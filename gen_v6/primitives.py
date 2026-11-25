"""v6 action primitives on the Panda + parallel gripper.

Every primitive runs from the env's settled state with the arm at home, logs each 20 Hz control step, returns
the arm home and lets the scene settle. The logs keep v5's keys (so v4/v5 tooling reads them) and add the
gripper and v6's contact summaries:

    phase, push_idx, pusher_pos, obj_pos, obj_quat, dis_pos, contact, max_speed, min_dist   (v3)
    ee_state (x, y, z, roll, pitch, yaw, gripper width), q, target                         (v5)
    grip (commanded opening), held (objects between both pads), pair (object-object contact),
    table (object-table contact), fx_contact (object-fixture contact), fx_state (fixture joints)  (v6)

Actions (all positions in the table frame, m; angles in rad):
    push(x0, y0, theta, length)             fingers closed, slide along theta at 1.2 cm
    pull(x0, y0, theta, length)             the same motion from beyond the object back toward the robot
    sweep(x0, y0, theta, length, opening)   fingers open across the motion: a two-pronged rake through a group
    poke(x0, y0, theta, length, speed)      a fast short strike that stops dead; the object coasts on
    topple(x0, y0, theta, length, z)        a push high up a tall object
    pick_place(grasp, place)                grasp = (x, y, yaw, z), place = (x, y, yaw, z): place, drop (high z),
                                            stack (z on top of another object), unstack, twist (same x, y)
    stand_up(grasp, place)                  grasp a lying object across its middle, turn it upright in the air,
                                            set it down on its end
Phases: travel, descend, grasp, lift, turn, carry, lower, release, push, strike, swing, throw, retreat, return,
settle.
"""
import mujoco
import numpy as np

from gen_v3 import primitives as v3p
from gen_v5.primitives import minjerk, wrap, yaw_rot
from gen_v6.scene_xml import BASE

HIGH_Z = 0.30  # joint-space moves start / end with the tool this high
TRAVEL_Z = 0.20  # tool height that clears the tallest object (the 16 cm bottle) before descending
CARRY_Z = 0.24  # tool height while carrying
TRAVEL_SPEED = 0.25
DESCEND_SPEED = 0.12
LIFT_SPEED = 0.15
CARRY_SPEED = 0.20
PUSH_SPEED = v3p.PUSH_SPEED  # 0.20 m/s
POKE_SPEED = 0.35  # m/s average over the stroke
TURN_SPEED = 2.5  # rad/s, wrist yaw while travelling
SLOW_TURN = 0.6  # rad/s, tilting the hand sideways (several joints swing a long way)
JOINT_SPEED = 0.8
PUSH_Z = 0.012  # fingertip height during slides: below every object's centre of mass
GRASP_STEPS = 12  # 0.6 s for the fingers to close and squeeze
RELEASE_STEPS = 10
SETTLE_STEPS = 30  # 1.5 s: drops bounce and roll longer than pushes settle
CHECK_STEPS = v3p.CHECK_STEPS
OPEN = 0.08
LOG_KEYS = v3p.LOG_KEYS + ("ee_state", "q", "target", "grip", "held", "pair", "table", "fx_contact", "fx_state")


# ----------------------------------------------------------------------------- rotations
def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def tool_rot(yaw, tilt=0.0):
    """Hand pointing down with the fingers closing along yaw, then tilted by `tilt` about the closing axis."""
    return yaw_rot(yaw) @ rot_x(tilt)


def _quat(R):
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R).ravel())
    return q


def _slerp(q0, q1, s):
    if np.dot(q0, q1) < 0:
        q1 = -q1
    d = np.clip(np.dot(q0, q1), -1, 1)
    if d > 0.9995:
        q = q0 + s * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(d)
    return (np.sin((1 - s) * th) * q0 + np.sin(s * th) * q1) / np.sin(th)


def _mat(q):
    R = np.empty(9)
    mujoco.mju_quat2Mat(R, q)
    return R.reshape(3, 3)


def _angle(q0, q1):
    return 2 * np.arccos(min(1.0, abs(float(np.dot(q0, q1)))))


def near_yaw(theta, current):
    """theta or theta + pi (a parallel grasp is symmetric), whichever is closer to the current tool yaw."""
    a, b = wrap(theta), wrap(theta + np.pi)
    return a if abs(wrap(a - current)) <= abs(wrap(b - current)) else b


def _tool_yaw(env):
    R = env.tool_rot()
    return float(np.arctan2(R[1, 0], R[0, 0]))


# ----------------------------------------------------------------------------- logging and motion
def record(env, phase, log, push_idx, target):
    v3p.record(env, phase, log, push_idx)
    log["ee_state"].append(env.ee_state())
    log["q"].append(env.data.qpos[env.jq].copy())
    log["target"].append(target)
    log["grip"].append(env.grip_cmd)
    h = np.zeros(len(env.active), bool)
    h[env.holding()] = True
    log["held"].append(h)
    pair, table, fx = env.contact_summary()
    log["pair"].append(pair)
    log["table"].append(table)
    log["fx_contact"].append(fx)
    log["fx_state"].append(env.fixture_state())


class Motion:
    """Runs and logs one primitive: straight tool moves (position and orientation), joint-space moves, holds."""

    def __init__(self, env, log, push_idx, video_fn):
        self.env, self.log, self.idx, self.video_fn = env, log, push_idx, video_fn

    def _after(self, target):
        record(self.env, self.phase, self.log, self.idx, target)
        if self.video_fn:
            self.video_fn()

    def move(self, p0, p1, R0, R1, speed, phase, smooth=True, turn_speed=None, on_ctrl=None):
        """Tool target in a straight line from (p0, R0) to (p1, R1). smooth: minimum-jerk timing; else constant
        speed with an abrupt start and stop (slides and strikes). on_ctrl(c, n): called before control step c
        of n (a toss opens the fingers mid-move)."""
        env = self.env
        self.phase = phase
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        q0, q1 = _quat(R0), _quat(R1)
        t = max(np.linalg.norm(p1 - p0) / speed, _angle(q0, q1) / (turn_speed or TURN_SPEED), 1e-3)
        n_sub = max(1, int(np.ceil(t / env.model.opt.timestep)))
        n_ctrl = int(np.ceil(n_sub / v3p.SUBSTEPS))
        for c in range(n_ctrl):
            def frac(k, c=c):
                s = min(1.0, (c * v3p.SUBSTEPS + k + 1) / n_sub)
                return minjerk(s) if smooth else s

            def target(k):
                s = frac(k)
                return p0 + s * (p1 - p0), _mat(_slerp(q0, q1, s))
            if on_ctrl is not None:
                on_ctrl(c, n_ctrl)
            env.step(target)
            p, R = target(v3p.SUBSTEPS - 1)
            self._after(np.r_[p, np.arctan2(R[1, 0], R[0, 0])])
        return p1, R1

    def joint(self, q1, phase):
        env = self.env
        self.phase = phase
        q0 = env.q_des.copy()
        t = max(np.abs(q1 - q0).max() / JOINT_SPEED, 0.2)
        n_sub = max(1, int(np.ceil(t / env.model.opt.timestep)))
        n_ctrl = int(np.ceil(n_sub / v3p.SUBSTEPS))
        for c in range(n_ctrl):
            def q_fn(k, c=c):
                s = min(1.0, (c * v3p.SUBSTEPS + k + 1) / n_sub)
                return q0 + minjerk(s) * (q1 - q0)
            env.step_joint(q_fn)
            self._after(np.full(4, np.nan))

    def hold(self, n, phase):
        self.phase = phase
        for _ in range(n):
            self.env.step()
            self._after(np.full(4, np.nan))

    def tool(self):
        return self.env.pusher_pos().copy(), self.env.tool_rot()


def _start(env, log, video_fn, frame_fn, names, push_idx):
    keep = log is not None
    log = log if keep else {k: [] for k in LOG_KEYS}
    if frame_fn and names[0]:
        frame_fn(names[0])
    if video_fn:
        video_fn()
    return keep, Motion(env, log, push_idx, video_fn)


def _approach(M, xy, R, z, phase_down="descend"):
    """Joint-space move to above xy, straight down to TRAVEL_Z, then slowly down to z."""
    env = M.env
    q_high, _ = env.solve_ik_multi(np.r_[xy, HIGH_Z], R)
    M.joint(q_high, "travel")
    M.move(np.r_[xy, HIGH_Z], np.r_[xy, TRAVEL_Z], R, R, TRAVEL_SPEED, "travel")
    M.move(np.r_[xy, TRAVEL_Z], np.r_[xy, z], R, R, DESCEND_SPEED, phase_down)


def _finish(M, keep, frame_fn, names, up=True):
    env = M.env
    if up:
        p, R = M.tool()
        M.move(p, np.r_[p[:2], HIGH_Z], R, R, LIFT_SPEED, "retreat")
    M.joint(env.home["q"], "return")
    M.hold(SETTLE_STEPS, "settle")
    if frame_fn and names[-1]:
        frame_fn(names[-1])
    return M.log if keep else {k: np.asarray(v) for k, v in M.log.items()}


# ----------------------------------------------------------------------------- slides
def slide(env, x0, y0, theta, length, opening=0.0, z=PUSH_Z, finger_yaw=None, speed=PUSH_SPEED, phase="push",
          frame_fn=None, log=None, push_idx=0, video_fn=None, frame_names=("initial", "after_push", "final")):
    """Come down at (x0, y0, z) and slide `length` along theta at `speed`. finger_yaw: closing-axis direction
    (default across the motion, so the two fingers side by side form the face); opening: finger gap."""
    keep, M = _start(env, log, video_fn, frame_fn, frame_names, push_idx)
    fy = near_yaw(theta + np.pi / 2 if finger_yaw is None else finger_yaw, _tool_yaw(env))
    R = tool_rot(fy)
    u = np.array([np.cos(theta), np.sin(theta)])
    s, e = np.array([x0, y0]), np.array([x0, y0]) + u * length
    env.set_grip(opening)
    _approach(M, s, R, z)
    M.move(np.r_[s, z], np.r_[e, z], R, R, speed, phase, smooth=(phase == "strike"))
    if phase == "strike":  # stop dead and hold still while the object coasts away
        M.hold(4, "strike")
    if frame_fn and frame_names[1]:
        frame_fn(frame_names[1])
    return _finish(M, keep, frame_fn, frame_names)


def push(env, x0, y0, theta, length, opening=0.0, z=PUSH_Z, **kw):
    return slide(env, x0, y0, theta, length, opening=opening, z=z, **kw)


def pull(env, x0, y0, theta, length, **kw):
    """Hook behind an object and draw it toward the robot: the sampler starts beyond the object and points
    theta back toward the arm's base; the motion itself is a closed-finger slide."""
    return slide(env, x0, y0, theta, length, opening=0.0, **kw)


def sweep(env, x0, y0, theta, length, opening=0.07, **kw):
    return slide(env, x0, y0, theta, length, opening=opening, **kw)


def poke(env, x0, y0, theta, length, speed=POKE_SPEED, **kw):
    """A fast stroke: minimum-jerk timing at `speed` on average, so the fingers peak near 1.9 x speed halfway
    along and then brake hard. The sampler puts the object's face near the middle of the stroke, so it is hit
    at full speed and coasts on alone (how far depends on its mass, friction and shape)."""
    return slide(env, x0, y0, theta, length, opening=0.0, speed=speed, phase="strike", **kw)


def topple(env, x0, y0, theta, length, z, **kw):
    return slide(env, x0, y0, theta, length, opening=0.0, z=z, **kw)


# ----------------------------------------------------------------------------- grasps
def pick_place(env, grasp, place, open_width=OPEN, frame_fn=None, log=None, push_idx=0, video_fn=None,
               frame_names=("initial", "after_grasp", "after_release", "final"), carry_z=CARRY_Z):
    """Grasp at grasp = (x, y, yaw, z), carry, and open the fingers at place = (x, y, yaw, z)."""
    keep, M = _start(env, log, video_fn, frame_fn, frame_names, push_idx)
    gx, gy, gyaw, gz = grasp
    px, py, pyaw, pz = place
    gyaw = near_yaw(gyaw, _tool_yaw(env))
    Rg = tool_rot(gyaw)
    env.set_grip(open_width)
    _approach(M, (gx, gy), Rg, gz)
    env.set_grip(0.0)
    M.hold(GRASP_STEPS, "grasp")
    if frame_fn and frame_names[1]:
        frame_fn(frame_names[1])
    M.move(np.r_[gx, gy, gz], np.r_[gx, gy, carry_z], Rg, Rg, LIFT_SPEED, "lift")
    pyaw = gyaw + wrap(near_yaw(pyaw, gyaw) - gyaw)  # a parallel grasp: turn the shorter way
    Rp = tool_rot(pyaw)
    M.move(np.r_[gx, gy, carry_z], np.r_[px, py, carry_z], Rg, Rp, CARRY_SPEED, "carry")
    M.move(np.r_[px, py, carry_z], np.r_[px, py, pz], Rp, Rp, DESCEND_SPEED, "lower")
    env.set_grip(open_width)
    M.hold(RELEASE_STEPS, "release")
    if frame_fn and frame_names[2]:
        frame_fn(frame_names[2])
    return _finish(M, keep, frame_fn, frame_names)


def stand_up(env, grasp, place, tilt=np.pi / 2, open_width=OPEN, frame_fn=None, log=None, push_idx=0, video_fn=None,
             frame_names=("initial", "after_grasp", "after_release", "final")):
    """Grasp a lying object across its middle (grasp = (x, y, yaw, z), closing across the object's axis), lift,
    turn the hand by `tilt` about the closing axis so the object's axis points down, and set it on its end at
    place = (x, y, approach_yaw, z): z is the pads' height when the object's lower end touches the table."""
    keep, M = _start(env, log, video_fn, frame_fn, frame_names, push_idx)
    gx, gy, gyaw, gz = grasp
    px, py, _, pz = place
    gyaw = near_yaw(gyaw, _tool_yaw(env))
    Rg = tool_rot(gyaw)
    env.set_grip(open_width)
    _approach(M, (gx, gy), Rg, gz)
    env.set_grip(0.0)
    M.hold(GRASP_STEPS, "grasp")
    if frame_fn and frame_names[1]:
        frame_fn(frame_names[1])
    up_z = max(CARRY_Z, pz + 0.12)
    M.move(np.r_[gx, gy, gz], np.r_[gx, gy, up_z], Rg, Rg, LIFT_SPEED, "lift")
    # Turn whichever way leaves the wrist on the robot's side (tool z, which points from the pads back to the
    # wrist, toward the arm's base); the other way would fold the arm over the hand.
    to_base = np.r_[BASE[:2] - np.array([px, py]), 0.0]
    Rt = max((tool_rot(gyaw, s * abs(tilt)) for s in (1, -1)), key=lambda R: R[:, 2] @ to_base)
    M.move(np.r_[gx, gy, up_z], np.r_[gx, gy, up_z], Rg, Rt, CARRY_SPEED, "turn", turn_speed=SLOW_TURN)
    M.move(np.r_[gx, gy, up_z], np.r_[px, py, up_z], Rt, Rt, CARRY_SPEED, "carry")
    M.move(np.r_[px, py, up_z], np.r_[px, py, pz], Rt, Rt, DESCEND_SPEED, "lower")
    env.set_grip(open_width)
    M.hold(RELEASE_STEPS, "release")
    if frame_fn and frame_names[2]:
        frame_fn(frame_names[2])
    # back off along the hand's approach axis (away from the object) before going up
    p, R = M.tool()
    back = R[:, 2] * 0.08  # tool z points away from the fingertips
    M.move(p, p + back, R, R, LIFT_SPEED, "retreat")
    return _finish(M, keep, frame_fn, frame_names)


# ----------------------------------------------------------------------------- stage 2: lid, toss
def _axis_rot(axis, angle):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


LID_STEPS = 8  # straight pieces along the lid's arc
LID_TILT = np.radians(30)


def lid_open(env, grasp, hinge, axis, angle, open_width=OPEN, frame_fn=None, log=None, push_idx=0, video_fn=None,
             frame_names=("initial", "after_grasp", "after_release", "final")):
    """Grasp a lid's knob at grasp = (x, y, yaw, z) (closing along the hinge), then carry it `angle` along its arc
    about the hinge line (point `hinge`, direction `axis`) and let go. The hand turns by at most LID_TILT with the
    lid (the wrist runs into its stop beyond that); the knob pivots between the pads for the rest, which it can,
    since the pads close along the hinge line."""
    keep, M = _start(env, log, video_fn, frame_fn, frame_names, push_idx)
    gx, gy, gyaw, gz = grasp
    gyaw = near_yaw(gyaw, _tool_yaw(env))
    Rg = tool_rot(gyaw)
    env.set_grip(open_width)
    _approach(M, (gx, gy), Rg, gz)
    env.set_grip(0.0)
    M.hold(GRASP_STEPS, "grasp")
    if frame_fn and frame_names[1]:
        frame_fn(frame_names[1])
    k0, h = np.array([gx, gy, gz]), np.asarray(hinge, float)
    p_prev, R_prev = k0, Rg
    for j in range(1, LID_STEPS + 1):
        A = _axis_rot(axis, angle * j / LID_STEPS)
        p, R = h + A @ (k0 - h), _axis_rot(axis, min(angle, LID_TILT) * j / LID_STEPS) @ Rg
        M.move(p_prev, p, R_prev, R, LIFT_SPEED, "swing", turn_speed=SLOW_TURN)
        p_prev, R_prev = p, R
    env.set_grip(open_width)
    M.hold(RELEASE_STEPS, "release")
    if frame_fn and frame_names[2]:
        frame_fn(frame_names[2])
    p, R = M.tool()
    M.move(p, p + R[:, 2] * 0.06, R, R, LIFT_SPEED, "retreat")
    return _finish(M, keep, frame_fn, frame_names)


TOSS_Z = 0.14


def toss(env, grasp, theta, stroke, speed, release, open_width=OPEN, frame_fn=None, log=None, push_idx=0,
         video_fn=None, frame_names=("initial", "after_grasp", "after_release", "final")):
    """Grasp at grasp = (x, y, yaw, z), lift to TOSS_Z, draw back, then swing `stroke` m along theta at `speed`
    (m/s, average; minimum-jerk, so the peak is about 1.9x) and open the fingers `release` of the way through."""
    keep, M = _start(env, log, video_fn, frame_fn, frame_names, push_idx)
    gx, gy, gyaw, gz = grasp
    gyaw = near_yaw(gyaw, _tool_yaw(env))
    Rg = tool_rot(gyaw)
    env.set_grip(open_width)
    _approach(M, (gx, gy), Rg, gz)
    env.set_grip(0.0)
    M.hold(GRASP_STEPS, "grasp")
    if frame_fn and frame_names[1]:
        frame_fn(frame_names[1])
    u = np.r_[np.cos(theta), np.sin(theta), 0.0]
    top = np.r_[gx, gy, TOSS_Z]
    M.move(np.r_[gx, gy, gz], top, Rg, Rg, LIFT_SPEED, "lift")
    back = top - 0.4 * stroke * u
    M.move(top, back, Rg, Rg, CARRY_SPEED, "carry")

    def let_go(c, n):
        if c == int(round(release * n)):
            env.set_grip(open_width)
    M.move(back, back + stroke * u, Rg, Rg, speed, "throw", on_ctrl=let_go)
    env.set_grip(open_width)
    M.hold(RELEASE_STEPS, "release")
    if frame_fn and frame_names[2]:
        frame_fn(frame_names[2])
    return _finish(M, keep, frame_fn, frame_names)
