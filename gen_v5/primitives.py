"""v5 push primitive: the same 4-number push as v3 (x0, y0, theta, length), executed by the Panda.

The tool travels above the objects to (x0, y0) while turning the V toward theta, descends, slides
`length` along theta, lifts, returns home, and the scene settles. v3's floating pusher teleported to
the start instead; a real arm has to travel, so travel happens high enough to clear every object.

Logs per 20 Hz control step: v3's keys (so gen_v4.labels works unchanged) plus the 7-D end-effector
state, joint positions and commanded tool targets. frame_fn(name) gets the v3 keyframe names;
video_fn() is called every control step so the caller can sample frames at any rate.
"""
import numpy as np

from gen_v3 import primitives as v3p
from gen_v3 import scene_xml as sx
from gen_v5.env import SUBSTEPS

TRAVEL_Z = 0.16  # tool height from which the push descends: plates clear the tallest distractor (0.09 m)
HIGH_Z = 0.30  # joint-space moves start / end with the tool this high above the push start / end
JOINT_SPEED = 0.8  # rad/s, fastest joint during joint-space moves (average)
PUSH_Z = v3p.PUSH_Z
TRAVEL_SPEED = 0.25  # m/s (average; non-contact moves use a smooth minimum-jerk profile)
DESCEND_SPEED = v3p.DESCEND_SPEED
PUSH_SPEED = v3p.PUSH_SPEED
LIFT_SPEED = 0.20
TURN_SPEED = 2.5  # rad/s, wrist yaw while travelling
SETTLE_STEPS = v3p.SETTLE_STEPS
LOG_KEYS = v3p.LOG_KEYS + ("ee_state", "q", "target")


def yaw_rot(yaw):
    """Tool rotation: V opening toward yaw, flange pointing down (tool z = world up)."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def minjerk(s):
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def segment(env, p0, p1, yaw0, yaw1, speed, phase, log, push_idx, video_fn, smooth=True):
    """Move the tool target in a straight line (and turn its yaw); log each control step. smooth: minimum-jerk
    timing (starts and stops gently, as a real arm's trajectory generator does); else constant speed (the push)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    dyaw = wrap(yaw1 - yaw0)
    t = max(np.linalg.norm(p1 - p0) / speed, abs(dyaw) / TURN_SPEED, 1e-3)
    n_sub = max(1, int(np.ceil(t / env.model.opt.timestep)))
    n_ctrl = int(np.ceil(n_sub / SUBSTEPS))
    for c in range(n_ctrl):
        def frac(k, c=c):
            s = min(1.0, (c * SUBSTEPS + k + 1) / n_sub)
            return minjerk(s) if smooth else s

        def target(k):
            s = frac(k)
            return p0 + s * (p1 - p0), yaw_rot(yaw0 + s * dyaw)
        env.step(target)
        record(env, phase, log, push_idx, np.r_[target(SUBSTEPS - 1)[0], yaw0 + frac(SUBSTEPS - 1) * dyaw])
        if video_fn:
            video_fn()
    return yaw0 + dyaw


def joint_move(env, q1, phase, log, push_idx, video_fn):
    """Joint-space move (smooth, minimum-jerk timing) from the current joint targets to q1."""
    q0 = env.q_des.copy()
    t = max(np.abs(q1 - q0).max() / JOINT_SPEED, 0.2)
    n_sub = max(1, int(np.ceil(t / env.model.opt.timestep)))
    n_ctrl = int(np.ceil(n_sub / SUBSTEPS))
    for c in range(n_ctrl):
        def q_fn(k, c=c):
            s = min(1.0, (c * SUBSTEPS + k + 1) / n_sub)
            return q0 + minjerk(s) * (q1 - q0)
        env.step_joint(q_fn)
        record(env, phase, log, push_idx, np.full(4, np.nan))
        if video_fn:
            video_fn()


def hold(env, n, phase, log, push_idx, video_fn):
    for _ in range(n):
        env.step()
        record(env, phase, log, push_idx, np.full(4, np.nan))
        if video_fn:
            video_fn()


def record(env, phase, log, push_idx, target):
    v3p.record(env, phase, log, push_idx)
    log["ee_state"].append(env.ee_state())
    log["q"].append(env.data.qpos[env.jq].copy())
    log["target"].append(target)


def execute_push(env, x0, y0, theta, length, frame_fn=None, log=None, push_idx=0, frame_names=None, video_fn=None):
    """Run one push from the env's current (settled, arm at home) state. Returns per-step logs as arrays."""
    names = frame_names or ("initial", "after_push", "final")
    keep = log is not None
    log = log if keep else {k: [] for k in LOG_KEYS}
    u = np.array([np.cos(theta), np.sin(theta)])
    start, end = np.array([x0, y0]), np.array([x0, y0]) + u * length
    if frame_fn and names[0]:
        frame_fn(names[0])
    if video_fn:
        video_fn()
    q_high, _ = env.solve_ik_multi(np.r_[start, HIGH_Z], yaw_rot(theta))  # picks a reachable arm branch
    joint_move(env, q_high, "travel", log, push_idx, video_fn)
    yaw = theta
    segment(env, np.r_[start, HIGH_Z], np.r_[start, TRAVEL_Z], yaw, yaw, TRAVEL_SPEED, "travel", log, push_idx, video_fn)
    segment(env, np.r_[start, TRAVEL_Z], np.r_[start, PUSH_Z], yaw, yaw, DESCEND_SPEED, "descend", log, push_idx, video_fn)
    segment(env, np.r_[start, PUSH_Z], np.r_[end, PUSH_Z], yaw, yaw, PUSH_SPEED, "push", log, push_idx, video_fn,
            smooth=False)
    if frame_fn:
        frame_fn(names[1])
    segment(env, np.r_[end, PUSH_Z], np.r_[end, HIGH_Z], yaw, yaw, LIFT_SPEED, "lift", log, push_idx, video_fn)
    joint_move(env, env.home["q"], "return", log, push_idx, video_fn)
    hold(env, SETTLE_STEPS, "settle", log, push_idx, video_fn)
    if frame_fn:
        frame_fn(names[2])
    return log if keep else {k: np.asarray(v) for k, v in log.items()}
