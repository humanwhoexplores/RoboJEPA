"""Scripted approach-and-push primitive and outcome labeling.

A push is three numbers relative to the cube's current center:
  theta  - push direction in the table plane (radians, world frame)
  dist   - how far the fingertip travels past first contact (m), roughly the
           cube displacement for a clean push
  offset - sideways shift of the push line from the cube center (m); a large
           offset makes the fingers miss the cube

The arm hovers behind the cube, descends, pushes along theta, lifts away, and
the scene is left to settle before the outcome is checked.
"""
import numpy as np

from envs.push_env import CUBE_HALF, TABLE_TOP_Z, TARGET_RADIUS, TARGETS

POS_SCALE = 0.05  # OSC delta: action 1.0 = 5 cm per control step (robosuite default)
ROT_SCALE = 0.5  # OSC delta: action 1.0 = 0.5 rad per control step (robosuite default)
HOVER_Z = TABLE_TOP_Z + 0.12
HOME = np.array([-0.10, 0.0, 1.01])  # eef position at reset
PUSH_Z = TABLE_TOP_Z + 0.013  # fingertips ~3 mm above the table, hand underside above the cube
STANDOFF = 0.05  # gap between fingertip and cube face before the push starts
SETTLE_STEPS = 20
CHECK_STEPS = 10  # last N settle steps used for the success check


def eef_pos(env):
    return np.array(env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]])


def eef_yaw(env):
    """Yaw of the eef site's local y axis (the finger-opening axis) in the table plane."""
    xmat = env.sim.data.site_xmat[env.robots[0].eef_site_id["right"]].reshape(3, 3)
    return float(np.arctan2(xmat[1, 1], xmat[0, 1]))


def wrap_half_pi(a):
    """Wrap to [-pi/2, pi/2): the closed hand is symmetric under a half turn."""
    return (a + np.pi / 2) % np.pi - np.pi / 2


def push_yaw(theta):
    """Hand yaw that puts the long side of the hand across the push direction.

    The finger-opening axis is the hand's long axis; keeping it perpendicular to
    the push stops the hand from reaching over the cube and landing on its top.
    """
    return theta + np.pi / 2


def waypoints(cube_xy, theta, dist, offset):
    u = np.array([np.cos(theta), np.sin(theta)])
    perp = np.array([-u[1], u[0]])
    start = cube_xy - u * (CUBE_HALF + STANDOFF) + perp * offset
    end = cube_xy - u * CUBE_HALF + u * dist + perp * offset
    return [
        ("hover", np.r_[start, HOVER_Z], 0.6),
        ("descend", np.r_[start, PUSH_Z], 0.4),
        ("push", np.r_[end, PUSH_Z], 0.25),
        ("retract", np.r_[end, HOVER_Z], 0.5),
        ("home", HOME, 0.6),  # out of the camera's view before the final frame
    ]


def execute_push(env, theta, dist, offset, keyframe_fn=None, max_steps_per_wp=100, tol=0.008):
    """Run one push from the env's current state. Returns per-step logs.

    keyframe_fn(name) is called at each phase boundary so the caller can render.
    """
    cube_xy = env.cube_state()[0][:2]
    log = {k: [] for k in ("phase", "action", "joint_pos", "joint_vel", "eef_pos", "eef_quat",
                           "gripper_qpos", "cube_pos", "cube_quat", "cube_vel", "contact", "contact_force")}
    robot = env.robots[0]

    def record(phase, action):
        cp, cq, cv = env.cube_state()
        hit, force = env.gripper_cube_contact()
        log["phase"].append(phase)
        log["action"].append(action)
        log["joint_pos"].append(np.array(env.sim.data.qpos[robot._ref_joint_pos_indexes]))
        log["joint_vel"].append(np.array(env.sim.data.qvel[robot._ref_joint_vel_indexes]))
        log["eef_pos"].append(eef_pos(env))
        log["eef_quat"].append(np.array(env.sim.data.body_xquat[env.sim.model.body_name2id(robot.robot_model.eef_name["right"])]))
        log["gripper_qpos"].append(np.array(env.sim.data.qpos[robot._ref_gripper_joint_pos_indexes["right"]]))
        log["cube_pos"].append(cp)
        log["cube_quat"].append(cq)
        log["cube_vel"].append(cv)
        log["contact"].append(hit)
        log["contact_force"].append(force)

    if keyframe_fn:
        keyframe_fn("initial")
    yaw_goal = push_yaw(theta)
    for name, target, speed in waypoints(cube_xy, theta, dist, offset):
        for _ in range(max_steps_per_wp):
            err = target - eef_pos(env)
            yaw_err = wrap_half_pi(yaw_goal - eef_yaw(env))
            if np.linalg.norm(err) < tol and abs(yaw_err) < 0.05:
                break
            delta = err / POS_SCALE * 2.0
            norm = np.linalg.norm(delta)
            if norm > speed:  # scale the whole vector so the hand moves in a straight line
                delta *= speed / norm
            rot = np.clip(yaw_err / ROT_SCALE * 2.0, -0.3, 0.3)
            action = np.r_[delta, 0.0, 0.0, rot, 1.0]  # yaw only, gripper closed
            env.step(action)
            record(name, action)
        if keyframe_fn:
            keyframe_fn(f"after_{name}")
    for _ in range(SETTLE_STEPS):
        action = np.r_[np.zeros(6), 1.0]
        env.step(action)
        record("settle", action)
    if keyframe_fn:
        keyframe_fn("final")
    return {k: np.asarray(v) for k, v in log.items()}


def label_outcome(cube_pos_traj, contact, phases, start_xy):
    """Labels from the stored cube trajectory alone (validate.py recomputes these).

    Success for a target: the cube center is inside the disc on every one of the
    last CHECK_STEPS settle steps and the cube is still on the table.
    """
    settle = np.flatnonzero(phases == "settle")[-CHECK_STEPS:]
    xy = cube_pos_traj[settle, :2]
    z = cube_pos_traj[settle, 2]
    on_table = bool(np.all(z > TABLE_TOP_Z))
    final_xy = cube_pos_traj[-1, :2]
    touched = bool(np.any(contact))
    out = {"touched": touched, "on_table": on_table, "final_x": float(final_xy[0]), "final_y": float(final_xy[1]),
           "displacement": float(np.linalg.norm(final_xy - start_xy))}
    inside = {}
    for side, center in TARGETS.items():
        inside[side] = bool(on_table and np.all(np.linalg.norm(xy - center, axis=1) <= TARGET_RADIUS))
        out[f"success_{side}"] = int(inside[side])
    for side, center in TARGETS.items():
        other = "right" if side == "left" else "left"
        if inside[side]:
            mode = "success"
        elif not on_table:
            mode = "off_table"
        elif not touched:
            mode = "missed_contact"
        elif inside[other]:
            mode = "wrong_target"
        else:
            to_target = center - start_xy
            dist_needed = np.linalg.norm(to_target)
            progress = float(np.dot(final_xy - start_xy, to_target / dist_needed))
            lateral = float(np.linalg.norm((final_xy - start_xy) - progress * to_target / dist_needed))
            if lateral > TARGET_RADIUS:
                mode = "off_course"
            elif progress < dist_needed - TARGET_RADIUS:
                mode = "insufficient"
            else:
                mode = "overshoot"
        out[f"failure_mode_{side}"] = mode
    return out
