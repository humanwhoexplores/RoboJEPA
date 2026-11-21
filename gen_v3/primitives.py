"""Floating-pusher push primitive and per-step logging.

A push action is pusher-centric and never names an object:
    x0, y0  - where the pusher comes down (table frame, m)
    theta   - push direction in the table plane (rad)
    length  - how far the pusher travels along theta after touching down (m)

The V pusher appears above (x0, y0), opening toward theta, descends,
slides `length` along theta, lifts, and is parked out of view; then the scene
settles before the final frame.
gen_v3.candidates turns object-centric intents (push object A by d toward T)
into these four numbers.
"""
import numpy as np

from gen_v3 import scene_xml as sx
from gen_v3.env import SUBSTEPS

HOVER_Z = 0.10  # pusher center height while travelling above the objects
PUSH_Z = sx.PUSHER_HALF_LEN + 0.003  # pusher bottom 3 mm above the table
DESCEND_SPEED = 0.15  # m/s
PUSH_SPEED = 0.20  # m/s, close to v1's push speed
LIFT_SPEED = 0.30
SETTLE_STEPS = 20  # 1 s at 20 Hz
CHECK_STEPS = 10  # last N settle steps used for labels (same as v1)

LOG_KEYS = ("phase", "push_idx", "pusher_pos", "obj_pos", "obj_quat", "dis_pos", "contact", "max_speed", "min_dist")


def segment(env, start, end, speed, phase, log, push_idx=0):
    """Move the pusher in a straight line at constant speed, logging every control step."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    dist = np.linalg.norm(end - start)
    n_sub = max(1, int(np.ceil(dist / (speed * env.model.opt.timestep))))
    n_ctrl = int(np.ceil(n_sub / SUBSTEPS))
    for c in range(n_ctrl):
        def path(k, c=c):
            s = min(1.0, (c * SUBSTEPS + k + 1) / n_sub)
            return start + s * (end - start)
        env.step(path)
        record(env, phase, log, push_idx)


def record(env, phase, log, push_idx=0):
    pos, quat = env.object_poses()
    touch, deepest = env.contacts()
    log["phase"].append(phase)
    log["push_idx"].append(push_idx)
    log["dis_pos"].append(env.distractor_poses())
    log["pusher_pos"].append(env.pusher_pos())
    log["obj_pos"].append(pos)
    log["obj_quat"].append(quat)
    log["contact"].append(touch)
    log["max_speed"].append(float(env.object_speeds().max()))
    log["min_dist"].append(deepest)


def execute_push(env, x0, y0, theta, length, frame_fn=None, log=None, push_idx=0, frame_names=None):
    """Run one push from the env's current state. Returns per-control-step logs as arrays.

    frame_fn(name) is called at "initial", "after_push" and "final" so the caller can render
    (frame_names renames them, e.g. for the first push of a plan). With log given, steps are
    appended to it (as lists) and nothing is converted: used by execute_plan.
    """
    names = frame_names or ("initial", "after_push", "final")
    keep = log is not None
    log = log if keep else {k: [] for k in LOG_KEYS}
    u = np.array([np.cos(theta), np.sin(theta)])
    start = np.array([x0, y0])
    end = start + u * length
    if frame_fn and names[0]:
        frame_fn(names[0])
    env.teleport_pusher(np.r_[start, HOVER_Z], yaw=theta)  # V opens toward the push
    segment(env, np.r_[start, HOVER_Z], np.r_[start, PUSH_Z], DESCEND_SPEED, "descend", log, push_idx)
    segment(env, np.r_[start, PUSH_Z], np.r_[end, PUSH_Z], PUSH_SPEED, "push", log, push_idx)
    if frame_fn:
        frame_fn(names[1])
    segment(env, np.r_[end, PUSH_Z], np.r_[end, HOVER_Z], LIFT_SPEED, "lift", log, push_idx)
    env.teleport_pusher(env.park)
    for _ in range(SETTLE_STEPS):
        env.step()
        record(env, "settle", log, push_idx)
    if frame_fn:
        frame_fn(names[2])
    return log if keep else {k: np.asarray(v) for k, v in log.items()}


def execute_plan(env, pushes, frame_fn=None, plan_fn=None):
    """Run several pushes in a row, each after the scene has settled from the previous one.

    pushes: list of (x0, y0, theta, length), or None entries to be filled by plan_fn(k, env)
    once the scene has settled from push k - 1 (closed-loop re-planning between pushes).
    Frames: "initial", "after_push1", "mid1" (settled after push 1), ..., "after_pushK", "final".
    Returns (log arrays with a per-step push_idx, the pushes actually run).
    """
    log = {k: [] for k in LOG_KEYS}
    done = []
    n = len(pushes)
    for k, p in enumerate(pushes):
        if p is None:
            p = plan_fn(k, env)
        done.append(tuple(float(v) for v in p))
        names = ("initial" if k == 0 else None, f"after_push{k + 1}", "final" if k == n - 1 else f"mid{k + 1}")
        execute_push(env, *p, frame_fn=frame_fn, log=log, push_idx=k, frame_names=names)
    return {k: np.asarray(v) for k, v in log.items()}, done
