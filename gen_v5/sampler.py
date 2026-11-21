"""v4's goal-blind pushes, restricted to what the Panda can reach.

v4's floating pusher could start and end anywhere (push end points reach |x|, |y| = 0.53 m, i.e. off the
table). A real arm on one side of the table cannot. Each v4 candidate is checked with IK at its start
(push height and travel height) and its end (push height), with the V turned toward theta:
    end out of reach    -> the push is shortened (1 cm steps) until it is reachable
    start out of reach  -> the candidate is dropped; replacements come from further v4 sampler draws
Still goal-blind: the check reads only the push and the robot, never a goal.
"""
import numpy as np

from gen_v4.sampler import N_LINES, N_TOTAL, sample_pushes as v4_sample
from gen_v5.primitives import HIGH_Z, PUSH_Z, yaw_rot

POS_TOL, ROT_TOL = 0.003, 0.02
# rad kept from each joint limit along a push. The elbow (joint 4) folds up near its limit when a push runs
# toward the base; the arm is then nearly singular and tracking breaks down, so it gets a wide margin. The
# wrist twist (joint 7) hardly moves during a push, so a small margin suffices there.
JOINT_MARGIN = np.array([0.1, 0.1, 0.1, 0.3, 0.1, 0.1, 0.05])
MIN_LENGTH = 0.03


def reachable(env, xy, theta, z):
    return env.solve_ik_multi(np.r_[xy, z], yaw_rot(theta), pos_tol=POS_TOL, rot_tol=ROT_TOL)[1]


def follow(env, q, path, rot, iters=30):
    """Kinematic dry run: from joints q, follow the tool path (list of points) with the same IK the
    controller uses. Returns the index of the last point tracked within tolerance (-1 if none)."""
    last = -1
    for i, p in enumerate(path):
        for _ in range(iters):
            q, ep, er = env.ik_step(q, p, rot)
            if ep < POS_TOL / 3 and er < ROT_TOL / 3:
                break
        margin = np.minimum(q - env.jrange[:, 0], env.jrange[:, 1] - q)
        if ep > POS_TOL or er > ROT_TOL or (margin < JOINT_MARGIN).any():
            break
        last = i
    return last


def fit(env, cand):
    """The candidate, shortened to what the arm can follow continuously, or None if its start is out of
    reach. Same motion as the primitive: IK branch chosen above the start, then straight down and along."""
    x0, y0, th, L = cand["x0"], cand["y0"], cand["theta"], cand["length"]
    s = np.array([x0, y0])
    rot = yaw_rot(th)
    q_high, ok = env.solve_ik_multi(np.r_[s, HIGH_Z], rot, pos_tol=POS_TOL, rot_tol=ROT_TOL)
    if not ok:
        return None
    down = [np.r_[s, z] for z in np.linspace(HIGH_Z, PUSH_Z, 14)]
    if follow(env, q_high, down, rot) < len(down) - 1:
        return None
    q = q_high
    for p in down:  # joints at the push start, along the same branch
        for _ in range(30):
            q, _, _ = env.ik_step(q, p, rot)
    u = np.array([np.cos(th), np.sin(th)])
    steps = np.arange(0.01, L + 1e-9, 0.01)
    last = follow(env, q, [np.r_[s + u * d, PUSH_Z] for d in steps], rot)
    L_fit = L if last == len(steps) - 1 else (float(steps[last]) if last >= 0 else 0.0)
    if L_fit < MIN_LENGTH <= L:
        return None
    return {**cand, "length": float(L_fit), "v4_length": float(L), "shortened": bool(L_fit < L)}


def sample_pushes(env, scene, pos, dis_pos, rng_seed, n_total=N_TOTAL, n_lines=N_LINES, max_draws=8):
    """n_total reachable goal-blind pushes (fewer only if max_draws v4 draws cannot supply them)."""
    out, dropped = [], 0
    for r in range(max_draws):
        for c in v4_sample(scene, pos, dis_pos, np.random.default_rng([*rng_seed, r]), n_total, n_lines):
            f = fit(env, c)
            if f is None:
                dropped += 1
            elif len(out) < n_total:
                out.append({**f, "draw": r})
        if len(out) == n_total:
            break
    return out, dropped
