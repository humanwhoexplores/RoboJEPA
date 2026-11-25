"""Where the fingers can hold an object, read from its current pose in the simulator.

grasp_options(env, i) lists (x, y, yaw, z, width) grasps for active object i: the pads centred at (x, y, z),
closing along yaw, across `width` of the object. The closing axis passes through the centre of mass (or a
pad slides the object round it). Nothing here knows about goals.

    round about its axis    ball, cylinder, tube, can, bottle: any yaw when upright; across the axis when lying
    box-like                cube, domino, milk, cereal, bread, lemon, a block lying on its side: across a
                            horizontal face pair narrower than the gripper (narrowest first)
    shaped blocks upright   L, T, hex, bar: the grasps from the gripper pilot (gen_v6.pilot_grasp.ideal_grasp)
    hammer                  across the handle at the centre of mass, near the head
    prism                   none: the fingers cannot hold a triangle (it stays a push-only object)
"""
import numpy as np

from gen_v6 import catalog as cat

MAX_WIDTH = 0.07  # the gripper opens 8 cm; leave room to come down around the object
MIN_Z = 0.014  # pads' centre at least this high: the fingertips clear the table
FINGER_REACH = 0.026  # the palm's underside is 3.1 cm above the pads' centre: keep it clear of the top
ROUND = {"ball", "cylinder", "tube", "can", "bottle"}
BOXLIKE = {"cube", "domino", "milk", "cereal", "bread", "lemon"}
SHAPED = {"L", "T", "hex", "bar"}


def body_frame(env, i):
    n = env.active[i]
    b = env.body_id[n]
    return env.data.xpos[b].copy(), env.data.xmat[b].reshape(3, 3).copy(), env._box[n]


def centre(env, i):
    """World centre of object i's collision box (the body origin can sit off-centre for robosuite meshes)."""
    p, R, box = body_frame(env, i)
    return p + R @ ((box.min(0) + box.max(0)) / 2)


def upright(R, tol=0.9):
    return abs(R[2, 2]) > tol


def lying(R, tol=0.35):
    return abs(R[2, 2]) < tol


def _yaw(v):
    return float(np.arctan2(v[1], v[0]))


def grasp_options(env, i, spec_obj, rng=None, n_round=2):
    """[(x, y, yaw, z, width), ...] for active object i (empty if it cannot be held as it lies now)."""
    kind = spec_obj.get("kind", spec_obj.get("shape"))
    p, R, box = body_frame(env, i)
    c = centre(env, i)
    top = float((p + box @ R.T)[:, 2].max())
    # Pads at the centre of mass, but no lower than 2.6 cm under the top: the palm's underside sits 3.1 cm above
    # the pads' centre, so tall things are held near their top.
    z = max(c[2], top - FINGER_REACH, MIN_Z)
    half = (box.max(0) - box.min(0)) / 2
    rng = rng if rng is not None else np.random.default_rng(0)
    out = []
    if kind == "prism":
        return out
    if kind in ROUND:
        diam = 2 * min(half[0], half[1])  # across the round axis (body z)
        if diam > MAX_WIDTH:
            return out
        if upright(R):
            for _ in range(n_round):
                out.append((c[0], c[1], float(rng.uniform(-np.pi, np.pi)), z, diam))
        elif lying(R):
            out.append((c[0], c[1], _yaw(R[:, 2]) + np.pi / 2, z, diam))
        return out
    if kind in SHAPED and upright(R):
        from gen_v6.pilot_grasp import centre_height, grasp_width, ideal_grasp
        yaw_body = float(np.arctan2(R[1, 0], R[0, 0]))
        q = np.array([np.cos(yaw_body / 2), 0, 0, np.sin(yaw_body / 2)])
        o = {"shape": kind, "size": spec_obj["size"]}
        gx, gy, gyaw, gz = ideal_grasp(o, p, q)
        # the pilot's height is for an object standing on the table; shift it to where the object is now
        out.append((gx, gy, gyaw, max(gz - centre_height(o) + c[2], MIN_Z), grasp_width(o)))
        return out
    if kind == "hammer":
        # at the centre of mass (60% of the mass is in the head), or the head swings down in the grip
        h = cat.HAMMER
        com = 0.6 * (h["handle_half"][0] + h["head_half"][0])
        at = p + R @ np.array([min(com, h["handle_half"][0] - 0.03), 0, 0])
        out.append((at[0], at[1], _yaw(R[:, 1]), max(at[2], MIN_Z), 2 * cat.HAMMER["handle_half"][1]))
        return out
    # box-like (and any block knocked onto its side): close across a horizontal pair of faces
    cands = []
    for k in range(3):
        axis = R[:, k]
        if abs(axis[2]) < 0.3 and 2 * half[k] <= MAX_WIDTH:
            cands.append((2 * half[k], _yaw(axis)))
    for w, yaw in sorted(cands):
        out.append((c[0], c[1], yaw, z, w))
    return out


def open_width(width):
    return float(min(0.08, width + 0.025))
