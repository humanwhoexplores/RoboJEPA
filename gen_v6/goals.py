"""v6 goals and their success checks, computed from stored logs only (positions, orientations, contacts,
fixture joints), so validate.py can recompute every label.

A state S holds arrays with any leading time shape (...):
    pos (..., N, 3), quat (..., N, 4) wxyz      objects
    pair (..., N, N), table (..., N)            object-object and object-table contact
    fxc (..., N, F), fxs (..., F)               object-fixture contact, fixture joint positions (NaN: none)

Goal types (a, b, c: objects; f: a fixture index into spec["fixtures"]; z: a zone):
    v3:          in_zone(a, z), left_of / right_of / in_front_of / behind / next_to (a, b), between(a, b, c)
    placement:   on_top(a, b)            a rests on b (touching it, above it, off the table)
                 off_of(a, b)            a started on b and now rests on the table
                 inside(a, f)            in a tray / bin / bowl / the drawer / the lidded box, or on the platform
                 count_in(f, n)          at least n objects inside container f
    pose:        upright(a), lying(a)    a tall object standing on its end / knocked onto its side
                 rotated(a)              turned in place by at least 45 degrees (moved under 3 cm)
    consequence: off_table(a), touching(a, b), moved_untouched(a) (moved 2 cm with no robot contact)
    stage 2:     drawer_open(f), drawer_closed(f), lid_open(f), lid_closed(f), hidden(a, f) (inside, shut)
    compound:    and(parts), for two- and three-step decisions (a tower is and(on_top(a, b), on_top(b, c)))
Each goal is enumerated only if it is false at the scene start.
"""
import itertools

import numpy as np

from gen_v3 import scene_xml as sx
from gen_v3 import tasks as v3t
from gen_v6 import catalog as cat

V3_TYPES = list(v3t.GOAL_TYPES)
GOAL_TYPES = V3_TYPES + ["on_top", "off_of", "inside", "count_in", "upright", "lying", "rotated", "off_table",
                         "touching", "moved_untouched", "drawer_open", "drawer_closed", "lid_open", "lid_closed",
                         "hidden"]
UPRIGHT_COS = np.cos(np.radians(25))
LYING_COS = np.cos(np.radians(65))
ROTATE_MIN = np.radians(45)
ROTATE_STAY = 0.03
MOVE_MIN = 0.02
DRAWER_OPEN, DRAWER_SHUT = 0.06, 0.01
LID_OPEN, LID_SHUT = 1.0, 0.1
FLAT_TOP = {"cube", "cylinder", "hex", "bar", "can", "milk", "cereal", "bread"}  # something can rest on these
NOT_ROUND = {"cube", "prism", "L", "T", "hex", "bar", "domino", "milk", "cereal", "bread", "hammer"}
# goals whose success is a change of state, not a move of a named object: no "agent moved" requirement
NO_AGENT = {"upright", "lying", "rotated", "count_in", "drawer_open", "drawer_closed", "lid_open", "lid_closed",
            "moved_untouched"}


def kind(o):
    return o.get("kind", o.get("shape"))


def zz(quat):
    """World z component of the body z axis."""
    q = np.asarray(quat)
    return 1 - 2 * (q[..., 1] ** 2 + q[..., 2] ** 2)


def yaw_of(quat):
    """Heading of the body x axis in the table plane."""
    w, x, y, z = (np.asarray(quat)[..., k] for k in range(4))
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def on_table(pos):
    return v3t.on_table(pos)


# ----------------------------------------------------------------------------- fixtures
def fx_local(spec, f, xy):
    """xy (..., 2) in fixture f's frame."""
    fx = spec["fixtures"][f]
    c, s = np.cos(fx["yaw"]), np.sin(fx["yaw"])
    d = np.asarray(xy) - np.asarray(fx["xy"])
    return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], -1)


def inside_fx(spec, f, pos, fxs=None):
    """Object positions pos (..., N, 3) inside fixture f -> bool (..., N)."""
    k = spec["fixtures"][f]["kind"]
    lp = fx_local(spec, f, pos[..., :2])
    ax, ay, z = np.abs(lp[..., 0]), np.abs(lp[..., 1]), pos[..., 2]
    if k == "tray":
        ix, iy = cat.TRAY_IN
        return (ax <= ix) & (ay <= iy) & (z < cat.TRAY_WALL_H + 0.06)
    if k == "bin":
        ix, iy = cat.BIN_IN
        return (ax <= ix) & (ay <= iy) & (z < cat.BIN_WALL_H + 0.03)
    if k == "bowl":
        return (np.hypot(lp[..., 0], lp[..., 1]) <= cat.BOWL_R - 0.005) & (z < cat.BOWL_H + 0.04)
    if k == "platform":
        hx, hy, hz = cat.PLATFORM_HALF
        return (ax <= hx) & (ay <= hy) & (z > 2 * hz)
    if k == "cabinet":
        ix, iy, iz = cat.DRAWER_IN
        slide = 0.0 if fxs is None else np.nan_to_num(fxs[..., f])[..., None]
        dx = cat.CAB_HALF[0] - ix - 0.008 + slide
        return (np.abs(lp[..., 0] - dx) <= ix) & (ay <= iy) & (z < 2 * cat.CAB_HALF[2])
    if k == "lidbox":
        ix, iy, iz = cat.LIDBOX_IN
        return (ax <= ix) & (ay <= iy) & (z < 2 * iz + 0.02)
    return np.zeros(np.shape(z), bool)


def container_fixtures(spec):
    return [f for f, fx in enumerate(spec["fixtures"]) if fx["kind"] in cat.CONTAINERS]


# ----------------------------------------------------------------------------- predicates
def holds(goal, S, spec, start=None):
    """Predicate of one goal on state S (leading time dims ...). start: the state the action began in (for
    rotated). Returns bool (...)."""
    t = goal["type"]
    if t == "and":
        return np.logical_and.reduce([holds(p, S, spec, start) for p in goal["parts"]])
    pos = S["pos"]
    if t in V3_TYPES:
        return v3t.holds(goal, pos, spec)
    if t == "on_top":
        a, b = goal["a"], goal["b"]
        d = np.linalg.norm(pos[..., a, :2] - pos[..., b, :2], axis=-1)
        r = spec["objects"][b]["radius"] + 0.01
        return S["pair"][..., a, b] & ~S["table"][..., a] & (pos[..., a, 2] > pos[..., b, 2] + 0.01) & (d < r)
    if t == "off_of":
        a, b = goal["a"], goal["b"]
        return ~S["pair"][..., a, b] & S["table"][..., a] & (pos[..., a, 2] < pos[..., b, 2] + 0.01)
    if t == "inside":
        return inside_fx(spec, goal["f"], pos, S.get("fxs"))[..., goal["a"]]
    if t == "count_in":
        return inside_fx(spec, goal["f"], pos, S.get("fxs")).sum(-1) >= goal["n"]
    if t == "upright":
        return np.abs(zz(S["quat"][..., goal["a"], :])) > UPRIGHT_COS
    if t == "lying":
        return np.abs(zz(S["quat"][..., goal["a"], :])) < LYING_COS
    if t == "rotated":
        a = goal["a"]
        dyaw = np.abs((yaw_of(S["quat"][..., a, :]) - yaw_of(start["quat"][a]) + np.pi) % (2 * np.pi) - np.pi)
        stay = np.linalg.norm(pos[..., a, :2] - start["pos"][a, :2], axis=-1) < ROTATE_STAY
        upright = np.abs(zz(S["quat"][..., a, :])) > UPRIGHT_COS
        return (dyaw >= ROTATE_MIN) & stay & upright
    if t == "off_table":
        return ~on_table(pos[..., goal["a"], :])
    if t == "touching":
        return S["pair"][..., goal["a"], goal["b"]]
    if t == "moved_untouched":
        raise ValueError("moved_untouched is judged on a whole rollout: use label_sets")
    if t in ("drawer_open", "drawer_closed", "lid_open", "lid_closed"):
        q = np.nan_to_num(S["fxs"][..., goal["f"]])
        return {"drawer_open": q >= DRAWER_OPEN, "drawer_closed": q <= DRAWER_SHUT,
                "lid_open": q >= LID_OPEN, "lid_closed": q <= LID_SHUT}[t]
    if t == "hidden":
        f = goal["f"]
        q = np.nan_to_num(S["fxs"][..., f])
        shut = q <= (DRAWER_SHUT if spec["fixtures"][f]["kind"] == "cabinet" else LID_SHUT)
        return inside_fx(spec, f, pos, S.get("fxs"))[..., goal["a"]] & shut
    raise ValueError(t)


def moved_objects(goal):
    """Objects a goal asks to move (they must end on the table)."""
    t = goal["type"]
    if t == "and":
        return [a for p in goal["parts"] for a in moved_objects(p)]
    if t in ("off_table", "count_in", "drawer_open", "drawer_closed", "lid_open", "lid_closed"):
        return []
    return [goal["a"]]


def references(goal):
    """Objects a goal names as references (b, c): the strict label wants them to stay put."""
    if goal["type"] == "and":
        return sorted({r for p in goal["parts"] for r in references(p)} - set(moved_objects(goal)))
    if goal["type"] in ("in_zone", "inside", "count_in", "hidden") or goal["type"] in NO_AGENT:
        return []
    return [goal[k] for k in ("b", "c") if k in goal]


def agents(goal):
    """Objects that must move at least MOVE_MIN for the default label (single goals; see labels for "and")."""
    return [] if goal["type"] in NO_AGENT else [goal["a"]]


def goal_id(g):
    if g["type"] == "and":
        return "and(" + "+".join(goal_id(p) for p in g["parts"]) + ")"
    keys = [k for k in ("a", "b", "c", "f", "n") if k in g]
    return "|".join([g["type"]] + [f"{k}{g[k]}" if k in ("f", "n") else str(g[k]) for k in keys])


# ----------------------------------------------------------------------------- enumeration
def initial_state(env):
    pos, quat = env.object_poses()
    pair, table, fxc = env.contact_summary()
    return {"pos": pos, "quat": quat, "pair": pair, "table": table, "fxc": fxc, "fxs": env.fixture_state()}


def enumerate_goals(spec, S0, stage2=True):
    """Every goal of the scene that is NOT already true in S0 (a state without time dims)."""
    objs = spec["objects"]
    n = len(objs)
    fx = spec["fixtures"]
    goals = []
    one = {k: v[None] for k, v in S0.items()}
    for a in range(n):
        ka = kind(objs[a])
        for z in range(len(spec["zones"])):
            goals.append({"type": "in_zone", "a": a, "b": z})
        for b in range(n):
            if b == a:
                continue
            for t in ("left_of", "right_of", "in_front_of", "behind", "next_to", "touching"):
                goals.append({"type": t, "a": a, "b": b})
            if kind(objs[b]) in FLAT_TOP and ka != "hammer" and abs(zz(S0["quat"][b])) > UPRIGHT_COS:
                goals.append({"type": "on_top", "a": a, "b": b})
            if S0["pair"][a, b] and S0["pos"][a, 2] > S0["pos"][b, 2] + 0.01 and not S0["table"][a]:
                goals.append({"type": "off_of", "a": a, "b": b})
        for b, c in itertools.combinations(range(n), 2):
            if a not in (b, c) and np.linalg.norm(S0["pos"][b, :2] - S0["pos"][c, :2]) >= v3t.BETWEEN_MIN_SPAN:
                goals.append({"type": "between", "a": a, "b": b, "c": c})
        for f, fxd in enumerate(fx):
            if fxd["kind"] in ("tray", "bin", "bowl", "platform") or (stage2 and fxd["kind"] in ("cabinet", "lidbox")):
                if ka != "hammer":
                    goals.append({"type": "inside", "a": a, "f": f})
                if stage2 and fxd["kind"] in ("cabinet", "lidbox") and ka != "hammer":
                    goals.append({"type": "hidden", "a": a, "f": f})
        if ka in cat.TALL:
            goals.append({"type": "upright", "a": a})
            goals.append({"type": "lying", "a": a})
        if ka in NOT_ROUND and abs(zz(S0["quat"][a])) > UPRIGHT_COS:
            goals.append({"type": "rotated", "a": a})
        goals.append({"type": "off_table", "a": a})
        goals.append({"type": "moved_untouched", "a": a})
    for f, fxd in enumerate(fx):
        if fxd["kind"] in ("tray", "bin", "bowl", "platform"):
            cur = int(inside_fx(spec, f, S0["pos"][None], None)[0].sum())
            for k in (1, 2):
                if cur + k <= n:
                    goals.append({"type": "count_in", "f": f, "n": cur + k})
        if stage2 and fxd["kind"] == "cabinet":
            goals += [{"type": "drawer_open", "f": f}, {"type": "drawer_closed", "f": f}]
        if stage2 and fxd["kind"] == "lidbox":
            goals += [{"type": "lid_open", "f": f}, {"type": "lid_closed", "f": f}]
    out = []
    for g in goals:
        if g["type"] in ("rotated", "moved_untouched") or not holds(g, one, spec, start=S0)[0]:
            g["goal_id"] = goal_id(g)
            out.append(g)
    return out


def target_point(goal, spec, S0):
    """Where object a should end up (for in-view checks and the no-physics rule), or None."""
    t = goal["type"]
    if t in V3_TYPES:
        return v3t.target_point(goal, spec, S0["pos"])
    if t in ("on_top", "touching"):
        return S0["pos"][goal["b"], :2]
    if t in ("inside", "hidden", "count_in", "drawer_open", "drawer_closed", "lid_open", "lid_closed"):
        return np.asarray(spec["fixtures"][goal["f"]]["xy"])
    if t == "off_table":
        p = S0["pos"][goal["a"], :2]
        k = int(np.argmax(np.abs(p)))
        out = p.copy()
        out[k] = np.sign(p[k]) * (sx.TABLE_HALF + 0.05)
        return out
    if t == "and":
        return target_point(goal["parts"][-1], spec, S0)
    return S0["pos"][goal["a"], :2]
