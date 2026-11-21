"""Goals and their success checks, computed from stored object positions only.

Goal types. Directions are in the table frame, as seen from the robot's side of the
table toward cam0 (cam0 sits at +x looking back at the table): left = world -y,
in front = world +x (closer to cam0). The 5 training cameras are within 20 degrees of cam0's
azimuth (held-out ones within 35), so "left" in the table frame is "left" in every image.
    in_zone      a, z      object a's center inside zone z
    left_of      a, b      a is left of b, roughly level with it front-to-back
    right_of     a, b      the mirror image
    in_front_of  a, b      a is in front of b (closer to cam0), roughly level with it sideways
    behind       a, b      the mirror image
    next_to      a, b      the gap between a and b's footprints is at most NEXT_GAP
    between      a, b, c   a's center lies in the corridor between b and c
Compound goals for two-push plans (milestone 5):
    and          parts     every part holds (parts move different objects)

A goal holds if its predicate is true on every one of the last CHECK_STEPS settle
steps and every object it moves is still on the table (same rule as v1's label_outcome).
validate.py recomputes every label with label_goals() and must find 0 mismatches.
"""
import numpy as np

from gen_v3 import scene_xml as sx
from gen_v3.primitives import CHECK_STEPS

GOAL_TYPES = ["in_zone", "left_of", "right_of", "in_front_of", "behind", "next_to", "between"]
SIDE_GAP = 0.05  # directional relations: at least 5 cm apart along the direction ...
LEVEL_TOL = 0.05  # ... and at most 5 cm apart across it
NEXT_GAP = 0.025
RELATION_OFFSET = 0.08  # where an ideal directional push aims, beside b
BETWEEN_T = (0.2, 0.8)  # a's projection onto b -> c, as a fraction of the b-c distance
BETWEEN_TOL = 0.03  # max distance of a's center from the b-c line
BETWEEN_MIN_SPAN = 0.14  # b and c must start at least this far apart (else there is no room for a)
DIRECTIONS = {"left_of": (1, -1), "right_of": (1, 1), "in_front_of": (0, 1), "behind": (0, -1)}  # (axis, sign)


def on_table(pos):
    """pos (..., 3) -> bool: center above the table top and inside its edges."""
    return (pos[..., 2] > 0.0) & (np.abs(pos[..., 0]) < sx.TABLE_HALF) & (np.abs(pos[..., 1]) < sx.TABLE_HALF)


def moved_objects(goal):
    """Objects a goal asks to move (they must stay on the table)."""
    if goal["type"] == "and":
        return [p["a"] for p in goal["parts"]]
    return [goal["a"]]


def holds(goal, pos, scene):
    """Predicate of one goal on object positions pos (..., N, 3). Returns bool (...)."""
    t = goal["type"]
    if t == "and":
        return np.logical_and.reduce([holds(p, pos, scene) for p in goal["parts"]])
    a = goal["a"]
    pa = pos[..., a, :2]
    if t == "in_zone":
        z = np.asarray(scene["zones"][goal["b"]]["xy"])
        return np.linalg.norm(pa - z, axis=-1) <= sx.ZONE_RADIUS
    pb = pos[..., goal["b"], :2]
    d = pa - pb
    if t in DIRECTIONS:
        axis, sign = DIRECTIONS[t]
        return (sign * d[..., axis] >= SIDE_GAP) & (np.abs(d[..., 1 - axis]) <= LEVEL_TOL)
    if t == "next_to":
        ra = scene["objects"][a]["radius"]
        rb = scene["objects"][goal["b"]]["radius"]
        return np.linalg.norm(d, axis=-1) - ra - rb <= NEXT_GAP
    if t == "between":
        pc = pos[..., goal["c"], :2]
        bc = pc - pb
        span = np.linalg.norm(bc, axis=-1)
        frac = np.sum(d * bc, axis=-1) / np.maximum(span, 1e-9) ** 2
        off = np.abs(d[..., 0] * bc[..., 1] - d[..., 1] * bc[..., 0]) / np.maximum(span, 1e-9)
        return (frac >= BETWEEN_T[0]) & (frac <= BETWEEN_T[1]) & (off <= BETWEEN_TOL)
    raise ValueError(t)


def goal_id(g):
    if g["type"] == "and":
        return "and(" + "+".join(goal_id(p) for p in g["parts"]) + ")"
    return "|".join([g["type"], str(g["a"]), str(g["b"])] + ([str(g["c"])] if "c" in g else []))


def enumerate_goals(scene, init_pos):
    """Every single-push goal of the scene that is NOT already true at the start (init_pos: (N, 3))."""
    n = len(scene["objects"])
    goals = []
    for a in range(n):
        for z in range(len(scene["zones"])):
            goals.append({"type": "in_zone", "a": a, "b": z})
        for b in range(n):
            if b != a:
                for t in ("left_of", "right_of", "in_front_of", "behind", "next_to"):
                    goals.append({"type": t, "a": a, "b": b})
        for b in range(n):
            for c in range(b + 1, n):
                if a not in (b, c) and np.linalg.norm(init_pos[b, :2] - init_pos[c, :2]) >= BETWEEN_MIN_SPAN:
                    goals.append({"type": "between", "a": a, "b": b, "c": c})
    out = []
    for g in goals:
        if not holds(g, init_pos[None], scene)[0]:
            g["goal_id"] = goal_id(g)
            out.append(g)
    return out


def target_point(goal, scene, init_pos):
    """Where object a should end up for an ideal push toward this goal."""
    pa = init_pos[goal["a"], :2]
    t = goal["type"]
    if t == "in_zone":
        return np.asarray(scene["zones"][goal["b"]]["xy"])
    pb = init_pos[goal["b"], :2]
    if t in DIRECTIONS:
        axis, sign = DIRECTIONS[t]
        off = np.zeros(2)
        off[axis] = sign * RELATION_OFFSET
        return pb + off
    if t == "next_to":
        ra = scene["objects"][goal["a"]]["radius"]
        rb = scene["objects"][goal["b"]]["radius"]
        v = pa - pb
        return pb + v / np.linalg.norm(v) * (ra + rb + 0.01)
    if t == "between":
        return (pb + init_pos[goal["c"], :2]) / 2
    raise ValueError(t)


def label_goals(goals, obj_pos, phases, scene):
    """obj_pos (T, N, 3), phases (T,) -> int array (len(goals),) of 0/1 labels."""
    settle = np.flatnonzero(phases == "settle")[-CHECK_STEPS:]
    pos = obj_pos[settle]
    labels = []
    for g in goals:
        ok = all(on_table(pos[:, a]).all() for a in moved_objects(g)) and holds(g, pos, scene).all()
        labels.append(int(ok))
    return np.array(labels, dtype=np.int8)
