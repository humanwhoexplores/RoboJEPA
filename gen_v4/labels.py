"""Three labels per (push, goal), all computed from stored object positions only.

    state    the v3 rule: the goal's predicate holds on every one of the last CHECK_STEPS settle
             steps, and every object the goal names as moving is still on the table
    default  state AND every object the instruction says to push moved at least MOVE_MIN.
             In v3, "push the red cube next to the blue can" counted as a success when the can was
             shoved next to a cube that never moved: 15% of v3's positives were of that kind.
    strict   default AND no reference object (b, c of a relation) moved more than REF_MAX

Movement is measured from the state the push started in (the scene start, or a two-push
decision's middle state). For a compound goal judged after its second push, the first part's
object is measured from the scene start (origin_pos) and the second part's from the middle state.
validate.py recomputes all three for every rollout and must find 0 mismatches.
"""
import numpy as np

from gen_v3.primitives import CHECK_STEPS
from gen_v3.tasks import holds, moved_objects, on_table

MOVE_MIN = 0.02  # m the named object must move
REF_MAX = 0.03  # m a reference object may move under the strict label
LABELS = ("default", "state", "strict")


def references(goal):
    if goal["type"] == "and":
        return sorted({r for p in goal["parts"] for r in references(p)} - set(moved_objects(goal)))
    if goal["type"] == "in_zone":
        return []
    return [goal[k] for k in ("b", "c") if k in goal]


def label_sets(goals, obj_pos, phases, scene, start_pos, origin_pos=None):
    """obj_pos (T, N, 3), phases (T,), start_pos (N, 3) -> {label name: int8 array (len(goals),)}."""
    settle = np.flatnonzero(np.asarray(phases) == "settle")[-CHECK_STEPS:]
    pos = obj_pos[settle]
    final = obj_pos[-1, :, :2]
    moved = np.linalg.norm(final - np.asarray(start_pos)[:, :2], axis=1)
    moved_origin = moved if origin_pos is None else np.linalg.norm(final - np.asarray(origin_pos)[:, :2], axis=1)
    out = {k: [] for k in LABELS}
    for g in goals:
        state = all(on_table(pos[:, a]).all() for a in moved_objects(g)) and bool(holds(g, pos, scene).all())
        parts = g["parts"] if g["type"] == "and" else [g]
        agents = all((moved_origin if (i == 0 and len(parts) > 1) else moved)[p["a"]] >= MOVE_MIN
                     for i, p in enumerate(parts))
        refs_still = all(moved[r] <= REF_MAX for r in references(g))
        out["state"].append(state)
        out["default"].append(state and agents)
        out["strict"].append(state and agents and refs_still)
    return {k: np.array(v, dtype=np.int8) for k, v in out.items()}
