"""Three labels per (rollout, goal), computed from a rollout's stored log only (validate.py recomputes them).

    state    the goal's predicate holds on every one of the last CHECK_STEPS settle steps, and every object the
             goal asks to move ends on the table (off_table asks for the opposite)
    default  state AND every agent (the object the instruction names as the one to move) moved at least
             MOVE_MIN. Goals that are a change of state (upright, lying, rotated, drawer / lid, count_in,
             moved_untouched) have no agent: their predicate already says what must change.
    strict   default AND no reference object (b, c of a relation) moved more than REF_MAX

moved_untouched(a) is judged on the whole rollout: a moved at least MOVE_MIN and the robot never touched it.
rotated(a) is judged against the rollout's start state. Movement is measured from the state the rollout started
in; for a compound goal judged after its last step, earlier parts' agents are measured from the decision's
start state (origin), as in v4.
"""
import numpy as np

from gen_v3.primitives import CHECK_STEPS
from gen_v6 import goals as G

MOVE_MIN = G.MOVE_MIN  # m the named object must move
REF_MAX = 0.03  # m a reference object may move under the strict label
LABELS = ("default", "state", "strict")


def states(log, steps):
    """The goal-checking state (goals module layout) at control steps `steps` of a rollout log."""
    return {"pos": np.asarray(log["obj_pos"])[steps], "quat": np.asarray(log["obj_quat"])[steps],
            "pair": np.asarray(log["pair"])[steps].astype(bool), "table": np.asarray(log["table"])[steps].astype(bool),
            "fxc": np.asarray(log["fx_contact"])[steps].astype(bool), "fxs": np.asarray(log["fx_state"])[steps]}


def check_steps(log):
    return np.flatnonzero(np.asarray(log["phase"]) == "settle")[-CHECK_STEPS:]


def _single_state(g, S, spec, start, log, moved):
    if g["type"] == "moved_untouched":
        a = g["a"]
        return bool(moved[a] >= MOVE_MIN and not np.asarray(log["contact"])[:, a].any()
                    and G.on_table(S["pos"][-1, a]))
    if g["type"] == "and":
        return all(_single_state(p, S, spec, start, log, moved) for p in g["parts"])
    ok = bool(G.holds(g, S, spec, start).all())
    return ok and all(G.on_table(S["pos"][:, a]).all() for a in G.moved_objects(g))


def label_sets(goals, log, spec, start, origin=None):
    """goals: list of goal dicts; log: one rollout's log (arrays over control steps); start: the state the rollout
    began in (goals.initial_state layout, no time dims); origin: the decision's start state for compound goals
    judged after a later step (default: start). Returns {label name: int8 array (len(goals),)}."""
    S = states(log, check_steps(log))
    final = np.asarray(log["obj_pos"])[-1, :, :2]
    moved = np.linalg.norm(final - np.asarray(start["pos"])[:, :2], axis=1)
    origin = start if origin is None else origin
    moved_origin = np.linalg.norm(final - np.asarray(origin["pos"])[:, :2], axis=1)
    out = {k: [] for k in LABELS}
    for g in goals:
        state = _single_state(g, S, spec, start, log, moved)
        parts = g["parts"] if g["type"] == "and" else [g]
        last = len(parts) - 1
        agents = all((moved if k == last else moved_origin)[a] >= MOVE_MIN
                     for k, p in enumerate(parts) for a in G.agents(p))
        refs_still = all(moved_origin[r] <= REF_MAX for r in G.references(g))
        out["state"].append(state)
        out["default"].append(state and agents)
        out["strict"].append(state and agents and refs_still)
    return {k: np.array(v, dtype=np.int8) for k, v in out.items()}
