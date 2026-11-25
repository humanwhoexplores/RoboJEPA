"""v6 splits. Scene splits (train / val / test_id and six one-factor tests) are drawn in gen_v6.scene_spec; this
module adds the held-out action-goal pairs (test_act), which are a label mask rather than a scene split.

test_act: for each pair (action, goal type) below, no rollout of that action may be used as evidence about goals
of that type in train or val scenes, whether it succeeded or not. Every other pair stays, so the model sees each
held-out action work for other goals and each held-out goal type reached by other actions. At test time (test_id
scenes) the masked entries are the test: does the model rank, say, a drop above a push for "get the cube into
the red zone" without having seen a drop land in a zone?

Masked entries are flagged act_heldout in every instruction row, and the scene's label file carries the mask.
"""
import numpy as np

from gen_v6.scene_spec import HELD_COMBOS, HELD_REGION, SPLIT_P  # noqa: F401 (re-exported for the config)

HELD_ACT = {("drop", "in_zone"), ("drop", "on_top"), ("place", "between")}


def act_heldout(action, goal_type):
    return (action, goal_type) in HELD_ACT


def act_mask(actions, goal_types):
    """(n_rollouts, n_goals) bool: True where (action, goal type) is held out. For a compound goal, any held-out
    part counts."""
    def types(t):
        return t if isinstance(t, (list, tuple)) else [t]
    return np.array([[any(act_heldout(a, t) for t in types(g)) for g in goal_types] for a in actions], bool)
