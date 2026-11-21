"""Which instructions become decisions, picked from the scene alone (never from the pushes' results).

Per scene, 8 single-push decisions:
    v3_focus  the 4 goals v3 asked in this exact scene, rebuilt with v3's own code and random stream,
              with v3's exact instruction words: a paired subset where only the pushes differ.
    open      4 more goals drawn with v3's type mix from every goal whose target is in view, WITHOUT
              v3's straight-path filter, so instructions that need more than a clean straight slide
              appear too. Different objects are preferred, as in v3.
Plus v3's 2 compound goals (pairs of v3 focus goals on different objects) for two-push decisions.

Every goal of the scene also gets an instruction text, so any goal can be used for training.
Each goal records in_view and straight_path, so results can be split by them.
"""
import numpy as np

from gen_v3.candidates import FOCUS_MIX, REACH, choose_focus, compound_goals, path_clear
from gen_v3.language import instruction
from gen_v3.tasks import target_point

N_OPEN = 4


def annotate(goals, spec, pos, dis):
    for g in goals:
        g["in_view"] = bool(np.all(np.abs(target_point(g, spec, pos)) < REACH))
        g["straight_path"] = bool(g["in_view"] and path_clear(g, spec, pos, dis))
    return goals


def v3_focus(goals, spec, pos, dis, seed, idx):
    """v3's 4 focus goals: choose_focus is the first draw from v3's candidate stream [seed, idx, 1]."""
    return choose_focus(goals, spec, pos, np.random.default_rng([seed, idx, 1]), dis)


def open_goals(goals, focus, rng, n=N_OPEN):
    """n more in-view goals by FOCUS_MIX type weights, objects not yet used first. Reads no labels."""
    taken = {g["goal_id"] for g in focus}
    pool_all = [g for g in goals if g["in_view"] and g["goal_id"] not in taken]
    used_a = {g["a"] for g in focus}
    types, p = list(FOCUS_MIX), np.array(list(FOCUS_MIX.values()))
    out = []
    for _ in range(300):
        if len(out) == n or not pool_all:
            break
        t = types[rng.choice(len(types), p=p)]
        pool = [g for g in pool_all if g["type"] == t and g not in out]
        fresh = [g for g in pool if g["a"] not in used_a]
        pool = fresh or pool
        if not pool:
            continue
        g = pool[rng.integers(len(pool))]
        out.append(g)
        used_a.add(g["a"])
    return out


def all_texts(goals, focus, compound, spec, pos, seed, idx):
    """(texts, templates) by goal id. Focus and compound goals use v3's stream [seed, idx, 2] in v3's
    order, so their words are identical to v3; every other goal gets a text from stream [seed, idx, 12]."""
    texts, templates = {}, {}
    rng = np.random.default_rng([seed, idx, 2])
    for g in focus + compound:
        texts[g["goal_id"]], templates[g["goal_id"]] = instruction(g, spec, spec["split"], rng, pos)
    rng = np.random.default_rng([seed, idx, 12])
    for g in goals:
        if g["goal_id"] not in texts:
            texts[g["goal_id"]], templates[g["goal_id"]] = instruction(g, spec, spec["split"], rng, pos)
    return texts, templates


def pick(goals, spec, pos, dis, seed, idx):
    """Returns (eval goals [dicts with 'source'], compound goals, texts, templates)."""
    annotate(goals, spec, pos, dis)
    focus = v3_focus(goals, spec, pos, dis, seed, idx)
    compound = compound_goals(focus)
    extra = open_goals(goals, focus, np.random.default_rng([seed, idx, 22]))
    evals = [dict(g, source="v3_focus") for g in focus] + [dict(g, source="open") for g in extra]
    texts, templates = all_texts(goals, focus, compound, spec, pos, seed, idx)
    return evals, compound, texts, templates
