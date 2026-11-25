"""Which instructions become decisions, picked from the scene alone (never from the rollouts' results).

Per scene:
    single     N_SINGLE one-step goals drawn by DECISION_MIX type weights from the goals whose objects can be
               named and whose target is in reach, different objects first
    probe      in a scene with a look-alike pair: one goal on a look-alike, named by its hidden property
               ("put the heavier red cube in the bin"). Its two-step version starts from a rollout that slid
               or poked one of the pair (a probe), so the model sees how one of them moves before it acts.
    compound   N_COMPOUND pairs of single goals on different objects, done in two steps
    tower      in a scene with three stackable objects: and(on_top(b, c), on_top(a, b)), two placements
Every goal of the scene also gets an instruction text where its objects can be named, so any goal can be used
for training. Goals record in_view.
"""
import numpy as np

from gen_v6 import goals as G
from gen_v6 import language as L

N_SINGLE = 8
N_COMPOUND = 2
REACH = 0.26
DECISION_MIX = {
    "in_zone": 0.10, "left_of": 0.03, "right_of": 0.03, "in_front_of": 0.03, "behind": 0.03, "next_to": 0.04,
    "between": 0.04, "on_top": 0.11, "off_of": 0.05, "inside": 0.13, "count_in": 0.04, "upright": 0.05,
    "lying": 0.06, "rotated": 0.04, "off_table": 0.04, "touching": 0.04, "moved_untouched": 0.04,
    "drawer_open": 0.025, "drawer_closed": 0.025, "lid_open": 0.025, "lid_closed": 0.025, "hidden": 0.02,
}
PROBE_TYPES = ("inside", "on_top", "in_zone", "off_table", "next_to")
COMPOUND_TYPES = ("in_zone", "on_top", "inside", "next_to", "lying", "upright", "off_of", "drawer_open", "lid_open")


def annotate(goals, spec, S0):
    for g in goals:
        p = G.target_point(g, spec, S0)
        g["in_view"] = bool(p is not None and np.all(np.abs(p) < REACH))
    return goals


def all_texts(goals, spec, S0, seed, idx):
    """(texts, templates) by goal id; None where an object cannot be named."""
    rng = np.random.default_rng([seed, idx, 12])
    texts, templates = {}, {}
    for g in goals:
        texts[g["goal_id"]], templates[g["goal_id"]] = L.instruction(g, spec, spec["split"], rng, S0["pos"])
    return texts, templates


def _objects(g):
    if g["type"] == "and":
        return {o for p in g["parts"] for o in _objects(p)}
    return {g[k] for k in ("a", "b", "c") if k in g}


def single_goals(goals, texts, rng, n=N_SINGLE, exclude=()):
    pool_all = [g for g in goals if g["in_view"] and texts.get(g["goal_id"]) and g["goal_id"] not in exclude]
    types = [t for t in DECISION_MIX if any(g["type"] == t for g in pool_all)]
    if not types:
        return []
    p = np.array([DECISION_MIX[t] for t in types])
    p /= p.sum()
    out, used = [], set()
    for _ in range(400):
        if len(out) == n:
            break
        t = types[rng.choice(len(types), p=p)]
        pool = [g for g in pool_all if g["type"] == t and g not in out]
        fresh = [g for g in pool if not (_objects(g) & used)]
        pool = fresh or pool
        if not pool:
            continue
        g = pool[rng.integers(len(pool))]
        out.append(g)
        used |= _objects(g)
    return out


def probe_goal(goals, spec, S0, rng):
    """A goal on a look-alike, worded by its hidden property; None without a pair."""
    look = spec.get("lookalike")
    if not look:
        return None
    pool = [g for g in goals if g["in_view"] and g["type"] in PROBE_TYPES and g["a"] in look["pair"]
            and not (set(look["pair"]) <= _objects(g))]
    if not pool:
        return None
    g = dict(pool[rng.integers(len(pool))])
    lrng = np.random.default_rng(rng.integers(1 << 31))
    old = L.PROPERTY_P
    L.PROPERTY_P = 1.0  # name the look-alike by its hidden property
    try:
        text, template = L.instruction(g, spec, spec["split"], lrng, S0["pos"])
    finally:
        L.PROPERTY_P = old
    if text is None:
        return None
    g["text"], g["template"] = text, template
    return g


def compound_goals(singles, rng, n=N_COMPOUND):
    pairs = [(g1, g2) for k, g1 in enumerate(singles) for g2 in singles[k + 1:]
             if g1["type"] in COMPOUND_TYPES and g2["type"] in COMPOUND_TYPES and not (_objects(g1) & _objects(g2))]
    out = []
    for k in rng.permutation(len(pairs))[:n]:
        g1, g2 = pairs[k]
        g = {"type": "and", "parts": [g1, g2]}
        g["goal_id"] = G.goal_id(g)
        out.append(g)
    return out


def tower_goal(goals, spec, S0, rng):
    """and(on_top(b, c), on_top(a, b)) over three objects whose on_top goals exist, or None."""
    on = {(g["a"], g["b"]): g for g in goals if g["type"] == "on_top" and g["in_view"]}
    trip = [(a, b, c) for (b, c) in on for (a, b2) in on if b2 == b and a != c]
    if not trip:
        return None
    a, b, c = trip[rng.integers(len(trip))]
    g = {"type": "and", "parts": [on[(b, c)], on[(a, b)]], "tower": [a, b, c]}
    g["goal_id"] = G.goal_id(g)
    return g


def pick(goals, spec, S0, seed, idx):
    """Returns (single decisions [dicts with 'source'], multi-step decisions, texts, templates)."""
    annotate(goals, spec, S0)
    texts, templates = all_texts(goals, spec, S0, seed, idx)
    rng = np.random.default_rng([seed, idx, 22])
    probe = probe_goal(goals, spec, S0, rng)
    singles = single_goals(goals, texts, rng, N_SINGLE - (probe is not None))
    evals = [dict(g, source="single") for g in singles]
    if probe is not None:
        evals.append(dict(probe, source="probe"))
    multi = [dict(g, source="compound") for g in compound_goals(singles, rng)]
    tower = tower_goal(goals, spec, S0, rng)
    if tower is not None and all(texts.get(p["goal_id"]) for p in tower["parts"]):
        multi.append(dict(tower, source="tower"))
    if probe is not None:
        multi.append({"type": "probe", "goal_id": "probe(" + probe["goal_id"] + ")", "parts": [probe],
                      "source": "probe", "pair": list(spec["lookalike"]["pair"])})
    lrng = np.random.default_rng([seed, idx, 13])
    for g in multi:
        if g["type"] == "and":
            texts[g["goal_id"]], templates[g["goal_id"]] = L.instruction(g, spec, spec["split"], lrng, S0["pos"])
        else:
            texts[g["goal_id"]], templates[g["goal_id"]] = probe["text"], probe["template"]
    return evals, multi, texts, templates
