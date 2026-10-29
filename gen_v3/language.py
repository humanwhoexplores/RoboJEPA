"""Instructions and hindsight captions for v3 goals (milestone 3).

Referring expressions name an object with the fewest attributes that single it out
("the cylinder", "the red cube", "the small blue L-block"), or, some of the time, with
a relation that singles it out ("the cube nearest the green zone", "the leftmost cylinder").
Shape words have synonyms; test scenes use held-out templates AND held-out shape words,
so test language is new in two ways. Compound goals join two instructions.

Hindsight captions describe what a rollout actually did, from its stored positions
("the red cube moved 12 cm to the left and ended in the blue zone"), so every rollout,
including failed ones, comes with a true description of its outcome.
"""
import itertools

import numpy as np

from gen_v3 import scene_xml as sx
from gen_v3.tasks import NEXT_GAP, on_table

SHAPE_WORDS = {  # train / val words
    "cube": ["cube", "box"], "cylinder": ["cylinder", "can"], "prism": ["triangle block"],
    "L": ["L-block"], "T": ["T-block"], "hex": ["hexagon block"], "bar": ["bar", "long block"],
}
TEST_SHAPE_WORDS = {  # held out for test scenes
    "cube": ["square block"], "cylinder": ["round block"], "prism": ["triangular prism"],
    "L": ["L-shaped piece"], "T": ["T-shaped piece"], "hex": ["hex prism"], "bar": ["long beam"],
}
SIZE_WORDS = {"s": "small", "l": "large"}
RELATIONAL_P = 0.3  # share of references that use a relation when one singles the object out
MARGIN = 0.03  # a relational reference needs the runner-up to be at least 3 cm worse

TRAIN = {
    "in_zone": ["push {a} into the {zone} zone", "move {a} to the {zone} area", "slide {a} onto the {zone} circle",
                "put {a} in the {zone} zone"],
    "left_of": ["push {a} to the left of {b}", "move {a} so it is left of {b}", "slide {a} over to the left of {b}"],
    "right_of": ["push {a} to the right of {b}", "move {a} so it is right of {b}", "slide {a} over to the right of {b}"],
    "in_front_of": ["push {a} in front of {b}", "move {a} so it is in front of {b}"],
    "behind": ["push {a} behind {b}", "move {a} so it is behind {b}"],
    "next_to": ["push {a} next to {b}", "move {a} up against {b}", "slide {a} until it is beside {b}"],
    "between": ["push {a} between {b} and {c}", "move {a} into the gap between {b} and {c}"],
    "and": ["{p1}, then {p2}", "{p1} and {p2}", "first {p1}, then {p2}"],
}
TEST = {
    "in_zone": ["get {a} inside the {zone} spot", "nudge {a} until it sits in the {zone} region"],
    "left_of": ["place {a} on the left side of {b}"],
    "right_of": ["place {a} on the right side of {b}"],
    "in_front_of": ["place {a} on the near side of {b}"],
    "behind": ["place {a} on the far side of {b}"],
    "next_to": ["nudge {a} until it touches {b}"],
    "between": ["place {a} in the middle of {b} and {c}"],
    "and": ["{p1}; after that, {p2}"],
}


def shape_word(shape, split, rng):
    words = (TEST_SHAPE_WORDS if split == "test" else SHAPE_WORDS)[shape]
    return words[rng.integers(len(words))]


def _relational(i, objects, pos, zones, split, rng):
    """A relation that singles out object i among the objects sharing its shape, or None."""
    o = objects[i]
    same = [k for k, p in enumerate(objects) if p["shape"] == o["shape"]]
    if len(same) < 2:  # a relation is only needed (and only natural) when the shape alone is ambiguous
        return None
    word = shape_word(o["shape"], split, rng)
    options = []
    for z in zones:
        d = {k: np.linalg.norm(pos[k, :2] - z["xy"]) for k in same}
        best = sorted(d, key=d.get)
        if best[0] == i and d[best[1]] - d[i] >= MARGIN:
            options.append(f"the {word} nearest the {z['color']} zone")
    ys = {k: pos[k, 1] for k in same}
    order = sorted(ys, key=ys.get)
    if order[0] == i and ys[order[1]] - ys[i] >= MARGIN:
        options.append(f"the leftmost {word}")
    if order[-1] == i and ys[i] - ys[order[-2]] >= MARGIN:
        options.append(f"the rightmost {word}")
    return options[rng.integers(len(options))] if options else None


def refer(i, objects, split="train", rng=None, pos=None, zones=None):
    """Shortest attribute description of object i that no other object matches (or a relational one)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    o = objects[i]
    if pos is not None and rng.random() < RELATIONAL_P:
        rel = _relational(i, objects, pos, zones or [], split, rng)
        if rel:
            return rel
    for keys in (("shape",), ("color", "shape"), ("size", "shape"), ("size", "color", "shape")):
        matches = [p for p in objects if all(p[k] == o[k] for k in keys)]
        if len(matches) == 1:
            words = {"size": SIZE_WORDS[o["size"]], "color": o["color"], "shape": shape_word(o["shape"], split, rng)}
            return "the " + " ".join(words[k] for k in keys)
    raise ValueError("objects are not distinguishable")  # scene_spec keeps (shape, size, color) unique


def instruction(goal, scene, split, rng, pos=None):
    """(text, template) for a goal. pos: start positions, enables relational references."""
    bank = TEST if split == "test" else TRAIN
    objs, zones = scene["objects"], scene["zones"]
    template = bank[goal["type"]][rng.integers(len(bank[goal["type"]]))]
    if goal["type"] == "and":
        parts = [instruction(p, scene, split, rng, pos)[0] for p in goal["parts"]]
        return template.format(p1=parts[0], p2=parts[1]), template
    ref = lambda k: refer(k, objs, split, rng, pos, zones)
    a = ref(goal["a"])
    if goal["type"] == "in_zone":
        return template.format(a=a, zone=zones[goal["b"]]["color"]), template
    if goal["type"] == "between":
        return template.format(a=a, b=ref(goal["b"]), c=ref(goal["c"])), template
    return template.format(a=a, b=ref(goal["b"])), template


# ----------------------------------------------------------------------------- hindsight captions
def _direction(v):
    """Dominant table-frame direction of a displacement, in the words the goals use."""
    if abs(v[1]) >= abs(v[0]):
        return "to the left" if v[1] < 0 else "to the right"
    return "forward" if v[0] > 0 else "back"


def caption(scene, init_pos, final_pos, rng=None):
    """True one-sentence description of what changed between init_pos and final_pos ((N, 3) each)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    objs, zones = scene["objects"], scene["zones"]
    moved = np.linalg.norm(final_pos[:, :2] - init_pos[:, :2], axis=1)
    clauses = []
    for i in np.argsort(-moved):
        if moved[i] < 0.02:
            continue
        name = refer(int(i), objs, "train", rng)
        if not on_table(final_pos[i]):
            clauses.append(f"{name} fell off the table")
            continue
        text = f"{name} moved {100 * moved[i]:.0f} cm {_direction(final_pos[i, :2] - init_pos[i, :2])}"
        for z in zones:
            was = np.linalg.norm(init_pos[i, :2] - z["xy"]) <= sx.ZONE_RADIUS
            now = np.linalg.norm(final_pos[i, :2] - z["xy"]) <= sx.ZONE_RADIUS
            if now and not was:
                text += f" and ended in the {z['color']} zone"
            elif was and not now:
                text += f" and left the {z['color']} zone"
        gaps = [(np.linalg.norm(final_pos[i, :2] - final_pos[k, :2]) - objs[i]["radius"] - objs[k]["radius"], k)
                for k in range(len(objs)) if k != i]
        if gaps:
            gap, k = min(gaps)
            before = np.linalg.norm(init_pos[i, :2] - init_pos[k, :2]) - objs[i]["radius"] - objs[k]["radius"]
            if gap <= NEXT_GAP < before:
                text += f", next to {refer(k, objs, 'train', rng)}"
        clauses.append(text)
    if not clauses:
        return "nothing moved"
    return "; ".join(clauses)


# Held-out templates and shape words must never appear in training.
assert not set(itertools.chain(*TRAIN.values())) & set(itertools.chain(*TEST.values()))
assert not set(itertools.chain(*SHAPE_WORDS.values())) & set(itertools.chain(*TEST_SHAPE_WORDS.values()))
