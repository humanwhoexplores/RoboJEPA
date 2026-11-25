"""Instructions and hindsight captions for v6 goals.

Instructions say what should be true, never how to do it ("get the red cube into the bin", not "pick up..."
or "push..."): the action is the model's choice, and the held-out action-goal pairs (gen_v6.splits) test
whether it picks one it never saw succeed for that goal.

References name an object with the fewest attributes that single it out ("the lemon", "the red cube", "the small
blue L-block"), or, some of the time, with a relation ("the leftmost domino", "the second domino from the
left", "the cube nearest the green zone", "the ball in the bowl"). Two look-alikes (same kind, size and colour,
different hidden mass or friction) are told apart by the hidden property ("the heavier red cube", "the more
slippery blue ball") or by a relation. An object nothing singles out gets no instruction (None).

Test scenes (split "test") use held-out templates AND held-out kind and fixture words. Compound goals join the
parts' instructions. Hindsight captions describe what a rollout actually did, from its stored states.
"""
import itertools

import numpy as np

from gen_v6 import goals as G

KIND_WORDS = {  # train / val
    "cube": ["cube"], "cylinder": ["cylinder"], "prism": ["triangle block"], "L": ["L-block"], "T": ["T-block"],
    "hex": ["hexagon block"], "bar": ["bar", "long block"], "ball": ["ball"], "tube": ["tube"],
    "domino": ["domino"], "can": ["can"], "milk": ["milk carton", "carton"], "cereal": ["cereal box"],
    "bread": ["bread", "loaf"], "lemon": ["lemon"], "bottle": ["bottle"], "hammer": ["hammer"],
}
TEST_KIND_WORDS = {  # held out for test scenes
    "cube": ["square block"], "cylinder": ["round block"], "prism": ["triangular prism"], "L": ["L-shaped piece"],
    "T": ["T-shaped piece"], "hex": ["hex prism"], "bar": ["long beam"], "ball": ["sphere"], "tube": ["rod"],
    "domino": ["tile"], "can": ["tin"], "milk": ["juice pack"], "cereal": ["snack box"], "bread": ["bun"],
    "lemon": ["citrus fruit"], "bottle": ["flask"], "hammer": ["mallet"],
}
FIXTURE_WORDS = {"tray": ["tray"], "bin": ["bin"], "bowl": ["bowl"], "platform": ["platform"], "ramp": ["ramp"],
                 "wall": ["wall"], "seesaw": ["seesaw"], "cabinet": ["drawer"], "lidbox": ["lidded box"]}
TEST_FIXTURE_WORDS = {"tray": ["shallow dish"], "bin": ["crate"], "bowl": ["basin"], "platform": ["pedestal"],
                      "ramp": ["incline"], "wall": ["low barrier"], "seesaw": ["teeter board"],
                      "cabinet": ["sliding drawer"], "lidbox": ["chest"]}
SIZE_WORDS = {"s": "small", "l": "large"}
PROP_WORDS = {"mass": ("lighter", "heavier"), "friction": ("more slippery", "grippier")}
TEST_PROP_WORDS = {"mass": ("less heavy", "weightier"), "friction": ("smoother", "less slippery")}
ORDINALS = ["", "second", "third", "fourth", "fifth"]
RELATIONAL_P = 0.3  # share of references that use a relation when one singles the object out
PROPERTY_P = 0.6  # share of look-alike references that use the hidden property (the rest: a relation)
MARGIN = 0.03  # a relational reference needs the runner-up to be at least 3 cm worse

TRAIN = {
    "in_zone": ["get {a} into the {zone} zone", "move {a} to the {zone} area", "{a} should end up in the {zone} zone"],
    "left_of": ["get {a} to the left of {b}", "move {a} so it is left of {b}"],
    "right_of": ["get {a} to the right of {b}", "move {a} so it is right of {b}"],
    "in_front_of": ["get {a} in front of {b}", "move {a} so it is in front of {b}"],
    "behind": ["get {a} behind {b}", "move {a} so it is behind {b}"],
    "next_to": ["move {a} next to {b}", "get {a} beside {b}"],
    "between": ["move {a} between {b} and {c}", "get {a} into the gap between {b} and {c}"],
    "on_top": ["put {a} on top of {b}", "get {a} onto {b}", "stack {a} on {b}"],
    "off_of": ["get {a} off {b}", "take {a} off {b}"],
    "inside": ["put {a} {prep} the {f}", "get {a} {into} the {f}", "{a} should end up {prep} the {f}"],
    "count_in": ["get at least {n} things {into} the {f}", "make sure {n} or more objects are {prep} the {f}"],
    "upright": ["stand {a} up", "get {a} standing upright"],
    "lying": ["knock {a} over", "get {a} lying on its side"],
    "rotated": ["turn {a} around where it is", "rotate {a} in place"],
    "off_table": ["get {a} off the table", "make {a} end up off the table"],
    "touching": ["make {a} touch {b}", "get {a} in contact with {b}"],
    "moved_untouched": ["move {a} without touching it", "get {a} moving without the gripper touching it"],
    "drawer_open": ["open the {f}", "get the {f} open"],
    "drawer_closed": ["close the {f}", "get the {f} shut"],
    "lid_open": ["open the lid of the {f}", "get the {f} open"],
    "lid_closed": ["close the lid of the {f}", "get the {f} shut"],
    "hidden": ["hide {a} in the {f}", "put {a} in the {f} and shut it"],
    "and": ["{parts}", "first {parts}"],
}
TEST = {
    "in_zone": ["have {a} finish inside the {zone} spot"],
    "left_of": ["{a} needs to finish on the left side of {b}"],
    "right_of": ["{a} needs to finish on the right side of {b}"],
    "in_front_of": ["{a} needs to finish on the near side of {b}"],
    "behind": ["{a} needs to finish on the far side of {b}"],
    "next_to": ["have {a} finish right beside {b}"],
    "between": ["have {a} finish in the middle of {b} and {c}"],
    "on_top": ["have {a} resting on {b}"],
    "off_of": ["have {a} no longer resting on {b}"],
    "inside": ["have {a} finish {prep} the {f}"],
    "count_in": ["have the {f} hold {n} or more objects"],
    "upright": ["have {a} standing on its end"],
    "lying": ["have {a} toppled onto its side"],
    "rotated": ["have {a} face a new way without leaving its spot"],
    "off_table": ["have {a} leave the tabletop"],
    "touching": ["have {a} resting against {b}"],
    "moved_untouched": ["have {a} shift without the hand ever reaching it"],
    "drawer_open": ["have the {f} pulled out"],
    "drawer_closed": ["have the {f} pushed in"],
    "lid_open": ["have the {f} standing open"],
    "lid_closed": ["have the {f} closed up"],
    "hidden": ["have {a} shut away in the {f}"],
    "and": ["{parts}; that is all"],
}
JOIN = {"train": [", then ", " and then "], "test": ["; after that, "]}


def _bank(split):
    test = split == "test"
    return (TEST if test else TRAIN, TEST_KIND_WORDS if test else KIND_WORDS,
            TEST_FIXTURE_WORDS if test else FIXTURE_WORDS, TEST_PROP_WORDS if test else PROP_WORDS)


def kind_word(kind, split, rng):
    words = _bank(split)[1][kind]
    return words[rng.integers(len(words))]


def fixture_word(kind, split, rng):
    words = _bank(split)[2][kind]
    return words[rng.integers(len(words))]


def _attrs(o):
    return {"kind": G.kind(o), "size": o["size"], "color": o.get("color")}


def _plain(i, objects, keys, split, rng):
    o = _attrs(objects[i])
    words = {"size": SIZE_WORDS[o["size"]], "color": o["color"], "kind": kind_word(o["kind"], split, rng)}
    return "the " + " ".join(words[k] for k in keys)


def _same(i, objects):
    o = _attrs(objects[i])
    return [k for k in range(len(objects)) if _attrs(objects[k]) == o]


def _full_keys(o):
    return ("size", "color", "kind") if o["color"] else ("size", "kind")


def _relational(i, spec, pos, split, rng):
    """Relations that single out object i among the objects that look the same as it (or share its kind)."""
    objs = spec["objects"]
    same = _same(i, objs)
    if len(same) < 2:
        same = [k for k in range(len(objs)) if G.kind(objs[k]) == G.kind(objs[i])]
        if len(same) < 2:
            return []
        word = kind_word(G.kind(objs[i]), split, rng)
    else:
        word = _plain(i, objs, _full_keys(_attrs(objs[i]))[1:], split, rng)[4:]  # "red cube", "can"
    out = []
    ys = sorted(same, key=lambda k: pos[k, 1])
    gaps = np.diff([pos[k, 1] for k in ys])
    if len(gaps) and gaps.min() >= MARGIN:
        r = ys.index(i)
        if r == 0:
            out.append(f"the leftmost {word}")
        elif r == len(ys) - 1:
            out.append(f"the rightmost {word}")
        elif len(ys) <= 5:
            from_left = r <= len(ys) - 1 - r
            k = r if from_left else len(ys) - 1 - r
            out.append(f"the {ORDINALS[k]} {word} from the {'left' if from_left else 'right'}")
    for z in spec["zones"]:
        d = {k: np.linalg.norm(pos[k, :2] - np.asarray(z["xy"])) for k in same}
        best = sorted(d, key=d.get)
        if best[0] == i and d[best[1]] - d[i] >= MARGIN:
            out.append(f"the {word} nearest the {z['color']} zone")
    for f, fx in enumerate(spec["fixtures"]):
        if fx["kind"] in G.cat.CONTAINERS:
            ins = G.inside_fx(spec, f, pos[None])[0]
            if ins[i] and sum(ins[k] for k in same) == 1:
                prep = "on" if fx["kind"] == "platform" else "in"
                out.append(f"the {word} {prep} the {fixture_word(fx['kind'], split, rng)}")
    return out


def refer(i, spec, split="train", rng=None, pos=None):
    """A description that singles out object i, or None. pos: positions (N, 3) the description is true for."""
    rng = rng if rng is not None else np.random.default_rng(0)
    objs = spec["objects"]
    if i >= len(objs):  # in_zone goals pass a zone index as "b" (35/1200 scenes crashed here)
        return None
    o = _attrs(objs[i])
    look = spec.get("lookalike")
    if look and i in look["pair"] and len(_same(i, objs)) == 2:
        if rng.random() < PROPERTY_P or pos is None:
            words = _bank(split)[3][look["property"]]
            w = words[1] if i == look["high"] else words[0]
            return f"the {w} " + _plain(i, objs, _full_keys(o)[1:], split, rng)[4:]
    if pos is not None and rng.random() < RELATIONAL_P:
        rel = _relational(i, spec, pos, split, rng)
        if rel:
            return rel[rng.integers(len(rel))]
    for keys in (("kind",), ("color", "kind"), ("size", "kind"), ("size", "color", "kind")):
        if "color" in keys and not o["color"]:
            continue
        if sum(all(_attrs(p)[k] == o[k] for k in keys) for p in objs) == 1:
            return _plain(i, objs, keys, split, rng)
    if pos is not None:
        rel = _relational(i, spec, pos, split, rng)
        if rel:
            return rel[rng.integers(len(rel))]
    return None


def instruction(goal, spec, split, rng, pos=None):
    """(text, template) for a goal, or (None, None) if an object in it cannot be singled out."""
    bank = _bank(split)[0]
    t = goal["type"]
    if t == "and":
        parts = [instruction(p, spec, split, rng, pos)[0] for p in goal["parts"]]
        if any(p is None for p in parts):
            return None, None
        joins = JOIN["test" if split == "test" else "train"]
        joined = parts[0]
        for p in parts[1:]:
            joined += joins[rng.integers(len(joins))] + p
        template = bank["and"][rng.integers(len(bank["and"]))]
        return template.format(parts=joined), template
    template = bank[t][rng.integers(len(bank[t]))]
    fill = {}
    for k in ("a", "b", "c"):
        if k in goal:
            fill[k] = refer(goal[k], spec, split, rng, pos)
            if fill[k] is None:
                return None, None
    if t == "in_zone":
        fill["zone"] = spec["zones"][goal["b"]]["color"]
        fill.pop("b", None)
    if "f" in goal:
        fk = spec["fixtures"][goal["f"]]["kind"]
        fill["f"] = fixture_word(fk, split, rng)
        fill["prep"], fill["into"] = ("on", "onto") if fk == "platform" else ("in", "into")
    if "n" in goal:
        fill["n"] = str(goal["n"])
    return template.format(**fill), template


def all_instruction_words():
    """Every template string, for the train / test overlap checks."""
    return set(itertools.chain(*TRAIN.values())), set(itertools.chain(*TEST.values()))


# ----------------------------------------------------------------------------- hindsight captions
def _direction(v):
    """Dominant table-frame direction of a displacement, in the words the goals use."""
    if abs(v[1]) >= abs(v[0]):
        return "to the left" if v[1] < 0 else "to the right"
    return "forward" if v[0] > 0 else "back"


def caption(spec, S0, S1, rng=None):
    """True one-sentence description of what changed from state S0 to state S1 (goals.initial_state dicts)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    objs = spec["objects"]
    p0, p1 = np.asarray(S0["pos"]), np.asarray(S1["pos"])
    moved = np.linalg.norm(p1[:, :2] - p0[:, :2], axis=1)
    one0 = {k: np.asarray(v)[None] for k, v in S0.items()}
    one1 = {k: np.asarray(v)[None] for k, v in S1.items()}
    clauses = []
    for f, fx in enumerate(spec["fixtures"]):
        q0, q1 = np.nan_to_num(S0["fxs"][f]), np.nan_to_num(S1["fxs"][f])
        word = fixture_word(fx["kind"], "train", rng)
        if fx["kind"] == "cabinet" and abs(q1 - q0) > 0.02:
            clauses.append(f"the {word} was {'pulled open' if q1 > q0 else 'pushed shut'}")
        if fx["kind"] == "lidbox" and abs(q1 - q0) > 0.3:
            clauses.append(f"the {word} was {'opened' if q1 > q0 else 'closed'}")
        if fx["kind"] == "seesaw" and abs(q1 - q0) > 0.15:
            clauses.append("the seesaw tipped")
    for i in np.argsort(-moved):
        i = int(i)
        name = refer(i, spec, "train", rng, p0)
        if name is None:
            continue
        up0, up1 = abs(G.zz(S0["quat"][i])), abs(G.zz(S1["quat"][i]))
        turned = abs((G.yaw_of(S1["quat"][i]) - G.yaw_of(S0["quat"][i]) + np.pi) % (2 * np.pi) - np.pi)
        if not G.on_table(p1[i]):
            clauses.append(f"{name} fell off the table")
            continue
        bits = []
        if moved[i] >= 0.02:
            bits.append(f"moved {100 * moved[i]:.0f} cm {_direction(p1[i, :2] - p0[i, :2])}")
        if G.kind(objs[i]) in G.cat.TALL and up0 > G.UPRIGHT_COS and up1 < G.LYING_COS:
            bits.append("was knocked over")
        elif G.kind(objs[i]) in G.cat.TALL and up0 < G.LYING_COS and up1 > G.UPRIGHT_COS:
            bits.append("was stood up")
        elif moved[i] < 0.02 and turned > np.radians(30) and up1 > G.UPRIGHT_COS:
            bits.append("turned in place")
        if not bits:
            continue
        for z, zone in enumerate(spec["zones"]):
            g = {"type": "in_zone", "a": i, "b": z}
            was, now = G.holds(g, one0, spec)[0], G.holds(g, one1, spec)[0]
            if now and not was:
                bits.append(f"ended in the {zone['color']} zone")
            elif was and not now:
                bits.append(f"left the {zone['color']} zone")
        for f, fx in enumerate(spec["fixtures"]):
            if fx["kind"] not in G.cat.CONTAINERS:
                continue
            was = G.inside_fx(spec, f, p0[None], np.asarray(S0["fxs"])[None])[0, i]
            now = G.inside_fx(spec, f, p1[None], np.asarray(S1["fxs"])[None])[0, i]
            word = fixture_word(fx["kind"], "train", rng)
            prep = "on" if fx["kind"] == "platform" else "in"
            if now and not was:
                bits.append(f"ended {prep} the {word}")
            elif was and not now:
                bits.append(f"came out of the {word}" if prep == "in" else f"came off the {word}")
        for k in range(len(objs)):
            if k == i:
                continue
            g = {"type": "on_top", "a": i, "b": k}
            if G.holds(g, one1, spec)[0] and not G.holds(g, one0, spec)[0]:
                other = refer(k, spec, "train", rng, p0)
                if other:
                    bits.append(f"ended on top of {other}")
        clauses.append(f"{name} " + (", ".join(bits[:-1]) + " and " + bits[-1] if len(bits) > 1 else bits[0]))
    return "; ".join(clauses) if clauses else "nothing moved"


# Held-out templates and words must never appear in training.
assert not set(itertools.chain(*TRAIN.values())) & set(itertools.chain(*TEST.values()))
assert not set(itertools.chain(*KIND_WORDS.values())) & set(itertools.chain(*TEST_KIND_WORDS.values()))
assert not set(itertools.chain(*FIXTURE_WORDS.values())) & set(itertools.chain(*TEST_FIXTURE_WORDS.values()))
assert set(TRAIN) == set(TEST) == set(G.GOAL_TYPES) | {"and"}
