"""Sample a v6 scene: family, split, objects with hidden properties, fixtures, zones, camera, look. Pure numpy.

Scene families (scene idx % 6 picks the family, so every family gets the same number of scenes):
    blocks      v5-style blocks, zones and distractors; a low wall in some; a look-alike pair in a third of scenes
    containers  a tray, bin and / or bowl with everyday objects, balls and blocks; some start inside
    stacking    tall things (tubes, bottles, cartons, dominoes) upright or lying, blocks, stacks, a platform
    rolling     a ramp, balls, lying tubes and cans, blocks, sometimes a wall
    chain       a domino row, a seesaw with something on one end, tightly packed blocks
    tools       (stage 2) a drawer cabinet and / or a lidded box, a hammer, everyday objects

Splits (one factor changes in each test split, as in v5):
    train / val / test_id   in distribution
    test_lang   held-out instruction templates and words        test_cam    a held-out camera view
    test_pos    an object starts inside the held-out region      test_obj    a held-out kind + colour combination
    test_phys   an object with an extreme hidden mass or friction (never in any other split)
    test_family a held-out combination of things: a ramp that feeds a container (never in any other split)
The action-goal hold-out (test_act) is a label mask, not a scene split: see gen_v6.splits.
"""
import numpy as np

from gen_v3 import scene_spec as v3s
from gen_v3 import scene_xml as sx
from gen_v6 import catalog as cat

FAMILIES = ["blocks", "containers", "stacking", "rolling", "chain", "tools"]
SPLIT_P = {"train": 0.58, "val": 0.06, "test_id": 0.06, "test_lang": 0.05, "test_cam": 0.05, "test_pos": 0.05,
           "test_obj": 0.05, "test_phys": 0.05, "test_family": 0.05}
HELD_REGION = (np.array([0.09, -0.09]), 0.075)  # v5's
HELD_COMBOS = {("cube", "purple"), ("cylinder", "orange"), ("prism", "red"), ("L", "blue"), ("T", "yellow"),
               ("hex", "green"), ("bar", "red"),  # v5's
               ("ball", "green"), ("tube", "red"), ("domino", "blue")}
FIELD = 0.18  # object centres within +-FIELD
FX_FIELD = 0.13  # fixture centres within +-FX_FIELD
CAB_X_MIN = 0.03  # the cabinet's centre at least this far out (see fam_tools)
LID_X_MIN = -0.06  # and the lid box's
GAP = 0.025
LOOKALIKE_P = {"blocks": 0.5, "rolling": 0.35, "containers": 0.3, "stacking": 0.25}  # scenes with a look-alike pair


def footprint(kind, size, pose="upright"):
    """Radius of a disc that covers the object's footprint."""
    if kind in sx.SHAPES:
        return v3s.bounding_radius(kind, size)
    lying = pose == "lying"
    return {"ball": cat.BALL_R.get(size, 0.022), "tube": cat.TUBE_HALF if lying else cat.TUBE_R,
            "domino": float(np.hypot(*cat.DOMINO_HALF[:2])), "can": 0.042 if lying else 0.026,
            "milk": 0.03, "cereal": 0.053, "bread": 0.032, "lemon": 0.035, "bottle": 0.085 if lying else 0.03,
            "hammer": 0.12}[kind]


def _rect_dist(xy, f):
    """Distance from point xy to fixture f's footprint rectangle."""
    hx, hy = cat.fixture_footprint(f["kind"])
    c, s = np.cos(f["yaw"]), np.sin(f["yaw"])
    d = np.asarray(xy) - np.asarray(f["xy"])
    local = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
    return float(np.linalg.norm(np.maximum(np.abs(local) - [hx, hy], 0)))


class Layout:
    """Tracks what is already on the table, so new things go on free space."""

    def __init__(self, rng):
        self.rng, self.discs, self.fixtures = rng, [], []

    def free(self, xy, r, gap=GAP, skip_fixture=None):
        if any(np.linalg.norm(np.asarray(xy) - d) < r + rd + gap for d, rd in self.discs):
            return False
        return all(_rect_dist(xy, f) > r + gap for f in self.fixtures if f["kind"] != skip_fixture)

    def place(self, r, field=FIELD, tries=300, gap=GAP, near=None, region=None):
        for _ in range(tries):
            if near is not None:
                xy = np.asarray(near[0]) + self.rng.uniform(-near[1], near[1], 2)
            else:
                xy = self.rng.uniform(-field, field, 2)
            if np.abs(xy).max() > field + 0.03:
                continue
            if region is not None and not region(xy):
                continue
            if self.free(xy, r, gap):
                self.discs.append((xy, r))
                return xy
        return None

    def add_fixture(self, kind, tries=300, yaw=None, xy=None, beside=None):
        """beside: a fixture this one may sit right next to (its own footprints are still kept apart)."""
        hx, hy = cat.fixture_footprint(kind)
        for _ in range(tries):
            f_xy = np.asarray(xy) if xy is not None else self.rng.uniform(-FX_FIELD, FX_FIELD, 2)
            f_yaw = float(self.rng.uniform(-np.pi, np.pi)) if yaw is None else float(yaw)
            f = {"kind": str(kind), "xy": f_xy, "yaw": f_yaw}
            corners = [np.asarray(f_xy) + np.array([[np.cos(f_yaw), -np.sin(f_yaw)], [np.sin(f_yaw), np.cos(f_yaw)]]) @ np.array([sx_, sy_])
                       for sx_ in (-hx, hx) for sy_ in (-hy, hy)]
            if np.abs(corners).max() > 0.26:
                if xy is not None:
                    return None
                continue
            if any(_rect_dist(d, f) < rd + GAP for d, rd in self.discs):
                if xy is not None:
                    return None
                continue
            if any(np.linalg.norm(f_xy - np.asarray(g["xy"])) < np.hypot(hx, hy) + np.hypot(*cat.fixture_footprint(g["kind"])) + 0.01
                   for g in self.fixtures if g is not beside):
                if xy is not None:
                    return None
                continue
            self.fixtures.append(f)
            return f
        return None


def in_region(xy):
    c, r = HELD_REGION
    return np.linalg.norm(np.asarray(xy) - c) < r


class Builder:
    def __init__(self, rng, split, stage2):
        self.rng, self.split, self.stage2 = rng, split, stage2
        self.L = Layout(rng)
        self.objects = []
        self.used_pool = set()
        self.used_attrs = set()

    # pool bookkeeping
    def _pool(self, kind, size):
        n = {"ball": 2, "tube": 2, "domino": cat.N_DOMINOES, "hammer": 1}.get(kind)
        if kind in cat.RS_OBJECTS:
            n = cat.RS_OBJECTS[kind][0]
        if kind in sx.SHAPES:
            n = sx.COPIES
        for k in range(n):
            name = f"{kind}_{size}_{k}"
            if name not in self.used_pool:
                return name
        return None

    def _color(self, kind, avoid_held=True):
        cols = list(sx.COLORS)
        for _ in range(50):
            c = str(self.rng.choice(cols))
            if avoid_held and self.split != "test_obj" and (kind, c) in HELD_COMBOS:
                continue
            return c
        return cols[0]

    def region_ok(self, xy):
        return self.split == "test_pos" or not in_region(xy)

    def add(self, kind, size="s", color=None, pose="upright", xy=None, yaw=None, on=None, props=None, attrs_unique=True,
            near=None, gap=GAP):
        """Add an object; returns its index or None if there is no room / no pool body left."""
        name = self._pool(kind, size)
        if name is None:
            return None
        colourable = kind in cat.COLOURABLE
        if colourable and color is None:
            for _ in range(30):
                color = self._color(kind)
                if not attrs_unique or (kind, size, color) not in self.used_attrs:
                    break
            else:
                return None
        if colourable and attrs_unique and (kind, size, color) in self.used_attrs:
            return None
        r = footprint(kind, size, pose)
        if xy is None:
            xy = self.L.place(r, near=near, gap=gap, region=self.region_ok)
            if xy is None:
                return None
        else:
            xy = np.asarray(xy, float)
            if on is None:
                self.L.discs.append((xy, r))
        self.used_pool.add(name)
        if colourable:
            self.used_attrs.add((kind, size, color))
        o = {"pool_name": name, "kind": kind, "shape": kind, "size": size, "color": color if colourable else None,
             "xy": [float(v) for v in xy], "yaw": float(self.rng.uniform(-np.pi, np.pi) if yaw is None else yaw),
             "pose": pose, "radius": float(r), "props": props or cat.sample_props(self.rng)}
        if on is not None:
            o["on"] = on
        self.objects.append(o)
        return len(self.objects) - 1

    def block(self, **kw):
        shape = str(self.rng.choice(sx.SHAPES))
        size = str(self.rng.choice(list(sx.SIZES)))
        return self.add(shape, size, **kw)

    def any_small(self, kinds=("block", "ball", "can", "lemon", "bread"), **kw):
        k = str(self.rng.choice(kinds))
        if k == "block":
            return self.block(**kw)
        if k == "ball":
            return self.add("ball", str(self.rng.choice(list(cat.BALL_R))), **kw)
        return self.add(k, "s", **kw)


def _zones(rng, L, n, field=0.2):
    zones = []
    for color in rng.permutation(list(sx.COLORS))[:n]:
        for _ in range(200):
            xy = rng.uniform(-field, field, 2)
            if any(np.linalg.norm(xy - np.asarray(z["xy"])) < 2 * sx.ZONE_RADIUS + 0.03 for z in zones):
                continue
            if any(_rect_dist(xy, f) < sx.ZONE_RADIUS + 0.01 for f in L.fixtures):
                continue
            zones.append({"color": str(color), "xy": xy})
            break
    return zones


# ----------------------------------------------------------------------------- families
def fam_blocks(B):
    rng = B.rng
    if rng.random() < 0.3:
        B.L.add_fixture("wall")
    n = int(rng.integers(3, 7))
    if rng.random() < LOOKALIKE_P["blocks"]:
        lookalike(B)
    for _ in range(40):
        if len(B.objects) >= n:
            break
        B.block()


def lookalike(B, kinds=None):
    """Two objects that look identical but differ in one hidden property (mass or friction)."""
    rng = B.rng
    for _ in range(20):
        if kinds is None:
            kind, size = str(rng.choice(["cube", "cylinder", "hex", "ball"])), "s" if rng.random() < 0.5 else "l"
        else:
            kind, size = kinds
        color = B._color(kind)
        if kind in cat.RS_OBJECTS:
            color = None
        prop = str(rng.choice(["mass", "friction"]))
        lo = cat.sample_props(rng, extreme_ok=B.split == "test_phys")
        hi = dict(lo)
        if prop == "mass":
            lo["mass_scale"], hi["mass_scale"] = float(rng.uniform(0.4, 0.6)), float(rng.uniform(2.0, 2.5))
        else:
            lo["friction"], hi["friction"] = float(rng.uniform(0.3, 0.4)), float(rng.uniform(0.95, 1.1))
        n0 = len(B.objects)
        i = B.add(kind, size, color=color, props=lo, attrs_unique=False)
        j = B.add(kind, size, color=color, props=hi, attrs_unique=False) if i is not None else None
        if i is None or j is None:
            del B.objects[n0:]
            continue
        if rng.random() < 0.5:  # which of the two is listed first carries no information
            B.objects[i], B.objects[j] = B.objects[j], B.objects[i]
        key = "mass_scale" if prop == "mass" else "friction"
        low, high = sorted((i, j), key=lambda k: B.objects[k]["props"][key])
        B.lookalike = {"pair": [i, j], "property": prop, "low": low, "high": high}
        return True
    return False


def fam_containers(B, with_ramp=False):
    rng = B.rng
    kinds = list(rng.permutation(["tray", "bin", "bowl"]))[: int(rng.integers(1, 3))]
    fxs = [B.L.add_fixture(k) for k in kinds]
    fxs = [f for f in fxs if f is not None]
    n_rs = int(rng.integers(1, 4))
    for k in rng.permutation(["can", "lemon", "bread", "milk", "cereal", "bottle"])[:n_rs]:
        B.add(str(k), "s")
    if rng.random() < LOOKALIKE_P["containers"]:
        lookalike(B)
    for _ in range(int(rng.integers(1, 3))):
        B.any_small(kinds=("block", "ball"))
    for f in fxs:  # sometimes something already sits in a container
        if rng.random() < 0.25 and f["kind"] in ("tray", "bin", "bowl"):
            kind = str(rng.choice(["lemon", "ball", "cube"]))
            size = "s"
            name = B._pool(kind, size)
            if name is not None:
                B.add(kind, size, xy=f["xy"], on=f["kind"], gap=0)
    if rng.random() < 0.3 and len(B.objects) < 6:
        B.block()


def fam_stacking(B):
    rng = B.rng
    if rng.random() < 0.5:
        B.L.add_fixture("platform")
    for k in rng.permutation(["tube", "bottle", "milk", "cereal", "can", "domino", "tube"])[: int(rng.integers(1, 4))]:
        k = str(k)
        pose = "lying" if k in ("tube", "can", "bottle") and rng.random() < 0.4 else "upright"
        B.add(k, "s", pose=pose)
    if rng.random() < LOOKALIKE_P["stacking"]:
        lookalike(B, (str(rng.choice(["cube", "cylinder"])), "s"))
    for _ in range(int(rng.integers(1, 3))):  # stacks of two
        base_shape = str(rng.choice(["cube", "cylinder", "hex"]))
        base = B.add(base_shape, "l")
        if base is None:
            continue
        top_shape = str(rng.choice(["cube", "cylinder", "hex", "T", "L"]))
        B.add(top_shape, "s", xy=B.objects[base]["xy"], on=base)
    for _ in range(int(rng.integers(0, 3))):
        B.block()


def fam_rolling(B, with_container=False):
    rng = B.rng
    # the ramp's low end points roughly at the middle of the table, so things roll into the open
    box = str(rng.choice(["bin", "bowl", "tray"])) if with_container else None
    for _ in range(200):
        xy = rng.uniform(-FX_FIELD, FX_FIELD, 2)
        to_mid = -xy / max(np.linalg.norm(xy), 1e-6)
        down = np.arctan2(to_mid[1], to_mid[0]) + rng.uniform(-0.6, 0.6)
        ramp = B.L.add_fixture("ramp", xy=xy, yaw=down + np.pi)  # ramp +x (its high end) points away from the middle
        if ramp is None or box is None:
            if ramp is not None:
                break
            continue
        # test_family: a container right at the foot of the ramp, so things that roll down can land in it
        u = np.array([np.cos(ramp["yaw"]), np.sin(ramp["yaw"])])
        foot = np.asarray(ramp["xy"]) - u * (cat.RAMP_LEN / 2 + cat.fixture_footprint(box)[0] + 0.012)
        if B.L.add_fixture(box, xy=foot, yaw=ramp["yaw"] + np.pi, beside=ramp) is not None:
            break
        B.L.fixtures.remove(ramp)
        ramp = None
    if rng.random() < 0.5:
        B.L.add_fixture("wall")
    if rng.random() < LOOKALIKE_P["rolling"]:  # two balls alike but for mass or friction: watch one roll first
        lookalike(B, ("ball", str(rng.choice(list(cat.BALL_R)))))
    for _ in range(int(rng.integers(1, 3))):
        B.add("ball", str(rng.choice(list(cat.BALL_R))))
    if rng.random() < 0.6:
        B.add(str(rng.choice(["tube", "can"])), "s", pose="lying")
    for _ in range(int(rng.integers(1, 4))):
        B.block()


def fam_chain(B):
    rng = B.rng
    # a domino row: 3 to 5 slabs facing along a line, 3.5 to 4.5 cm apart
    n = int(rng.integers(3, 6))
    gap = float(rng.uniform(0.035, 0.045))
    for _ in range(100):
        c = rng.uniform(-0.12, 0.12, 2)
        th = rng.uniform(-np.pi, np.pi)
        u = np.array([np.cos(th), np.sin(th)])
        pts = [c + u * gap * (k - (n - 1) / 2) for k in range(n)]
        r = footprint("domino", "s")
        if all(np.abs(p).max() < FIELD and B.L.free(p, r, 0.005) and B.region_ok(p) for p in pts):
            color = B._color("domino")
            for p in pts:
                B.add("domino", "s", color=color, xy=p, yaw=th, attrs_unique=False)
            break
    if rng.random() < 0.6:
        f = B.L.add_fixture("seesaw")
        if f is not None:
            u = np.array([np.cos(f["yaw"]), np.sin(f["yaw"])])
            end = np.asarray(f["xy"]) + u * (cat.SEESAW_PLANK[0] - 0.03) * (1 if rng.random() < 0.5 else -1)
            kind = str(rng.choice(["ball", "cube", "cylinder"]))
            B.add(kind, "s", xy=end, on="seesaw")
    # a tight cluster of 2 or 3 blocks
    first = B.block()
    if first is not None:
        o = B.objects[first]
        for _ in range(int(rng.integers(1, 3))):
            for _ in range(30):
                shape = str(rng.choice(["cube", "cylinder", "hex"]))
                size = str(rng.choice(list(sx.SIZES)))
                r = footprint(shape, size)
                ang = rng.uniform(-np.pi, np.pi)
                d = o["radius"] + r + rng.uniform(0.003, 0.008)
                xy = np.asarray(o["xy"]) + d * np.array([np.cos(ang), np.sin(ang)])
                if np.abs(xy).max() < FIELD and B.L.free(xy, r, 0.002) and B.region_ok(xy):
                    if B.add(shape, size, xy=xy) is not None:
                        break
    for _ in range(int(rng.integers(0, 2))):
        B.block()


def fam_tools(B):
    rng = B.rng
    has_cab = rng.random() < 0.55  # both rarely fit now that each keeps clear of the base
    has_box = (not has_cab) or rng.random() < 0.5
    if has_cab:
        # the drawer faces the robot (the cabinet's +x points at it, roughly): the fingers hook behind the handle
        # and pull toward the base, and the arm never reaches over the cabinet
        # and the cabinet stands at x >= CAB_X_MIN: nearer the base the elbow runs into its limit as the hand
        # pulls the drawer toward it (a sweep of cabinet poses: drawer actions all feasible from x = 0.03 on)
        f = None
        for _ in range(100):
            xy = np.array([rng.uniform(CAB_X_MIN, FX_FIELD), rng.uniform(-FX_FIELD, FX_FIELD)])
            f = B.L.add_fixture("cabinet", tries=1, yaw=float(np.pi + rng.uniform(-0.5, 0.5)), xy=xy)
            if f is not None:
                break
        if f is not None:
            f["open"] = float(rng.choice([0.0, 0.0, rng.uniform(0.04, cat.DRAWER_TRAVEL)]))
    if has_box:
        f = None
        for _ in range(100):  # the lid box too stays clear of the base (lid actions feasible from x = -0.06 on)
            xy = np.array([rng.uniform(LID_X_MIN, FX_FIELD), rng.uniform(-FX_FIELD, FX_FIELD)])
            f = B.L.add_fixture("lidbox", tries=1, yaw=float(rng.uniform(-0.6, 0.6)), xy=xy)
            if f is not None:
                break
        if f is not None:
            f["open"] = float(rng.choice([0.0, np.radians(rng.uniform(80, 105))]))
    B.add("hammer", "s")
    for k in rng.permutation(["can", "lemon", "bread", "ball", "cube"])[: int(rng.integers(2, 4))]:
        k = str(k)
        B.add(k, "s") if k != "ball" else B.add("ball", "s")


BUILD = {"blocks": fam_blocks, "containers": fam_containers, "stacking": fam_stacking, "rolling": fam_rolling,
         "chain": fam_chain, "tools": fam_tools}


def ok(spec, split):
    objs = spec["objects"]
    held_obj = [(o["kind"], o["color"]) in HELD_COMBOS for o in objs]
    in_reg = [in_region(o["xy"]) for o in objs]
    extreme = [cat.extreme(o["props"]) for o in objs]
    if split == "test_obj":
        return any(held_obj) and not any(in_reg) and not any(extreme)
    if split == "test_pos":
        return any(in_reg) and not any(held_obj) and not any(extreme)
    if split == "test_phys":
        return any(extreme) and not any(held_obj) and not any(in_reg)
    return not any(held_obj) and not any(in_reg) and not any(extreme)


def families(stage2=True):
    return FAMILIES if stage2 else FAMILIES[:5]


def sample_scene(seed, idx, stage2=True, family=None, retry=0):
    """retry > 0 draws a different layout with the same split, family and look (for a scene that did not settle)."""
    rng0 = np.random.default_rng([seed, idx, 60])
    split = str(rng0.choice(list(SPLIT_P), p=list(SPLIT_P.values())))
    fams = families(stage2)
    family = family or fams[idx % len(fams)]
    if split == "test_family":
        family = "rolling"  # with a container at the foot of the ramp (fam_rolling(with_container=True))
    cameras = v3s.HELD_CAMERAS if split == "test_cam" else v3s.TRAIN_CAMERAS
    for attempt in range(500):
        rng = np.random.default_rng([seed, idx, 61, attempt] + ([retry] if retry else []))
        B = Builder(rng, split, stage2)
        B.lookalike = None
        if split == "test_family":
            fam_rolling(B, with_container=True)
        else:
            BUILD[family](B)
        if split == "test_phys":  # make one object extreme
            cand = [i for i, o in enumerate(B.objects) if not (B.lookalike and i in B.lookalike["pair"])]
            if cand:
                k = int(rng.choice(cand))
                p = B.objects[k]["props"]
                if rng.random() < 0.5:
                    p["mass_scale"] = float(rng.choice([rng.uniform(0.3, 0.38), rng.uniform(2.7, 3.0)]))
                else:
                    p["friction"] = float(rng.choice([rng.uniform(0.2, 0.27), rng.uniform(1.13, 1.2)]))
        if len(B.objects) < 2:
            continue
        zones = _zones(rng, B.L, int(rng.integers(2, 4)) if family != "blocks" else int(rng.integers(3, 5)))
        spec = {"objects": B.objects}
        if not ok(spec, split):
            continue
        break
    else:
        raise RuntimeError(f"no scene for {family} / {split} (seed {seed}, idx {idx})")
    vrng = np.random.default_rng([seed, idx, 63])
    clutter = []
    n_d = int(vrng.integers(0, 3 if family == "blocks" else 2))
    for name in vrng.permutation(sx.distractor_names()):
        if len(clutter) == n_d:
            break
        kind = str(name).split("_")[0]
        r = sx.DISTRACTORS[kind][0]
        xy = B.L.place(r, field=FIELD + 0.03, region=B.region_ok)
        if xy is not None:
            clutter.append({"pool_name": str(name), "kind": kind, "radius": r, "xy": [float(v) for v in xy],
                            "color": str(vrng.choice(list(sx.DISTRACTOR_COLORS)))})
    for o in B.objects:
        o["rgb_jitter"] = [float(v) for v in vrng.uniform(-v3s.RGB_JITTER, v3s.RGB_JITTER, 3)]
    for z in zones:
        z["xy"] = [float(v) for v in z["xy"]]
    fixtures = [{**f, "xy": [float(v) for v in f["xy"]], "yaw": float(f["yaw"])} for f in B.L.fixtures]
    return {
        "scene_id": f"v6s{seed}_{idx:05d}", "scene_idx": int(idx), "seed": int(seed), "family": family,
        "v6_split": split, "split": "test" if split == "test_lang" else ("val" if split == "val" else "train"),
        "camera": str(vrng.choice(list(cameras))), "objects": B.objects, "distractors": clutter, "zones": zones,
        "fixtures": fixtures, "lookalike": B.lookalike, "visual": v3s.sample_visual(vrng), "attempt": attempt,
        "retry": int(retry),
    }
