"""Sample a v3 scene: objects, distractors, zones, camera, look and split. Pure numpy, no simulator.

A spec is a JSON-able dict; env.set_scene(spec) builds it in MuJoCo.

Visual variety (milestone 4), all drawn per scene and stored in the spec:
    camera      one of the training views; test scenes use a held-out view with probability held_share
    visual      table and floor color, key / fill light strength, key light direction
    rgb_jitter  a small per-object color shift (names stay valid: "red" is still clearly red)
    distractors 0 to 2 neutral pucks / posts that can be pushed but are never named in a goal
"""
import numpy as np

from gen_v3 import scene_xml as sx

SPLIT_P = {"train": 0.70, "val": 0.15, "test": 0.15}
N_OBJECTS = (3, 6)  # inclusive
N_ZONES = (3, 4)
N_DISTRACTORS = (0, 2)
OBJ_FIELD = 0.18  # object centers in [-0.18, 0.18]^2
ZONE_FIELD = 0.20  # zone centers; with the radius, everything stays inside |x|, |y| <= 0.25
OBJ_GAP = 0.03  # min free space between object footprints at start
ZONE_GAP = 0.03
TRAIN_CAMERAS = ("cam0", "cam1", "cam2", "cam3", "cam4")
HELD_CAMERAS = ("held1", "held2")
RGB_JITTER = 0.05


def bounding_radius(shape, size):
    a = sx.SIZES[size]
    return a * {"cube": 1.42, "L": 1.42, "T": 1.42, "bar": 1.68}.get(shape, 1.0)


def sample_visual(rng):
    gray = rng.uniform(0.35, 0.75)
    tint = rng.uniform(-0.06, 0.06, 3)
    return {
        "table_rgb": [float(v) for v in np.clip(gray + tint, 0, 1)],
        "floor_rgb": [float(v) for v in rng.uniform(0.15, 0.40, 3)],
        "key_light": float(rng.uniform(0.45, 0.9)),
        "fill_light": float(rng.uniform(0.15, 0.4)),
        "key_dir": [float(rng.uniform(-0.5, 0.5)), float(rng.uniform(-0.5, 0.5)), -1.0],
    }


def sample_scene(seed, scene_idx, cameras=("cam0",), held_cameras=(), held_share=0.0, visual=False,
                 distractors=False):
    """cameras: views for train / val (and test, unless a held-out view is drawn)."""
    rng = np.random.default_rng([seed, scene_idx])
    split = str(rng.choice(list(SPLIT_P), p=list(SPLIT_P.values())))

    zones = []
    zone_colors = rng.permutation(list(sx.COLORS))[: rng.integers(N_ZONES[0], N_ZONES[1] + 1)]
    for color in zone_colors:
        for _ in range(200):
            xy = rng.uniform(-ZONE_FIELD, ZONE_FIELD, 2)
            if all(np.linalg.norm(xy - z["xy"]) > 2 * sx.ZONE_RADIUS + ZONE_GAP for z in zones):
                zones.append({"color": str(color), "xy": xy})
                break

    n_obj = int(rng.integers(N_OBJECTS[0], N_OBJECTS[1] + 1))
    objects, used_pool, used_attrs = [], set(), set()
    for _ in range(500):
        if len(objects) == n_obj:
            break
        shape = str(rng.choice(sx.SHAPES))
        size = str(rng.choice(list(sx.SIZES)))
        color = str(rng.choice(list(sx.COLORS)))
        if (shape, size, color) in used_attrs:
            continue
        free = [k for k in range(sx.COPIES) if f"{shape}_{size}_{k}" not in used_pool]
        if not free:
            continue
        xy = rng.uniform(-OBJ_FIELD, OBJ_FIELD, 2)
        r = bounding_radius(shape, size)
        if any(np.linalg.norm(xy - o["xy"]) < r + o["radius"] + OBJ_GAP for o in objects):
            continue
        # Mostly start objects outside zones, so zone goals are not already met.
        if any(np.linalg.norm(xy - z["xy"]) < sx.ZONE_RADIUS for z in zones) and rng.random() < 0.8:
            continue
        pool_name = f"{shape}_{size}_{free[0]}"
        used_pool.add(pool_name)
        used_attrs.add((shape, size, color))
        objects.append({"pool_name": pool_name, "shape": shape, "size": size, "color": color,
                        "xy": xy, "yaw": float(rng.uniform(-np.pi, np.pi)), "radius": r})

    # Drawn from a separate stream so turning these options on leaves objects and zones unchanged.
    vrng = np.random.default_rng([seed, scene_idx, 3])
    clutter = []
    if distractors:
        n_d = int(vrng.integers(N_DISTRACTORS[0], N_DISTRACTORS[1] + 1))
        names = sx.distractor_names()
        for name in vrng.permutation(names):
            if len(clutter) == n_d:
                break
            kind = str(name).split("_")[0]
            r = sx.DISTRACTORS[kind][0]
            for _ in range(100):
                xy = vrng.uniform(-OBJ_FIELD - 0.03, OBJ_FIELD + 0.03, 2)
                if all(np.linalg.norm(xy - o["xy"]) > r + o["radius"] + OBJ_GAP for o in objects + clutter):
                    clutter.append({"pool_name": str(name), "kind": kind, "radius": r, "xy": xy,
                                    "color": str(vrng.choice(list(sx.DISTRACTOR_COLORS)))})
                    break
    if visual:
        for o in objects:
            o["rgb_jitter"] = [float(v) for v in vrng.uniform(-RGB_JITTER, RGB_JITTER, 3)]
    camera = str(rng.choice(list(cameras)))
    if split == "test" and held_cameras and vrng.random() < held_share:
        camera = str(vrng.choice(list(held_cameras)))

    for o in objects + clutter:
        o["xy"] = [float(v) for v in o["xy"]]
    for z in zones:
        z["xy"] = [float(v) for v in z["xy"]]
    return {
        "scene_id": f"v3s{seed}_{scene_idx:05d}",
        "scene_idx": int(scene_idx),
        "seed": int(seed),
        "split": split,
        "camera": camera,
        "objects": objects,
        "distractors": clutter,
        "zones": zones,
        "visual": sample_visual(vrng) if visual else None,
    }
