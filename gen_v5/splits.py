"""v5 scene sampling with explicit, one-factor OOD test splits.

Each scene's split is drawn first; the scene itself comes from v3's sampler (gen_v3.scene_spec) and is
redrawn until it satisfies the split's constraint. Every OOD split changes exactly one factor:

    train / val / test_id  in distribution: training cameras, templates and words; no held-out object
                           combination; no object starts inside the held-out region
    test_lang   held-out instruction templates and shape words (v3's "test" language)
    test_cam    a held-out camera view (held1 / held2)
    test_pos    at least one object starts inside the held-out table region (a disc nobody starts in otherwise)
    test_obj    at least one object has a held-out shape+color combination (never seen in any other split)

spec["split"] keeps v3's meaning (it selects the language bank: "test" only for test_lang), and
spec["v5_split"] holds the split above.
"""
import numpy as np

from gen_v3 import scene_spec as v3s

SPLIT_P = {"train": 0.64, "val": 0.06, "test_id": 0.06, "test_lang": 0.06, "test_cam": 0.06, "test_pos": 0.06,
           "test_obj": 0.06}
HELD_REGION = (np.array([0.09, -0.09]), 0.075)  # center, radius (m): object start positions held out
HELD_COMBOS = {("cube", "purple"), ("cylinder", "orange"), ("prism", "red"), ("L", "blue"), ("T", "yellow"),
               ("hex", "green"), ("bar", "red")}  # one per shape, 35 of 42 combinations stay in training


def in_region(xy):
    c, r = HELD_REGION
    return np.linalg.norm(np.asarray(xy) - c) < r


def ok(spec, split):
    objs = spec["objects"]
    held_obj = [(o["shape"], o["color"]) in HELD_COMBOS for o in objs]
    in_reg = [in_region(o["xy"]) for o in objs]
    if split == "test_obj":
        return any(held_obj) and not any(in_reg)
    if split == "test_pos":
        return any(in_reg) and not any(held_obj)
    return not any(held_obj) and not any(in_reg)


def sample_scene(seed, idx, visual=True, distractors=True):
    rng = np.random.default_rng([seed, idx, 50])
    split = str(rng.choice(list(SPLIT_P), p=list(SPLIT_P.values())))
    cameras = v3s.HELD_CAMERAS if split == "test_cam" else v3s.TRAIN_CAMERAS
    for attempt in range(2000):
        spec = v3s.sample_scene(seed * 1000 + attempt, idx, cameras=cameras, visual=visual, distractors=distractors)
        if ok(spec, split):
            break
    else:
        raise RuntimeError(f"no scene for split {split} (seed {seed}, idx {idx})")
    spec.update(scene_id=f"v5s{seed}_{idx:05d}", seed=int(seed), scene_idx=int(idx), v3_attempt=attempt,
                split="test" if split == "test_lang" else ("val" if split == "val" else "train"), v5_split=split)
    return spec
