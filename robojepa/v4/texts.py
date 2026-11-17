"""Instruction strings for every goal in the v4 feature cache, with paraphrases for training.

Train goals get K texts: v4's stored instruction plus K-1 more drawn with v4's own language code
(gen_v3.language, training templates and shape words only, random referring expressions). Val and
test goals keep exactly the stored instruction. Row order matches <split>_meta.npz.

    python -m robojepa.v4.texts --features features_v4 --code data/ext_v4/code
"""
import argparse
import json
import sys

import numpy as np

K_TRAIN = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ext_v4/data/v4")
    ap.add_argument("--code", default="data/ext_v4/code")
    ap.add_argument("--features", default="features_v4")
    a = ap.parse_args()
    sys.path.insert(0, a.code)
    from gen_v3.language import instruction
    from robojepa.v4.oracle import split_names

    for split in split_names(a.features):
        m = np.load(f"{a.features}/{split}_meta.npz")
        k = K_TRAIN if split == "train" else 1
        out, scenes = [], {}
        for s, gid in zip(m["g_scene"], m["g_id"]):
            sid = str(m["scene_ids"][s])
            if sid not in scenes:
                scenes[sid] = json.load(open(f"{a.data}/scenes/{sid}.json"))
            sc = scenes[sid]
            texts = [sc["instructions"][gid]]
            goal = next(g for g in sc["goals"] if g["goal_id"] == gid)
            rng = np.random.default_rng([int(s), len(out), 31])
            for _ in range(20 * k):
                if len(texts) == k:
                    break
                t, _ = instruction(goal, sc["spec"], "train", rng, np.asarray(sc["init_pos"]))
                if t not in texts:
                    texts.append(t)
            out.append(texts + [texts[-1]] * (k - len(texts)))
        np.save(f"{a.features}/{split}_texts.npy", np.array(out))
        print(f"{split}: {len(out)} goals x {k} texts; e.g. {out[0]}")


if __name__ == "__main__":
    main()
