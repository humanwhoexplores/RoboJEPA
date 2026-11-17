"""Cache frozen DINOv2 + goal tables for the v5 dataset (gen_v5, branch generator-v5: Franka Panda).

Same layout as robojepa.v4.features (start frame + 64 settled finals per scene, int8 + per-token scale), so
the v4 oracle / pilot code runs on it unchanged. Differences:
  * splits come from v5 itself (spec["v5_split"]): train, val, test_id and four one-factor OOD tests
    (test_lang, test_cam, test_pos, test_obj); the list is written to <out>/splits.json.
  * v5 renders 256 px; frames are resized to 224 (DINOv2-small, 16x16 patches) on the GPU.
  * meta gains `bad` (n_scenes, 64): pushes whose end effector left the commanded line by > 15 mm.

    python -m robojepa.v5.features --data data/v5 --code ../RoboJEPA-v5 --out features_v5
    python -m robojepa.v4.texts --data data/v5 --code ../RoboJEPA-v5 --features features_v5
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from robojepa.features import VisionEncoder
from robojepa.v4 import features as v4f

SPLITS = ("train", "val", "test_id", "test_lang", "test_cam", "test_pos", "test_obj")
BAD_TRACK_MM = 15.0


class Resized(VisionEncoder):
    @torch.no_grad()
    def __call__(self, imgs_uint8):
        x = torch.from_numpy(imgs_uint8).to(self.device).permute(0, 3, 1, 2).float() / 255
        x = F.interpolate(x, size=(224, 224), mode="bilinear", antialias=True, align_corners=False)
        x = ((x - self.mean) / self.std).half()
        return self.model(pixel_values=x).last_hidden_state[:, 1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/v5")
    ap.add_argument("--code", default="../RoboJEPA-v5", help="folder containing gen_v3/ (reference rule)")
    ap.add_argument("--out", default="features_v5")
    ap.add_argument("--limit", type=int, default=None, help="first N scenes per split (smoke test)")
    a = ap.parse_args()
    sys.path.insert(0, a.code)
    os.makedirs(a.out, exist_ok=True)
    sids = sorted(f[:-5] for f in os.listdir(f"{a.data}/scenes") if f.endswith(".json") and "rows" not in f)
    by_split = {s: [] for s in SPLITS}
    for s in sids:
        by_split[json.load(open(f"{a.data}/scenes/{s}.json"))["spec"]["v5_split"]].append(s)
    err = {r["traj_id"]: float(r["push_track_err_mm"]) for r in csv.DictReader(open(f"{a.data}/manifest.csv"))}
    json.dump(list(SPLITS), open(f"{a.out}/splits.json", "w"))
    enc, texts = Resized("cuda"), {}
    for split in SPLITS:
        ss = by_split[split][: a.limit]
        v4f.encode_split(enc, a.data, ss, a.out, split, texts)
        bad = np.array([[err[f"{s}_c{k:02d}"] > BAD_TRACK_MM for k in range(v4f.N_CAND)] for s in ss])
        m = dict(np.load(f"{a.out}/{split}_meta.npz"))
        np.savez(f"{a.out}/{split}_meta.npz", **m, bad=bad)
        print(f"{split}: {bad.sum()} pushes flagged bad (> {BAD_TRACK_MM} mm)", flush=True)
    v4f.encode_texts(texts, a.out, "cuda")


if __name__ == "__main__":
    main()
