"""Turn trajectories into instruction rows (one push outcome judged under 4 instructions).

    python -m data.instructions --data data/v1

Writes <data>/instruction_rows.csv with columns:
    traj_id, scene_id, split, kind (position|color), target_side, target_color, instruction, label
Train/val use TRAIN templates; test uses the held-out TEST templates.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN = {
    "position": ["push the cube to the {side} region", "move the block into the {side} zone",
                 "slide the cube over to the {side} target", "put the cube in the {side} area"],
    "color": ["push the cube to the {color} region", "move the block into the {color} zone",
              "slide the cube over to the {color} target", "put the cube in the {color} area"],
}
TEST = {
    "position": ["get the yellow block onto the {side}-hand circle", "nudge the cube until it sits in the {side} spot"],
    "color": ["get the yellow block onto the {color} circle", "nudge the cube until it sits in the {color} spot"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pilot")
    ap.add_argument("--seed", type=int, default=590)
    args = ap.parse_args()
    root = Path(args.data)
    m = pd.read_csv(root / "manifest.csv")
    rng = np.random.default_rng(args.seed)
    rows = []
    for r in m.itertuples():
        bank = TEST if r.split == "test" else TRAIN
        for side in ("left", "right"):
            color = r.left_color if side == "left" else r.right_color
            for kind in ("position", "color"):
                text = rng.choice(bank[kind]).format(side=side, color=color)
                rows.append({"traj_id": r.traj_id, "scene_id": r.scene_id, "split": r.split, "kind": kind,
                             "target_side": side, "target_color": color, "instruction": text,
                             "label": int(getattr(r, f"success_{side}"))})
    out = pd.DataFrame(rows)
    out.to_csv(root / "instruction_rows.csv", index=False)
    print(out.groupby(["split", "kind"]).label.agg(["count", "mean"]).round(3))


if __name__ == "__main__":
    main()
