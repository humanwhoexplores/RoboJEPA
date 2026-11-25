"""Split a generated v6 batch into N symlink "views" so gen_v6.validate can check each one.

gen_v6.validate keeps every trajectory npz open while it runs (finals[tid] = z). A 200-scene batch
has 12,800 of them, and macOS caps a process at 10,240 open files (kern.maxfilesperproc), so the
validator dies with "Too many open files" on a whole batch. Its checks are all per scene, so
validating two 100-scene views covers the same ground. Nothing is copied: the views symlink the
batch's files and hold the batch's CSVs filtered to their scenes.

    python scripts/v6/split_view.py data/v6/part_00 2   # -> data/v6/_val_part_00_h0, _h1
"""
import os
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

CSVS = ("manifest.csv", "instruction_rows.csv", "all_goals.csv", "step2_manifest.csv", "step2_instruction_rows.csv")

src = Path(sys.argv[1]).resolve()
nparts = int(sys.argv[2])
sids = sorted(p.name[:-len(".rows.json")] for p in (src / "scenes").glob("*.rows.json"))
for k in range(nparts):
    keep = set(sids[k * len(sids) // nparts:(k + 1) * len(sids) // nparts])
    dst = src.parent / f"_val_{src.name}_h{k}"
    if dst.exists():
        shutil.rmtree(dst)
    for sub in ("trajs", "scenes", "steps", "videos"):
        (dst / sub).mkdir(parents=True)
        for f in os.listdir(src / sub):
            m = re.match(r"(v6s\d+_\d{5})", f)
            if m and m.group(1) in keep:
                os.symlink(src / sub / f, dst / sub / f)
    shutil.copy(src / "config.json", dst / "config.json")
    for c in CSVS:
        if (src / c).exists():
            df = pd.read_csv(src / c)
            df[df.scene_id.isin(keep)].to_csv(dst / c, index=False)
    print(dst, len(keep), "scenes", len(os.listdir(dst / "trajs")), "trajs")
