"""Validate a generated dataset and write a report + contact sheet.

    python -m data.validate --data data/pilot
    python -m data.validate --data data/v1 --determinism 5

Fails (exit code 1) on label mismatches, split leakage, non-determinism,
NaNs, or a cube that left the table / moved impossibly fast.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from data.generate import RESERVED_X, run_trajectory
from envs.push_primitive import label_outcome

MAX_CUBE_SPEED = 2.0  # m/s; anything above this is a physics blow-up


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pilot")
    ap.add_argument("--determinism", type=int, default=3, help="re-simulate N random trajectories")
    ap.add_argument("--sheet", type=int, default=24, help="trajectories in the contact sheet")
    args = ap.parse_args()
    root = Path(args.data)
    m = pd.read_csv(root / "manifest.csv")
    errors, report = [], {"n_trajectories": len(m), "n_scenes": int(m.scene_id.nunique())}

    # 1. Labels recomputed from the stored cube trajectory must match.
    mismatches, bad_physics = [], []
    for row in m.itertuples():
        z = np.load(root / "trajs" / f"{row.traj_id}.npz")
        meta = json.loads(str(z["meta"]))
        lab = label_outcome(z["cube_pos"], z["contact"], z["phase"], np.array(meta["scene"]["cube_xy"]))
        for k in ("success_left", "success_right", "failure_mode_left", "failure_mode_right"):
            if lab[k] != getattr(row, k):
                mismatches.append((row.traj_id, k, lab[k], getattr(row, k)))
        speed = np.linalg.norm(z["cube_vel"], axis=1).max()
        if not np.isfinite(z["cube_pos"]).all() or speed > MAX_CUBE_SPEED:
            bad_physics.append((row.traj_id, float(speed)))
    report["label_mismatches"] = len(mismatches)
    report["bad_physics"] = bad_physics
    if mismatches:
        errors.append(f"{len(mismatches)} label mismatches, e.g. {mismatches[:3]}")
    if bad_physics:
        errors.append(f"physics problems: {bad_physics[:3]}")

    # 2. Split leakage: scenes in exactly one split; reserved band only in test.
    per_scene = m.groupby("scene_id").split.nunique()
    if (per_scene > 1).any():
        errors.append(f"scenes in several splits: {list(per_scene[per_scene > 1].index[:5])}")
    leak = m[(m.cube_x0 >= RESERVED_X[0]) & (m.split != "test")]
    if len(leak):
        errors.append(f"{len(leak)} non-test trajectories start in the reserved band")

    # 3. Contact sanity: misses should not touch, real pushes should.
    contact_rate = m.assign(touched=m.contact_steps > 0).groupby("candidate_type").touched.mean()
    report["contact_rate_by_type"] = contact_rate.round(3).to_dict()
    if contact_rate.get("miss", 0) > 0.2:
        errors.append(f"miss candidates touch the cube {contact_rate['miss']:.0%} of the time")

    # 4. Balance and outcome mix.
    report["split_trajectories"] = m.split.value_counts().to_dict()
    report["success_rate"] = {k: round(float(m[k].mean()), 3) for k in
                              ("success_left", "success_right", "success_red", "success_blue")}
    report["success_by_type"] = m.groupby("candidate_type")[["success_left", "success_right"]].mean().round(3).to_dict()
    modes = pd.concat([m.failure_mode_left, m.failure_mode_right]).value_counts()
    report["outcome_modes_per_instruction"] = modes.to_dict()
    report["left_color_red_fraction"] = round(float((m.left_color == "red").mean()), 3)

    # 5. Determinism: re-simulate a few trajectories and compare the final cube position.
    rng = np.random.default_rng(0)
    det = []
    for tid in rng.choice(m.traj_id, size=min(args.determinism, len(m)), replace=False):
        z = np.load(root / "trajs" / f"{tid}.npz")
        meta = json.loads(str(z["meta"]))
        log, _, _, _ = run_trajectory(meta["scene"], meta["candidate"], with_images=False)
        det.append(float(np.abs(log["cube_pos"][-1] - z["cube_pos"][-1]).max()))
    report["determinism_max_abs_diff"] = det
    if det and max(det) > 1e-6:
        errors.append(f"non-deterministic rerun, max diff {max(det):.2e} m")

    # 6. Contact sheet: initial | final with labels, for eyeballing.
    pick = m.sample(n=min(args.sheet, len(m)), random_state=0)
    tiles = []
    for row in pick.itertuples():
        z = np.load(root / "trajs" / f"{row.traj_id}.npz")
        names = list(z["keyframe_names"])
        pair = np.concatenate([z["keyframes"][names.index("initial")], z["keyframes"][names.index("final")]], 1)
        im = Image.fromarray(pair)
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, im.width, 26], fill=(0, 0, 0))
        d.text((4, 2), f"{row.candidate_type}  L={row.failure_mode_left}  R={row.failure_mode_right}", fill=(255, 255, 255))
        d.text((4, 13), f"left={row.left_color}  {row.split}  {row.traj_id}", fill=(200, 200, 200))
        tiles.append(np.asarray(im))
    cols = 3
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.concatenate([np.concatenate(tiles[i:i + cols], 1) for i in range(0, len(tiles), cols)], 0)
    Image.fromarray(grid).save(root / "contact_sheet.png")

    report["errors"] = errors
    (root / "validation_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    print("\nFAILED" if errors else "\nPASSED", f"-> {root}/validation_report.json, {root}/contact_sheet.png")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
