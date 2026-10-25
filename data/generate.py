"""Generate push trajectories: N scenes x 8 candidate pushes each.

    python -m data.generate --n-scenes 10 --out data/pilot          # quick pilot (80 trajectories)
    python -m data.generate --n-scenes 80 --out data/v1 --workers 6  # v1 (640 trajectories)

Each scene (a random cube pose and color layout) is replayed from the same start
for every candidate. Outputs:
    <out>/trajs/<traj_id>.npz  per-step logs + keyframe images + scene/params JSON
    <out>/manifest.csv         one row per trajectory (ids, split, params, labels)
    <out>/config.json          generation settings and library versions
"""
import argparse
import csv
import json
import os
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

CANDIDATE_TYPES = ["ideal_left", "ideal_right", "short_left", "short_right",
                   "over_left", "over_right", "miss", "random"]
SPLIT_P = {"train": 0.70, "val": 0.15, "test": 0.15}
CUBE_X = (-0.06, 0.06)
CUBE_Y = (-0.05, 0.05)
RESERVED_X = (0.03, 0.06)  # cube start band seen only in test scenes
NOISE = {"theta": 0.12, "dist_frac": 0.10, "offset": 0.01}
IMAGE_SIZE = 224

_ENV = None


def get_env():
    global _ENV
    if _ENV is None:
        from envs.push_env import make_env
        _ENV = make_env()
    return _ENV


def sample_scene(seed, scene_idx):
    rng = np.random.default_rng([seed, scene_idx])
    split = rng.choice(list(SPLIT_P), p=list(SPLIT_P.values()))
    if split == "test" and rng.random() < 0.5:
        x = rng.uniform(*RESERVED_X)
    else:
        x = rng.uniform(CUBE_X[0], RESERVED_X[0])
    y = rng.uniform(*CUBE_Y)
    return {
        "scene_id": f"s{seed}_{scene_idx:05d}",
        "scene_idx": int(scene_idx),
        "split": str(split),
        "cube_xy": [float(x), float(y)],
        "cube_yaw": float(rng.uniform(-np.pi / 4, np.pi / 4)),
        "left_color": str(rng.choice(["red", "blue"])),
        "reserved_band": bool(x >= RESERVED_X[0]),
    }


def make_candidates(scene, seed):
    from envs.push_env import CUBE_HALF, TARGETS

    rng = np.random.default_rng([seed, scene["scene_idx"], 1])
    c = np.array(scene["cube_xy"])
    cands = []
    for ctype in CANDIDATE_TYPES:
        if ctype == "random":
            theta, dist, offset = rng.uniform(-np.pi, np.pi), rng.uniform(0.05, 0.25), 0.0
        elif ctype == "miss":
            side = rng.choice(["left", "right"])
            v = TARGETS[side] - c
            theta, dist = np.arctan2(v[1], v[0]), np.linalg.norm(v)
            # Push line passes beside the cube: offset beyond cube half-width + finger half-width.
            offset = rng.choice([-1, 1]) * (CUBE_HALF + 0.04 + rng.uniform(0.0, 0.03))
        else:
            kind, side = ctype.split("_")
            v = TARGETS[side] - c
            theta, dist, offset = np.arctan2(v[1], v[0]), np.linalg.norm(v), 0.0
            dist *= {"ideal": 1.0, "short": rng.uniform(0.3, 0.6), "over": rng.uniform(1.6, 2.0)}[kind]
        intended = (float(theta), float(dist), float(offset))
        theta += rng.normal(0, NOISE["theta"])
        dist *= 1 + rng.normal(0, NOISE["dist_frac"])
        offset += rng.normal(0, NOISE["offset"])
        cands.append({"candidate_type": ctype, "intended": intended,
                      "theta": float(theta), "dist": float(max(dist, 0.0)), "offset": float(offset)})
    return cands


def run_trajectory(scene, cand, with_images=True):
    """Reset, place the scene, execute one push. Returns (log, keyframes, labels, colors)."""
    from envs.push_primitive import execute_push, label_outcome

    env = get_env()
    env.reset()
    colors = env.set_scene(scene["cube_xy"], scene["cube_yaw"], scene["left_color"])
    frames = {}
    kf = (lambda name: frames.__setitem__(name, env.camera_image(size=IMAGE_SIZE))) if with_images else None
    log = execute_push(env, cand["theta"], cand["dist"], cand["offset"], keyframe_fn=kf)
    labels = label_outcome(log["cube_pos"], log["contact"], log["phase"], np.array(scene["cube_xy"]))
    return log, frames, labels, colors


def run_scene(job):
    seed, scene_idx, out = job
    scene = sample_scene(seed, scene_idx)
    rows = []
    for k, cand in enumerate(make_candidates(scene, seed)):
        traj_id = f"{scene['scene_id']}_c{k}"
        t0 = time.time()
        log, frames, labels, colors = run_trajectory(scene, cand)
        labels["success_red"] = labels[f"success_{'left' if colors['left'] == 'red' else 'right'}"]
        labels["success_blue"] = labels[f"success_{'left' if colors['left'] == 'blue' else 'right'}"]
        meta = {"traj_id": traj_id, "scene": scene, "candidate": cand, "colors": colors, "labels": labels}
        names = list(frames)
        np.savez_compressed(
            Path(out) / "trajs" / f"{traj_id}.npz",
            meta=json.dumps(meta),
            keyframe_names=np.array(names),
            keyframes=np.stack([frames[n] for n in names]).astype(np.uint8),
            **{k: v for k, v in log.items()},
        )
        rows.append({
            "traj_id": traj_id, "scene_id": scene["scene_id"], "split": scene["split"],
            "reserved_band": scene["reserved_band"], "candidate_id": k, "candidate_type": cand["candidate_type"],
            "cube_x0": scene["cube_xy"][0], "cube_y0": scene["cube_xy"][1], "cube_yaw0": scene["cube_yaw"],
            "left_color": colors["left"], "right_color": colors["right"],
            "theta": cand["theta"], "dist": cand["dist"], "offset": cand["offset"],
            "n_steps": len(log["phase"]), "contact_steps": int(log["contact"].sum()),
            "seconds": round(time.time() - t0, 3), **labels,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-scenes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=590)
    ap.add_argument("--out", default="data/pilot")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "trajs").mkdir(parents=True, exist_ok=True)
    import mujoco
    import robosuite

    from envs import push_env, push_primitive

    config = {
        "seed": args.seed, "n_scenes": args.n_scenes, "candidate_types": CANDIDATE_TYPES,
        "split_p": SPLIT_P, "cube_x": CUBE_X, "cube_y": CUBE_Y, "reserved_x": RESERVED_X, "noise": NOISE,
        "image_size": IMAGE_SIZE, "camera": push_env.CAMERA, "targets": {k: v.tolist() for k, v in push_env.TARGETS.items()},
        "target_radius": push_env.TARGET_RADIUS, "cube_half": push_env.CUBE_HALF,
        "settle_steps": push_primitive.SETTLE_STEPS, "check_steps": push_primitive.CHECK_STEPS,
        "mujoco": mujoco.__version__, "robosuite": robosuite.__version__,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))

    jobs = [(args.seed, i, str(out)) for i in range(args.n_scenes)]
    t0 = time.time()
    rows = []
    if args.workers > 1:
        with get_context("spawn").Pool(args.workers) as pool:
            for i, r in enumerate(pool.imap_unordered(run_scene, jobs)):
                rows += r
                print(f"scene {i + 1}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    else:
        for i, job in enumerate(jobs):
            rows += run_scene(job)
            print(f"scene {i + 1}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)

    rows.sort(key=lambda r: r["traj_id"])
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} trajectories in {time.time() - t0:.0f}s -> {out}/manifest.csv")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
