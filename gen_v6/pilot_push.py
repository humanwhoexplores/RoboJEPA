"""Milestone 1 pilot, part 2: how well does the gripper push compared with v5's V pusher?

Every object in each scene is pushed 10 cm in a random free direction, starting 3 cm behind it, by:
    v5_V          v5's rigid V pusher (gen_v5)
    grip_closed   the gripper, fingers closed, side by side across the push
    grip_spread   the gripper, fingers 6 cm apart across the push (two contact points)
Reports how often the object moves 7 to 13 cm along the push with less than 2 cm sideways drift.

    MUJOCO_GL=osmesa python -m gen_v6.pilot_push --scenes 12 --workers 4
"""
import argparse
import json
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from gen_v3 import scene_xml as sx

D = 0.10
VARIANTS = ("v5_V", "grip_closed", "grip_spread")
_ENV = {}


def env(kind):
    if kind not in _ENV:
        if kind == "v5":
            from gen_v5.env import PandaSceneEnv
            _ENV[kind] = PandaSceneEnv()
        else:
            from gen_v6.env import GripperSceneEnv
            _ENV[kind] = GripperSceneEnv()
    return _ENV[kind]


def clear_dir(rng, spec, pos, i):
    for _ in range(200):
        th = rng.uniform(-np.pi, np.pi)
        u = np.array([np.cos(th), np.sin(th)])
        end = pos[i, :2] + u * D
        if np.abs(end).max() > 0.24:
            continue
        ok = True
        for j in range(len(spec["objects"])):
            if j == i:
                continue
            # keep other objects off the swept corridor
            v = pos[j, :2] - pos[i, :2]
            along = v @ u
            across = abs(v[0] * u[1] - v[1] * u[0])
            if -0.08 < along < D + 0.06 and across < 0.07:
                ok = False
        if ok:
            return th
    return None


def run_scene(job):
    seed, idx = job
    from gen_v5.primitives import execute_push
    from gen_v5.splits import sample_scene
    from gen_v6.primitives import push
    spec = sample_scene(seed, idx)
    rng = np.random.default_rng([seed, idx, 62])
    e5, e6 = env("v5"), env("v6")
    e5.set_scene(spec)
    e6.set_scene(spec)
    s5, s6 = e5.get_state(), e6.get_state()
    pos = e6.object_poses()[0]
    rows = []
    for i, o in enumerate(spec["objects"]):
        th = clear_dir(rng, spec, pos, i)
        if th is None:
            continue
        u = np.array([np.cos(th), np.sin(th)])
        r = sx.footprint_radius(o["shape"], o["size"])
        for var in VARIANTS:
            if var == "v5_V":
                e5.set_state(s5)
                seat = sx.seat_x(r)
                st = pos[i, :2] - u * (seat + 0.03)
                t0 = time.time()
                log = execute_push(e5, st[0], st[1], th, D + 0.03)
            else:
                e6.set_state(s6)
                st = pos[i, :2] - u * (r + 0.012 + 0.03)  # finger half-thickness ~1.2 cm in front of the tool site
                t0 = time.time()
                log = push(e6, st[0], st[1], th, D + 0.03, opening=0.0 if var == "grip_closed" else 0.06)
            fin = log["obj_pos"][-1, i, :2]
            v = fin - pos[i, :2]
            along, lateral = float(v @ u), float(abs(v[0] * u[1] - v[1] * u[0]))
            others = [j for j in range(len(spec["objects"])) if j != i]
            moved_others = float(max([np.linalg.norm(log["obj_pos"][-1, j, :2] - pos[j, :2]) for j in others] or [0]))
            rows.append({"scene": spec["scene_id"], "obj": i, "shape": o["shape"], "size": o["size"], "variant": var,
                         "along": round(along, 4), "lateral": round(lateral, 4), "moved_others": round(moved_others, 4),
                         "ok": bool(0.07 <= along <= 0.13 and lateral < 0.02), "seconds": round(time.time() - t0, 2)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=12)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="pilot_v6")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(args.seed, i) for i in range(args.scenes)]
    with get_context("spawn").Pool(args.workers) as pool:
        rows = [r for part in pool.imap_unordered(run_scene, jobs) for r in part]
    (out / "push_trials.json").write_text(json.dumps(rows, indent=1))
    lines = []
    for var in VARIANTS:
        rs = [r for r in rows if r["variant"] == var]
        lines.append(f"{var:12s} ok {np.mean([r['ok'] for r in rs]):.2f}  along {np.median([r['along'] for r in rs]) * 100:.1f} cm"
                     f"  lateral {np.median([r['lateral'] for r in rs]) * 100:.1f} cm (p90 {np.percentile([r['lateral'] for r in rs], 90) * 100:.1f})  n={len(rs)}")
        for shape in sx.SHAPES:
            ss = [r for r in rs if r["shape"] == shape]
            if ss:
                lines.append(f"     {shape:9s} ok {np.mean([r['ok'] for r in ss]):.2f}  lateral {np.median([r['lateral'] for r in ss]) * 100:.1f} cm  n={len(ss)}")
    text = "\n".join(lines)
    (out / "push_summary.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
