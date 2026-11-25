"""Milestone 1 pilot: can the Panda gripper pick, place, drop and stack our objects reliably?

For each scene (v5's scene sampler, so the same objects and layouts), every object is picked with a scripted ideal
grasp and (a) placed on a free spot, (b) dropped from DROP_H above it, (c) stacked on another object. Each trial
restores the scene's start state. Reports success rates per shape and per trial type.

    MUJOCO_GL=osmesa python -m gen_v6.pilot_grasp --scenes 20 --workers 4 --out pilot_v6
"""
import argparse
import json
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from gen_v3 import scene_xml as sx

DROP_H = 0.12
PLACE_TOL = 0.015
TILT_MAX = np.radians(15)
_ENV = {}


def env():
    if "e" not in _ENV:
        from gen_v6.env import GripperSceneEnv
        _ENV["e"] = GripperSceneEnv()
    return _ENV["e"]


def yaw_of(q):
    return float(2 * np.arctan2(q[3], q[0]))


def tilt_of(q):
    w, x, y, z = q
    zz = 1 - 2 * (x * x + y * y)  # world z of the body z axis
    return float(np.arccos(np.clip(zz, -1, 1)))


def centre_height(o):
    h = sx.HEIGHT_HALF[o["size"]]
    return 0.8 * h if o["shape"] == "bar" else h


def ideal_grasp(o, pos, quat):
    """(x, y, yaw, z) of a grasp that squeezes two parallel faces of object o at pose (pos, quat)."""
    a = sx.SIZES[o["size"]]
    t = a / 3
    # The closing axis must pass through the centre of mass, or the object pivots between the pads.
    # L: across its upright bar, 1 cm above the centre of mass (-a/3, -a/3) so the pads clear the other bar;
    # the pads' torsional friction holds the small offset. T: end to end through the stem.
    # Prism: from its top vertex to the opposite face (two faces at 60 degrees squeeze it out upward).
    dx, dy, ang = {"cube": (0, 0, 0), "cylinder": (0, 0, 0), "prism": (0, a / 4, np.pi / 2), "hex": (0, 0, np.pi / 2),
                   "bar": (0, 0, np.pi / 2), "L": (-(a - t), -a / 3 + 0.01, 0), "T": (0, 0, np.pi / 2)}[o["shape"]]
    yaw = yaw_of(quat)
    c, s = np.cos(yaw), np.sin(yaw)
    return (pos[0] + c * dx - s * dy, pos[1] + s * dx + c * dy, yaw + ang, max(centre_height(o), 0.016))


def grasp_width(o):
    """Object width between the pads of ideal_grasp (m)."""
    a = sx.SIZES[o["size"]]
    return {"cube": 2 * a, "cylinder": 2 * a, "prism": 1.5 * a, "hex": np.sqrt(3) * a, "bar": a, "L": 2 * a / 3,
            "T": 2 * a}[o["shape"]]


def free_spot(rng, spec, pos, i, e, tries=300):
    r_i = spec["objects"][i]["radius"]
    for _ in range(tries):
        xy = rng.uniform(-0.2, 0.2, 2)
        if np.linalg.norm(xy - pos[i, :2]) < 0.08:
            continue
        dpos = e.distractor_poses()
        if all(np.linalg.norm(xy - pos[j, :2]) > r_i + o["radius"] + 0.03 for j, o in enumerate(spec["objects"]) if j != i) \
                and all(np.linalg.norm(xy - dpos[j, :2]) > r_i + d["radius"] + 0.03 for j, d in enumerate(spec.get("distractors", []))):
            return xy
    return None


def run_trial(e, spec, s0, i, kind, target_xy, base=None):
    from gen_v6.primitives import pick_place
    e.set_state(s0)
    pos, quat = e.object_poses()
    o = spec["objects"][i]
    g = ideal_grasp(o, pos[i], quat[i])
    hc = centre_height(o)
    if kind == "stack":
        b = spec["objects"][base]
        support = 2 * sx.HEIGHT_HALF[b["size"]] * (0.8 if b["shape"] == "bar" else 1.0)
        target_xy = pos[base, :2]
    else:
        support = 0.0
    z_rel = support + hc + (DROP_H if kind == "drop" else 0.004)
    off = np.array(g[:2]) - pos[i, :2]  # pads relative to the body centre; the yaw is kept, so is this offset
    t0 = time.time()
    log = pick_place(e, g, (target_xy[0] + off[0], target_xy[1] + off[1], g[2], z_rel),
                     open_width=min(0.08, grasp_width(o) + 0.025))
    dt = time.time() - t0
    P, Q, ph = log["obj_pos"], log["obj_quat"], log["phase"]
    carry = ph == "carry"
    lifted = bool(carry.any() and (P[carry, i, 2] > pos[i, 2] + 0.05).all())
    fin, fq = P[-1, i], Q[-1, i]
    others = [j for j in range(len(spec["objects"])) if j != i and j != base]
    moved_others = float(max([np.linalg.norm(P[-1, j, :2] - pos[j, :2]) for j in others] or [0.0]))
    xy_err = float(np.linalg.norm(fin[:2] - target_xy))
    tilt = tilt_of(fq)
    expect_z = support + sx.HEIGHT_HALF[o["size"]] * (0.8 if o["shape"] == "bar" else 1.0)
    on_level = abs(fin[2] - expect_z) < 0.012
    res = {"scene": spec["scene_id"], "obj": i, "shape": o["shape"], "size": o["size"], "kind": kind,
           "lifted": lifted, "xy_err": round(xy_err, 4), "tilt_deg": round(float(np.degrees(tilt)), 1),
           "z": round(float(fin[2]), 4), "on_level": bool(on_level), "moved_others": round(moved_others, 4),
           "steps": int(len(ph)), "seconds": round(dt, 2)}
    if kind == "stack":
        res["base_shape"] = spec["objects"][base]["shape"]
        res["base_moved"] = round(float(np.linalg.norm(P[-1, base, :2] - pos[base, :2])), 4)
        res["ok"] = bool(lifted and on_level and xy_err < 0.02 and tilt < TILT_MAX and res["base_moved"] < 0.01)
    elif kind == "place":
        res["ok"] = bool(lifted and on_level and xy_err < PLACE_TOL and tilt < TILT_MAX and moved_others < 0.01)
    else:  # drop: a lift and a landing on the table; where it ends up is the physics we want, not an error
        res["ok"] = bool(lifted and fin[2] > 0)
        res["landed_xy_err"] = round(xy_err, 4)
    return res


def run_scene(job):
    seed, idx = job
    from gen_v5.splits import sample_scene
    e = env()
    spec = sample_scene(seed, idx)
    e.set_scene(spec)
    s0 = e.get_state()
    pos = e.object_poses()[0]
    rng = np.random.default_rng([seed, idx, 61])
    out = []
    n = len(spec["objects"])
    for i in range(n):
        spot = free_spot(rng, spec, pos, i, e)
        if spot is None:
            continue
        out.append(run_trial(e, spec, s0, i, "place", spot))
        out.append(run_trial(e, spec, s0, i, "drop", spot))
        base = int(rng.choice([j for j in range(n) if j != i]))
        out.append(run_trial(e, spec, s0, i, "stack", None, base=base))
    return out


def summarize(rows):
    lines = []
    for kind in ("place", "drop", "stack"):
        rs = [r for r in rows if r["kind"] == kind]
        if not rs:
            continue
        lines.append(f"{kind}: ok {np.mean([r['ok'] for r in rs]):.3f}  lifted {np.mean([r['lifted'] for r in rs]):.3f}"
                     f"  n={len(rs)}  mean {np.mean([r['seconds'] for r in rs]):.1f}s wall, {np.mean([r['steps'] for r in rs]):.0f} steps")
        for shape in sx.SHAPES:
            ss = [r for r in rs if r["shape"] == shape]
            if ss:
                lines.append(f"   {shape:9s} ok {np.mean([r['ok'] for r in ss]):.2f}  lifted {np.mean([r['lifted'] for r in ss]):.2f}"
                             f"  xy_err {np.median([r['xy_err'] for r in ss]) * 100:.1f} cm  tilt {np.median([r['tilt_deg'] for r in ss]):.0f} deg  n={len(ss)}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="pilot_v6")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(args.seed, i) for i in range(args.scenes)]
    t0 = time.time()
    if args.workers > 1:
        with get_context("spawn").Pool(args.workers) as pool:
            rows = [r for part in pool.imap_unordered(run_scene, jobs) for r in part]
    else:
        rows = [r for j in jobs for r in run_scene(j)]
    (out / "grasp_trials.json").write_text(json.dumps(rows, indent=1))
    text = summarize(rows)
    (out / "grasp_summary.txt").write_text(text + f"\n{len(rows)} trials in {time.time() - t0:.0f}s\n")
    print(text)
    print(f"{len(rows)} trials in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
