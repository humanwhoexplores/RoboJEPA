"""Generate v4: the same scenes as v3 (same seed), goal-blind pushes, every push labeled against every goal.

    python -m gen_v4.generate --n-scenes 40 --out data/v4_mini --workers 6
    python -m gen_v4.generate --n-scenes 1000 --out data/v4 --workers 6        # add --resume after a stop

Per scene (defaults):
    64 goal-blind pushes from the settled start state (gen_v4.sampler), in random order
    labels for every push x every goal of the scene, three ways (gen_v4.labels)
    8 single-push decisions picked from the scene only (gen_v4.decisions): v3's 4 + 4 open ones
    2 two-push decisions (v3's compound goals): if some start push solves the first part, one of them
        is picked at random, and 32 new goal-blind pushes are run from its settled end state

Outputs:
    <out>/scenes/<scene_id>.json          spec, start positions, every goal (+ in_view, straight_path,
                                          instruction, template), eval goals, compound goals, step-2 info
    <out>/scenes/<scene_id>_initial.npz   the start image, stored once per scene (all pushes share it)
    <out>/scenes/<scene_id>_labels.npz    pushes x goals label matrices: default, state, strict
    <out>/trajs/<scene_id>_cNN.npz        one push: per-step logs, keyframes [after_push, final], labels
    <out>/steps/<scene_id>_kJ_sNN.npz     step 2 of two-push decision J: same layout; its start image is
                                          the final frame of the step-1 push named in meta
    <out>/manifest.csv                    one row per push (no goal fields: the pushes know no goal)
    <out>/instruction_rows.csv            one row per push x eval decision (v3 columns + label variants)
    <out>/all_goals.csv                   one row per scene goal, with its instruction and positive counts
    <out>/step2_manifest.csv              one row per step-2 push
    <out>/step2_instruction_rows.csv      one row per step-2 push x its compound goal
    <out>/config.json                     settings and library versions
"""
import argparse
import csv
import json
import os
import time
from multiprocessing import TimeoutError, get_context
from pathlib import Path

import numpy as np

IMAGE_SIZE = 224
STEP_FRAMES = (None, "after_push", "final")  # the start frame is stored once per scene instead


def scene_ids(spec, seed, idx):
    spec["v3_scene_id"] = spec["scene_id"]
    spec["scene_id"] = f"v4s{seed}_{idx:05d}"
    return spec


def run_push(env, spec, state, cand, goals, start_pos, origin_pos=None, with_images=True):
    """Restore state, run one push, label it. Returns (log, frames, labels by name, settled end state)."""
    from gen_v3.primitives import execute_push
    from gen_v4.labels import label_sets

    env.set_state(state)
    frames = {}
    fn = (lambda name: frames.__setitem__(name, env.render(spec["camera"], IMAGE_SIZE))) if with_images else None
    log = execute_push(env, cand["x0"], cand["y0"], cand["theta"], cand["length"], frame_fn=fn,
                       frame_names=STEP_FRAMES)
    labels = label_sets(goals, log["obj_pos"], log["phase"], spec, start_pos, origin_pos)
    return log, frames, labels, env.get_state()


def save_rollout(path, meta, frames, labels, log):
    names = list(frames)
    np.savez_compressed(
        path, meta=json.dumps(meta), keyframe_names=np.array(names),
        keyframes=np.stack([frames[n] for n in names]).astype(np.uint8),
        goal_labels=labels["default"], goal_labels_state=labels["state"], goal_labels_strict=labels["strict"],
        **log)


def push_row(cand, log, start_pos, labels):
    moved = np.linalg.norm(log["obj_pos"][-1, :, :2] - start_pos[:, :2], axis=1)
    s = cand["sampler"]
    return {"candidate_type": cand["candidate_type"], "sampler_object": s["object"] if s else -1,
            "x0": cand["x0"], "y0": cand["y0"], "theta": cand["theta"], "length": cand["length"],
            "n_steps": len(log["phase"]), "touched": int(log["contact"].any()),
            "n_touched": int(log["contact"].any(0).sum()), "n_moved": int((moved > 0.01).sum()),
            "n_goals_true": int(labels["default"].sum()), "n_goals_true_state": int(labels["state"].sum())}


def run_scene(job):
    from gen_v3.generate import get_env, start_scene
    from gen_v3.language import caption
    from gen_v3.tasks import enumerate_goals
    from gen_v4.decisions import pick
    from gen_v4.sampler import sample_pushes

    seed, idx, out, opts = job
    out = Path(out)
    spec, init_pos, init_dis, state0 = start_scene(seed, idx, opts)
    spec = scene_ids(spec, seed, idx)
    sid = spec["scene_id"]
    env = get_env(opts["arm"])
    goals = enumerate_goals(spec, init_pos)
    gid = [g["goal_id"] for g in goals]
    gindex = {g: i for i, g in enumerate(gid)}
    evals, compound, texts, templates = pick(goals, spec, init_pos, init_dis, seed, idx)
    cands = sample_pushes(spec, init_pos, init_dis, np.random.default_rng([seed, idx, 21]),
                          n_total=opts["n_pushes"], n_lines=opts["n_lines"])
    env.set_state(state0)
    np.savez_compressed(out / "scenes" / f"{sid}_initial.npz", initial=env.render(spec["camera"], IMAGE_SIZE),
                        camera=spec["camera"])
    cap_rng = np.random.default_rng([seed, idx, 5])
    base = {"scene_id": sid, "v3_scene_id": spec["v3_scene_id"], "split": spec["split"], "camera": spec["camera"]}

    rows, instr_rows, states = [], [], []
    mats = {k: [] for k in ("default", "state", "strict")}
    for k, cand in enumerate(cands):
        traj_id = f"{sid}_c{k:02d}"
        t0 = time.time()
        log, frames, labels, end_state = run_push(env, spec, state0, cand, goals, init_pos)
        states.append(end_state)
        for name in mats:
            mats[name].append(labels[name])
        save_rollout(out / "trajs" / f"{traj_id}.npz", {"traj_id": traj_id, "scene_id": sid, "candidate": cand},
                     frames, labels, log)
        rows.append({"traj_id": traj_id, **base, "n_objects": len(spec["objects"]), "n_zones": len(spec["zones"]),
                     "n_distractors": len(spec["distractors"]), "candidate_id": k,
                     "in_eval32": int(k < opts["n_pushes"] // 2), **push_row(cand, log, init_pos, labels),
                     "seconds": round(time.time() - t0, 3),
                     "caption": caption(spec, init_pos, log["obj_pos"][-1], cap_rng)})
        for g in evals:
            i = gindex[g["goal_id"]]
            instr_rows.append({"traj_id": traj_id, **base, "goal_id": g["goal_id"], "goal_type": g["type"],
                               "source": g["source"], "straight_path": int(g["straight_path"]),
                               "instruction": texts[g["goal_id"]], "template": templates[g["goal_id"]],
                               "in_eval32": int(k < opts["n_pushes"] // 2), "label": int(labels["default"][i]),
                               "label_state": int(labels["state"][i]), "label_strict": int(labels["strict"][i])})
    mats = {k: np.stack(v) for k, v in mats.items()}
    np.savez_compressed(out / "scenes" / f"{sid}_labels.npz", goal_ids=np.array(gid),
                        traj_ids=np.array([r["traj_id"] for r in rows]), **mats)

    # Two-push decisions: step 1 = the start pushes; step 2 = new goal-blind pushes from a solved middle state.
    step2_rows, step2_instr, step2_info = [], [], []
    for j, g in enumerate(compound if opts["step2"] else []):
        p1, p2 = g["parts"]
        solved = np.flatnonzero(mats["default"][:, gindex[p1["goal_id"]]])
        info = {"goal_id": g["goal_id"], "step1_solvable": bool(len(solved)), "step1_solvers": len(solved)}
        step2_info.append(info)
        if not len(solved):
            continue
        k1 = int(np.random.default_rng([seed, idx, 23, j]).choice(solved))
        info["step1_traj_id"] = rows[k1]["traj_id"]
        mid = states[k1]
        env.set_state(mid)
        mid_pos, mid_dis = env.object_poses()[0], env.distractor_poses()
        cands2 = sample_pushes(spec, mid_pos, mid_dis, np.random.default_rng([seed, idx, 24, j]),
                               n_total=opts["n_step2"], n_lines=opts["n_step2_lines"])
        goals2 = goals + [g]
        for k, cand in enumerate(cands2):
            traj_id = f"{sid}_k{j}_s{k:02d}"
            t0 = time.time()
            log, frames, labels, _ = run_push(env, spec, mid, cand, goals2, mid_pos, origin_pos=init_pos)
            save_rollout(out / "steps" / f"{traj_id}.npz",
                         {"traj_id": traj_id, "scene_id": sid, "compound_goal": g["goal_id"],
                          "step1_traj_id": info["step1_traj_id"], "candidate": cand}, frames, labels, log)
            step2_rows.append({"traj_id": traj_id, **base, "compound_goal": g["goal_id"],
                               "step1_traj_id": info["step1_traj_id"], "candidate_id": k,
                               **push_row(cand, log, mid_pos, {n: v[:-1] for n, v in labels.items()}),
                               "seconds": round(time.time() - t0, 3),
                               "caption": caption(spec, mid_pos, log["obj_pos"][-1], cap_rng)})
            step2_instr.append({"traj_id": traj_id, **base, "goal_id": g["goal_id"], "goal_type": "and",
                                "instruction": texts[g["goal_id"]], "template": templates[g["goal_id"]],
                                "step1_traj_id": info["step1_traj_id"],
                                "part2_label": int(labels["default"][gindex[p2["goal_id"]]]),
                                "label": int(labels["default"][-1]), "label_state": int(labels["state"][-1]),
                                "label_strict": int(labels["strict"][-1])})

    pos_counts = mats["default"].sum(0)
    goal_rows = [{**base, "goal_id": g["goal_id"], "goal_type": g["type"], "a": g["a"], "b": g.get("b", -1),
                  "c": g.get("c", -1), "in_view": int(g["in_view"]), "straight_path": int(g["straight_path"]),
                  "instruction": texts[g["goal_id"]], "template": templates[g["goal_id"]],
                  "n_pos": int(pos_counts[i]), "n_pos_state": int(mats["state"][:, i].sum()),
                  "n_pos_strict": int(mats["strict"][:, i].sum())} for i, g in enumerate(goals)]
    (out / "scenes" / f"{sid}.json").write_text(json.dumps({
        "spec": spec, "init_pos": init_pos.tolist(), "init_distractor_pos": init_dis.tolist(), "goals": goals,
        "eval_goals": [{"goal_id": g["goal_id"], "source": g["source"]} for g in evals], "compound": compound,
        "instructions": texts, "templates": templates, "step2": step2_info}))
    result = (rows, instr_rows, goal_rows, step2_rows, step2_instr)
    (out / "scenes" / f"{sid}.rows.json").write_text(json.dumps(result))  # marks the scene done (--resume)
    return result


SCENE_TIMEOUT = 900  # s; a scene takes about 1 min on one core


def run_scene_tagged(job):
    return job[1], run_scene(job)


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-scenes", type=int, default=40)
    ap.add_argument("--first", type=int, default=0, help="index of the first scene (default 0, as v3)")
    ap.add_argument("--seed", type=int, default=7, help="7 rebuilds v3's scenes")
    ap.add_argument("--out", default="data/v4_mini")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--pushes", type=int, default=64, help="goal-blind pushes per scene")
    ap.add_argument("--lines", type=int, default=8, help="of which random pusher lines")
    ap.add_argument("--step2", type=int, default=32, help="pushes at step 2 of each two-push decision (0: none)")
    ap.add_argument("--step2-lines", type=int, default=4)
    ap.add_argument("--cameras", default="cam0,cam1,cam2,cam3,cam4")
    ap.add_argument("--held-cameras", default="held1,held2")
    ap.add_argument("--held-share", type=float, default=0.5)
    ap.add_argument("--no-arm", dest="arm", action="store_false")
    ap.add_argument("--no-visual", dest="visual", action="store_false")
    ap.add_argument("--no-distractors", dest="distractors", action="store_false")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    for sub in ("trajs", "scenes", "steps"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    import mujoco

    from gen_v3 import language, primitives, scene_spec, scene_xml, tasks
    from gen_v4 import decisions, labels, sampler

    opts = {"cameras": args.cameras.split(","), "held_cameras": [c for c in args.held_cameras.split(",") if c],
            "held_share": args.held_share, "plans": 0, "arm": args.arm, "visual": args.visual,
            "distractors": args.distractors, "n_pushes": args.pushes, "n_lines": args.lines,
            "step2": args.step2 > 0, "n_step2": args.step2, "n_step2_lines": args.step2_lines}
    config = {
        "generator": "gen_v4", "seed": args.seed, "first": args.first, "n_scenes": args.n_scenes,
        "image_size": IMAGE_SIZE, **opts, "split_p": scene_spec.SPLIT_P, "n_objects": scene_spec.N_OBJECTS,
        "n_zones": scene_spec.N_ZONES, "n_distractors": scene_spec.N_DISTRACTORS, "shapes": scene_xml.SHAPES,
        "sizes": scene_xml.SIZES, "colors": list(scene_xml.COLORS), "zone_radius": scene_xml.ZONE_RADIUS,
        "sampler": {"dist_min": sampler.DIST_MIN, "dist_max": sampler.DIST_MAX, "offset_frac": sampler.OFFSET_FRAC},
        "labels": {"move_min": labels.MOVE_MIN, "ref_max": labels.REF_MAX, "default": "default"},
        "n_open_decisions": decisions.N_OPEN, "goal_types": tasks.GOAL_TYPES,
        "settle_steps": primitives.SETTLE_STEPS, "check_steps": primitives.CHECK_STEPS,
        "relational_p": language.RELATIONAL_P, "mujoco": mujoco.__version__,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))

    jobs = [(args.seed, i, str(out), opts) for i in range(args.first, args.first + args.n_scenes)]
    t0 = time.time()
    acc = ([], [], [], [], [])
    done = 0

    def collect(res):
        nonlocal done
        for a, part in zip(acc, res):
            a += part
        done += 1
        print(f"scene {done}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)

    jobs_left = []
    for job in jobs:
        f = out / "scenes" / f"v4s{args.seed}_{job[1]:05d}.rows.json"
        if args.resume and f.exists():
            collect(json.loads(f.read_text()))
        else:
            jobs_left.append(job)
    if args.workers > 1:
        # A worker that dies makes Pool wait forever, so wait at most SCENE_TIMEOUT for the next scene and
        # restart the pool on the scenes not yet done.
        while jobs_left:
            pending = {job[1]: job for job in jobs_left}
            pool = get_context("spawn").Pool(args.workers)
            it = pool.imap_unordered(run_scene_tagged, jobs_left)
            try:
                for _ in range(len(jobs_left)):
                    i, res = it.next(timeout=SCENE_TIMEOUT)
                    collect(res)
                    pending.pop(i)
                pool.close()
            except TimeoutError:
                print(f"no scene finished in {SCENE_TIMEOUT}s; restarting the pool for {len(pending)} scenes", flush=True)
            finally:
                pool.terminate()
                pool.join()
            jobs_left = list(pending.values())
    else:
        for job in jobs_left:
            collect(run_scene(job))

    rows, instr_rows, goal_rows, step2_rows, step2_instr = acc
    for lst, key in ((rows, ("traj_id",)), (instr_rows, ("traj_id", "goal_id")), (goal_rows, ("scene_id", "goal_id")),
                     (step2_rows, ("traj_id",)), (step2_instr, ("traj_id",))):
        lst.sort(key=lambda r: tuple(r[k] for k in key))
    write_csv(out / "manifest.csv", rows)
    write_csv(out / "instruction_rows.csv", instr_rows)
    write_csv(out / "all_goals.csv", goal_rows)
    write_csv(out / "step2_manifest.csv", step2_rows)
    write_csv(out / "step2_instruction_rows.csv", step2_instr)
    print(f"{len(rows)} pushes + {len(step2_rows)} step-2 pushes in {time.time() - t0:.0f}s -> {out}/manifest.csv")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
