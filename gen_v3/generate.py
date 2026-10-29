"""Generate v3 rollouts: N scenes x (32 single pushes + 8 two-push plans), all from each scene's saved start state.

    python -m gen_v3.generate --n-scenes 20 --out data/v3_mini --workers 6
    python -m gen_v3.generate --n-scenes 20 --out data/v3_mini --workers 6 --no-arm --no-visual   # closer to v2

Outputs (single pushes keep the v2 layout, so v2 tools read them):
    <out>/scenes/<scene_id>.json       spec, settled start positions, goals, focus / compound goals, instructions
    <out>/trajs/<traj_id>.npz          single push: per-step logs + keyframes (initial, after_push, final) + labels
    <out>/plans/<traj_id>.npz          two-push plan: logs with push_idx + keyframes
                                       (initial, after_push1, mid1, after_push2, final) + labels
    <out>/manifest.csv                 one row per single push (v2 columns + caption)
    <out>/instruction_rows.csv         one row per single push x focus goal (+ template)
    <out>/plan_manifest.csv            one row per plan (both pushes, compound-goal label, caption)
    <out>/plan_instruction_rows.csv    one row per plan x compound goal
    <out>/config.json                  settings and library versions
"""
import argparse
import csv
from multiprocessing import TimeoutError
import json
import os
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

IMAGE_SIZE = 224

_ENV = {}


def get_env(arm):
    if arm not in _ENV:
        from gen_v3.env import SceneEnv
        _ENV[arm] = SceneEnv(arm=arm)
    return _ENV[arm]


def scene_kwargs(opts):
    return {"cameras": tuple(opts["cameras"]), "held_cameras": tuple(opts["held_cameras"]),
            "held_share": opts["held_share"], "visual": opts["visual"], "distractors": opts["distractors"]}


def start_scene(seed, scene_idx, opts):
    """Sample, build and settle a scene. Returns (spec, settled positions, distractor positions, saved state)."""
    from gen_v3.scene_spec import sample_scene

    env = get_env(opts["arm"])
    spec = sample_scene(seed, scene_idx, **scene_kwargs(opts))
    env.set_scene(spec)
    return spec, env.object_poses()[0], env.distractor_poses(), env.get_state()


def run_candidate(spec, state0, cand, goals, opts, with_images=True):
    """Restore the start state and run one single push. Returns (log, frames, goal labels)."""
    from gen_v3.primitives import execute_push
    from gen_v3.tasks import label_goals

    env = get_env(opts["arm"])
    env.set_state(state0)
    frames = {}
    fn = (lambda name: frames.__setitem__(name, env.render(spec["camera"], IMAGE_SIZE))) if with_images else None
    log = execute_push(env, cand["x0"], cand["y0"], cand["theta"], cand["length"], frame_fn=fn)
    return log, frames, label_goals(goals, log["obj_pos"], log["phase"], spec)


def run_plan(spec, state0, plan, goals, opts, rng=None, with_images=True):
    """Restore the start state and run a two-push plan. Pushes left as None are planned from the settled
    middle state (needs rng). Returns (log, frames, goal labels, the pushes actually run)."""
    from gen_v3.candidates import second_push
    from gen_v3.primitives import execute_plan
    from gen_v3.tasks import label_goals

    env = get_env(opts["arm"])
    env.set_state(state0)
    frames = {}
    fn = (lambda name: frames.__setitem__(name, env.render(spec["camera"], IMAGE_SIZE))) if with_images else None
    run = []

    def plan_fn(k, env):
        c = second_push(plan, k, spec, env.object_poses()[0], env.distractor_poses(), rng)
        run.append(c)
        return c["x0"], c["y0"], c["theta"], c["length"]

    pushes = []
    for c in plan["pushes"]:
        pushes.append(None if c is None else (c["x0"], c["y0"], c["theta"], c["length"]))
    fixed = iter([c for c in plan["pushes"]])
    log, _ = execute_plan(env, pushes, frame_fn=fn, plan_fn=plan_fn)
    done, planned = [], iter(run)
    for c in fixed:
        done.append(c if c is not None else next(planned))
    return log, frames, label_goals(goals, log["obj_pos"], log["phase"], spec), done


def run_scene(job):
    from gen_v3.candidates import compound_goals, make_candidates, make_plans
    from gen_v3.language import caption, instruction
    from gen_v3.tasks import enumerate_goals, label_goals

    seed, scene_idx, out, opts = job
    out = Path(out)
    spec, init_pos, init_dis, state0 = start_scene(seed, scene_idx, opts)
    goals = enumerate_goals(spec, init_pos)
    rng = np.random.default_rng([seed, scene_idx, 1])
    focus, cands = make_candidates(spec, goals, init_pos, rng, init_dis)
    compound = compound_goals(focus) if opts["plans"] else []
    plan_rng = np.random.default_rng([seed, scene_idx, 4])
    plans = make_plans(spec, compound, init_pos, init_dis, plan_rng)[:opts["plans"]] if opts["plans"] else []
    goal_index = {g["goal_id"]: i for i, g in enumerate(goals)}
    lang_rng = np.random.default_rng([seed, scene_idx, 2])
    texts, templates = {}, {}
    for g in focus + compound:
        texts[g["goal_id"]], templates[g["goal_id"]] = instruction(g, spec, spec["split"], lang_rng, init_pos)
    cap_rng = np.random.default_rng([seed, scene_idx, 5])

    rows, instr_rows = [], []
    for k, cand in enumerate(cands):
        traj_id = f"{spec['scene_id']}_c{k:02d}"
        t0 = time.time()
        log, frames, labels = run_candidate(spec, state0, cand, goals, opts)
        names = list(frames)
        np.savez_compressed(
            out / "trajs" / f"{traj_id}.npz",
            meta=json.dumps({"traj_id": traj_id, "scene_id": spec["scene_id"], "candidate": cand}),
            keyframe_names=np.array(names),
            keyframes=np.stack([frames[n] for n in names]).astype(np.uint8),
            goal_labels=labels,
            **log,
        )
        own = cand["focus_goal"]
        rows.append({
            "traj_id": traj_id, "scene_id": spec["scene_id"], "split": spec["split"], "camera": spec["camera"],
            "n_objects": len(spec["objects"]), "n_zones": len(spec["zones"]),
            "n_distractors": len(spec["distractors"]), "candidate_id": k,
            "candidate_type": cand["candidate_type"], "focus_goal": own,
            "x0": cand["x0"], "y0": cand["y0"], "theta": cand["theta"], "length": cand["length"],
            "n_steps": len(log["phase"]), "touched": int(log["contact"].any()),
            "touched_intended": int(log["contact"][:, cand["intended"]["object"]].any()) if cand["intended"] else -1,
            "n_moved": int((np.linalg.norm(log["obj_pos"][-1, :, :2] - init_pos[:, :2], axis=1) > 0.01).sum()),
            "own_goal_success": int(labels[goal_index[own]]) if own else -1,
            "n_goals_true": int(labels.sum()), "seconds": round(time.time() - t0, 3),
            "caption": caption(spec, init_pos, log["obj_pos"][-1], cap_rng),
        })
        for g in focus:
            instr_rows.append({"traj_id": traj_id, "scene_id": spec["scene_id"], "split": spec["split"],
                               "goal_id": g["goal_id"], "goal_type": g["type"], "instruction": texts[g["goal_id"]],
                               "template": templates[g["goal_id"]], "label": int(labels[goal_index[g["goal_id"]]])})

    plan_rows, plan_instr = [], []
    all_goals = goals + compound
    for k, plan in enumerate(plans):
        traj_id = f"{spec['scene_id']}_p{k:02d}"
        t0 = time.time()
        log, frames, labels, done = run_plan(spec, state0, plan, all_goals, opts, rng=plan_rng)
        names = list(frames)
        np.savez_compressed(
            out / "plans" / f"{traj_id}.npz",
            meta=json.dumps({"traj_id": traj_id, "scene_id": spec["scene_id"], "plan_type": plan["plan_type"],
                             "focus_goal": plan["focus_goal"], "pushes": done}),
            keyframe_names=np.array(names),
            keyframes=np.stack([frames[n] for n in names]).astype(np.uint8),
            goal_labels=labels,
            **log,
        )
        mid = np.flatnonzero(log["push_idx"] == 0)[-1]
        own = plan["focus_goal"]
        row = {"traj_id": traj_id, "scene_id": spec["scene_id"], "split": spec["split"], "camera": spec["camera"],
               "n_objects": len(spec["objects"]), "plan_id": k, "plan_type": plan["plan_type"], "focus_goal": own}
        for j, c in enumerate(done):
            row.update({f"x0_{j + 1}": c["x0"], f"y0_{j + 1}": c["y0"], f"theta_{j + 1}": c["theta"],
                        f"length_{j + 1}": c["length"], f"type_{j + 1}": c["candidate_type"]})
        row.update({
            "n_steps": len(log["phase"]),
            "own_goal_success": int(labels[all_goals.index(next(g for g in compound if g["goal_id"] == own))]) if own else -1,
            "part1_success": int(labels[goal_index[own.split("(")[1].split("+")[0]]]) if own else -1,
            "part1_mid_success": int(label_goals([goals[goal_index[own.split("(")[1].split("+")[0]]]],
                                                 log["obj_pos"][:mid + 1], log["phase"][:mid + 1], spec)[0]) if own else -1,
            "n_goals_true": int(labels[:len(goals)].sum()), "seconds": round(time.time() - t0, 3),
            "caption_mid": caption(spec, init_pos, log["obj_pos"][mid], cap_rng),
            "caption": caption(spec, init_pos, log["obj_pos"][-1], cap_rng),
        })
        plan_rows.append(row)
        for g in compound:
            plan_instr.append({"traj_id": traj_id, "scene_id": spec["scene_id"], "split": spec["split"],
                               "goal_id": g["goal_id"], "goal_type": "and", "instruction": texts[g["goal_id"]],
                               "template": templates[g["goal_id"]], "label": int(labels[all_goals.index(g)])})

    (out / "scenes" / f"{spec['scene_id']}.json").write_text(json.dumps({
        "spec": spec, "init_pos": init_pos.tolist(), "init_distractor_pos": init_dis.tolist(), "goals": goals,
        "focus": [g["goal_id"] for g in focus], "compound": compound,
        "instructions": texts, "templates": templates}))
    result = (rows, instr_rows, plan_rows, plan_instr)
    (out / "scenes" / f"{spec['scene_id']}.rows.json").write_text(json.dumps(result))  # marks the scene done (--resume)
    return result


SCENE_TIMEOUT = 600  # s; a scene takes about 30 s on one core


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
    ap.add_argument("--n-scenes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/v3_mini")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--cameras", default="cam0,cam1,cam2,cam3,cam4", help="comma-separated training views")
    ap.add_argument("--held-cameras", default="held1,held2", help="views only used for test scenes ('' for none)")
    ap.add_argument("--held-share", type=float, default=0.5, help="share of test scenes shown from a held-out view")
    ap.add_argument("--plans", type=int, default=8, help="two-push plans per scene (0 for none)")
    ap.add_argument("--no-arm", dest="arm", action="store_false", help="floating pusher only, as in v2")
    ap.add_argument("--no-visual", dest="visual", action="store_false", help="fixed table color and lighting")
    ap.add_argument("--no-distractors", dest="distractors", action="store_false")
    ap.add_argument("--resume", action="store_true", help="keep scenes already finished in --out")
    args = ap.parse_args()

    out = Path(args.out)
    for sub in ("trajs", "scenes", "plans"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    import mujoco

    from gen_v3 import candidates, language, primitives, scene_spec, scene_xml, tasks

    opts = {"cameras": args.cameras.split(","), "held_cameras": [c for c in args.held_cameras.split(",") if c],
            "held_share": args.held_share, "plans": args.plans, "arm": args.arm, "visual": args.visual,
            "distractors": args.distractors}
    config = {
        "generator": "gen_v3", "seed": args.seed, "n_scenes": args.n_scenes, "image_size": IMAGE_SIZE, **opts,
        "split_p": scene_spec.SPLIT_P, "n_objects": scene_spec.N_OBJECTS, "n_zones": scene_spec.N_ZONES,
        "n_distractors": scene_spec.N_DISTRACTORS, "shapes": scene_xml.SHAPES, "sizes": scene_xml.SIZES,
        "colors": list(scene_xml.COLORS), "zone_radius": scene_xml.ZONE_RADIUS, "noise": candidates.NOISE,
        "focus_mix": candidates.FOCUS_MIX, "goal_types": tasks.GOAL_TYPES, "settle_steps": primitives.SETTLE_STEPS,
        "check_steps": primitives.CHECK_STEPS, "relational_p": language.RELATIONAL_P, "mujoco": mujoco.__version__,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))

    jobs = [(args.seed, i, str(out), opts) for i in range(args.n_scenes)]
    t0 = time.time()
    rows, instr_rows, plan_rows, plan_instr = [], [], [], []

    def collect(res, i):
        for acc, part in zip((rows, instr_rows, plan_rows, plan_instr), res):
            acc += part
        print(f"scene {i + 1}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)


    done = 0
    if args.resume:
        todo = []
        for job in jobs:
            f = out / "scenes" / f"v3s{args.seed}_{job[1]:05d}.rows.json"
            if f.exists():
                collect(json.loads(f.read_text()), done)
                done += 1
            else:
                todo.append(job)
        jobs_left = todo
    else:
        jobs_left = jobs
    if args.workers > 1:
        # A worker that dies (e.g. a crash inside the renderer) makes Pool wait forever. So: wait at most
        # SCENE_TIMEOUT for the next scene; if none arrives, restart the pool on the scenes not yet done.
        while jobs_left:
            pending = {job[1]: job for job in jobs_left}
            pool = get_context("spawn").Pool(args.workers)
            it = pool.imap_unordered(run_scene_tagged, jobs_left)
            try:
                for _ in range(len(jobs_left)):
                    idx, res = it.next(timeout=SCENE_TIMEOUT)
                    collect(res, done)
                    done += 1
                    pending.pop(idx)
                pool.close()
            except TimeoutError:
                print(f"no scene finished in {SCENE_TIMEOUT}s; restarting the pool for {len(pending)} scenes", flush=True)
            finally:
                pool.terminate()
                pool.join()
            jobs_left = list(pending.values())
    else:
        for job in jobs_left:
            collect(run_scene(job), done)
            done += 1

    rows.sort(key=lambda r: r["traj_id"])
    instr_rows.sort(key=lambda r: (r["traj_id"], r["goal_id"]))
    plan_rows.sort(key=lambda r: r["traj_id"])
    plan_instr.sort(key=lambda r: (r["traj_id"], r["goal_id"]))
    write_csv(out / "manifest.csv", rows)
    write_csv(out / "instruction_rows.csv", instr_rows)
    write_csv(out / "plan_manifest.csv", plan_rows)
    write_csv(out / "plan_instruction_rows.csv", plan_instr)
    print(f"{len(rows)} single pushes + {len(plan_rows)} plans in {time.time() - t0:.0f}s -> {out}/manifest.csv")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
