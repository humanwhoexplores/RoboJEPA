"""Generate v5: v4's design (goal-blind pushes, every push labeled against every goal, decisions picked
from the scene only) on a simulated Franka Panda, with per-step video and end-effector logs and
one-factor OOD test splits.

    python -m gen_v5.generate --n-scenes 8 --out data/v5_mini --workers 4
    python -m gen_v5.generate --n-scenes 1000 --out data/v5 --workers 6 --resume

What changes from v4 (gen_v4.generate):
    robot      a Panda moves v3's V pusher (gen_v5.env / primitives); pushes the arm cannot follow are
               shortened or replaced (gen_v5.sampler); arm contacts with objects count as touches
    splits     train / val / test_id + test_lang / test_cam / test_pos / test_obj (gen_v5.splits)
    images     256 px keyframes (start, after_push, final), as v4 but larger
    video      every push also as an mp4 of the scene camera at VIDEO_FPS (videos/<traj_id>.mp4)
    logs       v4's per-step logs plus ee_state (x, y, z, roll, pitch, yaw, gripper), joint positions q,
               tool targets, and video_step (the control steps the video frames were taken at)
Same file layout as v4 otherwise, so robojepa.v4.features reads it (with --image-size 256).
"""
import argparse
import csv
import json
import os
import time
from multiprocessing import TimeoutError, get_context
from pathlib import Path

import numpy as np

IMAGE_SIZE = 256
CONTROL_HZ = 20
VIDEO_FPS = 5  # every 4th control step; RoboJEPA's DROID loader samples 1.875 fps from 15
STEP_FRAMES = (None, "after_push", "final")
_ENV = {}


def get_env():
    if "env" not in _ENV:
        from gen_v5.env import PandaSceneEnv
        _ENV["env"] = PandaSceneEnv()
    return _ENV["env"]


def run_push(env, spec, state, cand, goals, start_pos, video_path, origin_pos=None):
    """Restore state, run one push with video, label it. Returns (log, keyframes, labels, end state)."""
    from gen_v4.labels import label_sets
    from gen_v5.primitives import execute_push
    from gen_v5.video import write_mp4

    env.set_state(state)
    frames, video, steps = {}, [], []
    every = CONTROL_HZ // VIDEO_FPS
    count = [0]

    def video_fn():
        if count[0] % every == 0:
            video.append(env.render(spec["camera"], IMAGE_SIZE))
            steps.append(count[0])
        count[0] += 1

    log = execute_push(env, cand["x0"], cand["y0"], cand["theta"], cand["length"],
                       frame_fn=lambda name: frames.__setitem__(name, env.render(spec["camera"], IMAGE_SIZE)),
                       frame_names=STEP_FRAMES, video_fn=video_fn)
    log["video_step"] = np.array(steps) - 1  # frame k shows the state after control step video_step[k] (-1: start)
    write_mp4(video_path, video, VIDEO_FPS)
    labels = label_sets(goals, log["obj_pos"], log["phase"], spec, start_pos, origin_pos)
    return log, frames, labels, env.get_state()


def save_rollout(path, meta, frames, labels, log):
    names = list(frames)
    # float32 halves the per-step log (sub-micron precision at table scale); ints/bools/strings unchanged
    log = {k: v.astype(np.float32) if np.asarray(v).dtype == np.float64 else v for k, v in log.items()}
    np.savez_compressed(
        path, meta=json.dumps(meta), keyframe_names=np.array(names),
        keyframes=np.stack([frames[n] for n in names]).astype(np.uint8),
        goal_labels=labels["default"], goal_labels_state=labels["state"], goal_labels_strict=labels["strict"],
        **log)


def push_row(cand, log, start_pos, labels):
    moved = np.linalg.norm(log["obj_pos"][-1, :, :2] - start_pos[:, :2], axis=1)
    s = cand["sampler"]
    push = log["phase"] == "push"
    tg, ee = log["target"][push], log["ee_state"][push]
    return {"candidate_type": cand["candidate_type"], "sampler_object": s["object"] if s else -1,
            "x0": cand["x0"], "y0": cand["y0"], "theta": cand["theta"], "length": cand["length"],
            "v4_length": cand["v4_length"], "shortened": int(cand["shortened"]), "draw": cand["draw"],
            "n_steps": len(log["phase"]), "n_video_frames": len(log["video_step"]),
            "push_track_err_mm": round(1000 * float(np.linalg.norm(tg[:, :3] - ee[:, :3], axis=1).max()), 2) if push.any() else -1,
            "touched": int(log["contact"].any()), "n_touched": int(log["contact"].any(0).sum()),
            "n_moved": int((moved > 0.01).sum()),
            "n_goals_true": int(labels["default"].sum()), "n_goals_true_state": int(labels["state"].sum())}


def run_scene(job):
    from gen_v3.language import caption
    from gen_v3.tasks import enumerate_goals
    from gen_v4.decisions import pick
    from gen_v5.sampler import sample_pushes
    from gen_v5.splits import sample_scene

    seed, idx, out, opts = job
    out = Path(out)
    env = get_env()
    spec = sample_scene(seed, idx, visual=opts["visual"], distractors=opts["distractors"])
    env.set_scene(spec)
    init_pos, init_dis, state0 = env.object_poses()[0], env.distractor_poses(), env.get_state()
    sid = spec["scene_id"]
    goals = enumerate_goals(spec, init_pos)
    gid = [g["goal_id"] for g in goals]
    gindex = {g: i for i, g in enumerate(gid)}
    evals, compound, texts, templates = pick(goals, spec, init_pos, init_dis, seed, idx)
    cands, n_dropped = sample_pushes(env, spec, init_pos, init_dis, (seed, idx, 21), n_total=opts["n_pushes"],
                                     n_lines=opts["n_lines"])
    env.set_state(state0)
    np.savez_compressed(out / "scenes" / f"{sid}_initial.npz", initial=env.render(spec["camera"], IMAGE_SIZE),
                        camera=spec["camera"])
    cap_rng = np.random.default_rng([seed, idx, 5])
    base = {"scene_id": sid, "split": spec["v5_split"], "lang_split": spec["split"], "camera": spec["camera"]}

    rows, instr_rows, states = [], [], []
    mats = {k: [] for k in ("default", "state", "strict")}
    for k, cand in enumerate(cands):
        traj_id = f"{sid}_c{k:02d}"
        t0 = time.time()
        log, frames, labels, end_state = run_push(env, spec, state0, cand, goals, init_pos,
                                                  out / "videos" / f"{traj_id}.mp4")
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
        cands2, _ = sample_pushes(env, spec, mid_pos, mid_dis, (seed, idx, 24, j), n_total=opts["n_step2"],
                                  n_lines=opts["n_step2_lines"])
        goals2 = goals + [g]
        for k, cand in enumerate(cands2):
            traj_id = f"{sid}_k{j}_s{k:02d}"
            t0 = time.time()
            log, frames, labels, _ = run_push(env, spec, mid, cand, goals2, mid_pos, out / "videos" / f"{traj_id}.mp4",
                                              origin_pos=init_pos)
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
        "instructions": texts, "templates": templates, "step2": step2_info, "n_pushes_dropped": n_dropped}))
    result = (rows, instr_rows, goal_rows, step2_rows, step2_instr)
    (out / "scenes" / f"{sid}.rows.json").write_text(json.dumps(result))  # marks the scene done (--resume)
    return result


SCENE_TIMEOUT = 1800  # s


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
    ap.add_argument("--n-scenes", type=int, default=8)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="data/v5_mini")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--pushes", type=int, default=64)
    ap.add_argument("--lines", type=int, default=8)
    ap.add_argument("--step2", type=int, default=16, help="pushes at step 2 of each two-push decision (0: none)")
    ap.add_argument("--step2-lines", type=int, default=4)
    ap.add_argument("--no-visual", dest="visual", action="store_false")
    ap.add_argument("--no-distractors", dest="distractors", action="store_false")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    for sub in ("trajs", "scenes", "steps", "videos"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    import mujoco

    from gen_v3 import language, scene_spec, scene_xml, tasks
    from gen_v4 import decisions, labels
    from gen_v5 import primitives, sampler, scene_xml as vx, splits

    opts = {"visual": args.visual, "distractors": args.distractors, "n_pushes": args.pushes, "n_lines": args.lines,
            "step2": args.step2 > 0, "n_step2": args.step2, "n_step2_lines": args.step2_lines}
    config = {
        "generator": "gen_v5", "seed": args.seed, "first": args.first, "n_scenes": args.n_scenes,
        "image_size": IMAGE_SIZE, "video_fps": VIDEO_FPS, "control_hz": CONTROL_HZ, **opts,
        "splits": splits.SPLIT_P, "held_region": [splits.HELD_REGION[0].tolist(), splits.HELD_REGION[1]],
        "held_combos": sorted(splits.HELD_COMBOS), "train_cameras": scene_spec.TRAIN_CAMERAS,
        "held_cameras": scene_spec.HELD_CAMERAS, "n_objects": scene_spec.N_OBJECTS, "n_zones": scene_spec.N_ZONES,
        "shapes": scene_xml.SHAPES, "sizes": scene_xml.SIZES, "colors": list(scene_xml.COLORS),
        "robot": {"model": "robosuite Panda", "base": vx.BASE.tolist(), "tool_len": vx.TOOL_LEN, "kp": vx.KP, "kv": vx.KV,
                  "home_q": vx.HOME_Q.tolist(), "push_speed": primitives.PUSH_SPEED, "travel_z": primitives.TRAVEL_Z,
                  "high_z": primitives.HIGH_Z, "joint_margin": sampler.JOINT_MARGIN.tolist()},
        "labels": {"move_min": labels.MOVE_MIN, "ref_max": labels.REF_MAX}, "n_open_decisions": decisions.N_OPEN,
        "goal_types": tasks.GOAL_TYPES, "relational_p": language.RELATIONAL_P, "mujoco": mujoco.__version__,
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
        f = out / "scenes" / f"v5s{args.seed}_{job[1]:05d}.rows.json"
        if args.resume and f.exists():
            collect(json.loads(f.read_text()))
        else:
            jobs_left.append(job)
    if args.workers > 1:
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
    os.environ.setdefault("MUJOCO_GL", "egl")
    main()
