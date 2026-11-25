"""Generate v6: goal-blind rollouts of 14 kinds of action on a simulated Franka Panda with a gripper, every
rollout labeled against every goal of its scene, decisions picked from the scene only.

    python -m gen_v6.generate --n-scenes 12 --out data/v6_mini --workers 4
    python -m gen_v6.generate --n-scenes 1200 --out data/v6 --workers 8 --resume

What changes from v5 (gen_v5.generate):
    actions    push, pull, sweep, poke, topple (slides), place, drop, stack, unstack, twist, stand_up (grasps),
               drawer, lid (open or shut a fixture) and toss, drawn goal-blind per scene family (gen_v6.sampler);
               64 per scene
    scenes     six families with containers, ramps, a seesaw, dominoes, a drawer cabinet and a lidded box,
               everyday objects, and hidden mass / friction / bounce (gen_v6.scene_spec)
    goals      22 goal types (gen_v6.goals); action-neutral instructions (gen_v6.language)
    decisions  8 single-step, 2 two-step, a two-placement tower, and a probe-then-act decision on a look-alike
               pair (gen_v6.decisions); 12 step-2 rollouts each
    splits     nine scene splits plus held-out action-goal pairs (gen_v6.splits)
    settling   a scene that is still moving 6 s after it is built is redrawn (spec["retry"])
Same file layout as v5, so the v5 feature scripts read it; the candidate is stored in each rollout's meta.
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
VIDEO_FPS = 5
SETTLE_TRIES = 3  # rounds of SETTLE_CHECK_STEPS idle steps before a scene counts as unsettled
SETTLE_CHECK_STEPS = 40
SETTLE_TOL = 0.002  # m an object may move in the last second of a settle round
N_RETRY = 6
SPARE = 8  # extra candidates drawn to replace rollouts the physics could not run
_ENV = {}


def get_env():
    if "env" not in _ENV:
        from gen_v6.env import GripperSceneEnv
        _ENV["env"] = GripperSceneEnv()
    return _ENV["env"]


def settle(env):
    """Idle until nothing moves; False if objects are still moving after SETTLE_TRIES rounds."""
    for _ in range(SETTLE_TRIES):
        trace = []
        for _ in range(SETTLE_CHECK_STEPS):
            env.step()
            trace.append(np.r_[env.object_poses()[0].ravel(), np.nan_to_num(env.fixture_state())])
        last = np.array(trace[-CONTROL_HZ:])
        if np.abs(last - last[0]).max() < SETTLE_TOL and not env.unstable():
            return True
    return False


def build_scene(env, seed, idx, stage2):
    from gen_v6.scene_spec import sample_scene
    for retry in range(N_RETRY):
        spec = sample_scene(seed, idx, stage2=stage2, retry=retry)
        env.set_scene(spec)
        if settle(env):
            return spec
    raise RuntimeError(f"scene {idx} never settled")


def run_rollout(env, spec, state, cand, goals, start, video_path, origin=None):
    """Restore state, run one candidate with video, label it. Returns (log, keyframes, labels, end state) or None
    if the physics blew up."""
    from gen_v5.video import write_mp4
    from gen_v6.labels import label_sets
    from gen_v6.sampler import GRASP_ACTIONS, execute

    env.set_state(state)
    frames, video, steps = {}, [], []
    every = CONTROL_HZ // VIDEO_FPS
    count = [0]

    def video_fn():
        if count[0] % every == 0:
            video.append(env.render(spec["camera"], IMAGE_SIZE))
            steps.append(count[0])
        count[0] += 1

    grasp = cand["action"] in GRASP_ACTIONS or (cand["action"] == "lid" and cand["params"]["mode"] == "open")
    names = (None, "grasped", "after_action", "final") if grasp else (None, "after_action", "final")
    log = execute(env, cand, frame_fn=lambda name: frames.__setitem__(name, env.render(spec["camera"], IMAGE_SIZE)),
                  frame_names=names, video_fn=video_fn)
    if env.unstable() or not np.isfinite(log["obj_pos"]).all():
        return None
    log["video_step"] = np.array(steps) - 1  # frame k shows the state after control step video_step[k] (-1: start)
    write_mp4(video_path, video, VIDEO_FPS)
    labels = label_sets(goals, log, spec, start, origin)
    return log, frames, labels, env.get_state()


def save_rollout(path, meta, frames, labels, log):
    names = list(frames)
    log = {k: v.astype(np.float32) if np.asarray(v).dtype == np.float64 else v for k, v in log.items()}
    np.savez_compressed(
        path, meta=json.dumps(meta), keyframe_names=np.array(names),
        keyframes=np.stack([frames[n] for n in names]).astype(np.uint8),
        goal_labels=labels["default"], goal_labels_state=labels["state"], goal_labels_strict=labels["strict"],
        **log)


CONTACT_PHASES = ("push", "strike", "lower", "lift", "carry", "turn", "grasp")


def rollout_row(cand, log, start_pos, labels):
    moved = np.linalg.norm(log["obj_pos"][-1, :, :2] - start_pos[:, :2], axis=1)
    ph = np.asarray(log["phase"])
    err = np.linalg.norm(log["target"][:, :3] - log["ee_state"][:, :3], axis=1)
    act = np.isin(ph, CONTACT_PHASES) & np.isfinite(err)
    s = cand["sampler"]
    return {"action": cand["action"], "action_id": int(cand["vec"][0]), "sampler_object": s.get("object", -1),
            "sampler_target": s.get("target", ""), "params": json.dumps(cand["params"]),
            **{f"v{k}": round(float(v), 5) for k, v in enumerate(cand["vec"])},
            "n_steps": len(ph), "n_video_frames": len(log["video_step"]),
            "track_err_mm": round(1000 * float(err[act].max()), 2) if act.any() else -1,
            "touched": int(log["contact"].any()), "n_touched": int(log["contact"].any(0).sum()),
            "held": int(log["held"].any()), "n_moved": int((moved > 0.01).sum()),
            "n_off_table": int((log["obj_pos"][-1, :, 2] < 0).sum()),
            "n_goals_true": int(labels["default"].sum()), "n_goals_true_state": int(labels["state"].sum())}


def goal_types(g):
    return [p["type"] for p in g["parts"]] if g["type"] in ("and", "probe") else g["type"]


def run_scene(job):
    from gen_v6 import decisions, splits
    from gen_v6.goals import enumerate_goals, goal_id, initial_state
    from gen_v6.language import caption
    from gen_v6.sampler import sample

    seed, idx, out, opts = job
    out = Path(out)
    env = get_env()
    spec = build_scene(env, seed, idx, opts["stage2"])
    state0 = env.get_state()
    S0 = initial_state(env)
    init_pos = S0["pos"]
    sid = spec["scene_id"]
    goals = enumerate_goals(spec, S0, opts["stage2"])
    gid = [g["goal_id"] for g in goals]
    gindex = {g: i for i, g in enumerate(gid)}
    evals, multi, texts, templates = decisions.pick(goals, spec, S0, seed, idx)
    n_want = opts["n_rollouts"]
    cands, n_dropped = sample(env, spec, np.random.default_rng([seed, idx, 21]), n_total=n_want + SPARE,
                              n_lines=opts["n_lines"])
    env.set_state(state0)
    np.savez_compressed(out / "scenes" / f"{sid}_initial.npz", initial=env.render(spec["camera"], IMAGE_SIZE),
                        camera=spec["camera"])
    cap_rng = np.random.default_rng([seed, idx, 5])
    base = {"scene_id": sid, "split": spec["v6_split"], "lang_split": spec["split"], "family": spec["family"],
            "camera": spec["camera"]}

    rows, instr_rows, states, kept, n_unstable = [], [], [], [], 0
    mats = {k: [] for k in ("default", "state", "strict")}
    for cand in cands:
        if len(rows) == n_want:
            break
        k = len(rows)
        traj_id = f"{sid}_c{k:02d}"
        t0 = time.time()
        res = run_rollout(env, spec, state0, cand, goals, S0, out / "videos" / f"{traj_id}.mp4")
        if res is None:
            n_unstable += 1
            continue
        log, frames, labels, end_state = res
        states.append(end_state)
        kept.append(cand)
        for name in mats:
            mats[name].append(labels[name])
        S1 = initial_state(env)
        save_rollout(out / "trajs" / f"{traj_id}.npz", {"traj_id": traj_id, "scene_id": sid, "candidate": cand},
                     frames, labels, log)
        rows.append({"traj_id": traj_id, **base, "n_objects": len(spec["objects"]), "n_fixtures": len(spec["fixtures"]),
                     "n_zones": len(spec["zones"]), "n_distractors": len(spec["distractors"]), "candidate_id": k,
                     "in_eval32": int(k < n_want // 2), **rollout_row(cand, log, init_pos, labels),
                     "seconds": round(time.time() - t0, 3), "caption": caption(spec, S0, S1, cap_rng)})
        for g in evals:
            i = gindex[g["goal_id"]]
            instr_rows.append({"traj_id": traj_id, **base, "goal_id": g["goal_id"], "goal_type": g["type"],
                               "source": g["source"], "instruction": g.get("text") or texts[g["goal_id"]],
                               "template": g.get("template") or templates[g["goal_id"]],
                               "action": cand["action"], "act_heldout": int(splits.act_heldout(cand["action"], g["type"])),
                               "in_eval32": int(k < n_want // 2), "label": int(labels["default"][i]),
                               "label_state": int(labels["state"][i]), "label_strict": int(labels["strict"][i])})
    mats = {k: np.stack(v) for k, v in mats.items()}
    mask = splits.act_mask([c["action"] for c in kept], [g["type"] for g in goals])
    np.savez_compressed(out / "scenes" / f"{sid}_labels.npz", goal_ids=np.array(gid),
                        traj_ids=np.array([r["traj_id"] for r in rows]), act_heldout=mask, **mats)

    step2_rows, step2_instr, step2_info = [], [], []
    for j, g in enumerate(multi if opts["n_step2"] else []):
        info = {"goal_id": g["goal_id"], "source": g["source"]}
        step2_info.append(info)
        rng = np.random.default_rng([seed, idx, 23, j])
        if g["type"] == "probe":
            # step 1: a slide or poke of one of the look-alikes, chosen by its design only
            first = [k for k, c in enumerate(kept) if c["action"] in ("push", "pull", "poke")
                     and c["sampler"].get("object", -1) in g["pair"]]
            target = g["parts"][0]
        else:
            # step 1: a rollout that reached one part and not the other. A tower is built bottom first; the two
            # parts of a compound are independent, so either may come first (part 0 if any rollout reached it).
            solved = [mats["default"][:, gindex[p["goal_id"]]].astype(bool) for p in g["parts"]]
            order = [0] if "tower" in g else [0, 1]
            first, part = [], 0
            for part in order:
                both = solved[part] & solved[1 - part]
                first = list(np.flatnonzero(solved[part] & ~both)) or list(np.flatnonzero(solved[part]))
                if first:
                    break
            if part == 1 and first:  # store the decision in the order it was done (labels judge the last part
                old = g["goal_id"]  # from the step-2 start, the earlier ones from the decision's start)
                g["parts"] = g["parts"][::-1]
                g["goal_id"] = info["goal_id"] = goal_id(g)
                texts[g["goal_id"]], templates[g["goal_id"]] = texts.pop(old), templates.pop(old)
            target = g
        info.update(step1_candidates=len(first))
        if not first:
            continue
        k1 = int(rng.choice(first))
        info["step1_traj_id"] = rows[k1]["traj_id"]
        mid = states[k1]
        env.set_state(mid)
        Smid = initial_state(env)
        cands2, _ = sample(env, spec, np.random.default_rng([seed, idx, 24, j]), n_total=opts["n_step2"] + 2,
                           n_lines=opts["n_step2_lines"])
        goals2 = goals + ([target] if target["goal_id"] not in gindex else [])
        ti = gindex.get(target["goal_id"], len(goals))
        n2 = 0
        for cand in cands2:
            if n2 == opts["n_step2"]:
                break
            traj_id = f"{sid}_k{j}_s{n2:02d}"
            t0 = time.time()
            res = run_rollout(env, spec, mid, cand, goals2, Smid, out / "videos" / f"{traj_id}.mp4", origin=S0)
            if res is None:
                n_unstable += 1
                continue
            log, frames, labels, _ = res
            S2 = initial_state(env)
            save_rollout(out / "steps" / f"{traj_id}.npz",
                         {"traj_id": traj_id, "scene_id": sid, "decision": g["goal_id"], "source": g["source"],
                          "step1_traj_id": info["step1_traj_id"], "candidate": cand}, frames, labels, log)
            step2_rows.append({"traj_id": traj_id, **base, "decision": g["goal_id"], "source": g["source"],
                               "step1_traj_id": info["step1_traj_id"], "candidate_id": n2,
                               **rollout_row(cand, log, Smid["pos"], {n: v[:len(goals)] for n, v in labels.items()}),
                               "seconds": round(time.time() - t0, 3), "caption": caption(spec, Smid, S2, cap_rng)})
            lab = {n: int(v[ti]) for n, v in labels.items()}
            step2_instr.append({"traj_id": traj_id, **base, "decision": g["goal_id"], "goal_id": target["goal_id"],
                                "goal_type": "and" if target["type"] == "and" else target["type"], "source": g["source"],
                                "instruction": texts[g["goal_id"]], "template": templates[g["goal_id"]],
                                "step1_traj_id": info["step1_traj_id"], "action": cand["action"],
                                "act_heldout": int(any(splits.act_heldout(cand["action"], t)
                                                       for t in np.atleast_1d(goal_types(g)))),
                                "label": lab["default"], "label_state": lab["state"], "label_strict": lab["strict"]})
            n2 += 1

    pos_counts = mats["default"].sum(0)
    goal_rows = [{**base, "goal_id": g["goal_id"], "goal_type": g["type"], "a": g.get("a", -1), "b": g.get("b", -1),
                  "c": g.get("c", -1), "f": g.get("f", -1), "n": g.get("n", -1), "in_view": int(g["in_view"]),
                  "instruction": texts[g["goal_id"]] or "", "template": templates[g["goal_id"]] or "",
                  "n_pos": int(pos_counts[i]), "n_pos_state": int(mats["state"][:, i].sum()),
                  "n_pos_strict": int(mats["strict"][:, i].sum()),
                  "n_pos_act_heldout": int((mats["default"][:, i] & mask[:, i]).sum())} for i, g in enumerate(goals)]
    (out / "scenes" / f"{sid}.json").write_text(json.dumps({
        "spec": spec, "init_state": {k: np.asarray(v).tolist() for k, v in S0.items()},
        "goals": goals, "eval_goals": [{"goal_id": g["goal_id"], "source": g["source"],
                                        "instruction": g.get("text") or texts[g["goal_id"]]} for g in evals],
        "multi_step": multi, "instructions": texts, "templates": templates, "step2": step2_info,
        "n_candidates_dropped": n_dropped, "n_unstable": n_unstable}, default=_json))
    result = (rows, instr_rows, goal_rows, step2_rows, step2_instr)
    (out / "scenes" / f"{sid}.rows.json").write_text(json.dumps(result, default=_json))  # marks the scene done
    return result


def _json(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


SCENE_TIMEOUT = 3600  # s


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


def config(args, opts):
    import mujoco

    from gen_v3 import scene_xml as sx
    from gen_v6 import catalog, decisions, goals, labels, language, primitives, sampler, scene_spec, scene_xml, splits
    return {
        "generator": "gen_v6", "seed": args.seed, "first": args.first, "n_scenes": args.n_scenes,
        "image_size": IMAGE_SIZE, "video_fps": VIDEO_FPS, "control_hz": CONTROL_HZ, **opts,
        "families": scene_spec.families(opts["stage2"]), "splits": scene_spec.SPLIT_P,
        "held_region": [scene_spec.HELD_REGION[0].tolist(), scene_spec.HELD_REGION[1]],
        "held_combos": sorted(scene_spec.HELD_COMBOS), "held_act": sorted(splits.HELD_ACT),
        "actions": sampler.ACTIONS, "action_mix": sampler.ACTION_MIX, "goal_types": goals.GOAL_TYPES,
        "decision_mix": decisions.DECISION_MIX, "n_single_decisions": decisions.N_SINGLE,
        "objects": catalog.object_names(opts["stage2"]), "fixtures": catalog.FIXTURES + catalog.FIXTURES_STAGE2,
        "hidden_props": {"mass_range": catalog.MASS_RANGE, "friction_range": catalog.FRICTION_RANGE,
                         "mass_train": catalog.MASS_TRAIN, "friction_train": catalog.FRICTION_TRAIN},
        "labels": {"move_min": labels.MOVE_MIN, "ref_max": labels.REF_MAX},
        "robot": {"model": "robosuite Panda + Panda gripper", "high_z": primitives.HIGH_Z,
                  "travel_z": primitives.TRAVEL_Z, "carry_z": primitives.CARRY_Z, "push_z": primitives.PUSH_Z},
        "relational_p": language.RELATIONAL_P, "property_p": language.PROPERTY_P,
        "settle": {"tries": SETTLE_TRIES, "steps": SETTLE_CHECK_STEPS, "tol": SETTLE_TOL, "retries": N_RETRY},
        "mujoco": mujoco.__version__,
        "physics": {"timestep": sx.TIMESTEP, "cone": "elliptic", "impratio": scene_xml.IMPRATIO},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-scenes", type=int, default=12)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="data/v6_mini")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--rollouts", type=int, default=64)
    ap.add_argument("--lines", type=int, default=4)
    ap.add_argument("--step2", type=int, default=12, help="rollouts at step 2 of each multi-step decision (0: none)")
    ap.add_argument("--step2-lines", type=int, default=2)
    ap.add_argument("--no-stage2", dest="stage2", action="store_false", help="no tools family, drawer or lid")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    for sub in ("trajs", "scenes", "steps", "videos"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    opts = {"stage2": args.stage2, "n_rollouts": args.rollouts, "n_lines": args.lines, "n_step2": args.step2,
            "n_step2_lines": args.step2_lines}
    (out / "config.json").write_text(json.dumps(config(args, opts), indent=2, default=_json))

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
        f = out / "scenes" / f"v6s{args.seed}_{job[1]:05d}.rows.json"
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
    print(f"{len(rows)} rollouts + {len(step2_rows)} step-2 rollouts in {time.time() - t0:.0f}s -> {out}/manifest.csv")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("MUJOCO_GL", "cgl" if os.uname().sysname == "Darwin" else "egl")  # macOS has no EGL
    main()
