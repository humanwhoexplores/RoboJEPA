"""Validate a v3 dataset and write validation_report.json + contact_sheet.png.

    python -m gen_v3.validate --data data/v3_mini

Fails (exit code 1) on label mismatches (single pushes and plans), split, template or
shape-word leakage, non-determinism, the arm changing the physics, broken plan logs,
NaNs, or an object that moved impossibly fast while still on the table.
"""
import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from gen_v3.generate import run_candidate, run_plan, start_scene
from gen_v3.language import SHAPE_WORDS, TEST, TEST_SHAPE_WORDS, TRAIN
from gen_v3.tasks import label_goals, on_table

MAX_SPEED = 2.0  # m/s for an object still on the table; above this is a physics blow-up
DEEP_PENETRATION = -0.03  # m; reported, not fatal
PLAN_FRAMES = ["initial", "after_push1", "mid1", "after_push2", "final"]


def template_ok(text, template, goal_type, split):
    """The row's template is in its split's bank, and the text is that template filled in."""
    bank = TEST if split == "test" else TRAIN
    if template not in bank[goal_type]:
        return False
    pattern = "^" + re.sub(r"\\\{(a|b|c|p1|p2)\\\}", ".+", re.escape(template)).replace(r"\{zone\}", r"\w+") + "$"
    return re.match(pattern, text) is not None


def words_in(text, words):
    return [w for w in words if re.search(r"\b" + re.escape(w) + r"\b", text)]


def check_rollout(z, goals, spec, labels_csv, errors_out, tid):
    """Shared physics checks. Returns recomputed labels."""
    lab = label_goals(goals, z["obj_pos"], z["phase"], spec)
    if not np.array_equal(lab, z["goal_labels"]):
        errors_out["mismatch"].append(tid)
    pos = z["obj_pos"]
    if not np.isfinite(pos).all():
        errors_out["blowup"].append((tid, "nan"))
    vel = np.linalg.norm(np.diff(pos, axis=0), axis=-1) * 20
    still_on = on_table(pos[1:]) & on_table(pos[:-1])
    if (vel[still_on] > MAX_SPEED).any():
        errors_out["blowup"].append((tid, float(vel[still_on].max())))
    errors_out["deep"] += int(z["min_dist"].min() < DEEP_PENETRATION)
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/v3_mini")
    ap.add_argument("--determinism", type=int, default=5, help="re-simulate N random rollouts (and N plans) from scratch")
    ap.add_argument("--sheet", type=int, default=12, help="single pushes in the contact sheet (plus 4 plans)")
    args = ap.parse_args()
    root = Path(args.data)
    m = pd.read_csv(root / "manifest.csv")
    ins = pd.read_csv(root / "instruction_rows.csv")
    has_plans = (root / "plan_manifest.csv").exists()
    pm = pd.read_csv(root / "plan_manifest.csv") if has_plans else pd.DataFrame()
    pins = pd.read_csv(root / "plan_instruction_rows.csv") if has_plans else pd.DataFrame()
    config = json.loads((root / "config.json").read_text())
    opts = {k: config[k] for k in ("cameras", "held_cameras", "held_share", "plans", "arm", "visual", "distractors")}
    scenes = {p.stem: json.loads(p.read_text()) for p in (root / "scenes").glob("*.json") if not p.name.endswith(".rows.json")}
    errors, report = [], {"n_rollouts": len(m), "n_plans": len(pm), "n_scenes": int(m.scene_id.nunique())}
    acc = {"mismatch": [], "blowup": [], "deep": 0}

    # 1. Every label recomputed from the stored object positions must match (single pushes).
    ins_by = {k: g for k, g in ins.groupby("traj_id")}
    for row in m.itertuples():
        z = np.load(root / "trajs" / f"{row.traj_id}.npz")
        sc = scenes[row.scene_id]
        lab = check_rollout(z, sc["goals"], sc["spec"], None, acc, row.traj_id)
        ids = [g["goal_id"] for g in sc["goals"]]
        if isinstance(row.focus_goal, str) and lab[ids.index(row.focus_goal)] != row.own_goal_success:
            acc["mismatch"].append(row.traj_id)
        for r in ins_by.get(row.traj_id, pd.DataFrame()).itertuples():
            if lab[ids.index(r.goal_id)] != r.label:
                acc["mismatch"].append(row.traj_id)

    # 1b. Plans: labels (single and compound goals), push_idx and frames.
    bad_plan_logs = []
    pins_by = {k: g for k, g in pins.groupby("traj_id")} if has_plans else {}
    for row in pm.itertuples():
        z = np.load(root / "plans" / f"{row.traj_id}.npz")
        sc = scenes[row.scene_id]
        goals = sc["goals"] + sc["compound"]
        lab = check_rollout(z, goals, sc["spec"], None, acc, row.traj_id)
        ids = [g["goal_id"] for g in goals]
        if isinstance(row.focus_goal, str) and lab[ids.index(row.focus_goal)] != row.own_goal_success:
            acc["mismatch"].append(row.traj_id)
        for r in pins_by.get(row.traj_id, pd.DataFrame()).itertuples():
            if lab[ids.index(r.goal_id)] != r.label:
                acc["mismatch"].append(row.traj_id)
        pi = z["push_idx"]
        settled_mid = (z["phase"][pi == 0][-1] == "settle") if (pi == 0).any() else False
        if not (np.all(np.diff(pi) >= 0) and set(pi.tolist()) == {0, 1} and settled_mid
                and z["keyframe_names"].tolist() == PLAN_FRAMES):
            bad_plan_logs.append(row.traj_id)
    mism = sorted(set(acc["mismatch"]))
    report["label_mismatches"] = len(mism)
    report["physics_blowups"] = acc["blowup"]
    report["rollouts_with_deep_penetration"] = acc["deep"]
    report["bad_plan_logs"] = len(bad_plan_logs)
    if mism:
        errors.append(f"{len(mism)} rollouts with label mismatches, e.g. {mism[:3]}")
    if acc["blowup"]:
        errors.append(f"physics blow-ups: {acc['blowup'][:3]}")
    if bad_plan_logs:
        errors.append(f"{len(bad_plan_logs)} plans with bad push_idx / frames, e.g. {bad_plan_logs[:3]}")

    # 2. Leakage: one split per scene; test uses only held-out templates and shape words, train/val never do.
    every = pd.concat([m[["scene_id", "split"]], pm[["scene_id", "split"]]]) if has_plans else m
    if (every.groupby("scene_id").split.nunique() > 1).any():
        errors.append("a scene appears in several splits")
    train_words = list(itertools.chain(*SHAPE_WORDS.values()))
    test_words = list(itertools.chain(*TEST_SHAPE_WORDS.values()))
    lang = pd.concat([ins, pins]) if has_plans else ins
    bad_tmpl, bad_words = [], []
    for r in lang.drop_duplicates(["scene_id", "goal_id"]).itertuples():
        if not template_ok(r.instruction, r.template, r.goal_type, r.split):
            bad_tmpl.append(r.instruction)
        if words_in(r.instruction, train_words if r.split == "test" else test_words):
            bad_words.append(r.instruction)
    report["instruction_template_errors"] = len(bad_tmpl)
    report["shape_word_leaks"] = len(bad_words)
    if bad_tmpl:
        errors.append(f"{len(bad_tmpl)} instructions not matching their template bank, e.g. {bad_tmpl[:2]}")
    if bad_words:
        errors.append(f"{len(bad_words)} instructions with the other split's shape words, e.g. {bad_words[:2]}")
    held = set(opts["held_cameras"])
    cam_leak = m[(m.split != "test") & m.camera.isin(held)]
    if len(cam_leak):
        errors.append(f"{cam_leak.scene_id.nunique()} train/val scenes use a held-out camera")

    # 3. Determinism: rebuild the scene from its seed and re-run (single pushes and plans).
    rng = np.random.default_rng(0)
    diffs, plan_diffs, arm_diffs = [], [], []
    for tid in rng.choice(m.traj_id, size=min(args.determinism, len(m)), replace=False):
        row = m[m.traj_id == tid].iloc[0]
        sc = scenes[row.scene_id]
        spec, _, _, state0 = start_scene(config["seed"], sc["spec"]["scene_idx"], opts)
        z = np.load(root / "trajs" / f"{tid}.npz")
        cand = json.loads(str(z["meta"]))["candidate"]
        log, _, _ = run_candidate(spec, state0, cand, sc["goals"], opts, with_images=False)
        stored = z["obj_pos"]
        diffs.append(float(np.abs(log["obj_pos"] - stored).max()) if log["obj_pos"].shape == stored.shape else float("inf"))
        if opts["arm"]:  # the arm is visual only: the same push without it must give the same motion
            no_arm = dict(opts, arm=False)
            spec2, _, _, s2 = start_scene(config["seed"], sc["spec"]["scene_idx"], no_arm)
            log2, _, _ = run_candidate(spec2, s2, cand, sc["goals"], no_arm, with_images=False)
            arm_diffs.append(float(np.abs(log2["obj_pos"] - stored).max()) if log2["obj_pos"].shape == stored.shape else float("inf"))
    if has_plans and len(pm):
        for tid in rng.choice(pm.traj_id, size=min(args.determinism, len(pm)), replace=False):
            row = pm[pm.traj_id == tid].iloc[0]
            sc = scenes[row.scene_id]
            spec, _, _, state0 = start_scene(config["seed"], sc["spec"]["scene_idx"], opts)
            z = np.load(root / "plans" / f"{tid}.npz")
            meta = json.loads(str(z["meta"]))
            plan = {"pushes": meta["pushes"], "second": None}
            log, _, _, _ = run_plan(spec, state0, plan, sc["goals"] + sc["compound"], opts, with_images=False)
            stored = z["obj_pos"]
            plan_diffs.append(float(np.abs(log["obj_pos"] - stored).max()) if log["obj_pos"].shape == stored.shape else float("inf"))
    report["determinism_max_abs_diff"] = diffs
    report["plan_determinism_max_abs_diff"] = plan_diffs
    report["arm_vs_no_arm_max_abs_diff"] = arm_diffs
    if any(d > 0 for d in diffs + plan_diffs):
        errors.append(f"non-deterministic re-simulation: {diffs} {plan_diffs}")
    if any(d > 0 for d in arm_diffs):
        errors.append(f"the arm changes the physics: {arm_diffs}")

    # 4. Balance and sanity numbers (reported, not fatal).
    focus_rows = m[m.own_goal_success >= 0].copy()
    focus_rows["goal_type"] = focus_rows.focus_goal.str.split("|").str[0]
    report["own_goal_success_by_candidate_type"] = focus_rows.groupby("candidate_type").own_goal_success.mean().round(3).to_dict()
    ideal = focus_rows[focus_rows.candidate_type == "ideal"]
    report["ideal_success_by_goal_type"] = ideal.groupby("goal_type").own_goal_success.mean().round(3).to_dict()
    report["touch_rate_by_candidate_type"] = m.groupby("candidate_type").touched.mean().round(3).to_dict()
    aimed = m[m.touched_intended >= 0]
    report["touched_intended_object_by_type"] = aimed.groupby("candidate_type").touched_intended.mean().round(3).to_dict()
    report["positive_rate_by_goal_type"] = ins.groupby("goal_type").label.mean().round(3).to_dict()
    solvable = ins.groupby(["scene_id", "goal_id"]).label.max()
    report["decisions"] = int(len(solvable))
    report["solvable_decision_share"] = round(float(solvable.mean()), 3)
    gt = ins.drop_duplicates(["scene_id", "goal_id"]).set_index(["scene_id", "goal_id"]).goal_type
    report["solvable_share_by_goal_type"] = solvable.groupby(gt.reindex(solvable.index).values).mean().round(3).to_dict()
    report["decisions_by_goal_type"] = gt.value_counts().to_dict()
    if has_plans and len(pm):
        fp = pm[pm.own_goal_success >= 0]
        report["plan_success_by_type"] = fp.groupby("plan_type").own_goal_success.mean().round(3).to_dict()
        report["plan_first_part_success_by_type"] = fp.groupby("plan_type").part1_success.mean().round(3).to_dict()
        report["plan_first_part_success_after_push1"] = fp.groupby("plan_type").part1_mid_success.mean().round(3).to_dict()
        psolv = pins.groupby(["scene_id", "goal_id"]).label.max()
        report["plan_decisions"] = int(len(psolv))
        report["plan_solvable_share"] = round(float(psolv.mean()), 3)
    report["split_scenes"] = m.drop_duplicates("scene_id").split.value_counts().to_dict()
    report["camera_by_split"] = {s: g.drop_duplicates("scene_id").camera.value_counts().to_dict() for s, g in m.groupby("split")}
    report["objects_per_scene"] = m.drop_duplicates("scene_id").n_objects.value_counts().sort_index().to_dict()
    report["distractors_per_scene"] = m.drop_duplicates("scene_id").n_distractors.value_counts().sort_index().to_dict()
    shapes = pd.Series([o["shape"] for s in scenes.values() for o in s["spec"]["objects"]])
    report["objects_by_shape"] = shapes.value_counts().to_dict()
    rel = lang.drop_duplicates(["scene_id", "goal_id"]).instruction.str.contains("nearest|leftmost|rightmost")
    report["instructions_with_relational_reference"] = round(float(rel.mean()), 3)
    report["unique_instructions"] = int(lang.instruction.nunique())
    report["caption_nothing_moved_share"] = round(float((m.caption == "nothing moved").mean()), 3)
    report["mean_seconds_per_rollout"] = round(float(m.seconds.mean()), 3)
    if has_plans and len(pm):
        report["mean_seconds_per_plan"] = round(float(pm.seconds.mean()), 3)
    report["errors"] = errors
    (root / "validation_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))

    # 5. Contact sheet: single pushes (initial | after_push | final) and plans (5 frames), with instruction and label.
    def tile(strip, top, bottom, ok):
        canvas = Image.new("RGB", (strip.shape[1], strip.shape[0] + 34), "white")
        canvas.paste(Image.fromarray(strip), (0, 0))
        d = ImageDraw.Draw(canvas)
        d.text((4, strip.shape[0] + 3), top, fill=(90, 90, 90))
        d.text((4, strip.shape[0] + 18), bottom, fill=(0, 120, 0) if ok else (170, 0, 0))
        return np.asarray(canvas)

    tiles = []
    pick = focus_rows.sample(n=min(args.sheet, len(focus_rows)), random_state=0).sort_values("traj_id")
    for row in pick.itertuples():
        z = np.load(root / "trajs" / f"{row.traj_id}.npz")
        strip = np.concatenate(list(z["keyframes"]), axis=1)
        strip = np.concatenate([strip, np.full((strip.shape[0], 2 * strip.shape[0], 3), 255, np.uint8)], axis=1)
        text = scenes[row.scene_id]["instructions"][row.focus_goal]
        tiles.append(tile(strip, f"{row.traj_id}  {row.candidate_type}  {row.camera}",
                          f"{text}  ->  {'SUCCESS' if row.own_goal_success else 'fail'}", row.own_goal_success))
    if has_plans and len(pm):
        for row in pm[pm.own_goal_success >= 0].sample(n=min(4, len(pm)), random_state=0).itertuples():
            z = np.load(root / "plans" / f"{row.traj_id}.npz")
            text = scenes[row.scene_id]["instructions"][row.focus_goal]
            tiles.append(tile(np.concatenate(list(z["keyframes"]), axis=1), f"{row.traj_id}  {row.plan_type}  {row.camera}",
                              f"{text[:110]}  ->  {'SUCCESS' if row.own_goal_success else 'fail'}", row.own_goal_success))
    grid = np.concatenate(tiles, axis=0)
    Image.fromarray(grid).save(root / "contact_sheet.png")

    if errors:
        print("FAILED:\n  " + "\n  ".join(errors))
        sys.exit(1)
    print(f"PASSED  ({root / 'validation_report.json'}, {root / 'contact_sheet.png'})")


if __name__ == "__main__":
    main()
