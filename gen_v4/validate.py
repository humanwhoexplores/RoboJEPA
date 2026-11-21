"""Validate a v4 dataset and write validation_report.json + contact_sheet.png.

    python -m gen_v4.validate --data data/v4_mini [--v3 data/v3]

Fails (exit code 1) on: label mismatches (any of the 3 labels, start pushes and step-2 pushes, npz vs
label matrix vs CSVs), a default positive whose named object moved less than MOVE_MIN, split / template /
shape-word / camera leakage, non-determinism, the arm changing the physics, the sampler not reproducing
the stored pushes from the scene alone, uneven pushes per object, NaNs or physics blow-ups, and (with
--v3) scenes that differ from v3's.

Reports (not fatal): the shortcut baselines (random, touches-named-object, no-physics rule, oracle) per
goal type and decision source, solvable shares, the candidate-index test, and positives per scene.
"""
import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from gen_v3.generate import get_env, start_scene
from gen_v3.language import SHAPE_WORDS, TEST_SHAPE_WORDS
from gen_v3.tasks import on_table, target_point
from gen_v3.validate import template_ok, words_in
from gen_v4.generate import run_push, scene_ids
from gen_v4.labels import LABELS, MOVE_MIN, label_sets
from gen_v4.sampler import sample_pushes

MAX_SPEED = 2.0
DEEP_PENETRATION = -0.03
V3_SCENE_TOL = 1e-3  # m; the same code on another machine can differ in the last float bits


def chi2_sf(x, k):
    """Upper tail of the chi-square distribution (Wilson-Hilferty approximation; no scipy needed)."""
    if k <= 0:
        return 1.0
    z = ((x / k) ** (1 / 3) - (1 - 2 / (9 * k))) / math.sqrt(2 / (9 * k))
    return 0.5 * math.erfc(z / math.sqrt(2))


def physics_checks(z, acc, tid):
    pos = z["obj_pos"]
    if not np.isfinite(pos).all():
        acc["blowup"].append((tid, "nan"))
    vel = np.linalg.norm(np.diff(pos, axis=0), axis=-1) * 20
    still_on = on_table(pos[1:]) & on_table(pos[:-1])
    if (vel[still_on] > MAX_SPEED).any():
        acc["blowup"].append((tid, float(vel[still_on].max())))
    acc["deep"] += int(z["min_dist"].min() < DEEP_PENETRATION)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/v4_mini")
    ap.add_argument("--v3", default="", help="v3 data folder: check that v4 rebuilt the same scenes")
    ap.add_argument("--determinism", type=int, default=5)
    ap.add_argument("--sheet", type=int, default=12)
    args = ap.parse_args()
    root = Path(args.data)
    m = pd.read_csv(root / "manifest.csv")
    ins = pd.read_csv(root / "instruction_rows.csv")
    has2 = (root / "step2_manifest.csv").exists()
    m2 = pd.read_csv(root / "step2_manifest.csv") if has2 else pd.DataFrame()
    ins2 = pd.read_csv(root / "step2_instruction_rows.csv") if has2 else pd.DataFrame()
    config = json.loads((root / "config.json").read_text())
    opts = {k: config[k] for k in ("cameras", "held_cameras", "held_share", "plans", "arm", "visual", "distractors",
                                   "n_pushes", "n_lines", "step2", "n_step2", "n_step2_lines")}
    scenes = {p.stem: json.loads(p.read_text()) for p in (root / "scenes").glob("*.json")
              if not p.name.endswith(".rows.json")}
    errors = []
    report = {"n_scenes": len(scenes), "n_pushes": len(m), "n_step2_pushes": len(m2)}
    acc = {"mismatch": [], "blowup": [], "deep": 0, "agent": []}

    # 1. Labels: recompute all three from stored positions; compare with npz, label matrix and CSVs.
    ins_by = {k: g for k, g in ins.groupby("traj_id")}
    touched = {}
    for sid, grp in m.groupby("scene_id"):
        sc = scenes[sid]
        goals, pos0 = sc["goals"], np.asarray(sc["init_pos"])
        ids = [g["goal_id"] for g in goals]
        mats = np.load(root / "scenes" / f"{sid}_labels.npz")
        if mats["goal_ids"].tolist() != ids:
            acc["mismatch"].append(f"{sid} goal order")
        for row in grp.itertuples():
            z = np.load(root / "trajs" / f"{row.traj_id}.npz")
            touched[row.traj_id] = z["contact"].any(0)
            lab = label_sets(goals, z["obj_pos"], z["phase"], sc["spec"], pos0)
            for name, key in zip(LABELS, ("goal_labels", "goal_labels_state", "goal_labels_strict")):
                if not (np.array_equal(lab[name], z[key]) and np.array_equal(lab[name], mats[name][row.candidate_id])):
                    acc["mismatch"].append(row.traj_id)
            moved = np.linalg.norm(z["obj_pos"][-1, :, :2] - pos0[:, :2], axis=1)
            for i in np.flatnonzero(lab["default"]):
                if moved[goals[i]["a"]] < MOVE_MIN:
                    acc["agent"].append(row.traj_id)
            for r in ins_by.get(row.traj_id, pd.DataFrame()).itertuples():
                i = ids.index(r.goal_id)
                if (r.label, r.label_state, r.label_strict) != tuple(int(lab[n][i]) for n in LABELS):
                    acc["mismatch"].append(row.traj_id)
            physics_checks(z, acc, row.traj_id)
    ins2_by = {k: g for k, g in ins2.groupby("traj_id")} if has2 else {}
    for row in m2.itertuples() if has2 else []:
        sc = scenes[row.scene_id]
        g = next(c for c in sc["compound"] if c["goal_id"] == row.compound_goal)
        goals = sc["goals"] + [g]
        z = np.load(root / "steps" / f"{row.traj_id}.npz")
        z1 = np.load(root / "trajs" / f"{row.step1_traj_id}.npz")
        mid = z1["obj_pos"][-1]
        if np.abs(z["obj_pos"][0] - mid).max() > 0.002:  # step 2 starts where step 1 settled
            acc["mismatch"].append(f"{row.traj_id} start")
        lab = label_sets(goals, z["obj_pos"], z["phase"], sc["spec"], mid, origin_pos=np.asarray(sc["init_pos"]))
        for name, key in zip(LABELS, ("goal_labels", "goal_labels_state", "goal_labels_strict")):
            if not np.array_equal(lab[name], z[key]):
                acc["mismatch"].append(row.traj_id)
        for r in ins2_by.get(row.traj_id, pd.DataFrame()).itertuples():
            if (r.label, r.label_state, r.label_strict) != tuple(int(lab[n][-1]) for n in LABELS):
                acc["mismatch"].append(row.traj_id)
        physics_checks(z, acc, row.traj_id)
    mism = sorted(set(acc["mismatch"]))
    report["label_mismatches"] = len(mism)
    report["default_positives_named_object_not_moved"] = len(acc["agent"])
    report["physics_blowups"] = acc["blowup"]
    report["rollouts_with_deep_penetration"] = acc["deep"]
    if mism:
        errors.append(f"{len(mism)} label mismatches, e.g. {mism[:3]}")
    if acc["agent"]:
        errors.append(f"{len(acc['agent'])} default positives where the named object did not move")
    if acc["blowup"]:
        errors.append(f"physics blow-ups: {acc['blowup'][:3]}")

    # 2. Leakage: one split per scene; test uses only held-out templates and words; held-out cameras only in test.
    if (m.groupby("scene_id").split.nunique() > 1).any():
        errors.append("a scene appears in several splits")
    goals_csv = pd.read_csv(root / "all_goals.csv")
    lang = pd.concat([goals_csv[["split", "goal_type", "instruction", "template"]],
                      ins2.drop_duplicates(["scene_id", "goal_id"])[["split", "goal_type", "instruction", "template"]]
                      if has2 and len(ins2) else pd.DataFrame()])
    train_words = list(itertools.chain(*SHAPE_WORDS.values()))
    test_words = list(itertools.chain(*TEST_SHAPE_WORDS.values()))
    bad_t = [r.instruction for r in lang.itertuples() if not template_ok(r.instruction, r.template, r.goal_type, r.split)]
    bad_w = [r.instruction for r in lang.itertuples() if words_in(r.instruction, train_words if r.split == "test" else test_words)]
    report["instruction_template_errors"] = len(bad_t)
    report["shape_word_leaks"] = len(bad_w)
    if bad_t:
        errors.append(f"{len(bad_t)} instructions not matching their template bank, e.g. {bad_t[:2]}")
    if bad_w:
        errors.append(f"{len(bad_w)} instructions with the other split's shape words, e.g. {bad_w[:2]}")
    if len(m[(m.split != "test") & m.camera.isin(set(opts["held_cameras"]))]):
        errors.append("train/val scenes use a held-out camera")

    # 3. Blindness, coverage and determinism: rebuild scenes from the seed, re-sample, re-simulate.
    rng = np.random.default_rng(0)
    seed = config["seed"]
    resample_bad, uneven, diffs, arm_diffs = [], [], [], []
    for sid in sorted(scenes):
        sc = scenes[sid]
        d = m[m.scene_id == sid].sort_values("candidate_id")
        obj = d[d.candidate_type != "random_line"].sampler_object.value_counts().reindex(range(len(sc["spec"]["objects"])), fill_value=0)
        if obj.max() - obj.min() > 1:
            uneven.append(sid)
    for sid in rng.choice(sorted(scenes), size=min(args.determinism, len(scenes)), replace=False):
        sc = scenes[sid]
        idx = sc["spec"]["scene_idx"]
        spec, pos0, dis0, state0 = start_scene(seed, idx, opts)
        spec = scene_ids(spec, seed, idx)
        # The sampler sees only the scene and its own stream: no goals are passed, so none can leak in.
        again = sample_pushes(spec, pos0, dis0, np.random.default_rng([seed, idx, 21]), opts["n_pushes"], opts["n_lines"])
        stored = [json.loads(str(np.load(root / "trajs" / f"{t}.npz")["meta"]))["candidate"] for t in
                  m[m.scene_id == sid].sort_values("candidate_id").traj_id]
        if again != stored:
            resample_bad.append(sid)
        k = int(rng.integers(len(stored)))
        tid = f"{sid}_c{k:02d}"
        z = np.load(root / "trajs" / f"{tid}.npz")
        log, _, _, _ = run_push(get_env(opts["arm"]), spec, state0, stored[k], sc["goals"], pos0, with_images=False)
        diffs.append(float(np.abs(log["obj_pos"] - z["obj_pos"]).max()) if log["obj_pos"].shape == z["obj_pos"].shape else float("inf"))
        if opts["arm"]:
            no_arm = dict(opts, arm=False)
            spec2, p2, _, s2 = start_scene(seed, idx, no_arm)
            log2, _, _, _ = run_push(get_env(False), scene_ids(spec2, seed, idx), s2, stored[k], sc["goals"], p2,
                                     with_images=False)
            arm_diffs.append(float(np.abs(log2["obj_pos"] - z["obj_pos"]).max()) if log2["obj_pos"].shape == z["obj_pos"].shape else float("inf"))
    report["sampler_reproduces_pushes_from_scene_only"] = not resample_bad
    report["scenes_with_uneven_pushes_per_object"] = len(uneven)
    report["determinism_max_abs_diff"] = diffs
    report["arm_vs_no_arm_max_abs_diff"] = arm_diffs
    if resample_bad:
        errors.append(f"sampler does not reproduce the stored pushes for {resample_bad[:3]}")
    if uneven:
        errors.append(f"{len(uneven)} scenes with uneven pushes per object, e.g. {uneven[:3]}")
    if any(d > 0 for d in diffs):
        errors.append(f"non-deterministic re-simulation: {diffs}")
    if any(d > 0 for d in arm_diffs):
        errors.append(f"the arm changes the physics: {arm_diffs}")

    # 4. Same scenes as v3 (optional): spec and settled start positions.
    if args.v3:
        v3dir = Path(args.v3) / "scenes"
        worst, missing, spec_diff = 0.0, 0, 0
        for sid, sc in scenes.items():
            f = v3dir / f"{sc['spec']['v3_scene_id']}.json"
            if not f.exists():
                missing += 1
                continue
            v3 = json.loads(f.read_text())
            worst = max(worst, float(np.abs(np.asarray(v3["init_pos"]) - np.asarray(sc["init_pos"])).max()))
            a = {k: v for k, v in sc["spec"].items() if k not in ("scene_id", "v3_scene_id")}
            b = {k: v for k, v in v3["spec"].items() if k != "scene_id"}
            spec_diff += int(a != b)
            v4focus = [e["goal_id"] for e in sc["eval_goals"] if e["source"] == "v3_focus"]
            spec_diff += int(v4focus != v3["focus"])
            spec_diff += int(any(sc["instructions"][g] != v3["instructions"][g] for g in v3["instructions"]))
        report["v3_scene_check"] = {"missing": missing, "spec_or_focus_or_text_differences": spec_diff,
                                    "init_pos_max_abs_diff": worst}
        if spec_diff or worst > V3_SCENE_TOL:
            errors.append(f"scenes differ from v3: {report['v3_scene_check']}")

    # 5. Bias report: shortcut baselines, solvable shares, candidate-index test.
    theta = dict(zip(m.traj_id, m.theta))
    rows = []
    for (sid, gid), d in ins.groupby(["scene_id", "goal_id"], sort=False):
        for sub, dd in (("all", d), ("eval32", d[d.in_eval32 == 1])):
            sc = scenes[sid]
            g = next(x for x in sc["goals"] if x["goal_id"] == gid)
            pos = np.asarray(sc["init_pos"])
            v = target_point(g, sc["spec"], pos) - pos[g["a"], :2]
            phi = math.atan2(v[1], v[0])
            lab = dd.label.to_numpy()
            tm = np.array([touched[t][g["a"]] for t in dd.traj_id])
            if tm.any():
                ix = np.flatnonzero(tm)
                th = np.array([theta[t] for t in dd.traj_id.to_numpy()[ix]])
                rule, ground = lab[ix[np.argmax(np.cos(th - phi))]], lab[tm].mean()
            else:
                rule = ground = lab.mean()
            rows.append({"subset": sub, "goal_type": g["type"], "source": d.source.iloc[0],
                         "straight_path": int(d.straight_path.iloc[0]), "random": lab.mean(),
                         "touches_named": ground, "no_physics_rule": rule, "oracle": lab.max(), "n_pos": lab.sum()})
    bd = pd.DataFrame(rows)
    cols = ["random", "touches_named", "no_physics_rule", "oracle"]
    full = bd[bd.subset == "all"]
    report["v3_reference_baselines"] = {"random": 0.131, "touches_named": 0.366, "no_physics_rule": 0.637, "oracle": 0.978}
    report["baselines"] = full[cols].mean().round(3).to_dict()
    report["baselines_eval32"] = bd[bd.subset == "eval32"][cols].mean().round(3).to_dict()
    report["baselines_by_source"] = {s: d[cols].mean().round(3).to_dict() for s, d in full.groupby("source")}
    report["baselines_by_goal_type"] = {t: d[cols].mean().round(3).to_dict() for t, d in full.groupby("goal_type")}
    solv = full[full.oracle > 0]
    rnd = report["baselines"]["random"]
    orc = report["baselines"]["oracle"]
    report["no_physics_rule_normalized"] = round((report["baselines"]["no_physics_rule"] - rnd) / max(orc - rnd, 1e-9), 3)
    report["no_physics_rule_on_solvable"] = round(float(solv.no_physics_rule.mean()), 3) if len(solv) else None
    report["decisions"] = int(len(full))
    report["solvable_share"] = round(float((full.oracle > 0).mean()), 3)
    report["solvable_share_by_source"] = full.groupby("source").oracle.apply(lambda s: round(float((s > 0).mean()), 3)).to_dict()
    report["solvable_share_by_goal_type"] = full.groupby("goal_type").oracle.apply(lambda s: round(float((s > 0).mean()), 3)).to_dict()
    report["solvable_share_by_straight_path"] = full.groupby("straight_path").oracle.apply(lambda s: round(float((s > 0).mean()), 3)).to_dict()
    report["positives_per_solvable_decision"] = round(float(solv.n_pos.mean()), 2) if len(solv) else 0
    report["label_rates"] = {n: round(float(ins[n].mean()), 4) for n in ("label", "label_state", "label_strict")}
    # Candidate-index test: is success rate the same at every index? (v3 fails this: index = push type)
    ct = ins.groupby(ins.traj_id.str[-2:].astype(int)).label.agg(["sum", "count"])
    p = ct["sum"].sum() / ct["count"].sum()
    exp1, exp0 = ct["count"] * p, ct["count"] * (1 - p)
    chi2 = float((((ct["sum"] - exp1) ** 2) / exp1.clip(lower=1e-9) + (((ct["count"] - ct["sum"]) - exp0) ** 2) / exp0.clip(lower=1e-9)).sum())
    report["index_test"] = {"chi2": round(chi2, 1), "dof": len(ct) - 1, "p_value": round(chi2_sf(chi2, len(ct) - 1), 4)}
    mats_pos = [int(np.load(root / "scenes" / f"{sid}_labels.npz")["default"].sum()) for sid in scenes]
    report["label_matrix_positives_per_scene"] = round(float(np.mean(mats_pos)), 1)
    report["goals_per_scene"] = round(len(goals_csv) / max(len(scenes), 1), 1)
    report["pushes_by_type"] = m.candidate_type.value_counts().to_dict()
    report["touch_rate"] = round(float(m.touched.mean()), 3)
    if has2:
        info = [i for sc in scenes.values() for i in sc["step2"]]
        s1 = [i for i in info if i["step1_solvable"]]
        report["two_push_decisions"] = len(info)
        report["two_push_step1_solvable_share"] = round(len(s1) / max(len(info), 1), 3)
        if len(ins2):
            per = ins2.groupby(["scene_id", "goal_id"]).agg(any_ok=("label", "max"), part2=("part2_label", "max"))
            report["two_push_step2_solvable_given_step1"] = round(float(per.any_ok.mean()), 3)
            report["two_push_solvable_share"] = round(float(per.any_ok.sum()) / max(len(info), 1), 3)
            report["two_push_random_given_step1"] = round(float(ins2.label.mean()), 4)
    report["split_scenes"] = m.drop_duplicates("scene_id").split.value_counts().to_dict()
    report["caption_nothing_moved_share"] = round(float((m.caption == "nothing moved").mean()), 3)
    report["mean_seconds_per_push"] = round(float(m.seconds.mean()), 3)
    report["errors"] = errors
    (root / "validation_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))

    # 6. Contact sheet: start frame | after_push | final, with one of the scene's eval instructions and its label.
    def tile(strip, top, bottom, ok):
        canvas = Image.new("RGB", (strip.shape[1], strip.shape[0] + 34), "white")
        canvas.paste(Image.fromarray(strip), (0, 0))
        dr = ImageDraw.Draw(canvas)
        dr.text((4, strip.shape[0] + 3), top, fill=(90, 90, 90))
        dr.text((4, strip.shape[0] + 18), bottom, fill=(0, 120, 0) if ok else (170, 0, 0))
        return np.asarray(canvas)

    pos_rows = ins[ins.label == 1]
    pick = pd.concat([pos_rows.sample(n=min(args.sheet // 2, len(pos_rows)), random_state=0),
                      ins.sample(n=min(args.sheet - args.sheet // 2, len(ins)), random_state=1)]).sort_values("traj_id")
    tiles = []
    for r in pick.itertuples():
        z = np.load(root / "trajs" / f"{r.traj_id}.npz")
        init = np.load(root / "scenes" / f"{r.scene_id}_initial.npz")["initial"]
        strip = np.concatenate([init] + list(z["keyframes"]), axis=1)
        strip = np.concatenate([strip, np.full((strip.shape[0], 2 * strip.shape[0], 3), 255, np.uint8)], axis=1)
        tiles.append(tile(strip, f"{r.traj_id}  {r.source}  {r.camera}",
                          f"{r.instruction}  ->  {'SUCCESS' if r.label else 'fail'}", r.label))
    if tiles:
        Image.fromarray(np.concatenate(tiles, axis=0)).save(root / "contact_sheet.png")

    if errors:
        print("FAILED:\n  " + "\n  ".join(errors))
        sys.exit(1)
    print(f"PASSED  ({root / 'validation_report.json'}, {root / 'contact_sheet.png'})")


if __name__ == "__main__":
    main()
