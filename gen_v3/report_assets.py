"""Build report assets for a validated v3 dataset: stats.json, samples.json, JPEG images, copies of the reports.

    python -m gen_v3.report_assets --data data/v3_mini --out gen_v3_report_assets

Run gen_v3.validate on the same --data first (it writes validation_report.json).
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from gen_v3 import scene_xml as sx
from gen_v3.env import SceneEnv

MAX_PX = 320


def save_jpg(arr, path):
    im = Image.fromarray(np.asarray(arr, np.uint8))
    if max(im.size) > MAX_PX:
        im.thumbnail((MAX_PX, MAX_PX))
    im.save(path, "JPEG", quality=80)


def crop_center(img, frac):
    h, w = img.shape[:2]
    ch, cw = int(h * frac), int(w * frac)
    return img[(h - ch) // 2:(h + ch) // 2, (w - cw) // 2:(w + cw) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/v3_mini")
    ap.add_argument("--out", default="gen_v3_report_assets")
    args = ap.parse_args()
    root, out = Path(args.data), Path(args.out)
    img = out / "images"
    img.mkdir(parents=True, exist_ok=True)
    rep = json.loads((root / "validation_report.json").read_text())
    config = json.loads((root / "config.json").read_text())
    shutil.copy(root / "validation_report.json", out / "validation_report.json")
    shutil.copy(root / "config.json", out / "config.json")
    m = pd.read_csv(root / "manifest.csv")
    ins = pd.read_csv(root / "instruction_rows.csv")
    pm = pd.read_csv(root / "plan_manifest.csv")
    pins = pd.read_csv(root / "plan_instruction_rows.csv")
    scenes = {p.stem: json.loads(p.read_text()) for p in (root / "scenes").glob("*.json") if not p.name.endswith(".rows.json")}

    # ------------------------------------------------------------------ stats
    size = subprocess.run(["du", "-sk", str(root)], capture_output=True, text=True).stdout.split()[0]
    per_scene = m.groupby("scene_id").size()
    first = m.drop_duplicates("scene_id")
    test_cams = first[first.split == "test"].camera
    lang = pd.concat([ins, pins]).drop_duplicates(["scene_id", "goal_id"])
    shapes = pd.Series([o["shape"] for s in scenes.values() for o in s["spec"]["objects"]])
    stats = {
        "dataset": str(root), "generator": config.get("generator"), "seed": config["seed"],
        "scenes": int(m.scene_id.nunique()), "single_push_rollouts": len(m), "two_push_plans": len(pm),
        "size_mb": round(int(size) / 1024, 1),
        "mean_seconds_per_rollout": rep["mean_seconds_per_rollout"], "mean_seconds_per_plan": rep.get("mean_seconds_per_plan"),
        "split_scenes": rep["split_scenes"],
        "single_candidates_per_scene": int(per_scene.iloc[0]),
        "single_candidates_by_type": m.candidate_type.value_counts().to_dict(),
        "single_candidates_by_type_per_scene": (m.candidate_type.str.replace("_blocked", "").value_counts() / len(per_scene)).round(2).to_dict(),
        "plans_per_scene": int(pm.groupby("scene_id").size().iloc[0]),
        "plans_by_type": pm.plan_type.value_counts().to_dict(),
        "own_goal_success_by_candidate_type": rep["own_goal_success_by_candidate_type"],
        "ideal_success_by_goal_type": rep["ideal_success_by_goal_type"],
        "positive_rate_by_goal_type": rep["positive_rate_by_goal_type"],
        "decisions": rep["decisions"], "decisions_by_goal_type": rep["decisions_by_goal_type"],
        "solvable_decision_share": rep["solvable_decision_share"],
        "solvable_share_by_goal_type": rep["solvable_share_by_goal_type"],
        "plan_success_by_type": rep.get("plan_success_by_type"),
        "plan_first_part_success_after_push1": rep.get("plan_first_part_success_after_push1"),
        "plan_decisions": rep.get("plan_decisions"), "plan_solvable_share": rep.get("plan_solvable_share"),
        "instruction_rows": len(ins) + len(pins), "distinct_instructions_per_decision": len(lang),
        "unique_instruction_texts": int(lang.instruction.nunique()),
        "relational_reference_share": rep["instructions_with_relational_reference"],
        "hindsight_captions": int(m.caption.notna().sum() + pm.caption.notna().sum() + pm.caption_mid.notna().sum()),
        "unique_captions": int(pd.concat([m.caption, pm.caption, pm.caption_mid]).nunique()),
        "caption_nothing_moved_share": rep["caption_nothing_moved_share"],
        "test_scenes_on_held_out_camera_share": round(float(test_cams.isin(config["held_cameras"]).mean()), 3),
        "camera_by_split": rep["camera_by_split"],
        "distractors_per_scene": rep["distractors_per_scene"],
        "objects_per_scene": rep["objects_per_scene"],
        "objects_by_shape": shapes.value_counts().to_dict(),
        "checks": {k: rep[k] for k in ("label_mismatches", "bad_plan_logs", "instruction_template_errors", "shape_word_leaks",
                                       "determinism_max_abs_diff", "plan_determinism_max_abs_diff",
                                       "arm_vs_no_arm_max_abs_diff", "physics_blowups", "rollouts_with_deep_penetration",
                                       "errors")},
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2, default=str))

    # ------------------------------------------------------------------ samples
    def pick(df, mask, n, seed=0):
        d = df[mask]
        return d.sample(n=min(n, len(d)), random_state=seed)

    rel = lang.instruction.str.contains("nearest|leftmost|rightmost")
    samples = {"instructions": [], "captions": []}
    for label, d in [("train", pick(lang, (lang.split == "train") & ~rel & (lang.goal_type != "and"), 4)),
                     ("train_relational", pick(lang, (lang.split != "test") & rel, 2)),
                     ("train_compound", pick(lang, (lang.split != "test") & (lang.goal_type == "and"), 1)),
                     ("test_held_out", pick(lang, (lang.split == "test") & (lang.goal_type != "and"), 3)),
                     ("test_held_out_relational", pick(lang, (lang.split == "test") & rel, 1)),
                     ("test_held_out_compound", pick(lang, (lang.split == "test") & (lang.goal_type == "and"), 1))]:
        for r in d.itertuples():
            samples["instructions"].append({"kind": label, "split": r.split, "goal_type": r.goal_type,
                                            "instruction": r.instruction, "template": r.template})
    for t in ["ideal", "over", "short", "miss", "wrong_object", "random_object", "random_line"]:
        for r in pick(m, m.candidate_type == t, 1, 1).itertuples():
            samples["captions"].append({"candidate_type": t, "own_goal_success": int(r.own_goal_success), "caption": r.caption})
    for r in pick(pm, pm.plan_type == "plan_closed", 1, 2).itertuples():
        samples["captions"].append({"candidate_type": "plan_closed (after push 1)", "caption": r.caption_mid})
        samples["captions"].append({"candidate_type": "plan_closed (final)", "own_goal_success": int(r.own_goal_success),
                                    "caption": r.caption})
    (out / "samples.json").write_text(json.dumps(samples, indent=2))

    # ------------------------------------------------------------------ images from the stored data
    def frames(kind, tid):
        z = np.load(root / kind / f"{tid}.npz")
        return dict(zip(z["keyframe_names"].tolist(), z["keyframes"]))

    # 4 scenes: table / lighting / distractor variety (initial frames)
    with_d = first[first.n_distractors > 0].scene_id.tolist()
    for k, sid in enumerate((with_d[:2] + first[first.n_distractors == 0].scene_id.tolist()[:2])):
        row = first[first.scene_id == sid].iloc[0]
        f = frames("trajs", f"{sid}_c00")["initial"]
        save_jpg(f, img / f"variety_{k + 1}_{sid}_{row.camera}_{row.n_distractors}distractors.jpg")
    # robot arm mid-push: after_push frame of an ideal push
    r = m[(m.candidate_type == "ideal") & (m.own_goal_success == 1)].iloc[0]
    save_jpg(frames("trajs", r.traj_id)["after_push"], img / f"arm_mid_push_{r.traj_id}.jpg")
    # one successful two-push plan, all 5 keyframes
    r = pm[(pm.plan_type == "plan_closed") & (pm.own_goal_success == 1)].iloc[0]
    for k, (name, f) in enumerate(frames("plans", r.traj_id).items()):
        save_jpg(f, img / f"plan_{r.traj_id}_{k + 1}_{name}.jpg")
    plan_text = scenes[r.scene_id]["instructions"][r.focus_goal]
    # ~8 candidates of one decision: final frames, with type and outcome in the name
    goal = ins[ins.label == 1].groupby(["scene_id", "goal_id"]).size().sort_values().index[-1]
    rows = ins[(ins.scene_id == goal[0]) & (ins.goal_id == goal[1])].merge(m[["traj_id", "candidate_type"]], on="traj_id")
    chosen = pd.concat([rows[rows.label == 1].head(3), rows[rows.label == 0].drop_duplicates("candidate_type").head(5)])
    decision_text = scenes[goal[0]]["instructions"][goal[1]]
    save_jpg(frames("trajs", rows.traj_id.iloc[0])["initial"], img / f"decision_{goal[0]}_0_initial.jpg")
    for k, rr in enumerate(chosen.itertuples()):
        save_jpg(frames("trajs", rr.traj_id)["final"],
                 img / f"decision_{goal[0]}_{k + 1}_{rr.candidate_type}_{'success' if rr.label else 'fail'}.jpg")

    # ------------------------------------------------------------------ freshly rendered images
    from gen_v3.generate import start_scene

    opts = {k: config[k] for k in ("cameras", "held_cameras", "held_share", "plans", "arm", "visual", "distractors")}
    sid = with_d[0]
    spec, _, _, _ = start_scene(config["seed"], scenes[sid]["spec"]["scene_idx"], opts)
    from gen_v3.generate import get_env
    env = get_env(opts["arm"])
    for cam in sx.CAMERAS:
        save_jpg(env.render(cam, MAX_PX), img / f"cameras_{sid}_{cam}.jpg")
    # new shapes, alone in the middle of the table, seen from the top-down view
    env_plain = SceneEnv(arm=False)
    for shape, color in (("hex", "blue"), ("bar", "orange")):
        obj = {"pool_name": f"{shape}_l_0", "shape": shape, "size": "l", "color": color, "xy": [0.0, 0.0], "yaw": 0.4,
               "radius": 0.05}
        env_plain.set_scene({"objects": [obj], "zones": []})
        save_jpg(crop_center(env_plain.render("cam3", 640), 0.4), img / f"shape_{shape}.jpg")

    notes = {"plan_image_instruction": plan_text, "decision_instruction": decision_text,
             "decision": {"scene_id": goal[0], "goal_id": goal[1], "solvable_candidates": int(rows.label.sum()),
                          "candidates": len(rows)}}
    (out / "image_captions.json").write_text(json.dumps(notes, indent=2))
    print("wrote", out, "with", len(list(img.glob("*.jpg"))), "images")


if __name__ == "__main__":
    main()
