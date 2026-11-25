"""Validate a v6 dataset and write validation_report.json + contact_sheet.png.

    python -m gen_v6.validate --data data/v6_mini

Fails (exit code 1) on: label mismatches (any of the 3 labels; first-step and step-2 rollouts; npz vs label
matrix vs CSVs), a default positive whose agent moved less than MOVE_MIN, split leakage (templates and words,
cameras, held-out object combos, the held-out region, extreme hidden properties, the test_family layout),
non-determinism, the sampler not reproducing the stored candidates from the scene alone, NaNs or blow-ups,
held-out action-goal flags that disagree with gen_v6.splits.

Reports (not fatal): positives per goal type, decision source, action and family; solvable shares; the random
baseline; tracking errors; held-out action-goal positives; captions.
"""
import argparse
import itertools
import json
import re
import tempfile
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from gen_v3 import scene_spec as v3s
from gen_v6 import catalog as cat
from gen_v6 import goals as G
from gen_v6 import language as L
from gen_v6 import splits
from gen_v6.labels import LABELS, MOVE_MIN, label_sets

MAX_SPEED = 4.0  # m/s for an object on the table (a dropped ball can reach 2 m/s)
TRACK_BAD_MM = 30


def last_state(z):
    return {"pos": z["obj_pos"][-1].astype(float), "quat": z["obj_quat"][-1].astype(float),
            "pair": z["pair"][-1].astype(bool), "table": z["table"][-1].astype(bool),
            "fxc": z["fx_contact"][-1].astype(bool), "fxs": z["fx_state"][-1].astype(float)}


def as_state(d):
    return {k: np.asarray(v, bool if k in ("pair", "table", "fxc") else float) for k, v in d.items()}


def log_of(z):
    return {k: z[k] for k in ("obj_pos", "obj_quat", "pair", "table", "fx_contact", "fx_state", "phase", "contact")}


def physics(z, tid, acc):
    pos = z["obj_pos"].astype(float)
    if not np.isfinite(pos).all():
        acc["blowup"].append((tid, "nan"))
        return
    vel = np.linalg.norm(np.diff(pos, axis=0), axis=-1) * 20
    on = G.on_table(pos[1:]) & G.on_table(pos[:-1])
    if (vel[on] > MAX_SPEED).any():
        acc["blowup"].append((tid, float(vel[on].max())))


def leakage(scenes, ins, errors):
    train_t, test_t = L.all_instruction_words()
    test_words = set(itertools.chain(*L.TEST_KIND_WORDS.values(), *L.TEST_FIXTURE_WORDS.values()))
    for r in ins.itertuples():
        test = r.lang_split == "test"
        if (r.template in test_t) != test or (r.template in train_t) == test:
            errors.append(f"template leak: {r.traj_id} {r.template!r} in a {r.lang_split} scene")
            return
        if not test and any(re.search(rf"\b{re.escape(w)}\b", r.instruction) for w in test_words):
            errors.append(f"test word in a training instruction: {r.traj_id} {r.instruction!r}")
            return
    for sid, d in scenes.items():
        s, split = d["spec"], d["spec"]["v6_split"]
        if (s["camera"] in v3s.HELD_CAMERAS) != (split == "test_cam"):
            errors.append(f"camera leak: {sid} {s['camera']} in {split}")
        held = [(o["kind"], o.get("color")) in splits.HELD_COMBOS for o in s["objects"]]
        if any(held) != (split == "test_obj"):
            errors.append(f"object-combo leak: {sid} {split}")
        region = [np.linalg.norm(np.asarray(o["xy"]) - splits.HELD_REGION[0]) < splits.HELD_REGION[1]
                  for o in s["objects"]]
        if any(region) and split != "test_pos":
            errors.append(f"held-out region leak: {sid} {split}")
        extreme = [cat.extreme(o["props"]) for o in s["objects"]]
        if any(extreme) and split != "test_phys":
            errors.append(f"extreme hidden property outside test_phys: {sid} {split}")
        kinds = {f["kind"] for f in s["fixtures"]}
        if "ramp" in kinds and kinds & {"tray", "bin", "bowl"} and split != "test_family":
            errors.append(f"test_family layout outside test_family: {sid} {split}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/v6_mini")
    ap.add_argument("--determinism", type=int, default=4, help="rollouts to re-run")
    ap.add_argument("--resample", type=int, default=3, help="scenes whose candidates are re-drawn")
    ap.add_argument("--sheet", type=int, default=12)
    args = ap.parse_args()
    root = Path(args.data)
    m = pd.read_csv(root / "manifest.csv")
    ins = pd.read_csv(root / "instruction_rows.csv")
    m2 = pd.read_csv(root / "step2_manifest.csv") if (root / "step2_manifest.csv").exists() else pd.DataFrame()
    ins2 = (pd.read_csv(root / "step2_instruction_rows.csv") if (root / "step2_instruction_rows.csv").exists()
            else pd.DataFrame())
    config = json.loads((root / "config.json").read_text())
    scenes = {p.stem: json.loads(p.read_text()) for p in (root / "scenes").glob("*.json")
              if not p.name.endswith(".rows.json")}
    errors = []
    report = {"n_scenes": len(scenes), "n_rollouts": len(m), "n_step2_rollouts": len(m2)}
    acc = {"mismatch": [], "blowup": [], "agent": []}

    # 1. Labels, recomputed from each stored log; compared with the npz, the scene's label matrix and the CSVs.
    ins_by = {k: g for k, g in ins.groupby("traj_id")}
    finals = {}
    for sid, d in sorted(scenes.items()):
        spec, goals = d["spec"], d["goals"]
        S0 = as_state(d["init_state"])
        lab = np.load(root / "scenes" / f"{sid}_labels.npz")
        gix = {g: i for i, g in enumerate(lab["goal_ids"])}
        actions = []
        for k, tid in enumerate(lab["traj_ids"]):
            z = np.load(root / "trajs" / f"{tid}.npz")
            meta = json.loads(str(z["meta"]))
            actions.append(meta["candidate"]["action"])
            physics(z, tid, acc)
            rec = label_sets(goals, log_of(z), spec, S0)
            finals[tid] = z
            for name, key in zip(LABELS, ("goal_labels", "goal_labels_state", "goal_labels_strict")):
                if not (np.array_equal(rec[name], z[key]) and np.array_equal(rec[name], lab[name][k])):
                    acc["mismatch"].append((tid, name))
            moved = np.linalg.norm(z["obj_pos"][-1, :, :2] - S0["pos"][:, :2], axis=1)
            for gi in np.flatnonzero(rec["default"]):
                if any(moved[a] < MOVE_MIN for a in G.agents(goals[gi])):
                    acc["agent"].append((tid, goals[gi]["goal_id"]))
            for r in ins_by.get(tid, pd.DataFrame()).itertuples():
                i = gix[r.goal_id]
                if (r.label, r.label_state, r.label_strict) != tuple(int(rec[n][i]) for n in LABELS):
                    acc["mismatch"].append((tid, "csv:" + r.goal_id))
                if r.act_heldout != int(splits.act_heldout(r.action, r.goal_type)):
                    errors.append(f"act_heldout flag wrong: {tid} {r.goal_id}")
        mask = splits.act_mask(actions, [g["type"] for g in goals])
        if not np.array_equal(mask, lab["act_heldout"]):
            errors.append(f"act_heldout mask wrong: {sid}")

    for r in m2.itertuples() if len(m2) else []:
        d = scenes[r.scene_id]
        z = np.load(root / "steps" / f"{r.traj_id}.npz")
        meta = json.loads(str(z["meta"]))
        physics(z, r.traj_id, acc)
        start = last_state(finals[r.step1_traj_id])
        multi = {g["goal_id"]: g for g in d["multi_step"]}[r.decision]
        target = multi["parts"][0] if multi["type"] == "probe" else multi
        goals2 = d["goals"] + ([target] if target["goal_id"] not in {g["goal_id"] for g in d["goals"]} else [])
        rec = label_sets(goals2, log_of(z), d["spec"], start, as_state(d["init_state"]))
        for name, key in zip(LABELS, ("goal_labels", "goal_labels_state", "goal_labels_strict")):
            if not np.array_equal(rec[name], z[key]):
                acc["mismatch"].append((r.traj_id, name))
        if meta["step1_traj_id"] != r.step1_traj_id:
            errors.append(f"step-1 link wrong: {r.traj_id}")
    if len(ins2):
        for r in ins2.itertuples():
            z = np.load(root / "steps" / f"{r.traj_id}.npz")
            d = scenes[r.scene_id]
            ids = [g["goal_id"] for g in d["goals"]]
            i = ids.index(r.goal_id) if r.goal_id in ids else len(ids)
            if (r.label, r.label_state, r.label_strict) != tuple(int(z[k][i]) for k in
                                                                 ("goal_labels", "goal_labels_state", "goal_labels_strict")):
                acc["mismatch"].append((r.traj_id, "step2 csv"))
    report["label_mismatches"] = len(acc["mismatch"])
    report["agent_violations"] = len(acc["agent"])
    report["blowups"] = len(acc["blowup"])
    if acc["mismatch"]:
        errors.append(f"{len(acc['mismatch'])} label mismatches, e.g. {acc['mismatch'][:3]}")
    if acc["agent"]:
        errors.append(f"{len(acc['agent'])} default positives whose agent did not move, e.g. {acc['agent'][:3]}")
    if acc["blowup"]:
        errors.append(f"{len(acc['blowup'])} physics blow-ups, e.g. {acc['blowup'][:3]}")

    # 2. Split leakage.
    leakage(scenes, pd.concat([ins, ins2]) if len(ins2) else ins, errors)

    # 3. Determinism and the sampler, from the scene alone.
    from gen_v6.generate import build_scene, get_env, run_rollout
    from gen_v6.goals import initial_state
    from gen_v6.sampler import sample
    env = get_env()
    rng = np.random.default_rng(0)
    det = []
    sids = sorted(scenes)
    for sid in [sids[k] for k in rng.permutation(len(sids))[:max(args.determinism, args.resample)]]:
        d = scenes[sid]
        spec = build_scene(env, d["spec"]["seed"], d["spec"]["scene_idx"], config["stage2"])
        if spec["retry"] != d["spec"]["retry"]:
            errors.append(f"scene rebuild differs: {sid}")
            continue
        state0 = env.get_state()
        S0 = initial_state(env)
        if np.abs(S0["pos"] - np.asarray(d["init_state"]["pos"])).max() > 1e-6:
            errors.append(f"start state differs: {sid}")
        stored = [json.loads(str(np.load(root / "trajs" / f"{t}.npz")["meta"]))["candidate"]
                  for t in m[m.scene_id == sid].traj_id]
        if len(det) < args.resample:
            redrawn, _ = sample(env, spec, np.random.default_rng([spec["seed"], spec["scene_idx"], 21]),
                                n_total=config["n_rollouts"] + 8, n_lines=config["n_lines"])
            env.set_state(state0)
            it = iter(json.loads(json.dumps(c, default=float)) for c in redrawn)
            if not all(any(c == s for c in it) for s in stored):
                errors.append(f"sampler does not reproduce the stored candidates: {sid}")
        k = int(rng.integers(len(stored)))
        tid = f"{sid}_c{k:02d}"
        with tempfile.TemporaryDirectory() as tmp:
            res = run_rollout(env, spec, state0, stored[k], d["goals"], S0, Path(tmp) / "det.mp4")
        z = finals[tid]
        diff = float(np.abs(res[0]["obj_pos"] - z["obj_pos"]).max()) if res and len(res[0]["obj_pos"]) == len(z["obj_pos"]) else 1.0
        det.append(diff)
    report["determinism_max_diff_m"] = max(det) if det else None
    if det and max(det) > 1e-4:
        errors.append(f"non-deterministic rollouts: max diff {max(det):.2e} m")

    # 4. Reports.
    goals_all = pd.read_csv(root / "all_goals.csv")
    report["rollouts_per_action"] = m.action.value_counts().to_dict()
    report["share_touched"] = round(float(m.touched.mean()), 3)
    report["share_held"] = round(float(m.held.mean()), 3)
    report["track_err_over_30mm"] = int((m.track_err_mm > TRACK_BAD_MM).sum())
    report["scenes_per_split"] = Counter(d["spec"]["v6_split"] for d in scenes.values())
    report["scenes_per_family"] = Counter(d["spec"]["family"] for d in scenes.values())
    report["scenes_redrawn_to_settle"] = sum(d["spec"]["retry"] > 0 for d in scenes.values())
    report["rollouts_replaced_unstable"] = sum(d["n_unstable"] for d in scenes.values())
    by_type = {}
    for t, g in ins.groupby("goal_type"):
        solv = g.groupby(["scene_id", "goal_id"]).label.max()
        by_type[t] = {"decisions": int(len(solv)), "solvable": round(float(solv.mean()), 3),
                      "random": round(float(g.label.mean()), 3)}
    report["decisions_by_goal_type"] = by_type
    src = {}
    for s, g in ins.groupby("source"):
        solv = g.groupby(["scene_id", "goal_id"]).label.max()
        src[s] = {"decisions": int(len(solv)), "solvable": round(float(solv.mean()), 3),
                  "random": round(float(g.label.mean()), 3)}
    report["decisions_by_source"] = src
    pos_by_action = defaultdict(int)
    for sid in scenes:
        lab = np.load(root / "scenes" / f"{sid}_labels.npz")
        acts = m.set_index("traj_id").loc[lab["traj_ids"], "action"].to_numpy()
        for a, row in zip(acts, lab["default"]):
            pos_by_action[a] += int(row.sum())
    report["goal_positives_per_rollout_by_action"] = {
        a: round(pos_by_action[a] / max(1, int((m.action == a).sum())), 2) for a in sorted(pos_by_action)}
    report["goal_types_with_any_positive"] = {
        t: int((g.n_pos > 0).sum()) for t, g in goals_all.groupby("goal_type")}
    report["act_heldout_positives"] = int(goals_all.n_pos_act_heldout.sum())
    if len(ins2):
        s2 = {}
        for s, g in ins2.groupby("source"):
            solv = g.groupby(["scene_id", "decision"]).label.max()
            s2[s] = {"decisions": int(len(solv)), "solvable_at_step2": round(float(solv.mean()), 3),
                     "random": round(float(g.label.mean()), 3)}
        report["step2_by_source"] = s2
    report["caption_examples"] = list(m.caption.sample(min(8, len(m)), random_state=0))
    report["instruction_examples"] = list(ins.drop_duplicates("goal_id").instruction.sample(
        min(12, ins.goal_id.nunique()), random_state=0))

    # 5. Contact sheet: start and final frames of a few rollouts.
    tiles = []
    for tid in m.sample(min(args.sheet, len(m)), random_state=1).traj_id:
        z = np.load(root / "trajs" / f"{tid}.npz")
        sid = "_".join(tid.split("_")[:2])
        start = np.load(root / "scenes" / f"{sid}_initial.npz")["initial"]
        names = list(z["keyframe_names"])
        final = z["keyframes"][names.index("final")]
        pair = Image.fromarray(np.concatenate([start, final], 1))
        ImageDraw.Draw(pair).text((4, 4), f"{tid} {m.set_index('traj_id').loc[tid, 'action']}", fill=(255, 255, 255))
        tiles.append(np.asarray(pair))
    if tiles:
        rows = [np.concatenate(tiles[k:k + 2], 1) for k in range(0, len(tiles) - len(tiles) % 2, 2)]
        Image.fromarray(np.concatenate(rows, 0)).save(root / "contact_sheet.png")

    report["errors"] = errors
    report["passed"] = not errors
    (root / "validation_report.json").write_text(json.dumps(report, indent=2, default=lambda o: o if not isinstance(o, np.integer) else int(o)))
    print(json.dumps(report, indent=2, default=int))
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
