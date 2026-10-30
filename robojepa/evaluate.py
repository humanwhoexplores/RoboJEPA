"""Offline action-ranking evaluation.

For every test scene and every instruction, each method scores the scene's candidate
sequences; we execute (look up) the argmax and check whether it satisfied the instruction.

    python -m robojepa.evaluate --predictor runs/predictor --scorer runs/scorer --direct runs/direct --out results/main
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from robojepa.data import TARGETS, Split, Texts
from robojepa.models import DirectClassifier, Predictor, Scorer
from robojepa.train import auc

METHODS = ["random", "direct", "jepa", "oracle"]


def load(cls, run, dev):
    m = cls().to(dev).eval()
    m.load_state_dict(torch.load(os.path.join(run, "model.pt"), map_location=dev))
    return m


@torch.no_grad()
def batched(fn, n, bs=512):
    return torch.cat([fn(slice(i, min(i + bs, n))) for i in range(0, n, bs)])


@torch.no_grad()
def evaluate_split(split, texts, pred, scorer, direct, rng):
    s, dev = split, split.device
    n = len(s.n_push)
    n_push_t = torch.from_numpy(s.n_push).to(dev)
    z0 = lambda sl: s.z[s.start_row[sl]].float()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        z_pred = batched(lambda sl: pred.rollout(z0(sl), s.seqs[sl], n_push_t[sl]).float(), n).half()
    z_true = s.z[s.term_row]

    # World-model error by sequence length (and the trivial "nothing changes" baseline).
    wm = {}
    for k in (1, 2):
        m = torch.from_numpy(s.n_push == k).to(dev)
        if m.any():
            err = ((z_pred[m].float() - z_true[m].float()) ** 2).mean((1, 2))
            copy = ((s.z[s.start_row[m.cpu().numpy()]].float() - z_true[m].float()) ** 2).mean((1, 2))
            wm[f"{k}push"] = {"pred_mse": err.mean().item(), "copy_mse": copy.mean().item()}

    rows = []  # one per (scene, instruction, candidate-subset)
    cand_scores = {m: [] for m in METHODS[1:]}
    cand_labels = []
    for ti in range(len(texts.text)):
        t = TARGETS.index(texts.target[ti])
        e = texts.emb[ti:ti + 1]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            sc = {
                "oracle": batched(lambda sl: scorer(z_true[sl].float(), e.expand(sl.stop - sl.start, -1)), n),
                "jepa": batched(lambda sl: scorer(z_pred[sl].float(), e.expand(sl.stop - sl.start, -1)), n),
                "direct": batched(lambda sl: direct(z0(sl), s.seqs[sl], n_push_t[sl], e.expand(sl.stop - sl.start, -1)), n),
            }
        sc = {k: v.float().cpu().numpy() for k, v in sc.items()}
        sc["random"] = rng.random(n)
        y = s.Y[:, t]
        if texts.split[ti] == "train":  # candidate-level AUC on seen templates
            for k in cand_scores:
                cand_scores[k].append(sc[k])
            cand_labels.append(y)
        for scene in np.unique(s.scene):
            idx = np.where(s.scene == scene)[0]
            for subset, mask in (("all", np.ones(len(idx), bool)), ("1push", s.n_push[idx] == 1), ("2push", s.n_push[idx] == 2)):
                j = idx[mask]
                if len(j) == 0:
                    continue
                row = dict(scene=int(scene), instr=ti, kind=str(texts.split[ti]), target=TARGETS[t],
                           tkind="pos" if t < 2 else "color", subset=subset, solvable=bool(y[j].any()))
                for m in METHODS:
                    row[m] = float(y[j[np.argmax(sc[m][j])]])
                rows.append(row)
    aucs = {k: auc(np.concatenate(v), np.concatenate(cand_labels)) for k, v in cand_scores.items()}
    return rows, wm, aucs


def summarize(rows, solvable_only):
    out = {}
    groups = {
        "all": lambda r: r["subset"] == "all",
        "seen_templates": lambda r: r["subset"] == "all" and r["kind"] == "train",
        "heldout_paraphrases": lambda r: r["subset"] == "all" and r["kind"] == "heldout",
        "position_instr": lambda r: r["subset"] == "all" and r["tkind"] == "pos",
        "color_instr": lambda r: r["subset"] == "all" and r["tkind"] == "color",
        "1push_candidates": lambda r: r["subset"] == "1push",
        "2push_candidates": lambda r: r["subset"] == "2push",
    }
    for g, f in groups.items():
        sel = [r for r in rows if f(r) and (r["solvable"] or not solvable_only)]
        if sel:
            out[g] = {m: float(np.mean([r[m] for r in sel])) for m in METHODS}
            out[g]["n"] = len(sel)
            out[g]["solvable_frac"] = float(np.mean([r["solvable"] for r in rows if f(r)]))
    return out


@torch.no_grad()
def latency(pred, scorer, direct, split, texts, reps=50):
    s, dev = split, split.device
    idx = np.where(s.scene == s.scene[0])[0]
    z = s.z[s.start_row[idx]].float()
    seq, n_push = s.seqs[idx], torch.from_numpy(s.n_push[idx]).to(dev)
    e = texts.emb[:1].expand(len(idx), -1)
    fns = {"direct": lambda: direct(z, seq, n_push, e),
           "jepa": lambda: scorer(pred.rollout(z, seq, n_push), e)}
    res = {}
    for k, f in fns.items():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(5):
                f()
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(reps):
                f()
            torch.cuda.synchronize()
        res[k] = (time.perf_counter() - t) / reps * 1000
    res["n_candidates"] = len(idx)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features")
    ap.add_argument("--predictor", required=True)
    ap.add_argument("--scorer", required=True)
    ap.add_argument("--direct", required=True)
    ap.add_argument("--splits", nargs="+", default=["test_id", "test_ood"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    texts = Texts(a.features, dev)
    pred, scorer, direct = load(Predictor, a.predictor, dev), load(Scorer, a.scorer, dev), load(DirectClassifier, a.direct, dev)
    results = {"args": vars(a)}
    for sp in a.splits:
        split = Split(a.features, sp, dev)
        rows, wm, aucs = evaluate_split(split, texts, pred, scorer, direct, np.random.default_rng(0))
        results[sp] = {"selection_success": summarize(rows, False), "selection_success_solvable": summarize(rows, True),
                       "candidate_auc": aucs, "world_model_mse": wm, "latency_ms": latency(pred, scorer, direct, split, texts)}
        del split
        torch.cuda.empty_cache()
    os.makedirs(a.out, exist_ok=True)
    json.dump(results, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    print(report(results))
    open(os.path.join(a.out, "results.md"), "w").write(report(results))


def report(res):
    lines = []
    for sp in [k for k in res if k.startswith("test")]:
        r = res[sp]
        lines.append(f"\n## {sp}\n")
        lines.append("Selection success (solvable scene/instruction pairs; random/direct/jepa/oracle):\n")
        lines.append("| group | n | solvable | random | direct | jepa | oracle |\n|---|---|---|---|---|---|---|")
        for g, v in r["selection_success_solvable"].items():
            lines.append(f"| {g} | {v['n']} | {v['solvable_frac']:.2f} | " + " | ".join(f"{v[m]:.3f}" for m in METHODS) + " |")
        lines.append("\nCandidate-level AUC: " + ", ".join(f"{k} {v:.3f}" for k, v in r["candidate_auc"].items()))
        lines.append("World-model feature MSE: " + ", ".join(
            f"{k}: pred {v['pred_mse']:.3f} vs copy {v['copy_mse']:.3f}" for k, v in r["world_model_mse"].items()))
        lat = r["latency_ms"]
        lines.append(f"Decision latency ({lat['n_candidates']} candidates): direct {lat['direct']:.1f} ms, jepa {lat['jepa']:.1f} ms")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
