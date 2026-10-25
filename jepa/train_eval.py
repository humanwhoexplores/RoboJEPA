"""Train the JEPA world model, the scorer and the direct classifier; compare action selection.

    python -m jepa.train_eval --data data/v1            # needs data/v1/features.npz (jepa.features)

Models (same conv trunk and width, so capacity is matched):
  predictor  P(z_initial, action)      -> z_final_hat   (DINOv2 patch grid, trained with MSE)
  scorer     S(z_final, instruction)   -> success logit (trained on ACTUAL final features)
  direct     D(z_initial, action, instr) -> success logit (no forecasting)

A decision = one scene + one instruction; the 8 candidate pushes of that scene are
ranked and the top one is "executed" (its stored label says whether it succeeded).
Methods compared on each decision:
  direct        D(z0, a, t)
  jepa          S(P(z0, a), t)        <- the proposal's method
  jepa_oracle   S(z_final, t)         scorer on the real outcome: separates scorer error from world-model error
  state_ref     gradient boosting on privileged cube/target coordinates + action (upper reference)
  random        expected success of a uniformly random candidate
  best_possible 1 if any candidate succeeds

Writes <data>/jepa_results.json and <data>/jepa_results.md.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

GRID = 16
TARGET_XY = {"left": (0.0, -0.15), "right": (0.0, 0.15)}  # envs.push_env.TARGETS


# ----------------------------------------------------------------------------- data
def action_vec(m):
    return np.stack([np.cos(m.theta), np.sin(m.theta), m.dist / 0.2, m.offset / 0.05], 1).astype(np.float32)


def build_decisions(m, rows):
    """One decision per (scene, kind, side); one instruction text shared by all 8 candidates."""
    first = rows.drop_duplicates(["scene_id", "kind", "target_side"])
    decisions = []
    for r in first.itertuples():
        cand = m.index[m.scene_id == r.scene_id].values
        decisions.append({"scene_id": r.scene_id, "split": r.split, "kind": r.kind, "side": r.target_side,
                          "instruction": r.instruction, "traj_idx": cand,
                          "labels": m.loc[cand, f"success_{r.target_side}"].values.astype(int),
                          "reserved": bool(m.loc[cand[0], "reserved_band"])})
    return decisions


# ----------------------------------------------------------------------------- models
class FiLMConvTrunk(nn.Module):
    def __init__(self, in_ch, d, cond_dim, blocks=3):
        super().__init__()
        self.inp = nn.Conv2d(in_ch + 2, d, 1)
        self.convs = nn.ModuleList([nn.Conv2d(d, d, 3, padding=1) for _ in range(blocks)])
        self.norms = nn.ModuleList([nn.GroupNorm(8, d) for _ in range(blocks)])
        self.film = nn.Linear(cond_dim, 2 * d * blocks)
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, GRID), torch.linspace(-1, 1, GRID), indexing="ij")
        self.register_buffer("coords", torch.stack([xs, ys])[None])

    def forward(self, z, cond):
        h = self.inp(torch.cat([z, self.coords.expand(len(z), -1, -1, -1)], 1))
        gammas_betas = self.film(cond).chunk(2 * len(self.convs), 1)
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            g, b = gammas_betas[2 * i], gammas_betas[2 * i + 1]
            h = h + F.gelu(norm(conv(h)) * (1 + g[..., None, None]) + b[..., None, None])
        return h


class Predictor(nn.Module):
    def __init__(self, zdim, d, adim):
        super().__init__()
        self.act = nn.Sequential(nn.Linear(adim, 64), nn.GELU(), nn.Linear(64, 64))
        self.trunk = FiLMConvTrunk(zdim, d, 64)
        self.out = nn.Conv2d(d, zdim, 1)

    def forward(self, z0, a):
        return z0 + self.out(self.trunk(z0, self.act(a)))  # residual: predict the change


class Classifier(nn.Module):
    """Scorer when adim == 0, direct classifier when adim > 0."""

    def __init__(self, zdim, d, tdim, adim=0):
        super().__init__()
        self.adim = adim
        self.act = nn.Sequential(nn.Linear(adim, 64), nn.GELU(), nn.Linear(64, 64)) if adim else None
        self.trunk = FiLMConvTrunk(zdim, d, tdim + (64 if adim else 0))
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, z, t, a=None):
        cond = torch.cat([t, self.act(a)], 1) if self.adim else t
        h = self.trunk(z, cond)
        return self.head(torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1)).squeeze(1)


# ----------------------------------------------------------------------------- training
def fit(model, make_batch, n_train, val_loss_fn, epochs, lr, bs, seed, label):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    best, best_state, t0 = float("inf"), None, time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_train, generator=g)
        for i in range(0, n_train, bs):
            loss = make_batch(perm[i:i + bs])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = val_loss_fn()
        if v < best:
            best, best_state = v, {k: x.detach().clone() for k, x in model.state_dict().items()}
    model.load_state_dict(best_state)
    print(f"  {label}: best val loss {best:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    return best


def run_seed(seed, D, args, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    z0, z1, A, T = D["z0"], D["z1"], D["A"], D["T"]
    tr_t, va_t = D["train_traj"], D["val_traj"]
    tr_r, va_r = D["train_rows"], D["val_rows"]
    rows = D["rows_arr"]  # (n_rows, 3): traj_idx, text_idx, label
    zdim = z0.shape[1]

    # Predictor ------------------------------------------------------------
    P = Predictor(zdim, args.width, A.shape[1]).to(device)
    def p_batch(idx):
        j = tr_t[idx]
        return F.mse_loss(P(z0[j], A[j]), z1[j])
    def p_val():
        return F.mse_loss(P(z0[va_t], A[va_t]), z1[va_t]).item()
    fit(P, p_batch, len(tr_t), p_val, args.epochs_pred, 1e-3, 32, seed, "predictor")

    # Scorer on actual final features ---------------------------------------
    S = Classifier(zdim, args.width, T.shape[1]).to(device)
    def s_logits(r, z):
        return S(z[rows[r, 0]], T[rows[r, 1]])
    def s_batch(idx):
        r = tr_r[idx]
        return F.binary_cross_entropy_with_logits(s_logits(r, z1), rows[r, 2].float())
    def s_val():
        return F.binary_cross_entropy_with_logits(s_logits(va_r, z1), rows[va_r, 2].float()).item()
    fit(S, s_batch, len(tr_r), s_val, args.epochs_cls, 1e-3, 64, seed, "scorer")

    # Direct classifier ----------------------------------------------------
    Dm = Classifier(zdim, args.width, T.shape[1], adim=A.shape[1]).to(device)
    def d_logits(r):
        j = rows[r, 0]
        return Dm(z0[j], T[rows[r, 1]], A[j])
    def d_batch(idx):
        r = tr_r[idx]
        return F.binary_cross_entropy_with_logits(d_logits(r), rows[r, 2].float())
    def d_val():
        return F.binary_cross_entropy_with_logits(d_logits(va_r), rows[va_r, 2].float()).item()
    fit(Dm, d_batch, len(tr_r), d_val, args.epochs_cls, 1e-3, 64, seed, "direct")

    return P, S, Dm


@torch.no_grad()
def score_decisions(decisions, P, S, Dm, D, device):
    z0, z1, A, T = D["z0"], D["z1"], D["A"], D["T"]
    out = {"direct": [], "jepa": [], "jepa_oracle": []}
    lat = {"direct": [], "jepa": []}
    for dec in decisions:
        j = torch.as_tensor(np.array(dec["traj_idx"]), device=device)
        t = T[dec["text_idx"]].expand(len(j), -1)
        t0 = time.perf_counter()
        out["direct"].append(torch.sigmoid(Dm(z0[j], t, A[j])).cpu().numpy())
        lat["direct"].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        out["jepa"].append(torch.sigmoid(S(P(z0[j], A[j]), t)).cpu().numpy())
        lat["jepa"].append(time.perf_counter() - t0)
        out["jepa_oracle"].append(torch.sigmoid(S(z1[j], t)).cpu().numpy())
    return out, {k: 1000 * float(np.median(v)) for k, v in lat.items()}


def selection_metrics(decisions, scores):
    labels = [d["labels"] for d in decisions]
    res = {"n_decisions": len(decisions)}
    for name, s in scores.items():
        picks = np.array([lab[np.argmax(sc)] for lab, sc in zip(labels, s)])
        res[name] = float(picks.mean())
        y, p = np.concatenate(labels), np.concatenate(s)
        res[f"{name}_auroc"] = float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else None
    res["random"] = float(np.mean([lab.mean() for lab in labels]))
    res["best_possible"] = float(np.mean([lab.max() for lab in labels]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/v1")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--epochs-pred", type=int, default=100)
    ap.add_argument("--epochs-cls", type=int, default=40)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    from jepa.features import pick_device

    device = pick_device(args.device)
    root = Path(args.data)
    m = pd.read_csv(root / "manifest.csv")
    rows = pd.read_csv(root / "instruction_rows.csv")
    f = np.load(root / "features.npz", allow_pickle=False)
    assert f["traj_ids"].tolist() == m.traj_id.tolist(), "features.npz is out of date; rerun jepa.features"
    print("device:", device, "| encoder:", str(f["visual"]), "| text:", str(f["text"]))

    # Normalize DINOv2 features per channel with train statistics, lay out as (N, C, 16, 16).
    train_mask = (m.split == "train").values
    z0_np, z1_np = f["z_initial"].astype(np.float32), f["z_final"].astype(np.float32)
    mu = np.concatenate([z0_np[train_mask], z1_np[train_mask]]).mean((0, 1))
    sd = np.concatenate([z0_np[train_mask], z1_np[train_mask]]).std((0, 1)) + 1e-6
    to_grid = lambda z: torch.from_numpy(((z - mu) / sd).transpose(0, 2, 1).reshape(len(z), -1, GRID, GRID)).to(device)
    instr_idx = {s: i for i, s in enumerate(f["instructions"])}
    traj_idx = {t: i for i, t in enumerate(m.traj_id)}
    D = {
        "z0": to_grid(z0_np), "z1": to_grid(z1_np),
        "A": torch.from_numpy(action_vec(m)).to(device),
        "T": torch.from_numpy(f["text_emb"]).to(device),
        "train_traj": torch.as_tensor(np.flatnonzero(train_mask), device=device),
        "val_traj": torch.as_tensor(np.flatnonzero((m.split == "val").values), device=device),
    }
    rows_arr = np.stack([rows.traj_id.map(traj_idx), rows.instruction.map(instr_idx), rows.label], 1)
    D["rows_arr"] = torch.as_tensor(rows_arr, device=device)
    D["train_rows"] = torch.as_tensor(np.flatnonzero(rows.split == "train"), device=device)
    D["val_rows"] = torch.as_tensor(np.flatnonzero(rows.split == "val"), device=device)

    decisions = build_decisions(m, rows)
    for d in decisions:
        d["text_idx"] = instr_idx[d["instruction"]]

    # Privileged state reference: cube start + requested target position + action -> success.
    def state_X(sub):
        tm = m.loc[sub.traj_id.map(traj_idx).values]
        txy = np.array([TARGET_XY[s] for s in sub.target_side])
        return np.column_stack([tm.cube_x0, tm.cube_y0, txy, action_vec(tm)])
    tr = rows[rows.split == "train"]
    gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(state_X(tr), tr.label)

    # Predictor quality baselines: how far is "nothing changes" (z0) from the real outcome?
    results = {"config": vars(args) | {"encoder": str(f["visual"]), "text_encoder": str(f["text"])}, "seeds": []}
    for seed in range(args.seeds):
        print(f"seed {seed}", flush=True)
        P, S, Dm = run_seed(seed, D, args, device)
        P.eval(), S.eval(), Dm.eval()
        seed_res = {}
        for split in ("val", "test"):
            decs = [d for d in decisions if d["split"] == split]
            if not decs:
                continue
            scores, lat = score_decisions(decs, P, S, Dm, D, device)
            scores["state_ref"] = []
            for d in decs:
                sub = pd.DataFrame({"traj_id": m.traj_id.values[d["traj_idx"]], "target_side": d["side"]})
                scores["state_ref"].append(gbm.predict_proba(state_X(sub))[:, 1])
            split_res = selection_metrics(decs, scores)
            for kind in ("position", "color"):
                k = [i for i, d in enumerate(decs) if d["kind"] == kind]
                split_res[f"by_kind_{kind}"] = selection_metrics([decs[i] for i in k], {n: [s[i] for i in k] for n, s in scores.items()})
            if split == "test":
                for flag, name in ((True, "reserved_band"), (False, "regular_band")):
                    k = [i for i, d in enumerate(decs) if d["reserved"] == flag]
                    if k:
                        split_res[f"by_{name}"] = selection_metrics([decs[i] for i in k], {n: [s[i] for i in k] for n, s in scores.items()})
            split_res["latency_ms_per_decision"] = lat
            ti = torch.as_tensor(np.flatnonzero((m.split == split).values), device=device)
            with torch.no_grad():
                pred = P(D["z0"][ti], D["A"][ti])
                split_res["predictor_mse"] = F.mse_loss(pred, D["z1"][ti]).item()
                split_res["copy_initial_mse"] = F.mse_loss(D["z0"][ti], D["z1"][ti]).item()
                split_res["predictor_cos"] = F.cosine_similarity(pred, D["z1"][ti], dim=1).mean().item()
                split_res["copy_initial_cos"] = F.cosine_similarity(D["z0"][ti], D["z1"][ti], dim=1).mean().item()
            seed_res[split] = split_res
        results["seeds"].append(seed_res)

    # Mean / std over seeds for the headline numbers.
    methods = ["direct", "jepa", "jepa_oracle", "state_ref", "random", "best_possible"]
    summary = {}
    for split in ("val", "test"):
        if split not in results["seeds"][0]:
            continue
        get = lambda key, sub=None: np.array([(s[split][sub] if sub else s[split])[key] for s in results["seeds"]], dtype=float)
        summary[split] = {"n_decisions": results["seeds"][0][split]["n_decisions"]}
        for meth in methods:
            v = get(meth)
            summary[split][meth] = [round(v.mean(), 3), round(v.std(), 3)]
            if meth in ("direct", "jepa", "jepa_oracle", "state_ref"):
                a = get(f"{meth}_auroc")
                summary[split][f"{meth}_auroc"] = [round(np.nanmean(a), 3), round(np.nanstd(a), 3)]
            for kind in ("position", "color"):
                summary[split].setdefault(f"{kind}", {})[meth] = round(get(meth, f"by_kind_{kind}").mean(), 3)
        for key in ("predictor_mse", "copy_initial_mse", "predictor_cos", "copy_initial_cos"):
            summary[split][key] = round(get(key).mean(), 4)
        summary[split]["latency_ms"] = {k: round(np.mean([s[split]["latency_ms_per_decision"][k] for s in results["seeds"]]), 2) for k in ("direct", "jepa")}
    results["summary"] = summary
    (root / "jepa_results.json").write_text(json.dumps(results, indent=2, default=float))

    lines = [f"# JEPA vs direct: action selection ({args.seeds} seeds, mean ± std)", "",
             f"Encoder: {results['config']['encoder']}, text: {results['config']['text_encoder']}", ""]
    for split, s in summary.items():
        lines += [f"## {split} ({s['n_decisions']} decisions, 8 candidates each)", "",
                  "| method | selection success | AUROC | position | color |", "|---|---|---|---|---|"]
        for meth in methods:
            auc = s.get(f"{meth}_auroc")
            lines.append(f"| {meth} | {s[meth][0]:.3f} ± {s[meth][1]:.3f} | "
                         f"{'' if auc is None else f'{auc[0]:.3f}'} | {s['position'][meth]:.3f} | {s['color'][meth]:.3f} |")
        lines += ["", f"Predictor: MSE {s['predictor_mse']} vs copy-initial {s['copy_initial_mse']}; "
                      f"cosine {s['predictor_cos']} vs copy-initial {s['copy_initial_cos']}",
                  f"Latency per decision (ms): {s['latency_ms']}", ""]
    (root / "jepa_results.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
