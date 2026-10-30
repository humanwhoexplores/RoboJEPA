"""Train one model on cached features.

    python -m robojepa.train --model predictor --out runs/predictor
    python -m robojepa.train --model scorer    --out runs/scorer
    python -m robojepa.train --model direct    --out runs/direct --scenes 100

Every model trains for the same number of optimizer steps regardless of --scenes, so the
data-efficiency sweep varies data, not training length. The kept checkpoint is the one with the best
val loss (predictor) or val AUC (scorer, direct classifier; they are used to rank candidates).
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from robojepa.data import TARGETS, Split, Texts
from robojepa.models import DirectClassifier, Predictor, Scorer


def batches(n, bs, rng):
    while True:
        perm = rng.permutation(n)
        for i in range(0, n - bs + 1 if n >= bs else 1, bs):
            yield perm[i:i + bs]


def make_model(kind):
    return {"predictor": Predictor, "scorer": Scorer, "direct": DirectClassifier}[kind]()


def auc(scores, labels):
    """ROC AUC via rank statistic."""
    labels = np.asarray(labels).astype(bool)
    if labels.all() or not labels.any():
        return float("nan")
    r = np.argsort(np.argsort(scores)) + 1
    return (r[labels].sum() - labels.sum() * (labels.sum() + 1) / 2) / (labels.sum() * (~labels).sum())


class Task:
    """Wraps per-model data, loss and validation metric."""

    def __init__(self, kind, split, texts, rng):
        self.kind, self.s, self.texts, self.rng = kind, split, texts, rng
        if kind == "predictor":
            self.src, self.act, self.dst = split.transitions()
        elif kind == "scorer":
            self.rows, self.Y = split.scorer_frames()
        else:
            self.Y = split.Y
        self.n = len(self.src) if kind == "predictor" else len(self.Y)
        if kind != "predictor":
            p = self.Y.mean()
            self.pos_weight = torch.tensor((1 - p) / p, device=split.device)

    def loss(self, model, idx, text_split="train", reduce=True):
        s = self.s
        if self.kind == "predictor":
            pred = model(s.z[self.src[idx]].float(), self.act[idx])
            err = F.mse_loss(pred, s.z[self.dst[idx]].float(), reduction="none").mean((1, 2))
            return err.mean() if reduce else err
        # Every frame/candidate is paired with all targets, so the same outcome is a positive for one
        # instruction and a negative for another; this forces the model to read the instruction.
        t = np.tile(np.arange(len(TARGETS)), len(idx))
        idx = np.repeat(idx, len(TARGETS))
        y = torch.from_numpy(self.Y[idx, t]).to(s.device)
        text = self.texts.sample(t, text_split, self.rng)
        if self.kind == "scorer":
            logit = model(s.z[self.rows[idx]].float(), text)
        else:
            logit = model(s.z[s.start_row[idx]].float(), s.seqs[idx], torch.from_numpy(s.n_push[idx]).to(s.device), text)
        l = F.binary_cross_entropy_with_logits(logit, y, pos_weight=self.pos_weight, reduction="none")
        return (l.mean() if reduce else l), logit, y

    @torch.no_grad()
    def validate(self, model, bs=256):
        model.eval()
        losses, logits, ys = [], [], []
        for i in range(0, self.n, bs):
            idx = np.arange(i, min(i + bs, self.n))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = self.loss(model, idx, reduce=False)
            if self.kind == "predictor":
                losses.append(out.float())
            else:
                losses.append(out[0].float()), logits.append(out[1].float()), ys.append(out[2])
        model.train()
        res = {"loss": torch.cat(losses).mean().item()}
        if logits:
            res["auc"] = auc(torch.cat(logits).cpu().numpy(), torch.cat(ys).cpu().numpy())
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["predictor", "scorer", "direct"])
    ap.add_argument("--features", default="features")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", type=int, default=None, help="use only the first N training scenes")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--eval_every", type=int, default=125)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = "cuda"
    texts = Texts(a.features, dev)
    train = Task(a.model, Split(a.features, "train", dev, a.scenes), texts, rng)
    val = Task(a.model, Split(a.features, "val", dev), texts, np.random.default_rng(123))
    model = make_model(a.model).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    os.makedirs(a.out, exist_ok=True)
    print(f"{a.model}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, "
          f"{train.n} train examples ({train.s.n_scenes} scenes), {val.n} val, {a.steps} steps")

    best, log, t0, tot, cnt = float("inf"), [], time.time(), 0.0, 0
    for step, idx in enumerate(batches(train.n, a.bs, rng), 1):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = train.loss(model, idx)
        loss = out if a.model == "predictor" else out[0]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(), sched.step()
        tot, cnt = tot + loss.item(), cnt + 1
        if step % a.eval_every == 0 or step == a.steps:
            v = val.validate(model)
            log.append({"step": step, "train_loss": tot / cnt, **{f"val_{k}": x for k, x in v.items()}})
            print(f"step {step:5d}  train {tot / cnt:.4f}  " + "  ".join(f"val_{k} {x:.4f}" for k, x in v.items())
                  + f"  {time.time() - t0:.0f}s", flush=True)
            tot, cnt = 0.0, 0
            crit = v["loss"] if a.model == "predictor" else -v["auc"]
            if crit < best:
                best = crit
                torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))
        if step == a.steps:
            break
    key = (lambda r: -r["val_loss"]) if a.model == "predictor" else (lambda r: r["val_auc"])
    summary = {"args": vars(a), "log": log, "best_val": abs(best), "best_step": max(log, key=key)["step"]}
    json.dump(summary, open(os.path.join(a.out, "log.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
