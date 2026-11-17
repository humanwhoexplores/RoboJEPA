"""JEPA vs direct pilot on v4 (single goal-blind pushes), with the T5 token text pathway for both.

  predictor  start DINOv2 tokens + push -> final tokens (MSE). JEPA = frozen oracle TokenScorer
             (runs_v4/enc_t5/scorer.pt, trained on TRUE final frames) applied to the predicted finals.
  direct     start tokens + push + instruction tokens -> success logit, trained end-to-end on the labels.
             10 layers, 8.3M params, matched to predictor + scorer (5.1M + 3.5M).

Both pick the top-scoring of the 64 pushes per decision; checkpoints are chosen on val top-1, as in
the oracle run. Resumable after a crash (resume.pt).

    python -m robojepa.v4.pilot --model predictor --out runs_v4/pilot_jepa
    python -m robojepa.v4.pilot --model direct    --out runs_v4/pilot_direct
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from robojepa.models import DirectTokenClassifier, Predictor, TokenScorer
from robojepa.v4.oracle import N_CAND, TokenText, V4Split, evaluate, label_subset, split_names, summarize

T5 = "google-t5/t5-base"


def push(sp, scene, cand):
    """v4 push (x0, y0, theta, L) -> (x0, y0, L cos theta, L sin theta): start point and displacement, m."""
    x0, y0, th, L = torch.from_numpy(sp.meta["actions"][scene, cand]).to(sp.device).unbind(-1)
    return torch.stack([x0, y0, L * th.cos(), L * th.sin()], -1)


@torch.no_grad()
def predict_finals(pred, sp, scene):
    """Predicted final tokens of all 64 pushes in one scene, (64, 256, 384)."""
    s = np.full(N_CAND, scene)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return pred(sp.frames(s, np.full(N_CAND, -1)), push(sp, s, np.arange(N_CAND))).float()


@torch.no_grad()
def score_decisions(kind, model, scorer, sp, text, goals):
    """(len(goals), 64) scores of every push under each goal's instruction, plus prediction quality.
    kind predictor: model is the predictor, scorer ranks its predicted finals. direct: model scores."""
    model.eval(), scorer.eval()
    out, cache, err = [], {}, []
    for g in goals:
        sc = sp.scene[g]
        t = [x.expand(N_CAND, *x.shape[1:]) for x in text(sp, g)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if kind == "predictor":
                if sc not in cache:
                    cache.clear()
                    cache[sc] = predict_finals(model, sp, sc)
                    true = sp.frames(np.full(N_CAND, sc), np.arange(N_CAND))
                    err.append(F.cosine_similarity(cache[sc], true, dim=-1).mean().item())
                if scorer.two_frame:
                    out.append(scorer(cache[sc], *t, z0=sp.frames(np.full(N_CAND, sc), np.full(N_CAND, -1))).float())
                else:
                    out.append(scorer(cache[sc], *t).float())
            else:
                s = np.full(N_CAND, sc)
                out.append(model(sp.frames(s, np.full(N_CAND, -1)), push(sp, s, np.arange(N_CAND)), *t).float())
    model.train(), scorer.train()
    return torch.stack(out).cpu().numpy(), (float(np.mean(err)) if err else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["predictor", "direct", "scorer_ft"], required=True,
                    help="scorer_ft: fine-tune --scorer on the frozen --predictor's predicted finals")
    ap.add_argument("--features", default="features_v4")
    ap.add_argument("--scorer", default="runs_v4/enc_t5/scorer.pt")
    ap.add_argument("--scene_frac", type=float, default=1.0,
                    help="use this fraction of training scenes (labels for direct/scorer_ft, pushes for predictor)")
    ap.add_argument("--two_frame", action="store_true", help="--scorer is a two-frame (start + final) scorer")
    ap.add_argument("--predictor", default="runs_v4/pilot_jepa/model.pt", help="frozen predictor for scorer_ft")
    ap.add_argument("--eval_only", action="store_true", help="evaluate --out/model.pt (or --predictor) without training")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=None, help="default: predictor 6, direct 10")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--bs", type=int, default=None, help="default: predictor 64, direct 256")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dropout", type=float, default=0.0, help="direct: transformer dropout")
    ap.add_argument("--token_drop", type=float, default=0.0, help="direct: fraction of image patches hidden in training")
    ap.add_argument("--wd", type=float, default=0.05)
    a = ap.parse_args()
    kind = a.model
    bs = a.bs or (64 if kind == "predictor" else 256)
    if kind == "scorer_ft" and a.lr == 3e-4:
        a.lr = 1e-4

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = "cuda"
    text = TokenText(T5, dev)
    scorer = TokenScorer(text.dim, layers=4, two_frame=a.two_frame).to(dev).eval()
    scorer.load_state_dict(torch.load(a.scorer))
    scorer.requires_grad_(kind == "scorer_ft")
    train, val = V4Split(a.features, "train", dev), V4Split(a.features, "val", dev)
    pos, pool = label_subset(train, a.scene_frac)
    scenes = np.unique(train.scene[pool])  # predictor: these scenes' pushes (labels unused)
    val_goals = np.flatnonzero(val.meta["g_eval"])
    model = (DirectTokenClassifier(text.dim, layers=a.layers or 10, dropout=a.dropout, token_drop=a.token_drop) if kind == "direct"
             else Predictor(layers=a.layers or 6)).to(dev)
    if kind == "scorer_ft" or a.eval_only and kind == "predictor":
        model.load_state_dict(torch.load(a.predictor))
    if kind == "scorer_ft":  # train the scorer; the predictor stays frozen
        model.requires_grad_(False)
    train_mod = scorer if kind == "scorer_ft" else model
    save = lambda: torch.save(train_mod.state_dict(), f"{a.out}/model.pt")
    opt = torch.optim.AdamW(train_mod.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    os.makedirs(a.out, exist_ok=True)
    print(f"{kind}: {sum(p.numel() for p in train_mod.parameters()) / 1e6:.2f}M params, {len(scenes)} train scenes "
          f"x {N_CAND} pushes, {len(val_goals)} val decisions, bs {bs}, {a.steps} steps", flush=True)

    best, log, t0, tot, start = -1.0, [], time.time(), 0.0, 1
    ckpt = f"{a.out}/resume.pt"
    if os.path.exists(ckpt):
        c = torch.load(ckpt, weights_only=False)
        train_mod.load_state_dict(c["model"]), opt.load_state_dict(c["opt"]), sched.load_state_dict(c["sched"])
        best, log, start, rng.bit_generator.state = c["best"], c["log"], c["step"] + 1, c["rng"]
        torch.set_rng_state(c["torch_rng"])
        print(f"resumed at step {c['step']} (best val top1 {best:.3f})", flush=True)
    if a.eval_only:
        start = a.steps + 1
        if kind == "predictor":
            save()
    for step in range(start, a.steps + 1):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if kind == "predictor":  # uniform over all (scene, push) transitions
                s, k = scenes[rng.integers(len(scenes), size=bs)], rng.integers(N_CAND, size=bs)
                p = model(train.frames(s, np.full(bs, -1)), push(train, s, k))
                loss = F.mse_loss(p.float(), train.frames(s, k))
            elif kind == "scorer_ft":  # oracle sampling, but frames are the predictor's predicted finals
                pi = rng.integers(len(pos[0]), size=bs // 2)
                g = np.concatenate([pos[0][pi], pool[rng.integers(len(pool), size=bs - bs // 2)]])
                k = np.concatenate([pos[1][pi], rng.integers(N_CAND, size=bs - bs // 2)])
                s = train.scene[g]
                z0 = train.frames(s, np.full(bs, -1))
                with torch.no_grad():
                    zp = model(z0, push(train, s, k))
                y = torch.from_numpy(train.lab[g, k]).float().to(dev)
                logit = scorer(zp, *text(train, g, rng), **({"z0": z0} if a.two_frame else {}))
                loss = F.binary_cross_entropy_with_logits(logit.float(), y)
            else:  # half positive (goal, push) pairs, half random pushes of the same scene
                pi = rng.integers(len(pos[0]), size=bs // 2)
                g = np.concatenate([pos[0][pi], pool[rng.integers(len(pool), size=bs - bs // 2)]])
                k = np.concatenate([pos[1][pi], rng.integers(N_CAND, size=bs - bs // 2)])
                s = train.scene[g]
                logit = model(train.frames(s, np.full(bs, -1)), push(train, s, k), *text(train, g, rng))
                y = torch.from_numpy(train.lab[g, k]).float().to(dev)
                loss = F.binary_cross_entropy_with_logits(logit.float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(), sched.step()
        tot += loss.item()
        if step % a.eval_every == 0 or step == a.steps:
            sc, cos = score_decisions("direct" if kind == "direct" else "predictor", model, scorer, val, text, val_goals)
            v = summarize(evaluate(sc, val, val_goals))
            log.append({"step": step, "train_loss": tot / a.eval_every, "val_cos": cos, **{f"val_{k}": x for k, x in v.items()}})
            print(f"step {step:6d}  train {tot / a.eval_every:.4f}  val top1 {v['top1']:.3f}  top3 {v['top3']:.3f}  "
                  f"auc {v['auc']:.3f}" + (f"  cos {cos:.4f}" if cos else "") + f"  {time.time() - t0:.0f}s", flush=True)
            tot = 0.0
            if v["top1"] > best:
                best = v["top1"]
                save()
            torch.save({"model": train_mod.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "best": best, "log": log, "step": step, "rng": rng.bit_generator.state,
                        "torch_rng": torch.get_rng_state()}, ckpt + ".tmp")
            os.replace(ckpt + ".tmp", ckpt)
    json.dump(log, open(f"{a.out}/log.json", "w"), indent=1)
    del train
    torch.cuda.empty_cache()

    train_mod.load_state_dict(torch.load(f"{a.out}/model.pt"))
    kind = "predictor" if kind == "scorer_ft" else kind
    results = {}
    for split in split_names(a.features)[1:]:
        sp = V4Split(a.features, split, dev)
        goals = np.flatnonzero(sp.meta["g_eval"])
        sc, cos = score_decisions(kind, model, scorer, sp, text, goals)
        rows = evaluate(sc, sp, goals)
        res = {"all": {**summarize(rows), "pred_cos": cos}}
        if split == "test":
            res.update({f"camera_{c}": summarize([r for r in rows if r["camera"] == c]) for c in ("train", "held")})
        res.update({f"type_{t}": summarize([r for r in rows if r["type"] == t]) for t in sorted({r["type"] for r in rows})})
        results[split] = res
        del sp
        torch.cuda.empty_cache()
    json.dump(results, open(f"{a.out}/results.json", "w"), indent=1)
    print(f"\n{kind}: selection success on solvable decisions")
    print(f"{'split / subset':28s} {'n':>5s} {'top1':>7s} {'top3':>6s} {'auc':>6s} {'rule':>6s} {'random':>7s}")
    for split, res in results.items():
        for sub, r in res.items():
            if r["solvable"]:
                print(f"{split + ' / ' + sub:28s} {r['solvable']:5d} {r['top1']:7.3f} {r['top3']:6.3f} {r['auc']:6.3f} "
                      f"{r['rule']:6.3f} {r['random']:7.3f}")


if __name__ == "__main__":
    main()
