"""Oracle check on v4: train the instruction scorer on TRUE final frames, then rank each decision's 64
goal-blind pushes by the scorer applied to their true final frames.

This is the JEPA pipeline with a perfect predictor, so it bounds what any world model can reach with
this encoder and scorer. If it is far below 1.0 on solvable decisions, perception or language is the
bottleneck and must be fixed before predictors are worth training.

    python -m robojepa.v4.oracle --features features_v4 --out runs_v4/oracle                 # pooled MiniLM
    python -m robojepa.v4.oracle --text tokens --layers 4 --out runs_v4/oracle_tok            # token-level text

--text pooled: one mean-pooled sentence vector per instruction (the midterm setup).
--text tokens: every token's contextual embedding from a frozen encoder (--text_model), computed on the
fly, with training paraphrases from robojepa.v4.texts.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from robojepa.models import Scorer, TokenScorer
from robojepa.train import auc

N_CAND = 64
HELD_CAMERAS = ("held1", "held2")


class V4Split:
    """int8 features (start frame + 64 final frames per scene) and the goal table. Features live on the
    GPU, or with on_gpu=False stay memory-mapped on disk (to leave GPU memory for a large text encoder)."""

    def __init__(self, feat_dir, split, device, on_gpu=True):
        m = np.load(f"{feat_dir}/{split}_meta.npz")
        self.meta = {k: m[k] for k in m.files}
        self.on_gpu = on_gpu
        path = f"{feat_dir}/{split}_z.npy"
        z = np.load(path, mmap_mode="r")
        if on_gpu:  # stream the file through one small buffer so RAM never holds the whole array
            self.z = torch.empty(z.shape, dtype=torch.int8, device=device)
            row, chunk = z[0].nbytes, 512
            buf = np.empty((chunk,) + z.shape[1:], np.int8)
            with open(path, "rb") as f:
                f.seek(z.offset)
                for i in range(0, len(z), chunk):
                    m = min(chunk, len(z) - i)
                    f.readinto(memoryview(buf)[:m].cast("B"))
                    self.z[i:i + m] = torch.from_numpy(buf[:m]).to(device)
            del z
        else:
            self.z = z
        self.s = torch.from_numpy(np.load(f"{feat_dir}/{split}_scale.npy")).to(device)
        self.scene, self.text, self.lab = self.meta["g_scene"], self.meta["g_text"], self.meta["g_lab"]
        self.strings = np.load(f"{feat_dir}/{split}_texts.npy") if os.path.exists(f"{feat_dir}/{split}_texts.npy") else None
        self.device = device
        self.name = split

    def frames(self, scene, cand):
        """cand -1 is the start frame. Returns (B,256,384) float."""
        rows = scene * (N_CAND + 1) + cand + 1
        z = self.z[torch.from_numpy(rows).to(self.device)] if self.on_gpu else torch.from_numpy(self.z[rows]).to(self.device)
        return z.float() * self.s[torch.from_numpy(rows).to(self.device)].float().unsqueeze(-1)


class PooledText:
    """Cached mean-pooled MiniLM vectors (features.py), one fixed text per goal."""

    def __init__(self, feat_dir, device):
        self.emb = torch.from_numpy(np.load(f"{feat_dir}/text.npz")["emb"]).to(device)

    def __call__(self, sp, g, rng=None):
        return (self.emb[sp.text[g]],)


class TokenText:
    """Frozen text encoder; returns per-token contextual embeddings (last layer) and a padding mask.
    In training (rng given) each goal uses a random one of its paraphrases.

    Works for encoders (MiniLM, T5) and for decoder LLMs (Qwen3.5, Gemma 4). For multimodal checkpoints
    only the language model is kept; image and audio towers are dropped."""

    def __init__(self, name, device):
        from transformers import AutoModel, AutoTokenizer, T5EncoderModel
        self.tok = AutoTokenizer.from_pretrained(name)
        self.tok.padding_side = "right"  # text positions in TokenScorer assume right padding
        llm = "t5" not in name and "MiniLM" not in name
        model = (T5EncoderModel if "t5" in name else AutoModel).from_pretrained(  # straight to GPU: low host RAM
            name, dtype=torch.bfloat16 if llm else torch.float32, device_map=device)
        model = getattr(model, "language_model", model)
        self.model = model.to(device).eval()
        cfg = self.model.config
        self.dim = getattr(cfg, "hidden_size", None) or cfg.d_model
        self.device = device

    @torch.no_grad()
    def __call__(self, sp, g, rng=None):
        g = np.atleast_1d(g)
        k = rng.integers(sp.strings.shape[1], size=len(g)) if rng is not None else np.zeros(len(g), int)
        b = self.tok(list(sp.strings[g, k]), padding=True, return_tensors="pt").to(self.device)
        h = self.model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
        return h.float(), b["attention_mask"] == 0


class CachedText:
    """Per-token features precomputed by robojepa.v4.text_cache (same outputs as TokenText, and the same
    random paraphrase draws, so a cached run matches a live one)."""

    def __init__(self, cache_dir, device):
        self.tok = np.load(f"{cache_dir}/tok.npy", mmap_mode="r")
        self.scale = np.load(f"{cache_dir}/scale.npy", mmap_mode="r")
        self.off = np.load(f"{cache_dir}/offsets.npy")
        self.uid = {s: np.load(f"{cache_dir}/{s}_uid.npy") for s in ("train", "val", "test_id", "test")}
        self.dim, self.device = self.tok.shape[1], device

    def __call__(self, sp, g, rng=None):
        g = np.atleast_1d(g)
        u = self.uid[sp.name]
        k = rng.integers(u.shape[1], size=len(g)) if rng is not None else np.zeros(len(g), int)
        ids = u[g, k]
        n = self.off[ids + 1] - self.off[ids]
        L = int(n.max())
        h = np.zeros((len(g), L, self.dim), np.float32)
        for r, (i, m) in enumerate(zip(ids, n)):
            o = self.off[i]
            h[r, :m] = self.tok[o:o + m] * self.scale[o:o + m, None].astype(np.float32)
        pad = torch.from_numpy(np.arange(L)[None] >= n[:, None])
        return torch.from_numpy(h).to(self.device), pad.to(self.device)


def split_names(feat_dir):
    """Splits of a feature cache: <feat_dir>/splits.json (v5) or v4's four. The first is the training split."""
    p = f"{feat_dir}/splits.json"
    return json.load(open(p)) if os.path.exists(p) else ["train", "val", "test_id", "test"]


def label_subset(sp, frac, seed=0):
    """Positive (goal, push) pairs and the goal pool of a fixed random fraction of training scenes
    (data-efficiency runs); frac 1 keeps everything."""
    n = len(sp.meta["scene_ids"])
    keep = np.zeros(n, bool)
    keep[np.random.default_rng([seed, 7]).permutation(n)[:max(1, round(frac * n))]] = True
    pool = np.flatnonzero(keep[sp.scene])
    g, k = np.nonzero(sp.lab[pool])
    return (pool[g], k), pool


def sample_batch(sp, pos, bs, rng, pool=None):
    """Half positive (goal, push) pairs; the rest random pairs from the same scene (almost all negative),
    so the same frame is a positive for one instruction and a negative for others. 1/16 start frames."""
    pi = rng.integers(len(pos[0]), size=bs // 2)
    pool = np.arange(len(sp.scene)) if pool is None else pool
    g = np.concatenate([pos[0][pi], pool[rng.integers(len(pool), size=bs - bs // 2)]])
    k = np.concatenate([pos[1][pi], rng.integers(N_CAND, size=bs - bs // 2)])
    k[bs // 2:][rng.random(bs - bs // 2) < 1 / 16] = -1
    y = np.where(k >= 0, sp.lab[g, np.maximum(k, 0)], 0)
    return g, k, y


@torch.no_grad()
def score_decisions(model, sp, text, goals):
    """Scores (len(goals), 64) of every push's true final frame under each goal's instruction."""
    model.eval()
    out = []
    for g in goals:
        z = sp.frames(np.full(N_CAND, sp.scene[g]), np.arange(N_CAND))
        t = [x.expand(N_CAND, *x.shape[1:]) for x in text(sp, g)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z0 = sp.frames(np.full(N_CAND, sp.scene[g]), np.full(N_CAND, -1)) if getattr(model, "two_frame", False) else None
            out.append(model(z, *t, z0=z0).float() if z0 is not None else model(z, *t).float())
    model.train()
    return torch.stack(out).cpu().numpy()


def evaluate(scores, sp, goals):
    """Per-decision rows: oracle pick, random and rule references."""
    rows = []
    cams = sp.meta["camera"]
    for sc, g in zip(scores, goals):
        lab = sp.lab[g]
        order = np.argsort(-sc)
        rule = sp.meta["g_rule"][g]
        rows.append({"solvable": bool(lab.any()), "n_pos": int(lab.sum()), "top1": int(lab[order[0]]),
                     "top3": int(lab[order[:3]].any()), "random": float(lab.mean()),
                     "rule": float(lab[rule]) if rule >= 0 else float(lab.mean()),
                     "type": str(sp.meta["g_type"][g]), "source": str(sp.meta["g_source"][g]),
                     "camera": "held" if cams[sp.scene[g]] in HELD_CAMERAS else "train",
                     "auc": auc(sc, lab) if 0 < lab.sum() < N_CAND else float("nan")})
    return rows


def summarize(rows):
    s = [r for r in rows if r["solvable"]]
    mean = lambda k, rr: round(float(np.mean([r[k] for r in rr])), 3) if rr else None
    return {"decisions": len(rows), "solvable": len(s), "top1": mean("top1", s), "top3": mean("top3", s),
            "random": mean("random", s), "rule": mean("rule", s),
            "auc": round(float(np.nanmean([r["auc"] for r in s])), 3) if s else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features_v4")
    ap.add_argument("--out", default="runs_v4/oracle")
    ap.add_argument("--text", choices=["pooled", "tokens"], default="pooled")
    ap.add_argument("--text_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--text_cache", default=None, help="dir from robojepa.v4.text_cache; replaces --text_model")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--scene_frac", type=float, default=1.0, help="train on labels of this fraction of scenes")
    ap.add_argument("--two_frame", action="store_true", help="token scorer also sees the start frame")
    ap.add_argument("--features_on_disk", action="store_true", help="keep image features memory-mapped, not on GPU")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = "cuda"
    text = (PooledText(a.features, dev) if a.text == "pooled" else
            CachedText(a.text_cache, dev) if a.text_cache else TokenText(a.text_model, dev))
    on_gpu = not a.features_on_disk
    train, val = V4Split(a.features, "train", dev, on_gpu), V4Split(a.features, "val", dev, on_gpu)
    pos, pool = label_subset(train, a.scene_frac)
    val_goals = np.flatnonzero(val.meta["g_eval"])
    model = (Scorer(layers=a.layers) if a.text == "pooled" else TokenScorer(text.dim, layers=a.layers, two_frame=a.two_frame)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    os.makedirs(a.out, exist_ok=True)
    print(f"scorer ({a.text} text) {a.layers} layers: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, "
          f"{len(train.scene)} train goals, {len(pos[0])} positive pairs, {len(val_goals)} val decisions", flush=True)

    best, log, t0, tot, start = -1.0, [], time.time(), 0.0, 1
    ckpt = f"{a.out}/resume.pt"
    if os.path.exists(ckpt):  # continue an interrupted run from its last evaluation
        c = torch.load(ckpt, weights_only=False)
        model.load_state_dict(c["model"]), opt.load_state_dict(c["opt"]), sched.load_state_dict(c["sched"])
        best, log, start, rng.bit_generator.state = c["best"], c["log"], c["step"] + 1, c["rng"]
        torch.set_rng_state(c["torch_rng"])
        print(f"resumed at step {c['step']} (best val top1 {best:.3f})", flush=True)
    for step in range(start, a.steps + 1):
        g, k, y = sample_batch(train, pos, a.bs, rng, pool)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z0 = train.frames(train.scene[g], np.full(len(g), -1)) if a.two_frame else None
            logit = model(train.frames(train.scene[g], k), *text(train, g, rng), **({"z0": z0} if a.two_frame else {}))
        loss = F.binary_cross_entropy_with_logits(logit.float(), torch.from_numpy(y).float().to(dev))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(), sched.step()
        tot += loss.item()
        if step % a.eval_every == 0 or step == a.steps:
            v = summarize(evaluate(score_decisions(model, val, text, val_goals), val, val_goals))
            log.append({"step": step, "train_loss": tot / a.eval_every, **{f"val_{k}": x for k, x in v.items()}})
            print(f"step {step:6d}  train {tot / a.eval_every:.4f}  val top1 {v['top1']:.3f}  top3 {v['top3']:.3f}  "
                  f"auc {v['auc']:.3f}  {time.time() - t0:.0f}s", flush=True)
            tot = 0.0
            if v["top1"] > best:
                best = v["top1"]
                torch.save(model.state_dict(), f"{a.out}/scorer.pt")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "best": best, "log": log, "step": step, "rng": rng.bit_generator.state,
                        "torch_rng": torch.get_rng_state()}, ckpt + ".tmp")
            os.replace(ckpt + ".tmp", ckpt)
    json.dump(log, open(f"{a.out}/log.json", "w"), indent=1)
    del train
    torch.cuda.empty_cache()

    model.load_state_dict(torch.load(f"{a.out}/scorer.pt"))
    results = {}
    for split in split_names(a.features)[1:]:
        sp = V4Split(a.features, split, dev, on_gpu)
        goals = np.flatnonzero(sp.meta["g_eval"])
        rows = evaluate(score_decisions(model, sp, text, goals), sp, goals)
        res = {"all": summarize(rows)}
        if split == "test":
            res.update({f"camera_{c}": summarize([r for r in rows if r["camera"] == c]) for c in ("train", "held")})
        res.update({f"source_{s}": summarize([r for r in rows if r["source"] == s]) for s in ("v3_focus", "open")})
        res.update({f"type_{t}": summarize([r for r in rows if r["type"] == t]) for t in sorted({r["type"] for r in rows})})
        results[split] = res
        del sp
        torch.cuda.empty_cache()
    json.dump(results, open(f"{a.out}/results.json", "w"), indent=1)
    print("\nselection success on solvable decisions (oracle = scorer on TRUE final frames)")
    print(f"{'split / subset':28s} {'n':>5s} {'oracle':>7s} {'top3':>6s} {'auc':>6s} {'rule':>6s} {'random':>7s}")
    for split, res in results.items():
        for sub, r in res.items():
            if not r["solvable"]:
                continue
            print(f"{split + ' / ' + sub:28s} {r['solvable']:5d} {r['top1']:7.3f} {r['top3']:6.3f} {r['auc']:6.3f} "
                  f"{r['rule']:6.3f} {r['random']:7.3f}")


if __name__ == "__main__":
    main()
