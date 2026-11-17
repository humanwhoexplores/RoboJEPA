"""Pretrained RoboJEPA (Meta, github.com/facebookresearch/robo_jepa) on v5: JEPA vs a direct competitor with the
same pretrained backbone. Features come from robojepa.v5.rj_cache (frozen V-JEPA 2.1 ViT-G, 16x16x1664 per frame);
goals, labels and instructions from features_v5 (robojepa.v5.features + robojepa.v4.texts).

Each push is a 4-step `franka` rollout (0.8 s steps, robot-base EE pose deltas as actions):
start frame (arm home) -> push start -> +0.8 s -> +1.6 s -> settled final.

  --mode wm         fine-tune the pretrained predictor on cached sequences (L1, teacher forcing + full rollout;
                    their loss). No labels. --steps 0 keeps the pretrained weights (zero-shot).
  --mode scorer     two-frame TokenScorer (start, final) on TRUE final ViT-G frames of the cached pushes.
  --mode scorer_ft  fine-tune --scorer on the --wm's predicted finals (all 64 pushes per scene have labels).
  --mode direct     same pretrained predictor + a fresh TokenScorer head on its own rollout, trained end to end on
                    labels only (no latent loss): the "pretrained direct" competitor.
  --mode eval       JEPA: roll out --wm for all 64 pushes, rank with --scorer, report like robojepa.v4.pilot.

    python -m robojepa.v5.rj_train --mode wm --out runs_rj/wm100m
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

from robojepa.models import TokenScorer
from robojepa.v4.oracle import N_CAND, TokenText, evaluate, label_subset, split_names, summarize

T5 = "google-t5/t5-base"
C, HW, T_STEPS = 1664, 16, 4
RJ = os.path.expanduser("~/Public/robo_jepa")
DEV_TYPE = ["cuda"]  # autocast device type (set from --device)


def actions_from_poses(p):
    """(..., 5, 7) poses -> (..., 4, 7) RoboJEPA state-delta actions: dxyz, euler of R_{t+1} R_t^T, dgripper."""
    sh = p.shape[:-2]
    p = p.reshape(-1, 5, 7)
    R = Rotation.from_euler("xyz", p[:, :, 3:6].reshape(-1, 3)).as_matrix().reshape(-1, 5, 3, 3)
    dR = R[:, 1:] @ np.swapaxes(R[:, :-1], -1, -2)
    rot = Rotation.from_matrix(dR.reshape(-1, 3, 3)).as_euler("xyz").reshape(-1, 4, 3)
    a = np.concatenate([p[:, 1:, :3] - p[:, :-1, :3], rot, p[:, 1:, 6:] - p[:, :-1, 6:]], -1)
    return a.reshape(*sh, 4, 7).astype(np.float32)


class Goals:
    """Goal table of one split (features_v5 meta + instruction texts), the interface TokenText/evaluate expect."""

    def __init__(self, feat_dir, split, limit=None):
        m = np.load(f"{feat_dir}/{split}_meta.npz")
        self.meta = {k: m[k] for k in m.files}
        self.strings = np.load(f"{feat_dir}/{split}_texts.npy")
        if limit is not None:  # first `limit` scenes only (smoke tests against a partial cache)
            n, keep = len(self.meta["scene_ids"]), self.meta["g_scene"] < limit
            self.meta = {k: v[keep] if k.startswith("g_") else v[:limit] if len(v) == n else v for k, v in self.meta.items()}
            self.strings = self.strings[keep]
        self.scene, self.text, self.lab = self.meta["g_scene"], self.meta["g_text"], self.meta["g_lab"]
        self.name = split


class RJSplit:
    """Cached ViT-G features: start frames on the GPU, sequences memory-mapped (train: ~20 GB)."""

    def __init__(self, rj_dir, feat_dir, split, device):
        m = np.load(f"{rj_dir}/{split}_meta.npz")
        n = len(m["scene_ids"])
        self.g = Goals(feat_dir, split, limit=n)
        assert (m["scene_ids"] == self.g.meta["scene_ids"]).all(), "scene order differs from features_v5"
        self.poses, self.sub = m["poses"], m["sub"].astype(int)
        self.act = actions_from_poses(self.poses)  # (n, 64, 4, 7)
        self.init = torch.from_numpy(np.load(f"{rj_dir}/{split}_init_z.npy")).to(device)
        self.init_s = torch.from_numpy(np.load(f"{rj_dir}/{split}_init_scale.npy")).to(device)
        self.has_seq = self.sub.shape[1] > 0
        if self.has_seq:
            self.seq = np.load(f"{rj_dir}/{split}_seq_z.npy", mmap_mode="r")
            self.seq_s = np.load(f"{rj_dir}/{split}_seq_scale.npy")
            self.slot = np.full((len(self.sub), N_CAND), -1)
            for i, s in enumerate(self.sub):
                self.slot[i, s] = np.arange(len(s))
        self.device = device

    def start(self, scene):
        sc = torch.from_numpy(np.asarray(scene)).to(self.device)
        return self.init[sc].float() * self.init_s[sc].float().unsqueeze(-1)

    def frames(self, scene, j, t=None):
        """Cached frames (B, 4, 256, C) of sequence slot j (push sub[scene, j]); t=3 only the final (B, 256, C)."""
        idx = np.argsort(scene * 1000 + j)  # sorted reads from the memmap
        sl = slice(None) if t is None else t
        z = np.empty((len(scene),) + ((T_STEPS,) if t is None else ()) + (HW * HW, C), np.int8)
        z[idx] = self.seq[scene[idx], j[idx], sl]
        s = torch.from_numpy(self.seq_s[scene, j, sl]).to(self.device).float()
        return torch.from_numpy(z).to(self.device).float() * s.unsqueeze(-1)

    def ctrl(self, scene, cand):
        """Padded actions and states (B, 4, 64) for the world model."""
        a = torch.from_numpy(self.act[scene, cand]).to(self.device)
        s = torch.from_numpy(self.poses[scene, cand, :4]).to(self.device)
        return F.pad(a, (0, 64 - 7)), F.pad(s, (0, 64 - 7))


def load_wm(ckpt_dir, weights, device):
    sys.path.insert(0, RJ)
    from app.robo_jepa.world_model import WorldModel
    wm = WorldModel.load_pretrained(ckpt_dir, weights_filename=weights, load_decoder=False, load_encoder=False,
                                    device=device)
    return wm.transition_model.to(device)


def wm_forward(model, feats, a, s):
    """feats (B,T,256,C) -> predictions (B,T,256,C) of the next frame at every step."""
    B, T = feats.shape[:2]
    emb = torch.zeros(B, 1, dtype=torch.long, device=feats.device)  # `franka` (embodiment 0)
    vm = torch.zeros(B, 1, dtype=torch.long, device=feats.device)  # view 0: DROID's exterior camera
    out = model(feats.view(B, T, 1, HW, HW, C), actions=a[:, :T], states=s[:, :T], view_map=vm, embodiment_ids=emb)[0]
    return out.view(B, T, HW * HW, C)


def rollout(model, z0, a, s, detach=True):
    """Autoregressive 4-step rollout from the start frame; returns (B,4,256,C) predicted frames."""
    feats = z0[:, None]
    for t in range(T_STEPS):
        nxt = wm_forward(model, feats, a, s)[:, -1:]
        feats = torch.cat([feats, nxt.detach() if detach and t < T_STEPS - 1 else nxt], 1)
    return feats[:, 1:]


def l1(p, y):
    return (p.float() - y).abs().mean()


@torch.no_grad()
def wm_val(model, sp, n=256, bs=16):
    """Teacher-forcing L1 and final-frame rollout L1 / cosine on the first n val sequences."""
    model.eval()
    sc, j = np.divmod(np.arange(min(n, sp.sub.size)), sp.sub.shape[1])
    tf, ro, cos, cp = [], [], [], []
    for i in range(0, len(sc), bs):
        s_, j_ = sc[i:i + bs], j[i:i + bs]
        y = sp.frames(s_, j_)
        a, st = sp.ctrl(s_, sp.sub[s_, j_])
        z0 = sp.start(s_)
        with torch.autocast(DEV_TYPE[0], dtype=torch.bfloat16):
            p = wm_forward(model, torch.cat([z0[:, None], y[:, :-1]], 1), a, st)
            r = rollout(model, z0, a, st)
        tf.append(l1(p, y).item()), ro.append(l1(r[:, -1], y[:, -1]).item())
        cos.append(F.cosine_similarity(r[:, -1].float(), y[:, -1], dim=-1).mean().item())
        cp.append(l1(z0, y[:, -1]).item())  # baseline: predict "nothing changed" (final = start frame)
    model.train()
    return float(np.mean(tf)), float(np.mean(ro)), float(np.mean(cos)), float(np.mean(cp))


@torch.no_grad()
def predicted_finals(model, sp, scene, bs=32):
    """(64, 256, C) predicted settled final frames of all pushes in one scene."""
    out = []
    for i in range(0, N_CAND, bs):
        c = np.arange(i, min(i + bs, N_CAND))
        s = np.full(len(c), scene)
        a, st = sp.ctrl(s, c)
        with torch.autocast(DEV_TYPE[0], dtype=torch.bfloat16):
            out.append(rollout(model, sp.start(s), a, st)[:, -1].float())
    return torch.cat(out)


@torch.no_grad()
def score_decisions(model, scorer, sp, text, goals, oracle_sub=False):
    """(len(goals), 64) scores. model None + oracle_sub: true finals of the cached pushes only (others -inf).
    scorer None: model is a direct model (scores itself via its head)."""
    if model is not None:
        model.eval()
    (scorer or model).eval()
    out, cache = [], {}
    for g in goals:
        sc = int(sp.g.scene[g])
        if sc not in cache:
            cache.clear()
            z0 = sp.start(np.array([sc]))
            if oracle_sub:
                fin = sp.frames(np.full(sp.sub.shape[1], sc), np.arange(sp.sub.shape[1]), t=3)
            elif scorer is None:
                fin = None
            else:
                fin = predicted_finals(model, sp, sc)
            cache[sc] = (z0, fin)
        z0, fin = cache[sc]
        t = text(sp.g, g)
        with torch.autocast(DEV_TYPE[0], dtype=torch.bfloat16):
            if scorer is None:
                scores = model.score_scene(sp, sc, t)
            else:
                n = len(fin)
                tt = [x.expand(n, *x.shape[1:]) for x in t]
                s = scorer(fin, *tt, z0=z0.expand(n, -1, -1)).float()
                if oracle_sub:
                    full = torch.full((N_CAND,), -1e9, device=s.device)
                    full[torch.from_numpy(sp.sub[sc]).to(s.device)] = s
                    s = full
                scores = s
        out.append(scores)
    (scorer or model).train()
    if model is not None:
        model.train()
    return torch.stack(out).cpu().numpy()


class Direct(torch.nn.Module):
    """Pretrained RoboJEPA predictor rolled out from the start frame + a TokenScorer head on (start, last
    predicted frame) + instruction; trained end to end on success labels only."""

    def __init__(self, wm, text_dim):
        super().__init__()
        self.wm = wm
        self.head = TokenScorer(text_dim, layers=4, two_frame=True, feat_dim=C)

    def forward(self, z0, a, s, text, pad):
        fin = rollout(self.wm, z0, a, s, detach=False)[:, -1]
        return self.head(fin, text, pad, z0=z0)

    def score_scene(self, sp, sc, t, bs=32):
        out = []
        for i in range(0, N_CAND, bs):
            c = np.arange(i, min(i + bs, N_CAND))
            s = np.full(len(c), sc)
            a, st = sp.ctrl(s, c)
            tt = [x.expand(len(c), *x.shape[1:]) for x in t]
            out.append(self(sp.start(s), a, st, *tt).float())
        return torch.cat(out)


def report(results, name):
    print(f"\n{name}: selection success on solvable decisions")
    print(f"{'split / subset':28s} {'n':>5s} {'top1':>7s} {'top3':>6s} {'auc':>6s} {'rule':>6s} {'random':>7s}")
    for split, res in results.items():
        for sub, r in res.items():
            if r["solvable"]:
                print(f"{split + ' / ' + sub:28s} {r['solvable']:5d} {r['top1']:7.3f} {r['top3']:6.3f} {r['auc']:6.3f} "
                      f"{r['rule']:6.3f} {r['random']:7.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["wm", "scorer", "scorer_ft", "direct", "eval"], required=True)
    ap.add_argument("--rj", default="features_rj")
    ap.add_argument("--features", default="features_v5")
    ap.add_argument("--ckpt", default=f"{RJ}/checkpoints/wm-100m")
    ap.add_argument("--weights", default="cooldown_100m_150k.pth.tar")
    ap.add_argument("--wm", default=None, help="fine-tuned predictor state dict (scorer_ft, eval); default pretrained")
    ap.add_argument("--scorer", default=None)
    ap.add_argument("--scene_frac", type=float, default=1.0, help="labels from this fraction of training scenes")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--eval_every", type=int, default=None)
    ap.add_argument("--bs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    D = {"wm": (3000, 250, 16, 2e-5), "scorer": (20000, 1000, 64, 3e-4), "scorer_ft": (4000, 500, 32, 1e-4),
         "direct": (6000, 500, 16, 2e-5), "eval": (0, 1, 1, 0)}[a.mode]
    steps = D[0] if a.steps is None else a.steps
    every, bs, lr = a.eval_every or D[1], a.bs or D[2], a.lr or D[3]
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = a.device
    DEV_TYPE[0] = torch.device(dev).type
    train, val = RJSplit(a.rj, a.features, "train", dev), RJSplit(a.rj, a.features, "val", dev)
    text = TokenText(T5, dev) if a.mode != "wm" else None

    def get_wm():
        m = load_wm(a.ckpt, a.weights, dev)
        if a.wm:
            m.load_state_dict(torch.load(a.wm))
        return m

    wm = scorer = None
    if a.mode in ("wm", "scorer_ft", "direct", "eval"):
        wm = get_wm()
    if a.mode in ("scorer", "scorer_ft", "eval"):
        scorer = TokenScorer(text.dim, layers=4, two_frame=True, feat_dim=C).to(dev)
        if a.mode != "scorer":
            scorer.load_state_dict(torch.load(a.scorer))
    if a.mode == "direct":
        model = Direct(wm, text.dim).to(dev)
        train_mod = model
    else:
        model = wm
        train_mod = {"wm": wm, "scorer": scorer, "scorer_ft": scorer, "eval": scorer}[a.mode]
    if a.mode in ("scorer_ft", "eval"):
        wm.requires_grad_(False), wm.eval()
    elif wm is not None:  # load_pretrained freezes the predictor
        wm.requires_grad_(True)

    pos, pool = label_subset(train.g, a.scene_frac)
    keep = np.unique(train.g.scene[pool])
    val_goals = np.flatnonzero(val.g.meta["g_eval"])
    # scorer: only (goal, push) pairs whose final frame is cached
    if a.mode == "scorer":
        cached = train.slot[train.g.scene[pos[0]], pos[1]] >= 0
        pos = (pos[0][cached], pos[1][cached])
    print(f"{a.mode}: {sum(p.numel() for p in train_mod.parameters()) / 1e6:.1f}M trainable params, {len(keep)} train "
          f"scenes, {len(pos[0])} positive pairs, {len(val_goals)} val decisions, bs {bs}, {steps} steps", flush=True)

    def val_metric():
        if a.mode == "wm":
            tf, ro, cos, cp = wm_val(wm, val)
            return -ro, f"tf L1 {tf:.4f}  rollout L1 {ro:.4f} (copy-start {cp:.4f})  cos {cos:.4f}"
        if a.mode == "scorer":
            v = summarize(evaluate(score_decisions(None, scorer, val, text, val_goals, oracle_sub=True), val.g, val_goals))
            return v["top1"], f"val(cached pushes) top1 {v['top1']:.3f} auc {v['auc']:.3f}"
        sc = score_decisions(wm, scorer, val, text, val_goals) if a.mode == "scorer_ft" else \
            score_decisions(model, None, val, text, val_goals)
        v = summarize(evaluate(sc, val.g, val_goals))
        return v["top1"], f"val top1 {v['top1']:.3f} top3 {v['top3']:.3f} auc {v['auc']:.3f}"

    opt = torch.optim.AdamW([p for p in train_mod.parameters() if p.requires_grad], lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=max(steps, 1), pct_start=0.05)
    ckpt, best, log, start, t0, tot = f"{a.out}/resume.pt", -1e9, [], 1, time.time(), 0.0
    save = lambda: torch.save(train_mod.state_dict(), f"{a.out}/model.pt")
    if os.path.exists(ckpt):
        c = torch.load(ckpt, weights_only=False)
        train_mod.load_state_dict(c["model"]), opt.load_state_dict(c["opt"]), sched.load_state_dict(c["sched"])
        best, log, start, rng.bit_generator.state = c["best"], c["log"], c["step"] + 1, c["rng"]
        print(f"resumed at step {c['step']}", flush=True)
    if steps == 0 and a.mode != "eval":
        save()
    train_mod.train()
    for step in range(start, steps + 1):
        with torch.autocast(DEV_TYPE[0], dtype=torch.bfloat16):
            if a.mode == "wm":  # uniform over cached sequences of all training scenes (no labels)
                s, j = rng.integers(len(train.sub), size=bs), rng.integers(train.sub.shape[1], size=bs)
                y = train.frames(s, j)
                act, st = train.ctrl(s, train.sub[s, j])
                z0 = train.start(s)
                p = wm_forward(wm, torch.cat([z0[:, None], y[:, :-1]], 1), act, st)
                r = rollout(wm, z0, act, st)
                loss = 0.5 * l1(p, y) + 0.5 * l1(r, y)
            else:  # half positive (goal, push) pairs, half random pushes of the same scene (as robojepa.v4)
                pi = rng.integers(len(pos[0]), size=bs // 2)
                g = np.concatenate([pos[0][pi], pool[rng.integers(len(pool), size=bs - bs // 2)]])
                s = train.g.scene[g]
                if a.mode == "scorer":
                    jr = rng.integers(train.sub.shape[1], size=bs - bs // 2)
                    k = np.concatenate([pos[1][pi], train.sub[s[bs // 2:], jr]])
                    j = train.slot[s, k]
                    fin = train.frames(s, j, t=3)
                    logit = scorer(fin, *text(train.g, g, rng), z0=train.start(s))
                else:
                    k = np.concatenate([pos[1][pi], rng.integers(N_CAND, size=bs - bs // 2)])
                    act, st = train.ctrl(s, k)
                    z0 = train.start(s)
                    if a.mode == "scorer_ft":
                        with torch.no_grad():
                            fin = rollout(wm, z0, act, st)[:, -1]
                        logit = scorer(fin, *text(train.g, g, rng), z0=z0)
                    else:
                        logit = model(z0, act, st, *text(train.g, g, rng))
                y = torch.from_numpy(train.g.lab[g, k]).float().to(dev)
                loss = F.binary_cross_entropy_with_logits(logit.float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_mod.parameters(), 1.0)
        opt.step(), sched.step()
        tot += loss.item()
        if step % every == 0 or step == steps:
            m, msg = val_metric()
            log.append({"step": step, "train_loss": tot / every, "val": m, "msg": msg})
            print(f"step {step:6d}  train {tot / every:.4f}  {msg}  {time.time() - t0:.0f}s", flush=True)
            tot = 0.0
            if m > best:
                best = m
                save()
            torch.save({"model": train_mod.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "best": best, "log": log, "step": step, "rng": rng.bit_generator.state}, ckpt + ".tmp")
            os.replace(ckpt + ".tmp", ckpt)
    json.dump(log, open(f"{a.out}/log.json", "w"), indent=1)
    if a.mode in ("wm", "scorer"):
        return  # no decision eval: wm has no scorer; scorer's oracle covers cached pushes only

    if a.mode != "eval":
        train_mod.load_state_dict(torch.load(f"{a.out}/model.pt"))
    del train
    torch.cuda.empty_cache()
    results = {}
    for split in split_names(a.features)[1:]:
        sp = RJSplit(a.rj, a.features, split, dev)
        goals = np.flatnonzero(sp.g.meta["g_eval"])
        sc = score_decisions(model, None, sp, text, goals) if a.mode == "direct" else \
            score_decisions(wm, scorer, sp, text, goals)
        rows = evaluate(sc, sp.g, goals)
        results[split] = {"all": summarize(rows),
                          **{f"type_{t}": summarize([r for r in rows if r["type"] == t]) for t in sorted({r["type"] for r in rows})}}
        print(f"{split} / all  {results[split]['all']}", flush=True)
        del sp
    json.dump(results, open(f"{a.out}/results.json", "w"), indent=1)
    report(results, a.mode)


if __name__ == "__main__":
    main()
