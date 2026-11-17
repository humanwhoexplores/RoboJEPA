"""Cache frozen V-JEPA 2.1 ViT-G/384 features of v5 for Meta's RoboJEPA world models (github.com/facebookresearch/robo_jepa).

A frame is 16x16 tokens x 1664 (256 px input), ~426 KB at int8, so only the frames the experiments use are kept:

  init   every scene's start frame (arm at home)                                   all splits
  seq    for K pushes per scene, the frames at push start, +0.8 s, +1.6 s, and the
         settled final (arm back home)                                             train (K=24), val (K=8)
  poses  for all 64 pushes, the tool pose at home, push start, +0.8 s, +1.6 s, final:
         (x, y, z) in the robot base frame + euler xyz + gripper 0, RoboJEPA's `franka` convention
         (one WM step = 0.8 s). Actions are their differences (app/robo_jepa/world_model.poses_to_diffs).

Scene order per split matches features_v5 (sorted scene ids). The +0.8 s / +1.6 s frames come from the 5 fps mp4
(nearest frame); the start and final frames are the lossless keyframes.

    python -m robojepa.v5.rj_cache --encoder ~/Public/robo_jepa/checkpoints/vjepa2_1_vitG_384_target_bf16.pt --out features_rj
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

SPLITS = ("train", "val", "test_id", "test_lang", "test_cam", "test_pos", "test_obj")
K_PUSH = {"train": 24, "val": 8}
N_CAND, CONTROL_HZ, STEP_S = 64, 20, 0.8
BASE = np.array([-0.52, 0.0, 0.0])  # gen_v5.scene_xml.BASE: robot base in the world (no rotation)
GRID, DIM = 16, 1664


def load_encoder(path, robo_jepa, device):
    sys.path.insert(0, robo_jepa)
    from app.robo_jepa.utils import load_vjepa2_1_encoder
    torch.set_default_dtype(torch.bfloat16)  # build straight on the GPU in bf16: an fp32 CPU copy (7.4 GB) OOMs here
    with torch.device(device):
        vit = load_vjepa2_1_encoder()
    torch.set_default_dtype(torch.float32)
    vit.load_state_dict(torch.load(path, map_location="cpu", mmap=True))
    return vit.to(device, torch.bfloat16).eval()


@torch.no_grad()
def encode(vit, imgs, device, bs=16):
    """(N,256,256,3) uint8 -> (N,256,1664) layer-normed tokens (WorldModel.encode with encoder_layernorm)."""
    mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(imgs), bs):
        x = torch.from_numpy(imgs[i:i + bs]).to(device).permute(0, 3, 1, 2).float() / 255
        x = ((x - mean) / std).unsqueeze(2)  # (B,3,T=1,H,W): V-JEPA 2.1 takes one frame per image
        with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
            h = vit(x).float()
        out.append(F.layer_norm(h, (h.size(-1),)))
    return torch.cat(out)


def quantize(z):
    s = z.abs().amax(-1, keepdim=True).clamp_min(1e-6) / 127
    return (z / s).round().clamp(-127, 127).to(torch.int8).cpu().numpy(), s.squeeze(-1).half().cpu().numpy()


def push_times(log):
    """Control-step indices of: home, push start, +0.8 s, +1.6 s, final."""
    ph = log["phase"]
    t0 = int(np.flatnonzero(ph == "push")[0]) - 1
    d = int(STEP_S * CONTROL_HZ)
    n = len(ph)
    return np.array([0, t0, min(t0 + d, n - 1), min(t0 + 2 * d, n - 1), n - 1])


def pose(ee):
    """v5 ee_state rows -> RoboJEPA `franka` state. v5 logs the pusher frame (z up while the flange points down,
    roll ~ 0); DROID logs the gripper frame, z along the fingers (roll ~ +-pi). Convert: R_flange = R_tool Rx(pi)."""
    from scipy.spatial.transform import Rotation
    p = np.array(ee, np.float64)[..., :7].copy()
    p[..., :3] -= BASE
    R = Rotation.from_euler("xyz", p[..., 3:6].reshape(-1, 3)) * Rotation.from_euler("x", np.pi)
    p[..., 3:6] = R.as_euler("xyz").reshape(p[..., 3:6].shape)
    return p.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/v5")
    ap.add_argument("--encoder", required=True, help="target-encoder state dict (see extract_encoder below)")
    ap.add_argument("--robo_jepa", default=os.path.expanduser("~/Public/robo_jepa"))
    ap.add_argument("--out", default="features_rj")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--k", type=int, default=None, help="override sequences per scene (smoke test)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    sys.path.insert(0, os.path.expanduser("~/Public/RoboJEPA-v5"))
    from gen_v5.video import read_mp4
    os.makedirs(a.out, exist_ok=True)
    dev = a.device
    vit = load_encoder(a.encoder, a.robo_jepa, dev)
    sids = sorted(f[:-5] for f in os.listdir(f"{a.data}/scenes") if f.endswith(".json") and "rows" not in f)
    by_split = {s: [] for s in SPLITS}
    for s in sids:
        by_split[json.load(open(f"{a.data}/scenes/{s}.json"))["spec"]["v5_split"]].append(s)

    for split in SPLITS:
        ss = by_split[split][: a.limit]
        n, k = len(ss), K_PUSH.get(split, 0)
        k = min(k, a.k) if a.k is not None else k
        if os.path.exists(f"{a.out}/{split}_done"):
            print(f"{split}: done", flush=True)
            continue
        iz = np.lib.format.open_memmap(f"{a.out}/{split}_init_z.npy", "w+", np.int8, (n, GRID * GRID, DIM))
        iscale = np.zeros((n, GRID * GRID), np.float16)
        poses = np.zeros((n, N_CAND, 5, 7), np.float32)
        sub = np.zeros((n, k), np.int16)
        if k:
            sz = np.lib.format.open_memmap(f"{a.out}/{split}_seq_z.npy", "w+", np.int8, (n, k, 4, GRID * GRID, DIM))
            sscale = np.zeros((n, k, 4, GRID * GRID), np.float16)
        for i, sid in enumerate(ss):
            lab = np.load(f"{a.data}/scenes/{sid}_labels.npz")
            tids = [str(t) for t in lab["traj_ids"]]
            assert tids == [f"{sid}_c{c:02d}" for c in range(N_CAND)]
            rng = np.random.default_rng([i, 11])
            sub[i] = np.sort(rng.choice(N_CAND, k, replace=False)) if k else sub[i]
            imgs = [np.load(f"{a.data}/scenes/{sid}_initial.npz")["initial"]]
            for c, tid in enumerate(tids):
                t = np.load(f"{a.data}/trajs/{tid}.npz")
                times = push_times(t)
                poses[i, c] = pose(t["ee_state"][times])
                if c in sub[i]:
                    vid = read_mp4(f"{a.data}/videos/{tid}.mp4")
                    vs = t["video_step"]
                    names = list(t["keyframe_names"])
                    mid = [vid[int(np.argmin(np.abs(vs - times[j])))] for j in (1, 2, 3)]
                    imgs += mid + [t["keyframes"][names.index("final")]]
            q, s = quantize(encode(vit, np.stack(imgs), dev))
            iz[i], iscale[i] = q[0], s[0]
            if k:
                sz[i] = q[1:].reshape(k, 4, *q.shape[1:])
                sscale[i] = s[1:].reshape(k, 4, -1)
            if i % 20 == 0:
                print(f"{split}: {i}/{n}", flush=True)
        iz.flush()
        np.save(f"{a.out}/{split}_init_scale.npy", iscale)
        np.savez(f"{a.out}/{split}_meta.npz", scene_ids=np.array(ss), poses=poses, sub=sub)
        if k:
            sz.flush()
            np.save(f"{a.out}/{split}_seq_scale.npy", sscale)
        open(f"{a.out}/{split}_done", "w").write("ok")
        print(f"{split}: {n} scenes, {k} sequences each", flush=True)


def extract_encoder(src, dst):
    """Keep only the frozen target encoder of the 30 GB V-JEPA 2.1 training checkpoint, in bf16 (~3.7 GB)."""
    sd = torch.load(src, map_location="cpu", mmap=True)["target_encoder"]
    torch.save({k.replace("module.backbone.", ""): v.to(torch.bfloat16) for k, v in sd.items()}, dst)


if __name__ == "__main__":
    main()
