"""Load cached features for a split onto the GPU and build training examples."""
import os

import numpy as np
import torch

TARGETS = ["left", "right", "blue", "green"]


class Split:
    def __init__(self, feat_dir, split, device, max_scenes=None):
        m = np.load(os.path.join(feat_dir, f"{split}_meta.npz"))
        keep = np.ones(len(m["scene"]), bool) if max_scenes is None else m["scene"] < max_scenes
        self.device = device
        self.z = torch.from_numpy(np.load(os.path.join(feat_dir, f"{split}_patches.npy"))).to(device)
        g = lambda k: m[k][keep]
        self.scene, self.start_row, self.step_row = g("scene"), g("start_row"), g("step_row")
        self.n_push, self.region, self.colors = g("n_push").astype(np.int64), g("region"), g("colors")
        self.seqs = torch.from_numpy(g("seqs")).float().to(device)
        self.term_row = self.step_row[np.arange(len(self.n_push)), self.n_push - 1]
        # Y[i, t] = candidate i satisfies instruction target t.
        self.Y = np.stack([self.region == side_of(t, self.colors) for t in TARGETS], 1).astype(np.float32)
        self.n_scenes = len(np.unique(self.scene))

    def transitions(self):
        """(src_row, push, dst_row) for every executed push."""
        src, act, dst = [], [], []
        for i in range(len(self.n_push)):
            for k in range(self.n_push[i]):
                src.append(self.start_row[i] if k == 0 else self.step_row[i, k - 1])
                act.append(self.seqs[i, k])
                dst.append(self.step_row[i, k])
        return np.array(src), torch.stack(act), np.array(dst)

    def scorer_frames(self):
        """Frames with known outcome labels: every terminal frame, plus each scene's start frame (no success)."""
        _, first = np.unique(self.scene, return_index=True)
        rows = np.concatenate([self.term_row, self.start_row[first]])
        Y = np.concatenate([self.Y, np.zeros((len(first), len(TARGETS)), np.float32)])
        return rows, Y


def side_of(target, colors):
    """Region index (0 left, 1 right) that a target word refers to, per candidate."""
    if target in ("left", "right"):
        return np.full(len(colors), 0 if target == "left" else 1)
    return np.where(colors[:, 0] == target, 0, 1)


class Texts:
    def __init__(self, feat_dir, device):
        t = np.load(os.path.join(feat_dir, "text.npz"))
        self.emb = torch.from_numpy(t["emb"]).to(device)
        self.target, self.split, self.text = t["target"], t["split"], t["text"]

    def ids(self, target, split):
        return np.where((self.target == target) & (self.split == split))[0]

    def sample(self, target_idx, split, rng):
        """Random instruction embedding for each target index in target_idx."""
        pools = [self.ids(t, split) for t in TARGETS]
        return self.emb[[rng.choice(pools[t]) for t in target_idx]]
