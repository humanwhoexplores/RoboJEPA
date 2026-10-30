"""Cache frozen DINOv2 patch features for every stored frame, and frozen text embeddings for instructions.

    python -m robojepa.features --data data --out features
"""
import argparse
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F

VISION_MODEL = "facebook/dinov2-small"
TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Instruction templates. {t} is a side ("left"/"right") or a color ("blue"/"green").
TRAIN_TEMPLATES = [
    "push the cube to the {t} region",
    "move the red block into the {t} area",
    "slide the cube onto the {t} square",
    "put the cube in the {t} zone",
    "get the red cube to the {t} target",
]
HELDOUT_TEMPLATES = [  # paraphrases never seen in training
    "shove the block over to the {t} patch",
    "the cube should end up on the {t} mat",
]
TARGETS = {"left": "pos", "right": "pos", "blue": "color", "green": "color"}


def instructions():
    rows = []
    for split, temps in (("train", TRAIN_TEMPLATES), ("heldout", HELDOUT_TEMPLATES)):
        for ti, tmpl in enumerate(temps):
            for t, kind in TARGETS.items():
                rows.append(dict(text=tmpl.format(t=t), target=t, kind=kind, template=ti, split=split))
    return rows


@torch.no_grad()
def encode_text(out, device):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TEXT_MODEL)
    model = AutoModel.from_pretrained(TEXT_MODEL).to(device).eval()
    rows = instructions()
    batch = tok([r["text"] for r in rows], padding=True, return_tensors="pt").to(device)
    h = model(**batch).last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1).float()
    emb = F.normalize((h * mask).sum(1) / mask.sum(1), dim=-1).cpu().numpy().astype(np.float32)
    np.savez(os.path.join(out, "text.npz"), emb=emb, **{k: np.array([r[k] for r in rows]) for k in rows[0]})
    print(f"text: {len(rows)} instructions, dim {emb.shape[1]}")


class VisionEncoder:
    def __init__(self, device):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(VISION_MODEL, torch_dtype=torch.float16).to(device).eval()
        self.device = device
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, imgs_uint8):
        """(B,H,W,3) uint8 with H=W=224 -> (B,256,384) float16 patch tokens."""
        x = torch.from_numpy(imgs_uint8).to(self.device).permute(0, 3, 1, 2).float() / 255
        x = ((x - self.mean) / self.std).half()
        return self.model(pixel_values=x).last_hidden_state[:, 1:]


def encode_split(enc, data_dir, out, split, bs=128):
    files = sorted(glob.glob(os.path.join(data_dir, split, "scene_*.npz")))
    # First pass: build the index of frames.
    metas, n_frames = [], 0
    for f in files:
        d = np.load(f)
        m = len(d["region"])
        step_idx = np.full((m, d["imgs"].shape[1]), -1, np.int64)
        start, n_frames = n_frames, n_frames + 1
        for i in range(m):
            for k in range(d["n_push"][i]):
                step_idx[i, k] = n_frames
                n_frames += 1
        metas.append(dict(start=start, step_idx=step_idx, d={k: d[k] for k in d.files if k not in ("imgs", "start_img")}))

    feats = np.lib.format.open_memmap(os.path.join(out, f"{split}_patches.npy"), mode="w+",
                                      dtype=np.float16, shape=(n_frames, 256, 384))
    buf_img, buf_row = [], []

    def flush():
        if buf_img:
            feats[np.array(buf_row)] = enc(np.stack(buf_img)).cpu().numpy()
            buf_img.clear(), buf_row.clear()

    for f, meta in zip(files, metas):
        d = np.load(f)
        buf_img.append(d["start_img"]), buf_row.append(meta["start"])
        for i, row in np.argwhere(meta["step_idx"] >= 0):
            buf_img.append(d["imgs"][i, row]), buf_row.append(meta["step_idx"][i, row])
            if len(buf_img) >= bs:
                flush()
    flush()
    feats.flush()

    # Flatten per-candidate metadata.
    cat = lambda k: np.concatenate([m["d"][k] for m in metas])
    n_per = [len(m["d"]["region"]) for m in metas]
    np.savez(
        os.path.join(out, f"{split}_meta.npz"),
        scene=np.repeat(np.arange(len(metas)), n_per),
        start_row=np.repeat([m["start"] for m in metas], n_per),
        step_row=np.concatenate([m["step_idx"] for m in metas]),
        seqs=cat("seqs"), n_push=cat("n_push"), region=cat("region"), final_xy=cat("final_xy"),
        start_xy=np.repeat(np.stack([m["d"]["start_xy"] for m in metas]), n_per, axis=0),
        colors=np.repeat(np.stack([m["d"]["colors"] for m in metas]), n_per, axis=0),
    )
    print(f"{split}: {len(files)} scenes, {sum(n_per)} candidates, {n_frames} frame rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="features")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test_id", "test_ood"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encode_text(a.out, device)
    enc = VisionEncoder(device)
    for s in a.splits:
        encode_split(enc, a.data, a.out, s)


if __name__ == "__main__":
    main()
