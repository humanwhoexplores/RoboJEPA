"""Cache frozen DINOv2 patch features and instruction embeddings.

    python -m jepa.features --data data/v1

Writes <data>/features.npz with
    traj_ids        (N,)
    z_initial       (N, 256, 384) float16  DINOv2-S/14 patch tokens of the initial frame (16x16 grid)
    z_final         (N, 256, 384) float16  same for the final, settled frame
    instructions    (M,)  unique instruction strings
    text_emb        (M, 384) float32       all-MiniLM-L6-v2 sentence embeddings

Only the initial and final frames are encoded; the arm is out of view in both.
The first run downloads model weights from Hugging Face (~100 MB).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():  # also true for ROCm builds
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_visual_encoder(name, device):
    if name == "random":  # offline stand-in with the same output shape, for testing the pipeline only
        torch.manual_seed(0)
        conv = torch.nn.Conv2d(3, 384, kernel_size=14, stride=14).to(device).eval()
        return lambda x: conv(x).flatten(2).transpose(1, 2)
    from transformers import AutoModel

    model = AutoModel.from_pretrained(f"facebook/{name}").to(device).eval()
    return lambda x: model(pixel_values=x).last_hidden_state[:, 1:]  # drop CLS, keep 256 patch tokens


def load_text_encoder(name, device):
    if name == "random":
        rng = np.random.default_rng(0)
        return lambda texts: np.stack([rng.standard_normal(384).astype(np.float32) for _ in texts])
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(f"sentence-transformers/{name}", device=str(device))
    return lambda texts: model.encode(list(texts), normalize_embeddings=True)


@torch.no_grad()
def encode_frames(encoder, frames, device, batch=32):
    out = []
    for i in range(0, len(frames), batch):
        x = torch.from_numpy(frames[i:i + batch]).float().permute(0, 3, 1, 2) / 255.0
        x = ((x - IMAGENET_MEAN) / IMAGENET_STD).to(device)
        out.append(encoder(x).float().cpu().numpy().astype(np.float16))
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/v1")
    ap.add_argument("--visual", default="dinov2-small", help="dinov2-small | dinov2-base | random")
    ap.add_argument("--text", default="all-MiniLM-L6-v2", help="sentence-transformers model | random")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    root = Path(args.data)
    device = pick_device(args.device)
    print("device:", device)

    m = pd.read_csv(root / "manifest.csv")
    first = np.load(root / "trajs" / f"{m.traj_id[0]}.npz")
    names = list(first["keyframe_names"])
    i0, i1 = names.index("initial"), names.index("final")
    initial = np.stack([np.load(root / "trajs" / f"{t}.npz")["keyframes"][i0] for t in m.traj_id])
    final = np.stack([np.load(root / "trajs" / f"{t}.npz")["keyframes"][i1] for t in m.traj_id])

    enc = load_visual_encoder(args.visual, device)
    z0 = encode_frames(enc, initial, device)
    z1 = encode_frames(enc, final, device)
    print("visual features:", z0.shape)

    rows = pd.read_csv(root / "instruction_rows.csv")
    instructions = np.array(sorted(rows.instruction.unique().tolist()), dtype=str)
    text_emb = np.asarray(load_text_encoder(args.text, device)(instructions), dtype=np.float32)
    print("instructions:", len(instructions), "text dim:", text_emb.shape[1])

    np.savez(root / "features.npz", traj_ids=np.array(m.traj_id.tolist(), dtype=str), z_initial=z0, z_final=z1,
             instructions=instructions, text_emb=text_emb, visual=args.visual, text=args.text)
    print("saved", root / "features.npz")


if __name__ == "__main__":
    main()
