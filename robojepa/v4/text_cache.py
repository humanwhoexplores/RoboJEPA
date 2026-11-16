"""Precompute per-token text features once, for encoders too slow to run inside training (Qwen3.5's
Gated DeltaNet layers have no fast kernels on our ROCm build).

Every distinct instruction string across all splits is encoded once (last layer, as in TokenText) and
stored packed: tok.npy (Ntok, D) int8 + scale.npy (Ntok,) fp16, offsets.npy (Nstr+1). <split>_uid.npy
maps each split's text table (same shape as <split>_texts.npy) to string ids. Resumable after a crash.

    python -m robojepa.v4.text_cache --model Qwen/Qwen3.5-2B-Base --out features_v4/text_qwen35_2b
"""
import argparse
import os

import numpy as np
import torch

from robojepa.v4.oracle import TokenText

SPLITS = ("train", "val", "test_id", "test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--features", default="features_v4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=256)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tables = {s: np.load(f"{a.features}/{s}_texts.npy") for s in SPLITS}
    strings = sorted(set(np.concatenate([t.ravel() for t in tables.values()]).tolist()))
    sid = {s: i for i, s in enumerate(strings)}
    for s, t in tables.items():
        np.save(f"{a.out}/{s}_uid.npy", np.vectorize(sid.get)(t).astype(np.int32))

    enc = TokenText(a.model, "cuda")
    lens = np.array([len(enc.tok(x)["input_ids"]) for x in strings])
    off = np.concatenate([[0], np.cumsum(lens)])
    np.save(f"{a.out}/offsets.npy", off)
    np.save(f"{a.out}/strings.npy", np.array(strings))
    mode = "r+" if os.path.exists(f"{a.out}/tok.npy") else "w+"
    tok = np.lib.format.open_memmap(f"{a.out}/tok.npy", mode=mode, dtype=np.int8, shape=(off[-1], enc.dim))
    scale = np.lib.format.open_memmap(f"{a.out}/scale.npy", mode=mode, dtype=np.float16, shape=(off[-1],))
    prog = f"{a.out}/progress.txt"
    start = int(open(prog).read()) if os.path.exists(prog) else 0
    print(f"{len(strings)} strings, {off[-1]} tokens, dim {enc.dim}, resuming at {start}", flush=True)

    with torch.no_grad():
        for i in range(start, len(strings), a.bs):
            j = min(i + a.bs, len(strings))
            b = enc.tok(strings[i:j], padding=True, return_tensors="pt").to("cuda")
            h = enc.model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state.float()
            n = b["attention_mask"].sum(1).tolist()
            assert n == lens[i:j].tolist()
            h = torch.cat([h[r, :n[r]] for r in range(j - i)])
            s = h.abs().amax(-1, keepdim=True).clamp_min(1e-6) / 127
            tok[off[i]:off[j]] = (h / s).round().clamp(-127, 127).to(torch.int8).cpu().numpy()
            scale[off[i]:off[j]] = s.squeeze(-1).half().cpu().numpy()
            if (i // a.bs) % 20 == 0 or j == len(strings):
                tok.flush(), scale.flush()
                open(prog, "w").write(str(j))
                print(f"{j}/{len(strings)}", flush=True)
    open(f"{a.out}/done", "w").write("ok")


if __name__ == "__main__":
    main()
