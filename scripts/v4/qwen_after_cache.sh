#!/bin/bash
# After robojepa.v4.text_cache finishes: spot-check cache vs live Qwen, then the matched oracle run.
# Safe to re-run after a crash (oracle resumes from resume.pt; finished run is skipped).
cd /home/deck/Public/RoboJEPA
PY=.venv/bin/python C=features_v4/text_qwen35_2b
until [ -f $C/done ]; do sleep 30; done
$PY - <<'P'
import numpy as np, torch
from robojepa.v4.oracle import TokenText, CachedText, V4Split
c = CachedText("features_v4/text_qwen35_2b", "cuda"); live = TokenText("Qwen/Qwen3.5-2B-Base", "cuda")
class S: pass
for split in ("train", "test"):
    sp = S(); sp.name = split; sp.strings = np.load(f"features_v4/{split}_texts.npy")
    g = np.random.default_rng(1).integers(len(sp.strings), size=16)
    (a, pa), (b, pb) = c(sp, g), live(sp, g)
    assert (pa == pb).all()
    err = ((a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-6))[~pb]
    print(f"cache check {split}: rel err mean {err.mean():.4f} max {err.max():.4f}")
    assert err.max() < 0.05
P
[ $? -eq 0 ] || { echo "=== cache check FAILED"; exit 1; }
if ! grep -q "^test / all" runs_v4/enc_qwen35_2b_cached.log 2>/dev/null; then
  echo "=== qwen35_2b (cached) start $(date +%T)"
  $PY -m robojepa.v4.oracle --text tokens --text_cache $C --layers 4 --out runs_v4/enc_qwen35_2b_cached >> runs_v4/enc_qwen35_2b_cached.log 2>&1
  echo "=== exit $? $(date +%T)"
fi
grep -E "^(val|test_id|test) / all" runs_v4/enc_qwen35_2b_cached.log
