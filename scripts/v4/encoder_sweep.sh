#!/bin/bash
# Language-encoder ablation for the v4 oracle, one run at a time (RAM/GPU allow only one).
# Safe to re-run after a crash: finished runs are skipped, interrupted ones resume.
cd /home/deck/Public/RoboJEPA
PY=.venv/bin/python
run() {  # name model [extra args]; skips finished runs, resumes interrupted ones (resume.pt)
  if grep -q "^test / all" runs_v4/enc_$1.log 2>/dev/null; then echo "=== $1 already done"; return; fi
  echo "=== $1 start $(date +%T)"
  $PY -m robojepa.v4.oracle --text tokens --text_model "$2" --layers 4 --out runs_v4/enc_$1 "${@:3}" >> runs_v4/enc_$1.log 2>&1
  echo "=== $1 exit $? $(date +%T)"; grep -E "^(val|test_id|test) / all" runs_v4/enc_$1.log
}
dl() { $PY -c "from huggingface_hub import snapshot_download as s; s('$1', allow_patterns=['*.json','*.safetensors','*.model','tokenizer*','*.jinja','*.txt'])" >/dev/null 2>&1; echo "=== downloaded $1 ($?)"; df -h / | tail -1; }
rmcache() {  # model folder plus its files in the shared hub/blobs store (newer huggingface_hub layout)
  local m=models--${1//\//--} h=~/.cache/huggingface/hub
  for r in $(grep -l "$m/" $h/blobs/*/*.refs 2>/dev/null); do rm -f "${r%.refs}" "$r" "${r%.refs}.lock"; done
  rm -rf $h/$m; echo "=== removed cache $1"; }
run minilm sentence-transformers/all-MiniLM-L6-v2
run t5 google-t5/t5-base
run gemma4_e2b google/gemma-4-E2B --features_on_disk
grep -q "^test / all" runs_v4/enc_gemma4_e2b.log && rmcache google/gemma-4-E2B
dl Qwen/Qwen3.5-2B-Base
run qwen35_2b Qwen/Qwen3.5-2B-Base --features_on_disk
echo "=== all done"
