#!/usr/bin/env bash
# Midterm experiments: main comparison on all training scenes + data-efficiency sweep.
# Assumes data/ has been collected (python -m robojepa.collect).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTHONWARNINGS=ignore

mkdir -p runs results
# Step budgets chosen on val (see v2/ for the 5k-step runs): predictor and direct keep improving to ~15k.
declare -A STEPS=([predictor]=15000 [scorer]=5000 [direct]=15000)
[ -f features/test_ood_meta.npz ] || $PY -m robojepa.features --data data --out features

for n in 400 200 100 50; do
  for m in predictor scorer direct; do
    out=runs/n${n}/$m
    [ -f $out/log.json ] || $PY -m robojepa.train --model $m --scenes $n --steps ${STEPS[$m]} --out $out | tee runs/n${n}_$m.log
  done
  $PY -m robojepa.evaluate --predictor runs/n${n}/predictor --scorer runs/n${n}/scorer --direct runs/n${n}/direct \
    --out results/n${n}
done
$PY -m robojepa.figures
