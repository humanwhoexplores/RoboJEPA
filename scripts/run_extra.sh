#!/usr/bin/env bash
# Seed variance for the midterm (seed 0 comes from run_midterm.sh). Same step budgets as the sweep.
# The direct-baseline tuning that chose those budgets is archived in v2/.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTHONWARNINGS=ignore
mkdir -p runs/extra results/extra
declare -A STEPS=([predictor]=15000 [scorer]=5000 [direct]=15000)

for n in 400 100; do
  for s in 1 2; do
    for m in predictor scorer direct; do
      out=runs/extra/n${n}_s${s}_$m
      [ -f $out/log.json ] || $PY -m robojepa.train --model $m --scenes $n --seed $s --steps ${STEPS[$m]} --out $out \
        > $out.log
    done
    $PY -m robojepa.evaluate --predictor runs/extra/n${n}_s${s}_predictor --scorer runs/extra/n${n}_s${s}_scorer \
      --direct runs/extra/n${n}_s${s}_direct --out results/extra/n${n}_s${s} > /dev/null
  done
done
echo EXTRA DONE
