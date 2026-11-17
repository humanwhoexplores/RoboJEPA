#!/bin/bash
# Regularized direct classifier at 25% labels (a fairer low-data baseline). Crash-safe: resumes.
cd /home/deck/Public/RoboJEPA
n=de0.25_direct_reg
grep -q "^test / all" runs_v4/$n.log 2>/dev/null && { echo "=== $n already done"; exit; }
echo "=== $n start $(date +%T)"
.venv/bin/python -m robojepa.v4.pilot --model direct --scene_frac 0.25 --dropout 0.1 --token_drop 0.25 --wd 0.1 --out runs_v4/$n >> runs_v4/$n.log 2>&1
echo "=== $n exit $? $(date +%T)"
