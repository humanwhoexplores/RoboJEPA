#!/bin/bash
# JEPA vs direct pilot on v4, one run at a time. Re-run after a crash: finished runs skip, others resume.
cd /home/deck/Public/RoboJEPA
run() {
  if grep -q "^test / all" runs_v4/pilot_$1.log 2>/dev/null; then echo "=== $1 already done"; return; fi
  echo "=== $1 start $(date +%T)"
  .venv/bin/python -m robojepa.v4.pilot --model $2 --out runs_v4/pilot_$1 >> runs_v4/pilot_$1.log 2>&1
  echo "=== $1 exit $? $(date +%T)"; grep -E "^(val|test_id|test) / all" runs_v4/pilot_$1.log
}
run jepa predictor
run direct direct
echo "=== all done"
