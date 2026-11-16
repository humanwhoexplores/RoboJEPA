#!/bin/bash
# Scorer fixes for the JEPA pipeline, one run at a time. Re-run after a crash: done runs skip, others resume.
cd /home/deck/Public/RoboJEPA
PY=.venv/bin/python
run() {  # name module args...
  if grep -q "^test / all" runs_v4/$1.log 2>/dev/null; then echo "=== $1 already done"; return; fi
  echo "=== $1 start $(date +%T)"
  $PY -m "${@:2}" --out runs_v4/$1 >> runs_v4/$1.log 2>&1
  echo "=== $1 exit $? $(date +%T)"; grep -E "^(val|test_id|test) / all" runs_v4/$1.log
}
run ft1 robojepa.v4.pilot --model scorer_ft --steps 5000 --eval_every 500
run oracle2_t5 robojepa.v4.oracle --text tokens --text_model google-t5/t5-base --layers 4 --two_frame
run jepa2 robojepa.v4.pilot --model predictor --eval_only --two_frame --scorer runs_v4/oracle2_t5/scorer.pt
run ft2 robojepa.v4.pilot --model scorer_ft --two_frame --scorer runs_v4/oracle2_t5/scorer.pt --steps 5000 --eval_every 500
echo "=== all done"
