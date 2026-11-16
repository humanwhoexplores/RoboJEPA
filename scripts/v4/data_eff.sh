#!/bin/bash
# Data efficiency: labels from a fraction of training scenes for scorer and direct; JEPA's predictor uses
# pushes from all scenes (no labels needed) or, as a control, only the labeled ones.
# Waits for runs_v4/scorer_fix.sh. Re-run after a crash: done runs skip, others resume.
cd /home/deck/Public/RoboJEPA
PY=.venv/bin/python
until grep -q "=== all done" runs_v4/scorer_fix.log; do sleep 60; done
run() {  # name module args...
  if grep -q "^test / all" runs_v4/$1.log 2>/dev/null; then echo "=== $1 already done"; return; fi
  echo "=== $1 start $(date +%T)"
  $PY -m "${@:2}" --out runs_v4/$1 >> runs_v4/$1.log 2>&1
  echo "=== $1 exit $? $(date +%T)"; grep -E "^(val|test_id|test) / all" runs_v4/$1.log
}
best() { $PY -c "import json; print(json.load(open('runs_v4/$1/results.json'))['val']['all']['top1'])"; }
if [ ! -f runs_v4/data_eff.cfg ]; then  # fixed once, so a resumed run keeps the same choice
  if $PY -c "import sys; sys.exit(not $(best ft2) > $(best ft1))"; then echo two > runs_v4/data_eff.cfg; else echo one > runs_v4/data_eff.cfg; fi
fi
CFG=$(cat runs_v4/data_eff.cfg); TF=""; [ $CFG = two ] && TF="--two_frame"
echo "=== scorer config: $CFG-frame (ft1 $(best ft1), ft2 $(best ft2))"
run jepa2 robojepa.v4.pilot --model predictor --eval_only --two_frame --scorer runs_v4/oracle2_t5/scorer.pt
for f in 0.25 0.1; do
  run de${f}_oracle robojepa.v4.oracle --text tokens --text_model google-t5/t5-base --layers 4 $TF --scene_frac $f
  run de${f}_ft_allpred robojepa.v4.pilot --model scorer_ft $TF --scorer runs_v4/de${f}_oracle/scorer.pt --scene_frac $f --steps 5000 --eval_every 500
  run de${f}_pred robojepa.v4.pilot --model predictor --scene_frac $f
  run de${f}_ft_subpred robojepa.v4.pilot --model scorer_ft $TF --scorer runs_v4/de${f}_oracle/scorer.pt --predictor runs_v4/de${f}_pred/model.pt --scene_frac $f --steps 5000 --eval_every 500
  run de${f}_direct robojepa.v4.pilot --model direct --scene_frac $f
done
echo "=== all done"
