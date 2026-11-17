#!/bin/bash
# Scratch models on v5 (same recipes as v4): two-frame oracle scorer, predictor, JEPA (frozen / fine-tuned
# scorer), direct (plain and regularized), then 25% labels. Waits for the feature cache. Crash-safe.
cd /home/deck/Public/RoboJEPA
PY=.venv/bin/python; F="--features features_v5"; R=runs_v5
until grep -q FEATURES_DONE $R/features.log; do sleep 60; done
run() {  # name module args...
  if grep -q "^test_obj / all" $R/$1.log 2>/dev/null; then echo "=== $1 already done"; return; fi
  echo "=== $1 start $(date +%T)"
  $PY -m "${@:2}" $F --out $R/$1 >> $R/$1.log 2>&1
  echo "=== $1 exit $? $(date +%T)"; grep -E "^[a-z_]+ / all" $R/$1.log
}
T5="--text tokens --text_model google-t5/t5-base --layers 4 --two_frame"
run oracle2 robojepa.v4.oracle $T5
run pred robojepa.v4.pilot --model predictor --two_frame --scorer $R/oracle2/scorer.pt
run ft2 robojepa.v4.pilot --model scorer_ft --two_frame --scorer $R/oracle2/scorer.pt --predictor $R/pred/model.pt --steps 5000 --eval_every 500
run direct robojepa.v4.pilot --model direct --scorer $R/oracle2/scorer.pt --two_frame
run direct_reg robojepa.v4.pilot --model direct --scorer $R/oracle2/scorer.pt --two_frame --dropout 0.1 --token_drop 0.25 --wd 0.1
f=0.25
run de${f}_oracle robojepa.v4.oracle $T5 --scene_frac $f
run de${f}_noft robojepa.v4.pilot --model predictor --eval_only --two_frame --scorer $R/de${f}_oracle/scorer.pt --predictor $R/pred/model.pt --scene_frac $f
run de${f}_ft robojepa.v4.pilot --model scorer_ft --two_frame --scorer $R/de${f}_oracle/scorer.pt --predictor $R/pred/model.pt --scene_frac $f --steps 5000 --eval_every 500
run de${f}_direct_reg robojepa.v4.pilot --model direct --scorer $R/oracle2/scorer.pt --two_frame --scene_frac $f --dropout 0.1 --token_drop 0.25 --wd 0.1
echo "=== all done"
