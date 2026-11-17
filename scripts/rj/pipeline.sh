#!/bin/bash
# Pretrained RoboJEPA-100M on v5 (robojepa.v5.rj_train). Crash-safe: finished runs skip, others resume.
cd /home/deck/Public/RoboJEPA
PY=.venv/bin/python; R=runs_rj
run() {  # name args...
  if grep -q "^test_obj / all\|^=== trained" $R/$1.log 2>/dev/null; then echo "=== $1 already done"; return; fi
  echo "=== $1 start $(date +%T)"
  $PY -m robojepa.v5.rj_train "${@:2}" --out $R/$1 >> $R/$1.log 2>&1 && \
    { grep -q "^test_obj / all" $R/$1.log || echo "=== trained" >> $R/$1.log; }
  echo "=== $1 exit $? $(date +%T)"; grep -E "^[a-z_]+ / all" $R/$1.log
}
run scorer --mode scorer --steps 15000                                   # oracle: true ViT-G finals (cached pushes)
run zeroshot --mode eval --scorer $R/scorer/model.pt                     # pretrained predictor, no v5 training
run wm --mode wm                                                         # fine-tune on v5 frames, no labels
run jepa --mode eval --scorer $R/scorer/model.pt --wm $R/wm/model.pt
run jepa_ft --mode scorer_ft --scorer $R/scorer/model.pt --wm $R/wm/model.pt
run direct --mode direct                                                 # same backbone + head, labels only
f=0.25
run de${f}_scorer --mode scorer --steps 15000 --scene_frac $f
run de${f}_jepa_ft --mode scorer_ft --scorer $R/de${f}_scorer/model.pt --wm $R/wm/model.pt --scene_frac $f
run de${f}_direct --mode direct --scene_frac $f
echo "=== all done"
