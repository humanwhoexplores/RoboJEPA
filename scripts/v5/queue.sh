#!/bin/bash
# GPU queue: finish oracle2 (already running), cache RoboJEPA features, RoboJEPA runs, then the remaining scratch runs.
cd /home/deck/Public/RoboJEPA
while kill -0 3332903 2>/dev/null; do sleep 60; done
echo "=== oracle2 finished $(date +%T)"
if [ ! -f features_rj/test_obj_done ]; then
  .venv/bin/python -m robojepa.v5.rj_cache --encoder ~/Public/robo_jepa/checkpoints/vjepa2_1_vitG_384_target_bf16.pt --out features_rj >> runs_v5/rj_cache.log 2>&1
  echo "=== rj_cache exit $? $(date +%T)"
fi
[ -f features_rj/test_obj_done ] && runs_rj/pipeline.sh > runs_rj/pipeline.log 2>&1
runs_v5/main.sh
