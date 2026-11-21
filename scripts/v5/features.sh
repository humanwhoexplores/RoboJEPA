#!/bin/bash
set -e
cd /home/deck/Public/RoboJEPA
.venv/bin/python -m robojepa.v5.features --data data/v5 --code . --out features_v5
.venv/bin/python -m robojepa.v4.texts --data data/v5 --code . --features features_v5
echo FEATURES_DONE
