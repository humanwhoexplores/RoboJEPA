#!/bin/bash
# v6 dataset in 6 batches of 200 scenes (1,200 total), as it was produced on an 8 GB Mac:
# generate -> validate (two 100-scene halves, see split_view.py) -> zip (stored, verified)
# -> drop the raw batch -> optionally move the zip to a synced folder (e.g. Google Drive).
# Resume-safe: rerun the same command after a crash, sleep or reboot.
#
#   cd <repo root> && nohup caffeinate -ims scripts/v6/run_batches.sh > gen_v6_batches.log 2>&1 &   # macOS
#   cd <repo root> && nohup scripts/v6/run_batches.sh > gen_v6_batches.log 2>&1 &                   # Linux
#
# Env: PY (python, default python), WORKERS (default 4; 6 swapped heavily on 8 GB RAM),
#      DATA (default data/v6), ZIPS (default data/v6_zips), DEST (optional folder for finished zips).
# Batches use --first, so scene ids and seeds equal one 1,200-scene run (seed 11, scenes 0-1199).
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=. MUJOCO_GL=${MUJOCO_GL:-$([ "$(uname)" = Darwin ] && echo cgl || echo egl)}
PY=${PY:-python}; WORKERS=${WORKERS:-4}; DATA=${DATA:-data/v6}; ZIPS=${ZIPS:-data/v6_zips}; DEST=${DEST:-}
mkdir -p "$DATA" "$ZIPS"; ZIPS=$(cd "$ZIPS" && pwd)
free_gb() { df -Pk . | awk 'NR==2 {print int($4/1048576)}'; }
to_dest() {
  [ -n "$DEST" ] || return 0
  mkdir -p "$DEST"
  for z in "$ZIPS"/*.zip; do [ -f "$z.done" ] && mv "$z" "$DEST/" && mv "$z.done" "$z.moved" && echo "moved $(basename "$z") to $DEST"; done
}
for k in 0 1 2 3 4 5; do
  part=$(printf "part_%02d" $k); zipf="$ZIPS/RoboJEPA_gen_v6_$part.zip"
  if [ -f "$zipf.done" ] || [ -f "$zipf.moved" ]; then continue; fi
  to_dest
  while [ "$(free_gb)" -lt 7 ]; do echo "$(date) waiting: only $(free_gb) GB free"; to_dest; sleep 600; done
  echo "$(date) generating $part"
  $PY -m gen_v6.generate --first $((200*k)) --n-scenes 200 --workers "$WORKERS" --out "$DATA/$part" --resume || exit 1
  $PY scripts/v6/split_view.py "$DATA/$part" 2 || exit 1
  : > "$DATA/$part.validate.log"
  for h in 0 1; do
    v="$DATA/_val_${part}_h$h"
    $PY -m gen_v6.validate --data "$v" --determinism 2 --resample 1 >> "$DATA/$part.validate.log" 2>&1 \
      || { echo "$(date) validation FAILED for $part half $h"; exit 1; }
    cp "$v/validation_report.json" "$DATA/$part/validation_report_half$h.json"; rm -rf "$v"
  done
  (cd "$DATA" && zip -r -0 -q "$zipf" "$part" "$part.validate.log") && unzip -tq "$zipf" \
    && touch "$zipf.done" && rm -rf "${DATA:?}/$part" && echo "$(date) $part zipped"
done
to_dest
echo "$(date) ALL DONE"
