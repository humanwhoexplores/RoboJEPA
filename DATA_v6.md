# v6 dataset: where it is and how it was made

The generator is `gen_v6/` (design and output format: [README_v6.md](README_v6.md)). This page records
the actual 1,200-scene run (Nov 23-25, 2026) and what to know before using it.

## Download

Google Drive folder **RoboJEPA_gen_v6_data**:
<https://drive.google.com/drive/folders/1GXtkvv_Tfd84Lwj5YYW6QX8MVU5IIYCT> (ask Krishan for access).

| zip | scenes | rollouts | step-2 rollouts | bytes |
|---|---|---|---|---|
| RoboJEPA_gen_v6_part_00.zip | 0-199 | 12,800 | 540 | 2,856,442,399 |
| RoboJEPA_gen_v6_part_01.zip | 200-399 | 12,800 | 588 | 2,910,364,406 |
| RoboJEPA_gen_v6_part_02.zip | 400-599 | 12,800 | 420 | 2,821,774,645 |
| RoboJEPA_gen_v6_part_03.zip | 600-799 | 12,800 | 456 | 2,824,741,200 |
| RoboJEPA_gen_v6_part_04.zip | 800-999 | 12,800 | 588 | 2,902,886,076 |
| RoboJEPA_gen_v6_part_05.zip | 1000-1199 | 12,800 | 480 | 2,846,922,349 |
| **total** | **1,200** | **76,800** | **3,072** | **~17.2 GB** |

Each zip is stored (no compression; the npz/mp4 inside are already compressed) and unpacks to
`part_XX/` with the usual layout (`trajs/ scenes/ steps/ videos/`, `manifest.csv`,
`instruction_rows.csv`, `all_goals.csv`, `step2_manifest.csv`, `step2_instruction_rows.csv`,
`config.json`), plus `validation_report_half0.json`, `validation_report_half1.json` and
`part_XX.validate.log`.

Merging: the batches were made with `--first 200*k --n-scenes 200` and seed 11, so scene ids
(`v6s11_00000` ... `v6s11_01199`) and every rollout are identical to one 1,200-scene run. Unzip all
six into one folder and concatenate each CSV (one header). Per-file arrays need no changes.

## Settings

`gen_v6.generate` defaults: 64 rollouts per scene (4 lines), 12 step-2 rollouts per multi-step
decision, stage-2 families on (tools, drawer, lid), 256 px images, 5 fps video, 20 Hz control.
MuJoCo 3.3.7, robosuite 1.5.2, numpy 1.26.4, Python 3.11, macOS (MUJOCO_GL=cgl), 4 workers.

## Validation

Every batch passed `gen_v6.validate --determinism 2 --resample 1` ("passed": true, no errors) on
both halves. Batches were validated as two 100-scene views (`scripts/v6/split_view.py`) because the
validator then kept every trajectory file open and a 200-scene batch (12,800 files) exceeds macOS's
10,240 open files per process. The checks are per scene, so the halves cover the whole batch.

`gen_v6/validate.py` has since been fixed to keep only each rollout's final state, so a whole batch
(or all 1,200 scenes) can be validated in one call and the split view is optional. The checks
themselves are unchanged; `scripts/v6/run_batches.sh` still validates in halves, which also works.

## Code change during the run (read this)

Scene 65 crashed the generator with `IndexError` in `gen_v6/language.py` `refer()`:
`instruction()` calls `refer(goal[k])` for k in a, b, c, but for **in_zone** goals `b` is a zone
index, not an object index. When the zone index was >= the number of objects (tools scenes with
2 objects), `objs[b]` was out of range. 35 of the 1,200 scenes hit it.

Fix applied (commit on this branch): `refer()` returns `None` when `i >= len(objs)`. In those 35
scenes the affected in_zone goals get no instruction text (the existing `None` path). Every scene
that never reaches that line is byte-identical to the unpatched generator.

**Known, not fixed:** in every scene, in_zone goals still pass the zone index through `refer()` as if
it were an object. So some in_zone instructions are dropped (when that "object" is not uniquely
referable) and the RNG draws for later goals shift. The clean fix is to skip `b` for in_zone in
`language.instruction()`; it changes instruction text, so it needs a regenerated dataset.

## Reproducing the run

```bash
pip install mujoco==3.3.7 robosuite==1.5.2 numpy==1.26.4 scipy pandas pillow av   # Python 3.11
scripts/v6/run_batches.sh              # 6 x 200 scenes -> data/v6_zips/; WORKERS, DEST, PY via env
```

Throughput on an 8 GB M-series Mac with 4 workers: about 100-140 s per scene (about 6-7.5 h per
batch); 6 workers swapped heavily. Keep a laptop on AC power with the lid open: caffeinate cannot
stop clamshell sleep on battery.
