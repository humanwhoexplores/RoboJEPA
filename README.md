# RoboJEPA

Language-guided push selection with a JEPA-style world model vs. a direct classifier
(CSE590 project). A Panda arm in robosuite pushes a cube toward one of two colored regions;
instructions name the target by side ("left"/"right") or color ("blue"/"green"), and the
colors swap between scenes.

## Pipeline

| step | module | output |
|---|---|---|
| 1. collect | `robojepa.collect` | `data/{split}/scene_*.npz`: start image, candidate push sequences, image after each push, outcome region |
| 2. features | `robojepa.features` | `features/`: frozen DINOv2-S patch tokens (256x384, fp16) per frame + MiniLM instruction embeddings |
| 3. train | `robojepa.train` | `runs/`: predictor (z, push -> z'), scorer (z_T, text -> success), direct classifier (z_0, pushes, text -> success) |
| 4. evaluate | `robojepa.evaluate` | `results/`: offline action-ranking comparison |
| 5. figures | `robojepa.figures` | `figures/` |

`scripts/run_midterm.sh` runs steps 2-5, including the data-efficiency sweep (50/100/200/400 training scenes).

### Splits
- `train` / `val`: 400 / 50 scenes, 16 random candidates each (70% single push, 30% two pushes)
- `test_id`: 100 new scenes from the training cube-position range
- `test_ood`: 60 scenes with the cube starting in a held-out x range
- Test scenes use a fixed candidate set: 8 directions x 3 lengths single pushes + 24 two-push sequences
- Held-out instruction paraphrases are only used at evaluation

### Methods compared (same candidates, same scorer architecture)
- **Random**: uniform choice among candidates (floor)
- **Direct classifier**: scores (start features, action sequence, instruction) without predicting the future
- **JEPA**: predictor rolls the start features forward through each push; the scorer scores the predicted terminal features
- **Oracle**: the same scorer on the *true* terminal features (ceiling; isolates world-model error from scorer error)

Candidates are placed around a noisy cube-position estimate (std 1 cm), standing in for a detector.
No method sees object coordinates when scoring.

## Setup (Ubuntu 24.04, ROCm 7.2, RX 9070 XT)

```bash
uv venv --python 3.12 .venv
B=https://wheels.vllm.ai/rocm/db9527a46873454610df6dbedf79a36d6bf1a7f6
uv pip install --python .venv "torch @ $B/torch-2.13.0%2Bgit733fca1-cp312-cp312-manylinux_2_39_x86_64.whl" \
    "torchvision @ $B/torchvision-0.28.0%2B8fb8771-cp312-cp312-manylinux_2_39_x86_64.whl"
uv pip install --python .venv "mujoco==3.3.0" robosuite transformers numpy pillow matplotlib tqdm
```

The vLLM ROCm wheels link against the system ROCm in `/opt/rocm` (~2 GB), unlike the
pytorch.org ROCm wheels, which bundle it (~6 GB download). robosuite 1.5.2 fails with the latest mujoco (3.17); 3.3.0 works.
Rendering is headless via `MUJOCO_GL=egl`.

```bash
.venv/bin/python -m robojepa.collect --out data --workers 5   # ~30 min
scripts/run_midterm.sh
```
