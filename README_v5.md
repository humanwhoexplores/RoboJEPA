# v5: Panda dataset, scratch JEPA vs direct, and pretrained RoboJEPA (Meta)

Status as of 2026-11-18: the v5 dataset is generated and every experiment below has run to completion.
This file is the handoff: what exists, how to get or rebuild it, how to rerun everything, what we found,
and what is still open. Run every command from the repository root.

## 1. Layout

| path | what |
|---|---|
| `gen_v3/`, `gen_v4/` | earlier generators; v5 imports their scene specs, language, goal labels and decision picker |
| `gen_v5/` | v5 generator: v4's goal-blind design on a simulated Franka Panda (robosuite 1.5.2 assets, raw MuJoCo) |
| `robojepa/v4/` | scratch pipeline (oracle scorer, predictor, JEPA, direct, data efficiency); runs on v5 unchanged |
| `robojepa/v5/features.py` | frozen DINOv2-small + T5 text cache for v5 → `features_v5/` |
| `robojepa/v5/rj_cache.py` | frozen V-JEPA 2.1 ViT-G/384 cache for RoboJEPA → `features_rj/` |
| `robojepa/v5/rj_train.py` | pretrained RoboJEPA experiments: `--mode wm / scorer / scorer_ft / direct / eval` |
| `scripts/v5/` | `features.sh`, `main.sh` (scratch chain), `queue.sh` (whole GPU queue) |
| `scripts/rj/pipeline.sh` | RoboJEPA chain |

Not in git (see `.gitignore`): `data/`, `features_v5/` (~5 GB), `features_rj/` (~21 GB), `runs_v5/`, `runs_rj/`.
Every training script appends to `<run>.log`; a run counts as done once its log has a `test_obj / all` line
(or `=== trained`), and the chains skip finished runs, so a killed chain can simply be started again.

## 2. The v5 dataset

v4's design: goal-blind pushes, every push labeled against every goal, decisions picked from the scene only.
What v5 changes:

- **Robot.** A Franka Panda drives v3's V-shaped pusher. Pushes the arm cannot follow are shortened or replaced.
- **Images.** 256 px keyframes (start and final), plus an mp4 of every push at 5 fps.
- **Logs.** Per-step logs at 20 Hz now include `ee_state` (x, y, z, roll, pitch, yaw, gripper), joint positions `q` and tool targets.
- **Splits.** Seven splits: train / val / test_id, plus four one-factor OOD tests:

  | split | what changes |
  |---|---|
  | `test_lang` | held-out phrasings |
  | `test_cam` | held-out camera views |
  | `test_pos` | held-out object region |
  | `test_obj` | held-out shape/colour combinations |

Size and composition:

- **Scenes:** 800, seed 11. By split: train 504, val 53, test_id 56, test_lang 43, test_cam 49, test_pos 47, test_obj 48.
- **Pushes:** 64 single pushes per scene (51,200 total), plus 15,952 step-2 pushes.
- **Contact and quality:** 88% of pushes touch an object. 21 pushes have a tracking error above 15 mm; `features_v5` flags them in meta `bad`.
- **On disk:** 7.7 GB in 137k files.

Full config: `data/v5/config.json`.

```
data/v5/
  config.json  manifest.csv  instruction_rows.csv  all_goals.csv  step2_manifest.csv  step2_instruction_rows.csv
  scenes/<sid>.json          scene spec (spec["v5_split"] is the split)
  scenes/<sid>_initial.npz   start image (256,256,3)
  scenes/<sid>_labels.npz    traj_ids + labels of all 64 pushes against every goal
  trajs/<sid>_cNN.npz        one push: keyframes (after_push, final), goal_labels(_state/_strict), per-step
                             phase, pusher_pos, obj_pos/quat, contact, ee_state, q, target, video_step
  videos/<sid>_cNN.mp4       the push at 5 fps (gen_v5.video.read_mp4)
  steps/                     step-2 pushes from the after-push state
```

**Getting it.** The dataset is too large for this repo. Either:

- copy `data/v5/` from the shared archive (ask Wasiq for the link), or
- regenerate it. The run takes about 11 h on 6 CPU cores, uses the same seed and config, and is `--resume` safe:

  ```bash
  MUJOCO_GL=egl python -m gen_v5.generate --n-scenes 8 --out data/v5_mini --workers 4      # smoke test
  MUJOCO_GL=egl python -m gen_v5.generate --n-scenes 800 --out data/v5 --workers 6 --resume
  ```

## 3. Environment

Our machine:

- **Hardware:** RX 9070 XT 16 GB (ROCm), 14 GB RAM, 6 cores.
- **Python:** 3.12 in `.venv`; see `README.md` for the ROCm torch wheels.
- **Packages:** torch 2.13, transformers 5.20.2, mujoco 3.3.0, robosuite 1.5.2, numpy 1.26.4, scipy 1.17.1, av 19.0.2, timm 1.0.31, einops 0.8.2.
- **Pins:** robosuite 1.5.2 breaks on newer mujoco, so keep mujoco at 3.3.x.

Any CUDA GPU with at least 16 GB should work.

RoboJEPA needs Meta's repo and two checkpoints:

1. **Clone Meta's repo:**
   ```bash
   git clone https://github.com/facebookresearch/robo_jepa ~/Public/robo_jepa   # or set ROBO_JEPA=/path
   ```
2. **Download the predictor** (0.4 GB) and the matching config:
   ```bash
   mkdir -p ~/Public/robo_jepa/checkpoints/wm-100m
   wget -P ~/Public/robo_jepa/checkpoints/wm-100m https://dl.fbaipublicfiles.com/robojepa/wm/wm-100m/cooldown_100m_150k.pth.tar
   cp ~/Public/robo_jepa/configs/released/wm-100m/config.json ~/Public/robo_jepa/checkpoints/wm-100m/
   ```
3. **Download the encoder** (30 GB) and keep only its frozen target encoder in bf16 (3.7 GB):
   ```bash
   wget https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitG_384.pt
   python -c "from robojepa.v5.rj_cache import extract_encoder as e; e('vjepa2_1_vitG_384.pt', \
     '$HOME/Public/robo_jepa/checkpoints/vjepa2_1_vitG_384_target_bf16.pt')"
   ```
   You can delete the 30 GB file afterwards.

Memory pitfalls we hit with the encoder:

- **mmap fails.** With 14 GB of RAM, `torch.load(..., mmap=True)` of the 30 GB file failed until we set `torch.serialization.set_default_mmap_options(mmap.MAP_SHARED)`.
- **OOM kill on build.** Building the ViT-G in fp32 on the CPU gets OOM-killed. `rj_cache.load_encoder` builds it directly on the GPU in bf16.
- **Dtype mismatch.** RoPE upcasts q/k, so the encoder must run under autocast.

## 4. Reproducing

```bash
scripts/v5/features.sh             # DINOv2 + text cache -> features_v5 (~5 GB)
python -m robojepa.v5.rj_cache --encoder ~/Public/robo_jepa/checkpoints/vjepa2_1_vitG_384_target_bf16.pt --out features_rj   # 21 GB
scripts/rj/pipeline.sh > runs_rj/pipeline.log      # RoboJEPA chain, ~16 h (direct runs are 5.5 h each)
scripts/v5/main.sh >> runs_v5/queue.log            # scratch chain, ~9 h
```

The GPU holds only one large job at a time; the chains run their jobs one after another.

### Scratch models (DINOv2-small, from `robojepa/v4`)

| run | what |
|---|---|
| `oracle2` | two-frame token scorer (start + final frame, T5 token text) on the *true* final frames: the ceiling |
| `pred` | predictor (MSE) rolls the start tokens through the push; JEPA = frozen `oracle2` on predictions |
| `ft2` | JEPA with the scorer fine-tuned on predicted frames |
| `direct` | direct token classifier (start frame + push + text → label), no world model |
| `direct_reg` | regularized direct (`--dropout 0.1 --token_drop 0.25 --wd 0.1`) |
| `de0.25_*` | labels from 25% of training scenes; the predictor is still trained on all scenes (unlabeled video is cheap) |

### Pretrained RoboJEPA 100M (`robojepa/v5/rj_train.py`, frozen V-JEPA 2.1 ViT-G)

**Inputs.**
- **Frames:** 16×16 tokens × 1664.
- **Embodiment:** `franka`; one world-model step is 0.8 s, so a push is 4 steps: push start → +0.8 s → +1.6 s → settled final.
- **State:** end-effector pose in the robot base frame, converted to DROID's gripper convention (`R_tool·Rx(π)`, so roll ≈ ±π). Actions are the state differences.
- **Cached pushes:** only 24 of the 64 pushes per train scene (8 per val scene) have full sequences cached, because of disk space.

**Runs.**

| run | what |
|---|---|
| `scorer` | `NormScorer` (per-channel standardized ViT-G tokens) on true finals of cached pushes |
| `zeroshot` | Meta's predictor as released + that scorer |
| `wm` | predictor fine-tuned on v5 frames (L1, teacher forcing + rollout), no labels |
| `jepa`, `jepa_ft` | fine-tuned predictor + frozen / fine-tuned scorer |
| `direct` | same predictor and head trained end to end on labels (backprop through the rollout) |
| `de0.25_*` | 25% labels, predictor from `wm` |

## 5. Results

Top-1 accuracy among solvable decisions. `rule` = no-physics reference rule (pick a push that touched the
named object and heads toward the goal); random ≈ .04. Full per-type tables are in each run's log.

**Scratch, full labels**

| split | oracle | JEPA | JEPA ft | direct | direct reg | rule |
|---|---|---|---|---|---|---|
| val | .866 | .789 | .804 | .789 | .746 | .641 |
| test_id | .871 | .800 | .808 | .854 | .742 | .646 |
| test_lang | .566 | .537 | .486 | .440 | .320 | .543 |
| test_cam | .741 | .481 | .492 | .534 | .487 | .577 |
| test_pos | .799 | .742 | .768 | .753 | .675 | .531 |
| test_obj | .804 | .771 | .793 | .704 | .609 | .603 |

**Scratch, 25% labels**

| split | oracle | JEPA | JEPA ft | direct reg |
|---|---|---|---|---|
| val | .502 | .440 | .545 | .464 |
| test_id | .537 | .471 | .546 | .529 |
| test_lang | .183 | .200 | .263 | .331 |
| test_cam | .376 | .259 | .270 | .360 |
| test_pos | .454 | .320 | .433 | .448 |
| test_obj | .374 | .346 | .453 | .464 |

**Pretrained RoboJEPA 100M**

| split | zero-shot | JEPA | JEPA ft | direct | JEPA ft 25% | direct 25% |
|---|---|---|---|---|---|---|
| val | .053 | .254 | .287 | .167 | .225 | .177 |
| test_id | .029 | .183 | .237 | .158 | .258 | .146 |
| test_lang | .040 | .137 | .183 | .149 | .154 | .109 |
| test_cam | .032 | .069 | .079 | .111 | .074 | .159 |
| test_pos | .041 | .160 | .144 | .098 | .186 | .072 |
| test_obj | .045 | .207 | .212 | .084 | .218 | .089 |

What we read from this:

1. **Scratch models, full labels.** JEPA and direct are close. Direct is ahead on test_id and test_cam; JEPA is ahead on test_lang, test_pos and test_obj.
2. **Scratch models, 25% labels.** JEPA with the fine-tuned scorer leads in distribution. The regularized direct model ties or wins on the OOD splits. The v4 gap (.81 vs .20 at 25%) was mostly an unregularized direct baseline.
3. **Camera and language shift are unsolved.** Every model is at or below the no-physics rule on test_cam and test_lang.
4. **Pretrained RoboJEPA is much weaker than scratch DINO.**
   - Zero-shot is at chance.
   - After fine-tuning, the predictor's rollout L1 is 0.083, worse than simply copying the start frame (0.064), yet its predictions still rank pushes usefully.
   - Even on true finals, the ViT-G scorer underfits: cached-push AUC is about 0.79 (the DINO oracle reaches .986 decision AUC on val).

   The frozen V-JEPA features seem to carry too little colour and shape information for this task. JEPA and direct share the same features, so the comparison between them is still fair.

## 6. Open items

- **Unregularized direct at 25% on v5** (`robojepa.v4.pilot --model direct --scene_frac 0.25`). Without it, the v4 and v5 data-efficiency results can't be compared like for like.
- **10% labels on v5**, as in v4.
- **RoboJEPA world model never beat copy-start.** Try a residual/delta prediction head, or the 300M checkpoint (1.2 GB).
- **The RoboJEPA scorer sees only 24/64 cached pushes per train scene.** A bigger cache needs about 35 GB more disk.
- **Camera generalization** (`test_cam`) for both families: augmentation, or several training views per scene.
- **Seeds.** Every number above is a single seed; with 175–240 solvable decisions per split, differences of a few points are noise.
