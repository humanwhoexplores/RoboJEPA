# v6: gripper actions, hidden physics and two-step decisions

Status as of 2026-11-22: the generator is built and a 42-scene check (7 scenes per family) passed validation.
The full run has not started yet. Run every command from the repository root.

v6 answers the professor's feedback on v5 ("add diversity via actions, not just objects and distractors, like
pick up and drop"). It keeps v4/v5's design: candidates are drawn goal-blind, every rollout is labeled against
every goal of its scene, and decisions are picked from the scene alone, never from the rollouts' results.

## 1. What changes from v5

| | v5 | v6 |
|---|---|---|
| robot | Panda holding a rigid V pusher | Panda with robosuite's parallel gripper (pushes with closed fingers) |
| actions | push | 14 kinds (below), 64 per scene, mix set per scene family |
| scenes | blocks, zones, distractors | 6 families with trays, bins, a bowl, a platform, a ramp, a wall, a seesaw, dominoes, a drawer cabinet and a lidded box; robosuite everyday objects (can, bottle, milk, cereal, bread, lemon) and a hammer |
| physics | fixed | hidden mass (x0.3 to x3), friction (0.2 to 1.2) and bounce per object; look-alike pairs that differ only in mass or friction (24% of scenes) |
| goals | v3's 7 relation types | 22 types: v3's plus on_top, off_of, inside, count_in, upright, lying, rotated, off_table, touching, moved_untouched, drawer_open/closed, lid_open/closed, hidden |
| decisions | 8 single + 2 two-push | 8 single (one of them a probe on a look-alike), 2 compound, a two-placement tower, and probe-then-act; 12 step-2 rollouts each |
| splits | 7 | 9 scene splits plus the test_act label mask (section 4) |

Actions (`gen_v6/sampler.py`, `ACTIONS`):

| group | actions |
|---|---|
| slides | push, pull (hook behind and draw toward the base), sweep (open fingers), poke (fast strike; the object coasts), topple (push high on a tall object) |
| grasps | place (free spot, another object's top, or a container), drop (released 5 to 20 cm up), stack, unstack, twist (turn in place), stand_up (lift a lying tube, can or bottle onto its end) |
| fixtures | drawer (pull the handle out or push it shut), lid (lift the knob and swing it open, or push it over shut) |
| throw | toss (swing and release mid-stroke) |

The hammer is a normal object (picked, placed, pushed, tossed). A hammer-rake action was dropped: the level grip
puts the handle on the object it should rake.

Physics: MuJoCo 3.3.0 with elliptic friction cones and impratio 10. With v5's pyramidal cones a held ball crept
down between the pads and fell out during the lift; now placed objects land within 1 to 2 mm of the target.

## 2. Running it

```bash
# smoke test, about 13 min on 4 cores
MUJOCO_GL=egl python -m gen_v6.generate --n-scenes 12 --out data/v6_mini --workers 4
# full run: resume-safe, so a killed run can simply be started again
MUJOCO_GL=egl python -m gen_v6.generate --n-scenes 1200 --out data/v6 --workers 6 --resume
# checks: label recomputation, determinism, sampler reproduction, leakage between splits; writes
# validation_report.json and contact_sheet.png into the data folder and exits non-zero on failure
MUJOCO_GL=egl python -m gen_v6.validate --data data/v6 --determinism 8 --resample 4
```

Use `MUJOCO_GL=osmesa` where EGL is missing. `--no-stage2` leaves out the tools family, drawer and lid.

Cost measured in the cloud check (4 cores): about 4.2 min of one core per scene including step 2 (3 s per
rollout), and 11 MB per scene. So 1,200 scenes is about 84 core-hours (about 14 h on 6 cores) and about 13.5 GB.

Packages: mujoco 3.3.0, robosuite 1.5.2 (installed with `--no-deps`; its assets only), numpy, scipy, av, pillow,
pandas (validate only).

## 3. Output

Same layout as v5, so v5's feature scripts read it. New columns and arrays are marked.

```
data/v6/
  config.json                 every setting, including the action mix, splits, held-out pairs and physics
  manifest.csv                one row per rollout: action, action_id, sampler_object, sampler_target, params,
                              v0..v11 (fixed-length action vector), touched, held, track_err_mm, caption, ...
  instruction_rows.csv        rollout x decision goal: instruction, template, label(_state/_strict),
                              action, act_heldout (new)
  all_goals.csv               every goal of every scene with its instruction and positive counts
  step2_manifest.csv          step-2 rollouts, with decision, source and step1_traj_id
  step2_instruction_rows.csv
  scenes/<sid>.json           spec (spec["v6_split"] is the split), goals, decisions, multi_step, step2 info
  scenes/<sid>_initial.npz    start image (256, 256, 3)
  scenes/<sid>_labels.npz     64 rollouts x all goals: default, state, strict, act_heldout (new)
  trajs/<sid>_cNN.npz         one rollout: keyframes, goal_labels(_state/_strict), per-step obj_pos/quat,
                              ee_state (gripper value = real opening), q, target, contact, and new: grip, held,
                              pair (object-object contact), table, fx_contact, fx_state (drawer slide, lid angle)
  videos/<sid>_cNN.mp4        5 fps
  steps/                      step-2 rollouts from the step-1 end state
```

Labels (`gen_v6/labels.py`): `state` = the goal holds on the last settle steps; `default` = state and the named
object moved at least 2 cm; `strict` = default and no reference object moved more than 3 cm.

## 4. Splits

Scene splits (`gen_v6/scene_spec.py`, `SPLIT_P`): train 58%, val 6%, test_id 6%, and one-factor tests of 5% each:
test_lang (held-out phrasings and words), test_cam (held-out cameras), test_pos (held-out region), test_obj
(held-out shape/colour pairs), test_phys (one object with mass or friction outside the training range),
test_family (a container at the foot of the ramp, a layout no training scene has).

test_act (`gen_v6/splits.py`) is a label mask, not a scene split: the pairs (drop, in_zone), (drop, on_top) and
(place, between) must not be used as evidence in train or val, whether they succeeded or not. Each held-out
action still appears for other goals and each goal type is still reached by other actions. Masked entries are
flagged `act_heldout` in the instruction rows and in each scene's label file.

## 5. The 42-scene check

Validation passed: 0 label mismatches, 0 blow-ups, determinism within 1e-8 m.

- 2,688 rollouts plus 600 step-2 rollouts; 90% touch an object, 46% hold one.
- Rollouts by action: push 831, place 484, drop 323, poke 234, stack 153, topple 123, toss 107, sweep 106,
  pull 96, lid 74, unstack 54, twist 51, drawer 36, stand_up 16.
- Single decisions: 37% solvable by at least one of the 64 rollouts, against 2% for a random rollout.
- Probe decisions: 33% solvable. Two-step decisions solvable at step 2: compound 21%, probe 25%, tower 0 of 4.
- 138 held-out (action, goal) positives that the test_act mask hides from training.
- Lid open/shut works in nearly every lid rollout; drawer open in 13 of 18, shut in 9 of 17 (the shut stroke
  has since been lengthened by 1 cm).
