# Generator v4: goal-blind candidates

v4 answers Wasiq's feedback on v3 ("goal-blind candidates, so there is less bias in the dataset").
In v3, 28 of the 32 pushes per scene were computed from the scene's 4 instructions, and a rule with no
physics ("touch the named object, point at the target") picked a working push 64% of the time.
In v4 the pushes never see a goal:

1. **Same scenes as v3.** Seed 7 and v3's scene code rebuild the exact v3 scenes (objects, layout,
   camera, lighting, split). `gen_v4` imports the unchanged parts from `gen_v3`, which stays untouched.
2. **64 goal-blind pushes per scene** (`sampler.py`): 56 object pushes split evenly over the objects
   (directions evenly spaced with a random rotation, distances 3 to 25 cm stratified, random contact
   offset) plus 8 random pusher lines. The sampler has no goal argument. The list is shuffled.
3. **Every push labeled against every goal** (about 115 per scene), three ways (`labels.py`):
   `default` = v3's rule + the named object moved at least 2 cm; `state` = exactly v3's rule;
   `strict` = default + reference objects moved at most 3 cm.
4. **Decisions picked from the scene only** (`decisions.py`): v3's 4 instructions (same words, a paired
   subset) + 4 "open" goals without v3's straight-path filter. The pick never reads the labels.
5. **Two-push decisions, both steps goal-blind:** for each of v3's 2 compound goals, if a start push
   solves the first part, one such push is picked at random and 32 new goal-blind pushes run from its
   settled end state.
6. **The start image is stored once per scene** (`scenes/<id>_initial.npz`); push files keep
   `after_push` and `final`.

## Run (from the folder that contains `gen_v3/` and `gen_v4/`)

```bash
python -m pytest gen_v3/tests gen_v4/tests -q
python -m gen_v4.generate --n-scenes 40 --out data/v4_mini --workers 6     # check run, ~10 min
python -m gen_v4.validate --data data/v4_mini --v3 data/v3                 # must print PASSED
python -m gen_v4.generate --n-scenes 1000 --out data/v4 --workers 6        # full run; --resume after a stop
python -m gen_v4.validate --data data/v4 --v3 data/v3
```

Options: `--pushes` (64), `--lines` (8), `--step2` (32; 0 turns two-push decisions off), plus v3's
`--no-arm`, `--no-visual`, `--no-distractors`, `--cameras`, `--held-cameras`, `--held-share`.
On headless Linux, prefix with `MUJOCO_GL=egl` (or `osmesa`).

## Data format (what a loader needs)

- `manifest.csv`: one row per start push. No goal fields: `candidate_type` is only `object_push`,
  `object_push_blocked` (fell back to a random line) or `random_line`. `in_eval32` marks a fixed random
  half of the pushes for a 32-way ranking the same size as v3's.
- `instruction_rows.csv`: one row per push x eval decision: v3's columns plus `source`
  (`v3_focus` / `open`), `straight_path`, `label`, `label_state`, `label_strict`, `in_eval32`.
- `all_goals.csv`: every goal of every scene with an instruction text and its positive counts.
- `scenes/<id>_labels.npz`: `default`, `state`, `strict` matrices (pushes x goals), with `goal_ids` and
  `traj_ids`. Train on any goal from here.
- `scenes/<id>_initial.npz`: `initial` (224, 224, 3), the start image shared by all of the scene's pushes.
- `trajs/<id>_cNN.npz`: `keyframe_names` = [after_push, final], `goal_labels` (default),
  `goal_labels_state`, `goal_labels_strict`, and v3's per-step logs.
- `steps/<id>_kJ_sNN.npz` + `step2_manifest.csv` + `step2_instruction_rows.csv`: step 2 of two-push
  decision J. Its start image is the `final` frame of `step1_traj_id`. `label` is the whole compound
  goal; `part2_label` only the second part.
- `scenes/<id>.json`: spec (with `v3_scene_id`), start positions, goals (with `in_view`,
  `straight_path`), `eval_goals`, `compound`, `instructions`, `templates`, `step2`.

Compared with v3, a loader changes in two places: read the start image from `scenes/<id>_initial.npz`,
and do not expect `focus_goal`, `own_goal_success` or `intended` (they described the aiming that v4 removed).
