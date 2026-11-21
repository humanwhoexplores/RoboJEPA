# Generator v3 (milestones 3 to 5, plus a robot arm)

Builds on gen_v2 (unchanged, kept so the v2 dataset stays reproducible). Plain MuJoCo, no robosuite.

- 3 to 6 objects from 7 shapes (cube, cylinder, triangle block, L-block, T-block, hexagonal prism, bar),
  2 sizes, 6 colors; 3 or 4 colored zones; 0 to 2 neutral distractors that are never named
- 7 goal types: `in_zone`, `left_of`, `right_of`, `in_front_of`, `behind`, `next_to`, `between`,
  plus compound `and` goals for two-push plans
- per scene: 32 single pushes (ideal x3, short, over, miss per focus goal; wrong_object; random) and
  8 two-push plans (closed-loop, open-loop, half, random), all from the same saved start state
- 5 training cameras, 2 held-out cameras for half of the test scenes; randomized table, floor and lights
- a visual-only robot arm holding the V pusher (posed by IK, no collisions: physics identical with or without it)
- templates with synonyms, relational references, held-out test templates and shape words, and a hindsight
  caption for every rollout

## Run (from the folder that contains `gen_v3/`)

```bash
python -m pytest gen_v3/tests -q                                       # 12 checks, ~5 s
mjpython -m gen_v3.watch --scenes 1 --only ideal                       # watch pushes live (macOS)
python -m gen_v3.generate --n-scenes 1000 --out data/v3 --workers 6    # ~60 min, ~3.2 GB; add --resume after a stop
python -m gen_v3.validate --data data/v3                               # must print PASSED
python -m gen_v3.report_assets --data data/v3 --out gen_v3_report_assets
```

Options: `--no-arm`, `--no-visual`, `--no-distractors`, `--plans 0`, `--cameras`, `--held-cameras`, `--held-share`.
On headless Linux, prefix with `MUJOCO_GL=egl`. Data format and known limits: `gen_v3_report_assets/notes.md`.
