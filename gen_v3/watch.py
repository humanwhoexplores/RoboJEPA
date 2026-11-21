"""Watch v3 scenes and their candidate pushes live in the MuJoCo viewer.

    mjpython -m gen_v3.watch                               # 3 scenes x all 32 candidates
    mjpython -m gen_v3.watch --scenes 1 --only ideal over --speed 0.5

On macOS the viewer needs mjpython (ships with mujoco), not python.
Each candidate prints its type, its focus goal's instruction and whether it succeeded.
"""
import argparse
import time

import mujoco.viewer
import numpy as np

from gen_v3.candidates import make_candidates
from gen_v3.env import CONTROL_HZ, SceneEnv
from gen_v3.language import instruction
from gen_v3.primitives import execute_push
from gen_v3.scene_spec import sample_scene
from gen_v3.tasks import enumerate_goals, label_goals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--first", type=int, default=0, help="index of the first scene")
    ap.add_argument("--only", nargs="*", help="candidate types to show, e.g. ideal miss")
    ap.add_argument("--speed", type=float, default=1.0, help="1.0 = real time")
    ap.add_argument("--no-arm", dest="arm", action="store_false")
    args = ap.parse_args()

    env = SceneEnv(arm=args.arm)
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        base_step = env.step

        def step(*a, **k):  # sync the viewer after every control step
            base_step(*a, **k)
            env.pose_arm()
            viewer.sync()
            time.sleep(1.0 / CONTROL_HZ / args.speed)

        for idx in range(args.first, args.first + args.scenes):
            spec = sample_scene(args.seed, idx, visual=True, distractors=True)
            env.step = base_step  # settle at full speed
            env.set_scene(spec)
            viewer.sync()
            env.pose_arm()
            viewer.sync()
            init_pos = env.object_poses()[0]
            state0 = env.get_state()
            goals = enumerate_goals(spec, init_pos)
            focus, cands = make_candidates(spec, goals, init_pos, np.random.default_rng([args.seed, idx, 1]),
                                           env.distractor_poses())
            lang_rng = np.random.default_rng([args.seed, idx, 2])
            texts = {g["goal_id"]: instruction(g, spec, spec["split"], lang_rng, init_pos)[0] for g in focus}
            ids = [g["goal_id"] for g in goals]
            print(f"\nscene {spec['scene_id']} ({spec['split']}): {len(spec['objects'])} objects, "
                  f"{len(spec['zones'])} zones")
            for k, cand in enumerate(cands):
                if args.only and cand["candidate_type"].split("_blocked")[0] not in args.only:
                    continue
                if not viewer.is_running():
                    return
                env.set_state(state0)
                viewer.sync()
                env.step = step
                log = execute_push(env, cand["x0"], cand["y0"], cand["theta"], cand["length"])
                env.step = base_step
                labels = label_goals(goals, log["obj_pos"], log["phase"], spec)
                own = cand["focus_goal"]
                result = ("SUCCESS" if labels[ids.index(own)] else "fail") if own else "-"
                print(f"  c{k:02d} {cand['candidate_type']:<22} {texts.get(own, '(no focus goal)'):<70} {result}")
                time.sleep(0.5 / args.speed)


if __name__ == "__main__":
    main()
