"""Watch pushes live in the MuJoCo viewer.

    .venv/bin/mjpython -m scripts.watch                  # 3 random scenes, all 8 candidate pushes each
    .venv/bin/mjpython -m scripts.watch --scenes 1 --only ideal_left ideal_right
    .venv/bin/mjpython -m scripts.watch --speed 0.5      # half speed

On macOS the viewer needs `mjpython` (installed with mujoco) instead of `python`.
Mouse: left-drag rotates, right-drag pans, scroll zooms. Close the window to stop.
The terminal prints each push's candidate type and outcome.
"""
import argparse
import time

import numpy as np

from data.generate import CANDIDATE_TYPES, make_candidates, sample_scene
from envs.push_env import PushEnv, make_env
from envs.push_primitive import execute_push, label_outcome


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--only", nargs="*", choices=CANDIDATE_TYPES, help="candidate types to show")
    ap.add_argument("--speed", type=float, default=1.0, help="1.0 = real time")
    ap.add_argument("--no-viewer", action="store_true", help="run without a window (for testing)")
    args = ap.parse_args()

    if args.no_viewer:
        env = make_env()
    else:
        # robosuite's offscreen renderer opens a GLFW window off the main thread, which
        # crashes under mjpython on macOS (NSWindow must be on the main thread); the viewer doesn't need it.
        env = PushEnv(robots="Panda", has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
                      use_object_obs=True, control_freq=20, horizon=100000, ignore_done=True,
                      hard_reset=False, initialization_noise=None)
    env.reset()
    dt = 1.0 / env.control_freq
    viewer = None
    if not args.no_viewer:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(env.sim.model._model, env.sim.data._data)
        viewer.cam.lookat[:] = [0.0, 0.0, 0.8]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.3, 180.0, -35.0

    # Redraw the window after every control step, paced to real time.
    raw_step = env.step

    def step(action):
        t0 = time.time()
        out = raw_step(action)
        if viewer is not None:
            if not viewer.is_running():
                raise SystemExit
            viewer.sync()
            time.sleep(max(0.0, dt / args.speed - (time.time() - t0)))
        return out

    env.step = step

    for i in range(args.scenes):
        scene = sample_scene(args.seed, i)
        for cand in make_candidates(scene, args.seed):
            if args.only and cand["candidate_type"] not in args.only:
                continue
            env.reset()
            colors = env.set_scene(scene["cube_xy"], scene["cube_yaw"], scene["left_color"])
            if viewer is not None:
                viewer.sync()
                time.sleep(0.5 / args.speed)
            log = execute_push(env, cand["theta"], cand["dist"], cand["offset"])
            lab = label_outcome(log["cube_pos"], log["contact"], log["phase"], np.array(scene["cube_xy"]))
            print(f"scene {i}  left={colors['left']:<4}  {cand['candidate_type']:<11}  "
                  f"left: {lab['failure_mode_left']:<14}  right: {lab['failure_mode_right']}", flush=True)
            if viewer is not None:
                time.sleep(1.0 / args.speed)
    if viewer is not None:
        print("Done. Close the viewer window to exit.", flush=True)
        while viewer.is_running():
            time.sleep(0.1)


if __name__ == "__main__":
    main()
