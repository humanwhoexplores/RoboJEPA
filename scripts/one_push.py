"""Run one push to each target and save a filmstrip, to check the sim works.

    python -m scripts.one_push
"""
import numpy as np
from PIL import Image

from envs.push_env import TARGETS, make_env
from envs.push_primitive import execute_push, label_outcome

env = make_env()
strips = []
for side in ("left", "right"):
    env.reset()
    env.set_scene([0.03, 0.02], 0.2, "red")
    c = env.cube_state()[0][:2]
    v = TARGETS[side] - c
    frames = {}
    log = execute_push(env, np.arctan2(v[1], v[0]), np.linalg.norm(v), 0.0,
                       keyframe_fn=lambda n: frames.__setitem__(n, env.camera_image()))
    lab = label_outcome(log["cube_pos"], log["contact"], log["phase"], c)
    print(f"push {side}: success_left={lab['success_left']} success_right={lab['success_right']} "
          f"moved {lab['displacement'] * 100:.1f} cm, {len(log['phase'])} control steps")
    strips.append(np.concatenate([frames[k] for k in ("initial", "after_descend", "after_push", "final")], 1))
Image.fromarray(np.concatenate(strips, 0)).save("one_push.png")
print("saved one_push.png (rows: push left, push right)")
