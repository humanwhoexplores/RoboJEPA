"""Quick checks for generator v3:  python -m pytest gen_v3/tests -q"""
import numpy as np
import pytest

from gen_v3 import scene_xml as sx
from gen_v3.candidates import contact_radius, make_candidates, to_action
from gen_v3.env import SceneEnv
from gen_v3.language import caption, instruction, refer
from gen_v3.primitives import execute_plan, execute_push
from gen_v3.scene_spec import sample_scene
from gen_v3.tasks import enumerate_goals, holds, label_goals


@pytest.fixture(scope="module")
def env():
    return SceneEnv()


def one_object(shape="cube", size="s", xy=(0.0, 0.0)):
    obj = {"pool_name": f"{shape}_{size}_0", "shape": shape, "size": size, "color": "yellow",
           "xy": list(xy), "yaw": 0.3, "radius": sx.SIZES[size] * 1.42}
    return {"objects": [obj], "zones": [{"color": "red", "xy": [0.0, 0.15]}]}


def test_scene_builds_and_settles(env):
    spec = sample_scene(7, 0)
    env.set_scene(spec)
    pos, _ = env.object_poses()
    assert len(pos) == len(spec["objects"]) >= 3
    assert np.allclose(pos[:, :2], [o["xy"] for o in spec["objects"]], atol=0.005)  # nothing slid while settling
    assert env.object_speeds().max() < 1e-3


def test_ideal_push_reaches_zone(env):
    spec = one_object()
    env.set_scene(spec)
    p0 = env.object_poses()[0][0, :2]
    v = np.array(spec["zones"][0]["xy"]) - p0
    theta = np.arctan2(v[1], v[0])
    start, length = to_action(p0, contact_radius(spec["objects"][0]), theta, np.linalg.norm(v), 0.0)
    log = execute_push(env, start[0], start[1], theta, length)
    assert np.linalg.norm(log["obj_pos"][-1, 0, :2] - spec["zones"][0]["xy"]) < sx.ZONE_RADIUS
    assert log["contact"].any()


def test_miss_does_not_touch(env):
    spec = one_object()
    env.set_scene(spec)
    p0 = env.object_poses()[0][0, :2]
    offset = contact_radius(spec["objects"][0]) + sx.PUSHER_HALF_SPAN + 0.03
    start, length = to_action(p0, contact_radius(spec["objects"][0]), np.pi / 2, 0.15, offset)
    log = execute_push(env, start[0], start[1], np.pi / 2, length)
    assert not log["contact"].any()


def test_predicates_on_hand_made_positions():
    scene = {"objects": [{"radius": 0.03}, {"radius": 0.03}], "zones": [{"xy": [0.1, 0.1]}]}
    pos = np.array([[[0.1, 0.12, 0.02], [0.1, 0.24, 0.02]]])  # a in the zone, a left of b
    assert holds({"type": "in_zone", "a": 0, "b": 0}, pos, scene)[0]
    assert holds({"type": "left_of", "a": 0, "b": 1}, pos, scene)[0]
    assert not holds({"type": "right_of", "a": 0, "b": 1}, pos, scene)[0]
    assert not holds({"type": "next_to", "a": 0, "b": 1}, pos, scene)[0]  # 12 cm apart: a 6 cm gap


def test_rollout_is_deterministic(env):
    spec = sample_scene(7, 1)
    env.set_scene(spec)
    init = env.object_poses()[0]
    s0 = env.get_state()
    goals = enumerate_goals(spec, init)
    _, cands = make_candidates(spec, goals, init, np.random.default_rng(0), env.distractor_poses())
    assert len(cands) == 32
    runs = []
    for _ in range(2):
        env.set_state(s0)
        c = cands[0]
        runs.append(execute_push(env, c["x0"], c["y0"], c["theta"], c["length"]))
    assert np.array_equal(runs[0]["obj_pos"], runs[1]["obj_pos"])
    assert np.array_equal(label_goals(goals, runs[0]["obj_pos"], runs[0]["phase"], spec),
                          label_goals(goals, runs[1]["obj_pos"], runs[1]["phase"], spec))


def test_referring_expressions_are_unique():
    for i in range(30):
        objs = sample_scene(11, i)["objects"]
        names = [refer(k, objs) for k in range(len(objs))]
        assert len(set(names)) == len(names)


def test_new_relations_on_hand_made_positions():
    scene = {"objects": [{"radius": 0.03}] * 3, "zones": []}
    # a (index 0) is 10 cm in front of b (+x) and halfway between b and c
    pos = np.array([[[0.10, 0.0, 0.02], [0.0, 0.0, 0.02], [0.20, 0.0, 0.02]]])
    assert holds({"type": "in_front_of", "a": 0, "b": 1}, pos, scene)[0]
    assert not holds({"type": "behind", "a": 0, "b": 1}, pos, scene)[0]
    assert holds({"type": "behind", "a": 0, "b": 2}, pos, scene)[0]
    assert holds({"type": "between", "a": 0, "b": 1, "c": 2}, pos, scene)[0]
    pos[0, 0, 1] = 0.05  # 5 cm off the b-c line
    assert not holds({"type": "between", "a": 0, "b": 1, "c": 2}, pos, scene)[0]
    g = {"type": "and", "parts": [{"type": "in_front_of", "a": 0, "b": 1}, {"type": "behind", "a": 0, "b": 2}]}
    assert holds(g, pos, scene)[0]


@pytest.mark.parametrize("shape", ["hex", "bar"])
def test_new_shapes_push_cleanly(env, shape):
    spec = one_object(shape, "l")
    spec["objects"][0]["radius"] = 0.05
    env.set_scene(spec)
    p0 = env.object_poses()[0][0, :2]
    start, length = to_action(p0, contact_radius(spec["objects"][0]), np.pi / 2, 0.10, 0.0)
    log = execute_push(env, start[0], start[1], np.pi / 2, length)
    moved = log["obj_pos"][-1, 0, :2] - p0
    assert log["contact"].any() and 0.06 < moved[1] < 0.14 and abs(moved[0]) < 0.03


def test_arm_does_not_change_physics():
    spec = sample_scene(7, 2, visual=True, distractors=True)
    runs = []
    for arm in (False, True):
        e = SceneEnv(arm=arm)
        e.set_scene(spec)
        init = e.object_poses()[0]
        goals = enumerate_goals(spec, init)
        _, cands = make_candidates(spec, goals, init, np.random.default_rng(0), e.distractor_poses())
        c = cands[0]
        frames = []
        log = execute_push(e, c["x0"], c["y0"], c["theta"], c["length"], frame_fn=lambda n: frames.append(e.render()))
        runs.append((log, frames))
    assert np.array_equal(runs[0][0]["obj_pos"], runs[1][0]["obj_pos"])
    assert not np.array_equal(runs[0][1][1], runs[1][1][1])  # but the arm is in the picture


def test_two_push_plan_logs(env):
    spec = sample_scene(7, 3, distractors=True)
    env.set_scene(spec)
    init = env.object_poses()[0]
    goals = enumerate_goals(spec, init)
    _, cands = make_candidates(spec, goals, init, np.random.default_rng(0), env.distractor_poses())
    names = []
    p = [(c["x0"], c["y0"], c["theta"], c["length"]) for c in cands[:2]]
    log, done = execute_plan(env, p, frame_fn=names.append)
    assert names == ["initial", "after_push1", "mid1", "after_push2", "final"]
    assert set(log["push_idx"].tolist()) == {0, 1} and np.all(np.diff(log["push_idx"]) >= 0)
    assert log["phase"][log["push_idx"] == 0][-1] == "settle"
    assert log["dis_pos"].shape[1] == len(spec["distractors"])


def test_language_is_valid_and_split_aware():
    for i in range(30):
        spec = sample_scene(11, i)
        objs = spec["objects"]
        pos = np.array([o["xy"] + [0.02] for o in objs])
        rng = np.random.default_rng(i)
        for split in ("train", "test"):
            g = {"type": "between", "a": 0, "b": 1, "c": 2}
            text, template = instruction(g, spec, split, rng, pos)
            assert "{" not in text and template
        assert caption(spec, pos, pos) == "nothing moved"
        moved = pos.copy()
        moved[0, 1] -= 0.10
        assert caption(spec, pos, moved).startswith(refer(0, objs, "train", np.random.default_rng(0)).split()[0])
