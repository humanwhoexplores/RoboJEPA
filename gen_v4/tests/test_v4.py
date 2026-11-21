"""Quick checks for generator v4:  python -m pytest gen_v4/tests -q"""
import inspect

import numpy as np
import pytest

from gen_v3.generate import start_scene
from gen_v3.tasks import enumerate_goals, label_goals
from gen_v4 import sampler
from gen_v4.decisions import pick
from gen_v4.labels import label_sets, references
from gen_v4.sampler import per_object_counts, sample_pushes

OPTS = {"cameras": ["cam0", "cam1", "cam2", "cam3", "cam4"], "held_cameras": ["held1", "held2"], "held_share": 0.5,
        "visual": True, "distractors": True, "arm": False}


@pytest.fixture(scope="module")
def scene():
    spec, pos, dis, state = start_scene(7, 1, OPTS)
    return spec, pos, dis, state


def test_sampler_takes_no_goal():
    params = inspect.signature(sample_pushes).parameters
    assert list(params) == ["scene", "pos", "dis_pos", "rng", "n_total", "n_lines"]
    src = inspect.getsource(sampler)
    assert "target_point" not in src and "goal" not in inspect.getsource(sample_pushes).split('"""')[2]


def test_sampler_is_blind_to_goals_in_the_scene(scene):
    spec, pos, dis, _ = scene
    a = sample_pushes(spec, pos, dis, np.random.default_rng(3))
    spec2 = dict(spec, goals=[{"type": "in_zone", "a": 0, "b": 0}], zones=spec["zones"])  # extra keys are ignored
    b = sample_pushes(spec2, pos, dis, np.random.default_rng(3))
    assert a == b


def test_sampler_counts_and_shuffle(scene):
    spec, pos, dis, _ = scene
    c = sample_pushes(spec, pos, dis, np.random.default_rng(0))
    assert len(c) == 64
    assert sum(x["candidate_type"] == "random_line" for x in c) == 8
    objs = [x["sampler"]["object"] for x in c if x["sampler"]]
    counts = np.bincount(objs, minlength=len(spec["objects"]))
    assert counts.max() - counts.min() <= 1
    types = [x["candidate_type"] for x in c]
    assert types[:8] != ["random_line"] * 8 and types[-8:] != ["random_line"] * 8  # shuffled
    dists = sorted(x["sampler"]["dist"] for x in c if x["sampler"] and x["sampler"]["object"] == 0)
    assert dists[0] < 0.1 and dists[-1] > 0.18  # spread over the range


def test_per_object_counts():
    rng = np.random.default_rng(0)
    for n in range(3, 7):
        c = per_object_counts(56, n, rng)
        assert c.sum() == 56 and c.max() - c.min() <= 1


def test_decisions_rebuild_v3_and_add_open(scene):
    spec, pos, dis, _ = scene
    goals = enumerate_goals(spec, pos)
    evals, compound, texts, _ = pick(goals, spec, pos, dis, 7, 1)
    assert [e["source"] for e in evals] == ["v3_focus"] * 4 + ["open"] * 4
    assert len({e["goal_id"] for e in evals}) == 8
    assert all(e["in_view"] for e in evals) and all(e["straight_path"] for e in evals[:4])
    assert set(texts) >= {g["goal_id"] for g in goals} | {g["goal_id"] for g in compound}


def test_labels_need_the_named_object_to_move():
    scene = {"objects": [{"radius": 0.03}, {"radius": 0.03}], "zones": []}
    start = np.array([[0.0, 0.0, 0.02], [0.0, 0.20, 0.02]])
    end = np.array([[0.0, 0.0, 0.02], [0.0, 0.07, 0.02]])  # b was shoved next to a; a never moved
    traj = np.repeat(end[None], 12, axis=0)
    phases = np.array(["settle"] * 12)
    g = {"type": "next_to", "a": 0, "b": 1}
    lab = label_sets([g], traj, phases, scene, start)
    assert lab["state"][0] == 1 and lab["default"][0] == 0 and lab["strict"][0] == 0
    assert np.array_equal(lab["state"], label_goals([g], traj, phases, scene))  # state = v3 rule
    end2 = np.array([[0.0, 0.13, 0.02], [0.0, 0.20, 0.02]])  # a pushed up to b: a real success
    lab2 = label_sets([g], np.repeat(end2[None], 12, axis=0), phases, scene, start)
    assert lab2["default"][0] == 1 and lab2["strict"][0] == 1


def test_references():
    assert references({"type": "in_zone", "a": 0, "b": 1}) == []
    assert references({"type": "between", "a": 0, "b": 1, "c": 2}) == [1, 2]
    g = {"type": "and", "parts": [{"type": "left_of", "a": 0, "b": 1}, {"type": "next_to", "a": 1, "b": 2}]}
    assert references(g) == [2]
