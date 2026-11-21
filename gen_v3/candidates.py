"""Candidate pushes for one scene: 32 single pushes plus 8 two-push plans, all from the same saved start state.

Single pushes. Per focus goal (4 per scene, 24 total):
    ideal x3   aim object a at the goal's target point, full distance
    short      30 to 60% of the distance
    over       160 to 200% of the distance
    miss       the V's edge passes 2 to 4 cm beside object a (should not touch it)
Plus, across the scene:
    wrong_object x4   a focus goal's ideal motion applied to a different object
    random_object x2  a random object, random direction, 5 to 25 cm
    random_line x2    a random pusher line anywhere on the table
v2 had ideal x2 and 6 random_object pushes; v3 trades 3 random pushes for a third ideal and
4 more for 2 more relation types, so each decision has more positives.

Two-push plans (milestone 5). Per compound goal "A does g1 and B does g2" (2 per scene):
    plan_closed x2   ideal push for g1, then an ideal push for g2 planned from the settled middle state
    plan_open        both ideal pushes planned from the initial state (the second can be stale)
    plan_half        ideal push for g1, then a random push of B
If a scene has fewer than 2 compound goals, the remaining plans are random_plan (two random_object pushes).

Every push gets v1-style Gaussian noise on direction, distance and offset.
The stored action is pusher-centric (x0, y0, theta, length); the object-centric
intent is kept in the metadata for analysis only.
"""
import numpy as np

from gen_v3 import scene_xml as sx
from gen_v3.tasks import goal_id, target_point

N_FOCUS = 4
N_SINGLE = 32
N_PLANS = 8
N_COMPOUND = 2
FOCUS_MIX = {"in_zone": 0.3, "left_of": 0.1, "right_of": 0.1, "in_front_of": 0.1, "behind": 0.1,
             "next_to": 0.15, "between": 0.15}
NOISE = {"theta": 0.12, "dist_frac": 0.10, "offset": 0.01}  # same as v1
STANDOFF = 0.03  # gap between pusher and object face before the push starts
REACH = 0.22  # target points must lie inside the area every camera sees
CLEAR = 0.006  # pusher must come down at least this far from any object footprint
PATH_CLEAR = 0.005  # a focus goal's straight object path must pass at least this far from everything else


def contact_radius(obj):
    return sx.footprint_radius(obj["shape"], obj["size"])


def to_action(obj_xy, r_obj, theta, dist, offset):
    """Object-centric push (move an object of radius r_obj by dist along theta) -> pusher action."""
    u = np.array([np.cos(theta), np.sin(theta)])
    perp = np.array([-u[1], u[0]])
    back = sx.seat_x(r_obj) + STANDOFF
    start = obj_xy - u * back + perp * offset
    length = STANDOFF + max(dist, 0.0)
    return start, length


def v_plates(start, theta):
    """World-frame end points of the pusher's two plates when it touches down at start facing theta."""
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    plates = []
    for sign in (1, -1):
        center = np.array([0.0, sign * sx.PUSHER_PLATE_Y])
        along = sx.PUSHER_PLATE_HALF * np.array([np.cos(sx.PUSHER_ANGLE), sign * np.sin(sx.PUSHER_ANGLE)])
        plates.append((start + rot @ (center - along), start + rot @ (center + along)))
    return plates


def seg_dist(p, a, b):
    t = np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0.0, 1.0)
    return np.linalg.norm(p - (a + t * (b - a)))


def obstacles(scene, pos, dis_pos):
    """(xy, radius) of everything the pusher must not land on: objects and distractors."""
    obs = [(pos[i, :2], o["radius"]) for i, o in enumerate(scene["objects"])]
    obs += [(dis_pos[i, :2], o["radius"]) for i, o in enumerate(scene.get("distractors", []))]
    return obs


def is_clear(start, theta, obs):
    """True if neither plate lands on (or within CLEAR of) any obstacle's bounding circle."""
    plates = v_plates(start, theta)
    return all(seg_dist(xy, a, b) > r + sx.PUSHER_HALF_THICK + CLEAR for xy, r in obs for a, b in plates)


def clear_start(start, theta, obs, max_back=0.10):
    """Slide the touchdown point backwards along -theta until the V lands clear of every obstacle."""
    u = np.array([np.cos(theta), np.sin(theta)])
    for k in range(int(max_back / 0.005) + 1):
        p = start - u * 0.005 * k
        if is_clear(p, theta, obs):
            return p, 0.005 * k
    return None, None


def path_clear(g, scene, pos, dis_pos):
    """True if object a can slide straight to the goal's target point without running into anything else.

    Without this, e.g. "push A behind B" with A in front of B pushes A straight through B.
    """
    a = g["a"]
    pa, t = pos[a, :2], target_point(g, scene, pos)
    ra = scene["objects"][a]["radius"]
    if np.linalg.norm(t - pa) < 1e-6:
        return False
    obs = [ob for k, ob in enumerate(obstacles(scene, pos, dis_pos)) if k != a]
    return all(seg_dist(xy, pa, t) > ra + r + PATH_CLEAR for xy, r in obs)


def choose_focus(goals, scene, init_pos, rng, init_dis=None):
    """Up to N_FOCUS goals that one clean push can reach, drawn by type with FOCUS_MIX weights, distinct objects first."""
    init_dis = np.zeros((0, 3)) if init_dis is None else init_dis
    reachable = [g for g in goals if np.all(np.abs(target_point(g, scene, init_pos)) < REACH)
                 and path_clear(g, scene, init_pos, init_dis)]
    focus, used_a = [], set()
    types = list(FOCUS_MIX)
    p = np.array(list(FOCUS_MIX.values()))
    for _ in range(200):
        if len(focus) == N_FOCUS or not reachable:
            break
        t = types[rng.choice(len(types), p=p)]
        pool = [g for g in reachable if g["type"] == t and g not in focus]
        fresh = [g for g in pool if g["a"] not in used_a]
        pool = fresh or pool
        if not pool:
            continue
        g = pool[rng.integers(len(pool))]
        focus.append(g)
        used_a.add(g["a"])
    return focus


def ideal_intent(g, scene, pos):
    a = g["a"]
    v = target_point(g, scene, pos) - pos[a, :2]
    return a, float(np.arctan2(v[1], v[0])), float(np.linalg.norm(v))


def realize(ctype, gid, a, theta, dist, offset, scene, pos, dis_pos, rng):
    """Noisy object-centric intent -> pusher-centric candidate (falls back to a random line if boxed in)."""
    intended = {"object": a, "theta": theta, "dist": dist, "offset": offset}
    theta += rng.normal(0, NOISE["theta"])
    dist *= 1 + rng.normal(0, NOISE["dist_frac"])
    offset += rng.normal(0, NOISE["offset"])
    obs = obstacles(scene, pos, dis_pos)
    start, length = to_action(pos[a, :2], contact_radius(scene["objects"][a]), theta, dist, offset)
    start, extra = clear_start(start, theta, obs)
    if start is None:  # boxed in: fall back to a random line
        ctype = ctype + "_blocked"
        start, theta, length, extra = random_line(obs, rng)
    return {"candidate_type": ctype, "focus_goal": gid, "intended": intended,
            "x0": float(start[0]), "y0": float(start[1]), "theta": float(theta), "length": float(length + extra)}


def random_object_intent(scene, rng):
    return int(rng.integers(len(scene["objects"]))), float(rng.uniform(-np.pi, np.pi)), float(rng.uniform(0.05, 0.25))


def make_candidates(scene, goals, init_pos, rng, init_dis=None):
    init_dis = np.zeros((0, 3)) if init_dis is None else init_dis
    focus = choose_focus(goals, scene, init_pos, rng, init_dis)
    objs = scene["objects"]
    intents = []  # (type, focus goal id or "", object index, theta, dist, offset)
    for g in focus:
        a, theta, dist = ideal_intent(g, scene, init_pos)
        intents += [("ideal", g["goal_id"], a, theta, dist, 0.0)] * 3
        intents.append(("short", g["goal_id"], a, theta, dist * rng.uniform(0.3, 0.6), 0.0))
        intents.append(("over", g["goal_id"], a, theta, dist * rng.uniform(1.6, 2.0), 0.0))
        miss = rng.choice([-1, 1]) * (contact_radius(objs[a]) + sx.PUSHER_HALF_SPAN + rng.uniform(0.02, 0.04))
        intents.append(("miss", g["goal_id"], a, theta, dist, float(miss)))
    for g in focus:
        others = [i for i in range(len(objs)) if i != g["a"]]
        b = int(rng.choice(others))
        v = target_point(g, scene, init_pos) - init_pos[b, :2]
        intents.append(("wrong_object", g["goal_id"], b, float(np.arctan2(v[1], v[0])), float(np.linalg.norm(v)), 0.0))
    n_random_obj = N_SINGLE - len(intents) - 2
    for _ in range(max(n_random_obj, 0)):
        intents.append(("random_object", "", *random_object_intent(scene, rng), 0.0))

    cands = [realize(*it, scene, init_pos, init_dis, rng) for it in intents]
    obs = obstacles(scene, init_pos, init_dis)
    while len(cands) < N_SINGLE:
        start, theta, length, _ = random_line(obs, rng)
        cands.append({"candidate_type": "random_line", "focus_goal": "", "intended": None,
                      "x0": float(start[0]), "y0": float(start[1]), "theta": float(theta), "length": float(length)})
    return focus, cands


def compound_goals(focus):
    """Up to N_COMPOUND 'and' goals from pairs of focus goals that move different objects."""
    out = []
    for i in range(len(focus)):
        for j in range(i + 1, len(focus)):
            g1, g2 = focus[i], focus[j]
            if g1["a"] != g2["a"] and len(out) < N_COMPOUND:
                g = {"type": "and", "parts": [g1, g2]}
                g["goal_id"] = goal_id(g)
                out.append(g)
    return out


def make_plans(scene, compound, init_pos, init_dis, rng):
    """Two-push plans. Each is {plan_type, focus_goal, pushes: [push dict or None], replan: [...]}.

    A None push is planned when the plan runs, from the settled middle state, by second_push().
    """
    plans = []
    for g in compound:
        g1, g2 = g["parts"]
        p1 = lambda: realize("ideal", g1["goal_id"], *ideal_intent(g1, scene, init_pos), 0.0, scene, init_pos, init_dis, rng)
        for _ in range(2):
            plans.append({"plan_type": "plan_closed", "focus_goal": g["goal_id"], "pushes": [p1(), None],
                          "second": {"kind": "ideal", "goal": g2}})
        plans.append({"plan_type": "plan_open", "focus_goal": g["goal_id"],
                      "pushes": [p1(), realize("ideal", g2["goal_id"], *ideal_intent(g2, scene, init_pos), 0.0,
                                               scene, init_pos, init_dis, rng)], "second": None})
        plans.append({"plan_type": "plan_half", "focus_goal": g["goal_id"], "pushes": [p1(), None],
                      "second": {"kind": "random", "object": g2["a"]}})
    while len(plans) < N_PLANS:
        plans.append({"plan_type": "random_plan", "focus_goal": "", "pushes": [None, None],
                      "second": {"kind": "random", "object": None}})
    return plans


def second_push(plan, k, scene, pos, dis_pos, rng):
    """Plan push k of a plan from the current (settled) positions."""
    spec = plan["second"]
    if spec["kind"] == "ideal":
        g = spec["goal"]
        return realize("ideal", g["goal_id"], *ideal_intent(g, scene, pos), 0.0, scene, pos, dis_pos, rng)
    a, theta, dist = random_object_intent(scene, rng)
    if spec["object"] is not None:
        a = spec["object"]
    return realize("random_object", "", a, theta, dist, 0.0, scene, pos, dis_pos, rng)


def random_line(obs, rng):
    for _ in range(100):
        start = rng.uniform(-0.25, 0.25, 2)
        theta = float(rng.uniform(-np.pi, np.pi))
        p, extra = clear_start(start, theta, obs, max_back=0.0)
        if p is not None:
            return p, theta, float(rng.uniform(0.08, 0.30)), 0.0
    raise RuntimeError("no clear spot for a random line")
