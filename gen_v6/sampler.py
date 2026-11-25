"""Goal-blind candidate actions for one scene state: drawn from the objects and fixtures alone, never from a
goal or an instruction (the v4 rule, now for every action type).

Per call, n_total candidates of mixed types, in the proportions of the scene family's ACTION_MIX (types the
scene has nothing for, such as stand_up with nothing lying down, give their share to the others; at least 3
types and none above half). Objects are used in turn, in random order, so each gets a fair share. Targets of
grasp actions are free spots, tops of other objects and fixtures (container floors, the platform, the ramp's
top, a seesaw end), so some land in goal-relevant places and most do not, as with v4's pushes.

Each candidate:
    {"action": type, "params": {...}, "sampler": {"object": i, ...}, "vec": 12 numbers}
vec is a fixed-length summary for models that want one (the full 20 Hz end-effector trajectory is in the
rollout log): [type id, x0, y0, z0, x1, y1, z1, yaw0, yaw1, opening, speed, tilt].
Every candidate is checked with the arm's IK before it is kept: unreachable ones are redrawn.
"""
import numpy as np

from gen_v3 import scene_xml as sx
from gen_v5.primitives import wrap
from gen_v6 import catalog as cat
from gen_v6 import grasps as gr
from gen_v6 import primitives as P

ACTIONS = ["push", "pull", "sweep", "poke", "topple", "place", "drop", "stack", "unstack", "twist", "stand_up",
           "drawer", "lid", "toss"]
# No tool_pull (raking with the hammer): the parallel grip holds the handle level, so the handle comes down on the
# object it is meant to rake; the hammer is still picked, placed, pushed and tossed like any other object.
ACTION_ID = {a: k for k, a in enumerate(ACTIONS)}
ACTION_MIX = {
    "blocks": {"push": 0.28, "pull": 0.10, "sweep": 0.10, "poke": 0.14, "place": 0.16, "drop": 0.08, "stack": 0.08,
               "twist": 0.06},
    "containers": {"push": 0.16, "sweep": 0.10, "poke": 0.08, "place": 0.28, "drop": 0.20, "stack": 0.06,
                   "pull": 0.06, "toss": 0.06},
    "stacking": {"push": 0.12, "topple": 0.16, "place": 0.12, "stack": 0.20, "unstack": 0.12, "twist": 0.08,
                 "stand_up": 0.12, "drop": 0.08},
    "rolling": {"push": 0.22, "poke": 0.22, "pull": 0.08, "place": 0.18, "drop": 0.14, "sweep": 0.10, "toss": 0.06},
    "chain": {"push": 0.30, "poke": 0.18, "topple": 0.14, "place": 0.14, "drop": 0.14, "sweep": 0.10},
    "tools": {"drawer": 0.22, "lid": 0.18, "toss": 0.10, "place": 0.22, "push": 0.14,
              "drop": 0.10},
}
N_TOTAL = 64
N_LINES = 4  # pushes along random lines anywhere on the table (some touch nothing)
REACH = 0.24  # targets and slide starts stay within +-REACH
STANDOFF = 0.03  # gap between the fingers and the object at the start: the palm overhangs the fingers by 2.2 cm
FINGER_HALF = 0.012  # closed fingers: half their size along a slide
DIST = (0.03, 0.25)


# ----------------------------------------------------------------------------- scene geometry
class Scene:
    """What the sampler may read: object poses and sizes, fixture poses, distractors. No goals."""

    def __init__(self, env, spec):
        self.env, self.spec = env, spec
        self.pos, self.quat = env.object_poses()
        self.dis = env.distractor_poses()
        self.n = len(spec["objects"])
        self.r = np.array([env.half_width(n) for n in env.active])
        self.below = np.array([env.extent(i)[0] for i in range(self.n)])
        self.above = np.array([env.extent(i)[1] for i in range(self.n)])
        self.zz = np.array([1 - 2 * (q[1] ** 2 + q[2] ** 2) for q in self.quat])
        pair, table, _ = env.contact_summary()
        self.pair, self.table = pair, table
        self.fixtures = spec.get("fixtures", [])

    def kind(self, i):
        o = self.spec["objects"][i]
        return o.get("kind", o.get("shape"))

    def top(self, i):
        return float(self.pos[i, 2] + self.above[i])

    def obstacles(self, skip=()):
        obs = [(self.pos[i, :2], self.r[i]) for i in range(self.n) if i not in skip]
        obs += [(self.dis[k, :2], d["radius"]) for k, d in enumerate(self.spec.get("distractors", []))]
        return obs

    def clear(self, xy, r, skip=(), fixtures=True):
        if np.abs(xy).max() > REACH:
            return False
        if any(np.linalg.norm(xy - p) < r + q for p, q in self.obstacles(skip)):
            return False
        if fixtures:
            from gen_v6.scene_spec import _rect_dist
            return all(_rect_dist(xy, f) > r for f in self.fixtures)
        return True

    def stacked_on(self, i):
        """Index of the object i rests on, or None."""
        for j in range(self.n):
            if self.pair[i, j] and self.pos[i, 2] > self.pos[j, 2] + 0.01 and not self.table[i]:
                return j
        return None

    def has_on_top(self, j):
        return any(self.pair[i, j] and self.pos[i, 2] > self.pos[j, 2] + 0.01 for i in range(self.n) if i != j)


# ----------------------------------------------------------------------------- target surfaces for grasp actions
def fixture_target(S, f, rng):
    """(xy, surface z) of a spot on / in fixture f, or None."""
    fx = S.fixtures[f]
    k = fx["kind"]
    c, s = np.cos(fx["yaw"]), np.sin(fx["yaw"])
    R = np.array([[c, -s], [s, c]])
    if k in ("tray", "bin", "lidbox"):
        ix, iy = {"tray": cat.TRAY_IN, "bin": cat.BIN_IN, "lidbox": cat.LIDBOX_IN[:2]}[k]
        local, z = rng.uniform(-0.5, 0.5, 2) * [ix, iy], {"tray": 0.0, "bin": 0.006, "lidbox": 0.01}[k]
    elif k == "bowl":
        ang = rng.uniform(-np.pi, np.pi)
        local, z = rng.uniform(0, 0.02) * np.array([np.cos(ang), np.sin(ang)]), 0.004
    elif k == "platform":
        local, z = rng.uniform(-0.5, 0.5, 2) * np.array(cat.PLATFORM_HALF[:2]), 2 * cat.PLATFORM_HALF[2]
    elif k == "ramp":  # near the top of the incline
        x = cat.RAMP_LEN / 2 - rng.uniform(0.02, 0.06)
        local, z = np.array([x, rng.uniform(-0.02, 0.02)]), cat.RAMP_H * (x + cat.RAMP_LEN / 2) / cat.RAMP_LEN
    elif k == "seesaw":  # over one end of the plank (height read from the simulator)
        env = S.env
        b = env.model.body("fx_seesaw_plank").id
        sgn = 1 if rng.random() < 0.5 else -1
        end = env.data.xpos[b] + env.data.xmat[b].reshape(3, 3) @ np.array([sgn * (cat.SEESAW_PLANK[0] - 0.03), 0, cat.SEESAW_PLANK[2]])
        return end[:2], float(end[2])
    elif k == "cabinet":
        env = S.env
        slide = float(env.data.qpos[env.fx_joint["cabinet"][0]])
        ix, iy, iz = cat.DRAWER_IN
        if slide < 0.05:  # a closed drawer cannot take anything
            return None
        dx = cat.CAB_HALF[0] - ix - 0.008 + slide
        local = np.array([dx + rng.uniform(0, 0.6) * min(ix, slide - 0.03), rng.uniform(-0.5, 0.5) * iy])
        z = 0.02
    else:
        return None
    return np.asarray(fx["xy"]) + R @ local, float(z)


FIT_MARGIN = 0.004  # m between a placed object's footprint and a container's walls


def _interior(S, f):
    """Where a placed object's whole footprint must lie in fixture f's frame: ("box", (x0, x1, y0, y1)) or
    ("disc", radius); None for fixtures with no floor to fit in (ramp, seesaw: anything goes)."""
    k = S.fixtures[f]["kind"]
    if k in ("tray", "bin", "lidbox"):
        ix, iy = {"tray": cat.TRAY_IN, "bin": cat.BIN_IN, "lidbox": cat.LIDBOX_IN[:2]}[k]
        return "box", (-ix, ix, -iy, iy)
    if k == "platform":
        hx, hy, _ = cat.PLATFORM_HALF
        return "box", (-hx, hx, -hy, hy)
    if k == "cabinet":  # only the part of the drawer pulled out past the cabinet's front is open from above
        ix, iy, _ = cat.DRAWER_IN
        slide = float(S.env.data.qpos[S.env.fx_joint["cabinet"][0]])
        dx = cat.CAB_HALF[0] - ix - 0.008 + slide
        return "box", (max(cat.CAB_HALF[0], dx - ix), dx + ix, -iy, iy)
    if k == "bowl":
        return "disc", cat.BOWL_R
    return None


def fit_in(S, i, xy, turn, f):
    """Shift the target xy of object i's body origin (turned by `turn` about z on the way) so its whole
    footprint lies inside fixture f's interior with FIT_MARGIN to spare; None if it cannot fit at all."""
    inner = _interior(S, f)
    if inner is None:
        return np.asarray(xy, float)
    import mujoco
    R = np.empty(9)
    mujoco.mju_quat2Mat(R, np.asarray(S.quat[i], float))
    c, s_ = np.cos(turn), np.sin(turn)
    Rz = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
    off = (S.env._box[S.env.active[i]] @ (Rz @ R.reshape(3, 3)).T)[:, :2]  # footprint corners about the origin
    fx = S.fixtures[f]
    cf, sf = np.cos(fx["yaw"]), np.sin(fx["yaw"])
    Rf = np.array([[cf, -sf], [sf, cf]])
    loc = (np.asarray(xy, float) - np.asarray(fx["xy"], float)) @ Rf  # target in the fixture frame
    loff = off @ Rf
    m = FIT_MARGIN
    if inner[0] == "box":
        x0, x1, y0, y1 = inner[1]
        lo, hi = loff.min(0), loff.max(0)
        if hi[0] - lo[0] > x1 - x0 - 2 * m or hi[1] - lo[1] > y1 - y0 - 2 * m:
            return None
        loc = np.array([np.clip(loc[0], x0 + m - lo[0], x1 - m - hi[0]), np.clip(loc[1], y0 + m - lo[1], y1 - m - hi[1])])
    else:
        reach = float(np.linalg.norm(loff, axis=1).max())
        room = inner[1] - m - reach
        if room < 0:
            return None
        d = float(np.linalg.norm(loc))
        if d > room:
            loc = loc * room / d
    return np.asarray(fx["xy"], float) + Rf @ loc


def place_target(S, i, rng, mode):
    """(xy, surface z, target kind, target index) for putting object i down. mode: free | object | fixture."""
    if mode == "object":
        tops = [j for j in range(S.n) if j != i and S.kind(j) in ("cube", "cylinder", "hex", "bar", "can", "milk",
                                                                   "cereal", "bread") and S.zz[j] > 0.9
                and not S.has_on_top(j) and S.stacked_on(i) != j]
        if tops:
            j = int(rng.choice(tops))
            return S.pos[j, :2] + rng.normal(0, 0.004, 2), S.top(j), "object", j
        mode = "fixture"
    if mode == "fixture":
        fs = [f for f, fx in enumerate(S.fixtures) if fx["kind"] in ("tray", "bin", "bowl", "platform", "ramp",
                                                                    "seesaw", "cabinet", "lidbox")]
        if S.env is not None:
            fs = [f for f in fs if not (S.fixtures[f]["kind"] == "lidbox"
                                        and S.env.data.qpos[S.env.fx_joint["lidbox"][0]] < 1.0)]
        if fs:
            f = int(rng.choice(fs))
            t = fixture_target(S, f, rng)
            if t is not None:
                return t[0], t[1], "fixture", f
    for _ in range(100):
        xy = rng.uniform(-0.2, 0.2, 2)
        if S.clear(xy, S.r[i] + 0.01, skip=(i,)):
            return xy, 0.0, "table", -1
    return None


# ----------------------------------------------------------------------------- candidate builders
def _vec(action, x0, y0, z0, x1, y1, z1, yaw0=0.0, yaw1=0.0, opening=0.0, speed=0.0, tilt=0.0):
    return [float(ACTION_ID[action]), x0, y0, z0, x1, y1, z1, yaw0, yaw1, opening, speed, tilt]


def slide_cand(action, x0, y0, theta, length, obj, extra=None, opening=0.0, z=P.PUSH_Z, speed=P.PUSH_SPEED):
    u = np.array([np.cos(theta), np.sin(theta)])
    e = np.array([x0, y0]) + u * length
    params = {"x0": float(x0), "y0": float(y0), "theta": float(theta), "length": float(length), "z": float(z),
              "opening": float(opening), "speed": float(speed)}
    return {"action": action, "params": params, "sampler": {"object": obj, **(extra or {})},
            "vec": _vec(action, x0, y0, z, e[0], e[1], z, theta, theta, opening, speed)}


def _slide_start(S, i, theta, offset, back):
    u = np.array([np.cos(theta), np.sin(theta)])
    perp = np.array([-u[1], u[0]])
    return S.pos[i, :2] - u * back + perp * offset


def push_cand(S, i, rng, theta=None):
    theta = rng.uniform(-np.pi, np.pi) if theta is None else theta
    r = S.r[i]
    dist = rng.uniform(*DIST)
    off = rng.uniform(-0.5, 0.5) * r
    start = _slide_start(S, i, theta, off, r + FINGER_HALF + STANDOFF)
    if not S.clear(start, FINGER_HALF + 0.006, skip=()):
        return None
    return slide_cand("push", *start, theta, STANDOFF + dist, i, {"dist": float(dist), "offset": float(off)})


def line_cand(S, rng):
    for _ in range(50):
        start = rng.uniform(-0.2, 0.2, 2)
        if S.clear(start, FINGER_HALF + 0.006):
            th = rng.uniform(-np.pi, np.pi)
            return slide_cand("push", *start, th, rng.uniform(0.08, 0.25), -1, {"line": True})
    return None


def pull_cand(S, i, rng):
    to_base = np.arctan2(-S.pos[i, 1], -0.52 - S.pos[i, 0])
    theta = to_base + rng.uniform(-1.0, 1.0)
    r = S.r[i]
    dist = rng.uniform(0.03, 0.18)
    off = rng.uniform(-0.4, 0.4) * r
    start = _slide_start(S, i, theta, off, r + FINGER_HALF + STANDOFF)
    if not S.clear(start, FINGER_HALF + 0.006):
        return None
    return slide_cand("pull", *start, theta, STANDOFF + dist, i, {"dist": float(dist), "offset": float(off)})


def sweep_cand(S, i, rng):
    others = [j for j in range(S.n) if j != i and np.linalg.norm(S.pos[j, :2] - S.pos[i, :2]) < 0.14]
    j = int(rng.choice(others)) if others else i
    mid = (S.pos[i, :2] + S.pos[j, :2]) / 2
    d = S.pos[j, :2] - S.pos[i, :2]
    base = np.arctan2(d[1], d[0]) + np.pi / 2 if j != i else rng.uniform(-np.pi, np.pi)
    theta = base + rng.uniform(-0.5, 0.5) + (np.pi if rng.random() < 0.5 else 0)
    opening = rng.uniform(0.05, 0.08)
    u = np.array([np.cos(theta), np.sin(theta)])
    start = mid - u * (np.linalg.norm(d) / 2 + max(S.r[i], S.r[j]) + 0.03)
    if not S.clear(start, opening / 2 + 0.01):
        return None
    return slide_cand("sweep", *start, theta, rng.uniform(0.15, 0.25), i, {"with": j}, opening=opening)


def poke_cand(S, i, rng):
    theta = rng.uniform(-np.pi, np.pi)
    r = S.r[i]
    stroke = rng.uniform(0.08, 0.12)
    hit = rng.uniform(0.4, 0.55)  # where along the stroke the fingers meet the object's face
    off = rng.uniform(-0.3, 0.3) * r
    start = _slide_start(S, i, theta, off, r + FINGER_HALF + hit * stroke)
    if not S.clear(start, FINGER_HALF + 0.006):
        return None
    c = slide_cand("poke", *start, theta, stroke, i, {"hit": float(hit), "offset": float(off)},
                   speed=rng.uniform(0.25, 0.45))
    return c


def topple_cand(S, i, rng):
    if S.kind(i) not in cat.TALL or S.zz[i] < 0.9:
        return None
    height = S.top(i)
    theta = rng.uniform(-np.pi, np.pi)
    r = S.r[i]
    start = _slide_start(S, i, theta, rng.uniform(-0.3, 0.3) * r, r + FINGER_HALF + STANDOFF)
    if not S.clear(start, FINGER_HALF + 0.006):
        return None
    z = float(np.clip(height * rng.uniform(0.65, 0.85), 0.03, 0.15))
    return slide_cand("topple", *start, theta, STANDOFF + rng.uniform(0.05, 0.12), i, {}, z=z)


def _grasp(S, i, rng):
    opts = gr.grasp_options(S.env, i, S.spec["objects"][i], rng)
    if not opts:
        return None
    return opts[int(rng.integers(len(opts)))] if rng.random() < 0.3 else opts[0]


def grasp_cand(action, S, i, rng, g, target, drop=0.0, yaw_turn=None):
    gx, gy, gyaw, gz, w = g
    xy, surf, tkind, tidx = target
    dz = gz - S.pos[i, 2]  # pads above the body origin
    below = S.below[i]  # body origin above the object's lowest point
    yaw1 = gyaw + (yaw_turn if yaw_turn is not None else rng.uniform(-0.8, 0.8))
    if tkind == "fixture":
        xy = fit_in(S, i, xy, yaw1 - gyaw, tidx)
        if xy is None:
            return None
    off = np.array([gx, gy]) - S.pos[i, :2]  # pads relative to the body origin; turns with the object
    dy = yaw1 - gyaw
    rot = np.array([[np.cos(dy), -np.sin(dy)], [np.sin(dy), np.cos(dy)]])
    pxy = np.asarray(xy) + rot @ off
    pz = surf + below + dz + 0.005 + drop
    params = {"grasp": [float(gx), float(gy), float(gyaw), float(gz)],
              "place": [float(pxy[0]), float(pxy[1]), float(yaw1), float(min(pz, P.CARRY_Z - 0.02))],
              "open_width": gr.open_width(w), "drop": float(drop)}
    return {"action": action, "params": params,
            "sampler": {"object": i, "target": tkind, "target_index": int(tidx), "width": float(w)},
            "vec": _vec(action, gx, gy, gz, pxy[0], pxy[1], params["place"][3], gyaw, yaw1, params["open_width"])}


def place_cand(S, i, rng, action="place"):
    g = _grasp(S, i, rng)
    if g is None:
        return None
    mode = {"place": str(rng.choice(["free", "free", "fixture"])), "drop": str(rng.choice(["free", "fixture", "object"])),
            "stack": "object", "unstack": "free"}[action]
    if action == "unstack" and S.stacked_on(i) is None:
        return None
    t = place_target(S, i, rng, mode)
    if t is None:
        return None
    drop = rng.uniform(0.05, 0.2) if action == "drop" else 0.0
    return grasp_cand(action, S, i, rng, g, t, drop=drop)


def twist_cand(S, i, rng):
    if S.zz[i] < 0.9 or S.stacked_on(i) is not None:
        return None
    g = _grasp(S, i, rng)
    if g is None:
        return None
    turn = rng.choice([-1, 1]) * rng.uniform(np.pi / 4, np.pi)
    bottom = float(S.pos[i, 2] - S.below[i])  # turn it where it stands (on the table or the platform)
    return grasp_cand("twist", S, i, rng, g, (S.pos[i, :2], bottom, "in_place", -1), yaw_turn=float(wrap(turn)))


def stand_up_cand(S, i, rng):
    if S.kind(i) not in ("tube", "can", "bottle") or abs(S.zz[i]) > 0.35:
        return None
    opts = gr.grasp_options(S.env, i, S.spec["objects"][i], rng)
    if not opts:
        return None
    gx, gy, gyaw, gz, w = opts[0]
    half_len = float(np.max(np.abs(S.env._box[S.env.active[i]][:, 2])))  # body z (the axis) half extent
    for _ in range(30):
        xy = S.pos[i, :2] + rng.uniform(-0.08, 0.08, 2)
        if S.clear(xy, S.r[i] * 0.5, skip=(i,)):
            break
    else:
        return None
    pz = half_len + 0.006
    params = {"grasp": [float(gx), float(gy), float(gyaw), float(gz)], "place": [float(xy[0]), float(xy[1]), float(gyaw), pz],
              "open_width": gr.open_width(w), "tilt": float(np.pi / 2)}
    return {"action": "stand_up", "params": params, "sampler": {"object": i, "width": float(w)},
            "vec": _vec("stand_up", gx, gy, gz, xy[0], xy[1], pz, gyaw, gyaw, params["open_width"], 0.0, np.pi / 2)}


# ----------------------------------------------------------------------------- stage 2: drawer, lid, toss
def _fx_frame(S, f):
    fx = S.fixtures[f]
    c, s_ = np.cos(fx["yaw"]), np.sin(fx["yaw"])
    return np.asarray(fx["xy"], float), np.array([c, s_]), np.array([-s_, c])


def drawer_cand(S, f, rng):
    """Hook behind the handle and pull the drawer out, or push on the handle to shut it."""
    o, u, v = _fx_frame(S, f)
    slide = float(S.env.data.qpos[S.env.fx_joint["cabinet"][0]])
    hx = cat.CAB_HALF[0]
    can_open, can_close = slide < cat.DRAWER_TRAVEL - 0.03, slide > 0.02
    opening = can_open and (not can_close or rng.random() < 0.5)
    lat = rng.uniform(-0.01, 0.01)
    if opening:
        d = rng.uniform(0.03, cat.DRAWER_TRAVEL - slide)
        x0, length, th = hx + slide + cat.HANDLE_OUT - 0.021, 0.004 + d, 0.0
    else:
        d = rng.uniform(min(0.03, slide), slide)
        x0, length, th = hx + slide + cat.HANDLE_OUT + 0.027, 0.021 + d + 0.008, np.pi  # 1 cm past: it coasts short
    start = o + x0 * u + lat * v
    theta = float(np.arctan2(u[1], u[0]) + th)
    c = slide_cand("drawer", *start, theta, length, -1, {"fixture": int(f), "mode": "open" if opening else "close",
                                                       "amount": float(d)}, z=cat.HANDLE_Z)
    c["params"]["mode"] = c["sampler"]["mode"]
    return c


def lid_cand(S, f, rng):
    """Grasp the knob and swing the lid open, or push an open lid over so it falls shut."""
    env = S.env
    q = float(env.data.qpos[env.fx_joint["lidbox"][0]])
    lid = env.model.body("fx_lidbox_lid").id
    hinge = env.data.xpos[lid].copy()
    Rl = env.data.xmat[lid].reshape(3, 3)
    axis = Rl @ np.array([0.0, -1.0, 0.0])
    o, u, v = _fx_frame(S, f)
    if q < 0.3:
        knob = env.data.geom_xpos[env.model.geom("fx_lidbox_lid_g1").id].copy()
        yaw = float(np.arctan2(axis[1], axis[0]))
        gz = float(knob[2] + 0.003)
        angle = float(np.radians(rng.uniform(100, 108)))  # past upright, so it stays open when let go
        w = 2 * 0.02
        params = {"grasp": [float(knob[0]), float(knob[1]), yaw, gz], "hinge": hinge.tolist(), "axis": axis.tolist(),
                  "angle": angle, "open_width": gr.open_width(w), "mode": "open"}
        end = hinge + P._axis_rot(axis, angle) @ (np.r_[knob[:2], gz] - hinge)
        return {"action": "lid", "params": params, "sampler": {"object": -1, "fixture": int(f), "mode": "open"},
                "vec": _vec("lid", knob[0], knob[1], gz, end[0], end[1], end[2], yaw, yaw, params["open_width"],
                            P.LIFT_SPEED, angle)}
    if q > 1.0:
        h = rng.uniform(0.05, 0.08)
        lean = h / np.tan(q)  # where the lid is at that height, along the box's x from the hinge (negative: leaning back)
        x0 = float(lean - 0.011 - rng.uniform(0.03, 0.05))
        hz = float(hinge[2] + h)
        start = hinge[:2] + x0 * u + rng.uniform(-0.01, 0.01) * v
        c = slide_cand("lid", *start, float(np.arctan2(u[1], u[0])), float(-x0 + 0.06), -1,
                       {"fixture": int(f), "mode": "close"}, z=hz)
        c["params"]["mode"] = "close"
        return c
    return None


def toss_cand(S, i, rng):
    if S.stacked_on(i) is not None:
        return None
    g = _grasp(S, i, rng)
    if g is None:
        return None
    gx, gy, gyaw, gz, w = g
    conts = [f for f, fx in enumerate(S.fixtures) if fx["kind"] in ("tray", "bin", "bowl")]
    if conts and rng.random() < 0.3:
        aim = np.asarray(S.fixtures[int(rng.choice(conts))]["xy"], float)
    else:
        aim = S.pos[i, :2] + rng.uniform(0.1, 0.25) * np.array([np.cos(a := rng.uniform(-np.pi, np.pi)), np.sin(a)])
    d = aim - np.array([gx, gy])
    theta = float(np.arctan2(d[1], d[0]))
    stroke, speed, release = 0.16, float(rng.uniform(0.35, 0.55)), float(rng.uniform(0.45, 0.7))
    params = {"grasp": [float(gx), float(gy), float(gyaw), float(gz)], "theta": theta, "stroke": stroke,
              "speed": speed, "release": release, "open_width": gr.open_width(w)}
    e = np.array([gx, gy]) + 0.6 * stroke * np.array([np.cos(theta), np.sin(theta)])
    return {"action": "toss", "params": params, "sampler": {"object": i, "width": float(w), "aim": aim.tolist()},
            "vec": _vec("toss", gx, gy, gz, e[0], e[1], P.TOSS_Z, gyaw, gyaw, params["open_width"], speed, release)}


BUILDERS = {
    "push": push_cand, "pull": pull_cand, "sweep": sweep_cand, "poke": poke_cand, "topple": topple_cand,
    "place": lambda S, i, rng: place_cand(S, i, rng, "place"), "drop": lambda S, i, rng: place_cand(S, i, rng, "drop"),
    "stack": lambda S, i, rng: place_cand(S, i, rng, "stack"),
    "unstack": lambda S, i, rng: place_cand(S, i, rng, "unstack"), "twist": twist_cand, "stand_up": stand_up_cand,
    "drawer": drawer_cand, "lid": lid_cand, "toss": toss_cand,
}
FIXTURE_ACTIONS = {"drawer": "cabinet", "lid": "lidbox"}  # these act on a fixture: "object" is a fixture index


GRASP_ACTIONS = ("place", "drop", "stack", "unstack", "twist", "stand_up", "toss")
SLIDES = ("push", "pull", "sweep", "poke", "topple")
FLAT = ("cube", "cylinder", "hex", "bar", "can", "milk", "cereal", "bread")


def eligible(S, action):
    """Objects an action can be tried on in this state (by kind and pose only; the builders may still find no
    grasp or no free spot)."""
    if action in FIXTURE_ACTIONS:
        return [f for f, fx in enumerate(S.fixtures) if fx["kind"] == FIXTURE_ACTIONS[action]]
    out = []
    for i in range(S.n):
        k = S.kind(i)
        if action in GRASP_ACTIONS:
            if k == "prism" or S.has_on_top(i) or (k == "hammer" and action != "place"):
                continue
        if action == "topple" and (k not in cat.TALL or S.zz[i] < 0.9):
            continue
        if action == "stand_up" and (k not in ("tube", "can", "bottle") or abs(S.zz[i]) > 0.35):
            continue
        if action == "unstack" and S.stacked_on(i) is None:
            continue
        if action == "twist" and (k not in ("cube", "L", "T", "hex", "bar", "domino", "milk", "cereal", "bread")
                                  or S.zz[i] < 0.9):
            continue
        if action == "stack" and not any(j != i and S.kind(j) in FLAT and S.zz[j] > 0.9 for j in range(S.n)):
            continue
        out.append(i)
    return out


def counts(mix, avail, n, rng):
    """Integer counts per action type: mix renormalised over the available types (largest remainder)."""
    w = {a: p for a, p in mix.items() if a in avail}
    tot = sum(w.values())
    raw = {a: n * p / tot for a, p in w.items()}
    c = {a: int(np.floor(v)) for a, v in raw.items()}
    rest = n - sum(c.values())
    for a in sorted(raw, key=lambda a: raw[a] - c[a], reverse=True)[:rest]:
        c[a] += 1
    cap = n // 2
    over = {a: v - cap for a, v in c.items() if v > cap}
    for a, extra in over.items():  # none above half
        c[a] = cap
        others = [b for b in c if b != a]
        for k in range(extra):
            c[others[k % len(others)]] += 1
    return c


# ----------------------------------------------------------------------------- keeping the hand off fixtures
# Fixtures are fixed in place: a hand that runs into one stalls (the arm cannot push it) and the rollout is junk.
# Each fixture is a set of blocking boxes (local cx, cy, hx, hy, top z); the hand is a few discs (two fingers at the
# pads' height, the palm above them). Slides are cut short before they would hit a box; grasp and release poses
# that would hit one are redrawn.
def blocking_boxes(S, f):
    fx = S.fixtures[f]
    k = fx["kind"]
    t = 0.004
    if k == "tray":
        ix, iy = cat.TRAY_IN
        h = cat.TRAY_WALL_H
        return [(-(ix + t), 0, t, iy + 2 * t, h), (-t, iy + t, ix + t, t, h), (-t, -(iy + t), ix + t, t, h)]
    if k == "bin":
        ix, iy = cat.BIN_IN
        h = cat.BIN_WALL_H
        return [(-(ix + t), 0, t, iy + 2 * t, h), (ix + t, 0, t, iy + 2 * t, h), (0, iy + t, ix, t, h), (0, -(iy + t), ix, t, h)]
    if k == "bowl":
        return [("ring", cat.BOWL_R - 0.03, cat.BOWL_R + 0.004, cat.BOWL_H)]
    if k == "platform":
        return [(0, 0, *cat.PLATFORM_HALF[:2], 2 * cat.PLATFORM_HALF[2])]
    if k == "ramp":
        return [(0, 0, cat.RAMP_LEN / 2, cat.RAMP_W / 2 + 0.006, cat.RAMP_H + 0.004)]
    if k == "wall":
        return [(0, 0, *cat.WALL_HALF[:2], 2 * cat.WALL_HALF[2])]
    if k == "seesaw":
        return [(0, 0, cat.SEESAW_PLANK[0] + 0.005, cat.SEESAW_PLANK[1] + 0.006, 0.085)]
    if k == "cabinet":
        hx, hy, hz = cat.CAB_HALF
        slide = float(S.env.data.qpos[S.env.fx_joint["cabinet"][0]]) if S.env is not None else 0.0
        boxes = [(0, 0, hx, hy, 2 * hz)]
        if slide > 0.005:
            boxes.append((hx + slide / 2 + 0.01, 0, slide / 2 + 0.02, cat.DRAWER_IN[1] + 0.008, 2 * cat.DRAWER_IN[2] + 0.02))
        return boxes
    if k == "lidbox":
        ix, iy, iz = cat.LIDBOX_IN
        lid = float(S.env.data.qpos[S.env.fx_joint["lidbox"][0]]) if S.env is not None else 0.0
        top = 2 * iz + 0.03 + (2 * ix + 0.02) * np.sin(min(lid, np.pi / 2))
        if lid > 1.0:  # open: the walls block, the inside is free
            h = 2 * iz + 0.02
            walls = [(-(ix + 0.005), 0, 0.005, iy + 0.01, top), (ix + 0.005, 0, 0.005, iy + 0.01, h),
                     (0, iy + 0.005, ix, 0.005, h), (0, -(iy + 0.005), ix, 0.005, h)]
            return walls
        return [(0, 0, ix + 0.01, iy + 0.01, top)]
    return []


def _box_dist(local, box):
    if box[0] == "ring":
        r = float(np.linalg.norm(local))
        return 0.0 if box[1] <= r <= box[2] else min(abs(r - box[1]), abs(r - box[2]))
    cx, cy, hx, hy, _ = box
    return float(np.linalg.norm(np.maximum(np.abs(local - [cx, cy]) - [hx, hy], 0)))


# The hand, measured from the model in the tool frame (x = closing axis, z up from the pads' centre): the palm is
# 20 cm wide along the closing axis, 6.5 cm across, and its underside is only 3.1 cm above the pads' centre; each
# finger runs from opening/2 to opening/2 + 2.8 cm along x, 2 cm across, down to 0.9 cm below the pads' centre.
PALM = (0.102, 0.033, 0.031)
FINGER = (0.028, 0.011, -0.009)
HAND_MARGIN = 0.006
OBJ_TOL = 0.003  # the palm may graze an object's top by this much (blocks are 4.4 cm tall; the palm's underside is
#                  4.3 cm up when the fingers slide at PUSH_Z)


def _hand_points(xyz, yaw, opening):
    """[(points (N, 2), bottom z)] for the palm and the two fingers, sampled every ~2 cm (with a margin)."""
    a = np.array([np.cos(yaw), np.sin(yaw)])
    b = np.array([-a[1], a[0]])
    c = np.asarray(xyz[:2], float)
    m = HAND_MARGIN
    px, py, pz = PALM
    gx, gy = np.meshgrid(np.linspace(-px - m, px + m, 11), np.linspace(-py - m, py + m, 4))
    parts = [(c + gx.reshape(-1, 1) * a + gy.reshape(-1, 1) * b, xyz[2] + pz)]
    fx, fy, fz = FINGER
    for sgn in (-1, 1):
        x0 = opening / 2
        gx, gy = np.meshgrid(np.linspace(x0 - m / 2, x0 + fx + m, 3), np.linspace(-fy - m, fy + m, 3))
        parts.append((c + sgn * gx.reshape(-1, 1) * a + gy.reshape(-1, 1) * b, xyz[2] + fz))
    return parts


def hand_hits(S, xyz, yaw, opening, objects=False, skip=(), skip_fingers=()):
    """Would the hand at tool position xyz (closing along yaw, fingers `opening` apart) touch a fixture or, with
    objects=True, an object (other than `skip`; the fingers also ignore `skip_fingers`) or a tall distractor?"""
    parts = _hand_points(xyz, yaw, opening)
    for f, fx in enumerate(S.fixtures):
        c, s_ = np.cos(fx["yaw"]), np.sin(fx["yaw"])
        for box in blocking_boxes(S, f):
            top = box[3] if box[0] == "ring" else box[4]
            for pts, bottom in parts:
                if bottom - 0.003 >= top:
                    continue
                d = pts - np.asarray(fx["xy"])
                local = np.stack([c * d[:, 0] + s_ * d[:, 1], -s_ * d[:, 0] + c * d[:, 1]], 1)
                if box[0] == "ring":
                    r = np.linalg.norm(local, axis=1)
                    if np.any((r >= box[1]) & (r <= box[2])):
                        return True
                elif np.any(np.all(np.abs(local - [box[0], box[1]]) <= [box[2], box[3]], axis=1)):
                    return True
    if objects:
        for i in range(S.n):
            if i in skip:
                continue
            top = S.top(i) - OBJ_TOL
            for k, (pts, bottom) in enumerate(parts):
                if bottom >= top or (k > 0 and i in skip_fingers):
                    continue
                if np.any(np.linalg.norm(pts - S.pos[i, :2], axis=1) < S.r[i]):
                    return True
        for k, d in enumerate(S.spec.get("distractors", [])):
            top = 2 * float(S.dis[k, 2])
            for pts, bottom in parts:
                if bottom < top and np.any(np.linalg.norm(pts - S.dis[k, :2], axis=1) < d["radius"]):
                    return True
    return False


def fit_slide(S, c):
    """Cut a slide short of the first fixture it would run into; None if it hits one at the start or if less
    than 3 cm is left."""
    p = c["params"]
    th = p["theta"]
    u = np.array([np.cos(th), np.sin(th)])
    yaw = th + np.pi / 2
    s = np.array([p["x0"], p["y0"]])
    if hand_hits(S, np.r_[s, p["z"]], yaw, p["opening"], objects=True):
        return None  # the hand would come down on something
    if not S.fixtures:
        return c
    steps = np.arange(0.01, p["length"] + 1e-9, 0.01)
    for d in steps:
        if hand_hits(S, np.r_[s + u * d, p["z"]], yaw, p["opening"]):
            L = float(d - 0.015)
            if L < 0.03:
                return None
            e = s + u * L
            c = {**c, "params": {**p, "length": L, "cut": True}}
            c["vec"] = c["vec"][:4] + [float(e[0]), float(e[1])] + c["vec"][6:]
            return c
    return c


def grasp_clear(S, c):
    """The hand comes down around the object without landing on it or on anything else, and lowers it at the
    target without the hand hitting anything there (the held object itself is checked by place_target)."""
    p = c["params"]
    w = p["open_width"]
    i = c["sampler"]["object"]
    gx, gy, gyaw, gz = p["grasp"]
    px, py, pyaw, pz = p["place"]
    under = set(c["sampler"].get("under", ()))
    return not (hand_hits(S, (gx, gy, gz), gyaw, w, objects=True, skip_fingers=(i,))
                or hand_hits(S, (px, py, pz), pyaw, w, objects=True, skip=(i,) + tuple(under)))


# ----------------------------------------------------------------------------- reachability
POS_TOL, ROT_TOL = 0.004, 0.03


JOINT_MARGIN = 0.08  # rad; an IK answer this close to a joint stop means the arm cannot track the path there


def reachable(env, points, R):
    lo, hi = env.jrange[:, 0] + JOINT_MARGIN, env.jrange[:, 1] - JOINT_MARGIN
    for p in points:
        q, ok = env.solve_ik_multi(np.asarray(p, float), R, pos_tol=POS_TOL, rot_tol=ROT_TOL)
        if not ok or np.any(q < lo) or np.any(q > hi):
            return False
    return True


PATH_STEP = 0.01  # m per control step (the primitives move 0.6 to 1.5 cm per 50 ms step)
PATH_TOL = 0.012  # largest lag allowed on the way
END_TOL = 0.005


def path_ok(env, waypoints):
    """Follow the tool path the way the controller will (joint move to the first pose, then IK_ITERS damped
    least-squares iterations per ~1 cm step from the previous answer) and check it keeps up. The multi-start IK
    at single points misses paths that run the elbow into its stop on the way, e.g. rising close to the base.
    waypoints: [(pos, R), ...]."""
    from gen_v6.env import IK_ITERS
    p0, R0 = waypoints[0]
    q, ok = env.solve_ik_multi(np.asarray(p0, float), R0)  # the same call (and so the same posture) as _approach
    if not ok:
        return False
    lo, hi = env.jrange[:, 0] + JOINT_MARGIN, env.jrange[:, 1] - JOINT_MARGIN
    for (pa, Ra), (pb, Rb) in zip(waypoints[:-1], waypoints[1:]):
        pa, pb = np.asarray(pa, float), np.asarray(pb, float)
        qa, qb = P._quat(Ra), P._quat(Rb)
        n = max(1, int(np.ceil(max(np.linalg.norm(pb - pa) / PATH_STEP, P._angle(qa, qb) / 0.1))))
        for k in range(1, n + 1):
            t = k / n
            pt, Rt = pa + t * (pb - pa), P._mat(P._slerp(qa, qb, t))
            for _ in range(IK_ITERS):
                q, ep, er = env.ik_step(q, pt, Rt)
            if ep > PATH_TOL:
                return False
        for _ in range(4 * IK_ITERS):  # the primitives settle at a segment's end
            q, ep, er = env.ik_step(q, pb, Rb)
        if ep > END_TOL or er > 3 * ROT_TOL or np.any(q < lo - 0.06) or np.any(q > hi + 0.06):
            return False
    return True


def home_yaw(env):
    R = env.home["rot"]
    return float(np.arctan2(R[1, 0], R[0, 0]))


def feasible(env, c):
    """IK check at the motion's key points, with the hand turned exactly as the primitive will turn it."""
    a, p = c["action"], c["params"]
    hy = home_yaw(env)
    if a in ("push", "pull", "sweep", "poke", "topple", "drawer") or (a == "lid" and p["mode"] == "close"):
        th = p["theta"]
        R = P.tool_rot(P.near_yaw(th + np.pi / 2, hy))
        s = np.array([p["x0"], p["y0"]])
        e = s + p["length"] * np.array([np.cos(th), np.sin(th)])
        return path_ok(env, [(np.r_[s, P.HIGH_Z], R), (np.r_[s, p["z"]], R), (np.r_[e, p["z"]], R), (np.r_[e, P.HIGH_Z], R)])
    if a in ("place", "drop", "stack", "unstack", "twist"):
        gx, gy, gyaw, gz = p["grasp"]
        px, py, pyaw, pz = p["place"]
        gyaw2 = P.near_yaw(gyaw, hy)
        pyaw2 = gyaw2 + wrap(P.near_yaw(pyaw, gyaw2) - gyaw2)
        Rg, Rp = P.tool_rot(gyaw2), P.tool_rot(pyaw2)
        return path_ok(env, [(np.r_[gx, gy, P.HIGH_Z], Rg), (np.r_[gx, gy, gz], Rg), (np.r_[gx, gy, P.CARRY_Z], Rg),
                             (np.r_[px, py, P.CARRY_Z], Rp), (np.r_[px, py, pz], Rp), (np.r_[px, py, P.HIGH_Z], Rp)])
    if a == "lid":
        gx, gy, gyaw, gz = p["grasp"]
        R0 = P.tool_rot(P.near_yaw(gyaw, hy))
        k0, h = np.array([gx, gy, gz]), np.asarray(p["hinge"])
        arc = [(h + P._axis_rot(p["axis"], p["angle"] * j / P.LID_STEPS) @ (k0 - h),
                P._axis_rot(p["axis"], min(p["angle"], P.LID_TILT) * j / P.LID_STEPS) @ R0) for j in range(P.LID_STEPS + 1)]
        return path_ok(env, [(np.r_[gx, gy, P.HIGH_Z], R0)] + arc)
    if a == "toss":
        gx, gy, gyaw, gz = p["grasp"]
        R0 = P.tool_rot(P.near_yaw(gyaw, hy))
        u = np.r_[np.cos(p["theta"]), np.sin(p["theta"]), 0.0]
        top = np.r_[gx, gy, P.TOSS_Z]
        back = top - 0.4 * p["stroke"] * u
        return path_ok(env, [(np.r_[gx, gy, P.HIGH_Z], R0), (np.r_[gx, gy, gz], R0), (top, R0), (back, R0),
                             (back + p["stroke"] * u, R0), (back + p["stroke"] * u + np.r_[0, 0, P.HIGH_Z - P.TOSS_Z], R0)])
    if a == "stand_up":
        gx, gy, gyaw, gz = p["grasp"]
        gyaw = P.near_yaw(gyaw, hy)
        px, py, _, pz = p["place"]
        if not reachable(env, [np.r_[gx, gy, P.HIGH_Z], np.r_[gx, gy, gz]], P.tool_rot(gyaw)):
            return False
        to_base = np.r_[np.array([-0.52, 0.0]) - np.array([px, py]), 0.0]
        Rt = max((P.tool_rot(gyaw, s * np.pi / 2) for s in (1, -1)), key=lambda R: R[:, 2] @ to_base)
        return reachable(env, [np.r_[gx, gy, P.CARRY_Z], np.r_[px, py, P.CARRY_Z], np.r_[px, py, pz]], Rt)
    return True


# ----------------------------------------------------------------------------- the sampler
def sample(env, spec, rng, n_total=N_TOTAL, n_lines=N_LINES, family=None, max_tries=6):
    """n_total feasible goal-blind candidates for the env's current state. Returns (candidates, n_dropped)."""
    S = Scene(env, spec)
    mix = ACTION_MIX[family or spec["family"]]
    avail = {a for a in mix if a in BUILDERS and eligible(S, a)}
    if len(avail) < 3:  # fall back on the slides, which work on anything
        avail |= {"push", "poke", "pull"}
    n_obj = n_total - n_lines
    plan = counts(mix, avail, n_obj, rng)
    out, dropped = [], 0
    for a, k in plan.items():
        objs = eligible(S, a)
        got = 0
        order = list(rng.permutation(objs)) * (k // max(len(objs), 1) + 2)
        tries = 0
        while got < k and tries < k * max_tries and order:
            i = int(order[tries % len(order)])
            tries += 1
            c = BUILDERS[a](S, i, rng)
            if c is not None and a not in FIXTURE_ACTIONS:
                if a == "toss":
                    g = c["params"]["grasp"]
                    c = None if hand_hits(S, g[:2] + [g[3]], g[2], c["params"]["open_width"], objects=True,
                                          skip_fingers=(i,)) else c
                else:
                    c = fit_slide(S, c) if a in SLIDES else (c if grasp_clear(S, c) else None)
            if c is None or not feasible(env, c):
                dropped += c is not None
                continue
            out.append(c)
            got += 1
    pushable = eligible(S, "push")
    for t in range(20 * n_total):  # random lines, and whatever the other types could not fill (lines and pushes)
        if len(out) >= n_total:
            break
        c = line_cand(S, rng) if t % 2 == 0 or not pushable else push_cand(S, int(rng.choice(pushable)), rng)
        if c is None:
            continue
        c = fit_slide(S, c)
        if c is not None and feasible(env, c):
            out.append(c)
    return [out[k] for k in rng.permutation(len(out))], dropped


def execute(env, c, **kw):
    """Run candidate c with the matching primitive. kw: frame_fn, video_fn, log, push_idx, frame_names."""
    a, p = c["action"], c["params"]
    if a in ("push", "pull", "sweep", "topple"):
        return P.slide(env, p["x0"], p["y0"], p["theta"], p["length"], opening=p["opening"], z=p["z"], **kw)
    if a == "poke":
        return P.poke(env, p["x0"], p["y0"], p["theta"], p["length"], speed=p["speed"], **kw)
    if a in ("place", "drop", "stack", "unstack", "twist"):
        return P.pick_place(env, p["grasp"], p["place"], open_width=p["open_width"], **kw)
    if a == "stand_up":
        return P.stand_up(env, p["grasp"], p["place"], tilt=p["tilt"], open_width=p["open_width"], **kw)
    if a == "drawer" or (a == "lid" and p["mode"] == "close"):
        return P.slide(env, p["x0"], p["y0"], p["theta"], p["length"], opening=p["opening"], z=p["z"], **kw)
    if a == "lid":
        return P.lid_open(env, p["grasp"], p["hinge"], p["axis"], p["angle"], open_width=p["open_width"], **kw)
    if a == "toss":
        return P.toss(env, p["grasp"], p["theta"], p["stroke"], p["speed"], p["release"], open_width=p["open_width"],
                      **kw)
    raise ValueError(a)
