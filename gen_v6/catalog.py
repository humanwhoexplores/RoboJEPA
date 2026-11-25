"""What a v6 scene can contain: movable objects (a fixed pool of bodies, re-dressed per scene) and fixtures
(mocap bodies a scene places or parks off the table).

Objects
    blocks      v3's 7 shapes x 2 sizes x 2 copies (built by gen_v3.scene_xml, same geometry and density)
    ball        sphere, 2 sizes x 2 copies; rolls
    tube        tall thin cylinder x 2; stands upright or lies on its side (then it rolls)
    domino      thin tall slab x 6; for rows that knock each other over
    can, milk, cereal, bread, lemon, bottle
                robosuite 1.5.2's everyday objects (its meshes and textures), 2 cans, 1 of each other
    hammer      (stage 2) a handle and a head; a tool for pulling far objects
Colourable kinds take one of v3's six colours; robosuite objects keep their own texture and are named by kind.

Fixtures (one of each kind at most per scene; parked far away when unused)
    tray        three low walls and an open side: things can be pushed or swept in, or put in
    bin         four 5 cm walls: only a place or a drop gets something in
    bowl        a round bowl with sloped sides: dropped things roll to the middle
    platform    a raised 12 x 12 x 4 cm block: put things on it
    ramp        a 20 cm incline, 4 cm high at its top end: balls and lying tubes roll down
    wall        a low 24 cm wall: things bounce off it or get pinned against it
    seesaw      a plank on a hinge: dropping or pushing onto one end lifts the other
    cabinet     (stage 2) a box with a sliding drawer and a handle
    lidbox      (stage 2) a box with a hinged lid and a knob

Hidden physical properties (per object, per scene; none of it shows in the image)
    mass_scale  x0.3 to x3 of the kind's normal mass (log-uniform)
    friction    sliding friction 0.2 to 1.2 (normal 0.8)
    bounce      0 (dead, critically damped contact) to 1 (lively)
"""
import numpy as np

from gen_v3 import scene_xml as sx

# ----------------------------------------------------------------------------- movable objects
BLOCK_SHAPES = list(sx.SHAPES)
BALL_R = {"s": 0.022, "l": 0.028}
TUBE_R, TUBE_HALF = 0.017, 0.05
DOMINO_HALF = (0.006, 0.022, 0.04)  # thickness, width, height (half extents)
N_DOMINOES = 6
RS_OBJECTS = {  # robosuite kind: (copies, mass kg, words)
    "can": (2, 0.08, ["soda can", "can"]),
    "milk": (1, 0.10, ["milk carton", "milk"]),
    "cereal": (1, 0.09, ["cereal box", "cereal"]),
    "bread": (1, 0.03, ["bread", "loaf"]),
    "lemon": (1, 0.04, ["lemon"]),
    "bottle": (1, 0.08, ["bottle"]),
}
HAMMER = {"handle_half": (0.09, 0.009, 0.009), "head_half": (0.012, 0.035, 0.014), "mass": 0.12}
COLOURABLE = set(BLOCK_SHAPES) | {"ball", "tube", "domino"}
TALL = {"tube", "domino", "can", "milk", "cereal", "bottle"}  # can be knocked over / stood up
ROLLS = {"ball", "tube", "can", "bottle", "lemon"}  # round in some pose


def object_names(stage2=True):
    """Every pool body name, v3 blocks first (their names are v3's, e.g. 'cube_s_0')."""
    names = list(sx.object_names())
    names += [f"ball_{s}_{k}" for s in BALL_R for k in range(2)]
    names += [f"tube_s_{k}" for k in range(2)]
    names += [f"domino_s_{k}" for k in range(N_DOMINOES)]
    names += [f"{kind}_s_{k}" for kind, (n, _, _) in RS_OBJECTS.items() for k in range(n)]
    if stage2:
        names += ["hammer_s_0"]
    return names


def kind_of(pool_name):
    return pool_name.split("_")[0]


def size_of(pool_name):
    return pool_name.split("_")[1]


# ----------------------------------------------------------------------------- fixtures
FIXTURES = ["tray", "bin", "bowl", "platform", "ramp", "wall", "seesaw"]
FIXTURES_STAGE2 = ["cabinet", "lidbox"]
FIXTURE_RGBA = {
    "tray": [0.80, 0.68, 0.50, 1], "bin": [0.30, 0.34, 0.42, 1], "bowl": [0.93, 0.93, 0.90, 1],
    "platform": [0.62, 0.52, 0.40, 1], "ramp": [0.50, 0.52, 0.55, 1], "wall": [0.66, 0.36, 0.30, 1],
    "seesaw": [0.85, 0.75, 0.35, 1], "cabinet": [0.55, 0.42, 0.30, 1], "lidbox": [0.35, 0.55, 0.60, 1],
}
TRAY_IN = (0.07, 0.05)  # interior half extents (x along the open side's normal, y across)
TRAY_WALL_H = 0.018
BIN_IN = (0.06, 0.06)
BIN_WALL_H = 0.05
BOWL_R, BOWL_H, BOWL_FLOOR_R = 0.065, 0.035, 0.035
PLATFORM_HALF = (0.06, 0.06, 0.02)
RAMP_LEN, RAMP_W, RAMP_H = 0.20, 0.10, 0.04
WALL_HALF = (0.12, 0.008, 0.03)
SEESAW_PLANK = (0.11, 0.035, 0.005)
SEESAW_PIVOT_H = 0.035
# stage 2
CAB_HALF = (0.08, 0.10, 0.06)  # outer half extents of the cabinet (x = depth, drawer slides along +x)
DRAWER_IN = (0.06, 0.08, 0.035)  # drawer interior half extents
DRAWER_TRAVEL = 0.10
HANDLE_OUT = 0.06  # drawer handle bar: centre this far in front of the drawer front
HANDLE_HALF_LEN = 0.05
DRAWER_Z0 = 0.014  # drawer body above the cabinet floor (2 * wall + 2 mm)
HANDLE_Z = DRAWER_Z0 + DRAWER_IN[2] + 0.004  # handle height above the table
LIDBOX_IN = (0.05, 0.06, 0.03)
FIXTURE_WORDS = {"tray": ["tray"], "bin": ["bin", "box"], "bowl": ["bowl"], "platform": ["platform", "raised block"],
                 "ramp": ["ramp", "slope"], "wall": ["wall", "barrier"], "seesaw": ["seesaw"],
                 "cabinet": ["drawer"], "lidbox": ["lidded box", "box with the lid"]}
CONTAINERS = ("tray", "bin", "bowl", "platform", "cabinet", "lidbox")


def fixture_footprint(kind):
    """Half extents (x, y) of a fixture's footprint in its own frame, for keeping scenes uncluttered."""
    return {"tray": (TRAY_IN[0] + 0.008, TRAY_IN[1] + 0.008), "bin": (BIN_IN[0] + 0.008, BIN_IN[1] + 0.008),
            "bowl": (BOWL_R + 0.01, BOWL_R + 0.01), "platform": PLATFORM_HALF[:2],
            "ramp": (RAMP_LEN / 2, RAMP_W / 2), "wall": WALL_HALF[:2], "seesaw": SEESAW_PLANK[:2],
            "cabinet": (CAB_HALF[0] + HANDLE_OUT + 0.01, CAB_HALF[1]), "lidbox": (LIDBOX_IN[0] + 0.01, LIDBOX_IN[1] + 0.01)}[kind]


# ----------------------------------------------------------------------------- hidden properties
MASS_RANGE = (0.3, 3.0)
FRICTION_RANGE = (0.2, 1.2)
MASS_TRAIN = (0.4, 2.5)  # test_phys holds out the most extreme values (outside these) for test scenes only
FRICTION_TRAIN = (0.3, 1.1)
NORMAL = {"mass_scale": 1.0, "friction": 0.8, "bounce": 0.0}


def bounce_solref(b):
    """Contact (timeconst, dampratio) for bounciness b in [0, 1]: 1 -> lightly damped, springy."""
    return [0.02, float(1.0 - 0.85 * b)]  # b = 1: a block dropped from 23 cm rebounds about 9 cm


def extreme(props):
    m, f = props["mass_scale"], props["friction"]
    return not (MASS_TRAIN[0] <= m <= MASS_TRAIN[1] and FRICTION_TRAIN[0] <= f <= FRICTION_TRAIN[1])


def sample_props(rng, extreme_ok=False, normal_p=0.0):
    """Hidden properties of one object. extreme_ok: draw from the full ranges (test_phys), else from the
    training ranges."""
    if rng.random() < normal_p:
        return dict(NORMAL)
    lo_m, hi_m = MASS_RANGE if extreme_ok else MASS_TRAIN
    lo_f, hi_f = FRICTION_RANGE if extreme_ok else FRICTION_TRAIN
    return {"mass_scale": float(np.exp(rng.uniform(np.log(lo_m), np.log(hi_m)))),
            "friction": float(rng.uniform(lo_f, hi_f)), "bounce": float(rng.uniform(0, 1) ** 2)}
