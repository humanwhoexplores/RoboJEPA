"""MJCF for the generator v3 tabletop: built once, re-dressed per scene at runtime.

The model holds a fixed pool of objects (every shape x size, COPIES each), a pool
of distractors (never named in a goal), a pool of zone discs, the V pusher, all
cameras and, optionally, a robot arm holding the pusher. A scene picks some objects,
distractors and zones, places and recolors them; unused ones are parked far off the table.

The arm (build_xml(arm=True)) is a chain of mocap links posed by inverse kinematics so
its hand always meets the pusher. Its links do not collide with anything, so the push
physics are identical with or without it; it only changes what the cameras see
(including the occlusions a real arm causes).

World frame: table top at z = 0, table centered on the origin. The default
camera "cam0" sits at +x looking back at the table, so image-left is world -y
(same convention as v1's pushview).
"""
import numpy as np

TABLE_HALF = 0.35  # table is 0.70 x 0.70 m
FLOOR_Z = -0.75  # anything pushed off the table lands here

SHAPES = ["cube", "cylinder", "prism", "L", "T", "hex", "bar"]
SIZES = {"s": 0.024, "l": 0.032}  # footprint half-extent (m): about 15 to 20 px wide at 224 px
HEIGHT_HALF = {"s": 0.022, "l": 0.026}
COPIES = 2  # each (shape, size) can appear at most twice in a scene
DENSITY = 500  # same as v1's cube

COLORS = {
    "red": [0.85, 0.15, 0.15, 1.0],
    "green": [0.15, 0.65, 0.25, 1.0],
    "blue": [0.15, 0.30, 0.90, 1.0],
    "yellow": [0.95, 0.80, 0.20, 1.0],
    "purple": [0.55, 0.25, 0.75, 1.0],
    "orange": [0.95, 0.50, 0.10, 1.0],
}

# Distractors: neutral-colored clutter that can be pushed but is never named in a goal.
DISTRACTORS = {"puck": (0.020, 0.008), "post": (0.012, 0.045)}  # (radius, half-height)
DISTRACTOR_COPIES = 2
DISTRACTOR_COLORS = {
    "white": [0.92, 0.92, 0.90, 1.0],
    "gray": [0.55, 0.55, 0.57, 1.0],
    "black": [0.12, 0.12, 0.13, 1.0],
    "brown": [0.45, 0.30, 0.18, 1.0],
}

N_ZONES = 6
ZONE_RADIUS = 0.05

# The pusher is a V of two thin plates, opening in the push direction (body +x).
# A flat or round pusher lets objects slide sideways off the push line; the V
# keeps them centered (the classic "funnel" push).
PUSHER_HALF_THICK = 0.006  # plate half-thickness
PUSHER_PLATE_HALF = 0.04  # plate half-length
PUSHER_PLATE_Y = 0.03  # plate centers at y = +-this
PUSHER_ANGLE = np.radians(40)  # each plate's angle to the push direction
PUSHER_HALF_LEN = 0.03  # plate half-height
# Where the plates' inner faces meet on the push axis (pusher frame, m).
PUSHER_APEX_X = (PUSHER_HALF_THICK * np.sin(PUSHER_ANGLE)
                 - (PUSHER_PLATE_Y - PUSHER_HALF_THICK * np.cos(PUSHER_ANGLE)) / np.tan(PUSHER_ANGLE))
_tip = np.array([PUSHER_PLATE_HALF * np.cos(PUSHER_ANGLE), PUSHER_PLATE_Y + PUSHER_PLATE_HALF * np.sin(PUSHER_ANGLE)])
PUSHER_HALF_SPAN = float(_tip[1] + PUSHER_HALF_THICK)  # half-width of the V at its open end


def seat_x(r_obj):
    """x (pusher frame) of the center of an object of radius r_obj sitting in the V."""
    return float(PUSHER_APEX_X + r_obj / np.sin(PUSHER_ANGLE))


PARK = np.array([0.0, 0.0, 1.5])  # pusher parking spot: above and behind every camera
ARM_HOME = np.array([-0.28, 0.0, 0.22])  # with an arm, the pusher waits here instead (in view, behind the table)

# Robot arm (visual only): base column behind the table at -x, shoulder on top.
ARM_BASE = np.array([-0.42, 0.0])
ARM_SHOULDER_Z = 0.32
ARM_L1, ARM_L2 = 0.46, 0.46  # upper arm, forearm
ARM_WRIST = 0.09  # vertical link from the wrist down to the pusher center
ARM_RADIUS = 0.022

TIMESTEP = 0.002


def look_at(pos, target):
    """MuJoCo xyaxes for a camera at pos looking at target, with world +z as up."""
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    right = np.cross(f, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    return np.r_[right, up]


def orbit(az_deg, el_deg, dist, target):
    az, el = np.radians(az_deg), np.radians(el_deg)
    t = np.asarray(target)
    return t + dist * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]), t


# 5 training views and 2 held out for OOD tests (used from milestone 4 on). Each is
# framed so the whole working area (|x|, |y| <= 0.25 m) is in view at 224 px.
CAMERAS = {
    "cam0": orbit(0, 55, 0.76, (0.04, 0, 0)),
    "cam1": orbit(20, 58, 0.84, (0.04, 0, 0)),
    "cam2": orbit(-20, 58, 0.84, (0.04, 0, 0)),
    "cam3": orbit(0, 70, 0.72, (0.0, 0, 0)),
    "cam4": orbit(0, 45, 0.78, (0.04, 0, 0)),
    "held1": orbit(35, 52, 0.88, (0.02, 0, 0)),
    "held2": orbit(-35, 62, 0.90, (0.02, 0, 0)),
}


def object_names():
    """Pool object names, e.g. 'cube_s_0'."""
    return [f"{shape}_{size}_{k}" for shape in SHAPES for size in SIZES for k in range(COPIES)]


def distractor_names():
    """Pool distractor names, e.g. 'puck_0'."""
    return [f"{kind}_{k}" for kind in DISTRACTORS for k in range(DISTRACTOR_COPIES)]


ARM_LINKS = ("arm_upper", "arm_fore", "arm_wrist")


def footprint_radius(shape, size):
    """Distance from the body center to the face a push hits (approximate for L / T / prism)."""
    a = SIZES[size]
    return {"cube": a, "cylinder": a, "prism": 0.5 * a * 1.4, "L": a, "T": a, "hex": a * 0.93, "bar": 0.6 * a}[shape]


def _fmt(v):
    return " ".join(f"{x:.5g}" for x in v)


def _shape_geoms(name, shape, size):
    a, h = SIZES[size], HEIGHT_HALF[size]
    common = f'density="{DENSITY}" rgba="0.5 0.5 0.5 1" friction="0.8 0.005 0.0001"'
    if shape == "cube":
        return [f'<geom name="{name}_g0" type="box" size="{a} {a} {h}" {common}/>']
    if shape == "cylinder":
        return [f'<geom name="{name}_g0" type="cylinder" size="{a} {h}" {common}/>']
    if shape == "prism":
        return [f'<geom name="{name}_g0" type="mesh" mesh="prism_{size}" {common}/>']
    if shape == "hex":
        return [f'<geom name="{name}_g0" type="mesh" mesh="hex_{size}" {common}/>']
    if shape == "bar":  # long box: 3.2a x 1.0a footprint
        return [f'<geom name="{name}_g0" type="box" size="{1.6 * a} {0.5 * a} {0.8 * h}" {common}/>']
    t = a / 3  # bar half-thickness
    if shape == "L":
        return [
            f'<geom name="{name}_g0" type="box" size="{a} {t} {h}" pos="0 {-(a - t)} 0" {common}/>',
            f'<geom name="{name}_g1" type="box" size="{t} {a} {h}" pos="{-(a - t)} 0 0" {common}/>',
        ]
    if shape == "T":
        return [
            f'<geom name="{name}_g0" type="box" size="{a} {t} {h}" pos="0 {a - t} 0" {common}/>',
            f'<geom name="{name}_g1" type="box" size="{t} {a} {h}" pos="0 0 0" {common}/>',
        ]
    raise ValueError(shape)


def _prism_mesh(size, name="prism", angles=(90, 210, 330)):
    a, h = SIZES[size], HEIGHT_HALF[size]
    verts = []
    for z in (-h, h):
        for ang in angles:
            r = np.radians(ang)
            verts += [a * np.cos(r), a * np.sin(r), z]
    return f'<mesh name="{name}_{size}" vertex="{_fmt(verts)}"/>'


def build_xml(arm=False):
    parts = [
        '<mujoco model="gen_v3_table">',
        f'<option timestep="{TIMESTEP}" integrator="implicitfast" iterations="50"><flag multiccd="enable"/></option>',
        '<visual><global offwidth="640" offheight="640"/><quality shadowsize="2048"/></visual>',
        "<asset>",
        '<texture name="grid" type="2d" builtin="checker" rgb1=".55 .55 .55" rgb2=".5 .5 .5" width="64" height="64"/>',
        '<material name="table_mat" texture="grid" texrepeat="6 6" rgba="1 1 1 1"/>',
        *(_prism_mesh(s) for s in SIZES),
        *(_prism_mesh(s, "hex", range(0, 360, 60)) for s in SIZES),
        "</asset>",
        "<worldbody>",
        '<light name="key" pos="0.3 -0.3 1.2" dir="-0.3 0.3 -1" directional="true" diffuse="0.7 0.7 0.7" castshadow="false"/>',
        '<light name="fill" pos="-0.3 0.3 1.0" dir="0.3 -0.3 -1" directional="true" diffuse="0.3 0.3 0.3" castshadow="false"/>',
        f'<geom name="floor" type="plane" size="5 5 0.1" pos="0 0 {FLOOR_Z}" rgba="0.25 0.25 0.28 1"/>',
        f'<geom name="table" type="box" size="{TABLE_HALF} {TABLE_HALF} 0.02" pos="0 0 -0.02" material="table_mat" friction="0.8 0.005 0.0001"/>',
    ]
    for name, (pos, target) in CAMERAS.items():
        parts.append(f'<camera name="{name}" pos="{_fmt(pos)}" xyaxes="{_fmt(look_at(pos, target))}" fovy="45"/>')
    # Zones: visual-only discs on mocap bodies so a scene can move them.
    for i in range(N_ZONES):
        parts.append(
            f'<body name="zone_{i}" mocap="true" pos="{3 + i} 3 {FLOOR_Z}">'
            f'<geom name="zone_{i}_g" type="cylinder" size="{ZONE_RADIUS} 0.0005" contype="0" conaffinity="0" rgba="1 1 1 1"/>'
            "</body>"
        )
    # Floating pusher: a kinematic (mocap) V; its yaw is set per push.
    a = np.degrees(PUSHER_ANGLE)
    plate = f'type="box" size="{PUSHER_PLATE_HALF} {PUSHER_HALF_THICK} {PUSHER_HALF_LEN}" rgba="0.35 0.35 0.38 1" friction="0.8 0.005 0.0001"'
    parts.append(
        f'<body name="pusher" mocap="true" pos="{_fmt(PARK)}">'
        f'<geom name="pusher_l" pos="0 {PUSHER_PLATE_Y} 0" euler="0 0 {a:.4f}" {plate}/>'
        f'<geom name="pusher_r" pos="0 {-PUSHER_PLATE_Y} 0" euler="0 0 {-a:.4f}" {plate}/>'
        "</body>"
    )
    if arm:
        # Visual-only links (contype/conaffinity 0): posed every control step by SceneEnv.pose_arm.
        vis = 'contype="0" conaffinity="0" rgba="0.82 0.82 0.85 1"'
        parts.append(f'<geom name="arm_base" type="cylinder" size="0.05 {ARM_SHOULDER_Z / 2}" '
                     f'pos="{ARM_BASE[0]} {ARM_BASE[1]} {ARM_SHOULDER_Z / 2}" {vis}/>')
        parts.append(f'<geom name="arm_shoulder" type="sphere" size="0.04" pos="{ARM_BASE[0]} {ARM_BASE[1]} {ARM_SHOULDER_Z}" {vis}/>')
        for name, half in (("arm_upper", ARM_L1 / 2), ("arm_fore", ARM_L2 / 2), ("arm_wrist", ARM_WRIST / 2)):
            parts.append(f'<body name="{name}" mocap="true" pos="0 0 2">'
                         f'<geom name="{name}_g" type="capsule" size="{ARM_RADIUS} {half}" {vis}/>'
                         + (f'<geom name="{name}_joint" type="sphere" size="{ARM_RADIUS * 1.4}" pos="0 0 {-half}" '
                            f'contype="0" conaffinity="0" rgba="0.25 0.25 0.28 1"/>' if name != "arm_upper" else "")
                         + "</body>")
    # Object pool, parked in a row on the floor far from the table.
    for i, name in enumerate(object_names()):
        shape, size, _ = name.split("_")
        parts.append(f'<body name="{name}" pos="{-3 - 0.2 * i} -3 {FLOOR_Z + 0.05}"><freejoint name="{name}_j"/>')
        parts += _shape_geoms(name, shape, size)
        parts.append("</body>")
    for i, name in enumerate(distractor_names()):
        kind = name.split("_")[0]
        r, h = DISTRACTORS[kind]
        parts.append(f'<body name="{name}" pos="{3 + 0.2 * i} -3 {FLOOR_Z + 0.05}"><freejoint name="{name}_j"/>'
                     f'<geom name="{name}_g0" type="cylinder" size="{r} {h}" density="{DENSITY}" rgba="0.5 0.5 0.5 1" '
                     f'friction="0.8 0.005 0.0001"/></body>')
    parts += ["</worldbody>", "</mujoco>"]
    return "\n".join(parts)
