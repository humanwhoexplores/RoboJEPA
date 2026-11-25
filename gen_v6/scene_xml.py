"""MJCF for v6: v5's tabletop and Panda arm, with robosuite's Panda parallel gripper on the flange instead of
v5's rigid V pusher, plus v6's extra objects and fixtures (gen_v6.catalog).

The arm is built exactly as in gen_v5.scene_xml (robosuite Panda, joint position servos, gravity compensation).
The gripper is robosuite's panda_gripper.xml (two slide-joint fingers, 0 to 4 cm each, so an 8 cm opening;
position actuators, 20 N per finger). A "tool" site sits at the fingertip pads with:
    z = world up when the hand points down (opposite the approach direction)
    x = the closing axis (the line between the two pads)
so gen_v5.primitives.yaw_rot(theta) asks for a hand pointing down whose fingers close along theta.

Control vector: ctrl[0:7] arm joint targets, ctrl[7:9] finger targets (+w/2, -w/2 for an opening w).

Pool bodies are named "<kind>_<size>_<copy>" (v3's names for blocks); fixtures are mocap bodies "fx_<kind>", with
any moving part ("fx_seesaw_plank", "fx_cabinet_drawer", "fx_lidbox_lid") on a joint named "<body>_j".
"""
import os
import re

import mujoco
import numpy as np

from gen_v3 import scene_xml as sx
from gen_v5 import scene_xml as vx
from gen_v6 import catalog as cat

BASE = vx.BASE
HOME_Q = vx.HOME_Q
KP, KV, ARMATURE = vx.KP, vx.KV, vx.ARMATURE
MAX_OPEN = 0.08  # m, full gripper opening
N_ARM = 7
IMPRATIO = 10.0


def _robosuite_root():
    import importlib.util
    return os.path.dirname(importlib.util.find_spec("robosuite").origin)


def gripper_xml():
    return os.path.join(_robosuite_root(), "models/assets/grippers/panda_gripper.xml")


def _servo_arm(arm):
    """robosuite's torque-driven Panda -> stiff joint position servos (same as gen_v5.scene_xml.build_model)."""
    rad_range = {j.name: np.array(j.range) for j in arm.joints}
    for j in arm.joints:
        j.range = np.degrees(rad_range[j.name])
    for act in arm.actuators:
        k = int(act.target[-1]) - 1
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.gainprm[0] = KP[k]
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        act.biasprm[:3] = [0.0, -KP[k], -KV[k]]
        arm.joint(act.target).armature = ARMATURE[k]
        act.ctrlrange = rad_range[act.target]
        act.forcerange = [-87.0, 87.0] if act.target in ("joint1", "joint2", "joint3", "joint4") else [-12.0, 12.0]
        act.forcelimited = True
    for b in arm.bodies:
        b.gravcomp = 1.0


def build_spec(extra=None):
    """The v6 scene as an MjSpec (not compiled). extra(spec) may add bodies (objects, structures) before compiling."""
    xml = sx.build_xml(arm=False)
    xml = re.sub(r'<body name="pusher".*?</body>', "", xml, flags=re.S)  # v3's floating pusher
    spec = mujoco.MjSpec.from_string(xml)
    spec.option.timestep = sx.TIMESTEP
    # Elliptic friction cones with impratio 10 (MuJoCo's advice for grasping): with v3's pyramidal cones and
    # impratio 1 a held ball creeps down between the pads as it is lifted and drops out before the carry.
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = IMPRATIO

    arm = mujoco.MjSpec.from_file(vx.panda_xml())
    _servo_arm(arm)
    frame = spec.worldbody.add_frame(pos=BASE)
    frame.attach_body(arm.body("base"), "panda_", "")
    spec.worldbody.add_geom(name="pedestal", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                            size=[0.09, (0 - sx.FLOOR_Z) / 2, 0], pos=[BASE[0], BASE[1], sx.FLOOR_Z / 2],
                            rgba=[0.2, 0.2, 0.22, 1], contype=0, conaffinity=0)

    grip = mujoco.MjSpec.from_file(gripper_xml())
    for b in grip.bodies:
        b.gravcomp = 1.0  # the arm's servos track; the fingers' own position servos hold the opening
    for g in grip.geoms:
        if g.name.endswith("pad_collision"):
            # Torsional friction on the pads (robosuite leaves them at condim 3). Mesh objects touch a pad at a
            # single point, and without it they pivot about the closing axis and hang tilted.
            g.condim = 4
            g.priority = 2  # the pads' own friction holds every grasp, whatever an object's hidden friction
    for s in grip.sites:
        s.rgba = [0, 0, 0, 0]  # robosuite's visual markers (a 20 m green line along the approach axis, red dots)
    hand = spec.body("panda_right_hand")
    hand.add_frame().attach_body(grip.body("right_gripper"), "gripper0_", "")
    # Tool site at the pads: grip_site's frame turned half a turn about its x (the closing axis), so z points up.
    eef = spec.body("gripper0_eef")
    eef.add_site(name="tool", pos=[0, 0, 0], quat=[0.0, 1.0, 0.0, 0.0], size=[0.005, 0, 0], rgba=[0, 0, 0, 0])
    add_objects(spec)
    _hex_boxes(spec)
    for name in cat.object_names():
        for g in spec.body(name).geoms:
            # An object's own contact parameters win against the table, fixtures and fingers (priority 0), so its
            # hidden friction and bounce take effect (MuJoCo otherwise takes the larger friction of the two).
            g.priority = 1
    add_fixtures(spec)
    if extra is not None:
        extra(spec)
    return spec


def _hex_boxes(spec):
    """Collide hexes as three boxes turned 0, 60 and 120 degrees, whose union is exactly the hexagonal prism. The
    mesh stays for looks and mass: a heavy mesh hex rocked on one or two contact points and crept 3 cm in 5 s on a
    bare table, where box-box contacts hold it still."""
    for b in spec.bodies:
        if not b.name.startswith("hex_"):
            continue
        size = b.name.split("_")[1]
        a, h = sx.SIZES[size], sx.HEIGHT_HALF[size]
        mesh = [g for g in b.geoms if g.type == G.mjGEOM_MESH][0]
        mesh.contype, mesh.conaffinity = 0, 0
        for k, ang in enumerate((0, 60, 120)):
            r = np.radians(ang) / 2
            b.add_geom(name=f"{b.name}_c{k}", type=G.mjGEOM_BOX, size=[a / 2, a * np.sqrt(3) / 2, h],
                       quat=[np.cos(r), 0, 0, np.sin(r)], density=0, friction=list(mesh.friction), group=3,
                       rgba=[0, 0, 0, 0])


def build_model(extra=None):
    return build_spec(extra).compile()


# ----------------------------------------------------------------------------- v6 objects
G = mujoco.mjtGeom
BLOCK_FRICTION = [0.8, 0.005, 0.0001]  # v3's blocks
# Things that roll get rolling resistance (condim 6): a ball leaving the ramp
# as on a cloth-covered table, instead of rolling off the far edge (a 0.75 m/s ball stops in about 20 cm).
ROLL_FRICTION = [0.8, 0.005, 0.006]
PARK_ROW = (-3.0, -3.0)  # pool bodies wait in a row on the floor: x = PARK_ROW[0] - 0.2 * index


def _box(body, size, pos=(0, 0, 0), quat=None, **kw):
    g = body.add_geom(type=G.mjGEOM_BOX, size=list(size), pos=list(pos), **kw)
    if quat is not None:
        g.quat = list(quat)
    return g


def _free_body(spec, name, index):
    b = spec.worldbody.add_body(name=name, pos=[PARK_ROW[0] - 0.2 * index, PARK_ROW[1], sx.FLOOR_Z + 0.1])
    j = b.add_freejoint()
    j.name = f"{name}_j"
    return b


def _rs_xml(kind):
    import importlib.util
    root = os.path.dirname(importlib.util.find_spec("robosuite").origin)
    return os.path.join(root, f"models/assets/objects/{kind}.xml")


def add_objects(spec):
    """v6's extra pool bodies (v3's blocks are already in the XML)."""
    common = dict(density=sx.DENSITY, rgba=[0.5, 0.5, 0.5, 1], friction=BLOCK_FRICTION)
    start = len(sx.object_names())
    names = cat.object_names()[start:]
    for i, name in enumerate(names):
        kind, size = cat.kind_of(name), cat.size_of(name)
        idx = start + i
        if kind in cat.RS_OBJECTS:
            src = mujoco.MjSpec.from_file(_rs_xml(kind))
            obj = src.body("object")
            for g in obj.geoms:
                g.solref = [0.02, 1.0]
                g.solimp = [0.9, 0.95, 0.001, 0.5, 2.0]
                g.friction = ROLL_FRICTION if kind in cat.ROLLS else BLOCK_FRICTION
                g.condim = 6 if kind in cat.ROLLS else 3
                if g.contype == 0:
                    g.density = 0.0
                else:
                    g.mass = cat.RS_OBJECTS[kind][1]
            frame = spec.worldbody.add_frame(pos=[PARK_ROW[0] - 0.2 * idx, PARK_ROW[1], sx.FLOOR_Z + 0.1])
            b = frame.attach_body(obj, f"{name}_", "")
            b.name = name
            j = b.add_freejoint()
            j.name = f"{name}_j"
            continue
        b = _free_body(spec, name, idx)
        if kind == "ball":
            b.add_geom(name=f"{name}_g0", type=G.mjGEOM_SPHERE, size=[cat.BALL_R[size], 0, 0], condim=6,
                       **{**common, "friction": ROLL_FRICTION})
        elif kind == "tube":
            b.add_geom(name=f"{name}_g0", type=G.mjGEOM_CYLINDER, size=[cat.TUBE_R, cat.TUBE_HALF, 0], condim=6,
                       **{**common, "friction": ROLL_FRICTION})
        elif kind == "domino":
            _box(b, cat.DOMINO_HALF, name=f"{name}_g0", **common)
        elif kind == "hammer":
            h = cat.HAMMER
            hand = dict(common, density=0)
            g = _box(b, h["handle_half"], name=f"{name}_g0", **hand)
            g2 = _box(b, h["head_half"], pos=(h["handle_half"][0] + h["head_half"][0], 0, 0), name=f"{name}_g1", **hand)
            g.mass, g2.mass = 0.4 * h["mass"], 0.6 * h["mass"]
            g.rgba, g2.rgba = [0.55, 0.38, 0.22, 1], [0.35, 0.35, 0.38, 1]
        else:
            raise ValueError(kind)


# ----------------------------------------------------------------------------- fixtures
FX_PARK = (3.0, 3.0)
LIMITED = mujoco.mjtLimited.mjLIMITED_TRUE  # MjSpec does not switch limits on from a range by itself (ranges in degrees)


def fixture_names(stage2=True):
    return cat.FIXTURES + (cat.FIXTURES_STAGE2 if stage2 else [])


def _rot_y(a):
    return [np.cos(a / 2), 0, np.sin(a / 2), 0]


def _rot_x(a):
    return [np.cos(a / 2), np.sin(a / 2), 0, 0]


def _rot_z(a):
    return [np.cos(a / 2), 0, 0, np.sin(a / 2)]


def _quat_mul(a, b):
    out = np.empty(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out


def add_fixtures(spec):
    t = 0.004  # wall half thickness
    for i, kind in enumerate(fixture_names()):
        rgba = cat.FIXTURE_RGBA[kind]
        b = spec.worldbody.add_body(name=f"fx_{kind}", mocap=True,
                                    pos=[FX_PARK[0] + 0.6 * i, FX_PARK[1], sx.FLOOR_Z + 0.06])
        kw = dict(rgba=rgba, friction=BLOCK_FRICTION)
        n = lambda k: f"fx_{kind}_g{k}"
        if kind == "tray":
            ix, iy = cat.TRAY_IN
            h = cat.TRAY_WALL_H / 2
            _box(b, (t, iy + 2 * t, h), (-(ix + t), 0, h), name=n(0), **kw)
            _box(b, (ix + t, t, h), (-t, iy + t, h), name=n(1), **kw)
            _box(b, (ix + t, t, h), (-t, -(iy + t), h), name=n(2), **kw)
            _box(b, (ix, iy, 0.0006), (0, 0, 0.0006), name=n(3), contype=0, conaffinity=0,
                 rgba=[rgba[0] * 0.85, rgba[1] * 0.85, rgba[2] * 0.85, 1])  # floor: visual only
        elif kind == "bin":
            ix, iy = cat.BIN_IN
            h = cat.BIN_WALL_H / 2
            _box(b, (ix, iy, 0.003), (0, 0, 0.003), name=n(0), **kw)
            _box(b, (t, iy + 2 * t, h), (-(ix + t), 0, h), name=n(1), **kw)
            _box(b, (t, iy + 2 * t, h), (ix + t, 0, h), name=n(2), **kw)
            _box(b, (ix, t, h), (0, iy + t, h), name=n(3), **kw)
            _box(b, (ix, t, h), (0, -(iy + t), h), name=n(4), **kw)
        elif kind == "bowl":
            b.add_geom(name=n(0), type=G.mjGEOM_CYLINDER, size=[cat.BOWL_FLOOR_R + 0.004, 0.002, 0], pos=[0, 0, 0.002],
                       **kw)
            run = cat.BOWL_R - cat.BOWL_FLOOR_R
            slant = np.hypot(run, cat.BOWL_H)
            tilt = np.arctan2(run, cat.BOWL_H)  # from vertical
            r_mid = (cat.BOWL_R + cat.BOWL_FLOOR_R) / 2
            nseg = 16
            half_len = np.pi * r_mid / nseg * 1.15
            for k in range(nseg):
                a = 2 * np.pi * k / nseg
                # segment frame: x radial (outward), y tangent, z along the slant; tilted outward about y
                q = _quat_mul(_rot_z(a), _rot_y(tilt))
                _box(b, (0.003, half_len, slant / 2), (r_mid * np.cos(a), r_mid * np.sin(a), cat.BOWL_H / 2), q,
                     name=n(k + 1), **kw)
        elif kind == "platform":
            p = cat.PLATFORM_HALF
            _box(b, p, (0, 0, p[2]), name=n(0), **kw)
        elif kind == "ramp":
            a = np.arctan2(cat.RAMP_H, cat.RAMP_LEN)
            half_l = cat.RAMP_LEN / (2 * np.cos(a))
            th = 0.004
            # top surface runs from (-L/2, 0) up to (L/2, H); the plank sits just below it
            cz = cat.RAMP_H / 2 - th / np.cos(a)
            _box(b, (half_l, cat.RAMP_W / 2, th), (0, 0, cz), _rot_y(-a), name=n(0), **kw)
            _box(b, (0.004, cat.RAMP_W / 2, cat.RAMP_H / 2 - 0.004), (cat.RAMP_LEN / 2 - 0.006, 0, cat.RAMP_H / 2 - 0.004),
                 name=n(1), **kw)
            for k, sgn in enumerate((1, -1)):  # low side rails keep rolling things on the incline
                _box(b, (half_l, 0.003, 0.006), (0, sgn * (cat.RAMP_W / 2 + 0.003), cz + 0.008), _rot_y(-a),
                     name=n(2 + k), **kw)
        elif kind == "wall":
            w = cat.WALL_HALF
            _box(b, w, (0, 0, w[2]), name=n(0), **kw)
        elif kind == "seesaw":
            ph = cat.SEESAW_PIVOT_H
            _box(b, (0.012, cat.SEESAW_PLANK[1] + 0.006, ph / 2), (0, 0, ph / 2), name=n(0),
                 rgba=[0.3, 0.3, 0.32, 1], friction=BLOCK_FRICTION)
            p = b.add_body(name="fx_seesaw_plank", pos=[0, 0, ph + cat.SEESAW_PLANK[2]])
            # The joint stop catches the plank just before its end reaches the table: a thin box edge grinding
            # on the table under a load blew up the solver now and then (the plank spun at 1000 rad/s).
            lim = np.degrees(np.arcsin((ph - 0.004) / cat.SEESAW_PLANK[0]))
            p.add_joint(name="fx_seesaw_plank_j", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0],
                        range=[-lim, lim], limited=LIMITED, damping=0.002, armature=0.001)
            _box(p, cat.SEESAW_PLANK, name="fx_seesaw_plank_g0", density=600, **kw)
        elif kind == "cabinet":
            _cabinet(b, kw)
        elif kind == "lidbox":
            _lidbox(b, kw)
    # A moving part hangs off a mocap body, which MuJoCo welds to the world, so the usual parent-child contact
    # filter does not apply: without these exclusions the plank lies locked on its pivot's top face (and then
    # blows up), and the drawer and lid grind on their frames. The joint ranges stop them instead.
    for base, part in (("fx_seesaw", "fx_seesaw_plank"), ("fx_cabinet", "fx_cabinet_drawer"), ("fx_lidbox", "fx_lidbox_lid")):
        if any(bb.name == part for bb in spec.bodies):
            spec.add_exclude(bodyname1=base, bodyname2=part)


def _cabinet(b, kw):
    """A cabinet whose drawer slides out along +x (toward the camera side) by up to DRAWER_TRAVEL."""
    hx, hy, hz = cat.CAB_HALF
    t = 0.006
    n = lambda k: f"fx_cabinet_g{k}"
    _box(b, (hx, hy, t), (0, 0, 2 * hz - t), name=n(0), **kw)  # top
    _box(b, (hx, t, hz), (0, hy - t, hz), name=n(1), **kw)  # sides
    _box(b, (hx, t, hz), (0, -(hy - t), hz), name=n(2), **kw)
    _box(b, (t, hy, hz), (-(hx - t), 0, hz), name=n(3), **kw)  # back
    _box(b, (hx, hy, t), (0, 0, t), name=n(4), **kw)  # bottom
    ix, iy, iz = cat.DRAWER_IN
    d = b.add_body(name="fx_cabinet_drawer", pos=[hx - ix - 0.008, 0, 2 * t + 0.002])
    d.add_joint(name="fx_cabinet_drawer_j", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0],
                range=[0, cat.DRAWER_TRAVEL], limited=LIMITED, damping=2.0, frictionloss=0.05)
    dk = dict(kw, rgba=[0.75, 0.62, 0.45, 1], density=300)
    w = 0.004
    m = lambda k: f"fx_cabinet_drawer_g{k}"
    _box(d, (ix, iy, w), (0, 0, w), name=m(0), **dk)
    _box(d, (w, iy, iz), (ix + w, 0, iz + w), name=m(1), **dk)  # front
    _box(d, (w, iy, iz), (-(ix + w), 0, iz + w), name=m(2), **dk)
    _box(d, (ix, w, iz), (0, iy + w, iz + w), name=m(3), **dk)
    _box(d, (ix, w, iz), (0, -(iy + w), iz + w), name=m(4), **dk)
    # handle: a bar held HANDLE_OUT in front of the drawer on two posts; the closed fingers come down between the
    # bar and the drawer front and pull (the gap is wide enough that the palm stays clear of the cabinet)
    hk = dict(rgba=[0.2, 0.2, 0.22, 1], density=300, friction=BLOCK_FRICTION)
    hx_bar = ix + 2 * w + cat.HANDLE_OUT
    hz = iz + w
    _box(d, (0.006, cat.HANDLE_HALF_LEN, 0.008), (hx_bar, 0, hz), name=m(5), **hk)
    for k, sgn in enumerate((1, -1)):
        _box(d, (cat.HANDLE_OUT / 2, 0.004, 0.008), (ix + 2 * w + cat.HANDLE_OUT / 2, sgn * (cat.HANDLE_HALF_LEN - 0.004), hz),
             name=m(6 + k), **hk)


def _lidbox(b, kw):
    """An open-topped box with a lid hinged along its back (-x) edge; the lid's knob sits at the front."""
    ix, iy, iz = cat.LIDBOX_IN
    t = 0.005
    n = lambda k: f"fx_lidbox_g{k}"
    _box(b, (ix, iy, t), (0, 0, t), name=n(0), **kw)
    _box(b, (t, iy + 2 * t, iz), (ix + t, 0, iz + 2 * t), name=n(1), **kw)
    _box(b, (t, iy + 2 * t, iz), (-(ix + t), 0, iz + 2 * t), name=n(2), **kw)
    _box(b, (ix, t, iz), (0, iy + t, iz + 2 * t), name=n(3), **kw)
    _box(b, (ix, t, iz), (0, -(iy + t), iz + 2 * t), name=n(4), **kw)
    top = 2 * iz + 2 * t
    lid = b.add_body(name="fx_lidbox_lid", pos=[-(ix + 2 * t), 0, top])
    lid.add_joint(name="fx_lidbox_lid_j", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, -1, 0],
                  range=[0, 110], limited=LIMITED, damping=0.01)
    lk = dict(kw, rgba=[0.25, 0.45, 0.50, 1], density=300)
    _box(lid, (ix + 2 * t, iy + 2 * t, 0.003), (ix + 2 * t, 0, 0.003), name="fx_lidbox_lid_g0", **lk)
    _box(lid, (0.008, 0.02, 0.008), (2 * ix + 4 * t - 0.01, 0, 0.014), name="fx_lidbox_lid_g1",
         rgba=[0.2, 0.2, 0.22, 1], density=300, friction=BLOCK_FRICTION)
