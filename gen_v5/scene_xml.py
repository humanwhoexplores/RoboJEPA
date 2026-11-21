"""MJCF for v5: v3's tabletop (objects, zones, distractors, cameras, lights) with a simulated Franka Panda
holding v3's V pusher, instead of v3's floating (mocap) pusher.

The arm is robosuite's Panda model (meshes, inertias, joint limits), attached with MjSpec. The V pusher is a
rigid tool on the Panda flange: same plates, same size as v3, so gen_v3.candidates / gen_v4.sampler
(which place pushes from the V's geometry) stay valid. Arm and tool collide with everything, as a real
robot would. World frame as in v3: table top at z = 0, cameras at +x looking back toward the robot at -x.
"""
import os
import re

import mujoco
import numpy as np

from gen_v3 import scene_xml as sx

BASE = np.array([-0.52, 0.0, 0.0])  # Panda base on a pedestal behind the table (table edge at x = -0.35)
TOOL_LEN = 0.12  # flange to pusher center, along the flange axis
TOOL_YAW = 0.0  # tool mounting angle about the flange axis (rad): where the wrist's twist range sits
SHAFT_RADIUS = 0.012
HOME_Q = np.array([0.0, -0.6, 0.0, -2.4, 0.0, 1.85, 0.785])  # tool up and back, out of the work area


# Joint position servos, stiff like an industrial position-controlled arm. robosuite's Panda has zero
# armature (rotor inertia); a realistic one keeps the stiff servos stable at the 2 ms physics step.
KP = [4000, 4000, 4000, 3000, 1500, 1000, 600]
KV = [200, 200, 200, 150, 60, 40, 25]
ARMATURE = [0.3, 0.3, 0.3, 0.3, 0.1, 0.1, 0.1]


def panda_xml():
    import importlib.util
    root = os.path.dirname(importlib.util.find_spec("robosuite").origin)
    return os.path.join(root, "models/assets/robots/panda/robot.xml")


def build_model():
    """Compile the v5 scene. Returns an MjModel."""
    xml = sx.build_xml(arm=False)
    xml = re.sub(r'<body name="pusher".*?</body>', "", xml, flags=re.S)  # v3's floating pusher
    spec = mujoco.MjSpec.from_string(xml)
    spec.option.timestep = sx.TIMESTEP

    arm = mujoco.MjSpec.from_file(panda_xml())
    # v3's MJCF uses degrees and the merged model compiles with the parent's setting; robosuite's Panda
    # gives joint ranges in radians, so convert them (the servo ctrlrange below stays in radians).
    rad_range = {j.name: np.array(j.range) for j in arm.joints}
    for j in arm.joints:
        j.range = np.degrees(rad_range[j.name])
    for act in arm.actuators:  # turn robosuite's torque motors into joint position servos
        k = int(act.target[-1]) - 1
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED  # position servo: force = KP (ctrl - q) - KV qdot
        act.gainprm[0] = KP[k]
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        act.biasprm[:3] = [0.0, -KP[k], -KV[k]]
        arm.joint(act.target).armature = ARMATURE[k]
        act.ctrlrange = rad_range[act.target]
        act.forcerange = [-87.0, 87.0] if act.target in ("joint1", "joint2", "joint3", "joint4") else [-12.0, 12.0]
        act.forcelimited = True
    for b in arm.bodies:  # gravity compensation, so the servos only track
        b.gravcomp = 1.0
    frame = spec.worldbody.add_frame(pos=BASE)
    frame.attach_body(arm.body("base"), "panda_", "")
    # Pedestal under the base, down to the floor (visual only).
    spec.worldbody.add_geom(name="pedestal", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                            size=[0.09, (0 - sx.FLOOR_Z) / 2, 0], pos=[BASE[0], BASE[1], sx.FLOOR_Z / 2],
                            rgba=[0.2, 0.2, 0.22, 1], contype=0, conaffinity=0)

    hand = spec.body("panda_right_hand")
    # Tool frame: x = push direction (the V opens toward +x), z = world up when the flange points down.
    # quat: half turn about x (tool z up when the flange points down), then the mounting angle about z.
    qz = np.array([np.cos(TOOL_YAW / 2), 0, 0, np.sin(TOOL_YAW / 2)])
    q = np.empty(4)
    mujoco.mju_mulQuat(q, np.array([0.0, 1.0, 0.0, 0.0]), qz)
    tool = hand.add_body(name="pusher", pos=[0, 0, TOOL_LEN], quat=q)
    tool.add_site(name="tool", pos=[0, 0, 0], size=[0.005, 0, 0])
    shaft_half = (TOOL_LEN - sx.PUSHER_HALF_LEN) / 2
    tool.add_geom(name="shaft", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[SHAFT_RADIUS, shaft_half, 0],
                  pos=[0, 0, sx.PUSHER_HALF_LEN + shaft_half], rgba=[0.3, 0.3, 0.32, 1], density=500)
    a = sx.PUSHER_ANGLE
    for name, sgn in (("pusher_l", 1), ("pusher_r", -1)):
        tool.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[sx.PUSHER_PLATE_HALF, sx.PUSHER_HALF_THICK, sx.PUSHER_HALF_LEN],
                      pos=[0, sgn * sx.PUSHER_PLATE_Y, 0],
                      quat=[np.cos(sgn * a / 2), 0, 0, np.sin(sgn * a / 2)],
                      rgba=[0.35, 0.35, 0.38, 1], friction=[0.8, 0.005, 0.0001], density=500)
    return spec.compile()
