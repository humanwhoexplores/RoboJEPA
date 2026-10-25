"""Tabletop push task: one Panda, one cube, two colored target regions, fixed camera.

Built on robosuite's Lift task (same robot, table and controller), with:
  * a larger neutral-colored cube (so it is not confused with a red target and
    covers several DINOv2 patches at 224 px),
  * two visual-only target discs on the table, left and right,
  * per-scene control of cube pose and which side is red / blue.

Coordinates are robosuite world coordinates. The robot base sits at -x looking
toward +x. "Left" / "right" are defined as seen in the fixed camera CAMERA,
which looks back at the robot from +x, so image-left is world -y (checked by
rendering).
"""
import numpy as np

from robosuite.environments.manipulation.lift import Lift
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import new_element, new_geom

TABLE_TOP_Z = 0.8
# 4 cm cube. The Panda hand's underside sits only ~3 cm above the fingertips,
# so a taller cube gets pinned by the hand instead of pushed by the fingers.
CUBE_HALF = 0.02
TARGET_RADIUS = 0.06
TARGET_X = 0.0
LEFT_Y = -0.15  # world -y is the left side of the camera image
TARGETS = {
    "left": np.array([TARGET_X, LEFT_Y]),
    "right": np.array([TARGET_X, -LEFT_Y]),
}
COLORS = {
    "red": [0.85, 0.15, 0.15, 1.0],
    "blue": [0.15, 0.30, 0.90, 1.0],
}
CUBE_RGBA = [0.95, 0.80, 0.20, 1.0]  # yellow: neither target color
CAMERA = "pushview"  # fixed camera added below; tighter than robosuite's agentview


class PushEnv(Lift):
    """Lift with a bigger cube and two target discs; reward/success are unused."""

    def _load_model(self):
        # Skip Lift._load_model (it builds its own red cube); call the base class.
        super(Lift, self)._load_model()
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        arena.set_origin([0, 0, 0])

        # Visual-only target discs, a hair above the table surface.
        for side, xy in TARGETS.items():
            arena.worldbody.append(
                new_geom(
                    name=f"target_{side}",
                    type="cylinder",
                    size=[TARGET_RADIUS, 0.001],
                    pos=[xy[0], xy[1], TABLE_TOP_Z + 0.001],
                    rgba=COLORS["red"],
                    group=1,
                    contype=0,
                    conaffinity=0,
                )
            )

        # Fixed camera at +x looking back at the table center, framed on both targets
        # so the 4 cm cube spans more than one 14 px DINOv2 patch at 224 px.
        arena.worldbody.append(
            new_element(
                "camera",
                name=CAMERA,
                pos="0.45 0 1.15",
                xyaxes="0 1 0 -0.614 0 0.789",
                fovy="45",
            )
        )

        self.cube = BoxObject(
            name="cube",
            size_min=[CUBE_HALF] * 3,
            size_max=[CUBE_HALF] * 3,
            rgba=CUBE_RGBA,
            density=500,
            rng=self.rng,
        )
        # Lift's reset uses a placement sampler; we overwrite the pose ourselves in set_scene().
        from robosuite.utils.placement_samplers import UniformRandomSampler

        self.placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=self.cube,
            x_range=[0.0, 0.0],
            y_range=[0.0, 0.0],
            rotation=0.0,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=False,
            reference_pos=self.table_offset,
            z_offset=0.0,
            rng=self.rng,
        )
        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.cube,
        )

    def set_scene(self, cube_xy, cube_yaw, left_color):
        """Place the cube and color the targets. Call right after reset()."""
        quat_wxyz = [np.cos(cube_yaw / 2), 0.0, 0.0, np.sin(cube_yaw / 2)]
        pos = [cube_xy[0], cube_xy[1], TABLE_TOP_Z + CUBE_HALF]
        self.sim.data.set_joint_qpos(self.cube.joints[0], np.concatenate([pos, quat_wxyz]))
        right_color = "blue" if left_color == "red" else "red"
        for side, color in (("left", left_color), ("right", right_color)):
            gid = self.sim.model.geom_name2id(f"target_{side}")
            self.sim.model.geom_rgba[gid] = COLORS[color]
        self.sim.forward()
        return {"left": left_color, "right": right_color}

    # Convenience accessors ------------------------------------------------
    def cube_state(self):
        bid = self.cube_body_id
        return (
            np.array(self.sim.data.body_xpos[bid]),
            np.array(self.sim.data.body_xquat[bid]),
            np.array(self.sim.data.get_joint_qvel(self.cube.joints[0])[:3]),
        )

    def gripper_cube_contact(self):
        """(in_contact, total normal force) between any robot/gripper geom and the cube."""
        import mujoco

        cube_ids = {self.sim.model.geom_name2id(g) for g in self.cube.contact_geoms}
        robot_ids = set()
        for g in self.robots[0].robot_model.contact_geoms + self.robots[0].gripper["right"].contact_geoms:
            try:
                robot_ids.add(self.sim.model.geom_name2id(g))
            except ValueError:
                pass
        force = 0.0
        hit = False
        f6 = np.zeros(6)
        for i in range(self.sim.data.ncon):
            c = self.sim.data.contact[i]
            pair = {c.geom1, c.geom2}
            if pair & cube_ids and pair & robot_ids:
                hit = True
                mujoco.mj_contactForce(self.sim.model._model, self.sim.data._data, i, f6)
                force += abs(f6[0])
        return hit, force

    def camera_image(self, camera=CAMERA, size=224):
        import mujoco

        key = (camera, size)
        if getattr(self, "_renderers", None) is None:
            self._renderers = {}
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.sim.model._model, height=size, width=size)
        r = self._renderers[key]
        r.update_scene(self.sim.data._data, camera=camera)
        return r.render().copy()


def make_env(control_freq=20):
    return PushEnv(
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,  # we render keyframes ourselves
        use_object_obs=True,
        control_freq=control_freq,
        horizon=100000,
        ignore_done=True,
        hard_reset=False,
        initialization_noise=None,  # deterministic start pose
    )
