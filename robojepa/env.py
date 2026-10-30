"""Tabletop push environment: one cube, two colored target regions, scripted push primitives."""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import new_geom

TABLE_TOP = np.array([0.0, 0.0, 0.8])
CUBE_HALF = 0.02
# Regions are squares centered at (REGION_X, +-REGION_Y) relative to the table center.
REGION_X, REGION_Y, REGION_HALF = 0.0, 0.17, 0.065
# Cube spawn range (relative to table center), kept clear of both regions.
SPAWN_X, SPAWN_Y = (-0.08, 0.08), (-0.06, 0.06)
COLORS = {"blue": [0.1, 0.25, 0.95, 1], "green": [0.1, 0.8, 0.2, 1]}

Z_PUSH = TABLE_TOP[2] + 0.025  # gripper fingertip height while pushing
Z_HOVER = TABLE_TOP[2] + 0.12
CAMERA = "pushview"
CAM_POS, CAM_TARGET = [0.42, 0.0, 1.32], [-0.04, 0.0, 0.8]


class PushEnv(ManipulationEnv):
    """robosuite env with a cube and two flat target regions whose colors swap per scene.

    Side naming is in the camera image frame: "left" region is at world -y, "right" at +y.
    """

    def __init__(self, img_size=224, seed=None, **kwargs):
        self.img_size = img_size
        self.colors = ("blue", "green")  # colors of (left, right), set in reset
        super().__init__(
            robots="Panda",
            gripper_types="default",
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=False,
            render_camera=CAMERA,
            control_freq=20,
            horizon=10_000,
            ignore_done=True,
            seed=seed,
            **kwargs,
        )

    # ---------------- model ----------------
    def _load_model(self):
        super()._load_model()
        robot = self.robots[0]
        robot.robot_model.set_base_xpos(robot.robot_model.base_xpos_offset["table"](0.8))

        arena = TableArena(table_full_size=(0.8, 0.8, 0.05), table_friction=(1.0, 5e-3, 1e-4), table_offset=TABLE_TOP)
        arena.set_origin([0, 0, 0])
        # Fixed oblique camera in front of the table, looking back toward the robot.
        arena.set_camera(CAMERA, pos=CAM_POS, quat=T.convert_quat(
            T.mat2quat(_look_at(CAM_POS, CAM_TARGET)), to="wxyz"))
        for side, sy in (("left", -REGION_Y), ("right", REGION_Y)):
            arena.worldbody.append(new_geom(
                name=f"region_{side}", type="box", size=[REGION_HALF, REGION_HALF, 0.0005],
                pos=[TABLE_TOP[0] + REGION_X, sy, TABLE_TOP[2] + 0.0005],
                rgba=COLORS["blue"], contype=0, conaffinity=0, group=1))

        self.cube = BoxObject(name="cube", size=[CUBE_HALF] * 3, rgba=[0.9, 0.1, 0.1, 1],
                              friction=[1.0, 5e-3, 1e-4], density=300)
        self.model = ManipulationTask(mujoco_arena=arena, mujoco_robots=[r.robot_model for r in self.robots],
                                      mujoco_objects=self.cube)

    def _setup_references(self):
        super()._setup_references()
        self.cube_body_id = self.sim.model.body_name2id(self.cube.root_body)
        self.region_geom_ids = {s: self.sim.model.geom_name2id(f"region_{s}") for s in ("left", "right")}

    def reward(self, action=None):
        return 0.0

    # ---------------- scene control ----------------
    def reset_scene(self, rng):
        """Reset with a random cube pose and random color assignment. Returns scene params."""
        self.reset()
        self._ori_goal = self._eef_quat()
        xy = np.array([rng.uniform(*SPAWN_X), rng.uniform(*SPAWN_Y)])
        yaw = rng.uniform(-np.pi / 4, np.pi / 4)
        self.colors = ("blue", "green") if rng.random() < 0.5 else ("green", "blue")
        for side, c in zip(("left", "right"), self.colors):
            self.sim.model.geom_rgba[self.region_geom_ids[side]] = COLORS[c]
        quat = T.convert_quat(T.axisangle2quat(np.array([0, 0, yaw])), to="wxyz")
        self.sim.data.set_joint_qpos(self.cube.joints[0], np.concatenate([[*xy, TABLE_TOP[2] + CUBE_HALF], quat]))
        self.sim.forward()
        # Raise the gripper out of the camera's way and let the cube settle.
        self._move_to(np.array([-0.15, 0.0, Z_HOVER]), max_steps=40)
        self._hold(5)
        return dict(cube_xy=xy, cube_yaw=yaw, colors=self.colors)

    def get_state(self):
        return self.sim.get_state().flatten().copy(), self.colors

    def set_state(self, state):
        flat, colors = state
        self.colors = colors
        for side, c in zip(("left", "right"), colors):
            self.sim.model.geom_rgba[self.region_geom_ids[side]] = COLORS[c]
        self.sim.set_state_from_flattened(flat)
        self.sim.forward()
        for robot in self.robots:  # reset controller goals to the restored pose
            robot.composite_controller.reset()

    def render_image(self):
        img = self.sim.render(camera_name=CAMERA, width=self.img_size, height=self.img_size)
        return np.ascontiguousarray(img[::-1])

    @property
    def eef_pos(self):
        return np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id["right"]])

    @property
    def cube_pos(self):
        return np.array(self.sim.data.body_xpos[self.cube_body_id])

    def region_of(self, xy):
        """Return 'left', 'right' or None for a cube center xy (world)."""
        for side, sy in (("left", -REGION_Y), ("right", REGION_Y)):
            if abs(xy[0] - TABLE_TOP[0] - REGION_X) <= REGION_HALF and abs(xy[1] - sy) <= REGION_HALF:
                return side
        return None

    # ---------------- low-level motion ----------------
    def _step_toward(self, target, max_delta):
        act = np.zeros(self.action_dim)
        delta = target - self.eef_pos
        act[:3] = delta * min(1.0, max_delta / (np.linalg.norm(delta) + 1e-9)) / 0.05  # straight-line motion
        act[3:6] = np.clip(self._ori_err() / 0.5, -1, 1)
        act[-1] = 1.0  # closed gripper acts as a pusher
        self.step(act)

    def _ori_err(self):
        # Hold the home orientation (gripper pointing down) recorded at reset.
        return T.quat2axisangle(T.quat_multiply(self._ori_goal, T.quat_inverse(self._eef_quat())))

    def _eef_quat(self):
        return T.mat2quat(self.sim.data.site_xmat[self.robots[0].eef_site_id["right"]].reshape(3, 3))

    def _move_to(self, target, max_steps=60, max_delta=0.03, tol=0.006):
        for _ in range(max_steps):
            if np.linalg.norm(target - self.eef_pos) < tol:
                return True
            self._step_toward(target, max_delta)
        return False

    def _hold(self, n):
        target = self.eef_pos.copy()
        for _ in range(n):
            self._step_toward(target, 0.03)

    # ---------------- push primitive ----------------
    def execute_push(self, push, record=False):
        """Execute one push = (sx, sy, ex, ey) in table-centered coords.

        Hover above start, descend, slide to end, lift. Returns cube xy trace if record.
        """
        s = TABLE_TOP[:2] + np.asarray(push[:2])
        e = TABLE_TOP[:2] + np.asarray(push[2:4])
        trace = []
        self._move_to(np.array([*s, Z_HOVER]), max_steps=40)
        self._move_to(np.array([*s, Z_PUSH]), max_steps=25, max_delta=0.02)
        goal = np.array([*e, Z_PUSH])
        for _ in range(80):
            if np.linalg.norm(goal - self.eef_pos) < 0.006:
                break
            self._step_toward(goal, 0.025)
            if record:
                trace.append(self.cube_pos[:2].copy())
        self._move_to(np.array([*e, Z_HOVER]), max_steps=15, max_delta=0.03)
        return trace

    def execute_sequence(self, seq, settle_steps=15):
        for push in seq:
            self.execute_push(push)
        return self.settle(settle_steps)

    def settle(self, settle_steps=15):
        """Retract the arm; return the region the cube stayed in for the whole window (or None)."""
        regions = set()
        target = np.array([-0.15, 0.0, Z_HOVER])
        for _ in range(settle_steps):
            self._step_toward(target, 0.04)
            regions.add(self.region_of(self.cube_pos[:2]))
        return regions.pop() if len(regions) == 1 else None

def _look_at(eye, target, up=(0, 0, 1)):
    """MuJoCo camera rotation matrix (camera looks along -z, y is up)."""
    eye, target, up = map(np.asarray, (eye, target, up))
    f = target - eye
    f = f / np.linalg.norm(f)
    r = np.cross(f, up)
    r = r / np.linalg.norm(r)
    u = np.cross(r, f)
    return np.stack([r, u, -f], axis=1)
