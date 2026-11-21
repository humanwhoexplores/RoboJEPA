"""Runtime wrapper around the v3 MJCF: place a scene, step the pusher, render, save/restore.

    env = SceneEnv(arm=False)
    env.set_scene(spec)          # spec from gen_v3.scene_spec.sample_scene
    s0 = env.get_state()
    ...                          # run a candidate push (gen_v3.primitives)
    env.set_state(s0)            # back to the exact same start for the next candidate

The arm is posed only when rendering (pose_arm), from the pusher's current pose. Its
links never collide, so the simulated physics do not depend on it.
"""
import mujoco
import numpy as np

from gen_v3 import scene_xml as sx

CONTROL_HZ = 20
SUBSTEPS = int(round(1.0 / (CONTROL_HZ * sx.TIMESTEP)))  # 25 physics steps per control step
SETTLE_BEFORE = 40  # control steps to let objects come to rest after placement
STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


def quat_z_to(v):
    """wxyz quaternion rotating +z onto the unit vector v."""
    v = np.asarray(v, float) / np.linalg.norm(v)
    c = v[2]
    if c < -1 + 1e-9:
        return np.array([0.0, 1.0, 0.0, 0.0])
    axis = np.cross([0.0, 0.0, 1.0], v)
    q = np.r_[1 + c, axis]
    return q / np.linalg.norm(q)


class SceneEnv:
    def __init__(self, arm=False):
        self.arm = arm
        self.model = mujoco.MjModel.from_xml_string(sx.build_xml(arm=arm))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.pool = sx.object_names()
        self.dpool = sx.distractor_names()
        self.body_id = {n: m.body(n).id for n in self.pool + self.dpool}
        self.qpos_adr = {n: m.jnt_qposadr[m.joint(f"{n}_j").id] for n in self.pool + self.dpool}
        self.qvel_adr = {n: m.jnt_dofadr[m.joint(f"{n}_j").id] for n in self.pool + self.dpool}
        self.geom_ids = {n: [g for g in range(m.ngeom) if m.geom_bodyid[g] == self.body_id[n]] for n in self.pool + self.dpool}
        self.zone_mocap = [m.body_mocapid[m.body(f"zone_{i}").id] for i in range(sx.N_ZONES)]
        self.zone_geom = [m.geom(f"zone_{i}_g").id for i in range(sx.N_ZONES)]
        self.pusher_mocap = m.body_mocapid[m.body("pusher").id]
        self.pusher_geoms = {m.geom("pusher_l").id, m.geom("pusher_r").id}
        self.arm_mocap = {n: m.body_mocapid[m.body(n).id] for n in sx.ARM_LINKS} if arm else {}
        self.park = sx.ARM_HOME if arm else sx.PARK
        self.table_geom = m.geom("table").id
        self.floor_geom = m.geom("floor").id
        self.defaults = {"table": m.geom_rgba[self.table_geom].copy(), "floor": m.geom_rgba[self.floor_geom].copy(),
                         "light_diffuse": m.light_diffuse.copy(), "light_dir": m.light_dir.copy()}
        self.active = []  # pool names of the scene's objects, in spec order
        self.dactive = []  # pool names of the scene's distractors
        self._renderers = {}

    # ------------------------------------------------------------------ scene
    def _apply_visual(self, vis):
        m = self.model
        m.geom_rgba[self.table_geom] = self.defaults["table"]
        m.geom_rgba[self.floor_geom] = self.defaults["floor"]
        m.light_diffuse[:] = self.defaults["light_diffuse"]
        m.light_dir[:] = self.defaults["light_dir"]
        if not vis:
            return
        m.geom_rgba[self.table_geom, :3] = vis["table_rgb"]
        m.geom_rgba[self.floor_geom, :3] = vis["floor_rgb"]
        m.light_diffuse[0] = [vis["key_light"]] * 3
        m.light_diffuse[1] = [vis["fill_light"]] * 3
        d = np.asarray(vis["key_dir"], float)
        m.light_dir[0] = d / np.linalg.norm(d)

    def set_scene(self, spec):
        """Place spec's objects, distractors and zones, park everything else, and let it settle."""
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        self._apply_visual(spec.get("visual"))
        used = {o["pool_name"]: o for o in spec["objects"]}
        dused = {o["pool_name"]: o for o in spec.get("distractors", [])}
        for i, name in enumerate(self.pool):
            adr = self.qpos_adr[name]
            if name in used:
                o = used[name]
                yaw = o["yaw"]
                d.qpos[adr:adr + 7] = [o["xy"][0], o["xy"][1], sx.HEIGHT_HALF[o["size"]] + 0.001,
                                       np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
                rgba = np.array(sx.COLORS[o["color"]])
                rgba[:3] = np.clip(rgba[:3] + np.asarray(o.get("rgb_jitter", [0, 0, 0])), 0, 1)
                for g in self.geom_ids[name]:
                    m.geom_rgba[g] = rgba
            else:
                d.qpos[adr:adr + 7] = [-3 - 0.2 * i, -3, sx.FLOOR_Z + 0.05, 1, 0, 0, 0]
        for i, name in enumerate(self.dpool):
            adr = self.qpos_adr[name]
            if name in dused:
                o = dused[name]
                h = sx.DISTRACTORS[o["kind"]][1]
                d.qpos[adr:adr + 7] = [o["xy"][0], o["xy"][1], h + 0.001, 1, 0, 0, 0]
                for g in self.geom_ids[name]:
                    m.geom_rgba[g] = sx.DISTRACTOR_COLORS[o["color"]]
            else:
                d.qpos[adr:adr + 7] = [3 + 0.2 * i, -3, sx.FLOOR_Z + 0.05, 1, 0, 0, 0]
        for i in range(sx.N_ZONES):
            mid = self.zone_mocap[i]
            if i < len(spec["zones"]):
                z = spec["zones"][i]
                d.mocap_pos[mid] = [z["xy"][0], z["xy"][1], 0.0005]
                m.geom_rgba[self.zone_geom[i]] = sx.COLORS[z["color"]]
            else:
                d.mocap_pos[mid] = [3 + i, 3, sx.FLOOR_Z]
        d.mocap_pos[self.pusher_mocap] = self.park
        d.mocap_quat[self.pusher_mocap] = [1, 0, 0, 0]
        self.active = [o["pool_name"] for o in spec["objects"]]
        self.dactive = [o["pool_name"] for o in spec.get("distractors", [])]
        mujoco.mj_forward(m, d)
        for _ in range(SETTLE_BEFORE):
            self.step()

    # ------------------------------------------------------------------ stepping
    def step(self, pusher_path=None):
        """One control step. pusher_path(k) -> pusher position for physics substep k (or None to stay)."""
        for k in range(SUBSTEPS):
            if pusher_path is not None:
                self.data.mocap_pos[self.pusher_mocap] = pusher_path(k)
            mujoco.mj_step(self.model, self.data)

    def pusher_pos(self):
        return np.array(self.data.mocap_pos[self.pusher_mocap])

    def teleport_pusher(self, pos, yaw=0.0):
        self.data.mocap_pos[self.pusher_mocap] = pos
        self.data.mocap_quat[self.pusher_mocap] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------ state
    def object_poses(self):
        """(N, 3) positions and (N, 4) wxyz quaternions of the scene's objects."""
        pos = np.array([self.data.xpos[self.body_id[n]] for n in self.active])
        quat = np.array([self.data.xquat[self.body_id[n]] for n in self.active])
        return pos, quat

    def distractor_poses(self):
        """(D, 3) positions of the scene's distractors (D may be 0)."""
        return np.array([self.data.xpos[self.body_id[n]] for n in self.dactive]).reshape(-1, 3)

    def object_speeds(self):
        return np.array([np.linalg.norm(self.data.qvel[self.qvel_adr[n]:self.qvel_adr[n] + 3]) for n in self.active])

    def contacts(self):
        """(pusher touches object i: bool array, deepest penetration over all contacts in m)."""
        touch = np.zeros(len(self.active), bool)
        idx = {self.body_id[n]: i for i, n in enumerate(self.active)}
        deepest = 0.0
        for c in self.data.contact[:self.data.ncon]:
            deepest = min(deepest, c.dist)
            if c.geom1 in self.pusher_geoms or c.geom2 in self.pusher_geoms:
                other = c.geom2 if c.geom1 in self.pusher_geoms else c.geom1
                i = idx.get(self.model.geom_bodyid[other])
                if i is not None:
                    touch[i] = True
        return touch, deepest

    def get_state(self):
        size = mujoco.mj_stateSize(self.model, STATE_SPEC)
        s = np.empty(size)
        mujoco.mj_getState(self.model, self.data, s, STATE_SPEC)
        return s

    def set_state(self, s):
        mujoco.mj_setState(self.model, self.data, s, STATE_SPEC)
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------ arm
    def arm_points(self):
        """Shoulder, elbow, wrist and hand (pusher center) points for the current pusher pose (elbow-up IK)."""
        hand = self.pusher_pos()
        wrist = hand + [0.0, 0.0, sx.ARM_WRIST]
        sh = np.r_[sx.ARM_BASE, sx.ARM_SHOULDER_Z]
        v = wrist - sh
        r = np.linalg.norm(v[:2])
        h = v[:2] / r if r > 1e-9 else np.array([1.0, 0.0])
        dist = np.clip(np.linalg.norm(v), 1e-3, sx.ARM_L1 + sx.ARM_L2 - 1e-4)
        a = np.arctan2(v[2], r)
        b = np.arccos(np.clip((sx.ARM_L1 ** 2 + dist ** 2 - sx.ARM_L2 ** 2) / (2 * sx.ARM_L1 * dist), -1, 1))
        elbow = sh + sx.ARM_L1 * np.r_[np.cos(a + b) * h, np.sin(a + b)]
        # If the wrist is out of reach the forearm points at it and stops short.
        u = wrist - elbow
        wrist_reached = elbow + u / np.linalg.norm(u) * sx.ARM_L2
        return sh, elbow, wrist_reached, hand

    def pose_arm(self):
        """Place the visual arm links to meet the pusher, and update their world poses (kinematics only)."""
        if not self.arm:
            return
        sh, elbow, wrist, hand = self.arm_points()
        d = self.data
        for name, (p, q) in zip(sx.ARM_LINKS, ((sh, elbow), (elbow, wrist), (wrist, hand))):
            mid = self.arm_mocap[name]
            d.mocap_pos[mid] = (p + q) / 2
            d.mocap_quat[mid] = quat_z_to(q - p)
        mujoco.mj_kinematics(self.model, d)

    # ------------------------------------------------------------------ rendering
    def render(self, camera="cam0", size=224):
        self.pose_arm()
        key = size
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.model, height=size, width=size)
        r = self._renderers[key]
        r.update_scene(self.data, camera=camera)
        return r.render().copy()
