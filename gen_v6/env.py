"""v6 runtime: v5's Panda scene with a parallel gripper instead of the V pusher, v6's extra objects and fixtures,
and per-object hidden physical properties.

Same interface as gen_v5.env.PandaSceneEnv (set_scene, get_state / set_state, object_poses, contacts, render,
IK), plus:

    env.set_grip(width)     finger opening target in m (0 = closed, 0.08 = fully open); held until changed
    env.grip_width()        measured opening (distance between the finger joints)
    env.holding()           indices of active objects squeezed between both pads
    env.ee_state()          (x, y, z, roll, pitch, yaw, gripper) with the real opening as the gripper value
    env.contact_summary()   object-object, object-table and object-fixture contacts right now
    env.fixture_state()     joint position of each scene fixture's moving part (seesaw tilt, drawer, lid)
    env.extent(i, quat)     (bottom, top) of active object i below / above its centre in orientation quat

Scene specs (gen_v6.scene_spec) extend v5's: each object may carry
    kind    catalog kind (default: its v3 shape)          pose   "upright" (default) or "lying"
    on      index of an earlier object or a fixture kind it starts on (stacked / inside / on the platform)
    props   hidden properties {mass_scale, friction, bounce} (default: normal)
and spec["fixtures"] lists {kind, xy, yaw[, open]} (open: drawer slide in m or lid angle in rad).
"""
import mujoco
import numpy as np

from gen_v3 import env as v3env
from gen_v3 import scene_xml as sx
from gen_v5.env import PandaSceneEnv
from gen_v6 import catalog as cat
from gen_v6 import scene_xml as gx

SUBSTEPS = v3env.SUBSTEPS
IK_ITERS = 3
N_ARM = gx.N_ARM
MOVING_PART = {"seesaw": "fx_seesaw_plank_j", "cabinet": "fx_cabinet_drawer_j", "lidbox": "fx_lidbox_lid_j"}


def quat_yaw(yaw):
    return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])


def quat_mul(a, b):
    out = np.empty(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out


LYING = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0])  # body z turned onto world -y: lying on its side


def _has_body(m, name):
    try:
        m.body(name)
        return True
    except KeyError:
        return False


def pose_quat(pose, yaw):
    return quat_mul(quat_yaw(yaw), LYING) if pose == "lying" else quat_yaw(yaw)


class GripperSceneEnv(PandaSceneEnv):
    def __init__(self, extra=None):
        self.arm = False
        self.model = gx.build_model(extra)
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.pool = cat.object_names()
        self.dpool = sx.distractor_names()
        self.body_id = {n: m.body(n).id for n in self.pool + self.dpool}
        self.qpos_adr = {n: m.jnt_qposadr[m.joint(f"{n}_j").id] for n in self.pool + self.dpool}
        self.qvel_adr = {n: m.jnt_dofadr[m.joint(f"{n}_j").id] for n in self.pool + self.dpool}
        self.geom_ids = {n: [g for g in range(m.ngeom) if m.geom_bodyid[g] == self.body_id[n]] for n in self.pool + self.dpool}
        self.colourable = {n for n in self.pool if cat.kind_of(n) in cat.COLOURABLE}
        self.zone_mocap = [m.body_mocapid[m.body(f"zone_{i}").id] for i in range(sx.N_ZONES)]
        self.zone_geom = [m.geom(f"zone_{i}_g").id for i in range(sx.N_ZONES)]
        self.joints = [m.joint(f"panda_joint{k}").id for k in range(1, 8)]
        self.jq = np.array([m.jnt_qposadr[j] for j in self.joints])
        self.jv = np.array([m.jnt_dofadr[j] for j in self.joints])
        self.jrange = m.jnt_range[self.joints]
        self.tool_site = m.site("tool").id
        self.finger_q = np.array([m.jnt_qposadr[m.joint(f"gripper0_finger_joint{k}").id] for k in (1, 2)])
        self.pad_geoms = [m.geom("gripper0_finger1_pad_collision").id, m.geom("gripper0_finger2_pad_collision").id]
        self.pusher_geoms = {g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith("gripper0_")}
        self.arm_geoms = {g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith("panda_")}
        self.table_geom = m.geom("table").id
        self.floor_geom = m.geom("floor").id
        self.defaults = {"table": m.geom_rgba[self.table_geom].copy(), "floor": m.geom_rgba[self.floor_geom].copy(),
                         "light_diffuse": m.light_diffuse.copy(), "light_dir": m.light_dir.copy(),
                         "body_mass": m.body_mass.copy(), "body_inertia": m.body_inertia.copy(),
                         "geom_friction": m.geom_friction.copy(), "geom_solref": m.geom_solref.copy(),
                         "geom_solmix": m.geom_solmix.copy(), "geom_contype": m.geom_contype.copy(),
                         "geom_conaffinity": m.geom_conaffinity.copy(), "body_gravcomp": m.body_gravcomp.copy()}
        # fixtures: mocap body, every geom on it or its moving part, the moving part's joint
        self.fixture_kinds = gx.fixture_names()
        self.fx_body = {k: m.body(f"fx_{k}").id for k in self.fixture_kinds}
        self.fx_mocap = {k: m.body_mocapid[b] for k, b in self.fx_body.items()}
        self.fx_park = {k: m.body_pos[b].copy() for k, b in self.fx_body.items()}
        self.fx_geoms = {k: {g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith(f"fx_{k}")}
                         for k in self.fixture_kinds}
        self.fx_joint = {k: (m.jnt_qposadr[m.joint(j).id], m.jnt_dofadr[m.joint(j).id]) for k, j in MOVING_PART.items()}
        self.fx_active = []  # kinds of the scene's fixtures, in spec order
        self.fx_parts = {k: [m.body(b).id for b in (f"fx_{k}_plank", f"fx_{k}_drawer", f"fx_{k}_lid") if _has_body(m, b)]
                         for k in self.fixture_kinds}
        self._box = {n: self._body_box(n) for n in self.pool}
        self.active, self.dactive, self._renderers = [], [], {}
        self.lead = np.array(gx.KV, float) / np.array(gx.KP, float)
        self.grip_cmd = gx.MAX_OPEN
        self.home = self._solve_home()
        self.q_des = self.home["q"].copy()
        self.park = self.home["pos"]

    # ------------------------------------------------------------------ geometry
    def _body_box(self, name):
        """Corners (8k, 3) of the collision geoms' boxes in the body frame."""
        m = self.model
        pts = []
        for g in self.geom_ids[name]:
            if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
                continue
            c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
            R = np.empty(9)
            mujoco.mju_quat2Mat(R, m.geom_quat[g])
            R = R.reshape(3, 3)
            for sgn in np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3):
                pts.append(m.geom_pos[g] + R @ (c + sgn * h))
        return np.array(pts)

    def extent_of(self, name, quat):
        """(below, above): how far a pool body reaches below / above its centre in orientation quat."""
        R = np.empty(9)
        mujoco.mju_quat2Mat(R, np.asarray(quat, float))
        z = (self._box[name] @ R.reshape(3, 3).T)[:, 2]
        return float(-z.min()), float(z.max())

    def extent(self, i, quat=None):
        n = self.active[i]
        return self.extent_of(n, self.data.xquat[self.body_id[n]] if quat is None else quat)

    def half_width(self, name):
        """Largest horizontal reach of a pool body from its centre when upright (footprint radius)."""
        return float(np.linalg.norm(self._box[name][:, :2], axis=1).max())

    # ------------------------------------------------------------------ gripper
    def set_grip(self, width):
        self.grip_cmd = float(np.clip(width, 0.0, gx.MAX_OPEN))
        self.data.ctrl[N_ARM:N_ARM + 2] = [self.grip_cmd / 2, -self.grip_cmd / 2]

    def grip_width(self):
        q = self.data.qpos[self.finger_q]
        return float(q[0] - q[1])

    def pad_contacts(self):
        """set of active-object indices touched by each pad: (left, right)."""
        idx = {self.body_id[n]: i for i, n in enumerate(self.active)}
        out = (set(), set())
        for c in self.data.contact[:self.data.ncon]:
            for k, pad in enumerate(self.pad_geoms):
                if pad in (c.geom1, c.geom2):
                    other = c.geom2 if c.geom1 == pad else c.geom1
                    i = idx.get(self.model.geom_bodyid[other])
                    if i is not None:
                        out[k].add(i)
        return out

    def holding(self):
        """Indices of active objects squeezed between both pads."""
        left, right = self.pad_contacts()
        return sorted(left & right)

    # ------------------------------------------------------------------ scene
    def _reset_physics_props(self):
        m, d = self.model, self.defaults
        m.body_mass[:] = d["body_mass"]
        m.body_inertia[:] = d["body_inertia"]
        m.geom_friction[:] = d["geom_friction"]
        m.geom_solref[:] = d["geom_solref"]
        m.geom_solmix[:] = d["geom_solmix"]

    def _sleep(self, bodies, geoms):
        """Unused things float where they are parked, touching nothing: no contacts to solve each step."""
        m = self.model
        for b in bodies:
            m.body_gravcomp[b] = 1.0
        for g in geoms:
            m.geom_contype[g] = 0
            m.geom_conaffinity[g] = 0

    def _wake_all(self):
        m, d = self.model, self.defaults
        m.geom_contype[:] = d["geom_contype"]
        m.geom_conaffinity[:] = d["geom_conaffinity"]
        m.body_gravcomp[:] = d["body_gravcomp"]

    def _apply_props(self, name, props):
        m = self.model
        b = self.body_id[name]
        s = float(props.get("mass_scale", 1.0))
        m.body_mass[b] = self.defaults["body_mass"][b] * s
        m.body_inertia[b] = self.defaults["body_inertia"][b] * s
        for g in self.geom_ids[name]:
            m.geom_friction[g, 0] = float(props.get("friction", m.geom_friction[g, 0]))
            bounce = float(props.get("bounce", 0.0))
            if bounce > 0:
                m.geom_solref[g] = cat.bounce_solref(bounce)

    def _place_fixtures(self, fixtures):
        d = self.data
        self.fx_active = []
        for k in self.fixture_kinds:
            d.mocap_pos[self.fx_mocap[k]] = self.fx_park[k]
            d.mocap_quat[self.fx_mocap[k]] = [1, 0, 0, 0]
        for k, (qa, va) in self.fx_joint.items():
            d.qpos[qa], d.qvel[va] = 0.0, 0.0
        for f in fixtures:
            k = f["kind"]
            d.mocap_pos[self.fx_mocap[k]] = [f["xy"][0], f["xy"][1], 0.0]
            d.mocap_quat[self.fx_mocap[k]] = quat_yaw(f.get("yaw", 0.0))
            if k in self.fx_joint:
                d.qpos[self.fx_joint[k][0]] = float(f.get("open", 0.0))
            self.fx_active.append(k)

    def fixture_top(self, kind, xy=None):
        """Height of the surface an object rests on when it starts on / in a fixture."""
        return {"platform": 2 * cat.PLATFORM_HALF[2], "bin": 0.006, "bowl": 0.004, "tray": 0.0,
                "cabinet": 0.0, "lidbox": 0.01,
                "seesaw": cat.SEESAW_PIVOT_H + 0.05}.get(kind, 0.0)  # seesaw: dropped from above an end, settles

    def set_scene(self, spec, grip=gx.MAX_OPEN):
        """v5's set_scene with v6's objects (kinds, poses, stacks), fixtures, hidden properties, and the
        9-entry control vector (arm + fingers) with the fingers at `grip`."""
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        self._apply_visual(spec.get("visual"))
        self._reset_physics_props()
        self._wake_all()
        self._place_fixtures(spec.get("fixtures", []))
        mujoco.mj_kinematics(m, d)
        used = {o["pool_name"] for o in spec["objects"]}
        dused = {o["pool_name"]: o for o in spec.get("distractors", [])}
        for i, name in enumerate(self.pool):
            if name not in used:
                adr = self.qpos_adr[name]
                d.qpos[adr:adr + 7] = [gx.PARK_ROW[0] - 0.2 * i, gx.PARK_ROW[1], sx.FLOOR_Z + 0.1, 1, 0, 0, 0]
        tops = []  # top height of each placed object (for stacking later ones on it)
        for o in spec["objects"]:
            name = o["pool_name"]
            q = pose_quat(o.get("pose", "upright"), o["yaw"])
            below, above = self.extent_of(name, q)
            on = o.get("on")
            if on is None:
                base = 0.0
            elif isinstance(on, (int, np.integer)):
                base = tops[on]
            else:
                base = self.fixture_top(on)
            z = base + below + 0.001
            tops.append(z + above)
            adr = self.qpos_adr[name]
            d.qpos[adr:adr + 7] = [o["xy"][0], o["xy"][1], z, *q]
            if name in self.colourable and "color" in o and o["color"] in sx.COLORS:
                rgba = np.array(sx.COLORS[o["color"]])
                rgba[:3] = np.clip(rgba[:3] + np.asarray(o.get("rgb_jitter", [0, 0, 0])), 0, 1)
                for g in self.geom_ids[name]:
                    m.geom_rgba[g] = rgba
            if o.get("props"):
                self._apply_props(name, o["props"])
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
        d.qpos[self.jq] = self.home["q"]
        d.qpos[self.finger_q] = [grip / 2, -grip / 2]
        self.grip_cmd = float(grip)
        self.q_des = self.home["q"].copy()
        self._arm_ctrl(self.q_des)
        self.active = [o["pool_name"] for o in spec["objects"]]
        self.dactive = [o["pool_name"] for o in spec.get("distractors", [])]
        idle = [n for n in self.pool + self.dpool if n not in used and n not in dused]
        idle_fx = [k for k in self.fixture_kinds if k not in self.fx_active]
        self._sleep([self.body_id[n] for n in idle] + [b for k in idle_fx for b in self.fx_parts[k]],
                    [g for n in idle for g in self.geom_ids[n]] + [g for k in idle_fx for g in self.fx_geoms[k]])
        mujoco.mj_forward(m, d)
        for _ in range(v3env.SETTLE_BEFORE):
            self.step()

    # ------------------------------------------------------------------ contacts
    def contact_summary(self):
        """(pair (N, N) object-object, table (N,) object-table, fixture (N, F) object-fixture) contacts now,
        F = the scene's fixtures in spec order."""
        n, f = len(self.active), len(self.fx_active)
        pair, table, fx = np.zeros((n, n), bool), np.zeros(n, bool), np.zeros((n, f), bool)
        idx = {self.body_id[name]: i for i, name in enumerate(self.active)}
        fx_of = {}
        for j, k in enumerate(self.fx_active):
            for g in self.fx_geoms[k]:
                fx_of[g] = j
        bid = self.model.geom_bodyid
        for c in self.data.contact[:self.data.ncon]:
            a, b = idx.get(bid[c.geom1]), idx.get(bid[c.geom2])
            if a is not None and b is not None:
                pair[a, b] = pair[b, a] = True
                continue
            if a is None and b is None:
                continue
            i, other = (a, c.geom2) if a is not None else (b, c.geom1)
            if other == self.table_geom:
                table[i] = True
            elif other in fx_of:
                fx[i, fx_of[other]] = True
        return pair, table, fx

    def fixture_state(self):
        """Joint position of each scene fixture's moving part (NaN for fixtures without one)."""
        return np.array([self.data.qpos[self.fx_joint[k][0]] if k in self.fx_joint else np.nan for k in self.fx_active])

    # ------------------------------------------------------------------ stepping
    def _arm_ctrl(self, q):
        self.data.ctrl[:N_ARM] = q
        self.data.ctrl[N_ARM:N_ARM + 2] = [self.grip_cmd / 2, -self.grip_cmd / 2]

    def step(self, target_fn=None):
        """One 20 Hz control step. With a tool target, IK runs once per control step (IK_ITERS damped least-squares
        iterations toward the step's end pose) and the joint targets move linearly over the physics substeps,
        with v5's velocity lead. (v5 ran one IK iteration per substep: 25 per control step, 8x the cost.)"""
        d, dt = self.data, self.model.opt.timestep
        if target_fn is None:
            for k in range(SUBSTEPS):
                self._arm_ctrl(self.q_des)
                mujoco.mj_step(self.model, d)
            return
        pos, rot = target_fn(SUBSTEPS - 1)
        q0 = self.q_des.copy()
        q1 = q0
        for _ in range(IK_ITERS):
            q1, ep, er = self.ik_step(q1, pos, rot)
        dq = (q1 - q0) / (SUBSTEPS * dt)
        for k in range(SUBSTEPS):
            self.q_des = q0 + (k + 1) / SUBSTEPS * (q1 - q0)
            self._arm_ctrl(self.q_des + self.lead * dq)
            mujoco.mj_step(self.model, d)

    def step_joint(self, q_fn):
        d, dt = self.data, self.model.opt.timestep
        for k in range(SUBSTEPS):
            q_new = q_fn(k)
            dq = (q_new - self.q_des) / dt
            self.q_des = q_new
            self._arm_ctrl(q_new + self.lead * dq)
            mujoco.mj_step(self.model, d)

    def unstable(self):
        """True if MuJoCo hit a numerical blow-up since the last set_state / set_scene (it then resets the
        state on its own, so the rollout is garbage and must be dropped)."""
        return int(self.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number) > 0

    def clear_warnings(self):
        for w in self.data.warning:
            w.number = 0

    def set_state(self, s):
        self.clear_warnings()
        mujoco.mj_setState(self.model, self.data, s, self._spec())
        mujoco.mj_forward(self.model, self.data)
        self.q_des = self.data.ctrl[:N_ARM].copy()
        self.grip_cmd = float(self.data.ctrl[N_ARM] - self.data.ctrl[N_ARM + 1])

    def get_state(self):
        size = mujoco.mj_stateSize(self.model, self._spec())
        s = np.empty(size)
        mujoco.mj_getState(self.model, self.data, s, self._spec())
        return s

    @staticmethod
    def _spec():
        return mujoco.mjtState.mjSTATE_INTEGRATION | mujoco.mjtState.mjSTATE_CTRL

    def ee_state(self):
        e = super().ee_state()
        e[6] = self.grip_width()
        return e
