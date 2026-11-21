"""v5 runtime: v3's scene with a simulated Panda that moves v3's V pusher.

Same interface as gen_v3.env.SceneEnv where gen_v4 uses it (set_scene, get_state / set_state,
object_poses, distractor_poses, object_speeds, contacts, render), plus arm control:

    env.step(target_fn)  one 20 Hz control step; target_fn(k) -> (tool position, tool yaw) for physics
                         substep k. Each substep: one damped-least-squares IK step on the joint targets,
                         then the joint servos track them.
    env.ee_state()       7-D end-effector state (x, y, z, roll, pitch, yaw, gripper), V-JEPA 2-AC style.

The joint state is part of get_state(), so restoring a scene restores the arm too.
"""
import mujoco
import numpy as np

from gen_v3 import env as v3env
from gen_v3 import scene_xml as sx
from gen_v5 import scene_xml as vx

CONTROL_HZ = v3env.CONTROL_HZ
SUBSTEPS = v3env.SUBSTEPS
SETTLE_BEFORE = v3env.SETTLE_BEFORE
STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION | mujoco.mjtState.mjSTATE_CTRL
DOWN = np.diag([1.0, -1.0, -1.0])  # tool frame with yaw 0: x along +x, z down -> flange points down


def tool_rot(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ np.diag([1.0, 1.0, 1.0])


class PandaSceneEnv(v3env.SceneEnv):
    def __init__(self):
        self.arm = False
        self.model = vx.build_model()
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
        self.joints = [m.joint(f"panda_joint{k}").id for k in range(1, 8)]
        self.jq = np.array([m.jnt_qposadr[j] for j in self.joints])
        self.jv = np.array([m.jnt_dofadr[j] for j in self.joints])
        self.jrange = m.jnt_range[self.joints]
        self.tool_site = m.site("tool").id
        tool_body = m.body("pusher").id
        self.pusher_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == tool_body}
        self.arm_geoms = {g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith("panda_")}
        self.table_geom = m.geom("table").id
        self.floor_geom = m.geom("floor").id
        self.defaults = {"table": m.geom_rgba[self.table_geom].copy(), "floor": m.geom_rgba[self.floor_geom].copy(),
                         "light_diffuse": m.light_diffuse.copy(), "light_dir": m.light_dir.copy()}
        self.active, self.dactive, self._renderers = [], [], {}
        self.lead = np.array(vx.KV, float) / np.array(vx.KP, float)
        self.home = self._solve_home()
        self.q_des = self.home["q"].copy()
        self.park = self.home["pos"]

    # ------------------------------------------------------------------ arm kinematics
    def _jac(self):
        jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jp, jr, self.tool_site)
        return np.vstack([jp, jr])[:, self.jv]

    def _pose_err(self, pos, rot):
        d = self.data
        ep = pos - d.site_xpos[self.tool_site]
        # Rotation error as an angular velocity (quaternion log map): well defined up to 180 degrees,
        # unlike the cross-product form, which vanishes for a half turn and stalls the IK.
        qc, qt, er = np.empty(4), np.empty(4), np.empty(3)
        mujoco.mju_mat2Quat(qc, d.site_xmat[self.tool_site])
        mujoco.mju_mat2Quat(qt, np.ascontiguousarray(rot).ravel())
        mujoco.mju_subQuat(er, qt, qc)
        er = d.site_xmat[self.tool_site].reshape(3, 3) @ er  # local -> world frame
        return np.r_[ep, er]

    def ik_step(self, q, pos, rot, gain=1.0, damping=0.02):
        """One damped-least-squares step for joint targets q toward the tool pose (pos, rot). Uses a
        scratch copy of the kinematics so the simulated state is untouched."""
        d = self.data
        saved = d.qpos[self.jq].copy()
        d.qpos[self.jq] = q
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_comPos(self.model, d)
        e = self._pose_err(pos, rot)
        J = self._jac()
        dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), gain * e)
        # Null-space motion (elbow swivel) away from joint limits: a 7-joint arm can turn the tool to any
        # yaw this way, which the last joint alone (+-166 deg) cannot. Weak pull to the posture as well.
        N = np.eye(7) - np.linalg.pinv(J) @ J
        lo, hi = self.jrange[:, 0], self.jrange[:, 1]
        mid, half = (lo + hi) / 2, (hi - lo) / 2
        away = -((q - mid) / half) ** 3 / half  # grows sharply near either limit
        dq += N @ (0.3 * away + 0.01 * (vx.HOME_Q - q))
        q = np.clip(q + dq, self.jrange[:, 0] + 0.02, self.jrange[:, 1] - 0.02)
        d.qpos[self.jq] = saved
        mujoco.mj_kinematics(self.model, d)
        return q, float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]))

    def solve_ik(self, pos, rot, q0, iters=200, tol=1e-4):
        q = np.asarray(q0, float).copy()
        for _ in range(iters):
            q, ep, er = self.ik_step(q, pos, rot)
            if ep < tol and er < 1e-3:
                break
        return q, ep, er

    def ik_seeds(self):
        seeds = [vx.HOME_Q.copy()]
        for q1, q7 in ((-1.0, -2.4), (0.0, -2.4), (1.0, -2.4), (-1.0, 0.0), (1.0, 0.0), (0.0, 2.4), (1.0, 2.4), (-1.0, 2.4)):
            q = vx.HOME_Q.copy()
            q[0], q[6] = q1, q7
            seeds.append(q)
        return seeds

    def solve_ik_multi(self, pos, rot, iters=150, pos_tol=0.003, rot_tol=0.02):
        """IK from several start postures (different elbow / wrist branches). Returns (q, ok)."""
        best = None
        for q0 in self.ik_seeds():
            q, ep, er = self.solve_ik(pos, rot, q0, iters=iters)
            if ep < pos_tol and er < rot_tol:
                return q, True
            if best is None or ep + 0.1 * er < best[1]:
                best = (q, ep + 0.1 * er)
        return best[0], False

    def step_joint(self, q_fn):
        """One control step following a joint-space path: q_fn(k) -> joint targets for substep k."""
        d, dt = self.data, self.model.opt.timestep
        for k in range(SUBSTEPS):
            q_new = q_fn(k)
            dq = (q_new - self.q_des) / dt
            self.q_des = q_new
            d.ctrl[:] = q_new + self.lead * dq
            mujoco.mj_step(self.model, d)

    def _solve_home(self):
        d = self.data
        d.qpos[self.jq] = vx.HOME_Q
        mujoco.mj_forward(self.model, d)
        pos = d.site_xpos[self.tool_site].copy()
        return {"q": vx.HOME_Q.copy(), "pos": pos, "rot": d.site_xmat[self.tool_site].reshape(3, 3).copy()}

    # ------------------------------------------------------------------ scene
    def set_scene(self, spec):
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
        d.qpos[self.jq] = self.home["q"]
        d.ctrl[:] = self.home["q"]
        self.q_des = self.home["q"].copy()
        self.active = [o["pool_name"] for o in spec["objects"]]
        self.dactive = [o["pool_name"] for o in spec.get("distractors", [])]
        mujoco.mj_forward(m, d)
        for _ in range(SETTLE_BEFORE):
            self.step()

    # ------------------------------------------------------------------ stepping
    def step(self, target_fn=None):
        """One control step. target_fn(k) -> (tool position, tool rotation) for substep k, or None to hold.

        IK updates the joint targets q_des each substep; the servo command leads them by (KV / KP) * dq_des,
        which cancels the servo's velocity-damping lag (otherwise the tool trails a moving target by ~1 cm)."""
        d, dt = self.data, self.model.opt.timestep
        for k in range(SUBSTEPS):
            if target_fn is not None:
                pos, rot = target_fn(k)
                q_new, _, _ = self.ik_step(self.q_des, pos, rot)
                dq = (q_new - self.q_des) / dt
                self.q_des = q_new
                d.ctrl[:] = q_new + self.lead * dq
            else:
                d.ctrl[:] = self.q_des
            mujoco.mj_step(self.model, d)

    def pusher_pos(self):
        return np.array(self.data.site_xpos[self.tool_site])

    def tool_rot(self):
        return self.data.site_xmat[self.tool_site].reshape(3, 3).copy()

    def ee_state(self):
        """(x, y, z, roll, pitch, yaw, gripper) of the tool, gripper fixed at 0 (a rigid pusher)."""
        R = self.tool_rot()
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arcsin(-np.clip(R[2, 0], -1, 1))
        yaw = np.arctan2(R[1, 0], R[0, 0])
        return np.r_[self.pusher_pos(), roll, pitch, yaw, 0.0]

    def contacts(self):
        """(tool touches object i, deepest penetration); arm-link contacts with objects count as touches too."""
        touch = np.zeros(len(self.active), bool)
        idx = {self.body_id[n]: i for i, n in enumerate(self.active)}
        robot = self.pusher_geoms | self.arm_geoms
        deepest = 0.0
        for c in self.data.contact[:self.data.ncon]:
            deepest = min(deepest, c.dist)
            if c.geom1 in robot or c.geom2 in robot:
                other = c.geom2 if c.geom1 in robot else c.geom1
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
        self.q_des = self.data.ctrl[:7].copy()  # states are saved with the arm holding still

    def pose_arm(self):
        pass

    def render(self, camera="cam0", size=256):
        if size not in self._renderers:
            self._renderers[size] = mujoco.Renderer(self.model, height=size, width=size)
        r = self._renderers[size]
        r.update_scene(self.data, camera=camera)
        return r.render().copy()
