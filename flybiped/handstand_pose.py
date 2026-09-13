"""Find a statically balanced HANDSTAND pose: nose down, standing on the front legs (T1).

Kinematic random search over body pitch and the 8 actuated front-leg joints
(mirrored left/right), followed by a dynamic stability test with the position
actuators holding the pose. The result is saved to ``build/handstand_pose.json``
and used by the environment as the nominal "standing" state.
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from flybiped import model as fm

POSE_JSON = fm.BUILD_DIR / "handstand_pose.json"
LEG_JOINTS = ("coxa_abduct", "coxa_twist", "coxa", "femur_twist", "femur",
              "tibia", "tarsus", "tarsus2")
STANCE_WIDTH = 0.10   # cm, desired lateral distance between the two claws
FLOOR_MARGIN = 0.04   # cm, clearance required for every non-foot geom


class Kinematics:
    """Helper bundling the ids needed to evaluate a candidate handstand pose."""

    def __init__(self, m: mujoco.MjModel):
        self.m, self.d = m, mujoco.MjData(m)
        name = lambda t, n: mujoco.mj_name2id(m, t, n)  # noqa: E731
        self.jq = {}  # joint base name -> (qposadr_left, qposadr_right, range)
        for j in LEG_JOINTS:
            l = name(mujoco.mjtObj.mjOBJ_JOINT, f"{j}_T1_left")
            r = name(mujoco.mjtObj.mjOBJ_JOINT, f"{j}_T1_right")
            self.jq[j] = (m.jnt_qposadr[l], m.jnt_qposadr[r], m.jnt_range[l].copy())
        self.claws = [name(mujoco.mjtObj.mjOBJ_SITE, f"claw_T1_{s}") for s in ("left", "right")]
        self.tarsi = [name(mujoco.mjtObj.mjOBJ_SITE, f"tarsus_T1_{s}") for s in ("left", "right")]
        gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""  # noqa: E731
        foot = [g for g in range(m.ngeom) if "T1" in gname(g)]
        # Collision capsules of the tarsus chain (tarsus..tarsus4 + claw): the "sole".
        self.sole_geoms = np.array([g for g in foot if m.geom_contype[g]
                                    and ("tarsus" in gname(g) or "claw" in gname(g))])
        self.body_geoms = np.array([g for g in range(1, m.ngeom)
                                    if g not in foot and m.geom_contype[g]])
        self.wing_q = [(m.jnt_qposadr[j], float(m.qpos_spring[m.jnt_qposadr[j]]))
                       for j in range(m.njnt) if "wing" in mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
        self.ranges = np.array([self.jq[j][2] for j in LEG_JOINTS])

    def qpos(self, pitch: float, q_leg: np.ndarray, height: float = 0.0) -> np.ndarray:
        qpos = self.m.qpos0.copy()
        qpos[:3] = (0, 0, height)
        qpos[3:7] = (np.cos(pitch / 2), 0, np.sin(pitch / 2), 0)  # nose-down (positive pitch)
        for (l, r, _), q in zip(self.jq.values(), q_leg):
            qpos[l] = qpos[r] = q
        for adr, spring in self.wing_q:  # wings rest at their spring position
            qpos[adr] = spring
        return qpos

    def evaluate(self, pitch: float, q_leg: np.ndarray) -> tuple[float, float]:
        """Return (cost, ground_z) of the pose with the root at the origin."""
        d, m = self.d, self.m
        d.qpos[:] = self.qpos(pitch, q_leg)
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        claws = d.site_xpos[self.claws]
        sole_bottom = d.geom_xpos[self.sole_geoms, 2] - m.geom_size[self.sole_geoms, 0]
        ground = sole_bottom.min()
        com = d.subtree_com[1]
        mid = d.geom_xpos[self.sole_geoms].mean(0)                       # support-polygon centre
        cost = 40.0 * np.sum((com[:2] - mid[:2]) ** 2)                  # CoM over the feet
        cost += 40.0 * (abs(claws[0, 1] - claws[1, 1]) - STANCE_WIDTH) ** 2
        cost += 400.0 * np.sum((sole_bottom - ground) ** 2)              # whole sole on the floor
        cost -= 1.0 * (com[2] - ground)                                  # stand tall
        clearance = d.geom_xpos[self.body_geoms, 2] - m.geom_rbound[self.body_geoms] - ground
        cost += 100.0 * np.sum(np.clip(FLOOR_MARGIN - clearance, 0, None) ** 2) * 100
        centre = self.ranges.mean(1)
        cost += 0.05 * np.sum(((q_leg - centre) / np.ptp(self.ranges, axis=1)) ** 2)
        return float(cost), float(ground)


def search(kin: Kinematics, n_random: int = 30000, n_refine: int = 3000,
           n_keep: int = 12, seed: int = 0) -> list[tuple[float, np.ndarray]]:
    """Random search followed by hill-climbing; returns the ``n_keep`` best poses."""
    rng = np.random.default_rng(seed)
    lo, hi = kin.ranges[:, 0], kin.ranges[:, 1]
    cands = []
    for _ in range(n_random):
        pitch = rng.uniform(np.deg2rad(20), np.deg2rad(85))
        q = rng.uniform(lo, hi)
        cands.append((kin.evaluate(pitch, q)[0], pitch, q))
    cands.sort(key=lambda c: c[0])
    return [_refine(kin, rng, *c, n_refine) for c in cands[:n_keep]]


def _refine(kin, rng, c, pitch, q, n_refine):
    lo, hi = kin.ranges[:, 0], kin.ranges[:, 1]
    scale = 0.15
    for i in range(n_refine):
        p2 = np.clip(pitch + rng.normal(0, scale * 0.5), np.deg2rad(10), np.deg2rad(89))
        q2 = np.clip(q + rng.normal(0, scale, q.shape) * (hi - lo), lo, hi)
        c2, _ = kin.evaluate(p2, q2)
        if c2 < c:
            c, pitch, q = c2, p2, q2
        if i % 500 == 499:
            scale *= 0.6
    return pitch, q


def _hold_ctrl(m: mujoco.MjModel, qpos: np.ndarray) -> np.ndarray:
    """Actuator controls that target the pose joint angles (adhesion fully on)."""
    ctrl = np.zeros(m.nu)
    for i in range(m.nu):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT:
            ctrl[i] = np.clip(qpos[m.jnt_qposadr[m.actuator_trnid[i, 0]]], *m.actuator_ctrlrange[i])
        elif "adhere" in name:
            ctrl[i] = 1.0
    return ctrl


def _reset(m, d, qpos, ctrl) -> None:
    mujoco.mj_resetData(m, d)
    d.qpos[:] = qpos
    d.ctrl[:] = ctrl
    for i in range(m.nu):  # start the actuator filters at their targets
        if m.actuator_actadr[i] >= 0:
            d.act[m.actuator_actadr[i]] = ctrl[i]
    mujoco.mj_forward(m, d)


def calibrate(m: mujoco.MjModel, qpos: np.ndarray, iters: int = 25,
              window: float = 0.02) -> np.ndarray:
    """Gravity-compensate the position actuators.

    The leg actuators are proportional (force = gain * (ctrl - q)), so under
    load the joints sag. Iteratively offset ``ctrl`` by the measured sag over a
    short window so that the pose is held at the intended angles.
    """
    d = mujoco.MjData(m)
    ctrl = _hold_ctrl(m, qpos)
    leg = [i for i in range(m.nu) if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT
           and "T3" in mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)]
    qadr = np.array([m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in leg])
    for _ in range(iters):
        _reset(m, d, qpos, ctrl)
        for _ in range(int(window / m.opt.timestep)):
            mujoco.mj_step(m, d)
        ctrl[leg] = np.clip(ctrl[leg] + (qpos[qadr] - d.qpos[qadr]),
                            m.actuator_ctrlrange[leg, 0], m.actuator_ctrlrange[leg, 1])
    return ctrl


def _lowest_z(m: mujoco.MjModel, d: mujoco.MjData, geoms: np.ndarray) -> np.ndarray:
    """Lowest world-z point of primitive geoms (numpy twin of env._lowest_z)."""
    R = d.geom_xmat[geoms].reshape(-1, 3, 3)
    zrow = np.abs(R[:, 2, :])
    size, typ = m.geom_size[geoms], m.geom_type[geoms]
    drop = np.where(typ == mujoco.mjtGeom.mjGEOM_ELLIPSOID, np.sqrt(np.sum((size * zrow) ** 2, 1)),
           np.where(typ == mujoco.mjtGeom.mjGEOM_BOX, np.sum(size * zrow, 1),
           np.where(np.isin(typ, (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER)),
                    size[:, 0] + zrow[:, 2] * size[:, 1], size[:, 0])))
    return d.geom_xpos[geoms, 2] - drop


def stability(m: mujoco.MjModel, kin: Kinematics, qpos: np.ndarray, ctrl: np.ndarray,
              n_trials: int = 8, seconds: float = 0.6, seed: int = 0) -> tuple[float, np.ndarray]:
    """Mean survival fraction under random yaw/velocity/pitch perturbations.

    A trial "falls" when a non-foot geom touches the floor or the thorax drops
    below half its initial height. Also returns the unperturbed settled qpos.
    """
    rng = np.random.default_rng(seed)
    d = mujoco.MjData(m)
    thorax = m.site("thorax").id
    survived = []
    settled = None
    for t in range(n_trials + 1):
        q = qpos.copy()
        if t > 0:
            yaw = rng.uniform(-np.pi, np.pi)
            pitch = np.deg2rad(rng.normal(0, 2.0))
            qy = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
            qp = np.array([np.cos(pitch / 2), 0, np.sin(pitch / 2), 0])
            out = np.zeros(4)
            mujoco.mju_mulQuat(out, qy, q[3:7]); mujoco.mju_mulQuat(q[3:7], out, qp)
        _reset(m, d, q, ctrl)
        if t > 0:
            d.qvel[:6] = rng.normal(0, 0.3, 6)
        h0 = d.site_xpos[thorax, 2]
        steps = int(seconds / m.opt.timestep)
        alive = steps
        for k in range(steps):
            mujoco.mj_step(m, d)
            if k % 20 == 0 and (_lowest_z(m, d, kin.body_geoms).min() < 0 or d.site_xpos[thorax, 2] < 0.5 * h0):
                alive = k
                break
        if t == 0:
            settled = d.qpos.copy() if alive == steps else qpos.copy()
        else:
            survived.append(alive / steps)
    return float(np.mean(survived)), settled


def stance(m: mujoco.MjModel, seconds: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """Natural six-leg stance: joint angles 0, settled on the floor."""
    qpos = m.qpos0.copy()
    qpos[2] = 0.14
    for adr, spring in Kinematics(m).wing_q:
        qpos[adr] = spring
    ctrl = _hold_ctrl(m, qpos)
    d = mujoco.MjData(m)
    _reset(m, d, qpos, ctrl)
    for _ in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)
    return d.qpos.copy(), ctrl


def main() -> None:
    m = fm.load()
    kin = Kinematics(m)
    qpos_stance, ctrl_stance = stance(m)
    print(f"six-leg stance: root z={qpos_stance[2]:.3f}")
    best = None
    for pitch, q in search(kin):
        cost, ground = kin.evaluate(pitch, q)
        qpos = kin.qpos(pitch, q, height=-ground + 0.002)
        ctrl = calibrate(m, qpos)
        score, qpos_settled = stability(m, kin, qpos, ctrl)
        print(f"candidate pitch={np.rad2deg(pitch):.1f} cost={cost:.3f} "
              f"root z={qpos[2]:.3f} settled z={qpos_settled[2]:.3f} survival={score:.2f}")
        if best is None or score > best[-1]:
            best = (pitch, q, qpos, qpos_settled, ctrl, score)
    pitch, q, qpos, qpos_settled, ctrl, drift = best
    print(f"best: pitch={np.rad2deg(pitch):.1f} deg survival={drift:.2f}")
    POSE_JSON.write_text(json.dumps({
        "pitch_deg": float(np.rad2deg(pitch)),
        "leg_joints": dict(zip(LEG_JOINTS, q.tolist())),
        "qpos": qpos.tolist(),
        "qpos_settled": qpos_settled.tolist(),
        "ctrl": ctrl.tolist(),
        "qpos_stance": qpos_stance.tolist(),
        "ctrl_stance": ctrl_stance.tolist(),
        "drift": drift,
    }, indent=1))
    print("wrote", POSE_JSON)


if __name__ == "__main__":
    main()
