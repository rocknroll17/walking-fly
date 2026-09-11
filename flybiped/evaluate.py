"""Evaluate a trained policy on CPU MuJoCo: metrics + a rendered video.

Usage: python -m flybiped.evaluate --run runs/v1 [--episodes 20] [--video web/assets/demo.mp4]
Rolls out the deterministic policy with the exact observation/action code of
the JAX environment (re-implemented in numpy here so no GPU is needed).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio
import mujoco
import numpy as np

from flybiped import compat  # noqa: F401
from flybiped import model as fm
from flybiped.policy import NumpyPolicy, export_numpy, load_params

GEOM = mujoco.mjtGeom


_MODEL_CACHE: dict[str, mujoco.MjModel] = {}


class CpuEnv:
    """Numpy twin of flybiped.env.FlyBiped (single instance, for evaluation)."""

    def __init__(self, consts: dict, seed: int = 0, assist: float | None = None):
        self.E = consts
        self.assist = consts.get("assist", 0.0) if assist is None else assist
        if "model" not in _MODEL_CACHE:
            _MODEL_CACHE["model"] = fm.load()
        self.m = _MODEL_CACHE["model"]
        self.d = mujoco.MjData(self.m)
        self.rng = np.random.default_rng(seed)
        E = consts
        self.hind, self.fore, self.body = (np.array(E[k]) for k in ("hind_geoms", "fore_geoms", "body_geoms"))
        self.last_action = np.zeros(E["nu"], np.float32)
        self.goal = np.zeros(3)
        self.reached = 0
        self.hind_left = np.array([g for g in self.hind if "T3_left" in (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or "")])
        self.hind_right = np.array([g for g in self.hind if "T3_right" in (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or "")])
        self.gait = {"flight": 0, "walk_steps": 0, "td_alt": 0, "td_same": 0, "td_both": 0}
        self._last_fc = np.ones(2); self._last_td = -1

    # geometry helpers -----------------------------------------------------
    def lowest_z(self, geoms):
        m, d = self.m, self.d
        R = d.geom_xmat[geoms].reshape(-1, 3, 3)
        zrow = np.abs(R[:, 2, :]); size = m.geom_size[geoms]; typ = m.geom_type[geoms]
        drop = np.where(np.isin(typ, (GEOM.mjGEOM_CAPSULE, GEOM.mjGEOM_CYLINDER)), size[:, 0] + zrow[:, 2] * size[:, 1],
               np.where(typ == GEOM.mjGEOM_ELLIPSOID, np.sqrt(np.sum((size * zrow) ** 2, 1)),
               np.where(typ == GEOM.mjGEOM_BOX, np.sum(size * zrow, 1), size[:, 0])))
        return d.geom_xpos[geoms, 2] - drop

    def thorax(self):
        return self.d.site_xpos[self.E["thorax_site"]]

    def bipedal(self):
        E = self.E
        return (self.lowest_z(self.fore).min() > E["clearance"] and self.lowest_z(self.hind).min() < 0.003
                and self.lowest_z(self.body).min() >= 0 and self.thorax()[2] > E["biped_min_height"])

    def fell(self):
        return self.lowest_z(self.body).min() < 0

    def heading(self):
        q = self.d.xquat[self.E["thorax_body"]]
        fwd = np.zeros(3); mujoco.mju_rotVecQuat(fwd, np.array([1.0, 0, 0]), q)
        yaw = np.arctan2(fwd[1], fwd[0]); c, s = np.cos(yaw), np.sin(yaw)
        return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1.0]]), q

    # env API --------------------------------------------------------------
    def sample_goal(self):
        g = self.E["goal"]; p = self.thorax()
        ang = self.rng.uniform(-np.pi, np.pi); r = self.rng.uniform(g["dist_min"], g["dist_max"])
        self.goal = np.array([p[0] + r * np.cos(ang), p[1] + r * np.sin(ang), g["height"]])
        self.d.mocap_pos[0] = self.goal

    def reset(self, biped: bool = False, drop: bool = False, flip: bool = False):
        m, d, E = self.m, self.d, self.E
        mujoco.mj_resetData(m, d)
        q = np.array(E["q_biped"] if biped else E["q_stance"]); c = np.array(E["ctrl_biped"] if biped else E["ctrl_stance"])
        yaw = self.rng.uniform(-np.pi, np.pi); qy = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        mujoco.mju_mulQuat(q[3:7], qy, q[3:7].copy())
        if drop:   # random orientation from a height, like the training drop starts
            q[2] = 0.5; r = self.rng.normal(size=4); q[3:7] = r / np.linalg.norm(r)
        if flip:   # on its back
            q[2] = 0.25; qx = np.array([0.0, 1.0, 0.0, 0.0]); out = np.zeros(4); mujoco.mju_mulQuat(out, q[3:7].copy(), qx); q[3:7] = out
        d.qpos[:] = q; d.ctrl[:] = c
        for i, a in enumerate(E["act_adr"]):
            if a >= 0: d.act[a] = c[i]
        d.xfrc_applied[E["thorax_body"], 2] = self.assist * E.get("weight", 0.0)   # training harness, if any
        mujoco.mj_forward(m, d)
        if drop or flip:   # like training: let the fly land and settle before the policy takes over
            for _ in range(int(E.get("settle_time", 0.25) / m.opt.timestep)):
                mujoco.mj_step(m, d)
            d.time = 0.0
        self.last_action[:] = 0; self.reached = 0
        self.gait = {"flight": 0, "walk_steps": 0, "td_alt": 0, "td_same": 0, "td_both": 0}
        self._last_fc = np.ones(2); self._last_td = -1
        self.sample_goal()
        return self.observe()

    def see_goal(self):
        d = self.d; E = self.E
        hp = d.site_xpos[E["head_site"]]; R = d.site_xmat[E["head_site"]].reshape(3, 3)
        v = R.T @ (self.goal - hp); dist = np.linalg.norm(v) + 1e-6
        az = np.arctan2(v[1], v[0]); el = np.arcsin(np.clip(v[2] / dist, -1, 1))
        vis = float(abs(az) < np.deg2rad(E["azimuth_limit_deg"]))
        return np.array([vis, vis * np.sin(az), vis * np.cos(az), vis * np.sin(el), vis * 2 * np.arctan(E["goal_radius"] / dist)])

    def observe(self):
        d, E = self.d, self.E
        q = d.xquat[E["thorax_body"]]
        qinv = q * np.array([1, -1, -1, -1]); grav = np.zeros(3); mujoco.mju_rotVecQuat(grav, np.array([0, 0, -1.0]), qinv)
        sd = d.sensordata; g, a, v = E["gyro_adr"], E["accel_adr"], E["velocimeter_adr"]
        forces = np.concatenate([sd[i:i + 3] for i in E["force_adr"]]); touch = sd[E["touch_adr"]]
        return np.concatenate([
            d.qpos[7:], d.qvel[6:] * 0.05, grav, sd[g:g + 3] * 0.05, sd[a:a + 3] * 1e-3, sd[v:v + 3],
            np.tanh(forces), np.tanh(touch), self.see_goal(), self.last_action]).astype(np.float32)

    def step(self, action):
        d, E = self.d, self.E
        lo, hi = np.array(E["ctrl_lo"]), np.array(E["ctrl_hi"])
        ctrl = np.array(E["ctrl0"]) + action * 0.5 * (hi - lo) * E["action_scale"]
        tq = E.get("torque_act", E.get("wing_act", []))
        ctrl[tq] = action[tq]
        ctrl[E["adh_act"]] = 0.5 * (action[E["adh_act"]] + 1)
        d.ctrl[:] = np.clip(ctrl, lo, hi)
        for _ in range(E["n_substeps"]):
            mujoco.mj_step(self.m, d)
        self.last_action[:] = action
        # Gait bookkeeping: flight phases and touchdown alternation while standing on the hind legs.
        fc = np.array([self.lowest_z(self.hind_left).min() < 0.003, self.lowest_z(self.hind_right).min() < 0.003], float)
        bip = self.bipedal(); walking = bip and fc.sum() >= 1
        fore_clear = self.lowest_z(self.fore).min() > E["clearance"]
        body_clear = self.lowest_z(self.body).min() >= 0
        airborne = fore_clear and body_clear and fc.sum() == 0 and self.thorax()[2] > E["biped_min_height"]
        if bip or airborne:
            self.gait["walk_steps"] += 1; self.gait["flight"] += int(airborne)
        td = fc * (1 - self._last_fc)
        if td.sum() == 2:
            if bip: self.gait["td_both"] += 1
            self._last_td = -1
        elif td.sum() == 1:
            foot = int(td.argmax())
            if bip:
                if foot == self._last_td: self.gait["td_same"] += 1
                else: self.gait["td_alt"] += 1
            self._last_td = foot
        self._last_fc = fc
        dist = np.linalg.norm(self.thorax()[:2] - self.goal[:2])
        if dist < E["goal"]["reach_radius"] and walking:
            self.reached += 1; self.sample_goal()
        return self.observe(), self.fell()

    def gait_stats(self) -> dict:
        g = self.gait; td = g["td_alt"] + g["td_same"] + g["td_both"]
        return {"flight_frac": g["flight"] / max(1, g["walk_steps"]), "alternation": g["td_alt"] / max(1, td), "touchdowns": td}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--video", default=str(fm.ROOT / "web/assets/demo.mp4"))
    ap.add_argument("--fps", type=int, default=50)
    args = ap.parse_args()
    consts = json.loads((Path(args.run) / "env_constants.json").read_text())   # written by train.py; no GPU needed
    params = load_params(Path(args.run) / "policy.pkl")
    policy = NumpyPolicy(export_numpy(params, consts["nu"]), stochastic=True)
    env = CpuEnv(consts)
    E = env.E
    steps = int(args.seconds / E["ctrl_dt"])
    frames, stats = [], []
    renderer = mujoco.Renderer(env.m, 480, 640)
    for ep in range(args.episodes):
        obs = env.reset(); biped_steps = 0; fallen_steps = 0
        for t in range(steps):
            obs, fell = env.step(policy(obs))
            biped_steps += env.bipedal(); fallen_steps += fell
            if ep == 0 and t % max(1, int(1 / (args.fps * E["ctrl_dt"]))) == 0:
                renderer.update_scene(env.d, camera="track1"); frames.append(renderer.render())
        stats.append({"goals": env.reached, "bipedal_frac": biped_steps / steps, "fallen_frac": fallen_steps / steps})
        print(f"episode {ep}: goals={env.reached} bipedal={biped_steps / steps:.2f} fallen={fallen_steps / steps:.2f}")
    summary = {k: float(np.mean([s[k] for s in stats])) for k in stats[0]}
    print("mean:", summary)
    Path(args.run, "eval.json").write_text(json.dumps({"episodes": stats, "mean": summary}, indent=1))
    if frames:
        imageio.mimwrite(args.video, frames, fps=args.fps, codec="libx264", quality=7)
        print("wrote", args.video)


if __name__ == "__main__":
    main()
