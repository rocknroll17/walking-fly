"""Fruit-fly "stand up and walk on two legs" goal-reaching environment.

Runs on MJX with the MuJoCo Warp backend. The fly starts in its natural
six-leg stance (or, for a fraction of episodes, already reared up on its hind
legs) and must reach a visible goal sphere. Only progress made while the
front and middle legs are off the ground counts, so the policy has to learn
to rear up and walk bipedally. Falling does not end the episode (it is
penalised): the fly has to get up again by itself.
"""
from __future__ import annotations

import json
import os
from typing import Any

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math as mjx_math
from mujoco_playground._src import mjx_env

from flybiped import model as fm
from flybiped import pose as fp


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=2 * fm.CONTROL_DT,     # 4 ms: action repeat over the 2 ms flybody rate
        sim_dt=fm.PHYSICS_DT,
        episode_length=1000,           # 4 s
        action_scale=1.0,              # fraction of the actuator range around the stance pose
        naconmax=48 * 4096,            # contact arena shared by ALL worlds: set to ~48 * num_envs
        njmax=250,                     # constraint rows per world
        warn_overflow=False,           # print MuJoCo Warp arena-overflow warnings (debugging)
        goal=config_dict.create(
            dist_min=0.4, dist_max=1.5, height=0.2, reach_radius=0.2,
        ),
        init=config_dict.create(
            biped_start=0.0,           # start already reared up on the hind legs
            drop_start=0.3,            # dropped in a random orientation (fall recovery)
            flip_start=0.0,            # starts lying on its back (recovery from a backward fall)
            drop_height=0.5, drop_joint_noise=0.2, settle_time=0.25,
            joint_noise=0.03, pitch_noise_deg=2.0, vel_noise=0.3,
        ),
        # Per-step reward weights. No fall termination: contacts are penalised instead
        # (cf. MuJoCo Playground Go1 Getup, TumblerNet) so fall recovery is learnable.
        reward=config_dict.create(
            progress=2.0,      # cm/s towards the goal, only while bipedal
            reach=20.0,        # goal reached (bipedal), goal respawns
            facing=0.1,        # heading towards the goal, only while bipedal
            stall=-0.05,       # bipedal but not moving while far from the goal
            bipedal=1.0,       # strict "standing on the front legs" indicator
            height=0.0,        # thorax height is tricky for handstand, rely on orientation instead
            orientation=1.0,   # handstand orientation from handstand_pose.json
            upright=1.0,       # self-righting toward handstand
            posture=1.0,       # keep T1 planted
            fore_contact=-5.0, # any front/middle leg touching the floor -> Extreme penalty (lava floor for front legs)
            body_contact=-1.0, # thorax/head/abdomen/wings touching the floor
            action_rate=-0.002,
            wing_effort=0.0,     # (Removed) Let it use wings freely for balance
            wing_pose=0.0,       # (Removed) Let it unfold wings for balance
            leg_effort=-0.0005,
            # Standard locomotion regularisers (MuJoCo Playground Go1 joystick / legged_gym),
            # re-scaled to fly units (cm, cm/s, rad/s) so each is O(0.1) per step.
            lin_vel_z=-0.05,     # hopping
            double_air=-1.0,     # whole fly airborne after standing (a hop), not walking (CyberDog2 gait reg.)
            single_support=0.2,  # exactly one hind foot down while bipedal and moving: alternating gait
            alternate=0.5,       # touchdown on the other foot than last time (+), same foot again or both at once (-)
            ang_vel_xy=-0.0002,  # wobble
            dof_pos_limits=-1.0, # joints pushed past 95 % of their range
            feet_slip=-0.1,      # hind feet sliding while in contact, only while bipedal
            feet_air_time=2.0,   # reward proper swing phases of the hind feet, only while bipedal
            dof_vel=-2e-5,       # joint speeds beyond dof_vel_limit (rad/s): unrealistic flailing (Hwangbo 2019 style)
        ),
        dof_vel_limit=60.0,     # rad/s; fly step period ~110 ms, swing ~50 ms -> joint peaks of tens of rad/s (JEB 2013)
        feet_air_time_min=0.05, # s, swing shorter than this earns nothing
        push=config_dict.create(       # random external pushes on the thorax (robustness, cf. Playground Go1 pert_config)
            enable=True, interval_min=0.8, interval_max=2.0, duration=0.02, magnitude=0.8,  # magnitude x body weight
        ),
        assist=0.0,                    # stand-up assist: upward pull on the thorax as a fraction of body weight
                                       # (curriculum aid after HoST 2025; the autopilot decays it to 0)
        noise=config_dict.create(      # observation noise (MuJoCo Playground Go1 scales), training only
            level=1.0, joint_pos=0.03, joint_vel=1.5, gyro=0.2, gravity=0.05, linvel=0.1,
        ),
        vision=config_dict.create(     # geometric "what the eyes see" target detection, head frame
            azimuth_limit_deg=155.0,   # Drosophila: ~50 deg posterior blind spot (Zhao et al., Nature 2025)
        ),
        height_stance=0.12,
        height_target=0.18,
        biped_min_height=0.14,
        clearance=0.03,        # cm, lowered because T1 handstand brings the whole body closer to the floor
    )


class FlyBiped(mjx_env.MjxEnv):

    def __init__(self, config: config_dict.ConfigDict | None = None,
                 config_overrides: dict[str, Any] | None = None, physics: bool = True):
        """``physics=False`` skips the GPU model (for exporting constants on a CPU-only machine)."""
        config = config or default_config()
        super().__init__(config, config_overrides)
        self._xml_path = str(fm.BIPED_XML)
        self._mj_model = fm.load()
        self._mj_model.opt.timestep = self._config.sim_dt
        self._mjx_model = None
        if physics:
            # graph_mode NONE: the default Warp graph capture re-captures whenever JAX donates buffers to new
            # addresses, which leaked GPU memory every training iteration (OOM after the third one).
            # Graph capture mode (FLYBIPED_GRAPH_MODE): NONE is leak-free but launches every kernel separately;
            # WARP_STAGED captures once with staging buffers (MJX docs' recommendation) and is faster if stable.
            from mujoco.mjx.warp import types as mjxw_types
            mode = getattr(mjxw_types.GraphMode, os.environ.get("FLYBIPED_GRAPH_MODE", "NONE"))
            self._mjx_model = mjx.put_model(self._mj_model, impl="warp", graph_mode=mode)
            if not self._config.warn_overflow:
                self._mjx_model = self._mjx_model.tree_replace({"opt._impl.warn_overflow": 0})
        self._init_geometry()
        self._init_pose()

    @property
    def static_observation_size(self) -> int:
        """Observation length without touching the GPU (mirrors _get_obs)."""
        m = self._mj_model
        return ((m.nq - 7) + (m.nv - 6) + 3 + 3 + 3 + 3 + 3 * len(self._force_adr) + len(self._touch_adr)
                + 5 + m.nu)

    # ------------------------------------------------------------------ setup
    def _init_geometry(self) -> None:
        m = self._mj_model
        gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""  # noqa: E731
        coll = [g for g in range(1, m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
        self._hind_geoms = jp.array([g for g in coll if "T1" in gname(g) and "tars" in gname(g)])
        self._fore_geoms = jp.array([g for g in coll if "T3" in gname(g) or "T2" in gname(g)])
        self._body_geoms = jp.array([g for g in coll if not any(t in gname(g) for t in ("T1", "T2", "T3")) or ("T1" in gname(g) and "tars" not in gname(g))])
        self._hind_left = jp.array([g for g in coll if "T1_left" in gname(g) and "tars" in gname(g)])
        self._hind_right = jp.array([g for g in coll if "T1_right" in gname(g) and "tars" in gname(g)])
        self._claw_sites = jp.array([m.site("claw_T1_left").id, m.site("claw_T1_right").id])
        lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]          # hinge joints (free joint excluded)
        c, r = 0.5 * (lo + hi), hi - lo
        self._soft_lo, self._soft_hi = jp.array(c - 0.475 * r), jp.array(c + 0.475 * r)
        self._geom_size = jp.array(m.geom_size)
        self._geom_type = jp.array(m.geom_type)
        self._thorax = m.body("thorax").id
        self._abdomen = m.body("abdomen").id
        self._thorax_site = m.site("thorax").id
        self._head_site = m.site("head").id
        self._goal_geom = m.geom("goal").id
        self._gyro = int(m.sensor("gyro").adr[0])
        self._accel = int(m.sensor("accelerometer").adr[0])
        self._velocimeter = int(m.sensor("velocimeter").adr[0])
        names = [m.sensor(i).name for i in range(m.nsensor)]
        self._force_adr = jp.array([int(m.sensor_adr[i]) for i, n in enumerate(names) if n.startswith("force_")])
        self._touch_adr = jp.array([int(m.sensor_adr[i]) for i, n in enumerate(names) if n.startswith("touch_")])
        aname = [m.actuator(i).name for i in range(m.nu)]
        # Wings are position servos like the legs (see model._wings_as_position_servos), so no
        # actuator gets torque-style treatment; the list is kept for the wing effort term/export.
        self._wing_act = jp.array([i for i, n in enumerate(aname) if "wing" in n])
        self._torque_act = jp.array([], dtype=jp.int32)
        self._adh_act = jp.array([i for i, n in enumerate(aname) if "adhere" in n])
        self._leg_act = jp.array([i for i, n in enumerate(aname) if "_T" in n and "adhere" not in n])
        self._leg_qadr = jp.array([m.jnt_qposadr[j] for j in range(m.njnt) if "_T" in m.joint(j).name])
        self._hind_qadr = jp.array([m.jnt_qposadr[j] for j in range(m.njnt) if "_T1" in m.joint(j).name])
        self._weight = float(m.body_subtreemass[self._thorax] * -m.opt.gravity[2])   # dyn
        self._ctrl_lo = jp.array(m.actuator_ctrlrange[:, 0])
        self._ctrl_hi = jp.array(m.actuator_ctrlrange[:, 1])
        self._act_adr = jp.array(m.actuator_actadr)
        wing_j = [j for j in range(m.njnt) if "wing" in m.joint(j).name]
        self._wing_qadr = jp.array([m.jnt_qposadr[j] for j in wing_j])
        self._wing_rest = jp.array([m.qpos_spring[m.jnt_qposadr[j]] for j in wing_j])

    def _init_pose(self) -> None:
        pose = json.loads((fm.BUILD_DIR / "handstand_pose.json").read_text())
        self._q_stance = jp.array(pose["qpos_stance"])
        self._ctrl_stance = jp.array(pose["ctrl_stance"])
        self._q_biped = jp.array(pose["qpos_settled"])
        self._ctrl_biped = jp.array(pose["ctrl"])
        self._ctrl0 = self._ctrl_stance   # action centre
        self._leg_q_stance = self._q_stance[self._leg_qadr]
        self._hind_q_stance = self._q_stance[self._hind_qadr]
        q = self._q_biped[3:7]
        self._gravity_biped = mjx_math.rotate(jp.array([0.0, 0, -1]), mjx_math.quat_inv(q))

    # -------------------------------------------------------------- geometry
    def _lowest_z(self, data: mjx.Data, geoms: jax.Array) -> jax.Array:
        """Lowest world-z point of each primitive geom (sphere/capsule/ellipsoid/box)."""
        pos = data.geom_xpos[geoms]
        R = data.geom_xmat[geoms].reshape(-1, 3, 3)
        size = self._geom_size[geoms]
        typ = self._geom_type[geoms]
        zrow = jp.abs(R[:, 2, :])                       # world-z components of local axes
        r = size[:, 0]
        capsule = r + zrow[:, 2] * size[:, 1]
        ellipsoid = jp.sqrt(jp.sum((size * zrow) ** 2, axis=1))
        box = jp.sum(size * zrow, axis=1)
        drop = jp.where(typ == mujoco.mjtGeom.mjGEOM_CAPSULE, capsule,
               jp.where(typ == mujoco.mjtGeom.mjGEOM_ELLIPSOID, ellipsoid,
               jp.where(typ == mujoco.mjtGeom.mjGEOM_BOX, box,
               jp.where(typ == mujoco.mjtGeom.mjGEOM_CYLINDER, capsule, r))))
        return pos[:, 2] - drop

    # ---------------------------------------------------------------- reset
    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, k_mode, k_yaw, k_pitch, k_joint, k_vel, k_goal, k_drop = jax.random.split(rng, 8)
        cfg = self._config.init
        u = jax.random.uniform(k_mode)
        biped = u < cfg.biped_start
        drop = (u >= cfg.biped_start) & (u < cfg.biped_start + cfg.drop_start)
        flip = (u >= cfg.biped_start + cfg.drop_start) & (u < cfg.biped_start + cfg.drop_start + cfg.flip_start)
        qpos = jp.where(biped, self._q_biped, self._q_stance)
        ctrl = jp.where(biped, self._ctrl_biped, self._ctrl_stance)
        yaw = jax.random.uniform(k_yaw, (), minval=-jp.pi, maxval=jp.pi)
        dpitch = jp.deg2rad(cfg.pitch_noise_deg) * jax.random.normal(k_pitch, ())
        quat = mjx_math.quat_mul(mjx_math.axis_angle_to_quat(jp.array([0.0, 0, 1]), yaw), qpos[3:7])
        quat = mjx_math.quat_mul(quat, mjx_math.axis_angle_to_quat(jp.array([0.0, 1, 0]), dpitch))
        qpos = qpos.at[3:7].set(quat)
        qpos = qpos.at[7:].add(cfg.joint_noise * jax.random.normal(k_joint, (qpos.shape[0] - 7,)))
        # Dropped start: random orientation from a height, larger joint noise.
        kq, kj = jax.random.split(k_drop)
        rand_quat = jax.random.normal(kq, (4,)); rand_quat = rand_quat / (jp.linalg.norm(rand_quat) + 1e-6)
        q_drop = self._q_stance.at[2].set(cfg.drop_height).at[3:7].set(rand_quat)
        q_drop = q_drop.at[7:].add(cfg.drop_joint_noise * jax.random.normal(kj, (qpos.shape[0] - 7,)))
        qpos = jp.where(drop, q_drop, qpos)
        # Flipped start: six-leg pose rotated onto its back, just above the floor.
        q_flip = self._q_stance.at[2].set(0.25).at[3:7].set(
            mjx_math.quat_mul(mjx_math.quat_mul(mjx_math.axis_angle_to_quat(jp.array([0.0, 0, 1]), yaw),
                                                self._q_stance[3:7]),
                              mjx_math.axis_angle_to_quat(jp.array([1.0, 0, 0]), jp.pi)))
        qpos = jp.where(flip, q_flip, qpos)
        drop = drop | flip   # both need the settle phase below
        qvel = jp.zeros(self.mjx_model.nv).at[:6].set(cfg.vel_noise * jax.random.normal(k_vel, (6,)))
        act = jp.zeros(self.mjx_model.na).at[self._act_adr].set(ctrl)
        data = mjx_env.make_data(
            self._mj_model, qpos=qpos, qvel=qvel, ctrl=ctrl, act=act,
            impl=self.mjx_model.impl.value,
            naconmax=self._config.naconmax, njmax=self._config.njmax)
        lift = self._config.assist * self._weight
        data = data.replace(xfrc_applied=data.xfrc_applied.at[self._abdomen, 2].set(lift))
        data = mjx.forward(self.mjx_model, data)
        # Let dropped flies land and settle (all worlds pay the cost; only drops need it).
        settle_steps = int(cfg.settle_time / self._config.sim_dt)
        settled = mjx_env.step(self.mjx_model, data, ctrl, settle_steps)
        data = data.where(drop, settled)   # MJX-aware merge (contact arena is shared across worlds)
        data = data.replace(time=jp.zeros(()))
        goal = self._sample_goal(k_goal, data.qpos[:3])
        data = data.replace(mocap_pos=goal[None])
        data = mjx.forward(self.mjx_model, data)
        info = {
            "rng": rng,
            "goal": goal,
            "last_action": jp.zeros(self.action_size),
            "last_dist": self._goal_dist(data, goal),
            "feet_air_time": jp.zeros(2),
            "last_td_foot": -jp.ones(()),      # -1 none, 0 left, 1 right: which hind foot touched down last
            "push_t": jp.zeros(()),            # time until the next push starts
            "push_left": jp.zeros(()),         # remaining duration of the current push
            "push_force": jp.zeros(3),
            "last_feet_contact": jp.ones(2),
            "last_claw_xy": data.site_xpos[self._claw_sites][:, :2],
        }
        metrics = {f"reward/{k}": jp.zeros(()) for k in self._config.reward.keys()}
        metrics.update({"goals_reached": jp.zeros(()), "bipedal_frac": jp.zeros(()), "walking_frac": jp.zeros(()),
                        "airborne_frac": jp.zeros(()),
                        "body_contact_frac": jp.zeros(()), "height": jp.zeros(()), "nan": jp.zeros(())})
        obs = self._get_obs(data, info)
        return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    def _sample_goal(self, rng: jax.Array, origin: jax.Array) -> jax.Array:
        cfg = self._config.goal
        k1, k2 = jax.random.split(rng)
        ang = jax.random.uniform(k1, (), minval=-jp.pi, maxval=jp.pi)
        dist = jax.random.uniform(k2, (), minval=cfg.dist_min, maxval=cfg.dist_max)
        return jp.array([origin[0] + dist * jp.cos(ang), origin[1] + dist * jp.sin(ang), cfg.height])

    # ----------------------------------------------------------------- step
    def _push(self, data: mjx.Data, info: dict) -> mjx.Data:
        """Random horizontal shoves on the thorax; the assist lift stays on z."""
        cfg = self._config.push
        rng, k1, k2 = jax.random.split(info["rng"], 3)
        info["rng"] = rng
        start = (info["push_t"] <= 0.0) & (info["push_left"] <= 0.0)
        ang = jax.random.uniform(k1, (), minval=-jp.pi, maxval=jp.pi)
        mag = cfg.magnitude * self._weight * float(cfg.enable)
        new_force = jp.array([mag * jp.cos(ang), mag * jp.sin(ang), 0.0])
        info["push_force"] = jp.where(start, new_force, info["push_force"])
        info["push_left"] = jp.where(start, cfg.duration, jp.maximum(info["push_left"] - self.dt, 0.0))
        info["push_t"] = jp.where(start, jax.random.uniform(k2, (), minval=cfg.interval_min, maxval=cfg.interval_max),
                                  info["push_t"] - self.dt)
        active = (info["push_left"] > 0.0).astype(jp.float32)
        xy = info["push_force"][:2] * active
        return data.replace(xfrc_applied=data.xfrc_applied.at[self._thorax, :2].set(xy))

    def _fresh_info(self, data: mjx.Data, info: dict) -> dict:
        """Re-initialise per-episode bookkeeping right after an auto-reset.

        Brax's AutoResetWrapper restores ``data``/``obs`` from the first reset but
        keeps ``info`` from the finished episode. A freshly reset world has
        ``data.time == 0``; its goal is recoverable from the restored mocap body.
        """
        fresh = data.time <= 0.0
        goal = jp.where(fresh, data.mocap_pos[0], info["goal"])
        pick = lambda new, old: jp.where(fresh, new, old)  # noqa: E731
        info = dict(info)
        info.update(
            goal=goal,
            last_dist=pick(self._goal_dist(data, goal), info["last_dist"]),
            last_action=pick(jp.zeros_like(info["last_action"]), info["last_action"]),
            feet_air_time=pick(jp.zeros(2), info["feet_air_time"]),
            last_feet_contact=pick(jp.ones(2), info["last_feet_contact"]),
            last_claw_xy=pick(data.site_xpos[self._claw_sites][:, :2], info["last_claw_xy"]),
            last_td_foot=pick(-jp.ones(()), info["last_td_foot"]),
            push_t=pick(jp.zeros(()), info["push_t"]),
            push_left=pick(jp.zeros(()), info["push_left"]),
        )
        return info

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        action = jp.clip(action, -1.0, 1.0)
        ctrl = self._action_to_ctrl(action)
        state.info.update(self._fresh_info(state.data, state.info))
        data = self._push(state.data, state.info)
        data = mjx_env.step(self.mjx_model, data, ctrl, self.n_substeps)

        goal = state.info["goal"]
        dist = self._goal_dist(data, goal)
        contacts = self._contacts(data)
        bipedal = contacts["bipedal"]
        reached = (dist < self._config.goal.reach_radius) & bipedal
        nan = jp.any(~jp.isfinite(data.qpos)) | jp.any(~jp.isfinite(data.qvel))

        feet_contact = jp.array([jp.min(self._lowest_z(data, self._hind_left)) < 0.003,
                                 jp.min(self._lowest_z(data, self._hind_right)) < 0.003]).astype(jp.float32)
        # Walking = standing on the hind legs with at least one foot on the ground.
        # Airborne = the whole fly off the floor (a hop): body clear, front legs clear, no hind contact.
        walking = bipedal & (jp.sum(feet_contact) >= 1.0)
        airborne = contacts["airborne"]
        reached = reached & walking
        touchdown = feet_contact * (1.0 - state.info["last_feet_contact"])
        n_td = jp.sum(touchdown)
        td_foot = jp.argmax(touchdown).astype(jp.float32)
        last_foot = state.info["last_td_foot"]
        alternate = jp.where(n_td == 2.0, -1.0,                                   # both feet land together: hop
                    jp.where(n_td == 1.0, jp.where(td_foot == last_foot, -0.5, 1.0), 0.0)) * walking.astype(jp.float32)
        new_last = jp.where(n_td == 1.0, td_foot, jp.where(n_td == 2.0, -1.0, last_foot))
        rewards = self._rewards(data, state.info, action, dist, reached, contacts, feet_contact, walking, alternate, airborne)
        reward = sum(rewards[k] * v for k, v in self._config.reward.items())
        reward = jp.where(nan, 0.0, reward)

        # Spawn a fresh goal once the current one is reached.
        rng, k_goal = jax.random.split(state.info["rng"])
        new_goal = self._sample_goal(k_goal, data.qpos[:3])
        goal = jp.where(reached, new_goal, goal)
        data = data.replace(mocap_pos=goal[None])
        dist = jp.where(reached, self._goal_dist(data, goal), dist)

        air = (state.info["feet_air_time"] + self._config.ctrl_dt) * (1.0 - feet_contact)
        dist = jp.where(nan, 0.0, dist)                 # never carry NaN into the next episode's progress
        state.info.update(rng=rng, goal=goal, last_action=action, last_dist=dist,
                          feet_air_time=air, last_feet_contact=feet_contact, last_td_foot=new_last,
                          last_claw_xy=jp.nan_to_num(data.site_xpos[self._claw_sites][:, :2]))
        state.metrics.update({f"reward/{k}": v for k, v in rewards.items()})
        # Brax sums metrics over an episode: log per-step indicators, not running totals.
        state.metrics.update(goals_reached=reached.astype(jp.float32),
                             bipedal_frac=bipedal.astype(jp.float32), walking_frac=walking.astype(jp.float32),
                             airborne_frac=airborne.astype(jp.float32),
                             body_contact_frac=contacts["body"].astype(jp.float32),
                             height=data.site_xpos[self._thorax_site][2])
        obs = self._get_obs(data, state.info)
        # A diverged world is terminated and must not leak NaNs into the observation
        # normaliser or the metrics; the auto-reset wrapper restores it next step.
        obs = jp.where(nan, jp.zeros_like(obs), obs)
        state.metrics.update({k: jp.where(nan, 0.0, v) for k, v in state.metrics.items()})
        state.metrics.update(nan=nan.astype(jp.float32))
        return state.replace(data=data, obs=obs, reward=reward, done=nan.astype(jp.float32))

    def _action_to_ctrl(self, action: jax.Array) -> jax.Array:
        half = 0.5 * (self._ctrl_hi - self._ctrl_lo)
        ctrl = self._ctrl0 + action * half * self._config.action_scale   # position actuators
        ctrl = ctrl.at[self._torque_act].set(action[self._torque_act])      # (none: wings are servos now)
        ctrl = ctrl.at[self._adh_act].set(0.5 * (action[self._adh_act] + 1.0))  # adhesion 0..1
        return jp.clip(ctrl, self._ctrl_lo, self._ctrl_hi)

    # ------------------------------------------------------------ quantities
    def _goal_dist(self, data: mjx.Data, goal: jax.Array) -> jax.Array:
        return jp.linalg.norm(data.site_xpos[self._thorax_site][:2] - goal[:2])

    def _contacts(self, data: mjx.Data) -> dict[str, jax.Array]:
        """Floor-contact indicators and the strict "standing on the hind legs" test."""
        cfg = self._config
        fore_low = self._lowest_z(data, self._fore_geoms)
        hind_low = self._lowest_z(data, self._hind_geoms)
        body_low = self._lowest_z(data, self._body_geoms)
        height = data.site_xpos[self._thorax_site][2]
        fore = jp.min(fore_low) < 0.003
        body = jp.min(body_low) < 0.0
        hind = jp.min(hind_low) < 0.003
        fore_clear = jp.min(fore_low) > cfg.clearance
        bipedal = fore_clear & hind & ~body & (height > cfg.biped_min_height)
        airborne = fore_clear & ~hind & ~body & (height > cfg.biped_min_height)
        return {"fore": fore, "body": body, "hind": hind, "bipedal": bipedal, "airborne": airborne}

    def _heading_frame(self, data: mjx.Data):
        """Rotation matrix of the thorax yaw-only frame and the thorax quaternion."""
        quat = data.xquat[self._thorax]
        fwd = mjx_math.rotate(jp.array([1.0, 0, 0]), quat)
        yaw = jp.arctan2(fwd[1], fwd[0])
        c, s = jp.cos(yaw), jp.sin(yaw)
        return jp.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]), quat

    def _rewards(self, data, info, action, dist, reached, contacts, feet_contact, walking, alternate, airborne) -> dict[str, jax.Array]:
        cfg = self._config
        R, quat = self._heading_frame(data)
        to_goal = info["goal"] - data.site_xpos[self._thorax_site]
        dir_xy = to_goal[:2] / (jp.linalg.norm(to_goal[:2]) + 1e-6)
        progress = jp.clip((info["last_dist"] - dist) / cfg.ctrl_dt, -3.0, 3.0)
        height = data.site_xpos[self._thorax_site][2]
        gravity = mjx_math.rotate(jp.array([0.0, 0, -1]), mjx_math.quat_inv(quat))
        speed = jp.linalg.norm(data.sensordata[self._velocimeter:self._velocimeter + 3])
        b = contacts["bipedal"].astype(jp.float32)
        w = walking.astype(jp.float32)            # progress/facing/reach only count while actually walking
        far = dist > cfg.goal.reach_radius + 0.1
        gyro = data.sensordata[self._gyro:self._gyro + 3]
        vz = data.qvel[2]                                   # world-frame vertical velocity of the root
        q = data.qpos[7:]
        limit_violation = jp.sum(jp.clip(self._soft_lo - q, 0, None) + jp.clip(q - self._soft_hi, 0, None))
        claw_xy = data.site_xpos[self._claw_sites][:, :2]
        foot_speed = jp.linalg.norm(claw_xy - info["last_claw_xy"], axis=1) / cfg.ctrl_dt
        slip = jp.sum(foot_speed * feet_contact) * b
        touchdown = feet_contact * (1.0 - info["last_feet_contact"])
        air_time = jp.sum(jp.clip(info["feet_air_time"] - cfg.feet_air_time_min, 0, None) * touchdown) * b
        over = jp.clip(jp.abs(data.qvel[6:]) - cfg.dof_vel_limit, 0.0, None)
        n_down = jp.sum(feet_contact)
        moving = (speed > 0.3).astype(jp.float32)
        return {
            "double_air": airborne.astype(jp.float32),
            "single_support": b * moving * (n_down == 1).astype(jp.float32),
            "dof_vel": jp.sum(jp.square(over)),
            "lin_vel_z": jp.square(vz),
            "ang_vel_xy": jp.sum(jp.square(gyro[:2])),
            "dof_pos_limits": limit_violation,
            "feet_slip": slip,
            "feet_air_time": air_time,
            "progress": progress * w,
            "reach": reached.astype(jp.float32),
            "facing": jp.dot(R[0, :2], dir_xy) * w,
            "alternate": alternate,
            "stall": b * far.astype(jp.float32) * (speed < 0.1).astype(jp.float32),
            "bipedal": b,
            "height": jp.clip((height - cfg.height_stance) / (cfg.height_target - cfg.height_stance), 0.0, 1.0),
            "orientation": jp.exp(-2.0 * jp.sum(jp.square(gravity - self._gravity_biped))),
            "upright": 0.5 * (1.0 + jp.dot(gravity, self._gravity_biped)),   # -1 (on its back) .. +1 (upright) -> 0..1
            # Once the body faces up, pull the hind legs back to the stance angles so it stands instead of lying on them.
            # We only apply this to hind legs (_T3) so front legs aren't encouraged to point at the ground.
            "posture": (jp.dot(gravity, self._gravity_biped) > 0.5).astype(jp.float32)
                       * jp.exp(-0.5 * jp.sum(jp.square(data.qpos[self._hind_qadr] - self._hind_q_stance))),
            "fore_contact": contacts["fore"].astype(jp.float32),
            "body_contact": contacts["body"].astype(jp.float32),
            "action_rate": jp.sum(jp.square(action - info["last_action"])),
            "wing_effort": jp.sum(jp.square(action[self._wing_act])),
            # Folded-wing penalty only while not lying on the body: insects self-right by pushing with open wings.
            "wing_pose": jp.sum(jp.square(data.qpos[self._wing_qadr] - self._wing_rest)) * (1.0 - contacts["body"].astype(jp.float32)),
            "leg_effort": jp.sum(jp.square(action[self._leg_act])),
        }

    def _see_goal(self, data: mjx.Data, goal: jax.Array) -> jax.Array:
        """Goal as seen from the head: [visible, sin/cos azimuth, sin elevation, angular size].

        This stands in for a visual target detector: only geometry the eyes could
        resolve is exposed, and nothing outside the compound eyes' field of view.
        """
        head_pos = data.site_xpos[self._head_site]
        R = data.site_xmat[self._head_site].reshape(3, 3)      # head frame axes in world
        d_world = goal - head_pos
        d = R.T @ d_world                                        # head-frame vector
        dist = jp.linalg.norm(d) + 1e-6
        az = jp.arctan2(d[1], d[0])
        el = jp.arcsin(jp.clip(d[2] / dist, -1.0, 1.0))
        visible = jp.abs(az) < jp.deg2rad(self._config.vision.azimuth_limit_deg)
        size = 2.0 * jp.arctan(fm.GOAL_RADIUS / dist)
        v = visible.astype(jp.float32)
        return jp.array([v, v * jp.sin(az), v * jp.cos(az), v * jp.sin(el), v * size])

    def _get_obs(self, data: mjx.Data, info: dict) -> jax.Array:
        """Sensor-only observation (nothing a real fly/robot could not measure):
        joint encoders, IMU (gravity, gyro, accelerometer), body-velocity estimate,
        tarsal force/touch sensors, the goal as seen by the eyes, last action.
        Uniform sensor noise is added during training (level 0 = clean, as in the viewer)."""
        quat = data.xquat[self._thorax]
        gravity = mjx_math.rotate(jp.array([0.0, 0, -1]), mjx_math.quat_inv(quat))
        sd = data.sensordata
        gyro = sd[self._gyro:self._gyro + 3]
        accel = sd[self._accel:self._accel + 3]
        linvel = sd[self._velocimeter:self._velocimeter + 3]
        forces = jp.concatenate([sd[a:a + 3] for a in np.asarray(self._force_adr)])
        touch = sd[self._touch_adr]
        qpos, qvel = data.qpos[7:], data.qvel[6:]
        nz = self._config.noise
        if nz.level > 0.0:
            rng, k1, k2, k3, k4, k5 = jax.random.split(info["rng"], 6)
            info["rng"] = rng
            u = lambda k, shape, scale: (2.0 * jax.random.uniform(k, shape) - 1.0) * nz.level * scale  # noqa: E731
            qpos = qpos + u(k1, qpos.shape, nz.joint_pos)
            qvel = qvel + u(k2, qvel.shape, nz.joint_vel)
            gyro = gyro + u(k3, (3,), nz.gyro)
            gravity = gravity + u(k4, (3,), nz.gravity)
            linvel = linvel + u(k5, (3,), nz.linvel)
        return jp.concatenate([
            qpos,
            qvel * 0.05,
            gravity, gyro * 0.05, accel * 1e-3, linvel,
            jp.tanh(forces), jp.tanh(touch),
            self._see_goal(data, info["goal"]),
            info["last_action"],
        ])

    # ------------------------------------------------------------ properties
    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mj_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
