"""Export a trained policy plus the environment constants for the browser viewer.

Usage: python -m flybiped.export --run runs/v1 [--out web/assets/policy.json]
Without --run an untrained (random) policy is written so the viewer can be
exercised before training finishes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from flybiped import compat  # noqa: F401
from flybiped import model as fm
from flybiped.env import FlyBiped
from flybiped.policy import export_numpy, load_params


def env_constants(env: FlyBiped) -> dict:
    """Everything the JS side needs to reproduce observations/actions exactly."""
    m = env.mj_model
    cfg = env._config
    return {
        "ctrl_dt": cfg.ctrl_dt, "sim_dt": cfg.sim_dt, "n_substeps": int(round(cfg.ctrl_dt / cfg.sim_dt)),
        "assist": float(cfg.assist), "weight": env._weight, "settle_time": float(cfg.init.settle_time),
        "walk_gate_time": float(cfg.walk_gate_time),
        "action_scale": cfg.action_scale, "clearance": cfg.clearance,
        "goal": dict(cfg.goal),
        "ctrl0": np.asarray(env._ctrl0).tolist(),
        "ctrl_lo": m.actuator_ctrlrange[:, 0].tolist(), "ctrl_hi": m.actuator_ctrlrange[:, 1].tolist(),
        "torque_act": np.asarray(env._torque_act).tolist(),  # actuators driven as raw torque (none now)
        "adh_act": np.asarray(env._adh_act).tolist(),
        "act_adr": m.actuator_actadr.tolist(),
        "hind_geoms": np.asarray(env._hind_geoms).tolist(), "fore_geoms": np.asarray(env._fore_geoms).tolist(),
        "body_geoms": np.asarray(env._body_geoms).tolist(),
        "thorax_body": env._thorax, "thorax_site": env._thorax_site, "head_site": env._head_site,
        "gyro_adr": env._gyro, "accel_adr": env._accel, "velocimeter_adr": env._velocimeter,
        "force_adr": np.asarray(env._force_adr).tolist(), "touch_adr": np.asarray(env._touch_adr).tolist(),
        "goal_radius": fm.GOAL_RADIUS, "azimuth_limit_deg": cfg.vision.azimuth_limit_deg,
        "height_stance": cfg.height_stance, "height_target": cfg.height_target, "biped_min_height": cfg.biped_min_height,
        "q_stance": np.asarray(env._q_stance).tolist(), "ctrl_stance": np.asarray(env._ctrl_stance).tolist(),
        "q_biped": np.asarray(env._q_biped).tolist(), "ctrl_biped": np.asarray(env._ctrl_biped).tolist(),
        "nq": m.nq, "nv": m.nv, "nu": m.nu, "na": m.na,
    }


def random_params(env: FlyBiped, sizes=(512, 256, 128)):
    """Untrained PPO parameters in the same layout Brax produces."""
    from brax.training.acme import running_statistics, specs
    obs = env.static_observation_size
    normalizer = running_statistics.init_state(specs.Array((obs,), np.float32))
    key = jax.random.PRNGKey(0)
    layers, dims = {}, [obs, *sizes, 2 * env.action_size]
    for i in range(len(dims) - 1):
        key, k = jax.random.split(key)
        layers[f"hidden_{i}"] = {"kernel": 0.05 * jax.random.normal(k, (dims[i], dims[i + 1])),
                                 "bias": np.zeros(dims[i + 1])}
    return normalizer, {"params": layers}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="run dir with policy.pkl (from watch_export)")
    ap.add_argument("--checkpoint", default=None, help="orbax checkpoint dir (e.g. runs/imported/checkpoints/000001234)")
    ap.add_argument("--constants", default=None, help="env_constants.json to embed (default: computed from the model)")
    ap.add_argument("--step", type=int, default=0, help="cumulative step count to display")
    ap.add_argument("--out", default=str(fm.ROOT / "web/assets/policy.json"))
    args = ap.parse_args()
    env = FlyBiped(config_overrides={"naconmax": 48}, physics=False)   # constants only: no GPU needed
    if args.checkpoint:
        from brax.training.agents.ppo import checkpoint as ckpt
        params = ckpt.load(str(Path(args.checkpoint).resolve()))
    elif args.run:
        params = load_params(Path(args.run) / "policy.pkl")
    else:
        params = random_params(env)
    spec = export_numpy(params, env.action_size)
    spec["env"] = json.loads(Path(args.constants).read_text()) if args.constants else env_constants(env)
    spec["trained"] = bool(args.run or args.checkpoint)
    spec["step"] = args.step
    Path(args.out).write_text(json.dumps(spec))
    print(f"wrote {args.out} ({Path(args.out).stat().st_size/1e6:.1f} MB), obs={env.static_observation_size} act={env.action_size}")


if __name__ == "__main__":
    main()
