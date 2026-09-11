"""Train the bipedal fly with Brax PPO on MuJoCo Warp.

Usage:  python -m flybiped.train --run runs/v1 [--timesteps 150e6] [--restore runs/v0]
Checkpoints, TensorBoard-style scalar logs (JSON lines) and the final policy
are written under the run directory.
"""
from __future__ import annotations

import argparse
import functools
import json
import time
from pathlib import Path

import jax
import numpy as np

from flybiped import compat  # noqa: F401  (JAX/Brax version shims)
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import wrapper

from flybiped.env import FlyBiped, default_config


def ppo_config(num_timesteps: int, num_envs: int, episode_length: int, eval_every: int) -> dict:
    return dict(
        num_timesteps=num_timesteps,
        num_envs=num_envs,
        episode_length=episode_length,
        num_evals=max(1, num_timesteps // eval_every),   # each eval also writes a checkpoint
        reward_scaling=1.0,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=32,
        num_minibatches=32,
        num_updates_per_batch=4,
        discounting=0.996,             # ~1 s horizon at 4 ms control (gamma = 1 - dt/T)
        gae_lambda=0.95,
        learning_rate=3e-4,
        entropy_cost=5e-3,
        batch_size=num_envs // 32,
        clipping_epsilon=0.2,
        max_grad_norm=1.0,
        learning_rate_schedule="ADAPTIVE_KL",   # legged_gym convention: lr x/÷1.5 to track desired_kl
        desired_kl=0.01,
        num_resets_per_eval=0,   # keep env states across epochs: episodes end by episode_length, not every epoch
    )


make_networks = functools.partial(
    ppo_networks.make_ppo_networks,
    policy_hidden_layer_sizes=(512, 256, 128),
    value_hidden_layer_sizes=(512, 256, 128))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--timesteps", type=float, default=150e6)
    ap.add_argument("--num_envs", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=float, default=1e6, help="steps between evals/checkpoints")
    ap.add_argument("--restore", default=None, help="run dir to restore params from")
    ap.add_argument("--override", default="{}", help="JSON env config overrides")
    args = ap.parse_args()

    run = Path(args.run).resolve()   # orbax needs absolute checkpoint paths
    run.mkdir(parents=True, exist_ok=True)
    overrides = json.loads(args.override)
    overrides.setdefault("naconmax", 48 * args.num_envs)
    env = FlyBiped(config_overrides=overrides)
    (run / "env_config.json").write_text(json.dumps(env._config.to_dict(), indent=1))
    from flybiped.export import env_constants
    (run / "env_constants.json").write_text(json.dumps(env_constants(env)))  # used by watch_export (no GPU)

    cfg = ppo_config(int(args.timesteps), args.num_envs, env._config.episode_length, int(args.eval_every))
    (run / "ppo_config.json").write_text(json.dumps(cfg, indent=1))
    log = open(run / "progress.jsonl", "a")
    t0 = time.time()

    def progress(step, metrics):
        row = {"step": int(step), "time": time.time() - t0,
               **{k: float(v) for k, v in metrics.items()}}
        log.write(json.dumps(row) + "\n"); log.flush()
        keys = ("training/sps", "training/policy_loss", "training/v_loss", "training/entropy_loss", "training/learning_rate")
        print(f"[{row['time']/60:6.1f} min] step {step/1e6:7.2f}M  " +
              "  ".join(f"{k.split('/')[-1]}={row[k]:.3f}" for k in keys if k in row), flush=True)

    restore = None
    if args.restore:
        from brax.training.agents.ppo import checkpoint as ckpt
        ckpts = sorted(p for p in Path(args.restore).resolve().joinpath("checkpoints").iterdir()
                       if p.is_dir() and p.name.isdigit())      # skip orbax *tmp* dirs left by killed runs
        restore = ckpt.load(str(ckpts[-1]))

    make_inference_fn, params, _ = ppo.train(
        environment=env, eval_env=None,   # no GPU evaluator (run_evals=False); metrics come from watch_export
        wrap_env_fn=wrapper.wrap_for_brax_training,
        network_factory=make_networks,
        progress_fn=progress,
        seed=args.seed,
        save_checkpoint_path=str(run / "checkpoints"),
        restore_params=restore,
        log_training_metrics=True,
        run_evals=False,      # the Brax evaluator leaks GPU memory per call on Warp; metrics come from watch_export (CPU)
        **cfg)
    from flybiped.policy import save_params
    save_params(params, run / "policy.pkl")
    print("done; wall time %.1f min" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
