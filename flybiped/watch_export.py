"""Keep web/assets/policy.json in sync with the newest training checkpoint.

Usage: python -m flybiped.watch_export --run runs/v1 [--interval 120]
Polls the run's checkpoint directory; whenever a new checkpoint appears it is
exported for the browser viewer and a copy of the params is saved as
``policy.pkl`` (so evaluation works on partially trained runs too).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from flybiped import compat  # noqa: F401
from flybiped import model as fm
from flybiped.policy import export_numpy, save_params


def newest_checkpoint(run: Path) -> Path | None:
    ckpts = [p for p in (run / "checkpoints").glob("[0-9]*") if p.is_dir() and "tmp" not in p.name]
    return max(ckpts, key=lambda p: int(p.name)) if ckpts else None


def cpu_eval(spec: dict, step: int, seconds: float = 3.0) -> dict:
    """Deterministic CPU rollouts of the exported policy from the three start modes."""
    from flybiped.evaluate import CpuEnv
    from flybiped.policy import NumpyPolicy
    policy = NumpyPolicy(spec, stochastic=True, seed=step)   # same action noise as training rollouts
    modes = [("stance", {}), ("biped", {"biped": True}), ("drop", {"drop": True}), ("flip", {"flip": True})]
    n = int(seconds / spec["env"]["ctrl_dt"])
    assist = spec["env"].get("assist", 0.0)
    res = {"step": step, "assist": assist}
    for tag, a in (("", assist), ("_noassist", 0.0)):     # as trained (harness on) and the real thing
        if tag and a == assist:
            break                                          # no harness in training: one pass is enough
        g = b = f = 0.0
        per_mode = {}; flight = alt = td = 0.0
        for i, (name, kw) in enumerate(modes):
            env = CpuEnv(spec["env"], seed=100 + i, assist=a)
            obs = env.reset(**kw); bb = ff = 0
            for _ in range(n):
                obs, fell = env.step(policy(obs)); bb += env.bipedal(); ff += fell
            g += env.reached / len(modes); b += bb / n / len(modes); f += ff / n / len(modes)
            gs = env.gait_stats()
            per_mode[name] = {"goals": env.reached, "bipedal": bb / n, "body": ff / n, **gs}
            flight += gs["flight_frac"] / len(modes); alt += gs["alternation"] * gs["touchdowns"]; td += gs["touchdowns"]
        res["goals" + tag] = g; res["bipedal" + tag] = b; res["body_contact" + tag] = f
        res["flight" + tag] = flight; res["alternation" + tag] = alt / max(1.0, td)
        res["modes" + tag] = per_mode
    for k in ("goals", "bipedal", "body_contact", "flight", "alternation", "modes"):
        res.setdefault(k + "_noassist", res[k])
    return res


def _export_progress(run: Path, out: Path) -> None:
    """Status-page data: CPU evaluation rows plus training throughput."""
    rows = [json.loads(l) for l in (run / "cpu_eval.jsonl").read_text().splitlines()] if (run / "cpu_eval.jsonl").exists() else []
    sps = None
    for line in (run / "progress.jsonl").read_text().splitlines() if (run / "progress.jsonl").exists() else []:
        r = json.loads(line)
        sps = r.get("training/sps", sps)
    off = run / "step_offset.json"
    offset = json.loads(off.read_text())["offset"] if off.exists() else 0
    out.write_text(json.dumps({"run": run.name, "sps": sps, "offset": offset, "rows": rows}))


def _render_clip(spec: dict, out_dir: Path, name: str, seconds: float = 3.0, fps: int = 25) -> None:
    """Short side-by-side clip (six-leg start | reared-up start) of the exported policy."""
    import imageio
    import mujoco
    import numpy as np
    from flybiped.evaluate import CpuEnv
    from flybiped.policy import NumpyPolicy
    out_dir.mkdir(parents=True, exist_ok=True)
    policy = NumpyPolicy(spec, stochastic=True, seed=1)
    envs = [CpuEnv(spec["env"], seed=0), CpuEnv(spec["env"], seed=1)]
    obs = [envs[0].reset(biped=False), envs[1].reset(biped=True)]
    renderer = mujoco.Renderer(envs[0].m, 300, 400)
    every = max(1, int(round(1 / (fps * spec["env"]["ctrl_dt"]))))
    frames = []
    for t in range(int(seconds / spec["env"]["ctrl_dt"])):
        row = []
        for i, env in enumerate(envs):
            obs[i], _ = env.step(policy(obs[i]))
            if t % every == 0:
                renderer.update_scene(env.d, camera="track1"); row.append(renderer.render())
        if row:
            frames.append(np.concatenate(row, axis=1))
    imageio.mimwrite(out_dir / f"{name}.mp4", frames, fps=fps, codec="libx264", quality=6, macro_block_size=1)
    clips = sorted(p.name for p in out_dir.glob("*.mp4"))
    (out_dir / "index.json").write_text(json.dumps(clips))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="run dir to watch")
    ap.add_argument("--follow", default=None, help="file containing the run dir to watch (autopilot switches it)")
    ap.add_argument("--interval", type=float, default=120)
    ap.add_argument("--out", default=str(fm.ROOT / "web/assets/policy.json"))
    ap.add_argument("--no_clips", action="store_true")
    args = ap.parse_args()
    from brax.training.agents.ppo import checkpoint
    last = None
    # First run after a resume/migration: publish the newest checkpoint of ANY run immediately,
    # so the viewer does not show the untrained placeholder until the first new checkpoint.
    out_path = Path(args.out)
    untrained = not out_path.exists() or not json.loads(out_path.read_text()).get("trained", False)
    if untrained:
        cands = [(newest_checkpoint(r), r) for r in (fm.ROOT / "runs").glob("*/") if (r / "env_constants.json").exists()]
        cands = [(c, r) for c, r in cands if c is not None]
        if cands:
            ck, run = max(cands, key=lambda cr: cr[0].stat().st_mtime)
            try:
                consts = json.loads((run / "env_constants.json").read_text())
                params = checkpoint.load(str(ck))
                spec = export_numpy(params, consts["nu"]); spec["env"] = consts; spec["trained"] = True
                off = run / "step_offset.json"
                spec["step"] = int(ck.name) + (json.loads(off.read_text())["offset"] if off.exists() else 0)
                spec["run"] = run.name
                out_path.write_text(json.dumps(spec))
                print(f"published existing checkpoint {run.name}/{ck.name} for the viewer", flush=True)
            except Exception as e:
                print(f"initial publish skipped: {e}", flush=True)
    while True:
        if args.follow and not Path(args.follow).exists():
            time.sleep(args.interval); continue          # nothing is training yet
        run = Path(Path(args.follow).read_text().strip() if args.follow else args.run).resolve()
        if not (run / "env_constants.json").exists():
            time.sleep(args.interval); continue
        consts = json.loads((run / "env_constants.json").read_text())   # written by train.py; keeps this process off the GPU
        auto = fm.ROOT / "runs/autopilot.json"
        if auto.exists():   # autopilot state for the status page
            (Path(args.out).parent / "autopilot.json").write_text(auto.read_text())
        ck = newest_checkpoint(run)
        if ck is not None and ck != last:
            try:
                params = checkpoint.load(str(ck))
                spec = export_numpy(params, consts["nu"])
                spec["env"] = consts
                spec["trained"] = True
                off = run / "step_offset.json"
                spec["step"] = int(ck.name) + (json.loads(off.read_text())["offset"] if off.exists() else 0)
                spec["run"] = run.name
                Path(args.out).write_text(json.dumps(spec))
                save_params(params, run / "policy.pkl")
                ev = cpu_eval(spec, int(ck.name))
                with open(run / "cpu_eval.jsonl", "a") as f:
                    f.write(json.dumps(ev) + "\n")
                print(f"cpu eval @ {ck.name}: goals {ev['goals']:.2f} bipedal {ev['bipedal']:.2f} body {ev['body_contact']:.2f}"
                      f" | no assist: goals {ev['goals_noassist']:.2f} bipedal {ev['bipedal_noassist']:.2f}"
                      f" | gait(no assist): flight {ev['flight_noassist']:.2f} alternation {ev['alternation_noassist']:.2f}", flush=True)
                _export_progress(run, Path(args.out).parent / "progress.json")
                last = ck                                    # evaluated: never re-append this checkpoint
                print(f"exported checkpoint {ck.name} -> {args.out}", flush=True)
                if not args.no_clips:
                    try:
                        _render_clip(spec, Path(args.out).parent / "clips", f"{run.name}_{int(ck.name):09d}")
                    except Exception as e:                   # no EGL / ffmpeg: clips are optional
                        print(f"clip skipped: {e}", flush=True)
            except Exception as e:  # checkpoint may still be being written
                print(f"skip {ck.name}: {e}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
