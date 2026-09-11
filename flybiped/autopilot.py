"""Unattended training loop with a stand-up curriculum.

Runs training in chunks, reads the evaluation metrics after each chunk and
adapts the environment config:

* stuck (fly never stands on its hind legs)  -> escalate through STAGES
  (stronger stand-up shaping, then an upward assist force on the thorax,
  cf. HoST 2025);
* standing works                              -> decay the assist to zero;
* walking to goals with no assist             -> done.

State is kept in runs/autopilot.json; the active run dir is written to
runs/ACTIVE so watch_export/status page follow along.

Usage: python -m flybiped.autopilot --start runs/v6 [--chunk 15e6] [--budget 400e6]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from flybiped import model as fm

RUNS = fm.ROOT / "runs"
STEP_OFFSET = [0]   # cumulative steps before the current chunk (for display)
ARGS = None         # parsed CLI args (num_envs, eval_every, mem_fraction)
# Skill-first curriculum: learn to WALK from the reared-up pose (with a harness),
# then learn to STAND UP into that skill from the six-leg stance / falls.
WALK_STAGE = {"name": "walk", "override": {"init.biped_start": 1.0, "init.drop_start": 0.0, "assist": 0.5}}
WALK_ASSIST_STEP = 0.1       # harness is reduced by this much whenever walking with it is solid
WALK_ASSIST_MIN = 0.2
WALK_GOALS_SOLID = 2.5       # goals/episode (with harness) considered solid walking
WALK_OK_NOASSIST = 1.5       # goals/episode WITHOUT harness that ends the walk stage
WALK_MAX_CHUNKS = 6
# Fall recovery drilled on its own (Go1-Getup style): every episode starts fallen, no harness, no pushes,
# orientation/height shaping doubled. Ends when the fly rights itself from its back reliably.
GETUP_STAGE = {"name": "getup", "override": {"init.biped_start": 0.0, "init.drop_start": 0.5, "init.flip_start": 0.5,
                                             "assist": 0.0, "push.enable": False,
                                             "reward.orientation": 1.0, "reward.upright": 2.0, "reward.height": 2.0,
                                             "reward.posture": 1.0, "reward.action_rate": -0.001}}
GETUP_OK_FLIP = 0.3      # bipedal fraction from the flipped start (no harness) that ends the drill
GETUP_MAX_CHUNKS = 4
STAGES = [  # stand-up escalation ladder (used after the walk stage); assist starts where the walk stage left it
    {"name": "standup", "override": {"init.biped_start": 0.3, "init.drop_start": 0.25, "init.flip_start": 0.15}},
    {"name": "shape", "override": {"init.biped_start": 0.3, "init.drop_start": 0.25, "init.flip_start": 0.15,
                                   "reward.bipedal": 2.0, "reward.height": 2.0, "reward.orientation": 1.0}},
    {"name": "assist80", "override": {"init.biped_start": 0.4, "init.drop_start": 0.25, "init.flip_start": 0.15, "assist": 0.8,
                                      "reward.bipedal": 2.0, "reward.height": 2.0, "reward.orientation": 1.0}},
]
ASSIST_LADDER = [0.8, 0.5, 0.3, 0.15, 0.0]
STAND_OK = 0.25        # bipedal fraction of the episode that counts as "can stand"
STAND_STUCK = 0.05
GOALS_DONE = 2.0       # goals per 4 s episode, with assist 0 -> success


def last_evals(run: Path, n: int = 3) -> dict:
    """Latest CPU evaluations written by watch_export (waits briefly for the last checkpoint)."""
    f = run / "cpu_eval.jsonl"
    steps = 0
    if (run / "progress.jsonl").exists():
        for line in (run / "progress.jsonl").read_text().splitlines():
            steps = max(steps, json.loads(line)["step"])
    for _ in range(60):   # give the watcher up to ~10 min to evaluate the final checkpoint
        rows = [json.loads(l) for l in f.read_text().splitlines()] if f.exists() else []
        if rows and rows[-1]["step"] >= steps - 1:
            break
        time.sleep(10)
    if not rows:
        return {}
    rows = rows[-n:]
    mean = lambda k: sum(r.get(k, 0.0) for r in rows) / len(rows)  # noqa: E731
    flip = [r.get("modes_noassist", {}).get("flip", {}).get("bipedal", 0.0) for r in rows]
    drop = [r.get("modes_noassist", {}).get("drop", {}).get("bipedal", 0.0) for r in rows]
    return {"bipedal": mean("bipedal"), "goals": mean("goals"), "reward": mean("bipedal") + mean("goals"),
            "goals_noassist": mean("goals_noassist"), "bipedal_noassist": mean("bipedal_noassist"),
            "flight_noassist": mean("flight_noassist"), "alternation_noassist": mean("alternation_noassist"),
            "flip_bipedal": sum(flip) / len(flip), "drop_bipedal": sum(drop) / len(drop), "steps": steps}


def train_chunk(run: Path, restore: Path | None, steps: float, override: dict) -> None:
    run.mkdir(parents=True, exist_ok=True)
    (run / "step_offset.json").write_text(json.dumps({"offset": int(STEP_OFFSET[0])}))
    (RUNS / "ACTIVE").write_text(str(run))
    cmd = [sys.executable, "-m", "flybiped.train", "--run", str(run), "--timesteps", str(steps),
           "--num_envs", "4096", "--eval_every", "5e5", "--override", json.dumps(override)]
    if restore:
        cmd += ["--restore", str(restore)]
    with open(run.parent / f"{run.name}.log", "w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False,
                       env={**__import__("os").environ, "MUJOCO_GL": "egl", "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.45"})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None, help="run dir whose latest checkpoint to continue from")
    ap.add_argument("--chunk", type=float, default=15e6)
    ap.add_argument("--budget", type=float, default=400e6)
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--tag", default=time.strftime("%H%M"), help="prefix so run dirs never collide with earlier launches")
    ap.add_argument("--step_offset", type=float, default=0, help="steps already trained before this launch (display only)")
    ap.add_argument("--assist", type=float, default=0.5, help="harness level to start the walk stage with")
    ap.add_argument("--phase", default="walk", choices=["walk", "standup", "getup"])
    ap.add_argument("--num_envs", type=int, default=int(os.environ.get("FLYBIPED_NUM_ENVS", 4096)))
    ap.add_argument("--eval_every", type=float, default=5e5, help="steps between checkpoints")
    ap.add_argument("--mem_fraction", type=float, default=float(os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", 0.45)))
    args = ap.parse_args()
    global ARGS
    ARGS = args
    STEP_OFFSET[0] = int(args.step_offset)
    state_file = RUNS / "autopilot.json"
    state = {"phase": args.phase, "stage": args.stage, "assist_idx": None, "spent": 0.0, "history": [], "prev": None,
             "walk_assist": args.assist}
    restore = Path(args.start).resolve() if args.start else None
    k = 0
    walk_chunks = 0
    while state["spent"] < args.budget:
        if state["phase"] == "walk":
            stage = WALK_STAGE
        elif state["phase"] == "getup":
            stage = GETUP_STAGE
        else:
            stage = STAGES[min(state["stage"], len(STAGES) - 1)]
        override = dict(stage["override"])
        if state["phase"] == "getup":
            pass                                            # fixed drill settings
        elif state["phase"] == "walk":
            override["assist"] = state["walk_assist"]
        elif state["assist_idx"] is not None:
            override["assist"] = ASSIST_LADDER[state["assist_idx"]]
        else:
            override.setdefault("assist", state["walk_assist"])   # stand-up stage starts at the walk stage's level
        suffix = "" if state["assist_idx"] is None else f"_assist{override['assist']:.2f}"
        run = RUNS / f"auto{args.tag}_{k:02d}_{stage['name']}{suffix}"
        run.mkdir(parents=True, exist_ok=True)
        print(f"== chunk {k}: {run.name} override={override} restore={restore}", flush=True)
        state_file.write_text(json.dumps(state, indent=1))   # phase/assist visible while the chunk runs (migrate.sh)
        train_chunk(run, restore, args.chunk, override)
        ev = last_evals(run)
        if ev:
            STEP_OFFSET[0] += ev["steps"]
        if not ev or ev["steps"] < 0.8 * args.chunk:
            # The trainer died early (e.g. GPU OOM): resume the same chunk from its last checkpoint.
            retries = state.get("retries", 0) + 1; state["retries"] = retries
            print(f"   chunk ended early ({ev.get('steps', 0) if ev else 0} steps); retry {retries}/3", flush=True)
            if retries > 3:
                print("trainer keeps dying; see the chunk log. stopping.", flush=True); break
            restore = run if (run / "checkpoints").exists() and any((run / "checkpoints").iterdir()) else restore
            k += 1
            continue
        state["retries"] = 0
        state["spent"] += args.chunk
        state["history"].append({"run": run.name, "override": override, **ev})
        state_file.write_text(json.dumps(state, indent=1))
        print(f"   result: {ev}", flush=True)
        restore = run
        assist = override.get("assist", 0.0)
        if state["phase"] == "getup":
            getup_chunks = state.get("getup_chunks", 0) + 1; state["getup_chunks"] = getup_chunks
            if ev["flip_bipedal"] >= GETUP_OK_FLIP or getup_chunks >= GETUP_MAX_CHUNKS:
                print(f"get-up drill done after {getup_chunks} chunks (flip->bipedal {ev['flip_bipedal']:.2f}, "
                      f"drop->bipedal {ev['drop_bipedal']:.2f}); back to mixed stand-up at assist 0.15", flush=True)
                state["phase"] = "standup"; state["prev"] = None
                state["assist_idx"] = ASSIST_LADDER.index(0.15)
            k += 1
            continue
        if state["phase"] == "walk":
            walk_chunks += 1
            if ev["goals_noassist"] >= WALK_OK_NOASSIST or walk_chunks >= WALK_MAX_CHUNKS:
                print(f"walk stage done after {walk_chunks} chunks (goals w/o harness {ev['goals_noassist']:.2f}); "
                      f"switching to stand-up at assist {state['walk_assist']:.2f}", flush=True)
                state["phase"] = "standup"; state["prev"] = None
            elif ev["goals"] >= WALK_GOALS_SOLID and state["walk_assist"] > WALK_ASSIST_MIN + 1e-6:
                state["walk_assist"] = round(max(WALK_ASSIST_MIN, state["walk_assist"] - WALK_ASSIST_STEP), 2)
                print(f"walking is solid with harness; lowering harness to {state['walk_assist']:.2f}", flush=True)
                state["prev"] = ev
            else:
                state["prev"] = ev
            k += 1
            continue
        if (assist == 0.0 and ev["goals"] >= GOALS_DONE
                and ev["flight_noassist"] <= 0.10 and ev["alternation_noassist"] >= 0.70):
            print("SUCCESS: alternating bipedal walking to goals without assist", flush=True); break
        if ev["bipedal"] >= STAND_OK:
            # Standing works: decay the assist (or keep going if already 0).
            if assist > 0.0:
                lower = [i for i, a in enumerate(ASSIST_LADDER) if a < assist - 1e-9]
                state["assist_idx"] = lower[0] if lower else len(ASSIST_LADDER) - 1
        elif ev["bipedal"] < STAND_STUCK and state["prev"] is not None and ev["reward"] <= state["prev"]["reward"] * 1.05:
            state["stage"] = min(state["stage"] + 1, len(STAGES) - 1)   # stuck: escalate
            state["assist_idx"] = None
        elif ev["bipedal"] < STAND_STUCK and state["prev"] is None:
            pass  # first chunk: give the current stage one more chunk
        state["prev"] = ev
        k += 1
    print("autopilot finished", flush=True)


if __name__ == "__main__":
    main()
