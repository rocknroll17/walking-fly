#!/usr/bin/env bash
# Start / stop / inspect the three background jobs:
#   trainer  : flybiped.autopilot (curriculum loop, spawns flybiped.train chunks on the GPU)
#   watcher  : flybiped.watch_export (CPU: exports checkpoints for the viewer, evaluates, renders clips)
#   web      : static server for the viewer and status page
# Usage: bash scripts/run.sh start|stop|status|logs|watcher   [--port 8765] [--fresh] [--resume bundle.tar.gz]
#   watcher: restart only the checkpoint watcher (after a git pull), training keeps running
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PORT=8765; FRESH=0; RESUME=""; CMD="${1:-status}"; shift || true
while [[ $# -gt 0 ]]; do case "$1" in --port) PORT="$2"; shift 2;; --fresh) FRESH=1; shift;; --resume) RESUME="$2"; shift 2;; *) shift;; esac; done
export MUJOCO_GL=${MUJOCO_GL:-egl}
mkdir -p runs
pid_of() { pgrep -f "$1" | head -1 || true; }

start_web() {
  if [[ -z "$(pid_of "http.server $PORT")" ]]; then
    (cd web && setsid nohup python3 -m http.server "$PORT" --bind 0.0.0.0 > ../runs/web.log 2>&1 < /dev/null &)
  fi
}
start_watcher() {
  if [[ -z "$(pid_of 'flybiped[.]watch_export')" ]]; then
    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" setsid nohup .venv/bin/python -m flybiped.watch_export --follow runs/ACTIVE --interval 10 > runs/watch_export.log 2>&1 < /dev/null &
  fi
}
start_trainer() {
  [[ -n "$(pid_of 'flybiped[.]autopilot')" ]] && return
  local args=(--chunk "${FLYBIPED_CHUNK:-10e6}" --budget "${FLYBIPED_BUDGET:-inf}")   # runs until stopped by default
  if [[ -n "$RESUME" ]]; then
    # Continue from a bundle made by scripts/migrate.sh on another machine.
    rm -rf runs/imported && mkdir -p runs/imported && tar -xzf "$RESUME" -C runs/imported --strip-components=1
    echo "{\"offset\": $(python3 -c "import json;print(json.load(open('runs/imported/resume.json'))['step_offset'] - int(json.load(open('runs/imported/resume.json'))['checkpoint']))")}" > runs/imported/step_offset.json
    local phase assist off
    phase=$(python3 -c "import json;print(json.load(open('runs/imported/resume.json'))['phase'])")
    assist=$(python3 -c "import json;print(json.load(open('runs/imported/resume.json'))['assist'])")
    off=$(python3 -c "import json;print(json.load(open('runs/imported/resume.json'))['step_offset'])")
    args+=(--start runs/imported --phase "$phase" --assist "$assist" --step_offset "$off")
    echo "resuming from bundle: phase=$phase assist=$assist steps=$off"
    # Publish the bundled policy to the viewer right away (no need to wait for the first new checkpoint).
    local ck; ck=$(ls -d runs/imported/checkpoints/[0-9]* | tail -1)
    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" .venv/bin/python -m flybiped.export --checkpoint "$ck" --constants runs/imported/env_constants.json --step "$off" >/dev/null 2>&1 \
      && echo "viewer policy set to the bundled checkpoint" || echo "note: could not export the bundled checkpoint for the viewer"
  elif [[ $FRESH -eq 0 ]]; then
    local latest; latest=$(ls -td runs/*/checkpoints 2>/dev/null | head -1 | xargs -r dirname || true)
    if [[ -n "$latest" ]]; then
      local off=0; [[ -f "$latest/step_offset.json" ]] && off=$(python3 -c "import json;print(json.load(open('$latest/step_offset.json'))['offset'])")
      args+=(--start "$latest" --step_offset "$off")
      [[ -f runs/autopilot.json ]] && args+=(--phase "$(python3 -c "import json;print(json.load(open('runs/autopilot.json')).get('phase','walk'))")")
    fi
  fi
  setsid nohup .venv/bin/python -m flybiped.autopilot "${args[@]}" > runs/autopilot.log 2>&1 < /dev/null &
}
stop_all() {
  for pat in 'flybiped[.]autopilot' 'flybiped[.]train' 'flybiped[.]watch_export' "http.server $PORT"; do
    pgrep -f "$pat" | xargs -r kill 2>/dev/null || true
  done
}
status() {
  echo "trainer : $([[ -n "$(pid_of 'flybiped[.]autopilot')" ]] && echo running || echo stopped)   (chunk: $(cat runs/ACTIVE 2>/dev/null | xargs -r basename))"
  echo "watcher : $([[ -n "$(pid_of 'flybiped[.]watch_export')" ]] && echo running || echo stopped)"
  echo "web     : $([[ -n "$(pid_of "http.server $PORT")" ]] && echo "http://$(hostname -I 2>/dev/null | awk '{print $1}'):$PORT/" || echo stopped)"
  grep "cpu eval" runs/watch_export.log 2>/dev/null | tail -1 || true
}
case "$CMD" in
  start)  start_web; start_watcher; start_trainer; sleep 1; status ;;
  stop)   stop_all; echo stopped ;;
  status) status ;;
  logs)
    active_log=""; [[ -f runs/ACTIVE ]] && active_log="runs/$(basename "$(cat runs/ACTIVE)").log"
    tail -n 20 runs/autopilot.log $active_log runs/watch_export.log 2>/dev/null ;;
  watcher) pgrep -f 'flybiped[.]watch_export' | xargs -r kill 2>/dev/null; sleep 1; start_watcher; echo "watcher restarted" ;;
  *) echo "usage: bash scripts/run.sh start|stop|status|logs|watcher [--port N] [--fresh] [--resume bundle]"; exit 1 ;;
esac
