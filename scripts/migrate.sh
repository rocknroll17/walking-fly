#!/usr/bin/env bash
# Pack the newest checkpoint + curriculum state into one file so training can
# continue on another machine:   bash scripts/migrate.sh pack [out.tar.gz]
# On the target machine:          bash scripts/run.sh start --resume bundle.tar.gz
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CMD="${1:-pack}"; OUT="${2:-runs/bundle.tar.gz}"
[[ "$CMD" == "pack" ]] || { echo "usage: bash scripts/migrate.sh pack [out.tar.gz]"; exit 1; }
# newest run dir that has a finished checkpoint
run=""; ck=""
for d in $(ls -td runs/*/ 2>/dev/null); do
  c=$(ls -d "$d"checkpoints/[0-9]* 2>/dev/null | grep -v tmp | sort | tail -1 || true)
  if [[ -n "$c" ]]; then run="${d%/}"; ck="$c"; break; fi
done
[[ -n "$ck" ]] || { echo "no checkpoint found under runs/"; exit 1; }
phase=$(python3 -c "import json,os; p='runs/autopilot.json'; print(json.load(open(p)).get('phase','walk') if os.path.exists(p) else 'walk')")
assist=$(python3 -c "import json; print(json.load(open('$run/env_config.json')).get('assist',0.0))")
offset=$(python3 -c "import json,os; p='$run/step_offset.json'; print(json.load(open(p))['offset'] if os.path.exists(p) else 0)")
step=$(basename "$ck"); total=$((offset + 10#$step))
stage=$(mktemp -d)
mkdir -p "$stage/bundle/checkpoints"
cp -r "$ck" "$stage/bundle/checkpoints/"
cp "$run/env_constants.json" "$run/env_config.json" "$stage/bundle/" 2>/dev/null || true
cat > "$stage/bundle/resume.json" <<JSON
{"source_run": "$(basename "$run")", "checkpoint": "$step", "phase": "$phase", "assist": $assist, "step_offset": $total}
JSON
tar -czf "$OUT" -C "$stage" bundle
rm -rf "$stage"
echo "packed $(basename "$run")/$step (phase=$phase assist=$assist, ~$((total/1000000))M steps) -> $OUT ($(du -h "$OUT" | cut -f1))"
