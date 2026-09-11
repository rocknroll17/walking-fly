#!/usr/bin/env bash
# One-shot setup: checks the machine, creates the Python env, fetches the fly model,
# builds the bipedal model/poses/web assets. Safe to re-run.
#   bash scripts/setup.sh            # full setup (needs an NVIDIA GPU for training)
#   bash scripts/setup.sh --viewer   # viewer-only setup (no GPU needed)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
VIEWER_ONLY=0; [[ "${1:-}" == "--viewer" ]] && VIEWER_ONLY=1
FLYBODY_COMMIT=d015e9bfe441bd90ae431bac24c55cb74bdbce26

say() { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }
die() { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

say "checking the machine"
[[ "$(uname -s)" == "Linux" ]] || die "Linux is required (MuJoCo Warp / EGL)."
command -v git >/dev/null || die "git is required."
command -v curl >/dev/null || die "curl is required."
if [[ $VIEWER_ONLY -eq 0 ]]; then
  command -v nvidia-smi >/dev/null || die "nvidia-smi not found: training needs an NVIDIA GPU (run with --viewer for the viewer only)."
  cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
  drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  echo "GPU compute capability $cap, driver $drv"
  awk -v c="$cap" 'BEGIN{ if (c+0 < 7.0) { print "compute capability < 7.0: MuJoCo Warp needs a Volta or newer GPU"; exit 1 } }' || die "unsupported GPU"
  awk -v d="$drv" 'BEGIN{ split(d,a,"."); if (a[1]+0 < 525) { exit 1 } }' || die "NVIDIA driver >= 525 is required for the CUDA 12 JAX wheels (found $drv)."
fi

say "python environment (uv + Python 3.12)"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
[[ -n "${LD_LIBRARY_PATH:-}" ]] && echo "note: LD_LIBRARY_PATH is set; the JAX CUDA wheels ship their own CUDA libraries, unset it if JAX fails to see the GPU"
uv sync --locked --python 3.12 --no-dev   # exact versions from uv.lock (the CUDA wheels install on any Linux x86-64; JAX falls back to CPU without a GPU)

say "fruit fly model (flybody, pinned commit)"
if [[ ! -d ext/flybody ]]; then
  git clone -q https://github.com/TuragaLab/flybody.git ext/flybody
fi
git -C ext/flybody checkout -q "$FLYBODY_COMMIT"

say "building the bipedal model, standing poses and web meshes"
export MUJOCO_GL=${MUJOCO_GL:-egl}
.venv/bin/python -m flybiped.model
.venv/bin/python -m flybiped.pose
.venv/bin/python -m flybiped.web_assets

say "browser viewer dependencies (node)"
if command -v npm >/dev/null; then
  (cd web && npm install --silent --no-audit --no-fund)
else
  echo "npm not found: skipping viewer install (install Node.js >= 18 and re-run to enable the viewer)"
fi

say "exporting an untrained policy so the viewer works before training"
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" .venv/bin/python -m flybiped.export
say "done. next: bash scripts/run.sh start"
