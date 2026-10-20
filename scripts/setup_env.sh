#!/usr/bin/env bash
# Set up the wrc_demo development environment.
#
# This script complements `environment.yml`: the conda route is the
# recommended path (see environment.yml + README), but this script gives
# you a uv-managed venv at ./.venv for users who prefer uv or can't
# install conda (e.g. on a strict CI runner).
#
# Usage:
#   bash scripts/setup_env.sh                # create .venv, install wrc-demo[dev]
#   bash scripts/setup_env.sh --all          # create .venv + install ALL extras
#   bash scripts/setup_env.sh --info         # show what would be installed
#
# This is a *fallback* / convenience wrapper — the primary install path
# is `conda env create -f environment.yml && conda activate wrc-demo`
# (the conda env gets every hardware SDK already, see environment.yml).
#
# The .venv created here is a mock-friendly environment: it has pytest
# + wrc-demo[dev] but NOT ultralytics / motorbridge / pin / pyorbbecsdk.
# Use it for:
#   - mock tests (`pytest tests/`)
#   - linting / type-checking
#   - local development without GPU
#
# For real-hardware runs use the conda `wrc-demo` env instead.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"
PROFILE="dev"   # default: dev only

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)
            PROFILE="perception,llm,kinematics,arm,dev"
            shift
            ;;
        --info)
            echo "venv:    $VENV_DIR"
            echo "profile: $PROFILE"
            echo "extras:  -e .[$PROFILE]"
            exit 0
            ;;
        *)
            echo "unknown flag $1" >&2
            exit 1
            ;;
    esac
done

# Pick a venv manager: prefer uv (fast), fall back to python -m venv + pip.
if command -v uv >/dev/null 2>&1; then
    echo "[setup] using uv at $(command -v uv)"
    if [[ ! -d "$VENV_DIR" ]]; then
        echo "[setup] creating venv at $VENV_DIR"
        uv venv "$VENV_DIR" --python 3.10
    fi
    echo "[setup] installing wrc-demo[$PROFILE] editable"
    VIRTUAL_ENV="$VENV_DIR" uv pip install -e "$REPO_DIR[$PROFILE]"
else
    echo "[setup] uv not found; falling back to 'python -m venv' + pip"
    if [[ ! -d "$VENV_DIR" ]]; then
        echo "[setup] creating venv at $VENV_DIR"
        python3 -m venv "$VENV_DIR"
    fi
    echo "[setup] installing wrc-demo[$PROFILE] editable"
    "$VENV_DIR/bin/pip" install -e "$REPO_DIR[$PROFILE]"
fi

echo ""
echo "Done. done.  Activate with:"
echo "    source $VENV_DIR/bin/activate"
echo "Then run the mock tests:"
echo "    pytest tests/test_extrinsics_loader.py tests/test_hand_eye_compensation_per_camera.py tests/test_demo_yaml_llm_profile.py -v"
echo ""
echo "Note: this venv is mock-friendly only. For real hardware (Orbbec /"
echo "reBot Arm / RealSense), use the conda 'wrc-demo' env from"
echo "environment.yml — that one has motorbridge / pyorbbecsdk / pin."