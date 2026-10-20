#!/usr/bin/env bash
# GraspGen-X ZMQ grasp server for wrc_demo (set grasp.backend: graspgenx).
#
#   ./scripts/serve_graspgenx.sh [gripper] [port]
#
# Model + deps live in their own directory (defaults to ~/graspgenx-venv for
# the Python interpreter and the GraspGenX source tree; override with
# GRASPGENX_ROOT). GraspGenX pins torch independently from wrc_demo, so it
# lives in its own venv. Checkpoints auto-download on first import.
#
# Args:
#   $1 gripper description (default: franka_panda; reBot RS uses the
#      swept-volume defined in configs/demo.yaml grasp.graspgenx.sweep)
#   $2 port (default: 5556)
#
# Environment:
#   GRASPGENX_ROOT          root dir for both the venv and the GraspGenX
#                            source tree (default: $HOME/graspgenx-root)
#   GRASPGENX_VENV          Python venv with GraspGenX installed
#                            (default: $GRASPGENX_ROOT/.venv/bin/python)
#   GRASPGENX_SRC           path to the GraspGenX source checkout
#                            (default: $GRASPGENX_ROOT/GraspGenX)
set -euo pipefail

# Determine repo-relative paths for the defaults. ROOT here is the directory
# one level up from scripts/ — i.e. the wrc_demo checkout. GraspGenX lives
# alongside it (default), so we look for it as a sibling.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default GRASPGENX_ROOT to a sibling of wrc_demo (so they share checkpoints
# and asset directories); override with the env var to put it elsewhere.
GRASPGENX_ROOT="${GRASPGENX_ROOT:-$(dirname "$REPO_DIR")/graspgenx-root}"
GRASPGENX_VENV="${GRASPGENX_VENV:-$GRASPGENX_ROOT/.venv/bin/python}"
GRASPGENX_SRC="${GRASPGENX_SRC:-$GRASPGENX_ROOT/GraspGenX}"

export GRASPGENX_CHECKPOINT_DIR="$GRASPGENX_SRC/ext/graspgenx_checkpoints"

if [[ ! -x "$GRASPGENX_VENV" ]]; then
    echo "" >&2
    echo "ERROR: GraspGenX venv not found at $GRASPGENX_VENV" >&2
    echo "" >&2
    echo "One-time setup:" >&2
    echo "    git clone https://github.com/NVlabs/GraspGenX.git $GRASPGENX_SRC" >&2
    echo "    uv venv $GRASPGENX_ROOT/.venv --python 3.10" >&2
    echo "    source $GRASPGENX_ROOT/.venv/bin/activate" >&2
    echo "    uv pip install -e $GRASPGENX_SRC" >&2
    echo "" >&2
    echo "Or point at an existing install via GRASPGENX_VENV." >&2
    exit 1
fi

if [[ ! -f "$GRASPGENX_SRC/client-server/graspgenx_server.py" ]]; then
    echo "ERROR: GraspGenX source not found at $GRASPGENX_SRC" >&2
    echo "  Set GRASPGENX_SRC=/path/to/GraspGenX" >&2
    exit 1
fi

exec "$GRASPGENX_VENV" \
    "$GRASPGENX_SRC/client-server/graspgenx_server.py" \
    --config "$GRASPGENX_CHECKPOINT_DIR/release/gen/config.yaml" \
    --assets_dir "$GRASPGENX_SRC/ext/gripper_descriptions" \
    --default_gripper "${1:-franka_panda}" \
    --port "${2:-5556}"