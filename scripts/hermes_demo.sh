#!/usr/bin/env bash
# One-shot Hermes demo: register the wrc-demo MCP server, verify the
# connection, and drop into a Hermes chat where you can talk to the arm.
#
#   ./scripts/hermes_demo.sh                              # default model
#   ./scripts/hermes_demo.sh --model anthropic/claude-sonnet-4-5
#   ./scripts/hermes_demo.sh --arm rebot_rs               # real arm (onsite!)
#
# The BRAIN is whatever vision-capable model Hermes is pointed at:
#   - Claude:      needs ANTHROPIC_API_KEY known to Hermes (`hermes model`),
#                  then --model anthropic/<claude model>
#   - GPT (Codex): already logged in on this rig (gpt-5.6-sol) - just works
#   - local VLM:   serve Qwen3-VL (scripts/start_cosmos3_server.sh --bg),
#                  define a custom provider in ~/.hermes/config.yaml
#                  `providers:` pointing at http://127.0.0.1:8080/v1, then
#                  --provider it
#
# Try saying:
#   "take a camera snapshot and tell me what you see"
#   "what objects are on the table?"
#   "grasp the red cube and place it on the plate"
set -euo pipefail

CAMERA="l515"
ARM="mock"
MODEL=""
PROVIDER=""
REPO="$(cd "$(dirname "$0")/.." && pwd)"

# Pick a Python interpreter. Prefer $WRC_DEMO_PY; fall back to the conda
# `wrc-demo` env (the project's standard env); fall back to whatever
# `python` is on PATH. Mirrors the same fallback chain as start_all.sh.
WRC_DEMO_PY="${WRC_DEMO_PY:-${CONDA_PREFIX:-}/bin/python}"
if [[ -z "${CONDA_PREFIX:-}" ]] && [[ ! -x "$WRC_DEMO_PY" ]]; then
    WRC_DEMO_PY="$(command -v python)"
fi
if [[ -z "$WRC_DEMO_PY" ]] || [[ ! -x "$WRC_DEMO_PY" ]]; then
    echo "[!] No Python interpreter found. Activate the conda 'wrc-demo' env" >&2
    echo "    (\`conda activate wrc-demo\`) or set WRC_DEMO_PY=/path/to/python" >&2
    exit 1
fi
PY="$WRC_DEMO_PY"

# Default detector: prefer an open-vocab YOLOE checkpoint in models/ if
# present, else fall back to closed-set yolo11n.pt (ultralytics auto-
# downloads on first use).
DETECTOR="${WRC_DETECTOR_MODEL:-$REPO/models/yolo11n.pt}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --camera) CAMERA="$2"; shift 2 ;;
        --arm) ARM="$2"; shift 2 ;;
        --detector) DETECTOR="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --provider) PROVIDER="$2"; shift 2 ;;
        *) echo "unknown flag $1"; exit 1 ;;
    esac
done

# If the ultralytics CLIP fork is installed, prefer the open-vocabulary
# YOLOE model (free-text vocabulary); otherwise stay on closed-set yolo11n.
if "$PY" -c "import clip" 2>/dev/null; then
    DETECTOR="${WRC_DETECTOR_MODEL:-$REPO/models/yoloe-11s-seg.pt}"
    echo "[+] CLIP available: using open-vocabulary detector $DETECTOR"
else
    echo "[i] CLIP fork not installed: using closed-set COCO detector."
    echo "    For YOUR OWN vocabulary (any object by name), run once:"
    echo "    pip install --python $PY 'git+https://github.com/ultralytics/CLIP.git'"
fi

echo "[+] registering MCP server (camera=$CAMERA arm=$ARM)"
hermes mcp remove wrc-demo >/dev/null 2>&1 || true
hermes mcp add wrc-demo \
    --command "$PY" \
    --env "PYTHONPATH=$REPO/src" "WRC_CAMERAS=$CAMERA" "WRC_ARM=$ARM" \
          "WRC_DETECTOR_MODEL=$DETECTOR" "DISPLAY=${DISPLAY:-:1}" \
    --args -m wrc_demo.apps.mcp_server

echo "[+] testing the connection"
hermes mcp test wrc-demo

CHAT_ARGS=()
[[ -n "$MODEL" ]] && CHAT_ARGS+=(-m "$MODEL")
[[ -n "$PROVIDER" ]] && CHAT_ARGS+=(--provider "$PROVIDER")

cat <<'EOF'
[+] opening interactive chat (ctrl+d to exit). Ask things like:
      take a camera snapshot and tell me what you see
      what objects are on the table?
      localize the bottle
      grasp the red cube
EOF
exec hermes chat "${CHAT_ARGS[@]}"
