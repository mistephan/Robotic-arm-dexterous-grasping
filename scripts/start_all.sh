#!/usr/bin/env bash
# One-shot bring-up: vLLM + wrc_demo (interactive).
#
# Usage:
#   bash scripts/start_all.sh                            # mock stack, no sudo
#   bash scripts/start_all.sh --real                     # real hardware (needs sudo)
#   bash scripts/start_all.sh --real --task "wave"       # real + single task
#   bash scripts/start_all.sh --dashboard                # mock + web dashboard on :8090
#   bash scripts/start_all.sh --dashboard --real         # real + web dashboard (order-agnostic)
#   bash scripts/start_all.sh --repl                     # foreground stdin REPL (mock stack)
#   bash scripts/start_all.sh --repl --real              # foreground stdin REPL (real hardware)
#   bash scripts/start_all.sh --stop                     # stop everything
#   bash scripts/start_all.sh --status                   # check status
#
# Architecture (ADR-0007, updated 2026-08-14 to use Qwen3-VL):
#   cosmos3-venv (uv)        wrc_demo (DM env)
#   ┌──────────────┐         ┌──────────────┐
#   │   vLLM 0.27  │  HTTP   │  wrc_demo    │
#   │  Qwen3-VL-2B │◄────────┤  OpenAICompat│
#   │   AWQ-INT4   │ :8080  │  Client      │
#   └──────────────┘         └──────────────┘
#
# vLLM lives in its own uv venv because cu13 + transformers v5 + vllm
# 0.27 are incompatible with the wrc_demo DM env (cu12, older stacks).
# They talk via the OpenAI HTTP API on localhost:8080.
#
# Default profile is `local_qwen3_vl` (Qwen3-VL-2B-Instruct AWQ-4bit,
# AWQ INT4 fits on 8 GB and natively supports tool calling). Override
# with `--llm <profile>` after editing the DEMO_ARGS line below.
#
# See ADR-0007 / docs/SPEC-local-llm.md for details.

set -euo pipefail

# Resolve the vLLM script relative to this script's location so the launcher
# works regardless of where the repo is checked out. Override with WRC_VLLM_SCRIPT.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VLLM_SCRIPT="${WRC_VLLM_SCRIPT:-$SCRIPT_DIR/start_cosmos3_server.sh}"
VLLM_PID_FILE="/tmp/vllm_cosmos3.pid"
VLLM_LOG="/tmp/vllm_cosmos3.log"
DEMO_PID_FILE="/tmp/wrc_demo.pid"
DEMO_LOG="/tmp/wrc_demo.log"

# ── Subcommand dispatch ───────────────────────────────────────────
case "${1:-start}" in
    --stop|stop)
        # Stop demo first, then vllm
        if [[ -f "$DEMO_PID_FILE" ]]; then
            pid=$(cat "$DEMO_PID_FILE")
            if kill -0 "$pid" 2>/dev/null; then
                echo "[stop] killing wrc_demo PID $pid"
                kill "$pid" 2>/dev/null || true
                sleep 1
                kill -9 "$pid" 2>/dev/null || true
            fi
            rm -f "$DEMO_PID_FILE"
        fi
        pkill -f 'wrc_demo.apps.demo' 2>/dev/null || true
        pkill -f 'wrc_demo.apps.mcp' 2>/dev/null || true
        if [[ -x "$VLLM_SCRIPT" ]]; then
            bash "$VLLM_SCRIPT" --stop 2>&1 | head -3
        fi
        echo "[stop] done"
        exit 0
        ;;
    --status|status)
        echo "=== vLLM (cosmos3-llm @ :8080) ==="
        if [[ -f "$VLLM_PID_FILE" ]] && kill -0 "$(cat "$VLLM_PID_FILE")" 2>/dev/null; then
            echo "  running, PID $(cat "$VLLM_PID_FILE")"
            curl -sf http://localhost:8080/v1/models 2>&1 | head -c 200
            echo
        else
            echo "  NOT running"
        fi
        echo ""
        echo "=== wrc_demo ==="
        if [[ -f "$DEMO_PID_FILE" ]] && kill -0 "$(cat "$DEMO_PID_FILE")" 2>/dev/null; then
            echo "  running, PID $(cat "$DEMO_PID_FILE")"
        else
            echo "  NOT running"
        fi
        exit 0
        ;;
esac

# ── Start mode ─────────────────────────────────────────────────────
MODE="${1:-mock}"   # --real, --dashboard, mock, or first arg = task string
shift || true
TASK=""
NEED_SUDO=0
CAMERAS="mock"
ARM="mock"
LAUNCH_DASHBOARD=0
REPL_MODE=""

case "$MODE" in
    --real|real)
        NEED_SUDO=1
        CAMERAS="orbbec_overhead"
        ARM="rebot_rs_o6"
        # Accept "--real --dashboard" or "--real --repl" in either order.
        if [[ "${1:-}" == "--dashboard" ]]; then
            LAUNCH_DASHBOARD=1
            shift
        elif [[ "${1:-}" == "--repl" ]]; then
            REPL_MODE=1
            shift
        fi
        ;;
    --dashboard|dashboard)
        # Long-running kiosk mode: stay alive serving :8090 MJPEG +
        # web "send" button.  Use --real --dashboard together for real
        # hardware; without --real you get the mock stack.
        LAUNCH_DASHBOARD=1
        if [[ "${1:-}" == "--real" ]]; then
            NEED_SUDO=1
            CAMERAS="orbbec_overhead"
            ARM="rebot_rs_o6"
            shift
        fi
        ;;
    --repl|repl)
        # Foreground REPL: stdin is the real terminal (no nohup), so the
        # user can type tasks directly at the `task>` prompt.  Mimics the
        # legacy wrc_demo direct-invocation UX:
        #   sudo env ... python -u -m wrc_demo.apps.demo ... --interactive
        # Without --real you get the mock stack (no sudo needed).
        REPL_MODE=1
        # Accept "--repl --real" and "--repl --dashboard" in either order
        # (mutual exclusion for --dashboard is checked below).
        if [[ "${1:-}" == "--real" ]]; then
            NEED_SUDO=1
            CAMERAS="orbbec_overhead"
            ARM="rebot_rs_o6"
            shift
        elif [[ "${1:-}" == "--dashboard" ]]; then
            LAUNCH_DASHBOARD=1
            shift
        fi
        ;;
    *)
        # Treat $MODE as the task string (one-shot mode)
        TASK="$MODE"
        ;;
esac

# Mutual exclusion: --repl is foreground (stdin REPL); --dashboard is
# background (web kiosk). They have different exit semantics — pick one.
if [[ -n "${REPL_MODE:-}" && $LAUNCH_DASHBOARD -eq 1 ]]; then
    echo "  ERROR: --repl and --dashboard are mutually exclusive."
    echo "         (foreground stdin REPL vs background web kiosk)"
    exit 1
fi

# Optional positional --task
while [[ $# -gt 0 ]]; do
    case "$1" in
        --task) TASK="$2"; shift 2 ;;
        *) shift ;;
    esac
done

# --task is incompatible with --repl (REPL is interactive; --task is one-shot)
if [[ -n "${REPL_MODE:-}" && -n "$TASK" ]]; then
    echo "  ERROR: --repl and --task are mutually exclusive."
    echo "         (interactive stdin vs one-shot command)"
    exit 1
fi

# ── Step 0: ensure YOLOE text-encoder symlinks in CWD ─────────────────
# ultralytics loads `mobileclip*.ts` from the current working directory
# (NOT from models/). The launchers (start_all.sh, run.sh) all run
# with CWD = the project root, so we mirror models/*.ts into CWD here
# so detection works without a manual symlink step.
#
# Idempotent: skips if symlink already correct, repairs broken links.
for _ts in mobileclip2_b.ts mobileclip_blt.ts; do
    if [[ -L "$_ts" && ! -e "$_ts" ]]; then
        # Broken symlink (target missing). Remove so ultralytics
        # doesn't waste a download attempt on a path it can't reach.
        rm -f "$_ts"
    fi
    if [[ ! -e "$_ts" && -e "models/$_ts" ]]; then
        ln -sf "models/$_ts" "$_ts"
        echo "[setup] symlinked models/$_ts -> ./$_ts (ultralytics CWD lookup)"
    fi
done

echo "============================================================"
echo "  wrc_demo + Qwen3-VL-AWQ-4bit one-shot launcher"
echo "============================================================"
echo "  mode:    $([[ $NEED_SUDO -eq 1 ]] && echo 'real hardware' || echo 'mock stack')"
echo "  camera:  $CAMERAS"
echo "  arm:     $ARM"
echo "  task:    ${TASK:-<interactive>}"
echo ""

# ── Step 1: ensure vLLM is running ───────────────────────────────
echo "[1/3] check vLLM..."
if curl -sf --max-time 3 http://localhost:8080/v1/models > /dev/null 2>&1; then
    echo "  vLLM already running (skipped)"
else
    echo "  vLLM not running, starting..."
    # The legacy script name "start_cosmos3_server.sh" is preserved because
    # operation docs and many user scripts reference it; it now serves the
    # Qwen3-VL AWQ model (see ADR-0007 + §17/§18 of wrc_demo_操作文档.md).
    MAX_LEN=8192 GPU_UTIL=0.85 bash "$VLLM_SCRIPT" --bg
    # Wait for readiness
    for i in {1..40}; do
        sleep 3
        if curl -sf --max-time 3 http://localhost:8080/v1/models > /dev/null 2>&1; then
            echo "  vLLM ready (after ${i}*3s)"
            break
        fi
        if [[ $i -eq 40 ]]; then
            echo "  vLLM failed to start in 120s; see $VLLM_LOG"
            exit 1
        fi
    done
fi

# ── Step 2: launch wrc_demo ─────────────────────────────────────
echo ""
echo "[2/3] start wrc_demo..."
RUN_DIR="${WRC_RUN_DIR:-$REPO_DIR/runs/live_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_DIR"

# Pick a Python interpreter. Prefer $WRC_DEMO_PY; otherwise use the
# conda `wrc-demo` env's python (the project's standard).  Do NOT
# fall back to `command -v python` (PATH lookup) — that may resolve
# to a user-installed `~/.local/bin/python` whose site-packages
# shadow the conda env, breaking torch/ultralytics imports.
WRC_DEMO_PY="${WRC_DEMO_PY:-/home/seeed/miniforge3/envs/wrc-demo/bin/python}"
if [[ ! -x "$WRC_DEMO_PY" ]]; then
    echo "  ERROR: $WRC_DEMO_PY not found. Set WRC_DEMO_PY or fix the hardcoded path."
    exit 1
fi


# Build env-passthrough list for LLM API keys. The `env VAR=val cmd`
# form below resets all environment variables; anything needed by
# the subprocess must be listed explicitly. These match the `env_key:`
# fields in configs/llm/*.yaml (see ADR-0006).
_API_KEY_ARGS=()
[[ -n "${MINIMAX_API_KEY:-}"   ]] && _API_KEY_ARGS+=(MINIMAX_API_KEY="$MINIMAX_API_KEY")
[[ -n "${ANTHROPIC_API_KEY:-}" ]] && _API_KEY_ARGS+=(ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY")
[[ -n "${OPENAI_API_KEY:-}"    ]] && _API_KEY_ARGS+=(OPENAI_API_KEY="$OPENAI_API_KEY")

# Default LLM profile is local_qwen3_vl (current production model — Qwen3-VL-2B
# AWQ-4bit served by start_cosmos3_server.sh). Override with --llm <profile>.
# Build demo args as a bash array (preserves quoting of $TASK).
# LLM profile resolution (priority high→low, see _resolve_llm_profile):
#   1. `WRC_LLM` env var → pass as --llm (overrides everything)
#   2. NOT passing --llm → falls through to configs/demo.yaml
#   3. demo.yaml `llm.profile` (lowest priority)
# This way the user can edit demo.yaml once and have it take effect.
if [[ $LAUNCH_DASHBOARD -eq 1 ]]; then
    BOOT_TASK="${TASK:-${WRC_DASHBOARD_TASK:-pick and place pink object}}"
    DEMO_ARGS=(scripts/dashboard_runner.py \
        --cameras "$CAMERAS" --arm "$ARM" \
        --task "$BOOT_TASK")
elif [[ -n "${REPL_MODE:-}" ]]; then
    # Foreground REPL: --interactive reads tasks from stdin (real terminal).
    # No --no-serve: the MJPEG stream dashboard stays up on :8090 so the
    # user can also watch the live view while typing.
    DEMO_ARGS=(-m wrc_demo.apps.demo \
        --cameras "$CAMERAS" --arm "$ARM" --no-view --interactive)
else
    # Non-dashboard, non-REPL: run wrc_demo.apps.demo as a module. The
    # script name moved out of the nohup line (session 2 had dropped it)
    # and into DEMO_ARGS so the same nohup form works for both modules
    # and script-file invocations.
    DEMO_ARGS=(-m wrc_demo.apps.demo \
        --cameras "$CAMERAS" --arm "$ARM" --no-view)
    if [[ -n "$TASK" ]]; then
        DEMO_ARGS+=(--no-serve --task "$TASK")
    else
        # nohup redirects stdin to /dev/null (GNU nohup default), so
        # --interactive would read EOF and exit immediately. Without
        # --dashboard (web UI), --repl (foreground REPL), or --task
        # (one-shot), there's nothing to drive the demo and it silently
        # dies right after "✓ READY" prints — confusing because the
        # launcher already exited. Bail loudly.
        echo "  ERROR: non-interactive mode needs --task \"<cmd>\","
        echo "         --dashboard (web UI), or --repl (foreground REPL)."
        echo "  nohup'd --interactive would read EOF on /dev/null and exit."
        exit 1
    fi
fi
# Only pass --llm if WRC_LLM is set; otherwise let demo.yaml decide.
if [[ -n "${WRC_LLM:-}" ]]; then
    DEMO_ARGS+=(--llm "$WRC_LLM")
fi

# === Build env-prefix array (sudo or no-sudo) =========================
# This used to be inlined into the nohup line and duplicated for sudo vs
# no-sudo; factored out so the REPL path can reuse it via exec().
if [[ $NEED_SUDO -eq 1 ]]; then
    # Real hardware: needs sudo for CAN bus + WRC_STREAM=1, WRC_VIEW=1.
    # `nohup sudo`/`exec sudo` can't prompt for a password (no tty). Require
    # the operator to have cached sudo credentials first via `sudo -v` in an
    # interactive terminal; we verify non-interactively with `sudo -n true`.
    if ! sudo -n true 2>/dev/null; then
        echo ""
        echo "  ERROR: --real needs sudo (CAN bus, etc.) but no cached credentials."
        echo "  Run this in an interactive terminal first to cache sudo:"
        echo ""
        echo "      sudo -v"
        echo ""
        echo "  Then re-run: bash $0 --real ${TASK:+--task \"$TASK\"}"
        exit 1
    fi
    # LD_LIBRARY_PATH must be passed explicitly: `sudo` (and `nohup`) reset
    # it for security, and the conda activate.d hook doesn't run under sudo.
    _LD_PATHS=""
    for _sub in cu13 cuda_runtime cudnn cublas cufft cusparse curand cusolver \
                cudnn_frontend nvjitlink; do
        _p="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/$_sub/lib"
        [[ -d "$_p" ]] && _LD_PATHS="${_LD_PATHS}:${_p}"
    done
    export LD_LIBRARY_PATH="${_LD_PATHS}${LD_LIBRARY_PATH:-}"

    _SUFFIX_PRE=(sudo env HOME="$HOME" PYTHONNOUSERSITE=1 \
        "${_API_KEY_ARGS[@]}" \
        YOLO_OFFLINE=True \
        PYTHONPATH="$REPO_DIR/src" \
        LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
        WRC_RUN_DIR="$RUN_DIR" WRC_STREAM=1 WRC_VIEW=1)
else
    _SUFFIX_PRE=(env HOME="$HOME" PYTHONNOUSERSITE=1 \
        "${_API_KEY_ARGS[@]}" \
        YOLO_OFFLINE=True \
        PYTHONPATH="$REPO_DIR/src" \
        WRC_RUN_DIR="$RUN_DIR")
fi

# === Foreground REPL: exec, no redirection, no & ========================
# stdin/stdout/stderr stay attached to the user's terminal; signals
# (Ctrl+C) propagate directly to python. Replaces the shell so when
# python exits, the script exits — no orphan launcher.
if [[ -n "${REPL_MODE:-}" ]]; then
    if [[ ! -t 0 ]]; then
        echo "  ERROR: --repl requires a real terminal (stdin is not a tty)."
        echo "         Use --dashboard for headless operation."
        exit 1
    fi
    exec "${_SUFFIX_PRE[@]}" "$WRC_DEMO_PY" -s -u "${DEMO_ARGS[@]}"
    # exec never returns.
fi

# === Background nohup ====================================================
nohup "${_SUFFIX_PRE[@]}" "$WRC_DEMO_PY" -s -u "${DEMO_ARGS[@]}" > "$DEMO_LOG" 2>&1 &

DEMO_PID=$!
echo $DEMO_PID > "$DEMO_PID_FILE"
echo "  wrc_demo PID: $DEMO_PID"
echo "  log:          $DEMO_LOG"
echo "  run dir:      $RUN_DIR"

# ── Step 3: wait + summary ───────────────────────────────────────
echo ""
echo "[3/3] wait for demo to start..."
for i in {1..30}; do
    sleep 2
    if ! kill -0 $DEMO_PID 2>/dev/null; then
        echo "  ERROR: wrc_demo died early. Last 30 lines of log:"
        tail -30 "$DEMO_LOG" 2>&1
        exit 1
    fi
    # Check log for "task>" or completed task report
    if grep -q "task> \|task report\|success: True\|step budget" "$DEMO_LOG" 2>/dev/null; then
        break
    fi
done

echo ""
echo "============================================================"
echo "  ✓ READY"
echo "============================================================"
echo "  wrc_demo:    http://localhost:8090/  (dashboard, if --no-view not set)"
echo "  vLLM:        http://localhost:8080/v1  (Qwen3-VL-2B-AWQ-4bit via start_cosmos3_server.sh)"
echo "  log:         $DEMO_LOG"
echo "  tail:        tail -f $DEMO_LOG"
echo "  status:      bash $0 --status"
echo "  stop:        bash $0 --stop"
echo "============================================================"
