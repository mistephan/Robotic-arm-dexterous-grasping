"""MCP stdio server: expose the arm's skill runtime to any MCP agent.

This turns the demo into a tool surface for agent platforms -- Hermes
(~/.hermes/config.yaml), Claude Code (.mcp.json), Claude Desktop, Codex CLI.
The platform's agent replaces wrc_demo's built-in orchestrator as the brain;
the safety harness, tracing, episodic memory and skills are identical.

Protocol: MCP over stdio, newline-delimited JSON-RPC 2.0. Implemented by
hand (initialize / tools/list / tools/call / ping) so the demo gains no new
dependency. stdout carries ONLY protocol frames; everything else goes to
stderr.

Configuration via environment (set in the MCP server entry):
    WRC_CAMERA          camera profile (default: mock)
    WRC_CAMERAS         comma-separated camera profiles; first one is the
                        manipulation camera (overrides WRC_CAMERA)
    WRC_ARM             arm profile    (default: mock)
    WRC_RUN_DIR         trace directory (default: <repo>/runs/mcp_<pid>)
    WRC_DETECTOR_MODEL  override detector weights (e.g. a yolo11n.pt path
                        for closed-set COCO until the CLIP fork is installed)
    WRC_DETECT_CLASSES  comma-separated default vocabulary for observations
    WRC_VIEW            "0" disables the live camera window (default: open it
                        whenever DISPLAY is set, so the audience always sees
                        what the camera sees)
    WRC_PREWARM         "0" disables perception pre-warm at startup
                        (default: cameras + detector + world model come up
                        immediately so the first command is fast)
    WRC_STREAM          "0" disables the MJPEG livestream dashboard
    WRC_STREAM_PORT     dashboard port (default: from configs/demo.yaml)

Latency contract (why this server is fast): perception pre-warms in the
background the moment the gateway starts -- N camera streams, the detector,
and the WorldWatcher that keeps the belief store hot. The ARM stays
unpowered until the first motion tool call (LazyArm). A routine command like
"pick and place pink object" is ONE pick_and_place tool call that resolves
against the warm world model and runs deterministically: no LLM round-trips
inside the loop, total time ~ arm motion time.
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
import sys
import threading
import time
from pathlib import Path

PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
SERVER_INFO = {"name": "wrc-demo", "version": "0.1.0"}

# task_done belongs to the built-in loop; MCP agents manage their own tasks.
_EXCLUDED_TOOLS = {"task_done"}

_EXTRA_TOOLS = [
    {
        "name": "camera_snapshot",
        "description": (
            "Capture a camera frame and return it as an image, with the "
            "depth source noted. Use this to SEE the workspace. Optional "
            "`camera` selects one of the rig cameras (see world_state)."
        ),
        "parameters": {
            "type": "object",
            "properties": {"camera": {"type": "string"}},
            "required": [],
        },
    },
    {
        "name": "world_state",
        "description": (
            "INSTANT text snapshot of the live world model: every object "
            "with color + 3D position + freshness, what the gripper holds, "
            "camera FPS, and the livestream URL. Perception runs "
            "continuously, so prefer this over camera_snapshot when you "
            "only need to know WHAT is where -- it costs no image tokens "
            "and returns immediately."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "live_view_url",
        "description": (
            "URL of the live dashboard (N camera MJPEG streams + robot "
            "narration feed). Share it with the human so they can watch "
            "the robot work."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "emergency_stop",
        "description": (
            "Soft-stop the arm immediately: freeze in place and reject all "
            "further motion until reset_stop is called."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "reset_stop",
        "description": "Clear a previous emergency_stop so motion tools work again.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
]


class McpSkillServer:
    def __init__(self):
        self._runtime = None
        self._arm = None
        self._init_error: str | None = None
        self._init_lock = threading.Lock()

    # ── runtime lifecycle ────────────────────────────────────────────────

    def _ensure_runtime(self):
        with self._init_lock:
            if self._runtime is not None:
                return self._runtime
            if self._init_error is not None:
                raise RuntimeError(f"hardware init failed earlier: {self._init_error}")
            from ..apps.demo import build_runtime
            from ..config import PACKAGE_ROOT, load_demo_config

            cameras = [
                c.strip()
                for c in os.environ.get(
                    "WRC_CAMERAS", os.environ.get("WRC_CAMERA", "mock")
                ).split(",")
                if c.strip()
            ]
            arm = os.environ.get("WRC_ARM", "mock")
            run_dir = Path(
                os.environ.get("WRC_RUN_DIR", PACKAGE_ROOT / "runs" / f"mcp_{os.getpid()}")
            )
            try:
                # Anything the stack prints must not corrupt the protocol stream.
                with contextlib.redirect_stdout(sys.stderr):
                    cfg = load_demo_config(cameras=cameras, arm=arm, llm="mock")
                    det_model = os.environ.get("WRC_DETECTOR_MODEL")
                    if det_model:
                        cfg._data["detector"]["model"] = det_model
                    classes = os.environ.get("WRC_DETECT_CLASSES")
                    if classes:
                        cfg._data["detect_classes"] = [
                            c.strip() for c in classes.split(",") if c.strip()
                        ]
                    port = os.environ.get("WRC_STREAM_PORT")
                    if port:
                        cfg._data.setdefault("stream", {})["port"] = int(port)
                    view = os.environ.get("WRC_VIEW", "1") != "0" and bool(
                        os.environ.get("DISPLAY")
                    )
                    serve = os.environ.get("WRC_STREAM", "1") != "0"
                    # lazy_arm: perception comes up now; motors stay untouched
                    # until the first motion command.
                    self._runtime, self._arm = build_runtime(
                        cfg, run_dir, view=view, lazy_arm=True, serve=serve
                    )
                url = (
                    self._runtime.stream_server.url
                    if self._runtime.stream_server is not None else "disabled"
                )
                print(
                    f"[wrc-mcp] runtime up: cameras={cameras} arm={arm} (lazy) "
                    f"livestream={url}",
                    file=sys.stderr,
                )
            except Exception as e:
                self._init_error = f"{type(e).__name__}: {e}"
                raise
            return self._runtime

    def prewarm_async(self) -> None:
        """Bring perception up in the background so the first command is
        instant. A prewarm failure must NOT permanently poison the server:
        transient conditions (camera enumerating, port busy) often clear by
        the time a human sends the first command, so the poison flag is
        reset and the first tools/call rebuilds from scratch."""

        def _warm():
            try:
                self._ensure_runtime()
            except Exception as e:
                print(f"[wrc-mcp] prewarm failed (will retry on first call): {e}",
                      file=sys.stderr)
                with self._init_lock:
                    self._init_error = None

        threading.Thread(target=_warm, daemon=True, name="wrc-prewarm").start()

    def shutdown(self):
        with contextlib.redirect_stdout(sys.stderr):
            if self._runtime is not None:
                from ..apps.demo import shutdown_runtime

                shutdown_runtime(self._runtime, self._arm)

    # ── tool surface ─────────────────────────────────────────────────────

    def list_tools(self) -> list[dict]:
        from ..skills.runtime import TOOL_SPECS

        specs = [t for t in TOOL_SPECS if t["name"] not in _EXCLUDED_TOOLS]
        specs = specs + _EXTRA_TOOLS
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "inputSchema": t["parameters"],
            }
            for t in specs
        ]

    def call_tool(self, name: str, arguments: dict) -> dict:
        runtime = self._ensure_runtime()
        with contextlib.redirect_stdout(sys.stderr):
            if name == "camera_snapshot":
                return self._camera_snapshot(runtime, (arguments or {}).get("camera"))
            if name == "world_state":
                return _text_result(self._world_state(runtime))
            if name == "live_view_url":
                url = (
                    runtime.stream_server.url
                    if runtime.stream_server is not None else None
                )
                if url:
                    return _text_result({"ok": True, "url": url})
                return _text_result(
                    {"ok": False,
                     "error": "livestream not running: disabled via WRC_STREAM=0 "
                              "or the port was taken at startup (see gateway "
                              "stderr; set WRC_STREAM_PORT to change it)"},
                    is_error=True,
                )
            if name == "emergency_stop":
                runtime.arm.stop()
                return _text_result({"ok": True, "stopped": True,
                                     "note": "arm frozen; call reset_stop to resume"})
            if name == "reset_stop":
                runtime.arm.resume()
                return _text_result({"ok": True, "stopped": False})
            if name == "pick_and_place":  # narrate on the dashboard
                obj = (arguments or {}).get("object", "?")
                dest = (arguments or {}).get("destination")
                runtime.current_task = f"pick and place {obj}" + (f" -> {dest}" if dest else "")
                try:
                    result = runtime.execute(name, arguments or {})
                finally:
                    runtime.current_task = None
            else:
                result = runtime.execute(name, arguments or {})
        return _text_result(result, is_error=not result.get("ok", False))

    def _world_state(self, runtime) -> dict:
        from ..apps.demo import _runtime_state

        state = _runtime_state(runtime)
        state["cameras"] = runtime.rig.stats() if getattr(runtime, "rig", None) else {}
        if runtime.stream_server is not None:
            state["live_view_url"] = runtime.stream_server.url
        state["ok"] = True
        return state

    def _camera_snapshot(self, runtime, camera: str | None = None) -> dict:
        rig = getattr(runtime, "rig", None)
        age_s = 0.0
        if camera and rig is not None:
            try:
                stream = rig.get(camera)
            except KeyError as e:
                return _text_result({"ok": False, "error": str(e)}, is_error=True)
            frame = stream.latest()
            if frame is None:
                return _text_result({"ok": False, "error": "no frame yet"}, is_error=True)
            # latest() never blocks -- do not silently serve a pre-glitch
            # frame as if it were live.
            age_s = round(time.monotonic() - frame.t, 1)
            if stream.last_error and age_s > 2.0:
                return _text_result(
                    {"ok": False,
                     "error": f"camera {camera!r} is not delivering frames "
                              f"(last error: {stream.last_error}; newest frame "
                              f"is {age_s}s old)"},
                    is_error=True,
                )
            import cv2

            ok, buf = cv2.imencode(".jpg", frame.rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])
            jpeg = buf.tobytes() if ok else None
        else:
            frame = runtime.observe()
            jpeg = runtime.frame_jpeg()
        if not jpeg:
            return _text_result({"ok": False, "error": "no frame available"}, is_error=True)
        return {
            "content": [
                {
                    "type": "image",
                    "data": base64.b64encode(jpeg).decode(),
                    "mimeType": "image/jpeg",
                },
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "camera": camera or (runtime.rig.primary.name
                                                 if getattr(runtime, "rig", None) else "primary"),
                            "depth_source": frame.depth_source,
                            "frame_age_s": age_s,
                            "t": time.time(),
                        }
                    ),
                },
            ],
            "isError": False,
        }


def _text_result(payload: dict, is_error: bool = False) -> dict:
    return {
        "content": [{"type": "text", "text": json.dumps(payload)}],
        "isError": is_error,
    }


# ── JSON-RPC plumbing ────────────────────────────────────────────────────


def _response(req_id, result=None, error=None) -> dict:
    msg: dict = {"jsonrpc": "2.0", "id": req_id}
    if error is not None:
        msg["error"] = error
    else:
        msg["result"] = result
    return msg


def handle_message(server: McpSkillServer, msg: dict) -> dict | None:
    """-> response dict, or None for notifications."""
    method = msg.get("method")
    req_id = msg.get("id")
    is_notification = req_id is None

    try:
        if method == "initialize":
            client_ver = (msg.get("params") or {}).get("protocolVersion", "")
            version = client_ver if client_ver in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
            return _response(
                req_id,
                {
                    "protocolVersion": version,
                    "capabilities": {"tools": {}},
                    "serverInfo": SERVER_INFO,
                },
            )
        if method in ("notifications/initialized", "initialized"):
            return None
        if method == "ping":
            return _response(req_id, {})
        if method == "tools/list":
            return _response(req_id, {"tools": server.list_tools()})
        if method == "tools/call":
            params = msg.get("params") or {}
            name = params.get("name", "")
            args = params.get("arguments") or {}
            try:
                return _response(req_id, server.call_tool(name, args))
            except Exception as e:
                return _response(
                    req_id,
                    _text_result(
                        {"ok": False, "error": f"{type(e).__name__}: {e}"}, is_error=True
                    ),
                )
        if is_notification:
            return None
        return _response(req_id, error={"code": -32601, "message": f"method not found: {method}"})
    except Exception as e:  # never kill the server on one bad message
        if is_notification:
            return None
        return _response(req_id, error={"code": -32603, "message": f"internal error: {e}"})


def main() -> int:
    server = McpSkillServer()
    # The prewarm thread wraps its build in redirect_stdout(sys.stderr),
    # which swaps the PROCESS-GLOBAL sys.stdout. Protocol frames must go
    # through a reference captured before that thread starts, or the
    # initialize/tools/list responses land on stderr and the client hangs.
    protocol_out = sys.stdout
    print("[wrc-mcp] wrc-demo MCP server on stdio", file=sys.stderr)
    if os.environ.get("WRC_PREWARM", "1") != "0":
        server.prewarm_async()
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                print(f"[wrc-mcp] bad JSON frame: {line[:120]}", file=sys.stderr)
                continue
            resp = handle_message(server, msg)
            if resp is not None:
                protocol_out.write(json.dumps(resp) + "\n")
                protocol_out.flush()
    finally:
        server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
