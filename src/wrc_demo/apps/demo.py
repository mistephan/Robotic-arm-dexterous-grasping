"""Main demo entry point.

    wrc-demo --task "pick and place pink object" \
             --cameras l515,uvc --arm rebot_rs --llm anthropic

Defaults are the fully-offline stack (mock camera/arm/llm) so the wiring can
always be exercised without hardware or network. `--interactive` keeps the
session open for multiple tasks with persistent memory and beliefs.

Every run now livestreams: N cameras pump continuously (CameraRig), the
WorldWatcher keeps the belief store warm, and an MJPEG dashboard serves the
whole rig to any browser. Routine commands run on the reflex/experience fast
path without an LLM call; only novel tasks reach the model.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import threading
import time
import warnings

import numpy as np
from pathlib import Path

from ..agent.advisor import Advisor
from ..agent.llm import make_llm
from ..agent.orchestrator import AgentOrchestrator
from ..agent.reflex import ExperienceMemory, FastPlanner
from ..agent.trace import TraceLogger
from ..config import CONFIG_DIR, Cfg, PACKAGE_ROOT, load_demo_config
from ..control import make_arm
from ..memory import BeliefStore, EpisodicMemory
from ..perception.camera_base import make_camera
from ..perception.depth_provider import DepthProvider
from ..perception.grounding import Extrinsics
from ..perception.stream import CameraRig, CameraStream
from ..perception.world import LockedDetector, WatchedCamera, WorldWatcher
from ..safety.harness import Kinematics, SafetyHarness, SafetyLimits
from ..skills.runtime import SkillRuntime


def _redirect_warnings_to_file(run_dir: Path) -> Path:
    """Route Python warnings to ``run_dir/warnings.log`` instead of stderr.

    Warnings remain available for diagnosis, but asynchronous CUDA/torch
    notices can no longer overwrite the interactive ``task>`` prompt.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    warning_path = run_dir / "warnings.log"
    write_lock = threading.Lock()

    def _showwarning(message, category, filename, lineno, file=None, line=None):
        rendered = warnings.formatwarning(
            message, category, filename, lineno, line=line
        )
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with write_lock, warning_path.open("a", encoding="utf-8") as log:
            log.write(f"[{timestamp}] {rendered}")

    warnings.showwarning = _showwarning
    return warning_path


def _task_prompt(run_dir: Path):
    """Build an editable prompt and an async-output-safe context manager."""
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.patch_stdout import patch_stdout
    except ImportError:
        return lambda: input("task> "), contextlib.nullcontext()

    history_path = Path(run_dir).parent / ".wrc_task_history"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    session = PromptSession(
        history=FileHistory(str(history_path)),
        auto_suggest=AutoSuggestFromHistory(),
        enable_history_search=True,
        multiline=False,
    )
    return lambda: session.prompt("task> "), patch_stdout(raw=True)


def _resolve_llm_profile(args_llm):
    """Determine the effective LLM profile name.

    Priority: CLI --llm (if provided) > demo.yaml llm.profile > "mock".

    This lets users switch the demo's brain by editing one line of
    configs/demo.yaml (e.g. ``profile: minimax`` for the cloud API)
    instead of remembering the --llm CLI flag. CLI flag still wins
    for ad-hoc overrides and tests.
    """
    if args_llm:
        return args_llm
    # Look up at call time so tests can monkeypatch wrc_demo.config.CONFIG_DIR
    from .. import config as _config
    demo_yaml_path = _config.CONFIG_DIR / "demo.yaml"
    if demo_yaml_path.exists():
        import yaml as _yaml
        with open(demo_yaml_path) as f:
            demo_yaml = _yaml.safe_load(f) or {}
        profile = (demo_yaml.get("llm") or {}).get("profile")
        if profile:
            return str(profile)
    return "mock"


def _camera_cfgs(cfg) -> list[Cfg]:
    """All camera profiles (cfg.cameras when present, else [cfg.camera])."""
    raw = cfg.get("cameras")
    if raw:
        return [Cfg(c) if isinstance(c, dict) else c for c in raw]
    return [cfg.camera]


def build_runtime(
    cfg,
    run_dir: Path,
    view: bool = False,
    lazy_arm: bool = False,
    serve: bool = False,
) -> tuple[SkillRuntime, object]:
    """Build the runtime, rolling back powered hardware on any failure."""
    resources: dict[str, object] = {}

    def _remember(name: str, resource: object) -> None:
        resources[name] = resource

    try:
        return _build_runtime_unchecked(
            cfg, run_dir, view=view, lazy_arm=lazy_arm, serve=serve,
            _on_resource=_remember,
        )
    except BaseException:
        runtime = resources.get("runtime")
        arm = resources.get("arm")
        if runtime is not None and arm is not None:
            shutdown_runtime(runtime, arm)
        else:
            import contextlib

            rig = resources.get("rig")
            if rig is not None:
                with contextlib.suppress(Exception, KeyboardInterrupt):
                    rig.close()
            if arm is not None:
                _disconnect_arm(arm)
        raise


def _build_runtime_unchecked(
    cfg,
    run_dir: Path,
    view: bool = False,
    lazy_arm: bool = False,
    serve: bool = False,
    _on_resource=lambda _name, _resource: None,
) -> tuple[SkillRuntime, object]:
    kin = Kinematics(
        urdf_path=str(cfg.arm.urdf),
        ee_frame=str(cfg.arm.get("ee_frame", "gripper_end")),
        arm_joint_count=int(cfg.arm.get("n_joints", 6)),
    )
    # ``make_arm`` returns the right arm type (MockArm / SafeArm
    # wrapping the vendored SDK) and connects it. For real arms this
    # powers the motors immediately; the ``lazy_arm`` flag is now a
    # no-op (kept in the signature for back-compat with the MCP server).
    arm = make_arm(cfg, kinematics=kin)
    _on_resource("arm", arm)

    # ── the camera rig: N continuous streams, first = manipulation ──────
    cam_cfgs = _camera_cfgs(cfg)
    streams, watched = [], []
    # detector init moved AFTER rig.open() to avoid USB reset during YOLO load

    def fk():
        # Eye-in-hand extrinsics need live FK -- but the 3 Hz watcher must
        # NEVER be the thing that triggers an arm re-connect (the SDK's
        # start() powers the motors). The watcher skips this camera until
        # the arm reports connected.
        if not getattr(arm, "connected", True):
            from ..types import SkillError

            raise SkillError("eye-in-hand extrinsics need the arm (still in standby)")
        return kin.fk(arm.get_state().q)

    for i, ccfg in enumerate(cam_cfgs):
        stream = CameraStream(
            make_camera(ccfg),
            name=str(ccfg.get("name", f"cam{i}")),
            rate_hz=float(ccfg.get("fps", 30.0)),
        )
        streams.append(stream)
        # requires_calibration + no on-disk extrinsics -> force fuse=False.
        # The strict loader above already returns identity T for missing
        # files; this branch fails closed *before* WatchedCamera ever sees
        # the camera, so 3D beliefs stay clean during bring-up.
        requires_calib = bool(ccfg.get("requires_calibration", False))
        extr_cfg = ccfg.get("extrinsics", _empty_cfg())
        if requires_calib:
            src_path = str(extr_cfg.get("source") or extr_cfg.get("hand_eye_npz") or "")
            have_src = bool(src_path) and Path(str(src_path)).exists()
            if not have_src:
                print(
                    f"[wrc-demo] {ccfg.get('name', f'cam{i}')}: "
                    "requires_calibration=true but no extrinsics source "
                    f"({src_path!r}); fusing beliefs is disabled for this camera",
                    file=sys.stderr,
                )
                fuse = False
            else:
                fuse = bool(ccfg.get("fuse_beliefs", "extrinsics" in ccfg))
        else:
            fuse = bool(ccfg.get("fuse_beliefs", "extrinsics" in ccfg))
        watched.append(
            WatchedCamera(
                stream=stream,
                depth=DepthProvider(ccfg),
                extrinsics=Extrinsics.from_config(extr_cfg, fk_tcp2base=fk),
                # A camera without calibrated extrinsics must not fuse 3D
                # beliefs (garbage base-frame positions); it still streams
                # video + overlays + heartbeats.
                fuse=fuse,
            )
        )
    rig = CameraRig(streams)
    _on_resource("rig", rig)
    rig.open()
    # Detector init AFTER camera open to avoid USB reset during YOLO/GPU load
    detector = LockedDetector(_make_detector(cfg))
    try:
        rig.primary.warm_up(int(cam_cfgs[0].get("warmup_frames", 5)))
    except Exception:
        rig.close()  # a partial build must not leak open camera streams
        raise

    memory = EpisodicMemory(horizon_s=float(cfg.memory.get("horizon_s", 15.0)))
    # Belief store tuning from the optional demo.yaml `beliefs:` block.
    # Defaults preserve the documented object-permanence semantics; see
    # the block's comments for the anti-accumulation knobs.
    bcfg = cfg.get("beliefs", None)
    forget = bcfg.get("forget_after_s", None) if bcfg is not None else None
    beliefs = BeliefStore(
        **({
            "match_radius_m": float(bcfg.get("match_radius_m", 0.08)),
            "stale_revive_s": float(bcfg.get("stale_revive_s", 5.0)),
            "max_per_label": int(bcfg.get("max_per_label", 2)),
            **({"forget_after_s": float(forget)} if forget else {}),
        } if bcfg is not None else {})
    )
    trace = TraceLogger(run_dir)
    runtime = SkillRuntime(
        rig.primary, watched[0].depth, detector, watched[0].extrinsics,
        kin, arm, memory, beliefs, trace, cfg,
    )
    runtime.rig = rig
    _on_resource("runtime", runtime)

    # Voice announcements: key events -> Piper TTS on the AGX -> USB
    # speaker. Real rig only -- mock e2e tests drive build_runtime and
    # memory.add must never trigger network/audio from the test stack.
    runtime.speaker = None
    if str(cfg.arm.get("type", "mock")) != "mock":
        from ..voice.asr import ensure_remote_voice_services

        # AGX reboot leaves the voice service down; start it now so the
        # mic (ASR) and announcements (TTS) both work without any manual
        # step. No-op (one 2.5 s health GET) when already running.
        ensure_remote_voice_services(cfg.get("voice", None))

        from ..voice.tts import make_speaker

        speaker = make_speaker(cfg.get("voice", None))
        if speaker is not None:
            runtime.speaker = speaker
            _orig_add = memory.add

            def _speaking_add(kind, text, **kw):
                try:
                    speaker.observe(kind, text)   # speak key events
                except Exception:
                    pass                          # narration never blocks
                return _orig_add(kind, text, **kw)

            memory.add = _speaking_add
            print(f"[wrc-demo] voice announcements on ({speaker.tts_url})",
                  file=sys.stderr)

    pcfg = cfg.get("perception_loop", _empty_cfg())
    if bool(pcfg.get("enabled", True)):
        watcher = WorldWatcher(
            watched, detector, beliefs,
            classes=list(cfg.get("detect_classes", [])) or ["cup", "bottle", "box"],
            rate_hz=float(pcfg.get("rate_hz", 3.0)),
            harness=arm.harness,  # MockArm / SafeArm both expose .harness
        )
        watcher.start()
        runtime.watcher = watcher

    runtime.stream_server = None
    scfg = cfg.get("stream", _empty_cfg())
    if serve and bool(scfg.get("enabled", True)):
        from .stream_server import StreamServer

        server = StreamServer(
            rig,
            state_fn=lambda: _runtime_state(runtime),
            host=str(scfg.get("host", "0.0.0.0")),
            advertise_host=(
                os.environ.get("WRC_STREAM_ADVERTISE_HOST")
                or scfg.get("advertise_host")
                or None
            ),
            port=int(scfg.get("port", 8090)),
            fps=float(scfg.get("fps", 15.0)),
            quality=int(scfg.get("quality", 80)),
            tls=bool(scfg.get("tls", False)),
        )
        try:
            server.start()
            runtime.stream_server = server
        except OSError as e:  # port taken: the demo must keep running
            print(f"[wrc-demo] livestream disabled ({e}); "
                  "set stream.port or WRC_STREAM_PORT", file=sys.stderr)

        # Voice input: mic button on the dashboard. Degrades gracefully --
        # faster-whisper is a post-install pip dep; if it is missing the
        # button reports the error (or stays hidden via /state voice flag).
        if runtime.stream_server is not None:
            from ..voice.asr import make_asr
            asr = make_asr(cfg.get("voice", None))
            if asr is not None:
                runtime.stream_server.set_asr_fn(asr.transcribe)
                detail = getattr(asr, "model_name", None) or getattr(asr, "url", "")
                print(f"[wrc-demo] voice input enabled "
                      f"({type(asr).__name__} {detail})", file=sys.stderr)

    if view:
        from .live_view import RigViewer

        runtime.viewer = RigViewer(rig)
        runtime.viewer.start()

    return runtime, arm


def _runtime_state(runtime) -> dict:
    """Live world state for the dashboard/MCP (never touches the lazy arm)."""
    _, status = runtime.camera.overlay() if hasattr(runtime.camera, "overlay") else ([], "n/a")
    out = {
        "agent_status": status,
        "task": runtime.current_task,
        "holding": runtime.held_object,
        "objects": runtime.beliefs.summary(),
        "arm_connected": getattr(runtime.arm, "connected", True),
        # human-readable narration: newest events last (observations, skill
        # calls, outcomes). Reads the never-pruned display history, NOT the
        # 15 s agent digest window -- otherwise narration vanished from the
        # screen ~15 s after it happened (horizon_s prunes the agent store).
        "events": runtime.memory.display_digest(max_lines=40).splitlines(),
        "grasp_debug": runtime.grasp_debug_snapshot(),
    }
    # Agent telemetry: LLM task decomposition (milestones) + per-skill-call
    # timeline. Absent on minimal runtimes (no agent wired).
    agent = getattr(runtime, "agent", None)
    if agent is not None and hasattr(agent, "progress_snapshot"):
        try:
            out["agent"] = agent.progress_snapshot()
        except Exception:
            pass
    if runtime.watcher is not None:
        out["perception"] = runtime.watcher.stats()
    # Projection context for the primary camera; diagnostics only.
    frame = runtime.camera.latest() if hasattr(runtime.camera, "latest") else None
    if frame is not None:
        try:
            T_base_cam = (frame.T_base_cam if frame.T_base_cam is not None
                          else runtime.extrinsics.cam_to_base())
            out["grasp_projection"] = {
                "camera": getattr(runtime.camera, "name", None),
                "T_base_cam": np.asarray(T_base_cam, dtype=float).tolist(),
            }
        except Exception:
            pass
    return out


def _disconnect_arm(arm) -> None:
    """Disable an arm during partial startup; never raises."""
    import contextlib

    end_effector = getattr(arm, "end_effector", None)
    if end_effector is not None:
        with contextlib.suppress(Exception, KeyboardInterrupt):
            end_effector.close()

    with contextlib.suppress(Exception, KeyboardInterrupt):
        if hasattr(arm, "disconnect_gently"):
            arm.disconnect_gently()
        else:
            arm.disconnect()


def shutdown_runtime(runtime, arm) -> None:
    """Stop threads and hardware in dependency order; never raises.

    Real RS hardware gets a gradual impedance/gravity-support ramp before
    the final motor disable.  Emergency stop remains a separate immediate
    operation; this path is for normal EOF/empty-line/process shutdown.
    """
    import contextlib

    for step in (
        lambda: runtime.watcher.stop() if runtime.watcher is not None else None,
        lambda: runtime.speaker.stop() if getattr(runtime, "speaker", None) else None,
        lambda: runtime.stream_server.stop() if getattr(runtime, "stream_server", None) else None,
        lambda: runtime.viewer.stop() if getattr(runtime, "viewer", None) else None,
        lambda: runtime.rig.close() if getattr(runtime, "rig", None) else runtime.camera.close(),
        (
            lambda: (runtime.skill_open_gripper()
                 if getattr(arm, "end_effector", None) is not None
                 else runtime.skill_close_gripper())
            if not getattr(runtime, "held_object", None) else None
        ),
        lambda: _disconnect_arm(arm),
    ):
        with contextlib.suppress(Exception, KeyboardInterrupt):
            step()


def recover_after_user_stop(runtime) -> None:
    """After a user STOP has aborted the task, resume only to park at home."""
    reasons = list(getattr(runtime.arm.harness, "violations", []))
    if not any(
        reason in {"safe_arm.stop()", "chat cancel"} for reason in reasons
    ):
        return
    try:
        runtime.arm.resume()
        runtime.observe()
        runtime.skill_move_home(park_gripper=False)
        runtime.memory.add("note", "STOP recovery complete; arm returned home")
    except Exception as exc:
        runtime.memory.add(
            "warning", f"STOP recovery could not return home: {exc}"
        )


def _empty_cfg():
    from ..config import Cfg

    return Cfg({})


def _make_detector(cfg):
    # A camera profile may pin its own detector (the mock camera uses the
    # mock detector so offline runs never load model weights).
    dcfg = cfg.camera.get("detector") or cfg.detector
    if dcfg.type == "mock":
        from ..perception.detector import MockDetector

        return MockDetector(label=dcfg.get("label", "red cube"))
    from ..perception.detector import OpenVocabDetector

    return OpenVocabDetector(
        model_path=dcfg.model,
        device=dcfg.get("device", "cuda:0"),
        conf=float(dcfg.get("conf", 0.25)),
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="WRC agentic grasping demo")
    p.add_argument("--task", default=None, help="natural-language task")
    p.add_argument("--camera", default="mock", help="camera profile (mock|l515|d435i|uvc)")
    p.add_argument("--cameras", default=None,
                   help="comma-separated camera profiles; first = manipulation "
                        "camera (e.g. l515,uvc). Overrides --camera.")
    p.add_argument("--arm", default="mock", help="arm profile (mock|rebot_rs)")
    p.add_argument("--llm", default=None,
               help="llm profile (overrides demo.yaml llm.profile); "
                    "one of: mock|local_qwen3_vl|minimax|anthropic|openai")
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--run-dir", default=None, help="trace output dir")
    p.add_argument("--interactive", action="store_true", help="multi-task REPL")
    p.add_argument("--no-view", action="store_true",
                   help="do not open the live camera window")
    p.add_argument("--no-serve", action="store_true",
                   help="do not start the MJPEG livestream dashboard")
    args = p.parse_args(argv)

    cameras = [c.strip() for c in args.cameras.split(",")] if args.cameras else None
    llm_profile = _resolve_llm_profile(args.llm)
    cfg = load_demo_config(camera=args.camera, cameras=cameras, arm=args.arm, llm=llm_profile)
    run_dir = Path(args.run_dir) if args.run_dir else (
        PACKAGE_ROOT / "runs" / time.strftime("%Y%m%d_%H%M%S")
    )
    import os

    view = not args.no_view and bool(os.environ.get("DISPLAY"))
    print(f"[wrc-demo] cameras={cameras or [args.camera]} arm={args.arm} "
          f"llm={llm_profile} view={view}")
    print(f"[wrc-demo] traces -> {run_dir}")
    _redirect_warnings_to_file(run_dir)

    runtime, arm = build_runtime(cfg, run_dir, view=view, serve=not args.no_serve)
    if runtime.stream_server is not None:
        print(f"[wrc-demo] LIVESTREAM dashboard: {runtime.stream_server.url}")

    # Ctrl+C = soft stop (freeze + latch e-stop, no free-fall); a second
    # Ctrl+C raises KeyboardInterrupt and tears the process down.
    import signal

    def _sigint(_sig, _frm):
        print("\n[wrc-demo] SIGINT: soft-stopping the arm (Ctrl+C again to exit)")
        runtime.arm.stop()
        signal.signal(signal.SIGINT, signal.default_int_handler)

    signal.signal(signal.SIGINT, _sigint)
    from ..agent.llm import MockLLM

    llm = make_llm(cfg.llm)
    is_mock = isinstance(llm, MockLLM)
    advisor = Advisor(llm) if (llm.supports_vision and not is_mock) else None
    experience = ExperienceMemory(PACKAGE_ROOT / "runs" / "experience.json")
    from ..skills.library import SkillLibrary
    skill_library = SkillLibrary(PACKAGE_ROOT / "skills_library")
    # ADR-0008: optional LLM fallback chain. Read from cfg._data
    # (sanctioned per CLAUDE.md §Config system) so a YAML block like
    #   llm:
    #     fallback: [local_qwen3_vl, mock]
    # wires without any new API surface. Default empty chain = the
    # pre-ADR-0008 behavior (raise on primary failure).
    fallback_chain = list(cfg._data.get("llm", {}).get("fallback", []) or [])
    agent = AgentOrchestrator(
        llm, runtime, advisor=advisor, max_steps=args.max_steps,
        decompose=not is_mock, fast_planner=FastPlanner(experience),
        cfg=cfg, fallback_chain=fallback_chain,
        skill_library=skill_library,
    )
    # Live agent telemetry (task decomposition + skill-call timeline) for
    # the dashboard /state poll; _runtime_state reads it back.
    runtime.agent = agent

    def _run(task: str):
        runtime.current_task = task
        _spk = getattr(runtime, "speaker", None)
        if _spk is not None:
            _spk.say(task)  # echo verbatim; AGX picks zh/en voice by script
        try:
            report = agent.run_task(task)
            if _spk is not None and report is not None:
                _spk.say("任务完成" if report.success else "任务失败")
            return report
        finally:
            runtime.current_task = None
            recover_after_user_stop(runtime)

    # Wire the dashboard "send"/"stop" buttons to the live agent so you can
    # drive the robot from the browser (http://<ip>:8090) as well as the REPL.
    _srv = getattr(runtime, "stream_server", None)
    if _srv is not None:
        _srv.set_task_fn(lambda t: _print_report(_run(t)))
        _srv.set_cancel_fn(runtime.arm.stop)

    try:
        if args.interactive:
            print("Type a task (empty line to quit).")
            read_task, prompt_output = _task_prompt(run_dir)
            with prompt_output:
                while True:
                    try:
                        task = read_task().strip()
                    except EOFError:  # Ctrl+D
                        break
                    except KeyboardInterrupt:  # Ctrl+C clears the edit buffer
                        print("^C (input cleared)")
                        continue
                    if not task:
                        break
                    # Clear e-stop before each task so a previous Ctrl+C
                    # does not permanently block motion.
                    if hasattr(runtime, "arm") and hasattr(runtime.arm, "harness"):
                        runtime.arm.harness.reset_estop()
                    _print_report(_run(task))
        else:
            task = args.task or "look at the table and report what objects you see"
            report = _run(task)
            _print_report(report)
            return 0 if report.success else 1
    finally:
        shutdown_runtime(runtime, arm)
    return 0


def _print_report(report) -> None:
    print("\n=== task report ===")
    print(f"task:    {report.task}")
    print(f"success: {report.success}")
    print(f"path:    {report.path} ({report.duration_s}s)")
    print(f"steps:   {report.steps}")
    print(f"summary: {report.summary}")
    # Reflex summaries historically printed only the skill name and hid the
    # structured return value. That made a successful `look around` appear
    # to contain no observation at all. Show each tool result in a stable,
    # human-readable form for both reflex and LLM paths.
    for item in getattr(report, "tool_log", []) or []:
        result = item.get("result") or {}
        visible = {k: v for k, v in result.items() if k != "ok"}
        if visible:
            print(
                f"result[{item.get('tool', 'tool')}]: "
                + json.dumps(visible, ensure_ascii=False, indent=2)
            )
    # 2026-08-13: explicit "task done" signal — without this users see the
    # arm still hovering at ready pose and (incorrectly) assume the task
    # is still running, then re-issue the same command and trigger a fresh
    # pick_and_place cycle on the object they just placed.
    status = "✓ done" if report.success else "✗ failed"
    print(f"\n[{status}] task complete. "
          f"Enter a new task or press Enter to quit.")


if __name__ == "__main__":
    sys.exit(main())
