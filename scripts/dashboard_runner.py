"""Persistent wrc_demo runtime for a long-running demo (booth / kiosk mode).

Loads the runtime, runs ONE scripted pick task to verify everything wires
together, then stays alive serving the MJPEG livestream dashboard on :8090
so a remote operator (or browser) can drive further tasks via the dashboard
"send" button or via the MCP server.

The deliberation tier is auto-detected:
  - `anthropic` if ANTHROPIC_API_KEY is set
  - `local_qwen3_vl` if a vLLM server is reachable at http://127.0.0.1:8080
  - `mock` otherwise (beh in the transparent about it in the narration)

Usage:
    python scripts/dashboard_runner.py
    python scripts/dashboard_runner.py --cameras orbbec --arm rebot_rs

Environment:
    WRC_DASHBOARD_TASK  task to run on boot (default: "pick and place pink object")
    WRC_STREAM_HOST     dashboard bind host (default: 0.0.0.0)
    WRC_STREAM_PORT     dashboard bind port (default: 8090)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# YOLOE re-embeds text prompts on class-list changes; without offline mode
# ultralytics phones GitHub on those paths (rate-limited at the booth ->
# seconds-long stalls that starve the 3 Hz watcher).
os.environ.setdefault("YOLO_OFFLINE", "True")
os.environ.setdefault("ULTRALYTICS_OFFLINE", "True")
# YOLOE's text encoder (mobileclip2_b.ts) is resolved relative to the CWD:
# a runner launched from the wrong directory silently loses ALL detections
# ("mobileclip2_b.ts does not exist" per watcher tick). Pin the cwd to the
# repo, where the checkpoint lives.
os.chdir(REPO)

from wrc_demo.agent.llm import make_llm
from wrc_demo.agent.orchestrator import AgentOrchestrator
from wrc_demo.agent.reflex import ExperienceMemory, FastPlanner
from wrc_demo.apps.demo import (
    build_runtime, recover_after_user_stop, shutdown_runtime,
)
from wrc_demo.config import load_demo_config
from wrc_demo.skills.library import SkillLibrary


def main() -> int:
    p = argparse.ArgumentParser(description="Long-running wrc_demo runner")
    p.add_argument("--cameras", default="mock",
                   help="comma-separated camera profiles (first = manipulation)")
    p.add_argument("--arm", default="mock", help="arm profile (mock|rebot_rs)")
    p.add_argument("--llm", default=None,
                   help="force a specific LLM profile; default = auto-detect")
    p.add_argument("--task", default=os.environ.get("WRC_DASHBOARD_TASK",
                                                     "pick and place pink object"),
                   help="task to run once on boot")
    args = p.parse_args()

    # Auto-detect the deliberation tier when not explicitly overridden.
    # Reflex/experience tiers work the same either way; only the LLM tier
    # needs a real backend to actually plan novel tasks.
    # Priority: CLI --llm > configs/demo.yaml llm.profile > env-detect fallback.
    if args.llm:
        llm_profile = args.llm
    else:
        # Try configs/demo.yaml first (the user's source of truth).
        llm_profile = None
        try:
            import yaml as _yaml
            _demo_yaml = _yaml.safe_load((REPO / "configs" / "demo.yaml").read_text()) or {}
            llm_profile = (_demo_yaml.get("llm") or {}).get("profile")
        except Exception:
            pass
        if not llm_profile:
            # Fallback: env-detect
            qwen_up = False
            try:
                urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=2)
                qwen_up = True
            except Exception:
                pass
            if os.environ.get("ANTHROPIC_API_KEY"):
                llm_profile = "anthropic"
            elif qwen_up:
                llm_profile = "local_qwen3_vl"
            else:
                llm_profile = "mock"
    print(f"LLM TIER: {llm_profile} (source: {'cli/demo-yaml' if args.llm else 'auto'})", flush=True)

    cameras = [c.strip() for c in args.cameras.split(",")]
    cfg = load_demo_config(cameras=cameras, arm=args.arm, llm=llm_profile)
    # pastel cubes on RTX renders sit ~0.15; lower the threshold for the
    # booth setting where objects are typically uniform-colored.
    cfg._data["detector"]["conf"] = 0.12
    # name the booth objects explicitly: beliefs are stored under DETECTOR
    # labels, and "banana" cannot resolve a belief labeled "fruit"
    cfg._data["detect_classes"] = [
        "banana", "cracker box", "soup can", "cube", "box", "bottle", "toy",
    ]

    runtime, arm = build_runtime(cfg, REPO / "runs" / "dashboard", view=False, serve=True)
    print("DASHBOARD:", runtime.stream_server.url if runtime.stream_server else "OFF",
          flush=True)

    experience = ExperienceMemory(REPO / "runs" / "experience.json")
    skill_library = SkillLibrary(REPO / "skills_library")
    fallback_chain = list(cfg._data.get("llm", {}).get("fallback", []) or [])
    agent = AgentOrchestrator(
        make_llm(cfg.llm), runtime, decompose=True,
        fast_planner=FastPlanner(experience), cfg=cfg,
        fallback_chain=fallback_chain, skill_library=skill_library,
    )
    # Live agent telemetry (LLM task decomposition + skill-call timeline)
    # for the dashboard's /state poll. decompose=True: the user-facing panel
    # shows the LLM's milestones; reflex/experience fast paths return
    # before _decompose ever calls the LLM, so simple commands pay nothing.
    runtime.agent = agent

    def run_task(task: str):
        runtime.current_task = task
        _spk = getattr(runtime, "speaker", None)
        if _spk is not None:
            _spk.say(task)  # echo verbatim; AGX TTS picks zh/en by script
        try:
            r = agent.run_task(task)
            if _spk is not None:
                _spk.say("任务完成" if r.success else f"任务失败：{r.summary[:60]}")
            if r.path == "llm" and llm_profile == "mock":
                # The web runner's deliberation tier is a MOCK: be honest in
                # the narration instead of silently "succeeding".
                runtime.memory.add(
                    "note",
                    f"I could not act on {task!r} with my reflexes. For "
                    "free-form language, talk to me through Claude CLI "
                    "(cd wrc_demo && claude), or start the web runner with a "
                    "real LLM (ANTHROPIC_API_KEY + llm='anthropic').",
                )
            print(f"TASK {task!r}: success={r.success} path={r.path} {r.duration_s}s",
                  flush=True)
            print(f"  SUMMARY: {r.summary[:400]}", flush=True)
            for entry in (r.tool_log or [])[-6:]:
                import json as _json

                print(f"  TOOL {entry.get('tool')}: "
                      f"{_json.dumps(entry.get('result'))[:220]}", flush=True)
            for ev in runtime.memory.events()[-10:]:
                print(f"  NOTE[{ev.kind}]: {ev.text}"[:240], flush=True)
            return r
        finally:
            runtime.current_task = None
            if runtime.arm.harness.estopped:
                recover_after_user_stop(runtime)

    def cancel_task():
        runtime.memory.add("outcome", "USER pressed STOP -- aborting the current action")
        runtime.arm.stop()

    if runtime.stream_server is not None:
        runtime.stream_server.set_task_fn(run_task)  # the web chat box
        runtime.stream_server.set_cancel_fn(cancel_task)  # the STOP button

    # Brief settle period so the watcher has time to localize at least one
    # of the requested labels before we kick off the boot task.
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and runtime.beliefs.find("pink object") is None:
        time.sleep(0.3)
    print("BELIEFS:", [(o["label"], o["color"], [round(p, 2) for p in o["position"]])
                       for o in runtime.beliefs.summary()], flush=True)

    r = run_task(args.task)
    print(f"RESULT: success={r.success} path={r.path} {r.duration_s}s", flush=True)
    print("SUMMARY:", r.summary[:300], flush=True)
    if r.tool_log:
        import json

        print("LAST:", json.dumps(r.tool_log[-1]["result"])[:400], flush=True)
    print("BELIEFS-AFTER:", [(o["label"], o["color"], [round(p, 2) for p in o["position"]])
                             for o in runtime.beliefs.summary()], flush=True)
    print("DASHBOARD-LIVE (Ctrl+C to stop)", flush=True)
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown_runtime(runtime, arm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
