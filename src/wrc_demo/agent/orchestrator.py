"""The agent loop: reflex first, then decompose -> act -> verify -> recover.

Inference-time composition of the two papers' ideas plus the Anthropic
robotics study's latency lesson:
- reflex/habit tier (agent.reflex): routine commands compile straight to
  skill calls (µs) or replay a proven plan from experience memory -- the LLM
  never blocks the hot path;
- ASPIRE: curated skill API + per-call multimodal traces + honest failure
  reporting back into context;
- Agentic-VLA: LLM task decomposition into checkable milestones, and a VLM
  advisor consulted only after failures whose one-sentence spatial suggestion
  is appended to the context for the retry.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from ..skills.runtime import TOOL_SPECS, SkillRuntime
from .advisor import Advisor
from .llm import LLMClient, LLMResponse, ToolCall
from .prompts import DECOMPOSE_PROMPT, SYSTEM_PROMPT
from .reflex import FastPlanner, parse_command


def _short_args(args: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in args.items())


def _as_bool(value) -> bool:
    """Schema-lax backends send booleans as strings ('false' is truthy!)."""
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    return bool(value)


@dataclass
class TaskReport:
    task: str
    success: bool
    summary: str
    steps: int
    milestones: list[str] = field(default_factory=list)
    tool_log: list[dict] = field(default_factory=list)
    path: str = "llm"  # "reflex" | "experience" | "llm"
    duration_s: float = 0.0


class AgentOrchestrator:
    def __init__(
        self,
        llm: LLMClient,
        runtime: SkillRuntime,
        advisor: Advisor | None = None,
        max_steps: int = 30,
        decompose: bool = True,
        attach_images: bool = True,
        fast_planner: FastPlanner | None = None,
        cfg=None,
        fallback_chain: tuple[str, ...] = (),
        skill_library=None,
    ):
        self.llm = llm
        self.runtime = runtime
        self.advisor = advisor
        self.max_steps = max_steps
        self.decompose = decompose
        self.attach_images = attach_images and llm.supports_vision
        self.fast_planner = fast_planner
        # ADR-0008: profile switching + fallback. `cfg` is the live
        # wrc_demo.config.Cfg (or any object with a writable `_data["llm"]`
        # dict); `fallback_chain` is an ordered list of profile names tried
        # when the primary chat() raises a transient SDK error. Both default
        # to inert values, so existing callers (demo.py, tests) are
        # backward-compatible.
        self.cfg = cfg
        self.fallback_chain = list(fallback_chain)
        self.skill_library = skill_library
        self.tool_max_tokens = (
            int(cfg.llm.get("max_tokens", 512)) if cfg is not None else 512
        )
        # Live task progress for the dashboard/MCP (see progress_snapshot).
        # One writer (the run_task thread), many readers (HTTP poll); dict
        # field writes are atomic enough under the GIL, and the snapshot
        # deep-copies so a renderer never sees a half-updated entry.
        self.progress: dict = {
            "state": "idle",           # idle | running | done
            "task": None,
            "path": None,              # reflex | experience | llm
            "milestones": [],          # LLM task decomposition lines
            "steps": [],               # compact per-skill-call records
            "current": None,           # skill currently executing
            "last_llm_text": "",
            "success": None,
            "summary": None,
            "elapsed_s": 0.0,
        }

    # ───────────────────────────────────────────────────────────────────
    # ADR-0008: profile switching + transient-error fallback
    # ───────────────────────────────────────────────────────────────────

    def switch_llm(self, profile_name: str) -> "LLMClient":
        """Swap the live LLM client for a new one built from `cfg`.

        `profile_name` overrides `cfg.llm["profile"]` (or sets it if
        absent) before re-deriving the client. The new client becomes
        `self.llm`; the next `chat()` uses it. An `llm_switch` event is
        recorded in the trace.
        """
        from .llm import make_llm
        if self.cfg is None:
            raise RuntimeError(
                "switch_llm requires the orchestrator to be constructed "
                "with a `cfg` argument (ADR-0008)."
            )
        sub_cfg = dict(self.cfg._data["llm"])
        sub_cfg["profile"] = profile_name
        new_llm = make_llm(sub_cfg)
        self.llm = new_llm
        if getattr(self.runtime, "trace", None) is not None:
            self.runtime.trace.record_event(
                "llm_switch", {"profile": profile_name}
            )
        return new_llm

    def _chat_with_fallback(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Try `self.llm` first; on a transient SDK error, walk
        `self.fallback_chain` in order. Stops on success or chain
        exhaustion. `LLMConfigError` (raised from `make_llm` itself)
        is NOT caught here — misconfiguration is a start-up issue that
        should surface, not be silently masked by a fallback.
        """
        from .llm import LLMConfigError
        import openai
        transient = (
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.RateLimitError,
            openai.AuthenticationError,
            openai.InternalServerError,
            openai.BadRequestError,  # 4xx often means the model can't parse the request — recoverable
        )
        last_exc: Exception | None = None
        chain: list[tuple[str, LLMClient]] = [("primary", self.llm)]
        for profile_name in self.fallback_chain:
            try:
                chain.append((profile_name, self._client_for(profile_name)))
            except LLMConfigError:
                # Misconfigured fallback profile — surface immediately
                raise
            except Exception as e:
                # Building the fallback itself failed; record and move on
                if getattr(self.runtime, "trace", None) is not None:
                    self.runtime.trace.record_event(
                        "llm_fallback_build_failed",
                        {"profile": profile_name, "err": str(e)[:200]},
                    )
                continue
        for label, candidate in chain:
            try:
                return candidate.chat(
                    system, messages, tools=tools, max_tokens=max_tokens
                )
            except transient as e:
                # Local OpenAI-compatible servers may report an oversized
                # input as "Input length (N) exceeds ... (M)" rather than the
                # token-count wording handled by OpenAICompatClient. Retry
                # once with task intent plus the newest complete exchange.
                err = str(e).lower()
                context_too_long = (
                    "maximum context length" in err
                    or ("input length" in err and "exceeds" in err)
                )
                if context_too_long:
                    compact = self._compact_chat_history(messages)
                    if compact != messages:
                        try:
                            if getattr(self.runtime, "trace", None) is not None:
                                self.runtime.trace.record_event(
                                    "llm_context_compacted",
                                    {
                                        "from_messages": len(messages),
                                        "to_messages": len(compact),
                                    },
                                )
                            return candidate.chat(
                                system, compact, tools=tools,
                                max_tokens=min(max_tokens, 256),
                            )
                        except transient as compact_error:
                            e = compact_error
                last_exc = e
                if getattr(self.runtime, "trace", None) is not None:
                    self.runtime.trace.record_event(
                        "llm_fallback",
                        {"from": label, "err": str(e)[:200]},
                    )
                continue
        # Chain exhausted
        assert last_exc is not None
        raise last_exc

    @staticmethod
    def _compact_chat_history(messages: list[dict]) -> list[dict]:
        """Keep task intent and the newest protocol-complete tool exchange."""
        if len(messages) <= 3:
            return messages
        first = messages[0]
        start = len(messages) - 1
        while start > 1 and messages[start].get("role") != "assistant":
            start -= 1
        tail = (
            messages[start:]
            if messages[start].get("role") == "assistant"
            else messages[-1:]
        )
        return [first, *tail]

    def _client_for(self, profile_name: str) -> "LLMClient":
        """Build an LLMClient for a named profile (used by the fallback chain)."""
        from .llm import make_llm
        if self.cfg is None:
            raise RuntimeError(
                "fallback_chain requires the orchestrator to be constructed "
                "with a `cfg` argument (ADR-0008)."
            )
        sub_cfg = dict(self.cfg._data["llm"])
        sub_cfg["profile"] = profile_name
        return make_llm(sub_cfg)

    def run_task(self, task: str) -> TaskReport:
        """Lifecycle wrapper around _run_task_impl: resets and finalizes the
        live progress dict so every exit path (early return, LLM failure,
        forced task_done, max_steps, exception) leaves a consistent state
        for the dashboard poll."""
        self.progress.update({
            "state": "running", "task": task, "path": None,
            "milestones": [], "steps": [], "current": None,
            "last_llm_text": "", "success": None, "summary": None,
            "elapsed_s": 0.0,
        })
        try:
            report = self._run_task_impl(task)
        except Exception:
            self.progress.update({
                "state": "done", "success": False,
                "summary": "orchestrator exception (see runtime_errors.log)",
                "current": None,
            })
            raise
        self.progress.update({
            "state": "done",
            "path": report.path,
            "success": bool(report.success),
            "summary": str(report.summary)[:400],
            "duration_s": float(report.duration_s),
            "elapsed_s": float(report.duration_s),
            "current": None,
        })
        # Keep the key the panel reads uniform whether running or done.
        self.progress.setdefault("duration_s", float(report.duration_s))
        return report

    def progress_snapshot(self) -> dict:
        """Deep-copied, JSON-safe progress for _runtime_state / MCP."""
        import copy as _copy

        snap = _copy.deepcopy(self.progress)
        if snap.get("state") == "running" and snap.get("t_start_monotonic"):
            snap["elapsed_s"] = round(
                time.monotonic() - float(snap["t_start_monotonic"]), 2
            )
        snap.pop("t_start_monotonic", None)
        return snap

    def _sync_progress_steps(self, tool_log: list[dict]) -> None:
        """Compact the authoritative tool_log into progress["steps"].
        Called once per loop iteration so every append site (normal calls,
        forced task_done, fast path) is covered without touching them."""
        steps = []
        for e in tool_log:
            result = e.get("result") or {}
            steps.append({
                "step": e.get("step"),
                "tool": e.get("tool"),
                "args": _short_args(e.get("args") or {}),
                "ok": bool(result.get("ok", result.get("task_complete", False))),
                "error": str(result.get("error", ""))[:120] or None,
            })
        self.progress["steps"] = steps

    def _run_task_impl(self, task: str) -> TaskReport:
        t_start = time.monotonic()
        self.progress["t_start_monotonic"] = t_start
        requested_reflex = parse_command(task)
        hold_only_request = bool(
            requested_reflex
            and requested_reflex.calls
            and requested_reflex.calls[0][0] == "grasp_object"
        )
        fast_note = None
        if self.fast_planner is not None:
            report, fast_note = self._try_fast_path(task, t_start)
            if report is not None:
                return report

        # Batch flags must be known BEFORE _decompose: a batch task's plan
        # is the runtime-owned frozen list, so the LLM decomposition round
        # trip (~6-7 s on the AGX) buys nothing — skip it (2026-09-22,
        # log 170536: command → first arm motion was ~28 s, ~16 s of it
        # pure LLM decision chain before any perception or motion).
        batch_words = (
            "all fruit", "all the fruit", "fruit on the table",
            "所有水果", "全部水果", "桌上的水果",
        )
        task_lower = task.lower()
        self._batch_pick_place = (
            any(word in task_lower for word in batch_words)
            or (
                any(word in task_lower for word in ("所有", "全部", "每个"))
                and "水果" in task_lower
            )
        )
        self._requested_container_label = (
            "wicker basket"
            if any(word in task for word in ("筐", "框", "篮", "basket"))
            else None
        )
        # Chinese-container batch commands name an unambiguous physical
        # fixture (this rig has ONE wicker basket): the whole opening chain
        # — observe (list_objects), freeze the list, first pick — is fully
        # determined by the task text + the list result, so it runs without
        # any LLM round trip. English "left basket" style tasks keep the
        # LLM deciding the destination (positional qualifiers matter).
        self._batch_bootstrap = bool(
            self._batch_pick_place
            and self._requested_container_label
            and any(w in task for w in ("筐", "框", "篮子", "篮"))
        )

        milestones = [] if self._batch_pick_place else (
            self._decompose(task) if self.decompose else []
        )
        try:
            self.progress["milestones"] = [str(m)[:160] for m in milestones]
            self.progress["path"] = "llm"
            if milestones:
                self.runtime.trace.record_event("task_decomposition", {
                    "source": "llm", "task": task,
                    "milestones": list(self.progress["milestones"]),
                })
        except Exception:
            pass
        messages: list[dict] = []
        tool_log: list[dict] = []

        intro = f"Task: {task}\n"
        if milestones:
            intro += "Milestones:\n" + "\n".join(f"{i+1}. {m}" for i, m in enumerate(milestones)) + "\n"
        if fast_note:
            intro += f"\n{fast_note}\n"
        if self.skill_library is not None:
            learned = self.skill_library.relevant(task)
            if learned:
                intro += "\nRelevant validated skill-library guidance:\n" + "\n\n".join(learned) + "\n"
        intro += (
            "\nMemory (last 15 s):\n" + self.runtime.memory.digest(max_lines=5)
            + "\n\nBegin. Observe first, then act. Call one tool now."
        )
        messages.append({"role": "user", "content": intro})

        consecutive_failures = 0
        # Track exact successful placements.  This prevents the LLM from
        # repeating one placement while allowing a batch task to move several
        # different objects into the same destination.
        self._completed_pick_places: set[tuple[str, str]] = set()
        self._pending_batch_objects: list[str] | None = None
        self._batch_destination: str | None = None
        self._failed_batch_objects: list[str] = []
        # Batch autopilot: after a successful batch pick_and_place with
        # pending objects left, the next call is synthesized locally (the
        # frozen list is authoritative) instead of paying a per-fruit LLM
        # round trip (~6.5 s on the AGX). Cleared on failure, list
        # exhaustion, or any non-pp call — the LLM regains control then.
        # Reset per task so autopilot never leaks across tasks.
        self._batch_auto_next: ToolCall | None = None
        if self._batch_bootstrap:
            # Bootstrap the loop with a LOCAL list_objects — no LLM round
            # trip before the world is even observed. The pending list and
            # the first pick get synthesized from its result (see the
            # list_objects handler below).
            self._batch_auto_next = ToolCall("list_objects", {})
            try:
                self.runtime.trace.record_event("batch_autopilot_bootstrap", {
                    "container": self._requested_container_label,
                })
            except Exception:
                pass
        # 2026-09-03: same shape for failures. Small local LLMs (Qwen3-VL-2B
        # AWQ-4bit) sometimes ignore the "if pick_and_place fails, call
        # task_done" rule in prompts.py and re-issue the same failing call
        # until max_steps (30) is hit (~2 min of wasted GPU). Cap consecutive
        # pick_and_place failures and force task_done instead.
        # Also: some small LLMs stop calling tools and just emit text
        # ("I see a red cube...") over and over. Track consecutive text-only
        # turns and force task_done.
        self._consecutive_pick_failures = 0
        self._consecutive_text_only = 0
        # One call already owns the complete internal retry budget.
        # Reissuing it would multiply three attempts into six or nine.
        PICK_AND_PLACE_FAILURE_LIMIT = 1
        # Bumped 2026-09-03: minimax-M3 cloud LLM does more reasoning per turn
        # than local Qwen3-VL did; 3 turns was too tight — the LLM was on
        # the verge of correctly calling task_done(success=false, "no wave
        # tool") when the counter fired. 5 gives one more thinking turn
        # of headroom without losing the runaway-loop protection.
        TEXT_ONLY_LIMIT = 5
        for step in range(1, self.max_steps + 1):
            if self._batch_auto_next is not None:
                # Batch autopilot: synthesize the next call as a normal
                # LLMResponse so every downstream guard, the assistant
                # message protocol, and progress bookkeeping keep working
                # unchanged — only the network round trip is skipped.
                auto = self._batch_auto_next
                self._batch_auto_next = None
                resp = LLMResponse(
                    text=(
                        "(batch auto-continue: picking "
                        f"{auto.arguments.get('object')!r} from the frozen list)"
                    ),
                    tool_calls=[auto],
                )
                try:
                    self.progress["last_llm_text"] = resp.text[:400]
                    self.runtime.trace.record_event("batch_auto_continue", {
                        "object": auto.arguments.get("object"),
                        "remaining": list(self._pending_batch_objects or []),
                    })
                except Exception:
                    pass
            else:
                try:
                    resp = self._chat_with_fallback(
                        system=SYSTEM_PROMPT,
                        messages=self._prune_images(messages),
                        tools=TOOL_SPECS,
                        # Reasoning plus a JSON tool call share this output
                        # budget. A 128-token cap truncated Hermes JSON on
                        # AGX; honor the active LLM profile (agx_qwen3vl
                        # sets max_tokens: 512).
                        max_tokens=self.tool_max_tokens,
                    )
                    try:
                        self.progress["last_llm_text"] = (resp.text or "")[:400]
                        self.runtime.trace.record_event("llm_decision", {
                            "step": step,
                            "text": (resp.text or "")[:400],
                            "tool_calls": [
                                {"name": c.name, "arguments": c.arguments}
                                for c in resp.tool_calls[:1]
                            ],
                        })
                    except Exception:
                        pass
                except Exception as exc:
                    # An unavailable/overloaded LLM must fail this task, not
                    # tear down the interactive demo and disable the robot
                    # connection.
                    summary = f"LLM request failed: {str(exc)[:300]}"
                    trace = getattr(self.runtime, "trace", None)
                    if trace is not None:
                        trace.record_event(
                            "llm_task_failed", {"error": str(exc)[:500]}
                        )
                        trace.finish(
                            f"task: {task}\nsuccess: false\n{summary}"
                        )
                    return TaskReport(
                        task, False, summary, step, milestones, tool_log,
                        duration_s=round(time.monotonic() - t_start, 2),
                    )
            if not resp.tool_calls:
                # 2026-09-03: small local LLMs sometimes forget to emit
                # tool calls and just chat. Track consecutive text-only
                # turns and force task_done after TEXT_ONLY_LIMIT (=5).
                self._consecutive_text_only += 1
                if self._consecutive_text_only >= TEXT_ONLY_LIMIT:
                    last_text = (resp.text or "(no text)")[:200]
                    if self.runtime.trace is not None:
                        self.runtime.trace.record_event(
                            "llm_force_task_done",
                            {
                                "reason": (
                                    f"LLM emitted no tool calls for "
                                    f"{self._consecutive_text_only} consecutive turns"
                                ),
                                "last_text": last_text,
                            },
                        )
                    forced_call = ToolCall(
                        name="task_done",
                        arguments={
                            "success": False,
                            "summary": (
                                f"LLM did not emit any tool call for "
                                f"{self._consecutive_text_only} consecutive turns; aborting. "
                                f"(last text: {last_text[:80]})"
                            ),
                        },
                    )
                    forced_result = self.runtime.execute(
                        forced_call.name, forced_call.arguments
                    )
                    tool_log.append({
                        "step": step, "tool": forced_call.name,
                        "args": forced_call.arguments, "result": forced_result,
                        "forced_after_text_only": self._consecutive_text_only,
                    })
                    return TaskReport(
                        task, False, forced_call.arguments["summary"],
                        str(step), milestones, tool_log,
                        duration_s=round(time.monotonic() - t_start, 2),
                    )
                messages.append({"role": "assistant", "content": resp.text or "(no text)"})
                messages.append(
                    {
                        "role": "user",
                        "content": "Respond with exactly one tool call (use task_done to finish).",
                    }
                )
                continue
            # Reset text-only counter when LLM does emit a tool call.
            self._consecutive_text_only = 0

            # Execute only the first call, but echo ALL calls and answer each
            # one -- OpenAI-style APIs reject the next request if any
            # tool_call in the assistant message lacks a matching result.
            call = resp.tool_calls[0]

            # Preserve the user's terminal intent across LLM recovery. A plain
            # "pick up/grab X" means hold the object; it must never silently
            # become pick_and_place after an earlier grasp failure.
            if hold_only_request and call.name == "pick_and_place":
                original = dict(call.arguments or {})
                label = original.get("object") or requested_reflex.object_query
                trace = getattr(self.runtime, "trace", None)
                if trace is not None:
                    trace.record_event(
                        "llm_preserve_pickup_semantics",
                        {
                            "reason": "plain pick-up request must keep object held",
                            "original_call": {
                                "name": call.name, "arguments": original,
                            },
                            "replacement_call": {
                                "name": "grasp_object", "arguments": {"label": label},
                            },
                        },
                    )
                call = ToolCall("grasp_object", {"label": label})

            # Preserve an explicitly named basket across LLM translation.
            # The model often emits generic "box" for Chinese 筐, while the
            # detector vocabulary and physical fixture say "wicker basket".
            if call.name == "pick_and_place" and self._requested_container_label:
                normalized_args = dict(call.arguments or {})
                destination = " ".join(
                    str(normalized_args.get("destination", "")).lower().split()
                )
                if destination in {"", "box", "basket", "筐", "框", "篮子"}:
                    normalized_args["destination"] = self._requested_container_label
                    call = ToolCall(call.name, normalized_args, id=call.id)

            # A small LLM may declare a batch complete after the first fruit.
            # The runtime-owned pending list is authoritative.
            if (
                call.name == "task_done"
                and self._batch_pick_place
                and self._pending_batch_objects
                and self._batch_destination
            ):
                next_object = self._pending_batch_objects[0]
                self.runtime.trace.record_event(
                    "batch_task_continue",
                    {"next_object": next_object,
                     "remaining": list(self._pending_batch_objects)},
                )
                call = ToolCall("pick_and_place", {
                    "object": next_object,
                    "destination": self._batch_destination,
                })

            if (
                call.name == "task_done"
                and self._batch_pick_place
                and not self._pending_batch_objects
                and self._failed_batch_objects
            ):
                call = ToolCall("task_done", {
                    "success": False,
                    "summary": (
                        "Batch completed with failed objects: "
                        + ", ".join(self._failed_batch_objects)
                    ),
                })

            # The initial visible-object snapshot is authoritative for a batch.
            # Never chase remembered aliases or fruit newly seen inside the
            # basket after the original pending list has been exhausted.
            if (
                call.name == "pick_and_place"
                and self._batch_pick_place
                and self._pending_batch_objects is not None
            ):
                batch_args = dict(call.arguments or {})
                requested_object = " ".join(
                    str(batch_args.get("object", "")).lower().split()
                )
                if requested_object not in self._pending_batch_objects:
                    self.runtime.trace.record_event(
                        "batch_out_of_scope_pick_suppressed",
                        {
                            "requested_object": requested_object,
                            "remaining": list(self._pending_batch_objects),
                            "initially_visible_only": True,
                        },
                    )
                    if self._pending_batch_objects:
                        batch_args["object"] = self._pending_batch_objects[0]
                        call = ToolCall(call.name, batch_args, id=call.id)
                    else:
                        call = ToolCall("task_done", {
                            "success": not self._failed_batch_objects,
                            "summary": (
                                "Initial fruit list completed"
                                if not self._failed_batch_objects
                                else "Initial fruit list attempted; failed: "
                                + ", ".join(self._failed_batch_objects)
                            ),
                        })

            # Suppress only an exact duplicate object/destination pair.
            # Batch tasks must remain free to place other fruits into the same
            # basket; the old global boolean stopped after the first fruit.
            if call.name == "pick_and_place":
                pp_args = dict(call.arguments or {})
                pp_signature = (
                    " ".join(str(pp_args.get("object", "")).lower().split()),
                    " ".join(str(pp_args.get("destination", "")).lower().split()),
                )
                if pp_signature in self._completed_pick_places:
                    self.runtime.trace.record_event(
                        "duplicate_pick_and_place_suppressed",
                        {"signature": list(pp_signature), "original_call": pp_args},
                    )
                    if self._batch_pick_place and self._pending_batch_objects:
                        call = ToolCall("pick_and_place", {
                            "object": self._pending_batch_objects[0],
                            "destination": self._batch_destination or pp_signature[1],
                        })
                    else:
                        # Honest close: a batch with failed objects must not
                        # report success just because the duplicate was
                        # suppressed after the last successful placement.
                        batch_failed = list(self._failed_batch_objects or [])
                        call = ToolCall(
                            name="task_done",
                            arguments={
                                "success": not batch_failed,
                                "summary": (
                                    f"Duplicate placement of {pp_signature[0]!r} "
                                    f"to {(pp_signature[1] or 'the drop zone')!r} "
                                    "suppressed; that exact placement already succeeded."
                                    + (
                                        "; failed objects: "
                                        + ", ".join(batch_failed)
                                        if batch_failed else ""
                                    )
                                ),
                            },
                        )

            messages.append(
                {"role": "assistant", "content": resp.text, "tool_calls": list(resp.tool_calls)}
            )
            try:
                self.progress["current"] = {
                    "step": step, "tool": call.name,
                    "args": _short_args(call.arguments),
                }
            except Exception:
                pass
            result = self.runtime.execute(call.name, call.arguments)
            tool_log.append({"step": step, "tool": call.name, "args": call.arguments, "result": result})
            try:
                self._sync_progress_steps(tool_log)
                self.progress["current"] = None
            except Exception:
                pass

            if call.name == "list_objects" and self._batch_pick_place:
                container_words = ("basket", "box", "bin", "tray", "筐", "框", "盒")
                visible = []
                for obj in result.get("objects", []):
                    label = " ".join(str(obj.get("label", "")).lower().split())
                    if (
                        label
                        and obj.get("state") == "visible"
                        and not any(word in label for word in container_words)
                        and label not in visible
                    ):
                        visible.append(label)
                if self._pending_batch_objects is None:
                    self._pending_batch_objects = visible
                    self.runtime.trace.record_event(
                        "batch_pending_initialized", {"objects": visible}
                    )
                    # The frozen list IS the batch plan — surface it as the
                    # task milestones (decompose was skipped for batch).
                    try:
                        self.progress["milestones"] = [
                            f"place {name} into "
                            f"{self._batch_destination or self._requested_container_label or 'the container'}"
                            for name in visible
                        ]
                    except Exception:
                        pass
                    # Bootstrap fast-start: with the container named in the
                    # task text, the first pick needs no LLM decision.
                    if (
                        self._batch_bootstrap
                        and self._batch_auto_next is None
                        and visible
                    ):
                        self._batch_destination = (
                            self._batch_destination
                            or self._requested_container_label
                        )
                        self._batch_auto_next = ToolCall("pick_and_place", {
                            "object": visible[0],
                            "destination": self._batch_destination,
                        })

            harness = getattr(getattr(self.runtime, "arm", None), "harness", None)
            if harness is not None and harness.estopped:
                summary = "Task cancelled by STOP; current action aborted."
                self.runtime.trace.record_event(
                    "task_cancelled", {"tool": call.name, "step": step}
                )
                return TaskReport(
                    task=task, success=False, summary=summary, steps=step,
                    milestones=milestones, tool_log=tool_log,
                    duration_s=time.monotonic() - t_start,
                )

            if (
                hold_only_request
                and call.name == "grasp_object"
                and bool(result.get("ok"))
                and bool(result.get("held"))
            ):
                held = str(result.get("held"))
                summary = f"Picked up {held} and kept it held."
                self.runtime.trace.record_event(
                    "pickup_auto_complete",
                    {"held": held, "reason": "successful hold-only grasp"},
                )
                done_args = {"success": True, "summary": summary}
                done_result = self.runtime.execute("task_done", done_args)
                tool_log.append({
                    "step": step, "tool": "task_done", "args": done_args,
                    "result": done_result, "forced": True,
                })
                return TaskReport(
                    task=task,
                    success=True,
                    summary=summary,
                    steps=step,
                    milestones=milestones,
                    tool_log=tool_log,
                    duration_s=time.monotonic() - t_start,
                )

            # Remember the exact successful object/destination pair. Other
            # objects may still be placed at the same destination in a batch.
            if call.name == "pick_and_place" and bool(result.get("ok")):
                done_args = dict(call.arguments or {})
                done_object = " ".join(
                    str(done_args.get("object", "")).lower().split()
                )
                done_destination = " ".join(
                    str(done_args.get("destination", "")).lower().split()
                )
                self._completed_pick_places.add((done_object, done_destination))
                if self._batch_pick_place:
                    self._batch_destination = done_destination or self._batch_destination
                    if self._pending_batch_objects is not None:
                        self._pending_batch_objects = [
                            item for item in self._pending_batch_objects
                            if item != done_object
                        ]
                        if self._pending_batch_objects:
                            # Autopilot: the frozen list is authoritative, so
                            # the next fruit does not need an LLM decision.
                            self._batch_auto_next = ToolCall("pick_and_place", {
                                "object": self._pending_batch_objects[0],
                                "destination": (
                                    self._batch_destination or done_destination
                                ),
                            })
                self._consecutive_pick_failures = 0
            elif call.name == "pick_and_place" and not bool(result.get("ok")):
                failed_args = dict(call.arguments or {})
                failed_object = " ".join(
                    str(failed_args.get("object", "")).lower().split()
                )
                if self._batch_pick_place:
                    self._batch_destination = (
                        " ".join(str(failed_args.get("destination", "")).lower().split())
                        or self._batch_destination
                    )
                    if failed_object not in self._failed_batch_objects:
                        self._failed_batch_objects.append(failed_object)
                    if self._pending_batch_objects is not None:
                        self._pending_batch_objects = [
                            item for item in self._pending_batch_objects
                            if item != failed_object
                        ]
                    self._consecutive_pick_failures = 0
                else:
                    self._consecutive_pick_failures += 1
                if self._consecutive_pick_failures >= PICK_AND_PLACE_FAILURE_LIMIT:
                    # Force task_done: small local LLMs ignore the
                    # "fail-then-task_done" rule in prompts.py and retry
                    # pick_and_place until max_steps is hit (~2 min waste).
                    last_err = str(result.get("error", ""))[:200]
                    if self.runtime.trace is not None:
                        self.runtime.trace.record_event(
                            "llm_force_task_done",
                            {
                                "reason": (
                                    f"pick_and_place failed "
                                    f"{self._consecutive_pick_failures} times "
                                    "consecutively; forcing task_done"
                                ),
                                "last_result_error": last_err,
                            },
                        )
                    # Actually execute task_done (the prior pick_and_place
                    # has already run; the result var holds its outcome).
                    forced_call = ToolCall(
                        name="task_done",
                        arguments={
                            "success": False,
                            "summary": (
                                f"pick_and_place failed "
                                f"{self._consecutive_pick_failures} times in a row "
                                f"(last: {last_err[:80]}); aborting."
                            ),
                        },
                    )
                    forced_result = self.runtime.execute(
                        forced_call.name, forced_call.arguments
                    )
                    tool_log.append({
                        "step": step, "tool": forced_call.name,
                        "args": forced_call.arguments, "result": forced_result,
                        "forced_after_failures": self._consecutive_pick_failures,
                    })
                    if forced_result.get("task_complete"):
                        self.runtime.trace.finish(
                            f"task: {task}\n"
                            "success: false\n"
                            f"steps: {step}\n"
                            f"{forced_call.arguments['summary']}"
                        )
                        return TaskReport(
                            task, False, forced_call.arguments["summary"],
                            str(step), milestones, tool_log,
                            duration_s=round(time.monotonic() - t_start, 2),
                        )
                    # If task_done itself errored, fall through to next step
                    # (rare; we trust max_steps as the final backstop).
            elif call.name == "move_home":
                # moving home between attempts resets the failure streak
                self._consecutive_pick_failures = 0

            if call.name == "task_done" and result.get("task_complete"):
                summary = str(call.arguments.get("summary", ""))
                success = _as_bool(call.arguments.get("success", False))
                self.runtime.trace.finish(
                    f"task: {task}\nsuccess: {success}\nsteps: {step}\n{summary}"
                )
                return TaskReport(
                    task, success, summary, step, milestones, tool_log,
                    duration_s=round(time.monotonic() - t_start, 2),
                )

            ok = bool(result.get("ok"))
            consecutive_failures = 0 if ok else consecutive_failures + 1
            if self.advisor is not None:
                self.advisor.note_outcome(ok)

            content = json.dumps(result)
            messages.append(
                {"role": "tool", "tool_call_id": call.id or "call_0", "name": call.name, "content": content}
            )
            # Sliding window: keep intro + last N to bound context growth
            # (default N=8 ~= 4 assistant turns + 4 tool results). Each turn
            # appends up to 3 entries (assistant w/ tool_calls, tool, extras);
            # N=8 caps at ~3 turns of history which is plenty for short tasks.
            _MAX_HISTORY = 8
            if len(messages) > _MAX_HISTORY:
                messages = messages[:1] + messages[-( _MAX_HISTORY - 1):]

            for extra in resp.tool_calls[1:]:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": extra.id or "call_x",
                        "name": extra.name,
                        "content": json.dumps(
                            {"ok": False, "error": "skipped: one tool call per turn; re-issue if still needed"}
                        ),
                    }
                )

            # Failure escalation: one visual suggestion from the advisor.
            if not ok and self.advisor is not None and self.advisor.should_consult:
                jpeg = self.runtime.frame_jpeg()
                if jpeg:
                    suggestion = self.advisor.suggest(
                        task, jpeg, failure=str(result.get("error", ""))
                    )
                    if suggestion:
                        self.runtime.memory.add("note", f"advisor: {suggestion}")
                        messages.append(
                            {"role": "user", "content": f"Advisor suggestion: {suggestion}"}
                        )

            # Refresh the agent's situational context periodically.
            extra: dict = {}
            if self.attach_images and self.runtime.last_frame is not None:
                extra = {"images": [self.runtime.frame_jpeg()]}
            if not ok and consecutive_failures >= 2:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Two consecutive failures. Change strategy: re-observe, "
                            "try a different approach, or finish with task_done("
                            "success=false, honest summary).\nMemory:\n"
                            + self.runtime.memory.digest(max_lines=5)
                        ),
                        **extra,
                    }
                )

        self.runtime.trace.finish(f"task: {task}\nsuccess: false\nran out of steps ({self.max_steps})")
        return TaskReport(
            task, False, "step budget exhausted", self.max_steps, milestones, tool_log,
            duration_s=round(time.monotonic() - t_start, 2),
        )

    def _try_fast_path(self, task: str, t_start: float) -> tuple[TaskReport | None, str | None]:
        """Reflex/experience execution; (report, None) on success, or
        (None, note-for-the-LLM) when the fast attempt failed or no fast
        plan exists."""
        plan = self.fast_planner.plan(task)
        if plan is None:
            return None, None
        tool_log: list[dict] = []
        try:
            self.progress["path"] = str(plan.source)
            self.progress["milestones"] = [
                f"{name}({_short_args(args)})" for name, args in plan.calls
            ]
            self.runtime.trace.record_event("task_decomposition", {
                "source": str(plan.source),
                "task": task,
                "milestones": list(self.progress["milestones"]),
            })
        except Exception:
            pass
        for i, (name, args) in enumerate(plan.calls, start=1):
            try:
                self.progress["current"] = {"step": i, "tool": name,
                                            "args": _short_args(args)}
            except Exception:
                pass
            result = self.runtime.execute(name, args)
            tool_log.append({"step": i, "tool": name, "args": args, "result": result})
            try:
                self._sync_progress_steps(tool_log)
                self.progress["current"] = None
            except Exception:
                pass
            if not result.get("ok", False):
                self.fast_planner.note_outcome(
                    task, plan.calls, False, time.monotonic() - t_start
                )
                note = (
                    f"(A fast {plan.source} plan was tried first and FAILED at "
                    f"{name}({json.dumps(args)}): {str(result.get('error', ''))[:200]}. "
                    "Diagnose before retrying the same thing.)"
                )
                return None, note
        duration = round(time.monotonic() - t_start, 2)
        self.fast_planner.note_outcome(task, plan.calls, True, duration)
        summary = (
            f"done via {plan.source} path in {duration}s: "
            + "; ".join(f"{n}({_short_args(a)})" for n, a in plan.calls)
        )
        self.runtime.trace.finish(
            f"task: {task}\nsuccess: true\npath: {plan.source}\nduration_s: {duration}\n{summary}"
        )
        return (
            TaskReport(
                task, True, summary, len(plan.calls), [], tool_log,
                path=plan.source, duration_s=duration,
            ),
            None,
        )

    @staticmethod
    def _prune_images(messages: list[dict]) -> list[dict]:
        """Keep only the newest image in context: frames are ~50-100 KB of
        base64 each and every retained one is re-sent on every request."""
        last_with_images = None
        for i, m in enumerate(messages):
            if m.get("images"):
                last_with_images = i
        if last_with_images is None:
            return messages
        pruned = []
        for i, m in enumerate(messages):
            if m.get("images") and i != last_with_images:
                m = {k: v for k, v in m.items() if k != "images"}
            pruned.append(m)
        return pruned

    def _decompose(self, task: str) -> list[str]:
        try:
            resp: LLMResponse = self._chat_with_fallback(
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": DECOMPOSE_PROMPT.format(task=task)}],
                max_tokens=300,
            )
        except Exception:
            return []
        lines = []
        for raw in resp.text.splitlines():
            line = raw.strip().lstrip("0123456789.-* ").strip()
            if line:
                lines.append(line)
        return lines[:6]
