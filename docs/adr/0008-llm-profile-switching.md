# ADR-0008: Runtime LLM profile switching and per-profile fallback

**Status:** Accepted 2026-08-11
**Deciders:** wrc_demo maintainer
**Depends on:** [ADR-0006 — secrets in env vars](./0006-move-secrets-to-env-vars.md),
[ADR-0007 — local LLM strategy](./0007-local-llm-strategy.md)

## Context and Problem Statement

Today, switching the LLM requires restarting the demo
(`--llm <profile>` is a CLI flag processed at startup; see
`apps/demo.py:282-289`). A real pick-and-place run may need to swap
mid-session:

- The cloud profile (`anthropic`/`openai`) is rate-limited mid-task;
  we want to fail over to a local profile without losing the held
  object or restarting the world watcher.
- A new local profile (`local_cosmos3`) is added; we want to
  A/B-test it against `local_qwen` without bouncing the rig.
- The operator wants to inspect "what would anthropic have done
  here?" by replaying the trajectory against a different profile.

Currently impossible: the `LLMClient` is constructed once in
`AgentOrchestrator.__init__` and never replaced.

Worse, **there is no fallback on LLM failure.** `orchestrator.py:121`
calls `llm.chat(...)` inside the main loop without a try/except;
any HTTP error or auth failure raises straight out of `run_task()`,
leaving the held object dangling and the watcher running with no
agent in charge.

## Decision

1. **Add `AgentOrchestrator.switch_llm(new_profile_name: str)`** that
   constructs a fresh `LLMClient` from `cfg`, swaps `self.llm`, and
   records the switch in the trace logger. Cheap because the new
   client reuses the same `_call_with_param_fallback` machinery.

2. **Add a `fallback_chain: list[str]` field to `AgentOrchestrator`**
   (loaded from `cfg.get("llm.fallback", [])` at construction). When
   the primary `llm.chat()` raises a transport / auth / timeout
   exception, the orchestrator walks the chain in order, swapping
   `self.llm` on each attempt. Stops on success or chain exhaustion.
   **Does not catch `LLMConfigError`** — misconfiguration must be
   surfaced, not silently masked.

3. **Expose both via MCP** as `llm.switch` and `llm.status` tools.
   MCP is the operator-facing surface; CLI flag stays for
   cold-start selection. New tools go in
   `_EXTRA_TOOLS` in `apps/mcp_server.py:55-105` (existing convention).

4. **Add `tests/test_profile_switching.py`** that uses
   `ExplodingLLM` (already exists in `tests/test_streaming_fastpath.py:307`)
   to verify:
   - 1-step chain: exploding → mock → completes the task.
   - Empty chain: exploding raises.
   - Mid-loop switch: primary mock, switch to mock with a different
     script after 3 tool calls, the new script drives the next 3.
   - `LLMConfigError` (bad env var) propagates immediately, even if
     a fallback is configured.

5. **Per-profile health probe at startup.** `make_llm()` is wrapped
   by `build_runtime()` to do a 1-token `chat()` call. On failure,
   the profile is marked unhealthy in the runtime state and excluded
   from the fallback chain for this session. Prevents a "fallback to
   a dead profile" failure mode.

## Implementation

### AgentOrchestrator changes

```python
# src/wrc_demo/agent/orchestrator.py

class AgentOrchestrator:
    def __init__(self, llm, ..., fallback_chain=None, cfg=None, ...):
        self.llm = llm
        self.fallback_chain = list(fallback_chain or [])
        self.cfg = cfg
        # ... existing fields ...

    def switch_llm(self, profile_name: str) -> LLMClient:
        """Swap the live LLM client for a new one built from cfg.
        Records the switch in the trace; the next chat() uses the new
        client. Returns the new client for confirmation."""
        from .llm import make_llm
        new_llm = make_llm(self.cfg._data["llm"])  # re-derive from current cfg
        # If profile_name refers to a sub-profile, swap cfg.llm first:
        self.cfg._data["llm"]["profile"] = profile_name
        new_llm = make_llm(self.cfg._data["llm"])
        self.llm = new_llm
        self.trace.record_event("llm_switch", {"profile": profile_name})
        return new_llm

    def _chat_with_fallback(self, system, messages, tools=None, max_tokens=1024):
        """Try self.llm first; on transport/auth/timeout, walk fallback_chain."""
        import openai  # the exception types live in the SDK
        transient_errors = (
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.RateLimitError,
            openai.AuthenticationError,
            openai.InternalServerError,
        )
        chain = [self.llm] + [self._client_for(p) for p in self.fallback_chain]
        last_exc = None
        for candidate in chain:
            try:
                return candidate.chat(system, messages, tools=tools, max_tokens=max_tokens)
            except transient_errors as e:
                last_exc = e
                self.trace.record_event("llm_fallback", {"from": candidate, "err": str(e)[:200]})
                continue
        raise last_exc  # chain exhausted; propagate the last transient error

    def _client_for(self, profile_name: str) -> LLMClient:
        from .llm import make_llm
        sub_cfg = dict(self.cfg._data["llm"])
        sub_cfg["profile"] = profile_name
        # NOTE: this re-reads env vars; profile must follow ADR-0006
        return make_llm(sub_cfg)
```

### build_runtime changes

```python
# src/wrc_demo/apps/demo.py

fallback_chain = list(cfg.get("llm.fallback", []))
agent = AgentOrchestrator(
    llm, runtime, advisor=advisor, max_steps=30,
    decompose=True, fast_planner=FastPlanner(experience),
    fallback_chain=fallback_chain, cfg=cfg,
)
```

### Per-profile health probe

```python
# In build_runtime(), after llm is constructed:
def _probe(llm):
    try:
        llm.chat("ping", [{"role": "user", "content": "ping"}], max_tokens=1)
        return True
    except Exception:
        return False

if not _probe(llm):
    log.warning("LLM profile %s failed health probe; excluding from chain", llm_profile_name)
    fallback_chain = [p for p in fallback_chain if p != llm_profile_name]
```

### MCP extras

```python
# apps/mcp_server.py::_EXTRA_TOOLS — add:
{
    "name": "llm_switch",
    "description": "Swap the active LLM profile mid-session. Returns the new profile name.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "profile": {"type": "string", "enum": ["mock", "anthropic", "openai",
                                                    "local_qwen", "local_cosmos3",
                                                    "minimax"]}
        },
        "required": ["profile"],
    },
},
{
    "name": "llm_status",
    "description": "Report the active LLM profile and the fallback chain (with health).",
    "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
},
```

## Consequences

### Positive

- **Resilient to transient LLM failures.** A 503 from
  `api.minimax.chat` no longer kills the task; the agent falls
  over to `local_cosmos3` and keeps going.
- **A/B testing without restart.** MCP `llm_switch` lets the
  operator try `local_cosmos3` mid-session to see if it handles the
  current scene better than the current profile.
- **Per-profile health probing.** Catches the "fallback to a dead
  local server" failure mode before it wastes user time.
- **One mechanism for two needs.** Both runtime switch and fallback
  use the same `make_llm(cfg)` path; they compose.

### Negative

- **`make_llm()` is called more often.** Currently once per session;
  now once per switch + once per chain entry. Each call re-reads env
  vars and re-instantiates the underlying SDK client. Cost is
  ~10-50 ms (negligible).
- **Switching mid-loop loses the previous conversation's
  accumulated reasoning.** The new LLM sees the same `messages`
  history but its own reasoning tokens are not present. For tool
  calls this is fine (the tool args are in the history); for
  free-form reasoning, the new model has to reconstruct context.
- **Health probe can false-negative.** A 1-token `chat()` is fast
  but not exhaustive; some profiles work for `chat()` but fail on
  long-context calls. Mitigation: the fallback catches the failure
  on the actual call anyway, so a false-negative only costs one
  retry, not a hang.
- **`LLMConfigError` propagates from `make_llm()` itself.** If a
  fallback profile has a missing env var, the chain construction
  raises before any chat happens. Acceptable: misconfiguration
  should fail loud, not silently swap to a profile that wasn't
  supposed to run.

### Neutral

- **Backward compatible.** Existing code that doesn't pass
  `fallback_chain` gets an empty chain, which is the same behavior
  as today (no fallback, raises on failure).
- **CLI flag stays.** `--llm <profile>` still controls the cold-start
  selection. `llm_switch` is for runtime overrides only.
- **Profile name is a string, not an enum.** `make_llm()` already
  accepts any `type:` string; new profiles slot in without code
  changes.

## Verification

- `tests/test_profile_switching.py::test_fallback_chain_one_step` —
  primary LLM raises `APIConnectionError`, fallback succeeds, task
  completes.
- `tests/test_profile_switching.py::test_empty_chain_propagates` —
  no fallback configured, primary's exception bubbles up.
- `tests/test_profile_switching.py::test_mid_loop_switch` — switch
  to a new mock after 3 tool calls; the new mock's script drives
  the next 3.
- `tests/test_profile_switching.py::test_llm_config_error_propagates` —
  misconfigured profile raises `LLMConfigError` from `make_llm()`
  even if a fallback exists.
- `tests/test_profile_switching.py::test_health_probe_excludes_dead_profile` —
  profile that 1-token-probes as dead is removed from chain.

## Cross-references

- ADR-0006: the env-var loading pattern means fallback profiles
  must have their env vars set too. If not, the chain fails to
  construct — surfacing the misconfig.
- ADR-0007: `local_cosmos3` is the canonical example of a fallback
  target (when the cloud profile is rate-limited, fall over to the
  local 4B model).
- `apps/mcp_server.py::_EXTRA_TOOLS`: existing convention for
  operator-facing MCP extensions.
