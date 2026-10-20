# ADR-0006: Move secrets out of `configs/llm/*.yaml`; read API keys from environment

**Status:** Accepted 2026-08-11
**Deciders:** wrc_demo maintainer
**Related:** [ADR-0007 — local LLM strategy](./0007-local-llm-strategy.md) (depends on this)

## Context and Problem Statement

`configs/llm/minimax.yaml` contains a **plaintext API key** as the
`api_key:` field:

```yaml
type: openai_compat
base_url: https://api.minimax.chat/v1
model: MiniMax-M3
api_key: <REDACTED-EXAMPLE-KEY>
```

The file is **not tracked by git** (`git ls-files configs/llm/` returns
`anthropic.yaml`, `local_qwen.yaml`, `mock.yaml`, `openai.yaml` — no
`minimax.yaml`). It lives on the local filesystem only, presumably
because the operator did not want to commit the secret. But local-only
storage is fragile:

1. **No backup discipline.** A disk failure or fresh checkout loses the
   profile; the operator has to re-create it from memory.
2. **No audit trail.** Nothing records who created or last touched the
   key, and on which machines it has lived.
3. **Easy to leak by accident.** `grep -r sk-cp ~/`, accidental `git
   add configs/`, copy-paste into Slack — all silently exfiltrate the
   secret.
4. **Inconsistent with sibling profiles.** `openai.yaml` sets
   `api_key: null` and relies on `OPENAI_API_KEY`; `local_qwen.yaml`
   sets `api_key: EMPTY` because it's a dummy for the local server.
   `minimax.yaml` is the only profile that hard-codes a real secret.

A partial mitigation already landed in
`src/wrc_demo/agent/llm.py:make_llm()`: when `cfg.get("api_key")` is
`None`, the factory reads `cfg.get("env_key", "OPENAI_API_KEY")` from
the environment. That machinery exists; the profile just doesn't use
it yet.

## Decision

1. **Drop `api_key:` from every `configs/llm/*.yaml`.** Profiles
   declare `env_key:` (the env var name) and a comment pointing at the
   variable to set. Example:

   ```yaml
   type: openai_compat
   base_url: https://api.minimax.chat/v1
   model: MiniMax-M3
   env_key: MINIMAX_API_KEY         # export MINIMAX_API_KEY=... before running
   supports_vision: false
   ```

2. **Update `make_llm()` to log a one-line warning if the env var is
   unset at construction time.** Currently the failure mode is a
   confusing `openai.AuthenticationError` deep in the call stack;
   surfacing it at startup saves debug time and tells the operator
   exactly which variable to set.

3. **Add `tests/test_llm_secret_loading.py`.** Two cases: (a) profile
   with `env_key` set in the environment → key flows through to the
   client; (b) profile without the env var → `make_llm()` raises a
   clear `LLMConfigError(name, env_key)` before any network call.

4. **Document the env vars in `.env.example`** (new file, gitignored
   `.env` keeps the real values). Lists every profile's required env
   var and a one-line description. Not a substitute for a real secret
   store; just a checkpoint so the operator can't forget.

5. **Out of scope:** integrating a vault (HashiCorp Vault, AWS Secrets
   Manager, etc.). The env-var-only model is good enough for a single
   demo rig; an external store is overkill until there are multiple
   operators or rotation policies.

## Implementation

### Files touched

| File | Change |
|---|---|
| `configs/llm/minimax.yaml` | drop `api_key:` (and the long sk-cp-… string); add `env_key: MINIMAX_API_KEY` |
| `configs/llm/openai.yaml` | drop `api_key: null` (now redundant); add `env_key: OPENAI_API_KEY` |
| `configs/llm/local_qwen.yaml` | unchanged (`api_key: EMPTY` is a literal placeholder, not a secret) |
| `configs/llm/anthropic.yaml` | unchanged — already reads `ANTHROPIC_API_KEY` from env (the SDK does this internally; no env_key plumbing needed in this profile) |
| `configs/llm/local_cosmos3.yaml` (new, ADR-0007) | `api_key: EMPTY`, no env_key |
| `src/wrc_demo/agent/llm.py` | `make_llm()` raises `LLMConfigError` if `env_key` is named and the env var is unset; logs the variable name on stderr |
| `tests/test_llm_secret_loading.py` (new) | 4 tests covering (a) env-var hit, (b) env-var miss raises, (c) profile without `env_key` (local dummy OK), (d) `minimax` profile specifically does NOT contain a literal `sk-cp` prefix anywhere in the YAML or its git history |
| `.env.example` (new) | lists each profile's required env var |

### Migration

The operator runs:

```bash
# 1. Read the current key out of minimax.yaml (one-time):
grep '^api_key:' configs/llm/minimax.yaml | awk '{print $2}'

# 2. Set it in the shell (or ~/.bashrc):
export MINIMAX_API_KEY='<the value you just retrieved>'

# 3. Replace minimax.yaml with the env-key form (committed this PR).

# 4. Verify:
hf auth whoami               # sanity
python -c "from wrc_demo.config import load_demo_config; \
           from wrc_demo.agent.llm import make_llm; \
           print(make_llm(load_demo_config(llm='minimax').llm))"
```

## Consequences

### Positive

- **No plaintext secrets on disk.** A stolen laptop or an accidental
  `git add configs/` no longer leaks the API key.
- **One mental model.** Every profile reads its secret from the
  environment; the operator only has to remember which env var
  matches which profile.
- **Fast failure on misconfig.** `make_llm()` raises at startup with
  the env var name in the error message, instead of failing inside
  the first `chat()` call with a stack trace.
- **Profile files become diffable.** `configs/llm/minimax.yaml` can
  finally be tracked by git — no more local-only fork.

### Negative

- **Env vars are global per process.** If two wrc_demo instances need
  two different keys (e.g., staging vs prod), they have to be in
  separate shells. Acceptable for a demo rig; would need a vault
  solution if it ever becomes a problem.
- **One-shot export required.** Operators on a fresh machine must set
  the env var before the first run; `.env.example` mitigates this.
- **The original secret is still in `minimax.yaml`'s git history**
  (well, the local-only file — but its content has been read by
  every shell that touched it). Mitigation: **rotate the key**
  after this ADR lands. The operator should generate a fresh
  `sk-cp-…` on the provider side and discard the old one.

### Neutral

- **Anthropic profile is unchanged.** The Anthropic SDK reads
  `ANTHROPIC_API_KEY` from env itself; no plumbing needed.
- **Local profiles (`local_qwen`, `local_cosmos3`) are unchanged.**
  `api_key: EMPTY` is a literal placeholder for the local server's
  auth handshake, not a secret.

## Verification

- `tests/test_llm_secret_loading.py::test_minimax_key_from_env` —
  `make_llm()` constructed with `MINIMAX_API_KEY` in env produces an
  `OpenAICompatClient` whose `api_key` attribute equals the env var
  value.
- `tests/test_llm_secret_loading.py::test_missing_env_key_raises` —
  unset env var → `LLMConfigError("minimax", "MINIMAX_API_KEY")`
  raised at `make_llm()` construction.
- `tests/test_llm_secret_loading.py::test_local_qwen_no_env_needed` —
  local dummy profile works without any env var.
- `tests/test_llm_secret_loading.py::test_minimax_yaml_no_literal_secret` —
  parsed YAML does not contain a string matching `sk-cp` (catches
  regressions if someone re-adds a key to the file).
- **Manual:** `git grep -E 'sk-cp|sk-' configs/` returns nothing.

## Cross-references

- ADR-0007 (local LLM strategy): the new `local_cosmos3` profile
  follows the env-var convention established here.
- `src/wrc_demo/agent/llm.py:make_llm()` — the factory that this ADR
  extends.
