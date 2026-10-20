# SPEC: Local LLM Operations

**Status:** Draft 2026-08-11
**Owner:** wrc_demo maintainer
**Related:** [ADR-0006](./adr/0006-move-secrets-to-env-vars.md),
[ADR-0007](./adr/0007-local-llm-strategy.md),
[ADR-0008](./adr/0008-llm-profile-switching.md)

## Scope

Operational procedures for running wrc_demo with a **local LLM profile**
(`local_cosmos3`, `local_qwen`). Covers:

1. Model acquisition (download + quantization)
2. Server bring-up (vLLM)
3. Profile selection (`--llm <profile>`)
4. Runtime switching (`llm_switch` MCP tool)
5. Health and failure modes

Out of scope: cloud profiles (`anthropic`, `openai`, `minimax`) — see
`configs/llm/*.yaml` for their env-var requirements.

## Hardware target

| Tier | GPU | VRAM | Recommended profile |
|---|---|---|---|
| Dev (this rig) | RTX 5070 (Blackwell sm_120) | 8 GB | `local_cosmos3` (W4A16) |
| Prod | H100 / B200 | 80 GB+ | `local_qwen` (BF16) or `minimax` |

The `local_cosmos3` recipe assumes the dev tier. Bumping to the prod
tier with `local_cosmos3` works — just raise `--max-model-len` and
`--gpu-memory-utilization` and drop `--quantization`.

## Acquisition

### Step 1 — download

```bash
# One-time, ~9 GB, 30 min to 2 h depending on bandwidth.
# Token must have access to nvidia/Cosmos3-Edge (gated).
hf download nvidia/Cosmos3-Edge \
    --local-dir /home/seeed/wrc_demo/models/cosmos3-edge \
    --max-workers 8
```

Required env: `HF_TOKEN` (or `huggingface-cli login`).

Disk footprint: **9.13 GB** total (BF16). All 54 files; vLLM's weight
filter handles exclusion of the Generation tower at load time.

### Step 2 — quantize (dev tier only)

```bash
# ~5–15 min on RTX 5070. Output: ~3 GB on disk.
python /home/seeed/wrc_demo/scripts/quantize_cosmos3.py \
    --src /home/seeed/wrc_demo/models/cosmos3-edge \
    --dst /home/seeed/wrc_demo/models/cosmos3-edge-w4a16 \
    --verify
```

`--verify` runs a 5-token generation to confirm the quantization
didn't break anything (no NaN logits, sensible top-1). If verify
fails the script deletes the broken output; re-run after fixing
the cause.

Quantization recipe (encoded in `scripts/quantize_cosmos3.py`):

| Param | Value | Why |
|---|---|---|
| Algorithm | AWQ | Activation-aware Weight Quantization; the standard choice for LLMs |
| `w_bit` | 4 | Best size/quality trade-off |
| `q_group_size` | 128 | AWQ default; per-channel is more accurate but bigger |
| `zero_point` | True | Symmetric quantization loses ~1% quality at INT4 |
| Calib dataset | `pileval` | The AWQ-recommended calibration set |
| Exempted | `embed_tokens`, `lm_head`, vision encoder, Generation tower | These either dominate size (embed) or are precision-sensitive (lm_head, vision) |

### Step 3 — serve

```bash
bash /home/seeed/wrc_demo/scripts/serve_cosmos3_vllm.sh
```

Logs land in stdout (vLLM's default — `tail -f` if backgrounded).

**Critical flags** (do not change without re-reading ADR-0007):

| Flag | Reason |
|---|---|
| `--enable-auto-tool-choice` | Without this, tool calls land in `content` as raw text instead of the structured `tool_calls` field. The orchestrator's tool dispatch will silently fail. |
| `--tool-call-parser qwen3_xml` | Matches Cosmos3-Edge's chat template (byte-for-byte Qwen3-Coder). Other parsers (hermes, llama3_json) will misinterpret the XML protocol. |
| `--reasoning-parser qwen3` | The template defaults `enable_thinking=true`; without this parser, `<think>...</think>` text leaks into the tool-call arguments. |
| `--quantization awq_marlin` | Marlin INT4 GEMM kernels. Required for the W4A16 model. |
| `--limit-mm-per-prompt '{"image": 2, "video": 0}'` | Cap per-request images at 2; never accept video (we don't load the video processor anyway). The cap prevents vision-activation OOM. |
| `--mm-processor-kwargs '{"do_resize": true, "min_pixels": 4096, "max_pixels": 1048576}'` | Force the image down to ≤1024×1024 — anything bigger blows through VRAM. The default cap (16M pixels) is 16× too loose for 8 GB. |
| `--max-model-len 16384` | Native context is 131 K, which needs 15 GB just for KV cache. 16 K is the largest power-of-two-ish setting that fits in 8 GB. |

### Memory budget (dev tier)

| Component | Bytes | Notes |
|---|---|---|
| Reasoner W4A16 weights | ~2.75 GB | Linear layers INT4, embeddings/lm_head BF16 |
| Vision encoder (BF16) | ~0.83 GB | Always loaded; required for advisor |
| Projector (BF16) | ~0.15 GB | Vision → LM bridge |
| KV cache (16 K context) | ~1.5 GB | BF16 KV, 8 heads × 128 dim × 28 layers |
| Subtotal (static) | **~5.23 GB** | — |
| Available for activations | ~2.77 GB | 8 GB total − static, with `--gpu-memory-utilization 0.92` |
| Per-request activations | ~0.5–1.5 GB | Depends on batch size, image size, seq len |

If you see OOM in vLLM logs, the first knob to turn is
`--max-model-len` (lower) or `--gpu-memory-utilization` (raise, up to
0.95). If still OOM, switch to `--enable-layerwise-offload` (slower
but reliable).

## Profile selection

### Cold start

```bash
python -m wrc_demo.apps.demo \
    --llm local_cosmos3 \
    --cameras orbbec_overhead \
    --arm rebot_rs \
    --llm-fallback local_qwen,mock   # optional; see ADR-0008
```

`--llm <profile>` is required and must match a file in
`configs/llm/<profile>.yaml`. Unknown profile → `make_llm()` raises
`LLMConfigError`.

### Runtime switch

Via MCP (the operator-facing surface):

```json
{"tool": "llm_switch", "arguments": {"profile": "local_cosmos3"}}
{"tool": "llm_status", "arguments": {}}
```

`llm_switch` re-derives the LLMClient from the current `cfg` and
swaps `orchestrator.llm`. The next `chat()` uses the new client.
The trace records `"llm_switch"` events.

### Fallback chain

Set via `cfg.get("llm.fallback", [])` (a list of profile names).
When `chat()` raises a transient SDK exception (network, auth,
timeout), the orchestrator walks the chain in order. Stops on
success or chain exhaustion. The last exception propagates if the
chain is exhausted.

`LLMConfigError` (missing env var) does **not** trigger fallback —
misconfiguration must surface.

## Health and monitoring

### Startup probe

`build_runtime()` does a 1-token `chat("ping", [{role:user, content:ping}])`
call after constructing the LLM. On failure, the profile is excluded
from the fallback chain for this session. Prevents "fallback to a dead
local server" failures.

The probe is **fast** (1 token, ~50–200 ms on a local server) and
**best-effort** — a false negative just means one wasted retry, not
a hang.

### Runtime signals

| Signal | Where | Action |
|---|---|---|
| `trace.jsonl::"llm_switch"` events | `runs/<ts>/trace.jsonl` | Audit who switched when |
| `trace.jsonl::"llm_fallback"` events | `runs/<ts>/trace.jsonl` | Transient failures; check chain |
| vLLM TTFT | vLLM stdout (log file) | >2 s on local Cosmos3-Edge → investigate quantization |
| vLLM throughput | vLLM stdout | <5 tok/s on dev tier → check `--quantization awq_marlin` is set |
| orchestrator `max_steps` exhaustion | `TaskReport(success=False)` | Model couldn't converge; check `trace.jsonl` for stuck reasoning |

### Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `LLMConfigError(minimax, MINIMAX_API_KEY)` at startup | Env var not set | `export MINIMAX_API_KEY=...`; see ADR-0006 |
| vLLM OOM at `--max-model-len 16384` | KV cache too large for 8 GB | Lower to 8192, or add `--enable-layerwise-offload` |
| Tool calls land in `content` as raw text | Missing `--enable-auto-tool-choice` | Restart vLLM with the flag (see ADR-0007) |
| `<think>...</think>` text inside `tool_calls[].function.arguments` | Missing `--reasoning-parser qwen3` | Restart vLLM with the flag |
| `local_cosmos3` profile not in `--help` choices | Profile file not in `configs/llm/` | Verify `configs/llm/local_cosmos3.yaml` exists |
| Mid-task profile switch "loses" context | New LLM doesn't see the old model's reasoning tokens | Use `llm_switch` only at logical break points (after `task_done`); see ADR-0008 |

## Testing

### Offline (no server required)

```bash
# All wrc_demo unit tests should pass without any LLM server.
# They use MockLLM by default (see tests/conftest.py::demo_cfg).
cd /home/seeed/wrc_demo
python -m pytest tests/ -q -m "not cosmos3"
```

Expected: **281+ passed, 0 failed** (CLAUDE.md baseline).

### Online smoke test (decisive for ADR-0007)

```bash
# Server must be running first.
bash scripts/serve_cosmos3_vllm.sh &

# Run the 4-step smoke test.
cd /home/seeed/wrc_demo
python -m pytest tests/test_cosmos3_smoke.py -v -m cosmos3
```

Expected:
- All 4 tests pass → Cosmos3-Edge is a viable local LLM.
- Test ① fails → STOP and pick fallback per ADR-0007 §"Fallback plan".

### Per-profile unit tests

```bash
cd /home/seeed/wrc_demo
python -m pytest tests/test_profiles.py -v
```

Covers `make_llm()` construction for each profile, env-var loading
per ADR-0006, and the fallback chain construction per ADR-0008.

### End-to-end

```bash
# Real demo rig, mock stack first (no CAN bus needed).
python -m wrc_demo.apps.demo \
    --llm local_cosmos3 \
    --cameras mock \
    --arm mock \
    --task "pick up the banana and put it in the box" \
    --no-view --no-serve

# Then on the real rig.
python -m wrc_demo.apps.demo \
    --llm local_cosmos3 \
    --cameras orbbec_overhead \
    --arm rebot_rs \
    --interactive
```

## Versioning

The recipe is pinned:

| Component | Version |
|---|---|
| `nvidia/Cosmos3-Edge` | revision `6f58f6b4c91288838e60b6bcb2cc45d997e961de` (initial release, 2026-07-20) |
| `autoawq` | 0.2.9 (compat with torch 2.10) |
| `vllm` | ≥ 0.10.0 (Cosmos3-Edge merged in PR #48291, July 2026) |
| AWQ `q_group_size` | 128 |
| AWQ `w_bit` | 4 |
| Calib dataset | `pileval` |

When NVIDIA pushes a new revision to `main`, the quantization
output **must be regenerated** before reusing the old server.
Pinning happens in two places:
1. The HF download: `hf download ... --revision <sha>` (TODO: add to
   `scripts/quantize_cosmos3.py`)
2. The YAML `model:` field points at the local quantized dir, not at
   HF, so NVIDIA updates don't accidentally change the live model.

## Future work

- **Streaming output** (CLAUDE.md Gotcha §"no streaming"). Add
  `stream=True` to the OpenAI client + an SSE consumer in
  `AgentOrchestrator` to surface partial responses to the dashboard.
- **Vision-only fine-tune.** The Physical AI focus of Cosmos3-Edge
  makes it a candidate for LoRA on wrc_demo-specific scenes.
- **Quantized mmproj.** The vision encoder is BF16 (0.83 GB).
  INT8-quantizing it would free ~0.4 GB. Out of scope for ADR-0007.
