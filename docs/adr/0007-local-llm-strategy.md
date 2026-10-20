# ADR-0007: Add `local_cosmos3` profile — Cosmos3-Edge 4B as a local LLM

**Status:** Accepted 2026-08-11
**Deciders:** wrc_demo maintainer
**Depends on:** [ADR-0006 — secrets in env vars](./0006-move-secrets-to-env-vars.md)
**Supersedes:** the implicit "local_qwen is the only local LLM" assumption
in `CLAUDE.md` §"Known unresolved config/code mismatch"

## Context and Problem Statement

The `local_qwen` profile (`configs/llm/local_qwen.yaml`) is the only
on-device LLM option today. It has been half-dead since 2026-07-20:

| Symptom | Root cause |
|---|---|
| YAML `model: Qwen3VL-30B-A3B-Instruct-Q4_K_M` | `scripts/serve_qwen_*.sh` actually serve `Qwen/Qwen3.6-27B`; llama.cpp is permissive about names, **vLLM is strict and rejects the mismatch** |
| `scripts/serve_qwen_vllm.sh` has no multimodal flag | vLLM silently drops images; advisor escalation breaks |
| `scripts/serve_qwen_vllm.sh` has no `--enable-auto-tool-choice` | tool calls land as raw text instead of structured `tool_calls` |
| Qwen3.6-27B BF16 needs ~54 GB | doesn't fit any consumer GPU |

Even if all four are fixed, **Qwen3.6-27B is a heavyweight model** for
a 6-DoF tabletop arm that does mostly pick-and-place. The
reasoning-per-watt ratio is poor.

`nvidia/Cosmos3-Edge` (released 2026-07-20) is a candidate:

- **4B parameters** (vs Qwen3.6-27B's ~27B)
- **Physical AI focus** — robot/AV reasoning, well-aligned with the
  wrc_demo task distribution
- **Reasoner tower is text-only at inference** — drops the 4.5 GB
  diffusion/VAE weights that the chat path never touches
- **`chat_template.jinja` is byte-for-byte the Qwen3-Coder XML tool
  protocol** — proven tool calling format, vLLM ships a
  `qwen3_xml` parser out of the box
- **Blackwell support** (sm_120 = RTX 5070) — verified in HF PBR
  benchmarks on RTX PRO 4500 / 6000 Blackwell

The catch: **no public evidence that anyone has used Cosmos3-Edge for
tool calling**, even though the template supports it. NVIDIA
inherited the Qwen3-Coder chat template when training; whether the
weights themselves were SFT'd for tool use is unverified.

## Decision

1. **Add `configs/llm/local_cosmos3.yaml`** as a new profile. OpenAI-
   compatible (`type: openai_compat`), points at a vLLM server on
   `127.0.0.1:8080`, with the four flags the Qwen3-Coder protocol
   needs (`--enable-auto-tool-choice --tool-call-parser qwen3_xml
   --reasoning-parser qwen3`).

2. **Quantize to AWQ INT4 (W4A16) for the 8 GB target.** Per the
   model card, "Only BF16 precision is tested" — quantization is a
   deviation from the tested path and must be validated before
   production. The `scripts/quantize_cosmos3.py` script encodes the
   quantization recipe.

3. **Smoke-test before committing.** A 30-minute Go/No-Go test on 4
   progressively harder scenarios determines whether the template
   capability is matched by actual capability:
   - ① 2 tools + plain text → does the model return structured
     `tool_calls`?
   - ② All 21 TOOL_SPECS → can it pick the right one?
   - ③ Add an image → does VLM + tool coexist?
   - ④ Multi-turn `tool_response` loop → does the conversation
     converge?

   **① failure is a hard stop.** If the model can't generate a
   structured tool call with a 2-tool toy setup, the whole profile
   is replaced by a fallback (see §"Fallback plan" below).

4. **Pin to a single quantization format and revision.** Update
   `local_cosmos3.yaml` to reference a specific HF revision hash
   (not `main`); quantization produces a separate repo
   (`your-org/Cosmos3-Edge-W4A16` or local mirror). Avoids silent
   drift when NVIDIA pushes new commits.

5. **Coexist with `local_qwen`.** Do not delete `local_qwen.yaml`
   even if `local_cosmos3` works. The two profiles serve different
   hardware budgets (Qwen3.6-27B = 24+ GB; Cosmos3-Edge = 8 GB), and
   some operators may have the larger GPU.

## Implementation

### New files

| Path | Purpose |
|---|---|
| `configs/llm/local_cosmos3.yaml` | Profile: `base_url: http://127.0.0.1:8080/v1`, `model: Cosmos3-Edge-W4A16`, `env_key` not needed (local dummy) |
| `scripts/quantize_cosmos3.py` | AWQ INT4 quantization pipeline (CLI entry point: `python scripts/quantize_cosmos3.py --src <bf16-dir> --dst <w4a16-dir>`) |
| `scripts/serve_cosmos3_vllm.sh` | vLLM startup with the four tool-calling flags + Blackwell-aware memory flags |
| `tests/test_cosmos3_smoke.py` | 4-step Go/No-Go smoke test (gated by `pytest -m cosmos3`) |

### New profile content

```yaml
# Local Cosmos3-Edge (nvidia/Cosmos3-Edge, 4B, AWQ W4A16).
# Hardware target: RTX 5070 (Blackwell sm_120), 8 GB VRAM.
# Start the server with scripts/serve_cosmos3_vllm.sh.
type: openai_compat
base_url: http://127.0.0.1:8080/v1
model: Cosmos3-Edge-W4A16            # quantized; see scripts/quantize_cosmos3.py
api_key: EMPTY                       # vLLM accepts any non-empty string
supports_vision: true                # Reasoner tower is text+image; video tower not loaded
temperature: 0.2
```

### Quantization recipe (in `scripts/quantize_cosmos3.py`)

```python
# Quantize with AWQ Marlin for Blackwell-friendly INT4 GEMM kernels.
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model = AutoAWQForCausalLM.from_pretrained(
    SRC_DIR, safetensors=True,
    trust_remote_code=False,          # standard Qwen2-like arch, no remote code needed
)
tokenizer = AutoTokenizer.from_pretrained(SRC_DIR, trust_remote_code=False)

quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM",                # use Marlin kernels at inference
}
model.quantize(tokenizer, quant_config=quant_config, calib_data="pileval")
model.save_pretrained(DST_DIR)
```

### Server command (in `scripts/serve_cosmos3_vllm.sh`)

```bash
vllm serve "${QUANT_DIR:-Cosmos3-Edge-W4A16}" \
    --host 0.0.0.0 --port 8080 \
    --quantization awq_marlin \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.92 \
    --limit-mm-per-prompt '{"image": 2, "video": 0}' \
    --mm-processor-kwargs '{"do_resize": true, "min_pixels": 4096, "max_pixels": 1048576}' \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3
```

### Fallback plan (if smoke ① fails)

In priority order, picking exactly one:

1. **Swap to a Qwen3-Coder fine-tune.** Same `qwen3_xml` parser,
   same chat template lineage, but weights are explicitly SFT'd for
   tool use. Candidates: `Qwen/Qwen3-Coder-4B-Instruct-AWQ`,
   `Qwen/Qwen3-Coder-1.7B-Instruct-AWQ` (smaller, more headroom on
   8 GB). Switching cost: 1 file change in
   `local_cosmos3.yaml` (`model:` field).

2. **Two-model split.** Cosmos3-Edge handles vision (the task it's
   designed for); a smaller tool-calling model (Qwen3-Coder-1.7B or
   `gpt-oss-20b`) handles dispatch. Cost: new `local_cosmos3_vision`
   + `local_cosmos3_dispatch` profiles, plus routing logic in
   `AgentOrchestrator._decompose()` to call the vision model first.

3. **Text-based tool dispatch (NOT recommended).** Prompt the model
   to emit `<tool_call>` blocks as text and parse with a regex.
   Fragile; only viable if the model emits the format but the parser
   doesn't recognize it. Skip if ① fails outright.

4. **Reflex-only.** Drop the LLM tier entirely; rely on
   `reflex._RULES` + `ExperienceMemory` for everything. Already
   implemented, already works for the routine commands. Bad fit for
   novel tasks.

## Consequences

### Positive

- **8 GB fits.** W4A16 brings the chat path down to ~2.75 GB; KV
  cache for 16K context is ~1.5 GB; leaves ~3 GB headroom for the
  vision encoder activations. No consumer GPU upgrade required.
- **OpenAI-compatible.** No new client code; reuse
  `OpenAICompatClient`. The 21 TOOL_SPECS work as-is.
- **Reasoning model.** `enable_thinking=true` (default in the
  template) gives chain-of-thought before each tool call — should
  help tool selection at the cost of some latency. A
  `chat_template_kwargs={"enable_thinking": false}` opt-out is
  available for time-sensitive tasks.
- **Qwen3-Coder protocol.** A widely-deployed parser means community
  examples, blog posts, and known failure modes. Less unknown
  territory than a custom protocol would be.

### Negative

- **Tool-calling is unverified.** Until the smoke test passes,
  everything above is theoretical. The smoke test is **the gating
  event** for this ADR.
- **AWQ quantization is unsupported upstream.** The model card says
  only BF16 is tested. INT4 behavior may drift; we accept the risk
  in exchange for fitting on 8 GB.
- **Quantized model lives outside `nvidia/Cosmos3-Edge`.** Either
  push to your own HF org, or keep on local disk. The local-disk
  option is fine for a single demo rig but does not survive a fresh
  checkout.
- **NVIDIA may break the chat template.** Inheriting Qwen3-Coder's
  template is convenient now, but a future NVIDIA release could
  fork the protocol. The pinned-revision-hash mitigation limits the
  blast radius, but is not zero.

### Neutral

- **Two local profiles coexist.** Operators pick `--llm local_cosmos3`
  or `--llm local_qwen` based on their GPU. The dispatcher is
  profile-agnostic.
- **The model card's `--enable-layerwise-offload` is not used.** 4.87
  GB BF16 fits on 8 GB with `--gpu-memory-utilization 0.92`; offload
  would only hurt TTFT. Kept in the recipe as a fallback if W4A16
  ever proves unstable.

## Verification

- **Smoke test gates adoption.** `pytest -m cosmos3 tests/test_cosmos3_smoke.py`
  must pass all 4 steps before this ADR is considered live. CI
  marker `cosmos3` is opt-in (no auto-run) so it doesn't break
  machines without the local server.
- **Per-profile test matrix.** `tests/test_profiles.py` (new, see
  ADR-0009) adds `test_local_cosmos3_loads_with_quantization_flag`
  and `test_local_cosmos3_openai_compat_parses_tool_calls`.
- **End-to-end demo.** A real `--llm local_cosmos3` run of the
  standard pick-and-place flow on the demo rig produces a green
  `runs/<timestamp>/trace.jsonl` with `tool_name` entries matching
  the agent's plan.

## Cross-references

- ADR-0006 (secrets): `local_cosmos3.yaml` has no secret; uses
  `api_key: EMPTY` (dummy for local server auth).
- ADR-0009 (profile switching): the new profile participates in the
  runtime switch + fallback infrastructure.
- `docs/SPEC-local-llm.md` (new): operational details (startup
  order, log locations, health probes).
- `scripts/serve_cosmos3_vllm.sh` and `scripts/quantize_cosmos3.py`:
  implementation.
