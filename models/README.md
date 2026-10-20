# Bundled Model Weights

This directory is **gitignored** (see root `.gitignore` → `models/*`). Run
`python scripts/setup_models.py` after cloning to fetch / verify the
weights your config needs.

## What's here

| File | Purpose | Source |
|------|---------|--------|
| `yoloe-11s-seg.pt` | Default open-vocabulary detector (matches `configs/demo.yaml`) | `rebot_grasp/models/` (sibling project) |
| `yoloe-26l-seg.pt` | Heavier YOLOE; better recognition, 2x cost | `rebot_grasp/models/` |
| `mobileclip2_b.ts` | YOLOE text encoder (torchscript) | Qualcomm MobileCLIP2 → exported to .ts |
| `mobileclip_blt.ts` | Larger text-encoder variant | Same |

## LLM weights

The local LLM (`local_qwen3_vl` profile) is served separately by
`scripts/start_cosmos3_server.sh` (which forwards to `start_reason2_server.sh`).
The model files themselves live under `models/qwen3-vl-2b-awq-4bit/` and
are also gitignored. See ADR-0007 in the docs for why AWQ-4bit was picked
over BF16 / FP8.

## Adding a new entry

Edit `scripts/setup_models.py`'s `MANIFEST` list — add `{path, min_mb, use,
auto, manual}` so the script knows about it. Keep `min_mb` slightly below
the actual size so a corrupted partial download still fails the check.
