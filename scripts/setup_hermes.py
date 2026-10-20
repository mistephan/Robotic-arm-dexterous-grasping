#!/usr/bin/env python3
"""Register the wrc-demo MCP server with Hermes (and print other hosts' blocks).

Hermes reads MCP servers from ~/.hermes/config.yaml under `mcp_servers.<name>`
(stdio transport, absolute paths required). By default this prints the block;
--write merges it into the config, preserving everything else.

Usage:
    python scripts/setup_hermes.py                 # print the YAML block
    python scripts/setup_hermes.py --write         # merge into ~/.hermes/config.yaml
    python scripts/setup_hermes.py --camera l515 --arm rebot_rs --write
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
SERVER_KEY = "wrc-demo:"
MCP_KEY = "mcp_servers:"


def yaml_block(cameras: str, arm: str, python: str) -> str:
    return f"""{MCP_KEY}
  {SERVER_KEY}
    command: "{python}"
    args: ["-m", "wrc_demo.apps.mcp_server"]
    env:
      PYTHONPATH: "{REPO / 'src'}"
      WRC_CAMERAS: "{cameras}"
      WRC_ARM: "{arm}"
    connect_timeout: 60
    timeout: 300
"""


def upsert(existing: str | None, block: str) -> str:
    """Insert/replace the wrc-demo entry, preserving other YAML content
    (same line-surgery approach AgenticROS uses for this file)."""
    trimmed = block.rstrip() + "\n"
    if not existing or not existing.strip():
        return trimmed
    lines = existing.split("\n")
    mcp_idx = next((i for i, l in enumerate(lines) if l.strip() == MCP_KEY), -1)
    if mcp_idx < 0:
        sep = "" if existing.endswith("\n") else "\n"
        return existing + sep + trimmed
    # find our server entry under mcp_servers
    start = -1
    for i in range(mcp_idx + 1, len(lines)):
        if lines[i] and not lines[i][0].isspace() and lines[i].strip() != MCP_KEY:
            break
        if lines[i].strip() == SERVER_KEY:
            start = i
            break
    entry_lines = [l for l in trimmed.split("\n")[1:] if l.strip()]
    if start < 0:
        return "\n".join(lines[: mcp_idx + 1] + entry_lines + lines[mcp_idx + 1 :]).rstrip() + "\n"
    end = len(lines)
    for i in range(start + 1, len(lines)):
        l = lines[i]
        if l.strip() and (not l.startswith("    ")) :
            end = i
            break
    return "\n".join(lines[:start] + entry_lines + lines[end:]).rstrip() + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--camera", default="mock")
    p.add_argument("--cameras", default=None,
                   help="comma-separated camera profiles (first = manipulation "
                        "camera); overrides --camera")
    p.add_argument("--arm", default="mock")
    p.add_argument("--python", default=PYTHON, help="interpreter with wrc_demo deps")
    p.add_argument("--write", action="store_true", help="merge into ~/.hermes/config.yaml")
    p.add_argument("--config", default=str(Path.home() / ".hermes" / "config.yaml"))
    args = p.parse_args()

    block = yaml_block(args.cameras or args.camera, args.arm, args.python)
    if not args.write:
        print("# Add to ~/.hermes/config.yaml (or re-run with --write):\n")
        print(block)
        print("# Claude Code equivalent lives in this repo's .mcp.json")
        return 0

    cfg_path = Path(args.config)
    existing = cfg_path.read_text() if cfg_path.exists() else None
    merged = upsert(existing, block)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(merged)
    print(f"[+] wrote mcp_servers.wrc-demo -> {cfg_path}")
    print(f"    cameras={args.cameras or args.camera} arm={args.arm}")
    print("    restart Hermes to pick it up")
    print("    livestream dashboard will print its URL on the gateway's stderr")
    return 0


if __name__ == "__main__":
    sys.exit(main())
