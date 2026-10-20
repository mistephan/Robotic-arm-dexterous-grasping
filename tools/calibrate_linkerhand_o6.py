#!/usr/bin/env python3
"""Inspect/calibrate LinkerHand O6 without ever commanding the robot arm.

The default ``inspect`` mode is read-only. Every finger command requires the
explicit ``--allow-motion`` interlock; closing actions additionally require
``--object-present`` so they cannot silently run against the table or air.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Sequence

from wrc_demo.config import load_demo_config
from wrc_demo.end_effectors.linkerhand_o6 import FINGERS, LinkerHandO6, O6Config


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", default="rebot_rs_o6")
    p.add_argument("--mode", choices=("inspect", "baseline", "action"), default="inspect")
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--action", choices=("open", "power_grasp", "pinch"))
    p.add_argument("--allow-motion", action="store_true")
    p.add_argument("--object-present", action="store_true")
    p.add_argument("--output", type=Path)
    return p


def validate_args(args: argparse.Namespace) -> None:
    if args.samples < 1:
        raise SystemExit("--samples must be at least 1")
    if args.mode == "action" and not args.action:
        raise SystemExit("--action is required with --mode action")
    if args.mode == "action" and not args.allow_motion:
        raise SystemExit("finger motion is locked; add --allow-motion after clearing the workspace")
    if args.action in {"power_grasp", "pinch"} and not args.object_present:
        raise SystemExit("closing is locked; place a compliant object and add --object-present")


def o6_config(arm_cfg) -> O6Config:
    raw = arm_cfg.end_effector.as_dict()
    allowed = {f.name for f in fields(O6Config)}
    return O6Config(**{key: value for key, value in raw.items() if key in allowed})


def summarize(frames) -> dict:
    return {
        finger: {
            "peak": max(frame.peak(finger) for frame in frames),
            "total_max": max(frame.total(finger) for frame in frames),
        }
        for finger in FINGERS
    }


def run(argv: Sequence[str] | None = None) -> dict:
    args = parser().parse_args(argv)
    validate_args(args)
    cfg = load_demo_config(camera="mock", arm=args.arm, llm="mock")
    if cfg.arm.end_effector.type != "linkerhand_o6":
        raise SystemExit(f"arm profile {args.arm!r} is not LinkerHand O6")
    hand = LinkerHandO6(o6_config(cfg.arm))
    try:
        if args.mode == "inspect":
            frames = [hand.tactile(compensated=False) for _ in range(args.samples)]
            result = {"mode": "inspect", "read_only": True, "samples": args.samples,
                      "fingers": summarize(frames)}
        elif args.mode == "baseline":
            result = {"mode": "baseline", "read_only": True,
                      **hand.calibrate_tactile_baseline(args.samples)}
        else:
            result = {"mode": "action", "read_only": False,
                      "result": getattr(hand, args.action)()}
    finally:
        hand.close()
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return result


if __name__ == "__main__":
    run()
