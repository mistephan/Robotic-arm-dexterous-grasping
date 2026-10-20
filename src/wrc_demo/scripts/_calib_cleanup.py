"""Shared cleanup helpers for hand-eye calibration CLIs.

The two calibration scripts (``calib_top_orbbec.py``, ``calib_wrist.py``)
both drive a reBot arm: a missing or wrong cleanup path lets the script
exit with the motors still enabled, then on the next garbage-collection
cycle (or via the SDK ``disconnect()`` racing with daemon thread kill)
the controller drops torque and a loaded arm free-falls. The footgun is
explicitly called out in ``CLAUDE.md`` ("Park the arm (move_home) before
disconnecting the RS arm -- disconnect() disables torque and a loaded
arm falls.") and in HANDOFF §3.

This module exposes one primitive:

- ``safe_park_and_disconnect(park_fn, disconnect_fn, *, fallback_park_fn,
  label)`` -- runs ``park_fn()``; if it raises or returns falsy, runs
  ``fallback_park_fn()`` automatically (so a hardware "joint 0 hits a
  physical limit" refusal doesn't leave the arm in its last collection
  pose when ``disconnect()`` releases torque). Then calls
  ``disconnect_fn()``. Every call is wrapped fail-soft: any raise or
  refusal is logged with the ``[label]`` prefix and never propagated.

Callers compose the *what* of the park (joint-zero vs ready-pose) and
the *how* (SafeArm vs RebotArmEndPose) -- the helper only enforces the
"fail soft, log, never raise" contract and the "try a fallback before
giving up" promise so a refused joint-zero (e.g. RS's J3 lower limit
pins it to >= 0.020 rad, so moving to literal 0 rad never reaches the
harness's valid range) doesn't silently drop the arm.
"""

from __future__ import annotations

import sys
from typing import Callable, Optional


def safe_park_and_disconnect(
    park_fn: Optional[Callable[[], Optional[bool]]],
    disconnect_fn: Optional[Callable[[], None]],
    *,
    fallback_park_fn: Optional[Callable[[], Optional[bool]]] = None,
    label: str = "calibration",
) -> bool:
    """Park the arm (park_fn), retry on failure with fallback_park_fn,
    then disconnect. Without ever raising.

    Parameters
    ----------
    park_fn
        First-attempt park routine. Returns truthy on success, falsy on
        refused (e.g. harness rejected the pose).
    fallback_park_fn
        Second-attempt park routine. Run only if ``park_fn`` raised or
        returned falsy. Same return semantics. Common choice: a
        ``safe_arm.move_joints(home_q, ...)`` fallback that targets the
        arm profile's ``home_q`` (which is inside the URDF envelope by
        construction).
    disconnect_fn
        Final teardown -- typically ``arm.disconnect()`` which releases
        torque. Runs unconditionally after the park phase, even if both
        parks refused (the arm ends up at the last commanded pose in
        that case -- this is the footgun we're guarding against, but
        we still log a clear warning so the operator can recover with
        ``safe_arm.move_home()``).
    label
        Prefixes every log line so the operator can correlate.

    Returns
    -------
    bool
        True if the arm ended up at the primary target, False if only
        the fallback (or neither) succeeded. Always returns; never
        raises.
    """
    primary_ok = _try_park(park_fn, label, primary=True)
    success = bool(primary_ok)
    if not primary_ok and fallback_park_fn is not None:
        print(
            f"[{label}] primary park refused; retrying with fallback pose",
            file=sys.stderr,
        )
        fallback_ok = _try_park(fallback_park_fn, label, primary=False)
        if not fallback_ok:
            print(
                f"[{label}] both park attempts failed; "
                "disconnecting at the arm's last collection pose -- "
                "use safe_arm.move_home() before disconnect to recover",
                file=sys.stderr,
            )
        success = bool(fallback_ok)
    _do_disconnect(disconnect_fn, label)
    return success


def _try_park(
    fn: Optional[Callable[[], Optional[bool]]],
    label: str,
    *,
    primary: bool,
) -> bool:
    """Invoke ``fn`` once and report success/failure. Failures are logged
    but not raised."""
    if fn is None:
        return False
    tag = "primary" if primary else "fallback"
    print(f"[{label}] attempting {tag} park ...", file=sys.stderr)
    try:
        ok = fn()
    except Exception as e:
        print(f"[{label}] {tag} park raised: {e!r}", file=sys.stderr)
        return False
    if ok is False:
        print(f"[{label}] {tag} park refused; arm at last commanded pose", file=sys.stderr)
        return False
    print(f"[{label}] {tag} park ok", file=sys.stderr)
    return True


def _do_disconnect(
    fn: Optional[Callable[[], None]],
    label: str,
) -> None:
    if fn is None:
        return
    print(f"[{label}] disconnecting arm", file=sys.stderr)
    try:
        fn()
        print(f"[{label}] disconnected", file=sys.stderr)
    except Exception as e:
        print(f"[{label}] disconnect raised: {e!r}", file=sys.stderr)
