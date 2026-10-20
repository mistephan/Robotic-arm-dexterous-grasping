"""Safe LinkerHand O6 adapter with tactile-terminated grasp primitives.

One object owns both commands and tactile reads: the vendor SDK must never be
opened concurrently by the pressure dashboard and motion runtime on one CAN
interface.
"""

from __future__ import annotations

import copy
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from ..types import SafetyViolation

FINGERS = ("thumb", "index", "middle", "ring", "little")
FLEX_JOINT = {"thumb": 0, "index": 2, "middle": 3, "ring": 4, "little": 5}


@dataclass(frozen=True)
class TactileFrame:
    matrices: dict[str, tuple[tuple[int, ...], ...]]
    timestamp: float

    def peak(self, finger: str) -> int:
        return max(value for row in self.matrices[finger] for value in row)

    def total(self, finger: str) -> int:
        return sum(value for row in self.matrices[finger] for value in row)


@dataclass
class O6Config:
    sdk_root: str = str(
        Path(__file__).resolve().parents[3] / "sdk" / "linkerhand-python-sdk"
    )
    side: str = "right"
    can_interface: str = "can_hand"
    open_pose: list[int] = field(default_factory=lambda: [255, 70, 255, 255, 255, 255])
    # Conservative starting points; calibrate on the installed right hand.
    power_target: list[int] = field(default_factory=lambda: [80, 70, 35, 35, 40, 45])
    pinch_target: list[int] = field(default_factory=lambda: [75, 70, 55, 255, 255, 255])
    speed: list[int] = field(default_factory=lambda: [255] * 6)
    soft_torque: list[int] = field(default_factory=lambda: [55] * 6)
    step: int = 8
    baseline_samples: int = 3
    baseline_max: int = 20
    contact_peak: int = 35
    contact_sum: int = 120
    contact_confirm_frames: int = 3
    # The sensor ADC is 12-bit internally, but O6 CAN B1..B5 transmits one
    # byte per taxel and the vendor SDK exposes values in 0..255. This
    # baseline-subtracted near-saturation guard stops only the affected
    # finger immediately. Ordinary contact thresholds remain independently
    # calibrated empirical values.
    hard_peak: int = 245
    timeout_s: float = 4.0
    sample_period_s: float = 0.08
    verified_hold_settle_s: float = 0.0
    verified_hold_contact_count: int = 3
    hold_torque: list[int] = field(default_factory=lambda: [45] * 6)
    pinch_max_width_m: float = 0.04


class LinkerHandGraspAdapter:
    """Expose O6 through the existing RebotGraspBridge driver contract."""

    park_after_release = False

    def __init__(self, hand: "LinkerHandO6") -> None:
        self.hand = hand
        self._telemetry: dict = {}

    def open_gripper(self, **_kwargs) -> None:
        self._telemetry = self.hand.open()

    def close_gripper(self, **_kwargs) -> None:
        self._telemetry = self.hand.power_grasp()

    def grasp(
        self, *, force: float | None = None, min_width_m: float | None = None,
        close_force: float | None = None, **_kwargs,
    ) -> bool:
        # The autonomous grasp pipeline always uses the calibrated O6 power
        # grasp. ``min_width_m`` is a parallel-jaw safety floor, not a request
        # to switch this multi-finger hand to index pinch. Pinch remains an
        # explicit hand_action only.
        mode = "power_grasp"
        torque = None
        if force is not None:
            value = max(1, min(255, int(round(float(force) * 255.0))))
            torque = [value] * 6
        try:
            result = self.hand.power_grasp(torque=torque)
        except SafetyViolation:
            self._telemetry = {"action": mode, "stop_reason": "aborted"}
            raise
        except TimeoutError:
            self._telemetry = {"action": mode, "stop_reason": "timeout"}
            raise
        except RuntimeError as exc:
            reason = (
                "hard_pressure" if "hard-pressure" in str(exc)
                else "target_reached_without_contact"
                if "without tactile contact" in str(exc)
                else "runtime_error"
            )
            self._telemetry = {
                "action": mode, "stop_reason": reason, "error": str(exc),
            }
            if reason == "target_reached_without_contact":
                return False
            raise
        result = dict(result)
        result.setdefault("stop_reason", "tactile_contact")
        result.update({
            "grasp_mode": mode,
            "requested_force": force,
            "requested_close_force": close_force,
            "requested_min_width_m": min_width_m,
            "commanded_torque": torque or list(self.hand.cfg.soft_torque),
            "contact_count": len(result.get("contacted", [])),
            "contacted_joint_indices": [
                FLEX_JOINT[name] for name in result.get("contacted", [])
            ],
        })
        self._telemetry = result
        return result["stop_reason"] == "tactile_contact" and bool(
            result.get("contacted")
        )

    def release_gripper(self) -> None:
        self._telemetry = self.hand.release()

    def set_position_target(self, _position: float) -> None:
        # The parallel gripper parks closed after release. O6 stays open so
        # fingers cannot curl into the table or a just-released object.
        return None

    def last_grasp_telemetry(self) -> dict:
        return dict(self._telemetry)


class LinkerHandO6:
    """O6 action primitives; hardware connection is lazy and injectable."""

    def __init__(self, cfg: O6Config | None = None,
                 sdk_factory: Callable[[], Any] | None = None,
                 abort_requested: Callable[[], bool] | None = None) -> None:
        self.cfg = cfg or O6Config()
        self._sdk_factory = sdk_factory
        self._abort_requested = abort_requested
        self._sdk: Any = None
        self._pose = list(self.cfg.open_pose)
        # LinkerHand's public API prints every speed/torque write.  Tactile
        # closing calls move() many times, so cache controller settings and
        # only transmit them when they actually change.
        self._last_speed: list[int] | None = None
        self._last_torque: list[int] | None = None
        self._baseline: tuple[tuple[tuple[int, ...], ...], ...] | None = None

    def connect(self) -> None:
        if self._sdk is not None:
            return
        if self._sdk_factory is not None:
            self._sdk = self._sdk_factory()
            return
        root = str(Path(self.cfg.sdk_root).expanduser())
        if root not in sys.path:
            sys.path.insert(0, root)
        from LinkerHand.linker_hand_api import LinkerHandApi
        self._sdk = LinkerHandApi(
            hand_type=self.cfg.side, hand_joint="O6",
            modbus="None", can=self.cfg.can_interface,
        )

    def close(self) -> None:
        sdk, self._sdk = self._sdk, None
        if sdk is not None:
            close = getattr(getattr(sdk, "hand", None), "close_can_interface", None)
            if callable(close):
                close()

    def _check_abort(self) -> None:
        if self._abort_requested is not None and self._abort_requested():
            raise SafetyViolation("LinkerHand O6 action aborted by e-stop")

    def _require(self) -> Any:
        self.connect()
        return self._sdk

    @staticmethod
    def _matrix(matrix: Any) -> tuple[tuple[int, ...], ...]:
        rows = matrix.tolist() if hasattr(matrix, "tolist") else matrix
        if len(rows) != 10 or any(len(row) != 4 for row in rows):
            raise RuntimeError("O6 tactile matrix must be 10x4")
        values = tuple(tuple(int(v) for v in row) for row in rows)
        if any(v < 0 or v > 255 for row in values for v in row):
            raise RuntimeError("O6 CAN tactile value outside 0..255")
        return values

    def _raw_tactile(self) -> tuple[tuple[tuple[int, ...], ...], ...]:
        self._check_abort()
        raw = self._require().get_matrix_touch_v2()
        if len(raw) != 5:
            raise RuntimeError(f"O6 expected five tactile matrices, got {len(raw)}")
        return tuple(self._matrix(copy.deepcopy(value)) for value in raw)

    def _raw_tactile_fingers(
        self, fingers: set[str],
    ) -> dict[str, tuple[tuple[int, ...], ...]]:
        """Read only requested finger matrices through CAN B1..B5."""
        self._check_abort()
        sdk = self._require()
        getters = {
            "thumb": "get_thumb_matrix_touch",
            "index": "get_index_matrix_touch",
            "middle": "get_middle_matrix_touch",
            "ring": "get_ring_matrix_touch",
            "little": "get_little_matrix_touch",
        }
        if not all(callable(getattr(sdk, getters[f], None)) for f in fingers):
            raw = self._raw_tactile()
            return {finger: raw[FINGERS.index(finger)] for finger in fingers}
        return {
            finger: self._matrix(copy.deepcopy(getattr(sdk, getters[finger])()))
            for finger in FINGERS if finger in fingers
        }

    def calibrate_tactile_baseline(self, samples: int | None = None) -> dict:
        """Capture an unloaded per-taxel median; reject a loaded hand."""
        count = max(int(samples or self.cfg.baseline_samples), 1)
        frames = [self._raw_tactile() for _ in range(count)]
        baseline = []
        for finger in range(5):
            rows = []
            for row in range(10):
                values = []
                for col in range(4):
                    cell = sorted(frame[finger][row][col] for frame in frames)
                    values.append(cell[len(cell) // 2])
                rows.append(tuple(values))
            baseline.append(tuple(rows))
        peak = max(value for finger in baseline for row in finger for value in row)
        if peak > self.cfg.baseline_max:
            raise RuntimeError(
                f"O6 tactile baseline is loaded or noisy: peak={peak} > {self.cfg.baseline_max}"
            )
        self._baseline = tuple(baseline)
        return {"samples": count, "peak": peak}

    def tactile(
        self, compensated: bool = True, fingers: set[str] | None = None,
    ) -> TactileFrame:
        requested = set(FINGERS) if fingers is None else set(fingers)
        unknown = requested - set(FINGERS)
        if unknown:
            raise ValueError(f"unknown O6 tactile fingers: {sorted(unknown)}")
        raw_by_finger = self._raw_tactile_fingers(requested)
        if not compensated:
            matrices = raw_by_finger
        else:
            if self._baseline is None:
                self.calibrate_tactile_baseline()
            matrices = {
                finger: tuple(
                    tuple(max(
                        0,
                        raw_by_finger[finger][r][c]
                        - self._baseline[FINGERS.index(finger)][r][c],
                    ) for c in range(4))
                    for r in range(10)
                )
                for finger in requested
            }
        return TactileFrame(dict(matrices), time.monotonic())

    def holding_contact(self, samples: int = 2) -> dict:
        """Read-only tactile confirmation used before rejecting a new pick."""
        if self._baseline is None:
            raise RuntimeError("O6 tactile baseline unavailable for held-object check")
        contact_frames = {finger: 0 for finger in FINGERS}
        pressure = {}
        count = max(int(samples), 1)
        for sample in range(count):
            frame = self.tactile()
            for finger in FINGERS:
                peak, total = frame.peak(finger), frame.total(finger)
                pressure[finger] = {"peak": peak, "total": total}
                if peak >= self.cfg.contact_peak or total >= self.cfg.contact_sum:
                    contact_frames[finger] += 1
            if sample + 1 < count:
                time.sleep(self.cfg.sample_period_s)
        return {
            "contacted": sorted(
                finger for finger, frames in contact_frames.items()
                if frames == count
            ),
            "pressure": pressure,
            "sample_count": count,
            "pose": list(self._pose),
            "commanded_open": list(self._pose) == list(self.cfg.open_pose),
        }

    def move(self, pose: Sequence[int], *, torque: Sequence[int] | None = None,
             speed: Sequence[int] | None = None) -> None:
        target = [int(v) for v in pose]
        if len(target) != 6 or any(v < 0 or v > 255 for v in target):
            raise ValueError("O6 pose must contain six values in 0..255")
        self._check_abort()
        sdk = self._require()
        requested_speed = list(speed or self.cfg.speed)
        requested_torque = list(torque or self.cfg.soft_torque)
        if requested_speed != self._last_speed:
            sdk.set_speed(requested_speed)
            self._last_speed = requested_speed
        if requested_torque != self._last_torque:
            sdk.set_torque(requested_torque)
            self._last_torque = requested_torque
        self._check_abort()
        sdk.finger_move(target)
        self._pose = target

    def open(self) -> dict:
        self.move(self.cfg.open_pose)
        return {"action": "open", "pose": list(self._pose)}

    def release(self) -> dict:
        return self.open()

    def close_hand(self) -> dict:
        """Close smoothly to the calibrated fist pose.

        This is a pose command, not an object-grasp primitive. Sending one
        target lets the hand controller generate a continuous motion, just as
        :meth:`open` does. ``power_grasp`` and ``pinch`` retain the slower
        tactile-stepped behavior needed to stop individual fingers on contact.
        """
        self.move(self.cfg.power_target)
        return {
            "action": "close",
            "contacted": [],
            "pose": list(self._pose),
            "target_reached": True,
            "stop_reason": "target_reached",
        }

    def _apply_hold_torque(self) -> list[int]:
        """Drop from closing torque to a gentle static hold without moving."""
        requested = [int(v) for v in self.cfg.hold_torque]
        if requested != self._last_torque:
            self._check_abort()
            self._require().set_torque(requested)
            self._last_torque = requested
        return requested

    def power_grasp(self, torque: Sequence[int] | None = None) -> dict:
        return self._tactile_close(
            self.cfg.power_target, set(FINGERS), "power_grasp", torque=torque
        )

    def pinch(self, torque: Sequence[int] | None = None) -> dict:
        return self._tactile_close(
            self.cfg.pinch_target, {"thumb", "index"}, "pinch", torque=torque
        )

    def _tactile_close(
        self, target: Sequence[int], active: set[str], action: str,
        require_contact: bool = True, torque: Sequence[int] | None = None,
    ) -> dict:
        """Close all active fingers independently until contact or target.

        Each finger freezes after a debounced tactile contact while fingers
        that have not touched continue toward their configured target. A
        near-saturation taxel freezes only that finger immediately. The grasp
        succeeds when every active finger is either contacted or at target and
        at least one finger has tactile contact.
        """
        target = [int(v) for v in target]
        if self._baseline is None:
            self.calibrate_tactile_baseline()
        if torque is not None:
            self._check_abort()
            sdk = self._require()
            requested_speed = list(self.cfg.speed)
            requested_torque = list(torque)
            if requested_speed != self._last_speed:
                sdk.set_speed(requested_speed)
                self._last_speed = requested_speed
            if requested_torque != self._last_torque:
                sdk.set_torque(requested_torque)
                self._last_torque = requested_torque
        pose = list(self._pose)
        contacted: set[str] = set()
        hard_contacted: set[str] = set()
        target_reached: set[str] = set()
        contact_streak = {finger: 0 for finger in active}
        last_pressure = {
            finger: {"peak": 0, "total": 0} for finger in active
        }
        confirm_frames = max(int(self.cfg.contact_confirm_frames), 1)
        deadline = time.monotonic() + self.cfg.timeout_s
        verified_hold_since: float | None = None
        verified_contact_count = min(
            max(int(self.cfg.verified_hold_contact_count), 2), len(active)
        )
        while time.monotonic() < deadline:
            self._check_abort()
            moving_fingers = active - contacted - target_reached
            frame = self.tactile(fingers=moving_fingers)
            for finger in moving_fingers:
                peak, total = frame.peak(finger), frame.total(finger)
                last_pressure[finger] = {"peak": peak, "total": total}
                if peak >= self.cfg.hard_peak:
                    contacted.add(finger)
                    hard_contacted.add(finger)
                    contact_streak[finger] = confirm_frames
                    continue
                above_contact = (
                    peak >= self.cfg.contact_peak
                    or total >= self.cfg.contact_sum
                )
                contact_streak[finger] = (
                    contact_streak[finger] + 1 if above_contact else 0
                )
                if contact_streak[finger] >= confirm_frames:
                    contacted.add(finger)
            if (
                "thumb" in contacted
                and bool(contacted - {"thumb"})
                and len(contacted) >= verified_contact_count
            ):
                if verified_hold_since is None:
                    verified_hold_since = time.monotonic()
            else:
                verified_hold_since = None
            for finger in active - contacted - target_reached:
                joint = FLEX_JOINT[finger]
                if target[joint] == pose[joint] and contact_streak[finger] == 0:
                    target_reached.add(finger)
            if active <= contacted | target_reached:
                if require_contact and not contacted:
                    raise RuntimeError(
                        f"O6 {action} reached all finger targets without tactile contact"
                    )
                hold_torque = self._apply_hold_torque() if contacted else None
                return {
                    "action": action, "contacted": sorted(contacted),
                    "hold_torque": hold_torque,
                    "hard_contacted": sorted(hard_contacted),
                    "target_reached_fingers": sorted(target_reached),
                    "pressure": last_pressure,
                    "pose": pose,
                    "target_reached": bool(target_reached),
                    "stop_reason": (
                        "tactile_contact" if contacted else "target_reached"
                    ),
                }
            if (
                verified_hold_since is not None
                and time.monotonic() - verified_hold_since
                >= float(self.cfg.verified_hold_settle_s)
            ):
                hold_torque = self._apply_hold_torque()
                return {
                    "action": action,
                    "contacted": sorted(contacted),
                    "hold_torque": hold_torque,
                    "hard_contacted": sorted(hard_contacted),
                    "target_reached_fingers": sorted(target_reached),
                    "pressure": last_pressure,
                    "pose": pose,
                    "target_reached": False,
                    "stop_reason": "tactile_contact",
                    "partial_wrap": True,
                }
            changed = False
            for finger in active - contacted - target_reached:
                joint = FLEX_JOINT[finger]
                delta = target[joint] - pose[joint]
                if delta:
                    pose[joint] += max(-self.cfg.step, min(self.cfg.step, delta))
                    changed = True
            if not changed and any(
                contact_streak[finger] > 0
                for finger in active - contacted - target_reached
            ):
                time.sleep(self.cfg.sample_period_s)
                continue
            if not changed:
                raise RuntimeError(
                    f"O6 {action} made no progress; contacted={sorted(contacted)}, "
                    f"target_reached={sorted(target_reached)}"
                )
            self._check_abort()
            self.move(pose, torque=torque)
            time.sleep(self.cfg.sample_period_s)
        # A dexterous power grasp is already physically valid when the thumb
        # and at least one opposing finger have stable tactile contact. Do not
        # turn that confirmed hold into a task failure merely because the
        # remaining wrapping fingers did not reach their configured end pose
        # before the supervisory deadline.
        opposing_contact = bool(contacted - {"thumb"})
        if "thumb" in contacted and opposing_contact:
            hold_torque = self._apply_hold_torque()
            return {
                "action": action,
                "contacted": sorted(contacted),
                "hold_torque": hold_torque,
                "hard_contacted": sorted(hard_contacted),
                "target_reached_fingers": sorted(target_reached),
                "pressure": last_pressure,
                "pose": pose,
                "target_reached": False,
                "stop_reason": "tactile_contact_timeout",
                "partial_wrap": True,
            }
        raise TimeoutError(
            f"O6 {action} tactile close timed out; contacted={sorted(contacted)}"
        )
