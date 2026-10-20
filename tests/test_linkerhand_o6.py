from __future__ import annotations

import pytest

from wrc_demo.end_effectors.linkerhand_o6 import (
    FINGERS, LinkerHandGraspAdapter, LinkerHandO6, O6Config,
)
from wrc_demo.types import SafetyViolation


class FakeO6:
    def __init__(self, frames):
        self.frames = iter(frames)
        self.moves = []
        self.speeds = []
        self.torques = []

    def get_matrix_touch_v2(self):
        return next(self.frames)

    def finger_move(self, pose):
        self.moves.append(list(pose))

    def set_speed(self, speed):
        self.speeds.append(list(speed))

    def set_torque(self, torque):
        self.torques.append(list(torque))


class FakePerFingerO6(FakeO6):
    def __init__(self):
        super().__init__([])
        self.reads = []

    def _read(self, finger):
        self.reads.append(finger)
        return matrices([40])[0]

    def get_thumb_matrix_touch(self):
        return self._read("thumb")

    def get_index_matrix_touch(self):
        return self._read("index")

    def get_middle_matrix_touch(self):
        return self._read("middle")

    def get_ring_matrix_touch(self):
        return self._read("ring")

    def get_little_matrix_touch(self):
        return self._read("little")


def matrices(peaks=None):
    peaks = peaks or [0] * 5
    result = []
    for peak in peaks:
        matrix = [[0] * 4 for _ in range(10)]
        matrix[0][0] = peak
        result.append(matrix)
    return result


def filled_matrices(value):
    return [[([value] * 4) for _ in range(10)] for _ in range(5)]


def config():
    return O6Config(sample_period_s=0, timeout_s=1, step=100, baseline_samples=1,
                    contact_peak=30, contact_sum=100, contact_confirm_frames=1,
                    hard_peak=245)


def test_open_uses_soft_limits():
    fake = FakeO6([matrices()])
    hand = LinkerHandO6(config(), sdk_factory=lambda: fake)
    result = hand.open()
    assert result["pose"] == [255, 70, 255, 255, 255, 255]
    assert fake.moves[-1] == result["pose"]
    assert fake.torques[-1] == [55] * 6


def test_close_hand_reaches_power_target_without_requiring_contact():
    fake = FakeO6([])
    hand = LinkerHandO6(config(), sdk_factory=lambda: fake)
    result = hand.close_hand()
    assert result["action"] == "close"
    assert result["target_reached"] is True
    assert result["contacted"] == []
    assert result["pose"] == hand.cfg.power_target
    assert fake.moves == [hand.cfg.power_target]


def test_repeated_moves_do_not_retransmit_unchanged_speed_and_torque():
    fake = FakeO6([])
    hand = LinkerHandO6(config(), sdk_factory=lambda: fake)
    hand.move(hand.cfg.open_pose)
    hand.move(hand.cfg.power_target)
    assert fake.speeds == [[255] * 6]
    assert fake.torques == [[55] * 6]


def test_pinch_freezes_thumb_and_index_on_contact():
    fake = FakeO6([matrices(), matrices([40, 35, 0, 0, 0])])
    hand = LinkerHandO6(config(), sdk_factory=lambda: fake)
    result = hand.pinch()
    assert result["contacted"] == ["index", "thumb"]
    assert fake.moves == []  # contact was present before the first close step
    assert result["pose"][3:] == [255, 255, 255]


def test_power_grasp_freezes_hard_contacted_fingers_without_global_abort():
    fake = FakeO6([matrices(), matrices([250] * 5)])
    hand = LinkerHandO6(config(), sdk_factory=lambda: fake)
    result = hand.power_grasp()
    assert fake.moves == []
    assert result["stop_reason"] == "tactile_contact"
    assert result["hard_contacted"] == [
        "index", "little", "middle", "ring", "thumb"
    ]


def test_ordinary_8bit_contact_is_not_misclassified_as_hard_pressure():
    fake = FakeO6([matrices(), matrices([209] * 5)])
    hand = LinkerHandO6(config(), sdk_factory=lambda: fake)
    result = hand.power_grasp()
    assert result["stop_reason"] == "tactile_contact"
    assert result["contacted"] == ["index", "little", "middle", "ring", "thumb"]


def test_tactile_accepts_full_can_byte_range_and_rejects_overflow():
    hand = LinkerHandO6(config(), sdk_factory=lambda: FakeO6([matrices([255] * 5)]))
    assert hand.tactile(compensated=False).peak("thumb") == 255

    overflow = LinkerHandO6(
        config(), sdk_factory=lambda: FakeO6([matrices([256] * 5)])
    )
    with pytest.raises(RuntimeError, match="0..255"):
        overflow.tactile(compensated=False)


def test_tactile_can_read_only_requested_fingers():
    fake = FakePerFingerO6()
    hand = LinkerHandO6(config(), sdk_factory=lambda: fake)

    frame = hand.tactile(compensated=False, fingers={"thumb", "middle"})

    assert set(frame.matrices) == {"thumb", "middle"}
    assert fake.reads == ["thumb", "middle"]


def test_estop_aborts_before_any_hand_command():
    fake = FakeO6([matrices()])
    hand = LinkerHandO6(
        config(), sdk_factory=lambda: fake, abort_requested=lambda: True,
    )
    with pytest.raises(SafetyViolation, match="e-stop"):
        hand.open()
    assert fake.moves == []


def test_estop_interrupts_an_in_progress_tactile_close():
    fake = FakeO6([matrices()] * 8)
    checks = {"n": 0}

    def abort_requested():
        checks["n"] += 1
        return checks["n"] >= 6

    hand = LinkerHandO6(
        config(), sdk_factory=lambda: fake, abort_requested=abort_requested,
    )
    with pytest.raises(SafetyViolation, match="e-stop"):
        hand.power_grasp()
    assert len(fake.moves) <= 1


def test_adapter_uses_power_grasp_even_for_narrow_target_and_reports_contact():
    fake = FakeO6([matrices(), matrices([40, 35, 35, 35, 35])])
    hand = LinkerHandO6(config(), sdk_factory=lambda: fake)
    adapter = LinkerHandGraspAdapter(hand)
    assert adapter.grasp(force=0.4, min_width_m=0.02, close_force=0.7)
    telemetry = adapter.last_grasp_telemetry()
    assert telemetry["grasp_mode"] == "power_grasp"
    assert telemetry["stop_reason"] == "tactile_contact"
    assert telemetry["contacted"] == ["index", "little", "middle", "ring", "thumb"]
    assert telemetry["contacted_joint_indices"] == [2, 5, 3, 4, 0]
    assert telemetry["commanded_torque"] == [102] * 6


def test_adapter_reports_air_grasp_without_raising_runtime_error():
    fake = FakeO6([matrices()] * 12)
    hand = LinkerHandO6(config(), sdk_factory=lambda: fake)
    adapter = LinkerHandGraspAdapter(hand)

    assert adapter.grasp(force=0.35) is False
    telemetry = adapter.last_grasp_telemetry()
    assert telemetry["stop_reason"] == "target_reached_without_contact"
    assert "without tactile contact" in telemetry["error"]


def test_rejects_invalid_tactile_shape():
    fake = FakeO6([[[[0] * 4 for _ in range(9)] for _ in range(5)]])
    hand = LinkerHandO6(config(), sdk_factory=lambda: fake)
    with pytest.raises(RuntimeError, match="10x4"):
        hand.tactile()


def test_tactile_baseline_subtracts_idle_sensor_offset():
    baseline = filled_matrices(7)
    touched = filled_matrices(7)
    touched[1][2][3] = 42
    fake = FakeO6([baseline, touched])
    hand = LinkerHandO6(config(), sdk_factory=lambda: fake)
    info = hand.calibrate_tactile_baseline()
    frame = hand.tactile()
    assert info == {"samples": 1, "peak": 7}
    assert frame.peak("thumb") == 0
    assert frame.peak("index") == 35


def test_pinch_steps_then_freezes_each_contacted_finger():
    cfg = config()
    cfg.step = 20
    fake = FakeO6([
        matrices(),
        matrices(),
        matrices([40, 0, 0, 0, 0]),
        matrices([40, 35, 0, 0, 0]),
    ])
    hand = LinkerHandO6(cfg, sdk_factory=lambda: fake)
    result = hand.pinch()
    assert result["contacted"] == ["index", "thumb"]
    assert len(fake.moves) == 2
    assert fake.moves[1][0] == fake.moves[0][0]
    assert fake.moves[1][2] < fake.moves[0][2]


def test_power_grasp_freezes_contacted_fingers_while_others_reach_target():
    fake = FakeO6([
        matrices(),
        matrices([40, 0, 0, 0, 0]),
        matrices([40, 35, 0, 0, 0]),
        matrices([40, 35, 0, 0, 0]),
        matrices([40, 35, 0, 0, 0]),
    ])
    hand = LinkerHandO6(config(), sdk_factory=lambda: fake)

    result = hand.power_grasp()

    assert result["contacted"] == ["index", "thumb"]
    assert result["target_reached_fingers"] == ["little", "middle", "ring"]
    assert result["stop_reason"] == "tactile_contact"
    assert result["pose"][0] == 255  # thumb froze before the first move
    assert result["pose"][2] == 155  # index froze after one close step
    assert result["pose"][3:] == hand.cfg.power_target[3:]
    assert len(fake.moves) == 3


def test_timeout_with_thumb_and_opposing_contact_is_verified_hold():
    cfg = config()
    cfg.step = 1
    cfg.timeout_s = 0.002
    cfg.sample_period_s = 0.001
    fake = FakeO6(
        [matrices()] + [matrices([40, 35, 0, 0, 0])] * 20
    )
    hand = LinkerHandO6(cfg, sdk_factory=lambda: fake)

    result = hand.power_grasp()

    assert result["stop_reason"] == "tactile_contact_timeout"
    assert result["partial_wrap"] is True
    assert result["contacted"] == ["index", "thumb"]


def test_contact_requires_configured_number_of_consecutive_frames():
    cfg = config()
    cfg.contact_confirm_frames = 2
    cfg.step = 20
    fake = FakeO6([
        matrices(),
        matrices([40, 0, 0, 0, 0]),  # one-frame thumb spike
        matrices(),
        matrices([40, 35, 35, 35, 35]),
        matrices([40, 35, 35, 35, 35]),
    ])
    hand = LinkerHandO6(cfg, sdk_factory=lambda: fake)

    result = hand.power_grasp()

    assert set(result["contacted"]) == set(FINGERS)
    assert len(fake.moves) == 3  # the isolated first-frame spike did not freeze


def test_contact_can_finish_debounce_after_finger_reaches_target():
    cfg = config()
    cfg.step = 255
    cfg.contact_confirm_frames = 3
    fake = FakeO6([
        matrices(),
        matrices(),
        matrices([40, 0, 0, 0, 0]),
        matrices([40, 0, 0, 0, 0]),
        matrices([40, 0, 0, 0, 0]),
    ])
    hand = LinkerHandO6(cfg, sdk_factory=lambda: fake)

    result = hand.power_grasp()

    assert result["contacted"] == ["thumb"]
    assert result["target_reached_fingers"] == ["index", "little", "middle", "ring"]


def test_holding_contact_reads_tactile_without_moving_fingers():
    fake = FakeO6([
        matrices(),
        matrices([45, 0, 0, 0, 0]),
        matrices([50, 0, 0, 0, 0]),
    ])
    hand = LinkerHandO6(config(), sdk_factory=lambda: fake)
    hand.calibrate_tactile_baseline()

    result = hand.holding_contact(samples=2)

    assert result["contacted"] == ["thumb"]
    assert result["sample_count"] == 2
    assert fake.moves == []


def test_power_grasp_stops_after_three_finger_hold_and_reduces_torque():
    fake = FakeO6([
        matrices(),
        matrices([40, 35, 35, 0, 0]),
    ])
    hand = LinkerHandO6(config(), sdk_factory=lambda: fake)

    result = hand.power_grasp(torque=[89] * 6)

    assert result["contacted"] == ["index", "middle", "thumb"]
    assert result["partial_wrap"] is True
    assert result["stop_reason"] == "tactile_contact"
    assert fake.moves == []
    assert fake.torques == [[89] * 6, [45] * 6]
