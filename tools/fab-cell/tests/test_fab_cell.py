"""Comprehensive unit & adversarial tests for Fabrication Cell #1 Twin, Safety Gate, and Dataset."""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import pytest

_FAB_ROOT = Path(__file__).resolve().parent.parent
if str(_FAB_ROOT) not in sys.path:
    sys.path.insert(0, str(_FAB_ROOT))

from data.fab_dataset import (
    build_fab_dataset,
    export_fab_jsonl,
    reference_controller_policy,
    split_fab_dataset,
)
from safety.gate import (
    GateState,
    SafetyGate,
    SafetyGateConfig,
    validate_whitelisted_gcode,
)
from sim.fluidnc_twin import VALID_FAULTS, FluidNCTwin, WorkEnvelope


def test_fluidnc_twin_determinism_tick_rates_and_fault_injection() -> None:
    t1 = FluidNCTwin(tick_hz=25.0, seed=7)
    t2 = FluidNCTwin(tick_hz=25.0, seed=7)
    f1 = t1.step(ticks=5)
    f2 = t2.step(ticks=5)
    assert f1.to_json() == f2.to_json()
    assert f1.tick == 5 and f1.timestamp_ms == 200 and f1.tick_hz == 25.0

    # Every injected fault is caught by reference_controller_policy and passes SafetyGate cleanly
    expected_actions = {
        "none": ("NOMINAL", "CONTINUE", GateState.RUNNING),
        "thermal_drift": ("WARNING", "RETRACT_SAFE_Z", GateState.RUNNING),
        "spindle_overload": ("WARNING", "REDUCE_FEED", GateState.RUNNING),
        "tool_chatter": ("WARNING", "SPINDLE_ADJUST", GateState.RUNNING),
        "step_skip": ("FAULT", "FEED_HOLD", GateState.HOLD),
        "limit_breach": ("FAULT", "ESTOP", GateState.ESTOP),
    }

    for fault in VALID_FAULTS:
        twin = FluidNCTwin(tick_hz=10.0, seed=11)
        if fault == "thermal_drift":
            twin.inject_fault(fault, severity=0.5)
        elif fault != "none":
            twin.inject_fault(fault, severity=1.0, axis="Y")
        frame = twin.step(ticks=1)
        action = reference_controller_policy(frame)
        exp_status, exp_act, exp_state = expected_actions[fault]
        assert action["status"] == exp_status
        assert action["fault_type"] == fault
        assert action["action"] == exp_act

        gate = SafetyGate()
        gate.start()
        dec = gate.evaluate(json.dumps(action), telemetry=frame, inference_latency_ms=12.0)
        assert dec.approved is True
        assert dec.state == exp_state
        assert dec.violations == []
        twin.apply_command(dec.emitted_command)

    # Verify command application on twin
    ctrl = FluidNCTwin(tick_hz=10.0, seed=0)
    assert ctrl.apply_command("!") == "Hold"
    assert ctrl.feedrate_mm_min == 0.0
    assert ctrl.apply_command("~") == "Run"
    assert ctrl.apply_command("M3 S12000") == "Run" and ctrl.spindle_rpm == 12000.0
    assert ctrl.apply_command("G1 X50.0 Y60.0 Z-4.0 F500") == "Run"
    assert ctrl.pos_mm == {"x": 50.0, "y": 60.0, "z": -4.0}
    assert ctrl.apply_command("\x18") == "Alarm" and ctrl.spindle_rpm == 0.0


@pytest.mark.parametrize(
    "bad_raw",
    [
        "",
        "   ",
        "not json at all",
        '{"status": "NOMINAL", "fault_type": "none", "action": "CONTINUE"',  # truncated
        '```json\n{"status": "NOMINAL", "fault_type": "none", "action": "CONTINUE", "gcode": "~", "confidence": 0.9}\n```',
        'Sure! {"status": "NOMINAL", "fault_type": "none", "action": "CONTINUE", "gcode": "~", "confidence": 0.9}',
        '[{"status": "NOMINAL", "fault_type": "none", "action": "CONTINUE", "gcode": "~", "confidence": 0.9}]',
        '{"status": "NOMINAL", "fault_type": "none", "action": "CONTINUE", "gcode": "~", "confidence": 0.9, "extra": 1}',
        '{"status": "NOMINAL", "fault_type": "none", "action": "CONTINUE", "gcode": "~"}',  # missing confidence
        '{"status": "NOMINAL", "fault_type": "none", "action": "CONTINUE", "gcode": "~", "confidence": NaN}',
        '{"status": "NOMINAL", "fault_type": "none", "action": "CONTINUE", "gcode": "~", "confidence": 1.5}',
        '{"status": "NOMINAL", "fault_type": "none", "action": "CONTINUE", "gcode": "~", "confidence": true}',
        '{"status": "INVALID", "fault_type": "none", "action": "CONTINUE", "gcode": "~", "confidence": 0.9}',
    ],
)
def test_safety_gate_strict_json_and_schema_rejects_malformed_inputs(bad_raw: str) -> None:
    gate = SafetyGate()
    gate.start()
    dec = gate.evaluate(bad_raw, inference_latency_ms=5.0)
    assert dec.approved is False
    assert dec.state == GateState.HOLD
    assert dec.emitted_command == "!"
    assert len(dec.violations) > 0


@pytest.mark.parametrize(
    "unsafe_gcode",
    [
        "G1 X-0.01 Y10.0 Z-5.0 F500",
        "G1 X300.01 Y10.0 Z-5.0 F500",
        "G1 X10.0 Y-0.5 Z-5.0 F500",
        "G1 X10.0 Y300.01 Z-5.0 F500",
        "G1 X10.0 Y10.0 Z-50.01 F500",
        "G1 X10.0 Y10.0 Z0.01 F500",
        "G1 X10.0 Y10.0 Z-5.0 F0",
        "G1 X10.0 Y10.0 Z-5.0 F-100",
        "G1 X10.0 Y10.0 Z-5.0 F3000.1",
        "G1 X10.0 Y10.0 Z-5.0",  # missing F
        "G0 X10.0 Y10.0 Z-5.0 F500",  # unbounded rapid
        "G28",
        "G28 X0 Y0",
        "$RST=*",
        "$H",
        "$X",
        "G91",
        "G92 X0 Y0 Z0",
        "G53 G0 Z0",
        "M3 S24001",
        "M3 S-100",
        "M4 S12000",
        "G4 P0",
        "G4 P30.1",
        "G1 X10.0 F500\nG28",
        "G1 X10.0 F500; $RST=*",
        "G1 X10.0 G28 F500",
        "G1 X10.0 X350.0 F500",
    ],
)
def test_safety_gate_soft_limits_feedrate_and_whitelist_rejections(unsafe_gcode: str) -> None:
    gate = SafetyGate()
    gate.start()
    payload = json.dumps(
        {
            "status": "WARNING",
            "fault_type": "tool_chatter",
            "action": "REDUCE_FEED",
            "gcode": unsafe_gcode,
            "confidence": 0.9,
        }
    )
    dec = gate.evaluate(payload, inference_latency_ms=10.0)
    assert dec.approved is False
    assert dec.state == GateState.HOLD
    assert dec.emitted_command == "!"
    assert len(dec.violations) > 0


def test_safety_gate_thermal_ceiling_and_watchdog_timeout() -> None:
    # 1. Thermal ceiling (>= 80C) blocks motion/resume, forces HOLD
    twin = FluidNCTwin(seed=3)
    twin.inject_fault("thermal_drift", severity=1.5)  # ~83C
    hot_frame = twin.step(ticks=1)
    assert 80.0 <= hot_frame.spindle_temp_c < 92.0

    gate = SafetyGate()
    gate.start()
    move_cmd = json.dumps(
        {
            "status": "WARNING",
            "fault_type": "thermal_drift",
            "action": "RETRACT_SAFE_Z",
            "gcode": "G1 Z0 F600",
            "confidence": 0.95,
        }
    )
    dec_hot = gate.evaluate(move_cmd, telemetry=hot_frame, inference_latency_ms=15.0)
    assert dec_hot.approved is False
    assert dec_hot.state == GateState.HOLD
    assert dec_hot.emitted_command == "!"

    # Critical thermal (>= 92C) forces ESTOP
    twin.inject_fault("thermal_drift", severity=3.0)
    crit_frame = twin.step(ticks=1)
    assert crit_frame.spindle_temp_c >= 92.0
    gate2 = SafetyGate()
    gate2.start()
    dec_crit = gate2.evaluate(move_cmd, telemetry=crit_frame, inference_latency_ms=15.0)
    assert dec_crit.approved is False
    assert dec_crit.state == GateState.ESTOP
    assert dec_crit.emitted_command == "\x18"

    # 2. Watchdog timeout (> 250ms) forces HOLD
    gate3 = SafetyGate(SafetyGateConfig(latency_budget_ms=250.0))
    gate3.start()
    nominal_cmd = json.dumps(
        {
            "status": "NOMINAL",
            "fault_type": "none",
            "action": "CONTINUE",
            "gcode": "~",
            "confidence": 0.99,
        }
    )
    dec_late = gate3.evaluate(nominal_cmd, inference_latency_ms=250.1)
    assert dec_late.approved is False
    assert dec_late.state == GateState.HOLD
    assert dec_late.emitted_command == "!"

    # guard_inference wall-clock watchdog & exception safety
    gate4 = SafetyGate(SafetyGateConfig(latency_budget_ms=20.0))
    gate4.start()
    dec_slow = gate4.guard_inference(lambda: (time.sleep(0.04), nominal_cmd)[1])
    assert dec_slow.approved is False
    assert dec_slow.state == GateState.HOLD
    assert dec_slow.emitted_command == "!"

    gate5 = SafetyGate()
    gate5.start()
    dec_crash = gate5.guard_inference(lambda: (_ for _ in ()).throw(RuntimeError("oom")))
    assert dec_crash.approved is False
    assert dec_crash.state == GateState.HOLD
    assert dec_crash.emitted_command == "!"


def test_safety_gate_state_machine_and_manual_estop_reset() -> None:
    gate = SafetyGate()
    assert gate.state == GateState.IDLE
    assert gate.start() == GateState.RUNNING

    # Transition to HOLD via FEED_HOLD
    hold_cmd = json.dumps(
        {"status": "FAULT", "fault_type": "step_skip", "action": "FEED_HOLD", "gcode": "!", "confidence": 0.97}
    )
    dec_hold = gate.evaluate(hold_cmd, inference_latency_ms=5.0)
    assert dec_hold.approved is True and dec_hold.state == GateState.HOLD and dec_hold.emitted_command == "!"

    # Resume HOLD -> RUNNING
    assert gate.resume() == GateState.RUNNING

    # Trigger ESTOP -> cannot resume or evaluate until manual_reset(operator_ack=True)
    estop_cmd = json.dumps(
        {"status": "FAULT", "fault_type": "limit_breach", "action": "ESTOP", "gcode": "M5", "confidence": 0.99}
    )
    dec_estop = gate.evaluate(estop_cmd, inference_latency_ms=5.0)
    assert dec_estop.approved is True and dec_estop.state == GateState.ESTOP

    with pytest.raises(RuntimeError):
        gate.resume()
    with pytest.raises(RuntimeError):
        gate.start()

    nominal_cmd = json.dumps(
        {"status": "NOMINAL", "fault_type": "none", "action": "CONTINUE", "gcode": "~", "confidence": 0.99}
    )
    dec_latched = gate.evaluate(nominal_cmd, inference_latency_ms=5.0)
    assert dec_latched.approved is False
    assert dec_latched.state == GateState.ESTOP
    assert dec_latched.emitted_command == "\x18"

    assert gate.manual_reset(operator_ack=False) is False
    assert gate.state == GateState.ESTOP
    assert gate.manual_reset(operator_ack=True) is True
    assert gate.state == GateState.IDLE


def test_safety_gate_fuzz_and_adversarial_zero_unsafe_gcode() -> None:
    rng = random.Random(2026)
    env = WorkEnvelope()
    adversarial_gcodes = [
        "$J=G91 X1000 F9999",
        "G2 X50 Y50 I10 J10 F500",
        "G3 X50 Y50 I10 J10 F500",
        "G38.2 Z-60 F100",
        "G1 X10 Y10 Z-5 F500 (comment)",
        "G1 X10 Y10 Z-5 F500\r\n$RST=*",
        "G1\tX10\tF500",
        "G1 X1e99 F500",
        "G1 Xnan F500",
        "G1 Xinf F500",
        "M30",
        "M2",
        "M8",
        "G1 X0 Y0 Z-5 F100\x00G28",
    ]
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .-+;$*()\n\t"
    for _ in range(500):
        length = rng.randint(1, 36)
        adversarial_gcodes.append("".join(rng.choice(alphabet) for _ in range(length)))

    for gcode in adversarial_gcodes:
        ok, _ = validate_whitelisted_gcode(gcode, env)
        gate = SafetyGate()
        gate.start()
        raw = json.dumps(
            {
                "status": "WARNING",
                "fault_type": "tool_chatter",
                "action": "REDUCE_FEED",
                "gcode": gcode,
                "confidence": 0.9,
            }
        )
        dec = gate.evaluate(raw, inference_latency_ms=5.0)
        if dec.approved:
            assert ok is True
            assert dec.emitted_command == gcode
            assert not any(bad in gcode for bad in ("G28", "$", "G0 ", "G91", "G92", ";", "\n", "\r"))
        else:
            assert dec.emitted_command in ("!", "\x18")
            assert dec.state in (GateState.HOLD, GateState.ESTOP)


def test_fab_dataset_60_samples_and_tensor_compressor_wiring(tmp_path: Path) -> None:
    samples = build_fab_dataset()
    assert len(samples) == 60
    calib, ev = split_fab_dataset(samples)
    assert len(calib) == 45 and len(ev) == 15

    # Every ground-truth sample must pass SafetyGate with 0 violations
    for s in samples:
        assert set(s["expected"].keys()) == {"status", "fault_type", "action", "gcode", "confidence"}
        assert "Return ONLY JSON." in s["prompt"]
        gate = SafetyGate()
        gate.start()
        dec = gate.evaluate(
            json.dumps(s["expected"]),
            telemetry=s["telemetry"],
            inference_latency_ms=10.0,
        )
        assert dec.approved is True, f"Sample {s['id']} rejected: {dec.violations}"
        assert dec.violations == []

    # Verify direct compatibility with tools/tensor-compressor/common/calib_data.py
    tc_root = _FAB_ROOT.parent / "tensor-compressor"
    if str(tc_root) not in sys.path:
        sys.path.insert(0, str(tc_root))
    from common.calib_data import load_calibration_dataset

    jsonl_path = export_fab_jsonl(tmp_path / "fab_cell_60.jsonl")
    loaded = load_calibration_dataset(jsonl_path)
    assert len(loaded) == 60
    assert loaded[0]["domain"] == "fab_cell_fluidnc"
    assert loaded[0]["expected"] == samples[0]["expected"]
