"""Deterministic 60-sample calibration & eval dataset for Fabrication Cell #1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sim.fluidnc_twin import FaultType, FluidNCTwin, TelemetryFrame, WorkEnvelope

FIELDS: list[str] = ["status", "fault_type", "action", "gcode", "confidence"]

_SYSTEM_PROMPT = (
    'Evaluate the ESP32-S3 FluidNC telemetry frame and emit a valid JSON object '
    'with keys ["status", "fault_type", "action", "gcode", "confidence"]. '
    "Return ONLY JSON.\n\nTelemetry:\n"
)


def reference_controller_policy(
    frame: TelemetryFrame | dict[str, Any],
    envelope: WorkEnvelope = WorkEnvelope(),
) -> dict[str, Any]:
    """Deterministic reference controller mapping telemetry -> safe corrective action JSON."""
    t = frame.to_dict() if isinstance(frame, TelemetryFrame) else frame
    pos = t.get("pos_mm", {})
    limit_hit = bool(t.get("limit_triggered", False)) or (
        float(pos.get("x", 0.0)) < envelope.x_min_mm
        or float(pos.get("x", 0.0)) > envelope.x_max_mm
        or float(pos.get("y", 0.0)) < envelope.y_min_mm
        or float(pos.get("y", 0.0)) > envelope.y_max_mm
        or float(pos.get("z", 0.0)) < envelope.z_min_mm
        or float(pos.get("z", 0.0)) > envelope.z_max_mm
    )
    if limit_hit:
        return {
            "status": "FAULT",
            "fault_type": "limit_breach",
            "action": "ESTOP",
            "gcode": "M5",
            "confidence": 0.99,
        }

    if float(t.get("pos_error_mm", 0.0)) >= 0.25:
        return {
            "status": "FAULT",
            "fault_type": "step_skip",
            "action": "FEED_HOLD",
            "gcode": "!",
            "confidence": 0.97,
        }

    if float(t.get("spindle_temp_c", 0.0)) >= envelope.temp_warning_c:
        if float(t.get("spindle_temp_c", 0.0)) >= envelope.temp_ceiling_c:
            return {
                "status": "FAULT",
                "fault_type": "thermal_drift",
                "action": "FEED_HOLD",
                "gcode": "!",
                "confidence": 0.98,
            }
        return {
            "status": "WARNING",
            "fault_type": "thermal_drift",
            "action": "RETRACT_SAFE_Z",
            "gcode": "G1 Z0 F600",
            "confidence": 0.95,
        }

    if float(t.get("motor_current_a", 0.0)) >= 4.5 or float(t.get("motor_load_pct", 0.0)) >= 80.0:
        return {
            "status": "WARNING",
            "fault_type": "spindle_overload",
            "action": "REDUCE_FEED",
            "gcode": "G1 Z-2.0 F400",
            "confidence": 0.96,
        }

    if float(t.get("imu_vibration_rms_g", 0.0)) >= 1.2:
        return {
            "status": "WARNING",
            "fault_type": "tool_chatter",
            "action": "SPINDLE_ADJUST",
            "gcode": "M3 S12000",
            "confidence": 0.94,
        }

    return {
        "status": "NOMINAL",
        "fault_type": "none",
        "action": "CONTINUE",
        "gcode": "~",
        "confidence": 0.98,
    }


def _make_fab_sample(
    sample_id: str,
    split: str,
    frame: TelemetryFrame,
) -> dict[str, Any]:
    doc = frame.to_json()
    expected = reference_controller_policy(frame)
    prompt = f"{_SYSTEM_PROMPT}{doc}\n\nJSON:\n"
    completion = json.dumps(expected, ensure_ascii=False)
    return {
        "id": sample_id,
        "domain": "fab_cell_fluidnc",
        "split": split,
        "document": doc,
        "telemetry": frame.to_dict(),
        "fields": list(FIELDS),
        "expected": expected,
        "prompt": prompt,
        "text": prompt + completion,
    }


def build_fab_dataset() -> list[dict[str, Any]]:
    """Build 60 deterministic stratified samples (45 calib / 15 eval) across all 6 regimes."""
    regimes: tuple[FaultType, ...] = (
        "none",
        "thermal_drift",
        "spindle_overload",
        "tool_chatter",
        "step_skip",
        "limit_breach",
    )
    axes = ("X", "Y", "Z")
    tick_rates = (10.0, 20.0, 25.0, 50.0)

    # 10 scenarios per regime, interleaved round-robin so 0..44 (calib) and 45..59 (eval) are stratified
    per_regime: dict[FaultType, list[TelemetryFrame]] = {r: [] for r in regimes}
    for r_idx, regime in enumerate(regimes):
        for k in range(10):
            hz = tick_rates[(r_idx + k) % len(tick_rates)]
            twin = FluidNCTwin(tick_hz=hz, seed=100 * r_idx + k)
            twin.pos_mm = {
                "x": round(30.0 + 22.0 * k, 2),
                "y": round(40.0 + 18.0 * k, 2),
                "z": round(-3.0 - 2.5 * (k % 6), 2),
            }
            twin.feedrate_mm_min = 900.0 + 150.0 * (k % 5)
            twin.spindle_rpm = 14000.0 + 1000.0 * (k % 5)
            twin.active_gcode = (
                f"G1 X{twin.pos_mm['x'] + 5.0:.1f} Y{twin.pos_mm['y']:.1f} "
                f"Z{twin.pos_mm['z']:.1f} F{int(twin.feedrate_mm_min)}"
            )
            twin.step(ticks=2 + (k % 4))

            if regime == "thermal_drift":
                # Keep thermal_drift samples in warning band (68..78C) so RETRACT_SAFE_Z is valid
                twin.inject_fault("thermal_drift", severity=0.25 + 0.06 * k)
            elif regime != "none":
                twin.inject_fault(
                    regime,
                    severity=0.75 + 0.12 * k,
                    axis=axes[k % len(axes)],
                )
            frame = twin.step(ticks=1)
            per_regime[regime].append(frame)

    samples: list[dict[str, Any]] = []
    idx = 0
    for k in range(10):
        for regime in regimes:
            idx += 1
            split = "calib" if idx <= 45 else "eval"
            samples.append(_make_fab_sample(f"fab-{idx:03d}", split, per_regime[regime][k]))

    return samples


def split_fab_dataset(
    samples: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return deterministic (45 calibration, 15 held-out evaluation) split."""
    data = samples if samples is not None else build_fab_dataset()
    calib = [s for s in data if s.get("split") == "calib"]
    ev = [s for s in data if s.get("split") == "eval"]
    if len(calib) == 45 and len(ev) == 15:
        return calib, ev
    return data[:45], data[45:]


def export_fab_jsonl(output_path: str | Path) -> Path:
    """Write the 60-sample dataset as JSONL compatible with tools/tensor-compressor/."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    samples = build_fab_dataset()
    lines = [json.dumps(row, ensure_ascii=False) for row in samples]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
