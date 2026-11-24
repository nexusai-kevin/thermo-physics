"""Deterministic ESP32-S3 + FluidNC 3-axis digital twin and fault injector."""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import asdict, dataclass
from typing import Any, Literal

FaultType = Literal[
    "none",
    "thermal_drift",
    "spindle_overload",
    "tool_chatter",
    "step_skip",
    "limit_breach",
]
VALID_FAULTS: tuple[FaultType, ...] = (
    "none",
    "thermal_drift",
    "spindle_overload",
    "tool_chatter",
    "step_skip",
    "limit_breach",
)

_G1_WORD_RE = re.compile(r"\b([XYZF])([+-]?[0-9]+(?:\.[0-9]+)?)\b")
_M3_RPM_RE = re.compile(r"\bS([0-9]+(?:\.[0-9]+)?)\b")


@dataclass(frozen=True)
class WorkEnvelope:
    """Physical soft-limits and operating ceilings for Fabrication Cell #1."""

    x_min_mm: float = 0.0
    x_max_mm: float = 300.0
    y_min_mm: float = 0.0
    y_max_mm: float = 300.0
    z_min_mm: float = -50.0
    z_max_mm: float = 0.0
    max_feedrate_mm_min: float = 3000.0
    max_spindle_rpm: float = 24000.0
    temp_warning_c: float = 65.0
    temp_ceiling_c: float = 80.0


@dataclass(frozen=True)
class TelemetryFrame:
    """Normalized JSON-serializable telemetry snapshot from FluidNC on ESP32-S3."""

    cell_id: str
    mcu: str
    firmware: str
    tick: int
    timestamp_ms: int
    tick_hz: float
    controller_state: Literal["Idle", "Run", "Hold", "Alarm"]
    pos_mm: dict[str, float]
    pos_error_mm: float
    feedrate_mm_min: float
    spindle_rpm: float
    spindle_temp_c: float
    mcu_temp_c: float
    motor_current_a: float
    motor_load_pct: float
    imu_vibration_rms_g: float
    limit_triggered: bool
    limit_pins: str
    active_gcode: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))


class FluidNCTwin:
    """Deterministic 3-axis FluidNC kinematic & sensor simulator with fault injection."""

    def __init__(
        self,
        tick_hz: float = 10.0,
        envelope: WorkEnvelope = WorkEnvelope(),
        seed: int = 0,
    ) -> None:
        if tick_hz <= 0.0 or not math.isfinite(tick_hz):
            raise ValueError(f"tick_hz must be finite and > 0, got {tick_hz}")
        self.tick_hz = float(tick_hz)
        self.envelope = envelope
        self._rng = random.Random(seed)
        self._dt_ms = max(1, int(round(1000.0 / self.tick_hz)))

        self.tick: int = 0
        self.timestamp_ms: int = 0
        self.controller_state: Literal["Idle", "Run", "Hold", "Alarm"] = "Run"
        self.pos_mm: dict[str, float] = {"x": 150.0, "y": 150.0, "z": -5.0}
        self.pos_error_mm: float = 0.01
        self.feedrate_mm_min: float = 1200.0
        self.spindle_rpm: float = 16000.0
        self.spindle_temp_c: float = 38.0
        self.mcu_temp_c: float = 42.0
        self.motor_current_a: float = 2.1
        self.motor_load_pct: float = 35.0
        self.imu_vibration_rms_g: float = 0.18
        self.limit_triggered: bool = False
        self.limit_pins: str = ""
        self.active_gcode: str = "G1 X160.0 Y150.0 Z-5.0 F1200"

        self._active_fault: FaultType = "none"
        self._fault_severity: float = 1.0
        self._fault_axis: str = "X"

    def inject_fault(
        self,
        fault_type: FaultType,
        severity: float = 1.0,
        axis: str = "X",
    ) -> None:
        """Activate a deterministic physical fault mode."""
        if fault_type not in VALID_FAULTS:
            raise ValueError(f"Unsupported fault_type={fault_type!r}; expected one of {VALID_FAULTS}")
        if severity <= 0.0 or not math.isfinite(severity):
            raise ValueError(f"severity must be finite and > 0, got {severity}")
        axis_upper = axis.upper()
        if axis_upper not in ("X", "Y", "Z"):
            raise ValueError(f"axis must be X, Y, or Z, got {axis!r}")
        self._active_fault = fault_type
        self._fault_severity = float(severity)
        self._fault_axis = axis_upper

    def clear_fault(self) -> None:
        """Clear any injected fault and restore nominal sensor baselines."""
        self._active_fault = "none"
        self._fault_severity = 1.0
        self.limit_triggered = False
        self.limit_pins = ""
        self.pos_error_mm = 0.01

    def step(self, ticks: int = 1) -> TelemetryFrame:
        """Advance the digital twin by `ticks` and emit the latest TelemetryFrame."""
        if ticks < 1:
            raise ValueError(f"ticks must be >= 1, got {ticks}")

        for _ in range(ticks):
            self.tick += 1
            self.timestamp_ms += self._dt_ms

            if self.controller_state == "Run":
                step_mm = (self.feedrate_mm_min / 60.0) * (self._dt_ms / 1000.0)
                if self._active_fault != "limit_breach":
                    # Gentle bounded oscillation inside the safe work envelope
                    phase = 0.25 * self.tick
                    nx = self.pos_mm["x"] + min(step_mm, 1.5) * math.cos(phase)
                    ny = self.pos_mm["y"] + min(step_mm, 1.5) * math.sin(phase)
                    self.pos_mm["x"] = max(self.envelope.x_min_mm + 1.0, min(self.envelope.x_max_mm - 1.0, nx))
                    self.pos_mm["y"] = max(self.envelope.y_min_mm + 1.0, min(self.envelope.y_max_mm - 1.0, ny))

            jitter = self._rng.uniform(-0.02, 0.02)
            sev = self._fault_severity

            if self._active_fault == "none":
                self.spindle_temp_c = round(40.0 + 3.0 * math.sin(0.1 * self.tick) + jitter * 10.0, 2)
                self.mcu_temp_c = round(42.5 + 1.5 * math.cos(0.1 * self.tick) + jitter * 5.0, 2)
                self.motor_current_a = round(2.10 + 0.25 * math.sin(0.2 * self.tick) + jitter, 3)
                self.motor_load_pct = round(35.0 + 4.0 * math.sin(0.2 * self.tick) + jitter * 10.0, 2)
                self.imu_vibration_rms_g = round(max(0.05, 0.18 + 0.04 * math.sin(0.3 * self.tick) + jitter * 0.5), 3)
                self.pos_error_mm = round(max(0.002, 0.010 + abs(jitter) * 0.2), 4)
                self.limit_triggered = False
                self.limit_pins = ""
            elif self._active_fault == "thermal_drift":
                self.spindle_temp_c = round(min(95.0, 68.0 + 10.0 * sev + jitter * 5.0), 2)
                self.mcu_temp_c = round(min(84.0, 54.0 + 6.0 * sev + jitter * 3.0), 2)
                self.motor_current_a = round(2.45 + 0.20 * sev + jitter, 3)
                self.motor_load_pct = round(42.0 + 5.0 * sev + jitter * 10.0, 2)
                self.imu_vibration_rms_g = round(0.24 + 0.05 * sev + abs(jitter) * 0.5, 3)
                self.pos_error_mm = round(0.025 + 0.010 * sev, 4)
            elif self._active_fault == "spindle_overload":
                self.motor_current_a = round(min(9.5, 4.8 + 1.8 * sev + abs(jitter)), 3)
                self.motor_load_pct = round(min(100.0, 84.0 + 8.0 * sev + jitter * 5.0), 2)
                self.spindle_rpm = round(max(4000.0, self.spindle_rpm * (1.0 - 0.06 * min(sev, 2.0))), 1)
                self.spindle_temp_c = round(54.0 + 4.0 * sev + jitter * 5.0, 2)
                self.imu_vibration_rms_g = round(0.45 + 0.15 * sev, 3)
                self.pos_error_mm = round(0.03 + 0.01 * sev, 4)
            elif self._active_fault == "tool_chatter":
                self.imu_vibration_rms_g = round(min(4.5, 1.5 + 1.2 * sev + abs(jitter)), 3)
                self.motor_current_a = round(2.8 + 0.4 * sev + jitter, 3)
                self.motor_load_pct = round(min(75.0, 48.0 + 8.0 * sev), 2)
                self.spindle_temp_c = round(46.0 + 2.0 * sev, 2)
                self.pos_error_mm = round(0.04 + 0.02 * sev, 4)
            elif self._active_fault == "step_skip":
                self.pos_error_mm = round(0.35 + 0.85 * sev + abs(jitter) * 0.5, 4)
                self.imu_vibration_rms_g = round(0.55 + 0.25 * sev, 3)
                self.motor_current_a = round(3.4 + 0.5 * sev, 3)
                self.motor_load_pct = round(min(78.0, 58.0 + 8.0 * sev), 2)
            elif self._active_fault == "limit_breach":
                if self._fault_axis == "X":
                    self.pos_mm["x"] = round(self.envelope.x_max_mm + 1.2 * sev, 3)
                elif self._fault_axis == "Y":
                    self.pos_mm["y"] = round(self.envelope.y_max_mm + 1.2 * sev, 3)
                else:
                    self.pos_mm["z"] = round(self.envelope.z_min_mm - 1.0 * sev, 3)
                self.limit_triggered = True
                self.limit_pins = self._fault_axis
                self.controller_state = "Alarm"
                self.feedrate_mm_min = 0.0

        return self.snapshot()

    def snapshot(self) -> TelemetryFrame:
        """Return an immutable snapshot of the current twin state."""
        return TelemetryFrame(
            cell_id="fab-cell-01",
            mcu="ESP32-S3",
            firmware="FluidNC-3.9",
            tick=self.tick,
            timestamp_ms=self.timestamp_ms,
            tick_hz=self.tick_hz,
            controller_state=self.controller_state,
            pos_mm={k: round(v, 3) for k, v in self.pos_mm.items()},
            pos_error_mm=self.pos_error_mm,
            feedrate_mm_min=round(self.feedrate_mm_min, 2),
            spindle_rpm=round(self.spindle_rpm, 1),
            spindle_temp_c=self.spindle_temp_c,
            mcu_temp_c=self.mcu_temp_c,
            motor_current_a=self.motor_current_a,
            motor_load_pct=self.motor_load_pct,
            imu_vibration_rms_g=self.imu_vibration_rms_g,
            limit_triggered=self.limit_triggered,
            limit_pins=self.limit_pins,
            active_gcode=self.active_gcode,
        )

    def apply_command(self, cmd: str) -> str:
        """Apply a Safety-Gate-approved FluidNC command or real-time override."""
        if cmd == "\x18":
            self.controller_state = "Alarm"
            self.feedrate_mm_min = 0.0
            self.spindle_rpm = 0.0
            self.active_gcode = "RESET"
            return self.controller_state
        if cmd == "!":
            if self.controller_state != "Alarm":
                self.controller_state = "Hold"
            self.feedrate_mm_min = 0.0
            return self.controller_state
        if cmd == "~":
            if self.controller_state != "Alarm":
                self.controller_state = "Run"
                if self.feedrate_mm_min <= 0.0:
                    self.feedrate_mm_min = 1200.0
            return self.controller_state
        if cmd == "M5":
            self.spindle_rpm = 0.0
            self.feedrate_mm_min = 0.0
            self.active_gcode = "M5"
            return self.controller_state
        if cmd.startswith("M3"):
            m = _M3_RPM_RE.search(cmd)
            if m:
                self.spindle_rpm = float(m.group(1))
            self.active_gcode = cmd
            return self.controller_state
        if cmd.startswith(("G4", "G04")):
            self.feedrate_mm_min = 0.0
            self.active_gcode = cmd
            return self.controller_state
        if cmd.startswith(("G1", "G01")):
            for axis, val_str in _G1_WORD_RE.findall(cmd):
                val = float(val_str)
                if axis == "F":
                    self.feedrate_mm_min = val
                else:
                    self.pos_mm[axis.lower()] = val
            if self.controller_state != "Alarm":
                self.controller_state = "Run"
            self.active_gcode = cmd
            return self.controller_state
        return self.controller_state
