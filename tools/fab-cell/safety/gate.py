"""Deterministic Safety Gate sitting strictly between LLM output and FluidNC."""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from sim.fluidnc_twin import FaultType, TelemetryFrame, WorkEnvelope


class GateState(str, Enum):
    """Safety Gate finite-state machine."""

    IDLE = "IDLE"
    RUNNING = "RUNNING"
    HOLD = "HOLD"
    ESTOP = "ESTOP"


ActionType = Literal[
    "CONTINUE",
    "REDUCE_FEED",
    "COOLING_PAUSE",
    "SPINDLE_ADJUST",
    "FEED_HOLD",
    "RETRACT_SAFE_Z",
    "ESTOP",
]


@dataclass(frozen=True)
class SafetyGateConfig:
    """Deterministic hardware & latency safety limits."""

    envelope: WorkEnvelope = WorkEnvelope()
    latency_budget_ms: float = 250.0
    temp_ceiling_c: float = 80.0
    estop_temp_c: float = 92.0
    mcu_temp_ceiling_c: float = 85.0


class ControllerCommand(BaseModel):
    """Strict schema for edge LLM corrective controller outputs."""

    model_config = ConfigDict(extra="forbid", strict=True)

    status: Literal["NOMINAL", "WARNING", "FAULT"]
    fault_type: FaultType
    action: ActionType
    gcode: str
    confidence: float

    @field_validator("confidence", mode="before")
    @classmethod
    def _validate_confidence(cls, v: Any) -> float:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError("confidence must be a finite number in [0.0, 1.0]")
        fv = float(v)
        if not math.isfinite(fv) or fv < 0.0 or fv > 1.0:
            raise ValueError("confidence must be finite and within [0.0, 1.0]")
        return fv


@dataclass
class GateDecision:
    """Result of passing raw model output through the Safety Gate."""

    approved: bool
    state: GateState
    emitted_command: str
    parsed: ControllerCommand | None
    violations: list[str] = field(default_factory=list)
    latency_ms: float = 0.0


_TOKEN_RE = re.compile(r"^[A-Z][+-]?[0-9]+(?:\.[0-9]+)?$")
_FORBIDDEN_CHARS = frozenset(';$%*#,"\'\\()')
_FORBIDDEN_CONTROL = frozenset("\n\r\t\x00")


def _reject_nonfinite(constant: str) -> Any:
    raise ValueError(f"Non-finite JSON constant not allowed: {constant}")


def validate_whitelisted_gcode(gcode: str, envelope: WorkEnvelope) -> tuple[bool, list[str]]:
    """Strictly validate a FluidNC/G-code command against the safe whitelist and soft-limits."""
    if not isinstance(gcode, str) or not gcode or len(gcode) > 64:
        return False, ["invalid_gcode_length_or_type"]

    if gcode == "\x18":
        return True, []

    if any(ord(ch) < 32 or ord(ch) >= 127 or ch in _FORBIDDEN_CONTROL for ch in gcode):
        return False, ["forbidden_control_or_non_ascii_char"]

    if any(ch in _FORBIDDEN_CHARS for ch in gcode):
        return False, ["forbidden_gcode_metacharacter"]

    if gcode in ("!", "~", "M5"):
        return True, []

    if gcode != gcode.strip() or "  " in gcode:
        return False, ["malformed_gcode_whitespace"]

    tokens = gcode.split(" ")
    if not tokens or any(not _TOKEN_RE.match(tok) for tok in tokens):
        return False, ["malformed_gcode_token"]

    head = tokens[0]

    # 1. Coordinated linear move: G1 / G01 with explicit F and bounded XYZ
    if head in ("G1", "G01"):
        if len(tokens) < 2:
            return False, ["g1_missing_parameters"]
        seen: set[str] = set()
        params: dict[str, float] = {}
        for tok in tokens[1:]:
            letter = tok[0]
            if letter not in ("X", "Y", "Z", "F"):
                return False, [f"g1_disallowed_word:{letter}"]
            if letter in seen:
                return False, [f"g1_duplicate_word:{letter}"]
            seen.add(letter)
            val = float(tok[1:])
            if not math.isfinite(val):
                return False, [f"g1_nonfinite_value:{letter}"]
            params[letter] = val

        if "F" not in params:
            return False, ["g1_missing_feedrate"]
        if params["F"] <= 0.0 or params["F"] > envelope.max_feedrate_mm_min:
            return False, [f"feedrate_out_of_bounds:{params['F']}"]

        if "X" in params and not (envelope.x_min_mm <= params["X"] <= envelope.x_max_mm):
            return False, [f"x_soft_limit_breach:{params['X']}"]
        if "Y" in params and not (envelope.y_min_mm <= params["Y"] <= envelope.y_max_mm):
            return False, [f"y_soft_limit_breach:{params['Y']}"]
        if "Z" in params and not (envelope.z_min_mm <= params["Z"] <= envelope.z_max_mm):
            return False, [f"z_soft_limit_breach:{params['Z']}"]

        return True, []

    # 2. Spindle speed adjust: M3 S<rpm>
    if head == "M3":
        if len(tokens) != 2 or not tokens[1].startswith("S"):
            return False, ["m3_invalid_syntax"]
        rpm = float(tokens[1][1:])
        if not math.isfinite(rpm) or rpm < 0.0 or rpm > envelope.max_spindle_rpm:
            return False, [f"spindle_rpm_out_of_bounds:{rpm}"]
        return True, []

    # 3. Bounded dwell: G4 / G04 P<seconds>
    if head in ("G4", "G04"):
        if len(tokens) != 2 or not tokens[1].startswith("P"):
            return False, ["g4_invalid_syntax"]
        dwell = float(tokens[1][1:])
        if not math.isfinite(dwell) or dwell <= 0.0 or dwell > 30.0:
            return False, [f"dwell_out_of_bounds:{dwell}"]
        return True, []

    return False, [f"unwhitelisted_command:{head}"]


class SafetyGate:
    """Deterministic safety barrier enforcing schema, soft-limits, thermal caps, and watchdog."""

    def __init__(self, config: SafetyGateConfig = SafetyGateConfig()) -> None:
        self.config = config
        self.state: GateState = GateState.IDLE
        self.estop_reason: str | None = None

    def start(self) -> GateState:
        """Transition IDLE -> RUNNING."""
        if self.state == GateState.ESTOP:
            raise RuntimeError("Cannot start from ESTOP without manual_reset(operator_ack=True)")
        self.state = GateState.RUNNING
        return self.state

    def resume(self) -> GateState:
        """Transition HOLD -> RUNNING."""
        if self.state == GateState.ESTOP:
            raise RuntimeError("Cannot resume from ESTOP without manual_reset(operator_ack=True)")
        if self.state == GateState.HOLD:
            self.state = GateState.RUNNING
        return self.state

    def trigger_estop(self, reason: str = "manual_estop") -> GateDecision:
        """Latch ESTOP state and emit FluidNC soft-reset (\x18)."""
        self.state = GateState.ESTOP
        self.estop_reason = reason
        return GateDecision(
            approved=False,
            state=self.state,
            emitted_command="\x18",
            parsed=None,
            violations=[reason],
            latency_ms=0.0,
        )

    def manual_reset(self, operator_ack: bool = False) -> bool:
        """Explicit manual reset required to clear ESTOP -> IDLE."""
        if self.state != GateState.ESTOP:
            return True
        if operator_ack is not True:
            return False
        self.state = GateState.IDLE
        self.estop_reason = None
        return True

    def _reject_hold(
        self,
        reason: str | list[str],
        latency_ms: float,
        parsed: ControllerCommand | None = None,
    ) -> GateDecision:
        self.state = GateState.HOLD
        v = [reason] if isinstance(reason, str) else list(reason)
        return GateDecision(
            approved=False,
            state=self.state,
            emitted_command="!",
            parsed=parsed,
            violations=v,
            latency_ms=latency_ms,
        )

    def _reject_estop(
        self,
        reason: str | list[str],
        latency_ms: float,
        parsed: ControllerCommand | None = None,
    ) -> GateDecision:
        self.state = GateState.ESTOP
        v = [reason] if isinstance(reason, str) else list(reason)
        self.estop_reason = v[0] if v else "safety_estop"
        return GateDecision(
            approved=False,
            state=self.state,
            emitted_command="\x18",
            parsed=parsed,
            violations=v,
            latency_ms=latency_ms,
        )

    def evaluate(
        self,
        raw_output: str,
        telemetry: TelemetryFrame | dict[str, Any] | None = None,
        inference_latency_ms: float = 0.0,
    ) -> GateDecision:
        """Validate raw LLM JSON output against schema, telemetry, whitelist, and watchdog."""
        if self.state == GateState.ESTOP:
            return GateDecision(
                approved=False,
                state=GateState.ESTOP,
                emitted_command="\x18",
                parsed=None,
                violations=["estop_latched_manual_reset_required"],
                latency_ms=inference_latency_ms,
            )

        if (
            not isinstance(inference_latency_ms, (int, float))
            or not math.isfinite(float(inference_latency_ms))
            or float(inference_latency_ms) < 0.0
            or float(inference_latency_ms) > self.config.latency_budget_ms
        ):
            return self._reject_hold(
                f"watchdog_timeout:{inference_latency_ms}",
                float(inference_latency_ms) if isinstance(inference_latency_ms, (int, float)) else 0.0,
            )

        lat_ms = float(inference_latency_ms)

        # Strict JSON parsing (no partial-JSON repair, no markdown fences, no trailing prose)
        if not isinstance(raw_output, str):
            return self._reject_hold("malformed_json:non_string", lat_ms)

        stripped = raw_output.strip()
        if not (stripped.startswith("{") and stripped.endswith("}")):
            return self._reject_hold("malformed_json:not_pure_json_object", lat_ms)

        try:
            obj = json.loads(stripped, parse_constant=_reject_nonfinite)
        except Exception:
            return self._reject_hold("malformed_json:parse_error", lat_ms)

        if not isinstance(obj, dict):
            return self._reject_hold("malformed_json:not_dict", lat_ms)

        try:
            cmd = ControllerCommand.model_validate(obj)
        except ValidationError as exc:
            return self._reject_hold(f"schema_validation_failed:{len(exc.errors())}", lat_ms)

        # Telemetry hardware safety cross-checks
        if telemetry is not None:
            t_dict = telemetry.to_dict() if isinstance(telemetry, TelemetryFrame) else telemetry
            pos = t_dict.get("pos_mm", {})
            env = self.config.envelope
            limit_hit = bool(t_dict.get("limit_triggered", False))
            out_of_env = (
                float(pos.get("x", 0.0)) < env.x_min_mm
                or float(pos.get("x", 0.0)) > env.x_max_mm
                or float(pos.get("y", 0.0)) < env.y_min_mm
                or float(pos.get("y", 0.0)) > env.y_max_mm
                or float(pos.get("z", 0.0)) < env.z_min_mm
                or float(pos.get("z", 0.0)) > env.z_max_mm
            )
            if limit_hit or out_of_env:
                self.state = GateState.ESTOP
                self.estop_reason = "telemetry_limit_breach"
                if cmd.action == "ESTOP" and cmd.gcode in ("M5", "\x18", "!"):
                    return GateDecision(
                        approved=True,
                        state=GateState.ESTOP,
                        emitted_command="\x18",
                        parsed=cmd,
                        violations=[],
                        latency_ms=lat_ms,
                    )
                return self._reject_estop("telemetry_limit_breach_estop", lat_ms, parsed=cmd)

            spindle_t = float(t_dict.get("spindle_temp_c", 0.0))
            mcu_t = float(t_dict.get("mcu_temp_c", 0.0))
            if spindle_t >= self.config.estop_temp_c or mcu_t >= self.config.mcu_temp_ceiling_c:
                self.state = GateState.ESTOP
                self.estop_reason = "critical_thermal_estop"
                if cmd.action == "ESTOP" and cmd.gcode in ("M5", "\x18", "!"):
                    return GateDecision(
                        approved=True,
                        state=GateState.ESTOP,
                        emitted_command="\x18",
                        parsed=cmd,
                        violations=[],
                        latency_ms=lat_ms,
                    )
                return self._reject_estop("critical_thermal_estop", lat_ms, parsed=cmd)

            if spindle_t >= self.config.temp_ceiling_c and cmd.gcode not in ("!", "M5", "\x18"):
                return self._reject_hold(f"thermal_ceiling_exceeded:{spindle_t}C", lat_ms, parsed=cmd)

        # Whitelist + soft-limit validation
        ok, gcode_violations = validate_whitelisted_gcode(cmd.gcode, self.config.envelope)
        if not ok:
            return self._reject_hold(gcode_violations, lat_ms, parsed=cmd)

        # Action-command coherence
        if cmd.action == "FEED_HOLD" and cmd.gcode != "!":
            return self._reject_hold("action_gcode_mismatch:FEED_HOLD", lat_ms, parsed=cmd)
        if cmd.action == "ESTOP" and cmd.gcode not in ("M5", "\x18", "!"):
            return self._reject_estop("action_gcode_mismatch:ESTOP", lat_ms, parsed=cmd)

        # Approved state transition
        if cmd.action == "ESTOP" or cmd.gcode == "\x18":
            self.state = GateState.ESTOP
            self.estop_reason = f"commanded_estop:{cmd.fault_type}"
            emitted = "\x18" if cmd.gcode == "\x18" else cmd.gcode
        elif cmd.action == "FEED_HOLD" or cmd.gcode == "!":
            self.state = GateState.HOLD
            emitted = "!"
        else:
            self.state = GateState.RUNNING
            emitted = cmd.gcode

        return GateDecision(
            approved=True,
            state=self.state,
            emitted_command=emitted,
            parsed=cmd,
            violations=[],
            latency_ms=lat_ms,
        )

    def guard_inference(
        self,
        model_fn: Callable[[], str],
        telemetry: TelemetryFrame | dict[str, Any] | None = None,
    ) -> GateDecision:
        """Execute `model_fn()` under strict wall-clock watchdog and evaluate output."""
        t0 = time.perf_counter()
        try:
            raw_output = model_fn()
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return self._reject_hold(f"inference_exception:{type(exc).__name__}", elapsed_ms)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return self.evaluate(raw_output, telemetry=telemetry, inference_latency_ms=elapsed_ms)
