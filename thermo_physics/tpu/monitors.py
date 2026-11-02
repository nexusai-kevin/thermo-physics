"""
thermo_physics.tpu.monitors
===========================
Runtime Diagnostics, Failure Detection, and Stability Monitors for Hybrid TPU Simulation.
Flags:
1. Numerical Divergence: NaNs, Infinities, coordinate overflows, and unphysical kinetic spikes.
2. Ergodicity Breaking & Metastable Trapping: Identifies frozen spins trapped in local non-convex wells.
3. Fluctuation-Dissipation Theorem (FDT) Violations:
   - Kinetic Temperature: T_kin = (m / (k_B * N)) * sum_i v_i^2
   - Configurational Temperature: k_B T_conf = <||grad H(x)||^2> / <Delta H(x)>
   Tracks kinetic vs configurational temperature divergence under high stiffness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from thermo_physics.tpu.hamiltonian import NonConvexIsingSAT


class NumericalDivergenceError(RuntimeError):
    """Raised or logged when stochastic integration encounters numerical instability (NaN/Inf/Overflow)."""
    pass


class ErgodicityTrapWarning:
    """Diagnostic notice when an ensemble trajectory is frozen in a metastable false attractor."""
    def __init__(self, step: int, trapped_indices: List[int], energy: float, variance: float):
        self.step = step
        self.trapped_indices = trapped_indices
        self.energy = energy
        self.variance = variance

    def __repr__(self) -> str:
        return (
            f"ErgodicityTrapWarning(step={self.step}, trapped={len(self.trapped_indices)} spins, "
            f"energy={self.energy:.3f}, var={self.variance:.2e})"
        )


class FDTViolationWarning:
    """Diagnostic notice when equipartition theorem or configurational temperature is violated beyond tolerance."""
    def __init__(
        self,
        step: int,
        t_bath: float,
        t_kinetic: float,
        t_configurational: float,
        relative_error: float,
    ):
        self.step = step
        self.t_bath = t_bath
        self.t_kinetic = t_kinetic
        self.t_configurational = t_configurational
        self.relative_error = relative_error

    def __repr__(self) -> str:
        return (
            f"FDTViolationWarning(step={self.step}, T_bath={self.t_bath:.3f}, "
            f"T_kin={self.t_kinetic:.3f}, T_conf={self.t_configurational:.3f}, "
            f"err={self.relative_error * 100:.1f}%)"
        )


@dataclass
class DiagnosticReport:
    """Summary of all runtime diagnostics across the simulation."""
    divergence_detected: bool = False
    divergence_step: Optional[int] = None
    divergence_reason: Optional[str] = None
    ergodicity_traps_count: int = 0
    ergodicity_warnings: List[ErgodicityTrapWarning] = field(default_factory=list)
    fdt_violations_count: int = 0
    fdt_warnings: List[FDTViolationWarning] = field(default_factory=list)
    max_fdt_relative_error: float = 0.0
    mean_fdt_relative_error: float = 0.0
    mean_t_conf_relative_error: float = 0.0
    mean_t_conf: float = 0.0
    mean_t_kin: float = 0.0
    mean_t_bath: float = 0.0
    min_observed_temperature: float = float("inf")
    max_observed_temperature: float = 0.0
    t_conf_history: List[float] = field(default_factory=list)
    t_kin_history: List[float] = field(default_factory=list)
    t_bath_history: List[float] = field(default_factory=list)


class DiagnosticMonitor:
    """Active runtime monitor inspecting trajectory health at each timestep."""

    def __init__(
        self,
        num_variables: int,
        batch_size: int,
        hamiltonian: Optional[NonConvexIsingSAT] = None,
        mass: float = 1.0,
        kb: float = 1.0,
        x_max: float = 25.0,
        v_max: float = 50.0,
        fdt_tolerance: float = 0.40,
        ergodicity_window: int = 200,
        ergodicity_var_threshold: float = 1e-4,
    ):
        self.n = num_variables
        self.b = batch_size
        self.hamiltonian = hamiltonian
        self.mass = mass
        self.kb = kb
        self.x_max = x_max
        self.v_max = v_max
        self.fdt_tolerance = fdt_tolerance
        self.ergodicity_window = ergodicity_window
        self.ergodicity_var_threshold = ergodicity_var_threshold

        self.report = DiagnosticReport()

        self._pos_buffer: List[np.ndarray] = []
        self._fdt_errors: List[float] = []
        self._t_conf_errors: List[float] = []
        self._t_confs: List[float] = []
        self._t_kins: List[float] = []
        self._t_baths: List[float] = []

    def check_step(
        self,
        step: int,
        x: np.ndarray,
        v: np.ndarray,
        t_bath: float,
        energy: Optional[np.ndarray] = None,
        alpha_well: Optional[float] = None,
        lambda_sat: Optional[float] = None,
        force_cached: Optional[np.ndarray] = None,
    ) -> None:
        """Inspect current state and record any numerical divergence, ergodicity traps, or FDT violations."""
        # 1. Numerical Divergence Check
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(v)):
            self.report.divergence_detected = True
            self.report.divergence_step = step
            self.report.divergence_reason = f"Non-finite value (NaN or Inf) detected in state at step {step}"
            return

        max_x = float(np.max(np.abs(x)))
        max_v = float(np.max(np.abs(v)))
        if max_x > self.x_max:
            self.report.divergence_detected = True
            self.report.divergence_step = step
            self.report.divergence_reason = f"Position overflow |x| = {max_x:.2f} > {self.x_max} at step {step}"
            return
        if max_v > self.v_max:
            self.report.divergence_detected = True
            self.report.divergence_step = step
            self.report.divergence_reason = f"Velocity explosion |v| = {max_v:.2f} > {self.v_max} at step {step}"
            return

        # 2. Kinetic Temperature: T_kin = (m / (kb * N)) * sum_i v_i^2
        ke_per_sample = 0.5 * self.mass * np.sum(v**2, axis=-1)
        t_kin_per_sample = (2.0 * ke_per_sample) / (self.n * self.kb)
        mean_t_kin = float(np.mean(t_kin_per_sample))

        self.report.min_observed_temperature = min(self.report.min_observed_temperature, mean_t_kin)
        self.report.max_observed_temperature = max(self.report.max_observed_temperature, mean_t_kin)

        # 3. Configurational Temperature: k_B T_conf = <||grad H||^2> / <Delta H>
        mean_t_conf = mean_t_kin  # fallback if hamiltonian not supplied
        if self.hamiltonian is not None:
            if force_cached is not None:
                grad = -force_cached
            else:
                grad = self.hamiltonian.gradient(x, alpha_well=alpha_well, lambda_sat=lambda_sat)

            grad_sq = np.sum(grad**2, axis=-1)  # shape (B,)
            laplacian = self.hamiltonian.laplacian(x, alpha_well=alpha_well, lambda_sat=lambda_sat)  # shape (B,)

            mean_grad_sq = float(np.mean(grad_sq))
            mean_laplacian = float(np.mean(laplacian))

            if abs(mean_laplacian) > 1e-6:
                mean_t_conf = float(mean_grad_sq / (self.kb * mean_laplacian))
            else:
                mean_t_conf = mean_t_kin

        self._t_confs.append(mean_t_conf)
        self._t_kins.append(mean_t_kin)
        self._t_baths.append(t_bath)

        if t_bath > 1e-3:
            rel_err_kin = abs(mean_t_kin - t_bath) / t_bath
            rel_err_conf = abs(mean_t_conf - t_bath) / t_bath
            self._fdt_errors.append(rel_err_kin)
            self._t_conf_errors.append(rel_err_conf)

            self.report.max_fdt_relative_error = max(self.report.max_fdt_relative_error, rel_err_kin)

            # Flag violation only after initial transient steps
            if step > 100 and rel_err_kin > self.fdt_tolerance:
                self.report.fdt_violations_count += 1
                if len(self.report.fdt_warnings) < 10:
                    self.report.fdt_warnings.append(
                        FDTViolationWarning(step, t_bath, mean_t_kin, mean_t_conf, rel_err_kin)
                    )

        # 4. Ergodicity Breaking / Metastable Trapping Check
        self._pos_buffer.append(x.copy())
        if len(self._pos_buffer) > self.ergodicity_window:
            self._pos_buffer.pop(0)

            stacked = np.stack(self._pos_buffer, axis=0)
            spin_vars = np.var(stacked, axis=0)  # shape (B, N)
            mean_spin_var = float(np.mean(spin_vars))

            if mean_spin_var < self.ergodicity_var_threshold and step < 0.7 * self.ergodicity_window * 10:
                current_e = float(np.mean(energy)) if energy is not None else 0.0
                trapped_idx = list(np.where(np.mean(spin_vars, axis=0) < self.ergodicity_var_threshold)[0])
                self.report.ergodicity_traps_count += 1
                if len(self.report.ergodicity_warnings) < 10:
                    self.report.ergodicity_warnings.append(
                        ErgodicityTrapWarning(step, trapped_idx, current_e, mean_spin_var)
                    )

    def finalize(self) -> DiagnosticReport:
        """Finalize diagnostic summaries."""
        if self._fdt_errors:
            self.report.mean_fdt_relative_error = float(np.mean(self._fdt_errors))
        if self._t_conf_errors:
            self.report.mean_t_conf_relative_error = float(np.mean(self._t_conf_errors))
        if self._t_confs:
            self.report.mean_t_conf = float(np.mean(self._t_confs))
            self.report.mean_t_kin = float(np.mean(self._t_kins))
            self.report.mean_t_bath = float(np.mean(self._t_baths))
            self.report.t_conf_history = list(self._t_confs)
            self.report.t_kin_history = list(self._t_kins)
            self.report.t_bath_history = list(self._t_baths)
        return self.report
