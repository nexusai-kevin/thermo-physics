"""
thermo_physics.tpu.integrator
=============================
Symplectic Stochastic Integrators and Thermal Annealing Schedulers for Hybrid TPU Simulation.
Features:
1. BAOAB (Brünger-Brooks-Karplus / Leimkuhler-Matthews) splitting scheme for underdamped Langevin dynamics.
2. Euler-Maruyama baseline integrator for numerical instability and stiffness comparison.
3. Adaptive and parametric thermal and barrier schedulers (Linear, Exponential, Cosine, Adaptive Reheat).
4. Full support for non-conservative dynamic clause multiplier vector fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Tuple, Union

import numpy as np

from thermo_physics.tpu.hamiltonian import NonConvexIsingSAT


class ScheduleType(str, Enum):
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    COSINE = "cosine"
    ADAPTIVE = "adaptive"


@dataclass
class ThermalSchedule:
    """Thermal annealing and parameter ramp schedule configuration."""
    t_start: float = 3.0              # Initial high temperature k_B T
    t_end: float = 0.05               # Final low temperature k_B T
    total_steps: int = 5000           # Total number of integration steps
    schedule_type: ScheduleType = ScheduleType.COSINE
    alpha_start: float = 0.2          # Initial confinement barrier height
    alpha_end: float = 2.5            # Final confinement barrier height
    lambda_sat_start: float = 0.5     # Initial 3-SAT penalty
    lambda_sat_end: float = 4.0       # Final 3-SAT penalty
    reheat_threshold: float = 0.05    # Velocity variance trigger for adaptive reheating
    reheat_boost: float = 1.8         # Temperature multiplier upon reheat

    def get_temperature(self, step: int) -> float:
        """Compute bath temperature T(t) at given step."""
        fraction = min(1.0, max(0.0, step / max(1, self.total_steps)))
        if self.schedule_type == ScheduleType.LINEAR:
            return self.t_start - fraction * (self.t_start - self.t_end)
        elif self.schedule_type == ScheduleType.EXPONENTIAL:
            ratio = max(1e-6, self.t_end / max(1e-6, self.t_start))
            return self.t_start * (ratio ** fraction)
        elif self.schedule_type == ScheduleType.COSINE:
            return self.t_end + 0.5 * (self.t_start - self.t_end) * (1.0 + np.cos(np.pi * fraction))
        elif self.schedule_type == ScheduleType.ADAPTIVE:
            return self.t_end + 0.5 * (self.t_start - self.t_end) * (1.0 + np.cos(np.pi * fraction))
        return self.t_end

    def get_barrier_alpha(self, step: int) -> float:
        """Compute confinement barrier alpha(t) at given step."""
        fraction = min(1.0, max(0.0, step / max(1, self.total_steps)))
        return self.alpha_start + fraction * (self.alpha_end - self.alpha_start)

    def get_lambda_sat(self, step: int) -> float:
        """Compute 3-SAT penalty weight lambda_sat(t) at given step."""
        fraction = min(1.0, max(0.0, step / max(1, self.total_steps)))
        return self.lambda_sat_start + fraction * (self.lambda_sat_end - self.lambda_sat_start)


class AdaptiveThermalScheduler:
    """Manages dynamic thermal schedule with optional reheating when ergodicity drops."""

    def __init__(self, schedule: ThermalSchedule):
        self.schedule = schedule
        self.current_step = 0
        self.reheat_multiplier = 1.0
        self.reheat_decay = 0.98

    def step(self, velocity_variance: Optional[float] = None) -> Tuple[float, float, float]:
        """Advance one step and return (temperature, alpha_well, lambda_sat)."""
        base_t = self.schedule.get_temperature(self.current_step)

        # Decay any active reheat multiplier
        self.reheat_multiplier = 1.0 + (self.reheat_multiplier - 1.0) * self.reheat_decay

        # Check for adaptive reheat trigger
        if self.schedule.schedule_type == ScheduleType.ADAPTIVE and velocity_variance is not None:
            if velocity_variance < self.schedule.reheat_threshold and self.current_step < 0.8 * self.schedule.total_steps:
                self.reheat_multiplier = max(self.reheat_multiplier, self.schedule.reheat_boost)

        effective_t = base_t * self.reheat_multiplier
        alpha = self.schedule.get_barrier_alpha(self.current_step)
        lam = self.schedule.get_lambda_sat(self.current_step)

        self.current_step += 1
        return effective_t, alpha, lam


class BAOABIntegrator:
    """BAOAB Symplectic Stochastic Splitting Integrator for Underdamped Langevin Dynamics.

    Continuous SDE:
      dx_i = v_i dt
      m dv_i = (- dH/dx_i - gamma * v_i) dt + sqrt(2 * gamma * k_B * T) dW_i

    BAOAB Step (time step dt):
      B: v <- v + 0.5 * (dt / m) * F(x, t)
      A: x <- x + 0.5 * dt * v
      O: Exact Ornstein-Uhlenbeck:
         c1 = exp(-gamma * dt / m)
         c2 = sqrt(k_B * T / m * (1 - c1^2))
         v <- c1 * v + c2 * xi, where xi ~ N(0, I)
      A: x <- x + 0.5 * dt * v
      B: v <- v + 0.5 * (dt / m) * F(x, t + dt)
    """

    def __init__(
        self,
        dt: float = 0.01,
        gamma: float = 1.0,
        mass: float = 1.0,
        kb: float = 1.0,
    ):
        self.dt = dt
        self.gamma = gamma
        self.mass = mass
        self.kb = kb

    def step(
        self,
        x: np.ndarray,
        v: np.ndarray,
        hamiltonian: NonConvexIsingSAT,
        temperature: float,
        alpha_well: float,
        lambda_sat: float,
        rng: np.random.Generator,
        f_cached: Optional[np.ndarray] = None,
        clause_weights: Optional[np.ndarray] = None,
        colored_noise: Optional[np.ndarray] = None,
        clipper: Optional[Any] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Perform a single BAOAB step for single state or batch of states (B, N)."""
        dt = self.dt
        m = self.mass
        gamma = self.gamma
        half_dt = 0.5 * dt

        # B step: half-step velocity update using current force
        if f_cached is None:
            f = -hamiltonian.gradient(x, alpha_well=alpha_well, lambda_sat=lambda_sat, clause_weights=clause_weights)
        else:
            f = f_cached

        v_b1 = v + (half_dt / m) * f
        if clipper is not None:
            v_b1 = clipper.clip_velocity(v_b1)

        # A step: half-step position update
        x_a1 = x + half_dt * v_b1
        if clipper is not None:
            x_a1 = clipper.clip_position(x_a1)

        # O step: exact Ornstein-Uhlenbeck fluctuation-dissipation (or colored noise bath)
        c1 = np.exp(-gamma * dt / m)
        if colored_noise is not None:
            # Under colored noise eta(t), integration of dv = (-gamma*v + eta) dt / m
            coeff_eta = (1.0 - c1) / gamma if gamma > 1e-12 else (dt / m)
            v_o = c1 * v_b1 + coeff_eta * colored_noise
            noise = colored_noise
        else:
            variance_factor = max(0.0, 1.0 - c1**2)
            c2 = np.sqrt((self.kb * max(0.0, temperature) / m) * variance_factor)
            noise = rng.standard_normal(size=x.shape)
            v_o = c1 * v_b1 + c2 * noise

        if clipper is not None:
            v_o = clipper.clip_velocity(v_o)

        # A step: second half-step position update
        x_next = x_a1 + half_dt * v_o
        if clipper is not None:
            x_next = clipper.clip_position(x_next)

        # B step: second half-step velocity update using force at x_next
        f_next = -hamiltonian.gradient(x_next, alpha_well=alpha_well, lambda_sat=lambda_sat, clause_weights=clause_weights)
        v_next = v_o + (half_dt / m) * f_next
        if clipper is not None:
            v_next = clipper.clip_velocity(v_next)

        return x_next, v_next, f_next, noise


class EulerMaruyamaIntegrator:
    """Standard non-symplectic Euler-Maruyama integrator for comparison and failure stress-testing."""

    def __init__(
        self,
        dt: float = 0.01,
        gamma: float = 1.0,
        mass: float = 1.0,
        kb: float = 1.0,
    ):
        self.dt = dt
        self.gamma = gamma
        self.mass = mass
        self.kb = kb

    def step(
        self,
        x: np.ndarray,
        v: np.ndarray,
        hamiltonian: NonConvexIsingSAT,
        temperature: float,
        alpha_well: float,
        lambda_sat: float,
        rng: np.random.Generator,
        clause_weights: Optional[np.ndarray] = None,
        colored_noise: Optional[np.ndarray] = None,
        clipper: Optional[Any] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Perform Euler-Maruyama step."""
        dt = self.dt
        m = self.mass
        gamma = self.gamma

        f = -hamiltonian.gradient(x, alpha_well=alpha_well, lambda_sat=lambda_sat, clause_weights=clause_weights)
        if colored_noise is not None:
            noise = colored_noise
            v_next = v + (dt / m) * (f - gamma * v + colored_noise)
        else:
            noise = rng.standard_normal(size=x.shape)
            diffusion = np.sqrt(2.0 * gamma * self.kb * max(0.0, temperature) * dt) / m
            v_next = v + (dt / m) * (f - gamma * v) + diffusion * noise

        x_next = x + v * dt
        if clipper is not None:
            x_next = clipper.clip_position(x_next)
            v_next = clipper.clip_velocity(v_next)

        f_next = -hamiltonian.gradient(x_next, alpha_well=alpha_well, lambda_sat=lambda_sat, clause_weights=clause_weights)
        return x_next, v_next, f_next, noise
