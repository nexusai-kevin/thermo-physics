"""
thermo_physics.tpu.hardware
===========================
Silicon Realism & Hardware Imperfection Module.
Simulates physical analog hardware imperfections in Thermodynamic Processing Units (TPUs)
and stochastic p-bit crossbar arrays:
1. Finite-Resolution DAC Quantization (4-bit, 6-bit, 8-bit fixed-point representation of
   couplings J_ij, biases h_i, and dynamic auxiliary clause weights y_m(t)).
2. Operational Amplifier Voltage Rail Clipping (physical saturation of state voltages
   x_i in [-V_rail, V_rail], velocity slew rates in [-V_max, V_max], and multiplier ceilings).
3. Colored 1/f Flicker Noise Bath (Ornstein-Uhlenbeck stochastic processes with finite
   autocorrelation time tau_color, replacing ideal white Wiener noise increments).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from thermo_physics.tpu.hamiltonian import NonConvexIsingSAT, ProblemConfig


@dataclass
class SiliconRealismConfig:
    """Hardware imperfection parameters for physical silicon TPU simulation."""
    # DAC Quantization
    dac_bits: Optional[int] = None       # None or 32 for float, 4, 6, 8 for fixed-point
    quantize_hamiltonian: bool = True     # Quantize J and h
    quantize_dynamic_multipliers: bool = True  # Quantize y_m(t) during integration
    dac_w_max: Optional[float] = None     # Optional explicit dynamic range clamp for DAC

    # Op-Amp Saturation & Rail Clipping
    enable_rail_clipping: bool = False    # Enable physical voltage saturation
    v_rail: float = 1.5                  # Supply rail voltage in Volts: x in [-V_rail, V_rail]
    v_max: float = 10.0                  # Slew-rate velocity saturation: v in [-V_max, V_max]
    y_max: float = 15.0                  # Dynamic multiplier capacitor saturation: y in [lambda_0, y_max]

    # Colored 1/f Flicker Noise (Ornstein-Uhlenbeck Bath)
    enable_colored_noise: bool = False    # Replace white Wiener noise with colored noise
    tau_color: float = 0.05              # Noise correlation time (seconds / simulation units)
    use_pink_noise_bank: bool = False    # Sum-of-OUs multi-decade 1/f pink noise synthesis


class DACQuantizer:
    """Finite-Resolution Digital-to-Analog Converter (DAC) fixed-point quantizer."""

    @staticmethod
    def quantize(
        w: np.ndarray,
        bits: Optional[int] = None,
        w_min: Optional[float] = None,
        w_max: Optional[float] = None,
    ) -> np.ndarray:
        """Discretize continuous weights into K-bit fixed-point representation.

        Formula:
          Delta_DAC = (w_max - w_min) / (2^K - 1)
          w_quant = w_min + Delta_DAC * round((clip(w, w_min, w_max) - w_min) / Delta_DAC)
        """
        if bits is None or bits >= 32:
            return w.copy() if isinstance(w, np.ndarray) else np.asarray(w, dtype=np.float64)

        if bits < 1:
            raise ValueError(f"DAC bits must be >= 1, got {bits}")

        w_arr = np.asarray(w, dtype=np.float64)
        if w_arr.size == 0:
            return w_arr.copy()

        actual_min = float(np.min(w_arr)) if w_min is None else float(w_min)
        actual_max = float(np.max(w_arr)) if w_max is None else float(w_max)

        if actual_max <= actual_min:
            return np.full_like(w_arr, actual_min)

        levels = (1 << bits) - 1
        delta = (actual_max - actual_min) / float(levels)

        if delta < 1e-15:
            return w_arr.copy()

        clipped = np.clip(w_arr, actual_min, actual_max)
        normalized = (clipped - actual_min) / delta
        quantized = actual_min + np.round(normalized) * delta
        return quantized

    @staticmethod
    def quantize_symmetric(
        w: np.ndarray,
        bits: Optional[int] = None,
        max_val: Optional[float] = None,
    ) -> np.ndarray:
        """Symmetric zero-centered quantization for bipolar couplings J_ij and biases h_i.

        Formula:
          Delta_DAC = (w_max - w_min) / (2^K - 1) where w_max = -w_min = max_val
          w_quant = Delta_DAC * round(clip(w, -max_val, max_val) / Delta_DAC)
        """
        if bits is None or bits >= 32:
            return w.copy() if isinstance(w, np.ndarray) else np.asarray(w, dtype=np.float64)

        w_arr = np.asarray(w, dtype=np.float64)
        if w_arr.size == 0:
            return w_arr.copy()

        bound = float(np.max(np.abs(w_arr))) if max_val is None else float(abs(max_val))
        if bound <= 1e-15:
            return np.zeros_like(w_arr)

        return DACQuantizer.quantize(w_arr, bits=bits, w_min=-bound, w_max=bound)

    @classmethod
    def quantize_hamiltonian(
        cls,
        hamiltonian: NonConvexIsingSAT,
        bits: Optional[int] = None,
        max_val: Optional[float] = None,
    ) -> NonConvexIsingSAT:
        """Create a new NonConvexIsingSAT with DAC-quantized J matrix and h bias vector."""
        if bits is None or bits >= 32:
            return hamiltonian

        j_mat = getattr(hamiltonian, "j_matrix", getattr(hamiltonian, "j", None))
        h_vec = getattr(hamiltonian, "h_bias", getattr(hamiltonian, "h", None))

        j_quant = cls.quantize_symmetric(j_mat, bits=bits, max_val=max_val)
        # Ensure zero diagonal and exact symmetry
        np.fill_diagonal(j_quant, 0.0)
        j_quant = 0.5 * (j_quant + j_quant.T)

        h_quant = cls.quantize_symmetric(h_vec, bits=bits, max_val=max_val)

        import copy
        new_config = copy.deepcopy(hamiltonian.config)

        return NonConvexIsingSAT(
            num_variables=hamiltonian.n,
            j_matrix=j_quant,
            h_bias=h_quant,
            clauses=list(hamiltonian.clauses),
            config=new_config,
        )


class OpAmpVoltageClipper:
    """Operational Amplifier Saturation & Voltage Rail Limiter.

    Models physical power supply limits:
    - Position voltages clamped to [-V_rail, V_rail] (e.g. +/- 1.5 V)
    - Velocities clamped to [-V_max, V_max] (slew rate)
    - Dynamic multipliers clamped to [lambda_0, Y_max] (capacitor saturation)
    """

    def __init__(
        self,
        v_rail: float = 1.5,
        v_max: float = 10.0,
        y_max: float = 15.0,
        lambda_0: float = 1.0,
    ):
        self.v_rail = float(v_rail)
        self.v_max = float(v_max)
        self.y_max = float(y_max)
        self.lambda_0 = float(lambda_0)

        # Telemetry counters
        self.position_clips: int = 0
        self.velocity_clips: int = 0
        self.multiplier_clips: int = 0

    def clip_position(self, x: np.ndarray) -> np.ndarray:
        """Saturate state coordinates x_i to power supply rails [-V_rail, V_rail]."""
        clipped = np.clip(x, -self.v_rail, self.v_rail)
        self.position_clips += int(np.sum(np.abs(x) > self.v_rail))
        return clipped

    def clip_velocity(self, v: np.ndarray) -> np.ndarray:
        """Saturate velocity / current slew rate to [-V_max, V_max]."""
        clipped = np.clip(v, -self.v_max, self.v_max)
        self.velocity_clips += int(np.sum(np.abs(v) > self.v_max))
        return clipped

    def clip_multiplier(self, y: np.ndarray) -> np.ndarray:
        """Saturate auxiliary dynamic clause multipliers to [lambda_0, y_max]."""
        clipped = np.clip(y, self.lambda_0, self.y_max)
        self.multiplier_clips += int(np.sum(y > self.y_max))
        return clipped

    def clip_state(
        self, x: np.ndarray, v: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply combined voltage rail and velocity clipping."""
        return self.clip_position(x), self.clip_velocity(v)


class OrnsteinUhlenbeckNoise:
    """Colored 1/f Flicker Noise Bath (Ornstein-Uhlenbeck process).

    Governed by the stochastic differential equation:
      d eta_i = - (1 / tau_color) * eta_i * dt + sigma_color * dW_i

    Stationary variance:
      <eta_i^2> = sigma_eta^2 = (gamma * k_B * T) / tau_color
    which ensures the low-frequency noise power matches the white noise thermal bath (2 * gamma * k_B * T).

    Exact analytical transition over interval dt:
      eta(t+dt) = eta(t) * exp(-dt / tau_color) + sqrt(sigma_eta^2 * (1 - exp(-2*dt/tau_color))) * xi
      where xi ~ N(0, I).
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        tau_color: float = 0.05,
        use_pink_noise_bank: bool = False,
    ):
        self.shape = shape
        self.tau_color = float(max(1e-5, tau_color))
        self.use_pink_noise_bank = use_pink_noise_bank

        if use_pink_noise_bank:
            # 3-pole sum-of-OUs with log-spaced time constants covering ~3 decades
            self.tau_poles = [self.tau_color * 0.2, self.tau_color * 1.0, self.tau_color * 5.0]
            self.weights = [0.333, 0.333, 0.334]
            self.eta_bank = [np.zeros(shape, dtype=np.float64) for _ in self.tau_poles]
        else:
            self.tau_poles = [self.tau_color]
            self.weights = [1.0]
            self.eta_bank = [np.zeros(shape, dtype=np.float64)]

    @property
    def current_noise(self) -> np.ndarray:
        """Return instantaneous colored noise vector."""
        if self.use_pink_noise_bank:
            return sum(w * e for w, e in zip(self.weights, self.eta_bank))
        return self.eta_bank[0]

    def reset(self, rng: Optional[np.random.Generator] = None) -> None:
        """Reset noise state to zero or stationary distribution."""
        for i in range(len(self.eta_bank)):
            self.eta_bank[i].fill(0.0)

    def step(
        self,
        dt: float,
        temperature: float,
        gamma: float,
        kb: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Advance the colored noise state by timestep dt and return current noise realization.

        Matches thermodynamic equilibrium bath power:
          sigma_k^2 = (gamma * kb * T) / tau_k
        """
        temp = max(0.0, temperature)

        for i, tau in enumerate(self.tau_poles):
            c1 = np.exp(-dt / tau)
            var_stationary = (gamma * kb * temp) / tau
            noise_std = np.sqrt(max(0.0, var_stationary * (1.0 - c1**2)))

            xi = rng.standard_normal(size=self.shape)
            self.eta_bank[i] = c1 * self.eta_bank[i] + noise_std * xi

        return self.current_noise
