"""
thermo_physics.tpu.jax_silicon
==============================
Phase 2: Differentiable JAX Primitives for Silicon Realism & Analog Imperfections.

Provides custom JAX forward/backward primitives (@jax.custom_vjp) and vectorized
stochastic noise generators to simulate and train Energy-Based Models (EBMs) under
physical Thermodynamic Sampling Unit (TSU) hardware constraints:
1. Finite-Resolution DAC Quantization (`jax_dac_quantize`, `jax_dac_quantize_symmetric`)
   with Straight-Through Estimator (STE) backward pass.
2. Op-Amp Voltage Rail Clipping (`jax_opamp_clip`) with smooth surrogate hyperbolic-secant
   backward pass to prevent dead gradients during saturated analog training.
3. 1/f Pink Noise (3-Pole Ornstein-Uhlenbeck Bank) + Thermal Gaussian Bath
   (`JaxOrnsteinUhlenbeckState`, `jax_step_pink_thermal_noise`).
4. Vectorized JAX Silicon Realism Sampler (`JaxSiliconRealismSampler`) implementing
   `BackendType.SIMULATED_NOISE` for `sampler_adapter.py`.
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass, field
from typing import Any, Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np


# ============================================================================
# STEP 1: CUSTOM JAX PRIMITIVES FOR ANALOG IMPERFECTIONS (FORWARD + BACKWARD)
# ============================================================================

@functools.partial(jax.custom_vjp, nondiff_argnums=(1,))
def jax_dac_quantize_symmetric(w: jnp.ndarray, bits: int, max_val: float) -> jnp.ndarray:
    """Symmetric zero-centered K-bit DAC quantization with Straight-Through Estimator (STE).

    Forward Pass:
      Discretizes continuous tensor `w` into (2^K - 1) uniform analog voltage/conductance
      steps over [-max_val, +max_val].
    Backward Pass:
      Passes gradients straight through within [-max_val, +max_val] (clipped STE) so
      noise-aware EBM training can optimize weights through discrete DAC steps.
    """
    if bits <= 0 or bits >= 32:
        return w
    bound = jnp.maximum(jnp.abs(max_val), 1e-12)
    levels = float((1 << bits) - 1)
    delta = (2.0 * bound) / levels
    clipped = jnp.clip(w, -bound, bound)
    quantized = -bound + delta * jnp.round((clipped + bound) / delta)
    return quantized


def _dac_quantize_fwd(
    w: jnp.ndarray, bits: int, max_val: float
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, float]]:
    out = jax_dac_quantize_symmetric(w, bits, max_val)
    return out, (w, max_val)


def _dac_quantize_bwd(
    bits: int,
    res: Tuple[jnp.ndarray, float],
    g: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    w, max_val = res
    bound = jnp.maximum(jnp.abs(max_val), 1e-12)
    mask = jnp.where(jnp.abs(w) <= bound, 1.0, 0.0)
    grad_w = g * mask
    # Gradient w.r.t max_val (zero for fixed hardware rail)
    grad_max_val = jnp.sum(jnp.zeros_like(g))
    return grad_w, grad_max_val


jax_dac_quantize_symmetric.defvjp(_dac_quantize_fwd, _dac_quantize_bwd)


@jax.custom_vjp
def jax_opamp_clip(x: jnp.ndarray, v_rail: float) -> jnp.ndarray:
    """Op-Amp physical voltage rail saturation in [-v_rail, +v_rail] with surrogate gradient.

    Forward Pass:
      Hard physical voltage rail clipping: clip(x, -v_rail, +v_rail).
    Backward Pass:
      Uses a smooth sech^2 surrogate derivative outside the linear regime (with leak_alpha=0.05)
      so trajectories that transiently hit the op-amp supply rails still receive a restoring
      gradient during backprop / contrastive divergence.
    """
    rail = jnp.maximum(jnp.abs(v_rail), 1e-6)
    return jnp.clip(x, -rail, rail)


def _opamp_clip_fwd(
    x: jnp.ndarray, v_rail: float
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, float]]:
    return jax_opamp_clip(x, v_rail), (x, v_rail)


def _opamp_clip_bwd(
    res: Tuple[jnp.ndarray, float],
    g: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    x, v_rail = res
    rail = jnp.maximum(jnp.abs(v_rail), 1e-6)
    in_linear_region = jnp.abs(x) <= rail
    # Smooth sech^2 decay beyond the rail + 5% leak floor
    normalized_excess = (jnp.abs(x) - rail) / rail
    surrogate_slope = 0.05 + 0.95 * (1.0 / jnp.cosh(normalized_excess)) ** 2
    slope = jnp.where(in_linear_region, 1.0, surrogate_slope)
    grad_x = g * slope
    grad_rail = jnp.sum(jnp.where(x > rail, g, jnp.where(x < -rail, -g, 0.0)))
    return grad_x, grad_rail


jax_opamp_clip.defvjp(_opamp_clip_fwd, _opamp_clip_bwd)


# ============================================================================
# STEP 2: JAX 1/F PINK NOISE + THERMAL GAUSSIAN BATH GENERATOR
# ============================================================================

class JaxPinkNoiseState(NamedTuple):
    """Functional state container for 3-pole Ornstein-Uhlenbeck 1/f pink noise + thermal bath."""
    eta_pole1: jnp.ndarray  # Fast pole: 0.2 * tau_color
    eta_pole2: jnp.ndarray  # Mid pole:  1.0 * tau_color
    eta_pole3: jnp.ndarray  # Slow pole: 5.0 * tau_color
    combined_pink: jnp.ndarray
    thermal_white: jnp.ndarray
    total_noise: jnp.ndarray


def init_jax_pink_noise_state(shape: Tuple[int, ...], dtype: Any = jnp.float32) -> JaxPinkNoiseState:
    """Initialize functional 3-pole Ornstein-Uhlenbeck pink noise state in JAX."""
    zeros = jnp.zeros(shape, dtype=dtype)
    return JaxPinkNoiseState(
        eta_pole1=zeros,
        eta_pole2=zeros,
        eta_pole3=zeros,
        combined_pink=zeros,
        thermal_white=zeros,
        total_noise=zeros,
    )


def jax_step_pink_thermal_noise(
    state: JaxPinkNoiseState,
    key: jax.Array,
    dt: float = 0.015,
    temperature: float = 1.0,
    gamma: float = 1.0,
    tau_color: float = 0.05,
    pink_fraction: float = 0.5,
) -> Tuple[JaxPinkNoiseState, jax.Array]:
    """Advance the 3-pole 1/f pink noise bank and thermal Gaussian bath by `dt`.

    Combines:
      1. Multi-decade 1/f flicker noise synthesized via a 3-pole Ornstein-Uhlenbeck bank
         with log-spaced correlation times [0.2 * tau, 1.0 * tau, 5.0 * tau].
      2. White Johnson-Nyquist thermal Gaussian noise N(0, 2 * gamma * k_B * T / dt).
    """
    k1, k2, k3, k_white, next_key = jax.random.split(key, 5)
    temp = jnp.maximum(0.0, temperature)
    tau_base = jnp.maximum(1e-5, tau_color)

    tau1 = tau_base * 0.2
    tau2 = tau_base * 1.0
    tau3 = tau_base * 5.0

    def _ou_step(eta: jnp.ndarray, tau: jnp.ndarray, subkey: jax.Array) -> jnp.ndarray:
        c1 = jnp.exp(-dt / tau)
        var_stat = (gamma * temp) / tau
        std = jnp.sqrt(jnp.maximum(0.0, var_stat * (1.0 - c1 ** 2)))
        xi = jax.random.normal(subkey, shape=eta.shape, dtype=eta.dtype)
        return c1 * eta + std * xi

    p1 = _ou_step(state.eta_pole1, tau1, k1)
    p2 = _ou_step(state.eta_pole2, tau2, k2)
    p3 = _ou_step(state.eta_pole3, tau3, k3)

    # Equal-weight superposition across 3 decades yields 1/f^alpha (alpha ~ 1) spectrum
    pink = (p1 + p2 + p3) / 3.0

    # White thermal noise realization scaled to match stationary bath variance
    thermal_std = jnp.sqrt(jnp.maximum(0.0, 2.0 * gamma * temp))
    white = thermal_std * jax.random.normal(k_white, shape=p1.shape, dtype=p1.dtype)

    pf = jnp.clip(pink_fraction, 0.0, 1.0)
    # Normalize pink contribution so total effective bath power stays calibrated
    pink_norm = pink * jnp.sqrt(tau_base)
    total = pf * pink_norm + (1.0 - pf) * white

    new_state = JaxPinkNoiseState(
        eta_pole1=p1,
        eta_pole2=p2,
        eta_pole3=p3,
        combined_pink=pink,
        thermal_white=white,
        total_noise=total,
    )
    return new_state, next_key


# ============================================================================
# JAX SILICON REALISM TELEMETRY & LANGEVIN / GLAUBER STEP
# ============================================================================

@dataclass
class JaxSiliconTelemetry:
    """Telemetry metrics captured during a JAX silicon-realistic simulation run."""
    dac_bits: int
    v_rail: float
    tau_color: float
    pink_fraction: float
    j_quantization_rms_error: float
    h_quantization_rms_error: float
    total_rail_clips: int
    rail_clip_fraction: float
    mean_pink_noise_power: float
    mean_thermal_noise_power: float
    housekeeping_heat_joules: float
    step_clip_history: list[float] = field(default_factory=list)
    step_noise_history: list[float] = field(default_factory=list)
    step_energy_history: list[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dac_bits": self.dac_bits,
            "v_rail": self.v_rail,
            "tau_color": self.tau_color,
            "pink_fraction": self.pink_fraction,
            "j_quantization_rms_error": round(self.j_quantization_rms_error, 6),
            "h_quantization_rms_error": round(self.h_quantization_rms_error, 6),
            "total_rail_clips": int(self.total_rail_clips),
            "rail_clip_fraction": round(self.rail_clip_fraction, 6),
            "mean_pink_noise_power": round(self.mean_pink_noise_power, 6),
            "mean_thermal_noise_power": round(self.mean_thermal_noise_power, 6),
            "housekeeping_heat_joules": round(self.housekeeping_heat_joules, 6),
            "step_clip_history": [round(float(x), 4) for x in self.step_clip_history[-64:]],
            "step_noise_history": [round(float(x), 4) for x in self.step_noise_history[-64:]],
            "step_energy_history": [round(float(x), 4) for x in self.step_energy_history[-64:]],
        }


def run_jax_silicon_langevin(
    couplings: np.ndarray,
    biases: np.ndarray,
    n_samples: int = 500,
    burn_in: int = 100,
    stride: int = 1,
    beta: float = 1.0,
    dac_bits: int = 6,
    v_rail: float = 1.5,
    tau_color: float = 0.05,
    pink_fraction: float = 0.5,
    dt: float = 0.02,
    seed: int = 42,
    external_noise_injection: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, JaxSiliconTelemetry]:
    """Run continuous-voltage analog subthreshold p-bit Langevin dynamics in JAX
    with DAC quantization, Op-Amp voltage rail clipping, and 1/f + thermal noise.

    Returns:
      (discrete_samples, continuous_voltages, silicon_telemetry)
    """
    n_nodes = biases.shape[0]
    j_orig = jnp.asarray(couplings, dtype=jnp.float32)
    h_orig = jnp.asarray(biases, dtype=jnp.float32)

    j_max = float(jnp.maximum(jnp.max(jnp.abs(j_orig)), 1e-6))
    h_max = float(jnp.maximum(jnp.max(jnp.abs(h_orig)), 1e-6))

    # 1. Apply Differentiable K-bit DAC Quantization to J and h
    if dac_bits and dac_bits < 32:
        j_quant = jax_dac_quantize_symmetric(j_orig, dac_bits, j_max)
        # Preserve zero diagonal and symmetry
        j_quant = 0.5 * (j_quant + j_quant.T)
        j_quant = j_quant * (1.0 - jnp.eye(n_nodes, dtype=jnp.float32))
        h_quant = jax_dac_quantize_symmetric(h_orig, dac_bits, h_max)

        j_rms = float(jnp.sqrt(jnp.mean((j_orig - j_quant) ** 2)) / j_max)
        h_rms = float(jnp.sqrt(jnp.mean((h_orig - h_quant) ** 2)) / h_max)
    else:
        j_quant = j_orig
        h_quant = h_orig
        j_rms = 0.0
        h_rms = 0.0

    temperature = 1.0 / max(1e-4, float(beta))
    total_steps = burn_in + (n_samples * stride)

    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    x_init = jax.random.uniform(init_key, shape=(n_nodes,), minval=-0.5 * v_rail, maxval=0.5 * v_rail)
    noise_state_init = init_jax_pink_noise_state((n_nodes,), dtype=jnp.float32)

    if external_noise_injection is not None:
        ext_arr = jnp.asarray(external_noise_injection, dtype=jnp.float32)
        if ext_arr.shape[0] < total_steps:
            reps = (total_steps // ext_arr.shape[0]) + 1
            ext_arr = jnp.tile(ext_arr, (reps, 1))[:total_steps, :n_nodes]
        else:
            ext_arr = ext_arr[:total_steps, :n_nodes]
    else:
        ext_arr = jnp.zeros((total_steps, n_nodes), dtype=jnp.float32)

    def scan_step(carry, ext_noise_vec):
        x, n_state, k = carry
        # Step 1/f pink + thermal Gaussian noise bath
        n_state, k = jax_step_pink_thermal_noise(
            n_state,
            k,
            dt=dt,
            temperature=temperature,
            gamma=1.0,
            tau_color=tau_color,
            pink_fraction=pink_fraction,
        )

        # Analog transconductance readout with tanh p-bit activation
        s_soft = jnp.tanh(x * 1.5)
        analog_field = jnp.dot(j_quant, s_soft) + h_quant
        if dac_bits and dac_bits < 32:
            field_bound = jnp.maximum(jnp.max(jnp.abs(analog_field)), 1.0)
            analog_field = jax_dac_quantize_symmetric(analog_field, dac_bits, field_bound)

        # Double-well confining restoring force: -alpha * (x^3 - x)
        restoring_force = -1.2 * (x ** 3 - x)
        drift = restoring_force + analog_field

        # Combined stochastic noise + optional external edge-node hardware entropy
        noise_force = n_state.total_noise + ext_noise_vec
        x_unclipped = x + dt * drift + jnp.sqrt(dt) * noise_force

        # Op-Amp voltage rail clipping in [-v_rail, +v_rail]
        clip_mask = jnp.abs(x_unclipped) > v_rail
        clip_count = jnp.sum(clip_mask.astype(jnp.int32))
        x_next = jax_opamp_clip(x_unclipped, v_rail)

        # Housekeeping heat dissipated by rail saturation and non-conservative drift
        q_hk_step = jnp.sum((x_unclipped - x_next) ** 2) / dt

        # Discrete Ising readout via comparator sign(x_next)
        s_disc = jnp.where(x_next >= 0.0, 1.0, -1.0)
        e_step = -0.5 * jnp.sum((s_disc @ j_orig) * s_disc) - jnp.sum(h_orig * s_disc)

        pink_pwr = jnp.mean(n_state.combined_pink ** 2)
        therm_pwr = jnp.mean(n_state.thermal_white ** 2)

        metrics = (s_disc, x_next, clip_count, pink_pwr, therm_pwr, q_hk_step, e_step)
        return (x_next, n_state, k), metrics

    _, (s_trace, x_trace, clip_counts, pink_pwrs, therm_pwrs, q_hks, e_trace) = jax.lax.scan(
        scan_step,
        (x_init, noise_state_init, key),
        ext_arr,
    )
    s_trace.block_until_ready()

    valid_s = np.asarray(s_trace[burn_in::stride][:n_samples])
    valid_x = np.asarray(x_trace[burn_in::stride][:n_samples])

    clip_counts_np = np.asarray(clip_counts)
    total_clips = int(np.sum(clip_counts_np))
    clip_fraction = float(total_clips) / float(max(1, total_steps * n_nodes))

    telemetry = JaxSiliconTelemetry(
        dac_bits=dac_bits if dac_bits else 32,
        v_rail=float(v_rail),
        tau_color=float(tau_color),
        pink_fraction=float(pink_fraction),
        j_quantization_rms_error=j_rms,
        h_quantization_rms_error=h_rms,
        total_rail_clips=total_clips,
        rail_clip_fraction=clip_fraction,
        mean_pink_noise_power=float(np.mean(np.asarray(pink_pwrs))),
        mean_thermal_noise_power=float(np.mean(np.asarray(therm_pwrs))),
        housekeeping_heat_joules=float(np.sum(np.asarray(q_hks))),
        step_clip_history=(clip_counts_np / float(n_nodes)).tolist(),
        step_noise_history=np.asarray(pink_pwrs).tolist(),
        step_energy_history=np.asarray(e_trace).tolist(),
    )

    return valid_s, valid_x, telemetry


# Backward-compatibility alias
run_jax_silicon_langevin_sampler = run_jax_silicon_langevin
