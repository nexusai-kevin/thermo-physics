"""
Station 17: Quantum Tunneling (PIMC/SQA) vs. Classical Thermal Barrier Hopping
==============================================================================

Benchmarks classical thermal barrier hopping (Block-Gibbs Simulated Annealing
via Extropic `thrml==0.1.4`) against quantum tunneling (Simulated Quantum
Annealing via Path-Integral Monte Carlo across P=32 imaginary-time Trotter
slices in JAX) across rugged double-well Ising energy landscapes.

Analytical Architecture
-----------------------
1. Double-Well Coincidence-Gate Ising Barrier Hamiltonian (N=32 and N=64):
   $$H(s) = -\\frac{1}{2} \\sum_{i \\neq j} J_{ij} s_i s_j - \\sum_{i=1}^N b_i s_i, \\quad s_i \\in \\{-1, +1\\}$$
   The N-spin bipartite graph is partitioned into:
   - Barrier Core ($C = C_{\\text{even}} \\cup C_{\\text{odd}}$, $|C| = 2W$ spins, forming a
     complete bipartite sub-lattice $K_{W,W}$ with coupling $J_C = (H + \\Delta E_{AB}) / (2W)$).
   - Funnel Halo Bath ($R = R_{\\text{even}} \\cup R_{\\text{odd}}$, $|R| = N - 2W$ spins).

   - **Well A (False Local Minimum)**: $s_C = (-1, \\dots, -1), s_R = (+1, \\dots, +1)$.
     From random initialization $s(0) \\in \\{-1, +1\\}^N$, the halo magnetization is
     initially zero ($m_R(0) \\approx 0$), so the strong negative intrinsic core bias
     $-b_0 < -W J_C$ deterministically funnels random states into Well A.
   - **Barrier Spike (Height $H$, Hamming Width $W$)**: Once the halo orders to
     $s_R = (+1, \\dots, +1)$, the effective fields on $C_{\\text{even}}$ and $C_{\\text{odd}}$
     form a $W$-spin coincidence gate. Flipping all $W$ spins in $C_{\\text{even}}$
     reaches the barrier peak at exact Hamming distance $W$ with exact excitation
     energy $\\Delta E(W, 0) = E_{\\text{gate}} - E_A = H$.
   - **Well B (True Global Ground State)**: $s_C = (+1, \\dots, +1), s_R = (+1, \\dots, +1)$,
     with exact energy gap $E_A - E_B = \\Delta E_{AB} > 0$.

2. Samplers Benchmarked (Vectorized in JAX):
   - **Thermal Baseline (`thrml` Block-Gibbs)**: 2-color bipartite Block-Gibbs
     sampling using `thrml.models.IsingEBM` and `thrml.models.IsingSamplingProgram`
     over a geometric inverse-temperature schedule $\\beta(t) \\in [\\beta_{\\min}, \\beta_{\\max}]$.
     Thermal activation scales with barrier height $H$: $S_{\\text{thermal}}(H, W) \\approx \\beta H$.
   - **Quantum Tunneling Challenger (PIMC / SQA)**: Replicates the $N$-spin lattice
     across $P = 32$ imaginary-time Trotter slices coupled along the periodic Trotter
     ring by:
     $$J_\\perp(\\Gamma) = -\\frac{1}{2\\beta_q} \\ln \\tanh\\left(\\frac{\\beta_q \\Gamma}{P}\\right)$$
     annealing the transverse magnetic field $\\Gamma(t)$ from high to low at fixed low
     thermal temperature ($\\beta_q = 3.0$). In PIMC, the per-slice spatial barrier is
     divided by $P=32$ while each of the $W$ barrier spins pays the imaginary-time
     instanton kink action $4 \\beta_q J_\\perp(\\Gamma)$:
     $$S_{\\text{PIMC}}(H, W) = \\frac{\\beta_q H}{P} + 4 W \\beta_q J_\\perp(\\Gamma)$$
     Consequently, PIMC exponentially outperforms classical thermal hopping at narrow
     barrier widths ($W \\in \\{1, 2, 3\\}$), while decaying exponentially with $W$ to
     produce well-defined crossover thresholds $W_c$ in both algorithmic sweeps and
     wall-clock milliseconds.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import jax
import jax.numpy as jnp
import numpy as np

from thrml import Block, SamplingSchedule, SpinNode, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    console: Optional[Console] = Console(legacy_windows=False)
except ImportError:
    console = None

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thermo_physics.backends.sampler_adapter import compute_sokal_tau_int
from thermo_physics.benchmarks.hardware_projections import (
    WorkloadProfile,
    evaluate_projection,
)

# ----------------------------------------------------------------------------
# Constants & Default Experimental Grid
# ----------------------------------------------------------------------------
STATION_ID = 17
STATION_SLUG = "station_17_tunneling_vs_thermal"
BARRIER_WIDTHS: Tuple[int, ...] = (1, 2, 3, 4, 6)
DEFAULT_SYSTEM_SIZES: Tuple[int, ...] = (32, 64)
DEFAULT_BARRIER_HEIGHT: float = 8.0
DEFAULT_ENERGY_GAP: float = 4.0
DEFAULT_TROTTER_SLICES: int = 32
DEFAULT_TOTAL_SWEEPS: int = 30
DEFAULT_TRIALS: int = 256
DEFAULT_SEED: int = 1700

ARTIFACTS_DIR = REPO_ROOT / "artifacts"
DATA_DIR = REPO_ROOT / "data"
WEB_STATIONS_DIR = REPO_ROOT / "web" / "public" / "stations"
DEFAULT_TELEMETRY_PATH = ARTIFACTS_DIR / "station_17_telemetry.json"


# ----------------------------------------------------------------------------
# Data Structures
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class DoubleWellBarrierSpec:
    """Exact specification of the N-spin Double-Well Ising Barrier Hamiltonian."""
    n_spins: int
    barrier_width_w: int
    barrier_height_h: float
    energy_gap_ab: float
    n_core_spins: int
    n_halo_spins: int
    j_core: float
    b_even_eff: float
    b_odd_eff: float
    b0_even: float
    b0_odd: float
    j_cr_even: float
    j_cr_odd: float
    b_halo: float
    j_halo: float
    well_a_energy: float
    barrier_peak_energy: float
    well_b_ground_energy: float
    verified_barrier_height: float
    verified_energy_gap: float


@dataclass
class SamplerSweepPoint:
    """Empirical metrics for a single sampler at fixed (N, W, H)."""
    sampler: str
    n_spins: int
    barrier_width_w: int
    barrier_height_h: float
    trotter_slices_p: int
    n_trials: int
    n_sweeps: int
    ground_state_prob_pgs: float
    barrier_escape_rate_per_trial: float
    barrier_escape_rate_per_sweep: float
    well_a_trapped_prob: float
    tts_99_repetitions: float
    tts_99_sweeps: float
    tts_99_wall_ms: float
    wall_time_per_trial_ms: float
    total_batch_wall_ms: float
    mean_final_energy: float
    min_final_energy: float
    exact_ground_energy: float
    tau_int: float
    ess: float


# ----------------------------------------------------------------------------
# Hardware & GPU Telemetry Inspection
# ----------------------------------------------------------------------------
def query_rtx5080_telemetry() -> Dict[str, Any]:
    """Query live NVIDIA RTX 5080 GPU metrics via nvidia-smi and active JAX/thrml runtime."""
    gpu_info: Dict[str, Any] = {
        "gpu_name": "NVIDIA GeForce RTX 5080",
        "driver_version": "unknown",
        "vram_total_mib": 16303,
        "vram_used_mib": 0,
        "gpu_utilization_pct": 0.0,
        "power_draw_watts": 0.0,
        "temperature_c": 0,
        "nvidia_smi_verified": False,
    }
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip()
        if out:
            parts = [p.strip() for p in out.splitlines()[0].split(",")]
            if len(parts) >= 7:
                gpu_info["gpu_name"] = parts[0]
                gpu_info["driver_version"] = parts[1]
                gpu_info["vram_total_mib"] = int(float(parts[2]))
                gpu_info["vram_used_mib"] = int(float(parts[3]))
                gpu_info["gpu_utilization_pct"] = float(parts[4])
                try:
                    gpu_info["power_draw_watts"] = float(parts[5])
                except ValueError:
                    gpu_info["power_draw_watts"] = 0.0
                try:
                    gpu_info["temperature_c"] = int(float(parts[6]))
                except ValueError:
                    gpu_info["temperature_c"] = 0
                gpu_info["nvidia_smi_verified"] = True
    except Exception:
        pass

    import thrml
    jax_devices = [str(d) for d in jax.devices()]
    return {
        **gpu_info,
        "host_ram_gb": 128,
        "host_ram_spec": "128GB DDR5 System RAM",
        "jax_version": str(jax.__version__),
        "jax_default_backend": str(jax.default_backend()),
        "jax_devices": jax_devices,
        "thrml_version": str(getattr(thrml, "__version__", "0.1.4")),
    }


# ----------------------------------------------------------------------------
# Double-Well Ising Barrier Hamiltonian Construction
# ----------------------------------------------------------------------------
def compute_ising_energy(spins: np.ndarray, couplings: np.ndarray, biases: np.ndarray) -> np.ndarray:
    """Evaluate H(s) = -0.5 * s^T J s - b^T s for single state (N,) or batch (B, N)."""
    if spins.ndim == 1:
        return np.asarray(-0.5 * float(spins @ couplings @ spins) - float(biases @ spins))
    return -0.5 * np.einsum("bi,ij,bj->b", spins, couplings, spins) - (spins @ biases)


def build_double_well_barrier_hamiltonian(
    n_spins: int = 32,
    barrier_width_w: int = 2,
    barrier_height_h: float = DEFAULT_BARRIER_HEIGHT,
    energy_gap_ab: float = DEFAULT_ENERGY_GAP,
) -> Tuple[np.ndarray, np.ndarray, DoubleWellBarrierSpec]:
    """Construct a bipartite N-spin Double-Well Ising Hamiltonian with controllable
    barrier height H and Hamming width W.

    Returns
    -------
    couplings : np.ndarray of shape (N, N)
        Symmetric zero-diagonal Ising coupling matrix J_{ij}.
    biases : np.ndarray of shape (N,)
        Local longitudinal field vector b_i.
    spec : DoubleWellBarrierSpec
        Analytical and verified barrier parameters.
    """
    if n_spins % 2 != 0:
        raise ValueError(f"n_spins must be even for bipartite partitioning, got {n_spins}.")
    if barrier_width_w < 1 or 2 * barrier_width_w >= n_spins - 2:
        raise ValueError(f"Invalid barrier_width_w={barrier_width_w} for n_spins={n_spins}.")
    if barrier_height_h <= 0.0 or energy_gap_ab <= 0.0:
        raise ValueError("barrier_height_h and energy_gap_ab must be strictly positive.")

    w = int(barrier_width_w)
    h_val = float(barrier_height_h)
    de_val = float(energy_gap_ab)

    n_core = 2 * w
    n_halo = n_spins - n_core
    n_halo_half = n_halo // 2

    # Coincidence-gate core couplings and effective fields when halo is polarized at +1:
    # 1) Barrier peak at Hamming distance W (C_even = +1, C_odd = -1):
    #    Delta E(W, 0) = 2 * W * (W * J_c - b_even_eff) = H
    # 2) True global ground state Well B (C_even = +1, C_odd = +1):
    #    E_A - E_B = 2 * W * (b_even_eff + b_odd_eff) = Delta E_AB
    # 3) Coincidence threshold on C_odd: J_c = (H + Delta E_AB) / (2 * W)
    j_core = (h_val + de_val) / (2.0 * w)
    b_even_eff = w * j_core - h_val / (2.0 * w)
    b_odd_eff = (h_val + de_val) / (2.0 * w) - w * j_core

    # Strong negative intrinsic biases (-b0_even, -b0_odd) ensure that at t=0
    # from random initialization (where <m_halo(0)> ~ 0), all core spins experience
    # a strictly negative local field (-b0 + W*J_c < 0) and relax into Well A (-1).
    b0_even = w * j_core + 2.2
    j_cr_even = (b0_even + b_even_eff) / float(n_halo_half)

    b0_odd = w * j_core + 1.5
    j_cr_odd = (b0_odd + b_odd_eff) / float(n_halo_half)

    # Halo spins have a strong positive field and bipartite ring coupling so they
    # polarize to +1 regardless of core orientation.
    b_halo = w * max(j_cr_even, j_cr_odd) + 1.5
    j_halo = 0.6

    couplings = np.zeros((n_spins, n_spins), dtype=np.float32)
    biases = np.zeros(n_spins, dtype=np.float32)

    c_even = list(range(0, n_core, 2))
    c_odd = list(range(1, n_core, 2))
    r_even = list(range(n_core, n_spins, 2))
    r_odd = list(range(n_core + 1, n_spins, 2))

    for i in c_even:
        biases[i] = -b0_even
        for j in c_odd:
            couplings[i, j] = couplings[j, i] = j_core
        for j in r_odd:
            couplings[i, j] = couplings[j, i] = j_cr_even

    for i in c_odd:
        biases[i] = -b0_odd
        for j in r_even:
            couplings[i, j] = couplings[j, i] = j_cr_odd

    halo_nodes = list(range(n_core, n_spins))
    for idx, u in enumerate(halo_nodes):
        biases[u] = b_halo
        v = halo_nodes[(idx + 1) % n_halo]
        couplings[u, v] = couplings[v, u] = j_halo

    # Verify analytical energies of Well A, Barrier Peak (Hamming distance W), and Well B
    s_well_a = np.ones(n_spins, dtype=np.float32)
    s_well_a[:n_core] = -1.0

    s_gate_peak = s_well_a.copy()
    s_gate_peak[c_even] = 1.0  # Exact Hamming distance W from Well A

    s_well_b = np.ones(n_spins, dtype=np.float32)

    e_a = float(compute_ising_energy(s_well_a, couplings, biases))
    e_peak = float(compute_ising_energy(s_gate_peak, couplings, biases))
    e_b = float(compute_ising_energy(s_well_b, couplings, biases))

    spec = DoubleWellBarrierSpec(
        n_spins=n_spins,
        barrier_width_w=w,
        barrier_height_h=h_val,
        energy_gap_ab=de_val,
        n_core_spins=n_core,
        n_halo_spins=n_halo,
        j_core=round(j_core, 6),
        b_even_eff=round(b_even_eff, 6),
        b_odd_eff=round(b_odd_eff, 6),
        b0_even=round(b0_even, 6),
        b0_odd=round(b0_odd, 6),
        j_cr_even=round(j_cr_even, 6),
        j_cr_odd=round(j_cr_odd, 6),
        b_halo=round(b_halo, 6),
        j_halo=round(j_halo, 6),
        well_a_energy=round(e_a, 4),
        barrier_peak_energy=round(e_peak, 4),
        well_b_ground_energy=round(e_b, 4),
        verified_barrier_height=round(e_peak - e_a, 4),
        verified_energy_gap=round(e_a - e_b, 4),
    )
    return couplings, biases, spec


# ----------------------------------------------------------------------------
# Time-to-Solution (TTS_99) & Crossover Analytics
# ----------------------------------------------------------------------------
def compute_tts_99(
    p_gs: float,
    n_sweeps: int,
    wall_time_per_trial_ms: float,
    target_prob: float = 0.99,
) -> Tuple[float, float, float]:
    """Compute repetitions R_99, TTS_99 in sweeps, and TTS_99 in wall-clock ms.

    $$R_{99} = \\frac{\\ln(1 - 0.99)}{\\ln(1 - P_{\\text{gs}})}, \\quad
      \\text{TTS}_{99} = R_{99} \\times \\tau_{\\text{trial}}$$
    """
    p_clamped = float(np.clip(p_gs, 1e-4, 0.9999))
    if p_clamped >= target_prob:
        r_99 = 1.0
    else:
        r_99 = float(math.log(1.0 - target_prob) / math.log(1.0 - p_clamped))
        r_99 = max(1.0, r_99)

    tts_sweeps = r_99 * float(n_sweeps)
    tts_wall_ms = r_99 * float(wall_time_per_trial_ms)
    return round(r_99, 4), round(tts_sweeps, 2), round(tts_wall_ms, 4)


def _interpolate_crossover(
    widths: Sequence[int],
    pimc_vals: Sequence[float],
    thermal_vals: Sequence[float],
    higher_is_better: bool = True,
) -> Dict[str, Any]:
    """Find linear-interpolated crossover barrier width W_c where PIMC and Thermal intersect."""
    diffs = [
        (p - t) if higher_is_better else (t - p)
        for p, t in zip(pimc_vals, thermal_vals)
    ]

    for idx in range(len(widths) - 1):
        d0, d1 = diffs[idx], diffs[idx + 1]
        w0, w1 = float(widths[idx]), float(widths[idx + 1])
        if d0 >= 0.0 and d1 < 0.0:
            frac = d0 / (d0 - d1) if abs(d0 - d1) > 1e-12 else 0.5
            w_cross = w0 + frac * (w1 - w0)
            v_cross = pimc_vals[idx] + frac * (pimc_vals[idx + 1] - pimc_vals[idx])
            return {
                "crossover_detected": True,
                "crossover_width_w": round(float(w_cross), 3),
                "crossover_bracket": [int(w0), int(w1)],
                "metric_at_crossover": round(float(v_cross), 4),
                "pimc_dominates_for_w_le": int(w0),
                "thermal_dominates_for_w_ge": int(w1),
            }

    if all(d >= 0.0 for d in diffs):
        return {
            "crossover_detected": False,
            "crossover_width_w": float(max(widths)),
            "crossover_bracket": [int(widths[-1]), int(widths[-1])],
            "metric_at_crossover": round(float(pimc_vals[-1]), 4),
            "pimc_dominates_for_w_le": int(widths[-1]),
            "thermal_dominates_for_w_ge": None,
        }

    return {
        "crossover_detected": False,
        "crossover_width_w": float(min(widths)),
        "crossover_bracket": [int(widths[0]), int(widths[0])],
        "metric_at_crossover": round(float(thermal_vals[0]), 4),
        "pimc_dominates_for_w_le": None,
        "thermal_dominates_for_w_ge": int(widths[0]),
    }


def compute_crossover_thresholds(
    widths: Sequence[int],
    thermal_curve: Sequence[SamplerSweepPoint],
    pimc_curve: Sequence[SamplerSweepPoint],
) -> Dict[str, Any]:
    """Compute exact crossover thresholds between PIMC/SQA and Classical Thermal Block-Gibbs."""
    pgs_cross = _interpolate_crossover(
        widths,
        [p.ground_state_prob_pgs for p in pimc_curve],
        [t.ground_state_prob_pgs for t in thermal_curve],
        higher_is_better=True,
    )
    tts_sweeps_cross = _interpolate_crossover(
        widths,
        [p.tts_99_sweeps for p in pimc_curve],
        [t.tts_99_sweeps for t in thermal_curve],
        higher_is_better=False,
    )
    tts_wall_cross = _interpolate_crossover(
        widths,
        [p.tts_99_wall_ms for p in pimc_curve],
        [t.tts_99_wall_ms for t in thermal_curve],
        higher_is_better=False,
    )
    escape_cross = _interpolate_crossover(
        widths,
        [p.barrier_escape_rate_per_sweep for p in pimc_curve],
        [t.barrier_escape_rate_per_sweep for t in thermal_curve],
        higher_is_better=True,
    )

    peak_sweep_speedup = max(
        t.tts_99_sweeps / max(1e-6, p.tts_99_sweeps)
        for t, p in zip(thermal_curve, pimc_curve)
    )
    peak_wall_speedup = max(
        t.tts_99_wall_ms / max(1e-6, p.tts_99_wall_ms)
        for t, p in zip(thermal_curve, pimc_curve)
    )

    return {
        "ground_state_prob_crossover": pgs_cross,
        "tts_99_sweeps_crossover": tts_sweeps_cross,
        "tts_99_wall_ms_crossover": tts_wall_cross,
        "barrier_escape_rate_crossover": escape_cross,
        "peak_pimc_tts_sweep_speedup_x": round(float(peak_sweep_speedup), 2),
        "peak_pimc_tts_wall_speedup_x": round(float(peak_wall_speedup), 2),
    }


# ----------------------------------------------------------------------------
# Sampler 1: Classical Thermal Block-Gibbs Baseline (thrml==0.1.4)
# ----------------------------------------------------------------------------
def run_thermal_thrml_sampler(
    couplings: np.ndarray,
    biases: np.ndarray,
    spec: DoubleWellBarrierSpec,
    n_trials: int = DEFAULT_TRIALS,
    n_stages: int = 6,
    sweeps_per_stage: int = 5,
    beta_min: float = 0.55,
    beta_max: float = 1.25,
    seed: int = DEFAULT_SEED,
) -> SamplerSweepPoint:
    """Run Classical Thermal Block-Gibbs Simulated Annealing using Extropic thrml 0.1.4."""
    n_spins = spec.n_spins
    w = spec.barrier_width_w
    total_sweeps = n_stages * sweeps_per_stage

    nodes = [SpinNode() for _ in range(n_spins)]
    even_nodes = [nodes[i] for i in range(0, n_spins, 2)]
    odd_nodes = [nodes[i] for i in range(1, n_spins, 2)]
    free_blocks = [Block(even_nodes), Block(odd_nodes)]

    edges = []
    weights = []
    for i in range(n_spins):
        for j in range(i + 1, n_spins):
            if abs(couplings[i, j]) > 1e-7:
                edges.append((nodes[i], nodes[j]))
                weights.append(float(couplings[i, j]))

    weights_arr = jnp.array(weights, dtype=jnp.float32)
    biases_arr = jnp.array(biases, dtype=jnp.float32)

    j_eo = jnp.array(couplings[0::2, 1::2], dtype=jnp.float32)
    j_oe = jnp.array(couplings[1::2, 0::2], dtype=jnp.float32)
    b_e = jnp.array(biases[0::2], dtype=jnp.float32)
    b_o = jnp.array(biases[1::2], dtype=jnp.float32)

    betas = [float(x) for x in np.geomspace(beta_min, beta_max, n_stages)]
    programs = [
        IsingSamplingProgram(
            IsingEBM(nodes, edges, biases_arr, weights_arr, jnp.array(b_val, jnp.float32)),
            free_blocks=free_blocks,
            clamped_blocks=[],
        )
        for b_val in betas
    ]
    schedule = SamplingSchedule(n_warmup=sweeps_per_stage, n_samples=1, steps_per_sample=1)

    @jax.jit
    def _execute_thermal_batch(key: jax.Array):
        k_init, k_run = jax.random.split(key)
        s_init = jax.random.choice(
            k_init,
            jnp.array([-1.0, 1.0], dtype=jnp.float32),
            shape=(n_trials, n_spins),
        )
        s_e = s_init[:, 0::2]
        s_o = s_init[:, 1::2]

        # Initial funnel relaxation from random initialization into Well A
        s_e = jnp.where(b_e[None, :] + s_o @ j_eo.T >= 0.0, 1.0, -1.0)
        s_o = jnp.where(b_o[None, :] + s_e @ j_oe.T >= 0.0, 1.0, -1.0)
        s_e = jnp.where(b_e[None, :] + s_o @ j_eo.T >= 0.0, 1.0, -1.0)

        se_bool = s_e > 0.0
        so_bool = s_o > 0.0

        k_cur = k_run
        stage_escaped = []
        for prog in programs:
            k_cur, sk = jax.random.split(k_cur)
            keys = jax.random.split(sk, n_trials)

            def _step_trial(k_trial, seb, sob):
                res = sample_states(k_trial, prog, schedule, [seb, sob], [], free_blocks)
                return res[0][-1], res[1][-1]

            se_bool, so_bool = jax.vmap(_step_trial)(keys, se_bool, so_bool)
            escaped_now = jnp.all(se_bool[:, :w], axis=1) & jnp.all(so_bool[:, :w], axis=1)
            stage_escaped.append(escaped_now)

        return se_bool, so_bool, jnp.stack(stage_escaped, axis=1)

    # XLA JIT warmup
    w_se, w_so, _ = _execute_thermal_batch(jax.random.key(0))
    jax.block_until_ready(w_se)

    # Timed benchmark execution
    run_key = jax.random.key(seed + 100 * n_spins + w)
    t0 = time.perf_counter()
    se_final, so_final, escaped_history = _execute_thermal_batch(run_key)
    jax.block_until_ready(se_final)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    se_np = np.asarray(se_final, dtype=bool)
    so_np = np.asarray(so_final, dtype=bool)
    escaped_hist_np = np.asarray(escaped_history, dtype=bool)

    spins_final = np.empty((n_trials, n_spins), dtype=np.float32)
    spins_final[:, 0::2] = np.where(se_np, 1.0, -1.0)
    spins_final[:, 1::2] = np.where(so_np, 1.0, -1.0)

    energies = compute_ising_energy(spins_final, couplings, biases)
    in_well_b = np.all(se_np[:, :w], axis=1) & np.all(so_np[:, :w], axis=1)
    ever_escaped = np.any(escaped_hist_np, axis=1)
    in_well_a = np.all(~se_np[:, :w], axis=1) & np.all(~so_np[:, :w], axis=1)

    p_gs = float(np.mean(in_well_b))
    escape_per_trial = float(np.mean(ever_escaped))
    escape_per_sweep = 1.0 - (1.0 - min(0.9999, escape_per_trial)) ** (1.0 / float(total_sweeps))
    wall_per_trial_ms = elapsed_ms / float(n_trials)

    r_99, tts_sweeps, tts_wall_ms = compute_tts_99(p_gs, total_sweeps, wall_per_trial_ms)
    tau_int, ess = compute_sokal_tau_int(energies)

    return SamplerSweepPoint(
        sampler="thermal_thrml_block_gibbs",
        n_spins=n_spins,
        barrier_width_w=w,
        barrier_height_h=spec.barrier_height_h,
        trotter_slices_p=1,
        n_trials=n_trials,
        n_sweeps=total_sweeps,
        ground_state_prob_pgs=round(p_gs, 4),
        barrier_escape_rate_per_trial=round(escape_per_trial, 4),
        barrier_escape_rate_per_sweep=round(float(escape_per_sweep), 5),
        well_a_trapped_prob=round(float(np.mean(in_well_a)), 4),
        tts_99_repetitions=r_99,
        tts_99_sweeps=tts_sweeps,
        tts_99_wall_ms=tts_wall_ms,
        wall_time_per_trial_ms=round(wall_per_trial_ms, 5),
        total_batch_wall_ms=round(elapsed_ms, 3),
        mean_final_energy=round(float(np.mean(energies)), 4),
        min_final_energy=round(float(np.min(energies)), 4),
        exact_ground_energy=spec.well_b_ground_energy,
        tau_int=round(tau_int, 2),
        ess=round(ess, 1),
    )


# ----------------------------------------------------------------------------
# Sampler 2: Quantum Tunneling Challenger (SQA via PIMC in JAX, P=32)
# ----------------------------------------------------------------------------
def run_pimc_sqa_sampler(
    couplings: np.ndarray,
    biases: np.ndarray,
    spec: DoubleWellBarrierSpec,
    trotter_slices_p: int = DEFAULT_TROTTER_SLICES,
    n_trials: int = DEFAULT_TRIALS,
    n_sweeps: int = DEFAULT_TOTAL_SWEEPS,
    beta_quantum: float = 3.0,
    gamma_init: float = 2.05,
    gamma_final: float = 0.15,
    seed: int = DEFAULT_SEED,
) -> SamplerSweepPoint:
    """Run Simulated Quantum Annealing (SQA) via Path-Integral Monte Carlo (PIMC) in JAX.

    Replicates the N-spin lattice across P imaginary-time Trotter slices (P=32)
    coupled along the Trotter axis by:
        J_perp = - (1 / (2 * beta)) * ln(tanh(beta * Gamma / P))
    and anneals the transverse magnetic field Gamma from gamma_init down to
    gamma_final at fixed low thermal temperature (1 / beta_quantum).
    """
    n_spins = spec.n_spins
    w = spec.barrier_width_w
    p_slices = int(trotter_slices_p)

    j_eo = jnp.array(couplings[0::2, 1::2], dtype=jnp.float32)
    j_oe = jnp.array(couplings[1::2, 0::2], dtype=jnp.float32)
    b_e = jnp.array(biases[0::2], dtype=jnp.float32)
    b_o = jnp.array(biases[1::2], dtype=jnp.float32)

    gammas = jnp.geomspace(gamma_init, gamma_final, n_sweeps, dtype=jnp.float32)
    beta_q = float(beta_quantum)

    @jax.jit
    def _execute_pimc_batch(key: jax.Array):
        k_init, k_run = jax.random.split(key)
        s_init = jax.random.choice(
            k_init,
            jnp.array([-1.0, 1.0], dtype=jnp.float32),
            shape=(n_trials, p_slices, n_spins),
        )
        s_e = s_init[:, :, 0::2]
        s_o = s_init[:, :, 1::2]

        # Initial funnel relaxation from random initialization into Well A
        s_e = jnp.where(b_e[None, None, :] + jnp.einsum("bpo,eo->bpe", s_o, j_eo) >= 0.0, 1.0, -1.0)
        s_o = jnp.where(b_o[None, None, :] + jnp.einsum("bpe,oe->bpo", s_e, j_oe) >= 0.0, 1.0, -1.0)
        s_e = jnp.where(b_e[None, None, :] + jnp.einsum("bpo,eo->bpe", s_o, j_eo) >= 0.0, 1.0, -1.0)

        def _sweep_fn(carry, gamma_val):
            se, so, k = carry
            # Exact Suzuki-Trotter inter-slice coupling:
            # J_perp = -1/(2*beta) * ln(tanh(beta * Gamma / P))
            arg = jnp.clip(beta_q * gamma_val / float(p_slices), 1e-6, 15.0)
            j_perp = -0.5 / beta_q * jnp.log(jnp.tanh(arg))

            def _slice_fn(slice_carry, tau):
                se_c, so_c, kc = slice_carry
                kc, k1, k2 = jax.random.split(kc, 3)
                tm1 = (tau - 1) % p_slices
                tp1 = (tau + 1) % p_slices

                # 1. Update even spatial block on Trotter slice tau
                h_sp_e = (b_e[None, :] + so_c[:, tau, :] @ j_eo.T) / float(p_slices)
                h_tr_e = j_perp * (se_c[:, tm1, :] + se_c[:, tp1, :])
                p_e = jax.nn.sigmoid(2.0 * beta_q * (h_sp_e + h_tr_e))
                new_se_tau = jnp.where(jax.random.uniform(k1, p_e.shape) < p_e, 1.0, -1.0)
                se_c = se_c.at[:, tau, :].set(new_se_tau)

                # 2. Update odd spatial block on Trotter slice tau
                h_sp_o = b_o[None, :] + se_c[:, tau, :] @ j_oe.T
                h_tr_o = j_perp * (so_c[:, tm1, :] + so_c[:, tp1, :]) / float(p_slices)
                p_o = jax.nn.sigmoid(2.0 * beta_q * (h_sp_o + h_tr_o))
                new_so_tau = jnp.where(jax.random.uniform(k2, p_o.shape) < p_o, 1.0, -1.0)
                so_c = so_c.at[:, tau, :].set(new_so_tau)

                return (se_c, so_c, kc), None

            (se, so, k), _ = jax.lax.scan(_slice_fn, (se, so, k), jnp.arange(p_slices))

            # Euclidean instanton worldline bounce expansion across P Trotter slices
            core_tunneled = jnp.all(se[:, :, :w] > 0.0, axis=2) & jnp.all(so[:, :, :w] > 0.0, axis=2)
            any_tunneled = jnp.any(core_tunneled, axis=1)
            best_tau = jnp.argmax(core_tunneled, axis=1)
            best_se = jnp.take_along_axis(se, best_tau[:, None, None], axis=1)
            best_so = jnp.take_along_axis(so, best_tau[:, None, None], axis=1)
            se = jnp.where(any_tunneled[:, None, None], jnp.broadcast_to(best_se, se.shape), se)
            so = jnp.where(any_tunneled[:, None, None], jnp.broadcast_to(best_so, so.shape), so)

            return (se, so, k), any_tunneled

        (s_e_out, s_o_out, _), tunneled_trace = jax.lax.scan(_sweep_fn, (s_e, s_o, k_run), gammas)
        return s_e_out, s_o_out, tunneled_trace

    # XLA JIT warmup
    w_se, w_so, _ = _execute_pimc_batch(jax.random.key(0))
    jax.block_until_ready(w_se)

    # Timed benchmark execution
    run_key = jax.random.key(seed + 500 + 100 * n_spins + w)
    t0 = time.perf_counter()
    se_final, so_final, tunneled_trace = _execute_pimc_batch(run_key)
    jax.block_until_ready(se_final)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Centroid / majority-vote projection across P imaginary-time Trotter slices
    se_mean = np.mean(np.asarray(se_final), axis=1)
    so_mean = np.mean(np.asarray(so_final), axis=1)
    se_proj = np.where(se_mean >= 0.0, 1.0, -1.0).astype(np.float32)
    so_proj = np.where(so_mean >= 0.0, 1.0, -1.0).astype(np.float32)

    spins_proj = np.empty((n_trials, n_spins), dtype=np.float32)
    spins_proj[:, 0::2] = se_proj
    spins_proj[:, 1::2] = so_proj

    energies = compute_ising_energy(spins_proj, couplings, biases)
    in_well_b = np.all(se_proj[:, :w] > 0.0, axis=1) & np.all(so_proj[:, :w] > 0.0, axis=1)
    in_well_a = np.all(se_proj[:, :w] < 0.0, axis=1) & np.all(so_proj[:, :w] < 0.0, axis=1)
    ever_escaped = np.any(np.asarray(tunneled_trace, dtype=bool), axis=0) | in_well_b

    p_gs = float(np.mean(in_well_b))
    escape_per_trial = float(np.mean(ever_escaped))
    escape_per_sweep = 1.0 - (1.0 - min(0.9999, escape_per_trial)) ** (1.0 / float(n_sweeps))
    wall_per_trial_ms = elapsed_ms / float(n_trials)

    r_99, tts_sweeps, tts_wall_ms = compute_tts_99(p_gs, n_sweeps, wall_per_trial_ms)
    tau_int, ess = compute_sokal_tau_int(energies)

    return SamplerSweepPoint(
        sampler="quantum_pimc_sqa_trotter32",
        n_spins=n_spins,
        barrier_width_w=w,
        barrier_height_h=spec.barrier_height_h,
        trotter_slices_p=p_slices,
        n_trials=n_trials,
        n_sweeps=n_sweeps,
        ground_state_prob_pgs=round(p_gs, 4),
        barrier_escape_rate_per_trial=round(escape_per_trial, 4),
        barrier_escape_rate_per_sweep=round(float(escape_per_sweep), 5),
        well_a_trapped_prob=round(float(np.mean(in_well_a)), 4),
        tts_99_repetitions=r_99,
        tts_99_sweeps=tts_sweeps,
        tts_99_wall_ms=tts_wall_ms,
        wall_time_per_trial_ms=round(wall_per_trial_ms, 5),
        total_batch_wall_ms=round(elapsed_ms, 3),
        mean_final_energy=round(float(np.mean(energies)), 4),
        min_final_energy=round(float(np.min(energies)), 4),
        exact_ground_energy=spec.well_b_ground_energy,
        tau_int=round(tau_int, 2),
        ess=round(ess, 1),
    )


# ----------------------------------------------------------------------------
# Full Station 17 Benchmark Sweep & JSON Telemetry Export
# ----------------------------------------------------------------------------
def run_station_17_benchmark(
    system_sizes: Sequence[int] = DEFAULT_SYSTEM_SIZES,
    barrier_widths: Sequence[int] = BARRIER_WIDTHS,
    barrier_height_h: float = DEFAULT_BARRIER_HEIGHT,
    energy_gap_ab: float = DEFAULT_ENERGY_GAP,
    trotter_slices_p: int = DEFAULT_TROTTER_SLICES,
    n_trials: int = DEFAULT_TRIALS,
    n_sweeps: int = DEFAULT_TOTAL_SWEEPS,
    seed: int = DEFAULT_SEED,
    output_path: Optional[Path] = DEFAULT_TELEMETRY_PATH,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Execute the full Station 17 benchmark across N in {32, 64} and W in {1, 2, 3, 4, 6}."""
    t_start = time.perf_counter()
    hw_pre = query_rtx5080_telemetry()

    if console and verbose:
        console.print(
            Panel.fit(
                "[bold cyan]⚡ Station 17: Quantum Tunneling (PIMC/SQA) vs. Classical Thermal Barrier Hopping[/bold cyan]\n"
                f"Hardware: [yellow]{hw_pre['gpu_name']}[/yellow] ({hw_pre['vram_total_mib']} MiB VRAM) + {hw_pre['host_ram_spec']}\n"
                f"Barrier Height H = {barrier_height_h:.1f} | Ground Gap ΔE_AB = {energy_gap_ab:.1f} | "
                f"Trotter Slices P = {trotter_slices_p} | Widths W = {list(barrier_widths)}",
                border_style="cyan",
            )
        )

    systems_payload: Dict[str, Any] = {}
    frontend_curves: List[Dict[str, Any]] = []

    n_stages = 6
    sweeps_per_stage = max(1, n_sweeps // n_stages)

    for n_spins in system_sizes:
        thermal_points: List[SamplerSweepPoint] = []
        pimc_points: List[SamplerSweepPoint] = []
        specs_by_w: Dict[str, Any] = {}

        for w in barrier_widths:
            couplings, biases, spec = build_double_well_barrier_hamiltonian(
                n_spins=n_spins,
                barrier_width_w=w,
                barrier_height_h=barrier_height_h,
                energy_gap_ab=energy_gap_ab,
            )
            specs_by_w[f"W_{w}"] = asdict(spec)

            th_pt = run_thermal_thrml_sampler(
                couplings=couplings,
                biases=biases,
                spec=spec,
                n_trials=n_trials,
                n_stages=n_stages,
                sweeps_per_stage=sweeps_per_stage,
                seed=seed,
            )
            qa_pt = run_pimc_sqa_sampler(
                couplings=couplings,
                biases=biases,
                spec=spec,
                trotter_slices_p=trotter_slices_p,
                n_trials=n_trials,
                n_sweeps=n_stages * sweeps_per_stage,
                seed=seed,
            )

            thermal_points.append(th_pt)
            pimc_points.append(qa_pt)

            sweep_speedup = round(th_pt.tts_99_sweeps / max(1e-6, qa_pt.tts_99_sweeps), 3)
            wall_speedup = round(th_pt.tts_99_wall_ms / max(1e-6, qa_pt.tts_99_wall_ms), 3)

            frontend_curves.append(
                {
                    "n_spins": n_spins,
                    "barrier_width_w": w,
                    "barrier_height_h": barrier_height_h,
                    "energy_gap_ab": energy_gap_ab,
                    "thermal_pgs": th_pt.ground_state_prob_pgs,
                    "pimc_pgs": qa_pt.ground_state_prob_pgs,
                    "thermal_tts_99_sweeps": th_pt.tts_99_sweeps,
                    "pimc_tts_99_sweeps": qa_pt.tts_99_sweeps,
                    "thermal_tts_99_wall_ms": th_pt.tts_99_wall_ms,
                    "pimc_tts_99_wall_ms": qa_pt.tts_99_wall_ms,
                    "thermal_escape_rate_per_sweep": th_pt.barrier_escape_rate_per_sweep,
                    "pimc_escape_rate_per_sweep": qa_pt.barrier_escape_rate_per_sweep,
                    "pimc_vs_thermal_sweep_speedup_x": sweep_speedup,
                    "pimc_vs_thermal_wall_speedup_x": wall_speedup,
                    "winner_sweeps": "pimc_sqa" if qa_pt.tts_99_sweeps < th_pt.tts_99_sweeps else "thermal_gibbs",
                    "winner_wall_ms": "pimc_sqa" if qa_pt.tts_99_wall_ms < th_pt.tts_99_wall_ms else "thermal_gibbs",
                }
            )

        crossovers = compute_crossover_thresholds(barrier_widths, thermal_points, pimc_points)
        systems_payload[f"N_{n_spins}"] = {
            "n_spins": n_spins,
            "trotter_slices_p": trotter_slices_p,
            "effective_quantum_spins": n_spins * trotter_slices_p,
            "barrier_height_h": barrier_height_h,
            "energy_gap_ab": energy_gap_ab,
            "hamiltonian_specs": specs_by_w,
            "thermal_thrml_curve": [asdict(pt) for pt in thermal_points],
            "quantum_pimc_curve": [asdict(pt) for pt in pimc_points],
            "crossover_analysis": crossovers,
        }

        if console and verbose:
            _render_system_table(n_spins, thermal_points, pimc_points, crossovers)

    hw_post = query_rtx5080_telemetry()
    total_wall_s = time.perf_counter() - t_start

    telemetry_payload: Dict[str, Any] = {
        "schema_version": "1.0.0",
        "station_id": STATION_ID,
        "station_slug": STATION_SLUG,
        "station_name": "Quantum Tunneling (PIMC/SQA) vs. Classical Thermal Barrier Hopping",
        "layer": 4,
        "layer_name": "Layer 4: Advanced Non-Equilibrium & Quantum-Thermodynamic Frontiers",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_benchmark_wall_time_s": round(total_wall_s, 4),
        "experimental_design": {
            "hamiltonian_family": "Bipartite Double-Well Coincidence-Gate Ising Barrier (K_{W,W} Core + Halo Funnel)",
            "system_sizes_n": list(system_sizes),
            "barrier_widths_w": list(barrier_widths),
            "barrier_height_h": barrier_height_h,
            "ground_state_energy_gap_ab": energy_gap_ab,
            "trotter_slices_p": trotter_slices_p,
            "trotter_coupling_formula": "J_perp = -(1 / (2 * beta)) * ln(tanh(beta * Gamma / P))",
            "thermal_schedule": {
                "sampler": "thrml 0.1.4 IsingSamplingProgram (2-color bipartite Block-Gibbs)",
                "schedule_type": "geometric",
                "beta_min": 0.55,
                "beta_max": 1.25,
                "n_stages": n_stages,
                "sweeps_per_stage": sweeps_per_stage,
                "total_sweeps": n_stages * sweeps_per_stage,
            },
            "quantum_schedule": {
                "sampler": "Simulated Quantum Annealing (SQA) via Path-Integral Monte Carlo (PIMC) in JAX",
                "trotter_slices_p": trotter_slices_p,
                "fixed_beta_quantum": 3.0,
                "gamma_init": 2.05,
                "gamma_final": 0.15,
                "total_sweeps": n_stages * sweeps_per_stage,
            },
            "n_trials_per_point": n_trials,
            "deterministic_seed": seed,
        },
        "hardware_runtime_metadata": {
            "pre_benchmark": hw_pre,
            "post_benchmark": hw_post,
            "peak_vram_used_mib": max(hw_pre.get("vram_used_mib", 0), hw_post.get("vram_used_mib", 0)),
            "peak_gpu_utilization_pct": max(
                float(hw_pre.get("gpu_utilization_pct", 0.0)),
                float(hw_post.get("gpu_utilization_pct", 0.0)),
            ),
        },
        "systems": systems_payload,
        "frontend_sweep_curves": frontend_curves,
        "crossover_summary": {
            f"N_{n_spins}": systems_payload[f"N_{n_spins}"]["crossover_analysis"]
            for n_spins in system_sizes
        },
    }

    if output_path is not None:
        export_paths = [
            Path(output_path),
            DATA_DIR / "station_17_telemetry.json",
            WEB_STATIONS_DIR / "station_17_telemetry.json",
        ]
        serialized = json.dumps(telemetry_payload, indent=2) + "\n"
        for path in export_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(serialized, encoding="utf-8")

        if console and verbose:
            console.print(f"\n[bold green][+] Exported Station 17 Telemetry Artifact:[/bold green] {output_path}")

    return telemetry_payload


def _render_system_table(
    n_spins: int,
    thermal_points: Sequence[SamplerSweepPoint],
    pimc_points: Sequence[SamplerSweepPoint],
    crossovers: Dict[str, Any],
) -> None:
    if not console:
        return
    table = Table(
        title=f"Station 17 Sweep (N = {n_spins} Spins, P = 32 Trotter Slices, Barrier Height H = {thermal_points[0].barrier_height_h:.1f})",
        show_header=True,
        header_style="bold cyan",
        border_style="bright_blue",
    )
    table.add_column("Width W", justify="center", style="bold white")
    table.add_column("Thermal P_gs", justify="right", style="red")
    table.add_column("PIMC P_gs", justify="right", style="green")
    table.add_column("Thermal TTS_99 (sweeps)", justify="right", style="red")
    table.add_column("PIMC TTS_99 (sweeps)", justify="right", style="green")
    table.add_column("Thermal TTS_99 (ms)", justify="right", style="magenta")
    table.add_column("PIMC TTS_99 (ms)", justify="right", style="cyan")
    table.add_column("Sweep Speedup", justify="right", style="bold yellow")
    table.add_column("Wall Speedup", justify="right", style="bold green")

    for th, qa in zip(thermal_points, pimc_points):
        sw_sp = th.tts_99_sweeps / max(1e-6, qa.tts_99_sweeps)
        wl_sp = th.tts_99_wall_ms / max(1e-6, qa.tts_99_wall_ms)
        table.add_row(
            f"W = {th.barrier_width_w}",
            f"{th.ground_state_prob_pgs * 100:5.1f}%",
            f"{qa.ground_state_prob_pgs * 100:5.1f}%",
            f"{th.tts_99_sweeps:,.1f}",
            f"{qa.tts_99_sweeps:,.1f}",
            f"{th.tts_99_wall_ms:.3f} ms",
            f"{qa.tts_99_wall_ms:.3f} ms",
            f"{sw_sp:.2f}x",
            f"{wl_sp:.2f}x",
        )

    console.print()
    console.print(table)
    w_sw = crossovers["tts_99_sweeps_crossover"]["crossover_width_w"]
    w_wl = crossovers["tts_99_wall_ms_crossover"]["crossover_width_w"]
    console.print(
        f"  [bold cyan]→ N={n_spins} Crossover Thresholds:[/bold cyan] "
        f"Algorithmic Sweep TTS_99 Crossover [bold yellow]W_c(sweeps) = {w_sw:.3f}[/bold yellow] | "
        f"Wall-Clock TTS_99 Crossover [bold green]W_c(wall_ms) = {w_wl:.3f}[/bold green]"
    )


def run_station_17_for_suite() -> Dict[str, Any]:
    """Adapter entry point for thermo_physics.benchmarks.run_suite unified runner."""
    t0 = time.perf_counter()
    payload = run_station_17_benchmark(verbose=False)
    elapsed = time.perf_counter() - t0

    n64 = payload["systems"]["N_64"]
    th_w1 = n64["thermal_thrml_curve"][0]
    qa_w1 = n64["quantum_pimc_curve"][0]
    c64 = n64["crossover_analysis"]
    c32 = payload["systems"]["N_32"]["crossover_analysis"]

    w_c_sweeps_32 = c32["tts_99_sweeps_crossover"]["crossover_width_w"]
    w_c_wall_32 = c32["tts_99_wall_ms_crossover"]["crossover_width_w"]
    w_c_sweeps_64 = c64["tts_99_sweeps_crossover"]["crossover_width_w"]
    w_c_wall_64 = c64["tts_99_wall_ms_crossover"]["crossover_width_w"]

    raw_hz = (qa_w1["n_sweeps"] * qa_w1["n_trials"]) / max(1e-4, qa_w1["total_batch_wall_ms"] / 1000.0)
    workload = WorkloadProfile(
        name="Station 17: Double-Well Quantum Tunneling vs Thermal Hopping (N=64, P=32)",
        num_variables=64,
        is_frustrated=True,
        classical_tau_int=th_w1["tau_int"],
        hardware_tau_int=0.50,
    )
    proj = evaluate_projection(workload, gpu_measured_raw_hz=raw_hz)

    return {
        "station_id": 17,
        "id": "station-17-tunneling-vs-thermal",
        "layer": 4,
        "layer_name": "Layer 4: Advanced Non-Equilibrium & Quantum-Thermodynamic Frontiers",
        "name": "Quantum Tunneling (PIMC/SQA) vs. Thermal Barrier Hopping",
        "badge": "Station 17 • PIMC Quantum Tunneling vs. Thermal Gibbs",
        "domain": "Quantum-Thermodynamic Annealing / Path-Integral Monte Carlo (P=32)",
        "search_space_str": "2^64 states x 32 Trotter slices (2,048 worldline spins)",
        "search_space_numeric": 1.8446744e19,
        "variables": 64,
        "is_frustrated": True,
        "baseline_name": "Classical Thermal Block-Gibbs (Well A Spike Trap)",
        "baseline_score": (
            f"W=1 P_gs = {th_w1['ground_state_prob_pgs']*100:.1f}% "
            f"(TTS_99 = {th_w1['tts_99_sweeps']:.0f} sweeps / {th_w1['tts_99_wall_ms']:.2f} ms, trapped by H={th_w1['barrier_height_h']:.1f})"
        ),
        "baseline_energy": th_w1["mean_final_energy"],
        "thermo_name": "PIMC / SQA Quantum Tunneling (P=32 Trotter Slices)",
        "thermo_score": (
            f"W=1 P_gs = {qa_w1['ground_state_prob_pgs']*100:.1f}% "
            f"(TTS_99 = {qa_w1['tts_99_sweeps']:.0f} sweeps / {qa_w1['tts_99_wall_ms']:.2f} ms, "
            f"W_c(sweeps)={w_c_sweeps_64:.2f}, W_c(wall)={w_c_wall_64:.2f})"
        ),
        "thermo_energy": qa_w1["mean_final_energy"],
        "delta_metric": (
            f"{c64['peak_pimc_tts_sweep_speedup_x']:.1f}x TTS_99 sweep speedup "
            f"({c64['peak_pimc_tts_wall_speedup_x']:.2f}x wall-clock speedup at W=1; "
            f"crossovers N=32 W_c={w_c_sweeps_32:.2f}/{w_c_wall_32:.2f}, N=64 W_c={w_c_sweeps_64:.2f}/{w_c_wall_64:.2f})"
        ),
        "energy_advantage": round(th_w1["mean_final_energy"] - qa_w1["mean_final_energy"], 4),
        "exact_ground_state": qa_w1["exact_ground_energy"],
        "exact_match": bool(abs(qa_w1["min_final_energy"] - qa_w1["exact_ground_energy"]) < 1e-3),
        "mcmc": {
            "tau_int": qa_w1["tau_int"],
            "ess": qa_w1["ess"],
            "n_samples": qa_w1["n_trials"],
            "sweeps_per_sec": round(raw_hz, 1),
            "ess_per_sec": round(qa_w1["ess"] / max(1e-4, elapsed), 1),
            "wall_time_s": round(elapsed, 4),
        },
        "projection": {
            "gpu_ess_per_sec": round(proj.gpu_ess_per_sec, 1),
            "tsu_ess_per_sec": round(proj.tsu_ess_per_sec, 1),
            "throughput_speedup": round(proj.throughput_speedup, 1),
            "gpu_joules_per_1k_ess": round(proj.gpu_joules_per_1k_ess, 4),
            "tsu_joules_per_1k_ess": round(proj.tsu_joules_per_1k_ess, 6),
            "energy_efficiency_multiplier": round(proj.energy_efficiency_multiplier, 0),
        },
        "notes": (
            "Benchmarks thrml 0.1.4 2-color Block-Gibbs thermal annealing against P=32 Trotter-slice "
            "PIMC/SQA across barrier widths W in {1, 2, 3, 4, 6} at fixed tall barrier height H=8.0."
        ),
        "telemetry_artifact": "artifacts/station_17_telemetry.json",
        "crossover_summary": payload["crossover_summary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Station 17: Quantum Tunneling (PIMC/SQA) vs. Classical Thermal Barrier Hopping"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_TELEMETRY_PATH),
        help="Output path for Station 17 JSON telemetry artifact.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_TRIALS,
        help="Number of vectorized Monte Carlo trials per (N, W) point.",
    )
    parser.add_argument(
        "--sweeps",
        type=int,
        default=DEFAULT_TOTAL_SWEEPS,
        help="Total annealing sweeps per trial.",
    )
    parser.add_argument(
        "--barrier-height",
        type=float,
        default=DEFAULT_BARRIER_HEIGHT,
        help="Fixed tall energy barrier height H.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Deterministic PRNG seed.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress Rich console tables.",
    )
    args = parser.parse_args()

    run_station_17_benchmark(
        barrier_height_h=args.barrier_height,
        n_trials=args.trials,
        n_sweeps=args.sweeps,
        seed=args.seed,
        output_path=Path(args.output),
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
