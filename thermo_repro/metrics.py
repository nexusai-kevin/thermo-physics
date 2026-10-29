"""Thermodynamic Reproducibility Metrics:
1. Mixing Time & Integrated Autocorrelation Time (tau_int)
2. Energy Trajectory Tracking & Exact Boltzmann-Gibbs Verification
3. Normalized Hardware-Aware Energy Efficiency (Classical GPU vs. Thermodynamic Sampling Unit)
"""

from __future__ import annotations

from typing import Any, Sequence
import time
import numpy as np

from thermo_repro.backend import (
    Block,
    IsingEBM,
    IsingSamplingProgram,
    SamplingSchedule,
    SpinNode,
    compute_energy,
    exact_boltzmann_distribution,
    hinton_init,
    make_rng_key,
    sample_states,
    split_rng_key,
)
from thermo_repro.coloring import (
    greedy_graph_coloring_blocks,
    make_single_site_blocks,
)


# ---------------------------------------------------------------------------
# 1. Autocorrelation & Mixing Time (tau_int)
# ---------------------------------------------------------------------------


def compute_autocorrelation(series: Sequence[float] | np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """Compute normalized autocorrelation function rho(k) for lags k = 0, 1, ..., max_lag.

    Uses FFT-based autocovariance for O(T log T) speed.
    """
    x = np.asarray(series, dtype=np.float64).reshape(-1)
    n = len(x)
    if n < 4:
        return np.ones((1,), dtype=np.float64)

    if max_lag is None:
        max_lag = min(n // 2, 200)
    max_lag = max(1, min(max_lag, n - 1))

    mean = np.mean(x)
    centered = x - mean
    var = np.var(x)
    if var < 1e-15:
        res = np.zeros((max_lag + 1,), dtype=np.float64)
        res[0] = 1.0
        return res

    # Zero-padded FFT autocorrelation
    n_fft = 1 << (2 * n - 1).bit_length()
    fft_x = np.fft.rfft(centered, n=n_fft)
    acf = np.fft.irfft(fft_x * np.conj(fft_x), n=n_fft)[:n]
    acf /= np.arange(n, 0, -1)  # unbiased normalization by (N - k)
    acf /= acf[0]
    return acf[: max_lag + 1]


def compute_integrated_autocorrelation_time(
    series: Sequence[float] | np.ndarray,
    c_window: float = 5.0,
) -> dict[str, Any]:
    """Calculate the Integrated Autocorrelation Time (tau_int) using Sokal's automatic windowing:
        tau_int(W) = 0.5 + sum_{k=1}^{W} rho(k)
    where W is the first lag such that W >= c_window * tau_int(W) (or where rho(k) <= 0).

    Returns:
        - `tau_int`: integrated autocorrelation time in sweeps
        - `ess`: Effective Sample Size = N / (2 * tau_int)
        - `window`: cutoff lag W
        - `acf`: autocorrelation array rho(k)
    """
    x = np.asarray(series, dtype=np.float64).reshape(-1)
    n = len(x)
    acf = compute_autocorrelation(x)

    tau_int = 0.5
    window = 0
    for k in range(1, len(acf)):
        if acf[k] <= 0.0:
            window = k
            break
        tau_int += float(acf[k])
        window = k
        if k >= c_window * tau_int:
            break

    tau_int = max(0.5, tau_int)
    ess = float(n / (2.0 * tau_int))
    return {
        "tau_int": float(tau_int),
        "ess": ess,
        "window": int(window),
        "acf": acf,
    }


# ---------------------------------------------------------------------------
# 2. Energy Trajectory & Exact Boltzmann Verification
# ---------------------------------------------------------------------------


def verify_boltzmann_equilibrium(
    nodes: Sequence[SpinNode],
    edges: Sequence[tuple[SpinNode, SpinNode]],
    biases: Any,
    weights: Any,
    beta: float = 1.0,
    free_blocks: Sequence[Block] | None = None,
    n_warmup: int = 300,
    n_samples: int = 8000,
    steps_per_sample: int = 2,
    seed: int = 42,
) -> dict[str, Any]:
    """Run Block Gibbs on a subgraph (N <= 16), track the energy trajectory E(s_t),
    and compare the empirical state histogram against the exact brute-force Boltzmann
    distribution P(s) = exp(-beta E(s)) / Z.
    """
    n = len(nodes)
    if free_blocks is None:
        free_blocks = greedy_graph_coloring_blocks(nodes, edges)

    model = IsingEBM(nodes, edges, biases, weights, beta)
    program = IsingSamplingProgram(model, free_blocks=list(free_blocks), clamped_blocks=[])

    key = make_rng_key(seed)
    k_init, k_samp = split_rng_key(key, 2)
    init_state = hinton_init(k_init, model, list(free_blocks), ())

    schedule = SamplingSchedule(
        n_warmup=n_warmup,
        n_samples=n_samples,
        steps_per_sample=steps_per_sample,
    )
    samples = sample_states(k_samp, program, schedule, init_state, [], [Block(nodes)])
    samples_bool = np.asarray(samples[0], dtype=bool)
    energies = compute_energy(samples_bool, nodes, edges, biases, weights)

    # Compute exact Boltzmann distribution via brute-force enumeration of all 2^N states
    exact = exact_boltzmann_distribution(nodes, edges, biases, weights, beta)

    # Map each sampled boolean row to its integer state index in [0, 2^N - 1]
    bit_weights = (1 << np.arange(n - 1, -1, -1, dtype=np.int64))
    sample_indices = samples_bool.astype(np.int64) @ bit_weights
    counts = np.bincount(sample_indices, minlength=1 << n).astype(np.float64)
    empirical_probs = counts / float(n_samples)

    theo_probs = exact["probs"]
    tv_distance = float(0.5 * np.sum(np.abs(empirical_probs - theo_probs)))

    # KL divergence D_KL(P_emp || P_theo)
    mask = empirical_probs > 0
    kl_div = float(
        np.sum(empirical_probs[mask] * np.log(empirical_probs[mask] / np.clip(theo_probs[mask], 1e-15, 1.0)))
    )

    tau_stats = compute_integrated_autocorrelation_time(energies)

    return {
        "n_nodes": n,
        "n_edges": len(edges),
        "n_blocks": len(free_blocks),
        "beta": float(beta),
        "n_samples": n_samples,
        "exact_Z": exact["Z"],
        "exact_log_Z": exact["log_Z"],
        "exact_mean_energy": exact["expected_energy"],
        "empirical_mean_energy": float(np.mean(energies)),
        "energy_abs_error": float(abs(np.mean(energies) - exact["expected_energy"])),
        "total_variation_distance": tv_distance,
        "kl_divergence": kl_div,
        "tau_int_energy": tau_stats["tau_int"],
        "effective_sample_size": tau_stats["ess"],
        "energy_trajectory_head": energies[:50].tolist(),
        "empirical_probs": empirical_probs,
        "theoretical_probs": theo_probs,
    }


def compare_block_vs_sequential_gibbs(
    nodes: Sequence[SpinNode],
    edges: Sequence[tuple[SpinNode, SpinNode]],
    biases: Any,
    weights: Any,
    beta: float = 1.0,
    parallel_blocks: Sequence[Block] | None = None,
    n_warmup: int = 50,
    n_samples: int = 2000,
    seed: int = 7,
) -> dict[str, Any]:
    """Compare Parallel Multi-Color Block Gibbs against Sequential Single-Site Gibbs.

    Evaluates:
    - Number of sequential clock steps required per full graph sweep (C colors vs. N sites)
    - Integrated Autocorrelation Time (tau_int) in sweeps and in hardware clock cycles
    - Wall-clock execution time for software simulation
    """
    if parallel_blocks is None:
        parallel_blocks = greedy_graph_coloring_blocks(nodes, edges)
    single_site_blocks = make_single_site_blocks(nodes)

    model = IsingEBM(nodes, edges, biases, weights, beta)
    schedule = SamplingSchedule(n_warmup=n_warmup, n_samples=n_samples, steps_per_sample=1)

    key = make_rng_key(seed)
    k_init1, k_samp1, k_init2, k_samp2 = split_rng_key(key, 4)

    # 1. Parallel Block Gibbs
    prog_block = IsingSamplingProgram(model, free_blocks=list(parallel_blocks), clamped_blocks=[])
    init_block = hinton_init(k_init1, model, list(parallel_blocks), ())
    t0 = time.perf_counter()
    samp_block = sample_states(k_samp1, prog_block, schedule, init_block, [], [Block(nodes)])[0]
    t_block = time.perf_counter() - t0

    spins_block = np.where(np.asarray(samp_block, dtype=bool), 1.0, -1.0)
    mag_block = np.mean(spins_block, axis=1)
    energy_block = compute_energy(spins_block, nodes, edges, biases, weights)
    tau_block_e = compute_integrated_autocorrelation_time(energy_block)
    tau_block_m = compute_integrated_autocorrelation_time(mag_block)

    # 2. Sequential Single-Site Gibbs
    prog_seq = IsingSamplingProgram(model, free_blocks=single_site_blocks, clamped_blocks=[])
    init_seq = hinton_init(k_init2, model, single_site_blocks, ())
    t1 = time.perf_counter()
    samp_seq = sample_states(k_samp2, prog_seq, schedule, init_seq, [], [Block(nodes)])[0]
    t_seq = time.perf_counter() - t1

    spins_seq = np.where(np.asarray(samp_seq, dtype=bool), 1.0, -1.0)
    mag_seq = np.mean(spins_seq, axis=1)
    energy_seq = compute_energy(spins_seq, nodes, edges, biases, weights)
    tau_seq_e = compute_integrated_autocorrelation_time(energy_seq)
    tau_seq_m = compute_integrated_autocorrelation_time(mag_seq)

    n_colors = len(parallel_blocks)
    n_nodes = len(nodes)

    # On physical TSU hardware, updating one independent Block takes 1 parallel relaxation cycle.
    # Thus 1 sweep = n_colors clock cycles for Block Gibbs vs. n_nodes cycles for Single-Site Gibbs.
    hw_cycles_to_decorrelate_block = tau_block_e["tau_int"] * n_colors
    hw_cycles_to_decorrelate_seq = tau_seq_e["tau_int"] * n_nodes

    return {
        "n_nodes": n_nodes,
        "n_edges": len(edges),
        "block_gibbs": {
            "n_blocks_colors": n_colors,
            "wall_time_sec": float(t_block),
            "tau_int_energy_sweeps": tau_block_e["tau_int"],
            "tau_int_magnetization_sweeps": tau_block_m["tau_int"],
            "ess_energy": tau_block_e["ess"],
            "hardware_clock_cycles_per_sweep": n_colors,
            "hardware_cycles_per_independent_sample": float(2.0 * hw_cycles_to_decorrelate_block),
            "mean_energy": float(np.mean(energy_block)),
        },
        "sequential_gibbs": {
            "n_blocks_colors": n_nodes,
            "wall_time_sec": float(t_seq),
            "tau_int_energy_sweeps": tau_seq_e["tau_int"],
            "tau_int_magnetization_sweeps": tau_seq_m["tau_int"],
            "ess_energy": tau_seq_e["ess"],
            "hardware_clock_cycles_per_sweep": n_nodes,
            "hardware_cycles_per_independent_sample": float(2.0 * hw_cycles_to_decorrelate_seq),
            "mean_energy": float(np.mean(energy_seq)),
        },
        "parallel_cycle_speedup_factor": float(n_nodes / max(n_colors, 1)),
        "decorrelation_cycle_speedup_factor": float(
            hw_cycles_to_decorrelate_seq / max(hw_cycles_to_decorrelate_block, 1e-9)
        ),
        "software_wall_time_speedup": float(t_seq / max(t_block, 1e-9)),
    }


# ---------------------------------------------------------------------------
# 3. Normalized Hardware-Aware Energy Efficiency Benchmark
# ---------------------------------------------------------------------------


def profile_hardware_energy_efficiency(
    n_nodes: int,
    n_edges: int,
    n_blocks: int,
    total_sweeps: int,
    measured_wall_time_sec: float,
    gpu_tdp_watts: float = 300.0,
    tsu_array_power_watts: float = 1.0,
    tsu_pbit_capacity: int = 1_000_000,
    tsu_block_relaxation_ns: float = 5.0,
) -> dict[str, Any]:
    """Compare measured classical GPU/CPU MCMC energy usage against projected
    Thermodynamic Sampling Unit (TSU) hardware operating in CMOS subthreshold thermal regimes.

    Parameters:
    - `gpu_tdp_watts`: Typical GPU power draw under tensor/MCMC load (200-400W, default 300W)
    - `tsu_array_power_watts`: Projected TSU power (~1W per multi-million p-bit array)
    - `tsu_block_relaxation_ns`: Physical analog RC/thermal relaxation time per block (~5 ns)
    """
    total_spin_updates = n_nodes * total_sweeps
    total_block_steps = n_blocks * total_sweeps

    # Classical hardware metrics (from measured wall-clock execution)
    classical_joules = gpu_tdp_watts * measured_wall_time_sec
    classical_joules_per_sweep = classical_joules / max(total_sweeps, 1)
    classical_joules_per_spin_flip = classical_joules / max(total_spin_updates, 1)

    # TSU physical relaxation projection:
    # Active power scales with the fraction of the p-bit array utilized (with a baseline floor)
    utilization_fraction = max(n_nodes / float(tsu_pbit_capacity), 1e-4)
    active_tsu_watts = tsu_array_power_watts * min(1.0, utilization_fraction * 10.0)
    tsu_total_time_sec = total_block_steps * (tsu_block_relaxation_ns * 1e-9)
    tsu_joules = active_tsu_watts * tsu_total_time_sec
    tsu_joules_per_sweep = tsu_joules / max(total_sweeps, 1)
    tsu_femtojoules_per_spin_flip = (tsu_joules / max(total_spin_updates, 1)) * 1e15

    return {
        "problem_scale": {
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "n_blocks": n_blocks,
            "total_sweeps": total_sweeps,
            "total_spin_updates": total_spin_updates,
        },
        "classical_hardware_measured": {
            "wall_time_sec": float(measured_wall_time_sec),
            "sweeps_per_sec": float(total_sweeps / max(measured_wall_time_sec, 1e-9)),
            "assumed_power_watts": float(gpu_tdp_watts),
            "total_energy_joules": float(classical_joules),
            "joules_per_sweep": float(classical_joules_per_sweep),
            "nanojoules_per_spin_update": float(classical_joules_per_spin_flip * 1e9),
        },
        "tsu_projected": {
            "relaxation_time_sec": float(tsu_total_time_sec),
            "sweeps_per_sec": float(total_sweeps / max(tsu_total_time_sec, 1e-15)),
            "active_power_watts": float(active_tsu_watts),
            "total_energy_joules": float(tsu_joules),
            "joules_per_sweep": float(tsu_joules_per_sweep),
            "femtojoules_per_spin_update": float(tsu_femtojoules_per_spin_flip),
        },
        "tsu_vs_classical_ratios": {
            "latency_speedup_x": float(measured_wall_time_sec / max(tsu_total_time_sec, 1e-15)),
            "energy_reduction_x": float(classical_joules / max(tsu_joules, 1e-20)),
        },
    }
