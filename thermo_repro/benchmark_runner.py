"""Unified CLI and Reproducibility Suite Runner for Thermodynamic Computing."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any
import numpy as np

from thermo_repro.backend import (
    USING_NATIVE_THRML,
    Block,
    IsingEBM,
    IsingSamplingProgram,
    SamplingSchedule,
    SpinNode,
    compute_energy,
    hinton_init,
    make_rng_key,
    normal_random,
    sample_states,
    split_rng_key,
)
from thermo_repro.coloring import build_2d_lattice_topology
from thermo_repro.metrics import (
    compare_block_vs_sequential_gibbs,
    profile_hardware_energy_efficiency,
    verify_boltzmann_equilibrium,
)
from thermo_repro.project_a_classifier import run_project_a_benchmark
from thermo_repro.project_b_dtm import run_project_b_benchmark


def run_minimal_bipartite_repro(
    n_visible: int = 64,
    n_hidden: int = 32,
    n_warmup: int = 100,
    n_samples: int = 500,
    steps_per_sample: int = 2,
) -> dict[str, Any]:
    """Run the Section 3 minimal bipartite RBM reproduction pattern and profile energy efficiency."""
    visible_nodes = [SpinNode(f"v_{i}") for i in range(n_visible)]
    hidden_nodes = [SpinNode(f"h_{j}") for j in range(n_hidden)]
    all_nodes = visible_nodes + hidden_nodes

    edges = [(v, h) for v in visible_nodes for h in hidden_nodes]
    biases = np.zeros((len(all_nodes),), dtype=np.float32)
    weights = np.asarray(normal_random(make_rng_key(42), (len(edges),))) * 0.1
    beta = 1.0

    model = IsingEBM(all_nodes, edges, biases, weights, beta)
    free_blocks = [Block(visible_nodes), Block(hidden_nodes)]
    program = IsingSamplingProgram(model, free_blocks=free_blocks, clamped_blocks=[])

    key = make_rng_key(0)
    k_init, k_samp = split_rng_key(key, 2)
    init_state = hinton_init(k_init, model, free_blocks, ())
    schedule = SamplingSchedule(
        n_warmup=n_warmup,
        n_samples=n_samples,
        steps_per_sample=steps_per_sample,
    )

    t0 = time.perf_counter()
    samples = sample_states(k_samp, program, schedule, init_state, [], [Block(all_nodes)])
    elapsed = time.perf_counter() - t0

    samples_arr = np.asarray(samples[0], dtype=bool)
    energies = compute_energy(samples_arr, all_nodes, edges, biases, weights)
    total_sweeps = n_warmup + n_samples * steps_per_sample

    hw_profile = profile_hardware_energy_efficiency(
        n_nodes=len(all_nodes),
        n_edges=len(edges),
        n_blocks=len(free_blocks),
        total_sweeps=total_sweeps,
        measured_wall_time_sec=elapsed,
    )

    return {
        "samples_shape": list(samples_arr.shape),
        "mean_energy": float(np.mean(energies)),
        "min_energy": float(np.min(energies)),
        "max_energy": float(np.max(energies)),
        "wall_time_sec": float(elapsed),
        "hardware_efficiency_profile": hw_profile,
    }


def run_full_reproducibility_suite(output_json: str | None = None) -> dict[str, Any]:
    """Execute all verification rubrics, Project A, and Project B, returning a complete report."""
    print("=" * 78)
    print(" THERMODYNAMIC COMPUTING REFERENCE & REPRODUCIBILITY SUITE")
    print(f" Backend: {'Native Extropic thrml' if USING_NATIVE_THRML else 'Built-in JAX/NumPy thrml-Compatible Engine'}")
    print("=" * 78)

    # 1. Section 3 Minimal Pattern + Hardware Energy Profile
    print("\n[1/5] Running Bipartite EBM Benchmark (64 Visible x 32 Hidden) & TSU Energy Profile...")
    bipartite_bench = run_minimal_bipartite_repro()
    hw = bipartite_bench["hardware_efficiency_profile"]
    print(
        f"  -> Generated samples shape: {tuple(bipartite_bench['samples_shape'])} "
        f"| <E> = {bipartite_bench['mean_energy']:.3f} "
        f"| Wall time = {bipartite_bench['wall_time_sec']*1000:.1f} ms"
    )
    print(
        f"  -> Classical (300W GPU): {hw['classical_hardware_measured']['joules_per_sweep']*1e3:.3f} mJ/sweep "
        f"vs. Projected TSU (1W array): {hw['tsu_projected']['femtojoules_per_spin_update']:.2f} fJ/spin-update "
        f"({hw['tsu_vs_classical_ratios']['energy_reduction_x']:.2e}x energy reduction)"
    )

    # 2. Exact Boltzmann Equilibrium Verification on an 8-Spin 2x4 Lattice (256 states)
    print("\n[2/5] Verifying Exact Boltzmann-Gibbs Equilibrium (Brute-Force Z on 2x4 Spin Lattice)...")
    nodes_small, edges_small, blocks_small = build_2d_lattice_topology(2, 4, periodic=False)
    rng = np.random.default_rng(12)
    biases_small = rng.normal(0.0, 0.25, size=(len(nodes_small),)).astype(np.float32)
    weights_small = rng.normal(0.35, 0.15, size=(len(edges_small),)).astype(np.float32)

    boltzmann_check = verify_boltzmann_equilibrium(
        nodes_small,
        edges_small,
        biases_small,
        weights_small,
        beta=0.85,
        free_blocks=blocks_small,
        n_warmup=300,
        n_samples=12000,
        steps_per_sample=2,
    )
    print(
        f"  -> Exact <E> = {boltzmann_check['exact_mean_energy']:.4f} "
        f"| Empirical <E> = {boltzmann_check['empirical_mean_energy']:.4f} "
        f"(|err| = {boltzmann_check['energy_abs_error']:.4f})"
    )
    print(
        f"  -> Total Variation Distance TV(P_emp, P_exact) = {boltzmann_check['total_variation_distance']:.4f} "
        f"| KL Divergence = {boltzmann_check['kl_divergence']:.5f}"
    )

    # 3. Mixing Time & Integrated Autocorrelation Time (tau_int): Block vs. Sequential Gibbs
    print("\n[3/5] Comparing Mixing Time & Autocorrelation (tau_int): Parallel Block vs. Sequential Gibbs...")
    nodes_grid, edges_grid, blocks_grid = build_2d_lattice_topology(6, 6, periodic=False)
    biases_grid = np.zeros((len(nodes_grid),), dtype=np.float32)
    weights_grid = np.full((len(edges_grid),), 0.35, dtype=np.float32)
    mixing_comp = compare_block_vs_sequential_gibbs(
        nodes_grid,
        edges_grid,
        biases_grid,
        weights_grid,
        beta=0.9,
        parallel_blocks=blocks_grid,
        n_warmup=80,
        n_samples=2000,
    )
    print(
        f"  -> 2-Color Block Gibbs: tau_int = {mixing_comp['block_gibbs']['tau_int_energy_sweeps']:.2f} sweeps "
        f"({mixing_comp['block_gibbs']['hardware_clock_cycles_per_sweep']} clock cycles/sweep)"
    )
    print(
        f"  -> Sequential Gibbs:    tau_int = {mixing_comp['sequential_gibbs']['tau_int_energy_sweeps']:.2f} sweeps "
        f"({mixing_comp['sequential_gibbs']['hardware_clock_cycles_per_sweep']} clock cycles/sweep)"
    )
    print(
        f"  -> Hardware Parallel Cycle Speedup: {mixing_comp['parallel_cycle_speedup_factor']:.1f}x per sweep "
        f"| {mixing_comp['decorrelation_cycle_speedup_factor']:.1f}x per independent sample"
    )

    # 4. Project A: Associative Memory & Robust Classification
    print("\n[4/5] Running Project A: Associative Memory & Robust EBM Classification (Bipartite V-H-L)...")
    proj_a = run_project_a_benchmark()
    print(
        f"  -> Clean Classification Accuracy: {proj_a['classification_metrics']['clean_accuracy']*100:.1f}% "
        f"| Noisy (20% spin flips): {proj_a['classification_metrics']['noisy_accuracy']*100:.1f}%"
    )
    print(
        f"  -> Associative Memory Occluded Bit Recovery: "
        f"{proj_a['associative_memory_metrics']['occluded_bit_recovery_accuracy']*100:.1f}% "
        f"| Occluded Class Accuracy: {proj_a['associative_memory_metrics']['occluded_classification_accuracy']*100:.1f}%"
    )

    # 5. Project B: Denoising Thermodynamic Model (DTM)
    print("\n[5/5] Running Project B: Denoising Thermodynamic Model (DTM) on 12x12 Spin Lattice...")
    proj_b = run_project_b_benchmark(height=12, width=12, flip_prob=0.25)
    b_dtm = proj_b["dtm_checkerboard_block_gibbs"]
    print(
        f"  -> Corrupted Input Accuracy: {proj_b['initial_noisy_accuracy']*100:.1f}% "
        f"-> DTM Denoised Accuracy: {b_dtm['denoised_accuracy']*100:.1f}% "
        f"({b_dtm['bit_error_reduction_pct']:.1f}% bit-error reduction)"
    )
    print(
        f"  -> Energy Relaxation: {b_dtm['initial_energy']:.2f} -> {b_dtm['final_energy']:.2f} "
        f"| Hardware Clock Cycle Advantage vs. Sequential MCMC: {proj_b['hardware_cycle_advantage_x']:.1f}x"
    )

    # Strip large arrays from JSON summary
    boltzmann_summary = {
        k: v
        for k, v in boltzmann_check.items()
        if k not in ("empirical_probs", "theoretical_probs")
    }
    proj_b_summary = {k: v for k, v in proj_b.items() if k != "ascii_preview"}

    report = {
        "using_native_thrml": USING_NATIVE_THRML,
        "bipartite_benchmark": bipartite_bench,
        "boltzmann_verification": boltzmann_summary,
        "mixing_time_comparison": mixing_comp,
        "project_a_associative_classifier": proj_a,
        "project_b_dtm_denoising": proj_b_summary,
    }

    if output_json:
        out_path = Path(output_json)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved full JSON benchmark report to: {out_path.resolve()}")

    print("=" * 78)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Thermodynamic Computing Reference Implementations & Reproducibility Suite"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="benchmark_results.json",
        help="Path to write JSON benchmark report (default: benchmark_results.json)",
    )
    parser.add_argument(
        "--show-dtm-ascii",
        action="store_true",
        help="Print ASCII visualization of Clean, Noisy, and DTM-Denoised 2D spin lattices",
    )
    args = parser.parse_args()
    run_full_reproducibility_suite(output_json=args.output_json)
    if args.show_dtm_ascii:
        proj_b = run_project_b_benchmark(height=12, width=12, flip_prob=0.25)
        print("\n--- Clean 12x12 Lattice ---")
        print(proj_b["ascii_preview"]["clean"])
        print("\n--- Corrupted (25% Spin Flips) ---")
        print(proj_b["ascii_preview"]["noisy"])
        print("\n--- DTM Relaxed Reconstruction ---")
        print(proj_b["ascii_preview"]["denoised"])


if __name__ == "__main__":
    main()
