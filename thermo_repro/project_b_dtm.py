"""Project B: Denoising Thermodynamic Model (DTM) Suite on 2D Spin Lattices.

Replaces neural reverse-diffusion score evaluations with physical thermal relaxation
over a 2D nearest-neighbor factor graph using 2-color checkerboard Block Gibbs sampling.

Energy functional for denoising a noisy spin observation y in {-1, +1}^{H x W}:
    E(s | y) = - J_prior * sum_{<i, j>} s_i s_j - sum_i h_i(y_i) s_i
where h_i(y_i) = 0.5 * log((1 - p_noise) / p_noise) * y_i is the exact Bayes log-odds field.
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
    compute_energy,
    make_rng_key,
    sample_states,
    split_rng_key,
)
from thermo_repro.coloring import (
    build_2d_lattice_topology,
    make_single_site_blocks,
)


def create_synthetic_2d_spin_image(height: int = 12, width: int = 12) -> np.ndarray:
    """Create a structured 2D binary spin image in {-1.0, +1.0}^{H x W} (centered block motif)."""
    img = -np.ones((height, width), dtype=np.float32)
    r0, r1 = height // 4, height - height // 4
    c0, c1 = width // 4, width - width // 4
    img[r0:r1, c0:c1] = 1.0
    return img


def corrupt_spin_image(
    clean_img: np.ndarray,
    flip_prob: float = 0.25,
    seed: int = 123,
) -> np.ndarray:
    """Corrupt a clean {-1, +1} spin image by flipping spins independently with probability `flip_prob`."""
    rng = np.random.default_rng(seed)
    flips = rng.random(clean_img.shape) < flip_prob
    return np.where(flips, -clean_img, clean_img).astype(np.float32)


def render_ascii_spin_grid(grid: np.ndarray) -> str:
    """Render a 2D spin array in {-1, +1} as a cross-platform ASCII diagram."""
    lines = []
    for row in grid:
        line = "".join("##" if val > 0 else ".." for val in row)
        lines.append(line)
    return "\n".join(lines)


def run_dtm_annealed_denoising(
    noisy_img: np.ndarray,
    clean_img: np.ndarray | None = None,
    flip_prob: float = 0.25,
    j_prior: float = 0.24,
    obs_coupling_scale: float = 1.00,
    beta_schedule: Sequence[float] = (0.7, 1.4, 2.2),
    sweeps_per_stage: int = 18,
    use_checkerboard_blocks: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    """Run Denoising Thermodynamic Model (DTM) relaxation across an inverse-temperature
    annealing schedule `beta_schedule`.

    At each stage k:
    - Formulates the `IsingEBM` at inverse temperature `beta_k`
    - Initializes the spin lattice from the previous stage's relaxed state (starting at `noisy_img`)
    - Samples via 2-color Checkerboard Block Gibbs (or single-site sequential MCMC if `use_checkerboard_blocks=False`)
    """
    height, width = noisy_img.shape
    nodes, edges, checkerboard_blocks = build_2d_lattice_topology(height, width, periodic=False)
    free_blocks = checkerboard_blocks if use_checkerboard_blocks else make_single_site_blocks(nodes)

    # Scaled Bayes log-odds external field from noisy observation y
    # Balances local observation pinning against 4-neighbor spatial Ising coupling J_prior
    p_clipped = float(np.clip(flip_prob, 0.01, 0.49))
    lambda_obs = obs_coupling_scale * 0.5 * float(np.log((1.0 - p_clipped) / p_clipped))

    y_flat = noisy_img.reshape(-1).astype(np.float32)
    biases = (lambda_obs * y_flat).astype(np.float32)
    weights = np.full((len(edges),), float(j_prior), dtype=np.float32)

    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    current_spins = y_flat.copy()

    energy_trajectory: list[float] = [
        float(compute_energy(current_spins, nodes, edges, biases, weights))
    ]
    accuracy_trajectory: list[float] = []
    if clean_img is not None:
        clean_flat = clean_img.reshape(-1)
        accuracy_trajectory.append(float(np.mean(current_spins == clean_flat)))

    key = make_rng_key(seed)
    stage_keys = split_rng_key(key, len(beta_schedule))
    stage_samples_list: list[np.ndarray] = []

    t0 = time.perf_counter()
    for stage_idx, beta_val in enumerate(beta_schedule):
        model = IsingEBM(nodes, edges, biases, weights, float(beta_val))
        program = IsingSamplingProgram(model, free_blocks=free_blocks, clamped_blocks=[])

        # Prepare current_spins as block boolean init states
        init_state = [
            (current_spins[[node_to_idx[n] for n in blk]] > 0)
            for blk in free_blocks
        ]
        schedule = SamplingSchedule(
            n_warmup=0,
            n_samples=sweeps_per_stage,
            steps_per_sample=1,
        )
        samples = sample_states(
            stage_keys[stage_idx],
            program,
            schedule,
            init_state,
            [],
            [Block(nodes)],
        )[0]
        samples_bool = np.asarray(samples, dtype=bool)
        samples_spins = np.where(samples_bool, 1.0, -1.0).astype(np.float32)
        stage_samples_list.append(samples_spins)

        stage_energies = compute_energy(samples_spins, nodes, edges, biases, weights)
        energy_trajectory.extend(stage_energies.tolist())

        if clean_img is not None:
            clean_flat = clean_img.reshape(-1)
            stage_accs = np.mean(samples_spins == clean_flat[None, :], axis=1)
            accuracy_trajectory.extend(stage_accs.tolist())

        current_spins = samples_spins[-1].copy()

    wall_time = time.perf_counter() - t0
    final_stage_spins = stage_samples_list[-1]
    # Posterior marginal sign estimate over the lowest-temperature stage
    denoised_flat = np.where(np.mean(final_stage_spins, axis=0) >= 0.0, 1.0, -1.0).astype(np.float32)
    denoised_img = denoised_flat.reshape(height, width)

    total_sweeps = len(beta_schedule) * sweeps_per_stage
    n_colors = len(free_blocks)
    total_hw_clock_cycles = total_sweeps * n_colors

    result: dict[str, Any] = {
        "height": height,
        "width": width,
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "n_blocks": n_colors,
        "total_sweeps": total_sweeps,
        "total_hardware_clock_cycles": total_hw_clock_cycles,
        "wall_time_sec": float(wall_time),
        "initial_energy": float(energy_trajectory[0]),
        "final_energy": float(compute_energy(denoised_flat, nodes, edges, biases, weights)),
        "energy_trajectory": energy_trajectory,
        "accuracy_trajectory": accuracy_trajectory,
        "denoised_image": denoised_img,
    }
    if clean_img is not None:
        noisy_acc = float(np.mean(noisy_img == clean_img))
        final_acc = float(np.mean(denoised_img == clean_img))
        noisy_ber = 1.0 - noisy_acc
        final_ber = 1.0 - final_acc
        result.update(
            {
                "initial_pixel_accuracy": noisy_acc,
                "denoised_pixel_accuracy": final_acc,
                "initial_bit_error_rate": noisy_ber,
                "final_bit_error_rate": final_ber,
                "error_reduction_percent": float(
                    100.0 * (noisy_ber - final_ber) / max(noisy_ber, 1e-9)
                ),
            }
        )
    return result


def run_project_b_benchmark(
    height: int = 12,
    width: int = 12,
    flip_prob: float = 0.25,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare 2-Color Checkerboard DTM Relaxation vs. Classical Sequential MCMC on a 2D Lattice."""
    clean_img = create_synthetic_2d_spin_image(height=height, width=width)
    noisy_img = corrupt_spin_image(clean_img, flip_prob=flip_prob, seed=seed)

    # 1. Parallel 2-Color Checkerboard Block Gibbs (DTM)
    dtm_res = run_dtm_annealed_denoising(
        noisy_img=noisy_img,
        clean_img=clean_img,
        flip_prob=flip_prob,
        use_checkerboard_blocks=True,
        seed=seed + 1,
    )

    # 2. Classical Single-Site Sequential Gibbs MCMC
    seq_res = run_dtm_annealed_denoising(
        noisy_img=noisy_img,
        clean_img=clean_img,
        flip_prob=flip_prob,
        use_checkerboard_blocks=False,
        seed=seed + 2,
    )

    return {
        "project": "Project B: Denoising Thermodynamic Model (DTM) 2D Lattice Suite",
        "lattice_shape": [height, width],
        "n_spins": height * width,
        "noise_flip_prob": float(flip_prob),
        "initial_noisy_accuracy": dtm_res["initial_pixel_accuracy"],
        "dtm_checkerboard_block_gibbs": {
            "n_parallel_blocks": dtm_res["n_blocks"],
            "total_sweeps": dtm_res["total_sweeps"],
            "hardware_clock_cycles": dtm_res["total_hardware_clock_cycles"],
            "wall_time_sec": dtm_res["wall_time_sec"],
            "initial_energy": dtm_res["initial_energy"],
            "final_energy": dtm_res["final_energy"],
            "denoised_accuracy": dtm_res["denoised_pixel_accuracy"],
            "bit_error_reduction_pct": dtm_res["error_reduction_percent"],
        },
        "classical_sequential_mcmc": {
            "n_parallel_blocks": seq_res["n_blocks"],
            "total_sweeps": seq_res["total_sweeps"],
            "hardware_clock_cycles": seq_res["total_hardware_clock_cycles"],
            "wall_time_sec": seq_res["wall_time_sec"],
            "initial_energy": seq_res["initial_energy"],
            "final_energy": seq_res["final_energy"],
            "denoised_accuracy": seq_res["denoised_pixel_accuracy"],
            "bit_error_reduction_pct": seq_res["error_reduction_percent"],
        },
        "hardware_cycle_advantage_x": float(
            seq_res["total_hardware_clock_cycles"]
            / max(dtm_res["total_hardware_clock_cycles"], 1)
        ),
        "ascii_preview": {
            "clean": render_ascii_spin_grid(clean_img),
            "noisy": render_ascii_spin_grid(noisy_img),
            "denoised": render_ascii_spin_grid(dtm_res["denoised_image"]),
        },
    }
