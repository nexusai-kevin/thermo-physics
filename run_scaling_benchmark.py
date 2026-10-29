"""
Automated Multi-Scale Hardware Benchmark for Extropic THRML
Measures JAX JIT compilation time vs. pure parallel 2-color Block Gibbs execution time
and spin-flip throughput (million spin-updates / sec) as 2D lattice size scales:
    L = 16 (256 spins) -> L = 32 (1,024 spins) -> L = 64 (4,096 spins) -> L = 128 (16,384 spins).
Generates thrml_scaling_benchmark.png.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init


def build_lattice_program(L: int, beta_val: float = 0.44):
    nodes_2d = [[SpinNode() for _ in range(L)] for _ in range(L)]
    flat_nodes = [nodes_2d[r][c] for r in range(L) for c in range(L)]

    even_nodes = [nodes_2d[r][c] for r in range(L) for c in range(L) if (r + c) % 2 == 0]
    odd_nodes = [nodes_2d[r][c] for r in range(L) for c in range(L) if (r + c) % 2 == 1]
    free_blocks = [Block(even_nodes), Block(odd_nodes)]

    edges = []
    for r in range(L):
        for c in range(L):
            u = nodes_2d[r][c]
            edges.append((u, nodes_2d[(r + 1) % L][c]))
            edges.append((u, nodes_2d[r][(c + 1) % L]))

    biases = jnp.zeros((L * L,), dtype=jnp.float32)
    weights = jnp.ones((len(edges),), dtype=jnp.float32)
    beta = jnp.array(beta_val, dtype=jnp.float32)

    ebm = IsingEBM(nodes=flat_nodes, edges=edges, biases=biases, weights=weights, beta=beta)
    program = IsingSamplingProgram(ebm, free_blocks=free_blocks, clamped_blocks=[])
    return program, ebm, free_blocks, flat_nodes


def main():
    lattice_sizes = [16, 32, 64, 128]
    n_warmup = 100
    n_samples = 400
    total_sweeps = n_warmup + n_samples  # 500 full sweeps

    compile_times = []
    exec_times = []
    sweeps_per_sec = []
    m_spin_updates_per_sec = []
    sequential_speedup_factors = []

    key = jax.random.key(2026)
    device_name = str(jax.devices()[0])
    print(f"[*] Running Multi-Scale Hardware Benchmark on JAX Device: {device_name}")
    print(f"[*] Schedule per Lattice Size: {total_sweeps} full 2-color Block Gibbs sweeps ({n_warmup} warmup + {n_samples} samples)\n")

    for L in lattice_sizes:
        n_spins = L * L
        key, k_init, k_compile, k_run = jax.random.split(key, 4)

        program, ebm, free_blocks, flat_nodes = build_lattice_program(L=L, beta_val=0.4407)
        init_state = hinton_init(k_init, ebm, free_blocks, ())
        schedule = SamplingSchedule(n_warmup=n_warmup, n_samples=n_samples, steps_per_sample=2)
        obs_blocks = [Block(flat_nodes)]

        # 1. Measure First Call (JIT Compilation + Execution)
        t0 = time.perf_counter()
        out_first = sample_states(k_compile, program, schedule, init_state, [], obs_blocks)
        jax.block_until_ready(out_first)
        t_first = time.perf_counter() - t0

        # 2. Measure Second Call (Pure Compiled XLA Sampling Execution)
        t1 = time.perf_counter()
        out_second = sample_states(k_run, program, schedule, init_state, [], obs_blocks)
        jax.block_until_ready(out_second)
        t_exec = time.perf_counter() - t1

        t_jit = max(t_first - t_exec, 0.001)
        sps = total_sweeps / t_exec
        msu = (total_sweeps * n_spins) / (t_exec * 1e6)
        hw_cycle_speedup = n_spins / 2.0  # 2 parallel cycles vs N_spins sequential updates

        compile_times.append(t_jit)
        exec_times.append(t_exec)
        sweeps_per_sec.append(sps)
        m_spin_updates_per_sec.append(msu)
        sequential_speedup_factors.append(hw_cycle_speedup)

        print(
            f"L = {L:3d} ({n_spins:5d} spins) | JIT Compile: {t_jit:5.3f}s | Pure Exec: {t_exec:5.3f}s | "
            f"Throughput: {sps:7.1f} sweeps/s ({msu:6.2f} M spin-updates/s) | Clock Speedup: {hw_cycle_speedup:6.0f}x"
        )

    # Visualization: 2-Panel Scaling Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    x_labels = [f"{L}×{L}\n({L*L:,} spins)" for L in lattice_sizes]
    x_pos = np.arange(len(lattice_sizes))
    width = 0.36

    # Panel 1: JIT Compilation Time vs. Pure Sampling Execution Time
    axes[0].bar(x_pos - width / 2, compile_times, width, color="#6366f1", edgecolor="black", label="XLA JIT Compilation Time (s)")
    axes[0].bar(x_pos + width / 2, exec_times, width, color="#06b6d4", edgecolor="black", label=f"Pure Sampling Time ({total_sweeps} sweeps)")
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(x_labels)
    axes[0].set_title("JAX Compilation vs. Sampling Execution Time")
    axes[0].set_ylabel("Wall Time (seconds)")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()

    # Panel 2: Spin-Flip Throughput (Million Spin-Updates / sec) & Parallel Clock Speedup
    ax2 = axes[1]
    line1 = ax2.plot(x_pos, m_spin_updates_per_sec, "o-", color="#10b981", lw=2.8, markersize=9, label="Throughput (M Spin-Updates / sec)")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels)
    ax2.set_title("Parallel Block Gibbs Throughput & Hardware Clock Speedup")
    ax2.set_ylabel("Million Spin-Updates / Sec (MSU/s)", color="#10b981")
    ax2.grid(True, alpha=0.3)

    ax2_right = ax2.twinx()
    line2 = ax2_right.plot(x_pos, sequential_speedup_factors, "s--", color="#f59e0b", lw=2.2, markersize=8, label="Parallel Clock Speedup (N/2 Cycles)")
    ax2_right.set_ylabel("Parallel Clock Cycle Speedup vs. Sequential MCMC (×)", color="#f59e0b")
    ax2_right.set_yscale("log")

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc="upper left")

    plt.tight_layout()
    plt.savefig("thrml_scaling_benchmark.png", dpi=200)
    print("\n[+] Saved scaling benchmark visualization: thrml_scaling_benchmark.png")


if __name__ == "__main__":
    main()
