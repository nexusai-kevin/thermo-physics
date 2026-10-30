"""
Station 12: Grid SCUC QUBO Engine
=================================
Security-Constrained Unit Commitment (SCUC) mapped to an Ising Hamiltonian.
Demonstrates quadratic penalty formulation for power grid resource dispatch.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from thermo_physics.backends.sampler_adapter import (
    BackendType,
    HamiltonianGraph,
    SamplingConfig,
    get_sampler,
)


@dataclass
class Generator:
    id: str
    capacity_mw: float
    marginal_cost_per_mwh: float
    p_min_mw: float = 0.0


def build_scuc_qubo(
    generators: List[Generator],
    demand_mw: List[float],
    penalty_weight: float = 10.0,
) -> HamiltonianGraph:
    """Compile multi-period SCUC into a binary/bipolar HamiltonianGraph."""
    num_gen = len(generators)
    num_time = len(demand_mw)
    n_vars = num_gen * num_time

    # Variable index: x[g, t] -> g * num_time + t
    def idx(g: int, t: int) -> int:
        return g * num_time + t

    # Initialize QUBO matrix Q: min x^T Q x
    q_mat = np.zeros((n_vars, n_vars), dtype=np.float64)

    # 1. Linear generation costs: sum_{g, t} C_g * P_g * x_{g, t}
    for g in range(num_gen):
        for t in range(num_time):
            i = idx(g, t)
            # Cost per time slice (assume 1-hour intervals)
            q_mat[i, i] += generators[g].marginal_cost_per_mwh * generators[g].capacity_mw * 0.01

    # 2. Demand penalty: lambda * sum_t (sum_g P_g * x_{g, t} - D_t)^2
    # Expanding: lambda * [ sum_g (P_g^2 - 2 D_t P_g) x_{g, t} + 2 sum_{g < g'} P_g P_g' x_{g, t} x_{g', t} + D_t^2 ]
    const_offset = 0.0
    for t in range(num_time):
        d_t = demand_mw[t]
        const_offset += penalty_weight * (d_t ** 2)

        for g in range(num_gen):
            p_g = generators[g].capacity_mw
            i = idx(g, t)
            q_mat[i, i] += penalty_weight * (p_g ** 2 - 2.0 * d_t * p_g)

            for g_prime in range(g + 1, num_gen):
                p_gp = generators[g_prime].capacity_mw
                j = idx(g_prime, t)
                q_mat[i, j] += penalty_weight * (2.0 * p_g * p_gp)
                q_mat[j, i] += penalty_weight * (2.0 * p_g * p_gp)

    # Map binary QUBO (x in {0, 1}) to Ising spins (s in {-1, +1}):
    # x_i = (s_i + 1) / 2
    # x^T Q x = 1/4 s^T Q s + 1/2 (Q_diag + sum_j Q_ij) s + const
    couplings = -0.5 * q_mat.copy()  # Energy(s) = - 0.5 s^T J s - h^T s
    np.fill_diagonal(couplings, 0.0)

    biases = -0.5 * (np.diag(q_mat) + np.sum(q_mat, axis=1) - np.diag(q_mat))
    
    return HamiltonianGraph(
        couplings=couplings.astype(np.float32),
        biases=biases.astype(np.float32),
        spin_type="bipolar",
        offset=float(const_offset),
    )


def run_scuc_experiment():
    print("=" * 60)
    print("Station 12: Grid Security-Constrained Unit Commitment (SCUC)")
    print("=" * 60)

    # 4 Generators: Base Nuclear, Coal, Combined-Cycle Gas, Peaker Gas Turbine
    generators = [
        Generator("Nuclear_Base", capacity_mw=400.0, marginal_cost_per_mwh=12.0),
        Generator("Coal_Base", capacity_mw=250.0, marginal_cost_per_mwh=28.0),
        Generator("CCGT_Flex", capacity_mw=150.0, marginal_cost_per_mwh=45.0),
        Generator("Peaker_Gas", capacity_mw=100.0, marginal_cost_per_mwh=85.0),
    ]

    # 6-hour dynamic load curve (MW)
    demand_mw = [450.0, 520.0, 680.0, 750.0, 620.0, 480.0]

    graph = build_scuc_qubo(generators, demand_mw, penalty_weight=0.005)
    print(f"Compiled SCUC Grid Graph: {graph.num_nodes} spins (4 gens x 6 time slots)")

    config = SamplingConfig(n_samples=5000, burn_in=500, beta=2.0, seed=42)

    # 1. Classical JAX Gibbs Sampler with Sokal Autocorrelation Diagnostics
    jax_sampler = get_sampler(BackendType.JAX_BLOCK_GIBBS)
    jax_result = jax_sampler.sample(graph, config)
    print("\n--- JAX Block Gibbs Baseline ---")
    print(jax_result.summary())

    # 2. Extropic TSU Drop-in Backend
    hw_sampler = get_sampler(BackendType.EXTROPIC_TSU, mock_mode=True)
    hw_result = hw_sampler.sample(graph, config)
    print("\n--- Extropic TSU Hardware Adapter ---")
    print(hw_result.summary())

    # Decode best solution
    best_idx = np.argmin(jax_result.energies)
    best_spins = jax_result.samples[best_idx]
    best_binary = ((best_spins + 1) // 2).reshape(len(generators), len(demand_mw))

    print("\nOptimal Generator Dispatch Schedule (Gens x Time):")
    for g_idx, gen in enumerate(generators):
        commitments = "".join([" [ON] " if best_binary[g_idx, t] else " [..] " for t in range(len(demand_mw))])
        print(f"  {gen.id:<14}: {commitments}")


if __name__ == "__main__":
    run_scuc_experiment()
