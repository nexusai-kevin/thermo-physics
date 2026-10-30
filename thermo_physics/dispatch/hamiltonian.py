"""
thermo_physics.dispatch.hamiltonian
===================================
Energy-Based Potts and QUBO Hamiltonian formulation for multi-agent LLM task dispatch.
Vectorized compilation of multi-objective SLA, rate limit, capacity, and cost constraints into
a standardized JAX/THRML Ising HamiltonianGraph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from thermo_physics.backends.sampler_adapter import HamiltonianGraph
from thermo_physics.dispatch.models import AgentTask, ModelEndpoint


@dataclass
class DispatchHamiltonian:
    """Compiled dispatch energy formulation."""
    num_tasks: int
    num_endpoints: int
    num_variables: int
    qubo_matrix: np.ndarray        # Shape: (N, N) where N = M * K
    linear_costs: np.ndarray       # Shape: (N,)
    constant_offset: float
    graph: HamiltonianGraph        # Mapped to bipolar spins s in {-1, +1}
    task_index_map: Dict[Tuple[int, int], int]  # (task_idx, endpoint_idx) -> var_idx


def build_dispatch_hamiltonian(
    tasks: List[AgentTask],
    endpoints: List[ModelEndpoint],
    lambda_one: float = 25.0,        # Exactly-one assignment penalty
    lambda_tpm: float = 12.0,        # Quadratic TPM capacity limit penalty
    lambda_rpm: float = 8.0,         # Quadratic RPM rate limit penalty
    lambda_cost: float = 1.0,        # Economic cost weight
    lambda_latency: float = 18.0,    # Latency SLA breach penalty weight
    lambda_capability: float = 50.0, # Missing capability penalty
) -> DispatchHamiltonian:
    """Compile M tasks and K endpoints into an Ising HamiltonianGraph via vectorized NumPy operations.

    Decision Variable:
      x_{i, k} in {0, 1}: task i assigned to endpoint k.
      Flat index: v = i * K + k
    """
    m = len(tasks)
    k_endpoints = len(endpoints)
    n_vars = m * k_endpoints

    def idx(i: int, k: int) -> int:
        return i * k_endpoints + k

    task_map: Dict[Tuple[int, int], int] = {
        (i, k): i * k_endpoints + k for i in range(m) for k in range(k_endpoints)
    }

    q_mat = np.zeros((n_vars, n_vars), dtype=np.float64)
    c_vec = np.zeros(n_vars, dtype=np.float64)
    const_offset = float(m) * lambda_one

    tokens_arr = np.array([float(t.est_tokens) for t in tasks], dtype=np.float64)

    # ------------------------------------------------------------------------
    # 1. Vectorized Exactly-One Constraint: lambda_one * sum_i (sum_k x_{i, k} - 1)^2
    # ------------------------------------------------------------------------
    for i in range(m):
        u_start = i * k_endpoints
        u_end = u_start + k_endpoints
        c_vec[u_start:u_end] -= lambda_one
        indices_i = np.arange(u_start, u_end)
        # Off-diagonal cross terms between different endpoints for the same task
        q_mat[np.ix_(indices_i, indices_i)] += lambda_one
        for u in indices_i:
            q_mat[u, u] -= lambda_one  # diagonal stays 0 from this term since x^2 = x

    # ------------------------------------------------------------------------
    # 2. Vectorized Rate Limits & Capacity (TPM and RPM)
    # ------------------------------------------------------------------------
    for k in range(k_endpoints):
        ep = endpoints[k]
        tpm_cap = max(1000.0, float(ep.remaining_tpm if ep.remaining_tpm > 0 else ep.tpm_limit))
        rpm_cap = max(10.0, float(ep.remaining_rpm if ep.remaining_rpm > 0 else ep.rpm_limit))

        scale_tpm = lambda_tpm / (tpm_cap ** 2)
        scale_rpm = lambda_rpm / (rpm_cap ** 2)

        # Full cross-task outer product for endpoint k
        block_k = scale_tpm * np.outer(tokens_arr, tokens_arr) + scale_rpm
        indices_k = np.arange(k, n_vars, k_endpoints)
        q_mat[np.ix_(indices_k, indices_k)] += block_k

    # ------------------------------------------------------------------------
    # 3, 4, 5. Linear Costs: Economic, Latency SLA, and Capabilities
    # ------------------------------------------------------------------------
    for i in range(m):
        task = tasks[i]
        t_tokens = tokens_arr[i]
        t_sla = float(task.max_latency_ms)
        p_weight = float(task.priority_weight)
        req_caps = set(task.required_capabilities)

        for k in range(k_endpoints):
            ep = endpoints[k]
            u = idx(i, k)

            # 3. Economic Cost
            cost_usd = (t_tokens / 1000.0) * ep.cost_per_1k_tokens
            c_vec[u] += lambda_cost * cost_usd

            # 4. Latency SLA
            if ep.latency_p50_ms > t_sla:
                excess_ratio = (ep.latency_p50_ms - t_sla) / t_sla
                lat_penalty = lambda_latency * p_weight * ((excess_ratio + 0.5) ** 2)
                c_vec[u] += lat_penalty

            # 5. Capabilities
            sup_caps = set(ep.supported_capabilities)
            missing = req_caps - sup_caps
            if missing:
                c_vec[u] += lambda_capability * (float(len(missing)) ** 2)

    # ------------------------------------------------------------------------
    # Convert QUBO to Ising Spin Model
    # J_uv = - 0.5 * Q_uv
    # h_u = - 0.5 * [ c_u + sum_v Q_uv ]
    # ------------------------------------------------------------------------
    q_off = q_mat.copy()
    np.fill_diagonal(q_off, 0.0)
    couplings = - 0.5 * q_off.astype(np.float32)

    row_sums = np.sum(q_mat, axis=1)
    biases = - 0.5 * (c_vec + row_sums).astype(np.float32)

    total_q_sum = float(np.sum(q_mat))
    total_c_sum = float(np.sum(c_vec))
    ising_offset = const_offset + 0.25 * total_q_sum + 0.5 * total_c_sum

    graph = HamiltonianGraph(
        couplings=couplings,
        biases=biases,
        spin_type="bipolar",
        offset=ising_offset,
    )

    return DispatchHamiltonian(
        num_tasks=m,
        num_endpoints=k_endpoints,
        num_variables=n_vars,
        qubo_matrix=q_mat,
        linear_costs=c_vec,
        constant_offset=const_offset,
        graph=graph,
        task_index_map=task_map,
    )
