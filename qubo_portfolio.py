"""
Experiment 7: QUBO Financial Portfolio Optimization in Extropic THRML
Maps a constrained Markowitz portfolio selection problem (select exactly K=5 out of N=20 assets)
into an Ising Hamiltonian sampled across a thermal annealing schedule via Block Gibbs in JAX.
"""

import time
from itertools import combinations
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram


def generate_market_instance(n_assets: int = 20, seed: int = 42):
    """
    Generates expected returns (mu) and covariance matrix (Sigma) for N=20 assets.
    """
    rng = np.random.default_rng(seed)

    # Annualized expected returns between 5% and 20%
    mu = rng.uniform(0.05, 0.20, size=n_assets).astype(np.float64)
    # Plant a high-Sharpe 5-asset portfolio so the exact optimum is crisp and verifiable
    star_assets = [2, 5, 9, 14, 18]
    for idx in star_assets:
        mu[idx] += 0.12

    # Positive-definite covariance matrix with moderate cross-asset correlations
    factor_loadings = rng.normal(0.0, 0.08, size=(n_assets, 4))
    for k, idx in enumerate(star_assets):
        factor_loadings[idx] = np.eye(4)[k % 4] * 0.04

    idiosyncratic = np.diag(rng.uniform(0.03, 0.06, size=n_assets))
    sigma = factor_loadings @ factor_loadings.T + idiosyncratic
    return mu, sigma


def qubo_to_ising(
    mu: np.ndarray,
    sigma: np.ndarray,
    k_target: int = 5,
    risk_aversion: float = 1.0,
    penalty_lambda: float = 0.65,
):
    """
    Converts the Markowitz QUBO objective:
        min_{x in {0,1}^N}  q * x^T Sigma x - mu^T x + lambda * (sum_i x_i - K)^2
    via x_i = (s_i + 1) / 2, s_i in {-1, +1} into THRML's IsingEBM convention:
        E_thrml(s) = -beta * (sum_i b_i s_i + sum_{i<j} J_{ij} s_i s_j)

    Expanding Q(x) = sum_i Q_ii x_i + 2 * sum_{i<j} Q_ij x_i x_j + const:
        Q_ii = q * Sigma_ii - mu_i + lambda * (1 - 2*K)
        Q_ij = q * Sigma_ij + lambda   (for i != j)

    Substituting x_i = (s_i + 1)/2 gives:
        E(s) = sum_{i<j} (Q_ij / 2) s_i s_j + sum_i (Q_ii / 2 + sum_{j != i} Q_ij / 4) s_i + const
    Matching E(s) = -(sum_i b_i s_i + sum_{i<j} J_{ij} s_i s_j) yields:
        J_{ij} = -0.5 * Q_ij
        b_i    = -0.5 * Q_ii - 0.5 * sum_{j != i} Q_ij
    """
    n = len(mu)
    q_mat = risk_aversion * sigma.copy() + penalty_lambda * np.ones((n, n), dtype=np.float64)
    np.fill_diagonal(
        q_mat,
        risk_aversion * np.diag(sigma) - mu + penalty_lambda * (1.0 - 2.0 * k_target),
    )

    biases = np.zeros(n, dtype=np.float32)
    edges_idx = []
    weights = []

    for i in range(n):
        off_diag_sum = np.sum(q_mat[i, :]) - q_mat[i, i]
        biases[i] = float(-0.5 * q_mat[i, i] - 0.5 * off_diag_sum)
        for j in range(i + 1, n):
            j_val = float(-0.5 * q_mat[i, j])
            if abs(j_val) > 1e-9:
                edges_idx.append((i, j))
                weights.append(j_val)

    return biases, edges_idx, np.array(weights, dtype=np.float32), q_mat


def evaluate_portfolio(
    x: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    k_target: int = 5,
    risk_aversion: float = 1.0,
    penalty_lambda: float = 0.65,
):
    """Computes expected return, volatility sqrt(x^T Sigma x), and total QUBO energy."""
    ret = float(mu @ x)
    var = float(x.T @ sigma @ x)
    risk = float(np.sqrt(max(var, 0.0)))
    card_err = int(np.sum(x)) - k_target
    qubo_obj = risk_aversion * var - ret + penalty_lambda * (card_err ** 2)
    return ret, risk, qubo_obj


def brute_force_exact(
    mu: np.ndarray,
    sigma: np.ndarray,
    k_target: int = 5,
    risk_aversion: float = 1.0,
    penalty_lambda: float = 0.65,
):
    """Enumerates all (N choose K) feasible portfolios to find the exact global optimum."""
    n = len(mu)
    best_x = None
    best_obj = float("inf")
    feasible_cloud = []

    for comb in combinations(range(n), k_target):
        x = np.zeros(n, dtype=np.int32)
        x[list(comb)] = 1
        ret, risk, obj = evaluate_portfolio(x, mu, sigma, k_target, risk_aversion, penalty_lambda)
        feasible_cloud.append((risk, ret, obj, x))
        if obj < best_obj:
            best_obj = obj
            best_x = x.copy()

    return best_x, best_obj, feasible_cloud


def build_qubo_program(n_assets: int, biases: np.ndarray, edges_idx: list, weights: np.ndarray, beta_val: float):
    """
    Builds the THRML IsingEBM and partitions nodes via greedy chromatic coloring.
    """
    G = nx.Graph()
    G.add_nodes_from(range(n_assets))
    G.add_edges_from(edges_idx)

    spin_nodes = [SpinNode() for _ in range(n_assets)]
    coloring = nx.coloring.greedy_color(G, strategy="largest_first")
    num_colors = max(coloring.values()) + 1

    free_blocks = []
    for c in range(num_colors):
        block_nodes = [spin_nodes[i] for i in range(n_assets) if coloring[i] == c]
        if block_nodes:
            free_blocks.append(Block(block_nodes))

    edges = [(spin_nodes[u], spin_nodes[v]) for u, v in edges_idx]
    ebm = IsingEBM(
        nodes=spin_nodes,
        edges=edges,
        biases=jnp.array(biases, dtype=jnp.float32),
        weights=jnp.array(weights, dtype=jnp.float32),
        beta=jnp.array(beta_val, dtype=jnp.float32),
    )
    program = IsingSamplingProgram(ebm, free_blocks=free_blocks, clamped_blocks=[])
    return program, ebm, free_blocks, spin_nodes, num_colors


def main():
    n_assets = 20
    k_target = 5
    risk_aversion = 1.0
    penalty_lambda = 0.65

    print(f"[*] Initializing QUBO Portfolio Optimizer (Select K={k_target} from N={n_assets} assets)...")
    mu, sigma = generate_market_instance(n_assets=n_assets, seed=42)
    biases, edges_idx, weights, _ = qubo_to_ising(
        mu, sigma, k_target=k_target, risk_aversion=risk_aversion, penalty_lambda=penalty_lambda
    )

    # Exact brute-force benchmark over all (20 choose 5) = 15,504 feasible portfolios
    exact_x, exact_obj, feasible_cloud = brute_force_exact(
        mu, sigma, k_target=k_target, risk_aversion=risk_aversion, penalty_lambda=penalty_lambda
    )
    exact_ret, exact_risk, _ = evaluate_portfolio(
        exact_x, mu, sigma, k_target, risk_aversion, penalty_lambda
    )
    exact_indices = np.where(exact_x == 1)[0].tolist()
    print(f"[*] Brute-Force Global Optimum (C(20,5)=15,504 combinations):")
    print(f"    -> Selected Assets: {exact_indices} | Objective: {exact_obj:.4f} | Return: {exact_ret*100:.2f}% | Risk: {exact_risk:.4f}")

    # Thermal annealing schedule in THRML
    betas = [0.4, 0.8, 1.5, 2.5, 5.0, 12.0]
    key = jax.random.key(777)

    # Initialize near the target cardinality (p = K/N = 0.25) so high-T exploration starts in the K=5 basin
    rng_init = np.random.default_rng(42)
    init_bits = np.zeros(n_assets, dtype=bool)
    init_bits[rng_init.choice(n_assets, size=k_target, replace=False)] = True

    best_sampled_x = None
    best_sampled_obj = float("inf")
    best_any_x = None
    best_any_obj = float("inf")

    print("\n--- Running THRML QUBO Thermodynamic Annealing Sweep ---")
    start_time = time.perf_counter()

    current_bits = init_bits.copy()
    for b in betas:
        key, k_samp = jax.random.split(key)
        program, ebm, free_blocks, spin_nodes, n_colors = build_qubo_program(
            n_assets, biases, edges_idx, weights, beta_val=b
        )
        node_to_idx = {spin_nodes[i]: i for i in range(n_assets)}
        # Each 1D free_block of size M must be initialized with a 1D array of shape (M,)
        init_state = [
            jnp.array([bool(current_bits[node_to_idx[node]]) for node in blk], dtype=jnp.bool_)
            for blk in free_blocks
        ]
        schedule = SamplingSchedule(n_warmup=60, n_samples=350, steps_per_sample=1)

        raw_samples = sample_states(k_samp, program, schedule, init_state, [], [Block(spin_nodes)])
        x_samples = np.array(raw_samples[0], dtype=np.int32)

        card_counts = np.sum(x_samples, axis=1)
        feasible_mask = card_counts == k_target
        feas_rate = float(np.mean(feasible_mask) * 100.0)

        objs = []
        stage_best_bits = current_bits
        stage_best_obj = float("inf")
        for idx_s, x_vec in enumerate(x_samples):
            _, _, obj = evaluate_portfolio(x_vec, mu, sigma, k_target, risk_aversion, penalty_lambda)
            objs.append(obj)
            if obj < stage_best_obj:
                stage_best_obj = obj
                stage_best_bits = x_vec.astype(bool)
            if obj < best_any_obj:
                best_any_obj = obj
                best_any_x = x_vec.copy()
            if np.sum(x_vec) == k_target and obj < best_sampled_obj:
                best_sampled_obj = obj
                best_sampled_x = x_vec.copy()

        # Seed the next colder stage from the lowest-energy state discovered so far
        current_bits = best_sampled_x.astype(bool) if best_sampled_x is not None else stage_best_bits
        mean_obj = float(np.mean(objs))
        print(
            f"beta = {b:4.1f} | Budget Feasible (sum x_i = {k_target}): {feas_rate:5.1f}% | Mean QUBO Energy: {mean_obj:+.4f} | Min Energy: {stage_best_obj:+.4f}"
        )

    elapsed = time.perf_counter() - start_time
    if best_sampled_x is None:
        best_sampled_x = best_any_x
        best_sampled_obj = best_any_obj

    sampled_indices = np.where(best_sampled_x == 1)[0].tolist()
    sampled_ret, sampled_risk, _ = evaluate_portfolio(
        best_sampled_x, mu, sigma, k_target, risk_aversion, penalty_lambda
    )
    optimality_ratio = (best_sampled_obj / exact_obj) * 100.0 if exact_obj != 0 else 100.0

    print(f"\n[+] Total Optimization Time: {elapsed:.3f}s")
    print(f"[+] THRML Sampled Optimal Portfolio: {sampled_indices} (Sum x_i = {int(np.sum(best_sampled_x))})")
    print(f"[+] Sampled Objective: {best_sampled_obj:.4f} vs Exact: {exact_obj:.4f} ({optimality_ratio:.1f}% Match)")

    # Generate 2-panel visualization artifact
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: Risk-Return Pareto Frontier & Feasible Portfolio Cloud
    cloud_risks = [c[0] for c in feasible_cloud]
    cloud_rets = [c[1] * 100.0 for c in feasible_cloud]
    cloud_objs = [c[2] for c in feasible_cloud]

    sc = axes[0].scatter(
        cloud_risks, cloud_rets, c=cloud_objs, cmap="viridis_r", s=12, alpha=0.45, label="Feasible Portfolios (K=5)"
    )
    plt.colorbar(sc, ax=axes[0], label="QUBO Objective Energy")

    axes[0].scatter(
        [exact_risk],
        [exact_ret * 100.0],
        color="#ef4444",
        marker="*",
        s=260,
        edgecolor="black",
        zorder=5,
        label=f"Exact Global Optimum {exact_indices}",
    )
    axes[0].scatter(
        [sampled_risk],
        [sampled_ret * 100.0],
        color="#10b981",
        marker="o",
        s=120,
        facecolor="none",
        linewidth=2.5,
        zorder=6,
        label=f"THRML Sampled Optimum {sampled_indices}",
    )
    axes[0].set_title("Risk-Return Pareto Landscape (N=20, K=5)")
    axes[0].set_xlabel("Portfolio Volatility Risk (√(xᵀΣx))")
    axes[0].set_ylabel("Expected Return (%)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="lower right", fontsize=9)

    # Panel 2: Asset Allocation & Expected Returns Bar Chart
    asset_ids = np.arange(n_assets)
    bar_colors = ["#10b981" if best_sampled_x[i] == 1 else "#94a3b8" for i in range(n_assets)]
    axes[1].bar(asset_ids, mu * 100.0, color=bar_colors, edgecolor="#1e293b", width=0.7)
    for idx in sampled_indices:
        axes[1].text(idx, mu[idx] * 100.0 + 0.6, "★", ha="center", va="bottom", color="#047857", fontsize=12)

    axes[1].set_xticks(asset_ids)
    axes[1].set_title(f"Optimal K={k_target} Asset Allocation (Green = Selected by THRML)")
    axes[1].set_xlabel("Asset Index (0–19)")
    axes[1].set_ylabel("Expected Annual Return (%)")
    axes[1].set_ylim(0, max(mu * 100.0) * 1.15)
    axes[1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("thrml_qubo_portfolio.png", dpi=200)
    print("[+] Saved visualization artifact: thrml_qubo_portfolio.png")


if __name__ == "__main__":
    main()
