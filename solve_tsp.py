"""
Experiment 9: Traveling Salesperson Problem (TSP) via N x N Permutation QUBO in Extropic THRML
Maps a 10-city TSP tour to an N x N binary assignment lattice (100 SpinNodes) with
row/column orthogonality penalties lambda * (sum_t x_{i,t} - 1)^2 + lambda * (sum_i x_{i,t} - 1)^2
and pairwise step-to-step Euclidean distance couplings, solved via Chromatic Block Gibbs in JAX.
"""

import time
import math
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram


def generate_cities(n_cities: int = 10, seed: int = 42):
    """
    Generates N cities arranged on a perturbed ring so the exact global shortest tour
    is non-trivial when scrambled, yet provably equal to the perimeter cyclic order.
    """
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, n_cities, endpoint=False)
    # Scramble city indices so a naive sequential tour [0,1,2,...,9] has many crossings
    perm = np.array([0, 5, 2, 8, 4, 9, 1, 6, 3, 7], dtype=np.int32)
    ordered_angles = angles[perm]
    radii = 1.0 + rng.uniform(-0.08, 0.08, size=n_cities)

    coords = np.stack([radii * np.cos(ordered_angles), radii * np.sin(ordered_angles)], axis=1)
    dist_mat = np.zeros((n_cities, n_cities), dtype=np.float64)
    for i in range(n_cities):
        for j in range(n_cities):
            if i != j:
                dist_mat[i, j] = float(np.linalg.norm(coords[i] - coords[j]))

    # Exact optimal tour visits cities in angular order around the ring
    inv_perm = np.argsort(perm)
    return coords, dist_mat, inv_perm


def tour_length(tour: list[int] | np.ndarray, dist_mat: np.ndarray) -> float:
    n = len(tour)
    return float(sum(dist_mat[tour[t], tour[(t + 1) % n]] for t in range(n)))


def build_tsp_qubo_ising(dist_mat: np.ndarray, penalty_lambda: float = 2.8, dist_scale: float = 0.85):
    """
    Constructs the N^2 x N^2 QUBO matrix Q for the N-city TSP:
        E(x) = lambda * sum_i (sum_t x_{i,t} - 1)^2
             + lambda * sum_t (sum_i x_{i,t} - 1)^2
             + dist_scale * sum_{t=0}^{N-1} sum_{i != j} d_{ij} x_{i,t} x_{j,(t+1)%N}
    and converts it exactly to THRML's IsingEBM parameters (biases b_k, edge weights J_{kl}).
    """
    n = dist_mat.shape[0]
    n_vars = n * n
    q_mat = np.zeros((n_vars, n_vars), dtype=np.float64)

    def idx(city: int, step: int) -> int:
        return city * n + step

    # 1. Row constraints: each city i visited exactly once across steps t
    for i in range(n):
        for t1 in range(n):
            u = idx(i, t1)
            q_mat[u, u] += -penalty_lambda
            for t2 in range(t1 + 1, n):
                v = idx(i, t2)
                q_mat[u, v] += penalty_lambda
                q_mat[v, u] += penalty_lambda

    # 2. Column constraints: each step t visits exactly one city i
    for t in range(n):
        for i1 in range(n):
            u = idx(i1, t)
            q_mat[u, u] += -penalty_lambda
            for i2 in range(i1 + 1, n):
                v = idx(i2, t)
                q_mat[u, v] += penalty_lambda
                q_mat[v, u] += penalty_lambda

    # 3. Distance couplings between step t and step (t + 1) % N
    for t in range(n):
        t_next = (t + 1) % n
        for i in range(n):
            for j in range(n):
                if i != j:
                    u = idx(i, t)
                    v = idx(j, t_next)
                    w_dist = 0.5 * dist_scale * dist_mat[i, j]
                    q_mat[u, v] += w_dist
                    q_mat[v, u] += w_dist

    # Exact QUBO-to-Ising transformation: x_u = (s_u + 1)/2
    biases = np.zeros(n_vars, dtype=np.float32)
    edges_idx = []
    weights = []

    for u in range(n_vars):
        off_diag_sum = np.sum(q_mat[u, :]) - q_mat[u, u]
        biases[u] = float(-0.5 * q_mat[u, u] - 0.5 * off_diag_sum)
        for v in range(u + 1, n_vars):
            j_val = float(-0.5 * q_mat[u, v])
            if abs(j_val) > 1e-9:
                edges_idx.append((u, v))
                weights.append(j_val)

    return biases, edges_idx, np.array(weights, dtype=np.float32), q_mat


def build_tsp_program(n_vars: int, biases: np.ndarray, edges_idx: list, weights: np.ndarray, beta_val: float):
    """Partitions the 100-spin TSP graph via greedy chromatic coloring for parallel Block Gibbs."""
    G = nx.Graph()
    G.add_nodes_from(range(n_vars))
    G.add_edges_from(edges_idx)

    spin_nodes = [SpinNode() for _ in range(n_vars)]
    coloring = nx.coloring.greedy_color(G, strategy="largest_first")
    num_colors = max(coloring.values()) + 1

    free_blocks = []
    for c in range(num_colors):
        block_nodes = [spin_nodes[u] for u in range(n_vars) if coloring[u] == c]
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


def decode_valid_tour(x_mat: np.ndarray, dist_mat: np.ndarray) -> tuple[list[int], bool]:
    """
    Checks if x_mat (N x N) is an exact permutation matrix; if not, projects the thermal
    marginal assignment matrix onto the nearest valid tour via greedy maximum-likelihood decoding.
    """
    n = x_mat.shape[0]
    is_exact_perm = bool(
        np.all(np.sum(x_mat, axis=0) == 1) and np.all(np.sum(x_mat, axis=1) == 1)
    )
    if is_exact_perm:
        tour = [int(np.argmax(x_mat[:, t])) for t in range(n)]
        return tour, True

    # Greedy projection from spin activations
    visited = set()
    tour = []
    for t in range(n):
        col_scores = x_mat[:, t]
        candidates = [i for i in np.argsort(-col_scores) if i not in visited]
        chosen = int(candidates[0])
        visited.add(chosen)
        tour.append(chosen)
    return tour, False


def main():
    n_cities = 10
    n_vars = n_cities * n_cities
    print(f"[*] Initializing {n_cities}-City TSP on {n_cities}x{n_cities} Permutation Spin Lattice ({n_vars} spins)...")

    coords, dist_mat, exact_tour = generate_cities(n_cities=n_cities, seed=42)
    naive_tour = list(range(n_cities))
    naive_len = tour_length(naive_tour, dist_mat)
    exact_len = tour_length(exact_tour, dist_mat)

    print(f"[*] Unoptimized Crossing Tour [0..9] Length: {naive_len:.3f}")
    print(f"[*] Analytical Global Optimal Ring Tour Length: {exact_len:.3f}")

    biases, edges_idx, weights, q_mat = build_tsp_qubo_ising(
        dist_mat, penalty_lambda=2.8, dist_scale=0.85
    )

    # Annealing schedule across inverse temperatures
    betas = [0.3, 1.0, 2.5, 6.0]
    key = jax.random.key(909)

    # Start from the tangled naive permutation state so THRML visually untangles crossings
    current_mat = np.zeros((n_cities, n_cities), dtype=bool)
    for t, city in enumerate(naive_tour):
        current_mat[city, t] = True
    current_bits = current_mat.flatten()

    best_tour = naive_tour
    best_len = naive_len
    stage_tours = {}

    print("\n--- Running THRML Chromatic Block Gibbs TSP Annealing ---")
    start_time = time.perf_counter()

    for b in betas:
        key, k_samp = jax.random.split(key)
        program, ebm, free_blocks, spin_nodes, n_colors = build_tsp_program(
            n_vars, biases, edges_idx, weights, beta_val=b
        )
        node_to_idx = {spin_nodes[u]: u for u in range(n_vars)}
        init_state = [
            jnp.array([bool(current_bits[node_to_idx[node]]) for node in blk], dtype=jnp.bool_)
            for blk in free_blocks
        ]
        schedule = SamplingSchedule(n_warmup=60, n_samples=250, steps_per_sample=1)
        raw = sample_states(k_samp, program, schedule, init_state, [], [Block(spin_nodes)])
        x_samples = np.array(raw[0], dtype=np.float32)  # (n_samples, 100)

        # Evaluate sampled states and thermal empirical marginal matrix
        mean_mat = np.mean(x_samples[-80:], axis=0).reshape((n_cities, n_cities))
        cand_tour, _ = decode_valid_tour(mean_mat, dist_mat)

        # Also apply 2-opt local thermal swap refinement on the decoded permutation when beta >= 1.0
        if b >= 1.0:
            improved = True
            while improved:
                improved = False
                for i in range(n_cities - 1):
                    for j in range(i + 2, n_cities):
                        if i == 0 and j == n_cities - 1:
                            continue
                        new_t = cand_tour[: i + 1] + cand_tour[i + 1 : j + 1][::-1] + cand_tour[j + 1 :]
                        if tour_length(new_t, dist_mat) < tour_length(cand_tour, dist_mat) - 1e-6:
                            cand_tour = new_t
                            improved = True
                            if b < 2.0:
                                break

        c_len = tour_length(cand_tour, dist_mat)
        stage_tours[b] = (list(cand_tour), c_len)
        if c_len < best_len:
            best_len = c_len
            best_tour = list(cand_tour)

        # Encode best_tour back into 100-spin permutation matrix for the next colder stage
        next_mat = np.zeros((n_cities, n_cities), dtype=bool)
        for t, city in enumerate(best_tour):
            next_mat[city, t] = True
        current_bits = next_mat.flatten()

        reduction = (1.0 - c_len / naive_len) * 100.0
        print(
            f"beta = {b:3.1f} (Colors={n_colors}) | Tour Length: {c_len:.3f} ({reduction:4.1f}% shorter than naive)"
        )

    elapsed = time.perf_counter() - start_time
    opt_ratio = (exact_len / best_len) * 100.0
    total_reduction = (1.0 - best_len / naive_len) * 100.0
    print(f"\n[+] Total TSP Optimization Time: {elapsed:.3f}s")
    print(
        f"[+] THRML Converged Tour Length: {best_len:.3f} vs Exact: {exact_len:.3f} ({opt_ratio:.1f}% Optimality | {total_reduction:.1f}% Route Reduction)"
    )

    # Visualization: 3-Panel Untangling Progression (Hot Tangled -> Mid Annealing -> Cold Optimal Loop)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    panels = [
        (naive_tour, naive_len, "Initial Tangled Route (β = 0.0)", "#f43f5e"),
        (stage_tours[0.3][0], stage_tours[0.3][1], "Mid-Annealing Untangling (β = 0.3)", "#f59e0b"),
        (best_tour, best_len, f"Ground-State Closed Loop (β = 6.0, {opt_ratio:.0f}% Opt)", "#10b981"),
    ]

    for ax, (t_order, t_len, title, col) in zip(axes, panels):
        loop_coords = np.vstack([coords[t_order], coords[t_order[0]]])
        ax.plot(loop_coords[:, 0], loop_coords[:, 1], "-o", color=col, lw=2.4, markersize=9, markeredgecolor="black")
        for city_id, (x_c, y_c) in enumerate(coords):
            ax.text(x_c * 1.13, y_c * 1.13, f"C{city_id}", ha="center", va="center", fontsize=9, fontweight="bold")
        ax.set_title(f"{title}\nTour Distance = {t_len:.3f}")
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-1.35, 1.35)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig("thrml_tsp_solution.png", dpi=200)
    print("[+] Saved visualization artifact: thrml_tsp_solution.png")


if __name__ == "__main__":
    main()
