"""
Experiment 9: Traveling Salesperson Problem (TSP) via N x N Permutation QUBO in Extropic THRML

Maps a 10-city tour to an N x N binary assignment lattice (100 SpinNodes) with row/column permutation penalties
    lambda * sum_i (sum_t x_{i,t} - 1)^2 + lambda * sum_t (sum_i x_{i,t} - 1)^2
and step-to-step Euclidean distance couplings, sampled with chromatic block Gibbs in JAX.

Two things are reported separately so the thermodynamic contribution is not conflated with classical post-processing:

  * PURE sampling: tours read directly from sampled states that are exact permutation matrices. No 2-opt, no repair.
  * HYBRID: the same pure result followed by classical 2-opt local search.

Two annealing schedules are compared over several independent restarts:

  * fixed  : penalty lambda constant (2.8) while beta cools 0.2 -> 8.0
  * ramped : penalty lambda ramps 0.5 -> 5.0 (geometrically) while beta cools 0.2 -> 8.0

The reference optimum is computed exactly with the Held-Karp dynamic program.
"""

import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from thrml import Block, SamplingSchedule, SpinNode, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram

N_CITIES = 10
N_STAGES = 6
BETA_RANGE = (0.2, 8.0)
LAMBDA_RAMP = (0.5, 5.0)
LAMBDA_FIXED = 2.8
DIST_SCALE = 0.85
N_RESTARTS = 5
N_WARMUP = 60
N_SAMPLES = 200


def generate_cities(n_cities: int = N_CITIES, seed: int = 42):
    """Cities on a perturbed ring with scrambled labels, so the naive tour [0..N-1] is heavily tangled."""
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, n_cities, endpoint=False)
    perm = np.array([0, 5, 2, 8, 4, 9, 1, 6, 3, 7], dtype=np.int32)
    radii = 1.0 + rng.uniform(-0.08, 0.08, size=n_cities)
    coords = np.stack([radii * np.cos(angles[perm]), radii * np.sin(angles[perm])], axis=1)
    dist_mat = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    ring_tour = [int(c) for c in np.argsort(perm)]
    return coords, dist_mat, ring_tour


def tour_length(tour, dist_mat: np.ndarray) -> float:
    n = len(tour)
    return float(sum(dist_mat[tour[t], tour[(t + 1) % n]] for t in range(n)))


def held_karp(dist_mat: np.ndarray) -> float:
    """Exact optimal closed-tour length via the Held-Karp dynamic program (O(2^n n^2), fine for n = 10)."""
    n = dist_mat.shape[0]
    full = 1 << (n - 1)  # subsets of cities 1..n-1; city 0 is the fixed start
    dp = np.full((full, n - 1), np.inf)
    for j in range(n - 1):
        dp[1 << j, j] = dist_mat[0, j + 1]
    for mask in range(full):
        for j in range(n - 1):
            if not (mask >> j) & 1 or not np.isfinite(dp[mask, j]):
                continue
            for k in range(n - 1):
                if (mask >> k) & 1:
                    continue
                nxt = mask | (1 << k)
                cand = dp[mask, j] + dist_mat[j + 1, k + 1]
                if cand < dp[nxt, k]:
                    dp[nxt, k] = cand
    return float(min(dp[full - 1, j] + dist_mat[j + 1, 0] for j in range(n - 1)))


def two_opt(tour, dist_mat: np.ndarray):
    """Classical 2-opt local search, run to convergence. Used ONLY for the hybrid comparison."""
    tour = list(tour)
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                candidate = tour[: i + 1] + tour[i + 1 : j + 1][::-1] + tour[j + 1 :]
                if tour_length(candidate, dist_mat) < tour_length(tour, dist_mat) - 1e-12:
                    tour = candidate
                    improved = True
    return tour


def build_tsp_qubo_ising(dist_mat: np.ndarray, penalty_lambda: float, dist_scale: float = DIST_SCALE):
    """
    QUBO over x_{i,t} (city i at step t), index u = i * N + t:
        E(x) = lambda * sum_i (sum_t x_{i,t} - 1)^2 + lambda * sum_t (sum_i x_{i,t} - 1)^2
             + dist_scale * sum_t sum_{i != j} d_ij x_{i,t} x_{j,(t+1) mod N}
    converted exactly to THRML's IsingEBM convention E = -beta (sum b_u s_u + sum_{u<v} J_uv s_u s_v) via
    x = (s + 1) / 2:   J_uv = -Q_uv / 2,   b_u = -Q_uu / 2 - (1/2) sum_{v != u} Q_uv.
    """
    n = dist_mat.shape[0]
    n_vars = n * n
    q_mat = np.zeros((n_vars, n_vars), dtype=np.float64)

    def idx(city: int, step: int) -> int:
        return city * n + step

    for i in range(n):  # each city visited exactly once
        for t1 in range(n):
            q_mat[idx(i, t1), idx(i, t1)] += -penalty_lambda
            for t2 in range(t1 + 1, n):
                q_mat[idx(i, t1), idx(i, t2)] += penalty_lambda
                q_mat[idx(i, t2), idx(i, t1)] += penalty_lambda
    for t in range(n):  # each step hosts exactly one city
        for i1 in range(n):
            q_mat[idx(i1, t), idx(i1, t)] += -penalty_lambda
            for i2 in range(i1 + 1, n):
                q_mat[idx(i1, t), idx(i2, t)] += penalty_lambda
                q_mat[idx(i2, t), idx(i1, t)] += penalty_lambda
    for t in range(n):  # tour distance between consecutive steps
        t_next = (t + 1) % n
        for i in range(n):
            for j in range(n):
                if i != j:
                    w = 0.5 * dist_scale * dist_mat[i, j]
                    q_mat[idx(i, t), idx(j, t_next)] += w
                    q_mat[idx(j, t_next), idx(i, t)] += w

    off_diag_sums = q_mat.sum(axis=1) - np.diag(q_mat)
    biases = (-0.5 * np.diag(q_mat) - 0.5 * off_diag_sums).astype(np.float32)
    edges, weights = [], []
    for u in range(n_vars):
        for v in range(u + 1, n_vars):
            if abs(q_mat[u, v]) > 1e-9:
                edges.append((u, v))
                weights.append(-0.5 * q_mat[u, v])
    return biases, edges, np.array(weights, dtype=np.float32)


def build_tsp_program(n_vars: int, biases, edges_idx, weights, beta_val: float):
    """Greedy chromatic colouring of the interaction graph -> conditionally independent free blocks."""
    graph = nx.Graph()
    graph.add_nodes_from(range(n_vars))
    graph.add_edges_from(edges_idx)
    coloring = nx.coloring.greedy_color(graph, strategy="largest_first")
    n_colors = max(coloring.values()) + 1

    nodes = [SpinNode() for _ in range(n_vars)]
    free_blocks = [Block([nodes[u] for u in range(n_vars) if coloring[u] == c]) for c in range(n_colors)]
    ebm = IsingEBM(
        nodes,
        [(nodes[u], nodes[v]) for u, v in edges_idx],
        jnp.asarray(biases),
        jnp.asarray(weights),
        jnp.array(beta_val, dtype=jnp.float32),
    )
    program = IsingSamplingProgram(ebm, free_blocks, clamped_blocks=[])
    return program, free_blocks, nodes, n_colors


def build_stages(dist_mat: np.ndarray, ramped: bool):
    """Pre-build one sampling program per (beta, lambda) stage; they are reused across restarts."""
    betas = np.geomspace(*BETA_RANGE, N_STAGES)
    lams = np.geomspace(*LAMBDA_RAMP, N_STAGES) if ramped else np.full(N_STAGES, LAMBDA_FIXED)
    n_vars = dist_mat.shape[0] ** 2
    stages = []
    for beta, lam in zip(betas, lams):
        biases, edges, weights = build_tsp_qubo_ising(dist_mat, float(lam))
        program, free_blocks, nodes, n_colors = build_tsp_program(n_vars, biases, edges, weights, float(beta))
        stages.append(
            {
                "beta": float(beta),
                "lam": float(lam),
                "program": program,
                "free_blocks": free_blocks,
                "nodes": nodes,
                "n_colors": n_colors,
            }
        )
    return stages


def valid_permutation_mask(x_samples: np.ndarray, n: int) -> np.ndarray:
    """Boolean mask of samples whose N x N assignment matrix is an exact permutation matrix."""
    x = x_samples.reshape(-1, n, n)  # [sample, city, step]
    return (x.sum(axis=2) == 1).all(axis=1) & (x.sum(axis=1) == 1).all(axis=1)


def anneal_once(stages, dist_mat: np.ndarray, restart: int):
    """One annealing run. Returns the best exact-permutation tour sampled (no repair), and per-stage statistics."""
    n = dist_mat.shape[0]
    rng = np.random.default_rng(1000 + restart)
    start_tour = [int(c) for c in rng.permutation(n)]
    current = np.zeros((n, n), dtype=bool)
    for t, city in enumerate(start_tour):
        current[city, t] = True
    current = current.flatten()

    key = jax.random.key(909 + restart)
    best_tour, best_len = None, np.inf
    stage_stats = []
    for stage in stages:
        key, subkey = jax.random.split(key)
        position = {node: u for u, node in enumerate(stage["nodes"])}
        init_state = [
            jnp.array([bool(current[position[node]]) for node in block], dtype=jnp.bool_)
            for block in stage["free_blocks"]
        ]
        schedule = SamplingSchedule(n_warmup=N_WARMUP, n_samples=N_SAMPLES, steps_per_sample=1)
        raw = sample_states(subkey, stage["program"], schedule, init_state, [], [Block(stage["nodes"])])
        samples = np.array(raw[0], dtype=np.int32)
        current = samples[-1].astype(bool)  # pure annealing: next stage continues from the last state

        valid = valid_permutation_mask(samples, n)
        stage_best = np.inf
        for x in samples[valid]:
            tour = [int(np.argmax(x.reshape(n, n)[:, t])) for t in range(n)]
            length = tour_length(tour, dist_mat)
            stage_best = min(stage_best, length)
            if length < best_len:
                best_tour, best_len = tour, length
        stage_stats.append({"valid_rate": 100.0 * float(valid.mean()), "best": stage_best})
    return start_tour, best_tour, best_len, stage_stats


def run_schedule(name: str, stages, dist_mat: np.ndarray):
    print(f"\n--- Schedule: {name} (pure block Gibbs, no 2-opt) ---")
    for s in stages:
        print(f"  stage beta = {s['beta']:5.2f}, lambda = {s['lam']:4.2f}, colour blocks = {s['n_colors']}")
    runs = []
    for r in range(N_RESTARTS):
        start_tour, tour, length, stats = anneal_once(stages, dist_mat, r)
        final = stats[-1]
        status = f"L_pure = {length:.3f}" if tour is not None else "no exact permutation ever sampled"
        print(f"  restart {r}: {status} | final-stage valid-permutation rate = {final['valid_rate']:5.1f}%")
        runs.append({"start": start_tour, "tour": tour, "len": length, "final_valid": final["valid_rate"]})
    return runs


def summarize(name: str, runs, dist_mat: np.ndarray, l_opt: float):
    found = [r for r in runs if r["tour"] is not None]
    summary = {"name": name, "n_found": len(found), "n_runs": len(runs)}
    if not found:
        summary.update(pure_best=None, pure_mean=None, hyb_best=None, hyb_mean=None, valid=0.0)
        return summary, None
    pure = np.array([r["len"] for r in found])
    hybrid_tours = [two_opt(r["tour"], dist_mat) for r in found]
    hybrid = np.array([tour_length(t, dist_mat) for t in hybrid_tours])
    best_idx = int(np.argmin(pure))
    summary.update(
        pure_best=float(pure.min()),
        pure_mean=float(pure.mean()),
        hyb_best=float(hybrid.min()),
        hyb_mean=float(hybrid.mean()),
        valid=float(np.mean([r["final_valid"] for r in runs])),
        hyb_hits_opt=int(np.sum(hybrid <= l_opt + 1e-6)),
        pure_hits_opt=int(np.sum(pure <= l_opt + 1e-6)),
        best_pure_tour=found[best_idx]["tour"],
        best_hybrid_tour=hybrid_tours[best_idx],
        best_start=found[best_idx]["start"],
    )
    return summary, summary


def main():
    n = N_CITIES
    print(f"[*] {n}-city TSP on a {n}x{n} permutation spin lattice ({n * n} spins), {N_RESTARTS} restarts per schedule")
    coords, dist_mat, ring_tour = generate_cities(n)
    l_opt = held_karp(dist_mat)
    l_ring = tour_length(ring_tour, dist_mat)
    l_naive = tour_length(list(range(n)), dist_mat)
    rand_mean = float(np.mean([tour_length(list(np.random.default_rng(s).permutation(n)), dist_mat) for s in range(200)]))
    print(f"[*] Exact optimum (Held-Karp): {l_opt:.3f} (angular ring tour: {l_ring:.3f})")
    print(f"[*] Naive tour [0..9]: {l_naive:.3f} | mean random tour: {rand_mean:.3f}")

    t0 = time.perf_counter()
    results = {}
    for name, ramped in (("fixed lambda = 2.8", False), ("ramped lambda 0.5 -> 5.0", True)):
        stages = build_stages(dist_mat, ramped)
        runs = run_schedule(name, stages, dist_mat)
        results[name] = summarize(name, runs, dist_mat, l_opt)[0]
    elapsed = time.perf_counter() - t0

    print(f"\n=== Pure sampling vs. hybrid (+2-opt); exact optimum L_opt = {l_opt:.3f} ===")
    for name, s in results.items():
        if s["pure_best"] is None:
            print(f"{name}: no restart produced an exact permutation sample")
            continue
        gap_p = 100.0 * (s["pure_best"] / l_opt - 1.0)
        gap_h = 100.0 * (s["hyb_best"] / l_opt - 1.0)
        print(
            f"{name}: pure best {s['pure_best']:.3f} (+{gap_p:.1f}%), mean {s['pure_mean']:.3f}, hit optimum {s['pure_hits_opt']}/{s['n_found']} | "
            f"hybrid best {s['hyb_best']:.3f} (+{gap_h:.1f}%), mean {s['hyb_mean']:.3f}, hit optimum {s['hyb_hits_opt']}/{s['n_found']} | "
            f"restarts with a valid tour {s['n_found']}/{s['n_runs']}, final-stage valid rate {s['valid']:.1f}%"
        )
    print(f"[+] Total time: {elapsed:.1f}s")

    ramped = results["ramped lambda 0.5 -> 5.0"]
    fixed = results["fixed lambda = 2.8"]
    print(
        "RESULT "
        f"opt={l_opt:.3f} naive={l_naive:.3f} "
        f"ramped_pure_best={ramped['pure_best'] if ramped['pure_best'] is not None else float('nan'):.3f} "
        f"ramped_pure_mean={ramped['pure_mean'] if ramped['pure_mean'] is not None else float('nan'):.3f} "
        f"ramped_hybrid_best={ramped['hyb_best'] if ramped['hyb_best'] is not None else float('nan'):.3f} "
        f"ramped_valid={ramped['valid']:.1f} "
        f"fixed_pure_best={fixed['pure_best'] if fixed['pure_best'] is not None else float('nan'):.3f} "
        f"fixed_valid={fixed['valid']:.1f}"
    )

    # Figure: random start -> best pure-sampled tour -> hybrid, for the best schedule that produced a valid tour.
    chosen = ramped if ramped["pure_best"] is not None else fixed
    if chosen["pure_best"] is None:
        return
    start = chosen["best_start"]
    panels = [
        (start, "Random start (permutation state)", "#f43f5e"),
        (chosen["best_pure_tour"], f"PURE sampling ({chosen['name']})", "#f59e0b"),
        (chosen["best_hybrid_tour"], "HYBRID: pure + 2-opt", "#10b981"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (order, title, color) in zip(axes, panels):
        loop = np.vstack([coords[order], coords[order[0]]])
        ax.plot(loop[:, 0], loop[:, 1], "-o", color=color, lw=2.4, markersize=9, markeredgecolor="black")
        for city, (x_c, y_c) in enumerate(coords):
            ax.text(x_c * 1.13, y_c * 1.13, f"C{city}", ha="center", va="center", fontsize=9, fontweight="bold")
        ax.set_title(f"{title}\nL = {tour_length(order, dist_mat):.3f} (optimum {l_opt:.3f})")
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-1.35, 1.35)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig("thrml_tsp_solution.png", dpi=200)
    print("[+] Saved visualization artifact: thrml_tsp_solution.png")


if __name__ == "__main__":
    main()
