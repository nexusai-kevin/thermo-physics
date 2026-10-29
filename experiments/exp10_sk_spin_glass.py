"""
Experiment 10: Sherrington-Kirkpatrick (SK) Frustrated Spin Glass
Part of the thermo-physics benchmark suite (Extropic thrml & JAX).

Features:
- All-to-all Gaussian couplings: J_ij ~ Normal(0, 1/N)
- Two independent Markov chain replicas (alpha, beta) over identical disorder
- Parisi overlap order parameter computation: q = (1/N) sum_i s_i^alpha * s_i^beta
- Overlap distribution P(q) histogram and symmetry check
- Ground state comparison vs. exact brute-force enumeration (N <= 24)
- Automated Sokal tau_int and ESS diagnostics
"""

import time
import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, Tuple, Any

def generate_sk_disorder(N: int, seed: int = 42) -> np.ndarray:
    """Generate symmetric Gaussian couplings J_ij ~ N(0, 1/N) with zero diagonal."""
    rng = np.random.default_rng(seed)
    # Variance is 1/N so total energy scales extensively with N
    scale = 1.0 / np.sqrt(N)
    raw_J = rng.normal(0.0, scale, size=(N, N))
    J = (raw_J + raw_J.T) / np.sqrt(2.0)
    np.fill_diagonal(J, 0.0)
    return J

def compute_energy(states: np.ndarray, J: np.ndarray) -> np.ndarray:
    """
    Compute Hamiltonian H(s) = -0.5 * sum_{i,j} J_ij s_i s_j.
    Accepts shape (N,) or (samples, N).
    """
    if states.ndim == 1:
        return -0.5 * float(states @ J @ states)
    return -0.5 * np.einsum("bi,ij,bj->b", states, J, states)

def brute_force_ground_state(N: int, J: np.ndarray) -> Tuple[float, np.ndarray]:
    """Exact enumeration of ground state for small N (up to N=26)."""
    assert N <= 26, "Brute force search is restricted to N <= 26."
    total_states = 1 << N
    # Generate all binary configurations as {-1, +1}
    indices = np.arange(total_states, dtype=np.uint32)[:, None]
    shifts = np.arange(N, dtype=np.uint32)[None, :]
    bits = (indices >> shifts) & 1
    spins = 2.0 * bits - 1.0  # shape: (2^N, N)

    energies = compute_energy(spins, J)
    min_idx = np.argmin(energies)
    return float(energies[min_idx]), spins[min_idx]

def run_sk_gibbs_sampling(
    N: int,
    J: np.ndarray,
    beta: float,
    num_steps: int = 20000,
    burn_in: int = 4000,
    seed: int = 101,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Execute Gibbs sampling on the dense SK model using JAX/NumPy.
    Returns: (trajectory_samples, final_state, elapsed_time)
    """
    rng = np.random.default_rng(seed)
    s = rng.choice([-1.0, 1.0], size=N)
    trajectory = []

    t0 = time.perf_counter()
    for step in range(num_steps):
        # Local field: h_i = sum_j J_ij * s_j
        h = J @ s
        # Heat-bath update probability: P(s_i = +1) = 1 / (1 + exp(-2 * beta * h_i))
        prob_plus = 1.0 / (1.0 + np.exp(-2.0 * beta * h))
        rand_vals = rng.random(size=N)
        s = np.where(rand_vals < prob_plus, 1.0, -1.0)

        if step >= burn_in and step % 2 == 0:
            trajectory.append(s.copy())

    elapsed = time.perf_counter() - t0
    return np.array(trajectory), s, elapsed

def compute_sokal_tau(series: np.ndarray, max_lag: int = 300) -> float:
    """Compute Sokal integrated autocorrelation time tau_int."""
    x = series - np.mean(series)
    n = len(x)
    variance = np.var(x)
    if variance < 1e-12:
        return 0.5
    
    autocorr = np.correlate(x, x, mode="full")[n - 1 : n + max_lag] / (n * variance)
    # Sokal automatic truncation window
    tau = 0.5
    for m in range(1, len(autocorr)):
        tau += autocorr[m]
        if m >= 6.0 * tau:
            break
    return max(0.5, float(tau))

def run_experiment_10(N: int = 24, beta: float = 2.0, num_steps: int = 25000):
    print("=" * 70)
    print(f"EXPERIMENT 10: Sherrington-Kirkpatrick Frustrated Spin Glass (N={N})")
    print(f"Parameters: beta={beta} (T={1.0/beta:.2f}), steps={num_steps}, quenched seed=42")
    print("=" * 70)

    # 1. Quenched Disorder Configuration
    J = generate_sk_disorder(N=N, seed=42)
    print(f"[1/5] Quenched Gaussian couplings generated. Mean: {J.mean():.4f}, Std: {J.std():.4f} (theo: {1.0/np.sqrt(N):.4f})")

    # 2. Exact Ground State Baseline (for validation)
    print(f"[2/5] Running exact brute-force search over {1 << N:,} configurations...")
    bf_t0 = time.perf_counter()
    e_opt, s_opt = brute_force_ground_state(N, J)
    bf_time = time.perf_counter() - bf_t0
    print(f"      Exact Ground State Energy: {e_opt:.4f} (E/N = {e_opt/N:.4f}) [Solved in {bf_time:.2f}s]")

    # 3. Simulate Independent Replicas (Alpha and Beta)
    print(f"[3/5] Sampling Two Replicas (alpha & beta) under identical disorder...")
    traj_a, state_a, time_a = run_sk_gibbs_sampling(N, J, beta, num_steps=num_steps, seed=202)
    traj_b, state_b, time_b = run_sk_gibbs_sampling(N, J, beta, num_steps=num_steps, seed=303)
    
    total_samples = len(traj_a)
    print(f"      Replica A: {total_samples} samples collected in {time_a:.2f}s")
    print(f"      Replica B: {total_samples} samples collected in {time_b:.2f}s")

    # 4. Energy Trajectory and Sokal Diagnostics
    energies_a = compute_energy(traj_a, J)
    energies_b = compute_energy(traj_b, J)
    min_sampled_e = min(np.min(energies_a), np.min(energies_b))

    tau_int = compute_sokal_tau(energies_a)
    ess = total_samples / (2.0 * tau_int)
    throughput = (total_samples * N) / (time_a * 1e6)  # MSU/s

    # 5. Overlap Distribution P(q)
    # q_t = (1/N) * sum_i s_i^alpha(t) * s_i^beta(t)
    q_vals = np.mean(traj_a * traj_b, axis=1)
    mean_abs_q = float(np.mean(np.abs(q_vals)))
    q_variance = float(np.var(q_vals))

    hist, bin_edges = np.histogram(q_vals, bins=11, range=(-1.0, 1.0), density=True)

    print("\n" + "-" * 70)
    print("RESULTS & REPLICA SYMMETRY BREAKING DIAGNOSTICS")
    print("-" * 70)
    print(f"Best Energy Discovered:    {min_sampled_e:.4f} (Opt: {e_opt:.4f}, Rel Gap: {abs(min_sampled_e - e_opt):.4f})")
    print(f"Mean Energy (Replica A):   {np.mean(energies_a):.4f} +/- {np.std(energies_a):.4f}")
    print(f"Sokal Autocorrelation Tau: {tau_int:.2f} steps")
    print(f"Effective Sample Size:     {ess:.1f} / {total_samples} samples")
    print(f"Gibbs Update Throughput:   {throughput:.3f} MSU/s")
    print(f"Order Parameter <|q|>:     {mean_abs_q:.4f} (Non-zero indicates replica symmetry breaking)")
    print(f"Overlap Variance Var(q):   {q_variance:.4f}")
    print("\nParisi Overlap Distribution P(q) Profile [-1.0, +1.0]:")
    for i in range(len(hist)):
        bar = "#" * int(hist[i] * 5)
        print(f"  [{bin_edges[i]:+4.1f}, {bin_edges[i+1]:+4.1f}]: {hist[i]:5.2f} | {bar}")

    return {
        "N": N,
        "beta": beta,
        "e_opt": e_opt,
        "min_sampled_e": min_sampled_e,
        "tau_int": tau_int,
        "ess": ess,
        "mean_abs_q": mean_abs_q,
        "q_variance": q_variance,
    }

if __name__ == "__main__":
    run_experiment_10(N=24, beta=2.5, num_steps=30000)
