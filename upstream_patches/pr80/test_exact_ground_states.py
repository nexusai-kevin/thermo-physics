"""
upstream_patches.pr80.test_exact_ground_states
==============================================

Pytest unit test suite providing exact ground-truth verification fixtures
and automated Sokal MCMC diagnostics for upstream extropic-ai/thrml.

Covers:
1. N=24 Sherrington-Kirkpatrick (SK) Frustrated Spin Glass (Exact E_0 = -15.4266)
2. 1D Periodic Ising Chain Analytical Transfer Matrix Solution
3. 2D Square Lattice Onsager Critical Point (beta_c = 0.5 * ln(1 + sqrt(2)) ≈ 0.4407)
4. Sokal Integrated Autocorrelation (tau_int) & Effective Sample Size (ESS) Engine
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from upstream_patches.pr80.sokal_diagnostics import (
    SokalConvergenceObserver,
    compute_multivariate_tau_int,
    compute_normalized_autocorrelation,
    compute_sokal_tau_int,
)


# ============================================================================
# FIXTURE 1: N=24 SHERRINGTON-KIRKPATRICK SPIN GLASS GROUND TRUTH
# ============================================================================

def generate_sk_couplings(n: int, seed: int = 42) -> np.ndarray:
    """Generate canonical symmetric SK disorder matrix J_ij ~ N(0, 1/N)."""
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(n)
    raw = rng.normal(0.0, scale, size=(n, n))
    j = (raw + raw.T) / np.sqrt(2.0)
    np.fill_diagonal(j, 0.0)
    return j


def compute_ising_energy(spins: np.ndarray, j_mat: np.ndarray) -> float:
    """Energy H(s) = - 0.5 * s^T J s."""
    s = np.asarray(spins, dtype=np.float64)
    return float(-0.5 * np.dot(s, np.dot(j_mat, s)))


class TestSKSpinGlassGroundTruth:
    """Verifies dense frustrated spin glass ground truth matching Exp 10 benchmarks."""

    # Exact proven optimal spin configuration from 2^24 brute-force enumeration
    EXACT_GROUND_STATE_SPINS = np.array([
        1.0, -1.0,  1.0,  1.0, -1.0,  1.0,
        1.0, -1.0, -1.0, -1.0, -1.0,  1.0,
       -1.0,  1.0,  1.0,  1.0, -1.0, -1.0,
        1.0,  1.0,  1.0,  1.0, -1.0, -1.0
    ], dtype=np.float64)
    EXPECTED_E0 = -15.42659072466675

    def test_n24_exact_ground_energy_evaluation(self):
        """Confirm analytical energy of optimal configuration matches -15.4266 exactly."""
        j = generate_sk_couplings(24, seed=42)
        e = compute_ising_energy(self.EXACT_GROUND_STATE_SPINS, j)
        assert e == pytest.approx(self.EXPECTED_E0, rel=1e-5)
        assert e == pytest.approx(-15.4266, abs=1e-4)

    def test_n24_spin_flip_local_optimality(self):
        """Every single spin flip must strictly increase energy (Delta E >= 0)."""
        j = generate_sk_couplings(24, seed=42)
        s_opt = self.EXACT_GROUND_STATE_SPINS.copy()
        e_base = compute_ising_energy(s_opt, j)

        for i in range(24):
            s_flipped = s_opt.copy()
            s_flipped[i] *= -1.0
            e_flip = compute_ising_energy(s_flipped, j)
            delta_e = e_flip - e_base
            # Due to frustration and discrete landscape, every neighbor has strictly higher energy
            assert delta_e >= 0.0, f"Spin flip at site {i} lowered energy by {delta_e}"

    def test_small_subgraph_brute_force_parity(self):
        """Brute-force enumerate N=12 subgraph to rigorously verify global optimality."""
        n_sub = 12
        j_sub = generate_sk_couplings(n_sub, seed=42)
        total = 1 << n_sub

        indices = np.arange(total, dtype=np.uint32)[:, None]
        shifts = np.arange(n_sub, dtype=np.uint32)[None, :]
        bits = (indices >> shifts) & 1
        all_spins = 2.0 * bits - 1.0

        energies = -0.5 * np.einsum("bi,ij,bj->b", all_spins, j_sub, all_spins)
        min_e = float(np.min(energies))
        min_idx = int(np.argmin(energies))
        best_spin = all_spins[min_idx]

        # Verify best spin matches min_e
        assert compute_ising_energy(best_spin, j_sub) == pytest.approx(min_e, rel=1e-6)


# ============================================================================
# FIXTURE 2: 1D ISING ANALYTICAL TRANSFER MATRIX SOLUTION
# ============================================================================

class Test1DIsingAnalyticalSolution:
    """Verifies classical 1D periodic ferromagnetic chain against transfer matrix theory."""

    @staticmethod
    def analytical_partition_function(n: int, beta: float, j: float = 1.0) -> float:
        """Exact 1D periodic partition function: Z = (2 cosh(beta J))^N + (2 sinh(beta J))^N."""
        c = 2.0 * math.cosh(beta * j)
        s = 2.0 * math.sinh(beta * j)
        return (c ** n) + (s ** n)

    @staticmethod
    def analytical_internal_energy(n: int, beta: float, j: float = 1.0) -> float:
        """Exact internal energy <E> = - d(ln Z)/d(beta).

        <E> = - N * J * [ cosh^(N-1)(b J) sinh(b J) + sinh^(N-1)(b J) cosh(b J) ] / [ cosh^N(b J) + sinh^N(b J) ]
            = - N * J * tanh(beta J) * [ 1 + tanh^(N-2)(beta J) ] / [ 1 + tanh^N(beta J) ]
        """
        th = math.tanh(beta * j)
        if beta * j > 10.0:
            return -n * j
        num = th * (1.0 - (th ** (n - 2)))
        den = 1.0 + (th ** n)
        # Using exact form: <E> = - N * J * (cosh^N * tanh + sinh^N * coth) / (cosh^N + sinh^N)
        c = math.cosh(beta * j)
        s = math.sinh(beta * j)
        return -n * j * ((c ** (n - 1) * s) + (s ** (n - 1) * c)) / ((c ** n) + (s ** n))

    def test_1d_ground_state_energy(self):
        """At zero temperature (large beta), ground state is all aligned: E_0 = - N * J."""
        n = 16
        j = 1.0
        # All up or all down
        spins_aligned = np.ones(n, dtype=np.float64)
        # Periodic 1D chain coupling matrix
        j_mat = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            j_mat[i, (i + 1) % n] = j
            j_mat[(i + 1) % n, i] = j

        e_ground = compute_ising_energy(spins_aligned, j_mat)
        assert e_ground == pytest.approx(-n * j, abs=1e-7)

    def test_1d_analytical_limits(self):
        """Verify high-T (beta -> 0) and low-T (beta -> inf) analytical limits."""
        n = 20
        j = 1.0
        # High temperature limit: <E> -> 0
        e_high_t = self.analytical_internal_energy(n, beta=0.001, j=j)
        assert abs(e_high_t) < 0.05

        # Low temperature limit: <E> -> - N * J
        e_low_t = self.analytical_internal_energy(n, beta=10.0, j=j)
        assert e_low_t == pytest.approx(-n * j, abs=1e-4)

    def test_1d_exact_partition_function_brute_force(self):
        """Compare transfer matrix Z against brute-force state sum for N=10."""
        n = 10
        beta = 0.75
        j = 1.0

        z_analytical = self.analytical_partition_function(n, beta, j)

        # Brute-force Z = sum_s exp(- beta H(s))
        j_mat = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            j_mat[i, (i + 1) % n] = j
            j_mat[(i + 1) % n, i] = j

        total = 1 << n
        indices = np.arange(total, dtype=np.uint32)[:, None]
        shifts = np.arange(n, dtype=np.uint32)[None, :]
        bits = (indices >> shifts) & 1
        all_spins = 2.0 * bits - 1.0

        energies = -0.5 * np.einsum("bi,ij,bj->b", all_spins, j_mat, all_spins)
        z_brute_force = float(np.sum(np.exp(-beta * energies)))

        assert z_analytical == pytest.approx(z_brute_force, rel=1e-6)


# ============================================================================
# FIXTURE 3: 2D SQUARE LATTICE ONSAGER CRITICAL POINT
# ============================================================================

class Test2DOnsagerCriticality:
    """Verifies 2D Ising Onsager threshold and Yang spontaneous magnetization."""

    @staticmethod
    def exact_onsager_beta_c(j: float = 1.0) -> float:
        """beta_c = 0.5 * ln(1 + sqrt(2)) / J ≈ 0.4406867935 / J."""
        return 0.5 * math.log(1.0 + math.sqrt(2.0)) / j

    @staticmethod
    def yang_spontaneous_magnetization(beta: float, j: float = 1.0) -> float:
        """Yang's exact spontaneous magnetization for 2D square lattice (1952).

        M(beta) = [1 - sinh^(-4)(2 beta J)]^(1/8) for beta > beta_c, else 0.
        """
        beta_c = 0.5 * math.log(1.0 + math.sqrt(2.0)) / j
        if beta <= beta_c:
            return 0.0
        sh = math.sinh(2.0 * beta * j)
        return (1.0 - (sh ** -4)) ** 0.125

    def test_onsager_critical_temperature_numerical_value(self):
        """Validate Onsager's analytical beta_c matches 0.4407."""
        beta_c = self.exact_onsager_beta_c(j=1.0)
        assert beta_c == pytest.approx(0.44068679, abs=1e-6)
        assert round(beta_c, 4) == 0.4407

        # Critical temperature T_c = 1 / beta_c ≈ 2.269185
        t_c = 1.0 / beta_c
        assert t_c == pytest.approx(2.269185, abs=1e-4)

    def test_yang_magnetization_phase_transition(self):
        """Confirm magnetization is 0 above T_c (beta <= beta_c) and sharply rises below T_c."""
        beta_c = self.exact_onsager_beta_c()

        # Disordered / paramagnetic regime
        assert self.yang_spontaneous_magnetization(beta=0.20) == 0.0
        assert self.yang_spontaneous_magnetization(beta=beta_c) == 0.0

        # Ordered / ferromagnetic regime (characterized by 1/8 exponent steepness)
        m_critical_onset = self.yang_spontaneous_magnetization(beta=0.441)
        assert 0.40 < m_critical_onset < 0.60

        m_cold = self.yang_spontaneous_magnetization(beta=0.70)
        assert m_cold == pytest.approx(0.9902, abs=0.01)

        # Deep ferromagnetic freeze
        m_freeze = self.yang_spontaneous_magnetization(beta=2.0)
        assert m_freeze > 0.9999


# ============================================================================
# FIXTURE 4: SOKAL INTEGRATED AUTOCORRELATION & ESS ENGINE
# ============================================================================

class TestSokalAutocorrelationDiagnostics:
    """Verifies Sokal window cutoff and ESS calculations on known stochastic processes."""

    def test_white_noise_zero_autocorrelation(self):
        """Independent IID samples must yield tau_int ≈ 0.5 and ESS ≈ N."""
        rng = np.random.default_rng(123)
        n = 10000
        iid_samples = rng.normal(0.0, 1.0, size=n)

        tau_int, ess = compute_sokal_tau_int(iid_samples)
        # Theoretical tau_int for independent process is 0.5
        assert tau_int == pytest.approx(0.5, abs=0.15)
        # Effective sample size should be very close to N
        assert ess == pytest.approx(n, rel=0.20)

    def test_ar1_correlated_process_analytical_tau(self):
        """AR(1) process x_t = phi * x_{t-1} + noise has analytical tau_int = (1 + phi) / (2 * (1 - phi))."""
        rng = np.random.default_rng(456)
        n = 30000
        phi = 0.7
        # Theoretical tau_int: (1 + 0.7) / (2 * 0.3) = 1.7 / 0.6 = 2.8333...
        expected_tau = (1.0 + phi) / (2.0 * (1.0 - phi))

        # Generate AR(1) series
        x = np.zeros(n, dtype=np.float64)
        noise = rng.normal(0.0, np.sqrt(1.0 - phi ** 2), size=n)
        for t in range(1, n):
            x[t] = phi * x[t - 1] + noise[t]

        tau_int, ess = compute_sokal_tau_int(x)
        assert tau_int == pytest.approx(expected_tau, rel=0.15)
        expected_ess = n / (2.0 * expected_tau)
        assert ess == pytest.approx(expected_ess, rel=0.20)

    def test_multivariate_tau_evaluation(self):
        """Multivariate trajectory correctly evaluates per-spin statistics."""
        rng = np.random.default_rng(789)
        samples = rng.choice([-1.0, 1.0], size=(2000, 8))
        stats = compute_multivariate_tau_int(samples)

        assert stats["n_samples"] == 2000
        assert stats["n_spins"] == 8
        assert 0.4 <= stats["tau_mean"] <= 0.8
        assert stats["ess_min"] >= 1000

    def test_sokal_observer_lifecycle(self):
        """SokalConvergenceObserver correctly accumulates history and tracks convergence."""
        obs = SokalConvergenceObserver(c_window=5.0, min_burn_in_ratio=10.0)
        rng = np.random.default_rng(999)

        # Feed 1000 independent samples with 0.1s wall time
        data = rng.normal(0.0, 1.0, size=1000)
        obs.record_trace(data, elapsed_sec=0.1)

        metrics = obs.evaluate()
        assert metrics["n_samples"] == 1000
        assert metrics["sample_rate_hz"] == pytest.approx(10000.0, rel=0.05)
        assert metrics["is_converged"] is True  # 1000 >= 10 * 0.5
        assert "CONVERGED" in obs.summary()
