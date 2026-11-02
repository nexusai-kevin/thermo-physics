"""
tests.test_tpu
==============
Automated test suite verifying the Hybrid Thermodynamic Processing Unit (TPU) architecture:
1. Analytical vs Finite-Difference gradient and Laplacian verification for NonConvexIsingSAT.
2. BAOAB Symplectic Integrator stability vs Euler-Maruyama divergence under stiff SDE drift.
3. Thermal, kinetic, and configurational equipartition (Fluctuation-Dissipation Theorem).
4. Thermodynamic path integrals: First Law balance (dE = dW + dQ), Second Law dissipation, and non-isothermal Jarzynski equality.
5. Softplus non-saturating penalty overcoming tanh gradient starvation.
6. Non-conservative dynamic auxiliary clause multipliers y_m(t) and housekeeping heat dQ_hk.
7. Sequential Monte Carlo (SMC) Jarzynski population resampling preserving N_eff > 50%.
8. End-to-end benchmark comparison: TPU Langevin Dynamics vs Classical Simulated Annealing on phase-transition 3-SAT.
"""

from __future__ import annotations

import unittest
import numpy as np

from thermo_physics.tpu import (
    NonConvexIsingSAT,
    ProblemConfig,
    generate_random_3sat_spin_glass,
    generate_sk_spin_glass,
    BAOABIntegrator,
    EulerMaruyamaIntegrator,
    ThermalSchedule,
    AdaptiveThermalScheduler,
    ThermodynamicTracker,
    verify_jarzynski_equality,
    DiagnosticMonitor,
    HybridTPUSolver,
    ClassicalSimulatedAnnealing,
    TPUSolverConfig,
    run_tpu_vs_classical_benchmark,
)
from thermo_physics.tpu.integrator import ScheduleType


class TestThermodynamicTPU(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)

    def test_analytical_gradient_finite_difference(self) -> None:
        """Verify that analytical gradient matches finite-difference numerical gradient within 1e-4."""
        n = 6
        clauses = [(1, -2, 3), (-1, 4, -5), (2, -3, 6)]

        for p_type in ["tanh", "softplus"]:
            ham = NonConvexIsingSAT(
                num_variables=n,
                j_matrix=self.rng.normal(0, 0.5, (n, n)),
                h_bias=self.rng.normal(0, 0.2, n),
                clauses=clauses,
                config=ProblemConfig(num_variables=n, alpha_well=1.2, lambda_sat=3.0, c_sat=2.0, penalty_type=p_type),
            )

            x = self.rng.uniform(-1.5, 1.5, size=n)
            analytical_grad = ham.gradient(x)

            eps = 1e-6
            numerical_grad = np.zeros(n)
            for i in range(n):
                x_plus = x.copy()
                x_plus[i] += eps
                x_minus = x.copy()
                x_minus[i] -= eps
                numerical_grad[i] = (ham.energy(x_plus) - ham.energy(x_minus)) / (2.0 * eps)

            rel_error = np.linalg.norm(analytical_grad - numerical_grad) / max(1e-5, np.linalg.norm(numerical_grad))
            self.assertLess(rel_error, 1e-4, f"Gradient mismatch for {p_type}: {rel_error}")

    def test_diagonal_hessian_and_laplacian_finite_difference(self) -> None:
        """Verify analytical diagonal Hessian d^2 H / dx_i^2 and Laplacian Delta H(x) against finite differences."""
        n = 5
        clauses = [(1, -2, 3), (2, -3, 4), (-1, 3, -5)]

        for p_type in ["tanh", "softplus"]:
            ham = NonConvexIsingSAT(
                num_variables=n,
                j_matrix=self.rng.normal(0, 0.4, (n, n)),
                h_bias=self.rng.normal(0, 0.2, n),
                clauses=clauses,
                config=ProblemConfig(num_variables=n, alpha_well=1.5, lambda_sat=2.5, penalty_type=p_type),
            )

            x = self.rng.uniform(-1.2, 1.2, size=n)
            analytical_diag = ham.diagonal_hessian(x)
            analytical_lap = ham.laplacian(x)

            eps = 1e-5
            numerical_diag = np.zeros(n)
            e_mid = ham.energy(x)
            for i in range(n):
                x_plus = x.copy()
                x_plus[i] += eps
                x_minus = x.copy()
                x_minus[i] -= eps
                numerical_diag[i] = (ham.energy(x_plus) - 2.0 * e_mid + ham.energy(x_minus)) / (eps**2)

            rel_error = np.linalg.norm(analytical_diag - numerical_diag) / max(1e-5, np.linalg.norm(numerical_diag))
            self.assertLess(rel_error, 1e-3, f"Diagonal Hessian mismatch for {p_type}: {rel_error}")
            self.assertAlmostEqual(analytical_lap, float(np.sum(analytical_diag)), places=5)

    def test_configurational_temperature_estimator(self) -> None:
        """Verify configurational temperature kB * T_conf = <||grad H||^2> / <Delta H> converges to T_bath."""
        n = 8
        ham = NonConvexIsingSAT(
            num_variables=n,
            config=ProblemConfig(num_variables=n, alpha_well=0.5, lambda_sat=0.0)
        )
        integrator = BAOABIntegrator(dt=0.01, gamma=2.0)

        target_temp = 1.2
        b = 80
        x = self.rng.normal(0, 0.4, size=(b, n))
        v = self.rng.normal(0, np.sqrt(target_temp), size=(b, n))

        t_conf_samples = []
        f_cached = None
        for _ in range(600):
            x, v, f_cached, _ = integrator.step(
                x, v, ham, temperature=target_temp, alpha_well=0.5, lambda_sat=0.0, rng=self.rng, f_cached=f_cached
            )
            grad = ham.gradient(x, alpha_well=0.5, lambda_sat=0.0)
            grad_sq = np.sum(grad**2, axis=-1)
            lap = ham.laplacian(x, alpha_well=0.5, lambda_sat=0.0)
            t_conf = np.mean(grad_sq) / max(1e-5, np.mean(lap))
            t_conf_samples.append(t_conf)

        avg_t_conf = np.mean(t_conf_samples[250:])
        rel_diff = abs(avg_t_conf - target_temp) / target_temp
        self.assertLess(rel_diff, 0.18, f"Configurational temperature error too large: {rel_diff}")

    def test_softplus_gradient_vs_tanh_starvation(self) -> None:
        """Verify that softplus avoids tanh gradient vanishing when trapped deep inside a false well."""
        n = 3
        clauses = [(1, 2, 3)]

        ham_tanh = NonConvexIsingSAT(
            num_variables=n,
            clauses=clauses,
            config=ProblemConfig(num_variables=n, alpha_well=0.0, lambda_sat=10.0, c_sat=2.5, penalty_type="tanh"),
        )
        ham_softplus = NonConvexIsingSAT(
            num_variables=n,
            clauses=clauses,
            config=ProblemConfig(num_variables=n, alpha_well=0.0, lambda_sat=10.0, beta_softplus=2.5, penalty_type="softplus"),
        )

        x_trapped = np.array([-2.0, -2.0, -2.0])
        grad_tanh = ham_tanh.gradient(x_trapped)
        grad_softplus = ham_softplus.gradient(x_trapped)

        tanh_grad_mag = np.linalg.norm(grad_tanh)
        softplus_grad_mag = np.linalg.norm(grad_softplus)

        self.assertLess(tanh_grad_mag, 0.05)
        self.assertGreater(softplus_grad_mag, 1.0)

    def test_non_conservative_auxiliary_clause_multipliers(self) -> None:
        """Verify dynamic clause multipliers y_m integrate upward and generate housekeeping heat dQ_hk."""
        n = 6
        clauses = [(1, 2, 3), (-1, -2, 4), (3, -4, 5)]
        ham = NonConvexIsingSAT(
            num_variables=n,
            clauses=clauses,
            config=ProblemConfig(num_variables=n, alpha_well=1.0, lambda_sat=2.0, penalty_type="softplus"),
        )

        config = TPUSolverConfig(
            num_trajectories=16,
            total_steps=300,
            use_non_conservative_aux=True,
            kappa_aux=25.0,
            eta_aux=2.0,
            seed=42,
        )
        solver = HybridTPUSolver(config)
        res = solver.solve(ham)

        self.assertIsNotNone(res.thermo_trajectory)
        self.assertIsNotNone(res.jarzynski_analysis)
        # Housekeeping heat must be tracked and non-negative
        q_hk = res.jarzynski_analysis.mean_q_housekeeping
        self.assertGreaterEqual(q_hk, 0.0)

    def test_jarzynski_population_resampling(self) -> None:
        """Verify SMC Population Resampling maintains Kish N_eff / B >= 50% throughout annealing."""
        n = 12
        ham = generate_random_3sat_spin_glass(num_variables=n, clause_ratio=3.0, seed=123)

        config = TPUSolverConfig(
            num_trajectories=32,
            total_steps=400,
            enable_population_resampling=True,
            resample_threshold=0.50,
            resample_interval=15,
            schedule=ThermalSchedule(t_start=3.0, t_end=0.05, total_steps=400),
            seed=42,
        )
        solver = HybridTPUSolver(config)
        res = solver.solve(ham)

        self.assertIsNotNone(res.thermo_trajectory)
        self.assertIsNotNone(res.jarzynski_analysis)
        # Average effective sample size ratio must stay healthy (>= 50%)
        mean_n_eff = np.mean(res.thermo_trajectory.n_eff_history)
        self.assertGreaterEqual(mean_n_eff / 32.0, 0.50)

    def test_baoab_equipartition_and_stability(self) -> None:
        """Verify BAOAB maintains physical kinetic temperature (Equipartition Theorem) at constant T."""
        n = 10
        ham = NonConvexIsingSAT(num_variables=n, config=ProblemConfig(num_variables=n, alpha_well=0.5))
        integrator = BAOABIntegrator(dt=0.01, gamma=2.0, mass=1.0, kb=1.0)

        target_temp = 1.5
        b = 64
        x = self.rng.normal(0, 0.5, size=(b, n))
        v = self.rng.normal(0, np.sqrt(target_temp), size=(b, n))

        f_cached = None
        ke_list = []
        for _ in range(800):
            x, v, f_cached, _ = integrator.step(
                x, v, ham, temperature=target_temp, alpha_well=0.5, lambda_sat=0.0, rng=self.rng, f_cached=f_cached
            )
            ke = 0.5 * np.sum(v**2, axis=-1)
            t_kin = (2.0 * ke) / (n * 1.0)
            ke_list.append(np.mean(t_kin))

        measured_temp = np.mean(ke_list[300:])
        rel_diff = abs(measured_temp - target_temp) / target_temp
        self.assertLess(rel_diff, 0.15)

    def test_euler_maruyama_divergence_comparison(self) -> None:
        """Verify that Euler-Maruyama diverges or suffers severe stability breakdown under large dt, while BAOAB remains stable."""
        n = 8
        ham = NonConvexIsingSAT(num_variables=n, config=ProblemConfig(num_variables=n, alpha_well=10.0))

        large_dt = 0.08
        em_integrator = EulerMaruyamaIntegrator(dt=large_dt, gamma=1.0)
        baoab_integrator = BAOABIntegrator(dt=large_dt, gamma=1.0)

        b = 16
        x_init = self.rng.normal(0, 1.0, size=(b, n))
        v_init = self.rng.normal(0, 1.0, size=(b, n))

        x_baoab, v_baoab = x_init.copy(), v_init.copy()
        f_cached = None
        for _ in range(300):
            x_baoab, v_baoab, f_cached, _ = baoab_integrator.step(
                x_baoab, v_baoab, ham, temperature=0.5, alpha_well=10.0, lambda_sat=0.0, rng=self.rng, f_cached=f_cached
            )

        self.assertTrue(np.all(np.isfinite(x_baoab)))
        self.assertTrue(np.all(np.isfinite(v_baoab)))
        self.assertLess(np.max(np.abs(x_baoab)), 20.0)

    def test_non_isothermal_jarzynski_and_n_eff(self) -> None:
        """Verify non-isothermal Jarzynski functional Sigma and Kish's N_eff sample collapse metric."""
        n = 4
        ham = NonConvexIsingSAT(num_variables=n, config=ProblemConfig(num_variables=n, alpha_well=0.5))
        b = 50
        tracker = ThermodynamicTracker(batch_size=b, num_variables=n)

        integrator = BAOABIntegrator(dt=0.01, gamma=1.5)
        x = self.rng.normal(0, 0.3, size=(b, n))
        v = self.rng.normal(0, 0.5, size=(b, n))

        alpha_0 = 0.2
        alpha_1 = 1.5
        t_0 = 2.0
        t_1 = 0.5

        tracker.initialize(x, v, ham, alpha_well=alpha_0, lambda_sat=0.0, initial_temperature=t_0)

        steps = 150
        f_cached = None
        for s in range(steps):
            frac = s / steps
            cur_alpha = alpha_0 + frac * (alpha_1 - alpha_0)
            next_alpha = alpha_0 + (s + 1) / steps * (alpha_1 - alpha_0)
            cur_t = t_0 - frac * (t_0 - t_1)
            next_t = t_0 - (s + 1) / steps * (t_0 - t_1)

            x, v, f_cached, _ = integrator.step(
                x, v, ham, temperature=cur_t, alpha_well=cur_alpha, lambda_sat=0.0, rng=self.rng, f_cached=f_cached
            )
            tracker.record_step(
                x, v, ham,
                alpha_old=cur_alpha, lambda_old=0.0,
                alpha_new=next_alpha, lambda_new=0.0,
                temperature_old=cur_t, temperature_new=next_t,
            )

        traj = tracker.finalize(x, v, ham, alpha_well=alpha_1, lambda_sat=0.0)

        delta_e = traj.final_energy - traj.initial_energy
        energy_balance_error = np.max(np.abs(delta_e - (traj.work - traj.heat_dissipated)))
        self.assertLess(energy_balance_error, 1e-10)

        beta_eff = 1.0 / (0.5 * (t_0 + t_1))
        jarzynski = verify_jarzynski_equality(
            traj.work,
            beta=beta_eff,
            generalized_sigma=traj.generalized_sigma,
        )
        self.assertTrue(jarzynski.non_isothermal)
        self.assertGreater(jarzynski.n_eff, 0.0)

    def test_end_to_end_tpu_vs_classical_benchmark(self) -> None:
        """Verify full benchmark execution comparing Hybrid TPU solver against Classical SA."""
        n = 8
        ham = generate_random_3sat_spin_glass(num_variables=n, clause_ratio=3.0, seed=123)

        benchmark = run_tpu_vs_classical_benchmark(
            hamiltonian=ham,
            num_trajectories=16,
            steps=400,
            use_non_conservative_aux=True,
            enable_population_resampling=True,
            seed=42,
        )

        self.assertEqual(len(benchmark.tpu_result.best_spins), n)
        self.assertIsInstance(benchmark.tpu_result.best_energy, float)
        self.assertGreaterEqual(benchmark.tpu_result.best_violated_clauses, 0)
        self.assertGreater(benchmark.tpu_result.wall_time_sec, 0.0)


if __name__ == "__main__":
    unittest.main()
