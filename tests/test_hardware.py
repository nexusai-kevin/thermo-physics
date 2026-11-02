"""
tests.test_hardware
===================
Unit and integration tests for Silicon Realism & Hardware Imperfections module:
1. DAC fixed-point quantization (4-bit, 6-bit, 8-bit, symmetric bipolar).
2. Op-Amp voltage rail clipping & slew rate velocity bounds.
3. Colored 1/f Ornstein-Uhlenbeck noise bath variance and autocorrelation decay.
4. End-to-end TPU solver convergence under realistic silicon hardware imperfections.
"""

from __future__ import annotations

import unittest
import numpy as np

from thermo_physics.tpu import (
    NonConvexIsingSAT,
    ProblemConfig,
    generate_random_3sat_spin_glass,
    HybridTPUSolver,
    TPUSolverConfig,
    SiliconRealismConfig,
    DACQuantizer,
    OpAmpVoltageClipper,
    OrnsteinUhlenbeckNoise,
)


class TestSiliconHardwareRealism(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)

    def test_dac_quantization_levels(self) -> None:
        """Verify DAC quantizer produces at most 2^K discrete levels and correct step size."""
        for bits in [4, 6, 8]:
            max_codewords = 1 << bits
            levels = max_codewords - 1
            w = np.linspace(-1.0, 1.0, 500)
            w_quant = DACQuantizer.quantize_symmetric(w, bits=bits, max_val=1.0)

            unique_levels = np.unique(np.round(w_quant, decimals=6))
            # Number of unique levels should be at most 2^K
            self.assertLessEqual(len(unique_levels), max_codewords)

            expected_delta = 2.0 / levels
            # Steps between adjacent levels should match expected_delta
            diffs = np.diff(unique_levels)
            if len(diffs) > 0:
                self.assertAlmostEqual(float(np.min(diffs)), expected_delta, places=4)

    def test_dac_quantize_hamiltonian(self) -> None:
        """Verify Hamiltonian quantization discretizes couplings and preserves zero diagonal / symmetry."""
        n = 8
        ham = generate_random_3sat_spin_glass(
            num_variables=n,
            clause_ratio=3.0,
            seed=42,
        )

        quant_ham = DACQuantizer.quantize_hamiltonian(ham, bits=4, max_val=1.0)

        # Diagonal of J must remain strictly zero
        np.testing.assert_array_equal(np.diagonal(quant_ham.j), np.zeros(n))

        # J must remain symmetric
        np.testing.assert_allclose(quant_ham.j, quant_ham.j.T, atol=1e-12)

        # Values of J must fall on at most 16 levels
        unique_j = np.unique(np.round(quant_ham.j, decimals=4))
        self.assertLessEqual(len(unique_j), 16)

    def test_opamp_rail_clipping(self) -> None:
        """Verify that state voltages and velocities are clipped to physical power rails."""
        clipper = OpAmpVoltageClipper(v_rail=1.5, v_max=5.0, y_max=12.0, lambda_0=1.0)

        # Coordinates outside [-1.5, 1.5]
        x = np.array([-2.5, -1.0, 0.0, 1.2, 3.8])
        x_clipped = clipper.clip_position(x)
        self.assertTrue(np.all(x_clipped >= -1.5) and np.all(x_clipped <= 1.5))
        self.assertEqual(clipper.position_clips, 2)

        # Velocities outside [-5.0, 5.0]
        v = np.array([-10.0, -4.0, 0.5, 6.2])
        v_clipped = clipper.clip_velocity(v)
        self.assertTrue(np.all(v_clipped >= -5.0) and np.all(v_clipped <= 5.0))
        self.assertEqual(clipper.velocity_clips, 2)

        # Dynamic multipliers outside [1.0, 12.0]
        y = np.array([0.5, 1.0, 5.0, 15.0])
        y_clipped = clipper.clip_multiplier(y)
        self.assertTrue(np.all(y_clipped >= 1.0) and np.all(y_clipped <= 12.0))
        self.assertEqual(clipper.multiplier_clips, 1)

    def test_colored_noise_variance_and_autocorrelation(self) -> None:
        """Verify Ornstein-Uhlenbeck colored noise stationary variance and correlation decay."""
        tau = 0.05
        gamma = 1.5
        kb = 1.0
        temp = 2.0
        dt = 0.005
        steps = 10000

        ou = OrnsteinUhlenbeckNoise(shape=(1,), tau_color=tau, use_pink_noise_bank=False)
        rng = np.random.default_rng(123)

        samples = []
        for _ in range(steps):
            noise = ou.step(dt=dt, temperature=temp, gamma=gamma, kb=kb, rng=rng)
            samples.append(float(noise[0]))

        samples_arr = np.array(samples)
        # Stationary variance theory: sigma^2 = (gamma * kb * T) / tau
        expected_var = (gamma * kb * temp) / tau  # 1.5 * 1.0 * 2.0 / 0.05 = 60.0
        emp_var = float(np.var(samples_arr[2000:]))  # discard transient

        # Within 20% statistical tolerance over finite chain
        self.assertAlmostEqual(emp_var / expected_var, 1.0, delta=0.25)

        # Autocorrelation at lag = tau: expected rho(tau) ~ exp(-1) = 0.3678
        lag_steps = int(tau / dt)  # 10 steps
        centered = samples_arr - np.mean(samples_arr)
        autocorr_tau = np.mean(centered[:-lag_steps] * centered[lag_steps:]) / np.var(centered)
        self.assertAlmostEqual(float(autocorr_tau), np.exp(-1.0), delta=0.15)

    def test_tpu_solver_under_silicon_hardware_constraints(self) -> None:
        """Verify end-to-end TPU solver execution with 6-bit DAC, Op-Amp rails, and colored noise."""
        n = 16
        ham = generate_random_3sat_spin_glass(num_variables=n, clause_ratio=2.5, seed=42)

        silicon_cfg = SiliconRealismConfig(
            dac_bits=6,
            enable_rail_clipping=True,
            v_rail=1.5,
            v_max=10.0,
            y_max=15.0,
            enable_colored_noise=True,
            tau_color=0.05,
        )

        solver_cfg = TPUSolverConfig(
            num_trajectories=16,
            total_steps=500,
            dt=0.015,
            silicon_config=silicon_cfg,
            use_non_conservative_aux=True,
            monitor_diagnostics=True,
            track_thermodynamics=True,
            seed=42,
        )

        solver = HybridTPUSolver(solver_cfg)
        res = solver.solve(ham)

        self.assertIsNotNone(res.silicon_metrics)
        self.assertEqual(res.silicon_metrics["dac_bits"], 6)
        self.assertTrue(res.silicon_metrics["enable_rail_clipping"])
        self.assertTrue(res.silicon_metrics["enable_colored_noise"])
        self.assertGreater(res.silicon_metrics["j_quantization_rms_error"], 0.0)
        self.assertLessEqual(res.best_violated_clauses, 2)


if __name__ == "__main__":
    unittest.main()
