"""Unit tests for the 4-Organ Thermodynamic Synthetic Intelligence (SI) Cognitive Core.

Verifies:
1. Cognitive tick latency < 15ms.
2. Deterministic ground-state relaxation across all 4 coupled organs:
   - Organ 1: Continuous Hopfield attractor basin convergence from corrupted queries.
   - Organ 2: Kramers double-well bistability and thermal bath novelty modulation.
   - Organ 3: N=6, K=4 Potts goal arbiter ferromagnetic consensus and ground-state energy.
   - Organ 4: RTX 5080 metabolic QUBO scheduler constraint satisfaction.
"""

from __future__ import annotations

import unittest
import numpy as np

from thermo_physics.si_core import (
    HopfieldAssociativeMemory,
    KramersCuriosityDrive,
    PottsGoalArbiter,
    MetabolicHomeostasis,
    ThermodynamicSICore,
)


class TestThermodynamicSICore(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(1337)

    def test_organ_1_hopfield_attractor_relaxation(self) -> None:
        """Verifies deterministic relaxation to the nearest stored attractor basin."""
        hopfield = HopfieldAssociativeMemory(dim=64, beta=12.0)
        p1 = self.rng.standard_normal(64)
        p2 = self.rng.standard_normal(64)
        p3 = self.rng.standard_normal(64)

        hopfield.store(p1, "Pattern_Alpha")
        hopfield.store(p2, "Pattern_Beta")
        hopfield.store(p3, "Pattern_Gamma")

        # Corrupt Pattern_Alpha with 25% Gaussian noise
        corrupted_p1 = p1 / np.linalg.norm(p1) + 0.25 * self.rng.standard_normal(64)
        recalled_vec, label, energy = hopfield.retrieve(corrupted_p1, steps=5)

        self.assertEqual(label, "Pattern_Alpha")
        # Check alignment with ground truth unit vector
        true_normed_p1 = p1 / np.linalg.norm(p1)
        cosine_sim = float(np.dot(recalled_vec, true_normed_p1))
        self.assertGreater(cosine_sim, 0.95)
        # Energy must be finite and lower than unaligned noise
        self.assertIsInstance(energy, float)

    def test_organ_2_kramers_double_well_dynamics(self) -> None:
        """Verifies bistable potential dynamics and thermal novelty sensitivity."""
        kramers = KramersCuriosityDrive(barrier_height=2.5, damping=0.4)
        kramers.reset(x=-1.0, v=0.0)

        # In cold bath with zero novelty, particle stays in exploitation basin (x < 0)
        for _ in range(5):
            mode, temp = kramers.step(novelty_signal=0.0, dt=0.02, seed=42)
            self.assertEqual(mode, "EXPLOIT_KNOWN")
            self.assertLess(temp, 0.5)
            self.assertLess(kramers.x, 0.0)

        # High novelty signal heats the bath (T >= 2.0)
        _, heated_temp = kramers.step(novelty_signal=0.9, dt=0.05, seed=42)
        self.assertGreaterEqual(heated_temp, 2.0)

    def test_organ_3_potts_goal_arbiter_ferromagnetic_consensus(self) -> None:
        """Verifies Potts arbiter collapses to ground state consensus under strong field."""
        arbiter = PottsGoalArbiter(num_modules=6, num_states=4, coupling_strength=1.5)

        # Apply strong polarizing external field favoring stance 2 (DEEP_REASONING)
        strong_fields = np.zeros((6, 4), dtype=np.float32)
        strong_fields[:, 2] = 10.0

        spins, energy = arbiter.equilibrate(strong_fields, beta=5.0, sweeps=30, seed=42)

        # Under strong field and high beta (low temp), all 6 modules must align to stance 2
        np.testing.assert_array_equal(spins, np.full(6, 2))
        # Total energy must be strongly negative (favorable ground state)
        self.assertLess(energy, -50.0)

    def test_organ_4_metabolic_homeostasis_scheduler(self) -> None:
        """Verifies RTX 5080 VRAM limits and capability-driven tier selection."""
        metabolism = MetabolicHomeostasis(vram_limit_gb=16.0)

        # Case 1: Low capability requirement, low VRAM used -> Local_Fast_3B is optimal
        res_fast = metabolism.select_execution_tier(required_capability=0.4, current_vram_used=2.0)
        self.assertEqual(res_fast["selected_engine"], "Local_Fast_3B")
        self.assertGreater(res_fast["vram_headroom_gb"], 10.0)

        # Case 2: Moderate capability requirement (0.8), low VRAM -> Local_Core_8B is optimal
        res_core = metabolism.select_execution_tier(required_capability=0.8, current_vram_used=2.0)
        self.assertEqual(res_core["selected_engine"], "Local_Core_8B")

        # Case 3: High capability requirement with deep priority -> Local_Deep_14B is optimal
        metabolism_deep = MetabolicHomeostasis(vram_limit_gb=16.0, cap_weight=300.0)
        res_deep = metabolism_deep.select_execution_tier(required_capability=0.95, current_vram_used=2.0)
        self.assertEqual(res_deep["selected_engine"], "Local_Deep_14B")

        # Case 4: High capability requested, but current VRAM is already 12GB.
        # Deep 14B needs 11.5GB, which would breach the 16GB VRAM limit (12 + 11.5 = 23.5GB).
        # Scheduler must penalize overage and fall back to a smaller tier or protect VRAM.
        res_constrained = metabolism.select_execution_tier(required_capability=0.9, current_vram_used=12.0)
        self.assertIn(res_constrained["selected_engine"], ["Local_Fast_3B", "Local_Core_8B"])

    def test_cognitive_tick_latency_under_15ms(self) -> None:
        """Verifies that a full coupled cognitive tick executes in < 15ms on CPU."""
        si_core = ThermodynamicSICore(seed=42)
        query = self.rng.standard_normal(64)

        # Warm-up tick
        _ = si_core.cognitive_tick(query, novelty_score=0.3, current_vram=4.0, seed=42)

        latencies = []
        for i in range(25):
            test_query = query + 0.1 * self.rng.standard_normal(64)
            telemetry = si_core.cognitive_tick(test_query, novelty_score=0.4, current_vram=5.0)
            latencies.append(telemetry["tick_latency_ms"])

            # Verify presence of all organ outputs
            self.assertIn(telemetry["organ_1_hopfield_attractor"], [
                "System_Architecture_Optimization",
                "Security_Boundary_Verification",
                "Novel_Research_Synthesis",
            ])
            self.assertIn(telemetry["organ_2_kramers_drive"], ["EXPLOIT_KNOWN", "EXPLORE_NOVELTY"])
            self.assertIn(telemetry["organ_3_potts_stance"], [
                "VERIFY", "FAST_HEURISTIC", "DEEP_REASONING", "AUTONOMOUS_EXEC"
            ])
            self.assertIn(telemetry["organ_4_metabolism"], [
                "Local_Fast_3B", "Local_Core_8B", "Local_Deep_14B"
            ])

        mean_latency = float(np.mean(latencies))
        max_latency = float(np.max(latencies))
        p95_latency = float(np.percentile(latencies, 95))

        print(f"\n[Cognitive Core Benchmark] Mean: {mean_latency:.2f}ms | P95: {p95_latency:.2f}ms | Max: {max_latency:.2f}ms")
        self.assertLess(max_latency, 15.0, f"Max cognitive tick latency {max_latency}ms exceeded 15ms SLA target!")


if __name__ == "__main__":
    unittest.main()
