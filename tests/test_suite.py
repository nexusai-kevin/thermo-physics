"""Automated verification tests for the Thermodynamic Computing Reproducibility Suite."""

from __future__ import annotations

import unittest
import numpy as np

from thermo_repro.backend import (
    Block,
    SpinNode,
    compute_energy,
    exact_boltzmann_distribution,
)
from thermo_repro.coloring import (
    build_2d_lattice_topology,
    build_vhl_bipartite_topology,
    verify_independent_blocks,
)
from thermo_repro.metrics import (
    compare_block_vs_sequential_gibbs,
    compute_integrated_autocorrelation_time,
    profile_hardware_energy_efficiency,
    verify_boltzmann_equilibrium,
)
from thermo_repro.project_a_classifier import run_project_a_benchmark
from thermo_repro.project_b_dtm import run_project_b_benchmark


class TestThermodynamicSuite(unittest.TestCase):
    def test_independent_block_coloring_validation(self) -> None:
        nodes, edges, blocks = build_2d_lattice_topology(4, 4, periodic=False)
        self.assertEqual(len(nodes), 16)
        self.assertEqual(len(blocks), 2)
        self.assertTrue(verify_independent_blocks(blocks, edges))

        # Putting all nodes in a single block must fail independent-set validation
        with self.assertRaises(ValueError):
            verify_independent_blocks([Block(nodes)], edges)

    def test_vhl_bipartite_topology(self) -> None:
        topo = build_vhl_bipartite_topology(n_visible=16, n_hidden=8, n_labels=4)
        self.assertEqual(len(topo["all_nodes"]), 28)
        self.assertEqual(len(topo["all_edges"]), 16 * 8 + 8 * 4)
        self.assertTrue(verify_independent_blocks(topo["free_all_blocks"], topo["all_edges"]))

    def test_exact_boltzmann_convergence(self) -> None:
        nodes = [SpinNode(f"s_{i}") for i in range(6)]
        edges = [(nodes[i], nodes[i + 1]) for i in range(5)]
        biases = np.array([0.1, -0.2, 0.0, 0.15, -0.1, 0.05], dtype=np.float32)
        weights = np.array([0.4, -0.3, 0.5, 0.25, 0.35], dtype=np.float32)
        blocks = [Block(nodes[::2]), Block(nodes[1::2])]

        res = verify_boltzmann_equilibrium(
            nodes,
            edges,
            biases,
            weights,
            beta=0.9,
            free_blocks=blocks,
            n_warmup=250,
            n_samples=6000,
            steps_per_sample=2,
            seed=123,
        )
        self.assertLess(res["total_variation_distance"], 0.05)
        self.assertLess(res["energy_abs_error"], 0.08)
        self.assertGreater(res["effective_sample_size"], 500.0)

    def test_autocorrelation_and_hardware_profile(self) -> None:
        rng = np.random.default_rng(0)
        # Construct AR(1) process with known autocorrelation decay
        phi = 0.6
        x = np.zeros(2000, dtype=np.float64)
        for t in range(1, len(x)):
            x[t] = phi * x[t - 1] + rng.normal()
        stats = compute_integrated_autocorrelation_time(x)
        # Theoretical tau_int for AR(1) is 0.5 + phi / (1 - phi) = 0.5 + 1.5 = 2.0
        self.assertAlmostEqual(stats["tau_int"], 2.0, delta=0.5)

        hw = profile_hardware_energy_efficiency(
            n_nodes=96,
            n_edges=2048,
            n_blocks=2,
            total_sweeps=1100,
            measured_wall_time_sec=0.05,
        )
        self.assertGreater(hw["tsu_vs_classical_ratios"]["energy_reduction_x"], 1e6)

    def test_project_a_associative_classifier(self) -> None:
        res = run_project_a_benchmark(n_visible=16, n_hidden=16, n_trials_per_class=6, seed=42)
        self.assertGreaterEqual(res["classification_metrics"]["clean_accuracy"], 0.75)
        self.assertGreaterEqual(res["classification_metrics"]["noisy_accuracy"], 0.75)
        self.assertGreater(
            res["associative_memory_metrics"]["occluded_bit_recovery_accuracy"],
            0.70,
        )

    def test_project_b_dtm_denoising(self) -> None:
        res = run_project_b_benchmark(height=10, width=10, flip_prob=0.22, seed=42)
        dtm = res["dtm_checkerboard_block_gibbs"]
        self.assertGreater(dtm["denoised_accuracy"], res["initial_noisy_accuracy"])
        self.assertLess(dtm["final_energy"], dtm["initial_energy"])
        self.assertEqual(dtm["n_parallel_blocks"], 2)


if __name__ == "__main__":
    unittest.main()
