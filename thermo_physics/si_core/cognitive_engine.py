"""Thermodynamic Synthetic Intelligence (SI) Cognitive Engine.

Coupled 4-Organ Architecture:
- Organ 1: Continuous Hopfield Associative Memory (128GB DDR5 episodic capacity)
- Organ 2: Kramers Double-Well Curiosity & Drive (Langevin thermal bath modulation)
- Organ 3: N=6, K=4 Categorical Potts Goal Arbiter (Categorical Gibbs consensus)
- Organ 4: QUBO Metabolic Homeostasis Scheduler (RTX 5080 16GB VRAM budget governor)
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


# ============================================================================
# ORGAN 1: Modern Continuous Hopfield Network (Associative Memory)
# Runs in 128GB System RAM for massive episodic capacity
# ============================================================================
class HopfieldAssociativeMemory:
    """Continuous modern Hopfield network with exponential storage capacity."""

    def __init__(self, dim: int = 64, beta: float = 8.0) -> None:
        self.dim = dim
        self.beta = beta
        # Stored memory patterns X: shape (M, dim)
        self.patterns = np.empty((0, dim), dtype=np.float32)
        self.labels: List[str] = []

    def store(self, state_vec: np.ndarray, label: str) -> None:
        """Stores a normalized exemplar vector in episodic memory."""
        vec = np.asarray(state_vec, dtype=np.float32).ravel()
        if vec.shape[0] != self.dim:
            raise ValueError(f"State vector dimension {vec.shape[0]} does not match memory dim {self.dim}")
        norm = np.linalg.norm(vec)
        norm_vec = vec / (norm + 1e-8)
        self.patterns = np.vstack([self.patterns, norm_vec.reshape(1, -1)])
        self.labels.append(label)

    def energy(self, state_vec: np.ndarray) -> float:
        """Calculates Ramsauer-Hopfield energy E(xi)."""
        if len(self.patterns) == 0:
            return 0.0
        xi = state_vec / (np.linalg.norm(state_vec) + 1e-8)
        projections = self.patterns @ xi
        # Numerical stability via log-sum-exp
        max_proj = np.max(projections)
        lse = max_proj + np.log(np.sum(np.exp(self.beta * (projections - max_proj)))) / self.beta
        return float(-lse + 0.5 * np.dot(xi, xi))

    def retrieve(self, query_vec: np.ndarray, steps: int = 5) -> Tuple[np.ndarray, str, float]:
        """Relaxes corrupted or partial input to the nearest fixed-point attractor basin."""
        if len(self.patterns) == 0:
            return query_vec, "uninitialized", 0.0

        xi = np.asarray(query_vec, dtype=np.float32).ravel()
        xi = xi / (np.linalg.norm(xi) + 1e-8)

        for _ in range(steps):
            # Softmax attention over stored patterns (Modern Hopfield update)
            scores = self.beta * (self.patterns @ xi)
            scores -= np.max(scores)  # numerical stability
            weights = np.exp(scores) / (np.sum(np.exp(scores)) + 1e-12)
            xi = self.patterns.T @ weights
            xi /= (np.linalg.norm(xi) + 1e-8)

        best_idx = int(np.argmax(self.patterns @ xi))
        e = self.energy(xi)
        return xi, self.labels[best_idx], e


# ============================================================================
# ORGAN 2: Kramers Double-Well Drive (Curiosity & Exploration)
# Modulates system temperature T based on environmental novelty / boredom
# ============================================================================
class KramersCuriosityDrive:
    """Overdamped/underdamped Langevin dynamics in a bistable potential."""

    def __init__(self, barrier_height: float = 2.5, damping: float = 0.4) -> None:
        self.x = -1.0  # Start in Exploitation well (x = -1: Exploit, x = +1: Explore)
        self.v = 0.0
        self.barrier = barrier_height
        self.gamma = damping

    def reset(self, x: float = -1.0, v: float = 0.0) -> None:
        self.x = float(x)
        self.v = float(v)

    def step(self, novelty_signal: float, dt: float = 0.05, seed: Optional[int] = None) -> Tuple[str, float]:
        """
        Langevin dynamics on U(x) = barrier * (x^2 - 1)^2 - tilt * x.
        High novelty/prediction error heats the bath, inducing Kramers barrier hopping.
        """
        rng = np.random.default_rng(seed)
        temp = float(np.clip(0.2 + 2.5 * novelty_signal, 0.1, 5.0))

        # Force: -dU/dx = -4 * barrier * x * (x^2 - 1) + tilt
        force = -4.0 * self.barrier * self.x * (self.x**2 - 1.0) + 0.5 * novelty_signal
        noise = np.sqrt(2.0 * self.gamma * temp / dt) * rng.standard_normal()

        self.v = self.v + (-self.gamma * self.v + force + noise) * dt
        self.x = float(np.clip(self.x + self.v * dt, -2.0, 2.0))

        mode = "EXPLORE_NOVELTY" if self.x > 0.0 else "EXPLOIT_KNOWN"
        return mode, temp


# ============================================================================
# ORGAN 3: Multistate Potts Goal Arbiter (Contradiction Solver)
# Resolves competing agent drives (Speed, Accuracy, Safety, Autonomy, Cost)
# ============================================================================
class PottsGoalArbiter:
    """6 Cognitive Modules choosing among 4 discrete behavioral stances:

    0: VERIFY (Conservative/Verify)
    1: FAST_HEURISTIC (Fast/Low Latency)
    2: DEEP_REASONING (Deep Deliberation)
    3: AUTONOMOUS_EXEC (Direct Action)
    """

    STANCE_NAMES = ["VERIFY", "FAST_HEURISTIC", "DEEP_REASONING", "AUTONOMOUS_EXEC"]

    def __init__(self, num_modules: int = 6, num_states: int = 4, coupling_strength: float = 1.2) -> None:
        self.N = num_modules
        self.K = num_states
        # Ferromagnetic coupling encourages coherent behavioral alignment across modules
        self.J = np.ones((self.N, self.N), dtype=np.float32) * coupling_strength
        np.fill_diagonal(self.J, 0.0)

    def equilibrate(
        self,
        external_fields: np.ndarray,
        beta: float,
        sweeps: int = 40,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, float]:
        """Vectorized categorical Gibbs sampling with ground-state energy computation."""
        rng = np.random.default_rng(seed)
        spins = rng.integers(0, self.K, size=self.N)

        for _ in range(sweeps):
            for i in range(self.N):
                # Local field for each possible state k in {0..K-1}
                state_energies = np.zeros(self.K, dtype=np.float32)
                for k in range(self.K):
                    # Alignment bonus with neighbors in state k + external bias
                    neighbor_match = np.sum(self.J[i, :] * (spins == k))
                    state_energies[k] = -neighbor_match - external_fields[i, k]

                logits = -beta * (state_energies - np.min(state_energies))
                exp_logits = np.exp(logits)
                probs = exp_logits / np.sum(exp_logits)
                spins[i] = rng.choice(self.K, p=probs)

        # Compute total ground-state Hamiltonian energy
        total_energy = 0.0
        for i in range(self.N):
            total_energy -= external_fields[i, spins[i]]
            for j in range(i + 1, self.N):
                if spins[i] == spins[j]:
                    total_energy -= self.J[i, j]

        return spins, float(total_energy)


# ============================================================================
# ORGAN 4: Metabolic Homeostasis Scheduler (Resource QUBO)
# Prevents VRAM/CPU thrashing and enforces compute budgets
# Calibrated for NVIDIA GeForce RTX 5080 (16GB VRAM) + 128GB DDR5 System RAM
# ============================================================================
class MetabolicHomeostasis:
    """QUBO-inspired metabolic resource governor for local inference tiers."""

    def __init__(
        self,
        vram_limit_gb: float = 16.0,
        latency_weight: float = 0.01,
        cap_weight: float = 50.0,
        vram_weight: float = 200.0,
    ) -> None:
        self.vram_limit = float(vram_limit_gb)
        self.latency_weight = latency_weight
        self.cap_weight = cap_weight
        self.vram_weight = vram_weight
        self.endpoints = [
            {"name": "Local_Fast_3B", "vram_gb": 2.2, "latency_ms": 120, "capability": 0.5},
            {"name": "Local_Core_8B", "vram_gb": 6.5, "latency_ms": 350, "capability": 0.8},
            {"name": "Local_Deep_14B", "vram_gb": 11.5, "latency_ms": 900, "capability": 0.95},
        ]

    def select_execution_tier(self, required_capability: float, current_vram_used: float) -> Dict[str, Any]:
        """Solves single-variable metabolic Hamiltonian over available hardware tiers."""
        best_ep = self.endpoints[0]
        best_H = float("inf")

        for ep in self.endpoints:
            vram_overage = max(0.0, (current_vram_used + ep["vram_gb"]) - self.vram_limit)
            cap_deficit = max(0.0, required_capability - ep["capability"])
            # Metabolic Hamiltonian: latency penalty + capability deficit penalty + hard VRAM cliff penalty
            H = (
                self.latency_weight * ep["latency_ms"]
                + self.cap_weight * (cap_deficit**2)
                + self.vram_weight * (vram_overage**2)
            )
            if H < best_H:
                best_H = H
                best_ep = ep

        allocated = best_ep["vram_gb"]
        headroom = max(0.0, self.vram_limit - (current_vram_used + allocated))

        return {
            "selected_engine": best_ep["name"],
            "metabolic_energy": round(best_H, 3),
            "vram_allocated_gb": allocated,
            "vram_headroom_gb": round(headroom, 2),
            "capability": best_ep["capability"],
            "latency_ms": best_ep["latency_ms"],
        }


# ============================================================================
# UNIFIED SYNTHETIC INTELLIGENCE (SI) COGNITIVE LOOP
# ============================================================================
class ThermodynamicSICore:
    """Master controller that orchestrates the 4 thermodynamic organs in a unified cognitive tick."""

    def __init__(self, seed: Optional[int] = 42) -> None:
        self.memory = HopfieldAssociativeMemory(dim=64, beta=8.0)
        self.curiosity = KramersCuriosityDrive(barrier_height=2.5, damping=0.4)
        self.arbiter = PottsGoalArbiter(num_modules=6, num_states=4, coupling_strength=1.2)
        self.metabolism = MetabolicHomeostasis(vram_limit_gb=16.0)

        # Seed initial episodic attractors
        rng = np.random.default_rng(seed)
        self.memory.store(rng.standard_normal(64), "System_Architecture_Optimization")
        self.memory.store(rng.standard_normal(64), "Security_Boundary_Verification")
        self.memory.store(rng.standard_normal(64), "Novel_Research_Synthesis")

    def cognitive_tick(
        self,
        sensory_embedding: np.ndarray,
        novelty_score: float,
        current_vram: float = 4.0,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Executes a single coupled non-von-Neumann cognitive tick.

        Returns comprehensive telemetry across all 4 organs.
        """
        t0 = time.perf_counter()

        # 1. Associative Memory Recall (Hopfield Attractor Basin Convergence)
        recalled_vec, memory_label, mem_energy = self.memory.retrieve(sensory_embedding)

        # 2. Intrinsic Drive & Temperature Modulation (Kramers Double-Well)
        drive_mode, bath_temp = self.curiosity.step(novelty_score, seed=seed)
        beta = 1.0 / bath_temp

        # 3. Multi-Objective Goal Arbitration (Potts Spin Relaxation)
        # Bias fields derived from memory context + curiosity drive
        fields = np.zeros((self.arbiter.N, self.arbiter.K), dtype=np.float32)
        if drive_mode == "EXPLORE_NOVELTY":
            fields[:, 2] += 1.5  # Favor DEEP_REASONING
            fields[:, 3] += 1.2  # Favor AUTONOMOUS_EXEC
        else:
            fields[:, 0] += 1.5  # Favor VERIFY
            fields[:, 1] += 1.0  # Favor FAST_HEURISTIC

        goal_spins, potts_energy = self.arbiter.equilibrate(fields, beta=beta, sweeps=40, seed=seed)
        dominant_stance_idx = int(np.bincount(goal_spins, minlength=4).argmax())
        dominant_stance = PottsGoalArbiter.STANCE_NAMES[dominant_stance_idx]

        # 4. Metabolic Resource Allocation (Homeostasis QUBO on RTX 5080)
        req_cap = 0.9 if dominant_stance_idx == 2 else (0.8 if dominant_stance_idx == 3 else 0.6)
        metabolic_choice = self.metabolism.select_execution_tier(req_cap, current_vram)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "tick_latency_ms": round(elapsed_ms, 3),
            "organ_1_hopfield_attractor": memory_label,
            "organ_1_energy": round(mem_energy, 4),
            "organ_2_kramers_drive": drive_mode,
            "organ_2_bath_temp_T": round(bath_temp, 3),
            "organ_2_particle_x": round(self.curiosity.x, 3),
            "organ_2_particle_v": round(self.curiosity.v, 3),
            "organ_3_potts_stance": dominant_stance,
            "organ_3_potts_energy": round(potts_energy, 4),
            "organ_3_spins": goal_spins.tolist(),
            "organ_4_metabolism": metabolic_choice["selected_engine"],
            "organ_4_energy": metabolic_choice["metabolic_energy"],
            "organ_4_vram_allocated_gb": metabolic_choice["vram_allocated_gb"],
            "organ_4_vram_headroom_gb": metabolic_choice["vram_headroom_gb"],
            "organ_4_latency_ms": metabolic_choice["latency_ms"],
        }


if __name__ == "__main__":
    si = ThermodynamicSICore()
    sample_input = np.random.randn(64)
    print("Executing Autonomous SI Cognitive Tick...")
    telemetry = si.cognitive_tick(sample_input, novelty_score=0.85, current_vram=3.8)
    for k, v in telemetry.items():
        print(f"  {k}: {v}")
