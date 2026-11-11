"""
thermo_physics.router.solver
============================
Solves multi-task model assignment using an Energy-Based Potts/QUBO formulation.
Compares:
  - Thermodynamic Gibbs Sampler (annealed categorical Block Gibbs)
  - Greedy Baseline (isolated per-request cheapest/fastest assignment)

Objective function strictly balances:
  - Marginal token cost ($/1M tokens)
  - Latency SLA penalties (USD penalty proportional to SLA excess)
  - TPM / RPM rate-limit barrier penalties (prevents 429 errors)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from thermo_physics.router.config import EndpointConfig, ObjectiveWeights, RouterConfig
from thermo_physics.router.headroom import HeadroomSnapshot
from thermo_physics.router.tasks import RouteTask


@dataclass
class AssignmentDecision:
    task: RouteTask
    endpoint_name: str
    expected_cost_usd: float
    expected_latency_ms: float
    sla_satisfied: bool
    quality_satisfied: bool
    capabilities_satisfied: bool


@dataclass
class BatchRoutingSolution:
    strategy: str
    assignments: List[AssignmentDecision]
    total_cost_usd: float
    avg_latency_ms: float
    max_latency_ms: float
    sla_miss_count: int
    cap_mismatch_count: int
    total_energy: float
    solve_time_ms: float


class ThermodynamicRouterSolver:
    """Categorical Potts energy-based solver for batch LLM routing."""

    def __init__(self, config: RouterConfig):
        self.config = config
        self.endpoints = config.endpoints
        self.weights = config.weights

    def _compute_cost_and_penalties(
        self,
        task: RouteTask,
        ep: EndpointConfig,
        snapshot: HeadroomSnapshot,
    ) -> float:
        """Computes single-task assignment energy terms."""
        w = self.weights

        # 1. Hard feasibility checks
        if ep.quality < task.min_quality:
            return w.infeasible_usd
        if not set(task.required_caps).issubset(set(ep.capabilities)):
            return w.infeasible_usd
        if task.total_tokens_est > ep.context_window:
            return w.infeasible_usd
        if snapshot.blocked_until > task.ts:
            return w.infeasible_usd

        # 2. Financial cost
        cost = ep.cost_usd(task.in_tokens, task.out_tokens_est)

        # 3. Latency SLA penalty
        est_lat = ep.est_latency_ms(task.out_tokens_est)
        sla_penalty = 0.0
        if est_lat > task.max_latency_ms:
            excess = (est_lat - task.max_latency_ms) / max(1.0, task.max_latency_ms)
            sla_penalty = w.sla_miss_usd * task.priority * (1.0 + excess)

        return cost + sla_penalty

    def solve_greedy(
        self,
        tasks: List[RouteTask],
        snapshots: Dict[str, HeadroomSnapshot],
    ) -> BatchRoutingSolution:
        """Baseline: Greedy assignment considering each task sequentially."""
        t0 = time.perf_counter()
        assignments: List[AssignmentDecision] = []
        k_count = len(self.endpoints)

        # Track rolling simulated load within batch
        active_tpm = {ep.name: snapshots[ep.name].active_tpm for ep in self.endpoints}
        active_rpm = {ep.name: snapshots[ep.name].active_rpm for ep in self.endpoints}

        for task in tasks:
            best_ep = self.endpoints[0]
            best_val = float("inf")

            for ep in self.endpoints:
                snap = snapshots[ep.name]
                base_energy = self._compute_cost_and_penalties(task, ep, snap)

                # Rate limit penalty under current batch state
                proj_tpm = active_tpm[ep.name] + task.total_tokens_est
                proj_rpm = active_rpm[ep.name] + 1
                rl_penalty = 0.0
                if proj_tpm > ep.tpm_limit:
                    rl_penalty += self.weights.rate_limit_usd * ((proj_tpm - ep.tpm_limit) / ep.tpm_limit)
                if proj_rpm > ep.rpm_limit:
                    rl_penalty += self.weights.rate_limit_usd * ((proj_rpm - ep.rpm_limit) / ep.rpm_limit)

                total_score = base_energy + rl_penalty
                if total_score < best_val:
                    best_val = total_score
                    best_ep = ep

            active_tpm[best_ep.name] += task.total_tokens_est
            active_rpm[best_ep.name] += 1

            cost = best_ep.cost_usd(task.in_tokens, task.out_tokens_est)
            lat = best_ep.est_latency_ms(task.out_tokens_est)
            assignments.append(
                AssignmentDecision(
                    task=task,
                    endpoint_name=best_ep.name,
                    expected_cost_usd=cost,
                    expected_latency_ms=lat,
                    sla_satisfied=lat <= task.max_latency_ms,
                    quality_satisfied=best_ep.quality >= task.min_quality,
                    capabilities_satisfied=set(task.required_caps).issubset(set(best_ep.capabilities)),
                )
            )

        elapsed = (time.perf_counter() - t0) * 1000.0
        return self._format_solution("Greedy Baseline", assignments, elapsed)

    def solve_thermo(
        self,
        tasks: List[RouteTask],
        snapshots: Dict[str, HeadroomSnapshot],
    ) -> BatchRoutingSolution:
        """Global energy minimizer via categorical Gibbs sampling over the batch Potts model."""
        t0 = time.perf_counter()
        m = len(tasks)
        k_count = len(self.endpoints)
        if m == 0:
            return self._format_solution("Thermodynamic Gibbs", [], 0.0)

        solver_cfg = self.config.solver
        rng = np.random.default_rng(solver_cfg.seed)

        # Precompute base linear energy matrix: (M, K)
        linear_costs = np.zeros((m, k_count), dtype=np.float64)
        for i, task in enumerate(tasks):
            for k, ep in enumerate(self.endpoints):
                linear_costs[i, k] = self._compute_cost_and_penalties(task, ep, snapshots[ep.name])

        task_tokens = np.array([t.total_tokens_est for t in tasks], dtype=np.float64)
        tpm_limits = np.array([ep.tpm_limit for ep in self.endpoints], dtype=np.float64)
        base_tpm = np.array([snapshots[ep.name].active_tpm for ep in self.endpoints], dtype=np.float64)

        # Initial state: pick lowest linear cost per task
        state = np.argmin(linear_costs, axis=1)

        # Running tally of tokens per endpoint
        curr_tpm = base_tpm.copy()
        for i in range(m):
            curr_tpm[state[i]] += task_tokens[i]

        def compute_global_energy(curr_state: np.ndarray, tpm_tracker: np.ndarray) -> float:
            total = 0.0
            for i in range(m):
                total += linear_costs[i, curr_state[i]]
            for k in range(k_count):
                if tpm_tracker[k] > tpm_limits[k]:
                    over = (tpm_tracker[k] - tpm_limits[k]) / tpm_limits[k]
                    total += self.weights.rate_limit_usd * over
                elif tpm_tracker[k] > self.weights.soft_utilization * tpm_limits[k]:
                    soft_over = (tpm_tracker[k] - self.weights.soft_utilization * tpm_limits[k]) / tpm_limits[k]
                    total += self.weights.headroom_usd * soft_over
            return total

        best_state = state.copy()
        best_energy = compute_global_energy(state, curr_tpm)

        # Annealed Categorical Block Gibbs Sweeps
        for step in range(solver_cfg.n_steps):
            beta = solver_cfg.beta_start * ((solver_cfg.beta_end / solver_cfg.beta_start) ** (step / max(1, solver_cfg.n_steps - 1)))

            # Randomly update a fraction of tasks
            indices = rng.permutation(m)
            for i in indices:
                old_k = state[i]
                tok = task_tokens[i]

                # Evaluate unnormalized log-probabilities for all K endpoints
                logits = np.zeros(k_count, dtype=np.float64)
                for k in range(k_count):
                    delta_tpm = (0.0 if k == old_k else tok)
                    proj_tpm = curr_tpm[k] + delta_tpm
                    penalty = 0.0
                    if proj_tpm > tpm_limits[k]:
                        penalty += self.weights.rate_limit_usd * ((proj_tpm - tpm_limits[k]) / tpm_limits[k])
                    elif proj_tpm > self.weights.soft_utilization * tpm_limits[k]:
                        penalty += self.weights.headroom_usd * ((proj_tpm - self.weights.soft_utilization * tpm_limits[k]) / tpm_limits[k])

                    logits[k] = -beta * (linear_costs[i, k] + penalty)

                # Softmax with max subtraction
                shift = logits - np.max(logits)
                probs = np.exp(shift)
                probs /= np.sum(probs)

                # Sample categorical update
                new_k = rng.choice(k_count, p=probs)
                if new_k != old_k:
                    curr_tpm[old_k] -= tok
                    curr_tpm[new_k] += tok
                    state[i] = new_k

            step_energy = compute_global_energy(state, curr_tpm)
            if step_energy < best_energy:
                best_energy = step_energy
                best_state = state.copy()

        elapsed = (time.perf_counter() - t0) * 1000.0

        # Construct assignments
        assignments = []
        for i in range(m):
            ep = self.endpoints[best_state[i]]
            task = tasks[i]
            cost = ep.cost_usd(task.in_tokens, task.out_tokens_est)
            lat = ep.est_latency_ms(task.out_tokens_est)
            assignments.append(
                AssignmentDecision(
                    task=task,
                    endpoint_name=ep.name,
                    expected_cost_usd=cost,
                    expected_latency_ms=lat,
                    sla_satisfied=lat <= task.max_latency_ms,
                    quality_satisfied=ep.quality >= task.min_quality,
                    capabilities_satisfied=set(task.required_caps).issubset(set(ep.capabilities)),
                )
            )

        return self._format_solution("Thermodynamic Gibbs", assignments, elapsed, best_energy)

    def _format_solution(
        self,
        strategy: str,
        assignments: List[AssignmentDecision],
        elapsed_ms: float,
        energy: float = 0.0,
    ) -> BatchRoutingSolution:
        total_cost = sum(a.expected_cost_usd for a in assignments)
        lats = [a.expected_latency_ms for a in assignments]
        avg_lat = float(np.mean(lats)) if lats else 0.0
        max_lat = float(np.max(lats)) if lats else 0.0
        sla_misses = sum(1 for a in assignments if not a.sla_satisfied)
        cap_misses = sum(1 for a in assignments if not a.capabilities_satisfied)

        return BatchRoutingSolution(
            strategy=strategy,
            assignments=assignments,
            total_cost_usd=total_cost,
            avg_latency_ms=avg_lat,
            max_latency_ms=max_lat,
            sla_miss_count=sla_misses,
            cap_mismatch_count=cap_misses,
            total_energy=energy,
            solve_time_ms=elapsed_ms,
        )
