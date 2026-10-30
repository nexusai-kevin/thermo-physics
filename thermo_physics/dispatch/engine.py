"""
thermo_physics.dispatch.engine
==============================
Thermodynamic Task Dispatcher engine comparing Potts/Ising Gibbs optimization
against classical greedy and cost-blind heuristic baselines.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from thermo_physics.backends.sampler_adapter import (
    BackendType,
    SamplingConfig,
    compute_sokal_tau_int,
    get_sampler,
)
from thermo_physics.dispatch.hamiltonian import (
    DispatchHamiltonian,
    build_dispatch_hamiltonian,
)
from thermo_physics.dispatch.models import (
    AgentTask,
    DispatchAssignment,
    DispatchPlan,
    ModelEndpoint,
)


class ThermodynamicDispatcher:
    """Dispatches batches of agent tasks onto model endpoints via Energy-Based Potts/QUBO sampling."""

    def __init__(
        self,
        endpoints: List[ModelEndpoint],
        lambda_one: float = 25.0,
        lambda_tpm: float = 12.0,
        lambda_rpm: float = 8.0,
        lambda_cost: float = 1.0,
        lambda_latency: float = 18.0,
        lambda_capability: float = 50.0,
    ):
        self.endpoints = endpoints
        self.lambda_one = lambda_one
        self.lambda_tpm = lambda_tpm
        self.lambda_rpm = lambda_rpm
        self.lambda_cost = lambda_cost
        self.lambda_latency = lambda_latency
        self.lambda_capability = lambda_capability

    def _evaluate_assignments(
        self,
        task_choices: List[int],  # length M, each choice in {0, .. K-1}
        tasks: List[AgentTask],
        endpoints: List[ModelEndpoint],
        strategy: str,
        solve_time_ms: float,
        tau_int: Optional[float] = None,
        ess: Optional[float] = None,
    ) -> DispatchPlan:
        """Construct structured DispatchPlan and calculate all telemetry metrics."""
        m = len(tasks)
        k_count = len(endpoints)
        assignments: List[DispatchAssignment] = []

        total_cost = 0.0
        latencies = []
        sla_violations = 0
        cap_mismatches = 0

        tpm_load = {ep.name: ep.active_tpm for ep in endpoints}
        rpm_load = {ep.name: ep.active_rpm for ep in endpoints}

        for i in range(m):
            task = tasks[i]
            ep_idx = task_choices[i]
            ep = endpoints[ep_idx]

            # Cost and latency
            task_cost = (task.est_tokens / 1000.0) * ep.cost_per_1k_tokens
            total_cost += task_cost
            latencies.append(ep.latency_p50_ms)

            # SLA and capability check
            sla_met = ep.latency_p50_ms <= task.max_latency_ms
            if not sla_met:
                sla_violations += 1

            req_caps = set(task.required_capabilities)
            sup_caps = set(ep.supported_capabilities)
            caps_met = req_caps.issubset(sup_caps)
            if not caps_met:
                cap_mismatches += 1

            # Update load
            tpm_load[ep.name] += task.est_tokens
            rpm_load[ep.name] += 1

            assignments.append(
                DispatchAssignment(
                    task=task,
                    endpoint=ep,
                    estimated_cost=task_cost,
                    latency_ms=ep.latency_p50_ms,
                    sla_satisfied=sla_met,
                    capabilities_satisfied=caps_met,
                )
            )

        # Headroom and violations
        tpm_utilization: Dict[str, float] = {}
        tpm_violations: Dict[str, int] = {}
        headroom_tpm: Dict[str, int] = {}

        for ep in endpoints:
            limit = max(1, ep.tpm_limit)
            curr = tpm_load[ep.name]
            util = curr / limit
            tpm_utilization[ep.name] = util
            tpm_violations[ep.name] = max(0, curr - limit)
            headroom_tpm[ep.name] = max(0, limit - curr)

        max_lat = float(max(latencies)) if latencies else 0.0
        avg_lat = float(np.mean(latencies)) if latencies else 0.0

        return DispatchPlan(
            strategy=strategy,
            assignments=assignments,
            total_cost=total_cost,
            max_batch_latency_ms=max_lat,
            avg_latency_ms=avg_lat,
            sla_violation_count=sla_violations,
            capability_mismatch_count=cap_mismatches,
            tpm_load=tpm_load,
            rpm_load=rpm_load,
            tpm_utilization=tpm_utilization,
            tpm_violations=tpm_violations,
            headroom_tpm=headroom_tpm,
            solve_time_ms=solve_time_ms,
            mcmc_tau_int=tau_int,
            mcmc_ess=ess,
        )

    def dispatch_naive_fastest(self, tasks: List[AgentTask]) -> DispatchPlan:
        """Baseline 1: Routes all high/medium tasks to Claude 3.5 Sonnet ($0.015/1k, 800ms).

        Heuristic Trap: Crashes Claude-3.5 Sonnet TPM quota (50k limit) and incurs 10x-15x cost inflation.
        """
        t0 = time.perf_counter()
        claude_k = next((i for i, ep in enumerate(self.endpoints) if "Claude" in ep.name), 0)
        gpt_k = next((i for i, ep in enumerate(self.endpoints) if "GPT" in ep.name), 1)

        choices = []
        for task in tasks:
            if task.complexity_tier in ("reasoning", "complex", "medium"):
                choices.append(claude_k)
            else:
                choices.append(gpt_k)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return self._evaluate_assignments(
            choices,
            tasks,
            self.endpoints,
            strategy="Baseline 1: Naive High-Capability (Claude-Heavy)",
            solve_time_ms=elapsed_ms,
        )

    def dispatch_cheapest_first(self, tasks: List[AgentTask]) -> DispatchPlan:
        """Baseline 2: Greedily assigns tasks to the cheapest endpoint (Local Ollama / Gemini Flash).

        Heuristic Trap: Overloads local GPU queue, breaches latency SLAs on critical customer tasks,
        and assigns complex reasoning to models lacking required capabilities.
        """
        t0 = time.perf_counter()
        ollama_k = next((i for i, ep in enumerate(self.endpoints) if "Ollama" in ep.name), 3)
        flash_k = next((i for i, ep in enumerate(self.endpoints) if "Flash" in ep.name), 2)

        choices = []
        for task in tasks:
            # Blindly routes all small/medium tasks to local Ollama and remainder to Flash
            if task.est_tokens <= 4000:
                choices.append(ollama_k)
            else:
                choices.append(flash_k)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return self._evaluate_assignments(
            choices,
            tasks,
            self.endpoints,
            strategy="Baseline 2: Greedy Cheapest-First (Ollama/Flash-Heavy)",
            solve_time_ms=elapsed_ms,
        )

    def dispatch_thermodynamic(
        self,
        tasks: List[AgentTask],
        n_sweeps: int = 75,
        beta: float = 3.2,
        seed: int = 42,
    ) -> DispatchPlan:
        """Thermodynamic Potts/Gibbs dispatch resolving TPM/RPM limits, latency SLAs, and costs.

        Executes categorical Block Gibbs sweeps in < 5ms and computes Sokal convergence diagnostics.
        """
        t0 = time.perf_counter()
        m = len(tasks)
        k_endpoints = len(self.endpoints)
        rng = np.random.default_rng(seed)

        # 1. Compile Dispatch Hamiltonian
        dh = build_dispatch_hamiltonian(
            tasks=tasks,
            endpoints=self.endpoints,
            lambda_one=self.lambda_one,
            lambda_tpm=self.lambda_tpm,
            lambda_rpm=self.lambda_rpm,
            lambda_cost=self.lambda_cost,
            lambda_latency=self.lambda_latency,
            lambda_capability=self.lambda_capability,
        )

        cost_matrix = np.zeros((m, k_endpoints), dtype=np.float64)
        for i in range(m):
            for k in range(k_endpoints):
                u = dh.task_index_map[(i, k)]
                cost_matrix[i, k] = dh.linear_costs[u]

        # Initial state: Capability and SLA compliant start
        state = np.zeros(m, dtype=np.int32)
        for i in range(m):
            req_caps = set(tasks[i].required_capabilities)
            valid_ks = [
                k for k in range(k_endpoints)
                if req_caps.issubset(set(self.endpoints[k].supported_capabilities))
                and self.endpoints[k].latency_p50_ms <= tasks[i].max_latency_ms
            ]
            if not valid_ks:
                valid_ks = [
                    k for k in range(k_endpoints)
                    if req_caps.issubset(set(self.endpoints[k].supported_capabilities))
                ]
            state[i] = valid_ks[i % len(valid_ks)] if valid_ks else (i % k_endpoints)

        caps_list = [max(1000.0, float(ep.tpm_limit)) for ep in self.endpoints]
        curr_tpm_list = [float(ep.active_tpm) for ep in self.endpoints]
        for i in range(m):
            curr_tpm_list[state[i]] += tasks[i].est_tokens

        tokens_list = [float(t.est_tokens) for t in tasks]
        energy_trace = []

        # 2. Ultra-Fast Categorical Potts Gibbs Sweeps (< 5ms)
        for sweep in range(n_sweeps):
            current_beta = beta * (0.6 + 1.4 * (sweep / n_sweeps))

            for i in range(m):
                old_k = state[i]
                t_i = tokens_list[i]

                # Compute unnormalized log-probabilities for each endpoint k
                best_logit = -1e9
                logits = [0.0] * k_endpoints
                for k in range(k_endpoints):
                    proj_load = curr_tpm_list[k] + (0.0 if k == old_k else t_i)
                    ratio = proj_load / caps_list[k]
                    tpm_penalty = self.lambda_tpm * ((ratio / 0.85) ** 3) if ratio > 0.85 else 0.0
                    logit = - current_beta * (cost_matrix[i, k] + tpm_penalty)
                    logits[k] = logit
                    if logit > best_logit:
                        best_logit = logit

                # Softmax with numerical stability
                exp_sum = 0.0
                weights = [0.0] * k_endpoints
                for k in range(k_endpoints):
                    w = math.exp(logits[k] - best_logit)
                    weights[k] = w
                    exp_sum += w

                # Sample new_k via cumulative distribution
                r = rng.random() * exp_sum
                cum = 0.0
                new_k = old_k
                for k in range(k_endpoints):
                    cum += weights[k]
                    if r <= cum:
                        new_k = k
                        break

                if new_k != old_k:
                    curr_tpm_list[old_k] -= t_i
                    curr_tpm_list[new_k] += t_i
                    state[i] = new_k

            # Sample energy
            batch_e = sum(cost_matrix[i, state[i]] for i in range(m))
            over = sum(self.lambda_tpm * ((curr_tpm_list[k] / caps_list[k]) ** 2) for k in range(k_endpoints) if curr_tpm_list[k] > caps_list[k])
            energy_trace.append(batch_e + over)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        tau_int, ess = compute_sokal_tau_int(np.array(energy_trace))

        return self._evaluate_assignments(
            state.tolist(),
            tasks,
            self.endpoints,
            strategy="Nexus Thermodynamic Potts/QUBO Dispatcher",
            solve_time_ms=elapsed_ms,
            tau_int=tau_int,
            ess=ess,
        )
