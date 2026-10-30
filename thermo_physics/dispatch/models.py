"""
thermo_physics.dispatch.models
==============================
Data structures for agent tasks, model endpoints, assignments, and dispatch plans.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ComplexityTier(str, Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    REASONING = "reasoning"


@dataclass
class ModelEndpoint:
    """LLM or Agent execution substrate specification."""
    name: str
    provider: str
    cost_per_1k_tokens: float
    latency_p50_ms: float
    rpm_limit: int
    tpm_limit: int
    active_tpm: int = 0
    active_rpm: int = 0
    context_window: int = 128_000
    supported_capabilities: List[str] = field(default_factory=list)

    @property
    def remaining_tpm(self) -> int:
        return max(0, self.tpm_limit - self.active_tpm)

    @property
    def remaining_rpm(self) -> int:
        return max(0, self.rpm_limit - self.active_rpm)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentTask:
    """Autonomous agent task requesting LLM inference."""
    task_id: str
    complexity_tier: str
    est_tokens: int
    max_latency_ms: float
    priority_weight: float = 1.0  # 1.0 (low/batch) to 5.0 (critical interactive)
    required_capabilities: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DispatchAssignment:
    """Concrete routing decision mapping a task to an endpoint."""
    task: AgentTask
    endpoint: ModelEndpoint
    estimated_cost: float
    latency_ms: float
    sla_satisfied: bool
    capabilities_satisfied: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task.task_id,
            "complexity_tier": self.task.complexity_tier,
            "est_tokens": self.task.est_tokens,
            "endpoint": self.endpoint.name,
            "provider": self.endpoint.provider,
            "estimated_cost": round(self.estimated_cost, 6),
            "latency_ms": round(self.latency_ms, 1),
            "sla_satisfied": self.sla_satisfied,
            "capabilities_satisfied": self.capabilities_satisfied,
        }


@dataclass
class DispatchPlan:
    """Global multi-task batch allocation solution and telemetry."""
    strategy: str
    assignments: List[DispatchAssignment]
    total_cost: float
    max_batch_latency_ms: float
    avg_latency_ms: float
    sla_violation_count: int
    capability_mismatch_count: int
    tpm_load: Dict[str, int]
    rpm_load: Dict[str, int]
    tpm_utilization: Dict[str, float]
    tpm_violations: Dict[str, int]
    headroom_tpm: Dict[str, int]
    solve_time_ms: float = 0.0
    mcmc_tau_int: Optional[float] = None
    mcmc_ess: Optional[float] = None

    def summary(self) -> str:
        tpm_summary = ", ".join(
            f"{k}: {self.tpm_load[k]:,}/{self.tpm_load[k] + self.headroom_tpm[k]:,} TPM ({self.tpm_utilization[k]*100:.1f}%)"
            for k in sorted(self.tpm_load.keys())
        )
        violations_str = ", ".join(
            f"{k}: +{v:,} OVER" for k, v in self.tpm_violations.items() if v > 0
        ) or "None (0 violations)"

        return (
            f"=== Dispatch Plan: {self.strategy} ===\n"
            f"Tasks Dispatched       : {len(self.assignments)}\n"
            f"Total Batch Cost       : ${self.total_cost:.4f}\n"
            f"Latency SLA Violations : {self.sla_violation_count}/{len(self.assignments)} "
            f"({'CLEAN' if self.sla_violation_count == 0 else 'BREACHED'})\n"
            f"Capability Mismatches  : {self.capability_mismatch_count}\n"
            f"Max Batch Latency      : {self.max_batch_latency_ms:.1f} ms (Avg: {self.avg_latency_ms:.1f} ms)\n"
            f"Rate Limit Violations  : {violations_str}\n"
            f"TPM Utilization        : {tpm_summary}\n"
            f"Optimization Solve Time: {self.solve_time_ms:.2f} ms"
            + (f" | MCMC tau_int: {self.mcmc_tau_int:.2f}" if self.mcmc_tau_int else "")
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "total_tasks": len(self.assignments),
            "total_cost": round(self.total_cost, 4),
            "max_batch_latency_ms": round(self.max_batch_latency_ms, 1),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "sla_violation_count": self.sla_violation_count,
            "capability_mismatch_count": self.capability_mismatch_count,
            "tpm_load": self.tpm_load,
            "rpm_load": self.rpm_load,
            "tpm_utilization": {k: round(v, 4) for k, v in self.tpm_utilization.items()},
            "tpm_violations": self.tpm_violations,
            "headroom_tpm": self.headroom_tpm,
            "solve_time_ms": round(self.solve_time_ms, 2),
            "mcmc_tau_int": round(self.mcmc_tau_int, 2) if self.mcmc_tau_int else None,
            "mcmc_ess": round(self.mcmc_ess, 1) if self.mcmc_ess else None,
            "assignments": [a.to_dict() for a in self.assignments],
        }
