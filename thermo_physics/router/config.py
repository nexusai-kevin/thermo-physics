"""
thermo_physics.router.config
============================
Endpoint, tier-policy, and objective-weight configuration for the Nexus
rate-limit-aware LLM router. Loaded from a JSON file so partners can edit
prices / limits without touching code.

All objective weights are denominated in USD so the optimisation target is
explainable to a customer: "we treat one missed SLA as costing $X".
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class EndpointConfig:
    """One routable model endpoint (OpenAI-compatible or Ollama-native upstream)."""

    name: str                      # Router-facing id (what clients may request)
    provider: str
    base_url: str                  # e.g. https://api.openai.com/v1 or http://localhost:11434/v1
    model: str                     # Upstream model id sent to the provider
    api_key_env: Optional[str] = None  # Env var holding the key; None -> pass client auth through
    input_cost_per_1m: float = 0.0
    output_cost_per_1m: float = 0.0
    ttft_ms: float = 500.0         # Time-to-first-token estimate
    ms_per_output_token: float = 10.0
    rpm_limit: int = 500
    tpm_limit: int = 100_000
    context_window: int = 128_000
    quality: int = 2               # 1 = small/fast, 2 = mid, 3 = frontier
    capabilities: List[str] = field(default_factory=lambda: ["chat"])
    aliases: List[str] = field(default_factory=list)

    def cost_usd(self, input_tokens: float, output_tokens: float) -> float:
        return (input_tokens * self.input_cost_per_1m + output_tokens * self.output_cost_per_1m) / 1e6

    def est_latency_ms(self, output_tokens: float) -> float:
        return self.ttft_ms + output_tokens * self.ms_per_output_token

    def matches(self, model_id: Optional[str]) -> bool:
        if not model_id:
            return False
        return model_id in (self.name, self.model) or model_id in self.aliases


@dataclass
class TierPolicy:
    """Defaults applied when a request does not carry explicit routing headers."""

    min_quality: int
    max_latency_ms: float
    default_output_tokens: int
    priority: float = 1.0


DEFAULT_TIERS: Dict[str, TierPolicy] = {
    "simple": TierPolicy(min_quality=1, max_latency_ms=8_000, default_output_tokens=256, priority=1.0),
    "medium": TierPolicy(min_quality=2, max_latency_ms=15_000, default_output_tokens=512, priority=1.0),
    "complex": TierPolicy(min_quality=2, max_latency_ms=30_000, default_output_tokens=1_024, priority=2.0),
    "reasoning": TierPolicy(min_quality=3, max_latency_ms=60_000, default_output_tokens=2_048, priority=2.0),
}

QUALITY_TO_TIER = {1: "simple", 2: "medium", 3: "reasoning"}


@dataclass
class ObjectiveWeights:
    """USD-denominated penalty weights for the routing energy function."""

    sla_miss_usd: float = 0.02         # Per missed SLA, multiplied by task priority
    rate_limit_usd: float = 0.05       # Per request expected to overflow a rate limit (-> 429)
    headroom_usd: float = 0.01         # Soft push away from running an endpoint near its limit
    soft_utilization: float = 0.80     # Utilisation above which the headroom penalty starts
    infeasible_usd: float = 1_000.0    # Capability / quality / context-window mismatch


@dataclass
class SolverSettings:
    backend: str = "numpy"             # "numpy" (lowest latency) or "jax" (jit, hardware-swap path)
    n_replicas: int = 16
    n_steps: int = 150
    beta_start: float = 0.3
    beta_end: float = 30.0
    update_fraction: float = 0.3
    seed: int = 0


@dataclass
class RouterConfig:
    endpoints: List[EndpointConfig]
    mode: str = "shadow"               # "shadow" (observe only) or "route" (act on decisions)
    strategy: str = "thermo"           # "thermo" or "greedy"
    batch_window_ms: float = 50.0
    max_batch: int = 64
    log_path: str = "logs/router_decisions.jsonl"
    ollama_native_url: Optional[str] = "http://localhost:11434"
    retry_penalty_ms: float = 2_000.0  # Replay: latency added to a request that hits a 429
    tiers: Dict[str, TierPolicy] = field(default_factory=lambda: dict(DEFAULT_TIERS))
    weights: ObjectiveWeights = field(default_factory=ObjectiveWeights)
    solver: SolverSettings = field(default_factory=SolverSettings)

    def __post_init__(self) -> None:
        if self.mode not in ("shadow", "route"):
            raise ValueError(f"mode must be 'shadow' or 'route', got {self.mode!r}")
        if self.strategy not in ("thermo", "greedy"):
            raise ValueError(f"strategy must be 'thermo' or 'greedy', got {self.strategy!r}")
        if not self.endpoints:
            raise ValueError("RouterConfig needs at least one endpoint")
        names = [e.name for e in self.endpoints]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate endpoint names: {names}")

    def find_endpoint(self, model_id: Optional[str]) -> Optional[EndpointConfig]:
        for ep in self.endpoints:
            if ep.matches(model_id):
                return ep
        return None

    def endpoint_index(self, name: str) -> int:
        for i, ep in enumerate(self.endpoints):
            if ep.name == name:
                return i
        raise KeyError(name)

    def tier(self, name: str) -> TierPolicy:
        return self.tiers.get(name, self.tiers["medium"])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def config_from_dict(data: Dict[str, Any]) -> RouterConfig:
    endpoints = [EndpointConfig(**e) for e in data["endpoints"]]
    tiers = dict(DEFAULT_TIERS)
    for name, t in (data.get("tiers") or {}).items():
        tiers[name] = TierPolicy(**t)
    kwargs = {k: v for k, v in data.items() if k not in ("endpoints", "tiers", "weights", "solver", "_comment")}
    return RouterConfig(
        endpoints=endpoints,
        tiers=tiers,
        weights=ObjectiveWeights(**(data.get("weights") or {})),
        solver=SolverSettings(**(data.get("solver") or {})),
        **kwargs,
    )


def load_config(path: str | Path) -> RouterConfig:
    with open(path, "r", encoding="utf-8") as fh:
        return config_from_dict(json.load(fh))
