"""
thermo_physics.router
=====================
Production-ready rate-limit-aware thermodynamic batch dispatcher and shadow proxy.
"""

from thermo_physics.router.config import (
    DEFAULT_TIERS,
    EndpointConfig,
    ObjectiveWeights,
    RouterConfig,
    SolverSettings,
    TierPolicy,
    config_from_dict,
    load_config,
)
from thermo_physics.router.headroom import (
    HeadroomSnapshot,
    HeadroomTracker,
    parse_reset_seconds,
    parse_retry_after,
)
from thermo_physics.router.proxy import create_router_app
from thermo_physics.router.replay import (
    CounterfactualReplayEngine,
    ReplayMetrics,
    ReplayReport,
)
from thermo_physics.router.solver import (
    AssignmentDecision,
    BatchRoutingSolution,
    ThermodynamicRouterSolver,
)
from thermo_physics.router.tasks import (
    RouteTask,
    build_task,
    task_from_ollama_body,
    task_from_openai_body,
)

__all__ = [
    "EndpointConfig",
    "TierPolicy",
    "ObjectiveWeights",
    "SolverSettings",
    "RouterConfig",
    "DEFAULT_TIERS",
    "load_config",
    "config_from_dict",
    "HeadroomSnapshot",
    "HeadroomTracker",
    "parse_retry_after",
    "parse_reset_seconds",
    "RouteTask",
    "build_task",
    "task_from_openai_body",
    "task_from_ollama_body",
    "AssignmentDecision",
    "BatchRoutingSolution",
    "ThermodynamicRouterSolver",
    "ReplayMetrics",
    "ReplayReport",
    "CounterfactualReplayEngine",
    "create_router_app",
]
