"""
thermo_physics.dispatch
=======================
Nexus AI Thermodynamic Agent & LLM Task Dispatcher (nexus-dispatcher).
Maps multi-agent task routing, rate-limit headroom, and latency SLA compliance
to an Energy-Based Potts/QUBO Hamiltonian solved via JAX Block Gibbs sampling.
"""

from thermo_physics.dispatch.models import (
    AgentTask,
    ComplexityTier,
    DispatchAssignment,
    DispatchPlan,
    ModelEndpoint,
)
from thermo_physics.dispatch.hamiltonian import (
    DispatchHamiltonian,
    build_dispatch_hamiltonian,
)
from thermo_physics.dispatch.engine import (
    ThermodynamicDispatcher,
)

__all__ = [
    "AgentTask",
    "ComplexityTier",
    "DispatchAssignment",
    "DispatchPlan",
    "ModelEndpoint",
    "DispatchHamiltonian",
    "build_dispatch_hamiltonian",
    "ThermodynamicDispatcher",
]
