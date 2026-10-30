"""
thermo_physics.backends
=======================
Backend abstractions and hardware/software execution targets.
"""

from thermo_physics.backends.sampler_adapter import (
    BackendType,
    BaseThermodynamicSampler,
    ExtropicHardwareSampler,
    HamiltonianGraph,
    JaxBlockGibbsSampler,
    SamplingConfig,
    SamplingResult,
    compute_sokal_tau_int,
    get_sampler,
)

__all__ = [
    "BackendType",
    "BaseThermodynamicSampler",
    "ExtropicHardwareSampler",
    "HamiltonianGraph",
    "JaxBlockGibbsSampler",
    "SamplingConfig",
    "SamplingResult",
    "compute_sokal_tau_int",
    "get_sampler",
]
