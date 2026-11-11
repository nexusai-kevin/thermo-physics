"""
upstream_patches.pr81
=====================
Exportable PR #81 contribution bundle for `extropic-ai/thrml`:
  - Differentiable JAX Silicon Realism primitives (`jax_dac_quantize_symmetric`, `jax_opamp_clip`)
  - 3-Pole 1/f Pink + Thermal Gaussian Noise Bath (`jax_step_pink_thermal_noise`)
  - Next-Gen Thermodynamic Architectures (`ContinuousVariableEBM`, `DeepBoltzmannMachine`)
  - Hardware Energy-per-Op Profiler (`ThermodynamicEnergyProfiler`)
  - Unified Hardware Abstraction Layer (`ThermoHAL`, `ThermoHALRouter`)
"""

from thermo_physics.benchmarks.energy_profiler import (
    EnergyProfileReport,
    ThermodynamicEnergyProfiler,
)
from thermo_physics.benchmarks.next_gen_ebm import (
    CVEBMConfig,
    ContinuousVariableEBM,
    DBMConfig,
    DeepBoltzmannMachine,
)
from thermo_physics.hal.thermo_hal import (
    ExecutionTarget,
    PhysicalSignalPacket,
    ThermoHAL,
    ThermoHALRouter,
)
from thermo_physics.tpu.jax_silicon import (
    JaxPinkNoiseState,
    JaxSiliconTelemetry,
    init_jax_pink_noise_state,
    jax_dac_quantize_symmetric,
    jax_opamp_clip,
    jax_step_pink_thermal_noise,
    run_jax_silicon_langevin,
)

__all__ = [
    "CVEBMConfig",
    "ContinuousVariableEBM",
    "DBMConfig",
    "DeepBoltzmannMachine",
    "EnergyProfileReport",
    "ExecutionTarget",
    "JaxPinkNoiseState",
    "JaxSiliconTelemetry",
    "PhysicalSignalPacket",
    "ThermoHAL",
    "ThermoHALRouter",
    "ThermodynamicEnergyProfiler",
    "init_jax_pink_noise_state",
    "jax_dac_quantize_symmetric",
    "jax_opamp_clip",
    "jax_step_pink_thermal_noise",
    "run_jax_silicon_langevin",
]
