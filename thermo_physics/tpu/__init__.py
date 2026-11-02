"""
thermo_physics.tpu
==================
Non-Equilibrium Langevin Dynamics for Multi-Modal Continuous-Discrete Optimization.
Simulates a Hybrid Thermodynamic Processing Unit (TPU) solving constrained non-convex
3-SAT and Ising Spin-Glass problems via Underdamped Langevin Dynamics with BAOAB symplectic
splitting, adaptive thermal scheduling, non-equilibrium work dissipation tracking (Jarzynski/Sekimoto),
and automated failure/ergodicity/FDT diagnostics.
"""

from thermo_physics.tpu.hamiltonian import (
    NonConvexIsingSAT,
    ProblemConfig,
    generate_random_3sat_spin_glass,
    generate_sk_spin_glass,
)
from thermo_physics.tpu.integrator import (
    BAOABIntegrator,
    EulerMaruyamaIntegrator,
    ThermalSchedule,
    ScheduleType,
    AdaptiveThermalScheduler,
)
from thermo_physics.tpu.thermodynamics import (
    ThermodynamicTracker,
    JarzynskiAnalysis,
    verify_jarzynski_equality,
)
from thermo_physics.tpu.monitors import (
    DiagnosticMonitor,
    NumericalDivergenceError,
    ErgodicityTrapWarning,
    FDTViolationWarning,
)
from thermo_physics.tpu.hardware import (
    DACQuantizer,
    OpAmpVoltageClipper,
    OrnsteinUhlenbeckNoise,
    SiliconRealismConfig,
)
from thermo_physics.tpu.solver import (
    HybridTPUSolver,
    ClassicalSimulatedAnnealing,
    TPUSolverConfig,
    TPUSolutionResult,
    BenchmarkComparison,
    run_tpu_vs_classical_benchmark,
)

__all__ = [
    "NonConvexIsingSAT",
    "ProblemConfig",
    "generate_random_3sat_spin_glass",
    "generate_sk_spin_glass",
    "BAOABIntegrator",
    "EulerMaruyamaIntegrator",
    "ThermalSchedule",
    "ScheduleType",
    "AdaptiveThermalScheduler",
    "ThermodynamicTracker",
    "JarzynskiAnalysis",
    "verify_jarzynski_equality",
    "DiagnosticMonitor",
    "NumericalDivergenceError",
    "ErgodicityTrapWarning",
    "FDTViolationWarning",
    "HybridTPUSolver",
    "ClassicalSimulatedAnnealing",
    "TPUSolverConfig",
    "TPUSolutionResult",
    "BenchmarkComparison",
    "run_tpu_vs_classical_benchmark",
    "SiliconRealismConfig",
    "DACQuantizer",
    "OpAmpVoltageClipper",
    "OrnsteinUhlenbeckNoise",
]
