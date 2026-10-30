"""Thermodynamic Synthetic Intelligence (SI) Cognitive Core.

Implements the 4 coupled non-von-Neumann thermodynamic organs:
1. Continuous Hopfield Associative Memory (Organ 1)
2. Kramers Double-Well Curiosity & Drive (Organ 2)
3. Categorical Potts Goal Arbiter (Organ 3)
4. QUBO Metabolic Homeostasis Scheduler (Organ 4)
"""

from thermo_physics.si_core.cognitive_engine import (
    HopfieldAssociativeMemory,
    KramersCuriosityDrive,
    PottsGoalArbiter,
    MetabolicHomeostasis,
    ThermodynamicSICore,
)

__all__ = [
    "HopfieldAssociativeMemory",
    "KramersCuriosityDrive",
    "PottsGoalArbiter",
    "MetabolicHomeostasis",
    "ThermodynamicSICore",
]
