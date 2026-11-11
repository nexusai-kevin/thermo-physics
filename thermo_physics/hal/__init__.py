"""
thermo_physics.hal
==================
Hardware Abstraction Layer (HAL) and hardware-aware task router for physical TSUs
and ESP32-S3 / repurposed analog entropy edge nodes.
"""

from thermo_physics.hal.thermo_hal import (
    EdgeNodeStatus,
    ExecutionTarget,
    PhysicalSignalPacket,
    RepurposedEntropyHarvester,
    ThermoHAL,
    ThermoHALRouter,
)

__all__ = [
    "EdgeNodeStatus",
    "ExecutionTarget",
    "PhysicalSignalPacket",
    "RepurposedEntropyHarvester",
    "ThermoHAL",
    "ThermoHALRouter",
]
