"""
thermo_hal.py
=============
Top-level Hardware Abstraction Layer (HAL) entry point for `thermo-physics`.
Re-exports `ThermoHAL`, `ThermoHALRouter`, `PhysicalSignalPacket`, `ExecutionTarget`,
and `RepurposedEntropyHarvester` from `thermo_physics.hal.thermo_hal`.
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


if __name__ == "__main__":
    import jax.numpy as jnp

    hal = ThermoHAL(dac_bits=6, v_rail=1.5)
    router = ThermoHALRouter(hal=hal)

    j_mat = jnp.array([[0.0, -0.8, 0.5, -0.4], [-0.8, 0.0, -0.6, 0.7], [0.5, -0.6, 0.0, -0.9], [-0.4, 0.7, -0.9, 0.0]])
    h_vec = jnp.array([0.2, -0.3, 0.1, -0.15])

    res, packet, meta = router.dispatch((j_mat, h_vec))
    print(res.summary())
    print("HAL Signal Packet:", packet.summary_dict())
    print("Routing Decision:", meta["selected_target"], "| Entropy Source:", meta["entropy_source"])
