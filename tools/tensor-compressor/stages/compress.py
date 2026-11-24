"""Compatibility alias module re-exporting Stage 2 compression and rank allocation APIs."""

from __future__ import annotations

from stages.decompose import (
    MoECompressionHook,
    allocate_energy_optimal_ranks,
    compute_target_rank,
    run_decomposition,
    whitened_svd_factorize,
)

run_compression = run_decomposition

__all__ = [
    "MoECompressionHook",
    "allocate_energy_optimal_ranks",
    "compute_target_rank",
    "run_compression",
    "run_decomposition",
    "whitened_svd_factorize",
]
