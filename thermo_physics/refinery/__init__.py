"""
thermo_physics.refinery
=======================

5-phase Live-Stream Data Refinery transforming public empirical feeds into
validated thermodynamic Hamiltonians, benchmark payloads, and historical vault snapshots:
  Phase 1 (`collect.py`): Fault-tolerant public API harvester with retry & local cache
  Phase 2 (`clean.py`): Time-alignment, normalization, consensus pruning & binarization
  Phase 3 (`aggregate.py`): Empirical Ising, RNA Potts, QUBO/Ising & AffilAI graph inference
  Phase 4 (`productize.py`): Thermodynamic sampling, Pydantic validation & 5-target export
  Phase 5 (`compound.py`): Idempotent gzip vault archival & benchmark_manifest enrichment
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from thermo_physics.refinery.aggregate import (
    aggregate_affilai_product_graph,
    aggregate_ecological_ising,
    aggregate_market_qubo_ising,
    aggregate_rna_potts_tensor,
)
from thermo_physics.refinery.clean import (
    clean_market_stream,
    clean_rfam_msa,
    clean_shopify_catalogs,
    clean_usgs_stream,
)
from thermo_physics.refinery.collect import (
    collect_all_streams,
    collect_quant_and_polymarket,
    collect_rfam_rna_msa,
    collect_shopify_dtc_catalogs,
    collect_usgs_water_telemetry,
)
from thermo_physics.refinery.compound import (
    BENCHMARK_MANIFEST_TARGETS,
    archive_vault_snapshot,
    compound_empirical_refinery,
    compute_payload_hash,
    update_benchmark_manifests,
)
from thermo_physics.refinery.productize import (
    LIVE_STREAM_TARGETS,
    productize_empirical_streams,
)
from thermo_physics.refinery.schemas import LiveEmpiricalStreamsManifest, SourceMode


def run_live_refinery_pipeline(
    *,
    offline_only: bool = False,
    seed: int = 42,
    cache_dir: Optional[Path] = None,
    stream_targets: Optional[Sequence[Path]] = None,
    vault_dir: Optional[Path] = None,
    manifest_targets: Optional[Sequence[Path]] = None,
) -> Dict[str, Any]:
    """Execute Phases 1 through 5 end-to-end and return summary telemetry."""
    raw = collect_all_streams(offline_only=offline_only, cache_dir=cache_dir)

    cleaned_usgs = clean_usgs_stream(raw["usgs"])
    cleaned_rfam = clean_rfam_msa(raw["rfam"])
    cleaned_market = clean_market_stream(raw["market"])
    cleaned_shopify = clean_shopify_catalogs(raw["shopify"])

    agg_eco = aggregate_ecological_ising(cleaned_usgs)
    agg_rna = aggregate_rna_potts_tensor(cleaned_rfam)
    agg_mkt = aggregate_market_qubo_ising(cleaned_market)
    agg_aff = aggregate_affilai_product_graph(cleaned_shopify)

    prod_res = productize_empirical_streams(
        {
            "ecological_ising": agg_eco,
            "rna_potts": agg_rna,
            "market_qubo": agg_mkt,
            "affilai_product_graph": agg_aff,
        },
        targets=stream_targets,
        seed=seed,
    )

    comp_res = compound_empirical_refinery(
        prod_res["manifest"],
        vault_dir=vault_dir,
        manifest_targets=manifest_targets,
    )

    return {
        "manifest": prod_res["manifest"],
        "byte_size": prod_res["byte_size"],
        "targets_written": prod_res["targets_written"],
        "vault": comp_res["vault"],
        "updated_benchmark_manifests": comp_res["updated_benchmark_manifests"],
    }


__all__ = [
    "BENCHMARK_MANIFEST_TARGETS",
    "LIVE_STREAM_TARGETS",
    "LiveEmpiricalStreamsManifest",
    "SourceMode",
    "aggregate_affilai_product_graph",
    "aggregate_ecological_ising",
    "aggregate_market_qubo_ising",
    "aggregate_rna_potts_tensor",
    "archive_vault_snapshot",
    "clean_market_stream",
    "clean_rfam_msa",
    "clean_shopify_catalogs",
    "clean_usgs_stream",
    "collect_all_streams",
    "collect_quant_and_polymarket",
    "collect_rfam_rna_msa",
    "collect_shopify_dtc_catalogs",
    "collect_usgs_water_telemetry",
    "compound_empirical_refinery",
    "compute_payload_hash",
    "productize_empirical_streams",
    "run_live_refinery_pipeline",
    "update_benchmark_manifests",
]
