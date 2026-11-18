"""
Phase 5 (COMPOUND): Historical vault archival and benchmark_manifest.json enrichment.

Handles:
  1) Idempotent gzipped JSON snapshots in `data/historical_vault/refinery_snapshot_<TS>.json.gz`
     with SHA-256 deduplication against `vault_index.jsonl`
  2) Non-destructive updates to `benchmark_manifest.json` across all 4 targets, adding
     `metadata.live_empirical_refinery` and `live_empirical` blocks on Stations 2, 4, 6, 7, 10, 11, 13
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_VAULT_DIR = PACKAGE_ROOT / "data" / "historical_vault"

_EXTERNAL_MANIFEST_ROOTS: List[Path] = [
    Path(r"C:\Users\mcder\stellar_logos_seed_swarm\demos\nexus-site"),
    Path(r"C:\Users\mcder\Desktop\nexus-site-temp"),
]


def resolve_benchmark_manifest_targets() -> List[Path]:
    """Return active benchmark_manifest.json targets (repo data/ & docs/, plus sibling web repos on Windows)."""
    targets: List[Path] = [
        PACKAGE_ROOT / "data" / "benchmark_manifest.json",
        PACKAGE_ROOT / "docs" / "benchmark_manifest.json",
    ]
    if sys.platform == "win32":
        for root in _EXTERNAL_MANIFEST_ROOTS:
            if root.exists():
                targets.append(root / "public" / "benchmark_manifest.json")
    return targets


BENCHMARK_MANIFEST_TARGETS: List[Path] = resolve_benchmark_manifest_targets()


def compute_payload_hash(manifest: Dict[str, Any]) -> str:
    """Compute canonical SHA-256 hash ignoring volatile `generated_at_utc` wrapper timestamp."""
    hashable = {
        k: v
        for k, v in manifest.items()
        if k != "generated_at_utc"
    }
    canonical = json.dumps(hashable, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def archive_vault_snapshot(
    manifest: Dict[str, Any],
    *,
    vault_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Idempotently archive compressed `.json.gz` snapshot and append to `vault_index.jsonl`."""
    v_dir = Path(vault_dir) if vault_dir is not None else DEFAULT_VAULT_DIR
    v_dir.mkdir(parents=True, exist_ok=True)
    index_path = v_dir / "vault_index.jsonl"

    payload_hash = compute_payload_hash(manifest)
    existing_records: List[Dict[str, Any]] = []
    if index_path.exists():
        for line in index_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rec = json.loads(stripped)
                if isinstance(rec, dict):
                    existing_records.append(rec)
            except Exception:
                continue

    for rec in existing_records:
        if rec.get("payload_hash") == payload_hash:
            snap_name = rec.get("snapshot_file", "")
            snap_path = v_dir / snap_name
            if snap_name and snap_path.exists():
                return {
                    "archived": False,
                    "duplicate": True,
                    "payload_hash": payload_hash,
                    "snapshot_file": snap_name,
                    "snapshot_path": str(snap_path),
                    "index_path": str(index_path),
                    "total_snapshots": len(existing_records),
                }

    ts_compact = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snap_filename = f"refinery_snapshot_{ts_compact}.json.gz"
    snap_path = v_dir / snap_filename
    if snap_path.exists():
        snap_filename = f"refinery_snapshot_{ts_compact}_{payload_hash[:8]}.json.gz"
        snap_path = v_dir / snap_filename

    serialized = json.dumps(manifest, separators=(",", ":"), ensure_ascii=False)
    tmp_gz = snap_path.with_suffix(".gz.tmp")
    with gzip.open(tmp_gz, "wt", encoding="utf-8", compresslevel=6) as gz:
        gz.write(serialized)
    tmp_gz.replace(snap_path)

    eco = manifest["ecological_ising"]
    rna = manifest["rna_potts"]
    mkt = manifest["market_qubo"]
    aff = manifest["affilai_product_graph"]

    record = {
        "timestamp_utc": manifest["generated_at_utc"],
        "payload_hash": payload_hash,
        "snapshot_file": snap_filename,
        "json_bytes": len(serialized.encode("utf-8")),
        "gzip_bytes": snap_path.stat().st_size,
        "source_modes": {
            "ecological_ising": eco["provenance"]["source_mode"],
            "rna_potts": rna["provenance"]["source_mode"],
            "market_qubo": mkt["provenance"]["source_mode"],
            "affilai_product_graph": aff["provenance"]["source_mode"],
        },
        "summary": {
            "eco_exact_e0": eco["exact_ground_state_energy"],
            "eco_tau_int": eco["sampler_benchmark"]["tau_int"],
            "rna_retained_seqs": rna["provenance"]["retained_sequences"],
            "rna_tau_int": rna["sampler_benchmark"]["tau_int"],
            "market_selected": mkt["exact_optimum"]["selected_tickers"],
            "market_tau_int": mkt["sampler_benchmark"]["tau_int"],
            "affil_top_brand": max(aff["brands"], key=lambda b: b["composite_deal_score"])["brand"],
        },
    }

    with open(index_path, "a", encoding="utf-8") as idx_f:
        idx_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "archived": True,
        "duplicate": False,
        "payload_hash": payload_hash,
        "snapshot_file": snap_filename,
        "snapshot_path": str(snap_path),
        "index_path": str(index_path),
        "gzip_bytes": record["gzip_bytes"],
        "total_snapshots": len(existing_records) + 1,
    }


def _build_station_live_blocks(manifest: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """Build targeted `live_empirical` enrichment blocks for Stations 2, 4, 6, 7, 10, 11, 13."""
    eco = manifest["ecological_ising"]
    rna = manifest["rna_potts"]
    mkt = manifest["market_qubo"]
    aff = manifest["affilai_product_graph"]

    top_brand = max(aff["brands"], key=lambda b: b["composite_deal_score"])
    aff_mat = aff["brand_affinity_J"]
    n_b = len(aff_mat)
    off_diag_vals = [aff_mat[i][j] for i in range(n_b) for j in range(i + 1, n_b)]
    mean_aff = round(sum(off_diag_vals) / max(1, len(off_diag_vals)), 4)

    top_contact = rna["top_contacts"][0] if rna["top_contacts"] else {}

    return {
        2: {
            "stream_key": "ecological_ising",
            "source_label": eco["provenance"]["source_label"],
            "source_mode": eco["provenance"]["source_mode"],
            "captured_at_utc": eco["provenance"]["captured_at_utc"],
            "station_count": eco["provenance"]["station_count"],
            "channel_count": eco["provenance"]["channel_count"],
            "magnetization": eco["magnetization"],
            "frustration_pct": eco["frustration_pct"],
            "exact_ground_state_energy": eco["exact_ground_state_energy"],
            "tau_int": eco["sampler_benchmark"]["tau_int"],
            "ess": eco["sampler_benchmark"]["ess"],
        },
        4: {
            "stream_key": "rna_potts",
            "source_label": rna["provenance"]["source_label"],
            "source_mode": rna["provenance"]["source_mode"],
            "captured_at_utc": rna["provenance"]["captured_at_utc"],
            "family_id": rna["provenance"]["family_id"],
            "consensus_length": rna["provenance"]["consensus_length"],
            "q_states": rna["provenance"]["q_states"],
            "retained_sequences": rna["provenance"]["retained_sequences"],
            "top_contact_frobenius": top_contact.get("frobenius_score", 0.0),
            "tau_int": rna["sampler_benchmark"]["tau_int"],
            "ess": rna["sampler_benchmark"]["ess"],
        },
        6: {
            "stream_key": "affilai_product_graph",
            "source_label": aff["provenance"]["source_label"],
            "source_mode": aff["provenance"]["source_mode"],
            "captured_at_utc": aff["provenance"]["captured_at_utc"],
            "brand_count": aff["provenance"]["brand_count"],
            "total_products_polled": aff["provenance"]["total_products_polled"],
            "total_variants_polled": aff["provenance"]["total_variants_polled"],
            "top_partner_brand": top_brand["brand"],
            "top_composite_deal_score": top_brand["composite_deal_score"],
            "mean_affinity_coupling": mean_aff,
        },
        7: {
            "stream_key": "market_qubo",
            "source_label": mkt["provenance"]["source_label"],
            "source_mode": mkt["provenance"]["source_mode"],
            "captured_at_utc": mkt["provenance"]["captured_at_utc"],
            "n_assets": mkt["provenance"]["n_assets"],
            "k_target": mkt["provenance"]["k_target"],
            "polymarket_markets_polled": mkt["provenance"]["polymarket_markets_polled"],
            "selected_tickers": mkt["exact_optimum"]["selected_tickers"],
            "exact_qubo_energy": mkt["exact_optimum"]["qubo_energy"],
            "expected_return": mkt["exact_optimum"]["expected_return"],
            "volatility": mkt["exact_optimum"]["volatility"],
            "budget_feasibility_pct": mkt["sampler_benchmark"]["budget_feasibility_pct"],
            "pareto_match_pct": mkt["sampler_benchmark"]["pareto_match_pct"],
            "tau_int": mkt["sampler_benchmark"]["tau_int"],
            "ess": mkt["sampler_benchmark"]["ess"],
        },
        10: {
            "stream_key": "ecological_ising",
            "source_label": eco["provenance"]["source_label"],
            "source_mode": eco["provenance"]["source_mode"],
            "captured_at_utc": eco["provenance"]["captured_at_utc"],
            "variables": eco["provenance"]["channel_count"],
            "frustration_pct": eco["frustration_pct"],
            "frustrated_triangles_pct": eco["frustrated_triangles_pct"],
            "exact_ground_state_energy": eco["exact_ground_state_energy"],
            "min_sampled_energy": eco["sampler_benchmark"]["min_energy"],
            "tau_int": eco["sampler_benchmark"]["tau_int"],
            "ess": eco["sampler_benchmark"]["ess"],
        },
        11: {
            "stream_key": "rna_potts",
            "source_label": rna["provenance"]["source_label"],
            "source_mode": rna["provenance"]["source_mode"],
            "captured_at_utc": rna["provenance"]["captured_at_utc"],
            "family_id": rna["provenance"]["family_id"],
            "raw_sequences": rna["provenance"]["raw_sequences"],
            "retained_sequences": rna["provenance"]["retained_sequences"],
            "consensus_length": rna["provenance"]["consensus_length"],
            "entropy_bits": rna["sampler_benchmark"]["entropy_bits"],
            "top_contacts": rna["top_contacts"][:4],
            "tau_int": rna["sampler_benchmark"]["tau_int"],
            "ess": rna["sampler_benchmark"]["ess"],
        },
        13: {
            "stream_key": "rna_potts",
            "source_label": rna["provenance"]["source_label"],
            "source_mode": rna["provenance"]["source_mode"],
            "captured_at_utc": rna["provenance"]["captured_at_utc"],
            "family_id": rna["provenance"]["family_id"],
            "q_states": rna["provenance"]["q_states"],
            "entropy_bits": rna["sampler_benchmark"]["entropy_bits"],
            "min_energy": rna["sampler_benchmark"]["min_energy"],
            "tau_int": rna["sampler_benchmark"]["tau_int"],
            "ess": rna["sampler_benchmark"]["ess"],
        },
    }


def update_benchmark_manifests(
    manifest: Dict[str, Any],
    vault_info: Optional[Dict[str, Any]] = None,
    *,
    targets: Optional[Sequence[Path]] = None,
) -> List[str]:
    """Enrich `benchmark_manifest.json` files non-destructively with refinery metadata & live station blocks."""
    out_targets = list(targets) if targets is not None else list(BENCHMARK_MANIFEST_TARGETS)
    payload_hash = compute_payload_hash(manifest)
    station_blocks = _build_station_live_blocks(manifest)

    refinery_meta = {
        "schema_version": manifest["schema_version"],
        "generated_at_utc": manifest["generated_at_utc"],
        "payload_hash": payload_hash,
        "stream_json_path": "/data/live_empirical_streams.json",
        "stations_enriched": sorted(station_blocks.keys()),
        "source_modes": {
            "ecological_ising": manifest["ecological_ising"]["provenance"]["source_mode"],
            "rna_potts": manifest["rna_potts"]["provenance"]["source_mode"],
            "market_qubo": manifest["market_qubo"]["provenance"]["source_mode"],
            "affilai_product_graph": manifest["affilai_product_graph"]["provenance"]["source_mode"],
        },
        "vault_snapshot": (vault_info or {}).get("snapshot_file"),
    }

    updated_paths: List[str] = []
    for path in out_targets:
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            bm = json.load(f)

        bm.setdefault("metadata", {})["live_empirical_refinery"] = refinery_meta
        for stn in bm.get("stations", []):
            sid = stn.get("station_id")
            if sid in station_blocks:
                stn["live_empirical"] = station_blocks[sid]

        tmp_path = path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(bm, f, indent=2, ensure_ascii=False)
        tmp_path.replace(path)
        updated_paths.append(str(path))

    return updated_paths


def compound_empirical_refinery(
    manifest: Dict[str, Any],
    *,
    vault_dir: Optional[Path] = None,
    manifest_targets: Optional[Sequence[Path]] = None,
) -> Dict[str, Any]:
    """Execute Phase 5 archival + benchmark manifest enrichment."""
    vault_info = archive_vault_snapshot(manifest, vault_dir=vault_dir)
    updated_manifests = update_benchmark_manifests(
        manifest,
        vault_info=vault_info,
        targets=manifest_targets,
    )
    return {
        "vault": vault_info,
        "updated_benchmark_manifests": updated_manifests,
    }
