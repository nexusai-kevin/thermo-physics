"""
CLI runner for the 5-Phase Live-Stream Data Refinery.

Usage:
    python scripts/run_live_refinery.py [--offline-only] [--verify] [--seed 42]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from thermo_physics.refinery import (
    BENCHMARK_MANIFEST_TARGETS,
    LIVE_STREAM_TARGETS,
    LiveEmpiricalStreamsManifest,
    compute_payload_hash,
    run_live_refinery_pipeline,
)
from thermo_physics.refinery.productize import resolve_live_stream_targets

console = Console(legacy_windows=False)


def verify_refinery_artifacts(result: Dict[str, Any]) -> List[str]:
    """Run strict integrity assertions across all exported JSON streams, vault snapshots, and manifests."""
    errors: List[str] = []
    manifest = result["manifest"]
    expected_hash = compute_payload_hash(manifest)
    stream_targets = resolve_live_stream_targets() or list(LIVE_STREAM_TARGETS)
    file_sha256s: set[str] = set()

    for target in stream_targets:
        if not target.exists():
            errors.append(f"Missing live stream output: {target}")
            continue
        try:
            raw_bytes = target.read_bytes()
            file_sha256s.add(hashlib.sha256(raw_bytes).hexdigest())
            loaded = json.loads(raw_bytes.decode("utf-8"))
            LiveEmpiricalStreamsManifest.model_validate(loaded)
            if compute_payload_hash(loaded) != expected_hash:
                errors.append(f"Payload hash mismatch at {target}")
        except Exception as exc:
            errors.append(f"Validation error at {target}: {exc}")

    if len(file_sha256s) > 1:
        errors.append(f"Exported stream targets do not share identical raw SHA-256 bytes ({len(file_sha256s)} digests)")

    vault_info = result["vault"]
    snap_path = Path(vault_info["snapshot_path"])
    index_path = Path(vault_info["index_path"])
    if not snap_path.exists():
        errors.append(f"Missing gzipped vault snapshot: {snap_path}")
    else:
        try:
            with gzip.open(snap_path, "rt", encoding="utf-8") as gz:
                gz_data = json.load(gz)
            if compute_payload_hash(gz_data) != expected_hash:
                errors.append(f"Vault snapshot hash mismatch: {snap_path}")
        except Exception as exc:
            errors.append(f"Corrupt vault gzip snapshot {snap_path}: {exc}")

    if not index_path.exists():
        errors.append(f"Missing vault index: {index_path}")

    required_stations = {2, 4, 6, 7, 10, 11, 13}
    for bm_path in BENCHMARK_MANIFEST_TARGETS:
        if not bm_path.exists():
            errors.append(f"Missing benchmark_manifest target: {bm_path}")
            continue
        try:
            bm = json.loads(bm_path.read_text(encoding="utf-8"))
            if "live_empirical_refinery" not in bm.get("metadata", {}):
                errors.append(f"Missing metadata.live_empirical_refinery in {bm_path}")
            stations = bm.get("stations", [])
            if len(stations) < 16:
                errors.append(f"Expected >= 16 stations in {bm_path}, got {len(stations)}")
            found_enriched = {
                s["station_id"]
                for s in stations
                if s.get("station_id") in required_stations and isinstance(s.get("live_empirical"), dict)
            }
            missing = required_stations - found_enriched
            if missing:
                errors.append(f"Missing live_empirical blocks on stations {sorted(missing)} in {bm_path}")
        except Exception as exc:
            errors.append(f"Invalid benchmark manifest {bm_path}: {exc}")

    return errors


def render_summary(result: Dict[str, Any]) -> None:
    manifest = result["manifest"]
    eco = manifest["ecological_ising"]
    rna = manifest["rna_potts"]
    mkt = manifest["market_qubo"]
    aff = manifest["affilai_product_graph"]
    vault = result["vault"]

    console.print(
        Panel.fit(
            f"[bold cyan]Live-Stream Data Refinery — 5-Phase Execution Summary[/bold cyan]\n"
            f"Schema: [bold]{manifest['schema_version']}[/bold] | "
            f"Payload Size: [bold green]{result['byte_size'] / 1024.0:.1f} KB[/bold green] | "
            f"Vault: [bold]{vault['snapshot_file']}[/bold] "
            f"({'deduped' if vault.get('duplicate') else 'archived'})",
            border_style="cyan",
        )
    )

    table = Table(show_header=True, header_style="bold cyan", border_style="bright_blue")
    table.add_column("Domain Stream", style="bold")
    table.add_column("Source Mode", style="magenta")
    table.add_column("Dimensions / Scope", style="cyan")
    table.add_column("Key Thermodynamic / Graph Output", style="yellow")
    table.add_column("tau_int / ESS", justify="right", style="green")

    table.add_row(
        "Ecological Ising (USGS)",
        eco["provenance"]["source_mode"],
        f"{eco['provenance']['station_count']} stations x {eco['provenance']['channel_count']} ch",
        f"E0 = {eco['exact_ground_state_energy']:.4f} | Frust = {eco['frustration_pct']:.1f}%",
        f"{eco['sampler_benchmark']['tau_int']:.2f} / {eco['sampler_benchmark']['ess']:.1f}",
    )
    table.add_row(
        f"RNA Potts ({rna['provenance']['family_id']})",
        rna["provenance"]["source_mode"],
        f"{rna['provenance']['retained_sequences']}/{rna['provenance']['raw_sequences']} seqs, L={rna['provenance']['consensus_length']}, q={rna['provenance']['q_states']}",
        f"E_min = {rna['sampler_benchmark']['min_energy']:.3f} | S = {rna['sampler_benchmark']['entropy_bits']:.2f} bits",
        f"{rna['sampler_benchmark']['tau_int']:.2f} / {rna['sampler_benchmark']['ess']:.1f}",
    )
    table.add_row(
        "Market + Polymarket QUBO",
        mkt["provenance"]["source_mode"],
        f"N={mkt['provenance']['n_assets']}, K={mkt['provenance']['k_target']}, Poly={mkt['provenance']['polymarket_markets_polled']}",
        f"{','.join(mkt['exact_optimum']['selected_tickers'])} | Feas={mkt['sampler_benchmark']['budget_feasibility_pct']:.1f}%",
        f"{mkt['sampler_benchmark']['tau_int']:.2f} / {mkt['sampler_benchmark']['ess']:.1f}",
    )
    top_brand = max(aff["brands"], key=lambda b: b["composite_deal_score"])
    table.add_row(
        "AffilAI Shopify Graph",
        aff["provenance"]["source_mode"],
        f"{aff['provenance']['brand_count']} brands, {aff['provenance']['total_products_polled']} prods, {aff['provenance']['total_variants_polled']} vars",
        f"Top: {top_brand['brand']} ({top_brand['composite_deal_score']:.1f}, {top_brand['recommended_commission_tier']})",
        "15x15 J_aff",
    )
    console.print(table)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execute the 5-Phase Live-Stream Data Refinery.")
    parser.add_argument("--offline-only", action="store_true", help="Skip network polling; use refinery_cache or seeded fallback.")
    parser.add_argument("--verify", action="store_true", help="Validate all exported JSON streams, vault snapshots, and manifests.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic sampler seed.")
    args = parser.parse_args(argv)

    result = run_live_refinery_pipeline(offline_only=args.offline_only, seed=args.seed)
    render_summary(result)

    if args.verify:
        errors = verify_refinery_artifacts(result)
        if errors:
            for err in errors:
                console.print(f"[bold red]VERIFY FAIL:[/bold red] {err}")
            return 1
        stream_count = len(resolve_live_stream_targets() or LIVE_STREAM_TARGETS)
        console.print(
            f"[bold green]✓ Verification passed:[/bold green] All {stream_count} stream targets, "
            f"vault archive, and {len(BENCHMARK_MANIFEST_TARGETS)} benchmark manifests validated."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
