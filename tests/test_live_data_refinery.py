"""End-to-end verification test suite for the 5-phase Live-Stream Data Refinery."""

from __future__ import annotations

import gzip
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

import httpx
import numpy as np
import pytest

from thermo_physics.refinery import (
    BENCHMARK_MANIFEST_TARGETS,
    LIVE_STREAM_TARGETS,
    LiveEmpiricalStreamsManifest,
    aggregate_affilai_product_graph,
    aggregate_ecological_ising,
    aggregate_market_qubo_ising,
    aggregate_rna_potts_tensor,
    archive_vault_snapshot,
    clean_market_stream,
    clean_rfam_msa,
    clean_shopify_catalogs,
    clean_usgs_stream,
    collect_all_streams,
    collect_quant_and_polymarket,
    collect_rfam_rna_msa,
    collect_shopify_dtc_catalogs,
    collect_usgs_water_telemetry,
    compute_payload_hash,
    productize_empirical_streams,
    run_live_refinery_pipeline,
    update_benchmark_manifests,
)
from thermo_physics.refinery.collect import SHOPIFY_BRAND_SPECS, YAHOO_TICKERS

REPO_ROOT = Path(__file__).resolve().parent.parent
VALID_SOURCE_MODES = {"live_public_stream", "cached_empirical", "seeded_fallback"}


def _assert_all_finite(obj: Any, path: str = "root") -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        assert math.isfinite(float(obj)), f"Non-finite number at {path}: {obj}"
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            _assert_all_finite(item, f"{path}[{idx}]")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _assert_all_finite(v, f"{path}.{k}")


def _build_mock_transport(*, fail_all: bool = False, retry_once_for_usgs: bool = False) -> httpx.MockTransport:
    usgs_attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if fail_all:
            return httpx.Response(503, json={"error": "upstream unavailable"})

        if "waterservices.usgs.gov" in url:
            usgs_attempts["count"] += 1
            if retry_once_for_usgs and usgs_attempts["count"] == 1:
                return httpx.Response(429, headers={"Retry-After": "0"}, json={"error": "rate limited"})

            sites = ["01463500", "01646500", "01372058", "11447650"]
            params = [("00010", "Temperature, water", "deg C"), ("00095", "Specific conductance", "uS/cm"),
                      ("00300", "Dissolved oxygen", "mg/l"), ("00400", "pH, field", "std units")]
            ts_blocks = []
            for s_idx, site in enumerate(sites):
                for p_idx, (p_cd, p_name, unit) in enumerate(params):
                    obs = [
                        {
                            "dateTime": f"2026-11-18T{k // 2:02d}:{(k % 2) * 30:02d}:00.000Z",
                            "value": f"{10.0 + s_idx + 0.5 * p_idx + 0.2 * math.sin(k * 0.4 + p_idx):.3f}",
                        }
                        for k in range(24)
                    ]
                    ts_blocks.append(
                        {
                            "sourceInfo": {"siteCode": [{"value": site}], "siteName": f"Station {site}"},
                            "variable": {
                                "variableCode": [{"value": p_cd}],
                                "variableName": p_name,
                                "unit": {"unitCode": unit},
                            },
                            "values": [{"value": obs}],
                        }
                    )
            return httpx.Response(200, json={"value": {"timeSeries": ts_blocks}})

        if "rfam.org/family/" in url:
            seqs = []
            base = "GGGAAACCCUUUGGGAAACCCUUUGGGAAACCCUUU"
            for i in range(20):
                chars = list(base)
                if i % 3 == 0:
                    chars[2] = "A"
                    chars[10] = "U"
                seqs.append(f">MOCK_{i}/1-{len(base)}\n{''.join(chars)}")
            return httpx.Response(200, text="\n".join(seqs))

        if "query1.finance.yahoo.com" in url:
            base_ts = 1794960000
            timestamps = [base_ts + 3600 * k for k in range(24)]
            closes = [100.0 + 0.4 * k + math.sin(k * 0.5) for k in range(24)]
            return httpx.Response(
                200,
                json={"chart": {"result": [{"timestamp": timestamps, "indicators": {"quote": [{"close": closes}]}}]}},
            )

        if "gamma-api.polymarket.com" in url:
            markets = [
                {"conditionId": f"cid-{i:02d}", "question": f"Mock Market {i}?", "clobTokenIds": [f"tok-{i:02d}"]}
                for i in range(16)
            ]
            return httpx.Response(200, json=markets)

        if "sampling-simplified-markets" in url:
            return httpx.Response(200, json={"data": []})

        if "clob.polymarket.com/book" in url:
            return httpx.Response(
                200,
                json={
                    "bids": [{"price": "0.51", "size": "1200"}, {"price": "0.50", "size": "800"}],
                    "asks": [{"price": "0.53", "size": "900"}, {"price": "0.54", "size": "700"}],
                },
            )

        if "/products.json" in url:
            return httpx.Response(
                200,
                json={
                    "products": [
                        {
                            "id": 101,
                            "title": "Mock Product",
                            "product_type": "Accessories",
                            "updated_at": "2026-11-18T05:00:00Z",
                            "tags": ["Featured", "Sustainable"],
                            "variants": [
                                {"id": 1001, "sku": "MOCK-01", "price": "48.00", "compare_at_price": "60.00", "available": True},
                                {"id": 1002, "sku": "MOCK-02", "price": "56.00", "compare_at_price": None, "available": True},
                            ],
                        }
                    ]
                },
            )

        return httpx.Response(404)

    return httpx.MockTransport(handler)


# ---------------------------------------------------------------------------
# Phase 1: Collect functions & cache fallback integrity
# ---------------------------------------------------------------------------

def test_phase1_collect_live_cache_and_seeded_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("thermo_physics.refinery.collect.time.sleep", lambda *_args, **_kwargs: None)

    cache_dir = tmp_path / "refinery_cache"
    empty_cache = tmp_path / "empty_cache"

    # 1. Mocked live public stream + retry recovery (429 on first USGS attempt)
    live_transport = _build_mock_transport(retry_once_for_usgs=True)
    with httpx.Client(transport=live_transport, timeout=5.0) as client:
        live_all = collect_all_streams(offline_only=False, cache_dir=cache_dir, client=client)

    for key, expected_file in (
        ("usgs", "usgs_water_latest.json"),
        ("rfam", "rfam_rna_latest.json"),
        ("market", "quant_polymarket_latest.json"),
        ("shopify", "shopify_dtc_latest.json"),
    ):
        assert live_all[key]["source_mode"] == "live_public_stream"
        assert (cache_dir / expected_file).exists()

    assert len(live_all["usgs"]["channels"]) == 16
    assert set(live_all["rfam"]["families"].keys()) == {"RF00005", "RF00001"}
    assert len(live_all["market"]["yahoo_series"]) == len(YAHOO_TICKERS) == 16
    assert len(live_all["market"]["polymarket_books"]) == 16
    assert len(live_all["shopify"]["brands"]) == len(SHOPIFY_BRAND_SPECS) == 15

    # 2. Network failure with populated cache -> cached_empirical fallback
    fail_transport = _build_mock_transport(fail_all=True)
    with httpx.Client(transport=fail_transport, timeout=5.0) as fail_client:
        cached_usgs = collect_usgs_water_telemetry(offline_only=False, cache_dir=cache_dir, client=fail_client)
        cached_rfam = collect_rfam_rna_msa(offline_only=False, cache_dir=cache_dir, client=fail_client)
        cached_mkt = collect_quant_and_polymarket(offline_only=False, cache_dir=cache_dir, client=fail_client)
        cached_shop = collect_shopify_dtc_catalogs(offline_only=False, cache_dir=cache_dir, client=fail_client)

    for stream in (cached_usgs, cached_rfam, cached_mkt, cached_shop):
        assert stream["source_mode"] == "cached_empirical"

    # 3. Offline mode with empty cache -> deterministic seeded_fallback
    seeded_all = collect_all_streams(offline_only=True, cache_dir=empty_cache)
    for stream in seeded_all.values():
        assert stream["source_mode"] == "seeded_fallback"
        assert stream["source_mode"] in VALID_SOURCE_MODES

    assert len(seeded_all["usgs"]["channels"]) == 16
    assert set(seeded_all["rfam"]["families"].keys()) == {"RF00005", "RF00001"}
    assert len(seeded_all["market"]["yahoo_series"]) == 16
    assert len(seeded_all["market"]["polymarket_books"]) == 16
    assert len(seeded_all["shopify"]["brands"]) == 15


# ---------------------------------------------------------------------------
# Phase 2: Sanitization, interpolation, binarization & RNA 5-state pruning
# ---------------------------------------------------------------------------

def test_phase2_clean_usgs_sanitization_interpolation_and_binarization(tmp_path: Path) -> None:
    raw_usgs = collect_usgs_water_telemetry(offline_only=True, cache_dir=tmp_path / "empty")
    # Inject sentinels, NaN, malformed string, and drop intermediate timestamps on channel 0
    obs0 = raw_usgs["channels"][0]["observations"]
    injected = [
        {"dateTime": obs0[0]["dateTime"], "value": -999999},
        {"dateTime": obs0[1]["dateTime"], "value": float("nan")},
        {"dateTime": obs0[2]["dateTime"], "value": "not-a-number"},
    ]
    # Keep sparse subset (every 2nd point after index 3) to test forward-fill/interpolation to T=64
    raw_usgs["channels"][0]["observations"] = injected + obs0[3::2]

    cleaned = clean_usgs_stream(raw_usgs, target_steps=64, window=8)
    assert cleaned["values_aligned"].shape == (64, 16)
    assert cleaned["z_scores"].shape == (64, 16)
    assert cleaned["spins"].shape == (64, 16)
    assert cleaned["binary"].shape == (64, 16)
    assert np.all(np.isfinite(cleaned["values_aligned"]))
    assert np.all(np.isfinite(cleaned["z_scores"]))
    assert np.min(cleaned["values_aligned"]) > -1000.0
    assert set(np.unique(cleaned["spins"]).tolist()).issubset({-1, 1})
    assert set(np.unique(cleaned["binary"]).tolist()).issubset({0, 1})
    assert np.array_equal(cleaned["binary"], (cleaned["spins"] + 1) // 2)


def test_phase2_clean_rfam_msa_five_state_and_gap_pruning() -> None:
    # 30 columns total: first 24 low-gap consensus columns, last 6 columns >50% gap (must be excluded)
    # Plus 2 high-gap sequences (>25% gap in consensus region) that must be pruned
    good_seq_dna = "acgtacgtacgtacgtacgtacgt------"  # lowercase + T->U + trailing gaps
    good_seq_rna = "UGCAUGCAUGCAUGCAUGCAUGCA......"
    noisy_seq_1 = "ACGT----ACGT----ACGT----ACGTAC"   # 12/24 = 50% gap in first 24 cols -> pruned
    noisy_seq_2 = "NNNNNNNNACGTACGTACGTACGTACGTAC"   # 8/24 = 33.3% N(->gap) in first 24 cols -> pruned

    sequences = [{"header": f"good_{i}", "sequence": good_seq_dna if i % 2 == 0 else good_seq_rna} for i in range(12)]
    sequences.append({"header": "prune_1", "sequence": noisy_seq_1})
    sequences.append({"header": "prune_2", "sequence": noisy_seq_2})

    raw_rfam = {
        "source_label": "Unit Test Rfam",
        "families": {
            "RF00005": {
                "family_id": "RF00005",
                "source_mode": "live_public_stream",
                "captured_at_utc": "2026-11-18T00:00:00+00:00",
                "sequences": sequences,
            }
        },
    }

    cleaned = clean_rfam_msa(raw_rfam, consensus_len=24, max_col_gap=0.30, max_seq_gap=0.25)
    assert cleaned["alphabet"] == ["A", "C", "G", "U", "-"]
    assert cleaned["consensus_length"] == 24
    assert cleaned["raw_sequences"] == 14
    assert cleaned["retained_sequences"] == 12
    assert cleaned["consensus_columns"] == list(range(24))

    msa = cleaned["msa_encoded"]
    assert msa.shape == (12, 24)
    assert set(np.unique(msa).tolist()).issubset({0, 1, 2, 3, 4})
    # Row 0 came from good_seq_dna: 't' must map to 'U' (state 3), no gaps in consensus cols
    assert msa[0, :4].tolist() == [0, 1, 2, 3]
    assert np.all(msa != 4)


def test_phase2_clean_market_and_shopify(tmp_path: Path) -> None:
    raw_mkt = collect_quant_and_polymarket(offline_only=True, cache_dir=tmp_path / "empty")
    # Drop two interior hours from ticker 0 to exercise common-grid forward-fill
    raw_mkt["yahoo_series"][0]["timestamps"] = (
        raw_mkt["yahoo_series"][0]["timestamps"][:5] + raw_mkt["yahoo_series"][0]["timestamps"][7:]
    )
    raw_mkt["yahoo_series"][0]["closes"] = (
        raw_mkt["yahoo_series"][0]["closes"][:5] + raw_mkt["yahoo_series"][0]["closes"][7:]
    )
    # Inject malformed zero/negative levels into Polymarket book 0
    raw_mkt["polymarket_books"][0]["bids"].append({"price": "0", "size": "9999"})
    raw_mkt["polymarket_books"][0]["asks"].append({"price": "bad", "size": "-5"})

    cleaned_mkt = clean_market_stream(raw_mkt, rolling_window=12)
    n_hours = len(cleaned_mkt["common_hours"])
    assert cleaned_mkt["aligned_prices"].shape == (n_hours, 16)
    assert cleaned_mkt["log_returns"].shape == (n_hours - 1, 16)
    assert np.all(np.isfinite(cleaned_mkt["aligned_prices"]))
    assert np.all(np.isfinite(cleaned_mkt["log_returns"]))
    assert set(np.unique(cleaned_mkt["spins"]).tolist()).issubset({-1, 1})
    assert set(np.unique(cleaned_mkt["binary"]).tolist()).issubset({0, 1})
    assert np.array_equal(cleaned_mkt["binary"], (cleaned_mkt["spins"] + 1) // 2)
    for bk in cleaned_mkt["polymarket_books"]:
        assert bk["bid_depth"] >= 0.0
        assert bk["ask_depth"] >= 0.0
        assert 0.0 < bk["midpoint"] < 1.0
        assert bk["spread"] >= 0.0
        assert -1.0 <= bk["obi"] <= 1.0

    raw_shop = {
        "source_label": "Unit Test Shopify",
        "source_mode": "live_public_stream",
        "captured_at_utc": "2026-11-18T00:00:00+00:00",
        "brands": [
            {
                "index": 0,
                "brand": "TestBrand",
                "domain": "example.com",
                "vertical": "Apparel",
                "source_mode": "live_public_stream",
                "products": [
                    {
                        "id": 1,
                        "product_type": "  Shoes ",
                        "updated_at": "2026-11-18T00:00:00Z",
                        "tags": ["  BESTSELLER ", "brand::cfid => abcdef1234567890abcdef123456", "Eco-Friendly"],
                        "variants": [
                            {"id": 1, "sku": "dup-1", "price": "0.00", "available": True},
                            {"id": 2, "sku": "dup-2", "price": "-10.00", "available": True},
                            {"id": 3, "sku": "dup-3", "price": "bad", "available": True},
                            {"id": 4, "sku": " SKU-A ", "price": "75.00", "compare_at_price": "90.00", "available": True},
                            {"id": 5, "sku": "sku-a", "price": "80.00", "available": True},
                        ],
                    }
                ],
            }
        ],
    }
    cleaned_shop = clean_shopify_catalogs(raw_shop)
    brand = cleaned_shop["brands"][0]
    assert brand["product_count"] == 1
    assert brand["variant_count"] == 1
    assert brand["prices"] == [75.0]
    assert all(tag == tag.strip().lower() for tag in brand["top_tags"])
    assert all("=>" not in tag for tag in brand["top_tags"])
    assert {"bestseller", "eco-friendly", "shoes"}.issubset(set(brand["top_tags"]))


# ---------------------------------------------------------------------------
# Phase 3: Tensor symmetry, Potts normalization, QUBO->Ising exactness, AffilAI
# ---------------------------------------------------------------------------

def test_phase3_ecological_ising_symmetry_and_ground_state(tmp_path: Path) -> None:
    raw_usgs = collect_usgs_water_telemetry(offline_only=True, cache_dir=tmp_path / "empty")
    cleaned_usgs = clean_usgs_stream(raw_usgs)
    agg = aggregate_ecological_ising(cleaned_usgs)

    j_mat = np.asarray(agg["J"], dtype=np.float64)
    h_vec = np.asarray(agg["h"], dtype=np.float64)
    cov = np.asarray(agg["spin_covariance"], dtype=np.float64)

    assert j_mat.shape == (16, 16)
    assert h_vec.shape == (16,)
    assert cov.shape == (16, 16)
    assert np.all(np.isfinite(j_mat))
    assert np.all(np.isfinite(h_vec))
    assert np.allclose(j_mat, j_mat.T, atol=1e-12)
    assert np.allclose(np.diag(j_mat), 0.0, atol=1e-12)
    assert float(np.max(np.abs(j_mat))) <= 1.0 + 1e-12
    assert 0.0 <= agg["frustration_pct"] <= 100.0
    assert 0.0 <= agg["frustrated_triangles_pct"] <= 100.0

    gs_spins = np.asarray(agg["exact_ground_state_spins"], dtype=np.float64)
    assert set(np.unique(gs_spins).tolist()).issubset({-1.0, 1.0})
    gs_energy = -0.5 * float(gs_spins @ j_mat @ gs_spins) - float(h_vec @ gs_spins)
    assert math.isclose(gs_energy, agg["exact_ground_state_energy"], rel_tol=1e-9, abs_tol=1e-9)

    rng = np.random.default_rng(7)
    for _ in range(25):
        s = rng.choice([-1.0, 1.0], size=16)
        e_rand = -0.5 * float(s @ j_mat @ s) - float(h_vec @ s)
        assert agg["exact_ground_state_energy"] <= e_rand + 1e-9


def test_phase3_rna_potts_sum_to_one_and_connected_covariance(tmp_path: Path) -> None:
    raw_rfam = collect_rfam_rna_msa(offline_only=True, cache_dir=tmp_path / "empty")
    cleaned_rfam = clean_rfam_msa(raw_rfam)
    agg = aggregate_rna_potts_tensor(cleaned_rfam, alpha=0.5)

    f_i = np.asarray(agg["f_i"], dtype=np.float64)
    h_i = np.asarray(agg["h_i"], dtype=np.float64)
    c_ij = np.asarray(agg["C_ij"], dtype=np.float64)
    frob = np.asarray(agg["frobenius_couplings"], dtype=np.float64)

    assert f_i.shape == (24, 5)
    assert h_i.shape == (24, 5)
    assert c_ij.shape == (24, 24, 5, 5)
    assert frob.shape == (24, 24)

    assert np.allclose(f_i.sum(axis=1), 1.0, atol=1e-10)
    assert np.all(f_i > 0.0)
    assert np.allclose(h_i, np.log(f_i), atol=1e-10)

    # Reconstruct f_ij = C_ij + outer(f_i, f_j) and verify against direct MSA pair marginals
    f_ij_reconstructed = c_ij + np.einsum("ia,jb->ijab", f_i, f_i)
    assert np.allclose(f_ij_reconstructed.sum(axis=(-2, -1)), 1.0, atol=1e-10)

    msa = np.asarray(cleaned_rfam["families"]["RF00005"]["msa_encoded"], dtype=np.int64)
    one_hot = np.eye(5, dtype=np.float64)[msa]
    denom = float(msa.shape[0]) + 0.5
    f_i_expected = (np.sum(one_hot, axis=0) + 0.5 / 5.0) / denom
    f_ij_expected = (np.einsum("mia,mjb->ijab", one_hot, one_hot) + 0.5 / 25.0) / denom
    f_i_expected /= np.sum(f_i_expected, axis=1, keepdims=True)
    f_ij_expected /= np.sum(f_ij_expected, axis=(2, 3), keepdims=True)
    c_ij_expected = f_ij_expected - np.einsum("ia,jb->ijab", f_i_expected, f_i_expected)

    assert np.allclose(f_i, f_i_expected, atol=1e-10)
    assert np.allclose(f_ij_expected.sum(axis=(-2, -1)), 1.0, atol=1e-10)
    assert np.allclose(c_ij, c_ij_expected, atol=1e-10)
    assert np.allclose(frob, frob.T, atol=1e-12)
    assert np.allclose(np.diag(frob), 0.0, atol=1e-12)


def test_phase3_market_qubo_to_ising_exact_mathematical_equivalence(tmp_path: Path) -> None:
    raw_mkt = collect_quant_and_polymarket(offline_only=True, cache_dir=tmp_path / "empty")
    cleaned_mkt = clean_market_stream(raw_mkt)
    agg = aggregate_market_qubo_ising(cleaned_mkt, k_target=5)

    q_mat = np.asarray(agg["Q"], dtype=np.float64)
    j_mat = np.asarray(agg["J"], dtype=np.float64)
    h_vec = np.asarray(agg["h"], dtype=np.float64)
    n = q_mat.shape[0]
    assert q_mat.shape == (16, 16)
    assert j_mat.shape == (16, 16)
    assert h_vec.shape == (16,)
    assert np.allclose(q_mat, q_mat.T, atol=1e-12)
    assert np.allclose(j_mat, j_mat.T, atol=1e-12)
    assert np.allclose(np.diag(j_mat), 0.0, atol=1e-12)

    rng = np.random.default_rng(2026)
    diffs: List[float] = []
    for _ in range(32):
        s = rng.choice([-1.0, 1.0], size=n)
        x = (s + 1.0) / 2.0
        e_qubo = float(x @ q_mat @ x)
        e_ising = -0.5 * float(s @ j_mat @ s) - float(h_vec @ s)
        diffs.append(e_qubo - e_ising)

    spread = max(diffs) - min(diffs)
    assert spread < 1e-9, f"QUBO <-> Ising offset not constant across spins: spread={spread}"

    expected_shift = float(agg["qubo_ising_offset"]) - float(agg["penalty_lambda"]) * 25.0
    assert np.allclose(diffs, expected_shift, atol=1e-9)

    sel = agg["exact_optimum"]["selected_indices"]
    assert len(sel) == 5
    assert len(set(sel)) == 5


def test_phase3_affilai_brand_scores_and_affinity_matrix(tmp_path: Path) -> None:
    raw_shop = collect_shopify_dtc_catalogs(offline_only=True, cache_dir=tmp_path / "empty")
    cleaned_shop = clean_shopify_catalogs(raw_shop)
    agg = aggregate_affilai_product_graph(cleaned_shop)

    brands = agg["brands"]
    assert len(brands) == 15
    for b in brands:
        for axis in ("price_elasticity", "variant_depth", "inventory_velocity", "tag_taxonomy_richness"):
            assert 0.0 <= b[axis] <= 1.0
        assert 0.0 <= b["composite_deal_score"] <= 100.0
        assert b["recommended_commission_tier"] in {"PARTNER_15_20", "PARTNER_12_15", "STANDARD_8_12"}

    j_aff = np.asarray(agg["brand_affinity_J"], dtype=np.float64)
    assert j_aff.shape == (15, 15)
    assert np.allclose(j_aff, j_aff.T, atol=1e-12)
    assert np.allclose(np.diag(j_aff), 0.0, atol=1e-12)
    assert np.all((j_aff >= 0.0) & (j_aff <= 1.0))


# ---------------------------------------------------------------------------
# Phase 4 & 5: Productization, Vault Idempotency & Benchmark Manifests
# ---------------------------------------------------------------------------

def test_phase4_and_5_isolated_end_to_end_pipeline_and_idempotency(tmp_path: Path) -> None:
    empty_cache = tmp_path / "cache"
    stream_out = tmp_path / "live_empirical_streams.json"
    vault_dir = tmp_path / "vault"
    bm_copy = tmp_path / "benchmark_manifest.json"
    shutil.copy2(REPO_ROOT / "data" / "benchmark_manifest.json", bm_copy)

    res = run_live_refinery_pipeline(
        offline_only=True,
        seed=42,
        cache_dir=empty_cache,
        stream_targets=[stream_out],
        vault_dir=vault_dir,
        manifest_targets=[bm_copy],
    )

    manifest = res["manifest"]
    validated = LiveEmpiricalStreamsManifest.model_validate(manifest)
    assert validated.schema_version == "1.0.0"
    assert set(manifest.keys()) == {
        "schema_version",
        "generated_at_utc",
        "doctrine_disclosure",
        "ecological_ising",
        "rna_potts",
        "market_qubo",
        "affilai_product_graph",
    }
    _assert_all_finite(manifest)

    # Sokal tau_int >= 0.5 across all 3 thermodynamic sampler benchmarks
    assert manifest["ecological_ising"]["sampler_benchmark"]["tau_int"] >= 0.5
    assert manifest["rna_potts"]["sampler_benchmark"]["tau_int"] >= 0.5
    assert manifest["market_qubo"]["sampler_benchmark"]["tau_int"] >= 0.5

    # Exported rounded market Q, J, h still preserve QUBO <-> Ising equivalence
    q_exp = np.asarray(manifest["market_qubo"]["Q"], dtype=np.float64)
    j_exp = np.asarray(manifest["market_qubo"]["J"], dtype=np.float64)
    h_exp = np.asarray(manifest["market_qubo"]["h"], dtype=np.float64)
    rng = np.random.default_rng(99)
    exp_diffs = []
    for _ in range(16):
        s = rng.choice([-1.0, 1.0], size=16)
        x = (s + 1.0) / 2.0
        exp_diffs.append(float(x @ q_exp @ x) - (-0.5 * float(s @ j_exp @ s) - float(h_exp @ s)))
    assert (max(exp_diffs) - min(exp_diffs)) < 1e-3

    # Phase 5: Vault archival & idempotency check
    snap_path = Path(res["vault"]["snapshot_path"])
    index_path = Path(res["vault"]["index_path"])
    assert snap_path.exists()
    assert index_path.exists()
    with gzip.open(snap_path, "rt", encoding="utf-8") as gz:
        gz_manifest = json.load(gz)
    assert compute_payload_hash(gz_manifest) == compute_payload_hash(manifest)

    # Archiving an identical payload (even with a newer generated_at_utc) must be a no-op duplicate
    dup_manifest = dict(manifest, generated_at_utc="2099-01-01T00:00:00+00:00")
    dup_res = archive_vault_snapshot(dup_manifest, vault_dir=vault_dir)
    assert dup_res["archived"] is False
    assert dup_res["duplicate"] is True
    assert len(list(vault_dir.glob("refinery_snapshot_*.json.gz"))) == 1
    assert len(index_path.read_text(encoding="utf-8").splitlines()) == 1

    # Mutating payload produces a second distinct archive snapshot
    changed_manifest = json.loads(json.dumps(manifest))
    changed_manifest["ecological_ising"]["magnetization"] = round(
        changed_manifest["ecological_ising"]["magnetization"] + 0.01, 5
    )
    new_res = archive_vault_snapshot(changed_manifest, vault_dir=vault_dir)
    assert new_res["archived"] is True
    assert new_res["duplicate"] is False
    assert len(list(vault_dir.glob("refinery_snapshot_*.json.gz"))) == 2
    assert len(index_path.read_text(encoding="utf-8").splitlines()) == 2

    # Benchmark manifest integrity: 16 stations preserved + live_empirical blocks on Stations {2,4,6,7,10,11,13}
    bm = json.loads(bm_copy.read_text(encoding="utf-8"))
    assert "live_empirical_refinery" in bm["metadata"]
    assert bm["metadata"]["live_empirical_refinery"]["stations_enriched"] == [2, 4, 6, 7, 10, 11, 13]
    assert len(bm["stations"]) == 16
    assert "exp12_associative_recall" in bm["experiments"]
    by_id = {s["station_id"]: s for s in bm["stations"]}
    for sid in (2, 4, 6, 7, 10, 11, 13):
        assert isinstance(by_id[sid].get("live_empirical"), dict)


def test_phase4_and_5_persisted_repo_artifacts_and_targets() -> None:
    # Always verify core repo artifacts; on Windows also enforce all sibling targets when present
    core_stream = REPO_ROOT / "data" / "live_empirical_streams.json"
    assert core_stream.exists(), f"Missing {core_stream}"

    if sys.platform == "win32":
        stream_targets = list(LIVE_STREAM_TARGETS)
        manifest_targets = list(BENCHMARK_MANIFEST_TARGETS)
        expected_ecosystem = [
            (Path(r"C:\Users\mcder\AffilAI"), Path(r"C:\Users\mcder\AffilAI\public\data\live_empirical_streams.json")),
            (Path(r"C:\Users\mcder\poseidon_echo_dojo"), Path(r"C:\Users\mcder\poseidon_echo_dojo\data\live_empirical_streams.json")),
            (Path(r"C:\Users\mcder\freshfish-collectr"), Path(r"C:\Users\mcder\freshfish-collectr\public\data\live_empirical_streams.json")),
        ]
        for root_dir, expected_out in expected_ecosystem:
            if root_dir.exists():
                assert expected_out in stream_targets, f"Expected ecosystem target missing from LIVE_STREAM_TARGETS: {expected_out}"
    else:
        stream_targets = [core_stream]
        manifest_targets = [REPO_ROOT / "data" / "benchmark_manifest.json", REPO_ROOT / "docs" / "benchmark_manifest.json"]

    hashes = set()
    raw_bytes_set = set()
    for target in stream_targets:
        assert target.exists(), f"Missing exported stream target: {target}"
        raw_text = target.read_text(encoding="utf-8")
        raw_bytes_set.add(raw_text)
        data = json.loads(raw_text)
        LiveEmpiricalStreamsManifest.model_validate(data)
        _assert_all_finite(data)
        assert data["ecological_ising"]["sampler_benchmark"]["tau_int"] >= 0.5
        assert data["rna_potts"]["sampler_benchmark"]["tau_int"] >= 0.5
        assert data["market_qubo"]["sampler_benchmark"]["tau_int"] >= 0.5
        hashes.add(compute_payload_hash(data))
    assert len(hashes) == 1, "All live_empirical_streams.json targets must share identical payload hash"
    assert len(raw_bytes_set) == 1, "All live_empirical_streams.json targets must be byte-identical"

    vault_dir = REPO_ROOT / "data" / "historical_vault"
    index_path = vault_dir / "vault_index.jsonl"
    assert index_path.exists(), f"Missing {index_path}"
    entries = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(entries) >= 1
    for rec in entries:
        snap_file = vault_dir / rec["snapshot_file"]
        assert snap_file.exists(), f"Missing vault snapshot referenced in index: {snap_file}"
        with gzip.open(snap_file, "rt", encoding="utf-8") as gz:
            gz_payload = json.load(gz)
        LiveEmpiricalStreamsManifest.model_validate(gz_payload)
        assert compute_payload_hash(gz_payload) == rec["payload_hash"]

    required_stations = {2, 4, 6, 7, 10, 11, 13}
    for bm_path in manifest_targets:
        assert bm_path.exists(), f"Missing benchmark_manifest target: {bm_path}"
        bm = json.loads(bm_path.read_text(encoding="utf-8"))
        assert "live_empirical_refinery" in bm.get("metadata", {})
        stations = bm.get("stations", [])
        assert len(stations) >= 16
        enriched_ids = {
            s["station_id"] for s in stations if s.get("station_id") in required_stations and isinstance(s.get("live_empirical"), dict)
        }
        assert enriched_ids == required_stations
