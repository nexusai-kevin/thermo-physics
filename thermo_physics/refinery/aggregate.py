"""
Phase 3 (AGGREGATE): Empirical tensor inference & exact thermodynamic benchmarks.

Converts cleaned matrices into physical energy-based models:
  - `aggregate_ecological_ising`: empirical spin covariance, Tikhonov inverse-Ising (J_ij, h_i),
    frustration %, and exact 2^16 ground state enumeration
  - `aggregate_rna_potts_tensor`: pseudocount alpha=0.5 site marginals f_i(a), pairwise f_ij(a,b),
    connected covariance tensor C_ij(a,b), fields h_i(a), and 24x24 Frobenius coupling matrix F_ij
  - `aggregate_market_qubo_ising`: annualized (mu, Sigma) + Polymarket OBI penalties -> QUBO Q ->
    exact Ising (J, h, offset) + exact C(16,5)=4,368 portfolio enumeration
  - `aggregate_affilai_product_graph`: 4 normalized scoring axes, composite deal score,
    recommended commission tier, and 15x15 brand affinity coupling matrix
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Stream 1 Aggregator: Ecological Inverse-Ising (N=16, 2^16 exact search)
# ---------------------------------------------------------------------------

def aggregate_ecological_ising(
    cleaned_usgs: Dict[str, Any],
    tikhonov_reg: float = 0.15,
) -> Dict[str, Any]:
    """Infer regularized mean-field Ising (J, h) from aligned spin trajectories and solve 2^16 exactly."""
    spins = np.asarray(cleaned_usgs["spins"], dtype=np.float64)
    t_steps, n_nodes = spins.shape
    if n_nodes > 20:
        raise ValueError(f"Brute-force ecological Ising supports N<=20, got {n_nodes}")

    m = np.mean(spins, axis=0)
    cov = (spins.T @ spins) / float(t_steps) - np.outer(m, m)
    cov = 0.5 * (cov + cov.T)

    theta = np.linalg.inv(cov + tikhonov_reg * np.eye(n_nodes))
    j_raw = -theta
    j_sym = 0.5 * (j_raw + j_raw.T)
    np.fill_diagonal(j_sym, 0.0)

    max_abs = float(np.max(np.abs(j_sym)))
    j_mat = j_sym / max_abs if max_abs > 1e-12 else j_sym
    np.fill_diagonal(j_mat, 0.0)

    h_vec = np.arctanh(np.clip(m, -0.95, 0.95)) - (j_mat @ m)

    # Exact brute-force enumeration over 2^N states
    n_states = 1 << n_nodes
    indices = np.arange(n_states, dtype=np.uint32)[:, None]
    shifts = np.arange(n_nodes, dtype=np.uint32)[None, :]
    all_spins = 2.0 * ((indices >> shifts) & 1).astype(np.float64) - 1.0
    energies = -0.5 * np.sum((all_spins @ j_mat) * all_spins, axis=1) - (all_spins @ h_vec)
    min_idx = int(np.argmin(energies))
    exact_e0 = float(energies[min_idx])
    exact_spins = all_spins[min_idx].astype(int).tolist()

    # Edge frustration in ground state & geometric triangle frustration
    frustrated_edges = 0
    active_edges = 0
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            j_val = j_mat[i, j]
            if abs(j_val) > 1e-8:
                active_edges += 1
                if j_val * exact_spins[i] * exact_spins[j] < 0.0:
                    frustrated_edges += 1
    frustration_pct = (100.0 * frustrated_edges / active_edges) if active_edges > 0 else 0.0

    frustrated_tri = 0
    active_tri = 0
    for i, j, k in combinations(range(n_nodes), 3):
        prod = j_mat[i, j] * j_mat[j, k] * j_mat[k, i]
        if abs(prod) > 1e-12:
            active_tri += 1
            if prod < 0.0:
                frustrated_tri += 1
    frustrated_triangles_pct = (100.0 * frustrated_tri / active_tri) if active_tri > 0 else 0.0

    return {
        "provenance": {
            "source_label": cleaned_usgs["source_label"],
            "source_mode": cleaned_usgs["source_mode"],
            "captured_at_utc": cleaned_usgs["captured_at_utc"],
            "station_count": cleaned_usgs["station_count"],
            "channel_count": cleaned_usgs["channel_count"],
        },
        "nodes": cleaned_usgs["channels"],
        "J": j_mat.tolist(),
        "h": h_vec.tolist(),
        "spin_covariance": cov.tolist(),
        "magnetization": float(np.mean(m)),
        "frustration_pct": float(frustration_pct),
        "frustrated_triangles_pct": float(frustrated_triangles_pct),
        "exact_ground_state_energy": exact_e0,
        "exact_ground_state_spins": exact_spins,
    }


# ---------------------------------------------------------------------------
# Stream 2 Aggregator: RNA Co-Evolution Potts Tensor (L=24, q=5)
# ---------------------------------------------------------------------------

def _aggregate_single_potts(cleaned_family: Dict[str, Any], alpha: float = 0.5) -> Dict[str, Any]:
    msa = np.asarray(cleaned_family["msa_encoded"], dtype=np.int64)
    n_seqs, l_len = msa.shape
    alphabet = cleaned_family.get("alphabet", ["A", "C", "G", "U", "-"])
    q = len(alphabet)

    one_hot = np.eye(q, dtype=np.float64)[msa]  # (M, L, Q)
    counts_1 = np.sum(one_hot, axis=0)          # (L, Q)
    counts_2 = np.einsum("mia,mjb->ijab", one_hot, one_hot)  # (L, L, Q, Q)

    denom = float(n_seqs) + alpha
    f_i = (counts_1 + (alpha / q)) / denom
    f_ij = (counts_2 + (alpha / (q * q))) / denom

    # Enforce exact normalization to absorb floating-point rounding
    f_i = f_i / np.sum(f_i, axis=1, keepdims=True)
    f_ij = f_ij / np.sum(f_ij, axis=(2, 3), keepdims=True)

    c_ij = f_ij - np.einsum("ia,jb->ijab", f_i, f_i)
    h_i = np.log(np.maximum(f_i, 1e-12))

    frobenius = np.sqrt(np.sum(c_ij ** 2, axis=(2, 3)))
    frobenius = 0.5 * (frobenius + frobenius.T)
    np.fill_diagonal(frobenius, 0.0)

    cons_cols = cleaned_family.get("consensus_columns", list(range(l_len)))
    contact_candidates: List[Dict[str, Any]] = []
    for i in range(l_len):
        for j in range(i + 2, l_len):
            score = float(frobenius[i, j])
            best_pair_flat = int(np.argmax(counts_2[i, j]))
            a_idx, b_idx = divmod(best_pair_flat, q)
            contact_candidates.append(
                {
                    "i": i,
                    "j": j,
                    "pos_i": int(cons_cols[i]),
                    "pos_j": int(cons_cols[j]),
                    "frobenius_score": score,
                    "consensus_pair": f"{alphabet[a_idx]}-{alphabet[b_idx]}",
                }
            )
    contact_candidates.sort(key=lambda item: item["frobenius_score"], reverse=True)

    return {
        "provenance": {
            "source_label": cleaned_family.get("source_label", f"Rfam {cleaned_family['family_id']} Alignment"),
            "source_mode": cleaned_family["source_mode"],
            "captured_at_utc": cleaned_family["captured_at_utc"],
            "family_id": cleaned_family["family_id"],
            "raw_sequences": int(cleaned_family["raw_sequences"]),
            "retained_sequences": int(cleaned_family["retained_sequences"]),
            "consensus_length": int(l_len),
            "q_states": int(q),
        },
        "alphabet": list(alphabet),
        "f_i": f_i.tolist(),
        "h_i": h_i.tolist(),
        "frobenius_couplings": frobenius.tolist(),
        "C_ij": c_ij.tolist(),
        "top_contacts": contact_candidates[:12],
        "consensus_columns": [int(c) for c in cons_cols],
    }


def aggregate_rna_potts_tensor(
    cleaned_msa: Dict[str, Any],
    alpha: float = 0.5,
) -> Dict[str, Any]:
    """Compute pseudocounted RNA Potts site/pair marginals, C_ij(a,b), h_i(a), and Frobenius matrix."""
    if "families" in cleaned_msa and isinstance(cleaned_msa["families"], dict):
        primary_id = cleaned_msa.get("primary_family_id", "RF00005")
        primary_cleaned = cleaned_msa["families"][primary_id]
        primary_cleaned = dict(primary_cleaned, source_label=cleaned_msa.get("source_label", "Rfam MSA"))
        result = _aggregate_single_potts(primary_cleaned, alpha=alpha)

        other_ids = [fid for fid in cleaned_msa["families"] if fid != primary_id]
        if other_ids:
            sec_id = other_ids[0]
            sec_agg = _aggregate_single_potts(cleaned_msa["families"][sec_id], alpha=alpha)
            result["secondary_family"] = {
                "family_id": sec_id,
                "source_mode": sec_agg["provenance"]["source_mode"],
                "raw_sequences": sec_agg["provenance"]["raw_sequences"],
                "retained_sequences": sec_agg["provenance"]["retained_sequences"],
                "consensus_length": sec_agg["provenance"]["consensus_length"],
                "top_contacts": sec_agg["top_contacts"][:6],
            }
        return result

    return _aggregate_single_potts(cleaned_msa, alpha=alpha)


# ---------------------------------------------------------------------------
# Stream 3 Aggregator: Market + Polymarket OBI QUBO / Exact Ising Mapping
# ---------------------------------------------------------------------------

def aggregate_market_qubo_ising(
    cleaned_market: Dict[str, Any],
    k_target: int = 5,
    risk_aversion: float = 1.0,
) -> Dict[str, Any]:
    """Build Markowitz+OBI QUBO Q, map exactly to Ising (J, h, offset), and solve C(16,5) exactly."""
    log_rets = np.asarray(cleaned_market["log_returns"], dtype=np.float64)
    aligned_prices = np.asarray(cleaned_market["aligned_prices"], dtype=np.float64)
    tickers: List[str] = list(cleaned_market["tickers"])
    books: List[Dict[str, Any]] = list(cleaned_market["polymarket_books"])
    n_assets = len(tickers)

    span_hours = max(1, len(cleaned_market["common_hours"]) - 1)
    periods_per_year = (365.25 * 24.0) / float(span_hours)

    # Empirical annualized return (smoothly bounded via tanh to keep 5-day swings well-scaled)
    raw_ann_returns = np.sum(log_rets, axis=0) * periods_per_year
    mu = 0.45 * np.tanh(raw_ann_returns / 1.5)

    # Annualized positive-definite covariance matrix
    hourly_cov = np.cov(log_rets, rowvar=False)
    if hourly_cov.ndim == 0:
        hourly_cov = np.array([[float(hourly_cov)]], dtype=np.float64)
    sigma = hourly_cov * float(span_hours) * periods_per_year
    sigma = 0.5 * (sigma + sigma.T)
    min_eig = float(np.min(np.linalg.eigvalsh(sigma)))
    ridge = max(1e-3, -min_eig + 1e-3)
    sigma = sigma + ridge * np.eye(n_assets)
    ann_vols = np.sqrt(np.maximum(np.diag(sigma), 1e-8))

    # Map Polymarket OBI penalties onto the 16 assets
    pi_vec = np.zeros(n_assets, dtype=np.float64)
    obis = np.zeros(n_assets, dtype=np.float64)
    spreads = np.zeros(n_assets, dtype=np.float64)
    for i in range(n_assets):
        bk = books[i % len(books)]
        obis[i] = float(bk["obi"])
        spreads[i] = float(bk["spread"])
        pi_vec[i] = 0.03 * (1.0 - obis[i]) + 0.04 * spreads[i]

    pi_mat = 0.015 * np.outer(obis, obis)
    np.fill_diagonal(pi_mat, 0.0)
    pi_mat = 0.5 * (pi_mat + pi_mat.T)

    # Cardinality penalty scaled to dominate linear return magnitudes cleanly
    penalty_lambda = max(0.65, float(1.8 * np.max(np.abs(mu)) + 0.5))

    q_mat = risk_aversion * sigma + pi_mat + penalty_lambda * np.ones((n_assets, n_assets), dtype=np.float64)
    np.fill_diagonal(
        q_mat,
        risk_aversion * np.diag(sigma) - mu + pi_vec + penalty_lambda * (1.0 - 2.0 * k_target),
    )
    q_mat = 0.5 * (q_mat + q_mat.T)

    # Exact Ising mapping matching qubo_portfolio.py and sampler_adapter.py:
    # E_qubo(x) = x^T Q x + penalty_lambda * K^2 == -0.5 s^T J s - h^T s + const_offset
    j_mat = -0.5 * q_mat
    np.fill_diagonal(j_mat, 0.0)
    h_vec = np.zeros(n_assets, dtype=np.float64)
    for i in range(n_assets):
        off_diag_sum = float(np.sum(q_mat[i, :]) - q_mat[i, i])
        h_vec[i] = -0.5 * q_mat[i, i] - 0.5 * off_diag_sum

    const_offset = (
        penalty_lambda * float(k_target ** 2)
        + 0.5 * float(np.sum(np.diag(q_mat)))
        + 0.25 * float(np.sum(q_mat) - np.sum(np.diag(q_mat)))
    )

    # Verify QUBO <-> Ising equivalence across multiple states
    rng_chk = np.random.default_rng(123)
    for _ in range(16):
        x_chk = rng_chk.integers(0, 2, size=n_assets).astype(np.float64)
        s_chk = 2.0 * x_chk - 1.0
        e_qubo_direct = (
            risk_aversion * float(x_chk @ sigma @ x_chk)
            - float(mu @ x_chk)
            + float(pi_vec @ x_chk)
            + float(x_chk @ pi_mat @ x_chk)
            + penalty_lambda * ((float(np.sum(x_chk)) - k_target) ** 2)
        )
        e_ising_mapped = -0.5 * float(s_chk @ j_mat @ s_chk) - float(h_vec @ s_chk) + const_offset
        if abs(e_qubo_direct - e_ising_mapped) > 1e-7:
            raise AssertionError(
                f"QUBO-to-Ising mapping mismatch: direct={e_qubo_direct}, mapped={e_ising_mapped}"
            )

    # Exact brute-force enumeration over C(N, K) feasible portfolios
    combos = list(combinations(range(n_assets), k_target))
    x_feasible = np.zeros((len(combos), n_assets), dtype=np.float64)
    for row_idx, comb in enumerate(combos):
        x_feasible[row_idx, list(comb)] = 1.0

    objs = (
        np.sum((x_feasible @ q_mat) * x_feasible, axis=1)
        + penalty_lambda * float(k_target ** 2)
    )
    best_row = int(np.argmin(objs))
    best_x = x_feasible[best_row]
    sel_indices = np.where(best_x > 0.5)[0].astype(int).tolist()
    sel_tickers = [tickers[i] for i in sel_indices]
    exact_ret = float(mu @ best_x)
    exact_vol = float(np.sqrt(max(0.0, float(best_x @ sigma @ best_x))))
    exact_obj = float(objs[best_row])

    assets_out: List[Dict[str, Any]] = []
    for i, ticker in enumerate(tickers):
        bk = books[i % len(books)]
        assets_out.append(
            {
                "index": i,
                "ticker": ticker,
                "latest_price": float(aligned_prices[-1, i]),
                "annualized_return": float(mu[i]),
                "annualized_vol": float(ann_vols[i]),
                "polymarket_market": bk["question"],
                "obi": float(bk["obi"]),
                "spread": float(bk["spread"]),
                "obi_penalty": float(pi_vec[i]),
                "bid_depth": float(bk["bid_depth"]),
                "ask_depth": float(bk["ask_depth"]),
                "midpoint": float(bk["midpoint"]),
            }
        )

    return {
        "provenance": {
            "source_label": cleaned_market["source_label"],
            "source_mode": cleaned_market["source_mode"],
            "captured_at_utc": cleaned_market["captured_at_utc"],
            "n_assets": n_assets,
            "k_target": k_target,
            "polymarket_markets_polled": len(books),
        },
        "assets": assets_out,
        "Q": q_mat.tolist(),
        "J": j_mat.tolist(),
        "h": h_vec.tolist(),
        "qubo_ising_offset": float(const_offset),
        "penalty_lambda": float(penalty_lambda),
        "exact_optimum": {
            "selected_indices": sel_indices,
            "selected_tickers": sel_tickers,
            "qubo_energy": exact_obj,
            "expected_return": exact_ret,
            "volatility": exact_vol,
        },
    }


# ---------------------------------------------------------------------------
# Stream 4 Aggregator: AffilAI Product Graph (15 DTC Brands)
# ---------------------------------------------------------------------------

def _bounded_minmax(values: List[float], lo: float = 0.25, hi: float = 0.96) -> List[float]:
    arr = np.asarray(values, dtype=np.float64)
    v_min = float(np.min(arr))
    v_max = float(np.max(arr))
    if v_max - v_min <= 1e-12:
        return [0.5 * (lo + hi)] * len(values)
    scaled = (arr - v_min) / (v_max - v_min)
    return [float(np.clip(lo + s * (hi - lo), 0.0, 1.0)) for s in scaled]


def aggregate_affilai_product_graph(cleaned_shopify: Dict[str, Any]) -> Dict[str, Any]:
    """Score 15 Shopify DTC brands and build the 15x15 brand affinity coupling matrix."""
    raw_brands = cleaned_shopify.get("brands", [])
    n_brands = len(raw_brands)
    if n_brands == 0:
        raise ValueError("cleaned_shopify contains no brands")

    mean_prices: List[float] = []
    spread_ratios: List[float] = []
    raw_elasticity: List[float] = []
    raw_depth: List[float] = []
    raw_velocity: List[float] = []
    raw_taxonomy: List[float] = []

    all_updated = [ts for b in raw_brands for ts in b.get("updated_timestamps", [])]
    max_ts = max(all_updated) if all_updated else 0.0

    for b in raw_brands:
        prices = np.asarray(b["prices"], dtype=np.float64)
        mean_p = float(np.mean(prices))
        p_lo, p_hi = float(np.percentile(prices, 10)), float(np.percentile(prices, 90))
        spread_ratio = min(5.0, (p_hi - p_lo) / max(mean_p, 1.0))
        disc_list = b.get("discount_ratios", [])
        mean_disc = float(np.mean(disc_list)) if disc_list else 0.12

        mean_prices.append(mean_p)
        spread_ratios.append(spread_ratio)
        raw_elasticity.append(0.65 * min(1.0, spread_ratio / 2.2) + 0.35 * min(1.0, mean_disc * 2.5))

        p_cnt = max(1, int(b["product_count"]))
        v_cnt = max(1, int(b["variant_count"]))
        raw_depth.append(0.55 * min(1.0, (v_cnt / p_cnt) / 6.0) + 0.45 * min(1.0, np.log1p(v_cnt) / np.log1p(150.0)))

        avail = b.get("available_flags", [True])
        avail_frac = float(np.mean([1.0 if f else 0.0 for f in avail]))
        upds = b.get("updated_timestamps", [])
        if upds and max_ts > 0.0:
            median_age_days = max(0.0, (max_ts - float(np.median(upds))) / 86400.0)
            recency_score = float(np.exp(-median_age_days / 30.0))
        else:
            recency_score = 0.70
        raw_velocity.append(0.60 * avail_frac + 0.40 * recency_score)

        n_tags = len(b.get("tag_counts", {}))
        raw_taxonomy.append(0.55 * min(1.0, n_tags / 35.0) + 0.45 * min(1.0, (n_tags / p_cnt) / 3.0))

    norm_elast = _bounded_minmax(raw_elasticity, lo=0.28, hi=0.95)
    norm_depth = _bounded_minmax(raw_depth, lo=0.26, hi=0.96)
    norm_vel = _bounded_minmax(raw_velocity, lo=0.32, hi=0.96)
    norm_tax = _bounded_minmax(raw_taxonomy, lo=0.25, hi=0.95)

    brands_out: List[Dict[str, Any]] = []
    tag_sets: List[set[str]] = []

    for idx, b in enumerate(raw_brands):
        pe = norm_elast[idx]
        vd = norm_depth[idx]
        iv = norm_vel[idx]
        tr = norm_tax[idx]
        composite = round(100.0 * (0.28 * pe + 0.27 * vd + 0.25 * iv + 0.20 * tr), 2)
        if composite >= 58.0:
            tier = "PARTNER_15_20"
        elif composite >= 42.0:
            tier = "PARTNER_12_15"
        else:
            tier = "STANDARD_8_12"

        featured_products = list(b.get("featured_products", []))
        hero_product = featured_products[0] if featured_products else {}
        hero_image_url = str(hero_product.get("image_url", ""))
        hero_product_title = str(hero_product.get("title") or f"{b['brand']} Signature Drop")
        hero_product_price = float(hero_product.get("price_usd", mean_prices[idx]))
        hero_product_url = str(hero_product.get("product_url") or f"https://{b['domain']}")

        tag_sets.append(set(b.get("tag_counts", {}).keys()))
        brands_out.append(
            {
                "index": idx,
                "brand": b["brand"],
                "vertical": b["vertical"],
                "domain": b["domain"],
                "source_mode": b["source_mode"],
                "product_count": int(b["product_count"]),
                "variant_count": int(b["variant_count"]),
                "mean_price_usd": float(mean_prices[idx]),
                "price_spread_ratio": float(spread_ratios[idx]),
                "price_elasticity": pe,
                "variant_depth": vd,
                "inventory_velocity": iv,
                "tag_taxonomy_richness": tr,
                "composite_deal_score": composite,
                "recommended_commission_tier": tier,
                "top_tags": list(b.get("top_tags", [])),
                "hero_image_url": hero_image_url,
                "hero_product_title": hero_product_title,
                "hero_product_price_usd": hero_product_price,
                "hero_product_url": hero_product_url,
                "featured_products": featured_products,
            }
        )

    # 15x15 symmetric brand affinity coupling matrix (zero diagonal, values in [0, 1])
    affinity = np.zeros((n_brands, n_brands), dtype=np.float64)
    for i in range(n_brands):
        vert_tokens_i = set(brands_out[i]["vertical"].lower().replace("&", " ").split())
        for j in range(i + 1, n_brands):
            union = tag_sets[i] | tag_sets[j]
            jaccard = (len(tag_sets[i] & tag_sets[j]) / len(union)) if union else 0.0
            price_sim = float(np.exp(-abs(np.log(max(mean_prices[i], 1.0)) - np.log(max(mean_prices[j], 1.0)))))
            vert_tokens_j = set(brands_out[j]["vertical"].lower().replace("&", " ").split())
            vert_overlap = len(vert_tokens_i & vert_tokens_j) / max(1, len(vert_tokens_i | vert_tokens_j))
            score = float(np.clip(0.45 * jaccard + 0.35 * price_sim + 0.20 * vert_overlap, 0.0, 1.0))
            affinity[i, j] = score
            affinity[j, i] = score

    return {
        "provenance": {
            "source_label": cleaned_shopify["source_label"],
            "source_mode": cleaned_shopify["source_mode"],
            "captured_at_utc": cleaned_shopify["captured_at_utc"],
            "brand_count": n_brands,
            "total_products_polled": int(cleaned_shopify["total_products_polled"]),
            "total_variants_polled": int(cleaned_shopify["total_variants_polled"]),
        },
        "brands": brands_out,
        "brand_affinity_J": affinity.tolist(),
    }
