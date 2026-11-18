"""
Phase 2 (CLEAN): Deterministic normalization, time-alignment, and discretization.

Transforms raw heterogeneous public API feeds into aligned numerical matrices:
  - `clean_usgs_stream`: sentinel stripping, T=64 grid alignment, z-score, w=8 rolling baseline, {-1,+1} / {0,1} spins
  - `clean_rfam_msa`: RNA uppercase + T->U, 5-state {A,C,G,U,gap}, L=24 consensus selection, sequence gap pruning
  - `clean_market_stream`: common hourly grid forward-fill, log returns, w=12 baseline spins, Polymarket OBI & spread
  - `clean_shopify_catalogs`: zero-price/malformed removal, SKU deduplication, price & tag normalization
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

RNA_ALPHABET: List[str] = ["A", "C", "G", "U", "-"]
RNA_CHAR_TO_STATE: Dict[str, int] = {"A": 0, "C": 1, "G": 2, "U": 3}


def _parse_iso_epoch(dt_str: str) -> float:
    clean = dt_str.strip()
    if clean.endswith("Z"):
        clean = clean[:-1] + "+00:00"
    dt = datetime.fromisoformat(clean)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Compute causal rolling mean along axis=0 with min_periods=1."""
    t_len, n_cols = arr.shape
    out = np.empty_like(arr, dtype=np.float64)
    w = max(1, int(window))
    for t in range(t_len):
        start = max(0, t - w + 1)
        out[t] = np.mean(arr[start : t + 1], axis=0)
    return out


def _binarize_against_baseline(series: np.ndarray, baseline: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Binarize series into bipolar spins {-1, +1} and discrete binary {0, 1}."""
    diff = series - baseline
    # Tie-break exact zero differences (e.g., t=0 where baseline[0] == series[0]) by sign of series
    spins = np.where(np.abs(diff) > 1e-12, np.where(diff >= 0.0, 1, -1), np.where(series >= 0.0, 1, -1))
    spins_i8 = spins.astype(np.int8)
    binary_i8 = ((spins_i8 + 1) // 2).astype(np.int8)
    return spins_i8, binary_i8


# ---------------------------------------------------------------------------
# Stream 1 Cleaner: USGS Water Telemetry
# ---------------------------------------------------------------------------

def clean_usgs_stream(
    raw_usgs: Dict[str, Any],
    target_steps: int = 64,
    window: int = 8,
) -> Dict[str, Any]:
    """Strip sentinels (<=-99999, NaN), align to T=64 grid, z-score, and binarize spins."""
    raw_channels = raw_usgs.get("channels", [])
    if not raw_channels:
        raise ValueError("raw_usgs contains no channels to clean")

    parsed_series: List[Tuple[Dict[str, Any], np.ndarray, np.ndarray]] = []
    min_epochs: List[float] = []
    max_epochs: List[float] = []

    for ch in raw_channels[:16]:
        pairs: List[Tuple[float, float]] = []
        for obs in ch.get("observations", []):
            dt_str = obs.get("dateTime")
            val_raw = obs.get("value")
            if dt_str is None or val_raw is None:
                continue
            try:
                val = float(val_raw)
                ts = _parse_iso_epoch(str(dt_str))
            except Exception:
                continue
            if not math.isfinite(val) or val <= -99999.0:
                continue
            pairs.append((ts, val))

        if not pairs:
            raise ValueError(f"Channel {ch.get('channel_id')} has no valid finite observations")

        pairs.sort(key=lambda item: item[0])
        # Deduplicate identical timestamps (keep last value)
        dedup_ts: List[float] = []
        dedup_vals: List[float] = []
        for ts, val in pairs:
            if dedup_ts and abs(ts - dedup_ts[-1]) < 1e-6:
                dedup_vals[-1] = val
            else:
                dedup_ts.append(ts)
                dedup_vals.append(val)

        ts_arr = np.asarray(dedup_ts, dtype=np.float64)
        val_arr = np.asarray(dedup_vals, dtype=np.float64)
        min_epochs.append(float(ts_arr[0]))
        max_epochs.append(float(ts_arr[-1]))
        parsed_series.append((ch, ts_arr, val_arr))

    overlap_start = max(min_epochs)
    overlap_end = min(max_epochs)
    if overlap_end - overlap_start >= 3600.0:
        grid_start, grid_end = overlap_start, overlap_end
    else:
        grid_start, grid_end = min(min_epochs), max(max_epochs)

    if grid_end <= grid_start:
        grid_end = grid_start + 3600.0 * max(1, target_steps - 1)

    t_grid = np.linspace(grid_start, grid_end, target_steps, dtype=np.float64)
    timestamps_utc = [datetime.fromtimestamp(t, tz=timezone.utc).isoformat() for t in t_grid]

    n_channels = len(parsed_series)
    values_aligned = np.empty((target_steps, n_channels), dtype=np.float64)
    z_scores = np.empty((target_steps, n_channels), dtype=np.float64)

    channels_meta: List[Dict[str, Any]] = []
    sites_seen = set()

    for idx, (ch, ts_arr, val_arr) in enumerate(parsed_series):
        # Forward-fill across any sparse intermediate steps, then linear-interpolate onto uniform T=64 grid
        interp_vals = np.interp(t_grid, ts_arr, val_arr)
        values_aligned[:, idx] = interp_vals

        mu = float(np.mean(interp_vals))
        sigma = float(np.std(interp_vals))
        denom = sigma if sigma > 1e-6 else 1.0
        z_col = (interp_vals - mu) / denom
        z_scores[:, idx] = z_col

        site_code = str(ch.get("site_code", f"site_{idx}"))
        sites_seen.add(site_code)
        channels_meta.append(
            {
                "index": idx,
                "channel_id": str(ch.get("channel_id", f"{site_code}:{idx}")),
                "site_code": site_code,
                "site_name": str(ch.get("site_name", site_code)),
                "parameter_code": str(ch.get("parameter_code", "00000")),
                "parameter_name": str(ch.get("parameter_name", "Telemetry")),
                "unit": str(ch.get("unit", "unit")),
                "latest_value": float(interp_vals[-1]),
                "mean_value": mu,
                "std_value": sigma,
                "z_score_latest": float(z_col[-1]),
                "valid_points": int(len(val_arr)),
                "source_mode": ch.get("source_mode", raw_usgs.get("source_mode", "seeded_fallback")),
            }
        )

    baseline = _rolling_mean(z_scores, window=window)
    spins, binary = _binarize_against_baseline(z_scores, baseline)

    for idx, meta in enumerate(channels_meta):
        meta["spin_latest"] = int(spins[-1, idx])
        meta["binary_latest"] = int(binary[-1, idx])

    return {
        "source_label": raw_usgs.get("source_label", "USGS NWIS Water Telemetry"),
        "source_mode": raw_usgs.get("source_mode", "seeded_fallback"),
        "captured_at_utc": raw_usgs.get("captured_at_utc", datetime.now(timezone.utc).isoformat()),
        "station_count": len(sites_seen),
        "channel_count": n_channels,
        "timestamps_utc": timestamps_utc,
        "values_aligned": values_aligned,
        "z_scores": z_scores,
        "rolling_baseline": baseline,
        "spins": spins,
        "binary": binary,
        "channels": channels_meta,
    }


# ---------------------------------------------------------------------------
# Stream 2 Cleaner: Rfam RNA MSA (5-state consensus matrix)
# ---------------------------------------------------------------------------

def _clean_single_msa_family(
    family_payload: Dict[str, Any],
    consensus_len: int = 24,
    max_col_gap: float = 0.30,
    max_seq_gap: float = 0.25,
) -> Dict[str, Any]:
    raw_records = family_payload.get("sequences", [])
    if not raw_records:
        raise ValueError(f"Rfam family {family_payload.get('family_id')} has no sequences")

    raw_seqs = [str(rec.get("sequence", "")).strip().upper().replace("T", "U") for rec in raw_records if rec.get("sequence")]
    raw_count = len(raw_seqs)
    if raw_count == 0:
        raise ValueError("No non-empty sequences found in MSA")

    max_len = max(len(s) for s in raw_seqs)
    if max_len < consensus_len:
        raise ValueError(f"MSA width {max_len} < consensus_len {consensus_len}")

    encoded_full = np.full((raw_count, max_len), 4, dtype=np.int8)
    for r_idx, seq in enumerate(raw_seqs):
        for c_idx, ch in enumerate(seq):
            encoded_full[r_idx, c_idx] = RNA_CHAR_TO_STATE.get(ch, 4)

    col_gap_frac = np.mean(encoded_full == 4, axis=0)
    eligible_cols = np.where(col_gap_frac < max_col_gap)[0]

    if len(eligible_cols) >= consensus_len:
        # Pick L=24 highest-occupancy columns among eligible (< 0.30 gap fraction)
        ranked = eligible_cols[np.argsort(col_gap_frac[eligible_cols], kind="mergesort")]
        selected_cols = np.sort(ranked[:consensus_len])
    else:
        ranked_all = np.argsort(col_gap_frac, kind="mergesort")
        selected_cols = np.sort(ranked_all[:consensus_len])

    msa_consensus = encoded_full[:, selected_cols]
    seq_gap_ratio = np.mean(msa_consensus == 4, axis=1)
    keep_mask = seq_gap_ratio <= max_seq_gap

    if int(np.sum(keep_mask)) >= 5:
        msa_pruned = msa_consensus[keep_mask]
    else:
        # Degenerate guard: retain lowest-gap top sequences
        best_rows = np.argsort(seq_gap_ratio, kind="mergesort")[: max(5, min(20, raw_count))]
        msa_pruned = msa_consensus[best_rows]

    return {
        "family_id": str(family_payload.get("family_id", "RF00005")),
        "source_mode": family_payload.get("source_mode", "seeded_fallback"),
        "captured_at_utc": family_payload.get("captured_at_utc", datetime.now(timezone.utc).isoformat()),
        "alphabet": list(RNA_ALPHABET),
        "raw_sequences": raw_count,
        "retained_sequences": int(msa_pruned.shape[0]),
        "consensus_length": int(msa_pruned.shape[1]),
        "consensus_columns": [int(c) for c in selected_cols.tolist()],
        "msa_encoded": msa_pruned,
    }


def clean_rfam_msa(
    raw_rfam: Dict[str, Any],
    consensus_len: int = 24,
    max_col_gap: float = 0.30,
    max_seq_gap: float = 0.25,
) -> Dict[str, Any]:
    """Standardize Rfam MSA into 5-state {A,C,G,U,gap} and prune L=24 consensus columns."""
    if "families" in raw_rfam and isinstance(raw_rfam["families"], dict):
        cleaned_families: Dict[str, Dict[str, Any]] = {}
        for fam_id, fam_payload in raw_rfam["families"].items():
            cleaned_families[fam_id] = _clean_single_msa_family(
                fam_payload,
                consensus_len=consensus_len,
                max_col_gap=max_col_gap,
                max_seq_gap=max_seq_gap,
            )
        primary_id = "RF00005" if "RF00005" in cleaned_families else next(iter(cleaned_families))
        primary = cleaned_families[primary_id]
        return {
            **primary,
            "source_label": raw_rfam.get("source_label", "Rfam RNA Family Alignments"),
            "primary_family_id": primary_id,
            "families": cleaned_families,
        }

    return _clean_single_msa_family(
        raw_rfam,
        consensus_len=consensus_len,
        max_col_gap=max_col_gap,
        max_seq_gap=max_seq_gap,
    )


# ---------------------------------------------------------------------------
# Stream 3 Cleaner: Yahoo Finance Hourly Grid + Polymarket Order Books
# ---------------------------------------------------------------------------

def _clean_polymarket_book(book: Dict[str, Any]) -> Dict[str, Any]:
    bids_parsed: List[Tuple[float, float]] = []
    asks_parsed: List[Tuple[float, float]] = []

    for item in book.get("bids", []):
        try:
            p = float(item["price"])
            s = float(item["size"])
            if math.isfinite(p) and math.isfinite(s) and p > 0.0 and s > 0.0:
                bids_parsed.append((p, s))
        except Exception:
            continue

    for item in book.get("asks", []):
        try:
            p = float(item["price"])
            s = float(item["size"])
            if math.isfinite(p) and math.isfinite(s) and p > 0.0 and s > 0.0:
                asks_parsed.append((p, s))
        except Exception:
            continue

    bids_parsed.sort(key=lambda x: x[0], reverse=True)
    asks_parsed.sort(key=lambda x: x[0])

    best_bid = bids_parsed[0][0] if bids_parsed else (max(0.01, asks_parsed[0][0] - 0.02) if asks_parsed else 0.49)
    best_ask = asks_parsed[0][0] if asks_parsed else (min(0.99, best_bid + 0.02))
    if best_ask < best_bid:
        best_ask = best_bid

    # Top-5 near-touch levels for balanced local order-book pressure
    bid_depth = float(sum(sz for _, sz in bids_parsed[:5]))
    ask_depth = float(sum(sz for _, sz in asks_parsed[:5]))
    total_depth = bid_depth + ask_depth
    obi = float(np.clip((bid_depth - ask_depth) / (total_depth + 1e-9), -1.0, 1.0)) if total_depth > 0 else 0.0
    midpoint = float(np.clip(0.5 * (best_bid + best_ask), 0.001, 0.999))
    spread = float(max(0.0, best_ask - best_bid))

    return {
        "market_id": str(book.get("market_id", "")),
        "token_id": str(book.get("token_id", "")),
        "question": str(book.get("question", "Polymarket Active Market")),
        "best_bid": best_bid,
        "best_ask": best_ask,
        "bid_depth": bid_depth,
        "ask_depth": ask_depth,
        "midpoint": midpoint,
        "spread": spread,
        "obi": obi,
        "source_mode": book.get("source_mode", "seeded_fallback"),
    }


def clean_market_stream(
    raw_market: Dict[str, Any],
    rolling_window: int = 12,
) -> Dict[str, Any]:
    """Align 16 Yahoo hourly series with forward-fill, compute log returns/spins, and clean Polymarket CLOB books."""
    yahoo_list = raw_market.get("yahoo_series", [])
    if not yahoo_list:
        raise ValueError("raw_market has no yahoo_series")

    hourly_maps: List[Tuple[str, Dict[int, float], str]] = []
    min_hours: List[int] = []
    max_hours: List[int] = []

    for entry in yahoo_list:
        ts_list = entry.get("timestamps", [])
        cl_list = entry.get("closes", [])
        hour_map: Dict[int, float] = {}
        for ts, cl in zip(ts_list, cl_list):
            if ts is None or cl is None:
                continue
            try:
                f_cl = float(cl)
                h_ts = int(round(float(ts) / 3600.0) * 3600)
            except Exception:
                continue
            if math.isfinite(f_cl) and f_cl > 0.0:
                hour_map[h_ts] = f_cl
        if not hour_map:
            raise ValueError(f"Ticker {entry.get('ticker')} has no valid hourly closes")
        min_hours.append(min(hour_map.keys()))
        max_hours.append(max(hour_map.keys()))
        hourly_maps.append((str(entry["ticker"]), hour_map, str(entry.get("source_mode", "seeded_fallback"))))

    grid_start = min(min_hours)
    grid_end = max(max_hours)
    common_hours = list(range(grid_start, grid_end + 1, 3600))
    if len(common_hours) < 12:
        raise ValueError(f"Common hourly grid too short: {len(common_hours)}")

    n_hours = len(common_hours)
    n_assets = len(hourly_maps)
    aligned_prices = np.empty((n_hours, n_assets), dtype=np.float64)

    for j, (_, h_map, _) in enumerate(hourly_maps):
        sorted_keys = sorted(h_map.keys())
        last_val: Optional[float] = None
        first_known = h_map[sorted_keys[0]]
        for t_idx, h_ts in enumerate(common_hours):
            if h_ts in h_map:
                last_val = h_map[h_ts]
            aligned_prices[t_idx, j] = last_val if last_val is not None else first_known

    log_returns = np.diff(np.log(np.maximum(aligned_prices, 1e-9)), axis=0)
    baseline = _rolling_mean(log_returns, window=rolling_window)
    spins, binary = _binarize_against_baseline(log_returns, baseline)

    cleaned_books = [_clean_polymarket_book(b) for b in raw_market.get("polymarket_books", [])]

    return {
        "source_label": raw_market.get("source_label", "Yahoo Finance 1h + Polymarket CLOB"),
        "source_mode": raw_market.get("source_mode", "seeded_fallback"),
        "captured_at_utc": raw_market.get("captured_at_utc", datetime.now(timezone.utc).isoformat()),
        "tickers": [t for t, _, _ in hourly_maps],
        "ticker_modes": [m for _, _, m in hourly_maps],
        "common_hours": common_hours,
        "aligned_prices": aligned_prices,
        "log_returns": log_returns,
        "rolling_baseline": baseline,
        "spins": spins,
        "binary": binary,
        "polymarket_books": cleaned_books,
    }


# ---------------------------------------------------------------------------
# Stream 4 Cleaner: Shopify DTC Catalogs
# ---------------------------------------------------------------------------

def _normalize_tags(raw_tags: Any, product_type: Any = "") -> List[str]:
    tokens: List[str] = []
    if isinstance(raw_tags, list):
        tokens.extend(str(t) for t in raw_tags)
    elif isinstance(raw_tags, str):
        tokens.extend(raw_tags.split(","))
    if isinstance(product_type, str) and product_type.strip():
        tokens.append(product_type)

    cleaned: List[str] = []
    seen = set()
    for tok in tokens:
        t = tok.strip().lower()
        if not t:
            continue
        # Strip Shopify internal facet hashes such as `brand::cfid => ...`
        if "=>" in t:
            prefix, rhs = t.split("=>", 1)
            if "cfid" in prefix or len(rhs.strip()) >= 24:
                continue
            t = rhs.strip()
        elif "::" in t:
            t = t.split("::", 1)[1].strip()
        if not t or len(t) > 48:
            continue
        if t not in seen:
            seen.add(t)
            cleaned.append(t)
    return cleaned


def _extract_product_image_url(prod: Dict[str, Any]) -> str:
    """Extract and normalize the primary Shopify product image URL."""
    candidates: List[Any] = []
    images_list = prod.get("images")
    if isinstance(images_list, list):
        candidates.extend(images_list)
    single_img = prod.get("image")
    if single_img is not None:
        candidates.append(single_img)

    for entry in candidates:
        raw_src = entry.get("src") if isinstance(entry, dict) else (entry if isinstance(entry, str) else "")
        src = str(raw_src or "").strip()
        if not src:
            continue
        if src.startswith("//"):
            return f"https:{src}"
        if src.startswith(("http://", "https://")):
            return src
    return ""


def clean_shopify_catalogs(raw_shopify: Dict[str, Any]) -> Dict[str, Any]:
    """Strip malformed/zero-price variants, deduplicate SKUs, normalize prices and lowercase trimmed tags."""
    raw_brands = raw_shopify.get("brands", [])
    cleaned_brands: List[Dict[str, Any]] = []
    total_products = 0
    total_variants = 0

    for b_idx, brand_obj in enumerate(raw_brands):
        brand_name = str(brand_obj.get("brand", f"Brand_{b_idx}"))
        domain = str(brand_obj.get("domain", ""))
        vertical = str(brand_obj.get("vertical", "DTC Commerce"))
        b_mode = brand_obj.get("source_mode", raw_shopify.get("source_mode", "seeded_fallback"))

        seen_skus = set()
        valid_prices: List[float] = []
        discount_ratios: List[float] = []
        available_flags: List[bool] = []
        updated_timestamps: List[float] = []
        tag_counts: Dict[str, int] = {}
        valid_product_count = 0
        featured_candidates: List[Dict[str, Any]] = []

        for prod in brand_obj.get("products", []):
            if not isinstance(prod, dict):
                continue
            p_id = prod.get("id", "prod")
            p_tags = _normalize_tags(prod.get("tags", []), prod.get("product_type", ""))
            for t in p_tags:
                tag_counts[t] = tag_counts.get(t, 0) + 1

            upd = prod.get("updated_at")
            if isinstance(upd, str) and upd.strip():
                try:
                    updated_timestamps.append(_parse_iso_epoch(upd))
                except Exception:
                    pass

            prod_has_valid_variant = False
            first_price: Optional[float] = None
            first_compare: Optional[float] = None
            any_available = False
            prod_variant_count = 0

            for var in prod.get("variants", []):
                if not isinstance(var, dict):
                    continue
                try:
                    price = float(var.get("price", 0.0))
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(price) or price <= 0.0:
                    continue

                raw_sku = str(var.get("sku") or "").strip().upper()
                dedup_key = raw_sku if raw_sku else f"ID:{p_id}:{var.get('id', len(seen_skus))}"
                if dedup_key in seen_skus:
                    continue
                seen_skus.add(dedup_key)

                var_avail = bool(var.get("available", True))
                valid_prices.append(price)
                available_flags.append(var_avail)
                prod_has_valid_variant = True
                prod_variant_count += 1
                if first_price is None:
                    first_price = price
                if var_avail:
                    any_available = True

                cap_raw = var.get("compare_at_price")
                if cap_raw is not None:
                    try:
                        cap = float(cap_raw)
                        if math.isfinite(cap) and cap > price:
                            discount_ratios.append((cap - price) / cap)
                            if first_compare is None and first_price is not None and cap > first_price:
                                first_compare = cap
                    except (TypeError, ValueError):
                        pass

            if prod_has_valid_variant and first_price is not None:
                valid_product_count += 1
                img_url = _extract_product_image_url(prod)
                title = str(prod.get("title") or f"{brand_name} Featured Drop").strip()
                handle = str(prod.get("handle") or "").strip()
                product_url = f"https://{domain}/products/{handle}" if handle else f"https://{domain}"
                featured_candidates.append(
                    {
                        "title": title,
                        "handle": handle,
                        "product_url": product_url,
                        "image_url": img_url,
                        "price_usd": round(float(first_price), 2),
                        "compare_at_price_usd": round(float(first_compare), 2) if first_compare is not None else None,
                        "available": any_available,
                        "variant_count": prod_variant_count,
                    }
                )

        # Deterministic safe guard if a brand returned products without positive variant prices
        if not valid_prices:
            valid_prices = [48.0]
            available_flags = [True]

        featured_candidates.sort(
            key=lambda item: (bool(item["image_url"]), bool(item["available"])),
            reverse=True,
        )
        featured_products = featured_candidates[:3]

        total_products += valid_product_count
        total_variants += len(valid_prices)

        sorted_tags = sorted(tag_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        top_tags = [t for t, _ in sorted_tags[:6]]

        cleaned_brands.append(
            {
                "index": int(brand_obj.get("index", b_idx)),
                "brand": brand_name,
                "domain": domain,
                "vertical": vertical,
                "source_mode": b_mode,
                "product_count": valid_product_count,
                "variant_count": len(valid_prices),
                "prices": valid_prices,
                "discount_ratios": discount_ratios,
                "available_flags": available_flags,
                "updated_timestamps": updated_timestamps,
                "tag_counts": tag_counts,
                "top_tags": top_tags,
                "featured_products": featured_products,
            }
        )

    return {
        "source_label": raw_shopify.get("source_label", "AffilAI Shopify DTC Catalogs"),
        "source_mode": raw_shopify.get("source_mode", "seeded_fallback"),
        "captured_at_utc": raw_shopify.get("captured_at_utc", datetime.now(timezone.utc).isoformat()),
        "brand_count": len(cleaned_brands),
        "total_products_polled": total_products,
        "total_variants_polled": total_variants,
        "brands": cleaned_brands,
    }
