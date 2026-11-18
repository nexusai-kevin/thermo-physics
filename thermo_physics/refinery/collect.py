"""
Phase 1 (COLLECT): Fault-tolerant public-stream harvester for the Live-Stream Data Refinery.

Polls four orthogonal live empirical domains using `httpx` with bounded timeouts
(8.0s), retries (2 retries with 0.75s backoff), polite rate-limit pacing, automated
snapshot caching in `data/refinery_cache/<stream_id>_latest.json`, and deterministic
seeded fallbacks tagged cleanly via `source_mode`:
  - 'live_public_stream'
  - 'cached_empirical'
  - 'seeded_fallback'
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import httpx
import numpy as np

from thermo_physics.refinery.schemas import SourceMode

PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CACHE_DIR = PACKAGE_ROOT / "data" / "refinery_cache"

DEFAULT_TIMEOUT_S = 8.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_BACKOFF_S = 0.75

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Encoding": "gzip, deflate",
}

USGS_URL = (
    "https://waterservices.usgs.gov/nwis/iv/?format=json"
    "&sites=01463500,01646500,01372058,11447650,02171645,07374000,14211720,01578310"
    "&parameterCd=00010,00095,00300,00400&period=P2D"
)

RFAM_URLS = {
    "RF00005": "https://rfam.org/family/RF00005/alignment/fasta",
    "RF00001": "https://rfam.org/family/RF00001/alignment/fasta",
}

YAHOO_TICKERS: List[str] = [
    "NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSM", "AVGO",
    "BTC-USD", "ETH-USD", "SOL-USD", "GC=F", "CL=F", "SI=F", "HG=F", "NG=F",
]

POLYMARKET_CLOB_MARKETS_URL = "https://clob.polymarket.com/sampling-simplified-markets"
POLYMARKET_CLOB_BOOK_URL = "https://clob.polymarket.com/book"
POLYMARKET_GAMMA_URL = "https://gamma-api.polymarket.com/markets?limit=16&active=true&closed=false"

SHOPIFY_BRAND_SPECS: List[Tuple[str, str, str]] = [
    ("Allbirds", "www.allbirds.com", "Apparel & Footwear"),
    ("Brooklinen", "www.brooklinen.com", "Home & Bedding"),
    ("Tentree", "www.tentree.com", "Sustainable Apparel"),
    ("Olipop", "drinkolipop.com", "Functional Beverage"),
    ("MVMT", "www.mvmt.com", "Watches & Accessories"),
    ("ColourPop", "colourpop.com", "Beauty & Cosmetics"),
    ("KylieCosmetics", "kyliecosmetics.com", "Beauty & Cosmetics"),
    ("Glossier", "www.glossier.com", "Skincare & Beauty"),
    ("Native", "www.nativecos.com", "Personal Care"),
    ("Beardbrand", "www.beardbrand.com", "Grooming"),
    ("Keychron", "www.keychron.com", "Mechanical Keyboards & Tech"),
    ("Ridge", "www.ridge.com", "EDC & Wallets"),
    ("Ugmonk", "ugmonk.com", "Workspace & Design"),
    ("PelaCase", "pelacase.com", "Eco Phone Accessories"),
    ("Moment", "www.shopmoment.com", "Creator Gear & Optics"),
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_cache_path(stream_id: str, cache_dir: Optional[Path] = None) -> Path:
    base = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    return base / f"{stream_id}_latest.json"


def _save_cache(stream_id: str, payload: Dict[str, Any], cache_dir: Optional[Path] = None) -> None:
    path = _resolve_cache_path(stream_id, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    tmp_path.replace(path)


def _load_cache(stream_id: str, cache_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    path = _resolve_cache_path(stream_id, cache_dir)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def _fetch_with_retry(
    client: httpx.Client,
    url: str,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_s: float = DEFAULT_BACKOFF_S,
) -> httpx.Response:
    """Execute HTTP GET with retries, rate-limit handling, and exponential-linear backoff."""
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.get(url)
            if resp.status_code in (429, 502, 503, 504):
                if attempt < max_retries:
                    retry_after = resp.headers.get("Retry-After")
                    delay = backoff_s * (attempt + 1)
                    if retry_after:
                        try:
                            delay = min(4.0, max(delay, float(retry_after)))
                        except ValueError:
                            pass
                    time.sleep(delay)
                    continue
            resp.raise_for_status()
            return resp
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                time.sleep(backoff_s * (attempt + 1))
    assert last_exc is not None
    raise last_exc


def _create_client(client: Optional[httpx.Client]) -> Tuple[httpx.Client, bool]:
    if client is not None:
        return client, False
    return (
        httpx.Client(
            timeout=DEFAULT_TIMEOUT_S,
            headers=DEFAULT_HEADERS,
            follow_redirects=True,
        ),
        True,
    )


# ---------------------------------------------------------------------------
# Stream 1: USGS Water Services Telemetry
# ---------------------------------------------------------------------------

def _seeded_usgs_payload(captured_at_utc: str) -> Dict[str, Any]:
    """Deterministic multi-station estuarine sensor fallback matching NWIS channel layout."""
    rng = np.random.default_rng(20261118)
    sites = [
        ("01463500", "Delaware River at Trenton NJ"),
        ("02171645", "Rediv Canal at Santee River SC"),
        ("07374000", "Mississippi River at Baton Rouge LA"),
        ("11447650", "Sacramento River at Freeport CA"),
    ]
    params = [
        ("00010", "Temperature, water", "deg C", 14.5, 1.8),
        ("00095", "Specific conductance", "uS/cm", 290.0, 42.0),
        ("00300", "Dissolved oxygen", "mg/l", 9.1, 0.85),
        ("00400", "pH, field", "std units", 7.65, 0.22),
    ]
    base_dt = datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc)
    n_pts = 64
    t_grid = np.linspace(0.0, 2.0 * np.pi, n_pts)
    # Coupled regional hydrological wave modes driving realistic cross-channel Ising correlations
    basin_mode = np.sin(2.0 * t_grid)
    tidal_mode = np.cos(3.0 * t_grid + 0.4)

    channels: List[Dict[str, Any]] = []
    for s_idx, (site_code, site_name) in enumerate(sites):
        for p_idx, (p_code, p_name, unit, base_val, amp) in enumerate(params):
            phase = 0.45 * s_idx + 0.75 * p_idx
            # Dissolved O2 anti-correlates with temperature
            sign = -1.0 if p_code == "00300" else 1.0
            signal = (
                base_val
                + sign * amp * 0.65 * np.sin(t_grid + phase)
                + amp * 0.30 * basin_mode
                + amp * 0.20 * tidal_mode
                + rng.normal(0.0, amp * 0.14, size=n_pts)
            )
            observations = [
                {
                    "dateTime": (base_dt + timedelta(minutes=30 * k)).isoformat(),
                    "value": round(float(signal[k]), 4),
                }
                for k in range(n_pts)
            ]
            channels.append(
                {
                    "channel_id": f"{site_code}:{p_code}",
                    "site_code": site_code,
                    "site_name": site_name,
                    "parameter_code": p_code,
                    "parameter_name": p_name,
                    "unit": unit,
                    "observations": observations,
                    "source_mode": "seeded_fallback",
                }
            )

    return {
        "stream_id": "usgs_water",
        "source_label": "USGS National Water Information System (NWIS) Instantaneous Telemetry",
        "source_mode": "seeded_fallback",
        "captured_at_utc": captured_at_utc,
        "channels": channels,
    }


def _parse_usgs_json(raw_json: Dict[str, Any], captured_at_utc: str) -> Dict[str, Any]:
    """Extract and round-robin balance 16 active telemetry channels from USGS JSON."""
    ts_list = raw_json.get("value", {}).get("timeSeries", [])
    by_site: Dict[str, List[Dict[str, Any]]] = {}

    for ts in ts_list:
        source_info = ts.get("sourceInfo", {})
        site_codes = source_info.get("siteCode", [{}])
        site_code = str(site_codes[0].get("value", "unknown")) if site_codes else "unknown"
        site_name = str(source_info.get("siteName", site_code)).strip()

        var_info = ts.get("variable", {})
        var_codes = var_info.get("variableCode", [{}])
        p_code = str(var_codes[0].get("value", "00000")) if var_codes else "00000"
        p_name = str(var_info.get("variableName", p_code)).replace("&#176;", "deg ").strip()
        unit = str(var_info.get("unit", {}).get("unitCode", "unit")).strip()

        values_blocks = ts.get("values", [])
        raw_obs = values_blocks[0].get("value", []) if values_blocks else []
        valid_obs: List[Dict[str, Any]] = []
        for entry in raw_obs:
            dt_str = entry.get("dateTime")
            val_raw = entry.get("value")
            if not dt_str or val_raw is None:
                continue
            try:
                val = float(val_raw)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(val) or val <= -99999.0:
                continue
            valid_obs.append({"dateTime": str(dt_str), "value": val})

        if len(valid_obs) >= 4:
            by_site.setdefault(site_code, []).append(
                {
                    "channel_id": f"{site_code}:{p_code}",
                    "site_code": site_code,
                    "site_name": site_name,
                    "parameter_code": p_code,
                    "parameter_name": p_name,
                    "unit": unit,
                    "observations": valid_obs,
                    "source_mode": "live_public_stream",
                }
            )

    for site_code in by_site:
        by_site[site_code].sort(key=lambda ch: ch["parameter_code"])

    # Round-robin across stations to ensure broad geographic watershed diversity
    selected: List[Dict[str, Any]] = []
    site_keys = sorted(by_site.keys())
    round_idx = 0
    while len(selected) < 16 and site_keys:
        progressed = False
        for sk in site_keys:
            if round_idx < len(by_site[sk]):
                selected.append(by_site[sk][round_idx])
                progressed = True
                if len(selected) >= 16:
                    break
        if not progressed:
            break
        round_idx += 1

    if len(selected) < 16:
        raise ValueError(f"Insufficient active USGS channels ({len(selected)} < 16)")

    return {
        "stream_id": "usgs_water",
        "source_label": "USGS National Water Information System (NWIS) Instantaneous Telemetry",
        "source_mode": "live_public_stream",
        "captured_at_utc": captured_at_utc,
        "channels": selected[:16],
    }


def collect_usgs_water_telemetry(
    offline_only: bool = False,
    cache_dir: Optional[Path] = None,
    client: Optional[httpx.Client] = None,
) -> Dict[str, Any]:
    """Harvest 16 USGS NWIS water sensor channels with retry, cache, and seeded fallback."""
    stream_id = "usgs_water"
    now_iso = _utc_now_iso()

    if not offline_only:
        http_client, owns_client = _create_client(client)
        try:
            resp = _fetch_with_retry(http_client, USGS_URL)
            parsed = _parse_usgs_json(resp.json(), now_iso)
            _save_cache(stream_id, parsed, cache_dir=cache_dir)
            return parsed
        except Exception:
            pass
        finally:
            if owns_client:
                http_client.close()

    cached = _load_cache(stream_id, cache_dir=cache_dir)
    if cached and len(cached.get("channels", [])) >= 16:
        cached_copy = dict(cached)
        cached_copy["source_mode"] = "cached_empirical"
        for ch in cached_copy["channels"]:
            ch["source_mode"] = "cached_empirical"
        return cached_copy

    return _seeded_usgs_payload(now_iso)


# ---------------------------------------------------------------------------
# Stream 2: Rfam RNA Multiple Sequence Alignments (RF00005 & RF00001)
# ---------------------------------------------------------------------------

def _parse_fasta(text: str) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    cur_header: Optional[str] = None
    chunks: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(">"):
            if cur_header is not None and chunks:
                records.append({"header": cur_header, "sequence": "".join(chunks)})
            cur_header = stripped[1:].strip()
            chunks = []
        else:
            chunks.append(stripped)
    if cur_header is not None and chunks:
        records.append({"header": cur_header, "sequence": "".join(chunks)})
    return records


def _seeded_rfam_family(family_id: str) -> List[Dict[str, str]]:
    """Deterministic synthetic homologous RNA MSA with genuine stem-loop covariation."""
    rng = np.random.default_rng(1005 if family_id == "RF00005" else 1001)
    n_seqs = 64
    length = 72
    template = list("GGGGUAUAGCUCAGUUGGUAGAGCGUUGCCCUUUGCAAGGAGGACUCGGGGGUUCGAAUUCCCCCUACCCCA")
    wc_pairs = [(0, 71), (1, 70), (2, 69), (3, 68), (4, 67), (10, 25), (11, 24), (12, 23), (27, 43), (28, 42), (48, 64), (49, 63)]
    valid_pairs = [("G", "C"), ("C", "G"), ("A", "U"), ("U", "A"), ("G", "U"), ("U", "G")]
    bases = ["A", "C", "G", "U"]

    records: List[Dict[str, str]] = []
    for idx in range(n_seqs):
        seq = list(template[:length])
        for pos in range(length):
            if rng.random() < 0.08:
                seq[pos] = "-"
            elif rng.random() < 0.14:
                seq[pos] = str(rng.choice(bases))
        # Compensatory mutations on paired stems to create strong Potts co-evolution signal
        for i, j in wc_pairs:
            if rng.random() < 0.55:
                b1, b2 = valid_pairs[int(rng.integers(0, len(valid_pairs)))]
                seq[i] = b1
                seq[j] = b2
        records.append({"header": f"{family_id}_SEED_{idx:03d}/1-{length}", "sequence": "".join(seq)})
    return records


def _seeded_rfam_payload(captured_at_utc: str) -> Dict[str, Any]:
    return {
        "stream_id": "rfam_rna",
        "source_label": "Rfam RNA Family Stockholm/FASTA Alignments (RF00005 tRNA & RF00001 5S rRNA)",
        "source_mode": "seeded_fallback",
        "captured_at_utc": captured_at_utc,
        "families": {
            "RF00005": {
                "family_id": "RF00005",
                "name": "tRNA",
                "source_mode": "seeded_fallback",
                "captured_at_utc": captured_at_utc,
                "sequences": _seeded_rfam_family("RF00005"),
            },
            "RF00001": {
                "family_id": "RF00001",
                "name": "5S_rRNA",
                "source_mode": "seeded_fallback",
                "captured_at_utc": captured_at_utc,
                "sequences": _seeded_rfam_family("RF00001"),
            },
        },
    }


def collect_rfam_rna_msa(
    offline_only: bool = False,
    cache_dir: Optional[Path] = None,
    client: Optional[httpx.Client] = None,
) -> Dict[str, Any]:
    """Harvest Rfam RF00005 (tRNA) and RF00001 (5S rRNA) FASTA MSAs."""
    stream_id = "rfam_rna"
    now_iso = _utc_now_iso()
    cached = _load_cache(stream_id, cache_dir=cache_dir)

    if not offline_only:
        http_client, owns_client = _create_client(client)
        families: Dict[str, Dict[str, Any]] = {}
        try:
            for fam_id, url in RFAM_URLS.items():
                try:
                    resp = _fetch_with_retry(http_client, url)
                    seqs = _parse_fasta(resp.text)
                    if len(seqs) >= 10:
                        families[fam_id] = {
                            "family_id": fam_id,
                            "name": "tRNA" if fam_id == "RF00005" else "5S_rRNA",
                            "source_mode": "live_public_stream",
                            "captured_at_utc": now_iso,
                            "sequences": seqs,
                        }
                except Exception:
                    pass
                time.sleep(0.05)
        finally:
            if owns_client:
                http_client.close()

        if len(families) == len(RFAM_URLS):
            payload = {
                "stream_id": stream_id,
                "source_label": "Rfam RNA Family Stockholm/FASTA Alignments (RF00005 tRNA & RF00001 5S rRNA)",
                "source_mode": "live_public_stream",
                "captured_at_utc": now_iso,
                "families": families,
            }
            _save_cache(stream_id, payload, cache_dir=cache_dir)
            return payload

        if families:
            # Merge partial live fetch with cache or seeded fallback for missing family
            cached_fams = (cached or {}).get("families", {})
            for fam_id in RFAM_URLS:
                if fam_id not in families:
                    if fam_id in cached_fams and cached_fams[fam_id].get("sequences"):
                        fam_copy = dict(cached_fams[fam_id])
                        fam_copy["source_mode"] = "cached_empirical"
                        families[fam_id] = fam_copy
                    else:
                        families[fam_id] = {
                            "family_id": fam_id,
                            "name": "tRNA" if fam_id == "RF00005" else "5S_rRNA",
                            "source_mode": "seeded_fallback",
                            "captured_at_utc": now_iso,
                            "sequences": _seeded_rfam_family(fam_id),
                        }
            return {
                "stream_id": stream_id,
                "source_label": "Rfam RNA Family Stockholm/FASTA Alignments (RF00005 tRNA & RF00001 5S rRNA)",
                "source_mode": families["RF00005"]["source_mode"],
                "captured_at_utc": now_iso,
                "families": families,
            }

    if cached and cached.get("families"):
        cached_copy = dict(cached)
        cached_copy["source_mode"] = "cached_empirical"
        for fam in cached_copy["families"].values():
            fam["source_mode"] = "cached_empirical"
        return cached_copy

    return _seeded_rfam_payload(now_iso)


# ---------------------------------------------------------------------------
# Stream 3: Yahoo Finance 16-Ticker Hourly Bars + Polymarket CLOB Books
# ---------------------------------------------------------------------------

def _seeded_yahoo_series(ticker: str, idx: int) -> Dict[str, Any]:
    rng = np.random.default_rng(7000 + idx)
    base_prices = {
        "NVDA": 142.0, "AAPL": 228.0, "MSFT": 422.0, "GOOGL": 178.0,
        "AMZN": 208.0, "META": 585.0, "TSM": 194.0, "AVGO": 176.0,
        "BTC-USD": 89500.0, "ETH-USD": 3120.0, "SOL-USD": 218.0,
        "GC=F": 2610.0, "CL=F": 69.5, "SI=F": 30.8, "HG=F": 4.18, "NG=F": 2.95,
    }
    p0 = base_prices.get(ticker, 100.0)
    end_ts = 1795021200  # deterministic reference epoch aligned to whole hour
    n_hours = 110
    timestamps = [end_ts - (n_hours - 1 - k) * 3600 for k in range(n_hours)]
    # Common market factor + sector factor + idiosyncratic shock
    t_arr = np.arange(n_hours, dtype=np.float64)
    drift = 0.00025 if idx % 3 == 0 else 0.00008
    rets = drift + 0.004 * np.sin(0.18 * t_arr + idx * 0.4) + rng.normal(0.0, 0.007, size=n_hours)
    closes = p0 * np.exp(np.cumsum(rets))
    return {
        "ticker": ticker,
        "timestamps": timestamps,
        "closes": [round(float(x), 4) for x in closes],
        "source_mode": "seeded_fallback",
    }


def _seeded_polymarket_books() -> List[Dict[str, Any]]:
    rng = np.random.default_rng(8800)
    questions = [
        "Fed rate cut at upcoming FOMC meeting?",
        "US CPI YoY stays below 3.0% next print?",
        "Bitcoin trades above $95k this month?",
        "Ethereum ETF net inflows positive this week?",
        "US 10Y Treasury yield closes above 4.35%?",
        "Nvidia beats quarterly revenue consensus?",
        "WTI Crude settles below $72/bbl this week?",
        "Gold spot closes above $2,650/oz?",
        "S&P 500 posts weekly gain?",
        "TSMC monthly revenue growth > 25% YoY?",
        "Solana weekly DEX volume exceeds $15B?",
        "Copper futures rise > 2% over 5 sessions?",
        "Natural gas storage draw exceeds forecast?",
        "Apple services margin exceeds 74%?",
        "Meta Reality Labs capex guidance raised?",
        "Amazon AWS quarterly operating margin > 35%?",
    ]
    books: List[Dict[str, Any]] = []
    for idx, q in enumerate(questions):
        mid = float(np.clip(0.25 + 0.035 * idx + rng.normal(0.0, 0.04), 0.08, 0.92))
        spread = round(float(rng.uniform(0.002, 0.014)), 4)
        best_bid = round(max(0.01, mid - 0.5 * spread), 4)
        best_ask = round(min(0.99, mid + 0.5 * spread), 4)
        bids = [
            {"price": str(round(max(0.005, best_bid - k * 0.002), 4)), "size": str(round(float(rng.uniform(400, 4500)), 2))}
            for k in range(5)
        ]
        asks = [
            {"price": str(round(min(0.995, best_ask + k * 0.002), 4)), "size": str(round(float(rng.uniform(400, 4500)), 2))}
            for k in range(5)
        ]
        books.append(
            {
                "market_id": f"seeded-poly-{idx:02d}",
                "token_id": f"seeded-token-{idx:02d}",
                "question": q,
                "bids": bids,
                "asks": asks,
                "source_mode": "seeded_fallback",
            }
        )
    return books


def _seeded_quant_polymarket_payload(captured_at_utc: str) -> Dict[str, Any]:
    return {
        "stream_id": "quant_polymarket",
        "source_label": "Yahoo Finance 1h OHLCV (16 Assets) + Polymarket CLOB Order Books (16 Active Markets)",
        "source_mode": "seeded_fallback",
        "captured_at_utc": captured_at_utc,
        "yahoo_series": [_seeded_yahoo_series(t, i) for i, t in enumerate(YAHOO_TICKERS)],
        "polymarket_books": _seeded_polymarket_books(),
    }


def _fetch_polymarket_books(http_client: httpx.Client) -> Tuple[List[Dict[str, Any]], SourceMode]:
    """Fetch 16 active Polymarket order books combining CLOB sampling markets and Gamma titles."""
    gamma_titles: Dict[str, str] = {}
    gamma_list: List[Dict[str, Any]] = []
    try:
        r_gamma = _fetch_with_retry(http_client, POLYMARKET_GAMMA_URL)
        gamma_data = r_gamma.json()
        if isinstance(gamma_data, list):
            gamma_list = gamma_data
            for item in gamma_data:
                cid = item.get("conditionId")
                q = item.get("question")
                if cid and q:
                    gamma_titles[str(cid)] = str(q)
    except Exception:
        pass

    clob_items: List[Dict[str, Any]] = []
    try:
        r_clob = _fetch_with_retry(http_client, POLYMARKET_CLOB_MARKETS_URL)
        clob_json = r_clob.json()
        if isinstance(clob_json, dict):
            clob_items = clob_json.get("data", [])
    except Exception:
        pass

    candidates: List[Tuple[str, str, str]] = []  # (condition_id, token_id, question)
    seen_cids = set()

    # 1. Prioritize Gamma-titled active markets (preferring those present in CLOB sampling list)
    clob_by_cid = {
        str(m.get("condition_id")): m
        for m in clob_items
        if m.get("condition_id") and m.get("tokens") and m.get("active", True) and not m.get("closed", False)
    }
    for g in gamma_list:
        cid = str(g.get("conditionId", ""))
        if not cid or cid in seen_cids:
            continue
        toks = g.get("clobTokenIds")
        if isinstance(toks, str):
            try:
                toks = json.loads(toks)
            except Exception:
                toks = []
        tid = str(toks[0]) if isinstance(toks, list) and toks else ""
        if not tid and cid in clob_by_cid:
            tid = str(clob_by_cid[cid]["tokens"][0].get("token_id", ""))
        if tid:
            candidates.append((cid, tid, str(g.get("question", cid))))
            seen_cids.add(cid)

    # 2. Top up with active accepting-orders CLOB sampling markets if Gamma had < 16 usable tokens
    for m in clob_items:
        if len(candidates) >= 24:
            break
        cid = str(m.get("condition_id", ""))
        if not cid or cid in seen_cids:
            continue
        if not m.get("active", True) or m.get("closed", False) or not m.get("accepting_orders", True):
            continue
        toks = m.get("tokens", [])
        if not toks:
            continue
        tid = str(toks[0].get("token_id", ""))
        if not tid:
            continue
        outcome = str(toks[0].get("outcome", "Yes"))
        label = gamma_titles.get(cid, f"Polymarket CLOB {cid[:10]} ({outcome})")
        candidates.append((cid, tid, label))
        seen_cids.add(cid)

    books: List[Dict[str, Any]] = []
    for cid, tid, question in candidates:
        if len(books) >= 16:
            break
        try:
            r_book = _fetch_with_retry(http_client, f"{POLYMARKET_CLOB_BOOK_URL}?token_id={tid}")
            book_json = r_book.json()
            bids = book_json.get("bids", [])
            asks = book_json.get("asks", [])
            if bids or asks:
                books.append(
                    {
                        "market_id": cid,
                        "token_id": tid,
                        "question": question,
                        "bids": bids,
                        "asks": asks,
                        "source_mode": "live_public_stream",
                    }
                )
        except Exception:
            continue
        time.sleep(0.03)

    if books:
        return books, "live_public_stream"
    return _seeded_polymarket_books(), "seeded_fallback"


def collect_quant_and_polymarket(
    offline_only: bool = False,
    cache_dir: Optional[Path] = None,
    client: Optional[httpx.Client] = None,
) -> Dict[str, Any]:
    """Harvest Yahoo Finance 1h bars for 16 assets and 16 Polymarket CLOB order books."""
    stream_id = "quant_polymarket"
    now_iso = _utc_now_iso()
    cached = _load_cache(stream_id, cache_dir=cache_dir)

    if not offline_only:
        http_client, owns_client = _create_client(client)
        yahoo_series: List[Dict[str, Any]] = []
        cached_yahoo_map = {
            item["ticker"]: item for item in (cached or {}).get("yahoo_series", []) if "ticker" in item
        }
        try:
            for idx, ticker in enumerate(YAHOO_TICKERS):
                encoded = quote(ticker, safe="")
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded}?interval=1h&range=5d"
                fetched_item: Optional[Dict[str, Any]] = None
                try:
                    resp = _fetch_with_retry(http_client, url)
                    res_list = resp.json().get("chart", {}).get("result", [])
                    if res_list:
                        res0 = res_list[0]
                        timestamps = res0.get("timestamp", [])
                        closes = res0.get("indicators", {}).get("quote", [{}])[0].get("close", [])
                        clean_ts: List[int] = []
                        clean_cl: List[float] = []
                        for ts, cl in zip(timestamps, closes):
                            if ts is not None and cl is not None:
                                try:
                                    f_cl = float(cl)
                                except (TypeError, ValueError):
                                    continue
                                if math.isfinite(f_cl) and f_cl > 0.0:
                                    clean_ts.append(int(ts))
                                    clean_cl.append(f_cl)
                        if len(clean_ts) >= 8:
                            fetched_item = {
                                "ticker": ticker,
                                "timestamps": clean_ts,
                                "closes": clean_cl,
                                "source_mode": "live_public_stream",
                            }
                except Exception:
                    pass

                if fetched_item is not None:
                    yahoo_series.append(fetched_item)
                elif ticker in cached_yahoo_map:
                    c_item = dict(cached_yahoo_map[ticker])
                    c_item["source_mode"] = "cached_empirical"
                    yahoo_series.append(c_item)
                else:
                    yahoo_series.append(_seeded_yahoo_series(ticker, idx))
                time.sleep(0.03)

            poly_books, poly_mode = _fetch_polymarket_books(http_client)
            if poly_mode != "live_public_stream" and cached and cached.get("polymarket_books"):
                poly_books = [dict(b, source_mode="cached_empirical") for b in cached["polymarket_books"]]
                poly_mode = "cached_empirical"
            while len(poly_books) < 16:
                seeded_pad = _seeded_polymarket_books()
                poly_books.append(seeded_pad[len(poly_books)])
        finally:
            if owns_client:
                http_client.close()

        live_tickers = sum(1 for item in yahoo_series if item["source_mode"] == "live_public_stream")
        if live_tickers == len(YAHOO_TICKERS) and poly_mode == "live_public_stream":
            overall_mode: SourceMode = "live_public_stream"
        elif live_tickers >= 8 or poly_mode == "live_public_stream":
            overall_mode = "live_public_stream"
        elif any(item["source_mode"] == "cached_empirical" for item in yahoo_series):
            overall_mode = "cached_empirical"
        else:
            overall_mode = "seeded_fallback"

        payload = {
            "stream_id": stream_id,
            "source_label": "Yahoo Finance 1h OHLCV (16 Assets) + Polymarket CLOB Order Books (16 Active Markets)",
            "source_mode": overall_mode,
            "captured_at_utc": now_iso,
            "yahoo_series": yahoo_series,
            "polymarket_books": poly_books[:16],
        }
        if live_tickers >= 8:
            _save_cache(stream_id, payload, cache_dir=cache_dir)
        return payload

    if cached and cached.get("yahoo_series") and cached.get("polymarket_books"):
        cached_copy = dict(cached)
        cached_copy["source_mode"] = "cached_empirical"
        for item in cached_copy["yahoo_series"]:
            item["source_mode"] = "cached_empirical"
        for item in cached_copy["polymarket_books"]:
            item["source_mode"] = "cached_empirical"
        return cached_copy

    return _seeded_quant_polymarket_payload(now_iso)


# ---------------------------------------------------------------------------
# Stream 4: 15 Live Shopify DTC Brand Product Catalogs
# ---------------------------------------------------------------------------

def _seeded_shopify_brand(index: int, brand: str, domain: str, vertical: str) -> Dict[str, Any]:
    rng = np.random.default_rng(9100 + index)
    base_price = 28.0 + 18.0 * ((index * 5) % 9)
    tag_pool = [
        vertical.lower().split("&")[0].strip(),
        "bestseller",
        "sustainable",
        "everyday",
        "bundle",
        f"collection-{index % 4 + 1}",
        "dtc-core",
    ]
    products: List[Dict[str, Any]] = []
    for p_idx in range(12):
        price = round(float(base_price * (0.75 + 0.08 * p_idx) + rng.uniform(-3.0, 5.0)), 2)
        compare = round(price * 1.18, 2) if p_idx % 3 == 0 else None
        variants = []
        n_vars = 2 + (p_idx + index) % 4
        for v_idx in range(n_vars):
            variants.append(
                {
                    "id": index * 1000 + p_idx * 10 + v_idx,
                    "sku": f"{brand[:3].upper()}-{p_idx:02d}-{v_idx:02d}",
                    "price": f"{price + v_idx * 2.0:.2f}",
                    "compare_at_price": f"{compare:.2f}" if compare else None,
                    "available": bool((p_idx + v_idx + index) % 5 != 0),
                }
            )
        products.append(
            {
                "id": index * 100 + p_idx,
                "title": f"{brand} Signature Item {p_idx + 1}",
                "handle": f"{brand.lower().replace(' ', '-')}-item-{p_idx + 1}",
                "product_type": vertical,
                "updated_at": "2026-11-18T00:00:00Z",
                "tags": tag_pool[: 3 + (p_idx % 4)],
                "images": [
                    {
                        "src": "https://images.unsplash.com/photo-1523275335684-37898b6baf30?q=80&w=700&auto=format&fit=crop"
                    }
                ],
                "variants": variants,
            }
        )
    return {
        "index": index,
        "brand": brand,
        "domain": domain,
        "vertical": vertical,
        "source_mode": "seeded_fallback",
        "products": products,
    }


def _seeded_shopify_payload(captured_at_utc: str) -> Dict[str, Any]:
    return {
        "stream_id": "shopify_dtc",
        "source_label": "AffilAI Live Shopify DTC Storefront Product Catalogs (15 Brands)",
        "source_mode": "seeded_fallback",
        "captured_at_utc": captured_at_utc,
        "brands": [
            _seeded_shopify_brand(idx, brand, domain, vertical)
            for idx, (brand, domain, vertical) in enumerate(SHOPIFY_BRAND_SPECS)
        ],
    }


def collect_shopify_dtc_catalogs(
    offline_only: bool = False,
    cache_dir: Optional[Path] = None,
    client: Optional[httpx.Client] = None,
) -> Dict[str, Any]:
    """Harvest `/products.json?limit=25` across all 15 Shopify DTC brands."""
    stream_id = "shopify_dtc"
    now_iso = _utc_now_iso()
    cached = _load_cache(stream_id, cache_dir=cache_dir)

    if not offline_only:
        http_client, owns_client = _create_client(client)
        cached_by_brand = {
            b["brand"]: b for b in (cached or {}).get("brands", []) if "brand" in b
        }
        brands_out: List[Dict[str, Any]] = []
        try:
            for idx, (brand, domain, vertical) in enumerate(SHOPIFY_BRAND_SPECS):
                url = f"https://{domain}/products.json?limit=25"
                fetched_brand: Optional[Dict[str, Any]] = None
                try:
                    resp = _fetch_with_retry(http_client, url)
                    prods = resp.json().get("products", [])
                    if isinstance(prods, list) and len(prods) > 0:
                        fetched_brand = {
                            "index": idx,
                            "brand": brand,
                            "domain": domain,
                            "vertical": vertical,
                            "source_mode": "live_public_stream",
                            "products": prods,
                        }
                except Exception:
                    pass

                if fetched_brand is not None:
                    brands_out.append(fetched_brand)
                elif brand in cached_by_brand:
                    b_copy = dict(cached_by_brand[brand])
                    b_copy["index"] = idx
                    b_copy["source_mode"] = "cached_empirical"
                    brands_out.append(b_copy)
                else:
                    brands_out.append(_seeded_shopify_brand(idx, brand, domain, vertical))
                time.sleep(0.04)
        finally:
            if owns_client:
                http_client.close()

        live_count = sum(1 for b in brands_out if b["source_mode"] == "live_public_stream")
        cached_count = sum(1 for b in brands_out if b["source_mode"] == "cached_empirical")
        if live_count >= 8:
            overall_mode: SourceMode = "live_public_stream"
        elif cached_count > 0:
            overall_mode = "cached_empirical"
        else:
            overall_mode = "seeded_fallback"

        payload = {
            "stream_id": stream_id,
            "source_label": "AffilAI Live Shopify DTC Storefront Product Catalogs (15 Brands)",
            "source_mode": overall_mode,
            "captured_at_utc": now_iso,
            "brands": brands_out,
        }
        if live_count > 0:
            _save_cache(stream_id, payload, cache_dir=cache_dir)
        return payload

    if cached and len(cached.get("brands", [])) == len(SHOPIFY_BRAND_SPECS):
        cached_copy = dict(cached)
        cached_copy["source_mode"] = "cached_empirical"
        for b in cached_copy["brands"]:
            b["source_mode"] = "cached_empirical"
        return cached_copy

    return _seeded_shopify_payload(now_iso)


def collect_all_streams(
    offline_only: bool = False,
    cache_dir: Optional[Path] = None,
    client: Optional[httpx.Client] = None,
) -> Dict[str, Any]:
    """Execute Phase 1 collection across all 4 live empirical domains."""
    return {
        "usgs": collect_usgs_water_telemetry(offline_only=offline_only, cache_dir=cache_dir, client=client),
        "rfam": collect_rfam_rna_msa(offline_only=offline_only, cache_dir=cache_dir, client=client),
        "market": collect_quant_and_polymarket(offline_only=offline_only, cache_dir=cache_dir, client=client),
        "shopify": collect_shopify_dtc_catalogs(offline_only=offline_only, cache_dir=cache_dir, client=client),
    }
