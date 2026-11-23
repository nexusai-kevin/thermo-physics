"""Hugging Face Commercial Lead Scraper for 14B-70B Open-Weights Models."""

from __future__ import annotations

import argparse
import csv
import html
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from huggingface_hub import HfApi

SEARCH_QUERIES = [
    "legal",
    "medical",
    "finance",
    "security",
    "compliance",
    "code qwen",
    "agent qwen",
    "llama",
    "qwen",
    "mistral",
    "mixtral",
    "deepseek",
]

# Community quantization/repack accounts to skip so leads focus on primary builders/hosts
REPACKER_ACCOUNTS = {
    "bartowski",
    "thebloke",
    "mradermacher",
    "lmstudio-community",
    "mlx-community",
    "quantfactory",
    "unsloth",
    "maziyarpanahi",
    "hugging-quants",
    "cjpais",
    "turboderp",
    "lone-striker",
    "lonestriker",
    "second-state",
    "tensorblock",
    "gguf-org",
}

# Base model vendors / foundation labs to exclude when targeting vertical SaaS fine-tunes
EXCLUDED_BASE_ORGS = {
    "qwen",
    "nvidia",
    "redhatai",
    "meta-llama",
    "mistralai",
    "deepseek-ai",
    "litert-community",
}

_QUANT_REPO_PATTERN = re.compile(
    r"(?:^|[-_.])(fp8|nvfp4|awq|gptq|gguf|llama\.cpp|exl2|mlx|bnb-4bit)(?:$|[-_.])",
    re.IGNORECASE,
)

ACADEMIC_DOMAIN_SUFFIXES = (
    ".edu",
    ".ac.uk",
    ".ac.jp",
    ".ac.cn",
    ".ac.kr",
    ".ac.at",
    ".edu.cn",
    ".edu.au",
    ".edu.sg",
    ".gov",
    ".gov.uk",
)

# Exclude AI infra, router, kernel, and quantization labs so leads focus on vertical B2B buyers
EXCLUDED_INFRA_ORGS = {
    "bottlecapai",
    "orcarouter",
    "basecompute",
    "openvino",
    "intel-ai",
    "axera-tech",
    "cyankiwi",
    "inferact",
    "radixark",
    "obliteratus",
}

INFRA_LAB_KEYWORDS = (
    "router",
    "inference",
    "quant",
    "kernel",
    "benchmark",
    "llm api",
    "bottlecap",
    "orcarouter",
    "vllm",
    "tensorrt",
    "openvino",
    "exllama",
    "gguf",
    "awq",
    "gptq",
)

VERTICAL_DOMAIN_KEYWORDS = (
    "legal",
    "law",
    "health",
    "medical",
    "clinical",
    "bio",
    "pharma",
    "finance",
    "fintech",
    "bank",
    "compliance",
    "enterprise",
    "security",
    "cyber",
    "agent",
    "code",
    "coding",
    "audit",
    "insurance",
    "logistics",
)

ACADEMIC_KEYWORDS = (
    "university",
    "college",
    "school of ",
    "faculty of ",
    "department of ",
    "institute of technology",
    "polytechnic",
    "academy of sciences",
    "national laboratory",
    "academic",
)

SOCIAL_OR_HOST_DOMAINS = {
    "huggingface.co",
    "github.com",
    "gitlab.com",
    "twitter.com",
    "x.com",
    "linkedin.com",
    "discord.gg",
    "discord.com",
    "youtube.com",
    "arxiv.org",
    "t.me",
}


def _configure_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def estimate_params_b(model_info: Any) -> float | None:
    """Estimate parameter count in billions from safetensors metadata or model ID."""
    st = getattr(model_info, "safetensors", None)
    if st is not None:
        total = getattr(st, "total", None)
        if total and total > 0:
            return round(float(total) / 1e9, 2)

    mid = getattr(model_info, "id", "") or ""
    # Check MoE pattern like 8x7B or 16x3.8B
    moe_match = re.search(r"(\d+)\s*[xX]\s*(\d+(?:\.\d+)?)\s*[bB]\b", mid)
    if moe_match:
        n_exp = float(moe_match.group(1))
        per_exp = float(moe_match.group(2))
        return round(n_exp * per_exp * 0.85, 2)  # Shared attention/norms adjustment

    # Check standard pattern like 14B, 32B, 70B, 72B, 27B
    b_matches = re.findall(r"(?<![a-zA-Z0-9])(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z0-9])", mid)
    if b_matches:
        vals = [float(v) for v in b_matches if 1.0 <= float(v) <= 500.0]
        if vals:
            return round(max(vals), 2)
    return None


def classify_architecture(model_info: Any) -> str:
    """Classify model architecture as 'MoE' or 'Dense'."""
    mid = (getattr(model_info, "id", "") or "").lower()
    tags = [str(t).lower() for t in (getattr(model_info, "tags", None) or [])]
    cfg = getattr(model_info, "config", None) or {}
    if not isinstance(cfg, dict):
        cfg = {}

    model_type = str(cfg.get("model_type", "")).lower()
    archs = " ".join(str(a).lower() for a in (cfg.get("architectures") or []))
    num_experts = (
        cfg.get("num_local_experts")
        or cfg.get("n_routed_experts")
        or cfg.get("num_experts")
        or 0
    )

    moe_signals = (
        int(num_experts) > 1
        or "moe" in model_type
        or "mixtral" in model_type
        or "moe" in archs
        or "mixtral" in archs
        or "mixtral" in mid
        or "moe" in mid
        or bool(re.search(r"\d+x\d+(?:\.\d+)?b", mid))
        or bool(re.search(r"-a\d+(?:\.\d+)?b", mid))
        or any("moe" in t or "mixtral" in t for t in tags)
    )
    return "MoE" if moe_signals else "Dense"


def _is_repacker_or_quant(model_id: str) -> bool:
    if "/" not in model_id:
        return True
    owner, repo = model_id.split("/", 1)
    owner_l = owner.lower()
    if owner_l in REPACKER_ACCOUNTS or owner_l in EXCLUDED_BASE_ORGS:
        return True
    if _QUANT_REPO_PATTERN.search(repo):
        return True
    return False


def _extract_org_website_from_html(page_html: str) -> str | None:
    """Extract the custom external website link from an HF organization profile page."""
    # Pattern 1: HF Org header external link (<a ... href="https://..." rel="nofollow" target="_blank">)
    for m in re.finditer(
        r'<a[^>]+href="(https?://[^"]+)"[^>]*rel="nofollow"[^>]*target="_blank"',
        page_html,
    ):
        url = html.unescape(m.group(1)).strip()
        host = urlparse(url).netloc.lower()
        if host and not any(s in host for s in SOCIAL_OR_HOST_DOMAINS):
            return url

    # Pattern 2: Fallback reversed attribute order
    for m in re.finditer(
        r'<a[^>]+rel="nofollow"[^>]+href="(https?://[^"]+)"',
        page_html,
    ):
        url = html.unescape(m.group(1)).strip()
        host = urlparse(url).netloc.lower()
        if host and not any(s in host for s in SOCIAL_OR_HOST_DOMAINS):
            return url
    return None


def _extract_org_bio_from_html(page_html: str) -> str:
    """Extract organization bio/description text from HF org profile HTML."""
    for pat in (
        r'<meta[^>]+property="og:description"[^>]+content="([^"]*)"',
        r'<meta[^>]+name="description"[^>]+content="([^"]*)"',
    ):
        m = re.search(pat, page_html, flags=re.IGNORECASE)
        if m:
            return html.unescape(m.group(1)).strip()
    return ""


def is_excluded_infra_org(
    org_name: str,
    fullname: str,
    website: str,
    bio: str = "",
) -> bool:
    """Return True if the organization is an AI infra, router, or quantization lab."""
    if org_name.lower() in EXCLUDED_INFRA_ORGS:
        return True
    combined = f"{org_name} {fullname} {website} {bio}".lower()
    return any(kw in combined for kw in INFRA_LAB_KEYWORDS)


def score_vertical_relevance(
    org_info: dict[str, Any],
    model_id: str,
    tags: list[str] | None = None,
) -> tuple[int, list[str]]:
    """Score how strongly an org/model matches vertical B2B SaaS domains."""
    blob = " ".join(
        [
            str(org_info.get("org_name", "")),
            str(org_info.get("fullname", "")),
            str(org_info.get("website", "")),
            str(org_info.get("bio", "")),
            model_id,
            " ".join(str(t) for t in (tags or [])),
        ]
    ).lower()
    matched = [kw for kw in VERTICAL_DOMAIN_KEYWORDS if kw in blob]
    return len(matched), matched


def inspect_commercial_org(
    org_name: str,
    session: requests.Session,
    cache: dict[str, dict[str, Any] | None],
) -> dict[str, Any] | None:
    """Verify org exists, has a custom external website, and is not academic/infra."""
    if org_name in cache:
        return cache[org_name]

    if org_name.lower() in EXCLUDED_INFRA_ORGS or org_name.lower() in EXCLUDED_BASE_ORGS:
        cache[org_name] = None
        return None

    try:
        ov_resp = session.get(
            f"https://huggingface.co/api/organizations/{org_name}/overview",
            timeout=10,
        )
        if ov_resp.status_code != 200:
            cache[org_name] = None
            return None
        ov = ov_resp.json()
    except Exception:
        cache[org_name] = None
        return None

    org_type = str(ov.get("orgType") or "").lower()
    if org_type in ("university", "non-profit", "classroom"):
        cache[org_name] = None
        return None

    fullname = str(ov.get("fullname") or org_name)
    text_check = f"{org_name} {fullname}".lower()
    if any(kw in text_check for kw in ACADEMIC_KEYWORDS):
        cache[org_name] = None
        return None

    # Fetch org profile page to inspect custom website link and bio
    website = ov.get("website") or ov.get("details")
    bio = ""
    try:
        page_resp = session.get(f"https://huggingface.co/{org_name}", timeout=10)
        if page_resp.status_code == 200:
            if not website or not str(website).startswith("http"):
                website = _extract_org_website_from_html(page_resp.text)
            bio = _extract_org_bio_from_html(page_resp.text)
    except Exception:
        pass

    if not website:
        cache[org_name] = None
        return None

    host = urlparse(website).netloc.lower()
    if not host or any(host.endswith(suf) for suf in ACADEMIC_DOMAIN_SUFFIXES):
        cache[org_name] = None
        return None

    if is_excluded_infra_org(org_name, fullname, website, bio):
        cache[org_name] = None
        return None

    result = {
        "org_name": org_name,
        "fullname": fullname,
        "website": website,
        "bio": bio,
        "is_verified": bool(ov.get("isVerified", False)),
        "num_users": int(ov.get("numUsers", 0) or 0),
    }
    cache[org_name] = result
    return result


def scrape_commercial_leads(
    limit: int = 50,
    days: int = 180,
    min_params: float = 13.5,
    max_params: float = 72.5,
    min_downloads: int = 25,
    max_downloads: int = 50000,
    max_scan: int = 8000,
    output_path: str | Path = "tools/lead-scraper/leads.csv",
) -> list[dict[str, Any]]:
    """Query HF Hub for 14B-70B models updated recently by commercial organizations."""
    api = HfApi()
    session = requests.Session()
    session.headers.update({"User-Agent": "VerticalTensorCompression-LeadScraper/1.0"})

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    seen_models: set[str] = set()
    org_cache: dict[str, dict[str, Any] | None] = {}
    leads: list[dict[str, Any]] = []
    scanned = 0

    for query in SEARCH_QUERIES:
        if len(leads) >= limit or scanned >= max_scan:
            break

        # Query across downloads, trending, and recently modified to reach the mid-download band
        for sort_key in ("downloads", "trendingScore", "lastModified"):
            if len(leads) >= limit or scanned >= max_scan:
                break

            try:
                iterator = api.list_models(
                    search=query,
                    sort=sort_key,
                    limit=min(1000, max_scan - scanned),
                    expand=["safetensors", "tags", "config", "lastModified", "downloads"],
                )
            except Exception as exc:
                print(f"[Warn] list_models failed for query={query} sort={sort_key}: {exc}")
                continue

            for m in iterator:
                scanned += 1
                model_id = getattr(m, "id", "")
                if not model_id or model_id in seen_models:
                    continue
                seen_models.add(model_id)

                if _is_repacker_or_quant(model_id):
                    continue

                downloads = int(getattr(m, "downloads", 0) or 0)
                if not (min_downloads <= downloads <= max_downloads):
                    continue

                last_mod = getattr(m, "last_modified", None)
                if last_mod is None:
                    continue
                if last_mod.tzinfo is None:
                    last_mod = last_mod.replace(tzinfo=timezone.utc)
                if last_mod < cutoff:
                    continue

                params_b = estimate_params_b(m)
                if params_b is None or not (min_params <= params_b <= max_params):
                    continue

                org_name = model_id.split("/", 1)[0]
                org_info = inspect_commercial_org(org_name, session, org_cache)
                if org_info is None:
                    continue

                arch_type = classify_architecture(m)
                tags = list(getattr(m, "tags", None) or [])
                v_score, v_matches = score_vertical_relevance(org_info, model_id, tags)
                slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", model_id).strip("-").lower()
                compress_cmd = (
                    f"python tools/tensor-compressor/compress_pipeline.py "
                    f"--model {model_id} --ratio 0.35 --output-dir output/{slug}"
                )

                leads.append(
                    {
                        "_vertical_score": v_score,
                        "org_name": org_name,
                        "website": org_info["website"],
                        "model_id": model_id,
                        "est_params_b": params_b,
                        "architecture_type": arch_type,
                        "downloads_30d": downloads,
                        "last_modified": last_mod.strftime("%Y-%m-%d"),
                        "compress_cmd": compress_cmd,
                    }
                )
                v_tag = f" [vertical:{','.join(v_matches[:3])}]" if v_matches else ""
                print(
                    f"[{len(leads):02d}/{limit}] {org_name} ({org_info['website']}) | "
                    f"{model_id} ({params_b}B {arch_type}, {downloads:,} dl){v_tag}"
                )

                if len(leads) >= limit or scanned >= max_scan:
                    break

    # Prioritize vertical B2B domain matches, then downloads_30d descending
    leads.sort(key=lambda r: (r["_vertical_score"] > 0, r["_vertical_score"], r["downloads_30d"]), reverse=True)
    for r in leads:
        r.pop("_vertical_score", None)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "org_name",
        "website",
        "model_id",
        "est_params_b",
        "architecture_type",
        "downloads_30d",
        "last_modified",
        "compress_cmd",
    ]
    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(leads)

    print(f"\nExported {len(leads)} commercial leads (scanned {scanned} models) -> {out_file}")
    return leads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape Hugging Face Hub for commercial orgs running 14B-70B models."
    )
    parser.add_argument("--limit", type=int, default=50, help="Maximum commercial leads to export")
    parser.add_argument("--days", type=int, default=180, help="Recency window in days (default: 180)")
    parser.add_argument("--min-params", type=float, default=13.5, help="Minimum model size in B")
    parser.add_argument("--max-params", type=float, default=72.5, help="Maximum model size in B")
    parser.add_argument("--min-downloads", type=int, default=25, help="Minimum 30d downloads (default: 25)")
    parser.add_argument("--max-downloads", type=int, default=50000, help="Maximum 30d downloads (default: 50000)")
    parser.add_argument("--max-scan", type=int, default=8000, help="Max HF models to scan (default: 8000)")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "leads.csv"),
        help="Output CSV path",
    )
    return parser.parse_args()


def main() -> int:
    _configure_utf8_console()
    args = parse_args()
    scrape_commercial_leads(
        limit=args.limit,
        days=args.days,
        min_params=args.min_params,
        max_params=args.max_params,
        min_downloads=args.min_downloads,
        max_downloads=args.max_downloads,
        max_scan=args.max_scan,
        output_path=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
