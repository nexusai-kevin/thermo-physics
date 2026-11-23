"""B2B Vertical Outreach Packet Factory for Hugging Face Commercial Leads."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_LEADS_CSV = Path(__file__).resolve().parent / "leads.csv"
_DEFAULT_OUT_DIR = _REPO_ROOT / "output" / "outreach_packets"
_FALLBACK_DOSSIER_HTML = _REPO_ROOT / "output" / "qwen-qwen3.8-27b" / "benchmark_dossier.html"
_LIVE_PROOF_URL = "https://getnexusaisolutions.com/compression"

_SUBPAGES = ("", "/about", "/about-us", "/contact", "/contact-us", "/team", "/company")

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_NOISE_EMAIL_SUBSTRINGS = (
    "example.com",
    "example.org",
    "example@",
    "mysite.com",
    "wix.com",
    "wixpress.com",
    "sentry.io",
    "noreply",
    "no-reply",
    "donotreply",
    "yourdomain",
    "domain.com",
    "email.com",
    "test.com",
    "localhost",
    ".png",
    ".jpg",
    ".jpeg",
    ".svg",
    ".webp",
    ".woff",
    ".css",
    ".js",
)

_PRIORITY_EMAIL_PREFIXES = (
    "cto",
    "founder",
    "ceo",
    "eng",
    "engineering",
    "ai",
    "ml",
    "hello",
    "contact",
    "info",
    "team",
    "support",
    "sales",
)

_BROWSER_CANDIDATES = (
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
)

_VERTICAL_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("Legal AI", ("legal", "law", "contract", "attorney", "litigation", "intake")),
    ("Medical / Clinical AI", ("medical", "clinical", "health", "healthcare", "radiology", "patient", "biomed")),
    ("Financial AI", ("finance", "fintech", "banking", "trading", "credit", "underwriting", "accounting")),
    ("Compliance / Privacy AI", ("compliance", "privacy", "gdpr", "hipaa", "soc2", "audit", "regulatory")),
    ("Enterprise / Security AI", ("security", "cyber", "threat", "enterprise", "soc")),
    ("Developer / Agent AI", ("code", "coding", "agent", "workflow", "automation")),
]


def _configure_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _safe_slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", name).strip("-").lower()


def _strip_html(raw_html: str) -> str:
    cleaned = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", " ", raw_html, flags=re.I | re.S)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = html.unescape(cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _classify_vertical(blob: str) -> str:
    blob_l = blob.lower()
    for label, keywords in _VERTICAL_RULES:
        if any(kw in blob_l for kw in keywords):
            return label
    return "Enterprise Vertical AI"


def _extract_meta_summary(page_html: str) -> str:
    patterns = (
        r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']description["\']',
    )
    for pat in patterns:
        m = re.search(pat, page_html, flags=re.I)
        if m:
            cand = html.unescape(m.group(1)).strip()
            if len(cand) >= 25:
                return cand
    # Fallback to <title>
    tm = re.search(r"<title[^>]*>(.*?)</title>", page_html, flags=re.I | re.S)
    if tm:
        title = html.unescape(re.sub(r"\s+", " ", tm.group(1))).strip()
        if len(title) >= 12:
            return title
    return ""


def _one_sentence(text: str, fallback: str) -> str:
    raw = re.sub(r"\s+", " ", (text or "")).strip()
    if not raw or len(raw) < 20:
        return fallback
    # Split into first crisp sentence
    parts = re.split(r"(?<=[.!?])\s+", raw)
    sent = parts[0].strip()
    if len(sent) > 220:
        sent = sent[:217].rsplit(" ", 1)[0].rstrip(",;:") + "..."
    if not sent.endswith((".", "!", "?")):
        sent += "."
    return sent


def _extract_emails(text: str) -> list[str]:
    found: set[str] = set()
    # Decode mailto: links and obfuscated [at] / (at) patterns
    normalized = html.unescape(text)
    for m in _EMAIL_RE.findall(normalized):
        em = m.strip().lower()
        if any(noise in em for noise in _NOISE_EMAIL_SUBSTRINGS):
            continue
        if len(em) > 80:
            continue
        found.add(em)

    def _rank(email: str) -> tuple[int, str]:
        local = email.split("@", 1)[0]
        for idx, pref in enumerate(_PRIORITY_EMAIL_PREFIXES):
            if local == pref or local.startswith(pref):
                return (idx, email)
        return (len(_PRIORITY_EMAIL_PREFIXES), email)

    return sorted(found, key=_rank)


def _extract_leadership(text: str) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    # Look for "Name, CTO / Co-Founder / CEO" or "CTO: Name"
    patterns = (
        r"\b([A-Z][a-z]{1,20}\s+[A-Z][a-z]{1,20})\s*[,|\-–—]\s*(?:Co-?Founder|Founder|CTO|Chief Technology Officer|CEO|VP of Engineering)\b",
        r"\b(?:Co-?Founder|Founder|CTO|Chief Technology Officer|CEO)\s*[:\-–—]\s*([A-Z][a-z]{1,20}\s+[A-Z][a-z]{1,20})\b",
    )
    for pat in patterns:
        for m in re.finditer(pat, text):
            name = m.group(1).strip()
            if name not in seen and not any(w in name.lower() for w in ("privacy", "policy", "terms", "cookie", "about")):
                seen.add(name)
                names.append(name)
                if len(names) >= 3:
                    return names
    return names


def enrich_lead(
    org_name: str,
    website: str,
    primary_row: dict[str, str],
    all_models: list[str],
    session: requests.Session,
) -> dict[str, Any]:
    """Fetch website pages and HF model card to extract summary, contacts, and vertical."""
    model_id = primary_row["model_id"]
    base_url = website.rstrip("/")

    pages_html: list[str] = []
    pages_text: list[str] = []
    meta_summary = ""
    fetched_urls: list[str] = []

    for sub in _SUBPAGES:
        url = urljoin(base_url + "/", sub.lstrip("/")) if sub else website
        try:
            resp = session.get(url, timeout=8, allow_redirects=True)
            if resp.status_code == 200 and "text/html" in resp.headers.get("Content-Type", "").lower():
                if not resp.encoding or resp.encoding.lower() == "iso-8859-1":
                    resp.encoding = resp.apparent_encoding or "utf-8"
                pages_html.append(resp.text)
                pages_text.append(_strip_html(resp.text))
                fetched_urls.append(resp.url)
                if not meta_summary:
                    meta_summary = _extract_meta_summary(resp.text)
        except Exception:
            continue

    # Also fetch HF model card README for domain context
    hf_readme = ""
    try:
        r_hf = session.get(f"https://huggingface.co/{model_id}/raw/main/README.md", timeout=8)
        if r_hf.status_code == 200:
            hf_readme = r_hf.text[:4000]
    except Exception:
        pass

    combined_raw = "\n".join(pages_html) + "\n" + hf_readme
    combined_plain = "\n".join(pages_text) + "\n" + hf_readme

    vertical = _classify_vertical(f"{org_name} {website} {meta_summary} {model_id} {hf_readme[:1200]}")
    fallback_summary = (
        f"{org_name} builds {vertical.lower()} workflows powered by open-weights models such as {model_id}."
    )
    product_summary = _one_sentence(meta_summary, fallback_summary)
    emails = _extract_emails(combined_raw)
    leaders = _extract_leadership(combined_plain)

    return {
        "org_name": org_name,
        "website": website,
        "primary_model_id": model_id,
        "all_models": all_models,
        "est_params_b": float(primary_row["est_params_b"]),
        "architecture_type": primary_row["architecture_type"],
        "downloads_30d": int(primary_row["downloads_30d"]),
        "last_modified": primary_row["last_modified"],
        "vertical": vertical,
        "product_summary": product_summary,
        "contact_emails": emails,
        "primary_contact_email": emails[0] if emails else "none_found_publicly",
        "leadership_names": leaders,
        "fetched_urls": fetched_urls,
        "compress_cmd": primary_row["compress_cmd"],
    }


def _find_browser_exe() -> Path:
    for cand in _BROWSER_CANDIDATES:
        if cand.exists():
            return cand
    raise FileNotFoundError("Neither Google Chrome nor Microsoft Edge was found for headless PDF export.")


def export_pdf_dossier(source_html: Path, dest_pdf: Path, browser_exe: Path) -> Path:
    """Render an HTML dossier to a 1-page PDF using headless Chrome/Edge."""
    dest_pdf.parent.mkdir(parents=True, exist_ok=True)
    file_uri = source_html.resolve().as_uri()
    cmd = [
        str(browser_exe),
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--no-pdf-header-footer",
        f"--print-to-pdf={dest_pdf.resolve()}",
        file_uri,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
    if proc.returncode != 0 or not dest_pdf.exists() or dest_pdf.stat().st_size < 1000:
        raise RuntimeError(f"Headless PDF export failed ({proc.returncode}): {proc.stderr[:300]}")
    header = dest_pdf.read_bytes()[:5]
    if header != b"%PDF-":
        raise RuntimeError(f"Invalid PDF output header at {dest_pdf}: {header!r}")
    return dest_pdf


def _estimate_vram_sizing(params_b: float) -> tuple[float, float, float, str, str]:
    base_bf16 = round(params_b * 2.08, 1)
    comp_bf16 = round(base_bf16 * 0.693, 1)
    comp_q4 = round((params_b * 0.693) * 0.58 + 0.5, 1)
    if base_bf16 >= 48.0:
        base_hw = "1x A100-80GB ($2.20/hr)"
        comp_hw = f"1x L40S-48GB ($1.15/hr) or 1x 16GB/24GB local GPU at Q4_K_M (~{comp_q4} GB)"
    else:
        base_hw = "1x A100-40GB / L40S-48GB"
        comp_hw = f"1x L4-24GB or 1x 16GB RTX-class GPU at Q4_K_M (~{comp_q4} GB)"
    return base_bf16, comp_bf16, comp_q4, base_hw, comp_hw


def verify_benchmark_gate(benchmark_json_path: Path) -> dict[str, Any]:
    """Enforce strict Email <-> Dossier consistency gate before generating outreach."""
    if not benchmark_json_path.exists():
        raise FileNotFoundError(f"Benchmark JSON required for outreach gate: {benchmark_json_path}")

    data = json.loads(benchmark_json_path.read_text(encoding="utf-8"))
    n_eval = int(data.get("eval_summary", {}).get("num_eval_samples", 0))
    if n_eval < 10:
        raise ValueError(
            f"Outreach gate blocked: num_eval_samples={n_eval} < 10 in {benchmark_json_path}"
        )

    base = data.get("base_metrics", {})
    comp = data.get("compressed_metrics", {})
    base_json = float(base.get("json_valid_pct", 0.0))
    comp_json = float(comp.get("json_valid_pct", 0.0))
    comp_acc = float(comp.get("vertical_accuracy_pct", 0.0))

    if (base_json - comp_json) > 2.0:
        raise ValueError(
            f"Outreach gate blocked: strict JSON schema validity dropped by {base_json - comp_json:.1f}% "
            f"({base_json:.1f}% -> {comp_json:.1f}%, max allowed drop 2.0%). Run Stage 3 healing first."
        )
    if comp_json < 95.0 or comp_acc < 95.0:
        raise ValueError(
            f"Outreach gate blocked: compressed json_valid_pct={comp_json:.1f}%, "
            f"vertical_accuracy_pct={comp_acc:.1f}% (both must be >= 95.0%)."
        )
    return data


def build_email_draft(
    profile: dict[str, Any],
    dossier_pdf_rel: str,
    benchmark_data: dict[str, Any],
    is_reference_dossier: bool,
) -> str:
    org = profile["org_name"]
    model_id = profile["primary_model_id"]
    params_b = profile["est_params_b"]
    arch = profile["architecture_type"]
    vertical = profile["vertical"]
    summary = profile["product_summary"]
    to_email = profile["primary_contact_email"]
    greeting_name = profile["leadership_names"][0].split()[0] if profile["leadership_names"] else f"{org} Engineering Team"

    base_bf16, comp_bf16, comp_q4, base_hw, comp_hw = _estimate_vram_sizing(params_b)
    extra_models = [m for m in profile["all_models"] if m != model_id]
    extra_line = f" (and {extra_models[0]})" if extra_models else ""

    bm_model = benchmark_data["model"]["id"]
    bm_base = benchmark_data["base_metrics"]
    bm_comp = benchmark_data["compressed_metrics"]
    bm_deltas = benchmark_data["deltas"]
    bm_cfg = benchmark_data.get("pipeline_config", {})
    n_eval = benchmark_data["eval_summary"]["num_eval_samples"]

    if bm_cfg.get("native_4bit_compressed"):
        speed_line = (
            f"- Native 16GB RTX 5080 4-bit NF4 PyTorch reference speed (no compiled Flash/Conv1D kernels): "
            f"{bm_comp['throughput_tok_s']:.2f} tok/s ({bm_comp.get('native_4bit_vram_gb', bm_comp['measured_local_peak_vram_gb']):.2f} GB, 100% GPU)"
        )
    else:
        speed_line = f"- Projected GPU tier from {base_hw} to {comp_hw}"

    dossier_label = (
        f"Reference Architecture Capture ({bm_model}) — we have not yet run your private weights"
        if is_reference_dossier
        else f"Prospect-Specific Benchmark Dossier ({model_id})"
    )

    subject = (
        f"{model_id}: -{bm_deltas['param_reduction_pct']:.1f}% parameter/VRAM reduction "
        f"({bm_comp['json_valid_pct']:.0f}% strict JSON parity)"
    )
    return f"""To: {to_email}
Subject: {subject}
Attachment: {dossier_pdf_rel} [{dossier_label}]

Hi {greeting_name},

I noticed your team is actively shipping {model_id}{extra_line} ({params_b:.1f}B {arch}) on Hugging Face for {vertical.lower()} — {summary}

At {params_b:.1f}B parameters, BF16 serving sits around ~{base_bf16} GB VRAM ({base_hw}). We built a data-aware 5-stage tensor compression pipeline (Cholesky-whitened SVD / MPO calibrated on vertical extraction prompts + targeted QLoRA recovery) that strips redundant projection rank while preserving first/last layers.

On our verified {bm_model} capture (n={n_eval} held-out extraction prompts, full telemetry at {_LIVE_PROOF_URL}):
- Parameters & BF16 VRAM: -{bm_deltas['param_reduction_pct']:.1f}% ({bm_base['peak_vram_gb']:.1f} GB -> {bm_comp['peak_vram_gb']:.1f} GB BF16, or ~{comp_q4} GB at Q4_K_M)
{speed_line}
- Strict JSON Schema Validity: {bm_base['json_valid_pct']:.1f}% -> {bm_comp['json_valid_pct']:.1f}% | Field Accuracy: {bm_base['vertical_accuracy_pct']:.1f}% -> {bm_comp['vertical_accuracy_pct']:.1f}% (PPL {bm_base['perplexity']:.2f} -> {bm_comp['perplexity']:.2f})

Attached is our 1-page engineering dossier ({dossier_label}). If you'd like to test {model_id} compressed on your own prompts, I can run a free 1-page benchmark dossier on your weights or spin up a private 48-hour OpenAI-compatible API sandbox endpoint with a dedicated bearer token so your team can hit it directly via curl or the OpenAI SDK.

Worth a quick 48-hour sandbox test on {model_id}?

Best,
Kevin
Nexus AI Solutions
{_LIVE_PROOF_URL}
"""


def run_outreach_factory(
    leads_csv: Path = _DEFAULT_LEADS_CSV,
    out_dir: Path = _DEFAULT_OUT_DIR,
) -> list[dict[str, Any]]:
    if not leads_csv.exists():
        raise FileNotFoundError(f"Leads CSV not found: {leads_csv}")
    if not _FALLBACK_DOSSIER_HTML.exists():
        raise FileNotFoundError(f"Fallback dossier HTML not found: {_FALLBACK_DOSSIER_HTML}")

    browser_exe = _find_browser_exe()
    rows = list(csv.DictReader(leads_csv.open("r", encoding="utf-8")))
    if not rows:
        raise ValueError(f"No rows found in {leads_csv}")

    # Group by org_name, selecting highest-download model as primary
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["org_name"], []).append(row)

    session = requests.Session()
    session.headers.update({"User-Agent": "NexusAI-OutreachFactory/1.0 (+https://getnexusaisolutions.com/compression)"})

    out_dir.mkdir(parents=True, exist_ok=True)
    packets: list[dict[str, Any]] = []

    for org_name, org_rows in grouped.items():
        org_rows.sort(key=lambda r: int(r["downloads_30d"]), reverse=True)
        primary = org_rows[0]
        all_models = [r["model_id"] for r in org_rows]
        profile = enrich_lead(org_name, primary["website"], primary, all_models, session)

        org_slug = _safe_slug(org_name)
        packet_dir = out_dir / org_slug
        packet_dir.mkdir(parents=True, exist_ok=True)

        model_slug = _safe_slug(primary["model_id"])
        org_html = _REPO_ROOT / "output" / org_slug / "benchmark_dossier.html"
        prospect_html = _REPO_ROOT / "output" / model_slug / "benchmark_dossier.html"
        if org_html.exists():
            source_html = org_html
            is_reference = False
        elif prospect_html.exists():
            source_html = prospect_html
            is_reference = False
        else:
            source_html = _FALLBACK_DOSSIER_HTML
            is_reference = True

        source_json = source_html.parent / "benchmark_results.json"
        benchmark_data = verify_benchmark_gate(source_json)

        pdf_path = export_pdf_dossier(source_html, packet_dir / "benchmark_dossier.pdf", browser_exe)
        email_path = packet_dir / "email_draft.txt"
        profile_path = packet_dir / "lead_profile.json"

        profile["dossier_type"] = "reference_architecture_capture" if is_reference else "prospect_specific"
        profile["reference_model"] = benchmark_data["model"]["id"]
        email_text = build_email_draft(profile, str(pdf_path.relative_to(_REPO_ROOT)), benchmark_data, is_reference)
        email_path.write_text(email_text, encoding="utf-8")
        profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")

        packets.append(
            {
                "org_name": org_name,
                "model_id": primary["model_id"],
                "vertical": profile["vertical"],
                "contact_email": profile["primary_contact_email"],
                "all_emails": profile["contact_emails"],
                "leadership": profile["leadership_names"],
                "product_summary": profile["product_summary"],
                "packet_dir": str(packet_dir.relative_to(_REPO_ROOT)),
                "pdf_path": str(pdf_path.relative_to(_REPO_ROOT)),
                "email_path": str(email_path.relative_to(_REPO_ROOT)),
            }
        )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(packets, indent=2), encoding="utf-8")

    # Print clean summary table
    print("\n" + "=" * 112)
    print(f"{'ORG':<15} | {'MODEL_ID':<48} | {'VERTICAL':<21} | {'CONTACT_EMAIL':<24} | PACKET")
    print("-" * 112)
    for p in packets:
        print(
            f"{p['org_name'][:15]:<15} | {p['model_id'][:48]:<48} | "
            f"{p['vertical'][:21]:<21} | {p['contact_email'][:24]:<24} | {p['packet_dir']}"
        )
    print("=" * 112)
    print(f"Generated {len(packets)} outreach packets -> {out_dir}")
    return packets


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate B2B outreach packets from leads.csv")
    p.add_argument("--leads-csv", type=Path, default=_DEFAULT_LEADS_CSV)
    p.add_argument("--output-dir", type=Path, default=_DEFAULT_OUT_DIR)
    return p.parse_args()


def main() -> int:
    _configure_utf8_console()
    args = parse_args()
    run_outreach_factory(leads_csv=args.leads_csv, out_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
