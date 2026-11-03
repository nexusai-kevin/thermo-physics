#!/usr/bin/env python3
"""
Nexus AI Solutions — Backend AI Intent Monitoring & Live Reddit Community Scraper
=================================================================================
Scrapes REAL, live threads from Reddit's Atom/RSS search feeds across:
  - r/sweatystartup, r/Contractor, r/smallbusiness, r/HVAC, r/Plumbing,
    r/electricians, r/Roofing
Finds active threads where business owners are asking about:
  - Missing phone calls / calls going to voicemail
  - Answering services / virtual receptionists being too expensive
  - After-hours emergency dispatch / ServiceTitan / Housecall Pro / Jobber
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
import pathlib
import re
import subprocess
import sys
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from typing import List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TARGET_SUBREDDITS = [
    "sweatystartup",
    "Contractor",
    "smallbusiness",
    "HVAC",
    "Plumbing",
    "electricians",
    "Roofing",
]

SEARCH_QUERIES = [
    "answering service",
    "missed calls",
    "virtual receptionist",
    "voicemail contractor",
    "after hours calls",
]

HIGH_INTENT_PHRASES = {
    r"miss(?:ed|ing)\s+(?:phone\s+)?calls?": 35,
    r"answering\s+service": 35,
    r"virtual\s+receptionist": 35,
    r"receptionist": 25,
    r"voicemail": 30,
    r"after[\s-]hours": 30,
    r"missed[\s-]call\s+text": 40,
    r"answer(?:ing)?\s+(?:the\s+)?phone": 30,
    r"while\s+on\s+a\s+job": 25,
    r"ruby|smith\.ai|nexa|patlive|call\s+center|openphone|grasshopper": 30,
}

TRADE_AND_CRM_SIGNALS = {
    r"\b(?:plumb(?:ing|er)|hvac|electric(?:al|ian)|roof(?:ing|er)|contractor|handyman)\b": 15,
    r"\b(?:servicetitan|housecall\s*pro|jobber|fieldedge|google\s+calendar|crm)\b": 20,
    r"\b(?:emergency|dispatch|service\s+call|leads|customers)\b": 10,
}


@dataclass
class IntentLead:
    source: str
    community: str
    author: str
    title: str
    snippet: str
    url: str
    intent_score: int
    matched_triggers: List[str]
    detected_trade: str
    detected_crm: str
    suggested_reply: str
    captured_at: str
    post_date: str = ""


def strip_html(raw_html: str) -> str:
    text = html.unescape(raw_html or "")
    text = re.sub(r"<[^>]+>", " ", text)
    return " ".join(text.split())


def detect_trade(text: str, subreddit: str = "") -> str:
    low = f"{subreddit} {text}".lower()
    if "plumb" in low:
        return "Plumbing"
    if "hvac" in low or "furnace" in low or "air condition" in low:
        return "HVAC"
    if "electric" in low:
        return "Electrical"
    if "roof" in low:
        return "Roofing"
    if "contractor" in low or "sweatystartup" in low:
        return "Trade Contractor"
    return "Small Business / Home Services"


def detect_crm(text: str) -> str:
    low = text.lower()
    for crm in [
        "ServiceTitan",
        "Housecall Pro",
        "Jobber",
        "FieldEdge",
        "Google Calendar",
        "OpenPhone",
        "Grasshopper",
        "HighLevel",
    ]:
        if crm.lower() in low:
            return crm
    return "Direct Calendar / Phone"


def score_post(title: str, body: str) -> tuple[int, List[str]]:
    combined = f"{title}\n{body}"
    score = 0
    matches: List[str] = []

    for pattern, weight in HIGH_INTENT_PHRASES.items():
        found = re.findall(pattern, combined, flags=re.IGNORECASE)
        if found:
            score += weight
            matches.append(found[0].strip().lower())

    for pattern, weight in TRADE_AND_CRM_SIGNALS.items():
        found = re.findall(pattern, combined, flags=re.IGNORECASE)
        if found:
            score += weight
            matches.append(found[0].strip().lower())

    return min(score, 100), sorted(set(matches))


def build_suggested_reply(trade: str, crm: str, author: str) -> str:
    crm_hook = (
        f"that syncs straight with {crm}"
        if crm != "Direct Calendar / Phone"
        else "that plugs straight into Jobber, Housecall Pro, ServiceTitan, or Google Calendar"
    )
    return (
        f"Hey {author} — almost 80% of callers who hit voicemail hang up and call the next listing on "
        f"Google Maps within 45 seconds. Instead of paying $600–$1,200/mo for a call center that just "
        f"takes messages, we built an 8-second automated missed-call text-back & AI dispatcher for "
        f"{trade.lower()} shops {crm_hook}. You can test the live 8-second simulator or run your numbers "
        f"at https://getnexusaisolutions.com/funnel"
    )


def fetch_live_reddit_rss(limit_per_query: int = 12) -> List[dict]:
    """Fetches real, live Reddit threads via Reddit's public Atom RSS search endpoint."""
    posts: List[dict] = []
    sub_filter = "+".join(TARGET_SUBREDDITS)
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    for query in SEARCH_QUERIES:
        encoded_q = urllib.parse.quote(query)
        url = (
            f"https://www.reddit.com/r/{sub_filter}/search.rss"
            f"?q={encoded_q}&restrict_sr=on&sort=relevance&t=year"
        )
        try:
            res = subprocess.run(
                [
                    "curl.exe",
                    "-s",
                    "-L",
                    "-A",
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:131.0) Gecko/20100101 Firefox/131.0",
                    url,
                ],
                capture_output=True,
                timeout=10,
                check=False,
            )
            xml_text = res.stdout.decode("utf-8", errors="ignore").strip()
            if not xml_text.startswith("<?xml") and "<feed" not in xml_text:
                continue

            root = ET.fromstring(xml_text)
            for entry in root.findall("atom:entry", ns):
                title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
                link_el = entry.find("atom:link", ns)
                link = link_el.attrib.get("href", "") if link_el is not None else ""
                author_el = entry.find("atom:author/atom:name", ns)
                author = author_el.text.strip() if (author_el is not None and author_el.text) else "u/reddit_user"
                cat_el = entry.find("atom:category", ns)
                subreddit = (
                    f"r/{cat_el.attrib.get('term', 'Contractor')}"
                    if cat_el is not None
                    else "r/sweatystartup"
                )
                updated = (entry.findtext("atom:updated", default="", namespaces=ns) or "")[:10]
                content_raw = entry.findtext("atom:content", default="", namespaces=ns) or ""
                body = strip_html(content_raw)

                if title and link:
                    posts.append(
                        {
                            "source": "Reddit (Live)",
                            "community": subreddit,
                            "author": author,
                            "title": title,
                            "body": body,
                            "url": link,
                            "post_date": updated or "Recent",
                        }
                    )
        except Exception as exc:
            print(f"[!] RSS parse warning ({query}): {exc}", file=sys.stderr)
            continue

    return posts


def run_intent_monitor(include_live: bool = True, min_score: int = 35) -> List[IntentLead]:
    raw_posts = fetch_live_reddit_rss(limit_per_query=12) if include_live else []

    seen_urls = set()
    leads: List[IntentLead] = []
    now_iso = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    for post in raw_posts:
        url = post.get("url", "")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)

        title = post.get("title", "").strip()
        body = post.get("body", "").strip()
        score, triggers = score_post(title, body)
        if score < min_score:
            continue

        combined = f"{title} {body}"
        community = post.get("community", "")
        author = post.get("author", "")
        trade = detect_trade(combined, community)
        crm = detect_crm(combined)
        snippet = (body[:260] + "...") if len(body) > 260 else body

        leads.append(
            IntentLead(
                source=post.get("source", "Reddit (Live)"),
                community=community,
                author=author,
                title=title,
                snippet=snippet,
                url=url,
                intent_score=score,
                matched_triggers=triggers,
                detected_trade=trade,
                detected_crm=crm,
                suggested_reply=build_suggested_reply(trade, crm, author),
                captured_at=now_iso,
                post_date=post.get("post_date", "Recent"),
            )
        )

    leads.sort(key=lambda item: (item.intent_score, item.post_date), reverse=True)
    return leads[:25]


def export_leads(leads: List[IntentLead], out_dir: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "intent_leads.json"
    csv_path = out_dir / "intent_leads.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(lead) for lead in leads], f, indent=2)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "intent_score",
                "post_date",
                "source",
                "community",
                "author",
                "detected_trade",
                "detected_crm",
                "title",
                "matched_triggers",
                "suggested_reply",
                "url",
                "captured_at",
            ],
        )
        writer.writeheader()
        for lead in leads:
            row = asdict(lead)
            row["matched_triggers"] = ", ".join(lead.matched_triggers)
            row.pop("snippet", None)
            writer.writerow(row)

    return json_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Nexus AI Live Reddit Intent Monitor")
    parser.add_argument("--min-score", type=int, default=35, help="Minimum buying-intent score (0-100)")
    args = parser.parse_args()

    out_dir = pathlib.Path(__file__).resolve().parent
    leads = run_intent_monitor(include_live=True, min_score=args.min_score)
    json_path, csv_path = export_leads(leads, out_dir)

    print("=" * 88)
    print(f"NEXUS AI LIVE INTENT MONITOR — {len(leads)} REAL REDDIT LEADS CAPTURED")
    print("=" * 88)
    for idx, lead in enumerate(leads[:10], start=1):
        print(
            f"\n[{idx}] SCORE: {lead.intent_score}/100 | {lead.community} | {lead.author} ({lead.post_date})"
        )
        print(f"    Title   : {lead.title}")
        print(f"    URL     : {lead.url}")
        print(f"    Triggers: {', '.join(lead.matched_triggers)}")
    print("\n" + "-" * 88)
    print(f"Exported JSON: {json_path}")
    print(f"Exported CSV : {csv_path}")


if __name__ == "__main__":
    main()
