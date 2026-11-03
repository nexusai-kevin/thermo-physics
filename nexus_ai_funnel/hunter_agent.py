#!/usr/bin/env python3
"""
hunter_agent.py - Autonomous Prospecting & Audit Node for Nexus AI Solutions
Extracts contractor business listings, detects missing lead infrastructure,
and crafts contextual diagnostic hooks to the interactive sandbox.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


@dataclass
class ContractorLead:
    company_name: str
    trade: str
    phone: str
    website: str
    borough: str
    claims_24_7: bool
    negative_review_snippet: Optional[str]
    audit_risk: str
    personalized_hook: str


class ContractorHunter:
    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "llama3.1",
        funnel_url: str = "https://getnexusaisolutions.com/funnel#sandbox",
        demo_phone: str = "(212) 555-0199",
    ) -> None:
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.funnel_url = funnel_url
        self.demo_phone = demo_phone
        self._ollama_available: Optional[bool] = None

    def _call_ollama(self, prompt: str) -> Optional[str]:
        """Attempts local Ollama inference; caches availability so offline runs are instant."""
        if self._ollama_available is False:
            return None
        payload = json.dumps(
            {"model": self.model, "prompt": prompt, "stream": False}
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.ollama_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=2) as resp:
                self._ollama_available = True
                data = json.loads(resp.read().decode("utf-8", errors="ignore"))
                text = data.get("response", "").strip()
                return text if text else None
        except Exception:
            self._ollama_available = False
            return None

    def _build_precision_fallback_hook(
        self,
        name: str,
        trade: str,
        borough: str,
        review_snippet: str,
        claims_24_7: bool,
    ) -> str:
        """Crafts a review-specific <75-word diagnostic hook when Ollama is offline."""
        claim_clause = (
            f"Even though your site lists 24/7 {trade.lower()} coverage in {borough}, a recent caller noted: \"{review_snippet}\""
            if claims_24_7 and review_snippet
            else f"Noticed you handle {trade.lower()} across {borough}, where a recent review flagged: \"{review_snippet}\""
            if review_snippet
            else f"Hey {name} team—saw you cover emergency {trade.lower()} across {borough}."
        )
        return (
            f"{claim_clause} In NYC, when after-hours emergency calls hit voicemail, 78% of homeowners "
            f"dial the next Google Maps dispatcher within 45 seconds—costing $1,500+ per missed ticket. "
            f"Test our live 8-second AI dispatcher at {self.funnel_url} or call & hang up on {self.demo_phone}."
        )

    def audit_contractor_profile(
        self,
        name: str,
        trade: str,
        borough: str,
        review_snippet: str,
        claims_24_7: bool,
    ) -> dict:
        """Uses a local LLM (or precision fallback) to extract the vulnerability and draft an undeniable hook."""
        prompt = f"""
You are an operational auditor for Nexus AI Solutions in NYC.
Contractor: {name}
Trade: {trade} ({borough}, NY)
Claims 24/7 Service: {claims_24_7}
Recent Review/Pain Snippet: "{review_snippet}"

Write a concise, high-converting 3-sentence diagnostic note to the owner:
1. Mention their trade in {borough} and call out the risk of missed after-hours calls going directly to competing dispatchers.
2. If they claim 24/7 or have review complaints, cite how a voicemail delay loses high-ticket jobs.
3. Invite them to test our live 8-second interactive phone simulator at {self.funnel_url} or text our NYC test line ({self.demo_phone}).
Keep it strictly under 75 words, direct, and zero fluff.
"""
        hook = self._call_ollama(prompt)
        if not hook:
            hook = self._build_precision_fallback_hook(
                name=name,
                trade=trade,
                borough=borough,
                review_snippet=review_snippet,
                claims_24_7=claims_24_7,
            )

        if claims_24_7 and review_snippet:
            risk = "Critical: 24/7 Promise vs. Voicemail Abandonment"
        elif claims_24_7:
            risk = "High Voicemail Abandonment"
        else:
            risk = "Standard After-Hours Leakage"

        return {"audit_risk": risk, "hook": hook}

    def process_and_export(
        self,
        raw_prospects: List[dict],
        output_csv: Optional[str] = None,
        push_webhook_url: Optional[str] = None,
    ) -> List[ContractorLead]:
        """Audits all raw leads, exports CSV/JSON, and optionally pushes to webhook_server.py."""
        base_dir = pathlib.Path(__file__).resolve().parent
        csv_path = (
            pathlib.Path(output_csv)
            if output_csv
            else base_dir / "audited_hunter_leads.csv"
        )
        json_path = csv_path.with_suffix(".json")

        audited_leads: List[ContractorLead] = []
        for p in raw_prospects:
            audit = self.audit_contractor_profile(
                p["name"],
                p["trade"],
                p["borough"],
                p.get("review", ""),
                p.get("claims_24_7", False),
            )
            lead = ContractorLead(
                company_name=p["name"],
                trade=p["trade"],
                phone=p["phone"],
                website=p.get("website", ""),
                borough=p["borough"],
                claims_24_7=p.get("claims_24_7", False),
                negative_review_snippet=p.get("review"),
                audit_risk=audit["audit_risk"],
                personalized_hook=audit["hook"],
            )
            audited_leads.append(lead)

            if push_webhook_url:
                self._push_lead_to_webhook(lead, push_webhook_url)

        if audited_leads:
            keys = list(asdict(audited_leads[0]).keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for item in audited_leads:
                    writer.writerow(asdict(item))

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump([asdict(item) for item in audited_leads], f, indent=2)

        print(f"[+] Successfully audited {len(audited_leads)} contractors -> {csv_path}")
        print(f"[+] JSON feed exported -> {json_path}")
        return audited_leads

    @staticmethod
    def _push_lead_to_webhook(lead: ContractorLead, webhook_url: str) -> bool:
        payload = json.dumps(asdict(lead)).encode("utf-8")
        req = urllib.request.Request(
            webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False


DEFAULT_NYC_TARGETS = [
    {
        "name": "Bronx Master Emergency Plumbing",
        "trade": "Plumbing & Drain",
        "borough": "The Bronx",
        "phone": "718-555-0144",
        "website": "bronxmasterplumbing.com",
        "claims_24_7": True,
        "review": "Called at 10 PM for a sewer backup, nobody answered and got voicemail.",
    },
    {
        "name": "Kings County Heat & Cool",
        "trade": "HVAC",
        "borough": "Brooklyn",
        "phone": "718-555-0182",
        "website": "kingscountyhvac.com",
        "claims_24_7": True,
        "review": "Furnace failed during freeze, waited 4 hours for a call back.",
    },
    {
        "name": "Astoria Rapid Rooters & Mechanical",
        "trade": "Plumbing",
        "borough": "Queens",
        "phone": "718-555-0193",
        "website": "astoriarapidrooters.com",
        "claims_24_7": True,
        "review": "Website says 24hr emergency plumber but phone rang 8 times and disconnected.",
    },
    {
        "name": "Midtown Commercial Electric Corp",
        "trade": "Electrical",
        "borough": "Manhattan",
        "phone": "212-555-0167",
        "website": "midtownelectricnyc.com",
        "claims_24_7": False,
        "review": "Left two voicemails about a tripped main panel in our restaurant, ended up calling someone else.",
    },
    {
        "name": "Verrazzano Storm Roofing & Siding",
        "trade": "Roofing",
        "borough": "Staten Island",
        "phone": "718-555-0129",
        "website": "verrazzanoroofing.com",
        "claims_24_7": True,
        "review": "Active roof leak during Sunday storm, answering service took a message and nobody ever texted back.",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nexus AI Autonomous Hunter — NYC Contractor Vulnerability Auditor"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Path to output CSV file (defaults to nexus_ai_funnel/audited_hunter_leads.csv)",
    )
    parser.add_argument(
        "--webhook",
        type=str,
        default="http://127.0.0.1:8000/api/hunter-audit",
        help="Optional local webhook server URL to push audited leads into",
    )
    args = parser.parse_args()

    hunter = ContractorHunter()
    leads = hunter.process_and_export(
        DEFAULT_NYC_TARGETS,
        output_csv=args.output_csv,
        push_webhook_url=args.webhook,
    )

    print("\n" + "=" * 88)
    print("HUNTER AGENT — AUDITED NYC CONTRACTOR OUTBOUND HOOKS")
    print("=" * 88)
    for idx, lead in enumerate(leads, start=1):
        print(
            f"\n[{idx}] {lead.company_name} ({lead.trade} • {lead.borough}) | Phone: {lead.phone}"
        )
        print(f"    24/7 Claimed : {lead.claims_24_7} | Audit Risk: {lead.audit_risk}")
        print(f"    Review Signal: \"{lead.negative_review_snippet}\"")
        print(f"    Outbound Hook: {lead.personalized_hook}")
    print("\n" + "=" * 88)


if __name__ == "__main__":
    main()
