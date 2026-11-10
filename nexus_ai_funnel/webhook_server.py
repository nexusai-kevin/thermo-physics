#!/usr/bin/env python3
"""
Nexus AI Solutions — Unified Server, 8-Second SMS Qualifier & Visual Command Center API
=======================================================================================
Serves:
  - GET  /dashboard          -> Visual Command Center (1-click script runner & live pipeline view)
  - GET  /                   -> Contractor Capture Funnel (`index.html`)
  - GET  /health             -> Health & tunnel status check
  - GET  /api/state          -> Live state (Hunter leads, Intent leads, SMS sessions, logs)
  - POST /api/run-hunter     -> One-button execution of `hunter_agent.py`
  - POST /api/run-intent     -> One-button execution of `intent_monitor.py`
  - POST /api/run-simulation -> One-button 8-sec missed-call & 2-question SMS simulation
  - POST /api/audit-single   -> Live custom contractor audit via `ContractorHunter`
  - POST /api/calculator-lead-> Receives calculator leads from `index.html`
  - POST /twilio/voice-missed-> Twilio incoming missed call -> 8-second SMS recovery
  - POST /twilio/sms-reply   -> Twilio 2-question AI SMS qualifier & calendar booking
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import io
import json
import os
import pathlib
import sys
import urllib.parse
import urllib.request
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = pathlib.Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Lightweight environment loader
def load_simple_env(env_path: pathlib.Path) -> None:
    if not env_path.is_file():
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass

load_simple_env(BASE_DIR / ".env")
load_simple_env(BASE_DIR.parent / ".env")

import hunter_agent  # noqa: E402
import intent_monitor  # noqa: E402

OWNER_ALERT_PHONE = os.environ.get("NEXUS_OWNER_ALERT_PHONE", "+19298392194").strip()
CALENDLY_BOOKING_URL = os.environ.get(
    "NEXUS_BOOKING_URL", "https://cal.com/nexus-ai-nyc/15min-activation"
)
NYC_DEMO_NUMBER = os.environ.get("NEXUS_DEMO_NUMBER", "+12125550199")

TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "").strip()
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "").strip()
TWILIO_FROM_NUMBER = os.environ.get("TWILIO_FROM_NUMBER", NYC_DEMO_NUMBER).strip()


def send_twilio_sms(to_number: str, text: str) -> dict:
    """Attempts real carrier SMS via Twilio REST API if configured."""
    sid = os.environ.get("TWILIO_ACCOUNT_SID", "").strip()
    token = os.environ.get("TWILIO_AUTH_TOKEN", "").strip()
    from_num = os.environ.get("TWILIO_FROM_NUMBER", NYC_DEMO_NUMBER).strip()

    if not (sid and token and from_num):
        return {
            "delivered": False,
            "channel": "SYSTEM_SIMULATION",
            "status": "QUEUED_LOCAL",
            "note": "Alert saved & broadcasted. Add TWILIO_ACCOUNT_SID & TWILIO_AUTH_TOKEN in .env for live cellular SMS.",
        }

    try:
        url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
        auth_str = f"{sid}:{token}"
        auth_header = (
            "Basic " + base64.b64encode(auth_str.encode("utf-8")).decode("utf-8")
        )
        payload = urllib.parse.urlencode(
            {"To": to_number, "From": from_num, "Body": text}
        ).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload, headers={"Authorization": auth_header}
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return {
                "delivered": True,
                "channel": "TWILIO_CARRIER",
                "status": "DELIVERED",
                "sid": data.get("sid"),
            }
    except Exception as exc:
        return {
            "delivered": False,
            "channel": "TWILIO_ERROR",
            "status": "FAILED",
            "note": str(exc),
        }


def now_ts() -> str:
    return dt.datetime.now().strftime("%H:%M:%S")


@dataclass
class LeadSession:
    phone: str
    stage: str = "AWAITING_MISSED_CALLS"
    trade: str = "Trade Contractor"
    missed_calls_weekly: int = 0
    crm: str = ""
    history: list[dict] = field(default_factory=list)


class NexusQualifierEngine:
    """State machine that qualifies inbound contractor leads in 2 operational questions."""

    def __init__(self, booking_url: str = CALENDLY_BOOKING_URL) -> None:
        self.booking_url = booking_url
        self.sessions: Dict[str, LeadSession] = {}
        self.audited_prospects: Dict[str, dict] = {}
        self.owner_alerts: List[dict] = []
        self.load_persisted_alerts()
        self.activity_log: List[dict] = [
            {
                "time": now_ts(),
                "type": "SYSTEM",
                "message": f"Nexus AI Unified Funnel initialized. Text alerts linked to {OWNER_ALERT_PHONE}.",
            }
        ]

    def load_persisted_alerts(self) -> None:
        alerts_file = BASE_DIR / "hot_leads_alerts.json"
        if alerts_file.is_file():
            try:
                with open(alerts_file, "r", encoding="utf-8") as f:
                    self.owner_alerts = json.load(f)
            except Exception:
                self.owner_alerts = []

    def save_persisted_alerts(self) -> None:
        alerts_file = BASE_DIR / "hot_leads_alerts.json"
        try:
            with open(alerts_file, "w", encoding="utf-8") as f:
                json.dump(self.owner_alerts[:50], f, indent=2)
        except Exception:
            pass

    def dispatch_owner_alert(
        self, trigger_event: str, lead_phone: str, lead_summary: str
    ) -> dict:
        timestamp = dt.datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
        sms_text = (
            f"🔥 [NEXUS HOT LEAD ALERT]\n"
            f"Someone is interested in your AI reception service!\n\n"
            f"• Lead: {lead_phone}\n"
            f"• Event: {trigger_event}\n"
            f"• Details: {lead_summary}\n"
            f"• Time: {timestamp}\n\n"
            f"Call or text them back immediately to close the deal!"
        )

        delivery = send_twilio_sms(OWNER_ALERT_PHONE, sms_text)
        alert_record = {
            "id": f"alert-{len(self.owner_alerts) + 1}",
            "time": now_ts(),
            "timestamp": timestamp,
            "target_phone": OWNER_ALERT_PHONE,
            "event": trigger_event,
            "lead_phone": lead_phone,
            "details": lead_summary,
            "sms_body": sms_text,
            "status": delivery.get("status", "QUEUED"),
            "channel": delivery.get("channel", "SYSTEM"),
            "note": delivery.get("note", ""),
        }
        self.owner_alerts.insert(0, alert_record)
        self.save_persisted_alerts()
        self.log_event(
            "OWNER SMS ALERT",
            f"Dispatched text alert to {OWNER_ALERT_PHONE}: [{trigger_event}] {lead_phone} ({lead_summary})",
        )
        return alert_record

    def log_event(self, event_type: str, message: str) -> None:
        self.activity_log.insert(
            0, {"time": now_ts(), "type": event_type, "message": message}
        )
        del self.activity_log[50:]

    def register_hunter_prospect(self, prospect: dict) -> None:
        phone = prospect.get("phone", "")
        if phone:
            self.audited_prospects[phone] = prospect

    def trigger_missed_call_demo(
        self, caller_phone: str, trade: str = "Trade Contractor"
    ) -> str:
        session = LeadSession(
            phone=caller_phone, stage="AWAITING_MISSED_CALLS", trade=trade
        )
        self.sessions[caller_phone] = session
        msg = (
            "⚡ [Nexus AI 8-Sec Demo] You just called our NYC Demo Line and hung up! "
            "This is the exact instant text your emergency plumbing/HVAC/electrical callers get "
            "instead of hitting voicemail.\n\n"
            "Want to see what this would recover for your shop? Reply with roughly how many "
            "calls you miss per week (e.g. '6')."
        )
        session.history.append({"role": "ai", "text": msg, "time": now_ts()})
        self.log_event(
            "8-SEC DEMO",
            f"Missed call from {caller_phone} ({trade}) -> Fired 8-second SMS recovery.",
        )
        self.dispatch_owner_alert(
            "DEMO_CALL_TRIGGERED",
            caller_phone,
            f"Missed call from {trade} contractor. Fired 8-sec SMS demo.",
        )
        return msg

    def trigger_calculator_audit(
        self, phone: str, trade: str, weekly_loss: str
    ) -> str:
        session = LeadSession(
            phone=phone, stage="AWAITING_CRM", trade=trade, missed_calls_weekly=6
        )
        self.sessions[phone] = session
        msg = (
            f"📊 [Nexus AI {trade} Audit] Your estimated voicemail leak is {weekly_loss}/week.\n\n"
            f"Quick question so we can show you the exact setup: What CRM or dispatch calendar "
            f"do you currently use? (ServiceTitan, Housecall Pro, Jobber, Google Calendar, etc.)"
        )
        session.history.append({"role": "ai", "text": msg, "time": now_ts()})
        self.log_event(
            "CALC LEAD",
            f"Calculator audit requested by {phone} ({trade} | {weekly_loss}/wk).",
        )
        self.dispatch_owner_alert(
            "CALCULATOR_SUBMITTED",
            phone,
            f"Trade: {trade} | Estimated Loss: {weekly_loss}/wk",
        )
        return msg

    def handle_inbound_sms(self, from_phone: str, body: str) -> str:
        clean = body.strip()
        session = self.sessions.get(from_phone)

        if not session:
            session = LeadSession(phone=from_phone, stage="AWAITING_MISSED_CALLS")
            self.sessions[from_phone] = session
            session.history.append({"role": "user", "text": clean, "time": now_ts()})
            reply = (
                "⚡ Hey! This is the Nexus AI 24/7 Receptionist for NYC contractors. "
                "I qualify leads and book jobs in 8 seconds so you never lose a $1,500+ ticket to voicemail.\n\n"
                "Q1 (of 2): Roughly how many calls do you miss per week while on a job or after-hours?"
            )
            session.history.append({"role": "ai", "text": reply, "time": now_ts()})
            self.log_event("SMS IN", f"New inbound SMS from {from_phone}: '{clean}'")
            return reply

        session.history.append({"role": "user", "text": clean, "time": now_ts()})

        if session.stage == "AWAITING_MISSED_CALLS":
            digits = [int(s) for s in clean.replace("+", " ").split() if s.isdigit()]
            missed = digits[0] if digits else 5
            session.missed_calls_weekly = missed
            session.stage = "AWAITING_CRM"
            est_weekly = missed * 1500
            reply = (
                f"Got it—{missed} missed calls/wk at a $1,500 avg emergency ticket is ~${est_weekly:,}/wk "
                f"in potential jobs calling the next listing on Google Maps.\n\n"
                f"Q2 (of 2): What CRM or calendar do you use to schedule jobs? "
                f"(e.g. ServiceTitan, Housecall Pro, Jobber, Google Calendar)"
            )
            session.history.append({"role": "ai", "text": reply, "time": now_ts()})
            self.log_event(
                "QUALIFIED Q1",
                f"{from_phone} misses {missed} calls/wk (~${est_weekly:,}/wk leak). Asked Q2 (CRM).",
            )
            self.dispatch_owner_alert(
                "QUALIFIED_CALL_VOLUME",
                from_phone,
                f"Misses {missed} calls/wk (~${est_weekly:,}/wk leak). Moving to CRM stage.",
            )
            return reply

        if session.stage == "AWAITING_CRM":
            session.crm = clean
            session.stage = "BOOKED"
            est_monthly = max(session.missed_calls_weekly, 4) * 1500 * 4
            reply = (
                f"Perfect—we integrate directly with {session.crm} so qualified emergency jobs land "
                f"straight on your dispatch board (recovering ~${est_monthly:,}/mo).\n\n"
                f"📅 Grab a 15-min slot here to activate your dedicated NYC number:\n"
                f"{self.booking_url}\n\n"
                f"(Or reply 'MORNING' or 'AFTERNOON' and I'll text you 2 open times right here.)"
            )
            session.history.append({"role": "ai", "text": reply, "time": now_ts()})
            self.log_event(
                "QUALIFIED Q2",
                f"{from_phone} uses {session.crm}. Sent direct booking link ({self.booking_url}).",
            )
            self.dispatch_owner_alert(
                "QUALIFIED_CRM_PROVIDED",
                from_phone,
                f"Uses CRM: {session.crm} | Potential recovery: ~${est_monthly:,}/mo | Sent booking link",
            )
            return reply

        low = clean.lower()
        if "morning" in low or "afternoon" in low or "tomorrow" in low:
            slot = (
                "Tomorrow at 8:00 AM EST"
                if "morning" in low
                else "Tomorrow at 3:30 PM EST"
            )
            reply = (
                f"✅ Locked in for {slot}! We've sent a calendar invite and will have your "
                f"{session.crm or 'dispatch'} sandbox ready to test live."
            )
            session.history.append({"role": "ai", "text": reply, "time": now_ts()})
            self.log_event(
                "CALENDAR LOCKED",
                f"Booked {from_phone} ({session.crm}) for {slot}!",
            )
            self.dispatch_owner_alert(
                "CALENDAR_SLOT_LOCKED",
                from_phone,
                f"Locked in for {slot} | CRM: {session.crm}",
            )
            return reply

        reply = (
            f"You're all set! Use your direct booking link anytime to pick a 15-min activation window: "
            f"{self.booking_url}"
        )
        session.history.append({"role": "ai", "text": reply, "time": now_ts()})
        return reply


ENGINE = NexusQualifierEngine()


def load_json_file(filename: str) -> list:
    path = BASE_DIR / filename
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


class WebhookHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        # Keep console output clean
        pass

    def _send_json(self, payload: dict, status: int = 200) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(raw)

    def _send_html_file(self, filename: str) -> None:
        path = BASE_DIR / filename
        if not path.exists():
            self._send_json({"error": f"{filename} not found"}, status=404)
            return
        raw = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        route = parsed.path

        if route in ("/dashboard", "/command-center"):
            self._send_html_file("command_center.html")
            return

        if route in ("/", "/funnel", "/index.html"):
            self._send_html_file("index.html")
            return

        if route == "/health":
            self._send_json(
                {
                    "status": "online",
                    "service": "Nexus AI Solutions Unified Funnel & Command Center",
                    "active_sms_sessions": len(ENGINE.sessions),
                    "hunter_audited_prospects": len(
                        load_json_file("audited_hunter_leads.json")
                    ),
                    "intent_community_leads": len(load_json_file("intent_leads.json")),
                    "endpoints": [
                        "/dashboard",
                        "/funnel",
                        "/api/state",
                        "/api/run-hunter",
                        "/api/run-intent",
                        "/api/run-simulation",
                        "/twilio/voice-missed",
                        "/twilio/sms-reply",
                    ],
                }
            )
            return

        if route == "/api/state":
            hunter_leads = load_json_file("audited_hunter_leads.json")
            intent_leads = load_json_file("intent_leads.json")
            sessions_list = [asdict(s) for s in ENGINE.sessions.values()]
            self._send_json(
                {
                    "status": "online",
                    "owner_phone": OWNER_ALERT_PHONE,
                    "demo_number": NYC_DEMO_NUMBER,
                    "booking_url": CALENDLY_BOOKING_URL,
                    "hunter_leads": hunter_leads,
                    "intent_leads": intent_leads,
                    "sms_sessions": sessions_list,
                    "owner_alerts": ENGINE.owner_alerts,
                    "activity_log": ENGINE.activity_log,
                }
            )
            return

        self._send_json({"error": "Not found"}, status=404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length).decode("utf-8", errors="ignore")
        data = {}
        if raw_body.startswith("{"):
            try:
                data = json.loads(raw_body)
            except Exception:
                data = {}

        if self.path == "/api/run-hunter":
            buf = io.StringIO()
            with redirect_stdout(buf):
                leads = hunter_agent.discover_and_audit_real_nyc_contractors()
            for item in leads:
                ENGINE.register_hunter_prospect(asdict(item))
            ENGINE.log_event(
                "HUNTER AGENT",
                f"Live-audited {len(leads)} real NYC contractors -> audited_hunter_leads.csv",
            )
            self._send_json(
                {
                    "ok": True,
                    "count": len(leads),
                    "leads": [asdict(item) for item in leads],
                    "console_output": buf.getvalue(),
                }
            )
            return

        if self.path == "/api/audit-single":
            name = data.get("name", "Metro NYC Plumbing")
            trade = data.get("trade", "Plumbing")
            borough = data.get("borough", "Queens")
            phone = data.get("phone", "(718) 555-0199")
            website = data.get("website", "")
            claims_24_7 = bool(data.get("claims_24_7", True))
            review = data.get("review", "")
            hunter = hunter_agent.ContractorHunter()
            audit = hunter.audit_contractor_profile(
                name=name,
                trade=trade,
                borough=borough,
                website=website,
                review_snippet=review,
                claims_24_7=claims_24_7,
            )
            new_lead = hunter_agent.ContractorLead(
                company_name=name,
                trade=trade,
                phone=phone,
                website=website,
                borough=borough,
                claims_24_7=audit["claims_24_7"],
                negative_review_snippet=audit["finding"],
                audit_risk=audit["audit_risk"],
                personalized_hook=audit["hook"],
            )
            existing = load_json_file("audited_hunter_leads.json")
            existing.insert(0, asdict(new_lead))
            (BASE_DIR / "audited_hunter_leads.json").write_text(
                json.dumps(existing, indent=2), encoding="utf-8"
            )
            ENGINE.register_hunter_prospect(asdict(new_lead))
            ENGINE.log_event(
                "HUNTER CUSTOM",
                f"Live-audited {name} ({website or borough}) -> {audit['audit_risk']}",
            )
            self._send_json({"ok": True, "lead": asdict(new_lead), "all_leads": existing})
            return

        if self.path == "/api/run-intent":
            include_live = bool(data.get("include_live", True))
            leads = intent_monitor.run_intent_monitor(
                include_live=include_live, min_score=35
            )
            intent_monitor.export_leads(leads, BASE_DIR)
            ENGINE.log_event(
                "INTENT MONITOR",
                f"Scraped live Reddit RSS feeds -> Captured {len(leads)} real threads.",
            )
            self._send_json(
                {
                    "ok": True,
                    "count": len(leads),
                    "leads": [asdict(item) for item in leads],
                }
            )
            return

        if self.path == "/api/run-simulation":
            phone = data.get("phone", "+1 (917) 555-0188")
            trade = data.get("trade", "Plumbing")
            missed_reply = data.get(
                "missed_reply", "We miss around 6 calls a week when both trucks are on jobs"
            )
            crm_reply = data.get("crm_reply", "Housecall Pro")
            slot_reply = data.get("slot_reply", "MORNING works best")

            s1 = ENGINE.trigger_missed_call_demo(phone, trade=trade)
            s2 = ENGINE.handle_inbound_sms(phone, missed_reply)
            s3 = ENGINE.handle_inbound_sms(phone, crm_reply)
            s4 = ENGINE.handle_inbound_sms(phone, slot_reply)

            session = ENGINE.sessions[phone]
            self._send_json(
                {
                    "ok": True,
                    "phone": phone,
                    "session": asdict(session),
                    "steps": [s1, s2, s3, s4],
                }
            )
            return

        if self.path == "/api/sms-chat":
            phone = data.get("phone", "+1 (917) 555-0199")
            message = data.get("message", "")
            action = data.get("action", "reply")
            if action == "missed_call":
                reply = ENGINE.trigger_missed_call_demo(
                    phone, trade=data.get("trade", "Plumbing")
                )
            else:
                reply = ENGINE.handle_inbound_sms(phone, message)
            session = ENGINE.sessions.get(phone)
            self._send_json(
                {
                    "ok": True,
                    "reply": reply,
                    "session": asdict(session) if session else None,
                }
            )
            return

        if self.path == "/api/hunter-audit":
            ENGINE.register_hunter_prospect(data)
            ENGINE.log_event(
                "HUNTER INGEST",
                f"Registered {data.get('company_name')} ({data.get('borough')})",
            )
            self._send_json({"ok": True, "registered": data.get("company_name")})
            return

        if self.path == "/api/calculator-lead":
            phone = data.get("phone", "+19175550143")
            trade = data.get("trade", "Plumbing")
            weekly_loss = data.get("weekly_loss", "$9,000")
            sms_text = ENGINE.trigger_calculator_audit(phone, trade, weekly_loss)
            self._send_json({"ok": True, "sms_preview": sms_text})
            return

        if self.path == "/api/book-slot":
            name = data.get("name", "Interested Contractor").strip()
            contact = data.get("contact", "").strip()
            slot = data.get("slot", "Tomorrow @ 8:00 AM EST").strip()
            calls = data.get("calls", "4–8 calls/wk").strip()
            crm = data.get("crm", "ServiceTitan").strip()

            ENGINE.log_event(
                "BOOKING CONFIRMED",
                f"{name} ({contact}) booked {slot} | CRM: {crm} | Volume: {calls}",
            )
            alert = ENGINE.dispatch_owner_alert(
                "WEB_CALENDAR_BOOKING",
                contact,
                f"Name: {name} | Slot: {slot} | CRM: {crm} | Volume: {calls}",
            )
            self._send_json({"ok": True, "alert": alert})
            return

        if self.path == "/api/test-owner-alert":
            test_phone = data.get("test_lead_phone", "+1 (212) 555-0144")
            test_trade = data.get("trade", "HVAC")
            alert = ENGINE.dispatch_owner_alert(
                "TEST_HOT_LEAD",
                test_phone,
                f"{test_trade} Contractor requested immediate 8-sec emergency demo ($12,000/wk leak)",
            )
            self._send_json({"ok": True, "alert": alert})
            return

        if self.path == "/twilio/voice-missed":
            params = urllib.parse.parse_qs(raw_body)
            caller = params.get("From", ["+19175550199"])[0]
            sms_text = ENGINE.trigger_missed_call_demo(caller)
            self._send_json({"ok": True, "caller": caller, "sms_sent": sms_text})
            return

        if self.path == "/twilio/sms-reply":
            params = urllib.parse.parse_qs(raw_body)
            sender = params.get("From", ["+19175550199"])[0]
            body = params.get("Body", [""])[0]
            reply_text = ENGINE.handle_inbound_sms(sender, body)
            twiml = f'<?xml version="1.0" encoding="UTF-8"?><Response><Message>{reply_text}</Message></Response>'
            self.send_response(200)
            self.send_header("Content-Type", "application/xml")
            self.end_headers()
            self.wfile.write(twiml.encode("utf-8"))
            return

        self._send_json({"error": "Not found"}, status=404)


def run_simulation() -> None:
    print("=" * 84)
    print("NEXUS AI — 8-SECOND DEMO & 2-QUESTION SMS QUALIFIER SIMULATION")
    print("=" * 84)
    contractor_phone = "+19175550188"

    print(f"\n1. [CONTRACTOR CALLS {NYC_DEMO_NUMBER} & HANGS UP]")
    opening = ENGINE.trigger_missed_call_demo(contractor_phone, trade="Plumbing")
    print(f"   AI (+8.0s SMS): {opening}\n")

    reply_1 = "We probably miss 6 or 7 calls a week when both trucks are out"
    print(f'2. [CONTRACTOR REPLIES]: "{reply_1}"')
    ai_q2 = ENGINE.handle_inbound_sms(contractor_phone, reply_1)
    print(f"   AI: {ai_q2}\n")

    reply_2 = "Housecall Pro"
    print(f'3. [CONTRACTOR REPLIES]: "{reply_2}"')
    ai_book = ENGINE.handle_inbound_sms(contractor_phone, reply_2)
    print(f"   AI: {ai_book}\n")

    reply_3 = "MORNING works best"
    print(f'4. [CONTRACTOR REPLIES]: "{reply_3}"')
    ai_confirm = ENGINE.handle_inbound_sms(contractor_phone, reply_3)
    print(f"   AI: {ai_confirm}\n")
    print("=" * 84)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nexus AI Unified Webhook & Command Center Server"
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run end-to-end 2-question SMS simulation",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run HTTP server"
    )
    args = parser.parse_args()

    if args.simulate:
        run_simulation()
        return

    server = HTTPServer(("127.0.0.1", args.port), WebhookHandler)
    print("=" * 78)
    print(f"  NEXUS AI COMMAND CENTER : http://localhost:{args.port}/dashboard")
    print(f"  CONTRACTOR FUNNEL PAGE  : http://localhost:{args.port}/funnel")
    print(f"  TWILIO VOICE WEBHOOK    : http://localhost:{args.port}/twilio/voice-missed")
    print(f"  TWILIO SMS WEBHOOK      : http://localhost:{args.port}/twilio/sms-reply")
    print("=" * 78)
    server.serve_forever()


if __name__ == "__main__":
    main()
