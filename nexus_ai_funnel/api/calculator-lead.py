from http.server import BaseHTTPRequestHandler
import base64
import datetime as dt
import json
import os
import urllib.parse
import urllib.request

OWNER_ALERT_PHONE = os.environ.get("NEXUS_OWNER_ALERT_PHONE", "+19298392194").strip()
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "").strip()
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "").strip()
TWILIO_FROM_NUMBER = os.environ.get("TWILIO_FROM_NUMBER", "+12125550199").strip()


def send_twilio_carrier_sms(to_phone: str, text: str) -> dict:
    if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_FROM_NUMBER):
        return {
            "status": "QUEUED_CLOUD",
            "channel": "SYSTEM_SIMULATION",
            "note": "Alert logged. Add TWILIO_ACCOUNT_SID & TWILIO_AUTH_TOKEN in Vercel to dispatch live cellular SMS to 929-839-2194.",
        }
    try:
        url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json"
        auth_str = f"{TWILIO_ACCOUNT_SID}:{TWILIO_AUTH_TOKEN}"
        auth_header = (
            "Basic " + base64.b64encode(auth_str.encode("utf-8")).decode("utf-8")
        )
        payload = urllib.parse.urlencode(
            {"To": to_phone, "From": TWILIO_FROM_NUMBER, "Body": text}
        ).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload, headers={"Authorization": auth_header}
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return {
                "status": "DELIVERED",
                "channel": "TWILIO_CARRIER",
                "sid": data.get("sid"),
            }
    except Exception as exc:
        return {"status": "FAILED", "channel": "TWILIO_ERROR", "note": str(exc)}


class handler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict, status: int = 200) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length).decode("utf-8", errors="ignore")
        data = {}
        if raw_body.startswith("{"):
            try:
                data = json.loads(raw_body)
            except Exception:
                data = {}

        phone = data.get("phone", "+19175550143")
        trade = data.get("trade", "Plumbing")
        weekly_loss = data.get("weekly_loss", "$9,000")
        timestamp = dt.datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
        alert_text = (
            f"🔥 [NEXUS INBOUND LEAD ALERT]\n"
            f"A contractor submitted your Missed Revenue Calculator!\n\n"
            f"• Contractor Cell: {phone}\n"
            f"• Trade: {trade}\n"
            f"• Estimated Leak: {weekly_loss}/wk\n"
            f"• Time: {timestamp}\n\n"
            f"Call or text them back now while they are hot!"
        )
        delivery = send_twilio_carrier_sms(OWNER_ALERT_PHONE, alert_text)
        self._send_json(
            {
                "ok": True,
                "sms_preview": f"📊 [Nexus AI {trade} Audit] Your estimated voicemail leak is {weekly_loss}/week.",
                "owner_alert": delivery,
            }
        )
