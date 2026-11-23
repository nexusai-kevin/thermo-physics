"""Authenticated OpenAI-compatible reverse proxy for 48-hour prospect sandboxes."""

from __future__ import annotations

import argparse
import http.server
import json
import os
import secrets
import threading
import time
import urllib.error
import urllib.request
from typing import Any


class _AuthProxyHandler(http.server.BaseHTTPRequestHandler):
    upstream_base: str = "http://127.0.0.1:11434"
    bearer_token: str = ""
    expires_at_epoch: float = 0.0

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[SandboxProxy] {self.address_string()} - {fmt % args}", flush=True)

    def _check_auth(self) -> bool:
        if time.time() > self.expires_at_epoch:
            self._send_json(410, {"error": "Sandbox 48-hour TTL expired."})
            return False
        auth = self.headers.get("Authorization", "")
        expected = f"Bearer {self.bearer_token}"
        if not secrets.compare_digest(auth, expected):
            self._send_json(401, {"error": "Unauthorized: invalid or missing bearer token."})
            return False
        return True

    def _send_json(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _proxy(self, method: str) -> None:
        if self.path == "/health":
            self._send_json(
                200,
                {
                    "status": "ok",
                    "remaining_sec": max(0, int(self.expires_at_epoch - time.time())),
                },
            )
            return

        if not self._check_auth():
            return

        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length) if length > 0 else None
        target_url = f"{self.upstream_base.rstrip('/')}{self.path}"

        headers = {
            k: v
            for k, v in self.headers.items()
            if k.lower() not in ("host", "authorization", "content-length")
        }
        req = urllib.request.Request(target_url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                self.send_response(resp.status)
                for k, v in resp.headers.items():
                    if k.lower() not in ("transfer-encoding", "connection"):
                        self.send_header(k, v)
                self.end_headers()
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except urllib.error.HTTPError as err:
            err_body = err.read()
            self.send_response(err.code)
            self.send_header("Content-Type", err.headers.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(err_body)))
            self.end_headers()
            self.wfile.write(err_body)
        except Exception as exc:
            self._send_json(502, {"error": f"Upstream inference error: {exc}"})

    def do_GET(self) -> None:
        self._proxy("GET")

    def do_POST(self) -> None:
        self._proxy("POST")


def main() -> int:
    parser = argparse.ArgumentParser(description="Authenticated OpenAI-compatible sandbox proxy")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--upstream", default="http://127.0.0.1:11434")
    parser.add_argument("--token", required=True)
    parser.add_argument("--ttl-hours", type=float, default=48.0)
    args = parser.parse_args()

    _AuthProxyHandler.upstream_base = args.upstream
    _AuthProxyHandler.bearer_token = args.token
    _AuthProxyHandler.expires_at_epoch = time.time() + (args.ttl_hours * 3600.0)

    server = http.server.ThreadingHTTPServer(("0.0.0.0", args.port), _AuthProxyHandler)

    def _ttl_watchdog() -> None:
        time.sleep(max(1.0, args.ttl_hours * 3600.0))
        print("[SandboxProxy] TTL reached, shutting down.", flush=True)
        server.shutdown()
        os._exit(0)

    threading.Thread(target=_ttl_watchdog, daemon=True).start()
    print(f"[SandboxProxy] Listening on 0.0.0.0:{args.port} -> {args.upstream} (TTL={args.ttl_hours}h)", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
