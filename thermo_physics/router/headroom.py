"""
thermo_physics.router.headroom
==============================
Tracks token and request load per endpoint across rolling 60-second windows.
Ingests provider rate-limit response headers (OpenAI, Anthropic) to calibrate
true provider headroom against theoretical caps.
"""

from __future__ import annotations

import collections
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

from thermo_physics.router.config import EndpointConfig


def parse_retry_after(value: Optional[str]) -> Optional[float]:
    """Parses Retry-After header into delay seconds."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def parse_reset_seconds(value: Optional[str]) -> Optional[float]:
    """Parses rate limit reset strings like '12s', '250ms', '1m30s' or bare floats into seconds."""
    if not value:
        return None
    val = value.strip().lower()
    try:
        return max(0.0, float(val))
    except ValueError:
        pass

    total = 0.0
    matched = False
    for num, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(ms|s|m|h)?", val):
        matched = True
        n = float(num)
        if unit == "ms":
            total += n / 1000.0
        elif unit == "m":
            total += n * 60.0
        elif unit == "h":
            total += n * 3600.0
        else:  # 's' or empty
            total += n
    return total if matched else None


@dataclass
class HeadroomSnapshot:
    endpoint_name: str
    active_tpm: int
    active_rpm: int
    tpm_limit: int
    rpm_limit: int
    headroom_tpm: int
    headroom_rpm: int
    tpm_utilization: float
    rpm_utilization: float
    blocked_until: float


class HeadroomTracker:
    """Sliding-window tracker of active TPM and RPM across all managed endpoints."""

    def __init__(self, endpoints: List[EndpointConfig], window_sec: float = 60.0):
        self.endpoints = {ep.name: ep for ep in endpoints}
        self.window_sec = window_sec
        # deques hold tuples of (timestamp, token_count)
        self._token_events: Dict[str, collections.deque[Tuple[float, int]]] = {
            ep.name: collections.deque() for ep in endpoints
        }
        self._request_events: Dict[str, collections.deque[float]] = {
            ep.name: collections.deque() for ep in endpoints
        }
        # Ingested rate limit headers
        self._header_remaining_tokens: Dict[str, Optional[int]] = {ep.name: None for ep in endpoints}
        self._header_remaining_requests: Dict[str, Optional[int]] = {ep.name: None for ep in endpoints}
        self._blocked_until: Dict[str, float] = {ep.name: 0.0 for ep in endpoints}

    def _purge(self, ep_name: str, now: float) -> None:
        cutoff = now - self.window_sec
        toks = self._token_events[ep_name]
        while toks and toks[0][0] < cutoff:
            toks.popleft()
        reqs = self._request_events[ep_name]
        while reqs and reqs[0] < cutoff:
            reqs.popleft()

    def record_dispatched(self, ep_name: str, est_tokens: int, now: Optional[float] = None) -> None:
        """Reserve capacity at the moment of dispatch."""
        t = time.time() if now is None else now
        if ep_name in self._token_events:
            self._token_events[ep_name].append((t, est_tokens))
            self._request_events[ep_name].append(t)

    def record_completed(
        self,
        ep_name: str,
        actual_tokens: int,
        headers: Optional[Mapping[str, str]] = None,
        now: Optional[float] = None,
    ) -> None:
        """Update tracker with actual completion telemetry and provider rate-limit headers."""
        t = time.time() if now is None else now
        if ep_name not in self.endpoints:
            return

        if not headers:
            return

        norm_headers = {k.lower(): str(v) for k, v in headers.items()}

        # 1. OpenAI rate limits
        if "x-ratelimit-remaining-tokens" in norm_headers:
            try:
                self._header_remaining_tokens[ep_name] = int(norm_headers["x-ratelimit-remaining-tokens"])
            except ValueError:
                pass
        if "x-ratelimit-remaining-requests" in norm_headers:
            try:
                self._header_remaining_requests[ep_name] = int(norm_headers["x-ratelimit-remaining-requests"])
            except ValueError:
                pass

        # 2. Anthropic rate limits
        if "anthropic-ratelimit-tokens-remaining" in norm_headers:
            try:
                self._header_remaining_tokens[ep_name] = int(norm_headers["anthropic-ratelimit-tokens-remaining"])
            except ValueError:
                pass
        if "anthropic-ratelimit-requests-remaining" in norm_headers:
            try:
                self._header_remaining_requests[ep_name] = int(norm_headers["anthropic-ratelimit-requests-remaining"])
            except ValueError:
                pass

        # 3. 429 retry-after or reset limits
        delay = parse_retry_after(norm_headers.get("retry-after"))
        if delay is None:
            delay = parse_reset_seconds(norm_headers.get("x-ratelimit-reset-tokens"))
        if delay is not None and delay > 0.0:
            self._blocked_until[ep_name] = max(self._blocked_until[ep_name], t + delay)

    def record_429(self, ep_name: str, retry_after: Optional[float] = None, now: Optional[float] = None) -> None:
        """Record an explicit 429 rate limit breach."""
        t = time.time() if now is None else now
        penalty = retry_after if retry_after is not None and retry_after > 0 else 5.0
        self._blocked_until[ep_name] = max(self._blocked_until[ep_name], t + penalty)

    def get_snapshot(self, ep_name: str, now: Optional[float] = None) -> HeadroomSnapshot:
        t = time.time() if now is None else now
        ep = self.endpoints[ep_name]
        self._purge(ep_name, t)

        active_t = sum(tok for _, tok in self._token_events[ep_name])
        active_r = len(self._request_events[ep_name])

        # Blend window estimate with authoritative provider headers if recent
        headroom_t = max(0, ep.tpm_limit - active_t)
        if self._header_remaining_tokens[ep_name] is not None:
            headroom_t = min(headroom_t, self._header_remaining_tokens[ep_name])

        headroom_r = max(0, ep.rpm_limit - active_r)
        if self._header_remaining_requests[ep_name] is not None:
            headroom_r = min(headroom_r, self._header_remaining_requests[ep_name])

        tpm_util = min(1.0, active_t / max(1, ep.tpm_limit))
        rpm_util = min(1.0, active_r / max(1, ep.rpm_limit))

        return HeadroomSnapshot(
            endpoint_name=ep_name,
            active_tpm=active_t,
            active_rpm=active_r,
            tpm_limit=ep.tpm_limit,
            rpm_limit=ep.rpm_limit,
            headroom_tpm=headroom_t,
            headroom_rpm=headroom_r,
            tpm_utilization=tpm_util,
            rpm_utilization=rpm_util,
            blocked_until=self._blocked_until[ep_name],
        )

    def all_snapshots(self, now: Optional[float] = None) -> Dict[str, HeadroomSnapshot]:
        t = time.time() if now is None else now
        return {ep: self.get_snapshot(ep, now=t) for ep in self.endpoints}
