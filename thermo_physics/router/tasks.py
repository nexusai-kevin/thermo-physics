"""
thermo_physics.router.tasks
===========================
Turns an incoming chat request into routing metadata (`RouteTask`).

Privacy contract: request *content* is inspected in memory only to estimate
token counts and infer capabilities. Content is never stored or logged; only
the numeric/categorical metadata in `RouteTask` leaves this module.

Explicit client headers always override heuristics:
  x-nexus-tier            simple | medium | complex | reasoning
  x-nexus-capabilities    comma-separated, e.g. "code,tools"
  x-nexus-max-latency-ms  latency SLA for this request
  x-nexus-priority        1.0 (batch) .. 5.0 (critical interactive)
"""

from __future__ import annotations

import json
import math
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from thermo_physics.router.config import RouterConfig

CHARS_PER_TOKEN = 4.0
IMAGE_TOKEN_ESTIMATE = 800
_CODE_PATTERN = re.compile(r"```|\bdef \w+\(|\bclass \w+[:(]|\bfunction \w+\(|\bSELECT\b.+\bFROM\b|#include\s*<", re.S)


@dataclass
class RouteTask:
    request_id: str
    ts: float
    in_tokens: int
    out_tokens_est: int
    tier: str
    min_quality: int
    required_caps: List[str]
    max_latency_ms: float
    priority: float
    requested_model: Optional[str] = None
    # Filled in by replay when ground truth is known (never used for decisions):
    actual_out_tokens: Optional[int] = None

    @property
    def total_tokens_est(self) -> int:
        return self.in_tokens + self.out_tokens_est

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _content_parts(content: Any) -> List[Dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return [p if isinstance(p, dict) else {"type": "text", "text": str(p)} for p in content]
    return [{"type": "text", "text": str(content)}]


def _scan_messages(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    chars = 0
    images = 0
    has_code = False
    for msg in messages or []:
        for part in _content_parts(msg.get("content")):
            ptype = part.get("type", "text")
            if ptype in ("image_url", "image", "input_image"):
                images += 1
                continue
            text = part.get("text") or ""
            chars += len(text)
            if not has_code and _CODE_PATTERN.search(text):
                has_code = True
        # Ollama-native images field
        if msg.get("images"):
            images += len(msg["images"])
    return {"chars": chars, "images": images, "has_code": has_code, "n_messages": len(messages or [])}


def _header(headers: Mapping[str, str], key: str) -> Optional[str]:
    for k, v in headers.items():
        if k.lower() == key:
            return v
    return None


def build_task(
    *,
    messages: List[Dict[str, Any]],
    config: RouterConfig,
    headers: Optional[Mapping[str, str]] = None,
    requested_model: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    tools: Optional[List[Any]] = None,
    json_mode: bool = False,
    reasoning_requested: bool = False,
    request_id: Optional[str] = None,
    ts: Optional[float] = None,
) -> RouteTask:
    headers = headers or {}
    scan = _scan_messages(messages)

    tool_chars = len(json.dumps(tools)) if tools else 0
    in_tokens = int(math.ceil((scan["chars"] + tool_chars) / CHARS_PER_TOKEN)) + 4 * scan["n_messages"]
    in_tokens += IMAGE_TOKEN_ESTIMATE * scan["images"]

    caps: List[str] = []
    if scan["has_code"]:
        caps.append("code")
    if tools:
        caps.append("tools")
    if json_mode:
        caps.append("json")
    if scan["images"]:
        caps.append("vision")

    hdr_caps = _header(headers, "x-nexus-capabilities")
    if hdr_caps is not None:
        caps = [c.strip() for c in hdr_caps.split(",") if c.strip()]

    tier = (_header(headers, "x-nexus-tier") or "").strip().lower()
    if tier not in config.tiers:
        if reasoning_requested:
            tier = "reasoning"
        elif "code" in caps or "tools" in caps or in_tokens > 6_000:
            tier = "complex"
        elif in_tokens > 1_500 or "json" in caps:
            tier = "medium"
        else:
            tier = "simple"
    policy = config.tier(tier)

    max_latency = float(_header(headers, "x-nexus-max-latency-ms") or policy.max_latency_ms)
    priority = float(_header(headers, "x-nexus-priority") or policy.priority)
    out_est = int(max_output_tokens) if max_output_tokens else policy.default_output_tokens

    return RouteTask(
        request_id=request_id or uuid.uuid4().hex[:16],
        ts=time.time() if ts is None else ts,
        in_tokens=in_tokens,
        out_tokens_est=out_est,
        tier=tier,
        min_quality=policy.min_quality,
        required_caps=sorted(set(caps)),
        max_latency_ms=max_latency,
        priority=priority,
        requested_model=requested_model,
    )


def task_from_openai_body(body: Dict[str, Any], config: RouterConfig, headers: Mapping[str, str]) -> RouteTask:
    rf = body.get("response_format") or {}
    return build_task(
        messages=body.get("messages") or [],
        config=config,
        headers=headers,
        requested_model=body.get("model"),
        max_output_tokens=body.get("max_completion_tokens") or body.get("max_tokens"),
        tools=body.get("tools") or body.get("functions"),
        json_mode=isinstance(rf, dict) and rf.get("type") in ("json_object", "json_schema"),
        reasoning_requested=bool(body.get("reasoning_effort") or body.get("reasoning")),
    )


def task_from_ollama_body(body: Dict[str, Any], config: RouterConfig, headers: Mapping[str, str]) -> RouteTask:
    opts = body.get("options") or {}
    num_predict = opts.get("num_predict")
    return build_task(
        messages=body.get("messages") or [],
        config=config,
        headers=headers,
        requested_model=body.get("model"),
        max_output_tokens=num_predict if isinstance(num_predict, int) and num_predict > 0 else None,
        tools=body.get("tools"),
        json_mode=body.get("format") in ("json",) or isinstance(body.get("format"), dict),
        reasoning_requested=bool(body.get("think")),
    )
