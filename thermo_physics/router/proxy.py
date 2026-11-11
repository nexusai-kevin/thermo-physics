"""
thermo_physics.router.proxy
===========================
FastAPI-based OpenAI-Compatible Reverse Proxy & Shadow Router.

Endpoints:
  POST /v1/chat/completions (OpenAI Chat Completions protocol)
  POST /api/chat            (Ollama-native chat protocol)
  GET  /v1/models           (List available virtual and physical endpoints)
  GET  /nexus/telemetry     (Real-time rate-limit headroom & router savings)
  POST /nexus/replay        (Run counterfactual shadow replay over uploaded JSONL)

Modes:
  - "shadow": Forward request to client's requested model, solve optimal route in background,
              and log counterfactual delta (Zero risk, non-invasive design partner pilot).
  - "route": Actively dispatch request to the thermodynamically optimal endpoint.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import httpx
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from thermo_physics.router.config import EndpointConfig, RouterConfig, load_config
from thermo_physics.router.headroom import HeadroomTracker
from thermo_physics.router.replay import CounterfactualReplayEngine
from thermo_physics.router.solver import ThermodynamicRouterSolver
from thermo_physics.router.tasks import (
    RouteTask,
    task_from_ollama_body,
    task_from_openai_body,
)

logger = logging.getLogger("nexus.router.proxy")


def create_router_app(config: RouterConfig) -> FastAPI:
    app = FastAPI(
        title="Nexus Thermodynamic LLM Router & Shadow Proxy",
        description="Rate-limit-aware thermodynamic batch dispatcher and counterfactual shadow auditor.",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    tracker = HeadroomTracker(config.endpoints)
    solver = ThermodynamicRouterSolver(config)
    client = httpx.AsyncClient(timeout=120.0)

    # Queue for micro-batching
    batch_queue: List[Tuple[RouteTask, asyncio.Future, Dict[str, Any]]] = []
    queue_lock = asyncio.Lock()

    os.makedirs(os.path.dirname(config.log_path) or ".", exist_ok=True)

    def log_decision(entry: Dict[str, Any]) -> None:
        try:
            with open(config.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error("Failed to append decision log: %s", e)

    @app.on_event("shutdown")
    async def shutdown():
        await client.aclose()

    @app.get("/v1/models")
    async def list_models():
        data = []
        for ep in config.endpoints:
            data.append({
                "id": ep.name,
                "object": "model",
                "created": 1700000000,
                "owned_by": ep.provider,
                "permission": [],
                "root": ep.model,
            })
            for alias in ep.aliases:
                data.append({
                    "id": alias,
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": ep.provider,
                    "permission": [],
                    "root": ep.model,
                })
        return {"object": "list", "data": data}

    @app.get("/nexus/telemetry")
    async def get_telemetry():
        snaps = tracker.all_snapshots()
        return {
            "mode": config.mode,
            "strategy": config.strategy,
            "batch_window_ms": config.batch_window_ms,
            "headroom": {k: asdict(v) for k, v in snaps.items()},
        }

    async def _forward_upstream(
        ep: EndpointConfig,
        body: Dict[str, Any],
        client_auth: Optional[str] = None,
        stream: bool = False,
    ):
        target_url = ep.base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}

        # API Key precedence: config env var > client Auth header
        if ep.api_key_env:
            key = os.environ.get(ep.api_key_env)
            if key:
                headers["Authorization"] = f"Bearer {key}"
        elif client_auth:
            headers["Authorization"] = client_auth

        # Rewrite payload model to upstream's required model ID
        forward_body = dict(body)
        forward_body["model"] = ep.model

        t0 = time.perf_counter()
        if stream:
            req = client.build_request("POST", target_url, headers=headers, json=forward_body)
            resp = await client.send(req, stream=True)
            return resp, t0
        else:
            resp = await client.post(target_url, headers=headers, json=forward_body)
            return resp, t0

    @app.post("/v1/chat/completions")
    async def openai_chat_completions(
        request: Request,
        background_tasks: BackgroundTasks,
        authorization: Optional[str] = Header(None),
    ):
        body = await request.json()
        headers = dict(request.headers)
        task = task_from_openai_body(body, config, headers)
        stream = bool(body.get("stream", False))

        # Take headroom snapshot
        snaps = tracker.all_snapshots(now=task.ts)

        # Solve optimal route
        if config.strategy == "thermo":
            sol = solver.solve_thermo([task], snaps)
        else:
            sol = solver.solve_greedy([task], snaps)

        decision = sol.assignments[0]
        optimal_ep = config.find_endpoint(decision.endpoint_name) or config.endpoints[0]

        # In shadow mode, send to user's requested model if found; otherwise send to optimal
        if config.mode == "shadow":
            target_ep = config.find_endpoint(task.requested_model) or optimal_ep
        else:
            target_ep = optimal_ep

        # Track dispatch
        tracker.record_dispatched(target_ep.name, task.total_tokens_est, now=task.ts)

        # Forward request upstream
        try:
            upstream_resp, t0 = await _forward_upstream(target_ep, body, authorization, stream=stream)
        except Exception as e:
            logger.error("Upstream dispatch failed to %s: %s", target_ep.name, e)
            raise HTTPException(status_code=502, detail=f"Upstream dispatch error: {str(e)}")

        # Handle rate limits
        if upstream_resp.status_code == 429:
            tracker.record_429(target_ep.name, now=time.time())

        if stream:
            async def event_generator() -> AsyncGenerator[bytes, None]:
                total_bytes = 0
                async for chunk in upstream_resp.aiter_bytes():
                    total_bytes += len(chunk)
                    yield chunk

                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                approx_tokens = max(1, total_bytes // 4)
                tracker.record_completed(target_ep.name, approx_tokens, upstream_resp.headers)

                # Record decision log in background
                log_entry = {
                    "request_id": task.request_id,
                    "timestamp": task.ts,
                    "mode": config.mode,
                    "requested_model": task.requested_model,
                    "dispatched_model": target_ep.name,
                    "optimal_model": optimal_ep.name,
                    "actual_latency_ms": elapsed_ms,
                    "estimated_tokens": task.total_tokens_est,
                    "actual_tokens_approx": approx_tokens,
                    "target_cost": target_ep.cost_usd(task.in_tokens, approx_tokens),
                    "optimal_cost": optimal_ep.cost_usd(task.in_tokens, approx_tokens),
                    "sla_breach": elapsed_ms > task.max_latency_ms,
                    "status_code": upstream_resp.status_code,
                }
                log_decision(log_entry)

            return StreamingResponse(
                event_generator(),
                status_code=upstream_resp.status_code,
                media_type=upstream_resp.headers.get("content-type", "text/event-stream"),
            )
        else:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            data = upstream_resp.json()
            usage = data.get("usage") or {}
            out_tokens = usage.get("completion_tokens", task.out_tokens_est)
            tot_tokens = usage.get("total_tokens", task.in_tokens + out_tokens)

            tracker.record_completed(target_ep.name, tot_tokens, upstream_resp.headers)

            log_entry = {
                "request_id": task.request_id,
                "timestamp": task.ts,
                "mode": config.mode,
                "requested_model": task.requested_model,
                "dispatched_model": target_ep.name,
                "optimal_model": optimal_ep.name,
                "actual_latency_ms": elapsed_ms,
                "estimated_tokens": task.total_tokens_est,
                "actual_tokens": tot_tokens,
                "target_cost": target_ep.cost_usd(task.in_tokens, out_tokens),
                "optimal_cost": optimal_ep.cost_usd(task.in_tokens, out_tokens),
                "sla_breach": elapsed_ms > task.max_latency_ms,
                "status_code": upstream_resp.status_code,
            }
            background_tasks.add_task(log_decision, log_entry)

            response = JSONResponse(content=data, status_code=upstream_resp.status_code)
            # Inject Nexus audit headers
            response.headers["x-nexus-dispatched"] = target_ep.name
            response.headers["x-nexus-optimal"] = optimal_ep.name
            response.headers["x-nexus-mode"] = config.mode
            response.headers["x-nexus-headroom-tpm"] = str(snaps[target_ep.name].headroom_tpm)
            return response

    @app.post("/nexus/replay")
    async def run_replay(request: Request):
        """Run counterfactual replay audit on uploaded JSONL tasks."""
        body = await request.json()
        raw_tasks = body.get("tasks") or []
        partner_id = body.get("partner_id", "Pilot-Partner")

        engine = CounterfactualReplayEngine(config)
        tasks = [
            RouteTask(
                request_id=t.get("request_id", f"req-{i}"),
                ts=float(t.get("ts", time.time())),
                in_tokens=int(t.get("in_tokens", 200)),
                out_tokens_est=int(t.get("out_tokens_est", 256)),
                tier=t.get("tier", "medium"),
                min_quality=int(t.get("min_quality", 1)),
                required_caps=t.get("required_caps", []),
                max_latency_ms=float(t.get("max_latency_ms", 15_000.0)),
                priority=float(t.get("priority", 1.0)),
                requested_model=t.get("requested_model"),
                actual_out_tokens=int(t.get("actual_out_tokens", t.get("out_tokens_est", 256))),
            )
            for i, t in enumerate(raw_tasks)
        ]

        if not tasks:
            raise HTTPException(status_code=400, detail="No tasks found in replay payload")

        report = engine.replay(tasks, partner_id=partner_id)
        return report.to_dict()

    return app
