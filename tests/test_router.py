"""
Unit and integration tests for the Nexus Rate-Limit-Aware Router & Replay Engine.
"""

from __future__ import annotations

import time
import pytest
import numpy as np

from thermo_physics.router.config import (
    EndpointConfig,
    RouterConfig,
    load_config,
)
from thermo_physics.router.headroom import HeadroomTracker, parse_reset_seconds, parse_retry_after
from thermo_physics.router.replay import CounterfactualReplayEngine
from thermo_physics.router.solver import ThermodynamicRouterSolver
from thermo_physics.router.tasks import RouteTask, build_task, task_from_openai_body


@pytest.fixture
def test_config() -> RouterConfig:
    endpoints = [
        EndpointConfig(
            name="Cloud-Frontier",
            provider="Anthropic",
            base_url="https://api.anthropic.com/v1",
            model="claude-3-5-sonnet",
            input_cost_per_1m=3.0,
            output_cost_per_1m=15.0,
            ttft_ms=500.0,
            ms_per_output_token=10.0,
            rpm_limit=10,
            tpm_limit=5_000,
            quality=3,
            capabilities=["chat", "code", "reasoning"],
        ),
        EndpointConfig(
            name="Cloud-Fast",
            provider="Google",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            model="gemini-1.5-flash",
            input_cost_per_1m=0.1,
            output_cost_per_1m=0.4,
            ttft_ms=250.0,
            ms_per_output_token=6.0,
            rpm_limit=100,
            tpm_limit=50_000,
            quality=2,
            capabilities=["chat", "code", "json"],
        ),
        EndpointConfig(
            name="Local-Ollama",
            provider="On-Premise",
            base_url="http://localhost:11434/v1",
            model="qwen2.5:3b",
            input_cost_per_1m=0.0,
            output_cost_per_1m=0.0,
            ttft_ms=100.0,
            ms_per_output_token=15.0,
            rpm_limit=500,
            tpm_limit=10_000,
            quality=1,
            capabilities=["chat", "fast_chat"],
        ),
    ]
    return RouterConfig(
        endpoints=endpoints,
        mode="shadow",
        strategy="thermo",
        batch_window_ms=50.0,
        max_batch=32,
    )


def test_headroom_parsing():
    assert parse_retry_after("12") == 12.0
    assert parse_retry_after("invalid") is None
    assert parse_reset_seconds("150ms") == 0.150
    assert parse_reset_seconds("5s") == 5.0
    assert parse_reset_seconds("1m30s") == 90.0


def test_task_extraction_privacy(test_config):
    body = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a code assistant."},
            {"role": "user", "content": "Write a python function: ```def solve(): pass```"},
        ],
        "max_tokens": 512,
    }
    task = task_from_openai_body(body, test_config, {})
    assert task.in_tokens > 10
    assert task.out_tokens_est == 512
    assert "code" in task.required_caps
    assert task.tier in ("complex", "medium")
    # Verify content strings are NOT stored on task
    assert not hasattr(task, "messages")
    assert not hasattr(task, "prompt")


def test_headroom_tracker(test_config):
    tracker = HeadroomTracker(test_config.endpoints)
    now = 1000.0
    tracker.record_dispatched("Cloud-Frontier", 2000, now=now)
    snap = tracker.get_snapshot("Cloud-Frontier", now=now + 5.0)
    assert snap.active_tpm == 2000
    assert snap.active_rpm == 1
    assert snap.headroom_tpm == 3000

    # Test purging after 60s
    snap_after = tracker.get_snapshot("Cloud-Frontier", now=now + 65.0)
    assert snap_after.active_tpm == 0
    assert snap_after.active_rpm == 0


def test_thermo_solver_contention_diversion(test_config):
    """When Cloud-Frontier is near capacity, thermo solver diverts lower-priority tasks."""
    solver = ThermodynamicRouterSolver(test_config)
    tracker = HeadroomTracker(test_config.endpoints)

    # 10 burst tasks requiring chat/code
    tasks = [
        RouteTask(
            request_id=f"t-{i}",
            ts=100.0,
            in_tokens=300,
            out_tokens_est=500,
            tier="simple" if i < 6 else "reasoning",
            min_quality=1 if i < 6 else 3,
            required_caps=["code"],
            max_latency_ms=10000.0,
            priority=1.0 if i < 6 else 3.0,
        )
        for i in range(10)
    ]

    snaps = tracker.all_snapshots(now=100.0)
    sol = solver.solve_thermo(tasks, snaps)
    assert len(sol.assignments) == 10
    assert sol.solve_time_ms < 500.0  # Under 500ms even on un-JITed interpreted loops

    # Frontier should only be chosen for high-quality tasks due to limited 5,000 TPM limit
    frontier_tasks = [a for a in sol.assignments if a.endpoint_name == "Cloud-Frontier"]
    assert len(frontier_tasks) <= 4  # Didn't blow the 5k TPM limit


def test_counterfactual_replay_report(test_config):
    engine = CounterfactualReplayEngine(test_config)
    tasks = [
        RouteTask(
            request_id=f"req-{i}",
            ts=1000.0 + i * 2.0,  # 1 request every 2s (high burst)
            in_tokens=500,
            out_tokens_est=300,
            tier="medium",
            min_quality=1,
            required_caps=["chat"],
            max_latency_ms=8000.0,
            priority=1.0,
            requested_model="Cloud-Frontier",  # Partner naively sends everything to expensive frontier
            actual_out_tokens=250,
        )
        for i in range(25)
    ]

    report = engine.replay(tasks, partner_id="Unit-Test-Partner")
    assert report.num_requests_analyzed == 25
    assert report.dollar_savings_vs_baseline > 0.0
    assert report.percent_savings_vs_baseline > 50.0  # Massive savings diverting away from pure frontier
    # Check that summary markdown renders properly
    md = report.summary_markdown()
    assert "Net Advantage" in md
