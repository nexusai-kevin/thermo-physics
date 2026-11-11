"""
thermo_physics.router.replay
============================
Counterfactual Shadow-Replay Engine for Design Partner Pilots.

Ingests partner request logs (JSONL or CSV), reconstructs timestamp-ordered
sliding windows, and replays traffic simultaneously through:
  1. The Partner's Historical Baseline (what actually happened)
  2. Isolated Greedy Dispatch (cheapest/fastest per request)
  3. Nexus Thermodynamic Potts/Gibbs Batch Dispatch

Generates the audit-grade counterfactual report:
  - Exact $ spend comparison
  - Simulated 429 rate-limit breaches (excess over TPM/RPM limits)
  - SLA breach counts
  - Quantified net savings and headroom metrics
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from thermo_physics.router.config import EndpointConfig, RouterConfig
from thermo_physics.router.headroom import HeadroomTracker
from thermo_physics.router.solver import ThermodynamicRouterSolver
from thermo_physics.router.tasks import RouteTask


@dataclass
class ReplayMetrics:
    strategy_name: str
    total_requests: int
    total_cost_usd: float
    total_tokens: int
    sla_breaches: int
    rate_limit_breaches_429: int
    avg_latency_ms: float
    p95_latency_ms: float
    solve_overhead_ms_total: float
    model_breakdown: Dict[str, int]


@dataclass
class ReplayReport:
    partner_id: str
    log_source: str
    num_requests_analyzed: int
    time_span_hours: float
    baseline: ReplayMetrics
    greedy: ReplayMetrics
    thermo: ReplayMetrics
    dollar_savings_vs_baseline: float
    percent_savings_vs_baseline: float
    dollar_savings_vs_greedy: float
    percent_savings_vs_greedy: float
    breach_reduction_vs_baseline: int
    breach_reduction_vs_greedy: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary_markdown(self) -> str:
        pct_b = self.percent_savings_vs_baseline
        pct_g = self.percent_savings_vs_greedy
        return f"""
# Nexus Counterfactual Shadow-Replay Audit: {self.partner_id}
**Time Span:** {self.time_span_hours:.2f} hours | **Total Requests:** {self.num_requests_analyzed:,}

| Metric | Historical Baseline | Greedy Baseline | Nexus Thermo Dispatch | Net Advantage |
| :--- | :--- | :--- | :--- | :--- |
| **Total Cost ($)** | ${self.baseline.total_cost_usd:.4f} | ${self.greedy.total_cost_usd:.4f} | **${self.thermo.total_cost_usd:.4f}** | **{pct_b:+.1f}% vs Base (${self.dollar_savings_vs_baseline:+.2f})** |
| **429 Rate-Limit Breaches** | {self.baseline.rate_limit_breaches_429} | {self.greedy.rate_limit_breaches_429} | **{self.thermo.rate_limit_breaches_429}** | **-{self.breach_reduction_vs_baseline} fewer 429s** |
| **Latency SLA Misses** | {self.baseline.sla_breaches} | {self.greedy.sla_breaches} | **{self.thermo.sla_breaches}** | **{self.baseline.sla_breaches - self.thermo.sla_breaches} fewer misses** |
| **P95 Latency (ms)** | {self.baseline.p95_latency_ms:.1f} ms | {self.greedy.p95_latency_ms:.1f} ms | **{self.thermo.p95_latency_ms:.1f} ms** | - |
| **Avg Latency (ms)** | {self.baseline.avg_latency_ms:.1f} ms | {self.greedy.avg_latency_ms:.1f} ms | **{self.thermo.avg_latency_ms:.1f} ms** | - |
| **Routing Overhead (Avg)** | 0.0 ms | {self.greedy.solve_overhead_ms_total / max(1, self.num_requests_analyzed):.2f} ms | **{self.thermo.solve_overhead_ms_total / max(1, self.num_requests_analyzed):.2f} ms** | Ultra-low JAX/NumPy latency |

### Model Distribution Breakdown
- **Baseline:** `{json.dumps(self.baseline.model_breakdown)}`
- **Greedy:**   `{json.dumps(self.greedy.model_breakdown)}`
- **Thermo:**   `{json.dumps(self.thermo.model_breakdown)}`
"""


class CounterfactualReplayEngine:
    """Simulates multi-day or multi-hour production log traffic under different routing regimes."""

    def __init__(self, config: RouterConfig):
        self.config = config
        self.solver = ThermodynamicRouterSolver(config)

    def load_tasks_from_jsonl(self, path: Union[str, Path]) -> List[RouteTask]:
        """Loads RouteTask instances from a JSONL file."""
        tasks: List[RouteTask] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                # Map from raw logged fields to RouteTask
                tasks.append(
                    RouteTask(
                        request_id=d.get("request_id", d.get("id", f"req-{len(tasks)}")),
                        ts=float(d.get("timestamp", d.get("ts", len(tasks)))),
                        in_tokens=int(d.get("in_tokens", d.get("prompt_tokens", 100))),
                        out_tokens_est=int(d.get("out_tokens_est", d.get("max_tokens", 256))),
                        tier=d.get("tier", "medium"),
                        min_quality=int(d.get("min_quality", 1)),
                        required_caps=d.get("required_caps", []),
                        max_latency_ms=float(d.get("max_latency_ms", 15_000.0)),
                        priority=float(d.get("priority", 1.0)),
                        requested_model=d.get("model", d.get("requested_model")),
                        actual_out_tokens=int(d.get("out_tokens", d.get("completion_tokens", d.get("out_tokens_est", 256)))),
                    )
                )
        tasks.sort(key=lambda t: t.ts)
        return tasks

    def replay(self, tasks: List[RouteTask], partner_id: str = "Partner-0") -> ReplayReport:
        if not tasks:
            raise ValueError("No tasks provided for replay")

        time_span_h = max(0.01, (tasks[-1].ts - tasks[0].ts) / 3600.0)

        # 1. Run Baseline Replay (what partner did)
        base_metrics = self._simulate_baseline(tasks)

        # 2. Run Greedy Replay
        greedy_metrics = self._simulate_strategy(tasks, mode="greedy")

        # 3. Run Thermodynamic Gibbs Replay
        thermo_metrics = self._simulate_strategy(tasks, mode="thermo")

        savings_base = base_metrics.total_cost_usd - thermo_metrics.total_cost_usd
        pct_base = (savings_base / max(0.0001, base_metrics.total_cost_usd)) * 100.0

        savings_greedy = greedy_metrics.total_cost_usd - thermo_metrics.total_cost_usd
        pct_greedy = (savings_greedy / max(0.0001, greedy_metrics.total_cost_usd)) * 100.0

        return ReplayReport(
            partner_id=partner_id,
            log_source="Simulated/Ingested Log",
            num_requests_analyzed=len(tasks),
            time_span_hours=time_span_h,
            baseline=base_metrics,
            greedy=greedy_metrics,
            thermo=thermo_metrics,
            dollar_savings_vs_baseline=savings_base,
            percent_savings_vs_baseline=pct_base,
            dollar_savings_vs_greedy=savings_greedy,
            percent_savings_vs_greedy=pct_greedy,
            breach_reduction_vs_baseline=base_metrics.rate_limit_breaches_429 - thermo_metrics.rate_limit_breaches_429,
            breach_reduction_vs_greedy=greedy_metrics.rate_limit_breaches_429 - thermo_metrics.rate_limit_breaches_429,
        )

    def _simulate_baseline(self, tasks: List[RouteTask]) -> ReplayMetrics:
        """Evaluates what actually occurred in the partner's recorded log."""
        tracker = HeadroomTracker(self.config.endpoints)
        costs: List[float] = []
        lats: List[float] = []
        sla_misses = 0
        breaches_429 = 0
        breakdown: Dict[str, int] = {}
        total_tokens = 0

        for task in tasks:
            ep = self.config.find_endpoint(task.requested_model) or self.config.endpoints[0]
            breakdown[ep.name] = breakdown.get(ep.name, 0) + 1

            out_tok = task.actual_out_tokens or task.out_tokens_est
            tot_tok = task.in_tokens + out_tok
            total_tokens += tot_tok

            # Check if this request violated rate limits at dispatch time
            snap = tracker.get_snapshot(ep.name, now=task.ts)
            if (snap.active_tpm + tot_tok) > ep.tpm_limit or (snap.active_rpm + 1) > ep.rpm_limit:
                breaches_429 += 1
                tracker.record_429(ep.name, retry_after=5.0, now=task.ts)

            tracker.record_dispatched(ep.name, tot_tok, now=task.ts)

            cost = ep.cost_usd(task.in_tokens, out_tok)
            lat = ep.est_latency_ms(out_tok)
            costs.append(cost)
            lats.append(lat)
            if lat > task.max_latency_ms:
                sla_misses += 1

        p95 = float(np.percentile(lats, 95)) if lats else 0.0
        avg = float(np.mean(lats)) if lats else 0.0

        return ReplayMetrics(
            strategy_name="Historical Baseline",
            total_requests=len(tasks),
            total_cost_usd=sum(costs),
            total_tokens=total_tokens,
            sla_breaches=sla_misses,
            rate_limit_breaches_429=breaches_429,
            avg_latency_ms=avg,
            p95_latency_ms=p95,
            solve_overhead_ms_total=0.0,
            model_breakdown=breakdown,
        )

    def _simulate_strategy(self, tasks: List[RouteTask], mode: str = "thermo") -> ReplayMetrics:
        """Groups tasks into small batch windows and routes them."""
        tracker = HeadroomTracker(self.config.endpoints)
        batch_window_sec = self.config.batch_window_ms / 1000.0

        costs: List[float] = []
        lats: List[float] = []
        sla_misses = 0
        breaches_429 = 0
        breakdown: Dict[str, int] = {}
        total_tokens = 0
        total_solve_ms = 0.0

        # Chunk into time batches
        i = 0
        n = len(tasks)
        while i < n:
            batch_start_ts = tasks[i].ts
            batch: List[RouteTask] = []
            while i < n and (tasks[i].ts - batch_start_ts) <= batch_window_sec and len(batch) < self.config.max_batch:
                batch.append(tasks[i])
                i += 1

            snaps = tracker.all_snapshots(now=batch_start_ts)

            if mode == "thermo":
                sol = self.solver.solve_thermo(batch, snaps)
            else:
                sol = self.solver.solve_greedy(batch, snaps)

            total_solve_ms += sol.solve_time_ms

            for decision in sol.assignments:
                t = decision.task
                ep = self.config.find_endpoint(decision.endpoint_name)
                if ep is None:
                    ep = self.config.endpoints[0]
                breakdown[ep.name] = breakdown.get(ep.name, 0) + 1

                out_tok = t.actual_out_tokens or t.out_tokens_est
                tot_tok = t.in_tokens + out_tok
                total_tokens += tot_tok

                # Check if rate limit was breached
                snap = tracker.get_snapshot(ep.name, now=t.ts)
                if (snap.active_tpm + tot_tok) > ep.tpm_limit or (snap.active_rpm + 1) > ep.rpm_limit:
                    breaches_429 += 1
                    tracker.record_429(ep.name, retry_after=5.0, now=t.ts)

                tracker.record_dispatched(ep.name, tot_tok, now=t.ts)

                cost = ep.cost_usd(t.in_tokens, out_tok)
                lat = ep.est_latency_ms(out_tok)
                costs.append(cost)
                lats.append(lat)
                if lat > t.max_latency_ms:
                    sla_misses += 1

        p95 = float(np.percentile(lats, 95)) if lats else 0.0
        avg = float(np.mean(lats)) if lats else 0.0

        return ReplayMetrics(
            strategy_name=f"Nexus {mode.capitalize()}",
            total_requests=len(tasks),
            total_cost_usd=sum(costs),
            total_tokens=total_tokens,
            sla_breaches=sla_misses,
            rate_limit_breaches_429=breaches_429,
            avg_latency_ms=avg,
            p95_latency_ms=p95,
            solve_overhead_ms_total=total_solve_ms,
            model_breakdown=breakdown,
        )
