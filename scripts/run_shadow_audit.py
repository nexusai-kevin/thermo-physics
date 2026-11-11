"""
scripts/run_shadow_audit.py
===========================
Executes counterfactual shadow-replay audits on client log files (JSONL/CSV)
or runs a self-contained simulated burst demonstration.
"""

import argparse
import json
import time
from thermo_physics.router import (
    CounterfactualReplayEngine,
    RouteTask,
    load_config,
)


def run_synthetic_burst_demo(engine, partner_name: str = "Demo-Agent-Fleet"):
    """Simulates a 40-request burst of agent tasks arriving within 30 seconds."""
    tasks = []
    tiers = ["simple", "simple", "medium", "complex", "reasoning"]
    for i in range(50):
        t = tiers[i % len(tiers)]
        req_caps = ["code"] if t in ("complex", "reasoning") else ["chat"]
        tasks.append(
            RouteTask(
                request_id=f"burst-req-{i:03d}",
                ts=1000.0 + i * 0.6,  # ~1.6 req/s
                in_tokens=350 + (i % 5) * 200,
                out_tokens_est=256 if t == "simple" else (512 if t == "medium" else 1024),
                tier=t,
                min_quality=1 if t == "simple" else (2 if t == "medium" else 3),
                required_caps=req_caps,
                max_latency_ms=8000.0 if t == "simple" else 25000.0,
                priority=1.0 if t == "simple" else 3.0,
                requested_model="Claude-3.5-Sonnet",  # Naive baseline: sends everything to top frontier model
                actual_out_tokens=200 if t == "simple" else 400,
            )
        )

    report = engine.replay(tasks, partner_id=partner_name)
    print(report.summary_markdown())
    return report


def main():
    parser = argparse.ArgumentParser(description="Nexus Counterfactual Shadow Audit Runner")
    parser.add_argument("--config", default="configs/router_default.json", help="Path to router config")
    parser.add_argument("--logs", default=None, help="Path to input JSONL logs (omit for synthetic burst demo)")
    parser.add_argument("--partner", default="Partner-Trial", help="Partner identifier for report")
    parser.add_argument("--out", default=None, help="Path to export report JSON")
    args = parser.parse_args()

    cfg = load_config(args.config)
    engine = CounterfactualReplayEngine(cfg)

    if args.logs:
        tasks = engine.load_tasks_from_jsonl(args.logs)
        report = engine.replay(tasks, partner_id=args.partner)
        print(report.summary_markdown())
    else:
        print("[*] No log file specified. Running 50-task agent fleet burst simulation...")
        report = run_synthetic_burst_demo(engine, partner_name=args.partner)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"[*] Exported full audit report to: {args.out}")


if __name__ == "__main__":
    main()
