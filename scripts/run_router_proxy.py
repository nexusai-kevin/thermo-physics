"""
scripts/run_router_proxy.py
===========================
Standalone runner for the Nexus Thermodynamic Rate-Limit-Aware LLM Router & Shadow Proxy.

Usage:
  python scripts/run_router_proxy.py --port 8092 --mode shadow --config configs/router_default.json
"""

import argparse
import uvicorn
from thermo_physics.router import create_router_app, load_config


def main():
    parser = argparse.ArgumentParser(description="Nexus Thermodynamic LLM Router & Proxy")
    parser.add_argument("--host", default="0.0.0.0", help="Binding host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8092, help="Binding port (default: 8092)")
    parser.add_argument("--config", default="configs/router_default.json", help="Path to router config JSON")
    parser.add_argument("--mode", choices=["shadow", "route"], default=None, help="Override proxy mode")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.mode:
        cfg.mode = args.mode

    app = create_router_app(cfg)
    print(f"[*] Starting Nexus Router Proxy on {args.host}:{args.port}")
    print(f"[*] Operational Mode: {cfg.mode.upper()} | Strategy: {cfg.strategy.upper()}")
    print(f"[*] Managed Endpoints ({len(cfg.endpoints)}):")
    for ep in cfg.endpoints:
        print(f"    - {ep.name:22} | Provider: {ep.provider:10} | TPM: {ep.tpm_limit:,} | Quality: {ep.quality}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
