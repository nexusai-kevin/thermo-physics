# Cora Overnight App Factory & Dojo Memory Summary

Updated: 2026-11-18 15:19

## Verified Overnight Apps
Last night's Overnight App Factory produced 15 verified stateless PWAs (`READY_FOR_REVIEW`, passed `npm run typecheck`, `npm run build`, and port 3110 paywall smoke tests): bill-defender, sub-escape-hatch, deposit-shield, warranty-rescue, junk-fee-scanner, scam-shield, school-chaos-tamer, stain-spill-er, declutter-coach, leftover-remixer, listing-flip-kit, voice-to-quote, ghost-proof-followup, complaint-deescalator, brain-dump-extractor. Plus `fridge-to-table` (Mission 01 verified).

## Failure Lessons & Code Repairs
All 191 overnight builds initially failed Phase-3 typecheck because `src/lib/mockAnalyze.ts` imported `./types` or `@/types` or used `export default` instead of `import type { AnalyzeResponse, AppMode } from "@/types/app";` and named `export function buildMockAnalysis`. Additionally, 176 brainstormed `autonomous_queue` apps failed with repetitive slop names (`idea`, `auto`, `virtual`, `flash`, `-xN`). Rolling lessons are now written to `overnight_lessons.md` and injected into Phase 1 & Phase 3. Nightly Dojo misses (phantom_trap/quad_layer_gaslight_and_override_trap, tool_scavenger/repo_audit_chain, incident_wargame/vram_spike_and_fsm_rescue, duplex_and_isolation/mid_dispatch_redirect_rollback) are now permanently enrolled in `harness/dojo_regression_suite.jsonl`.

## Standing Factory Rules
Overnight App Factory standing invariants: strictly stateless (browser `IndexedDB` + `localStorage` only, zero cloud DBs or user accounts), dual one-time monetization (3 free uses, Lifetime Pass $19-$29, BYOK Pass $9-$14, zero subscriptions), zero liability (no medical/legal/tax/financial advice), 5-app nightly quality cap, 180s command timeout, and hard-ban on `Virtual`, `Auto`, `Flash`, `Idea`, `Bot`, `Pro`, `Lite`, `Ultimate`, and `-xN` suffixes.

## Refinery Vault Regime Telemetry
Vault snapshots archived: 9 (latest window: refinery_snapshot_20261118T172422Z.json.gz, refinery_snapshot_20261118T172952Z.json.gz, refinery_snapshot_20261118T195339Z.json.gz, refinery_snapshot_20261118T200104Z.json.gz; latest payload SHA-256: `f36444ab8d9dfb97`). Regime comparison: [ALERT] usgs_frustration_shift: Edge frustration 40.83% -> 35.00% (-5.83 pp); triangle frustration 57.50% -> 52.14%; E0 -14.2648 -> -12.1829.; [ALERT] usgs_critical_slowing_shift: Sokal tau_int 3.39 -> 11.19 sweeps (+229.7%, window peak 13.17, ESS 14.3).; [STABLE] polymarket_qubo_portfolio_flip: Stable optimum [AAPL, AMZN, META, GC=F, NG=F]; tau_int range 26.23-29.15 sweeps; peak Polymarket OBI 0.9932 (SI=F).; [STABLE] shopify_dtc_stockout_or_velocity_shift: No >=5% drops or >=0.05 velocity spikes; top brand Brooklinen (89.2 / PARTNER_15_20); high-velocity watch: ColourPop (0.960), Glossier (0.960), Native (0.863).
