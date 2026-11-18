# Public Observatory Proof Dossier: Live `/thermo-lab` Empirical Streams & Cross-Vertical Refinery

**Author:** Kevin McDermott (Nexus AI Solutions / Aquacellum / AffilAI Studio)  
**Date:** November 18, 2026  
**Live Observatory:** `https://getnexusaisolutions.com/thermo-lab`  
**Core Engine Repository:** `https://github.com/nexusai-kevin/thermo-physics` (commit `4ae0d38`)  
**Frontend Repository:** `https://github.com/nexusai-kevin/nexus-site` (commit `f4e0987`)  
**Audited Manifest SHA-256:** `ae4f50d4319fedc63e4bd0849c6574063d5699abf6957bd1a3bb647c9a0f1632` (`164,093` bytes, synced across 8 ecosystem targets)

---

## Executive Summary

Starting from a zero-permission, 5-phase Live-Stream Data Refinery (`Collect -> Clean -> Aggregate -> Productize -> Compound`), `/thermo-lab` operates as an interactive thermodynamic observatory ingesting **unauthenticated public telemetry every 6 hours**, compiling live Ising ($J_{ij}, h_i$), Potts ($C_{ij}, f_i$), and Markowitz QUBO ($Q$) Hamiltonians, archiving gzipped SHA-256 deduplicated snapshots into `data/historical_vault/`, and streaming identical payloads into downstream web applications.

| Empirical Stream | Live Public Upstream Source (`source_mode`) | Hamiltonian / Tensor Dimensions | Audited Snapshot Telemetry (`2026-11-18T20:01:04Z`) |
| :--- | :--- | :--- | :--- |
| **Ecological Ising** | USGS NWIS Instantaneous Water Telemetry (`live_public_stream`) | 5 river/estuary stations × 16 sensor channels ($N=16$) | Exact $E_0 = -12.18294$, edge frustration `35.0%` (`52.14%` frustrated triangles), Sokal $\tau_{\text{int}} = 11.19$, ESS $= 14.30$ |
| **RNA Potts Epistasis** | Rfam `RF00005` tRNA seed alignment (`live_public_stream`) | $954 / 954$ sequences retained, $L = 24$ consensus cols, $q = 5$ states | $E_{\min} = -3.33352$, conformational entropy $S = 31.36\text{ bits}$, Sokal $\tau_{\text{int}} = 1.27$, ESS $= 125.61$ |
| **Financial & Prediction QUBO** | Yahoo Finance 1h OHLCV (16 assets) + Polymarket CLOB (16 books) (`live_public_stream`) | $N = 16$ assets, target cardinality $K = 5$ (`Q` + exact Ising `(J, h)`) | Optimal subset `["AAPL", "AMZN", "META", "GC=F", "NG=F"]`, QUBO energy `1.4205`, budget feasibility `100.0%`, Pareto match `100.0%`, Sokal $\tau_{\text{int}} = 28.46$, ESS $= 5.62$ |
| **Creator / DTC Brand Graph** | 15 Shopify DTC `/products.json` storefronts (`live_public_stream`) | 15 brands, 361 products, 1,252 variants + `brand_affinity_J` | Top deal score: `Brooklinen` (`89.17`, `PARTNER_15_20`), followed by `Keychron` (`80.6`), `Moment` (`79.1`), `Tentree` (`73.67`), `Glossier` (`72.91`) |

---

## 1. Brief for Extropic (`thrml` / TSU Hardware & Algorithmic Partnership Track)

- **Real-World Disorder Without Manual Fixture Hand-Crafting:** Rather than evaluating hardware and classical Gibbs samplers solely on synthetic spin glasses, `thermo_physics/refinery/` pulls fresh multi-sensor environmental covariance, biomolecular co-evolution couplings ($C_{ij}$), and order-book-penalized QUBOs every 6 hours via GitHub Actions (`.github/workflows/live_data_refinery.yml`) and a workstation scheduled runner (`ThermoLab_LiveDataRefinery_6h`, interval `PT6H`).
- **Sampling-Quality Diagnostics Built In:** Every snapshot logs exact enumeration ground states alongside Glauber/Gibbs Sokal autocorrelation time ($\tau_{\text{int}}$) and effective sample size ($\text{ESS}$). Noticeable critical slowing down appears directly in live river telemetry ($\tau_{\text{int}}$ shifting from `3.39` to a `13.17` peak sweeps across the 4-snapshot vault window) and dense portfolio QUBOs ($\tau_{\text{int}} = 28.46$, $\text{ESS} = 5.62$).
- **Interactive Tri-Mode Browser Observatory + Inbound Corpus Trap:** On `/thermo-lab`, `LiveEmpiricalRefineryDeck.tsx` exposes three modes:
  1. `Synthetic Baseline`
  2. `Live Public Telemetry`
  3. `Upload Custom Matrix / FASTA` (in-browser CSV/JSON Ising/QUBO matrix and FASTA alignment parser running client-side sweeps + live $\tau_{\text{int}}$ estimation)
  - Contributors can optionally submit **anonymized, schema-validated tensor diagnostics** via `POST /api/thermo-refinery/contribute`, which strictly enforces `anonymized_consent: true` (`HTTP 400` if false) and rejects raw biological sequences (`raw_sequences`) or email PII.
- **Provenance Separation (v0.1.0 Canonical Lock Intact):** All `22` canonical CSVs (`224` rows) under `workspace/goals/thermo-computing/files/thermal-benchmark-datasets/` remain immutable and pass `validate_benchmarks.py` independently, while TSU silicon projections remain clearly labeled as emulated/analytical baselines per `data-refinery-doctrine.md`.

---

## 2. Brief for DOE GEMS & Ecological / Biophysical Reviewers

- **Unauthenticated Hydrological Sensor Coupling (USGS NWIS):**
  - **5 Monitored Stations (16 Multi-Parameter Channels):**
    - `01463500` — Delaware River at Trenton, NJ
    - `02171645` — Rediversion Canal at Santee River near St. Stephen, SC
    - `07374000` — Mississippi River at Baton Rouge, LA
    - `11447650` — Sacramento River at Freeport, CA
    - `14211720` — Willamette River at Portland, OR
  - Continuous instantaneous telemetry (Temperature, pH, Dissolved Oxygen, Specific Conductance) is rolling-window normalized, binarized into spins $s_i \in \{-1, +1\}$, and inverted via regularized mean-field approximation into frustrated coupling graphs ($J_{ij}, h_i$).
  - Across the 4-snapshot `data/historical_vault/` consolidation window, automated regime detection tracked edge frustration shifting from `40.83%` to `35.00%` (`-5.83 pp`, frustrated triangles `57.50% -> 52.14%`, $E_0: -14.2648 \to -12.1829$) alongside a $\tau_{\text{int}}$ shift (`3.39 -> 11.19` sweeps, peak `13.17`).
- **Biomolecular Epistasis & Sequence Alignment (Rfam `RF00005`):**
  - Parses real tRNA seed alignments (`954 / 954` sequences retained, $L = 24$, $q = 5$ alphabet `A, C, G, U, -`) into empirical single-site frequencies $f_i(a)$ and covariance tensors $C_{ij}(a,b)$ for Potts energy benchmarking.
- **Direct Downstream Environmental Telemetry Integration (`Aquacellum`):**
  - Both `poseidon_echo_dojo` (`GET /api/refinery-telemetry`, `GET /api/status`, and the fishroom dashboard telemetry panel; commit `1bbe330`) and `freshfish-collectr` (`/dashboard/water-log` baseline card; commit `652337d`) consume the exact same SHA-256-verified `live_empirical_streams.json` payload.

---

## 3. Brief for Veronica Freund (`@veronica_freund87`) & AffilAI Studio Partners

- **Live 15-Brand Shopify DTC Catalog Refinery:**
  - Every 6 hours, the refinery polls unauthenticated `/products.json` feeds across 15 high-velocity DTC Shopify brands (`361` products, `1,252` variants) to compute composite deal scores combining price elasticity, variant depth, inventory availability/velocity, and tag taxonomy richness, plus cross-brand `brand_affinity_J` couplings.
  - **Latest Top-Ranked Brands:** `Brooklinen` (`89.17`, `16` products / `170` variants, `PARTNER_15_20`), `Keychron` (`80.6`, `25` / `132`), `Moment` (`79.1`, `25` / `123`), `Tentree` (`73.67`, `25` / `116`), `Glossier` (`72.91`, `20` / `20`).
- **Live Inside AffilAI Studio (`affilai-studio`, commit `a632530`):**
  - **Compass / Beacon Dashboard (`/dashboard`, Tab 3):** Displays the live 15-brand graph, top `brand_affinity_J` pairings, and 1-click `+ Add Live Brand Drop to Storefront` with `PARTNER_15_20` (15–20% direct + 5% squad bonus) commission tiers.
  - **Brand Portal (`/brand`):** `Load from Live 15-Brand Shopify Refinery Graph` quick-picker auto-populates commission presets.
  - **Product Hunter (`/dashboard/product-hunter`):** Surfaces live 15-brand catalog telemetry and angles directly from `/data/live_empirical_streams.json`.

---

## 4. Open Verification & Reproducibility Commands

Run the full verification suite against local workspace checkouts (Python `.venv` in `C:\Users\mcder\Desktop\thermo-physics`):

```powershell
# 1. Verify all 8 downstream live_empirical_streams.json targets, vault archive, and schema hashes
C:\Users\mcder\Desktop\thermo-physics\.venv\Scripts\python.exe C:\Users\mcder\Desktop\thermo-physics\scripts\run_live_refinery.py --verify

# 2. Run refinery unit & end-to-end pipeline tests (10 passed)
C:\Users\mcder\Desktop\thermo-physics\.venv\Scripts\pytest.exe C:\Users\mcder\Desktop\thermo-physics\tests\test_live_data_refinery.py -v

# 3. Verify immutable v0.1.0 canonical baseline (22 CSVs, 224 rows)
C:\Users\mcder\Desktop\thermo-physics\.venv\Scripts\python.exe C:\Users\mcder\workspace\goals\thermo-computing\files\validate_benchmarks.py

# 4. Run Cora 4-snapshot historical vault consolidation & regime-shift detector
C:\Users\mcder\Desktop\thermo-physics\.venv\Scripts\python.exe C:\Users\mcder\Cora\harness\scripts\morning_consolidation.py
```
