# Application Memo: Reproducible Sampling Workloads on `thrml`

**Applicant:** Kevin McDermott (Founder & Systems Architect, Nexus AI Solutions / Aquacellum) · **Contact:** kevin@aquacellum.com · **Date:** October 29, 2026  
**Target Program:** Extropic Algorithmic Partnership & Open Research Grant Track  
**Repository:** https://github.com/nexusai-kevin/thermo-physics · **Live demo:** https://getnexusaisolutions.com/thermo-lab

## Summary

We build and independently check sampling workloads on Extropic's open-source `thrml` (v0.1.4, JAX). We are submitting this application for the **Extropic Algorithmic Partnership & Open Research Grant Track**, requesting technical feedback on workload alignment with thermodynamic hardware targets and early evaluation access on Extropic's thermodynamic compute units (TSUs).

## What exists today

1. **Reproducibility suite: 9 experiments, one command.** 2D Ising (Onsager criticality), Max-Cut, Potts (K=4),
   image denoising with clamped observations, RBM associative memory, a QUBO portfolio problem, a continuous
   double-well chain, and a 10-city TSP. `uv run thrml-bench run-all` runs all nine and regenerates
   [`BENCHMARKS.md`](../BENCHMARKS.md). Every figure below is produced by that command.
2. **Sampling-quality diagnostics.** The CLI reports integrated autocorrelation time and effective sample size
   (Sokal-windowed, validated against AR(1) processes with known τ_int). At the 2D-Ising critical point (β = 0.4407,
   32×32) τ_int = 75 sweeps, so 4,000 samples carry ESS ≈ 53. Away from criticality ESS is close to N. We report this
   because it is the number that determines how many samples a workload actually needs.
3. **Throughput measurement.** 20.18 M spin-updates/s at 128×128 (16,384 spins), CPU only, 500 sweeps. The
   8,192× figure in our tables is the *theoretical* ratio N/2 of sequential to two-colour updates per sweep, not a
   measured wall-clock speedup. Sweeps/s is flat across 16² to 128², so small lattices are overhead-bound.
4. **Upstream contribution.** A QUBO portfolio example (`examples/qubo_portfolio_optimization`) submitted upstream as [PR #79](https://github.com/extropic-ai/thrml/pull/79) on `extropic-ai/thrml`; it applies cleanly to `main` and conforms to upstream's black/isort/flake8 config.
5. **Educational front end.** The `/thermo-lab` page runs nine interactive stations in the browser. These are
   JavaScript Glauber/Metropolis re-implementations for teaching, not calls into `thrml`.

## Target workloads and where each stands

| Workload | Result | Honest status |
| --- | --- | --- |
| Portfolio risk balancing (N=20, K=5) | 99.1% budget feasibility; sampled energy 94.1% of brute-force optimum | Found 4 of the 5 optimal assets, not the global optimum. Constraint is soft (penalty). Graph is complete, so colouring yields 20 blocks and no spin-level parallelism. |
| Combinatorial routing (10-city TSP) | Reaches exact Held-Karp optimum (6.296) under pure sampling | Evaluated on 100-spin permutation lattice (28 colors). Fixed penalty ($\lambda=2.8$) yields pure best 8.438 (+34.0% gap, repaired to 6.296 via 2-opt); geometrically ramped penalty ($\lambda=0.5 \to 5.0$, $\beta=0.2 \to 8.0$) finds the exact optimum (6.296) purely thermodynamically (1/5 restarts, 100% valid permutation matrices) without classical repair. |
| Frustrated spin glass (SK model, N=24) | 100.0% exact ground-state match ($E_0 = -18.4120$) | Dense all-to-all Gaussian couplings (276 edges). Dual non-interacting replicas extract Parisi overlap distribution $P(q)$ with $\langle |q| \rangle = 0.6124$ and $\tau_{\text{int}} = 14.82$. Matches exact $2^{24}$ brute-force enumeration. |
| Max-Cut (50 nodes, 243 edges) | 70.8% of edges cut vs 49.9% random | Optimality gap to the true maximum not measured. |
| Biological sequence / provenance modelling | None yet | Planned. Upstream already has a codon-optimization notebook; we have not built a workload of our own. |

### Frontier Expansion: Frustrated Non-Planar Spin Glasses (SK Model)
Beyond standard sparse planar graphs, we have expanded the suite to dense, frustrated spin glasses via the Sherrington-Kirkpatrick (SK) model ($N=24, J_{ij} \sim \mathcal{N}(0, 1/N)$). By deploying dual non-interacting Markov replicas over identical quenched disorder, we extract the empirical Parisi overlap distribution $P(q)$ and autocorrelation diagnostics ($\tau_{\text{int}} = 14.82$). The sampler achieves 100% ground-state discovery matching exact $2^{24}$ enumeration while revealing multi-valley Parisi structures ($\langle |q| \rangle = 0.6124$). This establishes a baseline for comparing classical Gibbs mixing against continuous physical thermal fluctuations in high-frustration regimes.

## Known limitations

Single synthetic instances and single seeds, so these are demonstrations rather than statistical benchmarks against
classical solvers. The continuous double-well sampler (Exp 8) is a custom heuristic conditional, not an exact
Gibbs/Langevin kernel for the stated Hamiltonian. All timings are CPU.

## Request

Technical feedback on workload relevance and benchmarking protocols (particularly standardizing ESS, τ_int, and parallel speedups across thermodynamic hardware architectures), alongside developer/researcher access and partnership under the Extropic Algorithmic Partnership & Open Research Grant Track.
