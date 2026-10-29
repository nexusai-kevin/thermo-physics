# Application Memo: Reproducible Sampling Workloads on `thrml`

**Applicant:** [Name] · [Organization: Nexus AI Solutions] · [email] · **Date:** [date]
**Repository:** https://github.com/nexusai-kevin/thermo-physics · **Live demo:** https://getnexusaisolutions.com/thermo-lab

## Summary

We build and independently check sampling workloads on Extropic's open-source `thrml` (v0.1.4, JAX). We are asking for
technical feedback on whether these workloads are representative of what Extropic hardware targets, and for access to
whatever evaluation path Extropic offers independent researchers. [Edit this ask to match the program you are applying to.]

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
4. **Upstream contribution.** A QUBO portfolio example (`examples/qubo_portfolio_optimization`) packaged as a patch
   for `extropic-ai/thrml`; it applies cleanly to current `main` and passes upstream's black/isort/flake8 config.
5. **Educational front end.** The `/thermo-lab` page runs nine interactive stations in the browser. These are
   JavaScript Glauber/Metropolis re-implementations for teaching, not calls into `thrml`.

## Target workloads and where each stands

| Workload | Result | Honest status |
| --- | --- | --- |
| Portfolio risk balancing (N=20, K=5) | 99.1% budget feasibility; sampled energy 94.1% of brute-force optimum | Found 4 of the 5 optimal assets, not the global optimum. Constraint is soft (penalty). Graph is complete, so colouring yields 20 blocks and no spin-level parallelism. |
| Combinatorial routing (10-city TSP) | Reaches the optimal tour (6.296) | Final tours come from thrml sampling **plus classical 2-opt refinement** on decoded states; a pure-sampling TSP result is not yet demonstrated. 100 spins, 28 colour blocks. |
| Max-Cut (50 nodes, 243 edges) | 70.8% of edges cut vs 49.9% random | Optimality gap to the true maximum not measured. |
| Biological sequence / provenance modelling | None yet | Planned. Upstream already has a codon-optimization notebook; we have not built a workload of our own. |

## Known limitations

Single synthetic instances and single seeds, so these are demonstrations rather than statistical benchmarks against
classical solvers. The continuous double-well sampler (Exp 8) is a custom heuristic conditional, not an exact
Gibbs/Langevin kernel for the stated Hamiltonian. All timings are CPU.

## Request

Feedback on workload relevance and on how Extropic would want sampling quality (ESS, τ_int) reported for
cross-hardware comparison; and, if available, information on access for independent researchers. [Adjust to the actual
programme terms; they were not verified when this draft was written.]
