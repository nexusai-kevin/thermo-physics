# Thermodynamic Physics & Computational Roadmap
**Nexus AI Solutions × Aquacellum Research Initiative**  
*Author: Kevin McDermott (`nexusai-kevin`)*  
*Repository: `thermo-physics` | Frameworks: Extropic `thrml`, JAX*

---

## Executive Summary
This roadmap outlines the mathematical transition of the `thermo-physics` benchmark suite from foundational statistical mechanics verifications to applied computational frontiers where physical thermodynamic hardware (continuous stochastic Langevin accelerators) is designed to provide asymptotic and energy-efficiency advantages over classical digital computing.

+-----------------------------------------------------------------------------+
|                               RESEARCH LAYERS                               |
+-----------------------------------------------------------------------------+
| Layer 1: Foundational Baselines (Exp 1–9)                                  |
|   - 1D/2D Ising, Potts (k=4), RBM associative memory, Max-Cut, QUBO         |
|   - Status: Complete & Verified (Upstream PR #79 submitted)                 |
+-----------------------------------------------------------------------------+
│
▼
+-----------------------------------------------------------------------------+
| Layer 2: Dense Frustrated Systems (Exp 10)                                  |
|   - Sherrington-Kirkpatrick (SK) Spin Glass with Gaussian quenched disorder |
|   - Replica Symmetry Breaking (RSB) & empirical Parisi P(q) distributions   |
|   - Status: Complete & Verified (N=24 exact ground-state match)             |
+-----------------------------------------------------------------------------+
│
▼
+-----------------------------------------------------------------------------+
| Layer 3: Applied Macromolecular Biophysics (Exp 11)                         |
|   - Boltzmann ensemble sampling of RNA secondary structure & riboswitches   |
|   - Native vs. thermal melt entropy transitions                             |
|   - Status: Complete & Verified (N=24 exact; N=50 scalable sampling)        |
+-----------------------------------------------------------------------------+
│
▼
+-----------------------------------------------------------------------------+
| Layer 4: Frontier Expansions (Planned / In Design)                          |
|   - 4A: ViennaRNA McCaskill Parity & Antimicrobial Peptide (AMP) Rotamers   |
|   - 4B: Microgrid Security-Constrained Unit Commitment (SCUC) QUBO          |
+-----------------------------------------------------------------------------+


---

## 1. Domain Mapping: Mathematical Formulations & Hardware Advantage

### Domain A: Frustrated Energy Landscapes & Spin Glasses (Implemented)
* **The Classical Bottleneck:** Digital architectures face severe scaling challenges on dense, non-planar frustrated graphs due to cache thrashing, non-local pointer chases, and exponential barrier crossing times ($\sim e^{\Delta F / k_B T}$). Single-spin updates cannot readily exploit spatial parallelism on fully-connected topologies.
* **The Thermodynamic Advantage:** Physical hardware is designed to leverage native thermal fluctuation across all couplings simultaneously, sampling degenerate ground-state valleys in continuous physical time.
* **Current Deliverable (Exp 10):** Sherrington-Kirkpatrick model ($N=24, M=276$ dense Gaussian couplings). Extracted empirical Parisi order parameter $\langle |q| \rangle = 0.5215$ and broad bimodal $P(q)$ support at $\beta=2.5$, contrasting with paramagnetic single-mode collapse ($\langle |q| \rangle = 0.1629$) at $\beta=0.5$.
* **Next Milestone:** Scaling to 3D Edwards-Anderson lattices ($L \times L \times L$) where the spin glass transition is strictly NP-hard.

---

### Domain B: Macromolecular Structural Ensembles & Biological Provenance (Active)
* **The Classical Bottleneck:** Standard bioinformatics tools (ViennaRNA, RNAfold, dynamic programming) focus primarily on predicting a single Minimum Free Energy (MFE) structure. However, functional biology—specifically riboswitches, aptamers, and viral RNA—operates as dynamic Boltzmann conformational ensembles. Exact partition function sampling on complex energy landscapes requires heavy matrix recursions.
* **The Thermodynamic Advantage:** Physical stochastic accelerators sample configurations directly proportional to their Boltzmann weights $e^{-\beta H(x)}$, capturing multi-valley conformational equilibria at physiological temperatures ($37^\circ\text{C}$).
* **Current Deliverable (Exp 11):** 
  - *Tier 1 ($N=24, M=24$):* Verified exact global minimum ($-10.00\text{ kcal/mol}$) against $2^{24}$ enumeration, resolving bistable contact switching between $\text{G}_0 \leftrightarrow \text{C}_{20}$ ($48.7\%$) and $\text{G}_0 \leftrightarrow \text{C}_{12}$ ($33.9\%$).
  - *Tier 2 ($N=50, M=72$):* Scalable sampling identifying dominant native stem-loops ($-66.00\text{ kcal/mol}$) and demonstrating conformational entropy doubling from native ($3.06\text{ bits}$) to thermal melt ($14.56\text{ bits}$).
* **Next Milestones:**
  1. *ViennaRNA Ensemble Parity:* Benchmark empirical contact matrices against classical McCaskill base-pairing probability matrices ($B_{ij}$). *Current contact probabilities are generated from a simplified energy model and have not yet been compared to McCaskill base-pairing matrices from ViennaRNA.*
  2. *Peptide Rotamer & AMP Design:* Extend from 4-base RNA graphs to 20-letter amino acid contact graphs for cyclic peptide stability and epidermal antimicrobial peptides (AMPs), bridging directly into Aquacellum captive-breeding biological provenance frameworks.

---

### Domain C: Industrial Grid Combinatorics — Microgrid SCUC (In Design)
* **The Classical Bottleneck:** Security-Constrained Unit Commitment (SCUC) under high renewable penetration requires solving large-scale Mixed-Integer Quadratic Programs (MIQP) repeatedly across operational windows. Digital branch-and-bound solvers can take tens of minutes under high system stress or volatile forecast variance.
* **The Thermodynamic Advantage:** Quadratic generator heat-rate curves and ramp-rate penalties map directly to continuous and discrete Ising couplings. Thermodynamic sampling provides rapid exploration of near-optimal feasible solutions during emergency dispatch windows.
* **Mathematical Formulation (Planned):**
  $$\min \sum_{t=1}^T \sum_{i=1}^G \left[ C_i(P_{i,t}) u_{i,t} + S_{i,t} + \lambda_{\text{load}}\left(\sum_i P_{i,t} u_{i,t} - D_t\right)^2 + \lambda_{\text{ramp}}(P_{i,t} - P_{i,t-1})^2 \right]$$
* **Target Deliverable:** A 10-generator, 24-hour microgrid dispatch benchmark script comparing thermodynamic QUBO convergence against open-source branch-and-bound solvers.

---

## 2. Research & Engagement Horizons

* **Near-Term Horizons:**
  - Upstream review engagement on PR #79 (`extropic-ai/thrml`).
  - Community engagement across the 11-experiment suite and `/thermo-lab` deployment.
  - Initial comparative benchmarking against McCaskill partition functions.
* **Medium-Term Horizons:**
  - Peptide rotamer packing and epidermal antimicrobial peptide (AMP) models.
  - Microgrid SCUC benchmark specification and baseline comparisons.
  - Formal submission to Extropic Hardware Developer / Research Access initiatives.
  - Integration with Aquacellum biological provenance testnet architecture.

---

## 3. Engineering Rigor & Scientific Boundaries
To maintain high academic integrity, all thermodynamic benchmarks adhere to:
1. **Explicit Ground Truth Baselines:** Every discrete problem instance is verified against brute-force enumeration ($2^N$ for small $N$) or proven classical optima (e.g., Held-Karp $L=6.296$ on 10-City TSP).
2. **Standardized MCMC Convergence Metrics:** All sampling trajectories report Sokal-windowed integrated autocorrelation time ($\tau_{\text{int}}$) and Effective Sample Size (ESS).
3. **Transparent Model Scope:** Approximations (e.g., simplified RNA stacking vs. full Turner rules) are explicitly stated to delineate physics proofs-of-concept from production bioinformatics engines.
