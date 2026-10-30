# Scientific Methodology & Energy Accounting Standards

**Repository:** `thermo-physics`  
**Author:** Kevin McDermott (`nexusai-kevin`) — Nexus AI Solutions × Aquacellum Research  
**Scope:** Rigorous mathematical standards, MCMC convergence metrics, energy conventions, and hardware projection models.

---

## 1. Executive Standards: Exact vs. Empirical Benchmarking

To ensure absolute academic rigor and reproducible engineering claims, `thermo-physics` delineates between:
1. **Exact Analytical & Combinatorial Ground Truths** (where closed-form or exhaustive $2^N$ proofs exist).
2. **Empirical Sampling Approximations** (where continuous Langevin or discrete Markov processes estimate thermodynamic expectations).

Every benchmark station in this repository must anchor its empirical sampling against a mathematically rigorous baseline.

```
+-----------------------------------------------------------------------------------+
|                        BENCHMARK ANCHOR HIERARCHY                                 |
+-----------------------------------------------------------------------------------+
| Tier 1: Exact Analytical Solutions (1D Ising Z, 2D Onsager critical point beta_c) |
| Tier 2: Exhaustive Brute-Force Baselines (2^N exact state sums for N <= 24)       |
| Tier 3: Classical Proven Combinatorial Optima (e.g. Held-Karp L = 6.296 on TSP)    |
| Tier 4: Statistical Convergence Metrics (Sokal tau_int, ESS, TV Distance <= 0.05) |
+-----------------------------------------------------------------------------------+
```

---

## 2. Hamiltonian Energy Conventions & the 0.5 Factor

A frequent source of numerical discrepancy across Ising and QUBO software libraries is the factor of $0.5$ on pairwise coupling terms. `thermo-physics` strictly adheres to the standard statistical physics undirected convention.

### A. The Standard Undirected Ising Hamiltonian
For an undirected graph $G = (V, E)$ represented by a symmetric coupling matrix $J \in \mathbb{R}^{N \times N}$ ($J_{ij} = J_{ji}$ with zero diagonal $J_{ii} = 0$) and external magnetic field vector $h \in \mathbb{R}^N$:

$$E(s) = -\frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N J_{ij} s_i s_j - \sum_{i=1}^N h_i s_i + E_{\text{offset}} = -\frac{1}{2} s^T J s - h^T s + E_{\text{offset}}$$

where spins $s_i \in \{-1, +1\}$.

> [!IMPORTANT]
> **The 0.5 Pre-Factor Rule:**  
> When the double sum $\sum_{i=1}^N \sum_{j=1}^N J_{ij} s_i s_j$ is evaluated over a full symmetric matrix, every physical undirected edge $\{i, j\}$ is counted twice ($J_{ij} s_i s_j + J_{ji} s_j s_i$). The factor $\frac{1}{2}$ cancels this double counting:
> $$\frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N J_{ij} s_i s_j = \sum_{i < j} J_{ij} s_i s_j$$
> Omission of this factor artificially doubles interaction energies and distorts effective inverse temperature $\beta \to 2\beta$.

### B. QUBO to Ising Transformation
For binary decision variables $x_i \in \{0, 1\}$ and objective $\min x^T Q x$:
$$x_i = \frac{s_i + 1}{2}$$

Substituting into $x^T Q x$ yields:
$$x^T Q x = \frac{1}{4} s^T Q s + \frac{1}{2} \left(\operatorname{diag}(Q) + \sum_{j} Q_{ij} - Q_{ii}\right) s + \text{constant}$$

Mapping to our standardized Ising form:
$$J_{ij} = -0.5 \cdot Q_{ij} \quad (i \ne j), \quad h_i = -0.5 \cdot \left(Q_{ii} + \sum_{j \ne i} Q_{ij}\right)$$

---

## 3. Ground-Truth Exact Verification ($N \le 24$)

For system sizes up to $N = 24$, the total state space is:
$$|\Omega| = 2^{24} = 16,777,216 \text{ configurations}$$

### A. Exhaustive Vectorized Enumeration
Using bit-shift matrix indexing, all $2^{24}$ states are enumerated in vectorized 32-bit chunks:
$$E_0 = \min_{s \in \{-1, +1\}^N} \left(-\frac{1}{2} s^T J s - h^T s\right)$$

For Station 10 (Sherrington-Kirkpatrick model with quenched Gaussian seed 42):
$$E_0 = -15.42659072466675 \quad (E_0 / N = -0.6428)$$

### B. Single-Spin Flip Stability Criterion
Any candidate global minimum $s^*$ must satisfy strict local stability across all $N$ single-spin coordinate directions:
$$\Delta E_i = E(s^{* \oplus i}) - E(s^*) = 2 s_i^* \left(\sum_{j=1}^N J_{ij} s_j^* + h_i\right) \ge 0 \quad \forall i \in \{1, \dots, N\}$$

---

## 4. MCMC Mixing Diagnostics: Sokal-Windowed $\tau_{\text{int}}$ and ESS

Raw hardware clock rate (sample sweeps per second, Hz) is **not** an accurate measure of computational throughput on frustrated energy landscapes. Highly correlated samples provide diminishing statistical information.

### A. Normalized Autocorrelation via FFT
For a scalar observable sequence $\{X_t\}_{t=1}^N$ (e.g. instantaneous Hamiltonian energy $E_t$ or magnetization $M_t$), let $\bar{X} = \frac{1}{N} \sum X_t$. The sample autocorrelation at lag $k$ is:
$$\hat{\rho}(k) = \frac{\frac{1}{N-k} \sum_{t=1}^{N-k} (X_t - \bar{X})(X_{t+k} - \bar{X})}{\frac{1}{N} \sum_{t=1}^N (X_t - \bar{X})^2}$$

This is evaluated in $\mathcal{O}(N \log N)$ operations using zero-padded Fast Fourier Transforms:
$$\hat{C} = \operatorname{IFFT}\left(|\operatorname{FFT}(X - \bar{X})|^2\right)$$

### B. Sokal's Self-Consistent Truncation Window
The integrated autocorrelation time is defined as:
$$\tau_{\text{int}} = \frac{1}{2} + \sum_{k=1}^\infty \rho(k)$$

Summing over all lags $k \to N$ introduces high variance from noisy tails. Following Alan Sokal (1997), we truncate the summation at lag $W$, where $W$ is the smallest integer satisfying:
$$W \ge c_{\text{window}} \cdot \tau_{\text{int}}(W)$$

`thermo-physics` standardizes on $c_{\text{window}} = 5.0$, with a physical lower bound $\tau_{\text{int}} \ge 0.5$ (the theoretical value for independent, identically distributed samples).

### C. Effective Sample Size (ESS) and Independent Generation Rate
The number of statistically independent samples is:
$$\text{ESS} = \frac{N_{\text{samples}}}{2 \tau_{\text{int}}}$$

The true throughput of an execution backend is:
$$\text{Rate}_{\text{independent}} = \frac{\text{ESS}}{\Delta t_{\text{wallclock}}} \quad \left[\frac{\text{independent samples}}{\text{second}}\right]$$

---

## 5. Total Variation (TV) Distance for Distributional Verification

For small solvable graphs ($N \le 8$), we verify that empirical frequencies converge to the exact Boltzmann-Gibbs distribution:
$$P_{\text{Boltzmann}}(s) = \frac{e^{-\beta E(s)}}{Z(\beta)}, \quad Z(\beta) = \sum_{s \in \{-1, +1\}^N} e^{-\beta E(s)}$$

The Total Variation distance is:
$$D_{\text{TV}}(P_{\text{empirical}}, P_{\text{Boltzmann}}) = \frac{1}{2} \sum_{s \in \Omega} |P_{\text{empirical}}(s) - P_{\text{Boltzmann}}(s)|$$

A sampling run is certified as Boltzmann equilibrium if:
$$D_{\text{TV}} \le 0.05 \quad (\text{with } N_{\text{samples}} \ge 10,000)$$

---

## 6. Hardware Projections: Assumptions & Limitations

To maintain credibility when comparing classical GPUs to physical thermodynamic hardware (Extropic TSUs), all architectural projections are explicitly parameterized.

### Explicit Model Parameters (Extropic Z1 Baseline)
* **P-bit Capacity:** 269,568 stochastic analog p-bits per die (organized across 8 cores).
* **Sampling Rate:** 50 MHz native state-update clock.
* **Thermal Power Envelope:** $\le 1.0\text{ W}$ total die power.
* **GPU Baseline:** NVIDIA RTX 4090 / A100 (~300W–400W TDP, ~40,000–80,000 raw Gibbs sweeps/sec on dense graphs).

### Scientific Limitations & Caveats
1. **Analog Noise & Non-Idealities:** Physical TSUs operate via continuous Brownian thermal motion across analog capacitor/transistor networks. Thermal noise is subject to device mismatch, thermal drift, and 8-bit to 12-bit DAC quantization limits on coupling matrices $J_{ij}$.
2. **Topological Mapping:** Dense all-to-all graphs (e.g. SK model) require non-planar routing or virtual spin embedding if the physical substrate uses sparse chimera or king's graph topologies.
3. **The 10,000× Energy Efficiency Advantage:** The projected $10,000\times$ efficiency gain ($\text{Joules per independent sample}$) stems from:
   $$\frac{\text{GPU Power}}{\text{TSU Power}} \times \frac{\tau_{\text{int}}^{\text{GPU}}}{\tau_{\text{int}}^{\text{TSU}}} \approx \frac{300\text{ W}}{1\text{ W}} \times \frac{50}{1} \approx 15,000\times$$
   This advantage is realized predominantly on **dense, frustrated landscapes** where classical MCMC experiences severe critical slowing down ($\tau_{\text{int}} \gg 1$), whereas on unfrustrated high-temperature graphs the classical baseline has much lower $\tau_{\text{int}}$.
4. **RNA Secondary Structure Approximation Scope:** In Station 13, the stem-loop hairpin detector ($L_{\text{stem}} \ge 3$, loop $3\text{--}8\text{ nt}$) provides an efficient, zero-dependency first-order approximation of ribosome stall sites. Full production biophysics pipelines (e.g. ViennaRNA / NUPACK nearest-neighbor dynamic programming) evaluate detailed loop entropies ($\Delta G$), which can be substituted into the same Potts Hamiltonian factor graph when external bioinformatics C-libraries are linked.
