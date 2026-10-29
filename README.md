# Thermodynamic Computing Reference Implementations & Reproducibility Suite

A modular reference implementation and hardware-aware reproducibility benchmark suite for **Energy-Based Models (EBMs)**, **Factor Graphs**, **Multi-Color Block Gibbs Sampling**, and **Denoising Thermodynamic Models (DTMs)** compatible with Extropic's `thrml` software stack.

---

## 1. Architectural Overview: Deterministic Layers vs. Energy Landscapes

| Paradigm | State Representation | Forward / Inference Mechanism | Hardware Primitive |
| :--- | :--- | :--- | :--- |
| **Classical Neural Network** | Deterministic activations $x \in \mathbb{R}^n$ | Matrix-vector multiply $y = f(Wx + b)$ | GPU Tensor Core (MAC arrays, 200–400W) |
| **Thermodynamic EBM (`thrml`)** | Stochastic spin states $s_i \in \{-1, +1\}$ | Physical thermal relaxation $P(s) = \frac{1}{Z} e^{-\beta E(s; \theta)}$ | Thermodynamic Sampling Unit (p-bit CMOS subthreshold arrays, ~1W) |

### Graph Coloring & Parallel Block Gibbs
To sample from the Boltzmann distribution $P(s) = \frac{1}{Z(\theta)} e^{-\beta E(s; \theta)}$ in parallel without violating detailed balance, the factor graph is partitioned into **independent sets (`Block`s)** where no two nodes in the same `Block` share an edge:
- **Bipartite V-H-L Classifier (Project A):** Visible ($V$) and Label ($L$) nodes connect only to Hidden ($H$) nodes, enabling **2-cycle parallel updates** (`[Block(V + L), Block(H)]`).
- **2D Lattice DTM (Project B):** Nearest-neighbor 2D grids are 2-colorable (checkerboard parity `(r + c) % 2`), allowing an entire $H \times W$ spin array to relax in **2 parallel hardware clock cycles** per sweep regardless of lattice size.

---

## 2. Repository Structure

```text
thermo-physics/
├── pyproject.toml                     # Package metadata & dependencies
├── README.md                          # Architecture, mapping guide, and benchmark results
├── first_ising.py                     # Minimal 5-spin 1D Ising chain starter example
├── run_suite.py                       # Top-level CLI entrypoint for the full benchmark suite
├── benchmark_results.json             # Generated empirical reproducibility metrics
├── thermo_repro/
│   ├── __init__.py                    # Unified API exports
│   ├── backend.py                     # Native `thrml` loader + built-in JAX/NumPy compatible engine
│   ├── coloring.py                    # Independent-set graph coloring & block verifier
│   ├── metrics.py                     # Exact Z(θ), Autocorrelation (τ_int), and TSU vs. GPU energy profiler
│   ├── project_a_classifier.py        # Project A: Bipartite V-H-L EBM Classifier & Associative Memory (CD-k)
│   ├── project_b_dtm.py               # Project B: 2D Lattice Denoising Thermodynamic Model (DTM)
│   └── benchmark_runner.py            # End-to-end benchmark orchestrator & CLI
└── tests/
    └── test_suite.py                  # Unit & statistical equilibrium verification tests
```

---

## 3. Reference Implementations

### Project A: Associative Memory & Robust Classification (`thermo_repro/project_a_classifier.py`)
Maps a multi-class classifier and Hopfield associative memory onto a 3-layer bipartite factor graph with Visible nodes $V$, Hidden nodes $H$, and spin-encoded Label nodes $L$:

$$E(v, h, l) = -\sum_{i} a_i v_i - \sum_{j} b_j h_j - \sum_{k} c_k l_k - \sum_{i,j} W_{ij} v_i h_j - \sum_{j,k} U_{jk} h_j l_k$$

- **Training:** Contrastive Divergence ($\text{CD-}k$) comparing positive clamped correlations $\langle v_i h_j \rangle_{\text{pos}}, \langle h_j l_k \rangle_{\text{pos}}$ against free bipartite Block Gibbs trajectories.
- **Classification by Relaxation:** Clamps $V$ to a noisy input (e.g., 20% random spin flips) and relaxes `[Block(H), Block(L)]` to read out the class posterior $\arg\max_k \langle l_k \rangle$.
- **Associative Pattern Completion:** Clamps only the unoccluded visible spins $V_{\text{obs}}$ and jointly relaxes `[Block(V_masked + L), Block(H)]` to reconstruct occluded patterns.

### Project B: Denoising Thermodynamic Model Suite (`thermo_repro/project_b_dtm.py`)
Replaces score-based reverse diffusion with physical thermal annealing on a 2D spin lattice:

$$E(s \mid y) = -J_{\text{prior}} \sum_{\langle i,j \rangle} s_i s_j - \sum_{i} \lambda_{\text{obs}} y_i s_i, \quad \lambda_{\text{obs}} = \frac{1}{2}\ln\left(\frac{1 - p_{\text{noise}}}{p_{\text{noise}}}\right)$$

- **Execution:** Anneals inverse temperature $\beta_1 < \beta_2 < \dots < \beta_K$ using 2-color checkerboard `Block` updates (`[Block(even_sites), Block(odd_sites)]`) and compares hardware clock cycles against sequential single-site MCMC.

---

## 4. Reproducibility Benchmark Results

Running `python run_suite.py --show-dtm-ascii` executes all five verification rubrics:

| Benchmark Rubric | Metric | Measured Result |
| :--- | :--- | :--- |
| **1. Bipartite EBM & TSU Energy Profile** | $64 \times 32$ RBM (1,100 sweeps) | $\langle E \rangle = -20.338$, **104.2 fJ/spin-update** projected on TSU vs. **6.94 mJ/sweep** on 300W GPU (**$6.9 \times 10^8\times$** energy reduction) |
| **2. Exact Boltzmann Equilibrium** | Brute-force $Z(\theta)$ on $2 \times 4$ lattice (256 states) | Exact $\langle E \rangle = -1.7699$ vs. Empirical $\langle E \rangle = -1.7526$ ($\|\Delta E\| = 0.0174$, $D_{\text{TV}} = 0.0466$, $D_{\text{KL}} = 0.0102$) |
| **3. Mixing Time & Autocorrelation ($\tau_{\text{int}}$)** | $6 \times 6$ Lattice: 2-Color Block vs. Sequential Gibbs | $\tau_{\text{int}} = 0.98$ sweeps (2 clock cycles/sweep vs. 36 clock cycles/sweep $\rightarrow$ **18.0x parallel cycle speedup**) |
| **4. Project A: V-H-L EBM Classifier** | Clean / Noisy (20% flips) / Occluded (37.5% masked) | **100.0%** clean accuracy, **93.8%** noisy accuracy, **100.0%** occluded associative recall |
| **5. Project B: 2D Lattice DTM** | $12 \times 12$ Lattice with 25% spin corruption | Accuracy recovered from **75.0% $\rightarrow$ 83.3%** (**33.3%** bit-error reduction), **72.0x** fewer hardware clock cycles than sequential MCMC |

---

## 5. Quickstart Usage

```powershell
# Run the full reproducibility benchmark suite and export benchmark_results.json
.\.venv\Scripts\python.exe run_suite.py --show-dtm-ascii

# Run the automated unit & statistical verification test suite
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
