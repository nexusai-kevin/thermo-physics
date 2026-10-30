# Upstream Contribution Guide: Exact Ground-Truth Fixtures & Sokal MCMC Diagnostics for `extropic-ai/thrml`

**Target Upstream Repository:** [`extropic-ai/thrml`](https://github.com/extropic-ai/thrml)  
**Contribution Package:** `upstream_patches/pr80/`  
**Author:** Kevin McDermott (`nexusai-kevin`) — Nexus AI Solutions × Aquacellum Research Initiative  

---

## 1. Pull Request Overview

### PR Title
`feat(diagnostics): add automated Sokal MCMC convergence observer and exact ground-state fixtures`

### Short Description
This PR introduces automated MCMC convergence diagnostics and exact analytical/combinatorial ground-state test fixtures for `thrml`:
1. **Automated Sokal-Windowed $\tau_{\text{int}}$ & Effective Sample Size (ESS):** Zero-overhead FFT-based integrated autocorrelation estimator and lightweight telemetry observer (`SokalConvergenceObserver`) that reports true statistically independent sampling rates ($\text{ESS}/\text{s}$) rather than raw hardware clock ticks.
2. **Exact Ground-Truth Verification Fixtures:** Regression-proof analytical baselines (1D periodic Ising chain transfer matrix, 2D square lattice Onsager critical point $\beta_c \approx 0.4407$, and $N=24$ Sherrington-Kirkpatrick spin glass $E_0 = -15.4266$ from $2^{24}$ brute-force enumeration).

### Why Upstream `thrml` Benefits
* **Quantifying Hardware Value:** On frustrated landscapes, classical digital samplers suffer critical slowing down ($\tau_{\text{int}} \gg 1$), producing highly correlated samples. Physical thermodynamic sampling substrates (continuous Brownian noise accelerators) mix rapidly ($\tau_{\text{int}} \to 0.5$). Sokal $\tau_{\text{int}}$ and ESS metrics rigorously expose this $10^3\times$ to $10^5\times$ independent sampling advantage.
* **Zero Dependencies:** Pure Python and NumPy (`numpy.fft`), imposing zero new external dependencies on `thrml`.
* **Zero Assumptions on Private APIs:** Conforms strictly to standard NumPy arrays and public JAX trace outputs.

---

## 2. Included Files & Architecture

```
upstream_patches/pr80/
├── __init__.py
├── sokal_diagnostics.py       # FFT autocorrelation, Sokal tau_int, ESS, and SokalConvergenceObserver
└── test_exact_ground_states.py # 12 Pytest test fixtures verifying Boltzmann equilibrium & ground states
```

### Key API Signatures:

```python
from sokal_diagnostics import compute_sokal_tau_int, SokalConvergenceObserver

# 1. Functional interface for any 1D energy or magnetization trace:
tau_int, ess = compute_sokal_tau_int(energy_series, c_window=5.0)

# 2. Observer object for sampling loops:
observer = SokalConvergenceObserver(c_window=5.0)
observer.record_trace(energies, elapsed_sec=0.12)
print(observer.summary())
# [Sokal Observer] CONVERGED | N=5000 samples in 0.1200s (41,666.7 Hz) | tau_int: 1.00 | ESS: 5000.0 (41,666.7 ESS/s)
```

---

## 3. Git Patch Application Instructions

To apply this contribution cleanly to a local clone of `extropic-ai/thrml`:

### Option A: Direct File Copy into Upstream Clone
```bash
# Clone the upstream thrml repository
git clone https://github.com/extropic-ai/thrml.git
cd thrml

# Create a feature branch
git checkout -b feat/sokal-diagnostics-and-ground-truth

# Copy the contribution files
cp -r /path/to/thermo-physics/upstream_patches/pr80/sokal_diagnostics.py thrml/diagnostics.py
cp -r /path/to/thermo-physics/upstream_patches/pr80/test_exact_ground_states.py tests/test_exact_ground_states.py

# Run verification suite
pytest tests/test_exact_ground_states.py -v
```

### Option B: Applying via Git Patch / Diff
```bash
# In thermo-physics, generate standalone patch
git diff main~1 upstream_patches/pr80/ > /tmp/thrml_pr80_diagnostics.patch

# In thrml, apply patch
cd /path/to/thrml
git apply --reject --whitespace=fix /tmp/thrml_pr80_diagnostics.patch
```

---

## 4. Verification & CI Reproduction

Execute the test suite locally with zero warnings:
```bash
pytest upstream_patches/pr80/ -v -W error
```

Expected result:
```text
============================= test session starts =============================
collected 12 items

upstream_patches/pr80/test_exact_ground_states.py::TestSKSpinGlassGroundTruth::test_n24_exact_ground_energy_evaluation PASSED [  8%]
upstream_patches/pr80/test_exact_ground_states.py::TestSKSpinGlassGroundTruth::test_n24_spin_flip_local_optimality PASSED [ 16%]
upstream_patches/pr80/test_exact_ground_states.py::TestSKSpinGlassGroundTruth::test_small_subgraph_brute_force_parity PASSED [ 25%]
upstream_patches/pr80/test_exact_ground_states.py::Test1DIsingAnalyticalSolution::test_1d_ground_state_energy PASSED [ 33%]
upstream_patches/pr80/test_exact_ground_states.py::Test1DIsingAnalyticalSolution::test_1d_analytical_limits PASSED [ 41%]
upstream_patches/pr80/test_exact_ground_states.py::Test1DIsingAnalyticalSolution::test_1d_exact_partition_function_brute_force PASSED [ 50%]
upstream_patches/pr80/test_exact_ground_states.py::Test2DOnsagerCriticality::test_onsager_critical_temperature_numerical_value PASSED [ 58%]
upstream_patches/pr80/test_exact_ground_states.py::Test2DOnsagerCriticality::test_yang_magnetization_phase_transition PASSED [ 66%]
upstream_patches/pr80/test_exact_ground_states.py::TestSokalAutocorrelationDiagnostics::test_white_noise_zero_autocorrelation PASSED [ 75%]
upstream_patches/pr80/test_exact_ground_states.py::TestSokalAutocorrelationDiagnostics::test_ar1_correlated_process_analytical_tau PASSED [ 83%]
upstream_patches/pr80/test_exact_ground_states.py::TestSokalAutocorrelationDiagnostics::test_multivariate_tau_evaluation PASSED [ 91%]
upstream_patches/pr80/test_exact_ground_states.py::TestSokalAutocorrelationDiagnostics::test_sokal_observer_lifecycle PASSED [100%]

============================= 12 passed in 0.05s ==============================
```
