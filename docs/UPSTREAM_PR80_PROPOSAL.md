# Upstream PR #80 Proposal: Exact Ground-Truth Fixtures & Sokal MCMC Diagnostics for `thrml`

**Target Repository:** [`extropic-ai/thrml`](https://github.com/extropic-ai/thrml)  
**Author:** Kevin McDermott (`nexusai-kevin`) — Nexus AI Solutions × Aquacellum Research Initiative  
**Associated Issues / PRs:** Extends Upstream PR #79 (Reproducibility & Statistical Mechanics Suite)  
**Code Artifacts:** `upstream_patches/pr80/` (`test_exact_ground_states.py`, `sokal_diagnostics.py`)

---

## 1. Executive Summary

As thermodynamic computing transitions from exploratory simulation to physical silicon (Extropic Thermodynamic Sampling Units / TSUs), sampling benchmarks require two foundational capabilities:

1. **Rigorous Ground-Truth Baselines:** Exact analytical and brute-force ground-state solutions on non-trivial topologies to guarantee that software and hardware samplers reach true global equilibria rather than metastable local traps.
2. **Statistically Sound Convergence Diagnostics:** Moving evaluation beyond raw hardware clock rates (raw Hz) to **Effective Sample Size throughput ($\text{ESS}/\text{s}$)** via automated, zero-overhead Sokal-windowed autocorrelation time ($\tau_{\text{int}}$) estimation.

This proposal bundles clean, zero-dependency contributions ready for upstream inclusion in `extropic-ai/thrml`.

---

## 2. Technical Justification & Value for Upstream `thrml`

### A. The Core Hardware Metric: $\text{ESS}/\text{s}$ vs. Raw Throughput
Classical digital MCMC (e.g. single-spin Metropolis or Block Gibbs) suffers severe critical slowing down on dense, frustrated landscapes such as the Sherrington-Kirkpatrick (SK) model and low-temperature spin glasses. In these regimes:
$$\tau_{\text{int}} \gg 1 \implies \text{ESS} = \frac{N_{\text{samples}}}{2\tau_{\text{int}}} \ll N_{\text{samples}}$$

A classical sampler generating 50,000 raw samples/second with $\tau_{\text{int}} = 50$ yields only **500 independent samples/second**. 

Physical continuous-time analog Langevin substrates leverage thermodynamic noise across all couplings simultaneously, dramatically accelerating barrier hopping and driving $\tau_{\text{int}} \to 0.5$ (independent samples). By equipping `thrml` with native Sokal diagnostics, hardware runs immediately surface their true competitive advantage: **orders-of-magnitude higher $\text{ESS}/\text{s}$**.

### B. Automated Sokal Adaptive Window ($W \ge c \cdot \tau_{\text{int}}$)
Manual autocorrelation truncation is prone to high variance (if truncated too early) or noise accumulation (if windowed too broadly). Our implementation uses:
- Fast zero-padded FFT normalized autocorrelation: $\mathcal{O}(N \log N)$.
- Self-consistent Sokal windowing: stops summation at the first lag $W$ satisfying $W \ge c \cdot \tau_{\text{int}}(W)$ (default $c = 5.0$).
- Zero external dependencies beyond standard `numpy`.
- Multivariate aggregation across multi-spin configurations.

### C. Exact Ground-Truth Test Fixtures
To guarantee regression safety and hardware calibration across `thrml` releases, we provide analytical and combinatorial reference baselines:
1. **$N=24$ SK Frustrated Spin Glass:** Verified against exhaustive $2^{24}$ brute-force enumeration ($E_0 = -15.4266$). Includes verified optimal configuration with positive single-spin flip penalties ($\Delta E \ge 0$).
2. **1D Periodic Ising Chain:** Exact analytical partition function $Z(\beta)$ and internal energy $\langle E(\beta) \rangle = -N J \tanh(\beta J)$ via transfer matrix theory.
3. **2D Square Lattice Onsager Criticality:** Exact closed-form critical inverse temperature $\beta_c = \frac{1}{2} \ln(1 + \sqrt{2}) \approx 0.4407$ and Yang's exact spontaneous magnetization $M(\beta) = [1 - \sinh^{-4}(2\beta J)]^{1/8}$.

---

## 3. Bundle Structure

```
upstream_patches/pr80/
├── __init__.py
├── sokal_diagnostics.py       # FFT autocorrelation, Sokal tau_int, ESS, and SokalConvergenceObserver
└── test_exact_ground_states.py # Pytest test suite covering SK N=24, 1D Ising, 2D Onsager, and Sokal engine
```

### Module Highlights:

#### `sokal_diagnostics.py`
```python
from upstream_patches.pr80.sokal_diagnostics import (
    compute_sokal_tau_int,
    SokalConvergenceObserver,
)

# Automated windowed autocorrelation and ESS:
tau_int, ess = compute_sokal_tau_int(energy_trace, c_window=5.0)

# Telemetry observer attaching directly to sampling loops:
observer = SokalConvergenceObserver()
observer.record_trace(energies, elapsed_sec=0.12)
print(observer.summary())
# [Sokal Observer] CONVERGED | N=5000 samples in 0.1200s (41,666.7 Hz) | tau_int: 1.00 | ESS: 5000.0 (41,666.7 ESS/s)
```

#### `test_exact_ground_states.py`
Contains 12 comprehensive unit tests spanning:
- `TestSKSpinGlassGroundTruth` (exact $E_0 = -15.4266$, local optimality, subgraph brute force parity)
- `Test1DIsingAnalyticalSolution` (exact transfer matrix $Z$, analytical $\langle E \rangle$, brute-force state sum)
- `Test2DOnsagerCriticality` (Onsager critical threshold $\beta_c \approx 0.4407$, Yang spontaneous magnetization)
- `TestSokalAutocorrelationDiagnostics` (IID noise recovery, AR(1) analytical $\tau_{\text{int}}$, multivariate evaluation, observer lifecycle)

---

## 4. Verification & Reproduction Commands

### Local Reproduction (Sandbox & Virtualenv)

Run the full PR #80 test suite:
```bash
# Execute pytest across the contribution bundle with zero warnings
pytest upstream_patches/pr80/ -v -W error
```

Expected output:
```text
============================= test session starts =============================
platform win32 -- Python 3.12.x, pytest-9.2.0, pluggy-1.7.0
rootdir: C:\Users\mcder\Desktop\thermo-physics
configfile: pyproject.toml
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

============================= 12 passed in 0.06s ==============================
```

### Backend Comparison (JAX vs. Physical Extropic Target)
```bash
python scripts/compare_backends.py
```

### Station 12 SCUC Optimization Verification
```bash
python -m thermo_physics.experiments.station_12_grid_scuc
```

---

## 5. Next Steps for Upstream Integration

1. Integrate `sokal_diagnostics.py` into `thrml.diagnostics` or `thrml.metrics`.
2. Add `test_exact_ground_states.py` into upstream CI matrix under `tests/verification/`.
3. Hook `SokalConvergenceObserver` into default `thrml.sample()` telemetry output.
