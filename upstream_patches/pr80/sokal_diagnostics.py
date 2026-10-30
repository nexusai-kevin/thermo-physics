"""
upstream_patches.pr80.sokal_diagnostics
=======================================

Automated Sokal-windowed autocorrelation time (tau_int) and Effective Sample Size (ESS)
diagnostic engine for thermodynamic sampling traces in extropic-ai/thrml.

Features:
- Fast FFT-based normalized autocorrelation O(N log N)
- Automated Sokal adaptive window cutoff W >= c * tau_int(W)
- Zero external dependencies beyond standard NumPy
- Pluggable SokalConvergenceObserver designed to attach directly to thrml sampling traces
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


def compute_normalized_autocorrelation(series: np.ndarray) -> np.ndarray:
    """Compute normalized autocorrelation rho(t) = C(t) / C(0) via zero-padded FFT.

    Parameters
    ----------
    series : np.ndarray
        1D array of scalar observations (e.g. energy, magnetization, or spin trajectory).

    Returns
    -------
    np.ndarray
        Normalized autocorrelation vector of length N, with rho[0] = 1.0.
    """
    series = np.asarray(series, dtype=np.float64)
    n = len(series)
    if n == 0:
        return np.array([1.0], dtype=np.float64)

    mean = np.mean(series)
    zero_mean = series - mean
    var = np.var(zero_mean)
    if var <= 1e-15:
        # Uniform series has no autocorrelation decay
        return np.ones(n, dtype=np.float64)

    # FFT with 2N padding to avoid circular convolution artifact
    n_fft = 2 * n
    f = np.fft.fft(zero_mean, n=n_fft)
    autocorr = np.fft.ifft(f * np.conj(f))[:n].real
    autocorr = autocorr / (autocorr[0] + 1e-15)
    return autocorr


def compute_sokal_tau_int(
    series: np.ndarray,
    c_window: float = 5.0,
    min_tau: float = 0.5,
) -> Tuple[float, float]:
    """Compute integrated autocorrelation time tau_int using Sokal's self-consistent window.

    The integrated autocorrelation time is defined as:
        tau_int = 1/2 + sum_{t=1}^{W} rho(t)

    where W is chosen adaptively as the smallest lag satisfying:
        W >= c_window * tau_int(W)

    The Effective Sample Size (ESS) is then:
        ESS = N / (2 * tau_int)

    Parameters
    ----------
    series : np.ndarray
        1D array of measurements across sequential MCMC/Langevin iterations.
    c_window : float
        Sokal window scaling factor (standard literature values: 4.0 to 6.0; default 5.0).
    min_tau : float
        Lower bound for tau_int (0.5 for independent identically distributed samples).

    Returns
    -------
    Tuple[float, float]
        (tau_int, ess) where tau_int is the autocorrelation time and ess is Effective Sample Size.
    """
    series = np.asarray(series, dtype=np.float64)
    n = len(series)
    if n < 2:
        return min_tau, float(max(1, n))

    zero_mean = series - np.mean(series)
    var = np.var(zero_mean)
    if var <= 1e-15:
        return min_tau, float(n)

    autocorr = compute_normalized_autocorrelation(series)

    # Sokal self-consistent truncation
    tau_int = 0.5
    for lag in range(1, n):
        tau_int += autocorr[lag]
        if lag >= c_window * tau_int:
            break

    tau_int = max(min_tau, float(tau_int))
    ess = max(1.0, float(n / (2.0 * tau_int)))
    return tau_int, ess


def compute_multivariate_tau_int(
    samples: np.ndarray,
    c_window: float = 5.0,
) -> Dict[str, Any]:
    """Compute tau_int and ESS across all spin variables in a multivariate trajectory.

    Parameters
    ----------
    samples : np.ndarray
        Shape (n_samples, n_spins) array of sampled configurations.
    c_window : float
        Sokal window cutoff parameter.

    Returns
    -------
    Dict[str, Any]
        Dictionary with per-spin and aggregate convergence metrics (mean, max, min).
    """
    samples = np.asarray(samples)
    assert samples.ndim == 2, f"Expected 2D array (n_samples, n_spins), got {samples.shape}"
    n_samples, n_spins = samples.shape

    tau_list = []
    ess_list = []
    for j in range(n_spins):
        t_int, e = compute_sokal_tau_int(samples[:, j], c_window=c_window)
        tau_list.append(t_int)
        ess_list.append(e)

    tau_arr = np.array(tau_list)
    ess_arr = np.array(ess_list)

    return {
        "n_samples": n_samples,
        "n_spins": n_spins,
        "tau_mean": float(np.mean(tau_arr)),
        "tau_max": float(np.max(tau_arr)),
        "tau_min": float(np.min(tau_arr)),
        "ess_mean": float(np.mean(ess_arr)),
        "ess_min": float(np.min(ess_arr)),
        "per_spin_tau": tau_arr,
        "per_spin_ess": ess_arr,
    }


@dataclass
class SokalConvergenceObserver:
    """Pluggable telemetry observer for thermodynamic sampling engines.

    Attaches to thrml sampling routines, computes online or post-hoc MCMC mixing
    diagnostics, and quantifies true statistically independent sampling rates.
    """

    c_window: float = 5.0
    min_burn_in_ratio: float = 50.0  # Require N_samples >= min_burn_in_ratio * tau_int
    history: List[float] = field(default_factory=list)
    wall_clock_time: float = 0.0

    def record_step(self, value: float) -> None:
        """Record a single scalar step (e.g. instantaneous energy or order parameter)."""
        self.history.append(float(value))

    def record_trace(self, values: Union[List[float], np.ndarray], elapsed_sec: float = 0.0) -> None:
        """Record an entire trajectory and update execution wall-clock time."""
        self.history.extend([float(v) for v in values])
        self.wall_clock_time += elapsed_sec

    def evaluate(self) -> Dict[str, Any]:
        """Compute full convergence metrics on accumulated history."""
        series = np.asarray(self.history, dtype=np.float64)
        n = len(series)
        tau_int, ess = compute_sokal_tau_int(series, c_window=self.c_window)

        sample_rate_hz = (n / self.wall_clock_time) if self.wall_clock_time > 0 else 0.0
        ess_per_sec = (ess / self.wall_clock_time) if self.wall_clock_time > 0 else 0.0
        is_converged = (n >= self.min_burn_in_ratio * tau_int) if n > 0 else False

        return {
            "n_samples": n,
            "tau_int": tau_int,
            "ess": ess,
            "sample_rate_hz": sample_rate_hz,
            "ess_per_sec": ess_per_sec,
            "wall_clock_time": self.wall_clock_time,
            "is_converged": is_converged,
            "convergence_ratio": float(n / (self.min_burn_in_ratio * tau_int)) if tau_int > 0 else 0.0,
        }

    def summary(self) -> str:
        """Formatted human-readable summary of mixing and throughput."""
        m = self.evaluate()
        status = "CONVERGED" if m["is_converged"] else "INSUFFICIENT MIXING"
        return (
            f"[Sokal Observer] {status} | N={m['n_samples']} samples in {m['wall_clock_time']:.4f}s "
            f"({m['sample_rate_hz']:,.1f} Hz) | tau_int: {m['tau_int']:.2f} | "
            f"ESS: {m['ess']:.1f} ({m['ess_per_sec']:,.1f} ESS/s) | "
            f"Target {self.min_burn_in_ratio:.0f}x tau_int: {m['convergence_ratio']:.2f}x"
        )
