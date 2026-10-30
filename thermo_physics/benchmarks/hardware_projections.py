"""
thermo_physics.benchmarks.hardware_projections
==============================================

Analytical and empirical hardware projection model comparing physical
Thermodynamic Sampling Units (Extropic TSUs) against classical high-end GPUs.

Assumptions:
- Extropic Z1: Configurable chip specification (default: 269,568 p-bits, 8 cores, 50 MHz, 1.0W)
- Classical GPU Baseline: NVIDIA RTX 4090 / A100 (~350W TDP, JAX/CUDA Block Gibbs execution)
- True metric of merit: Energy consumption (Joules) per 1,000 statistically independent samples (ESS)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class HardwareSpec:
    """Configurable execution substrate specification."""
    name: str
    architecture: str
    total_spins: int
    clock_frequency_hz: float
    power_watts: float
    die_count: int = 1
    cores_per_die: int = 8
    dac_resolution_bits: int = 8
    notes: str = ""


@dataclass
class WorkloadProfile:
    """Problem instance characteristics defining MCMC mixing difficulty."""
    name: str
    num_variables: int
    is_frustrated: bool
    classical_tau_int: float   # Sokal autocorrelation time on GPU (sweeps)
    hardware_tau_int: float    # Projected autocorrelation on analog substrate (sweeps)
    samples_required: int = 100_000


@dataclass
class ProjectionResult:
    """Comparative projection metrics between digital GPU and thermodynamic TSU."""
    workload: str
    gpu_raw_hz: float
    tsu_raw_hz: float
    gpu_ess_per_sec: float
    tsu_ess_per_sec: float
    gpu_joules_per_1k_ess: float
    tsu_joules_per_1k_ess: float
    throughput_speedup: float
    energy_efficiency_multiplier: float
    assumptions: Dict[str, Any]

    def summary(self) -> str:
        return (
            f"=== Benchmark Projection: {self.workload} ===\n"
            f"Throughput (Independent ESS/s):\n"
            f"  - Classical GPU Baseline : {self.gpu_ess_per_sec:,.1f} ESS/s\n"
            f"  - Extropic TSU Substrate : {self.tsu_ess_per_sec:,.1f} ESS/s ({self.throughput_speedup:,.1f}x)\n"
            f"Energy per 1,000 Independent Samples:\n"
            f"  - Classical GPU Baseline : {self.gpu_joules_per_1k_ess:,.4f} Joules\n"
            f"  - Extropic TSU Substrate : {self.tsu_joules_per_1k_ess:,.6f} Joules\n"
            f"Net Energy Efficiency Gain: {self.energy_efficiency_multiplier:,.0f}x\n"
        )


def get_default_extropic_z1_spec(
    p_bits: int = 269_568,
    cores: int = 8,
    freq_mhz: float = 50.0,
    power_w: float = 1.0,
) -> HardwareSpec:
    """Factory creating parameterized Extropic Z1 substrate specification."""
    return HardwareSpec(
        name="Extropic Z1 Thermodynamic Sampling Unit",
        architecture="Analog Stochastic CMOS p-bit Substrate",
        total_spins=p_bits,
        clock_frequency_hz=freq_mhz * 1e6,
        power_watts=power_w,
        die_count=1,
        cores_per_die=cores,
        dac_resolution_bits=8,
        notes="Continuous Brownian thermal noise acceleration with subthreshold CMOS p-bits.",
    )


def get_default_gpu_spec(
    power_w: float = 350.0,
    measured_gibbs_hz: float = 50_000.0,
) -> HardwareSpec:
    """Factory creating parameterized NVIDIA high-end GPU baseline specification."""
    return HardwareSpec(
        name="NVIDIA High-End GPU (RTX 4090 / A100)",
        architecture="Digital SIMT Vectorized Tensor Cores",
        total_spins=16_384,  # Effective active register-mapped spins per streaming multiprocessor
        clock_frequency_hz=2.2e9,
        power_watts=power_w,
        die_count=1,
        cores_per_die=128,
        dac_resolution_bits=32,  # IEEE float32 arithmetic
        notes="Measured empirical JAX Block Gibbs sampler baseline at typical 350W TDP.",
    )


def evaluate_projection(
    workload: WorkloadProfile,
    tsu_spec: Optional[HardwareSpec] = None,
    gpu_spec: Optional[HardwareSpec] = None,
    gpu_measured_raw_hz: float = 42_000.0,
) -> ProjectionResult:
    """Compute throughput, latency, and Joules-per-independent-sample comparison."""
    tsu = tsu_spec or get_default_extropic_z1_spec()
    gpu = gpu_spec or get_default_gpu_spec()

    # Raw sampling throughput (sweeps/sec)
    # TSU executes continuous thermal relaxation at ~50 MHz update rate
    # For N-variable graph, raw full sweeps/sec = frequency / N (or 1 sweep per clock cycle in parallel cores)
    tsu_parallel_factor = min(1.0, float(tsu.total_spins) / workload.num_variables)
    tsu_raw_hz = (tsu.clock_frequency_hz / workload.num_variables) * tsu_parallel_factor
    tsu_raw_hz = min(tsu_raw_hz, 1_000_000.0)  # Conservative driver/bus bandwidth ceiling: 1 MSamples/s

    gpu_raw_hz = gpu_measured_raw_hz

    # Effective Sample Size per second: ESS/s = Raw_Hz / (2 * tau_int)
    gpu_ess_per_sec = gpu_raw_hz / (2.0 * workload.classical_tau_int)
    tsu_ess_per_sec = tsu_raw_hz / (2.0 * workload.hardware_tau_int)

    # Energy Consumption: Joules per 1,000 independent samples (ESS)
    # Power (Joules/sec) / (ESS/sec) = Joules per 1 ESS
    gpu_joules_per_ess = gpu.power_watts / (gpu_ess_per_sec + 1e-12)
    tsu_joules_per_ess = tsu.power_watts / (tsu_ess_per_sec + 1e-12)

    gpu_joules_per_1k = gpu_joules_per_ess * 1000.0
    tsu_joules_per_1k = tsu_joules_per_ess * 1000.0

    throughput_speedup = tsu_ess_per_sec / (gpu_ess_per_sec + 1e-12)
    energy_efficiency = gpu_joules_per_1k / (tsu_joules_per_1k + 1e-12)

    assumptions = {
        "tsu_power_w": tsu.power_watts,
        "gpu_power_w": gpu.power_watts,
        "classical_tau_int": workload.classical_tau_int,
        "hardware_tau_int": workload.hardware_tau_int,
        "gpu_measured_raw_hz": gpu_raw_hz,
        "tsu_effective_raw_hz": tsu_raw_hz,
        "is_frustrated_landscape": workload.is_frustrated,
    }

    return ProjectionResult(
        workload=workload.name,
        gpu_raw_hz=gpu_raw_hz,
        tsu_raw_hz=tsu_raw_hz,
        gpu_ess_per_sec=gpu_ess_per_sec,
        tsu_ess_per_sec=tsu_ess_per_sec,
        gpu_joules_per_1k_ess=gpu_joules_per_1k,
        tsu_joules_per_1k_ess=tsu_joules_per_1k,
        throughput_speedup=throughput_speedup,
        energy_efficiency_multiplier=energy_efficiency,
        assumptions=assumptions,
    )


def run_benchmark_projections():
    """Execute standard comparative projections across key benchmark stations."""
    print("=" * 80)
    print("THERMODYNAMIC HARDWARE PROJECTION SUITE: Extropic Z1 vs. NVIDIA High-End GPU")
    print("=" * 80)

    workloads = [
        WorkloadProfile(
            name="Station 10: Sherrington-Kirkpatrick Spin Glass (N=24)",
            num_variables=24,
            is_frustrated=True,
            classical_tau_int=54.61,  # Measured critical slowing down on GPU
            hardware_tau_int=0.50,    # Analog thermal fluctuations mix in O(1)
        ),
        WorkloadProfile(
            name="Station 11: RNA Aptamer Core Ensemble (N=50)",
            num_variables=50,
            is_frustrated=True,
            classical_tau_int=38.22,
            hardware_tau_int=0.65,
        ),
        WorkloadProfile(
            name="Station 12: Grid SCUC Resource Dispatch (N=24)",
            num_variables=24,
            is_frustrated=False,
            classical_tau_int=1.00,
            hardware_tau_int=0.50,
        ),
    ]

    for w in workloads:
        res = evaluate_projection(w)
        print(res.summary())


if __name__ == "__main__":
    run_benchmark_projections()
