"""
thermo_physics.benchmarks.energy_profiler
=========================================
Phase 3, Step 2: Silicon Realism Energy-per-Op & Time-to-Solution Profiler.

Calculates theoretical and empirical energy-per-op (Joules, femtojoules per synaptic op,
and Joules per 1,000 independent ESS) grounded in the physical silicon realism parameters:
1. K-bit Capacitive DAC Switching Energy:
   E_dac_op = 0.5 * C_unit * (2^K) * V_rail^2  (with 28nm CMOS unit capacitor C_unit ~ 0.5 fF)
2. Subthreshold Analog Transconductance & Op-Amp Slew Energy:
   E_opamp_step = I_bias * V_rail * dt_physical + E_clip_penalty * rail_clip_fraction
3. Classical GPU Digital Baseline (RTX 5080 / RTX 4090 at 320W-350W TDP) vs.
   Physical Thermodynamic Sampling Unit (Extropic TSU Z1 at ~1.0W).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from thermo_physics.backends.sampler_adapter import (
    BackendType,
    HamiltonianGraph,
    SamplingConfig,
    SamplingResult,
    get_sampler,
)


@dataclass
class EnergyProfileReport:
    """Comprehensive Energy-per-Op and Time-to-Solution benchmark profile."""
    workload_name: str
    num_nodes: int
    num_synaptic_edges: int
    dac_bits: int
    v_rail_volts: float
    total_synaptic_ops: int
    wall_clock_sec: float
    simulated_tsu_time_to_solution_us: float
    tau_int: float
    ess: float
    gpu_ess_per_sec: float
    tsu_ess_per_sec: float
    # Energy metrics
    gpu_total_joules: float
    tsu_total_joules: float
    gpu_energy_per_op_pj: float       # Picojoules per synaptic operation on GPU
    tsu_energy_per_op_fj: float       # Femtojoules per synaptic operation on TSU
    dac_switching_energy_fj: float    # Capacitive DAC contribution (fJ/op)
    opamp_static_slew_fj: float       # Op-amp + rail clipping dissipation (fJ/op)
    gpu_joules_per_1k_ess: float
    tsu_joules_per_1k_ess: float
    energy_efficiency_gain_x: float
    silicon_diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workload_name": self.workload_name,
            "num_nodes": self.num_nodes,
            "num_synaptic_edges": self.num_synaptic_edges,
            "dac_bits": self.dac_bits,
            "v_rail_volts": self.v_rail_volts,
            "total_synaptic_ops": self.total_synaptic_ops,
            "wall_clock_sec": round(self.wall_clock_sec, 5),
            "simulated_tsu_time_to_solution_us": round(self.simulated_tsu_time_to_solution_us, 3),
            "tau_int": round(self.tau_int, 3),
            "ess": round(self.ess, 2),
            "gpu_ess_per_sec": round(self.gpu_ess_per_sec, 1),
            "tsu_ess_per_sec": round(self.tsu_ess_per_sec, 1),
            "gpu_total_joules": round(self.gpu_total_joules, 5),
            "tsu_total_joules": round(self.tsu_total_joules, 8),
            "gpu_energy_per_op_pj": round(self.gpu_energy_per_op_pj, 3),
            "tsu_energy_per_op_fj": round(self.tsu_energy_per_op_fj, 3),
            "dac_switching_energy_fj": round(self.dac_switching_energy_fj, 3),
            "opamp_static_slew_fj": round(self.opamp_static_slew_fj, 3),
            "gpu_joules_per_1k_ess": round(self.gpu_joules_per_1k_ess, 5),
            "tsu_joules_per_1k_ess": round(self.tsu_joules_per_1k_ess, 7),
            "energy_efficiency_gain_x": round(self.energy_efficiency_gain_x, 1),
            "silicon_diagnostics": self.silicon_diagnostics,
        }

    def summary(self) -> str:
        return (
            f"[{self.workload_name}] (N={self.num_nodes}, {self.dac_bits}-bit DAC, +/-{self.v_rail_volts}V rails)\n"
            f"  Time-to-Solution : GPU {self.wall_clock_sec * 1e3:.2f} ms | Physical TSU {self.simulated_tsu_time_to_solution_us:.2f} us\n"
            f"  Energy-per-Op    : GPU {self.gpu_energy_per_op_pj:,.2f} pJ/op | TSU {self.tsu_energy_per_op_fj:,.2f} fJ/op "
            f"(DAC: {self.dac_switching_energy_fj:.2f} fJ + OpAmp/Clip: {self.opamp_static_slew_fj:.2f} fJ)\n"
            f"  Joules / 1k ESS  : GPU {self.gpu_joules_per_1k_ess:.4f} J | TSU {self.tsu_joules_per_1k_ess:.6f} J "
            f"-> {self.energy_efficiency_gain_x:,.1f}x Efficiency Gain"
        )


class ThermodynamicEnergyProfiler:
    """Profiles time-to-solution and physical Joules-per-Op across silicon realism settings."""

    def __init__(
        self,
        gpu_tdp_watts: float = 320.0,       # NVIDIA RTX 5080 typical load power
        tsu_clock_hz: float = 50.0e6,       # 50 MHz analog relaxation bandwidth
        c_unit_femtofarads: float = 0.45,   # 28nm CMOS unit DAC capacitor (fF)
        subthreshold_bias_na: float = 120.0,  # Subthreshold p-bit / transconductance bias current (nA)
    ):
        self.gpu_tdp_watts = gpu_tdp_watts
        self.tsu_clock_hz = tsu_clock_hz
        self.c_unit_f = c_unit_femtofarads * 1e-15
        self.i_bias_a = subthreshold_bias_na * 1e-9

    def profile_sampling_run(
        self,
        workload_name: str,
        graph: HamiltonianGraph,
        config: SamplingConfig,
        dac_bits: int = 6,
        v_rail: float = 1.5,
        tau_color: float = 0.05,
        pink_fraction: float = 0.5,
    ) -> EnergyProfileReport:
        """Execute a silicon-realistic sampling run and compute full Energy-per-Op metrics."""
        cfg = SamplingConfig(
            n_samples=config.n_samples,
            burn_in=config.burn_in,
            stride=config.stride,
            beta=config.beta,
            seed=config.seed,
            metadata={
                **config.metadata,
                "dac_bits": dac_bits,
                "v_rail": v_rail,
                "tau_color": tau_color,
                "pink_fraction": pink_fraction,
            },
        )
        sampler = get_sampler(
            BackendType.SIMULATED_NOISE,
            dac_bits=dac_bits,
            v_rail=v_rail,
            tau_color=tau_color,
            pink_fraction=pink_fraction,
        )
        res = sampler.sample(graph, cfg)

        n_nodes = graph.num_nodes
        n_edges = int(np.count_nonzero(graph.couplings)) // 2
        if n_edges == 0:
            n_edges = n_nodes

        total_steps = cfg.burn_in + (cfg.n_samples * cfg.stride)
        # Each step evaluates 2 * n_edges synaptic MACs + n_nodes bias updates
        total_synaptic_ops = max(1, total_steps * (2 * n_edges + n_nodes))

        # 1. Classical GPU energy (RTX 5080 executing JAX scan)
        gpu_total_joules = self.gpu_tdp_watts * res.wall_clock_time
        gpu_energy_per_op_j = gpu_total_joules / float(total_synaptic_ops)
        gpu_energy_per_op_pj = gpu_energy_per_op_j * 1e12  # picojoules

        # 2. Physical Silicon TSU Energy Model grounded in DAC bits, V_rail, and Rail Clipping
        effective_bits = min(12, max(1, dac_bits))
        # Capacitive DAC switching energy per synaptic transition (activity factor alpha = 0.15)
        e_dac_j = 0.15 * 0.5 * self.c_unit_f * float(1 << effective_bits) * (v_rail ** 2)

        # Op-Amp subthreshold bias + rail-clipping housekeeping heat penalty
        dt_phys = 1.0 / self.tsu_clock_hz  # 20 ns per physical relaxation cycle
        clip_frac = float(res.diagnostics.get("rail_clip_fraction", 0.0))
        e_opamp_static_j = self.i_bias_a * v_rail * dt_phys
        # Saturated rail events dissipate extra slewing charge
        e_clip_penalty_j = e_opamp_static_j * (2.5 * clip_frac)
        e_opamp_total_j = e_opamp_static_j + e_clip_penalty_j

        tsu_energy_per_op_j = e_dac_j + e_opamp_total_j
        tsu_energy_per_op_fj = tsu_energy_per_op_j * 1e15  # femtojoules
        tsu_total_joules = tsu_energy_per_op_j * float(total_synaptic_ops)

        # Simulated physical TSU time-to-solution (parallel analog relaxation across all edges)
        tsu_tts_sec = float(total_steps) * dt_phys
        tsu_tts_us = tsu_tts_sec * 1e6

        gpu_ess_per_sec = res.ess_per_sec
        tsu_ess_per_sec = res.ess / max(1e-12, tsu_tts_sec)

        gpu_j_per_1k_ess = (gpu_total_joules / max(1e-9, res.ess)) * 1000.0
        tsu_j_per_1k_ess = (tsu_total_joules / max(1e-9, res.ess)) * 1000.0
        efficiency_gain = gpu_j_per_1k_ess / max(1e-15, tsu_j_per_1k_ess)

        return EnergyProfileReport(
            workload_name=workload_name,
            num_nodes=n_nodes,
            num_synaptic_edges=n_edges,
            dac_bits=dac_bits,
            v_rail_volts=v_rail,
            total_synaptic_ops=total_synaptic_ops,
            wall_clock_sec=res.wall_clock_time,
            simulated_tsu_time_to_solution_us=tsu_tts_us,
            tau_int=res.tau_int,
            ess=res.ess,
            gpu_ess_per_sec=gpu_ess_per_sec,
            tsu_ess_per_sec=tsu_ess_per_sec,
            gpu_total_joules=gpu_total_joules,
            tsu_total_joules=tsu_total_joules,
            gpu_energy_per_op_pj=gpu_energy_per_op_pj,
            tsu_energy_per_op_fj=tsu_energy_per_op_fj,
            dac_switching_energy_fj=e_dac_j * 1e15,
            opamp_static_slew_fj=e_opamp_total_j * 1e15,
            gpu_joules_per_1k_ess=gpu_j_per_1k_ess,
            tsu_joules_per_1k_ess=tsu_j_per_1k_ess,
            energy_efficiency_gain_x=efficiency_gain,
            silicon_diagnostics=res.diagnostics,
        )

    def run_precision_energy_sweep(
        self,
        graph: HamiltonianGraph,
        config: Optional[SamplingConfig] = None,
        bit_depths: List[int] = [8, 6, 4],
        v_rail: float = 1.5,
    ) -> List[EnergyProfileReport]:
        """Sweep across 8-bit, 6-bit, and 4-bit DAC configurations to quantify Joule savings."""
        cfg = config or SamplingConfig(n_samples=300, burn_in=60, beta=1.2)
        reports = []
        for bits in bit_depths:
            rep = self.profile_sampling_run(
                workload_name=f"{bits}-Bit Silicon Crossbar",
                graph=graph,
                config=cfg,
                dac_bits=bits,
                v_rail=v_rail,
            )
            reports.append(rep)
        return reports
