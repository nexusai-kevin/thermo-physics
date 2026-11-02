"""
Experiment 18: Phase-4 Silicon Realism & Hardware Imperfections Benchmark
========================================================================
Stress-tests the Non-Conservative Auxiliary Hybrid TPU solver against real-world physical
analog hardware constraints:
1. Finite-Resolution DAC Quantization:
   Discretizes couplings J_ij, biases h_i, and dynamic clause weights y_m(t) into 4-bit, 6-bit,
   and 8-bit fixed-point representations. Measures degradation in 3-SAT satisfaction at N=100.
2. Operational Amplifier Voltage Rail Clipping:
   Enforces physical saturation boundaries on oscillator voltages (x in [-V_rail, V_rail]),
   velocity slew rates (v in [-V_max, V_max]), and dynamic multiplier ceilings (y_m <= Y_max).
   Tracks impact on non-conservative housekeeping heat Q_hk.
3. Colored 1/f Flicker Noise Bath (Ornstein-Uhlenbeck Process):
   Replaces white Wiener noise increments with colored Ornstein-Uhlenbeck noise with finite
   autocorrelation time tau_color, evaluating symplectic stability and ergodicity.
4. Full Realistic Silicon Stack:
   Combines 6-bit/4-bit DAC + Op-Amp voltage rails + 1/f colored noise to test complete
   end-to-end hardware resilience on phase-transition 3-SAT.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console(legacy_windows=False)

from thermo_physics.tpu import (
    NonConvexIsingSAT,
    ProblemConfig,
    generate_random_3sat_spin_glass,
    HybridTPUSolver,
    TPUSolverConfig,
    SiliconRealismConfig,
    DACQuantizer,
    OpAmpVoltageClipper,
    OrnsteinUhlenbeckNoise,
)
from thermo_physics.tpu.integrator import ThermalSchedule


def _convert_numpy(obj: Any) -> Any:
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_numpy(v) for v in obj]
    return obj


def create_benchmark_instance(
    num_variables: int = 100,
    clause_ratio: float = 4.0,
    seed: int = 42,
) -> NonConvexIsingSAT:
    """Generate canonical N=100 3-SAT + spin glass problem at the satisfiability threshold."""
    rng = np.random.default_rng(seed)
    m = int(round(clause_ratio * num_variables))

    clauses = []
    for _ in range(m):
        vars_chosen = rng.choice(num_variables, size=3, replace=False) + 1
        polarities = rng.choice([-1, 1], size=3)
        clauses.append((
            int(polarities[0] * vars_chosen[0]),
            int(polarities[1] * vars_chosen[1]),
            int(polarities[2] * vars_chosen[2]),
        ))

    j_raw = rng.normal(0.0, 0.2 / np.sqrt(num_variables), size=(num_variables, num_variables))
    j_mat = 0.5 * (j_raw + j_raw.T)
    np.fill_diagonal(j_mat, 0.0)
    h_bias = rng.normal(0.0, 0.05, size=num_variables)

    config = ProblemConfig(
        num_variables=num_variables,
        alpha_well=1.2,
        lambda_sat=5.0,
        c_sat=2.5,
        penalty_type="softplus",
        beta_softplus=2.5,
    )
    return NonConvexIsingSAT(num_variables, j_matrix=j_mat, h_bias=h_bias, clauses=clauses, config=config)


def get_solver_config(
    steps: int,
    num_trajectories: int,
    silicon_config: Optional[SiliconRealismConfig] = None,
    seed: int = 42,
) -> TPUSolverConfig:
    """Canonical Phase-3/Phase-4 solver configuration with non-conservative auxiliary memory."""
    return TPUSolverConfig(
        num_trajectories=num_trajectories,
        total_steps=steps,
        dt=0.015,
        schedule=ThermalSchedule(
            t_start=2.5,
            t_end=0.02,
            total_steps=steps,
            alpha_start=0.3,
            alpha_end=3.0,
            lambda_sat_start=1.0,
            lambda_sat_end=7.0,
        ),
        use_non_conservative_aux=True,
        kappa_aux=35.0,
        eta_aux=1.0,
        lambda_0_aux=1.0,
        silicon_config=silicon_config,
        seed=seed,
    )


def run_dac_quantization_sweep(
    hamiltonian: NonConvexIsingSAT,
    num_trajectories: int = 32,
    steps: int = 2000,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Benchmark 1: Precision degradation under 32-bit, 8-bit, 6-bit, and 4-bit DAC quantization."""
    console.print(f"\n[bold yellow]═══ Benchmark 1: Finite-Resolution DAC Quantization Sweep (N={hamiltonian.n}) ═══[/bold yellow]")
    console.print("[dim]Evaluating coupling J & field h fixed-point discretization down to 4-bit CMOS crossbars...[/dim]")

    dac_configs = [
        ("32-bit Float (Ideal)", None),
        ("8-bit DAC (255 levels)", 8),
        ("6-bit DAC (63 levels)", 6),
        ("4-bit DAC (15 levels)", 4),
    ]

    results = []
    table = Table(title="DAC Quantization Precision vs SAT Satisfaction", header_style="bold cyan", border_style="bright_blue")
    table.add_column("DAC Precision", style="cyan")
    table.add_column("Discrete Levels", justify="center")
    table.add_column("RMS Quant Error (J)", justify="center")
    table.add_column("Best Violated", justify="center")
    table.add_column("Best Energy", justify="right")
    table.add_column("Wall Time", justify="right")
    table.add_column("Status", justify="center")

    for label, bits in dac_configs:
        sil_cfg = SiliconRealismConfig(
            dac_bits=bits,
            quantize_hamiltonian=True,
            quantize_dynamic_multipliers=True if bits is not None else False,
        )
        solver_cfg = get_solver_config(
            steps=steps,
            num_trajectories=num_trajectories,
            silicon_config=sil_cfg,
            seed=seed,
        )
        solver = HybridTPUSolver(solver_cfg)
        res = solver.solve(hamiltonian)

        levels_str = f"2^{bits} - 1 = {(1 << bits) - 1}" if bits else "Continuous"
        rms_err = res.silicon_metrics["j_quantization_rms_error"] if res.silicon_metrics else 0.0

        if res.best_violated_clauses == 0:
            status_str = "[bold green]100% SAT (0 Violated)[/bold green]"
        elif res.best_violated_clauses <= 2:
            status_str = f"[bold green]Robust ({res.best_violated_clauses} Violated)[/bold green]"
        else:
            status_str = f"[yellow]Degraded ({res.best_violated_clauses} Violated)[/yellow]"

        table.add_row(
            label,
            levels_str,
            f"{rms_err * 100:.2f}%" if bits else "0.00%",
            str(res.best_violated_clauses),
            f"{res.best_energy:.2f}",
            f"{res.wall_time_sec:.2f}s",
            status_str,
        )

        results.append({
            "label": label,
            "dac_bits": bits,
            "levels": (1 << bits) - 1 if bits else 0,
            "rms_error": rms_err,
            "best_violated_clauses": res.best_violated_clauses,
            "best_energy": res.best_energy,
            "wall_time_sec": res.wall_time_sec,
            "success": res.success,
        })

    console.print(table)
    return results


def run_opamp_rail_clipping_benchmark(
    hamiltonian: NonConvexIsingSAT,
    num_trajectories: int = 32,
    steps: int = 2000,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Benchmark 2: Operational Amplifier Voltage Rail Clipping & Slew Saturation."""
    console.print(f"\n[bold yellow]═══ Benchmark 2: Op-Amp Voltage Rail Saturation (+/- 1.5V) ═══[/bold yellow]")
    console.print("[dim]Evaluating supply rail bounds on position voltages, slew rates, and multiplier capacitor ceilings...[/dim]")

    configs = [
        ("Unbounded Rails (Ideal)", False, 1.5, 10.0, 15.0),
        ("Op-Amp Rails (+/-1.5V, Slew=10V/s)", True, 1.5, 10.0, 15.0),
        ("Stiff Rails (+/-1.2V, Slew=5V/s)", True, 1.2, 5.0, 10.0),
    ]

    results = []
    table = Table(title="Op-Amp Rail Saturation vs Housekeeping Work Q_hk", header_style="bold cyan", border_style="bright_blue")
    table.add_column("Rail Setting", style="cyan")
    table.add_column("V_rail Bounds", justify="center")
    table.add_column("Voltage Clips", justify="center")
    table.add_column("Multiplier Clips", justify="center")
    table.add_column("Q_hk Injected", justify="right")
    table.add_column("Best Violated", justify="center")
    table.add_column("Status", justify="center")

    for label, enable_clip, v_rail, v_max, y_max in configs:
        sil_cfg = SiliconRealismConfig(
            enable_rail_clipping=enable_clip,
            v_rail=v_rail,
            v_max=v_max,
            y_max=y_max,
        )
        solver_cfg = get_solver_config(
            steps=steps,
            num_trajectories=num_trajectories,
            silicon_config=sil_cfg,
            seed=seed,
        )
        solver = HybridTPUSolver(solver_cfg)
        res = solver.solve(hamiltonian)

        q_hk = float(np.mean(res.thermo_trajectory.housekeeping_heat)) if (res.thermo_trajectory and len(res.thermo_trajectory.housekeeping_heat) > 0) else 0.0
        v_clips = res.silicon_metrics["position_clips"] if res.silicon_metrics else 0
        y_clips = res.silicon_metrics["multiplier_clips"] if res.silicon_metrics else 0

        status_str = "[bold green]SOLVED (0 Violated)[/bold green]" if res.best_violated_clauses == 0 else f"[yellow]{res.best_violated_clauses} Violated[/yellow]"

        table.add_row(
            label,
            f"[-{v_rail}V, +{v_rail}V]" if enable_clip else "Unbounded",
            f"{v_clips:,}" if enable_clip else "0",
            f"{y_clips:,}" if enable_clip else "0",
            f"{q_hk:.2f} J",
            str(res.best_violated_clauses),
            status_str,
        )

        results.append({
            "label": label,
            "enable_rail_clipping": enable_clip,
            "v_rail": v_rail,
            "v_clips": v_clips,
            "y_clips": y_clips,
            "housekeeping_heat_j": q_hk,
            "best_violated_clauses": res.best_violated_clauses,
            "best_energy": res.best_energy,
        })

    console.print(table)
    return results


def run_colored_noise_benchmark(
    hamiltonian: NonConvexIsingSAT,
    num_trajectories: int = 32,
    steps: int = 2000,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Benchmark 3: Colored 1/f Flicker Noise Bath (Ornstein-Uhlenbeck)."""
    console.print(f"\n[bold yellow]═══ Benchmark 3: Colored 1/f Flicker Noise Bath (Ornstein-Uhlenbeck) ═══[/bold yellow]")
    console.print("[dim]Evaluating symplectic BAOAB stability under finite correlation times tau_color...[/dim]")

    configs = [
        ("White Wiener Bath (Ideal)", False, 0.0, False),
        ("Colored OU Bath (tau=0.05)", True, 0.05, False),
        ("Colored OU Bath (tau=0.20, Stiff)", True, 0.20, False),
        ("3-Pole Pink Noise (Multi-Decade 1/f)", True, 0.05, True),
    ]

    results = []
    table = Table(title="Thermal Noise Color vs Symplectic Stability", header_style="bold cyan", border_style="bright_blue")
    table.add_column("Noise Model", style="cyan")
    table.add_column("Correlation Time (tau)", justify="center")
    table.add_column("Mean Dissipated Heat", justify="right")
    table.add_column("Best Violated", justify="center")
    table.add_column("Status", justify="center")

    for label, enable_color, tau, pink_bank in configs:
        sil_cfg = SiliconRealismConfig(
            enable_colored_noise=enable_color,
            tau_color=tau,
            use_pink_noise_bank=pink_bank,
        )
        solver_cfg = get_solver_config(
            steps=steps,
            num_trajectories=num_trajectories,
            silicon_config=sil_cfg,
            seed=seed,
        )
        solver = HybridTPUSolver(solver_cfg)
        res = solver.solve(hamiltonian)

        w_diss = (
            float(res.jarzynski_analysis.mean_dissipated_work)
            if (res.jarzynski_analysis and hasattr(res.jarzynski_analysis, "mean_dissipated_work"))
            else (float(np.mean(res.thermo_trajectory.heat_dissipated)) if res.thermo_trajectory else 0.0)
        )
        status_str = "[bold green]STABLE (0 Violated)[/bold green]" if res.best_violated_clauses == 0 else f"[yellow]STABLE ({res.best_violated_clauses} Violated)[/yellow]"

        table.add_row(
            label,
            f"{tau:.2f}s" if enable_color else "0.00s (Delta)",
            f"{w_diss:.2f} J",
            str(res.best_violated_clauses),
            status_str,
        )

        results.append({
            "label": label,
            "enable_colored_noise": enable_color,
            "tau_color": tau,
            "use_pink_noise_bank": pink_bank,
            "mean_dissipated_heat_j": w_diss,
            "best_violated_clauses": res.best_violated_clauses,
            "best_energy": res.best_energy,
        })

    console.print(table)
    return results


def run_full_silicon_stack_benchmark(
    hamiltonian: NonConvexIsingSAT,
    num_trajectories: int = 32,
    steps: int = 2000,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Benchmark 4: Combined Full Silicon Realism Stack (DAC + Rails + Colored Noise)."""
    console.print(f"\n[bold yellow]═══ Benchmark 4: Combined Silicon Realism Stack (DAC + Op-Amp + 1/f Noise) ═══[/bold yellow]")
    console.print("[dim]Subjecting the solver to concurrent 6-bit/4-bit quantization, +/-1.5V clipping, and pink noise...[/dim]")

    configs = [
        ("Ideal Floating Point Baseline", None, False, False),
        ("Realistic 8-bit Silicon Stack", 8, True, True),
        ("Realistic 6-bit Silicon Stack", 6, True, True),
        ("Aggressive 4-bit Silicon Stack", 4, True, True),
    ]

    results = []
    table = Table(title="Full Silicon Realism Stack Performance at N=100", header_style="bold cyan", border_style="bright_blue")
    table.add_column("Silicon Configuration", style="cyan")
    table.add_column("DAC Bits", justify="center")
    table.add_column("Op-Amp Rails", justify="center")
    table.add_column("1/f Noise", justify="center")
    table.add_column("Best Violated", justify="center")
    table.add_column("Best Energy", justify="right")
    table.add_column("Silicon Resilience", justify="center")

    for label, bits, rails, noise in configs:
        sil_cfg = SiliconRealismConfig(
            dac_bits=bits,
            quantize_hamiltonian=True,
            quantize_dynamic_multipliers=True if bits else False,
            enable_rail_clipping=rails,
            v_rail=1.5,
            v_max=10.0,
            y_max=15.0,
            enable_colored_noise=noise,
            tau_color=0.05,
            use_pink_noise_bank=noise,
        )
        solver_cfg = get_solver_config(
            steps=steps,
            num_trajectories=num_trajectories,
            silicon_config=sil_cfg,
            seed=seed,
        )
        solver = HybridTPUSolver(solver_cfg)
        res = solver.solve(hamiltonian)

        if res.best_violated_clauses == 0:
            resil_str = "[bold green]HIGH RESILIENCE (100% SAT)[/bold green]"
        elif res.best_violated_clauses <= 2:
            resil_str = f"[bold green]ROBUST ({res.best_violated_clauses} Violated)[/bold green]"
        else:
            resil_str = f"[yellow]MARGINAL ({res.best_violated_clauses} Violated)[/yellow]"

        table.add_row(
            label,
            f"{bits}-bit" if bits else "Float32",
            "+/-1.5V" if rails else "None",
            "Pink 1/f" if noise else "White",
            str(res.best_violated_clauses),
            f"{res.best_energy:.2f}",
            resil_str,
        )

        results.append({
            "label": label,
            "dac_bits": bits,
            "rail_clipping": rails,
            "colored_noise": noise,
            "best_violated_clauses": res.best_violated_clauses,
            "best_energy": res.best_energy,
            "silicon_metrics": res.silicon_metrics,
        })

    console.print(table)
    return results


def run_phase_4_silicon_realism(output_dir: Path = Path("data")) -> Dict[str, Any]:
    """Execute complete Phase-4 Silicon Realism & Hardware Imperfections stress-test."""
    t_start = time.perf_counter()
    console.print(Panel.fit(
        "[bold cyan]Phase-4 Silicon Realism & Hardware Imperfection Module[/bold cyan]\n"
        "[white]4-bit/6-bit DAC Quantization | Op-Amp Rail Saturation (+/-1.5V) | Colored 1/f Flicker Noise[/white]",
        border_style="cyan"
    ))

    ham = create_benchmark_instance(num_variables=100, clause_ratio=4.0, seed=42)

    bench1 = run_dac_quantization_sweep(ham, num_trajectories=32, steps=2000, seed=42)
    bench2 = run_opamp_rail_clipping_benchmark(ham, num_trajectories=32, steps=2000, seed=42)
    bench3 = run_colored_noise_benchmark(ham, num_trajectories=32, steps=2000, seed=42)
    bench4 = run_full_silicon_stack_benchmark(ham, num_trajectories=32, steps=2000, seed=42)

    total_wall = time.perf_counter() - t_start

    report_data = {
        "num_variables": ham.n,
        "num_clauses": len(ham.clauses),
        "total_wall_time_sec": total_wall,
        "benchmark_1_dac_quantization": bench1,
        "benchmark_2_opamp_rails": bench2,
        "benchmark_3_colored_noise": bench3,
        "benchmark_4_full_silicon_stack": bench4,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "phase4_silicon_realism_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(_convert_numpy(report_data), f, indent=2)

    console.print(f"\n[bold green]✓ Phase-4 Silicon Realism Report exported to {report_path} (Total Wall: {total_wall:.2f}s)[/bold green]\n")
    return report_data


if __name__ == "__main__":
    run_phase_4_silicon_realism()
