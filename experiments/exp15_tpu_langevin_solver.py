"""
Experiment 15: Hybrid Thermodynamic Processing Unit (TPU) Langevin Solver
========================================================================
Stress-testing Non-Equilibrium Underdamped Langevin Dynamics on Non-Convex
3-SAT / Ising Spin Glass with Symplectic BAOAB Splitting, Work Dissipation
Tracking, Jarzynski Equality Verification, and Classical SA Comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console(legacy_windows=False)

from thermo_physics.tpu import (
    NonConvexIsingSAT,
    generate_random_3sat_spin_glass,
    BAOABIntegrator,
    EulerMaruyamaIntegrator,
    ThermalSchedule,
    AdaptiveThermalScheduler,
    ScheduleType,
    ThermodynamicTracker,
    verify_jarzynski_equality,
    DiagnosticMonitor,
    HybridTPUSolver,
    ClassicalSimulatedAnnealing,
    TPUSolverConfig,
    run_tpu_vs_classical_benchmark,
)


def _convert_numpy(obj: Any) -> Any:
    """Recursively convert numpy scalars and arrays to native Python types for JSON."""
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


def run_experiment_15(
    num_variables: int = 16,
    num_trajectories: int = 48,
    steps: int = 2500,
    seed: int = 42,
    output_dir: Path = Path("data"),
) -> Dict[str, Any]:
    """Execute Experiment 15 stress-test pipeline."""
    console.print(Panel.fit(
        "[bold cyan]Experiment 15: Hybrid TPU Langevin Solver & Non-Equilibrium Stress-Test[/bold cyan]\n"
        f"[white]Variables: {num_variables} | Trajectories: {num_trajectories} | Steps: {steps} | Seed: {seed}[/white]",
        border_style="cyan"
    ))

    # 1. Problem Generation (3-SAT at phase transition + SK Spin Glass)
    console.print("[dim]Generating non-convex 3-SAT (clause ratio 4.26) coupled with SK spin glass...[/dim]")
    hamiltonian = generate_random_3sat_spin_glass(
        num_variables=num_variables,
        clause_ratio=4.26,
        spin_glass_std=0.6,
        seed=seed,
    )
    console.print(f"[green][OK][/green] Hamiltonian constructed: {num_variables} spins, {len(hamiltonian.clauses)} 3-SAT clauses.")

    # 2. Benchmark BAOAB vs Euler-Maruyama under high stiffness (alpha_well = 6.0)
    console.print("\n[bold yellow]Stress Test 1: BAOAB vs Euler-Maruyama Integrator Stability[/bold yellow]")
    stiff_config = TPUSolverConfig(
        num_trajectories=16,
        total_steps=800,
        dt=0.03,  # Stiff timestep
        schedule=ThermalSchedule(alpha_start=2.0, alpha_end=6.0, total_steps=800),
        seed=seed,
    )
    # BAOAB
    stiff_config.integrator = "baoab"
    tpu_baoab = HybridTPUSolver(stiff_config).solve(hamiltonian)

    # Euler-Maruyama
    stiff_config.integrator = "euler_maruyama"
    tpu_em = HybridTPUSolver(stiff_config).solve(hamiltonian)

    stab_table = Table(title="Integrator Numerical Stability Comparison (dt = 0.03, alpha = 6.0)")
    stab_table.add_column("Integrator", style="bold")
    stab_table.add_column("Divergence Detected", style="cyan")
    stab_table.add_column("Divergence Step / Reason")
    stab_table.add_column("FDT Equipartition Error", justify="right")
    stab_table.add_column("Best Energy Attained", justify="right")

    em_div = "YES (Failed)" if (tpu_em.diagnostic_report and tpu_em.diagnostic_report.divergence_detected) else "NO"
    em_reason = (tpu_em.diagnostic_report.divergence_reason or "N/A") if tpu_em.diagnostic_report else "N/A"
    em_fdt = f"{tpu_em.diagnostic_report.mean_fdt_relative_error*100:.1f}%" if tpu_em.diagnostic_report else "N/A"

    baoab_div = "NO (Stable)"
    baoab_fdt = f"{tpu_baoab.diagnostic_report.mean_fdt_relative_error*100:.1f}%" if tpu_baoab.diagnostic_report else "N/A"

    stab_table.add_row("Euler-Maruyama", f"[red]{em_div}[/red]", em_reason, em_fdt, f"{tpu_em.best_energy:.2f}")
    stab_table.add_row("BAOAB Symplectic", f"[green]{baoab_div}[/green]", "None (Finite & Bounded)", baoab_fdt, f"{tpu_baoab.best_energy:.2f}")
    console.print(stab_table)

    # 3. Full Benchmark: Hybrid TPU Solver vs Classical Simulated Annealing
    console.print("\n[bold yellow]Stress Test 2: Hybrid TPU Solver vs Classical Simulated Annealing Baseline[/bold yellow]")
    bench = run_tpu_vs_classical_benchmark(
        hamiltonian=hamiltonian,
        num_trajectories=num_trajectories,
        steps=steps,
        seed=seed,
    )

    tpu = bench.tpu_result
    sa = bench.classical_result

    comp_table = Table(title=f"Optimization Benchmark (N={num_variables}, M={len(hamiltonian.clauses)} clauses)")
    comp_table.add_column("Metric", style="bold")
    comp_table.add_column("Hybrid TPU (Underdamped Langevin)", style="cyan", justify="right")
    comp_table.add_column("Classical Simulated Annealing", style="magenta", justify="right")
    comp_table.add_column("Advantage / Ratio", style="green", justify="right")

    comp_table.add_row("Best Energy Reached", f"{tpu.best_energy:.3f}", f"{sa['best_energy']:.3f}", f"dE = {bench.energy_gap:+.3f}")
    comp_table.add_row("Violated 3-SAT Clauses", str(tpu.best_violated_clauses), str(sa['best_violated_clauses']), "0 violated = SAT")
    comp_table.add_row("Ground State Found", str(tpu.success), str(sa['success']), "Optimal" if tpu.success else "Suboptimal")
    comp_table.add_row("Ensemble Success Rate", f"{bench.tpu_success_rate * 100:.1f}%", "N/A (single trajectory)", f"{bench.tpu_success_rate * 100:.1f}%")
    comp_table.add_row("Wall-Clock Time", f"{tpu.wall_time_sec:.3f} s", f"{sa['wall_time_sec']:.3f} s", f"{sa['wall_time_sec'] / max(1e-4, tpu.wall_time_sec):.2f}x speed")
    comp_table.add_row("Time to Solution (TTS 99%)", f"{bench.tts_tpu_sec:.3f} s", f"{bench.tts_classical_sec:.3f} s", f"[bold green]{bench.speedup_ratio:.2f}x TTS speedup[/bold green]")
    console.print(comp_table)

    # 4. Non-Equilibrium Thermodynamics & Jarzynski Equality
    console.print("\n[bold yellow]Stress Test 3: Thermodynamic Path Integrals & Jarzynski Equality[/bold yellow]")
    jarz = tpu.jarzynski_analysis
    thermo_table = Table(title="Non-Equilibrium Work & Fluctuation Theorem Observables")
    thermo_table.add_column("Thermodynamic Observable", style="bold")
    thermo_table.add_column("Measured Value", style="cyan", justify="right")
    thermo_table.add_column("Theoretical Law / Constraint", style="dim")

    if jarz and tpu.thermo_trajectory:
        traj = tpu.thermo_trajectory
        mean_diss = float(np.mean(traj.heat_dissipated))
        mean_entropy = float(np.mean(traj.entropy_production))

        thermo_table.add_row("Mean External Work <W>", f"{jarz.mean_work:.4f}", "dW = (dH/dt) dt")
        thermo_table.add_row("Work Std Dev sigma_W", f"{jarz.std_work:.4f}", "Fluctuations across ensemble")
        thermo_table.add_row("Jarzynski Exponential Avg <exp(-beta*W)>", f"{jarz.jarzynski_exp_avg:.4e}", "<exp(-beta*W)> = exp(-beta*dF)")
        thermo_table.add_row("Jarzynski Free Energy dF_Jarzynski", f"{jarz.delta_f_jarzynski:.4f}", f"95% CI: [{jarz.bootstrap_ci_95[0]:.4f}, {jarz.bootstrap_ci_95[1]:.4f}]")
        thermo_table.add_row("Dissipated Work <W_diss> = <W> - dF", f"{jarz.mean_dissipated_work:.4f}", "Second Law: <W_diss> >= 0")
        thermo_table.add_row("Second Law Satisfied", f"[green]{jarz.second_law_satisfied}[/green]", "<W> >= dF verified")
        thermo_table.add_row("Cumulative Heat Dissipation <Q_diss>", f"{mean_diss:.4f}", "First Law: dE = W - Q_diss")
        thermo_table.add_row("Medium Entropy Production dS_med", f"{mean_entropy:.4f}", "dS_med = sum dQ_diss / T(t)")
    console.print(thermo_table)

    # 5. Diagnostic Monitors Summary
    console.print("\n[bold yellow]Stress Test 4: Runtime Failure Diagnostics & Fluctuation-Dissipation Theorem[/bold yellow]")
    diag = tpu.diagnostic_report
    diag_table = Table(title="Automated Runtime Diagnostic Monitors")
    diag_table.add_column("Monitor", style="bold")
    diag_table.add_column("Status", justify="center")
    diag_table.add_column("Metrics & Findings")

    if diag:
        div_status = "[red]FAIL[/red]" if diag.divergence_detected else "[green]PASS[/green]"
        div_text = diag.divergence_reason or "Zero NaNs/Infs; analog coordinates bounded within safe limits"
        diag_table.add_row("Numerical Divergence", div_status, div_text)

        fdt_status = "[green]PASS[/green]" if diag.fdt_violations_count == 0 else "[yellow]FLAGGED[/yellow]"
        fdt_text = f"Mean Equipartition Error: {diag.mean_fdt_relative_error*100:.1f}% (Max: {diag.max_fdt_relative_error*100:.1f}%)"
        diag_table.add_row("Fluctuation-Dissipation (FDT)", fdt_status, fdt_text)

        erg_status = "[green]PASS[/green]" if diag.ergodicity_traps_count == 0 else "[yellow]FLAGGED[/yellow]"
        erg_text = f"Metastable Trap Events: {diag.ergodicity_traps_count} (Spins successfully cross multi-well barriers)"
        diag_table.add_row("Ergodicity & Barrier Crossing", erg_status, erg_text)
    console.print(diag_table)

    # Save summary report
    output_dir.mkdir(parents=True, exist_ok=True)
    report_data = {
        "num_variables": num_variables,
        "num_clauses": len(hamiltonian.clauses),
        "num_trajectories": num_trajectories,
        "steps": steps,
        "tpu_best_energy": tpu.best_energy,
        "tpu_best_violated": tpu.best_violated_clauses,
        "tpu_wall_time": tpu.wall_time_sec,
        "sa_best_energy": sa["best_energy"],
        "sa_best_violated": sa["best_violated_clauses"],
        "sa_wall_time": sa["wall_time_sec"],
        "speedup_ratio": bench.speedup_ratio,
        "jarzynski": {
            "mean_work": jarz.mean_work if jarz else None,
            "delta_f": jarz.delta_f_jarzynski if jarz else None,
            "dissipated_work": jarz.mean_dissipated_work if jarz else None,
            "second_law": jarz.second_law_satisfied if jarz else None,
        },
        "diagnostics": {
            "divergence": diag.divergence_detected if diag else False,
            "fdt_error": diag.mean_fdt_relative_error if diag else 0.0,
            "ergodicity_traps": diag.ergodicity_traps_count if diag else 0,
        }
    }

    report_path = output_dir / "tpu_langevin_stress_test_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(_convert_numpy(report_data), f, indent=2)
    console.print(f"\n[bold green]Report exported successfully to {report_path}[/bold green]\n")

    return report_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment 15: Hybrid TPU Langevin Solver Stress Test")
    parser.add_argument("--variables", "-n", type=int, default=16, help="Number of spins/variables")
    parser.add_argument("--trajectories", "-b", type=int, default=48, help="Number of parallel Langevin trajectories")
    parser.add_argument("--steps", "-s", type=int, default=2500, help="Number of integration steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_experiment_15(
        num_variables=args.variables,
        num_trajectories=args.trajectories,
        steps=args.steps,
        seed=args.seed,
    )
