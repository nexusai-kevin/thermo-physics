"""
thermo_physics.tpu.solver
=========================
Hybrid Thermodynamic Processing Unit (TPU) Solver & Classical Simulated Annealing Benchmark.
Solves constrained non-convex 3-SAT and Ising spin-glass problems via Underdamped Langevin
Dynamics with BAOAB symplectic splitting, real-time thermodynamic dissipation tracking,
non-conservative auxiliary clause memory multipliers y_m(t), Sequential Monte Carlo (SMC)
population resampling for non-isothermal Jarzynski weight preservation, and runtime failure diagnostics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from thermo_physics.tpu.hamiltonian import NonConvexIsingSAT
from thermo_physics.tpu.hardware import (
    DACQuantizer,
    OpAmpVoltageClipper,
    OrnsteinUhlenbeckNoise,
    SiliconRealismConfig,
)
from thermo_physics.tpu.integrator import (
    BAOABIntegrator,
    EulerMaruyamaIntegrator,
    AdaptiveThermalScheduler,
    ScheduleType,
    ThermalSchedule,
)
from thermo_physics.tpu.monitors import (
    DiagnosticMonitor,
    DiagnosticReport,
    NumericalDivergenceError,
)
from thermo_physics.tpu.thermodynamics import (
    JarzynskiAnalysis,
    ThermodynamicTracker,
    ThermodynamicTrajectory,
    verify_jarzynski_equality,
)


@dataclass
class TPUSolverConfig:
    """Configuration options for Hybrid Thermodynamic Processing Unit (TPU) solver."""
    num_trajectories: int = 32         # Parallel stochastic trajectories
    total_steps: int = 3000           # Integration steps per trajectory
    dt: float = 0.015                 # Integration timestep
    gamma: float = 1.5                # Bath friction coefficient
    mass: float = 1.0                 # Oscillator mass
    kb: float = 1.0                   # Boltzmann constant
    integrator: str = "baoab"         # 'baoab' or 'euler_maruyama'
    schedule: ThermalSchedule = field(
        default_factory=lambda: ThermalSchedule(
            t_start=2.5,
            t_end=0.02,
            total_steps=3000,
            schedule_type=ScheduleType.COSINE,
            alpha_start=0.2,
            alpha_end=2.5,
            lambda_sat_start=0.5,
            lambda_sat_end=4.0,
        )
    )
    # Non-conservative chaotic auxiliary memory multipliers: dy_m = (kappa * C_m(x) - eta * (y_m - lambda_0)) dt
    use_non_conservative_aux: bool = False
    kappa_aux: float = 20.0           # Upward integration rate for violated clauses
    eta_aux: float = 2.0              # Relaxation rate toward base weight lambda_0
    lambda_0_aux: float = 1.0         # Base multiplier floor

    # SMC Population Resampling (Jarzynski-Neal AIS)
    enable_population_resampling: bool = False
    resample_threshold: float = 0.50  # Kish N_eff / B trigger threshold
    resample_interval: int = 25       # Step frequency to audit resampling

    # Silicon Realism & Hardware Imperfections (DAC quantization, Op-Amp rails, 1/f noise)
    silicon_config: Optional[SiliconRealismConfig] = None

    monitor_diagnostics: bool = True
    track_thermodynamics: bool = True
    seed: Optional[int] = None


@dataclass
class TPUSolutionResult:
    """Complete result from Hybrid TPU Solver execution."""
    best_spins: np.ndarray            # Best discrete configuration found, shape (N,)
    best_continuous_x: np.ndarray     # Best continuous state found, shape (N,)
    best_energy: float                # Best discrete energy attained
    best_violated_clauses: int        # Number of violated clauses in best state
    success: bool                     # True if 0 clauses violated (or optimal ground state)
    all_final_spins: np.ndarray       # All final discrete spins, shape (B, N)
    all_final_energies: np.ndarray    # Final discrete energies across ensemble, shape (B,)
    all_violated_clauses: np.ndarray  # Violated clause count across ensemble, shape (B,)
    wall_time_sec: float              # Total solver execution time in seconds
    thermo_trajectory: Optional[ThermodynamicTrajectory] = None
    jarzynski_analysis: Optional[JarzynskiAnalysis] = None
    diagnostic_report: Optional[DiagnosticReport] = None
    energy_history: List[float] = field(default_factory=list)
    silicon_metrics: Optional[Dict[str, Any]] = None


class HybridTPUSolver:
    """Hybrid Thermodynamic Processing Unit (TPU) simulator."""

    def __init__(self, config: Optional[TPUSolverConfig] = None):
        self.config = config or TPUSolverConfig()
        self.rng = np.random.default_rng(self.config.seed)

    def solve(self, hamiltonian: NonConvexIsingSAT) -> TPUSolutionResult:
        """Execute underdamped Langevin dynamics with BAOAB symplectic splitting across parallel trajectories."""
        start_time = time.perf_counter()
        n = hamiltonian.n
        b = self.config.num_trajectories
        dt = self.config.dt
        gamma = self.config.gamma
        mass = self.config.mass
        kb = self.config.kb
        steps = self.config.total_steps

        # Initialize integrator
        if self.config.integrator.lower() == "baoab":
            integrator: Union[BAOABIntegrator, EulerMaruyamaIntegrator] = BAOABIntegrator(
                dt=dt, gamma=gamma, mass=mass, kb=kb
            )
        else:
            integrator = EulerMaruyamaIntegrator(dt=dt, gamma=gamma, mass=mass, kb=kb)

        # Initialize scheduler
        scheduler = AdaptiveThermalScheduler(self.config.schedule)

        # Silicon Realism & Hardware Imperfections setup
        sil_cfg = self.config.silicon_config
        sim_hamiltonian = hamiltonian
        j_quant_rms = 0.0
        h_quant_rms = 0.0

        clipper: Optional[OpAmpVoltageClipper] = None
        ou_noise: Optional[OrnsteinUhlenbeckNoise] = None

        if sil_cfg is not None:
            if sil_cfg.dac_bits is not None and sil_cfg.quantize_hamiltonian:
                sim_hamiltonian = DACQuantizer.quantize_hamiltonian(
                    hamiltonian, bits=sil_cfg.dac_bits, max_val=sil_cfg.dac_w_max
                )
                j_norm = np.linalg.norm(hamiltonian.j)
                if j_norm > 1e-12:
                    j_quant_rms = float(np.linalg.norm(sim_hamiltonian.j - hamiltonian.j) / j_norm)
                h_norm = np.linalg.norm(hamiltonian.h)
                if h_norm > 1e-12:
                    h_quant_rms = float(np.linalg.norm(sim_hamiltonian.h - hamiltonian.h) / h_norm)

            if sil_cfg.enable_rail_clipping:
                clipper = OpAmpVoltageClipper(
                    v_rail=sil_cfg.v_rail,
                    v_max=sil_cfg.v_max,
                    y_max=sil_cfg.y_max,
                    lambda_0=self.config.lambda_0_aux,
                )

            if sil_cfg.enable_colored_noise:
                ou_noise = OrnsteinUhlenbeckNoise(
                    shape=(b, n),
                    tau_color=sil_cfg.tau_color,
                    use_pink_noise_bank=sil_cfg.use_pink_noise_bank,
                )

        # Initialize physical state: positions near origin (delocalized) and Maxwell-Boltzmann velocities
        init_temp = self.config.schedule.get_temperature(0)
        x = self.rng.normal(0.0, 0.3, size=(b, n))
        v = self.rng.normal(0.0, np.sqrt(kb * max(1e-4, init_temp) / mass), size=(b, n))

        # Initial Hamiltonian parameters
        cur_t, cur_alpha, cur_lam = scheduler.step()

        # Initialize non-conservative auxiliary clause multipliers y_m if requested
        m_clauses = len(hamiltonian.clauses)
        y_aux: Optional[np.ndarray] = None
        if self.config.use_non_conservative_aux and m_clauses > 0:
            y_aux = np.full((b, m_clauses), self.config.lambda_0_aux, dtype=np.float64)

        # Thermodynamic tracking
        thermo_tracker: Optional[ThermodynamicTracker] = None
        if self.config.track_thermodynamics:
            thermo_tracker = ThermodynamicTracker(
                batch_size=b,
                num_variables=n,
                mass=mass,
                kb=kb,
                enable_population_resampling=self.config.enable_population_resampling,
                resample_threshold=self.config.resample_threshold,
            )
            thermo_tracker.initialize(
                x, v, sim_hamiltonian, cur_alpha, cur_lam,
                initial_temperature=cur_t, clause_weights=y_aux
            )

        # Diagnostic monitor with exact configurational temperature auditor
        diagnostic_monitor: Optional[DiagnosticMonitor] = None
        if self.config.monitor_diagnostics:
            diagnostic_monitor = DiagnosticMonitor(
                num_variables=n,
                batch_size=b,
                hamiltonian=sim_hamiltonian,
                mass=mass,
                kb=kb,
            )

        energy_history: List[float] = []
        cached_force: Optional[np.ndarray] = None

        # Main Langevin dynamics integration loop
        for s in range(steps):
            next_t, next_alpha, next_lam = scheduler.step(velocity_variance=float(np.var(v)))

            # If dynamic auxiliary multipliers are active, advance y_m and compute housekeeping work
            dw_hk: Optional[np.ndarray] = None
            y_aux_old = y_aux.copy() if y_aux is not None else None
            if y_aux is not None:
                c_violation = sim_hamiltonian.clause_violations(x)  # shape (B, M)
                dy = (self.config.kappa_aux * c_violation - self.config.eta_aux * (y_aux - self.config.lambda_0_aux)) * dt
                y_aux = y_aux + dy
                if clipper is not None:
                    y_aux = clipper.clip_multiplier(y_aux)
                else:
                    y_aux = np.maximum(self.config.lambda_0_aux, y_aux)

                if sil_cfg is not None and sil_cfg.dac_bits is not None and sil_cfg.quantize_dynamic_multipliers:
                    y_aux = DACQuantizer.quantize(
                        y_aux,
                        bits=sil_cfg.dac_bits,
                        w_min=self.config.lambda_0_aux,
                        w_max=sil_cfg.y_max if sil_cfg.enable_rail_clipping else float(np.max(y_aux)),
                    )

                delta_y = y_aux - self.config.lambda_0_aux
                if np.max(delta_y) > 1e-4:
                    f_aux = -sim_hamiltonian.gradient(x, alpha_well=0.0, lambda_sat=0.0, clause_weights=delta_y)
                    dw_hk = np.sum(f_aux * v * dt, axis=1)

            # Evolve colored noise if enabled
            step_noise: Optional[np.ndarray] = None
            if ou_noise is not None:
                step_noise = ou_noise.step(dt=dt, temperature=cur_t, gamma=gamma, kb=kb, rng=self.rng)

            # Step integrator
            if isinstance(integrator, BAOABIntegrator):
                x, v, cached_force, _ = integrator.step(
                    x=x,
                    v=v,
                    hamiltonian=sim_hamiltonian,
                    temperature=cur_t,
                    alpha_well=cur_alpha,
                    lambda_sat=cur_lam,
                    rng=self.rng,
                    f_cached=cached_force,
                    clause_weights=y_aux,
                    colored_noise=step_noise,
                    clipper=clipper,
                )
            else:
                x, v, cached_force, _ = integrator.step(
                    x=x,
                    v=v,
                    hamiltonian=sim_hamiltonian,
                    temperature=cur_t,
                    alpha_well=cur_alpha,
                    lambda_sat=cur_lam,
                    rng=self.rng,
                    clause_weights=y_aux,
                    colored_noise=step_noise,
                    clipper=clipper,
                )

            # Record non-equilibrium thermodynamic work, heat dissipation, and non-isothermal Sigma
            if thermo_tracker is not None:
                thermo_tracker.record_step(
                    x=x,
                    v=v,
                    hamiltonian=sim_hamiltonian,
                    alpha_old=cur_alpha,
                    lambda_old=cur_lam,
                    alpha_new=next_alpha,
                    lambda_new=next_lam,
                    temperature_old=cur_t,
                    temperature_new=next_t,
                    clause_weights_old=y_aux_old,
                    clause_weights_new=y_aux,
                    dw_hk=dw_hk,
                )

                # Check and perform SMC population resampling if enabled
                if self.config.enable_population_resampling and (s % self.config.resample_interval == 0):
                    x, v, cached_force, y_aux, resampled, _ = thermo_tracker.check_and_resample(
                        x=x,
                        v=v,
                        f_cached=cached_force,
                        rng=self.rng,
                        y_aux=y_aux,
                    )

            # Runtime failure detection & diagnostics (Kinetic & Configurational Temperature)
            if diagnostic_monitor is not None:
                pe_step = hamiltonian.energy(x, alpha_well=cur_alpha, lambda_sat=cur_lam, clause_weights=y_aux)
                diagnostic_monitor.check_step(
                    step=s,
                    x=x,
                    v=v,
                    t_bath=cur_t,
                    energy=pe_step,
                    alpha_well=cur_alpha,
                    lambda_sat=cur_lam,
                    force_cached=cached_force,
                )
                if diagnostic_monitor.report.divergence_detected:
                    break

            if s % max(1, steps // 50) == 0:
                _, disc_e, _ = hamiltonian.evaluate_discrete(x)
                energy_history.append(float(np.min(disc_e)))

            # Advance parameters
            cur_t, cur_alpha, cur_lam = next_t, next_alpha, next_lam

        wall_time = time.perf_counter() - start_time

        # Finalize thermodynamics
        thermo_traj: Optional[ThermodynamicTrajectory] = None
        jarzynski_res: Optional[JarzynskiAnalysis] = None
        if thermo_tracker is not None:
            thermo_traj = thermo_tracker.finalize(x, v, hamiltonian, cur_alpha, cur_lam, clause_weights=y_aux)
            beta_eff = 1.0 / max(1e-4, 0.5 * (self.config.schedule.t_start + self.config.schedule.t_end))
            jarzynski_res = verify_jarzynski_equality(
                work_samples=thermo_traj.work,
                beta=beta_eff,
                generalized_sigma=thermo_traj.generalized_sigma,
                resampled_delta_beta_f=thermo_tracker.cumulative_resampled_delta_beta_f,
                resample_events_count=thermo_tracker.resample_events_count,
                housekeeping_heat=thermo_traj.housekeeping_heat,
                seed=self.config.seed,
            )

        # Finalize diagnostics
        diag_report: Optional[DiagnosticReport] = None
        if diagnostic_monitor is not None:
            diag_report = diagnostic_monitor.finalize()

        # Evaluate discrete solutions across all trajectories
        final_spins, final_energies, final_violated = hamiltonian.evaluate_discrete(x)
        best_idx = int(np.argmin(final_energies))

        best_spins = final_spins[best_idx]
        best_continuous_x = x[best_idx]
        best_energy = float(final_energies[best_idx])
        best_violated = int(final_violated[best_idx])
        success = (best_violated == 0)

        silicon_metrics: Optional[Dict[str, Any]] = None
        if sil_cfg is not None:
            silicon_metrics = {
                "dac_bits": sil_cfg.dac_bits,
                "j_quantization_rms_error": j_quant_rms,
                "h_quantization_rms_error": h_quant_rms,
                "enable_rail_clipping": sil_cfg.enable_rail_clipping,
                "position_clips": clipper.position_clips if clipper is not None else 0,
                "velocity_clips": clipper.velocity_clips if clipper is not None else 0,
                "multiplier_clips": clipper.multiplier_clips if clipper is not None else 0,
                "enable_colored_noise": sil_cfg.enable_colored_noise,
                "tau_color": sil_cfg.tau_color if sil_cfg.enable_colored_noise else 0.0,
                "use_pink_noise_bank": sil_cfg.use_pink_noise_bank,
            }

        return TPUSolutionResult(
            best_spins=best_spins,
            best_continuous_x=best_continuous_x,
            best_energy=best_energy,
            best_violated_clauses=best_violated,
            success=success,
            all_final_spins=final_spins,
            all_final_energies=final_energies,
            all_violated_clauses=final_violated,
            wall_time_sec=wall_time,
            thermo_trajectory=thermo_traj,
            jarzynski_analysis=jarzynski_res,
            diagnostic_report=diag_report,
            energy_history=energy_history,
            silicon_metrics=silicon_metrics,
        )


class ClassicalSimulatedAnnealing:
    """Discrete single-spin flip Metropolis-Hastings Simulated Annealing baseline."""

    def __init__(
        self,
        total_steps: int = 3000,
        t_start: float = 2.5,
        t_end: float = 0.02,
        seed: Optional[int] = None,
    ):
        self.total_steps = total_steps
        self.t_start = t_start
        self.t_end = t_end
        self.rng = np.random.default_rng(seed)

    def solve(self, hamiltonian: NonConvexIsingSAT) -> Dict[str, Any]:
        """Run simulated annealing on the discrete configuration s in {-1, +1}^N."""
        start_time = time.perf_counter()
        n = hamiltonian.n
        spins = self.rng.choice([-1.0, 1.0], size=n)

        _, current_energy, current_violated = hamiltonian.evaluate_discrete(spins)
        best_energy = current_energy
        best_violated = current_violated
        best_spins = spins.copy()

        energy_history: List[float] = [best_energy]

        for step in range(self.total_steps):
            fraction = step / max(1, self.total_steps)
            temp = self.t_start * ((self.t_end / self.t_start) ** fraction)

            flip_var = self.rng.integers(0, n)
            spins[flip_var] *= -1.0

            _, prop_energy, prop_violated = hamiltonian.evaluate_discrete(spins)
            delta_e = prop_energy - current_energy

            if delta_e <= 0.0 or self.rng.uniform() < np.exp(-delta_e / max(1e-4, temp)):
                current_energy = prop_energy
                current_violated = prop_violated
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_violated = current_violated
                    best_spins = spins.copy()
            else:
                spins[flip_var] *= -1.0

            if step % max(1, self.total_steps // 50) == 0:
                energy_history.append(float(best_energy))

        wall_time = time.perf_counter() - start_time
        return {
            "best_spins": best_spins,
            "best_energy": float(best_energy),
            "best_violated_clauses": int(best_violated),
            "success": (best_violated == 0),
            "wall_time_sec": wall_time,
            "energy_history": energy_history,
        }


@dataclass
class BenchmarkComparison:
    """Comparative benchmarking results: Hybrid TPU vs Classical SA."""
    tpu_result: TPUSolutionResult
    classical_result: Dict[str, Any]
    energy_gap: float                  # Classical best E - TPU best E
    tpu_success_rate: float            # Fraction of TPU trajectories finding satisfying solution
    tts_tpu_sec: float                 # Time to Solution (99% target)
    tts_classical_sec: float           # Classical Time to Solution
    speedup_ratio: float               # TTS_classical / TTS_tpu
    fdt_mean_error: float
    t_conf_error: float
    jarzynski_dissipation: float
    n_eff: float
    collapse_ratio: float
    resample_events: int = 0
    housekeeping_heat: float = 0.0


def run_tpu_vs_classical_benchmark(
    hamiltonian: NonConvexIsingSAT,
    num_trajectories: int = 32,
    steps: int = 3000,
    use_non_conservative_aux: bool = False,
    enable_population_resampling: bool = False,
    seed: Optional[int] = 42,
) -> BenchmarkComparison:
    """Execute end-to-end benchmark comparing TPU Langevin Dynamics vs Classical Simulated Annealing."""
    tpu_config = TPUSolverConfig(
        num_trajectories=num_trajectories,
        total_steps=steps,
        use_non_conservative_aux=use_non_conservative_aux,
        enable_population_resampling=enable_population_resampling,
        seed=seed,
    )
    tpu_solver = HybridTPUSolver(tpu_config)
    tpu_res = tpu_solver.solve(hamiltonian)

    sa_solver = ClassicalSimulatedAnnealing(
        total_steps=steps * num_trajectories // 4,
        seed=seed,
    )
    sa_res = sa_solver.solve(hamiltonian)

    p_succ_tpu = max(1e-4, np.mean(tpu_res.all_violated_clauses == 0))
    tts_tpu = tpu_res.wall_time_sec * np.log(1.0 - 0.99) / np.log(1.0 - p_succ_tpu)

    p_succ_sa = 0.99 if sa_res["success"] else 0.05
    tts_sa = sa_res["wall_time_sec"] * np.log(1.0 - 0.99) / np.log(1.0 - p_succ_sa)

    speedup = tts_sa / max(1e-5, tts_tpu)
    energy_gap = sa_res["best_energy"] - tpu_res.best_energy

    fdt_err = tpu_res.diagnostic_report.mean_fdt_relative_error if tpu_res.diagnostic_report else 0.0
    t_conf_err = tpu_res.diagnostic_report.mean_t_conf_relative_error if tpu_res.diagnostic_report else 0.0
    dissipation = tpu_res.jarzynski_analysis.mean_dissipated_work if tpu_res.jarzynski_analysis else 0.0
    n_eff = tpu_res.jarzynski_analysis.n_eff if tpu_res.jarzynski_analysis else 0.0
    collapse_ratio = tpu_res.jarzynski_analysis.collapse_ratio if tpu_res.jarzynski_analysis else 1.0
    resample_events = tpu_res.jarzynski_analysis.resample_events_count if tpu_res.jarzynski_analysis else 0
    q_hk = tpu_res.jarzynski_analysis.mean_q_housekeeping if tpu_res.jarzynski_analysis else 0.0

    return BenchmarkComparison(
        tpu_result=tpu_res,
        classical_result=sa_res,
        energy_gap=energy_gap,
        tpu_success_rate=float(p_succ_tpu),
        tts_tpu_sec=float(tts_tpu),
        tts_classical_sec=float(tts_sa),
        speedup_ratio=float(speedup),
        fdt_mean_error=float(fdt_err),
        t_conf_error=float(t_conf_err),
        jarzynski_dissipation=float(dissipation),
        n_eff=float(n_eff),
        collapse_ratio=float(collapse_ratio),
        resample_events=resample_events,
        housekeeping_heat=q_hk,
    )
