"use client";

import React, { useState, useEffect, useMemo, useCallback } from "react";
import Link from "next/link";
import { Cpu, Zap, RefreshCw, Layers, ArrowRight } from "lucide-react";

export type SystemSizeN = 32 | 64;
export type BarrierWidthW = 1 | 2 | 3 | 4 | 6;
export type ChartMetricId = "pgs" | "tts_sweeps" | "tts_wall_ms" | "escape_rate";

export interface Station17SamplerPoint {
  sampler: string;
  n_spins: number;
  barrier_width_w: number;
  barrier_height_h: number;
  trotter_slices_p: number;
  n_trials: number;
  n_sweeps: number;
  ground_state_prob_pgs: number;
  barrier_escape_rate_per_trial: number;
  barrier_escape_rate_per_sweep: number;
  well_a_trapped_prob: number;
  tts_99_repetitions: number;
  tts_99_sweeps: number;
  tts_99_wall_ms: number;
  wall_time_per_trial_ms: number;
  total_batch_wall_ms: number;
  mean_final_energy: number;
  min_final_energy: number;
  exact_ground_energy: number;
  tau_int: number;
  ess: number;
}

export interface CrossoverThresholdPoint {
  crossover_detected: boolean;
  crossover_width_w: number;
  crossover_bracket: [number, number];
  metric_at_crossover: number;
  pimc_dominates_for_w_le: number | null;
  thermal_dominates_for_w_ge: number | null;
}

export interface SystemCrossoverAnalysis {
  ground_state_prob_crossover: CrossoverThresholdPoint;
  tts_99_sweeps_crossover: CrossoverThresholdPoint;
  tts_99_wall_ms_crossover: CrossoverThresholdPoint;
  barrier_escape_rate_crossover: CrossoverThresholdPoint;
  peak_pimc_tts_sweep_speedup_x: number;
  peak_pimc_tts_wall_speedup_x: number;
}

export interface Station17SystemBlock {
  n_spins: number;
  trotter_slices_p: number;
  effective_quantum_spins: number;
  barrier_height_h: number;
  energy_gap_ab: number;
  thermal_thrml_curve: Station17SamplerPoint[];
  quantum_pimc_curve: Station17SamplerPoint[];
  crossover_analysis: SystemCrossoverAnalysis;
}

export interface Station17TelemetryPayload {
  schema_version: string;
  station_id: number;
  station_slug: string;
  station_name: string;
  generated_at_utc: string;
  total_benchmark_wall_time_s: number;
  experimental_design: {
    hamiltonian_family: string;
    system_sizes_n: number[];
    barrier_widths_w: number[];
    barrier_height_h: number;
    ground_state_energy_gap_ab: number;
    trotter_slices_p: number;
    trotter_coupling_formula: string;
    n_trials_per_point: number;
    deterministic_seed: number;
  };
  hardware_runtime_metadata: {
    post_benchmark?: {
      gpu_name?: string;
      driver_version?: string;
      vram_total_mib?: number;
      vram_used_mib?: number;
      host_ram_spec?: string;
      jax_version?: string;
      thrml_version?: string;
    };
    peak_vram_used_mib?: number;
    peak_gpu_utilization_pct?: number;
  };
  systems: {
    N_32: Station17SystemBlock;
    N_64: Station17SystemBlock;
  };
}

// Initial embedded snapshot matching web/public/stations/station_17_telemetry.json for zero-layout-shift SSR
const INITIAL_STATION_17_TELEMETRY: Station17TelemetryPayload = {
  schema_version: "1.0.0",
  station_id: 17,
  station_slug: "station_17_tunneling_vs_thermal",
  station_name: "Quantum Tunneling (PIMC/SQA) vs. Classical Thermal Barrier Hopping",
  generated_at_utc: "2026-11-22T21:10:48.212597+00:00",
  total_benchmark_wall_time_s: 13.886,
  experimental_design: {
    hamiltonian_family: "Bipartite Double-Well Coincidence-Gate Ising Barrier (K_{W,W} Core + Halo Funnel)",
    system_sizes_n: [32, 64],
    barrier_widths_w: [1, 2, 3, 4, 6],
    barrier_height_h: 8.0,
    ground_state_energy_gap_ab: 4.0,
    trotter_slices_p: 32,
    trotter_coupling_formula: "J_perp = -(1 / (2 * beta)) * ln(tanh(beta * Gamma / P))",
    n_trials_per_point: 256,
    deterministic_seed: 1700,
  },
  hardware_runtime_metadata: {
    post_benchmark: {
      gpu_name: "NVIDIA GeForce RTX 5080",
      driver_version: "616.92",
      vram_total_mib: 16303,
      vram_used_mib: 4552,
      host_ram_spec: "128GB DDR5 System RAM",
      jax_version: "0.12.0.1",
      thrml_version: "0.1.4",
    },
    peak_vram_used_mib: 4552,
    peak_gpu_utilization_pct: 15.0,
  },
  systems: {
    N_32: {
      n_spins: 32,
      trotter_slices_p: 32,
      effective_quantum_spins: 1024,
      barrier_height_h: 8.0,
      energy_gap_ab: 4.0,
      thermal_thrml_curve: [
        {
          sampler: "thermal_thrml_block_gibbs",
          n_spins: 32,
          barrier_width_w: 1,
          barrier_height_h: 8.0,
          trotter_slices_p: 1,
          n_trials: 256,
          n_sweeps: 30,
          ground_state_prob_pgs: 0.1289,
          barrier_escape_rate_per_trial: 0.1289,
          barrier_escape_rate_per_sweep: 0.00459,
          well_a_trapped_prob: 0.8711,
          tts_99_repetitions: 33.3694,
          tts_99_sweeps: 1001.08,
          tts_99_wall_ms: 1.0032,
          wall_time_per_trial_ms: 0.03006,
          total_batch_wall_ms: 7.697,
          mean_final_energy: -87.6527,
          min_final_energy: -91.4,
          exact_ground_energy: -91.4,
          tau_int: 0.5,
          ess: 256.0,
        },
        {
          sampler: "thermal_thrml_block_gibbs",
          n_spins: 32,
          barrier_width_w: 2,
          barrier_height_h: 8.0,
          trotter_slices_p: 1,
          n_trials: 256,
          n_sweeps: 30,
          ground_state_prob_pgs: 0.1602,
          barrier_escape_rate_per_trial: 0.1641,
          barrier_escape_rate_per_sweep: 0.00596,
          well_a_trapped_prob: 0.8164,
          tts_99_repetitions: 26.3847,
          tts_99_sweeps: 791.54,
          tts_99_wall_ms: 0.6203,
          wall_time_per_trial_ms: 0.02351,
          total_batch_wall_ms: 6.019,
          mean_final_energy: -118.0414,
          min_final_energy: -121.6,
          exact_ground_energy: -121.6,
          tau_int: 0.5,
          ess: 256.0,
        },
        {
          sampler: "thermal_thrml_block_gibbs",
          n_spins: 32,
          barrier_width_w: 3,
          barrier_height_h: 8.0,
          trotter_slices_p: 1,
          n_trials: 256,
          n_sweeps: 30,
          ground_state_prob_pgs: 0.1836,
          barrier_escape_rate_per_trial: 0.1875,
          barrier_escape_rate_per_sweep: 0.0069,
          well_a_trapped_prob: 0.7461,
          tts_99_repetitions: 22.7031,
          tts_99_sweeps: 681.09,
          tts_99_wall_ms: 0.7005,
          wall_time_per_trial_ms: 0.03086,
          total_batch_wall_ms: 7.899,
          mean_final_energy: -148.2836,
          min_final_energy: -151.8001,
          exact_ground_energy: -151.8,
          tau_int: 0.66,
          ess: 194.6,
        },
        {
          sampler: "thermal_thrml_block_gibbs",
          n_spins: 32,
          barrier_width_w: 4,
          barrier_height_h: 8.0,
          trotter_slices_p: 1,
          n_trials: 256,
          n_sweeps: 30,
          ground_state_prob_pgs: 0.2031,
          barrier_escape_rate_per_trial: 0.2227,
          barrier_escape_rate_per_sweep: 0.00836,
          well_a_trapped_prob: 0.5664,
          tts_99_repetitions: 20.282,
          tts_99_sweeps: 608.46,
          tts_99_wall_ms: 0.5031,
          wall_time_per_trial_ms: 0.0248,
          total_batch_wall_ms: 6.35,
          mean_final_energy: -178.3266,
          min_final_energy: -182.0002,
          exact_ground_energy: -182.0,
          tau_int: 0.51,
          ess: 248.6,
        },
        {
          sampler: "thermal_thrml_block_gibbs",
          n_spins: 32,
          barrier_width_w: 6,
          barrier_height_h: 8.0,
          trotter_slices_p: 1,
          n_trials: 256,
          n_sweeps: 30,
          ground_state_prob_pgs: 0.1602,
          barrier_escape_rate_per_trial: 0.2109,
          barrier_escape_rate_per_sweep: 0.00787,
          well_a_trapped_prob: 0.168,
          tts_99_repetitions: 26.3847,
          tts_99_sweeps: 791.54,
          tts_99_wall_ms: 0.6657,
          wall_time_per_trial_ms: 0.02523,
          total_batch_wall_ms: 6.459,
          mean_final_energy: -237.386,
          min_final_energy: -242.4001,
          exact_ground_energy: -242.4,
          tau_int: 0.52,
          ess: 244.3,
        },
      ],
      quantum_pimc_curve: [
        {
          sampler: "quantum_pimc_sqa_trotter32",
          n_spins: 32,
          barrier_width_w: 1,
          barrier_height_h: 8.0,
          trotter_slices_p: 32,
          n_trials: 256,
          n_sweeps: 30,
          ground_state_prob_pgs: 0.9922,
          barrier_escape_rate_per_trial: 0.9922,
          barrier_escape_rate_per_sweep: 0.14929,
          well_a_trapped_prob: 0.0078,
          tts_99_repetitions: 1.0,
          tts_99_sweeps: 30.0,
          tts_99_wall_ms: 0.1462,
          wall_time_per_trial_ms: 0.14615,
          total_batch_wall_ms: 37.415,
          mean_final_energy: -91.3687,
          min_final_energy: -91.4,
          exact_ground_energy: -91.4,
          tau_int: 0.5,
          ess: 256.0,
        },
        {
          sampler: "quantum_pimc_sqa_trotter32",
          n_spins: 32,
          barrier_width_w: 2,
          barrier_height_h: 8.0,
          trotter_slices_p: 32,
          n_trials: 256,
          n_sweeps: 30,
          ground_state_prob_pgs: 0.6797,
          barrier_escape_rate_per_trial: 0.6797,
          barrier_escape_rate_per_sweep: 0.03724,
          well_a_trapped_prob: 0.3203,
          tts_99_repetitions: 4.0449,
          tts_99_sweeps: 121.35,
          tts_99_wall_ms: 0.5692,
          wall_time_per_trial_ms: 0.14072,
          total_batch_wall_ms: 36.024,
          mean_final_energy: -120.3187,
          min_final_energy: -121.6,
          exact_ground_energy: -121.6,
          tau_int: 0.56,
          ess: 230.2,
        },
        {
          sampler: "quantum_pimc_sqa_trotter32",
          n_spins: 32,
          barrier_width_w: 3,
          barrier_height_h: 8.0,
          trotter_slices_p: 32,
          n_trials: 256,
          n_sweeps: 30,
          ground_state_prob_pgs: 0.4453,
          barrier_escape_rate_per_trial: 0.4453,
          barrier_escape_rate_per_sweep: 0.01945,
          well_a_trapped_prob: 0.543,
          tts_99_repetitions: 7.8133,
          tts_99_sweeps: 234.4,
          tts_99_wall_ms: 1.0905,
          wall_time_per_trial_ms: 0.13957,
          total_batch_wall_ms: 35.729,
          mean_final_energy: -149.5891,
          min_final_energy: -151.8001,
          exact_ground_energy: -151.8,
          tau_int: 0.53,
          ess: 242.7,
        },
        {
          sampler: "quantum_pimc_sqa_trotter32",
          n_spins: 32,
          barrier_width_w: 4,
          barrier_height_h: 8.0,
          trotter_slices_p: 32,
          n_trials: 256,
          n_sweeps: 30,
          ground_state_prob_pgs: 0.25,
          barrier_escape_rate_per_trial: 0.25,
          barrier_escape_rate_per_sweep: 0.00954,
          well_a_trapped_prob: 0.7266,
          tts_99_repetitions: 16.0079,
          tts_99_sweeps: 480.24,
          tts_99_wall_ms: 2.2851,
          wall_time_per_trial_ms: 0.14275,
          total_batch_wall_ms: 36.544,
          mean_final_energy: -178.9853,
          min_final_energy: -182.0002,
          exact_ground_energy: -182.0,
          tau_int: 0.5,
          ess: 256.0,
        },
        {
          sampler: "quantum_pimc_sqa_trotter32",
          n_spins: 32,
          barrier_width_w: 6,
          barrier_height_h: 8.0,
          trotter_slices_p: 32,
          n_trials: 256,
          n_sweeps: 30,
          ground_state_prob_pgs: 0.1094,
          barrier_escape_rate_per_trial: 0.1094,
          barrier_escape_rate_per_sweep: 0.00385,
          well_a_trapped_prob: 0.7617,
          tts_99_repetitions: 39.7575,
          tts_99_sweeps: 1192.72,
          tts_99_wall_ms: 5.7913,
          wall_time_per_trial_ms: 0.14567,
          total_batch_wall_ms: 37.291,
          mean_final_energy: -238.7073,
          min_final_energy: -242.4001,
          exact_ground_energy: -242.4,
          tau_int: 0.57,
          ess: 224.2,
        },
      ],
      crossover_analysis: {
        ground_state_prob_crossover: {
          crossover_detected: true,
          crossover_width_w: 4.96,
          crossover_bracket: [4, 6],
          metric_at_crossover: 0.1825,
          pimc_dominates_for_w_le: 4,
          thermal_dominates_for_w_ge: 6,
        },
        tts_99_sweeps_crossover: {
          crossover_detected: true,
          crossover_width_w: 4.484,
          crossover_bracket: [4, 6],
          metric_at_crossover: 652.8017,
          pimc_dominates_for_w_le: 4,
          thermal_dominates_for_w_ge: 6,
        },
        tts_99_wall_ms_crossover: {
          crossover_detected: true,
          crossover_width_w: 2.06,
          crossover_bracket: [2, 3],
          metric_at_crossover: 0.6251,
          pimc_dominates_for_w_le: 2,
          thermal_dominates_for_w_ge: 3,
        },
        barrier_escape_rate_crossover: {
          crossover_detected: true,
          crossover_width_w: 4.454,
          crossover_bracket: [4, 6],
          metric_at_crossover: 0.0082,
          pimc_dominates_for_w_le: 4,
          thermal_dominates_for_w_ge: 6,
        },
        peak_pimc_tts_sweep_speedup_x: 33.37,
        peak_pimc_tts_wall_speedup_x: 6.86,
      },
    },
    N_64: {
      n_spins: 64,
      trotter_slices_p: 32,
      effective_quantum_spins: 2048,
      barrier_height_h: 8.0,
      energy_gap_ab: 4.0,
      thermal_thrml_curve: [
        {
          sampler: "thermal_thrml_block_gibbs",
          n_spins: 64,
          barrier_width_w: 1,
          barrier_height_h: 8.0,
          trotter_slices_p: 1,
          n_trials: 256,
          n_sweeps: 30,
          ground_state_prob_pgs: 0.1289,
          barrier_escape_rate_per_trial: 0.1289,
          barrier_escape_rate_per_sweep: 0.00459,
          well_a_trapped_prob: 0.8711,
          tts_99_repetitions: 33.3694,
          tts_99_sweeps: 1001.08,
          tts_99_wall_ms: 1.7695,
          wall_time_per_trial_ms: 0.05303,
          total_batch_wall_ms: 13.575,
          mean_final_energy: -217.5293,
          min_final_energy: -221.64,
          exact_ground_energy: -221.64,
          tau_int: 0.5,
          ess: 256.0,
        },
        {
          sampler: "thermal_thrml_block_gibbs",
          n_spins: 64,
          barrier_width_w: 2,
          barrier_height_h: 8.0,
          trotter_slices_p: 1,
          n_trials: 256,
          n_sweeps: 30,
          ground_state_prob_pgs: 0.0859,
          barrier_escape_rate_per_trial: 0.0898,
          barrier_escape_rate_per_sweep: 0.00313,
          well_a_trapped_prob: 0.8828,
          tts_99_repetitions: 51.2502,
          tts_99_sweeps: 1537.51,
          tts_99_wall_ms: 2.7085,
          wall_time_per_trial_ms: 0.05285,
          total_batch_wall_ms: 13.529,
          mean_final_energy: -269.6713,
          min_final_energy: -273.1733,
          exact_ground_energy: -273.1733,
          tau_int: 0.5,
          ess: 256.0,
        },
        {
          sampler: "thermal_thrml_block_gibbs",
          n_spins: 64,
          barrier_width_w: 3,
          barrier_height_h: 8.0,
          trotter_slices_p: 1,
          n_trials: 256,
          n_sweeps: 30,
          ground_state_prob_pgs: 0.1328,
          barrier_escape_rate_per_trial: 0.1367,
          barrier_escape_rate_per_sweep: 0.00489,
          well_a_trapped_prob: 0.7852,
          tts_99_repetitions: 32.317,
          tts_99_sweeps: 969.51,
          tts_99_wall_ms: 1.6741,
          wall_time_per_trial_ms: 0.0518,
          total_batch_wall_ms: 13.261,
          mean_final_energy: -321.1573,
          min_final_energy: -324.7066,
          exact_ground_energy: -324.7067,
          tau_int: 0.56,
          ess: 226.7,
        },
        {
          sampler: "thermal_thrml_block_gibbs",
          n_spins: 64,
          barrier_width_w: 4,
          barrier_height_h: 8.0,
          trotter_slices_p: 1,
          n_trials: 256,
          n_sweeps: 30,
          ground_state_prob_pgs: 0.1445,
          barrier_escape_rate_per_trial: 0.1602,
          barrier_escape_rate_per_sweep: 0.0058,
          well_a_trapped_prob: 0.5664,
          tts_99_repetitions: 29.5003,
          tts_99_sweeps: 885.01,
          tts_99_wall_ms: 1.4904,
          wall_time_per_trial_ms: 0.05052,
          total_batch_wall_ms: 12.934,
          mean_final_energy: -372.4293,
          min_final_energy: -376.2401,
          exact_ground_energy: -376.24,
          tau_int: 0.5,
          ess: 256.0,
        },
        {
          sampler: "thermal_thrml_block_gibbs",
          n_spins: 64,
          barrier_width_w: 6,
          barrier_height_h: 8.0,
          trotter_slices_p: 1,
          n_trials: 256,
          n_sweeps: 30,
          ground_state_prob_pgs: 0.1133,
          barrier_escape_rate_per_trial: 0.1523,
          barrier_escape_rate_per_sweep: 0.00549,
          well_a_trapped_prob: 0.1562,
          tts_99_repetitions: 38.3037,
          tts_99_sweeps: 1149.11,
          tts_99_wall_ms: 2.007,
          wall_time_per_trial_ms: 0.0524,
          total_batch_wall_ms: 13.414,
          mean_final_energy: -474.167,
          min_final_energy: -479.3069,
          exact_ground_energy: -479.3067,
          tau_int: 0.5,
          ess: 256.0,
        },
      ],
      quantum_pimc_curve: [
        {
          sampler: "quantum_pimc_sqa_trotter32",
          n_spins: 64,
          barrier_width_w: 1,
          barrier_height_h: 8.0,
          trotter_slices_p: 32,
          n_trials: 256,
          n_sweeps: 30,
          ground_state_prob_pgs: 0.9727,
          barrier_escape_rate_per_trial: 0.9727,
          barrier_escape_rate_per_sweep: 0.11306,
          well_a_trapped_prob: 0.0273,
          tts_99_repetitions: 1.2793,
          tts_99_sweeps: 38.38,
          tts_99_wall_ms: 0.4462,
          wall_time_per_trial_ms: 0.3488,
          total_batch_wall_ms: 89.293,
          mean_final_energy: -221.5309,
          min_final_energy: -221.64,
          exact_ground_energy: -221.64,
          tau_int: 0.5,
          ess: 256.0,
        },
        {
          sampler: "quantum_pimc_sqa_trotter32",
          n_spins: 64,
          barrier_width_w: 2,
          barrier_height_h: 8.0,
          trotter_slices_p: 32,
          n_trials: 256,
          n_sweeps: 30,
          ground_state_prob_pgs: 0.6328,
          barrier_escape_rate_per_trial: 0.6328,
          barrier_escape_rate_per_sweep: 0.03284,
          well_a_trapped_prob: 0.3672,
          tts_99_repetitions: 4.5966,
          tts_99_sweeps: 137.9,
          tts_99_wall_ms: 1.6146,
          wall_time_per_trial_ms: 0.35126,
          total_batch_wall_ms: 89.922,
          mean_final_energy: -271.7046,
          min_final_energy: -273.1733,
          exact_ground_energy: -273.1733,
          tau_int: 0.56,
          ess: 230.2,
        },
        {
          sampler: "quantum_pimc_sqa_trotter32",
          n_spins: 64,
          barrier_width_w: 3,
          barrier_height_h: 8.0,
          trotter_slices_p: 32,
          n_trials: 256,
          n_sweeps: 30,
          ground_state_prob_pgs: 0.3359,
          barrier_escape_rate_per_trial: 0.3359,
          barrier_escape_rate_per_sweep: 0.01355,
          well_a_trapped_prob: 0.6641,
          tts_99_repetitions: 11.2491,
          tts_99_sweeps: 337.47,
          tts_99_wall_ms: 3.9371,
          wall_time_per_trial_ms: 0.34999,
          total_batch_wall_ms: 89.598,
          mean_final_energy: -322.0504,
          min_final_energy: -324.7066,
          exact_ground_energy: -324.7067,
          tau_int: 0.5,
          ess: 256.0,
        },
        {
          sampler: "quantum_pimc_sqa_trotter32",
          n_spins: 64,
          barrier_width_w: 4,
          barrier_height_h: 8.0,
          trotter_slices_p: 32,
          n_trials: 256,
          n_sweeps: 30,
          ground_state_prob_pgs: 0.1172,
          barrier_escape_rate_per_trial: 0.1172,
          barrier_escape_rate_per_sweep: 0.00415,
          well_a_trapped_prob: 0.8828,
          tts_99_repetitions: 36.9469,
          tts_99_sweeps: 1108.41,
          tts_99_wall_ms: 13.1883,
          wall_time_per_trial_ms: 0.35695,
          total_batch_wall_ms: 91.38,
          mean_final_energy: -372.7089,
          min_final_energy: -376.2401,
          exact_ground_energy: -376.24,
          tau_int: 0.5,
          ess: 256.0,
        },
        {
          sampler: "quantum_pimc_sqa_trotter32",
          n_spins: 64,
          barrier_width_w: 6,
          barrier_height_h: 8.0,
          trotter_slices_p: 32,
          n_trials: 256,
          n_sweeps: 30,
          ground_state_prob_pgs: 0.0078,
          barrier_escape_rate_per_trial: 0.0078,
          barrier_escape_rate_per_sweep: 0.00026,
          well_a_trapped_prob: 0.9883,
          tts_99_repetitions: 587.1562,
          tts_99_sweeps: 17614.69,
          tts_99_wall_ms: 224.4168,
          wall_time_per_trial_ms: 0.38221,
          total_batch_wall_ms: 97.846,
          mean_final_energy: -475.3318,
          min_final_energy: -479.3069,
          exact_ground_energy: -479.3067,
          tau_int: 0.5,
          ess: 256.0,
        },
      ],
      crossover_analysis: {
        ground_state_prob_crossover: {
          crossover_detected: true,
          crossover_width_w: 3.882,
          crossover_bracket: [3, 4],
          metric_at_crossover: 0.1431,
          pimc_dominates_for_w_le: 3,
          thermal_dominates_for_w_ge: 4,
        },
        tts_99_sweeps_crossover: {
          crossover_detected: true,
          crossover_width_w: 3.739,
          crossover_bracket: [3, 4],
          metric_at_crossover: 907.0774,
          pimc_dominates_for_w_le: 3,
          thermal_dominates_for_w_ge: 4,
        },
        tts_99_wall_ms_crossover: {
          crossover_detected: true,
          crossover_width_w: 2.326,
          crossover_bracket: [2, 3],
          metric_at_crossover: 2.3714,
          pimc_dominates_for_w_le: 2,
          thermal_dominates_for_w_ge: 3,
        },
        barrier_escape_rate_crossover: {
          crossover_detected: true,
          crossover_width_w: 3.84,
          crossover_bracket: [3, 4],
          metric_at_crossover: 0.0057,
          pimc_dominates_for_w_le: 3,
          thermal_dominates_for_w_ge: 4,
        },
        peak_pimc_tts_sweep_speedup_x: 26.08,
        peak_pimc_tts_wall_speedup_x: 3.97,
      },
    },
  },
};

const WIDTH_OPTIONS: BarrierWidthW[] = [1, 2, 3, 4, 6];

const METRIC_META: Record<
  ChartMetricId,
  {
    label: string;
    shortLabel: string;
    unit: string;
    logScale: boolean;
    higherIsBetter: boolean;
    description: string;
  }
> = {
  pgs: {
    label: "Ground-State Success (P_gs)",
    shortLabel: "P_gs (%)",
    unit: "%",
    logScale: false,
    higherIsBetter: true,
    description: "Probability of reaching the true global ground state (Well B) after 30 annealing sweeps.",
  },
  tts_sweeps: {
    label: "Time-to-Solution (TTS_99 Sweeps)",
    shortLabel: "TTS_99 (Sweeps)",
    unit: "sweeps",
    logScale: true,
    higherIsBetter: false,
    description: "Algorithmic sweeps required to reach Well B with 99% confidence: R_99 * 30 sweeps.",
  },
  tts_wall_ms: {
    label: "Time-to-Solution (TTS_99 Wall ms)",
    shortLabel: "TTS_99 (ms)",
    unit: "ms",
    logScale: true,
    higherIsBetter: false,
    description: "Measured wall-clock milliseconds on RTX 5080 host (includes P=32 Trotter slice factor).",
  },
  escape_rate: {
    label: "Barrier Escape Rate (/Sweep)",
    shortLabel: "Escape / Sweep",
    unit: "/sweep",
    logScale: true,
    higherIsBetter: true,
    description: "Per-sweep probability of escaping false local minimum Well A across height H=8.0.",
  },
};

export function QuantumTunnelingCrossover() {
  const [telemetry, setTelemetry] = useState<Station17TelemetryPayload>(INITIAL_STATION_17_TELEMETRY);
  const [fetchStatus, setFetchStatus] = useState<"ready" | "loading" | "fallback">("ready");
  const [systemN, setSystemN] = useState<SystemSizeN>(64);
  const [selectedW, setSelectedW] = useState<BarrierWidthW>(1);
  const [metricId, setMetricId] = useState<ChartMetricId>("pgs");

  const fetchTelemetryData = useCallback(async (): Promise<Station17TelemetryPayload | null> => {
    try {
      const res = await fetch("/stations/station_17_telemetry.json", { cache: "no-store" });
      if (!res.ok) {
        return null;
      }
      const data = (await res.json()) as Station17TelemetryPayload;
      if (data && data.systems && data.systems.N_32 && data.systems.N_64) {
        return data;
      }
      return null;
    } catch {
      return null;
    }
  }, []);

  const loadTelemetry = useCallback(async () => {
    setFetchStatus("loading");
    const data = await fetchTelemetryData();
    if (data) {
      setTelemetry(data);
      setFetchStatus("ready");
    } else {
      setFetchStatus("fallback");
    }
  }, [fetchTelemetryData]);

  useEffect(() => {
    let cancelled = false;
    void fetchTelemetryData().then((data) => {
      if (cancelled) return;
      if (data) {
        setTelemetry(data);
        setFetchStatus("ready");
      } else {
        setFetchStatus("fallback");
      }
    });
    return () => {
      cancelled = true;
    };
  }, [fetchTelemetryData]);

  const activeSystem: Station17SystemBlock = useMemo(
    () => (systemN === 32 ? telemetry.systems.N_32 : telemetry.systems.N_64),
    [telemetry, systemN]
  );

  const thermalCurve = activeSystem.thermal_thrml_curve;
  const pimcCurve = activeSystem.quantum_pimc_curve;
  const crossovers = activeSystem.crossover_analysis;

  const activeCrossover: CrossoverThresholdPoint = useMemo(() => {
    if (metricId === "pgs") return crossovers.ground_state_prob_crossover;
    if (metricId === "tts_sweeps") return crossovers.tts_99_sweeps_crossover;
    if (metricId === "tts_wall_ms") return crossovers.tts_99_wall_ms_crossover;
    return crossovers.barrier_escape_rate_crossover;
  }, [crossovers, metricId]);

  const selectedThermal = useMemo(
    () => thermalCurve.find((pt) => pt.barrier_width_w === selectedW) ?? thermalCurve[0],
    [thermalCurve, selectedW]
  );
  const selectedPimc = useMemo(
    () => pimcCurve.find((pt) => pt.barrier_width_w === selectedW) ?? pimcCurve[0],
    [pimcCurve, selectedW]
  );

  const getMetricValue = useCallback(
    (pt: Station17SamplerPoint): number => {
      if (metricId === "pgs") return pt.ground_state_prob_pgs * 100.0;
      if (metricId === "tts_sweeps") return pt.tts_99_sweeps;
      if (metricId === "tts_wall_ms") return pt.tts_99_wall_ms;
      return pt.barrier_escape_rate_per_sweep;
    },
    [metricId]
  );

  // SVG Sweep Chart Geometry
  const svgW = 540;
  const svgH = 235;
  const padL = 52;
  const padR = 20;
  const padT = 24;
  const padB = 34;
  const plotW = svgW - padL - padR;
  const plotH = svgH - padT - padB;

  const wMin = 1.0;
  const wMax = 6.0;
  const toX = useCallback(
    (w: number): number => padL + ((w - wMin) / (wMax - wMin)) * plotW,
    [plotW]
  );

  const { toY, yTicks } = useMemo(() => {
    const meta = METRIC_META[metricId];
    const allVals = [...thermalCurve.map(getMetricValue), ...pimcCurve.map(getMetricValue)];

    if (!meta.logScale) {
      return {
        toY: (v: number): number => {
          const clamped = Math.max(0, Math.min(100, v));
          return padT + ((100 - clamped) / 100) * plotH;
        },
        yTicks: [0, 25, 50, 75, 100].map((v) => ({ val: v, label: `${v}%` })),
      };
    }

    const minPos = Math.max(1e-5, Math.min(...allVals) * 0.75);
    const maxPos = Math.max(minPos * 10, Math.max(...allVals) * 1.25);
    const logMin = Math.log10(minPos);
    const logMax = Math.log10(maxPos);

    const mapY = (v: number): number => {
      const lv = Math.log10(Math.max(minPos, Math.min(maxPos, v)));
      return padT + ((logMax - lv) / Math.max(1e-6, logMax - logMin)) * plotH;
    };

    const ticks: { val: number; label: string }[] = [];
    for (let i = 0; i <= 4; i++) {
      const lv = logMin + (i / 4) * (logMax - logMin);
      const raw = Math.pow(10, lv);
      let label = raw.toFixed(1);
      if (metricId === "tts_sweeps") label = raw >= 100 ? `${Math.round(raw)}` : raw.toFixed(0);
      else if (metricId === "tts_wall_ms") label = raw >= 10 ? `${raw.toFixed(1)}` : raw.toFixed(2);
      else label = `${(raw * 100).toFixed(2)}%`;
      ticks.push({ val: raw, label });
    }
    return { toY: mapY, yTicks: ticks };
  }, [metricId, thermalCurve, pimcCurve, getMetricValue, plotH]);

  const thermalPath = useMemo(
    () =>
      thermalCurve
        .map((pt, i) => `${i === 0 ? "M" : "L"}${toX(pt.barrier_width_w).toFixed(1)},${toY(getMetricValue(pt)).toFixed(1)}`)
        .join(" "),
    [thermalCurve, getMetricValue, toX, toY]
  );

  const pimcPath = useMemo(
    () =>
      pimcCurve
        .map((pt, i) => `${i === 0 ? "M" : "L"}${toX(pt.barrier_width_w).toFixed(1)},${toY(getMetricValue(pt)).toFixed(1)}`)
        .join(" "),
    [pimcCurve, getMetricValue, toX, toY]
  );

  const crossoverX = toX(activeCrossover.crossover_width_w);
  const selectedSweepSpeedup = selectedThermal.tts_99_sweeps / Math.max(1e-6, selectedPimc.tts_99_sweeps);
  const selectedWallSpeedup = selectedThermal.tts_99_wall_ms / Math.max(1e-6, selectedPimc.tts_99_wall_ms);
  const pimcWinsSelected =
    METRIC_META[metricId].higherIsBetter
      ? getMetricValue(selectedPimc) >= getMetricValue(selectedThermal)
      : getMetricValue(selectedPimc) <= getMetricValue(selectedThermal);

  const gpuMeta = telemetry.hardware_runtime_metadata.post_benchmark;

  return (
    <div className="bg-[#05070f] border border-[rgba(250,249,245,0.10)] rounded-2xl p-6 text-[#faf9f5] shadow-2xl">
      {/* Top Header & Crossover Callout Badge */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-[rgba(250,249,245,0.08)] pb-4 mb-6">
        <div className="flex items-center gap-3">
          <div className="p-2.5 bg-[rgba(94,234,212,0.08)] border border-[rgba(94,234,212,0.24)] rounded-lg text-[#5eead4]">
            <Layers className="w-5 h-5" />
          </div>
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-[10px] font-mono uppercase tracking-[0.18em] text-[#5eead4]">
                Station 17 · Quantum-Thermodynamic Frontier
              </span>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-[rgba(250,249,245,0.06)] text-[rgba(250,249,245,0.65)]">
                P = {activeSystem.trotter_slices_p} Trotter Slices · H = {activeSystem.barrier_height_h.toFixed(1)}
              </span>
            </div>
            <h3 className="font-[family-name:var(--font-display)] text-xl font-normal text-[#faf9f5] tracking-tight mt-0.5">
              Quantum Tunneling (PIMC/SQA) vs. Classical Thermal Barrier Hopping
            </h3>
          </div>
        </div>

        {/* Highlighted Crossover Threshold Callout Badge */}
        <div className="flex flex-wrap items-center gap-2">
          <div className="px-3.5 py-1.5 rounded-full border border-[#5eead4]/45 bg-[rgba(94,234,212,0.10)] text-[#5eead4] font-mono text-xs flex items-center gap-2">
            <Zap className="w-3.5 h-3.5 shrink-0" />
            <span>
              Crossover Threshold: <strong>W_c(sweeps) = {crossovers.tts_99_sweeps_crossover.crossover_width_w.toFixed(2)}</strong> ·{" "}
              <strong>W_c(wall) = {crossovers.tts_99_wall_ms_crossover.crossover_width_w.toFixed(2)}</strong>
            </span>
          </div>
          <button
            type="button"
            onClick={() => void loadTelemetry()}
            className="p-2 rounded-full border border-[rgba(250,249,245,0.14)] bg-[#020308] text-[rgba(250,249,245,0.7)] hover:text-[#5eead4] hover:border-[#5eead4]/40 transition-colors"
            title="Reload /stations/station_17_telemetry.json"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${fetchStatus === "loading" ? "animate-spin text-[#5eead4]" : ""}`} />
          </button>
        </div>
      </div>

      {/* Control Strip: System Size N, Metric Selector, and Barrier Width W Inspector */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 mb-6 bg-[#020308]/75 p-4 rounded-xl border border-[rgba(250,249,245,0.08)]">
        {/* System Size N */}
        <div className="lg:col-span-3">
          <label className="text-[10px] font-mono uppercase tracking-[0.16em] text-[rgba(250,249,245,0.5)] block mb-2">
            Ising Lattice Dimension (N)
          </label>
          <div className="grid grid-cols-2 gap-1.5">
            {([32, 64] as SystemSizeN[]).map((nVal) => (
              <button
                key={nVal}
                type="button"
                onClick={() => setSystemN(nVal)}
                className={`py-2 px-2.5 rounded-lg border text-xs font-mono transition-colors ${
                  systemN === nVal
                    ? "bg-[#faf9f5] border-[#faf9f5] text-[#020308] font-medium"
                    : "bg-[#05070f] border-[rgba(250,249,245,0.10)] text-[rgba(250,249,245,0.68)] hover:border-[rgba(250,249,245,0.28)]"
                }`}
              >
                N = {nVal} ({nVal * 32} PIMC)
              </button>
            ))}
          </div>
        </div>

        {/* Sweep Metric Selector */}
        <div className="lg:col-span-5">
          <label className="text-[10px] font-mono uppercase tracking-[0.16em] text-[rgba(250,249,245,0.5)] block mb-2">
            Sweep Benchmark Metric
          </label>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-1.5">
            {(Object.keys(METRIC_META) as ChartMetricId[]).map((mKey) => (
              <button
                key={mKey}
                type="button"
                onClick={() => setMetricId(mKey)}
                className={`py-2 px-2 rounded-lg border text-[11px] font-mono transition-colors ${
                  metricId === mKey
                    ? "bg-[#5eead4] border-[#5eead4] text-[#020308] font-medium"
                    : "bg-[#05070f] border-[rgba(250,249,245,0.10)] text-[rgba(250,249,245,0.68)] hover:border-[rgba(250,249,245,0.28)]"
                }`}
              >
                {METRIC_META[mKey].shortLabel}
              </button>
            ))}
          </div>
        </div>

        {/* Barrier Hamming Width W Selector */}
        <div className="lg:col-span-4">
          <label className="text-[10px] font-mono uppercase tracking-[0.16em] text-[rgba(250,249,245,0.5)] block mb-2">
            Inspect Barrier Hamming Width (W)
          </label>
          <div className="grid grid-cols-5 gap-1.5">
            {WIDTH_OPTIONS.map((wVal) => (
              <button
                key={wVal}
                type="button"
                onClick={() => setSelectedW(wVal)}
                className={`py-2 rounded-lg border text-xs font-mono transition-colors ${
                  selectedW === wVal
                    ? "bg-[#34d399]/20 border-[#34d399] text-[#34d399] font-medium"
                    : "bg-[#05070f] border-[rgba(250,249,245,0.10)] text-[rgba(250,249,245,0.68)] hover:border-[rgba(250,249,245,0.28)]"
                }`}
              >
                W={wVal}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main Split: Left Double-Well & PIMC Worldline Schematic (5 cols) + Right Sweep Crossover Chart (7 cols) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 mb-6">
        {/* Left Column: Double-Well Barrier Profile & Instanton Worldline Schematic */}
        <div className="lg:col-span-5 bg-[#020308]/85 border border-[rgba(250,249,245,0.08)] rounded-xl p-4 flex flex-col justify-between">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[10px] font-mono uppercase tracking-[0.18em] text-[#5eead4]">
              Double-Well Landscape · W = {selectedW} ({2 * selectedW} Core Spins)
            </span>
            <span
              className={`text-[10px] font-mono px-2 py-0.5 rounded-full border ${
                pimcWinsSelected
                  ? "border-[#5eead4]/40 bg-[rgba(94,234,212,0.10)] text-[#5eead4]"
                  : "border-amber-400/40 bg-[rgba(245,158,11,0.10)] text-amber-300"
              }`}
            >
              {pimcWinsSelected ? "Quantum Tunneling Regime" : "Thermal Activation Regime"}
            </span>
          </div>

          {/* Double-Well Barrier SVG Diagram */}
          <svg
            viewBox="0 0 360 185"
            className="w-full h-44 bg-[#05070f] rounded-lg border border-[rgba(250,249,245,0.06)] p-2"
            role="img"
            aria-label={`Double-well Ising energy barrier of height H=8.0 and Hamming width W=${selectedW}`}
          >
            {/* Reference baseline E_A */}
            <line x1="28" y1="115" x2="332" y2="115" stroke="rgba(250,249,245,0.12)" strokeDasharray="3 3" />

            {/* Dynamic Double-Well Curve whose barrier width expands with selectedW */}
            {(() => {
              const halfW = 10 + selectedW * 9;
              const cx = 175;
              const leftBase = cx - halfW - 18;
              const rightBase = cx + halfW + 18;
              const pathD = [
                `M 24 68`,
                `Q 58 115, 86 115`,
                `L ${leftBase} 115`,
                `C ${cx - halfW} 115, ${cx - halfW * 0.85} 34, ${cx} 34`,
                `C ${cx + halfW * 0.85} 34, ${cx + halfW} 148, ${rightBase} 148`,
                `L 295 148`,
                `Q 322 148, 338 82`,
              ].join(" ");

              return (
                <>
                  {/* Shaded barrier core */}
                  <path
                    d={`M ${leftBase} 115 C ${cx - halfW} 115, ${cx - halfW * 0.85} 34, ${cx} 34 C ${cx + halfW * 0.85} 34, ${cx + halfW} 115, ${rightBase} 115 Z`}
                    fill="rgba(94,234,212,0.08)"
                  />
                  {/* Energy curve */}
                  <path d={pathD} fill="none" stroke="#faf9f5" strokeWidth="2.2" />

                  {/* Classical Thermal Over-the-Top Arrow */}
                  <path
                    d={`M 88 106 C ${cx - halfW - 10} 18, ${cx + halfW + 10} 18, 280 136`}
                    fill="none"
                    stroke="#f59e0b"
                    strokeWidth="1.6"
                    strokeDasharray="4 3"
                  />
                  <text x={cx} y="16" textAnchor="middle" fill="#f59e0b" fontSize="9.5" fontFamily="monospace">
                    Thermal Hopping: exp(-βH) · P_gs={(selectedThermal.ground_state_prob_pgs * 100).toFixed(1)}%
                  </text>

                  {/* Quantum Tunneling Through-the-Spike Arrow */}
                  <line
                    x1={leftBase - 6}
                    y1="115"
                    x2={rightBase + 6}
                    y2="115"
                    stroke="#5eead4"
                    strokeWidth="2.4"
                    strokeDasharray={selectedW >= 4 ? "3 4" : "none"}
                  />
                  <polygon points={`${rightBase + 10},115 ${rightBase + 2},111 ${rightBase + 2},119`} fill="#5eead4" />
                  <text x={cx} y="104" textAnchor="middle" fill="#5eead4" fontSize="9.5" fontFamily="monospace">
                    PIMC Tunneling (W={selectedW}): P_gs={(selectedPimc.ground_state_prob_pgs * 100).toFixed(1)}%
                  </text>

                  {/* Height H annotation */}
                  <line x1={cx} y1="34" x2={cx} y2="115" stroke="rgba(250,249,245,0.35)" strokeDasharray="2 2" />
                  <text x={cx + 5} y="74" fill="rgba(250,249,245,0.8)" fontSize="9" fontFamily="monospace">
                    H = 8.0
                  </text>
                </>
              );
            })()}

            {/* Well A & Well B Markers */}
            <circle cx="86" cy="115" r="4.5" fill="#f59e0b" />
            <text x="86" y="133" textAnchor="middle" fill="#f59e0b" fontSize="9.5" fontFamily="monospace">
              Well A (Trap)
            </text>
            <text x="86" y="145" textAnchor="middle" fill="rgba(250,249,245,0.5)" fontSize="8.5" fontFamily="monospace">
              E_A = {selectedThermal.exact_ground_energy + 4.0}
            </text>

            <circle cx="290" cy="148" r="5" fill="#34d399" />
            <text x="290" y="165" textAnchor="middle" fill="#34d399" fontSize="9.5" fontFamily="monospace">
              Well B (Ground)
            </text>
            <text x="290" y="176" textAnchor="middle" fill="rgba(250,249,245,0.5)" fontSize="8.5" fontFamily="monospace">
              E_B = {selectedThermal.exact_ground_energy}
            </text>
          </svg>

          {/* Suzuki-Trotter Action Breakdown for Selected W */}
          <div className="mt-3 pt-3 border-t border-[rgba(250,249,245,0.08)] space-y-2 text-xs font-mono">
            <div className="flex items-center justify-between text-[rgba(250,249,245,0.65)]">
              <span>Suzuki-Trotter Coupling:</span>
              <span className="text-[#5eead4]">J_⊥ = -(1/2β) ln tanh(βΓ/P)</span>
            </div>
            <div className="flex items-center justify-between text-[rgba(250,249,245,0.65)]">
              <span>Selected W = {selectedW} Sweep Speedup:</span>
              <span className={selectedSweepSpeedup >= 1.0 ? "text-[#34d399] font-medium" : "text-amber-300"}>
                {selectedSweepSpeedup.toFixed(2)}x ({selectedPimc.tts_99_sweeps.toFixed(0)} vs {selectedThermal.tts_99_sweeps.toFixed(0)} sw)
              </span>
            </div>
            <div className="flex items-center justify-between text-[rgba(250,249,245,0.65)]">
              <span>Selected W = {selectedW} Wall-Clock Speedup:</span>
              <span className={selectedWallSpeedup >= 1.0 ? "text-[#34d399] font-medium" : "text-amber-300"}>
                {selectedWallSpeedup.toFixed(2)}x ({selectedPimc.tts_99_wall_ms.toFixed(3)} vs {selectedThermal.tts_99_wall_ms.toFixed(3)} ms)
              </span>
            </div>
          </div>
        </div>

        {/* Right Column: Interactive Sweep Chart with Crossover Callout Marker (7 cols) */}
        <div className="lg:col-span-7 bg-[#020308]/85 border border-[rgba(250,249,245,0.08)] rounded-xl p-4 flex flex-col justify-between">
          <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
            <div>
              <span className="text-[10px] font-mono uppercase tracking-[0.18em] text-[#5eead4] block">
                {METRIC_META[metricId].label} vs. Barrier Hamming Width W (N = {systemN})
              </span>
              <p className="text-[11px] text-[rgba(250,249,245,0.55)]">{METRIC_META[metricId].description}</p>
            </div>
            <div className="flex items-center gap-4 text-[11px] font-mono">
              <span className="flex items-center gap-1.5 text-[#5eead4]">
                <span className="w-3 h-0.5 bg-[#5eead4] inline-block" /> PIMC / SQA (P=32)
              </span>
              <span className="flex items-center gap-1.5 text-[#f59e0b]">
                <span className="w-3 h-0.5 bg-[#f59e0b] inline-block" /> Thermal Gibbs (thrml)
              </span>
            </div>
          </div>

          {/* Interactive SVG Sweep Chart */}
          <svg
            viewBox={`0 0 ${svgW} ${svgH}`}
            className="w-full h-56 overflow-visible"
            role="img"
            aria-label={`${METRIC_META[metricId].label} across barrier widths W in {1, 2, 3, 4, 6} for N=${systemN}`}
          >
            {/* Shaded Quantum Tunneling Advantage Zone (left of crossoverX) */}
            {activeCrossover.crossover_detected && (
              <rect
                x={padL}
                y={padT}
                width={Math.max(0, Math.min(plotW, crossoverX - padL))}
                height={plotH}
                fill="rgba(94,234,212,0.05)"
              />
            )}

            {/* Horizontal Grid Lines & Y-Axis Labels */}
            {yTicks.map((t, idx) => {
              const y = toY(t.val);
              return (
                <g key={idx}>
                  <line
                    x1={padL}
                    y1={y}
                    x2={svgW - padR}
                    y2={y}
                    stroke="rgba(250,249,245,0.07)"
                    strokeDasharray="3 3"
                  />
                  <text
                    x={padL - 8}
                    y={y + 3}
                    textAnchor="end"
                    fill="rgba(250,249,245,0.55)"
                    fontSize="9.5"
                    fontFamily="monospace"
                  >
                    {t.label}
                  </text>
                </g>
              );
            })}

            {/* X-Axis Grid Lines & Width Tick Labels */}
            {WIDTH_OPTIONS.map((wVal) => {
              const x = toX(wVal);
              const isSel = wVal === selectedW;
              return (
                <g
                  key={wVal}
                  className="cursor-pointer"
                  onClick={() => setSelectedW(wVal)}
                >
                  <line
                    x1={x}
                    y1={padT}
                    x2={x}
                    y2={padT + plotH}
                    stroke={isSel ? "rgba(52,211,153,0.45)" : "rgba(250,249,245,0.06)"}
                    strokeWidth={isSel ? "1.5" : "1"}
                  />
                  <text
                    x={x}
                    y={svgH - 10}
                    textAnchor="middle"
                    fill={isSel ? "#34d399" : "rgba(250,249,245,0.65)"}
                    fontSize="10"
                    fontWeight={isSel ? "700" : "400"}
                    fontFamily="monospace"
                  >
                    W={wVal}
                  </text>
                </g>
              );
            })}

            {/* Vertical Crossover Threshold Callout Line */}
            {activeCrossover.crossover_detected && (
              <g>
                <line
                  x1={crossoverX}
                  y1={padT}
                  x2={crossoverX}
                  y2={padT + plotH}
                  stroke="#34d399"
                  strokeWidth="1.6"
                  strokeDasharray="4 3"
                />
                <rect
                  x={Math.min(svgW - padR - 118, Math.max(padL + 4, crossoverX - 58))}
                  y={padT + 4}
                  width="116"
                  height="18"
                  rx="4"
                  fill="#05070f"
                  stroke="#34d399"
                  strokeWidth="1"
                />
                <text
                  x={Math.min(svgW - padR - 60, Math.max(padL + 62, crossoverX))}
                  y={padT + 16}
                  textAnchor="middle"
                  fill="#34d399"
                  fontSize="9.5"
                  fontFamily="monospace"
                >
                  Crossover W_c = {activeCrossover.crossover_width_w.toFixed(2)}
                </text>
              </g>
            )}

            {/* Curves */}
            <path d={thermalPath} fill="none" stroke="#f59e0b" strokeWidth="2.2" />
            <path d={pimcPath} fill="none" stroke="#5eead4" strokeWidth="2.5" />

            {/* Interactive Data Points */}
            {thermalCurve.map((pt) => {
              const cx = toX(pt.barrier_width_w);
              const cy = toY(getMetricValue(pt));
              const isSel = pt.barrier_width_w === selectedW;
              return (
                <g
                  key={`th-${pt.barrier_width_w}`}
                  className="cursor-pointer"
                  onClick={() => setSelectedW(pt.barrier_width_w as BarrierWidthW)}
                >
                  <circle
                    cx={cx}
                    cy={cy}
                    r={isSel ? "5.5" : "3.8"}
                    fill="#f59e0b"
                    stroke="#05070f"
                    strokeWidth="1.5"
                  />
                </g>
              );
            })}

            {pimcCurve.map((pt) => {
              const cx = toX(pt.barrier_width_w);
              const cy = toY(getMetricValue(pt));
              const isSel = pt.barrier_width_w === selectedW;
              return (
                <g
                  key={`qa-${pt.barrier_width_w}`}
                  className="cursor-pointer"
                  onClick={() => setSelectedW(pt.barrier_width_w as BarrierWidthW)}
                >
                  <circle
                    cx={cx}
                    cy={cy}
                    r={isSel ? "6" : "4"}
                    fill="#5eead4"
                    stroke="#05070f"
                    strokeWidth="1.5"
                  />
                  {isSel && (
                    <circle
                      cx={cx}
                      cy={cy}
                      r="9"
                      fill="none"
                      stroke="#5eead4"
                      strokeOpacity="0.45"
                      strokeWidth="1.5"
                    />
                  )}
                </g>
              );
            })}
          </svg>

          {/* 4-Card KPI Readout Row for Active Width W */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 mt-3">
            <div className="bg-[#05070f] border border-[rgba(250,249,245,0.08)] rounded-lg p-2.5">
              <div className="text-[10px] font-mono text-[rgba(250,249,245,0.45)] uppercase">
                PIMC P_gs (W={selectedW})
              </div>
              <div className="text-base font-mono text-[#5eead4] mt-0.5">
                {(selectedPimc.ground_state_prob_pgs * 100).toFixed(1)}%
              </div>
              <div className="text-[10px] font-mono text-[rgba(250,249,245,0.45)]">
                vs {(selectedThermal.ground_state_prob_pgs * 100).toFixed(1)}% thermal
              </div>
            </div>

            <div className="bg-[#05070f] border border-[rgba(250,249,245,0.08)] rounded-lg p-2.5">
              <div className="text-[10px] font-mono text-[rgba(250,249,245,0.45)] uppercase">
                TTS_99 (Sweeps)
              </div>
              <div className="text-base font-mono text-[#34d399] mt-0.5">
                {selectedPimc.tts_99_sweeps.toFixed(1)}
              </div>
              <div className="text-[10px] font-mono text-[rgba(250,249,245,0.45)]">
                vs {selectedThermal.tts_99_sweeps.toFixed(1)} thermal
              </div>
            </div>

            <div className="bg-[#05070f] border border-[rgba(250,249,245,0.08)] rounded-lg p-2.5">
              <div className="text-[10px] font-mono text-[rgba(250,249,245,0.45)] uppercase">
                TTS_99 (RTX 5080 ms)
              </div>
              <div className="text-base font-mono text-[#faf9f5] mt-0.5">
                {selectedPimc.tts_99_wall_ms.toFixed(3)} ms
              </div>
              <div className="text-[10px] font-mono text-[rgba(250,249,245,0.45)]">
                vs {selectedThermal.tts_99_wall_ms.toFixed(3)} ms thermal
              </div>
            </div>

            <div className="bg-[#05070f] border border-[rgba(250,249,245,0.08)] rounded-lg p-2.5">
              <div className="text-[10px] font-mono text-[rgba(250,249,245,0.45)] uppercase">
                Active Metric W_c
              </div>
              <div className="text-base font-mono text-[#34d399] mt-0.5">
                W_c = {activeCrossover.crossover_width_w.toFixed(2)}
              </div>
              <div className="text-[10px] font-mono text-[rgba(250,249,245,0.45)]">
                Peak {crossovers.peak_pimc_tts_sweep_speedup_x.toFixed(1)}x sweep gain
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Complete Empirical Sweep Table & Hardware Runtime Metadata */}
      <div className="bg-[#020308]/75 border border-[rgba(250,249,245,0.08)] rounded-xl p-4">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
          <div className="flex items-center gap-2">
            <Cpu className="w-4 h-4 text-[#5eead4]" />
            <span className="text-[10px] font-mono uppercase tracking-[0.18em] text-[#5eead4]">
              Station 17 Empirical Sweep Table · N = {systemN} Spins ({activeSystem.effective_quantum_spins} Trotter Spins)
            </span>
          </div>
          <span className="text-[11px] font-mono text-[rgba(250,249,245,0.5)]">
            {gpuMeta?.gpu_name ?? "NVIDIA GeForce RTX 5080"} ({gpuMeta?.vram_total_mib ?? 16303} MiB VRAM) ·{" "}
            {gpuMeta?.host_ram_spec ?? "128GB DDR5"} · thrml {gpuMeta?.thrml_version ?? "0.1.4"}
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left font-mono text-xs border-collapse">
            <thead>
              <tr className="border-b border-[rgba(250,249,245,0.10)] text-[rgba(250,249,245,0.5)]">
                <th className="py-2 pr-3 font-normal">Width W</th>
                <th className="py-2 px-3 font-normal">Thermal P_gs</th>
                <th className="py-2 px-3 font-normal">PIMC P_gs</th>
                <th className="py-2 px-3 font-normal">Thermal TTS_99</th>
                <th className="py-2 px-3 font-normal">PIMC TTS_99</th>
                <th className="py-2 px-3 font-normal">Thermal (ms)</th>
                <th className="py-2 px-3 font-normal">PIMC (ms)</th>
                <th className="py-2 pl-3 font-normal text-right">Sweep / Wall Gain</th>
              </tr>
            </thead>
            <tbody>
              {thermalCurve.map((thPt, idx) => {
                const qaPt = pimcCurve[idx] ?? thPt;
                const wVal = thPt.barrier_width_w as BarrierWidthW;
                const swGain = thPt.tts_99_sweeps / Math.max(1e-6, qaPt.tts_99_sweeps);
                const wlGain = thPt.tts_99_wall_ms / Math.max(1e-6, qaPt.tts_99_wall_ms);
                const isSelected = wVal === selectedW;
                return (
                  <tr
                    key={wVal}
                    onClick={() => setSelectedW(wVal)}
                    className={`border-b border-[rgba(250,249,245,0.05)] cursor-pointer transition-colors ${
                      isSelected ? "bg-[rgba(94,234,212,0.08)]" : "hover:bg-[rgba(250,249,245,0.03)]"
                    }`}
                  >
                    <td className="py-2 pr-3 text-[#faf9f5] font-medium">
                      W = {wVal} ({2 * wVal} core)
                    </td>
                    <td className="py-2 px-3 text-amber-300">
                      {(thPt.ground_state_prob_pgs * 100).toFixed(1)}%
                    </td>
                    <td className="py-2 px-3 text-[#5eead4] font-medium">
                      {(qaPt.ground_state_prob_pgs * 100).toFixed(1)}%
                    </td>
                    <td className="py-2 px-3 text-[rgba(250,249,245,0.75)]">
                      {thPt.tts_99_sweeps.toFixed(1)} sw
                    </td>
                    <td className="py-2 px-3 text-[#34d399]">
                      {qaPt.tts_99_sweeps.toFixed(1)} sw
                    </td>
                    <td className="py-2 px-3 text-[rgba(250,249,245,0.75)]">
                      {thPt.tts_99_wall_ms.toFixed(3)} ms
                    </td>
                    <td className="py-2 px-3 text-[rgba(250,249,245,0.9)]">
                      {qaPt.tts_99_wall_ms.toFixed(3)} ms
                    </td>
                    <td className="py-2 pl-3 text-right">
                      <span className={swGain >= 1.0 ? "text-[#34d399]" : "text-amber-300"}>
                        {swGain.toFixed(2)}x sw
                      </span>{" "}
                      ·{" "}
                      <span className={wlGain >= 1.0 ? "text-[#5eead4]" : "text-[rgba(250,249,245,0.5)]"}>
                        {wlGain.toFixed(2)}x ms
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <div className="mt-3 pt-3 border-t border-[rgba(250,249,245,0.06)] flex flex-wrap items-center justify-between gap-2 text-[11px] font-mono text-[rgba(250,249,245,0.55)]">
          <span>
            Source: <code className="text-[#5eead4]">/stations/station_17_telemetry.json</code> · Seed{" "}
            {telemetry.experimental_design.deterministic_seed} · {telemetry.experimental_design.n_trials_per_point} trials/point
          </span>
          <Link
            href="/stations/17"
            className="text-[#5eead4] hover:underline inline-flex items-center gap-1"
          >
            Open standalone Station 17 apparatus <ArrowRight className="w-3 h-3" />
          </Link>
        </div>
      </div>
    </div>
  );
}

export default QuantumTunnelingCrossover;
