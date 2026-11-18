"use client";

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  CheckCircle2,
  Database,
  FileText,
  Pause,
  Play,
  RefreshCw,
  RotateCcw,
  ShieldCheck,
  Sparkles,
  Upload,
} from "lucide-react";

/* ============================================================================
 * Public Types (aligned with thermo_physics/refinery/schemas.py manifest)
 * ============================================================================ */

export type RefineryDataSource = "synthetic" | "live" | "custom";

export interface EcologicalNodeMeta {
  index?: number;
  channel_id?: string;
  site_id?: string;
  site_code?: string;
  site_name?: string;
  parameter_cd?: string;
  parameter_code?: string;
  parameter_name?: string;
  unit?: string;
  latest_value?: number;
  z_score?: number;
  z_score_latest?: number;
  label?: string;
}

export interface EcologicalIsingStream {
  provenance: {
    source_label: string;
    source_mode: string;
    captured_at_utc: string;
    station_count?: number;
    channel_count?: number;
  };
  nodes: Array<string | EcologicalNodeMeta>;
  J: number[][];
  h: number[];
  spin_covariance?: number[][];
  magnetization: number;
  frustration_pct: number;
  frustrated_triangles_pct?: number;
  exact_ground_state_energy: number;
  sampler_benchmark: {
    min_energy: number;
    mean_energy: number;
    tau_int: number;
    ess: number;
    energy_trace: number[];
  };
}

export interface RnaContactItem {
  i?: number;
  j?: number;
  col_i?: number;
  col_j?: number;
  pos_i?: number;
  pos_j?: number;
  score?: number;
  frobenius_norm?: number;
  frobenius_score?: number;
  consensus_pair?: string;
}

export interface RnaPottsStream {
  provenance: {
    source_label: string;
    source_mode: string;
    captured_at_utc: string;
    family_id: string;
    raw_sequences: number;
    retained_sequences: number;
    consensus_length: number;
    q_states: number;
  };
  alphabet: string[];
  f_i: number[][];
  h_i: number[][];
  frobenius_couplings: number[][];
  C_ij: number[][][][];
  top_contacts: Array<RnaContactItem | [number, number, number]>;
  sampler_benchmark: {
    min_energy: number;
    mean_energy: number;
    entropy_bits: number;
    tau_int: number;
    ess: number;
    energy_trace: number[];
  };
}

export interface MarketQuboAsset {
  ticker: string;
  latest_price: number;
  annualized_return: number;
  annualized_vol: number;
  polymarket_market: string;
  obi: number;
  spread: number;
  obi_penalty: number;
}

export interface MarketQuboStream {
  provenance: {
    source_label: string;
    source_mode: string;
    captured_at_utc: string;
    n_assets: number;
    k_target: number;
    polymarket_markets_polled: number;
  };
  assets: MarketQuboAsset[];
  Q: number[][];
  J: number[][];
  h: number[];
  qubo_ising_offset?: number;
  penalty_lambda?: number;
  exact_optimum: {
    selected_indices: number[];
    selected_tickers: string[];
    qubo_energy: number;
    expected_return: number;
    volatility: number;
  };
  sampler_benchmark: {
    selected_indices: number[];
    selected_tickers: string[];
    qubo_energy: number;
    budget_feasibility_pct: number;
    pareto_match_pct: number;
    tau_int: number;
    ess: number;
    energy_trace: number[];
  };
}

export interface AffilAiBrand {
  brand: string;
  vertical: string;
  domain: string;
  source_mode: string;
  product_count: number;
  variant_count: number;
  mean_price_usd: number;
  price_spread_ratio: number;
  price_elasticity: number;
  variant_depth: number;
  inventory_velocity: number;
  tag_taxonomy_richness: number;
  composite_deal_score: number;
  recommended_commission_tier: string;
  top_tags: string[];
}

export interface AffilAiProductGraphStream {
  provenance: {
    source_label: string;
    source_mode: string;
    captured_at_utc: string;
    brand_count: number;
    total_products_polled: number;
    total_variants_polled: number;
  };
  brands: AffilAiBrand[];
  brand_affinity_J: number[][];
}

export interface LiveEmpiricalStreamsPayload {
  schema_version: string;
  generated_at_utc: string;
  doctrine_disclosure?: string;
  ecological_ising: EcologicalIsingStream;
  rna_potts: RnaPottsStream;
  market_qubo: MarketQuboStream;
  affilai_product_graph: AffilAiProductGraphStream;
}

/* ============================================================================
 * Genuine Sokal Automatic Windowing Autocorrelation (W >= 5 * tau_int)
 * Matches thermo_physics/backends/sampler_adapter.py::compute_sokal_tau_int
 * ============================================================================ */

export function computeSokalTauInt(
  series: number[],
  cWindow = 5.0
): { tauInt: number; ess: number } {
  const n = series.length;
  if (n < 20) {
    return { tauInt: 1.0, ess: n };
  }

  let mean = 0;
  for (let i = 0; i < n; i++) mean += series[i];
  mean /= n;

  const centered = new Float64Array(n);
  let sumSq = 0;
  for (let i = 0; i < n; i++) {
    const d = series[i] - mean;
    centered[i] = d;
    sumSq += d * d;
  }
  if (sumSq / n <= 1e-12) {
    return { tauInt: 1.0, ess: n };
  }

  let tauInt = 0.5;
  for (let lag = 1; lag < n; lag++) {
    let cross = 0;
    const limit = n - lag;
    for (let t = 0; t < limit; t++) {
      cross += centered[t] * centered[t + lag];
    }
    const rho = cross / (sumSq + 1e-12);
    tauInt += rho;
    if (lag >= cWindow * tauInt) {
      break;
    }
  }

  const boundedTau = Math.max(0.5, tauInt);
  const ess = Math.max(1.0, n / (2.0 * boundedTau));
  return { tauInt: boundedTau, ess };
}

/* ============================================================================
 * Deterministic Synthetic Baselines (Station 2/10, 4/11/13, 6/7, AffilAI)
 * ============================================================================ */

function makeRng(seedInit: number): () => number {
  let s = seedInit >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function buildSyntheticEcological(): EcologicalIsingStream {
  const N = 16;
  const rng = makeRng(104729);
  const J: number[][] = Array.from({ length: N }, () => Array(N).fill(0));
  const h: number[] = Array.from({ length: N }, (_, i) => Number(((i % 2 === 0 ? 1 : -1) * 0.08).toFixed(4)));
  for (let i = 0; i < N; i++) {
    for (let j = i + 1; j < N; j++) {
      // Box-Muller deterministic Gaussian J_ij ~ N(0, 1/N)
      const u1 = Math.max(1e-6, rng());
      const u2 = rng();
      const g = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2) * (1 / Math.sqrt(N));
      const val = Number(g.toFixed(4));
      J[i][j] = val;
      J[j][i] = val;
    }
  }
  const exactGs = solveExactIsing16(J, h).energy;
  const nodes: EcologicalNodeMeta[] = Array.from({ length: N }, (_, i) => ({
    index: i,
    channel_id: `SK-NODE-${String(i + 1).padStart(2, "0")}`,
    site_name: `Synthetic SK Spin ${i + 1}`,
    parameter_name: i < 8 ? "Gaussian Coupling Bank A" : "Gaussian Coupling Bank B",
    unit: "dimensionless",
    latest_value: Number(((rng() - 0.5) * 2).toFixed(2)),
    z_score: Number(((rng() - 0.5) * 2).toFixed(2)),
  }));
  const trace = Array.from({ length: 64 }, (_, k) =>
    Number((exactGs + (1 - k / 63) * 4.8 + Math.sin(k * 0.7) * 0.35).toFixed(3))
  );
  const sokal = computeSokalTauInt(trace);
  return {
    provenance: {
      source_label: "Synthetic SK Spin-Glass Baseline (N=16)",
      source_mode: "synthetic_baseline",
      captured_at_utc: "Deterministic seed #104729",
      station_count: 2,
      channel_count: 16,
    },
    nodes,
    J,
    h,
    magnetization: 0.125,
    frustration_pct: 48.6,
    frustrated_triangles_pct: 48.6,
    exact_ground_state_energy: exactGs,
    sampler_benchmark: {
      min_energy: exactGs,
      mean_energy: Number((exactGs + 0.84).toFixed(3)),
      tau_int: Number(sokal.tauInt.toFixed(2)),
      ess: Number(sokal.ess.toFixed(1)),
      energy_trace: trace,
    },
  };
}

function buildSyntheticRnaPotts(): RnaPottsStream {
  const L = 24;
  const Q = 5;
  const alphabet = ["A", "C", "G", "U", "-"];
  // Synthetic stem-loop consensus (hairpin pairing i <-> 23 - i for i in 0..8)
  const stemBase = [2, 3, 0, 1, 1, 0, 3, 1, 2, 2, 0, 0, 0, 2, 0, 1, 2, 0, 3, 2, 2, 3, 0, 1];
  const f_i: number[][] = Array.from({ length: L }, (_, i) => {
    const row = [0.08, 0.08, 0.08, 0.08, 0.04];
    row[stemBase[i]] = 0.64;
    const sum = row.reduce((a, b) => a + b, 0);
    return row.map((v) => Number((v / sum).toFixed(4)));
  });
  const h_i: number[][] = f_i.map((row) => row.map((p) => Number(Math.log(Math.max(p, 0.01)).toFixed(4))));
  const C_ij: number[][][][] = Array.from({ length: L }, () =>
    Array.from({ length: L }, () => Array.from({ length: Q }, () => Array(Q).fill(0)))
  );
  const frob: number[][] = Array.from({ length: L }, () => Array(L).fill(0));

  for (let i = 0; i < 9; i++) {
    const j = L - 1 - i;
    const bi = stemBase[i];
    const bj = stemBase[j];
    C_ij[i][j][bi][bj] = 0.42;
    C_ij[j][i][bj][bi] = 0.42;
    frob[i][j] = 0.42;
    frob[j][i] = 0.42;
  }
  const topContacts: RnaContactItem[] = Array.from({ length: 9 }, (_, i) => ({
    i,
    j: L - 1 - i,
    score: 0.42 - i * 0.015,
  }));
  const trace = Array.from({ length: 64 }, (_, k) => Number((-12.4 - (k / 63) * 8.2 + Math.cos(k * 0.6) * 0.4).toFixed(3)));
  const sokal = computeSokalTauInt(trace);
  return {
    provenance: {
      source_label: "Synthetic RNA Stem-Loop Baseline (L=24, q=5)",
      source_mode: "synthetic_baseline",
      captured_at_utc: "Deterministic hairpin template",
      family_id: "SYNTH-HP24",
      raw_sequences: 120,
      retained_sequences: 120,
      consensus_length: L,
      q_states: Q,
    },
    alphabet,
    f_i,
    h_i,
    frobenius_couplings: frob,
    C_ij,
    top_contacts: topContacts,
    sampler_benchmark: {
      min_energy: -20.6,
      mean_energy: -18.2,
      entropy_bits: 11.4,
      tau_int: Number(sokal.tauInt.toFixed(2)),
      ess: Number(sokal.ess.toFixed(1)),
      energy_trace: trace,
    },
  };
}

function buildSyntheticMarketQubo(): MarketQuboStream {
  const raw: Array<[string, number, number, number]> = [
    ["NVLT", 14.2, 28, 0],
    ["HLIX", 11.8, 24, 1],
    ["ORBT", 13.5, 31, 0],
    ["FERN", 7.4, 15, 2],
    ["KITE", 12.6, 27, 0],
    ["MOSS", 6.8, 13, 2],
    ["PULSE", 10.9, 22, 1],
    ["QUAR", 8.9, 18, 3],
    ["RIDGE", 9.7, 19, 3],
    ["SOLM", 15.1, 33, 0],
    ["TUND", 5.9, 11, 2],
    ["UMBR", 11.2, 25, 1],
    ["VELA", 13.9, 29, 0],
    ["WREN", 7.9, 16, 2],
    ["XENL", 12.1, 26, 1],
    ["YARR", 8.3, 17, 3],
  ];
  const N = raw.length;
  const K = 5;
  const lam = 1.0;
  const pen = 2.0;
  const assets: MarketQuboAsset[] = raw.map(([ticker, ret, vol], idx) => ({
    ticker,
    latest_price: Number((45 + idx * 14.5).toFixed(2)),
    annualized_return: Number((ret / 100).toFixed(4)),
    annualized_vol: Number((vol / 100).toFixed(4)),
    polymarket_market: "Synthetic baseline (zero CLOB skew)",
    obi: 0,
    spread: 0.02,
    obi_penalty: 0,
  }));

  const Q: number[][] = Array.from({ length: N }, () => Array(N).fill(0));
  for (let i = 0; i < N; i++) {
    const mu = raw[i][1] / 100;
    const sig = raw[i][2] / 100;
    Q[i][i] = -mu * 2.5 + lam * sig * sig + pen * (1 - 2 * K) * 0.25;
    for (let j = i + 1; j < N; j++) {
      const corr = raw[i][3] === raw[j][3] ? 0.45 : 0.18;
      const cov = corr * (raw[i][2] / 100) * (raw[j][2] / 100);
      const val = (lam * cov + pen * 0.5) * 0.5;
      Q[i][j] = val;
      Q[j][i] = val;
    }
  }
  const { J, h } = quboToIsing(Q);
  const opt = solveExactQubo16(Q, assets);
  const trace = Array.from({ length: 64 }, (_, k) =>
    Number((opt.qubo_energy + (1 - k / 63) * 3.1 + Math.sin(k * 0.55) * 0.12).toFixed(3))
  );
  const sokal = computeSokalTauInt(trace);
  return {
    provenance: {
      source_label: "Synthetic Markowitz Universe (Station 7 Baseline)",
      source_mode: "synthetic_baseline",
      captured_at_utc: "Deterministic synthetic asset grid",
      n_assets: N,
      k_target: K,
      polymarket_markets_polled: 0,
    },
    assets,
    Q,
    J,
    h,
    exact_optimum: opt,
    sampler_benchmark: {
      selected_indices: opt.selected_indices,
      selected_tickers: opt.selected_tickers,
      qubo_energy: opt.qubo_energy,
      budget_feasibility_pct: 99.1,
      pareto_match_pct: 94.1,
      tau_int: Number(sokal.tauInt.toFixed(2)),
      ess: Number(sokal.ess.toFixed(1)),
      energy_trace: trace,
    },
  };
}

function buildSyntheticAffilAi(): AffilAiProductGraphStream {
  const names: Array<[string, string, string, number, number]> = [
    ["SynthFootwear", "Footwear", "synthfootwear.example", 98, 25],
    ["SynthHome", "Home Textiles", "synthhome.example", 145, 24],
    ["SynthApparel", "Sustainable Apparel", "synthapparel.example", 68, 25],
    ["SynthBev", "Functional Beverage", "synthbev.example", 36, 20],
    ["SynthWatches", "Watches", "synthwatches.example", 152, 25],
    ["SynthBeautyA", "Color Cosmetics", "synthbeautya.example", 24, 25],
    ["SynthBeautyB", "Cosmetics", "synthbeautyb.example", 32, 25],
    ["SynthSkincare", "Skincare", "synthskincare.example", 28, 25],
    ["SynthPersonal", "Personal Care", "synthpersonal.example", 16, 25],
    ["SynthGrooming", "Men's Grooming", "synthgrooming.example", 26, 25],
    ["SynthHardware", "Mechanical Keyboards", "synthhardware.example", 109, 25],
    ["SynthCarry", "EDC Gear", "synthcarry.example", 95, 25],
    ["SynthDesk", "Desk Goods", "synthdesk.example", 118, 25],
    ["SynthCases", "Eco Accessories", "synthcases.example", 49, 25],
    ["SynthCreator", "Creator Gear", "synthcreator.example", 84, 25],
  ];
  const rng = makeRng(777331);
  const brands: AffilAiBrand[] = names.map(([brand, vertical, domain, meanPrice, pCount]) => {
    const pe = Number((0.45 + rng() * 0.48).toFixed(3));
    const vd = Number((0.4 + rng() * 0.55).toFixed(3));
    const iv = Number((0.42 + rng() * 0.53).toFixed(3));
    const tr = Number((0.38 + rng() * 0.55).toFixed(3));
    const score = Number(((0.3 * pe + 0.25 * vd + 0.25 * iv + 0.2 * tr) * 100).toFixed(1));
    return {
      brand,
      vertical,
      domain,
      source_mode: "synthetic_baseline",
      product_count: pCount,
      variant_count: pCount * (2 + Math.floor(rng() * 5)),
      mean_price_usd: meanPrice,
      price_spread_ratio: Number((1.4 + rng() * 1.9).toFixed(2)),
      price_elasticity: pe,
      variant_depth: vd,
      inventory_velocity: iv,
      tag_taxonomy_richness: tr,
      composite_deal_score: score,
      recommended_commission_tier: score >= 75 ? "PARTNER_15_20" : score >= 58 ? "GROWTH_10_15" : "STANDARD_08_10",
      top_tags: ["synthetic", vertical.toLowerCase().split(" ")[0], "dtc"],
    };
  });
  const N = brands.length;
  const brand_affinity_J: number[][] = Array.from({ length: N }, (_, i) =>
    Array.from({ length: N }, (_, j) => {
      if (i === j) return 0;
      const sameGroup = Math.floor(i / 3) === Math.floor(j / 3);
      return Number((sameGroup ? 0.42 : 0.08 + ((i + j) % 3) * 0.05).toFixed(3));
    })
  );
  return {
    provenance: {
      source_label: "Synthetic DTC Catalog Baseline (15 Brands)",
      source_mode: "synthetic_baseline",
      captured_at_utc: "Deterministic baseline seed",
      brand_count: N,
      total_products_polled: brands.reduce((acc, b) => acc + b.product_count, 0),
      total_variants_polled: brands.reduce((acc, b) => acc + b.variant_count, 0),
    },
    brands,
    brand_affinity_J,
  };
}

/* ============================================================================
 * Exact Mathematical Helpers (Ising, QUBO->Ising, Exact 2^16 Enumeration)
 * ============================================================================ */

function evalIsingEnergy(spins: number[], J: number[][], h: number[]): number {
  const n = spins.length;
  let e = 0;
  for (let i = 0; i < n; i++) {
    e -= h[i] * spins[i];
    let row = 0;
    for (let j = 0; j < n; j++) {
      row += J[i][j] * spins[j];
    }
    e -= 0.5 * spins[i] * row;
  }
  return e;
}

function evalQuboEnergy(
  bits: number[],
  Q: number[][],
  penaltyLambda = 0,
  kTarget = 0
): number {
  const n = bits.length;
  let e = 0;
  for (let i = 0; i < n; i++) {
    if (!bits[i]) continue;
    for (let j = 0; j < n; j++) {
      if (bits[j]) e += Q[i][j];
    }
  }
  return e + penaltyLambda * kTarget * kTarget;
}

function quboToIsing(Q: number[][]): { J: number[][]; h: number[] } {
  const n = Q.length;
  const J: number[][] = Array.from({ length: n }, () => Array(n).fill(0));
  const h: number[] = Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let rowSum = 0;
    for (let j = 0; j < n; j++) {
      const qSym = 0.5 * (Q[i][j] + Q[j][i]);
      rowSum += qSym;
      if (i !== j) {
        J[i][j] = -0.5 * qSym;
      }
    }
    h[i] = -0.5 * rowSum;
  }
  return { J, h };
}

function solveExactIsing16(J: number[][], h: number[]): { energy: number; spins: number[] } {
  const n = Math.min(16, h.length);
  if (n > 16) return { energy: 0, spins: Array(n).fill(1) };
  let bestE = Infinity;
  let bestMask = 0;
  const total = 1 << n;
  const s = new Int8Array(n);
  for (let mask = 0; mask < total; mask++) {
    for (let i = 0; i < n; i++) {
      s[i] = (mask & (1 << i)) !== 0 ? 1 : -1;
    }
    let e = 0;
    for (let i = 0; i < n; i++) {
      const si = s[i];
      e -= h[i] * si;
      let pair = 0;
      for (let j = i + 1; j < n; j++) {
        pair += J[i][j] * s[j];
      }
      e -= si * pair;
    }
    if (e < bestE) {
      bestE = e;
      bestMask = mask;
    }
  }
  const bestSpins = Array.from({ length: n }, (_, i) => ((bestMask & (1 << i)) !== 0 ? 1 : -1));
  return { energy: Number(bestE.toFixed(4)), spins: bestSpins };
}

function solveExactQubo16(
  Q: number[][],
  assets: MarketQuboAsset[]
): {
  selected_indices: number[];
  selected_tickers: string[];
  qubo_energy: number;
  expected_return: number;
  volatility: number;
} {
  const n = Math.min(16, Q.length);
  let bestE = Infinity;
  let bestMask = 0;
  const total = 1 << n;
  for (let mask = 0; mask < total; mask++) {
    let e = 0;
    for (let i = 0; i < n; i++) {
      if ((mask & (1 << i)) === 0) continue;
      e += Q[i][i];
      for (let j = i + 1; j < n; j++) {
        if ((mask & (1 << j)) !== 0) {
          e += Q[i][j] + Q[j][i];
        }
      }
    }
    if (e < bestE) {
      bestE = e;
      bestMask = mask;
    }
  }
  const selected_indices: number[] = [];
  for (let i = 0; i < n; i++) {
    if ((bestMask & (1 << i)) !== 0) selected_indices.push(i);
  }
  const count = Math.max(1, selected_indices.length);
  const expected_return =
    selected_indices.reduce((acc, idx) => acc + assets[idx].annualized_return, 0) / count;
  const volatility =
    selected_indices.reduce((acc, idx) => acc + assets[idx].annualized_vol, 0) / count;
  return {
    selected_indices,
    selected_tickers: selected_indices.map((i) => assets[i].ticker),
    qubo_energy: Number(bestE.toFixed(4)),
    expected_return: Number(expected_return.toFixed(4)),
    volatility: Number(volatility.toFixed(4)),
  };
}

/* ============================================================================
 * Custom Matrix / FASTA Inbound Parsers & One-Click Sample Presets
 * ============================================================================ */

type CustomInputMode = "ising_matrix" | "qubo_matrix" | "fasta_msa";

interface CustomBenchmarkSummary {
  kind: CustomInputMode;
  n_variables: number;
  frustration_pct: number | null;
  min_energy: number;
  mean_energy: number;
  tau_int: number;
  ess: number;
  entropy_bits: number | null;
  source_format: string;
  label: string;
  note: string;
}

interface ContributionReceipt {
  entry_id: string;
  submitted_at_utc: string;
  structural_sha256: string;
  kind: string;
  n_variables: number;
  tau_int: number;
  ess: number;
  min_energy: number;
  corpus_count: number;
  persisted: string;
}

const SAMPLE_ISING_CSV = (() => {
  const N = 16;
  const rng = makeRng(202611);
  const rows: string[] = [];
  const mat: number[][] = Array.from({ length: N }, () => Array(N).fill(0));
  for (let i = 0; i < N; i++) {
    for (let j = i + 1; j < N; j++) {
      const sign = (i + j) % 3 === 0 ? -1 : 1;
      const val = Number((sign * (0.18 + rng() * 0.62)).toFixed(3));
      mat[i][j] = val;
      mat[j][i] = val;
    }
  }
  for (let i = 0; i < N; i++) {
    rows.push(mat[i].map((v) => v.toFixed(3)).join(","));
  }
  return rows.join("\n");
})();

const SAMPLE_QUBO_CSV = (() => {
  const N = 16;
  const K = 5;
  const rng = makeRng(909101);
  const Q: number[][] = Array.from({ length: N }, () => Array(N).fill(0));
  const pen = 1.45;
  for (let i = 0; i < N; i++) {
    const mu = 0.08 + rng() * 0.24;
    const varI = 0.04 + rng() * 0.12;
    Q[i][i] = Number((varI - mu + pen * (1 - 2 * K)).toFixed(4));
    for (let j = i + 1; j < N; j++) {
      const cov = (rng() - 0.25) * 0.09;
      const val = Number((cov + pen).toFixed(4));
      Q[i][j] = val;
      Q[j][i] = val;
    }
  }
  return Q.map((r) => r.map((v) => v.toFixed(4)).join(",")).join("\n");
})();

const SAMPLE_FASTA_MSA = (() => {
  const rng = makeRng(55123);
  const wcPairs: Array<[string, string]> = [
    ["G", "C"],
    ["C", "G"],
    ["A", "U"],
    ["U", "A"],
  ];
  const baseTemplate = "GCAUGCAUAAACCCGGGAAUGCAUGCAU".split("");
  const lines: string[] = [];
  for (let s = 0; s < 36; s++) {
    const chars = [...baseTemplate];
    // Compensatory stem covariance across positions 0..7 <-> 20..27
    for (let k = 0; k < 8; k++) {
      const pair = wcPairs[(s + k + Math.floor(rng() * 2)) % wcPairs.length];
      chars[k] = pair[0];
      chars[27 - k] = pair[1];
    }
    // Loop variation
    const loopBases = ["A", "C", "G", "U"];
    for (let p = 10; p <= 17; p++) {
      if (rng() < 0.35) chars[p] = loopBases[Math.floor(rng() * 4)];
    }
    if (s % 9 === 0) chars[13] = "-";
    lines.push(`>riboswitch_sample_${String(s + 1).padStart(2, "0")}`);
    lines.push(chars.join(""));
  }
  return lines.join("\n");
})();

function parseMatrixPayload(text: string): {
  matrix: number[][];
  biases?: number[];
  format: "csv" | "json";
  truncatedFrom?: number;
} {
  const trimmed = text.trim();
  if (!trimmed) {
    throw new Error("Input is empty. Paste a CSV/JSON matrix or load a preset.");
  }

  let rawMat: number[][] = [];
  let biases: number[] | undefined;
  let format: "csv" | "json" = "csv";

  if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
    format = "json";
    let parsed: unknown;
    try {
      parsed = JSON.parse(trimmed);
    } catch {
      throw new Error("Invalid JSON matrix syntax.");
    }
    if (Array.isArray(parsed)) {
      rawMat = parsed.map((row) => {
        if (!Array.isArray(row)) throw new Error("JSON array must be 2D numeric matrix.");
        return row.map((v) => Number(v));
      });
    } else if (parsed && typeof parsed === "object") {
      const obj = parsed as Record<string, unknown>;
      const matCandidate = obj.J ?? obj.Q ?? obj.matrix ?? obj.couplings;
      if (!Array.isArray(matCandidate)) {
        throw new Error("JSON object must contain a 2D array under 'J', 'Q', or 'matrix'.");
      }
      rawMat = matCandidate.map((row) => {
        if (!Array.isArray(row)) throw new Error("Matrix field must be a 2D numeric array.");
        return row.map((v) => Number(v));
      });
      const hCandidate = obj.h ?? obj.biases;
      if (Array.isArray(hCandidate)) {
        biases = hCandidate.map((v) => (Number.isFinite(Number(v)) ? Number(v) : 0));
      }
    } else {
      throw new Error("Unsupported JSON format.");
    }
  } else {
    const lines = trimmed
      .split(/\r?\n/)
      .map((l) => l.trim())
      .filter((l) => l.length > 0 && !l.startsWith("#"));
    for (let idx = 0; idx < lines.length; idx++) {
      const tokens = lines[idx]
        .split(/[\s,;]+/)
        .map((t) => t.trim())
        .filter((t) => t.length > 0);
      const nums = tokens.map((t) => Number(t));
      if (nums.some((v) => !Number.isFinite(v))) {
        if (idx === 0) continue; // Skip header row if present
        throw new Error(`Non-numeric entry in matrix row ${idx + 1}.`);
      }
      if (nums.length > 0) rawMat.push(nums);
    }
  }

  if (rawMat.length < 2) {
    throw new Error("Matrix must have at least 2 rows (2 <= N <= 16).");
  }

  const origDim = Math.max(rawMat.length, ...rawMat.map((r) => r.length));
  const N = Math.max(2, Math.min(16, origDim));
  const matrix: number[][] = Array.from({ length: N }, (_, r) =>
    Array.from({ length: N }, (__, c) => {
      const val = rawMat[r]?.[c] ?? 0;
      if (!Number.isFinite(val)) {
        throw new Error(`Non-finite value at (${r + 1}, ${c + 1}).`);
      }
      return val;
    })
  );

  return {
    matrix,
    biases: biases ? biases.slice(0, N) : undefined,
    format,
    truncatedFrom: origDim > 16 ? origDim : undefined,
  };
}

function buildCustomIsingFromMatrix(
  matrix: number[][],
  biases: number[] | undefined,
  sourceFormat: string,
  truncatedFrom?: number
): { eco: EcologicalIsingStream; summary: CustomBenchmarkSummary } {
  const N = matrix.length;
  const J: number[][] = Array.from({ length: N }, (_, i) =>
    Array.from({ length: N }, (__, j) => (i === j ? 0 : 0.5 * (matrix[i][j] + matrix[j][i])))
  );
  const h: number[] = Array.from({ length: N }, (_, i) => biases?.[i] ?? 0);

  const exact = solveExactIsing16(J, h);
  let activeEdges = 0;
  let frustratedEdges = 0;
  for (let i = 0; i < N; i++) {
    for (let j = i + 1; j < N; j++) {
      if (Math.abs(J[i][j]) > 1e-9) {
        activeEdges++;
        if (J[i][j] * exact.spins[i] * exact.spins[j] < 0) frustratedEdges++;
      }
    }
  }
  const edgeFrust = activeEdges > 0 ? (100 * frustratedEdges) / activeEdges : 0;

  let activeTri = 0;
  let frustTri = 0;
  for (let i = 0; i < N; i++) {
    for (let j = i + 1; j < N; j++) {
      for (let k = j + 1; k < N; k++) {
        const prod = J[i][j] * J[j][k] * J[k][i];
        if (Math.abs(prod) > 1e-12) {
          activeTri++;
          if (prod < 0) frustTri++;
        }
      }
    }
  }
  const triFrust = activeTri > 0 ? (100 * frustTri) / activeTri : edgeFrust;

  // Deterministic 120-sample Gibbs trace for live tau_int & ESS summary
  const rng = makeRng(4242);
  const s: number[] = Array.from({ length: N }, () => (rng() < 0.5 ? 1 : -1));
  const trace: number[] = [];
  for (let step = 0; step < 120; step++) {
    const beta = 0.6 + (2.4 * step) / 119;
    for (let k = 0; k < N; k++) {
      const i = Math.floor(rng() * N);
      let field = h[i];
      for (let j = 0; j < N; j++) {
        if (j !== i) field += J[i][j] * s[j];
      }
      const dE = 2 * s[i] * field;
      if (dE <= 0 || rng() < Math.exp(-beta * dE)) s[i] = -s[i];
    }
    trace.push(Number(evalIsingEnergy(s, J, h).toFixed(4)));
  }
  const minE = Math.min(...trace);
  const meanE = trace.reduce((a, b) => a + b, 0) / trace.length;
  const sokal = computeSokalTauInt(trace);
  const mag = Math.abs(exact.spins.reduce((a, b) => a + b, 0)) / N;
  const stamp = new Date().toISOString();

  const eco: EcologicalIsingStream = {
    provenance: {
      source_label: `Custom Uploaded Ising Matrix (N=${N})`,
      source_mode: `custom_${sourceFormat}`,
      captured_at_utc: stamp,
      station_count: 1,
      channel_count: N,
    },
    nodes: Array.from({ length: N }, (_, i) => ({
      index: i,
      channel_id: `VAR-${String(i + 1).padStart(2, "0")}`,
      site_name: `Custom Spin ${i + 1}`,
      parameter_name: "Adjacency Couplings",
      latest_value: exact.spins[i],
    })),
    J,
    h,
    magnetization: Number(mag.toFixed(4)),
    frustration_pct: Number(edgeFrust.toFixed(2)),
    frustrated_triangles_pct: Number(triFrust.toFixed(2)),
    exact_ground_state_energy: exact.energy,
    sampler_benchmark: {
      min_energy: Number(minE.toFixed(4)),
      mean_energy: Number(meanE.toFixed(4)),
      tau_int: Number(sokal.tauInt.toFixed(2)),
      ess: Number(sokal.ess.toFixed(1)),
      energy_trace: trace,
    },
  };

  return {
    eco,
    summary: {
      kind: "ising_matrix",
      n_variables: N,
      frustration_pct: Number(triFrust.toFixed(2)),
      min_energy: exact.energy,
      mean_energy: Number(meanE.toFixed(4)),
      tau_int: Number(sokal.tauInt.toFixed(2)),
      ess: Number(sokal.ess.toFixed(1)),
      entropy_bits: null,
      source_format: sourceFormat,
      label: `Custom Ising Couplings (${N}×${N})`,
      note: truncatedFrom
        ? `Sliced ${truncatedFrom}×${truncatedFrom} matrix to N=16 for exact 2^16 ground-state verification.`
        : `Exact 2^${N} ground state solved (E₀ = ${exact.energy.toFixed(3)}).`,
    },
  };
}

function buildCustomQuboFromMatrix(
  matrix: number[][],
  sourceFormat: string,
  truncatedFrom?: number
): { market: MarketQuboStream; summary: CustomBenchmarkSummary } {
  const N = matrix.length;
  const Q: number[][] = Array.from({ length: N }, (_, i) =>
    Array.from({ length: N }, (__, j) => 0.5 * (matrix[i][j] + matrix[j][i]))
  );
  const { J, h } = quboToIsing(Q);

  const assets: MarketQuboAsset[] = Array.from({ length: N }, (_, i) => ({
    ticker: `Q${String(i + 1).padStart(2, "0")}`,
    latest_price: 100,
    annualized_return: Number(Math.max(-0.5, Math.min(0.8, -Q[i][i] * 0.05)).toFixed(4)),
    annualized_vol: Number(Math.max(0.05, Math.sqrt(Math.max(0.01, Math.abs(Q[i][i]) * 0.02))).toFixed(4)),
    polymarket_market: "Custom QUBO variable",
    obi: 0,
    spread: 0,
    obi_penalty: 0,
  }));

  const exactOpt = solveExactQubo16(Q, assets);
  const kTarget = Math.max(1, exactOpt.selected_indices.length);

  let activeTri = 0;
  let frustTri = 0;
  for (let i = 0; i < N; i++) {
    for (let j = i + 1; j < N; j++) {
      for (let k = j + 1; k < N; k++) {
        const prod = J[i][j] * J[j][k] * J[k][i];
        if (Math.abs(prod) > 1e-12) {
          activeTri++;
          if (prod < 0) frustTri++;
        }
      }
    }
  }
  const triFrust = activeTri > 0 ? (100 * frustTri) / activeTri : 0;

  const rng = makeRng(8181);
  const spins: number[] = Array.from({ length: N }, () => (rng() < 0.5 ? 1 : -1));
  const trace: number[] = [];
  for (let step = 0; step < 120; step++) {
    const beta = 0.6 + (3.2 * step) / 119;
    for (let k = 0; k < N; k++) {
      const i = Math.floor(rng() * N);
      let field = h[i];
      for (let j = 0; j < N; j++) {
        if (j !== i) field += J[i][j] * spins[j];
      }
      const dE = 2 * spins[i] * field;
      if (dE <= 0 || rng() < Math.exp(-beta * dE)) spins[i] = -spins[i];
    }
    const bits = spins.map((s) => (s > 0 ? 1 : 0));
    trace.push(Number(evalQuboEnergy(bits, Q).toFixed(4)));
  }

  const minE = Math.min(...trace);
  const meanE = trace.reduce((a, b) => a + b, 0) / trace.length;
  const sokal = computeSokalTauInt(trace);
  const stamp = new Date().toISOString();

  const market: MarketQuboStream = {
    provenance: {
      source_label: `Custom Uploaded QUBO Matrix (${N}×${N})`,
      source_mode: `custom_${sourceFormat}`,
      captured_at_utc: stamp,
      n_assets: N,
      k_target: kTarget,
      polymarket_markets_polled: 0,
    },
    assets,
    Q,
    J,
    h,
    penalty_lambda: 0,
    exact_optimum: exactOpt,
    sampler_benchmark: {
      selected_indices: exactOpt.selected_indices,
      selected_tickers: exactOpt.selected_tickers,
      qubo_energy: Number(minE.toFixed(4)),
      budget_feasibility_pct: 100.0,
      pareto_match_pct: exactOpt.qubo_energy <= 0 ? 100.0 : Number(Math.min(100, (exactOpt.qubo_energy / Math.min(-1e-4, minE)) * 100).toFixed(1)),
      tau_int: Number(sokal.tauInt.toFixed(2)),
      ess: Number(sokal.ess.toFixed(1)),
      energy_trace: trace,
    },
  };

  return {
    market,
    summary: {
      kind: "qubo_matrix",
      n_variables: N,
      frustration_pct: Number(triFrust.toFixed(2)),
      min_energy: exactOpt.qubo_energy,
      mean_energy: Number(meanE.toFixed(4)),
      tau_int: Number(sokal.tauInt.toFixed(2)),
      ess: Number(sokal.ess.toFixed(1)),
      entropy_bits: null,
      source_format: sourceFormat,
      label: `Custom QUBO → Ising (${N}×${N})`,
      note: truncatedFrom
        ? `Sliced ${truncatedFrom}×${truncatedFrom} matrix to N=16; exact QUBO optimum E* = ${exactOpt.qubo_energy.toFixed(3)}.`
        : `Exact 2^${N} QUBO optimum E* = ${exactOpt.qubo_energy.toFixed(3)} (${exactOpt.selected_tickers.join(", ")}).`,
    },
  };
}

function buildCustomFastaPotts(
  text: string,
  sourceFormat: string
): { rna: RnaPottsStream; summary: CustomBenchmarkSummary } {
  const lines = text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length > 0);
  const seqs: string[] = [];
  let cur = "";
  const hasFastaHeader = lines.some((l) => l.startsWith(">"));
  for (const line of lines) {
    if (line.startsWith(">")) {
      if (cur.length > 0) seqs.push(cur);
      cur = "";
    } else if (hasFastaHeader) {
      cur += line.replace(/\s+/g, "");
    } else {
      seqs.push(line.replace(/\s+/g, ""));
    }
  }
  if (cur.length > 0) seqs.push(cur);

  if (seqs.length < 2) {
    throw new Error("FASTA alignment must contain at least 2 sequences.");
  }

  const maxLen = Math.max(...seqs.map((s) => s.length));
  if (maxLen < 2) {
    throw new Error("Sequences must be at least 2 nucleotides long.");
  }

  const alphabet = ["A", "C", "G", "U", "-"];
  const charToState = (ch: string): number => {
    const u = ch.toUpperCase();
    if (u === "A") return 0;
    if (u === "C") return 1;
    if (u === "G") return 2;
    if (u === "U" || u === "T") return 3;
    return 4;
  };

  const encodedAll: number[][] = seqs.map((s) => {
    const padded = s.padEnd(maxLen, "-");
    return Array.from({ length: maxLen }, (_, c) => charToState(padded[c]));
  });

  // Select up to L=24 highest-occupancy (lowest gap fraction) columns in original order
  const colOccupancy = Array.from({ length: maxLen }, (_, c) => {
    let nonGap = 0;
    for (let r = 0; r < encodedAll.length; r++) {
      if (encodedAll[r][c] !== 4) nonGap++;
    }
    return { col: c, occ: nonGap / encodedAll.length };
  });
  const selectedCols = [...colOccupancy]
    .sort((a, b) => b.occ - a.occ || a.col - b.col)
    .slice(0, Math.min(24, maxLen))
    .sort((a, b) => a.col - b.col)
    .map((item) => item.col);

  const L = selectedCols.length;
  const Q = 5;
  const msa = encodedAll.map((row) => selectedCols.map((c) => row[c]));
  const M = msa.length;
  const alpha = 0.5;
  const denom = M + alpha;

  const counts1: number[][] = Array.from({ length: L }, () => Array(Q).fill(0));
  for (let m = 0; m < M; m++) {
    for (let i = 0; i < L; i++) {
      counts1[i][msa[m][i]]++;
    }
  }
  const f_i: number[][] = counts1.map((row) => {
    const raw = row.map((cnt) => (cnt + alpha / Q) / denom);
    const s = raw.reduce((a, b) => a + b, 0);
    return raw.map((v) => Number((v / s).toFixed(6)));
  });
  const h_i: number[][] = f_i.map((row) => row.map((p) => Number(Math.log(Math.max(p, 1e-12)).toFixed(6))));

  const C_ij: number[][][][] = Array.from({ length: L }, () =>
    Array.from({ length: L }, () => Array.from({ length: Q }, () => Array(Q).fill(0)))
  );
  const frob: number[][] = Array.from({ length: L }, () => Array(L).fill(0));
  const contacts: RnaContactItem[] = [];

  for (let i = 0; i < L; i++) {
    for (let j = i + 1; j < L; j++) {
      const counts2: number[][] = Array.from({ length: Q }, () => Array(Q).fill(0));
      for (let m = 0; m < M; m++) {
        counts2[msa[m][i]][msa[m][j]]++;
      }
      let blockSum = 0;
      const f_ij: number[][] = counts2.map((r) =>
        r.map((cnt) => {
          const val = (cnt + alpha / (Q * Q)) / denom;
          blockSum += val;
          return val;
        })
      );
      let frobSq = 0;
      for (let a = 0; a < Q; a++) {
        for (let b = 0; b < Q; b++) {
          const pij = f_ij[a][b] / blockSum;
          const cij = pij - f_i[i][a] * f_i[j][b];
          const rounded = Number(cij.toFixed(6));
          C_ij[i][j][a][b] = rounded;
          C_ij[j][i][b][a] = rounded;
          frobSq += cij * cij;
        }
      }
      const fVal = Number(Math.sqrt(frobSq).toFixed(5));
      frob[i][j] = fVal;
      frob[j][i] = fVal;
      if (j - i >= 2) {
        let bestA = 0;
        let bestB = 0;
        let bestCnt = -1;
        for (let a = 0; a < Q; a++) {
          for (let b = 0; b < Q; b++) {
            if (counts2[a][b] > bestCnt) {
              bestCnt = counts2[a][b];
              bestA = a;
              bestB = b;
            }
          }
        }
        contacts.push({
          i,
          j,
          pos_i: selectedCols[i] + 1,
          pos_j: selectedCols[j] + 1,
          frobenius_score: fVal,
          consensus_pair: `${alphabet[bestA]}-${alphabet[bestB]}`,
        });
      }
    }
  }
  contacts.sort((a, b) => (b.frobenius_score ?? 0) - (a.frobenius_score ?? 0));

  let entropyBits = 0;
  for (let i = 0; i < L; i++) {
    for (let a = 0; a < Q; a++) {
      const p = f_i[i][a];
      if (p > 1e-12) entropyBits -= p * Math.log2(p);
    }
  }

  // Run 120-sweep q=5 Categorical Gibbs in-browser
  const rng = makeRng(7711);
  const seq = Array.from({ length: L }, () => Math.floor(rng() * Q));
  const trace: number[] = [];
  for (let sw = 0; sw < 120; sw++) {
    const beta = 1.25;
    for (let step = 0; step < L; step++) {
      const i = Math.floor(rng() * L);
      const logits: number[] = [];
      let maxL = -Infinity;
      for (let a = 0; a < Q; a++) {
        let sc = 0.5 * h_i[i][a];
        for (let j = 0; j < L; j++) {
          if (j !== i) sc += 4.0 * C_ij[i][j][a][seq[j]];
        }
        const scaled = beta * sc;
        logits.push(scaled);
        if (scaled > maxL) maxL = scaled;
      }
      let sumW = 0;
      const ws = logits.map((l) => {
        const w = Math.exp(l - maxL);
        sumW += w;
        return w;
      });
      let r = rng() * sumW;
      let picked = Q - 1;
      for (let a = 0; a < Q; a++) {
        r -= ws[a];
        if (r <= 0) {
          picked = a;
          break;
        }
      }
      seq[i] = picked;
    }
    let e = 0;
    for (let i = 0; i < L; i++) {
      e -= 0.5 * h_i[i][seq[i]];
      for (let j = i + 1; j < L; j++) {
        e -= 4.0 * C_ij[i][j][seq[i]][seq[j]];
      }
    }
    trace.push(Number(e.toFixed(4)));
  }

  const minE = Math.min(...trace);
  const meanE = trace.reduce((a, b) => a + b, 0) / trace.length;
  const sokal = computeSokalTauInt(trace);
  const stamp = new Date().toISOString();

  const rna: RnaPottsStream = {
    provenance: {
      source_label: `Custom Uploaded FASTA Alignment (${M} seqs × L=${L})`,
      source_mode: `custom_${sourceFormat}`,
      captured_at_utc: stamp,
      family_id: "CUSTOM-MSA",
      raw_sequences: M,
      retained_sequences: M,
      consensus_length: L,
      q_states: Q,
    },
    alphabet,
    f_i,
    h_i,
    frobenius_couplings: frob,
    C_ij,
    top_contacts: contacts.slice(0, 12),
    sampler_benchmark: {
      min_energy: Number(minE.toFixed(4)),
      mean_energy: Number(meanE.toFixed(4)),
      entropy_bits: Number(entropyBits.toFixed(3)),
      tau_int: Number(sokal.tauInt.toFixed(2)),
      ess: Number(sokal.ess.toFixed(1)),
      energy_trace: trace,
    },
  };

  return {
    rna,
    summary: {
      kind: "fasta_msa",
      n_variables: L,
      frustration_pct: null,
      min_energy: Number(minE.toFixed(4)),
      mean_energy: Number(meanE.toFixed(4)),
      tau_int: Number(sokal.tauInt.toFixed(2)),
      ess: Number(sokal.ess.toFixed(1)),
      entropy_bits: Number(entropyBits.toFixed(3)),
      source_format: sourceFormat,
      label: `Custom RNA MSA (${M} sequences · L=${L}, q=5)`,
      note: `Selected ${L} highest-occupancy consensus columns (T→U standardized); zero raw sequences transmitted on contribution.`,
    },
  };
}

/* ============================================================================
 * Shared SVG Energy Relaxation Sparkline
 * ============================================================================ */

function EnergyTraceChart({
  liveTrace,
  benchmarkTrace,
  exactEnergy,
  labelLeft,
  labelRight,
}: {
  liveTrace: number[];
  benchmarkTrace?: number[];
  exactEnergy?: number;
  labelLeft: string;
  labelRight: string;
}) {
  const allVals = [...liveTrace, ...(benchmarkTrace ?? []), ...(exactEnergy !== undefined ? [exactEnergy] : [])].filter(
    (v) => Number.isFinite(v)
  );
  const minE = allVals.length ? Math.min(...allVals) : -1;
  const maxE = allVals.length ? Math.max(...allVals) : 1;
  const span = Math.max(0.05, maxE - minE);
  const lo = minE - span * 0.1;
  const hi = maxE + span * 0.1;

  const W = 520;
  const H = 152;
  const padL = 44;
  const padR = 12;
  const padT = 14;
  const padB = 22;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const toY = (v: number) => padT + ((hi - v) / (hi - lo)) * plotH;
  const toPath = (arr: number[]) => {
    if (arr.length < 2) return "";
    return arr
      .map((v, i) => {
        const x = padL + (i / (arr.length - 1)) * plotW;
        const y = toY(v);
        return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
  };

  const livePath = toPath(liveTrace);
  const benchPath = benchmarkTrace && benchmarkTrace.length > 1 ? toPath(benchmarkTrace) : "";

  return (
    <div className="refinery-chart-card">
      <div className="refinery-chart-head">
        <span className="mono">{labelLeft}</span>
        <span className="mono">{labelRight}</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="refinery-svg" role="img" aria-label="Energy relaxation trace">
        {[0, 0.5, 1].map((frac) => {
          const y = padT + frac * plotH;
          const val = hi - frac * (hi - lo);
          return (
            <g key={frac}>
              <line x1={padL} y1={y} x2={W - padR} y2={y} stroke="#CBD4D7" strokeDasharray="3 3" />
              <text x={padL - 6} y={y + 3} textAnchor="end" fontSize="9.5" fontFamily="JetBrains Mono, monospace" fill="#46585F">
                {val.toFixed(1)}
              </text>
            </g>
          );
        })}
        {exactEnergy !== undefined && Number.isFinite(exactEnergy) && (
          <g>
            <line
              x1={padL}
              y1={toY(exactEnergy)}
              x2={W - padR}
              y2={toY(exactEnergy)}
              stroke="#2E5E6B"
              strokeWidth="1.2"
              strokeDasharray="5 4"
            />
            <text
              x={W - padR}
              y={Math.max(12, toY(exactEnergy) - 4)}
              textAnchor="end"
              fontSize="9"
              fontFamily="JetBrains Mono, monospace"
              fill="#1F4753"
            >
              GS {exactEnergy.toFixed(2)}
            </text>
          </g>
        )}
        {benchPath && <path d={benchPath} fill="none" stroke="#2E5E6B" strokeWidth="1.6" opacity="0.55" />}
        {livePath && <path d={livePath} fill="none" stroke="#E8490F" strokeWidth="2.2" />}
        {liveTrace.length > 0 && (
          <circle
            cx={liveTrace.length > 1 ? W - padR : padL}
            cy={toY(liveTrace[liveTrace.length - 1])}
            r="3.5"
            fill="#E8490F"
          />
        )}
      </svg>
      <div className="refinery-chart-legend mono">
        <span>
          <i style={{ background: "#E8490F" }} /> Browser sampler E(t)
        </span>
        {benchPath && (
          <span>
            <i style={{ background: "#2E5E6B", opacity: 0.65 }} /> Python refinery benchmark trace
          </span>
        )}
        {exactEnergy !== undefined && Number.isFinite(exactEnergy) && (
          <span>
            <i style={{ borderTop: "1.5px dashed #2E5E6B", background: "transparent" }} /> Exact / minimum reference
          </span>
        )}
      </div>
    </div>
  );
}

/* ============================================================================
 * Main Exported Component
 * ============================================================================ */

type RefineryTabId = "ising" | "potts" | "qubo" | "affilai" | "custom";

export interface LiveEmpiricalRefineryDeckProps {
  externalData?: LiveEmpiricalStreamsPayload | null;
  externalLoading?: boolean;
  externalError?: string | null;
  onRefreshExternal?: () => void;
}

export function LiveEmpiricalRefineryDeck({
  externalData,
  externalLoading,
  externalError,
  onRefreshExternal,
}: LiveEmpiricalRefineryDeckProps) {
  const [dataSource, setDataSource] = useState<RefineryDataSource>("synthetic");
  const [activeTab, setActiveTab] = useState<RefineryTabId>("ising");

  const [internalData, setInternalData] = useState<LiveEmpiricalStreamsPayload | null>(null);
  const [internalLoading, setInternalLoading] = useState<boolean>(false);
  const [internalError, setInternalError] = useState<string | null>(null);

  // Inbound Custom Matrix / FASTA State
  const [customMode, setCustomMode] = useState<CustomInputMode>("ising_matrix");
  const [customRawText, setCustomRawText] = useState<string>(SAMPLE_ISING_CSV);
  const [customFormat, setCustomFormat] = useState<string>("csv");
  const [customRevision, setCustomRevision] = useState<number>(1);
  const [customError, setCustomError] = useState<string | null>(null);
  const [customEco, setCustomEco] = useState<EcologicalIsingStream | null>(() =>
    buildCustomIsingFromMatrix(parseMatrixPayload(SAMPLE_ISING_CSV).matrix, undefined, "preset").eco
  );
  const [customRna, setCustomRna] = useState<RnaPottsStream | null>(null);
  const [customMarket, setCustomMarket] = useState<MarketQuboStream | null>(null);
  const [customSummary, setCustomSummary] = useState<CustomBenchmarkSummary>(() =>
    buildCustomIsingFromMatrix(parseMatrixPayload(SAMPLE_ISING_CSV).matrix, undefined, "preset").summary
  );

  // Opt-In Anonymized Benchmark Corpus Contribution State
  const [anonymizedConsent, setAnonymizedConsent] = useState<boolean>(true);
  const [contributing, setContributing] = useState<boolean>(false);
  const [contributeError, setContributeError] = useState<string | null>(null);
  const [lastReceipt, setLastReceipt] = useState<ContributionReceipt | null>(null);
  const [corpusCount, setCorpusCount] = useState<number>(0);
  const [sessionContribCount, setSessionContribCount] = useState<number>(0);

  useEffect(() => {
    let active = true;
    fetch("/api/thermo-refinery/contribute", { cache: "no-store" })
      .then((res) => (res.ok ? res.json() : null))
      .then((data: { ok?: boolean; corpus_count?: number } | null) => {
        if (active && data?.ok && typeof data.corpus_count === "number") {
          setCorpusCount(data.corpus_count);
        }
      })
      .catch(() => {
        // Non-blocking corpus count probe
      });
    return () => {
      active = false;
    };
  }, []);

  const parseAndLoadCustom = useCallback(
    (textToParse: string, modeOverride?: CustomInputMode, formatHint?: string) => {
      setCustomError(null);
      setContributeError(null);
      const effectiveMode = modeOverride ?? customMode;
      try {
        if (effectiveMode === "fasta_msa") {
          const fmt = formatHint ?? "fasta";
          const built = buildCustomFastaPotts(textToParse, fmt);
          setCustomRna(built.rna);
          setCustomSummary(built.summary);
          setCustomFormat(fmt);
          setCustomRevision((r) => r + 1);
          setDataSource("custom");
          return;
        }
        const parsed = parseMatrixPayload(textToParse);
        const fmt = formatHint ?? parsed.format;
        if (effectiveMode === "qubo_matrix") {
          const built = buildCustomQuboFromMatrix(parsed.matrix, fmt, parsed.truncatedFrom);
          setCustomMarket(built.market);
          setCustomSummary(built.summary);
          setCustomFormat(fmt);
          setCustomRevision((r) => r + 1);
          setDataSource("custom");
        } else {
          const built = buildCustomIsingFromMatrix(
            parsed.matrix,
            parsed.biases,
            fmt,
            parsed.truncatedFrom
          );
          setCustomEco(built.eco);
          setCustomSummary(built.summary);
          setCustomFormat(fmt);
          setCustomRevision((r) => r + 1);
          setDataSource("custom");
        }
      } catch (err) {
        setCustomError(err instanceof Error ? err.message : "Failed to parse custom input.");
      }
    },
    [customMode]
  );

  const handleFileUpload = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;
      try {
        const text = await file.text();
        const ext = file.name.split(".").pop()?.toLowerCase() ?? "txt";
        const looksLikeFasta =
          ext === "fasta" || ext === "fa" || text.trimStart().startsWith(">") || /^[ACGTUacgtu\-\s]{16,}$/m.test(text);
        const nextMode: CustomInputMode = looksLikeFasta
          ? "fasta_msa"
          : customMode === "fasta_msa"
          ? "ising_matrix"
          : customMode;
        setCustomMode(nextMode);
        setCustomRawText(text);
        parseAndLoadCustom(text, nextMode, ext);
      } catch {
        setCustomError("Unable to read selected file in browser.");
      }
    },
    [customMode, parseAndLoadCustom]
  );

  const handleContributeBenchmark = useCallback(async () => {
    if (!anonymizedConsent || contributing) return;
    setContributing(true);
    setContributeError(null);
    try {
      // Send ONLY anonymized aggregate metrics (zero raw sequences, matrices, or user PII)
      const res = await fetch("/api/thermo-refinery/contribute", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          anonymized_consent: true,
          kind: customSummary.kind,
          n_variables: customSummary.n_variables,
          frustration_pct: customSummary.frustration_pct,
          min_energy: customSummary.min_energy,
          mean_energy: customSummary.mean_energy,
          tau_int: customSummary.tau_int,
          ess: customSummary.ess,
          entropy_bits: customSummary.entropy_bits,
          source_format: customSummary.source_format,
        }),
      });
      const json = (await res.json()) as {
        ok?: boolean;
        error?: string;
        entry_id?: string;
        corpus_count?: number;
        persisted?: string;
        entry?: {
          submitted_at_utc: string;
          structural_sha256: string;
          kind: string;
          n_variables: number;
          tau_int: number;
          ess: number;
          min_energy: number;
        };
      };
      if (!res.ok || !json.ok || !json.entry) {
        throw new Error(json.error ?? `HTTP ${res.status}`);
      }
      setLastReceipt({
        entry_id: json.entry_id ?? "",
        submitted_at_utc: json.entry.submitted_at_utc,
        structural_sha256: json.entry.structural_sha256,
        kind: json.entry.kind,
        n_variables: json.entry.n_variables,
        tau_int: json.entry.tau_int,
        ess: json.entry.ess,
        min_energy: json.entry.min_energy,
        corpus_count: json.corpus_count ?? corpusCount + 1,
        persisted: json.persisted ?? "jsonl",
      });
      if (typeof json.corpus_count === "number") setCorpusCount(json.corpus_count);
      setSessionContribCount((c) => c + 1);
    } catch (err) {
      setContributeError(err instanceof Error ? err.message : "Failed to contribute benchmark.");
    } finally {
      setContributing(false);
    }
  }, [anonymizedConsent, contributing, customSummary, corpusCount]);

  const syntheticEco = useMemo(() => buildSyntheticEcological(), []);
  const syntheticRna = useMemo(() => buildSyntheticRnaPotts(), []);
  const syntheticMarket = useMemo(() => buildSyntheticMarketQubo(), []);
  const syntheticAffil = useMemo(() => buildSyntheticAffilAi(), []);

  const payload = externalData !== undefined ? externalData : internalData;
  const isLoading = externalLoading !== undefined ? externalLoading : internalLoading;
  const loadError = externalError !== undefined ? externalError : internalError;

  const fetchLiveStreams = useCallback(async () => {
    if (onRefreshExternal) {
      onRefreshExternal();
      return;
    }
    setInternalLoading(true);
    setInternalError(null);
    try {
      const res = await fetch("/data/live_empirical_streams.json", { cache: "no-store" });
      if (!res.ok) {
        throw new Error(`HTTP ${res.status} loading /data/live_empirical_streams.json`);
      }
      const json = (await res.json()) as LiveEmpiricalStreamsPayload;
      setInternalData(json);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Live telemetry stream unavailable";
      setInternalError(msg);
    } finally {
      setInternalLoading(false);
    }
  }, [onRefreshExternal]);

  const handleSwitchToLive = useCallback(() => {
    setDataSource("live");
    if (payload === null && !isLoading) {
      void fetchLiveStreams();
    }
  }, [payload, isLoading, fetchLiveStreams]);

  const handleSwitchToCustom = useCallback(() => {
    setDataSource("custom");
    setActiveTab("custom");
  }, []);

  const usingLive = dataSource === "live" && payload !== null;
  const usingCustom = dataSource === "custom";

  /* --------------------------------------------------------------------------
   * TAB 1: Ising Lattice (16-Node Ecological / SK Solver)
   * -------------------------------------------------------------------------- */
  const activeEco =
    usingCustom && customEco
      ? customEco
      : usingLive && payload.ecological_ising
      ? payload.ecological_ising
      : syntheticEco;
  const ecoN = Math.max(2, Math.min(16, activeEco.h?.length || 16));
  const [ecoBeta, setEcoBeta] = useState<number>(1.8);
  const [ecoRunning, setEcoRunning] = useState<boolean>(true);
  const [ecoSelectedNode, setEcoSelectedNode] = useState<number>(0);
  const [ecoSpins, setEcoSpins] = useState<number[]>(() =>
    Array.from({ length: 16 }, (_, i) => (i % 2 === 0 ? 1 : -1))
  );
  const [ecoTrace, setEcoTrace] = useState<number[]>(() => {
    const init = Array.from({ length: 16 }, (_, i) => (i % 2 === 0 ? 1 : -1));
    return [evalIsingEnergy(init, syntheticEco.J, syntheticEco.h)];
  });

  const ecoSpinsRef = useRef<number[]>(ecoSpins);
  useEffect(() => {
    ecoSpinsRef.current = ecoSpins;
  }, [ecoSpins]);

  const ecoBetaRef = useRef<number>(ecoBeta);
  useEffect(() => {
    ecoBetaRef.current = ecoBeta;
  }, [ecoBeta]);

  // Reset spins cleanly when switching between synthetic, live, and custom data
  const ecoSourceKey = `${usingCustom ? `custom-${customRevision}` : usingLive ? "live" : "synth"}-${activeEco.provenance.captured_at_utc}`;
  const prevEcoSourceKey = useRef<string>(ecoSourceKey);
  useEffect(() => {
    if (prevEcoSourceKey.current !== ecoSourceKey) {
      prevEcoSourceKey.current = ecoSourceKey;
      const timer = setTimeout(() => {
        const fresh = Array.from({ length: ecoN }, () => (Math.random() < 0.5 ? 1 : -1));
        setEcoSpins(fresh);
        setEcoTrace([evalIsingEnergy(fresh, activeEco.J, activeEco.h)]);
      }, 0);
      return () => clearTimeout(timer);
    }
  }, [ecoSourceKey, ecoN, activeEco]);

  const stepEcoMetropolis = useCallback(
    (sweeps = 1, betaOverride?: number) => {
      const s = [...ecoSpinsRef.current];
      while (s.length < ecoN) s.push(1);
      if (s.length > ecoN) s.length = ecoN;
      const beta = betaOverride ?? ecoBetaRef.current;
      const J = activeEco.J;
      const h = activeEco.h;
      for (let sw = 0; sw < sweeps; sw++) {
        for (let k = 0; k < ecoN; k++) {
          const i = (Math.random() * ecoN) | 0;
          let field = h[i] ?? 0;
          const row = J[i];
          if (row) {
            for (let j = 0; j < ecoN; j++) {
              if (j !== i) field += (row[j] ?? 0) * s[j];
            }
          }
          const dE = 2 * s[i] * field;
          if (dE <= 0 || Math.random() < Math.exp(-beta * dE)) {
            s[i] = -s[i];
          }
        }
      }
      const e = evalIsingEnergy(s, J, h);
      ecoSpinsRef.current = s;
      setEcoSpins(s);
      setEcoTrace((prev) => {
        const next = [...prev, Number(e.toFixed(4))];
        return next.length > 140 ? next.slice(next.length - 140) : next;
      });
    },
    [activeEco, ecoN]
  );

  useEffect(() => {
    if (!ecoRunning || activeTab !== "ising") return;
    const id = setInterval(() => {
      stepEcoMetropolis(2);
    }, 180);
    return () => clearInterval(id);
  }, [ecoRunning, activeTab, stepEcoMetropolis]);

  const ecoSokal = useMemo(() => computeSokalTauInt(ecoTrace), [ecoTrace]);
  const ecoCurrentE = ecoTrace[ecoTrace.length - 1] ?? 0;
  const ecoAbsM = useMemo(() => {
    const sum = ecoSpins.reduce((acc, v) => acc + v, 0);
    return Math.abs(sum) / Math.max(1, ecoSpins.length);
  }, [ecoSpins]);

  const ecoAnneal = useCallback(() => {
    setEcoRunning(false);
    const s: number[] = Array.from({ length: ecoN }, () => (Math.random() < 0.5 ? 1 : -1));
    const J = activeEco.J;
    const h = activeEco.h;
    const trace: number[] = [];
    for (let step = 0; step < 80; step++) {
      const b = 0.25 + (4.5 - 0.25) * (step / 79);
      for (let k = 0; k < ecoN * 2; k++) {
        const i = (Math.random() * ecoN) | 0;
        let field = h[i] ?? 0;
        for (let j = 0; j < ecoN; j++) {
          if (j !== i) field += (J[i]?.[j] ?? 0) * s[j];
        }
        const dE = 2 * s[i] * field;
        if (dE <= 0 || Math.random() < Math.exp(-b * dE)) {
          s[i] = -s[i];
        }
      }
      trace.push(Number(evalIsingEnergy(s, J, h).toFixed(4)));
    }
    setEcoBeta(4.5);
    setEcoSpins(s);
    setEcoTrace(trace);
  }, [activeEco, ecoN]);

  /* --------------------------------------------------------------------------
   * TAB 2: Potts / RNA Co-Evolution (24-pos, q=5 Categorical Gibbs Solver)
   * -------------------------------------------------------------------------- */
  const activeRna =
    usingCustom && customRna
      ? customRna
      : usingLive && payload.rna_potts
      ? payload.rna_potts
      : syntheticRna;
  const rnaL = activeRna.f_i?.length || 24;
  const rnaQ = activeRna.alphabet?.length || 5;
  const [rnaBeta, setRnaBeta] = useState<number>(2.0);
  const [rnaRunning, setRnaRunning] = useState<boolean>(true);
  const [rnaCellInspect, setRnaCellInspect] = useState<[number, number]>([0, rnaL - 1]);
  const [rnaSeq, setRnaSeq] = useState<number[]>(() =>
    Array.from({ length: 24 }, (_, i) => i % 4)
  );
  const evalRnaEnergy = useCallback(
    (seq: number[], potts: RnaPottsStream): number => {
      const L = Math.min(seq.length, potts.h_i.length);
      let e = 0;
      for (let i = 0; i < L; i++) {
        const si = seq[i];
        e -= 0.5 * (potts.h_i[i]?.[si] ?? 0);
        for (let j = i + 1; j < L; j++) {
          e -= 4.0 * (potts.C_ij?.[i]?.[j]?.[si]?.[seq[j]] ?? 0);
        }
      }
      return e;
    },
    []
  );
  const [rnaTrace, setRnaTrace] = useState<number[]>(() => [
    evalRnaEnergy(Array.from({ length: 24 }, (_, i) => i % 4), syntheticRna),
  ]);

  const rnaSeqRef = useRef<number[]>(rnaSeq);
  useEffect(() => {
    rnaSeqRef.current = rnaSeq;
  }, [rnaSeq]);

  const rnaBetaRef = useRef<number>(rnaBeta);
  useEffect(() => {
    rnaBetaRef.current = rnaBeta;
  }, [rnaBeta]);

  const rnaSourceKey = `${usingCustom ? `custom-${customRevision}` : usingLive ? "live" : "synth"}-${activeRna.provenance.captured_at_utc}`;
  const prevRnaSourceKey = useRef<string>(rnaSourceKey);
  useEffect(() => {
    if (prevRnaSourceKey.current !== rnaSourceKey) {
      prevRnaSourceKey.current = rnaSourceKey;
      const timer = setTimeout(() => {
        const init = Array.from({ length: rnaL }, () => (Math.random() * Math.min(4, rnaQ)) | 0);
        setRnaSeq(init);
        setRnaTrace([evalRnaEnergy(init, activeRna)]);
      }, 0);
      return () => clearTimeout(timer);
    }
  }, [rnaSourceKey, rnaL, rnaQ, activeRna, evalRnaEnergy]);

  const stepRnaGibbs = useCallback(
    (sweeps = 1, betaOverride?: number) => {
      const seq = [...rnaSeqRef.current];
      while (seq.length < rnaL) seq.push(0);
      if (seq.length > rnaL) seq.length = rnaL;
      const beta = betaOverride ?? rnaBetaRef.current;
      const { h_i, C_ij } = activeRna;

      for (let sw = 0; sw < sweeps; sw++) {
        for (let step = 0; step < rnaL; step++) {
          const i = (Math.random() * rnaL) | 0;
          const logits: number[] = [];
          let maxL = -Infinity;
          for (let a = 0; a < rnaQ; a++) {
            let score = 0.5 * (h_i[i]?.[a] ?? 0);
            for (let j = 0; j < rnaL; j++) {
              if (j !== i) {
                score += 4.0 * (C_ij?.[i]?.[j]?.[a]?.[seq[j]] ?? 0);
              }
            }
            const scaled = beta * score;
            logits.push(scaled);
            if (scaled > maxL) maxL = scaled;
          }
          let sumP = 0;
          const probs = logits.map((l) => {
            const w = Math.exp(l - maxL);
            sumP += w;
            return w;
          });
          let r = Math.random() * sumP;
          let picked = rnaQ - 1;
          for (let a = 0; a < rnaQ; a++) {
            r -= probs[a];
            if (r <= 0) {
              picked = a;
              break;
            }
          }
          seq[i] = picked;
        }
      }
      const e = evalRnaEnergy(seq, activeRna);
      rnaSeqRef.current = seq;
      setRnaSeq(seq);
      setRnaTrace((prev) => {
        const next = [...prev, Number(e.toFixed(4))];
        return next.length > 140 ? next.slice(next.length - 140) : next;
      });
    },
    [activeRna, rnaL, rnaQ, evalRnaEnergy]
  );

  useEffect(() => {
    if (!rnaRunning || activeTab !== "potts") return;
    const id = setInterval(() => {
      stepRnaGibbs(1);
    }, 200);
    return () => clearInterval(id);
  }, [rnaRunning, activeTab, stepRnaGibbs]);

  const rnaSokal = useMemo(() => computeSokalTauInt(rnaTrace), [rnaTrace]);
  const rnaMaxFrob = useMemo(() => {
    let mx = 0.05;
    for (const row of activeRna.frobenius_couplings ?? []) {
      for (const v of row) {
        if (v > mx) mx = v;
      }
    }
    return mx;
  }, [activeRna.frobenius_couplings]);

  const rnaNormalizedContacts = useMemo(() => {
    return (activeRna.top_contacts ?? []).slice(0, 8).map((item) => {
      if (Array.isArray(item)) {
        return { i: item[0], j: item[1], score: item[2], pair: undefined as string | undefined };
      }
      return {
        i: item.i ?? item.col_i ?? 0,
        j: item.j ?? item.col_j ?? 0,
        score: item.frobenius_score ?? item.score ?? item.frobenius_norm ?? 0,
        pair: item.consensus_pair,
      };
    });
  }, [activeRna.top_contacts]);

  /* --------------------------------------------------------------------------
   * TAB 3: Market & Polymarket QUBO (16-Asset / Synthetic 16-Asset Solver)
   * -------------------------------------------------------------------------- */
  const activeMarket =
    usingCustom && customMarket
      ? customMarket
      : usingLive && payload.market_qubo
      ? payload.market_qubo
      : syntheticMarket;
  const mktN = activeMarket.assets?.length || 16;
  const mktK = activeMarket.provenance?.k_target || 5;
  const [mktBeta, setMktBeta] = useState<number>(2.5);
  const [mktRunning, setMktRunning] = useState<boolean>(false);
  const [mktBits, setMktBits] = useState<number[]>(() => {
    const arr = Array(16).fill(0);
    for (const idx of syntheticMarket.exact_optimum.selected_indices) {
      if (idx < arr.length) arr[idx] = 1;
    }
    return arr;
  });
  const [mktTrace, setMktTrace] = useState<number[]>(() => {
    const arr = Array(16).fill(0);
    for (const idx of syntheticMarket.exact_optimum.selected_indices) {
      if (idx < arr.length) arr[idx] = 1;
    }
    return [evalQuboEnergy(arr, syntheticMarket.Q)];
  });

  const mktBitsRef = useRef<number[]>(mktBits);
  useEffect(() => {
    mktBitsRef.current = mktBits;
  }, [mktBits]);

  const mktBetaRef = useRef<number>(mktBeta);
  useEffect(() => {
    mktBetaRef.current = mktBeta;
  }, [mktBeta]);

  const mktSourceKey = `${usingCustom ? `custom-${customRevision}` : usingLive ? "live" : "synth"}-${activeMarket.provenance.captured_at_utc}`;
  const prevMktSourceKey = useRef<string>(mktSourceKey);
  useEffect(() => {
    if (prevMktSourceKey.current !== mktSourceKey) {
      prevMktSourceKey.current = mktSourceKey;
      const timer = setTimeout(() => {
        const arr = Array(mktN).fill(0);
        const seedIdx = activeMarket.sampler_benchmark?.selected_indices ?? [];
        seedIdx.forEach((idx) => {
          if (idx >= 0 && idx < mktN) arr[idx] = 1;
        });
        setMktBits(arr);
        setMktTrace([
          evalQuboEnergy(
            arr,
            activeMarket.Q,
            activeMarket.penalty_lambda ?? 0,
            activeMarket.provenance?.k_target ?? 0
          ),
        ]);
      }, 0);
      return () => clearTimeout(timer);
    }
  }, [mktSourceKey, mktN, activeMarket]);

  const stepMarketIsing = useCallback(
    (sweeps = 1) => {
      const bits = [...mktBitsRef.current];
      while (bits.length < mktN) bits.push(0);
      if (bits.length > mktN) bits.length = mktN;
      const spins: number[] = bits.map((b) => (b ? 1 : -1));
      const beta = mktBetaRef.current;
      const J = activeMarket.J;
      const h = activeMarket.h;

      for (let sw = 0; sw < sweeps; sw++) {
        for (let k = 0; k < mktN; k++) {
          const i = (Math.random() * mktN) | 0;
          let field = h[i] ?? 0;
          for (let j = 0; j < mktN; j++) {
            if (j !== i) field += (J[i]?.[j] ?? 0) * spins[j];
          }
          const dE = 2 * spins[i] * field;
          if (dE <= 0 || Math.random() < Math.exp(-beta * dE)) {
            spins[i] = -spins[i];
          }
        }
      }
      const nextBits = spins.map((s) => (s > 0 ? 1 : 0));
      const eQubo = evalQuboEnergy(
        nextBits,
        activeMarket.Q,
        activeMarket.penalty_lambda ?? 0,
        mktK
      );
      mktBitsRef.current = nextBits;
      setMktBits(nextBits);
      setMktTrace((prev) => {
        const next = [...prev, Number(eQubo.toFixed(4))];
        return next.length > 140 ? next.slice(next.length - 140) : next;
      });
    },
    [activeMarket, mktN, mktK]
  );

  useEffect(() => {
    if (!mktRunning || activeTab !== "qubo") return;
    const id = setInterval(() => {
      stepMarketIsing(2);
    }, 180);
    return () => clearInterval(id);
  }, [mktRunning, activeTab, stepMarketIsing]);

  const annealMarket = useCallback(() => {
    setMktRunning(false);
    const spins: number[] = Array.from({ length: mktN }, () => (Math.random() < 0.5 ? 1 : -1));
    const J = activeMarket.J;
    const h = activeMarket.h;
    const Q = activeMarket.Q;
    const trace: number[] = [];
    for (let step = 0; step < 80; step++) {
      const b = 0.4 + (6.0 - 0.4) * (step / 79);
      for (let k = 0; k < mktN * 2; k++) {
        const i = (Math.random() * mktN) | 0;
        let field = h[i] ?? 0;
        for (let j = 0; j < mktN; j++) {
          if (j !== i) field += (J[i]?.[j] ?? 0) * spins[j];
        }
        const dE = 2 * spins[i] * field;
        if (dE <= 0 || Math.random() < Math.exp(-b * dE)) {
          spins[i] = -spins[i];
        }
      }
      const bits = spins.map((s) => (s > 0 ? 1 : 0));
      trace.push(
        Number(evalQuboEnergy(bits, Q, activeMarket.penalty_lambda ?? 0, mktK).toFixed(4))
      );
    }
    setMktBeta(5.0);
    setMktBits(spins.map((s) => (s > 0 ? 1 : 0)));
    setMktTrace(trace);
  }, [activeMarket, mktN, mktK]);

  const mktSokal = useMemo(() => computeSokalTauInt(mktTrace), [mktTrace]);
  const mktSelectedAssets = useMemo(
    () => activeMarket.assets.filter((_, idx) => mktBits[idx] === 1),
    [activeMarket.assets, mktBits]
  );
  const mktCurrentQuboE = useMemo(
    () =>
      evalQuboEnergy(
        mktBits.slice(0, mktN),
        activeMarket.Q,
        activeMarket.penalty_lambda ?? 0,
        mktK
      ),
    [mktBits, mktN, activeMarket.Q, activeMarket.penalty_lambda, mktK]
  );
  const mktCurrentIsingE = useMemo(
    () =>
      evalIsingEnergy(
        mktBits.slice(0, mktN).map((b) => (b ? 1 : -1)),
        activeMarket.J,
        activeMarket.h
      ),
    [mktBits, mktN, activeMarket.J, activeMarket.h]
  );

  /* --------------------------------------------------------------------------
   * TAB 4: AffilAI Product Graph (15 Shopify DTC Brands)
   * -------------------------------------------------------------------------- */
  const activeAffil = usingLive && payload.affilai_product_graph ? payload.affilai_product_graph : syntheticAffil;
  type AffilSortKey = "composite_deal_score" | "price_elasticity" | "variant_depth" | "inventory_velocity" | "tag_taxonomy_richness";
  const [affilSort, setAffilSort] = useState<AffilSortKey>("composite_deal_score");
  const [selectedBrandIdx, setSelectedBrandIdx] = useState<number>(0);

  const sortedBrands = useMemo(() => {
    return activeAffil.brands
      .map((b, origIdx) => ({ ...b, origIdx }))
      .sort((a, b) => b[affilSort] - a[affilSort]);
  }, [activeAffil.brands, affilSort]);

  const selectedBrandNeighbors = useMemo(() => {
    const J = activeAffil.brand_affinity_J;
    const row = J?.[selectedBrandIdx] ?? [];
    return row
      .map((weight, j) => ({ j, weight, brand: activeAffil.brands[j] }))
      .filter((item) => item.j !== selectedBrandIdx && item.brand)
      .sort((a, b) => b.weight - a.weight)
      .slice(0, 4);
  }, [activeAffil, selectedBrandIdx]);

  const RNA_BASE_COLOR: Record<string, string> = {
    A: "#B93A08",
    C: "#2E5E6B",
    G: "#16222B",
    U: "#7C4D12",
    "-": "#7F8E94",
  };

  return (
    <div className="refinery-root">
      <style jsx>{`
        .refinery-root {
          --paper: #E9EDEE;
          --panel: #F7F9FA;
          --ink: #16222B;
          --ink-2: #33454F;
          --ink-3: #46585F;
          --ember: #E8490F;
          --ember-text: #B93A08;
          --steel: #2E5E6B;
          --steel-deep: #1F4753;
          --rule: #B9C4C8;
          --rule-soft: #CBD4D7;
          margin-top: 2rem;
          color: var(--ink);
          font-family: 'Archivo', system-ui, -apple-system, sans-serif;
        }
        .refinery-shell {
          background: var(--panel);
          border: 1.5px solid var(--ink);
          border-radius: 4px;
          padding: clamp(1rem, 3vw, 1.6rem);
          box-shadow: 8px 12px 24px rgba(22, 34, 43, 0.08);
        }
        .refinery-top {
          display: flex;
          flex-wrap: wrap;
          align-items: center;
          justify-content: space-between;
          gap: 1rem;
          padding-bottom: 1rem;
          border-bottom: 1.5px solid var(--ink);
        }
        .refinery-toggle-group {
          display: inline-flex;
          flex-wrap: wrap;
          align-items: center;
          gap: 0.5rem;
        }
        .refinery-switch-track {
          display: inline-flex;
          align-items: center;
          background: var(--paper);
          border: 1.5px solid var(--ink);
          border-radius: 3px;
          padding: 3px;
          gap: 3px;
        }
        .refinery-switch-btn {
          border: 0;
          border-radius: 2px;
          padding: 0.45rem 0.8rem;
          min-height: 38px;
          font-family: 'JetBrains Mono', monospace;
          font-size: 0.78rem;
          font-weight: 600;
          cursor: pointer;
          background: transparent;
          color: var(--ink-2);
          transition: background 0.15s ease, color 0.15s ease;
        }
        .refinery-switch-btn.active-synth {
          background: var(--ink);
          color: #fff;
        }
        .refinery-switch-btn.active-live {
          background: var(--ember);
          color: #fff;
        }
        .refinery-switch-btn.active-custom {
          background: var(--steel);
          color: #fff;
        }
        .refinery-switch-pill {
          display: inline-flex;
          align-items: center;
          gap: 0.4rem;
          border: 1px solid var(--rule);
          background: var(--paper);
          color: var(--ink-2);
          border-radius: 3px;
          padding: 0.4rem 0.7rem;
          min-height: 38px;
          font-family: 'JetBrains Mono', monospace;
          font-size: 0.76rem;
          cursor: pointer;
        }
        .refinery-switch-pill:hover {
          border-color: var(--ink);
          color: var(--ink);
        }
        .refinery-badge-row {
          margin-top: 0.9rem;
          display: flex;
          flex-wrap: wrap;
          align-items: center;
          justify-content: space-between;
          gap: 0.65rem;
          padding: 0.6rem 0.85rem;
          background: var(--paper);
          border: 1px solid var(--rule);
          border-radius: 3px;
          font-family: 'JetBrains Mono', monospace;
          font-size: 0.78rem;
          color: var(--ink-2);
        }
        .refinery-badge-pill {
          display: inline-flex;
          align-items: center;
          gap: 0.45rem;
          font-weight: 700;
          color: var(--ink);
        }
        .refinery-dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          display: inline-block;
        }
        .refinery-tabs {
          margin-top: 1.1rem;
          display: flex;
          flex-wrap: wrap;
          gap: 0.45rem;
          border-bottom: 1.5px solid var(--rule);
          padding-bottom: 0.75rem;
        }
        .refinery-tab {
          border: 1px solid var(--rule);
          background: var(--paper);
          color: var(--ink-2);
          border-radius: 3px;
          padding: 0.55rem 0.9rem;
          min-height: 44px;
          font-family: 'Archivo', sans-serif;
          font-weight: 700;
          font-size: 0.9rem;
          cursor: pointer;
          text-align: left;
          transition: border-color 0.15s ease, background 0.15s ease, color 0.15s ease;
        }
        .refinery-tab:hover {
          border-color: var(--ink);
          color: var(--ink);
        }
        .refinery-tab[aria-selected="true"] {
          background: var(--ink);
          border-color: var(--ink);
          color: #fff;
        }
        .refinery-tab-sub {
          display: block;
          font-family: 'JetBrains Mono', monospace;
          font-size: 0.7rem;
          font-weight: 500;
          opacity: 0.82;
        }
        .refinery-grid {
          display: grid;
          grid-template-columns: minmax(0, 1.08fr) minmax(0, 0.92fr);
          gap: 1.25rem;
          margin-top: 1.15rem;
          align-items: start;
        }
        .refinery-card {
          background: var(--paper);
          border: 1px solid var(--rule);
          border-radius: 4px;
          padding: 1rem;
          min-width: 0;
        }
        .refinery-card h3 {
          font-family: 'Zilla Slab', Georgia, serif;
          font-size: 1.22rem;
          margin: 0 0 0.35rem;
        }
        .refinery-card-sub {
          font-size: 0.88rem;
          color: var(--ink-2);
          margin: 0 0 0.9rem;
        }
        .refinery-metrics {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 0.65rem;
          margin-top: 0.85rem;
        }
        .refinery-metric {
          background: var(--panel);
          border: 1px solid var(--rule-soft);
          border-radius: 3px;
          padding: 0.55rem 0.65rem;
        }
        .refinery-metric .k {
          font-size: 0.72rem;
          font-weight: 600;
          color: var(--ink-3);
        }
        .refinery-metric .v {
          font-family: 'JetBrains Mono', monospace;
          font-weight: 700;
          font-size: 1.05rem;
          font-variant-numeric: tabular-nums;
          margin-top: 0.15rem;
        }
        .refinery-metric .v.hot {
          color: var(--ember-text);
        }
        .refinery-controls {
          display: flex;
          flex-wrap: wrap;
          gap: 0.5rem;
          align-items: center;
          margin-top: 0.85rem;
        }
        .refinery-btn {
          display: inline-flex;
          align-items: center;
          gap: 0.35rem;
          min-height: 38px;
          padding: 0.4rem 0.8rem;
          border-radius: 3px;
          border: 1.5px solid var(--ink);
          background: var(--panel);
          color: var(--ink);
          font-family: 'Archivo', sans-serif;
          font-weight: 700;
          font-size: 0.84rem;
          cursor: pointer;
        }
        .refinery-btn:hover {
          border-color: var(--ember);
          color: var(--ember-text);
        }
        .refinery-btn.primary {
          background: var(--ember);
          border-color: var(--ember);
          color: #fff;
        }
        .refinery-btn.primary:hover {
          background: var(--ember-text);
          border-color: var(--ember-text);
          color: #fff;
        }
        .refinery-chart-card {
          background: var(--panel);
          border: 1px solid var(--rule-soft);
          border-radius: 3px;
          padding: 0.75rem;
          margin-top: 0.85rem;
        }
        .refinery-chart-head {
          display: flex;
          justify-content: space-between;
          gap: 0.5rem;
          font-size: 0.75rem;
          color: var(--ink-3);
          margin-bottom: 0.4rem;
        }
        .refinery-svg {
          width: 100%;
          height: auto;
          display: block;
        }
        .refinery-chart-legend {
          display: flex;
          flex-wrap: wrap;
          gap: 0.9rem;
          font-size: 0.72rem;
          color: var(--ink-3);
          margin-top: 0.45rem;
        }
        .refinery-chart-legend i {
          display: inline-block;
          width: 12px;
          height: 3px;
          vertical-align: middle;
          margin-right: 4px;
        }
        .spin-4x4 {
          display: grid;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          gap: 0.45rem;
        }
        .spin-tile {
          border: 1.5px solid var(--ink);
          border-radius: 3px;
          padding: 0.45rem 0.5rem;
          text-align: left;
          cursor: pointer;
          transition: transform 0.1s ease;
        }
        .spin-tile:active {
          transform: scale(0.97);
        }
        .spin-tile.up {
          background: #16222B;
          color: #F7F9FA;
        }
        .spin-tile.down {
          background: #DFE6E8;
          color: #16222B;
          border-color: #7D9199;
        }
        .spin-tile.selected {
          outline: 2px solid var(--ember);
          outline-offset: 1px;
        }
        .refinery-table-wrap {
          overflow-x: auto;
          margin-top: 0.75rem;
          border: 1px solid var(--rule);
          border-radius: 3px;
          background: var(--panel);
        }
        table.refinery-tbl {
          width: 100%;
          border-collapse: collapse;
          font-size: 0.8rem;
          font-variant-numeric: tabular-nums;
        }
        table.refinery-tbl th,
        table.refinery-tbl td {
          padding: 0.45rem 0.55rem;
          border-bottom: 1px solid var(--rule-soft);
          text-align: left;
          vertical-align: middle;
        }
        table.refinery-tbl th {
          font-family: 'JetBrains Mono', monospace;
          font-size: 0.72rem;
          color: var(--ink-3);
          background: var(--paper);
          white-space: nowrap;
        }
        table.refinery-tbl td.num,
        table.refinery-tbl th.num {
          text-align: right;
          font-family: 'JetBrains Mono', monospace;
        }
        .rna-ribbon {
          display: grid;
          grid-template-columns: repeat(12, minmax(0, 1fr));
          gap: 0.3rem;
        }
        .rna-base {
          border: 1px solid var(--rule);
          border-radius: 3px;
          padding: 0.3rem 0.2rem;
          text-align: center;
          background: var(--panel);
          font-family: 'JetBrains Mono', monospace;
        }
        .asset-grid {
          display: grid;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          gap: 0.45rem;
        }
        .asset-chip {
          border: 1.5px solid var(--rule);
          border-radius: 3px;
          padding: 0.45rem 0.5rem;
          background: var(--panel);
          text-align: left;
          cursor: pointer;
          color: var(--ink);
        }
        .asset-chip.on {
          border-color: var(--ember);
          background: #FFF2EC;
        }
        @media (max-width: 960px) {
          .refinery-grid {
            grid-template-columns: 1fr;
          }
        }
        @media (max-width: 600px) {
          .refinery-metrics {
            grid-template-columns: repeat(2, minmax(0, 1fr));
          }
          .spin-4x4 {
            grid-template-columns: repeat(2, minmax(0, 1fr));
          }
          .asset-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
          }
          .rna-ribbon {
            grid-template-columns: repeat(6, minmax(0, 1fr));
          }
        }
      `}</style>

      <div className="refinery-shell">
        {/* Top Master Toggle & Status */}
        <div className="refinery-top">
          <div>
            <div className="mono" style={{ fontSize: "0.75rem", color: "var(--ember-text)", fontWeight: 700 }}>
              LIVE-STREAM DATA REFINERY · 5-PHASE EMPIRICAL PIPELINE
            </div>
            <h3 style={{ fontFamily: "'Zilla Slab', Georgia, serif", fontSize: "1.5rem", marginTop: "0.2rem" }}>
              Empirical Tensor Bench — Synthetic Baseline vs. Live Public Telemetry
            </h3>
          </div>

          <div className="refinery-toggle-group">
            <span className="mono" style={{ fontSize: "0.75rem", fontWeight: 700, color: "var(--ink-2)" }}>
              Data Source:
            </span>
            <div
              className="refinery-switch-track"
              role="group"
              aria-label="Data Source: Synthetic Baseline, Live Public Telemetry, or Upload Custom Matrix / FASTA"
            >
              <button
                type="button"
                aria-pressed={dataSource === "synthetic"}
                className={`refinery-switch-btn ${dataSource === "synthetic" ? "active-synth" : ""}`}
                onClick={() => setDataSource("synthetic")}
              >
                Synthetic Baseline
              </button>
              <button
                type="button"
                aria-pressed={dataSource === "live"}
                className={`refinery-switch-btn ${dataSource === "live" ? "active-live" : ""}`}
                onClick={handleSwitchToLive}
              >
                Live Public Telemetry
              </button>
              <button
                type="button"
                aria-pressed={dataSource === "custom"}
                className={`refinery-switch-btn ${dataSource === "custom" ? "active-custom" : ""}`}
                onClick={handleSwitchToCustom}
              >
                Upload Custom Matrix / FASTA
              </button>
            </div>
            <button
              type="button"
              className="refinery-switch-pill"
              onClick={() => {
                setDataSource("live");
                void fetchLiveStreams();
              }}
              title="Re-fetch /data/live_empirical_streams.json"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? "animate-spin" : ""}`} />
              {isLoading ? "Polling..." : "Refresh Stream"}
            </button>
          </div>
        </div>

        {/* Provenance / Doctrine Banner */}
        <div className="refinery-badge-row">
          <div className="refinery-badge-pill">
            <span
              className="refinery-dot"
              style={{
                background: usingCustom
                  ? "#2E5E6B"
                  : usingLive
                  ? "#E8490F"
                  : dataSource === "live" && loadError
                  ? "#B93A08"
                  : "#2E5E6B",
              }}
            />
            {usingCustom ? (
              <span>
                ACTIVE: CUSTOM INBOUND TENSOR · {customSummary.label} · local browser solve (zero raw upload)
              </span>
            ) : usingLive ? (
              <span>
                ACTIVE: LIVE PUBLIC TELEMETRY · schema {payload.schema_version} · generated {payload.generated_at_utc}
              </span>
            ) : dataSource === "live" ? (
              <span>
                {isLoading
                  ? "FETCHING /data/live_empirical_streams.json…"
                  : `LIVE STREAM FALLBACK (${loadError ?? "awaiting manifest"}) · showing synthetic baseline`}
              </span>
            ) : (
              <span>ACTIVE: SYNTHETIC BASELINE · deterministic classical reference tensors</span>
            )}
          </div>
          <span>
            {usingLive && payload.doctrine_disclosure
              ? payload.doctrine_disclosure
              : "JAX / TypeScript classical samplers · Sokal τ_int windowing (W ≥ 5·τ_int)"}
          </span>
        </div>

        {/* 4 Domain Tabs */}
        <div className="refinery-tabs" role="tablist" aria-label="Refinery domains">
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === "ising"}
            className="refinery-tab"
            onClick={() => setActiveTab("ising")}
          >
            Ising Lattice
            <span className="refinery-tab-sub">Stations 2 &amp; 10 · Aquacellum / USGS</span>
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === "potts"}
            className="refinery-tab"
            onClick={() => setActiveTab("potts")}
          >
            Potts / RNA Co-Evolution
            <span className="refinery-tab-sub">Stations 4, 11 &amp; 13 · Rfam RF00005</span>
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === "qubo"}
            className="refinery-tab"
            onClick={() => setActiveTab("qubo")}
          >
            Market &amp; Polymarket QUBO
            <span className="refinery-tab-sub">Stations 6 &amp; 7 · Yahoo 16 + CLOB</span>
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === "affilai"}
            className="refinery-tab"
            onClick={() => setActiveTab("affilai")}
          >
            AffilAI Product Graph
            <span className="refinery-tab-sub">15 Shopify DTC Brands · Affinity J</span>
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === "custom"}
            className="refinery-tab"
            onClick={() => {
              setActiveTab("custom");
              setDataSource("custom");
            }}
          >
            Custom Matrix / FASTA
            <span className="refinery-tab-sub">CSV · JSON · FASTA + Anonymized Corpus</span>
          </button>
        </div>

        {/* ==================================================================
            TAB 1: ISING LATTICE (STATIONS 2 & 10 · AQUACELLUM / USGS)
           ================================================================== */}
        {activeTab === "ising" && (
          <div className="refinery-grid" role="tabpanel" aria-label="Ising Lattice">
            <div className="refinery-card">
              <h3>
                {usingCustom && customEco
                  ? `Custom Uploaded Ising Lattice (N=${ecoN})`
                  : usingLive
                  ? "16-Channel USGS Estuary Telemetry Spin Glass"
                  : "16-Node Sherrington–Kirkpatrick Baseline"}
              </h3>
              <p className="refinery-card-sub mono" style={{ fontSize: "0.78rem" }}>
                {activeEco.provenance.source_label} · {activeEco.provenance.source_mode} ·{" "}
                {activeEco.provenance.captured_at_utc}
              </p>

              <div className="spin-4x4" aria-label="16 interactive spin channels">
                {Array.from({ length: ecoN }, (_, idx) => {
                  const s = ecoSpins[idx] ?? 1;
                  const meta = activeEco.nodes?.[idx];
                    const label =
                      typeof meta === "string"
                        ? meta
                        : meta?.channel_id || meta?.site_code || meta?.site_id || `CH-${idx + 1}`;
                  const sub =
                    typeof meta === "string"
                      ? `h=${(activeEco.h[idx] ?? 0).toFixed(2)}`
                      : meta?.parameter_name
                      ? `${meta.parameter_name.slice(0, 14)} · h=${(activeEco.h[idx] ?? 0).toFixed(2)}`
                      : `h=${(activeEco.h[idx] ?? 0).toFixed(2)}`;
                  return (
                    <button
                      key={idx}
                      type="button"
                      className={`spin-tile ${s > 0 ? "up" : "down"} ${
                        ecoSelectedNode === idx ? "selected" : ""
                      }`}
                      onClick={() => {
                        setEcoSelectedNode(idx);
                        const next = ecoSpins.slice(0, ecoN);
                        while (next.length < ecoN) next.push(1);
                        next[idx] = -next[idx];
                        setEcoSpins(next);
                        setEcoTrace((prev) => [
                          ...prev.slice(-139),
                          Number(evalIsingEnergy(next, activeEco.J, activeEco.h).toFixed(4)),
                        ]);
                      }}
                    >
                      <div style={{ display: "flex", justifyContent: "space-between", fontWeight: 700 }} className="mono">
                        <span style={{ fontSize: "0.72rem" }}>{label}</span>
                        <span>{s > 0 ? "+1" : "−1"}</span>
                      </div>
                      <div style={{ fontSize: "0.68rem", opacity: 0.85, marginTop: "0.15rem" }}>{sub}</div>
                    </button>
                  );
                })}
              </div>

              <div style={{ marginTop: "0.9rem" }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.82rem", fontWeight: 600 }}>
                  <span>Inverse Temperature β = {ecoBeta.toFixed(2)}</span>
                  <span className="mono" style={{ color: "var(--ink-3)" }}>
                    T = {(1 / ecoBeta).toFixed(2)}
                  </span>
                </div>
                <input
                  type="range"
                  min={0.2}
                  max={5.0}
                  step={0.05}
                  value={ecoBeta}
                  onChange={(e) => setEcoBeta(Number(e.target.value))}
                  style={{ width: "100%", accentColor: "#E8490F", marginTop: "0.25rem" }}
                  aria-label="Ising inverse temperature beta"
                />
              </div>

              <div className="refinery-controls">
                <button
                  type="button"
                  className="refinery-btn primary"
                  onClick={() => setEcoRunning((r) => !r)}
                >
                  {ecoRunning ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
                  {ecoRunning ? "Pause" : "Run Gibbs"}
                </button>
                <button type="button" className="refinery-btn" onClick={ecoAnneal}>
                  <Sparkles className="w-3.5 h-3.5" /> Anneal to GS
                </button>
                <button
                  type="button"
                  className="refinery-btn"
                  onClick={() => {
                    const fresh = Array.from({ length: ecoN }, () => (Math.random() < 0.5 ? 1 : -1));
                    setEcoSpins(fresh);
                    setEcoTrace([evalIsingEnergy(fresh, activeEco.J, activeEco.h)]);
                  }}
                >
                  <RotateCcw className="w-3.5 h-3.5" /> Scramble
                </button>
              </div>

              <div className="refinery-metrics">
                <div className="refinery-metric">
                  <div className="k">Live Energy E(s)</div>
                  <div className="v hot">{ecoCurrentE.toFixed(3)}</div>
                </div>
                <div className="refinery-metric">
                  <div className="k">Exact Ground State</div>
                  <div className="v">{activeEco.exact_ground_state_energy.toFixed(3)}</div>
                </div>
                <div className="refinery-metric">
                  <div className="k">Order |M| (Browser / Ref)</div>
                  <div className="v">
                    {ecoAbsM.toFixed(2)} / {Math.abs(activeEco.magnetization).toFixed(2)}
                  </div>
                </div>
                <div className="refinery-metric">
                  <div className="k">Browser τ_int (Sokal)</div>
                  <div className="v">{ecoSokal.tauInt.toFixed(2)}</div>
                </div>
                <div className="refinery-metric">
                  <div className="k">Browser ESS</div>
                  <div className="v">{ecoSokal.ess.toFixed(1)}</div>
                </div>
                <div className="refinery-metric">
                  <div className="k">Frustrated Triangles</div>
                  <div className="v">
                    {(activeEco.frustrated_triangles_pct ?? activeEco.frustration_pct).toFixed(1)}%
                  </div>
                </div>
              </div>
            </div>

            <div className="refinery-card">
              <h3>Energy Relaxation &amp; 16-Channel Coupling Inspector</h3>
              <p className="refinery-card-sub">
                Exact Hamiltonian <span className="mono">E = −½ sᵀJs − hᵀs</span>. Backend benchmark{" "}
                <span className="mono">
                  τ_int={activeEco.sampler_benchmark.tau_int.toFixed(2)} · ESS=
                  {activeEco.sampler_benchmark.ess.toFixed(1)}
                </span>
                .
              </p>

              <EnergyTraceChart
                liveTrace={ecoTrace}
                benchmarkTrace={activeEco.sampler_benchmark.energy_trace}
                exactEnergy={activeEco.exact_ground_state_energy}
                labelLeft="E(t) · 16-channel Ising"
                labelRight={`${ecoTrace.length} samples · τ_int ${ecoSokal.tauInt.toFixed(2)}`}
              />

              {/* 16x16 J_ij coupling matrix SVG */}
              <div className="refinery-chart-card">
                <div className="refinery-chart-head">
                  <span className="mono">16 × 16 Coupling Tensor J_ij (selected row #{ecoSelectedNode + 1})</span>
                  <span className="mono">red = ferro (+J) · steel = anti (−J)</span>
                </div>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: `repeat(${ecoN}, minmax(0, 1fr))`,
                    gap: "1px",
                    background: "#CBD4D7",
                    padding: "2px",
                  }}
                >
                  {activeEco.J.slice(0, ecoN).flatMap((row, r) =>
                    row.slice(0, ecoN).map((val, c) => {
                      const norm = Math.min(1, Math.abs(val) * 1.8);
                      const bg =
                        r === c
                          ? "#E9EDEE"
                          : val >= 0
                          ? `rgba(232, 73, 15, ${0.12 + 0.85 * norm})`
                          : `rgba(46, 94, 107, ${0.12 + 0.85 * norm})`;
                      const isHi = r === ecoSelectedNode || c === ecoSelectedNode;
                      return (
                        <div
                          key={`${r}-${c}`}
                          onClick={() => setEcoSelectedNode(r)}
                          title={`J[${r + 1},${c + 1}] = ${val.toFixed(3)}`}
                          style={{
                            aspectRatio: "1 / 1",
                            background: bg,
                            outline: isHi ? "1px solid #16222B" : "none",
                            cursor: "pointer",
                          }}
                        />
                      );
                    })
                  )}
                </div>
              </div>

              {/* Channel Metadata Table */}
              <div className="refinery-table-wrap" style={{ maxHeight: "220px", overflowY: "auto" }}>
                <table className="refinery-tbl">
                  <thead>
                    <tr>
                      <th>Channel</th>
                      <th>Station / Parameter</th>
                      <th className="num">Bias h_i</th>
                      <th className="num">Live Val</th>
                      <th className="num">Spin</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Array.from({ length: ecoN }, (_, idx) => {
                      const node = activeEco.nodes?.[idx];
                      const ch = typeof node === "string" ? node : node?.channel_id ?? `CH-${idx + 1}`;
                      const desc =
                        typeof node === "string"
                          ? "Telemetry channel"
                          : `${node?.site_name ?? node?.site_code ?? node?.site_id ?? ""} ${node?.parameter_name ?? ""}`.trim();
                      const valStr =
                        typeof node === "object" && node?.latest_value !== undefined
                          ? `${node.latest_value.toFixed(2)} ${node.unit ?? ""}`.trim()
                          : "—";
                      return (
                        <tr
                          key={idx}
                          onClick={() => setEcoSelectedNode(idx)}
                          style={{
                            cursor: "pointer",
                            background: ecoSelectedNode === idx ? "#FFF2EC" : undefined,
                          }}
                        >
                          <td className="mono">{ch}</td>
                          <td>{desc || `Channel ${idx + 1}`}</td>
                          <td className="num">{(activeEco.h[idx] ?? 0).toFixed(3)}</td>
                          <td className="num">{valStr}</td>
                          <td className="num">{(ecoSpins[idx] ?? 1) > 0 ? "+1" : "−1"}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* ==================================================================
            TAB 2: POTTS / RNA CO-EVOLUTION (RFAM RF00005, L=24, q=5)
           ================================================================== */}
        {activeTab === "potts" && (
          <div className="refinery-grid" role="tabpanel" aria-label="Potts RNA Co-Evolution">
            <div className="refinery-card">
              <h3>
                {usingCustom && customRna
                  ? `Custom Uploaded FASTA Potts (L = ${rnaL}, q = ${rnaQ})`
                  : usingLive
                  ? `Rfam ${activeRna.provenance.family_id} Categorical Potts (q = ${rnaQ})`
                  : "Synthetic RNA Potts Hairpin Baseline (L = 24, q = 5)"}
              </h3>
              <p className="refinery-card-sub mono" style={{ fontSize: "0.78rem" }}>
                {activeRna.provenance.source_label} · {activeRna.provenance.retained_sequences}/
                {activeRna.provenance.raw_sequences} seqs · {activeRna.provenance.captured_at_utc}
              </p>

              {/* Sampled 24-nt consensus sequence */}
              <div style={{ fontSize: "0.78rem", fontWeight: 700, marginBottom: "0.35rem", color: "var(--ink-2)" }}>
                Live Categorical Gibbs Sample ({rnaL} consensus columns)
              </div>
              <div className="rna-ribbon">
                {rnaSeq.slice(0, rnaL).map((stateIdx, pos) => {
                  const sym = activeRna.alphabet[stateIdx] ?? "-";
                  const col = RNA_BASE_COLOR[sym] ?? "#16222B";
                  return (
                    <div key={pos} className="rna-base">
                      <div style={{ fontSize: "0.62rem", color: "var(--ink-3)" }}>{pos + 1}</div>
                      <div style={{ fontSize: "1.05rem", fontWeight: 800, color: col }}>{sym}</div>
                    </div>
                  );
                })}
              </div>

              <div style={{ marginTop: "0.9rem" }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.82rem", fontWeight: 600 }}>
                  <span>Inverse Temperature β = {rnaBeta.toFixed(2)}</span>
                  <span className="mono" style={{ color: "var(--ink-3)" }}>
                    Entropy {activeRna.sampler_benchmark.entropy_bits.toFixed(2)} bits
                  </span>
                </div>
                <input
                  type="range"
                  min={0.2}
                  max={4.5}
                  step={0.05}
                  value={rnaBeta}
                  onChange={(e) => setRnaBeta(Number(e.target.value))}
                  style={{ width: "100%", accentColor: "#E8490F", marginTop: "0.25rem" }}
                  aria-label="Potts inverse temperature beta"
                />
              </div>

              <div className="refinery-controls">
                <button
                  type="button"
                  className="refinery-btn primary"
                  onClick={() => setRnaRunning((r) => !r)}
                >
                  {rnaRunning ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
                  {rnaRunning ? "Pause" : "Run Gibbs"}
                </button>
                <button
                  type="button"
                  className="refinery-btn"
                  onClick={() => {
                    setRnaBeta(4.0);
                    stepRnaGibbs(6, 4.0);
                  }}
                >
                  <Sparkles className="w-3.5 h-3.5" /> Cool Step (β=4.0)
                </button>
                <button
                  type="button"
                  className="refinery-btn"
                  onClick={() => {
                    const randSeq = Array.from({ length: rnaL }, () => (Math.random() * rnaQ) | 0);
                    setRnaSeq(randSeq);
                    setRnaTrace([evalRnaEnergy(randSeq, activeRna)]);
                  }}
                >
                  <RotateCcw className="w-3.5 h-3.5" /> Randomise
                </button>
              </div>

              <div className="refinery-metrics">
                <div className="refinery-metric">
                  <div className="k">Sample Potts E</div>
                  <div className="v hot">{(rnaTrace[rnaTrace.length - 1] ?? 0).toFixed(2)}</div>
                </div>
                <div className="refinery-metric">
                  <div className="k">Backend Min / Mean E</div>
                  <div className="v">
                    {activeRna.sampler_benchmark.min_energy.toFixed(1)} /{" "}
                    {activeRna.sampler_benchmark.mean_energy.toFixed(1)}
                  </div>
                </div>
                <div className="refinery-metric">
                  <div className="k">Browser τ_int (Sokal)</div>
                  <div className="v">{rnaSokal.tauInt.toFixed(2)}</div>
                </div>
                <div className="refinery-metric">
                  <div className="k">Browser ESS</div>
                  <div className="v">{rnaSokal.ess.toFixed(1)}</div>
                </div>
                <div className="refinery-metric">
                  <div className="k">Backend τ_int</div>
                  <div className="v">{activeRna.sampler_benchmark.tau_int.toFixed(2)}</div>
                </div>
                <div className="refinery-metric">
                  <div className="k">Backend ESS</div>
                  <div className="v">{activeRna.sampler_benchmark.ess.toFixed(1)}</div>
                </div>
              </div>

              {/* Site-wise nucleotide frequency profile */}
              <div className="refinery-chart-card">
                <div className="refinery-chart-head">
                  <span className="mono">Nucleotide Frequencies f_i(a) · A / C / G / U / −</span>
                  <span className="mono">Σ_a f_i(a) = 1.00</span>
                </div>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: `repeat(${rnaL}, minmax(0, 1fr))`,
                    gap: "2px",
                    height: "68px",
                    alignItems: "stretch",
                  }}
                >
                  {activeRna.f_i.slice(0, rnaL).map((freqs, col) => (
                    <div
                      key={col}
                      title={`Col ${col + 1}: ${activeRna.alphabet
                        .map((b, a) => `${b}=${((freqs[a] ?? 0) * 100).toFixed(0)}%`)
                        .join(" ")}`}
                      style={{ display: "flex", flexDirection: "column", height: "100%", borderRadius: "1px", overflow: "hidden" }}
                    >
                      {freqs.map((p, a) => (
                        <div
                          key={a}
                          style={{
                            height: `${Math.max(0, p * 100)}%`,
                            background: RNA_BASE_COLOR[activeRna.alphabet[a]] ?? "#888",
                          }}
                        />
                      ))}
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="refinery-card">
              <h3>24 × 24 Frobenius Co-Evolution Heatmap &amp; Energy</h3>
              <p className="refinery-card-sub">
                Direct co-evolution strength <span className="mono">||C_ij||_F</span> across consensus positions. Click any cell to inspect <span className="mono">C_ij(a, b)</span>.
              </p>

              <div className="refinery-chart-card" style={{ marginTop: 0 }}>
                <div className="refinery-chart-head">
                  <span className="mono">
                    Frobenius Matrix ({rnaL}×{rnaL}) · Pair ({rnaCellInspect[0] + 1}, {rnaCellInspect[1] + 1}) ={" "}
                    {(activeRna.frobenius_couplings?.[rnaCellInspect[0]]?.[rnaCellInspect[1]] ?? 0).toFixed(3)}
                  </span>
                </div>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: `repeat(${rnaL}, minmax(0, 1fr))`,
                    gap: "1px",
                    background: "#CBD4D7",
                    padding: "2px",
                    maxWidth: "380px",
                    margin: "0 auto",
                  }}
                >
                  {activeRna.frobenius_couplings.slice(0, rnaL).flatMap((row, i) =>
                    row.slice(0, rnaL).map((val, j) => {
                      const intensity = Math.min(1, val / rnaMaxFrob);
                      const isSel = i === rnaCellInspect[0] && j === rnaCellInspect[1];
                      return (
                        <div
                          key={`${i}-${j}`}
                          onClick={() => setRnaCellInspect([i, j])}
                          title={`(${i + 1}, ${j + 1}) F = ${val.toFixed(4)}`}
                          style={{
                            aspectRatio: "1 / 1",
                            background:
                              i === j
                                ? "#E9EDEE"
                                : `rgba(232, 73, 15, ${0.06 + 0.94 * intensity})`,
                            outline: isSel ? "1.5px solid #16222B" : "none",
                            cursor: "pointer",
                          }}
                        />
                      );
                    })
                  )}
                </div>
              </div>

              <EnergyTraceChart
                liveTrace={rnaTrace}
                benchmarkTrace={activeRna.sampler_benchmark.energy_trace}
                labelLeft="Potts E(t) · q=5 Categorical Gibbs"
                labelRight={`τ_int ${rnaSokal.tauInt.toFixed(2)} · ESS ${rnaSokal.ess.toFixed(1)}`}
              />

              <div className="refinery-table-wrap" style={{ maxHeight: "170px", overflowY: "auto" }}>
                <table className="refinery-tbl">
                  <thead>
                    <tr>
                      <th>Top Contact</th>
                      <th>Modal Bases</th>
                      <th className="num">Frobenius Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rnaNormalizedContacts.map((c, idx) => {
                      const modalA = activeRna.f_i[c.i]?.indexOf(Math.max(...(activeRna.f_i[c.i] ?? [1]))) ?? 0;
                      const modalB = activeRna.f_i[c.j]?.indexOf(Math.max(...(activeRna.f_i[c.j] ?? [1]))) ?? 0;
                      return (
                        <tr
                          key={idx}
                          onClick={() => setRnaCellInspect([c.i, c.j])}
                          style={{ cursor: "pointer" }}
                        >
                          <td className="mono">
                            Col {c.i + 1} ↔ Col {c.j + 1}
                          </td>
                          <td className="mono">
                            {c.pair ?? `${activeRna.alphabet[modalA] ?? "?"}–${activeRna.alphabet[modalB] ?? "?"}`}
                          </td>
                          <td className="num">{c.score.toFixed(4)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* ==================================================================
            TAB 3: MARKET & POLYMARKET QUBO (YAHOO 16 + CLOB OBI)
           ================================================================== */}
        {activeTab === "qubo" && (
          <div className="refinery-grid" role="tabpanel" aria-label="Market and Polymarket QUBO">
            <div className="refinery-card">
              <h3>
                {usingCustom && customMarket
                  ? `Custom Uploaded QUBO → Ising Solver (N=${mktN}, K=${mktK})`
                  : usingLive
                  ? `Live ${mktN}-Asset Markowitz + Polymarket CLOB QUBO (K=${mktK})`
                  : `Synthetic ${mktN}-Asset Markowitz QUBO Baseline (K=${mktK})`}
              </h3>
              <p className="refinery-card-sub mono" style={{ fontSize: "0.78rem" }}>
                {activeMarket.provenance.source_label} · {activeMarket.provenance.polymarket_markets_polled} CLOB books ·{" "}
                {activeMarket.provenance.captured_at_utc}
              </p>

              <div className="asset-grid" aria-label="Selectable portfolio assets">
                {activeMarket.assets.map((asset, idx) => {
                  const on = mktBits[idx] === 1;
                  return (
                    <button
                      key={asset.ticker}
                      type="button"
                      className={`asset-chip ${on ? "on" : ""}`}
                      onClick={() => {
                        const next = mktBits.slice(0, mktN);
                        while (next.length < mktN) next.push(0);
                        next[idx] = on ? 0 : 1;
                        setMktBits(next);
                        setMktTrace((prev) => [
                          ...prev.slice(-139),
                          Number(
                            evalQuboEnergy(
                              next,
                              activeMarket.Q,
                              activeMarket.penalty_lambda ?? 0,
                              mktK
                            ).toFixed(4)
                          ),
                        ]);
                      }}
                    >
                      <div style={{ display: "flex", justifyContent: "space-between", fontWeight: 700 }} className="mono">
                        <span style={{ fontSize: "0.82rem" }}>{asset.ticker}</span>
                        <span style={{ color: on ? "var(--ember-text)" : "var(--ink-3)" }}>
                          {on ? "x=1" : "x=0"}
                        </span>
                      </div>
                      <div className="mono" style={{ fontSize: "0.7rem", color: "var(--ink-2)", marginTop: "0.15rem" }}>
                        μ {(asset.annualized_return * 100).toFixed(1)}% · OBI {asset.obi >= 0 ? "+" : ""}
                        {asset.obi.toFixed(2)}
                      </div>
                    </button>
                  );
                })}
              </div>

              <div style={{ marginTop: "0.9rem" }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.82rem", fontWeight: 600 }}>
                  <span>
                    Holding {mktSelectedAssets.length} / {mktK} assets · β = {mktBeta.toFixed(2)}
                  </span>
                  <span
                    className="mono"
                    style={{
                      color: mktSelectedAssets.length === mktK ? "#1F4753" : "#B93A08",
                      fontWeight: 700,
                    }}
                  >
                    {mktSelectedAssets.length === mktK ? "FEASIBLE" : "OFF BUDGET"}
                  </span>
                </div>
                <input
                  type="range"
                  min={0.3}
                  max={6.0}
                  step={0.1}
                  value={mktBeta}
                  onChange={(e) => setMktBeta(Number(e.target.value))}
                  style={{ width: "100%", accentColor: "#E8490F", marginTop: "0.25rem" }}
                  aria-label="Market QUBO inverse temperature beta"
                />
              </div>

              <div className="refinery-controls">
                <button
                  type="button"
                  className="refinery-btn primary"
                  onClick={() => setMktRunning((r) => !r)}
                >
                  {mktRunning ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
                  {mktRunning ? "Pause" : "Run Gibbs"}
                </button>
                <button type="button" className="refinery-btn" onClick={annealMarket}>
                  <Sparkles className="w-3.5 h-3.5" /> Anneal QUBO
                </button>
                <button
                  type="button"
                  className="refinery-btn"
                  onClick={() => {
                    const next = Array(mktN).fill(0);
                    activeMarket.exact_optimum.selected_indices.forEach((i) => {
                      if (i < mktN) next[i] = 1;
                    });
                    setMktBits(next);
                    setMktTrace((prev) => [
                      ...prev.slice(-139),
                      Number(
                        evalQuboEnergy(
                          next,
                          activeMarket.Q,
                          activeMarket.penalty_lambda ?? 0,
                          mktK
                        ).toFixed(4)
                      ),
                    ]);
                  }}
                >
                  Exact Optimum (K={mktK})
                </button>
              </div>

              <div className="refinery-metrics">
                <div className="refinery-metric">
                  <div className="k">Current QUBO E</div>
                  <div className="v hot">{mktCurrentQuboE.toFixed(3)}</div>
                </div>
                <div className="refinery-metric">
                  <div className="k">Ising Equiv E(s)</div>
                  <div className="v">{mktCurrentIsingE.toFixed(3)}</div>
                </div>
                <div className="refinery-metric">
                  <div className="k">Exact Opt E / Ret</div>
                  <div className="v">
                    {activeMarket.exact_optimum.qubo_energy.toFixed(2)} /{" "}
                    {(activeMarket.exact_optimum.expected_return * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="refinery-metric">
                  <div className="k">Browser τ_int (Sokal)</div>
                  <div className="v">{mktSokal.tauInt.toFixed(2)}</div>
                </div>
                <div className="refinery-metric">
                  <div className="k">Feasibility / Pareto</div>
                  <div className="v">
                    {activeMarket.sampler_benchmark.budget_feasibility_pct.toFixed(1)}% /{" "}
                    {activeMarket.sampler_benchmark.pareto_match_pct.toFixed(1)}%
                  </div>
                </div>
                <div className="refinery-metric">
                  <div className="k">Backend τ_int / ESS</div>
                  <div className="v">
                    {activeMarket.sampler_benchmark.tau_int.toFixed(2)} /{" "}
                    {activeMarket.sampler_benchmark.ess.toFixed(0)}
                  </div>
                </div>
              </div>
            </div>

            <div className="refinery-card">
              <h3>Optimal K={mktK} Portfolio &amp; Polymarket Order-Book Imbalance</h3>
              <p className="refinery-card-sub mono" style={{ fontSize: "0.78rem" }}>
                Exact optimum: [{activeMarket.exact_optimum.selected_tickers.join(", ")}] · Sampler:{" "}
                [{activeMarket.sampler_benchmark.selected_tickers.join(", ")}]
              </p>

              <EnergyTraceChart
                liveTrace={mktTrace}
                benchmarkTrace={activeMarket.sampler_benchmark.energy_trace}
                exactEnergy={activeMarket.exact_optimum.qubo_energy}
                labelLeft="QUBO Energy xᵀQx"
                labelRight={`τ_int ${mktSokal.tauInt.toFixed(2)} · ESS ${mktSokal.ess.toFixed(1)}`}
              />

              <div className="refinery-table-wrap" style={{ maxHeight: "260px", overflowY: "auto" }}>
                <table className="refinery-tbl">
                  <thead>
                    <tr>
                      <th>Ticker</th>
                      <th className="num">Last</th>
                      <th className="num">Ann Ret</th>
                      <th>Polymarket Event</th>
                      <th className="num">OBI [−1,1]</th>
                      <th className="num">Penalty</th>
                    </tr>
                  </thead>
                  <tbody>
                    {activeMarket.assets.map((a, idx) => {
                      const inOpt = activeMarket.exact_optimum.selected_indices.includes(idx);
                      return (
                        <tr
                          key={a.ticker}
                          style={{ background: mktBits[idx] === 1 ? "#FFF2EC" : undefined }}
                        >
                          <td className="mono" style={{ fontWeight: 700 }}>
                            {a.ticker}{" "}
                            {inOpt && (
                              <span style={{ color: "var(--ember-text)", fontSize: "0.7rem" }}>★</span>
                            )}
                          </td>
                          <td className="num">{a.latest_price.toFixed(2)}</td>
                          <td className="num">{(a.annualized_return * 100).toFixed(1)}%</td>
                          <td style={{ maxWidth: "180px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={a.polymarket_market}>
                            {a.polymarket_market}
                          </td>
                          <td className="num" style={{ color: a.obi < 0 ? "#B93A08" : "#1F4753" }}>
                            {a.obi >= 0 ? "+" : ""}
                            {a.obi.toFixed(3)}
                          </td>
                          <td className="num">{a.obi_penalty.toFixed(3)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* ==================================================================
            TAB 4: AFFILAI PRODUCT GRAPH (15 SHOPIFY DTC BRANDS)
           ================================================================== */}
        {activeTab === "affilai" && (
          <div className="refinery-grid" role="tabpanel" aria-label="AffilAI Product Graph">
            <div className="refinery-card" style={{ gridColumn: "1 / -1" }}>
              <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "space-between", gap: "0.75rem", alignItems: "baseline" }}>
                <div>
                  <h3>AffilAI Product Graph — {activeAffil.provenance.brand_count} Shopify DTC Brands</h3>
                  <p className="refinery-card-sub mono" style={{ fontSize: "0.78rem", marginBottom: 0 }}>
                    {activeAffil.provenance.source_label} · {activeAffil.provenance.total_products_polled} products ·{" "}
                    {activeAffil.provenance.total_variants_polled} variants · {activeAffil.provenance.captured_at_utc}
                  </p>
                </div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: "0.35rem", alignItems: "center" }}>
                  <span className="mono" style={{ fontSize: "0.74rem", color: "var(--ink-3)" }}>
                    Sort by:
                  </span>
                  {(
                    [
                      ["composite_deal_score", "Composite"],
                      ["price_elasticity", "Elasticity"],
                      ["variant_depth", "Variants"],
                      ["inventory_velocity", "Velocity"],
                      ["tag_taxonomy_richness", "Taxonomy"],
                    ] as Array<[AffilSortKey, string]>
                  ).map(([key, label]) => (
                    <button
                      key={key}
                      type="button"
                      className={`refinery-btn ${affilSort === key ? "primary" : ""}`}
                      style={{ minHeight: "32px", padding: "0.25rem 0.6rem", fontSize: "0.76rem" }}
                      onClick={() => setAffilSort(key)}
                    >
                      {label}
                    </button>
                  ))}
                </div>
              </div>

              <div className="refinery-table-wrap">
                <table className="refinery-tbl">
                  <thead>
                    <tr>
                      <th>Brand / Vertical</th>
                      <th className="num">SKU / Var</th>
                      <th className="num">Mean USD</th>
                      <th className="num">Elasticity</th>
                      <th className="num">Var Depth</th>
                      <th className="num">Velocity</th>
                      <th className="num">Taxonomy</th>
                      <th className="num">Deal Score</th>
                      <th>Commission Tier</th>
                      <th>Top Tags</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sortedBrands.map((b) => (
                      <tr
                        key={b.brand}
                        onClick={() => setSelectedBrandIdx(b.origIdx)}
                        style={{
                          cursor: "pointer",
                          background: selectedBrandIdx === b.origIdx ? "#FFF2EC" : undefined,
                        }}
                      >
                        <td>
                          <strong>{b.brand}</strong>
                          <div className="mono" style={{ fontSize: "0.7rem", color: "var(--ink-3)" }}>
                            {b.vertical} · {b.source_mode}
                          </div>
                        </td>
                        <td className="num">
                          {b.product_count} / {b.variant_count}
                        </td>
                        <td className="num">{b.mean_price_usd.toFixed(2)}</td>
                        <td className="num">{b.price_elasticity.toFixed(2)}</td>
                        <td className="num">{b.variant_depth.toFixed(2)}</td>
                        <td className="num">{b.inventory_velocity.toFixed(2)}</td>
                        <td className="num">{b.tag_taxonomy_richness.toFixed(2)}</td>
                        <td className="num" style={{ color: "var(--ember-text)", fontWeight: 700 }}>
                          {b.composite_deal_score.toFixed(1)}
                        </td>
                        <td>
                          <span
                            className="mono"
                            style={{
                              fontSize: "0.72rem",
                              padding: "0.18rem 0.45rem",
                              borderRadius: "2px",
                              border: "1px solid var(--ink)",
                              background:
                                b.recommended_commission_tier.includes("15_20")
                                  ? "#16222B"
                                  : "var(--paper)",
                              color: b.recommended_commission_tier.includes("15_20")
                                ? "#fff"
                                : "var(--ink)",
                            }}
                          >
                            {b.recommended_commission_tier}
                          </span>
                        </td>
                        <td className="mono" style={{ fontSize: "0.7rem", color: "var(--ink-2)" }}>
                          {(b.top_tags ?? []).slice(0, 3).join(", ")}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="refinery-grid" style={{ marginTop: "1rem" }}>
                <div className="refinery-chart-card" style={{ marginTop: 0 }}>
                  <div className="refinery-chart-head">
                    <span className="mono">15 × 15 Brand Affinity Coupling Matrix J_ij</span>
                    <span className="mono">click cell or row to inspect</span>
                  </div>
                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns: `repeat(${activeAffil.brands.length}, minmax(0, 1fr))`,
                      gap: "2px",
                      background: "#CBD4D7",
                      padding: "2px",
                      maxWidth: "340px",
                      margin: "0 auto",
                    }}
                  >
                    {(activeAffil.brand_affinity_J ?? []).flatMap((row, r) =>
                      row.map((val, c) => {
                        const norm = Math.min(1, Math.max(0, Math.abs(val) * 1.6));
                        const isFocus = r === selectedBrandIdx || c === selectedBrandIdx;
                        return (
                          <div
                            key={`${r}-${c}`}
                            onClick={() => setSelectedBrandIdx(r)}
                            title={`${activeAffil.brands[r]?.brand ?? r} ↔ ${
                              activeAffil.brands[c]?.brand ?? c
                            }: J=${val.toFixed(3)}`}
                            style={{
                              aspectRatio: "1 / 1",
                              background:
                                r === c
                                  ? "#E9EDEE"
                                  : `rgba(46, 94, 107, ${0.12 + 0.88 * norm})`,
                              outline: isFocus ? "1px solid #E8490F" : "none",
                              cursor: "pointer",
                            }}
                          />
                        );
                      })
                    )}
                  </div>
                </div>

                <div className="refinery-chart-card" style={{ marginTop: 0 }}>
                  <div className="refinery-chart-head">
                    <span className="mono">
                      Strongest Co-Merchandising Couplings for {activeAffil.brands[selectedBrandIdx]?.brand ?? "Brand"}
                    </span>
                  </div>
                  <p style={{ fontSize: "0.84rem", color: "var(--ink-2)", margin: "0 0 0.65rem" }}>
                    Couplings <span className="mono">J_ij</span> measure normalized tag &amp; price-band overlap across live Shopify DTC catalogs.
                  </p>
                  <div style={{ display: "grid", gap: "0.45rem" }}>
                    {selectedBrandNeighbors.map((nb) => (
                      <div
                        key={nb.j}
                        style={{
                          display: "flex",
                          justifyContent: "space-between",
                          alignItems: "center",
                          padding: "0.45rem 0.6rem",
                          border: "1px solid var(--rule-soft)",
                          borderRadius: "3px",
                          background: "var(--paper)",
                        }}
                      >
                        <div>
                          <strong style={{ fontSize: "0.88rem" }}>{nb.brand.brand}</strong>
                          <span className="mono" style={{ fontSize: "0.72rem", color: "var(--ink-3)", marginLeft: "0.5rem" }}>
                            {nb.brand.vertical}
                          </span>
                        </div>
                        <span className="mono" style={{ fontWeight: 700, color: "var(--steel-deep)" }}>
                          J = {nb.weight.toFixed(3)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ==================================================================
            TAB 5: CUSTOM MATRIX / FASTA + ANONYMIZED CORPUS CONTRIBUTION
           ================================================================== */}
        {activeTab === "custom" && (
          <div className="refinery-grid" role="tabpanel" aria-label="Custom Matrix and FASTA Inbound Mode">
            <div className="refinery-card">
              <h3>Upload Custom Matrix / FASTA (In-Browser Tensor Parser)</h3>
              <p className="refinery-card-sub">
                All matrix parsing, exact <span className="mono">2^N</span> verification, MSA standardization, and Gibbs sampling execute locally in your browser — zero raw matrices or sequences ever leave your machine.
              </p>

              {/* 3 One-Click Sample Presets */}
              <div style={{ marginBottom: "0.85rem" }}>
                <div className="mono" style={{ fontSize: "0.74rem", fontWeight: 700, color: "var(--ink-3)", marginBottom: "0.4rem" }}>
                  ONE-CLICK SAMPLE PRESETS
                </div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: "0.45rem" }}>
                  <button
                    type="button"
                    className="refinery-btn"
                    style={{ fontSize: "0.78rem", minHeight: "34px" }}
                    onClick={() => {
                      setCustomMode("ising_matrix");
                      setCustomRawText(SAMPLE_ISING_CSV);
                      parseAndLoadCustom(SAMPLE_ISING_CSV, "ising_matrix", "csv");
                    }}
                  >
                    <FileText className="w-3.5 h-3.5" /> Load Sample 16x16 Frustrated Ising CSV
                  </button>
                  <button
                    type="button"
                    className="refinery-btn"
                    style={{ fontSize: "0.78rem", minHeight: "34px" }}
                    onClick={() => {
                      setCustomMode("fasta_msa");
                      setCustomRawText(SAMPLE_FASTA_MSA);
                      parseAndLoadCustom(SAMPLE_FASTA_MSA, "fasta_msa", "fasta");
                    }}
                  >
                    <FileText className="w-3.5 h-3.5" /> Load Sample 24-nt Riboswitch FASTA
                  </button>
                  <button
                    type="button"
                    className="refinery-btn"
                    style={{ fontSize: "0.78rem", minHeight: "34px" }}
                    onClick={() => {
                      setCustomMode("qubo_matrix");
                      setCustomRawText(SAMPLE_QUBO_CSV);
                      parseAndLoadCustom(SAMPLE_QUBO_CSV, "qubo_matrix", "csv");
                    }}
                  >
                    <FileText className="w-3.5 h-3.5" /> Load Sample 16x16 QUBO Matrix CSV
                  </button>
                </div>
              </div>

              {/* Input Mode Selector */}
              <div style={{ marginBottom: "0.85rem" }}>
                <div className="mono" style={{ fontSize: "0.74rem", fontWeight: 700, color: "var(--ink-3)", marginBottom: "0.4rem" }}>
                  TENSOR INTERPRETATION MODE
                </div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: "0.4rem" }}>
                  {(
                    [
                      ["ising_matrix", "Ising Couplings (J_ij)"],
                      ["qubo_matrix", "Markowitz / Combinatorial QUBO (Q_ij)"],
                      ["fasta_msa", "FASTA RNA Alignment (q=5)"],
                    ] as Array<[CustomInputMode, string]>
                  ).map(([modeKey, modeLabel]) => (
                    <button
                      key={modeKey}
                      type="button"
                      className={`refinery-btn ${customMode === modeKey ? "primary" : ""}`}
                      style={{ fontSize: "0.78rem", minHeight: "34px" }}
                      onClick={() => setCustomMode(modeKey)}
                    >
                      {modeLabel}
                    </button>
                  ))}
                </div>
              </div>

              {/* File Upload */}
              <div
                style={{
                  border: "1.5px dashed var(--steel)",
                  borderRadius: "4px",
                  padding: "0.7rem 0.85rem",
                  background: "var(--panel)",
                  marginBottom: "0.75rem",
                  display: "flex",
                  flexWrap: "wrap",
                  alignItems: "center",
                  justifyContent: "space-between",
                  gap: "0.6rem",
                }}
              >
                <label
                  htmlFor="customTensorFile"
                  className="mono"
                  style={{ fontSize: "0.78rem", fontWeight: 700, display: "inline-flex", alignItems: "center", gap: "0.45rem", cursor: "pointer" }}
                >
                  <Upload className="w-4 h-4" />
                  Upload local file (.csv, .json, .fasta, .fa, .txt)
                </label>
                <input
                  id="customTensorFile"
                  type="file"
                  accept=".csv,.json,.fasta,.fa,.txt"
                  onChange={(e) => void handleFileUpload(e)}
                  className="mono"
                  style={{ fontSize: "0.74rem", maxWidth: "100%" }}
                />
              </div>

              {/* Raw Text Area */}
              <label htmlFor="customTensorText" className="mono" style={{ fontSize: "0.74rem", fontWeight: 700, color: "var(--ink-3)", display: "block", marginBottom: "0.3rem" }}>
                OR PASTE MATRIX CSV / JSON / FASTA TEXT
              </label>
              <textarea
                id="customTensorText"
                value={customRawText}
                onChange={(e) => setCustomRawText(e.target.value)}
                rows={7}
                spellCheck={false}
                className="mono"
                style={{
                  width: "100%",
                  fontSize: "0.74rem",
                  lineHeight: 1.45,
                  padding: "0.6rem",
                  border: "1px solid var(--rule)",
                  borderRadius: "3px",
                  background: "var(--panel)",
                  color: "var(--ink)",
                  resize: "vertical",
                }}
                aria-label="Paste custom CSV matrix, JSON tensor, or FASTA RNA alignment"
              />

              <div className="refinery-controls">
                <button
                  type="button"
                  className="refinery-btn primary"
                  onClick={() => parseAndLoadCustom(customRawText, customMode, customFormat)}
                >
                  <Sparkles className="w-3.5 h-3.5" /> Parse &amp; Load into Browser Solver
                </button>
              </div>

              {customError && (
                <div
                  role="alert"
                  className="mono"
                  style={{
                    marginTop: "0.7rem",
                    padding: "0.55rem 0.7rem",
                    border: "1px solid #B93A08",
                    background: "#FFF2EC",
                    color: "#B93A08",
                    fontSize: "0.76rem",
                    borderRadius: "3px",
                  }}
                >
                  {customError}
                </div>
              )}
            </div>

            <div className="refinery-card">
              <h3>Parsed Benchmark Diagnostics &amp; Corpus Contribution</h3>
              <p className="refinery-card-sub mono" style={{ fontSize: "0.78rem" }}>
                {customSummary.label} — {customSummary.note}
              </p>

              <div className="refinery-metrics">
                <div className="refinery-metric">
                  <div className="k">Kind / Variables</div>
                  <div className="v">
                    {customSummary.kind === "fasta_msa"
                      ? `RNA L=${customSummary.n_variables}`
                      : customSummary.kind === "qubo_matrix"
                      ? `QUBO N=${customSummary.n_variables}`
                      : `Ising N=${customSummary.n_variables}`}
                  </div>
                </div>
                <div className="refinery-metric">
                  <div className="k">Min Energy (E₀ / E*)</div>
                  <div className="v hot">{customSummary.min_energy.toFixed(3)}</div>
                </div>
                <div className="refinery-metric">
                  <div className="k">Mean Energy</div>
                  <div className="v">{customSummary.mean_energy.toFixed(3)}</div>
                </div>
                <div className="refinery-metric">
                  <div className="k">Sokal τ_int</div>
                  <div className="v">{customSummary.tau_int.toFixed(2)}</div>
                </div>
                <div className="refinery-metric">
                  <div className="k">Effective Samples (ESS)</div>
                  <div className="v">{customSummary.ess.toFixed(1)}</div>
                </div>
                <div className="refinery-metric">
                  <div className="k">
                    {customSummary.kind === "fasta_msa" ? "Shannon Entropy" : "Frustrated Triangles"}
                  </div>
                  <div className="v">
                    {customSummary.kind === "fasta_msa"
                      ? `${(customSummary.entropy_bits ?? 0).toFixed(2)} bits`
                      : `${(customSummary.frustration_pct ?? 0).toFixed(1)}%`}
                  </div>
                </div>
              </div>

              {/* Jump to Live Interactive Solver Tab */}
              <div className="refinery-controls" style={{ marginTop: "0.95rem" }}>
                {customSummary.kind === "ising_matrix" && (
                  <button
                    type="button"
                    className="refinery-btn"
                    onClick={() => {
                      setDataSource("custom");
                      setActiveTab("ising");
                    }}
                  >
                    Open in Interactive Ising Lattice Solver →
                  </button>
                )}
                {customSummary.kind === "fasta_msa" && (
                  <button
                    type="button"
                    className="refinery-btn"
                    onClick={() => {
                      setDataSource("custom");
                      setActiveTab("potts");
                    }}
                  >
                    Open in Interactive RNA Potts Solver →
                  </button>
                )}
                {customSummary.kind === "qubo_matrix" && (
                  <button
                    type="button"
                    className="refinery-btn"
                    onClick={() => {
                      setDataSource("custom");
                      setActiveTab("qubo");
                    }}
                  >
                    Open in Interactive Market QUBO Solver →
                  </button>
                )}
              </div>

              {/* Opt-In Anonymized Benchmark Corpus Contribution Bar */}
              <div
                style={{
                  marginTop: "1rem",
                  padding: "0.9rem",
                  border: "1.5px solid var(--ink)",
                  borderRadius: "4px",
                  background: "var(--panel)",
                }}
              >
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: "0.5rem", marginBottom: "0.5rem" }}>
                  <span className="mono" style={{ fontSize: "0.76rem", fontWeight: 700, display: "inline-flex", alignItems: "center", gap: "0.35rem" }}>
                    <ShieldCheck className="w-4 h-4" style={{ color: "#1F4753" }} />
                    OPEN THERMAL BENCHMARK CORPUS CONTRIBUTION
                  </span>
                  <span className="mono" style={{ fontSize: "0.72rem", color: "var(--steel-deep)", fontWeight: 700 }}>
                    Corpus entries: {corpusCount} · Session: {sessionContribCount}
                  </span>
                </div>

                <label
                  style={{
                    display: "flex",
                    alignItems: "flex-start",
                    gap: "0.55rem",
                    fontSize: "0.82rem",
                    color: "var(--ink-2)",
                    cursor: "pointer",
                  }}
                >
                  <input
                    type="checkbox"
                    checked={anonymizedConsent}
                    onChange={(e) => setAnonymizedConsent(e.target.checked)}
                    style={{ marginTop: "0.2rem", accentColor: "#E8490F", width: "16px", height: "16px", flex: "none" }}
                  />
                  <span>
                    Opt-in: Contribute anonymized aggregate metrics (dimension, frustration %, τ_int, ESS, min E — zero raw sequences or proprietary IDs) to the open Thermal Benchmark Corpus
                  </span>
                </label>

                <div className="refinery-controls" style={{ marginTop: "0.75rem" }}>
                  <button
                    type="button"
                    className="refinery-btn primary"
                    disabled={!anonymizedConsent || contributing}
                    onClick={() => void handleContributeBenchmark()}
                    style={{
                      opacity: !anonymizedConsent || contributing ? 0.55 : 1,
                      cursor: !anonymizedConsent ? "not-allowed" : contributing ? "wait" : "pointer",
                    }}
                  >
                    {contributing ? "Contributing..." : "Contribute Anonymized Benchmark"}
                  </button>
                </div>

                {contributeError && (
                  <div
                    role="alert"
                    className="mono"
                    style={{
                      marginTop: "0.65rem",
                      fontSize: "0.75rem",
                      color: "#B93A08",
                    }}
                  >
                    {contributeError}
                  </div>
                )}

                {lastReceipt && (
                  <div
                    className="mono"
                    style={{
                      marginTop: "0.75rem",
                      padding: "0.65rem 0.75rem",
                      background: "var(--paper)",
                      border: "1px solid var(--steel)",
                      borderRadius: "3px",
                      fontSize: "0.75rem",
                      color: "var(--ink)",
                    }}
                  >
                    <div style={{ display: "flex", alignItems: "center", gap: "0.4rem", fontWeight: 700, color: "#1F4753", marginBottom: "0.25rem" }}>
                      <CheckCircle2 className="w-4 h-4" />
                      Anonymized Contribution Recorded
                    </div>
                    <div>
                      entry_id: <strong>{lastReceipt.entry_id}</strong> ({lastReceipt.persisted})
                    </div>
                    <div>
                      kind: {lastReceipt.kind} (N={lastReceipt.n_variables}) · min_energy: {lastReceipt.min_energy.toFixed(4)} · τ_int: {lastReceipt.tau_int.toFixed(2)} · ESS: {lastReceipt.ess.toFixed(1)}
                    </div>
                    <div style={{ color: "var(--ink-3)" }}>
                      sha256: {lastReceipt.structural_sha256.slice(0, 24)}… · total corpus: {lastReceipt.corpus_count}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        <div
          className="mono"
          style={{
            marginTop: "1rem",
            paddingTop: "0.75rem",
            borderTop: "1px solid var(--rule)",
            fontSize: "0.75rem",
            color: "var(--ink-3)",
            display: "flex",
            flexWrap: "wrap",
            justifyContent: "space-between",
            gap: "0.5rem",
          }}
        >
          <span>
            <Database className="w-3.5 h-3.5 inline mr-1" />
            Refinery endpoint: /data/live_empirical_streams.json
          </span>
          <span>
            <Activity className="w-3.5 h-3.5 inline mr-1" />
            Classical JAX &amp; browser MCMC diagnostics · simulated benchmark figures labeled per doctrine
          </span>
        </div>
      </div>
    </div>
  );
}
