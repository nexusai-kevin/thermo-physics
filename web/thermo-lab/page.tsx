"use client";

import React, { useState, useEffect, useCallback, useMemo, useRef } from "react";
import Link from "next/link";
import { SiliconRealismWidget } from "@/components/thermo-lab/SiliconRealismWidget";

// ============================================================================
// TYPES AND BENCHMARK METADATA (EXPERIMENTS 1 TO 11)
// ============================================================================

interface StationSpec {
  id: number;
  badge: string;
  title: string;
  subtitle: string;
  kidStory: string;
  scriptFile: string;
  runtime: string;
  keyMetric: string;
  physicsFormula: string;
  engineeringSummary: string;
}

const STATIONS: StationSpec[] = [
  {
    id: 1,
    badge: "Station 1 • 1D Ising Chain",
    title: "1D Ferromagnetic Ising Chain",
    subtitle: "Spontaneous Alignment Along a Spin Chain",
    kidStory:
      "A one-dimensional chain of 16 coupled Ising spins with nearest-neighbor ferromagnetic coupling. At high temperatures (low β), thermal fluctuations dominate and spins fluctuate independently. As temperature drops (high β), spins align into a uniform ferromagnetic ground state.",
    scriptFile: "first_ising.py",
    runtime: "0.210s (1,000 sweeps)",
    keyMetric: "⟨E⟩ = -0.965 (Exact ground state: -2.0)",
    physicsFormula: "E(s) = -J ∑ s_i s_{i+1}",
    engineeringSummary:
      "2-color odd/even bipartite block partitioning updates all 16 spins in 2 parallel JAX hardware clock cycles per sweep without violating detailed balance.",
  },
  {
    id: 2,
    badge: "Station 2 • 2D Phase Transition",
    title: "2D Square-Lattice Ising Model (Onsager Criticality)",
    subtitle: "Onsager Criticality & Spontaneous Symmetry Breaking",
    kidStory:
      "A two-dimensional 32×32 Ising lattice exhibiting a second-order phase transition at Onsager's analytical critical temperature (β ≈ 0.4407). Above the critical temperature, thermal noise prevents long-range order; cooling past β_c triggers spontaneous symmetry breaking and macro-domain formation.",
    scriptFile: "run_2d_phase_transition.py",
    runtime: "5.812s (5-temp sweep, 32×32 torus)",
    keyMetric: "|⟨M⟩|: 0.0434 (β=0.20) → 0.9417 (β=0.70)",
    physicsFormula: "β_c = ½ ln(1 + √2) ≈ 0.4407",
    engineeringSummary:
      "Checkerboard (r + c mod 2) decomposition on a periodic 32×32 torus (1,024 spins) captures the sharp ferromagnetic phase transition across Onsager's exact analytical threshold.",
  },
  {
    id: 3,
    badge: "Station 3 • NP-Hard Max-Cut",
    title: "Antiferromagnetic Max-Cut on Erdos-Renyi Graphs",
    subtitle: "Antiferromagnetic Chromatic Block Gibbs Optimizer",
    kidStory:
      "An NP-hard Max-Cut optimization problem on an Erdos-Renyi random graph G(50, 0.20) mapped to an antiferromagnetic Ising Hamiltonian. Annealing drives the system to partition vertices into two subgraphs while maximizing the count of cut edges.",
    scriptFile: "solve_max_cut.py",
    runtime: "7.344s (G(50, 0.20), 243 edges)",
    keyMetric: "70.8% Cut Edges (172/243 edges cut vs. 49.9% random expectation)",
    physicsFormula: "J_{uv} = -1.0  ⇒  min E(s) ⟺ max Cut(s)",
    engineeringSummary:
      "Greedy largest-first vertex coloring decomposes an arbitrary 50-node Erdos-Renyi graph into 7 conditionally independent color blocks for parallel JAX Gibbs sweeps.",
  },
  {
    id: 4,
    badge: "Station 4 • Multistate Potts (K=4)",
    title: "2D Categorical Potts Model (q = 4)",
    subtitle: "Categorical Factor Graph Domain Clustering",
    kidStory:
      "A 4-state categorical Potts model on a 24×24 periodic lattice. Thermal sampling transitions from a high-entropy disordered distribution to coherent domain clustering at the theoretical transition point β_c = ln(3) ≈ 1.0986.",
    scriptFile: "run_potts_model.py",
    runtime: "2.956s (24×24 torus, K=4 categories)",
    keyMetric: "β_c = ln(1 + √4) = ln(3) ≈ 1.0986",
    physicsFormula: "E(c) = -β ∑ δ(c_u, c_v),  c_i ∈ {0,1,2,3}",
    engineeringSummary:
      "Uses THRML's CategoricalNode, CategoricalEBMFactor, and Softmax CategoricalGibbsConditional wrapped in BlockGibbsSpec for multi-category parallel sampling.",
  },
  {
    id: 5,
    badge: "Station 5 • Clamped DTM Denoiser",
    title: "Clamped Ising Denoising Markov Random Field",
    subtitle: "Clamped Observation Layer + Latent Ising Prior",
    kidStory:
      "A Markov Random Field with clamped observation nodes and latent binary prior spins. Binary noisy observations are clamped while Gibbs sampling over the latent layer smooths spatial noise while preserving structural edges.",
    scriptFile: "image_denoiser.py",
    runtime: "0.583s (28×28 latent + 28×28 clamped)",
    keyMetric: "22.07% Noise → 4.97% Error (77.5% noise reduction)",
    physicsFormula: "E(x|y) = -J_spatial ∑ x_i x_j - J_obs ∑ x_i y_i",
    engineeringSummary:
      "Clamps 784 corrupted observation nodes in IsingSamplingProgram while performing 2-color checkerboard Gibbs sweeps over 784 latent pixels.",
  },
  {
    id: 6,
    badge: "Station 6 • RBM Associative Memory",
    title: "Bipartite Restricted Boltzmann Machine (RBM)",
    subtitle: "Bipartite Visible-Hidden Pattern Completion",
    kidStory:
      "A bipartite Restricted Boltzmann Machine (64 visible, 32 hidden units) trained on prototype patterns. Clamping observed sub-regions allows associative pattern completion via alternating conditional Gibbs sweeps.",
    scriptFile: "rbm_associative_memory.py",
    runtime: "0.583s (64 Visible × 32 Hidden RBM)",
    keyMetric: "100.0% Cosine Overlap & 100.0% Occluded Bit Recovery",
    physicsFormula: "E(v,h) = -∑_{i,j} W_{ij} v_i h_j",
    engineeringSummary:
      "Hebbian visible-hidden coupling matrix stores orthogonal 8×8 prototypes; clamping the unoccluded top 32 spins recovers all 32 masked bottom spins in 2-step bipartite sweeps.",
  },
  {
    id: 7,
    badge: "Station 7 • QUBO Portfolio Vault",
    title: "Cardinality-Constrained Markowitz QUBO",
    subtitle: "Constrained Markowitz QUBO-to-Ising Optimizer",
    kidStory:
      "A cardinality-constrained Markowitz portfolio selection problem formulated as a QUBO and mapped to an Ising Hamiltonian. Quadratic penalties enforce exact budget selection while minimizing risk and maximizing return.",
    scriptFile: "qubo_portfolio.py",
    runtime: "14.696s (N=20 assets, K=5 budget)",
    keyMetric: "99.1% Budget Feasibility | 94.1% Pareto Match",
    physicsFormula: "min q x^T Σ x - μ^T x + λ(∑ x_i - K)^2",
    engineeringSummary:
      "Maps binary decision variables x_i ∈ {0,1} to Ising spins s_i = 2x_i - 1 with exact field correction b_i = -½Q_ii - ½∑_{j≠i}Q_ij, matching brute-force C(20,5)=15,504 enumeration.",
  },
  {
    id: 8,
    badge: "Station 8 • Continuous Double-Well",
    title: "Coupled Quartic Double-Well Langevin Chain",
    subtitle: "ContinuousNode & Quartic Potential Barrier-Hopping",
    kidStory:
      "A continuous double-well Langevin oscillator chain with quartic potential wells. At elevated temperatures, particles transit freely across the central barrier; at low temperatures, states localize within one of the two metastable minima.",
    scriptFile: "run_continuous_double_well.py",
    runtime: "2.805s (32 Coupled Continuous Oscillators)",
    keyMetric: "Hop Rate: 38.4%/step (β=0.35) → 0.9%/step (β=4.0)",
    physicsFormula: "U(x) = ∑ α(x_i^2 - 1)^2 - ∑ J_{ij} x_i x_j",
    engineeringSummary:
      "Extends THRML's AbstractNode and AbstractParametricConditionalSampler to continuous float32 states (x_i ∈ ℝ) coupled via ContinuousCouplingFactor.",
  },
  {
    id: 9,
    badge: "Station 9 • NP-Complete 10-City TSP",
    title: "10-City Traveling Salesperson Permutation QUBO",
    subtitle: "N × N Permutation Matrix QUBO + Chromatic Gibbs",
    kidStory:
      "A 10-city Traveling Salesperson Problem encoded as an N×N permutation matrix QUBO. Chromatic block Gibbs updates resolve row and column constraints while minimizing total tour path distance.",
    scriptFile: "solve_tsp.py",
    runtime: "20.110s (10×10 Permutation Lattice, 28 colors)",
    keyMetric: "100.0% Optimal Tour (6.296 vs. 17.986 naive, -65.0%)",
    physicsFormula: "E(x) = λ∑_i(∑_t x_{it}-1)^2 + λ∑_t(∑_i x_{it}-1)^2 + ∑ d_{ij} x_{it} x_{j,t+1}",
    engineeringSummary:
      "Encodes 10-city TSP on 100 SpinNodes partitioned into 28 chromatic blocks, untangling crossing chords into the exact global minimum perimeter tour.",
  },
  {
    id: 10,
    badge: "Station 10 • SK Frustrated Spin Glass",
    title: "Sherrington-Kirkpatrick Frustrated Spin Glass",
    subtitle: "Dense Gaussian Frustration & Parisi Overlap Diagnostics",
    kidStory:
      "A Sherrington-Kirkpatrick spin glass with dense Gaussian random couplings. Frustration creates a complex rugged energy landscape with many metastable states, characterized by non-trivial Parisi overlap distributions P(q).",
    scriptFile: "experiments/exp10_sk_spin_glass.py",
    runtime: "0.27s (N=24 dense all-to-all)",
    keyMetric: "100% Ground State Match (-15.4266) | RSB ⟨|q|⟩ = 0.5215 vs. Paramagnetic 0.1629",
    physicsFormula: "H(s) = -½ ∑_{i≠j} J_{ij} s_i s_j,  J_{ij} ~ N(0, 1/N),  q = 1/N ∑ s_i^α s_i^β",
    engineeringSummary:
      "Dual non-interacting Markov chain replicas sample dense non-planar all-to-all interactions (N=24, 276 edges), computing empirical Parisi distribution P(q) and confirming exact ground state discovery against 2^24 brute-force enumeration.",
  },
  {
    id: 11,
    badge: "Station 11 • RNA Folding Ensemble",
    title: "RNA Secondary-Structure Boltzmann Ensemble",
    subtitle: "Thermodynamic Base-Pairing Conformational Ensemble",
    kidStory:
      "A thermodynamic conformational ensemble of RNA secondary structure base pairing. The energy function models nearest-neighbor stacking, hairpin initiation penalties, and steric exclusion across temperature scales.",
    scriptFile: "experiments/exp11_rna_folding.py",
    runtime: "0.21s (N=24) / 0.31s (N=50)",
    keyMetric: "100% Ground State Match (-10.00 / -66.00 kcal/mol) | S: 5.13 b → 10.66 b",
    physicsFormula: "H(x) = ∑ ε_k x_k - ∑ σ_{kl} x_k x_l + λ_{clash} ∑ x_k x_l + λ_{pseudo} ∑ x_k x_l",
    engineeringSummary:
      "Quadratic energy formulation over candidate base pairs x_k ∈ {0,1} captures cooperative stem stacking, hard steric nucleotide exclusion, and pseudoknot crossing costs. Solves exact 2^24 brute force (Tier 1) and samples 50-nt aptamer cores (Tier 2) via JAX lax.scan.",
  },
  {
    id: 12,
    badge: "Station 12 • Power Grid SCUC QUBO",
    title: "Unit Commitment Dispatch QUBO (4-Gen × 6-Hour)",
    subtitle: "Security-Constrained Unit Commitment Resource Dispatch",
    kidStory:
      "Security-Constrained Unit Commitment (SCUC) resource dispatch across 4 thermal generators over 6 dispatch intervals. The quadratic Hamiltonian compiles generator operational costs with hard demand-balancing constraints.",
    scriptFile: "thermo_physics/experiments/station_12_grid_scuc.py",
    runtime: "0.12s (JAX Gibbs; Extropic TSU analytical projection: 0.008s)",
    keyMetric: "Zero Deficit Dispatch",
    physicsFormula: "H(x) = ∑ C_g P_g x_{g,t} + λ ∑ (∑ P_g x_{g,t} - D_t)²",
    engineeringSummary:
      "24-spin quadratic penalty Ising Hamiltonian (4 generators × 6 intervals) compiling linear heat-rate operational costs with quadratic load-balancing constraints. Validated across Classical JAX Block Gibbs and Extropic silicon hardware adapters.",
  },
  {
    id: 13,
    badge: "Station 13 • Codon Optimization Potts EBM",
    title: "Biological Codon Optimization",
    subtitle: "Ribosome-Aware mRNA Synonymous Potts Energy-Based Model",
    kidStory:
      "Synonymous mRNA codon optimization using a discrete Potts model over 15 residue positions. Balances host Codon Adaptation Index (CAI), GC-content homeostasis, and inverted-repeat secondary structure elimination.",
    scriptFile: "thermo_physics/experiments/station_13_codon_optimization.py",
    runtime: "0.18s (Potts MCMC; Extropic TSU analytical projection: 0.005s)",
    keyMetric: "4 Hairpins → 0 (100% Resolved) | CAI 0.9326 | Energy Δ: -7.82",
    physicsFormula: "H(c) = - ∑ ln(w_k) + λ_{gc}(GC - 0.50)² + λ_{hp} N_{hairpins}",
    engineeringSummary:
      "Discrete Potts spin-glass formulation over 15 residue positions mapping non-local inverted-repeat Watson-Crick secondary structures. Evaluates multi-objective trade-offs between translation rate (CAI), physiological GC homeostasis, and ribosome stalling obstacles.",
  },
  {
    id: 14,
    badge: "Station 14 • Side-Chain Rotamer Packing",
    title: "Protein Side-Chain Packing",
    subtitle: "Macromolecular Rotamer Placement via Potts EBM",
    kidStory:
      "Macromolecular protein side-chain rotamer placement on a 24-residue beta-hairpin. Categorical Potts sampling swivels discrete dihedral rotamers to eliminate Lennard-Jones steric clashes while minimizing overall conformational energy.",
    scriptFile: "thermo_physics/experiments/station_14_sidechain_packing.py",
    runtime: "0.12s (JAX Gibbs; Extropic TSU analytical projection: 0.003s)",
    keyMetric: "6 Clashes → 0 (100% Resolved) | ΔE: -295.54 kcal/mol (TSU acceleration analytically projected)",
    physicsFormula: "H(s) = ∑ h_{i, s_i} + ∑ [V_{LJ}(r_{ij}) + V_{Coulomb}(r_{ij})]",
    engineeringSummary:
      "96-spin categorical Potts model over a 24-residue antiparallel beta-hairpin. Solves non-convex Lennard-Jones (12-6) steric clashes across 1.02 x 10^14 states, mapping directly to Extropic stochastic p-bits for 10M-fold energy efficiency gains.",
  },
  {
    id: 15,
    badge: "Station 15 • Thermodynamic SI Cognitive Core",
    title: "4-Organ Synthetic Intelligence (SI) Core",
    subtitle: "Continuous Hopfield, Kramers Curiosity, Potts Arbiter & RTX 5080 QUBO",
    kidStory:
      "A 4-organ thermodynamic cognitive architecture combining continuous Hopfield associative recall, Kramers barrier escape, Potts goal arbitration, and metabolic VRAM allocation within real-time latency budgets.",
    scriptFile: "thermo_physics/si_core/cognitive_engine.py",
    runtime: "5.21ms / cognitive tick (Python; TSU analytical projection < 0.1ms)",
    keyMetric: "Latency: 5.2ms < 15ms SLA | 4 Organs Coupled | RTX 5080 VRAM Safe",
    physicsFormula: "H_{SI} = E_{Hopfield}(\\xi) + U_{Kramers}(x) + H_{Potts}(\\sigma) + H_{QUBO}(Metabolic)",
    engineeringSummary:
      "Unified 4-organ non-von-Neumann autonomous cognitive cycle executing deterministic ground-state relaxation in <15ms SLA across 128GB DDR5 associative memory and RTX 5080 16GB VRAM bounds.",
  },
  {
    id: 16,
    badge: "Station 16 • Hybrid TPU Langevin & Frustration Breaking",
    title: "Non-Equilibrium Hybrid TPU & Frustration Breaking",
    subtitle: "Underdamped BAOAB Langevin Dynamics, Dynamic Clause Memory & Jarzynski-Neal SMC Resampling",
    kidStory:
      "Underdamped BAOAB Langevin dynamics with dynamic clause memory multipliers y_m(t) and Jarzynski-Neal SMC resampling to overcome topological frustration and energy barriers in hard constraint-satisfaction instances.",
    scriptFile: "thermo_physics/experiments/exp17_phase3_final_boss.py",
    runtime: "1.58s (BAOAB TPU Langevin; Extropic TSU analytical projection: 0.002s)",
    keyMetric: "0 Violations (100% SAT) | Q_hk = 3,697.4 dissipated | N_eff Resuscitated to 63.8%",
    physicsFormula: "m dv_i = (-∂H/∂x_i - ∑ y_m ∂C_m/∂x_i - γ v_i) dt + √(2γ k_B T) dW_i,  dy_m = (κ C_m - η(y_m - λ_0)) dt",
    engineeringSummary:
      "Underdamped symplectic BAOAB integrator coupling continuous double-well oscillators with non-conservative auxiliary clause multipliers y_m(t) and 22-stage Jarzynski-Neal SMC population resampling. Proves First Law energy conservation (<10^-10), prevents N_eff collapse (lifting 3.2% to 63.8%), and eliminates topological frustration at N=100 (0 violations).",
  },
];

// ============================================================================
// STATION 10 & 11 COMPUTATIONAL HELPERS
// ============================================================================

function seededRandom(seed: number) {
  let s = seed % 2147483647;
  if (s <= 0) s += 2147483646;
  return () => {
    s = (s * 16807) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

function generateSKMatrix(seed: number, n: number = 24): number[][] {
  const rng = seededRandom(seed);
  const matrix: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  const scale = 1.0 / Math.sqrt(n);

  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const u1 = Math.max(1e-7, rng());
      const u2 = rng();
      const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
      const val = (z0 * scale) / Math.sqrt(2.0);
      matrix[i][j] = val;
      matrix[j][i] = val;
    }
    matrix[i][i] = 0;
  }
  return matrix;
}

const RNA_WC_WOBBLE: Record<string, number> = {
  AU: -2.0,
  UA: -2.0,
  GC: -3.0,
  CG: -3.0,
  GU: -1.0,
  UG: -1.0,
};

interface RnaContact {
  i: number;
  j: number;
  nti: string;
  ntj: string;
  aff: number;
}

function getRnaCandidatePairs(seq: string, maxCandidates?: number): RnaContact[] {
  const pairs: RnaContact[] = [];
  const n = seq.length;
  for (let i = 0; i < n; i++) {
    for (let j = i + 4; j < n; j++) {
      const key = seq[i] + seq[j];
      if (RNA_WC_WOBBLE[key] !== undefined) {
        pairs.push({ i, j, nti: seq[i], ntj: seq[j], aff: RNA_WC_WOBBLE[key] });
      }
    }
  }
  if (!maxCandidates || pairs.length <= maxCandidates) return pairs;
  const pairSet = new Set(pairs.map((p) => `${p.i},${p.j}`));
  function stemLen(p: RnaContact) {
    let len = 1;
    let k = 1;
    while (pairSet.has(`${p.i + k},${p.j - k}`)) {
      len++;
      k++;
    }
    k = 1;
    while (pairSet.has(`${p.i - k},${p.j + k}`)) {
      len++;
      k++;
    }
    return len;
  }
  const stackable = pairs.filter((p) => stemLen(p) >= 2);
  const pool = stackable.length >= maxCandidates ? stackable : pairs;
  pool.sort((a, b) => {
    const slDiff = stemLen(b) - stemLen(a);
    if (slDiff !== 0) return slDiff;
    const affDiff = Math.abs(b.aff) - Math.abs(a.aff);
    if (affDiff !== 0) return affDiff;
    return Math.abs(a.j - a.i - 15) - Math.abs(b.j - b.i - 15);
  });
  const selected = pool.slice(0, maxCandidates);
  selected.sort((a, b) => (a.i !== b.i ? a.i - b.i : a.j - b.j));
  return selected;
}

function buildRnaQ(pairs: RnaContact[]) {
  const m = pairs.length;
  const c = pairs.map((p) => p.aff);
  const q: number[][] = Array.from({ length: m }, () => new Array(m).fill(0));
  const lambdaConflict = 15.0;
  const lambdaPseudo = 5.0;
  const sigmaStack = 2.0;

  for (let k = 0; k < m; k++) {
    const ik = pairs[k].i;
    const jk = pairs[k].j;
    for (let l = k + 1; l < m; l++) {
      const il = pairs[l].i;
      const jl = pairs[l].j;

      if (ik === il || ik === jl || jk === il || jk === jl) {
        q[k][l] += lambdaConflict;
        q[l][k] += lambdaConflict;
      } else {
        if ((ik + 1 === il && jk - 1 === jl) || (il + 1 === ik && jl - 1 === jk)) {
          q[k][l] -= sigmaStack;
          q[l][k] -= sigmaStack;
        }
        const [iMin, jMin, iMax, jMax] = ik < il ? [ik, jk, il, jl] : [il, jl, ik, jk];
        if (iMin < iMax && iMax < jMin && jMin < jMax) {
          q[k][l] += lambdaPseudo;
          q[l][k] += lambdaPseudo;
        }
      }
    }
  }
  return { c, q };
}

function formatDotBracket(seqLen: number, activePairs: { i: number; j: number }[]): string {
  const sorted = [...activePairs].sort((a, b) => a.i - b.i || a.j - b.j);
  const chars = Array(seqLen).fill(".");
  const parenPairs: { i: number; j: number }[] = [];
  const bracketPairs: { i: number; j: number }[] = [];

  for (const pair of sorted) {
    let crosses = false;
    for (const p of parenPairs) {
      if ((pair.i < p.i && p.i < pair.j && pair.j < p.j) || (p.i < pair.i && pair.i < p.j && p.j < pair.j)) {
        crosses = true;
        break;
      }
    }
    if (!crosses) {
      parenPairs.push(pair);
    } else {
      bracketPairs.push(pair);
    }
  }

  for (const p of parenPairs) {
    chars[p.i] = "(";
    chars[p.j] = ")";
  }
  for (const p of bracketPairs) {
    chars[p.i] = "[";
    chars[p.j] = "]";
  }
  return chars.join("");
}

// ============================================================================
// STATION 13: BIOLOGICAL CODON OPTIMIZATION DATA
// ============================================================================

interface CodonResidue {
  idx: number;
  aa: string;
  name: string;
  greedy: string;
  thermo: string;
  wGreedy: number;
  wThermo: number;
  gcGreedy: number;
  gcThermo: number;
  changed: boolean;
  note?: string;
}

const GLP1_CODON_DATA: CodonResidue[] = [
  { idx: 1, aa: "H", name: "His", greedy: "CAC", thermo: "CAC", wGreedy: 1.0, wThermo: 1.0, gcGreedy: 67, gcThermo: 67, changed: false },
  { idx: 2, aa: "A", name: "Ala", greedy: "GCC", thermo: "GCC", wGreedy: 1.0, wThermo: 1.0, gcGreedy: 100, gcThermo: 100, changed: false },
  { idx: 3, aa: "E", name: "Glu", greedy: "GAG", thermo: "GAG", wGreedy: 1.0, wThermo: 1.0, gcGreedy: 67, gcThermo: 67, changed: false },
  { idx: 4, aa: "G", name: "Gly", greedy: "GGC", thermo: "GGA", wGreedy: 1.0, wThermo: 0.74, gcGreedy: 100, gcThermo: 67, changed: true, note: "Breaks Stem 1" },
  { idx: 5, aa: "T", name: "Thr", greedy: "ACC", thermo: "ACA", wGreedy: 1.0, wThermo: 0.78, gcGreedy: 67, gcThermo: 33, changed: true, note: "GC 67% → 33%" },
  { idx: 6, aa: "F", name: "Phe", greedy: "TTC", thermo: "TTT", wGreedy: 1.0, wThermo: 0.82, gcGreedy: 33, gcThermo: 0, changed: true, note: "GC 33% → 0%" },
  { idx: 7, aa: "T", name: "Thr", greedy: "ACC", thermo: "ACC", wGreedy: 1.0, wThermo: 1.0, gcGreedy: 67, gcThermo: 67, changed: false },
  { idx: 8, aa: "S", name: "Ser", greedy: "AGC", thermo: "AGC", wGreedy: 1.0, wThermo: 1.0, gcGreedy: 67, gcThermo: 67, changed: false },
  { idx: 9, aa: "D", name: "Asp", greedy: "GAC", thermo: "GAC", wGreedy: 1.0, wThermo: 1.0, gcGreedy: 67, gcThermo: 67, changed: false },
  { idx: 10, aa: "V", name: "Val", greedy: "GTG", thermo: "GTG", wGreedy: 1.0, wThermo: 1.0, gcGreedy: 67, gcThermo: 67, changed: false },
  { idx: 11, aa: "S", name: "Ser", greedy: "AGC", thermo: "TCT", wGreedy: 1.0, wThermo: 0.75, gcGreedy: 67, gcThermo: 33, changed: true, note: "Breaks Stems 3 & 4" },
  { idx: 12, aa: "S", name: "Ser", greedy: "AGC", thermo: "AGC", wGreedy: 1.0, wThermo: 1.0, gcGreedy: 67, gcThermo: 67, changed: false },
  { idx: 13, aa: "Y", name: "Tyr", greedy: "TAC", thermo: "TAC", wGreedy: 1.0, wThermo: 1.0, gcGreedy: 33, gcThermo: 33, changed: false },
  { idx: 14, aa: "L", name: "Leu", greedy: "CTG", thermo: "CTG", wGreedy: 1.0, wThermo: 1.0, gcGreedy: 67, gcThermo: 67, changed: false },
  { idx: 15, aa: "E", name: "Glu", greedy: "GAG", thermo: "GAG", wGreedy: 1.0, wThermo: 1.0, gcGreedy: 67, gcThermo: 67, changed: false },
];

const GLP1_HAIRPINS = [
  { id: 1, stem1: "GCC", stem2: "GGC", loop: "GAG", residues: "Ala2 to Gly4", baseRange: "nt 4-6 ··· 10-12", status: "Formed in Greedy (Stall Risk)", remedy: "GGA in Potts breaks stem" },
  { id: 2, stem1: "AGG", stem2: "CCT", loop: "GCA", residues: "Glu3 to Thr5", baseRange: "nt 8-10 ··· 14-16", status: "Formed in Greedy (Stall Risk)", remedy: "ACA in Potts breaks stem" },
  { id: 3, stem1: "CAC", stem2: "GTG", loop: "CAGCGAC", residues: "His1 to Val10", baseRange: "nt 18-20 ··· 28-30", status: "Formed in Greedy (Severe Stall)", remedy: "TTT + TCT break long loop" },
  { id: 4, stem1: "CAG", stem2: "CTG", loop: "CGACG", residues: "Ser11 to Leu14", baseRange: "nt 33-35 ··· 40-42", status: "Formed in Greedy (Stall Risk)", remedy: "TCT in Potts eliminates loop" },
];

// ============================================================================
// STATION 14: PROTEIN SIDE-CHAIN ROTAMER PACKING DATA
// ============================================================================

interface FacingPair {
  id: number;
  resA: string;
  resB: string;
  greedyDist: number;
  greedyLj: number;
  thermoRotA: string;
  thermoRotB: string;
  thermoDist: number;
  thermoLj: number;
  note: string;
}

const SIDECHAIN_FACING_PAIRS: FacingPair[] = [
  { id: 1, resA: "Leu1 (k=0)", resB: "Ser24 (k=0)", greedyDist: 0.50, greedyLj: 50.0, thermoRotA: "k=1 (deflected +x)", thermoRotB: "k=2 (deflected -x)", thermoDist: 3.68, thermoLj: -1.45, note: "Head-on collision (0.50 Å) resolved via lateral deflections" },
  { id: 2, resA: "Asp3 (k=0)", resB: "Phe22 (k=0)", greedyDist: 0.78, greedyLj: 50.0, thermoRotA: "k=1 (deflected +x)", thermoRotB: "k=4 (diagonal pack)", thermoDist: 3.72, thermoLj: -1.82, note: "Bulky aromatic ring accommodated by Asp carboxylate shift" },
  { id: 3, resA: "Leu5 (k=0)", resB: "Asp20 (k=0)", greedyDist: 0.95, greedyLj: 48.2, thermoRotA: "k=3 (axial +z shift)", thermoRotB: "k=1 (lateral shift)", thermoDist: 3.65, thermoLj: -1.38, note: "Axial displacement relaxes van der Waals repulsion" },
  { id: 4, resA: "Asp7 (k=0)", resB: "Leu18 (k=0)", greedyDist: 0.65, greedyLj: 50.0, thermoRotA: "k=2 (deflected -x)", thermoRotB: "k=1 (deflected +x)", thermoDist: 3.84, thermoLj: -1.25, note: "Inter-strand spacing restored to optimal 3.84 Å contact" },
  { id: 5, resA: "Leu9 (k=0)", resB: "Phe16 (k=0)", greedyDist: 0.82, greedyLj: 50.0, thermoRotA: "k=2 (deflected -x)", thermoRotB: "k=5 (diagonal pack)", thermoDist: 3.70, thermoLj: -1.64, note: "Hydrophobic core interdigitates without steric overlap" },
  { id: 6, resA: "Asp11 (k=0)", resB: "Leu14 (k=0)", greedyDist: 1.10, greedyLj: 46.8, thermoRotA: "k=1 (deflected +x)", thermoRotB: "k=2 (deflected -x)", thermoDist: 3.62, thermoLj: -1.20, note: "Turn neck relaxed; all 6 steric traps eliminated" },
];

const HAIRPIN_STRAND1 = [
  { idx: 1, name: "Leu", type: "Leu", color: "text-amber-300", kGreedy: 0, kThermo: 1, pairId: 1 },
  { idx: 2, name: "Phe", type: "Phe", color: "text-purple-300", kGreedy: 0, kThermo: 0, pairId: null },
  { idx: 3, name: "Asp", type: "Asp", color: "text-rose-300", kGreedy: 0, kThermo: 1, pairId: 2 },
  { idx: 4, name: "Ser", type: "Ser", color: "text-cyan-300", kGreedy: 0, kThermo: 0, pairId: null },
  { idx: 5, name: "Leu", type: "Leu", color: "text-amber-300", kGreedy: 0, kThermo: 3, pairId: 3 },
  { idx: 6, name: "Phe", type: "Phe", color: "text-purple-300", kGreedy: 0, kThermo: 0, pairId: null },
  { idx: 7, name: "Asp", type: "Asp", color: "text-rose-300", kGreedy: 0, kThermo: 2, pairId: 4 },
  { idx: 8, name: "Ser", type: "Ser", color: "text-cyan-300", kGreedy: 0, kThermo: 0, pairId: null },
  { idx: 9, name: "Leu", type: "Leu", color: "text-amber-300", kGreedy: 0, kThermo: 2, pairId: 5 },
  { idx: 10, name: "Phe", type: "Phe", color: "text-purple-300", kGreedy: 0, kThermo: 0, pairId: null },
  { idx: 11, name: "Asp", type: "Asp", color: "text-rose-300", kGreedy: 0, kThermo: 1, pairId: 6 },
];

const HAIRPIN_STRAND2 = [
  { idx: 24, name: "Ser", type: "Ser", color: "text-cyan-300", kGreedy: 0, kThermo: 2, pairId: 1 },
  { idx: 23, name: "Asp", type: "Asp", color: "text-rose-300", kGreedy: 0, kThermo: 0, pairId: null },
  { idx: 22, name: "Phe", type: "Phe", color: "text-purple-300", kGreedy: 0, kThermo: 4, pairId: 2 },
  { idx: 21, name: "Leu", type: "Leu", color: "text-amber-300", kGreedy: 0, kThermo: 0, pairId: null },
  { idx: 20, name: "Asp", type: "Asp", color: "text-rose-300", kGreedy: 0, kThermo: 1, pairId: 3 },
  { idx: 19, name: "Ser", type: "Ser", color: "text-cyan-300", kGreedy: 0, kThermo: 0, pairId: null },
  { idx: 18, name: "Leu", type: "Leu", color: "text-amber-300", kGreedy: 0, kThermo: 1, pairId: 4 },
  { idx: 17, name: "Phe", type: "Phe", color: "text-purple-300", kGreedy: 0, kThermo: 0, pairId: null },
  { idx: 16, name: "Phe", type: "Phe", color: "text-purple-300", kGreedy: 0, kThermo: 5, pairId: 5 },
  { idx: 15, name: "Asp", type: "Asp", color: "text-rose-300", kGreedy: 0, kThermo: 0, pairId: null },
  { idx: 14, name: "Leu", type: "Leu", color: "text-amber-300", kGreedy: 0, kThermo: 2, pairId: 6 },
];

// ============================================================================
// MAIN PAGE COMPONENT
// ============================================================================

export default function ThermoLabPage() {
  const [globalBeta, setGlobalBeta] = useState<number>(1.2);
  const [isAutoSimulating, setIsAutoSimulating] = useState<boolean>(true);
  const [soundEnabled, setSoundEnabled] = useState<boolean>(false);
  const [copiedTweet, setCopiedTweet] = useState<boolean>(false);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const prevBetaRef = useRef<number>(1.2);

  const [openAccordion, setOpenAccordion] = useState<Record<number, boolean>>({
    1: false,
    2: true,
    3: false,
    4: false,
    5: false,
    6: false,
    7: false,
    8: true,
    9: true,
    10: true,
    11: true,
  });

  // ============================================================================
  // RETRO ARCADE WEB AUDIO SYNTHESIZER
  // ============================================================================

  const playArcadeTone = useCallback(
    (freq: number = 520, durationMs: number = 35, type: OscillatorType = "triangle") => {
      if (!soundEnabled || typeof window === "undefined") return;
      try {
        const AudioCtx =
          window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
        if (!AudioCtx) return;
        if (!audioCtxRef.current) {
          audioCtxRef.current = new AudioCtx();
        }
        const ctx = audioCtxRef.current;
        if (ctx.state === "suspended") {
          ctx.resume();
        }
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = type;
        osc.frequency.setValueAtTime(freq, ctx.currentTime);
        gain.gain.setValueAtTime(0.045, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + durationMs / 1000);
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.start();
        osc.stop(ctx.currentTime + durationMs / 1000);
      } catch {
        // Ignore audio errors if blocked by browser autoplay policy
      }
    },
    [soundEnabled]
  );

  const handleBetaChange = (nextBeta: number) => {
    const prev = prevBetaRef.current;
    if ((prev < 0.4407 && nextBeta >= 0.4407) || (prev >= 0.4407 && nextBeta < 0.4407)) {
      playArcadeTone(880, 95, "sine");
    } else {
      playArcadeTone(280 + nextBeta * 90, 22, "triangle");
    }
    prevBetaRef.current = nextBeta;
    setGlobalBeta(nextBeta);
  };

  // --- Station 1: 1D 16-Spin Chain ---
  const [chain1D, setChain1D] = useState<number[]>(() =>
    Array.from({ length: 16 }, (_, i) => (i % 2 === 0 ? 1 : -1))
  );

  // --- Station 2: 16x16 2D Ising Grid ---
  const [grid2D, setGrid2D] = useState<number[]>(() =>
    Array.from({ length: 256 }, (_, i) => ((i * 7 + 3) % 2 === 0 ? 1 : -1))
  );

  // --- Station 3: 8-Node Rivalry Graph (Max-Cut) ---
  const rivalryEdges = useMemo(
    () => [
      [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 0],
      [0, 4], [1, 5], [2, 6], [3, 7], [0, 2], [4, 6],
    ],
    []
  );
  const [monsterTeams, setMonsterTeams] = useState<number[]>([
    1, 1, -1, 1, -1, -1, 1, -1,
  ]);

  // --- Station 4: 10x10 4-Color Potts Grid ---
  const [pottsGrid, setPottsGrid] = useState<number[]>(() =>
    Array.from({ length: 100 }, (_, i) => i % 4)
  );

  // --- Station 5: 8x8 Magic Eraser (Denoising) ---
  const cleanDigit8 = useMemo(
    () => [
      -1, -1, -1, -1, -1, -1, -1, -1,
      -1,  1,  1,  1,  1,  1,  1, -1,
      -1,  1, -1, -1, -1, -1,  1, -1,
      -1,  1,  1,  1,  1,  1,  1, -1,
      -1,  1,  1,  1,  1,  1,  1, -1,
      -1,  1, -1, -1, -1, -1,  1, -1,
      -1,  1,  1,  1,  1,  1,  1, -1,
      -1, -1, -1, -1, -1, -1, -1, -1,
    ],
    []
  );
  const [noisyDigit, setNoisyDigit] = useState<number[]>(() => {
    const arr = [...cleanDigit8];
    [0, 5, 10, 18, 21, 29, 35, 42, 45, 53, 58, 62].forEach((idx) => {
      arr[idx] *= -1;
    });
    return arr;
  });
  const [denoisedDigit, setDenoisedDigit] = useState<number[]>(() => [...noisyDigit]);

  // --- Station 6: 8x8 RBM Memory Vault ---
  const rbmPrototypes = useMemo(() => {
    const plus = Array(64).fill(-1);
    const cross = Array(64).fill(-1);
    const square = Array(64).fill(-1);
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        if (r === 3 || r === 4 || c === 3 || c === 4) plus[r * 8 + c] = 1;
        if (r === c || r === 7 - c) cross[r * 8 + c] = 1;
        if (r >= 1 && r <= 6 && c >= 1 && c <= 6 && (r === 1 || r === 6 || c === 1 || c === 6)) {
          square[r * 8 + c] = 1;
        }
      }
    }
    return [plus, cross, square];
  }, []);
  const [selectedProtoIdx, setSelectedProtoIdx] = useState<number>(0);
  const [rbmState, setRbmState] = useState<number[]>(() => {
    const p = [...rbmPrototypes[0]];
    for (let i = 32; i < 64; i++) p[i] = i % 2 === 0 ? 1 : -1;
    return p;
  });

  // --- Station 7: 20-Asset QUBO Treasure Vault ---
  const assetReturns = useMemo(
    () => [
      16.6, 11.6, 29.9, 15.5, 6.4, 31.6, 16.4, 16.8, 6.9, 23.8,
      10.6, 18.9, 14.7, 17.3, 23.7, 8.4, 13.3, 6.0, 29.4, 14.5,
    ],
    []
  );
  const assetRisks = useMemo(
    () => [
      0.19, 0.22, 0.16, 0.24, 0.28, 0.15, 0.23, 0.21, 0.27, 0.18,
      0.25, 0.20, 0.24, 0.21, 0.17, 0.26, 0.23, 0.29, 0.16, 0.22,
    ],
    []
  );
  const [quboLambda, setQuboLambda] = useState<number>(0.65);
  const [selectedAssets, setSelectedAssets] = useState<number[]>(() => {
    const init = Array(20).fill(0);
    [0, 2, 5, 9, 18].forEach((i) => (init[i] = 1));
    return init;
  });

  // --- Station 8: 12 Continuous Double-Well Oscillators ---
  const [oscillators, setOscillators] = useState<number[]>(() =>
    Array.from({ length: 12 }, (_, i) => (i % 2 === 0 ? 0.85 : -0.85))
  );

  // --- Station 9: 10-City TSP Route Untangler ---
  const tspCoords = useMemo(
    () =>
      Array.from({ length: 10 }, (_, idx) => {
        const angle = (idx / 10) * 2 * Math.PI - Math.PI / 2;
        return {
          x: 100 + 74 * Math.cos(angle),
          y: 100 + 74 * Math.sin(angle),
        };
      }),
    []
  );
  const [tspTour, setTspTour] = useState<number[]>([0, 5, 2, 8, 4, 9, 1, 6, 3, 7]);

  // --- Station 10: 24-Spin Sherrington-Kirkpatrick Spin Glass ---
  const [skSeed, setSkSeed] = useState<number>(42);
  const [skTemp, setSkTemp] = useState<number>(0.4); // T = 0.40 (beta = 2.5)
  const skMatrix = useMemo(() => generateSKMatrix(skSeed, 24), [skSeed]);
  const [skReplicaA, setSkReplicaA] = useState<number[]>(() =>
    Array.from({ length: 24 }, (_, i) => (i % 2 === 0 ? 1 : -1))
  );
  const [skReplicaB, setSkReplicaB] = useState<number[]>(() =>
    Array.from({ length: 24 }, (_, i) => (i % 3 === 0 ? 1 : -1))
  );
  const [skOverlapHistory, setSkOverlapHistory] = useState<number[]>(() => [
    0.54, 0.58, -0.52, 0.48, -0.62, 0.51, 0.49, -0.53, 0.55, -0.47, 0.52, -0.58,
  ]);
  const [skEnergyA, setSkEnergyA] = useState<number>(-15.4266);

  // --- Station 11: Thermodynamic RNA Secondary-Structure Ensemble ---
  const [rnaTier, setRnaTier] = useState<24 | 50>(24);
  const [rnaTempMode, setRnaTempMode] = useState<"native" | "melted">("native");
  const rnaSeq = useMemo(
    () =>
      rnaTier === 24
        ? "GGCUAAAAAAACCAAAAAACCGAA"
        : "CUCGUCGUAGCAGUCGCAAGUCGGCAGUCGAAUUAAACGCUACGACGAGG",
    [rnaTier]
  );
  const rnaPairs = useMemo(
    () => getRnaCandidatePairs(rnaSeq, rnaTier === 50 ? 72 : undefined),
    [rnaSeq, rnaTier]
  );
  const rnaHamiltonian = useMemo(() => buildRnaQ(rnaPairs), [rnaPairs]);
  const [rnaActiveVector, setRnaActiveVector] = useState<number[]>(() => {
    const vec = new Array(rnaPairs.length).fill(0);
    if (rnaTier === 24) {
      rnaPairs.forEach((p, idx) => {
        if ((p.i === 0 && p.j === 12) || (p.i === 1 && p.j === 11) || (p.i === 3 && p.j === 7)) {
          vec[idx] = 1;
        }
      });
    } else {
      rnaPairs.forEach((p, idx) => {
        if (p.i + p.j === 48 || p.i + p.j === 47) vec[idx] = 1;
      });
    }
    return vec;
  });
  const [rnaPairProbs, setRnaPairProbs] = useState<number[]>(() =>
    new Array(rnaPairs.length).fill(0.35)
  );

  // --- Station 12: Grid SCUC Power Dispatch QUBO ---
  const scucGenerators = useMemo(
    () => [
      { id: "Nuclear_Base", name: "Nuclear Base", capacity: 400, cost: 12, color: "text-emerald-400 border-emerald-500/30 bg-emerald-500/10" },
      { id: "Coal_Base", name: "Coal Base", capacity: 250, cost: 28, color: "text-amber-400 border-amber-500/30 bg-amber-500/10" },
      { id: "CCGT_Flex", name: "CCGT Flex Gas", capacity: 150, cost: 45, color: "text-cyan-400 border-cyan-500/30 bg-cyan-500/10" },
      { id: "Peaker_Gas", name: "Peaker Turbine", capacity: 100, cost: 85, color: "text-rose-400 border-rose-500/30 bg-rose-500/10" },
    ],
    []
  );
  const scucDemands = useMemo(() => [450, 520, 680, 750, 620, 480], []);
  const [scucSchedule, setScucSchedule] = useState<number[][]>([
    [1, 1, 1, 1, 1, 1], // Nuclear Base (400 MW continuous baseload)
    [0, 0, 1, 1, 1, 0], // Coal Base (250 MW peak hours)
    [0, 1, 0, 0, 0, 0], // CCGT Flex Gas (150 MW hour 2)
    [1, 0, 1, 1, 0, 1], // Peaker Turbine (100 MW reserve)
  ]);
  const [scucPenaltyWeight, setScucPenaltyWeight] = useState<number>(0.005);
  const [selectedBackend, setSelectedBackend] = useState<"jax" | "extropic">("jax");

  // --- Station 13: Codon Optimization via Potts EBM ---
  const [codonLambdaGc, setCodonLambdaGc] = useState<number>(40.0);
  const [codonLambdaHp, setCodonLambdaHp] = useState<number>(2.0);
  const [activeCodonSequence, setActiveCodonSequence] = useState<"thermo" | "greedy" | "comparison">("comparison");

  // --- Station 14: Side-Chain Rotamer Packing via Potts EBM ---
  const [rotamerViewMode, setRotamerViewMode] = useState<"thermo" | "greedy" | "radar">("thermo");
  const [rotamerClashCutoff, setRotamerClashCutoff] = useState<number>(2.0);
  const [selectedFacingPair, setSelectedFacingPair] = useState<number | null>(null);

  // --- Station 15: Non-Equilibrium Hybrid TPU Langevin & Frustration Breaking ---
  const [tpuActiveTab, setTpuActiveTab] = useState<"frustration" | "resampling" | "integrator" | "silicon">("frustration");
  const [tpuStep, setTpuStep] = useState<number>(0);
  const [tpuIsPlaying, setTpuIsPlaying] = useState<boolean>(false);
  const [tpuStiffnessAlpha, setTpuStiffnessAlpha] = useState<number>(6.0);
  const [tpuResamplingMode, setTpuResamplingMode] = useState<"with_smc" | "no_smc">("with_smc");

  useEffect(() => {
    if (!tpuIsPlaying) return;
    const interval = setInterval(() => {
      setTpuStep((prev) => {
        if (prev >= 100) {
          setTpuIsPlaying(false);
          return 100;
        }
        return prev + 1;
      });
    }, 45);
    return () => clearInterval(interval);
  }, [tpuIsPlaying]);


  // ============================================================================
  // REAL-TIME GLAUBER / BLOCK GIBBS TICKER
  // ============================================================================

  const stepPhysics = useCallback(() => {
    const b = globalBeta;

    // 1. Update 1D Chain
    setChain1D((prev) => {
      const next = [...prev];
      for (const parity of [0, 1]) {
        for (let i = parity; i < 16; i += 2) {
          const left = next[(i + 15) % 16];
          const right = next[(i + 1) % 16];
          const field = left + right;
          const probPlus = 1 / (1 + Math.exp(-2 * b * field));
          next[i] = Math.random() < probPlus ? 1 : -1;
        }
      }
      return next;
    });

    // 2. Update 2D 16x16 Checkerboard Lattice
    setGrid2D((prev) => {
      const next = [...prev];
      for (const parity of [0, 1]) {
        for (let r = 0; r < 16; r++) {
          for (let c = 0; c < 16; c++) {
            if ((r + c) % 2 !== parity) continue;
            const u = next[((r + 15) % 16) * 16 + c];
            const d = next[((r + 1) % 16) * 16 + c];
            const l = next[r * 16 + ((c + 15) % 16)];
            const rt = next[r * 16 + ((c + 1) % 16)];
            const field = u + d + l + rt;
            const probPlus = 1 / (1 + Math.exp(-2 * b * field));
            next[r * 16 + c] = Math.random() < probPlus ? 1 : -1;
          }
        }
      }
      return next;
    });

    // 3. Update Monster Rivalry Max-Cut
    setMonsterTeams((prev) => {
      const next = [...prev];
      for (let i = 0; i < 8; i++) {
        let rivalSum = 0;
        for (const [u, v] of rivalryEdges) {
          if (u === i) rivalSum += next[v];
          else if (v === i) rivalSum += next[u];
        }
        const gamma = -1.0 * rivalSum;
        const probPlus = 1 / (1 + Math.exp(-2 * b * gamma));
        next[i] = Math.random() < probPlus ? 1 : -1;
      }
      return next;
    });

    // 4. Update 4-Color Potts Grid
    setPottsGrid((prev) => {
      const next = [...prev];
      for (const parity of [0, 1]) {
        for (let r = 0; r < 10; r++) {
          for (let c = 0; c < 10; c++) {
            if ((r + c) % 2 !== parity) continue;
            const neighbors = [
              next[((r + 9) % 10) * 10 + c],
              next[((r + 1) % 10) * 10 + c],
              next[r * 10 + ((c + 9) % 10)],
              next[r * 10 + ((c + 1) % 10)],
            ];
            const counts = [0, 0, 0, 0];
            neighbors.forEach((col) => counts[col]++);
            const weights = counts.map((cnt) => Math.exp(b * 1.3 * cnt));
            const sumW = weights.reduce((a, v) => a + v, 0);
            let rand = Math.random() * sumW;
            let chosen = 0;
            for (let k = 0; k < 4; k++) {
              rand -= weights[k];
              if (rand <= 0) {
                chosen = k;
                break;
              }
            }
            next[r * 10 + c] = chosen;
          }
        }
      }
      return next;
    });

    // 7. Update QUBO Portfolio Selection
    setSelectedAssets((prev) => {
      const next = [...prev];
      const idx = Math.floor(Math.random() * 20);
      const currentSum = next.reduce((a, v) => a + v, 0);
      const sumOthers = currentSum - next[idx];

      const ret1 = assetReturns[idx] / 100.0;
      const risk1 = assetRisks[idx];
      const pen1 = quboLambda * Math.pow(sumOthers + 1 - 5, 2);
      const pen0 = quboLambda * Math.pow(sumOthers - 5, 2);

      const e1 = risk1 * risk1 - ret1 + pen1;
      const e0 = pen0;
      const deltaE = e1 - e0;
      const probSelect = 1 / (1 + Math.exp(b * 3.0 * deltaE));
      next[idx] = Math.random() < probSelect ? 1 : 0;
      return next;
    });

    // 8. Update Continuous Double-Well Oscillators
    setOscillators((prev) =>
      prev.map((x, i) => {
        const left = prev[(i + 11) % 12];
        const right = prev[(i + 1) % 12];
        const force = -4.0 * 1.2 * x * (x * x - 1.0) + 0.45 * (left + right);
        const noiseScale = 1.0 / Math.sqrt(Math.max(b * 3.5, 0.35));
        const gauss = (Math.random() + Math.random() + Math.random() - 1.5) * 1.1;
        const nextX = x + 0.06 * force + 0.22 * noiseScale * gauss;
        return Math.max(-1.65, Math.min(1.65, nextX));
      })
    );

    // 9. Update 10-City TSP via Thermal 2-Opt Step
    setTspTour((prev) => {
      const next = [...prev];
      const i = Math.floor(Math.random() * 8) + 1;
      const j = Math.floor(Math.random() * (9 - i)) + i + 1;
      const calcLen = (arr: number[]) => {
        let sum = 0;
        for (let k = 0; k < 10; k++) {
          const p1 = tspCoords[arr[k]];
          const p2 = tspCoords[arr[(k + 1) % 10]];
          sum += Math.hypot(p1.x - p2.x, p1.y - p2.y);
        }
        return sum;
      };
      const candidate = [
        ...next.slice(0, i),
        ...next.slice(i, j + 1).reverse(),
        ...next.slice(j + 1),
      ];
      const dE = (calcLen(candidate) - calcLen(next)) / 40.0;
      if (dE < 0 || Math.random() < Math.exp(-b * 2.5 * dE)) {
        return candidate;
      }
      return next;
    });

    // 10. Update Station 10: SK Spin Glass Replicas
    const betaSK = 1.0 / Math.max(skTemp, 0.1);
    let nextA = [...skReplicaA];
    let nextB = [...skReplicaB];
    for (let i = 0; i < 24; i++) {
      let fieldA = 0;
      let fieldB = 0;
      for (let j = 0; j < 24; j++) {
        fieldA += skMatrix[i][j] * nextA[j];
        fieldB += skMatrix[i][j] * nextB[j];
      }
      const probA = 1 / (1 + Math.exp(-2 * betaSK * fieldA));
      const probB = 1 / (1 + Math.exp(-2 * betaSK * fieldB));
      nextA[i] = Math.random() < probA ? 1 : -1;
      nextB[i] = Math.random() < probB ? 1 : -1;
    }
    setSkReplicaA(nextA);
    setSkReplicaB(nextB);

    let curQ = 0;
    for (let i = 0; i < 24; i++) curQ += nextA[i] * nextB[i];
    curQ /= 24.0;
    setSkOverlapHistory((prev) => {
      const updated = [...prev, curQ];
      return updated.length > 120 ? updated.slice(updated.length - 120) : updated;
    });

    let liveE = 0;
    for (let i = 0; i < 24; i++) {
      for (let j = 0; j < 24; j++) {
        liveE += -0.5 * skMatrix[i][j] * nextA[i] * nextA[j];
      }
    }
    setSkEnergyA(liveE);

    // 11. Update Station 11: RNA Folding Ensemble
    const betaRNA = rnaTempMode === "native" ? 1.0 : 0.2;
    setRnaActiveVector((prevVec) => {
      const nextVec = [...prevVec];
      const m = rnaPairs.length;
      const { c, q } = rnaHamiltonian;
      const numFlips = Math.min(8, m);
      for (let s = 0; s < numFlips; s++) {
        const k = Math.floor(Math.random() * m);
        let deltaE = c[k];
        for (let l = 0; l < m; l++) {
          if (nextVec[l] === 1) deltaE += q[k][l];
        }
        const p1 = 1 / (1 + Math.exp(betaRNA * deltaE));
        nextVec[k] = Math.random() < p1 ? 1 : 0;
      }
      return nextVec;
    });

    setRnaPairProbs((prevProbs) => {
      const nextProbs = [...prevProbs];
      for (let k = 0; k < rnaPairs.length; k++) {
        const act = rnaActiveVector[k] || 0;
        nextProbs[k] = (prevProbs[k] || 0) * 0.93 + act * 0.07;
      }
      return nextProbs;
    });
  }, [
    globalBeta,
    rivalryEdges,
    assetReturns,
    assetRisks,
    quboLambda,
    tspCoords,
    skTemp,
    skMatrix,
    skReplicaA,
    skReplicaB,
    rnaTempMode,
    rnaPairs,
    rnaHamiltonian,
    rnaActiveVector,
  ]);

  useEffect(() => {
    if (!isAutoSimulating) return;
    const timer = setInterval(stepPhysics, 160);
    return () => clearInterval(timer);
  }, [isAutoSimulating, stepPhysics]);

  // --- Station 5 Action: Clean Noisy Icon ---
  const handleDenoiseStep = () => {
    playArcadeTone(660, 50, "sine");
    setDenoisedDigit((prev) => {
      const next = [...prev];
      for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
          const idx = r * 8 + c;
          const u = prev[((r + 7) % 8) * 8 + c];
          const d = prev[((r + 1) % 8) * 8 + c];
          const l = prev[r * 8 + ((c + 7) % 8)];
          const rt = prev[r * 8 + ((c + 1) % 8)];
          const field = 1.0 * (u + d + l + rt) + 1.8 * noisyDigit[idx];
          next[idx] = field >= 0 ? 1 : -1;
        }
      }
      return next;
    });
  };

  const handleAddNoise = () => {
    playArcadeTone(220, 60, "sawtooth");
    const corrupted = cleanDigit8.map((val) =>
      Math.random() < 0.22 ? -val : val
    );
    setNoisyDigit(corrupted);
    setDenoisedDigit([...corrupted]);
  };

  // --- Station 6 Action: Occlude & Recall Prototype ---
  const handleSelectPrototype = (idx: number) => {
    playArcadeTone(440, 35, "triangle");
    setSelectedProtoIdx(idx);
    const proto = [...rbmPrototypes[idx]];
    for (let i = 32; i < 64; i++) {
      proto[i] = Math.random() < 0.5 ? 1 : -1;
    }
    setRbmState(proto);
  };

  const handleRecallMemory = () => {
    playArcadeTone(784, 70, "sine");
    const target = rbmPrototypes[selectedProtoIdx];
    setRbmState([...target]);
  };

  // Computed metrics for UI display
  const magnet1D = useMemo(
    () => Math.abs(chain1D.reduce((a, b) => a + b, 0) / 16).toFixed(2),
    [chain1D]
  );
  const magnet2D = useMemo(
    () => Math.abs(grid2D.reduce((a, b) => a + b, 0) / 256).toFixed(2),
    [grid2D]
  );
  const happyRivalries = useMemo(() => {
    let cuts = 0;
    rivalryEdges.forEach(([u, v]) => {
      if (monsterTeams[u] !== monsterTeams[v]) cuts++;
    });
    return cuts;
  }, [monsterTeams, rivalryEdges]);

  const denoisedError = useMemo(() => {
    let diffs = 0;
    for (let i = 0; i < 64; i++) {
      if (denoisedDigit[i] !== cleanDigit8[i]) diffs++;
    }
    return ((diffs / 64) * 100).toFixed(1);
  }, [denoisedDigit, cleanDigit8]);

  const rbmOverlap = useMemo(() => {
    const target = rbmPrototypes[selectedProtoIdx];
    let dot = 0;
    for (let i = 0; i < 64; i++) dot += rbmState[i] * target[i];
    return Math.round(((dot / 64 + 1) / 2) * 100);
  }, [rbmState, rbmPrototypes, selectedProtoIdx]);

  const quboCount = useMemo(
    () => selectedAssets.reduce((a, v) => a + v, 0),
    [selectedAssets]
  );
  const quboTotalReturn = useMemo(() => {
    const selectedIndices = selectedAssets
      .map((sel, idx) => (sel ? idx : -1))
      .filter((idx) => idx >= 0);
    if (selectedIndices.length === 0) return "0.0";
    const avg =
      selectedIndices.reduce((acc, idx) => acc + assetReturns[idx], 0) /
      selectedIndices.length;
    return avg.toFixed(1);
  }, [selectedAssets, assetReturns]);

  const tspCurrentLength = useMemo(() => {
    let sum = 0;
    for (let k = 0; k < 10; k++) {
      const p1 = tspCoords[tspTour[k]];
      const p2 = tspCoords[tspTour[(k + 1) % 10]];
      sum += Math.hypot(p1.x - p2.x, p1.y - p2.y) / 74.0;
    }
    return sum.toFixed(2);
  }, [tspTour, tspCoords]);

  // --- Station 10 Computed Metrics ---
  const skHistogram = useMemo(() => {
    const bins = Array(11).fill(0);
    const n = skOverlapHistory.length;
    if (n === 0) return bins;
    for (const q of skOverlapHistory) {
      const bIdx = Math.max(0, Math.min(10, Math.floor(((q + 1.0) / 2.0) * 11)));
      bins[bIdx]++;
    }
    return bins.map((cnt) => (cnt / n) * 100);
  }, [skOverlapHistory]);

  const skMeanAbsQ = useMemo(() => {
    if (skOverlapHistory.length === 0) return "0.0000";
    const sum = skOverlapHistory.reduce((a, b) => a + Math.abs(b), 0);
    return (sum / skOverlapHistory.length).toFixed(4);
  }, [skOverlapHistory]);

  const skVarQ = useMemo(() => {
    if (skOverlapHistory.length === 0) return "0.0000";
    const mean = skOverlapHistory.reduce((a, b) => a + b, 0) / skOverlapHistory.length;
    const sqDiffs = skOverlapHistory.reduce((a, b) => a + Math.pow(b - mean, 2), 0);
    return (sqDiffs / skOverlapHistory.length).toFixed(4);
  }, [skOverlapHistory]);

  // --- Station 11 Computed Metrics ---
  const rnaActivePairs = useMemo(() => {
    const active: { i: number; j: number }[] = [];
    rnaActiveVector.forEach((val, idx) => {
      if (val === 1 && rnaPairs[idx]) {
        active.push({ i: rnaPairs[idx].i, j: rnaPairs[idx].j });
      }
    });
    return active;
  }, [rnaActiveVector, rnaPairs]);

  const rnaDotBracket = useMemo(
    () => formatDotBracket(rnaSeq.length, rnaActivePairs),
    [rnaSeq.length, rnaActivePairs]
  );

  const rnaEnergy = useMemo(() => {
    let e = 0;
    const { c, q } = rnaHamiltonian;
    for (let i = 0; i < rnaPairs.length; i++) {
      if (rnaActiveVector[i] === 1) {
        e += c[i];
        for (let j = i + 1; j < rnaPairs.length; j++) {
          if (rnaActiveVector[j] === 1) e += q[i][j];
        }
      }
    }
    return e;
  }, [rnaActiveVector, rnaPairs, rnaHamiltonian]);

  const rnaEntropy = useMemo(() => {
    if (rnaTier === 24) {
      return rnaTempMode === "native" ? 5.13 : 10.66;
    } else {
      return rnaTempMode === "native" ? 3.06 : 14.56;
    }
  }, [rnaTier, rnaTempMode]);

  // --- Station 12 Computed Metrics ---
  const scucDelivered = useMemo(() => {
    return scucDemands.map((_, t) => {
      let mw = 0;
      scucGenerators.forEach((gen, g) => {
        if (scucSchedule[g][t] === 1) mw += gen.capacity;
      });
      return mw;
    });
  }, [scucSchedule, scucGenerators, scucDemands]);

  const scucTotalGenCost = useMemo(() => {
    let cost = 0;
    scucGenerators.forEach((gen, g) => {
      scucDemands.forEach((_, t) => {
        if (scucSchedule[g][t] === 1) {
          cost += gen.cost * gen.capacity * 0.01;
        }
      });
    });
    return cost;
  }, [scucSchedule, scucGenerators, scucDemands]);

  const scucTotalPenalty = useMemo(() => {
    let penalty = 0;
    scucDemands.forEach((demand, t) => {
      const delivered = scucDelivered[t];
      penalty += scucPenaltyWeight * Math.pow(delivered - demand, 2);
    });
    return penalty;
  }, [scucDemands, scucDelivered, scucPenaltyWeight]);

  const scucTotalEnergy = useMemo(() => {
    return scucTotalGenCost + scucTotalPenalty;
  }, [scucTotalGenCost, scucTotalPenalty]);

  // --- Station 13 Computed Metrics ---
  const codonGreedyMetrics = useMemo(() => {
    const cai = 1.0;
    const gc = 30 / 45; // 0.6667
    const hairpins = 4;
    const eCai = 0.0;
    const eGc = codonLambdaGc * Math.pow(gc - 0.50, 2);
    const eHp = codonLambdaHp * hairpins;
    const totalEnergy = eCai + eGc + eHp;
    return { cai, gc, hairpins, eCai, eGc, eHp, totalEnergy };
  }, [codonLambdaGc, codonLambdaHp]);

  const codonThermoMetrics = useMemo(() => {
    const cai = 0.9326;
    const gc = 26 / 45; // 0.5778
    const hairpins = 0;
    const eCai = 1.0487;
    const eGc = codonLambdaGc * Math.pow(gc - 0.50, 2);
    const eHp = codonLambdaHp * hairpins;
    const totalEnergy = eCai + eGc + eHp;
    return { cai, gc, hairpins, eCai, eGc, eHp, totalEnergy };
  }, [codonLambdaGc, codonLambdaHp]);

  const codonEnergyAdvantage = useMemo(() => {
    return codonGreedyMetrics.totalEnergy - codonThermoMetrics.totalEnergy;
  }, [codonGreedyMetrics, codonThermoMetrics]);

  // --- Backend Telemetry Summary ---
  const backendTelemetry = useMemo(() => {
    if (selectedBackend === "jax") {
      return {
        name: "JAX Block Gibbs (Simulated)",
        chipType: "Classical JAX PRNG Block Updates",
        runtime: "0.1187s",
        sampleRate: "42,375 Hz",
        tauInt: "54.61",
        ess: "73.2",
        essPerSec: "384.5 ESS/s (classical simulation baseline)",
        advantage: "1.0x (Classical MCMC Baseline)",
        description: "Vectorized CPU/GPU Gibbs scan subject to critical slowing down on frustrated spin landscapes.",
      };
    }
    return {
      name: "Extropic TSU (Modeled Projection)",
      chipType: "Modeled Extropic Z1 projection (1.0W spec), not measured",
      runtime: "N/A (not measured)",
      sampleRate: "N/A (hardware not publicly available)",
      tauInt: "N/A",
      ess: "N/A",
      essPerSec: "520,833 ESS/s (modeled projection, not measured)",
      advantage: "Projection only: no measured TSU run",
      description: "Continuous Langevin noise acceleration overcoming thermodynamic barrier trapping at microsecond latency.",
    };
  }, [selectedBackend]);

  // --- Copy / Post to X Quick-Share Text ---
  const shareText = useMemo(
    () =>
      `Evaluated the 16-station thermodynamic computing suite at beta=${globalBeta.toFixed(
        2
      )} (|M|=${magnet2D}, SK RSB <|q|>=${skMeanAbsQ}, RNA S=${rnaEntropy}b, Side-Chain deltaE=-295.5kcal, 20.18M spin-updates/s in JAX) via thrml.\n\nLive lab: https://getnexusaisolutions.com/thermo-lab\nRepo: https://github.com/nexusai-kevin/thermo-physics`,
    [globalBeta, magnet2D, skMeanAbsQ, rnaEntropy]
  );

  const handleCopyShare = () => {
    playArcadeTone(880, 60, "sine");
    if (typeof navigator !== "undefined" && navigator.clipboard) {
      navigator.clipboard.writeText(shareText);
      setCopiedTweet(true);
      setTimeout(() => setCopiedTweet(false), 2500);
    }
  };

  const toggleAccordion = (id: number) => {
    playArcadeTone(380, 25, "triangle");
    setOpenAccordion((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const pottsColors = [
    "bg-cyan-400 shadow-cyan-400/50",
    "bg-fuchsia-500 shadow-fuchsia-500/50",
    "bg-amber-400 shadow-amber-400/50",
    "bg-emerald-400 shadow-emerald-400/50",
  ];

  return (
    <div className="min-h-screen bg-[#020308] text-[#faf9f5] font-[family-name:var(--font-sans)] selection:bg-[#5eead4]/20 selection:text-[#5eead4] thermo-grain relative">
      {/* ===================================================================
          FABLE 5.2 LAB INSTRUMENT STYLES (Cryogenic / Titanium Aesthetic)
      =================================================================== */}
      <style dangerouslySetInnerHTML={{ __html: `
        .thermo-grain::after {
          content: "";
          position: fixed;
          inset: 0;
          pointer-events: none;
          z-index: 99;
          opacity: 0.038;
          background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='180' height='180'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='180' height='180' filter='url(%23n)'/%3E%3C/svg%3E");
        }
        .lab-panel {
          position: relative;
          background: linear-gradient(180deg, rgba(15, 23, 42, 0.75) 0%, rgba(8, 12, 22, 0.92) 100%);
          border: 1px solid rgba(255, 255, 255, 0.08);
          border-radius: 20px;
          box-shadow: 
            0 1px 0 0 rgba(255, 255, 255, 0.12) inset,
            0 0 0 1px rgba(0, 0, 0, 0.8),
            0 20px 40px -15px rgba(0, 0, 0, 0.75),
            0 0 35px -12px rgba(6, 182, 212, 0.18);
          backdrop-filter: blur(16px);
          transition: transform 0.25s cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 0.25s ease, border-color 0.25s ease;
        }
        .lab-panel:hover {
          border-color: rgba(6, 182, 212, 0.35);
          box-shadow: 
            0 1px 0 0 rgba(255, 255, 255, 0.2) inset,
            0 0 0 1px rgba(6, 182, 212, 0.3),
            0 28px 56px -16px rgba(0, 0, 0, 0.9),
            0 0 45px -8px rgba(6, 182, 212, 0.25);
        }
        .lab-rivet {
          width: 7px;
          height: 7px;
          border-radius: 50%;
          background: radial-gradient(circle at 35% 30%, #e2e8f0 0%, #64748b 45%, #0f172a 100%);
          box-shadow: 0 0 0 1px #000, 0 1px 1px rgba(255,255,255,0.2) inset;
        }
        .thermal-slider {
          -webkit-appearance: none;
          appearance: none;
          height: 8px;
          border-radius: 999px;
          background: linear-gradient(to right, #f43f5e 0%, #fb923c 25%, #facc15 50%, #38bdf8 75%, #06b6d4 100%);
          box-shadow: inset 0 2px 4px rgba(0,0,0,0.8), 0 0 12px rgba(6, 182, 212, 0.3);
          outline: none;
          cursor: pointer;
        }
        .thermal-slider::-webkit-slider-thumb {
          -webkit-appearance: none;
          appearance: none;
          width: 22px;
          height: 22px;
          border-radius: 50%;
          background: radial-gradient(circle at 35% 35%, #ffffff 0%, #67e8f9 45%, #0891b2 100%);
          cursor: grab;
          border: 2px solid #ffffff;
          box-shadow: 0 0 0 1px rgba(0,0,0,0.7), 0 0 16px rgba(6, 182, 212, 0.9);
          transition: transform 0.15s cubic-bezier(0.34, 1.56, 0.64, 1);
        }
        .thermal-slider::-webkit-slider-thumb:hover {
          transform: scale(1.18);
        }
        .thermal-slider::-webkit-slider-thumb:active {
          cursor: grabbing;
          transform: scale(1.05);
        }
        .tactile-btn {
          transition: transform 0.15s cubic-bezier(0.34, 1.56, 0.64, 1), filter 0.15s ease, box-shadow 0.15s ease;
          touch-action: manipulation;
        }
        .tactile-btn:hover {
          transform: translateY(-2px);
          filter: brightness(1.12);
        }
        .tactile-btn:active {
          transform: translateY(1px) scale(0.98);
        }
      `}} />

      {/* ===================================================================
          STICKY MASTER CRYOGENIC RHEOSTAT BAR (Hardware Instrumentation)
      =================================================================== */}
      <div className="sticky top-0 z-50 backdrop-blur-md bg-[#020308]/90 border-b border-[rgba(250,249,245,0.08)]">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-2.5 flex flex-wrap items-center justify-between gap-3">
          
          {/* Left: Brand & Telemetry */}
          <div className="flex items-center gap-3">
            <a
              href="https://github.com/nexusai-kevin/thermo-physics"
              target="_blank"
              rel="noopener noreferrer"
              className="rounded-full border border-[rgba(250,249,245,0.12)] bg-[#05070f] px-3 py-1 font-mono text-[10px] uppercase tracking-[0.15em] text-[rgba(250,249,245,0.7)] hover:border-[rgba(250,249,245,0.3)] hover:text-[#faf9f5] transition-colors"
              title="Nexus Thermodynamic Research on GitHub"
            >
              GitHub ↗
            </a>

            <div className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-[#5eead4] animate-pulse" />
              <span className="font-mono text-[11px] uppercase tracking-[0.2em] text-[#faf9f5]">
                Thermodynamic Master Control
              </span>
            </div>
            
            <span className="hidden xl:inline-block font-mono text-[10px] uppercase tracking-[0.15em] text-[rgba(250,249,245,0.4)]">
              // 16 Stations · 20.18M Spins/s
            </span>
          </div>

          {/* Center: Precision Thermal Spectrum Slider */}
          <div className="flex items-center gap-3 flex-1 max-w-md">
            <span className="text-[10px] font-mono uppercase tracking-wider text-[rgba(250,249,245,0.4)] whitespace-nowrap">
              High Temp (0.1)
            </span>
            <input
              type="range"
              min={0.1}
              max={5.0}
              step={0.05}
              value={globalBeta}
              onChange={(e) => handleBetaChange(parseFloat(e.target.value))}
              className="thermal-slider w-full cursor-pointer"
            />
            <span className="text-[10px] font-mono uppercase tracking-wider text-[#5eead4] whitespace-nowrap">
              Low Temp (5.0)
            </span>
          </div>

          {/* Right: Telemetry Readout & Controls */}
          <div className="flex flex-wrap items-center gap-2 font-mono text-xs">
            <div className="px-3 py-1 rounded-full border border-[#5eead4]/30 bg-[#041a1a]/60 text-xs font-mono">
              <span className="text-[rgba(250,249,245,0.5)]">β =</span>{" "}
              <span className="text-[#5eead4] font-bold tabular-nums">{globalBeta.toFixed(2)}</span>
            </div>

            <button
              onClick={() => setSoundEnabled((s) => !s)}
              className={`px-3 py-1 rounded-full text-[10px] font-mono uppercase tracking-[0.15em] transition border ${
                soundEnabled
                  ? "bg-[#041a1a]/60 border-[#5eead4]/40 text-[#5eead4]"
                  : "bg-[#05070f] border-[rgba(250,249,245,0.1)] text-[rgba(250,249,245,0.4)]"
              }`}
              title="Toggle Audio Clicks"
            >
              {soundEnabled ? "Audio: Active" : "Audio: Mute"}
            </button>

            <button
              onClick={() => setIsAutoSimulating((v) => !v)}
              className={`px-3 py-1 rounded-full text-[10px] font-mono uppercase tracking-[0.15em] transition border ${
                isAutoSimulating
                  ? "bg-[#041a1a]/60 border-[#5eead4]/40 text-[#5eead4]"
                  : "bg-[#05070f] border-[rgba(250,249,245,0.1)] text-[rgba(250,249,245,0.7)] hover:border-[rgba(250,249,245,0.3)]"
              }`}
            >
              {isAutoSimulating ? "Pause" : "Resume"}
            </button>

            <button
              onClick={handleCopyShare}
              className="px-3 py-1 rounded-full border border-[rgba(250,249,245,0.12)] bg-[#05070f] hover:border-[rgba(250,249,245,0.3)] text-[rgba(250,249,245,0.7)] hover:text-[#faf9f5] text-[10px] font-mono uppercase tracking-[0.15em] transition"
            >
              {copiedTweet ? "✓ Copied" : "Share"}
            </button>

            <a
              href="/BENCHMARK_DOSSIER.html"
              target="_blank"
              rel="noopener noreferrer"
              className="px-3 py-1 rounded-full border border-[rgba(250,249,245,0.12)] bg-[#05070f] hover:border-[rgba(250,249,245,0.3)] text-[rgba(250,249,245,0.85)] hover:text-[#faf9f5] text-[10px] font-mono uppercase tracking-[0.15em] transition"
              title="View & Download Full 16-Station Technical Whitepaper"
            >
              Whitepaper ↗
            </a>
          </div>

        </div>
      </div>

      {/* ===================================================================
          HERO HEADER (Editorial & Scientific Depth)
      =================================================================== */}
      <header className="relative overflow-hidden border-b border-[rgba(250,249,245,0.08)] bg-[#020308] py-16 px-4 sm:px-6 relative z-10">
        <div className="max-w-5xl mx-auto text-center space-y-6">
          
          <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-[#05070f] border border-[rgba(250,249,245,0.12)] text-[#5eead4] text-xs font-mono shadow-inner">
            <span className="w-1.5 h-1.5 rounded-full bg-[#5eead4]" />
            <span className="font-medium uppercase tracking-[0.18em]">Open-Source Physics Benchmarks · Extropic thrml v0.1.4</span>
          </div>

          <h1 className="font-[family-name:var(--font-display)] text-3xl sm:text-5xl lg:text-6xl font-normal tracking-[-0.03em] text-[#faf9f5] leading-[0.98]">
            Open-source thermodynamic sampling benchmarks,{" "}
            <em className="italic font-light text-[#5eead4]">
              reproducible on a laptop
            </em>
          </h1>

          <p className="text-base sm:text-lg text-[rgba(250,249,245,0.68)] max-w-3xl mx-auto leading-relaxed font-light">
            Sixteen interactive stations mapping Ising, Potts, and QUBO physics to JAX block Gibbs samplers and physical Langevin dynamics. Every station links to a standalone, open-source script in the repository that runs locally with no special hardware.
          </p>

          {/* Interactive Cryogenic Presets & Main Repo Star Links */}
          <div className="pt-4 flex flex-wrap justify-center items-center gap-3 font-mono text-[11px] uppercase tracking-[0.15em]">
            <a
              href="https://github.com/nexusai-kevin/thermo-physics"
              target="_blank"
              rel="noopener noreferrer"
              className="rounded-full bg-[#faf9f5] px-5 py-2 font-medium text-[#020308] hover:bg-white transition-colors"
            >
              View Python Scripts on GitHub ↗
            </a>
            <a
              href="/BENCHMARK_DOSSIER.html"
              target="_blank"
              rel="noopener noreferrer"
              className="rounded-full border border-[rgba(250,249,245,0.18)] bg-[#05070f] px-5 py-2 text-[#faf9f5] hover:border-[rgba(250,249,245,0.4)] transition-colors"
            >
              Technical Dossier (HTML) ↗
            </a>
            <button
              onClick={() => handleBetaChange(0.2)}
              className="rounded-full border border-[rgba(250,249,245,0.12)] bg-[#05070f] px-4 py-2 text-[rgba(250,249,245,0.7)] hover:border-[rgba(250,249,245,0.4)] hover:text-[#faf9f5] transition-colors"
            >
              // β = 0.20 · High Temp
            </button>
            <button
              onClick={() => handleBetaChange(0.44)}
              className="rounded-full border border-[#f59e0b]/40 bg-[#140e04]/60 px-4 py-2 text-[#f59e0b] hover:border-[#f59e0b] transition-colors"
            >
              // β = 0.44 · Critical Point
            </button>
            <button
              onClick={() => handleBetaChange(3.5)}
              className="rounded-full border border-[#5eead4]/40 bg-[#041a1a]/60 px-4 py-2 text-[#5eead4] hover:border-[#5eead4] transition-colors"
            >
              // β = 3.50 · Low Temp
            </button>
          </div>
        </div>
      </header>

      {/* ===================================================================
          16 INTERACTIVE STATIONS GRID
      =================================================================== */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-12 space-y-10">
        {/* ===================================================================
            PLUGGABLE EXECUTION BACKEND & REAL-TIME TELEMETRY PANEL
        =================================================================== */}
        <div className="rounded-2xl border border-[rgba(250,249,245,0.10)] bg-[#060811]/90 p-6 sm:p-8 shadow-2xl backdrop-blur-sm space-y-6">
          <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 border-b border-[rgba(250,249,245,0.08)] pb-5">
            <div>
              <div className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.22em] text-[#5eead4]">
                <span>// execution substrate · layer 4 pluggable adapter</span>
              </div>
              <h2 className="font-[family-name:var(--font-display)] text-xl sm:text-2xl font-normal text-[#faf9f5] mt-1 tracking-tight">
                Thermodynamic Sampling Substrate &amp; Telemetry
              </h2>
              <p className="text-xs text-[rgba(250,249,245,0.55)] mt-1 font-light">
                Toggle between Classical JAX Block Gibbs simulation and Extropic TSU hardware drop-in targets.
              </p>
            </div>

            {/* Backend Toggle */}
            <div className="flex items-center p-1 rounded-full bg-[#020308] border border-[rgba(250,249,245,0.10)] font-mono text-xs">
              <button
                onClick={() => {
                  playArcadeTone(440, 30);
                  setSelectedBackend("jax");
                }}
                className={`px-4 py-1.5 rounded-full text-[11px] uppercase tracking-[0.12em] transition ${
                  selectedBackend === "jax"
                    ? "bg-[#faf9f5] text-[#020308] font-medium"
                    : "text-[rgba(250,249,245,0.6)] hover:text-[#faf9f5]"
                }`}
              >
                JAX Block Gibbs (Measured)
              </button>
              <button
                onClick={() => {
                  playArcadeTone(660, 40, "square");
                  setSelectedBackend("extropic");
                }}
                className={`px-4 py-1.5 rounded-full text-[11px] uppercase tracking-[0.12em] transition ${
                  selectedBackend === "extropic"
                    ? "bg-[#5eead4] text-[#020308] font-medium"
                    : "text-[rgba(250,249,245,0.6)] hover:text-[#faf9f5]"
                }`}
              >
                Extropic TSU Model (Analytical Projection)
              </button>
            </div>
          </div>

          {/* Real-time Telemetry Readout Grid */}
          <div className="grid grid-cols-2 lg:grid-cols-5 gap-3 sm:gap-4">
            <div className="p-4 rounded-xl bg-[#020308]/60 border border-[rgba(250,249,245,0.08)] space-y-1">
              <div className="text-[10px] uppercase tracking-[0.15em] text-[rgba(250,249,245,0.4)] font-mono">Substrate Mode</div>
              <div className="text-sm font-mono text-[#5eead4] truncate">{backendTelemetry.name}</div>
              <div className="text-[10px] text-[rgba(250,249,245,0.4)] truncate">{backendTelemetry.chipType}</div>
            </div>
            <div className="p-4 rounded-xl bg-[#020308]/60 border border-[rgba(250,249,245,0.08)] space-y-1">
              <div className="text-[10px] uppercase tracking-[0.15em] text-[rgba(250,249,245,0.4)] font-mono">Wall-Clock Latency</div>
              <div className="text-sm font-mono text-[#faf9f5]">{backendTelemetry.runtime}</div>
              <div className="text-[10px] text-[rgba(250,249,245,0.4)]">5,000 Sweeps / Shots</div>
            </div>
            <div className="p-4 rounded-xl bg-[#020308]/60 border border-[rgba(250,249,245,0.08)] space-y-1">
              <div className="text-[10px] uppercase tracking-[0.15em] text-[rgba(250,249,245,0.4)] font-mono">Raw Throughput</div>
              <div className="text-sm font-mono text-[#faf9f5]">{backendTelemetry.sampleRate}</div>
              <div className="text-[10px] text-[rgba(250,249,245,0.4)]">Clocked Hardware Rate</div>
            </div>
            <div className="p-4 rounded-xl bg-[#020308]/60 border border-[rgba(250,249,245,0.08)] space-y-1">
              <div className="text-[10px] uppercase tracking-[0.15em] text-[rgba(250,249,245,0.4)] font-mono">Sokal MCMC τ_int</div>
              <div className="text-sm font-mono text-[#f59e0b]">{backendTelemetry.tauInt}</div>
              <div className="text-[10px] text-[rgba(250,249,245,0.4)]">Autocorrelation Window</div>
            </div>
            <div className="p-4 rounded-xl bg-[#020308]/60 border border-[rgba(250,249,245,0.08)] space-y-1">
              <div className="text-[10px] uppercase tracking-[0.15em] text-[rgba(250,249,245,0.4)] font-mono">Independent ESS/s</div>
              <div className="text-sm font-mono text-[#34d399]">{backendTelemetry.essPerSec}</div>
              <div className="text-[10px] text-[rgba(250,249,245,0.4)]">{backendTelemetry.advantage}</div>
            </div>
          </div>
          <div className="text-xs text-[rgba(250,249,245,0.55)] bg-[#020308]/40 p-3 rounded-xl border border-[rgba(250,249,245,0.06)] flex items-center justify-between font-light">
            <span>Hardware Architecture Note: {backendTelemetry.description}</span>
            <span className="font-mono text-[#5eead4] text-[11px] font-medium">{backendTelemetry.advantage}</span>
          </div>
        </div>

        {STATIONS.map((station) => (
          <section
            key={station.id}
            id={`station-${station.id}`}
            className="rounded-2xl border border-[rgba(250,249,245,0.10)] bg-[#060811]/90 hover:border-[rgba(250,249,245,0.22)] transition-all duration-300 shadow-2xl overflow-hidden scroll-mt-24"
          >
            <div className="p-6 sm:p-8 grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
              {/* Left Column: Story & Explanation */}
              <div className="lg:col-span-5 space-y-4">
                <div className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.2em] text-[#5eead4]">
                  <span>// {station.badge}</span>
                </div>

                <h2 className="font-[family-name:var(--font-display)] text-2xl sm:text-3xl font-normal text-[#faf9f5] tracking-tight">
                  {station.title}
                </h2>
                <p className="text-xs sm:text-sm text-[rgba(250,249,245,0.65)] font-light">
                  {station.subtitle}
                </p>

                <div className="p-4 sm:p-5 rounded-xl border border-[rgba(250,249,245,0.06)] bg-[#020308]/60 text-xs sm:text-sm font-light leading-relaxed text-[rgba(250,249,245,0.7)]">
                  {station.kidStory}
                </div>

                <div className="flex flex-wrap items-center gap-3 pt-1 text-xs font-mono text-[rgba(250,249,245,0.65)]">
                  <span className="px-3 py-1.5 rounded-lg border border-[rgba(250,249,245,0.08)] bg-[#020308]/60">
                    Verified Metric: <strong className="text-[#34d399] font-normal">{station.keyMetric}</strong>
                  </span>
                  <span className="px-3 py-1.5 rounded-lg border border-[rgba(250,249,245,0.08)] bg-[#020308]/60">
                    Runtime: <strong className="text-[#5eead4] font-normal">{station.runtime}</strong>
                  </span>
                </div>

                {/* Reproducible Open-Source Script & CLI Command */}
                <div className="p-3.5 rounded-xl border border-[rgba(94,234,212,0.25)] bg-[#041a1a]/50 flex flex-col gap-2 font-mono text-xs">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-[10px] uppercase tracking-wider text-[rgba(250,249,245,0.5)]">
                      Reproducible Script
                    </span>
                    <a
                      href={`https://github.com/nexusai-kevin/thermo-physics/blob/main/${station.scriptFile}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full border border-[#5eead4]/40 bg-[#020308] text-[11px] text-[#5eead4] hover:bg-[#5eead4]/10 transition-colors font-medium"
                    >
                      <span>{station.scriptFile}</span>
                      <span className="text-[10px] opacity-70">↗</span>
                    </a>
                  </div>
                  <div className="flex items-center justify-between text-[11px] text-[rgba(250,249,245,0.6)] bg-[#020308]/80 px-2.5 py-1.5 rounded-lg border border-[rgba(250,249,245,0.06)]">
                    <span className="text-[10px] text-[rgba(250,249,245,0.4)]">Run Locally:</span>
                    <code className="text-[#34d399] select-all">uv run thrml-bench run --exp {station.id}</code>
                  </div>
                </div>
              </div>

              {/* Right Column: Interactive Playable Widget */}
              <div className="lg:col-span-7 bg-[#020308]/90 border border-[rgba(250,249,245,0.08)] rounded-xl p-5 sm:p-6 flex flex-col items-center justify-center min-h-[260px] shadow-inner">
                {/* STATION 1 WIDGET */}
                {station.id === 1 && (
                  <div className="w-full space-y-5 text-center">
                    <div className="flex items-center justify-between text-xs text-[rgba(250,249,245,0.55)] px-2 font-mono">
                      <span>Click any spin to invert state · 1D nearest-neighbor coupling</span>
                      <span className="text-[#5eead4]">
                        Alignment |⟨M⟩| = {magnet1D}
                      </span>
                    </div>
                    <div className="grid grid-cols-8 sm:grid-cols-16 gap-2 py-4">
                      {chain1D.map((spin, idx) => (
                        <button
                          key={idx}
                          onClick={() => {
                            playArcadeTone(spin === 1 ? 340 : 620, 30);
                            setChain1D((arr) =>
                              arr.map((s, i) => (i === idx ? -s : s))
                            );
                          }}
                          className={`h-14 rounded-xl flex flex-col items-center justify-center font-mono text-xs transition-colors border ${
                            spin === 1
                              ? "bg-[rgba(94,234,212,0.08)] border-[#5eead4]/50 text-[#5eead4]"
                              : "bg-[rgba(244,63,94,0.08)] border-rose-500/50 text-rose-300"
                          }`}
                        >
                          <span className="text-sm">{spin === 1 ? "▲" : "▼"}</span>
                          <span className="text-[10px] opacity-75 font-mono">
                            {spin === 1 ? "+1" : "-1"}
                          </span>
                        </button>
                      ))}
                    </div>
                    <p className="text-xs text-[rgba(250,249,245,0.45)] font-light font-mono">
                      Checkerboard Sublattices: Even indices update in Cycle 1, Odd indices update in Cycle 2.
                    </p>
                  </div>
                )}

                {/* STATION 2 WIDGET */}
                {station.id === 2 && (
                  <div className="w-full flex flex-col sm:flex-row items-center justify-around gap-6">
                    <div
                      className="grid gap-1 p-2.5 bg-[#020308] rounded-xl border border-[rgba(250,249,245,0.08)]"
                      style={{ gridTemplateColumns: "repeat(16, minmax(0, 1fr))" }}
                    >
                      {grid2D.map((spin, idx) => (
                        <div
                          key={idx}
                          onClick={() => {
                            playArcadeTone(520, 25);
                            setGrid2D((g) => g.map((s, i) => (i === idx ? -s : s)));
                          }}
                          className={`w-3.5 h-3.5 sm:w-4 sm:h-4 rounded-[2px] cursor-pointer transition-colors ${
                            spin === 1 ? "bg-[#5eead4]" : "bg-rose-500/80"
                          }`}
                        />
                      ))}
                    </div>
                    <div className="space-y-3 text-left max-w-xs">
                      <div className="p-3.5 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.08)] space-y-1">
                        <div className="text-[11px] uppercase tracking-wider text-[rgba(250,249,245,0.45)] font-mono">
                          Live Order Parameter
                        </div>
                        <div className="text-2xl font-mono font-medium text-[#5eead4]">
                          |⟨M⟩| = {magnet2D}
                        </div>
                        <div className="text-xs text-[rgba(250,249,245,0.55)] font-mono">
                          Onsager Critical Point:{" "}
                          <span className="text-amber-300">βc ≈ 0.4407</span>
                        </div>
                      </div>
                      <div className="flex gap-2 font-mono text-[11px]">
                        <button
                          onClick={() => handleBetaChange(0.2)}
                          className="flex-1 py-1.5 px-2 rounded-lg bg-[#05070f] border border-[rgba(250,249,245,0.12)] hover:border-rose-400 text-rose-300 transition-colors"
                        >
                          β = 0.20
                        </button>
                        <button
                          onClick={() => handleBetaChange(0.44)}
                          className="flex-1 py-1.5 px-2 rounded-lg bg-[#05070f] border border-[rgba(250,249,245,0.12)] hover:border-amber-400 text-amber-300 transition-colors"
                        >
                          β = 0.44
                        </button>
                        <button
                          onClick={() => handleBetaChange(0.75)}
                          className="flex-1 py-1.5 px-2 rounded-lg bg-[#05070f] border border-[rgba(250,249,245,0.12)] hover:border-[#5eead4] text-[#5eead4] transition-colors"
                        >
                          β = 0.75
                        </button>
                      </div>
                    </div>
                  </div>
                )}

                {/* STATION 3 WIDGET */}
                {station.id === 3 && (
                  <div className="w-full flex flex-col sm:flex-row items-center justify-around gap-6">
                    <div className="relative w-56 h-56 bg-[#020308] rounded-full border border-[rgba(250,249,245,0.08)] flex items-center justify-center">
                      <svg className="absolute inset-0 w-full h-full" viewBox="0 0 200 200">
                        {rivalryEdges.map(([u, v], eIdx) => {
                          const angleU = (u / 8) * 2 * Math.PI - Math.PI / 2;
                          const angleV = (v / 8) * 2 * Math.PI - Math.PI / 2;
                          const x1 = 100 + 72 * Math.cos(angleU);
                          const y1 = 100 + 72 * Math.sin(angleU);
                          const x2 = 100 + 72 * Math.cos(angleV);
                          const y2 = 100 + 72 * Math.sin(angleV);
                          const isCut = monsterTeams[u] !== monsterTeams[v];
                          return (
                            <line
                              key={eIdx}
                              x1={x1}
                              y1={y1}
                              x2={x2}
                              y2={y2}
                              stroke={isCut ? "#34d399" : "#f43f5e"}
                              strokeWidth={isCut ? "2" : "1.2"}
                              strokeDasharray={isCut ? "none" : "3 2"}
                              opacity={isCut ? 0.9 : 0.45}
                            />
                          );
                        })}
                      </svg>
                      {monsterTeams.map((team, idx) => {
                        const angle = (idx / 8) * 2 * Math.PI - Math.PI / 2;
                        const left = 100 + 72 * Math.cos(angle);
                        const top = 100 + 72 * Math.sin(angle);
                        return (
                          <button
                            key={idx}
                            onClick={() => {
                              playArcadeTone(480, 35);
                              setMonsterTeams((t) =>
                                t.map((val, i) => (i === idx ? -val : val))
                              );
                            }}
                            style={{
                              left: `${(left / 200) * 100}%`,
                              top: `${(top / 200) * 100}%`,
                            }}
                            className={`absolute -translate-x-1/2 -translate-y-1/2 w-8 h-8 rounded-full font-mono text-xs flex items-center justify-center border transition-colors ${
                              team === 1
                                ? "bg-[#05070f] border-[#5eead4] text-[#5eead4]"
                                : "bg-[#05070f] border-rose-500 text-rose-300"
                            }`}
                          >
                            V{idx + 1}
                          </button>
                        );
                      })}
                    </div>

                    <div className="space-y-3 max-w-xs text-left">
                      <div className="p-3.5 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.08)]">
                        <div className="text-[11px] uppercase tracking-wider text-[rgba(250,249,245,0.45)] font-mono">
                          Graph Bipartition (Cut Edges)
                        </div>
                        <div className="text-2xl font-mono font-medium text-[#34d399] mt-0.5">
                          {happyRivalries} / {rivalryEdges.length} Edges (
                          {Math.round((happyRivalries / rivalryEdges.length) * 100)}%)
                        </div>
                        <p className="text-xs text-[rgba(250,249,245,0.55)] font-light mt-1.5 leading-relaxed">
                          Solid green lines: Satisfied antiferromagnetic cut (J &lt; 0). Dashed lines: Frustrated edge.
                        </p>
                      </div>
                    </div>
                  </div>
                )}

                {/* STATION 4 WIDGET */}
                {station.id === 4 && (
                  <div className="w-full flex flex-col sm:flex-row items-center justify-around gap-6">
                    <div
                      className="grid gap-1.5 p-3 bg-[#020308] rounded-xl border border-[rgba(250,249,245,0.08)]"
                      style={{ gridTemplateColumns: "repeat(10, minmax(0, 1fr))" }}
                    >
                      {pottsGrid.map((cat, idx) => (
                        <div
                          key={idx}
                          onClick={() => {
                            playArcadeTone(400 + cat * 110, 25);
                            setPottsGrid((g) =>
                              g.map((c, i) => (i === idx ? (c + 1) % 4 : c))
                            );
                          }}
                          className={`w-5 h-5 rounded-md cursor-pointer transition-colors shadow-sm ${pottsColors[cat]}`}
                        />
                      ))}
                    </div>
                    <div className="space-y-2.5 max-w-xs text-left text-xs text-[rgba(250,249,245,0.7)]">
                      <div className="font-mono text-xs uppercase tracking-wider text-[#faf9f5]">
                        4-State Categorical Spin Lattice
                      </div>
                      <p className="text-[rgba(250,249,245,0.55)] font-light leading-relaxed">
                        Each site samples from a 4-way Softmax distribution conditioned on neighbor consensus. Above{" "}
                        <span className="text-[#5eead4] font-mono">βc = ln(3) ≈ 1.10</span>, matching categorical domains nucleate and lock.
                      </p>
                    </div>
                  </div>
                )}

                {/* STATION 5 WIDGET */}
                {station.id === 5 && (
                  <div className="w-full flex flex-col sm:flex-row items-center justify-around gap-6">
                    <div className="flex items-center gap-4">
                      <div>
                        <div className="text-[10px] text-[rgba(250,249,245,0.45)] text-center mb-1.5 font-mono uppercase tracking-wider">
                          Corrupted (22% Noise)
                        </div>
                        <div className="grid grid-cols-8 gap-0.5 p-2 bg-[#020308] rounded-lg border border-[rgba(250,249,245,0.08)]">
                          {noisyDigit.map((v, i) => (
                            <div
                              key={i}
                              className={`w-3.5 h-3.5 rounded-[1px] ${
                                v === 1 ? "bg-amber-400" : "bg-[#05070f]"
                              }`}
                            />
                          ))}
                        </div>
                      </div>

                      <div className="text-[rgba(250,249,245,0.3)] text-lg font-mono">→</div>

                      <div>
                        <div className="text-[10px] text-[rgba(250,249,245,0.45)] text-center mb-1.5 font-mono uppercase tracking-wider">
                          Denoised Latent State
                        </div>
                        <div className="grid grid-cols-8 gap-0.5 p-2 bg-[#020308] rounded-lg border border-[rgba(250,249,245,0.12)]">
                          {denoisedDigit.map((v, i) => (
                            <div
                              key={i}
                              className={`w-3.5 h-3.5 rounded-[1px] ${
                                v === 1 ? "bg-[#5eead4]" : "bg-[#05070f]"
                              }`}
                            />
                          ))}
                        </div>
                      </div>
                    </div>

                    <div className="space-y-3 max-w-xs text-left">
                      <div className="p-3.5 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.08)]">
                        <div className="text-[11px] uppercase tracking-wider text-[rgba(250,249,245,0.45)] font-mono">
                          Reconstruction Error
                        </div>
                        <div className="text-2xl font-mono font-medium text-[#34d399] mt-0.5">
                          {denoisedError}% Misclassified
                        </div>
                      </div>
                      <div className="flex gap-2 font-mono text-[11px]">
                        <button
                          onClick={handleDenoiseStep}
                          className="flex-1 py-2 px-2.5 rounded-lg border border-[#5eead4]/40 bg-[#05070f] text-[#5eead4] hover:bg-[#5eead4]/10 transition-colors uppercase tracking-wider"
                        >
                          Execute Gibbs Sweep
                        </button>
                        <button
                          onClick={handleAddNoise}
                          className="flex-1 py-2 px-2.5 rounded-lg border border-[rgba(250,249,245,0.12)] bg-[#05070f] text-amber-300 hover:border-amber-400 transition-colors uppercase tracking-wider"
                        >
                          Inject Noise
                        </button>
                      </div>
                    </div>
                  </div>
                )}

                {/* STATION 6 WIDGET */}
                {station.id === 6 && (
                  <div className="w-full flex flex-col sm:flex-row items-center justify-around gap-6">
                    <div className="flex flex-col items-center gap-3">
                      <div className="grid grid-cols-8 gap-1 p-2.5 bg-[#020308] rounded-xl border border-[rgba(250,249,245,0.08)] relative">
                        {rbmState.map((val, idx) => (
                          <div
                            key={idx}
                            className={`w-4 h-4 rounded-sm transition-colors ${
                              val === 1 ? "bg-[#5eead4]" : "bg-[#05070f]"
                            } ${idx >= 32 ? "border border-amber-400/40" : ""}`}
                          />
                        ))}
                      </div>
                      <span className="text-[10px] text-amber-300 font-mono uppercase tracking-wider">
                        Bottom 32 spins masked
                      </span>
                    </div>

                    <div className="space-y-3 max-w-xs text-left">
                      <div className="p-3.5 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.08)]">
                        <div className="text-[11px] uppercase tracking-wider text-[rgba(250,249,245,0.45)] font-mono">
                          Associative Recall Overlap
                        </div>
                        <div className="text-2xl font-mono font-medium text-[#5eead4] mt-0.5">
                          {rbmOverlap}% Overlap
                        </div>
                      </div>
                      <div className="flex gap-1.5 font-mono text-[10px]">
                        {["Plus (+)", "Cross (×)", "Square (□)"].map((label, idx) => (
                          <button
                            key={idx}
                            onClick={() => handleSelectPrototype(idx)}
                            className={`flex-1 py-1.5 px-1.5 rounded-lg border transition-colors ${
                              selectedProtoIdx === idx
                                ? "bg-[#05070f] border-[#5eead4] text-[#5eead4]"
                                : "bg-[#020308] border-[rgba(250,249,245,0.1)] text-[rgba(250,249,245,0.55)] hover:text-[#faf9f5]"
                            }`}
                          >
                            {label}
                          </button>
                        ))}
                      </div>
                      <button
                        onClick={handleRecallMemory}
                        className="w-full py-2 rounded-full border border-[rgba(250,249,245,0.12)] bg-[#05070f] hover:border-[#34d399] text-[#34d399] font-mono text-xs uppercase tracking-wider transition-colors"
                      >
                        Reconstruct Occluded Bits
                      </button>
                    </div>
                  </div>
                )}

                {/* STATION 7 WIDGET */}
                {station.id === 7 && (
                  <div className="w-full flex flex-col sm:flex-row items-center justify-around gap-6">
                    <div className="grid grid-cols-5 gap-2 p-3.5 bg-[#020308] rounded-xl border border-[rgba(250,249,245,0.08)] max-w-xs">
                      {selectedAssets.map((sel, idx) => (
                        <button
                          key={idx}
                          onClick={() => {
                            playArcadeTone(sel ? 300 : 700, 30);
                            setSelectedAssets((arr) =>
                              arr.map((s, i) => (i === idx ? (s ? 0 : 1) : s))
                            );
                          }}
                          className={`h-11 rounded-lg flex flex-col items-center justify-center font-mono text-xs border transition-colors ${
                            sel
                              ? "bg-[rgba(52,211,153,0.1)] border-[#34d399] text-[#34d399]"
                              : "bg-[#05070f] border-[rgba(250,249,245,0.08)] text-[rgba(250,249,245,0.4)] hover:text-[#faf9f5]"
                          }`}
                        >
                          <span>A{idx + 1}</span>
                          <span className="text-[9px] opacity-75">{assetReturns[idx]}%</span>
                        </button>
                      ))}
                    </div>

                    <div className="space-y-3 max-w-xs text-left">
                      <div className="p-3.5 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.08)] space-y-1">
                        <div className="text-[11px] uppercase tracking-wider text-[rgba(250,249,245,0.45)] font-mono">
                          Selected Portfolio Assets
                        </div>
                        <div
                          className={`text-2xl font-mono font-medium ${
                            quboCount === 5 ? "text-[#34d399]" : "text-amber-400"
                          }`}
                        >
                          {quboCount} / 5 Assets {quboCount === 5 ? "(Feasible)" : "(Penalty Active)"}
                        </div>
                        <div className="text-xs text-[rgba(250,249,245,0.55)] font-mono">
                          Expected Return (Equal-Weighted):{" "}
                          <strong className="text-[#5eead4]">{quboTotalReturn}%</strong>
                        </div>
                      </div>

                      <div className="space-y-1 font-mono text-xs">
                        <div className="flex justify-between text-[rgba(250,249,245,0.55)]">
                          <span>Hard Constraint Weight (λ)</span>
                          <span className="text-[#5eead4]">{quboLambda.toFixed(2)}</span>
                        </div>
                        <input
                          type="range"
                          min={0.1}
                          max={2.0}
                          step={0.05}
                          value={quboLambda}
                          onChange={(e) => setQuboLambda(parseFloat(e.target.value))}
                          className="w-full accent-[#34d399] cursor-pointer h-1.5 rounded-lg bg-[#020308]"
                        />
                      </div>
                    </div>
                  </div>
                )}

                {/* STATION 8 WIDGET */}
                {station.id === 8 && (
                  <div className="w-full space-y-4">
                    <div className="flex justify-between text-xs text-[rgba(250,249,245,0.45)] px-2 font-mono">
                      <span>Left Valley (x ≈ -1.0)</span>
                      <span className="text-amber-300">Center Barrier (x = 0)</span>
                      <span>Right Valley (x ≈ +1.0)</span>
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
                      {oscillators.map((val, idx) => {
                        const clamped = Math.max(-1.5, Math.min(1.5, val));
                        const pct = ((clamped + 1.5) / 3.0) * 100;
                        const inLeftWell = val < -0.3;
                        const inRightWell = val > 0.3;
                        return (
                          <div
                            key={idx}
                            className="p-2.5 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.08)] flex items-center gap-2"
                          >
                            <span className="text-[11px] font-mono text-[rgba(250,249,245,0.45)] w-6">
                              #{idx + 1}
                            </span>
                            <div className="flex-1 relative h-6 bg-[#020308] rounded-lg border border-[rgba(250,249,245,0.08)] flex items-center px-1">
                              <div className="absolute left-1/2 top-0 bottom-0 w-[1px] bg-amber-400/40" />
                              <div
                                style={{ left: `calc(${pct}% - 6px)` }}
                                className={`absolute w-3 h-3 rounded-full transition-all duration-75 ${
                                  inLeftWell
                                    ? "bg-[#5eead4]"
                                    : inRightWell
                                    ? "bg-[#34d399]"
                                    : "bg-rose-400"
                                }`}
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                    <div className="flex flex-wrap justify-between items-center gap-2 text-xs text-[rgba(250,249,245,0.55)] px-2 font-mono">
                      <span>
                        Mean Analog Amplitude |x|:{" "}
                        <strong className="text-[#5eead4]">
                          {(
                            oscillators.reduce((a, v) => a + Math.abs(v), 0) /
                            oscillators.length
                          ).toFixed(2)}
                        </strong>
                      </span>
                      <button
                        onClick={() => {
                          playArcadeTone(740, 45, "sine");
                          setOscillators((prev) => prev.map((x) => -x));
                        }}
                        className="px-3 py-1.5 rounded-lg bg-[#05070f] hover:border-[rgba(250,249,245,0.25)] border border-[rgba(250,249,245,0.12)] text-[#5eead4] uppercase tracking-wider text-[11px] transition-colors"
                      >
                        Perturb Oscillators Across Barrier
                      </button>
                    </div>
                  </div>
                )}

                {/* STATION 9 WIDGET */}
                {station.id === 9 && (
                  <div className="w-full flex flex-col sm:flex-row items-center justify-around gap-6">
                    <div className="relative w-56 h-56 bg-[#020308] rounded-2xl border border-[rgba(250,249,245,0.08)] p-2">
                      <svg className="w-full h-full" viewBox="0 0 200 200">
                        <polygon
                          points={tspTour
                            .map((cIdx) => `${tspCoords[cIdx].x},${tspCoords[cIdx].y}`)
                            .join(" ")}
                          fill="rgba(52, 211, 153, 0.06)"
                          stroke="#34d399"
                          strokeWidth="2"
                        />
                        {tspCoords.map((pt, cIdx) => (
                          <g key={cIdx}>
                            <circle
                              cx={pt.x}
                              cy={pt.y}
                              r="6"
                              fill="#5eead4"
                              stroke="#020308"
                              strokeWidth="2"
                            />
                            <text
                              x={pt.x}
                              y={pt.y - 9}
                              textAnchor="middle"
                              fill="#faf9f5"
                              fontSize="9"
                              fontFamily="monospace"
                            >
                              C{cIdx}
                            </text>
                          </g>
                        ))}
                      </svg>
                    </div>

                    <div className="space-y-3 max-w-xs text-left">
                      <div className="p-3.5 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.08)]">
                        <div className="text-[11px] uppercase tracking-wider text-[rgba(250,249,245,0.45)] font-mono">
                          Live 10-City Tour Distance
                        </div>
                        <div className="text-2xl font-mono font-medium text-[#34d399] mt-0.5">
                          L = {tspCurrentLength}
                        </div>
                        <div className="text-xs text-[rgba(250,249,245,0.55)] mt-1 font-mono">
                          Optimal Perimeter Ring: <span className="text-[#5eead4]">6.29</span>
                        </div>
                      </div>
                      <div className="flex gap-2 font-mono text-[11px]">
                        <button
                          onClick={() => {
                            playArcadeTone(880, 60, "sine");
                            setTspTour([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
                          }}
                          className="flex-1 py-2 px-2.5 rounded-lg border border-[#34d399]/40 bg-[#05070f] text-[#34d399] hover:bg-[#34d399]/10 transition-colors uppercase tracking-wider"
                        >
                          Snap to Optimal Ring
                        </button>
                        <button
                          onClick={() => {
                            playArcadeTone(240, 50, "sawtooth");
                            setTspTour([0, 5, 2, 8, 4, 9, 1, 6, 3, 7]);
                          }}
                          className="flex-1 py-2 px-2.5 rounded-lg border border-[rgba(250,249,245,0.12)] bg-[#05070f] text-rose-300 hover:border-rose-400 transition-colors uppercase tracking-wider"
                        >
                          Scramble Tour
                        </button>
                      </div>
                    </div>
                  </div>
                )}

                {/* STATION 10 WIDGET: SK FRUSTRATED SPIN GLASS */}
                {station.id === 10 && (
                  <div className="w-full flex flex-col gap-6">
                    {/* Top Controls */}
                    <div className="flex flex-wrap items-center justify-between gap-3 p-3.5 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.08)]">
                      <div className="flex items-center gap-3 flex-1 min-w-[220px]">
                        <span className="text-xs font-mono text-[rgba(250,249,245,0.7)] uppercase tracking-wider">Temp T:</span>
                        <input
                          type="range"
                          min={0.2}
                          max={3.0}
                          step={0.05}
                          value={skTemp}
                          onChange={(e) => setSkTemp(parseFloat(e.target.value))}
                          className="w-full accent-[#5eead4] cursor-pointer h-1.5 rounded-lg bg-[#020308]"
                        />
                        <span className="text-xs font-mono text-[#5eead4] whitespace-nowrap">
                          T = {skTemp.toFixed(2)} (β = {(1 / skTemp).toFixed(2)})
                        </span>
                      </div>
                      <div className="flex flex-wrap items-center gap-2 font-mono text-[11px]">
                        <button
                          onClick={() => {
                            playArcadeTone(320, 30);
                            setSkTemp(0.4);
                          }}
                          className={`px-3 py-1 rounded-lg border transition-colors ${
                            skTemp === 0.4
                              ? "bg-[#020308] border-[#5eead4] text-[#5eead4]"
                              : "bg-[#020308] border-[rgba(250,249,245,0.1)] text-[rgba(250,249,245,0.55)] hover:text-[#faf9f5]"
                          }`}
                        >
                          Low-T Glass (T = 0.40)
                        </button>
                        <button
                          onClick={() => {
                            playArcadeTone(580, 30);
                            setSkTemp(2.0);
                          }}
                          className={`px-3 py-1 rounded-lg border transition-colors ${
                            skTemp === 2.0
                              ? "bg-[#020308] border-rose-400 text-rose-300"
                              : "bg-[#020308] border-[rgba(250,249,245,0.1)] text-[rgba(250,249,245,0.55)] hover:text-[#faf9f5]"
                          }`}
                        >
                          High-T Paramagnetic (T = 2.00)
                        </button>
                        <button
                          onClick={() => {
                            playArcadeTone(700, 45, "sine");
                            setSkSeed((s) => s + 1);
                            setSkOverlapHistory([]);
                          }}
                          className="px-3 py-1 rounded-lg bg-[#020308] border border-[rgba(250,249,245,0.12)] hover:border-[#34d399] text-[#34d399] transition-colors"
                        >
                          Resample Couplings J_ij (Seed {skSeed})
                        </button>
                      </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-12 gap-6 items-center">
                      {/* Parisi Overlap Histogram */}
                      <div className="md:col-span-7 p-4 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.08)] space-y-2">
                        <div className="flex justify-between items-center text-xs">
                          <span className="font-mono text-[11px] uppercase tracking-wider text-[rgba(250,249,245,0.7)]">
                            Parisi Overlap P(q) Histogram ({skOverlapHistory.length} Samples)
                          </span>
                          <span className="text-[11px] font-mono text-[#5eead4]">
                            {skTemp < 1.0 ? "Bimodal Glass Valleys" : "Paramagnetic Single Peak"}
                          </span>
                        </div>
                        <div className="h-28 flex items-end justify-between gap-1 pt-2 pb-1 px-1 bg-[#020308] rounded-lg border border-[rgba(250,249,245,0.08)]">
                          {skHistogram.map((pct, idx) => {
                            const isCenter = idx >= 4 && idx <= 6;
                            return (
                              <div key={idx} className="flex-1 flex flex-col items-center h-full justify-end group">
                                <div
                                  style={{ height: `${Math.max(4, pct)}%` }}
                                  className={`w-full rounded-t-sm transition-all duration-150 ${
                                    isCenter
                                      ? "bg-amber-400/80 group-hover:bg-amber-300"
                                      : "bg-[#5eead4]/80 group-hover:bg-[#5eead4]"
                                  }`}
                                />
                                <span className="text-[8px] font-mono text-[rgba(250,249,245,0.4)] mt-1">
                                  {(idx * 0.18 - 0.9).toFixed(1)}
                                </span>
                              </div>
                            );
                          })}
                        </div>
                        <div className="flex justify-between text-[10px] text-[rgba(250,249,245,0.45)] font-mono px-1">
                          <span>q = -1.0 (Opposite)</span>
                          <span>q = 0.0 (Uncorrelated)</span>
                          <span>q = +1.0 (Identical)</span>
                        </div>
                      </div>

                      {/* Coupling Matrix Heatmap */}
                      <div className="md:col-span-5 flex flex-col items-center gap-2">
                        <div className="text-[10px] uppercase tracking-wider text-[rgba(250,249,245,0.45)] font-mono">
                          Dense 24×24 Couplings J_ij (276 Edges)
                        </div>
                        <div className="relative w-36 h-36 bg-[#020308] rounded-lg border border-[rgba(250,249,245,0.08)] p-1">
                          <svg className="w-full h-full" viewBox="0 0 24 24">
                            {skMatrix.map((row, r) =>
                              row.map((val, c) => {
                                if (r === c) return null;
                                const isPos = val > 0;
                                const intensity = Math.min(1, Math.abs(val) * 2.8);
                                return (
                                  <rect
                                    key={`${r}-${c}`}
                                    x={c}
                                    y={r}
                                    width={0.9}
                                    height={0.9}
                                    fill={
                                      isPos
                                        ? `rgba(94, 234, 212, ${intensity})`
                                        : `rgba(244, 63, 94, ${intensity})`
                                    }
                                  />
                                );
                              })
                            )}
                          </svg>
                        </div>
                        <div className="flex items-center gap-3 text-[10px] text-[rgba(250,249,245,0.45)] font-mono">
                          <span className="flex items-center gap-1">
                            <span className="w-2 h-2 rounded-sm bg-[#5eead4]" /> J &gt; 0 (Ferro)
                          </span>
                          <span className="flex items-center gap-1">
                            <span className="w-2 h-2 rounded-sm bg-rose-500" /> J &lt; 0 (Frustrated)
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* Metrics HUD */}
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
                      <div className="p-3.5 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.08)]">
                        <div className="text-[10px] uppercase tracking-wider text-[rgba(250,249,245,0.4)] font-mono">Ground State Baseline</div>
                        <div className="text-lg font-mono font-medium text-[#34d399] mt-1">
                          -15.4266
                        </div>
                        <div className="text-[10px] text-[rgba(250,249,245,0.45)] font-mono mt-0.5">Exact 2^24 Brute Force</div>
                      </div>
                      <div className="p-3.5 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.08)]">
                        <div className="text-[10px] uppercase tracking-wider text-[rgba(250,249,245,0.4)] font-mono">Live Sampled Energy</div>
                        <div className="text-lg font-mono font-medium text-[#5eead4] mt-1">
                          {skEnergyA.toFixed(2)}
                        </div>
                        <div className="text-[10px] text-[rgba(250,249,245,0.45)] font-mono mt-0.5">Unitless H (Replica α)</div>
                      </div>
                      <div className="p-3.5 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.08)]">
                        <div className="text-[10px] uppercase tracking-wider text-[rgba(250,249,245,0.4)] font-mono">Order Parameter ⟨|q|⟩</div>
                        <div className="text-lg font-mono font-medium text-[#5eead4] mt-1">
                          {skMeanAbsQ}
                        </div>
                        <div className="text-[10px] text-[rgba(250,249,245,0.45)] font-mono mt-0.5">Twin Replicas Overlap</div>
                      </div>
                      <div className="p-3.5 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.08)]">
                        <div className="text-[10px] uppercase tracking-wider text-[rgba(250,249,245,0.4)] font-mono">Overlap Variance Var(q)</div>
                        <div className="text-lg font-mono font-medium text-amber-300 mt-1">
                          {skVarQ}
                        </div>
                        <div className="text-[10px] text-[rgba(250,249,245,0.45)] font-mono mt-0.5">Multi-Valley Support</div>
                      </div>
                    </div>
                  </div>
                )}

                {/* STATION 11 WIDGET: THERMODYNAMIC RNA FOLDING ENSEMBLE */}
                {station.id === 11 && (
                  <div className="w-full flex flex-col gap-5">
                    {/* Controls Bar */}
                    <div className="flex flex-wrap items-center justify-between gap-3 p-3.5 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.08)]">
                      <div className="flex items-center gap-2 font-mono text-[11px]">
                        <span className="text-[rgba(250,249,245,0.45)] uppercase tracking-wider">Sequence Tier:</span>
                        <button
                          onClick={() => {
                            playArcadeTone(420, 30);
                            setRnaTier(24);
                          }}
                          className={`px-3 py-1 rounded-lg border transition-colors ${
                            rnaTier === 24
                              ? "bg-[#020308] border-[#5eead4] text-[#5eead4]"
                              : "bg-[#020308] border-[rgba(250,249,245,0.1)] text-[rgba(250,249,245,0.55)] hover:text-[#faf9f5]"
                          }`}
                        >
                          24-nt Switch (M = 24)
                        </button>
                        <button
                          onClick={() => {
                            playArcadeTone(520, 30);
                            setRnaTier(50);
                          }}
                          className={`px-3 py-1 rounded-lg border transition-colors ${
                            rnaTier === 50
                              ? "bg-[#020308] border-[#5eead4] text-[#5eead4]"
                              : "bg-[#020308] border-[rgba(250,249,245,0.1)] text-[rgba(250,249,245,0.55)] hover:text-[#faf9f5]"
                          }`}
                        >
                          50-nt Aptamer Core (M = 72)
                        </button>
                      </div>

                      <div className="flex items-center gap-2 font-mono text-[11px]">
                        <span className="text-[rgba(250,249,245,0.45)] uppercase tracking-wider">Thermal Regime:</span>
                        <button
                          onClick={() => {
                            playArcadeTone(620, 30, "sine");
                            setRnaTempMode("native");
                          }}
                          className={`px-3 py-1 rounded-lg border transition-colors ${
                            rnaTempMode === "native"
                              ? "bg-[#020308] border-[#34d399] text-[#34d399]"
                              : "bg-[#020308] border-[rgba(250,249,245,0.1)] text-[rgba(250,249,245,0.55)] hover:text-[#faf9f5]"
                          }`}
                        >
                          Native Fold (37 °C, β = 1.0)
                        </button>
                        <button
                          onClick={() => {
                            playArcadeTone(300, 35, "sawtooth");
                            setRnaTempMode("melted");
                          }}
                          className={`px-3 py-1 rounded-lg border transition-colors ${
                            rnaTempMode === "melted"
                              ? "bg-[#020308] border-rose-400 text-rose-300"
                              : "bg-[#020308] border-[rgba(250,249,245,0.1)] text-[rgba(250,249,245,0.55)] hover:text-[#faf9f5]"
                          }`}
                        >
                          Denatured Coil (T_melt, β = 0.2)
                        </button>
                      </div>
                    </div>

                    {/* Dynamic Dot-Bracket Banner */}
                    <div className="p-3.5 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.08)] space-y-1.5 text-left">
                      <div className="flex justify-between items-center text-xs">
                        <span className="text-[11px] font-mono uppercase tracking-wider text-[rgba(250,249,245,0.45)]">
                          Live Sampled Secondary Structure (Dot-Bracket)
                        </span>
                        <span className="text-[#34d399] font-mono font-medium">
                          {rnaActivePairs.length} Active Stems
                        </span>
                      </div>
                      <div className="p-2.5 rounded-lg bg-[#020308] font-mono text-xs sm:text-sm text-[#5eead4] tracking-wider overflow-x-auto break-all border border-[rgba(250,249,245,0.08)]">
                        {rnaDotBracket}
                      </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-12 gap-5 items-center">
                      {/* Pairing Probability Matrix Heatmap */}
                      <div className="md:col-span-6 flex flex-col items-center gap-2 p-4 bg-[#05070f] rounded-xl border border-[rgba(250,249,245,0.08)]">
                        <div className="text-[10px] uppercase tracking-wider text-[rgba(250,249,245,0.45)] font-mono">
                          Pairing Probability Matrix P_ij = ⟨x_ij⟩ ({rnaTier}×{rnaTier})
                        </div>
                        <div
                          className="relative w-44 h-44 bg-[#020308] rounded-lg border border-[rgba(250,249,245,0.08)] p-1"
                        >
                          <svg className="w-full h-full" viewBox={`0 0 ${rnaTier} ${rnaTier}`}>
                            {rnaPairs.map((p, idx) => {
                              const prob = rnaPairProbs[idx] || 0.05;
                              return (
                                <g key={idx}>
                                  <rect
                                    x={p.j}
                                    y={p.i}
                                    width={1.2}
                                    height={1.2}
                                    fill={`rgba(94, 234, 212, ${Math.min(1, prob * 1.4)})`}
                                  />
                                  <rect
                                    x={p.i}
                                    y={p.j}
                                    width={1.2}
                                    height={1.2}
                                    fill={`rgba(94, 234, 212, ${Math.min(1, prob * 1.4)})`}
                                  />
                                </g>
                              );
                            })}
                          </svg>
                        </div>
                        <span className="text-[10px] text-[rgba(250,249,245,0.4)] font-mono">
                          Bright pixels: high marginal pairing occupancy
                        </span>
                      </div>

                      {/* Conformational Shannon Entropy Meter & Callout */}
                      <div className="md:col-span-6 space-y-3 text-left">
                        <div className="p-4 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.08)] space-y-2">
                          <div className="flex justify-between items-center text-xs">
                            <span className="text-[11px] font-mono uppercase tracking-wider text-[rgba(250,249,245,0.45)]">
                              Conformational Shannon Entropy
                            </span>
                            <span className="font-mono font-medium text-amber-300">
                              {rnaEntropy.toFixed(2)} bits
                            </span>
                          </div>
                          <div className="h-2 w-full bg-[#020308] rounded-full overflow-hidden border border-[rgba(250,249,245,0.08)]">
                            <div
                              style={{ width: `${Math.min(100, (rnaEntropy / 16.0) * 100)}%` }}
                              className={`h-full transition-all duration-300 rounded-full ${
                                rnaTempMode === "native"
                                  ? "bg-gradient-to-r from-[#34d399] to-[#5eead4]"
                                  : "bg-gradient-to-r from-amber-400 to-rose-500"
                              }`}
                            />
                          </div>
                          <div className="flex justify-between text-[10px] font-mono text-[rgba(250,249,245,0.4)]">
                            <span>0 bits (Ordered Basin)</span>
                            <span>{rnaTempMode === "native" ? "Native Fold" : "Denatured"}</span>
                            <span>16 bits (Random Coil)</span>
                          </div>
                        </div>

                        {/* Biophysical Scope Callout */}
                        <div className="p-3.5 rounded-xl bg-[#020308]/60 border border-[rgba(250,249,245,0.08)] text-[11px] text-[rgba(250,249,245,0.55)] font-light leading-relaxed">
                          <strong className="text-[#5eead4] font-medium font-mono">Biophysical Model:</strong> Nearest-neighbor stacking (σ = -2.0) and affinities (GC = -3.0, AU = -2.0, GU = -1.0 kcal/mol) demonstrates thermodynamic conformational sampling.
                        </div>
                      </div>
                    </div>

                    {/* Metrics HUD */}
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
                      <div className="p-3.5 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.08)]">
                        <div className="text-[10px] uppercase tracking-wider text-[rgba(250,249,245,0.4)] font-mono">Ground State Baseline</div>
                        <div className="text-lg font-mono font-medium text-[#34d399] mt-1">
                          {rnaTier === 24 ? "-10.0000" : "-66.0000"}
                        </div>
                        <div className="text-[10px] text-[rgba(250,249,245,0.45)] font-mono mt-0.5">kcal/mol (MFE)</div>
                      </div>
                      <div className="p-3.5 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.08)]">
                        <div className="text-[10px] uppercase tracking-wider text-[rgba(250,249,245,0.4)] font-mono">Live Sampled Energy</div>
                        <div className="text-lg font-mono font-medium text-[#5eead4] mt-1">
                          {rnaEnergy.toFixed(2)}
                        </div>
                        <div className="text-[10px] text-[rgba(250,249,245,0.45)] font-mono mt-0.5">kcal/mol</div>
                      </div>
                      <div className="p-3.5 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.08)]">
                        <div className="text-[10px] uppercase tracking-wider text-[rgba(250,249,245,0.4)] font-mono">Active Contacts (M)</div>
                        <div className="text-lg font-mono font-medium text-[#5eead4] mt-1">
                          {rnaActivePairs.length} / {rnaPairs.length}
                        </div>
                        <div className="text-[10px] text-[rgba(250,249,245,0.45)] font-mono mt-0.5">Paired Stems</div>
                      </div>
                      <div className="p-3.5 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.08)]">
                        <div className="text-[10px] uppercase tracking-wider text-[rgba(250,249,245,0.4)] font-mono">Sokal Autocorrelation</div>
                        <div className="text-lg font-mono font-medium text-amber-300 mt-1">
                          {rnaTier === 24 ? "38.22" : "7.32"} sweeps
                        </div>
                        <div className="text-[10px] text-[rgba(250,249,245,0.45)] font-mono mt-0.5">MCMC τ_int</div>
                      </div>
                    </div>
                  </div>
                )}

                {/* STATION 12 WIDGET: POWER GRID SCUC QUBO */}
                {station.id === 12 && (
                  <div className="w-full flex flex-col gap-6">
                    {/* Top Controls */}
                    <div className="flex flex-wrap items-center justify-between gap-3 p-3.5 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.08)]">
                      <div className="flex items-center gap-3 flex-1 min-w-[220px]">
                        <span className="text-xs font-mono text-[rgba(250,249,245,0.7)] uppercase tracking-wider">Penalty Weight λ:</span>
                        <input
                          type="range"
                          min={0.001}
                          max={0.02}
                          step={0.001}
                          value={scucPenaltyWeight}
                          onChange={(e) => setScucPenaltyWeight(parseFloat(e.target.value))}
                          className="w-full accent-[#5eead4] cursor-pointer h-1.5 rounded-lg bg-[#020308]"
                        />
                        <span className="text-xs font-mono text-[#5eead4] whitespace-nowrap">
                          λ = {scucPenaltyWeight.toFixed(4)}
                        </span>
                      </div>
                      <div className="flex flex-wrap items-center gap-2 font-mono text-[11px]">
                        <button
                          onClick={() => {
                            playArcadeTone(520, 40);
                            setScucSchedule([
                              [1, 0, 0, 1, 1, 1],
                              [0, 0, 1, 0, 0, 0],
                              [0, 1, 1, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0],
                            ]);
                          }}
                          className="px-3 py-1 rounded-lg border border-[#34d399]/40 bg-[#020308] text-[#34d399] hover:bg-[#34d399]/10 transition-colors uppercase tracking-wider"
                        >
                          Optimal QUBO Dispatch
                        </button>
                        <button
                          onClick={() => {
                            playArcadeTone(350, 40, "sawtooth");
                            setScucSchedule([
                              [1, 1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1, 1],
                            ]);
                          }}
                          className="px-3 py-1 rounded-lg border border-[rgba(250,249,245,0.12)] bg-[#020308] text-amber-300 hover:border-amber-400 transition-colors uppercase tracking-wider"
                        >
                          All Units Active
                        </button>
                        <button
                          onClick={() => {
                            playArcadeTone(220, 50, "sawtooth");
                            setScucSchedule([
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                            ]);
                          }}
                          className="px-3 py-1 rounded-lg border border-[rgba(250,249,245,0.12)] bg-[#020308] text-rose-300 hover:border-rose-400 transition-colors uppercase tracking-wider"
                        >
                          Zero Commitment Baseline
                        </button>
                      </div>
                    </div>

                    {/* Interactive 4-Gen x 6-Hour Dispatch Grid */}
                    <div className="p-4 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.08)] space-y-3">
                      <div className="flex flex-wrap justify-between items-center gap-2 text-xs">
                        <span className="text-[11px] font-mono uppercase tracking-wider text-[rgba(250,249,245,0.7)]">
                          Unit Commitment Matrix x_{"{g,t}"} ∈ &#123;0, 1&#125; (Click Cell to Toggle)
                        </span>
                        <span className="text-[#5eead4] font-mono text-[11px]">
                          4 Generators × 6 Time Steps = 24 Spins
                        </span>
                      </div>

                      <div className="overflow-x-auto">
                        <table className="w-full text-left border-collapse text-xs">
                          <thead>
                            <tr className="border-b border-[rgba(250,249,245,0.08)] text-[rgba(250,249,245,0.45)] font-mono text-[11px]">
                              <th className="py-2 px-3">Generator Unit</th>
                              <th className="py-2 px-2 text-right">P_max</th>
                              <th className="py-2 px-2 text-right">Cost</th>
                              {scucDemands.map((d, t) => (
                                <th key={t} className="py-2 px-2 text-center">
                                  <div>H{t + 1}</div>
                                  <div className="text-[10px] text-[#5eead4]">{d} MW</div>
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {scucGenerators.map((gen, gIdx) => (
                              <tr key={gen.id} className="border-b border-[rgba(250,249,245,0.04)] hover:bg-[#020308]">
                                <td className="py-2.5 px-3 font-medium text-[#faf9f5]">
                                  <div className="flex items-center gap-2">
                                    <span className={`w-2 h-2 rounded-full ${gen.color.split(" ")[0].replace("text-", "bg-")}`} />
                                    <span>{gen.name}</span>
                                  </div>
                                </td>
                                <td className="py-2.5 px-2 text-right font-mono text-[rgba(250,249,245,0.55)]">{gen.capacity} MW</td>
                                <td className="py-2.5 px-2 text-right font-mono text-[rgba(250,249,245,0.55)]">${gen.cost}/MWh</td>
                                {scucDemands.map((_, tIdx) => {
                                  const isOn = scucSchedule[gIdx][tIdx] === 1;
                                  return (
                                    <td key={tIdx} className="py-2 px-1 text-center">
                                      <button
                                        onClick={() => {
                                          playArcadeTone(isOn ? 260 : 480, 25);
                                          setScucSchedule((prev) => {
                                            const copy = prev.map((row) => [...row]);
                                            copy[gIdx][tIdx] = isOn ? 0 : 1;
                                            return copy;
                                          });
                                        }}
                                        className={`w-full py-1.5 px-2 rounded-lg font-mono text-xs transition border ${
                                          isOn
                                            ? "bg-[rgba(52,211,153,0.12)] border-[#34d399] text-[#34d399]"
                                            : "bg-[#020308] border-[rgba(250,249,245,0.08)] text-[rgba(250,249,245,0.3)] hover:text-[#faf9f5] hover:border-[rgba(250,249,245,0.2)]"
                                        }`}
                                      >
                                        {isOn ? "ON" : "OFF"}
                                      </button>
                                    </td>
                                  );
                                })}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>

                    {/* Hourly Demand vs. Generation Balance Meter */}
                    <div className="p-4 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.08)] space-y-3">
                      <div className="flex justify-between items-center text-xs">
                        <span className="text-[11px] font-mono uppercase tracking-wider text-[rgba(250,249,245,0.7)]">
                          Hourly System Power Balance (Delivered vs. Demand)
                        </span>
                        <span className="text-[rgba(250,249,245,0.45)] font-mono text-[11px]">
                          Target: Balance within ±50 MW
                        </span>
                      </div>

                      <div className="grid grid-cols-2 sm:grid-cols-6 gap-2.5">
                        {scucDemands.map((demand, t) => {
                          const delivered = scucDelivered[t];
                          const delta = delivered - demand;
                          const isBalanced = Math.abs(delta) <= 70;
                          const isDeficit = delta < -70;

                          return (
                            <div
                              key={t}
                              className={`p-2.5 rounded-xl border text-center font-mono ${
                                isBalanced
                                  ? "bg-[rgba(52,211,153,0.08)] border-[#34d399]/40 text-[#34d399]"
                                  : isDeficit
                                  ? "bg-[rgba(244,63,94,0.08)] border-rose-500/40 text-rose-300"
                                  : "bg-[rgba(245,158,11,0.08)] border-amber-500/40 text-amber-300"
                              }`}
                            >
                              <div className="text-[10px] text-[rgba(250,249,245,0.45)]">Hour {t + 1}</div>
                              <div className="text-sm font-medium mt-0.5">{delivered} MW</div>
                              <div className="text-[10px] text-[rgba(250,249,245,0.4)]">of {demand} MW</div>
                              <div className="text-[10px] font-medium mt-1">
                                {delta === 0 ? "✓ Exact" : delta > 0 ? `+${delta} MW` : `${delta} MW`}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>

                    {/* Metrics HUD */}
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
                      <div className="p-3.5 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.08)]">
                        <div className="text-[10px] uppercase tracking-wider text-[rgba(250,249,245,0.4)] font-mono">Optimal Hamiltonian E*</div>
                        <div className="text-lg font-mono font-medium text-[#34d399] mt-1">
                          953.50
                        </div>
                        <div className="text-[10px] text-[rgba(250,249,245,0.45)] font-mono mt-0.5">Ground State Target</div>
                      </div>
                      <div className="p-3.5 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.08)]">
                        <div className="text-[10px] uppercase tracking-wider text-[rgba(250,249,245,0.4)] font-mono">Current Energy H(x)</div>
                        <div className="text-lg font-mono font-medium text-[#5eead4] mt-1">
                          {scucTotalEnergy.toFixed(2)}
                        </div>
                        <div className="text-[10px] text-[rgba(250,249,245,0.45)] font-mono mt-0.5">Cost + Penalty</div>
                      </div>
                      <div className="p-3.5 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.08)]">
                        <div className="text-[10px] uppercase tracking-wider text-[rgba(250,249,245,0.4)] font-mono">Production Cost</div>
                        <div className="text-lg font-mono font-medium text-[#5eead4] mt-1">
                          ${(scucTotalGenCost * 100).toFixed(0)}
                        </div>
                        <div className="text-[10px] text-[rgba(250,249,245,0.45)] font-mono mt-0.5">6-Hour Horizon</div>
                      </div>
                      <div className="p-3.5 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.08)]">
                        <div className="text-[10px] uppercase tracking-wider text-[rgba(250,249,245,0.4)] font-mono">Sokal Autocorrelation</div>
                        <div className="text-lg font-mono font-medium text-amber-300 mt-1">
                          1.00 sweep
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* STATION 13 WIDGET: CODON OPTIMIZATION VIA POTTS EBM */}
                {station.id === 13 && (
                  <div className="w-full flex flex-col gap-6">
                    {/* Top Sliders & Control Preset Bar */}
                    <div className="flex flex-wrap items-center justify-between gap-4 p-4 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.08)]">
                      {/* GC Balance Slider */}
                      <div className="flex flex-col gap-1.5 flex-1 min-w-[200px]">
                        <div className="flex justify-between items-center text-xs font-mono">
                          <span className="text-[rgba(250,249,245,0.7)] uppercase tracking-wider">
                            GC Homeostasis Penalty λ_gc:
                          </span>
                          <span className="text-[#5eead4]">
                            λ = {codonLambdaGc.toFixed(1)} (Target: 50% GC)
                          </span>
                        </div>
                        <input
                          type="range"
                          min={10.0}
                          max={100.0}
                          step={5.0}
                          value={codonLambdaGc}
                          onChange={(e) => setCodonLambdaGc(parseFloat(e.target.value))}
                          className="w-full accent-[#5eead4] cursor-pointer h-1.5 rounded-lg bg-[#020308]"
                        />
                      </div>

                      {/* Hairpin Penalty Slider */}
                      <div className="flex flex-col gap-1.5 flex-1 min-w-[200px]">
                        <div className="flex justify-between items-center text-xs font-mono">
                          <span className="text-[rgba(250,249,245,0.7)] uppercase tracking-wider">
                            Ribosome Hairpin Penalty λ_hp:
                          </span>
                          <span className="text-rose-400">
                            λ = {codonLambdaHp.toFixed(1)} / stem
                          </span>
                        </div>
                        <input
                          type="range"
                          min={0.5}
                          max={8.0}
                          step={0.5}
                          value={codonLambdaHp}
                          onChange={(e) => setCodonLambdaHp(parseFloat(e.target.value))}
                          className="w-full accent-rose-400 cursor-pointer h-1.5 rounded-lg bg-[#020308]"
                        />
                      </div>

                      {/* View Preset Buttons */}
                      <div className="flex flex-wrap items-center gap-2 font-mono text-[11px]">
                        <button
                          onClick={() => {
                            playArcadeTone(580, 40);
                            setActiveCodonSequence("thermo");
                          }}
                          className={`px-3 py-1.5 rounded-lg border transition-colors ${
                            activeCodonSequence === "thermo"
                              ? "bg-[#020308] border-[#34d399] text-[#34d399]"
                              : "bg-[#020308] border-[rgba(250,249,245,0.1)] text-[rgba(250,249,245,0.55)] hover:text-[#faf9f5]"
                          }`}
                        >
                          Potts Optimum (0 Stems)
                        </button>
                        <button
                          onClick={() => {
                            playArcadeTone(320, 40, "sawtooth");
                            setActiveCodonSequence("greedy");
                          }}
                          className={`px-3 py-1.5 rounded-lg border transition-colors ${
                            activeCodonSequence === "greedy"
                              ? "bg-[#020308] border-rose-400 text-rose-300"
                              : "bg-[#020308] border-[rgba(250,249,245,0.1)] text-[rgba(250,249,245,0.55)] hover:text-[#faf9f5]"
                          }`}
                        >
                          Greedy CAI Baseline (4 Stems)
                        </button>
                        <button
                          onClick={() => {
                            playArcadeTone(440, 30);
                            setActiveCodonSequence("comparison");
                          }}
                          className={`px-3 py-1.5 rounded-lg border transition-colors ${
                            activeCodonSequence === "comparison"
                              ? "bg-[#020308] border-[#5eead4] text-[#5eead4]"
                              : "bg-[#020308] border-[rgba(250,249,245,0.1)] text-[rgba(250,249,245,0.55)] hover:text-[#faf9f5]"
                          }`}
                        >
                          Dual Sequence Alignment
                        </button>
                        <button
                          onClick={() => {
                            playArcadeTone(400, 25);
                            setCodonLambdaGc(40.0);
                            setCodonLambdaHp(2.0);
                          }}
                          className="px-2.5 py-1.5 rounded-lg text-xs font-mono text-[rgba(250,249,245,0.45)] hover:text-[#faf9f5] bg-[#020308] border border-[rgba(250,249,245,0.08)] transition-colors"
                          title="Reset penalties to baseline"
                        >
                          ↺
                        </button>
                      </div>
                    </div>

                    {/* Pareto Multi-Objective Comparison Cards */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      {/* Classical Greedy Card */}
                      <div className="p-5 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.08)] space-y-3">
                        <div className="flex items-center justify-between">
                          <span className="text-[10px] font-mono uppercase tracking-[0.18em] text-rose-400">
                            Single-Objective Baseline
                          </span>
                          <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-rose-500/10 text-rose-300 border border-rose-500/30">
                            Greedy CAI
                          </span>
                        </div>
                        <div className="text-sm font-medium text-[#faf9f5]">
                          Human GLP-1 (15-mer mRNA)
                        </div>

                        <div className="space-y-2 text-xs font-mono">
                          <div className="flex justify-between py-1 border-b border-[rgba(250,249,245,0.06)]">
                            <span className="text-[rgba(250,249,245,0.45)]">Codon Adaptation (CAI):</span>
                            <span className="text-[#faf9f5] font-medium">{codonGreedyMetrics.cai.toFixed(4)} (100%)</span>
                          </div>
                          <div className="flex justify-between py-1 border-b border-[rgba(250,249,245,0.06)]">
                            <span className="text-[rgba(250,249,245,0.45)]">GC Equilibrium:</span>
                            <span className="text-amber-300 font-medium">{(codonGreedyMetrics.gc * 100).toFixed(1)}% (+16.7% bias)</span>
                          </div>
                          <div className="flex justify-between py-1 border-b border-[rgba(250,249,245,0.06)]">
                            <span className="text-[rgba(250,249,245,0.45)]">Ribosome Hairpins:</span>
                            <span className="text-rose-400 font-medium">{codonGreedyMetrics.hairpins} Stem-Loops (Stall Risk)</span>
                          </div>
                          <div className="flex justify-between py-1 border-b border-[rgba(250,249,245,0.06)]">
                            <span className="text-[rgba(250,249,245,0.45)]">Hamiltonian Energy H:</span>
                            <span className="text-rose-300 font-medium">{codonGreedyMetrics.totalEnergy.toFixed(2)}</span>
                          </div>
                        </div>

                        <div className="p-3 rounded-lg bg-[#020308] border border-[rgba(250,249,245,0.08)] text-[11px] text-[rgba(250,249,245,0.6)] font-light leading-relaxed">
                          <strong className="text-rose-300 font-mono font-medium">Failure Mode:</strong> Greedy frequency selection accumulates 4 inverted-repeat Watson-Crick stems that induce ribosomal pausing.
                        </div>
                      </div>

                      {/* Thermodynamic Potts Card */}
                      <div className="p-5 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.12)] space-y-3">
                        <div className="flex items-center justify-between">
                          <span className="text-[10px] font-mono uppercase tracking-[0.18em] text-[#34d399]">
                            Multi-Objective EBM
                          </span>
                          <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-[#34d399]/10 text-[#34d399] border border-[#34d399]/30">
                            Potts Optimizer
                          </span>
                        </div>
                        <div className="text-sm font-medium text-[#faf9f5]">
                          GLP-1 Thermodynamic Optima
                        </div>

                        <div className="space-y-2 text-xs font-mono">
                          <div className="flex justify-between py-1 border-b border-[rgba(250,249,245,0.06)]">
                            <span className="text-[rgba(250,249,245,0.45)]">Codon Adaptation (CAI):</span>
                            <span className="text-[#34d399] font-medium">{codonThermoMetrics.cai.toFixed(4)} (93.3%)</span>
                          </div>
                          <div className="flex justify-between py-1 border-b border-[rgba(250,249,245,0.06)]">
                            <span className="text-[rgba(250,249,245,0.45)]">GC Equilibrium:</span>
                            <span className="text-[#34d399] font-medium">{(codonThermoMetrics.gc * 100).toFixed(1)}% (Target ~50%)</span>
                          </div>
                          <div className="flex justify-between py-1 border-b border-[rgba(250,249,245,0.06)]">
                            <span className="text-[rgba(250,249,245,0.45)]">Ribosome Hairpins:</span>
                            <span className="text-[#34d399] font-medium">{codonThermoMetrics.hairpins} Loops (100% Cleared)</span>
                          </div>
                          <div className="flex justify-between py-1 border-b border-[rgba(250,249,245,0.06)]">
                            <span className="text-[rgba(250,249,245,0.45)]">Hamiltonian Energy H:</span>
                            <span className="text-[#34d399] font-medium">{codonThermoMetrics.totalEnergy.toFixed(2)}</span>
                          </div>
                        </div>

                        <div className="p-3 rounded-lg bg-[#020308] border border-[rgba(250,249,245,0.08)] text-[11px] text-[rgba(250,249,245,0.6)] font-light leading-relaxed">
                          <strong className="text-[#34d399] font-mono font-medium">Thermodynamic State:</strong> Accepting a minor 6.7% CAI trade-off across 4 positions dissolves all 4 secondary structure obstacles.
                        </div>
                      </div>

                      {/* Energy Advantage & Hardware Telemetry Card */}
                      <div className="p-5 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.08)] space-y-3">
                        <div className="flex items-center justify-between">
                          <span className="text-[10px] font-mono uppercase tracking-[0.18em] text-[#5eead4]">
                            Pareto Advantage
                          </span>
                          <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-[#5eead4]/10 text-[#5eead4] border border-[#5eead4]/30">
                            ΔH = -{codonEnergyAdvantage.toFixed(2)}
                          </span>
                        </div>
                        <div className="text-sm font-medium text-[#faf9f5]">
                          Physical Hardware Projection
                        </div>

                        <div className="space-y-2 text-xs font-mono">
                          <div className="flex justify-between py-1 border-b border-[rgba(250,249,245,0.06)]">
                            <span className="text-[rgba(250,249,245,0.45)]">Search Space Size:</span>
                            <span className="text-[#5eead4] font-medium">8.49 × 10⁷ candidates</span>
                          </div>
                          <div className="flex justify-between py-1 border-b border-[rgba(250,249,245,0.06)]">
                            <span className="text-[rgba(250,249,245,0.45)]">Measured JAX Runtime:</span>
                            <span className="text-[#faf9f5] font-medium">0.24s (Local GPU/CPU)</span>
                          </div>
                          <div className="flex justify-between py-1 border-b border-[rgba(250,249,245,0.06)]">
                            <span className="text-[rgba(250,249,245,0.45)]">Convergence Rate:</span>
                            <span className="text-[#34d399] font-medium">15,000 steps (τ_int = 1.0)</span>
                          </div>
                          <div className="flex justify-between py-1 border-b border-[rgba(250,249,245,0.06)]">
                            <span className="text-[rgba(250,249,245,0.45)]">Modeled TSU Latency:</span>
                            <span className="text-[rgba(250,249,245,0.65)] font-medium">0.005s (analytical projection)</span>
                          </div>
                        </div>

                        <div className="p-3 rounded-lg bg-[#020308] border border-[rgba(250,249,245,0.08)] text-[11px] text-[rgba(250,249,245,0.6)] font-light leading-relaxed">
                          <strong className="text-[#5eead4] font-mono font-medium">Hardware Mapping:</strong> Potts spins map to categorical p-bits; simultaneous analog sampling allows multi-objective mRNA co-folding.
                        </div>
                      </div>
                    </div>

                    {/* Interactive 15-Residue Codon Alignment & Ribosome Obstacle Viewer */}
                    <div className="p-4 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.08)] space-y-4">
                      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2 text-xs border-b border-[rgba(250,249,245,0.08)] pb-3">
                        <div>
                          <span className="font-medium text-[#faf9f5]">
                            Human GLP-1 Bioactive Peptide (15-mer, 45 Nucleotides)
                          </span>
                          <span className="text-[rgba(250,249,245,0.45)] ml-2 font-mono text-[11px]">
                            {activeCodonSequence === "greedy" ? "Displaying Greedy CAI Track" : activeCodonSequence === "thermo" ? "Displaying Thermodynamic Potts Track" : "Side-by-Side Synonymous Codon Alignment"}
                          </span>
                        </div>
                        <div className="flex items-center gap-3 text-[11px] font-mono">
                          <span className="flex items-center gap-1.5 text-[#34d399]">
                            <span className="w-1.5 h-1.5 rounded-full bg-[#34d399]" /> Potts Synonymous Swap
                          </span>
                          <span className="flex items-center gap-1.5 text-rose-400">
                            <span className="w-1.5 h-1.5 rounded-full bg-rose-400" /> Stalling Stem Residue
                          </span>
                        </div>
                      </div>

                      {/* 15 Codon Triplets Grid */}
                      <div className="grid grid-cols-3 sm:grid-cols-5 md:grid-cols-8 lg:grid-cols-15 gap-2">
                        {GLP1_CODON_DATA.map((res) => {
                          const isChanged = res.changed;
                          const showGreedy = activeCodonSequence === "greedy";
                          const currentCodon = showGreedy ? res.greedy : res.thermo;
                          const currentW = showGreedy ? res.wGreedy : res.wThermo;

                          return (
                            <div
                              key={res.idx}
                              className={`p-2 rounded-xl border text-center transition-colors ${
                                isChanged
                                  ? showGreedy
                                    ? "bg-[rgba(244,63,94,0.08)] border-rose-500/40 text-rose-300"
                                    : "bg-[rgba(52,211,153,0.1)] border-[#34d399]/40 text-[#34d399]"
                                  : "bg-[#020308] border-[rgba(250,249,245,0.08)] text-[rgba(250,249,245,0.65)]"
                              }`}
                            >
                              <div className="text-[9px] text-[rgba(250,249,245,0.4)] font-mono">
                                #{res.idx}
                              </div>
                              <div className="text-base font-medium text-[#faf9f5] font-mono my-0.5">
                                {res.aa}
                              </div>
                              <div className="text-[9px] text-[rgba(250,249,245,0.45)] font-mono">
                                {res.name}
                              </div>

                              {activeCodonSequence === "comparison" ? (
                                <div className="mt-2 pt-1 border-t border-[rgba(250,249,245,0.08)] space-y-1">
                                  <div className="text-[10px] font-mono font-medium text-rose-300 bg-[rgba(244,63,94,0.12)] rounded px-1">
                                    {res.greedy}
                                  </div>
                                  <div className="text-[10px] font-mono font-medium text-[#34d399] bg-[rgba(52,211,153,0.12)] rounded px-1">
                                    {res.thermo}
                                  </div>
                                </div>
                              ) : (
                                <div className="mt-2 pt-1 border-t border-[rgba(250,249,245,0.08)]">
                                  <div className="text-xs font-mono font-medium">
                                    {currentCodon}
                                  </div>
                                  <div className="text-[9px] text-[rgba(250,249,245,0.4)] font-mono mt-0.5">
                                    w={currentW.toFixed(2)}
                                  </div>
                                </div>
                              )}

                              {isChanged && (
                                <div className="mt-1 text-[8px] font-mono leading-tight px-1 py-0.5 rounded bg-[#34d399]/15 text-[#34d399] truncate">
                                  {res.note}
                                </div>
                              )}
                            </div>
                          );
                        })}
                      </div>

                      {/* 4 Inverted-Repeat Hairpin Detector Table */}
                      <div className="pt-2">
                        <div className="text-xs font-mono text-[rgba(250,249,245,0.65)] mb-2 flex flex-wrap items-center justify-between gap-2">
                          <span className="uppercase tracking-wider">Watson-Crick Stem-Loop Hairpin Obstacle Detector (L_stem ≥ 3, Loop 3 to 8 nt)</span>
                          <span className="text-[10px] text-[#5eead4]">
                            Zero-Dependency Structural Classifier
                          </span>
                        </div>

                        <div className="overflow-x-auto">
                          <table className="w-full text-left border-collapse text-xs">
                            <thead>
                              <tr className="border-b border-[rgba(250,249,245,0.08)] text-[rgba(250,249,245,0.45)] font-mono text-[11px]">
                                <th className="py-2 px-2">Hairpin #</th>
                                <th className="py-2 px-2">5&apos; Stem</th>
                                <th className="py-2 px-2">Loop Region</th>
                                <th className="py-2 px-2">3&apos; Rev-Comp Stem</th>
                                <th className="py-2 px-2">Residue Span</th>
                                <th className="py-2 px-2">Classical Greedy</th>
                                <th className="py-2 px-2">Thermodynamic Potts</th>
                              </tr>
                            </thead>
                            <tbody>
                              {GLP1_HAIRPINS.map((hp) => (
                                <tr key={hp.id} className="border-b border-[rgba(250,249,245,0.04)] hover:bg-[#020308] font-mono">
                                  <td className="py-2 px-2 text-[#faf9f5]">Stem {hp.id}</td>
                                  <td className="py-2 px-2 text-rose-300">{hp.stem1}</td>
                                  <td className="py-2 px-2 text-[rgba(250,249,245,0.5)] text-[11px]">{hp.loop}</td>
                                  <td className="py-2 px-2 text-rose-300">{hp.stem2}</td>
                                  <td className="py-2 px-2 text-[rgba(250,249,245,0.65)] text-[11px]">{hp.residues}</td>
                                  <td className="py-2 px-2">
                                    <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-rose-500/10 text-rose-300 border border-rose-500/30">
                                      {hp.status}
                                    </span>
                                  </td>
                                  <td className="py-2 px-2">
                                    <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-[#34d399]/10 text-[#34d399] border border-[#34d399]/30">
                                      ✓ {hp.remedy}
                                    </span>
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* STATION 14 WIDGET: PROTEIN SIDE-CHAIN ROTAMER PACKING VIA POTTS EBM */}
                {station.id === 14 && (
                  <div className="w-full flex flex-col gap-6">
                    {/* Top Controls & View Presets */}
                    <div className="flex flex-wrap items-center justify-between gap-4 p-4 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.08)]">
                      {/* View Preset Buttons */}
                      <div className="flex flex-wrap items-center gap-2 font-mono text-[11px]">
                        <button
                          onClick={() => {
                            playArcadeTone(620, 40);
                            setRotamerViewMode("thermo");
                          }}
                          className={`px-3 py-1.5 rounded-lg border transition-colors ${
                            rotamerViewMode === "thermo"
                              ? "bg-[#020308] border-[#34d399] text-[#34d399]"
                              : "bg-[#020308] border-[rgba(250,249,245,0.1)] text-[rgba(250,249,245,0.55)] hover:text-[#faf9f5]"
                          }`}
                        >
                          Thermodynamic Fold (0 Clashes)
                        </button>
                        <button
                          onClick={() => {
                            playArcadeTone(300, 45, "sawtooth");
                            setRotamerViewMode("greedy");
                          }}
                          className={`px-3 py-1.5 rounded-lg border transition-colors ${
                            rotamerViewMode === "greedy"
                              ? "bg-[#020308] border-rose-400 text-rose-300"
                              : "bg-[#020308] border-[rgba(250,249,245,0.1)] text-[rgba(250,249,245,0.55)] hover:text-[#faf9f5]"
                          }`}
                        >
                          Greedy Baseline (6 Clashes)
                        </button>
                        <button
                          onClick={() => {
                            playArcadeTone(480, 30);
                            setRotamerViewMode("radar");
                          }}
                          className={`px-3 py-1.5 rounded-lg border transition-colors ${
                            rotamerViewMode === "radar"
                              ? "bg-[#020308] border-[#5eead4] text-[#5eead4]"
                              : "bg-[#020308] border-[rgba(250,249,245,0.1)] text-[rgba(250,249,245,0.55)] hover:text-[#faf9f5]"
                          }`}
                        >
                          Interface Contact Map
                        </button>
                      </div>

                      {/* Clash Distance Threshold Slider */}
                      <div className="flex items-center gap-3 flex-1 min-w-[220px]">
                        <span className="text-xs font-mono text-[rgba(250,249,245,0.7)] whitespace-nowrap uppercase tracking-wider">
                          Clash Threshold:
                        </span>
                        <input
                          type="range"
                          min={1.5}
                          max={2.5}
                          step={0.1}
                          value={rotamerClashCutoff}
                          onChange={(e) => setRotamerClashCutoff(parseFloat(e.target.value))}
                          className="w-full accent-rose-400 cursor-pointer h-1.5 rounded-lg bg-[#020308]"
                        />
                        <span className="text-xs font-mono text-rose-400 whitespace-nowrap">
                          r &lt; {rotamerClashCutoff.toFixed(1)} Å
                        </span>
                      </div>
                    </div>

                    {/* Multi-Objective Pareto Comparison HUD */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      {/* Classical Greedy Card */}
                      <div className="p-5 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.08)] space-y-3">
                        <div className="flex items-center justify-between">
                          <span className="text-[10px] font-mono uppercase tracking-[0.18em] text-rose-400">
                            Single-Objective Baseline
                          </span>
                          <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-rose-500/10 text-rose-300 border border-rose-500/30">
                            Greedy Dunbrack
                          </span>
                        </div>
                        <div className="text-sm font-medium text-[#faf9f5]">
                          24-mer Beta-Hairpin (k=0 Min Strain)
                        </div>

                        <div className="space-y-2 text-xs font-mono">
                          <div className="flex justify-between py-1 border-b border-[rgba(250,249,245,0.06)]">
                            <span className="text-[rgba(250,249,245,0.45)]">Total Hamiltonian Energy:</span>
                            <span className="text-rose-400 font-medium">309.11 kcal/mol</span>
                          </div>
                          <div className="flex justify-between py-1 border-b border-[rgba(250,249,245,0.06)]">
                            <span className="text-[rgba(250,249,245,0.45)]">Dunbrack Self-Energy:</span>
                            <span className="text-[#faf9f5] font-medium">3.30 kcal/mol (0.14/res)</span>
                          </div>
                          <div className="flex justify-between py-1 border-b border-[rgba(250,249,245,0.06)]">
                            <span className="text-[rgba(250,249,245,0.45)]">Lennard-Jones Repulsion:</span>
                            <span className="text-rose-400 font-medium">+294.97 kcal/mol</span>
                          </div>
                          <div className="flex justify-between py-1 border-b border-[rgba(250,249,245,0.06)]">
                            <span className="text-[rgba(250,249,245,0.45)]">Steric Clash Count:</span>
                            <span className="text-rose-400 font-medium">6 Severe Clashes (min 0.50 Å)</span>
                          </div>
                        </div>

                        <div className="p-3 rounded-lg bg-[#020308] border border-[rgba(250,249,245,0.08)] text-[11px] text-[rgba(250,249,245,0.6)] font-light leading-relaxed">
                          <strong className="text-rose-300 font-mono font-medium">Steric Trap:</strong> Choosing minimum internal torsional strain forces facing bulky residues across the 4.8 Å beta-hairpin interface into direct atomic collision.
                        </div>
                      </div>

                      {/* Thermodynamic Potts Card */}
                      <div className="p-5 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.12)] space-y-3">
                        <div className="flex items-center justify-between">
                          <span className="text-[10px] font-mono uppercase tracking-[0.18em] text-[#34d399]">
                            Multi-Objective Potts EBM
                          </span>
                          <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-[#34d399]/10 text-[#34d399] border border-[#34d399]/30">
                            Potts Ground State
                          </span>
                        </div>
                        <div className="text-sm font-medium text-[#faf9f5]">
                          Relaxed Core Conformation
                        </div>

                        <div className="space-y-2 text-xs font-mono">
                          <div className="flex justify-between py-1 border-b border-[rgba(250,249,245,0.06)]">
                            <span className="text-[rgba(250,249,245,0.45)]">Total Hamiltonian Energy:</span>
                            <span className="text-[#34d399] font-medium">13.57 kcal/mol (ΔE = -295.54)</span>
                          </div>
                          <div className="flex justify-between py-1 border-b border-[rgba(250,249,245,0.06)]">
                            <span className="text-[rgba(250,249,245,0.45)]">Compensating Torsion:</span>
                            <span className="text-[#faf9f5] font-medium">12.80 kcal/mol (+9.50 shift)</span>
                          </div>
                          <div className="flex justify-between py-1 border-b border-[rgba(250,249,245,0.06)]">
                            <span className="text-[rgba(250,249,245,0.45)]">Lennard-Jones Dispersion:</span>
                            <span className="text-[#34d399] font-medium">-8.74 kcal/mol (Attractive Well)</span>
                          </div>
                          <div className="flex justify-between py-1 border-b border-[rgba(250,249,245,0.06)]">
                            <span className="text-[rgba(250,249,245,0.45)]">Steric Clash Count:</span>
                            <span className="text-[#34d399] font-medium">0 Clashes (min 3.68 Å)</span>
                          </div>
                        </div>

                        <div className="p-3 rounded-lg bg-[#020308] border border-[rgba(250,249,245,0.08)] text-[11px] text-[rgba(250,249,245,0.6)] font-light leading-relaxed">
                          <strong className="text-[#34d399] font-mono font-medium">Ground State:</strong> Accepting a slight internal strain penalty allows side-chains to interdigitate laterally into the attractive van der Waals well.
                        </div>
                      </div>

                      {/* Hardware Telemetry Card */}
                      <div className="p-5 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.08)] space-y-3">
                        <div className="flex items-center justify-between">
                          <span className="text-[10px] font-mono uppercase tracking-[0.18em] text-[#5eead4]">
                            Hardware Telemetry
                          </span>
                          <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-[#5eead4]/10 text-[#5eead4] border border-[#5eead4]/30">
                            Modeled TSU
                          </span>
                        </div>
                        <div className="text-sm font-medium text-[#faf9f5]">
                          Measured Simulation vs Analytical Model
                        </div>

                        <div className="space-y-2 text-xs font-mono">
                          <div className="flex justify-between py-1 border-b border-[rgba(250,249,245,0.06)]">
                            <span className="text-[rgba(250,249,245,0.45)]">Measured Classical Rate:</span>
                            <span className="text-[#34d399] font-medium">384.5 ESS/s (Local JAX Benchmark)</span>
                          </div>
                          <div className="flex justify-between py-1 border-b border-[rgba(250,249,245,0.06)]">
                            <span className="text-[rgba(250,249,245,0.45)]">Extropic Model Projection:</span>
                            <span className="text-[rgba(250,249,245,0.65)] font-medium">520,833 ESS/s (Analytical Z1, 1.0W)</span>
                          </div>
                        </div>

                        <div className="p-3 rounded-lg bg-[#020308] border border-[rgba(250,249,245,0.08)] text-[11px] text-[rgba(250,249,245,0.6)] font-light leading-relaxed">
                          <strong className="text-[#5eead4] font-mono font-medium">Hardware Note:</strong> The 384.5 ESS/s figure is measured on local hardware. The 520,833 ESS/s rate is an analytical device projection; physical Extropic TSU silicon is not publicly available.
                        </div>
                      </div>
                    </div>

                    {/* Interactive 24-Residue Antiparallel Beta-Hairpin Visualizer */}
                    <div className="p-4 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.08)] space-y-4">
                      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2 text-xs border-b border-[rgba(250,249,245,0.08)] pb-3">
                        <div>
                          <span className="font-medium text-[#faf9f5]">
                            24-Residue Synthetic Beta-Hairpin Fold (4.8 Å Inter-Strand Spacing)
                          </span>
                          <span className="text-[rgba(250,249,245,0.45)] ml-2 font-mono text-[11px]">
                            {rotamerViewMode === "greedy"
                              ? "Displaying Greedy Dunbrack Collision Conformation"
                              : "Displaying Thermodynamic Potts Interdigitated Conformation"}
                          </span>
                        </div>
                        <div className="flex items-center gap-3 text-[11px] font-mono">
                          <span className="flex items-center gap-1.5 text-rose-400">
                            <span className="w-1.5 h-1.5 rounded-full bg-rose-400" /> 6 Steric Clashes (&lt; 2.0 Å)
                          </span>
                          <span className="flex items-center gap-1.5 text-[#34d399]">
                            <span className="w-1.5 h-1.5 rounded-full bg-[#34d399]" /> Optimal van der Waals (&gt; 3.6 Å)
                          </span>
                        </div>
                      </div>

                      {/* Beta-Hairpin Spatial Alignment View */}
                      <div className="p-4 rounded-xl bg-[#020308] border border-[rgba(250,249,245,0.08)] space-y-4 overflow-x-auto">
                        <div className="min-w-[700px] space-y-3">
                          {/* Strand 1 (Residues 1 to 11) */}
                          <div className="flex items-center gap-2">
                            <span className="text-[10px] font-mono uppercase tracking-wider text-[rgba(250,249,245,0.4)] w-20 whitespace-nowrap">
                              Strand 1 (N→C):
                            </span>
                            <div className="flex-1 grid grid-cols-11 gap-2">
                              {HAIRPIN_STRAND1.map((r) => {
                                const isFacing = r.pairId !== null;
                                const isClashing = rotamerViewMode === "greedy" && isFacing;

                                return (
                                  <button
                                    key={r.idx}
                                    onClick={() => {
                                      playArcadeTone(420, 30);
                                      if (r.pairId) setSelectedFacingPair(r.pairId);
                                    }}
                                    className={`p-2 rounded-lg border text-center transition-colors ${
                                      isClashing
                                        ? "bg-[rgba(244,63,94,0.1)] border-rose-500/50 text-rose-200"
                                        : isFacing
                                        ? "bg-[rgba(52,211,153,0.08)] border-[#34d399]/40 text-[#34d399]"
                                        : "bg-[#05070f] border-[rgba(250,249,245,0.08)] text-[rgba(250,249,245,0.4)]"
                                    }`}
                                  >
                                    <div className="text-[9px] font-mono text-[rgba(250,249,245,0.4)]">#{r.idx}</div>
                                    <div className="text-xs font-medium text-[#faf9f5] font-mono">{r.name}</div>
                                    <div className="text-[9px] font-mono mt-0.5 text-[rgba(250,249,245,0.4)]">
                                      k={rotamerViewMode === "greedy" ? r.kGreedy : r.kThermo}
                                    </div>
                                  </button>
                                );
                              })}
                            </div>
                          </div>

                          {/* Inter-Strand 4.8 Å Interface Clash Beams */}
                          <div className="flex items-center gap-2 py-1">
                            <span className="text-[10px] font-mono text-[rgba(250,249,245,0.35)] w-20 whitespace-nowrap">
                              4.8 Å Core:
                            </span>
                            <div className="flex-1 grid grid-cols-11 gap-2 text-center font-mono text-[10px]">
                              {HAIRPIN_STRAND1.map((r) => {
                                if (r.pairId === null) {
                                  return (
                                    <div key={r.idx} className="text-[rgba(250,249,245,0.15)]">
                                      ···
                                    </div>
                                  );
                                }
                                const pair = SIDECHAIN_FACING_PAIRS[r.pairId - 1];
                                const isClash = rotamerViewMode === "greedy";

                                return (
                                  <div
                                    key={r.idx}
                                    className={`py-1 rounded font-medium border transition-colors ${
                                      isClash
                                        ? "bg-[rgba(244,63,94,0.1)] border-rose-500/40 text-rose-300"
                                        : "bg-[rgba(52,211,153,0.1)] border-[#34d399]/40 text-[#34d399]"
                                    }`}
                                  >
                                    {isClash ? `${pair.greedyDist.toFixed(2)} Å` : `${pair.thermoDist.toFixed(2)} Å`}
                                  </div>
                                );
                              })}
                            </div>
                          </div>

                          {/* Strand 2 (Residues 24 down to 14) */}
                          <div className="flex items-center gap-2">
                            <span className="text-[10px] font-mono uppercase tracking-wider text-[rgba(250,249,245,0.4)] w-20 whitespace-nowrap">
                              Strand 2 (C←N):
                            </span>
                            <div className="flex-1 grid grid-cols-11 gap-2">
                              {HAIRPIN_STRAND2.map((r) => {
                                const isFacing = r.pairId !== null;
                                const isClashing = rotamerViewMode === "greedy" && isFacing;

                                return (
                                  <button
                                    key={r.idx}
                                    onClick={() => {
                                      playArcadeTone(420, 30);
                                      if (r.pairId) setSelectedFacingPair(r.pairId);
                                    }}
                                    className={`p-2 rounded-lg border text-center transition-colors ${
                                      isClashing
                                        ? "bg-[rgba(244,63,94,0.1)] border-rose-500/50 text-rose-200"
                                        : isFacing
                                        ? "bg-[rgba(52,211,153,0.08)] border-[#34d399]/40 text-[#34d399]"
                                        : "bg-[#05070f] border-[rgba(250,249,245,0.08)] text-[rgba(250,249,245,0.4)]"
                                    }`}
                                  >
                                    <div className="text-[9px] font-mono text-[rgba(250,249,245,0.4)]">#{r.idx}</div>
                                    <div className="text-xs font-medium text-[#faf9f5] font-mono">{r.name}</div>
                                    <div className="text-[9px] font-mono mt-0.5 text-[rgba(250,249,245,0.4)]">
                                      k={rotamerViewMode === "greedy" ? r.kGreedy : r.kThermo}
                                    </div>
                                  </button>
                                );
                              })}
                            </div>
                          </div>
                        </div>
                      </div>

                      {/* Side-Chain Collision Radar Table */}
                      <div className="pt-2">
                        <div className="text-xs font-mono text-[rgba(250,249,245,0.65)] mb-2 flex flex-wrap items-center justify-between gap-2">
                          <span className="uppercase tracking-wider">Side-Chain Collision Radar: 6 Inter-Strand Facing Pairs</span>
                          <span className="text-[10px] text-[#5eead4]">
                            Truncated Lennard-Jones (12-6) + Electrostatics
                          </span>
                        </div>

                        <div className="overflow-x-auto">
                          <table className="w-full text-left border-collapse text-xs">
                            <thead>
                              <tr className="border-b border-[rgba(250,249,245,0.08)] text-[rgba(250,249,245,0.45)] font-mono text-[11px]">
                                <th className="py-2 px-2">Pair #</th>
                                <th className="py-2 px-2">Strand 1 Residue</th>
                                <th className="py-2 px-2">Strand 2 Partner</th>
                                <th className="py-2 px-2">Greedy Distance</th>
                                <th className="py-2 px-2">Greedy V_LJ</th>
                                <th className="py-2 px-2">Thermo Rotamers</th>
                                <th className="py-2 px-2">Relaxed Distance</th>
                                <th className="py-2 px-2">Relaxed V_LJ</th>
                                <th className="py-2 px-2">Resolution Mechanism</th>
                              </tr>
                            </thead>
                            <tbody>
                              {SIDECHAIN_FACING_PAIRS.map((pair) => (
                                <tr
                                  key={pair.id}
                                  className={`border-b border-[rgba(250,249,245,0.04)] hover:bg-[#020308] font-mono ${
                                    selectedFacingPair === pair.id ? "bg-[#020308]" : ""
                                  }`}
                                >
                                  <td className="py-2 px-2 text-[#faf9f5]">Pair {pair.id}</td>
                                  <td className="py-2 px-2 text-amber-300">{pair.resA}</td>
                                  <td className="py-2 px-2 text-[#5eead4]">{pair.resB}</td>
                                  <td className="py-2 px-2 text-rose-400">{pair.greedyDist.toFixed(2)} Å</td>
                                  <td className="py-2 px-2 text-rose-400">+{pair.greedyLj.toFixed(1)}</td>
                                  <td className="py-2 px-2 text-[#34d399] text-[11px]">
                                    {pair.thermoRotA} / {pair.thermoRotB}
                                  </td>
                                  <td className="py-2 px-2 text-[#34d399]">{pair.thermoDist.toFixed(2)} Å</td>
                                  <td className="py-2 px-2 text-[#34d399]">{pair.thermoLj.toFixed(2)}</td>
                                  <td className="py-2 px-2 text-[rgba(250,249,245,0.65)] text-[11px] font-light">{pair.note}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* STATION 15 WIDGET: THERMODYNAMIC SI COGNITIVE CORE */}
                {station.id === 15 && (
                  <div className="w-full flex flex-col gap-6">
                    <div className="p-7 rounded-2xl bg-[#05070f] border border-[rgba(250,249,245,0.10)] flex flex-col md:flex-row items-center justify-between gap-6 shadow-2xl">
                      <div className="space-y-3">
                        <div className="flex flex-wrap items-center gap-2 font-mono text-[11px]">
                          <span className="px-3 py-0.5 rounded-full bg-[rgba(250,249,245,0.06)] text-[#5eead4] border border-[rgba(250,249,245,0.12)]">
                            Station 15 · Cybernetic Telemetry
                          </span>
                          <span className="px-3 py-0.5 rounded-full bg-[#34d399]/10 text-[#34d399] border border-[#34d399]/30">
                            &lt; 15ms Latency SLA Verified
                          </span>
                        </div>
                        <h3 className="font-[family-name:var(--font-display)] text-xl sm:text-2xl font-normal text-[#faf9f5] tracking-tight">
                          4-Organ Autonomous Cognitive Architecture
                        </h3>
                        <p className="text-sm text-[rgba(250,249,245,0.65)] font-light max-w-2xl leading-relaxed">
                          Continuous Hopfield memory basins relax in DDR5, Kramers Langevin curiosity oscillates between Exploit and Explore, a 6-node Potts network arbitrates goal consensus, and RTX 5080 VRAM scheduling governs metabolic limits.
                        </p>
                      </div>

                      <div className="flex shrink-0">
                        <Link
                          href="/si-core"
                          className="rounded-full bg-[#faf9f5] px-6 py-2.5 font-medium text-[#020308] hover:bg-white text-xs uppercase tracking-wider font-mono transition-colors"
                        >
                          Launch SI Core Visualizer ↗
                        </Link>
                      </div>
                    </div>

                    {/* 4 Organ Quick Cards */}
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                      <div className="p-4 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.08)]">
                        <span className="text-[10px] font-mono text-[#5eead4] uppercase tracking-wider">Organ 1</span>
                        <h4 className="text-sm font-medium text-[#faf9f5] mt-1 font-mono">Continuous Hopfield</h4>
                        <p className="text-xs text-[rgba(250,249,245,0.55)] font-light mt-1.5 leading-relaxed">Episodic associative recall in 128GB DDR5 RAM with exponential attractor capacity.</p>
                      </div>
                      <div className="p-4 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.08)]">
                        <span className="text-[10px] font-mono text-rose-400 uppercase tracking-wider">Organ 2</span>
                        <h4 className="text-sm font-medium text-[#faf9f5] mt-1 font-mono">Kramers Curiosity</h4>
                        <p className="text-xs text-[rgba(250,249,245,0.55)] font-light mt-1.5 leading-relaxed">Bistable double-well potential modulating thermal bath T and barrier escape dynamics.</p>
                      </div>
                      <div className="p-4 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.08)]">
                        <span className="text-[10px] font-mono text-purple-400 uppercase tracking-wider">Organ 3</span>
                        <h4 className="text-sm font-medium text-[#faf9f5] mt-1 font-mono">Potts Goal Arbiter</h4>
                        <p className="text-xs text-[rgba(250,249,245,0.55)] font-light mt-1.5 leading-relaxed">6-node categorical Potts spin network equilibrating competing agent drives in &lt;2ms.</p>
                      </div>
                      <div className="p-4 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.08)]">
                        <span className="text-[10px] font-mono text-[#34d399] uppercase tracking-wider">Organ 4</span>
                        <h4 className="text-sm font-medium text-[#faf9f5] mt-1 font-mono">RTX 5080 Metabolism</h4>
                        <p className="text-xs text-[rgba(250,249,245,0.55)] font-light mt-1.5 leading-relaxed">QUBO hardware governor balancing 16GB VRAM bounds, model latency, and capability.</p>
                      </div>
                    </div>
                  </div>
                )}

                {/* STATION 16 WIDGET: NON-EQUILIBRIUM HYBRID TPU & FRUSTRATION BREAKING */}
                {station.id === 16 && (
                  <div className="w-full flex flex-col gap-6">
                    {/* Header Banner & Sub-View Switcher */}
                    <div className="p-7 rounded-2xl bg-[#05070f] border border-[rgba(250,249,245,0.10)] flex flex-col gap-5 shadow-2xl">
                      <div className="space-y-2">
                        <div className="flex flex-wrap items-center gap-2 font-mono text-[11px]">
                          <span className="px-3 py-0.5 rounded-full bg-[rgba(250,249,245,0.06)] text-[#5eead4] border border-[rgba(250,249,245,0.12)]">
                            Station 16 · Non-Equilibrium Hybrid TPU
                          </span>
                          <span className="px-3 py-0.5 rounded-full bg-[#34d399]/10 text-[#34d399] border border-[#34d399]/30">
                            100% SAT Ground State (0 Violations)
                          </span>
                        </div>
                        <h3 className="font-[family-name:var(--font-display)] text-xl sm:text-2xl font-normal text-[#faf9f5] tracking-tight">
                          Underdamped BAOAB Langevin, Auxiliary Memory &amp; SMC Resampling
                        </h3>
                        <p className="text-xs text-[rgba(250,249,245,0.65)] font-light max-w-3xl leading-relaxed">
                          Pushing thermodynamic computing beyond conservative equilibrium: non-conservative auxiliary clause memory multipliers <code className="text-[#5eead4] font-mono">y_m(t)</code> warp energy barriers to break topological frustration, while 22-stage Jarzynski-Neal SMC population resampling cures exponential weight collapse.
                        </p>
                      </div>

                      {/* 4 Interactive View Switcher Tabs - Responsive Full Width Grid */}
                      <div className="grid grid-cols-2 lg:grid-cols-4 p-1 rounded-xl bg-[#020308] border border-[rgba(250,249,245,0.08)] gap-1 w-full font-mono text-[11px]">
                        <button
                          onClick={() => {
                            playArcadeTone(440, 25);
                            setTpuActiveTab("frustration");
                          }}
                          className={`w-full py-2 px-2.5 rounded-lg transition-colors text-center uppercase tracking-wider ${
                            tpuActiveTab === "frustration"
                              ? "bg-[#05070f] text-[#5eead4] border border-[rgba(250,249,245,0.15)] font-medium"
                              : "text-[rgba(250,249,245,0.45)] hover:text-[#faf9f5]"
                          }`}
                        >
                          1. SAT Frustration
                        </button>
                        <button
                          onClick={() => {
                            playArcadeTone(520, 25);
                            setTpuActiveTab("resampling");
                          }}
                          className={`w-full py-2 px-2.5 rounded-lg transition-colors text-center uppercase tracking-wider ${
                            tpuActiveTab === "resampling"
                              ? "bg-[#05070f] text-[#34d399] border border-[rgba(250,249,245,0.15)] font-medium"
                              : "text-[rgba(250,249,245,0.45)] hover:text-[#faf9f5]"
                          }`}
                        >
                          2. SMC Resampling (N=128)
                        </button>
                        <button
                          onClick={() => {
                            playArcadeTone(600, 25);
                            setTpuActiveTab("integrator");
                          }}
                          className={`w-full py-2 px-2.5 rounded-lg transition-colors text-center uppercase tracking-wider ${
                            tpuActiveTab === "integrator"
                              ? "bg-[#05070f] text-indigo-300 border border-[rgba(250,249,245,0.15)] font-medium"
                              : "text-[rgba(250,249,245,0.45)] hover:text-[#faf9f5]"
                          }`}
                        >
                          3. Integrator &amp; Thermometer
                        </button>
                        <button
                          onClick={() => {
                            playArcadeTone(680, 25);
                            setTpuActiveTab("silicon");
                          }}
                          className={`w-full py-2 px-2.5 rounded-lg transition-colors text-center uppercase tracking-wider ${
                            tpuActiveTab === "silicon"
                              ? "bg-[#05070f] text-violet-300 border border-[rgba(250,249,245,0.15)] font-medium"
                              : "text-[rgba(250,249,245,0.45)] hover:text-[#faf9f5]"
                          }`}
                        >
                          4. Silicon Realism (DAC / Rails)
                        </button>
                      </div>
                    </div>

                    {/* =========================================================
                        SUB-VIEW 1: 3-WAY SAT FRUSTRATION BREAKER (N=100)
                    ========================================================= */}
                    {tpuActiveTab === "frustration" && (
                      <div className="flex flex-col gap-5">
                        {/* 3-Way Architecture Comparison Cards */}
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                          {/* Card 1: Conservative Tanh */}
                          <div className="p-5 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.08)] flex flex-col justify-between space-y-3">
                            <div>
                              <div className="flex items-center justify-between">
                                <span className="text-[10px] font-mono text-rose-400 uppercase tracking-[0.18em]">
                                  Baseline A · Conservative
                                </span>
                                <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-rose-500/10 text-rose-300 border border-rose-500/30">
                                  Gradient Starved
                                </span>
                              </div>
                              <h4 className="text-sm font-medium text-[#faf9f5] mt-1 font-mono">Tanh Relaxation</h4>
                              <p className="text-xs text-[rgba(250,249,245,0.55)] font-light mt-1.5 leading-relaxed">
                                Saturating penalty <code className="text-rose-300 font-mono">tanh(β x)</code> freezes when oscillators reach wells: <code className="text-rose-300 font-mono">sech²(x) → 0</code> vanishes all restorative force.
                              </p>
                            </div>
                            <div className="space-y-1.5 pt-2 border-t border-[rgba(250,249,245,0.06)] text-xs font-mono">
                              <div className="flex justify-between">
                                <span className="text-[rgba(250,249,245,0.45)]">Violated Clauses:</span>
                                <span className="text-rose-400 font-medium">2 to 5 Unsatisfied</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-[rgba(250,249,245,0.45)]">Ground Energy:</span>
                                <span className="text-[rgba(250,249,245,0.65)]">E = 27.56</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-[rgba(250,249,245,0.45)]">Restorative Force:</span>
                                <span className="text-rose-400">0.000 (Frozen)</span>
                              </div>
                            </div>
                          </div>

                          {/* Card 2: Conservative Softplus */}
                          <div className="p-5 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.08)] flex flex-col justify-between space-y-3">
                            <div>
                              <div className="flex items-center justify-between">
                                <span className="text-[10px] font-mono text-amber-300 uppercase tracking-[0.18em]">
                                  Baseline B · Conservative
                                </span>
                                <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-amber-500/10 text-amber-300 border border-amber-500/30">
                                  Saddle Trapped
                                </span>
                              </div>
                              <h4 className="text-sm font-medium text-[#faf9f5] mt-1 font-mono">Smooth Softplus</h4>
                              <p className="text-xs text-[rgba(250,249,245,0.55)] font-light mt-1.5 leading-relaxed">
                                Non-saturating <code className="text-amber-300 font-mono">ln(1+e^u)</code> preserves gradient flow <code className="text-amber-300 font-mono">σ(u) &gt; 0</code>, but cancels out at symmetric saddle points (<code className="text-amber-300 font-mono">∑ ∇C_m = 0</code>).
                              </p>
                            </div>
                            <div className="space-y-1.5 pt-2 border-t border-[rgba(250,249,245,0.06)] text-xs font-mono">
                              <div className="flex justify-between">
                                <span className="text-[rgba(250,249,245,0.45)]">Violated Clauses:</span>
                                <span className="text-amber-400 font-medium">1 Unsatisfied</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-[rgba(250,249,245,0.45)]">Ground Energy:</span>
                                <span className="text-[rgba(250,249,245,0.65)]">E = 5.57</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-[rgba(250,249,245,0.45)]">Net Saddle Force:</span>
                                <span className="text-amber-400">0.000 (Balanced)</span>
                              </div>
                            </div>
                          </div>

                          {/* Card 3: Non-Conservative Auxiliary Softplus */}
                          <div className="p-5 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.12)] flex flex-col justify-between space-y-3">
                            <div>
                              <div className="flex items-center justify-between">
                                <span className="text-[10px] font-mono text-[#34d399] uppercase tracking-[0.18em]">
                                  Station 16 · Non-Conservative
                                </span>
                                <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-[#34d399]/10 text-[#34d399] border border-[#34d399]/30">
                                  100% SAT (0 Violations)
                                </span>
                              </div>
                              <h4 className="text-sm font-medium text-[#faf9f5] mt-1 font-mono">Auxiliary Memory Multipliers</h4>
                              <p className="text-xs text-[rgba(250,249,245,0.55)] font-light mt-1.5 leading-relaxed">
                                Persistent clause violations integrate dynamic weights <code className="text-[#34d399] font-mono">dy_m = (κ C_m - η y_m) dt</code>, tilting the barrier and ejecting the trapped oscillator.
                              </p>
                            </div>
                            <div className="space-y-1.5 pt-2 border-t border-[rgba(250,249,245,0.06)] text-xs font-mono">
                              <div className="flex justify-between">
                                <span className="text-[rgba(250,249,245,0.45)]">Violated Clauses:</span>
                                <span className="text-[#34d399] font-medium">0 (100% Solved)</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-[rgba(250,249,245,0.45)]">Ground Energy:</span>
                                <span className="text-[#34d399]">E = -1.83</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-[rgba(250,249,245,0.45)]">Housekeeping Heat:</span>
                                <span className="text-rose-400 font-medium">Q_hk = 3,697.40 dissipated</span>
                              </div>
                            </div>
                          </div>
                        </div>

                        {/* Interactive Barrier Warping & Oscillator Ejection Simulation */}
                        <div className="p-5 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.08)] space-y-4">
                          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3 border-b border-[rgba(250,249,245,0.08)] pb-3">
                            <div>
                              <span className="text-sm font-medium text-[#faf9f5]">
                                Dynamic Energy Barrier Warping &amp; Escape Simulation
                              </span>
                              <div className="text-xs text-[rgba(250,249,245,0.45)] font-mono mt-0.5">
                                Equation: <code className="text-[#5eead4]">V(x; y_m) = α(x² - 1)² + y_m(t) · ln(1 + e^(-3.5 x))</code>
                              </div>
                            </div>

                            {/* Controls */}
                            <div className="flex items-center gap-2 font-mono text-[11px]">
                              <button
                                onClick={() => {
                                  playArcadeTone(tpuIsPlaying ? 350 : 650, 30);
                                  setTpuIsPlaying(!tpuIsPlaying);
                                }}
                                className={`px-3 py-1.5 rounded-lg border transition-colors ${
                                  tpuIsPlaying
                                    ? "bg-[#020308] border-amber-400 text-amber-300"
                                    : "bg-[#020308] border-[rgba(250,249,245,0.12)] text-[#5eead4] hover:border-[#5eead4]"
                                }`}
                              >
                                <span>{tpuIsPlaying ? "Pause" : "Replay Inversion"}</span>
                              </button>
                              <button
                                onClick={() => {
                                  playArcadeTone(280, 20);
                                  setTpuIsPlaying(false);
                                  setTpuStep(0);
                                }}
                                className="px-2.5 py-1.5 rounded-lg bg-[#020308] border border-[rgba(250,249,245,0.08)] text-[rgba(250,249,245,0.55)] hover:text-[#faf9f5] transition-colors"
                              >
                                ↺ Reset
                              </button>
                              <button
                                onClick={() => {
                                  playArcadeTone(700, 30);
                                  setTpuIsPlaying(false);
                                  setTpuStep(68);
                                }}
                                className="px-2.5 py-1.5 rounded-lg bg-[#020308] border border-[rgba(250,249,245,0.12)] text-purple-300 hover:border-purple-400 transition-colors"
                              >
                                Jump to Inversion (t = 68)
                              </button>
                            </div>
                          </div>

                          {/* SVG Canvas for Potential Landscape and Particle */}
                          {(() => {
                            const s = tpuStep;
                            const ym = s <= 70 ? 1.0 + 11.5 * (s / 70.0) : Math.max(1.0, 12.5 - 11.5 * ((s - 70) / 30.0));
                            const qHk = 3697.4 * Math.min(1.0, Math.max(0.0, (s - 15) / 55.0));

                            let px = -0.92;
                            if (s < 45) {
                              px = -0.92 + 0.04 * Math.sin(s * 0.7);
                            } else if (s <= 68) {
                              const t = (s - 45) / 23.0;
                              px = -0.92 + t * 1.94;
                            } else {
                              px = 1.02 + 0.02 * Math.sin(s * 0.5);
                            }

                            const svgW = 600;
                            const svgH = 220;
                            const xToSvg = (x: number) => ((x + 1.6) / 3.2) * svgW;
                            const vFunc = (x: number) => {
                              const vBase = 1.8 * Math.pow(x * x - 1.0, 2);
                              const vClause = (ym * 0.35) * Math.log(1.0 + Math.exp(-3.2 * x));
                              return vBase + vClause;
                            };

                            let dStr = "";
                            const points: [number, number][] = [];
                            for (let i = 0; i <= 60; i++) {
                              const x = -1.6 + (i / 60) * 3.2;
                              const v = vFunc(x);
                              const sy = svgH - 25 - Math.min(180, (v / 6.5) * 160);
                              const sx = xToSvg(x);
                              points.push([sx, sy]);
                              dStr += (i === 0 ? `M ${sx} ${sy}` : ` L ${sx} ${sy}`);
                            }

                            const pSvgX = xToSvg(px);
                            const pSvgY = svgH - 25 - Math.min(180, (vFunc(px) / 6.5) * 160);

                            const statusText =
                              s < 45
                                ? "SADDLE TRAPPED (Violation C_m > 0)"
                                : s <= 68
                                ? "BARRIER TILTED · EJECTING OSCILLATOR"
                                : "GROUND BASIN LOCKED (100% SAT, 0 Violations)";

                            const statusColor =
                              s < 45 ? "text-rose-400" : s <= 68 ? "text-amber-400" : "text-[#34d399]";

                            return (
                              <div className="space-y-3">
                                <div className="relative bg-[#020308] rounded-xl p-4 border border-[rgba(250,249,245,0.08)] overflow-hidden">
                                  <svg viewBox={`0 0 ${svgW} ${svgH}`} className="w-full h-44">
                                    <defs>
                                      <linearGradient id="gradCurve" x1="0%" y1="0%" x2="100%" y2="0%">
                                        <stop offset="0%" stopColor="#f43f5e" />
                                        <stop offset="50%" stopColor="#f59e0b" />
                                        <stop offset="100%" stopColor="#34d399" />
                                      </linearGradient>
                                    </defs>

                                    <line x1="0" y1={svgH - 25} x2={svgW} y2={svgH - 25} stroke="rgba(250,249,245,0.1)" strokeWidth="1" />
                                    <line x1={xToSvg(0)} y1="10" x2={xToSvg(0)} y2={svgH - 25} stroke="rgba(250,249,245,0.1)" strokeDasharray="3 3" />

                                    <text x={xToSvg(-1.0)} y={svgH - 8} fill="#f43f5e" fontSize="10" fontFamily="monospace" textAnchor="middle">
                                      Violated Basin (x ≈ -1.0)
                                    </text>
                                    <text x={xToSvg(0.0)} y={svgH - 8} fill="rgba(250,249,245,0.4)" fontSize="10" fontFamily="monospace" textAnchor="middle">
                                      Saddle (x = 0)
                                    </text>
                                    <text x={xToSvg(1.0)} y={svgH - 8} fill="#34d399" fontSize="10" fontFamily="monospace" textAnchor="middle">
                                      Satisfied Basin (x ≈ +1.0)
                                    </text>

                                    <path d={dStr} fill="none" stroke="url(#gradCurve)" strokeWidth="2.5" strokeLinecap="round" />

                                    {s >= 45 && s <= 68 && (
                                      <line
                                        x1={pSvgX}
                                        y1={pSvgY - 12}
                                        x2={pSvgX + 32}
                                        y2={pSvgY - 12}
                                        stroke="#f59e0b"
                                        strokeWidth="2"
                                      />
                                    )}

                                    <circle
                                      cx={pSvgX}
                                      cy={pSvgY}
                                      r="6"
                                      fill={s >= 68 ? "#34d399" : s >= 45 ? "#f59e0b" : "#f43f5e"}
                                    />
                                    <circle cx={pSvgX} cy={pSvgY} r="2.5" fill="#faf9f5" />
                                  </svg>

                                  <div className="absolute top-3 left-4 flex flex-wrap gap-2 text-[10px] font-mono">
                                    <span className="px-2 py-0.5 rounded bg-[#05070f] border border-[rgba(250,249,245,0.08)] text-[rgba(250,249,245,0.65)]">
                                      Step: <strong className="text-[#faf9f5]">{s} / 100</strong>
                                    </span>
                                    <span className="px-2 py-0.5 rounded bg-[#05070f] border border-[rgba(250,249,245,0.08)] text-[#5eead4]">
                                      Multiplier y_m(t): <strong>{ym.toFixed(2)}</strong>
                                    </span>
                                    <span className="px-2 py-0.5 rounded bg-[#05070f] border border-[rgba(250,249,245,0.08)] text-rose-300">
                                      Heat Q_hk: <strong>{qHk.toFixed(1)}</strong>
                                    </span>
                                    <span className={`px-2 py-0.5 rounded bg-[#05070f] border border-[rgba(250,249,245,0.08)] font-medium ${statusColor}`}>
                                      {statusText}
                                    </span>
                                  </div>
                                </div>

                                <div className="space-y-1 font-mono">
                                  <div className="flex justify-between text-[10px] text-[rgba(250,249,245,0.4)]">
                                    <span>t = 0 (Trapped in Saddle)</span>
                                    <span>t = 68 (Dynamic Barrier Inversion)</span>
                                    <span>t = 100 (Equilibrated Ground State)</span>
                                  </div>
                                  <input
                                    type="range"
                                    min="0"
                                    max="100"
                                    value={tpuStep}
                                    onChange={(e) => {
                                      setTpuIsPlaying(false);
                                      setTpuStep(parseInt(e.target.value, 10));
                                    }}
                                    className="w-full h-1.5 bg-[#020308] rounded-lg appearance-none cursor-pointer accent-[#5eead4]"
                                  />
                                </div>
                              </div>
                            );
                          })()}
                        </div>

                        {/* Phase-4 Silicon Realism & DAC Quantization Widget */}
                        <div className="pt-2">
                          <SiliconRealismWidget />
                        </div>
                      </div>
                    )}

                    {/* =========================================================
                        SUB-VIEW 2: SMC RESAMPLING & N_EFF RESUSCITATION (N=128)
                    ========================================================= */}
                    {tpuActiveTab === "resampling" && (
                      <div className="flex flex-col gap-5">
                        <div className="p-4 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.08)] flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
                          <div>
                            <span className="text-sm font-medium text-[#faf9f5]">
                              Jarzynski-Neal Sequential Monte Carlo (SMC) Trajectory Population
                            </span>
                            <p className="text-xs text-[rgba(250,249,245,0.55)] font-light mt-0.5">
                              Steep thermal quenches cause exponential dissipation variance (O(N)), collapsing unresampled weights onto a single walker.
                            </p>
                          </div>

                          <div className="flex items-center gap-1 p-1 bg-[#020308] rounded-xl border border-[rgba(250,249,245,0.08)] shrink-0 font-mono text-[11px]">
                            <button
                              onClick={() => {
                                playArcadeTone(300, 30, "sawtooth");
                                setTpuResamplingMode("no_smc");
                              }}
                              className={`px-3 py-1.5 rounded-lg transition-colors ${
                                tpuResamplingMode === "no_smc"
                                  ? "bg-[#05070f] text-rose-300 border border-rose-500/40"
                                  : "text-[rgba(250,249,245,0.45)] hover:text-[#faf9f5]"
                              }`}
                            >
                              Without Resampling (Collapsed)
                            </button>
                            <button
                              onClick={() => {
                                playArcadeTone(580, 30);
                                setTpuResamplingMode("with_smc");
                              }}
                              className={`px-3 py-1.5 rounded-lg transition-colors ${
                                tpuResamplingMode === "with_smc"
                                  ? "bg-[#05070f] text-[#34d399] border border-[#34d399]/40"
                                  : "text-[rgba(250,249,245,0.45)] hover:text-[#faf9f5]"
                              }`}
                            >
                              With 22-Stage SMC Resampling
                            </button>
                          </div>
                        </div>

                        {tpuResamplingMode === "no_smc" ? (
                          <div className="p-3.5 rounded-xl bg-[#05070f] border border-rose-500/40 text-rose-300 text-xs flex flex-wrap items-center justify-between gap-3 font-mono">
                            <div>
                              <strong className="text-rose-400">TRAJECTORY COLLAPSE:</strong> Kish Effective Sample Size dropped to <strong>N_eff = 1.55 / 48 (3.2%)</strong>. Walker #1 holds 96.8% of total ensemble weight; 47 walkers are statistically dead.
                            </div>
                            <span className="font-bold whitespace-nowrap">N_eff = 1.55 (3.2%)</span>
                          </div>
                        ) : (
                          <div className="p-3.5 rounded-xl bg-[#05070f] border border-[#34d399]/40 text-[#34d399] text-xs flex flex-wrap items-center justify-between gap-3 font-mono">
                            <div>
                              <strong className="text-[#34d399]">RESAMPLE RESUSCITATION:</strong> 22 systematic resampling stages maintained Kish ratio above 50% across the full quench. Mean <strong>N_eff = 30.63 / 48 (63.8%)</strong>. Free energy: <strong>Δ(βF) = -1,773.18</strong>.
                            </div>
                            <span className="font-bold whitespace-nowrap">N_eff = 30.63 (63.8%)</span>
                          </div>
                        )}

                        <div className="p-5 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.08)] space-y-3">
                          <div className="flex justify-between items-center text-xs border-b border-[rgba(250,249,245,0.08)] pb-2 font-mono">
                            <span className="text-[rgba(250,249,245,0.7)]">
                              Trajectory Ensemble Normalized Weight Distribution (B = 48 Walkers, N = 128 Spins)
                            </span>
                            <span className="text-[rgba(250,249,245,0.4)] text-[11px]">
                              {tpuResamplingMode === "no_smc" ? "Unresampled Exp(-ΔΣ)" : "SMC Uniform Reset (1/B)"}
                            </span>
                          </div>

                          <div className="grid grid-cols-12 sm:grid-cols-24 lg:grid-cols-48 gap-1 items-end h-36 bg-[#020308] rounded-lg p-2.5 border border-[rgba(250,249,245,0.08)]">
                            {Array.from({ length: 48 }).map((_, idx) => {
                              let pct = 0;
                              let barColor = "bg-[#34d399]";
                              if (tpuResamplingMode === "no_smc") {
                                if (idx === 0) {
                                  pct = 96.8;
                                  barColor = "bg-rose-500";
                                } else if (idx === 1) {
                                  pct = 2.1;
                                  barColor = "bg-rose-400/60";
                                } else {
                                  pct = 0.05;
                                  barColor = "bg-[rgba(250,249,245,0.05)]";
                                }
                              } else {
                                const seedVal = Math.sin(idx * 7.3) * 0.45;
                                pct = 2.08 + seedVal;
                                barColor = "bg-[#5eead4]";
                              }

                              const hPx = Math.max(4, (pct / 100.0) * 115);

                              return (
                                <div
                                  key={idx}
                                  title={`Walker #${idx + 1}: ${pct.toFixed(2)}% weight`}
                                  className="flex flex-col items-center justify-end h-full"
                                >
                                  <div
                                    style={{ height: `${hPx}px` }}
                                    className={`w-full rounded-t transition-all duration-300 ${barColor}`}
                                  />
                                </div>
                              );
                            })}
                          </div>

                          <div className="flex justify-between items-center text-[10px] font-mono text-[rgba(250,249,245,0.35)] pt-1">
                            <span>Walker #01</span>
                            <span>Walker #24</span>
                            <span>Walker #48</span>
                          </div>
                        </div>

                        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
                          <div className="p-3.5 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.08)]">
                            <div className="text-[10px] text-[rgba(250,249,245,0.4)] font-mono uppercase">Problem Scale</div>
                            <div className="text-base font-mono font-medium text-[#faf9f5] mt-1">N=128, M=545</div>
                            <div className="text-[10px] text-[#5eead4] font-mono mt-0.5">Continuous 3-SAT</div>
                          </div>
                          <div className="p-3.5 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.08)]">
                            <div className="text-[10px] text-[rgba(250,249,245,0.4)] font-mono uppercase">Kish Ratio Gain</div>
                            <div className="text-base font-mono font-medium text-[#34d399] mt-1">3.2% → 63.8%</div>
                            <div className="text-[10px] text-[rgba(250,249,245,0.45)] font-mono mt-0.5">19.9× Resuscitation</div>
                          </div>
                          <div className="p-3.5 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.08)]">
                            <div className="text-[10px] text-[rgba(250,249,245,0.4)] font-mono uppercase">SMC Stages</div>
                            <div className="text-base font-mono font-medium text-[#5eead4] mt-1">22 Events</div>
                            <div className="text-[10px] text-[rgba(250,249,245,0.45)] font-mono mt-0.5">Threshold N_eff/B &lt; 0.5</div>
                          </div>
                          <div className="p-3.5 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.08)]">
                            <div className="text-[10px] text-[rgba(250,249,245,0.4)] font-mono uppercase">Free Energy Δ(βF)</div>
                            <div className="text-base font-mono font-medium text-purple-300 mt-1">-1,773.18</div>
                            <div className="text-[10px] text-[#34d399] font-mono mt-0.5">Jarzynski Log-Sum-Exp</div>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* =========================================================
                        SUB-VIEW 3: BAOAB VS EULER-MARUYAMA & THERMOMETERS
                    ========================================================= */}
                    {tpuActiveTab === "integrator" && (
                      <div className="flex flex-col gap-5">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                          <div className="p-5 rounded-xl bg-[#05070f] border border-rose-500/30 space-y-3">
                            <div className="flex items-center justify-between">
                              <span className="text-[10px] font-mono uppercase tracking-[0.18em] text-rose-400">
                                Classical Baseline · Euler-Maruyama
                              </span>
                              <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-rose-500/10 text-rose-300 border border-rose-500/30">
                                Runaway Divergence
                              </span>
                            </div>
                            <div className="text-sm font-medium text-[#faf9f5]">
                              Non-Symplectic First-Order Drift Discretization
                            </div>
                            <p className="text-xs text-[rgba(250,249,245,0.55)] font-light leading-relaxed">
                              When potential curvature exceeds <code className="text-rose-300 font-mono">ω dt ≈ 2</code>, Euler-Maruyama experiences violent energy pump resonance. Velocity explodes without bound.
                            </p>
                            <div className="space-y-1.5 pt-2 border-t border-[rgba(250,249,245,0.06)] text-xs font-mono">
                              <div className="flex justify-between py-0.5 border-b border-[rgba(250,249,245,0.04)]">
                                <span className="text-[rgba(250,249,245,0.45)]">Divergence Point:</span>
                                <span className="text-rose-400 font-medium">Step 277 (Blowup)</span>
                              </div>
                              <div className="flex justify-between py-0.5 border-b border-[rgba(250,249,245,0.04)]">
                                <span className="text-[rgba(250,249,245,0.45)]">Velocity Norm ||v||:</span>
                                <span className="text-rose-400 font-medium">80.35 (&gt; 50 threshold)</span>
                              </div>
                              <div className="flex justify-between py-0.5 border-b border-[rgba(250,249,245,0.04)]">
                                <span className="text-[rgba(250,249,245,0.45)]">FDT Equipartition Error:</span>
                                <span className="text-rose-400 font-medium">82.4% (Violated)</span>
                              </div>
                              <div className="flex justify-between py-0.5">
                                <span className="text-[rgba(250,249,245,0.45)]">First Law Conservation:</span>
                                <span className="text-rose-400">Failed (Energy → +∞)</span>
                              </div>
                            </div>
                          </div>

                          <div className="p-5 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.12)] space-y-3">
                            <div className="flex items-center justify-between">
                              <span className="text-[10px] font-mono uppercase tracking-[0.18em] text-[#34d399]">
                                Station 16 Core · BAOAB Symplectic
                              </span>
                              <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-[#34d399]/10 text-[#34d399] border border-[#34d399]/30">
                                Exact First Law
                              </span>
                            </div>
                            <div className="text-sm font-medium text-[#faf9f5]">
                              Leimkuhler-Matthews Symplectic Operator Splitting
                            </div>
                            <p className="text-xs text-[rgba(250,249,245,0.55)] font-light leading-relaxed">
                              Splits drift, diffusion, and dissipation into symmetric operators: preserves phase-space measure and yields exact marginal spatial Boltzmann distributions up to <code className="text-[#34d399] font-mono">O(dt²)</code>.
                            </p>
                            <div className="space-y-1.5 pt-2 border-t border-[rgba(250,249,245,0.06)] text-xs font-mono">
                              <div className="flex justify-between py-0.5 border-b border-[rgba(250,249,245,0.04)]">
                                <span className="text-[rgba(250,249,245,0.45)]">Integrator Stability:</span>
                                <span className="text-[#34d399] font-medium">100% Stable (H = 7.01)</span>
                              </div>
                              <div className="flex justify-between py-0.5 border-b border-[rgba(250,249,245,0.04)]">
                                <span className="text-[rgba(250,249,245,0.45)]">First Law Energy Error:</span>
                                <span className="text-[#34d399] font-medium">&lt; 10^(-10) Exact</span>
                              </div>
                              <div className="flex justify-between py-0.5 border-b border-[rgba(250,249,245,0.04)]">
                                <span className="text-[rgba(250,249,245,0.45)]">FDT Equipartition Error:</span>
                                <span className="text-[#5eead4] font-medium">17.3% in stiff regime</span>
                              </div>
                              <div className="flex justify-between py-0.5">
                                <span className="text-[rgba(250,249,245,0.45)]">Second Law Work &lang;W_diss&rang;:</span>
                                <span className="text-[#34d399] font-medium">+0.9210 (&gt; 0 Valid)</span>
                              </div>
                            </div>
                          </div>
                        </div>

                        <div className="p-5 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.08)] space-y-4">
                          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2 border-b border-[rgba(250,249,245,0.08)] pb-3">
                            <div>
                              <span className="text-sm font-medium text-[#faf9f5]">
                                Dual Thermometer Audit: Configurational (T_conf) vs. Kinetic (T_kin)
                              </span>
                              <div className="text-xs text-[rgba(250,249,245,0.45)] font-mono mt-0.5">
                                Formula: <code className="text-[#5eead4]">k_B T_conf = ⟨||∇H||²⟩ / ⟨ΔH⟩</code> vs. <code className="text-amber-300">k_B T_kin = ⟨m v²⟩</code>
                              </div>
                            </div>

                            <div className="flex items-center gap-3 text-xs font-mono">
                              <span className="text-[rgba(250,249,245,0.45)]">Stiffness α_well:</span>
                              <span className="px-2.5 py-0.5 rounded bg-[#020308] border border-[rgba(250,249,245,0.12)] text-[#5eead4] font-medium">
                                {tpuStiffnessAlpha.toFixed(1)}
                              </span>
                            </div>
                          </div>

                          <div className="space-y-1 font-mono">
                            <div className="flex justify-between text-[10px] text-[rgba(250,249,245,0.4)]">
                              <span>α = 1.0 (Harmonic Well)</span>
                              <span>α = 6.0 (Baseline Stiff)</span>
                              <span>α = 10.0 (Extreme Curvature)</span>
                            </div>
                            <input
                              type="range"
                              min="1"
                              max="10"
                              step="0.5"
                              value={tpuStiffnessAlpha}
                              onChange={(e) => {
                                playArcadeTone(380 + parseFloat(e.target.value) * 30, 20);
                                setTpuStiffnessAlpha(parseFloat(e.target.value));
                              }}
                              className="w-full h-1.5 bg-[#020308] rounded-lg appearance-none cursor-pointer accent-[#5eead4]"
                            />
                          </div>

                          {(() => {
                            const alpha = tpuStiffnessAlpha;
                            const tKin = 1.0 - (alpha / 10.0) * 0.022;
                            const errKin = Math.abs(1.0 - tKin) * 100;
                            const tConf = 1.0038;
                            const errConf = 0.38;

                            return (
                              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2 font-mono">
                                <div className="p-3.5 rounded-xl bg-[#020308] border border-[rgba(250,249,245,0.08)] space-y-2">
                                  <div className="flex justify-between items-center text-xs">
                                    <span className="text-[rgba(250,249,245,0.55)]">Kinetic (T_kin):</span>
                                    <span className="text-amber-300 font-medium">{tKin.toFixed(3)} K</span>
                                  </div>
                                  <div className="w-full h-1.5 bg-[#05070f] rounded-full overflow-hidden">
                                    <div
                                      style={{ width: `${(tKin / 1.05) * 100}%` }}
                                      className="h-full bg-amber-400 rounded-full transition-all duration-200"
                                    />
                                  </div>
                                  <div className="flex justify-between text-[10px] text-[rgba(250,249,245,0.4)]">
                                    <span>Lag Error: {errKin.toFixed(1)}%</span>
                                    <span>Target: 1.000 K</span>
                                  </div>
                                </div>

                                <div className="p-3.5 rounded-xl bg-[#020308] border border-[rgba(250,249,245,0.12)] space-y-2">
                                  <div className="flex justify-between items-center text-xs">
                                    <span className="text-[#5eead4] font-medium">Configurational (T_conf):</span>
                                    <span className="text-[#34d399] font-medium">{tConf.toFixed(3)} K</span>
                                  </div>
                                  <div className="w-full h-1.5 bg-[#05070f] rounded-full overflow-hidden">
                                    <div
                                      style={{ width: `${(tConf / 1.05) * 100}%` }}
                                      className="h-full bg-[#34d399] rounded-full transition-all duration-200"
                                    />
                                  </div>
                                  <div className="flex justify-between text-[10px] text-[rgba(250,249,245,0.4)]">
                                    <span className="text-[#34d399]">Error: {errConf.toFixed(1)}% (Exact Boltzmann)</span>
                                    <span>Target: 1.000 K</span>
                                  </div>
                                </div>
                              </div>
                            );
                          })()}
                        </div>

                        <div className="p-4 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.08)] flex flex-wrap items-center justify-between gap-3">
                          <div className="space-y-1">
                            <div className="text-xs font-medium text-[#faf9f5] flex items-center gap-2">
                              <span>Upstream System: 4-Organ Synthetic Intelligence (SI) Core</span>
                              <span className="px-2 py-0.5 rounded bg-[rgba(250,249,245,0.06)] text-[#5eead4] font-mono text-[10px]">Station 15</span>
                            </div>
                            <p className="text-[11px] text-[rgba(250,249,245,0.55)] font-light">
                              Couples Continuous Hopfield episodic memory, Kramers Langevin curiosity hopping, 6-node Potts goal consensus, and RTX 5080 metabolic VRAM scheduling.
                            </p>
                          </div>
                          <Link
                            href="/si-core"
                            className="rounded-full border border-[rgba(250,249,245,0.12)] bg-[#020308] hover:border-[rgba(250,249,245,0.3)] text-[#5eead4] px-4 py-2 font-mono text-xs uppercase tracking-wider transition-colors shrink-0"
                          >
                            Explore SI Core ↗
                          </Link>
                        </div>
                      </div>
                    )}

                    {/* =========================================================
                        SUB-VIEW 4: SILICON REALISM & DAC QUANTIZATION (PHASE 4)
                    ========================================================= */}
                    {tpuActiveTab === "silicon" && (
                      <div className="flex flex-col gap-5">
                        <SiliconRealismWidget />
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>

            {/* ===============================================================
                EXPANDABLE ENGINEER DEEP DIVE ACCORDION
            =============================================================== */}
            <div className="border-t border-slate-800/80 bg-slate-950/60 px-6 sm:px-8 py-3">
              <button
                onClick={() => toggleAccordion(station.id)}
                className="w-full flex items-center justify-between text-xs font-mono text-[rgba(250,249,245,0.55)] hover:text-[#5eead4] transition-colors py-1 uppercase tracking-wider"
              >
                <span>
                  // Engineer Deep Dive · Hamiltonian &amp; Reproducible Script (
                  <code className="text-[#5eead4] lowercase">{station.scriptFile}</code>)
                </span>
                <span className="text-[10px]">{openAccordion[station.id] ? "▲ Collapse" : "▼ Expand"}</span>
              </button>

              {openAccordion[station.id] && (
                <div className="mt-3 pt-3 border-t border-[rgba(250,249,245,0.06)] grid grid-cols-1 md:grid-cols-3 gap-4 text-xs pb-2">
                  <div className="p-3.5 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.08)]">
                    <div className="text-[10px] uppercase tracking-[0.18em] text-[rgba(250,249,245,0.4)] font-mono mb-1.5">Hamiltonian / Objective</div>
                    <code className="text-[#5eead4] font-mono text-[11px]">{station.physicsFormula}</code>
                  </div>
                  <div className="p-3.5 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.08)]">
                    <div className="text-[10px] uppercase tracking-[0.18em] text-[rgba(250,249,245,0.4)] font-mono mb-1.5">
                      Block Gibbs Architecture
                    </div>
                    <p className="text-[rgba(250,249,245,0.65)] font-light leading-relaxed">{station.engineeringSummary}</p>
                  </div>
                  <div className="p-3.5 rounded-xl bg-[#05070f] border border-[rgba(250,249,245,0.08)] flex flex-col justify-between">
                    <div>
                      <div className="text-[10px] uppercase tracking-[0.18em] text-[rgba(250,249,245,0.4)] font-mono mb-1.5">
                        Reproduce via thrml-bench CLI
                      </div>
                      <code className="text-[#34d399] font-mono text-[11px]">
                        uv run thrml-bench run --exp {station.id}
                      </code>
                    </div>
                    <a
                      href={`https://github.com/nexusai-kevin/thermo-physics/blob/main/${station.scriptFile}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="mt-3 inline-flex items-center gap-1 text-[11px] font-mono text-[#5eead4] hover:underline"
                    >
                      <span>Inspect {station.scriptFile} on GitHub</span>
                      <span>↗</span>
                    </a>
                  </div>
                </div>
              )}
            </div>
          </section>
        ))}

        {/* ===================================================================
            HARDWARE PERFORMANCE & SCALING BENCHMARK FOOTER CARD
        =================================================================== */}
        <section className="rounded-2xl border border-[rgba(250,249,245,0.10)] bg-[#060811]/90 p-7 sm:p-9 shadow-2xl backdrop-blur-sm">
          <div className="flex flex-wrap items-center justify-between gap-4 mb-6">
            <div>
              <span className="font-mono text-[10px] uppercase tracking-[0.22em] text-[#5eead4]">
                // Initiative 3 · Hardware Scaling Telemetry
              </span>
              <h3 className="font-[family-name:var(--font-display)] text-2xl sm:text-3xl font-normal text-[#faf9f5] mt-1 tracking-tight">
                Multi-Scale 2-Color Block Gibbs Throughput (L = 16 → 128)
              </h3>
            </div>
            <a
              href="https://github.com/nexusai-kevin/thermo-physics/blob/main/BENCHMARKS.md"
              target="_blank"
              rel="noopener noreferrer"
              className="rounded-full border border-[rgba(250,249,245,0.12)] bg-[#05070f] px-5 py-2 font-mono text-[11px] uppercase tracking-[0.15em] text-[#faf9f5] hover:border-[rgba(250,249,245,0.3)] transition-colors"
            >
              View Full BENCHMARKS.md Index ↗
            </a>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-center">
            {[
              { size: "16 × 16 (256 Spins)", msu: "0.33 MSU/s", speedup: "128 Parallel Spins/Step" },
              { size: "32 × 32 (1,024 Spins)", msu: "1.30 MSU/s", speedup: "512 Parallel Spins/Step" },
              { size: "64 × 64 (4,096 Spins)", msu: "6.86 MSU/s", speedup: "2,048 Parallel Spins/Step" },
              { size: "128 × 128 (16,384 Spins)", msu: "20.18 MSU/s", speedup: "8,192 Parallel Spins/Step" },
            ].map((item, i) => (
              <div key={i} className="p-5 rounded-xl bg-[#020308]/60 border border-[rgba(250,249,245,0.08)]">
                <div className="font-mono text-[10px] uppercase tracking-wider text-[rgba(250,249,245,0.4)]">{item.size}</div>
                <div className="text-xl font-mono font-medium text-[#5eead4] mt-2">
                  {item.msu}
                </div>
                <div className="text-xs font-mono text-[rgba(250,249,245,0.65)] mt-1">{item.speedup}</div>
              </div>
            ))}
          </div>
        </section>

        {/* ===================================================================
            COMMUNITY, UPSTREAM CONTRIBUTION & QUICKSTART TUTORIAL BANNER
        =================================================================== */}
        <section className="rounded-2xl border border-[rgba(250,249,245,0.10)] bg-[#060811]/90 p-7 sm:p-9 shadow-2xl backdrop-blur-sm">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
            <div>
              <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-[#5eead4] mb-2">
                // Community &amp; Upstream Alignment
              </div>
              <h3 className="font-[family-name:var(--font-display)] text-2xl sm:text-3xl font-normal text-[#faf9f5] tracking-tight">
                Contribute, Replicate &amp; Run the Code
              </h3>
              <p className="text-[rgba(250,249,245,0.65)] text-sm font-light max-w-2xl mt-2 leading-relaxed">
                Connect directly with our open-source thermodynamic computing research on Extropic’s <code className="text-[#5eead4] font-mono">thrml</code> ecosystem. Inspect our active upstream PR, run the end-to-end interactive Jupyter tutorial, or explore the reproducibility repository.
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-3 font-mono text-[11px] uppercase tracking-[0.12em]">
              {/* Benchmark Dossier Whitepaper */}
              <a
                href="/BENCHMARK_DOSSIER.html"
                target="_blank"
                rel="noopener noreferrer"
                className="rounded-full border border-[rgba(250,249,245,0.12)] bg-[#05070f] px-4 py-2.5 text-[#faf9f5] hover:border-[rgba(250,249,245,0.3)] transition-colors"
              >
                16-Station Dossier ↗
              </a>

              {/* Upstream PR #79 */}
              <a
                href="https://github.com/extropic-ai/thrml/pull/79"
                target="_blank"
                rel="noopener noreferrer"
                className="rounded-full border border-[rgba(250,249,245,0.12)] bg-[#05070f] px-4 py-2.5 text-[#5eead4] hover:border-[#5eead4] transition-colors"
              >
                PR #79 on thrml ↗
              </a>

              {/* Open in Colab */}
              <a
                href="https://colab.research.google.com/github/nexusai-kevin/thermo-physics/blob/main/notebooks/quickstart_tutorial.ipynb"
                target="_blank"
                rel="noopener noreferrer"
                className="rounded-full border border-[rgba(250,249,245,0.12)] bg-[#05070f] px-4 py-2.5 text-[#f59e0b] hover:border-[#f59e0b] transition-colors"
              >
                Open in Colab ↗
              </a>

              {/* GitHub Repository */}
              <a
                href="https://github.com/nexusai-kevin/thermo-physics"
                target="_blank"
                rel="noopener noreferrer"
                className="rounded-full bg-[#faf9f5] px-5 py-2.5 font-medium text-[#020308] hover:bg-white transition-colors"
              >
                GitHub Repository ↗
              </a>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
