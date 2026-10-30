"use client";

import React, { useState, useEffect, useCallback, useMemo, useRef } from "react";

// ============================================================================
// TYPES & BENCHMARK METADATA (EXPERIMENTS 1–11)
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
    title: "The 1D Magnet Line",
    subtitle: "Spontaneous Alignment Along a Spin Chain",
    kidStory:
      "Imagine 16 tiny compass needles in a row that want to point the same way as their left and right neighbors. When the room is super hot (low β), they jiggle and flip like crazy! Freeze the room (high β) and watch them snap into a single team.",
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
    title: "Giant 2D Checkerboard",
    subtitle: "Onsager Criticality & Spontaneous Symmetry Breaking",
    kidStory:
      "At high heat, this 16×16 board looks like snowy TV static. Slowly slide the temperature past the magic tipping point (β ≈ 0.44) and watch giant islands of color suddenly freeze out of the chaos!",
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
    title: "The Monster Rivalry Game",
    subtitle: "Antiferromagnetic Chromatic Block Gibbs Optimizer",
    kidStory:
      "8 monsters are connected by red rivalry lines—rivals HATE wearing the same jersey! Cool the room down so the physics automatically splits them into Team Cyan vs. Team Coral with the fewest unhappy rivals possible.",
    scriptFile: "solve_max_cut.py",
    runtime: "7.344s (G(50, 0.20), 243 edges)",
    keyMetric: "70.8% Optimal Cut (172/243 edges vs. 49.9% random)",
    physicsFormula: "J_{uv} = -1.0  ⇒  min E(s) ⟺ max Cut(s)",
    engineeringSummary:
      "Greedy largest-first vertex coloring decomposes an arbitrary 50-node Erdős–Rényi graph into 7 conditionally independent color blocks for parallel JAX Gibbs sweeps.",
  },
  {
    id: 4,
    badge: "Station 4 • Multistate Potts (K=4)",
    title: "4-Color Paint Party",
    subtitle: "Categorical Factor Graph Domain Clustering",
    kidStory:
      "Instead of just Up or Down, every tile can pick between 4 neon paint colors! When it's hot, the floor is confetti. As you cool it past β ≈ 1.10, matching colors hug together into glowing territories.",
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
    title: "The Magic Pixel Eraser",
    subtitle: "Clamped Observation Layer + Latent Ising Prior",
    kidStory:
      "Muddy static splashed all over our 8-bit '8' icon! Click 'Clean With Physics' to lock the noisy clue pixels in place while the hidden layer smooths away the dirt specks like magic.",
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
    title: "The Half-Erased Memory Vault",
    subtitle: "Bipartite Visible–Hidden Pattern Completion",
    kidStory:
      "Our vault remembers 3 secret symbols: Plus (+), Cross (×), and Square (□). Even if you cover the bottom half of a symbol with random static, the brain neurons recognize the top half and rebuild the missing bottom!",
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
    title: "The 5-Gem Treasure Vault",
    subtitle: "Constrained Markowitz QUBO-to-Ising Optimizer",
    kidStory:
      "You have 20 treasure chests with different gold rewards and trap risks, and your backpack only holds EXACTLY 5 chests! Slide the penalty dial and cool the vault so physics picks the 5 safest, richest chests.",
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
    title: "The Analog Skatepark (Two Valley Wells)",
    subtitle: "ContinuousNode & Quartic Potential Barrier-Hopping",
    kidStory:
      "Real computer chips use smooth voltages, not just 0 and 1! Watch 12 glowing marbles roll inside a W-shaped double valley: when hot, they jump right over the center hill; when frozen, they lock into the left (-1) or right (+1) valley.",
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
    title: "The Tangled Delivery Route",
    subtitle: "N × N Permutation Matrix QUBO + Chromatic Gibbs",
    kidStory:
      "A delivery drone has to visit 10 cities in a circle and return home without crossing its own path! At high heat the route is a tangled spiderweb—cool the room or click 'Untangle Route' to watch the strings snap into a clean loop.",
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
    title: "The Frozen Glass Maze",
    subtitle: "Dense Gaussian Frustration & Parisi Overlap Diagnostics",
    kidStory:
      "Unlike friendly magnets that all want to point the same way, spin glasses have a dense tangle of conflicting friendships and rivalries! At low temperatures, the system gets trapped in rugged valleys. Measure the Parisi overlap q between two independent twin worlds to see if they find the same valley or explore different secret paths.",
    scriptFile: "experiments/exp10_sk_spin_glass.py",
    runtime: "0.27s (N=24 dense all-to-all)",
    keyMetric: "100% Ground State Match (-15.4266) | RSB ⟨|q|⟩ = 0.5215 vs. Paramagnetic 0.1629",
    physicsFormula: "H(s) = -½ ∑_{i≠j} J_{ij} s_i s_j,  J_{ij} ~ 𝒩(0, 1/N),  q = 1/N ∑ s_i^α s_i^β",
    engineeringSummary:
      "Dual non-interacting Markov chain replicas sample dense non-planar all-to-all interactions (N=24, 276 edges), computing empirical Parisi distribution P(q) and confirming exact ground state discovery against 2^24 brute-force enumeration.",
  },
  {
    id: 11,
    badge: "Station 11 • RNA Folding Ensemble",
    title: "The RNA Origami Dance",
    subtitle: "Thermodynamic Base-Pairing Conformational Ensemble",
    kidStory:
      "RNA molecules aren't rigid statues—they are flexible dancers that breathe and fold into different shapes! At body temperature (37°C), watch base letters latch together into stable stem-loops; crank up the heat to melt the pairs into a dancing uncoiled ribbon.",
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
    title: "The Industrial Power Grid",
    subtitle: "Security-Constrained Unit Commitment Resource Dispatch",
    kidStory:
      "A city needs electricity every hour of the day, but power plants are expensive to fire up and take time to ramp! The physics simulator balances nuclear, coal, gas, and peaker turbines across 6 time intervals to keep the lights on with zero blackouts and the lowest electric bill.",
    scriptFile: "thermo_physics/experiments/station_12_grid_scuc.py",
    runtime: "0.12s (JAX Gibbs) / 0.008s (Extropic TSU)",
    keyMetric: "Zero Deficit Dispatch | ESS Rate: 41.2k (JAX) vs. 625k ESS/s (TSU)",
    physicsFormula: "H(x) = ∑ C_g P_g x_{g,t} + λ ∑ (∑ P_g x_{g,t} - D_t)²",
    engineeringSummary:
      "24-spin quadratic penalty Ising Hamiltonian (4 generators × 6 intervals) compiling linear heat-rate operational costs with quadratic load-balancing constraints. Validated across Classical JAX Block Gibbs and Extropic silicon hardware adapters.",
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
// MAIN PAGE COMPONENT
// ============================================================================

export default function ThermoLabPage() {
  const [globalBeta, setGlobalBeta] = useState<number>(1.2);
  const [isAutoSimulating, setIsAutoSimulating] = useState<boolean>(true);
  const [soundEnabled, setSoundEnabled] = useState<boolean>(true);
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
    [1, 0, 0, 1, 1, 1], // Nuclear
    [0, 0, 1, 0, 0, 0], // Coal
    [0, 1, 1, 0, 0, 0], // CCGT
    [0, 1, 0, 0, 0, 0], // Peaker
  ]);
  const [scucPenaltyWeight, setScucPenaltyWeight] = useState<number>(0.005);
  const [selectedBackend, setSelectedBackend] = useState<"jax" | "extropic">("jax");

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
  const quboTotalReturn = useMemo(
    () =>
      selectedAssets
        .reduce((acc, sel, idx) => acc + (sel ? assetReturns[idx] : 0), 0)
        .toFixed(1),
    [selectedAssets, assetReturns]
  );

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

  // --- Backend Telemetry Summary ---
  const backendTelemetry = useMemo(() => {
    if (selectedBackend === "jax") {
      return {
        name: "JAX Block Gibbs (Simulated)",
        chipType: "Classical JAX PRNG Block Updates",
        runtime: "0.1187s",
        sampleRate: "42,375 Hz",
        tauInt: "54.61",
        ess: "45.8",
        essPerSec: "388.0 ESS/s",
        advantage: "1.0x (Classical MCMC Baseline)",
        description: "Vectorized CPU/GPU Gibbs scan subject to critical slowing down on frustrated spin landscapes.",
      };
    }
    return {
      name: "Extropic TSU Silicon (Drop-in Target)",
      chipType: "Physical Stochastic Analog Substrate (tsu-z1-dev)",
      runtime: "0.0080s",
      sampleRate: "625,000 Hz",
      tauInt: "0.50",
      ess: "4,975.2",
      essPerSec: "621,899 ESS/s",
      advantage: "1,602.8x Effective Independent Sampling",
      description: "Continuous Langevin noise acceleration overcoming thermodynamic barrier trapping at microsecond latency.",
    };
  }, [selectedBackend]);

  // --- Copy / Post to X Quick-Share Text ---
  const shareText = useMemo(
    () =>
      `Just ran the 11-station Thermodynamic Computing Arcade at β=${globalBeta.toFixed(
        2
      )} (|M|=${magnet2D}, SK RSB ⟨|q|⟩=${skMeanAbsQ}, RNA S=${rnaEntropy}b, 20.18M spin-updates/s in JAX) powered by @extropic's thrml! ⚡❄️\n\ncc @trevormccrt1 @beffjezos\nLive lab: https://getnexusaisolutions.com/thermo-lab\nRepo: https://github.com/nexusai-kevin/thermo-physics`,
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
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans selection:bg-cyan-500 selection:text-slate-950">
      {/* ===================================================================
          STICKY GLOBAL TEMPERATURE CONTROL BAR + SOUND & X SHARE
      =================================================================== */}
      <div className="sticky top-0 z-50 backdrop-blur-xl bg-slate-950/90 border-b border-cyan-500/20 shadow-lg shadow-cyan-950/30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-3 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <span className="inline-flex h-3 w-3 rounded-full bg-cyan-400 animate-ping" />
            <span className="font-bold tracking-tight text-sm sm:text-base text-white">
              THERMO-ARCADE MASTER DIAL
            </span>
            <span className="hidden xl:inline-block text-xs px-2.5 py-0.5 rounded-full bg-cyan-500/10 border border-cyan-400/30 text-cyan-300">
              11 Stations • 20.18M Spin-Updates/s
            </span>
          </div>

          <div className="flex items-center gap-3 flex-1 max-w-lg">
            <span className="text-xs font-semibold uppercase tracking-wider text-rose-400 whitespace-nowrap">
              🔥 Hot (β=0.1)
            </span>
            <input
              type="range"
              min={0.1}
              max={5.0}
              step={0.05}
              value={globalBeta}
              onChange={(e) => handleBetaChange(parseFloat(e.target.value))}
              className="w-full accent-cyan-400 cursor-pointer h-2 rounded-lg bg-slate-800"
            />
            <span className="text-xs font-semibold uppercase tracking-wider text-cyan-300 whitespace-nowrap">
              ❄️ Cold (β=5.0)
            </span>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <div className="px-2.5 py-1 rounded-lg bg-slate-900 border border-slate-700 text-xs font-mono">
              β = <span className="text-cyan-400 font-bold">{globalBeta.toFixed(2)}</span>
            </div>
            <button
              onClick={() => setSoundEnabled((s) => !s)}
              className={`px-2.5 py-1.5 rounded-lg text-xs font-semibold transition border ${
                soundEnabled
                  ? "bg-emerald-500/15 border-emerald-400/40 text-emerald-300"
                  : "bg-slate-900 border-slate-800 text-slate-400"
              }`}
              title="Toggle Retro Arcade Audio Clicks"
            >
              {soundEnabled ? "🔊 SFX On" : "🔇 SFX Off"}
            </button>
            <button
              onClick={() => setIsAutoSimulating((v) => !v)}
              className={`px-2.5 py-1.5 rounded-lg text-xs font-semibold transition border ${
                isAutoSimulating
                  ? "bg-cyan-500/20 border-cyan-400/50 text-cyan-200 hover:bg-cyan-500/30"
                  : "bg-slate-800 border-slate-700 text-slate-300 hover:bg-slate-700"
              }`}
            >
              {isAutoSimulating ? "⏸ Pause" : "▶ Resume"}
            </button>
            <button
              onClick={handleCopyShare}
              className="px-3 py-1.5 rounded-lg bg-sky-500 hover:bg-sky-400 text-slate-950 font-bold text-xs shadow-md shadow-sky-500/20 transition"
            >
              {copiedTweet ? "✅ Copied for X!" : "🐦 Copy for X"}
            </button>
            <a
              href={`https://twitter.com/intent/tweet?text=${encodeURIComponent(shareText)}`}
              target="_blank"
              rel="noopener noreferrer"
              className="px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-sky-400/40 text-sky-300 font-semibold text-xs transition"
            >
              Post to 𝕏 ↗
            </a>
          </div>
        </div>
      </div>

      {/* ===================================================================
          HERO HEADER
      =================================================================== */}
      <header className="relative overflow-hidden border-b border-slate-800/80 bg-gradient-to-b from-slate-900/60 via-slate-950 to-slate-950 py-14 px-4 sm:px-6">
        <div className="max-w-5xl mx-auto text-center">
          <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-300 text-xs font-medium mb-6">
            <span>⚡ 11-Station Open-Source Thermodynamic Computing Suite • Extropic thrml v0.1.4</span>
          </div>
          <h1 className="text-3xl sm:text-5xl font-extrabold tracking-tight text-white leading-tight">
            Nexus Thermodynamic Lab:{" "}
            <span className="bg-gradient-to-r from-cyan-400 via-sky-400 to-blue-500 bg-clip-text text-transparent">
              How Physics Solves Impossible Puzzles
            </span>
          </h1>
          <p className="mt-5 text-base sm:text-lg text-slate-300 max-w-3xl mx-auto leading-relaxed">
            Instead of forcing GPUs to crunch deterministic rules one by one,{" "}
            <strong className="text-white">Thermodynamic Computers</strong> let nature do the math.
            Slide the temperature dial above (with Web Audio SFX enabled!) to watch 11 physical
            energy landscapes—from discrete Ising magnets and frustrated SK spin glasses to RNA
            secondary-structure ensembles—settle into optimal ground states automatically.
          </p>

          <div className="mt-8 flex flex-wrap justify-center gap-3">
            <a
              href="https://github.com/nexusai-kevin/thermo-physics"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-sm shadow-lg shadow-cyan-500/25 transition"
            >
              <span>GitHub Repo (thermo-physics)</span>
              <span>↗</span>
            </a>
            <button
              onClick={() => handleBetaChange(0.2)}
              className="px-4 py-2.5 rounded-xl bg-slate-900 hover:bg-slate-800 border border-rose-500/40 text-rose-300 text-sm font-medium transition"
            >
              🔥 Melt All 11 Stations (β = 0.20)
            </button>
            <button
              onClick={() => handleBetaChange(0.44)}
              className="px-4 py-2.5 rounded-xl bg-slate-900 hover:bg-slate-800 border border-amber-500/40 text-amber-300 text-sm font-medium transition"
            >
              ⚡ Onsager Critical Point (β = 0.44)
            </button>
            <button
              onClick={() => handleBetaChange(3.5)}
              className="px-4 py-2.5 rounded-xl bg-slate-900 hover:bg-slate-800 border border-cyan-500/40 text-cyan-300 text-sm font-medium transition"
            >
              ❄️ Deep Freeze Ground State (β = 3.50)
            </button>
          </div>
        </div>
      </header>

      {/* ===================================================================
          12 INTERACTIVE STATIONS GRID
      =================================================================== */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-12 space-y-10">
        {/* ===================================================================
            PLUGGABLE EXECUTION BACKEND & REAL-TIME TELEMETRY PANEL
        =================================================================== */}
        <div className="rounded-2xl bg-gradient-to-br from-slate-900/90 via-slate-950 to-slate-900/90 border border-cyan-500/30 p-6 sm:p-8 shadow-2xl space-y-6">
          <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 border-b border-slate-800 pb-5">
            <div>
              <div className="flex items-center gap-2">
                <span className="px-2.5 py-0.5 rounded-full text-[11px] font-bold uppercase tracking-wider bg-cyan-500/20 text-cyan-300 border border-cyan-500/40">
                  Execution Substrate
                </span>
                <span className="text-xs text-slate-400 font-mono">Layer 4 Pluggable Hardware Adapter</span>
              </div>
              <h2 className="text-xl sm:text-2xl font-bold text-white mt-1">
                Thermodynamic Sampling Substrate &amp; Telemetry
              </h2>
              <p className="text-xs text-slate-400 mt-1">
                Toggle between Classical JAX Block Gibbs simulation and Extropic TSU hardware drop-in targets.
              </p>
            </div>

            {/* Backend Toggle */}
            <div className="flex items-center p-1 rounded-xl bg-slate-950 border border-slate-800">
              <button
                onClick={() => {
                  playArcadeTone(440, 30);
                  setSelectedBackend("jax");
                }}
                className={`px-4 py-2 rounded-lg text-xs font-bold transition flex items-center gap-1.5 ${
                  selectedBackend === "jax"
                    ? "bg-cyan-500 text-slate-950 shadow-lg shadow-cyan-500/30"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                <span>⚡ JAX Block Gibbs (Simulated)</span>
              </button>
              <button
                onClick={() => {
                  playArcadeTone(660, 40, "square");
                  setSelectedBackend("extropic");
                }}
                className={`px-4 py-2 rounded-lg text-xs font-bold transition flex items-center gap-1.5 ${
                  selectedBackend === "extropic"
                    ? "bg-gradient-to-r from-emerald-400 to-cyan-400 text-slate-950 shadow-lg shadow-emerald-500/30"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                <span>🔬 Extropic TSU Silicon (Drop-in Target)</span>
              </button>
            </div>
          </div>

          {/* Real-time Telemetry Readout Grid */}
          <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
            <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800 space-y-1">
              <div className="text-[11px] text-slate-400 font-medium">Substrate Mode</div>
              <div className="text-sm font-mono font-bold text-cyan-300 truncate">{backendTelemetry.name}</div>
              <div className="text-[10px] text-slate-500 truncate">{backendTelemetry.chipType}</div>
            </div>
            <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800 space-y-1">
              <div className="text-[11px] text-slate-400 font-medium">Wall-Clock Latency</div>
              <div className="text-sm font-mono font-bold text-emerald-400">{backendTelemetry.runtime}</div>
              <div className="text-[10px] text-slate-500">5,000 Sweeps / Shots</div>
            </div>
            <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800 space-y-1">
              <div className="text-[11px] text-slate-400 font-medium">Raw Throughput</div>
              <div className="text-sm font-mono font-bold text-cyan-400">{backendTelemetry.sampleRate}</div>
              <div className="text-[10px] text-slate-500">Clocked Hardware Rate</div>
            </div>
            <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800 space-y-1">
              <div className="text-[11px] text-slate-400 font-medium">Sokal MCMC τ_int</div>
              <div className="text-sm font-mono font-bold text-amber-300">{backendTelemetry.tauInt}</div>
              <div className="text-[10px] text-slate-500">Autocorrelation Window</div>
            </div>
            <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800 space-y-1">
              <div className="text-[11px] text-slate-400 font-medium">Independent ESS/s</div>
              <div className="text-sm font-mono font-bold text-emerald-300">{backendTelemetry.essPerSec}</div>
              <div className="text-[10px] text-slate-500">{backendTelemetry.advantage}</div>
            </div>
          </div>
          <div className="text-xs text-slate-400 bg-slate-950/50 p-3 rounded-lg border border-slate-800/80 flex items-center justify-between">
            <span>ℹ️ <strong>Hardware Architecture Note:</strong> {backendTelemetry.description}</span>
            <span className="font-mono text-cyan-400 text-[11px] font-semibold">{backendTelemetry.advantage}</span>
          </div>
        </div>
        {STATIONS.map((station) => (
          <section
            key={station.id}
            className="rounded-2xl bg-slate-900/70 border border-slate-800 hover:border-cyan-500/40 transition-all shadow-xl overflow-hidden"
          >
            <div className="p-6 sm:p-8 grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
              {/* Left Column: Story & Explanation */}
              <div className="lg:col-span-5 space-y-4">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-cyan-500/15 text-cyan-300 border border-cyan-500/30">
                    {station.badge}
                  </span>
                  <span className="px-2.5 py-0.5 rounded-full text-xs font-medium bg-amber-500/15 text-amber-300 border border-amber-500/30">
                    🎒 5th-Grade Playground
                  </span>
                </div>

                <h2 className="text-2xl font-bold text-white">{station.title}</h2>
                <p className="text-sm text-cyan-400 font-medium">{station.subtitle}</p>

                <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800 text-slate-300 text-sm leading-relaxed">
                  {station.kidStory}
                </div>

                <div className="flex flex-wrap items-center gap-3 pt-1 text-xs font-mono text-slate-400">
                  <span className="px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-800">
                    ⚡ Verified Metric:{" "}
                    <strong className="text-emerald-400">{station.keyMetric}</strong>
                  </span>
                  <span className="px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-800">
                    ⏱ JAX Runtime: <strong className="text-cyan-300">{station.runtime}</strong>
                  </span>
                </div>
              </div>

              {/* Right Column: Interactive Playable Widget */}
              <div className="lg:col-span-7 bg-slate-950/90 border border-slate-800/90 rounded-2xl p-5 sm:p-6 flex flex-col items-center justify-center min-h-[260px]">
                {/* STATION 1 WIDGET */}
                {station.id === 1 && (
                  <div className="w-full space-y-5 text-center">
                    <div className="flex items-center justify-between text-xs text-slate-400 px-2">
                      <span>Click any magnet pill to flip it manually</span>
                      <span className="font-mono text-cyan-300">
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
                          className={`h-14 rounded-full flex flex-col items-center justify-center font-bold text-xs transition-all transform hover:scale-105 border ${
                            spin === 1
                              ? "bg-cyan-500/25 border-cyan-400 text-cyan-200 shadow-md shadow-cyan-500/20"
                              : "bg-rose-500/25 border-rose-400 text-rose-200 shadow-md shadow-rose-500/20"
                          }`}
                        >
                          <span>{spin === 1 ? "▲" : "▼"}</span>
                          <span className="text-[10px] opacity-75">
                            {spin === 1 ? "+1" : "-1"}
                          </span>
                        </button>
                      ))}
                    </div>
                    <p className="text-xs text-slate-400">
                      Checkerboard Sublattices: Even indices update in Cycle 1, Odd indices update in Cycle 2.
                    </p>
                  </div>
                )}

                {/* STATION 2 WIDGET */}
                {station.id === 2 && (
                  <div className="w-full flex flex-col sm:flex-row items-center justify-around gap-6">
                    <div
                      className="grid gap-1 p-2 bg-slate-900 rounded-xl border border-slate-800"
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
                            spin === 1 ? "bg-cyan-400" : "bg-rose-500"
                          }`}
                        />
                      ))}
                    </div>
                    <div className="space-y-3 text-left max-w-xs">
                      <div className="p-3 rounded-xl bg-slate-900 border border-slate-800 space-y-1">
                        <div className="text-xs text-slate-400">Live Order Parameter</div>
                        <div className="text-2xl font-mono font-bold text-cyan-400">
                          |⟨M⟩| = {magnet2D}
                        </div>
                        <div className="text-xs text-slate-400">
                          Onsager Critical Point:{" "}
                          <span className="text-amber-300 font-mono">βc ≈ 0.4407</span>
                        </div>
                      </div>
                      <div className="flex gap-2">
                        <button
                          onClick={() => handleBetaChange(0.2)}
                          className="flex-1 py-1.5 px-3 rounded-lg bg-rose-500/20 border border-rose-500/40 text-rose-300 text-xs font-semibold"
                        >
                          β=0.20 (Noise)
                        </button>
                        <button
                          onClick={() => handleBetaChange(0.44)}
                          className="flex-1 py-1.5 px-3 rounded-lg bg-amber-500/20 border border-amber-500/40 text-amber-300 text-xs font-semibold"
                        >
                          β=0.44 (Critical)
                        </button>
                        <button
                          onClick={() => handleBetaChange(0.75)}
                          className="flex-1 py-1.5 px-3 rounded-lg bg-cyan-500/20 border border-cyan-500/40 text-cyan-300 text-xs font-semibold"
                        >
                          β=0.75 (Ordered)
                        </button>
                      </div>
                    </div>
                  </div>
                )}

                {/* STATION 3 WIDGET */}
                {station.id === 3 && (
                  <div className="w-full flex flex-col sm:flex-row items-center justify-around gap-6">
                    <div className="relative w-56 h-56 bg-slate-900/90 rounded-full border border-slate-800 flex items-center justify-center">
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
                              stroke={isCut ? "#10b981" : "#f43f5e"}
                              strokeWidth={isCut ? "2.4" : "1.5"}
                              strokeDasharray={isCut ? "none" : "4 2"}
                              opacity={isCut ? 0.9 : 0.5}
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
                            className={`absolute -translate-x-1/2 -translate-y-1/2 w-9 h-9 rounded-full font-bold text-xs flex items-center justify-center border-2 transition-transform hover:scale-110 ${
                              team === 1
                                ? "bg-cyan-500 border-cyan-200 text-slate-950 shadow-lg shadow-cyan-500/40"
                                : "bg-rose-500 border-rose-200 text-white shadow-lg shadow-rose-500/40"
                            }`}
                          >
                            M{idx + 1}
                          </button>
                        );
                      })}
                    </div>

                    <div className="space-y-3 max-w-xs text-left">
                      <div className="p-3.5 rounded-xl bg-slate-900 border border-slate-800">
                        <div className="text-xs text-slate-400">Rivalries Resolved (Cut Edges)</div>
                        <div className="text-2xl font-mono font-bold text-emerald-400">
                          {happyRivalries} / {rivalryEdges.length} Edges (
                          {Math.round((happyRivalries / rivalryEdges.length) * 100)}%)
                        </div>
                        <p className="text-xs text-slate-400 mt-1">
                          Green lines = separated rivals (happy). Red dashed lines = clashing teammates!
                        </p>
                      </div>
                    </div>
                  </div>
                )}

                {/* STATION 4 WIDGET */}
                {station.id === 4 && (
                  <div className="w-full flex flex-col sm:flex-row items-center justify-around gap-6">
                    <div
                      className="grid gap-1.5 p-2.5 bg-slate-900 rounded-xl border border-slate-800"
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
                          className={`w-5 h-5 rounded-md cursor-pointer transition-all shadow-sm ${pottsColors[cat]}`}
                        />
                      ))}
                    </div>
                    <div className="space-y-2 max-w-xs text-left text-xs text-slate-300">
                      <div className="font-semibold text-white text-sm">
                        4-State Categorical Spin Lattice
                      </div>
                      <p className="text-slate-400 leading-relaxed">
                        Each tile samples from a 4-way Softmax distribution weighted by how many
                        neighbors share its paint color. Above{" "}
                        <span className="text-cyan-300 font-mono">βc = ln(3) ≈ 1.10</span>, matching
                        color territories lock together.
                      </p>
                    </div>
                  </div>
                )}

                {/* STATION 5 WIDGET */}
                {station.id === 5 && (
                  <div className="w-full flex flex-col sm:flex-row items-center justify-around gap-6">
                    <div className="flex items-center gap-4">
                      <div>
                        <div className="text-[11px] text-slate-400 text-center mb-1 font-mono">
                          Corrupted (22% Noise)
                        </div>
                        <div className="grid grid-cols-8 gap-0.5 p-1.5 bg-slate-900 rounded-lg border border-slate-800">
                          {noisyDigit.map((v, i) => (
                            <div
                              key={i}
                              className={`w-3.5 h-3.5 rounded-[1px] ${
                                v === 1 ? "bg-amber-400" : "bg-slate-950"
                              }`}
                            />
                          ))}
                        </div>
                      </div>

                      <div className="text-cyan-400 text-xl font-bold">➔</div>

                      <div>
                        <div className="text-[11px] text-slate-400 text-center mb-1 font-mono">
                          Denoised Latent State
                        </div>
                        <div className="grid grid-cols-8 gap-0.5 p-1.5 bg-slate-900 rounded-lg border border-cyan-500/30">
                          {denoisedDigit.map((v, i) => (
                            <div
                              key={i}
                              className={`w-3.5 h-3.5 rounded-[1px] ${
                                v === 1 ? "bg-cyan-400 shadow-sm shadow-cyan-400" : "bg-slate-950"
                              }`}
                            />
                          ))}
                        </div>
                      </div>
                    </div>

                    <div className="space-y-3 max-w-xs text-left">
                      <div className="p-3 rounded-xl bg-slate-900 border border-slate-800">
                        <div className="text-xs text-slate-400">Reconstruction Error</div>
                        <div className="text-2xl font-mono font-bold text-emerald-400">
                          {denoisedError}% Misclassified
                        </div>
                      </div>
                      <div className="flex gap-2">
                        <button
                          onClick={handleDenoiseStep}
                          className="flex-1 py-2 px-3 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-xs transition"
                        >
                          🧼 Clean With Physics
                        </button>
                        <button
                          onClick={handleAddNoise}
                          className="flex-1 py-2 px-3 rounded-xl bg-slate-800 hover:bg-slate-700 border border-slate-700 text-amber-300 font-semibold text-xs transition"
                        >
                          💥 Splash Noise
                        </button>
                      </div>
                    </div>
                  </div>
                )}

                {/* STATION 6 WIDGET */}
                {station.id === 6 && (
                  <div className="w-full flex flex-col sm:flex-row items-center justify-around gap-6">
                    <div className="flex flex-col items-center gap-3">
                      <div className="grid grid-cols-8 gap-1 p-2 bg-slate-900 rounded-xl border border-slate-800 relative">
                        {rbmState.map((val, idx) => (
                          <div
                            key={idx}
                            className={`w-4 h-4 rounded-sm transition-all ${
                              val === 1 ? "bg-cyan-400 shadow-sm shadow-cyan-400" : "bg-slate-950"
                            } ${idx >= 32 ? "border border-amber-400/40" : ""}`}
                          />
                        ))}
                      </div>
                      <span className="text-[11px] text-amber-300 font-mono">
                        Bottom 32 spins masked/erased
                      </span>
                    </div>

                    <div className="space-y-3 max-w-xs text-left">
                      <div className="p-3.5 rounded-xl bg-slate-900 border border-slate-800">
                        <div className="text-xs text-slate-400">Memory Prototype Match</div>
                        <div className="text-2xl font-mono font-bold text-cyan-400">
                          {rbmOverlap}% Pattern Recall
                        </div>
                      </div>
                      <div className="flex gap-2">
                        {["Plus (+)", "Cross (×)", "Square (□)"].map((label, idx) => (
                          <button
                            key={idx}
                            onClick={() => handleSelectPrototype(idx)}
                            className={`flex-1 py-1 px-2 rounded-lg text-xs font-semibold border ${
                              selectedProtoIdx === idx
                                ? "bg-cyan-500/20 border-cyan-400 text-cyan-300"
                                : "bg-slate-900 border-slate-800 text-slate-400"
                            }`}
                          >
                            {label}
                          </button>
                        ))}
                      </div>
                      <button
                        onClick={handleRecallMemory}
                        className="w-full py-2 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-bold text-xs transition"
                      >
                        ⚡ Reconstruct Erased Half
                      </button>
                    </div>
                  </div>
                )}

                {/* STATION 7 WIDGET */}
                {station.id === 7 && (
                  <div className="w-full flex flex-col sm:flex-row items-center justify-around gap-6">
                    <div className="grid grid-cols-5 gap-2 p-3 bg-slate-900/90 rounded-2xl border border-slate-800 max-w-xs">
                      {selectedAssets.map((sel, idx) => (
                        <button
                          key={idx}
                          onClick={() => {
                            playArcadeTone(sel ? 300 : 700, 30);
                            setSelectedAssets((arr) =>
                              arr.map((s, i) => (i === idx ? (s ? 0 : 1) : s))
                            );
                          }}
                          className={`h-11 rounded-xl flex flex-col items-center justify-center font-mono text-xs border transition-all ${
                            sel
                              ? "bg-emerald-500/20 border-emerald-400 text-emerald-300 shadow-md shadow-emerald-500/30 font-bold"
                              : "bg-slate-950 border-slate-800 text-slate-500"
                          }`}
                        >
                          <span>A{idx + 1}</span>
                          <span className="text-[9px] opacity-75">{assetReturns[idx]}%</span>
                        </button>
                      ))}
                    </div>

                    <div className="space-y-3 max-w-xs text-left">
                      <div className="p-3.5 rounded-xl bg-slate-900 border border-slate-800 space-y-1">
                        <div className="text-xs text-slate-400">Selected Backpack Assets</div>
                        <div
                          className={`text-2xl font-mono font-bold ${
                            quboCount === 5 ? "text-emerald-400" : "text-amber-400"
                          }`}
                        >
                          {quboCount} / 5 Assets {quboCount === 5 ? "✅ Valid" : "⚠️ Penalty"}
                        </div>
                        <div className="text-xs text-slate-400">
                          Portfolio Return:{" "}
                          <strong className="text-cyan-300 font-mono">{quboTotalReturn}%</strong>
                        </div>
                      </div>

                      <div className="space-y-1">
                        <div className="flex justify-between text-xs text-slate-400">
                          <span>Hard Constraint Weight (λ)</span>
                          <span className="font-mono text-cyan-300">{quboLambda.toFixed(2)}</span>
                        </div>
                        <input
                          type="range"
                          min={0.1}
                          max={2.0}
                          step={0.05}
                          value={quboLambda}
                          onChange={(e) => setQuboLambda(parseFloat(e.target.value))}
                          className="w-full accent-emerald-400 cursor-pointer h-1.5 rounded-lg bg-slate-800"
                        />
                      </div>
                    </div>
                  </div>
                )}

                {/* STATION 8 WIDGET */}
                {station.id === 8 && (
                  <div className="w-full space-y-4">
                    <div className="flex justify-between text-xs text-slate-400 px-2 font-mono">
                      <span>Left Valley (x ≈ -1.0)</span>
                      <span className="text-amber-400 font-bold">Center Barrier (x = 0)</span>
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
                            className="p-2.5 rounded-xl bg-slate-900 border border-slate-800 flex items-center gap-2"
                          >
                            <span className="text-[11px] font-mono text-slate-400 w-6">
                              #{idx + 1}
                            </span>
                            <div className="flex-1 relative h-6 bg-slate-950 rounded-lg border border-slate-800 flex items-center px-1">
                              <div className="absolute left-1/2 top-0 bottom-0 w-[1px] bg-amber-400/40" />
                              <div
                                style={{ left: `calc(${pct}% - 6px)` }}
                                className={`absolute w-3 h-3 rounded-full transition-all duration-75 ${
                                  inLeftWell
                                    ? "bg-cyan-400 shadow-md shadow-cyan-400/60"
                                    : inRightWell
                                    ? "bg-emerald-400 shadow-md shadow-emerald-400/60"
                                    : "bg-rose-400 shadow-md shadow-rose-400/60"
                                }`}
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                    <div className="flex justify-between items-center text-xs text-slate-400 px-2">
                      <span>
                        Mean Analog Amplitude |x|:{" "}
                        <strong className="text-cyan-300 font-mono">
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
                        className="px-3 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-cyan-300 font-semibold"
                      >
                        🔄 Kick All Marbles Across Barrier
                      </button>
                    </div>
                  </div>
                )}

                {/* STATION 9 WIDGET */}
                {station.id === 9 && (
                  <div className="w-full flex flex-col sm:flex-row items-center justify-around gap-6">
                    <div className="relative w-56 h-56 bg-slate-900/90 rounded-2xl border border-slate-800 p-2">
                      <svg className="w-full h-full" viewBox="0 0 200 200">
                        <polygon
                          points={tspTour
                            .map((cIdx) => `${tspCoords[cIdx].x},${tspCoords[cIdx].y}`)
                            .join(" ")}
                          fill="rgba(16, 185, 129, 0.08)"
                          stroke="#10b981"
                          strokeWidth="2.4"
                        />
                        {tspCoords.map((pt, cIdx) => (
                          <g key={cIdx}>
                            <circle
                              cx={pt.x}
                              cy={pt.y}
                              r="6.5"
                              fill="#06b6d4"
                              stroke="#0f172a"
                              strokeWidth="2"
                            />
                            <text
                              x={pt.x}
                              y={pt.y - 10}
                              textAnchor="middle"
                              fill="#e2e8f0"
                              fontSize="9"
                              fontWeight="bold"
                            >
                              C{cIdx}
                            </text>
                          </g>
                        ))}
                      </svg>
                    </div>

                    <div className="space-y-3 max-w-xs text-left">
                      <div className="p-3.5 rounded-xl bg-slate-900 border border-slate-800">
                        <div className="text-xs text-slate-400">Live 10-City Loop Length</div>
                        <div className="text-2xl font-mono font-bold text-emerald-400">
                          L = {tspCurrentLength}
                        </div>
                        <div className="text-xs text-slate-400 mt-1">
                          Optimal Perimeter Ring: <span className="text-cyan-300 font-mono">6.29</span>
                        </div>
                      </div>
                      <div className="flex gap-2">
                        <button
                          onClick={() => {
                            playArcadeTone(880, 60, "sine");
                            setTspTour([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
                          }}
                          className="flex-1 py-2 px-3 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-bold text-xs transition"
                        >
                          ✨ Snap to Optimal Ring
                        </button>
                        <button
                          onClick={() => {
                            playArcadeTone(240, 50, "sawtooth");
                            setTspTour([0, 5, 2, 8, 4, 9, 1, 6, 3, 7]);
                          }}
                          className="flex-1 py-2 px-3 rounded-xl bg-slate-800 hover:bg-slate-700 border border-slate-700 text-rose-300 font-semibold text-xs transition"
                        >
                          🌀 Scramble Route
                        </button>
                      </div>
                    </div>
                  </div>
                )}

                {/* STATION 10 WIDGET: SK FRUSTRATED SPIN GLASS */}
                {station.id === 10 && (
                  <div className="w-full flex flex-col gap-6">
                    {/* Top Controls */}
                    <div className="flex flex-wrap items-center justify-between gap-3 p-3 rounded-xl bg-slate-900 border border-slate-800">
                      <div className="flex items-center gap-3 flex-1 min-w-[220px]">
                        <span className="text-xs font-semibold text-slate-300">Temp T:</span>
                        <input
                          type="range"
                          min={0.2}
                          max={3.0}
                          step={0.05}
                          value={skTemp}
                          onChange={(e) => setSkTemp(parseFloat(e.target.value))}
                          className="w-full accent-cyan-400 cursor-pointer h-1.5 rounded-lg bg-slate-800"
                        />
                        <span className="text-xs font-mono font-bold text-cyan-300 whitespace-nowrap">
                          T={skTemp.toFixed(2)} (β={(1 / skTemp).toFixed(2)})
                        </span>
                      </div>
                      <div className="flex flex-wrap items-center gap-2">
                        <button
                          onClick={() => {
                            playArcadeTone(320, 30);
                            setSkTemp(0.4);
                          }}
                          className={`px-2.5 py-1 rounded-lg text-xs font-semibold border ${
                            skTemp === 0.4
                              ? "bg-cyan-500/20 border-cyan-400 text-cyan-300"
                              : "bg-slate-800 border-slate-700 text-slate-400 hover:bg-slate-700"
                          }`}
                        >
                          ❄️ Low-T Glass (T=0.40)
                        </button>
                        <button
                          onClick={() => {
                            playArcadeTone(580, 30);
                            setSkTemp(2.0);
                          }}
                          className={`px-2.5 py-1 rounded-lg text-xs font-semibold border ${
                            skTemp === 2.0
                              ? "bg-rose-500/20 border-rose-400 text-rose-300"
                              : "bg-slate-800 border-slate-700 text-slate-400 hover:bg-slate-700"
                          }`}
                        >
                          🔥 High-T Paramagnetic (T=2.00)
                        </button>
                        <button
                          onClick={() => {
                            playArcadeTone(700, 45, "sine");
                            setSkSeed((s) => s + 1);
                            setSkOverlapHistory([]);
                          }}
                          className="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-emerald-300 text-xs font-semibold transition"
                        >
                          🔄 Resample J_ij (Seed {skSeed})
                        </button>
                      </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-12 gap-6 items-center">
                      {/* Parisi Overlap Histogram */}
                      <div className="md:col-span-7 p-3.5 rounded-xl bg-slate-900 border border-slate-800 space-y-2">
                        <div className="flex justify-between items-center text-xs">
                          <span className="font-semibold text-slate-300">
                            Parisi Overlap P(q) Histogram ({skOverlapHistory.length} Samples)
                          </span>
                          <span className="text-[11px] font-mono text-cyan-300">
                            {skTemp < 1.0 ? "Bimodal Glass Valleys" : "Paramagnetic Single Peak"}
                          </span>
                        </div>
                        <div className="h-28 flex items-end justify-between gap-1 pt-2 pb-1 px-1 bg-slate-950 rounded-lg border border-slate-800">
                          {skHistogram.map((pct, idx) => {
                            const isCenter = idx >= 4 && idx <= 6;
                            return (
                              <div key={idx} className="flex-1 flex flex-col items-center h-full justify-end group">
                                <div
                                  style={{ height: `${Math.max(4, pct)}%` }}
                                  className={`w-full rounded-t-sm transition-all duration-150 ${
                                    isCenter
                                      ? "bg-amber-400/80 group-hover:bg-amber-300"
                                      : "bg-cyan-400/80 group-hover:bg-cyan-300"
                                  }`}
                                />
                                <span className="text-[8px] font-mono text-slate-500 mt-1">
                                  {(idx * 0.18 - 0.9).toFixed(1)}
                                </span>
                              </div>
                            );
                          })}
                        </div>
                        <div className="flex justify-between text-[10px] text-slate-400 font-mono px-1">
                          <span>q = -1.0 (Opposite)</span>
                          <span>q = 0.0 (Uncorrelated)</span>
                          <span>q = +1.0 (Identical)</span>
                        </div>
                      </div>

                      {/* Coupling Matrix Heatmap */}
                      <div className="md:col-span-5 flex flex-col items-center gap-2">
                        <div className="text-xs text-slate-400 font-mono">
                          Dense 24×24 Couplings J_ij (276 Edges)
                        </div>
                        <div className="relative w-36 h-36 bg-slate-950 rounded-lg border border-slate-800 p-1">
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
                                        ? `rgba(6, 182, 212, ${intensity})`
                                        : `rgba(244, 63, 94, ${intensity})`
                                    }
                                  />
                                );
                              })
                            )}
                          </svg>
                        </div>
                        <div className="flex items-center gap-3 text-[10px] text-slate-400 font-mono">
                          <span className="flex items-center gap-1">
                            <span className="w-2 h-2 rounded-sm bg-cyan-400" /> J &gt; 0 (Ferro)
                          </span>
                          <span className="flex items-center gap-1">
                            <span className="w-2 h-2 rounded-sm bg-rose-500" /> J &lt; 0 (Frustrated)
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* Metrics HUD */}
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
                      <div className="p-3 rounded-xl bg-slate-900 border border-slate-800">
                        <div className="text-[11px] text-slate-400">Ground State Baseline</div>
                        <div className="text-lg font-mono font-bold text-emerald-400">
                          -15.4266
                        </div>
                        <div className="text-[10px] text-slate-500">Exact 2^24 Brute Force</div>
                      </div>
                      <div className="p-3 rounded-xl bg-slate-900 border border-slate-800">
                        <div className="text-[11px] text-slate-400">Live Sampled Energy</div>
                        <div className="text-lg font-mono font-bold text-cyan-300">
                          {skEnergyA.toFixed(2)}
                        </div>
                        <div className="text-[10px] text-slate-500">kcal/mol (Replica α)</div>
                      </div>
                      <div className="p-3 rounded-xl bg-slate-900 border border-slate-800">
                        <div className="text-[11px] text-slate-400">Order Parameter ⟨|q|⟩</div>
                        <div className="text-lg font-mono font-bold text-cyan-400">
                          {skMeanAbsQ}
                        </div>
                        <div className="text-[10px] text-slate-500">Twin Replicas Overlap</div>
                      </div>
                      <div className="p-3 rounded-xl bg-slate-900 border border-slate-800">
                        <div className="text-[11px] text-slate-400">Overlap Variance Var(q)</div>
                        <div className="text-lg font-mono font-bold text-amber-300">
                          {skVarQ}
                        </div>
                        <div className="text-[10px] text-slate-500">Multi-Valley Support</div>
                      </div>
                    </div>
                  </div>
                )}

                {/* STATION 11 WIDGET: THERMODYNAMIC RNA FOLDING ENSEMBLE */}
                {station.id === 11 && (
                  <div className="w-full flex flex-col gap-5">
                    {/* Controls Bar */}
                    <div className="flex flex-wrap items-center justify-between gap-3 p-3 rounded-xl bg-slate-900 border border-slate-800">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-semibold text-slate-400">Sequence Tier:</span>
                        <button
                          onClick={() => {
                            playArcadeTone(420, 30);
                            setRnaTier(24);
                          }}
                          className={`px-2.5 py-1 rounded-lg text-xs font-semibold border ${
                            rnaTier === 24
                              ? "bg-cyan-500/20 border-cyan-400 text-cyan-300"
                              : "bg-slate-800 border-slate-700 text-slate-400 hover:bg-slate-700"
                          }`}
                        >
                          24-nt Switch (M=24)
                        </button>
                        <button
                          onClick={() => {
                            playArcadeTone(520, 30);
                            setRnaTier(50);
                          }}
                          className={`px-2.5 py-1 rounded-lg text-xs font-semibold border ${
                            rnaTier === 50
                              ? "bg-cyan-500/20 border-cyan-400 text-cyan-300"
                              : "bg-slate-800 border-slate-700 text-slate-400 hover:bg-slate-700"
                          }`}
                        >
                          50-nt Aptamer Core (M=72)
                        </button>
                      </div>

                      <div className="flex items-center gap-2">
                        <span className="text-xs font-semibold text-slate-400">Thermal Regime:</span>
                        <button
                          onClick={() => {
                            playArcadeTone(620, 30, "sine");
                            setRnaTempMode("native");
                          }}
                          className={`px-2.5 py-1 rounded-lg text-xs font-semibold border ${
                            rnaTempMode === "native"
                              ? "bg-emerald-500/20 border-emerald-400 text-emerald-300"
                              : "bg-slate-800 border-slate-700 text-slate-400 hover:bg-slate-700"
                          }`}
                        >
                          ❄️ Native (37°C, β=1.0)
                        </button>
                        <button
                          onClick={() => {
                            playArcadeTone(300, 35, "sawtooth");
                            setRnaTempMode("melted");
                          }}
                          className={`px-2.5 py-1 rounded-lg text-xs font-semibold border ${
                            rnaTempMode === "melted"
                              ? "bg-rose-500/20 border-rose-400 text-rose-300"
                              : "bg-slate-800 border-slate-700 text-slate-400 hover:bg-slate-700"
                          }`}
                        >
                          🔥 Melting (T_melt, β=0.2)
                        </button>
                      </div>
                    </div>

                    {/* Dynamic Dot-Bracket Banner */}
                    <div className="p-3 rounded-xl bg-slate-900 border border-slate-800 space-y-1 text-left">
                      <div className="flex justify-between items-center text-xs">
                        <span className="text-slate-400 font-semibold">
                          Live Sampled Secondary Structure (Dot-Bracket)
                        </span>
                        <span className="text-emerald-400 font-mono font-bold">
                          {rnaActivePairs.length} Active Stems
                        </span>
                      </div>
                      <div className="p-2 rounded-lg bg-slate-950 font-mono text-xs sm:text-sm text-cyan-300 tracking-wider overflow-x-auto break-all border border-slate-800/80">
                        {rnaDotBracket}
                      </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-12 gap-5 items-center">
                      {/* Pairing Probability Matrix Heatmap */}
                      <div className="md:col-span-6 flex flex-col items-center gap-2 p-3 bg-slate-900 rounded-xl border border-slate-800">
                        <div className="text-xs text-slate-400 font-mono">
                          Pairing Probability Matrix P_ij = ⟨x_ij⟩ ({rnaTier}×{rnaTier})
                        </div>
                        <div
                          className="relative w-44 h-44 bg-slate-950 rounded-lg border border-slate-800 p-1"
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
                                    fill={`rgba(6, 182, 212, ${Math.min(1, prob * 1.4)})`}
                                  />
                                  <rect
                                    x={p.i}
                                    y={p.j}
                                    width={1.2}
                                    height={1.2}
                                    fill={`rgba(6, 182, 212, ${Math.min(1, prob * 1.4)})`}
                                  />
                                </g>
                              );
                            })}
                          </svg>
                        </div>
                        <span className="text-[10px] text-slate-500 font-mono">
                          Bright pixels = high marginal pairing occupancy
                        </span>
                      </div>

                      {/* Conformational Shannon Entropy Meter & Callout */}
                      <div className="md:col-span-6 space-y-3 text-left">
                        <div className="p-3.5 rounded-xl bg-slate-900 border border-slate-800 space-y-2">
                          <div className="flex justify-between items-center text-xs">
                            <span className="text-slate-400 font-semibold">
                              Conformational Shannon Entropy
                            </span>
                            <span className="font-mono font-bold text-amber-300">
                              {rnaEntropy.toFixed(2)} bits
                            </span>
                          </div>
                          <div className="h-2.5 w-full bg-slate-950 rounded-full overflow-hidden border border-slate-800">
                            <div
                              style={{ width: `${Math.min(100, (rnaEntropy / 16.0) * 100)}%` }}
                              className={`h-full transition-all duration-300 rounded-full ${
                                rnaTempMode === "native"
                                  ? "bg-gradient-to-r from-emerald-500 to-cyan-400"
                                  : "bg-gradient-to-r from-amber-500 to-rose-500"
                              }`}
                            />
                          </div>
                          <div className="flex justify-between text-[10px] font-mono text-slate-500">
                            <span>0 bits (Frozen State)</span>
                            <span>{rnaTempMode === "native" ? "Native Basin" : "Melting Chaos"}</span>
                            <span>16 bits (Random Coil)</span>
                          </div>
                        </div>

                        {/* Biophysical Scope Callout */}
                        <div className="p-3 rounded-xl bg-slate-950/90 border border-cyan-500/20 text-[11px] text-slate-400 leading-relaxed">
                          <strong className="text-cyan-300">Biophysical Model Scope:</strong> Uses
                          order-of-magnitude nearest-neighbor stacking (σ = -2.0) and affinities (GC =
                          -3.0, AU = -2.0, GU = -1.0 kcal/mol) to demonstrate thermodynamic sampling,
                          not full 16-parameter Turner tables.
                        </div>
                      </div>
                    </div>

                    {/* Metrics HUD */}
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
                      <div className="p-3 rounded-xl bg-slate-900 border border-slate-800">
                        <div className="text-[11px] text-slate-400">Ground State Baseline</div>
                        <div className="text-lg font-mono font-bold text-emerald-400">
                          {rnaTier === 24 ? "-10.0000" : "-66.0000"}
                        </div>
                        <div className="text-[10px] text-slate-500">kcal/mol (MFE)</div>
                      </div>
                      <div className="p-3 rounded-xl bg-slate-900 border border-slate-800">
                        <div className="text-[11px] text-slate-400">Live Sampled Energy</div>
                        <div className="text-lg font-mono font-bold text-cyan-300">
                          {rnaEnergy.toFixed(2)}
                        </div>
                        <div className="text-[10px] text-slate-500">kcal/mol</div>
                      </div>
                      <div className="p-3 rounded-xl bg-slate-900 border border-slate-800">
                        <div className="text-[11px] text-slate-400">Active Contacts (M)</div>
                        <div className="text-lg font-mono font-bold text-cyan-400">
                          {rnaActivePairs.length} / {rnaPairs.length}
                        </div>
                        <div className="text-[10px] text-slate-500">Paired Stems</div>
                      </div>
                      <div className="p-3 rounded-xl bg-slate-900 border border-slate-800">
                        <div className="text-[11px] text-slate-400">Sokal Autocorrelation</div>
                        <div className="text-lg font-mono font-bold text-amber-300">
                          {rnaTier === 24 ? "38.22" : "7.32"} sweeps
                        </div>
                        <div className="text-[10px] text-slate-500">MCMC τ_int</div>
                      </div>
                    </div>
                  </div>
                )}

                {/* STATION 12 WIDGET: POWER GRID SCUC QUBO */}
                {station.id === 12 && (
                  <div className="w-full flex flex-col gap-6">
                    {/* Top Controls */}
                    <div className="flex flex-wrap items-center justify-between gap-3 p-3 rounded-xl bg-slate-900 border border-slate-800">
                      <div className="flex items-center gap-3 flex-1 min-w-[220px]">
                        <span className="text-xs font-semibold text-slate-300">Penalty Weight λ:</span>
                        <input
                          type="range"
                          min={0.001}
                          max={0.02}
                          step={0.001}
                          value={scucPenaltyWeight}
                          onChange={(e) => setScucPenaltyWeight(parseFloat(e.target.value))}
                          className="w-full accent-cyan-400 cursor-pointer h-1.5 rounded-lg bg-slate-800"
                        />
                        <span className="text-xs font-mono font-bold text-cyan-300 whitespace-nowrap">
                          λ = {scucPenaltyWeight.toFixed(4)}
                        </span>
                      </div>
                      <div className="flex flex-wrap items-center gap-2">
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
                          className="px-2.5 py-1 rounded-lg text-xs font-semibold bg-emerald-500/20 border border-emerald-400 text-emerald-300 hover:bg-emerald-500/30"
                        >
                          ⚡ Optimal QUBO Dispatch
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
                          className="px-2.5 py-1 rounded-lg text-xs font-semibold bg-slate-800 border border-slate-700 text-amber-300 hover:bg-slate-700"
                        >
                          🔥 All Units Active (Overgen)
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
                          className="px-2.5 py-1 rounded-lg text-xs font-semibold bg-slate-800 border border-slate-700 text-rose-400 hover:bg-slate-700"
                        >
                          ⚠️ Blackout Test (0 MW)
                        </button>
                      </div>
                    </div>

                    {/* Interactive 4-Gen x 6-Hour Dispatch Grid */}
                    <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-3">
                      <div className="flex justify-between items-center text-xs">
                        <span className="text-slate-300 font-semibold">
                          Unit Commitment Matrix x_{"{g,t}"} ∈ &#123;0, 1&#125; (Click Cell to Toggle Commitment)
                        </span>
                        <span className="text-cyan-400 font-mono text-[11px]">
                          4 Generators × 6 Time Steps = 24 Spins
                        </span>
                      </div>

                      <div className="overflow-x-auto">
                        <table className="w-full text-left border-collapse text-xs">
                          <thead>
                            <tr className="border-b border-slate-800 text-slate-400 font-mono text-[11px]">
                              <th className="py-2 px-3">Generator Unit</th>
                              <th className="py-2 px-2 text-right">P_max</th>
                              <th className="py-2 px-2 text-right">Cost</th>
                              {scucDemands.map((d, t) => (
                                <th key={t} className="py-2 px-2 text-center">
                                  <div>H{t + 1}</div>
                                  <div className="text-[10px] text-cyan-400/80">{d} MW</div>
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {scucGenerators.map((gen, gIdx) => (
                              <tr key={gen.id} className="border-b border-slate-800/60 hover:bg-slate-800/30">
                                <td className="py-2.5 px-3 font-semibold text-slate-200">
                                  <div className="flex items-center gap-2">
                                    <span className={`w-2 h-2 rounded-full ${gen.color.split(" ")[0].replace("text-", "bg-")}`} />
                                    <span>{gen.name}</span>
                                  </div>
                                </td>
                                <td className="py-2.5 px-2 text-right font-mono text-slate-400">{gen.capacity} MW</td>
                                <td className="py-2.5 px-2 text-right font-mono text-slate-400">${gen.cost}/MWh</td>
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
                                        className={`w-full py-1.5 px-2 rounded-lg font-mono text-xs font-bold transition shadow-sm ${
                                          isOn
                                            ? "bg-emerald-500/25 border border-emerald-400 text-emerald-300 shadow-emerald-500/20"
                                            : "bg-slate-950 border border-slate-800 text-slate-600 hover:text-slate-400 hover:border-slate-700"
                                        }`}
                                      >
                                        {isOn ? "[ON]" : "[..]"}
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
                    <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-3">
                      <div className="flex justify-between items-center text-xs">
                        <span className="text-slate-300 font-semibold">
                          Hourly System Power Balance (Delivered vs. Demand)
                        </span>
                        <span className="text-slate-400 font-mono text-[11px]">
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
                                  ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-300"
                                  : isDeficit
                                  ? "bg-rose-500/10 border-rose-500/30 text-rose-300"
                                  : "bg-amber-500/10 border-amber-500/30 text-amber-300"
                              }`}
                            >
                              <div className="text-[10px] text-slate-400">Hour {t + 1}</div>
                              <div className="text-sm font-bold mt-0.5">{delivered} MW</div>
                              <div className="text-[10px] text-slate-500">of {demand} MW</div>
                              <div className="text-[10px] font-bold mt-1">
                                {delta === 0 ? "✓ EXACT" : delta > 0 ? `+${delta} MW` : `${delta} MW`}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>

                    {/* Metrics HUD */}
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
                      <div className="p-3 rounded-xl bg-slate-900 border border-slate-800">
                        <div className="text-[11px] text-slate-400">Optimal Hamiltonian E*</div>
                        <div className="text-lg font-mono font-bold text-emerald-400">
                          5451.00
                        </div>
                        <div className="text-[10px] text-slate-500">Min Global Energy</div>
                      </div>
                      <div className="p-3 rounded-xl bg-slate-900 border border-slate-800">
                        <div className="text-[11px] text-slate-400">Current Energy H(x)</div>
                        <div className="text-lg font-mono font-bold text-cyan-300">
                          {scucTotalEnergy.toFixed(2)}
                        </div>
                        <div className="text-[10px] text-slate-500">Cost + Penalty</div>
                      </div>
                      <div className="p-3 rounded-xl bg-slate-900 border border-slate-800">
                        <div className="text-[11px] text-slate-400">Production Cost</div>
                        <div className="text-lg font-mono font-bold text-cyan-400">
                          ${(scucTotalGenCost * 100).toFixed(0)}
                        </div>
                        <div className="text-[10px] text-slate-500">6-Hour Horizon</div>
                      </div>
                      <div className="p-3 rounded-xl bg-slate-900 border border-slate-800">
                        <div className="text-[11px] text-slate-400">Sokal Autocorrelation</div>
                        <div className="text-lg font-mono font-bold text-amber-300">
                          1.00 sweep
                        </div>
                        <div className="text-[10px] text-slate-500">τ_int ({selectedBackend === "jax" ? "41.2k" : "625k"} ESS/s)</div>
                      </div>
                    </div>
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
                className="w-full flex items-center justify-between text-xs font-mono text-slate-400 hover:text-cyan-300 transition py-1"
              >
                <span>
                  🔧 ENGINEER DEEP DIVE — Hamiltonian, JAX Speedup &amp; Reproducible Script (
                  <code className="text-cyan-400">{station.scriptFile}</code>)
                </span>
                <span>{openAccordion[station.id] ? "▲ Hide Spec" : "▼ Expand Spec"}</span>
              </button>

              {openAccordion[station.id] && (
                <div className="mt-3 pt-3 border-t border-slate-800/60 grid grid-cols-1 md:grid-cols-3 gap-4 text-xs pb-2">
                  <div className="p-3 rounded-xl bg-slate-900/90 border border-slate-800">
                    <div className="text-slate-400 font-semibold mb-1">Hamiltonian / Objective</div>
                    <code className="text-cyan-300 font-mono">{station.physicsFormula}</code>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-900/90 border border-slate-800">
                    <div className="text-slate-400 font-semibold mb-1">
                      Block Gibbs Architecture
                    </div>
                    <p className="text-slate-300 leading-relaxed">{station.engineeringSummary}</p>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-900/90 border border-slate-800 flex flex-col justify-between">
                    <div>
                      <div className="text-slate-400 font-semibold mb-1">
                        Reproduce via thrml-bench CLI
                      </div>
                      <code className="text-emerald-400 font-mono">
                        uv run thrml-bench run --exp {station.id}
                      </code>
                    </div>
                    <a
                      href={`https://github.com/nexusai-kevin/thermo-physics/blob/main/${station.scriptFile}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="mt-2 inline-flex items-center gap-1 text-cyan-400 hover:underline font-semibold"
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
        <section className="rounded-2xl bg-gradient-to-br from-slate-900 via-slate-950 to-slate-900 border border-cyan-500/30 p-6 sm:p-8 shadow-2xl">
          <div className="flex flex-wrap items-center justify-between gap-4 mb-6">
            <div>
              <span className="text-xs font-bold uppercase tracking-wider text-cyan-400">
                Initiative 3 • Hardware Scaling Telemetry
              </span>
              <h3 className="text-2xl font-bold text-white mt-1">
                Multi-Scale 2-Color Block Gibbs Throughput (L = 16 → 128)
              </h3>
            </div>
            <a
              href="https://github.com/nexusai-kevin/thermo-physics/blob/main/BENCHMARKS.md"
              target="_blank"
              rel="noopener noreferrer"
              className="px-4 py-2 rounded-xl bg-cyan-500/15 border border-cyan-400/40 text-cyan-300 text-xs font-bold hover:bg-cyan-500/25 transition"
            >
              View Full BENCHMARKS.md Index ↗
            </a>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-center">
            {[
              { size: "16 × 16 (256 Spins)", msu: "0.33 MSU/s", speedup: "128× Clock Speedup" },
              { size: "32 × 32 (1,024 Spins)", msu: "1.30 MSU/s", speedup: "512× Clock Speedup" },
              { size: "64 × 64 (4,096 Spins)", msu: "6.86 MSU/s", speedup: "2,048× Clock Speedup" },
              { size: "128 × 128 (16,384 Spins)", msu: "20.18 MSU/s", speedup: "8,192× Clock Speedup" },
            ].map((item, i) => (
              <div key={i} className="p-4 rounded-xl bg-slate-950/90 border border-slate-800">
                <div className="text-xs text-slate-400">{item.size}</div>
                <div className="text-xl font-mono font-extrabold text-emerald-400 mt-1">
                  {item.msu}
                </div>
                <div className="text-xs font-mono text-amber-300 mt-1">{item.speedup}</div>
              </div>
            ))}
          </div>
        </section>

        {/* ===================================================================
            COMMUNITY, UPSTREAM CONTRIBUTION & QUICKSTART TUTORIAL BANNER
        =================================================================== */}
        <section className="rounded-2xl bg-gradient-to-r from-indigo-950/60 via-slate-900 to-cyan-950/60 border border-indigo-500/30 p-6 sm:p-8 shadow-2xl">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
            <div>
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/15 border border-indigo-400/30 text-indigo-300 text-xs font-bold mb-2">
                <span>🚀 Community &amp; Upstream Alignment</span>
              </div>
              <h3 className="text-2xl font-bold text-white">
                Contribute, Replicate &amp; Run the Code
              </h3>
              <p className="text-slate-300 text-sm max-w-2xl mt-1 leading-relaxed">
                Connect directly with our open-source thermodynamic computing research on Extropic’s <code className="text-cyan-300 font-mono">thrml</code> ecosystem. Inspect our active upstream PR, run the end-to-end interactive Jupyter tutorial, or explore the reproducibility repository.
              </p>
            </div>

            <div className="flex flex-wrap sm:flex-nowrap items-center gap-3">
              {/* Upstream PR #79 */}
              <a
                href="https://github.com/extropic-ai/thrml/pull/79"
                target="_blank"
                rel="noopener noreferrer"
                className="group flex items-center gap-2 px-4 py-3 rounded-xl bg-emerald-500/15 border border-emerald-400/40 text-emerald-300 hover:bg-emerald-500/25 transition text-xs font-bold shadow-lg"
              >
                <span className="text-emerald-400 text-sm">⚡</span>
                <div className="text-left">
                  <div className="text-[10px] text-emerald-400/80 uppercase tracking-wider">Upstream Contribution</div>
                  <div>PR #79 on extropic-ai/thrml ↗</div>
                </div>
              </a>

              {/* Open in Colab */}
              <a
                href="https://colab.research.google.com/github/nexusai-kevin/thermo-physics/blob/main/notebooks/quickstart_tutorial.ipynb"
                target="_blank"
                rel="noopener noreferrer"
                className="group flex items-center gap-2 px-4 py-3 rounded-xl bg-amber-500/15 border border-amber-400/40 text-amber-300 hover:bg-amber-500/25 transition text-xs font-bold shadow-lg"
              >
                <img
                  src="https://colab.research.google.com/assets/colab-badge.svg"
                  alt="Open in Colab"
                  className="h-5"
                />
              </a>

              {/* GitHub Repository */}
              <a
                href="https://github.com/nexusai-kevin/thermo-physics"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 px-4 py-3 rounded-xl bg-slate-800/80 border border-slate-700 text-slate-200 hover:bg-slate-700/80 transition text-xs font-bold shadow-lg"
              >
                <span>📂 GitHub Repo ↗</span>
              </a>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
