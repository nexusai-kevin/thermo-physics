"use client";

import React, { useState, useEffect, useCallback, useMemo, useRef } from "react";

// ============================================================================
// TYPES & BENCHMARK METADATA (EXPERIMENTS 1–9)
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
];

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
    // Play a special phase-transition chime when crossing Onsager's critical point (0.4407)
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
  }, [globalBeta, rivalryEdges, assetReturns, assetRisks, quboLambda, tspCoords]);

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

  // --- Copy / Post to X Quick-Share Text ---
  const shareText = useMemo(
    () =>
      `Just ran the 9-station Thermodynamic Computing Arcade at β=${globalBeta.toFixed(
        2
      )} (|M|=${magnet2D}, Max-Cut=${Math.round(
        (happyRivalries / rivalryEdges.length) * 100
      )}%, 20.18M spin-updates/s in JAX) powered by @extropic's thrml! ⚡❄️\n\ncc @trevormccrt1 @beffjezos\nLive lab: https://getnexusaisolutions.com/thermo-lab\nRepo: https://github.com/nexusai-kevin/thermo-physics`,
    [globalBeta, magnet2D, happyRivalries, rivalryEdges.length]
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
              9 Stations • 20.18M Spin-Updates/s
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
            <span>⚡ 9-Station Open-Source Thermodynamic Computing Suite • Extropic thrml v0.1.4</span>
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
            Slide the temperature dial above (with Web Audio SFX enabled!) to watch 9 physical
            energy landscapes—from discrete Ising magnets to continuous analog wells and 10-city TSP
            routers—settle into optimal ground states automatically.
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
              🔥 Melt All 9 Stations (β = 0.20)
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
          9 INTERACTIVE STATIONS GRID
      =================================================================== */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-12 space-y-10">
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
                      <div className="text-center space-y-1.5">
                        <span className="text-xs text-slate-400">Noisy Input (Clamped)</span>
                        <div
                          className="grid gap-1 p-2 bg-slate-900 rounded-xl border border-slate-800"
                          style={{ gridTemplateColumns: "repeat(8, minmax(0, 1fr))" }}
                        >
                          {noisyDigit.map((v, i) => (
                            <div
                              key={i}
                              className={`w-4 h-4 rounded-[2px] ${
                                v === 1 ? "bg-white" : "bg-slate-950"
                              }`}
                            />
                          ))}
                        </div>
                      </div>

                      <span className="text-cyan-400 font-bold text-xl">→</span>

                      <div className="text-center space-y-1.5">
                        <span className="text-xs text-emerald-400 font-semibold">
                          Restored (Err: {denoisedError}%)
                        </span>
                        <div
                          className="grid gap-1 p-2 bg-slate-900 rounded-xl border border-emerald-500/40"
                          style={{ gridTemplateColumns: "repeat(8, minmax(0, 1fr))" }}
                        >
                          {denoisedDigit.map((v, i) => (
                            <div
                              key={i}
                              className={`w-4 h-4 rounded-[2px] transition-colors ${
                                v === 1 ? "bg-emerald-400" : "bg-slate-950"
                              }`}
                            />
                          ))}
                        </div>
                      </div>
                    </div>

                    <div className="flex flex-col gap-2.5 w-full sm:w-auto">
                      <button
                        onClick={handleDenoiseStep}
                        className="px-4 py-2.5 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-bold text-xs shadow-lg shadow-emerald-500/20 transition"
                      >
                        ✨ Clean With Physics (Relax Latent Spins)
                      </button>
                      <button
                        onClick={handleAddNoise}
                        className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium border border-slate-700 transition"
                      >
                        💥 Splash Fresh 22% Pixel Noise
                      </button>
                    </div>
                  </div>
                )}

                {/* STATION 6 WIDGET */}
                {station.id === 6 && (
                  <div className="w-full flex flex-col sm:flex-row items-center justify-around gap-6">
                    <div className="text-center space-y-2">
                      <div className="text-xs text-slate-400">
                        Top Half = Clamped Cue • Bottom Half = Free Spins
                      </div>
                      <div
                        className="grid gap-1 p-2.5 bg-slate-900 rounded-xl border border-cyan-500/30"
                        style={{ gridTemplateColumns: "repeat(8, minmax(0, 1fr))" }}
                      >
                        {rbmState.map((v, i) => (
                          <div
                            key={i}
                            className={`w-4 h-4 sm:w-5 sm:h-5 rounded-[2px] transition-colors ${
                              v === 1
                                ? i < 32
                                  ? "bg-cyan-300"
                                  : "bg-emerald-400"
                                : "bg-slate-950"
                            }`}
                          />
                        ))}
                      </div>
                    </div>

                    <div className="space-y-3 max-w-xs w-full">
                      <div className="text-xs font-mono text-cyan-300">
                        Memory Recall Overlap: <strong>{rbmOverlap}%</strong>
                      </div>
                      <div className="grid grid-cols-3 gap-2">
                        {["➕ Plus", "✖️ Cross", "⬜ Square"].map((label, idx) => (
                          <button
                            key={idx}
                            onClick={() => handleSelectPrototype(idx)}
                            className={`py-2 px-2 rounded-lg text-xs font-semibold border transition ${
                              selectedProtoIdx === idx
                                ? "bg-cyan-500/20 border-cyan-400 text-cyan-200"
                                : "bg-slate-900 border-slate-800 text-slate-400 hover:bg-slate-800"
                            }`}
                          >
                            {label}
                          </button>
                        ))}
                      </div>
                      <button
                        onClick={handleRecallMemory}
                        className="w-full py-2.5 px-4 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-xs shadow-lg shadow-cyan-500/25 transition"
                      >
                        🧠 Fire Hidden RBM Neurons (Restore 100%)
                      </button>
                    </div>
                  </div>
                )}

                {/* STATION 7 WIDGET */}
                {station.id === 7 && (
                  <div className="w-full space-y-4">
                    <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
                      <div className="flex items-center gap-3">
                        <span className="font-semibold text-white">
                          Selected Chests:{" "}
                          <span
                            className={`font-mono px-2 py-0.5 rounded ${
                              quboCount === 5
                                ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40"
                                : "bg-rose-500/20 text-rose-300 border border-rose-500/40"
                            }`}
                          >
                            {quboCount} / 5 Target
                          </span>
                        </span>
                        <span className="text-slate-400 font-mono">
                          Portfolio Return: <strong className="text-cyan-300">{quboTotalReturn}%</strong>
                        </span>
                      </div>

                      <div className="flex items-center gap-2">
                        <span className="text-slate-400">Budget Penalty λ:</span>
                        <input
                          type="range"
                          min={0.1}
                          max={2.0}
                          step={0.05}
                          value={quboLambda}
                          onChange={(e) => setQuboLambda(parseFloat(e.target.value))}
                          className="w-24 accent-emerald-400 cursor-pointer"
                        />
                        <span className="font-mono text-emerald-300">{quboLambda.toFixed(2)}</span>
                      </div>
                    </div>

                    <div className="grid grid-cols-5 sm:grid-cols-10 gap-2 pt-1">
                      {selectedAssets.map((sel, idx) => (
                        <button
                          key={idx}
                          onClick={() => {
                            playArcadeTone(sel ? 300 : 600, 30);
                            setSelectedAssets((arr) =>
                              arr.map((v, i) => (i === idx ? (v === 1 ? 0 : 1) : v))
                            );
                          }}
                          className={`p-2 rounded-xl border text-center transition-all ${
                            sel === 1
                              ? "bg-emerald-500/25 border-emerald-400 text-emerald-200 shadow-md shadow-emerald-500/15"
                              : "bg-slate-900 border-slate-800 text-slate-400 hover:border-slate-700"
                          }`}
                        >
                          <div className="text-[10px] font-mono opacity-75">#{idx}</div>
                          <div className="text-xs font-bold">{assetReturns[idx]}%</div>
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {/* STATION 8 WIDGET: CONTINUOUS DOUBLE-WELL OSCILLATORS */}
                {station.id === 8 && (
                  <div className="w-full space-y-4">
                    <div className="flex items-center justify-between text-xs text-slate-400 px-2">
                      <span>Left Well (x = -1.0)</span>
                      <span className="text-amber-300 font-mono">Barrier Hill (x = 0.0)</span>
                      <span>Right Well (x = +1.0)</span>
                    </div>
                    <div className="space-y-2 px-2">
                      {oscillators.map((xVal, idx) => {
                        const pct = Math.max(2, Math.min(98, ((xVal + 1.65) / 3.3) * 100));
                        const inRightWell = xVal >= 0;
                        return (
                          <div
                            key={idx}
                            className="relative h-4 bg-slate-900 rounded-full border border-slate-800 overflow-hidden"
                          >
                            {/* Center barrier marker */}
                            <div className="absolute left-1/2 top-0 bottom-0 w-0.5 bg-amber-400/40" />
                            {/* Well attractors at -1 and +1 */}
                            <div className="absolute left-[19.7%] top-0 bottom-0 w-0.5 bg-rose-400/25" />
                            <div className="absolute left-[80.3%] top-0 bottom-0 w-0.5 bg-cyan-400/25" />
                            <div
                              style={{ left: `${pct}%` }}
                              className={`absolute top-0.5 -translate-x-1/2 w-3 h-3 rounded-full transition-all duration-150 ${
                                inRightWell
                                  ? "bg-cyan-400 shadow-md shadow-cyan-400/60"
                                  : "bg-rose-400 shadow-md shadow-rose-400/60"
                              }`}
                            />
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

                {/* STATION 9 WIDGET: 10-CITY TSP ROUTER */}
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
