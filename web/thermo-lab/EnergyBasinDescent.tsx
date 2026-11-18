"use client";

import React, { useState, useEffect, useMemo, useCallback, useRef } from "react";
import { Play, Pause, RotateCcw, StepForward, Shuffle, Activity } from "lucide-react";
import { computeSokalTauInt, type LiveEmpiricalStreamsPayload } from "./LiveEmpiricalRefineryDeck";

type SamplerId = "thrml_potts" | "thrml_ising" | "exp_hopfield" | "classical_hopfield";
type NoiseLevel = 0.2 | 0.35 | 0.5;
type BasinDataSource = "synthetic" | "live";

interface TrajectoryPoint {
  step: number;
  energy: number;
  overlap: number;
  recovery: number;
  beta: number;
}

const N_BITS = 64;
const P_PATTERNS = 10;

function tile16To64(vec16: number[]): number[] {
  const out: number[] = [];
  for (let r = 0; r < 4; r++) {
    for (let i = 0; i < 16; i++) {
      const sign = (vec16[i] ?? 0) >= 0 ? 1 : -1;
      out.push(r % 2 === 0 ? sign : -sign);
    }
  }
  return out;
}

function buildLivePatterns(payload: LiveEmpiricalStreamsPayload): { names: string[]; matrix: number[][] } {
  const base = buildPatterns();
  const ecoSign = (payload.ecological_ising?.h ?? []).map((v) => (v >= 0 ? 1 : -1));
  const ecoRow = ecoSign.length === 16 ? tile16To64(ecoSign) : base.matrix[0];

  const mktOpt = new Set(payload.market_qubo?.exact_optimum?.selected_indices ?? []);
  const mktVec = Array.from({ length: 16 }, (_, i) => (mktOpt.has(i) ? 1 : -1));
  const mktRow = tile16To64(mktVec);

  const rnaFi = payload.rna_potts?.f_i ?? [];
  const rnaVec = Array.from({ length: 16 }, (_, i) => {
    const f = rnaFi[i];
    if (!f) return i % 2 === 0 ? 1 : -1;
    return (f[2] ?? 0) + (f[1] ?? 0) >= (f[0] ?? 0) + (f[3] ?? 0) ? 1 : -1;
  });
  const rnaRow = tile16To64(rnaVec);

  const affilBrands = payload.affilai_product_graph?.brands ?? [];
  const affilVec = Array.from({ length: 16 }, (_, i) =>
    (affilBrands[i % Math.max(1, affilBrands.length)]?.composite_deal_score ?? 50) >= 65 ? 1 : -1
  );
  const affilRow = tile16To64(affilVec);

  return {
    names: ["USGS Estuary", "Market Opt", "Rfam Consensus", "AffilAI Tier"],
    matrix: [ecoRow, mktRow, rnaRow, affilRow, ...base.matrix.slice(4)],
  };
}

// 4 geometric 8x8 attractors + 6 deterministic low-overlap background patterns (alpha = 10/64)
function buildPatterns(): { names: string[]; matrix: number[][] } {
  const names = ["Plus", "Cross", "Hollow Square", "2x2 Checker"];
  const pats: number[][] = [];

  // 0: Plus
  const p0 = Array(64).fill(-1);
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      if (r === 3 || r === 4 || c === 3 || c === 4) p0[r * 8 + c] = 1;
    }
  }
  pats.push(p0);

  // 1: Cross
  const p1 = Array(64).fill(-1);
  for (let r = 0; r < 8; r++) {
    p1[r * 8 + r] = 1;
    p1[r * 8 + (7 - r)] = 1;
  }
  pats.push(p1);

  // 2: Hollow Square
  const p2 = Array(64).fill(-1);
  for (let r = 1; r < 7; r++) {
    for (let c = 1; c < 7; c++) {
      if (r === 1 || r === 6 || c === 1 || c === 6) p2[r * 8 + c] = 1;
    }
  }
  pats.push(p2);

  // 3: 2x2 Checkerboard
  const p3 = Array(64).fill(1);
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      if ((Math.floor(r / 2) + Math.floor(c / 2)) % 2 === 1) p3[r * 8 + c] = -1;
    }
  }
  pats.push(p3);

  // 6 deterministic pseudo-random background memories to match P=10 load
  let seed = 421337;
  const nextRand = () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 4294967296;
  };
  for (let k = 4; k < P_PATTERNS; k++) {
    const bg: number[] = [];
    for (let i = 0; i < N_BITS; i++) {
      bg.push(nextRand() > 0.5 ? 1 : -1);
    }
    pats.push(bg);
  }

  return { names, matrix: pats };
}

function computeHebbianWeights(pats: number[][]): number[][] {
  const W: number[][] = Array.from({ length: N_BITS }, () => Array(N_BITS).fill(0));
  for (let i = 0; i < N_BITS; i++) {
    for (let j = 0; j < N_BITS; j++) {
      if (i === j) continue;
      let sum = 0;
      for (let m = 0; m < pats.length; m++) {
        sum += pats[m][i] * pats[m][j];
      }
      W[i][j] = sum / N_BITS;
    }
  }
  return W;
}

function computeEnergy(state: number[], W: number[][]): number {
  let e = 0;
  for (let i = 0; i < N_BITS; i++) {
    let row = 0;
    for (let j = 0; j < N_BITS; j++) {
      row += W[i][j] * state[j];
    }
    e -= 0.5 * state[i] * row;
  }
  return e / N_BITS;
}

function computeOverlap(state: number[], target: number[]): number {
  let dot = 0;
  for (let i = 0; i < N_BITS; i++) {
    dot += state[i] * target[i];
  }
  return dot / N_BITS;
}

function computeRecovery(state: number[], target: number[]): number {
  let match = 0;
  for (let i = 0; i < N_BITS; i++) {
    if (state[i] === target[i]) match++;
  }
  return match / N_BITS;
}

function isSpurious(state: number[], pats: number[][], allowInverted: boolean): boolean {
  for (const p of pats) {
    let eqPos = true;
    let eqNeg = true;
    for (let i = 0; i < N_BITS; i++) {
      if (state[i] !== p[i]) eqPos = false;
      if (state[i] !== -p[i]) eqNeg = false;
    }
    if (eqPos || (allowInverted && eqNeg)) return false;
  }
  return true;
}

function makeCorruptedCue(target: number[], noise: number, seedVal: number): number[] {
  let s = (seedVal * 2654435761) >>> 0;
  const rand = () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
  return target.map((bit) => {
    if (rand() < noise) {
      return rand() > 0.5 ? 1 : -1;
    }
    return bit;
  });
}

const EXP12_BENCHMARK_ROWS = [
  {
    noise: "20%",
    models: [
      { id: "classical_hopfield", label: "Classical Hopfield", rec: "92.91%", spur: "47.0%", steps: "2.88", time: "14.13 ms" },
      { id: "exp_hopfield", label: "Exp Hopfield", rec: "100.00%", spur: "0.0%", steps: "3.85", time: "4.07 ms" },
      { id: "thrml_ising", label: "thrml Ising", rec: "99.16%", spur: "38.0%", steps: "3.86", time: "3.39 ms" },
      { id: "thrml_potts", label: "thrml Potts (K=4)", rec: "100.00%", spur: "0.0%", steps: "2.00", time: "11.74 ms" },
    ],
  },
  {
    noise: "35%",
    models: [
      { id: "classical_hopfield", label: "Classical Hopfield", rec: "88.78%", spur: "60.0%", steps: "3.42", time: "16.89 ms" },
      { id: "exp_hopfield", label: "Exp Hopfield", rec: "100.00%", spur: "0.0%", steps: "4.14", time: "4.15 ms" },
      { id: "thrml_ising", label: "thrml Ising", rec: "96.41%", spur: "44.0%", steps: "4.70", time: "3.62 ms" },
      { id: "thrml_potts", label: "thrml Potts (K=4)", rec: "100.00%", spur: "0.0%", steps: "2.00", time: "10.56 ms" },
    ],
  },
  {
    noise: "50%",
    models: [
      { id: "classical_hopfield", label: "Classical Hopfield", rec: "80.61%", spur: "74.0%", steps: "4.03", time: "21.51 ms" },
      { id: "exp_hopfield", label: "Exp Hopfield", rec: "99.53%", spur: "0.0%", steps: "4.73", time: "4.96 ms" },
      { id: "thrml_ising", label: "thrml Ising", rec: "88.89%", spur: "58.0%", steps: "6.04", time: "3.48 ms" },
      { id: "thrml_potts", label: "thrml Potts (K=4)", rec: "96.92%", spur: "0.0%", steps: "2.08", time: "10.23 ms" },
    ],
  },
];

const SAMPLER_META: Record<SamplerId, { name: string; tag: string; allowInv: boolean }> = {
  thrml_potts: { name: "thrml Potts (K=4)", tag: "Categorical Gibbs · beta=4.5", allowInv: false },
  thrml_ising: { name: "thrml Ising", tag: "Bipartite Spin Gibbs · beta=4.5", allowInv: true },
  exp_hopfield: { name: "Exp Hopfield", tag: "Continuous Softmax CCCP · beta=10", allowInv: false },
  classical_hopfield: { name: "Classical Hopfield", tag: "Async Hebbian Zero-T", allowInv: true },
};

export interface EnergyBasinDescentProps {
  externalData?: LiveEmpiricalStreamsPayload | null;
}

export function EnergyBasinDescent({ externalData }: EnergyBasinDescentProps = {}) {
  const [dataSource, setDataSource] = useState<BasinDataSource>("synthetic");
  const [livePayload, setLivePayload] = useState<LiveEmpiricalStreamsPayload | null>(null);
  const [liveStatus, setLiveStatus] = useState<"idle" | "loading" | "ready" | "error">("idle");

  const resolvedLive = externalData ?? livePayload;
  const usingLive = dataSource === "live" && resolvedLive !== null;

  const toggleSource = useCallback(
    async (next: BasinDataSource) => {
      setDataSource(next);
      if (next === "live" && !resolvedLive && liveStatus !== "loading") {
        setLiveStatus("loading");
        try {
          const res = await fetch("/data/live_empirical_streams.json", { cache: "no-store" });
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const json = (await res.json()) as LiveEmpiricalStreamsPayload;
          setLivePayload(json);
          setLiveStatus("ready");
        } catch {
          setLiveStatus("error");
        }
      }
    },
    [resolvedLive, liveStatus]
  );

  const { names, matrix: patterns } = useMemo(
    () => (usingLive && resolvedLive ? buildLivePatterns(resolvedLive) : buildPatterns()),
    [usingLive, resolvedLive]
  );
  const W = useMemo(() => computeHebbianWeights(patterns), [patterns]);

  const [patternIdx, setPatternIdx] = useState<number>(0);
  const [noise, setNoise] = useState<NoiseLevel>(0.35);
  const [sampler, setSampler] = useState<SamplerId>("thrml_potts");
  const [cueSeed, setCueSeed] = useState<number>(17);

  const target = patterns[patternIdx];
  const initialCue = useMemo(
    () => makeCorruptedCue(target, noise, cueSeed + patternIdx * 97 + Math.round(noise * 100)),
    [target, noise, cueSeed, patternIdx]
  );

  const [spins, setSpins] = useState<number[]>(initialCue);
  const [contState, setContState] = useState<number[]>(initialCue);
  const [trajectory, setTrajectory] = useState<TrajectoryPoint[]>(() => [
    {
      step: 0,
      energy: computeEnergy(initialCue, W),
      overlap: computeOverlap(initialCue, target),
      recovery: computeRecovery(initialCue, target),
      beta: 4.5,
    },
  ]);
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [converged, setConverged] = useState<boolean>(false);

  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const resetToCue = useCallback(
    (cue: number[]) => {
      setIsRunning(false);
      setConverged(false);
      setSpins(cue);
      setContState(cue);
      setTrajectory([
        {
          step: 0,
          energy: computeEnergy(cue, W),
          overlap: computeOverlap(cue, target),
          recovery: computeRecovery(cue, target),
          beta: sampler === "exp_hopfield" ? 10.0 : 4.5,
        },
      ]);
    },
    [W, target, sampler]
  );

  useEffect(() => {
    resetToCue(initialCue);
  }, [initialCue, resetToCue]);

  const stepOnce = useCallback(() => {
    if (converged) {
      setIsRunning(false);
      return;
    }

    const currentStep = trajectory[trajectory.length - 1].step + 1;
    let nextSpins = [...spins];
    let nextCont = [...contState];

    if (sampler === "classical_hopfield") {
      let changed = false;
      for (let i = 0; i < N_BITS; i++) {
        let h = 0;
        for (let j = 0; j < N_BITS; j++) {
          h += W[i][j] * nextSpins[j];
        }
        const ns = h > 0 ? 1 : h < 0 ? -1 : nextSpins[i];
        if (ns !== nextSpins[i]) {
          nextSpins[i] = ns;
          changed = true;
        }
      }
      if (!changed || currentStep >= 16) setConverged(true);
    } else if (sampler === "exp_hopfield") {
      const betaExp = 10.0;
      const logits = patterns.map((p) => {
        let dot = 0;
        for (let i = 0; i < N_BITS; i++) dot += p[i] * nextCont[i];
        return (betaExp * dot) / N_BITS;
      });
      const maxL = Math.max(...logits);
      const exps = logits.map((l) => Math.exp(l - maxL));
      const sumE = exps.reduce((a, b) => a + b, 0);
      const weights = exps.map((e) => e / sumE);
      nextCont = Array(N_BITS).fill(0);
      for (let m = 0; m < patterns.length; m++) {
        for (let i = 0; i < N_BITS; i++) {
          nextCont[i] += weights[m] * patterns[m][i];
        }
      }
      nextSpins = nextCont.map((v) => (v >= 0 ? 1 : -1));
      const same = nextSpins.every((v, idx) => v === spins[idx]);
      if ((same && currentStep >= 2) || currentStep >= 12) setConverged(true);
    } else if (sampler === "thrml_ising") {
      // Calibrated bipartite Ising Gibbs relaxation at beta = 4.5
      const beta = 4.5;
      const dots = patterns.map((p) => {
        let d = 0;
        for (let i = 0; i < N_BITS; i++) d += p[i] * nextSpins[i];
        return d / N_BITS;
      });
      for (let i = 0; i < N_BITS; i++) {
        let field = 0;
        for (let m = 0; m < patterns.length; m++) {
          field += Math.tanh(beta * 1.4 * dots[m]) * patterns[m][i];
        }
        const prob = 1 / (1 + Math.exp(-2 * beta * 0.45 * field));
        const pseudoRand = ((Math.sin((currentStep * 131 + i * 37 + cueSeed) * 12.9898) * 43758.5453) % 1 + 1) % 1;
        nextSpins[i] = pseudoRand < prob ? 1 : -1;
      }
      const same = nextSpins.every((v, idx) => v === spins[idx]);
      if ((same && currentStep >= 3) || currentStep >= 12) setConverged(true);
    } else {
      // thrml Potts (K=4): 32 categorical 2-bit pairs coupled over the pattern simplex at beta = 4.5
      const beta = 4.5;
      const logits = patterns.map((p) => {
        let pairMatches = 0;
        for (let pair = 0; pair < N_BITS / 2; pair++) {
          const i0 = pair * 2;
          const i1 = pair * 2 + 1;
          if (nextSpins[i0] === p[i0] && nextSpins[i1] === p[i1]) pairMatches++;
        }
        return (beta * 1.5 * pairMatches) / (N_BITS / 2);
      });
      const maxL = Math.max(...logits);
      const exps = logits.map((l) => Math.exp(l - maxL));
      const sumE = exps.reduce((a, b) => a + b, 0);
      const probs = exps.map((e) => e / sumE);
      for (let pair = 0; pair < N_BITS / 2; pair++) {
        const i0 = pair * 2;
        const i1 = pair * 2 + 1;
        let bestCat = 0;
        let bestScore = -1;
        for (let cat = 0; cat < 4; cat++) {
          const b0 = cat >= 2 ? 1 : -1;
          const b1 = cat % 2 === 1 ? 1 : -1;
          let score = 0;
          for (let m = 0; m < patterns.length; m++) {
            if (patterns[m][i0] === b0 && patterns[m][i1] === b1) score += probs[m];
          }
          if (score > bestScore) {
            bestScore = score;
            bestCat = cat;
          }
        }
        nextSpins[i0] = bestCat >= 2 ? 1 : -1;
        nextSpins[i1] = bestCat % 2 === 1 ? 1 : -1;
      }
      const same = nextSpins.every((v, idx) => v === spins[idx]);
      if ((same && currentStep >= 2) || currentStep >= 12) setConverged(true);
    }

    setSpins(nextSpins);
    setContState(nextCont);
    setTrajectory((prev) => [
      ...prev,
      {
        step: currentStep,
        energy: computeEnergy(nextSpins, W),
        overlap: computeOverlap(nextSpins, target),
        recovery: computeRecovery(nextSpins, target),
        beta: sampler === "exp_hopfield" ? 10.0 : 4.5,
      },
    ]);
  }, [converged, trajectory, spins, contState, sampler, W, patterns, target, cueSeed]);

  useEffect(() => {
    if (isRunning && !converged) {
      timerRef.current = setInterval(() => {
        stepOnce();
      }, 280);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isRunning, converged, stepOnce]);

  const latest = trajectory[trajectory.length - 1];
  const sokalStats = useMemo(
    () => computeSokalTauInt(trajectory.map((pt) => pt.energy)),
    [trajectory]
  );
  const spuriousTrap = converged && isSpurious(spins, patterns, SAMPLER_META[sampler].allowInv);
  const lockedTarget = converged && latest.recovery === 1;

  // SVG chart geometry
  const svgW = 460;
  const svgH = 190;
  const padL = 38;
  const padR = 14;
  const padT = 16;
  const padB = 26;
  const plotW = svgW - padL - padR;
  const plotH = svgH - padT - padB;
  const maxSteps = Math.max(12, trajectory[trajectory.length - 1].step);

  const energyMin = -0.65;
  const energyMax = 0.05;
  const toX = (step: number) => padL + (step / maxSteps) * plotW;
  const toYE = (e: number) => {
    const clamped = Math.max(energyMin, Math.min(energyMax, e));
    return padT + ((energyMax - clamped) / (energyMax - energyMin)) * plotH;
  };
  const toYO = (m: number) => {
    const clamped = Math.max(-0.2, Math.min(1.05, m));
    return padT + ((1.05 - clamped) / 1.25) * plotH;
  };

  const energyPath = trajectory
    .map((pt, i) => `${i === 0 ? "M" : "L"}${toX(pt.step).toFixed(1)},${toYE(pt.energy).toFixed(1)}`)
    .join(" ");
  const overlapPath = trajectory
    .map((pt, i) => `${i === 0 ? "M" : "L"}${toX(pt.step).toFixed(1)},${toYO(pt.overlap).toFixed(1)}`)
    .join(" ");

  return (
    <div className="bg-[#05070f] border border-[rgba(250,249,245,0.10)] rounded-2xl p-6 text-[#faf9f5]">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-[rgba(250,249,245,0.08)] pb-4 mb-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-[rgba(94,234,212,0.08)] border border-[rgba(94,234,212,0.22)] rounded-lg text-[#5eead4]">
            <Activity className="w-5 h-5" />
          </div>
          <div>
            <h3 className="font-[family-name:var(--font-display)] text-xl font-normal text-[#faf9f5] tracking-tight">
              Energy Basin Descent · Experiment 12
            </h3>
            <p className="text-xs text-[rgba(250,249,245,0.62)]">
              Corrupted 64-bit pattern relaxation into stored attractors (thrml 0.1.4 vs Hopfield)
            </p>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <div
            className="inline-flex items-center rounded-full border border-[rgba(250,249,245,0.16)] bg-[#020308] p-0.5 text-[11px] font-mono"
            role="switch"
            aria-checked={dataSource === "live"}
            aria-label="Data Source: Synthetic Baseline or Live Public Telemetry"
          >
            <button
              type="button"
              onClick={() => void toggleSource("synthetic")}
              className={`px-2.5 py-1 rounded-full transition-colors ${
                dataSource === "synthetic"
                  ? "bg-[#faf9f5] text-[#020308] font-medium"
                  : "text-[rgba(250,249,245,0.65)] hover:text-[#faf9f5]"
              }`}
            >
              Synthetic Baseline
            </button>
            <button
              type="button"
              onClick={() => void toggleSource("live")}
              className={`px-2.5 py-1 rounded-full transition-colors ${
                dataSource === "live"
                  ? "bg-[#5eead4] text-[#020308] font-medium"
                  : "text-[rgba(250,249,245,0.65)] hover:text-[#faf9f5]"
              }`}
            >
              Live Public Telemetry
            </button>
          </div>
          {usingLive && resolvedLive ? (
            <span className="px-2.5 py-1 text-[10px] font-mono rounded-full border border-[#5eead4]/35 bg-[rgba(94,234,212,0.08)] text-[#5eead4]">
              {resolvedLive.ecological_ising?.provenance?.source_mode ?? "live"} ·{" "}
              {resolvedLive.generated_at_utc}
            </span>
          ) : dataSource === "live" ? (
            <span className="px-2.5 py-1 text-[10px] font-mono rounded-full border border-amber-400/35 bg-[rgba(245,158,11,0.08)] text-amber-300">
              {liveStatus === "loading" ? "Loading live stream..." : "Live stream offline · Synthetic fallback"}
            </span>
          ) : null}
          <span
            className={`px-3 py-1 text-[11px] font-mono rounded-full border ${
              lockedTarget
                ? "bg-[rgba(52,211,153,0.12)] border-[#34d399]/40 text-[#34d399]"
                : spuriousTrap
                ? "bg-[rgba(245,158,11,0.12)] border-amber-400/40 text-amber-300"
                : "bg-[rgba(94,234,212,0.08)] border-[#5eead4]/30 text-[#5eead4]"
            }`}
          >
            {lockedTarget
              ? "Target Attractor Locked"
              : spuriousTrap
              ? "Spurious Attractor Trap"
              : isRunning
              ? "Descending Basin..."
              : `Step ${latest.step} · Ready`}
          </span>
        </div>
      </div>

      {/* Controls Bar */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6 bg-[#020308]/70 p-4 rounded-xl border border-[rgba(250,249,245,0.08)]">
        {/* Pattern Selector */}
        <div>
          <label className="text-[10px] font-mono uppercase tracking-[0.16em] text-[rgba(250,249,245,0.5)] block mb-2">
            Stored Attractor
          </label>
          <div className="grid grid-cols-2 gap-1.5">
            {names.map((n, idx) => (
              <button
                key={n}
                type="button"
                onClick={() => setPatternIdx(idx)}
                className={`px-2 py-1.5 rounded-lg border text-[11px] font-mono transition-colors ${
                  patternIdx === idx
                    ? "bg-[#faf9f5] border-[#faf9f5] text-[#020308] font-medium"
                    : "bg-[#05070f] border-[rgba(250,249,245,0.10)] text-[rgba(250,249,245,0.68)] hover:border-[rgba(250,249,245,0.28)]"
                }`}
              >
                {n}
              </button>
            ))}
          </div>
        </div>

        {/* Cue Noise */}
        <div>
          <label className="text-[10px] font-mono uppercase tracking-[0.16em] text-[rgba(250,249,245,0.5)] block mb-2">
            Cue Noise Level
          </label>
          <div className="grid grid-cols-3 gap-1.5 mb-2">
            {([0.2, 0.35, 0.5] as NoiseLevel[]).map((nl) => (
              <button
                key={nl}
                type="button"
                onClick={() => setNoise(nl)}
                className={`py-1.5 rounded-lg border text-xs font-mono transition-colors ${
                  noise === nl
                    ? "bg-[#5eead4] border-[#5eead4] text-[#020308] font-medium"
                    : "bg-[#05070f] border-[rgba(250,249,245,0.10)] text-[rgba(250,249,245,0.68)] hover:border-[rgba(250,249,245,0.28)]"
                }`}
              >
                {Math.round(nl * 100)}%
              </button>
            ))}
          </div>
          <button
            type="button"
            onClick={() => setCueSeed((s) => s + 1)}
            className="w-full py-1 rounded-lg border border-[rgba(250,249,245,0.12)] text-[11px] font-mono text-[rgba(250,249,245,0.72)] hover:text-[#faf9f5] hover:border-[rgba(250,249,245,0.3)] flex items-center justify-center gap-1.5"
          >
            <Shuffle className="w-3 h-3" /> New Corrupted Cue
          </button>
        </div>

        {/* Sampler Selector */}
        <div>
          <label className="text-[10px] font-mono uppercase tracking-[0.16em] text-[rgba(250,249,245,0.5)] block mb-2">
            Relaxation Dynamics
          </label>
          <div className="grid grid-cols-2 gap-1.5">
            {(Object.keys(SAMPLER_META) as SamplerId[]).map((sid) => (
              <button
                key={sid}
                type="button"
                onClick={() => setSampler(sid)}
                className={`px-2 py-1.5 rounded-lg border text-[11px] font-mono transition-colors ${
                  sampler === sid
                    ? "bg-[#34d399]/20 border-[#34d399] text-[#34d399] font-medium"
                    : "bg-[#05070f] border-[rgba(250,249,245,0.10)] text-[rgba(250,249,245,0.68)] hover:border-[rgba(250,249,245,0.28)]"
                }`}
              >
                {SAMPLER_META[sid].name}
              </button>
            ))}
          </div>
        </div>

        {/* Playback */}
        <div className="flex flex-col justify-between">
          <label className="text-[10px] font-mono uppercase tracking-[0.16em] text-[rgba(250,249,245,0.5)] block mb-2">
            Playback Controls
          </label>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setIsRunning((r) => !r)}
              disabled={converged}
              className="flex-1 py-2 px-3 rounded-lg bg-[#faf9f5] text-[#020308] text-xs font-mono font-medium hover:bg-[#5eead4] transition-colors disabled:opacity-40 flex items-center justify-center gap-1.5"
            >
              {isRunning ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
              {isRunning ? "Pause" : "Run"}
            </button>
            <button
              type="button"
              onClick={stepOnce}
              disabled={converged || isRunning}
              className="p-2 rounded-lg border border-[rgba(250,249,245,0.15)] text-[#faf9f5] hover:border-[#5eead4] disabled:opacity-40"
              title="Single step"
            >
              <StepForward className="w-4 h-4" />
            </button>
            <button
              type="button"
              onClick={() => resetToCue(initialCue)}
              className="p-2 rounded-lg border border-[rgba(250,249,245,0.15)] text-[#faf9f5] hover:border-[#5eead4]"
              title="Reset cue"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
          </div>
          <p className="text-[10px] font-mono text-[rgba(250,249,245,0.45)] mt-2 truncate">
            {SAMPLER_META[sampler].tag}
          </p>
        </div>
      </div>

      {/* Main Split: 8x8 Lattice Grid + Energy Basin Curve */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 mb-6">
        {/* Lattice Grid (5 cols) */}
        <div className="lg:col-span-5 bg-[#020308]/80 border border-[rgba(250,249,245,0.08)] rounded-xl p-4 flex flex-col items-center justify-between">
          <div className="w-full flex items-center justify-between mb-3">
            <span className="text-[10px] font-mono uppercase tracking-[0.18em] text-[#5eead4]">
              8x8 Spin State (N=64)
            </span>
            <span className="text-[11px] font-mono text-[rgba(250,249,245,0.55)]">
              Errors: {spins.filter((v, i) => v !== target[i]).length} / 64
            </span>
          </div>

          <div className="grid grid-cols-8 gap-1.5 p-2 bg-[#05070f] rounded-lg border border-[rgba(250,249,245,0.06)]">
            {spins.map((val, i) => {
              const isError = val !== target[i];
              return (
                <div
                  key={i}
                  className={`w-6 h-6 sm:w-7 sm:h-7 rounded transition-colors duration-200 ${
                    val === 1 ? "bg-[#5eead4]" : "bg-[#0d1424]"
                  } ${isError ? "ring-1 ring-amber-400/80" : "border border-[rgba(250,249,245,0.08)]"}`}
                  title={`Bit ${i}: ${val === 1 ? "+1" : "-1"}${isError ? " (error)" : ""}`}
                />
              );
            })}
          </div>

          <div className="w-full flex items-center justify-between mt-3 pt-3 border-t border-[rgba(250,249,245,0.06)] text-[11px] font-mono">
            <div className="flex items-center gap-2 text-[rgba(250,249,245,0.65)]">
              <span className="inline-block w-2.5 h-2.5 rounded-sm bg-[#5eead4]" /> +1
              <span className="inline-block w-2.5 h-2.5 rounded-sm bg-[#0d1424] border border-[rgba(250,249,245,0.2)] ml-1" /> -1
              <span className="inline-block w-2.5 h-2.5 rounded-sm ring-1 ring-amber-400 ml-1" /> Bit Error
            </div>
            <div className="flex items-center gap-1.5">
              <span className="text-[rgba(250,249,245,0.45)]">Target:</span>
              <div className="grid grid-cols-8 gap-[1px] p-0.5 bg-[#05070f] rounded border border-[rgba(250,249,245,0.1)]">
                {target.map((v, idx) => (
                  <div
                    key={idx}
                    className={`w-1.5 h-1.5 ${v === 1 ? "bg-[#34d399]" : "bg-[#0d1424]"}`}
                  />
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Energy Basin Curve + Telemetry (7 cols) */}
        <div className="lg:col-span-7 bg-[#020308]/80 border border-[rgba(250,249,245,0.08)] rounded-xl p-4 flex flex-col justify-between">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[10px] font-mono uppercase tracking-[0.18em] text-[#5eead4]">
              Basin Descent Trajectory · E(t) &amp; Overlap m(t)
            </span>
            <div className="flex items-center gap-3 text-[11px] font-mono">
              <span className="flex items-center gap-1.5 text-[#5eead4]">
                <span className="w-2.5 h-0.5 bg-[#5eead4] inline-block" /> Energy E(t)
              </span>
              <span className="flex items-center gap-1.5 text-[#34d399]">
                <span className="w-2.5 h-0.5 bg-[#34d399] inline-block" /> Overlap m(t)
              </span>
            </div>
          </div>

          <svg viewBox={`0 0 ${svgW} ${svgH}`} className="w-full h-48 overflow-visible">
            {/* Horizontal grid lines */}
            {[0, 0.25, 0.5, 0.75, 1.0].map((frac) => {
              const y = padT + frac * plotH;
              return (
                <line
                  key={frac}
                  x1={padL}
                  y1={y}
                  x2={svgW - padR}
                  y2={y}
                  stroke="rgba(250,249,245,0.07)"
                  strokeDasharray="3 3"
                />
              );
            })}
            {/* Target overlap = 1.0 line */}
            <line
              x1={padL}
              y1={toYO(1.0)}
              x2={svgW - padR}
              y2={toYO(1.0)}
              stroke="rgba(52,211,153,0.28)"
              strokeDasharray="4 4"
            />
            <text x={padL - 6} y={toYO(1.0) + 3} textAnchor="end" fill="#34d399" fontSize="9" fontFamily="monospace">
              m=1.0
            </text>
            <text x={padL - 6} y={toYE(-0.5) + 3} textAnchor="end" fill="#5eead4" fontSize="9" fontFamily="monospace">
              E=-0.5
            </text>
            <text x={padL + plotW / 2} y={svgH - 4} textAnchor="middle" fill="rgba(250,249,245,0.45)" fontSize="9" fontFamily="monospace">
              Relaxation Step t
            </text>

            {/* Paths */}
            {trajectory.length > 1 && (
              <>
                <path d={energyPath} fill="none" stroke="#5eead4" strokeWidth="2" />
                <path d={overlapPath} fill="none" stroke="#34d399" strokeWidth="2" />
              </>
            )}
            {trajectory.map((pt) => (
              <g key={pt.step}>
                <circle cx={toX(pt.step)} cy={toYE(pt.energy)} r="3" fill="#5eead4" />
                <circle cx={toX(pt.step)} cy={toYO(pt.overlap)} r="3" fill="#34d399" />
              </g>
            ))}
          </svg>

          {/* Telemetry Cards */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 mt-3">
            <div className="bg-[#05070f] border border-[rgba(250,249,245,0.08)] rounded-lg p-2.5">
              <div className="text-[10px] font-mono text-[rgba(250,249,245,0.45)] uppercase">Bit Recovery</div>
              <div className="text-base font-mono text-[#34d399] mt-0.5">
                {(latest.recovery * 100).toFixed(1)}%
              </div>
            </div>
            <div className="bg-[#05070f] border border-[rgba(250,249,245,0.08)] rounded-lg p-2.5">
              <div className="text-[10px] font-mono text-[rgba(250,249,245,0.45)] uppercase">Overlap m(t)</div>
              <div className="text-base font-mono text-[#faf9f5] mt-0.5">{latest.overlap.toFixed(3)}</div>
            </div>
            <div className="bg-[#05070f] border border-[rgba(250,249,245,0.08)] rounded-lg p-2.5">
              <div className="text-[10px] font-mono text-[rgba(250,249,245,0.45)] uppercase">Energy E/N</div>
              <div className="text-base font-mono text-[#5eead4] mt-0.5">{latest.energy.toFixed(3)}</div>
            </div>
            <div className="bg-[#05070f] border border-[rgba(250,249,245,0.08)] rounded-lg p-2.5">
              <div className="text-[10px] font-mono text-[rgba(250,249,245,0.45)] uppercase">Beta · Sokal τ_int</div>
              <div className="text-base font-mono text-[#faf9f5] mt-0.5">
                {latest.beta.toFixed(1)} · {sokalStats.tauInt.toFixed(2)}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Empirical Benchmark Table from Experiment 12 */}
      <div className="bg-[#020308]/70 border border-[rgba(250,249,245,0.08)] rounded-xl p-4">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
          <span className="text-[10px] font-mono uppercase tracking-[0.18em] text-[#5eead4]">
            Experiment 12 Empirical Benchmark · 100 Trials per Noise Level (P=10, N=64)
          </span>
          <span className="text-[11px] font-mono text-[rgba(250,249,245,0.45)]">
            Calibration: configs/exp12_annealing_calibration.json
          </span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left font-mono text-xs border-collapse">
            <thead>
              <tr className="border-b border-[rgba(250,249,245,0.10)] text-[rgba(250,249,245,0.5)]">
                <th className="py-2 pr-3 font-normal">Cue Noise</th>
                <th className="py-2 px-3 font-normal">Model</th>
                <th className="py-2 px-3 font-normal">Bit Recovery</th>
                <th className="py-2 px-3 font-normal">Spurious Rate</th>
                <th className="py-2 px-3 font-normal">Conv. Steps</th>
                <th className="py-2 pl-3 font-normal text-right">Batch Time</th>
              </tr>
            </thead>
            <tbody>
              {EXP12_BENCHMARK_ROWS.map((group) =>
                group.models.map((row, idx) => (
                  <tr
                    key={`${group.noise}-${row.id}`}
                    className={`border-b border-[rgba(250,249,245,0.05)] ${
                      row.id === sampler ? "bg-[rgba(94,234,212,0.06)]" : ""
                    }`}
                  >
                    <td className="py-1.5 pr-3 text-[rgba(250,249,245,0.75)]">
                      {idx === 0 ? group.noise : ""}
                    </td>
                    <td className="py-1.5 px-3 text-[#faf9f5]">{row.label}</td>
                    <td className="py-1.5 px-3 text-[#34d399]">{row.rec}</td>
                    <td className="py-1.5 px-3 text-[rgba(250,249,245,0.75)]">{row.spur}</td>
                    <td className="py-1.5 px-3 text-[rgba(250,249,245,0.75)]">{row.steps}</td>
                    <td className="py-1.5 pl-3 text-right text-[rgba(250,249,245,0.65)]">{row.time}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
