"use client";

import React, { useEffect, useRef, useState, useCallback } from "react";
import Link from "next/link";

interface StationItem {
  id: number;
  name: string;
  desc: string;
  metric: string;
  runtime: string;
  command: string;
}

const STATIONS: StationItem[] = [
  {
    id: 1,
    name: "1D Ising Chain",
    desc: "Nearest-neighbour spins on a line — the first check against an exact solution.",
    metric: "E = −0.965 · exact ground state −2.0",
    runtime: "0.210s",
    command: "uv run thrml-bench run --exp 1",
  },
  {
    id: 2,
    name: "2D Ising — Onsager",
    desc: "The classic square-lattice magnet, swept across its phase transition.",
    metric: "|M| 0.0434 (β=0.20) → 0.9417 (β=0.70) · βc ≈ 0.4407",
    runtime: "5.812s",
    command: "uv run thrml-bench run --exp 2",
  },
  {
    id: 3,
    name: "Max-Cut",
    desc: "Partition a random graph so as many edges as possible cross the cut.",
    metric: "Erdős–Rényi G(50, 0.20) · 70.8% cut edges (172/243 vs 49.9% random)",
    runtime: "7.344s",
    command: "uv run thrml-bench run --exp 3",
  },
  {
    id: 4,
    name: "Potts, q = 4",
    desc: "Four-state spins in place of two — a sharper transition to locate.",
    metric: "βc = ln(3) ≈ 1.0986",
    runtime: "2.956s",
    command: "uv run thrml-bench run --exp 4",
  },
  {
    id: 5,
    name: "Clamped Ising denoiser",
    desc: "Clamp the known pixels, sample the rest, and let the field clean the image.",
    metric: "22.07% noise → 4.97% error · 77.5% reduction",
    runtime: "0.583s",
    command: "uv run thrml-bench run --exp 5",
  },
  {
    id: 6,
    name: "RBM",
    desc: "A restricted Boltzmann machine recovers hidden patterns and occluded bits.",
    metric: "100% cosine overlap · 100% occluded-bit recovery",
    runtime: "0.583s",
    command: "uv run thrml-bench run --exp 6",
  },
  {
    id: 7,
    name: "Markowitz QUBO",
    desc: "Portfolio selection written as a quadratic binary problem.",
    metric: "99.1% budget feasibility · 94.1% Pareto match",
    runtime: "14.696s",
    command: "uv run thrml-bench run --exp 7",
  },
  {
    id: 8,
    name: "Double-well Langevin",
    desc: "Continuous dynamics hopping between two wells as the temperature changes.",
    metric: "Hop rate 38.4%/step (β=0.35) → 0.9%/step (β=4.0)",
    runtime: "2.805s",
    command: "uv run thrml-bench run --exp 8",
  },
  {
    id: 9,
    name: "TSP, 10 cities",
    desc: "The travelling salesman problem, encoded as an energy to minimise.",
    metric: "100% optimal tour · 6.296 vs 17.986 naive (−65%)",
    runtime: "20.110s",
    command: "uv run thrml-bench run --exp 9",
  },
  {
    id: 10,
    name: "SK spin glass",
    desc: "Fully connected, disordered spins — the rugged landscape, searched whole.",
    metric: "100% ground-state match (−15.4266) · RSB |q| = 0.5215",
    runtime: "0.27s",
    command: "uv run thrml-bench run --exp 10",
  },
  {
    id: 11,
    name: "RNA folding",
    desc: "Secondary structure as a ground-state search, verified exactly at small N and scaled on a riboswitch.",
    metric: "100% ground-state match (−10.00 / −66.00 kcal/mol) · N=24 exact · N=50 scaled · 7/7 tests passing",
    runtime: "0.21s / 0.31s",
    command: "uv run thrml-bench run --exp 11",
  },
  {
    id: 12,
    name: "SCUC unit commitment",
    desc: "Security-constrained unit commitment: meet demand with no deficit. The microgrid section below is this station, written out in full.",
    metric: "Zero-deficit dispatch",
    runtime: "0.12s",
    command: "uv run thrml-bench run --exp 12",
  },
  {
    id: 13,
    name: "Codon Potts",
    desc: "Codon choice as a Potts model — remove the hairpins, keep the code strong. Written out in full in the codon section below.",
    metric: "4 hairpins → 0 · CAI 0.9326 · ΔE −7.82",
    runtime: "0.18s",
    command: "uv run thrml-bench run --exp 13",
  },
  {
    id: 14,
    name: "Rotamer packing",
    desc: "Protein side-chain packing that resolves clashes by lowering the energy.",
    metric: "6 clashes → 0 · ΔE −295.54 kcal/mol",
    runtime: "0.12s",
    command: "uv run thrml-bench run --exp 14",
  },
  {
    id: 15,
    name: "SI core",
    desc: "A coupled-organs simulation core, held to a strict latency budget.",
    metric: "Latency 5.2 ms under a 15 ms SLA · 4 organs coupled · RTX 5080 VRAM safe",
    runtime: "5.2 ms",
    command: "uv run thrml-bench run --exp 15",
  },
  {
    id: 16,
    name: "Hybrid TPU Langevin",
    desc: "Hybrid sampling, checked end to end for constraint violations.",
    metric: "0 violations (100% SAT) · housekeeping heat Qhk = 3,697.4",
    runtime: "1.58s",
    command: "uv run thrml-bench run --exp 16",
  },
];

export default function ThermoLabPage() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [sliderVal, setSliderVal] = useState<number>(70);
  const [tempVal, setTempVal] = useState<number>(3.5);
  const [betaVal, setBetaVal] = useState<string>("0.286");
  const [energyVal, setEnergyVal] = useState<string>("0.00");
  const [orderVal, setOrderVal] = useState<string>("0.00");
  const [stateVal, setStateVal] = useState<string>("Disordered");
  const [sweepsCount, setSweepsCount] = useState<string>("0");
  const [isAnnealing, setIsAnnealing] = useState<boolean>(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const latticeState = useRef<{
    N: number;
    spins: Int8Array;
    sweeps: number;
    temp: number;
    annealAnim: number | null;
  }>({
    N: 16,
    spins: new Int8Array(16 * 16),
    sweeps: 0,
    temp: 3.5,
    annealAnim: null,
  });

  const handleCopy = useCallback((text: string, id: string) => {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text);
    } else {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
    }
    setCopiedId(id);
    setTimeout(() => {
      setCopiedId((curr) => (curr === id ? null : curr));
    }, 1600);
  }, []);

  const setT = useCallback((v: number) => {
    const clampedT = Math.min(4.0, Math.max(0.2, v));
    latticeState.current.temp = clampedT;
    setTempVal(clampedT);
    setSliderVal(Math.round(420 - clampedT * 100));
    setBetaVal((1 / clampedT).toFixed(3));
  }, []);

  useEffect(() => {
    const N = 16;
    const spins = latticeState.current.spins;
    for (let i = 0; i < spins.length; i++) {
      spins[i] = Math.random() < 0.5 ? 1 : -1;
    }

    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const idx = (x: number, y: number) => ((y + N) % N) * N + ((x + N) % N);

    const sweep = (beta: number) => {
      for (let s = 0; s < N * N; s++) {
        const x = (Math.random() * N) | 0;
        const y = (Math.random() * N) | 0;
        const i = y * N + x;
        const nb =
          spins[idx(x + 1, y)] +
          spins[idx(x - 1, y)] +
          spins[idx(x, y + 1)] +
          spins[idx(x, y - 1)];
        const dE = 2 * spins[i] * nb;
        if (dE <= 0 || Math.random() < Math.exp(-dE * beta)) {
          spins[i] = -spins[i] as any;
        }
      }
      latticeState.current.sweeps++;
    };

    const wolff = (beta: number) => {
      const sx = (Math.random() * N) | 0;
      const sy = (Math.random() * N) | 0;
      const spin = spins[sy * N + sx];
      const padd = 1 - Math.exp(-2 * beta);
      const stack: [number, number][] = [[sx, sy]];
      const seen = new Uint8Array(N * N);
      seen[sy * N + sx] = 1;
      const members: [number, number][] = [[sx, sy]];

      while (stack.length) {
        const [cx, cy] = stack.pop()!;
        const nbs = [
          [cx + 1, cy],
          [cx - 1, cy],
          [cx, cy + 1],
          [cx, cy - 1],
        ];
        for (let k = 0; k < 4; k++) {
          const nx = (nbs[k][0] + N) % N;
          const ny = (nbs[k][1] + N) % N;
          const ni = ny * N + nx;
          if (!seen[ni] && spins[ni] === spin && Math.random() < padd) {
            seen[ni] = 1;
            stack.push([nx, ny]);
            members.push([nx, ny]);
          }
        }
      }
      for (let j = 0; j < members.length; j++) {
        spins[members[j][1] * N + members[j][0]] = -spin as any;
      }
    };

    const computeStats = () => {
      let e = 0;
      let m = 0;
      for (let y = 0; y < N; y++) {
        for (let x = 0; x < N; x++) {
          const s = spins[y * N + x];
          m += s;
          e -= s * (spins[idx(x + 1, y)] + spins[idx(x, y + 1)]);
        }
      }
      return { e: e / (N * N), m: Math.abs(m) / (N * N) };
    };

    const draw = () => {
      const cell = canvas.width / N;
      for (let y = 0; y < N; y++) {
        for (let x = 0; x < N; x++) {
          ctx.fillStyle = spins[y * N + x] > 0 ? "#16222B" : "#A9BEC4";
          ctx.fillRect(x * cell + 1, y * cell + 1, cell - 2, cell - 2);
        }
      }
    };

    let animId: number;
    const renderLoop = () => {
      const currentT = latticeState.current.temp;
      const beta = 1 / currentT;
      const per = reduced ? 1 : 3;
      for (let k = 0; k < per; k++) sweep(beta);
      if (currentT < 3.2) {
        const w = reduced ? 2 : 5;
        for (let q = 0; q < w; q++) wolff(beta);
      }
      draw();
      const st = computeStats();
      setEnergyVal(st.e.toFixed(2));
      setOrderVal(st.m.toFixed(2));
      setSweepsCount(latticeState.current.sweeps.toLocaleString("en-US"));
      setStateVal(
        st.m > 0.85 ? "Ground state" : st.m > 0.45 ? "Ordering" : "Disordered"
      );
      animId = requestAnimationFrame(renderLoop);
    };

    draw();
    animId = requestAnimationFrame(renderLoop);

    return () => {
      cancelAnimationFrame(animId);
      if (latticeState.current.annealAnim) {
        cancelAnimationFrame(latticeState.current.annealAnim);
      }
    };
  }, []);

  const triggerAnneal = useCallback(() => {
    if (isAnnealing) return;
    setIsAnnealing(true);

    const N = 16;
    const spins = latticeState.current.spins;
    for (let i = 0; i < spins.length; i++) {
      spins[i] = Math.random() < 0.5 ? 1 : -1;
    }
    latticeState.current.sweeps = 0;

    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const from = 3.8;
    const to = 0.25;
    const dur = reduced ? 60 : 4200;
    const t0 = performance.now();

    const step = (now: number) => {
      const p = Math.min(1, (now - t0) / dur);
      const e = 1 - Math.pow(1 - p, 3);
      setT(from + (to - from) * e);
      if (p < 1) {
        latticeState.current.annealAnim = requestAnimationFrame(step);
      } else {
        latticeState.current.annealAnim = null;
        setIsAnnealing(false);
      }
    };
    latticeState.current.annealAnim = requestAnimationFrame(step);
  }, [isAnnealing, setT]);

  return (
    <div className="thermo-lab-container">
      <style jsx global>{`
        .thermo-lab-container {
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
          --cell-up: #16222B;
          --cell-down: #A9BEC4;
          --focus: #E8490F;
          background: var(--paper);
          color: var(--ink);
          font-family: 'Archivo', system-ui, -apple-system, sans-serif;
          font-size: 1.0625rem;
          line-height: 1.62;
          -webkit-font-smoothing: antialiased;
          min-height: 100vh;
        }
        .thermo-lab-container ::selection {
          background: var(--ember);
          color: #fff;
        }
        .thermo-lab-container h1,
        .thermo-lab-container h2,
        .thermo-lab-container h3 {
          font-family: 'Zilla Slab', Georgia, serif;
          font-weight: 700;
          line-height: 1.08;
          letter-spacing: -0.015em;
          text-wrap: balance;
          margin: 0;
        }
        .thermo-lab-container .mono {
          font-family: 'JetBrains Mono', Consolas, monospace;
        }
        .thermo-lab-container .wrap {
          max-width: 76rem;
          margin: 0 auto;
          padding: 0 clamp(1.1rem, 4vw, 3rem);
        }
        .thermo-lab-container .measure {
          max-width: 70ch;
        }

        /* Topbar */
        .thermo-topbar {
          position: sticky;
          top: 0;
          z-index: 20;
          background: var(--paper);
          border-bottom: 2px solid var(--ink);
        }
        .thermo-topbar-in {
          display: flex;
          align-items: center;
          gap: 1.5rem;
          flex-wrap: wrap;
          padding: 0.65rem 0;
        }
        .thermo-brand {
          font-family: 'Zilla Slab', serif;
          font-weight: 700;
          font-size: 1.3rem;
          letter-spacing: 0.04em;
          color: var(--ink);
          text-decoration: none;
          display: flex;
          align-items: center;
          gap: 0.6rem;
          min-height: 44px;
        }
        .thermo-brand:hover {
          color: var(--ink);
        }
        .thermo-topnav {
          margin-left: auto;
          display: flex;
          flex-wrap: wrap;
          gap: 0.15rem;
        }
        .thermo-topnav a {
          color: var(--ink-2);
          text-decoration: none;
          font-weight: 600;
          font-size: 0.98rem;
          padding: 0.55rem 0.7rem;
          min-height: 44px;
          display: inline-flex;
          align-items: center;
          border-radius: 3px;
        }
        .thermo-topnav a:hover {
          color: var(--ember-text);
          background: var(--panel);
        }

        /* Hero */
        .thermo-hero {
          padding: clamp(2.6rem, 6vw, 4.8rem) 0 clamp(2.4rem, 5vw, 4rem);
          border-bottom: 2px solid var(--ink);
        }
        .thermo-hero-grid {
          display: grid;
          grid-template-columns: minmax(0, 1.05fr) minmax(0, 0.95fr);
          gap: clamp(2rem, 5vw, 4.5rem);
          align-items: start;
        }
        .thermo-hero h1 {
          font-size: clamp(3rem, 7.2vw, 5.6rem);
          line-height: 0.98;
          letter-spacing: -0.02em;
        }
        .thermo-hero h1 .heat {
          color: var(--ember-text);
        }
        .thermo-hero-lede {
          margin-top: 1.6rem;
          font-size: 1.22rem;
          line-height: 1.55;
          color: var(--ink);
          max-width: 34em;
        }
        .thermo-hero-copy {
          margin-top: 1.1rem;
          color: var(--ink-2);
          max-width: 36em;
        }
        .thermo-status {
          margin-top: 1.9rem;
          border-top: 1px solid var(--rule);
          border-bottom: 1px solid var(--rule);
          padding: 0.95rem 0;
          display: flex;
          gap: 0.8rem;
          align-items: flex-start;
          max-width: 36em;
        }
        .thermo-status svg {
          flex: none;
          margin-top: 0.3rem;
        }
        .thermo-status p {
          font-size: 1rem;
          color: var(--ink);
          margin: 0;
        }
        .thermo-hero-actions {
          margin-top: 1.7rem;
          display: flex;
          flex-wrap: wrap;
          gap: 0.8rem;
        }
        .thermo-btn {
          display: inline-flex;
          align-items: center;
          gap: 0.5rem;
          min-height: 48px;
          padding: 0.7rem 1.2rem;
          font-family: 'Archivo', sans-serif;
          font-weight: 700;
          font-size: 1rem;
          text-decoration: none;
          border-radius: 3px;
          border: 2px solid var(--ink);
          transition: background 0.18s ease, color 0.18s ease, border-color 0.18s ease;
          cursor: pointer;
        }
        .thermo-btn-solid {
          background: var(--ink);
          color: var(--panel);
        }
        .thermo-btn-solid:hover {
          background: var(--ember);
          border-color: var(--ember);
          color: #fff;
        }
        .thermo-btn-ghost {
          background: transparent;
          color: var(--ink);
        }
        .thermo-btn-ghost:hover {
          border-color: var(--ember-text);
          color: var(--ember-text);
          background: var(--panel);
        }

        /* Lattice Bench */
        .thermo-bench {
          background: var(--panel);
          border: 1.5px solid var(--ink);
          border-radius: 4px;
          padding: 1.2rem 1.2rem 1.1rem;
          box-shadow: 10px 14px 28px rgba(22, 34, 43, 0.14);
        }
        .thermo-bench-head {
          display: flex;
          justify-content: space-between;
          align-items: baseline;
          gap: 1rem;
          border-bottom: 1px solid var(--rule);
          padding-bottom: 0.7rem;
        }
        .thermo-bench-head h2 {
          font-size: 1.25rem;
          font-family: 'Zilla Slab', serif;
        }
        .thermo-bench-head .mono {
          font-size: 0.8rem;
          color: var(--ink-3);
        }
        canvas.thermo-lattice-canvas {
          display: block;
          width: 100%;
          height: auto;
          margin: 1rem 0 0.4rem;
          border: 1px solid var(--rule);
          background: var(--panel);
        }
        .thermo-readouts {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 0.8rem;
          border-top: 1px solid var(--rule);
          padding-top: 0.85rem;
        }
        .thermo-readout .lab {
          font-size: 0.78rem;
          font-weight: 600;
          color: var(--ink-3);
          letter-spacing: 0.02em;
        }
        .thermo-readout .val {
          font-family: 'JetBrains Mono', monospace;
          font-weight: 700;
          font-size: 1.28rem;
          font-variant-numeric: tabular-nums;
          line-height: 1.25;
        }
        .thermo-readout .val.hot {
          color: var(--ember-text);
        }
        .thermo-slider-row {
          margin-top: 1.05rem;
        }
        .thermo-slider-labels {
          display: flex;
          justify-content: space-between;
          font-size: 0.85rem;
          font-weight: 600;
          color: var(--ink-2);
        }
        .thermo-slider-labels .hot-lab {
          color: var(--ember-text);
        }
        .thermo-slider {
          width: 100%;
          min-height: 44px;
          accent-color: var(--ember);
          cursor: pointer;
          background: transparent;
          margin: 0.1rem 0 0;
        }
        .thermo-bench-foot {
          display: flex;
          flex-wrap: wrap;
          gap: 0.7rem;
          align-items: center;
          justify-content: space-between;
          margin-top: 0.4rem;
        }
        .thermo-bench-note {
          font-size: 0.88rem;
          color: var(--ink-3);
          max-width: 26em;
          margin: 0;
        }
        .thermo-btn-anneal {
          min-height: 48px;
          padding: 0.65rem 1.15rem;
          border-radius: 3px;
          cursor: pointer;
          font-family: 'Archivo', sans-serif;
          font-weight: 700;
          font-size: 0.98rem;
          background: var(--ember);
          color: #fff;
          border: 2px solid var(--ember);
          transition: background 0.18s ease, border-color 0.18s ease;
        }
        .thermo-btn-anneal:hover {
          background: var(--ember-text);
          border-color: var(--ember-text);
        }
        .thermo-btn-anneal[disabled] {
          opacity: 0.55;
          cursor: wait;
        }

        /* Sections */
        section.thermo-blk {
          padding: clamp(2.8rem, 6vw, 4.6rem) 0;
          border-bottom: 2px solid var(--ink);
        }
        .thermo-blk h2 {
          font-size: clamp(2rem, 4.4vw, 3.1rem);
          max-width: 22em;
        }
        .thermo-blk .intro {
          margin-top: 1.1rem;
          color: var(--ink-2);
          font-size: 1.1rem;
        }
        .thermo-cols {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: clamp(1.6rem, 4vw, 3rem);
          margin-top: 2.2rem;
        }
        .thermo-cols h3 {
          font-size: 1.35rem;
          margin-bottom: 0.55rem;
        }
        .thermo-cols p {
          color: var(--ink-2);
          margin: 0;
        }
        .thermo-note-line {
          margin-top: 2rem;
          border-top: 1px solid var(--rule);
          padding-top: 1rem;
          color: var(--ink-2);
          max-width: 72ch;
        }
        .thermo-note-line strong {
          color: var(--ink);
        }

        /* Experiment index */
        .exp-head,
        .exp-row {
          display: grid;
          grid-template-columns: 5.2rem minmax(0, 1.25fr) minmax(0, 1fr) 9.5rem;
          gap: 1.2rem;
          align-items: baseline;
        }
        .exp-head {
          padding: 0.6rem 0;
          font-size: 0.82rem;
          font-weight: 700;
          letter-spacing: 0.05em;
          color: var(--ink-3);
          border-bottom: 2px solid var(--ink);
        }
        .exp-row {
          padding: 1.05rem 0;
          border-bottom: 1px solid var(--rule);
        }
        .exp-row:hover {
          background: var(--panel);
        }
        .exp-no {
          font-family: 'JetBrains Mono', monospace;
          font-weight: 700;
          font-size: 1.02rem;
          color: var(--ink);
          white-space: nowrap;
        }
        .exp-no .hot {
          color: var(--ember-text);
        }
        .exp-name {
          font-family: 'Zilla Slab', serif;
          font-weight: 600;
          font-size: 1.22rem;
          line-height: 1.25;
        }
        .exp-desc {
          color: var(--ink-2);
          font-size: 0.98rem;
          margin-top: 0.2rem;
          max-width: 44ch;
        }
        .exp-metric {
          font-family: 'JetBrains Mono', monospace;
          font-size: 0.9rem;
          line-height: 1.55;
          color: var(--steel-deep);
          font-variant-numeric: tabular-nums;
        }
        .exp-side {
          text-align: right;
        }
        .exp-time {
          font-family: 'JetBrains Mono', monospace;
          font-weight: 700;
          font-size: 0.95rem;
          font-variant-numeric: tabular-nums;
        }
        .thermo-cmd {
          display: inline-flex;
          align-items: center;
          gap: 0.45rem;
          margin-top: 0.45rem;
          font-family: 'JetBrains Mono', monospace;
          font-size: 0.76rem;
          color: var(--ink-2);
          background: transparent;
          border: 1px solid var(--rule);
          border-radius: 3px;
          padding: 0.45rem 0.55rem;
          min-height: 36px;
          cursor: pointer;
          text-align: left;
          transition: border-color 0.15s ease, color 0.15s ease, background 0.15s ease;
        }
        .thermo-cmd:hover {
          border-color: var(--steel);
          color: var(--steel-deep);
          background: var(--panel);
        }
        .thermo-substrate {
          margin-top: 1.8rem;
          background: var(--panel);
          border: 1px solid var(--rule);
          border-radius: 4px;
          padding: 1.1rem 1.25rem;
          color: var(--ink-2);
        }
        .thermo-substrate .mono {
          color: var(--steel-deep);
          font-variant-numeric: tabular-nums;
        }
        .thermo-substrate strong {
          color: var(--ink);
        }

        /* Codon table */
        .thermo-featured {
          background: var(--panel);
          border-bottom: 2px solid var(--ink);
        }
        table.cmp {
          width: 100%;
          border-collapse: collapse;
          margin-top: 2rem;
          font-variant-numeric: tabular-nums;
        }
        table.cmp th,
        table.cmp td {
          padding: 0.75rem 0.8rem;
          border-bottom: 1px solid var(--rule);
          text-align: right;
          vertical-align: top;
        }
        table.cmp th:first-child,
        table.cmp td:first-child {
          text-align: left;
        }
        table.cmp thead th {
          font-family: 'Zilla Slab', serif;
          font-size: 1.05rem;
          border-bottom: 2px solid var(--ink);
        }
        table.cmp td {
          font-family: 'JetBrains Mono', monospace;
          font-size: 0.95rem;
          color: var(--ink-2);
        }
        table.cmp td:first-child {
          font-family: 'Archivo', sans-serif;
          font-weight: 600;
          color: var(--ink);
        }
        table.cmp .win {
          color: var(--ember-text);
          font-weight: 700;
        }
        .cmp-scroll {
          overflow-x: auto;
        }
        .thermo-two-col {
          display: grid;
          grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
          gap: clamp(1.8rem, 5vw, 4rem);
          margin-top: 2.6rem;
        }
        .thermo-two-col h3 {
          font-size: 1.5rem;
          margin-bottom: 0.8rem;
        }
        .thermo-two-col p {
          color: var(--ink-2);
          margin: 0;
        }
        ul.rules {
          list-style: none;
          margin: 0;
          padding: 0;
        }
        ul.rules li {
          padding: 0.62rem 0;
          border-bottom: 1px solid var(--rule-soft);
          display: flex;
          justify-content: space-between;
          gap: 1rem;
          align-items: baseline;
        }
        ul.rules li:last-child {
          border-bottom: none;
        }
        ul.rules .who {
          font-weight: 600;
        }
        ul.rules .what {
          font-family: 'JetBrains Mono', monospace;
          font-size: 0.88rem;
          color: var(--steel-deep);
          white-space: nowrap;
          font-variant-numeric: tabular-nums;
        }
        .thermo-formula {
          margin-top: 2.2rem;
          border: 1.5px solid var(--ink);
          border-radius: 4px;
          background: var(--paper);
          padding: 1.1rem 1.25rem;
          overflow-x: auto;
        }
        .thermo-formula .flab {
          font-size: 0.85rem;
          font-weight: 700;
          color: var(--ink-3);
          margin-bottom: 0.45rem;
        }
        .thermo-formula code {
          font-family: 'JetBrains Mono', monospace;
          font-size: 1rem;
          white-space: pre-wrap;
          word-break: break-word;
          line-height: 1.75;
          color: var(--ink);
        }
        .thermo-codeblock {
          margin-top: 1rem;
          background: var(--ink);
          color: #edf2f3;
          border-radius: 4px;
          padding: 1rem 1.15rem;
          display: flex;
          gap: 1rem;
          align-items: flex-start;
          justify-content: space-between;
        }
        .thermo-codeblock code {
          font-family: 'JetBrains Mono', monospace;
          font-size: 0.88rem;
          line-height: 1.6;
          white-space: pre-wrap;
          word-break: break-word;
        }
        .thermo-copy-btn {
          flex: none;
          min-height: 40px;
          padding: 0.5rem 0.8rem;
          border-radius: 3px;
          cursor: pointer;
          font-family: 'Archivo', sans-serif;
          font-weight: 700;
          font-size: 0.85rem;
          background: transparent;
          color: #edf2f3;
          border: 1.5px solid #5b7280;
          transition: border-color 0.15s ease, background 0.15s ease;
        }
        .thermo-copy-btn:hover {
          border-color: var(--ember);
          background: rgba(232, 73, 15, 0.18);
        }
        .thermo-trial {
          margin-top: 2.6rem;
          border-top: 2px solid var(--ink);
          padding-top: 1.6rem;
          display: grid;
          grid-template-columns: minmax(0, 1.2fr) minmax(0, 0.8fr);
          gap: clamp(1.8rem, 5vw, 4rem);
        }
        .thermo-trial h3 {
          font-size: 1.6rem;
          margin-bottom: 0.7rem;
        }
        .thermo-trial p {
          color: var(--ink-2);
          margin: 0;
        }
        .thermo-trial p + p {
          margin-top: 0.8rem;
        }
        .thermo-trial .side {
          border: 1px solid var(--rule);
          border-radius: 4px;
          background: var(--paper);
          padding: 1.1rem 1.2rem;
          align-self: start;
        }
        .thermo-trial .side .big {
          font-family: 'Zilla Slab', serif;
          font-weight: 700;
          font-size: 1.3rem;
          color: var(--ink);
          line-height: 1.3;
          margin-bottom: 0.5rem;
        }

        /* Dispatcher / Microgrid */
        .thermo-stat-rows {
          margin-top: 2rem;
          border-top: 2px solid var(--ink);
        }
        .thermo-stat-row {
          display: grid;
          grid-template-columns: minmax(0, 1fr) auto;
          gap: 1rem;
          align-items: baseline;
          padding: 0.85rem 0;
          border-bottom: 1px solid var(--rule);
        }
        .thermo-stat-row .k {
          color: var(--ink-2);
        }
        .thermo-stat-row .k strong {
          color: var(--ink);
        }
        .thermo-stat-row .v {
          font-family: 'JetBrains Mono', monospace;
          font-weight: 700;
          font-size: 1.12rem;
          font-variant-numeric: tabular-nums;
          text-align: right;
        }
        .thermo-stat-row .v.hot {
          color: var(--ember-text);
        }
        .thermo-chart-box {
          margin-top: 2.2rem;
          background: var(--panel);
          border: 1px solid var(--rule);
          border-radius: 4px;
          padding: 1.2rem 1.25rem;
        }
        .thermo-chart-box h3 {
          font-size: 1.3rem;
          margin-bottom: 1rem;
        }
        .thermo-chart-box svg {
          width: 100%;
          height: auto;
          display: block;
        }
        .thermo-chart-cap {
          margin-top: 0.8rem;
          font-size: 0.92rem;
          color: var(--ink-3);
        }
        .thermo-gen-list {
          display: grid;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          gap: 1.2rem;
          margin-top: 1.6rem;
        }
        .thermo-gen-list .g {
          border-top: 2px solid var(--steel);
          padding-top: 0.55rem;
        }
        .thermo-gen-list .g .gn {
          font-family: 'Zilla Slab', serif;
          font-weight: 600;
          font-size: 1.1rem;
        }
        .thermo-gen-list .g .gv {
          font-family: 'JetBrains Mono', monospace;
          font-size: 0.95rem;
          color: var(--steel-deep);
          font-variant-numeric: tabular-nums;
        }

        /* Reproduce & Footer */
        .thermo-repro-links {
          margin-top: 1.8rem;
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 0 clamp(1.8rem, 5vw, 4rem);
        }
        .thermo-repro-links a {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 1rem;
          padding: 1rem 0.2rem;
          border-bottom: 1px solid var(--rule);
          text-decoration: none;
          color: var(--ink);
          font-weight: 600;
          font-size: 1.05rem;
          min-height: 56px;
        }
        .thermo-repro-links a .sub {
          display: block;
          font-weight: 400;
          font-size: 0.9rem;
          color: var(--ink-3);
          font-family: 'JetBrains Mono', monospace;
          margin-top: 0.15rem;
        }
        .thermo-repro-links a:hover {
          color: var(--ember-text);
        }
        .thermo-footer {
          padding: 2.6rem 0 3rem;
          background: var(--paper);
        }
        .thermo-foot-grid {
          display: flex;
          flex-wrap: wrap;
          gap: 2rem;
          justify-content: space-between;
          align-items: flex-start;
        }
        .thermo-footer .fbrand {
          font-family: 'Zilla Slab', serif;
          font-weight: 700;
          font-size: 1.4rem;
        }
        .thermo-footer p {
          color: var(--ink-2);
          max-width: 44ch;
          margin-top: 0.5rem;
        }
        .thermo-foot-links {
          display: flex;
          flex-direction: column;
          gap: 0.2rem;
        }
        .thermo-foot-links a {
          padding: 0.45rem 0;
          min-height: 36px;
          display: inline-flex;
          align-items: center;
          text-decoration: none;
          font-weight: 600;
          color: var(--ink-2);
        }
        .thermo-foot-links a:hover {
          color: var(--ember-text);
        }
        .thermo-foot-mail {
          font-family: 'JetBrains Mono', monospace;
          font-size: 1rem;
        }
        .thermo-foot-base {
          margin-top: 2.2rem;
          border-top: 1px solid var(--rule);
          padding-top: 1rem;
          font-size: 0.9rem;
          color: var(--ink-3);
          display: flex;
          flex-wrap: wrap;
          gap: 1rem;
          justify-content: space-between;
        }

        @media (max-width: 1020px) {
          .thermo-hero-grid {
            grid-template-columns: 1fr;
          }
          .exp-head {
            display: none;
          }
          .exp-row {
            grid-template-columns: 4.6rem minmax(0, 1fr);
            grid-template-rows: auto auto;
          }
          .exp-metric {
            grid-column: 2;
          }
          .exp-side {
            grid-column: 2;
            text-align: left;
          }
          .thermo-cols {
            grid-template-columns: 1fr;
          }
          .thermo-two-col,
          .thermo-trial {
            grid-template-columns: 1fr;
          }
          .thermo-gen-list {
            grid-template-columns: repeat(2, minmax(0, 1fr));
          }
          .thermo-repro-links {
            grid-template-columns: 1fr;
          }
        }
        @media (max-width: 700px) {
          .thermo-formula {
            overflow-x: visible;
          }
          .thermo-formula code {
            white-space: normal;
            word-break: break-word;
            font-size: 0.92rem;
            line-height: 1.7;
          }
        }
        @media (max-width: 480px) {
          .thermo-readouts {
            gap: 0.5rem;
          }
          .thermo-readout .val {
            font-size: 1.05rem;
          }
          .exp-row {
            grid-template-columns: 1fr;
          }
          .exp-metric,
          .exp-side {
            grid-column: 1;
          }
          .thermo-topnav a {
            padding: 0.55rem 0.45rem;
          }
        }
      `}</style>

      {/* Topbar */}
      <header className="thermo-topbar">
        <div className="wrap thermo-topbar-in">
          <Link href="#top" className="thermo-brand" aria-label="Thermo Lab, back to top">
            <svg
              width="22"
              height="22"
              viewBox="0 0 24 24"
              fill="none"
              stroke="#E8490F"
              strokeWidth="2"
              strokeLinecap="round"
              aria-hidden="true"
            >
              <path d="M12 2.5v11" />
              <circle cx="12" cy="17.8" r="3.6" fill="#E8490F" stroke="none" />
            </svg>
            THERMO LAB
          </Link>
          <nav className="thermo-topnav" aria-label="Index">
            <a href="#experiments">Experiments</a>
            <a href="#codon">Codon</a>
            <a href="#dispatcher">Dispatcher</a>
            <a href="#microgrid">Microgrid</a>
            <a href="#reproduce">Reproduce</a>
            <a href="#contact">Contact</a>
          </nav>
        </div>
      </header>

      <main id="top">
        {/* Hero */}
        <div className="thermo-hero">
          <div className="wrap thermo-hero-grid">
            <div>
              <h1>
                <span className="heat">Heat</span> is the computer.
              </h1>
              <p className="thermo-hero-lede">
                Thermodynamic sampling uses energy and temperature to solve hard problems.
                Instead of forcing one answer step by step, it lets a system settle — the
                way a cooling material settles — into a low-energy state that <em>is</em>{" "}
                the answer.
              </p>
              <p className="thermo-hero-copy">
                This bench sheet records the work: sixteen benchmark experiments, a codon
                sampler for protein expression, a task dispatcher, and a power-grid solver.
                The tests aim for better power consumption and fail about 999 times out of
                1000. A single pass would be game-changing, which is why the tests keep
                running.
              </p>
              <div className="thermo-status">
                <svg
                  width="18"
                  height="18"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="#E8490F"
                  strokeWidth="2.2"
                  strokeLinecap="round"
                  aria-hidden="true"
                >
                  <circle cx="12" cy="12" r="4" fill="#E8490F" stroke="none" />
                  <path d="M12 2v3M12 19v3M2 12h3M19 12h3M4.9 4.9l2.1 2.1M17 17l2.1 2.1M19.1 4.9L17 7M7 17l-2.1 2.1" />
                </svg>
                <p>
                  <strong>Open academic evaluation:</strong> available for university
                  biofoundries and synthetic biology research facilities to evaluate
                  the Potts sampler offline against past expression constructs.{" "}
                  <a href="#codon">Read the evaluation notes</a>.
                </p>
              </div>
              <div className="thermo-hero-actions">
                <a className="thermo-btn thermo-btn-solid" href="#experiments">
                  Read the experiments
                </a>
                <a
                  className="thermo-btn thermo-btn-ghost"
                  href="https://github.com/nexusai-kevin/thermo-physics"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  Open the code on GitHub
                </a>
              </div>
            </div>

            <div className="thermo-bench" aria-label="Live annealing lattice demonstration">
              <div className="thermo-bench-head">
                <h2>Annealing lattice, live</h2>
                <span className="mono">16 × 16 Ising spins</span>
              </div>
              <canvas
                ref={canvasRef}
                className="thermo-lattice-canvas"
                width={512}
                height={512}
                role="img"
                aria-label="A 16 by 16 grid of spins. At high temperature the spins are mixed at random; as temperature falls they order into one aligned ground state."
              />
              <div className="thermo-readouts" aria-live="off">
                <div className="thermo-readout">
                  <div className="lab">Temperature T</div>
                  <div className="val hot">{tempVal.toFixed(2)}</div>
                </div>
                <div className="thermo-readout">
                  <div className="lab">Beta, 1 / T</div>
                  <div className="val">{betaVal}</div>
                </div>
                <div className="thermo-readout">
                  <div className="lab">Mean energy / spin</div>
                  <div className="val">{energyVal}</div>
                </div>
                <div className="thermo-readout">
                  <div className="lab">Order, |M|</div>
                  <div className="val">{orderVal}</div>
                </div>
                <div className="thermo-readout">
                  <div className="lab">State</div>
                  <div
                    className="val"
                    style={{
                      fontSize: "1rem",
                      paddingTop: ".2rem",
                      color: stateVal === "Ground state" ? "#B93A08" : undefined,
                    }}
                  >
                    {stateVal}
                  </div>
                </div>
                <div className="thermo-readout">
                  <div className="lab">Sweeps</div>
                  <div className="val" style={{ fontSize: "1rem", paddingTop: ".2rem" }}>
                    {sweepsCount}
                  </div>
                </div>
              </div>
              <div className="thermo-slider-row">
                <div className="thermo-slider-labels">
                  <span className="hot-lab">Hot — disordered</span>
                  <span>Cold — ground state</span>
                </div>
                <input
                  type="range"
                  className="thermo-slider"
                  min="20"
                  max="400"
                  value={sliderVal}
                  step="1"
                  aria-label="Temperature. Left is hot and disordered; drag right, toward cold, to anneal the lattice into its ground state."
                  onChange={(e) => {
                    const v = Number(e.target.value);
                    setSliderVal(v);
                    setT((420 - v) / 100);
                  }}
                />
              </div>
              <div className="thermo-bench-foot">
                <p className="thermo-bench-note">
                  Drag the slider from hot to cold and watch the spins order. Near the
                  ground state the grid settles into one aligned domain — that settling is
                  the computation.
                </p>
                <button
                  className="thermo-btn-anneal"
                  type="button"
                  disabled={isAnnealing}
                  onClick={triggerAnneal}
                >
                  {isAnnealing ? "Annealing…" : "Run anneal"}
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* The Idea */}
        <section className="thermo-blk" id="idea">
          <div className="wrap">
            <h2>A landscape, a temperature, and time to settle.</h2>
            <p className="intro measure">
              Every problem on this page is rewritten the same way: as an energy. Good
              answers sit low in the landscape; bad answers sit high. The sampler's job is
              to find the valleys without getting trapped in the first dip it meets.
            </p>
            <div className="thermo-cols">
              <div>
                <h3>Energy landscapes</h3>
                <p>
                  A solution — a folded RNA, a codon sequence, a dispatch schedule — is
                  scored by a single number, its energy. Optimisation becomes geography:
                  the best answer is the lowest point on the map, and the map can be
                  searched by physics instead of by brute force.
                </p>
              </div>
              <div>
                <h3>Boltzmann sampling</h3>
                <p>
                  At a given temperature the sampler visits states with probability set by
                  the Boltzmann distribution: low-energy states are favoured, but higher
                  ones are still reachable. That controlled randomness is what lets it
                  escape shallow local minima that stop a greedy search cold.
                </p>
              </div>
              <div>
                <h3>Annealing</h3>
                <p>
                  Start hot, where spins flip freely and the system explores. Cool slowly,
                  and the exploration narrows into order. The schedule — how fast the
                  temperature falls — is the difference between a frozen mess and the
                  ground state.
                </p>
              </div>
            </div>
            <p className="thermo-note-line">
              <strong>What runs today:</strong> every result on this page is a classical
              JAX simulation on ordinary computers — no special sampling hardware is
              required or used. Where dedicated thermodynamic hardware is discussed, the
              figures are analytical projections, and they are labelled as projections
              wherever they appear.
            </p>
          </div>
        </section>

        {/* Experiments */}
        <section className="thermo-blk" id="experiments">
          <div className="wrap">
            <h2>Sixteen stations, one method.</h2>
            <p className="intro measure">
              Each station states the same question a different way — can sampling find
              the low-energy answer, and can it prove it against an exact or known result?
              Every row carries its verified metric and the exact command that reproduces
              it.
            </p>

            <div style={{ marginTop: "2.2rem" }}>
              <div className="exp-head" aria-hidden="true">
                <span>Station</span>
                <span>Experiment</span>
                <span>Verified metric</span>
                <span style={{ textAlign: "right" }}>Runtime · Run it</span>
              </div>

              {STATIONS.map((exp) => (
                <div key={exp.id} className="exp-row">
                  <div className="exp-no">
                    EXP <span className="hot">{exp.id}</span>
                  </div>
                  <div>
                    <div className="exp-name">{exp.name}</div>
                    <p className="exp-desc">{exp.desc}</p>
                  </div>
                  <div className="exp-metric">{exp.metric}</div>
                  <div className="exp-side">
                    <div className="exp-time">{exp.runtime}</div>
                    <br />
                    <button
                      className="thermo-cmd"
                      type="button"
                      onClick={() => handleCopy(exp.command, `exp-${exp.id}`)}
                    >
                      {copiedId === `exp-${exp.id}` ? "Copied" : exp.command}
                    </button>
                  </div>
                </div>
              ))}
            </div>

            <div className="thermo-substrate">
              <strong>The substrate.</strong> All sixteen stations run on JAX Block Gibbs
              sampling at <span className="mono">384.5 ESS/s</span> — the classical
              baseline, measured on ordinary hardware. The Extropic TSU figure of{" "}
              <span className="mono">520,833 ESS/s</span> is an{" "}
              <strong>analytical projection</strong> for dedicated hardware, not a
              measurement. <a href="#reading">How to read the numbers</a>.
            </div>
          </div>
        </section>

        {/* Codon */}
        <section className="thermo-blk thermo-featured" id="codon">
          <div className="wrap">
            <h2>Codon design, scored like physics.</h2>
            <p className="intro measure">
              The codon sampler —{" "}
              <span className="mono" style={{ fontSize: ".95em" }}>
                thermo_physics/codon
              </span>{" "}
              — treats a synonymous coding sequence as a Potts energy problem. It scores
              designs for protein expression on 5′ hairpins, homopolymers, restriction
              sites, and the Codon Adaptation Index (CAI), then anneals toward low energy —
              generating diverse synonymous candidate sets instead of one greedy answer.
            </p>

            <div className="cmp-scroll">
              <table
                className="cmp"
                aria-label="Published run on human GLP-1, 15-mer: greedy CAI-only baseline compared with the Potts anneal"
              >
                <thead>
                  <tr>
                    <th scope="col">Human GLP-1 (15-mer)</th>
                    <th scope="col">Greedy, CAI only</th>
                    <th scope="col">Potts anneal</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>CAI</td>
                    <td>1.0000</td>
                    <td>0.9326</td>
                  </tr>
                  <tr>
                    <td>GC content</td>
                    <td>66.7%</td>
                    <td>57.8%</td>
                  </tr>
                  <tr>
                    <td>5′ hairpins</td>
                    <td>4</td>
                    <td className="win">0</td>
                  </tr>
                  <tr>
                    <td>Energy</td>
                    <td>9.1111</td>
                    <td className="win">1.2891</td>
                  </tr>
                  <tr>
                    <td>Energy advantage</td>
                    <td>—</td>
                    <td className="win">7.82</td>
                  </tr>
                  <tr>
                    <td>Search space</td>
                    <td colSpan={2}>84,934,656 synonymous transcripts</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <p className="thermo-note-line">
              The greedy baseline wins CAI and loses the construct: four hairpins near the
              5′ end are exactly the structures that stall expression. The anneal trades a
              little CAI for a clean 5′ end and a far lower total energy. That trade is the
              whole argument for sampling over greed.
            </p>

            <div
              className="thermo-formula"
              role="figure"
              aria-label="Codon Hamiltonian"
            >
              <div className="flab">THE HAMILTONIAN — WHAT THE SAMPLER MINIMISES</div>
              <code>
                {`H(s) = −w_cai · CAI(s) + w_doublet · E_doublet(s) + w_5p · ΔG_5p(s)\n     + w_hairpin · N_hairpins(s) + w_gc · (GC(s) − GC_target)²`}
              </code>
            </div>

            <div className="thermo-two-col">
              <div>
                <h3>Host profiles</h3>
                <p style={{ marginBottom: ".6rem" }}>
                  Each host carries its own codon usage and a target GC the sampler steers
                  toward.
                </p>
                <ul className="rules">
                  <li>
                    <span className="who">
                      <em>S. cerevisiae</em>
                    </span>
                    <span className="what">target GC 39%</span>
                  </li>
                  <li>
                    <span className="who">
                      <em>E. coli</em>
                    </span>
                    <span className="what">target GC 51%</span>
                  </li>
                  <li>
                    <span className="who">
                      <em>D. discoideum</em>
                    </span>
                    <span className="what">AT-rich · target GC 27%</span>
                  </li>
                  <li>
                    <span className="who">
                      <em>B. subtilis</em>
                    </span>
                    <span className="what">target GC 43%</span>
                  </li>
                  <li>
                    <span className="who">
                      <em>C. griseus</em>
                    </span>
                    <span className="what">target GC 50%</span>
                  </li>
                  <li>
                    <span className="who">
                      <em>H. sapiens</em>
                    </span>
                    <span className="what">target GC 52%</span>
                  </li>
                </ul>
              </div>
              <div>
                <h3>Exclusions, off by default</h3>
                <p>
                  Restriction-site screening is configurable and{" "}
                  <strong>disabled by default</strong>, because the foundry workflow
                  assembles by homology (Gibson), not Golden Gate. Switched on, the
                  sampler excludes Type IIS sites — BsaI, BsmBI/Esp3I, BbsI/BpiI,
                  SapI/LguI, AarI — and homopolymer runs of five or more.
                </p>
                <p style={{ marginTop: ".9rem" }}>
                  Nothing about a construct is hidden from the scoring: every term in the
                  Hamiltonian is a named, weighted, inspectable quantity, and the weights
                  ship in the open-source configuration.
                </p>
              </div>
            </div>

            <div className="thermo-trial">
              <div>
                <h3>Academic Biofoundry & Research Lab Pilot</h3>
                <p>
                  Built for university biofoundries and synthetic biology research
                  facilities evaluating offline codon optimization against historical
                  construct libraries: run the offline CLI against expression designs the
                  lab has already produced, and benchmark correlations in private.
                </p>
                <p>
                  The evaluation harness is designed specifically for research facilities
                  that cannot transmit proprietary or pre-publication sequences off-site.
                  It is MIT open-source, makes zero network calls, and collects zero
                  telemetry — running entirely on the laboratory's own infrastructure.
                  Summary exports contain only aggregate statistics: sample counts, Spearman
                  ρ, Pearson r, p-values, and ROC-AUC. Zero sequences and zero construct IDs
                  leave the laboratory.
                </p>
                <p>
                  <strong style={{ color: "var(--ink)" }}>Free permanently</strong> for
                  academic labs and university biofoundries.
                </p>
                <div className="thermo-hero-actions">
                  <a
                    className="thermo-btn thermo-btn-solid"
                    href="mailto:hello@getnexusaisolutions.com?subject=Codon%20sampler%20—%20academic%20evaluation"
                  >
                    Request the evaluation CLI
                  </a>
                  <a className="thermo-btn thermo-btn-ghost" href="/codon">
                    Read the full codon page
                  </a>
                </div>
              </div>
              <div className="side">
                <p className="big">Try it in three seconds, offline.</p>
                <p>The built-in demo needs no files and no network:</p>
                <div className="thermo-codeblock" style={{ marginTop: ".8rem" }}>
                  <code>python scripts/biofoundry_codon_benchmark.py demo</code>
                  <button
                    className="thermo-copy-btn"
                    type="button"
                    onClick={() =>
                      handleCopy(
                        "python scripts/biofoundry_codon_benchmark.py demo",
                        "bio-demo"
                      )
                    }
                  >
                    {copiedId === "bio-demo" ? "Copied" : "Copy"}
                  </button>
                </div>
              </div>
            </div>

            <h3 style={{ fontSize: "1.5rem", marginTop: "2.6rem" }}>
              The three commands a lab actually runs
            </h3>
            <div className="thermo-codeblock">
              <code>
                python scripts/biofoundry_codon_benchmark.py design --seq "MSKGEEL..."
                --host d_discoideum --candidates 5
              </code>
              <button
                className="thermo-copy-btn"
                type="button"
                onClick={() =>
                  handleCopy(
                    'python scripts/biofoundry_codon_benchmark.py design --seq "MSKGEEL..." --host d_discoideum --candidates 5',
                    "bio-cmd-1"
                  )
                }
              >
                {copiedId === "bio-cmd-1" ? "Copied" : "Copy"}
              </button>
            </div>
            <div className="thermo-codeblock">
              <code>
                python scripts/biofoundry_codon_benchmark.py correlate --csv constructs.csv
                --host s_cerevisiae --enzymes none --summary-out
                summary_correlations_only.json
              </code>
              <button
                className="thermo-copy-btn"
                type="button"
                onClick={() =>
                  handleCopy(
                    "python scripts/biofoundry_codon_benchmark.py correlate --csv constructs.csv --host s_cerevisiae --enzymes none --summary-out summary_correlations_only.json",
                    "bio-cmd-2"
                  )
                }
              >
                {copiedId === "bio-cmd-2" ? "Copied" : "Copy"}
              </button>
            </div>
            <p className="thermo-note-line">
              The first designs five synonymous candidates for a Dictyostelium host. The
              second correlates sampler scores against a lab's own construct history — and
              writes out only the aggregate summary named in the command.
            </p>
          </div>
        </section>

        {/* Dispatcher */}
        <section className="thermo-blk" id="dispatcher">
          <div className="wrap">
            <h2>Routing language models like a power grid.</h2>
            <p className="intro measure">
              <span className="mono" style={{ fontSize: ".95em" }}>
                thermo_physics.dispatch
              </span>{" "}
              (MIT) is a Potts rate-limit and SLA task router for batched LLM routing.
              Each task is scheduled on metadata only — prompt text is never a field on the
              task record — and there is no external telemetry. In the published 40-task
              simulation (97,841 tokens), the Gibbs sampler holds the limits that greedy
              routing breaks.
            </p>
            <div className="thermo-stat-rows">
              <div className="thermo-stat-row">
                <span className="k">
                  <strong>Potts / Gibbs dispatcher</strong> — SLA misses
                </span>
                <span className="v">3 / 40</span>
              </div>
              <div className="thermo-stat-row">
                <span className="k">Capability misses</span>
                <span className="v">0 / 40</span>
              </div>
              <div className="thermo-stat-row">
                <span className="k">Tokens over the TPM limit</span>
                <span className="v">0</span>
              </div>
              <div className="thermo-stat-row">
                <span className="k">Solve time</span>
                <span className="v">4.65 ms</span>
              </div>
              <div className="thermo-stat-row">
                <span className="k">
                  Operational model expenditure vs naive high-capability routing
                </span>
                <span className="v hot">36.3% lower</span>
              </div>
              <div className="thermo-stat-row">
                <span className="k">
                  <strong>Greedy cheapest-first</strong>, same 40 tasks — capability
                  misses
                </span>
                <span className="v">18 / 40</span>
              </div>
            </div>
            <h3 style={{ fontSize: "1.4rem", marginTop: "2.4rem" }}>
              Audit your own routing, in shadow
            </h3>
            <p className="measure" style={{ color: "var(--ink-2)", marginTop: ".6rem" }}>
              The shadow-audit CLI replays a request log against a local endpoint and
              reports where the current routing would have missed — without sending
              production traffic anywhere new.
            </p>
            <div className="thermo-codeblock">
              <code>
                python scripts/run_shadow_audit.py --log requests.jsonl --target-endpoint
                localhost:8000 --out audit_report.json
              </code>
              <button
                className="thermo-copy-btn"
                type="button"
                onClick={() =>
                  handleCopy(
                    "python scripts/run_shadow_audit.py --log requests.jsonl --target-endpoint localhost:8000 --out audit_report.json",
                    "shadow-audit"
                  )
                }
              >
                {copiedId === "shadow-audit" ? "Copied" : "Copy"}
              </button>
            </div>
            <div className="thermo-hero-actions">
              <a className="thermo-btn thermo-btn-ghost" href="/dispatcher">
                Read the full dispatcher page
              </a>
            </div>
          </div>
        </section>

        {/* Microgrid */}
        <section className="thermo-blk" id="microgrid">
          <div className="wrap">
            <h2>Unit commitment, solved to the ground state.</h2>
            <p className="intro measure">
              <span className="mono" style={{ fontSize: ".95em" }}>
                thermo_physics.microgrid
              </span>{" "}
              (MIT) writes Security-Constrained Unit Commitment as an Ising QUBO: 4
              generators × 6 intervals = 24 binary variables, and{" "}
              <span className="mono" style={{ fontSize: ".95em" }}>
                16,777,216
              </span>{" "}
              possible schedules. The sampler recovers the deterministic ground state
              across all 16.7 million of them, in 0.12 s of classical JAX.
            </p>

            <div className="thermo-chart-box">
              <h3>Demand to meet, by interval (MW)</h3>
              <svg
                viewBox="0 -18 720 258"
                role="img"
                aria-label="Bar chart of demand across six intervals: 450, 520, 680, 750, 620, and 480 megawatts. Peak demand is 750 megawatts in interval four."
              >
                <line
                  x1="56"
                  y1="196"
                  x2="700"
                  y2="196"
                  stroke="#16222B"
                  strokeWidth="2"
                />
                <g
                  fontFamily="JetBrains Mono, monospace"
                  fontSize="15"
                  fill="#16222B"
                  textAnchor="middle"
                >
                  <rect x="76" y="76" width="72" height="120" fill="#2E5E6B" />
                  <text x="112" y="62" fontWeight="700">
                    450
                  </text>
                  <text x="112" y="220" fill="#46585F">
                    t1
                  </text>

                  <rect x="180" y="57" width="72" height="139" fill="#2E5E6B" />
                  <text x="216" y="43" fontWeight="700">
                    520
                  </text>
                  <text x="216" y="220" fill="#46585F">
                    t2
                  </text>

                  <rect x="284" y="15" width="72" height="181" fill="#2E5E6B" />
                  <text x="320" y="8" fontWeight="700" dy="-4">
                    680
                  </text>
                  <text x="320" y="220" fill="#46585F">
                    t3
                  </text>

                  <rect x="388" y="0" width="72" height="196" fill="#E8490F" />
                  <text x="424" y="220" fill="#46585F">
                    t4
                  </text>
                  <text x="424" y="24" fontWeight="700" fill="#FFFFFF">
                    750
                  </text>

                  <rect x="492" y="31" width="72" height="165" fill="#2E5E6B" />
                  <text x="528" y="17" fontWeight="700">
                    620
                  </text>
                  <text x="528" y="220" fill="#46585F">
                    t5
                  </text>

                  <rect x="596" y="68" width="72" height="128" fill="#2E5E6B" />
                  <text x="632" y="54" fontWeight="700">
                    480
                  </text>
                  <text x="632" y="220" fill="#46585F">
                    t6
                  </text>
                </g>
              </svg>
              <p className="thermo-chart-cap">
                Peak demand — 750 MW in interval 4, shown in ember — is the binding
                constraint the schedule has to survive.
              </p>
            </div>

            <div className="thermo-gen-list" aria-label="Generator fleet">
              <div className="g">
                <div className="gn">Nuclear</div>
                <div className="gv">400 MW</div>
              </div>
              <div className="g">
                <div className="gn">Coal</div>
                <div className="gv">250 MW</div>
              </div>
              <div className="g">
                <div className="gn">CCGT</div>
                <div className="gv">150 MW</div>
              </div>
              <div className="g">
                <div className="gn">Peaker</div>
                <div className="gv">100 MW</div>
              </div>
            </div>

            <div className="thermo-stat-rows">
              <div className="thermo-stat-row">
                <span className="k">Uncoordinated baseline — energy score</span>
                <span className="v">8651.0</span>
              </div>
              <div className="thermo-stat-row">
                <span className="k">
                  <strong>Published schedule</strong> — energy score
                </span>
                <span className="v">5451.0</span>
              </div>
              <div className="thermo-stat-row">
                <span className="k">Improvement</span>
                <span className="v hot">3,200</span>
              </div>
              <div className="thermo-stat-row">
                <span className="k">Ground-state recovery across 16,777,216 schedules</span>
                <span className="v">100% deterministic</span>
              </div>
              <div className="thermo-stat-row">
                <span className="k">Classical JAX runtime</span>
                <span className="v">0.12 s</span>
              </div>
            </div>
            <div className="thermo-hero-actions">
              <a className="thermo-btn thermo-btn-ghost" href="/microgrid">
                Read the full microgrid page
              </a>
            </div>
          </div>
        </section>

        {/* How to Read */}
        <section className="thermo-blk" id="reading">
          <div className="wrap">
            <h2>How to read the numbers.</h2>
            <div className="thermo-cols" style={{ marginTop: "1.6rem" }}>
              <div>
                <h3>Simulations, not customers</h3>
                <p>
                  These are simulated benchmark results, and results may vary. They are
                  not customer results, and no foundry, grid operator, or routing customer
                  is implied by any figure on this page. The one outside trial — the
                  Genome Foundry's — is described as what it is: an agreement to test, with
                  results still to come.
                </p>
              </div>
              <div>
                <h3>Ordinary computers, today</h3>
                <p>
                  All three packages — the benchmark suite, the codon sampler and
                  dispatcher, and the microgrid solver — run today as JAX simulations on
                  ordinary computers. No special sampling hardware is required to
                  reproduce any number here; the run commands on this page are the whole
                  setup.
                </p>
              </div>
              <div>
                <h3>Projections are labelled</h3>
                <p>
                  The Extropic TSU rates are analytical device projections. Physical
                  Extropic TSU silicon is not publicly available, and no projected rate is
                  presented as a measurement anywhere on this page. Measured numbers say{" "}
                  <em>measured</em> by carrying a runtime; projected numbers say{" "}
                  <em>projection</em> in plain words.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Reproduce */}
        <section className="thermo-blk" id="reproduce">
          <div className="wrap">
            <h2>Reproduce everything on this page.</h2>
            <p className="intro measure">
              The code is MIT open source. Clone the repository, run a station, and check
              the metric yourself — that is the intended way to read this bench sheet.
            </p>
            <div className="thermo-repro-links">
              <a
                href="https://github.com/nexusai-kevin/thermo-physics"
                target="_blank"
                rel="noopener noreferrer"
              >
                <span>
                  The repository
                  <span className="sub">github.com/nexusai-kevin/thermo-physics</span>
                </span>
                <svg
                  width="20"
                  height="20"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  aria-hidden="true"
                >
                  <path d="M5 12h14M13 6l6 6-6 6" />
                </svg>
              </a>
              <a
                href="https://github.com/nexusai-kevin/thermo-physics"
                target="_blank"
                rel="noopener noreferrer"
              >
                <span>
                  The codon module<span className="sub">thermo_physics/codon</span>
                </span>
                <svg
                  width="20"
                  height="20"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  aria-hidden="true"
                >
                  <path d="M5 12h14M13 6l6 6-6 6" />
                </svg>
              </a>
              <a href="#experiments">
                <span>
                  The benchmark suite
                  <span className="sub">uv run thrml-bench run --exp N</span>
                </span>
                <svg
                  width="20"
                  height="20"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  aria-hidden="true"
                >
                  <path d="M5 12h14M13 6l6 6-6 6" />
                </svg>
              </a>
              <a href="#codon">
                <span>
                  The biofoundry CLI
                  <span className="sub">scripts/biofoundry_codon_benchmark.py</span>
                </span>
                <svg
                  width="20"
                  height="20"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  aria-hidden="true"
                >
                  <path d="M5 12h14M13 6l6 6-6 6" />
                </svg>
              </a>
            </div>
            <div className="thermo-codeblock">
              <code>
                git clone https://github.com/nexusai-kevin/thermo-physics.git && uv run
                thrml-bench run --exp 11
              </code>
              <button
                className="thermo-copy-btn"
                type="button"
                onClick={() =>
                  handleCopy(
                    "git clone https://github.com/nexusai-kevin/thermo-physics.git && uv run thrml-bench run --exp 11",
                    "clone-repro"
                  )
                }
              >
                {copiedId === "clone-repro" ? "Copied" : "Copy"}
              </button>
            </div>
          </div>
        </section>
      </main>

      <footer className="thermo-footer" id="contact">
        <div className="wrap">
          <div className="thermo-foot-grid">
            <div>
              <div className="fbrand">THERMO LAB</div>
              <p>
                Thermodynamic sampling research from Nexus AI Solutions, New York.
                Questions about the sampler, the trial, or a collaboration start with one
                email.
              </p>
              <p style={{ marginTop: "1rem" }}>
                <a
                  className="thermo-foot-mail"
                  href="mailto:hello@getnexusaisolutions.com"
                >
                  hello@getnexusaisolutions.com
                </a>
              </p>
              <div className="thermo-hero-actions">
                <a
                  className="thermo-btn thermo-btn-solid"
                  href="mailto:hello@getnexusaisolutions.com?subject=Thermo%20Lab%20—%20enquiry"
                >
                  Contact the lab
                </a>
              </div>
            </div>
            <nav className="thermo-foot-links" aria-label="Thermo Lab pages">
              <Link href="/thermo-lab">Thermo Lab</Link>
              <Link href="/codon">Codon</Link>
              <Link href="/dispatcher">Dispatcher</Link>
              <Link href="/microgrid">Microgrid</Link>
              <a
                href="https://github.com/nexusai-kevin/thermo-physics"
                target="_blank"
                rel="noopener noreferrer"
              >
                GitHub — thermo-physics
              </a>
            </nav>
          </div>
          <div className="thermo-foot-base">
            <span>MIT open source. Simulated benchmark results; results may vary.</span>
            <span className="mono">Nexus AI Solutions · New York</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
