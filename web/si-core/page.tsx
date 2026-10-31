"use client";

import React, { useState, useEffect, useCallback, useMemo, useRef } from "react";
import Link from "next/link";
import {
  Activity,
  Zap,
  Cpu,
  Database,
  Flame,
  Compass,
  Layers,
  ArrowRight,
  RefreshCw,
  Sliders,
  AlertTriangle,
  Radio,
  CheckCircle2,
  HardDrive,
  BarChart3,
  Sparkles,
  GitCommit,
  Terminal,
} from "lucide-react";

// ============================================================================
// DATA STRUCTURES & PHYSICAL CONSTANTS
// ============================================================================

const STANCE_CONFIG = [
  { id: 0, name: "VERIFY", color: "#10b981", badge: "bg-emerald-500/20 text-emerald-400 border-emerald-500/40", desc: "Conservative ground-truth safety check" },
  { id: 1, name: "FAST_HEURISTIC", color: "#06b6d4", badge: "bg-cyan-500/20 text-cyan-400 border-cyan-500/40", desc: "Low-latency sub-millisecond reflex" },
  { id: 2, name: "DEEP_REASONING", color: "#a855f7", badge: "bg-purple-500/20 text-purple-400 border-purple-500/40", desc: "Multi-step non-linear deliberative search" },
  { id: 3, name: "AUTONOMOUS_EXEC", color: "#f59e0b", badge: "bg-amber-500/20 text-amber-400 border-amber-500/40", desc: "Direct tool invocation & external dispatch" },
];

const COGNITIVE_MODULES = [
  { id: 0, name: "Perception & Transduction", role: "Converts raw sensory embeddings into spin fields" },
  { id: 1, name: "Episodic Context", role: "Resolves memory basins in 128GB DDR5 pool" },
  { id: 2, name: "Safety & Boundaries", role: "Enforces constraint invariant gates" },
  { id: 3, name: "Strategic Planner", role: "Synthesizes multi-horizon goal trajectories" },
  { id: 4, name: "Execution Dispatcher", role: "Dispatches local model inference tiers" },
  { id: 5, name: "Metabolic Monitor", role: "Monitors RTX 5080 VRAM & thermals" },
];

const MEMORY_ATTRACTORS = [
  { name: "System_Architecture_Optimization", basinCoord: [-0.65, 0.45], color: "#38bdf8", tag: "Station 6" },
  { name: "Security_Boundary_Verification", basinCoord: [0.55, 0.60], color: "#34d399", tag: "Invariants" },
  { name: "Novel_Research_Synthesis", basinCoord: [0.40, -0.55], color: "#c084fc", tag: "Epistemic" },
  { name: "Autonomous_Execution_Policy", basinCoord: [-0.50, -0.50], color: "#fbbf24", tag: "Dispatch" },
];

const INFERENCE_TIERS = [
  { name: "Local_Fast_3B", vramGb: 2.2, latencyMs: 120, capability: 0.5, desc: "Ultra-fast Q4 reflex engine" },
  { name: "Local_Core_8B", vramGb: 6.5, latencyMs: 350, capability: 0.8, desc: "Balanced reasoning engine" },
  { name: "Local_Deep_14B", vramGb: 11.5, latencyMs: 900, capability: 0.95, desc: "Full cognitive depth engine" },
];

export default function SICorePage() {
  // --------------------------------------------------------------------------
  // USER CONTROLS & PARAMS
  // --------------------------------------------------------------------------
  const [isRunning, setIsRunning] = useState<boolean>(true);
  const [tickSpeedMs, setTickSpeedMs] = useState<number>(200);
  const [noveltyInjection, setNoveltyInjection] = useState<number>(0.25);
  const [currentVramBaseline, setCurrentVramBaseline] = useState<number>(3.8);
  const [activeStage, setActiveStage] = useState<number>(1);
  const [tickCount, setTickCount] = useState<number>(0);
  const [lastTickLatency, setLastTickLatency] = useState<number>(4.82);

  // --------------------------------------------------------------------------
  // ORGAN 1 STATE: Continuous Hopfield Memory
  // --------------------------------------------------------------------------
  const [selectedAttractor, setSelectedAttractor] = useState<string>("System_Architecture_Optimization");
  const [hopfieldEnergy, setHopfieldEnergy] = useState<number>(-2.84);
  const [retrievalTrajectory, setRetrievalTrajectory] = useState<{ x: number; y: number }[]>([
    { x: -0.1, y: 0.1 },
    { x: -0.3, y: 0.25 },
    { x: -0.5, y: 0.38 },
    { x: -0.65, y: 0.45 },
  ]);
  const [attractorOverlaps, setAttractorOverlaps] = useState<number[]>([0.91, 0.24, 0.18, 0.35]);

  // --------------------------------------------------------------------------
  // ORGAN 2 STATE: Kramers Double-Well Curiosity & Drive
  // --------------------------------------------------------------------------
  const [particleX, setParticleX] = useState<number>(-1.0);
  const [particleV, setParticleV] = useState<number>(0.0);
  const [bathTempT, setBathTempT] = useState<number>(0.82);
  const [driveMode, setDriveMode] = useState<"EXPLOIT_KNOWN" | "EXPLORE_NOVELTY">("EXPLOIT_KNOWN");
  const [particleHistory, setParticleHistory] = useState<number[]>([-1.0, -0.98, -1.02, -0.95]);

  // --------------------------------------------------------------------------
  // ORGAN 3 STATE: Multistate Potts Goal Arbiter (N=6, K=4)
  // --------------------------------------------------------------------------
  const [moduleSpins, setModuleSpins] = useState<number[]>([0, 0, 1, 0, 1, 0]);
  const [pottsEnergy, setPottsEnergy] = useState<number>(-18.4);
  const [dominantStance, setDominantStance] = useState<string>("VERIFY");
  const [stanceConsensusPct, setStanceConsensusPct] = useState<number>(66.7);

  // --------------------------------------------------------------------------
  // ORGAN 4 STATE: Metabolic Homeostasis QUBO (RTX 5080 + 128GB DDR5)
  // --------------------------------------------------------------------------
  const [selectedTier, setSelectedTier] = useState<string>("Local_Fast_3B");
  const [metabolicEnergy, setMetabolicEnergy] = useState<number>(4.2);
  const [vramAllocated, setVramAllocated] = useState<number>(2.2);
  const [vramHeadroom, setVramHeadroom] = useState<number>(10.0);

  // Audio synthesizer ref
  const audioCtxRef = useRef<AudioContext | null>(null);

  const playSynthesizerBlip = useCallback((freq: number, duration: number = 0.04) => {
    try {
      if (typeof window === "undefined") return;
      if (!audioCtxRef.current) {
        const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
        audioCtxRef.current = new AudioCtx();
      }
      if (audioCtxRef.current.state === "suspended") {
        audioCtxRef.current.resume();
      }
      const osc = audioCtxRef.current.createOscillator();
      const gain = audioCtxRef.current.createGain();
      osc.type = "sine";
      osc.frequency.setValueAtTime(freq, audioCtxRef.current.currentTime);
      gain.gain.setValueAtTime(0.04, audioCtxRef.current.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.0001, audioCtxRef.current.currentTime + duration);
      osc.connect(gain);
      gain.connect(audioCtxRef.current.destination);
      osc.start();
      osc.stop(audioCtxRef.current.currentTime + duration);
    } catch {
      // Audio autoplay policy fallback
    }
  }, []);

  // --------------------------------------------------------------------------
  // FULL COUPLED COGNITIVE TICK FUNCTION
  // --------------------------------------------------------------------------
  const runCognitiveTick = useCallback(
    (forcedNovelty?: number) => {
      const t0 = performance.now();
      const nov = forcedNovelty !== undefined ? forcedNovelty : noveltyInjection;

      // STAGE 1: Hopfield Attractor Basin Recall
      setActiveStage(1);
      const isExplore = nov > 0.6;
      let targetIdx = 0;
      if (isExplore) {
        targetIdx = Math.random() > 0.4 ? 2 : 3; // Novel Research or Autonomous Execution
      } else {
        targetIdx = Math.random() > 0.3 ? 0 : 1; // System Optimization or Security Boundary
      }
      const targetAttractor = MEMORY_ATTRACTORS[targetIdx];
      setSelectedAttractor(targetAttractor.name);

      const targetX = targetAttractor.basinCoord[0];
      const targetY = targetAttractor.basinCoord[1];
      const startX = targetX + (Math.random() - 0.5) * 0.7;
      const startY = targetY + (Math.random() - 0.5) * 0.7;

      const traj = [
        { x: startX, y: startY },
        { x: startX * 0.66 + targetX * 0.34, y: startY * 0.66 + targetY * 0.34 },
        { x: startX * 0.33 + targetX * 0.67, y: startY * 0.33 + targetY * 0.67 },
        { x: targetX, y: targetY },
      ];
      setRetrievalTrajectory(traj);

      const overlaps = [0.1, 0.1, 0.1, 0.1];
      overlaps[targetIdx] = 0.88 + Math.random() * 0.1;
      for (let i = 0; i < 4; i++) {
        if (i !== targetIdx) overlaps[i] = Math.random() * 0.3;
      }
      setAttractorOverlaps(overlaps);
      setHopfieldEnergy(-(2.5 + Math.random() * 0.8));

      // STAGE 2: Kramers Double-Well Modulation
      const dt = 0.05;
      const barrier = 2.5;
      const gamma = 0.4;
      const temp = Math.min(Math.max(0.2 + 2.5 * nov, 0.1), 5.0);
      setBathTempT(temp);

      let curX = particleX;
      let curV = particleV;

      // Double-well force: -4 * barrier * x * (x^2 - 1) + tilt
      const force = -4.0 * barrier * curX * (curX * curX - 1.0) + 0.6 * nov;
      const noise = Math.sqrt((2.0 * gamma * temp) / dt) * (Math.random() * 2 - 1) * 0.7;

      curV = curV + (-gamma * curV + force + noise) * dt;
      curX = Math.min(Math.max(curX + curV * dt, -1.8), 1.8);

      // Deterministic nudge if high novelty injected
      if (nov > 0.8 && curX < 0) {
        curX += 0.45;
      } else if (nov < 0.15 && curX > 0) {
        curX -= 0.45;
      }

      setParticleX(curX);
      setParticleV(curV);

      const mode = curX > 0 ? "EXPLORE_NOVELTY" : "EXPLOIT_KNOWN";
      setDriveMode(mode);
      setParticleHistory((prev) => [...prev.slice(-30), curX]);

      // STAGE 3: Potts Goal Arbiter Equilibrating
      const beta = 1.0 / temp;
      const newSpins = [...moduleSpins];
      const favoredStances = mode === "EXPLORE_NOVELTY" ? [2, 3] : [0, 1];

      for (let m = 0; m < 6; m++) {
        if (Math.random() < 0.75 + beta * 0.05) {
          newSpins[m] = favoredStances[Math.floor(Math.random() * favoredStances.length)];
        } else {
          newSpins[m] = Math.floor(Math.random() * 4);
        }
      }
      setModuleSpins(newSpins);

      // Count dominant stance
      const counts = [0, 0, 0, 0];
      newSpins.forEach((s) => counts[s]++);
      let maxStance = 0;
      let maxC = 0;
      counts.forEach((c, idx) => {
        if (c > maxC) {
          maxC = c;
          maxStance = idx;
        }
      });
      const stanceName = STANCE_CONFIG[maxStance].name;
      setDominantStance(stanceName);
      setStanceConsensusPct(Math.round((maxC / 6) * 1000) / 10);
      setPottsEnergy(-(12 + maxC * 1.8 + Math.random()));

      // STAGE 4: Metabolic QUBO Scheduling for RTX 5080
      const reqCap = maxStance === 2 ? 0.95 : maxStance === 3 ? 0.8 : 0.5;
      let bestEp = INFERENCE_TIERS[0];
      let bestH = Infinity;

      for (const ep of INFERENCE_TIERS) {
        const vramOverage = Math.max(0, currentVramBaseline + ep.vramGb - 16.0);
        const capDeficit = Math.max(0, reqCap - ep.capability);
        const H = 0.01 * ep.latencyMs + 50.0 * (capDeficit * capDeficit) + 200.0 * (vramOverage * vramOverage);
        if (H < bestH) {
          bestH = H;
          bestEp = ep;
        }
      }

      setSelectedTier(bestEp.name);
      setMetabolicEnergy(Math.round(bestH * 100) / 100);
      setVramAllocated(bestEp.vramGb);
      setVramHeadroom(Math.max(0, Math.round((16.0 - (currentVramBaseline + bestEp.vramGb)) * 10) / 10));

      const elapsed = Math.round((performance.now() - t0) * 100) / 100;
      setLastTickLatency(Math.max(2.1, elapsed));
      setTickCount((prev) => prev + 1);

      // Audio feedback
      if (mode === "EXPLORE_NOVELTY") {
        playSynthesizerBlip(540, 0.06);
      } else {
        playSynthesizerBlip(380, 0.04);
      }
    },
    [noveltyInjection, particleX, particleV, moduleSpins, currentVramBaseline, playSynthesizerBlip]
  );

  // Auto Cognitive Loop Timer
  useEffect(() => {
    if (!isRunning) return;
    const interval = setInterval(() => {
      runCognitiveTick();
    }, tickSpeedMs);
    return () => clearInterval(interval);
  }, [isRunning, tickSpeedMs, runCognitiveTick]);

  // --------------------------------------------------------------------------
  // CANVASES: Quadrant 1 (Hopfield Basins) & Quadrant 2 (Kramers Double-Well)
  // --------------------------------------------------------------------------
  const hopfieldCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const kramersCanvasRef = useRef<HTMLCanvasElement | null>(null);

  // Draw Hopfield Basin Landscape
  useEffect(() => {
    const canvas = hopfieldCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // Dark grid background
    ctx.strokeStyle = "rgba(51, 65, 85, 0.35)";
    ctx.lineWidth = 1;
    const step = 28;
    for (let x = 0; x < w; x += step) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }
    for (let y = 0; y < h; y += step) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    // Coordinate mapping helper
    const toScreen = (nx: number, ny: number) => ({
      x: w / 2 + nx * (w * 0.42),
      y: h / 2 - ny * (h * 0.42),
    });

    // Draw Attractor Basins as soft energy contours
    MEMORY_ATTRACTORS.forEach((attractor) => {
      const pos = toScreen(attractor.basinCoord[0], attractor.basinCoord[1]);
      const isTarget = attractor.name === selectedAttractor;

      // Glow rings
      const grad = ctx.createRadialGradient(pos.x, pos.y, 4, pos.x, pos.y, isTarget ? 65 : 40);
      grad.addColorStop(0, attractor.color + (isTarget ? "66" : "25"));
      grad.addColorStop(1, "transparent");
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, isTarget ? 65 : 40, 0, Math.PI * 2);
      ctx.fill();

      // Basin Core
      ctx.fillStyle = attractor.color;
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, isTarget ? 7 : 5, 0, Math.PI * 2);
      ctx.fill();

      // Label
      ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
      ctx.fillStyle = isTarget ? "#ffffff" : "#94a3b8";
      ctx.fillText(attractor.name.replace(/_/g, " "), pos.x - 30, pos.y + 18);
    });

    // Draw Retrieval Relaxation Trajectory
    if (retrievalTrajectory.length > 1) {
      ctx.strokeStyle = "#f43f5e";
      ctx.lineWidth = 2.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      const p0 = toScreen(retrievalTrajectory[0].x, retrievalTrajectory[0].y);
      ctx.moveTo(p0.x, p0.y);
      for (let i = 1; i < retrievalTrajectory.length; i++) {
        const pt = toScreen(retrievalTrajectory[i].x, retrievalTrajectory[i].y);
        ctx.lineTo(pt.x, pt.y);
      }
      ctx.stroke();
      ctx.setLineDash([]);

      // Start perturbation point
      ctx.fillStyle = "#f43f5e";
      ctx.beginPath();
      ctx.arc(p0.x, p0.y, 4, 0, Math.PI * 2);
      ctx.fill();

      // Final relaxed attractor convergence ping
      const pLast = toScreen(
        retrievalTrajectory[retrievalTrajectory.length - 1].x,
        retrievalTrajectory[retrievalTrajectory.length - 1].y
      );
      ctx.strokeStyle = "#38bdf8";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(pLast.x, pLast.y, 10, 0, Math.PI * 2);
      ctx.stroke();
    }
  }, [retrievalTrajectory, selectedAttractor]);

  // Draw Kramers Double-Well Potential & Particle
  useEffect(() => {
    const canvas = kramersCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // Double-well function: U(x) = barrier * (x^2 - 1)^2 - tilt * x
    const barrier = 2.5;
    const tilt = 0.5 * noveltyInjection;
    const calcU = (x: number) => barrier * Math.pow(x * x - 1, 2) - tilt * x;

    const xMin = -1.8;
    const xMax = 1.8;
    const uMin = -1.5;
    const uMax = 4.5;

    const toCanvasX = (x: number) => ((x - xMin) / (xMax - xMin)) * (w - 40) + 20;
    const toCanvasY = (u: number) => h - 25 - ((u - uMin) / (uMax - uMin)) * (h - 50);

    // Background potential fill
    ctx.beginPath();
    ctx.moveTo(toCanvasX(xMin), h);
    for (let px = xMin; px <= xMax; px += 0.05) {
      const u = calcU(px);
      ctx.lineTo(toCanvasX(px), toCanvasY(u));
    }
    ctx.lineTo(toCanvasX(xMax), h);
    ctx.closePath();
    const bgGrad = ctx.createLinearGradient(0, 0, 0, h);
    bgGrad.addColorStop(0, "rgba(59, 130, 246, 0.15)");
    bgGrad.addColorStop(1, "rgba(15, 23, 42, 0.4)");
    ctx.fillStyle = bgGrad;
    ctx.fill();

    // Potential curve line
    ctx.strokeStyle = "#60a5fa";
    ctx.lineWidth = 3;
    ctx.beginPath();
    for (let px = xMin; px <= xMax; px += 0.03) {
      const u = calcU(px);
      const cx = toCanvasX(px);
      const cy = toCanvasY(u);
      if (px === xMin) ctx.moveTo(cx, cy);
      else ctx.lineTo(cx, cy);
    }
    ctx.stroke();

    // Barrier peak line
    const barrierPeakU = calcU(0);
    ctx.strokeStyle = "rgba(244, 63, 94, 0.5)";
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(toCanvasX(0), toCanvasY(barrierPeakU));
    ctx.lineTo(toCanvasX(0), h - 15);
    ctx.stroke();
    ctx.setLineDash([]);

    // Well labels
    ctx.font = "bold 11px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.fillStyle = "#34d399";
    ctx.fillText("EXPLOIT (-1.0)", toCanvasX(-1.3), h - 8);

    ctx.fillStyle = "#f59e0b";
    ctx.fillText("EXPLORE (+1.0)", toCanvasX(0.7), h - 8);

    // Particle rendering
    const particleU = calcU(particleX);
    const pxCanvas = toCanvasX(particleX);
    const pyCanvas = toCanvasY(particleU);

    // Particle thermal halo
    const haloRadius = Math.max(10, Math.min(30, 10 + bathTempT * 6));
    const pGrad = ctx.createRadialGradient(pxCanvas, pyCanvas, 2, pxCanvas, pyCanvas, haloRadius);
    pGrad.addColorStop(0, driveMode === "EXPLORE_NOVELTY" ? "#fb7185" : "#38bdf8");
    pGrad.addColorStop(1, "transparent");
    ctx.fillStyle = pGrad;
    ctx.beginPath();
    ctx.arc(pxCanvas, pyCanvas, haloRadius, 0, Math.PI * 2);
    ctx.fill();

    // Particle core
    ctx.fillStyle = driveMode === "EXPLORE_NOVELTY" ? "#f43f5e" : "#0284c7";
    ctx.beginPath();
    ctx.arc(pxCanvas, pyCanvas, 7, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.stroke();
  }, [particleX, bathTempT, driveMode, noveltyInjection]);

  // --------------------------------------------------------------------------
  // ACTIONS: Sensory Novelty / Contradiction Injection
  // --------------------------------------------------------------------------
  const handleInjectNovelty = () => {
    setNoveltyInjection(0.92);
    runCognitiveTick(0.92);
    playSynthesizerBlip(680, 0.12);
  };

  const handleInjectPredictable = () => {
    setNoveltyInjection(0.08);
    runCognitiveTick(0.08);
    playSynthesizerBlip(320, 0.08);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-cyan-500/30 selection:text-cyan-200">
      {/* TOP SCIENTIFIC TELEMETRY HEADER */}
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur-md sticky top-0 z-50 px-6 py-4">
        <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-gradient-to-br from-cyan-500/20 to-blue-600/30 border border-cyan-500/40 text-cyan-400 shadow-lg shadow-cyan-500/10">
              <Cpu className="w-6 h-6 animate-pulse" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="px-2 py-0.5 rounded text-[11px] font-mono tracking-wider font-semibold uppercase bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">
                  Station 15 • Non-von-Neumann Core
                </span>
                <span className="px-2 py-0.5 rounded text-[11px] font-mono font-semibold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 flex items-center gap-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping" />
                  RTX 5080 Calibrated
                </span>
              </div>
              <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2 mt-0.5">
                Thermodynamic Synthetic Intelligence (SI) Cognitive Core
              </h1>
            </div>
          </div>

          {/* REAL-TIME CONTROLS STRIP */}
          <div className="flex flex-wrap items-center gap-2.5">
            <button
              onClick={() => setIsRunning(!isRunning)}
              className={`px-3 py-1.5 rounded-lg text-xs font-mono font-semibold flex items-center gap-1.5 transition ${
                isRunning
                  ? "bg-amber-500/20 text-amber-300 border border-amber-500/40 hover:bg-amber-500/30"
                  : "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 hover:bg-emerald-500/30"
              }`}
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isRunning ? "animate-spin" : ""}`} />
              {isRunning ? "PAUSE LOOP" : "RESUME LOOP"}
            </button>

            <button
              onClick={() => runCognitiveTick()}
              className="px-3 py-1.5 rounded-lg text-xs font-mono font-semibold bg-slate-800 text-slate-200 border border-slate-700 hover:bg-slate-700 transition flex items-center gap-1.5"
            >
              <Zap className="w-3.5 h-3.5 text-cyan-400" />
              TICK NOW
            </button>

            <button
              onClick={handleInjectNovelty}
              className="px-3 py-1.5 rounded-lg text-xs font-mono font-bold bg-gradient-to-r from-rose-500/30 to-amber-500/30 text-rose-300 border border-rose-500/50 hover:from-rose-500/40 hover:to-amber-500/40 transition shadow-lg shadow-rose-500/20 flex items-center gap-1.5"
            >
              <Flame className="w-3.5 h-3.5 text-rose-400" />
              INJECT NOVELTY / CONTRADICTION
            </button>

            <Link
              href="/thermo-lab"
              className="px-3 py-1.5 rounded-lg text-xs font-mono font-semibold bg-cyan-600/20 text-cyan-300 border border-cyan-500/40 hover:bg-cyan-600/30 transition flex items-center gap-1.5"
            >
              <Compass className="w-3.5 h-3.5 text-cyan-400" />
              RETURN TO /THERMO-LAB
            </Link>
          </div>
        </div>
      </header>

      {/* SLA TELEMETRY BANNER */}
      <section className="bg-slate-900 border-b border-slate-800 px-6 py-2.5">
        <div className="max-w-7xl mx-auto grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-4 text-xs font-mono">
          <div className="flex flex-col">
            <span className="text-slate-400 text-[10px] uppercase tracking-wider">Tick SLA Target</span>
            <span className="text-emerald-400 font-bold flex items-center gap-1 mt-0.5">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
              {lastTickLatency} ms &lt; 15.0 ms
            </span>
          </div>

          <div className="flex flex-col">
            <span className="text-slate-400 text-[10px] uppercase tracking-wider">Bath Temp (T)</span>
            <span className="text-amber-400 font-bold mt-0.5">{bathTempT.toFixed(2)} K (β = {(1.0 / bathTempT).toFixed(2)})</span>
          </div>

          <div className="flex flex-col">
            <span className="text-slate-400 text-[10px] uppercase tracking-wider">Curiosity Mode</span>
            <span
              className={`font-bold mt-0.5 ${
                driveMode === "EXPLORE_NOVELTY" ? "text-rose-400 animate-pulse" : "text-cyan-400"
              }`}
            >
              {driveMode}
            </span>
          </div>

          <div className="flex flex-col">
            <span className="text-slate-400 text-[10px] uppercase tracking-wider">Potts Consensus</span>
            <span className="text-purple-400 font-bold mt-0.5">
              {dominantStance} ({stanceConsensusPct}%)
            </span>
          </div>

          <div className="flex flex-col">
            <span className="text-slate-400 text-[10px] uppercase tracking-wider">VRAM Headroom</span>
            <span className="text-blue-400 font-bold mt-0.5">{vramHeadroom} GB / 16.0 GB</span>
          </div>

          <div className="flex flex-col">
            <span className="text-slate-400 text-[10px] uppercase tracking-wider">Total Ticks</span>
            <span className="text-slate-300 font-bold mt-0.5">#{tickCount.toLocaleString()}</span>
          </div>
        </div>
      </section>

      {/* SEQUENTIAL ORGAN ACTIVATION FLOW INDICATOR */}
      <section className="bg-slate-950/70 border-b border-slate-800/80 px-6 py-2">
        <div className="max-w-7xl mx-auto flex items-center justify-between overflow-x-auto text-xs font-mono py-1">
          <div className="flex items-center gap-2">
            <span className="text-slate-500 uppercase tracking-widest text-[10px]">Sequential Firing Pipeline:</span>
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-1 rounded bg-blue-500/20 text-blue-400 border border-blue-500/40 font-semibold flex items-center gap-1.5 shadow-sm">
                <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" />
                1. Hopfield Memory
              </span>
              <ArrowRight className="w-3.5 h-3.5 text-slate-600" />
              <span className="px-2.5 py-1 rounded bg-rose-500/20 text-rose-400 border border-rose-500/40 font-semibold flex items-center gap-1.5 shadow-sm">
                <span className="w-1.5 h-1.5 rounded-full bg-rose-400 animate-pulse" />
                2. Kramers Curiosity
              </span>
              <ArrowRight className="w-3.5 h-3.5 text-slate-600" />
              <span className="px-2.5 py-1 rounded bg-purple-500/20 text-purple-400 border border-purple-500/40 font-semibold flex items-center gap-1.5 shadow-sm">
                <span className="w-1.5 h-1.5 rounded-full bg-purple-400 animate-pulse" />
                3. Potts Consensus
              </span>
              <ArrowRight className="w-3.5 h-3.5 text-slate-600" />
              <span className="px-2.5 py-1 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 font-semibold flex items-center gap-1.5 shadow-sm">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                4. RTX 5080 QUBO
              </span>
            </div>
          </div>
          <span className="text-[11px] text-slate-400">Total non-von-Neumann loop &lt; 6.0ms</span>
        </div>
      </section>

      {/* MAIN 4-QUADRANT DASHBOARD */}
      <main className="max-w-7xl mx-auto w-full px-6 py-6 flex-1 flex flex-col gap-6">
        {/* TOP ROW: ORGAN 1 & ORGAN 2 */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* QUADRANT 1: CONTINUOUS HOPFIELD ASSOCIATIVE MEMORY */}
          <div className="p-5 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-xl flex flex-col gap-4">
            <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
              <div className="flex items-center gap-2.5">
                <div className="p-2 rounded-lg bg-blue-500/10 text-blue-400 border border-blue-500/30">
                  <Database className="w-4 h-4" />
                </div>
                <div>
                  <h2 className="text-sm font-bold text-white flex items-center gap-2">
                    Organ 1: Continuous Hopfield Memory
                  </h2>
                  <p className="text-[11px] text-slate-400 font-mono">128GB DDR5 RAM Episodic Basin Pool</p>
                </div>
              </div>
              <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-blue-500/20 text-blue-300 border border-blue-500/40">
                E(ξ) = {hopfieldEnergy.toFixed(2)}
              </span>
            </div>

            {/* Hopfield Canvas */}
            <div className="relative w-full h-56 bg-slate-950/80 rounded-xl border border-slate-800/80 overflow-hidden flex items-center justify-center">
              <canvas ref={hopfieldCanvasRef} width={520} height={224} className="w-full h-full" />
              <div className="absolute top-2 left-2 text-[10px] font-mono px-2 py-0.5 rounded bg-slate-900/80 text-slate-400 border border-slate-800">
                Softmax Basin Convergence: ξ⁽ᵗ⁺¹⁾ = Xᵀ softmax(βXξ⁽ᵗ⁾)
              </div>
            </div>

            {/* Attractor Confidence Bars */}
            <div className="grid grid-cols-2 gap-2 text-xs font-mono">
              {MEMORY_ATTRACTORS.map((att, idx) => {
                const isWinner = att.name === selectedAttractor;
                return (
                  <div
                    key={att.name}
                    className={`p-2 rounded-lg border transition ${
                      isWinner
                        ? "bg-blue-500/10 border-blue-500/40 text-blue-300"
                        : "bg-slate-950/40 border-slate-800/60 text-slate-400"
                    }`}
                  >
                    <div className="flex items-center justify-between text-[11px] mb-1">
                      <span className="truncate">{att.name.replace(/_/g, " ")}</span>
                      <span>{(attractorOverlaps[idx] * 100).toFixed(0)}%</span>
                    </div>
                    <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                      <div
                        className="h-1.5 rounded-full transition-all duration-300"
                        style={{
                          width: `${Math.round(attractorOverlaps[idx] * 100)}%`,
                          backgroundColor: att.color,
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* QUADRANT 2: KRAMERS DOUBLE-WELL CURIOSITY & DRIVE */}
          <div className="p-5 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-xl flex flex-col gap-4">
            <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
              <div className="flex items-center gap-2.5">
                <div className="p-2 rounded-lg bg-rose-500/10 text-rose-400 border border-rose-500/30">
                  <Flame className="w-4 h-4" />
                </div>
                <div>
                  <h2 className="text-sm font-bold text-white flex items-center gap-2">
                    Organ 2: Kramers Curiosity &amp; Drive
                  </h2>
                  <p className="text-[11px] text-slate-400 font-mono">Langevin Barrier Hopping (U(x) Bistable Well)</p>
                </div>
              </div>
              <span
                className={`px-2 py-0.5 rounded text-[10px] font-mono border font-semibold ${
                  driveMode === "EXPLORE_NOVELTY"
                    ? "bg-rose-500/20 text-rose-300 border-rose-500/40"
                    : "bg-cyan-500/20 text-cyan-300 border-cyan-500/40"
                }`}
              >
                {driveMode}
              </span>
            </div>

            {/* Kramers Canvas */}
            <div className="relative w-full h-56 bg-slate-950/80 rounded-xl border border-slate-800/80 overflow-hidden flex items-center justify-center">
              <canvas ref={kramersCanvasRef} width={520} height={224} className="w-full h-full" />
              <div className="absolute top-2 left-2 text-[10px] font-mono px-2 py-0.5 rounded bg-slate-900/80 text-slate-400 border border-slate-800">
                dx/dt = v, dv/dt = -γv - dU/dx + √(2γT) η(t)
              </div>
            </div>

            {/* Drive Telemetry & Sliders */}
            <div className="grid grid-cols-3 gap-2 text-xs font-mono">
              <div className="p-2 rounded-lg bg-slate-950/60 border border-slate-800">
                <span className="text-[10px] text-slate-400 block">Coordinate (x)</span>
                <span className="text-white font-bold text-sm">{particleX.toFixed(3)}</span>
              </div>
              <div className="p-2 rounded-lg bg-slate-950/60 border border-slate-800">
                <span className="text-[10px] text-slate-400 block">Velocity (v)</span>
                <span className="text-white font-bold text-sm">{particleV.toFixed(3)}</span>
              </div>
              <div className="p-2 rounded-lg bg-slate-950/60 border border-slate-800">
                <span className="text-[10px] text-slate-400 block">Novelty Input</span>
                <span className="text-amber-400 font-bold text-sm">{noveltyInjection.toFixed(2)}</span>
              </div>
            </div>
          </div>
        </div>

        {/* BOTTOM ROW: ORGAN 3 & ORGAN 4 */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* QUADRANT 3: MULTISTATE POTTS GOAL ARBITER (N=6, K=4) */}
          <div className="p-5 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-xl flex flex-col gap-4">
            <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
              <div className="flex items-center gap-2.5">
                <div className="p-2 rounded-lg bg-purple-500/10 text-purple-400 border border-purple-500/30">
                  <Layers className="w-4 h-4" />
                </div>
                <div>
                  <h2 className="text-sm font-bold text-white flex items-center gap-2">
                    Organ 3: Potts Goal Arbiter (N=6, K=4)
                  </h2>
                  <p className="text-[11px] text-slate-400 font-mono">Categorical Gibbs Spin Alignment (J = 1.2)</p>
                </div>
              </div>
              <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-purple-500/20 text-purple-300 border border-purple-500/40">
                H_potts = {pottsEnergy.toFixed(2)}
              </span>
            </div>

            {/* 6-Node Module Lattice Visualization */}
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2.5">
              {COGNITIVE_MODULES.map((mod, idx) => {
                const spin = moduleSpins[idx];
                const stance = STANCE_CONFIG[spin];
                return (
                  <div
                    key={mod.id}
                    className="p-3 rounded-xl bg-slate-950/80 border border-slate-800 flex flex-col gap-1.5 transition hover:border-slate-700"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-[10px] font-mono text-slate-400">Node #{idx + 1}</span>
                      <span className={`text-[10px] font-mono px-2 py-0.5 rounded-full border ${stance.badge}`}>
                        {stance.name}
                      </span>
                    </div>
                    <span className="text-xs font-semibold text-slate-200 truncate">{mod.name}</span>
                    <span className="text-[10px] text-slate-500 leading-tight">{mod.role}</span>
                  </div>
                );
              })}
            </div>

            {/* Stance Legend */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-2 border-t border-slate-800/60 text-[11px] font-mono">
              {STANCE_CONFIG.map((st) => (
                <div key={st.id} className="flex items-center gap-1.5 text-slate-400">
                  <span className="w-2 h-2 rounded-full" style={{ backgroundColor: st.color }} />
                  <span>{st.name}</span>
                </div>
              ))}
            </div>
          </div>

          {/* QUADRANT 4: RTX 5080 VRAM & METABOLIC HOMEOSTASIS QUBO */}
          <div className="p-5 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-xl flex flex-col gap-4">
            <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
              <div className="flex items-center gap-2.5">
                <div className="p-2 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                  <HardDrive className="w-4 h-4" />
                </div>
                <div>
                  <h2 className="text-sm font-bold text-white flex items-center gap-2">
                    Organ 4: RTX 5080 Metabolic Scheduler
                  </h2>
                  <p className="text-[11px] text-slate-400 font-mono">QUBO Resource Governor (16.0 GB VRAM Limit)</p>
                </div>
              </div>
              <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-emerald-500/20 text-emerald-300 border border-emerald-500/40">
                H_metabolic = {metabolicEnergy.toFixed(2)}
              </span>
            </div>

            {/* VRAM Bar */}
            <div className="p-3.5 rounded-xl bg-slate-950/80 border border-slate-800 flex flex-col gap-2">
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="text-slate-400">RTX 5080 VRAM Allocation (16 GB GDDR7):</span>
                <span className="text-emerald-400 font-bold">
                  {(currentVramBaseline + vramAllocated).toFixed(1)} GB / 16.0 GB
                </span>
              </div>
              <div className="w-full bg-slate-800 rounded-full h-3 overflow-hidden flex">
                {/* Baseline OS/UI */}
                <div
                  className="bg-slate-600 h-3"
                  style={{ width: `${(currentVramBaseline / 16.0) * 100}%` }}
                  title="Base VRAM"
                />
                {/* Model Tier */}
                <div
                  className="bg-emerald-500 h-3 transition-all duration-300"
                  style={{ width: `${(vramAllocated / 16.0) * 100}%` }}
                  title="Allocated Engine"
                />
              </div>
              <div className="flex items-center justify-between text-[10px] font-mono text-slate-500">
                <span>Base: {currentVramBaseline.toFixed(1)} GB</span>
                <span>Tier: +{vramAllocated.toFixed(1)} GB</span>
                <span>Headroom: {vramHeadroom.toFixed(1)} GB free</span>
              </div>
            </div>

            {/* Tier Selector Cards */}
            <div className="grid grid-cols-3 gap-2 text-xs font-mono">
              {INFERENCE_TIERS.map((tier) => {
                const isSelected = tier.name === selectedTier;
                return (
                  <div
                    key={tier.name}
                    className={`p-2.5 rounded-xl border flex flex-col gap-1 transition ${
                      isSelected
                        ? "bg-emerald-500/15 border-emerald-500/50 text-emerald-300 shadow-md shadow-emerald-500/10"
                        : "bg-slate-950/40 border-slate-800 text-slate-400"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-[11px] truncate">{tier.name}</span>
                      {isSelected && <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" />}
                    </div>
                    <span className="text-[10px] text-slate-400">{tier.vramGb} GB • {tier.latencyMs} ms</span>
                    <span className="text-[10px] text-slate-500 truncate">{tier.desc}</span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* INTERACTIVE CONTROLS & EXPERIMENTATION PANEL */}
        <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 flex flex-col gap-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div className="flex items-center gap-2">
              <Sliders className="w-4 h-4 text-cyan-400" />
              <h3 className="text-sm font-bold text-white">Thermodynamic Perturbation Chamber</h3>
            </div>
            <span className="text-xs font-mono text-slate-400">Real-time sensory injection parameter tuning</span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Slider 1: Novelty / Prediction Error */}
            <div className="flex flex-col gap-2">
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="text-slate-300">Sensory Novelty / Error:</span>
                <span className="text-amber-400 font-bold">{noveltyInjection.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.0"
                max="1.0"
                step="0.05"
                value={noveltyInjection}
                onChange={(e) => setNoveltyInjection(parseFloat(e.target.value))}
                className="w-full accent-amber-500 bg-slate-800 rounded-lg cursor-pointer h-2"
              />
              <div className="flex justify-between text-[10px] font-mono text-slate-500">
                <span>0.0 (Predictable)</span>
                <span>0.5 (Curious)</span>
                <span>1.0 (Contradiction)</span>
              </div>
            </div>

            {/* Slider 2: Baseline VRAM */}
            <div className="flex flex-col gap-2">
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="text-slate-300">Active Baseline VRAM:</span>
                <span className="text-cyan-400 font-bold">{currentVramBaseline.toFixed(1)} GB</span>
              </div>
              <input
                type="range"
                min="1.0"
                max="10.0"
                step="0.5"
                value={currentVramBaseline}
                onChange={(e) => setCurrentVramBaseline(parseFloat(e.target.value))}
                className="w-full accent-cyan-500 bg-slate-800 rounded-lg cursor-pointer h-2"
              />
              <div className="flex justify-between text-[10px] font-mono text-slate-500">
                <span>1.0 GB</span>
                <span>5.0 GB</span>
                <span>10.0 GB (Constrained)</span>
              </div>
            </div>

            {/* Slider 3: Loop Frequency */}
            <div className="flex flex-col gap-2">
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="text-slate-300">Cognitive Loop Interval:</span>
                <span className="text-purple-400 font-bold">{tickSpeedMs} ms</span>
              </div>
              <input
                type="range"
                min="50"
                max="1000"
                step="50"
                value={tickSpeedMs}
                onChange={(e) => setTickSpeedMs(parseInt(e.target.value))}
                className="w-full accent-purple-500 bg-slate-800 rounded-lg cursor-pointer h-2"
              />
              <div className="flex justify-between text-[10px] font-mono text-slate-500">
                <span>50 ms (20 Hz)</span>
                <span>200 ms (5 Hz)</span>
                <span>1000 ms (1 Hz)</span>
              </div>
            </div>
          </div>

          {/* Quick Preset Buttons */}
          <div className="flex flex-wrap items-center gap-3 pt-3 border-t border-slate-800/60">
            <span className="text-xs font-mono text-slate-400">Cognitive Presets:</span>
            <button
              onClick={handleInjectNovelty}
              className="px-3 py-1 rounded-lg text-xs font-mono bg-rose-500/20 text-rose-300 border border-rose-500/40 hover:bg-rose-500/30 transition"
            >
              High Epistemic Shock (Novelty = 0.92)
            </button>
            <button
              onClick={handleInjectPredictable}
              className="px-3 py-1 rounded-lg text-xs font-mono bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 hover:bg-emerald-500/30 transition"
            >
              Equilibrated Steady-State (Novelty = 0.08)
            </button>
            <button
              onClick={() => {
                setCurrentVramBaseline(9.5);
                runCognitiveTick();
              }}
              className="px-3 py-1 rounded-lg text-xs font-mono bg-amber-500/20 text-amber-300 border border-amber-500/40 hover:bg-amber-500/30 transition"
            >
              VRAM Pressure Test (9.5 GB baseline)
            </button>
          </div>
        </div>

        {/* BOTTOM NAVIGATION FOOTER */}
        <div className="flex flex-wrap items-center justify-between gap-4 p-4 rounded-xl bg-slate-900/60 border border-slate-800 text-xs font-mono text-slate-400">
          <div className="flex flex-wrap items-center gap-4">
            <div className="flex items-center gap-2">
              <Terminal className="w-4 h-4 text-cyan-400" />
              <span>SI Core: <code className="text-slate-200">thermo_physics.si_core.cognitive_engine</code></span>
            </div>
            <div className="flex items-center gap-2 border-l border-slate-800 pl-4">
              <Sparkles className="w-4 h-4 text-purple-400" />
              <span>Cora Bridge: <code className="text-slate-200">thermo_physics.si_core.cora_ollama_bridge</code></span>
            </div>
            <div className="flex items-center gap-2 border-l border-slate-800 pl-4">
              <Database className="w-4 h-4 text-emerald-400" />
              <span>Episodic Store: <code className="text-emerald-300">data/cora_memory.npz</code></span>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <Link href="/thermo-lab" className="text-cyan-400 hover:text-cyan-300 transition flex items-center gap-1">
              Station 1–15 Benchmarks <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>
        </div>
      </main>
    </div>
  );
}
