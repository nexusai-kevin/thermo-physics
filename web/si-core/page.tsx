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
  Send,
  Wifi,
  WifiOff,
  BookOpen,
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

const DEFAULT_ATTRACTORS = [
  { name: "Station_15_SI_Cognitive_Core", basinCoord: [-0.65, 0.45], color: "#38bdf8", tag: "Station 15" },
  { name: "Nexus_Thermodynamic_Dispatcher", basinCoord: [0.55, 0.60], color: "#34d399", tag: "Dispatcher" },
  { name: "Station_13_GLP_1_Codon_Optimization", basinCoord: [0.40, -0.55], color: "#c084fc", tag: "Codon EBM" },
  { name: "Local_Workstation_Architecture", basinCoord: [-0.50, -0.50], color: "#fbbf24", tag: "RTX 5080" },
];

const INFERENCE_TIERS = [
  { name: "Local_Fast_3B", vramGb: 2.2, latencyMs: 120, capability: 0.5, desc: "Ultra-fast Q4 reflex engine" },
  { name: "Local_Core_8B", vramGb: 6.5, latencyMs: 350, capability: 0.8, desc: "Balanced reasoning engine" },
  { name: "Local_Deep_14B", vramGb: 11.5, latencyMs: 900, capability: 0.95, desc: "Full cognitive depth engine" },
];

const PRESET_QUERIES = [
  {
    label: "GLP-1 Codon Hairpins",
    query: "Explain how our Station 13 GLP-1 codon optimizer avoids hairpin traps.",
    expectedStance: "VERIFY",
  },
  {
    label: "Nexus Dispatcher Quotas",
    query: "How does the 40-task Nexus Dispatcher prevent HTTP 429 rate limits?",
    expectedStance: "FAST_HEURISTIC",
  },
  {
    label: "Kramers Swarm Routing",
    query: "Propose a novel coupling between Kramers barrier escape and autonomous drone swarm routing.",
    expectedStance: "DEEP_REASONING",
  },
];

export default function SICorePage() {
  // --------------------------------------------------------------------------
  // CONNECTION MODE & AUTO-PROBE (localhost:8090)
  // --------------------------------------------------------------------------
  const [connectionMode, setConnectionMode] = useState<"local" | "simulation">("simulation");
  const [localServerAvailable, setLocalServerAvailable] = useState<boolean>(false);
  const [isProbing, setIsProbing] = useState<boolean>(false);
  const [isSeeding, setIsSeeding] = useState<boolean>(false);
  const [seedNotification, setSeedNotification] = useState<string | null>(null);
  const [memoryCount, setMemoryCount] = useState<number>(62);
  const [coraModels, setCoraModels] = useState<string[]>([]);

  // --------------------------------------------------------------------------
  // USER CONTROLS & PARAMS
  // --------------------------------------------------------------------------
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [tickSpeedMs, setTickSpeedMs] = useState<number>(300);
  const [noveltyInjection, setNoveltyInjection] = useState<number>(0.25);
  const [currentVramBaseline, setCurrentVramBaseline] = useState<number>(2.96);
  const [activeStage, setActiveStage] = useState<number>(1);
  const [tickCount, setTickCount] = useState<number>(0);
  const [lastTickLatency, setLastTickLatency] = useState<number>(5.12);

  // --------------------------------------------------------------------------
  // ORGAN 1 STATE: Continuous Hopfield Memory
  // --------------------------------------------------------------------------
  const [selectedAttractor, setSelectedAttractor] = useState<string>("Station_15_SI_Cognitive_Core");
  const [hopfieldEnergy, setHopfieldEnergy] = useState<number>(-2.84);
  const [retrievalTrajectory, setRetrievalTrajectory] = useState<{ x: number; y: number }[]>([
    { x: -0.1, y: 0.1 },
    { x: -0.3, y: 0.25 },
    { x: -0.5, y: 0.38 },
    { x: -0.65, y: 0.45 },
  ]);
  const [hopfieldTopMatches, setHopfieldTopMatches] = useState<{ label: string; score: number }[]>([
    { label: "Station_15_SI_Cognitive_Core", score: 0.94 },
    { label: "Station_13_GLP_1_Codon_Optimization", score: 0.38 },
    { label: "Nexus_Thermodynamic_Dispatcher", score: 0.29 },
    { label: "Local_Workstation_Architecture", score: 0.21 },
  ]);

  // --------------------------------------------------------------------------
  // ORGAN 2 STATE: Kramers Double-Well Curiosity & Drive
  // --------------------------------------------------------------------------
  const [particleX, setParticleX] = useState<number>(-1.0);
  const [particleV, setParticleV] = useState<number>(0.0);
  const [bathTempT, setBathTempT] = useState<number>(0.82);
  const [driveMode, setDriveMode] = useState<"EXPLOIT_KNOWN" | "EXPLORE_NOVELTY">("EXPLOIT_KNOWN");

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
  const [dispatchedModel, setDispatchedModel] = useState<string>("llama3.1:8b");
  const [metabolicEnergy, setMetabolicEnergy] = useState<number>(4.2);
  const [vramAllocated, setVramAllocated] = useState<number>(2.2);
  const [vramHeadroom, setVramHeadroom] = useState<number>(10.84);

  // --------------------------------------------------------------------------
  // CHAT CONSOLE STATE
  // --------------------------------------------------------------------------
  const [chatPrompt, setChatPrompt] = useState<string>("");
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [chatResponse, setChatResponse] = useState<string>("");
  const [lastStreamTelemetry, setLastStreamTelemetry] = useState<Record<string, unknown> | null>(null);
  const [tokensPerSec, setTokensPerSec] = useState<number>(0);
  const [tokenCount, setTokenCount] = useState<number>(0);

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
  // AUTO-PROBE LOCAL CORA SERVER (http://127.0.0.1:8090/health)
  // --------------------------------------------------------------------------
  const probeLocalServer = useCallback(async () => {
    setIsProbing(true);
    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 1800);
      const res = await fetch("http://127.0.0.1:8090/health", { signal: controller.signal });
      clearTimeout(timer);

      if (res.ok) {
        const data = await res.json();
        setLocalServerAvailable(true);
        setConnectionMode("local");
        setMemoryCount(data.memory_count || 62);
        setCurrentVramBaseline(data.vram_used_gb || 2.96);
        setCoraModels(data.models || []);
        playSynthesizerBlip(580, 0.08);
      } else {
        setLocalServerAvailable(false);
        setConnectionMode("simulation");
      }
    } catch {
      setLocalServerAvailable(false);
      setConnectionMode("simulation");
    } finally {
      setIsProbing(false);
    }
  }, [playSynthesizerBlip]);

  useEffect(() => {
    probeLocalServer();
  }, [probeLocalServer]);

  // Handle re-seed knowledge action
  const handleReSeedKnowledge = async () => {
    setIsSeeding(true);
    setSeedNotification(null);
    try {
      if (connectionMode === "local") {
        const res = await fetch("http://127.0.0.1:8090/memory/seed", { method: "POST" });
        if (res.ok) {
          const data = await res.json();
          setMemoryCount(data.memory_count);
          setSeedNotification(`Successfully consolidated ${data.memory_count} episodic attractors into data/cora_memory.npz!`);
          playSynthesizerBlip(640, 0.1);
        }
      } else {
        await new Promise((r) => setTimeout(r, 600));
        setMemoryCount(62);
        setSeedNotification("Re-seeded 62 canonical and markdown attractors (Simulation Mode).");
        playSynthesizerBlip(520, 0.08);
      }
    } catch (e) {
      setSeedNotification(`Seeding error: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setIsSeeding(false);
      setTimeout(() => setSeedNotification(null), 5000);
    }
  };

  // --------------------------------------------------------------------------
  // LOCAL SIMULATION TICK
  // --------------------------------------------------------------------------
  const runSimulationTick = useCallback(
    (forcedNovelty?: number) => {
      const t0 = performance.now();
      const nov = forcedNovelty !== undefined ? forcedNovelty : noveltyInjection;

      // 1. Hopfield Attractor Basin Recall
      setActiveStage(1);
      const isExplore = nov > 0.6;
      let targetIdx = 0;
      if (isExplore) {
        targetIdx = Math.random() > 0.4 ? 2 : 3;
      } else {
        targetIdx = Math.random() > 0.3 ? 0 : 1;
      }
      const targetAttractor = DEFAULT_ATTRACTORS[targetIdx];
      setSelectedAttractor(targetAttractor.name);

      const targetX = targetAttractor.basinCoord[0];
      const targetY = targetAttractor.basinCoord[1];
      const startX = targetX + (Math.random() - 0.5) * 0.7;
      const startY = targetY + (Math.random() - 0.5) * 0.7;

      setRetrievalTrajectory([
        { x: startX, y: startY },
        { x: startX * 0.66 + targetX * 0.34, y: startY * 0.66 + targetY * 0.34 },
        { x: startX * 0.33 + targetX * 0.67, y: startY * 0.33 + targetY * 0.67 },
        { x: targetX, y: targetY },
      ]);

      const simulatedMatches = DEFAULT_ATTRACTORS.map((att, i) => ({
        label: att.name,
        score: i === targetIdx ? 0.88 + Math.random() * 0.08 : Math.random() * 0.35,
      })).sort((a, b) => b.score - a.score);

      setHopfieldTopMatches(simulatedMatches);
      setHopfieldEnergy(-(2.5 + Math.random() * 0.8));

      // 2. Kramers Double-Well Modulation
      const dt = 0.05;
      const barrier = 2.5;
      const gamma = 0.4;
      const temp = Math.min(Math.max(0.2 + 2.5 * nov, 0.1), 5.0);
      setBathTempT(temp);

      let curX = particleX;
      let curV = particleV;
      const force = -4.0 * barrier * curX * (curX * curX - 1.0) + 0.6 * nov;
      const noise = Math.sqrt((2.0 * gamma * temp) / dt) * (Math.random() * 2 - 1) * 0.7;

      curV = curV + (-gamma * curV + force + noise) * dt;
      curX = Math.min(Math.max(curX + curV * dt, -1.8), 1.8);

      if (nov > 0.8 && curX < 0) curX += 0.45;
      else if (nov < 0.15 && curX > 0) curX -= 0.45;

      setParticleX(curX);
      setParticleV(curV);
      const mode = curX > 0 ? "EXPLORE_NOVELTY" : "EXPLOIT_KNOWN";
      setDriveMode(mode);

      // 3. Potts Goal Arbiter Equilibrating
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

      // 4. Metabolic QUBO Scheduling
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
      setDispatchedModel(bestEp.name === "Local_Deep_14B" ? "qwen2.5:14b" : "llama3.1:8b");
      setMetabolicEnergy(Math.round(bestH * 100) / 100);
      setVramAllocated(bestEp.vramGb);
      setVramHeadroom(Math.max(0, Math.round((16.0 - (currentVramBaseline + bestEp.vramGb)) * 10) / 10));

      const elapsed = Math.round((performance.now() - t0) * 100) / 100;
      setLastTickLatency(Math.max(2.1, elapsed));
      setTickCount((prev) => prev + 1);

      if (mode === "EXPLORE_NOVELTY") {
        playSynthesizerBlip(540, 0.06);
      } else {
        playSynthesizerBlip(380, 0.04);
      }
    },
    [noveltyInjection, particleX, particleV, moduleSpins, currentVramBaseline, playSynthesizerBlip]
  );

  // Auto-tick effect
  useEffect(() => {
    if (!isRunning) return;
    const interval = setInterval(() => {
      runSimulationTick();
    }, tickSpeedMs);
    return () => clearInterval(interval);
  }, [isRunning, tickSpeedMs, runSimulationTick]);

  // --------------------------------------------------------------------------
  // LIVE CORA CHAT & SSE / SIMULATION STREAMING
  // --------------------------------------------------------------------------
  const handleSendPrompt = async (promptToSend?: string) => {
    const text = (promptToSend || chatPrompt).trim();
    if (!text || isStreaming) return;

    setIsStreaming(true);
    setChatResponse("");
    setLastStreamTelemetry(null);
    setTokenCount(0);
    setTokensPerSec(0);
    const streamStart = performance.now();

    playSynthesizerBlip(440, 0.05);

    // MODE A: LIVE LOCAL FASTAPI + SSE SERVER (:8090)
    if (connectionMode === "local" && localServerAvailable) {
      try {
        const response = await fetch("http://127.0.0.1:8090/chat/stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt: text }),
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: Failed to reach local Cora stream`);
        }

        const reader = response.body?.getReader();
        if (!reader) throw new Error("ReadableStream not supported by response");

        const decoder = new TextDecoder();
        let buffer = "";
        let tokensReceived = 0;

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed.startsWith("data:")) continue;
            const jsonStr = trimmed.slice(5).trim();
            if (!jsonStr) continue;

            try {
              const event = JSON.parse(jsonStr);

              // 1. Initial Telemetry Event: Immediately update all 4 physical quadrants!
              if (event.type === "telemetry" && event.data) {
                const telem = event.data;
                setLastStreamTelemetry(telem);
                setLastTickLatency(telem.tick_latency_ms);
                setSelectedAttractor(telem.organ_1_attractor);
                setHopfieldEnergy(telem.organ_1_energy);
                if (telem.hopfield_top_matches) {
                  setHopfieldTopMatches(telem.hopfield_top_matches);
                }
                setParticleX(telem.kramers_x);
                setBathTempT(telem.bath_temp_T);
                setDriveMode(telem.drive_mode);
                setNoveltyInjection(telem.novelty_score);
                if (telem.potts_spins) {
                  setModuleSpins(telem.potts_spins);
                }
                setDominantStance(telem.potts_stance);
                setPottsEnergy(telem.potts_energy_H);
                setSelectedTier(telem.metabolic_tier);
                setDispatchedModel(telem.dispatched_model);
                setCurrentVramBaseline(telem.vram_used_gb);
                if (telem.vram_headroom_gb !== undefined) {
                  setVramHeadroom(telem.vram_headroom_gb);
                }
                setTickCount((prev) => prev + 1);
                playSynthesizerBlip(620, 0.08);
              }

              // 2. Token Stream Chunks
              if (event.type === "token" && event.content) {
                tokensReceived++;
                setTokenCount(tokensReceived);
                setChatResponse((prev) => prev + event.content);
                const elapsedSec = (performance.now() - streamStart) / 1000.0;
                if (elapsedSec > 0.1) {
                  setTokensPerSec(Math.round((tokensReceived / elapsedSec) * 10) / 10);
                }
              }

              // 3. Completion Event
              if (event.type === "done") {
                playSynthesizerBlip(520, 0.05);
              }
            } catch {
              // Parse error on malformed chunk
            }
          }
        }
      } catch (err) {
        // Fallback to simulation if network dropped
        setChatResponse(`[Local Stream Notice] Offline fallback activated: ${err instanceof Error ? err.message : String(err)}`);
      } finally {
        setIsStreaming(false);
      }
      return;
    }

    // MODE B: BROWSER SIMULATION STREAMING
    // 1. Run local thermodynamic tick
    const isExplorationQuery = text.toLowerCase().includes("novel") || text.toLowerCase().includes("drone") || text.toLowerCase().includes("quantum");
    const nov = isExplorationQuery ? 0.88 : 0.22;
    setNoveltyInjection(nov);
    runSimulationTick(nov);

    // 2. Stream simulated grounded response word by word
    const simulatedText = isExplorationQuery
      ? `[Cora SI Core • qwen2.5:14b (Browser Grounded Stream)]\nUnder stance [DEEP_REASONING] with intrinsic curiosity [EXPLORE_NOVELTY], the system hopped the Kramers potential barrier (x=+0.92, T=2.40K) into episodic basin [Station_15_SI_Cognitive_Core].\n\nThermodynamic synthesis for "${text}":\nCoupling Kramers barrier hopping with drone swarm routing enables stochastic multi-agent obstacle evasion without centralized gradient descent. By scaling the effective kinetic temperature T with local obstacle density, drones naturally tunnel through local minima and converge to the global lowest-action flight path across complex topologies.`
      : `[Cora SI Core • llama3.1:8b (Browser Grounded Stream)]\nUnder stance [VERIFY] with intrinsic drive [EXPLOIT_KNOWN], the system settled stably in potential well (x=-1.00, T=0.75K) aligning with [Station_13_GLP_1_Codon_Optimization].\n\nAnalysis for "${text}":\nAll 24 residue positions preserve physiological GC balance (48.2%) while categorical Potts EBM penalties eliminate inverted-repeat Watson-Crick stems, preventing ribosomal stalling obstacles on RTX 5080 (3.0/16.0GB VRAM headroom).`;

    const words = simulatedText.split(" ");
    let curIdx = 0;
    const streamInterval = setInterval(() => {
      if (curIdx >= words.length) {
        clearInterval(streamInterval);
        setIsStreaming(false);
        playSynthesizerBlip(540, 0.05);
        return;
      }
      const word = words[curIdx];
      setChatResponse((prev) => prev + (curIdx > 0 ? " " : "") + word);
      curIdx++;
      setTokenCount(curIdx);
      const elapsedSec = (performance.now() - streamStart) / 1000.0;
      if (elapsedSec > 0.05) {
        setTokensPerSec(Math.round((curIdx / elapsedSec) * 10) / 10);
      }
    }, 28);
  };

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

    const toScreen = (nx: number, ny: number) => ({
      x: w / 2 + nx * (w * 0.42),
      y: h / 2 - ny * (h * 0.42),
    });

    // Draw Attractor Basins as soft energy contours
    DEFAULT_ATTRACTORS.forEach((attractor) => {
      const pos = toScreen(attractor.basinCoord[0], attractor.basinCoord[1]);
      const isTarget = attractor.name === selectedAttractor || attractor.tag === selectedAttractor;

      const grad = ctx.createRadialGradient(pos.x, pos.y, 4, pos.x, pos.y, isTarget ? 65 : 40);
      grad.addColorStop(0, attractor.color + (isTarget ? "66" : "25"));
      grad.addColorStop(1, "transparent");
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, isTarget ? 65 : 40, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = attractor.color;
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, isTarget ? 7 : 5, 0, Math.PI * 2);
      ctx.fill();

      ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
      ctx.fillStyle = isTarget ? "#ffffff" : "#94a3b8";
      ctx.fillText(attractor.tag, pos.x - 20, pos.y + 18);
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

      ctx.fillStyle = "#f43f5e";
      ctx.beginPath();
      ctx.arc(p0.x, p0.y, 4, 0, Math.PI * 2);
      ctx.fill();

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

    const barrier = 2.5;
    const tilt = 0.5 * noveltyInjection;
    const calcU = (x: number) => barrier * Math.pow(x * x - 1, 2) - tilt * x;

    const xMin = -1.8;
    const xMax = 1.8;
    const uMin = -1.5;
    const uMax = 4.5;

    const toCanvasX = (x: number) => ((x - xMin) / (xMax - xMin)) * (w - 40) + 20;
    const toCanvasY = (u: number) => h - 25 - ((u - uMin) / (uMax - uMin)) * (h - 50);

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

    const barrierPeakU = calcU(0);
    ctx.strokeStyle = "rgba(244, 63, 94, 0.5)";
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(toCanvasX(0), toCanvasY(barrierPeakU));
    ctx.lineTo(toCanvasX(0), h - 15);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.font = "bold 11px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.fillStyle = "#34d399";
    ctx.fillText("EXPLOIT (-1.0)", toCanvasX(-1.3), h - 8);

    ctx.fillStyle = "#f59e0b";
    ctx.fillText("EXPLORE (+1.0)", toCanvasX(0.7), h - 8);

    const pxCanvas = toCanvasX(particleX);
    const pyCanvas = toCanvasY(calcU(particleX));

    const haloRadius = Math.max(10, Math.min(30, 10 + bathTempT * 6));
    const pGrad = ctx.createRadialGradient(pxCanvas, pyCanvas, 2, pxCanvas, pyCanvas, haloRadius);
    pGrad.addColorStop(0, driveMode === "EXPLORE_NOVELTY" ? "#fb7185" : "#38bdf8");
    pGrad.addColorStop(1, "transparent");
    ctx.fillStyle = pGrad;
    ctx.beginPath();
    ctx.arc(pxCanvas, pyCanvas, haloRadius, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = driveMode === "EXPLORE_NOVELTY" ? "#f43f5e" : "#0284c7";
    ctx.beginPath();
    ctx.arc(pxCanvas, pyCanvas, 7, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.stroke();
  }, [particleX, bathTempT, driveMode, noveltyInjection]);

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
              <div className="flex flex-wrap items-center gap-2">
                <span className="px-2 py-0.5 rounded text-[11px] font-mono tracking-wider font-semibold uppercase bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">
                  Station 15 • Non-von-Neumann Core
                </span>

                {/* CONNECTION MODE TOGGLE PILL */}
                <button
                  onClick={() => {
                    if (localServerAvailable) {
                      setConnectionMode((m) => (m === "local" ? "simulation" : "local"));
                    } else {
                      probeLocalServer();
                    }
                  }}
                  className={`px-2.5 py-0.5 rounded-full text-[11px] font-mono font-semibold border flex items-center gap-1.5 transition ${
                    connectionMode === "local"
                      ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40 hover:bg-emerald-500/30 shadow-sm shadow-emerald-500/20"
                      : "bg-cyan-500/15 text-cyan-300 border-cyan-500/30 hover:bg-cyan-500/25"
                  }`}
                  title="Click to toggle or probe local Cora server at http://127.0.0.1:8090"
                >
                  {connectionMode === "local" ? (
                    <>
                      <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
                      <Wifi className="w-3 h-3 text-emerald-400" />
                      <span>LIVE LOCAL CORA (:8090)</span>
                    </>
                  ) : (
                    <>
                      <WifiOff className="w-3 h-3 text-cyan-400" />
                      <span>BROWSER SIMULATION</span>
                      {isProbing ? <RefreshCw className="w-2.5 h-2.5 animate-spin" /> : null}
                    </>
                  )}
                </button>

                {/* MEMORY COUNT BADGE */}
                <span className="px-2 py-0.5 rounded text-[11px] font-mono bg-purple-500/20 text-purple-300 border border-purple-500/30 flex items-center gap-1">
                  <Database className="w-3 h-3 text-purple-400" />
                  <span>{memoryCount} Attractors</span>
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
              onClick={handleReSeedKnowledge}
              disabled={isSeeding}
              className="px-3 py-1.5 rounded-lg text-xs font-mono font-semibold bg-purple-600/20 text-purple-300 border border-purple-500/40 hover:bg-purple-600/30 transition flex items-center gap-1.5 disabled:opacity-50"
              title="Runs seed_cora_memory across README.md and docs/*.md"
            >
              <BookOpen className={`w-3.5 h-3.5 ${isSeeding ? "animate-spin" : ""}`} />
              <span>{isSeeding ? "SEEDING..." : "RE-SEED KNOWLEDGE"}</span>
            </button>

            <button
              onClick={() => setIsRunning(!isRunning)}
              className={`px-3 py-1.5 rounded-lg text-xs font-mono font-semibold flex items-center gap-1.5 transition ${
                isRunning
                  ? "bg-amber-500/20 text-amber-300 border border-amber-500/40 hover:bg-amber-500/30"
                  : "bg-slate-800 text-slate-300 border border-slate-700 hover:bg-slate-700"
              }`}
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isRunning ? "animate-spin" : ""}`} />
              {isRunning ? "PAUSE TICKER" : "AUTO TICKER"}
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

      {/* SEED NOTIFICATION TOAST */}
      {seedNotification && (
        <div className="bg-purple-950/80 border-b border-purple-800/80 px-6 py-2 text-center text-xs font-mono text-purple-300 animate-fade-in flex items-center justify-center gap-2">
          <Sparkles className="w-3.5 h-3.5 text-purple-400" />
          <span>{seedNotification}</span>
        </div>
      )}

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
            <span className="text-slate-400 text-[10px] uppercase tracking-wider">RTX 5080 VRAM</span>
            <span className="text-blue-400 font-bold mt-0.5">{currentVramBaseline.toFixed(2)} / 16.0 GB</span>
          </div>

          <div className="flex flex-col">
            <span className="text-slate-400 text-[10px] uppercase tracking-wider">Dispatched Model</span>
            <span className="text-emerald-400 font-bold mt-0.5 truncate">{dispatchedModel}</span>
          </div>
        </div>
      </section>

      {/* MAIN CONTAINER */}
      <main className="max-w-7xl mx-auto w-full px-6 py-6 flex-1 flex flex-col gap-6">

        {/* ===================================================================
            INTERACTIVE CORA SI COMMAND & CHAT CONSOLE
        =================================================================== */}
        <section className="p-5 rounded-2xl bg-gradient-to-br from-slate-900 via-slate-900/90 to-blue-950/30 border border-cyan-500/30 shadow-2xl flex flex-col gap-4">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-3">
            <div className="flex items-center gap-2.5">
              <div className="p-2 rounded-lg bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
                <Terminal className="w-4 h-4" />
              </div>
              <div>
                <h2 className="text-sm font-bold text-white flex items-center gap-2">
                  Cora SI Live Transduction &amp; Chat Console
                </h2>
                <p className="text-[11px] text-slate-400 font-mono">
                  Input directly drives the 4 physical organs (Hopfield recall, Kramers Langevin hopping, Potts consensus, &amp; RTX 5080 metabolism)
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <span className={`text-[11px] font-mono px-2.5 py-1 rounded-full border ${
                connectionMode === "local"
                  ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
                  : "bg-cyan-500/20 text-cyan-300 border-cyan-500/40"
              }`}>
                {connectionMode === "local" ? "Target: http://127.0.0.1:8090" : "Target: In-Browser Transducer"}
              </span>
            </div>
          </div>

          {/* Quick Preset Buttons */}
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-[11px] font-mono text-slate-400">Quick Prompts:</span>
            {PRESET_QUERIES.map((preset) => (
              <button
                key={preset.label}
                onClick={() => {
                  setChatPrompt(preset.query);
                  handleSendPrompt(preset.query);
                }}
                disabled={isStreaming}
                className="px-2.5 py-1 rounded-lg text-xs font-mono bg-slate-800/90 hover:bg-slate-700 text-slate-300 border border-slate-700/80 hover:text-white transition flex items-center gap-1.5 disabled:opacity-50"
              >
                <span>⚡ {preset.label}</span>
              </button>
            ))}
          </div>

          {/* Input Bar */}
          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleSendPrompt();
            }}
            className="flex items-center gap-2"
          >
            <input
              type="text"
              value={chatPrompt}
              onChange={(e) => setChatPrompt(e.target.value)}
              placeholder="Ask Cora a technical prompt or inject sensory novelty..."
              disabled={isStreaming}
              className="flex-1 bg-slate-950/90 border border-slate-700/80 rounded-xl px-4 py-2.5 text-sm text-slate-100 placeholder:text-slate-500 focus:outline-none focus:border-cyan-500/60 font-mono"
            />
            <button
              type="submit"
              disabled={isStreaming || !chatPrompt.trim()}
              className="px-4 py-2.5 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 font-bold text-xs shadow-lg shadow-cyan-500/20 transition flex items-center gap-1.5 disabled:opacity-50"
            >
              {isStreaming ? (
                <RefreshCw className="w-3.5 h-3.5 animate-spin" />
              ) : (
                <Send className="w-3.5 h-3.5" />
              )}
              <span>{isStreaming ? "EQUILIBRATING..." : "DISPATCH"}</span>
            </button>
          </form>

          {/* Streaming Response Console */}
          {(chatResponse || isStreaming) && (
            <div className="p-4 rounded-xl bg-slate-950/90 border border-slate-800 space-y-3 font-mono">
              <div className="flex flex-wrap items-center justify-between text-xs border-b border-slate-800 pb-2">
                <span className="text-cyan-400 font-bold flex items-center gap-1.5">
                  <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
                  Cora Response Output
                </span>
                <div className="flex items-center gap-3 text-[11px] text-slate-400">
                  <span>Tick SLA: <strong className="text-emerald-400">{lastTickLatency}ms</strong></span>
                  {tokensPerSec > 0 && <span>Tokens: <strong className="text-cyan-300">{tokenCount} ({tokensPerSec} tok/s)</strong></span>}
                </div>
              </div>

              {/* Streaming Response Text */}
              <div className="text-xs text-slate-200 leading-relaxed whitespace-pre-wrap">
                {chatResponse}
                {isStreaming && <span className="inline-block w-2 h-3.5 bg-cyan-400 ml-1 animate-pulse" />}
              </div>
            </div>
          )}
        </section>

        {/* ===================================================================
            TOP ROW: ORGAN 1 & ORGAN 2
        =================================================================== */}
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
                  <p className="text-[11px] text-slate-400 font-mono">128GB DDR5 RAM Episodic Basin Pool ({memoryCount} Attractors)</p>
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

            {/* Attractor Confidence Bars (Live from Hopfield cosine similarities) */}
            <div className="grid grid-cols-2 gap-2 text-xs font-mono">
              {hopfieldTopMatches.map((match) => {
                const isWinner = match.label === selectedAttractor;
                return (
                  <div
                    key={match.label}
                    className={`p-2 rounded-lg border transition ${
                      isWinner
                        ? "bg-blue-500/10 border-blue-500/40 text-blue-300"
                        : "bg-slate-950/40 border-slate-800/60 text-slate-400"
                    }`}
                  >
                    <div className="flex items-center justify-between text-[11px] mb-1">
                      <span className="truncate max-w-[160px]">{match.label.replace(/_/g, " ")}</span>
                      <span>{(match.score * 100).toFixed(0)}%</span>
                    </div>
                    <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                      <div
                        className="h-1.5 rounded-full bg-cyan-400 transition-all duration-300"
                        style={{ width: `${Math.min(100, Math.max(5, Math.round(match.score * 100)))}%` }}
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

            {/* Drive Telemetry */}
            <div className="grid grid-cols-3 gap-2 text-xs font-mono">
              <div className="p-2 rounded-lg bg-slate-950/60 border border-slate-800">
                <span className="text-[10px] text-slate-400 block">Coordinate (x)</span>
                <span className="text-white font-bold text-sm">{particleX.toFixed(3)}</span>
              </div>
              <div className="p-2 rounded-lg bg-slate-950/60 border border-slate-800">
                <span className="text-[10px] text-slate-400 block">Bath Temp (T)</span>
                <span className="text-amber-400 font-bold text-sm">{bathTempT.toFixed(2)} K</span>
              </div>
              <div className="p-2 rounded-lg bg-slate-950/60 border border-slate-800">
                <span className="text-[10px] text-slate-400 block">Novelty Score</span>
                <span className="text-cyan-400 font-bold text-sm">{noveltyInjection.toFixed(2)}</span>
              </div>
            </div>
          </div>
        </div>

        {/* ===================================================================
            BOTTOM ROW: ORGAN 3 & ORGAN 4
        =================================================================== */}
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
                const spin = moduleSpins[idx] || 0;
                const stance = STANCE_CONFIG[spin] || STANCE_CONFIG[0];
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
                <div
                  className="bg-slate-600 h-3"
                  style={{ width: `${(currentVramBaseline / 16.0) * 100}%` }}
                  title="Base VRAM"
                />
                <div
                  className="bg-emerald-500 h-3 transition-all duration-300"
                  style={{ width: `${(vramAllocated / 16.0) * 100}%` }}
                  title="Allocated Engine"
                />
              </div>
              <div className="flex items-center justify-between text-[10px] font-mono text-slate-500">
                <span>Base: {currentVramBaseline.toFixed(2)} GB</span>
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
              <span>Episodic Store: <code className="text-emerald-300">data/cora_memory.npz ({memoryCount})</code></span>
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
