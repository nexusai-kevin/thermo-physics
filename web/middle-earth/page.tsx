"use client";

import React, { useState, useMemo, useCallback } from "react";
import Link from "next/link";
import {
  Compass,
  Shield,
  Eye,
  Flame,
  Mountain,
  Waves,
  Skull,
  Sparkles,
  Sliders,
  Volume2,
  VolumeX,
  ArrowLeft,
  ExternalLink,
  RefreshCw,
  Dices,
  CheckCircle2,
  ChevronRight,
  Info,
  Gauge,
  Zap,
  BookOpen,
} from "lucide-react";
import { Container } from "@/components/Container";
import {
  Waypoint,
  WAYPOINTS,
  WAYPOINT_MAP,
  EDGES,
  PATH_PRESETS,
  calculatePathAction,
  annealPath,
  evaluatePotentialEnergy,
  PathActionBreakdown,
  AnnealTelemetry,
} from "@/lib/middleEarthEngine";

// ============================================================================
// AUDIO SYNTHESIZER (Web Audio API)
// ============================================================================
let audioCtx: AudioContext | null = null;

function playMiddleEarthTone(
  freq: number,
  durationMs: number = 90,
  type: OscillatorType = "sine",
  enabled: boolean = true
) {
  if (!enabled || typeof window === "undefined") return;
  try {
    const AudioContextClass =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    if (!audioCtx) audioCtx = new AudioContextClass();
    if (audioCtx.state === "suspended") audioCtx.resume();

    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = type;
    osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
    gain.gain.setValueAtTime(0.05, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + durationMs / 1000);
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + durationMs / 1000);
  } catch {
    // Audio context not allowed or blocked
  }
}

export default function MiddleEarthPage() {
  // State
  const [currentPath, setCurrentPath] = useState<string[]>(() => PATH_PRESETS[0].path);
  const [temperature, setTemperature] = useState<number>(1.0); // Ring Temptation / Noise
  const [sauronVigilance, setSauronVigilance] = useState<number>(0.5); // 0.0 - 1.0
  const [selectedWaypointId, setSelectedWaypointId] = useState<string>("rivendell");
  const [sfxEnabled, setSfxEnabled] = useState<boolean>(true);
  const [isAnnealing, setIsAnnealing] = useState<boolean>(false);
  const [lastTelemetry, setLastTelemetry] = useState<AnnealTelemetry | null>(null);

  // Active Waypoint data
  const selectedWaypoint = useMemo(
    () => WAYPOINT_MAP[selectedWaypointId] || WAYPOINTS[0],
    [selectedWaypointId]
  );

  // Local continuous potential at selected waypoint
  const localPotential = useMemo(
    () => evaluatePotentialEnergy(selectedWaypoint.x, selectedWaypoint.y, sauronVigilance),
    [selectedWaypoint, sauronVigilance]
  );

  // Path Action Metrics
  const metrics: PathActionBreakdown = useMemo(
    () => calculatePathAction(currentPath, temperature, sauronVigilance),
    [currentPath, temperature, sauronVigilance]
  );

  // Handle Waypoint Click
  const handleSelectWaypoint = (id: string) => {
    setSelectedWaypointId(id);
    const wp = WAYPOINT_MAP[id];
    const pitch = wp ? 250 + (1000 - wp.y) * 0.4 : 440;
    playMiddleEarthTone(pitch, 60, "triangle", sfxEnabled);
  };

  // Run Langevin Path Annealer
  const handleAnnealPath = useCallback(() => {
    setIsAnnealing(true);
    playMiddleEarthTone(320, 100, "sawtooth", sfxEnabled);

    setTimeout(() => {
      const { annealedPath, telemetry } = annealPath(currentPath, temperature, sauronVigilance, 45);
      setCurrentPath(annealedPath);
      setLastTelemetry(telemetry);
      setIsAnnealing(false);
      playMiddleEarthTone(587.33, 140, "sine", sfxEnabled); // D5 elven chime
    }, 25);
  }, [currentPath, temperature, sauronVigilance, sfxEnabled]);

  // Load Preset
  const handleLoadPreset = (preset: typeof PATH_PRESETS[0]) => {
    playMiddleEarthTone(440, 80, "sine", sfxEnabled);
    setCurrentPath(preset.path);
    setLastTelemetry(null);
  };

  // Thermal Fluctuation (High noise leap)
  const handleThermalFluctuation = () => {
    playMiddleEarthTone(220, 110, "square", sfxEnabled);
    const { annealedPath, telemetry } = annealPath(currentPath, 4.0, sauronVigilance, 35);
    setCurrentPath(annealedPath);
    setLastTelemetry(telemetry);
  };

  // Temperature description
  const temperatureDesc = useMemo(() => {
    if (temperature <= 0.6)
      return { label: "Cold Stealth / Minimum Action Ground State", icon: Shield, color: "text-sky-300" };
    if (temperature <= 1.8)
      return { label: "Balanced Stochastic Fellowship Trajectory", icon: Compass, color: "text-cyan-400" };
    if (temperature <= 3.2)
      return { label: "Warm Temptation / Desperate Mountain Detours", icon: Flame, color: "text-amber-400" };
    return { label: "Wild Panic / High Kramers Barrier Escape", icon: Skull, color: "text-rose-400" };
  }, [temperature]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col selection:bg-cyan-500 selection:text-slate-950 font-sans">
      {/* Background glow effects */}
      <div
        aria-hidden="true"
        className="pointer-events-none fixed top-0 left-1/2 -translate-x-1/2 w-[1200px] h-[600px] bg-gradient-to-b from-amber-600/10 via-cyan-600/10 to-transparent blur-3xl rounded-full"
      />

      {/* TOP NAVIGATION & CROSS-LINKS BANNER */}
      <header className="sticky top-0 z-40 bg-slate-950/85 backdrop-blur-md border-b border-slate-800">
        <Container className="py-3 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <Link
              href="/thermo-lab"
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-300 hover:text-white text-xs font-medium transition"
            >
              <ArrowLeft size={14} />
              <span>Thermo-Lab</span>
            </Link>

            <div className="hidden sm:flex items-center gap-1 text-xs text-slate-400 font-mono">
              <span className="text-slate-600">/</span>
              <a
                href="/thermo-lab#station-8"
                className="px-2 py-1 rounded hover:bg-slate-900 hover:text-amber-400 transition"
                title="Jump to Station 8: Continuous Double-Well Kramers Hopping"
              >
                Station 8 (Kramers Double-Well) ↗
              </a>
              <span className="text-slate-600">/</span>
              <a
                href="/thermo-lab#station-9"
                className="px-2 py-1 rounded hover:bg-slate-900 hover:text-cyan-400 transition"
                title="Jump to Station 9: NP-Complete Combinatorial Routing"
              >
                Station 9 (Combinatorial Routing) ↗
              </a>
              <span className="text-slate-600">/</span>
              <Link
                href="/lol-draft"
                className="px-2 py-1 rounded hover:bg-slate-900 hover:text-purple-400 transition"
                title="LoL Thermodynamic Draft Solver"
              >
                LoL Draft (Potts K=4) ↗
              </Link>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={() => setSfxEnabled(!sfxEnabled)}
              className="p-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-400 hover:text-slate-200 text-xs flex items-center gap-1.5 transition"
              title={sfxEnabled ? "Mute Middle-earth SFX" : "Enable Middle-earth SFX"}
            >
              {sfxEnabled ? <Volume2 size={14} className="text-amber-400" /> : <VolumeX size={14} />}
              <span className="hidden md:inline">{sfxEnabled ? "SFX On" : "Muted"}</span>
            </button>

            <a
              href="https://github.com/nexusai-kevin/thermo-physics"
              target="_blank"
              rel="noopener noreferrer"
              className="hidden lg:inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-300 text-xs font-semibold hover:text-amber-300 transition"
            >
              <span>Source Repo</span>
              <ExternalLink size={12} />
            </a>
          </div>
        </Container>
      </header>

      {/* HERO SECTION */}
      <section className="pt-8 pb-3 relative">
        <Container className="text-center max-w-4xl mx-auto">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-amber-500/10 border border-amber-500/30 text-amber-300 text-xs font-semibold uppercase tracking-wider mb-3">
            <Compass size={13} className="text-amber-400" />
            <span>Kramers Barrier Escape &amp; Combinatorial Routing in Middle-earth</span>
          </div>
          <h1 className="text-3xl sm:text-5xl font-extrabold tracking-tight text-white leading-tight">
            Middle-earth{" "}
            <span className="bg-gradient-to-r from-amber-400 via-yellow-200 to-cyan-400 bg-clip-text text-transparent">
              Thermodynamic Routing Engine
            </span>
          </h1>
          <p className="mt-2.5 text-slate-400 text-sm sm:text-base max-w-2xl mx-auto">
            Model the Quest of the Ring as a <strong className="text-slate-200">Continuous Potential Energy Surface U(x, y)</strong>.
            Sample minimum-action paths across 18 canonical waypoints, tunneling under Kramers barriers
            and balancing the vigilance field of the Great Eye.
          </p>

          {/* MASTER DUAL SLIDERS: Sauron Vigilance & Ring Temptation */}
          <div className="mt-6 p-4 max-w-3xl mx-auto rounded-2xl bg-slate-900/80 border border-slate-800 backdrop-blur-md shadow-xl grid grid-cols-1 md:grid-cols-2 gap-4 text-left">
            {/* Slider 1: Sauron Vigilance */}
            <div className="space-y-2 p-2 rounded-xl bg-slate-950/60 border border-slate-800/80">
              <div className="flex items-center justify-between text-xs">
                <div className="flex items-center gap-1.5 font-semibold text-rose-400">
                  <Eye size={14} className="text-rose-500 animate-pulse" />
                  <span>Sauron's Eye Vigilance h(t)</span>
                </div>
                <span className="font-mono text-rose-300 font-bold px-1.5 py-0.5 rounded bg-rose-950/60 border border-rose-900/40">
                  {Math.round(sauronVigilance * 100)}%
                </span>
              </div>
              <input
                type="range"
                min="0.0"
                max="1.0"
                step="0.05"
                value={sauronVigilance}
                onChange={(e) => {
                  setSauronVigilance(parseFloat(e.target.value));
                  playMiddleEarthTone(180 + parseFloat(e.target.value) * 120, 20, "sawtooth", sfxEnabled);
                }}
                className="w-full accent-rose-500 cursor-pointer h-1.5 bg-slate-800 rounded-lg appearance-none"
              />
              <div className="flex justify-between text-[10px] font-mono text-slate-500">
                <span>Distracted (West)</span>
                <span>Unblinking Gaze (Gorgoroth)</span>
              </div>
            </div>

            {/* Slider 2: Ring Temptation / Temperature T */}
            <div className="space-y-2 p-2 rounded-xl bg-slate-950/60 border border-slate-800/80">
              <div className="flex items-center justify-between text-xs">
                <div className="flex items-center gap-1.5 font-semibold text-cyan-400">
                  <Sliders size={14} className="text-cyan-400" />
                  <span>Ring Temptation / Temp (T)</span>
                </div>
                <span className="font-mono text-cyan-300 font-bold px-1.5 py-0.5 rounded bg-cyan-950/60 border border-cyan-900/40">
                  T = {temperature.toFixed(1)}
                </span>
              </div>
              <input
                type="range"
                min="0.2"
                max="5.0"
                step="0.1"
                value={temperature}
                onChange={(e) => {
                  setTemperature(parseFloat(e.target.value));
                  playMiddleEarthTone(300 + parseFloat(e.target.value) * 70, 20, "sine", sfxEnabled);
                }}
                className="w-full accent-cyan-400 cursor-pointer h-1.5 bg-slate-800 rounded-lg appearance-none"
              />
              <div className="flex items-center justify-between text-[10px] font-mono">
                <span className="text-sky-400">Stealth Ground</span>
                <span className="text-rose-400">Panic Hopping</span>
              </div>
            </div>
          </div>
        </Container>
      </section>

      {/* QUICK PRESETS */}
      <section className="py-2">
        <Container className="flex flex-wrap items-center justify-center gap-2">
          <span className="text-xs text-slate-400 font-semibold mr-1">Canonical Quests:</span>
          {PATH_PRESETS.map((preset) => (
            <button
              key={preset.id}
              onClick={() => handleLoadPreset(preset)}
              className="px-3 py-1.5 rounded-xl bg-slate-900/90 hover:bg-slate-800 border border-slate-800 hover:border-amber-500/40 text-xs font-medium text-slate-300 hover:text-amber-300 transition shadow-sm"
              title={preset.description}
            >
              {preset.title}
            </button>
          ))}
        </Container>
      </section>

      {/* MAIN INTERACTIVE MAP & INSPECTOR DASHBOARD */}
      <main className="flex-1 py-4">
        <Container>
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
            {/* LEFT / CENTER: INTERACTIVE MIDDLE-EARTH MAP CANVAS (8 COLS) */}
            <div className="lg:col-span-8 space-y-3">
              <div className="p-3.5 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-2xl relative overflow-hidden">
                {/* Map Header Status */}
                <div className="flex flex-wrap items-center justify-between gap-3 mb-2 px-1">
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse" />
                    <span className="text-xs font-bold text-white tracking-wide uppercase">
                      Topological Landscape of the Third Age
                    </span>
                  </div>
                  <div className="flex items-center gap-2 text-xs font-mono">
                    <span className="text-slate-400">
                      Path Length: <strong className="text-white">{currentPath.length}</strong> nodes
                    </span>
                    <span className="text-slate-600">•</span>
                    <span className="text-slate-400">
                      Action S: <strong className="text-cyan-300">{metrics.totalAction}</strong>
                    </span>
                  </div>
                </div>

                {/* SVG MAP CANVAS */}
                <div className="relative w-full aspect-[10/8] bg-[#07090e] rounded-xl border border-slate-800/80 overflow-hidden shadow-inner">
                  {/* Parchment Grid Overlay */}
                  <div
                    aria-hidden="true"
                    className="absolute inset-0 opacity-[0.04] pointer-events-none"
                    style={{
                      backgroundImage: "radial-gradient(#f59e0b 1px, transparent 1px)",
                      backgroundSize: "24px 24px",
                    }}
                  />

                  {/* Mordor Threat Aura (Glow centered around Mount Doom) */}
                  <div
                    aria-hidden="true"
                    className="absolute pointer-events-none rounded-full blur-3xl transition-opacity duration-500"
                    style={{
                      left: "70%",
                      top: "50%",
                      width: "360px",
                      height: "360px",
                      transform: "translate(-50%, -50%)",
                      background: "radial-gradient(circle, rgba(225,29,72,0.22) 0%, rgba(159,18,57,0.10) 50%, transparent 80%)",
                      opacity: 0.3 + sauronVigilance * 0.7,
                    }}
                  />

                  <svg
                    viewBox="0 0 1000 800"
                    className="w-full h-full select-none"
                    style={{ filter: "drop-shadow(0 0 1px rgba(0,0,0,0.8))" }}
                  >
                    <defs>
                      {/* Ley Line Glowing Filter */}
                      <filter id="ley-glow" x="-20%" y="-20%" width="140%" height="140%">
                        <feGaussianBlur stdDeviation="3.5" result="blur" />
                        <feMerge>
                          <feMergeNode in="blur" />
                          <feMergeNode in="SourceGraphic" />
                        </feMerge>
                      </filter>

                      {/* Sauron Gaze Cone Gradient */}
                      <radialGradient id="sauron-eye-grad" cx="50%" cy="50%" r="50%">
                        <stop offset="0%" stopColor="#ef4444" stopOpacity="1" />
                        <stop offset="60%" stopColor="#b91c1c" stopOpacity="0.8" />
                        <stop offset="100%" stopColor="#450a0a" stopOpacity="0" />
                      </radialGradient>
                    </defs>

                    {/* =======================================================
                        1. MOUNTAIN RIDGES & REGION CONTOURS
                    ======================================================= */}
                    {/* Misty Mountains Ridge */}
                    <path
                      d="M 290 80 Q 305 160 300 220 T 330 310 T 360 410"
                      fill="none"
                      stroke="#334155"
                      strokeWidth="20"
                      strokeLinecap="round"
                      opacity="0.25"
                    />
                    <path
                      d="M 290 80 Q 305 160 300 220 T 330 310 T 360 410"
                      fill="none"
                      stroke="#475569"
                      strokeWidth="4"
                      strokeDasharray="6 8"
                      opacity="0.4"
                    />

                    {/* White Mountains Ridge (Ered Nimrais) */}
                    <path
                      d="M 370 550 Q 470 590 580 660"
                      fill="none"
                      stroke="#334155"
                      strokeWidth="16"
                      strokeLinecap="round"
                      opacity="0.25"
                    />

                    {/* Ephel Dúath (Mountains of Shadow - Mordor West) */}
                    <path
                      d="M 680 460 Q 695 540 730 620 L 780 670"
                      fill="none"
                      stroke="#4c0519"
                      strokeWidth="18"
                      strokeLinecap="round"
                      opacity="0.4"
                    />

                    {/* Ered Lithui (Ash Mountains - Mordor North) */}
                    <path
                      d="M 680 460 L 960 470"
                      fill="none"
                      stroke="#4c0519"
                      strokeWidth="16"
                      strokeLinecap="round"
                      opacity="0.4"
                    />

                    {/* The Great River Anduin */}
                    <path
                      d="M 460 320 Q 440 380 480 440 T 520 520 Q 560 620 620 720"
                      fill="none"
                      stroke="#0284c7"
                      strokeWidth="6"
                      strokeLinecap="round"
                      opacity="0.3"
                    />

                    {/* Region Labels */}
                    <text x="140" y="80" fill="#475569" fontSize="12" fontWeight="bold" letterSpacing="3">
                      ERIADOR
                    </text>
                    <text x="240" y="200" fill="#64748b" fontSize="10" fontStyle="italic">
                      Misty Mountains
                    </text>
                    <text x="440" y="320" fill="#eab308" fontSize="11" fontWeight="bold" opacity="0.6">
                      Lothlórien
                    </text>
                    <text x="590" y="420" fill="#64748b" fontSize="10" fontStyle="italic">
                      The Dead Marshes
                    </text>
                    <text x="460" y="600" fill="#475569" fontSize="12" fontWeight="bold" letterSpacing="2">
                      ROHAN
                    </text>
                    <text x="580" y="700" fill="#475569" fontSize="12" fontWeight="bold" letterSpacing="2">
                      GONDOR
                    </text>
                    <text x="820" y="440" fill="#e11d48" fontSize="14" fontWeight="bold" letterSpacing="4" opacity="0.7">
                      MORDOR
                    </text>

                    {/* =======================================================
                        2. INACTIVE GRAPH LEY EDGES
                    ======================================================= */}
                    {EDGES.map((edge, idx) => {
                      const from = WAYPOINT_MAP[edge.from];
                      const to = WAYPOINT_MAP[edge.to];
                      if (!from || !to) return null;

                      return (
                        <line
                          key={idx}
                          x1={from.x}
                          y1={from.y}
                          x2={to.x}
                          y2={to.y}
                          stroke="#334155"
                          strokeWidth="1.5"
                          strokeDasharray="4 4"
                          opacity="0.3"
                        />
                      );
                    })}

                    {/* =======================================================
                        3. ACTIVE THERMODYNAMIC LEY LINE TRAJECTORY
                    ======================================================= */}
                    {currentPath.length >= 2 && (
                      <polyline
                        points={currentPath
                          .map((id) => {
                            const wp = WAYPOINT_MAP[id];
                            return wp ? `${wp.x},${wp.y}` : "";
                          })
                          .filter(Boolean)
                          .join(" ")}
                        fill="none"
                        stroke="#06b6d4"
                        strokeWidth="3.5"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        filter="url(#ley-glow)"
                        className="animate-pulse"
                      />
                    )}

                    {/* =======================================================
                        4. DYNAMIC SAURON EYE VIGILANCE CONE
                    ======================================================= */}
                    <g transform="translate(880, 520)">
                      <circle
                        r={40 + sauronVigilance * 50}
                        fill="url(#sauron-eye-grad)"
                        opacity={0.3 + sauronVigilance * 0.7}
                      />
                      {/* Pupil slit */}
                      <ellipse
                        rx={3 + sauronVigilance * 3}
                        ry={14 + sauronVigilance * 8}
                        fill="#fef08a"
                        opacity={0.9}
                      />
                      <circle r="4" fill="#ef4444" />
                      <text
                        x="-30"
                        y="-25"
                        fill="#f87171"
                        fontSize="10"
                        fontFamily="monospace"
                        fontWeight="bold"
                        letterSpacing="1"
                      >
                        BARAD-DÛR
                      </text>
                    </g>

                    {/* =======================================================
                        5. WAYPOINT NODES
                    ======================================================= */}
                    {WAYPOINTS.map((wp) => {
                      const isSelected = selectedWaypointId === wp.id;
                      const isInPath = currentPath.includes(wp.id);
                      const isStart = wp.id === "rivendell";
                      const isEnd = wp.id === "mount_doom";

                      let nodeColor = "#64748b";
                      if (wp.hazardType === "sanctuary") nodeColor = "#10b981"; // Emerald
                      else if (wp.hazardType === "mountain_pass" || wp.hazardType === "dungeon")
                        nodeColor = "#f59e0b"; // Amber
                      else if (wp.region === "mordor") nodeColor = "#f43f5e"; // Rose

                      if (isStart) nodeColor = "#38bdf8";
                      if (isEnd) nodeColor = "#fb923c";

                      return (
                        <g
                          key={wp.id}
                          transform={`translate(${wp.x}, ${wp.y})`}
                          onClick={() => handleSelectWaypoint(wp.id)}
                          className="cursor-pointer group"
                        >
                          {/* Selection Halo */}
                          {isSelected && (
                            <circle
                              r="16"
                              fill="none"
                              stroke="#38bdf8"
                              strokeWidth="2"
                              strokeDasharray="4 2"
                              className="animate-spin origin-center"
                            />
                          )}

                          {/* Outer pulse for path members */}
                          {isInPath && (
                            <circle
                              r="10"
                              fill={nodeColor}
                              opacity="0.25"
                              className="animate-ping"
                            />
                          )}

                          {/* Main Node Dot */}
                          <circle
                            r={isSelected ? "8" : isInPath ? "6.5" : "5"}
                            fill={nodeColor}
                            stroke="#0f172a"
                            strokeWidth="2"
                            className="group-hover:scale-125 transition-transform"
                          />

                          {/* Node Label */}
                          <text
                            x="11"
                            y="4"
                            fill={isSelected ? "#38bdf8" : isInPath ? "#f1f5f9" : "#94a3b8"}
                            fontSize={isSelected ? "12" : isInPath ? "11" : "9.5"}
                            fontWeight={isSelected || isInPath ? "bold" : "normal"}
                            className="pointer-events-none drop-shadow"
                          >
                            {wp.name}
                          </text>
                        </g>
                      );
                    })}
                  </svg>
                </div>

                {/* Bottom Map Legend */}
                <div className="flex flex-wrap items-center justify-between gap-3 mt-3 pt-2 border-t border-slate-800 text-xs text-slate-400">
                  <div className="flex items-center gap-4">
                    <span className="flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 rounded-full bg-emerald-400" />
                      <span>Sanctuary Well (U &lt; 0)</span>
                    </span>
                    <span className="flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 rounded-full bg-amber-400" />
                      <span>Kramers Barrier (Caradhras/Moria)</span>
                    </span>
                    <span className="flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 rounded-full bg-rose-500" />
                      <span>Mordor Fortification / Eye</span>
                    </span>
                  </div>
                  <span className="text-[11px] font-mono text-cyan-300">
                    Click any waypoint to inspect field coordinates
                  </span>
                </div>
              </div>

              {/* ACTION CONSOLE */}
              <div className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800 flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-center gap-2">
                  <button
                    onClick={handleAnnealPath}
                    disabled={isAnnealing}
                    className="px-4 py-2.5 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 font-bold text-xs shadow-lg shadow-cyan-500/25 flex items-center gap-2 transition active:scale-95 disabled:opacity-50"
                  >
                    <RefreshCw size={15} className={isAnnealing ? "animate-spin" : ""} />
                    <span>Equilibrate Path (Langevin / Gibbs)</span>
                  </button>

                  <button
                    onClick={handleThermalFluctuation}
                    className="px-3.5 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 text-xs font-semibold flex items-center gap-1.5 transition"
                    title="Simulate high-heat chaotic excursion (T=4.0)"
                  >
                    <Dices size={15} />
                    <span>Thermal Fluctuation</span>
                  </button>
                </div>

                <div className={`text-xs font-medium flex items-center gap-1.5 ${temperatureDesc.color}`}>
                  <temperatureDesc.icon size={14} />
                  <span>{temperatureDesc.label}</span>
                </div>
              </div>
            </div>

            {/* RIGHT COLUMN: ESOTERIC SYSTEMS INSPECTOR & TELEMETRY (4 COLS) */}
            <div className="lg:col-span-4 space-y-4">
              {/* Waypoint Inspector Card */}
              <div className="p-5 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-xl space-y-3">
                <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                  <div className="flex items-center gap-2">
                    <Compass size={17} className="text-amber-400" />
                    <h3 className="text-base font-bold text-white tracking-wide">
                      {selectedWaypoint.name}
                    </h3>
                  </div>
                  <span className="text-xs font-mono text-amber-300 italic">
                    {selectedWaypoint.sindarinName}
                  </span>
                </div>

                <p className="text-xs text-slate-300 leading-relaxed">
                  {selectedWaypoint.lore}
                </p>

                {/* Quantitative Waypoint Fields */}
                <div className="grid grid-cols-2 gap-2 pt-2 text-xs font-mono">
                  <div className="p-2.5 rounded-xl bg-slate-950/70 border border-slate-800">
                    <span className="text-[10px] text-slate-500 block">LOCAL POTENTIAL U(x,y)</span>
                    <strong className={localPotential.uVal < 0 ? "text-emerald-400" : "text-rose-400"}>
                      {localPotential.uVal > 0 ? `+${localPotential.uVal}` : localPotential.uVal}
                    </strong>
                  </div>

                  <div className="p-2.5 rounded-xl bg-slate-950/70 border border-slate-800">
                    <span className="text-[10px] text-slate-500 block">KRAMERS BARRIER ΔU</span>
                    <strong className="text-amber-400">+{selectedWaypoint.barrierHeight}</strong>
                  </div>

                  <div className="p-2.5 rounded-xl bg-slate-950/70 border border-slate-800">
                    <span className="text-[10px] text-slate-500 block">DAMPING COEFFICIENT γ</span>
                    <strong className="text-slate-300">{selectedWaypoint.friction}x</strong>
                  </div>

                  <div className="p-2.5 rounded-xl bg-slate-950/70 border border-slate-800">
                    <span className="text-[10px] text-slate-500 block">EYE VIGILANCE EXPOSURE</span>
                    <strong className="text-rose-400">
                      {Math.round(selectedWaypoint.sauronExposure * 100)}%
                    </strong>
                  </div>
                </div>
              </div>

              {/* Real-time Path Hamiltonian Telemetry Card */}
              <div className="p-5 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-xl space-y-3">
                <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                  <div className="flex items-center gap-2">
                    <Gauge size={17} className="text-cyan-400" />
                    <h3 className="text-sm font-bold text-white tracking-wide uppercase">
                      Path Hamiltonian Telemetry
                    </h3>
                  </div>
                  <span className="text-xs font-mono text-cyan-300 font-bold">
                    S = {metrics.totalAction}
                  </span>
                </div>

                <div className="space-y-2.5 text-xs">
                  {/* Action Breakdown Bars */}
                  <div>
                    <div className="flex justify-between text-slate-400 mb-1">
                      <span>Distance Cost:</span>
                      <span className="font-mono text-slate-200">{metrics.distanceCost}</span>
                    </div>
                    <div className="flex justify-between text-slate-400 mb-1">
                      <span>Barrier Potential Energy (ΔU):</span>
                      <span className="font-mono text-amber-400">+{metrics.barrierCost}</span>
                    </div>
                    <div className="flex justify-between text-slate-400 mb-1">
                      <span>Viscous Terrain Dissipation (γ):</span>
                      <span className="font-mono text-slate-300">+{metrics.frictionDissipation}</span>
                    </div>
                    <div className="flex justify-between text-slate-400 mb-1">
                      <span>Sauron Sightline Exposure:</span>
                      <span className="font-mono text-rose-400">+{metrics.stealthRiskScore}</span>
                    </div>
                  </div>

                  {/* Kramers Escape Probability */}
                  <div className="pt-2 border-t border-slate-800/80">
                    <div className="flex justify-between text-slate-300 font-semibold mb-1">
                      <span>Kramers Barrier Escape Prob:</span>
                      <span className="font-mono text-emerald-400">
                        {(metrics.kramersEscapeProb * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="h-1.5 w-full bg-slate-950 rounded-full overflow-hidden">
                      <div
                        className="bg-emerald-400 h-full transition-all"
                        style={{ width: `${Math.min(100, metrics.kramersEscapeProb * 100)}%` }}
                      />
                    </div>
                  </div>

                  {/* Sauron Detection Probability */}
                  <div>
                    <div className="flex justify-between text-slate-300 font-semibold mb-1">
                      <span>Sauron Detection Probability:</span>
                      <span className="font-mono text-rose-400 font-bold">
                        {metrics.sauronDetectionProb}%
                      </span>
                    </div>
                    <div className="h-1.5 w-full bg-slate-950 rounded-full overflow-hidden">
                      <div
                        className="bg-rose-500 h-full transition-all"
                        style={{ width: `${metrics.sauronDetectionProb}%` }}
                      />
                    </div>
                  </div>

                  {/* Sokal Autocorrelation Time */}
                  <div className="flex justify-between text-slate-400 pt-1 font-mono text-[11px]">
                    <span>Sokal Correlation Time τ_int:</span>
                    <span className="text-cyan-300 font-bold">{metrics.sokalTauInt}</span>
                  </div>

                  {/* Active Hazards */}
                  {metrics.activeHazards.length > 0 && (
                    <div className="pt-2 border-t border-slate-800/80">
                      <span className="text-[10px] uppercase font-bold text-slate-500 tracking-wider block mb-1">
                        Active Terrain Hazards
                      </span>
                      <div className="flex flex-wrap gap-1">
                        {metrics.activeHazards.map((h, i) => (
                          <span
                            key={i}
                            className="px-2 py-0.5 rounded bg-rose-950/60 border border-rose-900/40 text-rose-300 text-[10px]"
                          >
                            ⚠️ {h}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Last Anneal Telemetry */}
                  {lastTelemetry && (
                    <div className="p-2.5 rounded-xl bg-cyan-950/30 border border-cyan-800/40 text-[11px] font-mono text-cyan-300">
                      ⚡ MCMC Anneal: {lastTelemetry.runtimeMs}ms • {lastTelemetry.stepsSampled} steps • {lastTelemetry.acceptanceRate}% acceptance (ΔS = {lastTelemetry.deltaAction})
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </Container>
      </main>

      {/* PHYSICS & ESOTERIC SYSTEMS DEEP DIVE */}
      <section className="py-12 border-t border-slate-800 bg-slate-950/60">
        <Container className="max-w-4xl mx-auto space-y-6">
          <div className="text-center space-y-2">
            <h3 className="text-2xl font-bold text-white">
              Thermodynamic Physics of Middle-earth Routing
            </h3>
            <p className="text-slate-400 text-sm max-w-2xl mx-auto">
              How Tolkien's geography directly maps to Kramers barrier escape rates (Station 8)
              and NP-hard path Hamiltonian optimization (Station 9).
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-4">
            {/* Station 8 Connection */}
            <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3">
              <div className="flex items-center justify-between">
                <span className="px-2.5 py-1 rounded-full bg-amber-500/10 border border-amber-500/30 text-amber-300 text-xs font-bold font-mono">
                  Station 8 Connection
                </span>
                <span className="text-xs text-slate-400 font-mono">Kramers Barrier Hopping</span>
              </div>
              <h4 className="text-base font-bold text-white">Caradhras vs. Moria as Double-Well Escape</h4>
              <p className="text-xs text-slate-300 leading-relaxed">
                In Station 8, continuous oscillators hop between two potential wells separated by a quartic hill.
                In Middle-earth, the Fellowship faces an identical Kramers dilemma: scale the high saddle point of
                Caradhras (overcoming storm thermal noise ΔU = 75), or tunnel through the subterranean double-well of
                Khazad-dûm, risking the localized Balrog activation barrier (ΔU = 65).
              </p>
              <div className="pt-2">
                <a
                  href="/thermo-lab#station-8"
                  className="inline-flex items-center gap-1.5 text-xs text-amber-400 hover:text-amber-300 font-semibold"
                >
                  <span>Explore Station 8 in Thermo-Lab</span>
                  <ChevronRight size={13} />
                </a>
              </div>
            </div>

            {/* Station 9 Connection */}
            <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3">
              <div className="flex items-center justify-between">
                <span className="px-2.5 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-300 text-xs font-bold font-mono">
                  Station 9 Connection
                </span>
                <span className="text-xs text-slate-400 font-mono">Combinatorial Routing</span>
              </div>
              <h4 className="text-base font-bold text-white">Path Action Minimization &amp; Viscous Damping</h4>
              <p className="text-xs text-slate-300 leading-relaxed">
                Station 9 uses 28 chromatic Gibbs update blocks to untangle the 10-city Traveling Salesperson QUBO.
                Here, path action S integrates Euclidean distance, terrain friction (the heavy damping γ = 5.0 of the Dead Marshes),
                and Sauron's gaze potential. At low temperature, Langevin annealing finds the global minimum action trajectory.
              </p>
              <div className="pt-2">
                <a
                  href="/thermo-lab#station-9"
                  className="inline-flex items-center gap-1.5 text-xs text-cyan-400 hover:text-cyan-300 font-semibold"
                >
                  <span>Explore Station 9 in Thermo-Lab</span>
                  <ChevronRight size={13} />
                </a>
              </div>
            </div>
          </div>
        </Container>
      </section>
    </div>
  );
}
