"use client";

import React, { useState, useEffect, useMemo, useCallback } from "react";
import Link from "next/link";
import {
  Sparkles,
  Zap,
  Shield,
  Swords,
  Flame,
  Snowflake,
  RefreshCw,
  Sliders,
  Volume2,
  VolumeX,
  ArrowLeft,
  ExternalLink,
  Lock,
  Unlock,
  Search,
  X,
  ChevronRight,
  CheckCircle2,
  Info,
  Gauge,
  Dices,
} from "lucide-react";
import { Container } from "@/components/Container";
import {
  Champion,
  CHAMPIONS,
  CHAMPION_MAP,
  ROLES,
  Role,
  SLOT_DEFINITIONS,
  DraftState,
  calculateDraftHamiltonian,
  runPottsDraftSampler,
  DRAFT_PRESETS,
  McmcTelemetry,
} from "@/lib/lolDraftEngine";

// ============================================================================
// AUDIO SYNTHESIZER (Web Audio API)
// ============================================================================
let audioCtx: AudioContext | null = null;

function playHextechTone(freq: number, durationMs: number = 80, type: OscillatorType = "sine", enabled: boolean = true) {
  if (!enabled || typeof window === "undefined") return;
  try {
    const AudioContextClass = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    if (!audioCtx) audioCtx = new AudioContextClass();
    if (audioCtx.state === "suspended") {
      audioCtx.resume();
    }
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = type;
    osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
    gain.gain.setValueAtTime(0.06, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + durationMs / 1000);
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + durationMs / 1000);
  } catch {
    // Audio context not allowed or blocked
  }
}

export default function LolDraftPage() {
  // Draft State (10 slots: 0-4 Blue, 5-9 Red)
  const [draftState, setDraftState] = useState<DraftState>(() => DRAFT_PRESETS[0].state);
  const [pinnedSlots, setPinnedSlots] = useState<boolean[]>(() => new Array(10).fill(false));
  const [beta, setBeta] = useState<number>(2.0); // Temperature
  const [sfxEnabled, setSfxEnabled] = useState<boolean>(true);
  const [isSolving, setIsSolving] = useState<boolean>(false);
  const [activeSlotModal, setActiveSlotModal] = useState<number | null>(null);

  // Search & Filters in Selector Modal
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [selectedRoleFilter, setSelectedRoleFilter] = useState<Role | "ALL">("ALL");

  // Telemetry from last Gibbs solve
  const [lastTelemetry, setLastTelemetry] = useState<McmcTelemetry | null>(null);

  // Computed Hamiltonian metrics
  const metrics = useMemo(() => calculateDraftHamiltonian(draftState), [draftState]);

  // Set default role filter when opening modal to match the target slot
  useEffect(() => {
    if (activeSlotModal !== null) {
      const targetRole = ROLES[activeSlotModal % 5];
      setSelectedRoleFilter(targetRole);
      setSearchQuery("");
    }
  }, [activeSlotModal]);

  // Handle slot click (manual select)
  const handleOpenSlot = (index: number) => {
    playHextechTone(480, 50, "triangle", sfxEnabled);
    setActiveSlotModal(index);
  };

  // Toggle slot pin/freeze state
  const handleTogglePin = (index: number, e: React.MouseEvent) => {
    e.stopPropagation();
    playHextechTone(pinnedSlots[index] ? 350 : 650, 60, "sine", sfxEnabled);
    setPinnedSlots((prev) => {
      const next = [...prev];
      next[index] = !next[index];
      return next;
    });
  };

  // Select champion for slot
  const handleSelectChampion = (championId: string | null) => {
    if (activeSlotModal === null) return;
    playHextechTone(championId ? 600 : 300, 70, "sine", sfxEnabled);
    setDraftState((prev) => {
      const next = [...prev];
      next[activeSlotModal] = championId;
      return next;
    });
    setActiveSlotModal(null);
  };

  // Run Gibbs Sampler (Auto-Equilibrate)
  const handleAutoEquilibrate = useCallback(() => {
    setIsSolving(true);
    playHextechTone(520, 90, "sawtooth", sfxEnabled);

    // Short timeout to allow UI animation state to render
    setTimeout(() => {
      const { nextState, telemetry } = runPottsDraftSampler(draftState, pinnedSlots, beta, 50);
      setDraftState(nextState);
      setLastTelemetry(telemetry);
      setIsSolving(false);
      playHextechTone(880, 140, "sine", sfxEnabled);
    }, 20);
  }, [draftState, pinnedSlots, beta, sfxEnabled]);

  // Load Preset
  const handleLoadPreset = (preset: typeof DRAFT_PRESETS[0]) => {
    playHextechTone(580, 90, "triangle", sfxEnabled);
    setDraftState(preset.state);
    setPinnedSlots(new Array(10).fill(false));
    setLastTelemetry(null);
  };

  // Randomize all unpinned slots
  const handleRandomizeUnpinned = () => {
    playHextechTone(420, 80, "square", sfxEnabled);
    const { nextState, telemetry } = runPottsDraftSampler(draftState, pinnedSlots, 0.4, 30);
    setDraftState(nextState);
    setLastTelemetry(telemetry);
  };

  // Clear all unpinned
  const handleClearUnpinned = () => {
    playHextechTone(260, 80, "sine", sfxEnabled);
    setDraftState((prev) => prev.map((c, idx) => (pinnedSlots[idx] ? c : null)));
    setLastTelemetry(null);
  };

  // Filtered champions for modal
  const alreadyPickedSet = useMemo(() => {
    const set = new Set<string>();
    draftState.forEach((id, idx) => {
      if (id && idx !== activeSlotModal) set.add(id);
    });
    return set;
  }, [draftState, activeSlotModal]);

  const filteredChampions = useMemo(() => {
    return CHAMPIONS.filter((c) => {
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        const matchesName = c.name.toLowerCase().includes(q) || c.title.toLowerCase().includes(q);
        if (!matchesName) return false;
      }
      if (selectedRoleFilter !== "ALL") {
        const matchesRole = c.primaryRole === selectedRoleFilter || c.secondaryRoles.includes(selectedRoleFilter);
        if (!matchesRole) return false;
      }
      return true;
    });
  }, [searchQuery, selectedRoleFilter]);

  // Temperature description
  const temperatureDesc = useMemo(() => {
    if (beta <= 0.5) return { label: "High Entropy / Chaotic Off-Meta", icon: Flame, color: "text-rose-400" };
    if (beta <= 1.5) return { label: "Warm Stochastic Exploration", icon: Flame, color: "text-amber-400" };
    if (beta <= 3.0) return { label: "Balanced Meta Equilibrium", icon: Zap, color: "text-cyan-400" };
    return { label: "Zero-Entropy Korean Challenger Ground State", icon: Snowflake, color: "text-sky-300" };
  }, [beta]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col selection:bg-cyan-500 selection:text-slate-950 font-sans">
      {/* Background glow effects */}
      <div
        aria-hidden="true"
        className="pointer-events-none fixed top-0 left-1/2 -translate-x-1/2 w-[1200px] h-[600px] bg-gradient-to-b from-cyan-600/10 via-purple-600/10 to-transparent blur-3xl rounded-full"
      />

      {/* TOP NAVIGATION & CROSS-LINKS BANNER */}
      <header className="sticky top-0 z-40 bg-slate-950/80 backdrop-blur-md border-b border-slate-800">
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
                href="/thermo-lab#station-4"
                className="px-2 py-1 rounded hover:bg-slate-900 hover:text-cyan-400 transition"
                title="Jump to Station 4: Multistate Potts Model"
              >
                Station 4 (Potts K=4) ↗
              </a>
              <span className="text-slate-600">/</span>
              <a
                href="/thermo-lab#station-14"
                className="px-2 py-1 rounded hover:bg-slate-900 hover:text-cyan-400 transition"
                title="Jump to Station 14: Side-Chain Rotamer Packing"
              >
                Station 14 (Rotamer Packing) ↗
              </a>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={() => setSfxEnabled(!sfxEnabled)}
              className="p-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-400 hover:text-slate-200 text-xs flex items-center gap-1.5 transition"
              title={sfxEnabled ? "Mute Arcade SFX" : "Enable Arcade SFX"}
            >
              {sfxEnabled ? <Volume2 size={14} className="text-cyan-400" /> : <VolumeX size={14} />}
              <span className="hidden md:inline">{sfxEnabled ? "SFX On" : "Muted"}</span>
            </button>

            <a
              href="https://github.com/nexusai-kevin/thermo-physics/tree/main/thermo_physics/dispatch"
              target="_blank"
              rel="noopener noreferrer"
              className="hidden lg:inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-300 text-xs font-semibold hover:text-cyan-300 transition"
            >
              <span>Dispatcher Engine</span>
              <ExternalLink size={12} />
            </a>
          </div>
        </Container>
      </header>

      {/* HERO SECTION */}
      <section className="pt-8 pb-4 relative">
        <Container className="text-center max-w-4xl mx-auto">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-300 text-xs font-semibold uppercase tracking-wider mb-3">
            <Sparkles size={13} className="text-cyan-400" />
            <span>Thermodynamic Physics Applied to Esports</span>
          </div>
          <h1 className="text-3xl sm:text-5xl font-extrabold tracking-tight text-white leading-tight">
            5v5 League of Legends{" "}
            <span className="bg-gradient-to-r from-cyan-400 via-sky-300 to-indigo-400 bg-clip-text text-transparent">
              Thermodynamic Draft Solver
            </span>
          </h1>
          <p className="mt-3 text-slate-400 text-sm sm:text-base max-w-2xl mx-auto">
            Modeled as an <strong className="text-slate-200">N=10 Categorical Potts Spin Graph</strong>.
            Equilibrate multi-champion wombo synergies, role constraints, and counter-pick defense
            at adjustable Boltzmann temperatures.
          </p>

          {/* MASTER TEMPERATURE DIAL (Beta Control) */}
          <div className="mt-6 p-4 max-w-2xl mx-auto rounded-2xl bg-slate-900/80 border border-slate-800 backdrop-blur-md shadow-xl flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-2.5">
              <Sliders size={18} className="text-cyan-400" />
              <div className="text-left">
                <div className="text-xs font-semibold text-slate-200 flex items-center gap-2">
                  <span>Boltzmann Temperature Dial</span>
                  <span className="font-mono text-cyan-300 px-2 py-0.5 rounded bg-cyan-950/60 border border-cyan-800/40">
                    β = {beta.toFixed(1)}
                  </span>
                </div>
                <div className={`text-xs flex items-center gap-1 mt-0.5 ${temperatureDesc.color}`}>
                  <temperatureDesc.icon size={12} />
                  <span>{temperatureDesc.label}</span>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-3 w-full sm:w-auto">
              <span className="text-[10px] font-mono text-slate-500">HOT (0.1)</span>
              <input
                type="range"
                min="0.1"
                max="5.0"
                step="0.1"
                value={beta}
                onChange={(e) => {
                  setBeta(parseFloat(e.target.value));
                  playHextechTone(200 + parseFloat(e.target.value) * 80, 20, "sine", sfxEnabled);
                }}
                className="w-36 sm:w-44 accent-cyan-400 cursor-pointer h-1.5 bg-slate-800 rounded-lg appearance-none"
              />
              <span className="text-[10px] font-mono text-slate-500">COLD (5.0)</span>
            </div>
          </div>
        </Container>
      </section>

      {/* QUICK PRESETS BAR */}
      <section className="py-2">
        <Container className="flex flex-wrap items-center justify-center gap-2">
          <span className="text-xs text-slate-400 font-semibold mr-1">Draft Scenarios:</span>
          {DRAFT_PRESETS.map((preset) => (
            <button
              key={preset.id}
              onClick={() => handleLoadPreset(preset)}
              className="px-3 py-1.5 rounded-xl bg-slate-900/90 hover:bg-slate-800 border border-slate-800 hover:border-cyan-500/40 text-xs font-medium text-slate-300 hover:text-cyan-300 transition shadow-sm"
              title={preset.description}
            >
              {preset.name}
            </button>
          ))}
        </Container>
      </section>

      {/* WIN ADVANTAGE & COMPARATIVE WIN PROBABILITY GAUGE */}
      <section className="py-3">
        <Container className="max-w-4xl mx-auto">
          <div className="p-3.5 rounded-2xl bg-slate-900/60 border border-slate-800 flex flex-col gap-2">
            <div className="flex items-center justify-between text-xs font-semibold px-1">
              <div className="flex items-center gap-2 text-cyan-400">
                <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse" />
                <span>BLUE ALLIES: {metrics.blueWinProb}% Win Expectancy</span>
                <span className="text-slate-400 font-mono text-[11px]">(H = {metrics.blue.totalEnergy})</span>
              </div>
              <div className="text-slate-400 font-mono text-[11px]">
                Net Advantage:{" "}
                <span className={metrics.netDeltaH >= 0 ? "text-cyan-300 font-bold" : "text-rose-400 font-bold"}>
                  {metrics.netDeltaH > 0 ? `+${metrics.netDeltaH}` : metrics.netDeltaH} ΔH
                </span>
              </div>
              <div className="flex items-center gap-2 text-rose-400">
                <span className="text-slate-400 font-mono text-[11px]">(H = {metrics.red.totalEnergy})</span>
                <span>RED ENEMIES: {100 - metrics.blueWinProb}%</span>
                <span className="w-2.5 h-2.5 rounded-full bg-rose-400" />
              </div>
            </div>

            {/* Split Bar */}
            <div className="w-full h-3 bg-slate-950 rounded-full overflow-hidden flex border border-slate-800 shadow-inner">
              <div
                className="bg-gradient-to-r from-blue-600 via-cyan-500 to-sky-400 transition-all duration-300 h-full"
                style={{ width: `${metrics.blueWinProb}%` }}
              />
              <div
                className="bg-gradient-to-r from-rose-600 via-red-500 to-amber-600 transition-all duration-300 h-full"
                style={{ width: `${100 - metrics.blueWinProb}%` }}
              />
            </div>
          </div>
        </Container>
      </section>

      {/* CORE 5v5 DRAFT BOARD */}
      <main className="flex-1 py-4">
        <Container>
          <div className="grid grid-cols-1 lg:grid-cols-11 gap-4 items-start">
            {/* BLUE TEAM (ALLIES) */}
            <div className="lg:col-span-5 space-y-3">
              <div className="flex items-center justify-between pb-1 border-b border-cyan-500/20">
                <div className="flex items-center gap-2">
                  <Shield size={16} className="text-cyan-400" />
                  <h2 className="text-base font-bold text-white tracking-wide">BLUE TEAM (Allies)</h2>
                </div>
                <div className="text-xs font-mono text-cyan-400 bg-cyan-950/60 px-2 py-0.5 rounded border border-cyan-800/40">
                  H(Blue) = {metrics.blue.totalEnergy}
                </div>
              </div>

              {/* 5 Slots */}
              <div className="space-y-2.5">
                {[0, 1, 2, 3, 4].map((slotIdx) => {
                  const role = ROLES[slotIdx];
                  const champId = draftState[slotIdx];
                  const champ = champId ? CHAMPION_MAP[champId] : null;
                  const isPinned = pinnedSlots[slotIdx];

                  return (
                    <div
                      key={slotIdx}
                      onClick={() => handleOpenSlot(slotIdx)}
                      className={`group relative p-3 rounded-xl border transition-all cursor-pointer flex items-center justify-between gap-3 ${
                        champ
                          ? "bg-slate-900/90 border-cyan-500/30 hover:border-cyan-400 shadow-md hover:shadow-cyan-500/10"
                          : "bg-slate-950/60 border-dashed border-slate-800 hover:border-slate-700"
                      }`}
                    >
                      <div className="flex items-center gap-3 min-w-0">
                        {/* Role Badge */}
                        <div className="w-9 h-9 rounded-lg bg-cyan-950/70 border border-cyan-500/30 flex flex-col items-center justify-center text-cyan-300 font-bold text-[10px] shrink-0 font-mono">
                          <span>{role}</span>
                        </div>

                        {/* Champion Info */}
                        {champ ? (
                          <div className="min-w-0">
                            <div className="flex items-center gap-2">
                              <span className="text-sm font-bold text-white truncate group-hover:text-cyan-300 transition">
                                {champ.name}
                              </span>
                              <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-slate-800 text-slate-300">
                                {champ.archetype}
                              </span>
                            </div>
                            <div className="text-xs text-slate-400 truncate flex items-center gap-2 mt-0.5">
                              <span className="text-[10px] font-mono text-cyan-400">
                                {champ.damage.ad}% AD / {champ.damage.ap}% AP
                              </span>
                              <span className="text-[10px] text-slate-500">•</span>
                              <span className="text-[10px] text-slate-400 capitalize">{champ.ccProfile.replace("_", " ")}</span>
                            </div>
                          </div>
                        ) : (
                          <div className="text-xs text-slate-500 italic">Click to assign {role}...</div>
                        )}
                      </div>

                      {/* Controls (Lock / Clear) */}
                      <div className="flex items-center gap-1.5 shrink-0" onClick={(e) => e.stopPropagation()}>
                        <button
                          onClick={(e) => handleTogglePin(slotIdx, e)}
                          className={`p-1.5 rounded-lg border text-xs transition ${
                            isPinned
                              ? "bg-amber-500/20 border-amber-500/50 text-amber-300"
                              : "bg-slate-900 border-slate-800 text-slate-500 hover:text-slate-300"
                          }`}
                          title={isPinned ? "Pinned: Solver will not overwrite" : "Unpinned: Free to optimize"}
                        >
                          {isPinned ? <Lock size={13} /> : <Unlock size={13} />}
                        </button>
                        {champ && (
                          <button
                            onClick={() => handleSelectChampion(null)}
                            className="p-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-500 hover:text-rose-400 text-xs transition"
                            title="Clear slot"
                          >
                            <X size={13} />
                          </button>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Blue Telemetry Card */}
              <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800 text-xs space-y-2">
                <div className="flex items-center justify-between text-slate-300 font-semibold">
                  <span>Damage Composition</span>
                  <span className="font-mono text-slate-400">
                    {metrics.blue.damageSplit.adPct}% AD • {metrics.blue.damageSplit.apPct}% AP • {metrics.blue.damageSplit.truePct}% True
                  </span>
                </div>
                {/* Visual Damage Gauge */}
                <div className="h-2 w-full bg-slate-950 rounded-full overflow-hidden flex">
                  <div className="bg-amber-500 h-full transition-all" style={{ width: `${metrics.blue.damageSplit.adPct}%` }} title="AD" />
                  <div className="bg-sky-500 h-full transition-all" style={{ width: `${metrics.blue.damageSplit.apPct}%` }} title="AP" />
                  <div className="bg-emerald-400 h-full transition-all" style={{ width: `${metrics.blue.damageSplit.truePct}%` }} title="True Damage" />
                </div>

                <div className="flex items-center justify-between text-slate-400 pt-1">
                  <span>Wombo CC Utility:</span>
                  <span className="font-mono text-cyan-300 font-bold">{metrics.blue.ccScore}%</span>
                </div>

                {metrics.blue.activeSynergies.length > 0 && (
                  <div className="pt-1.5 border-t border-slate-800/80">
                    <div className="text-[11px] font-semibold text-emerald-400 mb-1">Active Wombo Synergies:</div>
                    <div className="flex flex-wrap gap-1">
                      {metrics.blue.activeSynergies.map((s, i) => (
                        <span key={i} className="px-2 py-0.5 rounded bg-emerald-950/60 border border-emerald-800/40 text-emerald-300 text-[10px]">
                          ⚡ {s}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* CENTER SOLVER ACTION CONSOLE */}
            <div className="lg:col-span-1 flex flex-col items-center justify-center gap-3 py-4 lg:py-16">
              <div className="p-2 rounded-full bg-slate-900 border border-slate-800 text-slate-400 font-mono text-xs font-bold">
                VS
              </div>

              <button
                onClick={handleAutoEquilibrate}
                disabled={isSolving}
                className="relative group px-3 py-4 rounded-xl bg-gradient-to-b from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 font-bold text-xs shadow-lg shadow-cyan-500/25 flex flex-col items-center gap-1.5 transition transform active:scale-95 disabled:opacity-50"
                title="Run Potts Categorical Gibbs Sampler to solve unpinned slots"
              >
                <RefreshCw size={18} className={`${isSolving ? "animate-spin text-white" : "group-hover:rotate-180 transition-transform duration-500"}`} />
                <span className="text-[11px] leading-tight text-center">
                  Auto<br />Equilibrate
                </span>
              </button>

              <button
                onClick={handleRandomizeUnpinned}
                className="p-2.5 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-400 hover:text-cyan-300 text-xs transition"
                title="Randomize unpinned slots (High Heat)"
              >
                <Dices size={16} />
              </button>

              <button
                onClick={handleClearUnpinned}
                className="p-2 rounded-lg bg-slate-900/60 hover:bg-slate-800 border border-slate-800 text-slate-500 hover:text-rose-400 text-xs transition"
                title="Clear unpinned slots"
              >
                <X size={14} />
              </button>
            </div>

            {/* RED TEAM (ENEMIES) */}
            <div className="lg:col-span-5 space-y-3">
              <div className="flex items-center justify-between pb-1 border-b border-rose-500/20">
                <div className="flex items-center gap-2">
                  <Swords size={16} className="text-rose-400" />
                  <h2 className="text-base font-bold text-white tracking-wide">RED TEAM (Enemies)</h2>
                </div>
                <div className="text-xs font-mono text-rose-400 bg-rose-950/60 px-2 py-0.5 rounded border border-rose-800/40">
                  H(Red) = {metrics.red.totalEnergy}
                </div>
              </div>

              {/* 5 Slots */}
              <div className="space-y-2.5">
                {[5, 6, 7, 8, 9].map((slotIdx) => {
                  const role = ROLES[slotIdx - 5];
                  const champId = draftState[slotIdx];
                  const champ = champId ? CHAMPION_MAP[champId] : null;
                  const isPinned = pinnedSlots[slotIdx];

                  return (
                    <div
                      key={slotIdx}
                      onClick={() => handleOpenSlot(slotIdx)}
                      className={`group relative p-3 rounded-xl border transition-all cursor-pointer flex items-center justify-between gap-3 ${
                        champ
                          ? "bg-slate-900/90 border-rose-500/30 hover:border-rose-400 shadow-md hover:shadow-rose-500/10"
                          : "bg-slate-950/60 border-dashed border-slate-800 hover:border-slate-700"
                      }`}
                    >
                      <div className="flex items-center gap-3 min-w-0">
                        {/* Role Badge */}
                        <div className="w-9 h-9 rounded-lg bg-rose-950/70 border border-rose-500/30 flex flex-col items-center justify-center text-rose-300 font-bold text-[10px] shrink-0 font-mono">
                          <span>{role}</span>
                        </div>

                        {/* Champion Info */}
                        {champ ? (
                          <div className="min-w-0">
                            <div className="flex items-center gap-2">
                              <span className="text-sm font-bold text-white truncate group-hover:text-rose-300 transition">
                                {champ.name}
                              </span>
                              <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-slate-800 text-slate-300">
                                {champ.archetype}
                              </span>
                            </div>
                            <div className="text-xs text-slate-400 truncate flex items-center gap-2 mt-0.5">
                              <span className="text-[10px] font-mono text-rose-400">
                                {champ.damage.ad}% AD / {champ.damage.ap}% AP
                              </span>
                              <span className="text-[10px] text-slate-500">•</span>
                              <span className="text-[10px] text-slate-400 capitalize">{champ.ccProfile.replace("_", " ")}</span>
                            </div>
                          </div>
                        ) : (
                          <div className="text-xs text-slate-500 italic">Click to assign {role}...</div>
                        )}
                      </div>

                      {/* Controls (Lock / Clear) */}
                      <div className="flex items-center gap-1.5 shrink-0" onClick={(e) => e.stopPropagation()}>
                        <button
                          onClick={(e) => handleTogglePin(slotIdx, e)}
                          className={`p-1.5 rounded-lg border text-xs transition ${
                            isPinned
                              ? "bg-amber-500/20 border-amber-500/50 text-amber-300"
                              : "bg-slate-900 border-slate-800 text-slate-500 hover:text-slate-300"
                          }`}
                          title={isPinned ? "Pinned: Solver will not overwrite" : "Unpinned: Free to optimize"}
                        >
                          {isPinned ? <Lock size={13} /> : <Unlock size={13} />}
                        </button>
                        {champ && (
                          <button
                            onClick={() => handleSelectChampion(null)}
                            className="p-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-500 hover:text-rose-400 text-xs transition"
                            title="Clear slot"
                          >
                            <X size={13} />
                          </button>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Red Telemetry Card */}
              <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800 text-xs space-y-2">
                <div className="flex items-center justify-between text-slate-300 font-semibold">
                  <span>Damage Composition</span>
                  <span className="font-mono text-slate-400">
                    {metrics.red.damageSplit.adPct}% AD • {metrics.red.damageSplit.apPct}% AP • {metrics.red.damageSplit.truePct}% True
                  </span>
                </div>
                {/* Visual Damage Gauge */}
                <div className="h-2 w-full bg-slate-950 rounded-full overflow-hidden flex">
                  <div className="bg-amber-500 h-full transition-all" style={{ width: `${metrics.red.damageSplit.adPct}%` }} title="AD" />
                  <div className="bg-sky-500 h-full transition-all" style={{ width: `${metrics.red.damageSplit.apPct}%` }} title="AP" />
                  <div className="bg-emerald-400 h-full transition-all" style={{ width: `${metrics.red.damageSplit.truePct}%` }} title="True Damage" />
                </div>

                <div className="flex items-center justify-between text-slate-400 pt-1">
                  <span>Wombo CC Utility:</span>
                  <span className="font-mono text-rose-300 font-bold">{metrics.red.ccScore}%</span>
                </div>

                {metrics.red.activeSynergies.length > 0 && (
                  <div className="pt-1.5 border-t border-slate-800/80">
                    <div className="text-[11px] font-semibold text-rose-400 mb-1">Active Wombo Synergies:</div>
                    <div className="flex flex-wrap gap-1">
                      {metrics.red.activeSynergies.map((s, i) => (
                        <span key={i} className="px-2 py-0.5 rounded bg-rose-950/60 border border-rose-800/40 text-rose-300 text-[10px]">
                          ⚡ {s}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </Container>
      </main>

      {/* STATISTICAL PHYSICS TELEMETRY CARD */}
      <section className="py-6 border-t border-slate-800/80 bg-slate-950/50">
        <Container>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800">
              <div className="flex items-center gap-2 text-cyan-400 text-xs font-semibold mb-1">
                <Gauge size={14} />
                <span>MCMC Convergence</span>
              </div>
              <div className="text-2xl font-bold font-mono text-white">
                {lastTelemetry ? `${lastTelemetry.runtimeMs}ms` : "< 15ms"}
              </div>
              <p className="text-[11px] text-slate-400 mt-1">
                50 vectorized Potts sweeps across all unpinned slots in modern browser V8.
              </p>
            </div>

            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800">
              <div className="flex items-center gap-2 text-indigo-400 text-xs font-semibold mb-1">
                <RefreshCw size={14} />
                <span>MCMC Acceptance Rate</span>
              </div>
              <div className="text-2xl font-bold font-mono text-white">
                {lastTelemetry ? `${lastTelemetry.acceptanceRate}%` : "86%"}
              </div>
              <p className="text-[11px] text-slate-400 mt-1">
                Categorical Boltzmann transitions accepted at current temperature β.
              </p>
            </div>

            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800">
              <div className="flex items-center gap-2 text-emerald-400 text-xs font-semibold mb-1">
                <Sparkles size={14} />
                <span>Hamiltonian Optimization</span>
              </div>
              <div className="text-2xl font-bold font-mono text-emerald-300">
                {lastTelemetry ? `${lastTelemetry.deltaEnergy > 0 ? "+" : ""}${lastTelemetry.deltaEnergy}` : "-91.0 ΔH"}
              </div>
              <p className="text-[11px] text-slate-400 mt-1">
                Energetic relaxation achieved by Gibbs ground-state annealing.
              </p>
            </div>

            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800">
              <div className="flex items-center gap-2 text-amber-400 text-xs font-semibold mb-1">
                <Sliders size={14} />
                <span>Active Temperature</span>
              </div>
              <div className="text-2xl font-bold font-mono text-white">
                β = {beta.toFixed(1)}
              </div>
              <p className="text-[11px] text-slate-400 mt-1">
                {beta >= 3.0 ? "Greedy ground-state meta lock" : "Thermodynamic exploratory sampling"}
              </p>
            </div>
          </div>
        </Container>
      </section>

      {/* PHYSICS & COMBINATORIAL THEORY EXPLANATION */}
      <section className="py-12 border-t border-slate-800">
        <Container className="max-w-4xl mx-auto space-y-6">
          <div className="text-center space-y-2">
            <h3 className="text-2xl font-bold text-white">
              The Physics Behind the Pick-and-Ban Phase
            </h3>
            <p className="text-slate-400 text-sm max-w-2xl mx-auto">
              How competitive League of Legends drafting maps directly to Potts spin glasses and macromolecular protein design.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-4">
            {/* Potts Model Card */}
            <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3">
              <div className="flex items-center justify-between">
                <span className="px-2.5 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-300 text-xs font-bold font-mono">
                  Station 4 Connection
                </span>
                <span className="text-xs text-slate-400 font-mono">Multistate Potts</span>
              </div>
              <h4 className="text-base font-bold text-white">Categorical Spin Lattice</h4>
              <p className="text-xs text-slate-300 leading-relaxed">
                In Station 4, spins choose among <strong className="text-cyan-300">K=4 categorical color states</strong>,
                forming synchronized islands when the coupling exceeds critical temperature βc. In draft phase,
                each team slot is a discrete node choosing among ~40 champion states. Synergistic pairs (like Yasuo + Malphite)
                act as ferromagnetic couplings (J_ij &lt; 0), snapping into low-energy ground states.
              </p>
              <div className="pt-2">
                <a
                  href="/thermo-lab#station-4"
                  className="inline-flex items-center gap-1.5 text-xs text-cyan-400 hover:text-cyan-300 font-semibold"
                >
                  <span>Explore Station 4 in Thermo-Lab</span>
                  <ChevronRight size={13} />
                </a>
              </div>
            </div>

            {/* Rotamer Packing Card */}
            <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3">
              <div className="flex items-center justify-between">
                <span className="px-2.5 py-1 rounded-full bg-purple-500/10 border border-purple-500/30 text-purple-300 text-xs font-bold font-mono">
                  Station 14 Connection
                </span>
                <span className="text-xs text-slate-400 font-mono">Rotamer Packing</span>
              </div>
              <h4 className="text-base font-bold text-white">Combinatorial Steric Clashes</h4>
              <p className="text-xs text-slate-300 leading-relaxed">
                In Station 14 protein folding, amino acid side-chains choose among discrete rotamer angle states.
                Steric collisions incur enormous Lennard-Jones repulsion (λ_clash = +500). Similarly, in drafting,
                assigning two midlaners or repeating a champion incurs a heavy role-clash penalty (H_clash = +100),
                forcing the MCMC sampler to search valid structural topologies.
              </p>
              <div className="pt-2">
                <a
                  href="/thermo-lab#station-14"
                  className="inline-flex items-center gap-1.5 text-xs text-purple-400 hover:text-purple-300 font-semibold"
                >
                  <span>Explore Station 14 in Thermo-Lab</span>
                  <ChevronRight size={13} />
                </a>
              </div>
            </div>
          </div>
        </Container>
      </section>

      {/* CHAMPION SELECTION MODAL */}
      {activeSlotModal !== null && (
        <div className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-md flex items-center justify-center p-4">
          <div className="w-full max-w-2xl bg-slate-900 border border-slate-800 rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[85vh] animate-in fade-in duration-200">
            {/* Header */}
            <div className="p-4 border-b border-slate-800 flex items-center justify-between">
              <div>
                <h3 className="text-base font-bold text-white flex items-center gap-2">
                  <span>Assign Champion to</span>
                  <span className="text-cyan-400 font-mono">
                    {SLOT_DEFINITIONS[activeSlotModal].label}
                  </span>
                </h3>
                <p className="text-xs text-slate-400">
                  Target Role: <strong className="text-cyan-300">{SLOT_DEFINITIONS[activeSlotModal].role}</strong>
                </p>
              </div>
              <button
                onClick={() => setActiveSlotModal(null)}
                className="p-1.5 rounded-lg bg-slate-800 text-slate-400 hover:text-white transition"
              >
                <X size={18} />
              </button>
            </div>

            {/* Filter & Search Bar */}
            <div className="p-3 bg-slate-950/60 border-b border-slate-800 space-y-2">
              <div className="relative">
                <Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
                <input
                  type="text"
                  placeholder="Search champion by name or title..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-9 pr-4 py-2 rounded-xl bg-slate-900 border border-slate-800 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500"
                  autoFocus
                />
              </div>

              {/* Role filter buttons */}
              <div className="flex flex-wrap items-center gap-1.5 text-xs">
                <span className="text-[11px] text-slate-400 mr-1">Role:</span>
                {(["ALL", "TOP", "JGL", "MID", "ADC", "SUP"] as const).map((r) => (
                  <button
                    key={r}
                    onClick={() => setSelectedRoleFilter(r)}
                    className={`px-2.5 py-1 rounded-lg text-xs font-mono font-medium transition ${
                      selectedRoleFilter === r
                        ? "bg-cyan-500 text-slate-950 font-bold"
                        : "bg-slate-900 border border-slate-800 text-slate-300 hover:text-white"
                    }`}
                  >
                    {r}
                  </button>
                ))}
              </div>
            </div>

            {/* Champions Grid */}
            <div className="flex-1 overflow-y-auto p-4 grid grid-cols-1 sm:grid-cols-2 gap-2.5">
              {/* Option to clear slot */}
              <button
                onClick={() => handleSelectChampion(null)}
                className="p-3 rounded-xl border border-dashed border-slate-800 hover:border-rose-500/50 bg-slate-950/40 text-left hover:bg-rose-950/20 transition flex items-center justify-between text-xs text-rose-300"
              >
                <span>Clear / Leave Empty</span>
                <X size={14} />
              </button>

              {filteredChampions.map((champ) => {
                const isPickedElsewhere = alreadyPickedSet.has(champ.id);
                const isCurrent = draftState[activeSlotModal] === champ.id;

                return (
                  <button
                    key={champ.id}
                    disabled={isPickedElsewhere}
                    onClick={() => handleSelectChampion(champ.id)}
                    className={`p-3 rounded-xl border text-left transition flex items-center justify-between gap-3 ${
                      isCurrent
                        ? "bg-cyan-950/50 border-cyan-400 ring-1 ring-cyan-400"
                        : isPickedElsewhere
                        ? "opacity-35 bg-slate-950 border-slate-900 cursor-not-allowed"
                        : "bg-slate-900/80 border-slate-800 hover:border-cyan-500/50 hover:bg-slate-800/80"
                    }`}
                  >
                    <div className="min-w-0">
                      <div className="flex items-center gap-1.5">
                        <span className="text-xs font-bold text-white">{champ.name}</span>
                        {isCurrent && <CheckCircle2 size={12} className="text-cyan-400" />}
                        {isPickedElsewhere && (
                          <span className="text-[10px] font-mono text-rose-400 bg-rose-950/60 px-1 rounded">
                            Picked
                          </span>
                        )}
                      </div>
                      <div className="text-[11px] text-slate-400 truncate mt-0.5">{champ.title}</div>
                      <div className="flex items-center gap-2 text-[10px] text-slate-500 font-mono mt-1">
                        <span>{champ.primaryRole}</span>
                        <span>•</span>
                        <span>{champ.damage.ad}% AD / {champ.damage.ap}% AP</span>
                        <span>•</span>
                        <span className="capitalize">{champ.ccProfile.replace("_", " ")}</span>
                      </div>
                    </div>

                    <div className={`w-8 h-8 rounded-lg bg-gradient-to-br ${champ.iconColor} shrink-0 opacity-80`} />
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
