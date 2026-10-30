/**
 * League of Legends 5v5 Thermodynamic Draft Phase Solver
 *
 * Models competitive draft phase as an N=10 categorical Potts spin graph:
 * 5 Blue slots (TOP, JGL, MID, ADC, SUP) vs. 5 Red slots (TOP, JGL, MID, ADC, SUP).
 *
 * Solves the ground state configuration via categorical Potts Gibbs MCMC sampling
 * with temperature beta control (from beta=0.1 high entropy chaos to beta=5.0
 * Korean Challenger meta lock-in).
 */

export type Role = "TOP" | "JGL" | "MID" | "ADC" | "SUP";
export type DamageClass = "AD" | "AP" | "Hybrid" | "True";
export type CCProfile = "knockup" | "hard_cc" | "slow" | "none";
export type ScalingProfile = "early" | "mid" | "late";

export interface Champion {
  id: string;
  name: string;
  title: string;
  primaryRole: Role;
  secondaryRoles: Role[];
  damage: {
    ad: number;     // 0 - 100 percentage
    ap: number;     // 0 - 100 percentage
    trueDmg: number;// 0 - 100 percentage
  };
  ccProfile: CCProfile;
  scaling: ScalingProfile;
  archetype: "diver" | "juggernaut" | "hypercarry" | "enchanter" | "vanguard" | "assassin" | "mage" | "duelist";
  iconColor: string; // Tailwind gradient / accent color for badge
  splashBg: string;   // High-contrast dark hex background
}

export const ROLES: Role[] = ["TOP", "JGL", "MID", "ADC", "SUP"];

export const SLOT_DEFINITIONS: { team: "blue" | "red"; role: Role; label: string; index: number }[] = [
  { team: "blue", role: "TOP", label: "Blue Top", index: 0 },
  { team: "blue", role: "JGL", label: "Blue Jungle", index: 1 },
  { team: "blue", role: "MID", label: "Blue Mid", index: 2 },
  { team: "blue", role: "ADC", label: "Blue ADC", index: 3 },
  { team: "blue", role: "SUP", label: "Blue Support", index: 4 },
  { team: "red", role: "TOP", label: "Red Top", index: 5 },
  { team: "red", role: "JGL", label: "Red Jungle", index: 6 },
  { team: "red", role: "MID", label: "Red Mid", index: 7 },
  { team: "red", role: "ADC", label: "Red ADC", index: 8 },
  { team: "red", role: "SUP", label: "Red Support", index: 9 },
];

export const CHAMPIONS: Champion[] = [
  // --- TOP LANERS ---
  {
    id: "malphite",
    name: "Malphite",
    title: "Shard of the Monolith",
    primaryRole: "TOP",
    secondaryRoles: ["SUP"],
    damage: { ad: 15, ap: 75, trueDmg: 10 },
    ccProfile: "knockup",
    scaling: "mid",
    archetype: "vanguard",
    iconColor: "from-amber-600 to-stone-700",
    splashBg: "#451a03",
  },
  {
    id: "aatrox",
    name: "Aatrox",
    title: "The Darkin Blade",
    primaryRole: "TOP",
    secondaryRoles: ["MID"],
    damage: { ad: 90, ap: 0, trueDmg: 10 },
    ccProfile: "knockup",
    scaling: "mid",
    archetype: "juggernaut",
    iconColor: "from-red-600 to-rose-950",
    splashBg: "#450a0a",
  },
  {
    id: "ksante",
    name: "K'Sante",
    title: "The Heart of the Nazumah",
    primaryRole: "TOP",
    secondaryRoles: [],
    damage: { ad: 70, ap: 0, trueDmg: 30 },
    ccProfile: "knockup",
    scaling: "late",
    archetype: "vanguard",
    iconColor: "from-amber-500 to-emerald-900",
    splashBg: "#292524",
  },
  {
    id: "darius",
    name: "Darius",
    title: "The Hand of Noxus",
    primaryRole: "TOP",
    secondaryRoles: [],
    damage: { ad: 65, ap: 0, trueDmg: 35 },
    ccProfile: "slow",
    scaling: "early",
    archetype: "juggernaut",
    iconColor: "from-red-700 to-slate-900",
    splashBg: "#3f1010",
  },
  {
    id: "fiora",
    name: "Fiora",
    title: "The Grand Duelist",
    primaryRole: "TOP",
    secondaryRoles: [],
    damage: { ad: 50, ap: 0, trueDmg: 50 },
    ccProfile: "slow",
    scaling: "late",
    archetype: "duelist",
    iconColor: "from-pink-600 to-blue-900",
    splashBg: "#4c0519",
  },
  {
    id: "jax",
    name: "Jax",
    title: "Grandmaster at Arms",
    primaryRole: "TOP",
    secondaryRoles: ["JGL"],
    damage: { ad: 60, ap: 35, trueDmg: 5 },
    ccProfile: "hard_cc",
    scaling: "late",
    archetype: "duelist",
    iconColor: "from-purple-600 to-indigo-950",
    splashBg: "#2e1065",
  },
  {
    id: "rumble",
    name: "Rumble",
    title: "The Mechanized Menace",
    primaryRole: "TOP",
    secondaryRoles: ["MID"],
    damage: { ad: 10, ap: 90, trueDmg: 0 },
    ccProfile: "slow",
    scaling: "mid",
    archetype: "mage",
    iconColor: "from-orange-500 to-amber-700",
    splashBg: "#431407",
  },
  {
    id: "sion",
    name: "Sion",
    title: "The Undead Juggernaut",
    primaryRole: "TOP",
    secondaryRoles: [],
    damage: { ad: 70, ap: 25, trueDmg: 5 },
    ccProfile: "knockup",
    scaling: "late",
    archetype: "vanguard",
    iconColor: "from-zinc-600 to-rose-950",
    splashBg: "#18181b",
  },
  {
    id: "gwen",
    name: "Gwen",
    title: "The Hallowed Seamstress",
    primaryRole: "TOP",
    secondaryRoles: [],
    damage: { ad: 10, ap: 75, trueDmg: 15 },
    ccProfile: "slow",
    scaling: "late",
    archetype: "duelist",
    iconColor: "from-cyan-400 to-blue-950",
    splashBg: "#083344",
  },

  // --- JUNGLE ---
  {
    id: "jarvaniv",
    name: "Jarvan IV",
    title: "The Exemplar of Demacia",
    primaryRole: "JGL",
    secondaryRoles: ["TOP"],
    damage: { ad: 85, ap: 15, trueDmg: 0 },
    ccProfile: "knockup",
    scaling: "mid",
    archetype: "diver",
    iconColor: "from-yellow-500 to-amber-800",
    splashBg: "#451a03",
  },
  {
    id: "leesin",
    name: "Lee Sin",
    title: "The Blind Monk",
    primaryRole: "JGL",
    secondaryRoles: [],
    damage: { ad: 95, ap: 5, trueDmg: 0 },
    ccProfile: "knockup",
    scaling: "early",
    archetype: "diver",
    iconColor: "from-emerald-500 to-teal-900",
    splashBg: "#064e3b",
  },
  {
    id: "diana",
    name: "Diana",
    title: "Scorn of the Moon",
    primaryRole: "JGL",
    secondaryRoles: ["MID"],
    damage: { ad: 10, ap: 90, trueDmg: 0 },
    ccProfile: "knockup",
    scaling: "mid",
    archetype: "diver",
    iconColor: "from-cyan-300 to-indigo-950",
    splashBg: "#0f172a",
  },
  {
    id: "sejuani",
    name: "Sejuani",
    title: "Fury of the North",
    primaryRole: "JGL",
    secondaryRoles: ["TOP"],
    damage: { ad: 10, ap: 85, trueDmg: 5 },
    ccProfile: "knockup",
    scaling: "mid",
    archetype: "vanguard",
    iconColor: "from-sky-400 to-blue-900",
    splashBg: "#0c4a6e",
  },
  {
    id: "elise",
    name: "Elise",
    title: "The Spider Queen",
    primaryRole: "JGL",
    secondaryRoles: [],
    damage: { ad: 5, ap: 95, trueDmg: 0 },
    ccProfile: "hard_cc",
    scaling: "early",
    archetype: "assassin",
    iconColor: "from-red-600 to-indigo-950",
    splashBg: "#31102f",
  },
  {
    id: "viego",
    name: "Viego",
    title: "The Ruined King",
    primaryRole: "JGL",
    secondaryRoles: ["MID"],
    damage: { ad: 85, ap: 10, trueDmg: 5 },
    ccProfile: "hard_cc",
    scaling: "late",
    archetype: "assassin",
    iconColor: "from-emerald-400 to-slate-900",
    splashBg: "#022c22",
  },
  {
    id: "vi",
    name: "Vi",
    title: "The Piltover Enforcer",
    primaryRole: "JGL",
    secondaryRoles: [],
    damage: { ad: 90, ap: 0, trueDmg: 10 },
    ccProfile: "knockup",
    scaling: "mid",
    archetype: "diver",
    iconColor: "from-pink-500 to-rose-900",
    splashBg: "#500724",
  },
  {
    id: "poppy",
    name: "Poppy",
    title: "Keeper of the Hammer",
    primaryRole: "JGL",
    secondaryRoles: ["TOP", "SUP"],
    damage: { ad: 80, ap: 15, trueDmg: 5 },
    ccProfile: "knockup",
    scaling: "mid",
    archetype: "vanguard",
    iconColor: "from-amber-400 to-blue-900",
    splashBg: "#172554",
  },

  // --- MID LANERS ---
  {
    id: "yasuo",
    name: "Yasuo",
    title: "The Unforgiven",
    primaryRole: "MID",
    secondaryRoles: ["TOP", "ADC"],
    damage: { ad: 90, ap: 5, trueDmg: 5 },
    ccProfile: "knockup",
    scaling: "late",
    archetype: "duelist",
    iconColor: "from-sky-400 to-cyan-800",
    splashBg: "#082f49",
  },
  {
    id: "yone",
    name: "Yone",
    title: "The Unforgotten",
    primaryRole: "MID",
    secondaryRoles: ["TOP"],
    damage: { ad: 65, ap: 25, trueDmg: 10 },
    ccProfile: "knockup",
    scaling: "late",
    archetype: "assassin",
    iconColor: "from-rose-500 to-indigo-950",
    splashBg: "#3b0764",
  },
  {
    id: "orianna",
    name: "Orianna",
    title: "The Lady of Clockwork",
    primaryRole: "MID",
    secondaryRoles: [],
    damage: { ad: 5, ap: 95, trueDmg: 0 },
    ccProfile: "knockup",
    scaling: "late",
    archetype: "mage",
    iconColor: "from-blue-400 to-indigo-900",
    splashBg: "#1e1b4b",
  },
  {
    id: "sylas",
    name: "Sylas",
    title: "The Dethroned",
    primaryRole: "MID",
    secondaryRoles: ["TOP", "JGL"],
    damage: { ad: 10, ap: 90, trueDmg: 0 },
    ccProfile: "hard_cc",
    scaling: "late",
    archetype: "mage",
    iconColor: "from-cyan-500 to-slate-900",
    splashBg: "#0f172a",
  },
  {
    id: "ahri",
    name: "Ahri",
    title: "The Nine-Tailed Fox",
    primaryRole: "MID",
    secondaryRoles: [],
    damage: { ad: 0, ap: 85, trueDmg: 15 },
    ccProfile: "hard_cc",
    scaling: "mid",
    archetype: "mage",
    iconColor: "from-fuchsia-400 to-pink-900",
    splashBg: "#4a044e",
  },
  {
    id: "galio",
    name: "Galio",
    title: "The Colossus",
    primaryRole: "MID",
    secondaryRoles: ["SUP"],
    damage: { ad: 10, ap: 90, trueDmg: 0 },
    ccProfile: "knockup",
    scaling: "mid",
    archetype: "vanguard",
    iconColor: "from-yellow-400 to-slate-800",
    splashBg: "#1e293b",
  },
  {
    id: "leblanc",
    name: "LeBlanc",
    title: "The Deceiver",
    primaryRole: "MID",
    secondaryRoles: [],
    damage: { ad: 0, ap: 100, trueDmg: 0 },
    ccProfile: "hard_cc",
    scaling: "early",
    archetype: "assassin",
    iconColor: "from-purple-500 to-rose-950",
    splashBg: "#3b0764",
  },
  {
    id: "kassadin",
    name: "Kassadin",
    title: "The Void Walker",
    primaryRole: "MID",
    secondaryRoles: [],
    damage: { ad: 5, ap: 95, trueDmg: 0 },
    ccProfile: "slow",
    scaling: "late",
    archetype: "assassin",
    iconColor: "from-violet-600 to-purple-950",
    splashBg: "#2e1065",
  },
  {
    id: "ziggs",
    name: "Ziggs",
    title: "The Hexplosives Expert",
    primaryRole: "MID",
    secondaryRoles: ["ADC"],
    damage: { ad: 5, ap: 95, trueDmg: 0 },
    ccProfile: "knockup",
    scaling: "late",
    archetype: "mage",
    iconColor: "from-amber-400 to-red-800",
    splashBg: "#451a03",
  },
  {
    id: "xerath",
    name: "Xerath",
    title: "The Magus Ascendant",
    primaryRole: "MID",
    secondaryRoles: ["SUP"],
    damage: { ad: 0, ap: 100, trueDmg: 0 },
    ccProfile: "hard_cc",
    scaling: "late",
    archetype: "mage",
    iconColor: "from-cyan-400 to-blue-900",
    splashBg: "#082f49",
  },

  // --- ADC (BOT CARRIES) ---
  {
    id: "jinx",
    name: "Jinx",
    title: "The Loose Cannon",
    primaryRole: "ADC",
    secondaryRoles: [],
    damage: { ad: 95, ap: 5, trueDmg: 0 },
    ccProfile: "hard_cc",
    scaling: "late",
    archetype: "hypercarry",
    iconColor: "from-pink-500 to-cyan-500",
    splashBg: "#500724",
  },
  {
    id: "kaisa",
    name: "Kai'Sa",
    title: "Daughter of the Void",
    primaryRole: "ADC",
    secondaryRoles: ["MID"],
    damage: { ad: 50, ap: 45, trueDmg: 5 },
    ccProfile: "none",
    scaling: "late",
    archetype: "hypercarry",
    iconColor: "from-purple-500 to-fuchsia-950",
    splashBg: "#3b0764",
  },
  {
    id: "jhin",
    name: "Jhin",
    title: "The Virtuoso",
    primaryRole: "ADC",
    secondaryRoles: [],
    damage: { ad: 90, ap: 5, trueDmg: 5 },
    ccProfile: "hard_cc",
    scaling: "mid",
    archetype: "hypercarry",
    iconColor: "from-amber-300 to-stone-900",
    splashBg: "#1c1917",
  },
  {
    id: "lucian",
    name: "Lucian",
    title: "The Purifier",
    primaryRole: "ADC",
    secondaryRoles: ["MID"],
    damage: { ad: 90, ap: 10, trueDmg: 0 },
    ccProfile: "none",
    scaling: "early",
    archetype: "duelist",
    iconColor: "from-yellow-400 to-slate-900",
    splashBg: "#1e293b",
  },
  {
    id: "samira",
    name: "Samira",
    title: "The Desert Rose",
    primaryRole: "ADC",
    secondaryRoles: [],
    damage: { ad: 95, ap: 5, trueDmg: 0 },
    ccProfile: "knockup",
    scaling: "mid",
    archetype: "hypercarry",
    iconColor: "from-rose-600 to-amber-900",
    splashBg: "#4c0519",
  },
  {
    id: "xayah",
    name: "Xayah",
    title: "The Rebel",
    primaryRole: "ADC",
    secondaryRoles: [],
    damage: { ad: 95, ap: 5, trueDmg: 0 },
    ccProfile: "hard_cc",
    scaling: "late",
    archetype: "hypercarry",
    iconColor: "from-violet-500 to-rose-900",
    splashBg: "#3b0764",
  },
  {
    id: "kogmaw",
    name: "Kog'Maw",
    title: "The Mouth of the Abyss",
    primaryRole: "ADC",
    secondaryRoles: ["MID"],
    damage: { ad: 40, ap: 50, trueDmg: 10 },
    ccProfile: "slow",
    scaling: "late",
    archetype: "hypercarry",
    iconColor: "from-lime-500 to-purple-950",
    splashBg: "#1a2e05",
  },
  {
    id: "missfortune",
    name: "Miss Fortune",
    title: "The Bounty Hunter",
    primaryRole: "ADC",
    secondaryRoles: [],
    damage: { ad: 85, ap: 15, trueDmg: 0 },
    ccProfile: "slow",
    scaling: "mid",
    archetype: "hypercarry",
    iconColor: "from-red-500 to-amber-700",
    splashBg: "#450a0a",
  },
  {
    id: "caitlyn",
    name: "Caitlyn",
    title: "The Sheriff of Piltover",
    primaryRole: "ADC",
    secondaryRoles: [],
    damage: { ad: 95, ap: 5, trueDmg: 0 },
    ccProfile: "hard_cc",
    scaling: "late",
    archetype: "hypercarry",
    iconColor: "from-indigo-400 to-blue-950",
    splashBg: "#1e1b4b",
  },
  {
    id: "ezreal",
    name: "Ezreal",
    title: "The Prodigal Explorer",
    primaryRole: "ADC",
    secondaryRoles: ["MID"],
    damage: { ad: 70, ap: 30, trueDmg: 0 },
    ccProfile: "none",
    scaling: "mid",
    archetype: "duelist",
    iconColor: "from-amber-300 to-sky-700",
    splashBg: "#0c4a6e",
  },

  // --- SUPPORTS ---
  {
    id: "nautilus",
    name: "Nautilus",
    title: "The Titan of the Depths",
    primaryRole: "SUP",
    secondaryRoles: ["TOP"],
    damage: { ad: 10, ap: 80, trueDmg: 10 },
    ccProfile: "knockup",
    scaling: "mid",
    archetype: "vanguard",
    iconColor: "from-teal-600 to-slate-900",
    splashBg: "#042f2e",
  },
  {
    id: "leona",
    name: "Leona",
    title: "The Radiant Dawn",
    primaryRole: "SUP",
    secondaryRoles: [],
    damage: { ad: 5, ap: 90, trueDmg: 5 },
    ccProfile: "hard_cc",
    scaling: "early",
    archetype: "vanguard",
    iconColor: "from-amber-400 to-orange-700",
    splashBg: "#451a03",
  },
  {
    id: "thresh",
    name: "Thresh",
    title: "The Chain Warden",
    primaryRole: "SUP",
    secondaryRoles: [],
    damage: { ad: 15, ap: 80, trueDmg: 5 },
    ccProfile: "knockup",
    scaling: "mid",
    archetype: "vanguard",
    iconColor: "from-emerald-400 to-cyan-950",
    splashBg: "#022c22",
  },
  {
    id: "rakan",
    name: "Rakan",
    title: "The Charmer",
    primaryRole: "SUP",
    secondaryRoles: [],
    damage: { ad: 5, ap: 95, trueDmg: 0 },
    ccProfile: "knockup",
    scaling: "late",
    archetype: "enchanter",
    iconColor: "from-yellow-400 to-rose-800",
    splashBg: "#4c0519",
  },
  {
    id: "nami",
    name: "Nami",
    title: "The Tidecaller",
    primaryRole: "SUP",
    secondaryRoles: [],
    damage: { ad: 0, ap: 100, trueDmg: 0 },
    ccProfile: "knockup",
    scaling: "mid",
    archetype: "enchanter",
    iconColor: "from-teal-400 to-cyan-800",
    splashBg: "#042f2e",
  },
  {
    id: "lulu",
    name: "Lulu",
    title: "The Fae Sorceress",
    primaryRole: "SUP",
    secondaryRoles: ["MID"],
    damage: { ad: 5, ap: 95, trueDmg: 0 },
    ccProfile: "knockup",
    scaling: "late",
    archetype: "enchanter",
    iconColor: "from-purple-400 to-pink-900",
    splashBg: "#3b0764",
  },
  {
    id: "braum",
    name: "Braum",
    title: "The Heart of the Freljord",
    primaryRole: "SUP",
    secondaryRoles: [],
    damage: { ad: 10, ap: 85, trueDmg: 5 },
    ccProfile: "knockup",
    scaling: "late",
    archetype: "vanguard",
    iconColor: "from-sky-300 to-blue-900",
    splashBg: "#0c4a6e",
  },
  {
    id: "blitzcrank",
    name: "Blitzcrank",
    title: "The Great Steam Golem",
    primaryRole: "SUP",
    secondaryRoles: [],
    damage: { ad: 15, ap: 80, trueDmg: 5 },
    ccProfile: "knockup",
    scaling: "early",
    archetype: "vanguard",
    iconColor: "from-yellow-500 to-amber-900",
    splashBg: "#451a03",
  },
];

export const CHAMPION_MAP: Record<string, Champion> = CHAMPIONS.reduce((acc, c) => {
  acc[c.id] = c;
  return acc;
}, {} as Record<string, Champion>);

// ============================================================================
// SYNERGY & COUNTER COUPLING MATRICES (Potts Interactions)
// ============================================================================

export interface SynergyRule {
  a: string;
  b: string;
  weight: number; // Negative energy reward (e.g. -25.0)
  name: string;
  description: string;
}

export const SYNERGIES: SynergyRule[] = [
  // Yasuo knockup wombo combos
  { a: "yasuo", b: "malphite", weight: -28.0, name: "Last Breath Unstoppable Force", description: "Malphite Unstoppable Force guarantees Yasuo 5-man Last Breath" },
  { a: "yasuo", b: "diana", weight: -26.0, name: "Moonfall Last Breath", description: "Diana vortex group into Yasuo airborne suspension" },
  { a: "yasuo", b: "nautilus", weight: -24.0, name: "Depth Charge Target Lock", description: "Point-and-click Depth Charge gives Yasuo guaranteed airborne" },
  { a: "yasuo", b: "jarvaniv", weight: -25.0, name: "Cataclysm Arena Wombo", description: "EQ knockup and impassable terrain inside Cataclysm" },
  { a: "yasuo", b: "rakan", weight: -24.0, name: "Grand Entrance Wombo", description: "Charm into instant AoE knockup triggering Last Breath" },
  { a: "yasuo", b: "sion", weight: -20.0, name: "Unstoppable Onslaught Setup", description: "Sion charge gives massive follow-up knockup" },
  { a: "yasuo", b: "sejuani", weight: -20.0, name: "Glacial Prison Shock", description: "Sejuani permafrost and leap knockup setup" },
  { a: "yasuo", b: "vi", weight: -22.0, name: "Cease and Desist Slam", description: "Point-and-click dunk guarantees Yasuo ult on priority carry" },

  // Ball Delivery / Area-of-Effect Wombo
  { a: "jarvaniv", b: "orianna", weight: -28.0, name: "Shockwave Cataclysm", description: "The iconic delivery system: Ball on Jarvan EQ into Shockwave inside wall" },
  { a: "malphite", b: "orianna", weight: -26.0, name: "Ball Delivery Orbital Strike", description: "Instant unstoppable dash into 5-man Shockwave" },
  { a: "jarvaniv", b: "missfortune", weight: -25.0, name: "Bullet Time Trap", description: "Cataclysm traps targets inside entire Bullet Time channel" },
  { a: "malphite", b: "missfortune", weight: -24.0, name: "Unstoppable Bullet Time", description: "AoE airborne locks targets into Bullet Time cone" },
  { a: "diana", b: "orianna", weight: -25.0, name: "Moonfall Collapse", description: "Diana pulls all 5 enemies directly onto Orianna's ball" },

  // Botlane Iconic Duos
  { a: "lucian", b: "nami", weight: -28.0, name: "Electrocute Tidecaller Burst", description: "Nami E empowers Lucian double-shot for instant electrocute 1-shot" },
  { a: "xayah", b: "rakan", weight: -28.0, name: "Lover's Leap Synergy", description: "Bonus dash range, shared recalls, and synchronized Feather/Charm CC" },
  { a: "kogmaw", b: "lulu", weight: -28.0, name: "Protect the Kog'Maw", description: "Wild Growth, Pix on-hit damage, and Polymorph peel turns Kog into God" },
  { a: "jinx", b: "lulu", weight: -26.0, name: "Hypercarry Fairy Shield", description: "Max attack speed steroids and peel for passive resets" },
  { a: "samira", b: "nautilus", weight: -25.0, name: "All-in Style Stack Reset", description: "Passive knockup extension from Nautilus CC triggers instant S-rank" },
  { a: "samira", b: "leona", weight: -24.0, name: "Sunlight CC Lock", description: "Leona triple CC lockdown enables uninterrupted Inferno Trigger" },

  // Melee Permafrost Synergy
  { a: "sejuani", b: "aatrox", weight: -18.0, name: "Permafrost Drain Bruiser", description: "Melee auto-attacks quickly build Sejuani frost stacks" },
  { a: "sejuani", b: "yone", weight: -20.0, name: "Freljord Soul Unbound", description: "Rapid melee strikes rapidly trigger instant freeze" },
  { a: "sejuani", b: "ksante", weight: -18.0, name: "Unyielding Permafrost Frontline", description: "Dual tank crowd-control lockdown" },
  { a: "sejuani", b: "jax", weight: -19.0, name: "Counterstrike Freeze Chain", description: "Melee auto-reset stack generation" },

  // Frontline Peel & Hypercarry Enablers
  { a: "sion", b: "jinx", weight: -18.0, name: "Front-to-Back Space Maker", description: "Massive health meat-shield allows Jinx to safely ramp Get Excited" },
  { a: "ksante", b: "caitlyn", weight: -18.0, name: "Isolation Peel & Headshot", description: "Kidnaps threats away from Caitlyn's range zone" },
  { a: "braum", b: "lucian", weight: -22.0, name: "Lightslinger Concussive Blows", description: "Lucian passive applies 2 instant Braum stun stacks per spell" },
  { a: "thresh", b: "jhin", weight: -18.0, name: "Deadly Chain Execution", description: "Hook into Deadly Flourish and Curtain's Call sniper setup" },
];

export interface CounterRule {
  counter: string;     // Champion that counters
  victim: string;      // Champion being countered
  penalty: number;     // Positive energy penalty applied to victim's team (+20.0 to +30.0)
  name: string;
  reason: string;
}

export const COUNTERS: CounterRule[] = [
  // Heavy AP counters
  { counter: "galio", victim: "orianna", penalty: 24.0, name: "Anti-Magic Bulwark", reason: "Galio magic shield and taunt completely neutralizes mage burst" },
  { counter: "galio", victim: "leblanc", penalty: 26.0, name: "Mage-Buster Intervention", reason: "Hero's Entrance negates assassin dive" },
  { counter: "galio", victim: "ahri", penalty: 22.0, name: "Anti-Charm Fortress", reason: "Heavy MR scaling punishes AP burst" },
  { counter: "kassadin", victim: "orianna", penalty: 24.0, name: "Null Sphere Suppression", reason: "Kassadin scales freely into AP control mages" },
  { counter: "kassadin", victim: "ziggs", penalty: 24.0, name: "Riftwalk Artillary Flank", reason: "Closes the distance on long-range poke instantly" },

  // Anti-Tank Shredders
  { counter: "fiora", victim: "sion", penalty: 25.0, name: "Percent Max-HP True Shred", reason: "Fiora vitals ignore Sion's infinite bonus health" },
  { counter: "fiora", victim: "malphite", penalty: 22.0, name: "Riposte Unstoppable Force", reason: "Riposte can parry Malphite ult and stun him in return" },
  { counter: "gwen", victim: "sion", penalty: 24.0, name: "Snip Snip Max HP Melter", reason: "Needlework shreds immobile colossal tanks" },
  { counter: "gwen", victim: "ksante", penalty: 20.0, name: "Hallowed Mist Immunity", reason: "Gwen immune to outside CC and melts resists" },
  { counter: "kogmaw", victim: "sion", penalty: 24.0, name: "Bio-Arcane Tank Melter", reason: "Massive percentage max-HP magic damage eats meat-shields" },

  // Projectile Blockers
  { counter: "braum", victim: "missfortune", penalty: 26.0, name: "Unbreakable Bullet Time", reason: "Braum shield completely absorbs entire Miss Fortune ultimate" },
  { counter: "braum", victim: "jhin", penalty: 22.0, name: "Curtain Call Absorption", reason: "Shield absorbs all 4 sniper shots with ease" },
  { counter: "yasuo", victim: "missfortune", penalty: 25.0, name: "Wind Wall Bullet Blocker", reason: "Windwall erases full team Bullet Time cone" },
  { counter: "yasuo", victim: "orianna", penalty: 20.0, name: "Wind Wall Ball Blocker", reason: "Can intercept ball flight path in narrow chokepoints" },

  // Anti-Dive & Anti-Dash
  { counter: "poppy", victim: "yasuo", penalty: 24.0, name: "Steadfast Anti-Dash", reason: "Grounds and knocks up Yasuo whenever he attempts Sweeping Blade" },
  { counter: "poppy", victim: "lee sin", penalty: 24.0, name: "Sonic Wave Denied", reason: "Poppy W interrupts Resonating Strike completely" },
  { counter: "poppy", victim: "rakan", penalty: 25.0, name: "Grand Entrance Denied", reason: "Interrupts Rakan W mid-dash and grounds him" },
  { counter: "poppy", victim: "diana", penalty: 22.0, name: "Lunar Rush Block", reason: "Prevents Diana dive onto backline" },

  // Immobile Carry Hunters
  { counter: "jarvaniv", victim: "jinx", penalty: 24.0, name: "Cataclysm Cage Trap", reason: "Flashless Jinx has zero mobility spells to escape Cataclysm" },
  { counter: "jarvaniv", victim: "kogmaw", penalty: 24.0, name: "Immobile Carry Entombed", reason: "Kog'Maw trapped in wall with no way to kite" },
  { counter: "malphite", victim: "jinx", penalty: 22.0, name: "Unstoppable Assassination", reason: "Ground slam attack speed slow and point-blank knockup" },
  { counter: "diana", victim: "jhin", penalty: 20.0, name: "Moonfall Collapse onto Sniper", reason: "Dives onto stationary Curtain's Call" },
  { counter: "vi", victim: "jinx", penalty: 24.0, name: "Unavoidable Lockdown", reason: "Point-and-click R cannot be cleansed or dodged by Jinx" },
];

// ============================================================================
// DRAFT STATE & HAMILTONIAN CALCULATOR
// ============================================================================

export type DraftState = (string | null)[]; // Array of 10 champion IDs (or null if empty)

export interface TeamMetrics {
  roleClashEnergy: number;
  intraSynergyEnergy: number;
  counterAdvantageEnergy: number; // Negative is good for this team
  damageBalanceEnergy: number;
  ccChainEnergy: number;
  totalEnergy: number;
  damageSplit: {
    adPct: number;
    apPct: number;
    truePct: number;
  };
  ccScore: number; // 0 - 100 percentage
  activeSynergies: string[];
  activeCounters: string[];
}

export interface DraftMetrics {
  blue: TeamMetrics;
  red: TeamMetrics;
  netDeltaH: number; // H(Red) - H(Blue). Positive = Blue advantage!
  blueWinProb: number; // 0 - 100%
}

/**
 * Calculates the complete Hamiltonian energy for Blue and Red teams
 */
export function calculateDraftHamiltonian(state: DraftState): DraftMetrics {
  const blueIds = state.slice(0, 5).filter(Boolean) as string[];
  const redIds = state.slice(5, 10).filter(Boolean) as string[];

  const blueChamps = blueIds.map((id) => CHAMPION_MAP[id]).filter(Boolean);
  const redChamps = redIds.map((id) => CHAMPION_MAP[id]).filter(Boolean);

  // 1. DUPLICATE PICK PENALTY (Cannot pick same champ twice in draft)
  let blueDupPenalty = 0;
  let redDupPenalty = 0;
  const seenAll = new Set<string>();
  for (let i = 0; i < 10; i++) {
    const id = state[i];
    if (!id) continue;
    if (seenAll.has(id)) {
      if (i < 5) blueDupPenalty += 200.0;
      else redDupPenalty += 200.0;
    }
    seenAll.add(id);
  }

  // 2. ROLE CLASH PENALTY
  const evalRoleClash = (slots: (string | null)[], offset: number): number => {
    let clash = 0;
    for (let i = 0; i < 5; i++) {
      const id = slots[i];
      if (!id) continue;
      const champ = CHAMPION_MAP[id];
      if (!champ) continue;
      const targetRole = ROLES[i];
      if (champ.primaryRole === targetRole) {
        // Perfect fit
      } else if (champ.secondaryRoles.includes(targetRole)) {
        clash += 5.0; // Minor off-role flex tax
      } else {
        clash += 100.0; // Harsh role mismatch penalty
      }
    }
    return clash;
  };

  const blueRoleClash = evalRoleClash(state.slice(0, 5), 0) + blueDupPenalty;
  const redRoleClash = evalRoleClash(state.slice(5, 10), 5) + redDupPenalty;

  // 3. INTRA-TEAM SYNERGY (Negative reward)
  const evalIntraSynergy = (champs: Champion[]) => {
    let energy = 0;
    const active: string[] = [];
    const champSet = new Set(champs.map((c) => c.id));

    for (const syn of SYNERGIES) {
      if (champSet.has(syn.a) && champSet.has(syn.b)) {
        energy += syn.weight;
        active.push(`${syn.name} (${syn.weight.toFixed(0)})`);
      }
    }
    return { energy, active };
  };

  const blueSynergy = evalIntraSynergy(blueChamps);
  const redSynergy = evalIntraSynergy(redChamps);

  // 4. INTER-TEAM COUNTERS
  const evalCounters = (attackerTeam: Champion[], defenderTeam: Champion[]) => {
    let penaltyAgainstDefender = 0;
    const activeRules: string[] = [];
    const attSet = new Set(attackerTeam.map((c) => c.id));
    const defSet = new Set(defenderTeam.map((c) => c.id));

    for (const rule of COUNTERS) {
      if (attSet.has(rule.counter) && defSet.has(rule.victim)) {
        penaltyAgainstDefender += rule.penalty;
        activeRules.push(`${rule.name}: ${rule.counter} counters ${rule.victim} (+${rule.penalty})`);
      }
    }

    // Heavy AP composition check (>70% total team damage is AP against Galio / Kassadin)
    const defTotalAP = defenderTeam.reduce((acc, c) => acc + c.damage.ap, 0);
    const defTotal = defenderTeam.reduce((acc, c) => acc + c.damage.ad + c.damage.ap + c.damage.trueDmg, 0) || 1;
    const defApShare = defTotalAP / defTotal;

    if (defApShare >= 0.7 && (attSet.has("galio") || attSet.has("kassadin"))) {
      penaltyAgainstDefender += 30.0;
      activeRules.push(`Heavy AP Trap: Defending team is ${(defApShare * 100).toFixed(0)}% AP into Anti-Mage (+30.0)`);
    }

    return { penaltyAgainstDefender, activeRules };
  };

  const redCountersBlue = evalCounters(redChamps, blueChamps);
  const blueCountersRed = evalCounters(blueChamps, redChamps);

  // 5. DAMAGE BALANCE PENALTY
  const evalDamage = (champs: Champion[]) => {
    if (champs.length === 0) return { penalty: 0, split: { adPct: 33, apPct: 33, truePct: 34 } };
    let totalAd = 0;
    let totalAp = 0;
    let totalTrue = 0;
    for (const c of champs) {
      totalAd += c.damage.ad;
      totalAp += c.damage.ap;
      totalTrue += c.damage.trueDmg;
    }
    const sum = totalAd + totalAp + totalTrue || 1;
    const adPct = Math.round((totalAd / sum) * 100);
    const apPct = Math.round((totalAp / sum) * 100);
    const truePct = 100 - adPct - apPct;

    let penalty = 0;
    if (adPct >= 80) {
      penalty += 35.0 * ((adPct - 80) / 20); // Full AD penalty
    }
    if (apPct >= 80) {
      penalty += 35.0 * ((apPct - 80) / 20); // Full AP penalty
    }

    return { penalty, split: { adPct, apPct, truePct } };
  };

  const blueDmg = evalDamage(blueChamps);
  const redDmg = evalDamage(redChamps);

  // 6. CC CHAIN SCORE & UTILITY
  const evalCC = (champs: Champion[]) => {
    if (champs.length === 0) return { energy: 0, score: 0 };
    let knockups = 0;
    let hardCC = 0;
    let slows = 0;
    for (const c of champs) {
      if (c.ccProfile === "knockup") {
        knockups++;
        hardCC++;
      } else if (c.ccProfile === "hard_cc") {
        hardCC++;
      } else if (c.ccProfile === "slow") {
        slows++;
      }
    }

    let energy = 0;
    if (knockups >= 2 && hardCC >= 3) {
      energy -= 18.0; // Layered crowd control wombo reward
    } else if (hardCC >= 2) {
      energy -= 10.0;
    } else if (hardCC === 0) {
      energy += 30.0; // Crippling 0-CC penalty
    }

    // CC score 0 - 100%
    const score = Math.min(100, Math.round(knockups * 30 + (hardCC - knockups) * 18 + slows * 8));
    return { energy, score };
  };

  const blueCC = evalCC(blueChamps);
  const redCC = evalCC(redChamps);

  // TOTAL ENERGIES
  const blueTotal =
    blueRoleClash +
    blueSynergy.energy +
    redCountersBlue.penaltyAgainstDefender +
    blueDmg.penalty +
    blueCC.energy;

  const redTotal =
    redRoleClash +
    redSynergy.energy +
    blueCountersRed.penaltyAgainstDefender +
    redDmg.penalty +
    redCC.energy;

  // DELTA & WIN PROBABILITY (lower energy is superior)
  const netDeltaH = redTotal - blueTotal;
  // Logistic function with temperature scaling:
  const blueWinProb = Math.min(96, Math.max(4, Math.round(100 / (1 + Math.exp(-netDeltaH / 22.0)))));

  return {
    blue: {
      roleClashEnergy: Math.round(blueRoleClash * 10) / 10,
      intraSynergyEnergy: Math.round(blueSynergy.energy * 10) / 10,
      counterAdvantageEnergy: Math.round(redCountersBlue.penaltyAgainstDefender * 10) / 10,
      damageBalanceEnergy: Math.round(blueDmg.penalty * 10) / 10,
      ccChainEnergy: Math.round(blueCC.energy * 10) / 10,
      totalEnergy: Math.round(blueTotal * 10) / 10,
      damageSplit: blueDmg.split,
      ccScore: blueCC.score,
      activeSynergies: blueSynergy.active,
      activeCounters: blueCountersRed.activeRules,
    },
    red: {
      roleClashEnergy: Math.round(redRoleClash * 10) / 10,
      intraSynergyEnergy: Math.round(redSynergy.energy * 10) / 10,
      counterAdvantageEnergy: Math.round(blueCountersRed.penaltyAgainstDefender * 10) / 10,
      damageBalanceEnergy: Math.round(redDmg.penalty * 10) / 10,
      ccChainEnergy: Math.round(redCC.energy * 10) / 10,
      totalEnergy: Math.round(redTotal * 10) / 10,
      damageSplit: redDmg.split,
      ccScore: redCC.score,
      activeSynergies: redSynergy.active,
      activeCounters: redCountersBlue.activeRules,
    },
    netDeltaH: Math.round(netDeltaH * 10) / 10,
    blueWinProb,
  };
}

// ============================================================================
// VECTORIZED CATEGORICAL POTTS GIBBS SAMPLER
// ============================================================================

export interface McmcTelemetry {
  runtimeMs: number;
  sweepsCompleted: number;
  transitionCount: number;
  acceptanceRate: number; // 0 - 100%
  initialEnergy: number;
  finalEnergy: number;
  deltaEnergy: number;
}

/**
 * Executes high-performance client-side Potts Gibbs sampling.
 *
 * @param currentState Current 10-node draft state
 * @param pinnedSlots Boolean array of length 10 indicating which slots are frozen
 * @param beta Inverse temperature (0.1 = hot chaos, 5.0 = cold ground state)
 * @param numSweeps Number of Gibbs sweeps (default: 50 sweeps)
 */
export function runPottsDraftSampler(
  currentState: DraftState,
  pinnedSlots: boolean[],
  beta: number = 2.0,
  numSweeps: number = 50
): { nextState: DraftState; telemetry: McmcTelemetry } {
  const startTime = typeof performance !== "undefined" ? performance.now() : Date.now();
  const state: DraftState = [...currentState];

  const initialMetrics = calculateDraftHamiltonian(state);
  const initialEnergy = initialMetrics.blue.totalEnergy + initialMetrics.red.totalEnergy;

  let transitionCount = 0;
  let totalEvaluations = 0;

  // Unpinned slots to update
  const unpinnedIndices: number[] = [];
  for (let i = 0; i < 10; i++) {
    if (!pinnedSlots[i]) {
      unpinnedIndices.push(i);
    }
  }

  if (unpinnedIndices.length === 0) {
    const endTime = typeof performance !== "undefined" ? performance.now() : Date.now();
    return {
      nextState: state,
      telemetry: {
        runtimeMs: Math.round((endTime - startTime) * 100) / 100,
        sweepsCompleted: 0,
        transitionCount: 0,
        acceptanceRate: 100,
        initialEnergy: Math.round(initialEnergy * 10) / 10,
        finalEnergy: Math.round(initialEnergy * 10) / 10,
        deltaEnergy: 0,
      },
    };
  }

  // Pre-filter viable candidates by role to optimize combinatorial search
  const candidatesByRole: Record<Role, Champion[]> = {
    TOP: CHAMPIONS.filter((c) => c.primaryRole === "TOP" || c.secondaryRoles.includes("TOP")),
    JGL: CHAMPIONS.filter((c) => c.primaryRole === "JGL" || c.secondaryRoles.includes("JGL")),
    MID: CHAMPIONS.filter((c) => c.primaryRole === "MID" || c.secondaryRoles.includes("MID")),
    ADC: CHAMPIONS.filter((c) => c.primaryRole === "ADC" || c.secondaryRoles.includes("ADC")),
    SUP: CHAMPIONS.filter((c) => c.primaryRole === "SUP" || c.secondaryRoles.includes("SUP")),
  };

  // Run Gibbs sweeps
  for (let sweep = 0; sweep < numSweeps; sweep++) {
    for (const slotIndex of unpinnedIndices) {
      const targetRole = ROLES[slotIndex % 5];
      const candidates = candidatesByRole[targetRole];
      const isBlue = slotIndex < 5;

      // Filter out champions picked in other slots
      const pickedElsewhere = new Set<string>();
      for (let j = 0; j < 10; j++) {
        if (j !== slotIndex && state[j]) {
          pickedElsewhere.add(state[j]!);
        }
      }
      const legalCandidates = candidates.filter((c) => !pickedElsewhere.has(c.id));
      if (legalCandidates.length === 0) continue;

      // Compute conditional energies E(c)
      const energies: number[] = new Array(legalCandidates.length);
      let minEnergy = Infinity;

      for (let cIdx = 0; cIdx < legalCandidates.length; cIdx++) {
        state[slotIndex] = legalCandidates[cIdx].id;
        const metrics = calculateDraftHamiltonian(state);
        // Objective: optimize the current team's energy (Blue wants low Blue energy, Red wants low Red energy)
        const teamEnergy = isBlue ? metrics.blue.totalEnergy : metrics.red.totalEnergy;
        energies[cIdx] = teamEnergy;
        if (teamEnergy < minEnergy) {
          minEnergy = teamEnergy;
        }
      }

      // Boltzmann Gibbs distribution: P(c) ∝ exp(-beta * (E(c) - minE))
      const weights: number[] = new Array(legalCandidates.length);
      let sumWeight = 0;
      for (let cIdx = 0; cIdx < legalCandidates.length; cIdx++) {
        const delta = energies[cIdx] - minEnergy;
        const w = Math.exp(-beta * delta);
        weights[cIdx] = w;
        sumWeight += w;
      }

      // Sample next state
      const rand = Math.random() * sumWeight;
      let cumulative = 0;
      let chosenChampion = legalCandidates[0].id;
      for (let cIdx = 0; cIdx < legalCandidates.length; cIdx++) {
        cumulative += weights[cIdx];
        if (rand <= cumulative) {
          chosenChampion = legalCandidates[cIdx].id;
          break;
        }
      }

      const prev = state[slotIndex];
      state[slotIndex] = chosenChampion;
      totalEvaluations++;
      if (prev !== chosenChampion) {
        transitionCount++;
      }
    }
  }

  const finalMetrics = calculateDraftHamiltonian(state);
  const finalEnergy = finalMetrics.blue.totalEnergy + finalMetrics.red.totalEnergy;
  const endTime = typeof performance !== "undefined" ? performance.now() : Date.now();

  return {
    nextState: state,
    telemetry: {
      runtimeMs: Math.round((endTime - startTime) * 100) / 100,
      sweepsCompleted: numSweeps,
      transitionCount,
      acceptanceRate: totalEvaluations > 0 ? Math.round((transitionCount / totalEvaluations) * 100) : 100,
      initialEnergy: Math.round(initialEnergy * 10) / 10,
      finalEnergy: Math.round(finalEnergy * 10) / 10,
      deltaEnergy: Math.round((finalEnergy - initialEnergy) * 10) / 10,
    },
  };
}

// ============================================================================
// DRAFT PRESETS
// ============================================================================

export interface DraftPreset {
  id: string;
  name: string;
  tagline: string;
  description: string;
  state: DraftState;
}

export const DRAFT_PRESETS: DraftPreset[] = [
  {
    id: "wombo-knockup",
    name: "Wombo Combo Knockup",
    tagline: "Airborne Chain Ground State",
    description: "Iconic five-man knockup delivery: Malphite & Diana engage into Yasuo Last Breath and Samira follow-up.",
    state: [
      "malphite",   // Blue Top
      "diana",      // Blue JGL
      "yasuo",      // Blue MID
      "samira",     // Blue ADC
      "nautilus",   // Blue SUP
      "aatrox",     // Red Top
      "leesin",     // Red JGL
      "ahri",       // Red MID
      "jinx",       // Red ADC
      "thresh",     // Red SUP
    ],
  },
  {
    id: "hypercarry-protect",
    name: "Protect the Hyper-Carry",
    tagline: "Colossal Frontline + Enchanter Steroids",
    description: "Sion and Sejuani build an impenetrable meat-shield for Lulu-empowered Jinx or Kog'Maw.",
    state: [
      "sion",       // Blue Top
      "sejuani",    // Blue JGL
      "orianna",    // Blue MID
      "jinx",       // Blue ADC
      "lulu",       // Blue SUP
      "fiora",      // Red Top
      "jarvaniv",   // Red JGL
      "sylas",      // Red MID
      "kaisa",      // Red ADC
      "leona",      // Red SUP
    ],
  },
  {
    id: "splitpush-skirmish",
    name: "Split-Push Skirmishers",
    tagline: "High-Mobility Isolated Duels",
    description: "Fiora and Jax split-push sidelanes with Xayah and Rakan holding the central wave.",
    state: [
      "fiora",      // Blue Top
      "leesin",     // Blue JGL
      "yone",       // Blue MID
      "xayah",      // Blue ADC
      "rakan",      // Blue SUP
      "ksante",     // Red Top
      "vi",         // Red JGL
      "galio",      // Red MID
      "caitlyn",    // Red ADC
      "braum",      // Red SUP
    ],
  },
  {
    id: "full-ap-trap",
    name: "Full AP Trap (Countered)",
    tagline: "Severe Energetic Clashing",
    description: "Blue drafts an all-magic AP roster right into Galio and Kassadin magic shields, causing energy to spike.",
    state: [
      "rumble",     // Blue Top (90% AP)
      "elise",      // Blue JGL (95% AP)
      "leblanc",    // Blue MID (100% AP)
      "ziggs",      // Blue ADC (95% AP)
      "xerath",     // Blue SUP (100% AP)
      "ksante",     // Red Top
      "sejuani",    // Red JGL
      "galio",      // Red MID (Counterpick!)
      "jhin",       // Red ADC
      "nautilus",   // Red SUP
    ],
  },
];
