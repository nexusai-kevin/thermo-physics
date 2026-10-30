/**
 * Middle-earth Fellowship Thermodynamic Routing Engine & Esoteric Systems Atlas
 *
 * Models the Quest of the Ring across Middle-earth as a continuous-discrete
 * potential energy landscape U(x, y) with Kramers barrier-hopping (Station 8)
 * and combinatorial path action optimization (Station 9).
 */

export interface Waypoint {
  id: string;
  name: string;
  sindarinName: string;
  region: "eriador" | "misty_mountains" | "rhovanion" | "gondor" | "mordor";
  x: number; // 0 - 1000 coordinate
  y: number; // 0 - 800 coordinate
  basePotential: number; // U_0: Negative = Sanctuary, Positive = Threat/Hazard
  barrierHeight: number; // Delta U (Kramers activation barrier)
  friction: number; // Damping gamma (terrain resistance / marsh mud)
  sauronExposure: number; // 0 - 1 baseline visibility to the Eye
  lore: string;
  hazardType: "sanctuary" | "mountain_pass" | "dungeon" | "marsh" | "fortress" | "chokepoint" | "volcano";
}

export interface Edge {
  from: string;
  to: string;
  distance: number;
  terrainFriction: number;
  barrierCost: number;
  stealthRisk: number;
  lore: string;
}

export const WAYPOINTS: Waypoint[] = [
  {
    id: "rivendell",
    name: "Rivendell",
    sindarinName: "Imladris",
    region: "eriador",
    x: 180,
    y: 140,
    basePotential: -35.0, // Deep sanctuary well
    barrierHeight: 5.0,
    friction: 1.0,
    sauronExposure: 0.05,
    hazardType: "sanctuary",
    lore: "The Last Homely House East of the Sea. Protected by the Elven Ring Vilya and Lord Elrond's enchantments.",
  },
  {
    id: "hollin",
    name: "Hollin",
    sindarinName: "Eregion",
    region: "eriador",
    x: 230,
    y: 250,
    basePotential: 5.0,
    barrierHeight: 12.0,
    friction: 1.2,
    sauronExposure: 0.25,
    hazardType: "chokepoint",
    lore: "Ancient realm of the Elven-smiths. Watched by flocks of Crebain out of Fangorn and Dunland.",
  },
  {
    id: "caradhras",
    name: "Caradhras Pass",
    sindarinName: "Barazinbar",
    region: "misty_mountains",
    x: 320,
    y: 230,
    basePotential: 45.0,
    barrierHeight: 75.0, // Extreme saddle point barrier
    friction: 3.5,
    sauronExposure: 0.40,
    hazardType: "mountain_pass",
    lore: "The Redhorn Gate. Cruel mountain spirit wielding blizzard, howling winds, and crushing avalanches.",
  },
  {
    id: "west_gate_moria",
    name: "West Gate of Moria",
    sindarinName: "Dorn an Anduin",
    region: "misty_mountains",
    x: 275,
    y: 330,
    basePotential: 20.0,
    barrierHeight: 30.0,
    friction: 2.0,
    sauronExposure: 0.20,
    hazardType: "chokepoint",
    lore: "The Doors of Durin, inscribed in ithildin. Guarded by the tentacled Watcher in the Water.",
  },
  {
    id: "mazarbul",
    name: "Chamber of Mazarbul",
    sindarinName: "Khazad-dûm Depths",
    region: "misty_mountains",
    x: 345,
    y: 340,
    basePotential: 40.0,
    barrierHeight: 65.0, // Balrog Kramers barrier
    friction: 2.2,
    sauronExposure: 0.15,
    hazardType: "dungeon",
    lore: "Tomb of Balin and the Bridge of Khazad-dûm. Abode of the Flame of Udûn: Durin's Bane.",
  },
  {
    id: "lothlorien",
    name: "Lothlórien",
    sindarinName: "Laurelindórenan",
    region: "rhovanion",
    x: 430,
    y: 350,
    basePotential: -40.0, // Lowest energy sanctuary well
    barrierHeight: 8.0,
    friction: 0.8,
    sauronExposure: 0.02,
    hazardType: "sanctuary",
    lore: "The Golden Wood of mallorn trees. Shielded from the Eye by Galadriel's ring Nenya and the Mirror.",
  },
  {
    id: "great_river",
    name: "The Great River",
    sindarinName: "Anduin",
    region: "rhovanion",
    x: 480,
    y: 440,
    basePotential: -5.0,
    barrierHeight: 15.0,
    friction: 1.1, // Smooth fluvial drift
    sauronExposure: 0.35,
    hazardType: "chokepoint",
    lore: "Long journey by Elven boat between towering crags, drifting past the Argonath: Pillars of the Kings.",
  },
  {
    id: "amon_hen",
    name: "Amon Hen",
    sindarinName: "Hill of the Eye",
    region: "rhovanion",
    x: 520,
    y: 520,
    basePotential: 25.0,
    barrierHeight: 35.0,
    friction: 1.8,
    sauronExposure: 0.60,
    hazardType: "chokepoint",
    lore: "Seat of Seeing overlooking the Great River. Site of Boromir's fall and the Breaking of the Fellowship.",
  },
  {
    id: "dead_marshes",
    name: "The Dead Marshes",
    sindarinName: "Nindalf / Wetwang",
    region: "mordor",
    x: 620,
    y: 450,
    basePotential: 30.0,
    barrierHeight: 25.0,
    friction: 5.0, // Heavy viscous damping
    sauronExposure: 0.45,
    hazardType: "marsh",
    lore: "Vast fetid bogs where the dead faces of the Last Alliance float beneath murky pools. Guided by Gollum.",
  },
  {
    id: "black_gate",
    name: "The Black Gate",
    sindarinName: "Morannon",
    region: "mordor",
    x: 690,
    y: 460,
    basePotential: 85.0,
    barrierHeight: 95.0, // Immense military activation barrier
    friction: 3.0,
    sauronExposure: 0.98, // Extreme Gaze
    hazardType: "fortress",
    lore: "Iron ramparts, Towers of the Teeth, and legions of Orcs guarding the narrow pass of Cirith Gorgor.",
  },
  {
    id: "ithilien",
    name: "Ithilien",
    sindarinName: "Moon-land",
    region: "gondor",
    x: 630,
    y: 560,
    basePotential: -15.0, // Forest cover sanctuary
    barrierHeight: 15.0,
    friction: 1.4,
    sauronExposure: 0.30,
    hazardType: "sanctuary",
    lore: "Fair woodland of aromatic bay, myrtle, and cedar. Patrolled by Faramir's Rangers of Ithilien.",
  },
  {
    id: "henneth_annun",
    name: "Henneth Annûn",
    sindarinName: "Window of the Sunset",
    region: "gondor",
    x: 670,
    y: 570,
    basePotential: -25.0,
    barrierHeight: 10.0,
    friction: 1.0,
    sauronExposure: 0.08,
    hazardType: "sanctuary",
    lore: "Secret refuge behind a thundering curtain of water. Safe haven for the Ringbearer before the pass.",
  },
  {
    id: "minas_morgul",
    name: "Minas Morgul",
    sindarinName: "Tower of Sorcery",
    region: "mordor",
    x: 720,
    y: 600,
    basePotential: 65.0,
    barrierHeight: 60.0,
    friction: 2.5,
    sauronExposure: 0.85,
    hazardType: "fortress",
    lore: "City of the Nazgûl radiating a pale, ghastly light. Imbued with corrupting psychic terror.",
  },
  {
    id: "cirith_ungol",
    name: "Cirith Ungol",
    sindarinName: "Pass of the Spider",
    region: "mordor",
    x: 760,
    y: 560,
    basePotential: 55.0,
    barrierHeight: 65.0, // High narrow bottleneck
    friction: 3.8,
    sauronExposure: 0.50, // Screened from the eye, but deadly
    hazardType: "mountain_pass",
    lore: "The Straight and Winding Stairs leading to the foul tunnel of Torech Ungol: Shelob's Lair.",
  },
  {
    id: "gorgoroth",
    name: "Plains of Gorgoroth",
    sindarinName: "Plateau of Terror",
    region: "mordor",
    x: 830,
    y: 530,
    basePotential: 50.0,
    barrierHeight: 35.0,
    friction: 2.5,
    sauronExposure: 0.95, // Direct sightline of Barad-dûr
    hazardType: "chokepoint",
    lore: "Blasted volcanic plateau choking with ash, pits, and military camps under the unblinking Red Eye.",
  },
  {
    id: "mount_doom",
    name: "Mount Doom",
    sindarinName: "Orodruin / Amon Amarth",
    region: "mordor",
    x: 900,
    y: 560,
    basePotential: 20.0,
    barrierHeight: 45.0,
    friction: 3.0,
    sauronExposure: 0.90,
    hazardType: "volcano",
    lore: "The Cracks of Doom (Sammath Naur). The primordial forge where the One Ring was wrought and must be unmade.",
  },
  {
    id: "minas_tirith",
    name: "Minas Tirith",
    sindarinName: "Tower of Guard",
    region: "gondor",
    x: 580,
    y: 650,
    basePotential: -5.0,
    barrierHeight: 25.0,
    friction: 1.2,
    sauronExposure: 0.55,
    hazardType: "fortress",
    lore: "The White City of Gondor built in seven stone tiers against Mount Mindolluin.",
  },
  {
    id: "helms_deep",
    name: "Helm's Deep",
    sindarinName: "Aglarond Gorge",
    region: "rhovanion",
    x: 390,
    y: 540,
    basePotential: 5.0,
    barrierHeight: 20.0,
    friction: 1.3,
    sauronExposure: 0.20,
    hazardType: "fortress",
    lore: "The Hornburg stronghold guarding the Deeping Wall and the glittering caves of Aglarond.",
  },
];

export const WAYPOINT_MAP: Record<string, Waypoint> = WAYPOINTS.reduce((acc, wp) => {
  acc[wp.id] = wp;
  return acc;
}, {} as Record<string, Waypoint>);

// Predefined Network of Navigable Paths across Middle-earth
export const EDGES: Edge[] = [
  // Rivendell South
  { from: "rivendell", to: "hollin", distance: 110, terrainFriction: 1.2, barrierCost: 10, stealthRisk: 0.20, lore: "Traversing the Redhorn road south along the foothills" },

  // Misty Mountain Crossing Dilemma (Caradhras vs Moria)
  { from: "hollin", to: "caradhras", distance: 95, terrainFriction: 3.5, barrierCost: 75, stealthRisk: 0.35, lore: "Ascending the snow-blind slopes of the Redhorn Gate" },
  { from: "hollin", to: "west_gate_moria", distance: 90, terrainFriction: 2.0, barrierCost: 30, stealthRisk: 0.20, lore: "Marching to the Sirannon pool before the Doors of Durin" },
  { from: "caradhras", to: "lothlorien", distance: 140, terrainFriction: 3.0, barrierCost: 60, stealthRisk: 0.25, lore: "Descent through the Dimrill Stair into Nanduhirion" },
  { from: "west_gate_moria", to: "mazarbul", distance: 75, terrainFriction: 2.5, barrierCost: 45, stealthRisk: 0.15, lore: "Delving through the dark mines of the Dwarrowdelf" },
  { from: "mazarbul", to: "lothlorien", distance: 90, terrainFriction: 2.0, barrierCost: 40, stealthRisk: 0.10, lore: "Fleeing across the Bridge of Khazad-dûm into Dimrill Dale" },

  // Golden Wood & River
  { from: "lothlorien", to: "great_river", distance: 100, terrainFriction: 1.0, barrierCost: 10, stealthRisk: 0.15, lore: "Embarking in grey Elven boats along the Silverlode to the Great River" },
  { from: "great_river", to: "amon_hen", distance: 90, terrainFriction: 1.1, barrierCost: 15, stealthRisk: 0.30, lore: "Rowing past the towering sentinel Argonath to the lake of Nen Hithoel" },

  // Crossing from Amon Hen to Mordor or South
  { from: "amon_hen", to: "dead_marshes", distance: 120, terrainFriction: 4.8, barrierCost: 35, stealthRisk: 0.40, lore: "Navigating the Emyn Muil labyrinth into the corpse-candles of the Marshes" },
  { from: "amon_hen", to: "helms_deep", distance: 135, terrainFriction: 1.3, barrierCost: 15, stealthRisk: 0.25, lore: "The Three Hunters galloping across the plains of Rohan" },
  { from: "amon_hen", to: "minas_tirith", distance: 145, terrainFriction: 1.2, barrierCost: 20, stealthRisk: 0.40, lore: "Riding south toward the capital of Gondor" },
  { from: "helms_deep", to: "minas_tirith", distance: 210, terrainFriction: 1.1, barrierCost: 15, stealthRisk: 0.20, lore: "Ride of the Rohirrim along the northern edge of the White Mountains" },

  // Marshes to Black Gate or Ithilien
  { from: "dead_marshes", to: "black_gate", distance: 75, terrainFriction: 3.0, barrierCost: 95, stealthRisk: 0.95, lore: "Approaching the terrifying iron ramparts of the Morannon" },
  { from: "dead_marshes", to: "ithilien", distance: 110, terrainFriction: 1.5, barrierCost: 20, stealthRisk: 0.35, lore: "Slipping south into the sheltered scented groves of North Ithilien" },

  // Black Gate Frontal Breach
  { from: "black_gate", to: "gorgoroth", distance: 150, terrainFriction: 2.5, barrierCost: 90, stealthRisk: 0.98, lore: "Attempting to force the gates under the watchful eye of the Dark Tower" },

  // Ithilien to Gondor or Pass
  { from: "ithilien", to: "henneth_annun", distance: 45, terrainFriction: 1.1, barrierCost: 10, stealthRisk: 0.15, lore: "Blindfolded descent into the secret cave behind the waterfall" },
  { from: "ithilien", to: "minas_tirith", distance: 95, terrainFriction: 1.3, barrierCost: 25, stealthRisk: 0.50, lore: "Crossing the River Anduin at Osgiliath" },
  { from: "henneth_annun", to: "minas_morgul", distance: 60, terrainFriction: 2.2, barrierCost: 45, stealthRisk: 0.65, lore: "Advancing into the cursed valley of the Morgulduin" },

  // Minas Morgul to Cirith Ungol to Gorgoroth
  { from: "minas_morgul", to: "cirith_ungol", distance: 55, terrainFriction: 3.8, barrierCost: 65, stealthRisk: 0.55, lore: "Climbing the perilous Straight and Winding Stairs" },
  { from: "cirith_ungol", to: "gorgoroth", distance: 80, terrainFriction: 2.8, barrierCost: 40, stealthRisk: 0.70, lore: "Creeping past the Two Watchers down into the ash of Gorgoroth" },

  // Final Ascent
  { from: "gorgoroth", to: "mount_doom", distance: 80, terrainFriction: 3.2, barrierCost: 45, stealthRisk: 0.90, lore: "Crawling along Sauron's Road to the fiery entrance of Sammath Naur" },
];

// ============================================================================
// CONTINUOUS POTENTIAL ENERGY SURFACE U(x, y)
// ============================================================================

export interface PotentialQuery {
  uVal: number;
  gradientX: number;
  gradientY: number;
  sanctuaryField: number;
  sauronField: number;
}

/**
 * Computes the continuous 2D potential energy U(x, y) across Middle-earth.
 *
 * @param x Coordinate (0 - 1000)
 * @param y Coordinate (0 - 800)
 * @param sauronVigilance Multiplier 0.0 - 1.0 scaling the Eye's repulsive barrier
 */
export function evaluatePotentialEnergy(x: number, y: number, sauronVigilance: number = 0.5): PotentialQuery {
  let uVal = 10.0; // Baseline ambient field

  // 1. Sanctuaries (Rivendell, Lothlórien, Henneth Annûn) create deep attractive wells
  const sanctuaries = [
    { x: 180, y: 140, depth: -60.0, radius: 90 }, // Rivendell
    { x: 430, y: 350, depth: -75.0, radius: 100 }, // Lothlórien
    { x: 670, y: 570, depth: -40.0, radius: 60 },  // Henneth Annûn
  ];

  let sanctuaryField = 0;
  for (const s of sanctuaries) {
    const dx = x - s.x;
    const dy = y - s.y;
    const distSq = dx * dx + dy * dy;
    const well = s.depth * Math.exp(-distSq / (2 * s.radius * s.radius));
    uVal += well;
    sanctuaryField += well;
  }

  // 2. Mountain Ridge Barriers (Misty Mountains & Ephel Dúath)
  // Misty Mountains: runs roughly from (300, 100) down to (360, 420)
  const distToMisty = Math.abs(x - (300 + (y - 100) * 0.18));
  if (y > 80 && y < 450 && distToMisty < 80) {
    // Caradhras peak barrier around y = 230
    const caradhrasBump = 50 * Math.exp(-Math.pow(y - 230, 2) / 1200);
    uVal += (40 * Math.exp(-Math.pow(distToMisty, 2) / 800)) + caradhrasBump;
  }

  // 3. Mordor Border Mountains (Ephel Dúath & Ered Lithui)
  // Surrounding Mordor: x in [650, 950], y in [420, 680]
  const distToBlackGate = Math.hypot(x - 690, y - 460);
  const blackGateBarrier = 85 * Math.exp(-Math.pow(distToBlackGate, 2) / 3600);
  uVal += blackGateBarrier;

  // 4. Sauron's Gaze Field (Centered at Barad-dûr / Mount Doom)
  // Eye of Sauron is located near Barad-dûr / Mount Doom (x: 880, y: 530)
  const distToEye = Math.hypot(x - 880, y - 530);
  const eyeRadius = 260;
  let sauronField = 0;
  if (distToEye < eyeRadius * 1.5) {
    // Repulsive barrier scaling steeply as one approaches Gorgoroth
    sauronField = (90 * sauronVigilance) * Math.exp(-Math.pow(distToEye, 2) / (2 * eyeRadius * eyeRadius));
    uVal += sauronField;
  }

  // Approximate spatial gradients
  const eps = 2.0;
  const uPlusX = evaluatePotentialRaw(x + eps, y, sauronVigilance);
  const uMinusX = evaluatePotentialRaw(x - eps, y, sauronVigilance);
  const uPlusY = evaluatePotentialRaw(x, y + eps, sauronVigilance);
  const uMinusY = evaluatePotentialRaw(x, y - eps, sauronVigilance);

  return {
    uVal: Math.round(uVal * 10) / 10,
    gradientX: (uPlusX - uMinusX) / (2 * eps),
    gradientY: (uPlusY - uMinusY) / (2 * eps),
    sanctuaryField: Math.round(sanctuaryField * 10) / 10,
    sauronField: Math.round(sauronField * 10) / 10,
  };
}

function evaluatePotentialRaw(x: number, y: number, sauronVigilance: number): number {
  let u = 10.0;
  const sList = [
    { x: 180, y: 140, depth: -60.0, r: 90 },
    { x: 430, y: 350, depth: -75.0, r: 100 },
    { x: 670, y: 570, depth: -40.0, r: 60 },
  ];
  for (const s of sList) {
    u += s.depth * Math.exp(-(Math.pow(x - s.x, 2) + Math.pow(y - s.y, 2)) / (2 * s.r * s.r));
  }
  const distToEye = Math.hypot(x - 880, y - 530);
  if (distToEye < 390) {
    u += (90 * sauronVigilance) * Math.exp(-Math.pow(distToEye, 2) / (2 * 260 * 260));
  }
  return u;
}

// ============================================================================
// PATH HAMILTONIAN & THERMODYNAMIC LANGEVIN PATH SAMPLER
// ============================================================================

export interface PathActionBreakdown {
  path: string[]; // List of waypoint IDs
  totalAction: number; // S = integral L dt (Path Hamiltonian)
  distanceCost: number;
  barrierCost: number;
  stealthRiskScore: number;
  frictionDissipation: number;
  kramersEscapeProb: number; // P_escape = exp(-Delta U / kT)
  sauronDetectionProb: number; // 0 - 100%
  sokalTauInt: number; // Sokal integrated autocorrelation time
  activeHazards: string[];
}

/**
 * Calculates the path Hamiltonian action S for a sequence of waypoints.
 */
export function calculatePathAction(
  path: string[],
  temperature: number = 1.0,
  sauronVigilance: number = 0.5
): PathActionBreakdown {
  if (path.length < 2) {
    return {
      path,
      totalAction: 0,
      distanceCost: 0,
      barrierCost: 0,
      stealthRiskScore: 0,
      frictionDissipation: 0,
      kramersEscapeProb: 1.0,
      sauronDetectionProb: 0,
      sokalTauInt: 1.0,
      activeHazards: [],
    };
  }

  let distanceCost = 0;
  let barrierCost = 0;
  let stealthRiskScore = 0;
  let frictionDissipation = 0;
  let maxBarrierEncountered = 0;
  const activeHazards: string[] = [];

  for (let i = 0; i < path.length - 1; i++) {
    const fromId = path[i];
    const toId = path[i + 1];
    const wpA = WAYPOINT_MAP[fromId];
    const wpB = WAYPOINT_MAP[toId];

    if (!wpA || !wpB) continue;

    // Euclidean distance
    const dist = Math.hypot(wpB.x - wpA.x, wpB.y - wpA.y);
    distanceCost += dist;

    // Edge lookup or synthetic fallback
    const edge = EDGES.find(
      (e) => (e.from === fromId && e.to === toId) || (e.from === toId && e.to === fromId)
    );

    const stepBarrier = edge ? edge.barrierCost : Math.max(wpA.barrierHeight, wpB.barrierHeight);
    const stepFriction = edge ? edge.terrainFriction : Math.max(wpA.friction, wpB.friction);
    const stepRisk = edge ? edge.stealthRisk : (wpA.sauronExposure + wpB.sauronExposure) / 2;

    barrierCost += stepBarrier;
    frictionDissipation += stepFriction * (dist * 0.15);
    stealthRiskScore += stepRisk * (dist * 0.25) * (1.0 + sauronVigilance * 1.5);

    if (stepBarrier > maxBarrierEncountered) {
      maxBarrierEncountered = stepBarrier;
    }

    if (fromId === "caradhras" || toId === "caradhras") activeHazards.push("Caradhras Blizzard Saddle");
    if (fromId === "mazarbul" || toId === "mazarbul") activeHazards.push("Durin's Bane Balrog Chamber");
    if (fromId === "dead_marshes" || toId === "dead_marshes") activeHazards.push("Dead Marshes Viscous Damping");
    if (fromId === "black_gate" || toId === "black_gate") activeHazards.push("Morannon Fortification Spike");
    if (fromId === "cirith_ungol" || toId === "cirith_ungol") activeHazards.push("Shelob's Lair Chokepoint");
    if (fromId === "gorgoroth" || toId === "gorgoroth") activeHazards.push("Direct Sightline of Barad-dûr");
  }

  // Deduplicate hazard tags
  const uniqueHazards = Array.from(new Set(activeHazards));

  // Kramers Escape Probability: P_escape = exp(-Delta U / (k_B * T))
  const kB = 20.0; // Scaled Boltzmann constant for narrative units
  const kramersEscapeProb = Math.min(1.0, Math.exp(-maxBarrierEncountered / (kB * Math.max(0.1, temperature))));

  // Sauron Detection Probability: Logistic response to stealth risk
  const sauronDetectionProb = Math.min(
    99,
    Math.max(1, Math.round(100 / (1 + Math.exp(-(stealthRiskScore - 80) / 25))))
  );

  // Sokal Autocorrelation Time tau_int (Critical slowing down near high barriers)
  const sokalTauInt = Math.max(1.1, Math.round((1.0 + (maxBarrierEncountered / 35.0) * Math.exp(-temperature / 2.0)) * 10) / 10);

  // Total Action: S = Distance + Barriers + Stealth Risk + Viscous Dissipation
  const totalAction = Math.round(
    distanceCost * 0.5 + barrierCost * 1.2 + stealthRiskScore * 1.5 + frictionDissipation * 1.0
  );

  return {
    path,
    totalAction,
    distanceCost: Math.round(distanceCost),
    barrierCost: Math.round(barrierCost),
    stealthRiskScore: Math.round(stealthRiskScore),
    frictionDissipation: Math.round(frictionDissipation),
    kramersEscapeProb: Math.round(kramersEscapeProb * 1000) / 1000,
    sauronDetectionProb,
    sokalTauInt,
    activeHazards: uniqueHazards,
  };
}

// ============================================================================
// CANONICAL PATH PRESETS
// ============================================================================

export interface PathPreset {
  id: string;
  title: string;
  subtitle: string;
  description: string;
  path: string[];
}

export const PATH_PRESETS: PathPreset[] = [
  {
    id: "canonical_stealth",
    title: "Canonical Fellowship Stealth Route",
    subtitle: "Moria → Lothlórien → Cirith Ungol",
    description: "The canon trajectory of Frodo and Sam: subterranean transit through Moria, sanctuary in Lórien, bogs of the Dead Marshes, and the secret stairs of Cirith Ungol.",
    path: [
      "rivendell",
      "hollin",
      "west_gate_moria",
      "mazarbul",
      "lothlorien",
      "great_river",
      "amon_hen",
      "dead_marshes",
      "ithilien",
      "henneth_annun",
      "cirith_ungol",
      "gorgoroth",
      "mount_doom",
    ],
  },
  {
    id: "black_gate_siege",
    title: "The Black Gate Frontal Siege",
    subtitle: "High Barrier Morannon Breach",
    description: "A desperate frontal attempt through the Iron Teeth of Mordor. Encounters catastrophic barrier potential (ΔU = 95.0) and virtually 100% Sauron vigilance.",
    path: [
      "rivendell",
      "hollin",
      "west_gate_moria",
      "mazarbul",
      "lothlorien",
      "great_river",
      "amon_hen",
      "dead_marshes",
      "black_gate",
      "gorgoroth",
      "mount_doom",
    ],
  },
  {
    id: "caradhras_storm",
    title: "Caradhras Mountain Storm Route",
    subtitle: "Attempting the Redhorn Pass",
    description: "Attempting the freezing blizzard of Caradhras instead of Moria. Severe thermal noise and blizzard friction before descending to the Golden Wood.",
    path: [
      "rivendell",
      "hollin",
      "caradhras",
      "lothlorien",
      "great_river",
      "amon_hen",
      "dead_marshes",
      "ithilien",
      "cirith_ungol",
      "gorgoroth",
      "mount_doom",
    ],
  },
  {
    id: "gondor_military_march",
    title: "Gondor & Rohan Defense March",
    subtitle: "Helm's Deep → Minas Tirith → Morgul Vale",
    description: "The southern military path uniting the forces of the West at Helm's Deep and Minas Tirith, confronting the terrifying Witch-king citadel of Minas Morgul.",
    path: [
      "rivendell",
      "hollin",
      "west_gate_moria",
      "mazarbul",
      "lothlorien",
      "great_river",
      "amon_hen",
      "helms_deep",
      "minas_tirith",
      "ithilien",
      "minas_morgul",
      "cirith_ungol",
      "gorgoroth",
      "mount_doom",
    ],
  },
];

// ============================================================================
// FAST LANGEVIN / MCMC PATH ANNEALER
// ============================================================================

export interface AnnealTelemetry {
  runtimeMs: number;
  initialAction: number;
  finalAction: number;
  deltaAction: number;
  stepsSampled: number;
  transitionsAccepted: number;
  acceptanceRate: number;
}

/**
 * Runs a simulated Langevin / Gibbs path relaxation from Rivendell to Mount Doom.
 */
export function annealPath(
  currentPath: string[],
  temperature: number = 1.0,
  sauronVigilance: number = 0.5,
  steps: number = 40
): { annealedPath: string[]; telemetry: AnnealTelemetry } {
  const startTime = typeof performance !== "undefined" ? performance.now() : Date.now();
  const initialAction = calculatePathAction(currentPath, temperature, sauronVigilance).totalAction;

  let bestPath = [...currentPath];
  let bestAction = initialAction;
  let path = [...currentPath];
  let transitionsAccepted = 0;

  // Build adjacency list for valid graph exploration
  const adj: Record<string, string[]> = {};
  for (const wp of WAYPOINTS) adj[wp.id] = [];
  for (const e of EDGES) {
    if (!adj[e.from].includes(e.to)) adj[e.from].push(e.to);
    if (!adj[e.to].includes(e.from)) adj[e.to].push(e.from);
  }

  // Predefined candidate sub-paths for high-level Gibbs state transitions
  const alternativeSubpaths: { at: string; target: string; via: string[] }[] = [
    { at: "hollin", target: "lothlorien", via: ["caradhras"] },
    { at: "hollin", target: "lothlorien", via: ["west_gate_moria", "mazarbul"] },
    { at: "amon_hen", target: "ithilien", via: ["dead_marshes"] },
    { at: "amon_hen", target: "ithilien", via: ["minas_tirith"] },
    { at: "amon_hen", target: "gorgoroth", via: ["dead_marshes", "black_gate"] },
    { at: "dead_marshes", target: "gorgoroth", via: ["black_gate"] },
    { at: "dead_marshes", target: "gorgoroth", via: ["ithilien", "henneth_annun", "cirith_ungol"] },
    { at: "ithilien", target: "gorgoroth", via: ["henneth_annun", "cirith_ungol"] },
    { at: "ithilien", target: "gorgoroth", via: ["minas_morgul", "cirith_ungol"] },
  ];

  for (let s = 0; s < steps; s++) {
    // Pick a candidate alternative perturbation
    const pert = alternativeSubpaths[Math.floor(Math.random() * alternativeSubpaths.length)];
    const idxA = path.indexOf(pert.at);
    const idxB = path.indexOf(pert.target);

    if (idxA !== -1 && idxB !== -1 && idxA < idxB) {
      const candidate = [
        ...path.slice(0, idxA + 1),
        ...pert.via,
        ...path.slice(idxB),
      ];

      const currentAct = calculatePathAction(path, temperature, sauronVigilance).totalAction;
      const candAct = calculatePathAction(candidate, temperature, sauronVigilance).totalAction;
      const delta = candAct - currentAct;

      // Metropolis criterion: P = min(1, exp(-delta / (kB * T)))
      const kB = 25.0;
      const acceptProb = delta <= 0 ? 1.0 : Math.exp(-delta / (kB * Math.max(0.1, temperature)));

      if (Math.random() < acceptProb) {
        path = candidate;
        transitionsAccepted++;
        if (candAct < bestAction) {
          bestAction = candAct;
          bestPath = candidate;
        }
      }
    }
  }

  // At cold temperatures, prefer the minimal energy discovered state
  const finalPath = temperature <= 1.5 ? bestPath : path;
  const finalAction = calculatePathAction(finalPath, temperature, sauronVigilance).totalAction;
  const endTime = typeof performance !== "undefined" ? performance.now() : Date.now();

  return {
    annealedPath: finalPath,
    telemetry: {
      runtimeMs: Math.round((endTime - startTime) * 100) / 100,
      initialAction,
      finalAction,
      deltaAction: finalAction - initialAction,
      stepsSampled: steps,
      transitionsAccepted,
      acceptanceRate: steps > 0 ? Math.round((transitionsAccepted / steps) * 100) : 100,
    },
  };
}
