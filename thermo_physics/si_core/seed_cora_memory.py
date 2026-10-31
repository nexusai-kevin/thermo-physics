"""Cora Episodic Memory Seeder.

Scans the thermo-physics repository documentation (README.md, docs/*.md, BENCHMARKS.md),
chunks by section headers ('##'), and encodes domain knowledge into Hopfield
attractor basins in data/cora_memory.npz.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from thermo_physics.si_core.cora_ollama_bridge import CoraThermodynamicBridge


# ============================================================================
# CANONICAL FOUNDATIONAL ATTRACTORS
# ============================================================================

CANONICAL_ATTRACTORS: List[Tuple[str, str]] = [
    (
        "Station_1_to_9_Foundations",
        "Foundational thermodynamic computing stations 1 to 9: 1D Ising chain spin alignment, "
        "2D checkerboard phase transition with Onsager criticality at beta=0.4407, NP-hard Max-Cut "
        "graph partitioning with chromatic block Gibbs sampling, multistate Potts K=4 color party, "
        "clamped DTM image denoiser, continuous Hopfield and RBM associative memory, Markowitz QUBO "
        "portfolio optimizer, Kramers double-well continuous barrier escape with Langevin dynamics, "
        "and Held-Karp NP-complete 10-city TSP."
    ),
    (
        "Station_10_to_14_Advanced_Solvers",
        "Advanced thermodynamic solvers stations 10 to 14: Sherrington-Kirkpatrick (SK) frustrated "
        "spin glass (N=24) with replica symmetry breaking, RNA conformational secondary structure "
        "folding ensembles with Turner nearest-neighbor free energies, Power Grid SCUC Unit Commitment "
        "QUBO balancing generator ramp rates, mRNA GLP-1 codon optimization via discrete Potts EBM "
        "resolving ribosome stalling hairpins, and 24-residue antiparallel beta-hairpin protein side-chain "
        "rotamer packing eliminating Lennard-Jones steric clashes."
    ),
    (
        "Station_15_SI_Cognitive_Core",
        "Station 15 Synthetic Intelligence (SI) 4-Organ Cognitive Core: Non-von-Neumann autonomous cognitive "
        "cycle executing deterministic ground-state relaxation in under 15ms SLA. Organ 1: Continuous Hopfield "
        "Associative Memory in 128GB DDR5 RAM with exponential attractor capacity. Organ 2: Kramers Double-Well "
        "Curiosity & Drive modulating thermal bath temperature T and escape dynamics. Organ 3: N=6, K=4 "
        "Categorical Potts Goal Arbiter equilibrating competing agent drives (VERIFY, FAST_HEURISTIC, "
        "DEEP_REASONING, AUTONOMOUS_EXEC). Organ 4: RTX 5080 Metabolic Homeostasis QUBO scheduler governing "
        "16GB VRAM and dispatching model tiers."
    ),
    (
        "Nexus_Thermodynamic_Dispatcher",
        "Nexus Thermodynamic Agent Burst Dispatcher: Sub-5ms combinatorial scheduler modeling multi-model "
        "burst requests across 40 tasks. Solves non-linear TPU/GPU quota limits, TPM/RPM rate headroom, "
        "and provider cost trade-offs, delivering 36% API cost reduction and sub-5ms routing with zero "
        "constraint violations."
    ),
    (
        "LoL_Draft_And_Middle_Earth",
        "Interactive physics application showcases: League of Legends 5v5 competitive draft solver "
        "formulated as an N=10 categorical Potts spin model eliminating role clashes while maximizing "
        "team wombo combos; and Middle-earth Fellowship path routing engine navigating 18 waypoints via "
        "continuous Kramers barrier escape across Mordor surveillance and mountain pass energy landscapes."
    ),
    (
        "Local_Workstation_Architecture",
        "Local hardware allocation architecture: NVIDIA GeForce RTX 5080 (16GB GDDR7 VRAM) and 128GB DDR5 "
        "System RAM. GPU VRAM runs local Ollama semantic transducer (nomic-embed-text for 768-D spin vectors, "
        "Qwen2.5/Llama3.1 for grounded language transduction). 128GB system RAM hosts massive episodic Hopfield "
        "associative memory tensors for millions of state attractors without touching GPU memory limits."
    ),
]


def extract_markdown_sections(file_path: Path) -> List[Tuple[str, str]]:
    """Chunks markdown file by '## ' headers into clean label/content tuples."""
    if not file_path.exists():
        return []

    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception:
        return []

    base_name = file_path.stem.replace(".", "_")
    sections: List[Tuple[str, str]] = []

    # Match '## Section Title'
    pattern = r"(?:^|\n)##\s+([^\n]+)\n(.*?)(?=\n##\s+|$)"
    matches = re.findall(pattern, content, flags=re.DOTALL)

    for header, body in matches:
        clean_title = re.sub(r"[^\w\s-]", "", header).strip().replace(" ", "_")
        label = f"{base_name}_{clean_title}"[:50]
        # Clean markdown code blocks, links, extra whitespace
        clean_body = re.sub(r"```.*?```", "", body, flags=re.DOTALL)
        clean_body = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", clean_body)
        clean_body = re.sub(r"\s+", " ", clean_body).strip()
        if len(clean_body) > 60:
            sections.append((label, clean_body[:1000]))

    return sections


def seed_cora_memory(
    memory_path: str = "data/cora_memory.npz",
    base_dir: Optional[str] = None,
) -> int:
    """Populates memory archive with canonical attractors and markdown documentation sections."""
    repo_root = Path(base_dir) if base_dir else Path(__file__).resolve().parent.parent.parent
    bridge = CoraThermodynamicBridge(memory_path=memory_path, embedding_dim=768)

    existing_labels = set(bridge.memory.labels)
    added_count = 0

    # 1. Ingest Canonical Attractors
    for label, text in CANONICAL_ATTRACTORS:
        if label not in existing_labels:
            vec = bridge.embed_sensory_input(text)
            bridge.memory.store(vec, label)
            existing_labels.add(label)
            added_count += 1

    # 2. Ingest Markdown Documentation
    doc_paths = [
        repo_root / "README.md",
        repo_root / "BENCHMARKS.md",
    ]
    docs_dir = repo_root / "docs"
    if docs_dir.exists():
        doc_paths.extend(list(docs_dir.glob("*.md")))

    for doc_file in doc_paths:
        sections = extract_markdown_sections(doc_file)
        for label, text in sections:
            if label not in existing_labels:
                vec = bridge.embed_sensory_input(text)
                bridge.memory.store(vec, label)
                existing_labels.add(label)
                added_count += 1

    # Save consolidated memory archive to disk
    bridge.memory.save(memory_path)
    return len(bridge.memory.patterns)


if __name__ == "__main__":
    print("=" * 76)
    print("  SEEDING CORA EPISODIC MEMORY (HOPFIELD ATTRACTOR BASIN INGESTION)")
    print("=" * 76)

    target_path = "data/cora_memory.npz"
    total = seed_cora_memory(memory_path=target_path)
    print(f"\n[+] Successfully consolidated {total} episodic attractors into '{target_path}'.")
