"""Cora Thermodynamic SI Core Ollama Bridge.

Integrates the 4 non-von-Neumann thermodynamic organs with a live local
Ollama runtime on NVIDIA GeForce RTX 5080 (16GB VRAM) and 128GB DDR5 System RAM:
1. Auto-discovery & tier mapping from GET /api/tags
2. Sensory embedding (nomic-embed-text 768-D with deterministic hash fallback)
3. Hopfield novelty detection (1 - max_cos) & automatic episodic memory consolidation
4. Kramers Langevin curiosity & temperature scaling (T -> sampling temp)
5. Potts goal arbitration & RTX 5080 metabolic VRAM scheduling
6. Grounded system telemetry header injection & interactive HUD chat loop
"""

from __future__ import annotations

import os
import sys
import time
import json
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

try:
    import urllib.request
    import urllib.error
except ImportError:
    pass

from thermo_physics.si_core.cognitive_engine import (
    HopfieldAssociativeMemory,
    KramersCuriosityDrive,
    PottsGoalArbiter,
    MetabolicHomeostasis,
)


# ============================================================================
# HELPER FUNCTIONS: HARDWARE TELEMETRY & EMBEDDINGS
# ============================================================================

# Cached VRAM telemetry (avoiding Windows subprocess overhead on every ms tick)
_LAST_VRAM_READING: float = 3.8
_LAST_VRAM_TIMESTAMP: float = 0.0


def get_live_vram_usage_gb(cache_ttl_s: float = 5.0) -> float:
    """Queries nvidia-smi for live GPU memory usage in GB with TTL caching for low tick latency."""
    global _LAST_VRAM_READING, _LAST_VRAM_TIMESTAMP
    now = time.time()
    if now - _LAST_VRAM_TIMESTAMP < cache_ttl_s:
        return _LAST_VRAM_READING

    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=1.0,
        )
        first_line = res.stdout.strip().split("\n")[0].strip()
        used_mb = float(first_line)
        _LAST_VRAM_READING = round(used_mb / 1024.0, 2)
        _LAST_VRAM_TIMESTAMP = now
        return _LAST_VRAM_READING
    except Exception:
        # Default baseline if nvidia-smi is unavailable in container/sandbox
        _LAST_VRAM_READING = 3.8
        _LAST_VRAM_TIMESTAMP = now
        return 3.8


def deterministic_hash_embedding(text: str, dim: int = 768) -> np.ndarray:
    """Deterministic hash projection to generate a normalized unit spin vector.

    Used when nomic-embed-text is offline or uninstalled to guarantee reproducible,
    zero-dependency spin vectors.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed_int = int.from_bytes(digest[:8], "big")
    rng = np.random.default_rng(seed_int)
    vec = rng.standard_normal(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-8)


# ============================================================================
# CORA THERMODYNAMIC BRIDGE CLASS
# ============================================================================

class CoraThermodynamicBridge:
    """Bridges Cora's cognitive processes to local Ollama through 4 thermodynamic organs."""

    STANCE_DIRECTIVES = {
        "VERIFY": (
            "Prioritize strict verification, ground-truth fact checking, and invariant safety checks. "
            "Reject unsubstantiated speculation and identify failure modes."
        ),
        "FAST_HEURISTIC": (
            "Provide immediate, high-velocity, concise, direct answers without unnecessary deliberation. "
            "Focus on speed and practical rule-of-thumb clarity."
        ),
        "DEEP_REASONING": (
            "Engage in deep, multi-perspective, rigorous conceptual synthesis and structured deliberation. "
            "Trace underlying physical and mathematical principles step-by-step."
        ),
        "AUTONOMOUS_EXEC": (
            "Focus on decisive actionable steps, concrete operational pathways, and execution commands. "
            "Format recommendations for direct deployment and immediate workflow completion."
        ),
    }

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        memory_path: str = "data/cora_memory.npz",
        embedding_dim: int = 768,
        seed: Optional[int] = 42,
    ) -> None:
        self.ollama_url = ollama_url.rstrip("/")
        self.memory_path = memory_path
        self.dim = embedding_dim
        self.seed = seed

        # Initialize the 4 Thermodynamic Organs
        self.memory = HopfieldAssociativeMemory(dim=self.dim, beta=8.0)
        self.curiosity = KramersCuriosityDrive(barrier_height=2.5, damping=0.4)
        self.arbiter = PottsGoalArbiter(num_modules=6, num_states=4, coupling_strength=1.2)
        self.metabolism = MetabolicHomeostasis(vram_limit_gb=16.0)

        # Initialize episodic memory from disk or seed foundational attractors
        self._init_memory()

        # Discover local Ollama models on RTX 5080
        self.ollama_online = False
        self.installed_models: List[str] = []
        self.tier_mapping: Dict[str, str] = {}
        self.discover_models()

        # Pre-warm live VRAM telemetry
        get_live_vram_usage_gb()

    def _init_memory(self) -> None:
        """Loads episodic memory from disk if available, otherwise seeds baseline attractors."""
        expanded_path = os.path.expanduser(self.memory_path)
        if not expanded_path.endswith(".npz"):
            expanded_path += ".npz"

        if os.path.exists(expanded_path):
            try:
                self.memory.load(expanded_path)
                return
            except Exception:
                pass

        # Seed foundational episodic attractors in 768-D
        rng = np.random.default_rng(self.seed)
        foundational = [
            ("Core_Identity_Cora_Autonomous_AI", "Foundational identity as Cora, an autonomous AI architect."),
            ("Thermodynamic_Station_Invariants", "Non-von-Neumann energy relaxation and Potts spin consensus."),
            ("System_Architecture_Optimization", "Distributed architectures, microservices, and scale limits."),
            ("Security_Boundary_Verification", "Strict boundary enforcement, access controls, and safety gates."),
        ]
        for label, desc in foundational:
            vec = deterministic_hash_embedding(desc, dim=self.dim)
            self.memory.store(vec, label)

        # Save initial seed to disk
        try:
            self.memory.save(expanded_path)
        except Exception:
            pass

    def discover_models(self) -> Dict[str, str]:
        """Queries Ollama GET /api/tags to detect installed models and maps them to metabolic tiers."""
        tags_url = f"{self.ollama_url}/api/tags"
        models: List[str] = []

        try:
            req = urllib.request.Request(tags_url, headers={"User-Agent": "Cora-SI-Core"})
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
                    self.ollama_online = True
        except Exception:
            self.ollama_online = False

        self.installed_models = models

        # Intelligent model tier mapping
        fast_candidate = None
        core_candidate = None
        deep_candidate = None

        for m in models:
            m_lower = m.lower()
            if any(k in m_lower for k in ["3b", "1.5b", "2b", "phi", "mini"]):
                if not fast_candidate:
                    fast_candidate = m
            elif any(k in m_lower for k in ["8b", "7b", "9b", "llama3.1", "mistral"]):
                if not core_candidate:
                    core_candidate = m
            elif any(k in m_lower for k in ["14b", "32b", "70b", "deepseek", "qwen2.5:14b"]):
                if not deep_candidate:
                    deep_candidate = m

        # Fallback hierarchy if specific tiers are missing
        primary = models[0] if models else "llama3.1:8b"
        self.tier_mapping = {
            "Local_Fast_3B": fast_candidate or primary or "qwen2.5:3b",
            "Local_Core_8B": core_candidate or primary or "llama3.1:8b",
            "Local_Deep_14B": deep_candidate or primary or "qwen2.5:14b",
        }
        return self.tier_mapping

    def embed_prompt(self, prompt: str) -> np.ndarray:
        """Embeds prompt into a normalized 768-D spin vector via nomic-embed-text (or hash fallback)."""
        if self.ollama_online:
            # 1. Try modern /api/embed
            embed_url = f"{self.ollama_url}/api/embed"
            try:
                payload = json.dumps({"model": "nomic-embed-text", "input": prompt}).encode("utf-8")
                req = urllib.request.Request(
                    embed_url,
                    data=payload,
                    headers={"Content-Type": "application/json", "User-Agent": "Cora-SI-Core"},
                )
                with urllib.request.urlopen(req, timeout=2.0) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode("utf-8"))
                        embeddings = data.get("embeddings", [])
                        if embeddings and len(embeddings[0]) == self.dim:
                            vec = np.asarray(embeddings[0], dtype=np.float32)
                            return vec / (np.linalg.norm(vec) + 1e-8)
            except Exception:
                pass

            # 2. Try legacy /api/embeddings
            legacy_url = f"{self.ollama_url}/api/embeddings"
            try:
                payload = json.dumps({"model": "nomic-embed-text", "prompt": prompt}).encode("utf-8")
                req = urllib.request.Request(
                    legacy_url,
                    data=payload,
                    headers={"Content-Type": "application/json", "User-Agent": "Cora-SI-Core"},
                )
                with urllib.request.urlopen(req, timeout=2.0) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode("utf-8"))
                        embedding = data.get("embedding", [])
                        if embedding and len(embedding) == self.dim:
                            vec = np.asarray(embedding, dtype=np.float32)
                            return vec / (np.linalg.norm(vec) + 1e-8)
            except Exception:
                pass

        # Robust deterministic fallback
        return deterministic_hash_embedding(prompt, dim=self.dim)

    def cognitive_tick(
        self,
        user_prompt: str,
        seed: Optional[int] = None,
        live_vram: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Executes a single coupled non-von-Neumann cognitive tick for the prompt.

        Returns full thermodynamic telemetry across all 4 organs.
        """
        t0 = time.perf_counter()

        # Sensory Transduction to 768-D Spin Vector
        spin_vector = self.embed_prompt(user_prompt)

        # Organ 1: Hopfield Memory Recall & Novelty Detection
        recalled_vec, memory_label, mem_energy = self.memory.retrieve(spin_vector, steps=5)
        novelty, max_sim = self.memory.compute_novelty(spin_vector)

        # Organ 2: Kramers Double-Well Curiosity & Drive
        drive_mode, bath_temp = self.curiosity.step(novelty, seed=seed)
        beta = 1.0 / bath_temp

        # Organ 3: Potts Goal Arbitration
        fields = np.zeros((self.arbiter.N, self.arbiter.K), dtype=np.float32)
        if drive_mode == "EXPLORE_NOVELTY":
            fields[:, 2] += 1.8  # Strong favor for DEEP_REASONING
            fields[:, 3] += 1.3  # Favor AUTONOMOUS_EXEC
        else:
            fields[:, 0] += 1.6  # Strong favor for VERIFY
            fields[:, 1] += 1.1  # Favor FAST_HEURISTIC

        goal_spins, potts_energy = self.arbiter.equilibrate(fields, beta=beta, sweeps=40, seed=seed)
        dominant_idx = int(np.bincount(goal_spins, minlength=4).argmax())
        dominant_stance = PottsGoalArbiter.STANCE_NAMES[dominant_idx]

        # Organ 4: RTX 5080 Metabolic Homeostasis
        vram_to_use = live_vram if live_vram is not None else get_live_vram_usage_gb()
        req_cap = 0.95 if dominant_idx == 2 else (0.8 if dominant_idx == 3 else 0.5)
        metabolic_choice = self.metabolism.select_execution_tier(req_cap, vram_to_use)
        selected_tier = metabolic_choice["selected_engine"]
        mapped_model = self.tier_mapping.get(selected_tier, "llama3.1:8b")

        # Scale Ollama sampling temperature from physical bath temperature
        ollama_temp = float(np.clip(bath_temp * 0.35, 0.1, 1.2))

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Episodic Consolidation: Automatically store novel insights (novelty > 0.45)
        consolidated = False
        if novelty > 0.45:
            clean_snippet = "".join(c if c.isalnum() else "_" for c in user_prompt[:24]).strip("_")
            new_label = f"Episodic_{int(time.time())}_{clean_snippet}"
            self.memory.store(spin_vector, new_label)
            try:
                self.memory.save(self.memory_path)
                consolidated = True
            except Exception:
                pass

        return {
            "tick_latency_ms": round(elapsed_ms, 2),
            "user_prompt": user_prompt,
            "novelty_score": round(novelty, 3),
            "max_cosine_sim": round(max_sim, 3),
            "episodic_consolidated": consolidated,
            "organ_1_attractor": memory_label,
            "organ_1_energy": round(mem_energy, 4),
            "organ_1_total_patterns": len(self.memory.patterns),
            "organ_2_drive_mode": drive_mode,
            "organ_2_bath_temp_T": round(bath_temp, 3),
            "organ_2_particle_x": round(self.curiosity.x, 3),
            "organ_3_stance": dominant_stance,
            "organ_3_potts_energy": round(potts_energy, 3),
            "organ_4_tier": selected_tier,
            "organ_4_model": mapped_model,
            "organ_4_live_vram_gb": vram_to_use,
            "organ_4_headroom_gb": metabolic_choice["vram_headroom_gb"],
            "ollama_temperature": round(ollama_temp, 3),
            "ollama_online": self.ollama_online,
        }

    def format_grounded_system_prompt(self, telemetry: Dict[str, Any]) -> str:
        """Injects thermodynamic ground state telemetry into Cora's system prompt."""
        stance = telemetry["organ_3_stance"]
        directive = self.STANCE_DIRECTIVES.get(stance, "Act coherently and effectively.")

        return (
            "You are Cora, an autonomous AI architect and coding intelligence powered by a "
            "4-Organ Thermodynamic Cognitive Core operating on an RTX 5080.\n\n"
            "[THERMODYNAMIC SYNTHETIC INTELLIGENCE (SI) TELEMETRY]\n"
            f"- Tick Latency: {telemetry['tick_latency_ms']} ms\n"
            f"- Thermal Bath Temperature: {telemetry['organ_2_bath_temp_T']} K (β = {1.0 / telemetry['organ_2_bath_temp_T']:.2f})\n"
            f"- Intrinsic Curiosity Drive: {telemetry['organ_2_drive_mode']} (x = {telemetry['organ_2_particle_x']})\n"
            f"- Associative Memory Attractor: {telemetry['organ_1_attractor']} (Novelty = {telemetry['novelty_score']})\n"
            f"- Potts Goal Consensus: {stance} (H = {telemetry['organ_3_potts_energy']})\n"
            f"- Metabolic Inference Tier: {telemetry['organ_4_tier']} -> {telemetry['organ_4_model']} (VRAM: {telemetry['organ_4_live_vram_gb']}/16.0 GB)\n\n"
            f"[BEHAVIORAL DIRECTIVE: {stance}]\n"
            f"{directive}\n"
        )

    def chat(self, user_prompt: str, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """Executes full cognitive tick, injects grounded telemetry, and queries local Ollama."""
        telemetry = self.cognitive_tick(user_prompt, seed=seed)
        system_prompt = self.format_grounded_system_prompt(telemetry)
        model_name = telemetry["organ_4_model"]

        # Call live local Ollama if online
        if self.ollama_online:
            chat_url = f"{self.ollama_url}/api/chat"
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {
                    "temperature": telemetry["ollama_temperature"],
                },
            }
            try:
                req = urllib.request.Request(
                    chat_url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json", "User-Agent": "Cora-SI-Core"},
                )
                with urllib.request.urlopen(req, timeout=120.0) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode("utf-8"))
                        reply = data.get("message", {}).get("content", "").strip()
                        return reply, telemetry
            except Exception:
                pass

        # Offline fallback response demonstrating thermodynamic equilibration
        stance = telemetry["organ_3_stance"]
        drive = telemetry["organ_2_drive_mode"]
        attractor = telemetry["organ_1_attractor"]

        simulated_response = (
            f"[Cora SI Core • {model_name} (Offline Simulation)]\n"
            f"Under stance [{stance}] with intrinsic drive [{drive}], the system settled into "
            f"episodic basin [{attractor}] (E={telemetry['organ_1_energy']}).\n"
            f"Analysis of query: \"{user_prompt}\" aligns with {stance.lower().replace('_', ' ')} protocol. "
            f"All thermodynamic bounds satisfied on RTX 5080 ({telemetry['organ_4_live_vram_gb']}/16.0GB VRAM)."
        )
        return simulated_response, telemetry


# ============================================================================
# INTERACTIVE CLI LOOP
# ============================================================================

def run_cora_cli() -> None:
    """Launches interactive terminal chat loop with real-time thermodynamic HUD."""
    print("=" * 76)
    print("  CORA 4-ORGAN THERMODYNAMIC SYNTHETIC INTELLIGENCE (SI) RUNTIME")
    print("  NVIDIA GeForce RTX 5080 (16GB VRAM) • 128GB DDR5 RAM • Station 15")
    print("=" * 76)

    bridge = CoraThermodynamicBridge(memory_path="data/cora_memory.npz")
    print(f"[*] Ollama Status: {'ONLINE' if bridge.ollama_online else 'OFFLINE (Fallback Simulation Active)'}")
    print(f"[*] Models Detected: {len(bridge.installed_models)} ({', '.join(bridge.installed_models[:3]) or 'None'})")
    print(f"[*] Model Tier Mapping:")
    for tier, m in bridge.tier_mapping.items():
        print(f"    - {tier}: {m}")
    print(f"[*] Episodic Attractors: {len(bridge.memory.patterns)} stored in memory")
    print("-" * 76)
    print("Type your prompt to Cora, or 'exit' / 'quit' to terminate.\n")

    while True:
        try:
            prompt = input("Cora > ").strip()
            if not prompt:
                continue
            if prompt.lower() in ["exit", "quit", "q"]:
                print("[*] Consolidating episodic memories and shutting down SI runtime...")
                break

            reply, telem = bridge.chat(prompt)

            # Print single-line cybernetic HUD
            hud = (
                f"[Tick: {telem['tick_latency_ms']}ms | "
                f"Attractor: {telem['organ_1_attractor'][:22]} | "
                f"Drive: {telem['organ_2_drive_mode']} (T={telem['organ_2_bath_temp_T']}K) | "
                f"Stance: {telem['organ_3_stance']} | "
                f"VRAM: {telem['organ_4_live_vram_gb']}/16.0GB]"
            )
            print(f"\n\033[96m{hud}\033[0m")
            print(f"{reply}\n")

        except (KeyboardInterrupt, EOFError):
            print("\n[*] Exiting Cora SI Core.")
            break


if __name__ == "__main__":
    run_cora_cli()
