"""Cora Thermodynamic SI Core FastAPI & WebSocket Server.

Exposes the 4-organ thermodynamic cognitive runtime over REST, SSE, and WebSocket:
- GET /health: Hardware status, live VRAM, memory count, Ollama detection
- GET /telemetry/state: Resting telemetry state across all 4 organs
- POST /memory/store: Stores and persists new episodic attractors to disk
- POST /memory/seed: Triggers knowledge-base ingestion on demand
- POST /chat/sync: Synchronous grounded inference with full telemetry
- POST /chat/stream: SSE text/event-stream with immediate <10ms telemetry & token chunks
- WS /ws/cora: Full-duplex WebSocket session for continuous telemetry & streaming tokens
"""

from __future__ import annotations

import os
import time
import json
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from thermo_physics.si_core.cora_ollama_bridge import (
    CoraThermodynamicBridge,
    get_live_vram_usage_gb,
)
from thermo_physics.si_core.cognitive_engine import PottsGoalArbiter
from thermo_physics.si_core.seed_cora_memory import seed_cora_memory


# ============================================================================
# PYDANTIC SCHEMAS
# ============================================================================

class StoreMemoryRequest(BaseModel):
    text: str = Field(..., description="Sensory text to transduce and store in Hopfield memory")
    label: Optional[str] = Field(None, description="Optional custom attractor label")


class ChatRequest(BaseModel):
    prompt: str = Field(..., description="User query or system instruction")
    seed: Optional[int] = Field(None, description="Optional PRNG seed for deterministic relaxation")


# ============================================================================
# FASTAPI APPLICATION SETUP
# ============================================================================

app = FastAPI(
    title="Cora Thermodynamic SI Core Server",
    description="Asynchronous 4-Organ Thermodynamic Cognitive Runtime for Cora (RTX 5080 + 128GB DDR5)",
    version="0.1.0",
)

# Enable CORS for nexus-site and external clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared singleton runtime bridge
bridge = CoraThermodynamicBridge(memory_path="data/cora_memory.npz", embedding_dim=768)


# ============================================================================
# REST ENDPOINTS
# ============================================================================

@app.get("/health")
async def get_health() -> Dict[str, Any]:
    """Returns runtime health, RTX 5080 VRAM usage, memory capacity, and Ollama status."""
    live_vram = get_live_vram_usage_gb()
    return {
        "status": "online",
        "vram_used_gb": live_vram,
        "memory_count": len(bridge.memory.patterns),
        "ollama_online": bridge.ollama_online,
        "models": bridge.installed_models,
        "tier_mapping": bridge.tier_mapping,
        "device": "NVIDIA GeForce RTX 5080 (16GB GDDR7) + 128GB DDR5",
    }


@app.get("/telemetry/state")
async def get_telemetry_state() -> Dict[str, Any]:
    """Returns resting thermodynamic telemetry state across all 4 coupled organs."""
    live_vram = get_live_vram_usage_gb()
    top_labels = bridge.memory.labels[:12] if bridge.memory.labels else []

    return {
        "organ_1_memory_count": len(bridge.memory.patterns),
        "organ_1_top_labels": top_labels,
        "organ_2_kramers_x": round(bridge.curiosity.x, 3),
        "organ_2_drive_mode": "EXPLORE_NOVELTY" if bridge.curiosity.x > 0 else "EXPLOIT_KNOWN",
        "organ_2_bath_temp_T": round(bridge.curiosity.step(0.0)[1], 3),
        "organ_3_modules": bridge.arbiter.N,
        "organ_3_stances": PottsGoalArbiter.STANCE_NAMES,
        "organ_4_vram_limit_gb": bridge.metabolism.vram_limit,
        "organ_4_live_vram_gb": live_vram,
        "organ_4_headroom_gb": round(max(0.0, bridge.metabolism.vram_limit - live_vram), 2),
    }


@app.post("/memory/store")
async def store_memory(req: StoreMemoryRequest) -> Dict[str, Any]:
    """Stores a sensory vector into Hopfield associative memory and persists to disk."""
    label = req.label
    if not label:
        clean_text = "".join(c if c.isalnum() else "_" for c in req.text[:24]).strip("_")
        label = f"Episodic_{int(time.time())}_{clean_text}"

    vec = bridge.embed_sensory_input(req.text)
    bridge.memory.store(vec, label)
    try:
        bridge.memory.save(bridge.memory_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to persist memory: {e}")

    return {
        "status": "stored",
        "label": label,
        "memory_count": len(bridge.memory.patterns),
    }


@app.post("/memory/seed")
async def seed_memory() -> Dict[str, Any]:
    """Triggers repository documentation ingestion into Hopfield memory on demand."""
    try:
        total = seed_cora_memory(memory_path=bridge.memory_path)
        bridge.memory.load(bridge.memory_path)
        return {
            "status": "seeded",
            "memory_count": total,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to seed memory: {e}")


@app.post("/chat/sync")
async def chat_sync(req: ChatRequest) -> Dict[str, Any]:
    """Synchronous chat endpoint executing full 4-organ tick and grounded LLM completion."""
    reply, telemetry = bridge.chat(req.prompt, seed=req.seed)
    return {
        "reply": reply,
        "telemetry": telemetry,
    }


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """SSE streaming endpoint yielding immediate <10ms telemetry and streaming token chunks."""
    async def sse_event_generator():
        async for event in bridge.stream_step(req.prompt, seed=req.seed):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(sse_event_generator(), media_type="text/event-stream")


# ============================================================================
# WEBSOCKET FULL-DUPLEX ENDPOINT
# ============================================================================

@app.websocket("/ws/cora")
async def websocket_cora(websocket: WebSocket) -> None:
    """Full-duplex WebSocket session streaming telemetry frames and token chunks."""
    await websocket.accept()
    try:
        while True:
            raw_msg = await websocket.receive_text()
            prompt = raw_msg
            seed = None
            try:
                parsed = json.loads(raw_msg)
                if isinstance(parsed, dict):
                    prompt = parsed.get("prompt", raw_msg)
                    seed = parsed.get("seed")
            except Exception:
                pass

            async for event in bridge.stream_step(prompt, seed=seed):
                await websocket.send_text(json.dumps(event))

    except WebSocketDisconnect:
        pass


# ============================================================================
# SERVER RUNNER
# ============================================================================

def run_server() -> None:
    """Starts the Cora Thermodynamic SI Core FastAPI server on port 8090."""
    import uvicorn
    print("[*] Starting Cora Thermodynamic SI Core Server on http://127.0.0.1:8090 ...")
    uvicorn.run("thermo_physics.si_core.cora_server:app", host="127.0.0.1", port=8090, reload=False)


if __name__ == "__main__":
    run_server()
