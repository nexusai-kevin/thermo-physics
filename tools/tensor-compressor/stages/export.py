"""Stage 4: Export compressed Hugging Face checkpoint + GGUF/Ollama & vLLM scripts."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import torch.nn as nn


_RECONSTRUCT_SCRIPT = '''"""Reconstruct dense standard Hugging Face weights from factorized checkpoint for GGUF/vLLM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoTokenizer


def reconstruct(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((src_dir / "compression_manifest.json").read_text(encoding="utf-8"))

    state: dict[str, torch.Tensor] = {}
    single = src_dir / "model.safetensors"
    index = src_dir / "model.safetensors.index.json"
    if single.exists():
        state = load_file(str(single), device="cpu")
    elif index.exists():
        idx = json.loads(index.read_text(encoding="utf-8"))
        for sf in sorted(set(idx.get("weight_map", {}).values())):
            state.update(load_file(str(src_dir / sf), device="cpu"))
    else:
        raise FileNotFoundError(f"No safetensors found in {src_dir}")

    for layer in manifest.get("layers", []):
        for _, spec in layer.get("projections", {}).items():
            if spec.get("status") != "compressed":
                continue
            prefix = spec["full_name"]
            if spec.get("kind") == "low_rank":
                u_w = state.pop(f"{prefix}.u_proj.weight")
                v_w = state.pop(f"{prefix}.v_proj.weight")
                state[f"{prefix}.weight"] = (u_w.float() @ v_w.float()).to(u_w.dtype)
                if f"{prefix}.u_proj.bias" in state:
                    state[f"{prefix}.bias"] = state.pop(f"{prefix}.u_proj.bias")
            elif spec.get("kind") == "mpo":
                c1 = state.pop(f"{prefix}.core1")
                c2 = state.pop(f"{prefix}.core2")
                w = torch.einsum("oik,kpq->opiq", c1.float(), c2.float())
                state[f"{prefix}.weight"] = w.reshape(spec["out_features"], spec["in_features"]).to(c1.dtype)

    save_file(state, str(dst_dir / "model.safetensors"))
    cfg = AutoConfig.from_pretrained(str(src_dir))
    cfg.save_pretrained(str(dst_dir))
    tok = AutoTokenizer.from_pretrained(str(src_dir))
    tok.save_pretrained(str(dst_dir))
    print(f"Reconstructed standard HF checkpoint at {dst_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--dst", type=Path, required=True)
    args = p.parse_args()
    reconstruct(args.src, args.dst)
'''


def _dir_size_gb(path: Path) -> float:
    total_bytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total_bytes / (1024**3), 4)


def _slugify(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", name).strip("-").lower()


def run_export(
    model: nn.Module,
    tokenizer: Any,
    manifest: dict[str, Any],
    model_id: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Save compressed HF checkpoint and generate GGUF/Ollama and vLLM deploy scripts."""
    t0 = time.perf_counter()
    out_path = Path(output_dir)
    hf_dir = out_path / "compressed_hf"
    deploy_dir = out_path / "deploy"
    hf_dir.mkdir(parents=True, exist_ok=True)
    deploy_dir.mkdir(parents=True, exist_ok=True)

    # Save model & tokenizer in safetensors format
    model.to("cpu")
    model.save_pretrained(str(hf_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(hf_dir))

    # Write compression manifest & enrich config.json
    manifest_path = hf_dir / "compression_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    cfg_path = hf_dir / "config.json"
    if cfg_path.exists():
        cfg_json = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg_json["vertical_tensor_compression"] = {
            "source_model": model_id,
            "method": manifest["method"],
            "ratio_requested": manifest["ratio_requested"],
            "align_multiple": manifest.get("align_multiple", 64),
            "allocation_mode": manifest.get("allocation_mode", "energy_optimal"),
            "actual_total_reduction": manifest["actual_total_reduction"],
            "params_before": manifest["params_before"]["total"],
            "params_after": manifest["params_after"]["total"],
        }
        cfg_path.write_text(json.dumps(cfg_json, indent=2), encoding="utf-8")

    (hf_dir / "reconstruct_standard_hf.py").write_text(_RECONSTRUCT_SCRIPT, encoding="utf-8")

    slug = _slugify(model_id)
    est_b = manifest["params_after"]["total"] / 1e9
    q4_gb = round(est_b * 0.58 + 0.5, 2)

    # Ready-to-run GGUF / Ollama export scripts (Bash + PowerShell)
    gguf_sh = f"""#!/usr/bin/env bash
set -euo pipefail

# 1. Reconstruct standard HF tensor keys for llama.cpp conversion
python "{hf_dir.as_posix()}/reconstruct_standard_hf.py" \\
  --src "{hf_dir.as_posix()}" \\
  --dst "{deploy_dir.as_posix()}/standard_hf_tmp"

# 2. Convert to F16 GGUF via llama.cpp
python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" \\
  "{deploy_dir.as_posix()}/standard_hf_tmp" \\
  --outfile "{deploy_dir.as_posix()}/{slug}-compressed-f16.gguf" \\
  --outtype f16

# 3. Quantize to 4-bit Q4_K_M (~{q4_gb} GB VRAM)
"$LLAMA_CPP_DIR/build/bin/llama-quantize" \\
  "{deploy_dir.as_posix()}/{slug}-compressed-f16.gguf" \\
  "{deploy_dir.as_posix()}/{slug}-compressed-q4_k_m.gguf" \\
  Q4_K_M

# 4. Register in Ollama
ollama create "{slug}-compressed" -f "{deploy_dir.as_posix()}/Modelfile"
echo "Ready: ollama run {slug}-compressed"
"""

    gguf_ps1 = f"""$ErrorActionPreference = "Stop"
# Set $env:LLAMA_CPP_DIR to your local llama.cpp clone before running
python "{hf_dir / 'reconstruct_standard_hf.py'}" --src "{hf_dir}" --dst "{deploy_dir / 'standard_hf_tmp'}"
python "$env:LLAMA_CPP_DIR\\convert_hf_to_gguf.py" "{deploy_dir / 'standard_hf_tmp'}" --outfile "{deploy_dir / (slug + '-compressed-f16.gguf')}" --outtype f16
& "$env:LLAMA_CPP_DIR\\build\\bin\\Release\\llama-quantize.exe" "{deploy_dir / (slug + '-compressed-f16.gguf')}" "{deploy_dir / (slug + '-compressed-q4_k_m.gguf')}" Q4_K_M
ollama create "{slug}-compressed" -f "{deploy_dir / 'Modelfile'}"
"""

    modelfile = f"""FROM ./{slug}-compressed-q4_k_m.gguf
PARAMETER temperature 0.1
PARAMETER num_ctx 4096
SYSTEM "You are a high-precision enterprise structured extraction model."
"""

    vllm_sh = f"""#!/usr/bin/env bash
set -euo pipefail

# Reconstruct standard HF weights for vLLM serving (supports FP8/AWQ/tensor-parallel)
python "{hf_dir.as_posix()}/reconstruct_standard_hf.py" \\
  --src "{hf_dir.as_posix()}" \\
  --dst "{deploy_dir.as_posix()}/standard_hf_vllm"

vllm serve "{deploy_dir.as_posix()}/standard_hf_vllm" \\
  --served-model-name "{slug}-compressed" \\
  --dtype bfloat16 \\
  --max-model-len 4096 \\
  --gpu-memory-utilization 0.90 \\
  --port 8000
"""

    (deploy_dir / "export_gguf.sh").write_text(gguf_sh, encoding="utf-8")
    (deploy_dir / "export_gguf.ps1").write_text(gguf_ps1, encoding="utf-8")
    (deploy_dir / "Modelfile").write_text(modelfile, encoding="utf-8")
    (deploy_dir / "serve_vllm.sh").write_text(vllm_sh, encoding="utf-8")

    return {
        "compressed_hf_dir": str(hf_dir),
        "deploy_dir": str(deploy_dir),
        "compressed_hf_size_gb": _dir_size_gb(hf_dir),
        "estimated_q4_km_gguf_gb": q4_gb,
        "export_gguf_sh": str(deploy_dir / "export_gguf.sh"),
        "export_gguf_ps1": str(deploy_dir / "export_gguf.ps1"),
        "modelfile": str(deploy_dir / "Modelfile"),
        "serve_vllm_sh": str(deploy_dir / "serve_vllm.sh"),
        "duration_sec": round(time.perf_counter() - t0, 2),
    }
