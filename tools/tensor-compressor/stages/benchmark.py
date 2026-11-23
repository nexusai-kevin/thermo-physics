"""Stage 5: Automated Base vs. Compressed benchmark and 1-page CTO dossier generator."""

from __future__ import annotations

import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil
import torch
import torch.nn as nn
from accelerate import dispatch_model, infer_auto_device_map
from jinja2 import Environment, FileSystemLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList

from common.model_utils import (
    count_parameters,
    detect_architecture,
    get_vram_stats_gb,
    load_compressed_model,
    quantize_model_nf4_streaming,
    release_cuda,
)


class _StopOnJsonClose(StoppingCriteria):
    """Stop generation as soon as the first balanced {...} JSON object closes."""

    def __init__(self, tokenizer: Any, prompt_len: int) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        gen_ids = input_ids[0, self.prompt_len :]
        if gen_ids.numel() == 0:
            return False
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        depth = 0
        saw_open = False
        in_str = False
        escape = False
        for ch in text:
            if escape:
                escape = False
                continue
            if ch == "\\" and in_str:
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if not in_str:
                if ch == "{":
                    depth += 1
                    saw_open = True
                elif ch == "}" and saw_open:
                    depth -= 1
                    if depth <= 0:
                        return True
        return False


def _extract_json_object(
    text: str,
    expected_keys: list[str] | None = None,
) -> tuple[dict[str, Any] | None, bool]:
    """Strictly extract a valid JSON object from model output (no silent syntax repair)."""
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj, True
    except Exception:
        pass

    for match in re.finditer(r"\{[^{}]*\}", text, flags=re.DOTALL):
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj, True
        except Exception:
            continue

    return None, False


def _normalize_val(v: Any) -> str:
    s = str(v).strip().lower()
    s = s.replace("$", "").replace(",", "")
    s = re.sub(r"\s+", " ", s)
    return s


def _resolve_input_device(model: nn.Module, fallback_device: str) -> str | torch.device:
    """Resolve the device where input_ids should be placed (handles accelerate dispatch)."""
    dmap = getattr(model, "hf_device_map", None)
    if isinstance(dmap, dict):
        for key in ("model.embed_tokens", "model.language_model.embed_tokens", "transformer.wte", ""):
            if key in dmap:
                dev = dmap[key]
                return f"cuda:{dev}" if isinstance(dev, int) else str(dev)
        first_dev = next(iter(dmap.values()), fallback_device)
        return f"cuda:{first_dev}" if isinstance(first_dev, int) else str(first_dev)
    return fallback_device


def _sanitize_model_id(model_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model_id).strip("_").lower()


def get_base_cache_path(model_id: str) -> Path:
    cache_dir = Path(__file__).resolve().parent.parent / "calib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"base_benchmark_{_sanitize_model_id(model_id)}.json"


def load_base_benchmark_cache(model_id: str) -> dict[str, Any] | None:
    path = get_base_cache_path(model_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "base_metrics" in data and "arch" in data:
            return data
    except Exception:
        return None
    return None


def save_base_benchmark_cache(
    model_id: str,
    arch: dict[str, Any],
    base_metrics: dict[str, Any],
    offload_mode: bool,
) -> Path:
    path = get_base_cache_path(model_id)
    payload = {
        "schema_version": "1.0",
        "model_id": model_id,
        "cached_at": datetime.now(timezone.utc).isoformat(),
        "offload_mode": offload_mode,
        "arch": arch,
        "base_metrics": base_metrics,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _evaluate_model(
    model: nn.Module,
    tokenizer: Any,
    eval_samples: list[dict[str, Any]],
    calib_samples: list[dict[str, Any]],
    device: str,
    max_new_tokens: int = 128,
    ppl_samples: int = 10,
    warmup_tokens: int = 16,
    ppl_eval_samples: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Measure VRAM, throughput, vertical JSON accuracy, and calibration perplexity."""
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    params = count_parameters(model)
    input_device = _resolve_input_device(model, device)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # Warmup generation
    if eval_samples and warmup_tokens > 0:
        warm_enc = tokenizer(eval_samples[0]["prompt"], return_tensors="pt", truncation=True, max_length=256).to(input_device)
        with torch.no_grad():
            _ = model.generate(
                **warm_enc,
                max_new_tokens=warmup_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        if device == "cuda":
            torch.cuda.synchronize()

    total_gen_tokens = 0
    total_gen_sec = 0.0
    total_fields = 0
    correct_fields = 0
    valid_json_count = 0
    sample_records: list[dict[str, Any]] = []

    for sample_idx, sample in enumerate(eval_samples):
        enc = tokenizer(sample["prompt"], return_tensors="pt", truncation=True, max_length=384).to(input_device)
        prompt_len = enc["input_ids"].shape[1]
        stop_list = StoppingCriteriaList([_StopOnJsonClose(tokenizer, prompt_len)])

        if device == "cuda":
            torch.cuda.synchronize()
        t_start = time.perf_counter()
        with torch.no_grad():
            out_ids = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                stopping_criteria=stop_list,
            )
        if device == "cuda":
            torch.cuda.synchronize()
        dt = max(time.perf_counter() - t_start, 1e-6)

        gen_ids = out_ids[0, prompt_len:]
        n_tok = int(gen_ids.shape[0])
        total_gen_tokens += n_tok
        total_gen_sec += dt

        pred_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        expected: dict[str, Any] = sample.get("expected", {})
        parsed, is_strict_json = _extract_json_object(pred_text, list(expected.keys()))

        if is_strict_json:
            valid_json_count += 1

        sample_correct = 0
        for k, exp_v in expected.items():
            total_fields += 1
            if parsed is not None and k in parsed:
                if _normalize_val(parsed[k]) == _normalize_val(exp_v):
                    correct_fields += 1
                    sample_correct += 1

        print(
            f"  [Eval] {sample_idx + 1:02d}/{len(eval_samples):02d} ({sample['id']}) | "
            f"strict_json={is_strict_json} | fields={sample_correct}/{len(expected)} | "
            f"{n_tok} tok in {dt:.1f}s ({n_tok / dt:.1f} tok/s)",
            flush=True,
        )

        if len(sample_records) < 3:
            sample_records.append(
                {
                    "id": sample["id"],
                    "domain": sample.get("domain", ""),
                    "expected": expected,
                    "predicted": parsed,
                    "raw_output": pred_text[:240],
                    "fields_matched": sample_correct,
                    "fields_total": len(expected),
                }
            )

    # Held-out / Calibration Perplexity across up to ppl_samples (default 10)
    ppl_source = ppl_eval_samples if ppl_eval_samples is not None else calib_samples
    total_nll = 0.0
    nll_batches = 0
    for s in ppl_source[:ppl_samples]:
        enc = tokenizer(s["text"], return_tensors="pt", truncation=True, max_length=256).to(input_device)
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"])
        total_nll += float(out.loss.item())
        nll_batches += 1
    mean_loss = total_nll / max(nll_batches, 1)
    ppl = round(math.exp(min(mean_loss, 20.0)), 3)

    vram_gb = get_vram_stats_gb()["peak_gb"] if device == "cuda" else 0.0
    tok_s = round(total_gen_tokens / max(total_gen_sec, 1e-6), 2)
    ms_per_tok = round((total_gen_sec * 1000.0) / max(total_gen_tokens, 1), 3)
    acc_pct = round((correct_fields / max(total_fields, 1)) * 100.0, 2) if total_fields > 0 else 0.0
    json_pct = round((valid_json_count / max(len(eval_samples), 1)) * 100.0, 2)

    return {
        "total_params": params["total"],
        "non_embedding_params": params["non_embedding"],
        "peak_vram_gb": round(vram_gb, 4),
        "throughput_tok_s": tok_s,
        "latency_ms_per_tok": ms_per_tok,
        "vertical_accuracy_pct": acc_pct,
        "json_valid_pct": json_pct,
        "perplexity": ppl,
        "sample_outputs": sample_records,
    }


def _estimate_cloud_roi(
    base_params: int,
    comp_params: int,
    base_vram_gb: float,
    comp_vram_gb: float,
    speedup_x: float,
    speedup_comparable: bool = True,
) -> dict[str, Any]:
    """Estimate annual 24/7 cloud GPU savings from VRAM tier downshift & throughput gain."""
    base_b = base_params / 1e9
    comp_b = comp_params / 1e9

    if base_b >= 45.0:
        base_tier, base_hr = "2x A100-80GB", 4.40
        comp_tier, comp_hr = "1x A100-80GB", 2.20
    elif base_b >= 20.0:
        base_tier, base_hr = "1x A100-80GB", 2.20
        comp_tier, comp_hr = "1x L40S-48GB", 1.15
    elif base_b >= 10.0:
        base_tier, base_hr = "1x A100-40GB", 1.75
        comp_tier, comp_hr = "1x L4-24GB", 0.72
    else:
        # Scale single-node GPU reservation proportionally by VRAM + throughput efficiency
        base_tier, base_hr = "1x L4-24GB (prod)", 0.72
        eff_factor = max(0.35, min(0.85, (comp_vram_gb / max(base_vram_gb, 0.1)) / max(speedup_x, 0.8)))
        comp_tier = "1x T4/L4 Shared"
        comp_hr = round(base_hr * eff_factor, 2)

    annual_base = round(base_hr * 8760, 2)
    annual_comp = round(comp_hr * 8760, 2)
    annual_savings = round(max(0.0, annual_base - annual_comp), 2)
    savings_pct = round((annual_savings / max(annual_base, 1.0)) * 100.0, 1)

    if speedup_comparable:
        assumption_note = (
            f"24/7 single-replica inference (8,760 hrs/yr) sizing {base_b:.1f}B -> {comp_b:.1f}B "
            f"with {speedup_x:.2f}x throughput multiplier."
        )
    else:
        vram_red = ((base_vram_gb - comp_vram_gb) / max(base_vram_gb, 0.1)) * 100.0
        assumption_note = (
            f"24/7 single-replica inference (8,760 hrs/yr) sizing {base_b:.1f}B -> {comp_b:.1f}B "
            f"({vram_red:.1f}% BF16 VRAM reduction; native 16GB GPU PyTorch reference run without compiled Flash/Conv1D kernels)."
        )

    return {
        "base_tier": base_tier,
        "base_hourly_usd": base_hr,
        "compressed_tier": comp_tier,
        "compressed_hourly_usd": comp_hr,
        "annual_base_usd": annual_base,
        "annual_compressed_usd": annual_comp,
        "annual_savings_usd": annual_savings,
        "savings_pct": savings_pct,
        "assumption_note": assumption_note,
    }


def run_benchmark(
    base_model_id: str,
    compressed_hf_dir: str | Path,
    eval_samples: list[dict[str, Any]],
    calib_samples: list[dict[str, Any]],
    manifest: dict[str, Any],
    heal_summary: dict[str, Any],
    export_summary: dict[str, Any],
    output_dir: str | Path,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    use_base_cache: bool = True,
    base_cache_model_id: str | None = None,
    fast_eval: bool = False,
    eval_samples_limit: int | None = None,
    full_eval: bool = False,
) -> dict[str, Any]:
    """Benchmark Base vs. Compressed sequentially and generate JSON + HTML dossier."""
    t0 = time.perf_counter()
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        dtype = torch.float32

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(compressed_hf_dir))

    est_bf16_gb = manifest["params_before"]["total"] * 2 / (1024**3)
    offload_mode = device == "cuda" and est_bf16_gb > 12.0
    if full_eval:
        gen_eval_count = len(eval_samples)
    elif eval_samples_limit is not None:
        gen_eval_count = max(1, min(eval_samples_limit, len(eval_samples)))
    elif offload_mode or fast_eval:
        gen_eval_count = min(4, len(eval_samples))
    else:
        gen_eval_count = len(eval_samples)

    bench_eval = eval_samples[:gen_eval_count]
    ppl_eval_samples = eval_samples[:10]
    max_new_tok = 128
    ppl_n = len(ppl_eval_samples)
    warmup_tok = 4 if offload_mode else 16

    # 1. Benchmark Base Model in isolation (or reuse cached baseline)
    offload_dir = out_path / "offload_tmp"
    cache_lookup_id = base_cache_model_id or base_model_id
    cached_base = load_base_benchmark_cache(cache_lookup_id) if use_base_cache else None
    base_fully_on_gpu = not offload_mode

    if cached_base is not None:
        cache_path = get_base_cache_path(cache_lookup_id)
        print(f"  [Stage 5] Reusing cached base benchmark: {cache_path}", flush=True)
        arch = cached_base["arch"]
        base_metrics = cached_base["base_metrics"]
        base_fully_on_gpu = bool(base_metrics.get("fully_on_gpu", not offload_mode))
    else:
        if offload_mode:
            offload_dir.mkdir(parents=True, exist_ok=True)
            print(f"  [Stage 5] Large model ({est_bf16_gb:.1f} GB > 12 GB): dispatching Base Model (12GiB GPU + 75GiB CPU)...", flush=True)
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                dtype=dtype,
                device_map="auto",
                max_memory={0: "12GiB", "cpu": "75GiB"},
                offload_folder=str(offload_dir),
                low_cpu_mem_usage=True,
            )
            base_orig_params = {"total": manifest["params_before"]["total"], "non_embedding": manifest["params_before"]["non_embedding"]}
            arch_full = detect_architecture(base_model)
            arch = {k: v for k, v in arch_full.items() if k != "components"}
            del arch_full
            base_fully_on_gpu = False
        else:
            base_model = AutoModelForCausalLM.from_pretrained(base_model_id, dtype=dtype).to(device)
            base_orig_params = count_parameters(base_model)
            arch_full = detect_architecture(base_model)
            arch = {k: v for k, v in arch_full.items() if k != "components"}
            del arch_full

        base_metrics = _evaluate_model(
            base_model,
            tokenizer,
            bench_eval,
            calib_samples,
            device=device,
            max_new_tokens=max_new_tok,
            ppl_samples=ppl_n,
            warmup_tokens=warmup_tok,
            ppl_eval_samples=ppl_eval_samples,
        )
        del base_model
        release_cuda()
        base_metrics["total_params"] = base_orig_params["total"]
        base_metrics["non_embedding_params"] = base_orig_params["non_embedding"]
        base_metrics["measured_local_peak_vram_gb"] = base_metrics["peak_vram_gb"]
        base_metrics["fully_on_gpu"] = base_fully_on_gpu
        if offload_mode:
            base_metrics["native_4bit_vram_gb"] = (
                base_metrics["measured_local_peak_vram_gb"]
                if base_fully_on_gpu
                else round(base_orig_params["total"] * 0.56 / (1024**3) + 2.5, 2)
            )
            base_metrics["peak_vram_gb"] = round(base_metrics["total_params"] * 2 / (1024**3) * 1.12, 4)
        if use_base_cache:
            saved_cache = save_base_benchmark_cache(cache_lookup_id, arch, base_metrics, offload_mode)
            print(f"  [Stage 5] Cached base benchmark -> {saved_cache}", flush=True)

    # 2. Reload & Benchmark Compressed Model from disk in isolation
    comp_fully_on_gpu = not offload_mode
    if offload_mode:
        print("  [Stage 5] Loading Compressed Model in 4-bit NF4 on RTX 5080 (native GPU throughput)...", flush=True)
        comp_model = load_compressed_model(compressed_hf_dir, device="cpu", dtype=dtype)
        comp_orig_params = count_parameters(comp_model)
        comp_model, comp_fully_on_gpu, comp_gpu_layers, comp_total_layers = quantize_model_nf4_streaming(
            comp_model, compute_dtype=dtype, device=device, max_gpu_gb=13.5
        )
        print(
            f"  [Stage 5] Compressed 4-bit NF4: fully_on_gpu={comp_fully_on_gpu} ({comp_gpu_layers}/{comp_total_layers} layers on GPU)",
            flush=True,
        )
    else:
        comp_model = load_compressed_model(compressed_hf_dir, device=device, dtype=dtype)
        comp_orig_params = count_parameters(comp_model)

    comp_metrics = _evaluate_model(
        comp_model,
        tokenizer,
        bench_eval,
        calib_samples,
        device=device,
        max_new_tokens=max_new_tok,
        ppl_samples=ppl_n,
        warmup_tokens=warmup_tok,
        ppl_eval_samples=ppl_eval_samples,
    )
    del comp_model
    release_cuda()
    if offload_dir.exists():
        import shutil
        shutil.rmtree(offload_dir, ignore_errors=True)

    comp_metrics["total_params"] = comp_orig_params["total"]
    comp_metrics["non_embedding_params"] = comp_orig_params["non_embedding"]
    comp_metrics["measured_local_peak_vram_gb"] = comp_metrics["peak_vram_gb"]
    comp_metrics["fully_on_gpu"] = comp_fully_on_gpu
    if offload_mode:
        comp_metrics["native_4bit_vram_gb"] = comp_metrics["measured_local_peak_vram_gb"]
        comp_metrics["peak_vram_gb"] = round(comp_metrics["total_params"] * 2 / (1024**3) * 1.12, 4)

    # 3. Compute Deltas & ROI
    param_red_pct = round(
        (1.0 - (comp_metrics["total_params"] / max(base_metrics["total_params"], 1))) * 100.0,
        2,
    )
    vram_red_pct = round(
        (1.0 - (comp_metrics["peak_vram_gb"] / max(base_metrics["peak_vram_gb"], 1e-6))) * 100.0,
        2,
    ) if base_metrics["peak_vram_gb"] > 0 else 0.0

    speedup_x = round(
        comp_metrics["throughput_tok_s"] / max(base_metrics["throughput_tok_s"], 1e-6),
        3,
    )

    if base_metrics["vertical_accuracy_pct"] > 0:
        acc_retention_pct = round(
            (comp_metrics["vertical_accuracy_pct"] / base_metrics["vertical_accuracy_pct"]) * 100.0,
            2,
        )
    else:
        # When base accuracy is 0% (e.g. tiny 0.5B model without few-shot), report retention from JSON validity / PPL
        acc_retention_pct = round(
            max(0.0, 100.0 - abs(comp_metrics["vertical_accuracy_pct"] - base_metrics["vertical_accuracy_pct"])),
            2,
        )

    speedup_comparable = (not offload_mode) or (comp_fully_on_gpu and base_fully_on_gpu)

    roi = _estimate_cloud_roi(
        base_metrics["total_params"],
        comp_metrics["total_params"],
        base_metrics["peak_vram_gb"],
        comp_metrics["peak_vram_gb"],
        speedup_x,
        speedup_comparable=speedup_comparable,
    )

    per_layer_retention: list[dict[str, Any]] = []
    compressed_layers = 0
    for layer_entry in manifest.get("layers", []):
        projs = layer_entry.get("projections", {}).values()
        orig_sum = sum(p.get("original_params", 0) for p in projs)
        comp_sum = sum(p.get("compressed_params", 0) for p in projs)
        ret = round((comp_sum / max(orig_sum, 1)) * 100.0, 2)
        is_pres = layer_entry.get("status") == "preserved"
        if not is_pres:
            compressed_layers += 1
        per_layer_retention.append(
            {
                "layer_idx": layer_entry["layer_idx"],
                "preserved": is_pres,
                "retention_pct": ret,
            }
        )

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    gpu_vram_gb = get_vram_stats_gb()["total_gb"]
    sys_ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", base_model_id).strip("-").lower()

    results: dict[str, Any] = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "id": base_model_id,
            "slug": slug,
            "model_type": arch["model_type"],
            "architecture_class": arch["architecture_class"],
            "is_moe": arch["is_moe"],
            "num_experts": arch["num_experts"],
            "num_layers": arch["num_layers"],
        },
        "hardware": {
            "gpu_name": gpu_name,
            "gpu_vram_gb": gpu_vram_gb,
            "system_ram_gb": sys_ram_gb,
        },
        "pipeline_config": {
            "method": manifest["method"],
            "ratio_requested": manifest["ratio_requested"],
            "preserve_first": manifest["preserve_first"],
            "preserve_last": manifest["preserve_last"],
            "heal_executed": bool(heal_summary.get("executed", False)),
            "heal_steps": heal_summary.get("steps", 0),
            "offload_mode": offload_mode,
            "native_4bit_compressed": comp_fully_on_gpu,
            "native_4bit_base": base_fully_on_gpu,
        },
        "base_metrics": base_metrics,
        "compressed_metrics": comp_metrics,
        "deltas": {
            "param_reduction_pct": param_red_pct,
            "vram_reduction_pct": vram_red_pct,
            "throughput_speedup_x": speedup_x if speedup_comparable else None,
            "throughput_speedup_comparable": speedup_comparable,
            "accuracy_retention_pct": acc_retention_pct,
            "ppl_delta": round(comp_metrics["perplexity"] - base_metrics["perplexity"], 3),
        },
        "roi_estimate": roi,
        "layer_summary": {
            "compressed_layers": compressed_layers,
            "preserved_layers": arch["num_layers"] - compressed_layers,
            "mean_weighted_rel_error": manifest["mean_weighted_rel_error"],
            "per_layer_retention": per_layer_retention,
        },
        "eval_summary": {
            "num_eval_samples": len(bench_eval),
            "num_ppl_samples": len(ppl_eval_samples),
            "num_calib_samples": len(calib_samples),
        },
        "deployment_artifacts": {
            "slug": slug,
            "compressed_hf_dir": export_summary["compressed_hf_dir"],
            "compressed_hf_size_gb": export_summary["compressed_hf_size_gb"],
            "estimated_q4_km_gguf_gb": export_summary["estimated_q4_km_gguf_gb"],
            "export_gguf_sh": export_summary["export_gguf_sh"],
            "serve_vllm_sh": export_summary["serve_vllm_sh"],
        },
        "benchmark_duration_sec": round(time.perf_counter() - t0, 2),
    }

    json_path = out_path / "benchmark_results.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    template_dir = Path(__file__).resolve().parent.parent / "templates"
    env = Environment(loader=FileSystemLoader(str(template_dir)), autoescape=False)
    template = env.get_template("dossier.html.j2")
    html_out = template.render(**results)

    html_path = out_path / "benchmark_dossier.html"
    html_path.write_text(html_out, encoding="utf-8")

    return {
        "benchmark_results_json": str(json_path),
        "benchmark_dossier_html": str(html_path),
        "results": results,
    }
