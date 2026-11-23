"""Stage 1: Layer-by-layer streaming activation covariance calibration (16GB VRAM cap)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from common.model_utils import (
    detect_architecture,
    get_ram_rss_gb,
    get_target_linear_names,
    get_vram_stats_gb,
    release_cuda,
    to_device_recursive,
)


class _CatcherStop(Exception):
    """Sentinel exception to halt forward pass once all required layer types are captured."""


class _LayerCatcher(nn.Module):
    """Captures hidden_states (at layer 0) and kwargs per distinct layer_type."""

    def __init__(
        self,
        wrapped: nn.Module,
        layer_type: str,
        capture_hidden: bool,
        capture_kwargs: bool,
        stop_after: bool,
        captured_inputs: list[torch.Tensor],
        kwargs_by_type: dict[str, list[dict[str, Any]]],
    ) -> None:
        super().__init__()
        self.wrapped = wrapped
        self.layer_type = layer_type
        self.capture_hidden = capture_hidden
        self.capture_kwargs = capture_kwargs
        self.stop_after = stop_after
        self.captured_inputs = captured_inputs
        self.kwargs_by_type = kwargs_by_type

    def forward(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        if self.capture_hidden:
            self.captured_inputs.append(hidden_states.detach().cpu())
        if self.capture_kwargs:
            clean_kwargs = {}
            for k, v in kwargs.items():
                if k in ("past_key_value", "past_key_values", "cache"):
                    continue
                clean_kwargs[k] = to_device_recursive(v, "cpu")
            self.kwargs_by_type.setdefault(self.layer_type, []).append(clean_kwargs)
        if self.stop_after:
            raise _CatcherStop()
        return hidden_states


def run_calibration(
    model: nn.Module,
    tokenizer: Any,
    calib_samples: list[dict[str, Any]],
    output_dir: str | Path,
    device: str = "cuda",
    max_seq_len: int = 256,
    batch_size: int = 1,
) -> dict[str, Any]:
    """Stream calibration samples block-by-block and cache per-linear input covariances."""
    t0 = time.perf_counter()
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    out_path = Path(output_dir)
    cache_dir = out_path / "calib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    arch = detect_architecture(model)
    comps = arch["components"]
    layers: nn.ModuleList = comps["layers"]
    embed_tokens = comps["embed_tokens"]
    rotary_emb = comps["rotary_emb"]

    backbone_cfg = getattr(comps["backbone"], "config", model.config)
    layer_types: list[str] | None = getattr(backbone_cfg, "layer_types", None)
    if layer_types and len(layer_types) == len(layers):
        first_idx_by_type: dict[str, int] = {}
        for idx, lt in enumerate(layer_types):
            if lt not in first_idx_by_type:
                first_idx_by_type[lt] = idx
        max_catch_idx = max(first_idx_by_type.values())
    else:
        layer_types = ["default"] * len(layers)
        first_idx_by_type = {"default": 0}
        max_catch_idx = 0

    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = [s["text"] for s in calib_samples]
    batches: list[dict[str, torch.Tensor]] = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(
            texts[i : i + batch_size],
            return_tensors="pt",
            padding=(batch_size > 1),
            truncation=True,
            max_length=max_seq_len,
        )
        batches.append(enc)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Step 1: Catcher pass (ONLY embed_tokens + rotary_emb on GPU)
    if embed_tokens is not None:
        embed_tokens.to(device)
    if rotary_emb is not None:
        rotary_emb.to(device)

    captured_inputs: list[torch.Tensor] = []
    kwargs_by_type: dict[str, list[dict[str, Any]]] = {}
    originals: dict[int, nn.Module] = {}

    capture_indices = set(first_idx_by_type.values())
    for idx in range(max_catch_idx + 1):
        originals[idx] = layers[idx]
        layers[idx] = _LayerCatcher(
            wrapped=originals[idx],
            layer_type=layer_types[idx],
            capture_hidden=(idx == 0),
            capture_kwargs=(idx in capture_indices),
            stop_after=(idx == max_catch_idx),
            captured_inputs=captured_inputs,
            kwargs_by_type=kwargs_by_type,
        )

    try:
        with torch.no_grad():
            for batch in batches:
                batch_dev = {k: v.to(device) for k, v in batch.items()}
                try:
                    model(**batch_dev, use_cache=False)
                except _CatcherStop:
                    pass
    finally:
        for idx, orig in originals.items():
            layers[idx] = orig
        if embed_tokens is not None:
            embed_tokens.to("cpu")
        if rotary_emb is not None:
            rotary_emb.to("cpu")
        release_cuda()

    inps = captured_inputs
    peak_vram = get_vram_stats_gb()["peak_gb"]
    peak_ram = get_ram_rss_gb()

    moe_router_stats: dict[int, list[float]] = {}
    cov_dtype = torch.float32
    num_layers = len(layers)

    # Step 2: Strict per-block streaming loop
    for layer_idx, layer in enumerate(layers):
        layer.to(device)
        target_names = get_target_linear_names(layer)
        lt = layer_types[layer_idx]
        layer_kwargs = kwargs_by_type.get(lt, kwargs_by_type.get("default", []))

        # Deduplicate covariance accumulation by input tensor key across batches
        unique_grams: dict[str, torch.Tensor] = {}
        unique_counts: dict[str, int] = {}
        proj_to_group: dict[str, str] = {}
        handles: list[Any] = []

        # Per-batch cache: (data_ptr, d_in) -> (group_id, gram, n_tok)
        batch_gram_cache: dict[tuple[int, int], tuple[str, torch.Tensor, int]] = {}

        def _make_pre_hook(proj_name: str):
            def hook(_mod: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
                x = inputs[0]
                if not isinstance(x, torch.Tensor):
                    return
                x_2d = x.detach().reshape(-1, x.shape[-1])
                key = (x_2d.data_ptr(), x_2d.shape[-1])
                if key not in batch_gram_cache:
                    xf = x_2d.to(cov_dtype)
                    gram = xf.T @ xf
                    gid = proj_to_group.get(proj_name, proj_name)
                    batch_gram_cache[key] = (gid, gram, int(x_2d.shape[0]))
                gid, gram, n_tok = batch_gram_cache[key]
                if proj_name not in proj_to_group:
                    proj_to_group[proj_name] = gid
                    if gid == proj_name:
                        if gid not in unique_grams:
                            unique_grams[gid] = gram.clone()
                            unique_counts[gid] = n_tok
                        else:
                            unique_grams[gid].add_(gram)
                            unique_counts[gid] += n_tok
                elif proj_to_group[proj_name] == proj_name:
                    unique_grams[gid].add_(gram)
                    unique_counts[gid] += n_tok
            return hook

        for name in target_names:
            mod = layer.get_submodule(name)
            handles.append(mod.register_forward_pre_hook(_make_pre_hook(name)))

        # Optional MoE router frequency hook
        expert_counts = None
        if arch["is_moe"] and arch["num_experts"] > 0:
            expert_counts = torch.zeros(arch["num_experts"], dtype=torch.float64)
            for m_name, m_mod in layer.named_modules():
                if isinstance(m_mod, nn.Linear) and m_name.split(".")[-1] in ("gate", "router"):
                    def _router_hook(_m: nn.Module, _in: Any, out: Any) -> None:
                        logits = out[0] if isinstance(out, tuple) else out
                        if isinstance(logits, torch.Tensor) and logits.shape[-1] == arch["num_experts"]:
                            top_k = max(1, arch["num_experts_per_tok"])
                            topk_idx = torch.topk(logits.detach().reshape(-1, logits.shape[-1]), k=top_k, dim=-1).indices
                            counts = torch.bincount(topk_idx.reshape(-1), minlength=arch["num_experts"]).cpu().double()
                            expert_counts.add_(counts)
                    handles.append(m_mod.register_forward_hook(_router_hook))
                    break

        is_linear_attn = (
            getattr(layer, "layer_type", None) == "linear_attention"
            or lt == "linear_attention"
            or (hasattr(layer, "linear_attn") and not hasattr(layer, "self_attn"))
        )

        outs: list[torch.Tensor] = []
        with torch.no_grad():
            for b_idx, inp in enumerate(inps):
                batch_gram_cache.clear()
                inp_dev = inp.to(device)
                kw_dev = to_device_recursive(layer_kwargs[b_idx], device)
                if is_linear_attn:
                    am = kw_dev.get("attention_mask")
                    if batch_size == 1 or (isinstance(am, torch.Tensor) and am.ndim > 2):
                        kw_dev["attention_mask"] = None
                out = layer(inp_dev, **kw_dev)
                out_tensor = out[0] if isinstance(out, tuple) else out
                outs.append(out_tensor.detach().cpu())
                del inp_dev, kw_dev, out, out_tensor

        for h in handles:
            h.remove()
        batch_gram_cache.clear()

        # Normalize and persist deduplicated per-layer covariances to disk
        unique_covs: dict[str, torch.Tensor] = {}
        for gid, gram in unique_grams.items():
            denom = max(unique_counts.get(gid, 1), 1)
            unique_covs[gid] = (gram / float(denom)).cpu()

        torch.save(
            {"_matrices": unique_covs, "_aliases": proj_to_group},
            cache_dir / f"layer_{layer_idx:03d}.pt",
        )

        if expert_counts is not None and expert_counts.sum().item() > 0:
            probs = (expert_counts / expert_counts.sum()).tolist()
            moe_router_stats[layer_idx] = [round(p, 6) for p in probs]

        peak_vram = max(peak_vram, get_vram_stats_gb()["peak_gb"])
        peak_ram = max(peak_ram, get_ram_rss_gb())

        del unique_grams, unique_covs
        layer.to("cpu")
        release_cuda()
        inps = outs

        if (layer_idx + 1) % 4 == 0 or (layer_idx + 1) == num_layers:
            print(
                f"  [Stage 1] Layer {layer_idx + 1:02d}/{num_layers:02d} ({lt}) | "
                f"Peak VRAM={peak_vram:.3f} GB | RAM={peak_ram:.2f} GB | {time.perf_counter() - t0:.1f}s",
                flush=True,
            )

    elapsed = round(time.perf_counter() - t0, 2)
    return {
        "calib_cache_dir": str(cache_dir),
        "num_samples": len(calib_samples),
        "num_layers": num_layers,
        "max_seq_len": max_seq_len,
        "peak_vram_gb": round(peak_vram, 4),
        "peak_ram_gb": round(peak_ram, 4),
        "duration_sec": elapsed,
        "moe_router_stats": moe_router_stats,
    }
