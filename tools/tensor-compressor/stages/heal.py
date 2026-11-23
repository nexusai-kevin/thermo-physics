"""Stage 3: Optional lightweight LoRA recovery pass on calibration dataset."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors
from transformers import AutoTokenizer

from common.model_utils import (
    detect_architecture,
    get_ram_rss_gb,
    get_vram_stats_gb,
    load_compressed_model,
    release_cuda,
    replace_submodule,
    to_device_recursive,
)


def _compute_eval_loss(
    model: nn.Module,
    encoded_batches: list[dict[str, torch.Tensor]],
    device: str,
    dtype: torch.dtype,
) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in encoded_batches[:4]:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            with torch.amp.autocast("cuda", dtype=dtype, enabled=(device == "cuda")):
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += float(out.loss.item())
            count += 1
    return round(total_loss / max(count, 1), 4)


def run_healing(
    model: nn.Module,
    tokenizer: Any,
    calib_samples: list[dict[str, Any]],
    enabled: bool = False,
    steps: int = 20,
    lr: float = 1e-4,
    max_seq_len: int = 256,
    device: str = "cuda",
) -> dict[str, Any]:
    """Run optional LoRA recovery pass on compressed weights, or skip in free-proof mode."""
    if not enabled:
        return {
            "executed": False,
            "skipped_reason": "--heal flag not passed (fast free-proof mode)",
            "steps": 0,
            "loss_before": None,
            "loss_after": None,
            "duration_sec": 0.0,
            "peak_vram_gb": 0.0,
        }

    t0 = time.perf_counter()
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    if device == "cpu":
        amp_dtype = torch.float32

    encoded_batches: list[dict[str, torch.Tensor]] = []
    for sample in calib_samples:
        enc = tokenizer(
            sample["text"],
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_len,
            padding="max_length",
        )
        encoded_batches.append(enc)

    # Target inner u_proj nn.Linear inside LowRankLinear (avoids colliding with self_attn.v_proj)
    linear_names = {
        name.split(".")[-1]
        for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear)
    }
    if "u_proj" in linear_names:
        target_modules = ["u_proj"]
    else:
        target_modules = [t for t in ("q_proj", "o_proj") if t in linear_names] or ["q_proj"]

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model.to(device)
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass
    if hasattr(model, "enable_input_require_grads"):
        try:
            model.enable_input_require_grads()
        except Exception:
            pass

    loss_before = _compute_eval_loss(model, encoded_batches, device, amp_dtype)

    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=sorted(set(target_modules)),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_cfg)
    optimizer = torch.optim.AdamW(
        [p for p in peft_model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )

    peft_model.train()
    grad_accum = 4
    optimizer.zero_grad(set_to_none=True)

    for step in range(steps):
        batch = encoded_batches[step % len(encoded_batches)]
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device == "cuda")):
            out = peft_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss / grad_accum
        loss.backward()

        if (step + 1) % grad_accum == 0 or (step + 1) == steps:
            torch.nn.utils.clip_grad_norm_(peft_model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    loss_after = _compute_eval_loss(peft_model, encoded_batches, device, amp_dtype)
    peak_vram = get_vram_stats_gb()["peak_gb"]
    peak_ram = get_ram_rss_gb()

    merged_model = peft_model.merge_and_unload()
    if hasattr(merged_model, "gradient_checkpointing_disable"):
        try:
            merged_model.gradient_checkpointing_disable()
        except Exception:
            pass
    merged_model.to("cpu")
    del optimizer, peft_model
    release_cuda()

    return {
        "executed": True,
        "steps": steps,
        "lr": lr,
        "loss_before": loss_before,
        "loss_after": loss_after,
        "ppl_before": round(math.exp(min(loss_before, 20.0)), 2),
        "ppl_after": round(math.exp(min(loss_after, 20.0)), 2),
        "duration_sec": round(time.perf_counter() - t0, 2),
        "peak_vram_gb": round(peak_vram, 4),
        "peak_ram_gb": round(peak_ram, 4),
    }


def _prepare_masked_batches(
    tokenizer: Any,
    calib_samples: list[dict[str, Any]],
    max_seq_len: int,
) -> list[dict[str, torch.Tensor]]:
    """Tokenize calibration samples and mask prompt tokens with -100 in labels."""
    batches: list[dict[str, torch.Tensor]] = []
    for sample in calib_samples:
        full_enc = tokenizer(
            sample["text"],
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_len,
            padding=False,
        )
        prompt_enc = tokenizer(
            sample["prompt"],
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_len,
            padding=False,
        )
        input_ids = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]
        labels = input_ids.clone()
        prompt_len = min(int(prompt_enc["input_ids"].shape[1]), int(labels.shape[1]) - 1)
        if prompt_len > 0:
            labels[:, :prompt_len] = -100
        batches.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        )
    return batches


class _LoRALinear(nn.Module):
    """Lightweight LoRA wrapper around a frozen nn.Linear for fast tail-layer recovery."""

    def __init__(self, base_linear: nn.Linear, r: int = 16, lora_alpha: int = 32) -> None:
        super().__init__()
        self.base_linear = base_linear
        for p in self.base_linear.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scaling = float(lora_alpha) / float(r)
        dtype = base_linear.weight.dtype
        dev = base_linear.weight.device
        self.lora_A = nn.Parameter(torch.empty((r, base_linear.in_features), dtype=dtype, device=dev))
        self.lora_B = nn.Parameter(torch.zeros((base_linear.out_features, r), dtype=dtype, device=dev))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        return base_out + lora_out * self.scaling

    @torch.no_grad()
    def delta_weight_cpu(self) -> torch.Tensor:
        return (self.lora_B.detach().float().cpu() @ self.lora_A.detach().float().cpu()) * self.scaling


class _TailCatchStop(Exception):
    """Sentinel to stop prefix forward pass after capturing layer kwargs."""


class _PrefixCatcher(nn.Module):
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
            raise _TailCatchStop()
        return hidden_states


def run_checkpoint_healing(
    compressed_hf_dir: str | Path,
    calib_samples: list[dict[str, Any]],
    steps: int = 12,
    lr: float = 3e-4,
    max_seq_len: int = 256,
    device: str = "cuda",
) -> dict[str, Any]:
    """Stream prefix layers on GPU, then train LoRA on the last 4 layers in BF16 (<8GB VRAM)."""
    t0 = time.perf_counter()
    hf_path = Path(compressed_hf_dir)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(str(hf_path))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    if device == "cpu":
        amp_dtype = torch.float32

    print(f"  [Stage 3] Loading compressed checkpoint to CPU RAM ({hf_path})...", flush=True)
    model = load_compressed_model(hf_path, device="cpu", dtype=amp_dtype)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    arch = detect_architecture(model)
    comps = arch["components"]
    layers: nn.ModuleList = comps["layers"]
    embed_tokens = comps["embed_tokens"]
    rotary_emb = comps["rotary_emb"]
    final_norm = comps["final_norm"]
    lm_head = comps["lm_head"]
    layers_prefix = arch.get("layers_prefix", "model.layers")
    total_layers = len(layers)
    num_train_layers = min(4, total_layers)
    start_layer = total_layers - num_train_layers

    backbone_cfg = getattr(comps["backbone"], "config", model.config)
    layer_types: list[str] | None = getattr(backbone_cfg, "layer_types", None)
    if not layer_types or len(layer_types) != total_layers:
        layer_types = ["default"] * total_layers

    first_idx_by_type: dict[str, int] = {}
    for idx, lt in enumerate(layer_types):
        if lt not in first_idx_by_type:
            first_idx_by_type[lt] = idx
    max_catch_idx = max(first_idx_by_type.values())
    capture_indices = set(first_idx_by_type.values())

    train_samples = calib_samples[: max(steps, min(len(calib_samples), 12))]
    batches = _prepare_masked_batches(tokenizer, train_samples, max_seq_len)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # 1. Catcher pass (embed_tokens + rotary_emb on GPU)
    if embed_tokens is not None:
        embed_tokens.to(device)
    if rotary_emb is not None:
        rotary_emb.to(device)

    captured_inputs: list[torch.Tensor] = []
    kwargs_by_type: dict[str, list[dict[str, Any]]] = {}
    originals: dict[int, nn.Module] = {}
    for idx in range(max_catch_idx + 1):
        originals[idx] = layers[idx]
        layers[idx] = _PrefixCatcher(
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
            for b in batches:
                try:
                    model(
                        input_ids=b["input_ids"].to(device),
                        attention_mask=b["attention_mask"].to(device),
                        use_cache=False,
                    )
                except _TailCatchStop:
                    pass
    finally:
        for idx, orig in originals.items():
            layers[idx] = orig
        if embed_tokens is not None:
            embed_tokens.to("cpu")
        if rotary_emb is not None:
            rotary_emb.to("cpu")
        release_cuda()

    # 2. Stream frozen prefix layers 0..start_layer-1 one block at a time on GPU
    h_list = captured_inputs
    if start_layer > 0:
        print(f"  [Stage 3] Streaming frozen prefix layers 0..{start_layer - 1}...", flush=True)
    for layer_idx in range(start_layer):
        layer = layers[layer_idx]
        lt = layer_types[layer_idx]
        l_kwargs = kwargs_by_type.get(lt, kwargs_by_type.get("default", []))
        is_linear_attn = (
            getattr(layer, "layer_type", None) == "linear_attention"
            or lt == "linear_attention"
            or (hasattr(layer, "linear_attn") and not hasattr(layer, "self_attn"))
        )
        layer.to(device)
        outs: list[torch.Tensor] = []
        with torch.no_grad():
            for b_idx, h in enumerate(h_list):
                kw_dev = to_device_recursive(l_kwargs[b_idx], device)
                if is_linear_attn:
                    kw_dev["attention_mask"] = None
                out = layer(h.to(device), **kw_dev)
                out_t = out[0] if isinstance(out, tuple) else out
                outs.append(out_t.detach().cpu())
                del kw_dev, out, out_t
        layer.to("cpu")
        release_cuda()
        h_list = outs
        if (layer_idx + 1) % 10 == 0 or (layer_idx + 1) == start_layer:
            print(
                f"  [Stage 3] Prefix stream {layer_idx + 1}/{start_layer} | "
                f"Peak VRAM={get_vram_stats_gb()['peak_gb']:.2f} GB | {time.perf_counter() - t0:.1f}s",
                flush=True,
            )

    # 3. Move ONLY last 4 layers + final_norm + lm_head to GPU in BF16 (~5 GB VRAM)
    for p in model.parameters():
        p.requires_grad_(False)

    tail_lora_refs: list[tuple[str, _LoRALinear]] = []
    trainable_params: list[nn.Parameter] = []
    target_suffixes = ("u_proj", "o_proj", "down_proj")

    for layer_idx in range(start_layer, total_layers):
        layer = layers[layer_idx].to(device)
        for name, mod in list(layer.named_modules()):
            if isinstance(mod, nn.Linear) and name.split(".")[-1] in target_suffixes:
                wrapper = _LoRALinear(mod, r=16, lora_alpha=32)
                replace_submodule(layer, name, wrapper)
                tail_lora_refs.append((f"{layers_prefix}.{layer_idx}.{name}.weight", wrapper))
                trainable_params.extend([wrapper.lora_A, wrapper.lora_B])

    if final_norm is not None:
        final_norm.to(device)
    if lm_head is not None:
        lm_head.to(device)

    print(
        f"  [Stage 3] Tail LoRA active on layers {start_layer}..{total_layers - 1} | "
        f"Trainable params={sum(p.numel() for p in trainable_params):,} | VRAM={get_vram_stats_gb()['allocated_gb']:.2f} GB",
        flush=True,
    )

    def _tail_forward(b_idx: int) -> torch.Tensor:
        h = h_list[b_idx].to(device)
        for layer_idx in range(start_layer, total_layers):
            layer = layers[layer_idx]
            lt = layer_types[layer_idx]
            l_kwargs = kwargs_by_type.get(lt, kwargs_by_type.get("default", []))
            kw_dev = to_device_recursive(l_kwargs[b_idx], device)
            is_lin = (
                getattr(layer, "layer_type", None) == "linear_attention"
                or lt == "linear_attention"
                or (hasattr(layer, "linear_attn") and not hasattr(layer, "self_attn"))
            )
            if is_lin:
                kw_dev["attention_mask"] = None
            out = layer(h, **kw_dev)
            h = out[0] if isinstance(out, tuple) else out
        if final_norm is not None:
            h = final_norm(h)
        logits = lm_head(h) if lm_head is not None else h
        labels = batches[b_idx]["labels"].to(device)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            ignore_index=-100,
        )

    def _tail_eval_loss() -> float:
        tot = 0.0
        n = min(6, len(batches))
        with torch.no_grad():
            for idx in range(n):
                tot += float(_tail_forward(idx).item())
        return round(tot / max(n, 1), 4)

    loss_before = _tail_eval_loss()

    try:
        import bitsandbytes as bnb
        optimizer: torch.optim.Optimizer = bnb.optim.PagedAdamW8bit(trainable_params, lr=lr, weight_decay=0.01)
    except Exception:
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

    t_train = time.perf_counter()
    for step in range(steps):
        t_step = time.perf_counter()
        b_idx = step % len(batches)
        optimizer.zero_grad(set_to_none=True)
        loss = _tail_forward(b_idx)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        dt_step = time.perf_counter() - t_step
        avg_step = (time.perf_counter() - t_train) / float(step + 1)
        eta = avg_step * float(steps - step - 1)
        print(
            f"  [Stage 3] step {step + 1}/{steps} | loss={float(loss.item()):.4f} | "
            f"{dt_step:.2f}s/step | ETA {eta:.0f}s",
            flush=True,
        )

    loss_after = _tail_eval_loss()
    peak_vram = get_vram_stats_gb()["peak_gb"]
    peak_ram = get_ram_rss_gb()

    # 4. Merge LoRA deltas directly into tail layers on CPU and save sharded safetensors (5GB shards)
    for layer_idx in range(start_layer, total_layers):
        layer = layers[layer_idx]
        for name, mod in list(layer.named_modules()):
            if isinstance(mod, _LoRALinear):
                delta_cpu = mod.delta_weight_cpu()
                base_lin = mod.base_linear.to("cpu")
                with torch.no_grad():
                    base_lin.weight.data.copy_((base_lin.weight.data.float() + delta_cpu).to(amp_dtype))
                replace_submodule(layer, name, base_lin)
        layer.to("cpu")

    if final_norm is not None:
        final_norm.to("cpu")
    if lm_head is not None:
        lm_head.to("cpu")

    merged_count = len(tail_lora_refs)
    del optimizer, trainable_params, tail_lora_refs, h_list, batches
    release_cuda()

    single_st = hf_path / "model.safetensors"
    model.save_pretrained(str(hf_path), safe_serialization=True, max_shard_size="5GB")
    if single_st.exists() and (hf_path / "model.safetensors.index.json").exists():
        single_st.unlink(missing_ok=True)
    del model
    release_cuda()

    heal_summary = {
        "executed": True,
        "mode": "tail_4layer_bf16_lora",
        "train_layers": f"{start_layer}..{total_layers - 1}",
        "steps": steps,
        "lr": lr,
        "merged_tensors": merged_count,
        "loss_before": loss_before,
        "loss_after": loss_after,
        "ppl_before": round(math.exp(min(loss_before, 20.0)), 2),
        "ppl_after": round(math.exp(min(loss_after, 20.0)), 2),
        "duration_sec": round(time.perf_counter() - t0, 2),
        "peak_vram_gb": round(peak_vram, 4),
        "peak_ram_gb": round(peak_ram, 4),
    }

    manifest_path = hf_path / "compression_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["heal_summary"] = heal_summary
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return heal_summary
