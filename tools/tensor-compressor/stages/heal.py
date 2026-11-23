"""Stage 3: Optional lightweight LoRA recovery pass on calibration dataset."""

from __future__ import annotations

import math
import time
from typing import Any

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from common.model_utils import get_ram_rss_gb, get_vram_stats_gb, release_cuda


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
