"""Model introspection, factorized linear layers, and 16GB-VRAM streaming utilities."""

from __future__ import annotations

import gc
import json
import math
from pathlib import Path
from typing import Any

import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import init_empty_weights
from safetensors.torch import load_file as load_safetensors
from transformers import AutoConfig, AutoModelForCausalLM

TARGET_LINEAR_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "w1",
    "w2",
    "w3",
    "c_attn",
    "c_proj",
    "c_fc",
)


def get_vram_stats_gb() -> dict[str, float]:
    """Return current, peak, reserved, and total CUDA memory in GB."""
    if not torch.cuda.is_available():
        return {"allocated_gb": 0.0, "peak_gb": 0.0, "reserved_gb": 0.0, "total_gb": 0.0}
    props = torch.cuda.get_device_properties(0)
    return {
        "allocated_gb": round(torch.cuda.memory_allocated(0) / (1024**3), 4),
        "peak_gb": round(torch.cuda.max_memory_allocated(0) / (1024**3), 4),
        "reserved_gb": round(torch.cuda.memory_reserved(0) / (1024**3), 4),
        "total_gb": round(props.total_memory / (1024**3), 2),
    }


def get_ram_rss_gb() -> float:
    """Return current process resident set size (system RAM) in GB."""
    return round(psutil.Process().memory_info().rss / (1024**3), 4)


def release_cuda() -> None:
    """Force garbage collection and release cached CUDA VRAM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass


def to_device_recursive(obj: Any, device: str | torch.device) -> Any:
    """Recursively move tensors inside nested structures to target device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, tuple):
        return tuple(to_device_recursive(x, device) for x in obj)
    if isinstance(obj, list):
        return [to_device_recursive(x, device) for x in obj]
    if isinstance(obj, dict):
        return {k: to_device_recursive(v, device) for k, v in obj.items()}
    return obj


def get_submodule(root: nn.Module, dotted_name: str) -> nn.Module:
    """Retrieve a nested submodule by dotted path."""
    cur = root
    for part in dotted_name.split("."):
        cur = getattr(cur, part)
    return cur


def replace_submodule(root: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    """Replace a nested submodule in-place by dotted path."""
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _best_two_factors(n: int) -> tuple[int, int]:
    """Find balanced integer factors (a, b) such that a * b == n."""
    root = int(math.isqrt(n))
    for a in range(root, 1, -1):
        if n % a == 0:
            return a, n // a
    return 1, n


def snap_aligned_rank(raw_rank: float | int, max_rank: int, align_multiple: int = 64) -> int:
    """Snap a target rank/bond_dim to a hardware-aligned multiple (default: 64)."""
    if align_multiple < 1:
        raise ValueError("align_multiple must be >= 1")
    if align_multiple == 1:
        return max(1, min(int(float(raw_rank) + 0.5), int(max_rank)))
    m = int(align_multiple)
    max_aligned = (int(max_rank) // m) * m
    if max_aligned < m:
        return m
    snapped = int(float(raw_rank) / m + 0.5) * m
    return max(m, min(snapped, max_aligned))


class LowRankLinear(nn.Module):
    """Two-stage factorized linear projection: x -> v_proj(x) -> u_proj(h)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = False,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self._in_features = int(in_features)
        self._out_features = int(out_features)
        self._rank = int(rank)
        self.v_proj = nn.Linear(self._in_features, self._rank, bias=False, dtype=dtype)
        self.u_proj = nn.Linear(self._rank, self._out_features, bias=bias, dtype=dtype)

    @property
    def in_features(self) -> int:
        return self._in_features

    @property
    def out_features(self) -> int:
        return self._out_features

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def original_params(self) -> int:
        bias_p = self._out_features if self.u_proj.bias is not None else 0
        return self._in_features * self._out_features + bias_p

    @property
    def compressed_params(self) -> int:
        bias_p = self._out_features if self.u_proj.bias is not None else 0
        return self._rank * (self._in_features + self._out_features) + bias_p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.u_proj(self.v_proj(x))

    @classmethod
    @torch.no_grad()
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int | None = None,
        target_ratio: float = 0.35,
        align_multiple: int = 64,
    ) -> "LowRankLinear":
        """Factorize an nn.Linear via SVD with hardware-aligned bottleneck rank."""
        out_f, in_f = linear.weight.shape
        max_saving_r = min(min(out_f, in_f) - 1, (out_f * in_f - 1) // (out_f + in_f))
        raw_r = float(rank) if rank is not None else ((1.0 - target_ratio) * out_f * in_f) / (out_f + in_f)
        if align_multiple >= 1 and max_saving_r >= align_multiple:
            r = snap_aligned_rank(raw_r, max_saving_r, align_multiple=align_multiple)
        else:
            r = max(1, min(max_saving_r, int(raw_r + 0.5)))

        u, s, vh = torch.linalg.svd(linear.weight.detach().float(), full_matrices=False)
        r = min(r, s.numel())
        sq = torch.sqrt(s[:r].clamp_min(1e-12))
        layer = cls(in_f, out_f, r, bias=(linear.bias is not None), dtype=linear.weight.dtype)
        layer.u_proj.weight.data.copy_((u[:, :r] * sq.unsqueeze(0)).to(linear.weight.dtype))
        layer.v_proj.weight.data.copy_((sq.unsqueeze(1) * vh[:r, :]).to(linear.weight.dtype))
        if linear.bias is not None and layer.u_proj.bias is not None:
            layer.u_proj.bias.data.copy_(linear.bias.detach().to(linear.weight.dtype))
        return layer

    @torch.no_grad()
    def to_dense_linear(self) -> nn.Linear:
        """Reconstruct standard dense nn.Linear for GGUF/dense HF export."""
        has_bias = self.u_proj.bias is not None
        dense = nn.Linear(
            self._in_features,
            self._out_features,
            bias=has_bias,
            dtype=self.u_proj.weight.dtype,
            device=self.u_proj.weight.device,
        )
        w = self.u_proj.weight.float() @ self.v_proj.weight.float()
        dense.weight.copy_(w.to(self.u_proj.weight.dtype))
        if has_bias:
            dense.bias.copy_(self.u_proj.bias)
        return dense


class MPOLinear(nn.Module):
    """2-core Matrix Product Operator (MPO) factorized linear layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bond_dim: int,
        in_factors: tuple[int, int] | None = None,
        out_factors: tuple[int, int] | None = None,
        bias: bool = False,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self._in_features = int(in_features)
        self._out_features = int(out_features)
        self._bond_dim = int(bond_dim)

        i1, i2 = in_factors if in_factors is not None else _best_two_factors(self._in_features)
        o1, o2 = out_factors if out_factors is not None else _best_two_factors(self._out_features)
        if i1 * i2 != self._in_features or o1 * o2 != self._out_features:
            raise ValueError("MPO factors must multiply to in_features / out_features")

        self.in_factors = (i1, i2)
        self.out_factors = (o1, o2)
        self.core1 = nn.Parameter(torch.empty((o1, i1, self._bond_dim), dtype=dtype or torch.float32))
        self.core2 = nn.Parameter(torch.empty((self._bond_dim, o2, i2), dtype=dtype or torch.float32))
        if bias:
            self.bias = nn.Parameter(torch.zeros((self._out_features,), dtype=dtype or torch.float32))
        else:
            self.register_parameter("bias", None)

    @property
    def in_features(self) -> int:
        return self._in_features

    @property
    def out_features(self) -> int:
        return self._out_features

    @property
    def original_params(self) -> int:
        bias_p = self._out_features if self.bias is not None else 0
        return self._in_features * self._out_features + bias_p

    @property
    def compressed_params(self) -> int:
        bias_p = self._out_features if self.bias is not None else 0
        return self.core1.numel() + self.core2.numel() + bias_p

    @classmethod
    def from_dense_weight(
        cls,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        target_ratio: float,
        bond_dim: int | None = None,
        align_multiple: int = 64,
    ) -> "MPOLinear | LowRankLinear":
        """Factorize a 2D weight matrix (out, in) into a 2-core MPO via reshaped SVD."""
        out_f, in_f = weight.shape
        i1, i2 = _best_two_factors(in_f)
        o1, o2 = _best_two_factors(out_f)

        # Fallback to LowRankLinear if dimensions are prime or degenerate
        if min(i1, i2, o1, o2) < 4:
            max_saving_r = min(min(out_f, in_f) - 1, (out_f * in_f - 1) // (out_f + in_f))
            raw_r = float(bond_dim) if bond_dim is not None else ((1.0 - target_ratio) * out_f * in_f) / (out_f + in_f)
            if align_multiple >= 1 and max_saving_r >= align_multiple:
                rank = snap_aligned_rank(raw_r, max_saving_r, align_multiple=align_multiple)
            else:
                rank = max(1, min(max_saving_r, int(raw_r + 0.5)))
            layer = LowRankLinear(in_f, out_f, rank, bias=bias is not None, dtype=weight.dtype)
            u, s, vh = torch.linalg.svd(weight.float(), full_matrices=False)
            sq = torch.sqrt(s[:rank].clamp_min(1e-12))
            layer.u_proj.weight.data.copy_((u[:, :rank] * sq.unsqueeze(0)).to(weight.dtype))
            layer.v_proj.weight.data.copy_((sq.unsqueeze(1) * vh[:rank, :]).to(weight.dtype))
            if bias is not None:
                layer.u_proj.bias.data.copy_(bias.to(weight.dtype))
            return layer

        denom = o1 * i1 + o2 * i2
        max_saving_chi = min(o1 * i1, o2 * i2, (out_f * in_f - 1) // denom)
        raw_chi = float(bond_dim) if bond_dim is not None else ((1.0 - target_ratio) * out_f * in_f) / denom
        if align_multiple >= 1 and max_saving_chi >= align_multiple:
            chi = snap_aligned_rank(raw_chi, max_saving_chi, align_multiple=align_multiple)
        else:
            chi = max(4, min(max_saving_chi, int(raw_chi)))

        # Reshape (o1, o2, i1, i2) -> permute (o1, i1, o2, i2) -> matrix (o1*i1, o2*i2)
        w_4d = weight.float().reshape(o1, o2, i1, i2).permute(0, 2, 1, 3).contiguous()
        w_mat = w_4d.reshape(o1 * i1, o2 * i2)
        u, s, vh = torch.linalg.svd(w_mat, full_matrices=False)
        chi = min(chi, s.numel())
        sq = torch.sqrt(s[:chi].clamp_min(1e-12))
        c1 = (u[:, :chi] * sq.unsqueeze(0)).reshape(o1, i1, chi)
        c2 = (sq.unsqueeze(1) * vh[:chi, :]).reshape(chi, o2, i2)

        mpo = cls(
            in_features=in_f,
            out_features=out_f,
            bond_dim=chi,
            in_factors=(i1, i2),
            out_factors=(o1, o2),
            bias=bias is not None,
            dtype=weight.dtype,
        )
        mpo.core1.data.copy_(c1.to(weight.dtype))
        mpo.core2.data.copy_(c2.to(weight.dtype))
        if bias is not None and mpo.bias is not None:
            mpo.bias.data.copy_(bias.to(weight.dtype))
        return mpo

    def reconstruct_weight(self) -> torch.Tensor:
        w = torch.einsum("oik,kpq->opiq", self.core1, self.core2)
        return w.reshape(self._out_features, self._in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape[:-1]
        x_2d = x.reshape(-1, self._in_features)
        i1, i2 = self.in_factors
        o1, o2 = self.out_factors
        x_3d = x_2d.reshape(-1, i1, i2)
        # Contract over i2 with core2 (chi, o2, i2) -> (batch, i1, chi, o2)
        h = torch.einsum("bij,kqj->bikq", x_3d, self.core2)
        # Contract over (i1, chi) with core1 (o1, i1, chi) -> (batch, o1, o2)
        y = torch.einsum("bikq,oik->boq", h, self.core1)
        out = y.reshape(*orig_shape, self._out_features)
        if self.bias is not None:
            out = out + self.bias
        return out

    @torch.no_grad()
    def to_dense_linear(self) -> nn.Linear:
        dense = nn.Linear(
            self._in_features,
            self._out_features,
            bias=self.bias is not None,
            dtype=self.core1.dtype,
            device=self.core1.device,
        )
        dense.weight.copy_(self.reconstruct_weight().to(self.core1.dtype))
        if self.bias is not None:
            dense.bias.copy_(self.bias)
        return dense


def _locate_backbone_components(model: nn.Module) -> dict[str, Any]:
    """Locate transformer layers, embeddings, rotary_emb, norm, and lm_head across HF architectures."""
    backbone = getattr(model, "model", None) or getattr(model, "transformer", None) or model
    layers = None
    for attr in ("layers", "h", "blocks", "decoder_layers"):
        candidate = getattr(backbone, attr, None)
        if isinstance(candidate, nn.ModuleList):
            layers = candidate
            break
    if layers is None:
        raise ValueError(f"Could not locate transformer ModuleList on {type(model).__name__}")

    embed_tokens = (
        getattr(backbone, "embed_tokens", None)
        or getattr(backbone, "wte", None)
        or getattr(backbone, "word_embeddings", None)
    )
    rotary_emb = getattr(backbone, "rotary_emb", None)
    final_norm = (
        getattr(backbone, "norm", None)
        or getattr(backbone, "ln_f", None)
        or getattr(backbone, "final_layernorm", None)
    )
    lm_head = getattr(model, "lm_head", None)
    return {
        "backbone": backbone,
        "layers": layers,
        "embed_tokens": embed_tokens,
        "rotary_emb": rotary_emb,
        "final_norm": final_norm,
        "lm_head": lm_head,
    }


def detect_architecture(model: nn.Module) -> dict[str, Any]:
    """Inspect model config and module tree for dense vs. MoE structure."""
    cfg = getattr(model, "config", None)
    text_cfg = getattr(cfg, "text_config", cfg) if cfg is not None else None
    comps = _locate_backbone_components(model)
    layers: nn.ModuleList = comps["layers"]

    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers_prefix = "transformer.h"
    elif hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
        layers_prefix = "model.language_model.layers"
    else:
        layers_prefix = "model.layers"

    model_type = getattr(text_cfg, "model_type", None) or (getattr(cfg, "model_type", "unknown") if cfg else "unknown")
    archs = getattr(cfg, "architectures", []) if cfg else []
    num_experts = (
        getattr(text_cfg, "num_local_experts", None)
        or getattr(text_cfg, "n_routed_experts", None)
        or getattr(text_cfg, "num_experts", None)
    )
    num_experts_per_tok = (
        getattr(text_cfg, "num_experts_per_tok", None)
        or getattr(text_cfg, "num_selected_experts", None)
    )

    has_moe_modules = any(
        any(k in name for k in ("experts", "block_sparse_moe", "moe"))
        for name, _ in layers[0].named_modules()
    ) if len(layers) > 0 else False

    is_moe = bool((num_experts and num_experts > 1) or has_moe_modules or "moe" in str(model_type).lower())
    return {
        "model_type": str(model_type),
        "architecture_class": archs[0] if archs else type(model).__name__,
        "is_moe": is_moe,
        "num_experts": int(num_experts) if num_experts else 0,
        "num_experts_per_tok": int(num_experts_per_tok) if num_experts_per_tok else 0,
        "num_layers": len(layers),
        "hidden_size": int(getattr(text_cfg, "hidden_size", 0) or 0),
        "intermediate_size": int(getattr(text_cfg, "intermediate_size", 0) or 0),
        "vocab_size": int(getattr(text_cfg, "vocab_size", 0) or 0),
        "layers_prefix": layers_prefix,
        "components": comps,
    }


def get_target_linear_names(layer: nn.Module) -> list[str]:
    """Return dotted names of target nn.Linear projections within a transformer block."""
    targets: list[str] = []
    for name, mod in layer.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        leaf = name.split(".")[-1]
        if leaf in TARGET_LINEAR_SUFFIXES:
            targets.append(name)
    return targets


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count total, non-embedding, and transformer-block parameters."""
    total = sum(p.numel() for p in model.parameters())
    embed_params = 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Embedding):
            embed_params += sum(p.numel() for p in mod.parameters(recurse=False))
    block_params = 0
    try:
        layers = _locate_backbone_components(model)["layers"]
        block_params = sum(p.numel() for p in layers.parameters())
    except Exception:
        pass
    return {
        "total": int(total),
        "non_embedding": int(total - embed_params),
        "transformer_blocks": int(block_params),
    }


def load_compressed_model(
    hf_dir: str | Path,
    device: str = "cpu",
    dtype: torch.dtype | None = None,
) -> nn.Module:
    """Reload a compressed HF checkpoint with zero-copy meta initialization."""
    hf_path = Path(hf_dir)
    manifest_path = hf_path / "compression_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing compression_manifest.json in {hf_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = AutoConfig.from_pretrained(str(hf_path))

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
        # Swap target linears according to manifest on meta device (0 RAM)
        for layer_entry in manifest.get("layers", []):
            for _, spec in layer_entry.get("projections", {}).items():
                if spec.get("status") != "compressed":
                    continue
                dotted = f"{spec['full_name']}"
                orig_mod = get_submodule(model, dotted)
                has_bias = getattr(orig_mod, "bias", None) is not None
                mod_dtype = dtype or orig_mod.weight.dtype
                if spec.get("kind") == "mpo":
                    new_mod: nn.Module = MPOLinear(
                        in_features=spec["in_features"],
                        out_features=spec["out_features"],
                        bond_dim=spec["bond_dim"],
                        in_factors=tuple(spec["in_factors"]),
                        out_factors=tuple(spec["out_factors"]),
                        bias=has_bias,
                        dtype=mod_dtype,
                    )
                else:
                    new_mod = LowRankLinear(
                        in_features=spec["in_features"],
                        out_features=spec["out_features"],
                        rank=spec["rank"],
                        bias=has_bias,
                        dtype=mod_dtype,
                    )
                replace_submodule(model, dotted, new_mod)

    # Load safetensors weights (single or sharded) on CPU
    state_dict: dict[str, torch.Tensor] = {}
    single_st = hf_path / "model.safetensors"
    index_st = hf_path / "model.safetensors.index.json"
    if single_st.exists():
        state_dict = load_safetensors(str(single_st), device="cpu")
    elif index_st.exists():
        idx_data = json.loads(index_st.read_text(encoding="utf-8"))
        shard_files = sorted(set(idx_data.get("weight_map", {}).values()))
        for sf in shard_files:
            state_dict.update(load_safetensors(str(hf_path / sf), device="cpu"))
    else:
        bin_path = hf_path / "pytorch_model.bin"
        if not bin_path.exists():
            raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin found in {hf_path}")
        state_dict = torch.load(str(bin_path), map_location="cpu", weights_only=True)

    if not (hasattr(model, "model") and hasattr(model.model, "language_model")):
        if any(k.startswith("model.language_model.") for k in state_dict):
            state_dict = {
                (f"model.{k[len('model.language_model.'):]}" if k.startswith("model.language_model.") else k): v
                for k, v in state_dict.items()
            }

    if dtype is not None:
        for k, v in state_dict.items():
            if v.is_floating_point() and v.dtype != dtype:
                state_dict[k] = v.to(dtype)

    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    del state_dict

    # Re-initialize any non-persistent rotary_emb buffers left on meta
    for mod in model.modules():
        if hasattr(mod, "inv_freq") and isinstance(mod.inv_freq, torch.Tensor) and mod.inv_freq.device.type == "meta":
            mod.__init__(config=getattr(mod, "config", model.config), device="cpu")

    # Re-tie weights if needed (e.g., lm_head <-> embed_tokens)
    if getattr(model.config, "tie_word_embeddings", False) and hasattr(model, "tie_weights"):
        model.tie_weights()
    non_tied_missing = [k for k in missing if "lm_head.weight" not in k]
    if non_tied_missing:
        raise RuntimeError(f"Unexpected missing keys loading compressed model: {non_tied_missing[:8]}")
    if unexpected:
        raise RuntimeError(f"Unexpected extra keys loading compressed model: {unexpected[:8]}")

    if device != "cpu":
        model = model.to(device)
    return model


def quantize_model_nf4_streaming(
    model: nn.Module,
    compute_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    max_gpu_gb: float = 13.5,
) -> tuple[nn.Module, bool, int, int]:
    """Quantize transformer block linears to 4-bit NF4 and place on GPU up to max_gpu_gb.

    Returns (model, fully_on_gpu, gpu_layers_count, total_layers_count).
    If the 4-bit model exceeds max_gpu_gb (e.g. 27B base on 16GB VRAM), remaining
    layers stay on CPU and are dispatched cleanly via accelerate.
    """
    import bitsandbytes as bnb
    from accelerate import dispatch_model, infer_auto_device_map

    comps = _locate_backbone_components(model)
    layers: nn.ModuleList = comps["layers"]
    total_layers = len(layers)

    for layer in layers:
        for name, mod in list(layer.named_modules()):
            if not isinstance(mod, nn.Linear):
                continue
            # Keep tiny scalar/router gates in original dtype for numerical stability
            if min(mod.in_features, mod.out_features) < 64:
                continue
            q_mod = bnb.nn.Linear4bit(
                mod.in_features,
                mod.out_features,
                bias=(mod.bias is not None),
                compute_dtype=compute_dtype,
                compress_statistics=True,
                quant_type="nf4",
            )
            q_mod.weight = bnb.nn.Params4bit(
                mod.weight.data.cpu(),
                requires_grad=False,
                compress_statistics=True,
                quant_type="nf4",
            )
            if mod.bias is not None:
                q_mod.bias = nn.Parameter(mod.bias.data.cpu().to(compute_dtype), requires_grad=False)
            replace_submodule(layer, name, q_mod)

    if device != "cuda" or not torch.cuda.is_available():
        return model, False, 0, total_layers

    release_cuda()
    # Estimate if full 4-bit model fits under max_gpu_gb
    # Note: Params4bit quantizes to ~0.55 bytes/param once moved to CUDA
    est_4bit_gb = sum(
        (p.numel() * 0.56 if isinstance(p, bnb.nn.Params4bit) else p.numel() * p.element_size())
        for p in model.parameters()
    ) / (1024**3)

    if est_4bit_gb <= max_gpu_gb:
        model.to(device)
        release_cuda()
        return model, True, total_layers, total_layers

    # Hybrid GPU+CPU dispatch for oversized base models (e.g. 27B base > 13.5 GB)
    dm = infer_auto_device_map(
        model,
        max_memory={0: f"{int(max_gpu_gb)}GiB", "cpu": "75GiB"},
        no_split_module_classes=getattr(model, "_no_split_modules", None),
    )
    model = dispatch_model(model, device_map=dm)
    gpu_layers = sum(1 for k, v in dm.items() if "layers." in k and (v == 0 or v == "cuda:0"))
    return model, False, gpu_layers, total_layers
