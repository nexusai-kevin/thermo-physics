"""Stage 2: Data-aware Whitened SVD / MPO tensor factorization + MoE compression hook."""

from __future__ import annotations

import heapq
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from common.model_utils import (
    LowRankLinear,
    MPOLinear,
    _best_two_factors,
    count_parameters,
    detect_architecture,
    get_ram_rss_gb,
    get_target_linear_names,
    get_vram_stats_gb,
    release_cuda,
    replace_submodule,
    snap_aligned_rank,
)


_ATTN_ROLE_SUFFIXES = frozenset(
    ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj", "c_attn")
)
_MLP_ROLE_SUFFIXES = frozenset(("gate_proj", "up_proj", "down_proj", "w1", "w2", "w3", "c_fc", "c_proj"))


def compute_target_rank(
    out_features: int,
    in_features: int,
    ratio: float,
    align_multiple: int = 64,
) -> int:
    """Compute hardware-aligned low-rank bottleneck r achieving ~`ratio` parameter reduction."""
    if align_multiple < 1:
        raise ValueError("align_multiple must be >= 1")
    min_dim = min(out_features, in_features)
    if ratio <= 0.0:
        return (min_dim // align_multiple) * align_multiple if min_dim >= align_multiple else min_dim
    max_saving_r = min(min_dim - 1, (out_features * in_features - 1) // (out_features + in_features))
    raw_r = ((1.0 - ratio) * out_features * in_features) / (out_features + in_features)
    return snap_aligned_rank(raw_r, max_saving_r, align_multiple=align_multiple)


def _cholesky_whitening_factor(cov: torch.Tensor) -> torch.Tensor:
    """Compute upper-triangular whitening factor S such that S^T S ≈ cov + lambda*I.

    Uses adaptive ridge damping with eigh fallback for rank-deficient covariances.
    """
    d = cov.shape[0]
    cov_sym = 0.5 * (cov.float() + cov.float().T)
    diag_mean = float(torch.diag(cov_sym).mean().item())
    eye = torch.eye(d, device=cov_sym.device, dtype=torch.float32)

    lam = max(1e-4 * abs(diag_mean), 1e-6)
    for _ in range(7):
        try:
            L = torch.linalg.cholesky(cov_sym + lam * eye)
            # We want W_tilde = W @ L so ||(U V - W) L||_F^2 = Tr((UV-W) cov (UV-W)^T)
            return L
        except RuntimeError:
            lam *= 10.0

    # Symmetric eigendecomposition fallback
    eigvals, eigvecs = torch.linalg.eigh(cov_sym)
    eigvals_clamped = torch.clamp(eigvals, min=1e-5)
    return eigvecs * torch.sqrt(eigvals_clamped).unsqueeze(0)


def _load_layer_whitening_factors(cov_path: Path, device: str) -> dict[str, torch.Tensor]:
    """Load per-layer covariances and compute Cholesky whitening factors once per unique group."""
    if not cov_path.exists():
        return {}
    raw_cov = torch.load(cov_path, map_location="cpu", weights_only=True)
    if isinstance(raw_cov, dict) and "_matrices" in raw_cov and "_aliases" in raw_cov:
        matrices = raw_cov["_matrices"]
        aliases = raw_cov["_aliases"]
        group_factors: dict[str, torch.Tensor] = {
            gid: _cholesky_whitening_factor(mat.to(device=device, dtype=torch.float32))
            for gid, mat in matrices.items()
        }
        return {pname: group_factors[gid] for pname, gid in aliases.items() if gid in group_factors}
    return {
        pname: _cholesky_whitening_factor(mat.to(device=device, dtype=torch.float32))
        for pname, mat in raw_cov.items()
    }


@torch.no_grad()
def _profile_projection_spectrum(
    linear: nn.Linear,
    whitening_factor: torch.Tensor | None,
    method: str,
    device: str = "cuda",
) -> tuple[torch.Tensor, int, int]:
    """Compute exact singular-value squared energies via symmetric Gram eigvalsh."""
    out_f, in_f = linear.weight.shape
    w = linear.weight.detach().to(device=device, dtype=torch.float32)

    if method == "mpo":
        i1, i2 = _best_two_factors(in_f)
        o1, o2 = _best_two_factors(out_f)
        if min(i1, i2, o1, o2) >= 4:
            w_mat = w.reshape(o1, o2, i1, i2).permute(0, 2, 1, 3).contiguous().reshape(o1 * i1, o2 * i2)
            cost_per_rank = o1 * i1 + o2 * i2
            max_rank_cap = min(o1 * i1, o2 * i2)
        else:
            w_mat = w
            cost_per_rank = out_f + in_f
            max_rank_cap = min(out_f, in_f) - 1
    else:
        w_mat = (w @ whitening_factor.to(device=device, dtype=torch.float32)) if whitening_factor is not None else w
        cost_per_rank = out_f + in_f
        max_rank_cap = min(out_f, in_f) - 1

    try:
        gram = (w_mat.T @ w_mat) if w_mat.shape[0] >= w_mat.shape[1] else (w_mat @ w_mat.T)
        gram = 0.5 * (gram + gram.T)
        eigvals = torch.linalg.eigvalsh(gram).clamp_min(0.0).flip(0).cpu()
    except RuntimeError:
        eigvals = torch.linalg.svdvals(w_mat).square().cpu()

    return eigvals, int(cost_per_rank), int(max_rank_cap)


def allocate_energy_optimal_ranks(
    specs: list[dict[str, Any]],
    ratio: float,
    align_multiple: int = 64,
) -> dict[str, Any]:
    """Globally allocate hardware-aligned ranks to maximize weighted singular energy at target ratio."""
    if align_multiple < 1:
        raise ValueError("align_multiple must be >= 1")

    ranks: dict[Any, int | None] = {}
    meta: dict[Any, dict[str, float]] = {}
    pool: list[dict[str, Any]] = []

    for idx, spec in enumerate(specs):
        key = spec["key"]
        in_f = int(spec["in_features"])
        out_f = int(spec["out_features"])
        bias_p = int(spec.get("bias_params", 0))
        cost_per_rank = int(spec["cost_per_rank"])
        max_rank_cap = int(spec["max_rank_cap"])
        dense_w = in_f * out_f

        max_saving_rank = min(max_rank_cap, (dense_w - 1) // max(cost_per_rank, 1))
        r_max = (max_saving_rank // align_multiple) * align_multiple
        if r_max < align_multiple:
            ranks[key] = None
            continue

        spec_f64 = spec["energy_spectrum"].detach().to(dtype=torch.float64)
        d = max(int(spec_f64.numel()), 1)
        tot_e = float(spec_f64.sum().item())
        if tot_e > 1e-30:
            p = spec_f64 / tot_e
        else:
            p = torch.full((d,), 1.0 / float(d), dtype=torch.float64)

        if d > 1:
            entropy = float(-(p * torch.clamp(p, min=1e-15).log()).sum().item() / math.log(float(d)))
            entropy = max(0.0, min(1.0, entropy))
        else:
            entropy = 1.0
        f_entropy = 0.75 + 0.50 * entropy

        num_layers = max(int(spec.get("num_layers", 1)), 1)
        layer_idx = int(spec.get("layer_idx", 0))
        z = float(layer_idx) / float(max(num_layers - 1, 1))
        f_depth = 1.20 - 0.35 * math.sin(math.pi * z)

        leaf = str(spec["proj_name"]).split(".")[-1]
        if leaf in _ATTN_ROLE_SUFFIXES:
            f_role = 1.25
        elif leaf in _MLP_ROLE_SUFFIXES:
            f_role = 0.90
        else:
            f_role = 1.0

        moe_eff = spec.get("moe_eff_ratio")
        if moe_eff is not None and ratio < 1.0:
            f_moe = max(0.25, min(2.5, (1.0 - float(moe_eff)) / max(1.0 - ratio, 1e-4)))
        else:
            f_moe = 1.0

        sensitivity = f_entropy * f_depth * f_role * f_moe
        meta[key] = {
            "spectral_entropy": round(entropy, 6),
            "sensitivity_weight": round(sensitivity, 6),
        }

        n_steps = r_max // align_multiple
        step_cost = align_multiple * cost_per_rank
        scale = sensitivity * (float(dense_w) / float(step_cost))
        utilities: list[float] = []
        for k in range(n_steps):
            start = k * align_multiple
            end = (k + 1) * align_multiple
            chunk_e = float(p[start:end].sum().item()) if start < d else 0.0
            tie_floor = 1e-15 * (1.0 - float(k) / float(max(n_steps, 1)))
            utilities.append(scale * max(chunk_e, tie_floor))

        r_breakeven = float(dense_w) / float(cost_per_rank)
        raw_floor = int(0.25 * (1.0 - ratio) * r_breakeven)
        r_floor = min(r_max, max(align_multiple, (raw_floor // align_multiple) * align_multiple))

        pool.append(
            {
                "key": key,
                "layer_idx": layer_idx,
                "order": idx,
                "dense_params": dense_w + bias_p,
                "bias_params": bias_p,
                "cost_per_rank": cost_per_rank,
                "step_cost": step_cost,
                "r_max": r_max,
                "r_floor": r_floor,
                "utilities": utilities,
            }
        )

    pool_orig_params = sum(item["dense_params"] for item in pool)
    target_pool_params = int(round((1.0 - ratio) * float(pool_orig_params)))

    if pool:
        floor_params = sum(item["r_floor"] * item["cost_per_rank"] + item["bias_params"] for item in pool)
        if floor_params > target_pool_params:
            for item in pool:
                item["r_floor"] = align_multiple

    current_params = 0
    allocated: dict[Any, int] = {}
    heap: list[tuple[float, int, int, int]] = []

    for pool_idx, item in enumerate(pool):
        r0 = item["r_floor"]
        allocated[item["key"]] = r0
        current_params += r0 * item["cost_per_rank"] + item["bias_params"]
        if r0 + align_multiple <= item["r_max"]:
            step_idx = r0 // align_multiple
            heapq.heappush(
                heap,
                (-item["utilities"][step_idx], item["layer_idx"], item["order"], pool_idx),
            )

    best_overshoot_cand: tuple[int, int, int] | None = None  # (new_err, -utility_rank, pool_idx)

    while heap and current_params < target_pool_params:
        neg_u, _, _, pool_idx = heapq.heappop(heap)
        item = pool[pool_idx]
        step_cost = item["step_cost"]
        key = item["key"]

        if current_params + step_cost <= target_pool_params:
            new_r = allocated[key] + align_multiple
            allocated[key] = new_r
            current_params += step_cost
            if new_r + align_multiple <= item["r_max"]:
                step_idx = new_r // align_multiple
                heapq.heappush(
                    heap,
                    (-item["utilities"][step_idx], item["layer_idx"], item["order"], pool_idx),
                )
        else:
            overshoot_err = abs((current_params + step_cost) - target_pool_params)
            if best_overshoot_cand is None or overshoot_err < best_overshoot_cand[0]:
                best_overshoot_cand = (overshoot_err, int(neg_u * 1e9), pool_idx)

    # Optional single boundary rounding step if it strictly moves closer to target_pool_params
    if best_overshoot_cand is not None and current_params < target_pool_params:
        cur_err = abs(current_params - target_pool_params)
        if best_overshoot_cand[0] < cur_err:
            item = pool[best_overshoot_cand[2]]
            key = item["key"]
            if allocated[key] + align_multiple <= item["r_max"]:
                allocated[key] += align_multiple
                current_params += item["step_cost"]

    for item in pool:
        ranks[item["key"]] = allocated[item["key"]]

    return {
        "ranks": ranks,
        "meta": meta,
        "compressible_params_before": int(pool_orig_params),
        "compressible_params_after": int(current_params),
        "target_compressible_params": int(target_pool_params),
    }


@torch.no_grad()
def whitened_svd_factorize(
    linear: nn.Linear,
    cov: torch.Tensor | None,
    rank: int,
    device: str = "cuda",
    whitening_factor: torch.Tensor | None = None,
    align_multiple: int | None = None,
) -> tuple[LowRankLinear, float]:
    """Factorize nn.Linear via data-aware whitened SVD and return (LowRankLinear, rel_error)."""
    orig_dtype = linear.weight.dtype
    out_f, in_f = linear.weight.shape
    has_bias = linear.bias is not None

    if align_multiple is not None and align_multiple >= 1:
        rank = snap_aligned_rank(rank, min(out_f, in_f) - 1, align_multiple=align_multiple)

    w = linear.weight.detach().to(device=device, dtype=torch.float32)
    if whitening_factor is not None:
        S = whitening_factor.to(device=device, dtype=torch.float32)
        w_tilde = w @ S
        try:
            use_triangular = torch.allclose(S, torch.tril(S), atol=1e-5)
        except Exception:
            use_triangular = False
    elif cov is not None:
        cov_dev = cov.to(device=device, dtype=torch.float32)
        S = _cholesky_whitening_factor(cov_dev)
        w_tilde = w @ S
        use_triangular = True
        try:
            # Verify S is lower-triangular (from Cholesky) vs general matrix (from eigh)
            is_lower = torch.allclose(S, torch.tril(S), atol=1e-5)
            use_triangular = is_lower
        except Exception:
            use_triangular = False
    else:
        S = None
        w_tilde = w
        use_triangular = False

    u, s, vh = torch.linalg.svd(w_tilde, full_matrices=False)
    r = min(rank, s.numel())
    sqrt_s = torch.sqrt(s[:r].clamp_min(1e-12))
    u_r = u[:, :r] * sqrt_s.unsqueeze(0)
    v_tilde = sqrt_s.unsqueeze(1) * vh[:r, :]

    if S is not None:
        if use_triangular:
            # Solve V_r @ S = V_tilde  <=>  S^T @ V_r^T = V_tilde^T
            v_r = torch.linalg.solve_triangular(S.T, v_tilde.T, upper=True).T
        else:
            v_r = (torch.linalg.pinv(S.T) @ v_tilde.T).T
        denom = float(torch.linalg.norm(w_tilde).item()) + 1e-8
        err = float(torch.linalg.norm(((u_r @ v_r) - w) @ S).item()) / denom
    else:
        v_r = v_tilde
        denom = float(torch.linalg.norm(w).item()) + 1e-8
        err = float(torch.linalg.norm((u_r @ v_r) - w).item()) / denom

    lr_mod = LowRankLinear(
        in_features=in_f,
        out_features=out_f,
        rank=r,
        bias=has_bias,
        dtype=orig_dtype,
    )
    lr_mod.u_proj.weight.data.copy_(u_r.to("cpu", dtype=orig_dtype))
    lr_mod.v_proj.weight.data.copy_(v_r.to("cpu", dtype=orig_dtype))
    if has_bias and linear.bias is not None:
        lr_mod.u_proj.bias.data.copy_(linear.bias.detach().to("cpu", dtype=orig_dtype))

    return lr_mod, round(err, 6)


class MoECompressionHook:
    """Clean extension hook for TD-MoE expert decomposition & cold-expert pruning."""

    @staticmethod
    def _extract_expert_index(proj_name: str) -> int | None:
        parts = proj_name.split(".")
        for i, p in enumerate(parts):
            if p == "experts" and i + 1 < len(parts) and parts[i + 1].isdigit():
                return int(parts[i + 1])
        return None

    @classmethod
    def adjust_expert_ratio(
        cls,
        proj_name: str,
        base_ratio: float,
        router_probs: list[float] | None,
        prune_threshold: float = 0.01,
    ) -> tuple[float, bool]:
        """Return (adjusted_ratio, should_prune) for an MoE expert projection.

        Hot experts (above uniform traffic) retain more rank; cold experts below
        `prune_threshold` are flagged for pruning or aggressive 2x rank reduction.
        """
        exp_idx = cls._extract_expert_index(proj_name)
        if exp_idx is None or not router_probs or exp_idx >= len(router_probs):
            return base_ratio, False

        n_exp = len(router_probs)
        prob = router_probs[exp_idx]
        if prob < prune_threshold:
            return min(0.85, base_ratio + 0.35), True

        uniform = 1.0 / max(n_exp, 1)
        rel_traffic = prob / uniform
        # Hot experts keep more capacity; colder experts compress further
        adjusted = base_ratio * (1.15 - 0.3 * min(rel_traffic, 2.0))
        return max(0.10, min(0.75, adjusted)), False


def run_decomposition(
    model: nn.Module,
    calib_summary: dict[str, Any],
    ratio: float = 0.35,
    method: str = "whitened_svd",
    preserve_first: int = 1,
    preserve_last: int = 1,
    device: str = "cuda",
    moe_expert_prune_threshold: float = 0.01,
    align_multiple: int = 64,
    allocation: str = "energy_optimal",
) -> dict[str, Any]:
    """Compress transformer blocks layer-by-layer on CUDA without exceeding 16GB VRAM."""
    t0 = time.perf_counter()
    if align_multiple < 1:
        raise ValueError("align_multiple must be >= 1")
    if allocation not in ("energy_optimal", "uniform"):
        raise ValueError(f"Unsupported allocation strategy: {allocation}")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    arch = detect_architecture(model)
    layers: nn.ModuleList = arch["components"]["layers"]
    num_layers = len(layers)
    cache_dir = Path(calib_summary["calib_cache_dir"])
    router_stats: dict[int, list[float]] = calib_summary.get("moe_router_stats", {})

    params_before = count_parameters(model)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    peak_vram = get_vram_stats_gb()["peak_gb"]
    peak_ram = get_ram_rss_gb()

    layer_reports: list[dict[str, Any]] = []
    all_errors: list[float] = []
    moe_pruned_experts = 0
    moe_compressed_experts = 0

    backbone_prefix = arch.get("layers_prefix", "model.layers")

    alloc_result: dict[str, Any] | None = None
    if allocation == "energy_optimal" and 0.0 < ratio < 1.0:
        profile_specs: list[dict[str, Any]] = []
        for layer_idx, layer in enumerate(layers):
            preserved = (layer_idx < preserve_first) or (layer_idx >= num_layers - preserve_last)
            if preserved:
                continue
            target_names = get_target_linear_names(layer)
            cov_path = cache_dir / f"layer_{layer_idx:03d}.pt"
            layer_factors = _load_layer_whitening_factors(cov_path, device=device) if method != "mpo" else {}
            layer_router_probs = router_stats.get(layer_idx)

            for name in target_names:
                mod = layer.get_submodule(name)
                if not isinstance(mod, nn.Linear):
                    continue
                out_f, in_f = mod.weight.shape
                bias_p = mod.bias.numel() if mod.bias is not None else 0
                moe_eff_ratio: float | None = None
                if arch["is_moe"] and ".experts." in name:
                    moe_eff_ratio, _ = MoECompressionHook.adjust_expert_ratio(
                        name, ratio, layer_router_probs, moe_expert_prune_threshold
                    )
                wf = layer_factors.get(name) if method != "mpo" else None
                eigvals, cost_per_rank, max_rank_cap = _profile_projection_spectrum(
                    mod, wf, method=method, device=device
                )
                profile_specs.append(
                    {
                        "key": (layer_idx, name),
                        "layer_idx": layer_idx,
                        "num_layers": num_layers,
                        "proj_name": name,
                        "in_features": int(in_f),
                        "out_features": int(out_f),
                        "bias_params": int(bias_p),
                        "cost_per_rank": int(cost_per_rank),
                        "max_rank_cap": int(max_rank_cap),
                        "energy_spectrum": eigvals,
                        "moe_eff_ratio": moe_eff_ratio,
                    }
                )
                peak_vram = max(peak_vram, get_vram_stats_gb()["peak_gb"])

            del layer_factors
            peak_ram = max(peak_ram, get_ram_rss_gb())
            release_cuda()

        alloc_result = allocate_energy_optimal_ranks(
            profile_specs,
            ratio=ratio,
            align_multiple=align_multiple,
        )
        del profile_specs
        comp_before = max(alloc_result["compressible_params_before"], 1)
        comp_after = alloc_result["compressible_params_after"]
        print(
            f"  [Stage 2] Energy-optimal budget allocated (align={align_multiple}) | "
            f"Compressible: {comp_before:,} -> {comp_after:,} "
            f"(-{(1.0 - comp_after / comp_before) * 100:.2f}% vs target {ratio * 100:.1f}%) | "
            f"{time.perf_counter() - t0:.1f}s",
            flush=True,
        )

    for layer_idx, layer in enumerate(layers):
        preserved = (layer_idx < preserve_first) or (layer_idx >= num_layers - preserve_last)
        target_names = get_target_linear_names(layer)
        proj_reports: dict[str, Any] = {}
        cov_path = cache_dir / f"layer_{layer_idx:03d}.pt"

        if preserved:
            for name in target_names:
                mod = layer.get_submodule(name)
                p_count = mod.weight.numel() + (mod.bias.numel() if mod.bias is not None else 0)
                proj_reports[name] = {
                    "full_name": f"{backbone_prefix}.{layer_idx}.{name}",
                    "status": "preserved",
                    "original_params": int(p_count),
                    "compressed_params": int(p_count),
                }
            layer_reports.append(
                {
                    "layer_idx": layer_idx,
                    "status": "preserved",
                    "projections": proj_reports,
                }
            )
            if cov_path.exists():
                cov_path.unlink(missing_ok=True)
            continue

        layer_factors = _load_layer_whitening_factors(cov_path, device=device) if method != "mpo" else {}
        layer_router_probs = router_stats.get(layer_idx)

        for name in target_names:
            mod = layer.get_submodule(name)
            if not isinstance(mod, nn.Linear):
                continue

            out_f, in_f = mod.weight.shape
            eff_ratio = ratio
            is_pruned_expert = False
            if arch["is_moe"] and ".experts." in name:
                eff_ratio, is_pruned_expert = MoECompressionHook.adjust_expert_ratio(
                    name, ratio, layer_router_probs, moe_expert_prune_threshold
                )
                if is_pruned_expert:
                    moe_pruned_experts += 1
                else:
                    moe_compressed_experts += 1

            orig_p = mod.weight.numel() + (mod.bias.numel() if mod.bias is not None else 0)
            proj_meta = alloc_result["meta"].get((layer_idx, name), {}) if alloc_result is not None else {}

            if alloc_result is not None:
                rank = alloc_result["ranks"].get((layer_idx, name))
            else:
                rank = compute_target_rank(out_f, in_f, eff_ratio, align_multiple=align_multiple)

            if method == "mpo":
                i1, i2 = _best_two_factors(in_f)
                o1, o2 = _best_two_factors(out_f)
                cost_per_rank = (o1 * i1 + o2 * i2) if min(i1, i2, o1, o2) >= 4 else (out_f + in_f)
            else:
                cost_per_rank = out_f + in_f

            invalid_or_no_savings = (
                rank is None
                or rank < 1
                or (align_multiple > 1 and rank % align_multiple != 0)
                or (rank * cost_per_rank >= out_f * in_f)
            )
            if invalid_or_no_savings:
                proj_reports[name] = {
                    "full_name": f"{backbone_prefix}.{layer_idx}.{name}",
                    "status": "skipped_no_savings",
                    "original_params": int(orig_p),
                    "compressed_params": int(orig_p),
                }
                continue

            wf = layer_factors.get(name) if method != "mpo" else None
            if method == "mpo":
                w_dev = mod.weight.detach().to(device=device)
                b_dev = mod.bias.detach().to(device=device) if mod.bias is not None else None
                mpo_mod = MPOLinear.from_dense_weight(
                    w_dev,
                    b_dev,
                    eff_ratio,
                    bond_dim=rank,
                    align_multiple=align_multiple,
                )
                inner_dim = int(mpo_mod._bond_dim) if isinstance(mpo_mod, MPOLinear) else int(mpo_mod.rank)
                if mpo_mod.compressed_params >= orig_p or (align_multiple > 1 and inner_dim % align_multiple != 0):
                    del w_dev, b_dev, mpo_mod
                    proj_reports[name] = {
                        "full_name": f"{backbone_prefix}.{layer_idx}.{name}",
                        "status": "skipped_no_savings",
                        "original_params": int(orig_p),
                        "compressed_params": int(orig_p),
                    }
                    continue
                rel_err = float(
                    torch.linalg.norm(
                        (mpo_mod.reconstruct_weight().float() if isinstance(mpo_mod, MPOLinear) else (mpo_mod.u_proj.weight.float() @ mpo_mod.v_proj.weight.float()))
                        - w_dev.float()
                    ).item()
                    / (torch.linalg.norm(w_dev.float()).item() + 1e-8)
                )
                mpo_mod = mpo_mod.to("cpu")
                replace_submodule(layer, name, mpo_mod)
                new_p = mpo_mod.compressed_params
                spec: dict[str, Any] = {
                    "full_name": f"{backbone_prefix}.{layer_idx}.{name}",
                    "status": "compressed",
                    "kind": "mpo" if isinstance(mpo_mod, MPOLinear) else "low_rank",
                    "in_features": int(in_f),
                    "out_features": int(out_f),
                    "original_params": int(orig_p),
                    "compressed_params": int(new_p),
                    "weighted_rel_error": round(rel_err, 6),
                }
                if isinstance(mpo_mod, MPOLinear):
                    spec["bond_dim"] = int(mpo_mod._bond_dim)
                    spec["in_factors"] = list(mpo_mod.in_factors)
                    spec["out_factors"] = list(mpo_mod.out_factors)
                else:
                    spec["rank"] = int(mpo_mod.rank)
                if proj_meta:
                    spec.update(proj_meta)
                del w_dev, b_dev
            else:
                lr_mod, rel_err = whitened_svd_factorize(
                    mod,
                    None,
                    rank,
                    device=device,
                    whitening_factor=wf,
                    align_multiple=align_multiple,
                )
                replace_submodule(layer, name, lr_mod)
                new_p = lr_mod.compressed_params
                spec = {
                    "full_name": f"{backbone_prefix}.{layer_idx}.{name}",
                    "status": "compressed",
                    "kind": "low_rank",
                    "in_features": int(in_f),
                    "out_features": int(out_f),
                    "rank": int(lr_mod.rank),
                    "original_params": int(orig_p),
                    "compressed_params": int(new_p),
                    "weighted_rel_error": rel_err,
                    "moe_cold_pruned": is_pruned_expert,
                }
                if proj_meta:
                    spec.update(proj_meta)

            all_errors.append(rel_err)
            proj_reports[name] = spec
            peak_vram = max(peak_vram, get_vram_stats_gb()["peak_gb"])

        peak_ram = max(peak_ram, get_ram_rss_gb())
        del layer_factors
        if cov_path.exists():
            cov_path.unlink(missing_ok=True)
        layer.to("cpu")
        release_cuda()

        if (layer_idx + 1) % 4 == 0 or (layer_idx + 1) == num_layers:
            mean_e = sum(all_errors) / max(len(all_errors), 1)
            print(
                f"  [Stage 2] Layer {layer_idx + 1:02d}/{num_layers:02d} | "
                f"Mean Err={mean_e:.4f} | Peak VRAM={peak_vram:.3f} GB | {time.perf_counter() - t0:.1f}s",
                flush=True,
            )

        layer_reports.append(
            {
                "layer_idx": layer_idx,
                "status": "compressed",
                "projections": proj_reports,
            }
        )

    try:
        if cache_dir.exists() and not any(cache_dir.iterdir()):
            cache_dir.rmdir()
    except Exception:
        pass

    params_after = count_parameters(model)
    actual_ratio = 1.0 - (params_after["total"] / max(params_before["total"], 1))
    block_ratio = 1.0 - (
        params_after["transformer_blocks"] / max(params_before["transformer_blocks"], 1)
    )
    comp_before_total = (
        alloc_result["compressible_params_before"]
        if alloc_result is not None
        else sum(
            p.get("original_params", 0)
            for l_rep in layer_reports
            for p in l_rep.get("projections", {}).values()
            if p.get("status") == "compressed"
        )
    )
    comp_after_total = (
        alloc_result["compressible_params_after"]
        if alloc_result is not None
        else sum(
            p.get("compressed_params", 0)
            for l_rep in layer_reports
            for p in l_rep.get("projections", {}).values()
            if p.get("status") == "compressed"
        )
    )
    comp_ratio = 1.0 - (comp_after_total / max(comp_before_total, 1)) if comp_before_total > 0 else 0.0

    return {
        "schema_version": "1.0",
        "method": method,
        "ratio_requested": ratio,
        "align_multiple": int(align_multiple),
        "allocation_mode": allocation,
        "preserve_first": preserve_first,
        "preserve_last": preserve_last,
        "architecture": {
            "model_type": arch["model_type"],
            "architecture_class": arch["architecture_class"],
            "is_moe": arch["is_moe"],
            "num_experts": arch["num_experts"],
            "num_layers": num_layers,
            "hidden_size": arch["hidden_size"],
        },
        "params_before": params_before,
        "params_after": params_after,
        "compressible_params_before": int(comp_before_total),
        "compressible_params_after": int(comp_after_total),
        "actual_compressible_reduction": round(comp_ratio, 4),
        "actual_total_reduction": round(actual_ratio, 4),
        "actual_block_reduction": round(block_ratio, 4),
        "mean_weighted_rel_error": round(sum(all_errors) / max(len(all_errors), 1), 6),
        "peak_vram_gb": round(peak_vram, 4),
        "peak_ram_gb": round(peak_ram, 4),
        "duration_sec": round(time.perf_counter() - t0, 2),
        "moe_summary": {
            "is_moe": arch["is_moe"],
            "compressed_expert_projections": moe_compressed_experts,
            "cold_pruned_expert_projections": moe_pruned_experts,
        },
        "layers": layer_reports,
    }
