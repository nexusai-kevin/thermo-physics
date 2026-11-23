"""Stage 2: Data-aware Whitened SVD / MPO tensor factorization + MoE compression hook."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from common.model_utils import (
    LowRankLinear,
    MPOLinear,
    count_parameters,
    detect_architecture,
    get_ram_rss_gb,
    get_target_linear_names,
    get_vram_stats_gb,
    release_cuda,
    replace_submodule,
)


def compute_target_rank(out_features: int, in_features: int, ratio: float) -> int:
    """Compute low-rank bottleneck r achieving ~`ratio` parameter reduction."""
    if ratio <= 0.0:
        return min(out_features, in_features)
    raw_r = int(((1.0 - ratio) * out_features * in_features) / (out_features + in_features))
    # Round to multiple of 8 for tensor-core alignment
    aligned_r = max(8, ((raw_r + 4) // 8) * 8)
    max_r = min(out_features, in_features) - 1
    return min(aligned_r, max_r)


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


@torch.no_grad()
def whitened_svd_factorize(
    linear: nn.Linear,
    cov: torch.Tensor | None,
    rank: int,
    device: str = "cuda",
) -> tuple[LowRankLinear, float]:
    """Factorize nn.Linear via data-aware whitened SVD and return (LowRankLinear, rel_error)."""
    orig_dtype = linear.weight.dtype
    out_f, in_f = linear.weight.shape
    has_bias = linear.bias is not None

    w = linear.weight.detach().to(device=device, dtype=torch.float32)
    if cov is not None:
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
) -> dict[str, Any]:
    """Compress transformer blocks layer-by-layer on CUDA without exceeding 16GB VRAM."""
    t0 = time.perf_counter()
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

        raw_cov = torch.load(cov_path, map_location="cpu", weights_only=True) if cov_path.exists() else {}
        if isinstance(raw_cov, dict) and "_matrices" in raw_cov and "_aliases" in raw_cov:
            matrices = raw_cov["_matrices"]
            aliases = raw_cov["_aliases"]
            layer_covs: dict[str, torch.Tensor] = {
                pname: matrices[gid] for pname, gid in aliases.items() if gid in matrices
            }
        else:
            layer_covs = raw_cov
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

            rank = compute_target_rank(out_f, in_f, eff_ratio)
            orig_p = mod.weight.numel() + (mod.bias.numel() if mod.bias is not None else 0)
            if rank * (out_f + in_f) >= out_f * in_f:
                proj_reports[name] = {
                    "full_name": f"{backbone_prefix}.{layer_idx}.{name}",
                    "status": "skipped_no_savings",
                    "original_params": int(orig_p),
                    "compressed_params": int(orig_p),
                }
                continue

            cov = layer_covs.get(name)
            if method == "mpo":
                w_dev = mod.weight.detach().to(device=device)
                b_dev = mod.bias.detach().to(device=device) if mod.bias is not None else None
                mpo_mod = MPOLinear.from_dense_weight(w_dev, b_dev, eff_ratio)
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
                del w_dev, b_dev
            else:
                lr_mod, rel_err = whitened_svd_factorize(mod, cov, rank, device=device)
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

            all_errors.append(rel_err)
            proj_reports[name] = spec
            peak_vram = max(peak_vram, get_vram_stats_gb()["peak_gb"])

        peak_ram = max(peak_ram, get_ram_rss_gb())
        del layer_covs, raw_cov
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

    return {
        "schema_version": "1.0",
        "method": method,
        "ratio_requested": ratio,
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
