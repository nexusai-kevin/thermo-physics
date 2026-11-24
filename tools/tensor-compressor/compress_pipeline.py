"""5-Stage Vertical Tensor Compression & Benchmarking CLI (16GB VRAM + 128GB RAM)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure local imports work regardless of working directory
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from common.calib_data import load_calibration_dataset, split_calib_eval
from common.model_utils import detect_architecture, get_ram_rss_gb, release_cuda
from stages.benchmark import run_benchmark
from stages.calibrate import run_calibration
from stages.decompose import run_decomposition
from stages.export import run_export
from stages.heal import run_checkpoint_healing, run_healing


def _configure_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vertical Tensor Compression 5-Stage Pipeline (16GB VRAM Streaming)"
    )
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local path")
    parser.add_argument(
        "--calib-data",
        "--dataset",
        dest="calib_data",
        default=None,
        help="Optional path to JSONL vertical calibration dataset (defaults to built-in 50-sample set)",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.35,
        help="Target parameter compression ratio across decomposed linear layers (default: 0.35)",
    )
    parser.add_argument(
        "--align-multiple",
        type=int,
        default=64,
        help="Hardware rank alignment multiple for fused 4-bit CUDA kernels (default: 64)",
    )
    parser.add_argument(
        "--allocation",
        choices=["energy_optimal", "uniform"],
        default="energy_optimal",
        help="Rank budget allocation strategy across layers (default: energy_optimal)",
    )
    parser.add_argument(
        "--method",
        choices=["whitened_svd", "mpo"],
        default="whitened_svd",
        help="Tensor decomposition method (default: whitened_svd)",
    )
    parser.add_argument("--preserve-first", type=int, default=1, help="Number of leading blocks to preserve")
    parser.add_argument("--preserve-last", type=int, default=1, help="Number of trailing blocks to preserve")
    parser.add_argument("--heal", action="store_true", help="Run optional Stage 3 LoRA healing pass")
    parser.add_argument("--heal-steps", type=int, default=20, help="Number of LoRA healing micro-steps")
    parser.add_argument(
        "--output-dir",
        default=str(_THIS_DIR / "output"),
        help="Directory for compressed checkpoint, deploy scripts, and benchmark artifacts",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run fast verification subset (16 calib / 10 eval samples, shorter sequence length)",
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Skip Stages 1-4 and run Stage 5 benchmark on an existing compressed_hf/ checkpoint in --output-dir",
    )
    parser.add_argument(
        "--heal-only",
        action="store_true",
        help="Run Stage 3 QLoRA checkpoint healing on an existing compressed_hf/ checkpoint in --output-dir",
    )
    parser.add_argument(
        "--fast-eval",
        action="store_true",
        help="Cap Stage 5 generation eval to 4 held-out prompts (while keeping full 10-sample perplexity)",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=None,
        help="Explicit number of held-out prompts for Stage 5 generation eval",
    )
    parser.add_argument(
        "--full-eval",
        action="store_true",
        help="Run Stage 5 generation eval on all held-out prompts even in CPU offload mode",
    )
    parser.add_argument(
        "--no-base-cache",
        action="store_true",
        help="Disable reusing cached base model benchmark metrics",
    )
    parser.add_argument(
        "--base-cache-id",
        default=None,
        help="Optional base model ID to look up cached baseline for a derivative/fine-tune",
    )
    return parser.parse_args()


def main() -> int:
    _configure_utf8_console()
    args = parse_args()
    t_pipeline = time.perf_counter()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else (
        torch.float16 if device == "cuda" else torch.float32
    )

    print("=" * 72)
    print("VERTICAL TENSOR COMPRESSION // 5-STAGE PIPELINE")
    print(f"Model:       {args.model}")
    print(f"Device:      {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}) | dtype={dtype}")
    print(
        f"Config:      method={args.method}, ratio={args.ratio:.2f}, align={args.align_multiple}, "
        f"allocation={args.allocation}, heal={args.heal}, smoke_test={args.smoke_test}"
    )
    print(f"Output Dir:  {out_dir}")
    print("=" * 72)

    samples = load_calibration_dataset(args.calib_data)
    calib_samples, eval_samples = split_calib_eval(samples, smoke_test=args.smoke_test)
    max_seq_len = 160 if args.smoke_test else 768
    print(f"[Dataset] Loaded {len(samples)} samples -> {len(calib_samples)} calib, {len(eval_samples)} eval")

    if args.heal_only:
        print("\n[Stage 3/5] Running checkpoint QLoRA healing on existing compressed_hf/...")
        heal_steps = min(args.heal_steps, 6) if args.smoke_test else args.heal_steps
        heal_summary = run_checkpoint_healing(
            compressed_hf_dir=out_dir / "compressed_hf",
            calib_samples=calib_samples,
            steps=heal_steps,
            max_seq_len=max_seq_len,
            device=device,
        )
        print(
            f"[Stage 3/5] Healed in {heal_summary['duration_sec']}s ({heal_summary['steps']} steps, "
            f"{heal_summary['merged_tensors']} tensors merged) | "
            f"Loss: {heal_summary['loss_before']} -> {heal_summary['loss_after']}"
        )
        if not args.benchmark_only:
            return 0

    if args.benchmark_only:
        import json
        import re as _re
        hf_dir = out_dir / "compressed_hf"
        manifest = json.loads((hf_dir / "compression_manifest.json").read_text(encoding="utf-8"))
        heal_summary = manifest.get("heal_summary", {"executed": False, "steps": 0})
        slug = _re.sub(r"[^a-zA-Z0-9._-]+", "-", args.model).strip("-").lower()
        est_b = manifest["params_after"]["total"] / 1e9
        hf_size_gb = round(sum(f.stat().st_size for f in hf_dir.rglob("*") if f.is_file()) / (1024**3), 4)
        export_summary = {
            "compressed_hf_dir": str(hf_dir),
            "compressed_hf_size_gb": hf_size_gb,
            "estimated_q4_km_gguf_gb": round(est_b * 0.58 + 0.5, 2),
            "export_gguf_sh": str(out_dir / "deploy" / "export_gguf.sh"),
            "serve_vllm_sh": str(out_dir / "deploy" / "serve_vllm.sh"),
        }
    else:
        print("\n[Init] Loading base model into CPU system RAM (layer-by-layer streaming ready)...")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        arch = detect_architecture(model)
        print(
            f"[Init] Architecture={arch['architecture_class']} ({'MoE' if arch['is_moe'] else 'Dense'}), "
            f"layers={arch['num_layers']}, host_ram={get_ram_rss_gb():.2f} GB"
        )

        # Stage 1: Calibration
        print("\n[Stage 1/5] Running layer-by-layer streaming activation calibration...")
        calib_summary = run_calibration(
            model=model,
            tokenizer=tokenizer,
            calib_samples=calib_samples,
            output_dir=out_dir,
            device=device,
            max_seq_len=max_seq_len,
            batch_size=1,
        )
        print(
            f"[Stage 1/5] Done in {calib_summary['duration_sec']}s | "
            f"Peak VRAM={calib_summary['peak_vram_gb']:.3f} GB | Peak RAM={calib_summary['peak_ram_gb']:.2f} GB"
        )

        # Stage 2: Data-Aware Tensor/SVD Decomposition
        print("\n[Stage 2/5] Running data-aware tensor decomposition...")
        manifest = run_decomposition(
            model=model,
            calib_summary=calib_summary,
            ratio=args.ratio,
            method=args.method,
            preserve_first=args.preserve_first,
            preserve_last=args.preserve_last,
            device=device,
            align_multiple=args.align_multiple,
            allocation=args.allocation,
        )
        print(
            f"[Stage 2/5] Done in {manifest['duration_sec']}s | "
            f"Params: {manifest['params_before']['total']:,} -> {manifest['params_after']['total']:,} "
            f"(-{manifest['actual_total_reduction']*100:.1f}%, align={manifest['align_multiple']}, {manifest['allocation_mode']}) | "
            f"Mean SVD Err={manifest['mean_weighted_rel_error']:.4f} | Peak VRAM={manifest['peak_vram_gb']:.3f} GB"
        )

        # Stage 3: Optional Healing
        print("\n[Stage 3/5] Optional QLoRA/LoRA healing...")
        heal_steps = min(args.heal_steps, 6) if args.smoke_test else args.heal_steps
        heal_summary = run_healing(
            model=model,
            tokenizer=tokenizer,
            calib_samples=calib_samples,
            enabled=args.heal,
            steps=heal_steps,
            max_seq_len=max_seq_len,
            device=device,
        )
        if heal_summary["executed"]:
            print(
                f"[Stage 3/5] Healed in {heal_summary['duration_sec']}s ({heal_summary['steps']} steps) | "
                f"Loss: {heal_summary['loss_before']} -> {heal_summary['loss_after']}"
            )
        else:
            print(f"[Stage 3/5] Skipped: {heal_summary['skipped_reason']}")

        # Stage 4: Export HF + GGUF/vLLM Scripts
        print("\n[Stage 4/5] Exporting compressed HF checkpoint and GGUF/vLLM deploy scripts...")
        export_summary = run_export(
            model=model,
            tokenizer=tokenizer,
            manifest=manifest,
            model_id=args.model,
            output_dir=out_dir,
        )
        print(
            f"[Stage 4/5] Done in {export_summary['duration_sec']}s | "
            f"HF size={export_summary['compressed_hf_size_gb']:.3f} GB | "
            f"Est Q4_K_M={export_summary['estimated_q4_km_gguf_gb']:.2f} GB"
        )

        # Release in-memory model before Stage 5 isolated benchmarking
        del model
        release_cuda()

    # Stage 5: Benchmark & 1-Page Executive Dossier
    print("\n[Stage 5/5] Benchmarking Base vs. Compressed & rendering executive dossier...")
    bench = run_benchmark(
        base_model_id=args.model,
        compressed_hf_dir=export_summary["compressed_hf_dir"],
        eval_samples=eval_samples,
        calib_samples=calib_samples,
        manifest=manifest,
        heal_summary=heal_summary,
        export_summary=export_summary,
        output_dir=out_dir,
        device=device,
        dtype=dtype,
        use_base_cache=not args.no_base_cache,
        base_cache_model_id=args.base_cache_id,
        fast_eval=args.fast_eval,
        eval_samples_limit=args.eval_samples,
        full_eval=args.full_eval,
    )
    res = bench["results"]
    deltas = res["deltas"]
    roi = res["roi_estimate"]

    print("\n" + "=" * 72)
    print("PIPELINE COMPLETE")
    print(f"Total Runtime:       {time.perf_counter() - t_pipeline:.2f}s")
    print(f"Parameter Reduction: -{deltas['param_reduction_pct']:.2f}%")
    print(f"Peak VRAM Delta:     {res['base_metrics']['peak_vram_gb']:.3f} GB -> {res['compressed_metrics']['peak_vram_gb']:.3f} GB ({-deltas['vram_reduction_pct']:+.1f}%)")
    print(f"Throughput:          {res['base_metrics']['throughput_tok_s']:.1f} -> {res['compressed_metrics']['throughput_tok_s']:.1f} tok/s ({deltas['throughput_speedup_x']:.2f}x)")
    print(f"Vertical Accuracy:   {res['base_metrics']['vertical_accuracy_pct']:.1f}% -> {res['compressed_metrics']['vertical_accuracy_pct']:.1f}% (retention {deltas['accuracy_retention_pct']:.1f}%)")
    print(f"Est. Annual Savings: ${roi['annual_savings_usd']:,.0f}/yr ({roi['base_tier']} -> {roi['compressed_tier']})")
    print(f"JSON Artifact:       {bench['benchmark_results_json']}")
    print(f"HTML Dossier:        {bench['benchmark_dossier_html']}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
