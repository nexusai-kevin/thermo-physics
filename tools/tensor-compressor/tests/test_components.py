"""Unit tests for LowRankLinear, MPOLinear, MoECompressionHook, and calibration dataset."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.calib_data import BUILTIN_EXTRACTION_SAMPLES, split_calib_eval
from common.model_utils import LowRankLinear, MPOLinear
from stages.decompose import MoECompressionHook, whitened_svd_factorize


def test_builtin_dataset_50_samples() -> None:
    assert len(BUILTIN_EXTRACTION_SAMPLES) == 50
    for s in BUILTIN_EXTRACTION_SAMPLES:
        assert s["id"]
        assert isinstance(s["expected"], dict) and len(s["expected"]) >= 3
        assert "Return ONLY JSON" in s["prompt"]
    calib, ev = split_calib_eval(BUILTIN_EXTRACTION_SAMPLES, smoke_test=True)
    assert len(calib) == 16 and len(ev) == 10


def test_low_rank_linear_and_dense_reconstruction() -> None:
    torch.manual_seed(0)
    lin = nn.Linear(64, 128, bias=True)
    cov = torch.eye(64)
    lr, err = whitened_svd_factorize(lin, cov, rank=32, device="cpu")
    assert isinstance(lr, LowRankLinear)
    assert lr.compressed_params < lr.original_params
    assert err < 0.75

    x = torch.randn(2, 5, 64)
    y_lr = lr(x)
    dense = lr.to_dense_linear()
    y_dense = dense(x)
    assert torch.allclose(y_lr, y_dense, atol=1e-5)


def test_mpo_linear_forward_and_reconstruction() -> None:
    torch.manual_seed(1)
    lin = nn.Linear(64, 128, bias=True)
    mpo = MPOLinear.from_dense_weight(lin.weight.data, lin.bias.data, target_ratio=0.35)
    assert isinstance(mpo, MPOLinear)
    assert mpo.compressed_params < mpo.original_params

    x = torch.randn(3, 4, 64)
    y_mpo = mpo(x)
    dense = mpo.to_dense_linear()
    y_dense = dense(x)
    assert y_mpo.shape == (3, 4, 128)
    assert torch.allclose(y_mpo, y_dense, atol=1e-5)


def test_moe_compression_hook() -> None:
    probs = [0.60, 0.35, 0.045, 0.005]
    r_hot, prune_hot = MoECompressionHook.adjust_expert_ratio("mlp.experts.0.gate_proj", 0.35, probs, prune_threshold=0.01)
    r_cold, prune_cold = MoECompressionHook.adjust_expert_ratio("mlp.experts.3.gate_proj", 0.35, probs, prune_threshold=0.01)
    assert not prune_hot and r_hot < 0.35
    assert prune_cold and r_cold > 0.35


def test_hybrid_qwen35_pipeline_roundtrip(tmp_path: Path) -> None:
    from transformers import AutoModelForCausalLM, Qwen3_5TextConfig
    from common.model_utils import load_compressed_model
    from stages.calibrate import run_calibration
    from stages.decompose import run_decomposition
    from stages.export import run_export

    cfg = Qwen3_5TextConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    )
    model = AutoModelForCausalLM.from_config(cfg).to(torch.bfloat16)

    class _DummyTokenizer:
        pad_token_id = 0
        eos_token_id = 1
        eos_token = "</s>"

        def __call__(self, texts, return_tensors="pt", padding=True, truncation=True, max_length=32):
            n = len(texts)
            return {
                "input_ids": torch.randint(2, 200, (n, 16)),
                "attention_mask": torch.ones((n, 16), dtype=torch.long),
            }

        def save_pretrained(self, _dir: str) -> None:
            pass

    tok = _DummyTokenizer()
    samples = BUILTIN_EXTRACTION_SAMPLES[:4]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    calib_sum = run_calibration(model, tok, samples, output_dir=tmp_path, device=device, max_seq_len=32, batch_size=2)
    assert calib_sum["num_layers"] == 4

    manifest = run_decomposition(model, calib_sum, ratio=0.35, preserve_first=1, preserve_last=1, device=device)
    assert manifest["params_after"]["total"] < manifest["params_before"]["total"]

    export_sum = run_export(model, tok, manifest, "dummy/qwen35-hybrid", output_dir=tmp_path)
    reloaded = load_compressed_model(export_sum["compressed_hf_dir"], device=device, dtype=torch.bfloat16)
    out = reloaded(input_ids=torch.randint(2, 200, (1, 12), device=device), use_cache=False)
    assert out.logits.shape == (1, 12, 256)


def test_json_extraction_and_stop_criteria() -> None:
    from stages.benchmark import _StopOnJsonClose, _extract_json_object

    # 1. Strict JSON
    strict_obj, is_strict = _extract_json_object('{"product": "Snowflake", "annual_spend_usd": "144000"}')
    assert is_strict and strict_obj == {"product": "Snowflake", "annual_spend_usd": "144000"}

    # 2. Truncated base & compressed outputs from Qwen3.8-27B run
    base_trunc = '{\n  "product": "Snowflake Enterprise",\n  "account_id": "SNF-9921",\n  "annual_spend_usd": 144000,\n  "renew'
    comp_trunc = '{product": "Snowflake Enterprise", "account_id": "SNF-9921", "annual_spend_usd": 144000, "renewal_date": "2027-'
    keys = ["product", "account_id", "annual_spend_usd", "renewal_date"]

    b_obj, b_strict = _extract_json_object(base_trunc, keys)
    c_obj, c_strict = _extract_json_object(comp_trunc, keys)
    assert not b_strict and b_obj is not None and b_obj["product"] == "Snowflake Enterprise" and b_obj["account_id"] == "SNF-9921"
    assert not c_strict and c_obj is not None and c_obj["product"] == "Snowflake Enterprise" and c_obj["account_id"] == "SNF-9921"

    # 3. _StopOnJsonClose
    class _Tok:
        def __init__(self, s: str) -> None:
            self.s = s

        def decode(self, _ids, skip_special_tokens=True) -> str:
            return self.s

    dummy_ids = torch.zeros((1, 8), dtype=torch.long)
    assert _StopOnJsonClose(_Tok('{"a": 1}'), prompt_len=2)(dummy_ids, torch.zeros((1, 10))) is True
    assert _StopOnJsonClose(_Tok('{"a": 1'), prompt_len=2)(dummy_ids, torch.zeros((1, 10))) is False


def test_base_benchmark_cache_roundtrip(tmp_path: Path, monkeypatch) -> None:
    import stages.benchmark as bm

    monkeypatch.setattr(
        bm,
        "get_base_cache_path",
        lambda mid: tmp_path / f"base_benchmark_{bm._sanitize_model_id(mid)}.json",
    )
    arch = {"id": "org/test-27b", "num_layers": 64}
    metrics = {"total_params": 27000000000, "peak_vram_gb": 56.0, "vertical_accuracy_pct": 100.0}
    saved = bm.save_base_benchmark_cache("org/test-27b", arch, metrics, offload_mode=True)
    assert saved.exists()
    loaded = bm.load_base_benchmark_cache("org/test-27b")
    assert loaded is not None
    assert loaded["base_metrics"]["total_params"] == 27000000000


def test_lead_scraper_infra_exclusion_and_vertical_scoring() -> None:
    scraper_dir = _ROOT.parent / "lead-scraper"
    if str(scraper_dir) not in sys.path:
        sys.path.insert(0, str(scraper_dir))
    import hf_leads

    assert hf_leads.is_excluded_infra_org("orcarouter", "OrcaRouter", "https://orcarouter.ai", "") is True
    assert hf_leads.is_excluded_infra_org("bottlecapai", "BottleCap AI", "https://bottlecapai.com", "") is True
    assert hf_leads.is_excluded_infra_org("acme-quant", "Acme Quant", "https://acme.ai", "Fast LLM inference kernels") is True
    assert hf_leads.is_excluded_infra_org("lexcorp", "LexCorp Legal", "https://lexcorp.com", "Enterprise legal compliance AI") is False

    score, matches = hf_leads.score_vertical_relevance(
        {"org_name": "lexcorp", "fullname": "LexCorp", "website": "https://lexcorp.com", "bio": "Healthcare & legal compliance"},
        "lexcorp/Qwen3-32B-Legal-Contract",
        ["legal", "compliance"],
    )
    assert score >= 3 and "legal" in matches and "compliance" in matches
