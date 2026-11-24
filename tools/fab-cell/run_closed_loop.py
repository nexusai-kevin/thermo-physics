"""Closed-loop Fabrication Cell #1 harness: FluidNCTwin -> Compressed BF16 LLM -> SafetyGate."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, StaticCache

_FAB_ROOT = Path(__file__).resolve().parent
_TC_ROOT = _FAB_ROOT.parent / "tensor-compressor"
for _p in (str(_FAB_ROOT), str(_TC_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.model_utils import load_compressed_model
from data.fab_dataset import build_fab_dataset
from safety.gate import ControllerCommand, GateState, SafetyGate, SafetyGateConfig, validate_whitelisted_gcode
from sim.fluidnc_twin import FluidNCTwin, TelemetryFrame, WorkEnvelope


def _extract_complete_json(text: str) -> str:
    """Return substring up through the first balanced {...} object, or raw text."""
    depth = 0
    saw_open = False
    in_str = False
    escape = False
    for idx, ch in enumerate(text):
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
                    return text[: idx + 1].strip()
    return text.strip()


def _load_eval_rows(dataset_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if dataset_path.exists():
        rows = [
            json.loads(line)
            for line in dataset_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        rows = build_fab_dataset()
    calib = [r for r in rows if r.get("split") == "calib"] or rows[:45]
    ev = [r for r in rows if r.get("split") == "eval"] or rows[45:]
    return calib, ev


class FastBF16JsonDecoder:
    """Low-latency CUDA BF16 greedy JSON decoder using StaticCache + CUDAGraph."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        max_cache_len: int = 512,
        max_new_tokens: int = 48,
        common_prefix: str = "",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_cache_len = max_cache_len
        self.max_new_tokens = max_new_tokens
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.use_cudagraph = self.device.type == "cuda"
        self.prefix_str = common_prefix

        self.cache: StaticCache | None = None
        self.graph: torch.cuda.CUDAGraph | None = None
        self.cur_tok: torch.Tensor | None = None
        self.next_tok: torch.Tensor | None = None
        self.step_pos: torch.Tensor | None = None
        self.prefix_ids_len: int = 0
        self.prefix_kv_snapshot: list[tuple[torch.Tensor, torch.Tensor]] | None = None

        if self.use_cudagraph:
            self._init_cudagraph()

    def _init_cudagraph(self) -> None:
        try:
            self.cache = StaticCache(
                config=self.model.config,
                max_batch_size=1,
                max_cache_len=self.max_cache_len,
                device=str(self.device),
                dtype=self.dtype,
            )
            self.cur_tok = torch.zeros((1, 1), dtype=torch.long, device=self.device)
            self.step_pos = torch.zeros((1,), dtype=torch.long, device=self.device)

            # Prime StaticCache layers with a dummy prefill so all layer tensors are allocated
            dummy_ids = torch.ones((1, 4), dtype=torch.long, device=self.device)
            dummy_pos = torch.arange(4, device=self.device)
            with torch.no_grad():
                _ = self.model(
                    input_ids=dummy_ids,
                    cache_position=dummy_pos,
                    past_key_values=self.cache,
                    use_cache=True,
                )

            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s), torch.no_grad():
                for _ in range(3):
                    self.step_pos.fill_(4)
                    out = self.model(
                        input_ids=self.cur_tok,
                        cache_position=self.step_pos,
                        past_key_values=self.cache,
                        use_cache=True,
                    )
                    self.next_tok = out.logits[:, -1:, :].argmax(dim=-1)
            torch.cuda.current_stream().wait_stream(s)

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g), torch.no_grad():
                out = self.model(
                    input_ids=self.cur_tok,
                    cache_position=self.step_pos,
                    past_key_values=self.cache,
                    use_cache=True,
                )
                self.next_tok = out.logits[:, -1:, :].argmax(dim=-1)
            self.graph = g

            # Snapshot static system prompt prefix KV states if provided
            if self.prefix_str:
                p_ids = self.tokenizer(self.prefix_str, return_tensors="pt")["input_ids"].to(self.device)
                p_len = int(p_ids.shape[1])
                if 4 <= p_len < self.max_cache_len - self.max_new_tokens - 16:
                    self.cache.reset()
                    p_pos = torch.arange(p_len, device=self.device)
                    with torch.no_grad():
                        _ = self.model(
                            input_ids=p_ids,
                            cache_position=p_pos,
                            past_key_values=self.cache,
                            use_cache=True,
                        )
                    self.prefix_kv_snapshot = [
                        (layer.keys[:, :, :p_len, :].clone(), layer.values[:, :, :p_len, :].clone())
                        for layer in self.cache.layers
                    ]
                    self.prefix_ids_len = p_len
        except Exception:
            self.use_cudagraph = False
            self.graph = None
            self.prefix_kv_snapshot = None

    @torch.no_grad()
    def _decode_eager(self, prompt: str) -> tuple[str, int]:
        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_cache_len - self.max_new_tokens).to(self.device)
        out = self.model.generate(
            **enc,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        gen_ids = out[0, enc["input_ids"].shape[1] :]
        text = _extract_complete_json(self.tokenizer.decode(gen_ids, skip_special_tokens=True))
        return text, int(gen_ids.numel())

    @torch.no_grad()
    def decode(self, prompt: str) -> tuple[str, int]:
        if not self.use_cudagraph or self.graph is None or self.cache is None:
            return self._decode_eager(prompt)

        assert self.cur_tok is not None and self.next_tok is not None and self.step_pos is not None

        self.cache.reset()
        if (
            self.prefix_kv_snapshot is not None
            and self.prefix_str
            and prompt.startswith(self.prefix_str)
        ):
            suffix = prompt[len(self.prefix_str) :]
            suf_ids = self.tokenizer(suffix, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)
            p_len = self.prefix_ids_len
            for idx, (k_snap, v_snap) in enumerate(self.prefix_kv_snapshot):
                self.cache.layers[idx].keys[:, :, :p_len, :].copy_(k_snap)
                self.cache.layers[idx].values[:, :, :p_len, :].copy_(v_snap)
            suf_len = int(suf_ids.shape[1])
            suf_pos = torch.arange(p_len, p_len + suf_len, device=self.device)
            out = self.model(
                input_ids=suf_ids,
                cache_position=suf_pos,
                past_key_values=self.cache,
                use_cache=True,
            )
            seq_len = p_len + suf_len
        else:
            enc = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_cache_len - self.max_new_tokens).to(self.device)
            seq_len = int(enc["input_ids"].shape[1])
            c_pos = torch.arange(seq_len, device=self.device)
            out = self.model(
                input_ids=enc["input_ids"],
                cache_position=c_pos,
                past_key_values=self.cache,
                use_cache=True,
            )

        self.cur_tok.copy_(out.logits[:, -1:, :].argmax(dim=-1))
        gen_ids: list[int] = [int(self.cur_tok.item())]
        eos_id = self.tokenizer.eos_token_id

        # Batched 4-step decode between GPU->CPU brace checks to minimize sync overhead
        step = 0
        while len(gen_ids) < self.max_new_tokens:
            batch_steps = min(4, self.max_new_tokens - len(gen_ids))
            for _ in range(batch_steps):
                self.step_pos.fill_(seq_len + step)
                self.graph.replay()
                self.cur_tok.copy_(self.next_tok)
                tid = int(self.cur_tok.item())
                gen_ids.append(tid)
                step += 1
                if eos_id is not None and tid == eos_id:
                    break

            decoded = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            if eos_id is not None and gen_ids[-1] == eos_id:
                break
            if "}" in decoded:
                cand = _extract_complete_json(decoded)
                if cand.startswith("{") and cand.endswith("}"):
                    return cand, len(gen_ids)

        return _extract_complete_json(self.tokenizer.decode(gen_ids, skip_special_tokens=True)), len(gen_ids)

    def verify_equivalence(self, sample_prompt: str) -> None:
        """Verify CUDAGraph + prefix-cache output matches eager generate(); fallback if not."""
        if not self.use_cudagraph:
            return
        fast_txt, _ = self.decode(sample_prompt)
        eager_txt, _ = self._decode_eager(sample_prompt)
        if fast_txt != eager_txt:
            # Disable prefix snapshot first, retry, or fallback to eager
            self.prefix_kv_snapshot = None
            fast_txt2, _ = self.decode(sample_prompt)
            if fast_txt2 != eager_txt:
                self.use_cudagraph = False


def _percentile(vals: list[float], pct: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = min(len(s) - 1, max(0, int(math.ceil((pct / 100.0) * len(s))) - 1))
    return s[idx]


def run_closed_loop(
    checkpoint_dir: Path,
    dataset_path: Path,
    latency_budget_ms: float = 250.0,
    max_new_tokens: int = 48,
    output_json: Path | None = None,
) -> dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    manifest = json.loads((checkpoint_dir / "compression_manifest.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = load_compressed_model(checkpoint_dir, device=device, dtype=dtype)
    model.eval()
    steady_vram_gb = (torch.cuda.memory_allocated(0) / (1024**3)) if device == "cuda" else 0.0

    calib_rows, eval_rows = _load_eval_rows(dataset_path)
    common_prefix = ""
    if eval_rows and "Telemetry:\n" in eval_rows[0]["prompt"]:
        common_prefix = eval_rows[0]["prompt"].split("Telemetry:\n", 1)[0] + "Telemetry:\n"

    decoder = FastBF16JsonDecoder(
        model=model,
        tokenizer=tokenizer,
        max_cache_len=768,
        max_new_tokens=max_new_tokens,
        common_prefix=common_prefix,
    )

    # Warmup passes (and verify CUDAGraph equivalence against eager generate)
    for w_idx in range(3):
        _ = decoder.decode(calib_rows[w_idx % len(calib_rows)]["prompt"])
    decoder.verify_equivalence(calib_rows[0]["prompt"])
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    env = WorkEnvelope()
    gate = SafetyGate(SafetyGateConfig(envelope=env, latency_budget_ms=latency_budget_ms))
    gate.start()

    strict_json_valid = 0
    action_exact_matches = 0
    full_5field_matches = 0
    watchdog_cleared = 0
    unsafe_passthroughs = 0
    latencies_ms: list[float] = []
    total_tokens = 0
    total_sec = 0.0
    records: list[dict[str, Any]] = []

    for row in eval_rows:
        t_dict = row["telemetry"]
        twin = FluidNCTwin(tick_hz=float(t_dict.get("tick_hz", 10.0)), envelope=env)
        twin.pos_mm = dict(t_dict["pos_mm"])
        twin.pos_error_mm = float(t_dict["pos_error_mm"])
        twin.feedrate_mm_min = float(t_dict["feedrate_mm_min"])
        twin.spindle_rpm = float(t_dict["spindle_rpm"])
        twin.spindle_temp_c = float(t_dict["spindle_temp_c"])
        twin.mcu_temp_c = float(t_dict["mcu_temp_c"])
        twin.motor_current_a = float(t_dict["motor_current_a"])
        twin.motor_load_pct = float(t_dict["motor_load_pct"])
        twin.imu_vibration_rms_g = float(t_dict["imu_vibration_rms_g"])
        twin.limit_triggered = bool(t_dict["limit_triggered"])
        twin.limit_pins = str(t_dict["limit_pins"])
        twin.active_gcode = str(t_dict["active_gcode"])
        twin.controller_state = t_dict["controller_state"]
        frame: TelemetryFrame = twin.snapshot()

        last_raw = {"text": "", "n_tok": 0}

        def _infer() -> str:
            if device == "cuda":
                torch.cuda.synchronize()
            txt, n_tok = decoder.decode(row["prompt"])
            if device == "cuda":
                torch.cuda.synchronize()
            last_raw["text"] = txt
            last_raw["n_tok"] = n_tok
            return txt

        decision = gate.guard_inference(_infer, telemetry=frame)
        lat_ms = decision.latency_ms
        latencies_ms.append(lat_ms)
        total_tokens += last_raw["n_tok"]
        total_sec += max(lat_ms / 1000.0, 1e-6)

        if lat_ms <= latency_budget_ms:
            watchdog_cleared += 1

        # Strict JSON + Pydantic check on raw model output
        parsed_raw: ControllerCommand | None = None
        is_strict_valid = False
        try:
            obj = json.loads(last_raw["text"])
            parsed_raw = ControllerCommand.model_validate(obj)
            is_strict_valid = True
        except Exception:
            is_strict_valid = False

        if is_strict_valid:
            strict_json_valid += 1

        exp = row["expected"]
        if parsed_raw is not None:
            core_match = (
                parsed_raw.status == exp["status"]
                and parsed_raw.fault_type == exp["fault_type"]
                and parsed_raw.action == exp["action"]
                and parsed_raw.gcode == exp["gcode"]
            )
            if core_match:
                action_exact_matches += 1
            if core_match and abs(parsed_raw.confidence - float(exp["confidence"])) <= 0.05:
                full_5field_matches += 1

        # Verify SafetyGate invariant: zero unsafe G-code ever reaches FluidNC
        gcode_ok, _ = validate_whitelisted_gcode(decision.emitted_command, env)
        if not gcode_ok:
            unsafe_passthroughs += 1
        twin.apply_command(decision.emitted_command)

        records.append(
            {
                "id": row["id"],
                "expected_fault": exp["fault_type"],
                "expected_action": exp["action"],
                "expected_gcode": exp["gcode"],
                "raw_output": last_raw["text"],
                "strict_json_valid": is_strict_valid,
                "gate_approved": decision.approved,
                "gate_state": decision.state.value,
                "emitted_command": decision.emitted_command,
                "violations": decision.violations,
                "latency_ms": round(lat_ms, 2),
                "n_tokens": last_raw["n_tok"],
            }
        )

        # Reset gate cleanly between independent held-out eval scenarios
        if gate.state == GateState.ESTOP:
            gate.manual_reset(operator_ack=True)
            gate.start()
        elif gate.state == GateState.HOLD:
            gate.resume()

    peak_vram_gb = (torch.cuda.max_memory_allocated(0) / (1024**3)) if device == "cuda" else 0.0
    n_eval = len(eval_rows)
    param_red_pct = round(manifest["actual_total_reduction"] * 100.0, 2)
    p50_ms = round(_percentile(latencies_ms, 50.0), 2)
    p95_ms = round(_percentile(latencies_ms, 95.0), 2)
    max_ms = round(max(latencies_ms) if latencies_ms else 0.0, 2)
    mean_tok_s = round(total_tokens / max(total_sec, 1e-6), 2)

    scorecard = {
        "checkpoint_dir": str(checkpoint_dir),
        "decode_mode": "cudagraph_static_cache" if decoder.use_cudagraph else "eager_bf16",
        "eval_samples": n_eval,
        "param_reduction_pct": param_red_pct,
        "steady_state_bf16_vram_gb": round(steady_vram_gb, 3),
        "peak_bf16_vram_gb": round(peak_vram_gb, 3),
        "strict_json_valid_count": strict_json_valid,
        "strict_json_valid_pct": round(100.0 * strict_json_valid / max(n_eval, 1), 1),
        "fault_to_action_exact_count": action_exact_matches,
        "fault_to_action_exact_pct": round(100.0 * action_exact_matches / max(n_eval, 1), 1),
        "full_5field_match_count": full_5field_matches,
        "unsafe_gcode_passthroughs": unsafe_passthroughs,
        "latency_budget_ms": latency_budget_ms,
        "watchdog_cleared_count": watchdog_cleared,
        "latency_p50_ms": p50_ms,
        "latency_p95_ms": p95_ms,
        "latency_max_ms": max_ms,
        "mean_throughput_tok_s": mean_tok_s,
        "records": records,
    }

    print("\n" + "=" * 72)
    print("FABRICATION CELL #1 // CLOSED-LOOP SCORECARD (RTX 5080 BF16)")
    print("=" * 72)
    print(f"Checkpoint:              {checkpoint_dir}")
    print(f"Decode Path:             {scorecard['decode_mode']}")
    print(f"Parameter Reduction:     -{param_red_pct:.2f}% ({manifest['params_before']['total']:,} -> {manifest['params_after']['total']:,})")
    print(f"Steady-State BF16 VRAM:  {steady_vram_gb:.3f} GB (Peak: {peak_vram_gb:.3f} GB)")
    print(f"Strict JSON Validity:    {scorecard['strict_json_valid_pct']:.1f}% ({strict_json_valid}/{n_eval})")
    print(f"Fault-to-Action Match:   {scorecard['fault_to_action_exact_pct']:.1f}% ({action_exact_matches}/{n_eval})")
    print(f"Unsafe G-Code Passed:    {unsafe_passthroughs}")
    print(f"Decision Latency (ms):   p50={p50_ms:.2f} | p95={p95_ms:.2f} | max={max_ms:.2f} ({mean_tok_s:.1f} tok/s)")
    print(f"Watchdog (<={latency_budget_ms:.0f}ms):      {watchdog_cleared}/{n_eval} cleared")
    print("=" * 72)

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(scorecard, indent=2), encoding="utf-8")

    return scorecard


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Fabrication Cell #1 closed-loop verification.")
    parser.add_argument("--checkpoint", type=Path, default=Path("output/fab-cell-qwen2.5-3b/compressed_hf"))
    parser.add_argument("--dataset", type=Path, default=Path("tools/fab-cell/data/fab_dataset.jsonl"))
    parser.add_argument("--latency-budget-ms", type=float, default=250.0)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--output-json", type=Path, default=Path("output/fab-cell-qwen2.5-3b/closed_loop_scorecard.json"))
    args = parser.parse_args()

    run_closed_loop(
        checkpoint_dir=args.checkpoint,
        dataset_path=args.dataset,
        latency_budget_ms=args.latency_budget_ms,
        max_new_tokens=args.max_new_tokens,
        output_json=args.output_json,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
