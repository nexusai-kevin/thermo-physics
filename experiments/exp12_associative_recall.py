"""
Experiment 12: Associative Memory Recall Benchmark (thrml 0.1.4 Ising & Potts vs. Hopfield)
=========================================================================================

Benchmarks Extropic thrml 0.1.4 Ising and K=4 Potts block-Gibbs samplers against:
1. Classical Hopfield Network (Hebbian outer-product weights, async zero-T updates)
2. Continuous Exponential Hopfield Network (Modern Hopfield CCCP softmax updates)

Task:
- 64-bit (8x8) associative recall across P = 10 stored patterns (load alpha = 10/64 = 0.15625)
- Corrupted-pattern cue noise at 20%, 35%, and 50% (random bit replacement noise,
  corresponding to 10%, 17.5%, and 25% effective bit-flip rates)
- Inverse-temperature annealing schedule optimization & JSON calibration export
- Logs bit-error recovery, spurious attractor rates, convergence steps, and
  execution time into data/benchmark_manifest.json & docs/benchmark_manifest.json.
"""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from thrml import Block, BlockGibbsSpec, SamplingSchedule, SpinNode, sample_states
from thrml.factor import FactorSamplingProgram
from thrml.models import IsingEBM, IsingSamplingProgram
from thrml.models.discrete_ebm import CategoricalEBMFactor, CategoricalGibbsConditional
from thrml.pgm import CategoricalNode


ROOT_DIR = Path(__file__).resolve().parent.parent
CALIBRATION_PATH = ROOT_DIR / "configs" / "exp12_annealing_calibration.json"
MANIFEST_PATHS = [
    ROOT_DIR / "data" / "benchmark_manifest.json",
    ROOT_DIR / "docs" / "benchmark_manifest.json",
]

NOISE_LEVELS = [0.20, 0.35, 0.50]
N_BITS = 64
N_PATTERNS = 10
N_TRIALS_PER_NOISE = 100
CALIBRATION_TRIALS = 40


def hadamard_matrix(n: int) -> np.ndarray:
    """Construct an n x n Sylvester Hadamard matrix (n must be a power of 2)."""
    h = np.array([[1.0]], dtype=np.float32)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h


def generate_patterns(num_patterns: int = N_PATTERNS, n_bits: int = N_BITS, seed: int = 42) -> np.ndarray:
    """Generate 4 geometric 8x8 icons plus (num_patterns - 4) low-overlap random patterns."""
    rng = np.random.default_rng(seed)
    pats: List[np.ndarray] = []

    # 0: Plus
    p0 = -np.ones((8, 8), dtype=np.float32)
    p0[:, 3:5] = 1.0
    p0[3:5, :] = 1.0
    pats.append(p0.flatten())

    # 1: Cross
    p1 = -np.ones((8, 8), dtype=np.float32)
    for r in range(8):
        p1[r, r] = 1.0
        p1[r, 7 - r] = 1.0
    pats.append(p1.flatten())

    # 2: Hollow Square
    p2 = -np.ones((8, 8), dtype=np.float32)
    p2[1:7, 1:7] = 1.0
    p2[2:6, 2:6] = -1.0
    pats.append(p2.flatten())

    # 3: 2x2 Checkerboard
    p3 = np.ones((8, 8), dtype=np.float32)
    for r in range(8):
        for c in range(8):
            if ((r // 2) + (c // 2)) % 2 == 1:
                p3[r, c] = -1.0
    pats.append(p3.flatten())

    for _ in range(num_patterns - 4):
        while True:
            cand = rng.choice([-1.0, 1.0], size=n_bits).astype(np.float32)
            if max(abs(float(cand @ p)) / n_bits for p in pats) < 0.35:
                pats.append(cand)
                break

    return np.array(pats, dtype=np.float32)


def corrupt_cue(pattern: np.ndarray, noise_rate: float, rng: np.random.Generator) -> np.ndarray:
    """Replace each bit independently with a random {-1, +1} bit with probability noise_rate."""
    mask = rng.random(len(pattern)) < noise_rate
    rand_bits = rng.choice([-1.0, 1.0], size=len(pattern)).astype(np.float32)
    return np.where(mask, rand_bits, pattern)


def is_spurious_attractor(state: np.ndarray, patterns: np.ndarray, allow_inverted: bool = True) -> bool:
    """Return True if state does not match any stored pattern (or its negation when symmetric)."""
    for p in patterns:
        if np.array_equal(state, p):
            return False
        if allow_inverted and np.array_equal(state, -p):
            return False
    return True


def evaluate_predictions(
    recovered: np.ndarray,
    targets: np.ndarray,
    patterns: np.ndarray,
    steps: np.ndarray,
    elapsed_s: float,
    allow_inverted: bool = True,
) -> Dict[str, float]:
    """Compute bit-error recovery, BER, exact recall, spurious rate, convergence steps, and timing."""
    bit_recovery = float(np.mean(recovered == targets))
    bit_error_rate = 1.0 - bit_recovery
    exact_recall = float(np.mean(np.all(recovered == targets, axis=1)))
    spurious_count = sum(
        1 for s in recovered if is_spurious_attractor(s, patterns, allow_inverted=allow_inverted)
    )
    spurious_rate = spurious_count / float(len(recovered))
    return {
        "bit_error_recovery": round(bit_recovery, 4),
        "bit_error_rate": round(bit_error_rate, 4),
        "exact_pattern_recall": round(exact_recall, 4),
        "spurious_attractor_rate": round(spurious_rate, 4),
        "mean_convergence_steps": round(float(np.mean(steps)), 2),
        "execution_time_s": round(float(elapsed_s), 5),
        "ms_per_trial": round(1000.0 * float(elapsed_s) / float(len(recovered)), 4),
    }


def run_classical_hopfield(
    cues: np.ndarray,
    patterns: np.ndarray,
    max_sweeps: int = 40,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Classical Hopfield network with Hebbian weights and asynchronous zero-T updates."""
    p_count, n_bits = patterns.shape
    W = (patterns.T @ patterns) / float(n_bits)
    np.fill_diagonal(W, 0.0)

    recovered = np.empty_like(cues)
    steps = np.empty(len(cues), dtype=np.float32)

    t0 = time.perf_counter()
    for idx, cue in enumerate(cues):
        s = cue.copy()
        conv_step = max_sweeps
        for sweep in range(1, max_sweeps + 1):
            changed = False
            for i in range(n_bits):
                h = float(W[i] @ s)
                ns = 1.0 if h > 0.0 else (-1.0 if h < 0.0 else s[i])
                if ns != s[i]:
                    s[i] = ns
                    changed = True
            if not changed:
                conv_step = sweep
                break
        recovered[idx] = s
        steps[idx] = conv_step
    elapsed = time.perf_counter() - t0
    return recovered, steps, elapsed


def run_exponential_hopfield(
    cues: np.ndarray,
    patterns: np.ndarray,
    beta_exp: float = 10.0,
    max_steps: int = 30,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Continuous Exponential Hopfield Network (Modern Hopfield CCCP update)."""
    n_bits = patterns.shape[1]
    recovered = np.empty_like(cues)
    steps = np.empty(len(cues), dtype=np.float32)

    t0 = time.perf_counter()
    for idx, cue in enumerate(cues):
        x = cue.copy()
        conv_step = max_steps
        for step in range(1, max_steps + 1):
            logits = beta_exp * (patterns @ x) / float(n_bits)
            logits -= np.max(logits)
            weights = np.exp(logits)
            weights /= np.sum(weights)
            nx = patterns.T @ weights
            if np.all(np.sign(nx) == np.sign(x)) and np.linalg.norm(nx - x) < 1e-4:
                x = nx
                conv_step = step
                break
            x = nx
        s = np.where(x >= 0.0, 1.0, -1.0).astype(np.float32)
        recovered[idx] = s
        steps[idx] = conv_step
    elapsed = time.perf_counter() - t0
    return recovered, steps, elapsed


class ThrmlIsingAssociativeSampler:
    """Bipartite Ising associative memory using thrml 0.1.4 IsingSamplingProgram."""

    def __init__(self, patterns: np.ndarray, n_hidden: int = 32, weight_scale: float = 4.0):
        self.patterns = patterns
        self.p_count, self.n_vis = patterns.shape
        self.n_hid = n_hidden

        h_codes = hadamard_matrix(n_hidden)[1 : self.p_count + 1]
        xi_pinv = np.linalg.pinv(patterns)
        W = (xi_pinv @ h_codes) * weight_scale

        self.vis_nodes = [SpinNode() for _ in range(self.n_vis)]
        self.hid_nodes = [SpinNode() for _ in range(self.n_hid)]
        self.all_nodes = self.vis_nodes + self.hid_nodes
        self.edges = [(self.vis_nodes[i], self.hid_nodes[j]) for i in range(self.n_vis) for j in range(self.n_hid)]
        self.weights = jnp.array([float(W[i, j]) for i in range(self.n_vis) for j in range(self.n_hid)], jnp.float32)
        self.biases = jnp.zeros(self.n_vis + self.n_hid, jnp.float32)
        self.free_blocks = [Block(self.hid_nodes), Block(self.vis_nodes)]
        self._compiled_stages: Dict[Tuple[float, int], Any] = {}

    def _get_stage_fn(self, beta_val: float, sweeps_per_stage: int):
        key_id = (round(float(beta_val), 6), int(sweeps_per_stage))
        if key_id not in self._compiled_stages:
            ebm = IsingEBM(
                nodes=self.all_nodes,
                edges=self.edges,
                biases=self.biases,
                weights=self.weights,
                beta=jnp.array(beta_val, jnp.float32),
            )
            prog = IsingSamplingProgram(ebm, free_blocks=self.free_blocks, clamped_blocks=[])
            sched = SamplingSchedule(n_warmup=sweeps_per_stage, n_samples=1, steps_per_sample=1)
            free_blocks = self.free_blocks

            def step_fn(k, sh, sv):
                res = sample_states(k, prog, sched, [sh, sv], [], free_blocks)
                return res[0][-1], res[1][-1]

            self._compiled_stages[key_id] = jax.jit(jax.vmap(step_fn))
        return self._compiled_stages[key_id]

    def warmup(self, betas: List[float], sweeps_per_stage: int, batch_size: int) -> float:
        t0 = time.perf_counter()
        sv = jnp.zeros((batch_size, self.n_vis), jnp.bool_)
        sh = jnp.zeros((batch_size, self.n_hid), jnp.bool_)
        keys = jax.random.split(jax.random.key(0), batch_size)
        for b_val in betas:
            fn = self._get_stage_fn(b_val, sweeps_per_stage)
            sh, sv = fn(keys, sh, sv)
            jax.block_until_ready(sv)
        return time.perf_counter() - t0

    def sample(
        self,
        cues: np.ndarray,
        betas: List[float],
        sweeps_per_stage: int = 2,
        seed: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        batch_size = len(cues)
        sv = jnp.array(cues > 0.0, jnp.bool_)
        sh = jnp.zeros((batch_size, self.n_hid), jnp.bool_)
        key = jax.random.key(seed)

        trajectory = [np.array(cues, dtype=np.float32)]
        t0 = time.perf_counter()
        for b_val in betas:
            fn = self._get_stage_fn(b_val, sweeps_per_stage)
            key, sk = jax.random.split(key)
            keys = jax.random.split(sk, batch_size)
            sh, sv = fn(keys, sh, sv)
            jax.block_until_ready(sv)
            trajectory.append(2.0 * np.array(sv, dtype=np.float32) - 1.0)
        elapsed = time.perf_counter() - t0

        final_states = trajectory[-1]
        conv_steps = _estimate_convergence_steps(trajectory, sweeps_per_stage)
        return final_states, conv_steps, elapsed


class ThrmlPottsAssociativeSampler:
    """Bipartite K=4 Potts associative memory using thrml 0.1.4 FactorSamplingProgram."""

    def __init__(self, patterns: np.ndarray, n_hidden: int = 24, weight_scale: float = 1.5, seed: int = 777):
        self.patterns = patterns
        self.p_count, self.n_bits = patterns.shape
        self.n_vis = self.n_bits // 2
        self.n_hid = n_hidden
        self.k_cats = 4

        xi_potts = self.bits_to_potts(patterns)
        rng_h = np.random.default_rng(seed)
        h_codes = rng_h.integers(0, self.k_cats, size=(self.p_count, self.n_hid))

        W_potts = np.zeros((self.n_vis, self.n_hid, self.k_cats, self.k_cats), dtype=np.float32)
        for m in range(self.p_count):
            for i in range(self.n_vis):
                va = np.full(self.k_cats, -1.0 / self.k_cats, dtype=np.float32)
                va[xi_potts[m, i]] += 1.0
                for h in range(self.n_hid):
                    vb = np.full(self.k_cats, -1.0 / self.k_cats, dtype=np.float32)
                    vb[h_codes[m, h]] += 1.0
                    W_potts[i, h] += np.outer(va, vb)
        W_potts *= weight_scale

        pv = [CategoricalNode() for _ in range(self.n_vis)]
        ph = [CategoricalNode() for _ in range(self.n_hid)]
        u = [pv[i] for i in range(self.n_vis) for _ in range(self.n_hid)]
        v = [ph[h] for _ in range(self.n_vis) for h in range(self.n_hid)]
        self.edge_blocks = [Block(u), Block(v)]
        self.w_flat = jnp.array(W_potts.reshape(self.n_vis * self.n_hid, self.k_cats, self.k_cats), jnp.float32)
        self.free_blocks = [Block(ph), Block(pv)]
        self.conds = [CategoricalGibbsConditional(n_categories=self.k_cats) for _ in self.free_blocks]
        self.spec = BlockGibbsSpec(self.free_blocks, clamped_blocks=[])
        self._compiled_stages: Dict[Tuple[float, int], Any] = {}

    @staticmethod
    def bits_to_potts(bits: np.ndarray) -> np.ndarray:
        n_vis = bits.shape[-1] // 2
        b = (bits > 0.0).astype(np.uint8).reshape(*bits.shape[:-1], n_vis, 2)
        return b[..., 0] * 2 + b[..., 1]

    @staticmethod
    def potts_to_bits(cats: np.ndarray) -> np.ndarray:
        b0 = (cats // 2).astype(np.float32) * 2.0 - 1.0
        b1 = (cats % 2).astype(np.float32) * 2.0 - 1.0
        return np.stack([b0, b1], axis=-1).reshape(*cats.shape[:-1], cats.shape[-1] * 2)

    def _get_stage_fn(self, beta_val: float, sweeps_per_stage: int):
        key_id = (round(float(beta_val), 6), int(sweeps_per_stage))
        if key_id not in self._compiled_stages:
            fac = CategoricalEBMFactor(self.edge_blocks, float(beta_val) * self.w_flat)
            prog = FactorSamplingProgram(self.spec, self.conds, [fac], other_interaction_groups=[])
            sched = SamplingSchedule(n_warmup=sweeps_per_stage, n_samples=1, steps_per_sample=1)
            free_blocks = self.free_blocks

            def step_fn(k, kh, kv):
                res = sample_states(k, prog, sched, [kh, kv], [], free_blocks)
                return res[0][-1], res[1][-1]

            self._compiled_stages[key_id] = jax.jit(jax.vmap(step_fn))
        return self._compiled_stages[key_id]

    def warmup(self, betas: List[float], sweeps_per_stage: int, batch_size: int) -> float:
        t0 = time.perf_counter()
        sv = jnp.zeros((batch_size, self.n_vis), jnp.uint8)
        sh = jnp.zeros((batch_size, self.n_hid), jnp.uint8)
        keys = jax.random.split(jax.random.key(0), batch_size)
        for b_val in betas:
            fn = self._get_stage_fn(b_val, sweeps_per_stage)
            sh, sv = fn(keys, sh, sv)
            jax.block_until_ready(sv)
        return time.perf_counter() - t0

    def sample(
        self,
        cues: np.ndarray,
        betas: List[float],
        sweeps_per_stage: int = 2,
        seed: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        batch_size = len(cues)
        sv = jnp.array(self.bits_to_potts(cues), jnp.uint8)
        sh = jnp.zeros((batch_size, self.n_hid), jnp.uint8)
        key = jax.random.key(seed)

        trajectory = [np.array(cues, dtype=np.float32)]
        t0 = time.perf_counter()
        for b_val in betas:
            fn = self._get_stage_fn(b_val, sweeps_per_stage)
            key, sk = jax.random.split(key)
            keys = jax.random.split(sk, batch_size)
            sh, sv = fn(keys, sh, sv)
            jax.block_until_ready(sv)
            trajectory.append(self.potts_to_bits(np.array(sv)))
        elapsed = time.perf_counter() - t0

        final_states = trajectory[-1]
        conv_steps = _estimate_convergence_steps(trajectory, sweeps_per_stage)
        return final_states, conv_steps, elapsed


def _estimate_convergence_steps(trajectory: List[np.ndarray], sweeps_per_stage: int) -> np.ndarray:
    """Find the earliest sweep index after which the state matches the final state."""
    final = trajectory[-1]
    batch_size = len(final)
    conv = np.full(batch_size, (len(trajectory) - 1) * sweeps_per_stage, dtype=np.float32)
    for b in range(batch_size):
        for stage_idx in range(1, len(trajectory)):
            if np.array_equal(trajectory[stage_idx][b], final[b]):
                conv[b] = stage_idx * sweeps_per_stage
                break
    return conv


def build_candidate_schedules() -> Dict[str, List[float]]:
    """Define candidate inverse-temperature annealing schedules beta(t)."""
    n_stages = 6
    t = np.linspace(0.0, 1.0, n_stages)
    return {
        "constant_cold": [round(float(x), 4) for x in np.full(n_stages, 4.5)],
        "linear_warm_to_cold": [round(float(x), 4) for x in np.linspace(2.0, 7.5, n_stages)],
        "geometric_anneal": [round(float(x), 4) for x in np.geomspace(2.0, 8.0, n_stages)],
        "sigmoid_quench": [
            round(float(x), 4) for x in (2.2 + 5.6 / (1.0 + np.exp(-6.0 * (t - 0.45))))
        ],
    }


def calibrate_annealing_schedule(
    ising_sampler: ThrmlIsingAssociativeSampler,
    potts_sampler: ThrmlPottsAssociativeSampler,
    patterns: np.ndarray,
    sweeps_per_stage: int = 2,
) -> Tuple[str, List[float], Dict[str, Any]]:
    """Sweep candidate beta(t) schedules on validation cues (35% & 50% noise) to pick optimal schedule."""
    rng = np.random.default_rng(2026)
    val_cues = []
    val_targets = []
    for noise in [0.35, 0.50]:
        for t in range(CALIBRATION_TRIALS):
            p = patterns[t % len(patterns)]
            val_cues.append(corrupt_cue(p, noise, rng))
            val_targets.append(p)
    val_cues_arr = np.array(val_cues, dtype=np.float32)
    val_targets_arr = np.array(val_targets, dtype=np.float32)

    candidates = build_candidate_schedules()
    results: Dict[str, Any] = {}
    best_name = ""
    best_score = -1e9
    best_betas: List[float] = []

    for name, betas in candidates.items():
        ising_sampler.warmup(betas, sweeps_per_stage, len(val_cues_arr))
        potts_sampler.warmup(betas, sweeps_per_stage, len(val_cues_arr))

        rec_i, _, _ = ising_sampler.sample(val_cues_arr, betas, sweeps_per_stage=sweeps_per_stage, seed=11)
        rec_p, _, _ = potts_sampler.sample(val_cues_arr, betas, sweeps_per_stage=sweeps_per_stage, seed=11)

        acc_i = float(np.mean(rec_i == val_targets_arr))
        acc_p = float(np.mean(rec_p == val_targets_arr))
        spur_i = sum(1 for s in rec_i if is_spurious_attractor(s, patterns)) / len(rec_i)
        spur_p = sum(1 for s in rec_p if is_spurious_attractor(s, patterns, allow_inverted=False)) / len(rec_p)

        composite = 0.5 * (acc_i + acc_p) - 0.25 * 0.5 * (spur_i + spur_p)
        results[name] = {
            "betas": betas,
            "ising_bit_recovery": round(acc_i, 4),
            "potts_bit_recovery": round(acc_p, 4),
            "ising_spurious_rate": round(spur_i, 4),
            "potts_spurious_rate": round(spur_p, 4),
            "composite_score": round(composite, 4),
        }
        if composite > best_score:
            best_score = composite
            best_name = name
            best_betas = betas

    return best_name, best_betas, results


def detect_hardware_info() -> Dict[str, str]:
    """Query local NVIDIA GPU via nvidia-smi and active JAX runtime device."""
    gpu_name = "Unknown GPU"
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True,
            timeout=5,
        ).strip()
        if out:
            gpu_name = out.splitlines()[0]
    except Exception:
        pass
    jax_devs = [str(d) for d in jax.devices()]
    return {
        "host_gpu": gpu_name,
        "jax_backend": jax.default_backend(),
        "jax_devices": ", ".join(jax_devs),
    }


def update_manifests(exp12_payload: Dict[str, Any]) -> None:
    """Write Experiment 12 benchmark results into data/benchmark_manifest.json and docs/benchmark_manifest.json."""
    for path in MANIFEST_PATHS:
        if path.exists():
            manifest = json.loads(path.read_text(encoding="utf-8"))
        else:
            manifest = {"metadata": {}, "stations": []}

        manifest.setdefault("metadata", {})
        manifest["metadata"]["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest.setdefault("experiments", {})
        manifest["experiments"]["exp12_associative_recall"] = exp12_payload

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    print("[*] Experiment 12: Associative Memory Recall Benchmark (thrml 0.1.4)")
    hw = detect_hardware_info()
    print(f"    Host GPU: {hw['host_gpu']} | JAX backend: {hw['jax_backend']} ({hw['jax_devices']})")

    patterns = generate_patterns(N_PATTERNS, N_BITS, seed=42)
    max_off_diag = float(np.max(np.abs((patterns @ patterns.T) / N_BITS - np.eye(N_PATTERNS))))
    print(f"    Stored patterns: P={N_PATTERNS}, N={N_BITS} (alpha={N_PATTERNS/N_BITS:.5f}, max_overlap={max_off_diag:.4f})")

    ising_sampler = ThrmlIsingAssociativeSampler(patterns, n_hidden=32, weight_scale=4.0)
    potts_sampler = ThrmlPottsAssociativeSampler(patterns, n_hidden=24, weight_scale=1.5)
    sweeps_per_stage = 2

    print("[*] Calibrating optimal inverse-temperature annealing schedule...")
    best_name, best_betas, sweep_results = calibrate_annealing_schedule(
        ising_sampler, potts_sampler, patterns, sweeps_per_stage=sweeps_per_stage
    )
    print(f"    Optimal schedule: {best_name} -> betas={best_betas}")

    calibration_payload = {
        "experiment": "exp12_associative_recall",
        "thrml_version": "0.1.4",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "optimal_schedule_name": best_name,
        "sweeps_per_stage": sweeps_per_stage,
        "total_sweeps": len(best_betas) * sweeps_per_stage,
        "inverse_temperatures_beta": best_betas,
        "temperatures_T": [round(1.0 / b, 4) for b in best_betas],
        "candidate_sweep": sweep_results,
    }
    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_PATH.write_text(json.dumps(calibration_payload, indent=2) + "\n", encoding="utf-8")
    print(f"[+] Exported calibration schedule to {CALIBRATION_PATH.relative_to(ROOT_DIR)}")

    # Warm up JIT for optimal schedule at N_TRIALS_PER_NOISE
    ising_jit_s = ising_sampler.warmup(best_betas, sweeps_per_stage, N_TRIALS_PER_NOISE)
    potts_jit_s = potts_sampler.warmup(best_betas, sweeps_per_stage, N_TRIALS_PER_NOISE)

    rng = np.random.default_rng(12345)
    by_noise: Dict[str, Any] = {}
    total_exec_by_model = {
        "classical_hopfield": 0.0,
        "exponential_hopfield": 0.0,
        "thrml_ising": 0.0,
        "thrml_potts": 0.0,
    }

    for noise in NOISE_LEVELS:
        noise_key = f"{int(noise * 100)}%"
        cues = np.array([corrupt_cue(patterns[t % N_PATTERNS], noise, rng) for t in range(N_TRIALS_PER_NOISE)])
        targets = np.array([patterns[t % N_PATTERNS] for t in range(N_TRIALS_PER_NOISE)])

        rec_c, st_c, dt_c = run_classical_hopfield(cues, patterns)
        rec_e, st_e, dt_e = run_exponential_hopfield(cues, patterns, beta_exp=10.0)
        rec_i, st_i, dt_i = ising_sampler.sample(cues, best_betas, sweeps_per_stage=sweeps_per_stage, seed=int(noise * 1000))
        rec_p, st_p, dt_p = potts_sampler.sample(cues, best_betas, sweeps_per_stage=sweeps_per_stage, seed=int(noise * 1000))

        m_c = evaluate_predictions(rec_c, targets, patterns, st_c, dt_c, allow_inverted=True)
        m_e = evaluate_predictions(rec_e, targets, patterns, st_e, dt_e, allow_inverted=False)
        m_i = evaluate_predictions(rec_i, targets, patterns, st_i, dt_i, allow_inverted=True)
        m_p = evaluate_predictions(rec_p, targets, patterns, st_p, dt_p, allow_inverted=False)

        total_exec_by_model["classical_hopfield"] += dt_c
        total_exec_by_model["exponential_hopfield"] += dt_e
        total_exec_by_model["thrml_ising"] += dt_i
        total_exec_by_model["thrml_potts"] += dt_p

        by_noise[noise_key] = {
            "cue_noise_rate": noise,
            "effective_bit_flip_rate": noise * 0.5,
            "n_trials": N_TRIALS_PER_NOISE,
            "classical_hopfield": m_c,
            "exponential_hopfield": m_e,
            "thrml_ising": m_i,
            "thrml_potts": m_p,
        }
        print(
            f"    Noise {noise_key}: "
            f"Classical={m_c['bit_error_recovery']:.3f} (spur={m_c['spurious_attractor_rate']:.2f}) | "
            f"ExpHopfield={m_e['bit_error_recovery']:.3f} (spur={m_e['spurious_attractor_rate']:.2f}) | "
            f"thrml-Ising={m_i['bit_error_recovery']:.3f} (spur={m_i['spurious_attractor_rate']:.2f}) | "
            f"thrml-Potts={m_p['bit_error_recovery']:.3f} (spur={m_p['spurious_attractor_rate']:.2f})"
        )

    exp12_payload = {
        "experiment_id": 12,
        "name": "Associative Memory Recall (thrml 0.1.4 Ising & Potts vs. Hopfield)",
        "thrml_version": "0.1.4",
        "patterns": N_PATTERNS,
        "bits_per_pattern": N_BITS,
        "storage_load_alpha": round(N_PATTERNS / N_BITS, 5),
        "noise_model": "random_replacement",
        "noise_levels": ["20%", "35%", "50%"],
        "trials_per_noise": N_TRIALS_PER_NOISE,
        "calibration_file": "configs/exp12_annealing_calibration.json",
        "optimal_annealing_schedule": {
            "name": best_name,
            "betas": best_betas,
            "sweeps_per_stage": sweeps_per_stage,
            "total_sweeps": len(best_betas) * sweeps_per_stage,
        },
        "hardware": {
            "host_gpu": hw["host_gpu"],
            "jax_backend": hw["jax_backend"],
            "jax_devices": hw["jax_devices"],
            "rtx_5080_host_execution_time_s": {
                k: round(v, 5) for k, v in total_exec_by_model.items()
            },
            "jit_warmup_s": {
                "thrml_ising": round(ising_jit_s, 4),
                "thrml_potts": round(potts_jit_s, 4),
            },
        },
        "results_by_noise": by_noise,
    }

    update_manifests(exp12_payload)
    print("[+] Updated data/benchmark_manifest.json and docs/benchmark_manifest.json")


if __name__ == "__main__":
    main()
