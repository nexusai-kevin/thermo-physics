"""
thermo_physics.benchmarks.next_gen_ebm
======================================
Phase 3, Step 1: Next-Gen Thermodynamic Architectures & Noise-Resilient Training.

Moves beyond single-layer bipartite RBMs and discrete spin glasses (PR #79 / PR #80) to:
1. Continuous-Variable Energy-Based Models (`ContinuousVariableEBM`):
   Models continuous analog state coordinates x in [-V_rail, +V_rail]^N governed by
   coupled non-convex quartic potentials and bilinear analog synaptic crossbars,
   with noise-resilient Langevin Contrastive Divergence training using `@jax.custom_vjp`
   DAC quantization and Op-Amp rail clipping in the forward & backward passes.
2. Deep Boltzmann Machines (`DeepBoltzmannMachine`):
   Multi-layer deep hierarchy (Visible v <-> Hidden h1 <-> Deep Hidden h2) utilizing
   2-color alternating block partitioning (Block A: v U h2, Block B: h1) and trained
   with silicon-realistic K-bit DAC + 1/f pink noise injection for hardware resilience.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from thermo_physics.backends.sampler_adapter import compute_sokal_tau_int
from thermo_physics.tpu.jax_silicon import (
    init_jax_pink_noise_state,
    jax_dac_quantize_symmetric,
    jax_opamp_clip,
    jax_step_pink_thermal_noise,
)


# ============================================================================
# 1. CONTINUOUS-VARIABLE ENERGY-BASED MODEL (CV-EBM)
# ============================================================================

@dataclass
class CVEBMConfig:
    """Hyperparameters for Continuous-Variable Energy-Based Model."""
    num_dims: int = 16
    alpha_well: float = 1.25
    dac_bits: int = 6
    v_rail: float = 1.5
    tau_color: float = 0.05
    pink_fraction: float = 0.5
    beta: float = 1.5
    dt: float = 0.02
    langevin_steps: int = 40
    learning_rate: float = 0.03
    seed: int = 42


class ContinuousVariableEBM:
    """Continuous-Variable EBM with quartic double-well + continuous crossbar couplings.

    Energy Function:
      E(x; J, h) = sum_i [ (alpha / 4) * (x_i^2 - 1)^2 ] - 0.5 * x^T J_dac x - h_dac^T x
    where J_dac and h_dac pass through differentiable K-bit DAC quantization (`jax_dac_quantize_symmetric`)
    and state trajectories x(t) pass through Op-Amp rail clipping (`jax_opamp_clip`).
    """

    def __init__(self, config: CVEBMConfig):
        self.config = config
        rng = np.random.default_rng(config.seed)
        n = config.num_dims
        j_init = rng.normal(0.0, 0.15 / np.sqrt(n), size=(n, n)).astype(np.float32)
        j_init = 0.5 * (j_init + j_init.T)
        np.fill_diagonal(j_init, 0.0)
        self.j_weights = jnp.asarray(j_init)
        self.h_biases = jnp.zeros((n,), dtype=jnp.float32)

    @staticmethod
    def energy_fn(
        x: jnp.ndarray,
        j_weights: jnp.ndarray,
        h_biases: jnp.ndarray,
        alpha_well: float,
        dac_bits: int,
        v_rail: float,
    ) -> jnp.ndarray:
        """Evaluate silicon-quantized CV-EBM energy for batch or single state `x`."""
        x_clipped = jax_opamp_clip(x, v_rail)
        j_q = jax_dac_quantize_symmetric(j_weights, dac_bits, v_rail)
        h_q = jax_dac_quantize_symmetric(h_biases, dac_bits, v_rail)

        quartic = 0.25 * alpha_well * jnp.sum((x_clipped ** 2 - 1.0) ** 2, axis=-1)
        interaction = -0.5 * jnp.sum((x_clipped @ j_q) * x_clipped, axis=-1)
        bias_term = -jnp.sum(x_clipped * h_q, axis=-1)
        return quartic + interaction + bias_term

    def sample_langevin_chain(
        self,
        init_x: jnp.ndarray,
        key: jax.Array,
        n_steps: Optional[int] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, float]]:
        """Run silicon-realistic continuous Langevin MCMC with 1/f pink + thermal Gaussian noise."""
        cfg = self.config
        steps = n_steps or cfg.langevin_steps
        temp = 1.0 / max(1e-4, cfg.beta)
        noise_state = init_jax_pink_noise_state(init_x.shape, dtype=jnp.float32)

        grad_e_fn = jax.grad(
            lambda x_in: jnp.sum(
                self.energy_fn(x_in, self.j_weights, self.h_biases, cfg.alpha_well, cfg.dac_bits, cfg.v_rail)
            )
        )

        def body_fn(carry, _):
            x_curr, n_state, k = carry
            n_state, k = jax_step_pink_thermal_noise(
                n_state,
                k,
                dt=cfg.dt,
                temperature=temp,
                gamma=1.0,
                tau_color=cfg.tau_color,
                pink_fraction=cfg.pink_fraction,
            )
            grad_x = grad_e_fn(x_curr)
            x_unclipped = x_curr - cfg.dt * grad_x + jnp.sqrt(cfg.dt) * n_state.total_noise
            clips = jnp.sum((jnp.abs(x_unclipped) > cfg.v_rail).astype(jnp.int32))
            x_next = jax_opamp_clip(x_unclipped, cfg.v_rail)
            e_mean = jnp.mean(
                self.energy_fn(x_next, self.j_weights, self.h_biases, cfg.alpha_well, cfg.dac_bits, cfg.v_rail)
            )
            return (x_next, n_state, k), (e_mean, clips)

        (x_final, _, _), (e_trace, clip_trace) = jax.lax.scan(
            body_fn, (init_x, noise_state, key), None, length=steps
        )
        e_np = np.asarray(e_trace)
        tau_int, ess = compute_sokal_tau_int(e_np)
        metrics = {
            "final_mean_energy": float(e_np[-1]),
            "total_rail_clips": int(np.sum(np.asarray(clip_trace))),
            "tau_int": float(tau_int),
            "ess": float(ess),
        }
        return x_final, e_trace, metrics

    def train_noise_resilient(
        self,
        target_data: jnp.ndarray,
        epochs: int = 25,
    ) -> Dict[str, Any]:
        """Train CV-EBM via Langevin Contrastive Divergence through differentiable
        DAC quantization (`@jax.custom_vjp`) and Op-Amp rail clipping under 1/f noise.
        """
        cfg = self.config
        n = cfg.num_dims
        eye_mask = 1.0 - jnp.eye(n, dtype=jnp.float32)
        key = jax.random.PRNGKey(cfg.seed)

        def cd_loss(j_w: jnp.ndarray, h_b: jnp.ndarray, x_pos: jnp.ndarray, x_neg: jnp.ndarray) -> jnp.ndarray:
            e_pos = jnp.mean(self.energy_fn(x_pos, j_w, h_b, cfg.alpha_well, cfg.dac_bits, cfg.v_rail))
            e_neg = jnp.mean(self.energy_fn(x_neg, j_w, h_b, cfg.alpha_well, cfg.dac_bits, cfg.v_rail))
            l2_reg = 1e-3 * (jnp.sum(j_w ** 2) + jnp.sum(h_b ** 2))
            return (e_pos - e_neg) + l2_reg

        grad_fn = jax.grad(cd_loss, argnums=(0, 1))

        loss_history: List[float] = []
        mse_history: List[float] = []
        t0 = time.perf_counter()

        for _ in range(epochs):
            key, subkey, corr_key = jax.random.split(key, 3)
            # Corrupt target data with analog thermal + rail noise and relax via Langevin
            noisy_init = jax_opamp_clip(
                target_data + 0.35 * jax.random.normal(corr_key, target_data.shape),
                cfg.v_rail,
            )
            x_neg, _, _ = self.sample_langevin_chain(noisy_init, subkey, n_steps=cfg.langevin_steps)

            g_j, g_h = grad_fn(self.j_weights, self.h_biases, target_data, jax.lax.stop_gradient(x_neg))
            g_j_sym = 0.5 * (g_j + g_j.T) * eye_mask

            self.j_weights = jax_opamp_clip(self.j_weights - cfg.learning_rate * g_j_sym, cfg.v_rail)
            self.h_biases = jax_opamp_clip(self.h_biases - cfg.learning_rate * g_h, cfg.v_rail)

            loss_val = float(cd_loss(self.j_weights, self.h_biases, target_data, x_neg))
            recon_mse = float(jnp.mean((target_data - x_neg) ** 2))
            loss_history.append(loss_val)
            mse_history.append(recon_mse)

        elapsed = time.perf_counter() - t0
        return {
            "architecture": "Continuous-Variable EBM (CV-EBM)",
            "dac_bits": cfg.dac_bits,
            "v_rail": cfg.v_rail,
            "epochs": epochs,
            "initial_mse": mse_history[0] if mse_history else 0.0,
            "final_mse": mse_history[-1] if mse_history else 0.0,
            "loss_history": loss_history,
            "mse_history": mse_history,
            "wall_time_sec": elapsed,
        }


# ============================================================================
# 2. DEEP BOLTZMANN MACHINE (3-LAYER DBM: v <-> h1 <-> h2)
# ============================================================================

@dataclass
class DBMConfig:
    """Hyperparameters for 3-Layer Deep Boltzmann Machine."""
    n_visible: int = 16
    n_hidden1: int = 12
    n_hidden2: int = 8
    dac_bits: int = 6
    v_rail: float = 1.5
    beta: float = 1.5
    cd_k_steps: int = 15
    learning_rate: float = 0.04
    pink_noise_std: float = 0.12
    seed: int = 42


class DeepBoltzmannMachine:
    """3-Layer Deep Boltzmann Machine (v <-> h1 <-> h2) with 2-Color Block Gibbs Sampling.

    Exploits two-coloring across layers:
      - Color Block A: Visible layer `v` and Deep Hidden layer `h2` (conditionally independent given `h1`)
      - Color Block B: Intermediate Hidden layer `h1` (conditioned on both `v` and `h2`)
    All weights pass through differentiable K-bit DAC quantization (`jax_dac_quantize_symmetric`)
    and Op-Amp voltage rail clipping (`jax_opamp_clip`).
    """

    def __init__(self, config: DBMConfig):
        self.config = config
        rng = np.random.default_rng(config.seed)
        self.w1 = jnp.asarray(
            rng.normal(0.0, 0.2, size=(config.n_visible, config.n_hidden1)), dtype=jnp.float32
        )
        self.w2 = jnp.asarray(
            rng.normal(0.0, 0.2, size=(config.n_hidden1, config.n_hidden2)), dtype=jnp.float32
        )
        self.b_v = jnp.zeros((config.n_visible,), dtype=jnp.float32)
        self.b_h1 = jnp.zeros((config.n_hidden1,), dtype=jnp.float32)
        self.b_h2 = jnp.zeros((config.n_hidden2,), dtype=jnp.float32)

    def quantized_params(self) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Return K-bit DAC quantized and rail-clipped DBM parameters."""
        bits = self.config.dac_bits
        rail = self.config.v_rail
        w1_q = jax_opamp_clip(jax_dac_quantize_symmetric(self.w1, bits, rail), rail)
        w2_q = jax_opamp_clip(jax_dac_quantize_symmetric(self.w2, bits, rail), rail)
        bv_q = jax_opamp_clip(jax_dac_quantize_symmetric(self.b_v, bits, rail), rail)
        bh1_q = jax_opamp_clip(jax_dac_quantize_symmetric(self.b_h1, bits, rail), rail)
        bh2_q = jax_opamp_clip(jax_dac_quantize_symmetric(self.b_h2, bits, rail), rail)
        return w1_q, w2_q, bv_q, bh1_q, bh2_q

    def energy(self, v: jnp.ndarray, h1: jnp.ndarray, h2: jnp.ndarray) -> jnp.ndarray:
        """Joint DBM energy: E(v, h1, h2) = -v^T W1 h1 - h1^T W2 h2 - b_v^T v - b_h1^T h1 - b_h2^T h2."""
        w1_q, w2_q, bv_q, bh1_q, bh2_q = self.quantized_params()
        e_vh1 = -jnp.sum((v @ w1_q) * h1, axis=-1)
        e_h1h2 = -jnp.sum((h1 @ w2_q) * h2, axis=-1)
        e_bias = -jnp.sum(v * bv_q, axis=-1) - jnp.sum(h1 * bh1_q, axis=-1) - jnp.sum(h2 * bh2_q, axis=-1)
        return e_vh1 + e_h1h2 + e_bias

    def sample_block_gibbs(
        self,
        init_v: jnp.ndarray,
        key: jax.Array,
        n_sweeps: int = 50,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Execute 2-color alternating Block Gibbs sweeps across (v, h2) and (h1) with analog noise."""
        cfg = self.config
        w1_q, w2_q, bv_q, bh1_q, bh2_q = self.quantized_params()
        batch_size = init_v.shape[0]
        h2_init = jnp.ones((batch_size, cfg.n_hidden2), dtype=jnp.float32)

        def _sample_bipolar(field_in: jnp.ndarray, subkey: jax.Array) -> jnp.ndarray:
            k_noise, k_u = jax.random.split(subkey)
            # Inject analog thermal/pink field fluctuation before op-amp rail clipping
            noisy_field = jax_opamp_clip(
                field_in + cfg.pink_noise_std * jax.random.normal(k_noise, field_in.shape),
                cfg.v_rail,
            )
            prob_plus = jax.nn.sigmoid(2.0 * cfg.beta * noisy_field)
            u = jax.random.uniform(k_u, shape=field_in.shape)
            return jnp.where(u < prob_plus, 1.0, -1.0)

        def sweep_step(carry, _):
            v_curr, h2_curr, k = carry
            k, k_h1, k_v, k_h2 = jax.random.split(k, 4)

            # Color Block B: Update h1 conditioned on (v, h2) simultaneously
            field_h1 = (v_curr @ w1_q) + (h2_curr @ w2_q.T) + bh1_q
            h1_next = _sample_bipolar(field_h1, k_h1)

            # Color Block A: Update v and h2 in parallel conditioned on h1
            field_v = (h1_next @ w1_q.T) + bv_q
            field_h2 = (h1_next @ w2_q) + bh2_q
            v_next = _sample_bipolar(field_v, k_v)
            h2_next = _sample_bipolar(field_h2, k_h2)

            e_step = jnp.mean(self.energy(v_next, h1_next, h2_next))
            return (v_next, h2_next, k), (v_next, h1_next, h2_next, e_step)

        _, (v_tr, h1_tr, h2_tr, e_tr) = jax.lax.scan(
            sweep_step, (init_v, h2_init, key), None, length=n_sweeps
        )
        return v_tr[-1], h1_tr[-1], h2_tr[-1], e_tr

    def train_noise_resilient(
        self,
        patterns: jnp.ndarray,
        epochs: int = 30,
    ) -> Dict[str, Any]:
        """Train 3-Layer DBM with Contrastive Divergence through custom_vjp DAC & Op-Amp primitives."""
        cfg = self.config
        key = jax.random.PRNGKey(cfg.seed)

        def mean_field_positive(v_data: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            w1_q, w2_q, _, bh1_q, bh2_q = self.quantized_params()
            h2_mf = jnp.zeros((v_data.shape[0], cfg.n_hidden2), dtype=jnp.float32)
            h1_mf = jnp.zeros((v_data.shape[0], cfg.n_hidden1), dtype=jnp.float32)
            # 5 damped mean-field fixed-point iterations for positive phase
            for _ in range(5):
                f_h1 = jax_opamp_clip((v_data @ w1_q) + (h2_mf @ w2_q.T) + bh1_q, cfg.v_rail)
                h1_mf = jnp.tanh(cfg.beta * f_h1)
                f_h2 = jax_opamp_clip((h1_mf @ w2_q) + bh2_q, cfg.v_rail)
                h2_mf = jnp.tanh(cfg.beta * f_h2)
            return h1_mf, h2_mf

        def dbm_cd_objective(
            w1: jnp.ndarray,
            w2: jnp.ndarray,
            bv: jnp.ndarray,
            bh1: jnp.ndarray,
            bh2: jnp.ndarray,
            v_pos: jnp.ndarray,
            h1_pos: jnp.ndarray,
            h2_pos: jnp.ndarray,
            v_neg: jnp.ndarray,
            h1_neg: jnp.ndarray,
            h2_neg: jnp.ndarray,
        ) -> jnp.ndarray:
            bits = cfg.dac_bits
            rail = cfg.v_rail
            w1_q = jax_opamp_clip(jax_dac_quantize_symmetric(w1, bits, rail), rail)
            w2_q = jax_opamp_clip(jax_dac_quantize_symmetric(w2, bits, rail), rail)
            bv_q = jax_opamp_clip(jax_dac_quantize_symmetric(bv, bits, rail), rail)
            bh1_q = jax_opamp_clip(jax_dac_quantize_symmetric(bh1, bits, rail), rail)
            bh2_q = jax_opamp_clip(jax_dac_quantize_symmetric(bh2, bits, rail), rail)

            def _e(v_x, h1_x, h2_x):
                return (
                    -jnp.sum((v_x @ w1_q) * h1_x, axis=-1)
                    - jnp.sum((h1_x @ w2_q) * h2_x, axis=-1)
                    - jnp.sum(v_x * bv_q, axis=-1)
                    - jnp.sum(h1_x * bh1_q, axis=-1)
                    - jnp.sum(h2_x * bh2_q, axis=-1)
                )

            return jnp.mean(_e(v_pos, h1_pos, h2_pos)) - jnp.mean(_e(v_neg, h1_neg, h2_neg))

        grad_fn = jax.grad(dbm_cd_objective, argnums=(0, 1, 2, 3, 4))
        recall_history: List[float] = []
        t0 = time.perf_counter()

        for _ in range(epochs):
            key, subkey = jax.random.split(key)
            h1_pos, h2_pos = mean_field_positive(patterns)
            v_neg, h1_neg, h2_neg, _ = self.sample_block_gibbs(
                patterns, subkey, n_sweeps=cfg.cd_k_steps
            )

            gw1, gw2, gbv, gbh1, gbh2 = grad_fn(
                self.w1,
                self.w2,
                self.b_v,
                self.b_h1,
                self.b_h2,
                patterns,
                jax.lax.stop_gradient(h1_pos),
                jax.lax.stop_gradient(h2_pos),
                jax.lax.stop_gradient(v_neg),
                jax.lax.stop_gradient(h1_neg),
                jax.lax.stop_gradient(h2_neg),
            )

            lr = cfg.learning_rate
            self.w1 = jax_opamp_clip(self.w1 - lr * gw1, cfg.v_rail)
            self.w2 = jax_opamp_clip(self.w2 - lr * gw2, cfg.v_rail)
            self.b_v = jax_opamp_clip(self.b_v - lr * gbv, cfg.v_rail)
            self.b_h1 = jax_opamp_clip(self.b_h1 - lr * gbh1, cfg.v_rail)
            self.b_h2 = jax_opamp_clip(self.b_h2 - lr * gbh2, cfg.v_rail)

            overlap = float(jnp.mean(patterns == v_neg))
            recall_history.append(overlap)

        elapsed = time.perf_counter() - t0
        return {
            "architecture": "3-Layer Deep Boltzmann Machine (DBM)",
            "layers": [cfg.n_visible, cfg.n_hidden1, cfg.n_hidden2],
            "dac_bits": cfg.dac_bits,
            "v_rail": cfg.v_rail,
            "epochs": epochs,
            "initial_overlap": recall_history[0] if recall_history else 0.0,
            "final_overlap": recall_history[-1] if recall_history else 0.0,
            "recall_history": recall_history,
            "wall_time_sec": elapsed,
        }
