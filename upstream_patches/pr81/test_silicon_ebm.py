"""
upstream_patches.pr81.test_silicon_ebm
======================================
Comprehensive verification suite for Phases 1, 2, and 3 (Upstream PR #81):
  - Phase 1: `ThermoHAL` signal translation, binary wire framing (`THAL`), ADC decode,
             and `ThermoHALRouter` hardware-aware dispatch.
  - Phase 2: JAX `@custom_vjp` K-bit DAC quantization (forward + STE backward),
             Op-Amp rail clipping (forward + sech^2 surrogate backward),
             3-pole 1/f pink noise bath, and `BackendType.SIMULATED_NOISE`.
  - Phase 3: `ContinuousVariableEBM` and 3-layer `DeepBoltzmannMachine` noise-resilient
             training loops + `ThermodynamicEnergyProfiler` (fJ/op & Joules/1k ESS).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import thermo_hal
from thermo_physics.backends.sampler_adapter import (
    BackendType,
    HamiltonianGraph,
    SamplingConfig,
    get_sampler,
)
from thermo_physics.benchmarks.energy_profiler import ThermodynamicEnergyProfiler
from thermo_physics.benchmarks.next_gen_ebm import (
    CVEBMConfig,
    ContinuousVariableEBM,
    DBMConfig,
    DeepBoltzmannMachine,
)
from thermo_physics.tpu.jax_silicon import (
    init_jax_pink_noise_state,
    jax_dac_quantize_symmetric,
    jax_opamp_clip,
    jax_step_pink_thermal_noise,
)


class TestPhase1ThermoHAL:
    """Verify Phase 1: ThermoHAL base interface, wire protocol, and router."""

    def test_hal_translation_and_wire_packet(self) -> None:
        hal = thermo_hal.ThermoHAL(dac_bits=6, adc_bits=12, v_rail=1.5, v_ref_adc=3.3)
        j_mat = jnp.array(
            [
                [0.0, -0.75, 0.40, -0.25],
                [-0.75, 0.0, 0.60, -0.50],
                [0.40, 0.60, 0.0, -0.90],
                [-0.25, -0.50, -0.90, 0.0],
            ],
            dtype=jnp.float32,
        )
        h_vec = jnp.array([0.3, -0.6, 0.15, -0.45], dtype=jnp.float32)

        packet = hal.translate_to_physical_signals((j_mat, h_vec), beta=1.2)
        assert packet.num_nodes == 4
        assert packet.dac_bits == 6
        assert np.all(np.abs(packet.coupling_voltages) <= 1.5 + 1e-5)
        assert np.all(np.abs(packet.bias_voltages) <= 1.5 + 1e-5)
        # 6-bit DAC codewords must fall in [0, 63]
        assert np.all(packet.bias_dac_codewords <= 63)
        assert np.all(packet.coupling_dac_codewords <= 63)

        wire = packet.to_wire_frame(max_channels=4)
        assert wire[:4] == b"THAL"
        assert len(wire) == 16 + 4 * 2  # 16-byte header + 4 uint16 codewords

    def test_hal_adc_roundtrip_to_jax_arrays(self) -> None:
        hal = thermo_hal.ThermoHAL(dac_bits=6, adc_bits=12, v_rail=1.5, v_ref_adc=3.3)
        adc_codes = np.array([0, 2048, 4095], dtype=np.uint16)
        v_jax = hal.adc_codes_to_jax_voltages(adc_codes)
        assert v_jax.shape == (3,)
        assert float(v_jax[0]) == pytest.approx(-1.5, abs=1e-3)
        assert float(v_jax[1]) == pytest.approx(0.0, abs=1e-2)
        assert float(v_jax[2]) == pytest.approx(1.5, abs=1e-3)

        spins = hal.adc_codes_to_jax_spins(adc_codes, spin_type="bipolar")
        np.testing.assert_array_equal(np.asarray(spins), np.array([-1, 1, 1], dtype=np.int8))

    def test_hal_router_dispatch_and_entropy_injection(self) -> None:
        hal = thermo_hal.ThermoHAL(dac_bits=6, v_rail=1.5)
        router = thermo_hal.ThermoHALRouter(hal=hal)
        nodes = router.discover_nodes()
        assert len(nodes) >= 4

        j_mat = np.array([[0.0, -1.0, 0.5], [-1.0, 0.0, -0.8], [0.5, -0.8, 0.0]])
        h_vec = np.array([0.2, -0.2, 0.1])
        cfg = SamplingConfig(n_samples=80, burn_in=20, beta=1.2, seed=7)

        res, packet, meta = router.dispatch((j_mat, h_vec), config=cfg, prefer_silicon_realism=True)
        assert res.samples.shape == (80, 3)
        assert res.backend_type == BackendType.SIMULATED_NOISE
        assert "hal_routing" in res.diagnostics
        assert meta["selected_target"] == thermo_hal.ExecutionTarget.REPURPOSED_ENTROPY_HYBRID.value


class TestPhase2JaxSiliconPrimitives:
    """Verify Phase 2: Custom JAX forward/backward primitives and 1/f pink noise bath."""

    def test_jax_dac_quantize_forward_and_ste_backward(self) -> None:
        w = jnp.linspace(-2.0, 2.0, 65, dtype=jnp.float32)
        w_q = jax_dac_quantize_symmetric(w, 4, 1.5)
        # Clipped to [-1.5, 1.5] with at most 2^4 = 16 unique levels
        assert float(jnp.max(jnp.abs(w_q))) <= 1.5 + 1e-5
        unique_vals = np.unique(np.round(np.asarray(w_q), 4))
        assert len(unique_vals) <= 16

        # Verify custom_vjp Straight-Through Estimator gradient
        grad_fn = jax.grad(lambda x: jnp.sum(jax_dac_quantize_symmetric(x, 4, 1.5) ** 2))
        grads = grad_fn(jnp.array([-2.0, -0.8, 0.5, 2.0], dtype=jnp.float32))
        # Outside [-1.5, 1.5], STE gradient is masked to 0; inside, it is non-zero
        assert float(grads[0]) == 0.0
        assert float(grads[3]) == 0.0
        assert float(grads[1]) < 0.0
        assert float(grads[2]) > 0.0

    def test_jax_opamp_clip_forward_and_surrogate_backward(self) -> None:
        x = jnp.array([-3.0, -1.0, 0.0, 1.2, 4.0], dtype=jnp.float32)
        v_clipped = jax_opamp_clip(x, 1.5)
        np.testing.assert_allclose(
            np.asarray(v_clipped),
            np.array([-1.5, -1.0, 0.0, 1.2, 1.5], dtype=np.float32),
            atol=1e-5,
        )

        # Verify surrogate sech^2 backward gradient does not vanish outside [-1.5, 1.5]
        grad_fn = jax.grad(lambda u: jnp.sum(jax_opamp_clip(u, 1.5)))
        g = grad_fn(x)
        # Inside linear rail: exact 1.0 slope; outside rail: smooth positive surrogate > 0.05
        assert float(g[1]) == pytest.approx(1.0, abs=1e-5)
        assert float(g[2]) == pytest.approx(1.0, abs=1e-5)
        assert 0.05 <= float(g[0]) < 1.0
        assert 0.05 <= float(g[4]) < 1.0

    def test_jax_pink_thermal_noise_generator_and_sampler_adapter(self) -> None:
        state = init_jax_pink_noise_state((16,))
        key = jax.random.PRNGKey(99)
        state, key = jax_step_pink_thermal_noise(
            state, key, dt=0.02, temperature=1.5, tau_color=0.05, pink_fraction=0.6
        )
        assert state.total_noise.shape == (16,)
        assert float(jnp.var(state.total_noise)) > 0.0

        sampler = get_sampler(BackendType.SIMULATED_NOISE, dac_bits=6, v_rail=1.5)
        j_mat = np.array([[0.0, -1.0, 0.4], [-1.0, 0.0, -0.6], [0.4, -0.6, 0.0]])
        h_vec = np.array([0.1, -0.3, 0.2])
        graph = HamiltonianGraph(couplings=j_mat, biases=h_vec)
        res = sampler.sample(graph, SamplingConfig(n_samples=64, burn_in=16, beta=1.0))
        assert res.backend_type == BackendType.SIMULATED_NOISE
        assert "j_quantization_rms_error" in res.diagnostics
        assert "housekeeping_heat_joules" in res.diagnostics


class TestPhase3BenchmarksAndEnergyProfiler:
    """Verify Phase 3: CV-EBM, 3-Layer DBM, and Energy-per-Op Profiler."""

    def test_continuous_variable_ebm_noise_resilient_training(self) -> None:
        cfg = CVEBMConfig(num_dims=8, dac_bits=6, v_rail=1.5, langevin_steps=15, learning_rate=0.04)
        model = ContinuousVariableEBM(cfg)
        target = jnp.array([[1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]] * 8, dtype=jnp.float32)
        report = model.train_noise_resilient(target, epochs=12)
        assert report["dac_bits"] == 6
        assert len(report["loss_history"]) == 12
        assert np.isfinite(report["final_mse"])

    def test_deep_boltzmann_machine_training(self) -> None:
        cfg = DBMConfig(n_visible=8, n_hidden1=6, n_hidden2=4, dac_bits=6, v_rail=1.5, cd_k_steps=8)
        dbm = DeepBoltzmannMachine(cfg)
        patterns = jnp.array(
            [
                [1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0],
                [-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0],
            ]
            * 4,
            dtype=jnp.float32,
        )
        rep = dbm.train_noise_resilient(patterns, epochs=10)
        assert rep["layers"] == [8, 6, 4]
        assert len(rep["recall_history"]) == 10
        assert 0.0 <= rep["final_overlap"] <= 1.0

    def test_energy_per_op_profiler_sweep(self) -> None:
        profiler = ThermodynamicEnergyProfiler()
        rng = np.random.default_rng(42)
        n = 12
        j_raw = rng.normal(0.0, 0.4, size=(n, n))
        j_mat = 0.5 * (j_raw + j_raw.T)
        np.fill_diagonal(j_mat, 0.0)
        h_vec = rng.normal(0.0, 0.2, size=(n,))
        graph = HamiltonianGraph(couplings=j_mat, biases=h_vec)

        reports = profiler.run_precision_energy_sweep(
            graph,
            config=SamplingConfig(n_samples=60, burn_in=15, beta=1.2),
            bit_depths=[8, 6, 4],
            v_rail=1.5,
        )
        assert len(reports) == 3
        # 4-bit DAC capacitive switching energy must be strictly lower than 6-bit and 8-bit
        assert reports[2].dac_switching_energy_fj < reports[1].dac_switching_energy_fj
        assert reports[1].dac_switching_energy_fj < reports[0].dac_switching_energy_fj
        assert reports[1].energy_efficiency_gain_x > 100.0
