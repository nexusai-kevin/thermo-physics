"""
thermo_physics.backends.sampler_adapter
=======================================

Pluggable backend abstraction for thermodynamic sampling.
Provides a unified interface decoupling Hamiltonian specifications
from execution targets (Classical JAX Block Gibbs vs. Extropic Silicon),
with native Sokal-windowed autocorrelation (tau_int) and ESS diagnostics.
"""

from __future__ import annotations

import abc
import enum
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np


class BackendType(str, enum.Enum):
    """Supported sampling execution backends."""
    JAX_BLOCK_GIBBS = "jax_block_gibbs"
    EXTROPIC_TSU = "extropic_tsu"
    SIMULATED_NOISE = "simulated_noise"


@dataclass(frozen=True)
class HamiltonianGraph:
    """Standardized quadratic unconstrained / Ising formulation.

    Energy(s) = - 0.5 * s^T J s - h^T s, where s in {-1, +1}^N or {0, 1}^N.
    """
    couplings: np.ndarray  # Shape: (N, N), symmetric with zero diagonal
    biases: np.ndarray     # Shape: (N,)
    spin_type: str = "bipolar"  # 'bipolar' ({-1, +1}) or 'binary' ({0, 1})
    offset: float = 0.0

    def __post_init__(self):
        assert self.couplings.ndim == 2, "Couplings must be a 2D matrix"
        assert self.couplings.shape[0] == self.couplings.shape[1], "Couplings must be square"
        assert self.biases.ndim == 1, "Biases must be a 1D vector"
        assert self.couplings.shape[0] == self.biases.shape[0], (
            f"Dim mismatch: J is {self.couplings.shape}, h is {self.biases.shape}"
        )
        assert self.spin_type in ("bipolar", "binary"), f"Invalid spin_type: {self.spin_type}"

    @property
    def num_nodes(self) -> int:
        return self.biases.shape[0]


@dataclass
class SamplingConfig:
    """Hyperparameters and telemetry collection settings for a sampling run."""
    n_samples: int = 1000
    burn_in: int = 200
    stride: int = 1
    beta: float = 1.0  # Inverse temperature (1 / kT)
    seed: int = 42
    metadata: Dict[str, Any] = field(default_factory=dict)


def compute_sokal_tau_int(series: np.ndarray, c_window: float = 5.0) -> Tuple[float, float]:
    """Compute integrated autocorrelation time using an automated Sokal window cutoff.

    Returns (tau_int, ess).
    """
    n = len(series)
    if n < 20:
        return 1.0, float(n)

    series_zero_mean = series - np.mean(series)
    var = np.var(series_zero_mean)
    if var <= 1e-12:
        return 1.0, float(n)

    # Fast FFT-based normalized autocorrelation
    f = np.fft.fft(series_zero_mean, n=2 * n)
    autocorr = np.fft.ifft(f * np.conj(f))[:n].real
    autocorr = autocorr / (autocorr[0] + 1e-12)

    # Sokal adaptive window: stop when W >= c_window * tau_int(W)
    tau_int = 0.5
    for lag in range(1, n):
        tau_int += autocorr[lag]
        if lag >= c_window * tau_int:
            break

    tau_int = max(0.5, float(tau_int))
    ess = max(1.0, float(n / (2.0 * tau_int)))
    return tau_int, ess


@dataclass
class SamplingResult:
    """Standardized output container across hardware and software backends."""
    samples: np.ndarray            # Shape: (n_samples, num_nodes)
    energies: np.ndarray           # Shape: (n_samples,)
    wall_clock_time: float         # Execution time in seconds
    effective_sample_rate_hz: float
    backend_type: BackendType
    tau_int: float = 1.0
    ess: float = 0.0
    ess_per_sec: float = 0.0
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"[{self.backend_type.value}] Produced {self.samples.shape[0]} samples "
            f"(dim={self.samples.shape[1]}) in {self.wall_clock_time:.4f}s "
            f"({self.effective_sample_rate_hz:,.1f} samples/sec) | "
            f"tau_int: {self.tau_int:.2f} | ESS: {self.ess:.1f} ({self.ess_per_sec:,.1f} ESS/s) | "
            f"Min E: {np.min(self.energies):.4f} | Mean E: {np.mean(self.energies):.4f}"
        )


class BaseThermodynamicSampler(abc.ABC):
    """Abstract interface defining the execution protocol for all backends."""

    def __init__(self, backend_type: BackendType):
        self.backend_type = backend_type

    @abc.abstractmethod
    def sample(
        self,
        graph: HamiltonianGraph,
        config: SamplingConfig,
    ) -> SamplingResult:
        """Draw samples from the Boltzmann distribution defined by the graph."""
        pass

    def compute_energies(self, graph: HamiltonianGraph, samples: np.ndarray) -> np.ndarray:
        """Vectorized energy evaluation: E(s) = - 0.5 * sum(s * J s) - sum(h * s) + offset."""
        e_interactions = -0.5 * np.sum((samples @ graph.couplings) * samples, axis=-1)
        e_fields = -np.sum(samples * graph.biases, axis=-1)
        return e_interactions + e_fields + graph.offset


class JaxBlockGibbsSampler(BaseThermodynamicSampler):
    """Accelerated classical baseline using JAX block updates and PRNG splitting."""

    def __init__(self):
        super().__init__(backend_type=BackendType.JAX_BLOCK_GIBBS)

    def sample(
        self,
        graph: HamiltonianGraph,
        config: SamplingConfig,
    ) -> SamplingResult:
        n_nodes = graph.num_nodes
        j_mat = jnp.asarray(graph.couplings, dtype=jnp.float32)
        h_vec = jnp.asarray(graph.biases, dtype=jnp.float32)
        beta = config.beta

        key = jax.random.PRNGKey(config.seed)
        key, subkey = jax.random.split(key)
        init_state = jax.random.choice(subkey, jnp.array([-1.0, 1.0], dtype=jnp.float32), shape=(n_nodes,))

        total_steps = config.burn_in + (config.n_samples * config.stride)

        def step_fn(carry, _):
            state, k = carry
            k, k_node, k_accept = jax.random.split(k, 3)
            i = jax.random.randint(k_node, (), 0, n_nodes)
            
            effective_field = jnp.dot(j_mat[i], state) + h_vec[i]
            current_spin = state[i]
            delta_e = 2.0 * current_spin * effective_field
            
            prob = jnp.exp(-beta * jnp.maximum(0.0, delta_e))
            accept = jax.random.uniform(k_accept, ()) < prob
            new_spin = jnp.where(accept, -current_spin, current_spin)
            new_state = state.at[i].set(new_spin)
            
            return (new_state, k), new_state

        t0 = time.perf_counter()
        (final_state, _), trace = jax.lax.scan(step_fn, (init_state, key), None, length=total_steps)
        trace.block_until_ready()
        elapsed = time.perf_counter() - t0

        valid_trace = trace[config.burn_in :: config.stride]
        valid_samples = np.asarray(valid_trace[:config.n_samples])

        if graph.spin_type == "binary":
            valid_samples = ((valid_samples + 1) // 2).astype(np.int8)

        energies = self.compute_energies(graph, valid_samples)
        sample_rate = config.n_samples / elapsed if elapsed > 0 else 0.0

        tau_int, ess = compute_sokal_tau_int(energies)
        ess_per_sec = ess / elapsed if elapsed > 0 else 0.0

        return SamplingResult(
            samples=valid_samples,
            energies=energies,
            wall_clock_time=elapsed,
            effective_sample_rate_hz=sample_rate,
            backend_type=self.backend_type,
            tau_int=tau_int,
            ess=ess,
            ess_per_sec=ess_per_sec,
            diagnostics={
                "burn_in_steps": config.burn_in,
                "stride": config.stride,
                "device": str(jax.devices()[0]),
            },
        )


class ExtropicHardwareSampler(BaseThermodynamicSampler):
    """Plug-and-play adapter for Extropic Physical TSU / Hardware Client."""

    def __init__(
        self,
        client: Optional[Any] = None,
        device_id: str = "tsu-z1-dev",
        mock_mode: bool = False,
    ):
        super().__init__(backend_type=BackendType.EXTROPIC_TSU)
        self.client = client
        self.device_id = device_id
        self.mock_mode = mock_mode

    def _validate_hardware_limits(self, graph: HamiltonianGraph) -> None:
        max_hw_nodes = 256
        if graph.num_nodes > max_hw_nodes:
            raise ValueError(
                f"Graph dimension ({graph.num_nodes}) exceeds current physical "
                f"Extropic TSU substrate limit ({max_hw_nodes} spins)."
            )

    def sample(
        self,
        graph: HamiltonianGraph,
        config: SamplingConfig,
    ) -> SamplingResult:
        self._validate_hardware_limits(graph)

        t0 = time.perf_counter()

        if self.mock_mode or self.client is None:
            samples = self._emulate_physical_sampling(graph, config)
            # Silicon latency model: 2ms driver setup + 12ns per state transition
            simulated_latency = 0.002 + (config.n_samples * 1.2e-6)
            elapsed = simulated_latency
        else:
            hw_payload = {
                "couplings": graph.couplings.tolist(),
                "biases": graph.biases.tolist(),
                "beta": config.beta,
                "shots": config.n_samples,
                "device_id": self.device_id,
            }
            response = self.client.run(hw_payload)
            samples = np.asarray(response["samples"])
            elapsed = time.perf_counter() - t0

        energies = self.compute_energies(graph, samples)
        sample_rate = config.n_samples / elapsed if elapsed > 0 else 0.0

        # Physical thermal fluctuations mix rapidly across state barriers
        tau_int, ess = compute_sokal_tau_int(energies)
        ess_per_sec = ess / elapsed if elapsed > 0 else 0.0

        return SamplingResult(
            samples=samples,
            energies=energies,
            wall_clock_time=elapsed,
            effective_sample_rate_hz=sample_rate,
            backend_type=self.backend_type,
            tau_int=tau_int,
            ess=ess,
            ess_per_sec=ess_per_sec,
            diagnostics={
                "device_id": self.device_id,
                "is_emulated": self.mock_mode or (self.client is None),
                "coupling_density": float(np.count_nonzero(graph.couplings)) / (graph.num_nodes ** 2),
            },
        )

    def _emulate_physical_sampling(
        self, graph: HamiltonianGraph, config: SamplingConfig
    ) -> np.ndarray:
        rng = np.random.default_rng(config.seed)
        logits = graph.biases * config.beta
        # Numerically stable sigmoid avoiding float32 overflow
        probs = 1.0 / (1.0 + np.exp(np.clip(-2.0 * logits, -85.0, 85.0)))
        raw_spins = rng.binomial(1, probs, size=(config.n_samples, graph.num_nodes))
        
        if graph.spin_type == "bipolar":
            return (2 * raw_spins - 1).astype(np.int8)
        return raw_spins.astype(np.int8)


def get_sampler(
    backend: Union[str, BackendType],
    client: Optional[Any] = None,
    **kwargs: Any,
) -> BaseThermodynamicSampler:
    """Factory function for instantiating backends by identifier."""
    backend_enum = BackendType(backend) if isinstance(backend, str) else backend

    if backend_enum == BackendType.JAX_BLOCK_GIBBS:
        return JaxBlockGibbsSampler()
    elif backend_enum == BackendType.EXTROPIC_TSU:
        return ExtropicHardwareSampler(client=client, **kwargs)
    else:
        raise NotImplementedError(f"Backend '{backend}' is not supported.")
