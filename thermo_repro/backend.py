"""Backend abstraction for `thrml` and JAX/NumPy factor-graph Block Gibbs sampling.

Provides a drop-in interface matching Extropic's `thrml` library:
- `SpinNode`, `Block`, `SamplingSchedule`, `sample_states`
- `IsingEBM`, `IsingSamplingProgram`, `hinton_init`

When `thrml` is installed in the active environment, native `thrml` primitives are
imported directly. In environments where `thrml` is not yet installed, an exact
vectorized implementation of the same factor-graph Block Gibbs semantics is used
so all reference models and reproducibility benchmarks run out of the box.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence
import itertools
import math

try:
    import jax
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:  # pragma: no cover - fallback for bare environments
    import numpy as jnp  # type: ignore[no-redef]

    HAS_JAX = False

import numpy as np

try:
    from thrml import (
        Block as _ThrmlBlock,
        SamplingSchedule as _ThrmlSamplingSchedule,
        SpinNode as _ThrmlSpinNode,
        sample_states as _thrml_sample_states,
    )
    from thrml.models import (
        IsingEBM as _ThrmlIsingEBM,
        IsingSamplingProgram as _ThrmlIsingSamplingProgram,
        hinton_init as _thrml_hinton_init,
    )

    USING_NATIVE_THRML = True
except ImportError:
    USING_NATIVE_THRML = False


# ---------------------------------------------------------------------------
# PRNG compatibility helper (supports jax.random.key or numpy Generator)
# ---------------------------------------------------------------------------


def make_rng_key(seed: int = 0) -> Any:
    """Create a PRNG key compatible with JAX or NumPy fallback."""
    if HAS_JAX:
        return jax.random.key(seed)
    return np.random.default_rng(seed)


def split_rng_key(key: Any, num: int = 2) -> list[Any]:
    """Split a PRNG key into `num` independent streams."""
    if HAS_JAX and not isinstance(key, np.random.Generator):
        return list(jax.random.split(key, num))
    if isinstance(key, int):
        key = np.random.default_rng(key)
    seeds = key.integers(0, 2**31 - 1, size=num)
    return [np.random.default_rng(int(s)) for s in seeds]


def _uniform_random(key: Any, shape: tuple[int, ...]) -> Any:
    if HAS_JAX and not isinstance(key, np.random.Generator):
        return jax.random.uniform(key, shape=shape)
    if isinstance(key, int):
        key = np.random.default_rng(key)
    return key.random(shape)


def normal_random(key: Any, shape: tuple[int, ...]) -> Any:
    """Sample standard normal array of `shape`."""
    if HAS_JAX and not isinstance(key, np.random.Generator):
        return jax.random.normal(key, shape=shape)
    if isinstance(key, int):
        key = np.random.default_rng(key)
    return jnp.asarray(key.standard_normal(shape), dtype=jnp.float32)


# ---------------------------------------------------------------------------
# Reference implementations of `thrml` primitives
# ---------------------------------------------------------------------------


class _RefSpinNode:
    """Represents a binary Ising spin variable s_i in {-1, +1} (stored as bool)."""

    _counter = 0

    def __init__(self, name: str | None = None) -> None:
        self.uid = _RefSpinNode._counter
        _RefSpinNode._counter += 1
        self.name = name or f"s_{self.uid}"

    def __repr__(self) -> str:
        return f"SpinNode({self.name})"

    def __hash__(self) -> int:
        return hash(self.uid)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _RefSpinNode) and self.uid == other.uid


class _RefBlock:
    """An independent set of SpinNodes updated simultaneously in Block Gibbs."""

    def __init__(self, nodes: Sequence[_RefSpinNode]) -> None:
        self.nodes = list(nodes)

    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self):
        return iter(self.nodes)

    def __repr__(self) -> str:
        return f"Block(n={len(self.nodes)})"


@dataclass(frozen=True)
class _RefSamplingSchedule:
    """Configures warmup sweeps, sample count, and thinning interval."""

    n_warmup: int
    n_samples: int
    steps_per_sample: int = 1


class _RefIsingEBM:
    """Ising Energy-Based Model:
    E(s) = - sum_{(i,j) in edges} W_{ij} s_i s_j - sum_i b_i s_i
    with Boltzmann distribution P(s) = exp(-beta * E(s)) / Z(beta).
    """

    def __init__(
        self,
        nodes: Sequence[_RefSpinNode],
        edges: Sequence[tuple[_RefSpinNode, _RefSpinNode]],
        biases: Any,
        weights: Any,
        beta: Any,
    ) -> None:
        self.nodes = list(nodes)
        self.edges = list(edges)
        self.biases = jnp.asarray(biases, dtype=jnp.float32)
        self.weights = jnp.asarray(weights, dtype=jnp.float32)
        self.beta = jnp.asarray(beta, dtype=jnp.float32)

        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
        n = len(self.nodes)

        # Build symmetric coupling matrix J_{ij} for fast vectorized field calculation
        j_mat = np.zeros((n, n), dtype=np.float32)
        w_np = np.asarray(self.weights, dtype=np.float32)
        for k, (u, v) in enumerate(self.edges):
            ui = self.node_to_idx[u]
            vi = self.node_to_idx[v]
            j_mat[ui, vi] += w_np[k]
            j_mat[vi, ui] += w_np[k]
        self.coupling_matrix = jnp.asarray(j_mat)


class _RefIsingSamplingProgram:
    """Compiled Block Gibbs update program for an IsingEBM."""

    def __init__(
        self,
        model: _RefIsingEBM,
        free_blocks: Sequence[_RefBlock],
        clamped_blocks: Sequence[_RefBlock] = (),
    ) -> None:
        self.model = model
        self.free_blocks = list(free_blocks)
        self.clamped_blocks = list(clamped_blocks)

        self.free_block_indices = [
            np.array([model.node_to_idx[node] for node in blk.nodes], dtype=np.int32)
            for blk in self.free_blocks
        ]
        self.clamped_block_indices = [
            np.array([model.node_to_idx[node] for node in blk.nodes], dtype=np.int32)
            for blk in self.clamped_blocks
        ]


def _ref_hinton_init(
    key: Any,
    model: _RefIsingEBM,
    free_blocks: Sequence[_RefBlock],
    clamped_spec: Sequence[Any] = (),
) -> list[Any]:
    """Initialize each free block using independent marginal bias probabilities:
    P(s_i = +1) = sigmoid(2 * beta * b_i).
    Returns a list of boolean state arrays corresponding to `free_blocks`.
    """
    biases_np = np.asarray(model.biases, dtype=np.float32)
    beta_val = float(np.asarray(model.beta))
    subkeys = split_rng_key(key, max(len(free_blocks), 1))
    init_states = []
    for blk_idx, blk in enumerate(free_blocks):
        idxs = np.array([model.node_to_idx[n] for n in blk.nodes], dtype=np.int32)
        b_blk = biases_np[idxs]
        probs = 1.0 / (1.0 + np.exp(-2.0 * beta_val * b_blk))
        u = np.asarray(_uniform_random(subkeys[blk_idx], (len(idxs),)))
        state_bool = u < probs
        init_states.append(jnp.asarray(state_bool, dtype=bool))
    return init_states


def _ref_sample_states(
    key: Any,
    program: _RefIsingSamplingProgram,
    schedule: _RefSamplingSchedule,
    init_state: Sequence[Any],
    clamped_values: Sequence[Any],
    readout_blocks: Sequence[_RefBlock],
) -> list[Any]:
    """Run Block Gibbs sampling according to `schedule` and return samples for `readout_blocks`.

    Matches `thrml.sample_states` semantics:
    - Boolean `True` represents spin +1, `False` represents spin -1.
    - For each block B in `program.free_blocks`, all spins in B are updated simultaneously
      given the current states of all other spins:
        h_i^{eff} = b_i + sum_j J_{ij} s_j
        P(s_i = +1 | s_{-i}) = 1 / (1 + exp(-2 * beta * h_i^{eff}))
    """
    model = program.model
    n_nodes = len(model.nodes)
    j_mat = np.asarray(model.coupling_matrix, dtype=np.float32)
    biases = np.asarray(model.biases, dtype=np.float32)
    beta = float(np.asarray(model.beta))

    # Construct full spin vector s in {-1.0, +1.0}^n_nodes
    spins = np.ones((n_nodes,), dtype=np.float32)
    for blk_idxs, blk_state in zip(program.free_block_indices, init_state):
        blk_bool = np.asarray(blk_state, dtype=bool).reshape(-1)
        spins[blk_idxs] = np.where(blk_bool, 1.0, -1.0)

    for blk_idxs, c_val in zip(program.clamped_block_indices, clamped_values):
        c_bool = np.asarray(c_val, dtype=bool).reshape(-1)
        spins[blk_idxs] = np.where(c_bool, 1.0, -1.0)

    readout_indices = [
        np.array([model.node_to_idx[n] for n in blk.nodes], dtype=np.int32)
        for blk in readout_blocks
    ]
    recorded = [
        np.zeros((schedule.n_samples, len(idxs)), dtype=bool)
        for idxs in readout_indices
    ]

    total_sweeps = schedule.n_warmup + schedule.n_samples * schedule.steps_per_sample
    n_free_blocks = len(program.free_block_indices)

    # Pre-generate uniform random numbers for all sweeps for fast execution
    rng = (
        np.random.default_rng(int(jax.random.randint(key, (), 0, 2**31 - 1)))
        if (HAS_JAX and not isinstance(key, np.random.Generator))
        else (key if isinstance(key, np.random.Generator) else np.random.default_rng(0))
    )

    sample_idx = 0
    for sweep in range(total_sweeps):
        for b_idx in range(n_free_blocks):
            idxs = program.free_block_indices[b_idx]
            # Local effective field h_eff = b_i + sum_j J_{ij} s_j
            h_eff = biases[idxs] + j_mat[idxs] @ spins
            probs = 1.0 / (1.0 + np.exp(-np.clip(2.0 * beta * h_eff, -50.0, 50.0)))
            u = rng.random(len(idxs))
            new_bool = u < probs
            spins[idxs] = np.where(new_bool, 1.0, -1.0)

        if sweep >= schedule.n_warmup:
            offset = sweep - schedule.n_warmup
            if (offset + 1) % schedule.steps_per_sample == 0:
                for r_idx, r_idxs in enumerate(readout_indices):
                    recorded[r_idx][sample_idx] = spins[r_idxs] > 0
                sample_idx += 1

    return [jnp.asarray(arr, dtype=bool) for arr in recorded]


# ---------------------------------------------------------------------------
# Export unified symbols
# ---------------------------------------------------------------------------

if USING_NATIVE_THRML:
    def SpinNode(name: str | None = None) -> Any:  # noqa: N802
        """Create a native `thrml.SpinNode` (accepts optional label for convenience)."""
        return _ThrmlSpinNode()

    Block = _ThrmlBlock
    SamplingSchedule = _ThrmlSamplingSchedule

    def IsingEBM(  # noqa: N802
        nodes: Sequence[Any],
        edges: Sequence[tuple[Any, Any]],
        biases: Any,
        weights: Any,
        beta: Any,
    ) -> Any:
        """Create a native `thrml.models.IsingEBM` with JAX float32 arrays."""
        return _ThrmlIsingEBM(
            list(nodes),
            list(edges),
            jnp.asarray(biases, dtype=jnp.float32),
            jnp.asarray(weights, dtype=jnp.float32),
            jnp.asarray(beta, dtype=jnp.float32),
        )

    def IsingSamplingProgram(  # noqa: N802
        model: Any,
        free_blocks: Sequence[Any],
        clamped_blocks: Sequence[Any] = (),
    ) -> Any:
        """Create a native `thrml.models.IsingSamplingProgram`."""
        return _ThrmlIsingSamplingProgram(
            model,
            free_blocks=list(free_blocks),
            clamped_blocks=list(clamped_blocks),
        )

    def hinton_init(
        key: Any,
        model: Any,
        free_blocks: Sequence[Any],
        batch_shape: tuple[int, ...] = (),
    ) -> list[Any]:
        """Run native `thrml.models.hinton_init`."""
        shape = batch_shape if isinstance(batch_shape, tuple) else ()
        return _thrml_hinton_init(key, model, list(free_blocks), shape)

    def sample_states(
        key: Any,
        program: Any,
        schedule: Any,
        init_state: Sequence[Any],
        clamped_values: Sequence[Any],
        readout_blocks: Sequence[Any],
    ) -> list[Any]:
        """Run native `thrml.sample_states` ensuring JAX boolean arrays for init & clamp states."""
        init_jax = [jnp.asarray(s, dtype=bool) for s in init_state]
        clamp_jax = [jnp.asarray(c, dtype=bool) for c in clamped_values]
        return _thrml_sample_states(
            key,
            program,
            schedule,
            init_jax,
            clamp_jax,
            list(readout_blocks),
        )
else:
    SpinNode = _RefSpinNode
    Block = _RefBlock
    SamplingSchedule = _RefSamplingSchedule
    IsingEBM = _RefIsingEBM
    IsingSamplingProgram = _RefIsingSamplingProgram
    hinton_init = _ref_hinton_init
    sample_states = _ref_sample_states


# ---------------------------------------------------------------------------
# Energy & Exact Boltzmann Helper Functions
# ---------------------------------------------------------------------------


def bool_to_spins(state_bool: Any) -> Any:
    """Convert boolean spin representation (True/False) to {-1.0, +1.0}."""
    return jnp.where(jnp.asarray(state_bool), 1.0, -1.0)


def spins_to_bool(spins: Any) -> Any:
    """Convert {-1.0, +1.0} spin values to boolean (True for +1, False for -1)."""
    return jnp.asarray(spins) > 0


def extract_model_matrices(
    nodes: Sequence[Any],
    edges: Sequence[tuple[Any, Any]],
    biases: Any,
    weights: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (biases_np, symmetric_coupling_matrix_J) for any IsingEBM specification."""
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    n = len(nodes)
    j_mat = np.zeros((n, n), dtype=np.float32)
    w_np = np.asarray(weights, dtype=np.float32)
    b_np = np.asarray(biases, dtype=np.float32)
    for k, (u, v) in enumerate(edges):
        ui = node_to_idx[u]
        vi = node_to_idx[v]
        j_mat[ui, vi] += w_np[k]
        j_mat[vi, ui] += w_np[k]
    return b_np, j_mat


def compute_energy(
    spins_or_bool: Any,
    nodes: Sequence[Any],
    edges: Sequence[tuple[Any, Any]],
    biases: Any,
    weights: Any,
) -> np.ndarray:
    """Compute scalar Ising energy E(s) = - sum_{(i,j)} W_{ij} s_i s_j - sum_i b_i s_i.

    Accepts either a 1D array `(n_nodes,)` or 2D batch `(n_samples, n_nodes)` of booleans or {-1, +1} spins.
    """
    arr = np.asarray(spins_or_bool)
    if arr.dtype == bool:
        spins = np.where(arr, 1.0, -1.0).astype(np.float32)
    else:
        spins = arr.astype(np.float32)

    single = spins.ndim == 1
    if single:
        spins = spins[None, :]

    b_np, j_mat = extract_model_matrices(nodes, edges, biases, weights)
    # Pair energy: -0.5 * sum_{i,j} s_i J_{ij} s_j (since J is symmetric)
    interaction_energy = -0.5 * np.sum((spins @ j_mat) * spins, axis=1)
    bias_energy = -(spins @ b_np)
    total = interaction_energy + bias_energy
    return total[0] if single else total


def exact_boltzmann_distribution(
    nodes: Sequence[Any],
    edges: Sequence[tuple[Any, Any]],
    biases: Any,
    weights: Any,
    beta: float,
) -> dict[str, Any]:
    """Brute-force the exact partition function Z(theta) and state probabilities
    for low-dimensional factor graphs (N <= 20).

    Returns a dictionary containing:
    - `states_spins`: (2^N, N) array of all spin states in {-1, +1}
    - `states_bool`: (2^N, N) boolean array
    - `energies`: (2^N,) exact energy E(s) for each state
    - `probs`: (2^N,) exact Boltzmann probability P(s) = exp(-beta * E(s)) / Z
    - `log_Z`: log partition function log(Z)
    - `Z`: scalar partition function Z
    - `expected_energy`: theoretical <E>
    """
    n = len(nodes)
    if n > 20:
        raise ValueError(f"Brute-force partition function requires N <= 20, got N={n}.")

    all_bool = np.array(list(itertools.product([False, True], repeat=n)), dtype=bool)
    all_spins = np.where(all_bool, 1.0, -1.0).astype(np.float32)
    energies = compute_energy(all_spins, nodes, edges, biases, weights)

    neg_beta_e = -float(beta) * energies
    max_val = np.max(neg_beta_e)
    log_z = float(max_val + np.log(np.sum(np.exp(neg_beta_e - max_val))))
    z_val = float(math.exp(log_z)) if log_z < 80.0 else float("inf")
    probs = np.exp(neg_beta_e - log_z)
    expected_energy = float(np.sum(probs * energies))

    return {
        "states_spins": all_spins,
        "states_bool": all_bool,
        "energies": energies,
        "probs": probs,
        "log_Z": log_z,
        "Z": z_val,
        "expected_energy": expected_energy,
    }
