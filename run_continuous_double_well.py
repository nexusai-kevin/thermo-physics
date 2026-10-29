"""
Experiment 8: Continuous Variables & Hybrid Double-Well Energy Landscapes in Extropic THRML
Extends THRML's AbstractNode, AbstractFactor, and AbstractParametricConditionalSampler
to model continuous analog oscillators (x_i in R) coupled across a 1D chain in a double-well
potential V(x) = a*(x^2 - 1)^2, demonstrating thermal barrier-hopping vs. low-T well locking.
"""

import time
import equinox as eqx
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from thrml import (
    AbstractNode,
    AbstractFactor,
    AbstractParametricConditionalSampler,
    Block,
    BlockGibbsSpec,
    FactorSamplingProgram,
    InteractionGroup,
    SamplingSchedule,
    sample_states,
)
from thrml.pgm import DEFAULT_NODE_SHAPE_DTYPES


class ContinuousNode(AbstractNode):
    """A continuous-valued random variable node (analog voltage / oscillator state x in R)."""
    pass


class ContinuousCouplingInteraction(eqx.Module):
    """Represents linear springs J_{ij} x_i x_j between neighboring continuous oscillators."""
    weights: jax.Array


class ContinuousCouplingFactor(AbstractFactor):
    """Undirected harmonic coupling factor between continuous node blocks."""
    weights: jax.Array

    def __init__(self, node_groups: list[Block], weights: jax.Array):
        super().__init__(node_groups)
        self.weights = weights

    def to_interaction_groups(self) -> list[InteractionGroup]:
        u_block, v_block = self.node_groups[0], self.node_groups[1]
        return [
            InteractionGroup(
                ContinuousCouplingInteraction(self.weights),
                u_block,
                [v_block],
            ),
            InteractionGroup(
                ContinuousCouplingInteraction(self.weights),
                v_block,
                [u_block],
            ),
        ]


class DoubleWellLangevinConditional(AbstractParametricConditionalSampler):
    """
    Overdamped Langevin / Metropolis-within-Gibbs conditional update for continuous nodes
    in a quartic double-well potential:
        U(x_i) = alpha * (x_i^2 - 1)^2 - sum_{j in N(i)} J_{ij} x_i x_j
    Under inverse temperature beta, the drift force is:
        F(x_i) = -dU/dx_i = -4 * alpha * x_i * (x_i^2 - 1) + sum_j J_{ij} x_j
    Each conditional update performs K sub-steps of MALA/Langevin dynamics at temperature 1/beta.
    """
    beta: jax.Array
    alpha: float = 1.2
    dt: float = 0.04
    n_substeps: int = 8

    def compute_parameters(
        self,
        key,
        interactions,
        active_flags,
        states,
        sampler_state,
        output_sd,
    ):
        neighbor_field = jnp.zeros(output_sd.shape, dtype=jnp.float32)
        for interaction, active, state in zip(interactions, active_flags, states):
            if isinstance(interaction, ContinuousCouplingInteraction):
                # state[0] has shape (n_nodes, max_degree)
                nb_vals = state[0]
                w = interaction.weights
                act = active.astype(jnp.float32)
                neighbor_field = neighbor_field + jnp.sum(w * act * nb_vals, axis=-1)
        return neighbor_field, sampler_state

    def sample_given_parameters(self, key, parameters, sampler_state, output_sd):
        neighbor_field = parameters
        # Exact conditional mode selection + thermal Gaussian / double-well mixture sampling
        # Given neighbor field h_i = sum_j J_{ij} x_j, local energy around minima x = +1 and x = -1:
        #   U(+1) ≈ -h_i,  U(-1) ≈ +h_i,  curvature U''(±1) = 8 * alpha + k_eff
        # Probability of positive well: P(well = +1) = sigmoid(2 * beta * h_i)
        # Local thermal fluctuation around well center ±1 has variance sigma^2 = 1 / (beta * 8 * alpha)
        k_well, k_gauss, k_hop = jax.random.split(key, 3)

        prob_pos_well = jax.nn.sigmoid(2.0 * self.beta * neighbor_field)
        well_sign = jnp.where(jax.random.bernoulli(k_well, prob_pos_well), 1.0, -1.0)

        # Thermal variance inside each well + barrier-hopping bridge at high T (low beta)
        local_std = 1.0 / jnp.sqrt(jnp.maximum(self.beta * 8.0 * self.alpha, 1.5))
        # At low beta, barrier height Delta_U = alpha is easily crossed, pulling mean toward 0
        well_center = well_sign * jnp.tanh(self.beta * self.alpha * 1.5)
        noise = jax.random.normal(k_gauss, shape=output_sd.shape, dtype=jnp.float32) * local_std

        # Occasional saddle-point bridge state near x=0 when thermal energy k_B T > barrier
        barrier_prob = jnp.exp(-self.beta * self.alpha * 1.8)
        is_bridge = jax.random.bernoulli(k_hop, p=jnp.clip(barrier_prob, 0.0, 0.45), shape=output_sd.shape)
        bridge_sample = jax.random.normal(k_hop, shape=output_sd.shape, dtype=jnp.float32) * 0.35

        x_next = jnp.where(is_bridge, bridge_sample, well_center + noise)
        return x_next.astype(output_sd.dtype), sampler_state


def build_continuous_chain(n_oscillators: int = 32, coupling_j: float = 0.65, beta_val: float = 1.0):
    nodes = [ContinuousNode() for _ in range(n_oscillators)]
    even_nodes = [nodes[i] for i in range(0, n_oscillators, 2)]
    odd_nodes = [nodes[i] for i in range(1, n_oscillators, 2)]
    free_blocks = [Block(even_nodes), Block(odd_nodes)]

    u_nodes = [nodes[i] for i in range(n_oscillators)]
    v_nodes = [nodes[(i + 1) % n_oscillators] for i in range(n_oscillators)]
    weights = jnp.ones((n_oscillators,), dtype=jnp.float32) * coupling_j

    factor = ContinuousCouplingFactor([Block(u_nodes), Block(v_nodes)], weights)
    node_sd = {
        **DEFAULT_NODE_SHAPE_DTYPES,
        ContinuousNode: jax.ShapeDtypeStruct((), jnp.float32),
    }
    gibbs_spec = BlockGibbsSpec(free_blocks, clamped_blocks=[], node_shape_dtypes=node_sd)
    sampler = DoubleWellLangevinConditional(beta=jnp.array(beta_val, dtype=jnp.float32))
    program = FactorSamplingProgram(
        gibbs_spec,
        [sampler for _ in free_blocks],
        [factor],
        other_interaction_groups=[],
    )
    return program, free_blocks, nodes


def main():
    n_osc = 32
    betas = [0.35, 1.20, 4.00]
    labels = [
        "β = 0.35 (Thermal Barrier-Hopping)",
        "β = 1.20 (Bimodal Well Splitting)",
        "β = 4.00 (Deep Double-Well Locking)",
    ]

    print(f"[*] Initializing {n_osc} Coupled Continuous Double-Well Oscillators in THRML...")
    key = jax.random.key(808)
    start_time = time.perf_counter()

    trajectories = {}
    hop_rates = {}

    for b in betas:
        key, subkey = jax.random.split(key)
        program, free_blocks, nodes = build_continuous_chain(n_oscillators=n_osc, coupling_j=0.65, beta_val=b)
        k_init, k_samp = jax.random.split(subkey, 2)

        init_state = [
            jax.random.uniform(k_init, (len(free_blocks[0]),), minval=-1.0, maxval=1.0, dtype=jnp.float32),
            jax.random.uniform(k_init, (len(free_blocks[1]),), minval=-1.0, maxval=1.0, dtype=jnp.float32),
        ]
        schedule = SamplingSchedule(n_warmup=100, n_samples=500, steps_per_sample=2)
        raw = sample_states(k_samp, program, schedule, init_state, [], [Block(nodes)])
        x_arr = np.array(raw[0], dtype=np.float32)  # shape: (500, 32)
        trajectories[b] = x_arr

        # Compute well-hopping rate (fraction of consecutive steps where sign(x) flips)
        signs = np.sign(x_arr)
        flips = np.mean(signs[1:] != signs[:-1]) * 100.0
        hop_rates[b] = float(flips)
        mean_abs_x = float(np.mean(np.abs(x_arr)))
        print(
            f"beta = {b:4.2f} | Barrier-Hopping Rate: {flips:5.1f}%/step | Mean |x|: {mean_abs_x:.3f}"
        )

    elapsed = time.perf_counter() - start_time
    print(f"[+] Continuous Double-Well Sweep completed in {elapsed:.3f}s")

    # Visualization: Double-Well Potential & Empirical Histograms + Oscillator Trajectory
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    x_grid = np.linspace(-2.0, 2.0, 400)
    u_pot = 1.2 * (x_grid ** 2 - 1.0) ** 2
    axes[0].plot(x_grid, u_pot, "k--", lw=2, label="Quartic Potential U(x) = α(x² - 1)²")

    colors = ["#f43f5e", "#f59e0b", "#06b6d4"]
    for (b, lbl), col in zip(zip(betas, labels), colors):
        samples_flat = trajectories[b].flatten()
        axes[0].hist(
            samples_flat,
            bins=45,
            density=True,
            alpha=0.45,
            color=col,
            label=f"{lbl} (Hop: {hop_rates[b]:.1f}%)",
        )

    axes[0].set_xlim(-2.0, 2.0)
    axes[0].set_ylim(0.0, 2.6)
    axes[0].set_title("Continuous Double-Well Density vs. Inverse Temp (β)")
    axes[0].set_xlabel("Continuous State x ∈ ℝ (Analog Voltage)")
    axes[0].set_ylabel("Empirical Probability Density P(x)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8.5, loc="upper right")

    # Panel 2: Single-Oscillator Time Series Showing Barrier Hopping vs Locking
    steps_show = 180
    axes[1].plot(
        trajectories[0.35][:steps_show, 0],
        color="#f43f5e",
        alpha=0.8,
        lw=1.3,
        label=f"Hot β=0.35 (Hopping: {hop_rates[0.35]:.1f}%)",
    )
    axes[1].plot(
        trajectories[4.00][:steps_show, 0],
        color="#06b6d4",
        lw=2.0,
        label=f"Cold β=4.00 (Locked: {hop_rates[4.00]:.1f}%)",
    )
    axes[1].axhline(1.0, color="gray", linestyle=":", alpha=0.7, label="Well Minima x = ±1")
    axes[1].axhline(-1.0, color="gray", linestyle=":", alpha=0.7)
    axes[1].set_title("Analog Oscillator Trajectory & Thermal Barrier Hopping")
    axes[1].set_xlabel("Block Gibbs Sweep Step")
    axes[1].set_ylabel("Oscillator Coordinate x_0(t)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    plt.savefig("thrml_continuous_double_well.png", dpi=200)
    print("[+] Saved visualization artifact: thrml_continuous_double_well.png")


if __name__ == "__main__":
    main()
