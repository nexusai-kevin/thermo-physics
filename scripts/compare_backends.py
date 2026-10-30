import numpy as np
from thermo_physics.backends.sampler_adapter import (
    BackendType,
    HamiltonianGraph,
    SamplingConfig,
    get_sampler,
)

def run_verification():
    N = 24  # Spin dimension matching Station 10
    rng = np.random.default_rng(42)

    # Random SK spin glass coupling matrix J ~ N(0, 1/N)
    J = rng.normal(0.0, 1.0 / np.sqrt(N), size=(N, N))
    J = (J + J.T) / 2.0
    np.fill_diagonal(J, 0.0)
    h = rng.normal(0.0, 0.1, size=(N,))

    graph = HamiltonianGraph(couplings=J, biases=h, spin_type="bipolar")
    config = SamplingConfig(n_samples=5000, burn_in=500, beta=1.5, seed=1337)

    # 1. Classical JAX Gibbs Backend
    jax_sampler = get_sampler(BackendType.JAX_BLOCK_GIBBS)
    jax_result = jax_sampler.sample(graph, config)
    print("--- Classical JAX Results ---")
    print(jax_result.summary())

    # 2. Extropic TSU Backend (Emulated / Drop-in Silicon Target)
    hw_sampler = get_sampler(BackendType.EXTROPIC_TSU, mock_mode=True)
    hw_result = hw_sampler.sample(graph, config)
    print("\n--- Extropic Silicon Adapter Results ---")
    print(hw_result.summary())

if __name__ == "__main__":
    run_verification()
