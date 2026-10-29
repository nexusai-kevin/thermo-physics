"""
Experiment 4: Multistate Potts Model (K=4) in Extropic THRML
Demonstrating categorical domain clustering with CategoricalNode & CategoricalEBMFactor.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import jax
import jax.numpy as jnp

from thrml import Block, BlockGibbsSpec, SamplingSchedule, sample_states
from thrml.pgm import CategoricalNode
from thrml.models.discrete_ebm import CategoricalEBMFactor, CategoricalGibbsConditional
from thrml.factor import FactorSamplingProgram

def build_potts_grid(L: int = 24, n_cats: int = 4, beta_val: float = 1.8):
    """
    Builds an L x L periodic torus with K-state categorical spins
    partitioned into 2-color bipartite blocks.
    """
    G = nx.grid_2d_graph(L, L, periodic=True)
    coord_to_node = {c: CategoricalNode() for c in G.nodes()}
    flat_nodes = [coord_to_node[c] for c in G.nodes()]
    
    # 2-color bipartite checkerboard blocks
    even_nodes = [coord_to_node[(r, c)] for r, c in G.nodes() if (r + c) % 2 == 0]
    odd_nodes = [coord_to_node[(r, c)] for r, c in G.nodes() if (r + c) % 2 == 1]
    free_blocks = [Block(even_nodes), Block(odd_nodes)]
    
    # Directed edges (Right and Down)
    u_nodes = []
    v_nodes = []
    for (r, c) in G.nodes():
        u = coord_to_node[(r, c)]
        r_next = coord_to_node[((r + 1) % L, c)]
        c_next = coord_to_node[(r, (c + 1) % L)]
        u_nodes.extend([u, u])
        v_nodes.extend([r_next, c_next])
        
    # Standard Potts coupling: reward identical colors with energy -beta
    id_mat = jnp.eye(n_cats, dtype=jnp.float32)
    weights = beta_val * jnp.broadcast_to(jnp.expand_dims(id_mat, 0), (len(u_nodes), n_cats, n_cats))
    
    factor = CategoricalEBMFactor([Block(u_nodes), Block(v_nodes)], weights)
    conditionals = [CategoricalGibbsConditional(n_categories=n_cats) for _ in free_blocks]
    gibbs_spec = BlockGibbsSpec(free_blocks, clamped_blocks=[])
    program = FactorSamplingProgram(gibbs_spec, conditionals, [factor], other_interaction_groups=[])
    
    return program, free_blocks, flat_nodes, L

def main():
    L = 24
    K = 4
    betas = [0.4, 1.2, 2.2]
    
    print(f"[*] Initializing {L}x{L} Potts Lattice (K={K} categories, {L*L} nodes)...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    key = jax.random.key(777)
    start_time = time.perf_counter()
    
    for idx, b in enumerate(betas):
        key, subkey = jax.random.split(key)
        program, free_blocks, flat_nodes, _ = build_potts_grid(L=L, n_cats=K, beta_val=b)
        
        k_init, k_samp = jax.random.split(subkey, 2)
        n_even = len(free_blocks[0])
        n_odd = len(free_blocks[1])
        init_state = [
            jax.random.randint(k_init, (n_even,), 0, K, dtype=jnp.uint8),
            jax.random.randint(k_init, (n_odd,), 0, K, dtype=jnp.uint8)
        ]
        
        schedule = SamplingSchedule(n_warmup=200, n_samples=300, steps_per_sample=2)
        samples = sample_states(k_samp, program, schedule, init_state, [], [Block(flat_nodes)])
        
        final_lattice = np.array(samples[0][-1]).reshape((L, L))
        
        axes[idx].imshow(final_lattice, cmap="tab10", vmin=0, vmax=K-1)
        status = "Disordered Gas" if b < 0.8 else "Domain Formation" if b < 1.8 else "Macroscopic Domains"
        axes[idx].set_title(f"β = {b} ({status})")
        axes[idx].axis("off")
        print(f"beta = {b} -> Sampled 300 lattice draws successfully.")
        
    elapsed = time.perf_counter() - start_time
    print(f"[+] Multi-temperature sweep complete in {elapsed:.3f}s")
    
    plt.tight_layout()
    plt.savefig("thrml_potts_k4.png", dpi=200)
    print("[+] Saved visualization artifact: thrml_potts_k4.png")

if __name__ == "__main__":
    main()
