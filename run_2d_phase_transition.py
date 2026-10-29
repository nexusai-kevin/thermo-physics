"""
2D Ising Grid Phase Transition & Critical Opacity in Extropic THRML
Demonstrating spontaneous symmetry breaking across Onsager's critical point (beta_c ≈ 0.4407).
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

def build_2d_lattice(L: int = 32, J: float = 1.0, beta_val: float = 0.44):
    """
    Constructs an L x L periodic 2D Ising grid mapped to two checkerboard blocks
    for parallel 2-color block Gibbs sampling.
    """
    nodes_2d = [[SpinNode() for _ in range(L)] for _ in range(L)]
    flat_nodes = [nodes_2d[r][c] for r in range(L) for c in range(L)]
    
    # Partition into bipartite checkerboard blocks (Even: r+c % 2 == 0, Odd: r+c % 2 == 1)
    even_nodes = []
    odd_nodes = []
    for r in range(L):
        for c in range(L):
            if (r + c) % 2 == 0:
                even_nodes.append(nodes_2d[r][c])
            else:
                odd_nodes.append(nodes_2d[r][c])
                
    free_blocks = [Block(even_nodes), Block(odd_nodes)]
    
    # Periodic nearest-neighbor edges (Right and Down)
    edges = []
    for r in range(L):
        for c in range(L):
            u = nodes_2d[r][c]
            r_next = nodes_2d[(r + 1) % L][c]
            c_next = nodes_2d[r][(c + 1) % L]
            edges.append((u, r_next))
            edges.append((u, c_next))
            
    num_nodes = L * L
    num_edges = len(edges)
    
    biases = jnp.zeros((num_nodes,), dtype=jnp.float32)
    weights = jnp.ones((num_edges,), dtype=jnp.float32) * J
    beta = jnp.array(beta_val, dtype=jnp.float32)
    
    model = IsingEBM(flat_nodes, edges, biases, weights, beta)
    program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])
    
    return model, program, free_blocks, flat_nodes

def sample_magnetization(program, model, free_blocks, flat_nodes, key, n_samples=1000, warmup=200):
    k_init, k_samp = jax.random.split(key, 2)
    init_state = hinton_init(k_init, model, free_blocks, ())
    
    # 2 steps per sample guarantees a full sweep over both checkerboard sublattices
    schedule = SamplingSchedule(n_warmup=warmup, n_samples=n_samples, steps_per_sample=2)
    raw_samples = sample_states(k_samp, program, schedule, init_state, [], [Block(flat_nodes)])
    
    # THRML samples are boolean (False: -1, True: +1)
    spins = 2 * jnp.asarray(raw_samples[0], dtype=jnp.float32) - 1.0  # shape: (n_samples, L*L)
    
    # Calculate |M| per sample to capture spontaneous ordering without symmetry cancellation
    m_per_sample = jnp.abs(jnp.mean(spins, axis=-1))
    avg_m = float(jnp.mean(m_per_sample))
    
    # Extract the last lattice state for spatial domain visualization
    last_state = np.array(spins[-1])
    return avg_m, last_state

def main():
    L = 32
    print(f"[*] Initializing {L}x{L} 2D Lattice ({L*L} spins)...")
    
    # Onsager's analytical critical point: sinh(2*beta_c) = 1 => beta_c ≈ 0.4406868
    beta_c = 0.5 * math.log(1.0 + math.sqrt(2.0))
    print(f"[*] Onsager Critical Point (J=1.0): beta_c = {beta_c:.4f}")
    
    beta_sweep = [0.20, 0.35, 0.44, 0.50, 0.70]
    m_results = []
    snapshots = {}
    
    master_key = jax.random.key(42)
    
    print("\n--- Running Beta Sweep ---")
    for b in beta_sweep:
        master_key, subkey = jax.random.split(master_key)
        model, program, free_blocks, flat_nodes = build_2d_lattice(L=L, J=1.0, beta_val=b)
        
        m_val, last_grid = sample_magnetization(
            program, model, free_blocks, flat_nodes, subkey, n_samples=1200, warmup=300
        )
        m_results.append(m_val)
        snapshots[b] = last_grid.reshape((L, L))
        print(f"beta = {b:.2f} | |<M>| = {m_val:.4f} {'(Ordered Phase)' if b > beta_c else '(Disordered Phase)'}")
        
    # Generate verification figure
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Plot 1: Magnetization vs Beta
    axes[0].plot(beta_sweep, m_results, 'o-', color='#3b82f6', lw=2)
    axes[0].axvline(beta_c, color='#ef4444', linestyle='--', label=f'Onsager βc ≈ {beta_c:.3f}')
    axes[0].set_title("Order Parameter |<M>| vs Beta")
    axes[0].set_xlabel("Inverse Temp (β)")
    axes[0].set_ylabel("|<M>|")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plots 2-4: Spatial Domain States
    vis_betas = [0.20, 0.44, 0.70]
    titles = [f"β = 0.20 (Disordered)", f"β = 0.44 (Critical Point)", f"β = 0.70 (Domain Ordered)"]
    for idx, (vb, title) in enumerate(zip(vis_betas, titles), start=1):
        im = axes[idx].imshow(snapshots[vb], cmap='coolwarm', vmin=-1, vmax=1)
        axes[idx].set_title(title)
        axes[idx].axis('off')
        
    plt.tight_layout()
    plt.savefig("thrml_2d_phase_transition.png", dpi=200)
    print("\n[+] Verification artifact saved: thrml_2d_phase_transition.png")

if __name__ == "__main__":
    main()
