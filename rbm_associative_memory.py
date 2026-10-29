"""
Experiment 6: Associative Memory & Prototype Recall via Restricted Boltzmann Machine (RBM)
Stores prototype binary patterns in visible-to-hidden bipartite couplings and recovers
orthogonal memories from occluded inputs using 2-block parallel Gibbs sampling in Extropic THRML.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

def generate_orthogonal_prototypes(size: int = 8):
    """Generates 3 distinct binary 8x8 prototype patterns."""
    P1 = np.ones((size, size), dtype=np.int32) * -1
    P1[:, 3:5] = 1
    P1[3:5, :] = 1  # Plus symbol

    P2 = np.ones((size, size), dtype=np.int32) * -1
    for r in range(size):
        P2[r, r] = 1
        P2[r, size - 1 - r] = 1  # Cross symbol

    P3 = np.ones((size, size), dtype=np.int32) * -1
    P3[1:7, 1:7] = 1
    P3[2:6, 2:6] = -1  # Hollow square

    return [P1.flatten(), P2.flatten(), P3.flatten()]

def train_hebbian_rbm(prototypes, n_hidden: int = 32):
    """
    Computes visible-hidden weight matrix using generalized Hebbian projection.
    Maps N_vis (64) to N_hid (32).
    """
    n_vis = len(prototypes[0])
    rng = np.random.default_rng(1337)
    
    # Project prototypes onto random orthogonal hidden representations
    H_proj = rng.choice([-1.0, 1.0], size=(len(prototypes), n_hidden))
    
    W = np.zeros((n_vis, n_hidden), dtype=np.float32)
    for p_idx in range(len(prototypes)):
        v = prototypes[p_idx].astype(np.float32)
        h = H_proj[p_idx]
        W += np.outer(v, h) / float(len(prototypes))
        
    return W

def build_rbm_program(W: np.ndarray, n_clamped_vis: int = 32, beta_val: float = 2.0):
    n_vis, n_hid = W.shape
    vis_nodes = [SpinNode() for _ in range(n_vis)]
    hid_nodes = [SpinNode() for _ in range(n_hid)]
    
    all_nodes = vis_nodes + hid_nodes
    edges = []
    weights_list = []
    
    # Fully bipartite visible-hidden interactions (no visible-visible or hidden-hidden edges)
    for i in range(n_vis):
        for j in range(n_hid):
            edges.append((vis_nodes[i], hid_nodes[j]))
            weights_list.append(W[i, j])
            
    biases = jnp.zeros((len(all_nodes),), dtype=jnp.float32)
    weights = jnp.array(weights_list, dtype=jnp.float32)
    beta = jnp.array(beta_val, dtype=jnp.float32)
    
    ebm = IsingEBM(nodes=all_nodes, edges=edges, biases=biases, weights=weights, beta=beta)
    
    # Clamp the known (top-half) visible spins and freely sample the occluded (bottom-half) visible + hidden blocks
    clamped_vis_block = Block(vis_nodes[:n_clamped_vis])
    free_vis_block = Block(vis_nodes[n_clamped_vis:])
    hid_block = Block(hid_nodes)
    
    free_blocks = [free_vis_block, hid_block]
    program = IsingSamplingProgram(ebm, free_blocks=free_blocks, clamped_blocks=[clamped_vis_block])
    
    return program, ebm, free_blocks, vis_nodes, hid_nodes

def main():
    size = 8
    n_vis = size * size
    n_hid = 32
    n_clamped = n_vis // 2
    print(f"[*] Initializing RBM ({n_vis} visible nodes, {n_hid} hidden nodes)...")
    
    prototypes = generate_orthogonal_prototypes(size=size)
    W = train_hebbian_rbm(prototypes, n_hidden=n_hid)
    
    program, ebm, free_blocks, vis_nodes, hid_nodes = build_rbm_program(
        W, n_clamped_vis=n_clamped, beta_val=2.2
    )
    
    # Occlude 50% of Prototype 1 (mask the bottom half with random noise)
    target_proto = prototypes[0].copy()
    occluded_input = target_proto.copy()
    rng = np.random.default_rng(42)
    occluded_input[n_clamped:] = rng.choice([-1, 1], size=n_vis - n_clamped)
    
    print("[*] Running Associative Memory Recall via Bipartite Gibbs Sampling in THRML...")
    key = jax.random.key(101)
    k_init, k_samp = jax.random.split(key, 2)
    
    # Initialize free blocks with the noisy bottom-half query and hinton_init for hidden units
    init_state = hinton_init(k_init, ebm, free_blocks, ())
    init_state[0] = jnp.array(occluded_input[n_clamped:] == 1, dtype=jnp.bool_)
    clamped_state = [jnp.array(occluded_input[:n_clamped] == 1, dtype=jnp.bool_)]
    
    schedule = SamplingSchedule(n_warmup=150, n_samples=300, steps_per_sample=2)
    
    start_time = time.perf_counter()
    raw_samples = sample_states(k_samp, program, schedule, init_state, clamped_state, [Block(vis_nodes)])
    elapsed = time.perf_counter() - start_time
    
    vis_samples = 2 * np.array(raw_samples[0], dtype=np.float32) - 1.0
    # Average across Gibbs trajectory to find recalled ground state
    recalled_pattern = np.sign(np.mean(vis_samples[-50:], axis=0)).astype(np.int32)
    
    # Overlap / Cosine similarity with true prototype
    overlap = np.dot(recalled_pattern, target_proto) / float(n_vis)
    occluded_acc = np.mean(recalled_pattern[n_clamped:] == target_proto[n_clamped:]) * 100.0
    print(f"[+] Sampling completed in {elapsed:.3f}s")
    print(f"[+] Memory Recall Overlap with Ground Truth: {overlap * 100:.1f}% (Occluded Half Recovery: {occluded_acc:.1f}%)")
    
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    axes[0].imshow(target_proto.reshape((size, size)), cmap="gray", vmin=-1, vmax=1)
    axes[0].set_title("Ground Truth Memory")
    axes[0].axis("off")
    
    axes[1].imshow(occluded_input.reshape((size, size)), cmap="gray", vmin=-1, vmax=1)
    axes[1].set_title("Occluded Query (50% Noise)")
    axes[1].axis("off")
    
    axes[2].imshow(recalled_pattern.reshape((size, size)), cmap="gray", vmin=-1, vmax=1)
    axes[2].set_title(f"Recalled Memory ({overlap * 100:.0f}% Overlap)")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.savefig("thrml_rbm_memory.png", dpi=200)
    print("[+] Saved visualization artifact: thrml_rbm_memory.png")

if __name__ == "__main__":
    main()
