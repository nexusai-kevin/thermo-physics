"""
Experiment 5: Discrete Image Denoising via Clamped Thermodynamic Sampling in Extropic THRML
Reconstructs corrupted binary pixel patterns by coupling a latent 2D Ising prior
to a clamped observation layer.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

def generate_synthetic_digit(L: int = 28) -> np.ndarray:
    """Creates a high-contrast binary synthetic shape (28x28)."""
    img = np.zeros((L, L), dtype=np.int32)
    # Outer frame
    img[4:24, 6:10] = 1
    img[4:24, 18:22] = 1
    img[4:8, 6:22] = 1
    img[20:24, 6:22] = 1
    # Center crossbar
    img[12:16, 6:22] = 1
    return 2 * img - 1  # Map {0, 1} to {-1, +1}

def add_pixel_noise(clean_img: np.ndarray, flip_prob: float = 0.20, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    flips = rng.random(clean_img.shape) < flip_prob
    noisy = clean_img.copy()
    noisy[flips] *= -1
    return noisy

def build_denoising_ebm(noisy_img: np.ndarray, J_spatial: float = 1.0, J_obs: float = 1.8, beta_val: float = 1.2):
    L = noisy_img.shape[0]
    
    # 1. Create Free Latent Spins (x) and Clamped Observation Spins (y)
    latent_grid = [[SpinNode() for _ in range(L)] for _ in range(L)]
    obs_grid = [[SpinNode() for _ in range(L)] for _ in range(L)]
    
    flat_latent = [latent_grid[r][c] for r in range(L) for c in range(L)]
    flat_obs = [obs_grid[r][c] for r in range(L) for c in range(L)]
    
    # Partition latent spins into bipartite checkerboard blocks
    even_latent = [latent_grid[r][c] for r in range(L) for c in range(L) if (r + c) % 2 == 0]
    odd_latent = [latent_grid[r][c] for r in range(L) for c in range(L) if (r + c) % 2 == 1]
    
    free_blocks = [Block(even_latent), Block(odd_latent)]
    clamped_block = Block(flat_obs)
    
    # 2. Construct Edges & Couplings
    edges = []
    weights_list = []
    
    # Spatial smoothness prior between neighboring latent spins (periodic torus)
    for r in range(L):
        for c in range(L):
            u = latent_grid[r][c]
            r_next = latent_grid[(r + 1) % L][c]
            c_next = latent_grid[r][(c + 1) % L]
            
            edges.append((u, r_next))
            weights_list.append(J_spatial)
            edges.append((u, c_next))
            weights_list.append(J_spatial)
            
    # Observation coupling: each latent spin x_i is tied to noisy pixel y_i
    for r in range(L):
        for c in range(L):
            edges.append((latent_grid[r][c], obs_grid[r][c]))
            weights_list.append(J_obs)
            
    all_nodes = flat_latent + flat_obs
    biases = jnp.zeros((len(all_nodes),), dtype=jnp.float32)
    weights = jnp.array(weights_list, dtype=jnp.float32)
    beta = jnp.array(beta_val, dtype=jnp.float32)
    
    ebm = IsingEBM(nodes=all_nodes, edges=edges, biases=biases, weights=weights, beta=beta)
    program = IsingSamplingProgram(ebm, free_blocks=free_blocks, clamped_blocks=[clamped_block])
    
    return program, ebm, free_blocks, clamped_block, flat_latent, flat_obs

def main():
    L = 28
    noise_ratio = 0.20
    print(f"[*] Generating {L}x{L} ground truth image with {int(noise_ratio*100)}% noise injection...")
    
    clean_img = generate_synthetic_digit(L=L)
    noisy_img = add_pixel_noise(clean_img, flip_prob=noise_ratio, seed=42)
    
    print("[*] Compiling Clamped Conditioning Program in THRML...")
    program, ebm, free_blocks, clamped_block, flat_latent, flat_obs = build_denoising_ebm(
        noisy_img, J_spatial=1.0, J_obs=1.8, beta_val=1.2
    )
    
    # Pack clamped observation states: boolean array (True for +1, False for -1)
    flat_noisy = noisy_img.flatten()
    clamped_values = jnp.array(flat_noisy == 1, dtype=jnp.bool_)
    
    key = jax.random.key(2026)
    k_init, k_samp = jax.random.split(key, 2)
    
    # Initialize free latent spins with hinton_init (batch_shape=())
    init_state = hinton_init(k_init, ebm, free_blocks, ())
    schedule = SamplingSchedule(n_warmup=150, n_samples=300, steps_per_sample=2)
    
    print("[*] Running Thermodynamic Denoising Sweep...")
    start_time = time.perf_counter()
    raw_samples = sample_states(
        k_samp, program, schedule, init_state, [clamped_values], [Block(flat_latent)]
    )
    elapsed = time.perf_counter() - start_time
    
    # Average across samples to compute marginal spin expectations in [-1, +1]
    latent_samples = 2 * np.array(raw_samples[0], dtype=np.float32) - 1.0
    mean_reconstruction = np.mean(latent_samples, axis=0).reshape((L, L))
    binarized_output = np.where(mean_reconstruction >= 0, 1, -1)
    
    # Compute accuracy metrics
    corrupted_err = np.mean(noisy_img != clean_img) * 100
    recovered_err = np.mean(binarized_output != clean_img) * 100
    
    print(f"[+] Denoising completed in {elapsed:.3f}s")
    print(f"[+] Initial Corrupted Error: {corrupted_err:.2f}%")
    print(f"[+] Restored Reconstruction Error: {recovered_err:.2f}%")
    
    # Save visual comparison artifact
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(clean_img, cmap="gray", vmin=-1, vmax=1)
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")
    
    axes[1].imshow(noisy_img, cmap="gray", vmin=-1, vmax=1)
    axes[1].set_title(f"Corrupted ({corrupted_err:.1f}% Noise)")
    axes[1].axis("off")
    
    axes[2].imshow(binarized_output, cmap="gray", vmin=-1, vmax=1)
    axes[2].set_title(f"Restored Latent (Error: {recovered_err:.1f}%)")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.savefig("thrml_image_denoising.png", dpi=200)
    print("[+] Saved visualization artifact: thrml_image_denoising.png")

if __name__ == "__main__":
    main()
