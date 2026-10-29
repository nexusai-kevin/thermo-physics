"""
Day 3: NP-Hard Max-Cut Combinatorial Optimizer in Extropic THRML
Maps Max-Cut to an antiferromagnetic Ising Hamiltonian sampled via chromatic Block Gibbs in JAX.
"""

import time
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

def generate_problem_graph(n_nodes: int = 40, edge_prob: float = 0.25, seed: int = 42):
    """Generates an Erdős–Rényi graph G(n, p)."""
    G = nx.erdos_renyi_graph(n=n_nodes, p=edge_prob, seed=seed)
    # Ensure graph is connected
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for i in range(len(components) - 1):
            u = list(components[i])[0]
            v = list(components[i+1])[0]
            G.add_edge(u, v)
    return G

def build_max_cut_ebm(G: nx.Graph, beta_val: float = 2.5):
    """
    Translates graph cut objective into an antiferromagnetic Ising model (J = -1.0)
    partitioned by greedy chromatic vertex coloring.
    """
    node_list = list(G.nodes())
    node_to_spin = {i: SpinNode() for i in node_list}
    flat_nodes = [node_to_spin[i] for i in node_list]
    
    # 1. Chromatic Graph Coloring for Bipartite/Multipartite Independent Blocks
    coloring = nx.coloring.greedy_color(G, strategy="largest_first")
    num_colors = max(coloring.values()) + 1
    
    free_blocks = []
    for c in range(num_colors):
        color_spins = [node_to_spin[i] for i in node_list if coloring[i] == c]
        if color_spins:
            free_blocks.append(Block(color_spins))
            
    # 2. Build Antiferromagnetic Edges (J = -1.0)
    edges = []
    for u, v in G.edges():
        edges.append((node_to_spin[u], node_to_spin[v]))
        
    num_nodes = len(node_list)
    num_edges = len(edges)
    
    biases = jnp.zeros((num_nodes,), dtype=jnp.float32)
    # Antiferromagnetic coupling penalizes like-spins, rewarding cut edges
    weights = -1.0 * jnp.ones((num_edges,), dtype=jnp.float32)
    beta = jnp.array(beta_val, dtype=jnp.float32)
    
    ebm = IsingEBM(nodes=flat_nodes, edges=edges, biases=biases, weights=weights, beta=beta)
    # thrml v0.1.4 accepts positional: (ebm, free_blocks, clamped_blocks)
    program = IsingSamplingProgram(ebm, free_blocks, clamped_blocks=[])
    
    return program, ebm, free_blocks, flat_nodes, node_list, num_colors

def compute_cut_size(G: nx.Graph, spins_pm1: np.ndarray, node_list: list) -> int:
    """Calculates number of cut edges for a given {-1, +1} configuration."""
    spin_map = {node_list[i]: spins_pm1[i] for i in range(len(node_list))}
    cut = 0
    for u, v in G.edges():
        if spin_map[u] != spin_map[v]:
            cut += 1
    return cut

def main():
    n_nodes = 50
    edge_prob = 0.20
    seed = 1337
    
    print(f"[*] Generating G(n={n_nodes}, p={edge_prob}) test graph...")
    G = generate_problem_graph(n_nodes, edge_prob, seed=seed)
    total_edges = G.number_of_edges()
    print(f"[*] Total Edges in Problem: {total_edges}")
    
    # Random partition baseline (expectation is 50% of edges)
    rng = np.random.default_rng(seed)
    random_cuts = [compute_cut_size(G, rng.choice([-1, 1], size=n_nodes), list(G.nodes())) for _ in range(500)]
    avg_random_cut = np.mean(random_cuts)
    max_random_cut = np.max(random_cuts)
    print(f"[*] Random Baseline -> Mean Cut: {avg_random_cut:.1f} ({avg_random_cut/total_edges*100:.1f}%), Best: {max_random_cut}")
    
    # Sweep inverse temperature beta to demonstrate thermal annealing ground-state convergence
    betas = [0.2, 0.5, 1.0, 2.0, 3.5]
    best_overall_cut = 0
    best_spins = None
    cut_distributions = []
    
    key = jax.random.key(101)
    
    print("\n--- Running THRML Thermodynamic Annealing & Sampling ---")
    start_time = time.perf_counter()
    
    for b in betas:
        key, subkey = jax.random.split(key)
        program, ebm, free_blocks, flat_nodes, node_list, k_colors = build_max_cut_ebm(G, beta_val=b)
        
        k_init, k_samp = jax.random.split(subkey, 2)
        init_state = hinton_init(k_init, ebm, free_blocks, ())
        schedule = SamplingSchedule(n_warmup=150, n_samples=600, steps_per_sample=k_colors)
        
        raw_samples = sample_states(k_samp, program, schedule, init_state, [], [Block(flat_nodes)])
        # Convert boolean samples (False=-1, True=+1)
        spins_arr = 2 * np.array(raw_samples[0], dtype=np.int32) - 1  # shape: (n_samples, n_nodes)
        
        cuts = [compute_cut_size(G, spins_arr[i], node_list) for i in range(len(spins_arr))]
        cut_distributions.append(cuts)
        mean_cut = np.mean(cuts)
        max_cut = np.max(cuts)
        
        if max_cut > best_overall_cut:
            best_overall_cut = max_cut
            best_spins = spins_arr[np.argmax(cuts)]
            
        print(f"beta = {b:.1f} (Colors={k_colors}) | Mean Cut: {mean_cut:.1f} | Best Cut Found: {max_cut} ({max_cut/total_edges*100:.1f}%)")
        
    elapsed = time.perf_counter() - start_time
    print(f"\n[+] Total Optimization Time: {elapsed:.3f}s")
    print(f"[+] Global Optimal Cut Found: {best_overall_cut} / {total_edges} edges ({best_overall_cut/total_edges*100:.1f}%)")
    
    # Visualization: Graph Cut Layout + Convergence Curve
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Cut Ratio Distributions vs Beta
    box_data = [[c / total_edges * 100 for c in dist] for dist in cut_distributions]
    axes[0].boxplot(box_data, tick_labels=[str(b) for b in betas], patch_artist=True,
                    boxprops=dict(facecolor="#93c5fd", color="#1e40af"),
                    medianprops=dict(color="#b91c1c", linewidth=2))
    axes[0].axhline(avg_random_cut / total_edges * 100, color="gray", linestyle="--", label=f"Random Chance ({avg_random_cut/total_edges*100:.1f}%)")
    axes[0].set_title("Max-Cut Ratio vs Inverse Temperature (β)")
    axes[0].set_xlabel("Annealing Beta (β)")
    axes[0].set_ylabel("Cut Ratio (%)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot 2: Partitioned Graph
    pos = nx.spring_layout(G, seed=42)
    node_colors = ["#ef4444" if best_spins[i] == 1 else "#3b82f6" for i in range(n_nodes)]
    
    # Draw cut edges vs internal edges
    cut_edges = [(u, v) for u, v in G.edges() if best_spins[node_list.index(u)] != best_spins[node_list.index(v)]]
    internal_edges = [(u, v) for u, v in G.edges() if best_spins[node_list.index(u)] == best_spins[node_list.index(v)]]
    
    nx.draw_networkx_nodes(G, pos, ax=axes[1], node_color=node_colors, node_size=150)
    nx.draw_networkx_edges(G, pos, edgelist=internal_edges, ax=axes[1], edge_color="#cbd5e1", width=1.0, alpha=0.5)
    nx.draw_networkx_edges(G, pos, edgelist=cut_edges, ax=axes[1], edge_color="#10b981", width=1.8, label="Cut Edge")
    axes[1].set_title(f"Best Partition Found (Cut: {best_overall_cut}/{total_edges})")
    axes[1].axis("off")
    
    plt.tight_layout()
    plt.savefig("thrml_max_cut_solution.png", dpi=200)
    print("[+] Saved visualization artifact: thrml_max_cut_solution.png")

if __name__ == "__main__":
    main()
