"""Factor graph topologies and graph-coloring block partitioners for parallel Gibbs sampling.

In a thermodynamic sampler (`thrml`), nodes inside the same `Block` are updated simultaneously
in parallel. For this parallel update to preserve detailed balance with respect to the joint
Boltzmann distribution P(s) = exp(-beta E(s)) / Z, every `Block` must form an independent set
in the factor graph (i.e. no two nodes in the same Block share a pairwise factor edge).
"""

from __future__ import annotations

from typing import Any, Sequence
from thermo_repro.backend import Block, SpinNode


def verify_independent_blocks(
    blocks: Sequence[Block],
    edges: Sequence[tuple[SpinNode, SpinNode]],
) -> bool:
    """Verify that every Block is a valid independent set with respect to `edges`.

    Raises ValueError if any edge connects two nodes within the same Block.
    """
    node_to_block: dict[SpinNode, int] = {}
    for b_idx, blk in enumerate(blocks):
        for node in blk:
            node_to_block[node] = b_idx

    for u, v in edges:
        if u in node_to_block and v in node_to_block:
            if node_to_block[u] == node_to_block[v]:
                raise ValueError(
                    f"Invalid Block coloring: connected nodes {u} and {v} share Block {node_to_block[u]}."
                )
    return True


def make_single_site_blocks(nodes: Sequence[SpinNode]) -> list[Block]:
    """Create N single-node blocks for sequential single-site Gibbs sampling."""
    return [Block([n]) for n in nodes]


def greedy_graph_coloring_blocks(
    nodes: Sequence[SpinNode],
    edges: Sequence[tuple[SpinNode, SpinNode]],
) -> list[Block]:
    """Partition an arbitrary graph into minimal independent-set Blocks using Welsh-Powell greedy coloring."""
    adj: dict[SpinNode, set[SpinNode]] = {n: set() for n in nodes}
    for u, v in edges:
        if u in adj and v in adj:
            adj[u].add(v)
            adj[v].add(u)

    # Sort nodes by descending degree for compact coloring
    sorted_nodes = sorted(nodes, key=lambda n: len(adj[n]), reverse=True)
    node_color: dict[SpinNode, int] = {}
    color_groups: dict[int, list[SpinNode]] = {}

    for node in sorted_nodes:
        neighbor_colors = {
            node_color[nbr] for nbr in adj[node] if nbr in node_color
        }
        color = 0
        while color in neighbor_colors:
            color += 1
        node_color[node] = color
        color_groups.setdefault(color, []).append(node)

    blocks = [Block(color_groups[c]) for c in sorted(color_groups.keys())]
    verify_independent_blocks(blocks, edges)
    return blocks


def build_2d_lattice_topology(
    height: int,
    width: int,
    periodic: bool = False,
) -> tuple[list[SpinNode], list[tuple[SpinNode, SpinNode]], list[Block]]:
    """Construct a 2D Ising lattice of size (height x width) with 2-color checkerboard Blocks.

    Because a 2D nearest-neighbor grid is bipartite, the entire H*W spin array updates in
    just 2 parallel hardware clock steps (even parity `(r + c) % 2 == 0` and odd parity `(r + c) % 2 == 1`).
    """
    grid = [[SpinNode(f"s_{r}_{c}") for c in range(width)] for r in range(height)]
    nodes = [grid[r][c] for r in range(height) for c in range(width)]
    edges: list[tuple[SpinNode, SpinNode]] = []

    even_nodes: list[SpinNode] = []
    odd_nodes: list[SpinNode] = []

    for r in range(height):
        for c in range(width):
            node = grid[r][c]
            if (r + c) % 2 == 0:
                even_nodes.append(node)
            else:
                odd_nodes.append(node)

            # Right neighbor
            if c + 1 < width:
                edges.append((node, grid[r][c + 1]))
            elif periodic and width > 2:
                edges.append((node, grid[r][0]))

            # Down neighbor
            if r + 1 < height:
                edges.append((node, grid[r + 1][c]))
            elif periodic and height > 2:
                edges.append((node, grid[0][c]))

    checkerboard_blocks = [Block(even_nodes), Block(odd_nodes)]
    verify_independent_blocks(checkerboard_blocks, edges)
    return nodes, edges, checkerboard_blocks


def build_vhl_bipartite_topology(
    n_visible: int,
    n_hidden: int,
    n_labels: int,
) -> dict[str, Any]:
    """Construct the Project A Visible-Hidden-Label (V-H-L) bipartite factor graph.

    Energy function:
        E(v, h, l) = - sum_i a_i v_i - sum_j b_j h_j - sum_k c_k l_k
                     - sum_{i,j} W_{ij} v_i h_j - sum_{j,k} U_{jk} h_j l_k

    Note that Visible nodes `V` and Label nodes `L` only connect to Hidden nodes `H`
    and never directly to each other! Thus `(V union L)` and `H` form a 2-color bipartite
    partition:
        - Free inference (clamped V): `[Block(H), Block(L)]`
        - Free generative sampling (unclamped): `[Block(V + L), Block(H)]`
    """
    v_nodes = [SpinNode(f"v_{i}") for i in range(n_visible)]
    h_nodes = [SpinNode(f"h_{j}") for j in range(n_hidden)]
    l_nodes = [SpinNode(f"l_{k}") for k in range(n_labels)]
    all_nodes = v_nodes + h_nodes + l_nodes

    vh_edges: list[tuple[SpinNode, SpinNode]] = []
    for v in v_nodes:
        for h in h_nodes:
            vh_edges.append((v, h))

    hl_edges: list[tuple[SpinNode, SpinNode]] = []
    for h in h_nodes:
        for l in l_nodes:
            hl_edges.append((h, l))

    all_edges = vh_edges + hl_edges

    free_all_blocks = [Block(v_nodes + l_nodes), Block(h_nodes)]
    verify_independent_blocks(free_all_blocks, all_edges)

    return {
        "v_nodes": v_nodes,
        "h_nodes": h_nodes,
        "l_nodes": l_nodes,
        "all_nodes": all_nodes,
        "vh_edges": vh_edges,
        "hl_edges": hl_edges,
        "all_edges": all_edges,
        "free_all_blocks": free_all_blocks,
        "clamped_v_free_blocks": [Block(h_nodes), Block(l_nodes)],
    }
