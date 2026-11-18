"""
Phase 4 (PRODUCTIZE): Thermodynamic sampling, Pydantic validation, and multi-target export.

Runs real JAX/THRML Gibbs & annealing samplers + `compute_sokal_tau_int` across:
  1) Ecological inverse-Ising lattice (N=16) via `JaxBlockGibbsSampler`
  2) RNA co-evolution Potts tensor (L=24, q=5) via JAX Categorical Gibbs scan
  3) Market + Polymarket QUBO/Ising portfolio (N=16, K=5) via THRML `IsingSamplingProgram`

Validates the full payload through `LiveEmpiricalStreamsManifest` and writes
`live_empirical_streams.json` atomically to all resolved project targets.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
from thrml import Block, SamplingSchedule, SpinNode, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram

from thermo_physics.backends.sampler_adapter import (
    BackendType,
    HamiltonianGraph,
    SamplingConfig,
    compute_sokal_tau_int,
    get_sampler,
)
from thermo_physics.refinery.schemas import LiveEmpiricalStreamsManifest

PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent

_EXTERNAL_WEB_ROOTS: List[Path] = [
    Path(r"C:\Users\mcder\stellar_logos_seed_swarm\demos\nexus-site"),
    Path(r"C:\Users\mcder\Desktop\nexus-site-temp"),
]

_EXTERNAL_ECOSYSTEM_TARGETS: List[tuple[Path, Path]] = [
    (Path(r"C:\Users\mcder\AffilAI"), Path(r"C:\Users\mcder\AffilAI\public\data\live_empirical_streams.json")),
    (Path(r"C:\Users\mcder\poseidon_echo_dojo"), Path(r"C:\Users\mcder\poseidon_echo_dojo\data\live_empirical_streams.json")),
    (Path(r"C:\Users\mcder\freshfish-collectr"), Path(r"C:\Users\mcder\freshfish-collectr\public\data\live_empirical_streams.json")),
]


def resolve_live_stream_targets() -> List[Path]:
    """Return active live_empirical_streams.json targets (always repo data/, plus sibling Windows repos)."""
    targets: List[Path] = [PACKAGE_ROOT / "data" / "live_empirical_streams.json"]
    if sys.platform == "win32":
        for root in _EXTERNAL_WEB_ROOTS:
            if root.exists():
                targets.append(root / "public" / "data" / "live_empirical_streams.json")
                targets.append(root / "src" / "data" / "live_empirical_streams.json")
        for root, target_path in _EXTERNAL_ECOSYSTEM_TARGETS:
            if root.exists():
                targets.append(target_path)
    return targets


LIVE_STREAM_TARGETS: List[Path] = resolve_live_stream_targets()

DOCTRINE_DISCLOSURE = (
    "Empirical Telemetry & Modeling Scope Disclosure: External public telemetry streams "
    "(USGS NWIS water sensors, Rfam RF00005/RF00001 RNA alignments, Yahoo Finance 1h bars, "
    "Polymarket CLOB order books, and Shopify DTC product feeds) are ingested via fault-tolerant "
    "polling and transparently tagged per stream with source_mode in {'live_public_stream', "
    "'cached_empirical', 'seeded_fallback'}. Ecological Ising couplings reflect Tikhonov-regularized "
    "empirical spin inverse covariances rather than causal physical laws; RNA Potts tensors capture "
    "consensus statistical co-evolution rather than full Turner 3D tertiary folding; Market QUBO "
    "matrices use trailing 5-day hourly volatility and prediction-market OBI as a computational "
    "optimization benchmark (not investment advice); AffilAI scores derive from public storefront "
    "catalog heuristics."
)


def _round_floats(obj: Any, ndigits: int = 4) -> Any:
    """Recursively round floating-point scalars and arrays for compact ~150-250KB JSON export."""
    if isinstance(obj, float):
        if abs(obj) < 1e-12:
            return 0.0
        return round(obj, ndigits)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return 0.0 if abs(val) < 1e-12 else round(val, ndigits)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _round_floats(obj.tolist(), ndigits=ndigits)
    if isinstance(obj, list):
        return [_round_floats(item, ndigits=ndigits) for item in obj]
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits=ndigits) for k, v in obj.items()}
    return obj


# ---------------------------------------------------------------------------
# Thermodynamic Sampler 1: Ecological Ising (N=16) via JaxBlockGibbsSampler
# ---------------------------------------------------------------------------

def _benchmark_ecological_ising(agg_ising: Dict[str, Any], seed: int = 42) -> Dict[str, Any]:
    j_mat = np.asarray(agg_ising["J"], dtype=np.float64)
    h_vec = np.asarray(agg_ising["h"], dtype=np.float64)
    exact_e0 = float(agg_ising["exact_ground_state_energy"])

    graph = HamiltonianGraph(couplings=j_mat, biases=h_vec, spin_type="bipolar", offset=0.0)
    sampler = get_sampler(BackendType.JAX_BLOCK_GIBBS)
    cfg = SamplingConfig(n_samples=320, burn_in=120, stride=4, beta=1.5, seed=seed)
    res = sampler.sample(graph, cfg)

    min_e = float(np.min(res.energies))
    mean_e = float(np.mean(res.energies))
    abs_mag = float(np.mean(np.abs(np.mean(res.samples, axis=1))))

    return {
        "min_energy": min_e,
        "mean_energy": mean_e,
        "tau_int": float(res.tau_int),
        "ess": float(res.ess),
        "energy_trace": res.energies.astype(float).tolist(),
        "beta": cfg.beta,
        "n_samples": cfg.n_samples,
        "abs_magnetization": abs_mag,
        "exact_match": bool(abs(min_e - exact_e0) < 0.25),
    }


# ---------------------------------------------------------------------------
# Thermodynamic Sampler 2: RNA Potts (L=24, q=5) via JAX Categorical Gibbs
# ---------------------------------------------------------------------------

def _benchmark_rna_potts(agg_potts: Dict[str, Any], seed: int = 42) -> Dict[str, Any]:
    h_np = np.asarray(agg_potts["h_i"], dtype=np.float32)
    c_np = np.asarray(agg_potts["C_ij"], dtype=np.float32)
    l_len, q = h_np.shape

    # Zero self-interaction diagonal blocks for clean site-to-site Potts Hamiltonian
    c_clean = c_np.copy()
    for i in range(l_len):
        c_clean[i, i, :, :] = 0.0
    # Amplify statistical co-evolution tensor gently so Potts Gibbs exhibits clear energetic descent
    j_jax = jnp.asarray(c_clean * 4.0, dtype=jnp.float32)
    h_jax = jnp.asarray(h_np * 0.5, dtype=jnp.float32)

    beta = 1.25
    burn_in = 60
    n_samples = 320
    total_sweeps = burn_in + n_samples

    key = jax.random.PRNGKey(seed)
    key, k_init = jax.random.split(key)
    init_state = jax.random.randint(k_init, shape=(l_len,), minval=0, maxval=q, dtype=jnp.int32)

    def _sweep_site(site_idx: int, carry_inner: tuple) -> tuple:
        state, k_in = carry_inner
        k_in, k_cat = jax.random.split(k_in)
        # Compute field from all coupled neighbors j != site_idx: sum_j J[site_idx, j, :, state[j]]
        pair_contribs = jnp.sum(j_jax[site_idx, jnp.arange(l_len), :, state], axis=0)
        logits = beta * (h_jax[site_idx] + pair_contribs)
        new_val = jax.random.categorical(k_cat, logits).astype(jnp.int32)
        state = state.at[site_idx].set(new_val)
        return (state, k_in)

    def _gibbs_sweep(carry: tuple, _: Any) -> tuple:
        state, k_outer = carry
        state, k_outer = jax.lax.fori_loop(0, l_len, _sweep_site, (state, k_outer))
        # Exact Potts energy: - sum_i h_i[s_i] - 0.5 * sum_{i,j} J_{ij}[s_i, s_j]
        site_e = jnp.sum(h_jax[jnp.arange(l_len), state])
        pair_e = 0.5 * jnp.sum(j_jax[jnp.arange(l_len)[:, None], jnp.arange(l_len)[None, :], state[:, None], state[None, :]])
        energy = -(site_e + pair_e)
        return (state, k_outer), (state, energy)

    (_, _), (states_trace, energies_trace) = jax.lax.scan(
        _gibbs_sweep, (init_state, key), None, length=total_sweeps
    )
    states_trace.block_until_ready()

    valid_states = np.asarray(states_trace[burn_in:], dtype=np.int64)
    valid_energies = np.asarray(energies_trace[burn_in:], dtype=np.float64)

    tau_int, ess = compute_sokal_tau_int(valid_energies)

    # Marginal posterior Shannon entropy in bits across the L=24 sites
    one_hot = np.eye(q, dtype=np.float64)[valid_states]
    p_marg = np.mean(one_hot, axis=0)
    entropy_bits = float(-np.sum(p_marg * np.log2(np.maximum(p_marg, 1e-12))))

    return {
        "min_energy": float(np.min(valid_energies)),
        "mean_energy": float(np.mean(valid_energies)),
        "entropy_bits": entropy_bits,
        "tau_int": float(tau_int),
        "ess": float(ess),
        "energy_trace": valid_energies.tolist(),
        "beta": beta,
        "n_samples": n_samples,
    }


# ---------------------------------------------------------------------------
# Thermodynamic Sampler 3: Market QUBO / Ising via THRML IsingSamplingProgram
# ---------------------------------------------------------------------------

def _benchmark_market_qubo(agg_market: Dict[str, Any], seed: int = 42) -> Dict[str, Any]:
    q_mat = np.asarray(agg_market["Q"], dtype=np.float64)
    j_mat = np.asarray(agg_market["J"], dtype=np.float64)
    h_vec = np.asarray(agg_market["h"], dtype=np.float64)
    k_target = int(agg_market["provenance"]["k_target"])
    n_assets = int(agg_market["provenance"]["n_assets"])
    tickers = [a["ticker"] for a in agg_market["assets"]]
    exact_obj = float(agg_market["exact_optimum"]["qubo_energy"])
    penalty_lambda = float(agg_market.get("penalty_lambda", 0.65))

    edges_idx: List[tuple[int, int]] = []
    weights: List[float] = []
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            w = float(j_mat[i, j])
            if abs(w) > 1e-9:
                edges_idx.append((i, j))
                weights.append(w)

    g_nx = nx.Graph()
    g_nx.add_nodes_from(range(n_assets))
    g_nx.add_edges_from(edges_idx)
    coloring = nx.coloring.greedy_color(g_nx, strategy="largest_first")
    num_colors = max(coloring.values()) + 1

    betas = [0.6, 1.2, 2.5, 5.0, 9.0]
    key = jax.random.key(seed)
    rng_init = np.random.default_rng(seed)
    current_bits = np.zeros(n_assets, dtype=bool)
    current_bits[rng_init.choice(n_assets, size=k_target, replace=False)] = True

    energy_trace: List[float] = []
    final_stage_objs: List[float] = []
    final_stage_feas: List[bool] = []
    best_feasible_x: Optional[np.ndarray] = None
    best_feasible_obj = float("inf")
    best_any_x: Optional[np.ndarray] = None
    best_any_obj = float("inf")

    for beta_val in betas:
        key, k_samp = jax.random.split(key)
        spin_nodes = [SpinNode() for _ in range(n_assets)]
        free_blocks = [
            Block([spin_nodes[i] for i in range(n_assets) if coloring[i] == c])
            for c in range(num_colors)
            if any(coloring[i] == c for i in range(n_assets))
        ]
        node_to_idx = {spin_nodes[i]: i for i in range(n_assets)}
        edges = [(spin_nodes[u], spin_nodes[v]) for u, v in edges_idx]
        ebm = IsingEBM(
            nodes=spin_nodes,
            edges=edges,
            biases=jnp.array(h_vec, dtype=jnp.float32),
            weights=jnp.array(weights, dtype=jnp.float32),
            beta=jnp.array(beta_val, dtype=jnp.float32),
        )
        program = IsingSamplingProgram(ebm, free_blocks=free_blocks, clamped_blocks=[])
        init_state = [
            jnp.array([bool(current_bits[node_to_idx[node]]) for node in blk], dtype=jnp.bool_)
            for blk in free_blocks
        ]
        schedule = SamplingSchedule(n_warmup=30, n_samples=64, steps_per_sample=1)
        raw_samples = sample_states(k_samp, program, schedule, init_state, [], [Block(spin_nodes)])
        x_samples = np.asarray(raw_samples[0], dtype=np.float64)

        stage_objs = np.sum((x_samples @ q_mat) * x_samples, axis=1) + penalty_lambda * float(k_target ** 2)
        card_counts = np.sum(x_samples, axis=1).astype(int)

        for idx_s in range(len(x_samples)):
            obj_val = float(stage_objs[idx_s])
            energy_trace.append(obj_val)
            is_feas = bool(card_counts[idx_s] == k_target)
            if beta_val == betas[-1]:
                final_stage_objs.append(obj_val)
                final_stage_feas.append(is_feas)
            if obj_val < best_any_obj:
                best_any_obj = obj_val
                best_any_x = x_samples[idx_s].copy()
            if is_feas and obj_val < best_feasible_obj:
                best_feasible_obj = obj_val
                best_feasible_x = x_samples[idx_s].copy()

        current_bits = (best_feasible_x if best_feasible_x is not None else best_any_x).astype(bool)

    chosen_x = best_feasible_x if best_feasible_x is not None else best_any_x
    chosen_obj = best_feasible_obj if best_feasible_x is not None else best_any_obj
    assert chosen_x is not None

    sel_indices = np.where(chosen_x > 0.5)[0].astype(int).tolist()
    sel_tickers = [tickers[i] for i in sel_indices]
    feasibility_pct = 100.0 * float(np.mean(final_stage_feas)) if final_stage_feas else 0.0

    if exact_obj < 0.0 and chosen_obj < 0.0:
        pareto_match = float(np.clip((chosen_obj / exact_obj) * 100.0, 0.0, 100.0))
    elif abs(exact_obj) < 1e-9:
        pareto_match = 100.0 if abs(chosen_obj - exact_obj) < 1e-4 else 0.0
    else:
        pareto_match = float(np.clip((exact_obj / max(chosen_obj, 1e-9)) * 100.0, 0.0, 100.0))

    tau_int, ess = compute_sokal_tau_int(np.asarray(energy_trace, dtype=np.float64))

    return {
        "selected_indices": sel_indices,
        "selected_tickers": sel_tickers,
        "qubo_energy": float(chosen_obj),
        "budget_feasibility_pct": feasibility_pct,
        "pareto_match_pct": pareto_match,
        "tau_int": float(tau_int),
        "ess": float(ess),
        "energy_trace": energy_trace,
    }


# ---------------------------------------------------------------------------
# Phase 4 Productization & Multi-Target Atomic Export
# ---------------------------------------------------------------------------

def productize_empirical_streams(
    aggregated_streams: Dict[str, Any],
    *,
    targets: Optional[Sequence[Path]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run thermodynamic samplers, validate Pydantic manifest, and write live_empirical_streams.json to all 5 targets."""
    eco = dict(aggregated_streams["ecological_ising"])
    eco["sampler_benchmark"] = _benchmark_ecological_ising(eco, seed=seed)

    rna = dict(aggregated_streams["rna_potts"])
    rna["sampler_benchmark"] = _benchmark_rna_potts(rna, seed=seed)

    mkt = dict(aggregated_streams["market_qubo"])
    mkt["sampler_benchmark"] = _benchmark_market_qubo(mkt, seed=seed)

    affil = dict(aggregated_streams["affilai_product_graph"])

    raw_payload: Dict[str, Any] = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "doctrine_disclosure": DOCTRINE_DISCLOSURE,
        "ecological_ising": eco,
        "rna_potts": rna,
        "market_qubo": mkt,
        "affilai_product_graph": affil,
    }

    rounded_payload = _round_floats(raw_payload, ndigits=5)

    # Preserve exact row-sum=1.0 for RNA marginals after decimal rounding
    for row in rounded_payload["rna_potts"]["f_i"]:
        max_idx = max(range(len(row)), key=lambda idx: row[idx])
        diff = 1.0 - sum(row)
        row[max_idx] = round(row[max_idx] + diff, 6)

    # Preserve exact QUBO <-> Ising equivalence on exported JSON Q, J, h
    q_round = np.asarray(rounded_payload["market_qubo"]["Q"], dtype=np.float64)
    q_round = 0.5 * (q_round + q_round.T)
    j_round = -0.5 * q_round
    np.fill_diagonal(j_round, 0.0)
    h_round = np.array(
        [-0.5 * q_round[i, i] - 0.5 * (np.sum(q_round[i, :]) - q_round[i, i]) for i in range(q_round.shape[0])],
        dtype=np.float64,
    )
    rounded_payload["market_qubo"]["Q"] = np.round(q_round, 6).tolist()
    rounded_payload["market_qubo"]["J"] = np.round(j_round, 6).tolist()
    rounded_payload["market_qubo"]["h"] = np.round(h_round, 6).tolist()

    validated = LiveEmpiricalStreamsManifest.model_validate(rounded_payload)
    manifest_dict = validated.model_dump(mode="json")

    serialized = json.dumps(manifest_dict, separators=(",", ":"), ensure_ascii=False)
    out_targets = list(targets) if targets is not None else resolve_live_stream_targets()

    for target_path in out_targets:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target_path.with_suffix(".json.tmp")
        tmp_path.write_text(serialized, encoding="utf-8")
        tmp_path.replace(target_path)

    return {
        "manifest": manifest_dict,
        "byte_size": len(serialized.encode("utf-8")),
        "targets_written": [str(p) for p in out_targets],
    }
