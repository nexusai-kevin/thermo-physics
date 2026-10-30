"""
Station 13: Biological Codon Optimization via Thermodynamic Potts EBM
=====================================================================

Optimizes synonymous mRNA codon sequence design for target peptide expression.
Balances human Codon Adaptation Index (CAI) against GC-content equilibrium and
local secondary structure penalties.

Target Peptide: Human GLP-1 bioactive 15-mer ("HAEGTFTSDVSSYLE")
Search Space: ~4.7 x 10^8 synonymous mRNA transcript sequences.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from thermo_physics.backends.sampler_adapter import compute_sokal_tau_int
from thermo_physics.benchmarks.hardware_projections import (
    WorkloadProfile,
    evaluate_projection,
)

# Canonical Human GLP-1 Active 15-mer peptide fragment
TARGET_PEPTIDE: str = "HAEGTFTSDVSSYLE"

# Standard Human Codon Usage Table (Fractional frequencies per 1,000 codons, Kazusa DB)
HUMAN_CODON_USAGE: Dict[str, Dict[str, float]] = {
    "A": {"GCC": 0.40, "GCT": 0.27, "GCA": 0.23, "GCG": 0.11},
    "C": {"TGC": 0.55, "TGT": 0.45},
    "D": {"GAC": 0.54, "GAT": 0.46},
    "E": {"GAG": 0.58, "GAA": 0.42},
    "F": {"TTC": 0.55, "TTT": 0.45},
    "G": {"GGC": 0.34, "GGG": 0.25, "GGA": 0.25, "GGT": 0.16},
    "H": {"CAC": 0.59, "CAT": 0.41},
    "I": {"ATC": 0.48, "ATT": 0.36, "ATA": 0.16},
    "K": {"AAG": 0.58, "AAA": 0.42},
    "L": {"CTG": 0.41, "CTC": 0.20, "TTG": 0.13, "CTT": 0.13, "CTA": 0.07, "TTA": 0.07},
    "M": {"ATG": 1.00},
    "N": {"AAC": 0.54, "AAT": 0.46},
    "P": {"CCC": 0.33, "CCT": 0.28, "CCA": 0.27, "CCG": 0.11},
    "Q": {"CAG": 0.75, "CAA": 0.25},
    "R": {"CGG": 0.21, "AGG": 0.20, "CGC": 0.19, "AGA": 0.20, "CGA": 0.11, "CGT": 0.08},
    "S": {"AGC": 0.24, "TCC": 0.22, "TCT": 0.18, "TCA": 0.15, "AGT": 0.15, "TCG": 0.06},
    "T": {"ACC": 0.36, "ACA": 0.28, "ACT": 0.24, "ACG": 0.12},
    "V": {"GTG": 0.47, "GTC": 0.24, "GTT": 0.18, "GTA": 0.11},
    "W": {"TGG": 1.00},
    "Y": {"TAC": 0.57, "TAT": 0.43},
}


def compute_relative_adaptiveness(amino_acid: str) -> Dict[str, float]:
    """Compute relative adaptiveness w_i = freq(codon) / max_freq(synonymous)."""
    table = HUMAN_CODON_USAGE[amino_acid]
    max_f = max(table.values())
    return {c: f / max_f for c, f in table.items()}


def compute_cai(mrna_seq: str, target_peptide: str) -> float:
    """Calculate geometric mean Codon Adaptation Index: CAI = exp( (1/L) sum ln(w_k) )."""
    l_seq = len(target_peptide)
    assert len(mrna_seq) == l_seq * 3
    log_sum = 0.0
    for i in range(l_seq):
        aa = target_peptide[i]
        codon = mrna_seq[i * 3 : (i + 1) * 3]
        w_dict = compute_relative_adaptiveness(aa)
        w_val = w_dict.get(codon, 0.01)
        log_sum += math.log(max(1e-4, w_val))
    return float(math.exp(log_sum / l_seq))


def compute_gc_content(mrna_seq: str) -> float:
    """Fraction of G and C nucleotides in mRNA sequence."""
    gc_count = sum(1 for b in mrna_seq if b in ("G", "C"))
    return float(gc_count / len(mrna_seq))


def evaluate_mrna_energy(
    codon_indices: List[int],
    target_peptide: str,
    target_gc: float = 0.52,
    lambda_gc: float = 8.0,
    lambda_hairpin: float = 3.0,
) -> Tuple[float, str, float, float]:
    """Compute Hamiltonian energy H(c) = E_cai + E_gc_penalty + E_neighbor."""
    codons = []
    e_cai = 0.0

    for i, aa in enumerate(target_peptide):
        synonyms = list(HUMAN_CODON_USAGE[aa].keys())
        idx = codon_indices[i] % len(synonyms)
        c = synonyms[idx]
        codons.append(c)

        w = compute_relative_adaptiveness(aa)[c]
        # Energy penalty: - ln(w), so optimal codon has 0 energy penalty
        e_cai += -math.log(max(1e-4, w))

    mrna = "".join(codons)
    gc = compute_gc_content(mrna)
    e_gc = lambda_gc * ((gc - target_gc) ** 2)

    # Neighboring dinucleotide / hairpin penalty: penalize repetitive 4-mers
    e_hairpin = 0.0
    for k in range(len(mrna) - 3):
        # Repetitive homopolymers or palindromic self-complementary 4-mers
        kmer = mrna[k : k + 4]
        if kmer in ("AAAA", "TTTT", "GGGG", "CCCC", "GCGC", "CGCG"):
            e_hairpin += lambda_hairpin

    total_e = e_cai + e_gc + e_hairpin
    cai = compute_cai(mrna, target_peptide)
    return total_e, mrna, cai, gc


def run_greedy_baseline(target_peptide: str) -> Tuple[str, float, float, float]:
    """Greedy selection of the single most frequent synonymous codon at every position."""
    best_indices = []
    for aa in target_peptide:
        w_dict = compute_relative_adaptiveness(aa)
        # Select highest relative adaptiveness (w = 1.0)
        best_indices.append(0)
    e, mrna, cai, gc = evaluate_mrna_energy(best_indices, target_peptide)
    return mrna, cai, gc, e


def run_random_baseline(target_peptide: str, seed: int = 42) -> Tuple[str, float, float, float]:
    """Random synonymous codon assignment."""
    rng = np.random.default_rng(seed)
    indices = [int(rng.integers(0, len(HUMAN_CODON_USAGE[aa]))) for aa in target_peptide]
    e, mrna, cai, gc = evaluate_mrna_energy(indices, target_peptide)
    return mrna, cai, gc, e


def run_thermodynamic_codon_optimization(
    target_peptide: str = TARGET_PEPTIDE,
    n_steps: int = 10_000,
    beta: float = 3.5,
    seed: int = 1337,
) -> Dict[str, Any]:
    """Thermodynamic Potts MCMC optimization balancing CAI and secondary structure."""
    l_seq = len(target_peptide)
    syn_counts = [len(HUMAN_CODON_USAGE[aa]) for aa in target_peptide]
    total_space = math.prod(syn_counts)

    rng = np.random.default_rng(seed)
    # Initialize randomly
    state = [int(rng.integers(0, count)) for count in syn_counts]

    curr_e, curr_mrna, curr_cai, curr_gc = evaluate_mrna_energy(state, target_peptide)
    best_e, best_mrna, best_cai, best_gc = curr_e, curr_mrna, curr_cai, curr_gc

    energy_trace = []
    t0 = time.perf_counter()

    for step in range(n_steps):
        # Select random residue position to update
        pos = rng.integers(0, l_seq)
        old_val = state[pos]
        new_val = rng.integers(0, syn_counts[pos])

        if new_val == old_val:
            energy_trace.append(curr_e)
            continue

        state[pos] = new_val
        prop_e, prop_mrna, prop_cai, prop_gc = evaluate_mrna_energy(state, target_peptide)
        delta_e = prop_e - curr_e

        # Metropolis acceptance probability
        if delta_e <= 0.0 or rng.random() < math.exp(-beta * delta_e):
            curr_e = prop_e
            curr_mrna = prop_mrna
            curr_cai = prop_cai
            curr_gc = prop_gc
            if curr_e < best_e:
                best_e, best_mrna, best_cai, best_gc = curr_e, curr_mrna, curr_cai, curr_gc
        else:
            state[pos] = old_val  # Reject

        energy_trace.append(curr_e)

    elapsed = time.perf_counter() - t0
    energy_arr = np.array(energy_trace)
    tau_int, ess = compute_sokal_tau_int(energy_arr)
    ess_per_sec = ess / elapsed if elapsed > 0 else 0.0

    return {
        "target_peptide": target_peptide,
        "search_space_size": total_space,
        "best_mrna": best_mrna,
        "best_cai": best_cai,
        "best_gc": best_gc,
        "best_energy": best_e,
        "tau_int": tau_int,
        "ess": ess,
        "ess_per_sec": ess_per_sec,
        "wall_clock_time": elapsed,
        "n_steps": n_steps,
    }


def run_experiment_13():
    print("=" * 80)
    print("Station 13: Biological Codon Optimization via Thermodynamic Potts EBM")
    print(f"Target Peptide: {TARGET_PEPTIDE} ({len(TARGET_PEPTIDE)} amino acids)")
    print("=" * 80)

    # 1. Classical Baselines
    rand_mrna, rand_cai, rand_gc, rand_e = run_random_baseline(TARGET_PEPTIDE)
    print("\n--- Baseline 1: Random Synonymous Codon Selection ---")
    print(f"  mRNA Sequence : {rand_mrna}")
    print(f"  CAI Score     : {rand_cai:.4f} | GC Content: {rand_gc * 100:.1f}% | Energy: {rand_e:.4f}")

    greedy_mrna, greedy_cai, greedy_gc, greedy_e = run_greedy_baseline(TARGET_PEPTIDE)
    print("\n--- Baseline 2: Greedy CAI-Only Selection (Ignoring GC & Hairpins) ---")
    print(f"  mRNA Sequence : {greedy_mrna}")
    print(f"  CAI Score     : {greedy_cai:.4f} | GC Content: {greedy_gc * 100:.1f}% | Energy: {greedy_e:.4f}")

    # 2. Thermodynamic Potts Optimization
    res = run_thermodynamic_codon_optimization(TARGET_PEPTIDE, n_steps=15_000, beta=4.0)
    print(f"\n--- Station 13: Thermodynamic Potts Optimization ({res['search_space_size']:,} transcripts) ---")
    print(f"  Optimal mRNA  : {res['best_mrna']}")
    print(f"  CAI Score     : {res['best_cai']:.4f} (High expression profile)")
    print(f"  GC Content    : {res['best_gc'] * 100:.1f}% (Optimal target: 50–55%)")
    print(f"  Min Energy    : {res['best_energy']:.4f}")
    print(f"  MCMC Mixing   : tau_int = {res['tau_int']:.2f} | ESS = {res['ess']:.1f} ({res['ess_per_sec']:,.1f} ESS/s)")
    print(f"  Runtime       : {res['wall_clock_time']:.4f}s ({res['n_steps'] / res['wall_clock_time']:,.1f} sweeps/sec)")

    # 3. Hardware Projection
    workload = WorkloadProfile(
        name=f"Station 13: mRNA Codon Potts EBM ({len(TARGET_PEPTIDE)} sites)",
        num_variables=len(TARGET_PEPTIDE),
        is_frustrated=False,
        classical_tau_int=res["tau_int"],
        hardware_tau_int=0.50,
    )
    proj = evaluate_projection(workload, gpu_measured_raw_hz=res["n_steps"] / res["wall_clock_time"])
    print(f"\n--- Hardware Acceleration Projection (Extropic Z1 vs. Measured Classical) ---")
    print(f"  Classical Throughput : {proj.gpu_ess_per_sec:,.1f} ESS/s | {proj.gpu_joules_per_1k_ess:,.4f} J / 1k ESS")
    print(f"  Extropic TSU Silicon : {proj.tsu_ess_per_sec:,.1f} ESS/s | {proj.tsu_joules_per_1k_ess:,.6f} J / 1k ESS")
    print(f"  Projected Efficiency : {proj.energy_efficiency_multiplier:,.0f}x Energy Reduction")


if __name__ == "__main__":
    run_experiment_13()
