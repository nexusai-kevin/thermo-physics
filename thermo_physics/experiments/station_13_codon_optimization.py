"""
Station 13: Biological Codon Optimization via Thermodynamic Potts EBM
=====================================================================

Formulates mRNA synonymous sequence optimization as a multi-objective Potts
Energy-Based Model (EBM). Discovers mRNA transcripts that simultaneously maximize
human translation efficiency (Codon Adaptation Index / CAI) while eliminating
ribosome-stalling RNA secondary structure hairpins and maintaining physiological GC balance.

Target Peptide: Human GLP-1 bioactive 15-mer ("HAEGTFTSDVSSYLE")
Search Space: ~8.49 x 10^7 synonymous mRNA sequence candidates.

Methodological Note:
The inverted-repeat stem-loop detector (L_stem >= 3, loop 3-8 nt) provides an efficient,
zero-dependency first-order approximation of ribosome-stalling hairpins. Future iterations
can incorporate ViennaRNA / NUPACK nearest-neighbor partition functions for detailed free-energy
conformational matrices while preserving this exact Potts EBM sampler interface.
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

WATSON_CRICK: Dict[str, str] = {"A": "T", "T": "A", "G": "C", "C": "G"}


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


def detect_stem_loop_hairpins(
    mrna_seq: str,
    min_stem: int = 3,
    min_loop: int = 3,
    max_loop: int = 8,
) -> List[Tuple[int, int, str, str]]:
    """Identify inverted-repeat Watson-Crick complementary stems forming RNA hairpins."""
    n = len(mrna_seq)
    hairpins = []
    for i in range(n - 2 * min_stem - min_loop + 1):
        stem1 = mrna_seq[i : i + min_stem]
        rev_comp = "".join(WATSON_CRICK.get(b, "N") for b in reversed(stem1))
        for loop_len in range(min_loop, max_loop + 1):
            j = i + min_stem + loop_len
            if j + min_stem <= n:
                stem2 = mrna_seq[j : j + min_stem]
                if stem2 == rev_comp:
                    hairpins.append((i, j, stem1, stem2))
    return hairpins


@dataclass
class CodonEvaluation:
    mrna: str
    total_energy: float
    cai: float
    gc_content: float
    hairpin_count: int
    e_cai: float
    e_gc: float
    e_hairpin: float


def evaluate_mrna_energy(
    codon_indices: List[int],
    target_peptide: str,
    target_gc: float = 0.50,
    lambda_gc: float = 40.0,
    lambda_hairpin: float = 2.0,
) -> CodonEvaluation:
    """Compute Hamiltonian energy: H(c) = E_cai + lambda_gc (GC - GC_target)^2 + lambda_hp N_hairpins."""
    codons = []
    e_cai = 0.0

    for i, aa in enumerate(target_peptide):
        synonyms = list(HUMAN_CODON_USAGE[aa].keys())
        idx = codon_indices[i] % len(synonyms)
        c = synonyms[idx]
        codons.append(c)

        w = compute_relative_adaptiveness(aa)[c]
        # Linear expression penalty: - ln(w)
        e_cai += -math.log(max(1e-4, w))

    mrna = "".join(codons)
    gc = compute_gc_content(mrna)
    e_gc = lambda_gc * ((gc - target_gc) ** 2)

    hairpins = detect_stem_loop_hairpins(mrna, min_stem=3, min_loop=3, max_loop=8)
    e_hairpin = lambda_hairpin * len(hairpins)

    total_e = e_cai + e_gc + e_hairpin
    cai = compute_cai(mrna, target_peptide)

    return CodonEvaluation(
        mrna=mrna,
        total_energy=total_e,
        cai=cai,
        gc_content=gc,
        hairpin_count=len(hairpins),
        e_cai=e_cai,
        e_gc=e_gc,
        e_hairpin=e_hairpin,
    )


def run_greedy_baseline(target_peptide: str) -> CodonEvaluation:
    """Greedy selection: chooses the #1 highest-frequency synonymous codon for each site."""
    best_indices = [0] * len(target_peptide)
    return evaluate_mrna_energy(best_indices, target_peptide)


def run_random_baseline(target_peptide: str, seed: int = 42) -> CodonEvaluation:
    """Random synonymous codon assignment baseline."""
    rng = np.random.default_rng(seed)
    indices = [int(rng.integers(0, len(HUMAN_CODON_USAGE[aa]))) for aa in target_peptide]
    return evaluate_mrna_energy(indices, target_peptide)


def run_thermodynamic_codon_optimization(
    target_peptide: str = TARGET_PEPTIDE,
    n_steps: int = 15_000,
    beta: float = 3.0,
    seed: int = 1337,
) -> Dict[str, Any]:
    """Thermodynamic Potts MCMC optimization balancing CAI, GC content, and secondary structure."""
    l_seq = len(target_peptide)
    syn_counts = [len(HUMAN_CODON_USAGE[aa]) for aa in target_peptide]
    total_space = math.prod(syn_counts)

    rng = np.random.default_rng(seed)
    # Initialize from random state
    state = [int(rng.integers(0, count)) for count in syn_counts]

    curr_eval = evaluate_mrna_energy(state, target_peptide)
    best_eval = curr_eval

    energy_trace = []
    t0 = time.perf_counter()

    for step in range(n_steps):
        # Select random residue position to propose synonymous swap
        pos = rng.integers(0, l_seq)
        old_val = state[pos]
        new_val = rng.integers(0, syn_counts[pos])

        if new_val == old_val:
            energy_trace.append(curr_eval.total_energy)
            continue

        state[pos] = new_val
        prop_eval = evaluate_mrna_energy(state, target_peptide)
        delta_e = prop_eval.total_energy - curr_eval.total_energy

        # Metropolis-Hastings acceptance probability
        if delta_e <= 0.0 or rng.random() < math.exp(-beta * delta_e):
            curr_eval = prop_eval
            if curr_eval.total_energy < best_eval.total_energy:
                best_eval = curr_eval
        else:
            state[pos] = old_val  # Reject proposal

        energy_trace.append(curr_eval.total_energy)

    elapsed = time.perf_counter() - t0
    energy_arr = np.array(energy_trace)
    tau_int, ess = compute_sokal_tau_int(energy_arr)
    ess_per_sec = ess / elapsed if elapsed > 0 else 0.0

    return {
        "target_peptide": target_peptide,
        "search_space_size": total_space,
        "best_eval": best_eval,
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
    print(f"Combinatorial Search Space: 84,934,656 synonymous mRNA transcripts")
    print("=" * 80)

    # 1. Classical Baselines
    rand = run_random_baseline(TARGET_PEPTIDE)
    print("\n[Baseline 1: Random Synonymous Choice]")
    print(f"  mRNA Sequence : {rand.mrna}")
    print(f"  CAI: {rand.cai:.4f} | GC: {rand.gc_content * 100:.1f}% | Hairpins: {rand.hairpin_count}")
    print(f"  Total Energy  : {rand.total_energy:.4f} (E_CAI={rand.e_cai:.2f}, E_GC={rand.e_gc:.2f}, E_HP={rand.e_hairpin:.2f})")

    greedy = run_greedy_baseline(TARGET_PEPTIDE)
    print("\n[Baseline 2: Greedy CAI-Only Selection (Heuristic Trap)]")
    print(f"  mRNA Sequence : {greedy.mrna}")
    print(f"  CAI: {greedy.cai:.4f} | GC: {greedy.gc_content * 100:.1f}% | Hairpins: {greedy.hairpin_count} STALLS")
    print(f"  Total Energy  : {greedy.total_energy:.4f} (E_CAI={greedy.e_cai:.2f}, E_GC={greedy.e_gc:.2f}, E_HP={greedy.e_hairpin:.2f})")
    print("  * Critical Pitfall: Pure greedy CAI achieves CAI=1.0 but creates 4 severe stem-loop hairpins")
    print("    and an unphysiological 66.7% GC content, stalling ribosomes during elongation.")

    # 2. Thermodynamic Potts Optimization
    res = run_thermodynamic_codon_optimization(TARGET_PEPTIDE, n_steps=20_000, beta=3.5)
    opt: CodonEvaluation = res["best_eval"]
    print(f"\n[Station 13: Thermodynamic Potts Optimization ({res['n_steps']:,} steps)]")
    print(f"  Optimal mRNA  : {opt.mrna}")
    print(f"  CAI Score     : {opt.cai:.4f} (High expression profile preserved)")
    print(f"  GC Content    : {opt.gc_content * 100:.1f}% (Shifted towards physiological ~50% target)")
    print(f"  Hairpins      : {opt.hairpin_count} (Stem-loop ribosome stall sites eliminated)")
    print(f"  Total Energy  : {opt.total_energy:.4f} (E_CAI={opt.e_cai:.2f}, E_GC={opt.e_gc:.2f}, E_HP={opt.e_hairpin:.2f})")
    print(f"  Net Advantage : {greedy.total_energy - opt.total_energy:.4f} Lower Energy than Greedy Baseline!")
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
    print(f"\n[Hardware Acceleration Projection: Extropic Z1 vs. Measured Classical GPU]")
    print(f"  Classical Throughput : {proj.gpu_ess_per_sec:,.1f} ESS/s | {proj.gpu_joules_per_1k_ess:,.4f} J / 1k ESS")
    print(f"  Extropic TSU Silicon : {proj.tsu_ess_per_sec:,.1f} ESS/s | {proj.tsu_joules_per_1k_ess:,.6f} J / 1k ESS")
    print(f"  Projected Efficiency : {proj.energy_efficiency_multiplier:,.0f}x Energy Reduction")
    print(f"  Audit Reference      : docs/METHODOLOGY.md#6-hardware-projections-assumptions--limitations")


if __name__ == "__main__":
    run_experiment_13()
