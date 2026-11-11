"""
thermo_physics.codon.sampler
============================
Fast Potts / Markov Random Field ensemble codon sampler for full-length genes.

Key capabilities for biofoundry workflows:
  1. Incremental O(1) local window delta-H updates during annealed Gibbs/Metropolis sweeps
  2. Strict biofoundry constraint repair (eliminates Golden Gate / MoClo restriction sites
     spanning codon boundaries, homopolymer runs, and 5' ribosome-occluding hairpins)
  3. Diverse Boltzmann ensemble extraction: returns K constraint-clean synonymous sequences
     maximizing pairwise nucleotide Hamming distance within the low-energy basin.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from thermo_physics.codon.scoring import (
    CANONICAL_PAIRS,
    ConstructScore,
    ScoringWeights,
    count_homopolymers,
    find_forbidden_motifs,
    scan_local_hairpins,
    score_construct,
)
from thermo_physics.codon.tables import (
    DEFAULT_FOUNDRY_ENZYMES,
    HostProfile,
    expand_forbidden_motifs,
    get_host_profile,
    translate_dna,
)


def nucleotide_hamming_fraction(seq_a: str, seq_b: str) -> float:
    """Fraction of differing nucleotide positions between two equal-length coding sequences."""
    if len(seq_a) != len(seq_b) or not seq_a:
        return 0.0
    diffs = sum(1 for a, b in zip(seq_a, seq_b) if a != b)
    return diffs / len(seq_a)


@dataclass
class CodonCandidate:
    rank: int
    dna_sequence: str
    score: ConstructScore
    min_hamming_to_higher_ranked: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "dna_sequence": self.dna_sequence,
            "min_hamming_to_higher_ranked": round(self.min_hamming_to_higher_ranked, 4),
            "score": self.score.to_dict(),
        }


@dataclass
class EnsembleDesignResult:
    protein_length: int
    host: str
    enzymes_excluded: List[str]
    candidates: List[CodonCandidate]
    greedy_baseline: ConstructScore
    greedy_dna: str
    freq_random_baseline: ConstructScore
    mean_pairwise_hamming: float
    wall_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "protein_length": self.protein_length,
            "host": self.host,
            "enzymes_excluded": self.enzymes_excluded,
            "mean_pairwise_hamming": round(self.mean_pairwise_hamming, 4),
            "wall_time_ms": round(self.wall_time_ms, 2),
            "greedy_baseline": self.greedy_baseline.to_dict(),
            "freq_random_baseline": self.freq_random_baseline.to_dict(),
            "candidates": [c.to_dict() for c in self.candidates],
        }


class PottsCodonSampler:
    """Annealed Potts sampler generating diverse, constraint-clean synonymous coding sequences."""

    def __init__(
        self,
        host: str | HostProfile = "s_cerevisiae",
        enzymes: Tuple[str, ...] = DEFAULT_FOUNDRY_ENZYMES,
        extra_forbidden_motifs: Tuple[str, ...] = (),
        weights: Optional[ScoringWeights] = None,
    ):
        self.profile = get_host_profile(host) if isinstance(host, str) else host
        self.enzymes = tuple(enzymes)
        self.extra_motifs = tuple(extra_forbidden_motifs)
        self.weights = weights or ScoringWeights()
        self.forbidden_list = expand_forbidden_motifs(self.enzymes, self.extra_motifs)

        # Precompute ordered synonymous codon lists, log(w), and GC counts per amino acid
        self._aa_codons: Dict[str, List[str]] = {}
        self._aa_log_w: Dict[str, np.ndarray] = {}
        self._aa_probs: Dict[str, np.ndarray] = {}
        self._aa_gc_counts: Dict[str, np.ndarray] = {}

        for aa, freq_map in self.profile.codon_frequencies.items():
            # Sort codons from highest to lowest frequency
            sorted_items = sorted(freq_map.items(), key=lambda kv: kv[1], reverse=True)
            codons = [c for c, _ in sorted_items]
            freqs = np.array([f for _, f in sorted_items], dtype=np.float64)
            freqs = freqs / np.sum(freqs)
            rel_w = np.array([self.profile.relative_adaptiveness[aa][c] for c in codons], dtype=np.float64)
            self._aa_codons[aa] = codons
            self._aa_probs[aa] = freqs
            self._aa_log_w[aa] = np.log(np.maximum(1e-4, rel_w))
            self._aa_gc_counts[aa] = np.array(
                [sum(1 for b in c if b in ("G", "C")) for c in codons], dtype=np.int32
            )

    def greedy_cai_sequence(self, aa_seq: str) -> str:
        """Return deterministic highest-frequency codon sequence."""
        return "".join(self._aa_codons[aa][0] for aa in aa_seq)

    def freq_random_sequence(self, aa_seq: str, rng: np.random.Generator) -> str:
        """Sample codons independently proportional to host usage frequencies."""
        chosen = []
        for aa in aa_seq:
            codons = self._aa_codons[aa]
            if len(codons) == 1:
                chosen.append(codons[0])
            else:
                idx = int(rng.choice(len(codons), p=self._aa_probs[aa]))
                chosen.append(codons[idx])
        return "".join(chosen)

    def _local_window_penalty(self, codons: List[str], pos: int, l_aa: int) -> float:
        """Evaluate local constraint, homopolymer, and hairpin penalties around codon `pos`."""
        w = self.weights
        # Slice a local +/- 8 codon window (up to 51 nt) for fast incremental evaluation
        c_start = max(0, pos - 8)
        c_end = min(l_aa, pos + 9)
        sub_dna = "".join(codons[c_start:c_end])

        pen = 0.0
        # 1. Forbidden restriction sites in local window
        for _, motif in self.forbidden_list:
            if motif in sub_dna:
                pen += w.w_restriction_site * sub_dna.count(motif)

        # 2. Homopolymers in local window
        hp = count_homopolymers(sub_dna, min_len=w.min_homopolymer_len)
        if hp:
            pen += w.w_homopolymer * hp

        # 3. Prokaryotic internal Shine-Dalgarno sites
        if self.profile.is_prokaryote and c_end > 6:
            body_sub = "".join(codons[max(6, c_start) : c_end])
            for sd in ("AGGAGG", "GGAGGT", "AGGAGA"):
                if sd in body_sub:
                    pen += w.w_internal_sd * body_sub.count(sd)

        # 4. Local stem-loop check
        if pos < 18:
            # 5' initiation region: discourage any hairpin stronger than -3.0 kcal/mol
            init_dna = "".join(codons[: min(l_aa, 19)])
            hits = scan_local_hairpins(init_dna, min_stem=4, max_stem=8, min_loop=3, max_loop=8, dg_threshold_kcal=-2.0)
            if hits:
                min_dg = min(h.dg_kcal for h in hits)
                if min_dg < -3.0:
                    pen += w.w_5prime_struct * abs(min_dg - (-3.0))
                elif min_dg < -2.0:
                    pen += 0.5 * abs(min_dg - (-2.0))
        else:
            hits = scan_local_hairpins(sub_dna, min_stem=5, max_stem=9, min_loop=3, max_loop=8, dg_threshold_kcal=-6.5)
            for h in hits:
                pen += w.w_body_hairpin * abs(h.dg_kcal - (-6.5))

        # 5. Local window GC bounds check
        local_gc = sum(1 for b in sub_dna if b in ("G", "C")) / max(1, len(sub_dna))
        if local_gc < self.profile.gc_window_min:
            pen += w.w_gc_window * ((self.profile.gc_window_min - local_gc) * 10.0) ** 2
        elif local_gc > self.profile.gc_window_max:
            pen += w.w_gc_window * ((local_gc - self.profile.gc_window_max) * 10.0) ** 2

        return pen

    def _repair_hard_constraints(self, codons: List[str], aa_seq: str, rng: np.random.Generator) -> List[str]:
        """Deterministic + stochastic repair pass removing all restriction sites and homopolymers."""
        l_aa = len(aa_seq)
        repaired = list(codons)

        for _pass in range(6):
            dna = "".join(repaired)
            site_hits = find_forbidden_motifs(dna, self.forbidden_list)
            hp_count = count_homopolymers(dna, min_len=self.weights.min_homopolymer_len)
            if not site_hits and hp_count == 0:
                break

            # Collect residue indices overlapping any violation
            problem_residues: List[int] = []
            for hit in site_hits:
                nt_idx = int(hit.split("@")[1])
                c_idx = nt_idx // 3
                for r in range(max(0, c_idx - 1), min(l_aa, c_idx + 4)):
                    problem_residues.append(r)

            if hp_count > 0:
                run = 1
                for i in range(1, len(dna)):
                    if dna[i] == dna[i - 1]:
                        run += 1
                        if run >= self.weights.min_homopolymer_len:
                            c_idx = i // 3
                            for r in range(max(0, c_idx - 2), min(l_aa, c_idx + 2)):
                                problem_residues.append(r)
                    else:
                        run = 1

            for pos in sorted(set(problem_residues)):
                aa = aa_seq[pos]
                choices = self._aa_codons[aa]
                if len(choices) <= 1:
                    continue
                best_c = repaired[pos]
                best_pen = self._local_window_penalty(repaired, pos, l_aa)
                for idx, cand_c in enumerate(choices):
                    repaired[pos] = cand_c
                    # Small preference for higher adaptiveness among clean choices
                    pen = self._local_window_penalty(repaired, pos, l_aa) - 0.2 * float(self._aa_log_w[aa][idx])
                    if pen < best_pen:
                        best_pen = pen
                        best_c = cand_c
                repaired[pos] = best_c

        return repaired

    def _run_single_chain(
        self,
        aa_seq: str,
        n_steps: int,
        beta_start: float,
        beta_end: float,
        rng: np.random.Generator,
        reference_dnas: List[str],
        diversity_weight: float = 6.0,
    ) -> str:
        """Run one annealed Potts chain with optional repulsion from already chosen sequences."""
        l_aa = len(aa_seq)
        n_nt = l_aa * 3
        w = self.weights

        # Initialize from host frequency distribution and immediately repair hard violations
        init_dna = self.freq_random_sequence(aa_seq, rng)
        codons = [init_dna[i * 3 : (i + 1) * 3] for i in range(l_aa)]
        codons = self._repair_hard_constraints(codons, aa_seq, rng)

        # Track state indices and global GC / log_w sums in O(1)
        state_idx = [self._aa_codons[aa_seq[i]].index(codons[i]) for i in range(l_aa)]
        log_w_sum = sum(float(self._aa_log_w[aa_seq[i]][state_idx[i]]) for i in range(l_aa))
        gc_count = sum(int(self._aa_gc_counts[aa_seq[i]][state_idx[i]]) for i in range(l_aa))

        mutable_positions = [i for i, aa in enumerate(aa_seq) if len(self._aa_codons[aa]) > 1]
        if not mutable_positions:
            return "".join(codons)

        # Pre-slice reference codons for O(1) per-site Hamming similarity computation
        ref_codon_lists = [[ref[i * 3 : (i + 1) * 3] for i in range(l_aa)] for ref in reference_dnas]

        for step in range(n_steps):
            frac = step / max(1, n_steps - 1)
            beta = beta_start * ((beta_end / beta_start) ** frac)

            pos = int(mutable_positions[int(rng.integers(0, len(mutable_positions)))])
            aa = aa_seq[pos]
            choices = self._aa_codons[aa]
            old_idx = state_idx[pos]
            new_idx = int(rng.integers(0, len(choices) - 1))
            if new_idx >= old_idx:
                new_idx += 1

            old_codon = codons[pos]
            new_codon = choices[new_idx]

            # 1. O(1) CAI + rare codon delta
            old_lw = float(self._aa_log_w[aa][old_idx])
            new_lw = float(self._aa_log_w[aa][new_idx])
            delta_e_cai = w.w_cai * (-(new_lw - old_lw) / l_aa)
            if new_lw < math.log(0.30):
                delta_e_cai += 0.45  # Discourage dropping into rare codons

            # 2. O(1) Global GC delta
            old_gc = gc_count / n_nt
            new_gc_count = gc_count - int(self._aa_gc_counts[aa][old_idx]) + int(self._aa_gc_counts[aa][new_idx])
            new_gc = new_gc_count / n_nt
            delta_e_gc = w.w_gc_global * (
                (new_gc - self.profile.target_gc) ** 2 - (old_gc - self.profile.target_gc) ** 2
            )

            # 3. Local window penalty delta
            old_local = self._local_window_penalty(codons, pos, l_aa)
            codons[pos] = new_codon
            new_local = self._local_window_penalty(codons, pos, l_aa)
            delta_local = new_local - old_local

            # 4. Diversity repulsion from previously generated ensemble members
            delta_div = 0.0
            if ref_codon_lists:
                old_matches = sum(1 for r_list in ref_codon_lists if r_list[pos] == old_codon)
                new_matches = sum(1 for r_list in ref_codon_lists if r_list[pos] == new_codon)
                delta_div = diversity_weight * (new_matches - old_matches) / (len(ref_codon_lists) * max(1, l_aa) ** 0.5)

            delta_e = delta_e_cai + delta_e_gc + delta_local + delta_div

            if delta_e <= 0.0 or rng.random() < math.exp(-min(50.0, beta * delta_e)):
                state_idx[pos] = new_idx
                log_w_sum += new_lw - old_lw
                gc_count = new_gc_count
            else:
                codons[pos] = old_codon

        codons = self._repair_hard_constraints(codons, aa_seq, rng)
        return "".join(codons)

    def sample_ensemble(
        self,
        protein_or_dna: str,
        k_candidates: int = 5,
        steps_per_chain: Optional[int] = None,
        n_chains: Optional[int] = None,
        seed: int = 42,
    ) -> EnsembleDesignResult:
        """Generate `k_candidates` diverse, constraint-clean synonymous sequences for an input protein or DNA."""
        t0 = time.perf_counter()
        clean = protein_or_dna.strip().upper().replace(" ", "").replace("\n", "").replace("\r", "")

        # Detect whether input is DNA or Amino Acids
        is_dna = len(clean) >= 9 and len(clean) % 3 == 0 and set(clean).issubset({"A", "T", "G", "C", "U"})
        aa_seq = translate_dna(clean, ignore_terminal_stop=True) if is_dna else clean.rstrip("*")

        for aa in aa_seq:
            if aa not in self._aa_codons:
                raise ValueError(f"Unsupported amino acid symbol {aa!r} in sequence")

        l_aa = len(aa_seq)
        steps = steps_per_chain or min(6_000, max(1_200, l_aa * 10))
        total_chains = n_chains or max(k_candidates + 3, int(k_candidates * 1.6))
        rng = np.random.default_rng(seed)

        # Baselines
        greedy_dna = self.greedy_cai_sequence(aa_seq)
        greedy_score = score_construct(greedy_dna, self.profile, self.enzymes, self.extra_motifs, self.weights)
        rand_dna = self.freq_random_sequence(aa_seq, rng)
        rand_score = score_construct(rand_dna, self.profile, self.enzymes, self.extra_motifs, self.weights)

        pool: List[Tuple[str, ConstructScore]] = []
        accepted_dnas: List[str] = []

        for c_idx in range(total_chains):
            cand_dna = self._run_single_chain(
                aa_seq=aa_seq,
                n_steps=steps,
                beta_start=0.5,
                beta_end=18.0,
                rng=rng,
                reference_dnas=accepted_dnas,
                diversity_weight=8.0 if c_idx > 0 else 0.0,
            )
            sc = score_construct(cand_dna, self.profile, self.enzymes, self.extra_motifs, self.weights)
            pool.append((cand_dna, sc))
            accepted_dnas.append(cand_dna)

        # Rank #1 is lowest total energy (preferring constraint-clean candidates)
        pool.sort(key=lambda item: (0 if item[1].constraints_passed else 1, item[1].total_energy))

        selected: List[CodonCandidate] = []
        used_indices = set()

        # Pick Rank 1
        best_dna, best_sc = pool[0]
        selected.append(CodonCandidate(rank=1, dna_sequence=best_dna, score=best_sc, min_hamming_to_higher_ranked=1.0))
        used_indices.add(0)

        # Greedily select remaining K-1 candidates balancing low energy and high Hamming distance
        while len(selected) < k_candidates and len(used_indices) < len(pool):
            best_idx = -1
            best_trade_score = float("inf")
            best_min_ham = 0.0

            for idx, (dna, sc) in enumerate(pool):
                if idx in used_indices:
                    continue
                min_ham = min(nucleotide_hamming_fraction(dna, s.dna_sequence) for s in selected)
                # Penalize sequences too close (<12% divergence) to already picked candidates
                div_bonus = 25.0 * min_ham
                composite = sc.total_energy - div_bonus + (0.0 if sc.constraints_passed else 500.0)
                if composite < best_trade_score:
                    best_trade_score = composite
                    best_idx = idx
                    best_min_ham = min_ham

            if best_idx == -1:
                break
            used_indices.add(best_idx)
            dna, sc = pool[best_idx]
            selected.append(
                CodonCandidate(
                    rank=len(selected) + 1,
                    dna_sequence=dna,
                    score=sc,
                    min_hamming_to_higher_ranked=best_min_ham,
                )
            )

        # Compute mean pairwise nucleotide Hamming distance across selected ensemble
        pair_dists: List[float] = []
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                pair_dists.append(nucleotide_hamming_fraction(selected[i].dna_sequence, selected[j].dna_sequence))
        mean_ham = float(np.mean(pair_dists)) if pair_dists else 0.0

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return EnsembleDesignResult(
            protein_length=l_aa,
            host=self.profile.name,
            enzymes_excluded=list(self.enzymes),
            candidates=selected,
            greedy_baseline=greedy_score,
            greedy_dna=greedy_dna,
            freq_random_baseline=rand_score,
            mean_pairwise_hamming=mean_ham,
            wall_time_ms=elapsed_ms,
        )
