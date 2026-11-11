"""
thermo_physics.codon.scoring
============================
Biophysical energy decomposition and constraint evaluation for coding DNA/mRNA.

Energy terms in the Potts Hamiltonian H(s):
  1. E_cai        : Host codon adaptation potential (-mean ln w_k) + rare-codon doublet penalty
  2. E_5prime     : 5' translation initiation secondary structure penalty (nt 1..45, Turner 2004 kcal/mol)
  3. E_hairpin    : Internal strong stem-loop elongation barrier penalty (Turner 2004 kcal/mol)
  4. E_gc         : Global host GC deviation + 50-nt sliding-window GC excursion penalty
  5. E_constraint : Hard biofoundry assembly/synthesis violations (Type IIS/II restriction sites
                    on forward or reverse-complement strands, homopolymers, internal SD sites)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from thermo_physics.codon.tables import (
    DEFAULT_FOUNDRY_ENZYMES,
    HostProfile,
    expand_forbidden_motifs,
    get_host_profile,
    translate_dna,
)

# ---------------------------------------------------------------------------
# Turner 2004 Nearest-Neighbor RNA Base-Pair Stacking Parameters (37 C, kcal/mol)
# Key: ((5'-b1, 3'-b2), (5'-b3, 3'-b4)) where b1-b2 is outer pair and b3-b4 is inner pair
# Stored in DNA alphabet (T in place of U) for direct evaluation on coding strings.
# ---------------------------------------------------------------------------

CANONICAL_PAIRS = {
    ("A", "T"), ("T", "A"),
    ("G", "C"), ("C", "G"),
    ("G", "T"), ("T", "G"),  # G-U wobble in RNA
}

# Turner 2004 coaxial/helical stack free energies (kcal/mol)
TURNER_STACK_DG: Dict[Tuple[Tuple[str, str], Tuple[str, str]], float] = {
    # GC / CG stacks
    (("G", "C"), ("G", "C")): -3.26,
    (("G", "C"), ("C", "G")): -3.42,
    (("C", "G"), ("G", "C")): -2.36,
    (("C", "G"), ("C", "G")): -3.26,
    # GC with AT/TA
    (("G", "C"), ("A", "T")): -2.24,
    (("G", "C"), ("T", "A")): -2.08,
    (("C", "G"), ("A", "T")): -2.11,
    (("C", "G"), ("T", "A")): -2.35,
    (("A", "T"), ("G", "C")): -2.08,
    (("T", "A"), ("G", "C")): -2.24,
    (("A", "T"), ("C", "G")): -2.35,
    (("T", "A"), ("C", "G")): -2.11,
    # AT / TA stacks
    (("A", "T"), ("A", "T")): -0.93,
    (("A", "T"), ("T", "A")): -1.10,
    (("T", "A"), ("A", "T")): -1.33,
    (("T", "A"), ("T", "A")): -0.93,
    # Wobble GT / TG stacks
    (("G", "C"), ("G", "T")): -2.51,
    (("G", "C"), ("T", "G")): -1.53,
    (("C", "G"), ("G", "T")): -1.41,
    (("C", "G"), ("T", "G")): -2.11,
    (("G", "T"), ("G", "C")): -2.11,
    (("T", "G"), ("G", "C")): -1.41,
    (("G", "T"), ("C", "G")): -1.53,
    (("T", "G"), ("C", "G")): -2.51,
    (("A", "T"), ("G", "T")): -1.36,
    (("A", "T"), ("T", "G")): -1.00,
    (("T", "A"), ("G", "T")): -1.27,
    (("T", "A"), ("T", "G")): -0.55,
    (("G", "T"), ("A", "T")): -0.55,
    (("T", "G"), ("A", "T")): -1.27,
    (("G", "T"), ("T", "A")): -1.00,
    (("T", "G"), ("T", "A")): -1.36,
    (("G", "T"), ("G", "T")): -0.50,
    (("T", "G"), ("T", "G")): -0.50,
    (("G", "T"), ("T", "G")): +0.30,
    (("T", "G"), ("G", "T")): -1.29,
}

# Turner 2004 hairpin loop initiation free energy (kcal/mol) by loop length
HAIRPIN_LOOP_INIT_DG: Dict[int, float] = {
    3: 5.4,
    4: 5.6,
    5: 5.7,
    6: 5.4,
    7: 6.0,
    8: 5.5,
    9: 6.4,
    10: 6.5,
    11: 6.6,
    12: 6.7,
}


@dataclass(frozen=True)
class HairpinHit:
    start_5p: int       # 0-based start of 5' stem
    end_3p: int         # 0-based exclusive end of 3' stem
    stem_len: int
    loop_len: int
    dg_kcal: float      # Negative = stable hairpin


def scan_local_hairpins(
    dna: str,
    min_stem: int = 4,
    max_stem: int = 10,
    min_loop: int = 3,
    max_loop: int = 10,
    dg_threshold_kcal: float = -3.0,
) -> List[HairpinHit]:
    """Scan a DNA/RNA segment for stable local stem-loop hairpins using Turner 2004 stacking."""
    seq = dna.upper()
    n = len(seq)
    hits: List[HairpinHit] = []

    for loop_center in range(min_stem, n - min_stem - min_loop + 1):
        for loop_len in range(min_loop, min(max_loop + 1, n - loop_center - min_stem + 1)):
            # Extend stem outward from the hairpin loop (i going left, j going right)
            i_inner = loop_center - 1
            j_inner = loop_center + loop_len
            if (seq[i_inner], seq[j_inner]) not in CANONICAL_PAIRS:
                continue

            stem_len = 1
            stack_dg = 0.0
            while (
                stem_len < max_stem
                and i_inner - stem_len >= 0
                and j_inner + stem_len < n
                and (seq[i_inner - stem_len], seq[j_inner + stem_len]) in CANONICAL_PAIRS
            ):
                outer_pair = (seq[i_inner - stem_len], seq[j_inner + stem_len])
                inner_pair = (seq[i_inner - stem_len + 1], seq[j_inner + stem_len - 1])
                stack_dg += TURNER_STACK_DG.get((outer_pair, inner_pair), -1.0)
                stem_len += 1

            if stem_len < min_stem:
                continue

            init_dg = HAIRPIN_LOOP_INIT_DG.get(loop_len, 6.7)
            # Terminal AU/GU penalty (+0.45 kcal/mol per terminal non-GC pair)
            outer_p = (seq[i_inner - stem_len + 1], seq[j_inner + stem_len - 1])
            inner_p = (seq[i_inner], seq[j_inner])
            term_pen = 0.0
            if outer_p not in (("G", "C"), ("C", "G")):
                term_pen += 0.45
            if inner_p not in (("G", "C"), ("C", "G")):
                term_pen += 0.45

            total_dg = stack_dg + init_dg + term_pen
            if total_dg <= dg_threshold_kcal:
                hits.append(
                    HairpinHit(
                        start_5p=i_inner - stem_len + 1,
                        end_3p=j_inner + stem_len,
                        stem_len=stem_len,
                        loop_len=loop_len,
                        dg_kcal=round(total_dg, 3),
                    )
                )
    return hits


@dataclass
class ScoringWeights:
    """Weights for the Potts codon Hamiltonian H(s) (length-invariant expression terms)."""

    w_cai: float = 8.0                   # Scales mean(-ln w_k)
    w_rare_doublet: float = 6.0          # Scales rare-codon doublets per 100 codons
    w_5prime_struct: float = 1.2         # Per kcal/mol below -1.5 kcal/mol in 5' initiation region
    w_body_hairpin: float = 0.8          # Per kcal/mol below -6.5 kcal/mol per 100 codons
    w_gc_global: float = 25.0            # Scales (gc_global - target_gc)^2
    w_gc_window: float = 20.0            # Scales mean squared 50-nt window excursion outside [min, max]
    w_restriction_site: float = 250.0    # Hard penalty per forbidden restriction enzyme match
    w_homopolymer: float = 80.0          # Hard penalty per homopolymer run >= min_homopolymer
    w_internal_sd: float = 25.0          # Prokaryotic internal Shine-Dalgarno hexamer penalty
    min_homopolymer_len: int = 5
    gc_window_size: int = 50
    gc_window_step: int = 15
    five_prime_nt: int = 45


@dataclass
class ConstructScore:
    """Decomposed biophysical score and constraint audit for a single coding DNA sequence."""

    dna_length: int
    aa_length: int
    host: str
    total_energy: float
    # Individual energy components
    e_cai: float
    e_rare_doublet: float
    e_5prime_struct: float
    e_body_hairpin: float
    e_gc_global: float
    e_gc_window: float
    e_constraints: float
    # Raw interpretable biological metrics
    cai: float
    mean_log_w: float
    rare_codon_fraction: float
    rare_doublet_count: int
    gc_global: float
    gc_5prime: float
    gc_window_min_observed: float
    gc_window_max_observed: float
    gc_window_violations: int
    dg_5prime_min_kcal: float
    strong_body_hairpin_count: int
    min_body_hairpin_dg_kcal: float
    restriction_site_count: int
    restriction_sites_found: List[str] = field(default_factory=list)
    homopolymer_count: int = 0
    internal_sd_count: int = 0

    @property
    def constraints_passed(self) -> bool:
        return self.restriction_site_count == 0 and self.homopolymer_count == 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def count_homopolymers(dna: str, min_len: int = 5) -> int:
    """Count distinct homopolymer runs of identical nucleotides of length >= min_len."""
    if not dna:
        return 0
    count = 0
    run_len = 1
    for i in range(1, len(dna)):
        if dna[i] == dna[i - 1]:
            run_len += 1
        else:
            if run_len >= min_len:
                count += 1 + (run_len - min_len)
            run_len = 1
    if run_len >= min_len:
        count += 1 + (run_len - min_len)
    return count


def find_forbidden_motifs(
    dna: str,
    motif_list: List[Tuple[str, str]],
) -> List[str]:
    """Find occurrences of forbidden restriction or sequence motifs."""
    hits: List[str] = []
    seq = dna.upper()
    for label, motif in motif_list:
        start = 0
        while True:
            idx = seq.find(motif, start)
            if idx == -1:
                break
            hits.append(f"{label}@{idx}")
            start = idx + 1
    return hits


def score_construct(
    dna: str,
    host: str | HostProfile = "s_cerevisiae",
    enzymes: Tuple[str, ...] = DEFAULT_FOUNDRY_ENZYMES,
    extra_forbidden_motifs: Tuple[str, ...] = (),
    weights: Optional[ScoringWeights] = None,
) -> ConstructScore:
    """Compute the full Potts energy decomposition and biological metrics for a coding sequence."""
    profile = get_host_profile(host) if isinstance(host, str) else host
    w = weights or ScoringWeights()

    clean_dna = dna.upper().replace("U", "T").replace(" ", "").replace("\n", "").replace("\r", "")
    aa_seq = translate_dna(clean_dna, ignore_terminal_stop=True)
    l_aa = len(aa_seq)
    coding_dna = clean_dna[: l_aa * 3]

    # 1. Codon Adaptation Index & Rare Codon Doublets
    log_w_sum = 0.0
    rare_count = 0
    rare_doublets = 0
    prev_rare = False

    for k in range(l_aa):
        aa = aa_seq[k]
        codon = coding_dna[k * 3 : (k + 1) * 3]
        w_val = profile.relative_adaptiveness[aa].get(codon, 0.01)
        log_w_sum += math.log(max(1e-4, w_val))
        is_rare = w_val < 0.30
        if is_rare:
            rare_count += 1
            if prev_rare:
                rare_doublets += 1
        prev_rare = is_rare

    mean_log_w = log_w_sum / max(1, l_aa)
    cai = math.exp(mean_log_w)
    e_cai = w.w_cai * (-mean_log_w)
    e_rare_doublet = w.w_rare_doublet * (rare_doublets * 100.0 / max(1, l_aa))

    # 2. Global and Sliding-Window GC Content
    n_nt = len(coding_dna)
    gc_total = sum(1 for b in coding_dna if b in ("G", "C"))
    gc_global = gc_total / max(1, n_nt)
    e_gc_global = w.w_gc_global * ((gc_global - profile.target_gc) ** 2)

    five_p_slice = coding_dna[: min(n_nt, w.five_prime_nt)]
    gc_5prime = sum(1 for b in five_p_slice if b in ("G", "C")) / max(1, len(five_p_slice))

    win_size = min(w.gc_window_size, n_nt)
    win_gcs: List[float] = []
    win_violations = 0
    e_gc_window_sum = 0.0
    if win_size >= 15:
        for start in range(0, max(1, n_nt - win_size + 1), w.gc_window_step):
            sub = coding_dna[start : start + win_size]
            g_frac = sum(1 for b in sub if b in ("G", "C")) / len(sub)
            win_gcs.append(g_frac)
            if g_frac < profile.gc_window_min:
                diff = profile.gc_window_min - g_frac
                win_violations += 1
                e_gc_window_sum += w.w_gc_window * (diff * 10.0) ** 2
            elif g_frac > profile.gc_window_max:
                diff = g_frac - profile.gc_window_max
                win_violations += 1
                e_gc_window_sum += w.w_gc_window * (diff * 10.0) ** 2
    else:
        win_gcs.append(gc_global)
    e_gc_window = e_gc_window_sum / max(1, len(win_gcs))

    # 3. 5' Translation Initiation Structure (nt 1..54 covering start region)
    init_region = coding_dna[: min(n_nt, w.five_prime_nt + 12)]
    five_p_hits = scan_local_hairpins(init_region, min_stem=4, max_stem=9, min_loop=3, max_loop=9, dg_threshold_kcal=-1.5)
    dg_5prime_min = min((h.dg_kcal for h in five_p_hits), default=0.0)
    e_5prime_struct = 0.0
    if dg_5prime_min < -1.5:
        e_5prime_struct = w.w_5prime_struct * abs(dg_5prime_min - (-1.5))

    # 4. Internal Coding Body Strong Stem-Loops (normalized per 100 codons)
    body_region = coding_dna[min(n_nt, 30) :]
    body_hits = scan_local_hairpins(body_region, min_stem=5, max_stem=10, min_loop=3, max_loop=8, dg_threshold_kcal=-6.5)
    strong_body_count = len(body_hits)
    min_body_dg = min((h.dg_kcal for h in body_hits), default=0.0)
    raw_body_hp = sum(w.w_body_hairpin * abs(h.dg_kcal - (-6.5)) for h in body_hits)
    e_body_hairpin = raw_body_hp / max(1.0, l_aa / 100.0)

    # 5. Hard Biofoundry Assembly & Synthesis Constraints
    forbidden_list = expand_forbidden_motifs(enzymes=enzymes, extra_motifs=extra_forbidden_motifs)
    site_hits = find_forbidden_motifs(coding_dna, forbidden_list)
    homopolymers = count_homopolymers(coding_dna, min_len=w.min_homopolymer_len)

    sd_hits = 0
    if profile.is_prokaryote and n_nt > 30:
        # Internal Shine-Dalgarno hexamers stall prokaryotic ribosomes
        for sd_motif in ("AGGAGG", "GGAGGT", "AGGAGA"):
            sd_hits += coding_dna[18:].count(sd_motif)

    e_constraints = (
        w.w_restriction_site * len(site_hits)
        + w.w_homopolymer * homopolymers
        + w.w_internal_sd * sd_hits
    )

    total_e = (
        e_cai
        + e_rare_doublet
        + e_5prime_struct
        + e_body_hairpin
        + e_gc_global
        + e_gc_window
        + e_constraints
    )

    return ConstructScore(
        dna_length=n_nt,
        aa_length=l_aa,
        host=profile.name,
        total_energy=round(total_e, 4),
        e_cai=round(e_cai, 4),
        e_rare_doublet=round(e_rare_doublet, 4),
        e_5prime_struct=round(e_5prime_struct, 4),
        e_body_hairpin=round(e_body_hairpin, 4),
        e_gc_global=round(e_gc_global, 4),
        e_gc_window=round(e_gc_window, 4),
        e_constraints=round(e_constraints, 4),
        cai=round(cai, 4),
        mean_log_w=round(mean_log_w, 4),
        rare_codon_fraction=round(rare_count / max(1, l_aa), 4),
        rare_doublet_count=rare_doublets,
        gc_global=round(gc_global, 4),
        gc_5prime=round(gc_5prime, 4),
        gc_window_min_observed=round(min(win_gcs), 4),
        gc_window_max_observed=round(max(win_gcs), 4),
        gc_window_violations=win_violations,
        dg_5prime_min_kcal=round(dg_5prime_min, 3),
        strong_body_hairpin_count=strong_body_count,
        min_body_hairpin_dg_kcal=round(min_body_dg, 3),
        restriction_site_count=len(site_hits),
        restriction_sites_found=site_hits,
        homopolymer_count=homopolymers,
        internal_sd_count=sd_hits,
    )
