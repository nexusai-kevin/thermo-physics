"""
thermo_physics.codon.tables
===========================
Host organism codon usage tables, standard genetic code translations, and
biofoundry restriction enzyme / sequence constraint presets.

Supported Hosts:
  - s_cerevisiae : Saccharomyces cerevisiae (yeast biofoundry / Sc2.0 workflows)
  - e_coli       : Escherichia coli K-12
  - b_subtilis   : Bacillus subtilis 168
  - c_griseus    : Cricetulus griseus (CHO mammalian expression)
  - h_sapiens    : Homo sapiens
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}


def reverse_complement(dna: str) -> str:
    """Return reverse-complement of a DNA string (5' -> 3')."""
    return "".join(COMPLEMENT.get(b, "N") for b in reversed(dna.upper()))


# Standard Genetic Code (DNA codon -> 1-letter amino acid, '*' = stop)
CODON_TO_AA: Dict[str, str] = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


def translate_dna(dna: str, ignore_terminal_stop: bool = True) -> str:
    """Translate an in-frame DNA sequence (A/T/G/C, U converted to T) into amino acids."""
    clean = dna.upper().replace("U", "T").replace(" ", "").replace("\n", "").replace("\r", "")
    if len(clean) % 3 != 0:
        raise ValueError(f"DNA length ({len(clean)}) is not a multiple of 3")
    aas: List[str] = []
    for i in range(0, len(clean), 3):
        codon = clean[i : i + 3]
        aa = CODON_TO_AA.get(codon)
        if aa is None:
            raise ValueError(f"Invalid codon {codon!r} at nucleotide position {i}")
        aas.append(aa)
    if ignore_terminal_stop and aas and aas[-1] == "*":
        aas.pop()
    if "*" in aas:
        idx = aas.index("*")
        raise ValueError(f"Internal stop codon at residue index {idx}")
    return "".join(aas)


@dataclass(frozen=True)
class HostProfile:
    """Host organism codon usage and physiological GC profile."""

    name: str
    scientific_name: str
    target_gc: float
    gc_window_min: float
    gc_window_max: float
    is_prokaryote: bool
    codon_frequencies: Dict[str, Dict[str, float]]
    # Precomputed relative adaptiveness w(c) = freq(c) / max_synonymous_freq
    relative_adaptiveness: Dict[str, Dict[str, float]] = field(init=False)

    def __post_init__(self) -> None:
        rel: Dict[str, Dict[str, float]] = {}
        for aa, table in self.codon_frequencies.items():
            max_f = max(table.values())
            rel[aa] = {c: max(1e-4, f / max_f) for c, f in table.items()}
        object.__setattr__(self, "relative_adaptiveness", rel)


# ---------------------------------------------------------------------------
# Host Codon Usage Tables (Synonymous fractional frequencies)
# ---------------------------------------------------------------------------

S_CEREVISIAE_CODONS: Dict[str, Dict[str, float]] = {
    "A": {"GCT": 0.38, "GCC": 0.22, "GCA": 0.29, "GCG": 0.11},
    "C": {"TGT": 0.63, "TGC": 0.37},
    "D": {"GAT": 0.65, "GAC": 0.35},
    "E": {"GAA": 0.70, "GAG": 0.30},
    "F": {"TTT": 0.59, "TTC": 0.41},
    "G": {"GGT": 0.47, "GGA": 0.22, "GGC": 0.19, "GGG": 0.12},
    "H": {"CAT": 0.64, "CAC": 0.36},
    "I": {"ATT": 0.46, "ATC": 0.26, "ATA": 0.27},
    "K": {"AAA": 0.58, "AAG": 0.42},
    "L": {"TTG": 0.29, "TTA": 0.28, "CTT": 0.13, "CTA": 0.14, "CTG": 0.11, "CTC": 0.06},
    "M": {"ATG": 1.00},
    "N": {"AAT": 0.59, "AAC": 0.41},
    "P": {"CCA": 0.42, "CCT": 0.31, "CCC": 0.15, "CCG": 0.12},
    "Q": {"CAA": 0.69, "CAG": 0.31},
    "R": {"AGA": 0.48, "AGG": 0.21, "CGT": 0.14, "CGA": 0.07, "CGC": 0.06, "CGG": 0.04},
    "S": {"TCT": 0.26, "TCA": 0.21, "TCC": 0.16, "AGT": 0.16, "AGC": 0.11, "TCG": 0.10},
    "T": {"ACT": 0.35, "ACA": 0.30, "ACC": 0.22, "ACG": 0.14},
    "V": {"GTT": 0.39, "GTG": 0.21, "GTC": 0.21, "GTA": 0.20},
    "W": {"TGG": 1.00},
    "Y": {"TAT": 0.56, "TAC": 0.44},
}

E_COLI_CODONS: Dict[str, Dict[str, float]] = {
    "A": {"GCG": 0.36, "GCC": 0.27, "GCA": 0.21, "GCT": 0.16},
    "C": {"TGC": 0.56, "TGT": 0.44},
    "D": {"GAT": 0.63, "GAC": 0.37},
    "E": {"GAA": 0.69, "GAG": 0.31},
    "F": {"TTT": 0.57, "TTC": 0.43},
    "G": {"GGC": 0.41, "GGT": 0.34, "GGG": 0.15, "GGA": 0.11},
    "H": {"CAT": 0.57, "CAC": 0.43},
    "I": {"ATT": 0.51, "ATC": 0.42, "ATA": 0.07},
    "K": {"AAA": 0.76, "AAG": 0.24},
    "L": {"CTG": 0.50, "TTA": 0.13, "TTG": 0.13, "CTC": 0.10, "CTT": 0.10, "CTA": 0.04},
    "M": {"ATG": 1.00},
    "N": {"AAC": 0.55, "AAT": 0.45},
    "P": {"CCG": 0.53, "CCA": 0.19, "CCT": 0.16, "CCC": 0.12},
    "Q": {"CAG": 0.65, "CAA": 0.35},
    "R": {"CGC": 0.40, "CGT": 0.38, "CGG": 0.10, "CGA": 0.06, "AGA": 0.04, "AGG": 0.02},
    "S": {"AGC": 0.28, "TCT": 0.15, "TCC": 0.15, "TCG": 0.15, "AGT": 0.15, "TCA": 0.12},
    "T": {"ACC": 0.44, "ACG": 0.27, "ACT": 0.16, "ACA": 0.13},
    "V": {"GTG": 0.37, "GTT": 0.26, "GTC": 0.22, "GTA": 0.15},
    "W": {"TGG": 1.00},
    "Y": {"TAT": 0.57, "TAC": 0.43},
}

B_SUBTILIS_CODONS: Dict[str, Dict[str, float]] = {
    "A": {"GCA": 0.29, "GCT": 0.26, "GCG": 0.25, "GCC": 0.20},
    "C": {"TGC": 0.55, "TGT": 0.45},
    "D": {"GAT": 0.64, "GAC": 0.36},
    "E": {"GAA": 0.68, "GAG": 0.32},
    "F": {"TTT": 0.68, "TTC": 0.32},
    "G": {"GGA": 0.32, "GGC": 0.32, "GGT": 0.21, "GGG": 0.15},
    "H": {"CAT": 0.68, "CAC": 0.32},
    "I": {"ATC": 0.42, "ATT": 0.46, "ATA": 0.12},
    "K": {"AAA": 0.70, "AAG": 0.30},
    "L": {"CTG": 0.29, "CTT": 0.24, "TTA": 0.20, "TTG": 0.14, "CTC": 0.09, "CTA": 0.04},
    "M": {"ATG": 1.00},
    "N": {"AAT": 0.56, "AAC": 0.44},
    "P": {"CCG": 0.44, "CCT": 0.29, "CCA": 0.19, "CCC": 0.08},
    "Q": {"CAA": 0.52, "CAG": 0.48},
    "R": {"CGC": 0.27, "AGA": 0.25, "CGT": 0.19, "CGG": 0.11, "CGA": 0.10, "AGG": 0.08},
    "S": {"TCA": 0.26, "AGC": 0.23, "TCT": 0.21, "TCC": 0.13, "AGT": 0.11, "TCG": 0.06},
    "T": {"ACA": 0.42, "ACG": 0.27, "ACT": 0.17, "ACC": 0.14},
    "V": {"GTT": 0.29, "GTG": 0.27, "GTC": 0.26, "GTA": 0.18},
    "W": {"TGG": 1.00},
    "Y": {"TAT": 0.64, "TAC": 0.36},
}

C_GRISEUS_CODONS: Dict[str, Dict[str, float]] = {
    "A": {"GCC": 0.39, "GCT": 0.31, "GCA": 0.22, "GCG": 0.08},
    "C": {"TGC": 0.53, "TGT": 0.47},
    "D": {"GAC": 0.53, "GAT": 0.47},
    "E": {"GAG": 0.59, "GAA": 0.41},
    "F": {"TTC": 0.53, "TTT": 0.47},
    "G": {"GGC": 0.34, "GGA": 0.26, "GGG": 0.22, "GGT": 0.18},
    "H": {"CAC": 0.56, "CAT": 0.44},
    "I": {"ATC": 0.51, "ATT": 0.35, "ATA": 0.14},
    "K": {"AAG": 0.61, "AAA": 0.39},
    "L": {"CTG": 0.40, "CTC": 0.19, "CTT": 0.13, "TTG": 0.13, "CTA": 0.08, "TTA": 0.07},
    "M": {"ATG": 1.00},
    "N": {"AAC": 0.54, "AAT": 0.46},
    "P": {"CCC": 0.32, "CCT": 0.31, "CCA": 0.28, "CCG": 0.09},
    "Q": {"CAG": 0.76, "CAA": 0.24},
    "R": {"AGG": 0.22, "AGA": 0.20, "CGG": 0.19, "CGC": 0.18, "CGA": 0.12, "CGT": 0.09},
    "S": {"AGC": 0.23, "TCC": 0.22, "TCT": 0.20, "AGT": 0.15, "TCA": 0.14, "TCG": 0.06},
    "T": {"ACC": 0.37, "ACA": 0.29, "ACT": 0.25, "ACG": 0.09},
    "V": {"GTG": 0.48, "GTC": 0.24, "GTT": 0.17, "GTA": 0.11},
    "W": {"TGG": 1.00},
    "Y": {"TAC": 0.56, "TAT": 0.44},
}

H_SAPIENS_CODONS: Dict[str, Dict[str, float]] = {
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

# Dictyostelium discoideum (highly AT-rich slime mold genome, coding GC ~27%)
D_DISCOIDEUM_CODONS: Dict[str, Dict[str, float]] = {
    "A": {"GCT": 0.36, "GCA": 0.49, "GCC": 0.12, "GCG": 0.03},
    "C": {"TGT": 0.77, "TGC": 0.23},
    "D": {"GAT": 0.85, "GAC": 0.15},
    "E": {"GAA": 0.84, "GAG": 0.16},
    "F": {"TTT": 0.72, "TTC": 0.28},
    "G": {"GGT": 0.57, "GGA": 0.34, "GGC": 0.05, "GGG": 0.04},
    "H": {"CAT": 0.79, "CAC": 0.21},
    "I": {"ATT": 0.61, "ATA": 0.25, "ATC": 0.14},
    "K": {"AAA": 0.78, "AAG": 0.22},
    "L": {"TTA": 0.59, "TTG": 0.18, "CTT": 0.12, "CTA": 0.06, "CTC": 0.03, "CTG": 0.02},
    "M": {"ATG": 1.00},
    "N": {"AAT": 0.81, "AAC": 0.19},
    "P": {"CCA": 0.67, "CCT": 0.24, "CCC": 0.06, "CCG": 0.03},
    "Q": {"CAA": 0.89, "CAG": 0.11},
    "R": {"AGA": 0.65, "CGT": 0.18, "AGG": 0.08, "CGA": 0.05, "CGC": 0.02, "CGG": 0.02},
    "S": {"TCA": 0.37, "TCT": 0.26, "AGT": 0.23, "TCC": 0.06, "AGC": 0.05, "TCG": 0.03},
    "T": {"ACA": 0.49, "ACT": 0.37, "ACC": 0.11, "ACG": 0.03},
    "V": {"GTT": 0.54, "GTA": 0.28, "GTG": 0.10, "GTC": 0.08},
    "W": {"TGG": 1.00},
    "Y": {"TAT": 0.80, "TAC": 0.20},
}


HOST_PROFILES: Dict[str, HostProfile] = {
    "s_cerevisiae": HostProfile(
        name="s_cerevisiae",
        scientific_name="Saccharomyces cerevisiae",
        target_gc=0.39,
        gc_window_min=0.26,
        gc_window_max=0.56,
        is_prokaryote=False,
        codon_frequencies=S_CEREVISIAE_CODONS,
    ),
    "e_coli": HostProfile(
        name="e_coli",
        scientific_name="Escherichia coli K-12",
        target_gc=0.51,
        gc_window_min=0.35,
        gc_window_max=0.68,
        is_prokaryote=True,
        codon_frequencies=E_COLI_CODONS,
    ),
    "d_discoideum": HostProfile(
        name="d_discoideum",
        scientific_name="Dictyostelium discoideum",
        target_gc=0.27,
        gc_window_min=0.14,
        gc_window_max=0.45,
        is_prokaryote=False,
        codon_frequencies=D_DISCOIDEUM_CODONS,
    ),
    "b_subtilis": HostProfile(
        name="b_subtilis",
        scientific_name="Bacillus subtilis 168",
        target_gc=0.43,
        gc_window_min=0.30,
        gc_window_max=0.60,
        is_prokaryote=True,
        codon_frequencies=B_SUBTILIS_CODONS,
    ),
    "c_griseus": HostProfile(
        name="c_griseus",
        scientific_name="Cricetulus griseus (CHO)",
        target_gc=0.50,
        gc_window_min=0.35,
        gc_window_max=0.68,
        is_prokaryote=False,
        codon_frequencies=C_GRISEUS_CODONS,
    ),
    "h_sapiens": HostProfile(
        name="h_sapiens",
        scientific_name="Homo sapiens",
        target_gc=0.52,
        gc_window_min=0.35,
        gc_window_max=0.70,
        is_prokaryote=False,
        codon_frequencies=H_SAPIENS_CODONS,
    ),
}

HOST_ALIASES: Dict[str, str] = {
    "yeast": "s_cerevisiae",
    "scerevisiae": "s_cerevisiae",
    "s_cerevisiae": "s_cerevisiae",
    "ecoli": "e_coli",
    "e_coli": "e_coli",
    "dicty": "d_discoideum",
    "dictyostelium": "d_discoideum",
    "ddiscoideum": "d_discoideum",
    "d_discoideum": "d_discoideum",
    "bsubtilis": "b_subtilis",
    "b_subtilis": "b_subtilis",
    "cho": "c_griseus",
    "c_griseus": "c_griseus",
    "human": "h_sapiens",
    "hsapiens": "h_sapiens",
    "h_sapiens": "h_sapiens",
}


def get_host_profile(name: str) -> HostProfile:
    key = HOST_ALIASES.get(name.strip().lower().replace("-", "_").replace(" ", "_"))
    if not key or key not in HOST_PROFILES:
        valid = ", ".join(sorted(HOST_PROFILES.keys()))
        raise ValueError(f"Unknown host {name!r}. Valid hosts: {valid}")
    return HOST_PROFILES[key]


# ---------------------------------------------------------------------------
# Biofoundry Restriction Enzymes (Type IIS / Golden Gate / MoClo & Standard)
# ---------------------------------------------------------------------------

RESTRICTION_ENZYMES: Dict[str, str] = {
    # Type IIS Golden Gate / MoClo standard enzymes (non-palindromic -> checked on both strands)
    "BsaI": "GGTCTC",
    "BsmBI": "CGTCTC",
    "Esp3I": "CGTCTC",
    "BbsI": "GAAGAC",
    "BpiI": "GAAGAC",
    "SapI": "GCTCTTC",
    "LguI": "GCTCTTC",
    "AarI": "CACCTGC",
    "BtgZI": "GCGATG",
    # Common Type II cloning enzymes (palindromic)
    "EcoRI": "GAATTC",
    "BamHI": "GGATCC",
    "XbaI": "TCTAGA",
    "SpeI": "ACTAGT",
    "PstI": "CTGCAG",
    "NotI": "GCGGCCGC",
    "HindIII": "AAGCTT",
    "XhoI": "CTCGAG",
    "NdeI": "CATATG",
}

DEFAULT_FOUNDRY_ENZYMES: Tuple[str, ...] = ("BsaI", "BsmBI", "BbsI", "SapI")


def expand_forbidden_motifs(
    enzymes: Tuple[str, ...] = DEFAULT_FOUNDRY_ENZYMES,
    extra_motifs: Tuple[str, ...] = (),
) -> List[Tuple[str, str]]:
    """Return list of (label, motif_5_to_3) including reverse-complements for non-palindromic enzymes."""
    patterns: List[Tuple[str, str]] = []
    seen = set()
    for enz in enzymes:
        seq = RESTRICTION_ENZYMES.get(enz)
        if seq is None:
            # Allow raw sequence passed in enzymes list
            seq = enz.upper()
        for strand_label, motif in ((enz, seq), (f"{enz}_rc", reverse_complement(seq))):
            if motif not in seen:
                seen.add(motif)
                patterns.append((strand_label, motif))
    for m in extra_motifs:
        motif = m.upper().replace("U", "T")
        if motif and motif not in seen:
            seen.add(motif)
            patterns.append((f"motif:{motif}", motif))
    return patterns
