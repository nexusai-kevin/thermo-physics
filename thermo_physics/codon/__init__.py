"""
thermo_physics.codon
====================
Offline Potts Energy-Based Codon Sampler and Biofoundry Retrospective Benchmark Suite.

Zero network calls, zero learned neural weights, inspectable biophysical energy terms:
  - Host codon usage adaptation (S. cerevisiae, E. coli, B. subtilis, CHO, H. sapiens)
  - Turner 2004 5' translation initiation secondary structure and internal stem-loop scoring
  - Sliding-window GC homeostasis
  - Type IIS (Golden Gate / MoClo: BsaI, BsmBI, BbsI, SapI, AarI) & Type II restriction exclusions
  - Homopolymer and prokaryotic internal Shine-Dalgarno exclusions
"""

from thermo_physics.codon.benchmark import (
    AggregateCorrelationReport,
    FeatureCorrelation,
    LocalConstructRow,
    binary_roc_auc,
    evaluate_construct_dataset,
    load_construct_csv,
    pearson_r,
    spearman_rho,
)
from thermo_physics.codon.sampler import (
    CodonCandidate,
    EnsembleDesignResult,
    PottsCodonSampler,
    nucleotide_hamming_fraction,
)
from thermo_physics.codon.scoring import (
    ConstructScore,
    HairpinHit,
    ScoringWeights,
    count_homopolymers,
    find_forbidden_motifs,
    scan_local_hairpins,
    score_construct,
)
from thermo_physics.codon.tables import (
    CODON_TO_AA,
    DEFAULT_FOUNDRY_ENZYMES,
    HOST_PROFILES,
    RESTRICTION_ENZYMES,
    HostProfile,
    expand_forbidden_motifs,
    get_host_profile,
    reverse_complement,
    translate_dna,
)

__all__ = [
    "CODON_TO_AA",
    "DEFAULT_FOUNDRY_ENZYMES",
    "HOST_PROFILES",
    "RESTRICTION_ENZYMES",
    "HostProfile",
    "get_host_profile",
    "reverse_complement",
    "translate_dna",
    "expand_forbidden_motifs",
    "HairpinHit",
    "ScoringWeights",
    "ConstructScore",
    "scan_local_hairpins",
    "count_homopolymers",
    "find_forbidden_motifs",
    "score_construct",
    "CodonCandidate",
    "EnsembleDesignResult",
    "PottsCodonSampler",
    "nucleotide_hamming_fraction",
    "FeatureCorrelation",
    "AggregateCorrelationReport",
    "LocalConstructRow",
    "pearson_r",
    "spearman_rho",
    "binary_roc_auc",
    "evaluate_construct_dataset",
    "load_construct_csv",
]
