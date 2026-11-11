"""
Unit and verification tests for the offline Potts codon package (`thermo_physics.codon`).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from thermo_physics.codon import (
    DEFAULT_FOUNDRY_ENZYMES,
    PottsCodonSampler,
    evaluate_construct_dataset,
    load_construct_csv,
    nucleotide_hamming_fraction,
    reverse_complement,
    scan_local_hairpins,
    score_construct,
    translate_dna,
)


def test_translation_and_host_profiles():
    dna = "ATGAGTAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGAC"
    aa = translate_dna(dna)
    assert aa.startswith("MSKGEELFTG")
    assert len(aa) == len(dna) // 3

    for host in ("s_cerevisiae", "e_coli", "b_subtilis", "c_griseus", "h_sapiens"):
        sc = score_construct(dna, host=host)
        assert sc.aa_length == len(aa)
        assert 0.0 < sc.cai <= 1.0
        assert 0.0 < sc.gc_global < 1.0


def test_restriction_and_homopolymer_detection():
    # Construct containing BsaI forward (GGTCTC), BsmBI reverse-complement (GAGACG), and 6-G homopolymer
    # Frame: ATG GGT CTC (M G L) + GAG ACG (E T) + GGG GGG (G G) + GCT (A)
    dna = "ATGGGTCTCGAGACGGGGGGGGCT"
    sc = score_construct(dna, host="s_cerevisiae", enzymes=("BsaI", "BsmBI", "BbsI"))
    assert sc.restriction_site_count >= 2
    assert sc.homopolymer_count >= 1
    assert not sc.constraints_passed


def test_turner_hairpin_scanner():
    # Strong GC stem-loop: GGGCCC + TTTA (loop=4) + GGGCCC
    hairpin_dna = "ATG" + "GGGCCC" + "TTTA" + "GGGCCC" + "GCTGCTGCT"
    hits = scan_local_hairpins(hairpin_dna, min_stem=4, max_stem=8, min_loop=3, max_loop=8, dg_threshold_kcal=-3.0)
    assert len(hits) >= 1
    assert min(h.dg_kcal for h in hits) < -8.0


def test_potts_ensemble_eliminates_violations_and_preserves_protein():
    # Protein that produces BsaI/BsmBI sites or homopolymers under naive encodings
    protein = (
        "MGLERTGGLKKEEEAAAGGGPPPKKKLLLFFFYYYHQQNNSSTTVVVW"
        "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFIC"
    )
    sampler = PottsCodonSampler(host="s_cerevisiae", enzymes=DEFAULT_FOUNDRY_ENZYMES)
    res = sampler.sample_ensemble(protein, k_candidates=4, steps_per_chain=1200, seed=123)

    assert len(res.candidates) == 4
    assert res.mean_pairwise_hamming >= 0.12

    for cand in res.candidates:
        # 1. Must translate to the exact target protein
        assert translate_dna(cand.dna_sequence) == protein
        # 2. Must clear all Golden Gate restriction sites and homopolymers
        assert cand.score.restriction_site_count == 0
        assert cand.score.homopolymer_count == 0
        assert cand.score.constraints_passed
        # 3. Must beat Greedy CAI baseline on total Potts energy
        assert cand.score.total_energy < res.greedy_baseline.total_energy

    # Verify pairwise divergence between all returned candidates
    for i in range(len(res.candidates)):
        for j in range(i + 1, len(res.candidates)):
            ham = nucleotide_hamming_fraction(
                res.candidates[i].dna_sequence,
                res.candidates[j].dna_sequence,
            )
            assert ham >= 0.08


def test_retrospective_correlation_and_zero_leak_privacy(tmp_path: Path):
    csv_file = tmp_path / "secret_constructs.csv"
    protein = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTL"
    sampler = PottsCodonSampler(host="s_cerevisiae", enzymes=DEFAULT_FOUNDRY_ENZYMES)
    res = sampler.sample_ensemble(protein, k_candidates=5, steps_per_chain=800, seed=7)

    rows = []
    # High-performing constraint-clean Potts candidates
    for idx, c in enumerate(res.candidates):
        rows.append((f"SECRET_GENE_GOOD_{idx}", c.dna_sequence, 90.0 - idx * 3.0))

    # Poor-performing variants with strong 5' GC hairpin and rare codons
    bad_5p = "ATG" + "GGGCCCTTTAGGGCCC" + res.candidates[0].dna_sequence[18:]
    # Keep reading frame valid and same length
    for idx in range(5):
        rows.append((f"SECRET_GENE_BAD_{idx}", res.greedy_dna, 15.0 + idx * 1.5))

    with open(csv_file, "w", encoding="utf-8") as f:
        f.write("construct_id,dna_sequence,expression\n")
        for cid, seq, expr in rows:
            f.write(f"{cid},{seq},{expr}\n")

    loaded = load_construct_csv(csv_file)
    assert len(loaded) == 10

    report, local_rows = evaluate_construct_dataset(loaded, host="s_cerevisiae", redesign_bottom_n=2)
    assert report.n_constructs_evaluated == 10
    assert len(local_rows) == 10

    # Strict privacy assertion: neither construct IDs nor DNA sequences appear anywhere in the aggregate report
    serialized_json = json.dumps(report.to_dict())
    serialized_md = report.to_markdown()
    for cid, seq, _ in rows:
        assert cid not in serialized_json
        assert cid not in serialized_md
        assert seq[:24] not in serialized_json
        assert seq[:24] not in serialized_md
