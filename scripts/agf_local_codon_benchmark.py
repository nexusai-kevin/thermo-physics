#!/usr/bin/env python3
"""
scripts/agf_local_codon_benchmark.py
====================================
Offline, zero-telemetry CLI for evaluating Potts energy-based codon scoring
and diverse synonymous ensemble sampling on local biofoundry construct datasets.

Privacy & Security Guarantee:
  - Zero network imports or external HTTP calls (runs 100% air-gapped/offline).
  - No learned neural weights; uses explicit host codon usage tables, Turner 2004
    nearest-neighbor RNA stem-loop free energies, sliding-window GC bounds, and
    Type IIS/II restriction enzyme & homopolymer exclusions.
  - Aggregate reports (`--summary-out`) strip all DNA/protein sequences and construct IDs,
    emitting only sample counts, Spearman rho, Pearson r, p-values, and ROC-AUC.

Usage Examples:
  1. Run built-in S. cerevisiae verification demo (no input files needed):
     python scripts/agf_local_codon_benchmark.py demo

  2. Run retrospective correlation on a local CSV/TSV of past constructs:
     python scripts/agf_local_codon_benchmark.py correlate \
       --csv path/to/constructs.csv \
       --seq-col dna_sequence \
       --expr-col expression \
       --host s_cerevisiae \
       --enzymes BsaI,BsmBI,BbsI,SapI \
       --summary-out summary_correlations_only.json \
       --local-detail-out LOCAL_ONLY_construct_diagnostics.csv

  3. Design K diverse, Golden-Gate-clean synonymous encodings for a protein/CDS:
     python scripts/agf_local_codon_benchmark.py design \
       --seq "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK" \
       --host s_cerevisiae \
       --enzymes BsaI,BsmBI,BbsI,SapI \
       --candidates 5
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Ensure repo root is importable if invoked directly as `python scripts/agf_local_codon_benchmark.py`
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thermo_physics.codon import (
    DEFAULT_FOUNDRY_ENZYMES,
    HOST_PROFILES,
    PottsCodonSampler,
    evaluate_construct_dataset,
    load_construct_csv,
    score_construct,
)


def _parse_enzymes(raw: str) -> Tuple[str, ...]:
    if not raw or raw.strip().lower() in ("none", "null", "-"):
        return ()
    return tuple(e.strip() for e in raw.split(",") if e.strip())


def cmd_correlate(args: argparse.Namespace) -> None:
    enzymes = _parse_enzymes(args.enzymes)
    rows = load_construct_csv(
        args.csv,
        seq_col=args.seq_col,
        expr_col=args.expr_col,
        id_col=args.id_col,
    )
    report, local_rows = evaluate_construct_dataset(
        rows=rows,
        host=args.host,
        enzymes=enzymes,
        redesign_bottom_n=args.redesign_bottom,
    )

    print(report.to_markdown())

    if args.summary_out:
        out_path = Path(args.summary_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\n[OK] Wrote privacy-safe aggregate report (zero sequences/IDs) to: {out_path}")

    if args.local_detail_out:
        detail_path = Path(args.local_detail_out)
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        with open(detail_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "construct_id",
                "expression",
                "total_energy",
                "cai",
                "dg_5prime_min_kcal",
                "strong_body_hairpin_count",
                "gc_global",
                "gc_window_violations",
                "rare_codon_fraction",
                "rare_doublet_count",
                "restriction_site_count",
                "restriction_sites_found",
                "homopolymer_count",
                "redesigned_energy",
                "redesigned_cai",
                "redesigned_dg_5prime",
                "redesigned_dna",
            ])
            for r in local_rows:
                sc = r.score
                writer.writerow([
                    r.construct_id,
                    r.expression,
                    sc.total_energy,
                    sc.cai,
                    sc.dg_5prime_min_kcal,
                    sc.strong_body_hairpin_count,
                    sc.gc_global,
                    sc.gc_window_violations,
                    sc.rare_codon_fraction,
                    sc.rare_doublet_count,
                    sc.restriction_site_count,
                    ";".join(sc.restriction_sites_found),
                    sc.homopolymer_count,
                    r.redesigned_energy if r.redesigned_energy is not None else "",
                    r.redesigned_cai if r.redesigned_cai is not None else "",
                    r.redesigned_dg_5prime if r.redesigned_dg_5prime is not None else "",
                    r.redesigned_dna or "",
                ])
        print(f"[LOCAL ONLY] Wrote per-construct diagnostic table to: {detail_path}")


def _read_sequence_arg(seq_arg: Optional[str], fasta_arg: Optional[str]) -> str:
    if seq_arg:
        return seq_arg.strip()
    if fasta_arg:
        lines = Path(fasta_arg).read_text(encoding="utf-8").splitlines()
        return "".join(line.strip() for line in lines if line.strip() and not line.startswith(">"))
    raise ValueError("Provide either --seq <AA_OR_DNA> or --fasta <PATH>")


def cmd_design(args: argparse.Namespace) -> None:
    enzymes = _parse_enzymes(args.enzymes)
    seq = _read_sequence_arg(args.seq, args.fasta)
    sampler = PottsCodonSampler(host=args.host, enzymes=enzymes)
    result = sampler.sample_ensemble(
        protein_or_dna=seq,
        k_candidates=args.candidates,
        steps_per_chain=args.steps,
        seed=args.seed,
    )

    gb = result.greedy_baseline
    rb = result.freq_random_baseline
    print(f"# Potts Ensemble Codon Design (Host: `{result.host}`, Length: `{result.protein_length} aa / {result.protein_length * 3} nt`)")
    print(f"- **Excluded Enzymes:** `{', '.join(result.enzymes_excluded) or 'None'}`")
    print(f"- **Mean Pairwise Nucleotide Divergence Across Top-{len(result.candidates)} Ensemble:** `{result.mean_pairwise_hamming * 100:.1f}%`")
    print(f"- **Runtime:** `{result.wall_time_ms:.1f} ms`\n")

    print("| Candidate / Baseline | Total Energy H(s) | CAI | 5' min dG (kcal/mol) | Body Hairpins | Global GC | GC Win Viol | Restriction Hits | Homopolymers | Min Divergence |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    print(
        f"| **Baseline: Greedy CAI-1.0** | `{gb.total_energy:.2f}` | `{gb.cai:.3f}` | `{gb.dg_5prime_min_kcal:.2f}` | `{gb.strong_body_hairpin_count}` | `{gb.gc_global * 100:.1f}%` | `{gb.gc_window_violations}` | `{gb.restriction_site_count}` | `{gb.homopolymer_count}` | - |"
    )
    print(
        f"| **Baseline: Host-Freq Random** | `{rb.total_energy:.2f}` | `{rb.cai:.3f}` | `{rb.dg_5prime_min_kcal:.2f}` | `{rb.strong_body_hairpin_count}` | `{rb.gc_global * 100:.1f}%` | `{rb.gc_window_violations}` | `{rb.restriction_site_count}` | `{rb.homopolymer_count}` | - |"
    )
    for c in result.candidates:
        sc = c.score
        div_str = "-" if c.rank == 1 else f"{c.min_hamming_to_higher_ranked * 100:.1f}%"
        print(
            f"| **Potts Candidate #{c.rank}** | **`{sc.total_energy:.2f}`** | `{sc.cai:.3f}` | `{sc.dg_5prime_min_kcal:.2f}` | `{sc.strong_body_hairpin_count}` | `{sc.gc_global * 100:.1f}%` | `{sc.gc_window_violations}` | **`{sc.restriction_site_count}`** | **`{sc.homopolymer_count}`** | `{div_str}` |"
        )

    if args.fasta_out:
        out_p = Path(args.fasta_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            for c in result.candidates:
                sc = c.score
                f.write(
                    f">potts_rank_{c.rank} host={result.host} energy={sc.total_energy:.2f} "
                    f"cai={sc.cai:.3f} dg5p={sc.dg_5prime_min_kcal:.2f} gc={sc.gc_global:.3f} "
                    f"restriction_hits={sc.restriction_site_count} homopolymers={sc.homopolymer_count}\n"
                )
                f.write(f"{c.dna_sequence}\n")
        print(f"\n[OK] Wrote {len(result.candidates)} candidate FASTA records to: {out_p}")


def cmd_demo(args: argparse.Namespace) -> None:
    """Self-contained verification demo on a 180-aa reporter protein in S. cerevisiae."""
    rng = np.random.default_rng(2026)
    protein = (
        "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTL"
        "VTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLV"
        "NRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLAD"
    )
    sampler = PottsCodonSampler(host="s_cerevisiae", enzymes=DEFAULT_FOUNDRY_ENZYMES)

    print("=" * 84)
    print("1. ENSEMBLE DESIGN BENCHMARK (180-aa Reporter in S. cerevisiae, Golden Gate Enzymes)")
    print("=" * 84)
    design_res = sampler.sample_ensemble(protein, k_candidates=5, seed=42)
    gb = design_res.greedy_baseline
    rb = design_res.freq_random_baseline
    print(f"Greedy CAI-1.0 Baseline -> Energy: {gb.total_energy:6.2f} | CAI: {gb.cai:.3f} | 5' dG: {gb.dg_5prime_min_kcal:5.2f} kcal/mol | Restriction Hits: {gb.restriction_site_count} | Homopolymers: {gb.homopolymer_count}")
    print(f"Freq-Random Baseline    -> Energy: {rb.total_energy:6.2f} | CAI: {rb.cai:.3f} | 5' dG: {rb.dg_5prime_min_kcal:5.2f} kcal/mol | Restriction Hits: {rb.restriction_site_count} | Homopolymers: {rb.homopolymer_count}")
    for c in design_res.candidates:
        sc = c.score
        print(
            f"Potts Candidate #{c.rank}      -> Energy: {sc.total_energy:6.2f} | CAI: {sc.cai:.3f} | 5' dG: {sc.dg_5prime_min_kcal:5.2f} kcal/mol | Restriction Hits: {sc.restriction_site_count} | Homopolymers: {sc.homopolymer_count} | Min Div: {c.min_hamming_to_higher_ranked * 100:4.1f}%"
        )
    print(f"Mean Pairwise Nucleotide Divergence: {design_res.mean_pairwise_hamming * 100:.1f}% | Elapsed: {design_res.wall_time_ms:.1f} ms\n")

    print("=" * 84)
    print("2. SYNTHETIC RETROSPECTIVE CORRELATION DEMO (N=48 Synonymous Variants)")
    print("=" * 84)
    # Generate 48 synonymous variants spanning high-CAI/folded-5', balanced Potts, and random encodings
    synthetic_rows: List[Tuple[str, str, float]] = []
    yeast = HOST_PROFILES["s_cerevisiae"]
    for i in range(48):
        codons: List[str] = []
        for pos, aa in enumerate(protein):
            choices = list(yeast.codon_frequencies[aa].keys())
            if i < 16:
                # High-CAI biased but prone to 5' GC stems
                c = choices[0] if rng.random() < 0.80 else choices[int(rng.integers(0, len(choices)))]
            elif i < 32:
                # Random uniform synonymous choices (lower CAI, occasional rare clusters)
                c = choices[int(rng.integers(0, len(choices)))]
            else:
                # Frequency-weighted choices
                probs = np.array(list(yeast.codon_frequencies[aa].values()), dtype=np.float64)
                probs /= np.sum(probs)
                c = choices[int(rng.choice(len(choices), p=probs))]
            codons.append(c)

        # Inject a strong 5' hairpin in half of the high-CAI constructs (classic Kudla/Cambray effect:
        # high CAI alone fails when 5' secondary structure occludes ribosome initiation)
        if i < 10:
            # Synonymous GC-rich 5' codons that base-pair across codons 2..12
            for pos in range(1, 12):
                aa = protein[pos]
                gc_sorted = sorted(
                    yeast.codon_frequencies[aa].keys(),
                    key=lambda cd: sum(1 for b in cd if b in ("G", "C")),
                    reverse=(pos < 6),
                )
                codons[pos] = gc_sorted[0]

        dna = "".join(codons)
        sc = score_construct(dna, host=yeast, enzymes=DEFAULT_FOUNDRY_ENZYMES)
        # Simulated ground-truth log-expression driven by 5' openness + codon adaptation - internal hairpins + noise
        true_expr = (
            2.5 * sc.cai
            + 0.35 * sc.dg_5prime_min_kcal
            - 0.15 * sc.strong_body_hairpin_count
            - 1.8 * sc.rare_codon_fraction
            - 4.0 * ((sc.gc_global - yeast.target_gc) ** 2)
            + float(rng.normal(0.0, 0.18))
        )
        synthetic_rows.append((f"syn_var_{i + 1:02d}", dna, round(float(np.exp(true_expr)), 4)))

    report, _ = evaluate_construct_dataset(synthetic_rows, host="s_cerevisiae", enzymes=DEFAULT_FOUNDRY_ENZYMES)
    print(report.to_markdown())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline Potts Codon Benchmark & Ensemble Designer (Zero Network Calls)"
    )
    sub = parser.add_subparsers(dest="command", required=False)

    # 1. correlate
    p_corr = sub.add_parser("correlate", help="Run offline correlation analysis on a local CSV/TSV of constructs")
    p_corr.add_argument("--csv", required=True, help="Path to local CSV or TSV file")
    p_corr.add_argument("--seq-col", default=None, help="DNA sequence column name (auto-detected if omitted)")
    p_corr.add_argument("--expr-col", default=None, help="Expression metric column name (auto-detected if omitted)")
    p_corr.add_argument("--id-col", default=None, help="Construct ID column name (optional)")
    p_corr.add_argument("--host", default="s_cerevisiae", help="Host profile: s_cerevisiae, e_coli, d_discoideum, b_subtilis, c_griseus, h_sapiens")
    p_corr.add_argument("--enzymes", default="none", help="Optional comma-separated restriction enzymes to check (default: none)")
    p_corr.add_argument("--redesign-bottom", type=int, default=0, help="Locally redesign the N lowest-expressing constructs")
    p_corr.add_argument("--summary-out", default=None, help="Write privacy-safe aggregate JSON report (no sequences/IDs)")
    p_corr.add_argument("--local-detail-out", default=None, help="Write per-construct diagnostic CSV (stays local)")

    # 2. design
    p_des = sub.add_parser("design", help="Sample K diverse, constraint-clean synonymous sequences for a protein or CDS")
    p_des.add_argument("--seq", default=None, help="Amino acid or coding DNA sequence string")
    p_des.add_argument("--fasta", default=None, help="Path to input FASTA file")
    p_des.add_argument("--host", default="s_cerevisiae", help="Host profile: s_cerevisiae, e_coli, d_discoideum, b_subtilis, c_griseus, h_sapiens")
    p_des.add_argument("--enzymes", default="none", help="Optional comma-separated restriction enzymes to exclude (default: none)")
    p_des.add_argument("--candidates", "-k", type=int, default=5, help="Number of diverse ensemble candidates to return")
    p_des.add_argument("--steps", type=int, default=None, help="MCMC steps per chain")
    p_des.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    p_des.add_argument("--fasta-out", default=None, help="Optional path to export candidate FASTA")

    # 3. demo
    sub.add_parser("demo", help="Run built-in S. cerevisiae ensemble design and N=48 correlation benchmark")

    args = parser.parse_args()
    if args.command == "correlate":
        cmd_correlate(args)
    elif args.command == "design":
        cmd_design(args)
    else:
        cmd_demo(args)


if __name__ == "__main__":
    main()
