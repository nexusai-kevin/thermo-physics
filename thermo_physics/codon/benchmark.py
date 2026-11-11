"""
thermo_physics.codon.benchmark
==============================
Offline, privacy-preserving retrospective benchmark for biofoundries.

Evaluates a local CSV/TSV of past constructs (`dna_sequence`, `expression_value`)
completely offline and produces two strictly separated outputs:
  1. AggregateCorrelationReport: Contains ONLY sample counts, Spearman rho, Pearson r,
     two-sided p-values, and binary ROC-AUC across energy terms (ZERO sequences, ZERO
     construct IDs, safe to inspect and share).
  2. LocalConstructDiagnostic (optional, stays strictly on the user's machine):
     Per-construct energy breakdown and optional Potts ensemble redesigns.
"""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from thermo_physics.codon.sampler import PottsCodonSampler
from thermo_physics.codon.scoring import (
    ConstructScore,
    ScoringWeights,
    score_construct,
)
from thermo_physics.codon.tables import (
    DEFAULT_FOUNDRY_ENZYMES,
    get_host_profile,
)


def _rankdata(values: Sequence[float]) -> np.ndarray:
    """Compute average-tie ranks (1-based) in pure NumPy without scipy."""
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def pearson_r(x: Sequence[float], y: Sequence[float]) -> float:
    """Compute Pearson correlation coefficient r."""
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if len(xa) < 2:
        return 0.0
    xm = xa - np.mean(xa)
    ym = ya - np.mean(ya)
    denom = float(np.sqrt(np.sum(xm * xm) * np.sum(ym * ym)))
    if denom <= 1e-12:
        return 0.0
    return float(np.clip(np.sum(xm * ym) / denom, -1.0, 1.0))


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    """Compute Spearman rank correlation rho and approximate two-sided p-value."""
    n = len(x)
    if n < 3:
        return 0.0, 1.0
    rx = _rankdata(x)
    ry = _rankdata(y)
    rho = pearson_r(rx, ry)
    if abs(rho) >= 1.0 - 1e-12:
        return float(np.sign(rho)), 0.0
    # Student-t statistic with df = n - 2; normal/Cornish-Fisher tail approximation
    df = n - 2
    t_stat = abs(rho) * math.sqrt(df / max(1e-12, 1.0 - rho * rho))
    # Accurate two-sided p-value via error function on transformed t-statistic (Wallace approximation)
    z = math.sqrt(df * math.log(1.0 + (t_stat * t_stat) / df)) * (1.0 - 1.0 / (4.0 * df))
    p_val = math.erfc(z / math.sqrt(2.0))
    return float(rho), float(min(1.0, max(0.0, p_val)))


def binary_roc_auc(scores: Sequence[float], labels_binary: Sequence[int]) -> float:
    """Compute exact Mann-Whitney U / Wilcoxon rank-sum ROC-AUC."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels_binary, dtype=np.int32)
    pos_mask = y == 1
    neg_mask = y == 0
    n_pos = int(np.sum(pos_mask))
    n_neg = int(np.sum(neg_mask))
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = _rankdata(s)
    sum_pos_ranks = float(np.sum(ranks[pos_mask]))
    u_stat = sum_pos_ranks - n_pos * (n_pos + 1) / 2.0
    return float(u_stat / (n_pos * n_neg))


@dataclass
class FeatureCorrelation:
    feature_name: str
    description: str
    spearman_rho: float
    spearman_p_value: float
    pearson_r: float
    roc_auc_high_expression: float
    top_quartile_mean_expr: float
    bottom_quartile_mean_expr: float


@dataclass
class AggregateCorrelationReport:
    """Privacy-safe summary containing ZERO construct sequences or identifiers."""

    host: str
    enzymes_checked: List[str]
    n_constructs_evaluated: int
    n_constructs_skipped: int
    expression_min: float
    expression_median: float
    expression_max: float
    fraction_with_restriction_sites: float
    fraction_with_homopolymers: float
    fraction_with_5prime_hairpin: float
    feature_correlations: List[FeatureCorrelation]
    potts_delta_spearman_vs_cai: float
    potts_delta_auc_vs_cai: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [
            f"# Offline Retrospective Codon Benchmark Summary (Host: `{self.host}`)",
            "",
            f"- **Constructs Evaluated:** `{self.n_constructs_evaluated}` (Skipped invalid/out-of-frame: `{self.n_constructs_skipped}`)",
            f"- **Enzymes Checked:** `{', '.join(self.enzymes_checked) or 'None'}`",
            f"- **Expression Range:** min `{self.expression_min:.3f}` | median `{self.expression_median:.3f}` | max `{self.expression_max:.3f}`",
            f"- **Constraint Incidence in Dataset:** Restriction hits `{self.fraction_with_restriction_sites * 100:.1f}%` | Homopolymers `{self.fraction_with_homopolymers * 100:.1f}%` | Strong 5' Hairpins (dG < -3 kcal/mol) `{self.fraction_with_5prime_hairpin * 100:.1f}%`",
            f"- **Potts Composite vs. Standard CAI Alone:** `Delta Spearman rho = {self.potts_delta_spearman_vs_cai:+.3f}` | `Delta ROC-AUC = {self.potts_delta_auc_vs_cai:+.3f}`",
            "",
            "| Predictor / Biophysical Term (Higher = Better Predicted Expression) | Spearman rho | p-value | Pearson r | High-Expr ROC-AUC | Top-25% Expr | Bottom-25% Expr |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for fc in self.feature_correlations:
            lines.append(
                f"| **{fc.feature_name}** ({fc.description}) | `{fc.spearman_rho:+.3f}` | `{fc.spearman_p_value:.2e}` | `{fc.pearson_r:+.3f}` | `{fc.roc_auc_high_expression:.3f}` | `{fc.top_quartile_mean_expr:.3f}` | `{fc.bottom_quartile_mean_expr:.3f}` |"
            )
        lines.append("")
        lines.append("*Privacy notice: This report contains aggregate correlation statistics only. Zero sequences or construct IDs are included.*")
        return "\n".join(lines)


@dataclass
class LocalConstructRow:
    """Per-construct diagnostic kept strictly on the user's local machine."""

    construct_id: str
    expression: float
    score: ConstructScore
    redesigned_energy: Optional[float] = None
    redesigned_cai: Optional[float] = None
    redesigned_dg_5prime: Optional[float] = None
    redesigned_dna: Optional[str] = None


def evaluate_construct_dataset(
    rows: List[Tuple[str, str, float]],
    host: str = "s_cerevisiae",
    enzymes: Tuple[str, ...] = DEFAULT_FOUNDRY_ENZYMES,
    weights: Optional[ScoringWeights] = None,
    redesign_bottom_n: int = 0,
) -> Tuple[AggregateCorrelationReport, List[LocalConstructRow]]:
    """Score a list of (construct_id, dna_seq, expression_value) tuples and compute correlations."""
    profile = get_host_profile(host)
    w = weights or ScoringWeights()

    local_rows: List[LocalConstructRow] = []
    skipped = 0

    for cid, dna, expr in rows:
        try:
            sc = score_construct(dna, profile, enzymes=enzymes, weights=w)
            local_rows.append(LocalConstructRow(construct_id=cid, expression=float(expr), score=sc))
        except Exception:
            skipped += 1

    if len(local_rows) < 3:
        raise ValueError(f"Need at least 3 valid in-frame constructs to compute correlations (got {len(local_rows)})")

    expr_arr = np.array([r.expression for r in local_rows], dtype=np.float64)
    med_expr = float(np.median(expr_arr))
    high_labels = (expr_arr > med_expr).astype(np.int32)
    if np.all(high_labels == 0) or np.all(high_labels == 1):
        high_labels = (expr_arr >= med_expr).astype(np.int32)

    # Orient predictors so higher value = better predicted expression
    predictors: List[Tuple[str, str, np.ndarray]] = [
        (
            "Potts Composite (-H_expr)",
            "Combined CAI + 5' initiation + hairpin + GC + rare-doublet score",
            np.array(
                [
                    -(
                        r.score.e_cai
                        + r.score.e_rare_doublet
                        + r.score.e_5prime_struct
                        + r.score.e_body_hairpin
                        + r.score.e_gc_global
                        + r.score.e_gc_window
                    )
                    for r in local_rows
                ],
                dtype=np.float64,
            ),
        ),
        (
            "Potts Total (-H_total)",
            "Full Hamiltonian including assembly/homopolymer penalties",
            np.array([-r.score.total_energy for r in local_rows], dtype=np.float64),
        ),
        (
            "Standard CAI",
            "Geometric mean Codon Adaptation Index alone",
            np.array([r.score.cai for r in local_rows], dtype=np.float64),
        ),
        (
            "5' Initiation Openness",
            "5' folding free energy dG_5' (kcal/mol, less negative = more open)",
            np.array([r.score.dg_5prime_min_kcal for r in local_rows], dtype=np.float64),
        ),
        (
            "Body Stem-Loop Clearance",
            "Negative internal stem-loop elongation penalty (-E_hairpin)",
            np.array([-r.score.e_body_hairpin for r in local_rows], dtype=np.float64),
        ),
        (
            "GC Window Homeostasis",
            "Negative global + sliding-window GC penalty (-E_gc)",
            np.array([-(r.score.e_gc_global + r.score.e_gc_window) for r in local_rows], dtype=np.float64),
        ),
        (
            "Rare-Codon Avoidance",
            "1 - rare codon fraction (w >= 0.30)",
            np.array([1.0 - r.score.rare_codon_fraction for r in local_rows], dtype=np.float64),
        ),
    ]

    n = len(local_rows)
    q_size = max(1, n // 4)
    feature_corrs: List[FeatureCorrelation] = []
    rho_by_name: Dict[str, float] = {}
    auc_by_name: Dict[str, float] = {}

    for name, desc, pred_arr in predictors:
        rho, p_val = spearman_rho(pred_arr, expr_arr)
        pr = pearson_r(pred_arr, expr_arr)
        auc = binary_roc_auc(pred_arr, high_labels)
        order = np.argsort(pred_arr)
        bot_q = float(np.mean(expr_arr[order[:q_size]]))
        top_q = float(np.mean(expr_arr[order[-q_size:]]))
        rho_by_name[name] = rho
        auc_by_name[name] = auc
        feature_corrs.append(
            FeatureCorrelation(
                feature_name=name,
                description=desc,
                spearman_rho=round(rho, 4),
                spearman_p_value=float(f"{p_val:.4e}"),
                pearson_r=round(pr, 4),
                roc_auc_high_expression=round(auc, 4),
                top_quartile_mean_expr=round(top_q, 4),
                bottom_quartile_mean_expr=round(bot_q, 4),
            )
        )

    # Optional local redesign of the lowest-expressing constructs
    if redesign_bottom_n > 0:
        sampler = PottsCodonSampler(host=profile, enzymes=enzymes, weights=w)
        # Sort indices by lowest expression
        worst_indices = np.argsort(expr_arr)[: min(redesign_bottom_n, len(local_rows))]
        # Map original DNA from input rows
        dna_by_id = {cid: dna for cid, dna, _ in rows}
        for idx in worst_indices:
            row = local_rows[int(idx)]
            orig_dna = dna_by_id.get(row.construct_id)
            if orig_dna:
                res = sampler.sample_ensemble(orig_dna, k_candidates=1, steps_per_chain=1500, n_chains=2)
                best = res.candidates[0]
                row.redesigned_energy = best.score.total_energy
                row.redesigned_cai = best.score.cai
                row.redesigned_dg_5prime = best.score.dg_5prime_min_kcal
                row.redesigned_dna = best.dna_sequence

    frac_rest = sum(1 for r in local_rows if r.score.restriction_site_count > 0) / n
    frac_hp = sum(1 for r in local_rows if r.score.homopolymer_count > 0) / n
    frac_5p = sum(1 for r in local_rows if r.score.dg_5prime_min_kcal < -3.0) / n

    report = AggregateCorrelationReport(
        host=profile.name,
        enzymes_checked=list(enzymes),
        n_constructs_evaluated=n,
        n_constructs_skipped=skipped,
        expression_min=round(float(np.min(expr_arr)), 4),
        expression_median=round(med_expr, 4),
        expression_max=round(float(np.max(expr_arr)), 4),
        fraction_with_restriction_sites=round(frac_rest, 4),
        fraction_with_homopolymers=round(frac_hp, 4),
        fraction_with_5prime_hairpin=round(frac_5p, 4),
        feature_correlations=feature_corrs,
        potts_delta_spearman_vs_cai=round(
            rho_by_name["Potts Composite (-H_expr)"] - rho_by_name["Standard CAI"], 4
        ),
        potts_delta_auc_vs_cai=round(
            auc_by_name["Potts Composite (-H_expr)"] - auc_by_name["Standard CAI"], 4
        ),
    )
    return report, local_rows


def load_construct_csv(
    path: str | Path,
    seq_col: Optional[str] = None,
    expr_col: Optional[str] = None,
    id_col: Optional[str] = None,
) -> List[Tuple[str, str, float]]:
    """Load (construct_id, dna_seq, expression) rows from a local CSV or TSV file."""
    p = Path(path)
    delimiter = "\t" if p.suffix.lower() in (".tsv", ".tab") else ","
    with open(p, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError(f"Empty or headerless file: {p}")

        fields_lower = {name.strip().lower(): name for name in reader.fieldnames}

        def pick_col(explicit: Optional[str], candidates: Tuple[str, ...], label: str) -> str:
            if explicit:
                if explicit in reader.fieldnames:
                    return explicit
                if explicit.lower() in fields_lower:
                    return fields_lower[explicit.lower()]
                raise ValueError(f"Column {explicit!r} not found in {reader.fieldnames}")
            for cand in candidates:
                if cand in fields_lower:
                    return fields_lower[cand]
            raise ValueError(
                f"Could not auto-detect {label} column in {reader.fieldnames}. "
                f"Pass explicit column name via CLI."
            )

        s_col = pick_col(
            seq_col,
            ("dna_seq", "dna_sequence", "sequence", "seq", "cds", "mrna", "construct_seq"),
            "DNA sequence",
        )
        e_col = pick_col(
            expr_col,
            ("expression", "expr", "yield", "titer", "fluorescence", "gfp", "protein_level", "score"),
            "expression metric",
        )
        i_col = None
        if id_col and id_col in reader.fieldnames:
            i_col = id_col
        else:
            for cand in ("construct_id", "id", "name", "gene_id", "sample_id"):
                if cand in fields_lower:
                    i_col = fields_lower[cand]
                    break

        rows: List[Tuple[str, str, float]] = []
        for idx, row in enumerate(reader):
            raw_seq = (row.get(s_col) or "").strip()
            raw_expr = (row.get(e_col) or "").strip()
            if not raw_seq or not raw_expr:
                continue
            try:
                expr_val = float(raw_expr)
            except ValueError:
                continue
            cid = (row.get(i_col) or f"construct_{idx + 1:04d}").strip() if i_col else f"construct_{idx + 1:04d}"
            rows.append((cid, raw_seq, expr_val))
        return rows
