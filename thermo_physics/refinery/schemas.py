"""
Pydantic v2 data-contract schemas for the 5-phase Live-Stream Data Refinery.

Validates dimensions, finiteness, normalization, and provenance metadata across
all four empirical streams before productizing `live_empirical_streams.json`.
"""

from __future__ import annotations

import math
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

SourceMode = Literal["live_public_stream", "cached_empirical", "seeded_fallback"]


def _assert_finite_matrix(mat: List[List[float]], rows: int, cols: int, name: str) -> None:
    if len(mat) != rows:
        raise ValueError(f"{name}: expected {rows} rows, got {len(mat)}")
    for r_idx, row in enumerate(mat):
        if len(row) != cols:
            raise ValueError(f"{name}[{r_idx}]: expected {cols} cols, got {len(row)}")
        for val in row:
            if not math.isfinite(val):
                raise ValueError(f"{name}[{r_idx}] contains non-finite value: {val}")


def _assert_finite_vector(vec: List[float], length: int, name: str) -> None:
    if len(vec) != length:
        raise ValueError(f"{name}: expected length {length}, got {len(vec)}")
    for val in vec:
        if not math.isfinite(val):
            raise ValueError(f"{name} contains non-finite value: {val}")


class EcologicalProvenance(BaseModel):
    source_label: str
    source_mode: SourceMode
    captured_at_utc: str
    station_count: int = Field(..., ge=1)
    channel_count: int = Field(..., ge=1)


class EcologicalNode(BaseModel):
    index: int
    channel_id: str
    site_code: str
    site_name: str
    parameter_code: str
    parameter_name: str
    unit: str
    latest_value: float
    mean_value: float
    std_value: float
    z_score_latest: float
    spin_latest: int
    binary_latest: int
    valid_points: int
    source_mode: SourceMode


class IsingSamplerBenchmark(BaseModel):
    min_energy: float
    mean_energy: float
    tau_int: float = Field(..., gt=0.0)
    ess: float = Field(..., gt=0.0)
    energy_trace: List[float]
    beta: float = 1.5
    n_samples: int = 320
    abs_magnetization: float = 0.0
    exact_match: bool = False


class EcologicalIsingPayload(BaseModel):
    provenance: EcologicalProvenance
    nodes: List[EcologicalNode]
    J: List[List[float]]
    h: List[float]
    spin_covariance: List[List[float]]
    magnetization: float
    frustration_pct: float = Field(..., ge=0.0, le=100.0)
    frustrated_triangles_pct: float = Field(..., ge=0.0, le=100.0)
    exact_ground_state_energy: float
    exact_ground_state_spins: Optional[List[int]] = None
    sampler_benchmark: IsingSamplerBenchmark

    @model_validator(mode="after")
    def _validate_ising_tensors(self) -> "EcologicalIsingPayload":
        n = self.provenance.channel_count
        if len(self.nodes) != n:
            raise ValueError(f"ecological_ising.nodes must have {n} entries, got {len(self.nodes)}")
        _assert_finite_matrix(self.J, n, n, "ecological_ising.J")
        _assert_finite_matrix(self.spin_covariance, n, n, "ecological_ising.spin_covariance")
        _assert_finite_vector(self.h, n, "ecological_ising.h")
        for i in range(n):
            if abs(self.J[i][i]) > 1e-6:
                raise ValueError(f"ecological_ising.J[{i}][{i}] must be zero, got {self.J[i][i]}")
            for j in range(n):
                if abs(self.J[i][j]) > 1.0005:
                    raise ValueError(f"ecological_ising.J[{i}][{j}] exceeds |J|<=1.0: {self.J[i][j]}")
        return self


class RnaProvenance(BaseModel):
    source_label: str
    source_mode: SourceMode
    captured_at_utc: str
    family_id: str
    raw_sequences: int = Field(..., ge=1)
    retained_sequences: int = Field(..., ge=1)
    consensus_length: int = Field(..., ge=1)
    q_states: int = Field(..., ge=2)


class RnaContact(BaseModel):
    i: int
    j: int
    pos_i: int
    pos_j: int
    frobenius_score: float
    consensus_pair: str


class RnaSamplerBenchmark(BaseModel):
    min_energy: float
    mean_energy: float
    entropy_bits: float
    tau_int: float = Field(..., gt=0.0)
    ess: float = Field(..., gt=0.0)
    energy_trace: List[float]
    beta: float = 1.5
    n_samples: int = 320


class RnaSecondarySummary(BaseModel):
    family_id: str
    source_mode: SourceMode
    raw_sequences: int
    retained_sequences: int
    consensus_length: int
    top_contacts: List[RnaContact]


class RnaPottsPayload(BaseModel):
    provenance: RnaProvenance
    alphabet: List[str]
    f_i: List[List[float]]
    h_i: List[List[float]]
    frobenius_couplings: List[List[float]]
    C_ij: List[List[List[List[float]]]]
    top_contacts: List[RnaContact]
    consensus_columns: List[int]
    secondary_family: Optional[RnaSecondarySummary] = None
    sampler_benchmark: RnaSamplerBenchmark

    @model_validator(mode="after")
    def _validate_potts_tensors(self) -> "RnaPottsPayload":
        l_len = self.provenance.consensus_length
        q = self.provenance.q_states
        if len(self.alphabet) != q:
            raise ValueError(f"rna_potts.alphabet must have length {q}")
        _assert_finite_matrix(self.f_i, l_len, q, "rna_potts.f_i")
        _assert_finite_matrix(self.h_i, l_len, q, "rna_potts.h_i")
        _assert_finite_matrix(self.frobenius_couplings, l_len, l_len, "rna_potts.frobenius_couplings")
        for idx, row in enumerate(self.f_i):
            row_sum = sum(row)
            if abs(row_sum - 1.0) > 1e-2:
                raise ValueError(f"rna_potts.f_i[{idx}] must sum to 1.0, got {row_sum}")
        if len(self.C_ij) != l_len:
            raise ValueError(f"rna_potts.C_ij expected {l_len} outer rows, got {len(self.C_ij)}")
        for i in range(l_len):
            if len(self.C_ij[i]) != l_len:
                raise ValueError(f"rna_potts.C_ij[{i}] expected {l_len} cols")
            for j in range(l_len):
                _assert_finite_matrix(self.C_ij[i][j], q, q, f"rna_potts.C_ij[{i}][{j}]")
        return self


class MarketProvenance(BaseModel):
    source_label: str
    source_mode: SourceMode
    captured_at_utc: str
    n_assets: int = Field(..., ge=2)
    k_target: int = Field(..., ge=1)
    polymarket_markets_polled: int = Field(..., ge=0)


class MarketAssetEntry(BaseModel):
    index: int
    ticker: str
    latest_price: float
    annualized_return: float
    annualized_vol: float
    polymarket_market: str
    obi: float = Field(..., ge=-1.0, le=1.0)
    spread: float = Field(..., ge=0.0)
    obi_penalty: float
    bid_depth: float = 0.0
    ask_depth: float = 0.0
    midpoint: float = 0.5


class MarketExactOptimum(BaseModel):
    selected_indices: List[int]
    selected_tickers: List[str]
    qubo_energy: float
    expected_return: float
    volatility: float


class MarketSamplerBenchmark(BaseModel):
    selected_indices: List[int]
    selected_tickers: List[str]
    qubo_energy: float
    budget_feasibility_pct: float = Field(..., ge=0.0, le=100.0)
    pareto_match_pct: float = Field(..., ge=0.0, le=100.0)
    tau_int: float = Field(..., gt=0.0)
    ess: float = Field(..., gt=0.0)
    energy_trace: List[float]


class MarketQuboPayload(BaseModel):
    provenance: MarketProvenance
    assets: List[MarketAssetEntry]
    Q: List[List[float]]
    J: List[List[float]]
    h: List[float]
    qubo_ising_offset: float
    exact_optimum: MarketExactOptimum
    sampler_benchmark: MarketSamplerBenchmark

    @model_validator(mode="after")
    def _validate_market_tensors(self) -> "MarketQuboPayload":
        n = self.provenance.n_assets
        if len(self.assets) != n:
            raise ValueError(f"market_qubo.assets must have {n} entries, got {len(self.assets)}")
        _assert_finite_matrix(self.Q, n, n, "market_qubo.Q")
        _assert_finite_matrix(self.J, n, n, "market_qubo.J")
        _assert_finite_vector(self.h, n, "market_qubo.h")
        for i in range(n):
            if abs(self.J[i][i]) > 1e-6:
                raise ValueError(f"market_qubo.J[{i}][{i}] must be zero")
        if len(self.exact_optimum.selected_indices) != self.provenance.k_target:
            raise ValueError("market_qubo.exact_optimum must match k_target")
        return self


class AffilProvenance(BaseModel):
    source_label: str
    source_mode: SourceMode
    captured_at_utc: str
    brand_count: int = Field(..., ge=1)
    total_products_polled: int = Field(..., ge=0)
    total_variants_polled: int = Field(..., ge=0)


class AffilBrandEntry(BaseModel):
    index: int
    brand: str
    vertical: str
    domain: str
    source_mode: SourceMode
    product_count: int
    variant_count: int
    mean_price_usd: float
    price_spread_ratio: float
    price_elasticity: float = Field(..., ge=0.0, le=1.0)
    variant_depth: float = Field(..., ge=0.0, le=1.0)
    inventory_velocity: float = Field(..., ge=0.0, le=1.0)
    tag_taxonomy_richness: float = Field(..., ge=0.0, le=1.0)
    composite_deal_score: float = Field(..., ge=0.0, le=100.0)
    recommended_commission_tier: str
    top_tags: List[str]


class AffilProductGraphPayload(BaseModel):
    provenance: AffilProvenance
    brands: List[AffilBrandEntry]
    brand_affinity_J: List[List[float]]

    @model_validator(mode="after")
    def _validate_affil_tensors(self) -> "AffilProductGraphPayload":
        n = self.provenance.brand_count
        if len(self.brands) != n:
            raise ValueError(f"affilai_product_graph.brands must have {n} entries")
        _assert_finite_matrix(self.brand_affinity_J, n, n, "affilai_product_graph.brand_affinity_J")
        for i in range(n):
            if abs(self.brand_affinity_J[i][i]) > 1e-6:
                raise ValueError(f"affilai_product_graph.brand_affinity_J[{i}][{i}] must be zero")
        return self


class LiveEmpiricalStreamsManifest(BaseModel):
    schema_version: str = "1.0.0"
    generated_at_utc: str
    doctrine_disclosure: str
    ecological_ising: EcologicalIsingPayload
    rna_potts: RnaPottsPayload
    market_qubo: MarketQuboPayload
    affilai_product_graph: AffilProductGraphPayload
