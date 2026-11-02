"""
thermo_physics.tpu.hamiltonian
==============================
Non-Convex Coupled Continuous-Discrete Energy Hamiltonian for Hybrid Thermodynamic Processing Units.
Combines:
1. Multi-well bistable potential confining continuous states x_i toward binary attractors {-1, +1}.
2. Fully coupled Ising / Spin-Glass interaction matrix J_ij and external field h_i.
3. Continuous differentiable relaxation of constrained 3-SAT clause violation penalties:
   - "tanh": Saturating sigmoidal relaxation (exhibits vanishing gradients in deep false wells)
   - "softplus": Non-saturating smooth relaxation (maintains persistent restorative gradients)
4. Dynamic Non-Conservative Clause Multipliers: y_m(t) warping local landscapes to break topological frustration.
5. Soft confining boundary walls for analog voltage range stability.
6. Analytical exact diagonal Hessian and Laplacian Delta H(x) = sum_i d^2 H / dx_i^2 for configurational temperature audits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np


@dataclass
class ProblemConfig:
    """Configuration for NonConvexIsingSAT Hamiltonian."""
    num_variables: int
    alpha_well: float = 1.0           # Depth of double-well potential (x^2 - 1)^2
    lambda_sat: float = 2.0           # Penalty weight for violated 3-SAT clauses
    c_sat: float = 2.5                # Sharpness parameter for continuous tanh relaxation
    k_wall: float = 20.0              # Boundary stiffness
    x_bound: float = 2.2              # Boundary activation threshold
    coupling_scale: float = 1.0       # Global scaling for J matrix
    penalty_type: str = "tanh"        # "tanh" or "softplus"
    beta_softplus: float = 2.5        # Sharpness of softplus penalty


class NonConvexIsingSAT:
    """Non-convex Hamiltonian combining continuous double-wells, Ising spin glass, and 3-SAT constraints.

    Total Energy:
      H(x, t) = V_well(x, t) + H_ising(x) + H_sat(x, t) + V_boundary(x)
    where:
      V_well(x, t) = sum_i alpha(t) * (x_i^2 - 1)^2
      H_ising(x) = - 0.5 * x^T J x - h^T x
      V_boundary(x) = sum_i k_wall * max(0, |x_i| - x_bound)^4
      H_sat(x, t, y) = sum_{clause m} y_m(t) * prod_{k in clause} u_{m, k}(x)
    """

    def __init__(
        self,
        num_variables: int,
        j_matrix: Optional[np.ndarray] = None,
        h_bias: Optional[np.ndarray] = None,
        clauses: Optional[List[Tuple[int, int, int]]] = None,
        config: Optional[ProblemConfig] = None,
    ):
        self.n = num_variables
        self.config = config or ProblemConfig(num_variables=num_variables)

        # Ising interaction matrix J (symmetric, zero diagonal)
        if j_matrix is not None:
            assert j_matrix.shape == (self.n, self.n), f"J matrix shape mismatch: {j_matrix.shape}"
            self.j_matrix = np.array(j_matrix, dtype=np.float64)
            np.fill_diagonal(self.j_matrix, 0.0)
            self.j_matrix = 0.5 * (self.j_matrix + self.j_matrix.T) * self.config.coupling_scale
        else:
            self.j_matrix = np.zeros((self.n, self.n), dtype=np.float64)

        # Local magnetic fields h
        if h_bias is not None:
            assert len(h_bias) == self.n, f"h vector length mismatch: {len(h_bias)}"
            self.h_bias = np.array(h_bias, dtype=np.float64)
        else:
            self.h_bias = np.zeros(self.n, dtype=np.float64)

        # 3-SAT Clauses: list of tuples of 3 1-based signed integers, e.g. (1, -3, 5)
        self.clauses: List[Tuple[int, int, int]] = clauses or []
        self._parsed_clauses = self._precompute_clauses(self.clauses)

    @property
    def j(self) -> np.ndarray:
        return self.j_matrix

    @property
    def h(self) -> np.ndarray:
        return self.h_bias

    def _precompute_clauses(
        self, clauses: List[Tuple[int, int, int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert clauses into 0-indexed variable indices and +/-1 signs."""
        if not clauses:
            return np.zeros((0, 3), dtype=np.int32), np.zeros((0, 3), dtype=np.float64)

        var_indices = []
        signs = []
        for c in clauses:
            assert len(c) == 3, f"Clause must have 3 literals: {c}"
            vars_in_c = [abs(lit) - 1 for lit in c]
            for v in vars_in_c:
                assert 0 <= v < self.n, f"Variable index {v+1} out of bounds for N={self.n}"
            signs_in_c = [1.0 if lit > 0 else -1.0 for lit in c]
            var_indices.append(vars_in_c)
            signs.append(signs_in_c)

        return np.array(var_indices, dtype=np.int32), np.array(signs, dtype=np.float64)

    def _compute_literal_factors(
        self, x_mat: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute literal factors u, first derivative du/dx, and second derivative d2u/dx2 for all clauses.

        Returns:
          u: shape (B, M, 3)
          du_dx: shape (B, M, 3)
          d2u_dx2: shape (B, M, 3)
        """
        var_idx, signs = self._parsed_clauses  # (M, 3), (M, 3)

        if self.config.penalty_type == "softplus":
            beta_sp = self.config.beta_softplus
            x_lit = x_mat[:, var_idx]
            z = -beta_sp * signs[None, :, :] * x_lit  # (B, M, 3)

            z_pos = np.maximum(0.0, z)
            abs_z = np.abs(z)
            sp = z_pos + np.log1p(np.exp(-abs_z))
            u = sp / beta_sp

            sig = 0.5 * (1.0 + np.tanh(0.5 * z))
            du_dx = -signs[None, :, :] * sig
            d2u_dx2 = beta_sp * sig * (1.0 - sig)
            return u, du_dx, d2u_dx2
        else:
            c_sat = self.config.c_sat
            s_mat = np.tanh(c_sat * x_mat)  # (B, N)
            s_clauses = s_mat[:, var_idx]   # (B, M, 3)
            u = 0.5 * (1.0 - signs[None, :, :] * s_clauses)  # (B, M, 3)

            sech2_mat = 1.0 - s_mat**2
            sech2_clauses = sech2_mat[:, var_idx]  # (B, M, 3)
            du_dx = -0.5 * signs[None, :, :] * c_sat * sech2_clauses
            d2u_dx2 = signs[None, :, :] * (c_sat**2) * sech2_clauses * s_clauses
            return u, du_dx, d2u_dx2

    def clause_violations(self, x: np.ndarray) -> np.ndarray:
        """Compute continuous clause violation values C_m(x) in [0, inf) for all clauses.

        Returns:
          np.ndarray of shape (M,) for 1D state or (B, M) for 2D batch.
        """
        is_1d = (x.ndim == 1)
        x_mat = np.atleast_2d(x)
        if len(self.clauses) == 0:
            return np.zeros(0, dtype=np.float64) if is_1d else np.zeros((x_mat.shape[0], 0), dtype=np.float64)
        u, _, _ = self._compute_literal_factors(x_mat)
        c_vals = np.prod(u, axis=2)  # shape (B, M)
        return c_vals[0] if is_1d else c_vals

    def energy(
        self,
        x: np.ndarray,
        alpha_well: Optional[float] = None,
        lambda_sat: Optional[float] = None,
        clause_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute continuous total energy H(x) for single state (N,) or batch of states (B, N)."""
        is_1d = (x.ndim == 1)
        x_mat = np.atleast_2d(x)
        b, n = x_mat.shape
        assert n == self.n, f"Dimension mismatch: expected {self.n}, got {n}"

        alpha = self.config.alpha_well if alpha_well is None else alpha_well
        lam = self.config.lambda_sat if lambda_sat is None else lambda_sat
        k_wall = self.config.k_wall
        x_bound = self.config.x_bound

        # 1. Multi-well potential
        e_well = alpha * np.sum((x_mat**2 - 1.0) ** 2, axis=1)

        # 2. Ising spin glass
        jx = x_mat @ self.j_matrix
        e_ising = -0.5 * np.sum(x_mat * jx, axis=1) - np.sum(x_mat * self.h_bias, axis=1)

        # 3. Boundary walls
        exceed = np.maximum(0.0, np.abs(x_mat) - x_bound)
        e_boundary = k_wall * np.sum(exceed**4, axis=1)

        # 4. Continuous 3-SAT clause penalty
        e_sat = np.zeros(b, dtype=np.float64)
        if len(self.clauses) > 0:
            u, _, _ = self._compute_literal_factors(x_mat)
            clause_violations = np.prod(u, axis=2)  # shape (B, M)

            if clause_weights is not None:
                w_mat = np.atleast_2d(clause_weights) if clause_weights.ndim == 2 else clause_weights[None, :]
                e_sat = np.sum(w_mat * clause_violations, axis=1)
            elif lam > 1e-9:
                e_sat = lam * np.sum(clause_violations, axis=1)

        total_e = e_well + e_ising + e_sat + e_boundary
        return float(total_e[0]) if is_1d else total_e

    def gradient(
        self,
        x: np.ndarray,
        alpha_well: Optional[float] = None,
        lambda_sat: Optional[float] = None,
        clause_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute analytical gradient dH/dx for state (N,) or batch of states (B, N)."""
        is_1d = (x.ndim == 1)
        x_mat = np.atleast_2d(x)
        b, n = x_mat.shape
        assert n == self.n, f"Dimension mismatch: expected {self.n}, got {n}"

        alpha = self.config.alpha_well if alpha_well is None else alpha_well
        lam = self.config.lambda_sat if lambda_sat is None else lambda_sat
        k_wall = self.config.k_wall
        x_bound = self.config.x_bound

        # 1. Multi-well gradient
        grad = 4.0 * alpha * x_mat * (x_mat**2 - 1.0)

        # 2. Ising spin glass gradient
        jx = x_mat @ self.j_matrix
        grad -= (jx + self.h_bias)

        # 3. Boundary gradient
        diff = np.abs(x_mat) - x_bound
        exceed = np.maximum(0.0, diff)
        grad += 4.0 * k_wall * np.sign(x_mat) * (exceed**3)

        # 4. Continuous 3-SAT gradient
        if len(self.clauses) > 0 and (clause_weights is not None or lam > 1e-9):
            var_idx, _ = self._parsed_clauses
            u, du_dx, _ = self._compute_literal_factors(x_mat)

            prod_0 = u[:, :, 1] * u[:, :, 2]
            prod_1 = u[:, :, 0] * u[:, :, 2]
            prod_2 = u[:, :, 0] * u[:, :, 1]

            if clause_weights is not None:
                w_mat = np.atleast_2d(clause_weights) if clause_weights.ndim == 2 else clause_weights[None, :]
            else:
                w_mat = lam

            grad_sat = np.zeros_like(x_mat)

            for k, (prod_other, v_k) in enumerate([
                (prod_0, var_idx[:, 0]),
                (prod_1, var_idx[:, 1]),
                (prod_2, var_idx[:, 2]),
            ]):
                d_clause = w_mat * prod_other * du_dx[:, :, k]  # (B, M)
                np.add.at(grad_sat, (slice(None), v_k), d_clause)

            grad += grad_sat

        return grad[0] if is_1d else grad

    def diagonal_hessian(
        self,
        x: np.ndarray,
        alpha_well: Optional[float] = None,
        lambda_sat: Optional[float] = None,
        clause_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute exact diagonal elements of Hessian d^2 H / dx_i^2, shape (N,) or (B, N)."""
        is_1d = (x.ndim == 1)
        x_mat = np.atleast_2d(x)
        b, n = x_mat.shape

        alpha = self.config.alpha_well if alpha_well is None else alpha_well
        lam = self.config.lambda_sat if lambda_sat is None else lambda_sat
        k_wall = self.config.k_wall
        x_bound = self.config.x_bound

        diag_h = 4.0 * alpha * (3.0 * (x_mat**2) - 1.0)

        diff = np.abs(x_mat) - x_bound
        exceed = np.maximum(0.0, diff)
        diag_h += 12.0 * k_wall * (exceed**2)

        if len(self.clauses) > 0 and (clause_weights is not None or lam > 1e-9):
            var_idx, _ = self._parsed_clauses
            u, _, d2u_dx2 = self._compute_literal_factors(x_mat)

            prod_0 = u[:, :, 1] * u[:, :, 2]
            prod_1 = u[:, :, 0] * u[:, :, 2]
            prod_2 = u[:, :, 0] * u[:, :, 1]

            if clause_weights is not None:
                w_mat = np.atleast_2d(clause_weights) if clause_weights.ndim == 2 else clause_weights[None, :]
            else:
                w_mat = lam

            diag_sat = np.zeros_like(x_mat)

            for k, (prod_other, v_k) in enumerate([
                (prod_0, var_idx[:, 0]),
                (prod_1, var_idx[:, 1]),
                (prod_2, var_idx[:, 2]),
            ]):
                d2_clause = w_mat * prod_other * d2u_dx2[:, :, k]  # (B, M)
                np.add.at(diag_sat, (slice(None), v_k), d2_clause)

            diag_h += diag_sat

        return diag_h[0] if is_1d else diag_h

    def laplacian(
        self,
        x: np.ndarray,
        alpha_well: Optional[float] = None,
        lambda_sat: Optional[float] = None,
        clause_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute scalar Laplacian Delta H(x) = sum_i d^2 H / dx_i^2, shape scalar or (B,)."""
        diag_h = self.diagonal_hessian(x, alpha_well, lambda_sat, clause_weights)
        if x.ndim == 1:
            return float(np.sum(diag_h))
        return np.sum(diag_h, axis=1)

    def evaluate_discrete(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project continuous states x to discrete bipolar spins s = sign(x) in {-1, +1}^N."""
        is_1d = (x.ndim == 1)
        x_mat = np.atleast_2d(x)
        b, n = x_mat.shape

        spins = np.where(x_mat >= 0.0, 1.0, -1.0)
        js = spins @ self.j_matrix
        e_ising = -0.5 * np.sum(spins * js, axis=1) - np.sum(spins * self.h_bias, axis=1)

        num_violated = np.zeros(b, dtype=np.int32)
        if len(self.clauses) > 0:
            var_idx, signs = self._parsed_clauses
            s_clauses = spins[:, var_idx]  # (B, M, 3)
            lit_satisfied = (signs[None, :, :] * s_clauses) > 0  # (B, M, 3)
            clause_satisfied = np.any(lit_satisfied, axis=2)    # (B, M)
            num_violated = np.sum(~clause_satisfied, axis=1)    # (B,)

        discrete_energy = e_ising + self.config.lambda_sat * num_violated

        if is_1d:
            return spins[0], float(discrete_energy[0]), int(num_violated[0])
        return spins, discrete_energy, num_violated


def generate_random_3sat_spin_glass(
    num_variables: int,
    clause_ratio: float = 4.26,
    spin_glass_std: float = 0.5,
    penalty_type: str = "tanh",
    seed: Optional[int] = None,
) -> NonConvexIsingSAT:
    """Generate a challenging benchmark problem combining random 3-SAT at phase transition with an SK spin glass."""
    rng = np.random.default_rng(seed)
    n = num_variables
    m = int(round(clause_ratio * n))

    clauses: List[Tuple[int, int, int]] = []
    for _ in range(m):
        vars_chosen = rng.choice(n, size=3, replace=False) + 1
        polarities = rng.choice([-1, 1], size=3)
        clauses.append((
            int(polarities[0] * vars_chosen[0]),
            int(polarities[1] * vars_chosen[1]),
            int(polarities[2] * vars_chosen[2]),
        ))

    j_raw = rng.normal(0.0, spin_glass_std / np.sqrt(n), size=(n, n))
    j_mat = 0.5 * (j_raw + j_raw.T)
    np.fill_diagonal(j_mat, 0.0)

    h_bias = rng.normal(0.0, 0.2, size=n)

    config = ProblemConfig(
        num_variables=n,
        alpha_well=1.0,
        lambda_sat=4.0,
        c_sat=2.5,
        k_wall=20.0,
        x_bound=2.2,
        penalty_type=penalty_type,
    )
    return NonConvexIsingSAT(n, j_matrix=j_mat, h_bias=h_bias, clauses=clauses, config=config)


def generate_sk_spin_glass(
    num_variables: int,
    j_std: float = 1.0,
    seed: Optional[int] = None,
) -> NonConvexIsingSAT:
    """Generate a pure Sherrington-Kirkpatrick (SK) infinite-range spin-glass problem."""
    rng = np.random.default_rng(seed)
    n = num_variables
    j_raw = rng.normal(0.0, j_std / np.sqrt(n), size=(n, n))
    j_mat = 0.5 * (j_raw + j_raw.T)
    np.fill_diagonal(j_mat, 0.0)
    h_bias = rng.normal(0.0, 0.1, size=n)

    config = ProblemConfig(
        num_variables=n,
        alpha_well=1.0,
        lambda_sat=0.0,
        c_sat=2.5,
        k_wall=20.0,
        x_bound=2.2,
    )
    return NonConvexIsingSAT(n, j_matrix=j_mat, h_bias=h_bias, clauses=[], config=config)
