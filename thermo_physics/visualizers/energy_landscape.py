"""
thermo_physics.visualizers.energy_landscape
===========================================

Visualizer for thermodynamic potential landscapes, Langevin trajectory barrier hops,
and bipartite block Gibbs transitions. Exportable to static PNG, JSON data, and standalone HTML.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def simulate_langevin_double_well(
    n_steps: int = 1500,
    dt: float = 0.02,
    beta: float = 1.0,
    a: float = 1.5,
    lam: float = 1.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate overdamped Langevin dynamics in a double-well potential V(x).

    dX_t = - V'(X_t) dt + sqrt(2 / beta) dW_t
    V(x) = (lam / 4) * (x^2 - a^2)^2
    V'(x) = lam * x * (x^2 - a^2)
    """
    rng = np.random.default_rng(seed)
    x = np.zeros(n_steps, dtype=np.float64)
    x[0] = -a  # Start in left well

    sigma = np.sqrt(2.0 * dt / beta)
    for t in range(1, n_steps):
        grad = lam * x[t - 1] * (x[t - 1] ** 2 - a ** 2)
        noise = rng.normal(0.0, sigma)
        x[t] = x[t - 1] - grad * dt + noise
        # Stability clipping
        x[t] = np.clip(x[t], -3.0, 3.0)

    time_axis = np.arange(n_steps) * dt
    return time_axis, x


def generate_landscape_figure(output_png: str = "docs/energy_landscape.png") -> None:
    """Generate high-resolution 3-panel thermodynamic energy landscape figure."""
    os.makedirs(os.path.dirname(output_png), exist_ok=True)

    fig = plt.figure(figsize=(15, 5), dpi=150)
    fig.patch.set_facecolor("#0b0f19")

    # 1. Panel A: 1D Double-Well Potential V(x) and Kramers Hopping
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_facecolor("#0f172a")

    x_grid = np.linspace(-2.2, 2.2, 300)
    a, lam = 1.5, 1.0
    v_grid = (lam / 4.0) * (x_grid ** 2 - a ** 2) ** 2

    ax1.plot(x_grid, v_grid, color="#38bdf8", lw=2.5, label="Potential V(x)")
    ax1.axhline(0, color="#334155", ls="--", lw=0.8)
    ax1.axvline(-a, color="#34d399", ls=":", label=f"Left Well (-{a})")
    ax1.axvline(a, color="#34d399", ls=":", label=f"Right Well (+{a})")
    ax1.axvline(0, color="#f43f5e", ls="--", label=f"Barrier ΔV = {(lam * a**4 / 4):.2f}")

    ax1.set_title("1D Double-Well Potential & Barrier", color="#f8fafc", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Continuous State Position (x)", color="#94a3b8")
    ax1.set_ylabel("Energy Potential V(x)", color="#94a3b8")
    ax1.tick_params(colors="#94a3b8")
    for spine in ax1.spines.values():
        spine.set_color("#334155")
    ax1.legend(loc="upper center", facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc", fontsize=8)

    # 2. Panel B: High-Beta vs Low-Beta Langevin Trajectories
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_facecolor("#0f172a")

    t_cold, x_cold = simulate_langevin_double_well(n_steps=1200, beta=4.0, seed=101)
    t_hot, x_hot = simulate_langevin_double_well(n_steps=1200, beta=0.8, seed=202)

    ax2.plot(t_hot, x_hot, color="#fb7185", lw=1.2, alpha=0.85, label="High Temp (β=0.8, Active Hopping)")
    ax2.plot(t_cold, x_cold, color="#34d399", lw=1.5, alpha=0.9, label="Low Temp (β=4.0, Well Trapping)")
    ax2.axhline(-a, color="#64748b", ls=":")
    ax2.axhline(a, color="#64748b", ls=":")
    ax2.axhline(0, color="#f43f5e", ls="--", alpha=0.5)

    ax2.set_title("Langevin Trajectories across Kramers Barrier", color="#f8fafc", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Simulation Time (t)", color="#94a3b8")
    ax2.set_ylabel("State Coordinate X(t)", color="#94a3b8")
    ax2.tick_params(colors="#94a3b8")
    for spine in ax2.spines.values():
        spine.set_color("#334155")
    ax2.legend(loc="upper right", facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc", fontsize=8)

    # 3. Panel C: 2D Bipartite Spin Graph & Energy Contours
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_facecolor("#0f172a")

    s1_grid = np.linspace(-1.5, 1.5, 100)
    s2_grid = np.linspace(-1.5, 1.5, 100)
    S1, S2 = np.meshgrid(s1_grid, s2_grid)
    # Energy landscape of two coupled spins: E = - J s1 s2 + 0.5 (s1^2 - 1)^2 + 0.5 (s2^2 - 1)^2
    J_coup = 1.2
    E_2d = -J_coup * S1 * S2 + 0.5 * (S1 ** 2 - 1.0) ** 2 + 0.5 * (S2 ** 2 - 1.0) ** 2

    cs = ax3.contourf(S1, S2, E_2d, levels=18, cmap="viridis")
    cbar = fig.colorbar(cs, ax=ax3)
    cbar.ax.tick_params(colors="#94a3b8")
    cbar.set_label("Hamiltonian Energy E(s1, s2)", color="#94a3b8")

    # Mark discrete spin vertices
    ax3.scatter([1, -1, 1, -1], [1, -1, -1, 1], color=["#34d399", "#34d399", "#f43f5e", "#f43f5e"], s=100, zorder=5)
    ax3.annotate("Ground (+1,+1)", (1.05, 1.05), color="#34d399", fontsize=8, fontweight="bold")
    ax3.annotate("Ground (-1,-1)", (-1.45, -1.25), color="#34d399", fontsize=8, fontweight="bold")
    ax3.annotate("Frustrated (+1,-1)", (0.45, -1.25), color="#f43f5e", fontsize=8)

    ax3.set_title("2D Bipartite Spin Coupling & Sinks", color="#f8fafc", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Spin Sublattice A (s1)", color="#94a3b8")
    ax3.set_ylabel("Spin Sublattice B (s2)", color="#94a3b8")
    ax3.tick_params(colors="#94a3b8")
    for spine in ax3.spines.values():
        spine.set_color("#334155")

    plt.tight_layout()
    plt.savefig(output_png, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"[Visualizer] Energy landscape figure saved: {output_png}")


def export_landscape_json(output_json: str = "docs/energy_landscape_data.json") -> None:
    """Export potential energy curve and simulated trajectories to structured JSON."""
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    t_cold, x_cold = simulate_langevin_double_well(n_steps=500, beta=4.0, seed=101)
    t_hot, x_hot = simulate_langevin_double_well(n_steps=500, beta=0.8, seed=202)

    payload = {
        "metadata": {
            "title": "Thermodynamic Double-Well Langevin Barrier Trajectories",
            "potential": "V(x) = 0.25 * (x^2 - 1.5^2)^2",
            "kramers_barrier": 1.2656,
        },
        "trajectories": {
            "time": t_cold.tolist(),
            "cold_trajectory_beta_4_0": x_cold.tolist(),
            "hot_trajectory_beta_0_8": x_hot.tolist(),
        },
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[Visualizer] Landscape trajectory data exported: {output_json}")


def main():
    generate_landscape_figure("docs/energy_landscape.png")
    export_landscape_json("docs/energy_landscape_data.json")


if __name__ == "__main__":
    main()
