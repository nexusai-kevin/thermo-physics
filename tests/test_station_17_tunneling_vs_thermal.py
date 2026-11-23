"""Unit and integration verification tests for Station 17 (station_17_tunneling_vs_thermal)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from stations.station_17_tunneling_vs_thermal import (
    BARRIER_WIDTHS,
    DEFAULT_TELEMETRY_PATH,
    build_double_well_barrier_hamiltonian,
    compute_ising_energy,
    run_pimc_sqa_sampler,
    run_station_17_benchmark,
    run_thermal_thrml_sampler,
)


def test_double_well_barrier_exact_heights_and_gaps() -> None:
    """Verify exact analytical barrier height H and ground-state gap Delta E_AB across all W in {1, 2, 3, 4, 6}."""
    for n_spins in (32, 64):
        for w in BARRIER_WIDTHS:
            J, b, spec = build_double_well_barrier_hamiltonian(
                n_spins=n_spins,
                barrier_width_w=w,
                barrier_height_h=8.0,
                energy_gap_ab=4.0,
            )
            assert J.shape == (n_spins, n_spins)
            assert b.shape == (n_spins,)
            assert np.allclose(J, J.T)
            assert np.allclose(np.diag(J), 0.0)

            # Verify strictly bipartite topology (zero edges within even-even or odd-odd blocks)
            assert np.allclose(J[0::2, 0::2], 0.0)
            assert np.allclose(J[1::2, 1::2], 0.0)

            assert abs(spec.verified_barrier_height - 8.0) < 1e-3
            assert abs(spec.verified_energy_gap - 4.0) < 1e-3
            assert spec.well_b_ground_energy < spec.well_a_energy < spec.barrier_peak_energy

            # Verify Well A is trapped against any partial single-spin flip
            s_a = np.ones(n_spins, dtype=np.float32)
            s_a[: 2 * w] = -1.0
            e_a = float(compute_ising_energy(s_a, J, b))
            for idx in range(n_spins):
                s_flip = s_a.copy()
                s_flip[idx] *= -1.0
                assert float(compute_ising_energy(s_flip, J, b)) > e_a


def test_pimc_beats_thermal_on_narrow_spike_and_crosses_over_on_wide_barrier() -> None:
    """Verify PIMC/SQA beats classical thermal Gibbs at W=1 and thermal beats PIMC at W=6."""
    J_narrow, b_narrow, spec_narrow = build_double_well_barrier_hamiltonian(n_spins=32, barrier_width_w=1)
    th_w1 = run_thermal_thrml_sampler(J_narrow, b_narrow, spec_narrow, n_trials=64)
    qa_w1 = run_pimc_sqa_sampler(J_narrow, b_narrow, spec_narrow, n_trials=64)

    assert qa_w1.ground_state_prob_pgs > th_w1.ground_state_prob_pgs
    assert qa_w1.tts_99_sweeps < th_w1.tts_99_sweeps
    assert qa_w1.tts_99_wall_ms < th_w1.tts_99_wall_ms

    J_wide, b_wide, spec_wide = build_double_well_barrier_hamiltonian(n_spins=64, barrier_width_w=6)
    th_w6 = run_thermal_thrml_sampler(J_wide, b_wide, spec_wide, n_trials=64)
    qa_w6 = run_pimc_sqa_sampler(J_wide, b_wide, spec_wide, n_trials=64)

    assert th_w6.ground_state_prob_pgs > qa_w6.ground_state_prob_pgs
    assert th_w6.tts_99_sweeps < qa_w6.tts_99_sweeps
    assert th_w6.tts_99_wall_ms < qa_w6.tts_99_wall_ms


def test_station_17_telemetry_artifact_schema() -> None:
    """Verify artifacts/station_17_telemetry.json exists and contains complete sweep and crossover telemetry."""
    assert DEFAULT_TELEMETRY_PATH.exists(), "artifacts/station_17_telemetry.json must be exported"
    data = json.loads(DEFAULT_TELEMETRY_PATH.read_text(encoding="utf-8"))

    assert data["station_id"] == 17
    assert data["station_slug"] == "station_17_tunneling_vs_thermal"
    assert data["experimental_design"]["trotter_slices_p"] == 32
    assert data["experimental_design"]["barrier_widths_w"] == [1, 2, 3, 4, 6]
    assert len(data["frontend_sweep_curves"]) == 10  # 2 system sizes x 5 barrier widths

    for sys_key in ("N_32", "N_64"):
        sys_block = data["systems"][sys_key]
        assert len(sys_block["thermal_thrml_curve"]) == 5
        assert len(sys_block["quantum_pimc_curve"]) == 5
        cross = sys_block["crossover_analysis"]
        assert cross["tts_99_sweeps_crossover"]["crossover_detected"] is True
        assert cross["tts_99_wall_ms_crossover"]["crossover_detected"] is True
        assert 1.0 < cross["tts_99_wall_ms_crossover"]["crossover_width_w"] < 6.0
        assert 1.0 < cross["tts_99_sweeps_crossover"]["crossover_width_w"] < 6.0


def test_thermo_lab_station_17_frontend_wiring() -> None:
    """Verify Station 17 component, telemetry JSON, and station HTML pages are wired in web/."""
    repo_root = DEFAULT_TELEMETRY_PATH.parent.parent
    web_telemetry = repo_root / "web" / "public" / "stations" / "station_17_telemetry.json"
    component_tsx = repo_root / "web" / "thermo-lab" / "QuantumTunnelingCrossover.tsx"
    page_tsx = repo_root / "web" / "thermo-lab" / "page.tsx"
    station_html = repo_root / "web" / "public" / "stations" / "station-17.html"

    assert web_telemetry.exists(), "web/public/stations/station_17_telemetry.json must exist"
    assert component_tsx.exists(), "web/thermo-lab/QuantumTunnelingCrossover.tsx must exist"
    assert station_html.exists(), "web/public/stations/station-17.html must exist"

    comp_src = component_tsx.read_text(encoding="utf-8")
    assert "/stations/station_17_telemetry.json" in comp_src
    assert "Crossover Threshold:" in comp_src
    assert "TTS_99" in comp_src
    assert "P_gs" in comp_src

    page_src = page_tsx.read_text(encoding="utf-8")
    assert "QuantumTunnelingCrossover" in page_src
    assert "station-17-tunneling" in page_src
    assert 'href: "/stations/17"' in page_src

