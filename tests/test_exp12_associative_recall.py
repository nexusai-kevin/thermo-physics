import json
from pathlib import Path

import numpy as np

from experiments.exp12_associative_recall import (
    CALIBRATION_PATH,
    MANIFEST_PATHS,
    ThrmlIsingAssociativeSampler,
    ThrmlPottsAssociativeSampler,
    corrupt_cue,
    generate_patterns,
    run_classical_hopfield,
    run_exponential_hopfield,
)


def test_patterns_and_samplers_recall():
    patterns = generate_patterns(num_patterns=10, n_bits=64, seed=42)
    assert patterns.shape == (10, 64)
    assert set(np.unique(patterns)).issubset({-1.0, 1.0})

    rng = np.random.default_rng(7)
    cues = np.array([corrupt_cue(patterns[i], 0.20, rng) for i in range(4)], dtype=np.float32)
    targets = patterns[:4]

    rec_c, _, _ = run_classical_hopfield(cues, patterns)
    rec_e, _, _ = run_exponential_hopfield(cues, patterns)
    assert np.mean(rec_e == targets) == 1.0
    assert np.mean(rec_c == targets) >= 0.75

    ising = ThrmlIsingAssociativeSampler(patterns)
    potts = ThrmlPottsAssociativeSampler(patterns)
    betas = [4.5, 4.5, 4.5, 4.5, 4.5, 4.5]

    rec_i, _, _ = ising.sample(cues, betas, sweeps_per_stage=2, seed=1)
    rec_p, _, _ = potts.sample(cues, betas, sweeps_per_stage=2, seed=1)
    assert np.mean(rec_i == targets) >= 0.90
    assert np.mean(rec_p == targets) == 1.0


def test_calibration_and_manifests_exist_and_valid():
    assert CALIBRATION_PATH.exists()
    cal = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    assert cal["experiment"] == "exp12_associative_recall"
    assert cal["thrml_version"] == "0.1.4"
    assert len(cal["inverse_temperatures_beta"]) == 6

    for mpath in MANIFEST_PATHS:
        assert mpath.exists()
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        exp12 = manifest["experiments"]["exp12_associative_recall"]
        assert exp12["experiment_id"] == 12
        for noise_key in ("20%", "35%", "50%"):
            bucket = exp12["results_by_noise"][noise_key]
            for model in ("classical_hopfield", "exponential_hopfield", "thrml_ising", "thrml_potts"):
                assert "bit_error_recovery" in bucket[model]
                assert "spurious_attractor_rate" in bucket[model]
                assert "mean_convergence_steps" in bucket[model]
                assert "execution_time_s" in bucket[model]
