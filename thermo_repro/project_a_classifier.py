"""Project A: Associative Memory & Robust Classification via Bipartite V-H-L Factor Graphs.

Implements the Energy-Based Classifier and Associative Memory architecture:
    E(v, h, l) = - sum_i a_i v_i - sum_j b_j h_j - sum_k c_k l_k
                 - sum_{i,j} W_{ij} v_i h_j - sum_{j,k} U_{jk} h_j l_k

Includes:
1. Synthetic 2D spin glyph dataset (4 distinct 4x4 or 8x8 classes with spin-flip noise & occlusion masks)
2. Contrastive Divergence (CD-k) training over the bipartite factor graph
3. Inference via Block Gibbs thermal relaxation (clamping V, relaxing H and L)
4. Associative memory completion (clamping partial visible nodes V_obs, relaxing V_masked, H, L)
"""

from __future__ import annotations

from typing import Any
import time
import numpy as np

from thermo_repro.backend import (
    Block,
    IsingEBM,
    IsingSamplingProgram,
    SamplingSchedule,
    compute_energy,
    hinton_init,
    make_rng_key,
    sample_states,
    split_rng_key,
)
from thermo_repro.coloring import build_vhl_bipartite_topology


def generate_glyph_prototypes(n_visible: int = 16) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Create 4 orthogonal/low-overlap spin prototypes in {-1, +1}^{n_visible} and
    their one-hot spin label encodings in {-1, +1}^4 (where active class k has l_k = +1, others -1).
    """
    side = int(np.sqrt(n_visible))
    if side * side != n_visible:
        raise ValueError(f"n_visible must be a perfect square (e.g., 16 or 64), got {n_visible}")

    # Class 0: Horizontal stripes
    g0 = np.ones((side, side), dtype=np.float32)
    g0[1::2, :] = -1.0

    # Class 1: Vertical stripes
    g1 = np.ones((side, side), dtype=np.float32)
    g1[:, 1::2] = -1.0

    # Class 2: Center cross / ring
    g2 = -np.ones((side, side), dtype=np.float32)
    mid = side // 2
    g2[mid - 1 : mid + 1, :] = 1.0
    g2[:, mid - 1 : mid + 1] = 1.0

    # Class 3: Checkerboard quadrants
    g3 = -np.ones((side, side), dtype=np.float32)
    g3[:mid, :mid] = 1.0
    g3[mid:, mid:] = 1.0

    prototypes = np.stack([g0.reshape(-1), g1.reshape(-1), g2.reshape(-1), g3.reshape(-1)], axis=0)
    n_classes = 4
    labels = -np.ones((n_classes, n_classes), dtype=np.float32)
    np.fill_diagonal(labels, 1.0)
    names = ["Horizontal Bars", "Vertical Bars", "Center Cross", "Diagonal Quadrants"]
    return prototypes, labels, names


class ThermodynamicVHLClassifier:
    """Bipartite V-H-L Factor Graph Classifier and Associative Memory."""

    def __init__(
        self,
        n_visible: int = 16,
        n_hidden: int = 16,
        n_labels: int = 4,
        beta: float = 1.5,
        seed: int = 42,
    ) -> None:
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_labels = n_labels
        self.beta = float(beta)
        self.topology = build_vhl_bipartite_topology(n_visible, n_hidden, n_labels)

        rng = np.random.default_rng(seed)
        self.a = np.zeros((n_visible,), dtype=np.float32)
        self.b = np.zeros((n_hidden,), dtype=np.float32)
        self.c = np.zeros((n_labels,), dtype=np.float32)
        self.W = rng.normal(0.0, 0.15, size=(n_visible, n_hidden)).astype(np.float32)
        self.U = rng.normal(0.0, 0.15, size=(n_hidden, n_labels)).astype(np.float32)

    def pack_parameters(self) -> tuple[np.ndarray, np.ndarray]:
        """Pack (a, b, c) into `biases` and (W, U) into `weights` matching `topology['all_edges']`."""
        biases = np.concatenate([self.a, self.b, self.c]).astype(np.float32)
        # vh_edges iterate v in v_nodes, h in h_nodes -> W.reshape(-1)
        # hl_edges iterate h in h_nodes, l in l_nodes -> U.reshape(-1)
        weights = np.concatenate([self.W.reshape(-1), self.U.reshape(-1)]).astype(np.float32)
        return biases, weights

    def build_ebm(self, beta: float | None = None) -> IsingEBM:
        """Instantiate a `thrml.models.IsingEBM` with current parameters."""
        biases, weights = self.pack_parameters()
        b_val = self.beta if beta is None else float(beta)
        return IsingEBM(
            self.topology["all_nodes"],
            self.topology["all_edges"],
            biases,
            weights,
            b_val,
        )

    def train_contrastive_divergence(
        self,
        v_train: np.ndarray,
        l_train: np.ndarray,
        epochs: int = 35,
        lr: float = 0.08,
        cd_steps: int = 3,
        seed: int = 101,
    ) -> list[float]:
        """Train (W, U, a, b, c) via Contrastive Divergence (CD-k) + latent prototype alignment.

        For each training example (v, l):
        - Positive phase: clamp (v, l) and compute exact mean-field hidden expectation
          <h>_pos = tanh(beta * (b + v @ W + l @ U^T))
        - Negative phase: perform k bipartite Block Gibbs sweeps alternating H and (V, L)
        - Update weights along the free-energy gradient: delta_W = lr * (<v h^T>_pos - <v h^T>_neg)
        """
        rng = np.random.default_rng(seed)
        n_samples = v_train.shape[0]
        loss_history: list[float] = []

        # Assign anchor hidden units to class prototypes to break symmetry quickly
        units_per_class = max(1, self.n_hidden // self.n_labels)
        for cls_idx in range(min(self.n_labels, n_samples)):
            h_slice = slice(cls_idx * units_per_class, (cls_idx + 1) * units_per_class)
            self.W[:, h_slice] += 0.35 * v_train[cls_idx][:, None] / np.sqrt(self.n_visible)
            self.U[h_slice, :] += 0.45 * l_train[cls_idx][None, :]

        for _epoch in range(epochs):
            epoch_recon_err = 0.0
            for idx in range(n_samples):
                v_pos = v_train[idx]
                l_pos = l_train[idx]

                # Positive phase (clamped V and L)
                h_field_pos = self.b + (v_pos @ self.W) + (self.U @ l_pos)
                h_prob_pos = 1.0 / (1.0 + np.exp(-np.clip(2.0 * self.beta * h_field_pos, -40.0, 40.0)))
                h_pos_mean = 2.0 * h_prob_pos - 1.0

                # Sample hidden spin state to start CD-k chain
                h_curr = np.where(rng.random(self.n_hidden) < h_prob_pos, 1.0, -1.0).astype(np.float32)
                v_curr = v_pos.copy()
                l_curr = l_pos.copy()

                for _step in range(cd_steps):
                    # Update V and L in parallel (since V and L are conditionally independent given H)
                    v_field = self.a + (self.W @ h_curr)
                    l_field = self.c + (h_curr @ self.U)
                    v_prob = 1.0 / (1.0 + np.exp(-np.clip(2.0 * self.beta * v_field, -40.0, 40.0)))
                    l_prob = 1.0 / (1.0 + np.exp(-np.clip(2.0 * self.beta * l_field, -40.0, 40.0)))
                    v_curr = np.where(rng.random(self.n_visible) < v_prob, 1.0, -1.0).astype(np.float32)
                    l_curr = np.where(rng.random(self.n_labels) < l_prob, 1.0, -1.0).astype(np.float32)

                    # Update H given (V_curr, L_curr)
                    h_field_neg = self.b + (v_curr @ self.W) + (self.U @ l_curr)
                    h_prob_neg = 1.0 / (1.0 + np.exp(-np.clip(2.0 * self.beta * h_field_neg, -40.0, 40.0)))
                    h_curr = np.where(rng.random(self.n_hidden) < h_prob_neg, 1.0, -1.0).astype(np.float32)

                h_neg_mean = 2.0 * h_prob_neg - 1.0

                # Contrastive Divergence parameter updates (with gentle L2 weight decay)
                self.W += lr * (np.outer(v_pos, h_pos_mean) - np.outer(v_curr, h_neg_mean) - 0.01 * self.W)
                self.U += lr * (np.outer(h_pos_mean, l_pos) - np.outer(h_neg_mean, l_curr) - 0.01 * self.U)
                self.a += lr * 0.2 * (v_pos - v_curr)
                self.b += lr * 0.2 * (h_pos_mean - h_neg_mean)
                self.c += lr * 0.2 * (l_pos - l_curr)

                epoch_recon_err += float(np.mean((v_pos - (2.0 * v_prob - 1.0)) ** 2))

            loss_history.append(epoch_recon_err / n_samples)

        return loss_history

    def classify_by_relaxation(
        self,
        v_input_spins: np.ndarray,
        n_warmup: int = 30,
        n_samples: int = 120,
        steps_per_sample: int = 1,
        seed: int = 99,
    ) -> dict[str, Any]:
        """Classify an input spin vector `v_input_spins` in {-1, +1}^{n_visible} using `thrml`:
        - Clamp `V` to `v_input_spins`
        - Leave `H` and `L` unclamped (`free_blocks = [Block(H), Block(L)]`)
        - Sample via Block Gibbs and compute the empirical expectation <L> over the relaxed chain.
        """
        model = self.build_ebm()
        free_blocks = self.topology["clamped_v_free_blocks"]  # [Block(H), Block(L)]
        clamped_blocks = [Block(self.topology["v_nodes"])]

        program = IsingSamplingProgram(
            model,
            free_blocks=free_blocks,
            clamped_blocks=clamped_blocks,
        )
        key = make_rng_key(seed)
        k_init, k_samp = split_rng_key(key, 2)
        clamped_v_bool = [v_input_spins > 0]
        init_state = hinton_init(k_init, model, free_blocks, clamped_v_bool)

        schedule = SamplingSchedule(
            n_warmup=n_warmup,
            n_samples=n_samples,
            steps_per_sample=steps_per_sample,
        )
        readout_blocks = [Block(self.topology["l_nodes"]), Block(self.topology["all_nodes"])]
        samples = sample_states(k_samp, program, schedule, init_state, clamped_v_bool, readout_blocks)

        l_samples_bool = np.asarray(samples[0], dtype=bool)
        all_samples_bool = np.asarray(samples[1], dtype=bool)
        l_spins = np.where(l_samples_bool, 1.0, -1.0)
        mean_l_magnetization = np.mean(l_spins, axis=0)
        predicted_class = int(np.argmax(mean_l_magnetization))

        biases, weights = self.pack_parameters()
        energies = compute_energy(
            all_samples_bool,
            self.topology["all_nodes"],
            self.topology["all_edges"],
            biases,
            weights,
        )

        return {
            "predicted_class": predicted_class,
            "label_magnetizations": mean_l_magnetization.tolist(),
            "mean_relaxed_energy": float(np.mean(energies)),
            "min_relaxed_energy": float(np.min(energies)),
        }

    def reconstruct_masked_pattern(
        self,
        v_corrupted_spins: np.ndarray,
        observed_mask: np.ndarray,
        n_warmup: int = 40,
        n_samples: int = 150,
        seed: int = 202,
    ) -> dict[str, Any]:
        """Associative Memory Pattern Completion:
        Clamp only the observed visible spins `V_obs` (where `observed_mask == True`),
        leaving `V_masked`, `H`, and `L` free to relax jointly in two bipartite blocks:
            Block 1: V_masked + L
            Block 2: H
        """
        model = self.build_ebm()
        v_nodes = self.topology["v_nodes"]
        h_nodes = self.topology["h_nodes"]
        l_nodes = self.topology["l_nodes"]

        obs_nodes = [v_nodes[i] for i in range(self.n_visible) if observed_mask[i]]
        masked_nodes = [v_nodes[i] for i in range(self.n_visible) if not observed_mask[i]]

        free_blocks = [Block(masked_nodes + l_nodes), Block(h_nodes)]
        clamped_blocks = [Block(obs_nodes)]
        clamped_vals = [v_corrupted_spins[observed_mask] > 0]

        program = IsingSamplingProgram(model, free_blocks=free_blocks, clamped_blocks=clamped_blocks)
        key = make_rng_key(seed)
        k_init, k_samp = split_rng_key(key, 2)
        init_state = hinton_init(k_init, model, free_blocks, clamped_vals)
        schedule = SamplingSchedule(n_warmup=n_warmup, n_samples=n_samples, steps_per_sample=1)

        samples = sample_states(
            k_samp,
            program,
            schedule,
            init_state,
            clamped_vals,
            [Block(v_nodes), Block(l_nodes)],
        )
        v_samples_spins = np.where(np.asarray(samples[0], dtype=bool), 1.0, -1.0)
        l_samples_spins = np.where(np.asarray(samples[1], dtype=bool), 1.0, -1.0)

        v_reconstructed = np.where(np.mean(v_samples_spins, axis=0) >= 0.0, 1.0, -1.0)
        predicted_class = int(np.argmax(np.mean(l_samples_spins, axis=0)))

        return {
            "reconstructed_spins": v_reconstructed,
            "predicted_class": predicted_class,
            "v_magnetizations": np.mean(v_samples_spins, axis=0).tolist(),
        }


def run_project_a_benchmark(
    n_visible: int = 16,
    n_hidden: int = 16,
    noise_flip_prob: float = 0.20,
    mask_fraction: float = 0.375,
    n_trials_per_class: int = 8,
    seed: int = 42,
) -> dict[str, Any]:
    """Execute the full Project A training, noisy classification, and associative memory benchmark."""
    t0 = time.perf_counter()
    prototypes, labels, class_names = generate_glyph_prototypes(n_visible=n_visible)
    n_classes = len(class_names)

    clf = ThermodynamicVHLClassifier(
        n_visible=n_visible,
        n_hidden=n_hidden,
        n_labels=n_classes,
        beta=1.5,
        seed=seed,
    )
    loss_history = clf.train_contrastive_divergence(
        prototypes,
        labels,
        epochs=40,
        lr=0.08,
        cd_steps=3,
        seed=seed + 1,
    )

    rng = np.random.default_rng(seed + 10)
    correct_clean = 0
    correct_noisy = 0
    total_noisy = 0

    # 1. Clean & Noisy Classification Evaluation
    for cls_idx in range(n_classes):
        res_clean = clf.classify_by_relaxation(prototypes[cls_idx], seed=seed + 100 + cls_idx)
        if res_clean["predicted_class"] == cls_idx:
            correct_clean += 1

        for t in range(n_trials_per_class):
            flips = rng.random(n_visible) < noise_flip_prob
            noisy_v = np.where(flips, -prototypes[cls_idx], prototypes[cls_idx])
            res_noisy = clf.classify_by_relaxation(
                noisy_v,
                seed=seed + 200 + cls_idx * 50 + t,
            )
            if res_noisy["predicted_class"] == cls_idx:
                correct_noisy += 1
            total_noisy += 1

    # 2. Associative Memory Pattern Completion under Occlusion Mask
    n_masked = int(round(n_visible * mask_fraction))
    observed_mask = np.ones((n_visible,), dtype=bool)
    observed_mask[-n_masked:] = False  # Mask out bottom portion of the glyph

    recovery_hamming_accs: list[float] = []
    masked_class_correct = 0
    for cls_idx in range(n_classes):
        v_masked_in = prototypes[cls_idx].copy()
        v_masked_in[~observed_mask] = 1.0  # Destroy masked region
        rec = clf.reconstruct_masked_pattern(
            v_masked_in,
            observed_mask=observed_mask,
            seed=seed + 500 + cls_idx,
        )
        acc = float(np.mean(rec["reconstructed_spins"][~observed_mask] == prototypes[cls_idx][~observed_mask]))
        recovery_hamming_accs.append(acc)
        if rec["predicted_class"] == cls_idx:
            masked_class_correct += 1

    elapsed = time.perf_counter() - t0
    return {
        "project": "Project A: Associative Memory & Robust EBM Classification",
        "architecture": {
            "n_visible": n_visible,
            "n_hidden": n_hidden,
            "n_labels": n_classes,
            "n_total_nodes": len(clf.topology["all_nodes"]),
            "n_total_edges": len(clf.topology["all_edges"]),
            "bipartite_blocks": 2,
        },
        "training": {
            "method": "Contrastive Divergence (CD-3)",
            "initial_recon_mse": float(loss_history[0]),
            "final_recon_mse": float(loss_history[-1]),
        },
        "classification_metrics": {
            "clean_accuracy": float(correct_clean / n_classes),
            "noise_flip_probability": float(noise_flip_prob),
            "noisy_accuracy": float(correct_noisy / total_noisy),
            "n_noisy_trials": total_noisy,
        },
        "associative_memory_metrics": {
            "masked_fraction": float(mask_fraction),
            "occluded_bit_recovery_accuracy": float(np.mean(recovery_hamming_accs)),
            "occluded_classification_accuracy": float(masked_class_correct / n_classes),
        },
        "wall_time_sec": float(elapsed),
    }
