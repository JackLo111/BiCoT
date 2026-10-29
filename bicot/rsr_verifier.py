from typing import Optional

import numpy as np
import torch
from tqdm import tqdm


class RSRBlackBoxVerifier:

    def __init__(self, vocab_size: int, device: str = "cuda"):
        self.vocab_size = vocab_size
        self.device = device

        self.sentinel_indices: Optional[list[int]] = None

        self.w_sig: Optional[np.ndarray] = None

        self.L_base_mean: Optional[np.ndarray] = None

    def select_optimal_sentinels(
        self,
        model,
        tokenizer,
        secret_vectors: torch.Tensor,
        calibration_texts: list[str],
        n_sentinels: int = 100,
        min_visibility_rank: int = 1000,
    ):
        print(">>> Starting Optimal Sentinel Selection...")
        model.eval()

        if secret_vectors.dim() == 1:
            secret_vectors = secret_vectors.unsqueeze(0)

        secret_vectors = secret_vectors.to(self.device).double()

        print(f"   [1/2] Analyzing Token Visibility on {len(calibration_texts)} samples...")
        token_counts = torch.zeros(self.vocab_size, device=self.device, dtype=torch.float64)

        batch_size = 8
        for i in range(0, len(calibration_texts), batch_size):
            batch = calibration_texts[i : i + batch_size]
            enc = tokenizer(batch, add_special_tokens=False)["input_ids"]
            for seq in enc:
                indices = torch.tensor(seq, device=self.device)
                token_counts.index_add_(0, indices, torch.ones_like(indices, dtype=torch.float64))

        _, visible_candidates = torch.topk(token_counts, k=min_visibility_rank)
        visible_mask = torch.zeros(self.vocab_size, device=self.device, dtype=torch.bool)
        visible_mask[visible_candidates] = True
        visible_mask[:10] = False

        print("   [2/2] Computing Sensitivity for Candidate Tokens...")

        if hasattr(model, "lm_head"):
            lm_weights = model.lm_head.weight.data.double()
        else:
            lm_weights = model.embed_out.weight.data.double()

        all_projections = torch.matmul(secret_vectors, lm_weights.t())
        sensitivity_scores, _ = all_projections.abs().max(dim=0)

        if sensitivity_scores.shape[0] > self.vocab_size:
            sensitivity_scores = sensitivity_scores[: self.vocab_size]

        sensitivity_scores[~visible_mask] = -1.0

        _, top_indices = torch.topk(sensitivity_scores, k=n_sentinels)
        self.sentinel_indices = sorted(top_indices.cpu().numpy().tolist())

        print(f"✅ Selected {len(self.sentinel_indices)} optimal sentinels.")

    def precompute_reference_statistics(
        self,
        model,
        tokenizer,
        secret_vectors: torch.Tensor,
        ref_prompts: list[str],
        batch_size: int = 8,
    ):
        print(">>> Pre-computing Base Model Geometry (Source Point Cloud)...")
        model.eval()

        if secret_vectors.dim() == 1:
            secret_vectors = secret_vectors.unsqueeze(0)

        secret_vectors = secret_vectors.to(self.device).double()

        W_sent = model.lm_head.weight.data[self.sentinel_indices, :].double()
        self.w_sig = torch.matmul(secret_vectors, W_sent.t()).cpu().numpy().astype(np.float64)


        logits_accumulator = np.zeros((len(self.sentinel_indices),), dtype=np.float64)
        count = 0

        with torch.no_grad():
            for i in tqdm(range(0, len(ref_prompts), batch_size), desc="Collecting Base Logits"):
                batch = ref_prompts[i : i + batch_size]
                inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(
                    self.device
                )
                outputs = model(**inputs)

                batch_logits = (
                    outputs.logits[:, -1, :][:, self.sentinel_indices].double().cpu().numpy()
                )

                logits_accumulator += np.sum(batch_logits, axis=0)
                count += len(batch)

        self.L_base_mean = logits_accumulator / count
        print(f" Base Geometry Computed. Shape: {self.L_base_mean.shape}")

    def robust_subspace_registration(
        self, deployed_logits_ref: np.ndarray, residual_threshold: float = 3.0
    ) -> np.ndarray:
        base_center = self.L_base_mean

        deployed_center = np.median(deployed_logits_ref, axis=0)

        drift_vector = deployed_center - base_center

        global_median_drift = np.median(drift_vector)
        drift_deviation = np.abs(drift_vector - global_median_drift)

        mad = np.median(drift_deviation)

        robust_z_score = 0.6745 * drift_deviation / (mad + 1e-9)

        is_healthy = robust_z_score < 3.5

        len(drift_vector) - np.sum(is_healthy)

        w_robust_list = []
        for i in range(self.w_sig.shape[0]):
            original_sig = self.w_sig[i]

            robust_sig = original_sig + drift_vector

            robust_sig[~is_healthy] = 0.0

            w_robust_list.append(robust_sig)

        return np.array(w_robust_list)

    def verify_detection(self, anchor_logits: np.ndarray, w_robust: np.ndarray) -> dict:
        anchor_logits = anchor_logits.astype(np.float64)
        n_samples = anchor_logits.shape[0]
        w_robust.shape[0]

        sample_scores = np.zeros(n_samples)

        w_robust_norm = (w_robust - np.mean(w_robust, axis=1, keepdims=True)) / (
            np.std(w_robust, axis=1, keepdims=True) + 1e-9
        )

        for i in range(n_samples):
            logit_vec = anchor_logits[i]

            if np.std(logit_vec) < 1e-9:
                logit_vec_norm = logit_vec
            else:
                logit_vec_norm = (logit_vec - np.mean(logit_vec)) / np.std(logit_vec)

            corrs = np.dot(w_robust_norm, logit_vec_norm) / len(logit_vec)

            sample_scores[i] = np.mean(corrs)

        return {"sample_scores": sample_scores, "mean_score": np.mean(sample_scores)}
