import math
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from typing import Any, Tuple
from lightning.pytorch.utilities import grad_norm

from src.hyperbolic_math.src.manifolds import Euclidean
from src.models.abstract import AbstractModel
from src.models.utils import get_dtype_from_precision
from src.hyperbolic_math.src.utils.helpers import compute_pairwise_distances


class HyperbolicHCModel(AbstractModel):
    def __init__(
        self,
        model: AbstractModel,
        # Optimizer
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        lr_scheduler_params: dict | None = None,
        # Evaluation & Logging
        stage: str = "clustering",
        global_epoch: int = 0,
        log_interval: int = 5,
        evaluation_metrics: dict[str, bool] = {},
        # Hyperbolic Loss parameters
        hyperbolic_loss_start_weight: float = 1.0,
        hyperbolic_loss_end_weight: float = 1.0,
        hyperbolic_loss_weight_interval: int = 0,
        hyperbolic_loss_temperature: float = 0.05,
        lca_version: str = "ours",
        loss_version: str = "wang",
        similarity_pruning: dict | None = None,
        use_importance_sampling: bool = True,
        sampling_delta: float = 0.01,
        sampling_eps: float = 0.01,
        log_approximation_diff: bool = False,
        # Contrastive Loss parameters
        contrastive_loss_enabled: bool = False,
        contrastive_temperature: float = 0.5,
        contrastive_weight: float = 1.0,
        # Reconstruction Loss parameters
        recon_weight: float = 1.0,
        # Regularizer parameters
        regularize_to_origin_enabled: bool = False,
        regularizer_weight: float = 0.05,
        # Backbone kNN SNN similarity parameters
        use_backbone_knn_similarities: bool = False,
        snn_k: int = 10,
        freeze_encoder: bool = False,
    ):
        super().__init__(
            # Architecture
            model_type=model.hparams.model_type,
            params_dtype=model.hparams.params_dtype,
            activation_function=model.hparams.activation_function,
            feature_scaling=model.hparams.feature_scaling,
            hyperbolic_encoding=model.hparams.hyperbolic_encoding,
            latent_dim=model.hparams.latent_dim,
            manifold_name=model.hparams.manifold_name,
            manifold_precision=model.hparams.manifold_precision,
            curvature_magnitude=model.hparams.curvature_magnitude,
            hyperbolic_decoding=model.hparams.hyperbolic_decoding,
            # Optimizer
            optimizer_name=model.hparams.optimizer_name,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lr_scheduler_params=lr_scheduler_params,
            # Evaluation & Logging
            stage=stage,
            global_epoch=global_epoch,
            log_interval=log_interval,
            evaluation_metrics=evaluation_metrics,
            # (Hyperbolic-) Layer specific parameters
            clamping_factor=model.hparams.clamping_factor,
            smoothing_factor=model.hparams.smoothing_factor,
            hyperboloid_layer_learnable_scale=model.hparams.hyperboloid_layer_learnable_scale,
            FHCNN_normalize=model.hparams.FHCNN_normalize,
        )
        # Save hyperparameters while ignoring non-serializable model
        self.save_hyperparameters(ignore=["model"])

        # Load encoding modules with their weights and ensure weights are copied by loading state dict
        self.encoder = model.encoder  # type: ignore
        self.encoder.load_state_dict(model.encoder.state_dict())  # type: ignore
        self.hyperbolic_encoding_layer = model.hyperbolic_encoding_layer  # type: ignore
        self.hyperbolic_encoding_layer.load_state_dict(model.hyperbolic_encoding_layer.state_dict())  # type: ignore

        # Load decoding modules if decoding is used
        if self.hparams.recon_weight != 0.0 and model.decoder is not None:
            self.decoder = model.decoder  # type: ignore  # type: ignore
            self.decoder.load_state_dict(model.decoder.state_dict())  # type: ignore
            self.hyperbolic_decoding_layer = model.hyperbolic_decoding_layer  # type: ignore
            self.hyperbolic_decoding_layer.load_state_dict(model.hyperbolic_decoding_layer.state_dict()) # type: ignore
        else:
            self.decoder = None
            self.hyperbolic_decoding_layer = None

        # Feature scaling layer & rms
        if hasattr(model, "feature_scaling_layer"):
            self.feature_scaling_layer = model.feature_scaling_layer  # type: ignore
        self.rms = model.rms  # type: ignore

        # Hyperbolic loss scheduler
        if hyperbolic_loss_end_weight < 0 or hyperbolic_loss_start_weight < 0:
            raise ValueError("hyperbolic_loss_start_weight and hyperbolic_loss_end_weight must be non-negative.")
        if hyperbolic_loss_weight_interval < 0:
            raise ValueError("hyperbolic_loss_weight_interval must be non-negative.")
        if hyperbolic_loss_start_weight > hyperbolic_loss_end_weight:
            raise ValueError("hyperbolic_loss_start_weight must be less than or equal to hyperbolic_loss_end_weight.")

        # Freeze encoder backbone if requested (keeps SNN similarities stable across batches)
        if freeze_encoder:
            for param in self.encoder.parameters():  # type: ignore
                param.requires_grad_(False)

        # Rejection sampling caches (re-estimated each training step)
        self.cached_acceptance_rate = 0.3

    def configure_optimizers(self) -> Any:
        if self.hparams.optimizer_name == "Adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.learning_rate,
                eps=1e-5,
                weight_decay=self.hparams.weight_decay,
                amsgrad=True,
            )
        elif self.hparams.optimizer_name == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.learning_rate,
                momentum=0.9,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            raise ValueError(f"Invalid optimizer name: {self.hparams.optimizer_name}")

        if self.hparams.lr_scheduler_params is not None:
            if self.hparams.lr_scheduler_params["type"] is None:
                return optimizer
            if self.hparams.lr_scheduler_params["type"] == "warmup_cosine":
                scheduler = CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=self.hparams.lr_scheduler_params["interval"],
                    eta_min=self.hparams.lr_scheduler_params["cosine_min_factor"]
                    * self.hparams.learning_rate,
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "epoch",
                        "frequency": 1,
                    },
                }
        else:
            return optimizer

    def decode_triplet_indices_from_flattened_indices(self, indices: torch.Tensor, n: int) -> torch.Tensor:
        """Decode flat indices in [0, n*(n-1)*(n-2)) to triplets (i, j, k)."""
        batch_n = n
        i = indices % batch_n
        temp = indices // batch_n
        # Decode the second index 'j' from (n-1) options
        j = temp % (batch_n - 1)
        # Decode the third index 'k' from (n-2) options
        k = temp // (batch_n - 1) % (batch_n - 2)
        # Adjust j to avoid collision with i (i == j)
        j = j + (j >= i).long()
        # Adjust k to avoid collisions with i and j
        low = torch.min(i, j)
        high = torch.max(i, j)
        k = k + (k >= low).long() + (k >= high - 1).long()
        return torch.stack([i, j, k], dim=1)

    def symmetric_percentile_pruning(self, similarities: torch.Tensor, percentile: float) -> torch.Tensor:
        """Prunes a symmetric matrix by calculating the quantile on unique off-diagonal edges."""
        n = similarities.size(0)
        indices = torch.triu_indices(n, n, offset=1, device=similarities.device)
        unique_edges = similarities[indices[0], indices[1]]
        threshold = torch.quantile(unique_edges, percentile)
        condition = similarities < threshold
        similarities = torch.where(condition, torch.zeros_like(similarities), similarities)
        return similarities

    def symmetric_knn_pruning(self, similarities: torch.Tensor, k: int) -> torch.Tensor:
        """Prunes a symmetric matrix by only keeping edges to the top-k neighbors."""
        n = similarities.size(0)
        device = similarities.device
        # Extract the top k+1 NN (include self loops)
        k_plus_self = min(int(k) + 1, n)
        topk_values, topk_indices = torch.topk(similarities, k_plus_self, dim=1)
        pruned_matrix = torch.zeros_like(similarities)
        row_indices = torch.arange(n, device=device).unsqueeze(1).expand(-1, k_plus_self)
        pruned_matrix[row_indices, topk_indices] = topk_values
        # Reinforce symmetry
        pruned_matrix = torch.max(pruned_matrix, pruned_matrix.T)
        return pruned_matrix

    def compute_snn_similarities(self, features: torch.Tensor, k: int) -> torch.Tensor:
        """Compute Shared Nearest Neighbor (SNN) Jaccard similarity matrix in feature space.

        SNN_sim(i,j) = |kNN(i) ∩ kNN(j)| / |kNN(i) ∪ kNN(j)|

        Uses cosine similarity to find neighbors in the backbone feature space.
        """
        n = features.size(0)
        k = min(k, n - 1)
        with torch.no_grad():
            # Normalize for cosine-based neighbor lookup
            features_norm = F.normalize(features, p=2, dim=1)
            cosine_sim = features_norm @ features_norm.T  # (n, n)
            # Get k nearest neighbors (excluding self)
            _, knn_indices = torch.topk(cosine_sim, k + 1, dim=1)  # (n, k+1)
            knn_indices = knn_indices[:, 1:]  # drop self, (n, k)
            # Build binary membership matrix: membership[i, j] = 1 if j ∈ kNN(i)
            membership = torch.zeros(n, n, device=features.device, dtype=features.dtype)
            row_idx = torch.arange(n, device=features.device).unsqueeze(1).expand(-1, k)
            membership[row_idx, knn_indices] = 1.0
            # |kNN(i) ∩ kNN(j)| via dot product of membership rows
            intersection = membership @ membership.T  # (n, n)
            # |kNN(i) ∪ kNN(j)| = k + k - intersection (each row has exactly k neighbors)
            union = 2.0 * k - intersection
            similarities = intersection / union.clamp(min=1.0)
        return similarities

    def compute_full_loss(self, embeddings: torch.Tensor, similarities: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute the exact hyperbolic loss over all triplets."""
        n = embeddings.size(0)
        device = embeddings.device

        # Optional: Apply similarity pruning
        pruning_cfg = self.hparams.similarity_pruning
        if pruning_cfg is not None:
            version = pruning_cfg.get("version", None) if isinstance(pruning_cfg, dict) else None
            if version == "percentile_pruning":
                similarities = self.symmetric_percentile_pruning(
                    similarities, percentile=pruning_cfg.get("quantile", 0.2))
            elif version == "knn_pruning":
                similarities = self.symmetric_knn_pruning(
                    similarities, k=pruning_cfg.get("nearest_neighbors", 10))
            elif version == "threshold_pruning":
                threshold = pruning_cfg.get("threshold", 0.01)
                similarities = torch.where(similarities <= threshold, torch.zeros_like(similarities), similarities)

        # Compute the LCA depths
        if self.hparams.lca_version not in ["ours", "chami"]:
            raise ValueError(f"Invalid LCA version: {self.hparams.lca_version}")
        lcas = self.compute_lca_depth(embeddings, version=self.hparams.lca_version)

        # Sample all triplets
        all_indices = torch.arange(n, device=device)
        triplets = torch.combinations(all_indices, r=3)
        i, j, k = triplets[:, 0], triplets[:, 1], triplets[:, 2]

        # Compute the triplet loss N_t
        lca_stack = torch.stack((lcas[i, j], lcas[i, k], lcas[j, k]), dim=1)
        lca_max = lca_stack.max(dim=1, keepdim=True).values
        scaled_exp = torch.exp((lca_stack - lca_max) / self.hparams.hyperbolic_loss_temperature)
        softmax_stack = scaled_exp / scaled_exp.sum(dim=1, keepdim=True)

        sim_stack = torch.stack((similarities[i, j], similarities[i, k], similarities[j, k]), dim=1)
        triplet_cost = torch.sum(sim_stack * (1 - softmax_stack))

        # Compute the min/max triplet cost for normalization
        pairwise_sums = torch.stack(
            (similarities[i, j] + similarities[i, k],
             similarities[i, j] + similarities[j, k],
             similarities[i, k] + similarities[j, k]),
            dim=1,
        )
        triplet_cost_min = pairwise_sums.min(dim=1).values.sum()
        triplet_cost_max = pairwise_sums.max(dim=1).values.sum()
        similarities_sum = similarities.sum() - n  # off-diagonal sum

        # Loss variants
        normalized_cost = (triplet_cost - triplet_cost_min) / (triplet_cost_max - triplet_cost_min).clamp(min=1e-15)
        wang_cost = triplet_cost / triplet_cost_min.clamp(min=1e-15)
        chami_cost = triplet_cost + similarities_sum

        return {
            "wang_cost": wang_cost,
            "normalized_cost": normalized_cost,
            "chami_cost": chami_cost,
            "triplet_cost": triplet_cost,
            "triplet_cost_min": triplet_cost_min,
            "triplet_cost_max": triplet_cost_max,
            "similarities_sum": similarities_sum,
            "evaluated_triplets": torch.tensor(triplets.shape[0], device=device),
        }

    def compute_importance_sampling_loss(self, embeddings: torch.Tensor, importance_sample_size: int, similarities: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Compute hyperbolic loss via importance sampling."""
        n = embeddings.size(0)
        device = embeddings.device
        total_unordered_triplet_combinations = n * (n - 1) * (n - 2)

        if self.hparams.similarity_pruning.version not in [None, 'None']:
            print("Warning: Similarity pruning is not applied in the importance sampling loss estimator. "
                  "This may lead to a bias in the loss estimate if pruning is enabled.")

        # Note: We compute all similarities since the odds of not sampling an index is vanishingly
        # small for minimum sample sizes larger than 1_000 triplets
        if similarities is None:
            # Compute the pairwise similarities from hyperbolic distances
            distances = compute_pairwise_distances(embeddings, self.manifold)
            similarities = torch.reciprocal(1 + distances)

        # Compute the pairwise min/max similarities
        triu_idx = torch.triu_indices(n, n, offset=1, device=device)
        upper_sims = similarities[triu_idx[0], triu_idx[1]]
        min_sim = upper_sims.min()
        max_sim = upper_sims.max()
        max_min_delta = max_sim - min_sim

        # Compute the LCA depths
        if self.hparams.lca_version not in ["ours", "chami"]:
            raise ValueError(f"Invalid LCA version: {self.hparams.lca_version}")
        lcas = self.compute_lca_depth(embeddings, version=self.hparams.lca_version)

        if max_min_delta < 1e-15:
            print("Warning: Similarity range is very small. Embeddings have collapsed.")
            max_min_delta = 1e-15

        use_uniform_sampling_instead_of_importance_sampling = False
        if use_uniform_sampling_instead_of_importance_sampling:
            # Sample 1_000 triplets uniformly at random with replacement
            sample_size = 10
            sampled_indices = torch.randint(0, total_unordered_triplet_combinations, (sample_size,), device=device)
            accepted_trips = [self.decode_triplet_indices_from_flattened_indices(sampled_indices, n)]
        else:
            # Importance sampling loop
            accepted_trips: list[torch.Tensor] = []
            total_num_accepted = 0
            total_sampled = 0

            # Sample triplets in batches until we have accepted enough triplets for a low-variance estimate of the loss
            for _ in range(50):
                if total_num_accepted >= importance_sample_size:
                    break
                remaining_samples = importance_sample_size - total_num_accepted

                # Compute the required sample size to have a high probability of accepting the required number of triplets
                sample_size = min(int(remaining_samples / self.cached_acceptance_rate * 1.75), 200_000)

                # Sample random triplets uniformly at random from the set of all possible triplets
                sampled_indices = torch.randint(0, total_unordered_triplet_combinations, (sample_size,), device=device)
                triplets = self.decode_triplet_indices_from_flattened_indices(sampled_indices, n)
                i, j, k = triplets[:, 0], triplets[:, 1], triplets[:, 2]

                # Calculate the per-triplet spread for the sampled triplets
                with torch.no_grad():
                    pairwise_sums = torch.stack(
                        (similarities[i, j] + similarities[i, k],
                        similarities[i, j] + similarities[j, k],
                        similarities[i, k] + similarities[j, k]),
                        dim=1,
                    )
                    per_triplet_spread = pairwise_sums.max(dim=1).values - pairwise_sums.min(dim=1).values

                # Calculate the acceptance probability for the triplet candidates
                acceptance_probability = (per_triplet_spread / max_min_delta).clamp(min=0.0, max=1.0)

                # Accept/Reject the candidate triplet
                mask = torch.rand(sample_size, device=device) <= acceptance_probability
                accepted_indices = mask.nonzero(as_tuple=True)[0]

                # Take only as many accepted triplets as needed to reach the target sample size for the loss estimate
                num_accepted = min(accepted_indices.shape[0], remaining_samples)
                if num_accepted > 0:
                    accepted_trips.append(triplets[accepted_indices[:num_accepted]])
                    total_num_accepted += num_accepted
                total_sampled += sample_size

                # Update the acceptance rate cache for the next training step
                self.cached_acceptance_rate = max(total_num_accepted / total_sampled, 0.01)

        # Extract the accepted triplets and their corresponding indices
        triplets = torch.cat(accepted_trips, dim=0)
        i, j, k = triplets[:, 0], triplets[:, 1], triplets[:, 2]

        if use_uniform_sampling_instead_of_importance_sampling:
            # Compute the triplet loss N_t
            lca_stack = torch.stack((lcas[i, j], lcas[i, k], lcas[j, k]), dim=1)
            lca_max = lca_stack.max(dim=1, keepdim=True).values
            scaled_exp = torch.exp((lca_stack - lca_max) / self.hparams.hyperbolic_loss_temperature)
            softmax_stack = scaled_exp / scaled_exp.sum(dim=1, keepdim=True)

            sim_stack = torch.stack((similarities[i, j], similarities[i, k], similarities[j, k]), dim=1)
            triplet_cost = torch.sum(sim_stack * (1 - softmax_stack))

            # Compute the min/max triplet cost for normalization
            pairwise_sums = torch.stack(
                (similarities[i, j] + similarities[i, k],
                similarities[i, j] + similarities[j, k],
                similarities[i, k] + similarities[j, k]),
                dim=1,
            )
            triplet_cost_min = pairwise_sums.min(dim=1).values.sum()
            triplet_cost_max = pairwise_sums.max(dim=1).values.sum()
            similarities_sum = similarities.sum() - n  # off-diagonal sum

            # Loss variants
            normalized_cost = (triplet_cost - triplet_cost_min) / (triplet_cost_max - triplet_cost_min).clamp(min=1e-15)
            wang_cost = triplet_cost / triplet_cost_min.clamp(min=1e-15)
            chami_cost = triplet_cost + similarities_sum
        else:
            # Compute the triplet loss N_t
            lca_stack = torch.stack((lcas[i, j], lcas[i, k], lcas[j, k]), dim=1)
            lca_max = lca_stack.max(dim=1, keepdim=True).values
            scaled_exp = torch.exp((lca_stack - lca_max) / self.hparams.hyperbolic_loss_temperature)
            softmax_stack = scaled_exp / scaled_exp.sum(dim=1, keepdim=True)

            sim_stack = torch.stack((similarities[i, j], similarities[i, k], similarities[j, k]), dim=1)
            per_triplet_cost = torch.sum(sim_stack * (1 - softmax_stack), dim=1)

            # Compute the per-triplet min/max triplet cost for normalization
            pairwise_sums = torch.stack(
                (similarities[i, j] + similarities[i, k],
                similarities[i, j] + similarities[j, k],
                similarities[i, k] + similarities[j, k]),
                dim=1,
            )
            per_triplet_cost_min = pairwise_sums.min(dim=1).values
            per_triplet_cost_max = pairwise_sums.max(dim=1).values

            # Compute the normalized cost estimator (=X_t)
            per_triplet_range = (per_triplet_cost_max - per_triplet_cost_min).clamp(min=1e-15)
            X_t = (per_triplet_cost - per_triplet_cost_min) / per_triplet_range.detach() ## fix denom gradient
            X_t = torch.where(per_triplet_range > 1e-15, X_t, torch.zeros_like(X_t))
            normalized_cost = X_t.mean()

        if self.hparams.log_approximation_diff:
            # Compute the full cost
            true_results = self.compute_full_loss(embeddings, similarities)
            # Log the approximation error
            self.metrics.setdefault("clustering-importance_sampling/approximation_error_normalized", []).append((normalized_cost - true_results["normalized_cost"]).abs().detach().item())
            self.metrics.setdefault("clustering-importance_sampling/normalized_cost", []).append(normalized_cost.detach().item())
            self.metrics.setdefault("clustering-importance_sampling/true_normalized_cost", []).append(true_results["normalized_cost"].detach().item())

            if use_uniform_sampling_instead_of_importance_sampling:
                self.metrics.setdefault("clustering-importance_sampling/approximation_error_wang", []).append((wang_cost - true_results["wang_cost"]).abs().detach().item())
                self.metrics.setdefault("clustering-importance_sampling/wang_cost", []).append(wang_cost.detach().item())
                self.metrics.setdefault("clustering-importance_sampling/true_wang_cost", []).append(true_results["wang_cost"].detach().item())

                self.metrics.setdefault("clustering-importance_sampling/approximation_error_triplet_cost", []).append((triplet_cost - true_results["triplet_cost"]).abs().detach().item())
                self.metrics.setdefault("clustering-importance_sampling/triplet_cost", []).append(triplet_cost.detach().item())
                self.metrics.setdefault("clustering-importance_sampling/true_triplet_cost", []).append(true_results["triplet_cost"].detach().item())

            # Compute cosine similarity and the kullback-leibler divergence between gradients of importance-sampled and full loss
            is_grad = torch.autograd.grad(normalized_cost, embeddings, retain_graph=True, create_graph=False)[0]
            full_grad = torch.autograd.grad(true_results["normalized_cost"], embeddings, retain_graph=True, create_graph=False)[0]
            gradient_cos_sim = F.cosine_similarity(is_grad.reshape(1, -1), full_grad.reshape(1, -1)).item()
            self.metrics.setdefault("clustering-importance_sampling/gradient_cosine_similarity", []).append(gradient_cos_sim)
            gradient_kl_div = F.kl_div(F.log_softmax(is_grad.reshape(1, -1), dim=1), F.softmax(full_grad.reshape(1, -1), dim=1), reduction='batchmean').item()
            self.metrics.setdefault("clustering-importance_sampling/gradient_kl_divergence", []).append(gradient_kl_div)

            if use_uniform_sampling_instead_of_importance_sampling:
                is_grad = torch.autograd.grad(wang_cost, embeddings, retain_graph=True, create_graph=False)[0]
                full_grad = torch.autograd.grad(true_results["wang_cost"], embeddings, retain_graph=True, create_graph=False)[0]
                gradient_cos_sim = F.cosine_similarity(is_grad.reshape(1, -1), full_grad.reshape(1, -1)).item()
                self.metrics.setdefault("clustering-importance_sampling/gradient_cosine_similarity_wang", []).append(gradient_cos_sim)
                gradient_kl_div = F.kl_div(F.log_softmax(is_grad.reshape(1, -1), dim=1), F.softmax(full_grad.reshape(1, -1), dim=1), reduction='batchmean').item()
                self.metrics.setdefault("clustering-importance_sampling/gradient_kl_divergence_wang", []).append(gradient_kl_div)

                is_grad = torch.autograd.grad(triplet_cost, embeddings, retain_graph=True, create_graph=False)[0]
                full_grad = torch.autograd.grad(true_results["triplet_cost"], embeddings, retain_graph=True, create_graph=False)[0]
                gradient_cos_sim = F.cosine_similarity(is_grad.reshape(1, -1), full_grad.reshape(1, -1)).item()
                self.metrics.setdefault("clustering-importance_sampling/gradient_cosine_similarity_triplet", []).append(gradient_cos_sim)
                gradient_kl_div = F.kl_div(F.log_softmax(is_grad.reshape(1, -1), dim=1), F.softmax(full_grad.reshape(1, -1), dim=1), reduction='batchmean').item()
                self.metrics.setdefault("clustering-importance_sampling/gradient_kl_divergence_triplet", []).append(gradient_kl_div)

            # Log the cosine similarity of the last layer's gradients individually
            is_grad = torch.autograd.grad(normalized_cost, self.hyperbolic_encoding_layer.parameters(), retain_graph=True, create_graph=False)
            is_grad = torch.cat([g.reshape(-1) for g in is_grad])
            full_grad = torch.autograd.grad(true_results["normalized_cost"], self.hyperbolic_encoding_layer.parameters(), retain_graph=True, create_graph=False)
            full_grad = torch.cat([g.reshape(-1) for g in full_grad])
            gradient_cos_sim = F.cosine_similarity(is_grad.reshape(1, -1), full_grad.reshape(1, -1)).item()
            self.metrics.setdefault("clustering-importance_sampling/gradient_cosine_similarity_last_layer", []).append(gradient_cos_sim)
            gradient_kl_div = F.kl_div(F.log_softmax(is_grad.reshape(1, -1), dim=1), F.softmax(full_grad.reshape(1, -1), dim=1), reduction='batchmean').item()
            self.metrics.setdefault("clustering-importance_sampling/gradient_kl_divergence_last_layer", []).append(gradient_kl_div)

        return {
            "normalized_cost": normalized_cost,
            "similarity_min": min_sim.detach(),
            "similarity_max": max_sim.detach(),
            "acceptance_rate": torch.tensor(self.cached_acceptance_rate, device=device),
            "evaluated_triplets": torch.tensor(triplets.shape[0], device=device),
        }

    def compute_hyperbolic_loss(self, embeddings: torch.Tensor, similarities: torch.Tensor | None = None) -> torch.Tensor:
        n = embeddings.size(0)
        num_all_triplets = n * (n - 1) * (n - 2) // 6

        # Compute the minimum required sample size for importance sampling
        eps = self.hparams.sampling_eps
        delta = self.hparams.sampling_delta
        importance_sample_size = int(math.ceil((0.5 + 2/3 * eps) * math.log(2.0 / delta) / (eps ** 2)))

        if (num_all_triplets <= 2 * importance_sample_size) or not self.hparams.use_importance_sampling:
            if similarities is None:
                # Compute the pairwise similarities from hyperbolic distances
                distances = compute_pairwise_distances(embeddings, self.manifold)
                similarities = torch.reciprocal(1 + distances)
            # Compute the full cost
            results = self.compute_full_loss(embeddings, similarities)
            # Log the individual costs and cost components
            max_min_delta = results["triplet_cost_max"] - results["triplet_cost_min"]
            self.metrics.setdefault("clustering/max_min_delta", []).append(max_min_delta.detach().item())
            for key in results.keys():
                self.metrics.setdefault(f"clustering/{key}", []).append(results[key].detach().item())

            # Return the specified loss variant
            if self.hparams.loss_version == "wang":
                return results["wang_cost"] - 1.0
            elif self.hparams.loss_version == "normalized":
                return results["normalized_cost"]
            elif self.hparams.loss_version == "chami":
                return results["chami_cost"]
            else:
                raise ValueError(f"Invalid loss version: {self.hparams.loss_version}")
        else:
            # Approximate the true cost with importance sampling
            results = self.compute_importance_sampling_loss(embeddings, importance_sample_size, similarities=similarities)

            # Log the individual costs and cost components
            for key in results.keys():
                self.metrics.setdefault(f"clustering/{key}", []).append(results[key].detach().item())

            # Return the importance sampling loss
            if self.hparams.loss_version == "normalized":
                return results["normalized_cost"]
            else:
                raise ValueError(f"Invalid loss version for Importance Sampling: {self.hparams.loss_version}. "
                                  "For 'wang' or 'chami' disable importance sampling.")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.hparams.model_type == "feedforward":
            x = x.view(x.size(0), -1)
        hidden = self.encoder(x)  # type: ignore
        hidden = self.rms(hidden)  # type: ignore

        if self.hparams.feature_scaling and not isinstance(self.manifold, Euclidean):
            hidden = self.feature_scaling_layer(hidden)  # type: ignore
        manifold_z = self.hyperbolic_encoding_layer(hidden)  # type: ignore
        return manifold_z

    def decode(self, manifold_z: torch.Tensor) -> torch.Tensor:
        if self.decoder is None:
            raise ValueError("Decoding is not enabled in this model.")
        x = self.hyperbolic_decoding_layer(manifold_z)  # type: ignore
        # Cast back from the manifold precision to the model's precision
        x = x.to(dtype=get_dtype_from_precision(self.hparams.params_dtype))
        x_recon = self.decoder(x)  # type: ignore
        return x_recon

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        manifold_z = self.encode(x)
        if self.hparams.recon_weight == 0.0 or self.decoder is None:
            # Skip decoding
            return torch.zeros_like(x), manifold_z
        x_recon = self.decode(manifold_z)
        return x_recon, manifold_z

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x_aug_1, x_aug_2, x, _ = batch
        x_aug_1 = x_aug_1.to(dtype=get_dtype_from_precision(self.hparams.params_dtype))
        x_aug_2 = x_aug_2.to(dtype=get_dtype_from_precision(self.hparams.params_dtype))
        x = x.to(dtype=get_dtype_from_precision(self.hparams.params_dtype))

        # Hyperbolic loss
        manifold_z = self.encode(x)

        if not torch.isfinite(manifold_z).all():
            print("Warning: Non-finite values detected in manifold_z embeddings.")
            exit()

        # Optionally replace the similarity matrix with backbone SNN similarities.
        # The backbone features (before hyperbolic projection) carry richer semantic structure;
        # SNN Jaccard similarity in that space gives a more stable, informative signal.
        snn_similarities = None
        if self.hparams.use_backbone_knn_similarities:
            with torch.no_grad():
                backbone_features = self.encoder(x)  # type: ignore
                backbone_features = self.rms(backbone_features)  # type: ignore
            snn_similarities = self.compute_snn_similarities(backbone_features, k=self.hparams.snn_k)

        hyperbolic_loss = self.compute_hyperbolic_loss(manifold_z, similarities=snn_similarities)
        # Log the hyperbolic loss
        self.metrics.setdefault("clustering/hyperbolic_loss", []).append(hyperbolic_loss.detach().item())  # type: ignore
        # Compute the current hyperbolic loss weight
        if self.hparams.hyperbolic_loss_weight_interval > 0:
            hyperbolic_loss_weight = self.hparams.hyperbolic_loss_start_weight + (
                self.hparams.hyperbolic_loss_end_weight - self.hparams.hyperbolic_loss_start_weight
            ) * min(self.current_epoch / self.hparams.hyperbolic_loss_weight_interval, 1.0)
        else:
            hyperbolic_loss_weight = self.hparams.hyperbolic_loss_end_weight
        self.metrics["clustering/hyperbolic_loss_weight"] = hyperbolic_loss_weight
        # Add the weighted reconstruction loss to the total loss
        total_loss = hyperbolic_loss_weight * hyperbolic_loss

        # Reconstruction loss
        if self.hparams.recon_weight > 0.0 and self.decoder is not None:
            x_recon, manifold_z = self(x)
            if x_recon.shape != x.shape:
                x_recon = x_recon.view_as(x)
            recon_loss = F.mse_loss(x_recon, x)
            # Log the reconstruction loss
            self.metrics.setdefault("clustering/recon_loss", []).append(recon_loss.detach().item())  # type: ignore
            # Add the weighted reconstruction loss to the total loss
            total_loss += self.hparams.recon_weight * recon_loss

        # Contrastive loss: use (aug_1, aug_2) as the positive pair for proper two-view SimCLR
        if self.hparams.contrastive_weight > 0.0 and self.hparams.contrastive_loss_enabled:
            manifold_z_aug_1 = self.encode(x_aug_1)
            manifold_z_aug_2 = self.encode(x_aug_2)
            contrastive_loss = self.compute_contrastive_loss(manifold_z_aug_1, manifold_z_aug_2, temperature=self.hparams.contrastive_temperature)
            # Log the contrastive loss
            self.metrics.setdefault("clustering/contrastive_loss", []).append(contrastive_loss.detach().item())
            # Add the weighted contrastive loss to the total loss
            total_loss += self.hparams.contrastive_weight * contrastive_loss

        # Prevent boundary collapse by penalizing the mean embedding distance to the origin
        if self.hparams.regularize_to_origin_enabled:
            mean_dist_to_origin = self.manifold.dist_0(manifold_z).mean()
            regularizer_weight: float = getattr(self.hparams, 'regularizer_weight', 0.05)  # type: ignore
            total_loss += mean_dist_to_origin * regularizer_weight

        # Logging - DETACH tensors to prevent memory leaks
        self.metrics.setdefault("clustering/total_loss", []).append(total_loss.detach().item())
        # Manifold curvature
        c: torch.Tensor = self.manifold.c  # type: ignore
        self.metrics.setdefault("clustering/c", []).append(c.item())
        # Latent space statistics
        self.metrics.setdefault("latent_space/norm_min", []).append(manifold_z.norm(p=2, dim=-1).min().detach().item())
        self.metrics.setdefault("latent_space/norm_max", []).append(manifold_z.norm(p=2, dim=-1).max().detach().item())
        self.metrics.setdefault("latent_space/norm_mean", []).append(manifold_z.norm(p=2, dim=-1).mean().detach().item())
        self.metrics.setdefault("latent_space/norm_std", []).append(manifold_z.norm(p=2, dim=-1).std().detach().item())

        if self.hparams.hyperbolic_encoding in ["HyperbolicLinearHyperboloidFHCNN", "HyperbolicLinearHyperboloidFHNN"]:
            scale = self.hyperbolic_encoding_layer[1].scale  # type: ignore
            self.metrics.setdefault("hyperbolic_encoding_layer/scale", []).append(scale.detach().item())

        for key, value in grad_norm(self, norm_type=2).items():
            if key == "grad_2.0_norm_total":
                key = "grad_2.0_norm/total"
            self.metrics.setdefault(f"{key}", []).append(value)

        self.log(
            "train_loss",
            total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=False,
            sync_dist=False,
        )

        return total_loss
