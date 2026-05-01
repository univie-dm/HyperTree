import math
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from typing import Any, Tuple
from lightning.pytorch.utilities import grad_norm

from src.hyperbolic_math.src.manifolds import Euclidean
from src.models.abstract import AbstractModel, LARS
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
        hyperbolic_loss_end_weight: float = 3.0,
        hyperbolic_loss_weight_interval: int = 300,
        hyperbolic_loss_start_temperature: float = 0.2,
        hyperbolic_loss_end_temperature: float = 0.1,
        hyperbolic_loss_temperature_interval: int = 250,
        lca_version: str = "ours",
        loss_version: str = "normalized",
        sampling_strategy: str | None = "importance",
        sampling_eps: float = 1e-3,
        sampling_delta: float = 1e-2,
        sampling_alpha: float = 0.5,
        uniform_sample_size: int = 1_000,
        log_approximation_diff: bool = False,
        # Contrastive Loss parameters
        contrastive_loss_enabled: bool = False,
        contrastive_temperature: float = 0.5,
        contrastive_weight: float = 1.0,
        contrastive_euclidean: bool = False,
        # Reconstruction Loss parameters
        recon_weight: float = 1.0,
        # Regularizer parameters
        regularize_to_origin_enabled: bool = False,
        regularizer_weight: float = 0.05,
        regularizer_type: str = "origin",  # "origin" | "uniformity" | "off"
        # Head warmup (clustering stage): for the first N epochs, train the
        # hyperbolic head only — HypHC weight forced to 0 and contrastive routed
        # through the head (geodesic) to give it supervised signal before
        # HypHC takes over. Backbone freeze is handled by freeze_encoder.
        head_warmup_epochs: int = 0,
        # Backbone kNN SNN similarity parameters
        use_backbone_knn_similarities: bool = False,
        snn_k: int = 10,
        freeze_encoder: bool = False,
        unfreeze_backbone_from: str | None = None,
        backbone_lr_factor: float = 0.1,
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

            # Selectively unfreeze backbone layers for fine-tuning
            if unfreeze_backbone_from is not None:
                resnet_backbone = self.encoder[0]  # type: ignore  # First module in Sequential is the ResNet
                layer_names = ["layer1", "layer2", "layer3", "layer4"]
                if not all(hasattr(resnet_backbone, ln) for ln in layer_names):
                    raise NotImplementedError(
                        f"unfreeze_backbone_from is only wired for ResNet backbones; got "
                        f"{type(resnet_backbone).__name__} (e.g. ViT). Set unfreeze_backbone_from=null."
                    )
                if unfreeze_backbone_from not in layer_names:
                    raise ValueError(f"unfreeze_backbone_from must be one of {layer_names}, got '{unfreeze_backbone_from}'")
                start_idx = layer_names.index(unfreeze_backbone_from)
                for name in layer_names[start_idx:]:
                    for param in getattr(resnet_backbone, name).parameters():
                        param.requires_grad_(True)
                print(f"Unfroze backbone from {unfreeze_backbone_from} onwards", flush=True)

        # Rejection sampling caches (re-estimated each training step)
        self.cached_acceptance_rate = 0.2

    def configure_optimizers(self) -> Any:
        # Separate backbone params (unfrozen via unfreeze_backbone_from) from head params
        # so backbone gets a lower learning rate
        backbone_params = []
        head_params = []
        use_discriminative_lr = (
            self.hparams.unfreeze_backbone_from is not None
            and self.hparams.backbone_lr_factor != 1.0
        )
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if use_discriminative_lr and name.startswith("encoder.0."):
                backbone_params.append(param)
            else:
                head_params.append(param)

        if backbone_params:
            param_groups = [
                {"params": head_params},
                {"params": backbone_params, "lr": self.hparams.learning_rate * self.hparams.backbone_lr_factor},
            ]
        else:
            param_groups = [{"params": head_params}]

        if self.hparams.optimizer_name == "Adam":
            optimizer = torch.optim.Adam(
                param_groups,
                lr=self.hparams.learning_rate,
                eps=1e-5,
                weight_decay=self.hparams.weight_decay,
                amsgrad=True,
            )
        elif self.hparams.optimizer_name == "SGD":
            optimizer = torch.optim.SGD(
                param_groups,
                lr=self.hparams.learning_rate,
                momentum=0.9,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer_name == "LARS":
            optimizer = LARS(
                param_groups,
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
                warmup_epochs = self.hparams.lr_scheduler_params["interval"]
                total_epochs = self.trainer.max_epochs
                eta_min = (
                    self.hparams.lr_scheduler_params["cosine_min_factor"]
                    * self.hparams.learning_rate
                )
                warmup = LinearLR(
                    optimizer, start_factor=1e-3, total_iters=warmup_epochs
                )
                cosine = CosineAnnealingLR(
                    optimizer, T_max=total_epochs - warmup_epochs, eta_min=eta_min
                )
                scheduler = SequentialLR(
                    optimizer,
                    schedulers=[warmup, cosine],
                    milestones=[warmup_epochs],
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

    def get_power_weighted_sample_size(self, similarities: torch.Tensor, max_min_delta: float) -> int:
        """Dynamically estimates the minimum power-weighted sample size."""
        n = similarities.size(0)
        device = similarities.device
        eps = self.hparams.sampling_eps
        delta = self.hparams.sampling_delta
        alpha = self.hparams.sampling_alpha

        # Sample a lightweight uniform batch to estimate the expectations
        num_est_samples = min(5000, n * (n - 1) * (n - 2))
        sampled_indices = torch.randint(0, n * (n - 1) * (n - 2), (num_est_samples,), device=device)
        triplets = self.decode_triplet_indices_from_flattened_indices(sampled_indices, n)
        i, j, k = triplets[:, 0], triplets[:, 1], triplets[:, 2]

        # Calculate the per-triplet spread for the sampled triplets
        with torch.no_grad():
            pairwise_sums_stack = torch.stack(
                (similarities[i, j] + similarities[i, k],
                 similarities[i, j] + similarities[j, k],
                 similarities[i, k] + similarities[j, k]),
                dim=1,
            )
            triplet_spread_stack = pairwise_sums_stack.max(dim=1).values - pairwise_sums_stack.min(dim=1).values

            # Estimate mu_V^(alpha) and prevent division by zero (if similarities collapse)
            mean_triplet_spread_stack = triplet_spread_stack.mean().item()
            mean_triplet_spread_stack_alpha = (triplet_spread_stack ** alpha).mean().clamp(min=1e-10).item()
            mu_V = mean_triplet_spread_stack / mean_triplet_spread_stack_alpha

        # Calculate the required sample size
        m_required = (8 * (max_min_delta ** (2 * (1 - alpha))) * math.log(4.0 / delta)) / ((eps ** 2) * (mu_V ** 2))

        # Optional: Apply EMA to stabilize training
        apply_EMA_stabilizer = False
        if apply_EMA_stabilizer:
            if not hasattr(self, "running_pw_sample_size"):
                self.running_pw_sample_size = m_required
            else:
                self.running_pw_sample_size = 0.9 * self.running_pw_sample_size + 0.1 * m_required
        else:
            self.running_pw_sample_size = m_required

        # Prevent under-sampling
        final_m = max(int(self.running_pw_sample_size), 10_000)

        # Log the minimum sample size
        self.metrics.setdefault("sampling/target_pw_sample_size", []).append(final_m)
        self.metrics.setdefault("sampling/final_required_diff_pw_sample_size", []).append(abs(final_m - int(m_required)))
        return final_m

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

    def compute_full_loss(self, embeddings: torch.Tensor, similarities: torch.Tensor, hyperbolic_loss_temperature: float) -> dict[str, torch.Tensor]:
        """Compute the exact hyperbolic loss over all triplets."""
        n = embeddings.size(0)
        device = embeddings.device

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
        scaled_exp = torch.exp((lca_stack - lca_max) / hyperbolic_loss_temperature)
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

    def compute_importance_sampling_loss(self, embeddings: torch.Tensor, importance_sample_size: int, hyperbolic_loss_temperature: float, similarities: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Compute hyperbolic loss via importance sampling."""
        n = embeddings.size(0)
        device = embeddings.device
        num_all_triplets = n * (n - 1) * (n - 2) // 6
        total_unordered_triplet_combinations = n * (n - 1) * (n - 2)

        # Note: We compute all similarities since the odds of not sampling an index is vanishingly
        # small for minimum sample sizes larger than 1_000 triplets
        if similarities is None:
            # Compute the pairwise similarities from hyperbolic distances
            distances = compute_pairwise_distances(embeddings, self.manifold)
            similarities = torch.reciprocal(1 + distances)

        # Compute the pairwise min/max similarities
        triu_idx = torch.triu_indices(n, n, offset=1, device=device)
        upper_sims = similarities[triu_idx[0], triu_idx[1]]
        min_sim = upper_sims.min().detach()
        max_sim = upper_sims.max().detach()
        max_min_delta = (max_sim - min_sim).detach()
        self.metrics.setdefault("sampling/similarities_min", []).append(min_sim)
        self.metrics.setdefault("sampling/similarities_max", []).append(max_sim)
        self.metrics.setdefault("sampling/similarities_max_min_delta", []).append(max_min_delta)

        # Compute the LCA depths
        if self.hparams.lca_version not in ["ours", "chami"]:
            raise ValueError(f"Invalid LCA version: {self.hparams.lca_version}")
        lcas = self.compute_lca_depth(embeddings, version=self.hparams.lca_version)

        # Compute the minimum required sample size for importance/power_weighted sampling
        eps = self.hparams.sampling_eps
        delta = self.hparams.sampling_delta
        alpha = self.hparams.sampling_alpha

        # Evaluate the full triplet set when the sampling strategy requires at least all triplets
        use_full_triplet_cost = False
        if self.hparams.sampling_strategy in [None, "None"]:
            use_full_triplet_cost = True
        elif self.hparams.sampling_strategy == "importance":
            use_full_triplet_cost = importance_sample_size >= num_all_triplets
        elif self.hparams.sampling_strategy == "power_weighted":
            # Dynamically compute the power_weighted minimum sample size with a O(1) lookup
            power_weighted_sample_size = self.get_power_weighted_sample_size(similarities, max_min_delta.item())
            use_full_triplet_cost = power_weighted_sample_size >= num_all_triplets

        # Sample the triplets
        with torch.no_grad():
            if use_full_triplet_cost:
                # Sample all triplets
                all_indices = torch.arange(n, device=device)
                triplets = torch.combinations(all_indices, r=3)
            elif self.hparams.sampling_strategy == "uniform":
                # Sample triplets uniformly at random with replacement
                sampled_indices = torch.randint(0, total_unordered_triplet_combinations, (self.hparams.uniform_sample_size,), device=device)
                triplets = self.decode_triplet_indices_from_flattened_indices(sampled_indices, n)
            elif self.hparams.sampling_strategy == "importance":
                # Sample triplets using importance sampling
                accepted_triplets: list[torch.Tensor] = []
                total_num_accepted = 0
                total_num_sampled = 0

                # Sample triplets in batches until enough triplets are accepted (rejection sampling)
                for _ in range(50):
                    if total_num_accepted >= importance_sample_size:
                        break
                    remaining_samples = importance_sample_size - total_num_accepted

                    # Compute the required sample size to have a high probability of accepting the required number of triplets
                    sample_size = min(int(remaining_samples / self.cached_acceptance_rate * 1.5), 200_000)

                    # Sample random triplets uniformly at random from the set of all possible triplets
                    sampled_indices = torch.randint(0, total_unordered_triplet_combinations, (sample_size,), device=device)
                    triplets = self.decode_triplet_indices_from_flattened_indices(sampled_indices, n)
                    i, j, k = triplets[:, 0], triplets[:, 1], triplets[:, 2]

                    # Calculate the per-triplet spread for the sampled triplets
                    pairwise_sums_stack = torch.stack(
                        (similarities[i, j] + similarities[i, k],
                        similarities[i, j] + similarities[j, k],
                        similarities[i, k] + similarities[j, k]),
                        dim=1,
                    )
                    triplet_spread_stack = pairwise_sums_stack.max(dim=1).values - pairwise_sums_stack.min(dim=1).values

                    # Calculate the acceptance probability for the triplet candidates
                    acceptance_probability = (triplet_spread_stack / max_min_delta).clamp(min=0.0, max=1.0)

                    # Accept/Reject the candidate triplet
                    mask = torch.rand(sample_size, device=device) <= acceptance_probability
                    accepted_indices = mask.nonzero(as_tuple=True)[0]

                    # Take only as many accepted triplets as needed to reach the target sample size for the loss estimate
                    num_accepted = min(accepted_indices.shape[0], remaining_samples)
                    accepted_triplets.append(triplets[accepted_indices[:num_accepted]])
                    total_num_accepted += num_accepted
                    total_num_sampled += sample_size

                # Update the acceptance rate cache for the next training step
                self.cached_acceptance_rate = max(total_num_accepted / total_num_sampled, 0.01)
                triplets = torch.cat(accepted_triplets, dim=0)
                self.metrics.setdefault("sampling/acceptance_rate", []).append(self.cached_acceptance_rate)
            elif self.hparams.sampling_strategy == "power_weighted":
                # Sample triplets using power-weighted sampling
                accepted_triplets: list[torch.Tensor] = []
                total_num_accepted = 0
                total_num_sampled = 0

                # Sample triplets in batches until enough triplets are accepted (rejection sampling)
                for _ in range(50):
                    if total_num_accepted >= power_weighted_sample_size:
                        break
                    remaining_samples = power_weighted_sample_size - total_num_accepted

                    # Compute the required sample size to have a high probability of accepting the required number of triplets
                    sample_size = min(int(remaining_samples / self.cached_acceptance_rate * 1.5), 200_000)

                    # Sample random triplets uniformly at random from the set of all possible triplets
                    sampled_indices = torch.randint(0, total_unordered_triplet_combinations, (sample_size,), device=device)
                    triplets = self.decode_triplet_indices_from_flattened_indices(sampled_indices, n)
                    i, j, k = triplets[:, 0], triplets[:, 1], triplets[:, 2]

                    # Calculate the per-triplet spread for the sampled triplets
                    pairwise_sums_stack = torch.stack(
                        (similarities[i, j] + similarities[i, k],
                        similarities[i, j] + similarities[j, k],
                        similarities[i, k] + similarities[j, k]),
                        dim=1,
                    )
                    triplet_spread_stack = pairwise_sums_stack.max(dim=1).values - pairwise_sums_stack.min(dim=1).values

                    # Calculate the acceptance probability for the triplet candidates
                    base_prob = (triplet_spread_stack / max_min_delta).clamp(min=0.0, max=1.0)
                    acceptance_probability = base_prob ** alpha

                    # Accept/Reject the candidate triplet
                    mask = torch.rand(sample_size, device=device) <= acceptance_probability
                    accepted_indices = mask.nonzero(as_tuple=True)[0]

                    # Take only as many accepted triplets as needed to reach the target sample size for the loss estimate
                    num_accepted = min(accepted_indices.shape[0], remaining_samples)
                    accepted_triplets.append(triplets[accepted_indices[:num_accepted]])
                    total_num_accepted += num_accepted
                    total_num_sampled += sample_size

                # Update the acceptance rate cache for the next training step
                self.cached_acceptance_rate = max(total_num_accepted / total_num_sampled, 0.01)
                triplets = torch.cat(accepted_triplets, dim=0)
                self.metrics.setdefault("sampling/acceptance_rate", []).append(self.cached_acceptance_rate)
            else:
                raise ValueError(f"Invalid sampling method: {self.hparams.sampling_strategy}")

        # Log the number of triplets and extract the indices
        self.metrics.setdefault("sampling/evaluated_triplets", []).append(triplets.shape[0])
        i, j, k = triplets[:, 0], triplets[:, 1], triplets[:, 2]

        # Compute the triplet cost N_t
        lca_stack = torch.stack((lcas[i, j], lcas[i, k], lcas[j, k]), dim=1)
        lca_max = lca_stack.max(dim=1, keepdim=True).values
        scaled_exp = torch.exp((lca_stack - lca_max) / hyperbolic_loss_temperature)
        softmax_stack = scaled_exp / scaled_exp.sum(dim=1, keepdim=True)

        sim_stack = torch.stack((similarities[i, j], similarities[i, k], similarities[j, k]), dim=1)
        triplet_cost_stack = torch.sum(sim_stack * (1 - softmax_stack), dim=1)

        # Compute the per-triplet min/max triplet cost for normalization
        pairwise_sums_stack = torch.stack(
            (similarities[i, j] + similarities[i, k],
            similarities[i, j] + similarities[j, k],
            similarities[i, k] + similarities[j, k]),
            dim=1,
        )
        triplet_cost_min_stack = pairwise_sums_stack.min(dim=1).values
        triplet_cost_max_stack = pairwise_sums_stack.max(dim=1).values

        # Compute the hyperbolic loss
        if self.hparams.sampling_strategy in [None, "None", "uniform"]:
            triplet_cost_sum = triplet_cost_stack.sum() # sum of N_t
            triplet_cost_min_sum = triplet_cost_min_stack.sum() # sum of N_t^min
            triplet_cost_max_sum = triplet_cost_max_stack.sum() # sum of N_t^max
            similarities_sum = similarities.sum() - n  # sum of all off-diagonal elements

            # Loss variants
            normalized_cost = (triplet_cost_sum - triplet_cost_min_sum) / (triplet_cost_max_sum - triplet_cost_min_sum)
            wang_cost = (triplet_cost_sum / triplet_cost_min_sum) - 1.0
            chami_cost = triplet_cost_sum + similarities_sum

            results = {
                "wang_cost": wang_cost,
                "normalized_cost": normalized_cost,
                "chami_cost": chami_cost,
                "triplet_cost_sum": triplet_cost_sum,
                "triplet_cost_min_sum": triplet_cost_min_sum,
                "triplet_cost_max_sum": triplet_cost_max_sum,
                "similarities_sum": similarities_sum,
            }
        elif self.hparams.sampling_strategy == "importance":
            triplet_cost_range_stack = (triplet_cost_max_stack - triplet_cost_min_stack).clamp(min=1e-15)
            # Detach the denominator gradient such that the optimizer does not focus on maximizing the triplet spread
            triplet_cost_range_stack = triplet_cost_range_stack.detach()
            X_t = (triplet_cost_stack - triplet_cost_min_stack) / triplet_cost_range_stack
            X_t = torch.where(triplet_cost_range_stack > 1e-15, X_t, torch.zeros_like(X_t))
            normalized_cost = X_t.mean()
            results = {"normalized_cost": normalized_cost}
        elif self.hparams.sampling_strategy == "power_weighted":
            triplet_cost_range_stack = (triplet_cost_max_stack - triplet_cost_min_stack).clamp(min=1e-15)

            # Apply the bias correction to numerator and denominator
            num_terms = (triplet_cost_stack - triplet_cost_min_stack) * (triplet_cost_range_stack ** -alpha)
            denom_terms = triplet_cost_range_stack ** (1 - alpha)

            # Zero out numerically unstable triplets where spread is basically zero
            valid_mask = triplet_cost_range_stack > 1e-15
            num_terms = torch.where(valid_mask, num_terms, torch.zeros_like(num_terms))
            denom_terms = torch.where(valid_mask, denom_terms, torch.zeros_like(denom_terms))

            normalized_cost = num_terms.sum() / denom_terms.sum()
            results = {"normalized_cost": normalized_cost}

        # Log the individual costs and cost components
        for key in results.keys():
            self.metrics.setdefault(f"clustering/{key}", []).append(results[key].detach().item())

        # Compute and log the approximation quality
        if self.hparams.log_approximation_diff and not self.hparams.sampling_strategy in [None, "None"] and self.hparams.loss_version in ["normalized", "power_weighted"]:
            # Compute the full cost
            with torch.no_grad():
                all_indices = torch.arange(n, device=device)
                triplets = torch.combinations(all_indices, r=3)
            i, j, k = triplets[:, 0], triplets[:, 1], triplets[:, 2]

            lca_stack = torch.stack((lcas[i, j], lcas[i, k], lcas[j, k]), dim=1)
            lca_max = lca_stack.max(dim=1, keepdim=True).values
            scaled_exp = torch.exp((lca_stack - lca_max) / hyperbolic_loss_temperature)
            softmax_stack = scaled_exp / scaled_exp.sum(dim=1, keepdim=True)

            sim_stack = torch.stack((similarities[i, j], similarities[i, k], similarities[j, k]), dim=1)
            triplet_cost_sum = torch.sum(sim_stack * (1 - softmax_stack), dim=1).sum() # sum of N_t

            pairwise_sums_stack = torch.stack(
                (similarities[i, j] + similarities[i, k],
                similarities[i, j] + similarities[j, k],
                similarities[i, k] + similarities[j, k]),
                dim=1,
            )
            triplet_cost_min_sum = pairwise_sums_stack.min(dim=1).values.sum() # sum of N_t^min
            triplet_cost_max_sum = pairwise_sums_stack.max(dim=1).values.sum() # sum of N_t^max

            true_normalized_cost = (triplet_cost_sum - triplet_cost_min_sum) / (triplet_cost_max_sum - triplet_cost_min_sum)
            self.metrics.setdefault("sampling/true_normalized_cost", []).append(true_normalized_cost.detach().item())

            # Log the approximation error
            self.metrics.setdefault("sampling/normalized_cost_approximation_diff", []).append((results["normalized_cost"] - true_normalized_cost).abs().detach().item())

            # Compute the cosine similarity between the loss gradients of sampling vs. all triplets (=full cost)
            sampling_grad = torch.autograd.grad(results["normalized_cost"], embeddings, retain_graph=True, create_graph=False)[0]
            full_grad = torch.autograd.grad(true_normalized_cost, embeddings, retain_graph=True, create_graph=False)[0]
            gradient_cos_sim = F.cosine_similarity(sampling_grad.reshape(1, -1), full_grad.reshape(1, -1)).item()
            self.metrics.setdefault("sampling/gradient_cosine_similarity", []).append(gradient_cos_sim)

        return {
            "normalized_cost": normalized_cost,
            "similarity_min": min_sim.detach(),
            "similarity_max": max_sim.detach(),
            "acceptance_rate": torch.tensor(self.cached_acceptance_rate, device=device),
            "evaluated_triplets": torch.tensor(triplets.shape[0], device=device),
        }

    def compute_hyperbolic_loss(self, embeddings: torch.Tensor, hyperbolic_loss_temperature: float, similarities: torch.Tensor | None = None) -> torch.Tensor:
        n = embeddings.size(0)
        num_all_triplets = n * (n - 1) * (n - 2) // 6

        # Compute the minimum required sample size for importance sampling
        eps = self.hparams.sampling_eps
        delta = self.hparams.sampling_delta
        importance_sample_size = int(math.ceil((0.5 + 2/3 * eps) * math.log(2.0 / delta) / (eps ** 2)))

        use_full_triplet_cost = self.hparams.sampling_strategy in [None, "None"] or num_all_triplets <= importance_sample_size

        if use_full_triplet_cost:
            if similarities is None:
                # Compute the pairwise similarities from hyperbolic distances
                distances = compute_pairwise_distances(embeddings, self.manifold)
                similarities = torch.reciprocal(1 + distances)
            # Compute the full cost
            results = self.compute_full_loss(embeddings, similarities, hyperbolic_loss_temperature)
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
            results = self.compute_importance_sampling_loss(embeddings, importance_sample_size, hyperbolic_loss_temperature, similarities=similarities)

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

        manifold_z = self.encode(x)
        if not torch.isfinite(manifold_z).all():
            print("Warning: Non-finite values detected in manifold_z embeddings.")
            exit()

        # Hyperbolic loss
        ## Compute the current hyperbolic loss temperature
        if self.hparams.hyperbolic_loss_temperature_interval > 0:
            hyperbolic_loss_temperature = self.hparams.hyperbolic_loss_start_temperature + (
                self.hparams.hyperbolic_loss_end_temperature - self.hparams.hyperbolic_loss_start_temperature
            ) * min(self.current_epoch / self.hparams.hyperbolic_loss_temperature_interval, 1.0)
        else:
            hyperbolic_loss_temperature = self.hparams.hyperbolic_loss_end_temperature
        self.metrics["clustering/hyperbolic_loss_temperature"] = hyperbolic_loss_temperature

        # Optionally replace the similarity matrix with backbone SNN similarities.
        # The backbone features (before hyperbolic projection) carry richer semantic structure;
        # SNN Jaccard similarity in that space gives a more stable, informative signal.
        snn_similarities = None
        if self.hparams.use_backbone_knn_similarities:
            with torch.no_grad():
                backbone_features = self.encoder(x)  # type: ignore
                backbone_features = self.rms(backbone_features)  # type: ignore
            snn_similarities = self.compute_snn_similarities(backbone_features, k=self.hparams.snn_k)

        ## Compute the current hyperbolic loss weight
        in_head_warmup = self.current_epoch < int(getattr(self.hparams, "head_warmup_epochs", 0))
        if in_head_warmup:
            hyperbolic_loss_weight = 0.0
        elif self.hparams.hyperbolic_loss_weight_interval > 0:
            hyperbolic_loss_weight = self.hparams.hyperbolic_loss_start_weight + (
                self.hparams.hyperbolic_loss_end_weight - self.hparams.hyperbolic_loss_start_weight
            ) * min(self.current_epoch / self.hparams.hyperbolic_loss_weight_interval, 1.0)
        else:
            hyperbolic_loss_weight = self.hparams.hyperbolic_loss_end_weight
        self.metrics["clustering/hyperbolic_loss_weight"] = hyperbolic_loss_weight

        # Skip HypHC computation entirely when its weight is zero — avoids the
        # Euclidean LCA error path and saves work when running ablations/warmups.
        if hyperbolic_loss_weight == 0.0:
            hyperbolic_loss = torch.zeros((), device=manifold_z.device)
        else:
            hyperbolic_loss = self.compute_hyperbolic_loss(manifold_z, hyperbolic_loss_temperature, similarities=snn_similarities)
        self.metrics.setdefault("clustering/hyperbolic_loss", []).append(hyperbolic_loss.detach().item())  # type: ignore
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
            # During head warmup, route contrastive through the hyperbolic head
            # so its gradient trains the head (Euclidean path skips the head entirely).
            use_euclidean_contrastive = self.hparams.contrastive_euclidean and not in_head_warmup
            if use_euclidean_contrastive:
                # Apply contrastive on pre-hyperbolic features — geodesic NT-Xent pushes
                # negatives to the Poincaré boundary and destroys neighborhood structure.
                h_aug_1 = self.rms(self.encoder(x_aug_1))  # type: ignore
                h_aug_2 = self.rms(self.encoder(x_aug_2))  # type: ignore
                contrastive_loss = self._compute_euclidean_contrastive_loss(
                    h_aug_1, h_aug_2, temperature=self.hparams.contrastive_temperature
                )
            else:
                manifold_z_aug_1 = self.encode(x_aug_1)
                manifold_z_aug_2 = self.encode(x_aug_2)
                contrastive_loss = self.compute_contrastive_loss(manifold_z_aug_1, manifold_z_aug_2, temperature=self.hparams.contrastive_temperature)
            # Log the contrastive loss
            self.metrics.setdefault("clustering/contrastive_loss", []).append(contrastive_loss.detach().item())
            # Add the weighted contrastive loss to the total loss
            total_loss += self.hparams.contrastive_weight * contrastive_loss

        # Prevent boundary / dimensional collapse
        if self.hparams.regularize_to_origin_enabled:
            regularizer_weight: float = getattr(self.hparams, 'regularizer_weight', 0.05)  # type: ignore
            reg_type: str = getattr(self.hparams, 'regularizer_type', 'origin')  # type: ignore
            if reg_type == "origin":
                # Pull embeddings toward origin — works but fights hierarchy depth
                reg_loss = self.manifold.dist_0(manifold_z).mean()
            elif reg_type == "uniformity":
                # Wang & Isola uniformity on L2-normalized tangent vectors.
                # Angular-only: leaves radius (hierarchy depth) free. Motivated by
                # Yang et al. 2024 "outer shell isotropy" for hyperbolic contrastive.
                tan = self.manifold.logmap_0(manifold_z)
                tan = F.normalize(tan, p=2, dim=-1)
                sq_pdist = torch.pdist(tan, p=2).pow(2)
                reg_loss = sq_pdist.mul(-2.0).exp().mean().clamp_min(1e-12).log()
            else:
                reg_loss = torch.zeros((), device=manifold_z.device)
            self.metrics.setdefault(f"clustering/reg_{reg_type}", []).append(reg_loss.detach().item())
            total_loss += reg_loss * regularizer_weight

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
