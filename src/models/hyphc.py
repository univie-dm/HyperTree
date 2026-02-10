import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from typing import Any, Tuple
from lightning.pytorch.utilities import grad_norm

from src.hyperbolic_math.src.manifolds import Euclidean
from src.models.abstract import AbstractModel
from src.models.utils import get_dtype_from_precision, get_hyperbolic_encoding_layer_by_name
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
        triplet_sampling_strategy: str = "random_with_replacement",
        triplet_sample_size: str | int = 'auto',
        lca_version: str = "ours",
        loss_version: str = "wang",
        similarity_pruning_version: str = "percentile_pruning",
        similarity_pruning_threshold: float = 0.01,
        similarity_pruning_quantile: float = 0.2,
        similarity_pruning_nearest_neighbors: float = 10,
        # Contrastive Loss parameters
        contrastive_loss_enabled: bool = False,
        contrastive_temperature: float = 0.5,
        contrastive_weight: float = 1.0,
        # Reconstruction Loss parameters
        recon_weight: float = 1.0,
        # Regularizer parameters
        regularize_to_origin_enabled: bool = False,
        regularizer_weight: float = 0.05,
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
            curvature_magnitude=model.manifold.c
            if isinstance(model.manifold.c, torch.Tensor)
            else torch.tensor(model.manifold.c),
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

    def sample_triplets(self, num_pts: int, device: torch.device, sim_bounds: Tuple[float, float] | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample triplets for the Hyperbolic loss."""
        # Note: sampled_indices need to be sorted for tensor lookup in compute_hyperbolic_loss (torch.unique sorts by default)
        if self.hparams.triplet_sample_size == 'auto':
            # TODO: Calculate the minimum sampling size based on Theorem
            minimum_sample_size = 1000 # placeholder
            if self.hparams.triplet_sampling_strategy == "importance_sampling":
                # Importance sampling
                random_indices = torch.randint(0, num_pts*(num_pts-1)*(num_pts-2),
                                               (min(minimum_sample_size, num_pts),), device=device) ## TODO: needs adjustment
                sampled_triplet_indices = self.decode_triplet_indices_from_flattened_indices(random_indices, num_pts)
                sampled_indices = torch.unique(sampled_triplet_indices)
            elif self.hparams.triplet_sampling_strategy == "random_with_replacement":
                random_indices = torch.randint(0, num_pts*(num_pts-1)*(num_pts-2),
                                               (min(minimum_sample_size, num_pts),), device=device)
                sampled_triplet_indices = self.decode_triplet_indices_from_flattened_indices(random_indices, num_pts)
                sampled_indices = torch.unique(sampled_triplet_indices)
            else:
                raise ValueError(f"Invalid triplet sampling strategy for auto sample size: {self.hparams.triplet_sampling_strategy}")
        elif self.hparams.triplet_sampling_strategy == "all_subsample_triplets":
            sample_size: int = self.hparams.triplet_sample_size if isinstance(self.hparams.triplet_sample_size, int) else 256  # type: ignore
            sampled_indices = torch.randperm(num_pts, device=device)[:min(sample_size, num_pts)].sort().values
            sampled_triplet_indices = torch.combinations(sampled_indices, r=3)
        elif self.hparams.triplet_sampling_strategy == "random_with_replacement":
            sample_size: int = self.hparams.triplet_sample_size if isinstance(self.hparams.triplet_sample_size, int) else num_pts**2  # type: ignore
            random_indices = torch.randint(0, num_pts*(num_pts-1)*(num_pts-2), (min(sample_size, num_pts),), device=device)
            sampled_triplet_indices = self.decode_triplet_indices_from_flattened_indices(random_indices, num_pts)
            sampled_indices = torch.unique(sampled_triplet_indices)
        elif self.hparams.triplet_sampling_strategy == "chami":
            sample_size: int = self.hparams.triplet_sample_size if isinstance(self.hparams.triplet_sample_size, int) else num_pts  # type: ignore
            sampled_indices = torch.randperm(num_pts, device=device)[:min(sample_size, num_pts)]
            sampled_pair_indices = torch.combinations(sampled_indices, r=2)
            # Append a third random point to the torch tensor of 2-tuples
            random_indices = torch.randint(0, num_pts, (sampled_pair_indices.shape[0], 1), device=device)
            sampled_triplet_indices = torch.cat([sampled_pair_indices, random_indices], dim=1)
            sampled_indices = torch.unique(sampled_triplet_indices)
        else:
            raise ValueError(f"Invalid triplet sampling strategy: {self.hparams.triplet_sampling_strategy}")
        return sampled_indices, sampled_triplet_indices

    def decode_triplet_indices_from_flattened_indices(self, indices, n):
        # Indices: Tensor of shape (m,) with integer IDs in [0, n*(n-1)*(n-2)-1]
        # Decode the triplets (i, j, k) from the integer IDs
        ## Decode the first index 'i' from n options
        batch_n = n
        i = indices % batch_n
        temp = indices // batch_n
        ## Decode the second index 'j' from (n-1) options
        j = temp % (batch_n - 1)
        ## Decode the third index 'k' from (n-2) options
        k = temp // (batch_n - 1) % (batch_n - 2)
        # Adjust j to avoid collision with i (i == j)
        j = j + (j >= i).long()
        # Adjust k to avoid collisions with i and j
        low = torch.min(i, j)
        high = torch.max(i, j)
        k = k + (k >= low).long() + (k >= high - 1).long()
        return torch.stack([i, j, k], dim=1)

    def symmetric_percentile_pruning(self, similarities, percentile):
        """Prunes a symmetric matrix by calculating the quantile on unique off-diagonal edges."""
        n = similarities.size(0)
        indices = torch.triu_indices(n, n, offset=1, device=similarities.device)
        unique_edges = similarities[indices[0], indices[1]]
        threshold = torch.quantile(unique_edges, percentile)
        condition = similarities < threshold
        similarities = torch.where(condition, torch.zeros_like(similarities), similarities)
        # Log how many edges are pruned and the density of the remaining graph
        self.metrics.setdefault("clustering/pruned_edge_percent", []).append(condition.float().mean().item()*100)  # type: ignore
        self.metrics.setdefault("clustering/pruning_threshold", []).append(threshold.item())  # type: ignore
        return similarities

    def symmetric_knn_pruning(self, similarities, k):
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

    def compute_hyperbolic_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        n = embeddings.size(0)
        device = embeddings.device

        if self.hparams.triplet_sample_size == 'auto':
            # TODO: needs adjustment for pruning etc.
            # Calculate the minimum sampling size based on the min-/max-pairwise similarities
            distances = compute_pairwise_distances(embeddings, self.manifold)
            similarities = torch.reciprocal(1 + distances)
            similarities = torch.where(similarities <= self.hparams.similarity_pruning_threshold, torch.zeros_like(similarities), similarities)
            upper_triu_mask = torch.triu(torch.ones_like(similarities), diagonal=1).bool()
            min_sim = similarities[upper_triu_mask].min()
            max_sim = similarities[upper_triu_mask].max()
            # Sample triplets
            sampled_indices, sampled_triplet_indices = self.sample_triplets(n, device, sim_bounds=(min_sim.item(), max_sim.item()))  # type: ignore
            # Extract similarities for sampled indices
            similarities = similarities[sampled_indices][:, sampled_indices]
        else:
            # Sample triplets
            sampled_indices, sampled_triplet_indices = self.sample_triplets(n, device)
            # Compute the (smoothed) pairwise distances
            distances = compute_pairwise_distances(embeddings[sampled_indices], self.manifold)
            similarities = torch.reciprocal(1 + distances)
            # Prune low similarities
            if self.hparams.similarity_pruning_version == "percentile_pruning":
                similarities = self.symmetric_percentile_pruning(similarities, percentile=self.hparams.similarity_pruning_quantile)
            elif self.hparams.similarity_pruning_version == "knn_pruning":
                similarities = self.symmetric_knn_pruning(similarities, k=self.hparams.similarity_pruning_nearest_neighbors)
            elif self.hparams.similarity_pruning_version == "threshold_pruning":
                similarities = torch.where(similarities <= self.hparams.similarity_pruning_threshold, torch.zeros_like(similarities), similarities)
            else:
                raise ValueError(f"Invalid similarity pruning version: {self.hparams.similarity_pruning_version}")

        # Compute the pairwise lcas
        if not self.hparams.lca_version in ["ours", "chami"]:
            raise ValueError(f"Invalid LCA version: {self.hparams.lca_version}")
        lcas = self.compute_lca_depth(embeddings[sampled_indices], version=self.hparams.lca_version)

        # Keep track of the storage of triplet indices in embeddings[sampled_indices]
        # Note: sampled_indices need to be sorted for efficient indexing (torch.unique sorts by default)
        positions = torch.searchsorted(sampled_indices, sampled_triplet_indices.reshape(-1)).reshape(-1, 3)
        self.metrics.setdefault("clustering/evaluated_triplets", []).append(positions.shape[0])  # type: ignore
        i, j, k = positions.T  # positions[:, 0], positions[:, 1], positions[:, 2]

        # Compute the scaled safe softmax of the lcas
        lca_stack = torch.stack((lcas[i, j], lcas[i, k], lcas[j, k]), dim=1)
        lca_triplet_max, _ = torch.max(lca_stack, dim=1, keepdim=True)
        scaled_softmax = torch.exp((lca_stack - lca_triplet_max) / self.hparams.hyperbolic_loss_temperature)
        softmax_stack = scaled_softmax / scaled_softmax.sum(dim=1, keepdim=True)

        # Compute the triplet cost based on the sampled triplets
        sim_stack = torch.stack((similarities[i, j], similarities[i, k], similarities[j, k]), dim=1)
        triplet_cost = torch.sum(sim_stack * (1 - softmax_stack))

        # Compute the maximum and minimum triplet cost
        pairwise_sim_stack = torch.stack(
            (similarities[i, j] + similarities[i, k],
             similarities[i, j] + similarities[j, k],
             similarities[i, k] + similarities[j, k]),
            dim=1,
        )
        pairwise_sim_stack_min, _ = torch.min(pairwise_sim_stack, dim=1, keepdim=False)
        pairwise_sim_stack_max, _ = torch.max(pairwise_sim_stack, dim=1, keepdim=False)
        triplet_cost_min = pairwise_sim_stack_min.sum()
        triplet_cost_max = pairwise_sim_stack_max.sum()

        # Compute the pairwise similarities sum (only used in chami loss)
        # Note: this term needs adjustment based on the triplet sampling strategy employed
        similarities_sum = similarities.sum() - similarities.shape[0] # sum of all pairwise similarities + main diagonal twice

        # Compute different versions of the loss
        chami_cost = triplet_cost + similarities_sum
        normalized_cost = (triplet_cost - triplet_cost_min) / (triplet_cost_max - triplet_cost_min)
        wang_cost = triplet_cost / triplet_cost_min

        # Log the individual cost components
        self.metrics.setdefault("clustering/triplet_cost", []).append(triplet_cost.detach().item())
        self.metrics.setdefault("clustering/triplet_cost_min", []).append(triplet_cost_min.detach().item())
        self.metrics.setdefault("clustering/triplet_cost_max", []).append(triplet_cost_max.detach().item())
        self.metrics.setdefault("clustering/max_min_delta", []).append((triplet_cost_max - triplet_cost_min).detach().item())
        self.metrics.setdefault("clustering/similarities_sum", []).append(similarities_sum.detach().item())
        # Log the chami, normalized, and wang costs
        self.metrics.setdefault("clustering/chami_cost", []).append(chami_cost.detach().item())
        self.metrics.setdefault("clustering/normalized_cost", []).append(normalized_cost.detach().item())
        self.metrics.setdefault("clustering/wang_cost", []).append(wang_cost.detach().item())

        # Return the specified loss
        if self.hparams.loss_version == "chami":
            return chami_cost
        elif self.hparams.loss_version == "normalized":
            return normalized_cost
        elif self.hparams.loss_version == "wang":
            return wang_cost
        else:
            raise ValueError(f"Invalid loss version: {self.hparams.loss_version}")

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
        x_aug_1, _, x, _ = batch
        x_aug_1 = x_aug_1.to(dtype=get_dtype_from_precision(self.hparams.params_dtype))
        x = x.to(dtype=get_dtype_from_precision(self.hparams.params_dtype))

        # Hyperbolic loss
        manifold_z = self.encode(x)
        hyperbolic_loss = self.compute_hyperbolic_loss(manifold_z)
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

        # Contrastive loss
        if self.hparams.contrastive_weight > 0.0 and self.hparams.contrastive_loss_enabled:
            manifold_z_aug_1 = self.encode(x_aug_1)
            contrastive_loss = self.compute_contrastive_loss(manifold_z, manifold_z_aug_1, temperature=self.hparams.contrastive_temperature)
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
