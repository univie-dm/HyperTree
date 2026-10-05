import math
import os
import time
import torch
import torch.nn.functional as F

from lightning.pytorch.utilities import grad_norm

from src.models.abstract import AbstractModel
from src.models.utils import get_dtype_from_precision
from src.hyperbolic_math.src.utils.helpers import compute_pairwise_distances
from src.utils.lca import compute_lca_depth

# An image has three views: the un-augmented original and two augmented views.
# Views forming the positive pair of the contrastive loss
CONTRASTIVE_VIEWS = {"aug_aug": ("aug_1", "aug_2"), "orig_aug": ("original", "aug_1")}
# View the hierarchical loss is computed on
HIERARCHICAL_VIEWS = {"original": "original", "augmented": "aug_1"}


def get_training_views(
    hierarchical_loss_view: str, contrastive_positive_pair: str, use_contrastive_loss: bool
) -> tuple[str, tuple[str, ...]]:
    """Returns the view of the hierarchical loss and the views of the contrastive loss (none if it is not used).

    These are the views a training step consumes, i.e. the ones the dataloader has to provide.
    """
    hierarchical_view = HIERARCHICAL_VIEWS[hierarchical_loss_view]
    contrastive_views = CONTRASTIVE_VIEWS[contrastive_positive_pair] if use_contrastive_loss else ()
    return hierarchical_view, contrastive_views


class HyperTreeModel(AbstractModel):
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
        log_interval: int = 10,
        evaluation_metrics: dict[str, bool] = {},
        # Hierarchical Loss parameters
        hierarchical_loss_start_weight: float = 1.0,
        hierarchical_loss_end_weight: float = 3.0,
        hierarchical_loss_weight_interval: int = 300,
        hierarchical_loss_start_temperature: float = 0.2,
        hierarchical_loss_end_temperature: float = 0.1,
        hierarchical_loss_temperature_interval: int = 250,
        hierarchical_loss_view: str = "original",
        lca_version: str = "rectified",
        sampling_eps: float = 1e-3,
        sampling_delta: float = 1e-2,
        # Backbone kNN SNN similarity parameters
        use_backbone_knn_similarities: bool = False,
        snn_k: int = 10,
        # Contrastive Loss parameters
        contrastive_loss_enabled: bool = False,
        contrastive_temperature: float = 0.5,
        contrastive_weight: float = 1.0,
        contrastive_positive_pair: str = "aug_aug",
        # Encoder
        freeze_encoder: bool = False,
        train_projection_head: bool = False,
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
        )
        # Save hyperparameters while ignoring non-serializable model
        self.save_hyperparameters(ignore=["model"])

        # Reuse the pretrained encoding modules
        self.encoder = model.encoder  # type: ignore
        self.rms = model.rms  # type: ignore
        if hasattr(model, "feature_scaling_layer"):
            self.feature_scaling_layer = model.feature_scaling_layer  # type: ignore
        self.hyperbolic_encoding_layer = model.hyperbolic_encoding_layer  # type: ignore

        # Hierarchical loss scheduler
        if hierarchical_loss_end_weight < 0 or hierarchical_loss_start_weight < 0:
            raise ValueError("hierarchical_loss_start_weight and hierarchical_loss_end_weight must be non-negative.")
        if hierarchical_loss_weight_interval < 0:
            raise ValueError("hierarchical_loss_weight_interval must be non-negative.")
        if hierarchical_loss_start_weight > hierarchical_loss_end_weight:
            raise ValueError("hierarchical_loss_start_weight must be less than or equal to hierarchical_loss_end_weight.")
        if hierarchical_loss_view not in HIERARCHICAL_VIEWS:
            raise ValueError(f"hierarchical_loss_view must be one of {list(HIERARCHICAL_VIEWS)}, got '{hierarchical_loss_view}'")
        if contrastive_positive_pair not in CONTRASTIVE_VIEWS:
            raise ValueError(f"contrastive_positive_pair must be one of {list(CONTRASTIVE_VIEWS)}, got '{contrastive_positive_pair}'")

        # Freeze the encoder if requested: the backbone and, unless train_projection_head is set,
        # the projection head (a frozen projection head keeps SNN similarities stable across batches).
        # Frozen modules are also kept in eval mode, see AbstractModel.train().
        self.frozen_modules = list(model.frozen_modules)  # e.g. an externally pretrained backbone
        if freeze_encoder:
            for param in self.encoder.parameters():  # type: ignore
                param.requires_grad_(False)
            if train_projection_head:
                # encoder = Sequential(backbone, *projection head layers)
                for param in self.encoder[1:].parameters():  # type: ignore
                    param.requires_grad_(True)
                self.frozen_modules.append(self.encoder[0])  # type: ignore
            else:
                self.frozen_modules.append(self.encoder)  # type: ignore
        self.train()  # frozen modules go to eval mode

        # Acceptance rate of the rejection sampler (re-estimated each training step)
        self.cached_acceptance_rate = 0.2

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

    @torch.no_grad()
    def sample_triplets(self, similarities: torch.Tensor, sample_size: int) -> torch.Tensor:
        """Draw triplets t with probability proportional to N_t^max - N_t^min (importance sampling).

        Rejection sampling: candidates are drawn uniformly at random with replacement and accepted
        with probability (N_t^max - N_t^min) / (w_max - w_min), where w_min and w_max are the
        smallest and largest pairwise similarities of the batch.
        """
        n = similarities.size(0)
        device = similarities.device
        total_unordered_triplet_combinations = n * (n - 1) * (n - 2)

        # Compute the pairwise min/max similarities
        triu_idx = torch.triu_indices(n, n, offset=1, device=device)
        upper_sims = similarities[triu_idx[0], triu_idx[1]]
        min_sim = upper_sims.min()
        max_sim = upper_sims.max()
        max_min_delta = max_sim - min_sim
        self.metrics.setdefault("sampling/similarities_min", []).append(min_sim)
        self.metrics.setdefault("sampling/similarities_max", []).append(max_sim)
        self.metrics.setdefault("sampling/similarities_max_min_delta", []).append(max_min_delta)

        accepted_triplets: list[torch.Tensor] = []
        _prof = os.environ.get("HYP_PROFILE_SAMPLER")
        if _prof:
            if device.type == "cuda":
                torch.cuda.synchronize()
            _t_sampler = time.perf_counter()
        total_num_accepted = 0
        total_num_sampled = 0

        # Sample triplets in batches until enough triplets are accepted
        for _ in range(50):
            if total_num_accepted >= sample_size:
                break
            remaining_samples = sample_size - total_num_accepted

            # Compute the required number of candidates to have a high probability of accepting the required number of triplets
            num_candidates = min(int(remaining_samples / self.cached_acceptance_rate * 1.5), 200_000)

            # Sample random triplets uniformly at random from the set of all possible triplets
            sampled_indices = torch.randint(0, total_unordered_triplet_combinations, (num_candidates,), device=device)
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
            mask = torch.rand(num_candidates, device=device) <= acceptance_probability
            accepted_indices = mask.nonzero(as_tuple=True)[0]

            # Take only as many accepted triplets as needed to reach the target sample size for the loss estimate
            num_accepted = min(accepted_indices.shape[0], remaining_samples)
            accepted_triplets.append(triplets[accepted_indices[:num_accepted]])
            total_num_accepted += num_accepted
            total_num_sampled += num_candidates

        if _prof:
            if device.type == "cuda":
                torch.cuda.synchronize()
            self.metrics.setdefault("profile/sampler_ms", []).append(
                (time.perf_counter() - _t_sampler) * 1000.0)
        # Update the acceptance rate cache for the next training step
        self.cached_acceptance_rate = max(total_num_accepted / total_num_sampled, 0.01)
        self.metrics.setdefault("sampling/acceptance_rate", []).append(self.cached_acceptance_rate)
        self.metrics.setdefault("sampling/candidate_triplets", []).append(total_num_sampled)
        return torch.cat(accepted_triplets, dim=0)

    def compute_triplet_costs(
        self, embeddings: torch.Tensor, similarities: torch.Tensor, triplets: torch.Tensor, hierarchical_loss_temperature: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the triplet cost N_t and its bounds N_t^min and N_t^max for the given triplets."""
        n = embeddings.size(0)
        i, j, k = triplets[:, 0], triplets[:, 1], triplets[:, 2]

        # Compute the LCA depths only for the distinct pairs that occur in the triplets
        pair_rows = torch.cat((i, i, j))
        pair_cols = torch.cat((j, k, k))
        pair_ids = torch.minimum(pair_rows, pair_cols) * n + torch.maximum(pair_rows, pair_cols)
        unique_pair_ids, pair_positions = torch.unique(pair_ids, return_inverse=True)
        lca_depths = compute_lca_depth(
            embeddings, unique_pair_ids // n, unique_pair_ids % n, self.manifold, version=self.hparams.lca_version
        )
        self.metrics.setdefault("sampling/evaluated_lca_pairs", []).append(unique_pair_ids.shape[0])

        # Compute the triplet cost N_t
        lca_stack = lca_depths[pair_positions].view(3, -1).T  # columns: LCA depths of (i, j), (i, k), (j, k)
        lca_max = lca_stack.max(dim=1, keepdim=True).values
        scaled_exp = torch.exp((lca_stack - lca_max) / hierarchical_loss_temperature)
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
        return triplet_cost_stack, triplet_cost_min_stack, triplet_cost_max_stack

    def compute_hierarchical_loss(self, embeddings: torch.Tensor, similarities: torch.Tensor, hierarchical_loss_temperature: float) -> torch.Tensor:
        """Compute the normalized hierarchical loss, estimated via importance sampling of triplets."""
        n = embeddings.size(0)
        num_all_triplets = n * (n - 1) * (n - 2) // 6

        # Compute the minimum required sample size for importance sampling
        eps = self.hparams.sampling_eps
        delta = self.hparams.sampling_delta
        importance_sample_size = int(math.ceil((0.5 + 2/3 * eps) * math.log(2.0 / delta) / (eps ** 2)))

        if num_all_triplets <= importance_sample_size:
            # The batch holds fewer triplets than the sampling budget: compute the exact loss over all triplets
            triplets = torch.combinations(torch.arange(n, device=embeddings.device), r=3)
            triplet_cost_stack, triplet_cost_min_stack, triplet_cost_max_stack = self.compute_triplet_costs(
                embeddings, similarities, triplets, hierarchical_loss_temperature
            )
            triplet_cost_min = triplet_cost_min_stack.sum()
            normalized_cost = (triplet_cost_stack.sum() - triplet_cost_min) / (triplet_cost_max_stack.sum() - triplet_cost_min).clamp(min=1e-15)
        else:
            # Approximate the loss with importance sampling
            triplets = self.sample_triplets(similarities, importance_sample_size)
            triplet_cost_stack, triplet_cost_min_stack, triplet_cost_max_stack = self.compute_triplet_costs(
                embeddings, similarities, triplets, hierarchical_loss_temperature
            )
            triplet_cost_range_stack = (triplet_cost_max_stack - triplet_cost_min_stack).clamp(min=1e-15)
            # Detach the denominator gradient such that the optimizer does not focus on maximizing the triplet spread
            triplet_cost_range_stack = triplet_cost_range_stack.detach()
            X_t = (triplet_cost_stack - triplet_cost_min_stack) / triplet_cost_range_stack
            X_t = torch.where(triplet_cost_range_stack > 1e-15, X_t, torch.zeros_like(X_t))
            normalized_cost = X_t.mean()

        self.metrics.setdefault("sampling/evaluated_triplets", []).append(triplets.shape[0])
        return normalized_cost

    def on_train_batch_start(self, batch, batch_idx):  # noqa: D102
        if os.environ.get("HYP_PROFILE_SAMPLER"):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._prof_step_t0 = time.perf_counter()

    def on_train_batch_end(self, outputs, batch, batch_idx):  # noqa: D102
        if os.environ.get("HYP_PROFILE_SAMPLER") and hasattr(self, "_prof_step_t0"):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _step_ms = (time.perf_counter() - self._prof_step_t0) * 1000.0
            self.metrics.setdefault("profile/step_ms", []).append(_step_ms)
            _s = self.metrics.get("profile/sampler_ms", [])
            if _s:
                print(f"PROF sampler_ms={_s[-1]:.3f} step_ms={_step_ms:.3f}", flush=True)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        params_dtype = get_dtype_from_precision(self.hparams.params_dtype)

        use_contrastive_loss = self.hparams.contrastive_weight > 0.0 and self.hparams.contrastive_loss_enabled
        hierarchical_view, contrastive_views = get_training_views(
            self.hparams.hierarchical_loss_view, self.hparams.contrastive_positive_pair, use_contrastive_loss
        )

        # Encode every required view exactly once: view -> (backbone features, manifold embeddings)
        encoded = {
            view: self.encode_with_backbone(batch[view].to(dtype=params_dtype))
            for view in dict.fromkeys((hierarchical_view, *contrastive_views))
        }
        backbone_features, manifold_z = encoded[hierarchical_view]
        if not torch.isfinite(manifold_z).all():
            raise ValueError("Non-finite values detected in manifold_z embeddings.")

        # Contrastive loss
        contrastive_loss = None
        distances = None
        if use_contrastive_loss:
            manifold_z_1, manifold_z_2 = (encoded[view][1] for view in contrastive_views)
            contrastive_loss, contrastive_distances = self.compute_contrastive_loss(
                manifold_z_1, manifold_z_2, temperature=self.hparams.contrastive_temperature
            )
            # Log the contrastive loss
            self.metrics.setdefault("clustering/contrastive_loss", []).append(contrastive_loss.detach().item())
            # The pairwise distances of a contrastive view are part of the contrastive distance
            # matrix, so the hierarchical loss reuses them instead of recomputing them
            if hierarchical_view in contrastive_views:
                distances = self._local_view_distances(
                    contrastive_distances, contrastive_views.index(hierarchical_view), manifold_z.shape[0]
                )

        # Hierarchical loss
        ## Compute the current hierarchical loss temperature
        if self.hparams.hierarchical_loss_temperature_interval > 0:
            hierarchical_loss_temperature = self.hparams.hierarchical_loss_start_temperature + (
                self.hparams.hierarchical_loss_end_temperature - self.hparams.hierarchical_loss_start_temperature
            ) * min(self.current_epoch / self.hparams.hierarchical_loss_temperature_interval, 1.0)
        else:
            hierarchical_loss_temperature = self.hparams.hierarchical_loss_end_temperature
        self.metrics["clustering/hierarchical_loss_temperature"] = hierarchical_loss_temperature

        ## Compute the current hierarchical loss weight
        if self.hparams.hierarchical_loss_weight_interval > 0:
            hierarchical_loss_weight = self.hparams.hierarchical_loss_start_weight + (
                self.hparams.hierarchical_loss_end_weight - self.hparams.hierarchical_loss_start_weight
            ) * min(self.current_epoch / self.hparams.hierarchical_loss_weight_interval, 1.0)
        else:
            hierarchical_loss_weight = self.hparams.hierarchical_loss_end_weight
        self.metrics["clustering/hierarchical_loss_weight"] = hierarchical_loss_weight

        # Skip the hierarchical loss computation entirely when its weight is zero — avoids the
        # Euclidean LCA error path and saves work when running ablations/warmups.
        if hierarchical_loss_weight == 0.0:
            hierarchical_loss = torch.zeros((), device=manifold_z.device)
        else:
            if self.hparams.use_backbone_knn_similarities:
                # Replace the similarity matrix with backbone SNN similarities.
                # The backbone features (before hyperbolic projection) carry richer semantic structure;
                # SNN Jaccard similarity in that space gives a more stable, informative signal.
                similarities = self.compute_snn_similarities(backbone_features.detach(), k=self.hparams.snn_k)
            else:
                # Compute the pairwise similarities from hyperbolic distances
                if distances is None:
                    distances = compute_pairwise_distances(manifold_z, self.manifold)
                similarities = torch.reciprocal(1 + distances)
            hierarchical_loss = self.compute_hierarchical_loss(manifold_z, similarities, hierarchical_loss_temperature)
        self.metrics.setdefault("clustering/hierarchical_loss", []).append(hierarchical_loss.detach().item())  # type: ignore

        total_loss = hierarchical_loss_weight * hierarchical_loss
        if contrastive_loss is not None:
            # Add the weighted contrastive loss to the total loss
            total_loss = total_loss + self.hparams.contrastive_weight * contrastive_loss

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
