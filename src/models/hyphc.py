import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from typing import Any, Tuple
from lightning.pytorch.utilities import grad_norm
import wandb
import matplotlib.pyplot as plt

from src.hyperbolic_math.src.utils.horo_pca import frechet_center_data
from src.hyperbolic_math.src.manifolds import Euclidean, PoincareBall
from src.models.abstract import AbstractModel
from src.models.resnet import HyperbolicResNet
from src.models.utils import (
    get_dtype_from_precision,
    get_hyperbolic_encoding_layer_by_name,
)
from src.hyperbolic_math.src.utils.helpers import compute_pairwise_distances

# CIFAR100 Coarse Labels Mapping
_CIFAR100_COARSE_LABELS = [
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15, 0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
    5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
    10, 3, 2, 12, 12, 16, 12, 1, 9, 19, 2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19, 18, 1, 2, 15, 6, 0, 17, 8, 14, 13,
]


class HyperbolicHCModel(AbstractModel):
    def __init__(
        self,
        ae_model: AbstractModel,
        trainable_curvature: bool = False,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        log_interval: int = 5,
        stage: str = "clustering",
        global_epoch: int = 0,
        lr_scheduler_params: dict | None = None,
        hyphc_start_weight: float = 1.0,
        hyphc_end_weight: float = 1.0,
        hyphc_weight_interval: int = 0,
        hyphc_temperature: float = 0.05,
        hyphc_sampling_strategy: str = "all_subsample_triplets",
        hyphc_fine_triplet_ratio: float = 0.5,
        hyphc_triplet_subsampling_size: int = 100,
        hyphc_triplet_subsampling_delta: float = 0.05,
        lca_version: str = "ours",
        normalized_hyphc_loss: bool = False,
        distances_type: str = "geodesic",
        hyphc_dist_sum_loss: bool = True,
        evaluation_metrics: dict[str, bool] = {},
        freeze_encoder: bool = False,
        contrastive_enabled: bool = False,
        contrastive_weight: float = 1.0,
        contrastive_temperature: float = 0.5,
        triplet_margin: float = 8.0,
        hierarchy_margin: float = 0.1,
        hierarchy_weight: float = 0.2,
        byol: bool = False,
        recon_weight: float = 1.0,
        cluster_dim: int | None = None,
        hyphc_num_nn: int = 10,
        use_seperate_heads: bool = False,
        gt_similarity_mode: str = "none",  # "none", "hierarchical", "hybrid", "pseudo"
        gt_alpha: float = 0.7,  # Influence of GT vs. Geometry for hybrid mode
        s_fine: float = 1.0,  # Weight for same class
        s_coarse: float = 0.4,  # Weight for same superclass, different class
        s_diff: float = 0.05,  # Weight for completely different classes
        similarity_pruning_threshold: float = 0.0,  # Threshold below which similarities are ignored
    ):
        super().__init__(
            model_type=ae_model.model_type,
            hyperbolic_encoding=ae_model.hyperbolic_encoding,
            params_dtype=ae_model.params_dtype,
            latent_dim=ae_model.latent_dim,
            manifold_name=ae_model.manifold_name,
            manifold_precision=ae_model.manifold.dtype,
            curvature_magnitude=ae_model.manifold.c
            if isinstance(ae_model.manifold.c, torch.Tensor)
            else torch.tensor(ae_model.manifold.c),
            trainable_curvature=trainable_curvature,
            hyperbolic_decoding=ae_model.hyperbolic_decoding,
            optimizer_name=ae_model.optimizer_name,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            log_interval=log_interval,
            stage=stage,
            global_epoch=global_epoch,
            distances_type=distances_type,
            evaluation_metrics=evaluation_metrics,
        )
        # Save hyperparameters while ignoring non-serializable ae_model
        self.save_hyperparameters(ignore=["ae_model"])
        self.orginal_input_shape = ae_model.orginal_input_shape
        # HypHC parameters
        self.hyphc_start_weight = hyphc_start_weight
        self.hyphc_end_weight = hyphc_end_weight
        self.hyphc_weight_interval = hyphc_weight_interval
        self.hyphc_temperature = hyphc_temperature
        self.hyphc_sampling_strategy = hyphc_sampling_strategy
        self.hyphc_fine_triplet_ratio = hyphc_fine_triplet_ratio
        self.hyphc_triplet_subsampling_delta = hyphc_triplet_subsampling_delta
        self.hyphc_triplet_subsampling_size = int(hyphc_triplet_subsampling_size)
        self.lca_version = lca_version
        self.gt_similarity_mode = gt_similarity_mode
        self.gt_alpha = gt_alpha
        self.s_fine = s_fine
        self.s_coarse = s_coarse
        self.s_diff = s_diff
        self.similarity_pruning_threshold = similarity_pruning_threshold
        self.normalized_hyphc_loss = normalized_hyphc_loss
        self.distances_type = distances_type
        self.hyphc_dist_sum_loss = hyphc_dist_sum_loss
        self.lr_scheduler_params = lr_scheduler_params
        self.contrastive_enabled = contrastive_enabled
        self.contrastive_weight = contrastive_weight
        self.contrastive_temperature = contrastive_temperature
        self.triplet_margin = triplet_margin
        self.hierarchy_margin = hierarchy_margin
        self.hierarchy_weight = hierarchy_weight
        self.byol = byol
        self.recon_weight = recon_weight
        self.hyphc_num_nn = hyphc_num_nn
        self.use_seperate_heads = use_seperate_heads
        if cluster_dim is None:
            self.cluster_dim = self.latent_dim
        else:
            self.cluster_dim = cluster_dim

        # Load encoding and decoding modules with their weights
        self.encoder = ae_model.encoder  # type: ignore
        self.hyperbolic_encoding_layer = ae_model.hyperbolic_encoding_layer  # type: ignore
        self.hyperbolic_decoding_layer = ae_model.hyperbolic_decoding_layer  # type: ignore
        self.decoder = ae_model.decoder  # type: ignore
        if isinstance(ae_model, HyperbolicResNet):
            self.expansion = ae_model.expansion
        else:
            self.expansion = 1

        # Ensure weights are copied by loading state dict
        self.encoder.load_state_dict(ae_model.encoder.state_dict())  # type: ignore
        self.hyperbolic_encoding_layer.load_state_dict(
            ae_model.hyperbolic_encoding_layer.state_dict()
        )  # type: ignore
        if self.recon_weight != 0.0:
            self.hyperbolic_decoding_layer.load_state_dict(
                ae_model.hyperbolic_decoding_layer.state_dict()
            )  # type: ignore
            self.decoder.load_state_dict(ae_model.decoder.state_dict())  # type: ignore
        else:
            self.decoder = None
            self.hyperbolic_decoding_layer = None

        if hasattr(ae_model, "feature_scaling_layer"):
            self.feature_scaling_layer = ae_model.feature_scaling_layer  # type: ignore
        self.rms = ae_model.rms  # type: ignore

        if freeze_encoder:
            for param in self.encoder.parameters():  # type: ignore
                param.requires_grad = False

        if hyphc_end_weight < 0 or hyphc_start_weight < 0:
            raise ValueError(
                "hyphc_start_weight and hyphc_end_weight must be non-negative."
            )
        if hyphc_weight_interval < 0:
            raise ValueError("hyphc_weight_interval must be non-negative.")
        if hyphc_start_weight > hyphc_end_weight:
            raise ValueError(
                "hyphc_start_weight must be less than or equal to hyphc_end_weight."
            )

        if use_seperate_heads:
            print("Using separate cluster head with hyperbolic manifold.")
            if isinstance(self.manifold, Euclidean):
                self.cluster_head = nn.Linear(
                    self.encoder[-1].out_features,  # type: ignore
                    self.cluster_dim,
                )
            else:
                self.cluster_head = get_hyperbolic_encoding_layer_by_name(
                    layer_name=str(self.hyperbolic_encoding),
                    manifold=self.manifold,
                    input_dim=self.encoder[-1].out_features,  # type: ignore
                    output_dim=self.cluster_dim,
                    params_dtype=self.params_dtype,
                    clamping_factor=self.clamping_factor,
                    smoothing_factor=self.smoothing_factor,
                    learnable_scale=self.learnable_scale,
                    FHCNN_normalize=self.FHCNN_normalize,
                )
                print(f'Cluster head dim: {self.cluster_dim}, Encoder output dim: {self.encoder[-1].out_features}')  # type: ignore
                if self.cluster_dim == self.encoder[-1].out_features: # type: ignore
                    print("Initializing cluster head with AE hyperbolic encoding weights.")
                    self.cluster_head.load_state_dict(
                        ae_model.hyperbolic_encoding_layer.state_dict()
                    )

    def configure_optimizers(self) -> Any:
        if self.optimizer_name == "Adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                eps=1e-5,
                weight_decay=self.weight_decay,
                amsgrad=True,
            )
        elif self.optimizer_name == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f"Invalid optimizer name: {self.optimizer_name}")

        if self.lr_scheduler_params is not None:
            if self.lr_scheduler_params["type"] is None:
                return optimizer
            if self.lr_scheduler_params["type"] == "warmup_cosine":
                scheduler = CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=self.lr_scheduler_params["interval"],
                    eta_min=self.lr_scheduler_params["cosine_min_factor"]
                    * self.learning_rate,
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

    def _calculate_probabilistic_coverage_samples(self, n: int) -> int:
        """
        Calculates the number of triplets to sample for probabilistic coverage.
        """
        if n < 2:
            return 0
        num_pairs = n * (n - 1) / 2
        m = math.ceil(
            (n * (n - 1) / 6)
            * (math.log(num_pairs) - math.log(self.hyphc_triplet_subsampling_delta))
        )
        return m

    def _sample_ground_truth_triplets(
        self,
        num_pts: int,
        device: torch.device,
        labels: torch.Tensor,
        coarse_labels: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, np.ndarray]:
        num_total = self.hyphc_triplet_subsampling_size
        num_fine = int(num_total * self.hyphc_fine_triplet_ratio)
        if coarse_labels is None:
             num_fine = num_total
             num_coarse = 0
        else:
             num_coarse = num_total - num_fine
        
        triplets = []
        
        # Helper to get indices
        label_to_indices = {}
        for idx, lbl in enumerate(labels):
            lbl = lbl.item()
            if lbl not in label_to_indices:
                label_to_indices[lbl] = []
            label_to_indices[lbl].append(idx)
            
        coarse_label_to_indices = {}
        if coarse_labels is not None:
             for idx, lbl in enumerate(coarse_labels):
                lbl = lbl.item()
                if lbl not in coarse_label_to_indices:
                    coarse_label_to_indices[lbl] = []
                coarse_label_to_indices[lbl].append(idx)

        # Sampling Fine Triplets
        # Anchor i: random
        # Positive j: same label
        # Negative k: different fine label
        
        for _ in range(num_fine):
            # Pick anchor
            anchor_idx = torch.randint(0, num_pts, (1,)).item()
            fine_lbl = labels[anchor_idx].item()
            
            # Pick positive (same fine label)
            pos_candidates = label_to_indices[fine_lbl]
            if len(pos_candidates) < 2:
                 # Skip if only 1 sample alone
                 continue
            
            pos_idx = pos_candidates[torch.randint(0, len(pos_candidates), (1,)).item()]
            while pos_idx == anchor_idx:
                pos_idx = pos_candidates[torch.randint(0, len(pos_candidates), (1,)).item()]
                
            # Pick negative (diff fine label)
            # Just pick random until fine label is different
            neg_idx = torch.randint(0, num_pts, (1,)).item()
            while labels[neg_idx].item() == fine_lbl:
                 neg_idx = torch.randint(0, num_pts, (1,)).item()
            
            triplets.append([anchor_idx, pos_idx, neg_idx])
            
        # Sampling Coarse Triplets
        if coarse_labels is not None:
            for _ in range(num_coarse):
                 # Pick anchor
                anchor_idx = torch.randint(0, num_pts, (1,)).item()
                coarse_lbl = coarse_labels[anchor_idx].item()
                fine_lbl = labels[anchor_idx].item()
                
                # Pick positive (same coarse label, different fine label)
                # Candidates: indices with same coarse label
                # Exclude: indices with same fine label
                
                if coarse_lbl in coarse_label_to_indices:
                    coarse_group = coarse_label_to_indices[coarse_lbl]
                    # Filter out same fine label
                    pos_candidates = [idx for idx in coarse_group if labels[idx].item() != fine_lbl]
                    
                    if len(pos_candidates) == 0:
                        continue
                    
                    pos_idx = pos_candidates[torch.randint(0, len(pos_candidates), (1,)).item()]
                    
                    # Pick negative (different coarse label)
                    neg_idx = torch.randint(0, num_pts, (1,)).item()
                    while coarse_labels[neg_idx].item() == coarse_lbl:
                        neg_idx = torch.randint(0, num_pts, (1,)).item()
                        
                    triplets.append([anchor_idx, pos_idx, neg_idx])

        if not triplets:
             # Fallback to random if empty
             num_samples = self.hyphc_triplet_subsampling_size
             samples = []
             for _ in range(num_samples):
                triplet = [torch.randint(0, num_pts, (1,)).item() for _ in range(3)]
                samples.append(triplet)
             sampled_triplet_indices = torch.tensor(samples, device=device)
             sampled_indices = torch.unique(sampled_triplet_indices)
             np_sampled_triplet_indices = sampled_triplet_indices.cpu().numpy()
             return sampled_indices, np_sampled_triplet_indices
             
        sampled_triplet_indices = torch.tensor(triplets, device=device)
        sampled_indices = torch.unique(sampled_triplet_indices)
        np_sampled_triplet_indices = sampled_triplet_indices.cpu().numpy()
        
        return sampled_indices, np_sampled_triplet_indices

    def sample_triplets(
        self,
        num_pts: int,
        device: torch.device,
        points: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        coarse_labels: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """Sample triplets for the HypHC loss."""
        if self.hyphc_sampling_strategy == "ground_truth" and labels is not None:
            return self._sample_ground_truth_triplets(num_pts, device, labels, coarse_labels)
        
        if self.hyphc_sampling_strategy == "all_subsample_triplets":
            sampled_indices = torch.randperm(num_pts, device=device)[
                : min(self.hyphc_triplet_subsampling_size, num_pts)
            ]
            np_sampled_triplet_indices = (
                torch.combinations(sampled_indices, r=3).cpu().numpy()
            )
        elif self.hyphc_sampling_strategy == "random_with_replace":
            random_indices = torch.randint(0, num_pts*(num_pts-1)*(num_pts-2)-1, 
                                           (min(self.hyphc_triplet_subsampling_size, num_pts),), device=device)
            sampled_triplet_indices = self.decode_triplet_indices_new(random_indices)
            np_sampled_triplet_indices = sampled_triplet_indices.cpu().numpy()
            sampled_indices = torch.unique(sampled_triplet_indices)
        elif self.hyphc_sampling_strategy == "chami":
            sampled_indices = torch.randperm(num_pts, device=device)[
                : min(self.hyphc_triplet_subsampling_size, num_pts)
            ]
            sampled_pair_indices = torch.combinations(sampled_indices, r=2)
            # Append a third random point to the torch tensor of 2-tuples
            random_indices = torch.randint(
                0, num_pts, (sampled_pair_indices.shape[0], 1), device=device
            )
            sampled_triplet_indices = torch.cat(
                [sampled_pair_indices, random_indices], dim=1
            )
            np_sampled_triplet_indices = sampled_triplet_indices.cpu().numpy()
            sampled_indices = torch.unique(sampled_triplet_indices)
        elif self.hyphc_sampling_strategy == "probabilistic_coverage":
            num_samples = self._calculate_probabilistic_coverage_samples(num_pts)
            print(f"Sampling {num_samples} triplets for probabilistic coverage.")
            sampled_triplet_indices = torch.randint(
                0, num_pts, (num_samples, 3), device=device
            )
            for i in range(sampled_triplet_indices.shape[0]):
                while len(torch.unique(sampled_triplet_indices[i])) < 3:
                    sampled_triplet_indices[i] = torch.randint(
                        0, num_pts, (3,), device=device
                    )
            np_sampled_triplet_indices = sampled_triplet_indices.cpu().numpy()
            sampled_indices = torch.unique(sampled_triplet_indices)
        elif self.hyphc_sampling_strategy == "nn" and points is not None:
            anker_points = torch.randperm(num_pts, device=device)[
                : min(self.hyphc_triplet_subsampling_size, num_pts)
            ]
            sampled_triplets = []
            # Use cosine similarity for neighbor search (stable at ball edge)
            # Normalize points for cosine similarity computation
            points_normalized = F.normalize(points, p=2, dim=-1)
            cosine_similarities = torch.mm(points_normalized, points_normalized.T)
            
            for anker in anker_points:
                # Sort by cosine similarity (descending, higher is more similar)
                sorted_indices = torch.argsort(cosine_similarities[anker], descending=True)
                # Exclude the anker point itself (it will be first with similarity ~1.0)
                nn_idx = sorted_indices[
                    1 : min(self.hyphc_num_nn + 1, sorted_indices.shape[0])
                ]
                random_nn_1 = torch.randint(0, nn_idx.shape[0], (1,), device=device)
                random_nn_2 = torch.randint(0, nn_idx.shape[0], (1,), device=device)
                while random_nn_2 == random_nn_1:
                    random_nn_2 = torch.randint(0, nn_idx.shape[0], (1,), device=device)
                sampled_triplet = torch.tensor(
                    [
                        anker.item(),
                        nn_idx[random_nn_1].item(),
                        nn_idx[random_nn_2].item(),
                    ],
                    device=device,
                )
                sampled_triplets.append(sampled_triplet)
            sampled_triplet_indices = torch.stack(sampled_triplets)
            sampled_indices = torch.unique(sampled_triplet_indices)
            np_sampled_triplet_indices = sampled_triplet_indices.cpu().numpy()

        elif self.hyphc_sampling_strategy == "random":
            num_samples = self.hyphc_triplet_subsampling_size
            samples = []
            for _ in range(num_samples):
                triplet = [torch.randint(0, num_pts, (1,)).item() for _ in range(3)]
                samples.append(triplet)
            sampled_triplet_indices = torch.tensor(samples, device=device)
            sampled_indices = torch.unique(sampled_triplet_indices)
            np_sampled_triplet_indices = sampled_triplet_indices.cpu().numpy()

        else:
            raise ValueError(
                f"Invalid sampling strategy: {self.hyphc_sampling_strategy}"
            )
        return sampled_indices, np_sampled_triplet_indices
    
    def decode_triplet_indices_new(self, indices):
        # Indices: Tensor of shape (m,) with integer IDs in [0, n*(n-1)*(n-2)-1]
        # Decode the triplets (i, j, k) from the integer IDs
        ## Decode the first index 'i' from n options
        batch_n = int(self.trainer.datamodule.batch_size)
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

    def compute_lca_depth_chami(
        self, points: torch.Tensor, batch_size: int = 1_000
    ) -> torch.Tensor:
        """LCA depth computation of Chami et al. (2020)."""
        assert self.manifold.c == 1.0, (
            "Chami et al. (2020) requires a curvature of -1.0"
        )
        device = points.device
        lcas = torch.zeros((points.shape[0], points.shape[0]), dtype=points.dtype).to(
            device
        )
        indices = torch.triu_indices(points.shape[0], points.shape[0], 1).to(device)
        while indices.shape[1] > 0:
            x = points[indices[0, :batch_size], :]
            y = points[indices[1, :batch_size], :]
            # Chami LCA computation
            # 1) reflection_center(x)
            r = x / torch.sum(x**2, dim=-1, keepdim=True)
            # 2) isometric_transform(r, y)
            r2 = torch.sum(r**2, dim=-1, keepdim=True) - 1.0
            u = y - r
            b_inv = r2 / torch.sum(u**2, dim=-1, keepdim=True) * u + r
            # 3) Copy
            o_inv = x
            # 4) euc_reflection(o_inv, b_inv)
            xTa = torch.sum(o_inv * b_inv, dim=-1, keepdim=True)
            norm_a_sq = torch.sum(b_inv**2, dim=-1, keepdim=True).clamp_min(1e-15)
            proj = xTa * b_inv / norm_a_sq
            o_inv_ref = 2 * proj - o_inv
            # 5) isometric_transform(r, o_inv_ref)
            r2_ = torch.sum(r**2, dim=-1, keepdim=True) - 1.0
            u_ = o_inv_ref - r
            o_ref = r2_ / torch.sum(u_**2, dim=-1, keepdim=True) * u_ + r
            # 6) _halve(o_ref)
            proj = o_ref / (
                1.0 + torch.sqrt(1 - torch.sum(o_ref**2, dim=-1, keepdim=True))
            )
            # 7) manifold.dist_0(proj)
            lca_depth = self.manifold.dist_0(proj, version=self.distances_version)

            lca_depth = lca_depth.reshape(-1)
            lcas[indices[0, :batch_size], indices[1, :batch_size]] = lca_depth
            lcas[indices[1, :batch_size], indices[0, :batch_size]] = lca_depth
            indices = indices[:, batch_size:]
        return lcas

    def compute_contrastive_lca(
        self, x: torch.Tensor, x_aug1: torch.Tensor, x_aug2: torch.Tensor
    ) -> torch.Tensor:
        # Fréchet center x_aug1 & x_aug2 points around x
        x_aug1 = frechet_center_data(x_aug1, x, self.manifold)
        x_aug2 = frechet_center_data(x_aug2, x, self.manifold)

        if isinstance(self.manifold, PoincareBall):
            x_aug_inner = (x_aug1 * x_aug2).sum(dim=-1, keepdim=True)

            # Compute the center of geodesic circle Gamma through x_aug1 and x_aug2
            x_aug1_2 = x_aug1.pow(2).sum(dim=-1, keepdim=True)
            x_aug2_2 = x_aug2.pow(2).sum(dim=-1, keepdim=True)

            denom = (
                2 * self.manifold.c * (x_aug1_2 * x_aug2_2 - x_aug_inner**2)
            ).clamp(min=1e-15)
            alpha = (
                (1 + self.manifold.c * x_aug1_2) * x_aug2_2
                - (1 + self.manifold.c * x_aug2_2) * x_aug_inner
            ) / denom
            beta = (
                (1 + self.manifold.c * x_aug2_2) * x_aug1_2
                - (1 + self.manifold.c * x_aug1_2) * x_aug_inner
            ) / denom
            z = alpha * x_aug1 + beta * x_aug2

            # Compute the lca and its distance to the origin w.r.t. the full geodesic
            z2 = z.pow(2).sum(dim=-1, keepdim=True)
            z_norm = z.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-15)

            # Computation via conjugation to avoid numerical issues
            sqrt_term = (z2 - 1 / self.manifold.c).clamp(min=1e-15).sqrt()
            denom = self.manifold.c * z_norm * (z_norm + sqrt_term).clamp(min=1e-15)
            lca = z / denom

            dists = self.manifold.dist_0(lca, version=self.distances_version)
        else:
            raise NotImplementedError("Only PoincareBall manifold is implemented.")
        return dists.mean()

    def compute_hyphc_loss_wang(
        self, 
        embeddings: torch.Tensor,
        labels: torch.Tensor | None = None,
        coarse_labels: torch.Tensor | None = None
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n = embeddings.size(0)
        device = embeddings.device

        # Sample triplets
        sampled_indices, np_sampled_triplet_indices = self.sample_triplets(
            n,
            device,
            points=embeddings if self.hyphc_sampling_strategy == "nn" else None,
            labels=labels,
            coarse_labels=coarse_labels,
        )
        # Compute the pairwise distances
        if self.distances_type == "geodesic":
            pts = embeddings[sampled_indices]
            if pts.dim() > 2:
                pts = pts.squeeze(-1)
            distances = compute_pairwise_distances(
                pts,
                self.manifold,
                version=self.distances_version,
            )
            dist_similarities = torch.reciprocal(1 + distances)
            dist_similarities = torch.where(dist_similarities <= self.similarity_pruning_threshold, torch.zeros_like(dist_similarities), dist_similarities)
        elif self.distances_type == "cosine_similarity":
            embeddings_normalized = torch.nn.functional.normalize(
                embeddings[sampled_indices], p=2, dim=-1
            )
            distances = torch.mm(embeddings_normalized, embeddings_normalized.T).clamp(
                min=-1.0, max=1.0
            )
            dist_similarities = (1 - distances) / (1 + distances)
            dist_similarities[dist_similarities <= self.similarity_pruning_threshold] = 0.0
        else:
            raise ValueError(f"Invalid distances type: {self.distances_type}")
        
        # Compute similarities based on GT labels
        if (self.gt_similarity_mode == "hierarchical" or self.gt_similarity_mode == "pseudo") and labels is not None and coarse_labels is not None:
            # Multi-level hierarchical similarity using both fine and coarse labels
            fine_match = (labels[sampled_indices].unsqueeze(0) == labels[sampled_indices].unsqueeze(1)).float()
            coarse_match = (coarse_labels[sampled_indices].unsqueeze(0) == coarse_labels[sampled_indices].unsqueeze(1)).float()
            
            # Combine into a smooth, multi-level weight matrix
            # Points with same fine label get S_FINE
            # Points with same coarse label (but different fine) get S_COARSE
            # Others get S_DIFF
            similarties = fine_match * self.s_fine + \
                         (coarse_match - fine_match) * self.s_coarse + \
                         (1 - coarse_match) * self.s_diff
        
        elif self.gt_similarity_mode == "hybrid" and labels is not None:
            # Hybrid GT-Distance similarity: blend GT labels with manifold distances
            # This anchors GT labels to the current latent representation for smoother optimization
            label_match = (labels[sampled_indices].unsqueeze(0) == labels[sampled_indices].unsqueeze(1)).float()
            # Apply label smoothing to prevent sharp boundaries
            smooth_label_match = label_match * 0.9 + 0.05
            
            # If coarse labels are available, use hierarchical weights for better blending
            if coarse_labels is not None:
                fine_match = label_match
                coarse_match = (coarse_labels[sampled_indices].unsqueeze(0) == coarse_labels[sampled_indices].unsqueeze(1)).float()
                gt_similarities = fine_match * self.s_fine + \
                                 (coarse_match - fine_match) * self.s_coarse + \
                                 (1 - coarse_match) * self.s_diff
            else:
                gt_similarities = smooth_label_match
            
            # Blend GT-based similarity with distance-based similarity
            similarties = self.gt_alpha * gt_similarities + (1 - self.gt_alpha) * dist_similarities
        
        else:
            # Default: use only distance-based similarities
            similarties = dist_similarities

        # Compute the pairwise lcas
        lcas = self.compute_lca_depth(embeddings[sampled_indices])

        # Keep track of the storage of triplets
        lookup_table = {
            idx: pos for pos, idx in enumerate(sampled_indices.cpu().numpy())
        }
        _i, _j, _k = np_sampled_triplet_indices.T
        i = np.vectorize(lookup_table.get)(_i)
        j = np.vectorize(lookup_table.get)(_j)
        k = np.vectorize(lookup_table.get)(_k)

        # Compute the scaled safe softmax of the lcas
        lca_triplets = torch.stack((lcas[i, j], lcas[i, k], lcas[j, k]), dim=1)
        lca_triplet_max, _ = torch.max(lca_triplets, dim=1, keepdim=True)
        scaled_softmax = torch.exp(
            (lca_triplets - lca_triplet_max) / self.hyphc_temperature
        )
        scaled_softmax = scaled_softmax / scaled_softmax.sum(dim=1, keepdim=True)

        # Compute the triplet loss for the subsampled triplets
        sim_stack = torch.stack(
            (similarties[i, j], similarties[i, k], similarties[j, k]), dim=1
        )
        sim_stack2 = torch.stack(
            (
                similarties[i, j] + similarties[i, k],
                similarties[i, j] + similarties[j, k],
                similarties[i, k] + similarties[j, k],
            ),
            dim=1,
        )
        sim_stack2_min, _ = torch.min(sim_stack2, dim=1, keepdim=False)
        triplet_loss = torch.sum(sim_stack * (1 - scaled_softmax))

        scale_min = sim_stack2_min.sum()

        total_loss = triplet_loss / scale_min

        self.metrics.setdefault("clustering/scale_min", []).append(
            scale_min.detach().item()
        )

        return total_loss, triplet_loss, similarties.sum(), total_loss


    def compute_hyphc_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor | None = None,
        coarse_labels: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        if self.lca_version == "wang":
            return self.compute_hyphc_loss_wang(
                embeddings, 
                labels=labels, 
                coarse_labels=coarse_labels
            )
        
        n = embeddings.size(0)
        device = embeddings.device

        # Sample triplets
        sampled_indices, np_sampled_triplet_indices = self.sample_triplets(
            n,
            device,
            points=embeddings if self.hyphc_sampling_strategy == "nn" else None,
            labels=labels,
            coarse_labels=coarse_labels,
        )
        # Compute the pairwise distances
        if self.distances_type == "geodesic":
            distances = compute_pairwise_distances(
                embeddings[sampled_indices],
                self.manifold,
                version=self.distances_version,
            )
            similarties = torch.reciprocal(1 + distances)
        elif self.distances_type == "cosine_similarity":
            embeddings_normalized = torch.nn.functional.normalize(
                embeddings[sampled_indices], p=2, dim=-1
            )
            distances = torch.mm(embeddings_normalized, embeddings_normalized.T).clamp(
                min=-1.0, max=1.0
            )
            similarties = (1 - distances) / (1 + distances)
        else:
            raise ValueError(f"Invalid distances type: {self.distances_type}")
        similarties_sum = similarties.sum()

        # Compute the pairwise lcas
        if self.lca_version == "ours":
            lcas = self.compute_lca_depth(embeddings[sampled_indices])
        elif self.lca_version == "chami":
            lcas = self.compute_lca_depth_chami(embeddings[sampled_indices])
        else:
            raise ValueError(f"Invalid LCA version: {self.lca_version}")

        # Keep track of the storage of triplets
        lookup_table = {
            idx: pos for pos, idx in enumerate(sampled_indices.cpu().numpy())
        }
        _i, _j, _k = np_sampled_triplet_indices.T
        i = np.vectorize(lookup_table.get)(_i)
        j = np.vectorize(lookup_table.get)(_j)
        k = np.vectorize(lookup_table.get)(_k)

        # Compute the scaled safe softmax of the lcas
        lca_triplets = torch.stack((lcas[i, j], lcas[i, k], lcas[j, k]), dim=1)
        lca_triplet_max, _ = torch.max(lca_triplets, dim=1, keepdim=True)
        scaled_softmax = torch.exp(
            (lca_triplets - lca_triplet_max) / self.hyphc_temperature
        )
        scaled_softmax = scaled_softmax / scaled_softmax.sum(dim=1, keepdim=True)

        # Compute the triplet loss for the subsampled triplets
        sim_stack = torch.stack(
            (similarties[i, j], similarties[i, k], similarties[j, k]), dim=1
        )
        sim_stack2 = torch.stack(
            (
                similarties[i, j] + similarties[i, k],
                similarties[i, j] + similarties[j, k],
                similarties[i, k] + similarties[j, k],
            ),
            dim=1,
        )
        sim_stack2_min, _ = torch.min(sim_stack2, dim=1, keepdim=False)
        sim_stack2_max, _ = torch.max(sim_stack2, dim=1, keepdim=False)
        triplet_loss = torch.sum(sim_stack * (1 - scaled_softmax))

        scale_min = sim_stack2_min.sum()
        scale_max = sim_stack2_max.sum()
        sim_sum = 0

        if self.hyphc_dist_sum_loss:
            sim_sum = similarties_sum

        scale_min += 2 * sim_sum
        scale_max += 2 * sim_sum

        # Compute total loss
        total_loss = triplet_loss
        total_loss += 2 * sim_sum
        normalized_loss = (total_loss - scale_min) / (scale_max - scale_min)

        self.metrics.setdefault("clustering/scale_min", []).append(
            scale_min.detach().item()
        )
        self.metrics.setdefault("clustering/scale_max", []).append(
            scale_max.detach().item()
        )
        self.metrics.setdefault("clustering/min_max_delta", []).append(
            (scale_max - scale_min).detach().item()
        )

        return total_loss, triplet_loss, similarties_sum, normalized_loss

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.model_type == "feedforward":
            x = x.view(x.size(0), -1)
        hidden = self.encoder(x)  # type: ignore
        hidden = self.rms(hidden)
        hidden = torch.nn.functional.tanh(hidden)
        if self.feature_scaling and not isinstance(self.manifold, Euclidean):
            hidden, scaler = self.feature_scaling_layer(hidden)
            if self.feature_scaling_layer.version == "scalar":
                self.metrics.setdefault("clustering/feature_scale", []).append(
                    scaler.detach().item()
                )
        manifold_z = self.hyperbolic_encoding_layer(hidden)  # type: ignore
        if self.use_seperate_heads:
            cluster_z = self.cluster_head(hidden)
        else:
            cluster_z = manifold_z
        return manifold_z, cluster_z

    def decode(self, manifold_z: torch.Tensor) -> torch.Tensor:
        x = self.hyperbolic_decoding_layer(manifold_z)  # type: ignore
        # Cast back from the manifold precision to the model's precision
        x = x.to(dtype=get_dtype_from_precision(self.params_dtype))
        if self.model_type in [
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
        ]:
            x = x.view(-1, 512 * self.expansion, 4, 4)
        elif self.model_type == "poincare_resnet_ae":
            x = x.view(-1, 64, 8, 8)
        x_recon = self.decoder(x)  # type: ignore
        shape = self.orginal_input_shape
        if shape is not None:
            if not isinstance(shape, (tuple, list)):
                shape = (shape,)
            # Ensure all elements in shape are not None
            if all(s is not None for s in shape):
                x_recon = x_recon.view(-1, *shape)
        return x_recon

    def on_train_epoch_start(self):
        """
        Hook to generate pseudo-labels at the start of each epoch if enabled.
        This runs K-Means on the entire training set embeddings to generate
        fine and coarse pseudo-labels.
        """
        if self.gt_similarity_mode != "pseudo":
            return

        # We need to gather embeddings for the entire training set
        # Using the trainer's dataloader
        try:
            # Access the train dataloader from the datamodule if available, or trainer
            if hasattr(self.trainer, "datamodule") and self.trainer.datamodule is not None:
                dataloader = self.trainer.datamodule.train_dataloader()
            else:
                dataloader = self.trainer.train_dataloader

            # Store original mode
            training_mode = self.training
            self.eval()

            all_embeddings = []
            device = self.device
            
            # Note: dealing with DDP would require gathering embeddings across ranks.
            # Here we assume single process or that dataloader provides full data (e.g. valid for typical Hydra configs here)
            # Create a simplified loader to iterate all data without shuffling if possible
            # But reusing dataloader instance is safer for now
            
            with torch.no_grad():
                # We iterate the existing dataloader. 
                # Ideally we want to avoid shuffling to keep order? 
                # Actually order doesn't matter for clustering itself, but updating labels does.
                # If we update dataset.labels, we need to know which index corresponds to which sample.
                # If the dataset is indexed by integer, we assume order is preserved or we need indices.
                # Most standard datasets (CIFAR) are indexed 0..N.
                # If dataloader shuffles, the order we get embeddings is shuffled.
                # We can't just list all embeddings and cluster them, because we don't know which label to update.
                # Wait! We need to assign labels back to the dataset.
                # If we don't know the indices, we can't assign back.
                # We need the batch to contain indices?
                pass
            
            # Since I can't guarantee indices are in the batch, I should construct a DataLoader with shuffle=False
            dataset = dataloader.dataset
            temp_loader = torch.utils.data.DataLoader(
                dataset, 
                batch_size=dataloader.batch_size, 
                shuffle=False, 
                num_workers=dataloader.num_workers,
                collate_fn=dataloader.collate_fn,
                pin_memory=dataloader.pin_memory
            )
            
            for batch in temp_loader:
                # Generic batch unpacking - assuming (images, ..., ...)
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                elif isinstance(batch, dict):
                        # Try common keys
                        x = batch.get("image") or batch.get("x") or batch.get("data")
                else:
                    x = batch
                
                x = x.to(device)
                z = self.encode(x)[0] # self.encode returns (manifold_z, cluster_z)
                all_embeddings.append(z.cpu())

            all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
            
            # Run K-Means
            k_fine = 100
            k_coarse = 20
            
            kmeans_fine = KMeans(n_clusters=k_fine, n_init=10, random_state=self.current_epoch)
            labels_fine = kmeans_fine.fit_predict(all_embeddings)
            
            kmeans_coarse = KMeans(n_clusters=k_coarse, n_init=10, random_state=self.current_epoch)
            labels_coarse = kmeans_coarse.fit_predict(all_embeddings)
            
            # Update Dataset
            updated = False
            if hasattr(dataset, "targets"):
                if isinstance(dataset.targets, list):
                     dataset.targets = labels_fine.tolist()
                else:
                     dataset.targets = torch.tensor(labels_fine)
                     
                if hasattr(dataset, "coarse_targets"):
                     if isinstance(dataset.coarse_targets, list):
                         dataset.coarse_targets = labels_coarse.tolist()
                     else:
                         dataset.coarse_targets = torch.tensor(labels_coarse)
                elif hasattr(dataset, "targets_coarse"): # Check other naming
                     dataset.targets_coarse = torch.tensor(labels_coarse)

                updated = True
            elif hasattr(dataset, "labels"):
                # Handle numpy/tensor/list
                dataset.labels = labels_fine
                if hasattr(dataset, "coarse_labels"):
                    dataset.coarse_labels = labels_coarse
                updated = True
            
            if not updated:
                 print("Warning: Could not update dataset labels (attribute not found).")

            self.train(training_mode)
            
        except Exception as e:
            print(f"Error in pseudo-label generation: {e}")
            # Fallback or continue
            pass

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        manifold_z, cluster_z = self.encode(x)
        if self.recon_weight == 0.0:
            # Skip decoding if reconstruction loss is not used
            return torch.zeros_like(x), manifold_z, cluster_z
        x_recon = self.decode(manifold_z)
        return x_recon, manifold_z, cluster_z

    def compute_contrastive_triplet_loss(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor | None = None,
    ) -> torch.Tensor:

        d_pos = self.manifold.dist(anchor, positive, version=self.distances_version)

        dists_matrix = compute_pairwise_distances(anchor, self.manifold, version=self.distances_version)
        
        eye = torch.eye(dists_matrix.size(0), device=dists_matrix.device)
        dists_matrix = dists_matrix + eye * 1e9

        d_neg, _ = dists_matrix.min(dim=1)

        loss = torch.clamp(d_pos - d_neg + self.triplet_margin, min=0.0)

        return loss.mean()
    
    def compute_radius_hierarchy_loss(self, parent_z: torch.Tensor, child_z: torch.Tensor) -> torch.Tensor:
        d_parent = self.manifold.dist_0(parent_z, version=self.distances_version)
        d_child = self.manifold.dist_0(child_z, version=self.distances_version)
        loss = torch.clamp(d_parent - d_child + self.hierarchy_margin, min=0.0)
        return loss.mean()

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x_aug_1, x_aug_2, x, labels = batch
        x_aug_1 = x_aug_1.to(dtype=get_dtype_from_precision(self.params_dtype))
        x_aug_2 = x_aug_2.to(dtype=get_dtype_from_precision(self.params_dtype))
        x = x.to(dtype=get_dtype_from_precision(self.params_dtype))
        if self.recon_weight > 0.0:
            x_recon, manifold_z, cluster_z = self(x)
            if x_recon.shape != x.shape:
                x_recon = x_recon.view_as(x)
            x_recon_loss = F.mse_loss(x_recon, x)

            recon_loss = x_recon_loss
        else:
            recon_loss = 0.0
            x_recon = torch.zeros_like(x)

        # HypHC loss
        _, cluster_z_x = self.encode(x)
        _, cluster_z_aug_1 = self.encode(x_aug_1)
        _, cluster_z_aug_2 = self.encode(x_aug_2)


        h_loss_inner = self.compute_radius_hierarchy_loss(
            parent_z=cluster_z_aug_2,
            child_z=cluster_z_aug_1
        )


        h_loss_outer = self.compute_radius_hierarchy_loss(
            parent_z=cluster_z_aug_1,
            child_z=cluster_z_x
        )


        hierarchy_loss = (h_loss_inner) #+ h_loss_outer) / 2
        
        self.metrics.setdefault("clustering/hierarchy_loss", []).append(
            hierarchy_loss.detach().item()
        )

        if self.lca_version == "contrastive_lca":
            triplet_loss_val = self.compute_contrastive_triplet_loss(
                anchor=cluster_z_x, positive=cluster_z_aug_1
            )
            dist_to_center = self.manifold.dist_0(cluster_z_x, version=self.distances_version).mean()

            # triplet_loss_val += dist_to_center * 0.01
            self.metrics.setdefault("clustering/dist_to_center", []).append(
                dist_to_center.detach().item()
            )

            # triplet_loss_x = self.compute_contrastive_triplet_loss(cluster_z, cluster_z_aug_1)
            # triplet_loss_val = (triplet_loss_val + triplet_loss_x) / 2
            standard_hyphc_loss = triplet_loss_val


            if batch_idx == 0:
                x_0 = x[0].detach().clone()
                x_aug_1_0 = x_aug_1[0].detach().clone()
                x_aug_2_0 = x_aug_2[0].detach().clone()

                # Normalize each image to [0, 1] range
                x_0 = (x_0 - x_0.min()) / (x_0.max() - x_0.min()).clamp(min=1e-8)
                x_aug_1_0 = (x_aug_1_0 - x_aug_1_0.min()) / (
                    x_aug_1_0.max() - x_aug_1_0.min()
                ).clamp(min=1e-8)
                x_aug_2_0 = (x_aug_2_0 - x_aug_2_0.min()) / (
                    x_aug_2_0.max() - x_aug_2_0.min()
                ).clamp(min=1e-8)

                images = torch.cat([x_0, x_aug_1_0, x_aug_2_0], dim=1) * 255
                self.metrics["clustering/input_augmentations"] = wandb.Image(
                    images.to(dtype=torch.uint8)
                )

            triplet_loss = triplet_loss_val  # Zum Loggen
            similarties_sum = torch.zeros_like(triplet_loss_val)
            normalized_loss = triplet_loss_val

            # dists = self.compute_contrastive_lca(cluster_z, cluster_z_aug_1, cluster_z_aug_2)
            # standard_hyphc_loss += dists
        else:
            #TODO find out which one works better
            coarse_labels = None
            if self.hyphc_sampling_strategy == "ground_truth":
                is_cifar100 = False
                if hasattr(self, "trainer") and hasattr(self.trainer, "datamodule") and hasattr(self.trainer.datamodule, "hparams"):
                    if self.trainer.datamodule.hparams.dataset_name == "cifar100":
                        is_cifar100 = True
                
                if is_cifar100:
                    coarse_labels = torch.tensor([_CIFAR100_COARSE_LABELS[l.item()] for l in labels], device=labels.device)

            standard_hyphc_loss, triplet_loss, similarties_sum, normalized_loss = (
                self.compute_hyphc_loss(cluster_z_aug_1, labels=labels, coarse_labels=coarse_labels)
            )

        if self.normalized_hyphc_loss:
            hyphc_loss = normalized_loss
        else:
            hyphc_loss = standard_hyphc_loss

        hyphc_loss_weight = self.hyphc_start_weight
        if self.hyphc_weight_interval > 0:
            hyphc_loss_weight = self.hyphc_start_weight + (
                self.hyphc_end_weight - self.hyphc_start_weight
            ) * min(self.current_epoch / self.hyphc_weight_interval, 1.0)
        else:
            hyphc_loss_weight = self.hyphc_end_weight

        # Compute total loss
        if hierarchy_loss:
            hyphc_loss = hyphc_loss + hierarchy_loss * self.hierarchy_weight
        total_loss = recon_loss * self.recon_weight + hyphc_loss * hyphc_loss_weight

        dist_to_origin = self.manifold.dist_0(cluster_z_x).mean()
        #total_loss += dist_to_origin * 0.05  # Prevents boundary collapse

        if self.contrastive_enabled:
            # Encode both augmented views to get their hyperbolic embeddings

            if self.byol:
                proj1 = cluster_z_aug_1
                proj2 = cluster_z_aug_2
            else:
                proj1 = cluster_z_x
                proj2 = cluster_z_aug_1
            contrastive_loss = self.compute_contrastive_loss(
                proj1,
                proj2,
                temperature=self.contrastive_temperature,
                use_euclidean=False,
            )

            # Add the weighted contrastive loss to the total loss
            total_loss = total_loss + self.contrastive_weight * contrastive_loss

            # Log the new loss for monitoring
            self.metrics.setdefault("clustering/contrastive_loss", []).append(
                contrastive_loss.detach().item()
            )

        # Logging - DETACH tensors to prevent memory leaks
        if self.recon_weight > 0.0:
            self.metrics.setdefault("clustering/recon_loss", []).append(
                recon_loss.detach().item()  # type: ignore
            )
        self.metrics.setdefault("clustering/hyphc_loss", []).append(
            hyphc_loss.detach().item()
        )
        self.metrics.setdefault("clustering/standard_hyphc_loss", []).append(
            standard_hyphc_loss.detach().item()
        )
        self.metrics.setdefault("clustering/normalized_loss", []).append(
            normalized_loss.detach().item()
        )
        self.metrics.setdefault("clustering/triplet_loss", []).append(
            triplet_loss.detach().item()
        )
        self.metrics.setdefault("clustering/similarties_sum", []).append(
            similarties_sum.detach().item()
        )
        self.metrics.setdefault("clustering/total_loss", []).append(
            total_loss.detach().item()
        )
        if isinstance(self.manifold.c, torch.Tensor):
            self.metrics.setdefault("clustering/c", []).append(self.manifold.c.item())
        self.metrics.setdefault("latent_space/mean", []).append(
            cluster_z_x.mean().detach().item()
        )
        self.metrics.setdefault("latent_space/std", []).append(
            cluster_z_x.std().detach().item()
        )
        self.metrics.setdefault("latent_space/max", []).append(
            cluster_z_x.max().detach().item()
        )
        self.metrics.setdefault("latent_space/min", []).append(
            cluster_z_x.min().detach().item()
        )
        self.metrics.setdefault("latent_space/norm_mean", []).append(
            cluster_z_x.norm(p=2, dim=-1).mean().detach().item()
        )

        self.metrics["clustering/hyphc_loss_weight"] = hyphc_loss_weight

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
