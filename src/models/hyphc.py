import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from typing import Any, Tuple
from lightning.pytorch.utilities import grad_norm
import wandb

from src.hyperbolic_math.src.utils.horo_pca import frechet_center_data
from src.hyperbolic_math.src.manifolds import Euclidean, PoincareBall
from torch.optim import AdamW, SGD
from src.hyperbolic_math.src.optim import RiemannianAdam, RiemannianSGD
from src.models.abstract import AbstractModel
from src.models.resnet import HyperbolicResNet
from src.models.utils import (
    get_dtype_from_precision,
    get_hyperbolic_encoding_layer_by_name,
)
from src.hyperbolic_math.src.utils.helpers import compute_pairwise_distances


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
        hyphc_triplet_subsampling_size: int = 100,
        hyphc_triplet_subsampling_delta: float = 0.05,
        lca_version: str = "ours",
        normalized_hyphc_loss: bool = False,
        distances_type: str = "geodesic",
        hyphc_dist_sum_loss: bool = True,
        evaluation_metrics: dict[str, bool] = {},
        distances_version: str = "mobius_direct",
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
            distances_version=distances_version,
        )
        self.orginal_input_shape = ae_model.orginal_input_shape
        # HypHC parameters
        self.hyphc_start_weight = hyphc_start_weight
        self.hyphc_end_weight = hyphc_end_weight
        self.hyphc_weight_interval = hyphc_weight_interval
        self.hyphc_temperature = hyphc_temperature
        self.hyphc_sampling_strategy = hyphc_sampling_strategy
        self.hyphc_triplet_subsampling_delta = hyphc_triplet_subsampling_delta
        self.hyphc_triplet_subsampling_size = hyphc_triplet_subsampling_size
        self.lca_version = lca_version
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

    def configure_optimizers(self) -> Any:
        if self.optimizer_name == "RiemannianAdam":
            optimizer = RiemannianAdam(
                [{"params": self.cluster_head.parameters()}, {"params": self.encoder.parameters(), "lr": 1e-5}],
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name == "RiemannianSGD":
            optimizer = RiemannianSGD(
                [{"params": self.cluster_head.parameters()}, {"params": self.encoder.parameters(), "lr": 1e-5}],
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9,
            )
        elif self.optimizer_name == "Adam":
            optimizer = AdamW(
                [{"params": self.cluster_head.parameters()}, {"params": self.encoder.parameters(), "lr": 1e-5}],
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name == "SGD":
            optimizer = SGD(
                [{"params": self.cluster_head.parameters()}, {"params": self.encoder.parameters(), "lr": 1e-5}],
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9,
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

    def sample_triplets(
        self, num_pts: int, device: torch.device, points: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """Sample triplets for the HypHC loss."""
        if self.hyphc_sampling_strategy == "all_subsample_triplets":
            sampled_indices = torch.randperm(num_pts, device=device)[
                : min(self.hyphc_triplet_subsampling_size, num_pts)
            ]
            np_sampled_triplet_indices = (
                torch.combinations(sampled_indices, r=3).cpu().numpy()
            )
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
            dists = compute_pairwise_distances(
                points,
                self.manifold,
                version=self.distances_version,
            )
            for anker in anker_points:
                sorted_indices = torch.argsort(dists[anker])
                # Exclude the anker point itself
                nn_idx = sorted_indices[1 : min( self.hyphc_num_nn + 1, sorted_indices.shape[0] )]
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

        else:
            raise ValueError(
                f"Invalid sampling strategy: {self.hyphc_sampling_strategy}"
            )
        return sampled_indices, np_sampled_triplet_indices

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

    def compute_contrastive_lca(self, x: torch.Tensor, x_aug1: torch.Tensor, x_aug2: torch.Tensor) -> torch.Tensor:
        # Fréchet center x_aug1 & x_aug2 points around x
        x_aug1 = frechet_center_data(x_aug1, x, self.manifold)
        x_aug2 = frechet_center_data(x_aug2, x, self.manifold)

        if isinstance(self.manifold, PoincareBall):
            x_aug_inner = (x_aug1 * x_aug2).sum(dim=-1, keepdim=True)

            # Compute the center of geodesic circle Gamma through x_aug1 and x_aug2
            x_aug1_2 = x_aug1.pow(2).sum(dim=-1, keepdim=True)
            x_aug2_2 = x_aug2.pow(2).sum(dim=-1, keepdim=True)

            denom = (2 * self.manifold.c * (x_aug1_2 * x_aug2_2 - x_aug_inner**2)).clamp(min=1e-15)
            alpha = (
                (1 + self.manifold.c * x_aug1_2) * x_aug2_2 - (1 + self.manifold.c * x_aug2_2) * x_aug_inner
            ) / denom
            beta = (
                (1 + self.manifold.c * x_aug2_2) * x_aug1_2 - (1 + self.manifold.c * x_aug1_2) * x_aug_inner
            ) / denom
            z = alpha * x_aug1 + beta * x_aug2

            # Compute the lca and its distance to the origin w.r.t. the full geodesic
            z2 = z.pow(2).sum(dim=-1, keepdim=True)
            z_norm = z.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-15)

            # Computation via conjugation to avoid numerical issues
            sqrt_term = (z2 - 1 / self.manifold.c).clamp(min=1e-15).sqrt()
            denom = self.manifold.c * z_norm * (z_norm + sqrt_term).clamp(min=1e-15)
            lca = z / denom

            dists = self.manifold.dist_0(lca)
        else:
            raise NotImplementedError("Only PoincareBall manifold is implemented.")
        return dists.mean()

    def compute_hyphc_loss(
        self, embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n = embeddings.size(0)
        device = embeddings.device

        # Sample triplets
        sampled_indices, np_sampled_triplet_indices = self.sample_triplets(n, device, points=embeddings if self.hyphc_sampling_strategy=="nn" else None)
        # Compute the pairwise distances
        if self.distances_type == "geodesic":
            distances = compute_pairwise_distances(
                embeddings[sampled_indices],
                self.manifold,
                version=self.distances_version,
            )
            if self.distances_version == "lorentzian_proxy":
                # distances = distances + 2/self.manifold.c
                if torch.any(distances < 0):
                    print(
                        "Warning: Negative distances encountered with lorentzian_proxy distance version. Min:{}".format(
                            distances.min().item()
                        )
                    )
                distances = distances.clamp(min=0.0)
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
        negative: torch.Tensor | None = None
    ) -> torch.Tensor:

        d_pos = self.manifold.dist(anchor, positive)

        dists_matrix = compute_pairwise_distances(anchor, self.manifold)
        
        eye = torch.eye(dists_matrix.size(0), device=dists_matrix.device)
        dists_matrix = dists_matrix + eye * 1e9 

        d_neg, _ = dists_matrix.min(dim=1)

        loss = torch.clamp(d_pos - d_neg + self.triplet_margin, min=0.0)
        
        return loss.mean()
    
    def compute_radius_hierarchy_loss(self, parent_z: torch.Tensor, child_z: torch.Tensor) -> torch.Tensor:

        d_parent = self.manifold.dist_0(parent_z)
        d_child = self.manifold.dist_0(child_z)
        loss = torch.clamp(d_parent - d_child + self.hierarchy_margin, min=0.0)
        
        return loss.mean()

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x_aug_1, x_aug_2, x, _ = batch
        if self.recon_weight > 0.0:
            x_recon, manifold_z, cluster_z = self(x)
            if x_recon.shape != x.shape:
                x_recon = x_recon.view_as(x)
            x_recon_loss = F.mse_loss(x_recon, x)

            recon_loss = x_recon_loss
        else:
            recon_loss = 0.0
            x_recon = torch.zeros_like(x)
            manifold_z, cluster_z = self.encode(x)

        hierarchy_loss = None
        # HypHC loss
        _, cluster_z_x = self.encode(x)
        _, cluster_z_aug_1 = self.encode(x_aug_1)
        _, cluster_z_aug_2 = self.encode(x_aug_2)
        if self.lca_version == "contrastive_lca":

            triplet_loss_val = self.compute_contrastive_triplet_loss(
                anchor=cluster_z_x,
                positive=cluster_z_aug_1
            )
            dist_to_center = self.manifold.dist_0(cluster_z_x).mean()

            #triplet_loss_val += dist_to_center * 0.01 
            self.metrics.setdefault("clustering/dist_to_center", []).append(
                dist_to_center.detach().item()
            )

            # triplet_loss_x = self.compute_contrastive_triplet_loss(cluster_z, cluster_z_aug_1)
            # triplet_loss_val = (triplet_loss_val + triplet_loss_x) / 2
            standard_hyphc_loss = triplet_loss_val

            hierarchy_loss = torch.tensor(0.0, device=self.device)

            #h_loss_1 = self.compute_radius_hierarchy_loss(parent_z=cluster_z_aug_1, child_z=cluster_z_x)
            h_loss_2 = self.compute_radius_hierarchy_loss(parent_z=cluster_z_aug_1, child_z=cluster_z_aug_2)
            
            hierarchy_loss = h_loss_2 #(h_loss_1 + h_loss_2) / 2
            

            self.metrics.setdefault("clustering/hierarchy_loss", []).append(
                hierarchy_loss.detach().item()
            )

            if batch_idx == 0:
                x_0 = x[0].detach().clone()
                x_aug_1_0 = x_aug_1[0].detach().clone()
                x_aug_2_0 = x_aug_2[0].detach().clone()
                
                # Normalize each image to [0, 1] range
                x_0 = (x_0 - x_0.min()) / (x_0.max() - x_0.min()).clamp(min=1e-8)
                x_aug_1_0 = (x_aug_1_0 - x_aug_1_0.min()) / (x_aug_1_0.max() - x_aug_1_0.min()).clamp(min=1e-8)
                x_aug_2_0 = (x_aug_2_0 - x_aug_2_0.min()) / (x_aug_2_0.max() - x_aug_2_0.min()).clamp(min=1e-8)
                
                images = torch.cat([x_0, x_aug_1_0, x_aug_2_0], dim=1) * 255
                self.metrics["clustering/input_augmentations"] = wandb.Image(images.to(dtype=torch.uint8))
            

            triplet_loss = triplet_loss_val # Zum Loggen
            similarties_sum = torch.zeros_like(triplet_loss_val)
            normalized_loss = triplet_loss_val
            
            # dists = self.compute_contrastive_lca(cluster_z, cluster_z_aug_1, cluster_z_aug_2)
            # standard_hyphc_loss += dists
        else:
            standard_hyphc_loss, triplet_loss, similarties_sum, normalized_loss = (
                self.compute_hyphc_loss(cluster_z)
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

        if self.contrastive_enabled:
            # Encode both augmented views to get their hyperbolic embeddings

            if self.byol:
                proj1 = cluster_z_aug_1
                proj2 = cluster_z_aug_2
            else:
                proj1 = manifold_z
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
            manifold_z.mean().detach().item()
        )
        self.metrics.setdefault("latent_space/std", []).append(
            manifold_z.std().detach().item()
        )
        self.metrics.setdefault("latent_space/max", []).append(
            manifold_z.max().detach().item()
        )
        self.metrics.setdefault("latent_space/min", []).append(
            manifold_z.min().detach().item()
        )
        self.metrics.setdefault("latent_space/norm_mean", []).append(
            manifold_z.norm(p=2, dim=-1).mean().detach().item()
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
            sync_dist=True,
        )

        return total_loss