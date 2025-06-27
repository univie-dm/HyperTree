import torch
import torch.nn.functional as F
import numpy as np

from typing import Tuple
from lightning.pytorch.utilities import grad_norm
from torchmetrics.functional.pairwise import pairwise_cosine_similarity

from src.models.utils import get_dtype_from_precision
from src.models.resnet import HyperbolicResNet
from src.models.abstract import AbstractModel
from src.hyperbolic_math.src.utils.helpers import compute_pairwise_distances


class HyperbolicHCModel(AbstractModel):
    def __init__(
        self,
        ae_model: AbstractModel,
        trainable_curvature: bool = False,
        learning_rate: float = 1e-3,
        log_interval: int = 5,
        stage: str = 'clustering',
        #global_epoch: int = 0, ### just renamed
        learning_rate_scheduler: str | None = None,
        lr_scheduler_params: dict | None = None,
        #start_epoch: int = 0, ###
        # HypHC parameters
        hyphc_weight: float = 1.0,
        hyphc_temperature: float = 0.05,
        hyphc_sampling_strategy: str = 'all_subsample_triplets',
        hyphc_triplet_subsampling_size: int = 100,
        lca_version: str = 'ours',
        normalized_hyphc_loss: bool = False,
        distances_type: str = 'geodesic',
        hyphc_dist_sum_loss: bool = True,
        evaluation_metrics: list[str] | None = None,
        start_epoch: int = 0,  # This is used to set the global_epoch in the parent class
    ):
        super().__init__(
            model_type=ae_model.model_type,
            hyperbolic_encoding=ae_model.hyperbolic_encoding,
            params_dtype=ae_model.params_dtype,
            latent_dim=ae_model.latent_dim,
            manifold_name=ae_model.manifold_name,
            manifold_precision=ae_model.manifold_precision,
            curvature_magnitude=ae_model.manifold.c.clone().detach(),
            trainable_curvature=trainable_curvature,
            hyperbolic_decoding=ae_model.hyperbolic_decoding,
            optimizer_name=ae_model.optimizer_name,
            learning_rate=learning_rate,
            log_interval=log_interval,
            stage=stage,
            global_epoch=start_epoch,
            learning_rate_scheduler=learning_rate_scheduler,
            lr_scheduler_params=lr_scheduler_params,
            distances_type=distances_type,
            evaluation_metrics=evaluation_metrics,
        )
        #self.global_epoch = start_epoch ###
        self.orginal_input_shape = ae_model.orginal_input_shape
        # HypHC parameters
        self.hyphc_weight = hyphc_weight
        self.hyphc_temperature = hyphc_temperature
        self.hyphc_sampling_strategy = hyphc_sampling_strategy
        self.hyphc_triplet_subsampling_size = hyphc_triplet_subsampling_size
        self.lca_version = lca_version
        self.normalized_hyphc_loss = normalized_hyphc_loss
        self.distances_type = distances_type
        self.hyphc_dist_sum_loss = hyphc_dist_sum_loss
        self.save_hyperparameters()


        # Load encoding and decoding modules with their weights
        self.encoder: torch.nn.Sequential = ae_model.encoder
        self.hyperbolic_encoding_layer: torch.nn.Module = ae_model.hyperbolic_encoding_layer
        self.hyperbolic_decoding_layer: torch.nn.Module = ae_model.hyperbolic_decoding_layer
        self.decoder: torch.nn.Sequential = ae_model.decoder
        if isinstance(ae_model, HyperbolicResNet):
            self.expansion = ae_model.expansion
        else:
            self.expansion = 1

        # Ensure weights are copied by loading state dict
        self.encoder.load_state_dict(ae_model.encoder.state_dict())
        self.hyperbolic_encoding_layer.load_state_dict(ae_model.hyperbolic_encoding_layer.state_dict())
        self.hyperbolic_decoding_layer.load_state_dict(ae_model.hyperbolic_decoding_layer.state_dict())
        self.decoder.load_state_dict(ae_model.decoder.state_dict())

    def sample_triplets(self, num_pts: int, device: torch.device) -> Tuple[torch.Tensor, np.ndarray]:
        """Sample triplets for the HypHC loss."""
        if self.hyphc_sampling_strategy == 'all_subsample_triplets':
            sampled_indices = torch.randperm(num_pts, device=device)[:min(self.hyphc_triplet_subsampling_size, num_pts)]
            np_sampled_triplet_indices = torch.combinations(sampled_indices, r=3).cpu().numpy()
        elif self.hyphc_sampling_strategy == 'chami':
            sampled_indices = torch.randperm(num_pts, device=device)[:min(self.hyphc_triplet_subsampling_size, num_pts)]
            sampled_pair_indices = torch.combinations(sampled_indices, r=2)
            # Append a third random point to the torch tensor of 2-tuples
            random_indices = torch.randint(0, num_pts, (sampled_pair_indices.shape[0], 1), device=device)
            sampled_triplet_indices = torch.cat([sampled_pair_indices, random_indices], dim=1)
            np_sampled_triplet_indices = sampled_triplet_indices.cpu().numpy()
            sampled_indices = torch.unique(sampled_triplet_indices)
        else:
            raise ValueError(f"Invalid sampling strategy: {self.hyphc_sampling_strategy}")
        return sampled_indices, np_sampled_triplet_indices

    def compute_lca_depth(self, points: torch.Tensor, batch_size: int=1_000) -> torch.Tensor:
        """Generalized and corrected implementation of Chami et al.'s (2020) LCA depth computation using inverse geometry."""
        device = points.device
        lcas = torch.zeros((points.shape[0], points.shape[0]), dtype=points.dtype).to(device)
        indices = torch.triu_indices(points.shape[0], points.shape[0], 1).to(device)
        while indices.shape[1] > 0:
            x = points[indices[0,:batch_size], :]
            y = points[indices[1,:batch_size], :]

            # Compute the center of geodesic circle Gamma through x and y
            x2 = x.pow(2).sum(dim=-1, keepdim=True)
            y2 = y.pow(2).sum(dim=-1, keepdim=True)
            xy = (x * y).sum(dim=-1, keepdim=True)

            denom = (2 * self.manifold.c * (x2 * y2 - xy ** 2)).clamp(min=1e-15)
            alpha = ((1 + self.manifold.c * x2) * y2 - (1 + self.manifold.c * y2) * xy) / denom
            beta = ((1 + self.manifold.c * y2) * x2 - (1 + self.manifold.c * x2) * xy) / denom
            z = alpha * x + beta * y

            # Compute the lca and its distance to the origin w.r.t. the full geodesic
            z2 = z.pow(2).sum(dim=-1, keepdim=True)
            lca = (1 - (z2 - 1/self.manifold.c).clamp(min=1e-15).sqrt() / z.norm(p=2, dim=-1, keepdim=True)) * z
            lca_depth = self.manifold.dist_0(lca)

            # Compute the cosine of the angles at the origin in the triangles {x, origin, c}, {y, origin, c}  {x, origin, y}
            xlca = (x * lca).sum(dim=-1, keepdim=True)
            ylca = (y * lca).sum(dim=-1, keepdim=True)
            lca_norm = lca.norm(p=2, dim=-1, keepdim=True)

            cos_x0c = xlca / (x2.sqrt() * lca_norm).clamp(min=1e-15)
            cos_y0c = ylca / (y2.sqrt() * lca_norm).clamp(min=1e-15)
            cos_x0y = xy / (x2.sqrt() * y2.sqrt()).clamp(min=1e-15)

            # Get the lca depth w.r.t. the geodesic segment between x and y
            condition = torch.logical_or(torch.logical_or(cos_x0y > cos_x0c, cos_x0y > cos_y0c), cos_x0y == 1) # True iff the lca lies outside
            lca_depth = torch.where(condition, torch.min(self.manifold.dist_0(x), self.manifold.dist_0(y)), lca_depth)
            lca_depth = torch.where(cos_x0y == -1, torch.zeros_like(lca_depth), lca_depth) # Handle antipodal points

            lca_depth = lca_depth.reshape(-1)
            lcas[indices[0,:batch_size], indices[1,:batch_size]] = lca_depth
            lcas[indices[1,:batch_size], indices[0,:batch_size]] = lca_depth
            indices = indices[:, batch_size:]
        return lcas

    def compute_lca_depth_chami(self, points: torch.Tensor, batch_size: int=1_000) -> torch.Tensor:
        """LCA depth computation of Chami et al. (2020)."""
        device = points.device
        lcas = torch.zeros((points.shape[0], points.shape[0]), dtype=points.dtype).to(device)
        indices = torch.triu_indices(points.shape[0], points.shape[0], 1).to(device)
        while indices.shape[1] > 0:
            x = points[indices[0,:batch_size], :]
            y = points[indices[1,:batch_size], :]
            # Chami LCA computation
            # 1) reflection_center(x)
            r = x / torch.sum(x ** 2, dim=-1, keepdim=True)
            # 2) isometric_transform(r, y)
            r2 = torch.sum(r ** 2, dim=-1, keepdim=True) - 1.
            u = y - r
            b_inv =  r2 / torch.sum(u ** 2, dim=-1, keepdim=True) * u + r
            # 3) Copy
            o_inv = x
            # 4) euc_reflection(o_inv, b_inv)
            xTa = torch.sum(o_inv * b_inv, dim=-1, keepdim=True)
            norm_a_sq = torch.sum(b_inv ** 2, dim=-1, keepdim=True).clamp_min(1e-15)
            proj = xTa * b_inv / norm_a_sq
            o_inv_ref = 2 * proj - o_inv
            # 5) isometric_transform(r, o_inv_ref)
            r2_ = torch.sum(r ** 2, dim=-1, keepdim=True) - 1.
            u_ = o_inv_ref - r
            o_ref = r2_ / torch.sum(u_ ** 2, dim=-1, keepdim=True) * u_ + r
            # 6) _halve(o_ref)
            proj = o_ref / (1. + torch.sqrt(1 - torch.sum(o_ref ** 2, dim=-1, keepdim=True)))
            # 7) manifold.dist_0(proj)
            lca_depth = self.manifold.dist_0(proj)

            lca_depth = lca_depth.reshape(-1)
            lcas[indices[0,:batch_size], indices[1,:batch_size]] = lca_depth
            lcas[indices[1,:batch_size], indices[0,:batch_size]] = lca_depth
            indices = indices[:, batch_size:]
        return lcas

    def compute_hyphc_loss(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n = embeddings.size(0)
        device = embeddings.device

        # Sample triplets
        sampled_indices, np_sampled_triplet_indices = self.sample_triplets(n, device)
        # Compute the pairwise distances
        if self.distances_type == 'geodesic':
            distances = compute_pairwise_distances(embeddings[sampled_indices, :], self.manifold)  ## check if the same without ":"
            similarties = torch.reciprocal(1 + distances)
        elif self.distances_type == 'cosine_similarity':
            distances = pairwise_cosine_similarity(
                embeddings[sampled_indices, :],  ## check if the same without ":"
                zero_diagonal=False,
                reduction='none'
            )
            similarties = distances
        else:
            raise ValueError(f"Invalid distances type: {self.distances_type}")
        similarties_sum = similarties.sum()

        # Compute the pairwise lcas
        if self.lca_version == 'ours':
            lcas = self.compute_lca_depth(embeddings[sampled_indices, :]) ## check if the same without ":"
        elif self.lca_version == 'chami':
            lcas = self.compute_lca_depth_chami(embeddings[sampled_indices, :])  ## check if the same without ":"
        else:
            raise ValueError(f"Invalid LCA version: {self.lca_version}")

        # Keep track of the storage of triplets
        lookup_table = {idx: pos for pos,idx in enumerate(sampled_indices.cpu().numpy())}
        _i, _j, _k = np_sampled_triplet_indices.T
        i = np.vectorize(lookup_table.get)(_i)
        j = np.vectorize(lookup_table.get)(_j)
        k = np.vectorize(lookup_table.get)(_k)

        # Compute the scaled safe softmax of the lcas
        lca_triplets = torch.stack((lcas[i,j], lcas[i,k], lcas[j,k]), dim=1)
        lca_triplet_max, _ = torch.max(lca_triplets, dim=1, keepdim=True)
        scaled_softmax = torch.exp((lca_triplets - lca_triplet_max) / self.hyphc_temperature)
        scaled_softmax = scaled_softmax / scaled_softmax.sum(dim=1, keepdim=True)

        # Compute the triplet loss for the subsampled triplets
        sim_stack = torch.stack((similarties[i,j], similarties[i,k], similarties[j,k]), dim=1)
        sim_stack2 = torch.stack((similarties[i,j]+similarties[i,k], similarties[i,j]+similarties[j,k], similarties[i,k]+similarties[j,k]), dim=1)
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

        self.metrics.setdefault("clustering/scale_min", []).append(scale_min)
        self.metrics.setdefault("clustering/scale_max", []).append(scale_max)
        self.metrics.setdefault("clustering/min_max_delta", []).append(scale_max - scale_min)

        return total_loss, triplet_loss, similarties_sum, normalized_loss

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.model_type == 'feedforward':
            x = x.view(x.size(0), -1)
        hidden = self.encoder(x)
        manifold_z = self.hyperbolic_encoding_layer(hidden)
        return manifold_z

    def decode(self, manifold_z: torch.Tensor) -> torch.Tensor:
        x = self.hyperbolic_decoding_layer(manifold_z)
        # Cast back from the manifold precision to the model's precision
        x = x.to(dtype=get_dtype_from_precision(self.params_dtype))
        if self.model_type in ['resnet18', 'resnet34']:
            x = x.view(-1, 512 * self.expansion, 4, 4)
        x_recon = self.decoder(x)
        if isinstance(self.orginal_input_shape, tuple):
            x_recon = x_recon.view(-1, *self.orginal_input_shape)
        return x_recon

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x_aug, x, _ = batch
        x_recon, manifold_z = self(x)
        #x_aug_recon, manifold_z_aug = self(x_aug)

        # Reconstruction loss
        x_recon_loss = F.mse_loss(x_recon, x)
        #x_aug_recon_loss = F.mse_loss(x_aug_recon, x_aug)
        #recon_loss = (x_recon_loss + x_aug_recon_loss) / 2
        recon_loss = x_recon_loss

        # HypHC loss
        standard_hyphc_loss, triplet_loss, similarties_sum, normalized_loss = self.compute_hyphc_loss(manifold_z)

        if self.normalized_hyphc_loss:
            hyphc_loss = normalized_loss
        else:
            hyphc_loss = standard_hyphc_loss

        # Compute total loss
        total_loss = recon_loss + hyphc_loss * self.hyphc_weight

        # Logging
        self.metrics.setdefault("clustering/recon_loss", []).append(recon_loss)
        self.metrics.setdefault("clustering/hyphc_loss", []).append(hyphc_loss)
        self.metrics.setdefault("clustering/standard_hyphc_loss", []).append(standard_hyphc_loss)
        self.metrics.setdefault("clustering/normalized_loss", []).append(normalized_loss)
        self.metrics.setdefault("clustering/triplet_loss", []).append(triplet_loss)
        self.metrics.setdefault("clustering/similarties_sum", []).append(similarties_sum)
        self.metrics.setdefault("clustering/total_loss", []).append(total_loss)
        self.metrics.setdefault('clustering/c', []).append(self.manifold.c.item())
        self.metrics.setdefault('clustering/lr', []).append(self.optimizers().param_groups[0]['lr'])

        for key, value in grad_norm(self, norm_type=2).items():
            if key == "grad_2.0_norm_total":
                key = "grad_2.0_norm/total"
            self.metrics.setdefault(f"{key}", []).append(value)

        return total_loss

    def on_train_start(self) -> None:
        self.on_train_epoch_end()
