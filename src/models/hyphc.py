import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple
from lightning.pytorch.utilities import grad_norm
from src.models.resnet import HyperbolicResNet
from src.models.abstract import AbstractModel
from src.hyperbolic_math.src.utils.helpers import compute_pairwise_distances
from src.hyperbolic_math.src.utils.math_utils import arsinh, sinh, cosh


class HyperbolicHCModel(AbstractModel):
    def __init__(
        self,
        ae_model: AbstractModel,
        clustering_trainable_curvature: bool = False,
        learning_rate: float = 1e-3,
        hyphc_weight: float = 0.1,
        hyphc_temperature: float = 0.5,
        hyphc_sampling_strategy: str = 'all_subsample_triplets',
        hyphc_triplet_subsampling_size: int = 100,
        lca_version: str = 'ours',
        start_epoch: int = 0,
        log_interval: int = 5,
        stage: str = 'clustering',
        global_epoch: int = 0,
        learning_rate_scheduler: str | None = None,
        learnable_hyphc_weight: bool = False,
        freeze_similarities: bool = False,
        normalized_hyphc_loss: bool = False,
        lr_scheduler_params: dict | None = None,
    ):
        super().__init__(
            manifold_name=ae_model.manifold.name,
            curvature_magnitude=ae_model.manifold.c.clone().detach(),
            trainable_curvature=clustering_trainable_curvature,
            latent_dim=ae_model.latent_dim,
            learning_rate=learning_rate,
            log_interval=log_interval,
            stage=stage,
            global_epoch=global_epoch+1,
            learning_rate_scheduler=learning_rate_scheduler,
            lr_scheduler_params=lr_scheduler_params,
        )
        self.global_epoch = start_epoch
        self.hyphc_weight = torch.nn.Parameter(torch.tensor(hyphc_weight), requires_grad=learnable_hyphc_weight)
        self.hyphc_temperature = hyphc_temperature
        self.hyphc_sampling_strategy = hyphc_sampling_strategy
        self.hyphc_triplet_subsampling_size = hyphc_triplet_subsampling_size
        self.lca_version = lca_version
        self.freeze_similarities = freeze_similarities
        self.precomputed_embeddings = None
        self.normalized_hyphc_loss = normalized_hyphc_loss
        # Load encoder, decoder, and to_tangent modules with their weights
        self.encoder = ae_model.encoder
        self.decoder = ae_model.decoder
        self.to_tangent = ae_model.to_tangent
        if isinstance(ae_model, HyperbolicResNet):
            self.from_tangent = ae_model.from_tangent
            self.expansion = ae_model.expansion
        else:
            self.from_tangent = None
            self.expansion = 1

        # Ensure weights are copied by loading state dict
        self.encoder.load_state_dict(ae_model.encoder.state_dict())
        self.decoder.load_state_dict(ae_model.decoder.state_dict())
        self.to_tangent.load_state_dict(ae_model.to_tangent.state_dict())
        
        self.orginal_input_shape = ae_model.orginal_input_shape

    def sample_triplets(self, num_pts: int, device: torch.device) -> Tuple[torch.Tensor, np.ndarray]:
        """Sample triplets for the HypHC loss."""
        if self.hyphc_sampling_strategy == 'all_subsample_triplets':
            sampled_indices = torch.randperm(num_pts, device=device)[:min(self.hyphc_triplet_subsampling_size, num_pts)]
            np_sampled_triplet_indices = torch.combinations(sampled_indices, r=3).cpu().numpy()
        elif self.hyphc_sampling_strategy == 'random':
            pass
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
        """Corrected implementation of Chami et al.'s (2020) LCA depth computation."""
        device = points.device
        sqrt_c = self.manifold.c.sqrt()
        lcas = torch.zeros((points.shape[0], points.shape[0]), dtype=points.dtype).to(device)
        indices = torch.triu_indices(points.shape[0], points.shape[0], 1).to(device)
        while indices.shape[1] > 0:
            x = points[indices[0,:batch_size], :]
            y = points[indices[1,:batch_size], :]

            # Compute the cosine of all angles within the triangle {origin,x,y}
            # Alpha = subtended angle at x
            # Beta = subtended angle at y
            # Gamma = subtended angle at the origin
            dist_x = self.manifold.dist_0(x)
            dist_y = self.manifold.dist_0(y)
            dist_xy = self.manifold.dist(x, y, backproject=True) # type: ignore

            cosh_cx = cosh(sqrt_c * dist_x)
            cosh_cy = cosh(sqrt_c * dist_y)
            cosh_cxy = cosh(sqrt_c * dist_xy)

            sinh_cx = sinh(sqrt_c * dist_x)
            sinh_cy = sinh(sqrt_c * dist_y)
            sinh_cxy = sinh(sqrt_c * dist_xy)

            # Hyperbolic law of cosines
            cos_alpha = (cosh_cx * cosh_cxy - cosh_cy) / (sinh_cx * sinh_cxy)
            cos_beta = (cosh_cy * cosh_cxy - cosh_cx) / (sinh_cy * sinh_cxy)

            cos_alpha -= torch.floor(cos_alpha/(torch.pi*2)) * torch.pi*2
            cos_beta -= torch.floor(cos_beta/(torch.pi*2)) * torch.pi*2
            cos_alpha -= torch.pi
            cos_beta -= torch.pi

            xy = (x * y).sum(dim=-1, keepdim=True)
            xy_norm_prod = (x.norm(p=2, dim=-1, keepdim=True) * y.norm(p=2, dim=-1, keepdim=True)).clamp(min=1e-4)
            cos_gamma = xy / xy_norm_prod

            # Compute the lca depth w.r.t. the full geodesic
            sin_alpha = (1 - cos_alpha ** 2).clamp(min=0).sqrt()

            # Hyperbolic law of sines
            lca_depth = arsinh(sinh_cx * sin_alpha) / sqrt_c
            # Get the lca depth w.r.t. the geodesic segment between x and y
            lca_depth = torch.where(cos_alpha > 0, lca_depth, dist_x)
            lca_depth = torch.where(cos_beta > 0, lca_depth, dist_y)
            lca_depth = torch.where(cos_gamma == -1, torch.zeros_like(lca_depth), lca_depth)

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

    def compute_hyphc_loss(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = embeddings.size(0)
        device = embeddings.device

        # Sample triplets
        sampled_indices, triplet_indices = self.sample_triplets(n, device)
        sampled_embeddings = embeddings[sampled_indices]
        i, j, k = triplet_indices[:, 0], triplet_indices[:, 1], triplet_indices[:, 2]

        # Compute the pairwise distances
        distances = compute_pairwise_distances(sampled_embeddings, self.manifold)
        similarities = torch.reciprocal(1 + distances)
        similarities_sum = similarities.sum()

        # Compute the pairwise lcas
        if self.lca_version == 'ours':
            lcas = self.compute_lca_depth(sampled_embeddings)
        elif self.lca_version == 'chami':
            lcas = self.compute_lca_depth_chami(sampled_embeddings)
        else:
            raise ValueError(f"Invalid LCA version: {self.lca_version}")

        indices_map = torch.full((n,), -1, dtype=torch.long, device=device)
        indices_map[sampled_indices] = torch.arange(sampled_indices.size(0), device=device)
        
        # Map triplet indices to positions
        i_pos = indices_map[i]
        j_pos = indices_map[j]
        k_pos = indices_map[k]

        # Compute the scaled safe softmax of the lcas
        lca_triplets = torch.stack((lcas[i_pos, j_pos], lcas[i_pos, k_pos], lcas[j_pos, k_pos]), dim=1)
        lca_triplet_max, _ = torch.max(lca_triplets, dim=1, keepdim=True)
        scaled_softmax = torch.exp((lca_triplets - lca_triplet_max) / self.hyphc_temperature)
        scaled_softmax = scaled_softmax / scaled_softmax.sum(dim=1, keepdim=True)

        # Compute the triplet loss for the subsampled triplets
        sim_stack = torch.stack((similarities[i_pos, j_pos], similarities[i_pos, k_pos], similarities[j_pos, k_pos]), dim=1)
        triplet_loss = torch.sum(sim_stack * (1 - scaled_softmax))

        # Compute total loss
        total_loss = triplet_loss + 2 * similarities_sum

        return total_loss, triplet_loss, similarities_sum
    

    def compute_normalized_hyphc_loss(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n = embeddings.size(0)
        device = embeddings.device

        # Sample triplets
        sampled_indices, np_sampled_triplet_indices = self.sample_triplets(n, device)

        # Compute the pairwise distances

        distances = compute_pairwise_distances(embeddings[sampled_indices, :], self.manifold)
        similarties = torch.reciprocal(1 + distances)

        similarties_sum = similarties.sum()

        # Compute the pairwise lcas
        if self.lca_version == 'ours':
            lcas = self.compute_lca_depth(embeddings[sampled_indices, :])
        elif self.lca_version == 'chami':
            lcas = self.compute_lca_depth_chami(embeddings[sampled_indices, :])
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
        sim_stack2_max, _ = torch.max(sim_stack2, dim=1, keepdim=False)
        sim_stack2_min, _ = torch.min(sim_stack2, dim=1, keepdim=False)
        triplet_loss = torch.sum(sim_stack * (1 - scaled_softmax))
        
        scale_min = sim_stack2_min.sum() + 2 * similarties_sum
        scale_max = sim_stack2_max.sum() + 2 * similarties_sum

        # Compute total loss
        total_loss = triplet_loss + 2 * similarties_sum
        normalized_loss = (total_loss - scale_min) / (scale_max - scale_min)

        self.metrics.setdefault("clustering/scale_min", []).append(scale_min)
        self.metrics.setdefault("clustering/scale_max", []).append(scale_max)
        self.metrics.setdefault("clustering/min_max_delta", []).append(scale_max - scale_min)

        return total_loss, triplet_loss, similarties_sum, normalized_loss

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.from_tangent is None:
            x = x.view(x.size(0), -1)
        hidden = self.encoder(x)
        tangent_z = self.to_tangent(hidden)
        manifold_z = self.manifold.expmap_0(tangent_z)
        return manifold_z, tangent_z

    def decode(self, manifold_z: torch.Tensor) -> torch.Tensor:
        if self.from_tangent is not None:
            tangent_z = self.manifold.logmap_0(manifold_z)
            x = self.from_tangent(tangent_z)
            x = x.view(-1, 512 * self.expansion, 4, 4)
            x_recon = self.decoder(x)
        else:
            tangent_z = self.manifold.logmap_0(manifold_z)
            x_recon = self.decoder(tangent_z)
        x_recon = x_recon.view(-1, *self.orginal_input_shape)
        return x_recon

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x_aug, x, _ = batch
        x_recon, manifold_z, _ = self(x)
        #x_aug_recon, manifold_z_aug, _ = self(x_aug)

        # Reconstruction loss
        x_recon_loss = F.mse_loss(x_recon, x)
        #x_aug_recon_loss = F.mse_loss(x_aug_recon, x_aug)
        #recon_loss = (x_recon_loss + x_aug_recon_loss) / 2
        recon_loss = x_recon_loss

        # HypHC loss
        if self.precomputed_embeddings is not None:
            # Calculate indices for current batch
            batch_size = x.size(0)
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            manifold_z = self.precomputed_embeddings[start_idx:end_idx]
        if self.normalized_hyphc_loss:
            z_hyphc_loss, triplet_loss, similarties_sum, normalized_loss = self.compute_normalized_hyphc_loss(manifold_z)
        else:
            z_hyphc_loss, triplet_loss, similarties_sum = self.compute_hyphc_loss(manifold_z)
        #z_aug_hyphc_loss, triplet_loss_aug, similarties_sum_aug, normalized_loss_aug = self.compute_normalized_hyphc_loss(manifold_z_aug)
        #hyphc_loss = (z_hyphc_loss + z_aug_hyphc_loss) / 2
        hyphc_loss = z_hyphc_loss
        if self.normalized_hyphc_loss:
            total_loss = recon_loss + normalized_loss * self.hyphc_weight
        else:
            total_loss = recon_loss + z_hyphc_loss * self.hyphc_weight

        # Logging
        self.metrics.setdefault("clustering/recon_loss", []).append(recon_loss)
        self.metrics.setdefault("clustering/x_recon_loss", []).append(x_recon_loss)
        #self.metrics.setdefault("clustering/x_aug_recon_loss", []).append(x_aug_recon_loss)
        self.metrics.setdefault("clustering/hyphc_loss", []).append(hyphc_loss)
        self.metrics.setdefault("clustering/z_hyphc_loss", []).append(z_hyphc_loss)
        #self.metrics.setdefault("clustering/z_aug_hyphc_loss", []).append(z_aug_hyphc_loss)
        self.metrics.setdefault("clustering/triplet_loss", []).append(triplet_loss)
        #self.metrics.setdefault("clustering/triplet_loss_aug", []).append(triplet_loss_aug)
        self.metrics.setdefault("clustering/similarties_sum", []).append(similarties_sum)
        #self.metrics.setdefault("clustering/similarties_sum_aug", []).append(similarties_sum_aug)
        self.metrics.setdefault("clustering/total_loss", []).append(total_loss)
        if self.normalized_hyphc_loss:
            self.metrics.setdefault("clustering/normalized_loss", []).append(normalized_loss)
        #self.metrics.setdefault("clustering/normalized_loss_aug", []).append(normalized_loss_aug)
        #self.metrics.setdefault('clustering/dist_x2x_aug', []).append(self.manifold.dist(manifold_z, manifold_z_aug).mean())
        self.metrics.setdefault('clustering/c', []).append(self.manifold.c.item())
        self.metrics.setdefault('clustering/lr', []).append(self.optimizers().param_groups[0]['lr']) # type: ignore
        self.metrics.setdefault('clustering/hyphc_weight', []).append(self.hyphc_weight.item())

        for key, value in grad_norm(self, norm_type=2).items():
            if key == "grad_2.0_norm_total":
                key = "grad_2.0_norm/total"
            self.metrics.setdefault(f"{key}", []).append(value)

        return total_loss
    
    def on_train_start(self) -> None:
        self.on_train_epoch_end()
        if not self.freeze_similarities:
            return

        self.eval()
        self.precomputed_embeddings = None
        all_embeddings = []
        
        with torch.no_grad():
            for batch in self.trainer.train_dataloader: # type: ignore
                x, _, _ = batch
                x = x.to(self.device)
                manifold_z, _ = self.encode(x)
                all_embeddings.append(manifold_z)
            all_embeddings = torch.cat(all_embeddings, dim=0)
            self.precomputed_embeddings = all_embeddings
        self.train()

