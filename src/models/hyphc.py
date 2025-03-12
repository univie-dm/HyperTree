import torch
import wandb
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.utilities import grad_norm
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity
from src.hyperbolic_math.src.manifolds.euclidean import Euclidean
from src.models.abstract import AbstractModel
from src.hyperbolic_math.src.utils.helpers import compute_pairwise_distances, get_delta
from src.hyperbolic_math.src.utils.math_utils import arsinh, sinh
from src.hyperbolic_math.src.utils.vis_utils import create_figure


class HyperbolicHCModel(AbstractModel):
    def __init__(
        self,
        ae_model: AbstractModel,
        clustering_trainable_curvature: bool = False,
        hyphc_weight: float = 0.1,
        hyphc_temperature: float = 0.5,
        hyphc_sampling_strategy: str = 'all_subsample_triplets',
        hyphc_triplet_subsampling_size: int = 100,
        lca_version: str = 'ours',
        start_epoch: int = 0,
        learning_rate: float = 1e-3,
        log_interval: int = 10,
    ):
        super().__init__(
            manifold_name=ae_model.manifold.name,
            curvature_magnitude=ae_model.manifold.c,
            trainable_curvature=clustering_trainable_curvature,
            latent_dim=ae_model.latent_dim,
            learning_rate=learning_rate,
            log_interval=log_interval
        )
        self.global_epoch = start_epoch

        self.hyphc_weight = hyphc_weight
        self.hyphc_temperature = hyphc_temperature
        self.hyphc_sampling_strategy = hyphc_sampling_strategy
        self.hyphc_triplet_subsampling_size = hyphc_triplet_subsampling_size
        self.lca_version = lca_version

        # Load encoder, decoder, and to_tangent modules with their weights
        self.encoder = ae_model.encoder
        self.decoder = ae_model.decoder
        self.to_tangent = ae_model.to_tangent

        # Ensure weights are copied by loading state dict
        self.encoder.load_state_dict(ae_model.encoder.state_dict())
        self.decoder.load_state_dict(ae_model.decoder.state_dict())
        self.to_tangent.load_state_dict(ae_model.to_tangent.state_dict())

    def sample_triplets(self, num_pts: int, device: torch.device) -> np.ndarray:
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

    def compute_lca_depth(self, points: torch.Tensor, batch_size: int=1_000, debug=False) -> torch.Tensor:
        """Corrected implementation of Chami et al. (2020) LCA depth computation."""
        device = points.device
        lcas = torch.zeros((points.shape[0], points.shape[0]), dtype=points.dtype).to(device)
        indices = torch.triu_indices(points.shape[0], points.shape[0], 1).to(device)
        while indices.shape[1] > 0:
            x = points[indices[0,:batch_size], :]
            y = points[indices[1,:batch_size], :]

            xy = (x * y).sum(dim=-1, keepdim=True)
            xy_norm_prod = (x.norm(p=2, dim=-1, keepdim=True) * y.norm(p=2, dim=-1, keepdim=True)).clamp(min=1e-15)
            sinh_x = sinh(self.manifold.dist_0(x))
            sinh_y = sinh(self.manifold.dist_0(y))
            sinh_xy = sinh(self.manifold.dist(x, y))
            sin_gamma = (1 - (xy / xy_norm_prod) ** 2).clamp(min=0).sqrt()

            sin_alpha = sin_gamma * sinh_y / sinh_xy
            sin_beta = sin_gamma * sinh_x / sinh_xy

            if debug:
                print(f"sinh_x: {sinh_x}, sinh_y: {sinh_y}, sinh_xy: {sinh_xy}")
                print(f"sin_gamma: {sin_gamma}, sin_alpha: {sin_alpha}, sin_beta: {sin_beta}")

            # LCA depth is only correct if the LCA lies on the geodesic segment connecting x and y
            lca_depth = arsinh(sin_alpha * sinh_x)

            if debug:
                print(f"Uncorrected LCA depth: {lca_depth}")

            # Take the closest point to the orthogonal projection of the origin that lies on the geodesic segment
            lca_depth = torch.where(sin_alpha > 0, lca_depth, torch.min(self.manifold.dist_0(x), self.manifold.dist_0(y)))
            lca_depth = torch.where(sin_beta > 0, lca_depth, torch.min(self.manifold.dist_0(x), self.manifold.dist_0(y)))
            lca_depth = torch.where(torch.logical_and(sin_gamma == 0, xy < 0), torch.zeros_like(lca_depth), lca_depth)

            if debug:
                print(f"x_dist: {self.manifold.dist_0(x)}]")
                print(f"y_dist: {self.manifold.dist_0(y)}]")
                print(f"Final LCA depth: {lca_depth}")

            lca_depth = lca_depth.reshape(-1)
            lcas[indices[0,:batch_size], indices[1,:batch_size]] = lca_depth
            lcas[indices[1,:batch_size], indices[0,:batch_size]] = lca_depth
            indices = indices[:, batch_size:]
        return lcas

    def compute_lca_depth_chami(self, points: torch.Tensor, batch_size: int=1_000, debug=False) -> torch.Tensor:
        """LCA depth computation of Chami et al. (2020)."""
        device = points.device
        lcas = torch.zeros((points.shape[0], points.shape[0]), dtype=points.dtype).to(device)
        indices = torch.triu_indices(points.shape[0], points.shape[0], 1).to(device)
        while indices.shape[1] > 0:
            x = points[indices[0,:batch_size], :]
            y = points[indices[1,:batch_size], :]

            xy = (x * y).sum(dim=-1, keepdim=True)
            xy_norm_prod = (x.norm(p=2, dim=-1, keepdim=True) * y.norm(p=2, dim=-1, keepdim=True)).clamp(min=1e-15)
            sinh_x = sinh(self.manifold.dist_0(x))
            sinh_y = sinh(self.manifold.dist_0(y))
            sinh_xy = sinh(self.manifold.dist(x, y))
            sin_gamma = (1 - (xy / xy_norm_prod) ** 2).clamp(min=0).sqrt()

            sin_alpha = sin_gamma * sinh_y / sinh_xy
            sin_beta = sin_gamma * sinh_x / sinh_xy

            if debug:
                print(f"sinh_x: {sinh_x}, sinh_y: {sinh_y}, sinh_xy: {sinh_xy}")
                print(f"sin_gamma: {sin_gamma}, sin_alpha: {sin_alpha}, sin_beta: {sin_beta}")

            lca_depth = arsinh(sin_alpha * sinh_x)

            lca_depth = lca_depth.reshape(-1)
            lcas[indices[0,:batch_size], indices[1,:batch_size]] = lca_depth
            lcas[indices[1,:batch_size], indices[0,:batch_size]] = lca_depth
            indices = indices[:, batch_size:]
        return lcas

    def compute_hyphc_loss(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = embeddings.size(0)
        device = embeddings.device

        # Sample triplets
        sampled_indices, np_sampled_triplet_indices = self.sample_triplets(n, device)

        # Compute the pairwise distances
        distances = compute_pairwise_distances(embeddings[sampled_indices, :], self.manifold)
        similarties = torch.reciprocal(1 + distances)
        similarties_sum = similarties.sum()

        # Compute the pairwise lcas
        if self.lca_version == 'chami':
            lcas = self.compute_lca_depth_chami(embeddings[sampled_indices, :])
        elif self.lca_version == 'ours':
            lcas = self.compute_lca_depth(embeddings[sampled_indices, :])
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
        triplet_loss = torch.sum(sim_stack * (1 - scaled_softmax))

        # Compute total loss
        total_loss = triplet_loss #+ 2*similarties_sum

        return total_loss, triplet_loss, similarties_sum

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x)
        tangent_z = self.to_tangent(hidden)
        manifold_z = self.manifold.expmap_0(tangent_z)
        return manifold_z, tangent_z

    def decode(self, manifold_z: torch.Tensor) -> torch.Tensor:
        tangent_z = self.manifold.logmap_0(manifold_z)
        x_recon = self.decoder(tangent_z)
        return x_recon

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x_aug, x, _ = batch
        x = x.view(x.size(0), -1)
        x_aug = x_aug.view(x_aug.size(0), -1)

        x_recon, manifold_z, _ = self(x)
        x_aug_recon, manifold_z_aug, _ = self(x_aug)

        # Reconstruction loss
        x_recon_loss = F.mse_loss(x_recon, x)
        x_aug_recon_loss = F.mse_loss(x_aug_recon, x_aug)
        recon_loss = (x_recon_loss + x_aug_recon_loss) / 2

        # HypHC loss
        z_hyphc_loss, triplet_loss, similarties_sum = self.compute_hyphc_loss(manifold_z)
        z_aug_hyphc_loss, triplet_loss_aug, similarties_sum_aug = self.compute_hyphc_loss(manifold_z_aug)
        hyphc_loss = (z_hyphc_loss + z_aug_hyphc_loss) / 2
        total_loss = recon_loss + self.hyphc_weight * hyphc_loss

        # Logging
        self.metrics.setdefault("clustering/recon_loss", []).append(recon_loss)
        self.metrics.setdefault("clustering/x_recon_loss", []).append(x_recon_loss)
        self.metrics.setdefault("clustering/x_aug_recon_loss", []).append(x_aug_recon_loss)
        self.metrics.setdefault("clustering/hyphc_loss", []).append(hyphc_loss)
        self.metrics.setdefault("clustering/z_hyphc_loss", []).append(z_hyphc_loss)
        self.metrics.setdefault("clustering/z_aug_hyphc_loss", []).append(z_aug_hyphc_loss)
        self.metrics.setdefault("clustering/triplet_loss", []).append(triplet_loss)
        self.metrics.setdefault("clustering/triplet_loss_aug", []).append(triplet_loss_aug)
        self.metrics.setdefault("clustering/similarties_sum", []).append(similarties_sum)
        self.metrics.setdefault("clustering/similarties_sum_aug", []).append(similarties_sum_aug)
        self.metrics.setdefault("clustering/total_loss", []).append(total_loss)
        self.metrics.setdefault('clustering/dist_x2x_aug', []).append(self.manifold.dist(manifold_z, manifold_z_aug).mean())
        self.metrics.setdefault('clustering/c', []).append(self.manifold.c.item())

        for key, value in grad_norm(self, norm_type=2).items():
            if key == "grad_2.0_norm_total":
                key = "grad_2.0_norm/total"
            self.metrics.setdefault(f"{key}", []).append(value)

        return total_loss

    def on_train_epoch_end(self) -> None:
        if not isinstance(self.logger, WandbLogger):
            return
        if self.current_epoch % self.log_interval == 0:
            with torch.no_grad():
                test_loader = self.trainer.datamodule.test_dataloader() # type: ignore

                embeddings = []
                labels = []
                tangents = []

                for batch_idx, (x_aug, x, y) in enumerate(test_loader):
                    x = x.view(x.size(0), -1)
                    _, z, tangent_z = self(x.to(self.device))
                    embeddings.append(z)
                    labels.append(y)
                    tangents.append(tangent_z)

                all_embeddings = torch.cat(embeddings, dim=0)
                all_labels = torch.cat(labels, dim=0)
                all_tangents = torch.cat(tangents, dim=0)
                delta, diam, rel_delta = get_delta(all_tangents, Euclidean())

                # Move to CPU for NMI and ARI calculation
                all_embeddings_np = all_embeddings.cpu().numpy()
                all_labels_np = all_labels.cpu().numpy()

                pwd = compute_pairwise_distances(all_embeddings, manifold=self.manifold)
                cosine_sim_matrix = 1-cosine_similarity(all_embeddings_np)

                # Perform complete linkage clustering
                agglomerative_clustering = AgglomerativeClustering(
                    n_clusters=len(np.unique(all_labels_np)),
                    metric='precomputed',
                    linkage='complete'
                )

                # Perform clustering with cosine similarity on hyperbolic embeddings
                predicted_labels_cosine = agglomerative_clustering.fit_predict(cosine_sim_matrix)
                nmi = normalized_mutual_info_score(all_labels_np, predicted_labels_cosine)
                ari = adjusted_rand_score(all_labels_np, predicted_labels_cosine)

                # Perform clustering with geodesic distance on hyperbolic embeddings
                predicted_labels_geodesic = agglomerative_clustering.fit_predict(pwd.cpu().numpy())
                nmi_geodesic = normalized_mutual_info_score(all_labels_np, predicted_labels_geodesic)
                ari_geodesic = adjusted_rand_score(all_labels_np, predicted_labels_geodesic)

                # Create embeddings plot
                fig = create_figure(all_embeddings, manifold=self.manifold, labels=all_labels_np)

                # Log the metrics
                self.metrics.setdefault("test/nmi_cosine", []).append(nmi)
                self.metrics.setdefault("test/ari_cosine", []).append(ari)
                self.metrics.setdefault("test/nmi_geodesic", []).append(nmi_geodesic)
                self.metrics.setdefault("test/ari_geodesic", []).append(ari_geodesic)
                self.metrics.setdefault("test/delta", []).append(delta.item())
                self.metrics.setdefault("test/diam", []).append(diam.item())
                self.metrics.setdefault("test/rel_delta", []).append(rel_delta.item())
                self.metrics["test/Hyperbolic Embeddings"] = wandb.Image(fig)
                plt.close()

        self.metrics['clustering/epoch'] = self.current_epoch
        for key, value in self.metrics.items():
            if isinstance(value, list):
                self.metrics[key] = sum(value) / len(value)
        self.logger.experiment.log(self.metrics, step=self.global_epoch+self.current_epoch+1)
        self.metrics = {}

    def on_train_start(self) -> None:
        if not isinstance(self.logger, WandbLogger):
            return
        self.on_train_epoch_end()
