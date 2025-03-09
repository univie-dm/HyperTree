import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np
import wandb
import matplotlib.pyplot as plt
from lightning.pytorch.utilities import grad_norm


from typing import Tuple, Type
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from src.models.abstract import AbstractModel
from ..hyperbolic_math.src.utils.helpers import get_delta, compute_pairwise_distances
from ..hyperbolic_math.src.utils.math_utils import arsinh, sinh
from ..hyperbolic_math.src.utils.vis_utils import create_figure


class HyperbolicHCModel(AbstractModel):
    def __init__(
        self,
        ae_model: Type[AbstractModel],
        clustering_trainable_curvature: bool = False,
        hyphc_weight: float = 0.1,
        hyphc_temperature: float = 0.5,
        hyphc_triplet_subsampling_size: int = 100,
        start_epoch: int = 0,
    ):
        super().__init__(
            manifold_name=ae_model.manifold.name,
            curvature_magnitude=ae_model.manifold.c,
            trainable_curvature=clustering_trainable_curvature,
            latent_dim=ae_model.latent_dim,
            learning_rate=ae_model.learning_rate,
            log_interval=ae_model.log_interval
        )
        self.global_epoch = start_epoch

        self.hyphc_weight = hyphc_weight
        self.hyphc_temperature = hyphc_temperature
        self.hyphc_triplet_subsampling_size = hyphc_triplet_subsampling_size

        # Load encoder, decoder, and to_tangent modules with their weights
        self.encoder = ae_model.encoder
        self.decoder = ae_model.decoder
        self.to_tangent = ae_model.to_tangent

        # Ensure weights are copied by loading state dict
        self.encoder.load_state_dict(ae_model.encoder.state_dict())
        self.decoder.load_state_dict(ae_model.decoder.state_dict())
        self.to_tangent.load_state_dict(ae_model.to_tangent.state_dict())

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

    def compute_hyphc_loss(self, embeddings: torch.Tensor, subsample_size: int=1e6, sampling: str="all_subsample_triplets") -> torch.Tensor:
        n = embeddings.size(0)
        device = embeddings.device

        # Sample triplets
        if sampling == 'all_subsample_triplets':
            if subsample_size < n:
                sampled_indices = torch.unique(torch.randint(0, n, (subsample_size,), device=device))
                np_sampled_indices = sampled_indices.cpu().numpy()
                np_sampled_triplet_indices = torch.combinations(sampled_indices, r=3).cpu().numpy()
            else:
                sampled_indices = torch.arange(n, device=device)
                np_sampled_indices = sampled_indices.cpu().numpy()
                np_sampled_triplet_indices = torch.combinations(torch.arange(n, device=device), r=3).cpu().numpy()
        elif sampling == 'random':
            pass
        elif sampling == 'chami_1':
            n_nodes =0 ### added to avoid compilation error
            num_samples = int(num_samples)
            all_nodes = np.arange(n_nodes)
            mesh = np.array(np.meshgrid(all_nodes, all_nodes))
            pairs = mesh.T.reshape(-1, 2)
            pairs = pairs[pairs[:, 0] < pairs[:, 1]]
            n_pairs = pairs.shape[0]

            print("Generating all pairs subset")
            subset = np.random.choice(np.arange(n_pairs), num_samples, replace=False)
            pairs = pairs[subset]
        elif sampling == 'chami_2':
            n_nodes =0 ### added to avoid compilation error
            num_samples = int(num_samples)
            all_nodes = np.arange(n_nodes)
            mesh = np.array(np.meshgrid(all_nodes, all_nodes))
            pairs = mesh.T.reshape(-1, 2)
            pairs = pairs[pairs[:, 0] < pairs[:, 1]]
            n_pairs = pairs.shape[0]

            print("Generating all pairs superset")
            k_base = int(num_samples / n_pairs)
            k_rem = num_samples - (k_base * n_pairs)
            subset = np.random.choice(np.arange(n_pairs), k_rem, replace=False)
            pairs_rem = pairs[subset]
            pairs_base = np.repeat(np.expand_dims(pairs, 0), k_base, axis=0).reshape((-1, 2))
            pairs = np.concatenate([pairs_base, pairs_rem], axis=0)
            num_samples = pairs.shape[0]
            triples = np.concatenate([pairs, np.random.randint(n_nodes, size=(num_samples, 1))], axis=1)
        else:
            raise ValueError(f"Invalid sampling strategy: {sampling}")

        # Compute the pairwise distances
        distances = compute_pairwise_distances(embeddings[sampled_indices, :], self.manifold)
        similarties = torch.reciprocal(1 + distances)
        similarties_sum = similarties.sum()

        # Compute the pairwise lcas
        lcas = self.compute_lca_depth(embeddings[sampled_indices, :])

        # Keep track of the storage of triplets
        lookup_table = {idx: pos for pos,idx in enumerate(np_sampled_indices)}
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
        total_loss = triplet_loss + 2*similarties_sum

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
        z_hyphc_loss, triplet_loss, similarties_sum = self.compute_hyphc_loss(manifold_z, subsample_size=self.hyphc_triplet_subsampling_size)
        z_aug_hyphc_loss, triplet_loss_aug, similarties_sum_aug = self.compute_hyphc_loss(manifold_z_aug, subsample_size=self.hyphc_triplet_subsampling_size)
        hyphc_loss = (z_hyphc_loss + z_aug_hyphc_loss) / 2
        total_loss = recon_loss + self.hyphc_weight * hyphc_loss


        for key, value in grad_norm(self, norm_type=2).items():
            if key == "grad_2.0_norm_total":
                key = "grad_2.0_norm/total"
            self.metrics.setdefault(f"{key}", []).append(value)

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
        self.metrics.setdefault('clustering/c', []).append(self.manifold.c.detach())

        return total_loss

    def on_train_epoch_end(self) -> None:
        if not isinstance(self.logger, pl.loggers.WandbLogger):
            return
        if self.current_epoch % self.log_interval == 0:
            with torch.no_grad():
                test_loader = self.trainer.datamodule.test_dataloader()

                embeddings = []
                labels = []

                for batch_idx, (x_aug, x, y) in enumerate(test_loader):
                    x = x.view(x.size(0), -1)
                    _, z, _ = self(x.to(self.device))
                    embeddings.append(z)
                    labels.append(y)

                all_embeddings = torch.cat(embeddings, dim=0)
                all_labels = torch.cat(labels, dim=0)

                delta, diam, rel_delta = get_delta(all_embeddings, self.manifold)

                all_labels_np = all_labels.cpu().numpy()
                pdist = compute_pairwise_distances(all_embeddings, self.manifold).cpu().numpy()

                agglomerative_clustering = AgglomerativeClustering(n_clusters=4, metric='precomputed', linkage='single')
                agglomerative_clustering.fit(pdist)
                nmi = normalized_mutual_info_score(all_labels_np, agglomerative_clustering.labels_)
                ari = adjusted_rand_score(all_labels_np, agglomerative_clustering.labels_)

                # Calculate nearest neighbor accuracy
                pdist_tensor = torch.from_numpy(pdist).to(self.device)
                pdist_tensor.fill_diagonal_(float('inf'))  # Exclude self
                _, nn_indices = torch.min(pdist_tensor, dim=1)
                nn_labels = all_labels[nn_indices.to(all_labels.device)]
                label_matches = (all_labels == nn_labels).float()
                nn_accuracy = label_matches.mean()

                # Create embeddings plot
                fig = create_figure(all_embeddings, manifold=self.manifold, labels=all_labels_np)

                # Log the metrics
                self.metrics.setdefault("test/nmi", []).append(nmi)
                self.metrics.setdefault("test/ari", []).append(ari)
                self.metrics.setdefault("test/delta", []).append(delta.item())
                self.metrics.setdefault("test/diam", []).append(diam.item())
                self.metrics.setdefault("test/rel_delta", []).append(rel_delta.item())
                self.metrics.setdefault("test/nn_accuracy", []).append(nn_accuracy.item())
                self.metrics["test/Hyperbolic Embeddings"] = wandb.Image(fig)
                plt.close()

        self.metrics['clustering/epoch'] = self.current_epoch
        for key, value in self.metrics.items():
            if isinstance(value, list):
                self.metrics[key] = sum(value) / len(value)
        self.logger.experiment.log(self.metrics, step=self.global_epoch+self.current_epoch)
        self.metrics = {}
