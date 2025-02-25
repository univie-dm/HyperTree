import torch
import pytorch_lightning as pl
import numpy as np

from typing import Tuple
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from ..hyperbolic_math.src.manifolds import Euclidean, Hyperboloid, PoincareBall
from ..hyperbolic_math.src.utils.helpers import compute_pairwise_distances
from ..hyperbolic_math.src.utils.math_utils import arsinh, sinh


class AbstractModel(pl.LightningModule):
    def __init__(
        self,
        manifold_name: str = 'PoincareBall',
        curvature_magnitude: float = 1.0,
        trainable_curvature: bool = False,
        latent_dim: int = 2,
        learning_rate: float = 1e-3,
        hyphc_weight: float = 0.1,
        temperature: float = 0.05,
        log_interval: int = 5
    ):
        super().__init__()
        if manifold_name == 'Euclidean':
            self.manifold = Euclidean()
        elif manifold_name == 'Hyperboloid':
            self.manifold = Hyperboloid(c=torch.tensor(curvature_magnitude), trainable_c=trainable_curvature)
        elif manifold_name == 'PoincareBall':
            self.manifold = PoincareBall(c=torch.tensor(curvature_magnitude), trainable_c=trainable_curvature)
        else:
            raise ValueError(f"Invalid manifold name: {manifold_name}")
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.hyphc_weight = hyphc_weight
        self.temperature = temperature
        self.log_interval = log_interval
        self.metrics = {}

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def decode(self, manifold_z: torch.Tensor) -> torch.Tensor:
        pass

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        manifold_z, tangent_z = self.encode(x)
        x_recon = self.decode(manifold_z)
        return x_recon, manifold_z, tangent_z

    def training_step(self, batch: torch.Tensor, batch_idx):
        # TODO: typing?
        pass

    def validation_step(self, batch: torch.Tensor, batch_idx):
        # TODO: typing?
        pass

    def test_step(self, batch: torch.Tensor, batch_idx):
        # TODO: typing?
        pass

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def on_train_epoch_end(self):
        # TODO: typing?
        pass

    def on_validation_epoch_end(self):
        # TODO: typing?
        pass

    def on_train_epoch_end(self) -> None:
        if self.current_epoch % self.log_interval == 0:
            with torch.no_grad():
                test_loader = self.trainer.datamodule.test_dataloader()

                embeddings = []
                labels = []

                for batch_idx, (x_aug, x, y) in enumerate(test_loader):
                    x = x.view(x.size(0), -1)
                    _, z, _ = self(x.to(self.device))
                    embeddings.append(z.cpu())
                    labels.append(y.cpu())

                    all_embeddings = torch.cat(embeddings, dim=0)
                    all_labels = torch.cat(labels, dim=0)

                    # Convert to numpy
                    all_embeddings_np = all_embeddings.cpu().numpy()
                    all_labels_np = all_labels.cpu().numpy()

                    # Move to CPU for dendrogram purity computation
                    all_labels = all_labels.cpu()

                    agglomerative_clustering = AgglomerativeClustering(n_clusters=10)
                    agglomerative_clustering.fit(all_embeddings_np)
                    nmi = normalized_mutual_info_score(all_labels_np, agglomerative_clustering.labels_)
                    ari = adjusted_rand_score(all_labels_np, agglomerative_clustering.labels_)
                    # Log the metric
                    self.metrics.setdefault("pretrain/nmi", []).append(nmi)
                    self.metrics.setdefault("pretrain/ari", []).append(ari)

        self.metrics['pretrain/epoch'] = self.current_epoch
        for key, value in self.metrics.items():
            if isinstance(value, list):
                self.metrics[key] = sum(value) / len(value)
        self.logger.experiment.log(self.metrics, step=self.current_epoch)
        self.metrics = {}

    def on_train_epoch_start(self):
        # TODO: typing?
        pass

    def on_validation_epoch_start(self):
        # TODO: typing?
        pass

    def on_test_epoch_start(self):
        # TODO: typing?
        pass

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
        elif sampling == 'chami':
            pass
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
        scaled_softmax = torch.exp((lca_triplets - lca_triplet_max) / self.temperature)
        scaled_softmax = scaled_softmax / scaled_softmax.sum(dim=1, keepdim=True)

        # Compute the triplet loss for the subsampled triplets
        sim_stack = torch.stack((similarties[i,j], similarties[i,k], similarties[j,k]), dim=1)
        triplet_loss = torch.sum(sim_stack * (1 - scaled_softmax))

        # Compute total loss
        total_loss = triplet_loss + 2*similarties_sum

        return total_loss, triplet_loss, similarties_sum
