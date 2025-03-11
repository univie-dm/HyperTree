import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl

from typing import Tuple
from pytorch_lightning.loggers import WandbLogger
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity
from ..hyperbolic_math.src.manifolds import Euclidean, Hyperboloid, PoincareBall
from ..hyperbolic_math.src.utils.helpers import compute_pairwise_distances, get_delta
from src.hyperbolic_math.src.utils.vis_utils import create_figure


class AbstractModel(pl.LightningModule):
    def __init__(
        self,
        manifold_name: str = 'PoincareBall',
        curvature_magnitude: float = 1.0,
        trainable_curvature: bool = False,
        latent_dim: int = 2,
        learning_rate: float = 1e-3,
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

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def on_train_epoch_end(self) -> None:
        if not isinstance(self.logger, WandbLogger):
            return
        if self.current_epoch % self.log_interval == 0:
            with torch.no_grad():
                test_loader = self.trainer.datamodule.test_dataloader()

                embeddings = []
                euclidean = []###
                labels = []

                for batch_idx, (x_aug, x, y) in enumerate(test_loader):
                    _, z, tangent_z = self(x.to(self.device))
                    embeddings.append(z)
                    euclidean.append(tangent_z)###
                    labels.append(y)

                all_embeddings = torch.cat(embeddings, dim=0)
                all_euclidean = torch.cat(euclidean, dim=0)###
                all_labels = torch.cat(labels, dim=0)

                delta, diam, rel_delta = get_delta(all_embeddings, self.manifold)

                # Move to CPU for NMI and ARI calculation
                all_embeddings_np = all_embeddings.cpu().numpy()
                all_euclidean_np = all_euclidean.cpu().numpy()###
                all_labels_np = all_labels.cpu().numpy()

                # Compute the pairwise distances
                dist_version = 'euclidean_cosine_similarity'
                if dist_version == 'euclidean_cosine_similarity':
                    dist_matrix = 1 - cosine_similarity(all_euclidean_np)
                elif dist_version == 'hyperbolic_cosine_similarity':
                    dist_matrix = 1 - cosine_similarity(all_embeddings_np)
                elif dist_version == 'geodesic':
                    dist_matrix = compute_pairwise_distances(all_embeddings)
                else:
                    raise ValueError(f"Invalid pairwise distance version {dist_version}")

                # Perform complete linkage clustering
                agglomerative_clustering = AgglomerativeClustering(
                    n_clusters=len(np.unique(all_labels_np)),
                    metric='precomputed',
                    linkage='complete'
                )

                # Perform clustering with cosine similarity on hyperbolic embeddings
                predicted_labels = agglomerative_clustering.fit_predict(dist_matrix)
                nmi = normalized_mutual_info_score(all_labels_np, predicted_labels)
                ari = adjusted_rand_score(all_labels_np, predicted_labels)

                # Create embeddings plot
                fig = create_figure(all_embeddings, manifold=self.manifold, labels=all_labels_np)

                # Log the metrics
                self.metrics.setdefault("test/nmi", []).append(nmi)
                self.metrics.setdefault("test/ari", []).append(ari)
                self.metrics.setdefault("test/delta", []).append(delta.item())
                self.metrics.setdefault("test/diam", []).append(diam.item())
                self.metrics.setdefault("test/rel_delta", []).append(rel_delta.item())
                self.metrics["test/Hyperbolic Embeddings"] = wandb.Image(fig)
                plt.close()

        self.metrics['pretrain/epoch'] = self.current_epoch
        for key, value in self.metrics.items():
            if isinstance(value, list):
                self.metrics[key] = sum(value) / len(value)
        self.logger.experiment.log(self.metrics, step=self.current_epoch)
        self.metrics = {}
