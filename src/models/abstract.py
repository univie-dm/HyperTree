from unittest import TestLoader
import torch
import numpy as np
import pytorch_lightning as pl

from typing import Any, Mapping, Optional, Tuple, Union
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from ..hyperbolic_math.src.manifolds import Euclidean, Hyperboloid, PoincareBall
from ..hyperbolic_math.src.utils.helpers import compute_pairwise_distances, get_delta


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

    def training_step(self, batch: torch.Tensor, batch_idx) -> Optional[Union[torch.Tensor, Mapping[str, Any]]]:
        pass

    def validation_step(self, batch: torch.Tensor, batch_idx) -> Optional[Union[torch.Tensor, Mapping[str, Any]]]:
        pass

    def test_step(self, batch: torch.Tensor, batch_idx) -> Optional[Union[torch.Tensor, Mapping[str, Any]]]:
        pass

    def on_train_epoch_start(self) -> None:
        pass

    def on_train_epoch_end(self) -> None:
        if not isinstance(self.logger, pl.loggers.WandbLogger):
            return
        if self.current_epoch % self.log_interval == 0:
            with torch.no_grad():
                test_loader = self.trainer.datamodule.test_dataloader()

                embeddings = []
                labels = []

                for batch_idx, (x_aug, x, y) in enumerate(test_loader):
                    _, z, _ = self(x.to(self.device))
                    embeddings.append(z)
                    labels.append(y)

                all_embeddings = torch.cat(embeddings, dim=0)
                all_labels = torch.cat(labels, dim=0)

                delta, diam, rel_delta = get_delta(all_embeddings, self.manifold)

                # Move to CPU for NMI and ARI calculation
                
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

                # Log the metrics
                self.metrics.setdefault("test/nmi", []).append(nmi)
                self.metrics.setdefault("test/ari", []).append(ari)
                self.metrics.setdefault("test/delta", []).append(delta.item())
                self.metrics.setdefault("test/diam", []).append(diam.item())
                self.metrics.setdefault("test/rel_delta", []).append(rel_delta.item())
                self.metrics.setdefault("test/nn_accuracy", []).append(nn_accuracy.item())

        self.metrics['pretrain/epoch'] = self.current_epoch
        for key, value in self.metrics.items():
            if isinstance(value, list):
                self.metrics[key] = sum(value) / len(value)
        self.logger.experiment.log(self.metrics, step=self.current_epoch)
        self.metrics = {}

    def on_validation_epoch_start(self) -> None:
        pass

    def on_validation_epoch_end(self) -> None:
        pass

    def on_test_epoch_start(self) -> None:
        pass

    def on_test_epoch_end(self) -> None:
        pass
