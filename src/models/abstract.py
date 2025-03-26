import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl

from typing import Tuple, Any
from pytorch_lightning.loggers import WandbLogger
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity

from src.models.utils import dendrogram_purity
from ..hyperbolic_math.src.manifolds import Euclidean, Hyperboloid, PoincareBall
from ..hyperbolic_math.src.utils.helpers import compute_pairwise_distances, get_delta
from ..hyperbolic_math.src.utils.vis_utils import create_figure

class AbstractModel(pl.LightningModule):
    def __init__(
        self,
        manifold_name: str = 'PoincareBall',
        curvature_magnitude: float = 1.0,
        trainable_curvature: bool = False,
        latent_dim: int = 2,
        learning_rate: float = 1e-3,
        log_interval: int = 5,
        stage: str = 'pretrain',
        global_epoch: int = 0,
        learning_rate_scheduler: str | None = None,
        lr_scheduler_params: dict | None = None,
    ):
        super().__init__()
        if manifold_name == 'Euclidean' or curvature_magnitude == 0:
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
        self.stage = stage
        self.global_epoch = global_epoch
        self.learning_rate_scheduler = learning_rate_scheduler
        self.lr_scheduler_params = lr_scheduler_params or {}
        self.metrics = {}
        self.euclidean_manifold = Euclidean() # for logging
        self.old_nmi = 0
        self.old_ari = 0

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        manifold_z, tangent_z = self.encode(x)
        x_recon = self.decode(manifold_z)
        return x_recon, manifold_z, tangent_z

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        if self.learning_rate_scheduler is None:
            return optimizer
        elif self.learning_rate_scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.trainer.max_epochs if self.trainer.max_epochs is not None else 100
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        elif self.learning_rate_scheduler == 'linear':
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1,
                end_factor=0.1, 
                total_iters=self.trainer.max_epochs if self.trainer.max_epochs is not None else 100
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        elif self.learning_rate_scheduler == 'warmup_cosine':
            # Get scheduler parameters with defaults
            warmup_epochs = self.lr_scheduler_params.get('warmup_epochs', 10)
            warmup_factor = self.lr_scheduler_params.get('warmup_factor', 0.1)
            cosine_min_factor = self.lr_scheduler_params.get('cosine_min_factor', 0.01)
            cosine_cycles = self.lr_scheduler_params.get('cosine_cycles', 1)
            
            # Create a warmup scheduler that linearly increases the learning rate
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=warmup_factor,
                end_factor=1.0,
                total_iters=min(warmup_epochs, self.trainer.max_epochs // 10)  # Warmup for specified epochs or 10% of total epochs # type: ignore
            )
            
            # Create a cosine annealing scheduler
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=(self.trainer.max_epochs - warmup_scheduler.total_iters) // cosine_cycles if self.trainer.max_epochs is not None else 90,
                T_mult=1,  # Keep the same cycle length
                eta_min=self.learning_rate * cosine_min_factor
            )
            
            # Combine the schedulers
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': torch.optim.lr_scheduler.SequentialLR(
                        optimizer,
                        schedulers=[warmup_scheduler, cosine_scheduler],
                        milestones=[warmup_scheduler.total_iters]
                    ),
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        else:
            raise ValueError(f"Invalid learning rate scheduler: {self.learning_rate_scheduler}")
    
    def evaluate(self) -> None:
        if not isinstance(self.logger, WandbLogger):
            return
        if self.current_epoch % self.log_interval == 0:
            with torch.no_grad():
                test_loader = self.trainer.datamodule.test_dataloader() # type: ignore

                embeddings = []
                euclidean = []
                labels = []
                tangents = []

                for batch_idx, (x_aug, x, y) in enumerate(test_loader):
                    _, z, tangent_z = self(x.to(self.device))
                    embeddings.append(z)
                    euclidean.append(tangent_z)
                    labels.append(y)
                    tangents.append(tangent_z)

                all_embeddings = torch.cat(embeddings, dim=0)
                all_euclidean = torch.cat(euclidean, dim=0)
                all_labels = torch.cat(labels, dim=0)

                # Compute the delta hyperbolicity
                delta_eucl, diam_eucl, rel_delta_eucl = get_delta(all_euclidean, self.euclidean_manifold)
                delta_hyp, diam_hyp, rel_delta_hyp = get_delta(all_embeddings, self.manifold)

                # Move to CPU for NMI and ARI calculation
                all_euclidean_np = all_euclidean.cpu().numpy()
                all_labels_np = all_labels.cpu().numpy()

                # Compute the pairwise distances
                #dist_cos_sim = 1 - cosine_similarity(all_euclidean_np)
                #dist_geodesic = compute_pairwise_distances(all_embeddings, self.manifold).cpu().numpy()

                # Perform complete linkage clustering
                #agglomerative_clustering = AgglomerativeClustering(
                #    n_clusters=len(np.unique(all_labels_np)),
                #    metric='precomputed',
                #    linkage='complete'
                #)

                # Perform clustering with the different pairwise distances
                #pred_labels_cos_sim  = agglomerative_clustering.fit_predict(dist_cos_sim)
                #nmi_cos_sim = normalized_mutual_info_score(all_labels_np, pred_labels_cos_sim)
                #ari_cos_sim = adjusted_rand_score(all_labels_np, pred_labels_cos_sim)
                
                dp_score = dendrogram_purity(all_embeddings, all_labels, self.manifold, "complete")
        
                # Compute hierarchical purity metrics
                #purity_metrics = compute_hierarchical_purity(embeddings, labels, self.manifold, "single")
                
                # Create embeddings plot
                fig = create_figure(all_embeddings, manifold=self.manifold, labels=all_labels_np)

                # Log the metrics
                ## Euclidean hyperbolicity
                self.metrics.setdefault("test/delta_eucl", []).append(delta_eucl.item())
                self.metrics.setdefault("test/diam_eucl", []).append(diam_eucl.item())
                self.metrics.setdefault("test/rel_delta_eucl", []).append(rel_delta_eucl.item())
                ## Hyperbolic hyperbolicity
                self.metrics.setdefault("test/delta_hyp", []).append(delta_hyp.item())
                self.metrics.setdefault("test/diam_hyp", []).append(diam_hyp.item())
                self.metrics.setdefault("test/rel_delta_hyp", []).append(rel_delta_hyp.item())
                ## Cosine similarity clustering
                #self.metrics.setdefault("test/nmi_cos_sim", []).append(nmi_cos_sim)
                #self.metrics.setdefault("test/ari_cos_sim", []).append(ari_cos_sim)
                
                self.metrics.setdefault("test/dp_score", []).append(dp_score)
                #self.metrics.setdefault("test/purity_metrics", []).append(purity_metrics)

                ## NMI and ARI delta
                #self.metrics.setdefault("test/nmi_delta", []).append(nmi_cos_sim - self.old_nmi)
                #self.metrics.setdefault("test/ari_delta", []).append(ari_cos_sim - self.old_ari)
                #self.old_nmi = nmi_cos_sim
                #self.old_ari = ari_cos_sim
                ## Hyperbolic Embedding plot
                self.metrics["test/Hyperbolic Embeddings"] = wandb.Image(fig)
                plt.close()

    def on_train_epoch_end(self) -> None:
        self.evaluate()
        self.metrics[f'{self.stage}/epoch'] = self.current_epoch
        for key, value in self.metrics.items():
            if isinstance(value, list):
                self.metrics[key] = sum(value) / len(value)
        if isinstance(self.logger, WandbLogger):
            self.logger.experiment.log(self.metrics, step=self.current_epoch+self.global_epoch)
        self.metrics = {}
        self.train()

