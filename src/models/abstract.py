import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl

from typing import Tuple, Any
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics.pairwise import cosine_similarity

from src.models.utils import get_dtype_from_precision
from src.evaluation import evaluation
from ..hyperbolic_math.src.manifolds import Euclidean, Hyperboloid, PoincareBall
from ..hyperbolic_math.src.optim import RiemannianAdam, RiemannianSGD
from ..hyperbolic_math.src.utils.helpers import compute_pairwise_distances


class AbstractModel(pl.LightningModule):
    def __init__(
        self,
        model_type: str | None = None,
        hyperbolic_encoding: str | None = None,
        params_dtype: str = 'float32',
        latent_dim: int = 2,
        manifold_name: str = 'PoincareBall',
        manifold_precision: str | torch.dtype = 'float32',
        curvature_magnitude: float | torch.Tensor = 1.0,
        trainable_curvature: bool = False,
        hyperbolic_decoding: str | None = None,
        optimizer_name: str = 'RiemannianAdam',
        learning_rate: float = 1e-3,
        log_interval: int = 5,
        stage: str = 'pretrain',
        global_epoch: int = 0,
        learning_rate_scheduler: str | None = None,
        lr_scheduler_params: dict | None = None,
        distances_type: str = 'geodesic',
        evaluation_metrics: list[str] = None,
    ):
        super().__init__()
        self.model_type = model_type
        self.hyperbolic_encoding = hyperbolic_encoding
        self.params_dtype = params_dtype
        self.latent_dim = latent_dim
        self.manifold_name = manifold_name

        if isinstance(curvature_magnitude, torch.Tensor):
            # Ensure we have our own copy
            c_tensor = curvature_magnitude.clone().detach()
        else:
            c_tensor = torch.tensor(curvature_magnitude)

        if manifold_name == 'Euclidean' or curvature_magnitude == 0:
            self.manifold = Euclidean()
        elif manifold_name == 'Hyperboloid':
            self.manifold = Hyperboloid(c=c_tensor,
                                        trainable_c=trainable_curvature,
                                        dtype=manifold_precision)
        elif manifold_name == 'PoincareBall':
            self.manifold = PoincareBall(c=c_tensor,
                                         trainable_c=trainable_curvature,
                                         dtype=manifold_precision)
        else:
            raise ValueError(f"Invalid manifold name: {manifold_name}")
        self.hyperbolic_decoding = hyperbolic_decoding
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.log_interval = log_interval
        self.stage = stage
        self.global_epoch = global_epoch
        self.learning_rate_scheduler = learning_rate_scheduler
        self.lr_scheduler_params = lr_scheduler_params or {}
        self.orginal_input_shape = None
        # Evaluation & Logging
        self.metrics = {}
        self.euclidean_manifold = Euclidean()
        self.old_nmi = 0
        self.old_ari = 0
        self.distances_type = distances_type
        self.evaluation_metrics = evaluation_metrics


    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input tensor x into a manifold representation.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("The encode method must be implemented by subclasses.")

    def decode(self, manifold_z: torch.Tensor) -> torch.Tensor:
        """
        Decodes the manifold representation manifold_z back into the original input space.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("The decode method must be implemented by subclasses.")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        manifold_z = self.encode(x)
        x_recon = self.decode(manifold_z)
        return x_recon, manifold_z

    def configure_optimizers(self) -> Any:
        if self.optimizer_name == 'RiemannianAdam':
            optimizer = RiemannianAdam(self.parameters(), lr=self.learning_rate, eps=1e-5, expmap_update=True)
        elif self.optimizer_name == 'RiemannianSGD':
            optimizer = RiemannianSGD(self.parameters(), lr=self.learning_rate, momentum=0.9, expmap_update=True)
        else:
            raise ValueError(f"Invalid optimizer name: {self.optimizer_name}")

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
                total_iters=min(warmup_epochs, self.trainer.max_epochs // 10) # Warmup for specified epochs or 10% of total epochs
            )

            # Create a cosine annealing scheduler
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=(self.trainer.max_epochs - warmup_scheduler.total_iters) // cosine_cycles if self.trainer.max_epochs is not None else 90,
                T_mult=1, # Keep the same cycle length
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
                labels = []

                for batch_idx, (x_aug, x, y) in enumerate(test_loader):
                    _, z = self(x.to(self.device))
                    embeddings.append(z)
                    labels.append(y)

                all_embeddings = torch.cat(embeddings, dim=0)
                all_labels = torch.cat(labels, dim=0)

                # Compute the pairwise distances
                if self.distances_type == 'geodesic':
                    distance_matrix = compute_pairwise_distances(all_embeddings, self.manifold)
                elif self.distances_type == 'cosine_similarity':
                    distance_matrix = torch.tensor(1 - cosine_similarity(all_embeddings.cpu().numpy()), dtype=self.manifold.dtype)
                else:
                    raise ValueError(f"Invalid distances_type: {self.distances_type}")

                evaluation_metrics = evaluation(
                    X=all_embeddings,
                    Y=all_labels,
                    distance_matrix=distance_matrix,
                    manifold=self.manifold,
                    metrics=self.evaluation_metrics
                )
                for key, value in evaluation_metrics.items():
                    self.metrics[f"test/{key}"] = value
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
