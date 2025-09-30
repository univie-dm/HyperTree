import numpy as np
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl

from typing import Tuple, Any
from pytorch_lightning.loggers import WandbLogger


from src.evaluation import evaluation
from src.hyperbolic_math.src.manifolds import Euclidean, Hyperboloid, PoincareBall
from src.hyperbolic_math.src.optim import RiemannianAdam, RiemannianSGD
from src.hyperbolic_math.src.utils.helpers import compute_pairwise_distances


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
        weight_decay: float = 0.0,
        log_interval: int = 5,
        stage: str = 'pretrain',
        global_epoch: int = 0,
        distances_type: str = 'geodesic',
        evaluation_metrics: dict[str, bool] = {},
        activation_function: str = 'relu',
        distances_version: str = 'mobius_direct'
    ):
        super().__init__()
        self.model_type = model_type
        self.hyperbolic_encoding = hyperbolic_encoding
        self.params_dtype = params_dtype
        self.latent_dim = latent_dim
        self.manifold_name = manifold_name
        self.distances_version = distances_version
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

        if activation_function == 'relu':
            self.activation_function = torch.nn.ReLU()
        elif activation_function == 'swish':
            self.activation_function = torch.nn.SiLU()

        self.hyperbolic_decoding = hyperbolic_decoding
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.log_interval = log_interval
        self.stage = stage
        self.global_epoch = global_epoch
        self.orginal_input_shape = None
        # Evaluation & Logging
        self.metrics = {}
        self.euclidean_manifold = Euclidean()
        self.old_nmi = 0
        self.old_ari = 0
        self.distances_type = distances_type
        self.evaluation_metrics = evaluation_metrics
        self.weight_decay = weight_decay


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
            optimizer = RiemannianAdam(self.parameters(), lr=self.learning_rate, eps=5e-4, expmap_update=True, weight_decay=self.weight_decay)
        elif self.optimizer_name == 'RiemannianSGD':
            optimizer = RiemannianSGD(self.parameters(), lr=self.learning_rate, momentum=0.9, expmap_update=True, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Invalid optimizer name: {self.optimizer_name}")
        return optimizer

    def test_phase(self) -> None:
        if not isinstance(self.logger, WandbLogger):
            return
        if self.current_epoch % self.log_interval == 0 and any([i for i in self.evaluation_metrics.values()]):
            self.eval()
            with torch.no_grad():
                test_loader = self.trainer.datamodule.test_dataloader() # type: ignore

                embeddings = []
                labels = []
                str_labels = []
                labels_map = self.trainer.datamodule.get_class_names() # type: ignore

                for batch_idx, (x_aug, x, y) in enumerate(test_loader):
                    _, z = self(x.to(self.device))
                    embeddings.append(z)
                    labels.append(y)
                    if labels_map is not None:
                        batch_str_labels = []
                        for label in y:
                            label_value = labels_map[int(label)]
                            # For GTSRB, extract only the coarse/super label (second element of tuple)
                            if isinstance(label_value, tuple) and len(label_value) >= 2:
                                batch_str_labels.append(label_value[1])  # Use coarse label
                            else:
                                batch_str_labels.append(label_value)
                        str_labels.append(batch_str_labels)
                    
                if labels_map is not None:
                    # Flatten the list of lists into a single list
                    str_labels = [label for batch_labels in str_labels for label in batch_labels]
                    # Validate that str_labels has the same length as labels
                    if len(str_labels) != len(torch.cat(labels, dim=0)):
                        print(f"Warning: str_labels length ({len(str_labels)}) doesn't match labels length ({len(torch.cat(labels, dim=0))}). Using None.")
                        str_labels = None
                else:
                    str_labels = None
                    
                all_embeddings = torch.cat(embeddings, dim=0)
                all_labels = torch.cat(labels, dim=0)

                # Compute the pairwise distances
                if self.distances_type == 'geodesic':
                    distance_matrix = compute_pairwise_distances(all_embeddings, self.manifold, version=self.distances_version)
                elif self.distances_type == 'cosine_similarity':
                    all_embeddings_normalized = torch.nn.functional.normalize(all_embeddings, p=2, dim=-1)
                    similarties = torch.mm(all_embeddings_normalized, all_embeddings_normalized.T).clamp(min=-1.0, max=1.0)
                    distance_matrix = (1 - similarties) / (1 + similarties)
                else:
                    raise ValueError(f"Invalid distances_type: {self.distances_type}")

                str_labels_array = np.array(str_labels) if str_labels else None
                evaluation_results = evaluation(
                    X=all_embeddings,
                    Y=all_labels,
                    distance_matrix=distance_matrix,
                    manifold=self.manifold,
                    metrics=self.evaluation_metrics,
                    str_labels=str_labels_array
                )
                for key, value in evaluation_results.items():
                    self.metrics[f"test/{key}"] = value
                plt.close()
                self.train()

    def on_train_start(self) -> None:
        self.test_phase()
        self.metrics[f'{self.stage}/epoch'] = self.current_epoch
        for key, value in self.metrics.items():
            if isinstance(value, list):
                self.metrics[key] = sum(value) / len(value)
        if isinstance(self.logger, WandbLogger):
            self.logger.experiment.log(self.metrics, step=self.current_epoch+self.global_epoch)
        self.metrics = {}
        self.train()

    def on_train_end(self) -> None:
        self.test_phase()
        self.metrics[f'{self.stage}/epoch'] = self.current_epoch
        for key, value in self.metrics.items():
            if isinstance(value, list):
                self.metrics[key] = sum(value) / len(value)
        if isinstance(self.logger, WandbLogger):
            self.logger.experiment.log(self.metrics, step=self.current_epoch+self.global_epoch)
        self.metrics = {}
        self.train()

    def on_train_epoch_end(self) -> None:
        if self.current_epoch > 0:
            self.test_phase()
        self.metrics[f'{self.stage}/epoch'] = self.current_epoch+1
        for key, value in self.metrics.items():
            if isinstance(value, list):
                self.metrics[key] = sum(value) / len(value)
        if isinstance(self.logger, WandbLogger):
            self.logger.experiment.log(self.metrics, step=self.current_epoch+self.global_epoch+1)
        self.metrics = {}
        self.train()
