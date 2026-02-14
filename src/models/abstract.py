import numpy as np
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import wandb

from typing import Tuple, Any
from pytorch_lightning.loggers import WandbLogger

from src.evaluation import evaluation
from src.hyperbolic_math.src.manifolds import Euclidean, Hyperboloid, PoincareBall
from src.hyperbolic_math.src.utils.helpers import compute_pairwise_distances
from src.hyperbolic_math.src.utils.math_utils import atanh
from src.utils.wordnet_hierarchy import get_wordnet_labels_for_cifar100
from src.utils.imagenet_hierarchy import get_wordnet_labels_for_imagenet


class AbstractModel(pl.LightningModule):
    def __init__(
        self,
        # Architecture
        model_type: str | None = None,
        params_dtype: str = "float32",
        activation_function: str = "swish",
        feature_scaling: bool = True,
        hyperbolic_encoding: str | None = None,
        latent_dim: int = 32,
        manifold_name: str = "PoincareBall",
        manifold_precision: str | torch.dtype = "float64",
        curvature_magnitude: float | torch.Tensor = 1.0,
        hyperbolic_decoding: str | None = None,
        # Optimizer
        optimizer_name: str = "Adam",
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        lr_scheduler_params: dict | None = None,
        # Evaluation & Logging
        stage: str = "pretrain",
        global_epoch: int = 0,
        log_interval: int = 5,
        evaluation_metrics: dict[str, bool] = {},
        # (Hyperbolic-) Layer specific parameters
        clamping_factor: float = 1.0,
        smoothing_factor: float = 50.0,
        hyperboloid_layer_learnable_scale: bool = False,
        FHCNN_normalize: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["manifold_precision", "model"])

        if activation_function == "relu":
            self.activation_function = torch.nn.ReLU()
        elif activation_function == "swish":
            self.activation_function = torch.nn.SiLU()

        if isinstance(curvature_magnitude, torch.Tensor):
            # Ensure we have our own copy
            c_tensor = curvature_magnitude.clone().detach()
        else:
            c_tensor = torch.tensor(curvature_magnitude)

        if manifold_name == "Euclidean" or curvature_magnitude == 0:
            self.manifold = Euclidean()
        elif manifold_name == "Hyperboloid":
            self.manifold = Hyperboloid(c=c_tensor, dtype=manifold_precision, dist_version="smoothened")
        elif manifold_name == "PoincareBall":
            self.manifold = PoincareBall(c=c_tensor, dtype=manifold_precision)
        else:
            raise ValueError(f"Invalid manifold name: {manifold_name}")

        self.metrics = {}

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

    def compute_contrastive_loss(self, z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
        """
        Compute simplified NT-Xent (normalized temperature-scaled cross entropy) loss
        for hyperbolic or Euclidean embeddings. Memory-efficient implementation.

        Args:
            z_i: Embeddings from original images, shape (batch_size, latent_dim)
            z_j: Embeddings from augmented images, shape (batch_size, latent_dim)
            temperature: Temperature parameter for softmax

        Returns:
            Contrastive loss value
        """
        batch_size = z_i.shape[0]
        z_i = z_i.view(batch_size, -1)
        z_j = z_j.view(batch_size, -1)
        z = torch.cat([z_i, z_j], dim=0)  # Shape: (2*batch_size, latent_dim)
        # Compute pairwise geodesic distances
        distance_matrix = compute_pairwise_distances(z, self.manifold)
        similarity_matrix = -distance_matrix / temperature
        # Create labels for contrastive loss
        # For each sample i in first half, positive is at i+batch_size
        # For each sample i in second half, positive is at i-batch_size
        labels = torch.cat(
            [torch.arange(batch_size, 2 * batch_size, device=z.device),
             torch.arange(0, batch_size, device=z.device)],
            dim=0,
        )
        # Mask to remove self-similarities
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, float("-inf"))
        # Compute loss
        loss = torch.nn.functional.cross_entropy(similarity_matrix, labels)
        return loss

    def configure_optimizers(self) -> Any:
        if self.hparams.optimizer_name == "Adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.learning_rate,
                eps=1e-5,
                weight_decay=self.hparams.weight_decay,
                amsgrad=True,
            )
        elif self.hparams.optimizer_name == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.learning_rate,
                momentum=0.9,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            raise ValueError(f"Invalid optimizer name: {self.hparams.optimizer_name}")

        if self.hparams.lr_scheduler_params is None:
            return optimizer
        if self.hparams.lr_scheduler_params["type"] is None:
            return optimizer
        elif self.hparams.lr_scheduler_params["type"] == "warmup_cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.hparams.lr_scheduler_params["interval"],
                eta_min=self.hparams.lr_scheduler_params["cosine_min_factor"]
                * self.hparams.learning_rate,
            )
        elif self.hparams.lr_scheduler_params["type"] == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=self.hparams.lr_scheduler_params.get("mode", "min"),
                factor=self.hparams.lr_scheduler_params.get("factor", 0.1),
                patience=self.hparams.lr_scheduler_params.get("patience", 10),
            )
        else:
            raise ValueError(
                f"Invalid scheduler name: {self.hparams.lr_scheduler_params['scheduler_name']}"
            )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "loss"}

    def test_phase(self) -> None:
        if not isinstance(self.logger, WandbLogger):
            return
        if self.current_epoch % self.hparams.log_interval == 0 and any([i for i in self.hparams.evaluation_metrics.values()]):
            self.eval()
            with torch.no_grad():
                test_loader = self.trainer.datamodule.test_dataloader()  # type: ignore

                embeddings = []
                labels = []
                str_labels = []
                coarse_labels = []  # For hierarchical datasets like CIFAR-100, GTSRB
                labels_map = self.trainer.datamodule.get_class_names()  # type: ignore

                for batch_idx, (_, _, x, y) in enumerate(test_loader):
                    _, z = self(x.to(self.device))
                    embeddings.append(z)
                    labels.append(y)
                    if labels_map is not None:
                        batch_str_labels = []
                        batch_coarse_labels = []
                        for label in y:
                            label_value = labels_map[int(label)]
                            # Handle tuple format: (fine_label, coarse_label)
                            if isinstance(label_value, tuple) and len(label_value) >= 2:
                                fine_label, coarse_label = (label_value[0], label_value[1])
                                # For visualization, use coarse label if available
                                batch_str_labels.append(coarse_label if coarse_label is not None else fine_label)
                                # Store coarse labels for hierarchical DP
                                if coarse_label is not None:
                                    # Map string coarse label back to numeric ID
                                    # We'll need to create a mapping
                                    batch_coarse_labels.append(coarse_label)
                            else:
                                batch_str_labels.append(label_value)
                        str_labels.append(batch_str_labels)
                        if batch_coarse_labels:
                            coarse_labels.append(batch_coarse_labels)

                if labels_map is not None:
                    # Flatten the list of lists into a single list
                    str_labels = [label for batch_labels in str_labels for label in batch_labels]
                    # Validate that str_labels has the same length as labels
                    if len(str_labels) != len(torch.cat(labels, dim=0)):
                        print(
                            f"Warning: str_labels length ({len(str_labels)}) doesn't match labels length ({len(torch.cat(labels, dim=0))}). Using None."
                        )
                        str_labels = None
                else:
                    str_labels = None

                # Process coarse labels for hierarchical datasets
                coarse_labels_array = None
                if coarse_labels:
                    # Flatten coarse labels
                    coarse_labels_flat = [label for batch_labels in coarse_labels for label in batch_labels]
                    if len(coarse_labels_flat) == len(torch.cat(labels, dim=0)):
                        # Create numeric mapping for coarse labels
                        unique_coarse = sorted(set(coarse_labels_flat))
                        coarse_to_id = {label: i for i, label in enumerate(unique_coarse)}
                        coarse_labels_array = np.array([coarse_to_id[label] for label in coarse_labels_flat])
                    else:
                        print(
                            f"Warning: coarse_labels length ({len(coarse_labels_flat)}) doesn't match labels length ({len(torch.cat(labels, dim=0))}). Using None."
                        )

                all_embeddings = torch.cat(embeddings, dim=0)
                all_labels = torch.cat(labels, dim=0)

                # Override with WordNet labels if configured for CIFAR-100
                use_wordnet = self.hparams.evaluation_metrics.get("use_wordnet_labels", False)
                if use_wordnet:
                    try:
                        # Get dataset name from datamodule
                        dataset_name = None
                        if hasattr(self.trainer, 'datamodule') and self.trainer.datamodule is not None:  # type: ignore
                            datamodule_hparams = getattr(self.trainer.datamodule, 'hparams', None)  # type: ignore
                            if datamodule_hparams and hasattr(datamodule_hparams, 'dataset_name'):
                                dataset_name = datamodule_hparams.dataset_name

                        # Apply WordNet labels based on dataset
                        if dataset_name == "cifar100":
                            wordnet_path = self.hparams.evaluation_metrics.get("wordnet_hierarchy_path", "./data/cifar100_wordnet_hierarchy.txt")
                            download_if_missing = self.hparams.evaluation_metrics.get("wordnet_download_if_missing", True)

                            # Ensure types are correct
                            if not isinstance(wordnet_path, str):
                                wordnet_path = str(wordnet_path)
                            if not isinstance(download_if_missing, bool):
                                download_if_missing = bool(download_if_missing)

                            wordnet_labels, _ = get_wordnet_labels_for_cifar100(
                                all_labels.cpu().numpy(),
                                hierarchy_file=wordnet_path,
                                download_if_missing=download_if_missing
                            )
                            coarse_labels_array = wordnet_labels

                        elif dataset_name == "imagenet":
                            imagenet_hierarchy_path = self.hparams.evaluation_metrics.get("imagenet_hierarchy_path", "./data/imagenet_wordnet_hierarchy.txt")
                            setup_if_missing = self.hparams.evaluation_metrics.get("wordnet_download_if_missing", True)

                            # Ensure types are correct
                            if not isinstance(imagenet_hierarchy_path, str):
                                imagenet_hierarchy_path = str(imagenet_hierarchy_path)
                            if not isinstance(setup_if_missing, bool):
                                setup_if_missing = bool(setup_if_missing)

                            wordnet_labels, _ = get_wordnet_labels_for_imagenet(
                                all_labels.cpu().numpy(),
                                hierarchy_file=imagenet_hierarchy_path,
                                setup_if_missing=setup_if_missing
                            )
                            coarse_labels_array = wordnet_labels
                            
                    except Exception as e:
                        print(f"Warning: Failed to load WordNet labels: {e}")
                        print("Falling back to standard coarse labels (if available)")

                # Compute the (non-smoothened) pairwise distances
                distance_matrix = compute_pairwise_distances(all_embeddings, self.manifold, version="default")

                mask = torch.triu(torch.ones_like(distance_matrix), diagonal=1).bool()
                plt.hist(distance_matrix[mask].cpu().numpy().flatten(), bins=50)
                self.metrics["test/distance_histogram"] = wandb.Image(plt)
                plt.clf()

                self.metrics["test/mean_distance"] = distance_matrix[mask].mean().item()
                self.metrics["test/std_distance"] = distance_matrix[mask].std().item()
                self.metrics["test/min_distance"] = distance_matrix[mask].min().item()
                self.metrics["test/max_distance"] = distance_matrix[mask].max().item()

                str_labels_array = np.array(str_labels) if str_labels else None

                evaluation_results = evaluation(
                    X=all_embeddings,
                    Y=all_labels,
                    distance_matrix=distance_matrix,
                    manifold=self.manifold,
                    metrics=self.hparams.evaluation_metrics,
                    str_labels=str_labels_array,
                    Y_coarse=coarse_labels_array,
                )

                del all_embeddings, all_labels, distance_matrix, embeddings, labels

                for key, value in evaluation_results.items():
                    self.metrics[f"test/{key}"] = value
                plt.close()
                self.train()

    def on_train_start(self) -> None:
        self.test_phase()
        self.metrics[f"{self.hparams.stage}/epoch"] = self.current_epoch
        # Read learning rate from the optimizer's param groups (consistent with other hooks)
        try:
            self.metrics[f"{self.hparams.stage}/lr"] = self.optimizers().param_groups[0]["lr"]  # type: ignore
        except Exception:
            # Fallback to stored learning_rate if optimizer not yet available
            self.metrics[f"{self.hparams.stage}/lr"] = getattr(self.hparams, "learning_rate", None)
        for key, value in self.metrics.items():
            if isinstance(value, list):
                self.metrics[key] = sum(value) / len(value)
        if isinstance(self.logger, WandbLogger):
            self.logger.experiment.log(self.metrics, step=self.current_epoch + self.hparams.global_epoch)
        self.metrics = {}
        self.train()

    def on_train_end(self) -> None:
        self.test_phase()
        self.metrics[f"{self.hparams.stage}/epoch"] = self.current_epoch
        self.metrics[f"{self.hparams.stage}/lr"] = self.optimizers().param_groups[0]["lr"]  # type: ignore
        for key, value in self.metrics.items():
            if isinstance(value, list):
                self.metrics[key] = sum(value) / len(value)
        if isinstance(self.logger, WandbLogger):
            self.logger.experiment.log(self.metrics, step=self.current_epoch + self.hparams.global_epoch)
        self.metrics = {}
        self.train()

    def on_train_epoch_end(self) -> None:
        if self.current_epoch > 0:
            self.test_phase()
        self.metrics[f"{self.hparams.stage}/epoch"] = self.current_epoch + 1
        self.metrics[f"{self.hparams.stage}/lr"] = self.optimizers().param_groups[0]["lr"]  # type: ignore
        for key, value in self.metrics.items():
            if isinstance(value, list):
                self.metrics[key] = sum(value) / len(value)
        if isinstance(self.logger, WandbLogger):
            self.logger.experiment.log(self.metrics, step=self.current_epoch + self.hparams.global_epoch + 1)
        self.metrics = {}
        self.train()

    def compute_lca_depth(self, points: torch.Tensor, batch_size: int = 1_000_000, version: str = 'ours') -> torch.Tensor:
        """Generalized, corrected, and numerically stabilized implementation of
        Chami et al.'s (2020) LCA depth computation using inverse geometry."""
        device = points.device
        n = points.shape[0]
        c: torch.Tensor = self.manifold.c  # type: ignore
        lcas = torch.zeros((n, n), dtype=points.dtype, device=device)

        # Pre-compute point norms version=='ours'
        if version == 'ours':
            point_norms = points.norm(p=2, dim=-1, keepdim=True)
            point_dists_0 = self.manifold.dist_0(points)

        # Get the upper triangular indices and process in batches
        row_idx, col_idx = torch.triu_indices(n, n, 1, device=device)
        total_pairs = row_idx.shape[0]
        for start_idx in range(0, total_pairs, batch_size):
            end_idx = min(start_idx + batch_size, total_pairs)
            batch_row = row_idx[start_idx:end_idx]
            batch_col = col_idx[start_idx:end_idx]

            x = points[batch_row]
            y = points[batch_col]
            xy = (x * y).sum(dim=-1, keepdim=True)

            if isinstance(self.manifold, Hyperboloid):
                # Compute the orthogonal geodesic projection(s) onto the Minkowski subspace
                sqrt_c = c.sqrt()
                x0 = x.narrow(-1, 0, 1)
                y0 = y.narrow(-1, 0, 1)
                xy_mink = self.manifold._minkowski_inner(x, y)
                denom = (1 - (c * xy_mink).pow(2)).clamp(max=-1e-15)
                alpha = (sqrt_c * (x0 + c * xy_mink * y0)) / denom
                beta = (sqrt_c * (y0 + c * xy_mink * x0)) / denom
                z = alpha * x + beta * y

                # Rescale the vector to lie on the Hyperboloid
                z_mink = self.manifold._minkowski_inner(z, z)
                lca = z / (-c * z_mink).clamp(min=1e-15).sqrt()
            elif isinstance(self.manifold, PoincareBall):
                # Compute the center of geodesic circle Gamma through x and y
                x2 = x.pow(2).sum(dim=-1, keepdim=True)
                y2 = y.pow(2).sum(dim=-1, keepdim=True)

                denom = (2 * c * (x2 * y2 - xy.pow(2))).clamp(min=1e-15)
                alpha = ((1 + c * x2) * y2 - (1 + c * y2) * xy) / denom
                beta = ((1 + c * y2) * x2 - (1 + c * x2) * xy) / denom
                z = alpha * x + beta * y

                # Compute the lca and its distance to the origin w.r.t. the full geodesic
                z2 = z.pow(2).sum(dim=-1, keepdim=True)
                z_norm = z.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-15)

                # Computation via conjugation to avoid numerical issues
                sqrt_term = (z2 - 1 / c).clamp(min=1e-15).sqrt()
                denom = c * z_norm * (z_norm + sqrt_term).clamp(min=1e-15)
                lca = z / denom
            else:
                raise ValueError(f"Unsupported manifold: {self.manifold}")

            lca_depth = self.manifold.dist_0(lca)

            # Apply LCA correction
            if version == 'ours':
                # Use pre-computed norms
                x_norm = point_norms[batch_row]
                y_norm = point_norms[batch_col]

                # Compute the cosine of the angles at the origin in the triangles {x, origin, c}, {y, origin, c}  {x, origin, y}
                xlca = (x * lca).sum(dim=-1, keepdim=True)
                ylca = (y * lca).sum(dim=-1, keepdim=True)
                lca_norm = lca.norm(p=2, dim=-1, keepdim=True)

                cos_x0c = (xlca / (x_norm * lca_norm).clamp(min=1e-15)).clamp(-1.0, 1.0)
                cos_y0c = (ylca / (y_norm * lca_norm).clamp(min=1e-15)).clamp(-1.0, 1.0)
                cos_x0y = (xy / (x_norm * y_norm).clamp(min=1e-15)).clamp(-1.0, 1.0)

                # Get the lca depth w.r.t. the geodesic segment between x and y
                condition = (cos_x0y > cos_x0c) | (cos_x0y > cos_y0c) | (cos_x0y > 1-1e-15)  # True iff lca lies outside
                lca_depth = torch.where(condition, torch.min(point_dists_0[batch_row], point_dists_0[batch_col]), lca_depth)
                lca_depth = torch.where(cos_x0y == -1, torch.zeros_like(lca_depth), lca_depth)  # Handle antipodal points

            lca_depth = lca_depth.reshape(-1)
            lcas[batch_row, batch_col] = lca_depth
            lcas[batch_col, batch_row] = lca_depth

        return lcas

class FeatureScaling(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        curvature: torch.Tensor,
        max_norm: float = 0.99,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.dim = dim
        max_scale = atanh(torch.tensor(max_norm).to(dtype=dtype)) / curvature.sqrt()
        self.register_buffer('max_scale', max_scale)
        self.scale = torch.nn.Linear(self.dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / np.sqrt(x.shape[-1])
        max_scale: torch.Tensor = self.max_scale  # type: ignore
        normalized_scale = torch.nn.functional.sigmoid(self.scale(x)) * max_scale
        x = normalized_scale * x
        return x
