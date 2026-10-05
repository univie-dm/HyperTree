import numpy as np
import torch
import pytorch_lightning as pl

from pathlib import Path
from typing import Tuple, Any
from pytorch_lightning.loggers import WandbLogger

from src.evaluation import evaluation, build_hierarchy_clusters, write_results
from src.hyperbolic_math.src.manifolds import Euclidean, Hyperboloid, PoincareBall
from src.hyperbolic_math.src.utils.helpers import compute_pairwise_distances
from src.hyperbolic_math.src.utils.math_utils import atanh
from src.utils.wordnet_hierarchy import get_wordnet_labels_for_cifar100
from src.utils.imagenet_hierarchy import get_wordnet_labels_for_imagenet, get_labels_at_hierarchy_level, load_imagenet_hierarchy, setup_tiny_imagenet_hierarchy


class AbstractModel(pl.LightningModule):
    def __init__(
        self,
        # Architecture
        model_type: str | None = None,
        params_dtype: str = "float32",
        activation_function: str = "swish",
        feature_scaling: bool = True,
        hyperbolic_encoding: str | None = "HyperbolicLinearPP",
        latent_dim: int = 32,
        manifold_name: str = "PoincareBall",
        manifold_precision: str | torch.dtype = "float64",
        curvature_magnitude: float | torch.Tensor = 1.0,
        # Optimizer
        optimizer_name: str = "Adam",
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        lr_scheduler_params: dict | None = None,
        # Evaluation & Logging
        stage: str = "pretrain",
        global_epoch: int = 0,
        log_interval: int = 10,
        evaluation_metrics: dict[str, bool] = {},
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
        # Modules whose parameters are frozen; they are kept in eval mode (see train())
        self.frozen_modules: list[torch.nn.Module] = []

    def train(self, mode: bool = True):
        """Sets the training mode, except for the frozen modules, which stay in eval mode.

        In train mode BatchNorm normalizes with batch statistics and updates its running
        statistics, so the features of a frozen backbone would still change during training
        and differ from the ones used at evaluation. Keeping the frozen modules in eval mode
        makes their features the same during training and evaluation.
        """
        super().train(mode)
        for module in self.frozen_modules:
            module.eval()
        return self

    def encode_with_backbone(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (backbone_features, manifold_embeddings) in a single forward pass.

        Subclasses provide the modules: encoder, rms, hyperbolic_encoding_layer and,
        with feature scaling on a hyperbolic manifold, feature_scaling_layer.
        """
        hidden = self.encoder(x)  # type: ignore
        hidden = self.rms(hidden)  # type: ignore
        return hidden, self.embed_features(hidden)

    def embed_features(self, hidden: torch.Tensor) -> torch.Tensor:
        """Maps (RMS-normalized) backbone features onto the manifold: feature scaling and hyperbolic layer."""
        if self.hparams.feature_scaling and not isinstance(self.manifold, Euclidean):
            hidden = self.feature_scaling_layer(hidden)  # type: ignore
        return self.hyperbolic_encoding_layer(hidden)  # type: ignore

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes the input tensor x into a manifold representation."""
        _, manifold_z = self.encode_with_backbone(x)
        return manifold_z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        manifold_z = self.encode(x)
        return manifold_z

    @staticmethod
    def _gather_with_backprop(z: torch.Tensor) -> torch.Tensor:
        """Gather tensors from all DDP ranks, keeping gradients for the local shard."""
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return z
        world_size = torch.distributed.get_world_size()
        if world_size == 1:
            return z
        # Gather from all ranks (no grad through remote copies)
        gathered = [torch.zeros_like(z) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, z.contiguous())
        # Overwrite the local shard so its gradients flow through
        rank = torch.distributed.get_rank()
        gathered[rank] = z
        return torch.cat(gathered, dim=0)

    @staticmethod
    def _local_view_distances(distance_matrix: torch.Tensor, view: int, local_batch_size: int) -> torch.Tensor:
        """Pairwise distances among the local embeddings of one contrastive view.

        Selects the block of view 0 (z_i) or view 1 (z_j) from the distance matrix returned by
        compute_contrastive_loss, which under DDP covers the gathered batch of all ranks.
        """
        global_batch_size = distance_matrix.shape[0] // 2
        rank = 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        start = view * global_batch_size + rank * local_batch_size
        block = slice(start, start + local_batch_size)
        return distance_matrix[block, block]

    def compute_contrastive_loss(
        self, z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute simplified NT-Xent (normalized temperature-scaled cross entropy) loss
        for hyperbolic or Euclidean embeddings.

        Under DDP, embeddings are gathered across all ranks so every GPU sees the
        full global batch of negatives.  Gradients only flow through the local shard.

        Args:
            z_i: Embeddings of the first view, shape (batch_size, latent_dim)
            z_j: Embeddings of the second view, shape (batch_size, latent_dim)
            temperature: Temperature parameter for softmax

        Returns:
            Contrastive loss value and the pairwise geodesic distance matrix of [z_i; z_j]
            (with gradients), so callers can reuse it instead of recomputing the distances
            (see _local_view_distances).
        """
        local_batch_size = z_i.shape[0]
        z_i = z_i.view(local_batch_size, -1)
        z_j = z_j.view(local_batch_size, -1)

        # Gather across GPUs so every rank sees all negatives
        z_i = self._gather_with_backprop(z_i)
        z_j = self._gather_with_backprop(z_j)
        batch_size = z_i.shape[0]  # global batch after gather

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
        return loss, distance_matrix

    def _compute_euclidean_contrastive_loss(self, h_i: torch.Tensor, h_j: torch.Tensor, temperature: float) -> torch.Tensor:
        """NT-Xent with cosine similarity on Euclidean features — bounded, no boundary drift."""
        h_i = self._gather_with_backprop(h_i)
        h_j = self._gather_with_backprop(h_j)
        batch_size = h_i.shape[0]
        h = torch.cat([h_i, h_j], dim=0)
        h = torch.nn.functional.normalize(h, p=2, dim=1)
        sim = h @ h.T / temperature
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=h.device)
        sim = sim.masked_fill(mask, float("-inf"))
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=h.device),
            torch.arange(0, batch_size, device=h.device),
        ])
        return torch.nn.functional.cross_entropy(sim, labels)

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
            warmup_epochs = self.hparams.lr_scheduler_params["interval"]
            total_epochs = self.trainer.max_epochs
            eta_min = (
                self.hparams.lr_scheduler_params["cosine_min_factor"]
                * self.hparams.learning_rate
            )
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-3, total_iters=warmup_epochs
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_epochs - warmup_epochs, eta_min=eta_min
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
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
                f"Invalid scheduler name: {self.hparams.lr_scheduler_params['type']}"
            )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "loss"}

    def test_phase(self) -> None:
        if not self.trainer.is_global_zero:
            return
        if self.current_epoch % self.hparams.log_interval == 0 and any([i for i in self.hparams.evaluation_metrics.values()]):
            self.eval()
            with torch.no_grad():
                if not hasattr(self, '_cached_test_loader'):
                    self._cached_test_loader = self.trainer.datamodule.test_dataloader()  # type: ignore
                test_loader = self._cached_test_loader

                embeddings = []
                labels = []
                for batch in test_loader:
                    embeddings.append(self(batch["original"].to(self.device)))
                    labels.append(batch["label"])
                all_embeddings = torch.cat(embeddings, dim=0)
                all_labels = torch.cat(labels, dim=0)
                labels_np = all_labels.cpu().numpy()

                datamodule = self.trainer.datamodule  # type: ignore
                dataset_name = datamodule.hparams.dataset_name
                eval_cfg = self.hparams.evaluation_metrics

                # Native coarse labels (CIFAR-100: 20 superclasses)
                coarse_labels_array = None
                native_coarse_labels = datamodule.get_coarse_labels()
                if native_coarse_labels is not None:
                    coarse_labels_array = np.array([native_coarse_labels[int(label)] for label in labels_np])

                # Build the WordNet / taxonomy hierarchy for hierarchical dendrogram purity
                hierarchy_clusters = None
                if eval_cfg.get("use_wordnet_labels", False):
                    try:
                        leaf_ids = set(np.unique(labels_np).tolist())
                        if dataset_name == "cifar100":
                            _, parent_child = get_wordnet_labels_for_cifar100(
                                labels_np,
                                hierarchy_file=str(eval_cfg.get("wordnet_hierarchy_path", "./data/cifar100_wordnet_hierarchy.txt")),
                                download_if_missing=bool(eval_cfg.get("wordnet_download_if_missing", True)),
                            )
                            hierarchy_clusters = build_hierarchy_clusters(parent_child, leaf_ids)
                            # Keep native 20 superclasses in coarse_labels_array

                        elif dataset_name == "imagenet":
                            _, parent_child = get_wordnet_labels_for_imagenet(
                                labels_np,
                                hierarchy_file=str(eval_cfg.get("imagenet_hierarchy_path", "./data/imagenet_wordnet_hierarchy.txt")),
                                setup_if_missing=bool(eval_cfg.get("wordnet_download_if_missing", True)),
                            )
                            hierarchy_clusters = build_hierarchy_clusters(parent_child, leaf_ids)
                            # Aggregate to ~20 coarse classes
                            coarse_labels_array, _ = get_labels_at_hierarchy_level(
                                labels_np, parent_child, target_num_classes=20
                            )

                        elif dataset_name == "inat18":
                            inat_hierarchy_path = str(eval_cfg.get("inat18_hierarchy_path", "./data/inat18_taxonomy_hierarchy.txt"))
                            if Path(inat_hierarchy_path).exists():
                                parent_child = load_imagenet_hierarchy(inat_hierarchy_path, exclude_root=True)
                                hierarchy_clusters = build_hierarchy_clusters(parent_child, leaf_ids)
                                # Coarse labels gate the hierarchical dendrogram purity in
                                # evaluation(); without them, only flat DP is logged.
                                coarse_labels_array, _ = get_labels_at_hierarchy_level(
                                    labels_np, parent_child, target_num_classes=20
                                )
                            else:
                                print(f"iNat18 hierarchy file not found: {inat_hierarchy_path}")

                        elif dataset_name == "tiny_imagenet":
                            tiny_hierarchy_path = str(eval_cfg.get("tiny_imagenet_hierarchy_path", "./data/tiny_imagenet_wordnet_hierarchy.txt"))
                            # Build hierarchy if missing
                            if not Path(tiny_hierarchy_path).exists():
                                setup_tiny_imagenet_hierarchy(datamodule.hparams.data_dir, tiny_hierarchy_path)

                            _, parent_child = get_wordnet_labels_for_imagenet(
                                labels_np,
                                hierarchy_file=tiny_hierarchy_path,
                                setup_if_missing=False
                            )
                            hierarchy_clusters = build_hierarchy_clusters(parent_child, leaf_ids)
                            # Aggregate to ~20 coarse classes
                            coarse_labels_array, _ = get_labels_at_hierarchy_level(
                                all_labels.cpu().numpy(), parent_child, target_num_classes=20
                            )

                    except Exception as e:
                        print(f"Warning: Failed to load WordNet labels: {e}")
                        print("Falling back to standard coarse labels (if available)")

                # Compute the (non-smoothened) pairwise distances
                distance_matrix = compute_pairwise_distances(all_embeddings, self.manifold, version="default")

                mask = torch.triu(torch.ones_like(distance_matrix), diagonal=1).bool()
                self.metrics["test/mean_distance"] = distance_matrix[mask].mean().item()
                self.metrics["test/std_distance"] = distance_matrix[mask].std().item()
                self.metrics["test/min_distance"] = distance_matrix[mask].min().item()
                self.metrics["test/max_distance"] = distance_matrix[mask].max().item()

                evaluation_results = evaluation(
                    X=all_embeddings,
                    Y=all_labels,
                    distance_matrix=distance_matrix,
                    manifold=self.manifold,
                    metrics=eval_cfg,
                    Y_coarse=coarse_labels_array,
                    hierarchy_clusters=hierarchy_clusters,
                )

                del all_embeddings, all_labels, distance_matrix, embeddings, labels

                for key, value in evaluation_results.items():
                    self.metrics[f"test/{key}"] = value
                self.train()

    def _write_metrics_to_file(self, epoch: int) -> None:
        """Write current metrics to a JSON file."""
        if not self.trainer.is_global_zero:
            return
        results_dir = self.hparams.evaluation_metrics.get("results_dir", "./results")
        write_results(
            metrics=self.metrics,
            results_dir=results_dir,
            stage=self.hparams.stage,
            epoch=epoch,
        )

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
        self._write_metrics_to_file(self.current_epoch)
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
        self._write_metrics_to_file(self.current_epoch + 1)
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
        self._write_metrics_to_file(self.current_epoch)
        self.metrics = {}
        self.train()


class FeatureScaling(torch.nn.Module):
    """
    Scaling layer to stabilize training of hybrid Euclidean-hyperbolic models.

    Reference:
    Klein Timo, et al. "Understanding and improving hyperbolic deep reinforcement learning."
        International Conference on Learning Representations. Vol. 2026. 2026.
    """
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
