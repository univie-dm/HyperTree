import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import wandb

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
        params_dtype: str = "float32",
        latent_dim: int = 2,
        manifold_name: str = "PoincareBall",
        manifold_precision: str | torch.dtype = "float32",
        curvature_magnitude: float | torch.Tensor = 1.0,
        trainable_curvature: bool = False,
        hyperbolic_decoding: str | None = None,
        optimizer_name: str = "RiemannianAdam",
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        log_interval: int = 5,
        stage: str = "pretrain",
        global_epoch: int = 0,
        distances_type: str = "geodesic",
        evaluation_metrics: dict[str, bool] = {},
        activation_function: str = "relu",
        distances_version: str = "mobius_direct",
        lr_scheduler_params: dict | None = None,
        clamping_factor: float = 1.0,
        smoothing_factor: float = 50.0,
        learnable_scale: bool = False,
        FHCNN_normalize: bool = False,
        feature_scaling: bool = True,
    ):
        super().__init__()
        self.model_type = model_type
        self.hyperbolic_encoding = hyperbolic_encoding
        self.params_dtype = params_dtype
        self.latent_dim = latent_dim
        self.manifold_name = manifold_name
        self.distances_version = distances_version
        self.clamping_factor = clamping_factor
        self.smoothing_factor = smoothing_factor
        self.learnable_scale = learnable_scale
        self.FHCNN_normalize = FHCNN_normalize
        self.feature_scaling = feature_scaling

        if isinstance(curvature_magnitude, torch.Tensor):
            # Ensure we have our own copy
            c_tensor = curvature_magnitude.clone().detach()
        else:
            c_tensor = torch.tensor(curvature_magnitude)

        if manifold_name == "Euclidean" or curvature_magnitude == 0:
            self.manifold = Euclidean()
        elif manifold_name == "Hyperboloid":
            self.manifold = Hyperboloid(
                c=c_tensor, trainable_c=trainable_curvature, dtype=manifold_precision
            )
        elif manifold_name == "PoincareBall":
            self.manifold = PoincareBall(
                c=c_tensor, trainable_c=trainable_curvature, dtype=manifold_precision
            )
        else:
            raise ValueError(f"Invalid manifold name: {manifold_name}")

        if activation_function == "relu":
            self.activation_function = torch.nn.ReLU()
        elif activation_function == "swish":
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
        self.lr_scheduler_params = lr_scheduler_params
        self.save_hyperparameters()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input tensor x into a manifold representation.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError(
            "The encode method must be implemented by subclasses."
        )

    def decode(self, manifold_z: torch.Tensor) -> torch.Tensor:
        """
        Decodes the manifold representation manifold_z back into the original input space.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError(
            "The decode method must be implemented by subclasses."
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        manifold_z = self.encode(x)
        x_recon = self.decode(manifold_z)
        return x_recon, manifold_z

    def compute_contrastive_loss(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        temperature: float = 0.1,
        use_euclidean: bool = False,
    ) -> torch.Tensor:
        """
        Compute simplified NT-Xent (normalized temperature-scaled cross entropy) loss
        for hyperbolic or Euclidean embeddings. Memory-efficient implementation.

        Args:
            z_i: Embeddings from original images, shape (batch_size, latent_dim)
            z_j: Embeddings from augmented images, shape (batch_size, latent_dim)
            temperature: Temperature parameter for softmax
            use_euclidean: If True, use cosine similarity instead of geodesic distance
                          (useful when z_i, z_j are from projection head in Euclidean space)

        Returns:
            Contrastive loss value
        """
        batch_size = z_i.shape[0]
        z_i = z_i.view(batch_size, -1)
        z_j = z_j.view(batch_size, -1)
        z = torch.cat([z_i, z_j], dim=0)  # Shape: (2*batch_size, latent_dim)

        # Compute pairwise distances/similarities
        if use_euclidean or self.distances_type == "cosine_similarity":
            # Use cosine similarity for Euclidean space (SimCLR approach)
            z_normalized = torch.nn.functional.normalize(z, p=2, dim=-1)
            similarity_matrix = torch.mm(z_normalized, z_normalized.T) / temperature
        elif self.distances_type == "geodesic":
            # Use geodesic distance for hyperbolic embeddings
            distance_matrix = compute_pairwise_distances(
                z, self.manifold, version=self.distances_version
            )
            # Clamp distances to prevent numerical instability
            distance_matrix = distance_matrix.clamp(max=1e5)  # Prevent overflow in exp
            similarity_matrix = -distance_matrix / temperature
        else:
            raise ValueError(f"Invalid distances_type: {self.distances_type}")

        # Create labels for contrastive loss
        # For each sample i in first half, positive is at i+batch_size
        # For each sample i in second half, positive is at i-batch_size
        labels = torch.cat(
            [
                torch.arange(batch_size, 2 * batch_size, device=z.device),
                torch.arange(0, batch_size, device=z.device),
            ],
            dim=0,
        )

        # Mask to remove self-similarities
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, float("-inf"))

        # Compute log-softmax
        log_prob = torch.nn.functional.log_softmax(similarity_matrix, dim=1)

        # Compute loss
        loss = -log_prob[torch.arange(2 * batch_size, device=z.device), labels].mean()

        return loss

    def configure_optimizers(self) -> Any:
        if self.optimizer_name == "RiemannianAdam":
            optimizer = RiemannianAdam(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.learning_rate,
                eps=5e-4,
                expmap_update=True,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name == "RiemannianSGD":
            optimizer = RiemannianSGD(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.learning_rate,
                momentum=0.9,
                expmap_update=True,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f"Invalid optimizer name: {self.optimizer_name}")

        if self.lr_scheduler_params is None:
            return optimizer
        if self.lr_scheduler_params["type"] is None:
            return optimizer
        elif self.lr_scheduler_params["type"] == "warmup_cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.lr_scheduler_params["interval"],
                eta_min=self.lr_scheduler_params["cosine_min_factor"]
                * self.learning_rate,
            )
        elif self.lr_scheduler_params["type"] == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=self.lr_scheduler_params.get("mode", "min"),
                factor=self.lr_scheduler_params.get("factor", 0.1),
                patience=self.lr_scheduler_params.get("patience", 10),
            )
        else:
            raise ValueError(
                f"Invalid scheduler name: {self.lr_scheduler_params['scheduler_name']}"
            )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "loss"}

    def test_phase(self) -> None:
        if not isinstance(self.logger, WandbLogger):
            return
        if self.current_epoch % self.log_interval == 0 and any(
            [i for i in self.evaluation_metrics.values()]
        ):
            self.eval()
            with torch.no_grad():
                test_loader = self.trainer.datamodule.test_dataloader()  # type: ignore

                embeddings = []
                labels = []
                str_labels = []
                coarse_labels = []  # For hierarchical datasets like CIFAR-100, GTSRB
                labels_map = self.trainer.datamodule.get_class_names()  # type: ignore

                for batch_idx, (_, _, x, y) in enumerate(test_loader):
                    z = self(x.to(self.device))
                    embeddings.append(z[-1])
                    labels.append(y)
                    if labels_map is not None:
                        batch_str_labels = []
                        batch_coarse_labels = []
                        for label in y:
                            label_value = labels_map[int(label)]
                            # Handle tuple format: (fine_label, coarse_label)
                            if isinstance(label_value, tuple) and len(label_value) >= 2:
                                fine_label, coarse_label = (
                                    label_value[0],
                                    label_value[1],
                                )
                                # For visualization, use coarse label if available
                                batch_str_labels.append(
                                    coarse_label
                                    if coarse_label is not None
                                    else fine_label
                                )
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
                    str_labels = [
                        label for batch_labels in str_labels for label in batch_labels
                    ]
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
                    coarse_labels_flat = [
                        label
                        for batch_labels in coarse_labels
                        for label in batch_labels
                    ]
                    if len(coarse_labels_flat) == len(torch.cat(labels, dim=0)):
                        # Create numeric mapping for coarse labels
                        unique_coarse = sorted(set(coarse_labels_flat))
                        coarse_to_id = {
                            label: i for i, label in enumerate(unique_coarse)
                        }
                        coarse_labels_array = np.array(
                            [coarse_to_id[label] for label in coarse_labels_flat]
                        )
                    else:
                        print(
                            f"Warning: coarse_labels length ({len(coarse_labels_flat)}) doesn't match labels length ({len(torch.cat(labels, dim=0))}). Using None."
                        )

                all_embeddings = torch.cat(embeddings, dim=0)
                all_labels = torch.cat(labels, dim=0)

                # Compute the pairwise distances
                if self.distances_type == "geodesic":
                    distance_matrix = compute_pairwise_distances(
                        all_embeddings, self.manifold, version=self.distances_version
                    )
                elif self.distances_type == "cosine_similarity":
                    all_embeddings_normalized = torch.nn.functional.normalize(
                        all_embeddings, p=2, dim=-1
                    )
                    similarties = torch.mm(
                        all_embeddings_normalized, all_embeddings_normalized.T
                    ).clamp(min=-1.0, max=1.0)
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
        self.metrics[f"{self.stage}/epoch"] = self.current_epoch
        # Read learning rate from the optimizer's param groups (consistent with other hooks)
        try:
            self.metrics[f"{self.stage}/lr"] = self.optimizers().param_groups[0]["lr"]  # type: ignore
        except Exception:
            # Fallback to stored learning_rate if optimizer not yet available
            self.metrics[f"{self.stage}/lr"] = getattr(self, "learning_rate", None)
        for key, value in self.metrics.items():
            if isinstance(value, list):
                self.metrics[key] = sum(value) / len(value)
        if isinstance(self.logger, WandbLogger):
            self.logger.experiment.log(
                self.metrics, step=self.current_epoch + self.global_epoch
            )
        self.metrics = {}
        self.train()

    def on_train_end(self) -> None:
        self.test_phase()
        self.metrics[f"{self.stage}/epoch"] = self.current_epoch
        self.metrics[f"{self.stage}/lr"] = self.optimizers().param_groups[0]["lr"]  # type: ignore
        for key, value in self.metrics.items():
            if isinstance(value, list):
                self.metrics[key] = sum(value) / len(value)
        if isinstance(self.logger, WandbLogger):
            self.logger.experiment.log(
                self.metrics, step=self.current_epoch + self.global_epoch
            )
        self.metrics = {}
        self.train()

    def on_train_epoch_end(self) -> None:
        if self.current_epoch > 0:
            self.test_phase()
        self.metrics[f"{self.stage}/epoch"] = self.current_epoch + 1
        self.metrics[f"{self.stage}/lr"] = self.optimizers().param_groups[0]["lr"]  # type: ignore
        for key, value in self.metrics.items():
            if isinstance(value, list):
                self.metrics[key] = sum(value) / len(value)
        if isinstance(self.logger, WandbLogger):
            self.logger.experiment.log(
                self.metrics, step=self.current_epoch + self.global_epoch + 1
            )
        self.metrics = {}
        self.train()

    def _log_images_to_wandb(
        self, x: torch.Tensor, x_recon: torch.Tensor, x_aug: torch.Tensor | None = None
    ):
        """Log original and reconstructed images to wandb."""
        try:
            x_0 = x[0] + x[0].min().abs()
            x_0 = x_0 / x_0.max()
            x_0 = x_0 * 255.0

            x_recon_0 = x_recon[0] + x_recon[0].min().abs()
            x_recon_0 = x_recon_0 / x_recon_0.max()
            x_recon_0 = x_recon_0 * 255.0

            # Handle augmented images (could be concatenated x_aug_1 and x_aug_2)
            x_aug_1_0 = None
            x_aug_2_0 = None
            if x_aug is not None and x_aug.size(0) > 0:  # Check that x_aug is not empty
                batch_size = x.size(0)
                # First half is x_aug_1, second half is x_aug_2 (due to concatenation)
                x_aug_1_0 = x_aug[0] + x_aug[0].min().abs()
                x_aug_1_0 = x_aug_1_0 / x_aug_1_0.max()
                x_aug_1_0 = x_aug_1_0 * 255.0

                if x_aug.size(0) > batch_size:  # We have both aug views concatenated
                    x_aug_2_0 = x_aug[batch_size] + x_aug[batch_size].min().abs()
                    x_aug_2_0 = x_aug_2_0 / x_aug_2_0.max()
                    x_aug_2_0 = x_aug_2_0 * 255.0

            # Convert from [C, H, W] to [H, W, C] for wandb
            if len(x_0.shape) == 3 and x_0.shape[0] in [
                1,
                3,
            ]:  # Check if channels-first
                x_0_img = x_0.permute(1, 2, 0).cpu().detach().numpy().astype("uint8")
                x_recon_0 = (
                    (x_recon_0 + x_recon_0.min().abs()) / x_recon_0.max() * 255.0
                )
                x_recon_0_img = (
                    x_recon_0.permute(1, 2, 0).cpu().detach().numpy().astype("uint8")
                )

                if x_aug_1_0 is not None:
                    x_aug_1_0_img = (
                        x_aug_1_0.permute(1, 2, 0)
                        .cpu()
                        .detach()
                        .numpy()
                        .astype("uint8")
                    )
                    if x_aug_1_0_img.shape[2] == 1:  # Grayscale, squeeze channel dim
                        x_aug_1_0_img = x_aug_1_0_img.squeeze(2)

                if x_aug_2_0 is not None:
                    x_aug_2_0_img = (
                        x_aug_2_0.permute(1, 2, 0)
                        .cpu()
                        .detach()
                        .numpy()
                        .astype("uint8")
                    )
                    if x_aug_2_0_img.shape[2] == 1:  # Grayscale, squeeze channel dim
                        x_aug_2_0_img = x_aug_2_0_img.squeeze(2)

                if x_0_img.shape[2] == 1:  # Grayscale, squeeze channel dim
                    x_0_img = x_0_img.squeeze(2)
                    x_recon_0_img = x_recon_0_img.squeeze(2)
            else:
                x_0_img = ((x_0 + x_0.min().abs()) / x_0.max()) * 255.0
                x_recon_0 = (
                    (x_recon_0 + x_recon_0.min().abs()) / x_recon_0.max()
                ) * 255.0
                x_0_img = x_0.cpu().numpy().astype("uint8")
                x_recon_0_img = x_recon_0.cpu().numpy().astype("uint8")
                if x_aug_1_0 is not None:
                    x_aug_1_0 = (
                        (x_aug_1_0 + x_aug_1_0.min().abs()) / x_aug_1_0.max()
                    ) * 255.0
                    x_aug_1_0_img = x_aug_1_0.cpu().numpy().astype("uint8")
                if x_aug_2_0 is not None:
                    x_aug_2_0 = (
                        (x_aug_2_0 + x_aug_2_0.min().abs()) / x_aug_2_0.max()
                    ) * 255.0
                    x_aug_2_0_img = x_aug_2_0.cpu().numpy().astype("uint8")

            # Create combined image: [original | aug1 | aug2 | reconstruction]
            if x_aug_1_0 is not None and x_aug_2_0 is not None:
                # Show all four: original, aug1, aug2, reconstruction
                x_combined = torch.cat(
                    (x_0, x_aug_1_0, x_aug_2_0, x_recon_0), dim=2
                )  # Concatenate along width
            elif x_aug_1_0 is not None:
                # Show three: original, aug1, reconstruction
                x_combined = torch.cat(
                    (x_0, x_aug_1_0, x_recon_0), dim=2
                )  # Concatenate along width
            else:
                # Show two: original, reconstruction
                x_combined = torch.cat(
                    (x_0, x_recon_0), dim=2
                )  # Concatenate along width

            x_combined = x_combined + x_combined.min().abs()
            x_combined = x_combined / x_combined.max()
            x_combined = x_combined * 255.0
            if len(x_combined.shape) == 3 and x_combined.shape[0] in [
                1,
                3,
            ]:  # Check if channels-first
                x_combined = (
                    (x_combined + x_combined.min().abs()) / x_combined.max() * 255.0
                )
                x_combined_img = (
                    x_combined.permute(1, 2, 0).cpu().detach().numpy().astype("uint8")
                )
                if x_combined_img.shape[2] == 1:  # Grayscale, squeeze channel dim
                    x_combined_img = x_combined_img.squeeze(2)
            else:
                x_combined_img = x_combined.cpu().numpy().astype("uint8")

            # Create caption based on what's shown
            if x_aug_1_0 is not None and x_aug_2_0 is not None:
                caption = "Original | Aug1 | Aug2 | Reconstruction"
            elif x_aug_1_0 is not None:
                caption = "Original | Aug1 | Reconstruction"
            else:
                caption = "Original | Reconstruction"

            self.metrics["test/image"] = wandb.Image(x_combined_img, caption=caption)
        except Exception as e:
            print(f"Error logging images to wandb: {e}", flush=True)
            pass

    def compute_lca_depth(
        self, points: torch.Tensor, batch_size: int = 1_000
    ) -> torch.Tensor:
        """Generalized, corrected, and numerically stabilized implementation of
        Chami et al.'s (2020) LCA depth computation using inverse geometry."""

        if not isinstance(self.manifold.c, torch.Tensor):
            self.manifold.c = torch.tensor(
                self.manifold.c, dtype=points.dtype, device=points.device
            )
        device = points.device
        lcas = torch.zeros((points.shape[0], points.shape[0]), dtype=points.dtype).to(
            device
        )
        indices = torch.triu_indices(points.shape[0], points.shape[0], 1).to(device)
        while indices.shape[1] > 0:
            x = points[indices[0, :batch_size], :]
            y = points[indices[1, :batch_size], :]
            xy = (x * y).sum(dim=-1, keepdim=True)

            if isinstance(self.manifold, Hyperboloid):
                # Compute the orthogonal geodesic projection(s) onto the Minkowski subspace
                sqrt_c = self.manifold.c.sqrt()
                x0 = x.narrow(-1, 0, 1)
                y0 = y.narrow(-1, 0, 1)
                xy_mink = self.manifold._minkowski_inner(x, y)
                denom = 1 - (self.manifold.c * xy_mink).pow(2).clamp(min=1e-15)
                alpha = (sqrt_c * (x0 + self.manifold.c * xy_mink * y0)) / denom
                beta = (sqrt_c * (y0 + self.manifold.c * xy_mink * x0)) / denom
                z = alpha * x + beta * y
                # Rescale the vector to lie on the Hyperboloid
                z_mink = self.manifold._minkowski_inner(z, z)
                lca = z / (-self.manifold.c * z_mink).clamp(min=1e-15).sqrt()
            elif isinstance(self.manifold, PoincareBall):
                # Compute the center of geodesic circle Gamma through x and y
                x2 = x.pow(2).sum(dim=-1, keepdim=True)
                y2 = y.pow(2).sum(dim=-1, keepdim=True)

                denom = (2 * self.manifold.c * (x2 * y2 - xy**2)).clamp(min=1e-15)
                alpha = (
                    (1 + self.manifold.c * x2) * y2 - (1 + self.manifold.c * y2) * xy
                ) / denom
                beta = (
                    (1 + self.manifold.c * y2) * x2 - (1 + self.manifold.c * x2) * xy
                ) / denom
                z = alpha * x + beta * y

                # Compute the lca and its distance to the origin w.r.t. the full geodesic
                z2 = z.pow(2).sum(dim=-1, keepdim=True)
                z_norm = z.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-15)

                # Direct computation of the lca
                # second_term = (z2 - 1/self.manifold.c).clamp(min=1e-15).sqrt()
                # lca = (1 - second_term / z_norm) * z

                # Computation via conjugation to avoid numerical issues
                sqrt_term = (z2 - 1 / self.manifold.c).clamp(min=1e-15).sqrt()
                denom = self.manifold.c * z_norm * (z_norm + sqrt_term).clamp(min=1e-15)
                lca = z / denom
            else:
                raise ValueError(f"Unsupported manifold: {self.manifold}")

            lca_depth = self.manifold.dist_0(lca, version=self.distances_version)

            # Compute the cosine of the angles at the origin in the triangles {x, origin, c}, {y, origin, c}  {x, origin, y}
            xlca = (x * lca).sum(dim=-1, keepdim=True)
            ylca = (y * lca).sum(dim=-1, keepdim=True)
            lca_norm = lca.norm(
                p=2, dim=-1, keepdim=True
            )  # .clamp(min=1e-10) # see if this clamping is necessary

            x_norm = x.norm(
                p=2, dim=-1, keepdim=True
            )  # .clamp(min=1e-15) # see if this clamping is necessary
            y_norm = y.norm(
                p=2, dim=-1, keepdim=True
            )  # .clamp(min=1e-15) # see if this clamping is necessary
            cos_x0c = (xlca / (x_norm * lca_norm).clamp(min=1e-15)).clamp(
                min=-1.0, max=1.0
            )
            cos_y0c = (ylca / (y_norm * lca_norm).clamp(min=1e-15)).clamp(
                min=-1.0, max=1.0
            )
            cos_x0y = (xy / (x_norm * y_norm).clamp(min=1e-15)).clamp(min=-1.0, max=1.0)

            # Get the lca depth w.r.t. the geodesic segment between x and y
            condition = torch.logical_or(
                torch.logical_or(cos_x0y > cos_x0c, cos_x0y > cos_y0c), cos_x0y == 1
            )  # True iff the lca lies outside
            lca_depth = torch.where(
                condition,
                torch.min(self.manifold.dist_0(x), self.manifold.dist_0(y)),
                lca_depth,
            )
            lca_depth = torch.where(
                cos_x0y == -1, torch.zeros_like(lca_depth), lca_depth
            )  # Handle antipodal points

            lca_depth = lca_depth.reshape(-1)
            lcas[indices[0, :batch_size], indices[1, :batch_size]] = lca_depth
            lcas[indices[1, :batch_size], indices[0, :batch_size]] = lca_depth
            indices = indices[:, batch_size:]
        return lcas


class FeatureScaling(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        curvature: float = 1.0,
        max_norm: float = 0.95,
    ):
        super().__init__()
        self.dim = dim

        max_scale = math.atanh(max_norm) / math.sqrt(curvature)
        self.max_scale = max_scale
        self.scale = torch.nn.Parameter(torch.Tensor([1.0]))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x / np.sqrt(x.shape[-1])
        normalized_scale = torch.nn.functional.sigmoid(self.scale) * self.max_scale
        x = normalized_scale * x
        return x, normalized_scale
