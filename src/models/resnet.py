import torch
import torch.nn as nn

from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from typing import Tuple
from src.models.abstract import AbstractModel, FeatureScaling
from src.models.utils import get_hyperbolic_encoding_layer_by_name
from src.hyperbolic_math.src.manifolds import Euclidean


class HyperbolicResNet(AbstractModel):
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
        # ResNet specific parameters
        projection_layer_dims: list[int] = [512, 256, 128],
        input_size: int = 32,
        contrastive_loss_enabled: bool = False,
        contrastive_weight: float = 0.5,
        contrastive_temperature: float = 0.1,
        # Supervised learning parameters
        load_supervised_weights: bool = False,
        supervised: bool = False,
        # External checkpoint (e.g., DINO ResNet50 from Facebook)
        pretrained_weights_path: str | None = None,
    ):
        super().__init__(
            # Architecture
            model_type=model_type,
            params_dtype=params_dtype,
            activation_function=activation_function,
            feature_scaling=feature_scaling,
            hyperbolic_encoding=hyperbolic_encoding,
            latent_dim=latent_dim,
            manifold_name=manifold_name,
            manifold_precision=manifold_precision,
            curvature_magnitude=curvature_magnitude,
            hyperbolic_decoding=None, # not supported for ResNet
            # Optimizer
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lr_scheduler_params=lr_scheduler_params,
            # Evaluation & Logging
            stage=stage,
            global_epoch=global_epoch,
            log_interval=log_interval,
            evaluation_metrics=evaluation_metrics,
            # (Hyperbolic-) Layer specific parameters
            clamping_factor=clamping_factor,
            smoothing_factor=smoothing_factor,
            hyperboloid_layer_learnable_scale=hyperboloid_layer_learnable_scale,
            FHCNN_normalize=FHCNN_normalize,
        )
        self.save_hyperparameters()

        if self.hparams.load_supervised_weights:
            print("Loading supervised weights", flush=True)
            pretrained_weights = 'DEFAULT'
        else:
            pretrained_weights = None

        if self.hparams.supervised:
            print("Supervised learning enabled", flush=True)

        # Encoder
        if self.hparams.model_type == "resnet18":
            self.encoder = resnet18(weights=pretrained_weights)
        elif self.hparams.model_type == "resnet34":
            self.encoder = resnet34(weights=pretrained_weights)
        elif self.hparams.model_type == "resnet50":
            self.encoder = resnet50(weights=pretrained_weights)
        elif self.hparams.model_type == "resnet101":
            self.encoder = resnet101(weights=pretrained_weights)
        elif self.hparams.model_type == "resnet152":
            self.encoder = resnet152(weights=pretrained_weights)
        else:
            raise ValueError(f"Invalid model type: {self.hparams.model_type}")
        # Decoder
        self.decoder = None  # ResNet decoder not supported

        # Replace classification head with identity to use pooled features (2048 for ResNet50, 512 for ResNet18)
        backbone_dims = {"resnet18": 512, "resnet34": 512, "resnet50": 2048, "resnet101": 2048, "resnet152": 2048}
        backbone_dim = backbone_dims[self.hparams.model_type]
        self.encoder.fc = nn.Identity()

        # Modify encoder for CIFAR-10/100 (32x32 images)
        if self.hparams.input_size == 32:
            print("Modifying ResNet encoder for CIFAR-10/100 (32x32 images).", flush=True)
            # Overwrite the first convolutional layer and maxpooling layer
            self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.encoder.maxpool = nn.Identity()  # type: ignore
        elif self.hparams.input_size == 64:
            print("Modifying ResNet encoder for 64x64 images (removing maxpool).", flush=True)
            self.encoder.maxpool = nn.Identity()  # type: ignore

        if self.hparams.pretrained_weights_path is not None:
            print(f"Loading backbone weights from {self.hparams.pretrained_weights_path}", flush=True)
            state = torch.load(self.hparams.pretrained_weights_path, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
                state = state["model"]
            state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}
            missing, unexpected = self.encoder.load_state_dict(state, strict=False)
            print(f"Loaded external checkpoint: missing={len(missing)} unexpected={len(unexpected)}", flush=True)

        if self.hparams.load_supervised_weights or self.hparams.pretrained_weights_path is not None:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Build projection head layers progressively
        current_dim = backbone_dim
        #projection_layers: list[nn.Module] = [] # old
        projection_layers: list[nn.Module] = [nn.BatchNorm1d(current_dim), self.activation_function] # test if better
        for dim in self.hparams.projection_layer_dims:
            projection_layers.extend(
                [nn.Linear(current_dim, dim),
                 nn.BatchNorm1d(dim),
                 self.activation_function
                 ]
            )
            current_dim = dim

        self.encoder = nn.Sequential(self.encoder, *projection_layers[:-2])
        self.rms = nn.RMSNorm(current_dim, elementwise_affine=False)
        if self.hparams.feature_scaling and not isinstance(self.manifold, Euclidean):
            c: torch.Tensor = self.manifold.c  # type: ignore
            self.feature_scaling_layer = FeatureScaling(current_dim, c, dtype=self.manifold.dtype)

        # Add hyperbolic encoding layer(s)
        # Note: Euclidean manifold will default to nn.Linear
        self.hyperbolic_encoding_layer = get_hyperbolic_encoding_layer_by_name(
            layer_name=self.hparams.hyperbolic_encoding,
            manifold=self.manifold,
            input_dim=current_dim,
            output_dim=self.hparams.latent_dim,
            params_dtype=self.hparams.params_dtype,
            clamping_factor=self.hparams.clamping_factor,
            smoothing_factor=self.hparams.smoothing_factor,
            learnable_scale=self.hparams.hyperboloid_layer_learnable_scale,
            FHCNN_normalize=self.hparams.FHCNN_normalize,
        )



    def encode(self, x: torch.Tensor) -> torch.Tensor:
        hidden, manifold_z = self.encode_with_backbone(x)
        return manifold_z

    def encode_with_backbone(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (backbone_features, hyperbolic_embeddings) in a single forward pass."""
        hidden = self.encoder(x)
        hidden = self.rms(hidden)
        if self.hparams.feature_scaling and not isinstance(self.manifold, Euclidean):
            manifold_z = self.hyperbolic_encoding_layer(self.feature_scaling_layer(hidden))
        else:
            manifold_z = self.hyperbolic_encoding_layer(hidden)
        return hidden, manifold_z

    def forward(self, x: torch.Tensor) -> Tuple[None, torch.Tensor]:
        manifold_z = self.encode(x)
        return None, manifold_z

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x_aug_1, x_aug_2, x, y = batch

        if self.hparams.supervised:
            z_aug_1 = self.encode(x_aug_1)
            z_aug_2 = self.encode(x_aug_2)

            if not torch.isfinite(z_aug_1).all() or not torch.isfinite(z_aug_2).all():
                raise ValueError("Non-finite values encountered in supervised contrastive embeddings.")

            supcon_loss = self.compute_supervised_contrastive_loss(
                z_aug_1, z_aug_2, y, temperature=self.hparams.contrastive_temperature
            )
            loss = self.hparams.contrastive_weight * supcon_loss

            # Supervised contrastive metrics
            self.metrics.setdefault("pretrain/supcon_loss", []).append(supcon_loss.detach().item())
            dist_to_center = self.manifold.dist_0(z_aug_1).mean()
            self.metrics.setdefault("pretrain/dist_to_center", []).append(dist_to_center.detach().item())
        else:
            # Backbone features only — skip hyperbolic head during pretraining.
            # The hyperbolic head has enough capacity to minimize contrastive loss
            # without the backbone improving (gradient absorption). Euclidean-only
            # contrastive forces clean gradient through the backbone. The hyperbolic
            # head is trained from scratch during the clustering stage.
            h_aug_1 = self.rms(self.encoder(x_aug_1))
            h_aug_2 = self.rms(self.encoder(x_aug_2))

            if not torch.isfinite(h_aug_1).all() or not torch.isfinite(h_aug_2).all():
                raise ValueError("Non-finite values encountered in backbone features.")

            euclidean_contrastive_loss = self._compute_euclidean_contrastive_loss(
                h_aug_1, h_aug_2, temperature=self.hparams.contrastive_temperature
            )
            loss = self.hparams.contrastive_weight * euclidean_contrastive_loss

            self.metrics.setdefault("pretrain/euclidean_contrastive_loss", []).append(euclidean_contrastive_loss.detach().item())
            z_aug_1 = h_aug_1  # backbone features for latent space stats below

        # Logging
        self.metrics.setdefault("pretrain/total_loss", []).append(loss.detach().item())  # type: ignore
        self.metrics.setdefault("pretrain/c", []).append(
            self.manifold.c.detach().item()  # type: ignore
            if self.manifold.c.numel() == 1  # type: ignore
            else self.manifold.c.detach().cpu()  # type: ignore
        )  # type: ignore
        self.log(
            "loss",
            loss,
            logger=False,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=False,
        )
        self.metrics.setdefault("latent_space/mean", []).append(z_aug_1.mean().detach().item())
        self.metrics.setdefault("latent_space/std", []).append(z_aug_1.std().detach().item())
        self.metrics.setdefault("latent_space/max", []).append(z_aug_1.max().detach().item())
        self.metrics.setdefault("latent_space/min", []).append(z_aug_1.min().detach().item())
        self.metrics.setdefault("latent_space/norm_mean", []).append(z_aug_1.norm(p=2, dim=-1).mean().detach().item())

        if self.hparams.hyperbolic_encoding in ["HyperbolicLinearHyperboloidFHCNN", "HyperbolicLinearHyperboloidFHNN"]:
            scale = self.hyperbolic_encoding_layer[1].scale  # type: ignore
            self.metrics.setdefault("hyperbolic_encoding_layer/scale", []).append(scale.detach().item())

        if not isinstance(loss, torch.Tensor):
            loss = torch.tensor(loss, device=x.device)

        # Safety net: replace NaN/Inf with 0 while preserving the computation graph
        if not torch.isfinite(loss):
            print(f"Warning: Non-finite loss detected at epoch {self.current_epoch}, batch {batch_idx}. Replacing with 0.", flush=True)
        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

        return loss
