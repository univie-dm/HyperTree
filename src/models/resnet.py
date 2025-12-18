from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from src.models.abstract import AbstractModel, FeatureScaling
from src.models.utils import (
    get_dtype_from_precision,
    get_hyperbolic_encoding_layer_by_name,
    get_hyperbolic_decoding_layer_by_name,
)
from src.hyperbolic_math.src.manifolds import Euclidean


def resize_conv3x3(in_planes: int, out_planes: int, scale: float = 1):
    """upsample + 3x3 convolution with padding to avoid checkerboard artifact."""
    if scale == 1:
        return nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
    return nn.Sequential(
        nn.Upsample(scale_factor=scale, mode="nearest"),
        nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        ),
    )


def resize_conv1x1(in_planes: int, out_planes: int, scale: float = 1):
    """upsample + 1x1 convolution to avoid checkerboard artifact."""
    if scale == 1:
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)
    return nn.Sequential(
        nn.Upsample(scale_factor=scale, mode="nearest"),
        nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False),
    )


class DecoderBlock(nn.Module):
    """ResNet block with resize convolutions for decoder."""

    def __init__(self, inplanes, planes, scale=1, upsample=None):
        super().__init__()
        self.conv1 = resize_conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = resize_conv3x3(inplanes, planes, scale)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class HyperbolicResNet(AbstractModel):
    def __init__(
        self,
        projection_layer_dims: list[int],
        model_type: str = "resnet18",
        hyperbolic_encoding: str = "EuclideanLinear_Exp",
        params_dtype: str = "float32",
        latent_dim: int = 2,
        manifold_name: str = "PoincareBall",
        manifold_precision: str = "float32",
        curvature_magnitude: float = 1.0,
        trainable_curvature: bool = False,
        hyperbolic_decoding: str = "Log_EuclideanLinear",
        output_size: int = 32,
        optimizer_name: str = "Adam",
        learning_rate: float = 1e-3,
        log_interval: int = 5,
        stage: str = "pretrain",
        global_epoch: int = 0,
        evaluation_metrics: dict[str, bool] = {},
        activation_function: str = "relu",
        clamping_factor: float = 1.0,
        smoothing_factor: float = 50.0,
        distances_type: str = "geodesic",
        learnable_scale: bool = False,
        normalize: bool = False,
        contrastive_enabled: bool = False,
        contrastive_weight: float = 0.5,
        contrastive_temperature: float = 0.1,
        lr_scheduler_params: dict | None = None,
        byol: bool = False,
        recon_weight: float = 1.0,
        FHCNN_normalize: bool = False,
        feature_scaling: bool = True,
        euclidean_latentspace: bool = False,
    ):
        super().__init__(
            model_type=model_type,
            hyperbolic_encoding=hyperbolic_encoding,
            params_dtype=params_dtype,
            latent_dim=latent_dim,
            manifold_name=manifold_name,
            manifold_precision=manifold_precision,
            curvature_magnitude=curvature_magnitude,
            trainable_curvature=trainable_curvature,
            hyperbolic_decoding=hyperbolic_decoding,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            log_interval=log_interval,
            stage=stage,
            global_epoch=global_epoch,
            evaluation_metrics=evaluation_metrics,
            activation_function=activation_function,
            distances_type=distances_type,
            lr_scheduler_params=lr_scheduler_params,
            clamping_factor=clamping_factor,
            smoothing_factor=smoothing_factor,
            FHCNN_normalize=FHCNN_normalize,
            feature_scaling=feature_scaling,
        )
        self.output_size = output_size
        self.clamping_factor = clamping_factor
        self.smoothing_factor = smoothing_factor
        self.learnable_scale = learnable_scale
        self.normalize = normalize
        self.contrastive_enabled = contrastive_enabled
        self.contrastive_weight = contrastive_weight
        self.recon_weight = recon_weight
        self.contrastive_temperature = contrastive_temperature
        self.byol = byol
        self.projection_layer_dims = projection_layer_dims
        self.euclidean_latentspace = euclidean_latentspace

        self.save_hyperparameters()

        # Decoder params
        # ResNet18 and ResNet34 use BasicBlock (expansion=1)
        # ResNet50, ResNet101, ResNet152 use Bottleneck (expansion=4)
        if self.model_type in ["resnet18", "resnet34"]:
            self.expansion = 1
        elif self.model_type in ["resnet50", "resnet101", "resnet152"]:
            self.expansion = 4
        else:
            self.expansion = 1
        self.inplanes = 512 * self.expansion

        # Encoder
        if self.model_type == "resnet18":
            self.encoder = resnet18(weights=None)
        elif self.model_type == "resnet34":
            self.encoder = resnet34(weights=None)
        elif self.model_type == "resnet50":
            self.encoder = resnet50(weights=None)
        elif self.model_type == "resnet101":
            self.encoder = resnet101(weights=None)
        elif self.model_type == "resnet152":
            self.encoder = resnet152(weights=None)
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")

        # Modify encoder for CIFAR-10/100 (32x32 images)
        if self.output_size == 32:
            print(
                "Modifying ResNet encoder for CIFAR-10/100 (32x32 images).", flush=True
            )
            self.encoder.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.encoder.maxpool = nn.Identity()

        in_features = self.encoder.fc.in_features

        self.encoder.fc = nn.Identity()

        # Build projection head layers progressively
        projection_layers: list[nn.Module] = [nn.Linear(in_features, 1000)]
        current_dim = 1000
        for dim in self.projection_layer_dims:
            projection_layers.extend(
                [
                    nn.Linear(current_dim, dim),
                    nn.BatchNorm1d(dim),
                    self.activation_function,
                ]
            )
            current_dim = dim

        if self.euclidean_latentspace:
            self.manifold = Euclidean()

        self.encoder = nn.Sequential(self.encoder, *projection_layers[:-2])
        self.rms = nn.RMSNorm(current_dim, elementwise_affine=False)
        if self.feature_scaling and not isinstance(self.manifold, Euclidean):
            self.feature_scaling_layer = FeatureScaling(current_dim, self.manifold.c, dtype=self.manifold.dtype)

        # Add hyperbolic encoding layer(s)
        if isinstance(self.manifold, Euclidean) or hyperbolic_encoding is None:
            print(
                "Using Euclidean manifold or hyperbolic_encoding is 'None', "
                "hyperbolic encoding layers will default to nn.Linear.",
                flush=True,
            )
            self.hyperbolic_encoding_layer = nn.Linear(current_dim, self.latent_dim)
        else:
            self.hyperbolic_encoding_layer = get_hyperbolic_encoding_layer_by_name(
                layer_name=hyperbolic_encoding,
                manifold=self.manifold,
                input_dim=current_dim,
                output_dim=self.latent_dim,
                params_dtype=self.params_dtype,
                clamping_factor=self.clamping_factor,
                smoothing_factor=self.smoothing_factor,
                learnable_scale=self.learnable_scale,
                FHCNN_normalize=self.normalize,
            )

        # Contrastive projection head (separate from encoder, following SimCLR)
        # This head is only used for contrastive learning and discarded after pretraining
        if self.contrastive_enabled:
            print(
                "Adding contrastive projection head for SimCLR-style learning.",
                flush=True,
            )

        # Add hyperbolic decoding layer(s)
        if isinstance(self.manifold, Euclidean) or hyperbolic_decoding is None:
            print(
                "Using Euclidean manifold or hyperbolic_encoding is 'None', "
                "hyperbolic encoding layers will default to nn.Linear.",
                flush=True,
            )
            self.hyperbolic_decoding_layer = nn.Linear(
                self.latent_dim, self.inplanes * 4 * 4
            )
        else:
            self.hyperbolic_decoding_layer = get_hyperbolic_decoding_layer_by_name(
                layer_name=hyperbolic_decoding,
                manifold=self.manifold,
                input_dim=self.latent_dim,
                output_dim=self.inplanes * 4 * 4,
                params_dtype=self.params_dtype,
                clamping_factor=self.clamping_factor,
                smoothing_factor=self.smoothing_factor,
                learnable_scale=self.learnable_scale,
                FHCNN_normalize=self.normalize,
            )

        # Decoder
        if self.output_size == 28:  # For MNIST
            self.decoder = nn.Sequential(
                self._make_decoder_layer(256, 2, scale=2),  # 8x8
                self._make_decoder_layer(128, 2, scale=2),  # 16x16
                self._make_decoder_layer(
                    64, 2, scale=1.75
                ),  # 28x28 (using scale=1.75 instead of 2)
                nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False),
                nn.Tanh(),
            )
        elif self.output_size == 224:  # For ImageNet
            self.decoder = nn.Sequential(
                self._make_decoder_layer(256, 2, scale=2),  # 8x8
                self._make_decoder_layer(128, 2, scale=2),  # 16x16
                self._make_decoder_layer(64, 2, scale=2),  # 32x32
                self._make_decoder_layer(32, 2, scale=2),  # 64x64
                self._make_decoder_layer(16, 2, scale=2),  # 128x128
                self._make_decoder_layer(8, 2, scale=1.75),  # 224x224
                nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=1, bias=False),
                nn.Tanh(),
            )
        else:  # For 32x32 images
            self.decoder = nn.Sequential(
                self._make_decoder_layer(256, 2, scale=2),  # 8x8
                self._make_decoder_layer(128, 2, scale=2),  # 16x16
                self._make_decoder_layer(64, 2, scale=2),  # 32x32
                nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False),
                nn.Tanh(),
            )

        if self.recon_weight == 0.0:
            print(
                "recon_weight is set to 0.0, decoder will not be used during training.",
                flush=True,
            )
            self.decoder = None
            self.hyperbolic_decoding_layer = None

    def _make_decoder_layer(
        self, planes: int, blocks: int, scale: float = 1
    ) -> nn.Sequential:
        upsample = None
        if scale != 1 or self.inplanes != planes:
            upsample = nn.Sequential(
                resize_conv1x1(self.inplanes, planes, scale), nn.BatchNorm2d(planes)
            )
        layers = []
        layers.append(DecoderBlock(self.inplanes, planes, scale, upsample))  # type: ignore
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(DecoderBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(x)
        hidden = self.rms(hidden)
        if self.feature_scaling and not isinstance(self.manifold, Euclidean):
            hidden, scaler = self.feature_scaling_layer(hidden)
            if self.feature_scaling_layer.version == "scalar":
                self.metrics.setdefault("pretrain/feature_scale", []).append(
                    scaler.detach().item()
                )
        manifold_z = self.hyperbolic_encoding_layer(hidden)
        return manifold_z

    def decode(self, manifold_z: torch.Tensor) -> torch.Tensor:
        x = self.hyperbolic_decoding_layer(manifold_z)  # type: ignore
        # Cast back from the manifold precision to the model's precision
        x = x.to(dtype=get_dtype_from_precision(self.params_dtype))
        x = x.view(-1, 512 * self.expansion, 4, 4)
        x_recon = self.decoder(x)  # type: ignore
        return x_recon

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        manifold_z = self.encode(x)
        if self.recon_weight == 0.0:
            x_recon = torch.ones_like(x)
        else:
            x_recon = self.decode(manifold_z)
        return x_recon, manifold_z

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x_aug_1, x_aug_2, x, _ = batch
        loss = 0.0
        if self.recon_weight == 0.0:
            # When recon_weight is 0.0, skip decoding for efficiency
            x_recon = torch.ones_like(x)
        else:
            x_recon, z = self(x)

            # Reconstruction loss
            x_recon_loss = F.mse_loss(x_recon, x)
            loss = x_recon_loss
        # Contrastive loss (if enabled)
        if self.contrastive_enabled:
            if self.byol:
                z_x = self.encode(x_aug_1)
                z_aug_2 = self.encode(x_aug_2)
            else:
                z_x = self.encode(x)
                z_aug_2 = self.encode(x_aug_1)
                # z_aug_2 = self.augment_hyperbolic_features(self.encode(x))

            if not torch.isfinite(z_x).all() or not torch.isfinite(z_aug_2).all():
                raise ValueError(
                    "Non-finite values encountered in contrastive embeddings."
                )

            contrastive_loss = self.compute_contrastive_loss(
                z_x,
                z_aug_2,
                temperature=self.contrastive_temperature,
                use_euclidean=False,
            )
            loss = self.recon_weight * loss + self.contrastive_weight * contrastive_loss
            dist_to_center = self.manifold.dist_0(z_x, version=self.distances_version).mean()
            loss = loss + dist_to_center * 0.01  # Regularization term
            self.metrics.setdefault("pretrain/dist_to_center", []).append(
                dist_to_center.detach().item()
            )
            # Log contrastive-specific metrics
            self.metrics.setdefault("pretrain/contrastive_loss", []).append(
                contrastive_loss.detach().item()
            )

        # Logging
        if self.recon_weight > 0.0:
            self.metrics.setdefault("pretrain/x_recon_loss", []).append(
                x_recon_loss.detach().item()
            )
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
            sync_dist=True,
        )
        self.metrics.setdefault("latent_space/mean", []).append(
            z_x.mean().detach().item()
        )
        self.metrics.setdefault("latent_space/std", []).append(
            z_x.std().detach().item()
        )
        self.metrics.setdefault("latent_space/max", []).append(
            z_x.max().detach().item()
        )
        self.metrics.setdefault("latent_space/min", []).append(
            z_x.min().detach().item()
        )
        self.metrics.setdefault("latent_space/norm_mean", []).append(
            z_x.norm(p=2, dim=-1).mean().detach().item()
        )

        if not isinstance(loss, torch.Tensor):
            loss = torch.tensor(loss, device=x.device)

        return loss
