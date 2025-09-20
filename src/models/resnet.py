import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, resnet34, resnet50
from src.models.abstract import AbstractModel
from src.models.utils import get_dtype_from_precision, get_hyperbolic_encoding_layer_by_name, get_hyperbolic_decoding_layer_by_name
from src.hyperbolic_math.src.manifolds import Euclidean, Hyperboloid


def resize_conv3x3(in_planes: int, out_planes: int, scale: float=1):
    """upsample + 3x3 convolution with padding to avoid checkerboard artifact."""
    if scale == 1:
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
    return nn.Sequential(
        nn.Upsample(scale_factor=scale, mode='nearest'),
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
    )

def resize_conv1x1(in_planes: int, out_planes: int, scale: float=1):
    """upsample + 1x1 convolution to avoid checkerboard artifact."""
    if scale == 1:
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)
    return nn.Sequential(
        nn.Upsample(scale_factor=scale, mode='nearest'),
        nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)
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
        model_type: str = 'resnet18',
        hyperbolic_encoding: str = 'EuclideanLinear_Exp',
        params_dtype: str = 'float32',
        latent_dim: int = 2,
        manifold_name: str = 'PoincareBall',
        manifold_precision: str = 'float32',
        curvature_magnitude: float = 1.0,
        trainable_curvature: bool = False,
        hyperbolic_decoding: str = 'Log_EuclideanLinear',
        output_size: int = 32,
        optimizer_name: str = 'RiemannianAdam',
        learning_rate: float = 1e-3,
        log_interval: int = 5,
        stage: str = 'pretrain',
        global_epoch: int = 0,
        evaluation_metrics: dict[str, bool] = {},
        activation_function: str = 'relu',
        clamping_factor: float = 1.0,
        smoothing_factor: float = 50.0,
        distances_type: str = 'geodesic',
        learnable_scale: bool = False,
        normalize: bool = False
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
            distances_type=distances_type
        )
        self.output_size = output_size
        self.clamping_factor = clamping_factor
        self.smoothing_factor = smoothing_factor
        self.learnable_scale = learnable_scale
        self.normalize = normalize
        self.save_hyperparameters()

        # Decoder params
        self.expansion = 1
        self.inplanes = 512 * self.expansion

        # Encoder
        if self.model_type == 'resnet18':
            self.encoder = resnet18(weights=None)
        elif self.model_type == 'resnet34':
            self.encoder = resnet34(weights=None)
        elif self.model_type == 'resnet50':
            self.encoder = resnet50(weights=None)
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")
        
        self.projection_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1000, 512),
            nn.BatchNorm1d(512),
            self.activation_function,
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            self.activation_function,
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            self.activation_function,
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            self.activation_function
        )

        self.encoder = nn.Sequential(self.encoder, self.projection_head)

        # Add hyperbolic encoding layer(s)
        if isinstance(self.manifold, Euclidean) or hyperbolic_encoding is None:
            print("Using Euclidean manifold or hyperbolic_encoding is 'None', "
                  "hyperbolic encoding layers will default to nn.Linear.", flush=True)
            self.hyperbolic_encoding_layer = nn.Linear(64, self.latent_dim)
        else:
            self.hyperbolic_encoding_layer = get_hyperbolic_encoding_layer_by_name(
                layer_name=hyperbolic_encoding,
                manifold=self.manifold,
                input_dim=64,
                output_dim=self.latent_dim,
                params_dtype=self.params_dtype,
                clamping_factor=self.clamping_factor,
                smoothing_factor=self.smoothing_factor,
                learnable_scale=self.learnable_scale,
                normalize=self.normalize
            )

        # Add hyperbolic decoding layer(s)
        if isinstance(self.manifold, Euclidean) or hyperbolic_decoding is None:
            print("Using Euclidean manifold or hyperbolic_encoding is 'None', "
                  "hyperbolic encoding layers will default to nn.Linear.", flush=True)
            self.hyperbolic_decoding_layer = nn.Linear(self.latent_dim, self.inplanes * 4 * 4)
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
                normalize=self.normalize
            )

        # Decoder
        if self.output_size == 28:  # For MNIST
            self.decoder = nn.Sequential(
                self._make_decoder_layer(256, 2, scale=2),  # 8x8
                self._make_decoder_layer(128, 2, scale=2),  # 16x16
                self._make_decoder_layer(64, 2, scale=1.75),  # 28x28 (using scale=1.75 instead of 2)
                nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False),
                nn.Tanh()
            )
        elif self.output_size == 224:  # For ImageNet
            self.decoder = nn.Sequential(
                self._make_decoder_layer(256, 2, scale=2),  # 8x8
                self._make_decoder_layer(128, 2, scale=2),  # 16x16
                self._make_decoder_layer(64, 2, scale=2),   # 32x32
                self._make_decoder_layer(32, 2, scale=2),   # 64x64
                self._make_decoder_layer(16, 2, scale=2),   # 128x128
                self._make_decoder_layer(8, 2, scale=1.75), # 224x224
                nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=1, bias=False),
                nn.Tanh()
            )
        else:  # For 32x32 images
            self.decoder = nn.Sequential(
                self._make_decoder_layer(256, 2, scale=2),  # 8x8
                self._make_decoder_layer(128, 2, scale=2),  # 16x16
                self._make_decoder_layer(64, 2, scale=2),   # 32x32
                nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False),
                nn.Tanh()
            )

    def _make_decoder_layer(self, planes: int, blocks: int, scale: float = 1) -> nn.Sequential:
        upsample = None
        if scale != 1 or self.inplanes != planes:
            upsample = nn.Sequential(
                resize_conv1x1(self.inplanes, planes, scale),
                nn.BatchNorm2d(planes)
            )
        layers = []
        layers.append(DecoderBlock(self.inplanes, planes, scale, upsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(DecoderBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(x)
        manifold_z = self.hyperbolic_encoding_layer(hidden)
        return manifold_z

    def decode(self, manifold_z: torch.Tensor) -> torch.Tensor:
        x = self.hyperbolic_decoding_layer(manifold_z)
        # Cast back from the manifold precision to the model's precision
        x = x.to(dtype=get_dtype_from_precision(self.params_dtype))
        x = x.view(-1, 512 * self.expansion, 4, 4)
        x_recon = self.decoder(x)
        return x_recon

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x_aug, x, _ = batch
        x_recon, _ = self(x)
        #x_aug_recon, _ = self(x_aug)

        # Reconstruction loss
        x_recon_loss = F.mse_loss(x_recon, x)
        #x_aug_recon_loss = F.mse_loss(x_aug_recon, x_aug)
        #loss = (x_recon_loss + x_aug_recon_loss) / 2
        loss = x_recon_loss

        # Logging
        self.metrics.setdefault("pretrain/x_recon_loss", []).append(x_recon_loss)
        #self.metrics.setdefault("pretrain/x_aug_recon_loss", []).append(x_aug_recon_loss)
        self.metrics.setdefault("pretrain/total_loss", []).append(loss)
        self.metrics.setdefault("pretrain/c", []).append(self.manifold.c.detach())

        return loss
