import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple
from torchvision.models import resnet18, resnet34
from src.models.abstract import AbstractModel


def resize_conv3x3(in_planes: int, out_planes: int, scale: int=1):
    """upsample + 3x3 convolution with padding to avoid checkerboard artifact."""
    if scale == 1:
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
    return nn.Sequential(
        nn.Upsample(scale_factor=scale, mode='nearest'),
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
    )

def resize_conv1x1(in_planes: int, out_planes: int, scale: int=1):
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

        out += identity
        out = self.relu(out)

        return out

class HyperbolicResNet(AbstractModel):
    def __init__(
        self,
        type: str = 'resnet18',
        manifold_name: str = 'PoincareBall',
        curvature_magnitude: float = 1.0,
        trainable_curvature: bool = False,
        latent_dim: int = 2,
        learning_rate: float = 1e-3,
        hyphc_weight: float = 0.1,
        temperature: float = 0.05,
        hyphc_triplet_subsampling_size: int = 100,
        log_interval: int = 5,
        input_channels: int = 3,
        output_size: int = 32
    ):
        super().__init__(
            manifold_name=manifold_name,
            curvature_magnitude=curvature_magnitude,
            trainable_curvature=trainable_curvature,
            latent_dim=latent_dim,
            learning_rate=learning_rate,
            hyphc_weight=hyphc_weight,
            temperature=temperature,
            hyphc_triplet_subsampling_size=hyphc_triplet_subsampling_size,
            log_interval=log_interval
        )
        self.type = type
        self.input_channels = input_channels
        # Decoder params
        self.expansion = 1
        self.inplanes = 512 * self.expansion

        # Encoder
        if type == 'resnet18':
            self.encoder = resnet18(weights=None)
        elif type == 'resnet34':
            self.encoder = resnet34(weights=None)
        else:
            raise ValueError(f"Invalid model type: {type}")

        # Modify first conv layer to accept different number of input channels
        self.encoder.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.encoder_linear = nn.Linear(1000, self.latent_dim)

        # Project latent vector to starting feature map
        self.linear = nn.Linear(latent_dim, self.inplanes * 4 * 4)

        # Decoder
        if output_size == 28:  # For MNIST
            self.decoder = nn.Sequential(
                self._make_decoder_layer(256, 2, scale=2),  # 8x8
                self._make_decoder_layer(128, 2, scale=2),  # 16x16
                self._make_decoder_layer(64, 2, scale=1.75),  # 28x28 (using scale=1.75 instead of 2)
                nn.Conv2d(64, input_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.Tanh()
            )
        else:  # For 32x32 images
            self.decoder = nn.Sequential(
                self._make_decoder_layer(256, 2, scale=2),  # 8x8
                self._make_decoder_layer(128, 2, scale=2),  # 16x16
                self._make_decoder_layer(64, 2, scale=2),   # 32x32
                nn.Conv2d(64, input_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.Tanh()
            )

    def _make_decoder_layer(self, planes: int, blocks: int, scale: int=1) -> nn.Sequential:
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

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tangent_z = self.encoder(x)
        tangent_z = self.encoder_linear(tangent_z)
        manifold_z = self.manifold.expmap_0(tangent_z)
        return manifold_z, tangent_z

    def decode(self, manifold_z: torch.Tensor) -> torch.Tensor:
        tangent_z = self.manifold.logmap_0(manifold_z)
        x = self.linear(tangent_z)
        x = x.view(-1, 512 * self.expansion, 4, 4)
        x_recon = self.decoder(x)
        return x_recon

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x_aug, x, _ = batch
        
        x_recon, _, _ = self(x)
        x_aug_recon, _, _ = self(x_aug)

        # Reconstruction loss
        x_recon_loss = F.mse_loss(x_recon, x)
        x_aug_recon_loss = F.mse_loss(x_aug_recon, x_aug)
        loss = (x_recon_loss + x_aug_recon_loss) / 2

        # Logging
        self.metrics.setdefault("pretrain/x_recon_loss", []).append(x_recon_loss)
        self.metrics.setdefault("pretrain/x_aug_recon_loss", []).append(x_aug_recon_loss)
        self.metrics.setdefault("pretrain/total_loss", []).append(loss)
        self.metrics.setdefault("pretrain/c", []).append(self.manifold.c.detach())

        return loss
