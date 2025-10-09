import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, resnet34, resnet50
import wandb
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
        normalize: bool = False,
        distances_version: str = 'mobius_direct',
        contrastive_enabled: bool = False,
        contrastive_weight: float = 0.5,
        contrastive_temperature: float = 0.1,
        lr_scheduler_params: dict | None = None
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
            distances_version=distances_version,
            lr_scheduler_params=lr_scheduler_params
        )
        self.output_size = output_size
        self.clamping_factor = clamping_factor
        self.smoothing_factor = smoothing_factor
        self.learnable_scale = learnable_scale
        self.normalize = normalize
        self.contrastive_enabled = contrastive_enabled
        self.contrastive_weight = contrastive_weight
        self.contrastive_temperature = contrastive_temperature

        self.hidden_dims = [1000, 512, 128, self.latent_dim]

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
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            self.activation_function,
            nn.Linear(128, self.latent_dim),
        )

        self.encoder = nn.Sequential(self.encoder, self.projection_head)

        # Add hyperbolic encoding layer(s)
        if isinstance(self.manifold, Euclidean) or hyperbolic_encoding is None:
            print("Using Euclidean manifold or hyperbolic_encoding is 'None', "
                  "hyperbolic encoding layers will default to nn.Linear.", flush=True)
            self.hyperbolic_encoding_layer = nn.Identity()
        else:
            self.hyperbolic_encoding_layer = get_hyperbolic_encoding_layer_by_name(
                layer_name=hyperbolic_encoding,
                manifold=self.manifold,
                input_dim=self.latent_dim,
                output_dim=self.latent_dim,
                params_dtype=self.params_dtype,
                clamping_factor=self.clamping_factor,
                smoothing_factor=self.smoothing_factor,
                learnable_scale=self.learnable_scale,
                FHCNN_normalize=self.normalize
            )
        
        # Contrastive projection head (separate from encoder, following SimCLR)
        # This head is only used for contrastive learning and discarded after pretraining
        if self.contrastive_enabled:
            print("Adding contrastive projection head for SimCLR-style learning.", flush=True)
            # Projection dimension - typically same as latent_dim or 2x latent_dim
            proj_dim = max(128, self.latent_dim * 2)
            self.contrastive_projection = nn.Sequential(
                nn.Linear(self.latent_dim + (1 if isinstance(self.manifold, Hyperboloid) else 0), 
                         proj_dim),
                nn.BatchNorm1d(proj_dim),
                self.activation_function,
                nn.Linear(proj_dim, proj_dim),
                nn.BatchNorm1d(proj_dim),
            )
        else:
            self.contrastive_projection = nn.Identity()

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
                FHCNN_normalize=self.normalize
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
        x_recon, z = self(x)        
        
        # Reconstruction loss
        x_recon_loss = F.mse_loss(x_recon, x)
        loss = x_recon_loss
        
        # Contrastive loss (if enabled)
        if self.contrastive_enabled:
            # Encode augmented images
            z_aug = self.encode(x_aug)
            
            # Apply contrastive projection head (SimCLR-style)
            # These projections are only used for contrastive loss, not reconstruction
            z_proj = self.contrastive_projection(z.view(z.size(0), -1))
            z_aug_proj = self.contrastive_projection(z_aug.view(z_aug.size(0), -1))
            
            # Use Euclidean cosine similarity since projection head outputs are in Euclidean space
            contrastive_loss = self.compute_contrastive_loss(
                z_proj, z_aug_proj, 
                temperature=self.contrastive_temperature,
                use_euclidean=True  # Projection head outputs are Euclidean
            )
            loss = loss + self.contrastive_weight * contrastive_loss
            
            # Log contrastive-specific metrics
            self.metrics.setdefault("pretrain/contrastive_loss", []).append(contrastive_loss)

        # Logging
        self.metrics.setdefault("pretrain/x_recon_loss", []).append(x_recon_loss)
        self.metrics.setdefault("pretrain/total_loss", []).append(loss)
        self.metrics.setdefault("pretrain/c", []).append(self.manifold.c.detach().item() if self.manifold.c.numel() == 1 else self.manifold.c.detach().cpu())
        self.log('recon_loss', x_recon_loss, prog_bar=True, on_epoch=True, logger=False)
        self.log("loss", loss, logger=False, prog_bar=True, on_epoch=True, on_step=False)

        if batch_idx == 0:
            self._log_images_to_wandb(x, x_recon, x_aug if self.contrastive_enabled else None)

        return loss

    def _log_images_to_wandb(self, x: torch.Tensor, x_recon: torch.Tensor, x_aug: torch.Tensor = None):
        """Log original and reconstructed images to wandb."""
        try:
            x_0 = x[0] + x[0].min().abs()
            x_0 = x_0 / x_0.max()
            x_0 = x_0*255.0
            x_aug_0 = x_aug[0] + x_aug[0].min().abs() if x_aug is not None else None
            if x_aug_0 is not None:
                x_aug_0 = x_aug_0 / x_aug_0.max()
                x_aug_0 = x_aug_0*255.0
            x_recon_0 = x_recon[0] + x_recon[0].min().abs()
            x_recon_0 = x_recon_0 / x_recon_0.max()
            x_recon_0 = x_recon_0*255.0
            # Convert from [C, H, W] to [H, W, C] for wandb
            if len(x_0.shape) == 3 and x_0.shape[0] in [1, 3]:  # Check if channels-first
                x_0_img = x_0.permute(1, 2, 0).cpu().detach().numpy().astype("uint8")
                x_recon_0_img = x_recon_0.permute(1, 2, 0).cpu().detach().numpy().astype("uint8")
                if x_aug_0 is not None:
                    x_aug_0_img = x_aug_0.permute(1, 2, 0).cpu().detach().numpy().astype("uint8")
                    if x_aug_0_img.shape[2] == 1:  # Grayscale, squeeze channel dim
                        x_aug_0_img = x_aug_0_img.squeeze(2)
                if x_0_img.shape[2] == 1:  # Grayscale, squeeze channel dim
                    x_0_img = x_0_img.squeeze(2)
                    x_recon_0_img = x_recon_0_img.squeeze(2)
                    if x_aug_0 is not None:
                        x_aug_0_img = x_aug_0_img.squeeze(2)
            else:
                x_0_img = x_0.cpu().numpy().astype("uint8")
                x_recon_0_img = x_recon_0.cpu().numpy().astype("uint8")
                if x_aug_0 is not None:
                    x_aug_0_img = x_aug_0.cpu().numpy().astype("uint8")
            if x_aug_0 is not None:
                x_combined = torch.cat((x_0, x_aug_0, x_recon_0), dim=2)  # Concatenate along width
            else:
                x_combined = torch.cat((x_0, x_recon_0), dim=2)  # Concatenate along width
            x_combined = x_combined + x_combined.min().abs()
            x_combined = x_combined / x_combined.max()
            x_combined = x_combined*255.0
            if len(x_combined.shape) == 3 and x_combined.shape[0] in [1, 3]:  # Check if channels-first
                x_combined_img = x_combined.permute(1, 2, 0).cpu().detach().numpy().astype("uint8")
                if x_combined_img.shape[2] == 1:  # Grayscale, squeeze channel dim
                    x_combined_img = x_combined_img.squeeze(2)
            else:
                x_combined_img = x_combined.cpu().numpy().astype("uint8")
            self.metrics['test/image'] = wandb.Image(x_combined_img, caption="image")
        except Exception as e:
            print(f"Error logging images to wandb: {e}", flush=True)
            pass