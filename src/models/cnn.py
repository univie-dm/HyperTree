import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.utilities import grad_norm
from typing import Tuple

import wandb

from src.models.abstract import AbstractModel
from src.models.utils import get_dtype_from_precision, get_hyperbolic_encoding_layer_by_name, get_hyperbolic_decoding_layer_by_name
from src.hyperbolic_math.src.manifolds import Hyperboloid


class HyperbolicConvAE(AbstractModel):
    """
    A Convolutional Autoencoder that inherits from AbstractModel and includes
    hyperbolic latent space capabilities.

    This model uses a deterministic architecture and expects a fixed input image size.
    It consists of an encoder that compresses the image, a hyperbolic layer to project
    it into a latent space, and a decoder that reconstructs the image.
    """
    def __init__(
        self,
        image_size: int = 32, # This parameter is now a strict architectural requirement.
        input_channels: int = 3, # Number of input channels (3 for RGB, 1 for grayscale)
        # Parameters to match AbstractModel
        hyperbolic_encoding: str | None = None,
        params_dtype: str = 'float32',
        latent_dim: int = 2,
        manifold_name: str = 'PoincareBall',
        manifold_precision: str = 'float32',
        curvature_magnitude: float = 1.0,
        trainable_curvature: bool = False,
        hyperbolic_decoding: str | None = None,
        optimizer_name: str = 'RiemannianAdam',
        learning_rate: float = 1e-3,
        log_interval: int = 5,
        stage: str = 'pretrain',
        global_epoch: int = 0,
        distances_type: str = 'geodesic',
        evaluation_metrics: dict[str, bool] = {},
        activation_function: str = 'relu',
        clamping_factor: float = 1.0,
        smoothing_factor: float = 50.0,
        learnable_scale: bool = False,
        FHCNN_normalize: bool = False
    ):
        super().__init__(
            model_type='convolutional',
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
        self.image_size = image_size
        self.input_channels = input_channels
        self.hyperbolic_decoding = hyperbolic_decoding
        self.hyperbolic_encoding = hyperbolic_encoding
        self.params_dtype = params_dtype
        self.latent_dim = latent_dim
        self.clamping_factor = clamping_factor
        self.smoothing_factor = smoothing_factor
        self.learnable_scale = learnable_scale
        self.FHCNN_normalize = FHCNN_normalize

        # Encoder Layers
        self.enc_conv1 = self.conv_block(self.input_channels, 32)
        self.enc_conv2 = self.conv_block(32, 64)
        self.enc_conv3 = self.conv_block(64, 128)
        self.pool = nn.MaxPool2d(2, 2)

        flattened_dim = 128 * 4 * 4
        self.enc_flatten = nn.Flatten()
        self.enc_linear = nn.Linear(flattened_dim, 64)

        self.hyperbolic_encoding_layer = get_hyperbolic_encoding_layer_by_name(
                layer_name=hyperbolic_encoding,
                manifold=self.manifold,
                input_dim=64,
                output_dim=self.latent_dim,
                params_dtype=self.params_dtype,
                clamping_factor=self.clamping_factor,
                smoothing_factor=self.smoothing_factor,
                learnable_scale=self.learnable_scale,
                FHCNN_normalize=self.FHCNN_normalize
            )

        self.hyperbolic_decoding_layer = get_hyperbolic_decoding_layer_by_name(
            layer_name=hyperbolic_decoding,
            manifold=self.manifold,
            input_dim=self.latent_dim,
            output_dim=64,
            params_dtype=self.params_dtype,
            clamping_factor=self.clamping_factor,
            smoothing_factor=self.smoothing_factor,
            learnable_scale=self.learnable_scale,
            FHCNN_normalize=self.FHCNN_normalize
        )

        # Decoder Layers
        self.dec_linear = nn.Linear(64, flattened_dim)
        self.dec_unflatten = nn.Unflatten(1, (128, 4, 4))
        
        self.dec_up1 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.dec_conv1 = self.conv_block(256, 64)

        self.dec_up2 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec_conv2 = self.conv_block(128, 32)
        
        self.dec_up3 = nn.ConvTranspose2d(32, 32, 2, stride=2)
        self.dec_conv3 = self.conv_block(64, 16)
        
        self.dec_final = nn.Conv2d(16, self.input_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """ Encodes the input image into a manifold representation and returns skip connections. """
        s1 = self.enc_conv1(x)
        p1 = self.pool(s1)
        
        s2 = self.enc_conv2(p1)
        p2 = self.pool(s2)
        
        s3 = self.enc_conv3(p2)
        p3 = self.pool(s3)
        
        flat = self.enc_flatten(p3)
        hidden = self.enc_linear(flat)
        manifold_z = self.hyperbolic_encoding_layer(hidden)
        
        return manifold_z, (s1, s2, s3)

    def decode(self, manifold_z: torch.Tensor, skips: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """ Decodes the manifold representation back into an image, using skip connections. """
        s1, s2, s3 = skips
        
        hidden = self.hyperbolic_decoding_layer(manifold_z)
        hidden = hidden.to(dtype=get_dtype_from_precision(self.params_dtype))
        
        x = self.dec_unflatten(self.dec_linear(hidden))
        
        x = self.dec_up1(x)
        x = torch.cat([x, s3], dim=1)
        x = self.dec_conv1(x)
        
        x = self.dec_up2(x)
        x = torch.cat([x, s2], dim=1)
        x = self.dec_conv2(x)
        
        x = self.dec_up3(x)
        x = torch.cat([x, s1], dim=1)
        x = self.dec_conv3(x)

        x_recon = self.dec_final(x)
        
        return x_recon

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Defines the full autoencoder forward pass with skip connections.
        Assumes input `x` has the same spatial dimensions as `image_size` in init.
        """
        z, skips = self.encode(x)
        x_recon = self.decode(z, skips)
        return x_recon, z

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """ Defines a single training step. """
        if isinstance(batch, (list, tuple)):
            _, x, _ = batch
        else:
            x = batch
            
        x_recon, _ = self(x)

        loss = F.mse_loss(x_recon, x)
        if batch_idx == 0:
            try:
                x_0 = x[0] + x[0].min().abs()
                x_0 = x_0 / x_0.max()
                x_0 = x_0*255.0
                x_recon_0 = x_recon[0] + x_recon[0].min().abs()
                x_recon_0 = x_recon_0 / x_recon_0.max()
                x_recon_0 = x_recon_0*255.0
                # Convert from [C, H, W] to [H, W, C] for wandb
                if len(x_0.shape) == 3 and x_0.shape[0] in [1, 3]:  # Check if channels-first
                    x_0_img = x_0.permute(1, 2, 0).cpu().detach().numpy().astype("uint8")
                    x_recon_0_img = x_recon_0.permute(1, 2, 0).cpu().detach().numpy().astype("uint8")
                    if x_0_img.shape[2] == 1:  # Grayscale, squeeze channel dim
                        x_0_img = x_0_img.squeeze(2)
                        x_recon_0_img = x_recon_0_img.squeeze(2)
                else:
                    x_0_img = x_0.cpu().numpy().astype("uint8")
                    x_recon_0_img = x_recon_0.cpu().numpy().astype("uint8")
                self.metrics['input_image'] = wandb.Image(x_0_img, caption="Input image")
                self.metrics['reconstructed_image'] = wandb.Image(x_recon_0_img, caption="Reconstructed image")
            except Exception as e:
                # If image logging fails, just skip it and continue training
                print(f"Warning: Failed to log images to wandb: {e}")
            

        # Check for NaN or inf in loss and gradients
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN or inf loss detected: {loss}")
            return torch.tensor(0.0, requires_grad=True, device=loss.device)
            
        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        for key, value in grad_norm(self, norm_type=2).items():
            if key == "grad_2.0_norm_total":
                key = "grad_2.0_norm/total"
            # Check for NaN/inf in gradient norms
            if isinstance(value, torch.Tensor):
                if not (torch.isnan(value) or torch.isinf(value)):
                    self.metrics.setdefault(f"{key}", []).append(value)
            else:
                # Handle scalar values
                if not (torch.isnan(torch.tensor(value)) or torch.isinf(torch.tensor(value))):
                    self.metrics.setdefault(f"{key}", []).append(value)

        self.metrics.setdefault(f"{self.stage}/recon_loss", []).append(loss)
        if hasattr(self.manifold, 'c') and hasattr(self.manifold.c, 'requires_grad') and self.manifold.c.requires_grad:
            try:
                c_tensor = self.manifold.c
                c_detached = c_tensor.detach()
                c_value = c_detached.item() if c_detached.numel() == 1 else c_detached
                self.metrics.setdefault(f"{self.stage}/c", []).append(c_value)
            except Exception as e:
                print(f"Warning: Could not log curvature value: {e}")
        
        self.log("recon_loss", loss, prog_bar=True, logger=False)

        return loss