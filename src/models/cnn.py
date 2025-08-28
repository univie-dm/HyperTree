import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.utilities import grad_norm
from torch.nn.utils import spectral_norm
from typing import Tuple

# Assuming AbstractModel and utils are in a path like this based on your other files
from src.models.abstract import AbstractModel
from src.models.utils import get_dtype_from_precision, get_hyperbolic_encoding_layer_by_name, get_hyperbolic_decoding_layer_by_name
from src.hyperbolic_math.src.manifolds import Euclidean


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
        clamping_factor: float = 1.0,
        smoothing_factor: float = 50.0,
        activation_function: str = 'relu'
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


        conv1 = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=32,
            kernel_size=8,
            stride=4,
            padding=1
        )
        self.conv1 = spectral_norm(conv1)
        conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.conv2 = spectral_norm(conv2)
        conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.conv3 = spectral_norm(conv3)
        flattened_dim = 64 * 3 * 3

        self.encoder = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.conv2,
            nn.ReLU(),
            self.conv3,
            nn.ReLU(),
            nn.Flatten(),
            spectral_norm(nn.Linear(flattened_dim, 32))  # Linear layer to map to latent dimension
        )

        self.hyperbolic_encoding_layer = get_hyperbolic_encoding_layer_by_name(
                layer_name=hyperbolic_encoding,
                manifold=self.manifold,
                input_dim=32,
                output_dim=self.latent_dim,
                params_dtype=self.params_dtype,
                clamping_factor=self.clamping_factor,
                smoothing_factor=self.smoothing_factor
            )

        self.hyperbolic_decoding_layer = get_hyperbolic_decoding_layer_by_name(
            layer_name=hyperbolic_decoding,
            manifold=self.manifold,
            input_dim=self.latent_dim,
            output_dim=32,
            params_dtype=self.params_dtype,
            clamping_factor=self.clamping_factor,
            smoothing_factor=self.smoothing_factor
        )

        deconv1 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.deconv1 = spectral_norm(deconv1)
        deconv2 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=4,
            stride=2,
            padding=1,
            output_padding=1
        )
        self.deconv2 = spectral_norm(deconv2)
        deconv3 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=self.input_channels,
            kernel_size=8,
            stride=4,
            padding=1,
            output_padding=2
        )
        self.deconv3 = spectral_norm(deconv3)
        self.decoder = nn.Sequential(
            spectral_norm(nn.Linear(32, flattened_dim)),
            nn.Unflatten(1, (64, 3, 3)),  # Reshape to match the output of the encoder
            self.deconv1,
            nn.ReLU(),
            self.deconv2,
            nn.ReLU(),
            self.deconv3,
        )



    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """ Encodes the input image into a manifold representation. """
        hidden = self.encoder(x)
        manifold_z = self.hyperbolic_encoding_layer(hidden)
        return manifold_z

    def decode(self, manifold_z: torch.Tensor) -> torch.Tensor:
        """ Decodes the manifold representation back into an image. """
        hidden = self.hyperbolic_decoding_layer(manifold_z)
        hidden = hidden.to(dtype=get_dtype_from_precision(self.params_dtype))
        x_recon = self.decoder(hidden)
        return x_recon

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Defines the full autoencoder forward pass.
        Assumes input `x` has the same spatial dimensions as `image_size` in init.
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        # The F.interpolate call is removed as it can be non-deterministic and
        # is not needed if input and output sizes are designed to match.
        return x_recon, z

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """ Defines a single training step. """
        if isinstance(batch, (list, tuple)):
            _, x, _ = batch
        else:
            x = batch
            
        x_recon, _ = self(x)

        loss = F.mse_loss(x_recon, x)

        for key, value in grad_norm(self, norm_type=2).items():
            if key == "grad_2.0_norm_total":
                key = "grad_2.0_norm/total"
            self.metrics.setdefault(f"{key}", []).append(value)

        self.metrics.setdefault(f"{self.stage}/recon_loss", []).append(loss)
        if self.manifold.c.requires_grad:
            self.metrics.setdefault(f"{self.stage}/c", []).append(self.manifold.c.detach())

        return loss
