import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.utilities import grad_norm

from typing import List, Tuple
from src.models.abstract import AbstractModel
from src.models.utils import get_dtype_from_precision, get_hyperbolic_encoding_layer_by_name, get_hyperbolic_decoding_layer_by_name
from src.hyperbolic_math.src.manifolds import Euclidean


class HyperbolicAE(AbstractModel):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 256, 128],
        hyperbolic_encoding: str = 'EuclideanLinear_Exp',
        params_dtype: str = 'float32',
        latent_dim: int = 2,
        manifold_name: str = 'PoincareBall',
        manifold_precision: str = 'float32',
        curvature_magnitude: float = 1.0,
        trainable_curvature: bool = False,
        hyperbolic_decoding: str = 'Log_EuclideanLinear',
        optimizer_name: str = 'RiemannianAdam',
        learning_rate: float = 1e-3,
        log_interval: int = 5,
        distances_type: str = 'geodesic',
        stage: str = 'pretrain',
        global_epoch: int = 0,
        evaluation_metrics: dict[str, bool] = {},
        activation_function: str = 'relu',
        clamping_factor: float = 1.0,
        smoothing_factor: float = 50.0,
        learnable_scale: bool = False,
        FHCNN_normalize: bool = False,
        distances_version: str = 'mobius_direct',
        spectral_normalization: bool = False,
        lr_scheduler_params: dict | None = None
    ):
        super().__init__(
            model_type='feedforward',
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
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.clamping_factor = clamping_factor
        self.smoothing_factor = smoothing_factor
        self.learnable_scale = learnable_scale
        self.FHCNN_normalize = FHCNN_normalize
        self.save_hyperparameters()

        # Encoder
        encoder_layers = []
        prev_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            encoder_layers.extend([
                torch.nn.utils.spectral_norm(nn.Linear(prev_dim, hidden_dim)) if spectral_normalization else nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                self.activation_function
            ])
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers[:-2])  # Remove the last activation function
        self.encoder.extend([nn.Tanh()])  # Add a Tanh activation at the end of the encoder

        # Add hyperbolic encoding layer(s)
        if isinstance(self.manifold, Euclidean) or hyperbolic_encoding is None:
            print("Using Euclidean manifold or hyperbolic_encoding is 'None', "
                  "hyperbolic encoding layers will default to nn.Linear.", flush=True)
            self.hyperbolic_encoding_layer = nn.Linear(self.hidden_dims[-1], self.latent_dim)
        else:
            self.hyperbolic_encoding_layer = get_hyperbolic_encoding_layer_by_name(
                layer_name=hyperbolic_encoding,
                manifold=self.manifold,
                input_dim=self.hidden_dims[-1],
                output_dim=self.latent_dim,
                params_dtype=self.params_dtype,
                clamping_factor=self.clamping_factor,
                smoothing_factor=self.smoothing_factor,
                learnable_scale=self.learnable_scale,
                FHCNN_normalize=self.FHCNN_normalize
            )

        # Add hyperbolic decoding layer(s)
        if isinstance(self.manifold, Euclidean) or hyperbolic_decoding is None:
            print("Using Euclidean manifold or hyperbolic_encoding is 'None', "
                  "hyperbolic encoding layers will default to nn.Linear.", flush=True)
            self.hyperbolic_decoding_layer = nn.Sequential(
                torch.nn.utils.spectral_norm(nn.Linear(self.latent_dim, self.hidden_dims[-1])) if spectral_normalization else nn.Linear(self.latent_dim, self.hidden_dims[-1]),
                nn.BatchNorm1d(self.hidden_dims[-1]),
                self.activation_function
            )
        else:
            self.hyperbolic_decoding_layer = get_hyperbolic_decoding_layer_by_name(
                layer_name=hyperbolic_decoding,
                manifold=self.manifold,
                input_dim=self.latent_dim,
                output_dim=self.hidden_dims[-1],
                params_dtype=self.params_dtype,
                clamping_factor=self.clamping_factor,
                smoothing_factor=self.smoothing_factor,
                activation_function=self.activation_function,
                learnable_scale=self.learnable_scale,
                FHCNN_normalize=self.FHCNN_normalize
            )

        # Decoder
        decoder_layers = []
        prev_dim = self.hidden_dims[-1]
        for hidden_dim in reversed(self.hidden_dims[:-1]):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                self.activation_function
            ])
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(self.hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.original_input_shape = x.shape[1:]
        x = x.view(x.size(0), -1)
        hidden = self.encoder(x)
        hidden = hidden / (self.hidden_dims[-1] ** 0.5)  # Scale by input dimension
        manifold_z = self.hyperbolic_encoding_layer(hidden)
        return manifold_z

    def decode(self, manifold_z: torch.Tensor) -> torch.Tensor:
        x_recon = self.hyperbolic_decoding_layer(manifold_z)
        # Cast back from the manifold precision to the model's precision
        x_recon = x_recon.to(dtype=get_dtype_from_precision(self.params_dtype))
        x_recon = self.decoder(x_recon)
        x_recon = x_recon.view(-1, *self.original_input_shape)
        return x_recon

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        _, _, x, _ = batch
        # Feedforward model uses only clean images for reconstruction
        x_recon, _, = self(x)

        # Reconstruction loss
        if x_recon.shape != x.shape:
            x_recon = x_recon.view_as(x)
        x_recon_loss = F.mse_loss(x_recon, x)
        loss = x_recon_loss

        for key, value in grad_norm(self, norm_type=2).items():
            if key == "grad_2.0_norm_total":
                key = "grad_2.0_norm/total"
            self.metrics.setdefault(f"{key}", []).append(value)

        # Logging
        self.metrics.setdefault("pretrain/x_recon_loss", []).append(x_recon_loss)
        self.metrics.setdefault("pretrain/total_loss", []).append(loss)
        if isinstance(self.manifold.c, torch.Tensor):
            self.metrics.setdefault("pretrain/c", []).append(self.manifold.c.detach())
        self.log("loss", loss, logger=False, prog_bar=True, on_epoch=True, on_step=False)

        return loss
