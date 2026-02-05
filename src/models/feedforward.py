from math import tanh
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.utilities import grad_norm

from typing import List, Tuple
from src.models.abstract import AbstractModel, FeatureScaling
from src.models.utils import (
    get_dtype_from_precision,
    get_hyperbolic_encoding_layer_by_name,
    get_hyperbolic_decoding_layer_by_name,
)
from src.hyperbolic_math.src.manifolds import Euclidean


class HyperbolicAE(AbstractModel):
    def __init__(
        self,
        # Architecture
        input_dim: int,
        hidden_dims: List[int] = [512, 256, 128],
        params_dtype: str = "float32",
        activation_function: str = "swish",
        feature_scaling: bool = True,
        hyperbolic_encoding: str | None = None,
        latent_dim: int = 2,
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
    ):
        super().__init__(
            # Architecture
            model_type="feedforward",
            params_dtype=params_dtype,
            activation_function=activation_function,
            feature_scaling=feature_scaling,
            hyperbolic_encoding=hyperbolic_encoding,
            latent_dim=latent_dim,
            manifold_name=manifold_name,
            manifold_precision=manifold_precision,
            curvature_magnitude=curvature_magnitude,
            hyperbolic_decoding=hyperbolic_decoding,
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

        # Encoder
        encoder_layers = []
        prev_dim = self.hparams.input_dim
        for hidden_dim in self.hparams.hidden_dims:
            encoder_layers.extend(
                [nn.Linear(prev_dim, hidden_dim),
                 nn.BatchNorm1d(hidden_dim),
                 self.activation_function]
            )
            prev_dim = hidden_dim
        # Remove the last activation function
        self.encoder = nn.Sequential(*encoder_layers[:-2])

        self.rms = nn.RMSNorm(self.hparams.hidden_dims[-1], elementwise_affine=False)
        if self.hparams.feature_scaling and not isinstance(self.manifold, Euclidean):
            self.feature_scaling_layer = FeatureScaling(self.hparams.hidden_dims[-1], self.manifold.c, dtype=self.manifold.dtype)

        # Add hyperbolic encoding layer(s)
        self.hyperbolic_encoding_layer = get_hyperbolic_encoding_layer_by_name(
            layer_name=self.hparams.hyperbolic_encoding,
            manifold=self.manifold,
            input_dim=self.hparams.hidden_dims[-1],
            output_dim=self.hparams.latent_dim,
            params_dtype=self.hparams.params_dtype,
            clamping_factor=self.hparams.clamping_factor,
            smoothing_factor=self.hparams.smoothing_factor,
            learnable_scale=self.hparams.hyperboloid_layer_learnable_scale,
            FHCNN_normalize=self.hparams.FHCNN_normalize,
        )

        # Add hyperbolic decoding layer(s)
        self.hyperbolic_decoding_layer = get_hyperbolic_decoding_layer_by_name(
            layer_name=self.hparams.hyperbolic_decoding,
            manifold=self.manifold,
            input_dim=self.hparams.latent_dim,
            output_dim=self.hparams.hidden_dims[-1],
            params_dtype=self.hparams.params_dtype,
            clamping_factor=self.hparams.clamping_factor,
            smoothing_factor=self.hparams.smoothing_factor,
            activation_function=self.activation_function,
            learnable_scale=self.hparams.hyperboloid_layer_learnable_scale,
            FHCNN_normalize=self.hparams.FHCNN_normalize,
        )

        # Decoder
        decoder_layers = []
        prev_dim = self.hparams.hidden_dims[-1]
        for hidden_dim in reversed(self.hparams.hidden_dims[:-1]):
            decoder_layers.extend(
                [nn.Linear(prev_dim, hidden_dim),
                 nn.BatchNorm1d(hidden_dim),
                 self.activation_function]
            )
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(self.hparams.hidden_dims[0], self.hparams.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self.original_input_shape = x.shape[1:]
        x = x.view(x.size(0), -1)
        hidden = self.encoder(x)
        hidden = self.rms(hidden)
        hidden = torch.nn.functional.tanh(hidden)
        if self.hparams.feature_scaling and not isinstance(self.manifold, Euclidean):
            hidden = self.feature_scaling_layer(hidden)
        manifold_z = self.hyperbolic_encoding_layer(hidden)
        return manifold_z

    def decode(self, manifold_z: torch.Tensor) -> torch.Tensor:
        if self.decoder is None:
            raise ValueError("No decoder was provided.")
        x_recon = self.hyperbolic_decoding_layer(manifold_z)  # type: ignore
        # Cast back from the manifold precision to the model's precision
        x_recon = x_recon.to(dtype=get_dtype_from_precision(self.hparams.params_dtype))
        x_recon = self.decoder(x_recon)  # type: ignore
        x_recon = x_recon.view(-1, *self.hparams.input_dim)
        return x_recon

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        _, _, x, _ = batch
        x = x.to(dtype=get_dtype_from_precision(self.hparams.params_dtype))
        # Feedforward model uses only clean images for reconstruction
        x_recon, manifold_z = self(x)

        # Reconstruction loss
        if x_recon.shape != x.shape:
            x_recon = x_recon.view_as(x)
        x_recon_loss = F.mse_loss(x_recon, x)
        total_loss = x_recon_loss

        # Logging - DETACH tensors to prevent memory leaks
        self.metrics.setdefault("pretrain/x_recon_loss", []).append(x_recon_loss.detach().item())
        self.metrics.setdefault("pretrain/total_loss", []).append(total_loss.detach().item())
        # Manifold curvature
        self.metrics.setdefault("pretrain/c", []).append(self.manifold.c.item())
        # Latent space statistics
        self.metrics.setdefault("latent_space/min", []).append(manifold_z.min().detach().item())
        self.metrics.setdefault("latent_space/max", []).append(manifold_z.max().detach().item())
        self.metrics.setdefault("latent_space/mean", []).append(manifold_z.mean().detach().item())
        self.metrics.setdefault("latent_space/std", []).append(manifold_z.std().detach().item())
        self.metrics.setdefault("latent_space/norm_mean", []).append(manifold_z.norm(p=2, dim=-1).mean().detach().item())

        for key, value in grad_norm(self, norm_type=2).items():
            if key == "grad_2.0_norm_total":
                key = "grad_2.0_norm/total"
            self.metrics.setdefault(f"{key}", []).append(value)

        self.log(
            "train_loss",
            total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=False,
            sync_dist=False,
        )

        return total_loss
