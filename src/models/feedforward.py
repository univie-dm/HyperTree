import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.utilities import grad_norm

from typing import List, Tuple
from src.models.abstract import AbstractModel
from src.hyperbolic_math.src.manifolds import Euclidean
from src.hyperbolic_math.src.nn_layers import *


class HyperbolicAE(AbstractModel):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 256, 128],
        hyperbolic_encoding: str = 'EuclideanLinear+Exp',
        hyperbolic_decoding: str = 'Log+EuclideanLinear',
        manifold_name: str = 'PoincareBall',
        manifold_precision: str = 'float32',
        curvature_magnitude: float = 1.0,
        trainable_curvature: bool = False,
        latent_dim: int = 2,
        learning_rate: float = 1e-3,
        log_interval: int = 5,
        stage: str = 'pretrain',
        global_epoch: int = 0,
        params_dtype: str = 'float32',
    ):
        super().__init__(
            manifold_name=manifold_name,
            manifold_precision=manifold_precision,
            curvature_magnitude=curvature_magnitude,
            trainable_curvature=trainable_curvature,
            latent_dim=latent_dim,
            learning_rate=learning_rate,
            log_interval=log_interval,
            stage=stage,
            global_epoch=global_epoch,
            learning_rate_scheduler=None,
            lr_scheduler_params=None,
            model_type='feedforward',
        )
        self.input_dim = input_dim
        self.params_dtype = params_dtype
        self.save_hyperparameters()

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Add hyperbolic encoding layer(s)
        if isinstance(self.manifold, Euclidean):
            print("Using Euclidean manifold, hyperbolic encoding layers will default to nn.Linear.",
                  flush=True)
            self.hyperbolic_encoding_layer = nn.Linear(hidden_dims[-1], latent_dim)
        elif hyperbolic_encoding == 'EuclideanLinear+Exp':
            self.hyperbolic_encoding_layer = nn.Sequential(
                nn.Linear(hidden_dims[-1], latent_dim),
                Expmap_0(manifold=self.manifold)
            )
        elif hyperbolic_encoding == 'HyperbolicLinear':
            self.hyperbolic_encoding_layer = HyperbolicLinear(
                manifold=self.manifold,
                input_dim=hidden_dims[-1],
                output_dim=latent_dim,
                params_dtype=self.params_dtype,
                input_space='tangent')

        # Add hyperbolic decoding layer(s)
        if isinstance(self.manifold, Euclidean):
            print("Using Euclidean manifold, hyperbolic decoding layers will default to nn.Linear.",
                  flush=True)
            self.hyperbolic_decoding_layer = nn.Sequential(
                nn.Linear(latent_dim, hidden_dims[-1]),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            )
        elif hyperbolic_decoding == 'Log+EuclideanLinear':
            self.hyperbolic_decoding_layer = nn.Sequential(
                Logmap_0(manifold=self.manifold),
                nn.Linear(latent_dim, hidden_dims[-1]),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            )
        elif hyperbolic_decoding == 'HyperbolicLinear+Log':
            self.hyperbolic_decoding_layer = nn.Sequential(
                HyperbolicLinear(
                    manifold=self.manifold,
                    input_dim=latent_dim,
                    output_dim=hidden_dims[-1],
                    params_dtype=self.params_dtype,
                    input_space='manifold'),
                Logmap_0(manifold=self.manifold),
                #nn.BatchNorm1d(hidden_dim),
                #nn.ReLU()
            )
        elif hyperbolic_decoding == 'HyperbolicRegression':
            self.hyperbolic_decoding_layer = nn.Sequential(
                HyperbolicRegressionPoincare(
                    manifold=self.manifold,
                    input_dim=latent_dim,
                    output_dim=hidden_dims[-1],
                    params_dtype=self.params_dtype,
                    input_space='manifold'),
                #nn.BatchNorm1d(hidden_dim),
                #nn.ReLU()
            )
        elif hyperbolic_decoding == 'HyperbolicRegressionPP':
            self.hyperbolic_decoding_layer = nn.Sequential(
                HyperbolicRegressionPoincarePP(
                    manifold=self.manifold,
                    input_dim=latent_dim,
                    output_dim=hidden_dims[-1],
                    params_dtype=self.params_dtype,
                    input_space='manifold'),
                #nn.BatchNorm1d(hidden_dim),
                #nn.ReLU()
            )
        elif hyperbolic_decoding == 'HyperbolicLinearPP':
            self.hyperbolic_decoding_layer = nn.Sequential(
                HyperbolicFullyConnectedPoincarePP(
                    manifold=self.manifold,
                    input_dim=latent_dim,
                    output_dim=hidden_dims[-1],
                    params_dtype=self.params_dtype,
                    input_space='manifold'),
                #nn.BatchNorm1d(hidden_dim),
                #nn.ReLU()
            )

        # Decoder
        decoder_layers = []
        prev_dim = hidden_dims[-1]
        for hidden_dim in reversed(hidden_dims[:-1]):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.view(x.size(0), -1)
        hidden = self.encoder(x)
        manifold_z = self.hyperbolic_encoding_layer(hidden)
        return manifold_z

    def decode(self, manifold_z: torch.Tensor) -> torch.Tensor:
        x_recon = self.hyperbolic_decoding_layer(manifold_z)
        x_recon = self.decoder(x_recon)
        x_recon = x_recon.view(-1, *self.orginal_input_shape)
        return x_recon

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x_aug, x, _ = batch
        self.orginal_input_shape = x.shape[1:]
        x_recon, _, = self(x)
        #x_aug_recon, _, = self(x_aug)

        # Reconstruction loss
        x_recon_loss = F.mse_loss(x_recon, x)
        #x_aug_recon_loss = F.mse_loss(x_aug_recon, x_aug)
        #loss = (x_recon_loss + x_aug_recon_loss) / 2
        loss = x_recon_loss

        for key, value in grad_norm(self, norm_type=2).items():
            if key == "grad_2.0_norm_total":
                key = "grad_2.0_norm/total"
            self.metrics.setdefault(f"{key}", []).append(value)

        # Logging
        self.metrics.setdefault("pretrain/x_recon_loss", []).append(x_recon_loss)
        #self.metrics.setdefault("pretrain/x_aug_recon_loss", []).append(x_aug_recon_loss)
        self.metrics.setdefault("pretrain/total_loss", []).append(loss)
        self.metrics.setdefault("pretrain/c", []).append(self.manifold.c.detach())

        return loss
