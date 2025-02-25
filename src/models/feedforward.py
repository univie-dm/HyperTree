import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple
from src.models.abstract import AbstractModel


class HyperbolicAE(AbstractModel):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 256, 128],
        manifold_name: str = 'PoincareBall',
        curvature_magnitude: float = 1.0,
        trainable_curvature: bool = False,
        latent_dim: int = 2,
        learning_rate: float = 1e-3,
        hyphc_weight: float = 0.1,
        temperature: float = 0.05,
        log_interval: int = 5
    ):
        super().__init__(
            manifold_name,
            curvature_magnitude,
            trainable_curvature,
            latent_dim,
            learning_rate,
            hyphc_weight,
            temperature,
            log_interval
        )
        self.input_dim = input_dim
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
        self.to_tangent = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x)
        tangent_z = self.to_tangent(hidden)
        manifold_z = self.manifold.expmap_0(tangent_z)
        return manifold_z, tangent_z

    def decode(self, manifold_z: torch.Tensor) -> torch.Tensor:
        tangent_z = self.manifold.logmap_0(manifold_z)
        x_recon = self.decoder(tangent_z)
        return x_recon

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x_aug, x, _ = batch
        x = x.view(x.size(0), -1)
        x_aug = x_aug.view(x_aug.size(0), -1)

        x_recon, _, _ = self(x)
        x_aug_recon, _, _ = self(x_aug)

        recon_loss = F.mse_loss(x_recon, x)
        recon_loss_aug = F.mse_loss(x_aug_recon, x_aug)

        self.metrics.setdefault("pretrain/recon_loss", []).append(recon_loss)
        self.metrics.setdefault("pretrain/recon_loss_aug", []).append(recon_loss_aug)

        loss = (recon_loss + recon_loss_aug) / 2
        self.metrics.setdefault("pretrain/total_loss", []).append(loss)
        self.metrics.setdefault("pretrain/c", []).append(self.manifold.c.detach())

        # TODO: Add Hyp loss here?
        return loss
