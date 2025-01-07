import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import List, Tuple, Optional

from ..hyperbolic_math.src.manifolds import PoincareBall, ManifoldParameter
from ..hyperbolic_math.src.optim.riemann_adam import RiemannianAdam


class HyperbolicAE(pl.LightningModule):
    def __init__(
        self,
        hidden_dims: List[int] = [512, 256, 128],
        latent_dim: int = 2,
        learning_rate: float = 1e-3,
        triplet_weight: float = 0.1,
        curvature: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # FMNIST specific input dimension (28x28 = 784)
        self.input_dim = 784
        
        # Initialize Poincare ball manifold
        self.manifold = PoincareBall()
        self.c = torch.tensor(curvature)
        
        # Encoder
        encoder_layers = []
        prev_dim = self.input_dim
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
        decoder_layers.extend([
            nn.Linear(hidden_dims[0], self.input_dim),
            nn.Sigmoid()  # For FMNIST pixel values
        ])
        self.decoder = nn.Sequential(*decoder_layers)

    def hyperbolic_distance_matrix(self, z: torch.Tensor) -> torch.Tensor:
        """Compute pairwise hyperbolic distances between embeddings."""
        n = z.size(0)
        # Create matrices of all pairs
        z_i = z.unsqueeze(1).expand(-1, n, -1)  # Shape: (n, n, dim)
        z_j = z.unsqueeze(0).expand(n, -1, -1)  # Shape: (n, n, dim)
        
        # Compute all pairwise distances at once
        dists = self.manifold.dist(z_i, z_j, self.c)
        return dists

    def triplet_loss(self, z: torch.Tensor, similarities: torch.Tensor) -> torch.Tensor:
        """Compute triplet loss based on hyperbolic distances and input similarities."""
        hyperbolic_dists = self.hyperbolic_distance_matrix(z)
        n = z.size(0)
        loss = 0
        num_triplets = 0
        
        # Sample triplets based on class labels
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    w_ij = similarities[i,j]
                    w_ik = similarities[i,k]
                    w_jk = similarities[j,k]
                    
                    # Get the pair with highest similarity
                    max_sim = max(w_ij, w_ik, w_jk)
                    if max_sim == w_ij:
                        anchor, pos, neg = i, j, k
                    elif max_sim == w_ik:
                        anchor, pos, neg = i, k, j
                    else:
                        anchor, pos, neg = j, k, i
                    
                    # Compute triplet loss
                    d_pos = hyperbolic_dists[anchor, pos]
                    d_neg = hyperbolic_dists[anchor, neg]
                    loss += F.relu(d_pos - d_neg + 0.1)  # margin = 0.1
                    num_triplets += 1
        
        return loss / num_triplets if num_triplets > 0 else 0

    def compute_similarities(self, x: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity matrix between flattened inputs."""
        x_flat = x.view(x.size(0), -1)
        x_norm = F.normalize(x_flat, p=2, dim=1)
        return torch.mm(x_norm, x_norm.t())

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Map FMNIST images to the Poincare ball through tangent space."""
        # Flatten and encode
        x = x.view(x.size(0), -1)
        h = self.encoder(x)
        
        # Map to tangent space
        tangent_vecs = self.to_tangent(h)
        
        # Map to Poincare ball
        z = self.manifold.expmap_0(tangent_vecs, self.c)
        
        return z, tangent_vecs

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Map from Poincare ball back to FMNIST image space."""
        # Map from Poincare ball to tangent space
        tangent_vecs = self.manifold.logmap_0(z, self.c)
        
        # Decode and reshape to image
        x_recon = self.decoder(tangent_vecs)
        return x_recon.view(-1, 1, 28, 28)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encode
        z, tangent_vecs = self.encode(x)
        
        # Decode
        x_recon = self.decode(z)
        
        return x_recon, z, tangent_vecs

    def training_step(self, batch, batch_idx):
        # Unpack batch - ignoring x_aug for now
        x_aug, x, y = batch  # x_aug ignored for now
        
        # Compute pairwise similarities
        similarities = self.compute_similarities(x)
        
        # Forward pass
        x_recon, z, _ = self(x)
        
        # Reconstruction loss (binary cross entropy for FMNIST)
        recon_loss = F.binary_cross_entropy(x_recon, x)
        
        # Triplet loss for hierarchical structure
        trip_loss = self.triplet_loss(z, similarities)
        
        # Total loss
        loss = recon_loss + self.hparams.triplet_weight * trip_loss
        
        # Logging
        self.log('train/recon_loss', recon_loss)
        self.log('train/triplet_loss', trip_loss)
        self.log('train/total_loss', loss)
        
        return loss

    def configure_optimizers(self):
        optimizer = RiemannianAdam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            c=self.c,
            expmap_update=True,
            stabilize=10
        )
        return optimizer
