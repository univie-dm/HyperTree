import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from ..hyperbolic_math.src.manifolds import PoincareBall, Euclidean
from ..hyperbolic_math.src.optim.riemann_adam import RiemannianAdam
from ..hyperbolic_math.src.optim.riemann_sgd import RiemannianSGD
from ..eval import compute_dendrogram_purity

class HyperbolicAE(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,  # Explicit input dimension
        hidden_dims: List[int] = [512, 256, 128],
        latent_dim: int = 2,  
        learning_rate: float = 1e-3,
        triplet_weight: float = 0.1,
        magnitude_curvature: float = 1.0,
        is_image: bool = False  # Flag to indicate if input is image data
    ):
        super().__init__()
        self.save_hyperparameters()
        self.input_dim = input_dim
        self.is_image = is_image
        self.learning_rate = learning_rate
        self.triplet_weight = triplet_weight
        
        # Initialize Poincare ball manifold
        self.manifold = PoincareBall()
        self.c = torch.tensor(magnitude_curvature)

        self.current_epoch_labels = []
        self.current_epoch_embeddings = []
        
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
        
        # Final encoder layer to map to tangent space
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

    def hyperbolic_distance_matrix(self, z: torch.Tensor) -> torch.Tensor:
        """Compute pairwise hyperbolic distances between embeddings."""
        n = z.size(0)
        dists = torch.zeros((n, n), device=z.device)
        for i in range(n):
            for j in range(i+1, n):
                dist = self.manifold.dist(z[i:i+1], z[j:j+1], self.c)
                dists[i,j] = dists[j,i] = dist.item()
        return dists

    def triplet_loss(self, z: torch.Tensor, similarities: torch.Tensor) -> torch.Tensor:
        """Compute triplet loss based on hyperbolic distances and input similarities."""
        hyperbolic_dists = self.hyperbolic_distance_matrix(z)
        n = z.size(0)
        loss = 0
        num_triplets = 0
        
        # Sample triplets based on similarities
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    w_ij = similarities[i,j]
                    w_ik = similarities[i,k]
                    w_jk = similarities[j,k]
                    
                    # Get pair with highest similarity
                    max_sim = max(w_ij, w_ik, w_jk)
                    if max_sim == w_ij:
                        anchor, pos, neg = i, j, k
                    elif max_sim == w_ik:
                        anchor, pos, neg = i, k, j
                    else:
                        anchor, pos, neg = j, k, i
                    
                    # Compute triplet loss with hyperbolic distances
                    d_pos = hyperbolic_dists[anchor, pos]
                    d_neg = hyperbolic_dists[anchor, neg]
                    loss += F.relu(d_pos - d_neg + 0.1)  # margin = 0.1
                    num_triplets += 1
        
        return loss / num_triplets if num_triplets > 0 else 0

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Map inputs to the Poincare ball through tangent space."""
        # Ensure input is properly flattened
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        
        # Verify input dimension
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {x.shape[1]}")
            
        # Encode to hidden representation
        h = self.encoder(x)
        
        # Map to tangent space
        tangent_vecs = self.to_tangent(h)
        
        # Map to Poincare ball using exponential map
        z = self.manifold.expmap_0(tangent_vecs, self.c)
        
        return z, tangent_vecs

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Map from Poincare ball back to input space."""
        # Map from Poincare ball to tangent space
        tangent_vecs = self.manifold.logmap_0(z, self.c)
        
        # Decode
        x_recon = self.decoder(tangent_vecs)
        
        # If input was image, reshape output
        if self.is_image:
            x_recon = x_recon.view(-1, 1, 28, 28)
        
        return x_recon

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encode
        z, tangent_vecs = self.encode(x)
        
        # Decode
        x_recon = self.decode(z)
        
        return x_recon, z, tangent_vecs

    def compute_distance_matrix(self, z: torch.Tensor) -> torch.Tensor:
        """Compute pairwise hyperbolic distances between inputs after encoding."""
        with torch.no_grad():
            n = z.size(0)
            
            # Create indices for all pairs
            row_idx, col_idx = torch.triu_indices(n, n, offset=1)
            
            # Get unique pairs of points
            points1 = z[row_idx]
            points2 = z[col_idx]
            
            # Compute distances for all pairs at once
            distances = self.manifold.dist(points1, points2, self.c)
            
            # Convert to similarities
            similarities = torch.exp(-distances)
            
            # Create the full similarity matrix
            similarity_matrix = torch.zeros((n, n), device=z.device)
            similarity_matrix[row_idx, col_idx] = similarities
            similarity_matrix[col_idx, row_idx] = similarities  # Mirror the matrix
            
            # Set diagonal to 1.0 (self-similarity)
            similarity_matrix.fill_diagonal_(1.0)
            
            return similarity_matrix

    def training_step(self, batch, batch_idx):
        # Unpack batch - ignoring x_aug for now
        x_aug, x, y = batch
        

        # Forward pass
        x_recon, z, _ = self(x)
        similarities = self.compute_distance_matrix(z)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x)
        
        # Triplet loss for hierarchical structure
        trip_loss = self.triplet_loss(z, similarities)
        
        # Total loss
        loss = recon_loss + self.triplet_weight * trip_loss
        
        # Logging
        self.log('train/recon_loss', recon_loss)
        self.log('train/triplet_loss', 0)
        self.log('train/total_loss', loss)

        # Store the embeddings and labels
        self.current_epoch_labels.append(y)
        self.current_epoch_embeddings.append(z.detach())
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )
        return optimizer
    
    def on_train_epoch_end(self):
        # Compute dendrogram purity every 2 epochs
        if (self.current_epoch + 1) % 2 == 0 and self.current_epoch_labels:
            with torch.no_grad():
                # Concatenate all labels and embeddings
                all_labels = torch.cat(self.current_epoch_labels, dim=0)
                all_embeddings = torch.cat(self.current_epoch_embeddings, dim=0)
                
                # Compute full similarity matrix for all embeddings at once
                similarities = self.compute_distance_matrix(all_embeddings)
                
                # Compute dendrogram purity
                dp_score = compute_dendrogram_purity(
                    similarities,
                    all_labels,
                    method='complete'
                )
                
                # Log the metric
                self.log('metrics/dendrogram_purity', dp_score)
        
        # Clear stored data
        self.current_epoch_labels = []
        self.current_epoch_embeddings = []