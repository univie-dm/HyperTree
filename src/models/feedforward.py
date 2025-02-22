from src.models.abstract import AbstractModel
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from tqdm import tqdm, trange

from ..hyperbolic_math.src.manifolds import PoincareBall, Manifold

class HyperbolicAE(AbstractModel):
    def __init__(
        self,
        input_dim: int,
        manifold: Manifold,
        hidden_dims: List[int] = [512, 256, 128],
        latent_dim: int = 2,
        learning_rate: float = 1e-3,
        log_interval: int = 5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.log_interval = log_interval
        self.latent_dim = latent_dim
        self.metrics = {}
        
        # Initialize Poincare ball manifold
        
        self.manifold = manifold
        
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
        h = self.encoder(x)
        tangent_vecs = self.to_tangent(h)
        z = self.manifold.expmap_0(tangent_vecs)
        return z, tangent_vecs

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        tangent_vecs = self.manifold.logmap_0(z)
        x_recon = self.decoder(tangent_vecs)
        return x_recon

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, tangent_vecs = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z, tangent_vecs

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    

    def training_step(self, batch, batch_idx):
        x_aug, x, y = batch
        x = x.view(x.size(0), -1)
        x_aug = x_aug.view(x_aug.size(0), -1)
        x_recon, z, _ = self(x)
        x_aug_recon, z_aug, _ = self(x_aug)
        

        recon_loss = F.mse_loss(x_recon, x)
        recon_loss_aug = F.mse_loss(x_aug_recon, x_aug)
        self.metrics.setdefault("pretrain/recon_loss", []).append(recon_loss)
        self.metrics.setdefault("pretrain/recon_loss_aug", []).append(recon_loss_aug)
        
        loss = (recon_loss + recon_loss_aug)/2
        self.metrics.setdefault("pretrain/total_loss", []).append(loss)
        self.metrics.setdefault("pretrain/c", []).append(self.manifold.c.detach())
        
        return loss
    
    def on_train_epoch_end(self):
        if self.current_epoch % self.log_interval == 0:
            with torch.no_grad():
                test_loader = self.trainer.datamodule.test_dataloader()
                
                embeddings = []
                labels = []
                
                for batch_idx, (x_aug, x, y) in enumerate(test_loader):
                    x = x.view(x.size(0), -1)
                    _, z, _ = self(x.to(self.device))
                    embeddings.append(z.cpu())  # Move to CPU immediately
                    labels.append(y.cpu())
                # Concatenate all embeddings and labels

                    all_embeddings = torch.cat(embeddings, dim=0)
                    all_labels = torch.cat(labels, dim=0)
                    
                    # Convert to numpy
                    all_embeddings_np = all_embeddings.cpu().numpy()
                    all_labels_np = all_labels.cpu().numpy()
                                  
                    # Move to CPU for dendrogram purity computation
                    all_labels = all_labels.cpu()
                    
                    agglomerative_clustering = AgglomerativeClustering(n_clusters=10)
                    agglomerative_clustering.fit(all_embeddings_np)
                    nmi = normalized_mutual_info_score(all_labels_np, agglomerative_clustering.labels_)
                    ari = adjusted_rand_score(all_labels_np, agglomerative_clustering.labels_)
                    # Log the metric
                    self.metrics.setdefault("pretrain/nmi", []).append(nmi)                    
                    self.metrics.setdefault("pretrain/ari", []).append(ari)


        self.metrics['pretrain/epoch'] = self.current_epoch
        for key, value in self.metrics.items():
            if isinstance(value, list):
                self.metrics[key] = sum(value) / len(value)
        self.logger.experiment.log(self.metrics, step=self.current_epoch)
        self.metrics = {}
                    