import random
from matplotlib import pyplot as plt
import numpy as np
import pytorch_lightning as pl
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from tqdm import tqdm, trange
from lightning.pytorch.loggers import WandbLogger
import wandb

from ..hyperbolic_math.src.manifolds import PoincareBall, Euclidean
from ..hyperbolic_math.src.optim.riemann_adam import RiemannianAdam
from ..hyperbolic_math.src.optim.riemann_sgd import RiemannianSGD
from ..eval import compute_dendrogram_purity, plot_embeddings

class HyperbolicAE(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 256, 128],
        latent_dim: int = 2,
        learning_rate: float = 1e-3,
        hyphc_weight: float = 0.1,
        magnitude_curvature: float = 1.0,
        temperature: float = 0.05,
        is_image: bool = False,
        log_interval: int = 5,
        pretrain_epochs: int = 100,
        wandb_enabled: bool = False,
        plot_label_type: str = 'class',
    ):
        super().__init__()
        self.save_hyperparameters()
        self.input_dim = input_dim
        self.is_image = is_image
        self.learning_rate = learning_rate
        self.hyphc_weight = hyphc_weight
        self.temperature = temperature
        self.log_interval = log_interval
        self.pretrain_epochs = pretrain_epochs
        self.wandb_enabled = wandb_enabled
        self.plot_label_type = plot_label_type
        
        # Initialize Poincare ball manifold
        self.manifold = PoincareBall()
        self.c = torch.tensor(magnitude_curvature)
        
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

    def compute_distance_matrix(self, z: torch.Tensor, batch_size: int = 1024) -> torch.Tensor:
        """
        Compute pairwise hyperbolic distances between inputs after encoding in batches.
        Returns raw distances since for HypHC:
        - Higher distance means less similar points
        - Less similar points should merge higher in tree (smaller LCA depth)
        """
        n = z.size(0)
        distance_matrix = torch.zeros((n, n), device=z.device)
        
        with torch.no_grad():
            for i in range(0, n, batch_size):
                end_i = min(i + batch_size, n)
                for j in range(0, n, batch_size):
                    end_j = min(j + batch_size, n)
                    
                    distances = self.manifold.dist(
                        z[i:end_i].unsqueeze(1), 
                        z[j:end_j].unsqueeze(0),
                        self.c
                    )
                    
                    distances = distances.squeeze()
                    distance_matrix[i:end_i, j:end_j] = distances
                    
                    if i != j:
                        distance_matrix[j:end_j, i:end_i] = distance_matrix[i:end_i, j:end_j].t()
                
            # Set diagonal to 0 (self-distance)
            distance_matrix.fill_diagonal_(0.0)
            return distance_matrix

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {x.shape[1]}")
            
        h = self.encoder(x)
        tangent_vecs = self.to_tangent(h)
        z = self.manifold.expmap_0(tangent_vecs, self.c)
        return z, tangent_vecs

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        tangent_vecs = self.manifold.logmap_0(z, self.c)
        x_recon = self.decoder(tangent_vecs)
        return x_recon

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, tangent_vecs = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z, tangent_vecs

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def compute_lca_depth(self, x: torch.Tensor, y: torch.Tensor, debug=False) -> torch.Tensor:
        """Compute depth with debugging."""
        eps = 1e-6
        x_norm = torch.norm(x, dim=-1).clamp(min=eps)
        y_norm = torch.norm(y, dim=-1).clamp(min=eps)
        
        if debug:
            print(f"x_norm: {x_norm.item():.4f}, y_norm: {y_norm.item():.4f}")
        
        dot_product = torch.sum(x * y, dim=-1)
        cos_theta = torch.clamp(dot_product/(x_norm * y_norm), min=-1.0 + eps, max=1.0 - eps)
        theta = torch.acos(cos_theta)
        
        if debug:
            print(f"cos_theta: {cos_theta.item():.4f}, theta: {theta.item():.4f}")
        
        sin_theta = torch.sin(theta).clamp(min=eps)
        
        numerator = x_norm * (y_norm * y_norm + 1)
        denominator = y_norm * (x_norm * x_norm + 1)
        
        if debug:
            print(f"numerator: {numerator.item():.4f}, denominator: {denominator.item():.4f}")
        
        ones = torch.ones_like(theta)
        atan2_arg = ones/sin_theta * (numerator/denominator - cos_theta)
        alpha = torch.atan2(atan2_arg.clamp(min=-1e6, max=1e6), ones)
        
        if debug:
            print(f"atan2_arg: {atan2_arg.item():.4f}, alpha: {alpha.item():.4f}")
        
        cos_alpha = torch.cos(alpha).clamp(min=eps)
        R = torch.sqrt(((x_norm * x_norm + 1)/(2 * x_norm * cos_alpha))**2 - ones)
        
        if debug:
            print(f"R: {R.item():.4f}")
            
        sqrt_term = torch.sqrt(R * R + ones)
        depth = 2 * torch.atanh((sqrt_term - R).clamp(min=-1.0 + eps, max=1.0 - eps))
        
        if debug:
            print(f"Final depth: {depth.item():.4f}")
            
        return depth

    def compute_hyphc_loss(self, embeddings: torch.Tensor, distances: torch.Tensor):
        """Compute HypHC loss with debugging."""
        n = embeddings.size(0)
        sampled_n = n^2
        device = embeddings.device
        eps = 1e-6
        distances += 1
        distances /= 1
        distances = distances.clamp(min=eps, max=1e6)
        lcas = torch.zeros((n, n), device=device)
        triplet_terms = []
        all_triplets = [(i, j, k) for i in range(n) for j in range(n) for k in range(n) if i != j and j != k and i != k]
        sampled_triplets = random.sample(all_triplets, min(sampled_n, n))

        for i,j,k in sampled_triplets:

            w_ij = distances[i, j]
            w_ik = distances[i, k]
            w_jk = distances[j, k]
            
            # Debug every few triplets
            debug = False
            if debug:
                print(f"Distances - w_ij: {w_ij.item():.4f}, w_ik: {w_ik.item():.4f}, w_jk: {w_jk.item():.4f}")
            
            try:
                if lcas[i, j] == 0 or lcas[j, i] == 0:
                    lcas[i, j] = self.compute_lca_depth(embeddings[i], embeddings[j], debug=debug)
                    lcas[j, i] = lcas[i, j]
                if lcas[i, k] == 0 or lcas[k, i] == 0:
                    lcas[i, k] = self.compute_lca_depth(embeddings[i], embeddings[k], debug=debug)
                    lcas[k, i] = lcas[i, k]
                if lcas[j, k] == 0 or lcas[k, j] == 0:
                    lcas[j, k] = self.compute_lca_depth(embeddings[j], embeddings[k], debug=debug)
                    lcas[k, j] = lcas[j, k]

                d_ij = lcas[i, j]
                d_ik = lcas[i, k]
                d_jk = lcas[j, k]
                
                if debug:
                    print(f"LCA depths - d_ij: {d_ij.item():.4f}, d_ik: {d_ik.item():.4f}, d_jk: {d_jk.item():.4f}")
                
                sim_weights = torch.stack([w_ij, w_ik, w_jk])
                weights = torch.stack([w_ij, w_ik, w_jk])
                
                max_sim = torch.max(sim_weights)
                exp_sims = torch.exp((sim_weights - max_sim) / self.temperature)
                softmax_denom = exp_sims.sum().clamp(min=eps)
                softmax = exp_sims / softmax_denom
                
                if debug:
                    print(f"Softmax weights: {softmax.cpu().numpy()}")
                
                dot_product = torch.dot(weights, softmax)
                triplet_term = (w_ij + w_ik + w_jk) - dot_product
                
                if debug:
                    print(f"Triplet term: {triplet_term.item():.4f}")
                
                triplet_terms.append(triplet_term)
                    
            except RuntimeError as e:
                if debug:
                    print(f"Error in triplet {i}, {j}, {k}: {str(e)}")
                continue
        
        if len(triplet_terms) > 0:
            #triplet_loss = torch.stack(triplet_terms).mean()
            triplet_loss = torch.stack(triplet_terms).sum()
            constant_term = 2 * distances.sum() # / distances.max() not in paper
            self.log('train/constant_term', constant_term.detach())
            self.log('train/triplet_loss', triplet_loss.detach())
            total_loss = triplet_loss + constant_term
            if debug:
                print(f"\nFinal stats:")
                print(f"Triplet loss mean: {triplet_loss.item():.4f}")
                print(f"Constant term: {constant_term.item():.4f}")
                print(f"Total loss: {total_loss.item():.4f}")
            
            return total_loss
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)

    def training_step(self, batch, batch_idx):
        x_aug, x, y = batch

        if len(x.shape) > 2:
            x_flat = x.view(x.size(0), -1)
        else:
            x_flat = x

        x_recon, z, _ = self(x_flat)
        
        recon_loss = F.mse_loss(x_recon.view(x.size(0), -1), x_flat)
        self.log('train/recon_loss', recon_loss.detach())
        
        hyphc_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        if self.current_epoch >= self.pretrain_epochs:
            distances = self.compute_distance_matrix(z)
            hyphc_loss = self.compute_hyphc_loss(z, distances)
            self.log('train/hyphc_loss', hyphc_loss.detach())
        
        loss = recon_loss + self.hyphc_weight * hyphc_loss
        self.log('train/total_loss', loss.detach(), on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_train_epoch_end(self):
        # Compute dendrogram purity every log_interval epochs
        if (self.current_epoch + 1) % self.log_interval == 0:
            with torch.no_grad():
                test_loader = self.trainer.datamodule.test_dataloader()
                
                embeddings = []
                labels = []
                
                for batch_idx, (x_aug, x, y) in enumerate(test_loader):                        
                    try:
                        # Clear cache periodically
                        if batch_idx % 5 == 0:
                            torch.cuda.empty_cache()
                        
                        # Ensure input is properly flattened
                        if len(x.shape) > 2:
                            x = x.view(x.size(0), -1)
                        x = x.to(self.device)
                        
                        _, z, _ = self(x)
                        embeddings.append(z.cpu())  # Move to CPU immediately
                        labels.append(y.cpu())
                        
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print(f"OOM at batch {batch_idx}, continuing with collected samples")
                            torch.cuda.empty_cache()
                            break
                        raise e
                
                if not embeddings:
                    print("No embeddings collected, skipping dendrogram purity computation")
                    return
                    
                # Concatenate all embeddings and labels
                try:
                    all_embeddings = torch.cat(embeddings, dim=0)
                    all_labels = torch.cat(labels, dim=0)
                    
                    # Convert to numpy
                    all_embeddings_np = all_embeddings.cpu().numpy()
                    all_labels_np = all_labels.cpu().numpy()
                    if self.wandb_enabled:
                        class_names = self.trainer.datamodule.get_class_names()
                        fig = plot_embeddings(all_embeddings_np, all_labels_np, class_names, label_type=self.plot_label_type)

                        # Log and cleanup
                        self.logger.experiment.log({"Hyperbolic Embedding":[wandb.Image(fig, caption="Hyperbolic Embedding")]})
                        plt.close()
                    
                    # Free memory
                    del embeddings, labels
                    torch.cuda.empty_cache()
                    similarities = self.compute_distance_matrix(
                        all_embeddings.to(self.device), 
                        batch_size=256  # Smaller batch size for distance computation
                    )
                    
                    # Move to CPU for dendrogram purity computation
                    similarities = similarities.cpu()
                    all_labels = all_labels.cpu()
                    
                    # Free GPU memory
                    torch.cuda.empty_cache()
                    
                    # Compute dendrogram purity
                    dp_score = compute_dendrogram_purity(
                        similarities,
                        all_labels,
                        method='complete'
                    )
                    
                    # Log the metric
                    self.log('metrics/dendrogram_purity', dp_score, on_epoch=True, prog_bar=True)
                    
                except RuntimeError as e:
                    print(f"Error during computation: {str(e)}")
                    return
                    