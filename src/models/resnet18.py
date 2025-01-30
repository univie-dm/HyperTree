import math
import random
import time
import numpy as np
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.models import resnet18
from typing import List, Tuple
from tqdm import tqdm, trange
import wandb
from matplotlib import pyplot as plt

import warnings

from ..hyperbolic_math.src.utils.math_utils import arcosh
warnings.filterwarnings("ignore", ".*'n_iter' was renamed to 'max_iter'.*")


def resize_conv3x3(in_planes, out_planes, scale=1):
    """upsample + 3x3 convolution with padding to avoid checkerboard artifact."""
    if scale == 1:
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
    return nn.Sequential(
        nn.Upsample(scale_factor=scale, mode='nearest'),
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
    )

def resize_conv1x1(in_planes, out_planes, scale=1):
    """upsample + 1x1 convolution to avoid checkerboard artifact."""
    if scale == 1:
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)
    return nn.Sequential(
        nn.Upsample(scale_factor=scale, mode='nearest'),
        nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)
    )

class DecoderBlock(nn.Module):
    """ResNet block with resize convolutions for decoder."""
    
    expansion = 1
    
    def __init__(self, inplanes, planes, scale=1, upsample=None):
        super().__init__()
        self.conv1 = resize_conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = resize_conv3x3(inplanes, planes, scale)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.upsample is not None:
            identity = self.upsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out

from ..hyperbolic_math.src.manifolds import PoincareBall
from ..eval import compute_dendrogram_purity, plot_embeddings

class HyperbolicResNet(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 2,
        learning_rate: float = 1e-3,
        hyphc_weight: float = 0.1,
        magnitude_curvature: float = 1.0,
        temperature: float = 0.05,
        is_image: bool = True,
        log_interval: int = 5,
        pretrain_epochs: int = 100,
        wandb_enabled: bool = False,
        plot_label_type: str = 'class',
        triplet_sampled_n: int = 1e6,
        loss_batch_size: int = 1000,
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
        self.triplet_sampled_n = find_n_for_combinations(triplet_sampled_n, 3)
        self.loss_batch_size = int(loss_batch_size)
        self.metrics = {}
        
        # Initialize Poincare ball manifold
        self.manifold = PoincareBall()
        self.c = torch.tensor(magnitude_curvature)
        
        # Modified ResNet18 as encoder
        self.encoder = resnet18(weights=None)
        # Replace final FC layer with our latent dimension
        self.encoder.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        
        # Initialize decoder layers
        self.expansion = 1
        self.inplanes = 512 * self.expansion
        
        # Project latent vector to starting feature map
        self.linear = nn.Linear(latent_dim, self.inplanes * 4 * 4)
        
        # Decoder ResNet blocks with upsampling
        self.layer1 = self._make_decoder_layer(256, 2, scale=2)  # 8x8
        self.layer2 = self._make_decoder_layer(128, 2, scale=2)  # 16x16
        self.layer3 = self._make_decoder_layer(64, 2, scale=2)   # 32x32
        
        # Final convolution to get to 3 channels
        self.conv1 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.tanh = nn.Tanh()
        
    def _make_decoder_layer(self, planes, blocks, scale=1):
        upsample = None
        if scale != 1 or self.inplanes != planes:
            upsample = nn.Sequential(
                resize_conv1x1(self.inplanes, planes, scale),
                nn.BatchNorm2d(planes)
            )
            
        layers = []
        layers.append(DecoderBlock(self.inplanes, planes, scale, upsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(DecoderBlock(self.inplanes, planes))
            
        return nn.Sequential(*layers)

    def compute_distance_matrix(self, z: torch.Tensor, batch_size: int = 1024) -> torch.Tensor:
            """
            Compute pairwise hyperbolic distances between inputs after encoding in batches.
            Returns raw distances since for HypHC:
            - Higher distance means less similar points
            - Less similar points should merge higher in tree (smaller LCA depth)
            """
            n = z.size(0)
            manifold_min_enorm = self.manifold.min_enorm
            with torch.no_grad():
                new_dist = compute_strided_pairwise_distances(z, n, manifold_min_enorm, self.c, batch_size)
            return new_dist

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tangent_vecs = self.encoder(x)
        z = self.manifold.expmap_0(tangent_vecs, self.c)
        return z, tangent_vecs

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        tangent_vecs = self.manifold.logmap_0(z, self.c)
        # Project to initial feature map
        x = self.linear(tangent_vecs)
        x = x.view(-1, 512 * self.expansion, 4, 4)
        
        # Pass through decoder layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Final convolution and activation
        x = self.conv1(x)
        x = self.tanh(x)
        
        return x

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

    def compute_hyphc_loss(self, embeddings: torch.Tensor, distances: torch.Tensor, sampled_n: int = 1e6, batch_size: int = 1000) -> torch.Tensor:
        n = embeddings.size(0)
        device = embeddings.device
        distances = torch.reciprocal(1 + distances)

        '''# Generate all possible triplets efficiently
        i_idx = torch.arange(n, device=device).repeat(n * n)
        j_idx = torch.arange(n, device=device).repeat_interleave(n).repeat(n)
        k_idx = torch.arange(n, device=device).repeat_interleave(n * n)
        
        # Filter valid triplets
        mask = (i_idx != j_idx) & (j_idx != k_idx) & (i_idx != k_idx)
        i_idx, j_idx, k_idx = i_idx[mask], j_idx[mask], k_idx[mask]

        # Sample if requested
        if sampled_n > 0:
            num_triplets = len(i_idx)
            if sampled_n < num_triplets:
                perm = torch.randperm(num_triplets, device=device)[:int(sampled_n)]
                i_idx, j_idx, k_idx = i_idx[perm], j_idx[perm], k_idx[perm]
        '''
        sampled_idx = torch.randint(0, n, (min(sampled_n, n),))
        sampled_triplets = torch.combinations(sampled_idx, r=3).to(device)
        # Process in batches to manage memory
        num_batches = (len(sampled_triplets) + batch_size - 1) // batch_size
        triplet_terms = []
        lcas = torch.zeros((n, n), device=device)

        for b in range(num_batches):
            start_idx = b * batch_size
            end_idx = min((b + 1) * batch_size, len(sampled_triplets))
            
            batch_i, batch_j, batch_k = sampled_triplets[start_idx:end_idx].unbind(1)

            # Get weights for the batch
            w_ij = distances[batch_i, batch_j]
            w_ik = distances[batch_i, batch_k]
            w_jk = distances[batch_j, batch_k]

            # Compute LCAs for new pairs
            for i, j in zip(batch_i, batch_j):
                if lcas[i, j] == 0:
                    lcas[i, j] = self.compute_lca_depth(embeddings[i], embeddings[j])
                    lcas[j, i] = lcas[i, j]
            for i, k in zip(batch_i, batch_k):
                if lcas[i, k] == 0:
                    lcas[i, k] = self.compute_lca_depth(embeddings[i], embeddings[k])
                    lcas[k, i] = lcas[i, k]
            for j, k in zip(batch_j, batch_k):
                if lcas[j, k] == 0:
                    lcas[j, k] = self.compute_lca_depth(embeddings[j], embeddings[k])
                    lcas[k, j] = lcas[j, k]

            # Compute softmax and loss terms
            sim_weights = torch.stack([w_ij, w_ik, w_jk], dim=1)
            max_sim = torch.max(sim_weights, dim=1)[0].unsqueeze(1)
            exp_sims = torch.exp((sim_weights - max_sim) / self.temperature)
            softmax = exp_sims / exp_sims.sum(dim=1, keepdim=True)
            
            weights = torch.stack([w_ij, w_ik, w_jk], dim=1)
            dot_product = torch.sum(weights * softmax, dim=1)
            triplet_term = (w_ij + w_ik + w_jk) - dot_product
            triplet_terms.append(triplet_term)

        triplet_loss = torch.cat(triplet_terms).sum()

        # Compute distance sum efficiently

        pairs_i, pairs_j = torch.combinations(sampled_idx, r=2).unbind(1)
        sum_dists = distances[pairs_i, pairs_j].sum()

        total_loss = triplet_loss + sum_dists * 2
        return total_loss, triplet_loss, sum_dists

    def training_step(self, batch, batch_idx):
        x_aug, x, y = batch

        x_recon, z, _ = self(x)
        

        recon_loss = F.mse_loss(x_recon, x)
            
        if 'train/recon_loss' not in self.metrics:
            self.metrics['train/recon_loss'] = []
            
        self.metrics['train/recon_loss'].append(recon_loss.detach())
        
        hyphc_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        if self.current_epoch >= self.pretrain_epochs:
            distances = self.compute_distance_matrix(z, batch_size=int(1e3))
            hyphc_loss, triplet_loss, sum_dists = self.compute_hyphc_loss(z, distances, sampled_n=self.triplet_sampled_n, batch_size=self.loss_batch_size)
            if 'train/hyphc_loss' not in self.metrics:
                self.metrics['train/hyphc_loss'] = []
            self.metrics['train/hyphc_loss'].append(hyphc_loss.detach())
            if 'train/sum_dists' not in self.metrics:
                self.metrics['train/sum_dists'] = []
                self.metrics['train/triplet_loss'] = []
            self.metrics['train/sum_dists'].append(sum_dists.detach())
            self.metrics['train/triplet_loss'].append(triplet_loss.detach())
        
        loss = recon_loss + self.hyphc_weight * hyphc_loss
        if 'train/total_loss' not in self.metrics:
            self.metrics['train/total_loss'] = []

        self.metrics['train/total_loss'].append(loss.detach())
        
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
                        self.metrics["eval/Hyperbolic Embedding"] = wandb.Image(fig, caption="Hyperbolic Embeddings Epoch {}".format(self.current_epoch))
                        plt.close()
                        
                        tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
                        embeddings = tsne.fit_transform(all_embeddings_np)
                        fig = plot_embeddings(embeddings, all_labels_np, class_names, label_type=self.plot_label_type)

                        # Log and cleanup
                        self.metrics["eval/T-SNE Embedding"] = wandb.Image(fig, caption="T-SNE Embeddings Epoch {}".format(self.current_epoch))
                        plt.close()

                    # Free memory
                    del embeddings, labels
                    torch.cuda.empty_cache()
                    similarities = self.compute_distance_matrix(
                        all_embeddings.to(self.device), 
                        batch_size=int(1e3)  # Smaller batch size for distance computation
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
                    self.metrics['eval/dendrogram_purity'] = dp_score

                    hyhc_loss, triplet_loss, sum_dists = self.compute_hyphc_loss(all_embeddings, similarities, sampled_n=self.triplet_sampled_n, batch_size=self.loss_batch_size)
                    self.metrics['eval/hyphc_loss'] = hyhc_loss.item()
                    self.metrics['eval/triplet_loss'] = triplet_loss.item()
                    self.metrics['eval/sum_dists'] = sum_dists.item()

                except RuntimeError as e:
                    print(f"Error during computation: {str(e)}")
                    return

        for key, value in self.metrics.items():
            if isinstance(value, list):
                self.metrics[key] = torch.mean(torch.stack(value)).item()
        self.metrics['epoch'] = self.current_epoch
        if isinstance(self.logger, pl.loggers.WandbLogger):
            self.logger.experiment.log(self.metrics, step=self.current_epoch+1)
        self.metrics = {}

@torch.jit.script
def compute_strided_pairwise_distances(z: torch.Tensor, n: int, manifold_min_enorm: float, c: torch.Tensor, batch_size: int) -> torch.Tensor:
    
    z_sqnorm = z.pow(2).sum(dim=-1)
    distances = torch.zeros((n, n), device=z.device)
    
    for i in range(0, n, batch_size):
        end_i = min(i + batch_size, n)
        for j in range(i + 1, n, batch_size):
            end_j = min(j + batch_size, n)
            
            # Get blocks of points
            x = z[i:end_i]  # Shape: (batch_i, d)
            y = z[j:end_j]  # Shape: (batch_j, d)
            
            # Get corresponding squared norms and reshape for broadcasting
            x_sqnorm = z_sqnorm[i:end_i].view(-1, 1)      # Shape: (batch_i, 1)
            y_sqnorm = z_sqnorm[j:end_j].view(1, -1)      # Shape: (1, batch_j)
            
            # Compute block distances
            xy_diff = x.unsqueeze(1) - y.unsqueeze(0)     # Shape: (batch_i, batch_j, d)
            xy_diff_sqnorm = xy_diff.pow(2).sum(dim=-1)   # Shape: (batch_i, batch_j)
            
            # Compute denominator terms ensuring correct broadcasting
            denom1 = (1 - c * x_sqnorm)                   # Shape: (batch_i, 1)
            denom2 = (1 - c * y_sqnorm)                   # Shape: (1, batch_j)
            denominator = denom1 * denom2                  # Shape: (batch_i, batch_j)
            
            # Compute final result
            res = 1 + 2 * c * xy_diff_sqnorm / denominator
            condition = res < 1 + manifold_min_enorm
            block_distances = torch.where(condition,
                                        torch.zeros_like(res),
                                        arcosh(res) / c.sqrt())
            
            # Fill distance matrix
            distances[i:end_i, j:end_j] = block_distances
            distances[j:end_j, i:end_i] = block_distances.t()
    
    return distances


def find_n_for_combinations(target_combinations, r):
    n = r
    while math.comb(n, r) < target_combinations:
        n += 1
    return n