import math
import random
import time
from src.models.utils import embedddings_plot
import numpy as np
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.models import resnet34
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

class HyperbolicResNet34(pl.LightningModule):
    def __init__(
        self,
        manifold: PoincareBall,
        latent_dim: int = 2,
        learning_rate: float = 1e-3,
        log_interval: int = 5,
    ):
        super().__init__()
        self.manifold = manifold
        self.learning_rate = learning_rate
        self.log_interval = log_interval
        self.metrics = {}
        
        # Changed to ResNet-34
        self.encoder = resnet34(weights=None)
        # Keep the same FC layer modification
        self.encoder.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        
        # Keep the rest of the decoder and other components unchanged
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


    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tangent_vecs = self.encoder(x)
        z = self.manifold.expmap_0(tangent_vecs)
        return z, tangent_vecs

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        tangent_vecs = self.manifold.logmap_0(z)
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

    def training_step(self, batch, batch_idx):
        x_aug, x, y = batch
        x_recon, z, _ = self(x)
        x_aug_recon, z_aug, _ = self(x_aug)
        x_recon_loss = F.mse_loss(x_recon, x)
        x_dist_loss = torch.Tensor([max(self.manifold.dist_0(z).mean()-0.5, 0.0)]).to(self.device)
        x_aug_recon_loss = F.mse_loss(x_aug_recon, x_aug)
        x_aug_dist_loss = torch.Tensor([max(self.manifold.dist_0(z_aug).mean()-0.5, 0.0)]).to(self.device)
        recon_loss = (x_recon_loss + x_aug_recon_loss)/2
        dist_loss = (x_dist_loss + x_aug_dist_loss)/2
        loss = recon_loss #+ dist_loss
        
        self.metrics.setdefault('pretrain/x_recon_loss', []).append(x_recon_loss.detach())
        self.metrics.setdefault('pretrain/x_dist_loss', []).append(x_dist_loss)
        self.metrics.setdefault('pretrain/x_aug_recon_loss', []).append(x_aug_recon_loss.detach())
        self.metrics.setdefault('pretrain/x_aug_dist_loss', []).append(x_aug_dist_loss)
        self.metrics.setdefault('pretrain/recon_loss', []).append(recon_loss.detach())
        self.metrics.setdefault('pretrain/dist_loss', []).append(dist_loss)
        self.metrics.setdefault('pretrain/loss', []).append(loss.detach())
        self.metrics.setdefault('pretrain/dist_x2x_aug', []).append(self.manifold.dist(z, z_aug).mean())
        return loss
    
    def on_train_epoch_end(self):
        labels = []
        embeddings = []
        if not (self.log_interval > 0 and self.current_epoch % self.log_interval == 0):
            return
        if not isinstance(self.logger.experiment, wandb.sdk.wandb_run.Run):
            return
        with torch.no_grad():
            for x_aug, x, y in self.trainer.datamodule.test_dataloader():
                x_recon, z, tangent_vecs = self(x.to(self.device))
                labels.append(y.detach())
                embeddings.append(z.detach())
            labels = torch.cat(labels).cpu()
            embeddings = torch.cat(embeddings).cpu()
            em_plot = embedddings_plot(embeddings, labels, self.manifold)
            del embeddings
            del labels
            self.metrics['pretrain/embeddings'] = wandb.Image(em_plot)
            self.metrics['pretrain/epoch'] = self.current_epoch
            for key in self.metrics:
                if isinstance(self.metrics[key], List):
                    self.metrics[key] = torch.stack(self.metrics[key]).mean()
                self.logger.experiment.log(self.metrics, step=self.current_epoch)
            plt.close()
            self.metrics = {}

            return

