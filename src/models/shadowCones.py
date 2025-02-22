from typing import List, Tuple
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb
import matplotlib.pyplot as plt
from src.models.resnet import HyperbolicResNet
from src.models.abstract import AbstractModel
from src.hyperbolic_math.src.manifolds.poincare import PoincareBall
from src.hyperbolic_math.src.utils.math_utils import arcosh
from src.models.utils import embedddings_plot, ConeOverlapCalculator, sample_n_points_per_class
from src.hyperbolic_math.src.utils.helpers import compute_pairwise_distances
from src.eval import dendrogram_purity
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage

class HyperbolicContrastiveModel(AbstractModel):
    def __init__(
        self,
        ae_model: AbstractModel,
        manifold: PoincareBall,
        light_radius: float = 0.5,
        temperature: float = 0.1,
        contrastive_weight: float = 0.1,
        learning_rate: float = 1e-3,
        **kwargs
    ):

        super().__init__()
        self.global_epoch = ae_model.current_epoch
        
        # Load encoder weights
        self.encoder = ae_model.encoder
        
        # Load decoder attributes and weights
        self.decoder = ae_model.decoder
        self.to_tangent = ae_model.to_tangent
        
        self.light_radius = light_radius
        self.temperature = temperature
        self.contrastive_weight = contrastive_weight
        self.log_interval = ae_model.log_interval
        self.learning_rate = learning_rate
        self.manifold = manifold
        self.cone_overlapping_calculator = ConeOverlapCalculator(manifold, light_radius)
        self.metrics = {}
        
    def setup_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


    def batch_cone_membership(self, x: torch.Tensor, x_aug: torch.Tensor, points: torch.Tensor, light_radius: float, curvature: float = 1.0) -> torch.Tensor:
        eps = 1e-8
        batch_size = x.shape[0]
        num_points = points.shape[0]
        
        # Compute for x
        def is_in_cone(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            u_norm_sq = torch.sum(u**2, dim=1, keepdim=True) 
            d_uO = torch.arccosh(1 + (2 * curvature * u_norm_sq) / (1 - curvature * u_norm_sq + eps))
            
            # Compute theta
            sqrt_c = torch.sqrt(torch.tensor(curvature, device=u.device))
            numerator = torch.sinh(sqrt_c * light_radius)
            denominator = torch.sinh(sqrt_c * d_uO)
            theta_u = torch.arcsin(torch.clamp(numerator / (denominator + eps), -1+eps, 1-eps))
            
            # Compute phi
            dot_product = torch.sum(u * v, dim=1)
            u_norm = torch.norm(u, dim=1)
            v_norm = torch.norm(v, dim=1)
            uv_norm = torch.norm(u - v, dim=1)
            
            term1 = dot_product * (1 + curvature * u_norm**2)
            term2 = u_norm**2 * (1 + curvature * v_norm**2)
            numerator = term1 - term2
            
            denominator = u_norm * uv_norm * torch.sqrt(
                torch.clamp(1 + curvature**2 * u_norm**2 * v_norm**2 - 2 * curvature * dot_product, min=eps)
            )
            
            cos_phi = torch.clamp(numerator / (denominator + eps), -1+eps, 1-eps)
            phi = torch.arccos(cos_phi)
            
            return phi <= theta_u.squeeze()

        # Test each point against x and x_aug cones
        in_x_cone = torch.zeros((batch_size, num_points), device=x.device)
        in_x_aug_cone = torch.zeros((batch_size, num_points), device=x.device)
        
        for i in range(batch_size):
            in_x_cone[i] = is_in_cone(x[i].unsqueeze(0).expand(num_points, -1), points)
            in_x_aug_cone[i] = is_in_cone(x_aug[i].unsqueeze(0).expand(num_points, -1), points)
        
        # Compute intersection using point membership

        intersection = torch.sum(in_x_cone * in_x_aug_cone, dim=1).float()
        in_x_cone_sum = torch.sum(in_x_cone, dim=1)
        in_x_aug_cone_sum = torch.sum(in_x_aug_cone, dim=1)
        self.metrics.setdefault("clustering/in_x_cone", []).append(in_x_cone_sum.mean())
        self.metrics.setdefault("clustering/in_x_aug_cone", []).append(in_x_aug_cone_sum.mean())
        min_sum = torch.minimum(in_x_cone_sum, in_x_aug_cone_sum)    
        return torch.where(min_sum == 0, torch.zeros_like(intersection), intersection/min_sum)
    
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
        
        # Reconstruction loss
        x_recon, z, tangent_vecs = self(x)
        x_aug_recon, z_aug, _ = self(x_aug)
        x_recon_loss = F.mse_loss(x_recon, x)
        aug_recon_loss = F.mse_loss(x_aug_recon, x_aug)
        recon_loss = (x_recon_loss + aug_recon_loss) / 2
        
        '''
        points = []
        for _, z_batch, _ in self.trainer.datamodule.test_dataloader():
            _, z_points, _ = self(z_batch.to(self.device))
            points.append(z_points.detach())
        points = torch.cat(points)

        cone_similarities = self.batch_cone_membership(z, z_aug, points, self.light_radius)
        '''
        cone_similarities = self.cone_overlapping_calculator(z, z_aug, method='analytical')
        self.metrics.setdefault("clustering/similarities", []).append(cone_similarities.mean())
        cone_logits = cone_similarities / self.temperature 
        cone_pos_sim = cone_logits.diag()
        cone_exp_logits = torch.exp(cone_logits)
        cone_neg_sim = torch.log(cone_exp_logits.sum(dim=0))
        cone_contrastive_loss = -(cone_pos_sim - cone_neg_sim).mean()

        '''
        dist_similarities = self.manifold.dist(z, z_aug)
        dist_logits = dist_similarities / self.temperature
        dist_pos_sim = dist_logits.diag()
        dist_exp_logits = torch.exp(dist_logits)
        dist_neg_sim = torch.log(dist_exp_logits.sum(dim=0))
        dist_contrastive_loss = -(dist_pos_sim - dist_neg_sim).mean()
        '''
        contrastive_loss = cone_contrastive_loss #+ dist_contrastive_loss
        
        # Combined loss
        total_loss = recon_loss + self.contrastive_weight * contrastive_loss
        
        # Logging
        self.metrics.setdefault("clustering/recon_loss", []).append(recon_loss)
        self.metrics.setdefault("clustering/x_recon_loss", []).append(x_recon_loss)
        self.metrics.setdefault("clustering/aug_recon_loss", []).append(aug_recon_loss)
        self.metrics.setdefault("clustering/contrastive_loss", []).append(contrastive_loss)
        self.metrics.setdefault("clustering/cone_contrastive_loss", []).append(cone_contrastive_loss)
        #self.metrics.setdefault("train/dist_contrastive_loss", []).append(dist_contrastive_loss)
        self.metrics.setdefault("clustering/total_loss", []).append(total_loss)
        self.metrics.setdefault('clustering/dist_x2x_aug', []).append(self.manifold.dist(z, z_aug).mean())
        self.metrics.setdefault('clustering/c', []).append(self.manifold.c.detach())

        return total_loss
    
    def on_train_epoch_end(self):
        labels = []
        embeddings = []
        if not isinstance(self.logger.experiment, wandb.sdk.wandb_run.Run):
            return
        if self.current_epoch % self.log_interval == 0:
            with torch.no_grad():
                '''
                n = 100  # Number of points per class
                num_classes = len(self.trainer.datamodule.get_class_names())
                x_aug_sampled, x_sampled, y_sampled = sample_n_points_per_class(
                    self.trainer.datamodule.test_dataloader(), n, num_classes
                )
                x_recon, z, tangent_vecs = self(x_sampled.to(self.device))
                labels = y_sampled.detach().cpu()
                embeddings = z.detach().cpu()
                em_plot = embedddings_plot(embeddings, labels, self.manifold, self.trainer.datamodule.get_class_names())
                self.metrics['train/embeddings'] = wandb.Image(em_plot)
                '''
                embeddings = []
                labels = []
                euclidean_embeddings = []
                for x_aug, x, y in self.trainer.datamodule.test_dataloader():
                    x = x.view(x.size(0), -1)
                    x_aug = x_aug.view(x_aug.size(0), -1)
                    z, tangent_vecs = self.encode(x.to(self.device))
                    embeddings.append(z.detach())
                    euclidean_embeddings.append(tangent_vecs.detach())
                    labels.append(y.detach())

                embeddings = torch.cat(embeddings)
                labels = torch.cat(labels) 
                euclidean_embeddings = torch.cat(euclidean_embeddings).cpu().numpy()

                pair_dist = compute_pairwise_distances(embeddings, self.manifold)
                pair_dist = pair_dist.cpu().numpy()
                labels = labels.cpu().numpy()
                num_classes = len(self.trainer.datamodule.get_class_names())

                agglo_clustering = AgglomerativeClustering(n_clusters=num_classes, linkage="average", metric="precomputed")
                pred_labels = agglo_clustering.fit_predict(pair_dist)
                nmi = normalized_mutual_info_score(labels, pred_labels)
                self.metrics['clustering/hyperbolic_nmi'] = nmi
                ari = adjusted_rand_score(labels, pred_labels)
                self.metrics['clustering/hyperbolic_ari'] = ari

               

        self.metrics['clustering/epoch'] = self.current_epoch
        for key in self.metrics:
            if isinstance(self.metrics[key], List):
                self.metrics[key] = torch.stack(self.metrics[key]).mean()
            self.logger.experiment.log(self.metrics, step=self.current_epoch+self.global_epoch)
        plt.close()
        self.metrics = {}