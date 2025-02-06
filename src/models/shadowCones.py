import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from src.models.resnet34 import HyperbolicResNet34
from src.hyperbolic_math.src.manifolds.poincare import PoincareBall
from src.hyperbolic_math.src.utils.math_utils import arcosh

class HyperbolicContrastiveModel(HyperbolicResNet34):
    def __init__(
        self,
        resnet: HyperbolicResNet34,
        manifold: PoincareBall,
        light_radius: float = 0.5,
        temperature: float = 0.1,
        contrastive_weight: float = 0.1,
        **kwargs
    ):
        # Get latent_dim from resnet model
        latent_dim = resnet.encoder.fc[-1].out_features
        super().__init__(manifold=manifold, latent_dim=latent_dim, **kwargs)
        
        # Load encoder weights
        self.encoder.load_state_dict(resnet.encoder.state_dict())
        
        # Load decoder attributes and weights
        self.linear = resnet.linear
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2 
        self.layer3 = resnet.layer3
        self.conv1 = resnet.conv1
        self.tanh = resnet.tanh
        
        self.light_radius = light_radius
        self.temperature = temperature
        self.contrastive_weight = contrastive_weight
        self.save_hyperparameters(ignore=['resnet'])


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
        min_sum = torch.minimum(in_x_cone_sum, in_x_aug_cone_sum)    
        return torch.where(min_sum == 0, torch.zeros_like(intersection), intersection/min_sum)


    def training_step(self, batch, batch_idx):
        x_aug, x, y = batch
        
        # Reconstruction loss
        x_recon, z, tangent_vecs = self(x)
        recon_loss = F.mse_loss(x_recon, x)
        _, x_aug_z, _ = self(x_aug)
        
        points = []
        for _, z_batch, _ in self.trainer.datamodule.train_dataloader():
            _, z_points, _ = self(z_batch.to(self.device))
            points.append(z_points)
        points = torch.cat(points)

        similarities = self.batch_cone_membership(z, x_aug_z, points, self.light_radius)
        logits = similarities / self.temperature 
        pos_sim = logits.diag()
        exp_logits = torch.exp(logits)
        neg_sim = torch.log(exp_logits.sum(dim=0))
        contrastive_loss = -(pos_sim - neg_sim).mean()
        
        # Combined loss
        total_loss = recon_loss + self.contrastive_weight * contrastive_loss
        
        # Logging
        self.log('train/recon_loss', recon_loss, prog_bar=True)
        self.log('train/contrastive_loss', contrastive_loss, prog_bar=True)
        self.log('train/total_loss', total_loss)
        
        return total_loss