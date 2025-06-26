import torch
import pytorch_lightning as pl
import torch.nn.functional as F

from typing import Tuple, Type
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from src.models.abstract import AbstractModel
from ..hyperbolic_math.src.utils.helpers import get_delta


#TODO: Unsupported
class HyperbolicContrastiveModel(AbstractModel):
    def __init__(
        self,
        ae_model: AbstractModel,
        clustering_trainable_curvature: bool = False,
        learning_rate: float = 1e-3,
        contrastive_weight: float = 0.1,
        contrastive_temperature: float = 0.1,
        light_radius: float = 0.5,
        start_epoch: int = 0,
        log_interval: int = 5
    ):
        super().__init__(
            manifold_name=ae_model.manifold.name,
            curvature_magnitude=ae_model.manifold.c,
            trainable_curvature=clustering_trainable_curvature,
            latent_dim=ae_model.latent_dim,
            learning_rate=learning_rate,
            log_interval=log_interval
        )
        self.global_epoch = start_epoch

        self.contrastive_weight = contrastive_weight
        self.contrastive_temperature = contrastive_temperature
        self.light_radius = light_radius

        # Load encoder, decoder, and to_tangent modules with their weights
        self.encoder = ae_model.encoder
        self.decoder = ae_model.decoder
        self.to_tangent = ae_model.to_tangent

        # Ensure weights are copied by loading state dict
        self.encoder.load_state_dict(ae_model.encoder.state_dict())
        self.decoder.load_state_dict(ae_model.decoder.state_dict())
        self.to_tangent.load_state_dict(ae_model.to_tangent.state_dict())

    # TODO:
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

        x_recon, manifold_z, _ = self(x)
        x_aug_recon, manifold_z_aug, _ = self(x_aug)

        # Reconstruction loss
        x_recon_loss = F.mse_loss(x_recon, x)
        x_aug_recon_loss = F.mse_loss(x_aug_recon, x_aug)
        recon_loss = (x_recon_loss + x_aug_recon_loss) / 2

        # Shadow Cone contrastive loss
        # TODO:
        '''
        points = []
        for _, z_batch, _ in self.trainer.datamodule.test_dataloader():
            _, z_points, _ = self(z_batch.to(self.device))
            points.append(z_points.detach())
        points = torch.cat(points)

        cone_similarities = self.batch_cone_membership(z, z_aug, points, self.light_radius)
        '''
        #cone_similarities = self.cone_overlapping_calculator(manifold_z, manifold_z_aug, method='analytical')
        cone_similarities = self.manifold.dist(manifold_z, manifold_z_aug)
        self.metrics.setdefault("clustering/similarities", []).append(cone_similarities.mean())
        cone_logits = cone_similarities / self.contrastive_temperature
        cone_pos_sim = cone_logits.diag()
        cone_exp_logits = torch.exp(cone_logits)
        cone_neg_sim = torch.log(cone_exp_logits.sum(dim=0))
        cone_contrastive_loss = -(cone_pos_sim - cone_neg_sim).mean()

        '''
        dist_similarities = self.manifold.dist(manifold_z, manifold_z_aug)
        dist_logits = dist_similarities / self.contrastive_temperature
        dist_pos_sim = dist_logits.diag()
        dist_exp_logits = torch.exp(dist_logits)
        dist_neg_sim = torch.log(dist_exp_logits.sum(dim=0))
        dist_contrastive_loss = -(dist_pos_sim - dist_neg_sim).mean()

        contrastive_loss = cone_contrastive_loss #+ dist_contrastive_loss
        total_loss = recon_loss + self.contrastive_weight * contrastive_loss'
        '''
        contrastive_loss = torch.tensor(0.0, device=self.device)
        dist_contrastive_loss = torch.tensor(0.0, device=self.device)
        total_loss = torch.tensor(0.0, device=self.device)

        # Logging
        self.metrics.setdefault("clustering/recon_loss", []).append(recon_loss)
        self.metrics.setdefault("clustering/x_recon_loss", []).append(x_recon_loss)
        self.metrics.setdefault("clustering/x_aug_recon_loss", []).append(x_aug_recon_loss)

        self.metrics.setdefault("clustering/contrastive_loss", []).append(contrastive_loss)
        self.metrics.setdefault("clustering/cone_contrastive_loss", []).append(cone_contrastive_loss)
        self.metrics.setdefault("clustering/dist_contrastive_loss", []).append(dist_contrastive_loss)

        self.metrics.setdefault("clustering/total_loss", []).append(total_loss)
        self.metrics.setdefault('clustering/dist_x2x_aug', []).append(self.manifold.dist(manifold_z, manifold_z_aug).mean())
        self.metrics.setdefault('clustering/c', []).append(self.manifold.c.detach())

        return total_loss

    def on_train_epoch_end(self) -> None:
        if not isinstance(self.logger, pl.loggers.WandbLogger): # type: ignore
            return
        if self.current_epoch % self.log_interval == 0:
            with torch.no_grad():
                test_loader = self.trainer.datamodule.test_dataloader() # type: ignore

                embeddings = []
                labels = []

                for batch_idx, (x_aug, x, y) in enumerate(test_loader):
                    x = x.view(x.size(0), -1)
                    _, z, _ = self(x.to(self.device))
                    embeddings.append(z)
                    labels.append(y.cpu())

                all_embeddings = torch.cat(embeddings, dim=0)
                all_labels = torch.cat(labels, dim=0)

                delta, diam, rel_delta = get_delta(all_embeddings, self.manifold)

                # Move to CPU for NMI and ARI calculation
                all_embeddings_np = all_embeddings.cpu().numpy()
                all_labels_np = all_labels.cpu().numpy()

                agglomerative_clustering = AgglomerativeClustering(n_clusters=10)
                agglomerative_clustering.fit(all_embeddings_np)
                nmi = normalized_mutual_info_score(all_labels_np, agglomerative_clustering.labels_)
                ari = adjusted_rand_score(all_labels_np, agglomerative_clustering.labels_)

                # Log the metrics
                self.metrics.setdefault("test/nmi", []).append(nmi)
                self.metrics.setdefault("test/ari", []).append(ari)
                self.metrics.setdefault("test/delta", []).append(delta.item())
                self.metrics.setdefault("test/diam", []).append(diam.item())
                self.metrics.setdefault("test/rel_delta", []).append(rel_delta.item())

        self.metrics['clustering/epoch'] = self.current_epoch
        for key, value in self.metrics.items():
            if isinstance(value, list):
                self.metrics[key] = sum(value) / len(value)
        self.logger.experiment.log(self.metrics, step=self.global_epoch+self.current_epoch) # type: ignore
        self.metrics = {}
