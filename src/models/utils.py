import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from sklearn.manifold import TSNE
from typing import Type
from src.hyperbolic_math.src.manifolds import Manifold
from src.hyperbolic_math.src.utils.helpers import compute_pairwise_distances


def embedddings_plot(x: torch.Tensor, y: torch.Tensor, manifold: Type[Manifold], class_names=None) -> plt.Figure:
    if x.shape[1] > 2:
        pairwise_dists = compute_pairwise_distances(x, manifold).detach().cpu().numpy()
        x = TSNE(n_components=2, metric="precomputed", init="random").fit_transform(pairwise_dists)
    else:
        x = x.detach().cpu().numpy()

    # Ensure that labels are also on the CPU and numpy
    y = y.cpu().numpy()

    # Create figure
    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    # Get unique classes for proper legend handling
    unique_classes = np.unique(y)

    # Create scatter plot
    scatter = plt.scatter(x[:, 0], x[:, 1], c=y, alpha=0.6, cmap='tab10')

    # Add legend with labels
    if class_names is not None:
        # Create custom legend with class names
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=plt.cm.tab10(i / 10),
                                  label=class_names[i][0],
                                  markersize=10
                                  ) for i in unique_classes]
        ax.legend(handles=legend_elements, title="Classes", loc="upper right")
    else:
        # Use default numeric labels
        ax.legend(*scatter.legend_elements(), title="Classes", loc="upper right")

    # Set equal aspect ratio and expand limits to show full spread
    ax.set_aspect('equal')

    # Get the maximum absolute value for x and y coordinates
    max_val = max(abs(x).max(), abs(y).max())
    margin = max_val * 0.2  # Add 10% margin

    # Set limits symmetrically
    ax.set_xlim(-max_val - margin, max_val + margin)
    ax.set_ylim(-max_val - margin, max_val + margin)

    # Add title
    plt.title("Hyperbolic Embeddings")

    return plt.gcf()


# TODO:
class ConeOverlapCalculator:
    def __init__(self, manifold, light_radius=0.5, num_samples=1000):
        self.manifold = manifold
        self.light_radius = light_radius
        self.num_samples = num_samples
        
    def analytical_cone_intersection(self, x: torch.Tensor, x_aug: torch.Tensor) -> torch.Tensor:
        """
        Calculate cone intersection analytically using angle-based approach.
        Ensures similarity is bounded between 0 and 1.
        
        Args:
            x: Tensor of shape (batch_size, dim)
            x_aug: Tensor of shape (batch_size, dim)
            
        Returns:
            Tensor of shape (batch_size,) containing overlap scores bounded between 0 and 1
        """
        batch_size = x.shape[0]
        device = x.device
        eps = 1e-7
        
        # Calculate distances to origin for all points
        zero = torch.zeros_like(x)
        d_xO = self.manifold.dist(zero, x)  # shape: (batch_size,)
        d_x_augO = self.manifold.dist(zero, x_aug)  # shape: (batch_size,)
        
        # Calculate cone apertures (half-angles)
        sinh_light_radius = torch.sinh(torch.tensor(self.light_radius, device=device))
        
        # Calculate theta values - ensure they're non-negative
        theta_x = torch.abs(torch.arcsin(
            torch.clamp(
                sinh_light_radius / (torch.sinh(d_xO) + eps),
                -1 + eps,
                1 - eps
            )
        ))  # shape: (batch_size,)
        
        theta_x_aug = torch.abs(torch.arcsin(
            torch.clamp(
                sinh_light_radius / (torch.sinh(d_x_augO) + eps),
                -1 + eps,
                1 - eps
            )
        ))  # shape: (batch_size,)
        
        # Calculate angles between vectors
        x_norm = F.normalize(x, p=2, dim=1)  # shape: (batch_size, dim)
        x_aug_norm = F.normalize(x_aug, p=2, dim=1)  # shape: (batch_size, dim)
        
        # Compute cosine similarity
        cos_angle = torch.sum(x_norm * x_aug_norm, dim=1)  # shape: (batch_size,)
        angle_between = torch.arccos(torch.clamp(cos_angle, -1 + eps, 1 - eps))  # shape: (batch_size,)
        
        # Compute intersection score
        # If angle_between is greater than sum of apertures, no intersection
        # If angle_between is less than abs difference of apertures, complete overlap
        # Otherwise, partial overlap
        sum_apertures = theta_x + theta_x_aug
        diff_apertures = torch.abs(theta_x - theta_x_aug)
        
        # Calculate normalized overlap score
        overlap_ratio = torch.clamp(
            (sum_apertures - angle_between) / (2 * torch.maximum(theta_x, theta_x_aug)),
            0.0,
            1.0
        )
        
        return overlap_ratio

    def monte_carlo_cone_intersection(self, x: torch.Tensor, x_aug: torch.Tensor) -> torch.Tensor:
        """
        Calculate cone intersection using Monte Carlo sampling.
        Ensures similarity is bounded between 0 and 1.
        
        Args:
            x: Tensor of shape (batch_size, dim)
            x_aug: Tensor of shape (batch_size, dim)
            
        Returns:
            Tensor of shape (batch_size,) containing overlap scores bounded between 0 and 1
        """
        batch_size = x.shape[0]
        dim = x.shape[1]
        device = x.device
        
        results = torch.zeros(batch_size, device=device)
        
        for i in range(batch_size):
            # Generate random points on unit hypersphere
            samples = torch.randn(self.num_samples, dim, device=device)
            samples = F.normalize(samples, dim=1)
            
            # Scale and translate samples
            midpoint = (x[i] + x_aug[i]) / 2
            distance = torch.norm(x[i] - x_aug[i])
            scaled_samples = samples * distance * self.light_radius
            translated_samples = scaled_samples + midpoint
            
            # Test membership
            in_x_cone = self.test_cone_membership(translated_samples, x[i:i+1])
            in_x_aug_cone = self.test_cone_membership(translated_samples, x_aug[i:i+1])
            
            # Calculate intersection volume
            intersection = torch.sum(in_x_cone & in_x_aug_cone).float()
            union = torch.sum(in_x_cone | in_x_aug_cone).float()
            
            # Ensure result is between 0 and 1
            results[i] = torch.clamp(intersection / (union + eps), 0.0, 1.0)
        
        return results
    
    def test_cone_membership(self, points: torch.Tensor, apex: torch.Tensor) -> torch.Tensor:
        """
        Test if points lie within the cone defined by apex
        
        Args:
            points: Tensor of shape (num_points, dim)
            apex: Tensor of shape (1, dim)
            
        Returns:
            Boolean tensor of shape (num_points,)
        """
        eps = 1e-7
        vectors = points - apex
        
        # Normalize vectors
        vectors_norm = F.normalize(vectors, p=2, dim=1)
        apex_norm = F.normalize(apex, p=2, dim=1)
        
        # Calculate angles
        cos_angles = torch.sum(vectors_norm * apex_norm, dim=1)
        angles = torch.arccos(torch.clamp(cos_angles, -1 + eps, 1 - eps))
        
        # Calculate cone aperture
        zero = torch.zeros_like(apex)
        d_apex = self.manifold.dist(zero, apex)
        theta_apex = torch.arcsin(
            torch.clamp(
                torch.sinh(torch.tensor(self.light_radius, device=points.device)) /
                (torch.sinh(d_apex) + eps),
                -1 + eps,
                1 - eps
            )
        )
        
        return angles <= theta_apex

    def __call__(self, x: torch.Tensor, x_aug: torch.Tensor, method='analytical'):
        """
        Calculate cone overlap using specified method
        
        Args:
            x: Tensor of shape (batch_size, dim)
            x_aug: Tensor of shape (batch_size, dim)
            method: 'analytical' or 'monte_carlo'
            
        Returns:
            Tensor of shape (batch_size,) containing overlap scores bounded between 0 and 1
        """
        if method == 'analytical':
            overlap = self.analytical_cone_intersection(x, x_aug)
            # Double check bounds
            return torch.clamp(overlap, 0.0, 1.0)
        else:
            return self.monte_carlo_cone_intersection(x, x_aug)
        

