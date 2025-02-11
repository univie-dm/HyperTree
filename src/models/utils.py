import torch
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from src.hyperbolic_math.src.utils.math_utils import arcosh

def embedddings_plot(x, y, manifold, class_names=None):
    x = x.detach().cpu().numpy()
    if x.shape[1] > 2:
        pairwise_dists = compute_strided_pairwise_distances(torch.Tensor(x), x.shape[0], manifold.min_enorm, manifold.c.cpu(), 512)
        x = TSNE(n_components=2, metric="precomputed", init="random").fit_transform(pairwise_dists)
    # Normalize points to fit within the circle
    #x = x / np.linalg.norm(x, axis=1, keepdims=True)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    # Draw boundary circle
    circle = plt.Circle((0, 0), 1, fill=False, color='black', alpha=0.5)
    ax.add_artist(circle)

    # Scatter plot with normalized points
    scatter = plt.scatter(x[:, 0], x[:, 1], c=y, alpha=0.6, cmap='tab10')

    # Add legend with labels
    if class_names:
        handles, labels = scatter.legend_elements()
        labels = [class_names[int(label)] for label in labels]
        legend1 = ax.legend(handles, labels, title="Classes", loc="upper right")
    else:
        legend1 = ax.legend(*scatter.legend_elements(), title="Classes", loc="upper right")
    ax.add_artist(legend1)

    # Set equal aspect ratio and limits
    ax.set_aspect('equal')
    #ax.set_xlim(-1.1, 1.1)
    #ax.set_ylim(-1.1, 1.1)
    
    # Add title
    plt.title("Hyperbolic Embeddings")
    
    return plt.gcf()


@torch.jit.script
def compute_strided_pairwise_distances(z: torch.Tensor, n: int, manifold_min_enorm: float, c: torch.Tensor, batch_size: int) -> torch.Tensor:
    
    z_sqnorm = z.pow(2).sum(dim=-1)
    distances = torch.zeros((n, n)).cpu()
    
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
