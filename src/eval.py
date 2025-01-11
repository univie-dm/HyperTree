import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from typing import Tuple, List
import torch

def get_cluster_pairs(labels: np.ndarray) -> List[Tuple[int, int]]:
    """Get all pairs of points that belong to the same cluster."""
    n = len(labels)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                pairs.append((i, j))
    return pairs

def compute_dendrogram_purity(
    distances: torch.Tensor,
    true_labels: torch.Tensor,
    method: str = 'complete'
) -> float:
    """
    Compute the dendrogram purity metric.
    
    Args:
        distances: Pairwise distance matrix (n x n)
        true_labels: Ground truth labels (n)
        method: Linkage method for hierarchical clustering
        
    Returns:
        float: Dendrogram purity score
    """
    # Convert to numpy for scipy compatibility
    if isinstance(distances, torch.Tensor):
        distances = distances.cpu().numpy()
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.cpu().numpy()
    
    # Convert similarities to distances if needed
    if np.all(distances >= 0) and np.all(distances <= 1):
        distances = 1 - distances
    
    # Ensure the matrix is symmetric
    distances = (distances + distances.T) / 2
    
    # Compute linkage matrix
    condensed_dist = distances[np.triu_indices_from(distances, k=1)]
    Z = linkage(condensed_dist, method=method)
    
    # Get ground truth pairs
    true_pairs = set(map(tuple, map(sorted, get_cluster_pairs(true_labels))))
    
    if len(true_pairs) == 0:
        return 1.0  # Perfect purity if no true pairs exist
    
    # Evaluate purity at different cut heights
    purities = []
    cut_heights = np.unique(Z[:, 2])  # All possible merging heights
    
    for height in cut_heights:
        # Get cluster assignments at this height
        pred_labels = fcluster(Z, height, criterion='distance')
        pred_pairs = set(map(tuple, map(sorted, get_cluster_pairs(pred_labels))))
        
        if len(pred_pairs) == 0:
            continue
            
        # Compute purity at this level
        correct_pairs = len(true_pairs.intersection(pred_pairs))
        purity = correct_pairs / len(pred_pairs) if len(pred_pairs) > 0 else 0
        purities.append(purity)
    
    # Return weighted average purity across all levels
    return np.mean(purities) if purities else 0.0