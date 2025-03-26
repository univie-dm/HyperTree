import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from sklearn.manifold import TSNE
from typing import Type
from src.hyperbolic_math.src.manifolds import Manifold
from src.hyperbolic_math.src.utils.helpers import compute_pairwise_distances


# TODO:
def create_label_mapping(all_embeddings_np, all_labels_np, class_names, label_type='class'):
    # Create label mappings
    if class_names and isinstance(class_names[0], tuple):
        label_mapping = {}
        superclass_mapping = {}
        for label in np.unique(all_labels_np):
            class_name, superclass = class_names[label]
            superclass_mapping[label] = superclass if superclass else class_name
            if label_type == 'super':
                label_mapping[label] = superclass
            elif label_type == 'class':
                label_mapping[label] = class_name
            elif label_type == 'both':
                label_mapping[label] = f"{class_name} ({superclass})"
    else:
        label_mapping = {label: str(label) for label in np.unique(all_labels_np)}


def dendrogram_purity(embeddings, labels, manifold=None, linkage_method='average'):
    """
    Compute the dendrogram purity for hierarchical clustering evaluation.
    
    Args:
        embeddings (torch.Tensor or np.ndarray): Hyperbolic embeddings of shape (n_samples, embedding_dim)
        labels (torch.Tensor or np.ndarray): Ground truth labels of shape (n_samples,)
        manifold (Manifold, optional): The hyperbolic manifold used for distance computation.
            If None, assumes embeddings are already in the form of a distance matrix.
        linkage_method (str, optional): The linkage method for hierarchical clustering.
            Options: 'single', 'complete', 'average', 'ward'. Default: 'average'.
            
    Returns:
        float: Dendrogram purity score (between 0 and 1)
    """
    import torch
    import numpy as np
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    
    # Convert inputs to numpy arrays if they are torch tensors
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # Compute pairwise distances if a manifold is provided
    if manifold is not None:
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings, device=manifold.c.device)
        distances = compute_pairwise_distances(embeddings, manifold).detach().cpu().numpy()
    else:
        # Assume embeddings is already a distance matrix
        distances = embeddings
    
    # Ensure the distance matrix is symmetric
    if not np.allclose(distances, distances.T):
        distances = (distances + distances.T) / 2
    
    # Convert distance matrix to condensed form for scipy
    condensed_distances = squareform(distances)
    
    # Perform hierarchical clustering
    Z = linkage(condensed_distances, method=linkage_method)
    
    # Compute dendrogram purity
    num_samples = len(labels)
    unique_labels = np.unique(labels)
    
    # Create a mapping from label to a list of sample indices with that label
    label_to_samples = {label: np.where(labels == label)[0] for label in unique_labels}
    
    # Compute same-class pairs
    same_class_pairs = []
    for label in unique_labels:
        samples = label_to_samples[label]
        if len(samples) > 1:
            pairs = [(i, j) for i in samples for j in samples if i < j]
            same_class_pairs.extend(pairs)
    
    if not same_class_pairs:
        return 1.0  # Perfect purity if no same-class pairs
    
    # More direct approach using scipy's flat cluster assignments
    total_purity = 0.0
    n_pairs = len(same_class_pairs)
    
    # Use a range of different cluster counts to simulate different levels of the hierarchy
    # Start with many clusters (almost like leaves) and gradually merge them
    max_k = min(num_samples, 100)  # Don't go beyond number of samples
    ks = [max(2, max_k - i*5) for i in range(20)]  # Different levels
    ks = sorted(list(set(ks)))  # Remove duplicates and sort
    
    # Keep track of which pairs have been processed
    processed_pairs = set()
    
    for k in ks:
        # Get cluster assignments for this level of the hierarchy
        cluster_assignments = fcluster(Z, t=k, criterion='maxclust')
        
        # Create a mapping from cluster ID to its members
        clusters = {}
        for i, cluster_id in enumerate(cluster_assignments):
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(i)
        
        # Compute purity for each cluster
        cluster_purities = {}
        for cluster_id, members in clusters.items():
            # Count occurrences of each label in this cluster
            label_counts = {}
            for member in members:
                label = labels[member]
                label_counts[label] = label_counts.get(label, 0) + 1
            
            # Compute purity as ratio of most common label
            max_count = max(label_counts.values()) if label_counts else 0
            purity = max_count / len(members)
            cluster_purities[cluster_id] = purity
        
        # For each same-class pair, if they're in the same cluster at this level, 
        # add the purity to our total
        remaining_pairs = []
        for i, j in same_class_pairs:
            if (i, j) in processed_pairs:
                continue
                
            cluster_i = cluster_assignments[i]
            cluster_j = cluster_assignments[j]
            
            if cluster_i == cluster_j:
                total_purity += cluster_purities[cluster_i]
                processed_pairs.add((i, j))
            else:
                remaining_pairs.append((i, j))
        
        # Update the list of pairs to process
        same_class_pairs = remaining_pairs
        
        # If all pairs are processed, we're done
        if not same_class_pairs:
            break
    
    # If there are any remaining pairs, assign them a default low purity value
    # This happens if some pairs are never in the same cluster even at the root
    if same_class_pairs:
        for _ in same_class_pairs:
            total_purity += 0.5  # A reasonable default value
    
    # Return average purity
    return total_purity / n_pairs


def get_samples_in_cluster(Z, cluster_idx, num_samples):
    """
    Helper function to get all sample indices in a cluster.
    This is an iterative implementation to avoid recursion depth errors.
    
    This function should not be used directly. The dendrogram_purity function
    uses a more efficient approach to track clusters.
    """
    to_process = [cluster_idx]
    result = []
    
    while to_process:
        idx = to_process.pop()
        # If idx < num_samples, it's a leaf node (original data point)
        if idx < num_samples:
            result.append(idx)
        else:
            # It's a non-leaf node, add its children to the processing queue
            node_idx = idx - num_samples
            left_child = int(Z[node_idx, 0])
            right_child = int(Z[node_idx, 1])
            
            to_process.append(right_child)
            to_process.append(left_child)
    
    return result


def compute_hierarchical_purity(embeddings, labels, manifold=None, linkage_method='average'):
    """
    A simpler version of dendrogram purity that uses scikit-learn's implementation.
    This is more efficient but slightly less informative than the full dendrogram purity.
    
    Args:
        embeddings (torch.Tensor or np.ndarray): Hyperbolic embeddings
        labels (torch.Tensor or np.ndarray): Ground truth labels
        manifold (Manifold, optional): The hyperbolic manifold for distance computation
        linkage_method (str, optional): Linkage method for hierarchical clustering
        
    Returns:
        dict: A dictionary with purity scores at different levels of the hierarchy
    """
    import torch
    import numpy as np
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    from sklearn.metrics import normalized_mutual_info_score
    
    # Convert to numpy if needed
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # Compute distances
    if manifold is not None:
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings, device=manifold.c.device)
        distances = compute_pairwise_distances(embeddings, manifold).detach().cpu().numpy()
    else:
        distances = embeddings
    
    # Ensure symmetric
    if not np.allclose(distances, distances.T):
        distances = (distances + distances.T) / 2
    
    # Convert to condensed form
    condensed_distances = squareform(distances)
    
    # Perform hierarchical clustering
    Z = linkage(condensed_distances, method=linkage_method)
    
    # Compute purity at different levels by varying the distance threshold
    num_clusters_list = [2, 5, 10, 20, 50, 100]  # Adjust based on your dataset size
    result = {}
    
    for num_clusters in num_clusters_list:
        if num_clusters >= len(labels):
            continue
            
        # Cut the dendrogram to get the specified number of clusters
        predicted_labels = fcluster(Z, t=num_clusters, criterion='maxclust')
        
        # Compute NMI score
        nmi = normalized_mutual_info_score(labels, predicted_labels)
        result[f'nmi_{num_clusters}'] = nmi
        
        # Compute purity for each cluster
        cluster_purities = []
        for cluster_id in np.unique(predicted_labels):
            cluster_indices = np.where(predicted_labels == cluster_id)[0]
            cluster_labels = labels[cluster_indices]
            
            # Count occurrences of each label
            label_counts = np.bincount(cluster_labels, minlength=len(np.unique(labels)))
            cluster_purity = np.max(label_counts) / len(cluster_indices)
            cluster_purities.append(cluster_purity)
        
        # Average purity weighted by cluster size
        result[f'purity_{num_clusters}'] = np.mean(cluster_purities)
    
    # Overall average purity
    result['avg_purity'] = np.mean([value for key, value in result.items() if key.startswith('purity_')])
    result['avg_nmi'] = np.mean([value for key, value in result.items() if key.startswith('nmi_')])
    
    return result


def evaluate_hierarchical_clustering(embeddings, labels, manifold=None, linkage_methods=None):
    """
    Comprehensive evaluation of hierarchical clustering quality based on embeddings and labels.
    
    Args:
        embeddings (torch.Tensor or np.ndarray): Embeddings of data points
        labels (torch.Tensor or np.ndarray): Ground truth labels
        manifold (Manifold, optional): Hyperbolic manifold for distance computation
        linkage_methods (list, optional): List of linkage methods to evaluate.
            Default is ['single', 'complete', 'average', 'ward']
            
    Returns:
        dict: Dictionary with all evaluation metrics
    """
    if linkage_methods is None:
        linkage_methods = ['single', 'complete', 'average', 'ward']
    
    results = {}
    
    for method in linkage_methods:
        # Compute dendrogram purity
        dp_score = dendrogram_purity(embeddings, labels, manifold, method)
        results[f'dendrogram_purity_{method}'] = dp_score
        
        # Compute hierarchical purity metrics
        purity_metrics = compute_hierarchical_purity(embeddings, labels, manifold, method)
        for key, value in purity_metrics.items():
            results[f'{key}_{method}'] = value
    
    # Compute average metrics across all linkage methods
    dp_scores = [results[f'dendrogram_purity_{method}'] for method in linkage_methods]
    results['dendrogram_purity_avg'] = sum(dp_scores) / len(dp_scores)
    
    avg_purity_scores = [results[f'avg_purity_{method}'] for method in linkage_methods]
    results['avg_purity_overall'] = sum(avg_purity_scores) / len(avg_purity_scores)
    
    avg_nmi_scores = [results[f'avg_nmi_{method}'] for method in linkage_methods]
    results['avg_nmi_overall'] = sum(avg_nmi_scores) / len(avg_nmi_scores)
    
    return results

# TODO:
'''
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
'''