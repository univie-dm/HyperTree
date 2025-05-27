import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from sklearn.manifold import TSNE
from typing import Union, Optional
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


def dendrogram_purity(
    embeddings_or_distances: Union[np.ndarray, torch.Tensor],
    labels: Union[np.ndarray, torch.Tensor],
    # Replace 'Any' with Union[Euclidean, Hyperboloid, PoincareBall] or your base Manifold class if available
    manifold: Optional[Manifold] = None, 
    linkage_method: str = 'average'
) -> float:
    """
    Compute the dendrogram purity for hierarchical clustering evaluation.
    This version iterates through merges in the linkage matrix for efficiency.

    The purity is defined as the sum of purities of the clusters where each
    same-class pair first merges, divided by the total number of same-class pairs.

    Args:
        embeddings_or_distances (Union[np.ndarray, torch.Tensor]):
            - Embeddings of shape (n_samples, embedding_dim).
            - OR a precomputed square distance matrix of shape (n_samples, n_samples).
            - OR a precomputed condensed distance matrix (1D array).
        labels (Union[np.ndarray, torch.Tensor]): Ground truth labels of shape (n_samples,).
        manifold (Optional[Any]): The manifold object (e.g., PoincareBall, Euclidean)
            used for distance computation if embeddings are provided. If None,
            and embeddings_or_distances is 2D and square, it's assumed to be a
            Euclidean distance matrix. If embeddings are NxD (D!=N) and manifold is None,
            Euclidean distances will be computed.
        linkage_method (str): The linkage method for hierarchical clustering
            (e.g., 'single', 'complete', 'average', 'ward').

    Returns:
        float: Dendrogram purity score (between 0 and 1).
    """

    if isinstance(labels, torch.Tensor):
        labels_np = labels.detach().cpu().numpy()
    elif isinstance(labels, np.ndarray):
        labels_np = labels
    else:
        # Attempt conversion for other list-like inputs, e.g. Python lists
        try:
            labels_np = np.array(labels)
        except Exception as e:
            raise TypeError(f"Labels could not be converted to a NumPy array: {e}")


    num_samples = len(labels_np)

    if num_samples < 2: # Purity is typically 1.0 for less than 2 samples.
        return 1.0

    if isinstance(embeddings_or_distances, torch.Tensor):
        input_data_np = embeddings_or_distances.detach().cpu().numpy()
    elif isinstance(embeddings_or_distances, np.ndarray):
        input_data_np = embeddings_or_distances
    else:
        raise TypeError("embeddings_or_distances must be a NumPy array or PyTorch Tensor.")

    condensed_distances = None
    if input_data_np.ndim == 1:
        # Assume it's a condensed distance matrix
        expected_condensed_size = num_samples * (num_samples - 1) // 2
        if input_data_np.shape[0] == expected_condensed_size:
            condensed_distances = input_data_np
        else:
            raise ValueError(
                f"Input is 1D but not a valid condensed distance matrix for {num_samples} samples. "
                f"Expected size {expected_condensed_size}, got {input_data_np.shape[0]}."
            )
    elif input_data_np.ndim == 2:
        if input_data_np.shape[0] != num_samples:
            raise ValueError(
                f"Input is 2D, but number of rows ({input_data_np.shape[0]}) "
                f"does not match number of labels ({num_samples}). Expected (N, D) embeddings or (N, N) distance matrix."
            )
        
        if manifold is not None: # Assumed to be embeddings (N, D)
            # Determine device for manifold computation, fallback to CPU
            # This assumes manifold.c exists and is a tensor, or manifold has parameters.
            # A more direct way might be to check manifold type or pass device explicitly.
            target_device = 'cpu'
            if hasattr(manifold, 'c') and isinstance(manifold.c, torch.Tensor):
                target_device = manifold.c.device
            elif hasattr(manifold, 'parameters'):
                try:
                    target_device = next(manifold.parameters()).device
                except StopIteration: # No parameters
                    pass
            
            if isinstance(embeddings_or_distances, torch.Tensor):
                embeddings_for_manifold = embeddings_or_distances.to(target_device)
            else: # Was numpy, convert to tensor for manifold computation
                embeddings_for_manifold = torch.tensor(input_data_np, dtype=torch.float32, device=target_device) # Assuming float32
            
            square_dist_matrix = compute_pairwise_distances(embeddings_for_manifold, manifold).detach().cpu().numpy()
            # Ensure symmetry, as numerical precision can sometimes lead to tiny differences
            if not np.allclose(square_dist_matrix, square_dist_matrix.T, atol=1e-7):
                 square_dist_matrix = (square_dist_matrix + square_dist_matrix.T) / 2.0
            condensed_distances = squareform(square_dist_matrix, checks=False) # checks=False as we symmetrized
        
        else: # No manifold: could be NxD embeddings (use Euclidean) or NxN distance matrix
            if input_data_np.shape[0] == input_data_np.shape[1]: # Square matrix (NxN), assume it's a distance matrix
                square_dist_matrix = input_data_np
                if not np.allclose(square_dist_matrix, square_dist_matrix.T, atol=1e-7):
                     square_dist_matrix = (square_dist_matrix + square_dist_matrix.T) / 2.0
                condensed_distances = squareform(square_dist_matrix, checks=False)
            else: # NxD embeddings, D!=N, compute Euclidean distances
                # Using sklearn for robust Euclidean distance calculation
                from sklearn.metrics.pairwise import euclidean_distances 
                square_dist_matrix = euclidean_distances(input_data_np)
                # euclidean_distances result is already symmetric
                condensed_distances = squareform(square_dist_matrix, checks=False)
    else:
        raise ValueError("embeddings_or_distances must be a 1D (condensed distance matrix) or 2D (embeddings or square distance matrix) array.")

    if condensed_distances is None:
        # This should ideally be caught by earlier checks.
        raise ValueError("Could not determine or compute a condensed distance matrix from the input.")

    # Perform hierarchical clustering using the condensed distance matrix
    Z = linkage(condensed_distances, method=linkage_method)

    # cluster_to_samples_map: maps cluster ID to the set of original sample indices it contains.
    # Leaf nodes (original samples) are IDs 0 to num_samples-1.
    # Internal nodes (formed by merges in Z) will get IDs from num_samples upwards.
    cluster_to_samples_map = {i: {i} for i in range(num_samples)}
    
    # pair_purity_contributions: stores the purity for each same-class pair
    # when they are first merged. Key: sorted tuple (original_sample_idx1, original_sample_idx2).
    pair_purity_contributions = {}
    
    # Iterate through each merge operation in the linkage matrix Z.
    # Each row m in Z represents a merge. Z has (num_samples - 1) rows.
    # Z[m, 0] and Z[m, 1] are the IDs of the clusters being merged.
    # The new cluster formed by merge m gets ID (num_samples + m).
    for m in range(Z.shape[0]):
        merged_cluster_id_1 = int(Z[m, 0])
        merged_cluster_id_2 = int(Z[m, 1])
        
        samples_in_cluster_1 = cluster_to_samples_map[merged_cluster_id_1]
        samples_in_cluster_2 = cluster_to_samples_map[merged_cluster_id_2]
        
        newly_merged_samples_set = samples_in_cluster_1.union(samples_in_cluster_2)
        new_cluster_id = num_samples + m # Assign a new ID to the merged cluster
        cluster_to_samples_map[new_cluster_id] = newly_merged_samples_set
        
        # Calculate purity of this newly formed cluster
        labels_of_samples_in_new_cluster = labels_np[list(newly_merged_samples_set)]
        
        purity_of_new_cluster = 0.0
        if labels_of_samples_in_new_cluster.size > 0:
            # np.unique also returns counts, which is efficient
            _unique_cluster_labels, counts_in_cluster = np.unique(labels_of_samples_in_new_cluster, return_counts=True)
            if counts_in_cluster.size > 0: # Ensure there are counts (i.e., cluster is not empty)
                purity_of_new_cluster = np.max(counts_in_cluster) / len(labels_of_samples_in_new_cluster)
            
        # Attribute this purity to same-class pairs brought together by THIS merge
        # These are pairs where one sample is from cluster_1 and the other from cluster_2
        for s1_idx in samples_in_cluster_1:
            for s2_idx in samples_in_cluster_2:
                if labels_np[s1_idx] == labels_np[s2_idx]:
                    # Ensure consistent key order for the pair
                    pair_key = tuple(sorted((s1_idx, s2_idx)))
                    # This pair is formed by this merge. Since we iterate merges
                    # from bottom up (smallest distance first), this is where they first meet.
                    # We only store the purity of the cluster at their first meeting point.
                    if pair_key not in pair_purity_contributions: # Should always be true with this iterative structure
                         pair_purity_contributions[pair_key] = purity_of_new_cluster
        
        # Optional: Clean up map entries for already merged clusters to save memory,
        # especially if num_samples is very large.
        del cluster_to_samples_map[merged_cluster_id_1]
        del cluster_to_samples_map[merged_cluster_id_2]

    # Calculate the total number of actual same-class pairs in the dataset
    total_actual_same_class_pairs = 0
    _unique_true_labels, true_label_counts = np.unique(labels_np, return_counts=True)
    for count in true_label_counts:
        # For a class with 'count' samples, there are count * (count - 1) / 2 pairs
        total_actual_same_class_pairs += count * (count - 1) // 2

    if total_actual_same_class_pairs == 0:
        # If there are no same-class pairs (e.g., all samples belong to unique classes,
        # or num_samples < 2 where no pairs can be formed), dendrogram purity is conventionally 1.0.
        return 1.0

    # Sum of purities for all unique same-class pairs, each considered once at their merge point.
    sum_of_contributed_purities = sum(pair_purity_contributions.values())
    
    # The number of entries in pair_purity_contributions should ideally equal total_actual_same_class_pairs
    # if the hierarchy is complete and all same-class pairs are eventually merged into some cluster.
    # If some same-class pairs are never in the same cluster (e.g. due to linkage issues or disconnected components),
    # they won't contribute to sum_of_contributed_purities. The definition of purity might vary in such cases.
    # This definition assumes they contribute 0 if not merged, which is implicitly handled.
    
    return sum_of_contributed_purities / total_actual_same_class_pairs



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