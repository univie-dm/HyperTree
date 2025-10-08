from logging import warning
from matplotlib import pyplot as plt
import torch
import wandb
import numpy as np
import numpy.typing as npt

from typing import Union, Optional
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.base import BaseEstimator, ClusterMixin
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csgraph, identity
from collections import Counter

from src.hyperbolic_math.src.manifolds import Hyperboloid, PoincareBall, Manifold
from src.hyperbolic_math.src.utils.helpers import get_delta
from src.hyperbolic_math.src.utils.vis_utils import create_figure, pointsTo2dPoincare
from src.hyperbolic_math.src.utils.horo_pca import compute_frechet_mean, center_data


def evaluation(X: torch.Tensor, Y: torch.Tensor, distance_matrix: torch.Tensor, 
               manifold: Manifold, metrics: dict[str, bool] | None = None, str_labels:Optional[npt.ArrayLike]|None = None) -> dict:
    """
    Evaluate the performance of a model using specified metrics.

    Parameters
    ----------
    X : list
        List of predicted values
    Y : list
        List of true values
    distance_matrix : torch.Tensor
        Precomputed distance matrix
    manifold : Manifold
        The manifold on which the embeddings are defined
    metrics : list
        List of metric functions to evaluate

    Returns
    -------
        dict: A dictionary with metric names as keys and their computed values.
    """
    results = {}
    points_in_manifold = False
    if metrics is None:
        return results
    

    distance_matrix += 2/manifold.c
    if distance_matrix.min() < 0:
        warning("Distance matrix contains negative values. Clipping to zero. Minimum: {}".format(distance_matrix.min()))
        distance_matrix = distance_matrix.clamp(min=0)

    if metrics["visualization"]:
        settings = {
            "plot_manifold_dtype": torch.float64,
            "dim_red_method": "HoroPCA",
            "title": "Hyperbolic Embeddings",
            "show_origin": True,
            "save_figure": False,
            "file_name": "hyperbolic_embeddings",
            "file_path": None,
            "file_format": "png",
        }
        sample_size = metrics['visualization_sample_size']
        sample_size = min(sample_size, len(X))

        pca_sample_indices = np.random.choice(len(X), sample_size, replace=False)
        points_sampled = X[pca_sample_indices]

        
        poincare_64 = PoincareBall(c=manifold.c.detach().to(settings['plot_manifold_dtype']), dtype=settings['plot_manifold_dtype'])
        if manifold.name == 'PoincareBall':
            points_sampled = poincare_64.to_hyperboloid(points_sampled)
        points_sampled = points_sampled.to(torch.float64)
        two_d_pts, _ = pointsTo2dPoincare(x=points_sampled, poincare=poincare_64, hyperplanes=None, settings=settings)
        two_d_pts = two_d_pts.detach().clone().to(dtype=settings['plot_manifold_dtype'], device=manifold.c.device)
        if not poincare_64.is_in_manifold(two_d_pts):
            warning("Some points are out of the Poincare ball after dimensionality reduction. NO visualization generated")
        else:
            points_in_manifold = True


    if metrics.get("delta_hyperbolicity", False):
        delta_hyp, diam_hyp, rel_delta_hyp = get_delta(X, manifold)
        results["delta_hyperbolicity"] = delta_hyp.item()
        results["diameter_hyperbolicity"] = diam_hyp.item()
        results["relative_delta_hyperbolicity"] = rel_delta_hyp.item()
    if metrics.get("dendrogram_purity_single", False):
        # Compute dendrogram purity for single linkage
        dp_score_single = dendrogram_purity(distance_matrix, Y, "single")
        results["dendrogram_purity_single"] = dp_score_single
    if metrics.get("dendrogram_purity_complete", False):
        # Compute dendrogram purity for complete linkage
        dp_score_complete = dendrogram_purity(distance_matrix, Y, "complete")
        results["dendrogram_purity_complete"] = dp_score_complete
    if metrics.get("dendrogram_purity_average", False):
        # Compute dendrogram purity for average linkage
        dp_score_average = dendrogram_purity(distance_matrix, Y, "average")
        results["dendrogram_purity_average"] = dp_score_average
    if metrics.get("dendrogram_purity_ward", False):
        # Compute dendrogram purity for ward linkage
        dp_score_ward = dendrogram_purity(distance_matrix, Y, "ward")
        results["dendrogram_purity_ward"] = dp_score_ward
    if metrics.get("agglomerative_clustering_single", False):
        # Compute agglomerative clustering metrics
        agglomerative_clustering_single = AgglomerativeClustering(
            n_clusters=len(np.unique(Y)),
            metric='precomputed',
            linkage='single'
        )
        pred_labels_single = agglomerative_clustering_single.fit_predict(distance_matrix.cpu().numpy())
        nmi_single = normalized_mutual_info_score(Y, pred_labels_single)
        ari_single = adjusted_rand_score(Y, pred_labels_single)
        results["agglomerative_clustering_single_nmi"] = nmi_single
        results["agglomerative_clustering_single_ari"] = ari_single
        if metrics['visualization'] and points_in_manifold:
            settings = {
                "plot_manifold_dtype": torch.float64,
                "dim_red_method": "HoroPCA",
                "title": "Hyperbolic Embeddings Agglomerative Clustering Single Linkage",
                "show_origin": True,
                "save_figure": False,
                "file_name": "hyperbolic_embeddings",
                "file_path": None,
                "file_format": "png",
            }
            # Map predicted labels to string labels if available
            Y_numpy = Y.cpu().numpy() if isinstance(Y, torch.Tensor) else np.asarray(Y)
            str_labels_subset = str_labels[pca_sample_indices] if str_labels is not None else None
            visualization_labels = map_predicted_labels_to_str_labels(
                pred_labels_single[pca_sample_indices], 
                Y_numpy[pca_sample_indices], 
                str_labels_subset
            )
            fig = create_figure(two_d_pts, poincare_64, labels=visualization_labels)
            results["agglomerative_clustering_single_hyperbolic_embeddings"] = wandb.Image(fig)
            plt.close(fig)
    if metrics.get("agglomerative_clustering_complete", False):
        # Compute agglomerative clustering metrics
        agglomerative_clustering_complete = AgglomerativeClustering(
            n_clusters=len(np.unique(Y)),
            metric='precomputed',
            linkage='complete'
        )
        pred_labels_complete = agglomerative_clustering_complete.fit_predict(distance_matrix.cpu().numpy())
        nmi_complete = normalized_mutual_info_score(Y, pred_labels_complete)
        ari_complete = adjusted_rand_score(Y, pred_labels_complete)
        results["agglomerative_clustering_complete_nmi"] = nmi_complete
        results["agglomerative_clustering_complete_ari"] = ari_complete
        if metrics['visualization'] and points_in_manifold:
            settings = {
                    "plot_manifold_dtype": torch.float64,
                    "dim_red_method": "HoroPCA",
                    "title": "Hyperbolic Embeddings Agglomerative Clustering Complete Linkage",
                    "show_origin": True,
                    "save_figure": False,
                    "file_name": "hyperbolic_embeddings",
                    "file_path": None,
                    "file_format": "png",
            }
            # Map predicted labels to string labels if available
            Y_numpy = Y.cpu().numpy() if isinstance(Y, torch.Tensor) else np.asarray(Y)
            str_labels_subset = str_labels[pca_sample_indices] if str_labels is not None else None
            visualization_labels = map_predicted_labels_to_str_labels(
                pred_labels_complete[pca_sample_indices], 
                Y_numpy[pca_sample_indices], 
                str_labels_subset
            )
            fig = create_figure(two_d_pts, poincare_64, labels=visualization_labels)
            results["agglomerative_clustering_complete_hyperbolic_embeddings"] = wandb.Image(fig)
            plt.close(fig)
    if metrics.get("agglomerative_clustering_average", False):
        # Compute agglomerative clustering metrics
        agglomerative_clustering_average = AgglomerativeClustering(
            n_clusters=len(np.unique(Y)),
            metric='precomputed',
            linkage='average'
        )
        pred_labels_average = agglomerative_clustering_average.fit_predict(distance_matrix.cpu().numpy())
        nmi_average = normalized_mutual_info_score(Y, pred_labels_average)
        ari_average = adjusted_rand_score(Y, pred_labels_average)
        results["agglomerative_clustering_average_nmi"] = nmi_average
        results["agglomerative_clustering_average_ari"] = ari_average
        if metrics['visualization'] and points_in_manifold:
            settings = {
                "plot_manifold_dtype": torch.float64,
                "dim_red_method": "HoroPCA",
                "title": "Hyperbolic Embeddings Agglomerative Clustering Average Linkage",
                "show_origin": True,
                "save_figure": False,
                "file_name": "hyperbolic_embeddings",
                "file_path": None,
                "file_format": "png",
            }
            # Map predicted labels to string labels if available
            Y_numpy = Y.cpu().numpy() if isinstance(Y, torch.Tensor) else np.asarray(Y)
            str_labels_subset = str_labels[pca_sample_indices] if str_labels is not None else None
            visualization_labels = map_predicted_labels_to_str_labels(
                pred_labels_average[pca_sample_indices], 
                Y_numpy[pca_sample_indices], 
                str_labels_subset
            )
            fig = create_figure(two_d_pts, poincare_64, labels=visualization_labels)
            results["agglomerative_clustering_average_hyperbolic_embeddings"] = wandb.Image(fig)
            plt.close(fig)
    if metrics.get("agglomerative_clustering_ward", False):
        # Compute agglomerative clustering metrics
        #TODO: Implement agglomerative clustering with ward linkage. Sklearn does not support precomputed distance matrix with ward linkage.
        pass
    if metrics.get("visualization", False) and points_in_manifold:
        # Create a figure with the points and labels
        settings = {
            "plot_manifold_dtype": torch.float64,
            "dim_red_method": "HoroPCA",
            "title": "Hyperbolic Embeddings",
            "show_origin": True,
            "save_figure": False,
            "file_name": "hyperbolic_embeddings",
            "file_path": None,
            "file_format": "png",
        }
        Y_numpy = Y.cpu().numpy() if isinstance(Y, torch.Tensor) else np.asarray(Y)
        if str_labels is not None:
            str_labels_array = np.asarray(str_labels)
            labels = str_labels_array[pca_sample_indices]
            # Convert labels to strings to handle cases where they are tuples/arrays (e.g., GTSRB)
            # Also handle None values by converting them to "Unknown"
            labels = np.array([str(label) if isinstance(label, (tuple, list, np.ndarray)) else (label if label is not None else "Unknown") for label in labels])
        else:
            labels = Y_numpy[pca_sample_indices]
        fig = create_figure(two_d_pts, poincare_64, labels=labels)
        results["hyperbolic_embeddings"] = wandb.Image(fig)
        plt.close(fig)

    if metrics.get("nearest_neighbors_accuracy", False):
        # Compute nearest neighbors accuracy
        nn_accuracy = nearest_neighbors_accuracy(distance_matrix, Y, k=1)
        results["1_nearest_neighbors_accuracy"] = nn_accuracy
        nn_accuracy = nearest_neighbors_accuracy(distance_matrix, Y, k=5)
        results["5_nearest_neighbors_accuracy"] = nn_accuracy
        nn_accuracy = nearest_neighbors_accuracy(distance_matrix, Y, k=10)
        results["10_nearest_neighbors_accuracy"] = nn_accuracy
        nn_accuracy = nearest_neighbors_accuracy(distance_matrix, Y, k=100)
        results["100_nearest_neighbors_accuracy"] = nn_accuracy
        nn_accuracy = nearest_neighbors_accuracy(distance_matrix, Y, k=1000)
        results["1000_nearest_neighbors_accuracy"] = nn_accuracy

    if metrics.get("spectral_clustering", False):
        affinity_matrix = distance_matrix.cpu().numpy()
        spectral_clustering = SpectralClustering(
            n_clusters=len(np.unique(Y)),
            affinity='precomputed_nearest_neighbors',
            n_neighbors=metrics.get('spectral_nn', 10),
            assign_labels='kmeans',
            random_state=int(torch.get_rng_state()[0].item())
        )

        #spectral_clustering = HyperbolicSpectralClustering(n_clusters=len(np.unique(Y)))
        pred_labels_spectral = spectral_clustering.fit_predict(affinity_matrix)
        nmi_spectral = normalized_mutual_info_score(Y, pred_labels_spectral)
        ari_spectral = adjusted_rand_score(Y, pred_labels_spectral)
        results["spectral_clustering_nmi"] = nmi_spectral
        results["spectral_clustering_ari"] = ari_spectral
        if metrics['visualization'] and points_in_manifold:
            settings = {
                "plot_manifold_dtype": torch.float64,
                "dim_red_method": "HoroPCA",
                "title": "Hyperbolic Embeddings Spectral Clustering",
                "show_origin": True,
                "save_figure": False,
                "file_name": "hyperbolic_embeddings",
                "file_path": None,
                "file_format": "png",
            }
            # Map predicted labels to string labels if available
            Y_numpy = Y.cpu().numpy() if isinstance(Y, torch.Tensor) else np.asarray(Y)
            str_labels_subset = str_labels[pca_sample_indices] if str_labels is not None else None
            visualization_labels = map_predicted_labels_to_str_labels(
                pred_labels_spectral[pca_sample_indices], 
                Y_numpy[pca_sample_indices], 
                str_labels_subset
            )
            fig = create_figure(two_d_pts, poincare_64, labels=visualization_labels)
            results["spectral_clustering_hyperbolic_embeddings"] = wandb.Image(fig)
            plt.close(fig)
    if metrics.get("kmeans", False):
        # Compute k-means clustering metrics
        kmeans = KMeans(n_clusters=len(np.unique(Y)), n_init='auto', random_state=int(torch.get_rng_state()[0].item()))
        pred_labels_kmeans = kmeans.fit_predict(X.cpu().numpy())
        nmi_kmeans = normalized_mutual_info_score(Y, pred_labels_kmeans)
        ari_kmeans = adjusted_rand_score(Y, pred_labels_kmeans)
        results["kmeans_nmi"] = nmi_kmeans
        results["kmeans_ari"] = ari_kmeans
        if metrics['visualization'] and points_in_manifold:
            settings = {
                "plot_manifold_dtype": torch.float64,
                "dim_red_method": "HoroPCA",
                "title": "Hyperbolic Embeddings KMeans Clustering",
                "show_origin": True,
                "save_figure": False,
                "file_name": "hyperbolic_embeddings",
                "file_path": None,
                "file_format": "png",
            }
            # Map predicted labels to string labels if available
            Y_numpy = Y.cpu().numpy() if isinstance(Y, torch.Tensor) else np.asarray(Y)
            str_labels_subset = str_labels[pca_sample_indices] if str_labels is not None else None
            visualization_labels = map_predicted_labels_to_str_labels(
                pred_labels_kmeans[pca_sample_indices], 
                Y_numpy[pca_sample_indices], 
                str_labels_subset
            )
            fig = create_figure(two_d_pts, poincare_64, labels=visualization_labels)
            results["kmeans_hyperbolic_embeddings"] = wandb.Image(fig)
            plt.close(fig)

    if metrics.get("dasgupta_cost", False):
        # Compute Dasgupta cost for average linkage
        dasgupta_cost_average = dasgupta_cost(torch.reciprocal(distance_matrix+1), "average")
        results["dasgupta_cost"] = dasgupta_cost_average
    return results
class Node:
    def __init__(self, id, left=None, right=None):
        self.id = id
        self.left = left
        self.right = right
        self._leaves_count = None

    def get_leaves_count(self):
        # Memoize the result to avoid re-computation
        if self._leaves_count is not None:
            return self._leaves_count
        
        if self.left is None and self.right is None: # It's a leaf node
            self._leaves_count = 1
        else:
            self._leaves_count = self.left.get_leaves_count() + self.right.get_leaves_count()
        return self._leaves_count

def dasgupta_cost(
    similarity_matrix: torch.Tensor,
    linkage_method: str = 'average'
) -> float:
    """
    Computes the Dasgupta cost correctly for a hierarchy built from a similarity matrix.
    """
    num_samples = similarity_matrix.shape[0]
    if num_samples < 2:
        return 0.0

    # 1. Build the linkage matrix (same as before)
    distance_matrix = 1 - similarity_matrix
    condensed_distances = squareform(distance_matrix.cpu().numpy(), checks=False)
    Z = linkage(condensed_distances, method=linkage_method)

    # 2. Build a user-friendly tree structure from the linkage matrix
    nodes = [Node(i) for i in range(num_samples)]
    for i, merge in enumerate(Z):
        node1_id, node2_id = int(merge[0]), int(merge[1])
        new_node = Node(id=num_samples + i, left=nodes[node1_id], right=nodes[node2_id])
        nodes.append(new_node)
    root = nodes[-1]
    
    # Create a map from leaf_id to its parent nodes for efficient LCA finding
    parent_map = {root: None}
    q = [root]
    while q:
        node = q.pop(0)
        if node.left:
            parent_map[node.left] = node
            q.append(node.left)
        if node.right:
            parent_map[node.right] = node
            q.append(node.right)

    # 3. Iterate over all unique pairs of leaves
    total_cost = 0.0
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            # 4. Find the LCA for the pair (i, j)
            # Get ancestors of i
            ancestors_i = set()
            curr = nodes[i]
            while curr is not None:
                ancestors_i.add(curr)
                curr = parent_map.get(curr)

            # Find the first common ancestor by traversing up from j
            lca_node = None
            curr = nodes[j]
            while curr is not None:
                if curr in ancestors_i:
                    lca_node = curr
                    break
                curr = parent_map.get(curr)
            
            # 5. Get subtree size and calculate cost contribution
            if lca_node:
                num_lca_leaves = lca_node.get_leaves_count()
                similarity_ij = similarity_matrix[i, j].item()
                total_cost += similarity_ij * num_lca_leaves

    return total_cost

def map_predicted_labels_to_str_labels(predicted_labels: np.ndarray, true_labels: np.ndarray, str_labels: Optional[npt.ArrayLike] = None) -> np.ndarray:
    """
    Map predicted cluster labels to string labels using majority voting.
    
    Parameters
    ----------
    predicted_labels : np.ndarray
        Predicted cluster labels (integers)
    true_labels : np.ndarray
        True labels (integers)
    str_labels : Optional[npt.ArrayLike]
        String labels corresponding to true_labels
        
    Returns
    -------
    np.ndarray
        Mapped string labels for predicted clusters, or original predicted labels if str_labels is None
    """
    if str_labels is None:
        return predicted_labels
    
    # Ensure inputs are numpy arrays
    str_labels = np.asarray(str_labels)
    predicted_labels = np.asarray(predicted_labels)
    true_labels = np.asarray(true_labels)
    
    unique_clusters = np.unique(predicted_labels)
    mapped_labels = np.copy(predicted_labels).astype(object)
    
    for cluster_id in unique_clusters:
        # Get indices of points in this cluster
        cluster_mask = predicted_labels == cluster_id
        cluster_true_labels = true_labels[cluster_mask]
        cluster_str_labels = str_labels[cluster_mask]
        
        # Find the most common string label in this cluster via majority vote on true labels
        true_label_counts = Counter(cluster_true_labels)
        most_common_true_label = true_label_counts.most_common(1)[0][0]
        
        # Find the corresponding string label
        true_label_mask = true_labels == most_common_true_label
        representative_str_label = str_labels[true_label_mask][0]
        
        # Handle case where str_labels contains tuples (e.g., GTSRB with fine/coarse labels)
        # Note: For GTSRB, we now extract only coarse labels in abstract.py, so this should be strings
        if isinstance(representative_str_label, (tuple, list, np.ndarray)):
            # If still tuple/array, convert to string representation
            representative_str_label = str(representative_str_label)
        
        # Assign this string label to all points in the cluster
        mapped_labels[cluster_mask] = representative_str_label
    
    return mapped_labels

def nearest_neighbors_accuracy(
    distance_matrix: Union[np.ndarray, torch.Tensor],
    labels: Union[np.ndarray, torch.Tensor],
    k: int = 1
) -> float:
    """
    Computes the nearest neighbors accuracy based on the distance matrix and labels.
    
    Parameters
    ----------
    distance_matrix : np.ndarray or torch.Tensor
        Precomputed distance matrix.
    labels : np.ndarray or torch.Tensor
        True labels for the samples.
    k : int, optional
        Number of nearest neighbors to consider, by default 1.

    Returns
    -------
    float
        The accuracy of the nearest neighbors classification.
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    if isinstance(distance_matrix, torch.Tensor):
        distance_matrix = distance_matrix.detach().cpu().numpy()

    num_samples = len(labels)
    if num_samples < 2:
        return 1.0

    # Get indices of the k+1 nearest neighbors for each sample (including itself)
    # Then exclude the first one (which is the sample itself)
    nearest_indices = np.argsort(distance_matrix, axis=1)[:, 1:k+1]
    average_accuracy = 0.0
    for i in range(distance_matrix.shape[0]):
        counter = 0
        # Check if the nearest neighbors are correct
        for j in range(k):
            if labels[nearest_indices[i, j]] == labels[i]:
                counter += 1
        average_accuracy += counter / k
    return average_accuracy / num_samples


def dendrogram_purity(
    distance_matrix: torch.Tensor,
    labels: np.ndarray,
    linkage_method: str = 'average'
) -> float:
    """
    Computes the canonical Dendrogram Purity for a hierarchy built from a distance matrix.

    This function first builds a dendrogram using the specified linkage method and then
    calculates its purity against the ground-truth labels, adhering to the formula:
    DP(T) = (1 / |P*|) * Σ_{k} Σ_{xi, xj ∈ C*_k, i≠j} pur(lca(xi, xj), C*_k)

    Args:
        distance_matrix (np.ndarray): A square N x N matrix of pairwise distances.
        labels (np.ndarray): A 1D array of size N with ground-truth class labels.
        linkage_method (str): The linkage algorithm to use (e.g., 'average', 'single', 'complete').

    Returns:
        float: The Dendrogram Purity score, between 0 and 1.
    """
    num_samples = len(labels)
    if num_samples < 2:
        return 1.0

    # Ensure numpy arrays
    distance_matrix_np = distance_matrix.cpu().numpy()
    labels_np = np.asarray(labels)

    if distance_matrix_np.shape != (num_samples, num_samples):
        raise ValueError("Distance matrix must be square and match number of labels.")

    # --- Step 1: Calculate the Normalization Factor |P*| ---
    # This is the total number of unique, non-identical pairs of points
    # that belong to the same ground-truth class.
    _, class_counts = np.unique(labels_np, return_counts=True)
    num_same_class_pairs = sum(count * (count - 1) // 2 for count in class_counts)

    if num_same_class_pairs == 0:
        # By convention, if there are no same-class pairs, purity is perfect.
        return 1.0

    # --- Step 2: Build the Dendrogram (Linkage Matrix) ---
    # The metric evaluates a given tree. Here, we build it first.
    condensed_distances = squareform(distance_matrix_np, checks=False)
    Z = linkage(condensed_distances, method=linkage_method)

    # --- Step 3: Iterate Through Merges to Calculate Total Purity Sum ---
    total_purity_sum = 0.0

    # This map will store the label distribution (Counter) for each cluster node.
    # It starts with leaf nodes (the original data points).
    cluster_label_counts = {i: Counter({labels_np[i]: 1}) for i in range(num_samples)}

    # Iterate over each merge operation in the linkage matrix
    for i, merge in enumerate(Z):
        node1_id, node2_id = int(merge[0]), int(merge[1])
        new_node_id = num_samples + i

        counts1 = cluster_label_counts[node1_id]
        counts2 = cluster_label_counts[node2_id]

        # The new node's labels are the sum of its children's labels
        new_node_counts = counts1 + counts2
        cluster_label_counts[new_node_id] = new_node_counts
        
        # The current new_node_id is the LCA for any pair of points (x, y)
        # where x is from the subtree of node1 and y is from the subtree of node2.

        # Total points in the newly formed cluster (the LCA's leaves)
        lca_node_size = sum(new_node_counts.values())

        # Iterate through all ground-truth classes present in the merge
        all_involved_labels = set(counts1.keys()) | set(counts2.keys())
        
        for label in all_involved_labels:
            # Number of points with this label from each child cluster
            n1_k = counts1.get(label, 0)
            n2_k = counts2.get(label, 0)
            
            # The number of new same-class pairs whose LCA is this node
            num_new_pairs = n1_k * n2_k
            
            if num_new_pairs > 0:
                # Purity of the LCA node with respect to this specific class `label`
                # pur(lvs(lca), C*_k) = |lvs(lca) ∩ C*_k| / |lvs(lca)|
                count_of_label_in_lca = new_node_counts.get(label, 0)
                purity_of_lca_for_label = count_of_label_in_lca / lca_node_size
                
                # Add the contribution of all these new pairs to the total sum
                total_purity_sum += num_new_pairs * purity_of_lca_for_label

        # Free memory by deleting child node info that will no longer be used
        del cluster_label_counts[node1_id]
        del cluster_label_counts[node2_id]
    
    # --- Step 4: Final Normalization ---
    return total_purity_sum / num_same_class_pairs

class HyperbolicSpectralClustering(BaseEstimator, ClusterMixin):
    """
    A Spectral Clustering algorithm that operates on a pre-computed
    geodesic distance matrix from a hyperbolic space, with optimizations
    for performance and robustness, similar to sklearn's implementation.
    """
    def __init__(self, n_clusters=8, gamma=1.0, affinity='rbf', n_neighbors=10,
                 assign_labels='kmeans', n_init=20, random_state=None):
        """
        Initializes the HyperbolicSpectralClustering instance.

        Args:
            n_clusters (int): The number of clusters to form.
            gamma (float): Kernel coefficient for the RBF kernel.
            affinity (str): How to construct the affinity matrix.
                            'rbf': use the RBF kernel on the full distance matrix.
                            'nearest_neighbors': construct a sparse k-NN graph.
            n_neighbors (int): Number of neighbors for the k-NN graph.
            assign_labels (str): The strategy for assigning labels in the spectral space.
                                 'kmeans': use KMeans.
            n_init (int): Number of times KMeans will be run with different seeds.
            random_state (int or RandomState): Determines random number generation.
        """
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.affinity = affinity
        self.n_neighbors = n_neighbors
        self.assign_labels = assign_labels
        self.n_init = n_init
        self.random_state = random_state
        self.labels_ = None

    def fit(self, D, y=None):
        """
        Performs hyperbolic spectral clustering on the distance matrix.
        """
        self.fit_predict(D)
        return self

    def fit_predict(self, D, y=None):
        """
        Performs clustering and returns cluster labels.
        """
        if D.shape[0] != D.shape[1]:
            raise ValueError("Input D must be a square distance matrix.")
        n_samples = D.shape[0]

        # Step 1: Construct the affinity matrix
        if self.affinity == 'rbf':
            affinity_matrix = np.exp(-self.gamma * (D ** 2))
            np.fill_diagonal(affinity_matrix, 0)
        elif self.affinity == 'nearest_neighbors':
            # Construct a sparse k-NN graph
            connectivity = kneighbors_graph(D, n_neighbors=self.n_neighbors,
                                            mode='distance', metric='precomputed',
                                            include_self=False)
            # Make the graph symmetric
            affinity_matrix = 0.5 * (connectivity + connectivity.T)
            # Apply the RBF kernel to the sparse graph
            affinity_matrix.data = np.exp(-self.gamma * (affinity_matrix.data ** 2))
        else:
            raise ValueError(f"Unknown affinity type: {self.affinity}")

        # Step 2: Compute the Graph Laplacian (normalized)
        # csgraph.laplacian is optimized for both sparse and dense matrices
        laplacian, dd = csgraph.laplacian(affinity_matrix, normed=True, return_diag=True)

        # Step 3: Compute the eigenvectors of the Laplacian
        # We use a robust and efficient eigensolver (eigsh with a shift-invert mode)
        # to find the eigenvectors corresponding to the smallest eigenvalues.
        # Adding a small shift to the diagonal improves stability.
        laplacian_shifted = laplacian + 1e-10 * identity(laplacian.shape[0])
        _, eigenvectors = eigsh(
            laplacian_shifted,
            k=self.n_clusters,
            which='SM',  # Smallest Magnitude
            tol=1e-6,
            v0=np.ones(laplacian.shape[0]) # Initial vector for stability
        )


        # Step 4: Normalize the eigenvectors to form the spectral embedding
        norm = np.linalg.norm(eigenvectors, axis=1, keepdims=True)
        spectral_embedding = eigenvectors / (norm + 1e-10)

        # Step 5: Cluster the points in the new spectral space
        if self.assign_labels == 'kmeans':
            kmeans = KMeans(
                n_clusters=self.n_clusters,
                n_init=self.n_init,
                random_state=self.random_state,
            )
            self.labels_ = kmeans.fit_predict(spectral_embedding)
        else:
            raise ValueError(f"Unknown label assignment strategy: {self.assign_labels}")

        return self.labels_
