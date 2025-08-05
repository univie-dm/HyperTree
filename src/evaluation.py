from math import dist
from matplotlib import pyplot as plt
import torch
import wandb
import numpy as np

from typing import Union, Optional
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.hyperbolic_math.src.manifolds.poincare import PoincareBall
from src.hyperbolic_math.src.manifolds import Manifold
from src.hyperbolic_math.src.utils.helpers import get_delta
from src.hyperbolic_math.src.utils.vis_utils import create_figure, pointsTo2dPoincare

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
    distance_matrix: Union[np.ndarray, torch.Tensor],
    labels: Union[np.ndarray, torch.Tensor],
    linkage_method: str = 'average'
) -> float:
    """
    Computes dendrogram purity according to the specific formula provided in the paper.
    Gradient-based Hierarchical Clustering using Continuous Representations of Trees in Hyperbolic Space
    https://dl.acm.org/doi/pdf/10.1145/3292500.3330997
    Formula: DP(T) = (1 / |P*|) * Σ_{C* ∈ C*} Σ_{(xi, xj) ∈ C* x C*} pur(lca(xi, xj), C*)
    """
    if isinstance(labels, torch.Tensor):
        labels_np = labels.detach().cpu().numpy()
    else:
        labels_np = np.array(labels)

    num_samples = len(labels_np)
    if num_samples < 2:
        return 1.0

    if isinstance(distance_matrix, torch.Tensor):
        distance_matrix_np = distance_matrix.detach().cpu().numpy()
    else:
        distance_matrix_np = distance_matrix

    if distance_matrix_np.shape != (num_samples, num_samples):
        raise ValueError("Distance matrix must be square and match number of labels.")
    condensed_distances = squareform(distance_matrix_np, checks=False)
    Z = linkage(condensed_distances, method=linkage_method)

    cluster_to_samples_map = {i: {i} for i in range(num_samples)}
    pair_purity_contributions = {}

    for m in range(Z.shape[0]):
        merged_cluster_id_1 = int(Z[m, 0])
        merged_cluster_id_2 = int(Z[m, 1])

        samples_in_cluster_1 = cluster_to_samples_map[merged_cluster_id_1]
        samples_in_cluster_2 = cluster_to_samples_map[merged_cluster_id_2]

        newly_merged_samples_set = samples_in_cluster_1.union(samples_in_cluster_2)
        new_cluster_id = num_samples + m
        cluster_to_samples_map[new_cluster_id] = newly_merged_samples_set
                
        labels_of_samples_in_new_cluster = labels_np[list(newly_merged_samples_set)]
        node_size = len(labels_of_samples_in_new_cluster)

        for s1_idx in samples_in_cluster_1:
            for s2_idx in samples_in_cluster_2:
                if labels_np[s1_idx] == labels_np[s2_idx]:
                    pair_key = tuple(sorted((s1_idx, s2_idx)))
                    if pair_key not in pair_purity_contributions:
                        # This pair's ground truth label
                        pair_label = labels_np[s1_idx]
                        
                        # Count how many members of the new cluster share this specific label
                        count_of_pair_label_in_node = np.sum(labels_of_samples_in_new_cluster == pair_label)
                        
                        # Purity of the node w.r.t. the pair's true class
                        purity_for_this_pair = count_of_pair_label_in_node / node_size
                        
                        pair_purity_contributions[pair_key] = purity_for_this_pair

        del cluster_to_samples_map[merged_cluster_id_1]
        del cluster_to_samples_map[merged_cluster_id_2]

        # The sum of purities for non-self pairs (xi, xj) and (xj, xi).
        # Since pur(lca(xi,xj)) == pur(lca(xj,xi)), we multiply the sum by 2.
        sum_of_purities_non_self = 2 * sum(pair_purity_contributions.values())
        
        # The purity for self-pairs (xi, xi) is always 1. There are num_samples such pairs.
        sum_of_purities_self = num_samples

        total_purity_sum = sum_of_purities_non_self + sum_of_purities_self

        # The denominator |P*| is the sum of squares of class sizes.
        _unique_true_labels, true_label_counts = np.unique(labels_np, return_counts=True)
        total_num_ordered_pairs = np.sum(true_label_counts**2)

        if total_num_ordered_pairs == 0:
            return 1.0

        return float(total_purity_sum / total_num_ordered_pairs)

    return 0.0

def evaluation(X: torch.Tensor, Y: torch.Tensor, distance_matrix: torch.Tensor, manifold: Manifold, metrics: dict[str, bool] | None = None) -> dict:
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
    if metrics is None:
        return results
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
    if metrics.get("agglomerative_clustering_ward", False):
        # Compute agglomerative clustering metrics
        #TODO: Implement agglomerative clustering with ward linkage. Sklearn does not support precomputed distance matrix with ward linkage.
        pass
    if metrics.get("visualization", False):
        # Create embeddings plot
        default_settings = {
        "plot_manifold_dtype": torch.float64,
        "dim_red_method": "HoroPCA",
        "title": "Hyperbolic Embeddings",
        "show_origin": True,
        "save_figure": False,
        "file_name": "hyperbolic_embeddings",
        "file_path": None,
        "file_format": "png",
    }
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
        # Compute spectral clustering metrics
        spectral_clustering = SpectralClustering(
            n_clusters=len(np.unique(Y)),
            affinity='precomputed'
        )
        pred_labels_spectral = spectral_clustering.fit_predict(distance_matrix.cpu().numpy())
        nmi_spectral = normalized_mutual_info_score(Y, pred_labels_spectral)
        ari_spectral = adjusted_rand_score(Y, pred_labels_spectral)
        results["spectral_clustering_nmi"] = nmi_spectral
        results["spectral_clustering_ari"] = ari_spectral
        #fig = create_figure(points, manifold, labels=pred_labels_spectral)
        #results["spectral_hyperbolic_embeddings"] = wandb.Image(fig)
        #plt.close(fig)
    if metrics.get("kmeans", False):
        # Compute k-means clustering metrics
        kmeans = KMeans(n_clusters=len(np.unique(Y)))
        pred_labels_kmeans = kmeans.fit_predict(X.cpu().numpy())
        nmi_kmeans = normalized_mutual_info_score(Y, pred_labels_kmeans)
        ari_kmeans = adjusted_rand_score(Y, pred_labels_kmeans)
        results["kmeans_nmi"] = nmi_kmeans
        results["kmeans_ari"] = ari_kmeans
        #fig = create_figure(points, manifold, labels=pred_labels_kmeans)
        #results["kmeans_hyperbolic_embeddings"] = wandb.Image(fig)
        #plt.close(fig)
    return results
