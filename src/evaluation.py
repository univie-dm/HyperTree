import torch
import wandb
import numpy as np

from typing import Union, Optional
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.hyperbolic_math.src.manifolds import Manifold
from src.hyperbolic_math.src.utils.helpers import get_delta, compute_pairwise_distances
from src.hyperbolic_math.src.utils.vis_utils import create_figure


def dendrogram_purity(
    embeddings_or_distances: Union[np.ndarray, torch.Tensor],
    labels: Union[np.ndarray, torch.Tensor],
    manifold: Optional[Manifold] = None, 
    linkage_method: str = 'average'
) -> float:
    """
    Compute the dendrogram purity for hierarchical clustering evaluation.
    This version iterates through merges in the linkage matrix for efficiency.

    The purity is defined as the sum of purities of the clusters where each
    same-class pair first merges, divided by the total number of same-class pairs.

    Parameters
    ----------
    embeddings_or_distances : Union[np.ndarray, torch.Tensor]
        - Embeddings of shape (n_samples, embedding_dim)
        - OR a precomputed square distance matrix of shape (n_samples, n_samples)
        - OR a precomputed condensed distance matrix (1D array)
    labels : Union[np.ndarray, torch.Tensor]
        Ground truth labels of shape (n_samples,)
    manifold : Manifold (optional)
        The manifold object (e.g. Euclidean, Hyperboloid, PoincareBall)
        used for distance computation if embeddings are provided.
        If None, and embeddings_or_distances is 2D and square, it's assumed to be a
        Euclidean distance matrix. If embeddings are NxD (D!=N) and manifold is None,
        Euclidean distances will be computed.
    linkage_method : str
        The linkage method for hierarchical clustering (e.g. 'single', 'complete', 'average', 'ward')

    Returns
    -------
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

    if num_samples < 2: # Purity is typically 1.0 for less than 2 samples
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
                embeddings_for_manifold = torch.tensor(input_data_np, device=target_device)

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

def evaluation(X: torch.Tensor, Y: torch.Tensor, distance_matrix: torch.Tensor, manifold: Manifold, metrics: dict[str, bool] = None) -> dict:
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
        dp_score_single = dendrogram_purity(X, Y, manifold, "single")
        results["dendrogram_purity_single"] = dp_score_single.item()
    if metrics.get("dendrogram_purity_complete", False):
        # Compute dendrogram purity for complete linkage
        dp_score_complete = dendrogram_purity(X, Y, manifold, "complete")
        results["dendrogram_purity_complete"] = dp_score_complete.item()
    if metrics.get("dendrogram_purity_average", False):
        # Compute dendrogram purity for average linkage
        dp_score_average = dendrogram_purity(X, Y, manifold, "average")
        results["dendrogram_purity_average"] = dp_score_average.item()
    if metrics.get("dendrogram_purity_ward", False):
        # Compute dendrogram purity for ward linkage
        dp_score_ward = dendrogram_purity(X, Y, manifold, "ward")
        results["dendrogram_purity_ward"] = dp_score_ward.item()
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
        agglomerative_clustering_ward = AgglomerativeClustering(
            n_clusters=len(np.unique(Y)),
            metric='precomputed',
            linkage='ward'
        )
    if metrics.get("visualization", False):
        # Create embeddings plot
        fig = create_figure(X, manifold, labels=Y.cpu().numpy())
        results["hyperbolic_embeddings"] = wandb.Image(fig)
    return results
