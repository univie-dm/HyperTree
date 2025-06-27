from typing import List
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import torch
import wandb
from src.hyperbolic_math.src.utils.helpers import get_delta
from src.hyperbolic_math.src.utils.vis_utils import create_figure
from src.models.utils import dendrogram_purity
from src.hyperbolic_math.src.manifolds import Manifold

def evaluation(X: torch.Tensor, Y: torch.Tensor, distance_matrix: torch.Tensor, manifold: Manifold, metrics: dict[str, bool] = None) -> dict:
    """
    Evaluate the performance of a model using specified metrics.

    Parameters:
    X (list): List of predicted values.
    Y (list): List of true values.
    distance_matrix (torch.Tensor): Precomputed distance matrix.
    manifold (Manifold): The manifold on which the embeddings are defined.
    metrics (list): List of metric functions to evaluate.

    Returns:
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
        fig = create_figure(X, manifold, labels=Y)
        results["hyperbolic_embeddings"] = wandb.Image(fig)
    return results