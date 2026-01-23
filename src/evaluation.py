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
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    confusion_matrix,
)
from sklearn.base import BaseEstimator, ClusterMixin
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csgraph, identity
from collections import Counter

from src.hyperbolic_math.src.manifolds.euclidean import Euclidean
from src.hyperbolic_math.src.manifolds import PoincareBall, Manifold
from src.hyperbolic_math.src.utils.helpers import get_delta
from src.hyperbolic_math.src.utils.vis_utils import create_figure, pointsTo2dPoincare


def evaluation(
    X: torch.Tensor,
    Y: torch.Tensor,
    distance_matrix: torch.Tensor,
    manifold: Manifold,
    metrics: dict[str, bool] | None = None,
    str_labels: Optional[npt.ArrayLike] | None = None,
    Y_coarse: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> dict:
    """
    Evaluate the performance of a model using specified metrics.

    Parameters
    ----------
    X : torch.Tensor
        Embeddings of the data points
    Y : torch.Tensor
        Fine-grained labels (or single-level labels for flat datasets)
    distance_matrix : torch.Tensor
        Precomputed distance matrix
    manifold : Manifold
        The manifold on which the embeddings are defined
    metrics : dict[str, bool] | None
        Dictionary of metric names and whether to compute them
    str_labels : Optional[npt.ArrayLike] | None
        String labels for visualization
    Y_coarse : Optional[Union[np.ndarray, torch.Tensor]]
        Coarse-grained labels for hierarchical datasets (e.g., CIFAR-100, GTSRB)
        If provided, hierarchical dendrogram purity will be computed

    Returns
    -------
        dict: A dictionary with metric names as keys and their computed values.
    """
    results = {}
    points_in_manifold = False
    if metrics is None:
        return results

    def get_categorical_colors(labels):
        """Create a list of distinct colors for categorical labels."""
        unique_labels = np.unique(labels)
        n_labels = len(unique_labels)

        # Create color mapping
        if n_labels <= 10:
            cmap = plt.cm.get_cmap("tab10")
        elif n_labels <= 20:
            cmap = plt.cm.get_cmap("tab20")
        else:
            cmap = plt.cm.get_cmap("hsv", n_labels)

        # Map each unique label to a color index
        label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        colors = [cmap(label_to_idx[label] / max(n_labels - 1, 1)) for label in labels]
        return colors, label_to_idx, cmap

    def set_square_axes(ax, pts):
        """Set square axes with equal limits based on max range."""
        max_range = max(
            pts[:, 0].max() - pts[:, 0].min(), pts[:, 1].max() - pts[:, 1].min()
        )
        mid_x = (pts[:, 0].max() + pts[:, 0].min()) / 2
        mid_y = (pts[:, 1].max() + pts[:, 1].min()) / 2
        ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
        ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
        ax.set_aspect("equal", adjustable="box")

    def create_confusion_matrix_plot(
        y_true, y_pred, title="Confusion Matrix", str_labels=None
    ):
        """Create a confusion matrix visualization.

        Parameters
        ----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted cluster labels
        title : str
            Title for the plot
        str_labels : array-like, optional
            String labels for the classes

        Returns
        -------
        matplotlib.figure.Figure
            The confusion matrix figure
        """
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Normalize by row (true label)
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(cm_normalized, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        # Set labels
        if str_labels is not None:
            unique_true = np.unique(y_true)
            unique_pred = np.unique(y_pred)
            # Map true labels to string labels
            true_label_names = [
                str(str_labels[np.where(y_true == label)[0][0]])
                for label in unique_true
            ]
            predicted_label_names = [
                str(str_labels[np.where(y_true == label)[0][0]])
                for label in unique_pred
            ]
            tick_marks_true = np.arange(len(unique_true))
            tick_marks_pred = np.arange(len(unique_pred))
            ax.set_yticks(tick_marks_true)
            ax.set_yticklabels(true_label_names, fontsize=8)
            ax.set_xticks(tick_marks_pred)
            ax.set_xticklabels(predicted_label_names, fontsize=8)
        else:
            ax.set_yticks(np.arange(cm.shape[0]))
            ax.set_yticklabels(np.unique(y_true), fontsize=8)
            ax.set_xticks(np.arange(cm.shape[1]))
            ax.set_xticklabels(np.unique(y_pred), fontsize=8)

        ax.set_ylabel("True Label", fontsize=10)
        ax.set_xlabel("Predicted Cluster", fontsize=10)
        ax.set_title(title, fontsize=12)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.tight_layout()

        return fig

    if metrics.get("test_sample_size", 0) > 0 and metrics["test_sample_size"] < len(X):
        # Sample equal number of points from each class
        sample_indices = stratified_sample(Y, metrics["test_sample_size"])
        X = X[sample_indices]
        Y = Y[sample_indices]
        distance_matrix = distance_matrix[sample_indices][:, sample_indices]
        if str_labels is not None:
            str_labels = np.asarray(str_labels)[sample_indices]
        if Y_coarse is not None:
            Y_coarse_np = (
                Y_coarse.cpu().numpy()
                if isinstance(Y_coarse, torch.Tensor)
                else np.asarray(Y_coarse)
            )
            Y_coarse = Y_coarse_np[sample_indices]

    if not isinstance(manifold.c, torch.Tensor):
        manifold.c = torch.tensor(manifold.c, device=X.device, dtype=X.dtype)

    # Check for collapsed embeddings (all embeddings are very similar)
    X_np = X.cpu().numpy()
    embedding_std = np.std(X_np, axis=0).mean()
    embedding_mean_norm = np.linalg.norm(X_np, axis=1).mean()
    if embedding_std < 1e-6:
        warning(
            f"Embeddings appear collapsed (std={embedding_std:.2e}, mean_norm={embedding_mean_norm:.2e}). Clustering metrics may be unreliable."
        )

    if distance_matrix.min() < 0:
        warning(
            "Distance matrix contains negative values. Clipping to zero. Minimum: {}".format(
                distance_matrix.min()
            )
        )
        distance_matrix = distance_matrix.clamp(min=0)

    # Compute visualization: HoroPCA for hyperbolic, standard PCA for Euclidean
    if metrics["visualization"]:
        sample_size = metrics["visualization_sample_size"]
        sample_size = min(sample_size, len(X))

        # Sample equal number of points from each class for visualization
        pca_sample_indices = stratified_sample(Y, sample_size)
        points_sampled = X[pca_sample_indices]

        if isinstance(manifold, Euclidean):
            # Use standard sklearn PCA for Euclidean manifolds
            from sklearn.decomposition import PCA as SklearnPCA

            pca = SklearnPCA(n_components=2)
            points_sampled_np = points_sampled.detach().cpu().numpy()
            two_d_pts = pca.fit_transform(points_sampled_np)
            two_d_pts = torch.tensor(two_d_pts, dtype=X.dtype, device=X.device)
            points_in_manifold = True
            colormap = "tab10"
            if len(np.unique(Y)) > 10:
                colormap = "tab20"
            if len(np.unique(Y)) > 20:
                colormap = "hsv"
        else:
            # Use HoroPCA for non-Euclidean manifolds
            settings = {
                "plot_manifold_dtype": manifold.dtype,
                "dim_red_method": "HoroPCA",
                "title": "Hyperbolic Embeddings",
                "show_origin": True,
                "save_figure": False,
                "file_name": "hyperbolic_embeddings",
                "file_path": None,
                "file_format": "png",
            }
            poincare_64 = PoincareBall(
                c=manifold.c.detach().to(settings["plot_manifold_dtype"]),
                dtype=settings["plot_manifold_dtype"],
            )
            if manifold.name == "PoincareBall":
                points_sampled = poincare_64.to_hyperboloid(points_sampled)
            points_sampled = points_sampled.to(settings["plot_manifold_dtype"])
            two_d_pts, _ = pointsTo2dPoincare(
                x=points_sampled,
                poincare=poincare_64,
                hyperplanes=None,
                settings=settings,
            )
            two_d_pts = (
                two_d_pts.detach()
                .clone()
                .to(dtype=settings["plot_manifold_dtype"], device=manifold.c.device)
            )  # type: ignore
            if not poincare_64.is_in_manifold(two_d_pts):
                warning(
                    "Some points are out of the Poincare ball after dimensionality reduction. NO visualization generated"
                )
            else:
                points_in_manifold = True

    if metrics.get("delta_hyperbolicity", False):
        delta_hyp, diam_hyp, rel_delta_hyp = get_delta(
            X, manifold, len(X), version="average", batch_size=100
        )
        results["delta_hyperbolicity"] = delta_hyp.item()
        results["diameter_hyperbolicity"] = diam_hyp.item()
        results["relative_delta_hyperbolicity"] = rel_delta_hyp.item()
    if metrics.get("dendrogram_purity_single", False):
        # Compute dendrogram purity for single linkage
        if Y_coarse is not None:
            # Use hierarchical DP for datasets with coarse labels
            dp_score_single = hierarchical_dendrogram_purity(
                distance_matrix, Y, Y_coarse, "single"
            )
            results["hierarchical_dendrogram_purity_single"] = dp_score_single
        else:
            # Use flat DP for simple datasets
            dp_score_single = dendrogram_purity(distance_matrix, Y, "single")
            results["dendrogram_purity_single"] = dp_score_single
    if metrics.get("dendrogram_purity_complete", False):
        # Compute dendrogram purity for complete linkage
        if Y_coarse is not None:
            # Use hierarchical DP for datasets with coarse labels
            dp_score_complete = hierarchical_dendrogram_purity(
                distance_matrix, Y, Y_coarse, "complete"
            )
            results["hierarchical_dendrogram_purity_complete"] = dp_score_complete
        else:
            # Use flat DP for simple datasets
            dp_score_complete = dendrogram_purity(distance_matrix, Y, "complete")
            results["dendrogram_purity_complete"] = dp_score_complete
    if metrics.get("dendrogram_purity_average", False):
        # Compute dendrogram purity for average linkage
        if Y_coarse is not None:
            # Use hierarchical DP for datasets with coarse labels
            dp_score_average = hierarchical_dendrogram_purity(
                distance_matrix, Y, Y_coarse, "average"
            )
            results["hierarchical_dendrogram_purity_average"] = dp_score_average
        else:
            # Use flat DP for simple datasets
            dp_score_average = dendrogram_purity(distance_matrix, Y, "average")
            results["dendrogram_purity_average"] = dp_score_average
    if metrics.get("dendrogram_purity_ward", False):
        # Compute dendrogram purity for ward linkage
        if Y_coarse is not None:
            # Use hierarchical DP for datasets with coarse labels
            dp_score_ward = hierarchical_dendrogram_purity(
                distance_matrix, Y, Y_coarse, "ward"
            )
            results["hierarchical_dendrogram_purity_ward"] = dp_score_ward
        else:
            # Use flat DP for simple datasets
            dp_score_ward = dendrogram_purity(distance_matrix, Y, "ward")
            results["dendrogram_purity_ward"] = dp_score_ward
    if metrics.get("agglomerative_clustering_single", False):
        # Compute agglomerative clustering metrics
        Y_numpy = Y.cpu().numpy() if isinstance(Y, torch.Tensor) else np.asarray(Y)
        agglomerative_clustering_single = AgglomerativeClustering(
            n_clusters=len(np.unique(Y_numpy)), metric="precomputed", linkage="single"
        )
        pred_labels_single = agglomerative_clustering_single.fit_predict(
            distance_matrix.cpu().numpy()
        )
        nmi_single = normalized_mutual_info_score(Y_numpy, pred_labels_single)
        ari_single = adjusted_rand_score(Y_numpy, pred_labels_single)
        results["agglomerative_clustering_single_nmi"] = nmi_single
        results["agglomerative_clustering_single_ari"] = ari_single

        # Generate confusion matrix (use coarse labels if available for visualization)
        Y_for_cm = Y_coarse if Y_coarse is not None else Y_numpy
        Y_for_cm_np = (
            Y_for_cm.cpu().numpy()
            if isinstance(Y_for_cm, torch.Tensor)
            else np.asarray(Y_for_cm)
        )
        # Map predicted labels to coarse labels for confusion matrix
        if Y_coarse is not None:
            Y_coarse_np = (
                Y_coarse.cpu().numpy()
                if isinstance(Y_coarse, torch.Tensor)
                else np.asarray(Y_coarse)
            )
            # For each predicted cluster, assign it to the most common coarse label
            pred_labels_for_cm = np.copy(pred_labels_single)
            for cluster_id in np.unique(pred_labels_single):
                cluster_mask = pred_labels_single == cluster_id
                cluster_coarse_labels = Y_coarse_np[cluster_mask]
                most_common = Counter(cluster_coarse_labels).most_common(1)[0][0]
                pred_labels_for_cm[cluster_mask] = most_common
        else:
            pred_labels_for_cm = pred_labels_single
        cm_fig = create_confusion_matrix_plot(
            Y_for_cm_np,
            pred_labels_for_cm,
            title="Agglomerative Clustering (Single Linkage) - Confusion Matrix",
            str_labels=str_labels,
        )
        results["agglomerative_clustering_single_confusion_matrix"] = wandb.Image(
            cm_fig
        )
        plt.close(cm_fig)

        if metrics["visualization"] and points_in_manifold:
            # Map predicted labels to string labels if available
            Y_numpy = Y.cpu().numpy() if isinstance(Y, torch.Tensor) else np.asarray(Y)
            str_labels_subset = (
                np.asarray(str_labels)[pca_sample_indices]
                if str_labels is not None
                else None
            )
            visualization_labels = map_predicted_labels_to_str_labels(
                pred_labels_single[pca_sample_indices],
                Y_numpy[pca_sample_indices],
                str_labels_subset,
            )

            if isinstance(manifold, Euclidean):
                # Create PCA scatter plot for Euclidean
                fig, ax = plt.subplots(figsize=(10, 10))
                two_d_pts_np = (
                    two_d_pts.cpu().numpy()
                    if isinstance(two_d_pts, torch.Tensor)
                    else two_d_pts
                )
                colors, label_to_idx, cmap = get_categorical_colors(
                    visualization_labels
                )
                scatter = ax.scatter(
                    two_d_pts_np[:, 0],
                    two_d_pts_np[:, 1],
                    c=colors,
                    s=10,
                    alpha=0.6,
                )
                # Add legend
                unique_labels = np.unique(visualization_labels)
                handles = [
                    plt.Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor=cmap(
                            label_to_idx[label] / max(len(unique_labels) - 1, 1)
                        ),
                        markersize=8,
                        label=str(label),
                    )
                    for label in unique_labels
                ]
                ax.legend(
                    handles=handles,
                    loc="center left",
                    bbox_to_anchor=(1, 0.5),
                    frameon=True,
                )
                set_square_axes(ax, two_d_pts_np)
                ax.set_title("PCA Agglomerative Clustering Single Linkage")
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                plt.tight_layout()
                results["agglomerative_clustering_single_pca_embeddings"] = wandb.Image(
                    fig
                )
                plt.close(fig)
            else:
                # Create HoroPCA figure for hyperbolic
                settings = {
                    "plot_manifold_dtype": manifold.dtype,
                    "dim_red_method": "HoroPCA",
                    "title": "Hyperbolic Embeddings Agglomerative Clustering Single Linkage",
                    "show_origin": True,
                    "save_figure": False,
                    "file_name": "hyperbolic_embeddings",
                    "file_path": None,
                    "file_format": "png",
                }
                fig = create_figure(two_d_pts, poincare_64, labels=visualization_labels)
                results["agglomerative_clustering_single_hyperbolic_embeddings"] = (
                    wandb.Image(fig)
                )
                plt.close(fig)
    if metrics.get("agglomerative_clustering_complete", False):
        # Compute agglomerative clustering metrics
        Y_numpy = Y.cpu().numpy() if isinstance(Y, torch.Tensor) else np.asarray(Y)
        agglomerative_clustering_complete = AgglomerativeClustering(
            n_clusters=len(np.unique(Y_numpy)), metric="precomputed", linkage="complete"
        )
        pred_labels_complete = agglomerative_clustering_complete.fit_predict(
            distance_matrix.cpu().numpy()
        )
        nmi_complete = normalized_mutual_info_score(Y_numpy, pred_labels_complete)
        ari_complete = adjusted_rand_score(Y_numpy, pred_labels_complete)
        results["agglomerative_clustering_complete_nmi"] = nmi_complete
        results["agglomerative_clustering_complete_ari"] = ari_complete

        # Generate confusion matrix (use coarse labels if available for visualization)
        Y_for_cm = Y_coarse if Y_coarse is not None else Y_numpy
        Y_for_cm_np = (
            Y_for_cm.cpu().numpy()
            if isinstance(Y_for_cm, torch.Tensor)
            else np.asarray(Y_for_cm)
        )
        # Map predicted labels to coarse labels for confusion matrix
        if Y_coarse is not None:
            Y_coarse_np = (
                Y_coarse.cpu().numpy()
                if isinstance(Y_coarse, torch.Tensor)
                else np.asarray(Y_coarse)
            )
            pred_labels_for_cm = np.copy(pred_labels_complete)
            for cluster_id in np.unique(pred_labels_complete):
                cluster_mask = pred_labels_complete == cluster_id
                cluster_coarse_labels = Y_coarse_np[cluster_mask]
                most_common = Counter(cluster_coarse_labels).most_common(1)[0][0]
                pred_labels_for_cm[cluster_mask] = most_common
        else:
            pred_labels_for_cm = pred_labels_complete
        cm_fig = create_confusion_matrix_plot(
            Y_for_cm_np,
            pred_labels_for_cm,
            title="Agglomerative Clustering (Complete Linkage) - Confusion Matrix",
            str_labels=str_labels,
        )
        results["agglomerative_clustering_complete_confusion_matrix"] = wandb.Image(
            cm_fig
        )
        plt.close(cm_fig)

        if metrics["visualization"] and points_in_manifold:
            # Map predicted labels to string labels if available
            Y_numpy = Y.cpu().numpy() if isinstance(Y, torch.Tensor) else np.asarray(Y)
            str_labels_subset = (
                np.asarray(str_labels)[pca_sample_indices]
                if str_labels is not None
                else None
            )
            visualization_labels = map_predicted_labels_to_str_labels(
                pred_labels_complete[pca_sample_indices],
                Y_numpy[pca_sample_indices],
                str_labels_subset,
            )

            if isinstance(manifold, Euclidean):
                # Create PCA scatter plot for Euclidean
                fig, ax = plt.subplots(figsize=(10, 10))
                two_d_pts_np = (
                    two_d_pts.cpu().numpy()
                    if isinstance(two_d_pts, torch.Tensor)
                    else two_d_pts
                )
                colors, label_to_idx, cmap = get_categorical_colors(
                    visualization_labels
                )
                scatter = ax.scatter(
                    two_d_pts_np[:, 0],
                    two_d_pts_np[:, 1],
                    c=colors,
                    s=10,
                    alpha=0.6,
                )
                # Add legend
                unique_labels = np.unique(visualization_labels)
                handles = [
                    plt.Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor=cmap(
                            label_to_idx[label] / max(len(unique_labels) - 1, 1)
                        ),
                        markersize=8,
                        label=str(label),
                    )
                    for label in unique_labels
                ]
                ax.legend(
                    handles=handles,
                    loc="center left",
                    bbox_to_anchor=(1, 0.5),
                    frameon=True,
                )
                set_square_axes(ax, two_d_pts_np)
                ax.set_title("PCA Agglomerative Clustering Complete Linkage")
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                plt.tight_layout()
                results["agglomerative_clustering_complete_pca_embeddings"] = (
                    wandb.Image(fig)
                )
                plt.close(fig)
            else:
                # Create HoroPCA figure for hyperbolic
                settings = {
                    "plot_manifold_dtype": manifold.dtype,
                    "dim_red_method": "HoroPCA",
                    "title": "Hyperbolic Embeddings Agglomerative Clustering Complete Linkage",
                    "show_origin": True,
                    "save_figure": False,
                    "file_name": "hyperbolic_embeddings",
                    "file_path": None,
                    "file_format": "png",
                }
                # Map predicted labels to string labels if available
                Y_numpy = (
                    Y.cpu().numpy() if isinstance(Y, torch.Tensor) else np.asarray(Y)
                )
                str_labels_subset = (
                    np.asarray(str_labels)[pca_sample_indices]
                    if str_labels is not None
                    else None
                )
                visualization_labels = map_predicted_labels_to_str_labels(
                    pred_labels_complete[pca_sample_indices],
                    Y_numpy[pca_sample_indices],
                    str_labels_subset,
                )
                fig = create_figure(two_d_pts, poincare_64, labels=visualization_labels)
                results["agglomerative_clustering_complete_hyperbolic_embeddings"] = (
                    wandb.Image(fig)
                )
                plt.close(fig)
    if metrics.get("agglomerative_clustering_average", False):
        # Compute agglomerative clustering metrics
        Y_numpy = Y.cpu().numpy() if isinstance(Y, torch.Tensor) else np.asarray(Y)
        agglomerative_clustering_average = AgglomerativeClustering(
            n_clusters=len(np.unique(Y_numpy)), metric="precomputed", linkage="average"
        )
        pred_labels_average = agglomerative_clustering_average.fit_predict(
            distance_matrix.cpu().numpy()
        )
        nmi_average = normalized_mutual_info_score(Y_numpy, pred_labels_average)
        ari_average = adjusted_rand_score(Y_numpy, pred_labels_average)
        results["agglomerative_clustering_average_nmi"] = nmi_average
        results["agglomerative_clustering_average_ari"] = ari_average

        # Generate confusion matrix (use coarse labels if available for visualization)
        Y_for_cm = Y_coarse if Y_coarse is not None else Y_numpy
        Y_for_cm_np = (
            Y_for_cm.cpu().numpy()
            if isinstance(Y_for_cm, torch.Tensor)
            else np.asarray(Y_for_cm)
        )
        # Map predicted labels to coarse labels for confusion matrix
        if Y_coarse is not None:
            Y_coarse_np = (
                Y_coarse.cpu().numpy()
                if isinstance(Y_coarse, torch.Tensor)
                else np.asarray(Y_coarse)
            )
            pred_labels_for_cm = np.copy(pred_labels_average)
            for cluster_id in np.unique(pred_labels_average):
                cluster_mask = pred_labels_average == cluster_id
                cluster_coarse_labels = Y_coarse_np[cluster_mask]
                most_common = Counter(cluster_coarse_labels).most_common(1)[0][0]
                pred_labels_for_cm[cluster_mask] = most_common
        else:
            pred_labels_for_cm = pred_labels_average
        cm_fig = create_confusion_matrix_plot(
            Y_for_cm_np,
            pred_labels_for_cm,
            title="Agglomerative Clustering (Average Linkage) - Confusion Matrix",
            str_labels=str_labels,
        )
        results["agglomerative_clustering_average_confusion_matrix"] = wandb.Image(
            cm_fig
        )
        plt.close(cm_fig)

        if metrics["visualization"] and points_in_manifold:
            # Map predicted labels to string labels if available
            Y_numpy = Y.cpu().numpy() if isinstance(Y, torch.Tensor) else np.asarray(Y)
            str_labels_subset = (
                np.asarray(str_labels)[pca_sample_indices]
                if str_labels is not None
                else None
            )
            visualization_labels = map_predicted_labels_to_str_labels(
                pred_labels_average[pca_sample_indices],
                Y_numpy[pca_sample_indices],
                str_labels_subset,
            )

            if isinstance(manifold, Euclidean):
                # Create PCA scatter plot for Euclidean
                fig, ax = plt.subplots(figsize=(10, 10))
                two_d_pts_np = (
                    two_d_pts.cpu().numpy()
                    if isinstance(two_d_pts, torch.Tensor)
                    else two_d_pts
                )
                colors, label_to_idx, cmap = get_categorical_colors(
                    visualization_labels
                )
                scatter = ax.scatter(
                    two_d_pts_np[:, 0],
                    two_d_pts_np[:, 1],
                    c=colors,
                    s=10,
                    alpha=0.6,
                )
                # Add legend
                unique_labels = np.unique(visualization_labels)
                handles = [
                    plt.Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor=cmap(
                            label_to_idx[label] / max(len(unique_labels) - 1, 1)
                        ),
                        markersize=8,
                        label=str(label),
                    )
                    for label in unique_labels
                ]
                ax.legend(
                    handles=handles,
                    loc="center left",
                    bbox_to_anchor=(1, 0.5),
                    frameon=True,
                )
                set_square_axes(ax, two_d_pts_np)
                ax.set_title("PCA Agglomerative Clustering Average Linkage")
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                plt.tight_layout()
                results["agglomerative_clustering_average_pca_embeddings"] = (
                    wandb.Image(fig)
                )
                plt.close(fig)
            else:
                # Create HoroPCA figure for hyperbolic
                settings = {
                    "plot_manifold_dtype": manifold.dtype,
                    "dim_red_method": "HoroPCA",
                    "title": "Hyperbolic Embeddings Agglomerative Clustering Average Linkage",
                    "show_origin": True,
                    "save_figure": False,
                    "file_name": "hyperbolic_embeddings",
                    "file_path": None,
                    "file_format": "png",
                }
                # Map predicted labels to string labels if available
                Y_numpy = (
                    Y.cpu().numpy() if isinstance(Y, torch.Tensor) else np.asarray(Y)
                )
                str_labels_subset = (
                    np.asarray(str_labels)[pca_sample_indices]
                    if str_labels is not None
                    else None
                )
                visualization_labels = map_predicted_labels_to_str_labels(
                    pred_labels_average[pca_sample_indices],
                    Y_numpy[pca_sample_indices],
                    str_labels_subset,
                )
                fig = create_figure(two_d_pts, poincare_64, labels=visualization_labels)
                results["agglomerative_clustering_average_hyperbolic_embeddings"] = (
                    wandb.Image(fig)
                )
                plt.close(fig)
    if metrics.get("agglomerative_clustering_ward", False):
        # Compute agglomerative clustering metrics
        # TODO: Implement agglomerative clustering with ward linkage. Sklearn does not support precomputed distance matrix with ward linkage.
        pass
    if metrics.get("visualization", False) and points_in_manifold:
        # Create a figure with the points and labels
        Y_numpy = Y.cpu().numpy() if isinstance(Y, torch.Tensor) else np.asarray(Y)
        if str_labels is not None:
            str_labels_array = np.asarray(str_labels)
            labels = str_labels_array[pca_sample_indices]
            # Convert labels to strings to handle cases where they are tuples/arrays (e.g., GTSRB)
            # Also handle None values by converting them to "Unknown"
            labels = np.array(
                [
                    str(label)
                    if isinstance(label, (tuple, list, np.ndarray))
                    else (label if label is not None else "Unknown")
                    for label in labels
                ]
            )
        else:
            labels = Y_numpy[pca_sample_indices]

        if isinstance(manifold, Euclidean):
            # Create PCA scatter plot for Euclidean
            fig, ax = plt.subplots(figsize=(10, 10))
            two_d_pts_np = (
                two_d_pts.cpu().numpy()
                if isinstance(two_d_pts, torch.Tensor)
                else two_d_pts
            )
            colors, label_to_idx, cmap = get_categorical_colors(labels)
            scatter = ax.scatter(
                two_d_pts_np[:, 0],
                two_d_pts_np[:, 1],
                c=colors,
                s=10,
                alpha=0.6,
            )
            # Add legend
            unique_labels = np.unique(labels)
            handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=cmap(
                        label_to_idx[label] / max(len(unique_labels) - 1, 1)
                    ),
                    markersize=8,
                    label=str(label),
                )
                for label in unique_labels
            ]
            ax.legend(
                handles=handles,
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                frameon=True,
            )
            set_square_axes(ax, two_d_pts_np)
            ax.set_title("PCA Embeddings (Euclidean)")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            plt.tight_layout()
            results["pca_embeddings"] = wandb.Image(fig)
            plt.close(fig)
        else:
            # Create HoroPCA figure for hyperbolic
            settings = {
                "plot_manifold_dtype": manifold.dtype,
                "dim_red_method": "HoroPCA",
                "title": "Hyperbolic Embeddings",
                "show_origin": True,
                "save_figure": False,
                "file_name": "hyperbolic_embeddings",
                "file_path": None,
                "file_format": "png",
            }
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
        Y_numpy = Y.cpu().numpy() if isinstance(Y, torch.Tensor) else np.asarray(Y)
        affinity_matrix = distance_matrix.cpu().numpy()
        spectral_clustering = SpectralClustering(
            n_clusters=len(np.unique(Y_numpy)),
            affinity="precomputed_nearest_neighbors",
            n_neighbors=metrics.get("spectral_nn", 10),
            assign_labels="kmeans",
            random_state=int(torch.get_rng_state()[0].item()),
        )

        # spectral_clustering = HyperbolicSpectralClustering(n_clusters=len(np.unique(Y_numpy)))
        pred_labels_spectral = spectral_clustering.fit_predict(affinity_matrix)
        nmi_spectral = normalized_mutual_info_score(Y_numpy, pred_labels_spectral)
        ari_spectral = adjusted_rand_score(Y_numpy, pred_labels_spectral)
        results["spectral_clustering_nmi"] = nmi_spectral
        results["spectral_clustering_ari"] = ari_spectral

        # Generate confusion matrix (use coarse labels if available for visualization)
        Y_for_cm = Y_coarse if Y_coarse is not None else Y_numpy
        Y_for_cm_np = (
            Y_for_cm.cpu().numpy()
            if isinstance(Y_for_cm, torch.Tensor)
            else np.asarray(Y_for_cm)
        )
        # Map predicted labels to coarse labels for confusion matrix
        if Y_coarse is not None:
            Y_coarse_np = (
                Y_coarse.cpu().numpy()
                if isinstance(Y_coarse, torch.Tensor)
                else np.asarray(Y_coarse)
            )
            pred_labels_for_cm = np.copy(pred_labels_spectral)
            for cluster_id in np.unique(pred_labels_spectral):
                cluster_mask = pred_labels_spectral == cluster_id
                cluster_coarse_labels = Y_coarse_np[cluster_mask]
                most_common = Counter(cluster_coarse_labels).most_common(1)[0][0]
                pred_labels_for_cm[cluster_mask] = most_common
        else:
            pred_labels_for_cm = pred_labels_spectral
        cm_fig = create_confusion_matrix_plot(
            Y_for_cm_np,
            pred_labels_for_cm,
            title="Spectral Clustering - Confusion Matrix",
            str_labels=str_labels,
        )
        results["spectral_clustering_confusion_matrix"] = wandb.Image(cm_fig)
        plt.close(cm_fig)

        if metrics["visualization"] and points_in_manifold:
            # Map predicted labels to string labels if available
            Y_numpy = Y.cpu().numpy() if isinstance(Y, torch.Tensor) else np.asarray(Y)
            str_labels_subset = (
                np.asarray(str_labels)[pca_sample_indices]
                if str_labels is not None
                else None
            )
            visualization_labels = map_predicted_labels_to_str_labels(
                pred_labels_spectral[pca_sample_indices],
                Y_numpy[pca_sample_indices],
                str_labels_subset,
            )

            if isinstance(manifold, Euclidean):
                # Create PCA scatter plot for Euclidean
                fig, ax = plt.subplots(figsize=(10, 10))
                two_d_pts_np = (
                    two_d_pts.cpu().numpy()
                    if isinstance(two_d_pts, torch.Tensor)
                    else two_d_pts
                )
                colors, label_to_idx, cmap = get_categorical_colors(
                    visualization_labels
                )
                scatter = ax.scatter(
                    two_d_pts_np[:, 0],
                    two_d_pts_np[:, 1],
                    c=colors,
                    s=10,
                    alpha=0.6,
                )
                # Add legend
                unique_labels = np.unique(visualization_labels)
                handles = [
                    plt.Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor=cmap(
                            label_to_idx[label] / max(len(unique_labels) - 1, 1)
                        ),
                        markersize=8,
                        label=str(label),
                    )
                    for label in unique_labels
                ]
                ax.legend(
                    handles=handles,
                    loc="center left",
                    bbox_to_anchor=(1, 0.5),
                    frameon=True,
                )
                set_square_axes(ax, two_d_pts_np)
                ax.set_title("PCA Spectral Clustering")
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                plt.tight_layout()
                results["spectral_clustering_pca_embeddings"] = wandb.Image(fig)
                plt.close(fig)
            else:
                # Create HoroPCA figure for hyperbolic
                settings = {
                    "plot_manifold_dtype": manifold.dtype,
                    "dim_red_method": "HoroPCA",
                    "title": "Hyperbolic Embeddings Spectral Clustering",
                    "show_origin": True,
                    "save_figure": False,
                    "file_name": "hyperbolic_embeddings",
                    "file_path": None,
                    "file_format": "png",
                }
                fig = create_figure(two_d_pts, poincare_64, labels=visualization_labels)
                results["spectral_clustering_hyperbolic_embeddings"] = wandb.Image(fig)
                plt.close(fig)
    if metrics.get("kmeans", False):
        # Compute k-means clustering metrics
        Y_numpy = Y.cpu().numpy() if isinstance(Y, torch.Tensor) else np.asarray(Y)
        kmeans = KMeans(
            n_clusters=len(np.unique(Y_numpy)),
            n_init="auto",
            random_state=int(torch.get_rng_state()[0].item()),
        )
        pred_labels_kmeans = kmeans.fit_predict(X.cpu().numpy())
        nmi_kmeans = normalized_mutual_info_score(Y_numpy, pred_labels_kmeans)
        ari_kmeans = adjusted_rand_score(Y_numpy, pred_labels_kmeans)
        results["kmeans_nmi"] = nmi_kmeans
        results["kmeans_ari"] = ari_kmeans

        # Generate confusion matrix (use coarse labels if available for visualization)
        Y_for_cm = Y_coarse if Y_coarse is not None else Y_numpy
        Y_for_cm_np = (
            Y_for_cm.cpu().numpy()
            if isinstance(Y_for_cm, torch.Tensor)
            else np.asarray(Y_for_cm)
        )
        # Map predicted labels to coarse labels for confusion matrix
        if Y_coarse is not None:
            Y_coarse_np = (
                Y_coarse.cpu().numpy()
                if isinstance(Y_coarse, torch.Tensor)
                else np.asarray(Y_coarse)
            )
            pred_labels_for_cm = np.copy(pred_labels_kmeans)
            for cluster_id in np.unique(pred_labels_kmeans):
                cluster_mask = pred_labels_kmeans == cluster_id
                cluster_coarse_labels = Y_coarse_np[cluster_mask]
                most_common = Counter(cluster_coarse_labels).most_common(1)[0][0]
                pred_labels_for_cm[cluster_mask] = most_common
        else:
            pred_labels_for_cm = pred_labels_kmeans
        cm_fig = create_confusion_matrix_plot(
            Y_for_cm_np,
            pred_labels_for_cm,
            title="K-Means Clustering - Confusion Matrix",
            str_labels=str_labels,
        )
        results["kmeans_confusion_matrix"] = wandb.Image(cm_fig)
        plt.close(cm_fig)

        if metrics["visualization"] and points_in_manifold:
            # Map predicted labels to string labels if available
            Y_numpy = Y.cpu().numpy() if isinstance(Y, torch.Tensor) else np.asarray(Y)
            str_labels_subset = (
                np.asarray(str_labels)[pca_sample_indices]
                if str_labels is not None
                else None
            )
            visualization_labels = map_predicted_labels_to_str_labels(
                pred_labels_kmeans[pca_sample_indices],
                Y_numpy[pca_sample_indices],
                str_labels_subset,
            )

            if isinstance(manifold, Euclidean):
                # Create PCA scatter plot for Euclidean
                fig, ax = plt.subplots(figsize=(10, 10))
                two_d_pts_np = (
                    two_d_pts.cpu().numpy()
                    if isinstance(two_d_pts, torch.Tensor)
                    else two_d_pts
                )
                colors, label_to_idx, cmap = get_categorical_colors(
                    visualization_labels
                )
                scatter = ax.scatter(
                    two_d_pts_np[:, 0],
                    two_d_pts_np[:, 1],
                    c=colors,
                    s=10,
                    alpha=0.6,
                )
                # Add legend
                unique_labels = np.unique(visualization_labels)
                handles = [
                    plt.Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor=cmap(
                            label_to_idx[label] / max(len(unique_labels) - 1, 1)
                        ),
                        markersize=8,
                        label=str(label),
                    )
                    for label in unique_labels
                ]
                ax.legend(
                    handles=handles,
                    loc="center left",
                    bbox_to_anchor=(1, 0.5),
                    frameon=True,
                )
                set_square_axes(ax, two_d_pts_np)
                ax.set_title("PCA KMeans Clustering")
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                plt.tight_layout()
                results["kmeans_pca_embeddings"] = wandb.Image(fig)
                plt.close(fig)
            else:
                # Create HoroPCA figure for hyperbolic
                settings = {
                    "plot_manifold_dtype": manifold.dtype,
                    "dim_red_method": "HoroPCA",
                    "title": "Hyperbolic Embeddings KMeans Clustering",
                    "show_origin": True,
                    "save_figure": False,
                    "file_name": "hyperbolic_embeddings",
                    "file_path": None,
                    "file_format": "png",
                }
                fig = create_figure(two_d_pts, poincare_64, labels=visualization_labels)
                results["kmeans_hyperbolic_embeddings"] = wandb.Image(fig)
                plt.close(fig)

    if metrics.get("dasgupta_cost", False):
        # Compute Dasgupta cost for average linkage
        dasgupta_cost_average = dasgupta_cost(
            torch.reciprocal(distance_matrix + 1), "average"
        )
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

        if self.left is None and self.right is None:  # It's a leaf node
            self._leaves_count = 1
        else:
            self._leaves_count = (
                self.left.get_leaves_count() + self.right.get_leaves_count()
            )  # type: ignore
        return self._leaves_count


def dasgupta_cost(
    similarity_matrix: torch.Tensor, linkage_method: str = "average"
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
            parent_map[node.left] = node  # type: ignore
            q.append(node.left)
        if node.right:
            parent_map[node.right] = node  # type: ignore
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


def map_predicted_labels_to_str_labels(
    predicted_labels: np.ndarray,
    true_labels: np.ndarray,
    str_labels: Optional[npt.ArrayLike] = None,
) -> np.ndarray:
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
    k: int = 1,
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
    nearest_indices = np.argsort(distance_matrix, axis=1)[:, 1 : k + 1]
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
    labels: np.ndarray | torch.Tensor,
    linkage_method: str = "average",
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


def hierarchical_dendrogram_purity(
    distance_matrix: torch.Tensor,
    labels_fine: np.ndarray | torch.Tensor,
    labels_coarse: np.ndarray | torch.Tensor,
    linkage_method: str = "average",
) -> float:
    """
    Computes the Hierarchical Dendrogram Purity for a hierarchy built from a distance matrix.

    This function evaluates the dendrogram against multiple levels of ground-truth labels,
    calculating the purity for both fine-grained and coarse-grained classes.

    This is the correct metric for datasets with hierarchical structure like:
    - CIFAR-100 (100 fine classes nested in 20 coarse classes)
    - GTSRB (43 fine traffic signs nested in coarse categories)

    The formula computes:
    DP_hierarchical(T) = (1 / |P*|) * Σ_{all clusters C*} Σ_{xi, xj ∈ C*, i≠j} pur(lca(xi, xj), C*)

    where P* includes pairs from both fine AND coarse ground-truth clusters.

    Args:
        distance_matrix (torch.Tensor): A square N x N matrix of pairwise distances.
        labels_fine (np.ndarray | torch.Tensor): A 1D array of size N with fine-grained class labels.
        labels_coarse (np.ndarray | torch.Tensor): A 1D array of size N with coarse-grained class labels.
        linkage_method (str): The linkage algorithm to use (e.g., 'average', 'single', 'complete').

    Returns:
        float: The Hierarchical Dendrogram Purity score, between 0 and 1.

    Example:
        For CIFAR-100:
        - labels_fine: 100 fine classes (e.g., 'maple_tree', 'oak_tree', 'palm_tree', ...)
        - labels_coarse: 20 coarse classes (e.g., 'trees', 'flowers', 'people', ...)

        This will evaluate against all 120 ground-truth clusters (100 fine + 20 coarse).
    """
    num_samples = len(labels_fine)
    if num_samples < 2:
        return 1.0

    # Ensure numpy arrays
    distance_matrix_np = distance_matrix.cpu().numpy()
    labels_fine_np = np.asarray(labels_fine)
    labels_coarse_np = np.asarray(labels_coarse)

    if distance_matrix_np.shape != (num_samples, num_samples):
        raise ValueError("Distance matrix must be square and match number of labels.")

    if len(labels_fine_np) != len(labels_coarse_np):
        raise ValueError("Fine and coarse labels must have the same length.")

    # --- Step 1: Calculate the Normalization Factor |P*| ---
    # For hierarchical DP, we count pairs from BOTH fine and coarse levels

    # Count pairs from fine-grained classes
    _, fine_class_counts = np.unique(labels_fine_np, return_counts=True)
    num_fine_pairs = sum(count * (count - 1) // 2 for count in fine_class_counts)

    # Count pairs from coarse-grained classes
    _, coarse_class_counts = np.unique(labels_coarse_np, return_counts=True)
    num_coarse_pairs = sum(count * (count - 1) // 2 for count in coarse_class_counts)

    # Total pairs is the sum from both levels
    num_same_class_pairs = num_fine_pairs + num_coarse_pairs

    if num_same_class_pairs == 0:
        return 1.0

    # --- Step 2: Build the Dendrogram (Linkage Matrix) ---
    condensed_distances = squareform(distance_matrix_np, checks=False)
    Z = linkage(condensed_distances, method=linkage_method)

    # --- Step 3: Calculate Purity Sum for Both Fine and Coarse Levels ---

    # We'll compute the purity sum separately for fine and coarse labels
    # and then add them together

    def compute_purity_sum_for_labels(labels_np):
        """Helper function to compute purity sum for a single label set."""
        total_purity_sum = 0.0

        # Map to track label distribution for each cluster node
        cluster_label_counts = {
            i: Counter({labels_np[i]: 1}) for i in range(num_samples)
        }

        # Iterate over each merge operation in the linkage matrix
        for i, merge in enumerate(Z):
            node1_id, node2_id = int(merge[0]), int(merge[1])
            new_node_id = num_samples + i

            counts1 = cluster_label_counts[node1_id]
            counts2 = cluster_label_counts[node2_id]

            # The new node's labels are the sum of its children's labels
            new_node_counts = counts1 + counts2
            cluster_label_counts[new_node_id] = new_node_counts

            # Total points in the newly formed cluster
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
                    # Purity of the LCA node with respect to this specific class
                    count_of_label_in_lca = new_node_counts.get(label, 0)
                    purity_of_lca_for_label = count_of_label_in_lca / lca_node_size

                    # Add the contribution of all these new pairs to the total sum
                    total_purity_sum += num_new_pairs * purity_of_lca_for_label

            # Free memory by deleting child node info that will no longer be used
            del cluster_label_counts[node1_id]
            del cluster_label_counts[node2_id]

        return total_purity_sum

    # Compute purity sum for fine-grained labels
    fine_purity_sum = compute_purity_sum_for_labels(labels_fine_np)

    # Compute purity sum for coarse-grained labels
    coarse_purity_sum = compute_purity_sum_for_labels(labels_coarse_np)

    # Total purity sum is the sum from both levels
    total_purity_sum = fine_purity_sum + coarse_purity_sum

    # --- Step 4: Final Normalization ---
    return total_purity_sum / num_same_class_pairs


class HyperbolicSpectralClustering(BaseEstimator, ClusterMixin):
    """
    A Spectral Clustering algorithm that operates on a pre-computed
    geodesic distance matrix from a hyperbolic space, with optimizations
    for performance and robustness, similar to sklearn's implementation.
    """

    def __init__(
        self,
        n_clusters=8,
        gamma=1.0,
        affinity="rbf",
        n_neighbors=10,
        assign_labels="kmeans",
        n_init=20,
        random_state=None,
    ):
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

        # Step 1: Construct the affinity matrix
        if self.affinity == "rbf":
            affinity_matrix = np.exp(-self.gamma * (D**2))
            np.fill_diagonal(affinity_matrix, 0)
        elif self.affinity == "nearest_neighbors":
            # Construct a sparse k-NN graph
            connectivity = kneighbors_graph(
                D,
                n_neighbors=self.n_neighbors,
                mode="distance",
                metric="precomputed",
                include_self=False,
            )
            # Make the graph symmetric
            affinity_matrix = 0.5 * (connectivity + connectivity.T)  # type: ignore
            # Apply the RBF kernel to the sparse graph
            affinity_matrix.data = np.exp(-self.gamma * (affinity_matrix.data**2))
        else:
            raise ValueError(f"Unknown affinity type: {self.affinity}")

        # Step 2: Compute the Graph Laplacian (normalized)
        # csgraph.laplacian is optimized for both sparse and dense matrices
        laplacian, dd = csgraph.laplacian(
            affinity_matrix, normed=True, return_diag=True
        )

        # Step 3: Compute the eigenvectors of the Laplacian
        # We use a robust and efficient eigensolver (eigsh with a shift-invert mode)
        # to find the eigenvectors corresponding to the smallest eigenvalues.
        # Adding a small shift to the diagonal improves stability.
        laplacian_shifted = laplacian + 1e-10 * identity(laplacian.shape[0])
        _, eigenvectors = eigsh(
            laplacian_shifted,
            k=self.n_clusters,
            which="SM",  # Smallest Magnitude
            tol=1e-6,  # type: ignore
            v0=np.ones(laplacian.shape[0]),  # Initial vector for stability
        )

        # Step 4: Normalize the eigenvectors to form the spectral embedding
        norm = np.linalg.norm(eigenvectors, axis=1, keepdims=True)
        spectral_embedding = eigenvectors / (norm + 1e-10)

        # Step 5: Cluster the points in the new spectral space
        if self.assign_labels == "kmeans":
            kmeans = KMeans(
                n_clusters=self.n_clusters,
                n_init=self.n_init,
                random_state=self.random_state,
            )
            self.labels_ = kmeans.fit_predict(spectral_embedding)
        else:
            raise ValueError(f"Unknown label assignment strategy: {self.assign_labels}")

        return self.labels_


def stratified_sample(
    labels: Union[np.ndarray, torch.Tensor], sample_size: int
) -> np.ndarray:
    """
    Sample equal number of points from each class (stratified sampling).

    Parameters
    ----------
    labels : np.ndarray or torch.Tensor
        Class labels for all points
    sample_size : int
        Total number of samples to draw

    Returns
    -------
    np.ndarray
        Indices of sampled points
    """
    if isinstance(labels, torch.Tensor):
        labels_np = labels.cpu().numpy()
    else:
        labels_np = np.asarray(labels)

    unique_classes = np.unique(labels_np)
    n_classes = len(unique_classes)

    # Calculate samples per class (try to distribute evenly)
    samples_per_class = sample_size // n_classes
    remaining_samples = sample_size % n_classes

    sampled_indices = []

    for i, cls in enumerate(unique_classes):
        # Get all indices for this class
        class_indices = np.where(labels_np == cls)[0]

        # Determine how many samples to take from this class
        n_samples_this_class = samples_per_class
        if i < remaining_samples:
            n_samples_this_class += 1

        # Make sure we don't try to sample more than available
        n_samples_this_class = min(n_samples_this_class, len(class_indices))

        # Sample from this class
        if n_samples_this_class > 0:
            sampled = np.random.choice(
                class_indices, n_samples_this_class, replace=False
            )
            sampled_indices.extend(sampled)

    return np.array(sampled_indices)
