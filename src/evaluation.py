import json
import os
from datetime import datetime
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
from sklearn.manifold import TSNE
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
from pathlib import Path

from src.hyperbolic_math.src.manifolds.euclidean import Euclidean
from src.hyperbolic_math.src.manifolds import PoincareBall, Hyperboloid, Manifold


# ---------------------------------------------------------------------------
# Results file I/O — plain JSON readable by an LLM
# ---------------------------------------------------------------------------

def _serializable(value):
    """Convert a metric value to a JSON-serializable type."""
    if isinstance(value, (int, float, str, bool, type(None))):
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, wandb.Image):
        return "<wandb.Image>"
    return str(value)


def write_results(
    metrics: dict,
    results_dir: str | Path,
    stage: str = "eval",
    epoch: int | None = None,
) -> Path:
    """
    Write evaluation metrics to a human/LLM-readable JSON file.

    Two files are maintained:
    - ``<results_dir>/<stage>_results_latest.json`` — overwritten each call
      (always reflects the most recent evaluation).
    - ``<results_dir>/<stage>_results_history.jsonl`` — one JSON object per
      line, appended each call (full history of intermediate results).

    Parameters
    ----------
    metrics : dict
        The metrics dict (may contain wandb.Image values — they are skipped).
    results_dir : str | Path
        Directory to write into (created if it does not exist).
    stage : str
        Label such as ``"pretrain"`` or ``"clustering"``.
    epoch : int | None
        Current epoch number (included in the record if provided).

    Returns
    -------
    Path
        Path to the ``_latest.json`` file.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "stage": stage,
    }
    if epoch is not None:
        record["epoch"] = epoch

    for key, value in sorted(metrics.items()):
        serialized = _serializable(value)
        if serialized == "<wandb.Image>":
            continue
        record[key] = serialized

    latest_path = results_dir / f"{stage}_results_latest.json"
    history_path = results_dir / f"{stage}_results_history.jsonl"

    with open(latest_path, "w") as f:
        json.dump(record, f, indent=2)
        f.write("\n")

    with open(history_path, "a") as f:
        json.dump(record, f)
        f.write("\n")

    return latest_path


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _get_categorical_colors(labels):
    """Create a list of distinct colors for categorical labels.

    The special label 'Other' (if present) is always colored gray.
    """
    unique_labels = np.unique(labels)
    # Exclude "Other" from the colormap so it doesn't consume a slot
    real_labels = [l for l in unique_labels if str(l) != "Other"]
    has_other = len(real_labels) < len(unique_labels)
    n_real = len(real_labels)

    if n_real <= 10:
        cmap = plt.cm.get_cmap("tab10")
    elif n_real <= 20:
        cmap = plt.cm.get_cmap("tab20")
    else:
        cmap = plt.cm.get_cmap("hsv", n_real)

    label_to_idx = {label: i for i, label in enumerate(real_labels)}
    if has_other:
        label_to_idx["Other"] = -1  # sentinel

    gray = (0.7, 0.7, 0.7, 1.0)
    colors = [
        gray if str(label) == "Other"
        else cmap(label_to_idx[label] / max(n_real - 1, 1))
        for label in labels
    ]
    return colors, label_to_idx, cmap


def _set_square_axes(ax, pts):
    """Set square axes with equal limits based on max range."""
    max_range = max(
        pts[:, 0].max() - pts[:, 0].min(), pts[:, 1].max() - pts[:, 1].min()
    )
    mid_x = (pts[:, 0].max() + pts[:, 0].min()) / 2
    mid_y = (pts[:, 1].max() + pts[:, 1].min()) / 2
    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    ax.set_aspect("equal", adjustable="box")


def _aggregate_labels_top_k(labels, max_classes=20):
    """Keep the top-k most frequent labels, map the rest to 'Other'."""
    labels = np.asarray(labels)
    counts = Counter(labels)
    top_k = [l for l, _ in counts.most_common(max_classes)]
    top_set = set(top_k)
    return np.array([l if l in top_set else "Other" for l in labels])


def _create_scatter_plot(two_d_pts, labels, title, max_legend_classes=20):
    """Create a 2D scatter plot with categorical coloring and legend."""
    two_d_pts_np = two_d_pts.cpu().numpy() if isinstance(two_d_pts, torch.Tensor) else np.asarray(two_d_pts)
    labels = np.asarray(labels)
    n_unique = len(np.unique(labels))

    # Aggregate to top-k classes if too many for readable visualization
    display_labels = _aggregate_labels_top_k(labels, max_legend_classes) if n_unique > max_legend_classes else labels

    colors, label_to_idx, cmap = _get_categorical_colors(display_labels)
    unique_display = np.unique(display_labels)
    n_display = len(unique_display)
    n_real = len([l for l in unique_display if str(l) != "Other"])

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(two_d_pts_np[:, 0], two_d_pts_np[:, 1], c=colors, s=10, alpha=0.6)

    # Build legend — "Other" drawn last in gray
    legend_labels = sorted([l for l in unique_display if str(l) != "Other"])
    if "Other" in unique_display:
        legend_labels.append("Other")
    handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=(
                (0.7, 0.7, 0.7, 1.0) if str(label) == "Other"
                else cmap(label_to_idx[label] / max(n_real - 1, 1))
            ),
            markersize=8, label=str(label),
        )
        for label in legend_labels
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1, 0.5), frameon=True,
              fontsize=8 if n_display <= 25 else 6)

    if n_unique > max_legend_classes:
        title = f"{title} (top {max_legend_classes} classes shown)"

    _set_square_axes(ax, two_d_pts_np)
    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    plt.tight_layout()
    return fig


def _compute_2d_embeddings(X_sampled, manifold):
    """Compute 2D embeddings using log map + t-SNE (hyperbolic) or PCA (Euclidean)."""
    if isinstance(manifold, Euclidean):
        from sklearn.decomposition import PCA as SklearnPCA
        pca = SklearnPCA(n_components=2)
        pts_np = X_sampled.detach().cpu().float().numpy()
        two_d = pca.fit_transform(pts_np)
        return torch.tensor(two_d, dtype=X_sampled.dtype, device=X_sampled.device)

    # Hyperbolic: log map to tangent space at origin, then t-SNE
    if isinstance(manifold, Hyperboloid):
        # Convert to PoincareBall first for logmap_0
        poincare = PoincareBall(c=manifold.c.detach().clone(), dtype=manifold.dtype)
        pts_poincare = poincare.from_hyperboloid(X_sampled.to(manifold.dtype))
        tangent = poincare.logmap_0(pts_poincare)
    elif isinstance(manifold, PoincareBall):
        tangent = manifold.logmap_0(X_sampled.to(manifold.dtype))
    else:
        raise ValueError(f"Unsupported manifold: {manifold}")

    tangent_np = tangent.detach().cpu().float().numpy()
    # Handle edge case: if dim is already 2, skip t-SNE
    if tangent_np.shape[1] == 2:
        return torch.tensor(tangent_np, dtype=X_sampled.dtype, device=X_sampled.device)

    perplexity = min(30.0, max(5.0, tangent_np.shape[0] / 4.0))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca", learning_rate="auto")
    two_d = tsne.fit_transform(tangent_np)
    return torch.tensor(two_d, dtype=X_sampled.dtype, device=X_sampled.device)


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def _create_confusion_matrix_plot(y_true, y_pred, title="Confusion Matrix", str_labels=None, hierarchy_file=None, coarse_label_names=None):
    """Create a confusion matrix visualization, aggregating for large class counts."""
    max_display_classes = 50
    unique_true = np.unique(y_true)
    n_true_classes = len(unique_true)

    y_true_display = y_true
    y_pred_display = y_pred
    aggregated = False

    if n_true_classes > max_display_classes:
        hierarchy_path = hierarchy_file or "./data/imagenet_wordnet_hierarchy.txt"
        if Path(hierarchy_path).exists():
            try:
                from src.utils.imagenet_hierarchy import (
                    load_imagenet_hierarchy,
                    get_labels_at_hierarchy_level,
                )
                parent_child = load_imagenet_hierarchy(hierarchy_path, exclude_root=True)
                y_true_display, _ = get_labels_at_hierarchy_level(
                    np.asarray(y_true), parent_child, target_num_classes=max_display_classes
                )
                pred_to_agg = {}
                for pred_label in np.unique(y_pred):
                    mask = y_pred == pred_label
                    most_common = Counter(y_true_display[mask]).most_common(1)[0][0]
                    pred_to_agg[pred_label] = most_common
                y_pred_display = np.array([pred_to_agg[p] for p in y_pred])
                aggregated = True
                title = f"{title} (aggregated to ~{len(np.unique(y_true_display))} categories)"
            except Exception as e:
                warning(f"Could not aggregate labels using hierarchy: {e}")

    # Use the union of true and predicted labels so the matrix is always square
    all_labels = sorted(set(np.unique(y_true_display)) | set(np.unique(y_pred_display)))
    cm = confusion_matrix(y_true_display, y_pred_display, labels=all_labels)
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.divide(cm.astype("float"), row_sums, where=row_sums != 0, out=np.zeros_like(cm, dtype=float))

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_normalized, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    n_display = len(all_labels)

    # Resolve label IDs to human-readable names
    max_label_len = 40
    if coarse_label_names is not None:
        label_names = [coarse_label_names.get(int(l), str(l)) for l in all_labels]
    elif str_labels is not None and not aggregated:
        label_names = []
        for label in all_labels:
            idx = np.where(y_true == label)[0]
            if len(idx) > 0:
                label_names.append(str(str_labels[idx[0]]))
            else:
                label_names.append(str(label))
    else:
        label_names = [str(l) for l in all_labels]
    # Truncate long names for readability on axes
    label_names = [n if len(n) <= max_label_len else n[:max_label_len - 1] + "\u2026" for n in label_names]

    fontsize = 8 if n_display <= 50 else 6
    ax.set_yticks(np.arange(n_display))
    ax.set_yticklabels(label_names, fontsize=fontsize)
    ax.set_xticks(np.arange(n_display))
    ax.set_xticklabels(label_names, fontsize=fontsize)
    ax.set_ylabel("True Label", fontsize=10)
    ax.set_xlabel("Predicted Cluster", fontsize=10)
    ax.set_title(title, fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Clustering helper: runs agglomerative clustering + confusion matrix + vis
# ---------------------------------------------------------------------------

def _map_preds_to_coarse(pred_labels, Y_coarse_np):
    """Map each predicted cluster to the most common coarse label in that cluster."""
    mapped = np.copy(pred_labels)
    for cluster_id in np.unique(pred_labels):
        mask = pred_labels == cluster_id
        most_common = Counter(Y_coarse_np[mask]).most_common(1)[0][0]
        mapped[mask] = most_common
    return mapped


def _run_agglomerative(
    linkage_name, distance_matrix, X, Y_clustering, Y_coarse,
    str_labels, manifold, two_d_pts, vis_sample_indices, do_vis, results,
    coarse_label_names=None,
):
    """Run agglomerative clustering for one linkage type and store results."""
    Y_np = Y_clustering.cpu().numpy() if isinstance(Y_clustering, torch.Tensor) else np.asarray(Y_clustering)
    n_clusters = len(np.unique(Y_np))
    dist_np = distance_matrix.cpu().numpy()

    if linkage_name == "ward":
        # Ward requires Euclidean input, not precomputed distances.
        # Use log-map tangent vectors as Euclidean proxy.
        if isinstance(manifold, Euclidean):
            X_euc = X.detach().cpu().float().numpy()
        elif isinstance(manifold, Hyperboloid):
            poincare = PoincareBall(c=manifold.c.detach().clone(), dtype=manifold.dtype)
            pts_poincare = poincare.from_hyperboloid(X.to(manifold.dtype))
            X_euc = poincare.logmap_0(pts_poincare).detach().cpu().float().numpy()
        elif isinstance(manifold, PoincareBall):
            X_euc = manifold.logmap_0(X.to(manifold.dtype)).detach().cpu().float().numpy()
        else:
            warning(f"Ward linkage not supported for manifold {manifold}; skipping.")
            return
        clust = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        pred_labels = clust.fit_predict(X_euc)
    else:
        clust = AgglomerativeClustering(
            n_clusters=n_clusters, metric="precomputed", linkage=linkage_name,
        )
        pred_labels = clust.fit_predict(dist_np)

    prefix = f"agglomerative_clustering_{linkage_name}"

    nmi = normalized_mutual_info_score(Y_np, pred_labels)
    ari = adjusted_rand_score(Y_np, pred_labels)
    results[f"{prefix}_nmi"] = nmi
    results[f"{prefix}_ari"] = ari

    # Confusion matrix
    Y_coarse_np = (
        Y_coarse.cpu().numpy() if isinstance(Y_coarse, torch.Tensor) else np.asarray(Y_coarse)
    ) if Y_coarse is not None else None

    y_true_cm = Y_coarse_np if Y_coarse_np is not None else Y_np
    pred_cm = _map_preds_to_coarse(pred_labels, Y_coarse_np) if Y_coarse_np is not None else pred_labels
    cm_fig = _create_confusion_matrix_plot(
        y_true_cm, pred_cm,
        title=f"Agglomerative ({linkage_name.title()}) - Confusion Matrix",
        str_labels=str_labels,
        coarse_label_names=coarse_label_names,
    )
    results[f"{prefix}_confusion_matrix"] = wandb.Image(cm_fig)
    plt.close(cm_fig)

    # Visualization
    if do_vis and two_d_pts is not None:
        str_sub = np.asarray(str_labels)[vis_sample_indices] if str_labels is not None else None
        vis_labels = map_predicted_labels_to_str_labels(
            pred_labels[vis_sample_indices], Y_np[vis_sample_indices], str_sub,
        )
        fig = _create_scatter_plot(
            two_d_pts, vis_labels,
            title=f"Agglomerative ({linkage_name.title()}) Embeddings",
        )
        results[f"{prefix}_embeddings"] = wandb.Image(fig)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

def evaluation(
    X: torch.Tensor,
    Y: torch.Tensor,
    distance_matrix: torch.Tensor,
    manifold: Manifold,
    metrics: dict[str, bool] | None = None,
    str_labels: Optional[npt.ArrayLike] | None = None,
    Y_coarse: Optional[Union[np.ndarray, torch.Tensor]] = None,
    coarse_label_names: Optional[dict[int, str]] = None,
    hierarchy_clusters: list[set[int]] | None = None,
    visualization_labels: Optional[np.ndarray] = None,
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
        Coarse-grained labels for hierarchical datasets (e.g., CIFAR-100, ImageNet)
        If provided, hierarchical dendrogram purity will be computed
    coarse_label_names : Optional[dict[int, str]]
        Mapping from hierarchy node IDs to human-readable names.
        Built by ``get_hierarchy_node_names()`` from the WordNet hierarchy.
    hierarchy_clusters : list[set[int]] | None
        Full multi-level hierarchy clusters (each a set of leaf class IDs).
        Built by ``build_hierarchy_clusters()``.  When provided, hierarchical
        dendrogram purity uses all intermediate levels rather than just fine+coarse.
    visualization_labels : Optional[np.ndarray]
        Per-sample string labels used exclusively for the ground-truth t-SNE
        scatter plot.  When provided, these override coarse_label_names and
        str_labels for visualization only (metrics are unaffected).
    """
    results = {}
    Y_clustering = Y_coarse if Y_coarse is not None else Y
    if metrics is None:
        return results

    # --- Subsampling ---
    if metrics.get("test_sample_size", 0) > 0 and metrics["test_sample_size"] < len(X):
        sample_indices = stratified_sample(Y, metrics["test_sample_size"])
        X = X[sample_indices]
        Y = Y[sample_indices]
        distance_matrix = distance_matrix[sample_indices][:, sample_indices]
        if str_labels is not None:
            str_labels = np.asarray(str_labels)[sample_indices]
        if Y_coarse is not None:
            Y_coarse_np = (
                Y_coarse.cpu().numpy() if isinstance(Y_coarse, torch.Tensor) else np.asarray(Y_coarse)
            )
            Y_coarse = Y_coarse_np[sample_indices]
        if visualization_labels is not None:
            visualization_labels = np.asarray(visualization_labels)[sample_indices]
        Y_clustering = Y_coarse if Y_coarse is not None else Y

    if not isinstance(manifold.c, torch.Tensor):
        manifold.c = torch.tensor(manifold.c, device=X.device, dtype=X.dtype)

    # --- Sanity checks ---
    X_np = X.cpu().numpy()
    embedding_std = np.std(X_np, axis=0).mean()
    embedding_mean_norm = np.linalg.norm(X_np, axis=1).mean()
    if embedding_std < 1e-6:
        warning(
            f"Embeddings appear collapsed (std={embedding_std:.2e}, mean_norm={embedding_mean_norm:.2e}). "
            "Clustering metrics may be unreliable."
        )

    if distance_matrix.min() < 0:
        warning(f"Distance matrix contains negative values (min={distance_matrix.min():.4f}). Clipping to zero.")
        distance_matrix = distance_matrix.clamp(min=0)

    # --- Visualization: 2D embeddings ---
    two_d_pts = None
    vis_sample_indices = None
    do_vis = metrics.get("visualization", False)

    if do_vis:
        sample_size = min(metrics.get("visualization_sample_size", 5000), len(X))
        vis_sample_indices = stratified_sample(Y_clustering, sample_size)
        points_sampled = X[vis_sample_indices]
        try:
            two_d_pts = _compute_2d_embeddings(points_sampled, manifold)
        except Exception as e:
            warning(f"2D embedding computation failed: {e}")
            do_vis = False

    # --- Delta hyperbolicity ---
    if metrics.get("delta_hyperbolicity", False):
        from src.hyperbolic_math.src.utils.helpers import get_delta
        delta_hyp, diam_hyp, rel_delta_hyp = get_delta(
            X, manifold, len(X), version="average", batch_size=100
        )
        results["delta_hyperbolicity"] = delta_hyp.item()
        results["diameter_hyperbolicity"] = diam_hyp.item()
        results["relative_delta_hyperbolicity"] = rel_delta_hyp.item()

    # --- Dendrogram purity ---
    dp_use_root = metrics.get("dendrogrampurity_use_root", False)
    for dp_linkage in ("single", "complete", "average", "ward"):
        if not metrics.get(f"dendrogram_purity_{dp_linkage}", False):
            continue
        if Y_coarse is not None:
            dp_score = hierarchical_dendrogram_purity(
                distance_matrix, Y, Y_coarse, dp_linkage, use_root=dp_use_root,
                hierarchy_clusters=hierarchy_clusters,
            )
            results[f"hierarchical_dendrogram_purity_{dp_linkage}"] = dp_score
            # Flat DP: standard purity on fine-grained labels only (no WordNet hierarchy).
            # This matches the gHHC benchmark metric for a direct apples-to-apples comparison.
            flat_dp_score = dendrogram_purity(distance_matrix, Y, dp_linkage, use_root=dp_use_root)
            results[f"dendrogram_purity_{dp_linkage}"] = flat_dp_score
        else:
            dp_score = dendrogram_purity(distance_matrix, Y, dp_linkage, use_root=dp_use_root)
            results[f"dendrogram_purity_{dp_linkage}"] = dp_score

    # --- Agglomerative clustering ---
    for linkage_name in ("single", "complete", "average", "ward"):
        if metrics.get(f"agglomerative_clustering_{linkage_name}", False):
            _run_agglomerative(
                linkage_name, distance_matrix, X, Y, Y_coarse,
                str_labels, manifold, two_d_pts, vis_sample_indices, do_vis, results,
                coarse_label_names=coarse_label_names,
            )

    # --- Ground-truth embedding visualization ---
    if do_vis and two_d_pts is not None:
        Y_np = Y_clustering.cpu().numpy() if isinstance(Y_clustering, torch.Tensor) else np.asarray(Y_clustering)
        # Resolve per-sample labels to human-readable names for the scatter plot
        vis_Y = Y_np[vis_sample_indices]
        if visualization_labels is not None:
            # Explicit visualization labels (e.g. CIFAR-100 20 superclasses) take priority
            labels = np.asarray(visualization_labels)[vis_sample_indices]
        elif coarse_label_names is not None:
            labels = np.array([coarse_label_names.get(int(l), str(l)) for l in vis_Y])
        elif str_labels is not None:
            labels = np.asarray(str_labels)[vis_sample_indices]
            labels = np.array([
                str(l) if isinstance(l, (tuple, list, np.ndarray)) else (l if l is not None else "Unknown")
                for l in labels
            ])
        else:
            labels = vis_Y

        method = "t-SNE" if not isinstance(manifold, Euclidean) else "PCA"
        fig = _create_scatter_plot(two_d_pts, labels, title=f"{method} Embeddings (Ground Truth)")
        results[f"{'tsne' if method == 't-SNE' else 'pca'}_embeddings"] = wandb.Image(fig)
        plt.close(fig)

    # --- Nearest neighbors accuracy ---
    if metrics.get("nearest_neighbors_accuracy", False):
        for k in (1, 5, 10, 100, 1000):
            results[f"{k}_nearest_neighbors_accuracy"] = nearest_neighbors_accuracy(
                distance_matrix, Y, k=k,
            )

    # --- Spectral clustering ---
    if metrics.get("spectral_clustering", False):
        Y_np = Y.cpu().numpy() if isinstance(Y, torch.Tensor) else np.asarray(Y)
        n_clusters = len(np.unique(Y_np))
        spectral = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed_nearest_neighbors",
            n_neighbors=metrics.get("spectral_nn", 10),
            assign_labels="kmeans",
            random_state=int(torch.get_rng_state()[0].item()),
        )
        pred_labels_spectral = spectral.fit_predict(distance_matrix.cpu().numpy())
        results["spectral_clustering_nmi"] = normalized_mutual_info_score(Y_np, pred_labels_spectral)
        results["spectral_clustering_ari"] = adjusted_rand_score(Y_np, pred_labels_spectral)

        # Confusion matrix
        Y_coarse_np = (
            Y_coarse.cpu().numpy() if isinstance(Y_coarse, torch.Tensor) else np.asarray(Y_coarse)
        ) if Y_coarse is not None else None
        y_true_cm = Y_coarse_np if Y_coarse_np is not None else Y_np
        pred_cm = _map_preds_to_coarse(pred_labels_spectral, Y_coarse_np) if Y_coarse_np is not None else pred_labels_spectral
        cm_fig = _create_confusion_matrix_plot(y_true_cm, pred_cm, title="Spectral Clustering - Confusion Matrix", str_labels=str_labels, coarse_label_names=coarse_label_names)
        results["spectral_clustering_confusion_matrix"] = wandb.Image(cm_fig)
        plt.close(cm_fig)

        if do_vis and two_d_pts is not None:
            str_sub = np.asarray(str_labels)[vis_sample_indices] if str_labels is not None else None
            vis_labels = map_predicted_labels_to_str_labels(
                pred_labels_spectral[vis_sample_indices], Y_np[vis_sample_indices], str_sub,
            )
            fig = _create_scatter_plot(two_d_pts, vis_labels, title="Spectral Clustering Embeddings")
            results["spectral_clustering_embeddings"] = wandb.Image(fig)
            plt.close(fig)

    # --- K-means ---
    if metrics.get("kmeans", False):
        Y_np = Y.cpu().numpy() if isinstance(Y, torch.Tensor) else np.asarray(Y)
        n_clusters = len(np.unique(Y_np))
        kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=int(torch.get_rng_state()[0].item()))
        pred_labels_km = kmeans.fit_predict(X.cpu().numpy())
        results["kmeans_nmi"] = normalized_mutual_info_score(Y_np, pred_labels_km)
        results["kmeans_ari"] = adjusted_rand_score(Y_np, pred_labels_km)

        Y_coarse_np = (
            Y_coarse.cpu().numpy() if isinstance(Y_coarse, torch.Tensor) else np.asarray(Y_coarse)
        ) if Y_coarse is not None else None
        y_true_cm = Y_coarse_np if Y_coarse_np is not None else Y_np
        pred_cm = _map_preds_to_coarse(pred_labels_km, Y_coarse_np) if Y_coarse_np is not None else pred_labels_km
        cm_fig = _create_confusion_matrix_plot(y_true_cm, pred_cm, title="K-Means - Confusion Matrix", str_labels=str_labels, coarse_label_names=coarse_label_names)
        results["kmeans_confusion_matrix"] = wandb.Image(cm_fig)
        plt.close(cm_fig)

        if do_vis and two_d_pts is not None:
            str_sub = np.asarray(str_labels)[vis_sample_indices] if str_labels is not None else None
            vis_labels = map_predicted_labels_to_str_labels(
                pred_labels_km[vis_sample_indices], Y_np[vis_sample_indices], str_sub,
            )
            fig = _create_scatter_plot(two_d_pts, vis_labels, title="K-Means Embeddings")
            results["kmeans_embeddings"] = wandb.Image(fig)
            plt.close(fig)

    # --- Dasgupta cost ---
    if metrics.get("dasgupta_cost", False):
        results["dasgupta_cost"] = dasgupta_cost(torch.reciprocal(distance_matrix + 1), "average")

    return results


# ---------------------------------------------------------------------------
# Tree utilities
# ---------------------------------------------------------------------------

class Node:
    def __init__(self, id, left=None, right=None):
        self.id = id
        self.left = left
        self.right = right
        self._leaves_count = None

    def get_leaves_count(self):
        if self._leaves_count is not None:
            return self._leaves_count
        if self.left is None and self.right is None:
            self._leaves_count = 1
        else:
            self._leaves_count = self.left.get_leaves_count() + self.right.get_leaves_count()
        return self._leaves_count


def dasgupta_cost(similarity_matrix: torch.Tensor, linkage_method: str = "average") -> float:
    """Computes the Dasgupta cost for a hierarchy built from a similarity matrix."""
    num_samples = similarity_matrix.shape[0]
    if num_samples < 2:
        return 0.0

    distance_matrix = 1 - similarity_matrix
    condensed_distances = squareform(distance_matrix.cpu().numpy(), checks=False)
    Z = linkage(condensed_distances, method=linkage_method)

    nodes = [Node(i) for i in range(num_samples)]
    for i, merge in enumerate(Z):
        node1_id, node2_id = int(merge[0]), int(merge[1])
        new_node = Node(id=num_samples + i, left=nodes[node1_id], right=nodes[node2_id])
        nodes.append(new_node)
    root = nodes[-1]

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

    total_cost = 0.0
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            ancestors_i = set()
            curr = nodes[i]
            while curr is not None:
                ancestors_i.add(curr)
                curr = parent_map.get(curr)

            curr = nodes[j]
            while curr is not None:
                if curr in ancestors_i:
                    num_lca_leaves = curr.get_leaves_count()
                    total_cost += similarity_matrix[i, j].item() * num_lca_leaves
                    break
                curr = parent_map.get(curr)

    return total_cost


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

def map_predicted_labels_to_str_labels(
    predicted_labels: np.ndarray,
    true_labels: np.ndarray,
    str_labels: Optional[npt.ArrayLike] = None,
) -> np.ndarray:
    """Map predicted cluster labels to string labels using majority voting."""
    if str_labels is None:
        return predicted_labels

    str_labels = np.asarray(str_labels)
    predicted_labels = np.asarray(predicted_labels)
    true_labels = np.asarray(true_labels)

    unique_clusters = np.unique(predicted_labels)
    mapped_labels = np.copy(predicted_labels).astype(object)

    for cluster_id in unique_clusters:
        cluster_mask = predicted_labels == cluster_id
        cluster_true_labels = true_labels[cluster_mask]
        most_common_true_label = Counter(cluster_true_labels).most_common(1)[0][0]

        true_label_mask = true_labels == most_common_true_label
        representative_str_label = str_labels[true_label_mask][0]
        if isinstance(representative_str_label, (tuple, list, np.ndarray)):
            representative_str_label = str(representative_str_label)
        mapped_labels[cluster_mask] = representative_str_label

    return mapped_labels


# ---------------------------------------------------------------------------
# Nearest neighbors accuracy
# ---------------------------------------------------------------------------

def nearest_neighbors_accuracy(
    distance_matrix: Union[np.ndarray, torch.Tensor],
    labels: Union[np.ndarray, torch.Tensor],
    k: int = 1,
) -> float:
    """Computes the nearest neighbors accuracy based on the distance matrix and labels."""
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    if isinstance(distance_matrix, torch.Tensor):
        distance_matrix = distance_matrix.detach().cpu().numpy()

    num_samples = len(labels)
    if num_samples < 2:
        return 1.0

    nearest_indices = np.argsort(distance_matrix, axis=1)[:, 1 : k + 1]
    average_accuracy = 0.0
    for i in range(num_samples):
        counter = 0
        for j in range(k):
            if labels[nearest_indices[i, j]] == labels[i]:
                counter += 1
        average_accuracy += counter / k
    return average_accuracy / num_samples


# ---------------------------------------------------------------------------
# Dendrogram purity
# ---------------------------------------------------------------------------

def dendrogram_purity(
    distance_matrix: torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    linkage_method: str = "average",
    use_root: bool = False,
) -> float:
    """
    Computes the canonical Dendrogram Purity for a hierarchy built from a distance matrix.

    DP(T) = (1 / |P*|) * sum_k sum_{xi, xj in C*_k, i!=j} pur(lca(xi, xj), C*_k)
    """
    num_samples = len(labels)
    if num_samples < 2:
        return 1.0

    distance_matrix_np = distance_matrix.cpu().numpy()
    labels_np = np.asarray(labels)

    if distance_matrix_np.shape != (num_samples, num_samples):
        raise ValueError("Distance matrix must be square and match number of labels.")

    _, class_counts = np.unique(labels_np, return_counts=True)
    num_same_class_pairs = sum(count * (count - 1) // 2 for count in class_counts)
    if num_same_class_pairs == 0:
        return 1.0

    condensed_distances = squareform(distance_matrix_np, checks=False)
    Z = linkage(condensed_distances, method=linkage_method)

    total_purity_sum = 0.0
    cluster_label_counts = {i: Counter({labels_np[i]: 1}) for i in range(num_samples)}

    num_merges = len(Z)
    for i, merge in enumerate(Z):
        if not use_root and i == num_merges - 1:
            break
        node1_id, node2_id = int(merge[0]), int(merge[1])
        new_node_id = num_samples + i

        counts1 = cluster_label_counts[node1_id]
        counts2 = cluster_label_counts[node2_id]
        new_node_counts = counts1 + counts2
        cluster_label_counts[new_node_id] = new_node_counts

        lca_node_size = sum(new_node_counts.values())

        for label in set(counts1.keys()) | set(counts2.keys()):
            n1_k = counts1.get(label, 0)
            n2_k = counts2.get(label, 0)
            num_new_pairs = n1_k * n2_k
            if num_new_pairs > 0:
                purity = new_node_counts.get(label, 0) / lca_node_size
                total_purity_sum += num_new_pairs * purity

        del cluster_label_counts[node1_id]
        del cluster_label_counts[node2_id]

    return total_purity_sum / num_same_class_pairs


def build_hierarchy_clusters(parent_child: dict[int, list[int]], leaf_ids: set[int]) -> list[set[int]]:
    """Build leaf-set clusters from a parent-child hierarchy.

    Each internal node in the hierarchy defines a cluster consisting of all
    leaf descendants reachable from it.  Singleton clusters and the full-root
    cluster are excluded so that the metric captures intermediate structure.

    Parameters
    ----------
    parent_child : dict[int, list[int]]
        Parent → children mapping (numeric IDs).
    leaf_ids : set[int]
        The set of leaf node IDs (e.g. class indices 0-199).

    Returns
    -------
    list[set[int]]
        Unique, non-trivial clusters of leaf IDs.
    """
    node_leaves: dict[int, set[int]] = {}

    def get_leaves(u: int) -> set[int]:
        if u in node_leaves:
            return node_leaves[u]
        if u in leaf_ids and u not in parent_child:
            s = {u}
            node_leaves[u] = s
            return s
        leaves: set[int] = set()
        if u in parent_child:
            for v in parent_child[u]:
                leaves.update(get_leaves(v))
        # u might be a leaf that also appears as a parent (unlikely but safe)
        if u in leaf_ids:
            leaves.add(u)
        node_leaves[u] = leaves
        return leaves

    num_leaves = len(leaf_ids)
    seen: set[frozenset[int]] = set()
    unique_clusters: list[set[int]] = []
    all_nodes = set(parent_child.keys())
    for children in parent_child.values():
        all_nodes.update(children)

    for u in all_nodes:
        leaves = get_leaves(u)
        c_frozen = frozenset(leaves)
        if c_frozen not in seen and 1 < len(leaves) < num_leaves:
            seen.add(c_frozen)
            unique_clusters.append(leaves)

    return unique_clusters


def hierarchical_dendrogram_purity(
    distance_matrix: torch.Tensor,
    labels_fine: np.ndarray | torch.Tensor,
    labels_coarse: np.ndarray | torch.Tensor,
    linkage_method: str = "average",
    use_root: bool = False,
    hierarchy_clusters: list[set[int]] | None = None,
) -> float:
    """
    Computes the Hierarchical Dendrogram Purity for a hierarchy built from a distance matrix.

    When ``hierarchy_clusters`` is provided (recommended), the full multi-level
    WordNet hierarchy is used.  Each cluster is a set of leaf class IDs that
    share a common ancestor.  This works for any dataset (CIFAR-100, ImageNet,
    Tiny ImageNet, etc.).

    Falls back to 2-level (fine + coarse) DP when no hierarchy clusters are given.
    """
    num_samples = len(labels_fine)
    if num_samples < 2:
        return 1.0

    distance_matrix_np = distance_matrix.cpu().numpy()
    labels_fine_np = np.asarray(labels_fine)
    labels_coarse_np = np.asarray(labels_coarse)

    if distance_matrix_np.shape != (num_samples, num_samples):
        raise ValueError("Distance matrix must be square and match number of labels.")

    if hierarchy_clusters is not None and len(hierarchy_clusters) > 0:
        if use_root:
            root_cluster = set(np.unique(labels_fine_np).tolist())
            if root_cluster not in [set(c) for c in hierarchy_clusters]:
                hierarchy_clusters = list(hierarchy_clusters) + [root_cluster]

        _, fine_counts = np.unique(labels_fine_np, return_counts=True)
        fine_label_counts_map = dict(zip(np.unique(labels_fine_np), fine_counts))

        num_same_class_pairs = 0
        for cluster in hierarchy_clusters:
            count = sum(fine_label_counts_map.get(l, 0) for l in cluster)
            num_same_class_pairs += count * (count - 1) // 2

        if num_same_class_pairs == 0:
            return 1.0

        condensed_distances = squareform(distance_matrix_np, checks=False)
        Z = linkage(condensed_distances, method=linkage_method)

        total_purity_sum = 0.0
        cluster_label_counts = {i: Counter({labels_fine_np[i]: 1}) for i in range(num_samples)}

        num_merges = len(Z)
        for i, merge in enumerate(Z):
            if not use_root and i == num_merges - 1:
                break
            node1_id, node2_id = int(merge[0]), int(merge[1])
            new_node_id = num_samples + i

            counts1 = cluster_label_counts[node1_id]
            counts2 = cluster_label_counts[node2_id]
            new_node_counts = counts1 + counts2
            cluster_label_counts[new_node_id] = new_node_counts
            lca_node_size = sum(new_node_counts.values())

            for cluster in hierarchy_clusters:
                n1_k = sum(counts1[l] for l in cluster if l in counts1)
                n2_k = sum(counts2[l] for l in cluster if l in counts2)
                num_new_pairs = n1_k * n2_k
                if num_new_pairs > 0:
                    count_in_lca = n1_k + n2_k
                    purity = count_in_lca / lca_node_size
                    total_purity_sum += num_new_pairs * purity

            del cluster_label_counts[node1_id]
            del cluster_label_counts[node2_id]

        return total_purity_sum / num_same_class_pairs

    # Fallback: 2-level (fine + coarse) DP
    if len(labels_fine_np) != len(labels_coarse_np):
        raise ValueError("Fine and coarse labels must have the same length.")

    _, fine_class_counts = np.unique(labels_fine_np, return_counts=True)
    num_fine_pairs = sum(count * (count - 1) // 2 for count in fine_class_counts)
    _, coarse_class_counts = np.unique(labels_coarse_np, return_counts=True)
    num_coarse_pairs = sum(count * (count - 1) // 2 for count in coarse_class_counts)
    num_same_class_pairs = num_fine_pairs + num_coarse_pairs

    if num_same_class_pairs == 0:
        return 1.0

    condensed_distances = squareform(distance_matrix_np, checks=False)
    Z = linkage(condensed_distances, method=linkage_method)

    def compute_purity_sum_for_labels(labels_np):
        total_purity_sum = 0.0
        cluster_label_counts = {i: Counter({labels_np[i]: 1}) for i in range(num_samples)}
        num_merges = len(Z)
        for i, merge in enumerate(Z):
            if not use_root and i == num_merges - 1:
                break
            node1_id, node2_id = int(merge[0]), int(merge[1])
            new_node_id = num_samples + i

            counts1 = cluster_label_counts[node1_id]
            counts2 = cluster_label_counts[node2_id]
            new_node_counts = counts1 + counts2
            cluster_label_counts[new_node_id] = new_node_counts
            lca_node_size = sum(new_node_counts.values())

            for label in set(counts1.keys()) | set(counts2.keys()):
                n1_k = counts1.get(label, 0)
                n2_k = counts2.get(label, 0)
                num_new_pairs = n1_k * n2_k
                if num_new_pairs > 0:
                    purity = new_node_counts.get(label, 0) / lca_node_size
                    total_purity_sum += num_new_pairs * purity

            del cluster_label_counts[node1_id]
            del cluster_label_counts[node2_id]
        return total_purity_sum

    fine_purity_sum = compute_purity_sum_for_labels(labels_fine_np)
    coarse_purity_sum = compute_purity_sum_for_labels(labels_coarse_np)
    return (fine_purity_sum + coarse_purity_sum) / num_same_class_pairs


# ---------------------------------------------------------------------------
# Hyperbolic spectral clustering (alternative implementation)
# ---------------------------------------------------------------------------

class HyperbolicSpectralClustering(BaseEstimator, ClusterMixin):
    """Spectral Clustering on a pre-computed geodesic distance matrix from hyperbolic space."""

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
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.affinity = affinity
        self.n_neighbors = n_neighbors
        self.assign_labels = assign_labels
        self.n_init = n_init
        self.random_state = random_state
        self.labels_ = None

    def fit(self, D, y=None):
        self.fit_predict(D)
        return self

    def fit_predict(self, D, y=None):
        if D.shape[0] != D.shape[1]:
            raise ValueError("Input D must be a square distance matrix.")

        if self.affinity == "rbf":
            affinity_matrix = np.exp(-self.gamma * (D**2))
            np.fill_diagonal(affinity_matrix, 0)
        elif self.affinity == "nearest_neighbors":
            connectivity = kneighbors_graph(
                D, n_neighbors=self.n_neighbors, mode="distance",
                metric="precomputed", include_self=False,
            )
            affinity_matrix = 0.5 * (connectivity + connectivity.T)
            affinity_matrix.data = np.exp(-self.gamma * (affinity_matrix.data**2))
        else:
            raise ValueError(f"Unknown affinity type: {self.affinity}")

        laplacian, dd = csgraph.laplacian(affinity_matrix, normed=True, return_diag=True)
        laplacian_shifted = laplacian + 1e-10 * identity(laplacian.shape[0])
        _, eigenvectors = eigsh(
            laplacian_shifted, k=self.n_clusters, which="SM", tol=1e-6,
            v0=np.ones(laplacian.shape[0]),
        )

        norm = np.linalg.norm(eigenvectors, axis=1, keepdims=True)
        spectral_embedding = eigenvectors / (norm + 1e-10)

        if self.assign_labels == "kmeans":
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=self.n_init, random_state=self.random_state)
            self.labels_ = kmeans.fit_predict(spectral_embedding)
        else:
            raise ValueError(f"Unknown label assignment strategy: {self.assign_labels}")

        return self.labels_


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def stratified_sample(
    labels: Union[np.ndarray, torch.Tensor], sample_size: int
) -> np.ndarray:
    """Sample equal number of points from each class (stratified sampling)."""
    if isinstance(labels, torch.Tensor):
        labels_np = labels.cpu().numpy()
    else:
        labels_np = np.asarray(labels)

    unique_classes = np.unique(labels_np)
    n_classes = len(unique_classes)
    samples_per_class = sample_size // n_classes
    remaining_samples = sample_size % n_classes

    sampled_indices = []
    for i, cls in enumerate(unique_classes):
        class_indices = np.where(labels_np == cls)[0]
        n_samples_this_class = samples_per_class + (1 if i < remaining_samples else 0)
        n_samples_this_class = min(n_samples_this_class, len(class_indices))
        if n_samples_this_class > 0:
            sampled = np.random.choice(class_indices, n_samples_this_class, replace=False)
            sampled_indices.extend(sampled)

    return np.array(sampled_indices)
