import json
from datetime import datetime
from logging import warning
import torch
import numpy as np

from typing import Union, Optional
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import normalized_mutual_info_score
from collections import Counter
from pathlib import Path

from src.hyperbolic_math.src.manifolds import Euclidean, PoincareBall, Hyperboloid, Manifold
from src.utils.lca import compute_pairwise_lca_depths


# ---------------------------------------------------------------------------
# Results file I/O
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
    return str(value)


def write_results(
    metrics: dict,
    results_dir: str | Path,
    stage: str = "eval",
    epoch: int | None = None,
) -> Path:
    """
    Write evaluation metrics to a human-readable JSON file.

    Two files are maintained:
    - ``<results_dir>/<stage>_results_latest.json`` — overwritten each call
      (always reflects the most recent evaluation).
    - ``<results_dir>/<stage>_results_history.jsonl`` — one JSON object per
      line, appended each call (full history of intermediate results).

    Parameters
    ----------
    metrics : dict
        The metrics dict.
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
        record[key] = _serializable(value)

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
# Main evaluation entry point
# ---------------------------------------------------------------------------

def evaluation(
    X: torch.Tensor,
    Y: torch.Tensor,
    distance_matrix: torch.Tensor,
    manifold: Manifold,
    metrics: dict[str, bool] | None = None,
    Y_coarse: Optional[Union[np.ndarray, torch.Tensor]] = None,
    hierarchy_clusters: list[set[int]] | None = None,
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
    Y_coarse : Optional[Union[np.ndarray, torch.Tensor]]
        Coarse-grained labels for hierarchical datasets (e.g., CIFAR-100, ImageNet)
        If provided, hierarchical dendrogram purity will be computed
    hierarchy_clusters : list[set[int]] | None
        Full multi-level hierarchy clusters (each a set of leaf class IDs).
        Built by ``build_hierarchy_clusters()``.  When provided, hierarchical
        dendrogram purity uses all intermediate levels rather than just fine+coarse.
    """
    results = {}
    if metrics is None:
        return results

    # --- Subsampling ---
    if metrics.get("test_sample_size", 0) > 0 and metrics["test_sample_size"] < len(X):
        sample_indices = stratified_sample(Y, metrics["test_sample_size"])
        X = X[sample_indices]
        Y = Y[sample_indices]
        distance_matrix = distance_matrix[sample_indices][:, sample_indices]
        if Y_coarse is not None:
            Y_coarse_np = (
                Y_coarse.cpu().numpy() if isinstance(Y_coarse, torch.Tensor) else np.asarray(Y_coarse)
            )
            Y_coarse = Y_coarse_np[sample_indices]

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

    # --- Dendrogram purity ---
    # Each tree is decoded by agglomerating (name, dissimilarities, linkage criterion).
    dp_trees = [
        (dp_linkage, distance_matrix, dp_linkage)
        for dp_linkage in ("single", "complete", "average", "ward")
        if metrics.get(f"dendrogram_purity_{dp_linkage}", False)
    ]
    if metrics.get("dendrogram_purity_lca", False):
        if isinstance(manifold, (PoincareBall, Hyperboloid)):
            # LCA-induced tree: single linkage on the LCA dissimilarities, no linkage
            # criterion is applied to the embeddings themselves.
            dp_trees.append(("lca", lca_tree_distances(X, manifold), "single"))
        else:
            warning(f"The LCA-induced tree is not defined for manifold {manifold}; skipping.")

    dp_use_root = metrics.get("dendrogrampurity_use_root", False)
    for dp_name, dp_distances, dp_linkage in dp_trees:
        if Y_coarse is not None:
            results[f"hierarchical_dendrogram_purity_{dp_name}"] = hierarchical_dendrogram_purity(
                dp_distances, Y, Y_coarse, dp_linkage, use_root=dp_use_root,
                hierarchy_clusters=hierarchy_clusters,
            )
        # Flat DP: standard purity on fine-grained labels only (no WordNet hierarchy).
        results[f"dendrogram_purity_{dp_name}"] = dendrogram_purity(dp_distances, Y, dp_linkage, use_root=dp_use_root)

    # --- Agglomerative clustering ---
    for linkage_name in ("single", "complete", "average", "ward"):
        if metrics.get(f"agglomerative_clustering_{linkage_name}", False):
            nmi = agglomerative_clustering_nmi(linkage_name, distance_matrix, X, Y, manifold)
            if nmi is not None:
                results[f"agglomerative_clustering_{linkage_name}_nmi"] = nmi

    # --- Nearest neighbors accuracy and Recall@K ---
    do_knn = metrics.get("nearest_neighbors_accuracy", False)
    do_recall = metrics.get("recall_at_k", False)
    if (do_knn or do_recall) and len(Y) > 1:
        labels_np = Y.detach().cpu().numpy() if isinstance(Y, torch.Tensor) else np.asarray(Y)
        neighbor_labels = nearest_neighbor_labels(distance_matrix, labels_np)
        if do_knn:
            for k in (1, 5, 10, 100, 1000):
                results[f"{k}_nearest_neighbors_accuracy"] = nearest_neighbors_accuracy(neighbor_labels, labels_np, k=k)
        if do_recall:
            for k in (2, 10):
                results[f"recall_at_{k}"] = recall_at_k(neighbor_labels, labels_np, k=k)

    return results


# ---------------------------------------------------------------------------
# Agglomerative clustering
# ---------------------------------------------------------------------------

def agglomerative_clustering_nmi(
    linkage_name: str,
    distance_matrix: torch.Tensor,
    X: torch.Tensor,
    Y: Union[np.ndarray, torch.Tensor],
    manifold: Manifold,
) -> float | None:
    """NMI between the labels and an agglomerative clustering with as many clusters as classes."""
    Y_np = Y.cpu().numpy() if isinstance(Y, torch.Tensor) else np.asarray(Y)
    n_clusters = len(np.unique(Y_np))

    if linkage_name == "ward":
        # Ward requires Euclidean input, not precomputed distances.
        # Use log-map tangent vectors as Euclidean proxy.
        if isinstance(manifold, Euclidean):
            X_euc = X.detach().cpu().float().numpy()
        elif isinstance(manifold, Hyperboloid):
            poincare = PoincareBall(c=manifold.c.detach().clone(), dtype=manifold.dtype)
            pts_poincare = manifold.to_poincare(X.to(manifold.dtype))
            X_euc = poincare.logmap_0(pts_poincare).detach().cpu().float().numpy()
        elif isinstance(manifold, PoincareBall):
            X_euc = manifold.logmap_0(X.to(manifold.dtype)).detach().cpu().float().numpy()
        else:
            warning(f"Ward linkage not supported for manifold {manifold}; skipping.")
            return None
        clust = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        pred_labels = clust.fit_predict(X_euc)
    else:
        clust = AgglomerativeClustering(
            n_clusters=n_clusters, metric="precomputed", linkage=linkage_name,
        )
        pred_labels = clust.fit_predict(distance_matrix.cpu().numpy())

    return normalized_mutual_info_score(Y_np, pred_labels)


# ---------------------------------------------------------------------------
# Nearest neighbors accuracy
# ---------------------------------------------------------------------------

def nearest_neighbor_labels(
    distance_matrix: Union[np.ndarray, torch.Tensor],
    labels: np.ndarray,
) -> np.ndarray:
    """Labels of all other points, sorted by increasing distance to each point (self-match excluded)."""
    if isinstance(distance_matrix, torch.Tensor):
        distance_matrix = distance_matrix.detach().cpu().numpy()
    nearest_indices = np.argsort(distance_matrix, axis=1)[:, 1:]
    return labels[nearest_indices]


def nearest_neighbors_accuracy(neighbor_labels: np.ndarray, labels: np.ndarray, k: int = 1) -> float:
    """Average fraction of the k nearest neighbors that share the label of the query."""
    return float((neighbor_labels[:, :k] == labels[:, None]).mean())


def recall_at_k(neighbor_labels: np.ndarray, labels: np.ndarray, k: int = 1) -> float:
    """Metric-learning Recall@K: fraction of queries with at least one same-class
    neighbor among the k nearest (self-match excluded).
    """
    return float((neighbor_labels[:, :k] == labels[:, None]).any(axis=1).mean())


# ---------------------------------------------------------------------------
# Dendrogram purity
# ---------------------------------------------------------------------------

def lca_tree_distances(X: torch.Tensor, manifold: Manifold) -> torch.Tensor:
    """Pairwise dissimilarities whose single-linkage dendrogram is the LCA-induced tree.

    The hierarchical loss natively induces a tree by agglomerating on its own rectified-LCA
    depths. A deeper LCA means more related, so the dissimilarity is ``max(depth) - depth``:
    single linkage only depends on the ordering of the pairwise values, which makes it
    exactly "merge the pair with the deepest LCA first". Like the pairwise distances of
    the other metrics, the depths are computed with the non-smoothened distance.
    """
    lca_depth = compute_pairwise_lca_depths(X.to(manifold.dtype), manifold, version="rectified", dist_version="default")
    lca_distance = lca_depth.max() - lca_depth
    lca_distance.fill_diagonal_(0.0)
    return lca_distance


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


def build_hierarchy_clusters(parent_child: dict[int, list[int]], leaf_ids: set[int], include_singletons: bool = False) -> list[set[int]]:
    """Build leaf-set clusters from a parent-child hierarchy.

    Each internal node in the hierarchy defines a cluster consisting of all
    leaf descendants reachable from it.  The full-root cluster is always
    excluded.  When *include_singletons* is False (default), singleton
    clusters (individual leaf classes) are also excluded so that the metric
    captures intermediate structure only.  Set *include_singletons=True*
    to include all hierarchy levels.

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
        min_size = 1 if include_singletons else 2
        if c_frozen not in seen and min_size <= len(leaves) < num_leaves:
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
