from matplotlib import pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from typing import Tuple, List
from sklearn.decomposition import PCA
import torch
from tqdm import tqdm

def get_cluster_pairs(labels: np.ndarray) -> set:
    """Get all pairs of points that belong to the same cluster more efficiently."""
    unique_labels = np.unique(labels)
    pairs = set()
    
    for label in unique_labels:
        indices = np.where(labels == label)[0]
        if len(indices) > 1:
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    pairs.add(tuple(sorted([indices[i], indices[j]])))
    return pairs


def plot_embeddings(all_embeddings_np, all_labels_np, class_names=None, label_type='class'):
    """
    Plot embeddings with proper class labels in legend.
    
    Args:
        all_embeddings_np: numpy array of embeddings
        all_labels_np: numpy array of label indices
        class_names: dictionary mapping label indices to tuples of (class_name, superclass_name)
                    or list where index corresponds to label
        label_type: string indicating what to show in legend:
                   'class' - only class name
                   'super' - only superclass name
                   'both' - both class and superclass names
    """
    # PCA projection
    pca = PCA(n_components=2)
    twoD_embeddings = pca.fit_transform(all_embeddings_np)

    # Normalize to unit disk
    norms = np.sqrt((twoD_embeddings**2).sum(axis=1))
    twoD_embeddings = twoD_embeddings / np.maximum(norms[:, None], 1)

    # Create figure
    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    # Draw boundary circle
    circle = plt.Circle((0, 0), 1, fill=False, color='black', alpha=0.5)
    ax.add_artist(circle)

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

    # Get unique display labels and map to colors
    unique_display_labels = sorted(set(label_mapping.values()))
    if len(unique_display_labels) <= 10:
        cmap = plt.cm.tab10
    else:
        cmap = plt.cm.tab20
    colors = cmap(np.linspace(0, 1, len(unique_display_labels)))
    color_map = dict(zip(unique_display_labels, colors))

    # Plot points grouped by display labels
    for display_label in unique_display_labels:
        # Find all labels that map to this display label
        matching_labels = [label for label, mapped_label in label_mapping.items() 
                         if mapped_label == display_label]
        mask = np.isin(all_labels_np, matching_labels)
        
        plt.scatter(twoD_embeddings[mask, 0], 
                   twoD_embeddings[mask, 1],
                   c=[color_map[display_label]],
                   label=display_label,
                   alpha=0.6)

    # Add legend outside of the circle
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # Set equal aspect ratio and limits
    ax.set_aspect('equal')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    
    # Add title
    plt.title("Hyperbolic Embeddings")
    
    return plt.gcf()


import torch
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

def dendrogram_purity(linkage_matrix, labels):
    """
    Calculate dendrogram purity given a linkage matrix and ground truth labels.
    
    Args:
        linkage_matrix (numpy.ndarray or torch.Tensor): Linkage matrix from hierarchical clustering.
        labels (torch.Tensor or list): Ground truth labels for each data point.
    
    Returns:
        float: Dendrogram purity score.
    """
    # Convert inputs to appropriate types
    if isinstance(linkage_matrix, torch.Tensor):
        linkage_matrix = linkage_matrix.cpu().numpy()
    labels = torch.as_tensor(labels)

    n_samples = labels.size(0)
    unique_labels = torch.unique(labels)
    n_classes = len(unique_labels)

    # Handle edge cases
    if n_samples <= 1 or n_classes == 1:
        return 1.0 if n_samples == 1 else 0.0

    # Label-to-index mapping
    label_to_idx = {label.item(): idx for idx, label in enumerate(unique_labels)}

    # Calculate valid pairs
    label_counts = torch.bincount(labels).cpu().numpy()
    total_pairs = sum(c * (c - 1) // 2 for c in label_counts if c >= 2)
    if total_pairs == 0:
        return 0.0

    # Build tree structure
    n_merges = linkage_matrix.shape[0]  # Use .shape for numpy arrays
    n_nodes = n_samples + n_merges

    parent = torch.full((n_nodes,), -1, dtype=torch.int64)
    left_child = torch.zeros(n_nodes, dtype=torch.int64)
    right_child = torch.zeros(n_nodes, dtype=torch.int64)
    class_counts = torch.zeros((n_nodes, n_classes), dtype=torch.float32)

    # Initialize leaves
    for i in range(n_samples):
        lbl_idx = label_to_idx[labels[i].item()]
        class_counts[i, lbl_idx] = 1.0

    # Process merges
    for i in range(n_merges):
        new_cluster_id = n_samples + i
        left = int(linkage_matrix[i, 0])
        right = int(linkage_matrix[i, 1])
        
        left_child[new_cluster_id] = left
        right_child[new_cluster_id] = right
        parent[left] = new_cluster_id
        parent[right] = new_cluster_id
        
        class_counts[new_cluster_id] = class_counts[left] + class_counts[right]

    # Group indices by class
    class_indices = {}
    for label in unique_labels:
        mask = (labels == label)
        class_indices[label.item()] = torch.nonzero(mask, as_tuple=True)[0].tolist()

    sum_purity = 0.0

    # Compute purity for each pair
    for lbl, indices in class_indices.items():
        count = len(indices)
        if count < 2:
            continue
        
        lbl_idx = label_to_idx[lbl]
        
        for i in range(count):
            for j in range(i + 1, count):
                idx_i = indices[i]
                idx_j = indices[j]
                
                # Find LCA
                path_i = []
                current = idx_i
                while current != -1:
                    path_i.append(current)
                    current = parent[current]
                
                path_j_set = set()
                current = idx_j
                while current != -1:
                    path_j_set.add(current)
                    current = parent[current]
                
                # Find first common node
                lca_node = None
                for node in path_i:
                    if node in path_j_set:
                        lca_node = node
                        break
                
                if lca_node is None:
                    continue
                
                total = class_counts[lca_node].sum().item()
                purity = class_counts[lca_node, lbl_idx].item() / total if total != 0 else 0.0
                sum_purity += purity

    return sum_purity / total_pairs