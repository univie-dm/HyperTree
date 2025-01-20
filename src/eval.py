from matplotlib import pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from typing import Tuple, List
from sklearn.decomposition import PCA
import torch

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

def compute_dendrogram_purity(
    distances: torch.Tensor,
    true_labels: torch.Tensor,
    method: str = 'complete',
    max_clusters: int = 20  # Limit number of clusters to evaluate
) -> float:
    """
    Compute the dendrogram purity metric more efficiently.
    """
    # Convert to numpy for scipy compatibility
    if isinstance(distances, torch.Tensor):
        distances = distances.cpu().numpy()
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.cpu().numpy()
    
    # Convert similarities to distances if needed
    if np.all(distances >= 0) and np.all(distances <= 1):
        distances = 1 - distances
    
    # Ensure the matrix is symmetric
    distances = (distances + distances.T) / 2
    
    # Compute linkage matrix
    condensed_dist = distances[np.triu_indices_from(distances, k=1)]
    Z = linkage(condensed_dist, method=method)
    
    # Get ground truth pairs
    true_pairs = get_cluster_pairs(true_labels)
    
    if len(true_pairs) == 0:
        return 1.0
    
    # Evaluate purity at different cut heights
    purities = []
    # Use fewer cut heights for efficiency
    cut_heights = np.unique(Z[:, 2])[::max(1, len(Z)//max_clusters)]
    
    for height in cut_heights:
        pred_labels = fcluster(Z, height, criterion='distance')
        pred_pairs = get_cluster_pairs(pred_labels)
        
        if len(pred_pairs) == 0:
            continue
            
        correct_pairs = len(true_pairs.intersection(pred_pairs))
        purity = correct_pairs / len(pred_pairs) if len(pred_pairs) > 0 else 0
        purities.append(purity)
    
    return np.mean(purities) if purities else 0.0


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