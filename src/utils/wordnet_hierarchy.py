"""
Utility functions for loading and processing WordNet hierarchies for CIFAR-100.

The WordNet hierarchy from the semantic-embeddings repository provides a 
parent-child structure for CIFAR-100 classes:
- Nodes 0-99: CIFAR-100 fine-grained classes (leaf nodes)
- Nodes 100-162: WordNet parent categories (63 intermediate/parent nodes)

This allows for more fine-grained hierarchical evaluation compared to 
the standard CIFAR-100 coarse labels (20 superclasses).
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import requests
from pathlib import Path


def download_wordnet_hierarchy(save_path: Optional[str] = None) -> str:
    """
    Download the CIFAR-100 WordNet parent-child hierarchy from GitHub.
    
    Parameters
    ----------
    save_path : Optional[str]
        Path where to save the downloaded file. If None, saves to a default location.
        
    Returns
    -------
    str
        Path to the downloaded file
    """
    url = "https://raw.githubusercontent.com/cvjena/semantic-embeddings/master/Cifar-Hierarchy/cifar.parent-child.txt"
    
    if save_path is None:
        # Default save location in data directory
        save_path = "./data/cifar100_wordnet_hierarchy.txt"
    
    save_path_obj = Path(save_path)
    save_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # Download the file
    response = requests.get(url)
    response.raise_for_status()
    
    with open(save_path_obj, 'w') as f:
        f.write(response.text)
    
    return str(save_path_obj)


def load_wordnet_hierarchy(filepath: str) -> Dict[int, List[int]]:
    """
    Load WordNet parent-child hierarchy from file.
    
    The file format is: parent_id child_id (one pair per line)
    
    Parameters
    ----------
    filepath : str
        Path to the parent-child hierarchy file
        
    Returns
    -------
    Dict[int, List[int]]
        Dictionary mapping parent node IDs to lists of child node IDs
    """
    parent_child = {}
    
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                parent, child = int(parts[0]), int(parts[1])
                if parent not in parent_child:
                    parent_child[parent] = []
                parent_child[parent].append(child)
    
    return parent_child


def build_wordnet_label_mapping(
    parent_child: Dict[int, List[int]], 
    fine_labels: np.ndarray
) -> np.ndarray:
    """
    Build WordNet-based hierarchical labels for dendrogram purity evaluation.
    
    This function maps each CIFAR-100 fine-grained class (0-99) to its WordNet
    parent categories. For hierarchical dendrogram purity, we need labels that
    capture the hierarchical structure at multiple levels.
    
    Parameters
    ----------
    parent_child : Dict[int, List[int]]
        Parent-child hierarchy dictionary
    fine_labels : np.ndarray
        Array of fine-grained labels (0-99) for each sample
        
    Returns
    -------
    np.ndarray
        Array of WordNet parent labels for hierarchical evaluation.
        Maps each fine class (0-99) to its immediate parent in the WordNet hierarchy.
    """
    # Build reverse mapping: child -> parent
    child_to_parent = {}
    for parent, children in parent_child.items():
        for child in children:
            # Only store the immediate parent
            if child not in child_to_parent:
                child_to_parent[child] = parent
    
    # Map fine labels to their WordNet parent labels
    wordnet_labels = np.zeros_like(fine_labels)
    for i, fine_label in enumerate(fine_labels):
        if fine_label in child_to_parent:
            # Map to immediate parent
            wordnet_labels[i] = child_to_parent[fine_label]
        else:
            # If no parent found, keep the fine label (shouldn't happen for 0-99)
            wordnet_labels[i] = fine_label
    
    return wordnet_labels


def get_wordnet_labels_for_cifar100(
    fine_labels: np.ndarray,
    hierarchy_file: Optional[str] = None,
    download_if_missing: bool = True
) -> Tuple[np.ndarray, Dict[int, List[int]]]:
    """
    Convenience function to get WordNet labels for CIFAR-100 fine labels.
    
    This function handles downloading the hierarchy file if needed and
    builds the WordNet parent labels for hierarchical evaluation.
    
    Parameters
    ----------
    fine_labels : np.ndarray
        Array of CIFAR-100 fine-grained labels (0-99)
    hierarchy_file : Optional[str]
        Path to the WordNet hierarchy file. If None, uses default location.
    download_if_missing : bool
        Whether to download the hierarchy file if it doesn't exist
        
    Returns
    -------
    Tuple[np.ndarray, Dict[int, List[int]]]
        - WordNet parent labels for each sample
        - Parent-child hierarchy dictionary
        
    Example
    -------
    >>> fine_labels = np.array([0, 1, 2, 15, 16, 17])  # Some CIFAR-100 classes
    >>> wordnet_labels, hierarchy = get_wordnet_labels_for_cifar100(fine_labels)
    >>> # Now use wordnet_labels as Y_coarse in hierarchical_dendrogram_purity
    """
    if hierarchy_file is None:
        hierarchy_file = "./data/cifar100_wordnet_hierarchy.txt"
    
    # Download if file doesn't exist and download is enabled
    if not Path(hierarchy_file).exists():
        if download_if_missing:
            print(f"WordNet hierarchy file not found. Downloading to {hierarchy_file}...")
            hierarchy_file = download_wordnet_hierarchy(hierarchy_file)
        else:
            raise FileNotFoundError(
                f"WordNet hierarchy file not found: {hierarchy_file}. "
                "Set download_if_missing=True to download automatically."
            )
    
    # Load hierarchy
    parent_child = load_wordnet_hierarchy(hierarchy_file)
    
    # Build WordNet labels
    wordnet_labels = build_wordnet_label_mapping(parent_child, fine_labels)
    
    return wordnet_labels, parent_child


def get_hierarchy_depth(parent_child: Dict[int, List[int]], node: int) -> int:
    """
    Calculate the depth of a node in the hierarchy (for analysis purposes).
    
    Parameters
    ----------
    parent_child : Dict[int, List[int]]
        Parent-child hierarchy dictionary
    node : int
        Node ID to calculate depth for
        
    Returns
    -------
    int
        Depth of the node (0 for leaf nodes, increases for parent nodes)
    """
    # If node has children, it's a parent
    if node in parent_child:
        # Depth is 1 + max depth of children
        child_depths = [get_hierarchy_depth(parent_child, child) for child in parent_child[node]]
        return 1 + max(child_depths)
    else:
        # Leaf node
        return 0


def print_hierarchy_statistics(parent_child: Dict[int, List[int]]) -> None:
    """
    Print statistics about the WordNet hierarchy (for debugging/analysis).
    
    Parameters
    ----------
    parent_child : Dict[int, List[int]]
        Parent-child hierarchy dictionary
    """
    # Find all nodes
    all_nodes = set(parent_child.keys())
    for children in parent_child.values():
        all_nodes.update(children)
    
    # Find leaf nodes (CIFAR-100 classes)
    leaf_nodes = all_nodes - set(parent_child.keys())
    
    print("WordNet Hierarchy Statistics:")
    print(f"  Total nodes: {len(all_nodes)}")
    print(f"  Leaf nodes (CIFAR-100 classes): {len(leaf_nodes)} (range: {min(leaf_nodes)}-{max(leaf_nodes)})")
    print(f"  Parent nodes (WordNet categories): {len(parent_child)} (range: {min(parent_child.keys())}-{max(parent_child.keys())})")
    
    # Calculate max depth
    max_depth = max(get_hierarchy_depth(parent_child, node) for node in parent_child.keys())
    print(f"  Maximum hierarchy depth: {max_depth}")
    
    # Distribution of children per parent
    children_counts = [len(children) for children in parent_child.values()]
    print(f"  Avg children per parent: {np.mean(children_counts):.2f}")
    print(f"  Max children per parent: {max(children_counts)}")
    print(f"  Min children per parent: {min(children_counts)}")
