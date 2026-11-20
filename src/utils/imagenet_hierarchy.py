"""
Utility functions for loading and processing WordNet hierarchies for ImageNet.

The WordNet hierarchy from the cvjena/semantic-embeddings repository provides a 
parent-child structure for ImageNet classes:
- Uses WordNet synset IDs (e.g., n01440764)
- Need to map synset IDs to ImageNet class indices (0-999)
- Parent synsets form intermediate nodes in the hierarchy

This allows for hierarchical evaluation (Dendrogram Purity) that captures
semantic similarity between ImageNet classes based on WordNet.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import requests
from pathlib import Path


# WordNet hierarchy from cvjena/semantic-embeddings
# This is the official hierarchy file used by the semantic-embeddings paper
WORDNET_HIERARCHY_URL = "https://raw.githubusercontent.com/cvjena/semantic-embeddings/master/ILSVRC/wordnet.parent-child.mintree.txt"

# ImageNet synset to class index mapping URL (from cvjena/semantic-embeddings)
# The synset mapping can be derived from the ImageNet folder structure, but we provide
# a fallback URL from the same repo's class list for consistency
# Note: cvjena uses folder names directly, so we derive from the hierarchy leaves
IMAGENET_SYNSETS_URL = "https://raw.githubusercontent.com/cvjena/semantic-embeddings/master/datasets/ilsvrc.py"


def download_file(url: str, save_path: str) -> str:
    """
    Download a file from a URL.
    
    Parameters
    ----------
    url : str
        URL to download from
    save_path : str
        Path where to save the downloaded file
        
    Returns
    -------
    str
        Path to the downloaded file
    """
    save_path_obj = Path(save_path)
    save_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    response = requests.get(url)
    response.raise_for_status()
    
    with open(save_path_obj, 'w') as f:
        f.write(response.text)
    
    return str(save_path_obj)


def download_imagenet_synset_mapping(save_path: Optional[str] = None) -> str:
    """
    Download the ImageNet synset to index mapping.
    
    Uses the AWS S3 hosted imagenet_class_index.json which contains synset to 
    class index mapping in JSON format.
    
    Parameters
    ----------
    save_path : Optional[str]
        Path where to save the mapping. If None, uses default location.
        
    Returns
    -------
    str
        Path to the downloaded file
    """
    
    if save_path is None:
        save_path = "./data/imagenet_synset_to_idx.txt"
    
    # AWS S3 hosted ImageNet class index (reliable, widely used)
    s3_url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    
    try:
        response = requests.get(s3_url)
        response.raise_for_status()
        class_index = response.json()
        
        # Convert JSON to simple synset list (one per line, in order)
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path_obj, 'w') as f:
            for idx in range(len(class_index)):
                synset = class_index[str(idx)][0]
                f.write(f"{synset}\n")
        
        return str(save_path_obj)
    except Exception as e:
        raise RuntimeError(
            f"Could not download ImageNet synset mapping from {s3_url}: {e}. "
            "Please ensure you have ImageNet data downloaded and use "
            "build_synset_list_from_imagenet_folder() instead."
        )


def build_synset_list_from_imagenet_folder(
    imagenet_train_dir: str,
    save_path: Optional[str] = None
) -> str:
    """
    Build synset mapping from local ImageNet train folder structure.
    
    ImageNet stores images in folders named by synset ID (e.g., n01440764/).
    The alphabetical order of these folder names gives the standard class order.
    
    Parameters
    ----------
    imagenet_train_dir : str
        Path to ILSVRC2012_img_train directory
    save_path : Optional[str]
        Path to save the synset list. If None, uses default location.
        
    Returns
    -------
    str
        Path to the saved synset list file
    """
    
    if save_path is None:
        save_path = "./data/imagenet_synset_to_idx.txt"
    
    train_path = Path(imagenet_train_dir)
    if not train_path.exists():
        raise FileNotFoundError(f"ImageNet train directory not found: {imagenet_train_dir}")
    
    # Get all synset folder names and sort alphabetically
    synsets = sorted([
        d.name for d in train_path.iterdir()
        if d.is_dir() and d.name.startswith('n')
    ])
    
    if len(synsets) not in (200, 1000):
        print(f"Warning: Found {len(synsets)} synsets, expected 200 (Tiny ImageNet) or 1000 (ImageNet)")
    
    # Save to file
    save_path_obj = Path(save_path)
    save_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path_obj, 'w') as f:
        for synset in synsets:
            f.write(f"{synset}\n")
    
    return str(save_path_obj)


def download_imagenet_wordnet_hierarchy(save_path: Optional[str] = None) -> str:
    """
    Download the ImageNet WordNet parent-child hierarchy from cvjena/semantic-embeddings.
    
    This hierarchy file uses WordNet synset IDs and provides a proper tree structure
    (the mintree version ensures it's a tree, not a DAG).
    
    Parameters
    ----------
    save_path : Optional[str]
        Path where to save the downloaded file. If None, uses default location.
        
    Returns
    -------
    str
        Path to the downloaded file
    """
    if save_path is None:
        save_path = "./data/imagenet_wordnet_hierarchy_raw.txt"
    
    return download_file(WORDNET_HIERARCHY_URL, save_path)


def load_synset_to_idx_mapping(filepath: str) -> Dict[str, int]:
    """
    Load the synset to ImageNet class index mapping.
    
    Supports multiple formats:
    - One synset per line (in order of class index)
    - LOC_synset_mapping format: "synset_id description" per line
    
    Parameters
    ----------
    filepath : str
        Path to the synset file
        
    Returns
    -------
    Dict[str, int]
        Dictionary mapping synset IDs to class indices (0-999)
    """
    synset_to_idx = {}
    
    with open(filepath, 'r') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            # Handle LOC_synset_mapping.txt format: "n01440764 tench, Tinca tinca"
            # or simple format: "n01440764"
            synset = line.split()[0] if ' ' in line else line
            synset_to_idx[synset] = idx
    
    return synset_to_idx


def load_synset_hierarchy(filepath: str) -> Dict[str, List[str]]:
    """
    Load the WordNet hierarchy with synset IDs.
    
    Parameters
    ----------
    filepath : str
        Path to the parent-child hierarchy file (synset format)
        
    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping parent synset IDs to lists of child synset IDs
    """
    parent_child = {}
    
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                parent, child = parts[0], parts[1]
                if parent not in parent_child:
                    parent_child[parent] = []
                parent_child[parent].append(child)
    
    return parent_child


def build_imagenet_hierarchy_file(
    synset_to_idx: Dict[str, int],
    synset_hierarchy: Dict[str, List[str]],
    output_path: str
) -> str:
    """
    Build an ImageNet hierarchy file in numeric format (similar to CIFAR-100).
    
    The format is:
    - Class indices 0-999: ImageNet leaf classes
    - Class indices 1000+: WordNet parent categories
    
    Parameters
    ----------
    synset_to_idx : Dict[str, int]
        Mapping from synset IDs to ImageNet class indices
    synset_hierarchy : Dict[str, List[str]]
        Parent-child hierarchy with synset IDs
    output_path : str
        Path to save the output file
        
    Returns
    -------
    str
        Path to the output file
    """
    # Build complete node ID mapping
    # Start with ImageNet class synsets (indices 0-999)
    node_to_idx = dict(synset_to_idx)  # Copy the mapping
    
    # Find all synsets that appear as parents (intermediate nodes)
    all_parents = set(synset_hierarchy.keys())
    all_children = set()
    for children in synset_hierarchy.values():
        all_children.update(children)
    
    # Intermediate nodes are parents that are not ImageNet classes
    intermediate_nodes = all_parents - set(synset_to_idx.keys())
    # Also add any children that are not ImageNet classes (higher-level parents)
    intermediate_nodes.update(all_children - set(synset_to_idx.keys()))
    
    # Assign IDs to intermediate nodes (starting after the leaf classes)
    next_idx = len(synset_to_idx)
    for synset in sorted(intermediate_nodes):
        if synset not in node_to_idx:
            node_to_idx[synset] = next_idx
            next_idx += 1
    
    # Write the hierarchy file
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path_obj, 'w') as f:
        for parent_synset, children_synsets in synset_hierarchy.items():
            parent_idx = node_to_idx.get(parent_synset)
            if parent_idx is None:
                continue
                
            for child_synset in children_synsets:
                child_idx = node_to_idx.get(child_synset)
                if child_idx is None:
                    continue
                f.write(f"{parent_idx} {child_idx}\n")

    return str(output_path_obj)


def setup_imagenet_hierarchy(
    output_path: Optional[str] = None,
    force_download: bool = False,
    imagenet_train_dir: Optional[str] = None
) -> str:
    """
    Set up the ImageNet WordNet hierarchy file.
    
    Downloads required files from cvjena/semantic-embeddings and builds the numeric hierarchy.
    
    Parameters
    ----------
    output_path : Optional[str]
        Path for the final hierarchy file. If None, uses default location.
    force_download : bool
        If True, re-download files even if they exist
    imagenet_train_dir : Optional[str]
        Path to ILSVRC2012_img_train directory. Used as fallback for synset mapping
        if download fails.
        
    Returns
    -------
    str
        Path to the hierarchy file
    """
    if output_path is None:
        output_path = "./data/imagenet_wordnet_hierarchy.txt"
    
    # Check if output already exists
    if Path(output_path).exists() and not force_download:
        return output_path
    
    # Download synset mapping
    synset_path = "./data/imagenet_synset_to_idx.txt"
    if not Path(synset_path).exists() or force_download:
        print("Downloading ImageNet synset mapping...")
        try:
            download_imagenet_synset_mapping(synset_path)
        except Exception as e:
            if imagenet_train_dir:
                print(f"Download failed ({e}), building from local ImageNet folder...")
                build_synset_list_from_imagenet_folder(imagenet_train_dir, synset_path)
            else:
                raise
    
    # Download WordNet hierarchy from cvjena/semantic-embeddings
    hierarchy_raw_path = "./data/imagenet_wordnet_hierarchy_raw.txt"
    if not Path(hierarchy_raw_path).exists() or force_download:
        print("Downloading ImageNet WordNet hierarchy from cvjena/semantic-embeddings...")
        download_imagenet_wordnet_hierarchy(hierarchy_raw_path)
    
    # Load mappings
    synset_to_idx = load_synset_to_idx_mapping(synset_path)
    synset_hierarchy = load_synset_hierarchy(hierarchy_raw_path)
    
    # Build numeric hierarchy
    print(f"Building ImageNet hierarchy file: {output_path}")
    build_imagenet_hierarchy_file(synset_to_idx, synset_hierarchy, output_path)
    
    return output_path


def load_imagenet_hierarchy(
    filepath: str,
    exclude_root: bool = True
) -> Dict[int, List[int]]:
    """
    Load ImageNet WordNet parent-child hierarchy from file.
    
    Parameters
    ----------
    filepath : str
        Path to the parent-child hierarchy file (numeric format)
    exclude_root : bool
        If True, removes the single root node from the hierarchy to avoid
        artificially inflating DP scores (since all samples share the root).
        Default is True.
        
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
    
    if exclude_root:
        # Find root nodes (parents that are not children of anyone)
        all_parents = set(parent_child.keys())
        all_children = set()
        for children in parent_child.values():
            all_children.update(children)
        roots = all_parents - all_children
        
        # If there's a single root, remove it
        if len(roots) == 1:
            root = roots.pop()
            del parent_child[root]
    
    return parent_child


def build_imagenet_label_mapping(
    parent_child: Dict[int, List[int]], 
    fine_labels: np.ndarray
) -> np.ndarray:
    """
    Build WordNet-based hierarchical labels for ImageNet.
    
    Maps each ImageNet class (0-999) to its WordNet parent category.
    
    Parameters
    ----------
    parent_child : Dict[int, List[int]]
        Parent-child hierarchy dictionary
    fine_labels : np.ndarray
        Array of ImageNet labels (0-999) for each sample
        
    Returns
    -------
    np.ndarray
        Array of WordNet parent labels for hierarchical evaluation
    """
    # Build reverse mapping: child -> parent
    child_to_parent = {}
    for parent, children in parent_child.items():
        for child in children:
            if child not in child_to_parent:
                child_to_parent[child] = parent
    
    # Map fine labels to their WordNet parent labels
    wordnet_labels = np.zeros_like(fine_labels)
    for i, fine_label in enumerate(fine_labels):
        if fine_label in child_to_parent:
            wordnet_labels[i] = child_to_parent[fine_label]
        else:
            # If no parent found, keep the fine label
            wordnet_labels[i] = fine_label
    
    return wordnet_labels


def get_wordnet_labels_for_imagenet(
    fine_labels: np.ndarray,
    hierarchy_file: Optional[str] = None,
    setup_if_missing: bool = True,
    exclude_root: bool = True
) -> Tuple[np.ndarray, Dict[int, List[int]]]:
    """
    Get WordNet labels for ImageNet fine labels.
    
    Parameters
    ----------
    fine_labels : np.ndarray
        Array of ImageNet labels (0-999)
    hierarchy_file : Optional[str]
        Path to the WordNet hierarchy file. If None, uses default location.
    setup_if_missing : bool
        Whether to set up the hierarchy file if it doesn't exist
    exclude_root : bool
        If True, removes the single root node from the hierarchy to avoid
        artificially inflating DP scores. Default is True.
        
    Returns
    -------
    Tuple[np.ndarray, Dict[int, List[int]]]
        - WordNet parent labels for each sample
        - Parent-child hierarchy dictionary
    """
    if hierarchy_file is None:
        hierarchy_file = "./data/imagenet_wordnet_hierarchy.txt"
    
    # Set up if file doesn't exist
    if not Path(hierarchy_file).exists():
        if setup_if_missing:
            print(f"ImageNet hierarchy file not found. Setting up: {hierarchy_file}")
            setup_imagenet_hierarchy(hierarchy_file)
        else:
            raise FileNotFoundError(
                f"ImageNet hierarchy file not found: {hierarchy_file}. "
                "Set setup_if_missing=True to download and build automatically."
            )
    
    # Load hierarchy (exclude_root=True by default)
    parent_child = load_imagenet_hierarchy(hierarchy_file, exclude_root=exclude_root)
    
    # Build WordNet labels
    wordnet_labels = build_imagenet_label_mapping(parent_child, fine_labels)
    
    return wordnet_labels, parent_child


def get_labels_at_hierarchy_level(
    fine_labels: np.ndarray,
    parent_child: Dict[int, List[int]],
    target_num_classes: int = 50
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Map fine-grained labels to a higher hierarchy level with approximately
    *target_num_classes* using a **top-down greedy split**.

    Starting from the root(s), the algorithm repeatedly expands the active
    node that covers the most leaf classes until the target is reached.  This
    produces balanced, semantically meaningful superclasses (e.g. "animal",
    "food", "artifact") instead of the overly abstract categories that a
    uniform bottom-up approach yields.

    Parameters
    ----------
    fine_labels : np.ndarray
        Array of fine-grained labels (0-999 for ImageNet, 0-199 for
        Tiny ImageNet).
    parent_child : Dict[int, List[int]]
        Parent-child hierarchy dictionary.
    target_num_classes : int
        Approximate target number of classes for the output.

    Returns
    -------
    Tuple[np.ndarray, Dict[int, int]]
        - Coarse labels at the selected hierarchy level
        - Mapping from fine labels to coarse labels
    """
    unique_fine = set(np.unique(fine_labels).tolist())

    # Pre-compute leaf descendants for every node (cached).
    _desc_cache: Dict[int, set] = {}

    def _leaf_descendants(node: int) -> set:
        if node in _desc_cache:
            return _desc_cache[node]
        if node not in parent_child:
            _desc_cache[node] = {node}
            return _desc_cache[node]
        result: set = set()
        for child in parent_child[node]:
            result |= _leaf_descendants(child)
        _desc_cache[node] = result
        return result

    # Find root nodes (parents that are not children of anyone).
    all_children_set: set = set()
    for ch in parent_child.values():
        all_children_set.update(ch)
    roots = set(parent_child.keys()) - all_children_set

    # Active nodes partition the fine labels.  Start from the root(s).
    active_nodes = set(roots)

    # Greedily expand the most populated node until we reach the target.
    while len(active_nodes) < target_num_classes:
        # Score each expandable active node by how many fine labels it covers.
        best_node = None
        best_count = -1
        for node in active_nodes:
            if node not in parent_child:
                continue  # leaf – cannot expand
            count = len(_leaf_descendants(node) & unique_fine)
            if count > best_count:
                best_count = count
                best_node = node

        if best_node is None:
            break  # nothing left to expand

        children = parent_child[best_node]
        new_count = len(active_nodes) - 1 + len(children)
        if new_count > target_num_classes:
            # Expanding would overshoot – stop here.
            break

        active_nodes.remove(best_node)
        active_nodes.update(children)

    # Build fine_label → active_node mapping.
    label_to_active: Dict[int, int] = {}
    for node in active_nodes:
        for leaf in _leaf_descendants(node):
            if leaf in unique_fine:
                label_to_active[leaf] = node

    coarse_labels = np.array([label_to_active.get(int(l), int(l)) for l in fine_labels])
    return coarse_labels, label_to_active


def print_imagenet_hierarchy_statistics(parent_child: Dict[int, List[int]]) -> None:
    """
    Print statistics about the ImageNet WordNet hierarchy.
    
    Parameters
    ----------
    parent_child : Dict[int, List[int]]
        Parent-child hierarchy dictionary
    """
    # Find all nodes
    all_nodes = set(parent_child.keys())
    for children in parent_child.values():
        all_nodes.update(children)
    
    # Find leaf nodes (nodes that never appear as parents)
    all_children = set()
    for children in parent_child.values():
        all_children.update(children)
    leaf_nodes = [n for n in all_children if n not in parent_child]
    parent_nodes = list(parent_child.keys())
    
    print("ImageNet WordNet Hierarchy Statistics:")
    print(f"  Total nodes: {len(all_nodes)}")
    print(f"  Leaf nodes (ImageNet classes): {len(leaf_nodes)}")
    print(f"  Parent nodes (WordNet categories): {len(parent_nodes)}")
    
    if parent_nodes:
        print(f"  Parent node ID range: {min(parent_nodes)}-{max(parent_nodes)}")
    
    # Distribution of children per parent
    children_counts = [len(children) for children in parent_child.values()]
    print(f"  Avg children per parent: {np.mean(children_counts):.2f}")
    print(f"  Max children per parent: {max(children_counts)}")
    print(f"  Min children per parent: {min(children_counts)}")


def setup_tiny_imagenet_hierarchy(
    tiny_imagenet_dir: str,
    output_path: Optional[str] = None,
    force_download: bool = False,
) -> str:
    """
    Set up the WordNet hierarchy file for Tiny ImageNet (200-class subset).

    Downloads the full ImageNet WordNet hierarchy, then filters it to only
    include the 200 Tiny ImageNet leaf classes and their ancestors.

    Parameters
    ----------
    tiny_imagenet_dir : str
        Root directory of Tiny ImageNet dataset (containing wnids.txt or train/).
    output_path : Optional[str]
        Path for the output hierarchy file. If None, uses default location.
    force_download : bool
        If True, re-download/rebuild even if the output file exists.

    Returns
    -------
    str
        Path to the hierarchy file.
    """
    if output_path is None:
        output_path = "./data/tiny_imagenet_wordnet_hierarchy.txt"

    if Path(output_path).exists() and not force_download:
        return output_path

    # Read the 200 synset IDs
    tiny_dir = Path(tiny_imagenet_dir)
    wnids_file = tiny_dir / "wnids.txt"
    if wnids_file.exists():
        with open(wnids_file) as f:
            synsets = sorted(line.strip() for line in f if line.strip())
    else:
        synsets = sorted(
            d.name for d in (tiny_dir / "train").iterdir()
            if d.is_dir() and d.name.startswith("n")
        )

    # Build synset-to-index mapping for the 200 classes
    synset_to_idx = {s: i for i, s in enumerate(synsets)}

    # Download the full WordNet hierarchy
    hierarchy_raw_path = "./data/imagenet_wordnet_hierarchy_raw.txt"
    if not Path(hierarchy_raw_path).exists() or force_download:
        print("Downloading ImageNet WordNet hierarchy from cvjena/semantic-embeddings...")
        download_imagenet_wordnet_hierarchy(hierarchy_raw_path)

    synset_hierarchy = load_synset_hierarchy(hierarchy_raw_path)

    # Filter hierarchy to only include ancestors of the 200 leaf synsets.
    # Walk up from each leaf to find all relevant ancestor edges.
    # First, build child -> parent mapping from the full hierarchy
    child_to_parent_full: Dict[str, str] = {}
    for parent, children in synset_hierarchy.items():
        for child in children:
            child_to_parent_full[child] = parent

    # Find all ancestor synsets by walking up from each leaf
    relevant_synsets = set(synsets)
    for synset in synsets:
        current = synset
        while current in child_to_parent_full:
            parent = child_to_parent_full[current]
            relevant_synsets.add(parent)
            current = parent

    # Filter the hierarchy to only keep edges where both endpoints are relevant
    filtered_hierarchy: Dict[str, List[str]] = {}
    for parent, children in synset_hierarchy.items():
        if parent not in relevant_synsets:
            continue
        filtered_children = [c for c in children if c in relevant_synsets]
        if filtered_children:
            filtered_hierarchy[parent] = filtered_children

    # Build the numeric hierarchy file
    print(f"Building Tiny ImageNet hierarchy file ({len(synsets)} leaf classes): {output_path}")
    build_imagenet_hierarchy_file(synset_to_idx, filtered_hierarchy, output_path)

    return output_path


# Allow running as a script to set up the hierarchy
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Set up ImageNet WordNet hierarchy for hierarchical evaluation"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./data/imagenet_wordnet_hierarchy.txt",
        help="Output path for the hierarchy file"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print hierarchy statistics after setup"
    )
    
    args = parser.parse_args()
    
    output_path = setup_imagenet_hierarchy(args.output, force_download=args.force)
    print(f"ImageNet hierarchy file created: {output_path}")
    
    if args.stats:
        hierarchy = load_imagenet_hierarchy(output_path)
        print_imagenet_hierarchy_statistics(hierarchy)
