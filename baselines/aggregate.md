# Aggregate eval results

## scan-cifar100

- dataset: `cifar100`  metric: `cosine`  sample_size: 10000  seeds: [42, 56, 99] (n=3)

| metric | mean | std | n |
|---|---:|---:|---:|
| test/10_nearest_neighbors_accuracy | 0.4483 | 0.0000 | 3 |
| test/1_nearest_neighbors_accuracy | 0.4120 | 0.0000 | 3 |
| test/5_nearest_neighbors_accuracy | 0.4383 | 0.0000 | 3 |
| test/agglomerative_clustering_ward_acc | 0.2574 | 0.0000 | 3 |
| test/agglomerative_clustering_ward_ari | 0.1817 | 0.0000 | 3 |
| test/agglomerative_clustering_ward_nmi | 0.5035 | 0.0000 | 3 |
| test/all_dendrogram_purity_complete | 0.2196 | 0.0000 | 3 |
| test/all_dendrogram_purity_ward | 0.2389 | 0.0000 | 3 |
| test/coarse_1_nearest_neighbors_accuracy | 0.6076 | 0.0000 | 3 |
| test/coarse_recall_at_2 | 0.7158 | 0.0000 | 3 |
| test/coarse_recall_at_4 | 0.8117 | 0.0000 | 3 |
| test/coarse_recall_at_8 | 0.8879 | 0.0000 | 3 |
| test/dendrogram_purity_complete | 0.1636 | 0.0000 | 3 |
| test/dendrogram_purity_ward | 0.1807 | 0.0000 | 3 |
| test/hierarchical_dendrogram_purity_complete | 0.3295 | 0.0000 | 3 |
| test/hierarchical_dendrogram_purity_ward | 0.3531 | 0.0000 | 3 |
| test/recall_at_10 | 0.7737 | 0.0000 | 3 |
| test/recall_at_100 | 0.9627 | 0.0000 | 3 |
| test/recall_at_2 | 0.5232 | 0.0000 | 3 |
| test/recall_at_4 | 0.6394 | 0.0000 | 3 |
| test/recall_at_8 | 0.7447 | 0.0000 | 3 |

## secu-cifar100-dim128

- dataset: `cifar100`  metric: `cosine`  sample_size: 10000  seeds: [42, 99] (n=2)

| metric | mean | std | n |
|---|---:|---:|---:|
| test/10_nearest_neighbors_accuracy | 0.5088 | 0.0547 | 2 |
| test/1_nearest_neighbors_accuracy | 0.4754 | 0.0591 | 2 |
| test/5_nearest_neighbors_accuracy | 0.5034 | 0.0581 | 2 |
| test/agglomerative_clustering_ward_acc | 0.3823 | 0.0446 | 2 |
| test/agglomerative_clustering_ward_ari | 0.2441 | 0.0409 | 2 |
| test/agglomerative_clustering_ward_nmi | 0.5635 | 0.0412 | 2 |
| test/all_dendrogram_purity_complete | 0.2638 | 0.0710 | 2 |
| test/all_dendrogram_purity_ward | 0.2923 | 0.0509 | 2 |
| test/coarse_1_nearest_neighbors_accuracy | 0.6552 | 0.0622 | 2 |
| test/coarse_recall_at_2 | 0.7489 | 0.0491 | 2 |
| test/coarse_recall_at_4 | 0.8255 | 0.0347 | 2 |
| test/coarse_recall_at_8 | 0.8916 | 0.0207 | 2 |
| test/dendrogram_purity_complete | 0.2509 | 0.0649 | 2 |
| test/dendrogram_purity_ward | 0.2622 | 0.0369 | 2 |
| test/hierarchical_dendrogram_purity_complete | 0.3011 | 0.0658 | 2 |
| test/hierarchical_dendrogram_purity_ward | 0.3798 | 0.0380 | 2 |
| test/recall_at_10 | 0.7950 | 0.0409 | 2 |
| test/recall_at_100 | 0.9667 | 0.0057 | 2 |
| test/recall_at_2 | 0.5836 | 0.0597 | 2 |
| test/recall_at_4 | 0.6805 | 0.0533 | 2 |
| test/recall_at_8 | 0.7697 | 0.0455 | 2 |

## secu-cifar100-dim16

- dataset: `cifar100`  metric: `cosine`  sample_size: 10000  seeds: [42, 56, 99] (n=3)

| metric | mean | std | n |
|---|---:|---:|---:|
| test/10_nearest_neighbors_accuracy | 0.5088 | 0.0099 | 3 |
| test/1_nearest_neighbors_accuracy | 0.4686 | 0.0099 | 3 |
| test/5_nearest_neighbors_accuracy | 0.4979 | 0.0084 | 3 |
| test/agglomerative_clustering_ward_acc | 0.3976 | 0.0101 | 3 |
| test/agglomerative_clustering_ward_ari | 0.2600 | 0.0093 | 3 |
| test/agglomerative_clustering_ward_nmi | 0.5842 | 0.0072 | 3 |
| test/all_dendrogram_purity_complete | 0.3053 | 0.0128 | 3 |
| test/all_dendrogram_purity_ward | 0.3232 | 0.0072 | 3 |
| test/coarse_1_nearest_neighbors_accuracy | 0.6655 | 0.0071 | 3 |
| test/coarse_recall_at_2 | 0.7587 | 0.0064 | 3 |
| test/coarse_recall_at_4 | 0.8337 | 0.0020 | 3 |
| test/coarse_recall_at_8 | 0.8973 | 0.0062 | 3 |
| test/dendrogram_purity_complete | 0.2693 | 0.0109 | 3 |
| test/dendrogram_purity_ward | 0.2760 | 0.0059 | 3 |
| test/hierarchical_dendrogram_purity_complete | 0.3758 | 0.0170 | 3 |
| test/hierarchical_dendrogram_purity_ward | 0.4156 | 0.0109 | 3 |
| test/recall_at_10 | 0.8005 | 0.0037 | 3 |
| test/recall_at_100 | 0.9682 | 0.0006 | 3 |
| test/recall_at_2 | 0.5802 | 0.0070 | 3 |
| test/recall_at_4 | 0.6795 | 0.0033 | 3 |
| test/recall_at_8 | 0.7743 | 0.0037 | 3 |

## secu-cifar100-dim32

- dataset: `cifar100`  metric: `cosine`  sample_size: 10000  seeds: [42, 56, 99] (n=3)

| metric | mean | std | n |
|---|---:|---:|---:|
| test/10_nearest_neighbors_accuracy | 0.5421 | 0.0047 | 3 |
| test/1_nearest_neighbors_accuracy | 0.5090 | 0.0082 | 3 |
| test/5_nearest_neighbors_accuracy | 0.5358 | 0.0051 | 3 |
| test/agglomerative_clustering_ward_acc | 0.4198 | 0.0040 | 3 |
| test/agglomerative_clustering_ward_ari | 0.2812 | 0.0066 | 3 |
| test/agglomerative_clustering_ward_nmi | 0.5983 | 0.0045 | 3 |
| test/all_dendrogram_purity_complete | 0.3261 | 0.0115 | 3 |
| test/all_dendrogram_purity_ward | 0.3313 | 0.0061 | 3 |
| test/coarse_1_nearest_neighbors_accuracy | 0.6905 | 0.0111 | 3 |
| test/coarse_recall_at_2 | 0.7765 | 0.0079 | 3 |
| test/coarse_recall_at_4 | 0.8468 | 0.0034 | 3 |
| test/coarse_recall_at_8 | 0.9004 | 0.0014 | 3 |
| test/dendrogram_purity_complete | 0.2960 | 0.0082 | 3 |
| test/dendrogram_purity_ward | 0.2887 | 0.0043 | 3 |
| test/hierarchical_dendrogram_purity_complete | 0.3850 | 0.0180 | 3 |
| test/hierarchical_dendrogram_purity_ward | 0.4147 | 0.0119 | 3 |
| test/recall_at_10 | 0.8187 | 0.0042 | 3 |
| test/recall_at_100 | 0.9703 | 0.0007 | 3 |
| test/recall_at_2 | 0.6150 | 0.0090 | 3 |
| test/recall_at_4 | 0.7118 | 0.0057 | 3 |
| test/recall_at_8 | 0.7953 | 0.0055 | 3 |

## secu-tiny_imagenet

- dataset: `tiny_imagenet`  metric: `cosine`  sample_size: 10000  seeds: [42, 56, 99] (n=3)

| metric | mean | std | n |
|---|---:|---:|---:|
| test/10_nearest_neighbors_accuracy | 0.0866 | 0.0714 | 3 |
| test/1_nearest_neighbors_accuracy | 0.0688 | 0.0559 | 3 |
| test/5_nearest_neighbors_accuracy | 0.0754 | 0.0613 | 3 |
| test/agglomerative_clustering_ward_acc | 0.0874 | 0.0712 | 3 |
| test/agglomerative_clustering_ward_ari | 0.0330 | 0.0286 | 3 |
| test/agglomerative_clustering_ward_nmi | 0.2962 | 0.2178 | 3 |
| test/all_dendrogram_purity_complete | 0.0714 | 0.0098 | 3 |
| test/all_dendrogram_purity_ward | 0.1428 | 0.1047 | 3 |
| test/coarse_1_nearest_neighbors_accuracy | 0.1769 | 0.1452 | 3 |
| test/coarse_recall_at_2 | 0.2720 | 0.2273 | 3 |
| test/coarse_recall_at_4 | 0.3844 | 0.3243 | 3 |
| test/coarse_recall_at_8 | 0.4955 | 0.4205 | 3 |
| test/dendrogram_purity_complete | 0.0361 | 0.0128 | 3 |
| test/dendrogram_purity_ward | 0.1311 | 0.1488 | 3 |
| test/hierarchical_dendrogram_purity_complete | 0.1230 | 0.0087 | 3 |
| test/hierarchical_dendrogram_purity_ward | 0.1600 | 0.0404 | 3 |
| test/recall_at_10 | 0.2681 | 0.2283 | 3 |
| test/recall_at_100 | 0.5566 | 0.4647 | 3 |
| test/recall_at_2 | 0.1092 | 0.0907 | 3 |
| test/recall_at_4 | 0.1665 | 0.1405 | 3 |
| test/recall_at_8 | 0.2404 | 0.2045 | 3 |
