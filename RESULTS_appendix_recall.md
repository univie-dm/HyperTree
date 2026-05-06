# Appendix: Recall@K and kNN accuracy across all experiments

Aggregated from `results_baseline_embeddings/*.json` (frozen + external checkpoints) and wandb runs in `kevin-sidak-team/hyper_clustering` (tags filter: any of `paper_main, lca_ablation, hist_revert, viability, ours, manifold_ablation, lca_curv_sweep, poincare_d16_ours, poincare_d32_ours, hyperboloid`).

**Caveat:** `recall@K` is implemented in `src/evaluation.py` (gated by `evaluation_metrics.recall_at_k`) and recall_at_baseline_embeddings.py always computes it. Older runs and configs without the flag (notably iNat18 / ImageNet / TinyImageNet model runs prior to 2026-05-02) will show `—` in the R@K columns; the configs have since been patched. NN@k columns are kNN-majority accuracy (different metric — for k=1 it coincides with R@1).

## cifar100

| Source | Run / tag | hDP_ward | NMI_ward | NN@1 | NN@5 | NN@10 | NN@100 | R@2 | R@4 | R@8 | R@10 | R@100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | hier_c100_s1_d32_e5 | 0.896 | 0.746 | 0.708 | 0.684 | 0.671 | 0.527 | 0.795 | 0.862 | 0.906 | 0.917 | 0.987 |
| baseline | hier_c100_s3_d16_e5 | 0.851 | 0.719 | 0.658 | 0.639 | 0.626 | 0.482 | 0.756 | 0.826 | 0.884 | 0.899 | 0.987 |
| baseline | hier_c100_s2_d16_e75 | 0.851 | 0.810 | 0.761 | 0.756 | 0.752 | 0.645 | 0.834 | 0.881 | 0.916 | 0.924 | 0.986 |
| baseline | hier_c100_s2_d16_e5 | 0.851 | 0.717 | 0.653 | 0.634 | 0.621 | 0.475 | 0.750 | 0.828 | 0.886 | 0.901 | 0.985 |
| baseline | hier_c100_s3_d16_e30 | 0.838 | 0.804 | 0.762 | 0.752 | 0.749 | 0.641 | 0.834 | 0.884 | 0.916 | 0.925 | 0.985 |
| baseline | ours_c100_d16_s2_e40 | 0.836 | 0.594 | 0.510 | 0.489 | 0.474 | 0.308 | 0.614 | 0.701 | 0.773 | 0.794 | 0.937 |
| baseline | hier_c100_s1_d16_e5 | 0.834 | 0.717 | 0.657 | 0.636 | 0.623 | 0.476 | 0.753 | 0.829 | 0.883 | 0.898 | 0.984 |
| baseline | ours_c100_d16_s3_e45 | 0.829 | 0.571 | 0.504 | 0.483 | 0.467 | 0.303 | 0.607 | 0.690 | 0.765 | 0.786 | 0.935 |
| baseline | hier_c100_s2_d128_e5 | 0.829 | 0.768 | 0.741 | 0.720 | 0.707 | 0.563 | 0.818 | 0.870 | 0.912 | 0.922 | 0.986 |
| baseline | hier_c100_s2_d16_e15 | 0.828 | 0.784 | 0.737 | 0.726 | 0.720 | 0.603 | 0.816 | 0.870 | 0.907 | 0.919 | 0.986 |
| baseline | ours_c100_d16_s2_e50 | 0.827 | 0.579 | 0.500 | 0.473 | 0.452 | 0.232 | 0.596 | 0.683 | 0.760 | 0.782 | 0.941 |
| baseline | ours_c100_d16_s2_e45 | 0.826 | 0.582 | 0.507 | 0.481 | 0.465 | 0.273 | 0.605 | 0.692 | 0.770 | 0.790 | 0.938 |
| baseline | hier_c100_s2_d16_e90 | 0.825 | 0.809 | 0.761 | 0.756 | 0.752 | 0.645 | 0.834 | 0.881 | 0.916 | 0.925 | 0.986 |
| baseline | ours_c100_d16_s2_e35 | 0.816 | 0.593 | 0.512 | 0.493 | 0.477 | 0.315 | 0.615 | 0.703 | 0.776 | 0.798 | 0.934 |
| baseline | hier_c100_s3_d32_e15 | 0.814 | 0.806 | 0.775 | 0.763 | 0.758 | 0.644 | 0.842 | 0.887 | 0.918 | 0.926 | 0.987 |
| baseline | hier_c100_s2_d16_e30 | 0.814 | 0.806 | 0.759 | 0.753 | 0.749 | 0.641 | 0.831 | 0.878 | 0.914 | 0.924 | 0.985 |
| baseline | hier_c100_s3_d128_e5 | 0.813 | 0.762 | 0.740 | 0.718 | 0.706 | 0.560 | 0.819 | 0.874 | 0.911 | 0.923 | 0.987 |
| baseline | hier_c100_s2_d32_e95 | 0.812 | 0.818 | 0.791 | 0.784 | 0.784 | 0.676 | 0.854 | 0.892 | 0.921 | 0.928 | 0.985 |
| baseline | hier_c100_s3_d32_e5 | 0.809 | 0.745 | 0.705 | 0.683 | 0.671 | 0.532 | 0.791 | 0.856 | 0.904 | 0.917 | 0.987 |
| baseline | hier_c100_s2_d32_e15 | 0.809 | 0.807 | 0.770 | 0.765 | 0.759 | 0.644 | 0.842 | 0.888 | 0.917 | 0.926 | 0.985 |
| baseline | hier_c100_s2_d128_e45 | 0.806 | 0.823 | 0.806 | 0.801 | 0.797 | 0.684 | 0.863 | 0.899 | 0.923 | 0.929 | 0.987 |
| baseline | hier_c100_s1_d16_e30 | 0.804 | 0.806 | 0.759 | 0.750 | 0.747 | 0.641 | 0.829 | 0.879 | 0.912 | 0.922 | 0.984 |
| baseline | hier_c100_s3_d128_e15 | 0.804 | 0.814 | 0.795 | 0.788 | 0.783 | 0.664 | 0.856 | 0.894 | 0.924 | 0.930 | 0.988 |
| baseline | hier_c100_s2_d128_e15 | 0.802 | 0.810 | 0.796 | 0.784 | 0.780 | 0.660 | 0.858 | 0.895 | 0.921 | 0.929 | 0.987 |
| baseline | hier_c100_s2_d32_e5 | 0.799 | 0.750 | 0.711 | 0.691 | 0.679 | 0.537 | 0.796 | 0.861 | 0.906 | 0.917 | 0.988 |
| baseline | hier_c100_s1_d128_e5 | 0.797 | 0.768 | 0.745 | 0.726 | 0.713 | 0.572 | 0.819 | 0.873 | 0.911 | 0.921 | 0.987 |
| baseline | hier_c100_s3_d16_e15 | 0.795 | 0.786 | 0.744 | 0.730 | 0.724 | 0.602 | 0.819 | 0.872 | 0.912 | 0.921 | 0.986 |
| baseline | hier_c100_s2_d128_e30 | 0.795 | 0.823 | 0.806 | 0.799 | 0.796 | 0.683 | 0.861 | 0.899 | 0.923 | 0.929 | 0.987 |
| baseline | ours_c100_d16_s3_e40 | 0.793 | 0.583 | 0.512 | 0.488 | 0.471 | 0.312 | 0.612 | 0.694 | 0.765 | 0.785 | 0.937 |
| baseline | hier_c100_s3_d16_e75 | 0.792 | 0.807 | 0.764 | 0.755 | 0.752 | 0.646 | 0.838 | 0.884 | 0.917 | 0.925 | 0.985 |
| baseline | hier_c100_s3_d32_e90 | 0.791 | 0.817 | 0.790 | 0.783 | 0.779 | 0.673 | 0.854 | 0.894 | 0.921 | 0.927 | 0.986 |
| baseline | hier_c100_s3_d16_e45 | 0.790 | 0.805 | 0.764 | 0.755 | 0.752 | 0.646 | 0.836 | 0.883 | 0.918 | 0.926 | 0.986 |
| baseline | hier_c100_s1_d32_e45 | 0.790 | 0.821 | 0.788 | 0.783 | 0.780 | 0.675 | 0.854 | 0.892 | 0.920 | 0.928 | 0.985 |
| baseline | ours_c100_d16_s3_e50 | 0.789 | 0.570 | 0.503 | 0.475 | 0.458 | 0.276 | 0.602 | 0.684 | 0.761 | 0.782 | 0.936 |
| baseline | hier_c100_s3_d32_e30 | 0.788 | 0.817 | 0.788 | 0.779 | 0.776 | 0.670 | 0.852 | 0.892 | 0.919 | 0.927 | 0.987 |
| baseline | hier_c100_s1_d32_e30 | 0.788 | 0.821 | 0.785 | 0.781 | 0.779 | 0.672 | 0.852 | 0.892 | 0.921 | 0.928 | 0.985 |
| baseline | hier_c100_s2_d128_e75 | 0.786 | 0.822 | 0.806 | 0.801 | 0.797 | 0.684 | 0.864 | 0.900 | 0.923 | 0.929 | 0.987 |
| baseline | hier_c100_s2_d32_e45 | 0.786 | 0.819 | 0.790 | 0.784 | 0.783 | 0.675 | 0.853 | 0.892 | 0.921 | 0.929 | 0.985 |
| baseline | ours_c100_d16_s2_e30 | 0.785 | 0.601 | 0.512 | 0.492 | 0.478 | 0.318 | 0.614 | 0.700 | 0.773 | 0.794 | 0.932 |
| baseline | hier_c100_s1_d128_e55 | 0.783 | 0.827 | 0.813 | 0.805 | 0.801 | 0.688 | 0.866 | 0.903 | 0.924 | 0.931 | 0.986 |
| baseline | ours_c100_d16_s3_e30 | 0.783 | 0.592 | 0.511 | 0.490 | 0.475 | 0.313 | 0.613 | 0.696 | 0.768 | 0.788 | 0.934 |
| baseline | hier_c100_s1_d16_e45 | 0.783 | 0.808 | 0.765 | 0.754 | 0.750 | 0.645 | 0.833 | 0.880 | 0.914 | 0.924 | 0.984 |
| baseline | hier_c100_s1_d128_e75 | 0.783 | 0.827 | 0.813 | 0.805 | 0.801 | 0.688 | 0.866 | 0.902 | 0.924 | 0.931 | 0.987 |
| baseline | hier_c100_s2_d128_e90 | 0.783 | 0.821 | 0.806 | 0.801 | 0.797 | 0.684 | 0.864 | 0.900 | 0.923 | 0.929 | 0.987 |
| baseline | hier_c100_s3_d16_e95 | 0.783 | 0.806 | 0.764 | 0.755 | 0.752 | 0.646 | 0.837 | 0.883 | 0.917 | 0.925 | 0.985 |
| baseline | hier_c100_s1_d16_e75 | 0.782 | 0.809 | 0.763 | 0.754 | 0.751 | 0.646 | 0.833 | 0.880 | 0.914 | 0.923 | 0.984 |
| baseline | hier_c100_s1_d16_e75_root_false | 0.782 | 0.809 | 0.763 | 0.754 | 0.751 | 0.646 | 0.833 | 0.880 | 0.914 | 0.923 | 0.984 |
| baseline | ours_c100_d16_s1_e15 | 0.782 | 0.606 | 0.519 | 0.490 | 0.478 | 0.322 | 0.617 | 0.697 | 0.771 | 0.792 | 0.933 |
| baseline | hier_c100_s1_d16_e15 | 0.780 | 0.786 | 0.736 | 0.730 | 0.724 | 0.607 | 0.814 | 0.868 | 0.908 | 0.918 | 0.986 |
| baseline | hier_c100_s1_d32_e15 | 0.780 | 0.807 | 0.779 | 0.767 | 0.761 | 0.648 | 0.844 | 0.887 | 0.920 | 0.927 | 0.986 |
| baseline | ours_c100_d16_s3_e35 | 0.780 | 0.591 | 0.509 | 0.490 | 0.474 | 0.314 | 0.615 | 0.695 | 0.766 | 0.787 | 0.936 |
| baseline | hier_c100_s2_d32_e75 | 0.777 | 0.818 | 0.791 | 0.784 | 0.784 | 0.676 | 0.854 | 0.892 | 0.921 | 0.928 | 0.985 |
| baseline | hier_c100_s2_d128_e55 | 0.776 | 0.824 | 0.807 | 0.801 | 0.798 | 0.685 | 0.864 | 0.900 | 0.923 | 0.930 | 0.987 |
| baseline | ours_c100_d16_s1_e10 | 0.775 | 0.609 | 0.522 | 0.494 | 0.480 | 0.330 | 0.621 | 0.699 | 0.774 | 0.795 | 0.935 |
| baseline | ours_c100_d16_s3_e25 | 0.775 | 0.595 | 0.514 | 0.492 | 0.475 | 0.313 | 0.614 | 0.699 | 0.766 | 0.787 | 0.933 |
| baseline | hier_c100_s1_d128_e15 | 0.774 | 0.811 | 0.795 | 0.789 | 0.784 | 0.665 | 0.855 | 0.895 | 0.921 | 0.928 | 0.986 |
| baseline | hier_c100_s1_d128_e45 | 0.772 | 0.828 | 0.813 | 0.805 | 0.801 | 0.688 | 0.865 | 0.901 | 0.923 | 0.931 | 0.986 |
| baseline | hier_c100_s1_d128_e90 | 0.769 | 0.826 | 0.813 | 0.805 | 0.801 | 0.688 | 0.866 | 0.902 | 0.924 | 0.931 | 0.987 |
| baseline | ours_c100_d16_s1_e40 | 0.769 | 0.554 | 0.507 | 0.482 | 0.467 | 0.304 | 0.607 | 0.688 | 0.760 | 0.784 | 0.937 |
| baseline | hier_c100_s3_d128_e75 | 0.767 | 0.820 | 0.808 | 0.801 | 0.799 | 0.685 | 0.862 | 0.896 | 0.922 | 0.929 | 0.987 |
| baseline | hier_c100_s1_d128_e30 | 0.765 | 0.827 | 0.812 | 0.804 | 0.800 | 0.686 | 0.865 | 0.899 | 0.925 | 0.932 | 0.986 |
| baseline | ours_c100_d16_s1_e50 | 0.765 | 0.546 | 0.495 | 0.468 | 0.450 | 0.246 | 0.595 | 0.675 | 0.752 | 0.774 | 0.942 |
| baseline | hier_c100_s3_d128_e90 | 0.764 | 0.822 | 0.808 | 0.801 | 0.799 | 0.685 | 0.862 | 0.896 | 0.922 | 0.929 | 0.987 |
| baseline | ours_c100_d16_s3_e20 | 0.763 | 0.601 | 0.516 | 0.494 | 0.478 | 0.318 | 0.619 | 0.699 | 0.769 | 0.789 | 0.934 |
| baseline | ours_c100_d16_s3_e15 | 0.762 | 0.612 | 0.518 | 0.497 | 0.481 | 0.324 | 0.621 | 0.701 | 0.770 | 0.790 | 0.936 |
| baseline | ours_c100_d16_s2_e25 | 0.762 | 0.604 | 0.514 | 0.494 | 0.480 | 0.321 | 0.615 | 0.704 | 0.772 | 0.793 | 0.932 |
| baseline | hier_c100_s1_d128_e95 | 0.761 | 0.827 | 0.813 | 0.805 | 0.801 | 0.688 | 0.866 | 0.903 | 0.924 | 0.931 | 0.987 |
| baseline | ours_c100_d16_s1_e45 | 0.760 | 0.555 | 0.501 | 0.478 | 0.461 | 0.282 | 0.605 | 0.689 | 0.765 | 0.785 | 0.936 |
| baseline | ours_c100_d16_s1_e35 | 0.758 | 0.561 | 0.508 | 0.485 | 0.469 | 0.308 | 0.611 | 0.694 | 0.766 | 0.787 | 0.934 |
| baseline | ours_c100_d16_s3_e10 | 0.757 | 0.619 | 0.521 | 0.500 | 0.485 | 0.332 | 0.623 | 0.702 | 0.771 | 0.791 | 0.937 |
| baseline | ours_c100_d16_s1_e20 | 0.754 | 0.599 | 0.513 | 0.490 | 0.474 | 0.317 | 0.612 | 0.694 | 0.770 | 0.790 | 0.934 |
| baseline | ours_c100_d16_s2_e15 | 0.750 | 0.613 | 0.516 | 0.497 | 0.482 | 0.326 | 0.623 | 0.709 | 0.775 | 0.794 | 0.933 |
| baseline | ours_c100_d16_s1_e30 | 0.749 | 0.573 | 0.507 | 0.484 | 0.469 | 0.307 | 0.609 | 0.692 | 0.765 | 0.786 | 0.933 |
| baseline | ours_c100_d16_s1_e25 | 0.749 | 0.591 | 0.509 | 0.487 | 0.474 | 0.314 | 0.613 | 0.692 | 0.769 | 0.791 | 0.934 |
| baseline | ours_c100_d16_s2_e10 | 0.745 | 0.614 | 0.523 | 0.501 | 0.486 | 0.332 | 0.625 | 0.710 | 0.779 | 0.798 | 0.936 |
| baseline | ours_c100_d16_s2_e20 | 0.741 | 0.604 | 0.515 | 0.497 | 0.480 | 0.324 | 0.616 | 0.704 | 0.773 | 0.792 | 0.933 |
| wandb | giddy-fog-1387 [c0.01,d16,hist_revert,lca_curv_sweep,seed2] | 0.806 | 0.632 | 0.544 | 0.516 | 0.501 | 0.351 | 0.642 | 0.719 | 0.787 | 0.808 | 0.945 |
| wandb | devoted-sun-322 [baseline,hyperboloid] | 0.806 | — | 0.567 | 0.533 | 0.519 | 0.400 | — | — | — | — | — |
| wandb | toasty-lake-1401 [d16,hyperboloid,manifold_ablation,paper_main,seed2] | 0.774 | 0.246 | 0.104 | 0.085 | 0.076 | 0.046 | 0.167 | 0.253 | 0.369 | 0.411 | 0.874 |
| wandb | sandy-dawn-1438 [c100.0,d16,hist_revert,lca_curv_sweep,seed2] | 0.770 | 0.632 | 0.529 | 0.506 | 0.491 | 0.350 | 0.633 | 0.716 | 0.781 | 0.802 | 0.941 |
| wandb | exalted-shadow-1405 [d16,hyperboloid,manifold_ablation,paper_main,seed3] | 0.767 | 0.242 | 0.100 | 0.085 | 0.075 | 0.047 | 0.158 | 0.245 | 0.359 | 0.398 | 0.860 |
| wandb | worldly-oath-1224 [paper_main,poincare_d16_ours,seed3] | 0.765 | 0.210 | 0.086 | 0.064 | 0.057 | 0.034 | 0.138 | 0.205 | 0.306 | 0.347 | 0.837 |
| wandb | colorful-sun-1222 [paper_main,poincare_d16_ours,seed2] | 0.759 | 0.242 | 0.101 | 0.086 | 0.077 | 0.045 | 0.162 | 0.244 | 0.353 | 0.392 | 0.856 |
| wandb | ethereal-morning-1226 [paper_main,poincare_d32_ours,seed2] | 0.747 | 0.224 | 0.098 | 0.079 | 0.068 | 0.041 | 0.157 | 0.236 | 0.340 | 0.382 | 0.852 |
| wandb | feasible-resonance-1238 [paper_main,poincare_d16_ours,seed3] | 0.745 | 0.222 | 0.094 | 0.073 | 0.065 | 0.040 | 0.146 | 0.221 | 0.328 | 0.370 | 0.856 |
| wandb | magic-sunset-1223 [paper_main,poincare_d16_ours,seed1] | 0.742 | 0.190 | 0.047 | 0.040 | 0.036 | 0.026 | 0.084 | 0.140 | 0.227 | 0.264 | 0.801 |
| wandb | visionary-wind-1227 [paper_main,poincare_d32_ours,seed3] | 0.736 | 0.208 | 0.079 | 0.060 | 0.052 | 0.033 | 0.129 | 0.197 | 0.299 | 0.334 | 0.838 |
| wandb | lemon-gorge-1239 [paper_main,poincare_d32_ours,seed1] | 0.734 | 0.230 | 0.118 | 0.090 | 0.079 | 0.044 | 0.185 | 0.271 | 0.385 | 0.427 | 0.876 |
| wandb | lunar-fog-1225 [paper_main,poincare_d32_ours,seed1] | 0.732 | 0.198 | 0.076 | 0.060 | 0.054 | 0.033 | 0.127 | 0.199 | 0.298 | 0.340 | 0.834 |
| wandb | legendary-bee-1389 [d16,hyperboloid,manifold_ablation,paper_main,seed1] | 0.729 | 0.200 | 0.054 | 0.047 | 0.042 | 0.030 | 0.091 | 0.155 | 0.243 | 0.282 | 0.808 |


## tiny_imagenet

| Source | Run / tag | hDP_ward | NMI_ward | NN@1 | NN@5 | NN@10 | NN@100 | R@2 | R@4 | R@8 | R@10 | R@100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | ours_ti_d128_s44_e50 | 0.617 | 0.389 | 0.103 | 0.087 | 0.077 | 0.043 | 0.164 | 0.245 | 0.352 | 0.390 | 0.803 |
| baseline | ours_ti_d128_s44_e10 | 0.605 | 0.347 | 0.049 | 0.040 | 0.037 | 0.024 | 0.082 | 0.137 | 0.217 | 0.248 | 0.710 |
| baseline | ours_ti_d128_s42_e180 | 0.587 | 0.439 | 0.166 | 0.147 | 0.133 | 0.068 | 0.248 | 0.345 | 0.459 | 0.493 | 0.851 |
| baseline | ours_ti_d128_s43_e60 | 0.557 | 0.394 | 0.110 | 0.093 | 0.083 | 0.046 | 0.171 | 0.255 | 0.363 | 0.398 | 0.809 |
| baseline | ours_ti_d128_s43_e240 | 0.536 | 0.449 | 0.184 | 0.160 | 0.146 | 0.074 | 0.265 | 0.359 | 0.466 | 0.500 | 0.848 |
| baseline | ours_ti_d128_s44_e60 | 0.535 | 0.395 | 0.113 | 0.093 | 0.084 | 0.046 | 0.178 | 0.256 | 0.362 | 0.401 | 0.811 |
| baseline | ours_ti_d128_s43_e220 | 0.534 | 0.442 | 0.180 | 0.155 | 0.140 | 0.071 | 0.257 | 0.355 | 0.459 | 0.501 | 0.846 |
| baseline | ours_ti_d128_s43_e160 | 0.533 | 0.426 | 0.157 | 0.134 | 0.121 | 0.063 | 0.232 | 0.326 | 0.430 | 0.469 | 0.839 |
| baseline | ours_ti_d128_s42_e160 | 0.532 | 0.433 | 0.164 | 0.140 | 0.126 | 0.065 | 0.244 | 0.338 | 0.444 | 0.483 | 0.845 |
| baseline | ours_ti_d128_s43_e140 | 0.519 | 0.425 | 0.149 | 0.132 | 0.119 | 0.061 | 0.221 | 0.316 | 0.426 | 0.462 | 0.833 |
| baseline | ours_ti_d128_s42_e40 | 0.514 | 0.383 | 0.096 | 0.080 | 0.073 | 0.041 | 0.155 | 0.230 | 0.334 | 0.373 | 0.795 |
| baseline | ours_ti_d128_s42_e10 | 0.511 | 0.346 | 0.050 | 0.039 | 0.036 | 0.023 | 0.084 | 0.134 | 0.209 | 0.243 | 0.713 |
| baseline | ours_ti_d128_s43_e40 | 0.511 | 0.381 | 0.089 | 0.075 | 0.069 | 0.040 | 0.142 | 0.221 | 0.324 | 0.364 | 0.790 |
| baseline | ours_ti_d128_s44_e140 | 0.503 | 0.426 | 0.154 | 0.135 | 0.121 | 0.062 | 0.231 | 0.327 | 0.434 | 0.470 | 0.839 |
| baseline | ours_ti_d128_s42_e80 | 0.503 | 0.406 | 0.128 | 0.108 | 0.096 | 0.051 | 0.193 | 0.281 | 0.388 | 0.424 | 0.822 |
| baseline | ours_ti_d128_s43_e120 | 0.488 | 0.423 | 0.147 | 0.127 | 0.115 | 0.060 | 0.216 | 0.311 | 0.420 | 0.460 | 0.834 |
| baseline | hier_ti_s1_d128_e90 | 0.483 | 0.785 | 0.743 | 0.736 | 0.731 | 0.349 | 0.798 | 0.835 | 0.861 | 0.870 | 0.971 |
| baseline | ours_ti_d128_s43_e100 | 0.473 | 0.417 | 0.134 | 0.115 | 0.105 | 0.056 | 0.205 | 0.293 | 0.403 | 0.441 | 0.828 |
| baseline | ours_ti_d128_s43_e10 | 0.472 | 0.347 | 0.046 | 0.038 | 0.035 | 0.023 | 0.078 | 0.129 | 0.211 | 0.241 | 0.719 |
| baseline | ours_ti_d128_s42_e60 | 0.468 | 0.397 | 0.111 | 0.094 | 0.086 | 0.047 | 0.175 | 0.261 | 0.367 | 0.407 | 0.815 |
| baseline | ours_ti_d128_s43_e30 | 0.448 | 0.375 | 0.079 | 0.069 | 0.061 | 0.036 | 0.137 | 0.210 | 0.306 | 0.343 | 0.780 |
| baseline | hier_ti_d16_lowdim_s1_ep95 | 0.447 | 0.753 | 0.659 | 0.652 | 0.643 | 0.299 | 0.735 | 0.790 | 0.834 | 0.848 | 0.961 |
| baseline | hier_ti_s3_d128_e35_partial | 0.447 | 0.783 | 0.738 | 0.736 | 0.731 | 0.348 | 0.795 | 0.833 | 0.860 | 0.870 | 0.971 |
| baseline | ours_ti_d128_s43_e20 | 0.443 | 0.362 | 0.063 | 0.055 | 0.049 | 0.030 | 0.107 | 0.176 | 0.268 | 0.304 | 0.758 |
| baseline | hier_ti_s2_d128_e80 | 0.440 | 0.786 | 0.737 | 0.736 | 0.731 | 0.349 | 0.795 | 0.834 | 0.862 | 0.870 | 0.971 |
| baseline | ours_ti_d128_s42_e100 | 0.437 | 0.418 | 0.137 | 0.119 | 0.107 | 0.056 | 0.205 | 0.299 | 0.409 | 0.447 | 0.831 |
| baseline | ours_ti_d128_s44_e200 | 0.428 | 0.441 | 0.175 | 0.153 | 0.138 | 0.070 | 0.252 | 0.350 | 0.461 | 0.499 | 0.854 |
| baseline | hier_ti_d32_lowdim_s1_ep70 | 0.425 | 0.774 | 0.709 | 0.700 | 0.695 | 0.331 | 0.769 | 0.815 | 0.854 | 0.863 | 0.966 |
| baseline | ours_ti_d128_s43_e50 | 0.424 | 0.388 | 0.103 | 0.085 | 0.076 | 0.043 | 0.164 | 0.240 | 0.344 | 0.381 | 0.801 |
| baseline | ours_ti_d128_s44_e160 | 0.423 | 0.427 | 0.157 | 0.138 | 0.124 | 0.063 | 0.234 | 0.329 | 0.439 | 0.477 | 0.840 |
| baseline | ours_ti_d128_s43_e180 | 0.423 | 0.433 | 0.162 | 0.140 | 0.128 | 0.065 | 0.238 | 0.332 | 0.444 | 0.481 | 0.840 |
| baseline | ours_ti_d128_s44_e180 | 0.420 | 0.436 | 0.170 | 0.145 | 0.132 | 0.067 | 0.245 | 0.342 | 0.452 | 0.489 | 0.843 |
| baseline | ours_ti_d128_s44_e20 | 0.419 | 0.364 | 0.063 | 0.056 | 0.050 | 0.031 | 0.107 | 0.174 | 0.262 | 0.296 | 0.754 |
| baseline | ours_ti_d128_s42_e220 | 0.417 | 0.445 | 0.184 | 0.157 | 0.142 | 0.072 | 0.261 | 0.358 | 0.469 | 0.508 | 0.855 |
| baseline | ours_ti_d128_s43_e200 | 0.414 | 0.439 | 0.172 | 0.152 | 0.138 | 0.070 | 0.249 | 0.347 | 0.460 | 0.497 | 0.847 |
| baseline | ours_ti_d128_s44_e100 | 0.409 | 0.413 | 0.146 | 0.121 | 0.108 | 0.056 | 0.216 | 0.304 | 0.413 | 0.451 | 0.826 |
| baseline | ours_ti_d128_s42_e240 | 0.408 | 0.448 | 0.187 | 0.160 | 0.146 | 0.074 | 0.270 | 0.365 | 0.476 | 0.514 | 0.852 |
| baseline | ours_ti_d128_s42_e200 | 0.406 | 0.440 | 0.173 | 0.151 | 0.136 | 0.069 | 0.251 | 0.351 | 0.459 | 0.497 | 0.846 |
| baseline | ours_ti_d128_s42_e120 | 0.406 | 0.421 | 0.141 | 0.122 | 0.110 | 0.058 | 0.212 | 0.308 | 0.420 | 0.456 | 0.834 |
| baseline | ours_ti_d128_s44_e80 | 0.405 | 0.407 | 0.132 | 0.108 | 0.097 | 0.052 | 0.201 | 0.283 | 0.392 | 0.431 | 0.821 |
| baseline | ours_ti_d128_s44_e40 | 0.403 | 0.386 | 0.095 | 0.081 | 0.072 | 0.041 | 0.154 | 0.230 | 0.334 | 0.372 | 0.799 |
| baseline | ours_ti_d128_s42_e30 | 0.403 | 0.373 | 0.080 | 0.068 | 0.061 | 0.036 | 0.132 | 0.208 | 0.305 | 0.340 | 0.777 |
| baseline | ours_ti_d128_s42_e140 | 0.403 | 0.427 | 0.157 | 0.133 | 0.121 | 0.062 | 0.229 | 0.323 | 0.439 | 0.476 | 0.837 |
| baseline | ours_ti_d128_s44_e220 | 0.402 | 0.447 | 0.184 | 0.158 | 0.145 | 0.073 | 0.264 | 0.363 | 0.477 | 0.513 | 0.856 |
| baseline | ours_ti_d128_s44_e30 | 0.400 | 0.376 | 0.083 | 0.069 | 0.063 | 0.037 | 0.133 | 0.206 | 0.306 | 0.346 | 0.786 |
| baseline | ours_ti_d128_s43_e80 | 0.398 | 0.409 | 0.126 | 0.107 | 0.095 | 0.052 | 0.193 | 0.281 | 0.385 | 0.421 | 0.823 |
| baseline | ours_ti_d128_s42_e20 | 0.397 | 0.365 | 0.067 | 0.056 | 0.050 | 0.031 | 0.113 | 0.180 | 0.269 | 0.304 | 0.761 |
| baseline | ours_ti_d128_s42_e50 | 0.394 | 0.391 | 0.102 | 0.087 | 0.079 | 0.044 | 0.162 | 0.246 | 0.355 | 0.395 | 0.806 |
| baseline | ours_ti_d128_s44_e120 | 0.394 | 0.418 | 0.149 | 0.126 | 0.112 | 0.058 | 0.219 | 0.311 | 0.423 | 0.459 | 0.832 |
| baseline | ours_ti_d128_s44_e240 | 0.382 | 0.449 | 0.189 | 0.165 | 0.150 | 0.074 | 0.266 | 0.367 | 0.477 | 0.516 | 0.860 |
| wandb | playful-voice-1374 [d128,h2h_warm,hyperboloid,manifold_ablation,seed43] | 0.842 | 0.345 | 0.067 | 0.050 | 0.043 | 0.025 | — | — | — | — | — |
| wandb | fresh-puddle-1407 [d128,h2h_warm,lca_ablation,lca_chami,seed42] | 0.841 | 0.340 | 0.044 | 0.035 | 0.032 | 0.021 | — | — | — | — | — |
| wandb | worldly-shape-1410 [d128,h2h_warm,lca_ablation,lca_ours,seed44] | 0.840 | 0.338 | 0.037 | 0.033 | 0.031 | 0.020 | — | — | — | — | — |
| wandb | hopeful-sunset-1375 [d128,h2h_warm,hyperboloid,manifold_ablation,seed42] | 0.837 | 0.331 | 0.055 | 0.044 | 0.037 | 0.022 | — | — | — | — | — |
| wandb | serene-darkness-1407 [d128,h2h_warm,lca_ablation,lca_ours,seed43] | 0.836 | 0.341 | 0.038 | 0.033 | 0.030 | 0.021 | — | — | — | — | — |
| wandb | fragrant-frost-1404 [d128,h2h_warm,lca_ablation,lca_ours,seed42] | 0.833 | 0.340 | 0.042 | 0.035 | 0.032 | 0.021 | — | — | — | — | — |
| wandb | splendid-fog-1409 [d128,h2h_warm,lca_ablation,lca_chami,seed44] | 0.832 | 0.333 | 0.039 | 0.033 | 0.029 | 0.020 | — | — | — | — | — |
| wandb | swift-bush-1411 [d128,h2h_warm,lca_ablation,lca_chami,seed43] | 0.826 | 0.334 | 0.039 | 0.034 | 0.030 | 0.020 | — | — | — | — | — |
| wandb | vague-pyramid-1376 [d128,h2h_warm,hyperboloid,manifold_ablation,seed44] | 0.826 | 0.338 | 0.066 | 0.051 | 0.043 | 0.024 | — | — | — | — | — |
| wandb | upbeat-pine-1344 [d16,h2h_warm,paper_main,poincare_hyphc,seed44] | 0.747 | 0.516 | 0.269 | 0.246 | 0.230 | 0.107 | — | — | — | — | — |
| wandb | unique-tree-1436 [d32,h2h_warm,paper_main,poincare_hyphc,seed44] | 0.743 | 0.517 | 0.283 | 0.257 | 0.238 | 0.115 | — | — | — | — | — |
| wandb | confused-frost-1328 [d32,h2h_warm,paper_main,poincare_hyphc,seed43] | 0.738 | 0.516 | 0.289 | 0.258 | 0.241 | 0.116 | — | — | — | — | — |
| wandb | smooth-frog-1343 [d32,h2h_warm,paper_main,poincare_hyphc,seed44] | 0.734 | 0.521 | 0.282 | 0.256 | 0.239 | 0.116 | — | — | — | — | — |
| wandb | crimson-tree-1297 [d32,h2h_warm,paper_main,poincare_hyphc,seed42] | 0.730 | 0.522 | 0.287 | 0.260 | 0.243 | 0.117 | — | — | — | — | — |
| wandb | fresh-firebrand-1328 [d16,h2h_warm,paper_main,poincare_hyphc,seed43] | 0.729 | 0.511 | 0.263 | 0.240 | 0.224 | 0.103 | — | — | — | — | — |
| wandb | giddy-frost-1433 [d16,h2h_warm,paper_main,poincare_hyphc,seed43] | 0.724 | 0.508 | 0.268 | 0.241 | 0.224 | 0.101 | — | — | — | — | — |
| wandb | colorful-sea-1435 [d32,h2h_warm,paper_main,poincare_hyphc,seed43] | 0.721 | 0.521 | 0.289 | 0.261 | 0.244 | 0.117 | — | — | — | — | — |
| wandb | jolly-meadow-1294 [d16,h2h_warm,paper_main,poincare_hyphc,seed42] | 0.709 | 0.514 | 0.269 | 0.244 | 0.229 | 0.109 | — | — | — | — | — |
| wandb | bumbling-sky-1295 [d16,h2h_warm,paper_main,poincare_no_hyphc,seed42] | 0.477 | 0.513 | 0.268 | 0.243 | 0.224 | 0.104 | — | — | — | — | — |
| wandb | dutiful-dew-1345 [d16,h2h_warm,paper_main,poincare_no_hyphc,seed44] | 0.463 | 0.514 | 0.273 | 0.245 | 0.226 | 0.105 | — | — | — | — | — |
| wandb | copper-shape-1328 [d32,h2h_warm,paper_main,poincare_no_hyphc,seed43] | 0.458 | 0.516 | 0.273 | 0.248 | 0.229 | 0.106 | — | — | — | — | — |
| wandb | copper-totem-1346 [d32,h2h_warm,paper_main,poincare_no_hyphc,seed44] | 0.449 | 0.513 | 0.268 | 0.242 | 0.225 | 0.104 | — | — | — | — | — |
| wandb | blooming-cloud-1298 [d32,h2h_warm,paper_main,poincare_no_hyphc,seed42] | 0.435 | 0.511 | 0.271 | 0.245 | 0.226 | 0.105 | — | — | — | — | — |
| wandb | quiet-sun-1328 [d16,h2h_warm,paper_main,poincare_no_hyphc,seed43] | 0.428 | 0.515 | 0.273 | 0.243 | 0.226 | 0.103 | — | — | — | — | — |
| wandb | rose-tree-1350 [d32,euclidean_baseline,h2h_warm,paper_main,seed44] | 0.423 | 0.508 | 0.291 | 0.259 | 0.240 | 0.108 | — | — | — | — | — |
| wandb | eternal-shape-1333 [d16,euclidean_baseline,h2h_warm,paper_main,seed43] | 0.421 | 0.502 | 0.278 | 0.250 | 0.232 | 0.098 | — | — | — | — | — |
| wandb | comfy-sea-1362 [d16,euclidean_baseline,h2h_warm,paper_main,seed44] | 0.417 | 0.503 | 0.274 | 0.252 | 0.232 | 0.098 | — | — | — | — | — |
| wandb | pious-haze-1332 [d32,euclidean_baseline,h2h_warm,paper_main,seed43] | 0.412 | 0.511 | 0.291 | 0.260 | 0.243 | 0.108 | — | — | — | — | — |
| wandb | peachy-flower-1299 [d32,euclidean_baseline,h2h_warm,paper_main,seed42] | 0.411 | 0.508 | 0.282 | 0.258 | 0.239 | 0.107 | — | — | — | — | — |
| wandb | cool-breeze-1296 [d16,euclidean_baseline,h2h_warm,paper_main,seed42] | 0.393 | 0.497 | 0.281 | 0.249 | 0.230 | 0.098 | — | — | — | — | — |


## imagenet

| Source | Run / tag | hDP_ward | NMI_ward | NN@1 | NN@5 | NN@10 | NN@100 | R@2 | R@4 | R@8 | R@10 | R@100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wandb | cosmic-elevator-1429 [bk0,d128,h2h_warm,lca_ablation,lca_ours,seed44] | 0.793 | 0.591 | 0.007 | 0.006 | 0.005 | 0.002 | — | — | — | — | — |
| wandb | copper-wave-1428 [bk0,d128,h2h_warm,lca_ablation,lca_chami,seed44] | 0.792 | 0.589 | 0.008 | 0.006 | 0.005 | 0.002 | — | — | — | — | — |
| wandb | worthy-donkey-1394 [bk0,d128,h2h_warm,lca_ablation,lca_ours,seed42] | 0.791 | 0.589 | 0.007 | 0.006 | 0.006 | 0.002 | — | — | — | — | — |
| wandb | morning-darkness-1396 [bk0,d128,h2h_warm,lca_ablation,lca_chami,seed42] | 0.791 | 0.590 | 0.006 | 0.006 | 0.006 | 0.002 | — | — | — | — | — |
| wandb | quiet-microwave-1429 [bk0,d128,h2h_warm,lca_ablation,lca_chami,seed43] | 0.789 | 0.590 | 0.008 | 0.007 | 0.006 | 0.002 | — | — | — | — | — |
| wandb | apricot-water-1399 [bk0,d128,h2h_warm,lca_ablation,lca_ours,seed43] | 0.788 | 0.590 | 0.009 | 0.006 | 0.006 | 0.002 | — | — | — | — | — |
| wandb | distinctive-fire-1381 [bk0,d128,h2h_warm,hyperboloid,manifold_ablation,seed43] | 0.763 | 0.672 | 0.057 | 0.031 | 0.022 | 0.007 | — | — | — | — | — |
| wandb | colorful-sponge-1382 [bk0,d128,h2h_warm,hyperboloid,manifold_ablation,seed44] | 0.761 | 0.669 | 0.057 | 0.029 | 0.020 | 0.006 | — | — | — | — | — |
| wandb | honest-rain-1382 [bk0,d128,h2h_warm,hyperboloid,manifold_ablation,seed42] | 0.752 | 0.670 | 0.057 | 0.030 | 0.020 | 0.006 | — | — | — | — | — |
| wandb | jumping-shadow-1202 [d32,dino_frozen,h2h,paper_main,poincare_hyphc_cold,seed2] | 0.622 | 0.695 | 0.113 | 0.072 | 0.056 | 0.020 | — | — | — | — | — |
| wandb | sweet-terrain-1232 [d32,dino_frozen,h2h,paper_main,poincare_no_hyphc,seed3] | 0.609 | 0.696 | 0.114 | 0.077 | 0.060 | 0.023 | — | — | — | — | — |
| wandb | brisk-dew-1206 [d32,dino_frozen,h2h,paper_main,poincare_no_hyphc,seed2] | 0.602 | 0.696 | 0.114 | 0.079 | 0.062 | 0.024 | — | — | — | — | — |
| wandb | ethereal-frog-1215 [d32,dino_frozen,euclidean_baseline,h2h,paper_main,seed2] | 0.540 | 0.718 | 0.190 | 0.127 | 0.095 | 0.026 | — | — | — | — | — |
| wandb | distinctive-frog-1184 [d32,dino_frozen,h2h,paper_main,poincare_no_hyphc,seed1] | 0.530 | 0.697 | 0.115 | 0.078 | 0.062 | 0.023 | — | — | — | — | — |
| wandb | stilted-leaf-1183 [d32,dino_frozen,euclidean_baseline,h2h,paper_main,seed1] | 0.493 | 0.719 | 0.198 | 0.129 | 0.096 | 0.026 | — | — | — | — | — |
| wandb | silver-waterfall-1199 [d32,dino_frozen,h2h,paper_main,poincare_hyphc_warm,seed2] | 0.463 | 0.672 | 0.061 | 0.040 | 0.031 | 0.013 | — | — | — | — | — |
| wandb | worthy-wildflower-1200 [d32,dino_frozen,h2h,paper_main,poincare_hyphc_warm,seed3] | 0.442 | 0.670 | 0.060 | 0.038 | 0.029 | 0.011 | — | — | — | — | — |
| wandb | electric-pond-1231 [d32,dino_frozen,h2h,paper_main,poincare_hyphc_cold,seed3] | 0.436 | 0.695 | 0.109 | 0.070 | 0.054 | 0.019 | — | — | — | — | — |
| wandb | serene-moon-1184 [d32,dino_frozen,h2h,paper_main,poincare_hyphc_cold,seed1] | 0.433 | 0.696 | 0.108 | 0.071 | 0.055 | 0.019 | — | — | — | — | — |
| wandb | warm-fog-1233 [d32,dino_frozen,euclidean_baseline,h2h,paper_main,seed3] | 0.430 | 0.717 | 0.188 | 0.125 | 0.093 | 0.026 | — | — | — | — | — |
| wandb | rare-darkness-1198 [d32,dino_frozen,h2h,paper_main,poincare_hyphc_warm,seed1] | 0.428 | 0.667 | 0.069 | 0.044 | 0.034 | 0.013 | — | — | — | — | — |


## inat18

| Source | Run / tag | hDP_ward | NMI_ward | NN@1 | NN@5 | NN@10 | NN@100 | R@2 | R@4 | R@8 | R@10 | R@100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | frozen_dinov2_vitg14_res336_pca256 | 0.854 | 0.889 | 0.414 | 0.176 | 0.105 | 0.016 | 0.502 | 0.589 | 0.669 | 0.693 | 0.878 |
| baseline | frozen_dinov2_vitl14_res336_pca256 | 0.854 | 0.904 | 0.414 | 0.177 | 0.106 | 0.016 | 0.502 | 0.593 | 0.673 | 0.699 | 0.885 |
| baseline | frozen_dinov2_vitl14_res448_pca256 | 0.852 | 0.908 | 0.424 | 0.181 | 0.109 | 0.016 | 0.512 | 0.600 | 0.684 | 0.708 | 0.890 |
| baseline | frozen_dinov2_vitg14_res336_pca128 | 0.852 | 0.887 | 0.388 | 0.166 | 0.101 | 0.015 | 0.479 | 0.565 | 0.646 | 0.672 | 0.869 |
| baseline | frozen_dinov2_vitl14_res336_pca128 | 0.851 | 0.902 | 0.393 | 0.168 | 0.102 | 0.015 | 0.480 | 0.571 | 0.652 | 0.678 | 0.873 |
| baseline | frozen_dinov2_inat18_val_pca128 | 0.851 | 0.897 | 0.349 | 0.153 | 0.094 | 0.015 | 0.441 | 0.529 | 0.617 | 0.642 | 0.853 |
| baseline | frozen_dinov2_vitl14_res336_raw | 0.851 | 0.907 | 0.432 | 0.184 | 0.109 | 0.016 | 0.521 | 0.610 | 0.690 | 0.716 | 0.895 |
| baseline | frozen_dinov2_vitl14_res448_raw | 0.851 | 0.910 | 0.444 | 0.188 | 0.112 | 0.016 | 0.531 | 0.621 | 0.704 | 0.727 | 0.900 |
| baseline | frozen_dinov2_vitl14_res448_pca128 | 0.850 | 0.904 | 0.403 | 0.173 | 0.104 | 0.016 | 0.491 | 0.581 | 0.662 | 0.687 | 0.877 |
| baseline | frozen_dinov2_vitg14_res224_pca256 | 0.850 | 0.885 | 0.359 | 0.155 | 0.095 | 0.015 | 0.448 | 0.535 | 0.617 | 0.643 | 0.856 |
| baseline | frozen_dinov2_inat18_val_1024d | 0.849 | 0.902 | 0.387 | 0.167 | 0.101 | 0.015 | 0.477 | 0.570 | 0.653 | 0.677 | 0.875 |
| baseline | frozen_dinov2_vitg14_res336_raw | 0.849 | 0.893 | 0.450 | 0.186 | 0.110 | 0.016 | 0.536 | 0.620 | 0.696 | 0.719 | 0.896 |
| baseline | frozen_dinov2_vitg14_res448_pca256 | 0.848 | 0.890 | 0.438 | 0.185 | 0.110 | 0.016 | 0.524 | 0.611 | 0.690 | 0.714 | 0.886 |
| baseline | frozen_dinov2_inat18_val_pca256 | 0.847 | 0.899 | 0.374 | 0.162 | 0.098 | 0.015 | 0.463 | 0.553 | 0.639 | 0.664 | 0.864 |
| baseline | frozen_dinov2_vitg14_res448_raw | 0.846 | 0.894 | 0.473 | 0.195 | 0.115 | 0.016 | 0.557 | 0.641 | 0.715 | 0.739 | 0.904 |
| baseline | frozen_dinov2_vitg14_res224_raw | 0.845 | 0.889 | 0.392 | 0.166 | 0.100 | 0.015 | 0.479 | 0.565 | 0.648 | 0.671 | 0.876 |
| baseline | frozen_dinov2_vitg14_res224_pca128 | 0.843 | 0.884 | 0.335 | 0.146 | 0.090 | 0.014 | 0.420 | 0.506 | 0.593 | 0.617 | 0.843 |
| baseline | frozen_dinov2_vitg14_res448_pca128 | 0.842 | 0.888 | 0.411 | 0.176 | 0.105 | 0.016 | 0.501 | 0.591 | 0.670 | 0.695 | 0.877 |
| wandb | daily-frost-1307 [bk1,c03_warm,d128,paper_main,seed1] | 0.935 | 0.949 | 0.000 | 0.000 | 0.000 | 0.000 | — | — | — | — | — |
| wandb | zesty-brook-1377 [bk1,c03_warm,d128,paper_main,seed1] | 0.896 | 0.953 | 0.000 | 0.000 | 0.000 | 0.000 | — | — | — | — | — |
| wandb | swift-galaxy-1317 [bk1,c03_warm,d128,lca_ablation,lca_chami,seed1] | 0.866 | 0.951 | 0.000 | 0.000 | 0.000 | 0.000 | — | — | — | — | — |
| wandb | lemon-mountain-1316 [bk1,c03_warm,d128,lca_ablation,lca_ours,seed1] | 0.866 | 0.951 | 0.000 | 0.000 | 0.000 | 0.000 | — | — | — | — | — |
| wandb | different-oath-1379 [bk1,c03_warm,d128,paper_main,seed3] | 0.861 | 0.940 | 0.000 | 0.000 | 0.000 | 0.000 | — | — | — | — | — |
| wandb | hearty-spaceship-1378 [bk1,c03_warm,d128,paper_main,seed2] | 0.857 | 0.943 | 0.000 | 0.000 | 0.000 | 0.000 | — | — | — | — | — |
| wandb | proud-deluge-1312 [bk1,c03_warm,d128,paper_main,seed3] | 0.853 | 0.939 | 0.000 | 0.000 | 0.000 | 0.000 | — | — | — | — | — |
| wandb | playful-shadow-1392 [bk1,c03_warm,d128,paper_main,seed2] | 0.848 | 0.946 | 0.000 | 0.000 | 0.000 | 0.000 | — | — | — | — | — |
| wandb | comic-lion-1317 [bk1,c03_warm,d128,lca_ablation,lca_ours,seed2] | 0.820 | 0.942 | 0.000 | 0.000 | 0.000 | 0.000 | — | — | — | — | — |
| wandb | pious-music-1361 [c03_cw10_warm,diag_a,dino_frozen,h2h,paper_main,seed2] | 0.754 | 0.750 | 0.000 | 0.000 | 0.000 | 0.000 | — | — | — | — | — |
| wandb | charmed-thunder-1320 [c03_cw10_warm,diag_a,dino_frozen,h2h,paper_main,seed1] | 0.677 | 0.762 | 0.000 | 0.000 | 0.000 | 0.000 | — | — | — | — | — |
| wandb | dauntless-glitter-1357 [c03_d256_warm,diag_a,dino_frozen,h2h,paper_main,seed3] | 0.630 | 0.718 | 0.000 | 0.000 | 0.000 | 0.000 | — | — | — | — | — |
| wandb | atomic-darkness-1358 [c03_cw10_warm,diag_a,dino_frozen,h2h,paper_main,seed3] | 0.582 | 0.756 | 0.000 | 0.000 | 0.000 | 0.000 | — | — | — | — | — |
| wandb | fanciful-plasma-1281 [d128,dino_frozen,h2h,paper_main,poincare_c03_warm,rescue,seed3] | 0.560 | 0.708 | 0.000 | 0.000 | 0.000 | 0.000 | — | — | — | — | — |
| wandb | revived-hill-1356 [c03_d256_warm,diag_a,dino_frozen,h2h,paper_main,seed2] | 0.550 | 0.753 | 0.000 | 0.000 | 0.000 | 0.000 | — | — | — | — | — |
| wandb | genial-firebrand-1393 [c03_lr1e5_warm,diag_a,dino_frozen,h2h,paper_main,seed3] | 0.500 | 0.672 | 0.000 | 0.000 | 0.000 | 0.000 | — | — | — | — | — |
| wandb | wobbly-water-1358 [c03_lr1e5_warm,diag_a,dino_frozen,h2h,paper_main,seed2] | 0.459 | 0.706 | 0.000 | 0.000 | 0.000 | 0.000 | — | — | — | — | — |
| wandb | curious-aardvark-1424 [50epochs,d128,dinov2_vitb14_euclidean,frozen,viability,vitb] | 0.450 | 0.900 | 0.266 | 0.117 | 0.073 | 0.012 | — | — | — | — | — |
| wandb | morning-cloud-1277 [d128,dino_frozen,h2h,paper_main,poincare_c03_warm,rescue,seed1] | 0.449 | 0.731 | 0.000 | 0.000 | 0.000 | 0.000 | — | — | — | — | — |
| wandb | elated-cherry-1351 [c03_lr1e5_warm,diag_a,dino_frozen,h2h,paper_main,seed1] | 0.423 | 0.695 | 0.000 | 0.000 | 0.000 | 0.000 | — | — | — | — | — |
| wandb | vocal-fire-1319 [c03_d256_warm,diag_a,dino_frozen,h2h,paper_main,seed1] | 0.356 | 0.743 | 0.000 | 0.000 | 0.000 | 0.000 | — | — | — | — | — |
| wandb | leafy-darkness-1423 [50epochs,d128,dinov1_vitb16_euclidean,frozen,viability,vitb] | 0.354 | 0.883 | 0.095 | 0.044 | 0.029 | 0.007 | — | — | — | — | — |
| wandb | devoted-grass-1282 [d128,dino_frozen,h2h,paper_main,poincare_c03_warm,rescue,seed2] | 0.347 | 0.739 | 0.000 | 0.000 | 0.000 | 0.000 | — | — | — | — | — |
| wandb | generous-sun-1422 [50epochs,d128,dinov2_vitb14_poincare_c01,frozen,viability,vitb] | 0.331 | 0.812 | 0.019 | 0.010 | 0.007 | 0.002 | — | — | — | — | — |
| wandb | confused-terrain-1418 [50epochs,d128,dinov1_vitb16_poincare_c01,frozen,viability,vitb] | 0.303 | 0.826 | 0.019 | 0.009 | 0.006 | 0.002 | — | — | — | — | — |
| wandb | rural-waterfall-1282 [d128,dino_frozen,h2h,paper_main,poincare_c05_warm,rescue,seed2] | 0.270 | 0.816 | 0.001 | 0.000 | 0.000 | 0.000 | — | — | — | — | — |
| wandb | dutiful-shadow-1293 [d128,dino_frozen,h2h,paper_main,poincare_c05_warm,rescue,seed3] | 0.267 | 0.776 | 0.001 | 0.000 | 0.000 | 0.000 | — | — | — | — | — |
| wandb | quiet-galaxy-1279 [d128,dino_frozen,h2h,paper_main,poincare_c05_warm,rescue,seed1] | 0.261 | 0.818 | 0.001 | 0.000 | 0.000 | 0.000 | — | — | — | — | — |
| wandb | flowing-deluge-1391 [c03_hyperboloid_warm,diag_a,dino_frozen,h2h,paper_main,seed3] | 0.241 | 0.829 | 0.008 | 0.004 | 0.002 | 0.001 | — | — | — | — | — |
| wandb | volcanic-plasma-1282 [d128,dino_frozen,h2h,paper_main,poincare_c01_warm,rescue,seed2] | 0.234 | 0.804 | 0.002 | 0.001 | 0.001 | 0.000 | — | — | — | — | — |
| wandb | lilac-waterfall-1354 [c03_hyperboloid_warm,diag_a,dino_frozen,h2h,paper_main,seed2] | 0.212 | 0.840 | 0.008 | 0.003 | 0.002 | 0.001 | — | — | — | — | — |
| wandb | valiant-energy-1277 [d128,dino_frozen,h2h,paper_main,poincare_c01_warm,rescue,seed1] | 0.196 | 0.798 | 0.001 | 0.001 | 0.001 | 0.000 | — | — | — | — | — |
| wandb | different-leaf-1355 [c03_hyperboloid_warm,diag_a,dino_frozen,h2h,paper_main,seed1] | 0.179 | 0.839 | 0.008 | 0.003 | 0.002 | 0.001 | — | — | — | — | — |
| wandb | light-lake-1280 [d128,dino_frozen,h2h,paper_main,poincare_c01_warm,rescue,seed3] | 0.175 | 0.790 | 0.002 | 0.001 | 0.001 | 0.000 | — | — | — | — | — |
| wandb | dulcet-frost-1276 [d128,dino_frozen,h2h,paper_main,poincare_c03_warm,rescue,seed3] | — | 0.710 | 0.000 | 0.000 | 0.000 | 0.000 | — | — | — | — | — |
| wandb | snowy-resonance-1275 [d128,dino_frozen,h2h,paper_main,poincare_c01_warm,rescue,seed3] | — | 0.792 | 0.002 | 0.001 | 0.001 | 0.000 | — | — | — | — | — |
| wandb | fine-plasma-1274 [d128,dino_frozen,h2h,paper_main,poincare_c01_warm,rescue,seed1] | — | 0.799 | 0.002 | 0.001 | 0.001 | 0.000 | — | — | — | — | — |
| wandb | dandy-night-1273 [d128,dino_frozen,h2h,paper_main,poincare_c01_warm,rescue,seed2] | — | 0.805 | 0.002 | 0.001 | 0.001 | 0.000 | — | — | — | — | — |
| wandb | dutiful-planet-1271 [d128,dino_frozen,h2h,paper_main,poincare_c05_warm,rescue,seed1] | — | 0.796 | 0.001 | 0.000 | 0.000 | 0.000 | — | — | — | — | — |
| wandb | glorious-darkness-1272 [d128,dino_frozen,h2h,paper_main,poincare_c05_warm,rescue,seed2] | — | 0.816 | 0.001 | 0.001 | 0.000 | 0.000 | — | — | — | — | — |
| wandb | curious-wood-1270 [d128,dino_frozen,h2h,paper_main,poincare_c03_warm,rescue,seed2] | — | 0.738 | 0.000 | 0.000 | 0.000 | 0.000 | — | — | — | — | — |
| wandb | dashing-paper-1269 [d128,dino_frozen,h2h,paper_main,poincare_c03_warm,rescue,seed1] | — | 0.715 | 0.000 | 0.000 | 0.000 | 0.000 | — | — | — | — | — |
| wandb | serene-music-1268 [d128,dino_frozen,h2h,paper_main,poincare_c05_warm,rescue,seed3] | — | 0.776 | 0.001 | 0.000 | 0.000 | 0.000 | — | — | — | — | — |
| wandb | soft-water-1220 [d128,dino_frozen,euclidean_baseline,h2h,paper_main,seed3] | — | 0.877 | 0.043 | 0.020 | 0.014 | 0.004 | — | — | — | — | — |
| wandb | kind-gorge-1216 [d128,dino_frozen,h2h,paper_main,poincare_no_hyphc,seed3] | — | 0.802 | 0.001 | 0.000 | 0.000 | 0.000 | — | — | — | — | — |
| wandb | hopeful-microwave-1214 [d128,dino_frozen,h2h,paper_main,poincare_hyphc_warm,seed3] | — | 0.749 | 0.001 | 0.000 | 0.000 | 0.000 | — | — | — | — | — |
| wandb | splendid-dragon-1213 [d128,dino_frozen,euclidean_baseline,h2h,paper_main,seed2] | — | 0.877 | 0.043 | 0.021 | 0.014 | 0.004 | — | — | — | — | — |
| wandb | treasured-glade-1212 [d128,dino_frozen,h2h,paper_main,poincare_no_hyphc,seed2] | — | 0.798 | 0.001 | 0.000 | 0.000 | 0.000 | — | — | — | — | — |
| wandb | vocal-puddle-1210 [d128,dino_frozen,h2h,paper_main,poincare_hyphc_warm,seed2] | — | 0.754 | 0.001 | 0.000 | 0.000 | 0.000 | — | — | — | — | — |
| wandb | gentle-star-1211 [d128,dino_frozen,euclidean_baseline,h2h,paper_main,seed1] | — | 0.877 | 0.044 | 0.021 | 0.014 | 0.003 | — | — | — | — | — |
| wandb | swept-haze-1209 [d128,dino_frozen,h2h,paper_main,poincare_hyphc_warm,seed1] | — | 0.730 | 0.001 | 0.001 | 0.000 | 0.000 | — | — | — | — | — |
| wandb | worldly-brook-1208 [d128,dino_frozen,h2h,paper_main,poincare_no_hyphc,seed1] | — | 0.800 | 0.001 | 0.001 | 0.000 | 0.000 | — | — | — | — | — |


## cub200

| Source | Run / tag | hDP_ward | NMI_ward | NN@1 | NN@5 | NN@10 | NN@100 | R@2 | R@4 | R@8 | R@10 | R@100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wandb | autumn-star-988 [head_only_contrastive,hyperboloid,latent32,supervised_baseline] | 0.151 | 0.473 | 0.104 | 0.090 | 0.080 | 0.042 | 0.176 | 0.270 | 0.386 | 0.424 | 0.876 |

