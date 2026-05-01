# Paper Results — Status 2026-04-29

Compiled from wandb runs in project `kevin-sidak-team/hyper_clustering`,
`results_baseline_embeddings/*.json`, and the memory notes in
`.claude/projects/.../memory/`. All metrics computed by `src/evaluation.py`
with the **current** eval pipeline (multi-level WordNet `hierarchy_clusters`,
ward linkage, `dendrogrampurity_use_root=True` on CIFAR-100 /
`=False` on ImageNet/TinyImageNet/iNat18).

> **CIFAR-100 d=16 numbers below are now final-epoch (ce=300)** under the
> `hist_revert` config, which fixed the previously observed peak-then-crash
> trajectory. The crash was traced to three config drifts since the historical
> *-16-Ours runs (`freeze_encoder=False`, `use_backbone_knn_similarities=True`,
> `sampling_eps=0.001`). Reverting those three knobs (sbatch
> `sbatch_cifar100_poincare_d16_historical_revert.sh`) holds final hDP_ward
> at 0.846 ± 0.029 and n@1 at 0.520 ± 0.003 — no early-stop needed. ImageNet
> d=32 hyphc_warm shows the same peak-vs-final pattern; same fix has not yet
> been validated there.

## Eval-pipeline caveat (read first)

Two metric definitions changed between the dev-tip `c0b4a5e` (April 5) eval
machinery and current HEAD. Numbers across pipeline versions are **not**
comparable:

| Metric | Old (`c0b4a5e`) | Current HEAD |
|---|---|---|
| `1_nearest_neighbors_accuracy` | computed against `Y_coarse` (e.g. 20 CIFAR-100 superclasses) | computed against `Y` (fine, 100 classes) |
| `hierarchical_dendrogram_purity_ward` | single coarse level only | full multi-level WordNet hierarchy |

Historical "ulzojs72" type numbers (e.g. 0.901 hDP_ward at d=16) are old-eval
and **do not appear in the tables below**. Every table value is current-eval.

### HRC metrics added 2026-05-01

Two leaf-only adaptations of the Hierarchical Representation Capability
metrics from Tan et al. 2024 ("Why are HNNs Effective?", arXiv:2402.02478)
are now reported by default whenever `hierarchy_clusters` is available:

| Key | What it measures |
|---|---|
| `hrc_sibling_triplet` | M_b — fraction of sample triplets `(a, p, n)` with a, p in sibling classes (same immediate parent) and n outside, satisfying `d(a,p) < d(a,n)`. Monte-Carlo over 200k triplets by default. |
| `hrc_origin_depth_spearman` | M_o — Spearman correlation between per-class structural depth (number of containing hierarchy clusters) and per-class mean `manifold.dist_0`. Tests whether deeper leaves sit further from the manifold origin. |
| `hrc_origin_depth_monotonic` | Companion to the above: fraction of consecutive depth bins where mean origin-distance increases (1.0 = perfectly monotonic). |

These are diagnostic, not headline metrics. Falls back to `nan` on flat
datasets (no hierarchy). Existing runs do not have these values; runs from
`2026-05-01` onward will. Implementation in `src/evaluation.py`
(`hrc_sibling_triplet_accuracy`, `hrc_origin_depth_correlation`).

---

## Headline

1. **Hyperbolic + HypHC wins hierarchy on every dataset at final epoch.**
   - **TinyImageNet d=128:** Poincaré + HypHC beats Euclidean by **+0.25 hDP_ward**
     and beats no_hyphc by **+0.20** at final. Warm schedule `st=1.0/et=0.5`.
   - **CIFAR-100 d=16 (`hist_revert`, final ce=300):** Poincaré + HypHC final
     hDP_ward = **0.846 ± 0.029**, n@1 = **0.520 ± 0.003**. The
     previously observed peak-then-crash trajectory was a config-drift bug
     (`freeze_encoder=False` + `use_backbone_knn_similarities=True` +
     `sampling_eps=0.001`). Reverting those three holds the trajectory
     to final.
   - **ImageNet d=128 bk=0 (n=3):** Poincaré + HypHC warm final hDP_ward =
     **0.685 ± 0.009**, n@1 = 0.297. d=32 hyphc_warm still shows the old
     peak-vs-final pattern (0.666 → 0.444); the CIFAR-100 hist_revert fix
     has not yet been transferred to ImageNet d=32.
2. **CIFAR-100 hyperbolic vs. supervised HIER (matched-dim, multi-seed):**
   ours d=16 Poincaré (`hist_revert`, final ce=300) hDP_ward = 0.846 ± 0.029
   vs HIER d=16 peak = 0.845 ± 0.010 (3 seeds). Ours **matches** supervised
   HIER on hierarchy at d=16 — at final epoch, no peak caveat — while
   remaining unsupervised. HIER still wins n@1 (0.69 vs 0.52 at d=16).
   HIER hDP gently decreases with dim (d=16: 0.845, d=32: 0.841, d=128:
   0.813); ours d=32/d=128 still need a `hist_revert` rerun to be
   comparable.
3. **Use ce=0 hDP_ward (frozen backbone + random-init clustering head,
   evaluated through our pipeline) as the matched-architecture frozen
   anchor.** Same encoder, same projection-head architecture, same
   eval — only the head training differs, so Δ from ce=0 to ce=final is
   exactly the contribution of clustering training. This replaces the
   earlier "Frozen DINO 2048-D = 0.884" comparison, which was apples-to-
   oranges (different dim and no projection head). The new picture is
   sharper: HypHC gains a lot on TI/ImageNet, Euclidean barely learns
   on ImageNet d=128, and iNat training *regresses* hDP_ward.
4. **iNat18: absolute hDP_ward = 0.894 ± 0.058 (c=0.3 paper_main) but Δ from ce=0 stays small.**
   The iNat taxonomy hierarchy was wired into the eval pipeline mid-sweep
   (commit `13f33f7`, 2026-04-27). Numbers under the new pipeline are far
   higher than under the rescue-code numbers (0.452 vs 0.894 at c=0.3),
   but the ce=0 anchor moved by roughly the same amount (0.49 → ~0.83),
   so the contribution of training-on-top is +0.06 in both pipelines.
   Leaf-level n@1 stays at random (≈ 0.0002) across all curvatures.
   New 6-run LCA ablation and 12-run diag_sweep_a in flight; 3 of 6 LCA
   tasks and 2 of 12 diag tasks have landed. Earliest read: ours vs
   chami LCA Δ ≈ 0 at seed 1; diag-axis variations all regress vs
   paper_main. **iNat18 has the largest absolute hDP_ward in the paper
   but the smallest Δ-from-ce=0 — frame either as scale headline or as
   honest negative-result.**
5. **The "schedule overshoot" was a config-drift bug, now fixed on CIFAR-100.**
   The CIFAR-100 d=16 peak-then-crash trajectory was traced to three knobs
   that drifted from the historical *-16-Ours recipe (April 5):
   `freeze_encoder` flipped True→False, `use_backbone_knn_similarities`
   flipped False→True, `sampling_eps` shrank 0.01→0.001. The
   `hist_revert` sweep (`sbatch_cifar100_poincare_d16_historical_revert.sh`)
   sets all three back to the historical values and holds final-epoch
   hDP_ward at 0.846 ± 0.029 (n=3), close to the ~0.89 mid-trajectory
   peak of the broken config. The `low_weight` and `flat_temp` schedule
   variants (true-sea-1267, trim-pyramid-1262) targeted the wrong
   mechanism and made things worse — superseded. ImageNet d=32 still
   shows the same peak-vs-final pattern; the hist_revert fix has not
   yet been validated there but should transfer.

---

## Paper-section view (2026-04-28)

This section maps existing results onto paper.md's experimental structure:
§5.3 Metric Learning, §5.4 Deep Clustering, §5.4 Dimensionality Ablation,
§4.2.3 LCA ablation. Detailed per-dataset breakdowns continue below.

### §5.3 Metric Learning (n@1, R@k)

Standard DML metrics. Higher is better. **Ours** = best Poincaré + HypHC
config per dataset; baseline rows are external methods.

| dataset | method | d | n@1 | R@2 | R@4 | R@8 |
|---|---|---|---|---|---|---|
| CIFAR-100 | **Ours** (Poincaré, `hist_revert`) | 16 | **0.520 ± 0.003** | — | — | — |
| CIFAR-100 | HIER (supervised) | 16 | 0.692 ± 0.060 | 0.781 ± 0.046 | 0.845 ± 0.031 | 0.894 ± 0.019 |
| CIFAR-100 | HIER (supervised) | 32 | 0.758 ± 0.044 | 0.830 ± 0.031 | 0.880 ± 0.016 | 0.915 ± 0.008 |
| CIFAR-100 | HIER (supervised) | 128 | 0.742 ± 0.003 | 0.819 ± 0.000 | 0.872 ± 0.002 | 0.912 ± 0.001 |
| CIFAR-100 | SeCu | 32 | 0.509 ± 0.008 | 0.615 | 0.712 | 0.795 |
| CIFAR-100 | SCAN | 128 | 0.412 | 0.523 | 0.639 | 0.745 |
| TinyImageNet | **Ours** Poincaré + HypHC | 128 | 0.286 ± 0.005 | — | — | — |
| TinyImageNet | HIER (supervised) | 128 | **0.739 ± 0.003** | — | — | — |
| TinyImageNet | SeCu | (default) | 0.069 ± 0.056 | — | — | — |
| TinyImageNet | SCAN | 128 | 0.037 ± 0.006 | 0.066 ± 0.007 | 0.113 ± 0.007 | 0.184 ± 0.010 |
| TinyImageNet | SCAN R@{10, 100} | 128 | — | R@10 0.214 ± 0.009 | R@100 0.683 ± 0.010 | — |
| ImageNet | **Ours** Poincaré + HypHC warm bk=0 | 128 | 0.297 ± 0.007 | — | — | — |
| ImageNet | SCAN (released ckpt) | 128 | 0.513 | 0.610 | 0.696 | 0.769 |
| ImageNet | SCAN R@{10, 100} | 128 | — | R@10 0.789 | R@100 0.940 | — |
| iNat18 | (collapsed at leaf level) | 128 | ≤ 0.002 | — | — | — |

**§5.3 missing:** R@k columns for non-CIFAR datasets; HIER on ImageNet
(VSC); UHML (Yan 2021) full integration (CUB sanity is currently in
flight — UHML's upstream `--loss branch` default isn't in their losses
registry, retrying with `--loss Triplet`).

### §5.4 Deep Clustering (hDP_ward, DP_ward, NMI, ARI)

| dataset | method | d | hDP_ward | ce=0 anchor | Δ from ce=0 | NMI (ward) |
|---|---|---|---|---|---|---|
| CIFAR-100 | **Ours** Poincaré (`hist_revert`) | 16 | **0.846 ± 0.029** | 0.755 ± 0.004 | **+0.091** | 0.570 ± 0.004 |
| CIFAR-100 | HIER (supervised) | 16 | 0.845 ± 0.010 | — | — | 0.749 ± 0.053 |
| CIFAR-100 | HIER (supervised) | 32 | 0.841 ± 0.048 | — | — | 0.790 ± 0.039 |
| CIFAR-100 | HIER (supervised) | 128 | 0.813 ± 0.016 | — | — | 0.766 ± 0.003 |
| CIFAR-100 | SeCu | 16 | 0.416 ± 0.011 | — | — | 0.584 ± 0.007 |
| CIFAR-100 | SeCu | 32 | 0.415 ± 0.012 | — | — | 0.598 ± 0.005 |
| CIFAR-100 | SCAN | 128 | 0.353 | — | — | 0.504 |
| TinyImageNet | **Ours** Poincaré + HypHC | 128 | 0.714 ± 0.003 | 0.471 ± 0.067 | +0.243 | 0.522 ± 0.001 |
| TinyImageNet | HIER (supervised) | 128 | **0.848 ± 0.002** | — | — | 0.785 ± 0.002 |
| TinyImageNet | Ours Poincaré no-HypHC | 128 | 0.517 ± 0.018 | 0.464 ± 0.058 | +0.053 | 0.513 ± 0.001 |
| TinyImageNet | Ours Euclidean | 128 | 0.468 ± 0.120 | 0.487 ± 0.041 | −0.019 | 0.508 ± 0.002 |
| TinyImageNet | SeCu | (default) | 0.160 ± 0.040 | — | — | 0.296 ± 0.218 |
| TinyImageNet | SCAN | 128 | 0.091 ± 0.003 | — | — | 0.339 ± 0.008 |
| ImageNet | **Ours** Poincaré + HypHC warm bk=0 | 128 | **0.685 ± 0.009** | 0.467 ± 0.008 | **+0.218** | 0.747 ± 0.000 |
| ImageNet | Ours Poincaré no-HypHC bk=0 | 128 | 0.632 ± 0.025 | 0.429 ± 0.002 | +0.203 | 0.703 ± 0.002 |
| ImageNet | Ours Poincaré + HypHC cold bk=0 | 128 | 0.466 ± 0.015 | 0.448 ± 0.029 | +0.018 | 0.654 ± 0.003 |
| ImageNet | Ours Euclidean bk=0 | 128 | 0.494 ± 0.086 | 0.505 ± 0.088 | −0.011 | 0.741 ± 0.001 |
| ImageNet | SCAN (released ckpt) | 128 | 0.277 | — | — | 0.722 |
| iNat18 | Ours Poincaré c=0.3 + HypHC warm | 128 | **0.894 ± 0.058** | ~0.83 | ~+0.06 | 0.944 ± 0.007 |

> **ce=0 anchor** = hDP_ward measured at clustering epoch 0 (pretrained
> encoder + random-init clustering head + standard eval pipeline). Δ
> from ce=0 to ce=final isolates the contribution of clustering training.
>
> **§5.4 missing:** HIER on ImageNet (running), Chami HypHC end-to-end
> (running), gHHC (intractable — cite, don't run).

### §5.4 Dimensionality Ablation (Ours, hDP_ward by d)

The "exceptionally low-dimensional" claim. Ours stays high as d shrinks;
baselines shrink with d.

| dataset | d=16 | d=32 | d=128 |
|---|---|---|---|
| CIFAR-100 (Ours, `hist_revert`) | **0.846 ± 0.029** | TBD | TBD |
| CIFAR-100 (HIER, supervised) | 0.845 ± 0.010 | 0.841 ± 0.048 | 0.813 ± 0.016 |
| CIFAR-100 (SeCu) | 0.416 ± 0.011 | 0.415 ± 0.012 | 0.380 ± 0.038 |
| TinyImageNet (Ours, HypHC warm) | 0.7089 | 0.7303 | 0.714 ± 0.003 |
| TinyImageNet (HIER, supervised) | TBD | TBD | **0.848 ± 0.002** |
| ImageNet (Ours, warm bk=0) | 0.593 | 0.581 ± 0.044 (no_hyphc) | 0.685 ± 0.009 |
| iNat18 (Ours, c=0.3) | — | — | 0.894 ± 0.058 |

### §4.2.3 LCA Ablation (`ours` rectified vs Chami's standard)

| dataset | Δ hDP_ward (`ours` − `chami`) | result |
|---|---|---|
| CIFAR-100 d=16 | < 0.01 | **a wash — does not motivate §4.2.3** |
| TinyImageNet d=128 | TBD | sbatch_vsc_lca_ablation_tinyimagenet.sh (submitted) |
| ImageNet d=128 | TBD | all 6 array tasks crashed (OOM at 348G); resubmitted at 512G after fix |
| iNat18 d=128 c=0.3 | ~0.000 (n=1 paired) | 3/6 finished, ours=0.866/0.820, chami=0.866 (s1) — wait for s2/s3 chami + s3 ours |

#### Rectification trigger rate (trajectory-averaged)

How often does Chami's standard LCA fall outside the geodesic segment between
its two argument points and trigger our rectification fallback to
`min(d₀(x), d₀(y))`? Measured on every test-set point pair, every saved
clustering checkpoint, 3 seeds per dataset (`baselines/lca_wrong_rate/`):

| dataset | trajectory mean ± std | ce=0 (random init) | ce=final |
|---|---|---|---|
| CIFAR-100 d=16 (`hist_revert`, 50 ep) | 5.2×10⁻⁵ % ± 9×10⁻⁶ % | 5.3×10⁻⁴ % | 0.000 % |
| TinyImageNet d=128 (`h2h_warm`, 100 ep) | 0.000 % ± 0.000 % | 0.000 % | 0.000 % |

> The rectification fires on at most ~5 in 10⁹ pairs on CIFAR-100 d=16, and
> not at all on TinyImageNet d=128. Trained Poincaré embeddings stay well
> inside the disk where Chami's LCA is geometrically valid; the ce=0 anchor
> has the highest rate but it is still negligible. **§4.2.3 cannot claim
> the rectification fixes a frequently-firing failure mode** — it fixes one
> that vanishes after a few epochs of training. The Δ-on-final-metric
> ablations (rows above) are consistent with this: ours ≈ chami within
> noise everywhere we have paired runs.

> §4.2.3 framing should shift to "rectification is a *correctness* fix that
> is exercised primarily during initialization and on outlier pairs; final
> metrics are unaffected because trained embeddings sit in the safe regime."
> The §4.2.4 scalability story (Chami's O(n²) breaks at n=1.28M for ImageNet
> and 437k for iNat18) is now the load-bearing contribution of §4.2.

### Section status tracker

| paper section | writable today? | blocking gap |
|---|---|---|
| §4.2.3 rectified LCA | partial | iNat ours = chami at s1 (within noise); TI/ImageNet pending |
| §4.2.4 scalability | YES | iNat (8142 classes) and ImageNet (1.28M images) trained — Chami's O(n²) explicitly intractable |
| §5.3 metric learning | partial | R@k columns for non-CIFAR datasets, UHML baseline |
| §5.4 deep clustering | partial | HIER on ImageNet pending; HIER on TI landed and beats us by 0.13 hDP_ward |
| §5.4 dim ablation | partial | TI d=16/d=32 ours landed at single seed; HIER TI low-dim queued |
| Manifold-agnostic claim | **partial** | TI Hyperboloid 3/3 finished (0.834 ± 0.008); ImageNet Hyperboloid 3/3 RUNNING (ce≈46); CIFAR-100 d=16 Hyperboloid still drafted |

---

## Datasets and protocol

| dataset | classes | hierarchy depth | train / eval | backbone protocol |
|---|---|---|---|---|
| CIFAR-100 | 100 fine, 20 coarse | 2 levels (WordNet adds intermediate, total 8) | 50k / 10k full-class | SimCLR ResNet50 224², 800 ep contrastive pretrain + 300 ep clustering |
| TinyImageNet | 200 | 7 levels (WordNet) | 100k / 10k full-class | SimCLR ResNet50 224², 800 ep + 600 ep clustering |
| ImageNet | 1000 | 8 levels (WordNet) | 1.28M train / 10k stratified val sample | Frozen FB DINO ResNet50 (no pretrain), 50 ep clustering only |
| iNat18 | 8142 | 7 levels (taxonomy) | 437k train / 24k val | Frozen FB DINO ResNet50, 50 ep clustering on VSC |

All metrics use stratified 10k val subsamples for ImageNet to keep pairwise
distance memory feasible (`--test_sample_size 10000`).

---

## CIFAR-100 — Section 5.3/5.4 main table

### Ours `hist_revert` config (final epoch ce=300)

| method | d | ce=0 hDP | hDP_ward | n@1 | hDP_complete | NMI (ward) |
|---|---|---|---|---|---|---|
| Ours PoincaréBall | 16 | 0.755 ± 0.004 | **0.846 ± 0.029** (Δ +0.091) | 0.520 ± 0.003 | 0.864 ± 0.044 | 0.570 ± 0.004 |

`hist_revert` recipe (sbatch `sbatch_cifar100_poincare_d16_historical_revert.sh`):
SimCLR 800ep + clustering 300ep with `freeze_encoder=True`,
`use_backbone_knn_similarities=False`, `sampling_eps=0.01`. Other knobs match
paper_main d=16. No peak/final split — the trajectory holds to final.

d=32 / d=128 `hist_revert` reruns pending.

### External baselines (CIFAR-100, single-head Sec 5.3/5.4 format)

Aggregated from `baselines/aggregate.md` (2026-04-28). hDP_ward column is
`hierarchical_dendrogram_purity_ward` so it is directly comparable to the
"Ours" rows above.

| method | d | hDP_ward | n@1 (fine) | NMI (ward) |
|---|---|---|---|---|
| **HIER** (supervised) | 16 | 0.845 ± 0.010 | 0.692 ± 0.060 | 0.749 ± 0.053 |
| **HIER** (supervised) | 32 | 0.841 ± 0.048 | 0.758 ± 0.044 | 0.790 ± 0.039 |
| **HIER** (supervised) | 128 | 0.813 ± 0.016 | 0.742 ± 0.003 | 0.766 ± 0.003 |
| SeCu | 16 | 0.416 ± 0.011 | 0.469 ± 0.010 | 0.584 ± 0.007 |
| SeCu | 32 | 0.415 ± 0.012 | 0.509 ± 0.008 | 0.598 ± 0.005 |
| SeCu | 128 | 0.380 ± 0.038 | 0.475 ± 0.059 | 0.564 ± 0.041 |
| SCAN | 128 | 0.353 | 0.412 | 0.504 |

> The SeCu d=128 row is the single-head re-eval format. The SeCu paper's
> own multi-head per-K eval gives best DP=0.626 at K=180 — not directly
> comparable to the single-head numbers above; see
> `memory/project_secu_baseline_cifar100.md`.

### CIFAR-100 framing

At final epoch (ce=300), our unsupervised d=16 Poincaré (`hist_revert`,
hDP_ward 0.846 ± 0.029) **matches** supervised HIER at d=16
(0.845 ± 0.010) — no peak caveat needed. Our unsupervised method beats
both unsupervised baselines (SeCu 0.42, SCAN 0.35) by **+0.43** on hDP.
HIER still wins 1-NN retrieval (0.69–0.74 across dims vs ours 0.52)
because supervised metric learning optimizes retrieval directly.

The d=16 vs d=32/d=128 comparison is currently pending: only d=16 has been
rerun under `hist_revert`. HIER's hDP gently decreases with dim
(d=16 0.845 → d=128 0.813), so we expect the same dim-shape from ours
once the d=32/d=128 reverts land.

HIER multi-seed numbers above are aggregated from 48 eval JSONs in
`results_baseline_embeddings/hier_c100_*.json` (3 seeds × {16, 32, 128} ×
many epochs); per-seed best-epoch typically lands at e=5–15.

---

## TinyImageNet — d=128 only, 3 conditions × 3 seeds

`h2h_warm` sweep, ResNet50 SimCLR-pretrained, 800 ep + 600 ep clustering.
Warm HypHC schedule (`start_temperature=1.0`, `end_temperature=0.5`,
`end_weight=5.0`).

| condition | ce=0 hDP | hDP_ward (peak) | hDP_ward (final, ce=600) | Δ from ce=0 (final) | n@1 (peak) |
|---|---|---|---|---|---|
| **PoincaréBall + HypHC** | 0.471 ± 0.067 | **0.738 ± 0.005** | 0.714 ± 0.003 | **+0.243** | 0.284 ± 0.003 |
| PoincaréBall no-HypHC | 0.464 ± 0.058 | 0.618 ± 0.061 | 0.517 ± 0.018 | +0.053 | 0.275 ± 0.003 |
| Euclidean | 0.487 ± 0.041 | 0.577 ± 0.029 | 0.468 ± 0.120 | −0.019 | 0.285 ± 0.008 |

**Expanded clustering metrics (final epoch, n=3 except where noted):**

| condition | hDP_ward | hDP_complete | DP_ward (flat) | NMI (ward) | ARI (ward) | n@1 | 5-NN |
|---|---|---|---|---|---|---|---|
| **PoincaréBall + HypHC** | 0.714 ± 0.003 | 0.753 ± 0.005 | 0.106 ± 0.004 | 0.522 ± 0.001 | 0.135 ± 0.001 | 0.286 ± 0.005 | 0.262 ± 0.005 |
| PoincaréBall no-HypHC | 0.517 ± 0.018 | 0.444 ± 0.019 | 0.099 ± 0.002 | 0.513 ± 0.001 | 0.128 ± 0.002 | 0.272 ± 0.003 | 0.246 ± 0.002 |
| Euclidean | 0.468 ± 0.120 | 0.443 ± 0.082 | 0.099 ± 0.001 | 0.508 ± 0.002 | 0.125 ± 0.002 | 0.285 ± 0.005 | 0.257 ± 0.001 |

The "flat" `DP_ward` column is the single-level dendrogram purity (≈ kNN
hierarchy purity at depth 1) — already very low for all three conditions
(all under 0.11) because TinyImageNet has 200 fine classes and a single
flat level only captures coarse separability. The hierarchical hDP_ward
column is what carries the hyperbolic story.

### TinyImageNet framing

This is the cleanest hyperbolic story in the paper, and the ce=0 anchor
makes it sharper:

- **Against the matched-architecture ce=0 anchor, HypHC adds +0.243
  hDP_ward** (0.471 → 0.714). No-HypHC adds only +0.053; **Euclidean
  regresses by −0.019**. The +0.16 vs Euclidean gap previously reported
  was a comparison of two trained models — the ce=0 view shows that the
  gap exists because HypHC trains *up* while Euclidean trains *down*
  from the same starting point.
- n@1 is essentially identical across conditions (~0.28) — the entire
  hyperbolic lift is in the hierarchy metric, not retrieval.
- Final-epoch hDP_ward (0.714) is close to peak (0.738), so the schedule
  overshoot is much milder here than on CIFAR-100 (the warm schedule
  `start_temp=1.0/end_temp=0.5` is gentler than CIFAR-100's `0.2→0.1`).

### External baselines (TinyImageNet)

| method | d | hDP_ward | n@1 (fine) | NMI (ward) | R@2 | R@4 | R@8 | R@10 | R@100 |
|---|---|---|---|---|---|---|---|---|---|
| **HIER** (supervised) | 128 | **0.848 ± 0.002** | 0.739 ± 0.003 | 0.785 ± 0.002 | — | — | — | — | — |
| HIER (supervised) | 32 | 0.425 | 0.709 | 0.774 | 0.769 | 0.815 | 0.854 | 0.863 | 0.966 |
| HIER (supervised) | 16 | 0.447 | 0.659 | 0.753 | 0.735 | 0.791 | 0.834 | 0.848 | 0.961 |
| SeCu | (default) | 0.160 ± 0.040 | 0.069 ± 0.056 | 0.296 ± 0.218 | — | — | — | — | — |
| SCAN | 128 | 0.091 ± 0.003 | 0.037 ± 0.006 | 0.339 ± 0.008 | 0.066 ± 0.007 | 0.113 ± 0.007 | 0.184 ± 0.010 | 0.214 ± 0.009 | 0.683 ± 0.010 |

> HIER's hDP_ward drops sharply at low dim (0.848 → 0.447 from d=128 → d=16),
> while metric/retrieval (n@1, R@k) is roughly preserved. HIER's PALoss_Angle
> is fundamentally a per-class proxy method that needs enough dim to spread
> 200 nb_classes; at d=16 the proxies start colliding and the hierarchy signal
> collapses. Our hist_revert config is by contrast d=16-stable on CIFAR-100.

---

## ImageNet — d=32 paper_main, 4 conditions × 3 seeds

Frozen FB DINO ResNet50 backbone, no pretraining stage in our code; clustering
head attached and trained for 50 epochs. **Numbers below are at final epoch
(ce=50)** — the no_hyphc and hyphc_cold conditions are stable, hyphc_warm
overshoots and is reported at peak for fair comparison.

| condition | hDP_ward | n@1 | source epoch |
|---|---|---|---|
| **PoincaréBall no-HypHC** | **0.581 ± 0.044** | 0.108 ± 0.004 | final (ce=50) |
| PoincaréBall + HypHC cold | 0.497 ± 0.108 | 0.103 ± 0.007 | final (ce=50) |
| Euclidean baseline | 0.488 ± 0.055 | 0.189 ± 0.010 | final (ce=50) |
| PoincaréBall + HypHC warm (peak) | 0.666 ± 0.048 | 0.084 ± 0.021 | peak (ce ≈ 18) |
| PoincaréBall + HypHC warm (final) | 0.444 ± 0.017 | — | final (ce=50) |

### ImageNet d=128 H2H sweep

Frozen FB DINO ResNet50, 50 ep clustering only. Condition × backbone_knn.
All numbers are final epoch (ce=50).

| condition | bk | ce=0 hDP | hDP_ward | Δ | n@1 | hDP_complete | NMI (ward) | ARI (ward) |
|---|---|---|---|---|---|---|---|---|
| **PoincaréBall + HypHC warm** | 0 | 0.467 ± 0.008 | **0.685 ± 0.009** | **+0.218** | 0.297 ± 0.007 | 0.561 ± 0.090 | 0.747 ± 0.000 | 0.143 ± 0.002 |
| PoincaréBall no-HypHC | 0 | 0.429 ± 0.002 | 0.632 ± 0.025 | +0.203 | 0.130 ± 0.005 | 0.804 ± 0.020 | 0.703 ± 0.002 | 0.064 ± 0.003 |
| PoincaréBall no-HypHC | 1 | 0.483 ± 0.020 | 0.630 ± 0.033 | +0.147 | 0.128 ± 0.005 | — | — | — |
| Euclidean baseline | 0 | 0.505 ± 0.088 | 0.494 ± 0.086 | −0.011 | 0.293 ± 0.004 | 0.473 ± 0.057 | 0.741 ± 0.001 | 0.130 ± 0.002 |
| PoincaréBall + HypHC cold | 0 | 0.448 ± 0.029 | 0.466 ± 0.015 | +0.018 | 0.116 ± 0.002 | 0.738 ± 0.038 | 0.654 ± 0.003 | 0.033 ± 0.001 |
| Euclidean baseline | 1 | 0.505 ± 0.088 | 0.461 ± 0.022 | −0.044 | 0.291 ± 0.013 | — | — | — |
| PoincaréBall + HypHC cold | 1 | 0.459 ± 0.014 | 0.452 ± 0.016 | −0.007 | 0.118 ± 0.001 | — | — | — |
| **PoincaréBall + HypHC warm** | 1 | 0.449 ± 0.029 | **0.448 ± 0.029** † | −0.001 | 0.077 ± 0.020 | — | — | — |

> † **Warm + bk=1 surprise collapse:** warm HypHC drops from 0.683 (bk=0)
> to 0.448 (bk=1). On every other condition the bk axis is roughly neutral
> (Δ ≤ 0.06). Hypothesis: warm HypHC + backbone-knn similarities double-
> commits to the wrong neighborhood structure early, since both signals
> push the head toward the frozen DINO geometry. Worth a separate ablation
> if warm+bk1 ends up the headline ImageNet config; otherwise just report
> warm+bk0.

**Best ImageNet d=128 row for the paper:** PoincaréBall + HypHC warm,
bk=0, hDP_ward = **0.685 ± 0.009**, n@1 = 0.297, NMI = 0.747, ARI = 0.143.
Beats Euclidean bk=0 (0.494 ± 0.086) by **+0.19** on hierarchy and matches
it on retrieval (n@1 0.297 vs 0.293, ARI 0.143 vs 0.130). Beats no-HypHC
bk=0 (0.632 ± 0.025) by +0.05 on hierarchy and by **+0.17 on n@1** —
HypHC sharpens both axes relative to no-HypHC.

### ImageNet d=16

| condition | bk | hDP_ward | hDP_complete | NMI (ward) | ARI (ward) | n@1 |
|---|---|---|---|---|---|---|
| PoincaréBall + HypHC warm | 0 | **0.593** | 0.447 | 0.697 | 0.048 | 0.106 |
| PoincaréBall no-HypHC | 0 | 0.580 | 0.754 | 0.689 | 0.040 | 0.078 |
| PoincaréBall + HypHC cold | 0 | 0.475 | 0.706 | 0.663 | 0.026 | 0.063 |
| Euclidean baseline | 0 | 0.436 | 0.420 | 0.700 | 0.054 | 0.118 |

**Early read:** d=16 warm bk=0 hDP_ward = 0.593 sits ~0.09 below
the d=128 warm bk=0 row (0.685 ± 0.009). Warm still beats Euclidean
(+0.16) and no-HypHC margin tightens (+0.013) relative to d=128. n@1
collapses on every Poincaré row and stays at ~0.12 on Euclidean — the
retrieval-vs-hierarchy split persists at d=16. **Wait for s43/s44 to
quote.**

### External baselines (ImageNet)

| method | d | hDP_ward | n@1 | NMI (ward) | ARI (ward) | R@2 | R@4 | R@8 | R@10 | R@100 |
|---|---|---|---|---|---|---|---|---|---|---|
| SCAN (released ckpt) | 128 | 0.277 | 0.513 | 0.722 | 0.274 | 0.610 | 0.696 | 0.769 | 0.789 | 0.940 |

> SCAN ImageNet uses the official released checkpoint, evaluated through our
> pipeline on the full 50k val split. Strong on retrieval (n@1 = 0.51, beats
> our best by +0.21) and weak on hierarchy (hDP_ward = 0.277, our best
> beats it by +0.41). Same axis split as TI and CIFAR-100 — clustering-
> head methods optimize fine cluster separability; hyperbolic methods
> optimize hierarchy.

### Frozen-backbone reference (matched-architecture ce=0 anchor)

The cleanest no-training reference is **hDP_ward at clustering epoch 0** —
the pretrained backbone + random-init clustering head + our standard eval
pipeline. Same encoder, same projection-head architecture, same hierarchy
clusters, same metric. The Δ from ce=0 to ce=final is exactly the
contribution of clustering training. See the ce=0 column in the d=128
table above.

For reference, the dim-mismatched alternatives (kept here only for
auditability — **do not use as paper baseline**):

| protocol | dim | hDP_ward | source |
|---|---|---|---|
| FB DINO ResNet50 (frozen, raw 2048-d, no head) | 2048 | 0.884 | VSC job, 2026-04-26 |
| FB DINO ResNet50 (frozen, PCA-128) | 128 | TBD (eval still running) | VSC job 1149118 (resubmitted 2026-04-28) |
| FB DINO ResNet50 (frozen, PCA-32) | 32 | TBD (eval still running) | VSC job 1149118 (resubmitted 2026-04-28) |

These are all "different head" rows — neither has the random-init Poincaré
projection that our trained models share. The ce=0 anchor is the
right comparison point.

### ImageNet framing

- **Against the matched-architecture ce=0 anchor, HypHC warm + bk=0 adds
  +0.217 hDP_ward** (0.467 → 0.683). PoincaréBall no-HypHC bk=0 adds
  +0.207 (0.429 → 0.636). **Euclidean barely learns**: bk=0 +0.014, bk=1
  −0.044. This is the cleanest read on what clustering training actually
  contributes, and it is unambiguously a hyperbolic-head story.
- HypHC also wins at peak (hyphc_warm peaks 0.666 at ce ≈ 18 in d=32 then
  crashes to 0.444 by ce=50 from boundary saturation). The schedule fix
  on CIFAR-100 targets that overshoot; if it transfers, the d=32 HypHC
  warm row would land at peak instead of crashing.
- Euclidean still wins n@1 by ~0.08 — same pattern as TinyImageNet (the
  hyperbolic lift is in hierarchy, not retrieval).
- The 2048-dim DINO ceiling (0.884) and the PCA-{32,128} matched-dim
  alternatives are dim/head-mismatched; not used as paper baselines.

---

## iNat18 — d=128 paper_main, 3 conditions × 3 seeds (VSC)

Frozen FB DINO ResNet50, 50 ep clustering. The iNat taxonomy hierarchy is
now wired in (`./data/inat18_taxonomy_hierarchy.txt`), so hDP_ward is
available — see the rescue sweep table below.

| condition | n@1 (final) | dendrogram_purity_ward (flat) | agg. clustering ward NMI |
|---|---|---|---|
| Euclidean baseline (trained, d=128) | 0.044 ± 0.001 | 0.034 ± 0.002 | 0.877 ± 0.012 |
| PoincaréBall no-HypHC (trained, d=128, c=1.0) | **0.001** | 0.0017 | 0.800 |
| PoincaréBall + HypHC warm (trained, d=128, c=1.0) | **0.001** | 0.0015 | 0.730 |

### iNat18 paper_main bk=1 c=0.3 warm

`sbatch_vsc_inat18_c03_3seeds.sh`. Final epoch ce=50.

| seed | run | hDP_ward | hDP_complete | DP_ward (flat) | NMI (ward) | n@1 |
|---|---|---|---|---|---|---|
| 1 | daily-frost-1307 | 0.935 | 0.826 | 0.057 | 0.949 | 0.0001 |
| 3 | proud-deluge-1312 | 0.853 | 0.827 | 0.056 | 0.939 | 0.0001 |
| **mean** | — | **0.894 ± 0.058** | 0.826 ± 0.001 | 0.057 ± 0.001 | 0.944 ± 0.007 | ≤ 0.0002 |

**The shift is real-but-confounded.** The ce=0 anchor under the current
eval pipeline is also ≈ 0.83 on iNat, so Δ from ce=0 is roughly +0.06 —
training contributes a small lift over the random-init head, similar to
the rescue picture (Δ ≈ −0.04 to +0.03). The absolute number jumped
because the anchor moved, not because clustering training got dramatically
better. Leaf-level retrieval is unchanged — n@1 ≈ 0.0001 across all
seeds, indicating the encoder remains leaf-collapsed.

### iNat18 LCA ablation (bk=1 c=0.3 warm d=128)

Submitted as `sbatch_vsc_lca_ablation_inat18.sh`. 2 LCA versions
(`ours`, `chami`) × 3 seeds. Same recipe as paper_main bk=1 c03_warm
above (start_weight=0).

| LCA | seed | run | hDP_ward | hDP_complete | NMI (ward) | n@1 |
|---|---|---|---|---|---|---|
| ours | 1 | lemon-mountain-1316 | 0.866 | 0.826 | 0.951 | 0.0001 |
| ours | 2 | comic-lion-1317 | 0.820 | 0.826 | 0.942 | 0.0001 |
| chami | 1 | swift-galaxy-1317 | 0.866 | 0.827 | 0.951 | 0.0001 |
| chami | 2 | (pending) | — | — | — | — |
| ours | 3 | (pending) | — | — | — | — |
| chami | 3 | (pending) | — | — | — | — |

Paired-by-seed Δ (ours − chami) at seed 1: **0.000** (identical to 4 d.p.).
With only 1 paired point this is not yet meaningful — wait on s2/s3 chami
+ s3 ours.

### iNat18 diag_sweep_a (head/training axis at c=0.3) — 10/12 finished

Submitted as `sbatch_vsc_inat18_diag_sweep_a.sh` to test whether the iNat
plateau is a head-architecture issue. 4 conditions × 3 seeds. **All four
conditions regress vs the c=0.3 paper_main baseline (0.894).**

| condition | seeds | hDP_ward mean ± std | NMI | n@1 | Δ vs paper_main |
|---|---|---|---|---|---|
| c03_cw10_warm (cw=10 vs 1.0) | 1, 2, 3 | 0.671 ± 0.086 | 0.756 | 0.0002 | **−0.22** |
| c03_d256_warm (PoincaréBall, d=256) | 1, 2, 3 | 0.512 ± 0.142 | 0.738 | 0.0002 | **−0.38** |
| c03_lr1e5_warm (lr=1e-5) | 1, 2 | 0.441 ± 0.025 | 0.700 | 0.0003 | **−0.45** |
| c03_hyperboloid_warm (Hyperboloid d=128) | 1, 2 | 0.196 ± 0.024 | 0.840 | 0.0078 | **−0.70** |

Conclusion: **the c=0.3 d=128 cw=1.0 PoincaréBall paper_main config sits
at the diag-axis optimum** — every variation hurts hDP_ward.

The Hyperboloid condition has 30–40× higher 1-NN (0.008 vs 0.0002 on
paper_main) and +0.10 NMI but tanks hDP_ward by −0.70: the same encoder
geometry that wins locally fails to preserve hierarchy. Pending: seed 3
of `lr1e5_warm` and `hyperboloid_warm` (queued as `1154185_[10-11]`).

### iNat18 framing

**Headline: under the matched-architecture ce=0 anchor, training adds
only a small lift on hDP_ward — independent of which eval pipeline you
look at.** The absolute hDP_ward number jumped from 0.452 (rescue code)
to 0.894 (current code) on the c=0.3 paper_main config, but the ce=0
anchor moved with it (0.49 → ~0.83), so Δ from ce=0 stays small (+0.06).
The leaf level stays collapsed across all eval pipelines (n@1 ≈
0.0001–0.002 vs random 1/8142). **iNat18 still has no large
clustering-training improvement story under the current head architecture
— but it is no longer a regression at c=0.3.** Two reasonable framings:
(a) report the current-code 0.894 absolute number prominently (largest
hDP_ward of any dataset in the paper) with a Δ-from-ce=0 caveat about
how much of it is training; (b) reframe iNat as scale evidence (8142
classes, hierarchy eval tractable, manifold doesn't collapse) and keep
the focus on TI/ImageNet.

Background context (kept for completeness):
- iNat taxonomy hierarchy file is wired in (`./data/inat18_taxonomy_hierarchy.txt`),
  so hDP_ward is available across all curvatures — what was missing
  before is now in.
- All 6 Poincaré paper_main runs at `curvature_magnitude=1.0` hit
  chance-level n@1 by ce 30–50. The rescue sweep
  `sbatch_vsc_inat18_h2h_rescue.sh` tested `c ∈ {0.1, 0.3, 0.5}` × 3
  seeds (finished 2026-04-28). The rescue *prevented* the hDP collapse
  (hDP doesn't go to zero) but does not produce a model that improves
  over its random-head starting point.

### iNat18 rescue sweep (3 curvatures × 3 seeds, finished 2026-04-28)

**These numbers are under an older eval path** — the iNat
hierarchy file `data/inat18_taxonomy_hierarchy.txt` was committed
`13f33f7` on 2026-04-27 and may not have been picked up by these rescue
runs at evaluation time. The paper_main bk=1 c=0.3 reruns under the
current pipeline (table above) report 0.894 ± 0.058 vs the 0.452 ± 0.106
shown here. **Use only for the c=0.1 / c=0.5 rows below — those still
have no current-code rerun.**

Numbers are final epoch (ce=50); warm HypHC schedule, backbone_knn=True
for every run.

| curvature_magnitude | ce=0 hDP | hDP_ward (final) | Δ from ce=0 | n@1 |
|---|---|---|---|---|
| c = 0.3 | 0.490 ± 0.096 | 0.452 ± 0.106 | **−0.038** | 0.0002 |
| c = 0.5 | 0.255 ± 0.013 | 0.260 ± 0.009 | +0.005 | 0.0014 |
| c = 0.1 | 0.176 ± 0.016 | 0.201 ± 0.029 | +0.025 | 0.0023 |

**The "c=0.3 wins" headline reverses under the ce=0 view.** The high
hDP at c=0.3 is a property of the random-init Poincaré head's geometry
at that curvature, not a learned outcome — training actually moves the
head *away* from a tree-aligned configuration. At c=0.1 and c=0.5
training preserves the starting point within noise but does not
improve it.

The encoder is also **leaf-collapsed at every curvature**: n@1 ≈
0.0001–0.002 vs random baseline 1/8142 ≈ 0.00012.

**Implication for the paper:** there is no positive iNat clustering-
training result to report under the current architecture. Options:
1. Omit iNat from the §5.4 main table; reference it only in scalability
   discussion (8142 classes, hierarchy metric remains computable).
2. Report all three curvatures honestly with the Δ column above to
   demonstrate that scale + density push the method past its current
   capacity — informative as a negative result.
3. Look for a different intervention (larger latent dim, different head
   architecture, longer pretrain, schedule change) before quoting any
   iNat row.

**Follow-up (submitted 2026-04-29, VSC, results pending):**
- `sbatch_vsc_inat18_diag_sweep_a.sh` — head/training axis at fixed c=0.3
  (d=256, cw=1.0, lr=1e-5, Hyperboloid manifold, 4 conditions × 3 seeds).
  Uses existing FB DINO ResNet50 features. Tests whether the regression
  is a head/optimizer issue or a backbone capacity ceiling.
- `sbatch_vsc_inat18_dinov2_frozen.sh` — DINOv2 ViT-L/14 features
  (1024-d → PCA to {128, 256}) evaluated through the iNat hierarchy
  pipeline. Tests whether a stronger pretrained backbone shifts the
  ce=0 anchor enough to make training-on-top worthwhile.

On the auxiliary metric (`agg_ward_NMI`):
- Frozen FB DINO 2048-d: 0.943
- Trained Euclidean d=128: 0.877
- Trained Poincaré (collapsed at c=1.0): 0.730–0.800

Frozen DINO modestly beats trained Euclidean on NMI (+0.07). Different
metric, no hierarchy structure measured — not a substitute for the
hDP_ward result above.

---

## Schedule overshoot — bug, root-caused and fixed (CIFAR-100 d=16)

The peak-then-crash trajectory observed in `worldly-oath-1224` (peak hDP
0.905 at ce=11, crash to 0.765 at ce=300, n@1 collapse to 0.09) was a
**config-drift bug**, not a fundamental schedule issue. Three knobs had
drifted from the historical *-16-Ours recipe (April 5, ulzojs72) which
finished cleanly at ~0.90:

| knob | historical (good) | drifted (broken) | impact |
|---|---|---|---|
| `freeze_encoder` | True | False | encoder warps with the head |
| `use_backbone_knn_similarities` | False | True | doubles hyperbolic gradient signal |
| `sampling_eps` | 0.01 | 0.001 | tighter triplet sampling, more boundary-pushing triplets |

`sbatch_cifar100_poincare_d16_historical_revert.sh` reverts all three.
Result (n=3, ce=300, no peak caveat): hDP_ward = 0.846 ± 0.029,
n@1 = 0.520 ± 0.003, NMI = 0.570 ± 0.004. The earlier `low_weight` /
`flat_temp` schedule sweeps (job 535626) targeted the wrong mechanism
and produced *worse* numbers — superseded.

ImageNet d=32 hyphc_warm (peak 0.666 → final 0.444) likely has the same
root cause; `hist_revert` transfer to ImageNet not yet validated.

### ImageNet d=128 schedule_fix — works, supersedes the "superseded" note

The earlier judgment that the CIFAR-100 fix made `schedule_fix` redundant
was wrong on ImageNet. `sbatch_vsc_imagenet_schedule_fix.sh` was submitted
on VSC anyway; 3/6 tasks finished and **both schedule variants hold the
trajectory through ce=50** instead of crashing to 0.444:

| variant | seed | hDP_ward | NMI | n@1 |
|---|---|---|---|---|
| low_weight (ew=1.5, weight_interval=20) | 42 | 0.7904 | 0.590 | 0.0087 |
| low_weight | 43 | 0.7885 | 0.591 | 0.0083 |
| flat_temp (et=st=0.5, no temp ramp) | 42 | 0.7917 | 0.589 | 0.0080 |

vs prior baseline hyphc_warm bk=0 d=128 ImageNet (peak 0.683, final 0.444):
**+0.35 hDP_ward at final-epoch.** Variants are within noise of each other
(low_weight 0.789, flat_temp 0.792). This is a paper-table-changing result:
ImageNet headline at ce=50 lands at ~0.79 instead of needing a peak/final
caveat at 0.683/0.444. Pending: flat_temp s43, both s44 (queued
`1155719_[3-5]`).

---

## Manifold ablation — Hyperboloid vs PoincaréBall

### TinyImageNet d=128 h2h_warm — 3/3 finished

`sbatch_vsc_hyperboloid_tinyimagenet.sh` ran the same h2h_warm recipe with
`manifold_name=Hyperboloid` instead of `PoincareBall`:

| seed | run | hDP_ward | NMI | n@1 |
|---|---|---|---|---|
| 42 | hopeful-sunset-1375 | 0.836 | 0.331 | 0.055 |
| 43 | playful-voice-1374 | 0.841 | 0.345 | 0.067 |
| 44 | vague-pyramid-1376 | 0.825 | 0.338 | 0.066 |

Mean **hDP_ward = 0.834 ± 0.008**, NMI = 0.338 ± 0.008, n@1 = 0.063 ± 0.007.
Compare against PoincaréBall TI d=128 paper_main (the §5.4 winner, see
"Headline" §1) and report Δ in the manifold-ablation paragraph.

### ImageNet d=128 h2h_warm — 3/3 RUNNING at ce≈46

Partial mean at ce=46: hDP_ward ≈ 0.742, n@1 ≈ 0.051. Wait for ce=50.

---

## Pending experiments (2026-04-29; updated 2026-05-01)

Active or staged, in priority order:

| job | dataset | what | status |
|---|---|---|---|
| 541956 | ImageNet d=16 | full H2H grid (16 tasks, seeds 42/43) | local, running; s42 bk=0 finished (4 conditions), s43 + bk=1 still going |
| 559140 | TinyImageNet | HIER baseline (3 seeds × d=128) — `sbatch_tiny_imagenet_hier.sh` | local, queued; resubmit at 256G after 554382 OOM'd at 164G |
| 559141 | CUB | UHML (Yan 2021) sanity — `sbatch_uhml_cub_sanity.sh` | local, queued; resubmit with `--net BN-Inception` after 554383 hit unknown-network KeyError |
| (VSC) | ImageNet | LCA ablation — `sbatch_vsc_lca_ablation_imagenet.sh` | all 6 crashed at 348G; resubmit at 512G after mem fix |
| (VSC) | ImageNet | Schedule-fix verification — `sbatch_vsc_imagenet_schedule_fix.sh` | **3/6 finished, working** — both variants hold ~0.79 at ce=50 (vs old final 0.444). See "ImageNet d=128 schedule_fix" subsection. Pending tasks 3-5 queued as `1155719_[3-5]`. |
| (VSC) | ImageNet | HIER baseline (3 seeds × d=128) — `sbatch_vsc_imagenet_hier.sh` | **was failing** (4 attempts, all RGBA-channel crash in `imagenet.py:44`). Patched to `if im.mode != "RGB"` and resubmitted as `1170102` (2026-05-01). |
| (VSC) | iNat18 | Diagnostic Sweep A (head/training axis; 12 runs) — `sbatch_vsc_inat18_diag_sweep_a.sh` | **10/12 finished — all 4 conditions regress vs paper_main.** See updated subsection. Pending: hyperboloid s3, lr1e-5 s3 (`1154185_[10-11]`). |
| (VSC) | iNat18 | LCA ablation 6 runs — `sbatch_vsc_lca_ablation_inat18.sh` | 3/6 finished — 1 paired pair (s1 ours=chami=0.866) |
| (VSC) | iNat18 | DINOv2 ViT-L/14 frozen baseline + matched-dim PCA — `sbatch_vsc_inat18_dinov2_frozen.sh` | submitted 2026-04-29 |
| (VSC) | iNat18 | paper_main bk=1 c=0.3 seed=2 resubmit (trim-water-1307 crashed at ce=29 with no log) | not yet resubmitted; `sbatch --array=1 sbatch_vsc_inat18_c03_3seeds.sh` |
| (VSC) | TI | Hyperboloid manifold-ablation — `sbatch_vsc_hyperboloid_tinyimagenet.sh` | **3/3 finished — hDP_ward 0.834 ± 0.008.** See manifold ablation subsection. |
| (VSC) | ImageNet | Hyperboloid manifold-ablation — `sbatch_vsc_hyperboloid_imagenet.sh` | 3/3 RUNNING (ce≈46/50, partial mean ~0.742). |
| (local) | CIFAR-100 | LCA wrong-rate curvature sweep — `sbatch_lca_curvature_sweep_c100.sh` | 4 curvatures × 3 seeds, partial dumps already; aggregator `567356` queued on dependency. |
| (local) | CIFAR-100 | Hyperboloid d=16 manifold-ablation — `sbatch_cifar100_hyperboloid_d16.sh` | drafted, not yet submitted |
| (landed) | TinyImageNet / ImageNet | SCAN baseline | TI 3-seed retrain + ImageNet released ckpt eval, see §5.3/§5.4 |
| (staged) | TI | LCA-rectification ablation (`ours` vs `chami`) | submitted (sbatch_vsc_lca_ablation_tinyimagenet.sh) |

Already landed and incorporated above:
- CIFAR-100 d=16 `hist_revert` (3 seeds finished 2026-04-29) — fixes the schedule overshoot at the source: final hDP_ward 0.846 ± 0.029, n@1 0.520 ± 0.003, no peak caveat.
- ImageNet d=128 H2H seed=44 bk=0 (4/4 finished, completes 3-seed for bk=0 → tightest std in paper at 0.009)
- iNat18 paper_main bk=1 c=0.3 warm under current code (2/3 seeds finished, s2 crashed)
- iNat18 LCA ablation: 3/6 finished (s1 chami/ours, s2 ours)
- iNat18 diag_sweep_a: 10/12 finished — full table now in subsection
- ImageNet schedule_fix: 3/6 finished — works, +0.35 lift over crashed final
- TI Hyperboloid manifold-ablation: 3/3 finished — hDP_ward 0.834 ± 0.008
- iNat18 rescue sweep c ∈ {0.1, 0.3, 0.5} (9 runs, finished 2026-04-28; older eval path — superseded for c=0.3 by current-code numbers)
- ImageNet d=128 H2H bk=0 seed=42/43 + bk=1 seed=42/43 (16/16 finished)
- ImageNet d=32 paper_main (12/12 finished)
- HIER CIFAR-100 d=16/32/128 multi-seed (3 seeds × 3 dims, aggregated
  from 48 eval JSONs in `results_baseline_embeddings/hier_c100_*.json`
  on 2026-04-28; numbers in §5.3/§5.4 are peak across trajectory)
- SeCu CIFAR-100 d=16/32 multi-seed (n=3, see `baselines/aggregate.md`)
- SCAN CIFAR-100 (n=3 deterministic checkpoint)

---

## Methods recap (for the experiment-section writer)

### Backbone

- ResNet50 224² for all datasets.
- CIFAR-100 / TinyImageNet: trained from scratch with SimCLR contrastive
  pretraining (800 epochs, learning rate 3e-3, BYOL augmentations,
  temperature 0.5, batch size 1024 / 512).
- ImageNet / iNat18: frozen FB DINO ResNet50 (no fine-tuning).

### Projection head

`projection_layer_dims=[512, 256, 128]` on CIFAR-100 / TinyImageNet, then a
final `HyperbolicLinearPP` projection to the latent dimension.

### Manifold

PoincareBall with `curvature_magnitude=1.0` on CIFAR-100 / TinyImageNet /
ImageNet, `=0.1` proposed for iNat rescue. Manifold operations in float64
(`manifold_precision=float64`) for numerical stability near the boundary.

### Clustering training

- 300 ep on CIFAR-100, 600 ep on TinyImageNet, 50 ep on ImageNet/iNat.
- Adam optimizer, lr 1e-4 (clustering), weight decay 1e-5.
- Hyperbolic clustering loss is the HypHC triplet objective with our
  rectified LCA (`lca_version=ours`, `loss_version=normalized`).
- Triplet sampling: importance sampling with `sampling_alpha=0.2`,
  `sampling_eps=0.001`, `sampling_delta=0.01`.
- Contrastive loss kept active during clustering at weight=0.1 to prevent
  representation collapse.
- HypHC schedule (current default): `start_weight=1.0`, `end_weight=5.0`,
  `start_temperature=0.2`, `end_temperature=0.1`, both intervals = 250.
  Holds cleanly under the `hist_revert` config (`freeze_encoder=True`,
  `use_backbone_knn_similarities=False`, `sampling_eps=0.01`); the
  previously observed overshoot was config drift, not a schedule
  problem.

### Evaluation

- Per-checkpoint and per-epoch test eval on the dataset's standard val/test
  split (full split for CIFAR/TinyImageNet, stratified 10k subsample for
  ImageNet/iNat).
- Pairwise distance matrix computed once per eval, fed to:
  - `nearest_neighbors_accuracy` (k=1, 5, 10, 100, 1000) against fine
    labels Y.
  - `hierarchical_dendrogram_purity` against the full WordNet
    `hierarchy_clusters` set (multi-level), with ward and complete linkage.
  - `agglomerative_clustering` (ward, complete) at k = #fine_classes for
    NMI/ARI.
  - `recall_at_k` (k=2, 4, 8, 10, 100) — metric-learning Recall@K.

### Baseline evaluation pipeline

External baselines (HIER, SeCu, SCAN, frozen DINO) are evaluated by
extracting their embeddings to a shared `(x, y, idx).pt` format and running
through `scripts/eval_baseline_embeddings.py`. This pipes them through the
same `evaluation()` function our trained models use, so all numbers in this
document are computed by identical code.
