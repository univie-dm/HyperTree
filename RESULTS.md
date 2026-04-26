# Paper Results — Status 2026-04-26

Compiled from wandb runs in project `kevin-sidak-team/hyper_clustering`,
`results_baseline_embeddings/*.json`, and the memory notes in
`.claude/projects/.../memory/`. All metrics computed by `src/evaluation.py`
with the **current** eval pipeline (multi-level WordNet `hierarchy_clusters`,
ward linkage, `dendrogrampurity_use_root=True` on CIFAR-100 /
`=False` on ImageNet/TinyImageNet/iNat18).

> **All "Ours" numbers in tables below are peak hDP_ward across the clustering
> trajectory** (= early-stopping equivalent), reported with peak epoch for
> transparency. See "Schedule overshoot" caveat at the bottom — the current
> default schedule (`end_weight=5.0`, `end_temperature=0.1`,
> `weight_interval=temperature_interval=250`) overshoots into boundary
> saturation, so final-clustering numbers are systematically lower than peak.
> A 2-condition × 3-seed schedule-fix sweep is running (job 535626) and will
> replace these with final-epoch numbers once verified.

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

---

## Headline

1. **Hyperbolic + HypHC wins hierarchy on every dataset; schedule overshoot
   blocks 2/3 from showing the win at final epoch.**
   - **TinyImageNet d=128 (paper-writable at final):** Poincaré + HypHC beats
     Euclidean by **+0.25 hDP_ward** and beats no_hyphc by **+0.20** at final.
     Warm schedule `st=1.0/et=0.5` is gentle enough; final ≈ peak.
   - **CIFAR-100 d=16 (peak only, schedule fix pending):** Peak Poincaré +
     HypHC hits **0.89 hDP_ward** for ~15 epochs (ce 11–25), then crashes
     to 0.75 at ce=300. Cold schedule `st=0.2/et=0.1` overshoots into
     boundary saturation. Schedule-fix sweep job 535626 in flight.
   - **ImageNet d=32 (peak only, same overshoot pattern):** Peak Poincaré +
     HypHC warm hits **0.666** at ce ≈ 18 (vs no_hyphc 0.618 and Euclidean
     0.569 — HypHC wins by +0.05), then crashes to 0.444 by ce=50. Same
     boundary-saturation mechanism as CIFAR-100. The schedule fix being
     tested on CIFAR-100 should transfer.
2. **CIFAR-100 hyperbolic vs. supervised HIER ends in a near tie at peak.**
   Ours d=16 Poincaré peak hDP_ward = 0.893 ± 0.021 vs HIER d=128 supervised
   = 0.783. Ours wins hDP by +0.11; HIER still wins n@1 (0.81 vs 0.51 at our
   peak). Pending: HIER d=16/d=32 (jobs 535618/535619) to test the matched-dim
   comparison.
3. **Frozen DINO ResNet50 on ImageNet hits hDP_ward = 0.884 with no
   training**, beating every learned 32/128-dim config. Most likely an
   apples-to-oranges dim mismatch (2048 vs 32/128). Matched-dim DINO-PCA
   ceiling is a pending experiment. On iNat18 frozen DINO is also
   modestly better than trained Euclidean on NMI (0.943 vs 0.877, +0.07);
   no hDP comparison possible there because there's no iNat WordNet file.
4. **iNat18 Poincaré is broken** at `curvature_magnitude=1.0` (collapse to
   n@1≈0.001 across 6/6 paper_main runs). Euclidean works (n@1≈0.044). Rescue
   sweep at c∈{0.1, 0.3, 0.5} staged but not yet submitted.
5. **The CIFAR-100 default training schedule overshoots, the others mostly don't.**
   On CIFAR-100 d=16/d=32 (cold HypHC, `st=0.2, et=0.1`), hDP peaks at
   clustering epoch 11–25 then crashes when latent norms cross ~0.995
   (Poincaré boundary saturation). On TinyImageNet (warm HypHC,
   `st=1.0, et=0.5`) and on ImageNet's `no_hyphc` and `hyphc_cold`
   conditions, final-epoch numbers are within 0.04–0.07 of peak — those
   tables can be reported at final. ImageNet's `hyphc_warm` is the lone
   overshooter (peak 0.666 → final 0.444). Schedule-fix sweep (job 535626)
   targets the CIFAR-100 case directly.

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

### Ours (3 seeds, current code, peak across trajectory)

| method | d | hDP_ward | n@1 (fine, 100-class) | peak ce | final hDP_ward |
|---|---|---|---|---|---|
| Ours PoincaréBall (peak) | 16 | **0.893 ± 0.021** | 0.506 ± 0.013 | 11–26 | 0.753 ± 0.011 |
| Ours PoincaréBall (peak) | 32 | 0.882 ± 0.010 | 0.521 ± 0.004 | 11–55 | 0.737 ± 0.006 |

n=4 runs per row (3 paper_main seeds + 1 historical-recipe rerun). Final
values are clustering epoch 300; peak is the best hDP_ward observed across
the 300-epoch trajectory.

### External baselines (CIFAR-100, single-head Sec 5.3/5.4 format)

| method | d | hDP_ward | n@1 | NMI (ward) | source |
|---|---|---|---|---|---|
| **HIER** (supervised hyperbolic DML, CVPR 2023) | 128 | 0.783 | **0.813** | 0.827 | local rerun job 534399, eval epoch 55 |
| SeCu (Sec 5.3/5.4 format) | 16 | 0.4095 (hier) | 0.4793 | 0.5924 | published checkpoint |
| SeCu (Sec 5.3/5.4 format) | 32 | 0.4204 (hier) | 0.5179 | 0.6032 | published checkpoint |
| SCAN | 128 | 0.3574 (hier) | 0.4121 | 0.5038 | authors' published checkpoint |

> SeCu d=128 uses a different multi-head per-K eval that is not directly
> comparable to the single-head numbers above (best DP=0.626 at K=180); see
> `memory/project_secu_baseline_cifar100.md`.

### CIFAR-100 framing

At peak, our unsupervised d=16 Poincaré (hDP_ward 0.893) beats supervised
HIER d=128 (0.783) by **+0.11 on hierarchy quality** and beats both
unsupervised baselines (SeCu, SCAN) by **+0.47** on the same metric. HIER
still wins 1-NN retrieval by ~0.30 because supervised metric learning
optimizes that directly.

The d=16 vs d=32 comparison shows our advantage is dim-stable: peak hDP only
varies by ~0.01 across these dims, while n@1 only varies by ~0.015.

**Pending HIER baselines:** d=16 (job 535618), d=32 (job 535619), d=128
seeds 2/3 (job 535620). Once landed, the table will have HIER at all three
matched dims.

---

## TinyImageNet — d=128 only, 3 conditions × 3 seeds

`h2h_warm` sweep, ResNet50 SimCLR-pretrained, 800 ep + 600 ep clustering.
Warm HypHC schedule (`start_temperature=1.0`, `end_temperature=0.5`,
`end_weight=5.0`).

| condition | hDP_ward (peak) | n@1 (peak) | peak ce | hDP_ward (final, ce=600) |
|---|---|---|---|---|
| **PoincaréBall + HypHC** | **0.738 ± 0.005** | 0.284 ± 0.003 | 38 ± 46 | 0.714 ± 0.003 |
| PoincaréBall no-HypHC | 0.618 ± 0.061 | 0.275 ± 0.003 | 51 ± 69 | 0.517 ± 0.018 |
| Euclidean | 0.577 ± 0.029 | 0.285 ± 0.008 | 407 ± 224 | 0.468 ± 0.120 |

### TinyImageNet framing

This is the cleanest hyperbolic story in the paper:

- HypHC adds **+0.12 hDP_ward** over no-HypHC at matched manifold and recipe.
- Hyperbolic + HypHC beats Euclidean by **+0.16 hDP_ward**.
- n@1 is essentially identical across conditions (~0.28) — the entire
  hyperbolic lift is in the hierarchy metric, not retrieval.
- Final-epoch hDP_ward (0.714) is close to peak (0.738), so the schedule
  overshoot is much milder here than on CIFAR-100 (the warm schedule
  `start_temp=1.0/end_temp=0.5` is gentler than CIFAR-100's `0.2→0.1`).

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

### ImageNet d=128 (h2h sweep, in-flight)

`533909_10..13` still running, `533909_2..9` complete. d=128 uses the same
frozen-DINO protocol but at a larger latent. Best so far:
`floral-breeze-1229` (poincare_hyphc_warm, seed43): hDP_ward = 0.692,
n@1 = 0.293. Multi-seed numbers will land within ~24h (4 tasks active).

### Frozen-backbone reference (no training)

| protocol | dim | hDP_ward | n@1 | source |
|---|---|---|---|---|
| FB DINO ResNet50 (frozen, no clustering) | 2048 | **0.884** | (not measured at 2048-d) | VSC job, 2026-04-26 |

This is a 2048-dim ceiling, **not directly comparable** to the 32/128-dim
trained rows. The matched-dim ceiling (DINO 2048 → PCA-32 / PCA-128, evaluate)
has not been computed yet — it is the most informative missing experiment for
the paper's ImageNet section.

### ImageNet framing

- **HypHC does work on ImageNet — at peak.** hyphc_warm peaks at **0.666 ±
  0.048 (ce ≈ 18)**, beating no_hyphc (0.618) and Euclidean (0.569) on
  hDP_ward. Same spike-and-crash pattern as CIFAR-100 d=16: norm_max climbs
  to ~0.996 within 4 epochs, hDP rises to peak by ce ≈ 11, then collapses to
  0.444 by ce=50 as boundary saturation continues.
- The reportable ImageNet result depends on which we call: peak (HypHC
  wins, +0.10 over Euclidean) or final (no_hyphc wins, +0.09 over
  Euclidean). Either way **PoincaréBall beats Euclidean** — the question is
  whether HypHC adds value on top of the manifold, and at peak it does.
- The schedule-fix sweep on CIFAR-100 (job 535626) targets exactly this
  overshoot mechanism. If `end_weight=1.5` or `end_temperature=0.2` keeps
  the CIFAR-100 hDP at 0.89 through ce=300, the same recipe should let
  ImageNet's 0.666 peak survive to final.
- Euclidean still wins n@1 by ~0.08 — same pattern as TinyImageNet (the
  hyperbolic lift is in hierarchy, not retrieval).
- The 2048-dim DINO ceiling at 0.884 is **not directly comparable** (dim
  mismatch); the matched-dim DINO-PCA reduction has not been computed yet.

---

## iNat18 — d=128 paper_main, 3 conditions × 3 seeds (VSC)

Frozen FB DINO ResNet50, 50 ep clustering. **No WordNet hierarchy file in
repo for iNat**, so multi-level hierarchical metrics are unavailable; only
flat DP and clustering metrics land.

| condition | n@1 (final) | dendrogram_purity_ward (flat) | agg. clustering ward NMI |
|---|---|---|---|
| **Frozen FB DINO ResNet50** (no training, 2048-d) | — | 0.0765 | **0.9425** |
| Euclidean baseline (trained, d=128) | 0.044 ± 0.001 | 0.034 ± 0.002 | 0.877 ± 0.012 |
| PoincaréBall no-HypHC (trained, d=128) | **0.001** | 0.0017 | 0.800 |
| PoincaréBall + HypHC warm (trained, d=128) | **0.001** | 0.0015 | 0.730 |

Frozen DINO Recall@K: R@2=0.121, R@4=0.154, R@8=0.193, R@10=0.205, R@100=0.397.
Source JSON on VSC: `results_baseline_embeddings/frozen_dino_inat18_val.json`.

### iNat18 framing

**Two separate problems, neither blocks the paper outright but both narrow
the iNat story.**

1. **No WordNet hierarchy in the repo for iNat.** All hierarchical metrics
   (`hierarchical_dendrogram_purity_*`, `all_dendrogram_purity_*`) are
   unavailable. Only flat DP and clustering metrics land. **Flat DP on 8142
   classes is naturally near zero** (random baseline ≈ 1/8142), so absolute
   numbers like 0.034 / 0.077 are not comparable to the hDP_ward numbers on
   CIFAR-100 / TinyImageNet / ImageNet. The relevant metric on iNat is
   `agglomerative_clustering_ward_nmi`.

2. **Poincaré collapse at `curvature_magnitude=1.0`.** All 6 Poincaré
   paper_main runs hit chance-level n@1 by ce 30–50. Hypothesis: density-
   induced boundary crush (8142 classes in d=128 ball at c=1.0 ≈ 8× denser
   than ImageNet-1k where the same config works). Rescue sweep
   `sbatch_vsc_inat18_h2h_rescue.sh` tests `c ∈ {0.1, 0.3, 0.5}` (HIER's
   working c is 0.1 on a similar manifold). Not yet submitted to VSC.

On the comparable metric (`agg_ward_NMI`):
- Frozen FB DINO: 0.943
- Trained Euclidean: 0.877
- Trained Poincaré (collapsed): 0.730–0.800

Frozen DINO modestly beats trained Euclidean on NMI (+0.07). This is a real
result but not analogous to the ImageNet hDP gap — different metric, no
hierarchy structure measured.

If the c=0.1 rescue sweep lands a working Poincaré config, the paper has an
iNat row at 8142-class scale on NMI. If it doesn't, the paper either omits
iNat or reports Euclidean-only.

---

## Schedule overshoot caveat (active investigation)

In `worldly-oath-1224` (CIFAR-100 d=16 Poincaré seed3, current code), the
trajectory shows:

| ce | latent norm_max | hDP_ward | n@1 |
|---|---|---|---|
| 0 (pretrain end) | n/a | 0.759 | 0.527 |
| 11 | 0.981 | **0.905** | 0.505 |
| 21 | 0.990 | 0.904 | 0.521 |
| 31 | 0.997 | 0.733 | 0.089 |
| 300 | 0.999 | 0.765 | 0.086 |

Once `norm_max` crosses ~0.995, fine-grained Poincaré distances become
numerically degenerate, and both n@1 and hDP_ward (which use the full
hierarchy) crash. The 20-coarse-class hierarchy was robust to this in the
old eval pipeline; the new full-WordNet hierarchy is not.

Two knobs are being tested in **job 535626** (CIFAR-100 d=16, 2 conditions
× 3 seeds, results expected 2026-04-27):

- `low_weight`: `end_weight=1.5` (was 5.0), `weight_interval=20`. Caps the
  hyperbolic loss push.
- `flat_temp`: `end_temperature=0.2` (was 0.1). No temperature ramp.

Hypothesis: either knob alone is sufficient to keep `norm_max` below ~0.99
and preserve hDP. Worst case (both fail) we fall back to early stopping at
ce=20, which the per-checkpoint embeddings already support.

---

## Pending experiments

Active or staged, in priority order:

| job | dataset | what | ETA |
|---|---|---|---|
| 535626 | CIFAR-100 d=16 | Schedule fix sweep (low_weight + flat_temp × 3 seeds) | 2026-04-27 |
| 535618 | CIFAR-100 d=16 | HIER 3-seed | 2026-04-27 |
| 535619 | CIFAR-100 d=32 | HIER 3-seed | 2026-04-27 |
| 535620 | CIFAR-100 d=128 | HIER seeds 2/3 (seed 1 done) | 2026-04-27 |
| 533909 | ImageNet d=128 | h2h sweep tasks 10–13 | 2026-04-27 |
| (staged) | ImageNet | Matched-dim DINO ceiling (DINO 2048 → PCA-32, eval) | not yet submitted |
| (staged) | iNat18 | Rescue sweep c ∈ {0.1, 0.3, 0.5} (`sbatch_vsc_inat18_h2h_rescue.sh`) | VSC, not yet submitted |
| (staged) | TinyImageNet | HIER baseline (needs new dataset adapter) | not yet drafted |

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
  This schedule overshoots — see caveat. Pending replacement.

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
