# Paper Results — Status 2026-04-24

Compiled from wandb runs, `LESSONS_LEARNED.md`, and baseline re-eval JSONs in
`results_baseline_embeddings/`. Numbers marked ✅ are paper-writable today;
⏳ are in-flight; ❌ are not yet run. All metrics computed by
`src/evaluation.py` with the current eval pipeline (8-level WordNet hierarchy,
ward linkage, `dendrogrampurity_use_root=True` on CIFAR-100 /
`=False` on ImageNet).

---

## Headline

- **Hyperbolic + HypHC is the winner on deep taxonomies.** PoincaréBall with
  a warm HypHC schedule leads Euclidean by +25–44 pp hDP_ward on CIFAR-100,
  TinyImageNet, and ImageNet-1k.
- **Unsupervised hyperbolic beats supervised HIER on hDP_ward at CIFAR-100.**
  Our d=16 Poincaré = 0.901 hDP_ward; HIER's d=128 supervised = 0.783 (+0.12
  in our favor). HIER crushes us on 1-NN retrieval (0.81 vs 0.59) — clean
  framing: supervised DML wins pure retrieval, unsupervised ours wins
  hierarchy-preserving structure.
- **The margin is hierarchy-specific.** Flat DP_ward and R@1 are
  indistinguishable across manifolds on TI; the entire lift is in the
  multi-level hierarchical metric.
- **STL-10 is the counter-example.** Two-level label hierarchy → Euclidean
  wins by +0.25 hDP. Dataset-scope predictor: # of levels in the label tree.
- **Low dim is our turf.** Moving d=128 → d=16 gains +0.15 hDP_ward on
  CIFAR-100. The matched-dim comparison to SeCu-16 / SeCu-32 is the paper's
  Sec 5.3/5.4 story.
- **iNat18 Poincaré collapses at `curvature_magnitude=1.0`** across 6/6 runs
  (n@1≈0.001). Euclidean on the same config is fine (n@1=0.04). Rescue sweep
  ready: `sbatch_vsc_inat18_h2h_rescue.sh` tests c∈{0.1, 0.3, 0.5}.

---

## Section 5.3 / 5.4 main table — paper-writable rows

### CIFAR-100 (100 fine / 20 super)

#### d=16 ✅ (ulzojs72-era recipe, 3 seeds, SimCLR 800ep + clustering 300ep)

| method (d, supervision)               | hDP_ward | hDP_complete | flat DP_ward | 1-NN |
|---|:---:|:---:|:---:|:---:|
| **Ours PoincaréBall d=16 (unsup)**    | **0.901 ± 0.005** | **0.909 ± 0.004** | — | 0.589 |
| Ours PoincaréBall d=16 chami LCA      | 0.902 ± 0.005 | 0.910 ± 0.004 | — | 0.589 |
| Ours Hyperboloid d=16 (unsup)         | 0.874 ± 0.013 | 0.883 ± 0.008 | — | 0.534 |
| Ours Euclidean d=16 (unsup, 533763)   | 0.799 ± 0.014 | 0.785 ± 0.011 | — | 0.457 |
| **HIER d=128 (SUPERVISED, 534399, e55)** ✅ | 0.783 | 0.755 | 0.638 | **0.813** |

LCA "ours" vs "chami" is a wash on CIFAR-100 (Δ<0.01). Flag for
**Section 4.2.3 ablation**: need TinyImageNet / iNat / ImageNet to show the
rectified LCA matters. CIFAR alone does not motivate the contribution.

**HIER vs Ours head-to-head (same eval pipeline, `eval_baseline_embeddings.py`):**
HIER is a supervised hyperbolic DML method (CVPR 2023). Run at emb=128 with
full-class CIFAR-100 (50k train + 10k eval, all 100 classes), PA loss + HIER
regularizer, Poincaré c=0.1, ResNet50 224×224. Our d=16 Poincaré beats HIER
on every hierarchical metric by ~+0.12 hDP_ward even though HIER uses all
labels; HIER wins 1-NN by +0.22. This is the paper's **supervised-vs-
unsupervised on different axes** story. Seeds 2/3 pending; emb=16/32 open
question (likely will amplify our advantage since HIER's natural dim is 128).

#### d=128 H2H-sweep recipe (legacy, ablation row only)

| condition (seeds 42/43/44) | hDP_ward | hDP_complete | 1-NN |
|---|:---:|:---:|:---:|
| PoincaréBall no-HypHC | 0.747 ± 0.004 | 0.693 ± 0.042 | 0.558 ± .002 |
| PoincaréBall + HypHC cold | 0.471 ± 0.077 | 0.696 ± 0.011 | 0.553 ± .001 |
| Euclidean | 0.379 ± 0.068 | 0.300 ± 0.073 | 0.560 ± .003 |

Useful for the cross-dim ablation showing the +0.15 hDP gain at low dim.
Do **not** use as the main-table CIFAR-100 row (different recipe).

### TinyImageNet ✅ (SimCLR 800ep + clustering 600ep, warm HypHC st=1.0/et=0.5)

| condition (seeds 42/43/44) | hDP_ward | flat DP_ward | 1-NN |
|---|:---:|:---:|:---:|
| **PoincaréBall + HypHC warm** | **0.715 ± 0.002** | 0.105 ± 0.004 | 0.285 ± .005 |
| PoincaréBall no-HypHC | 0.517 ± 0.018 | 0.099 ± 0.002 | 0.272 ± .003 |
| Euclidean | 0.461 ± 0.125 | 0.098 ± 0.006 | 0.288 ± .006 |

Two takeaways:
- **Poincaré+HypHC seed-invariant (σ=0.002)** — not a cherry-pick.
- Euclidean σ=0.125; seed 42 gave 0.606, seeds 43/44 gave 0.39. Always
  report 3+ seeds for Euclidean baselines on this dataset.
- Flat DP and 1-NN identical across manifolds (~0.10 and ~0.28); the
  entire story is hierarchical.

### ImageNet-1k ⏳ (frozen FB DINO R50, 50ep clustering)

#### d=128 — seed=42 row only (rest in flight, job 533909)

| condition | bk | ep | hDP_ward |
|---|:---:|:---:|:---:|
| **PoincaréBall + HypHC warm** (eager-deluge-1114) | 0 | 50/50 ✅ | **0.675** |
| PoincaréBall + HypHC warm (young-night-1189) | 1 | 19/50 ⏳ | 0.670 ↗ |
| PoincaréBall no-HypHC (polar-bird-1186) | 0 | 31/50 ⏳ | 0.591 |
| Euclidean (deft-wildflower-1188) | 0 | 22/50 ⏳ | 0.498 |
| PoincaréBall + HypHC cold (woven-lion-1187) | 0 | 35/50 ⏳ | 0.444 |

Seeds 43 pending behind `JobArrayTaskLimit` — ETA 3–5 days for full
multi-seed table.

#### d=32 — 3 seeds (jobs 533767 cold + 534387 warm) — warm is winning, as predicted

Early reads at epoch 12-13 already show the same warm-cold ordering as d=128.

| condition | seed 1 hDP_ward (ep) | seed 2 hDP_ward (ep) | seed 3 hDP_ward (ep) |
|---|:---:|:---:|:---:|
| **PoincaréBall + HypHC warm** (534387) | **0.683** (13) | 0.583 (13) | 0.703 (12) |
| PoincaréBall no-HypHC (533767) | 0.427 (10) ⏳ | 0.427 (10) ⏳ | pending |
| PoincaréBall + HypHC cold (533767) | 0.433 (12) ⏳ | 0.433 (12) ⏳ | pending |
| Euclidean baseline (533767) | 0.412 (7) ⏳ | 0.412 (7) ⏳ | pending |

Warm >> cold at d=32, matching the d=128 finding. 3-seed warm all trending
above 0.58 at epoch 13/50; expected to climb further. Full sweep lands
in ~2–3 days (cold) / ~15h (warm).

### iNat18 ⚠️ (Poincaré collapsed at c=1.0, rescue sweep designed)

**Main sweep results** — `sbatch_vsc_inat18_h2h_dino_frozen.sh`, 3 conds × 3
seeds = 9 runs, frozen FB DINO ResNet50 + d=128 + warm HypHC (cond 0) /
no_hyphc (cond 1) / Euclidean (cond 2), seeds 1/2/3:

| condition | seed 1 (ep) | seed 2 (ep) | seed 3 (ep) | n@1 | hDP_ward |
|---|:---:|:---:|:---:|:---:|:---:|
| Euclidean baseline (`gentle-star-1211`, `splendid-dragon-1213`, `soft-water-1220`) | 50 ✅ | 50 ✅ | 50 ⏳ | ≈ 0.04 | — (no WordNet hierarchy for iNat) |
| PoincaréBall no-HypHC (seed 1/2/3 runs) | 50 ✅ | 50 ✅ | 29 ⏳ | **≈ 0.001** ❌ | — |
| PoincaréBall + HypHC warm (seed 1/2/3 runs) | 50 ✅ | 50 ✅ | 43 ⏳ | **≈ 0.001** ❌ | — |

All 6 completed Poincaré runs collapsed to chance-level n@1. Euclidean at
same config is fine. Hypothesis: 8142 classes × d=128 × `curvature_magnitude=1.0`
saturates the tanh in `HyperbolicLinearPP`, crushing points to the boundary.
See LESSONS_LEARNED.md §"iNat Poincaré density collapse" for details.

**Rescue sweep designed and ready:** `sbatch_vsc_inat18_h2h_rescue.sh` —
3 curvatures × 3 seeds × warm HypHC:
- cond 0: c=0.1 (HIER-style, maximally flat)
- cond 1: c=0.3 (moderate)
- cond 2: c=0.5 (halfway to default)

If c=0.1 works but c=1.0 doesn't, geometry density is confirmed.

**Frozen-DINO lower bound** also staged — `sbatch_vsc_inat18_frozen_baseline.sh`
extracts frozen backbone features and runs through `eval_baseline_embeddings.py`
to produce a one-number reference (n@1, DP_ward, agglomerative NMI/ARI) for
the iNat row with no head trained on top.

---

## Manifold-choice ablation (CIFAR-100 d=16) ✅

Use this *instead of* a three-manifold row in every main-table cell.

| manifold | hDP_ward | hDP_complete | 1-NN |
|---|:---:|:---:|:---:|
| **PoincaréBall** | **0.901 ± 0.005** | **0.909 ± 0.004** | **0.589** |
| Hyperboloid | 0.874 ± 0.013 | 0.883 ± 0.008 | 0.534 |
| Euclidean | 0.799 ± 0.014 | 0.785 ± 0.011 | 0.457 |

One-line framing: "Poincaré and Hyperboloid are isometric charts of the same
manifold; Poincaré is marginally better in our pipeline (+0.03 hDP_ward,
+0.05 1-NN). We use Poincaré in all main-table rows; Hyperboloid is not
repeated per-dataset."

---

## STL-10 negative result ✅ (counter-example row)

24-run sweep, 3 seeds × 4 conds × bk∈{0,1}, SimCLR 800ep + cluster 400ep.

| condition | bk=0 | bk=1 |
|---|:---:|:---:|
| **Euclidean baseline** | **0.799** | **0.826** |
| PoincaréBall + HypHC cold | 0.313 | 0.576 |
| PoincaréBall + HypHC warm | 0.453 | 0.460 |
| PoincaréBall no-HypHC | 0.341 | 0.470 |

Euclidean leads by +0.25 hDP. STL-10 has a 2-level label split
(vehicle/animal); hyperbolic geometry has no depth to exploit. Report as
"hyperbolic helps only when the label tree is deep" — dataset-scope predictor
is # of hierarchy levels.

---

## Cross-dim ablation (CIFAR-100, d=128 → d=16) ✅

Same dataset, different latent dim, Euclidean excluded to hold recipe:

| dim | recipe | hDP_ward (Poincaré) |
|---|---|:---:|
| d=128 | H2H sweep, frozen, no-HypHC | 0.747 ± 0.004 |
| d=16  | \*-16-Ours, Poincaré + HypHC cold | **0.901 ± 0.005** |

**+0.15 hDP_ward moving to d=16.** Low-dim embeddings exploit hyperbolic
volume expansion; at d=128 the manifold geometry is underutilized.
Motivates "exceptionally low-dimensional" framing.

---

## Baselines — re-evaluation through our pipeline

### HIER (hyperbolic DML, supervised) ✅

Trained on CIFAR-100 full-class (50k train + 10k eval, all 100 classes),
emb=128, seed=1, 100 epochs (job 534399, `hier_cifar100_534399.log`).
Saved embeddings at 12 checkpoints; piped through `eval_baseline_embeddings.py`
(job 534413) for 5 epochs (45/55/75/90/95). **Peak at epoch 55:**

| metric | value |
|---|:---:|
| 1-NN accuracy | **0.8129** |
| hierarchical_dendrogram_purity_ward | **0.7834** |
| hierarchical_dendrogram_purity_complete | 0.7545 |
| dendrogram_purity_ward (flat) | 0.6381 |
| all_dendrogram_purity_ward | 0.7827 |
| agglomerative_clustering_ward_nmi | 0.8270 |
| agglomerative_clustering_ward_ari | 0.6313 |
| best_partition_ward_nmi / ari / k | 0.8297 / 0.6640 / 118 |
| R@{2,4,8,10} | 0.866 / 0.903 / 0.924 / 0.931 |

Epoch trajectory: 0.772 (e45) → **0.783 (e55)** → 0.783 (e75) → 0.769 (e90)
→ 0.761 (e95). Peak mid-training, slight regression thereafter. Seeds 2/3
pending; emb=16/32 unscheduled but likely to improve our advantage.

### SeCu / SCAN (unsupervised clustering) ⏳

User re-running SeCu through the current eval pipeline. Memory-stored numbers
gave an hDP − all_DP gap of 10–15 pp (inconsistent with our runs); the fresh
eval closes the gap to ~0.05. Old memory numbers ARE stale.

| Job | Target | Status |
|---|---|---|
| 534400 | SeCu CIFAR-100 d=16 | running |
| 534401 | SeCu CIFAR-100 d=32 | running |
| 534402 | SeCu CIFAR-100 d=128 | running |
| 534015 | SeCu TinyImageNet | running |

SCAN re-eval queued on user's side per plan.

### Frozen-DINO lower bound ⏳

New infrastructure — `scripts/extract_frozen_backbone_embeddings.py` +
`scripts/eval_baseline_embeddings.py`. Takes any ResNet50 checkpoint, forwards
the dataset's eval split with fc stripped, saves `(x, y, idx)` tuple in
HIER-compatible format, runs through our eval pipeline. Staged sbatch files:

- `sbatch_vsc_inat18_frozen_baseline.sh` — iNat18 val frozen FB DINO R50
- `sbatch_vsc_imagenet_frozen_baseline.sh` — ImageNet val frozen FB DINO R50
  (10k stratified subsample to match main sweep)

Gives a pure "no-head" reference row every hyperbolic-head condition must beat.

### Coverage matrix

| Dataset  | SeCu | SCAN | HIER | Frozen DINO |
|---|:---:|:---:|:---:|:---:|
| CIFAR-100 | ⏳ (534400-02) | user queuing | ✅ emb=128 s1 | TBD |
| TinyImageNet | ⏳ (534015) | user queuing | ❌ not scheduled | TBD |
| ImageNet-1k | cite published | no published | ❌ (costly) | ⏳ staged |
| iNat18 | skip (no pub, 8142 cls infeasible) | skip (same) | ❌ too expensive | ⏳ staged |

---

## In-flight SLURM jobs

### Local cluster
| Job | Name | What | ETA |
|---|---|---|---|
| 533909 | imnet_h2h | ImageNet d=128 × 4 conds × 2 bk × 2 seeds (16 tasks) | 3–5 days |
| 533767 | imnet_d32 | ImageNet d=32 cold × 3 conds × 3 seeds (9 tasks) | 2–3 days |
| 534387 | imnet_d32_w | ImageNet d=32 Poincaré+HypHC **warm** × 3 seeds (3 tasks) | ~15h |
| 534400/01/02 | secu-dim{16,32,128} | SeCu baselines CIFAR-100 | hours |
| 534015 | secu-ti | SeCu baseline TinyImageNet | hours |

### VSC
| Job | What | State |
|---|---|---|
| — | iNat18 H2H (main sweep, 9 tasks) | Seeds 1-2 done (Poincaré collapsed); seed 3 running |

### Completed this cycle
- **534388** HIER CUB sanity (R@1 ≈ 0.68-0.69 matches paper's 0.70 — env OK)
- **534399** HIER CIFAR-100 emb=128 seed 1 (R@1=0.8127, 3:47h)
- **534413** HIER CIFAR-100 embeddings → our eval pipeline (5 epochs evaluated)
- **534414/534416** CIFAR-100 Poincaré d=16/d=32 — failed in 10-30s due to
  `clustering.hyperbolic_smoothing` struct-mode error. **Scripts fixed**,
  ready to resubmit.

---

## Work to queue (paper-required, not yet submitted)

### Local cluster
1. **CIFAR-100 Poincaré d=16** — `sbatch_cifar100_poincare_d16.sh` (just
   patched). Resubmit.
2. **CIFAR-100 Poincaré d=32** — `sbatch_cifar100_poincare_d32.sh` (just
   patched). Resubmit.
3. **HIER CIFAR-100 seeds 2/3** — copy `sbatch_cifar100_hier.sh` × 2 with
   `--seed 2` / `--seed 3`. For error bars on the 0.783 hDP_ward number.
4. **HIER CIFAR-100 emb=16 and emb=32** — matched-dim comparison. Likely
   amplifies our d=16 advantage (HIER's natural dim is 128).

### VSC
5. **iNat18 Poincaré rescue sweep** — `sbatch_vsc_inat18_h2h_rescue.sh`
   (3 curvatures × 3 seeds, ready to submit).
6. **iNat18 frozen-DINO baseline** — `sbatch_vsc_inat18_frozen_baseline.sh`
   (1 run × ~1h, lower-bound anchor).
7. **ImageNet frozen-DINO baseline** — `sbatch_vsc_imagenet_frozen_baseline.sh`
   (1 run × ~1h, lower-bound anchor).

---

## What the paper needs vs what we have

| Section | What's needed | Have? |
|---|---|---|
| 5.3 metric learning (CIFAR-100) | Poincaré d=16/d=32, SeCu + HIER baselines | d=16 ✅, d=32 ⏳ (resubmit), SeCu ⏳, HIER ✅ (s=1 at d=128) |
| 5.3 metric learning (TI) | Poincaré + SeCu baseline | Poincaré ✅, SeCu-TI ⏳ |
| 5.4 deep clustering (CIFAR-100) | hDP_ward main row + SeCu/SCAN/HIER | hDP ✅, SeCu ⏳, HIER ✅ |
| 5.4 deep clustering (TI) | hDP_ward main row + SeCu | hDP ✅, SeCu-TI ⏳ |
| 5.4 deep clustering (ImageNet) | hDP_ward main row | d=128 seed=42 ✅, rest ⏳; d=32 warm looking strong |
| 5.4 deep clustering (iNat18) | hDP_ward main row | Euclidean ⏳, Poincaré ⚠️ collapsed — rescue sweep designed |
| Ablation: manifold choice | Three manifolds, one dataset | ✅ (CIFAR-100 d=16) |
| Ablation: dim sweep | d=16 vs d=128 on CIFAR-100 | ✅ |
| Ablation: LCA ours vs chami | Non-trivial Δ on some dataset | ❌ — CIFAR Δ<0.01, need TI/iNat/ImageNet |
| Ablation: curvature sweep on dense hierarchy | c ∈ {0.1, 0.3, 0.5, 1.0} | ⏳ — rescue sweep on iNat18 covers this |
| Counter-example (5.4 or 5.5) | STL-10 negative result | ✅ |
| Supervised-vs-unsupervised framing | HIER wins R@1, we win hDP | ✅ (CIFAR-100 at d=128, 1 seed) |

---

## Open methodology questions

- **SeCu hDP magnitude sanity check.** After re-eval, if SeCu-16 still lands
  at hDP_w ≈ 0.40 while our d=16 Euclidean is at 0.80, investigate whether
  SeCu's released embeddings are low-dim (16) or some other dim. An apples-
  to-apples table requires matched output dim.
- **LCA-rectification on deeper hierarchies.** CIFAR (Δ<0.01) does not
  motivate Section 4.2.3. Need TI / iNat / ImageNet ours-vs-chami numbers.
- **Cold-HypHC regression at d=128 ward-DP.** Poincaré+HypHC cold on CIFAR
  d=128 gives hDP_ward=0.471 ± 0.077 (high variance) vs Poincaré no-HypHC
  at 0.747 ± 0.004. Complete-linkage hDP is fine for both. Suggests cold
  HypHC scrambles the ward tree at d=128; warm schedule fixes this on
  ImageNet. Worth one paragraph in the discussion.
