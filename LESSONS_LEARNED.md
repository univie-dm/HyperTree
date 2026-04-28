# Lessons Learned — Hyperbolic Clustering

Running log of non-obvious findings from debugging the CUB-200 R@1 collapse
and the CIFAR-100 sweeps. Read this before proposing fixes — several
"obvious" root causes have already been ruled out.

## Current state (2026-04-24)

**In flight on local cluster (use `squeue -u $USER`):**
- `533641` (array 0-15%4) — **ImageNet H2H**, frozen FB DINO ResNet50 + hyperbolic
  head. 16 runs = 4 conditions × 2 bk × 2 seeds. First run `eager-deluge-1114`
  already at clustering ep 37/50 with hDP_complete=0.403, hDP_ward=0.677. Wait
  for rest.
- `533759` — **SeCu baseline for hierarchical DP** (user-run, comparison
  baseline).
- `533763` (array 0-2%3) — **CIFAR-100 Euclidean d=16 × 3 seeds**. Fair
  baseline for the `*-16-Ours` recipe table. Submitted this session; numbers
  pending. Expected ~15h.
- `533767` (array 0-8%3) — **ImageNet H2H at d=32** × 3 conditions × 3 seeds
  (poincare_hyphc_cold, poincare_no_hyphc, euclidean_baseline; seeds 1/2/3;
  bk=True fixed). Paper-framing sweep: abstract promises "exceptionally
  low-dimensional" but the running d=128 sweep (533641) doesn't deliver.
  CIFAR-100 showed +0.15 hDP_ward moving d=128 → d=16; d=32 for 1000
  classes is the expected low-dim sweet spot. Frozen FB DINO, 50 clustering
  epochs only, so ~a day of wallclock with %3 throttle. `533641` stays as
  the cross-dim ablation. See `sbatch_imagenet_h2h_d32.sh`.

**In flight on VSC:**
- `sbatch_vsc_inat18_h2h.sh` (iNat18 DINO) and `sbatch_vsc_inat18_simclr.sh`
  (SimCLR) — status unknown; previously blocked by disk-quota on
  `/home/ks09437` when Lightning saved `last.ckpt`. User is adding a symlink
  from `data/checkpoints/` to a larger filesystem. After symlink is in
  place, corrupt `last.ckpt` files need deletion before resubmit:
  `find data/checkpoints -name 'last.ckpt' -delete` and
  `find data/checkpoints -name 'epoch=*.ckpt' -delete`. Then `sbatch`
  both scripts again — eval now uses full val set (commit b093a1a).

**Top-3 paper todos (order matters):**
1. **Section 5.4 hierarchical-DP main table.** Writable *today* with the
   numbers in this file: CIFAR-100 (`*-16-Ours` recipe row + ablation
   recipe row), TinyImageNet, STL-10. Use iNat18/ImageNet `TBD` placeholders.
   Primary metric hDP_ward.
2. **LCA-rectification ablation.** Your rectified-LCA contribution (paper
   Section 4.2.3) shows Δ<0.01 vs Chami's LCA on CIFAR-100 at d=16 (both
   manifolds). To motivate the section you need `ours` vs `chami` on
   TinyImageNet / iNat18 / ImageNet. Plan an ablation sweep.
3. **Sections 5.1 (datasets) + 5.2 (impl details) + Figure 1** —
   mechanical writeup, unblock reviewers.

**Skip for now:** GTSRB (not in paper's target dataset list paper.md:349-353).
CUB-200 SimCLR transfer *would* be a good paper addition but the Euclidean
d=16 baseline is higher priority — submit CUB only after 533763 lands.



## Pathologies

### 1. Gradient absorption by the hyperbolic head
When contrastive loss is computed on hyperbolic projections and the backbone
is frozen (or receives weak signal), the hyperbolic head absorbs all the
gradient. The backbone never learns, and the head drifts toward degenerate
(boundary-saturated) configurations. **Mitigation:** compute contrastive in
*Euclidean* space on pre-hyperbolic backbone features (`clustering.contrastive_loss.euclidean=true`),
or freeze the encoder and selectively unfreeze `layer4`.

### 2. Dip-then-recover is *not* caused by bad init
Classic bad-init signature (R@1 starts high, dips, slowly recovers) suggested
that random weight magnitudes were saturating the Poincaré boundary. Turned out
the factory at `src/models/utils.py:139` has been Kaiming-scaling
(`weight *= 1/√input_dim`) since commit 02d42a4 (Feb 2026). CIFAR-100 winners
and the current CUB runs have *identical* init. Init is not the root cause.

### 3. The real cause of the dip: hyperbolic head is random at clustering start
CIFAR-100 sweep era (`git show 7bd0904:src/models/resnet.py`) trained the
hyperbolic head during pretrain via a **dual** hyperbolic + Euclidean
contrastive loss. Current CUB runs (uncommitted skip-head branch) compute
contrastive *only* on Euclidean backbone features — so the hyperbolic head
drops into clustering with random weights.

**Current fix under test:** `head_warmup_epochs=N`. During warmup,
`hyperbolic_loss_weight=0` and contrastive routes to geodesic NT-Xent on the
hyperbolic head. After warmup, contrastive switches back to Euclidean and
HypHC ramps on. See `src/models/hyphc.py:751` and job **524086**.

## Benchmarks — what numbers actually mean

### CUB-200 supervised baseline (~42%) is inflated
The torchvision supervised ResNet50 was trained on ImageNet, which contains
**59 bird classes**. Label leakage inflates CUB R@1 dramatically. Fair
comparison — random hyperbolic head on supervised features but stripped of
bird-class semantics — is closer to **~18%**. Do not compare CVPR 2021's
**58.9%** benchmark against the leaky 42%; the true ceiling is lower.

### 42% from a *random* hyperbolic head is JL, not learning
Random linear projections preserve neighborhoods (Johnson–Lindenstrauss).
A ResNet50 with supervised features + random hyperbolic head gets ~42% not
because the hyperbolic head did anything useful, but because JL preserves
nearest-neighbor structure through a random projection. The hyperbolic head
must *beat* this, not match it.

### "Working on CIFAR-100" ≠ "all metrics go up"
Top-5 CIFAR-100 runs: R@1 trajectory **flat** (0.52 → 0.53), ARI drops
(−0.06), NMI/DP flat. Bottom-5: R@1 collapses 0.52 → 0.04. So "working" means
*not collapsing*, not "improving everything." Don't wait for a dramatic
R@1 jump — absence of collapse is the win signal.

## Sweep insights

### CIFAR-100 winners vs losers: `freeze_encoder` is the only clean knob
From `scripts/diff_cifar100_winners_losers.py`: top-5 all have
`freeze_encoder=True`, bottom-5 all have `False`. Contrastive enabled and
regularizer enabled also differ, but freeze_encoder is the cleanest
univariate signal. `unfreeze_backbone_from=layer4` with `backbone_lr_factor=0.1`
on top of `freeze_encoder=True` is the recipe used by the working runs.

### CUB-200 sweep flips the CIFAR-100 recipe
Running the same diff script on `regularizer_sweep + hyphc_magnitude_sweep`
(jobs 524061, 524070), top-5 winners (R@1 ≈ 0.17–0.20) share:
- `regularizer.type = uniformity` (origin is only in losers)
- `regularizer.weight = 0.05` (0.5 is only in losers)
- `unfreeze_backbone_from = None` — **fully frozen wins, `layer4` is in losers**
- `contrastive_loss.enabled = True`, `euclidean = True`

The `unfreeze layer4` knob that helped on CIFAR-100 *hurts* on CUB. Likely
because the supervised ResNet50 features on CUB are already bird-optimized
(ImageNet label leakage); unfreezing corrupts them. Do not transplant the
CIFAR-100 backbone-unfreeze recipe to CUB.

### Start HypHC weight from zero
`clustering.hyperbolic_loss.start_weight=0` with a long
`weight_interval` (≥250 epochs) avoids the early HypHC dominance that pushes
embeddings to the boundary before the contrastive/regularizer structure
stabilizes.

### HypHC ablation (end_weight=0) also collapses — and tells us why
Ran the winning recipe with `hyperbolic_loss.end_weight=0` to test whether
HypHC itself was destroying NN structure (job 524330, `sparkling-frost-984`).
R@1 dropped 0.42 → 0.10 by ep185 — same collapse as non-ablated runs.

But this is not evidence *for* HypHC. With `unfreeze_backbone_from=None`
and `contrastive.euclidean=True` (contrastive on pre-hyperbolic **backbone**
features, which are frozen), **the contrastive loss has zero gradient** — it
touches only frozen parameters. In the ablation, the only active signal
reaching the hyperbolic head is `uniformity@0.05`, which disperses embeddings
uniformly on the sphere with no class-preserving constraint. The collapse is
the uniformity regularizer spreading a random-ish head.

So under the "winner recipe" (fully frozen + Euclidean contrastive), **HypHC
is the only class-preserving signal** reaching the hyperbolic head. That also
reframes the 0.42 → 0.20 drop: HypHC preserves *some* structure; removing it
leaves only uniformity, which is purely destructive in this config. Real
fixes need either (a) a class-preserving signal that reaches the hyperbolic
head without HypHC (e.g. geodesic NT-Xent on the head's outputs), or
(b) unfrozen features so the backbone can reshape to the contrastive signal
— contradicting the "fully frozen" winner pattern.

## Paper-scope decisions

### Baselines: SCAN, SeCu, hyperbolic metric learning — none report DP
The paper comparison is against these three families on **Dendrogram Purity**.
None of them report DP in their papers — they report NMI/ACC (SCAN, SeCu) or
R@K (hyperbolic DML like Ermolov et al. 2022, HIER). So the baseline
evaluation is **not a citation exercise — we have to run their released
code/checkpoints, extract embeddings, and compute DP ourselves** via ward
linkage + WordNet coarse labels in our `evaluation.py`. Plan for this as a
real work item.

### Dataset overlap constrains the DP story to CIFAR-100 + TinyImageNet
SCAN/SeCu report CIFAR-10/100, STL-10, ImageNet. Hyperbolic DML is CUB/Cars.
The intersection with "has a WordNet hierarchy we can build coarse labels
from" is CIFAR-100 + TinyImageNet. CUB and Cars are fine-grained with no
WordNet tree; iNat has its own taxonomy (not WordNet). For the DP paper,
default to CIFAR-100 + TI; add ImageNet-1k only if compute budget allows.

## Workflow traps

### Double-scaling
My first "fix" was to add Kaiming scaling inside
`HyperbolicLinearPoincarePP.__init__`. The factory at `src/models/utils.py:139`
was *also* scaling. Net effect: weights got `1/input_dim` instead of
`1/√input_dim`, output norm became 0.075 (too small, vs correct ~0.87).
**Before adding init scaling anywhere, grep for existing scaling in the
factory or the layer — one or the other, not both.**

### Verify the fix is not a no-op *before* launching the sweep
The init fix changed nothing functionally. I only caught this by
`git show <sweep-era-commit>:<file>` and comparing against the current
working tree. When a fix targets a regression, always diff against the
last-known-good commit to confirm the thing you're "fixing" is actually
different.

### Bare digits in `wandb.tags` crash on Pydantic validation
`wandb.tags='[tiny_imagenet,dino,224,hyphc]'` passed through Hydra: the `224`
is parsed as `int`, not `str`. Wandb's Pydantic v2 settings rejects it with
`Input should be a valid string` on `run_tags.2`. Fix: rename any numeric
tag to include a letter prefix (e.g. `res224`). Don't bother quoting the
string in the bash source — Hydra re-parses.

### 224×224 + ResNet50 + hyperbolic head OOMs at batch=256
Frozen ImageNet ResNet50 + hyperbolic projection + HypHC soft-LCA pairwise
distance matrix (in float64) at 224² and batch=256 hits OOM on 80GB H100
*during clustering*, not pretraining. batch=64 fits. The soft-LCA pairwise
distance cost is `O(B² × D)` in float64; scales badly with batch.

### `unfreeze_backbone_from` requires `freeze_encoder=True`
The two hparams compose: `freeze_encoder=True` freezes everything, then
`unfreeze_backbone_from=layer4` selectively re-enables gradients for
`layer4` onward. Setting only `unfreeze_backbone_from` with
`freeze_encoder=False` does nothing (everything is already trainable).

### "HDP regressed on newer code" is almost always a config mismatch
The `*-16-Ours` April-5 Poincaré runs (`ulzojs72` et al.) at `c0b4a5e` sit
at hDP_complete ≈ 0.91 / hDP_ward ≈ 0.90. Recent CIFAR-100 clustering runs
on `imagenet`-branch HEAD (`b093a1a8` and later) look ~0.10–0.25 lower and
it's tempting to blame the 25-commit gap. It isn't the code:

1. `c0b4a5e` is **dev's tip**, not main. `main` is at `0908b98` from
   2025-11-29 (~5 months behind dev), so "should be the state of main"
   misreads the graph — ulzojs72 is actually the most forward code of the
   three.
2. The big `src/evaluation.py` refactor between `c0b4a5e` and HEAD moved
   WordNet hierarchy construction out of `hierarchical_dendrogram_purity`
   and into `AbstractModel.on_test_epoch_end` via `build_hierarchy_clusters`.
   The inner DP algorithm (linkage → merge traversal → `use_root` handling
   → normalization) at `src/evaluation.py:1033-1081` is identical to dev's.
   With `use_wordnet_labels=True` + `dendrogrampurity_use_root=True`,
   CIFAR-100 gets the same multi-level WordNet hierarchy + root cluster in
   both versions. The metric itself has not drifted.
3. No one has re-run Poincaré d=16 800ep since April 5. The post-refactor
   CIFAR-100 ld=16 runs (April 22, `b093a1a8`) are **Euclidean**, and the
   April 21 Poincaré runs are **d=128**. Both are expected to be lower on
   hDP from the H2H-sweep numbers above (Euclidean d=16 ≈ 0.78;
   Poincaré d=128 ≈ 0.65–0.72).

**Rule:** before blaming a hDP gap on code drift, `git log` the runs'
commits and diff the config. A "regression" across a manifold swap or
latent-dim change is not a regression. No Poincaré d=16 800ep rerun on
HEAD exists — job `533763` is Euclidean d=16 (the fair baseline for
the `*-16-Ours` table), not a code-regression test. If you actually want
to rule out code drift on the Poincaré recipe, rerun it on HEAD with the
ulzojs72 config (contrastive_weight=0.1, start_weight=1→end_weight=5,
temp 0.2→0.1, alpha=0.2, smoothing=50).

## Related work that informed interventions

- **CMHHC** (multi-view contrastive hyperbolic) — motivates per-view
  hyperbolic contrastive instead of single-view.
- **Ge et al. HCL** — hyperbolic contrastive learning with explicit
  boundary regularization.
- **Yang et al. (2024) outer-shell isotropy** — angular uniformity
  regularizer on L2-normalized tangent vectors (Wang & Isola style).
  Implemented as `regularizer.type=uniformity` in `src/models/hyphc.py`.

## HypHC tuning

### HypHC start_temperature was ~5× too cold on TinyImageNet
The zany-snow recipe used `start_temperature=0.2, end_temperature=0.1` (copied
forward from CIFAR-100). On TI this cratered hierarchical DP. Sweep 531497
tested three warmer variants at the same otherwise-winning recipe:

| variant | st/et | ew | hDP_ward |
|---|---|---|---|
| baseline  | 0.2/0.1 | 5 | 0.404 |
| warmer    | 0.5/0.3 | 5 | 0.717 |
| **much_warmer** | **1.0/0.5** | **5** | **0.731** |
| weaker_warmer | 0.5/0.3 | 1 | 0.724 |

**Warmer HypHC lifts hDP 1.8× and takes TI into CIFAR territory.** Flat DP only
moves from 0.096 → 0.104 (noise) because fine-class pairs still rarely share a
pure subtree at R@1=0.29 — the win is hierarchy-specific.

### start_temperature is the lever; end_weight is not
`weaker_warmer` (ew=1) nearly matched `much_warmer` (ew=5). All three warmer
variants tied to within 0.02 hDP regardless of end_weight. The dominant
sensitivity is `start_temperature` — it sets how diffuse the soft-LCA is during
the first epochs. Too cold = HypHC over-commits to noisy fine-grained pairs
before the backbone knn structure is usable.

### Re-validate HypHC temperature per-dataset, not transfer from CIFAR
The CIFAR-100 winners used `start_temperature=0.2`. Transferring this to TI
broke hDP until the hctemp sweep fixed it. Expect the same per-dataset tuning
requirement on CUB, iNat, ImageNet — do not assume a CIFAR hyperparameter is
portable.

## Dataset / backbone findings

### Flat DP is misleading on fine-grained datasets — use hierarchical DP
TinyImageNet flat `dendrogram_purity_ward` ≈ 0.10 vs CIFAR-100's 0.31, a 3.2×
gap that looked like pipeline failure. But `hierarchical_dendrogram_purity_ward`
(same DP, computed against WordNet coarse labels) tells a different story:
TI ≈ 0.49–0.53 vs CIFAR ≈ 0.62–0.77, only ~1.3× worse. Flat DP cratered because
with 200 fine classes and R@1 ~0.27, no pair of distinct fine classes shares a
pure subtree. **Always report `hierarchical_dendrogram_purity_*` and
`all_dendrogram_purity_*` alongside flat DP on multi-class datasets** — flat DP
on ≥200-class problems is a near-useless headline number and penalizes
hierarchy-preserving embeddings for not also solving fine-grained retrieval.

### DINO and SimCLR have opposite TinyImageNet profiles
At 64×64, same ResNet50 backbone:
- **SimCLR**: R@1 ≈ 0.27, flat DP_ward ≈ 0.10, **hDP_ward ≈ 0.49**
- **DINO**:   R@1 ≈ 0.14, flat DP_ward ≈ 0.04, **hDP_ward ≈ 0.85**

DINO is 1.7× better at hierarchy but half as good at fine discrimination. The
multi-crop teacher-student objective pushes embeddings toward coarse/global
structure; SimCLR's instance-discrimination pushes toward fine separability.
For a hierarchy-focused paper, DINO backbones are the natural pick — flip the
headline metric to hDP and DINO wins cleanly.

### DINO at 64×64 is multi-crop-starved
Standard DINO uses global=224 / local=96 (ratio ~2.3). At 64/32 the ratio is
2.0 but the *absolute* sizes are 3.5× smaller, and the 32×32 locals carry
almost no discriminative signal for the student. This is the likely cause of
DINO's TI fine-grained collapse to R@1 ≈ 0.14. **Upsampling TI to 224 with
local=96** (job 531502) is the standard fix.

### Skipping pretraining to stack supervised ImageNet features
Setting `pretraining.epochs=0` + `pretraining.load_supervised_weights=True`
produces a clustering-only run on frozen torchvision ResNet50. Useful as a
ceiling probe to isolate "backbone quality" vs "hyperbolic head" failure modes.
Hash differs from SimCLR/DINO runs because `pretrain_epochs` and
`load_supervised_weights` are in `get_model_path`. Not a fully-unsupervised
setup, so not a paper-final result — diagnostic only.

## Evaluation

### Recall@K (standard DML metric)
`src/evaluation.py:recall_at_k` — per-query 1 if *any* of the top-k neighbors
(self excluded) matches the query label, averaged. Gated by
`evaluation_metrics.recall_at_k`. Reports `test/recall_at_{2,4,8,10,100}`
alongside the existing `{k}_nearest_neighbors_accuracy` (which is
*accuracy*@k, not recall, and identical at k=1). Use R@k to compare against
DML benchmark tables (e.g. CVPR 2021's CUB R@1 = 58.9%).

## H2H manifold comparison (CIFAR-100, TinyImageNet, STL-10)

### CIFAR-100: two recipes, very different numbers — use the d=16 recipe for the paper
CIFAR-100 has two distinct sweep families on wandb. They disagree by +0.16
hDP_ward, so the recipe matters as much as the manifold choice. Don't quote
the H2H-sweep numbers as the paper main result.

**Paper-main recipe (`*-16-Ours` / `*-16-Chami` runs, d=16):**
SimCLR 800ep + clustering 300ep, latent_dim=**16**,
contrastive active during clustering (weight=0.1, enabled=True),
HypHC ramped `start_weight=1→end_weight=5` (not 0→5),
temperature `0.2→0.1` (cold), seeds {1, 2, 3}.

| manifold \\ LCA      | ours  s1/s2/s3            | chami s1/s2/s3            | mean hDP_ward | mean hDP_complete | mean 1-NN |
|---------------------|---------------------------|---------------------------|:-------------:|:-----------------:|:---------:|
| **PoincareBall**    | 0.906 / 0.902 / 0.896     | 0.906 / 0.902 / 0.897     | **0.901 ± 0.005** | **0.909 ± 0.004** | **0.589** |
| Hyperboloid         | 0.848 / 0.881 / 0.881     | 0.870 / 0.881 / 0.880     | 0.874 ± 0.013     | 0.883 ± 0.008     | 0.534     |

Key findings:
1. **Poincaré beats Hyperboloid** by +0.03 hDP_ward and +0.05 1-NN at d=16.
   Magnitude small but consistent across seeds and LCA variants.
2. **LCA "ours" vs "chami" is a wash on CIFAR-100** (Δ < 0.01 on any metric,
   in both manifolds). The paper's rectified LCA contribution does not
   manifest on this dataset. Needs TinyImageNet / iNat18 / ImageNet numbers
   to show it matters — flag as open question for experiments section.
3. **Low dim + contrastive-during-clustering + HypHC-from-ep0** gives ~+0.15
   hDP_ward over the H2H-sweep recipe below. Each knob contributes something,
   but we have not isolated which one drives the gap.
4. **No matching Euclidean d=16 baseline exists** — to claim
   "hyperbolic beats Euclidean at low dim" we need Euclidean d=16 with the
   exact same recipe. **Submitted as job 533763** (3 seeds × 1 Euclidean
   condition, `sbatch_cifar100_euclidean_d16.sh`). Numbers pending; will
   be filled in here once runs finish. Expected wallclock ~15h.

**H2H-sweep recipe (legacy, `euclidean_baseline`/`poincare_hyphc`/
`poincare_no_hyphc` tags, d=128):**
SimCLR 800ep + clustering 600ep, latent_dim=128, freeze_encoder=True,
contrastive off during clustering, HypHC 0→5, seeds {42,43,44}.

| condition              | hDP_ward s42/s43/s44  | mean ± sd         | hDP_complete mean |
|------------------------|-----------------------|:-----------------:|:-----------------:|
| PoincareBall (no HypHC)| 0.751 / 0.742 / 0.750 | 0.747 ± 0.004     | 0.693             |
| PoincareBall + HypHC   | 0.560 / 0.418 / 0.436 | 0.471 ± 0.077     | 0.696             |
| Euclidean (no HypHC)   | 0.424 / 0.413 / 0.301 | 0.379 ± 0.068     | 0.300             |

Useful as an ablation point showing (a) manifold alone clears Euclidean by
+0.37 even without HypHC, and (b) HypHC at d=128 with 0→5 ramp and
freeze_encoder *hurts* ward-DP. The 0→5 warm-start HypHC ramp is what
differs from the paper-main recipe; don't use this recipe for the main table.

Flat 1-NN is indistinguishable across Poincaré conditions (0.553–0.566) and
close to Euclidean (0.556–0.563) in the H2H sweep — hyperbolic lift is
entirely in hierarchical metrics, matching the TinyImageNet pattern.

### TinyImageNet: PoincareBall + HypHC beats Euclidean by +0.25 hDP (3 seeds)
`h2h_warm` recipe (SimCLR 800ep + clustering 600ep, warm HypHC st=1.0/et=0.5),
3 seeds × 3 conditions:

| condition | s42 | s43 | s44 | mean ± sd |
|---|---|---|---|---|
| **PoincareBall + HypHC(ew=5)** | 0.716 | 0.716 | 0.712 | **0.715 ± 0.002** |
| PoincareBall (no HypHC)        | 0.527 | 0.496 | 0.527 | 0.517 ± 0.018 |
| Euclidean (no HypHC)           | 0.606 | 0.392 | 0.386 | 0.461 ± 0.125 |

Two surprises:
1. **Poincaré+HypHC is almost seed-invariant (σ=0.002)** — this is a robust
   story, not a cherry-pick.
2. **Euclidean has huge seed variance (σ=0.125)** — seed 42 gave 0.606 which
   fooled the first-read, seeds 43/44 are 0.39. This means any paper table
   needs 3+ seeds for Euclidean baselines or we risk reporting an outlier as
   the mean.

HypHC adds +0.20 hDP over the no-HypHC Poincaré run; the hyperbolic manifold
on its own (no HypHC) is comparable to Euclidean. **HypHC is carrying the
hyperbolic advantage.** Flat DP and R@1 are indistinguishable across
conditions (~0.10 / ~0.28); the entire story is hierarchical.

### STL-10: Euclidean wins decisively (negative result for hyperbolic)
24-run VSC sweep (3 seeds × 4 conditions × backbone_knn {0,1}), SimCLR 800ep +
clustering 400ep. Mean hDP_ward:

| condition | bk=0 | bk=1 |
|---|---|---|
| **euclidean_baseline** | **0.799** | **0.826** |
| poincare_hyphc_cold    | 0.313 | 0.576 |
| poincare_hyphc_warm    | 0.453 | 0.460 |
| poincare_no_hyphc      | 0.341 | 0.470 |

Euclidean leads the best Poincaré config (cold+bk=1) by +0.25 hDP. Consistent
with the thesis: **hyperbolic geometry helps only when the label structure has
a genuine multi-level hierarchy**. STL-10's 2-level (vehicle/animal) split
doesn't give hyperbolic anything to grip, so the extra manifold machinery is
pure overhead. Report this in the paper as the counterpoint — not every
dataset benefits. Dataset-scope predictor: # of levels in the label hierarchy.

### `backbone_knn=True` uniformly helps across conditions
On STL-10, `use_backbone_knn_similarities=True` improved hDP in *every* of the
8 condition pairs (cold+Poincaré +0.26, no_hyphc+Poincaré +0.13,
warm+Poincaré +0.01, Euclidean +0.03). Default it to True unless the
intervention is specifically about the similarity source.

### HypHC warm schedule can be *worse* than cold on STL-10
Opposite of TinyImageNet: on STL-10 cold (0.2/0.1) + bk=1 beats warm
(1.0/0.5) + bk=1 by +0.12 hDP. Reinforces the per-dataset-tuning rule from the
HypHC section — don't transfer temperature schedules across datasets.

## Open questions

- Does `head_warmup_epochs=50` eliminate the dip? (job 524086, pending —
  less interesting now that HypHC is confirmed as the sole class signal
  under the winner recipe; head warmup still gives HypHC a cleaner starting
  geometry, but the supervised-baseline ceiling is leakage-capped anyway).
- Can geodesic NT-Xent on the hyperbolic head outputs provide a second
  class-preserving signal independent of HypHC? (requires
  `contrastive.euclidean=false` + backbone unfrozen OR a new projection).
- What is the true performance when ImageNet label leakage is removed?
  (requires 521970 → `configs/cub200_simclr_transfer.yaml`) — primary
  open question now.

## 2026-04-24 additions

### HIER env works with newer timm/torch — pinning unnecessary
HIER's `environment.yml` pins `torch==1.8.1+cu111`, `timm==0.4.12`,
`pytorch_metric_learning` unpinned. Installing into our `hyper` env gave us
`timm 1.0.26` and `pytorch_metric_learning 2.9.0` — way newer. Job 534388
(CUB sanity) still landed at R@1 ≈ 0.68-0.69, matching the paper's 0.70 (out
of 0.701 published). Don't spin up a separate frozen-dep env for HIER unless
a specific feature breaks; newer timm is fine for ResNet50 + Poincaré head.

### HIER's CIFAR-100 run needs ~4h / 164G mem for 50k images × 100 epochs
Our first sub at 64G hit OOM at epoch ≈1 (job 534398). Bumped to 164G and
the rerun (534399) finished clean in 3:47h on 1×H100. CIFAR-100 with
batch=90, 224² upsample, bn_freeze=True, fp16=True. Use 128-164G for any
torchvision dataset ≥50k images through their pipeline.

### HIER on CIFAR-100 (supervised) vs ours (unsupervised) — the paper story
Ran HIER with emb=128, 100 epochs, full-class split (50k train + 10k eval,
all 100 classes), seed 1. Piped best-epoch embeddings through
`scripts/eval_baseline_embeddings.py` (new, Poincaré c=0.1). At epoch 55:

- HIER: 1-NN = **0.813**, hDP_ward = 0.783, hDP_complete = 0.755.
- Ours PoincaréBall d=16: 1-NN = 0.589, hDP_ward = **0.901**.

HIER uses class labels and gets +0.22 on R@1. We use no labels and get +0.12
on hDP_ward. Cleanest framing for the paper: supervised DML wins pure
retrieval, unsupervised hyperbolic clustering wins hierarchy-preserving
structure. Different optimization targets produce different dominant axes.

### iNat Poincaré density collapse at `curvature_magnitude=1.0`
Main VSC sweep (9 runs, 3 conds × 3 seeds, `sbatch_vsc_inat18_h2h_dino_frozen.sh`)
produced 6 Poincaré runs (3 seeds × {warm HypHC, no HypHC}) — **all 6 collapsed
to n@1 ≈ 0.001** by epoch 29-50. Same config but `manifold_name=Euclidean`
gave n@1 ≈ 0.04. This is a Poincaré-specific failure: the only difference
between the two configs is the manifold.

Hypothesis: iNat has 8142 classes (vs ImageNet's 1000), so the class-density
per unit Poincaré-volume is ~8× higher at the same latent_dim=128. The
`HyperbolicLinearPP` projection head pushes all points to the boundary where
tanh saturates and they become numerically indistinguishable. Lowering
`curvature_magnitude` effectively flattens the ball (c→0 degenerates to
Euclidean), pulling points away from boundary.

**Rescue sweep ready**: `sbatch_vsc_inat18_h2h_rescue.sh` tests
c ∈ {0.1, 0.3, 0.5} at d=128, 3 seeds each. HIER itself uses c=0.1 on
Poincaré, so the low-end is a known-working value from a neighboring method.
If even c=0.1 fails, the problem isn't curvature density and we need a
different intervention (dim=256, different projection architecture, etc.).

**Rescue sweep result (finished 2026-04-28)**: 9 runs landed; on hDP_ward,
**c=0.3 wins, not the HIER-borrowed c=0.1**:

| c | hDP_ward (final ce=50) | n@1 |
|---|---|---|
| 0.3 | **0.452 ± 0.106** | 0.0002 |
| 0.5 | 0.260 ± 0.009 | 0.0014 |
| 0.1 | 0.201 ± 0.029 | 0.0023 |

Two takeaways:
1. **HIER's c=0.1 does not transfer.** Borrowing the curvature from a
   neighboring supervised method on a different dataset turned out to be
   the *worst* of the three. The "flatter ball ⇒ less boundary crush ⇒
   better" intuition broke around c=0.3 and reversed below it; at c=0.1
   the manifold is so close to Euclidean that the hyperbolic clustering
   loss has no geometry to exploit. The sweet spot is dataset-specific
   and has to be measured per dataset, not borrowed.
2. **Hierarchy can recover even when retrieval can't.** All 9 rescue
   runs are still leaf-collapsed: 1-NN ≈ 0.0001–0.002 vs random
   1/8142 ≈ 0.00012. But the Ward dendrogram still captures coarse
   taxonomy structure (8142 species → ~14 phyla), which is what hDP_ward
   measures. So on extreme-scale label sets the hyperbolic head can
   produce a tree-aligned partition without producing a discriminative
   embedding. For the paper this means iNat is reportable as a
   *hierarchy-only* result — claiming retrieval-grade learning at 8142
   classes would be a misread of the metrics.

Caveat before claiming a "win" on iNat: frozen FB DINO ResNet50 (no
training) on the same eval pipeline needs to land first as a comparison
point. If frozen DINO already hits hDP_ward ≈ 0.45+ at matched dim, the
rescue sweep adds nothing on top and the row should be dropped or
reframed as "trained does not beat frozen on iNat hierarchy."

### Hydra struct-mode trap: `hyperbolic_smoothing` lives under `architecture`
The sbatch `sbatch_cifar100_poincare_d{16,32}.sh` scripts originally had
`clustering.hyperbolic_smoothing=50.0` which dies instantly with:

```
Could not override 'clustering.hyperbolic_smoothing'.
Key 'hyperbolic_smoothing' is not in struct
    full_key: clustering.hyperbolic_smoothing
```

Correct override is `architecture.hyperbolic_smoothing=50.0`. Every other
config YAML (config.yaml:132, inat18_dino.yaml:16, imagenet_dino.yaml:16,
etc.) has it under `architecture:`. Jobs 534414/534416 failed in 10-30s from
this. Before submitting a new sbatch with `.yaml`-style dotted overrides,
`grep` the actual config to confirm the key's parent namespace.

### Frozen-backbone lower-bound infrastructure
New pair of scripts:
- `scripts/extract_frozen_backbone_embeddings.py` — loads any ResNet50
  state_dict (strips module./backbone./model. prefixes), replaces fc with
  Identity, forwards an eval split, saves `(x, y, idx)` in HIER-compatible
  `.pt` format. Supports `--dataset ∈ {cifar100, imagenet, tiny_imagenet, inat18}`.
- `scripts/eval_baseline_embeddings.py` — consumes the `.pt`, computes
  Euclidean or Poincaré pairwise distances, builds WordNet hierarchy clusters
  when a hierarchy file exists, runs `src/evaluation.py:evaluation()` with
  our standard metrics, writes JSON to `results_baseline_embeddings/`.

Same two scripts handle (a) frozen-DINO lower-bound rows, (b) SeCu/SCAN
checkpoint re-eval, (c) HIER saved embeddings. One pipeline for all
external baselines.

### Running scripts/*.py from anywhere: prepend repo root to sys.path
`uv run python scripts/foo.py` does NOT put the repo root in `sys.path` —
Python puts `scripts/` first, and `from src.evaluation import ...` dies with
`ModuleNotFoundError: No module named 'src'`. Fix: add
`sys.path.insert(0, str(Path(__file__).resolve().parent.parent))` near the
top of any script under `scripts/` that imports from `src.*`.

### Login-node memory is too small for eval pipeline (10k × 10k distance matrix)
Attempted to run `scripts/eval_baseline_embeddings.py` directly on the login
node. Process got SIGKILLed (exit 137) before emitting any output. Login
node has 15G total / ~6G free; CIFAR-100 test set is 10k × 10k float32
pairwise = 400M, plus sklearn agglomerative + torch/scipy imports easily
push past the limit. Run baseline eval under `sbatch` with ≥32G mem, not
interactively. `sbatch_eval_hier_cifar100.sh` is the template.

## 2026-04-28 additions

### SeCu cluster head is seed-sensitive — one bad seed can collapse training without crashing
`secu-tiny_imagenet-s99` (slurm 541124, 2026-04-28) trained 400 epochs
without crashing but produced an embedding-collapsed model:

| metric | s42 | s56 | **s99** |
|---|---:|---:|---:|
| 1-NN | 0.1092 | 0.0923 | **0.0050** |
| ward NMI | 0.4271 | 0.4167 | **0.0447** |
| ward acc | 0.1331 | 0.1237 | **0.0054** |

Random fine-class baseline is ≈ 1/200 = 0.005. The s99 training loss
plateaued 1.5–2 nats above sibling seeds from epoch 50 and climbed another
~2 nats between epochs 300–394 — **stuck-then-diverged cluster head, no
NaN/Inf in the log**. Same script + data + hparams converged under seeds
42 and 56; `worker_seed = 99 + rank` seeded a bad cluster-center init.

**Action taken:** removed `results/secu-tiny_imagenet-s99.{json,log}` so the
aggregator doesn't bake collapse into the SeCu mean/std; resubmitted with
seed=7 for a third good seed. Broken checkpoint kept at
`model/secu_tiny_imagenet_s99_*.pth.tar` for reference. See
`docs/secu_ti_seed99_collapse.md` for the full loss trajectory.

**Implication for baseline tables:** the SeCu TinyImageNet n=3 aggregate in
`baselines/aggregate.md` (mean hDP_ward = 0.160, sd = 0.040) **still
includes seed=99**. Reaggregate after seed=7 lands; expect the std to drop
sharply and the mean to rise as the collapse is excluded.

**If a fresh seed also diverges**, candidates to investigate before assuming
random failure: LR schedule on the cluster head (the climb starts at the
LR-decay boundary), cluster-init scheme (k-means warm-start vs. random),
gradient clipping on the head.

### SeCu CIFAR-100 d=16 ≈ d=32 on hierarchy, d=32 only wins n@1
The MEMORY note "latent=32 beats 16" is true on n@1 (+0.04) but **a wash
on hDP_ward** (within std). Aggregate from n=3 seeds {42, 56, 99}:

| dim | hDP_ward | n@1 | ward NMI |
|---|---|---|---|
| 16 | 0.4156 ± 0.011 | 0.4686 ± 0.010 | 0.5842 ± 0.007 |
| 32 | 0.4147 ± 0.012 | 0.5090 ± 0.008 | 0.5983 ± 0.005 |

When the SeCu comparison is hierarchy-focused, dim doesn't matter; it
matters for retrieval. SeCu d=128 uses a multi-head per-K eval that is
not directly comparable to the single-head numbers above.
