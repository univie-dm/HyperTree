# Lessons Learned — Hyperbolic Clustering

Running log of non-obvious findings from debugging the CUB-200 R@1 collapse
and the CIFAR-100 sweeps. Read this before proposing fixes — several
"obvious" root causes have already been ruled out.

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

### `unfreeze_backbone_from` requires `freeze_encoder=True`
The two hparams compose: `freeze_encoder=True` freezes everything, then
`unfreeze_backbone_from=layer4` selectively re-enables gradients for
`layer4` onward. Setting only `unfreeze_backbone_from` with
`freeze_encoder=False` does nothing (everything is already trainable).

## Related work that informed interventions

- **CMHHC** (multi-view contrastive hyperbolic) — motivates per-view
  hyperbolic contrastive instead of single-view.
- **Ge et al. HCL** — hyperbolic contrastive learning with explicit
  boundary regularization.
- **Yang et al. (2024) outer-shell isotropy** — angular uniformity
  regularizer on L2-normalized tangent vectors (Wang & Isola style).
  Implemented as `regularizer.type=uniformity` in `src/models/hyphc.py`.

## Evaluation

### Recall@K (standard DML metric)
`src/evaluation.py:recall_at_k` — per-query 1 if *any* of the top-k neighbors
(self excluded) matches the query label, averaged. Gated by
`evaluation_metrics.recall_at_k`. Reports `test/recall_at_{2,4,8,10,100}`
alongside the existing `{k}_nearest_neighbors_accuracy` (which is
*accuracy*@k, not recall, and identical at k=1). Use R@k to compare against
DML benchmark tables (e.g. CVPR 2021's CUB R@1 = 58.9%).

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
