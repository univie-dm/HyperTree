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
2. **CIFAR-100 hyperbolic vs. supervised HIER (matched-dim, multi-seed):**
   ours d=16 Poincaré peak hDP_ward = 0.893 ± 0.021 vs HIER d=16 peak =
   0.845 ± 0.010 (3 seeds). Ours wins hierarchy by +0.05 even at matched
   dim; HIER still wins n@1 (0.69 vs 0.51 at d=16). HIER hDP gently
   decreases with dim (d=16: 0.845, d=32: 0.841, d=128: 0.813), same
   pattern as ours. Both methods peak in the first ~15 epochs.
3. **Use ce=0 hDP_ward (frozen backbone + random-init clustering head,
   evaluated through our pipeline) as the matched-architecture frozen
   anchor.** Same encoder, same projection-head architecture, same
   eval — only the head training differs, so Δ from ce=0 to ce=final is
   exactly the contribution of clustering training. This replaces the
   earlier "Frozen DINO 2048-D = 0.884" comparison, which was apples-to-
   oranges (different dim and no projection head). The new picture is
   sharper: HypHC gains a lot on TI/ImageNet, Euclidean barely learns
   on ImageNet d=128, and iNat training *regresses* hDP_ward.
4. **iNat18: training does not improve hDP_ward at any tested curvature.**
   c=1.0 collapses 6/6 runs (n@1≈0.001). Rescue sweep c∈{0.1, 0.3, 0.5} ×
   3 seeds finished 2026-04-28. Against the matched-architecture ce=0
   anchor: c=0.3 ce=0=0.490 → final=0.452 (**Δ=−0.04, regressed**);
   c=0.1 0.176→0.201 (+0.025); c=0.5 0.255→0.260 (+0.005). The "c=0.3
   beats c=0.1" headline from the rescue table is real but reflects the
   random-head starting point at that curvature, not a learned win. No
   curvature beats its ce=0. **iNat18 has no clustering-training success
   story under the current head architecture** — drop or reframe.
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

## Paper-section view (2026-04-28)

This section maps existing results onto paper.md's experimental structure:
§5.3 Metric Learning, §5.4 Deep Clustering, §5.4 Dimensionality Ablation,
§4.2.3 LCA ablation. Detailed per-dataset breakdowns continue below.

### §5.3 Metric Learning (n@1, R@k)

Standard DML metrics. Higher is better. **Ours** = best Poincaré + HypHC
config per dataset; baseline rows are external methods.

| dataset | method | d | n@1 | R@2 | R@4 | R@8 | seeds |
|---|---|---|---|---|---|---|---|
| CIFAR-100 | **Ours** (peak Poincaré) | 16 | 0.506 ± 0.013 | 0.583 | 0.681 | 0.770 | 4 |
| CIFAR-100 | HIER (supervised, peak) | 16 | 0.692 ± 0.060 | 0.781 ± 0.046 | 0.845 ± 0.031 | 0.894 ± 0.019 | 3 |
| CIFAR-100 | HIER (supervised, peak) | 32 | 0.758 ± 0.044 | 0.830 ± 0.031 | 0.880 ± 0.016 | 0.915 ± 0.008 | 3 |
| CIFAR-100 | HIER (supervised, peak) | 128 | **0.742 ± 0.003** | 0.819 ± 0.000 | 0.872 ± 0.002 | 0.912 ± 0.001 | 3 |
| CIFAR-100 | SeCu | 32 | 0.5090 ± 0.008 | 0.615 | 0.712 | 0.795 | 3 |
| CIFAR-100 | SCAN | 128 | 0.4120 | 0.523 | 0.639 | 0.745 | 3 |
| TinyImageNet | **Ours** Poincaré + HypHC | 128 | 0.284 ± 0.003 | — | — | — | 3 |
| TinyImageNet | SeCu † | (default) | 0.0688 ± 0.056 † | — | — | — | 3 |
| TinyImageNet | SCAN | 128 | 0.037 ± 0.006 | 0.066 ± 0.007 | 0.113 ± 0.007 | 0.184 ± 0.010 | 3 |
| ImageNet | **Ours** Poincaré + HypHC warm bk=0 | 128 | 0.299 ± 0.008 | — | — | — | 2 |
| ImageNet | SCAN (released ckpt) | 128 | 0.513 | 0.610 | 0.696 | 0.769 | 1 |
| ImageNet | Frozen DINO 2048-d (no train) | 2048 | — | — | — | — | 1 |
| iNat18 | (collapsed at leaf level) | 128 | ≤ 0.002 | — | — | — | — |

> † SeCu TI n@1 includes the collapsed seed=99 (random for 200 classes).
> Reaggregate after seed=7 lands.
>
> HIER CIFAR-100 numbers are peak across the 100-epoch trajectory (matching
> our peak convention); peak typically lands at epoch 5–15. Source:
> 48 eval JSONs in `results_baseline_embeddings/hier_c100_*.json`,
> 3 seeds × {d=16, d=32, d=128}.
>
> **§5.3 missing:** R@k columns for non-CIFAR datasets; HIER on TI/ImageNet
> (TI: local job 554382 OOM'd at ~50min on all 3 seeds at 164G — eval_10.pt
> written but training crashed before epoch 100; resubmitted as job 559140
> with 256G mem; ImageNet: VSC, queued/running); UHML (Yan 2021) row
> (CUB sanity job 554383 failed instantly with `KeyError: VGG16-BN`
> — UHML's models registry only knows `BN-Inception`; resubmitted as
> job 559141; full integration on our datasets still requires writing
> UHML adapters analogous to HIER's `baselines/hier_extensions/`).

### §5.4 Deep Clustering (hDP_ward, DP_ward, NMI, ARI)

| dataset | method | d | hDP_ward | ce=0 anchor | Δ from ce=0 | seeds |
|---|---|---|---|---|---|---|
| CIFAR-100 | **Ours** Poincaré (peak) | 16 | **0.893 ± 0.021** | 0.755 ± 0.004 | **+0.138** | 4 |
| CIFAR-100 | **Ours** Poincaré (final ce=300) | 16 | 0.753 ± 0.011 | 0.755 ± 0.004 | −0.002 | 4 |
| CIFAR-100 | HIER (supervised, peak) | 16 | 0.845 ± 0.010 | — | NMI 0.749 ± 0.053 | 3 |
| CIFAR-100 | HIER (supervised, peak) | 32 | 0.841 ± 0.048 | — | NMI 0.790 ± 0.039 | 3 |
| CIFAR-100 | HIER (supervised, peak) | 128 | 0.813 ± 0.016 | — | NMI 0.766 ± 0.003 | 3 |
| CIFAR-100 | SeCu | 16 | 0.4156 ± 0.011 | — | — | 3 |
| CIFAR-100 | SeCu | 32 | 0.4147 ± 0.012 | — | — | 3 |
| CIFAR-100 | SCAN | 128 | 0.3531 | — | — | 3 |
| CIFAR-100 | Chami HypHC † | {16,32,128} | TBD | — | — | 3 |
| TinyImageNet | **Ours** Poincaré + HypHC (final) | 128 | **0.714 ± 0.003** | 0.471 ± 0.067 | **+0.243** | 3 |
| TinyImageNet | Ours Poincaré no-HypHC | 128 | 0.517 ± 0.018 | 0.464 ± 0.058 | +0.053 | 3 |
| TinyImageNet | Ours Euclidean | 128 | 0.468 ± 0.120 | 0.487 ± 0.041 | −0.019 | 3 |
| TinyImageNet | SeCu (incl. collapsed seed) | (default) | 0.160 ± 0.040 | — | — | 3 |
| TinyImageNet | SCAN | 128 | 0.091 ± 0.003 | — | NMI 0.339 ± 0.008 | 3 |
| ImageNet | **Ours** Poincaré + HypHC warm bk=0 (final) | 128 | **0.683 ± 0.012** | 0.467 ± 0.008 | **+0.217** | 2 |
| ImageNet | Ours Poincaré no-HypHC bk=0 | 128 | 0.636 ± 0.033 | 0.429 ± 0.002 | +0.207 | 2 |
| ImageNet | Ours Poincaré + HypHC cold bk=0 | 128 | 0.473 ± 0.011 | 0.448 ± 0.029 | +0.025 | 2 |
| ImageNet | Ours Euclidean bk=0 | 128 | 0.519 ± 0.105 | 0.505 ± 0.088 | +0.014 | 2 |
| ImageNet | SCAN (released ckpt) | 128 | 0.246 | — | NMI 0.722 / ARI 0.274 | 1 |
| iNat18 | Ours Poincaré c=0.3 + HypHC warm | 128 | 0.452 ± 0.106 | 0.490 ± 0.096 | **−0.038** | 3 |
| iNat18 | Ours Poincaré c=0.5 + HypHC warm | 128 | 0.260 ± 0.009 | 0.255 ± 0.013 | +0.005 | 3 |
| iNat18 | Ours Poincaré c=0.1 + HypHC warm | 128 | 0.201 ± 0.029 | 0.176 ± 0.016 | +0.025 | 3 |

> † Chami HypHC at d∈{16,32,128} × 3 seeds: **submitted as job 549712**
> (2026-04-28). Eval auto-runs after each train.
>
> **ce=0 anchor** = hDP_ward measured at clustering epoch 0, i.e. the
> pretrained encoder + random-init clustering head + our standard eval
> pipeline. Same architecture, same hierarchy clusters, same metric — only
> the head training differs. Δ from ce=0 to ce=final isolates the
> contribution of clustering training. This replaces the dim-mismatched
> "Frozen DINO 2048-D" comparison.
>
> **§5.4 missing:** HIER on TI/ImageNet, Chami end-to-end on
> any dataset (running), gHHC (paper.md says intractable — cite, don't run).
> SCAN now landed for TI (3 seeds × d=128 retrain) and ImageNet (released
> ckpt, n=1).

### §5.4 Dimensionality Ablation (Ours, hDP_ward by d)

The "exceptionally low-dimensional" claim. Ours stays high as d shrinks;
baselines shrink with d.

| dataset | d=16 | d=32 | d=128 | source |
|---|---|---|---|---|
| CIFAR-100 (Ours, peak) | **0.893 ± 0.021** | 0.882 ± 0.010 | TBD | n=4 / n=4 / pending |
| CIFAR-100 (SeCu) | 0.4156 ± 0.011 | 0.4147 ± 0.012 | 0.3798 ± 0.038 | n=3 / n=3 / n=2 |
| TinyImageNet (Ours) | TBD ‡ | TBD ‡ | 0.714 ± 0.003 | pending / pending / n=3 |
| ImageNet (Ours, frozen DINO) | TBD § | 0.581 ± 0.044 (no_hyphc) | 0.683 ± 0.012 (warm bk=0) | pending / n=3 / n=2 |
| iNat18 (Ours, c=0.3) | TBD | TBD | 0.452 ± 0.106 | only d=128 in flight |

> ‡ TI d=16 + d=32 **submitted as jobs 549710 / 549711** (2026-04-28).
> § ImageNet d=16 in flight (job 541956).

### §4.2.3 LCA Ablation (`ours` rectified vs Chami's standard)

| dataset | Δ hDP_ward (`ours` − `chami`) | result |
|---|---|---|
| CIFAR-100 d=16 | < 0.01 | **a wash — does not motivate §4.2.3** |
| TinyImageNet d=128 | TBD | sbatch_vsc_lca_ablation_tinyimagenet.sh (submitted) |
| ImageNet d=128 | TBD | sbatch_vsc_lca_ablation_imagenet.sh (submitted) |
| iNat18 d=128 c=0.3 | TBD | sbatch_vsc_lca_ablation_inat18.sh (submitted) |

> §4.2.3 is the paper's first contribution but currently has no measured
> support. The TI ablation is the most likely to show a real Δ — TI has
> the strongest hyperbolic story (+0.16 hDP over Euclidean), so any LCA
> effect should manifest there if anywhere.

### Section status tracker

| paper section | writable today? | blocking gap |
|---|---|---|
| §4.2.3 rectified LCA | **NO** | TI/ImageNet/iNat ablation submitted, awaiting results |
| §4.2.4 scalability | YES | iNat (8142 classes) and ImageNet (1.28M images) trained — Chami's O(n²) explicitly intractable |
| §5.3 metric learning | partial | R@k columns for non-CIFAR datasets, UHML baseline |
| §5.4 deep clustering | partial | HIER on TI/ImageNet, Chami end-to-end (running); SCAN landed for TI (n=3) and ImageNet (released ckpt, n=1) |
| §5.4 dim ablation | partial | TI d=16/d=32 (running), ImageNet d=16 (running) |
| Manifold-agnostic claim | **NO** | Hyperboloid runs only exist on CIFAR-100 |

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

| method | d | ce=0 hDP | hDP_ward (peak) | n@1 (peak) | peak ce | final hDP_ward |
|---|---|---|---|---|---|---|
| Ours PoincaréBall | 16 | 0.755 ± 0.004 | **0.893 ± 0.021** (Δ +0.138) | 0.506 ± 0.013 | 11–26 | 0.753 ± 0.011 (Δ −0.002) |
| Ours PoincaréBall | 32 | 0.768 ± 0.007 | 0.882 ± 0.010 (Δ +0.114) | 0.521 ± 0.004 | 11–55 | 0.737 ± 0.006 (Δ −0.031) |

n=4 runs per row (3 paper_main seeds + 1 historical-recipe rerun). Final
values are clustering epoch 300; peak is the best hDP_ward observed across
the 300-epoch trajectory.

### External baselines (CIFAR-100, single-head Sec 5.3/5.4 format)

Aggregated from `baselines/aggregate.md` (2026-04-28). hDP_ward column is
`hierarchical_dendrogram_purity_ward` so it is directly comparable to the
"Ours" rows above.

| method | d | hDP_ward | n@1 (fine) | NMI (ward) | n seeds | source |
|---|---|---|---|---|---|---|
| **HIER** (supervised, peak) | 16 | 0.845 ± 0.010 | 0.692 ± 0.060 | 0.749 ± 0.053 | 3 | seeds 1/2/3, peak across 100ep trajectory |
| **HIER** (supervised, peak) | 32 | 0.841 ± 0.048 | 0.758 ± 0.044 | 0.790 ± 0.039 | 3 | seeds 1/2/3, peak across 100ep trajectory |
| **HIER** (supervised, peak) | 128 | 0.813 ± 0.016 | 0.742 ± 0.003 | 0.766 ± 0.003 | 3 | seeds 1/2/3, peak across 100ep trajectory |
| SeCu | 16 | 0.4156 ± 0.0109 | 0.4686 ± 0.0099 | 0.5842 ± 0.0072 | 3 | published recipe, seeds {42, 56, 99} |
| SeCu | 32 | 0.4147 ± 0.0119 | 0.5090 ± 0.0082 | 0.5983 ± 0.0045 | 3 | published recipe, seeds {42, 56, 99} |
| SeCu | 128 | 0.3798 ± 0.0380 ‡ | 0.4754 ± 0.0591 | 0.5635 ± 0.0412 | 2 | published recipe, seeds {42, 99} |
| SCAN | 128 | 0.3531 | 0.4120 | 0.5035 | 3 | authors' published checkpoint, std=0 (deterministic ckpt) |

> ‡ The SeCu d=128 row is the single-head re-eval format. The SeCu paper's
> own multi-head per-K eval gives best DP=0.626 at K=180 — not directly
> comparable to the single-head numbers above; see
> `memory/project_secu_baseline_cifar100.md`. SeCu d=128 also has only 2
> seeds (42, 99) — seed 56 not yet rerun.

> SeCu d=16 ≈ d=32 on hDP_ward (within std); the dim story for SeCu is
> n@1-only, where d=32 beats d=16 by +0.04. See `LESSONS_LEARNED.md` for
> the full d=16 vs d=32 breakdown.

### CIFAR-100 framing

At peak, our unsupervised d=16 Poincaré (hDP_ward 0.893) beats supervised
HIER at every matched dim: HIER d=16 = 0.845 ± 0.010 (Δ +0.048), d=32 =
0.841 ± 0.048 (Δ +0.041 vs ours d=32 = 0.882), d=128 = 0.813 ± 0.016. Our
unsupervised method beats both unsupervised baselines (SeCu, SCAN) by
**+0.47** on hDP. HIER still wins 1-NN retrieval (0.69–0.74 across dims
vs ours 0.51) because supervised metric learning optimizes that directly.

The d=16 vs d=32 comparison shows our advantage is dim-stable: peak hDP only
varies by ~0.01 across these dims, while n@1 only varies by ~0.015. HIER's
hDP also gently decreases with dim, which is the same dim-shape as ours.

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

Aggregated from `baselines/aggregate.md` (2026-04-28). Default SeCu recipe,
locally retrained.

| method | d | hDP_ward | n@1 (fine) | NMI (ward) | n seeds | source |
|---|---|---|---|---|---|---|
| SeCu † | (default) | 0.1600 ± 0.0404 | 0.0688 ± 0.0559 | 0.2962 ± 0.2178 | 3 | published recipe, seeds {42, 56, 99} |
| SCAN   | 128 | 0.0914 ± 0.0033 | 0.0372 ± 0.0063 | 0.3389 ± 0.0080 | 3 | published recipe, seeds {42, 56, 99} |

> † **Caveat — the n=3 aggregate above still includes a collapsed seed.**
> `secu-tiny_imagenet-s99` (slurm 541124) trained 400 epochs without
> crashing but produced an embedding-collapsed model: 1-NN = 0.0050 (random
> for 200 classes), ward NMI = 0.0447, ward acc = 0.0054. Cluster head got
> stuck at init then diverged after the LR-decay boundary; seeds 42 and 56
> on the same script converge cleanly (1-NN ≈ 0.10, NMI ≈ 0.42). The
> collapsed `results/secu-tiny_imagenet-s99.{json,log}` have been removed
> from disk; resubmitted with seed=7 for a third good seed. Once seed=7
> lands, reaggregate — expect the means to rise sharply and the std to
> drop. See `docs/secu_ti_seed99_collapse.md` and `LESSONS_LEARNED.md`.

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

### ImageNet d=128 H2H sweep (16/16 finished, 4 conditions × 2 bk × 2 seeds)

Frozen FB DINO ResNet50, 50 ep clustering only. Condition × backbone_knn,
n=2 seeds (42, 43). All numbers are final epoch (ce=50).

| condition | bk | ce=0 hDP | hDP_ward | Δ | n@1 | seeds |
|---|---|---|---|---|---|---|
| **PoincaréBall + HypHC warm** | 0 | 0.467 ± 0.008 | **0.683 ± 0.012** | **+0.217** | 0.299 ± 0.008 | s42=0.675, s43=0.692 |
| PoincaréBall no-HypHC | 0 | 0.429 ± 0.002 | 0.636 ± 0.033 | +0.207 | 0.131 ± 0.006 | s42=0.613, s43=0.660 |
| PoincaréBall no-HypHC | 1 | 0.483 ± 0.020 | 0.630 ± 0.033 | +0.147 | 0.128 ± 0.005 | s42=0.606, s43=0.653 |
| Euclidean baseline | 0 | 0.505 ± 0.088 | 0.519 ± 0.105 | +0.014 | 0.294 ± 0.002 | s42=0.593, s43=0.445 |
| PoincaréBall + HypHC cold | 0 | 0.448 ± 0.029 | 0.473 ± 0.011 | +0.025 | 0.116 ± 0.002 | s42=0.481, s43=0.465 |
| Euclidean baseline | 1 | 0.505 ± 0.088 | 0.461 ± 0.022 | −0.044 | 0.291 ± 0.013 | s42=0.445, s43=0.476 |
| PoincaréBall + HypHC cold | 1 | 0.459 ± 0.014 | 0.452 ± 0.016 | −0.007 | 0.118 ± 0.001 | s42=0.463, s43=0.440 |
| **PoincaréBall + HypHC warm** | 1 | 0.449 ± 0.029 | **0.448 ± 0.029** † | −0.001 | 0.077 ± 0.020 | s42=0.468, s43=0.427 |

> † **Warm + bk=1 surprise collapse:** warm HypHC drops from 0.683 (bk=0)
> to 0.448 (bk=1). On every other condition the bk axis is roughly neutral
> (Δ ≤ 0.06). Hypothesis: warm HypHC + backbone-knn similarities double-
> commits to the wrong neighborhood structure early, since both signals
> push the head toward the frozen DINO geometry. Worth a separate ablation
> if warm+bk1 ends up the headline ImageNet config; otherwise just report
> warm+bk0.

**Best ImageNet d=128 row for the paper (final, no peak-vs-final
caveat):** PoincaréBall + HypHC warm, bk=0, hDP_ward = **0.683 ± 0.012**,
n@1 = 0.299. Beats Euclidean (best of bk=0 = 0.519) by **+0.16**, beats
no-HypHC (best bk=0 = 0.636) by **+0.05**.

Adding seed=44 to all 8 conditions is in flight as job 541955 (8 jobs at
%4 throttle, ETA ~24h on local cluster).

### External baselines (ImageNet)

| method | d | hDP_ward | n@1 (fine) | NMI (ward) | ARI (ward) | n seeds | source |
|---|---|---|---|---|---|---|---|
| SCAN (released ckpt) | 128 | 0.2465 | 0.5128 | 0.7219 | 0.2739 | 1 | authors' published checkpoint, deterministic eval (50k val) |

> SCAN ImageNet uses the official released checkpoint, evaluated through our
> pipeline on the full 50k val split. Strong on retrieval (n@1 = 0.51, beats
> our best by +0.21) and weak on hierarchy (hDP_ward = 0.246, our best
> beats it by +0.44). Same axis split as TI and CIFAR-100 — clustering-
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

### iNat18 framing

**Headline: under the matched-architecture ce=0 anchor, no curvature
beats its starting point on hDP_ward.** The c=0.3 row that previously
looked like a "win" (0.452 ± 0.106) is actually a regression (ce=0 was
0.490). The leaf level stays collapsed at every curvature (n@1 ≈
0.0001–0.002 vs random 1/8142). **iNat18 has no clustering-training
success story under the current head architecture.** Either the paper
omits iNat or reframes it as a scale-test of the pipeline (8142 classes,
hierarchy metric remains tractable, encoder doesn't collapse — but
training does not improve over random init).

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

After the `abstract.py` patch that wires `coarse_labels_array` for the
iNat hierarchical_dendrogram_purity_ward gate, all 9 runs now log hDP_ward
against the iNat taxonomy hierarchy. Numbers are final epoch (ce=50);
warm HypHC schedule, backbone_knn=True for every run.

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

## Pending experiments (2026-04-29)

Active or staged, in priority order:

| job | dataset | what | status |
|---|---|---|---|
| 541955 | ImageNet d=128 | seed=44 H2H (8 tasks) | local, running, ETA ~24h |
| 541956 | ImageNet d=16 | full H2H grid (16 tasks, seeds 42/43) | local, running, ETA ~5d |
| 559140 | TinyImageNet | HIER baseline (3 seeds × d=128) — `sbatch_tiny_imagenet_hier.sh` | local, queued; resubmit at 256G after 554382 OOM'd at 164G |
| 559141 | CUB | UHML (Yan 2021) sanity — `sbatch_uhml_cub_sanity.sh` | local, queued; resubmit with `--net BN-Inception` after 554383 hit unknown-network KeyError |
| (VSC) | ImageNet | HIER baseline (3 seeds × d=128) — `sbatch_vsc_imagenet_hier.sh` | submitted 2026-04-29 |
| (VSC) | iNat18 | Diagnostic Sweep A (head/training axis: d=256, cw=1.0, lr=1e-5, Hyperboloid; 12 runs) — `sbatch_vsc_inat18_diag_sweep_a.sh` | submitted 2026-04-29 |
| (VSC) | iNat18 | DINOv2 ViT-L/14 frozen baseline + matched-dim PCA — `sbatch_vsc_inat18_dinov2_frozen.sh` | submitted 2026-04-29 |
| (landed) | TinyImageNet / ImageNet | SCAN baseline | TI 3-seed retrain + ImageNet released ckpt eval, see §5.3/§5.4 |
| (staged) | TI / ImageNet / iNat18 | LCA-rectification ablation (`ours` vs `chami`) | submitted (lca_abl_ti, lca_abl_imnet running) |
| (staged) | ImageNet | Schedule-fix verification on hyphc_warm overshoot | not yet drafted |

Already landed and incorporated above:
- iNat18 rescue sweep c ∈ {0.1, 0.3, 0.5} (9 runs, finished 2026-04-28)
- ImageNet d=128 H2H (16/16 finished)
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
