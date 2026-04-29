# Anonymous Repo Cleanup — Instructions

Run these on a fresh branch (e.g. `anon`). The plan is: drop everything that
isn't strictly needed to reproduce the paper's experiments, scrub
identifying strings, inline the private submodule, then squash to a single
commit so `git log` is empty of author info. **Run on a copy** — these
operations are destructive.

```bash
# Start from clean state
git switch main && git pull
git switch -c anon
git status   # should be empty before continuing
```

## 1. Delete untracked / external repos

These were cloned into the working tree for baseline runs but aren't ours.

```bash
rm -rf HypHC/ baselines/HIER-CVPR23/ baselines/UnHyperML/
# baselines/hier_extensions/ stays — that's our adapter code
# baselines/aggregate.md stays only if you want a baseline-numbers file in the release; otherwise rm it
```

## 2. Delete internal docs and scratch artifacts

These contain author handles, internal results tracking, and meeting-style
narrative that shouldn't ship with a review repo.

```bash
rm -f LESSONS_LEARNED.md RESULTS.md paper.md ANONYMOUS_REPO_INSTRUCTIONS.md
rm -rf docs/ related_work/ baseline_dp_eval/ old_slurm/
rm -rf results/ results_baseline_embeddings/ lightning_logs/ logs/
rm -rf .claude/ .vscode/ .venv/ __pycache__/
rm -f imagenet.yaml environment.yaml CLAUDE.md   # legacy / Claude-Code-only
```

## 3. Trim sbatches

Reviewers don't need 56 SLURM scripts pinned to our cluster. Keep none, or
keep **one** minimal example renamed to `scripts/reproduce_example.sh`
showing the train.py CLI. Default: delete all.

```bash
rm -f sbatch_*.sh
```

If you keep one, strip `#SBATCH -A datamining`, `--constraint`, partition
names, and any cluster path references.

## 4. Trim scripts/

Drop scripts that depend on our wandb account or internal sweep logs. Keep
only what a reviewer would need to rebuild data artifacts or evaluate
baselines.

```bash
rm -f scripts/query_wandb.py \
      scripts/fetch_cifar100_trajectories.py \
      scripts/diff_cifar100_winners_losers.py \
      scripts/analyze_tiny_imagenet_sweep.py \
      scripts/smoke_inat_hierarchy.py \
      scripts/reorganize_inat18.py \
      scripts/download_inat18.sh
# Keep: build_inat18_taxonomy.py, eval_baseline_embeddings.py,
#       extract_frozen_backbone_embeddings.py, pca_reduce_embeddings.py,
#       run_chami_hyphc_cifar100.py
```

## 5. Trim configs/

Drop dataset configs not used in the paper. Keep CIFAR-100, TinyImageNet,
ImageNet, iNat18, and the base `config.yaml`.

```bash
# Delete dataset configs the paper doesn't use
rm -f configs/cub200_*.yaml \
      configs/imagenet_simclr.yaml configs/imagenet_poincare.yaml \
      configs/imagenet_dino_pretrain.yaml
rm -f configs/dataset/cub200.yaml \
      configs/dataset/cifar10.yaml configs/dataset/cifar100_20.yaml \
      configs/dataset/fmnist.yaml configs/dataset/kmnist.yaml \
      configs/dataset/mnist.yaml configs/dataset/iris.yaml \
      configs/dataset/gtsrb.yaml configs/dataset/stl10.yaml
# Keep: config.yaml + the four dataset_dino.yaml files + dataset/{cifar100,tiny_imagenet,imagenet,inat18}.yaml
```

## 6. Anonymize wandb in remaining configs

Either neutralize the entity or drop the wandb block entirely. Reviewers
won't have wandb access; either way works.

```bash
# Option A: neutralize entity (keeps wandb logging if user provides their own)
grep -rl "kevin-sidak-team" configs/ | xargs sed -i "s/kevin-sidak-team/anon/g"

# Option B: comment out the wandb block in each config
# (do this manually — search for `wandb:` and prefix lines with `#`)
```

## 7. Inline the private submodule

`src/hyperbolic_math` points to a private GitHub repo (`univie-dm/...`)
that reviewers can't access. Inline its contents and drop the submodule
metadata.

```bash
# 7a. Copy submodule contents into the parent repo as plain files
git submodule update --init --recursive
rm -rf src/hyperbolic_math/.git src/hyperbolic_math/.github 2>/dev/null
git rm --cached src/hyperbolic_math
rm -f .gitmodules
git add src/hyperbolic_math   # adds files directly, not as submodule

# 7b. Scrub identifying info in submodule's own README/pyproject
grep -rl "univie-dm\|kevin\|sidak" src/hyperbolic_math/ \
    | xargs sed -i "s/univie-dm/anon/g; s/kevin\\.sidak@\\S*/anon@anon/gi"
```

## 8. Final string scrub

Walk the remaining tree and confirm no identifying strings leak.

```bash
# Should produce zero output:
grep -rn "kevin\|sidak\|univie\|/srv/home/users\|ks09437\|datamining\|kevin-sidak-team" \
    --exclude-dir=.git --exclude-dir=.venv --exclude="ANONYMOUS_REPO_INSTRUCTIONS.md" .

# Common stragglers to fix manually:
# - pyproject.toml: `description = "Add your description here"` → set a neutral one
# - README.md: rewrite (see §10)
# - any TODO/FIXME comments naming a person
```

## 9. Replace README

The current `README.md` is two lines. Write a reproduction-focused replacement
that does not name authors, institutions, or the wandb account. Suggested
sections: Installation (uv sync), Datasets (where to put them), Training
example (one CLI), Evaluation, License. Keep it under ~80 lines.

## 10. Wipe git history

A reviewer reading `git log` should see no author names, dates that
correlate with internal milestones, or commit messages referencing internal
infrastructure. The cleanest way is an orphan branch with a single commit:

```bash
# Stage cleanup
git add -A && git commit -m "wip: cleanup"

# Create orphan branch with no history
git checkout --orphan anon-clean
git add -A
GIT_AUTHOR_NAME="Anonymous" GIT_AUTHOR_EMAIL="anon@anon.invalid" \
  GIT_COMMITTER_NAME="Anonymous" GIT_COMMITTER_EMAIL="anon@anon.invalid" \
  git commit -m "Initial submission"

# Verify single anonymous commit
git log --format="%h %an <%ae> %s"
# expected: one line with "Anonymous <anon@anon.invalid> Initial submission"
```

Do **not** push the `anon` branch (the one with cleanup history) anywhere
public — only push `anon-clean`.

## 11. Submission

Two common targets:

- **anonymous.4open.science**: paste a public GitHub URL. The service strips
  git history automatically, but it does serve files as-is, so steps 1–8
  still need to be done before the source tree is uploaded. The repo behind
  the link can be a fresh GitHub repo containing only the `anon-clean`
  branch's contents.
- **A fresh public GitHub repo with anonymous author**: create it under a
  throwaway account, push only `anon-clean`. The orphan-branch step makes
  the commit history acceptable.

## 12. Verification checklist

Before linking the repo in the submission:

- [ ] `git log` shows only anonymous commits
- [ ] `grep -rn "kevin\|sidak\|univie\|/srv/home\|ks09437\|datamining\|kevin-sidak-team" --exclude-dir=.git .` returns nothing
- [ ] No `.claude/`, `.vscode/`, `.venv/`, `data/`, `logs/`, `lightning_logs/`,
  `results/`, `results_baseline_embeddings/`, `__pycache__/` in tree
- [ ] No `LESSONS_LEARNED.md`, `RESULTS.md`, `paper.md`, `CLAUDE.md`, `ANONYMOUS_REPO_INSTRUCTIONS.md`
- [ ] `.gitmodules` is gone; `src/hyperbolic_math/` is plain files
- [ ] `README.md` reads as a reproduction guide, no author info
- [ ] `pyproject.toml` has a real description, no author field
- [ ] `uv sync && uv run python train.py --help` works on a fresh checkout
- [ ] One representative training command in the README runs end-to-end on
  a small dataset (sanity check)
