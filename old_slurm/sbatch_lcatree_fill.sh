#!/bin/bash
#SBATCH --job-name=lcatree_fill
#SBATCH --time=4:00:00
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH -x vader,galadriel
#SBATCH --output=logs/lcatree_fill_%A.log
#SBATCH -A datamining

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent
source /etc/profile.d/modules.sh
module load miniforge
export CUBLAS_WORKSPACE_CONFIG=:4096:8
set -e
cd /srv/home/users/sidakk95cs/hyperbolic-clustering

# ============================================================================
# Fill the missing cells of the LCA-induced-tree column for Reviewer echd.
#
# Already logged: CIFAR-100 d=16 (ours ce0 + ce300, HIER, random) and
# TinyImageNet d=128 (ours ce600, HIER). Missing and added here:
#   - TinyImageNet d=16, ours ce0 and ce600, three seeds
#   - TinyImageNet d=128, ours ce0 anchor, three seeds
#   - a random floor on both TinyImageNet dimensionalities
#
# The LCA-tree evaluation is root-EXCLUDED by construction
# (scripts/eval_lca_tree.py:124), so every number it produces must be read
# against a root-excluded Ward number, never against the root-inclusive Ward
# figures in Table 2. That is exactly the comparison echd made, and it is why
# the apparent Ward-over-LCA gap on CIFAR-100 is not what it looks like.
# ============================================================================

LCA="uv run python scripts/eval_lca_tree.py --manifold poincare --c 1.0 --lca_version ours"

# --- TinyImageNet d=16 ---
for s in 42 43 44; do
  for ce in 0 600; do
    f=baselines/rebuild/ti_poin_d16_seed${s}_clustering_ce${ce}.pt
    [ -f "$f" ] && $LCA --emb_path "$f" --dataset tiny_imagenet \
        --tag lcatree_ours_ti_d16_s${s}_ce${ce} || echo "absent $f"
  done
done

# --- TinyImageNet d=128, ce0 anchors (ce600 already logged) ---
for s in 42 43 44; do
  f=baselines/ti_d128_full600/ti_d128_seed${s}_ce0.pt
  [ -f "$f" ] && $LCA --emb_path "$f" --dataset tiny_imagenet \
      --tag lcatree_ours_ti_d128_s${s}_ce0 || echo "absent $f"
done

# --- random floors, one per dimensionality ---
python - <<'PY'
import torch
for src, name, d in [
    ("baselines/rebuild/ti_poin_d16_seed42_clustering_ce600.pt", "ti_d16", 16),
    ("baselines/ti_d128_full600/ti_d128_seed42_ce600.pt", "ti_d128", 128),
]:
    x, y, i = torch.load(src, weights_only=False)
    g = torch.Generator().manual_seed(0)
    r = torch.randn(x.shape[0], x.shape[1], generator=g) * 0.01
    torch.save((r, y, i), f"/tmp/lcafloor_{name}.pt")
    print(name, tuple(r.shape))
PY

$LCA --emb_path /tmp/lcafloor_ti_d16.pt  --dataset tiny_imagenet --tag lcatree_random_ti_d16
$LCA --emb_path /tmp/lcafloor_ti_d128.pt --dataset tiny_imagenet --tag lcatree_random_ti_d128

module purge
