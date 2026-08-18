#!/bin/bash
#SBATCH --job-name=eval_abl_aligned
#SBATCH --time=6:00:00
#SBATCH -c 16
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/eval_abl_aligned_%A.log
#SBATCH -A datamining
#SBATCH -x vader
export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent
source /etc/profile.d/modules.sh
module load miniforge
set -e
cd /srv/home/users/sidakk95cs/hyperbolic-clustering

# ============================================================================
# Score the loss-ablation triad on metrics ALIGNED with what L_H optimizes.
#
# Every comparison so far used Ward-linkage hDP, which L_H does not optimize:
# it builds a dendrogram by agglomerating the embedding, independent of the
# LCA geometry. L_H optimizes a relaxed Dasgupta cost defined through the
# rectified hyperbolic LCA. So "L_C-only beats joint on Ward-hDP" may simply
# mean Ward-hDP is blind to what L_H does.
#
# Two aligned measurements:
#   1. hDP on the LCA-INDUCED tree (scripts/eval_lca_tree.py) -- the hierarchy
#      decoded from the rectified LCA depths, i.e. the structure L_H shapes.
#   2. Dasgupta cost -- the discrete objective L_H is a relaxation of.
#
# If the joint arm wins on these while losing on Ward-hDP, that is a real and
# reportable finding: L_H shapes the geometry's own hierarchy rather than the
# agglomerative clustering of the embedding.
#
# All arms share one encoder and an identical ce=0 anchor.
# ============================================================================

LCA="uv run python scripts/eval_lca_tree.py --manifold poincare --c 1.0 --dataset cifar100"

for s in 1 2 3; do
  echo "=== LCA-tree: joint seed${s} ==="
  f=baselines/c100_d16_full300/c100_d16_seed${s}_clustering_ce300.pt
  [ -f "$f" ] && ${LCA} --emb_path "$f" --tag lcatree_abl_joint_s${s} || echo "skip $f"

  echo "=== LCA-tree: contrastive-only seed${s} ==="
  f=baselines/rebuild/c100_conly_d16_seed${s}_clustering_ce300.pt
  [ -f "$f" ] && ${LCA} --emb_path "$f" --tag lcatree_abl_conly_s${s} || echo "skip $f"

  echo "=== LCA-tree: L_H-only seed${s} ==="
  f=baselines/rebuild/c100_lhonly_d16_seed${s}_clustering_ce300.pt
  [ -f "$f" ] && ${LCA} --emb_path "$f" --tag lcatree_abl_lhonly_s${s} || echo "skip $f"
done

# Dasgupta cost (the discrete objective L_H relaxes), same arms.
echo "=== Dasgupta cost, all arms ==="
uv run python - <<'PY'
import json, torch, glob
from pathlib import Path
import sys
sys.path.insert(0, ".")
from src.evaluation import dasgupta_cost
from src.hyperbolic_math.src.manifolds import PoincareBall
from src.hyperbolic_math.src.utils.helpers import compute_pairwise_distances

arms = {
    "joint":  "baselines/c100_d16_full300/c100_d16_seed{s}_clustering_ce300.pt",
    "conly":  "baselines/rebuild/c100_conly_d16_seed{s}_clustering_ce300.pt",
    "lhonly": "baselines/rebuild/c100_lhonly_d16_seed{s}_clustering_ce300.pt",
    "anchor": "baselines/c100_d16_full300/c100_d16_seed{s}_clustering_ce0.pt",
}
dev = "cuda" if torch.cuda.is_available() else "cpu"
out = {}
for arm, tpl in arms.items():
    for s in (1, 2, 3):
        p = tpl.format(s=s)
        if not Path(p).exists():
            print(f"skip {p}"); continue
        x, y, _ = torch.load(p, map_location="cpu", weights_only=False)
        x = x.float().to(dev)
        man = PoincareBall(c=torch.tensor(1.0, device=dev, dtype=x.dtype), dtype=x.dtype)
        d = compute_pairwise_distances(x, man)
        cost = dasgupta_cost(torch.reciprocal(d + 1), "average")
        out.setdefault(arm, []).append(float(cost))
        print(f"{arm:8s} seed{s}  dasgupta={cost:.6f}", flush=True)
Path("results_baseline_embeddings").mkdir(exist_ok=True)
Path("results_baseline_embeddings/abl_dasgupta.json").write_text(json.dumps(out, indent=1))
print("[write] results_baseline_embeddings/abl_dasgupta.json")
PY

module purge
