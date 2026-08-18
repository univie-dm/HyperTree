#!/bin/bash
#ASC --vanilla
#SBATCH --job-name "inat18_frozen_matched_dim"
#SBATCH --gres=gpu:1
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=01:00:00
#SBATCH --error=logs/inat18_frozen_matched_dim_%j.err
#SBATCH --output=logs/inat18_frozen_matched_dim_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

# ============================================================================
# iNat18 frozen-DINO matched-dim ceiling — closes two gaps in RESULTS.md:
#
# (a) The current frozen-DINO row has NO hDP_ward number (only flat DP/NMI),
#     so our trained c=0.3 hDP_ward 0.452 ± 0.106 has no real reference. We
#     have data/inat18_taxonomy_hierarchy.txt; eval_baseline_embeddings.py
#     supports it — just re-run eval on the existing 2048-D .pt.
#
# (b) The frozen 2048-D vs trained d=128 comparison is dim-unfair. PCA-reduce
#     to {32, 128} and re-eval to get apples-to-apples numbers.
#
# Reuses the .pt file from job 1146716/1146741/1148527/1149095 (any of the
# COMPLETED inat18_frozen_baseline runs). Pure CPU work — eval_baseline_
# embeddings.py + sklearn PCA. ~30 min total.
# ============================================================================

OUT_2048="baselines/frozen_dino/inat18_val_dino_r50.pt"
HIER="./data/inat18_taxonomy_hierarchy.txt"

if [ ! -f "${OUT_2048}" ]; then
    echo "ERROR: ${OUT_2048} not found. Re-run sbatch_vsc_inat18_frozen_baseline.sh first."
    exit 1
fi
if [ ! -f "${HIER}" ]; then
    echo "ERROR: ${HIER} not found. Run scripts/build_inat18_taxonomy.py first."
    exit 1
fi

echo "=== [1/3] re-eval frozen 2048-D with iNat hierarchy (gives hDP_ward) ==="
uv run python scripts/eval_baseline_embeddings.py \
    --emb_path ${OUT_2048} \
    --manifold euclidean --c 0.0 \
    --dataset inat18 \
    --wordnet_path ${HIER} \
    --tag frozen_dino_inat18_val_2048d_hier

for D in 32 128; do
    OUT_PCA="baselines/frozen_dino/inat18_val_dino_r50_pca${D}d.pt"
    echo "=== [2/3] PCA reduce iNat 2048 -> ${D} ==="
    uv run python scripts/pca_reduce_embeddings.py \
        --in_path ${OUT_2048} \
        --out_path ${OUT_PCA} \
        --dim ${D}

    echo "=== [3/3] eval matched-dim d=${D} ==="
    uv run python scripts/eval_baseline_embeddings.py \
        --emb_path ${OUT_PCA} \
        --manifold euclidean --c 0.0 \
        --dataset inat18 \
        --wordnet_path ${HIER} \
        --tag frozen_dino_inat18_val_pca${D}
done
