#!/bin/bash
#SBATCH --job-name=c100_root
#SBATCH --time=00:30:00
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --gres=gpu:0
#SBATCH --output=logs/cifar_useroot_probe_%j.log
#SBATCH -A datamining

source /etc/profile.d/modules.sh
module load miniforge

CKPT=baselines/HIER-CVPR23/logs/CIFAR100/resnet50_seed1_emb128_eval_95.pt

echo "=== use_root=True ==="
uv run python scripts/eval_baseline_embeddings.py \
    --emb_path "$CKPT" \
    --manifold poincare --c 0.1 --dataset cifar100 \
    --tag hier_c100_s1_d128_root_T --use_root True \
    --out_dir /tmp/cifar_useroot_probe 2>&1 | tail -50

echo "=== use_root=False ==="
uv run python scripts/eval_baseline_embeddings.py \
    --emb_path "$CKPT" \
    --manifold poincare --c 0.1 --dataset cifar100 \
    --tag hier_c100_s1_d128_root_F --use_root False \
    --out_dir /tmp/cifar_useroot_probe 2>&1 | tail -50

echo "=== summary ==="
for f in /tmp/cifar_useroot_probe/hier_c100_s1_d128_root_T.json /tmp/cifar_useroot_probe/hier_c100_s1_d128_root_F.json; do
    if [ -f "$f" ]; then
        printf "%s\n" "$f"
        uv run python -c "import json; r=json.load(open('$f')); print('  hDP_ward:', r.get('test/hierarchical_dendrogram_purity_ward')); print('  hDP_complete:', r.get('test/hierarchical_dendrogram_purity_complete')); print('  flat_DP_ward:', r.get('test/dendrogram_purity_ward'))"
    fi
done

module purge
