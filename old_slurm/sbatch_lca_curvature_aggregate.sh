#!/bin/bash
#SBATCH --job-name=lca_curv_agg
#SBATCH --time=2:00:00
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/lca_curv_agg_%j.log
#SBATCH -A datamining

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

# Aggregate LCA wrong-rate per curvature for the §4.2.3 curvature sweep.
# Walks baselines/lca_curvature_sweep/c{C}/ and runs aggregate_lca_wrong_rate
# with the matching --c on each subdirectory. Writes per-curvature
# summary.json + a combined summary at baselines/lca_curvature_sweep/summary.json

ROOT=/srv/home/users/sidakk95cs/hyperbolic-clustering/baselines/lca_curvature_sweep
mkdir -p ${ROOT}

for CDIR in ${ROOT}/c*/; do
    [ -d "${CDIR}" ] || continue
    CURV=$(basename ${CDIR} | sed 's/^c//')
    echo ""
    echo "================================================================"
    echo "=== curvature c=${CURV}  dir=${CDIR}"
    echo "================================================================"
    uv run python -m scripts.aggregate_lca_wrong_rate \
        --dump_dir ${CDIR} \
        --c ${CURV} \
        --out_json ${CDIR}/summary.json
done

# Combine all per-curvature summaries into one
uv run python -c "
import json
from pathlib import Path
root = Path('${ROOT}')
combined = {'curvatures': {}}
for sub in sorted(root.glob('c*/summary.json')):
    curv = sub.parent.name[1:]
    combined['curvatures'][curv] = json.loads(sub.read_text())
out = root / 'summary.json'
out.write_text(json.dumps(combined, indent=2))
print(f'[wrote] {out}')

# Print a compact table
print()
print('=== Combined wrong-rate table ===')
print(f'{\"c\":>10}  {\"trajectory mean ± std\":>30}  {\"ce=0\":>15}  {\"ce=final\":>15}')
for curv in sorted(combined['curvatures'], key=float):
    s = combined['curvatures'][curv]
    if 'datasets' not in s or not s['datasets']:
        continue
    ds = next(iter(s['datasets'].values()))
    print(f'{curv:>10}  {ds[\"trajectory_mean_pct\"]:>15.6f} ± {ds[\"trajectory_std_pct\"]:>10.6f}  {ds[\"ce0_mean_pct\"]:>15.6f}  {ds[\"cefinal_mean_pct\"]:>15.6f}')
"

module purge
