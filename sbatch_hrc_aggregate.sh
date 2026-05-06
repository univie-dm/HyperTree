#!/bin/bash
#SBATCH --job-name=hrc_agg
#SBATCH --time=0:10:00
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH --output=logs/hrc_agg_%A.log
#SBATCH -A datamining

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

set -e
cd /srv/home/users/sidakk95cs/hyperbolic-clustering

uv run python scripts/aggregate_hrc_baselines.py | tee baselines/hrc_reeval/summary.txt

module purge
