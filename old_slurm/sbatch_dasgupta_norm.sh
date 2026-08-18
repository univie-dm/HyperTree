#!/bin/bash
#SBATCH --job-name=dasg_norm
#SBATCH --time=2:00:00
#SBATCH -c 16
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/dasg_norm_%A.log
#SBATCH -A datamining
export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent
source /etc/profile.d/modules.sh
module load miniforge
cd /srv/home/users/sidakk95cs/hyperbolic-clustering
uv run python - <<'PY'
import json, torch, sys
from pathlib import Path
sys.path.insert(0,'.')
from src.evaluation import dasgupta_cost
from src.hyperbolic_math.src.manifolds import PoincareBall
from src.hyperbolic_math.src.utils.helpers import compute_pairwise_distances
arms={'joint':'baselines/c100_d16_full300/c100_d16_seed{s}_clustering_ce300.pt',
      'conly':'baselines/rebuild/c100_conly_d16_seed{s}_clustering_ce300.pt',
      'lhonly':'baselines/rebuild/c100_lhonly_d16_seed{s}_clustering_ce300.pt',
      'anchor':'baselines/c100_d16_full300/c100_d16_seed{s}_clustering_ce0.pt'}
dev='cuda' if torch.cuda.is_available() else 'cpu'
out={}
for arm,tpl in arms.items():
    for s in (1,2,3):
        p=tpl.format(s=s)
        if not Path(p).exists(): continue
        x,y,_=torch.load(p,map_location='cpu',weights_only=False); x=x.float().to(dev)
        n=x.shape[0]
        man=PoincareBall(c=torch.tensor(1.0,device=dev,dtype=x.dtype),dtype=x.dtype)
        d=compute_pairwise_distances(x,man); w=torch.reciprocal(d+1)
        iu=torch.triu_indices(n,n,offset=1,device=dev)
        sw=w[iu[0],iu[1]].sum().item()
        raw=dasgupta_cost(w,'average')
        # normalized: w-weighted mean LCA size, as a fraction of n
        norm=raw/(sw*n)
        out.setdefault(arm,[]).append({'seed':s,'raw':raw,'sum_w':sw,'norm':norm})
        print(f'{arm:7s} s{s} raw={raw:.4g} sum_w={sw:.4g} norm={norm:.6f}',flush=True)
Path('results_baseline_embeddings/abl_dasgupta_normalized.json').write_text(json.dumps(out,indent=1))
print('[write] abl_dasgupta_normalized.json')
PY
module purge
