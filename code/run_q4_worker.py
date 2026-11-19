"""在独立输出目录执行原第四问调度器；提前开展上周基线和论文固定候选A回放。"""
from pathlib import Path
import sys,shutil
BASE=Path(__file__).resolve().parent;sys.path[:0]=[str(BASE/'runtime'),str(BASE/'reproduction/q4/code')]
import numpy as np
P=BASE/'reproduction/q4';method=sys.argv[1]
out=P/'workers'/method
for folder in ['data','results','qa']:(out/folder).mkdir(parents=True,exist_ok=True)
shutil.copytree(P/'data',out/'data',dirs_exist_ok=True)
prices=np.load(P/'data/prices.npz');p=prices['price'];fixed=prices['attachment1']
baseline=np.full((365,4,2,144),np.nan)
for d in range(365):
    for r in range(4):
        for h in range(2):
            baseline[d,r,h]=fixed if d==0 else p[d+h-7] if d>=7 else p[d-1]
            if r:baseline[d,r,0,:r*36]=np.nan
np.savez_compressed(out/'results/price_weekly_baseline.npz',predictions=baseline)
if method=='xgb_frozen':
    pred=np.load(P/'results/candidate_week90_d3.npz')['predictions'];pred[:31]=baseline[:31]
    np.savez_compressed(out/'results/price_xgb_frozen.npz',predictions=pred)
elif method=='xgb_adaptive':shutil.copy2(P/'results/price_xgb_adaptive.npz',out/'results/price_xgb_adaptive.npz')
elif method!='weekly_baseline':raise ValueError(method)
import dispatch
dispatch.ROOT=out;dispatch.RISK_CONTEXT=None;sys.argv=[sys.argv[0],'--methods',method]
dispatch.main()
