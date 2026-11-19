from pathlib import Path
import sys,shutil,json
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE/'runtime'))
import numpy as np,pandas as pd
P=BASE/'reproduction/q4';methods=['weekly_baseline','xgb_frozen','xgb_adaptive'];meta=[]
for method in methods:
    out=P/'workers'/method
    expected=np.load(P/'results'/f'price_{method}.npz')['predictions']
    used=np.load(out/'results'/f'price_{method}.npz')['predictions']
    assert np.array_equal(expected,used,equal_nan=True),(method,'提前回放的价格方案与完整重新选模结果不一致')
    meta.append(json.loads((out/'qa/dispatch_run.json').read_text(encoding='utf-8')))
    for f in (out/'results').glob('dispatch_*.npz'):shutil.copy2(f,P/'results'/f.name)
    if method=='xgb_adaptive':shutil.copytree(out/'qa/q2_candidates',P/'qa/q2_candidates',dirs_exist_ok=True)
for key in ['deployed_q3_mask','january_inventory_coefficient','q3_tau_by_mask','rho_grid']:
    assert all(x[key]==meta[0][key] for x in meta),key
for filename in ['q3_january_selection.csv','q3_tau_calibration.csv']:
    original=(P/'workers'/methods[0]/'results'/filename).read_bytes()
    assert all((P/'workers'/method/'results'/filename).read_bytes()==original for method in methods)
    (P/'results'/filename).write_bytes(original)
for folder,name in [('results','dispatch_daily.csv'),('results','dispatch_summary.csv'),('qa','q2_decisions.csv'),('qa','q3_lp_certificates.csv')]:
    pd.concat([pd.read_csv(P/'workers'/method/folder/name) for method in methods],ignore_index=True).to_csv(P/folder/name,index=False,encoding='utf-8-sig')
merged=meta[0].copy();merged.update(methods=methods,q2_lp_count=sum(x['q2_lp_count'] for x in meta),q3_lp_count=sum(x['q3_lp_count'] for x in meta),runtime_seconds=max(x['runtime_seconds'] for x in meta),execution='Three independent subprocesses; prices checked bitwise against complete refit outputs.')
merged['q2_lp_max_errors']={k:max(x['q2_lp_max_errors'][k] for x in meta) for k in meta[0]['q2_lp_max_errors']}
(P/'qa/dispatch_run.json').write_text(json.dumps(merged,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(merged,ensure_ascii=False,indent=2))
