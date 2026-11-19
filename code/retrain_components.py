"""复用提交源码的特征、拟合、前推校准与发布函数；解除对未交付旧候选快照的依赖。"""
from pathlib import Path
import sys, importlib.util, json, shutil, time
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE/'runtime'))
import numpy as np,pandas as pd,scipy,sklearn
from sklearn.linear_model import Ridge,Lasso
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
P=BASE/'reproduction';F=P/'q2/forecast_module'
spec=importlib.util.spec_from_file_location('component',F/'code/train_component_forecasts.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
for key in ['np','pd','scipy','sklearn','Ridge','Lasso','StandardScaler','threadpool_limits']:setattr(m,key,globals()[key])
data=dict(np.load(F/'data/rebuilt_prepared.npz'));names=json.loads((F/'data/rebuilt_protocol.json').read_text(encoding='utf-8'))['features']
start=time.time();selection={};final=data['baseline'].copy();fitlogs=[]
with threadpool_limits(limits=2):
    for v,var in enumerate(['load','pv']):
        scores=[]
        for alpha in m.GRID[var]:
            pred,logs=m.period(data,names,var,alpha,14,31)
            score=m.calibration_metrics(pred,data['actual'],var)
            scores.append({'alpha':alpha,**score});print(var,alpha,score,flush=True)
        chosen=min(scores,key=lambda r:r['rmse_kw'])
        selection[var]={'selected':chosen,'candidates':scores}
        pred,logs=m.period(data,names,var,chosen['alpha'],31,365)
        final[...,v]=pred;fitlogs.extend(logs)
        m.dump(F/f'metadata/reproduced_{var}_fits.json',{'fits':logs})
        print('COMPONENT COMPLETE',var,chosen['alpha'],time.time()-start,flush=True)
np.savez_compressed(F/'data/forecast_component_selected.npz',predictions=final,dates=data['dates'],units='kWh per 10-min slot',source='retrained in current audit')
for q in ['q2','q3','q4']:shutil.copy2(F/'data/forecast_component_selected.npz',P/q/'data/forecast_component_selected.npz')
rows,daily=m.formal_metrics(final,data['actual'],'reproduced_component62')
pd.DataFrame(rows).to_csv(BASE/'reproduced_forecast_metrics.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(daily).to_csv(BASE/'reproduced_forecast_daily.csv',index=False,encoding='utf-8-sig')
m.dump(BASE/'component_retraining.json',{'selection':selection,'versions':{'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},'fit_count':len(fitlogs),'all_fit_checks_passed':all(x['solver']['passed'] for x in fitlogs),'all_training_targets_strictly_past':all(x['training_targets_strictly_past'] for x in fitlogs),'elapsed_seconds':time.time()-start,'metrics':rows,'scope':'从原附件及现存建模函数重建，不曾获取或比对原始预测快照。'})
print('DONE',time.time()-start,flush=True)
