"""从原附件重建缺失的接口；所有输出是本次重建，不冒充原预测快照。"""
from pathlib import Path
import sys, shutil, importlib.util, json, hashlib
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'runtime'))
import numpy as np
import pandas as pd
from openpyxl import load_workbook
P=BASE/'reproduction'

def read(path,sheet=0):
    wb=load_workbook(path,read_only=True,data_only=True)
    ws=wb.worksheets[sheet] if isinstance(sheet,int) else wb[sheet]
    rows=list(ws.values);wb.close();return rows

def main():
    raw=P/'C题/附件'
    shutil.copytree(raw,P/'附件',dirs_exist_ok=True)
    load=read(raw/'附件2.xlsx','小区负载');pv=read(raw/'附件2.xlsx','光伏发电实际功率')
    dates=pd.DatetimeIndex([r[0] for r in load[1:]])
    assert dates.equals(pd.date_range('2025-01-01',periods=365))
    assert [r[0] for r in load[1:]]==[r[0] for r in pv[1:]]
    actual=np.stack([np.array([r[1:145] for r in x[1:]],float) for x in [load,pv]],axis=-1)
    assert actual.shape==(365,144,2) and np.isfinite(actual).all() and (actual>=0).all()
    frame=pd.DataFrame({'sample_date':np.repeat(dates.strftime('%Y-%m-%d'),144),'slot':np.tile(np.arange(1,145),365),'load_kw':actual[:,:,0].ravel(),'pv_kw':actual[:,:,1].ravel()})
    tariff=np.array([r[1] for r in read(raw/'附件1.xlsx')[1:]],float)
    price=pd.DataFrame({'slot':np.arange(1,145),'price_yuan_per_kwh':tariff})
    for q in ['q2','q3','q4']:
        for sub in ['data','results','qa','reference']:(P/q/sub).mkdir(parents=True,exist_ok=True)
        frame.to_csv(P/q/'data/actual.csv',index=False,encoding='utf-8-sig')
        price.to_csv(P/q/'data/price.csv',index=False,encoding='utf-8-sig')
        for f in raw.glob('附件*.xlsx'):
            shutil.copy2(f,P/q/'reference'/f.name)
            if q=='q4':shutil.copy2(f,P/q/'data'/f.name)
    spec=importlib.util.spec_from_file_location('forecast_source',P/'q2/forecast_module/code/train_component_forecasts.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.np=np;m.pd=pd
    energy=actual/6
    baseline=np.array([m.historical_baseline(energy[:d]) for d in range(365)])
    features=np.zeros((365,2,144,66),dtype=np.float32)
    for d in range(7,365):features[d],names=m.features(energy[:d],dates[d])
    fm=P/'q2/forecast_module';(fm/'data/source').mkdir(parents=True,exist_ok=True)
    frame.to_csv(fm/'data/source/actual.csv',index=False,encoding='utf-8-sig')
    np.savez_compressed(fm/'data/rebuilt_prepared.npz',actual=energy,dates=dates.to_numpy().astype('datetime64[D]'),baseline=baseline,features=features)
    (fm/'data/rebuilt_protocol.json').write_text(json.dumps({'features':names,'provenance':'Rebuilt in current audit from attachments, not original source snapshot.'},ensure_ascii=False,indent=2),encoding='utf-8')
    for q in ['q2','q3','q4']:
        np.savez_compressed(P/q/'data/baseline.npz',predictions=baseline)
    rows=read(raw/'附件3.xlsx');issued=np.empty((365,4,144));forecasts=np.array([r[2:26] for r in rows[1:]],float).reshape(365,4,24)
    flat=actual[:,:,1].ravel()
    for d in range(365):
        for ri in range(4):
            u=d*144+ri*36;p0=flat[u-1] if u else 0.
            nodes=np.r_[p0,forecasts[d,ri]];j=np.arange(6)
            issued[d,ri]=(((11-2*j)[None,:]*nodes[:-1,None]+(1+2*j)[None,:]*nodes[1:,None])/72).ravel()
    for q in ['q3','q4']:
        np.savez_compressed(P/q/'data/native_pv_energy.npz',issued=issued,deployed_method=np.full(365,'linear'),source='current audit analytical reconstruction')
    metrics={}
    for name,start,end in [('calibration',0,14*144),('formal',31*144,365*144)]:
        err=[]
        for d in range(365):
            for r in range(4):
                u=d*144+r*36;ids=u+np.arange(144);ok=(ids>=start)&(ids<end)
                err.extend((issued[d,r,ok]*6-flat[ids[ok]]).tolist())
        e=np.asarray(err);metrics[name]={'n':len(e),'rmse_kw':float(np.sqrt(np.mean(e*e))),'mae_kw':float(np.mean(abs(e))),'bias_kw':float(e.mean())}
    meta={'raw_shapes':{'actual':list(actual.shape),'native':list(issued.shape)},'negative_count':int((actual<0).sum()),'nonfinite_count':int((~np.isfinite(actual)).sum()),'linear_interpolation_metrics':metrics,'source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in raw.glob('附件*.xlsx')},'note':'原始源快照未随代码包交付；当前接口由附件和现存函数重建。'}
    (BASE/'input_reconstruction.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(meta,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
