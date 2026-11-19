"""训练四时点因果电价预测模型，并按历史月份误差选择当月模型。"""
from pathlib import Path
import sys,json,time,hashlib
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
import xgboost as xgb
START=pd.Timestamp('2025-01-01')
PARAMS=dict(learning_rate=.04,n_estimators=400,min_child_weight=20,reg_lambda=10.,
 subsample=.9,colsample_bytree=.9,tree_method='hist',n_jobs=2,
 objective='reg:squarederror',eval_metric='rmse',random_state=20260913)
CONFIGS=[dict(id='week90_d3',window=90,depth=3,year=False),
 dict(id='year90_d3',window=90,depth=3,year=True),
 dict(id='year56_d2',window=56,depth=2,year=True),
 dict(id='year180_d3',window=180,depth=3,year=True)]

def read_prices():
    frame=pd.read_excel(ROOT/'data/附件4.xlsx')
    p=frame.iloc[:,1:].to_numpy(float)
    dates=pd.DatetimeIndex(frame.iloc[:,0])
    fixed=pd.read_excel(ROOT/'data/附件1.xlsx').iloc[:,1].to_numpy(float)
    assert p.shape==(365,144) and np.isfinite(p).all() and (p>0).all()
    assert dates.equals(pd.date_range(START,periods=365)) and fixed.shape==(144,)
    return p,dates,fixed

def features(history,day,ri,h):
    u=day*144+ri*36
    assert len(history)==u and day>=7
    k=np.arange(144);t=day+h;date=START+pd.Timedelta(days=int(t));x={}
    def add(n,v):x[n]=np.broadcast_to(np.asarray(v,float),(144,)).copy()
    add('slot',k);add('lead_slots',h*144+k-ri*36);add('issue_hour',ri*6)
    for j in [1,2]:
        add('time_sin'+str(j),np.sin(2*np.pi*j*k/144));add('time_cos'+str(j),np.cos(2*np.pi*j*k/144))
    latest=day*144+k;latest=np.where(latest<u,latest,latest-144)
    add('latest_known_same_slot',history[latest]);add('target_lag2day',history[(t-2)*144+k])
    for name,fn in [('mean',np.mean),('std',np.std),('min',np.min),('max',np.max)]:add('last_complete_day_'+name,fn(history[(day-1)*144:day*144]))
    for j in range(7):add('weekday_'+str(j),float(date.dayofweek==j))
    for lag in [7,14,21,28]:
        idx=(t-lag)*144+k
        assert np.max(idx)<u
        value=np.full(144,np.nan);ok=idx>=0;value[ok]=history[idx[ok]];add('lag'+str(lag),value)
    for w in [7,28]:
        idx=latest[None,:]-np.arange(w)[:,None]*144
        values=np.full(idx.shape,np.nan);ok=idx>=0;values[ok]=history[idx[ok]]
        add('same_slot_mean'+str(w),np.nanmean(values,0));add('same_slot_std'+str(w),np.nanstd(values,0))
        add('recent_mean'+str(w),history[-w*144:].mean());add('recent_std'+str(w),history[-w*144:].std())
    add('week_level_change',history[-1008:].mean()-history[-2016:-1008].mean() if u>=2016 else 0)
    add('latest_price',history[-1]);add('recent_6h_mean',history[-36:].mean());add('recent_6h_std',history[-36:].std())
    add('recent_6h_week_difference',np.mean(history[-36:]-history[-1044:-1008]) if u>=1044 else 0)
    add('observed_today_slots',ri*36)
    add('today_prefix_week_difference',np.mean(history[day*144:u]-history[(day-7)*144:u-1008]) if ri else 0)
    phase=2*np.pi*(date.dayofyear-1)/365.25
    for j in [1,2]:
        add('year_sin'+str(j),np.sin(j*phase));add('year_cos'+str(j),np.cos(j*phase))
    return np.column_stack(list(x.values())).astype(np.float32),list(x)

def errors(pred,p,origins_start=31,target_start=None,target_stop=365):
    records=[]
    for d in range(origins_start,365):
        for ri in range(4):
            for h in range(2):
                target=d+h
                if target>=target_stop or (target_start is not None and target<target_start):continue
                lo=ri*36 if h==0 else 0
                e=pred[d,ri,h,lo:]-p[target,lo:]
                assert np.isfinite(e).all()
                records.append(dict(origin_day=d,issue_hour=ri*6,horizon=h,target_day=target,
                    n=len(e),SSE=float(e@e),SAE=float(abs(e).sum()),SE=float(e.sum())))
    return pd.DataFrame(records)

def stats(df):
    n=int(df.n.sum())
    return dict(n=n,RMSE=float(np.sqrt(df.SSE.sum()/n)),MAE=float(df.SAE.sum()/n),bias=float(df.SE.sum()/n))

def main():
    began=time.perf_counter();p,dates,fixed=read_prices();flat=p.ravel()
    _,names=features(flat[:1008],7,0,0)
    X=np.full((365,4,2,144,len(names)),np.nan,dtype=np.float32)
    for d in range(7,365):
        for r in range(4):
            for h in range(2):X[d,r,h]=features(flat[:d*144+r*36],d,r,h)[0]
    baseline=np.full((365,4,2,144),np.nan)
    for d in range(365):
        for r in range(4):
            for h in range(2):
                value=fixed if d==0 else p[d+h-7] if d>=7 else p[d-1]
                baseline[d,r,h]=value
                if r:baseline[d,r,0,:r*36]=np.nan
    np.savez_compressed(ROOT/'data/prices.npz',price=p,dates=dates.to_numpy(),attachment1=fixed)
    predictions={};fitlog=[];feature_checks=[]
    for d in [7,14,30,31,90,181,364]:
        for r in range(4):
            u=d*144+r*36;changed=flat.copy();changed[u:]=12345.
            for h in range(2):
                assert np.array_equal(features(changed[:u],d,r,h)[0],X[d,r,h],equal_nan=True)
                feature_checks.append(dict(day=d,issue_hour=r*6,horizon=h,passed=True))
    with threadpool_limits(limits=2):
        for cfg in CONFIGS:
            tick=time.perf_counter();cols=[i for i,n in enumerate(names) if cfg['year'] or not n.startswith('year_')]
            pred=baseline.copy();clipped=0
            for fit in range(14,365,7):
                for h in range(2):
                    targets=np.arange(max(7+h,fit-cfg['window']),fit)
                    xs=[];ys=[];targetids=[];originids=[]
                    for t in targets:
                        d=int(t-h)
                        for r in range(4):
                            lo=r*36 if h==0 else 0
                            xs.append(X[d,r,h,lo:][:,cols]);ys.append(p[t,lo:])
                            targetids.extend([t]*(144-lo));originids.extend([d*144+r*36]*(144-lo))
                    xx=np.concatenate(xs);yy=np.concatenate(ys);ts=np.asarray(targetids);os=np.asarray(originids)
                    rounds=100;validation_origin=None;train_last=None
                    if len(targets)>=21:
                        val=ts>=fit-7;validation_origin=int(os[val].min())
                        train=(ts+1)*144<=validation_origin
                        assert ((ts[train]+1)*144<=validation_origin).all()
                        early=xgb.XGBRegressor(**PARAMS,max_depth=cfg['depth'],early_stopping_rounds=30)
                        early.fit(xx[train],yy[train],eval_set=[(xx[val],yy[val])],verbose=False)
                        rounds=int(early.best_iteration)+1;train_last=int(ts[train].max())
                    model=xgb.XGBRegressor(**dict(PARAMS,n_estimators=rounds),max_depth=cfg['depth']).fit(xx,yy)
                    for d in range(fit,min(fit+7,365)):
                        for r in range(4):
                            lo=r*36 if h==0 else 0
                            hat=model.predict(X[d,r,h,lo:][:,cols]).astype(float)
                            clipped+=int((hat<.0001).sum());pred[d,r,h,lo:]=np.maximum(.0001,hat)
                    fitlog.append(dict(model=cfg['id'],fit_day=fit,horizon=h,train_target_first=int(targets.min()),
                        train_target_last=int(targets.max()),train_rows=len(yy),n_estimators=rounds,
                        validation_origin_slot=validation_origin,early_train_target_last=train_last))
                if fit in [14,98,196,294,364]:print(cfg['id'],fit,'seconds',round(time.perf_counter()-tick,1),flush=True)
            predictions[cfg['id']]=pred
            np.savez_compressed(ROOT/'results'/('candidate_'+cfg['id']+'.npz'),predictions=pred)
            print('CANDIDATE COMPLETE',cfg['id'],'clipped',clipped,'seconds',time.perf_counter()-tick,flush=True)
    ledgers={name:errors(pred,p,origins_start=14) for name,pred in predictions.items()}
    selection=[];scores=[];adaptive=baseline.copy();frozen=baseline.copy();frozen_name=None
    for month in range(2,13):
        first=int((pd.Timestamp(2025,month,1)-START).days)
        stop=int((pd.Timestamp(2025,month+1,1)-START).days) if month<12 else 365
        lower=max(14,first-28);options=[]
        for name,ledger in ledgers.items():
            use=ledger[(ledger.target_day>=lower)&(ledger.target_day<first)]
            assert use.target_day.max()<first and ((use.origin_day*144+use.issue_hour*6)<first*144).all()
            score=stats(use);options.append((score['RMSE'],score['MAE'],name))
            scores.append(dict(selection_day=first,month=month,model=name,first_target=int(use.target_day.min()),last_target=int(use.target_day.max()),**score))
        chosen=sorted(options)[0][2]
        if frozen_name is None:frozen_name=chosen
        adaptive[first:stop]=predictions[chosen][first:stop]
        frozen[first:stop]=predictions[frozen_name][first:stop]
        selection.append(dict(month=month,selection_day=first,chosen=chosen,frozen=frozen_name,
            target_first=lower,target_last=first-1,deploy_first=first,deploy_last=stop-1))
    methods={'weekly_baseline':baseline,'xgb_frozen':frozen,'xgb_adaptive':adaptive}
    all_errors=[];summary=[]
    for name,pred in {**predictions,**methods}.items():
        e=errors(pred,p);e['model']=name;all_errors.append(e)
        for label,part in [('all',e),*[(f'issue_{r}',e[e.issue_hour==r]) for r in [0,6,12,18]],*[(f'horizon_{h}',e[e.horizon==h]) for h in [0,1]]]:
            summary.append(dict(model=name,group=label,**stats(part)))
        if name in methods:np.savez_compressed(ROOT/'results'/('price_'+name+'.npz'),predictions=pred)
    pd.DataFrame(fitlog).to_csv(ROOT/'qa/price_training_boundaries.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(scores).to_csv(ROOT/'results/monthly_selection_scores.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(selection).to_csv(ROOT/'results/monthly_selected_models.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(summary).to_csv(ROOT/'results/price_metrics.csv',index=False,encoding='utf-8-sig')
    pd.concat(all_errors).to_csv(ROOT/'results/price_error_ledger.csv',index=False,encoding='utf-8-sig')
    meta=dict(configs=CONFIGS,parameters=PARAMS,feature_names=names,frozen_model=frozen_name,
        feature_future_mutation_checks=feature_checks,fit_count=len(fitlog),runtime_seconds=time.perf_counter()-began,
        source_sha256=hashlib.sha256((ROOT/'data/附件4.xlsx').read_bytes()).hexdigest(),
        software=dict(numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,xgboost=xgb.__version__),
        selection_policy='monthly selection from only completed-target out-of-sample predictions in trailing 28 days; first available origin January 15',
        warmup_policy='all dispatch variants use causal baseline during January',
        retrospective_research=True)
    (ROOT/'qa/price_model.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print('ALL COMPLETE',meta['runtime_seconds'],frozen_name,flush=True)

if __name__=='__main__':main()
