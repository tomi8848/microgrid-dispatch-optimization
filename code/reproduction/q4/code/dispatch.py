"""用已发布电价执行第四问两类调度，并汇总逐日费用与物理轨迹。"""
from pathlib import Path
import json,time,argparse
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import q2_planner as q2
from optimization import dispatch_lp as q3
from optimization.context import Context,ALPHAS
ROOT=Path(__file__).resolve().parents[1]
RISK_CONTEXT=None
METHODS=['weekly_baseline','xgb_frozen','xgb_adaptive']
SLOT_KEYS=['q0','normal','up','down','charge','discharge','emergency','unused','issue_hour','reserve_floor']

def inputs():
    frame=pd.read_csv(ROOT/'data/actual.csv')
    actual=frame[['load_kw','pv_kw']].to_numpy(float).reshape(365,144,2)/6
    forecast=np.load(ROOT/'data/forecast_component_selected.npz')['predictions']
    native=np.load(ROOT/'data/native_pv_energy.npz')['issued']
    prices=np.load(ROOT/'data/prices.npz')['price']
    net=actual[:,:,0]-actual[:,:,1];net2=forecast[:,:,:,0]-forecast[:,:,:,1]
    net3=np.empty((365,4,2,144))
    for d in range(365):
        for r in range(4):
            pv=forecast[d,:,:,1].reshape(288).copy();pv[r*36:r*36+144]=native[d,r]
            net3[d,r]=forecast[d,:,:,0]-pv.reshape(2,144)
    assert actual.shape==(365,144,2) and native.shape==(365,4,144)
    return actual,net,net2,net3,prices

def new_run(days=365):
    out={k:np.zeros((days,144)) for k in SLOT_KEYS};out['soc']=np.zeros((days,145));return out

def vector(mask):return '('+','.join(str((mask>>i)&1) for i in range(3))+')'

def q3_day(actual_net,issued_net,issued_price,day,mask,state,tau=.8):
    global RISK_CONTEXT
    if RISK_CONTEXT is None:RISK_CONTEXT=Context(ROOT)
    issued_net=issued_net.copy()
    for ri in range(4):issued_net[ri]+=RISK_CONTEXT.risk_margin(day,ri,tau)
    end=144 if day==364 else 288;certs=[]
    q0,c=q3.solve(issued_net[0].reshape(288)[:end],issued_price[0].reshape(288)[:end],state)
    certs.append(dict(issue_hour=0,**c));q0=q0[:144]
    run={k:np.zeros(144) for k in SLOT_KEYS};run['q0']=q0;run['normal']=q0.copy()
    soc=np.zeros(145);soc[0]=state
    hours=[0]+[r for i,r in enumerate([6,12,18]) if mask&(1<<i)]+[24]
    for r,nxt in zip(hours[:-1],hours[1:]):
        lo,hi=r*6,nxt*6
        if r:
            demand=issued_net[r//6].reshape(288)[lo:end].copy()
            price=issued_price[r//6].reshape(288)[lo:end].copy()
            plan,c=q3.solve_refund(demand,price,soc[lo],q0[lo:]);certs.append(dict(issue_hour=r,**c))
            run['normal'][lo:hi]=plan[:hi-lo]
        run['issue_hour'][lo:hi]=r
        for k in range(lo,hi):
            balance=run['normal'][k]-actual_net[k]
            if balance>=0:
                run['charge'][k]=min(balance,q3.FLOW_MAX,max((q3.SOC_MAX-soc[k])/.9,0))
                run['unused'][k]=balance-run['charge'][k]
            else:
                run['discharge'][k]=min(-balance,q3.FLOW_MAX,max(.9*(soc[k]-q3.SOC_MIN),0))
                run['emergency'][k]=-balance-run['discharge'][k]
            soc[k+1]=soc[k]+.9*run['charge'][k]-run['discharge'][k]/.9
    run['up']=np.maximum(run['normal']-q0,0);run['down']=np.maximum(q0-run['normal'],0)
    run['soc']=soc;run['reserve_floor'][:]=1200
    return run,certs

def q2_day(d,state,net,forecast,issued_price):
    decision=q2.evaluate_origin(d,state,net[:d],forecast[:d+1],issued_price[0,0],
        include_continuation=d<364,next_prices=issued_price[0,1])
    actual=q2.execute_actual(decision['plan'],net[d],state,issued_price[0,0],decision['reserve_strength'])
    # 模式4-2继承第二问跨日边界；state由上一日实际末库存传入。
    if abs(float(actual['soc'][0])-state)>1e-8:
        raise AssertionError(f'mode 4-2 inherited storage mismatch on day {d}')
    run={k:np.zeros(144) for k in SLOT_KEYS}
    run.update(q0=decision['plan'],normal=decision['plan'].copy(),charge=actual['charge'],
        discharge=actual['discharge'],emergency=actual['emergency'],unused=actual['unused'],
        soc=actual['soc'],reserve_floor=actual['reserve_floor'])
    return run,decision

def put(container,d,run):
    for k in container:container[k][d]=run[k]

def bill(run,p):
    return dict(plan_cost=p*run['q0'],up_cost=1.5*p*run['up'],down_cost=-.5*p*run['down'],emergency_cost=5*p*run['emergency'])

def daily_frame(run,p,method,mode,mask=-1):
    costs=bill(run,p)
    n=len(p);df=pd.DataFrame(dict(date=pd.date_range('2025-01-01',periods=n),day=np.arange(n),
        method=method,mode=mode,mask=mask,combination=vector(mask) if mask>=0 else '不调整',
        soc_start=run['soc'][:,0],soc_end=run['soc'][:,-1]))
    for k in SLOT_KEYS:
        if k not in ['issue_hour','reserve_floor']:df[k]=run[k].sum(1)
    for k,v in costs.items():df[k]=v.sum(1)
    df['total_cost']=sum(df[k] for k in costs);df['formal']=df.day>=31
    return df

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--methods',nargs='+',default=METHODS)
    args=parser.parse_args();start=time.perf_counter()
    actual,net,n2,n3,p=inputs();baseline=np.load(ROOT/'results/price_weekly_baseline.npz')['predictions']
    certificates=[];q2_audit=[];all_daily=[];q2_lp_count=0;q2_lp_max={}
    original=q2.solve_dispatch
    def checked(*a,**kw):
        nonlocal q2_lp_count
        result=original(*a,**kw);q2_lp_count+=1
        for k,v in result['verification'].items():
            if ('residual' in k or 'violation' in k or 'gap' in k or 'stationarity' in k or 'complementarity' in k) and isinstance(v,(int,float)):
                q2_lp_max[k]=max(q2_lp_max.get(k,0),v)
        return result
    q2.solve_dispatch=checked
    (ROOT/'qa/q2_candidates').mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=2):
        warm2=new_run(31);s=6000.
        for d in range(31):
            run,decision=q2_day(d,s,net,n2,baseline[d]);put(warm2,d,run);s=run['soc'][-1]
        np.savez_compressed(ROOT/'results/warmup_q2.npz',**warm2)
        warm3=[];warm_scores=[];tau_scores=[];selected_taus={}
        value=.9*float(np.load(ROOT/'data/prices.npz')['attachment1'][0])
        for mask in range(8):
            warm=new_run(31);s=6000.
            for d in range(31):
                run,cert=q3_day(net[d],n3[d],baseline[d],d,mask,s);put(warm,d,run);s=run['soc'][-1]
                certificates.extend(dict(method='common_warmup',day=d,mask=mask,**c) for c in cert)
            warm3.append(warm)
            for tau in ALPHAS:
                state=float(warm['soc'][14,0]);initial=state;cash=0.
                for d in range(14,31):
                    trial,cert=q3_day(net[d],n3[d],baseline[d],d,mask,state,tau)
                    state=float(trial['soc'][-1]);cash+=float(sum(v.sum() for v in bill(trial,p[d]).values()))
                    certificates.extend(dict(method='tau_calibration',day=d,mask=mask,**c) for c in cert)
                tau_scores.append(dict(mask=mask,combination=vector(mask),tau=tau,initial_soc=initial,final_soc=state,
                                       cash_cost=cash,selection_score=cash-value*(state-initial)))
            best=min((x for x in tau_scores if x['mask']==mask),key=lambda x:(x['selection_score'],x['tau']))
            selected_taus[mask]=best['tau']
            cost=float(sum(v.sum() for v in bill(warm,p[:31]).values()))
            warm_scores.append(dict(mask=mask,combination=vector(mask),cash_cost=cost,initial_soc=6000.,final_soc=float(s),
                inventory_value=value,selection_score=cost-value*(s-6000),selection_day=31,last_observed_day=30))
        chosen=sorted(warm_scores,key=lambda row:(row['selection_score'],row['cash_cost'],row['mask']))[0]['mask']
        pd.DataFrame(warm_scores).to_csv(ROOT/'results/q3_january_selection.csv',index=False,encoding='utf-8-sig')
        for row in tau_scores:row['selected']=row['tau']==selected_taus[row['mask']]
        pd.DataFrame(tau_scores).to_csv(ROOT/'results/q3_tau_calibration.csv',index=False,encoding='utf-8-sig')
        print('WARMUP COMPLETE, selected Q3',vector(chosen),'q2 SOC',warm2['soc'][-1,-1],flush=True)
        for method in args.methods:
            clock=time.perf_counter();price=np.load(ROOT/'results'/('price_'+method+'.npz'))['predictions']
            assert np.array_equal(price[:31],baseline[:31],equal_nan=True)
            out2=new_run()
            for k in out2:out2[k][:31]=warm2[k]
            s=float(warm2['soc'][-1,-1])
            for d in range(31,365):
                run,e=q2_day(d,s,net,n2,price[d]);put(out2,d,run);s=float(run['soc'][-1])
                cert=e.pop('certificate')
                if method=='xgb_adaptive':np.savez_compressed(ROOT/'qa/q2_candidates'/f'day_{d:03d}.npz',**cert)
                q2_audit.append(dict(method=method,day=d,reserve_strength=e['reserve_strength'],
                    candidate_count=e['candidate_count'],primary_minimum=e['primary_minimum_yuan'],
                    selected_primary_gap=e['selected_primary_gap_yuan'],primary_tie_count=e['primary_tie_count'],
                    selected_candidate=e['selected_candidate'],secondary_changed=e['secondary_changed_candidate'],
                    **e['tree_audit']))
                if d%50==0 or d==364:print(method,'Q2',d,'seconds',round(time.perf_counter()-clock,1),flush=True)
            np.savez_compressed(ROOT/'results'/('dispatch_q2_'+method+'.npz'),**out2)
            all_daily.append(daily_frame(out2,p,method,'4-2'))
            for mask in range(8):
                out3=new_run()
                for k in out3:out3[k][:31]=warm3[mask][k]
                s=float(out3['soc'][30,-1])
                for d in range(31,365):
                    run,cert=q3_day(net[d],n3[d],price[d],d,mask,s,selected_taus[mask]);put(out3,d,run);s=float(run['soc'][-1])
                    certificates.extend(dict(method=method,day=d,mask=mask,**c) for c in cert)
                np.savez_compressed(ROOT/'results'/f'dispatch_q3_{method}_mask{mask}.npz',**out3)
                all_daily.append(daily_frame(out3,p,method,'4-3',mask))
                print(method,'Q3',vector(mask),'seconds',round(time.perf_counter()-clock,1),flush=True)
            pd.concat(all_daily).to_csv(ROOT/'results/dispatch_daily.csv',index=False,encoding='utf-8-sig')
    df=pd.concat(all_daily);rows=[]
    for (method,mode,mask),group in df.groupby(['method','mode','mask']):
        formal=group[group.formal]
        rows.append(dict(method=method,mode=mode,mask=int(mask),combination=vector(int(mask)) if mask>=0 else '不调整',
            deployed=(mode=='4-2' or mask==chosen),formal_cost=float(formal.total_cost.sum()),
            annual_cost=float(group.total_cost.sum()),plan_cost=float(formal.plan_cost.sum()),
            adjustment_cost=float((formal.up_cost+formal.down_cost).sum()),emergency_cost=float(formal.emergency_cost.sum()),
            emergency_kwh=float(formal.emergency.sum()),feb1_soc=float(formal.soc_start.iloc[0]),final_soc=float(formal.soc_end.iloc[-1])))
    pd.DataFrame(rows).to_csv(ROOT/'results/dispatch_summary.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(q2_audit).to_csv(ROOT/'qa/q2_decisions.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(certificates).to_csv(ROOT/'qa/q3_lp_certificates.csv',index=False,encoding='utf-8-sig')
    meta=dict(deployed_q3_mask=chosen,deployed_q3_combination=vector(chosen),january_inventory_coefficient=value,
        q3_tau_by_mask=selected_taus,q3_tau_grid=list(ALPHAS),settlement='half_refund',
        q2_lp_count=q2_lp_count,q2_lp_max_errors=q2_lp_max,q3_lp_count=len(certificates),
        methods=args.methods,runtime_seconds=time.perf_counter()-start,
        rho_grid=q2.RESERVE_STRENGTHS.tolist(),primary_tie_atol=q2.PRIMARY_TIE_ATOL,
        future_real_price_used_for_decisions=False,continuous_soc=True)
    (ROOT/'qa/dispatch_run.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print('ALL DISPATCH COMPLETE',meta,flush=True)

if __name__=='__main__':main()
