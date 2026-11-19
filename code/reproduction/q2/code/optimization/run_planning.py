"""逐日执行问题二双日情景规划，并连续传递实际储能状态。
正式统计区间为 2 月 1 日至 12 月 31 日。"""
from pathlib import Path
import hashlib,json,platform,time
import numpy as np
import pandas as pd
import scipy
from optimization import planner as m
ROOT=Path(__file__).resolve().parents[2]
DATA,RESULTS,QA=ROOT/'data',ROOT/'results',ROOT/'qa'

def save_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def inputs():
    raw=pd.read_csv(DATA/'actual.csv')
    actual=raw[['load_kw','pv_kw']].to_numpy(float).reshape(365,144,2)/6
    dates=pd.to_datetime(raw.sample_date).dt.strftime('%Y-%m-%d').drop_duplicates().to_numpy()
    price=pd.read_csv(DATA/'price.csv').price_yuan_per_kwh.to_numpy(float)
    with np.load(DATA/'forecast_component_selected.npz',allow_pickle=False) as f:pred=f['predictions'].copy()
    with np.load(DATA/'baseline.npz',allow_pickle=False) as f:baseline=f['predictions'].copy()
    assert actual.shape==(365,144,2) and pred.shape==(365,2,144,2)
    assert np.array_equal(pred[:31],baseline[:31])
    return actual,pred,price,dates

def main():
    for path in [RESULTS,QA/'decisions']:path.mkdir(parents=True,exist_ok=True)
    actual,pred,prices,dates=inputs()
    net=actual[:,:,0]-actual[:,:,1];prednet=pred[:,:,:,0]-pred[:,:,:,1]
    state=6000.;daily=[];frames=[];audits=[];candidates=[];clock=time.perf_counter()
    starts=[f'{k//6:02d}:{k%6*10:02d}' for k in range(144)]
    ends=[f'{(k+1)//6:02d}:{(k+1)%6*10:02d}' for k in range(144)]
    for d in range(365):
        begin=time.perf_counter()
        # 跨日硬约束 S[d-1,144] = S[d,0]；不改变原有状态传递或候选规则。
        if d and abs(state-daily[-1]['soc_end_kwh'])>1e-8:
            raise AssertionError(f'cross-day storage discontinuity before day {d}')
        e=m.evaluate_origin(d,state,net[:d],prednet[:d+1],prices,
                            include_continuation=d<364,terminal_value=0.)
        q=e['plan'];run=m.execute_actual(q,net[d],state,prices,e['reserve_strength'])
        if abs(float(run['soc'][0])-state)>1e-8:
            raise AssertionError(f'execution initial state differs from inherited storage on day {d}')
        cert=e.pop('certificate')
        np.savez_compressed(QA/'decisions'/f'{dates[d]}.npz',**cert)
        candidates.append(pd.DataFrame(dict(sample_date=dates[d],candidate=np.arange(len(cert['primary_costs'])),
            normal_cost_yuan=cert['normal_costs'],expected_emergency_yuan=cert['emergency_costs']@cert['probabilities'],
            primary_cost_yuan=cert['primary_costs'],eligible=np.isin(np.arange(len(cert['primary_costs'])),cert['eligible']),
            secondary_cost_yuan=np.where(np.isfinite(cert['secondary_costs']),cert['secondary_costs'],np.nan),
            selected=np.arange(len(cert['primary_costs']))==int(cert['selected']),reserve_strength=cert['reserve_strengths'])))
        normal=float(q@prices);emergency=float(run['emergency']@(5*prices))
        daily.append(dict(sample_date=dates[d],origin_day0=d,period='warmup' if d<31 else 'formal',
            load_kwh=float(actual[d,:,0].sum()),pv_kwh=float(actual[d,:,1].sum()),planned_kwh=float(q.sum()),
            charge_kwh=float(run['charge'].sum()),discharge_kwh=float(run['discharge'].sum()),
            emergency_kwh=float(run['emergency'].sum()),unused_kwh=float(run['unused'].sum()),
            soc_start_kwh=state,soc_end_kwh=float(run['soc'][-1]),normal_cost_yuan=normal,
            emergency_cost_yuan=emergency,total_cost_yuan=normal+emergency,
            reserve_strength=e['reserve_strength'],scenario_pairs=e['tree_audit']['pair_count'],
            scenario_count=e['branch_count'],candidate_count=e['candidate_count'],
            expected_daily_cost_yuan=e['objective_yuan'],expected_emergency_yuan=e['expected_day1_emergency_yuan'],
            secondary_next_day_cost_yuan=e['expected_continuation_yuan'],primary_minimum_yuan=e['primary_minimum_yuan'],
            selected_primary_gap_yuan=e['selected_primary_gap_yuan'],primary_tie_count=e['primary_tie_count'],
            secondary_changed_candidate=e['secondary_changed_candidate'],secondary_changed_plan=e['secondary_changed_plan'],
            secondary_changed_feedback=e['secondary_changed_feedback'],planning_seconds=time.perf_counter()-begin))
        frame=pd.DataFrame(dict(sample_date=dates[d],slot=np.arange(1,145),interval_start=starts,interval_end=ends,
            price_yuan_per_kwh=prices,actual_load_kwh=actual[d,:,0],actual_pv_kwh=actual[d,:,1],actual_net_kwh=net[d],
            forecast_net_kwh=prednet[d,0],planned_kwh=q,charge_kwh=run['charge'],discharge_kwh=run['discharge'],
            emergency_kwh=run['emergency'],unused_kwh=run['unused'],soc_start_kwh=run['soc'][:-1],soc_end_kwh=run['soc'][1:],
            reserve_floor_kwh=run['reserve_floor'],reserve_strength=e['reserve_strength'],
            normal_cost_yuan=prices*q,emergency_cost_yuan=5*prices*run['emergency']))
        frame['total_cost_yuan']=frame.normal_cost_yuan+frame.emergency_cost_yuan
        frames.append(frame)
        record={k:v for k,v in e.items() if k not in ['branches','plan']}
        record.update(sample_date=dates[d],**e['tree_audit'])
        audits.append(record)
        state=float(run['soc'][-1])
        if d%20==0 or d in [30,31,364]:
            print(f'{d+1}/365 {dates[d]} bill={normal+emergency:.2f} soc={state:.3f} ties={e["primary_tie_count"]} time={time.perf_counter()-begin:.2f}',flush=True)
    yd=pd.DataFrame(daily);ys=pd.concat(frames,ignore_index=True);cc=pd.concat(candidates,ignore_index=True)
    continuity=yd.soc_start_kwh.to_numpy()[1:]-yd.soc_end_kwh.to_numpy()[:-1]
    if np.max(np.abs(continuity))>1e-8:
        raise AssertionError('S[d,144] = S[d+1,0] violated')
    save_json(QA/'cross_day_continuity.json',dict(constraint='S[d,144] = S[d+1,0]',boundaries=len(continuity),maximum_error_kwh=float(np.max(np.abs(continuity))),passed=True))
    fd=yd.iloc[31:].reset_index(drop=True);fs=ys.iloc[31*144:].reset_index(drop=True)
    for name,frame in [('all_year_daily',yd),('all_year_dispatch',ys),('daily',fd),('dispatch',fs),('candidate_objectives',cc)]:
        frame.to_csv(RESULTS/(name+'.csv'),index=False,encoding='utf-8-sig',float_format='%.15g')
    plan=ys.pivot(index='sample_date',columns='slot',values='planned_kwh')
    plan.columns=[f'slot_{k:03d}_kwh' for k in plan.columns]
    plan.to_csv(RESULTS/'all_year_plans_144.csv',encoding='utf-8-sig',float_format='%.15g')
    plan.iloc[31:].to_csv(RESULTS/'plans_144.csv',encoding='utf-8-sig',float_format='%.15g')
    sums=['load_kwh','pv_kwh','planned_kwh','charge_kwh','discharge_kwh','emergency_kwh','unused_kwh','normal_cost_yuan','emergency_cost_yuan','total_cost_yuan']
    totals={k:float(fd[k].sum()) for k in sums}
    totals.update(days=len(fd),intervals=len(fs),initial_soc_kwh=float(fd.soc_start_kwh.iloc[0]),final_soc_kwh=state,
        emergency_days=int((fd.emergency_kwh>1e-7).sum()),emergency_slots=int((fs.emergency_kwh>1e-7).sum()),
        numerical_tie_days=int((fd.primary_tie_count>1).sum()),secondary_changed_candidate_days=int(fd.secondary_changed_candidate.sum()),
        secondary_changed_plan_days=int(fd.secondary_changed_plan.sum()),secondary_changed_feedback_days=int(fd.secondary_changed_feedback.sum()),
        maximum_primary_gap_yuan=float(fd.selected_primary_gap_yuan.max()))
    summary=dict(totals=totals,warmup_initial_soc=6000.,warmup_final_soc=float(yd.soc_end_kwh.iloc[30]),
        parameters=dict(first_branches=4,conditional_children=3,lookback_pairs=90,eta=.9,soc_min=1200.,soc_max=10800.,
            flow_max_kwh=5000/6,emergency_multiplier=5,feedback_strengths=m.RESERVE_STRENGTHS.tolist(),
            primary_tie_atol_yuan=m.PRIMARY_TIE_ATOL,terminal_value=0),
        runtime_seconds=time.perf_counter()-clock,software=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__),
        input_sha256={f:hashlib.sha256((DATA/f).read_bytes()).hexdigest() for f in ['actual.csv','price.csv','forecast_component_selected.npz','baseline.npz']},
        objective='lexicographic: first minimize current-day expected bill; then next-day expected bill only among numerical current-day minimizers',
        scope='minimum within finite generated plans and causal reserve-feedback family under empirical scenarios; realized bills are chronological replay, not hindsight or global stochastic optima')
    save_json(QA/'planning_summary.json',summary);save_json(QA/'scenario_audit.json',audits)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
