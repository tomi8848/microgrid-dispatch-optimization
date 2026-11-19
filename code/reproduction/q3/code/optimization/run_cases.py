"""复算第三问八种全年更新策略及同状态单日反事实结果。"""
from pathlib import Path
import json,hashlib,time,platform
import numpy as np
import pandas as pd
import scipy
from optimization.context import Context,ROOT,ALPHAS
from optimization.simulation import simulate_day,daily_summary,case_id


def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def summarize_certificates(certs):
    return {k:max([c[k] for c in certs],default=0.) for k in ['balance_max','lower_violation','upper_violation','duality_gap']}


def make_frame(ctx,d,mask,alpha,run):
    df=pd.DataFrame(run);df.insert(0,'sample_date',ctx.dates[d]);df.insert(1,'slot',np.arange(144))
    def hhmm(minutes):return f'{minutes//60:02d}:{minutes%60:02d}'
    df.insert(2,'interval',[f'{hhmm(k*10)}--{hhmm((k+1)*10)}' for k in range(144)])
    df['strategy_id']=mask;df['case_id']=case_id(mask);df['alpha']=alpha
    df['price_yuan_per_kwh']=ctx.price;df['load_kwh']=ctx.actual[d,:,0];df['pv_kwh']=ctx.actual[d,:,1]
    return df


def main():
    began=time.perf_counter();ctx=Context();results=ROOT/'results';qa=ROOT/'qa'
    results.mkdir(parents=True,exist_ok=True);qa.mkdir(parents=True,exist_ok=True)
    all_daily=[];selection=[];main_runs=[];certificates=[];chosen={}
    for mask in range(8):
        state=6000.;days=[];frames=[];alpha=.8;virtual_initial=None
        for d in range(365):
            if d==14:virtual_initial=state
            if d==31:
                for candidate in ALPHAS:
                    vs=virtual_initial;cost=0.
                    for vd in range(14,31):
                        vr,cs=simulate_day(ctx,vd,mask,candidate,vs);certificates.extend(cs)
                        cost+=float(np.sum(vr['total_cost_yuan']));vs=float(vr['soc_end_kwh'][-1])
                    score=cost-.9*ctx.price[0]*(vs-virtual_initial)
                    selection.append(dict(strategy_id=mask,case_id=case_id(mask),alpha=candidate,
                        calibration_start='2025-01-15',calibration_end='2025-01-31',
                        soc_start_kwh=virtual_initial,soc_end_kwh=vs,cost_yuan=cost,selection_score_yuan=score))
                candidates=[x for x in selection if x['strategy_id']==mask]
                best=min(candidates,key=lambda x:(x['selection_score_yuan'],x['alpha']))
                alpha=best['alpha'];chosen[case_id(mask)]=alpha
                for x in candidates:x['selected']=x is best
            run,cs=simulate_day(ctx,d,mask,alpha,state);certificates.extend(cs)
            days.append(daily_summary(ctx,d,mask,alpha,run));frames.append(make_frame(ctx,d,mask,alpha,run))
            if mask==7:main_runs.append(run)
            state=float(run['soc_end_kwh'][-1])
        out=results/'cases'/case_id(mask);out.mkdir(parents=True,exist_ok=True)
        daily=pd.DataFrame(days);pd.concat(frames,ignore_index=True).to_csv(out/'dispatch.csv',index=False)
        daily.to_csv(out/'daily.csv',index=False);all_daily.extend(days)
        print(json.dumps(dict(case=case_id(mask),alpha=alpha,annual_cost=daily.total_cost_yuan.sum(),
            formal_cost=daily.loc[daily.phase=='formal','total_cost_yuan'].sum(),
            formal_initial=daily.iloc[31].soc_start_kwh,final_soc=state,elapsed_seconds=time.perf_counter()-began)),flush=True)
    daily=pd.DataFrame(all_daily);daily.to_csv(results/'strategies_daily.csv',index=False)
    daily.to_csv(results/'annual_daily.csv',index=False)
    pd.DataFrame(selection).to_csv(results/'january_selection.csv',index=False)
    dump(qa/'selection.json',dict(primary_case_id='mask111',selected_case_id='mask111',primary_strategy_id=7,
        primary_choice='Prespecified all-update demonstration, not selected from annual outcomes.',
        alpha_by_case=chosen,candidate_grid=list(ALPHAS),settlement='half_refund',actual_january_alpha=.8,formal_deployment_date='2025-02-01',
        calibration='Jan15-Jan31 virtual rollouts from each policy actual Jan15 initial SOC; no actual-state overwrite.',
        lambda_yuan_per_internal_kwh=.9*ctx.price[0]))

    # 反事实共享日初状态、合同和风险参数，且不跨日拼接。
    counter=[];frozen=[]
    for d in range(365):
        source=main_runs[d];state=float(source['soc_start_kwh'][0]);q0=source['original_plan_kwh']
        alpha=.8 if d<31 else chosen['mask111']
        qhash=hashlib.sha256(np.asarray(q0,dtype='<f8').tobytes()).hexdigest()
        for mask in range(8):
            run,cs=simulate_day(ctx,d,mask,alpha,state,q0_override=q0);certificates.extend(cs)
            row=daily_summary(ctx,d,mask,alpha,run);row.update(reference_strategy_id=7,q0_sha256=qhash)
            counter.append(row)
        run,cs=simulate_day(ctx,d,7,alpha,state,q0_override=q0,freeze=True);certificates.extend(cs)
        row=daily_summary(ctx,d,7,alpha,run);row.update(reference_strategy_id=7,q0_sha256=qhash,
            control='all_updates_frozen_midnight_forecast_and_risk_margin')
        frozen.append(row)
        if d%60==0:print(json.dumps(dict(counterfactual_day=d+1,elapsed_seconds=time.perf_counter()-began)),flush=True)
    pd.DataFrame(counter).to_csv(results/'counterfactual_daily.csv',index=False)
    pd.DataFrame(frozen).to_csv(results/'frozen_information_daily.csv',index=False)
    dump(qa/'planning_protocol.json',dict(passed=True,elapsed_seconds=time.perf_counter()-began,
        python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,
        lp_count=len(certificates),lp_certificate_max=summarize_certificates(certificates),
        all_year_case_days=len(all_daily),counterfactual_rows=len(counter),
        price='attachment1 fixed curve',charging_side='AC',eta=.9,
        initial_soc=6000,formal_period=['2025-02-01','2025-12-31'],
        year_end='No auxiliary January 2026 slots and no arbitrary terminal equality.',
        primary_is_prespecified=True,numerical_results_recomputed=True,interpolation='piecewise_linear_analytic_integration',
        input_sha256={name:hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest()
          for name in ['actual.csv','price.csv','forecast_component_selected.npz','native_pv_energy.npz']}))
    print(json.dumps(dict(done=True,elapsed_seconds=time.perf_counter()-began)),flush=True)


if __name__=='__main__':main()
