"""模拟第三问单日滚动调整、储能物理反馈和实际费用结算。"""
import numpy as np
from optimization.dispatch_lp import solve,solve_refund,ETA,SOC_MIN,SOC_MAX,FLOW_MAX
from optimization.settlement import settle


def case_id(mask):
    return 'mask'+''.join(str(int(bool(mask&(1<<i)))) for i in range(3))


def releases(mask):
    return [0]+[r for i,r in enumerate([6,12,18]) if mask&(1<<i)]


def simulate_day(ctx,d,mask,alpha,initial_soc,q0_override=None,freeze=False):
    certs=[]
    if q0_override is None:
        m,p=ctx.planning_data(d,0,alpha)
        plan,cert=solve(m,p,initial_soc);certs.append(cert);q0=plan[:144].copy()
    else:
        q0=np.asarray(q0_override,float).copy();assert q0.shape==(144,)
    normal=q0.copy();charge=np.zeros(144);discharge=np.zeros(144)
    emergency=np.zeros(144);unused=np.zeros(144);issues=np.zeros(144,dtype=int)
    soc=np.empty(145);soc[0]=initial_soc
    times=releases(mask)+[24]
    for r,rnext in zip(times[:-1],times[1:]):
        k0,k1=6*r,6*rnext
        if r:
            demand,price=ctx.planning_data(d,r//6,alpha,freeze=freeze)
            candidate,cert=solve_refund(demand,price,soc[k0],q0[k0:]);certs.append(cert)
            normal[k0:k1]=candidate[:k1-k0]
        issues[k0:k1]=r
        for k in range(k0,k1):
            residual=normal[k]-ctx.net[d,k]
            if residual>=0:
                charge[k]=min(residual,FLOW_MAX,max((SOC_MAX-soc[k])/ETA,0.))
                unused[k]=residual-charge[k]
            else:
                discharge[k]=min(-residual,FLOW_MAX,max(ETA*(soc[k]-SOC_MIN),0.))
                emergency[k]=-residual-discharge[k]
            soc[k+1]=soc[k]+ETA*charge[k]-discharge[k]/ETA
    run=dict(original_plan_kwh=q0,normal_kwh=normal,
        charge_kwh=charge,discharge_kwh=discharge,emergency_kwh=emergency,unused_kwh=unused,
        soc_start_kwh=soc[:-1],soc_end_kwh=soc[1:],issue_hour=issues)
    run.update(settle(ctx.price,q0,normal,emergency))
    return run,certs


def daily_summary(ctx,d,mask,alpha,run):
    row=dict(sample_date=ctx.dates[d],strategy_id=int(mask),case_id=case_id(mask),
        update_hours=','.join(map(str,releases(mask)[1:])) or 'none',alpha=float(alpha),
        phase='warmup' if d<31 else 'formal',soc_start_kwh=float(run['soc_start_kwh'][0]),
        soc_end_kwh=float(run['soc_end_kwh'][-1]),load_kwh=float(ctx.actual[d,:,0].sum()),
        pv_kwh=float(ctx.actual[d,:,1].sum()))
    for k,v in run.items():
        if k not in ['soc_start_kwh','soc_end_kwh','issue_hour']:row[k]=float(np.sum(v))
    row['planned_cost_yuan']=row['plan_cost_yuan']
    row['adjustment_cost_yuan']=row['up_cost_yuan']+row['down_cost_yuan']
    row['emergency_intervals']=int(np.sum(run['emergency_kwh']>1e-6))
    row['inventory_adjusted_cost_yuan']=row['total_cost_yuan']-.9*ctx.price[0]*(row['soc_end_kwh']-row['soc_start_kwh'])
    return row
