"""建立并求解问题一的确定性日前购电—储能调度 LP 模型。"""
from pathlib import Path
import os
import json
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import linprog

BASE = Path(__file__).resolve().parents[1]
df = pd.read_csv(BASE / 'data/input_day.csv')
n, dt = len(df), 1/6
eta_c = eta_d = 0.9
capacity = 12000.0
emin, emax, eini = 1200.0, 10800.0, 6000.0
pmax = 5000.0
p = df.price_yuan_kwh.to_numpy()
load = df.load_kw.to_numpy() * dt
pv = df.pv_forecast_kw.to_numpy() * dt
assert n == 144 and np.isfinite(df[['price_yuan_kwh','load_kw','pv_forecast_kw']]).all().all()
objective = np.r_[p, np.zeros(3*n)]
A_ub = np.zeros((n, 4*n))
A_eq = np.zeros((n, 4*n))
for t in range(n):
    A_ub[t, t] = -1
    A_ub[t, n+t] = 1
    A_ub[t, 2*n+t] = -1
    A_eq[t, n+t] = -eta_c
    A_eq[t, 2*n+t] = 1/eta_d
    A_eq[t, 3*n+t] = 1
    if t:
        A_eq[t, 3*n+t-1] = -1
b_ub = pv-load
b_eq = np.r_[eini, np.zeros(n-1)]
lower = np.r_[np.zeros(3*n), np.full(n, emin)]
upper = np.r_[np.full(n,np.inf), np.full(2*n,pmax*dt), np.full(n,emax)]
lower[-1] = upper[-1] = eini
res = linprog(objective, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
              bounds=list(zip(lower,upper)), method='highs-ds',
              options={'primal_feasibility_tolerance':1e-9, 'dual_feasibility_tolerance':1e-9})
if not res.success:
    raise RuntimeError(res.message)
q,c,d,e = np.split(res.x.copy(),4)
overlap_raw = int(np.sum((c>1e-7)&(d>1e-7)))
epsilon = np.minimum(c, d/(eta_c*eta_d))
c -= epsilon
d -= eta_c*eta_d*epsilon
for arr in (q,c,d):
    arr[np.abs(arr)<1e-9] = 0.0
estate = np.r_[eini,e]
slack = q+pv+d-load-c
dynamic_residual = np.diff(estate)-eta_c*c+d/eta_d
dual_obj = (b_ub @ res.ineqlin.marginals + b_eq @ res.eqlin.marginals
            + lower @ res.lower.marginals
            + upper[np.isfinite(upper)] @ res.upper.marginals[np.isfinite(upper)])
checks = dict(solver_status=res.message,
              minimum_supply_margin_kwh=float(slack.min()),
              maximum_storage_equation_error_kwh=float(np.abs(dynamic_residual).max()),
              minimum_energy_kwh=float(estate.min()), maximum_energy_kwh=float(estate.max()),
              initial_energy_kwh=float(estate[0]), terminal_energy_kwh=float(estate[-1]),
              terminal_error_kwh=float(abs(estate[-1]-eini)),
              maximum_charge_power_kw=float(c.max()/dt),
              maximum_discharge_power_kw=float(d.max()/dt),
              minimum_grid_energy_kwh=float(q.min()),
              simultaneous_charge_discharge_intervals=int(np.sum((c>1e-7)&(d>1e-7))),
              raw_simultaneous_intervals=overlap_raw,
              maximum_surplus_exceeding_pv_kwh=float(max(0,np.max(slack-pv))),
              grid_and_surplus_intervals=int(np.sum((q>1e-7)&(slack>1e-7))),
              lp_dual_lower_bound_yuan=float(dual_obj),
              primal_dual_gap_yuan=float(p@q-dual_obj))
loss = (1-eta_c)*c.sum() + (1/eta_d-1)*d.sum()
checks['whole_day_energy_balance_error_kwh'] = float(q.sum()+pv.sum()-load.sum()-loss-slack.sum()-(e[-1]-eini))
assert checks['minimum_supply_margin_kwh'] >= -1e-7
assert checks['maximum_storage_equation_error_kwh'] < 1e-7
assert estate.min() >= emin-1e-7 and estate.max() <= emax+1e-7
assert c.max()/dt <= pmax+1e-7 and d.max()/dt <= pmax+1e-7
assert checks['simultaneous_charge_discharge_intervals'] == 0
assert abs(checks['primal_dual_gap_yuan']) < 1e-6
assert checks['maximum_surplus_exceeding_pv_kwh'] < 1e-7
assert checks['grid_and_surplus_intervals'] == 0
strategy = df.copy()
for name,values in dict(load_kwh=load,pv_kwh=pv,grid_kwh=q,charge_kwh=c,discharge_kwh=d,
                        energy_start_kwh=estate[:-1],energy_end_kwh=e,
                        unused_surplus_kwh=np.maximum(slack,0),cost_yuan=p*q).items():
    strategy[name] = values
strategy.to_csv(BASE/'results/strategy_144.csv',index=False,encoding='utf-8-sig',float_format='%.12g')
four_hour = []
for start in range(0,24,4):
    mask = (df.start_minute >= start*60) & (df.start_minute < (start+4)*60)
    four_hour.append(dict(start_hour=start,end_hour=start+4,
                          charge_kwh=float(c[mask].sum()),discharge_kwh=float(d[mask].sum())))
selected = [strategy.loc[strategy.start_minute == hour*60].iloc[0].to_dict()
            for hour in (10,12,14,16,18,20)]
baseline = np.maximum(load-pv,0)
summary = dict(total_purchase_kwh=float(q.sum()), total_cost_yuan=float(p@q),
               total_load_kwh=float(load.sum()), total_pv_kwh=float(pv.sum()),
               total_charge_kwh=float(c.sum()), total_discharge_kwh=float(d.sum()),
               total_unused_surplus_kwh=float(np.maximum(slack,0).sum()),
               storage_loss_kwh=float(loss),
               baseline_cost_yuan=float(p@baseline), baseline_purchase_kwh=float(baseline.sum()),
               cost_saving_yuan=float(p@baseline-p@q), cost_saving_fraction=float(1-(p@q)/(p@baseline)))
out = dict(parameters=dict(dt_hours=dt,capacity_kwh=capacity,minimum_energy_kwh=emin,
                          maximum_energy_kwh=emax,maximum_ac_power_kw=pmax,eta_charge=eta_c,eta_discharge=eta_d),
           initial_energy_kwh=eini,terminal_energy_kwh=float(e[-1]),
           scipy_version=scipy.__version__,method='HiGHS dual simplex',
           summary=summary,checks=checks,selected_intervals=selected,
           four_hour=four_hour,strategy=strategy.to_dict(orient='records'))
(BASE/'results/solution.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(summary=summary,four_hour=four_hour,
                      selected=[dict(interval=r['interval'],grid_kwh=r['grid_kwh']) for r in selected],
                      checks=checks),ensure_ascii=False,indent=2))

from plot_dispatch import draw_dispatch
draw_dispatch(BASE)
