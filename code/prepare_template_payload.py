from pathlib import Path
import sys,json,datetime,csv,os
import numpy as np,pandas as pd
from openpyxl import load_workbook
C=Path(__file__).resolve().parent;B=Path(os.environ.get('TEMPLATE_RUN_DIR',str(C.parent)));P=C/'reproduction';T=P/'C题/附件/附件5'
ALIGN_TIME=os.environ.get('ALIGN_TEMPLATE_TIMES','1')=='1'
def read_series(name):
 if name=='result2.xlsx':
  f=pd.read_csv(P/'q2/results/dispatch.csv');a=lambda k:f[k].to_numpy().reshape(334,144)
  q0=a('planned_kwh');q=q0;charge=a('charge_kwh');dis=a('discharge_kwh');e=a('emergency_kwh');prices=a('price_yuan_per_kwh');soc=np.c_[a('soc_start_kwh')[:,0],a('soc_end_kwh')]
 elif name=='result3.xlsx':
  f=pd.read_csv(P/'q3/results/cases/mask111/dispatch.csv').iloc[31*144:];a=lambda k:f[k].to_numpy().reshape(334,144)
  q0=a('original_plan_kwh');q=a('normal_kwh');charge=a('charge_kwh');dis=a('discharge_kwh');e=a('emergency_kwh');prices=a('price_yuan_per_kwh');soc=np.c_[a('soc_start_kwh')[:,0],a('soc_end_kwh')]
 else:
  f=np.load(P/'q4/results'/('dispatch_q2_xgb_adaptive.npz' if name=='result4-2.xlsx' else 'dispatch_q3_xgb_adaptive_mask7.npz'))
  q0,q,charge,dis,e,soc=[f[k][31:] for k in ['q0','normal','charge','discharge','emergency','soc']];prices=np.load(P/'q4/data/prices.npz')['price'][31:]
 assert q0.shape==(334,144) and soc.shape==(334,145)
 assert np.max(abs(soc[1:,0]-soc[:-1,-1]))<1e-8
 return q0,q,charge,dis,e,soc,prices
def clock(k):return f'{k//6}:{k%6*10:02d}'
def body_style(sheet, width, period):
 rows=[]
 for i in range(period):
  cells=[]
  for j in range(width):
   c=sheet.cell(i+2,j+1)
   cells.append(dict(font=dict(name=c.font.name,size=c.font.sz,bold=bool(c.font.bold),italic=bool(c.font.italic)),horizontal=c.alignment.horizontal or 'general',vertical=c.alignment.vertical or 'bottom',borders={edge:getattr(c.border,edge).style if getattr(c.border,edge) else None for edge in ['left','right','top','bottom']}))
  rows.append(cells)
 return dict(rows=rows,heights=[sheet.row_dimensions[i+2].height or sheet.sheet_format.defaultRowHeight for i in range(period)])
def prepare(name):
 if name=='result1.xlsx':
  s=json.loads((P/'q1/results/solution.json').read_text(encoding='utf-8'))
  return dict(name=name,plan=[[r['grid_kwh']] for r in s['strategy']],storage=[[r['charge_kwh'],r['discharge_kwh']] for r in s['four_hour']],endpoints=[[s['initial_energy_kwh']],[s['terminal_energy_kwh']]],cost=s['summary']['total_cost_yuan'])
 q0,q,a,b,e,soc,p=read_series(name);dates=pd.date_range('2025-02-01','2025-12-31').strftime('%Y-%m-%d').tolist()
 fee=(p*q0).sum(1);adjust=(p*(1.5*np.maximum(q-q0,0)-.5*np.maximum(q0-q,0))).sum(1);efee=(5*p*e).sum(1)
 storage=[];events=[];event_groups=[]
 for d,date in enumerate(dates):
  for j in range(6):storage.append([date if j==0 else None,clock(j*24)+'-'+clock((j+1)*24),float(a[d,j*24:(j+1)*24].sum()),float(b[d,j*24:(j+1)*24].sum()),0 if j==0 else '24:00' if j==1 else None,float(soc[d,0]) if j==0 else float(soc[d,-1]) if j==1 else None])
  dayevents=[];k=0
  while k<144:
   if e[d,k]<=1e-7:k+=1;continue
   lo=k
   while k<144 and e[d,k]>1e-7:k+=1
   dayevents.append([date if not dayevents else None,clock(lo)+'-'+clock(k),float(e[d,lo:k].sum())])
  if not dayevents:dayevents=[[date,'无',0.]]
  event_groups.append(dict(start_row=len(events)+2,rows=len(dayevents)));events.extend(dayevents)
 frame=pd.DataFrame(dict(date=dates,soc_start=soc[:,0],soc_end=soc[:,-1],planned_kwh=q0.sum(1),normal_kwh=q.sum(1),charge_kwh=a.sum(1),discharge_kwh=b.sum(1),emergency_kwh=e.sum(1),plan_fee_yuan=fee,adjustment_fee_yuan=adjust,emergency_fee_yuan=efee,total_cost_yuan=fee+adjust+efee))
 (B/'辅助明细').mkdir(exist_ok=True);frame.to_csv(B/'辅助明细'/f'{name[:-5]}_逐日账单.csv',index=False,encoding='utf-8-sig')
 template=load_workbook(T/name)
 formats=[[template['充放电量'].cell(i%6+2,j+1).number_format for j in range(6)] for i in range(len(storage))]
 event_formats=[[template['紧急购电量'].cell(2 if r[0] is not None else 3,j+1).number_format for j in range(3)] for r in events]
 storage_style=body_style(template['充放电量'],6,6);event_style=body_style(template['紧急购电量'],3,3)
 template.close()
 return dict(name=name,dates=dates,plan=q0.tolist(),normal=q.tolist(),plan_fees=fee.tolist(),adjustment_fees=adjust.tolist(),storage=storage,storage_formats=formats,storage_style=storage_style,events=events,event_formats=event_formats,event_style=event_style,event_groups=event_groups,formal_cost=float((fee+adjust+efee).sum()))
def supplement():
 cols=['sample_date','soc_start_kwh','soc_end_kwh','original_plan_kwh','normal_kwh','charge_kwh','discharge_kwh','emergency_kwh','plan_cost_yuan','up_cost_yuan','down_cost_yuan','emergency_cost_yuan','total_cost_yuan','phase','alpha']
 cases={}
 for case in ['000','100','010','110','001','101','011','111']:
  f=pd.read_csv(P/f'q3/results/cases/mask{case}/daily.csv');cases[case]=f[cols].values.tolist()
 f=pd.read_csv(P/'q3/results/january_selection.csv');cal=f[['case_id','alpha','soc_start_kwh','soc_end_kwh','cost_yuan','selection_score_yuan','selected']].values.tolist()
 return dict(name='第三问_全年365天_八种情况.xlsx',headers=['日期','日初储电量/kWh','日末储电量/kWh','原计划购电/kWh','正常购电/kWh','充电量/kWh','放电量/kWh','紧急购电/kWh','计划费/元','上调费/元','下调退款/元','紧急费/元','总费用/元','运行阶段','余量参数'],cases=cases,calibration=cal,inventory_value=.38232)
def align_to_template(out,name):
 """Change only export windows: source clock intervals and solver arrays remain intact."""
 if name=='result1.xlsx':
  source=json.loads((P/'q1/results/solution.json').read_text(encoding='utf-8'))['strategy']
  out['plan']=out['plan'][1:]+[[None]]
  out['alignment']=dict(mode='literal_template_time',known_intervals=143,missing_cell='计划购电量!B145',missing_interval='次日00:00-00:10',excluded_first_kwh=source[0]['grid_kwh'],excluded_first_cost_yuan=source[0]['cost_yuan'],known_purchase_kwh=sum(r['grid_kwh'] for r in source[1:]),known_cost_yuan=sum(r['cost_yuan'] for r in source[1:]))
  return out
 q0,q,a,b,e,soc,prices=read_series(name)
 def shift(v):return np.r_[v.ravel()[1:],np.nan].reshape(v.shape)
 def rows(v):return [[None if not np.isfinite(x) else float(x) for x in r] for r in v]
 plan=shift(q0);normal=shift(q);plan_cost=prices*q0;adjust_cost=prices*(1.5*np.maximum(q-q0,0)-.5*np.maximum(q0-q,0))
 pc=shift(plan_cost);ac=shift(adjust_cost)
 out['plan']=rows(plan);out['normal']=rows(normal)
 out['plan_fees']=[float(x) if np.isfinite(x) else None for x in pc.sum(1)]
 out['adjustment_fees']=[float(x) if np.isfinite(x) else None for x in ac.sum(1)]
 out['alignment']=dict(mode='literal_template_time',known_intervals=int(np.isfinite(plan).sum()),missing_cell='EO335',missing_interval='2026-01-01 00:00-00:10',excluded_first_kwh=float(q0[0,0]),excluded_first_cost_yuan=float(plan_cost[0,0]),calendar_day_original_formal_cost=out['formal_cost'])
 frame=pd.DataFrame(dict(date=out['dates'],display_window_start=pd.date_range('2025-02-01 00:10',periods=334,freq='D').astype(str),display_window_end=pd.date_range('2025-02-02 00:10',periods=334,freq='D').astype(str),complete=np.isfinite(plan).all(1),known_slots=np.isfinite(plan).sum(1),original_calendar_plan_kwh=q0.sum(1),removed_current_day_first_kwh=q0[:,0],added_next_day_first_kwh=np.r_[q0[1:,0],np.nan],displayed_plan_kwh=plan.sum(1),known_plan_kwh=np.nansum(plan,axis=1),original_calendar_plan_fee=plan_cost.sum(1),removed_current_day_first_fee=plan_cost[:,0],added_next_day_first_fee=np.r_[plan_cost[1:,0],np.nan],displayed_plan_fee=pc.sum(1),known_plan_fee=np.nansum(pc,axis=1),original_calendar_adjustment_fee=adjust_cost.sum(1),displayed_adjustment_fee=ac.sum(1),known_adjustment_fee=np.nansum(ac,axis=1)))
 frame.to_csv(B/'辅助明细'/f'{name[:-5]}_时间校正对账.csv',index=False,encoding='utf-8-sig')
 return out
def preserve_presentation(out,name):
 directory=os.environ.get('TEMPLATE_SOURCE_DIR')
 if not directory:return out
 w=load_workbook(Path(directory)/name);meta={}
 for s in w:
  heights=[]
  for row in range(1,s.max_row+1):
   h=s.row_dimensions[row].height or s.sheet_format.defaultRowHeight
   if heights and heights[-1]['height']==h:heights[-1]['last_row']=row
   else:heights.append(dict(first_row=row,last_row=row,height=h))
  meta[s.title]=dict(rows=s.max_row,cols=s.max_column,number_formats=[[c.number_format for c in row] for row in s],widths=[dict(first_column=v.min,last_column=v.max,width=v.width) for v in s.column_dimensions.values()],heights=heights)
 out['presentation']=meta;w.close();return out
if __name__=='__main__':
 for folder in ['核验','辅助明细','预览']:(B/folder).mkdir(parents=True,exist_ok=True)
 names=sys.argv[1:] or ['result1.xlsx','result2.xlsx','result3.xlsx','result4-2.xlsx','result4-3.xlsx','第三问_全年365天_八种情况.xlsx']
 for name in names:
  out=supplement() if name.startswith('第三问') else prepare(name)
  if ALIGN_TIME and not name.startswith('第三问'):out=preserve_presentation(align_to_template(out,name),name)
  (B/'核验'/f'{name}.payload.json').write_text(json.dumps(out,ensure_ascii=False,allow_nan=False),encoding='utf-8');print('PREPARED',name)
 # Explicit mapping requested by the user; no changes to original template labels.
 w=load_workbook(T/'result2.xlsx',read_only=True)
 with (B/('模板时段对应_已校正.csv' if ALIGN_TIME else '模板时段映射.csv')).open('w',encoding='utf-8-sig',newline='') as f:
  writer=csv.writer(f);writer.writerow(['时段序号','Excel列','模板原表头','填充值实际时段','原始附件右端采样时刻'])
  from openpyxl.utils import get_column_letter
  for k in range(144):
   interval=(clock(k+1)+'-'+clock(k+2) if k<143 else '次日0:00-0:10') if ALIGN_TIME else clock(k)+'-'+clock(k+1)
   endpoint=(clock(k+2) if k<143 else '次日0:10') if ALIGN_TIME else clock(k+1)
   writer.writerow([k+1,get_column_letter(k+2),w['计划购电量'].cell(1,k+2).value,interval,endpoint])
 w.close()
