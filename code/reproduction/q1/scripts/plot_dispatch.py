"""读取问题一求解结果，绘制购电与储能调度图。"""
from pathlib import Path
import hashlib
import json
import os
import numpy as np

BASE = Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR'] = str(BASE / 'results/mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator


def draw_dispatch(base=BASE):
    base = Path(base)
    source = base / 'results/solution.json'
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    solution = json.loads(source.read_text(encoding='utf-8'))
    rows = solution['strategy']
    params = solution['parameters']
    dt = params['dt_hours']
    n = len(rows)
    edges = np.r_[[r['start_minute']/60 for r in rows], rows[-1]['end_minute']/60]
    assert n == 144 and np.allclose(np.diff(edges), dt)
    load = np.array([r['load_kw'] for r in rows])
    pv = np.array([r['pv_forecast_kw'] for r in rows])
    price = np.array([r['price_yuan_kwh'] for r in rows])
    grid = np.array([r['grid_kwh'] for r in rows])/dt
    charge = np.array([r['charge_kwh'] for r in rows])/dt
    discharge = np.array([r['discharge_kwh'] for r in rows])/dt
    energy = np.r_[solution['initial_energy_kwh'], [r['energy_end_kwh'] for r in rows]]
    emin, emax = params['minimum_energy_kwh'], params['maximum_energy_kwh']
    pmax = params['maximum_ac_power_kw']
    assert np.min(grid+pv+discharge-load-charge) >= -1e-7
    assert np.allclose(np.diff(energy), params['eta_charge']*charge*dt-discharge*dt/params['eta_discharge'], atol=1e-7)
    assert max(charge.max(),discharge.max()) <= pmax+1e-7
    assert energy.min() >= emin-1e-7 and energy.max() <= emax+1e-7
    assert abs(grid.sum()*dt-solution['summary']['total_purchase_kwh']) < 1e-7

    font_path = Path('C:/Windows/Fonts/msyh.ttc')
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
    else:
        installed_fonts = {font.name for font in font_manager.fontManager.ttflist}
        font_name = next((name for name in ('Noto Sans CJK SC', 'Noto Sans CJK JP', 'PingFang SC', 'SimHei')
                          if name in installed_fonts), 'sans-serif')
    plt.rcParams.update({
        'font.family':font_name, 'font.size':11, 'axes.labelsize':11,
        'axes.titlesize':12, 'legend.fontsize':10.5, 'axes.unicode_minus':False,
        'axes.spines.top':False, 'axes.spines.right':False,
        'axes.edgecolor':'#79AFB1', 'axes.linewidth':.7,
        'xtick.color':'#334155', 'ytick.color':'#334155',
        'text.color':'#1e293b', 'axes.labelcolor':'#334155',
        'svg.fonttype':'path', 'savefig.facecolor':'white',
    })
    colors = dict(load='#459688',pv='#E89E77',grid='#79AFB1',price='#D1A38C',
                  charge='#459688',discharge='#D98D67',energy='#459688',limit='#D1A38C')
    fig, axs = plt.subplots(4,1,figsize=(10.8,11.0),sharex=True)
    fig.subplots_adjust(left=.105,right=.888,top=.89,bottom=.065,hspace=.64)
    fig.suptitle('第一问 · 日前购电与储能调度',x=.105,y=.988,ha='left',fontsize=16,fontweight='bold')
    fig.text(.105,.957,'144 个 10 分钟时段  |  原始最优策略逐段展示',fontsize=10.5,color='#64748b')

    titles = ['(a) 已知负载与光伏预测','(b) 计划购电与电价','(c) 储能充放电控制','(d) 储电量与运行边界']
    legends = []
    for ax,title in zip(axs,titles):
        ax.set_title(title,loc='left',y=1.16,pad=0,fontweight='bold')
        ax.set_xlim(0,24)
        ax.set_axisbelow(True)
        ax.grid(axis='y',color='#E0E3DE',alpha=.6,lw=.6)
        ax.grid(axis='x',color='#EBEFED',alpha=.7,lw=.6)
        ax.xaxis.set_major_locator(MultipleLocator(4))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.tick_params(which='minor',length=2.5)
        ax.tick_params(which='major',length=3.5)
        ax.margins(x=0)

    def header_legend(ax, handles, labels, ncol):
        legend = ax.legend(handles,labels,ncol=ncol,loc='lower right',
                           bbox_to_anchor=(1,1.045),borderaxespad=0,frameon=False,
                           columnspacing=1.35,handlelength=2.0,handletextpad=.5)
        legends.append(legend)
        return legend

    axs[0].stairs(load,edges,baseline=None,color=colors['load'],lw=1.65)
    axs[0].stairs(pv,edges,baseline=None,color=colors['pv'],lw=1.7)
    axs[0].set_ylim(0,8500)
    axs[0].set_yticks([0,2000,4000,6000,8000])
    axs[0].set_ylabel('功率 / kW')
    header_legend(axs[0],[Line2D([],[],color=colors['load'],lw=1.7),
                         Line2D([],[],color=colors['pv'],lw=1.7)],['小区负载','光伏预测'],2)

    axs[1].bar(edges[:-1],grid,width=dt,align='edge',color=colors['grid'],
               alpha=.86,linewidth=0,zorder=2)
    axs[1].set_ylim(0,9200)
    axs[1].set_yticks([0,2000,4000,6000,8000])
    axs[1].set_ylabel('购电功率 / kW')
    price_ax = axs[1].twinx()
    price_ax.stairs(price,edges,baseline=None,color=colors['price'],lw=1.65,linestyle='--',zorder=3)
    price_ax.set_ylim(0,1.6)
    price_ax.set_yticks([0,.4,.8,1.2,1.6])
    price_ax.set_ylabel('电价 / (元/kWh)',color=colors['price'],labelpad=8)
    price_ax.tick_params(axis='y',colors=colors['price'],length=3.5)
    price_ax.spines['right'].set_visible(True)
    price_ax.spines['right'].set_color('#E0E3DE')
    header_legend(axs[1],[Patch(facecolor=colors['grid'],alpha=.86),
                         Line2D([],[],color=colors['price'],lw=1.65,ls='--')],
                         ['购电功率（左轴）','电价（右轴）'],2)

    axs[2].bar(edges[:-1],charge,width=dt,align='edge',color=colors['charge'],linewidth=0)
    axs[2].bar(edges[:-1],-discharge,width=dt,align='edge',color=colors['discharge'],linewidth=0)
    for limit in (-pmax,pmax):
        axs[2].axhline(limit,color=colors['limit'],ls=(0,(4,3)),lw=.95,zorder=3)
    axs[2].axhline(0,color='#79AFB1',lw=.8)
    axs[2].set_ylim(-6100,6100)
    axs[2].set_yticks([-5000,-2500,0,2500,5000])
    axs[2].set_ylabel('储能功率 / kW')
    header_legend(axs[2],[Patch(facecolor=colors['charge']),Patch(facecolor=colors['discharge']),
                         Line2D([],[],color=colors['limit'],lw=.95,ls='--')],
                         ['充电（正）','放电（负）','功率上限 ±5000'],3)

    axs[3].axhspan(emin,emax,color='#D1E2DA',alpha=.72,zorder=0)
    axs[3].plot(edges,energy,color=colors['energy'],lw=2.0,zorder=3)
    for limit in (emin,emax):
        axs[3].axhline(limit,color=colors['limit'],ls=(0,(4,3)),lw=.95,zorder=2)
    axs[3].scatter([0,24],[energy[0],energy[-1]],s=27,facecolor='white',
                   edgecolor=colors['energy'],linewidth=1.5,zorder=5,clip_on=False)
    axs[3].text(.25,energy[0]-350,'6000',ha='left',va='top',fontsize=10,color=colors['energy'])
    axs[3].text(23.75,energy[-1]+400,'6000',ha='right',va='bottom',fontsize=10,color=colors['energy'])
    axs[3].set_ylim(0,12000)
    axs[3].set_yticks([0,1200,6000,10800,12000])
    axs[3].set_ylabel('储电量 / kWh')
    header_legend(axs[3],[Line2D([],[],color=colors['energy'],lw=2),
                         Patch(facecolor='#D1E2DA',edgecolor='none')],
                         ['储电量','安全范围 1200–10800 kWh'],2)
    axs[3].set_xticks(np.arange(0,25,4),[f'{h:02d}:00' for h in range(0,25,4)])
    axs[3].set_xlabel('当天时间',labelpad=8)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_checks = []
    for i,(ax,legend) in enumerate(zip(axs,legends)):
        legend_box = legend.get_window_extent(renderer)
        axes_box = ax.get_window_extent(renderer)
        assert legend_box.y0 > axes_box.y1, (i,'legend overlaps data')
        assert not legend_box.overlaps(ax._left_title.get_window_extent(renderer)), (i,'legend overlaps title')
        legend_checks.append(dict(panel=i+1,legend_above_data=True))
    outdir = base/'figures'
    outdir.mkdir(parents=True,exist_ok=True)
    fig.savefig(outdir/'dispatch.png',dpi=320)
    fig.savefig(outdir/'dispatch.svg')
    plt.close(fig)
    assert source_hash == hashlib.sha256(source.read_bytes()).hexdigest()
    qa = dict(source_sha256=source_hash,unmodified_source=True,interval_count=n,
              no_smoothing=True,maximum_grid_power_kw=float(grid.max()),
              maximum_charge_power_kw=float(charge.max()),maximum_discharge_power_kw=float(discharge.max()),
              displayed_initial_energy_kwh=float(energy[0]),displayed_final_energy_kwh=float(energy[-1]),
              legends=legend_checks,png=str(outdir/'dispatch.png'),svg=str(outdir/'dispatch.svg'))
    (base/'results/plot_verification.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(qa,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    draw_dispatch()
