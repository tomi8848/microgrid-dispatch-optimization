"""按发布日期组织第三问预测数据，并用历史误差计算风险余量。
所有统计量仅使用决策时点之前已完成的目标日。"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
ALPHAS=(0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,1.0)


class Context:
    def __init__(self,root=ROOT):
        self.root=Path(root)
        frame=pd.read_csv(self.root/'data/actual.csv')
        self.actual=frame[['load_kw','pv_kw']].to_numpy(float).reshape(365,144,2)/6
        self.net=self.actual[:,:,0]-self.actual[:,:,1]
        self.dates=pd.to_datetime(frame.sample_date).dt.strftime('%Y-%m-%d').drop_duplicates().to_numpy()
        self.price=pd.read_csv(self.root/'data/price.csv').price_yuan_per_kwh.to_numpy(float)
        with np.load(self.root/'data/forecast_component_selected.npz') as z:self.historical=z['predictions'].copy()
        with np.load(self.root/'data/baseline.npz') as z:assert np.array_equal(z['predictions'][:31],self.historical[:31])
        with np.load(self.root/'data/native_pv_energy.npz') as z:
            self.native=z['issued'].copy()
            assert np.all(z['deployed_method']=='linear'), '本精简版必须以全年线性积分重新运行。'
        assert self.native.shape==(365,4,144) and self.historical.shape==(365,2,144,2)
        self.net_forecasts=np.empty((365,4,2,144))
        for d in range(365):
            for ri,r in enumerate([0,6,12,18]):
                pv=self.historical[d,:,:,1].reshape(288).copy()
                pv[r*6:r*6+144]=self.native[d,ri]
                self.net_forecasts[d,ri]=self.historical[d,:,:,0]-pv.reshape(2,144)
        self.margin_cache={}

    def risk_margin(self,d,ri,alpha):
        assert np.isfinite(alpha) and 0.0 <= alpha <= 1.0
        key=(int(d),int(ri),float(alpha))
        if key in self.margin_cache:return self.margin_cache[key]
        result=np.zeros((2,144));start=ri*36
        for h in range(2):
            targets=np.arange(max(0,d-28),d)
            targets=targets[targets-h>=7]
            if not len(targets):continue
            errors=self.net[targets]-self.net_forecasts[targets-h,ri,h]
            for b in range(6):
                lo=max(b*24,start if h==0 else 0);hi=(b+1)*24
                if lo>=hi:continue
                value=max(0.,float(np.quantile(errors[:,lo:hi].ravel(),alpha,method='linear')))
                result[h,b*24:(b+1)*24]=value
        self.margin_cache[key]=result
        return result

    def planning_data(self,d,ri,alpha,freeze=False):
        start=ri*36;end=144 if d==364 else 288
        source=0 if freeze else ri
        demand=(self.net_forecasts[d,source]+self.risk_margin(d,source,alpha)).reshape(288)[start:end]
        price=np.tile(self.price,2)[start:end].copy()
        return demand,price
