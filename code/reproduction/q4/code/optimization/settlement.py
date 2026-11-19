"""按增购加价、取消半额退款和紧急购电规则计算第三问账单。"""
import numpy as np

UP_FACTOR=1.5
REFUND_FRACTION=.5
EMERGENCY_FACTOR=5.

def settle(prices, original, normal, emergency):
    p,q0,q,e=(np.asarray(v,dtype=float) for v in (prices,original,normal,emergency))
    assert p.shape==q0.shape==q.shape==e.shape
    assert all(np.isfinite(v).all() for v in (p,q0,q,e)) and np.all(p>0)
    assert min(q0.min(),q.min(),e.min())>=-1e-7
    up=np.maximum(q-q0,0);down=np.maximum(q0-q,0)
    plan=p*q0;increase=UP_FACTOR*p*up;refund=-REFUND_FRACTION*p*down
    urgent=EMERGENCY_FACTOR*p*e
    return dict(up_kwh=up,down_kwh=down,plan_cost_yuan=plan,up_cost_yuan=increase,
                down_cost_yuan=refund,emergency_cost_yuan=urgent,
                total_cost_yuan=plan+increase+refund+urgent)
