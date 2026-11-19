"""求解第三问日初购电和日内调整的辅助线性规划，生成可执行候选计划。"""
from functools import lru_cache
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, hstack, vstack
from optimization.settlement import UP_FACTOR,REFUND_FRACTION

ETA=.9
SOC_MIN,SOC_MAX=1200.,10800.
FLOW_MAX=5000./6


@lru_cache(maxsize=8)
def structure(n):
    rr=[];cc=[];vv=[]
    for k in range(n):
        for col,val in [(k,1),(n+k,-1),(2*n+k,1),(3*n+k,-1)]:
            rr.append(k);cc.append(col);vv.append(val)
        for col,val in [(n+k,-ETA),(2*n+k,1/ETA),(4*n+k,-1),(4*n+k+1,1)]:
            rr.append(n+k);cc.append(col);vv.append(val)
    rr.append(2*n);cc.append(4*n);vv.append(1.)
    return coo_matrix((vv,(rr,cc)),shape=(2*n+1,5*n+1)).tocsr()


def solve(net,prices,initial_soc,lower_q=None):
    net=np.asarray(net,float);prices=np.asarray(prices,float);n=len(net)
    assert n and net.shape==prices.shape and np.isfinite(net).all()
    assert np.isfinite(prices).all() and np.min(prices)>0
    assert SOC_MIN-1e-7<=initial_soc<=SOC_MAX+1e-7
    floor=np.zeros(n) if lower_q is None else np.asarray(lower_q,float)
    assert floor.shape==net.shape and np.min(floor)>=-1e-8
    lower=np.r_[np.maximum(floor,0),np.zeros(3*n),np.full(n+1,SOC_MIN)]
    upper=np.r_[np.full(n,np.inf),np.full(2*n,FLOW_MAX),np.full(n,np.inf),np.full(n+1,SOC_MAX)]
    c=np.r_[prices,np.zeros(4*n+1)]
    rhs=np.r_[net,np.zeros(n),np.clip(initial_soc,SOC_MIN,SOC_MAX)]
    aeq=structure(n)
    res=linprog(c,A_eq=aeq,b_eq=rhs,bounds=np.column_stack([lower,upper]),
        method='highs',options={'primal_feasibility_tolerance':1e-8,'dual_feasibility_tolerance':1e-8})
    if not res.success:raise RuntimeError(f'LP failed: {res.message}')
    x=res.x;finite=np.isfinite(upper)
    dual=float(rhs@res.eqlin.marginals+lower@res.lower.marginals+
        upper[finite]@res.upper.marginals[finite])
    certificate=dict(balance_max=float(np.max(np.abs(aeq@x-rhs))),
        lower_violation=max(0.,float(np.max(lower-x))),
        upper_violation=max(0.,float(np.max(x[finite]-upper[finite]))),
        objective=float(c@x),duality_gap=abs(float(c@x)-dual))
    assert max(certificate[k] for k in ['balance_max','lower_violation','upper_violation','duality_gap'])<1e-5,certificate
    return np.maximum(x[:n],np.maximum(floor,0)),certificate


@lru_cache(maxsize=8)
def refund_structure(n, committed):
    total = 5*n+1+2*committed
    physical = hstack([structure(n), coo_matrix((2*n+1, 2*committed))], format='csr')
    slots = np.arange(committed)
    relation = coo_matrix((np.r_[np.ones(committed), -np.ones(committed), np.ones(committed)],
        (np.tile(slots, 3), np.r_[slots, 5*n+1+slots, 5*n+1+committed+slots])),
        shape=(committed, total)).tocsr()
    return vstack([physical, relation], format='csr')


def solve_refund(net, slot_prices, initial_soc, original_remainder):
    # 增购与取消同时为正会增加目标值，连续 LP 的最优解会自动互斥。
    net = np.asarray(net, float)
    prices = np.asarray(slot_prices, float)
    q0 = np.asarray(original_remainder, float)
    n, m = len(net), len(q0)
    assert 0 < m <= n and prices.shape == net.shape
    assert np.isfinite(net).all() and np.isfinite(prices).all() and np.min(prices) > 0
    assert q0.min() >= -1e-5 and SOC_MIN-1e-5 <= initial_soc <= SOC_MAX+1e-5
    q0 = np.maximum(q0, 0)
    lower = np.r_[np.zeros(4*n), np.full(n+1, SOC_MIN), np.zeros(2*m)]
    upper = np.r_[np.full(n, np.inf), np.full(2*n, FLOW_MAX), np.full(n, np.inf),
                  np.full(n+1, SOC_MAX), np.full(m, np.inf), q0]
    c = np.r_[np.zeros(m), prices[m:], np.zeros(4*n+1), UP_FACTOR*prices[:m], -REFUND_FRACTION*prices[:m]]
    rhs = np.r_[net, np.zeros(n), np.clip(initial_soc, SOC_MIN, SOC_MAX), q0]
    aeq = refund_structure(n, m)
    res = linprog(c, A_eq=aeq, b_eq=rhs, bounds=np.column_stack([lower, upper]),
                  method='highs', options={'primal_feasibility_tolerance': 1e-8,
                                           'dual_feasibility_tolerance': 1e-8})
    if not res.success:
        raise RuntimeError(f'Refund LP failed: {res.message}')
    x = res.x
    finite = np.isfinite(upper)
    dual = float(rhs @ res.eqlin.marginals + lower @ res.lower.marginals +
                 upper[finite] @ res.upper.marginals[finite])
    cert = dict(balance_max=float(np.max(np.abs(aeq @ x-rhs))),
                lower_violation=max(0., float(np.max(lower-x))),
                upper_violation=max(0., float(np.max(x[finite]-upper[finite]))),
                duality_gap=abs(float(c @ x)-dual), objective=float(c @ x),
                simultaneous_up_down=float(np.minimum(x[5*n+1:5*n+1+m], x[5*n+1+m:]).max()))
    assert max(cert[k] for k in cert if k != 'objective') < 1e-5, cert
    return np.maximum(x[:n], 0), cert
