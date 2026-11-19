"""求解确定性储能调度 LP，为情景规划生成候选并计算完美预见下界。"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


SOC_MIN = 1200.0
SOC_MAX = 10800.0
FLOW_MAX = 5000.0 / 6.0
TOL = 1e-6


@lru_cache(maxsize=16)
def _structure(n: int, eta: float, fixed_terminal: bool):
    t = np.arange(n)
    rows = [t, t, t, t, n + t, n + t, n + t, n + t,
            np.array([2 * n])]
    cols = [t, n + t, 2 * n + t, 3 * n + t, n + t, 2 * n + t,
            4 * n + t, 4 * n + t + 1, np.array([4 * n])]
    data = [np.ones(n), -np.ones(n), np.ones(n), -np.ones(n),
            np.full(n, -eta), np.full(n, 1.0 / eta), -np.ones(n),
            np.ones(n), np.ones(1)]
    if fixed_terminal:
        rows.append(np.array([2 * n + 1]))
        cols.append(np.array([5 * n]))
        data.append(np.ones(1))
    matrix = coo_matrix((np.concatenate(data),
                         (np.concatenate(rows), np.concatenate(cols))),
                        shape=(2 * n + 1 + int(fixed_terminal), 5 * n + 1)).tocsr()
    lower = np.concatenate([np.zeros(4 * n), np.full(n + 1, SOC_MIN)])
    upper = np.concatenate([np.full(n, np.inf), np.full(2 * n, FLOW_MAX),
                            np.full(n, np.inf), np.full(n + 1, SOC_MAX)])
    return matrix, lower, upper, np.column_stack([lower, upper])


def _max_abs(values) -> float:
    values = np.asarray(values)
    return float(np.max(np.abs(values))) if values.size else 0.0


def _soc_input(value: float, label: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value < SOC_MIN - TOL or value > SOC_MAX + TOL:
        raise ValueError(f"{label} must be within [{SOC_MIN}, {SOC_MAX}] kWh")
    return float(np.clip(value, SOC_MIN, SOC_MAX))


def solve_dispatch(net_kwh, prices, initial_soc=6000.0,
                   terminal_soc=None, eta=0.9) -> dict:
    net = np.asarray(net_kwh, dtype=float)
    price = np.asarray(prices, dtype=float)
    if net.ndim != 1 or not len(net) or price.shape != net.shape:
        raise ValueError("net_kwh and prices must be nonempty equal-length 1-D arrays")
    if not np.all(np.isfinite(net)) or not np.all(np.isfinite(price)):
        raise ValueError("net_kwh and prices must be finite")
    if np.any(price < 0):
        raise ValueError("Negative purchase prices need a bounded import model")
    eta = float(eta)
    if not 0 < eta <= 1:
        raise ValueError("eta must lie in (0, 1]")
    initial_soc = _soc_input(initial_soc, "initial_soc")
    if terminal_soc is not None:
        terminal_soc = _soc_input(terminal_soc, "terminal_soc")
    n = len(net)
    matrix, lower, upper, bounds = _structure(n, eta, terminal_soc is not None)
    rhs = np.zeros(matrix.shape[0])
    rhs[:n] = net
    rhs[2 * n] = initial_soc
    if terminal_soc is not None:
        rhs[-1] = terminal_soc
    objective = np.zeros(5 * n + 1)
    objective[:n] = price
    result = linprog(objective, A_eq=matrix, b_eq=rhs, bounds=bounds,
                     method="highs", options={"primal_feasibility_tolerance": 1e-8,
                                              "dual_feasibility_tolerance": 1e-8})
    if not result.success:
        raise RuntimeError(f"Dispatch LP failed (status {result.status}): {result.message}")

    raw = result.x
    # 原对偶与互补松弛证书只验证当前确定性 LP。
    y = np.asarray(result.eqlin.marginals)
    lm = np.asarray(result.lower.marginals)
    um = np.asarray(result.upper.marginals)
    finite_upper = np.isfinite(upper)
    dual_objective = float(rhs @ y + lower @ lm + upper[finite_upper] @ um[finite_upper])
    raw_cost = float(objective @ raw)
    dual_gap = abs(raw_cost - dual_objective)
    stationarity = _max_abs(objective - matrix.T @ y - lm - um)
    complementary = max(_max_abs(lm * (raw - lower)),
                         _max_abs(um[finite_upper] * (upper[finite_upper] - raw[finite_upper])))
    bound_violation = max(0.0, float(np.max(lower - raw)),
                          float(np.max(raw[finite_upper] - upper[finite_upper])))
    dual_sign_violation = max(0.0, float(np.max(-lm)), float(np.max(um)))

    q = raw[:n].copy()
    charge = raw[n:2 * n].copy()
    discharge = raw[2 * n:3 * n].copy()
    unused = raw[3 * n:4 * n].copy()
    soc = raw[4 * n:].copy()
    raw_simultaneous_count = int(np.sum((charge > TOL) & (discharge > TOL)))
    # 消除求解容差导致的同槽充放电循环，同时保持库存和购电费用不变。
    delta = np.minimum(charge, discharge / eta ** 2)
    charge -= delta
    discharge -= eta ** 2 * delta
    unused += (1.0 - eta ** 2) * delta
    for vector in (q, charge, discharge, unused):
        vector[np.abs(vector) < 1e-10] = 0.0
    cost = float(price @ q)
    balance = q + discharge - charge - unused - net
    dynamics = soc[1:] - soc[:-1] - eta * charge + discharge / eta
    energy_loss = (1.0 - eta) * charge + (1.0 / eta - 1.0) * discharge
    energy_identity = float(np.sum(q - net - unused) - (soc[-1] - soc[0]) - np.sum(energy_loss))
    terminal_error = 0.0 if terminal_soc is None else abs(float(soc[-1] - terminal_soc))
    verification = {
        "solver": "scipy.optimize.linprog(method='highs')",
        "scipy_version": scipy.__version__,
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "solver_iterations": int(result.nit),
        "n_slots": n,
        "eta_charge": eta,
        "eta_discharge": eta,
        "round_trip_efficiency": eta ** 2,
        "max_flow_kwh": FLOW_MAX,
        "raw_lp_objective_yuan": raw_cost,
        "raw_lp_dual_objective_yuan": dual_objective,
        "raw_lp_duality_gap_yuan": dual_gap,
        "raw_lp_relative_duality_gap": dual_gap / max(1.0, abs(raw_cost)),
        "raw_lp_max_equality_residual_kwh": _max_abs(matrix @ raw - rhs),
        "raw_lp_max_bound_violation_kwh": bound_violation,
        "raw_lp_kkt_stationarity_max": stationarity,
        "raw_lp_kkt_complementarity_max": complementary,
        "raw_lp_dual_sign_violation_max": dual_sign_violation,
        "raw_simultaneous_charge_discharge_slots": raw_simultaneous_count,
        "cycle_removed_charge_kwh": float(np.sum(delta)),
        "post_max_balance_residual_kwh": _max_abs(balance),
        "post_max_soc_recursion_residual_kwh": _max_abs(dynamics),
        "post_max_soc_bound_violation_kwh": max(0.0, float(np.max(SOC_MIN - soc)),
                                                  float(np.max(soc - SOC_MAX))),
        "post_max_nonnegativity_violation_kwh": max(0.0, -float(min(np.min(q), np.min(charge),
                                                                      np.min(discharge), np.min(unused)))),
        "post_max_flow_bound_violation_kwh": max(0.0, float(np.max(charge) - FLOW_MAX),
                                                   float(np.max(discharge) - FLOW_MAX)),
        "post_simultaneous_charge_discharge_slots": int(np.sum((charge > TOL) & (discharge > TOL))),
        "post_initial_soc_error_kwh": abs(float(soc[0] - initial_soc)),
        "post_terminal_soc_error_kwh": terminal_error,
        "post_objective_change_yuan": abs(cost - raw_cost),
        "post_energy_identity_residual_kwh": abs(energy_identity),
        "soc_min_kwh": float(np.min(soc)),
        "soc_max_kwh": float(np.max(soc)),
    }
    primal_checks = ["raw_lp_max_equality_residual_kwh", "raw_lp_max_bound_violation_kwh",
                     "post_max_balance_residual_kwh", "post_max_soc_recursion_residual_kwh",
                     "post_max_soc_bound_violation_kwh", "post_max_nonnegativity_violation_kwh",
                     "post_max_flow_bound_violation_kwh", "post_initial_soc_error_kwh",
                     "post_terminal_soc_error_kwh", "post_objective_change_yuan",
                     "post_energy_identity_residual_kwh"]
    verification["passed"] = bool(
        all(verification[key] <= TOL for key in primal_checks)
        and verification["post_simultaneous_charge_discharge_slots"] == 0
        and dual_gap <= 1e-5
        and stationarity <= TOL and complementary <= 1e-5 and dual_sign_violation <= TOL)
    if not verification["passed"]:
        raise AssertionError(f"Dispatch certificate failed: {json.dumps(verification)}")
    return {"q": q, "charge": charge, "discharge": discharge,
            "unused": unused, "soc": soc, "cost": cost,
            "verification": verification}
