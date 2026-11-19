"""执行第二问两日情景规划，完成误差构树、候选生成、因果反馈和字典序选优。"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib
import numpy as np
from scipy.spatial.distance import cdist
from q2_lp import solve_dispatch

ETA = 0.9
SOC_MIN = 1200.0
SOC_MAX = 10800.0
FLOW_MAX = 5000.0 / 6.0
EMERGENCY_MULTIPLIER = 5.0
LASSO_START = 14
FORMAL_START = 31
LOOKBACK_PAIRS = 90
FIRST_BRANCHES = 4
CHILDREN_PER_BRANCH = 3
ANCHOR_SOC = np.array([SOC_MIN, 6000.0, SOC_MAX])
RESERVE_STRENGTHS = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.0])
TERMINAL_VALUE = 0.0
TERMINAL_REFERENCE_SOC = 6000.0
TOL = 2e-5
PRIMARY_TIE_ATOL = 1e-6


@dataclass
class Branch:
    probability: float
    day1_net: np.ndarray
    day1_source: int | None
    next_forecast: np.ndarray
    child_nets: np.ndarray
    child_probabilities: np.ndarray
    child_sources: np.ndarray
    member_count: int


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def block_features(paths: np.ndarray) -> np.ndarray:
    paths = np.asarray(paths, dtype=float)
    if paths.ndim != 2 or paths.shape[1] != 144:
        raise ValueError("paths must have shape (n,144)")
    blocks = paths.reshape(len(paths), 6, 24)
    cumulative = np.cumsum(paths, axis=1)
    raw = np.concatenate([
        blocks.mean(axis=2),
        blocks.std(axis=2),
        blocks.max(axis=2),
        cumulative[:, [23, 47, 71, 95, 119, 143]],
    ], axis=1)
    scale = raw.std(axis=0)
    scale[scale < 1e-9] = 1.0
    return (raw - raw.mean(axis=0)) / scale


def kmedoids(paths: np.ndarray, groups: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths = np.asarray(paths, dtype=float)
    n = len(paths)
    if n == 0:
        raise ValueError("empty scenario bank")
    groups = max(1, min(int(groups), n))
    feat = block_features(paths)
    groups = min(groups, len(np.unique(feat, axis=0)))
    dist = cdist(feat, feat, metric="sqeuclidean")
    medoids = [int(np.argmin(dist.sum(axis=1)))]
    while len(medoids) < groups:
        nearest = np.min(dist[:, medoids], axis=1)
        nearest[medoids] = -1.0
        medoids.append(int(np.argmax(nearest)))
    medoids = np.asarray(medoids, dtype=int)
    for _ in range(30):
        assignment = np.argmin(dist[:, medoids], axis=1)
        updated = medoids.copy()
        for group in range(groups):
            members = np.flatnonzero(assignment == group)
            if len(members):
                within = dist[np.ix_(members, members)].sum(axis=1)
                updated[group] = int(members[np.argmin(within)])
        if np.array_equal(updated, medoids):
            break
        medoids = updated
    assignment = np.argmin(dist[:, medoids], axis=1)
    probabilities = np.bincount(assignment, minlength=groups).astype(float) / n
    if np.any(probabilities <= 0) or abs(probabilities.sum() - 1.0) > 1e-12:
        raise AssertionError("invalid cluster probabilities")
    return medoids, assignment, probabilities


def build_tree(
    origin: int,
    actual_net: np.ndarray,
    prediction_net: np.ndarray,
    first_branches: int = FIRST_BRANCHES,
    children_per_branch: int = CHILDREN_PER_BRANCH,
    lookback_pairs: int = LOOKBACK_PAIRS,
) -> tuple[list[Branch], dict[str, Any]]:
    pair_stop = origin - 1
    pair_start = max(LASSO_START, pair_stop - lookback_pairs)
    pair_origins = np.arange(pair_start, pair_stop, dtype=int)
    base1 = prediction_net[origin, 0]
    base2 = prediction_net[origin, 1]
    if len(pair_origins) == 0:
        branch = Branch(
            probability=1.0,
            day1_net=base1.copy(),
            day1_source=None,
            next_forecast=base2.copy(),
            child_nets=base2[None, :].copy(),
            child_probabilities=np.array([1.0]),
            child_sources=np.array([-1], dtype=int),
            member_count=1,
        )
        return [branch], {
            "pair_count": 0,
            "earliest_pair_origin": None,
            "latest_pair_origin": None,
            "latest_target_index_used": None,
            "all_targets_strictly_before_origin": True,
        }

    eps1 = actual_net[pair_origins] - prediction_net[pair_origins, 0]
    revision = prediction_net[pair_origins + 1, 0] - prediction_net[pair_origins, 1]
    eps2 = actual_net[pair_origins + 1] - prediction_net[pair_origins + 1, 0]
    first_medoid, assignment, first_prob = kmedoids(eps1, first_branches)
    branches: list[Branch] = []
    for group, medoid_local in enumerate(first_medoid):
        members = np.flatnonzero(assignment == group)
        member_origins = pair_origins[members]
        # 第一层分簇只使用首日误差，避免次日信息反向影响首日决策。
        representative_revision = revision[medoid_local]
        next_forecast = base2 + representative_revision
        combined_child_error = revision[members] + eps2[members] - representative_revision
        child_medoid, child_assignment, child_prob = kmedoids(
            combined_child_error, min(children_per_branch, len(members))
        )
        child_local = members[child_medoid]
        child_nets = base2[None, :] + revision[child_local] + eps2[child_local]
        branches.append(Branch(
            probability=float(first_prob[group]),
            day1_net=base1 + eps1[medoid_local],
            day1_source=int(pair_origins[medoid_local]),
            next_forecast=next_forecast,
            child_nets=child_nets,
            child_probabilities=child_prob,
            child_sources=pair_origins[child_local].astype(int),
            member_count=int(len(members)),
        ))
    if abs(sum(branch.probability for branch in branches) - 1.0) > 1e-12:
        raise AssertionError("first-day branch probabilities do not sum to one")
    for branch in branches:
        if abs(branch.child_probabilities.sum() - 1.0) > 1e-12:
            raise AssertionError("conditional probabilities do not sum to one")
    return branches, {
        "pair_count": int(len(pair_origins)),
        "earliest_pair_origin": int(pair_origins[0]),
        "latest_pair_origin": int(pair_origins[-1]),
        "latest_target_index_used": int(pair_origins[-1] + 1),
        "all_targets_strictly_before_origin": bool(pair_origins[-1] + 1 < origin),
    }


def unique_plans(plans: list[np.ndarray], digits: int = 7) -> np.ndarray:
    if not plans:
        raise ValueError("candidate generator produced no plans")
    array = np.asarray(plans, dtype=float)
    rounded = np.round(array, digits)
    _, indices = np.unique(rounded, axis=0, return_index=True)
    return array[np.sort(indices)]


def design_paths(base: np.ndarray, scenarios: np.ndarray, probabilities: np.ndarray) -> list[np.ndarray]:
    scenarios = np.asarray(scenarios, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    paths = [np.asarray(base, dtype=float)]
    paths.append(np.average(scenarios, axis=0, weights=probabilities))
    paths.extend(scenarios)
    cumulative_deficit = np.max(np.cumsum(scenarios - base[None, :], axis=1), axis=1)
    four_hour = (scenarios - base[None, :]).reshape(len(scenarios), 6, 24).sum(axis=2).max(axis=1)
    paths.append(scenarios[int(np.argmax(cumulative_deficit))])
    paths.append(scenarios[int(np.argmax(four_hour))])
    result = unique_plans(paths, digits=9)
    return [row for row in result]


def generate_q1_candidates(
    base: np.ndarray,
    branches: list[Branch],
    prices: np.ndarray,
    initial_soc: float,
) -> np.ndarray:
    nets = np.asarray([branch.day1_net for branch in branches])
    probs = np.asarray([branch.probability for branch in branches])
    designs = design_paths(base, nets, probs)
    targets = np.unique(np.clip([SOC_MIN, initial_soc, 6000.0, SOC_MAX], SOC_MIN, SOC_MAX))
    candidates: list[np.ndarray] = []
    for net in designs:
        candidates.append(solve_dispatch(net, prices, initial_soc, None)["q"])
        for target in targets:
            candidates.append(solve_dispatch(net, prices, initial_soc, float(target))["q"])
    return unique_plans(candidates)


def generate_q2_candidates(branch: Branch, prices: np.ndarray) -> np.ndarray:
    designs = design_paths(
        branch.next_forecast,
        branch.child_nets,
        branch.child_probabilities,
    )
    candidates: list[np.ndarray] = []
    anchor_pairs: list[tuple[float, float | None]] = [
        (SOC_MIN, None),
        (6000.0, None),
        (SOC_MAX, None),
        (6000.0, 6000.0),
        (6000.0, SOC_MAX),
    ]
    for net in designs:
        for initial, terminal in anchor_pairs:
            candidates.append(solve_dispatch(net, prices, initial, terminal)["q"])
    return unique_plans(candidates)


def simulate_feedback_matrix(
    plans: np.ndarray,
    nets: np.ndarray,
    initial_soc: float | np.ndarray,
    prices: np.ndarray,
    reserve_strength: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    plans = np.asarray(plans, dtype=float)
    nets = np.asarray(nets, dtype=float)
    if plans.ndim != 2 or nets.ndim != 2 or plans.shape[1] != 144 or nets.shape[1] != 144:
        raise ValueError("plans and nets must have 144 columns")
    p_count, n_count = len(plans), len(nets)
    if reserve_strength is None:
        reserve_strength = np.zeros(p_count, dtype=float)
    reserve_strength = np.asarray(reserve_strength, dtype=float)
    if reserve_strength.shape != (p_count,) or np.any((reserve_strength < 0) | (reserve_strength > 1)):
        raise ValueError("reserve_strength must have one value in [0,1] per plan")
    future_peak = np.maximum.accumulate(prices[::-1])[::-1]
    spread = max(float(prices.max() - prices.min()), 1e-12)
    opportunity = np.maximum(future_peak - prices, 0.0) / spread
    reserve = SOC_MIN + reserve_strength[:, None] * (SOC_MAX - SOC_MIN) * opportunity[None, :]
    state0 = np.asarray(initial_soc, dtype=float)
    if state0.ndim == 0:
        soc = np.full((p_count, n_count), float(state0))
    else:
        if state0.shape != (p_count,):
            raise ValueError("vector initial_soc must have one value per plan")
        soc = np.broadcast_to(state0[:, None], (p_count, n_count)).copy()
    emergency_cost = np.zeros((p_count, n_count), dtype=float)
    for k in range(144):
        residual = plans[:, None, k] - nets[None, :, k]
        surplus = np.maximum(residual, 0.0)
        deficit = np.maximum(-residual, 0.0)
        charge = np.minimum(np.minimum(surplus, FLOW_MAX), np.maximum((SOC_MAX - soc) / ETA, 0.0))
        discharge = np.minimum(np.minimum(deficit, FLOW_MAX), np.maximum(ETA * (soc - reserve[:, None, k]), 0.0))
        emergency = deficit - discharge
        soc += ETA * charge - discharge / ETA
        emergency_cost += EMERGENCY_MULTIPLIER * prices[k] * emergency
    return emergency_cost, soc


def next_day_value_scalar(
    initial_states: np.ndarray,
    q2_plans: np.ndarray,
    branch: Branch,
    prices: np.ndarray,
    terminal_value: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    initial_states = np.asarray(initial_states, dtype=float)
    option_plans = np.repeat(q2_plans, len(RESERVE_STRENGTHS), axis=0)
    option_reserve = np.tile(RESERVE_STRENGTHS, len(q2_plans))
    normal = option_plans @ prices
    values = np.empty(len(initial_states), dtype=float)
    selected = np.empty(len(initial_states), dtype=int)
    selected_terminal = np.empty(len(initial_states), dtype=float)
    for i, state in enumerate(initial_states):
        emergency, terminal = simulate_feedback_matrix(
            option_plans, branch.child_nets, float(state), prices, option_reserve
        )
        expected_terminal = terminal @ branch.child_probabilities
        terminal_penalty = (
            np.maximum(TERMINAL_REFERENCE_SOC - terminal, 0.0)
            @ branch.child_probabilities
        )
        expected = normal + emergency @ branch.child_probabilities + terminal_value * terminal_penalty
        selected[i] = int(np.argmin(expected))
        values[i] = float(expected[selected[i]])
        selected_terminal[i] = float(expected_terminal[selected[i]])
    return values, selected, selected_terminal


def evaluate_origin(
    origin: int,
    state: float,
    actual_net: np.ndarray,
    prediction_net: np.ndarray,
    prices: np.ndarray,
    first_branches: int = FIRST_BRANCHES,
    children_per_branch: int = CHILDREN_PER_BRANCH,
    lookback_pairs: int = LOOKBACK_PAIRS,
    include_continuation: bool = True,
    terminal_value: float = TERMINAL_VALUE,
    next_prices: np.ndarray | None = None,
) -> dict[str, Any]:
    if len(actual_net) != origin or len(prediction_net) != origin + 1:
        raise ValueError('decision input must end at observed yesterday / released today')
    if terminal_value != 0.0:
        raise ValueError('daily-priority objective excludes terminal inventory value')
    next_prices = prices if next_prices is None else np.asarray(next_prices, float)
    assert np.shape(prices) == (144,) and next_prices.shape == (144,)
    branches, tree_audit = build_tree(
        origin, actual_net, prediction_net,
        first_branches, children_per_branch, lookback_pairs,
    )
    q1_base_plans = generate_q1_candidates(prediction_net[origin, 0], branches, prices, state)
    q1_plans = np.repeat(q1_base_plans, len(RESERVE_STRENGTHS), axis=0)
    q1_reserve = np.tile(RESERVE_STRENGTHS, len(q1_base_plans))
    day1_nets = np.asarray([branch.day1_net for branch in branches])
    first_prob = np.asarray([branch.probability for branch in branches])
    emergency1, terminal1 = simulate_feedback_matrix(
        q1_plans, day1_nets, state, prices, q1_reserve
    )
    normal1 = q1_plans @ prices
    primary = normal1 + emergency1 @ first_prob
    primary_minimum = float(np.min(primary))
    eligible = np.flatnonzero(primary <= primary_minimum + PRIMARY_TIE_ATOL)
    primary_only_selected = int(np.argmin(primary))
    # 首日所有情景共用一个计划，次日计划只能随已实现的首日分支变化。
    continuation = np.zeros_like(emergency1)
    q2_selected = np.full(emergency1.shape, -1, dtype=int)
    q2_expected_terminal = np.zeros_like(emergency1)
    q2_counts: list[int] = []
    if include_continuation:
        for b, branch in enumerate(branches):
            q2_plans = generate_q2_candidates(branch, next_prices)
            q2_counts.append(int(len(q2_plans)))
            continuation[eligible, b], q2_selected[eligible, b], q2_expected_terminal[eligible, b] = next_day_value(
                terminal1[eligible, b], q2_plans, branch, next_prices, 0.0
            )
    else:
        q2_counts = [0 for _ in branches]
    # 第二级只在当日同优集合内比较，不能用加权和替代字典序。
    secondary = np.full(len(primary), np.inf)
    secondary[eligible] = continuation[eligible] @ first_prob
    selected = int(eligible[np.argmin(secondary[eligible])])
    primary_gap = float(primary[selected] - primary_minimum)
    assert primary_gap <= PRIMARY_TIE_ATOL + 1e-9
    order = np.argsort(primary, kind='stable')
    second = float(primary[order[1]]) if len(order)>1 else primary_minimum
    return {
        "plan": q1_plans[selected].copy(),
        "reserve_strength": float(q1_reserve[selected]),
        "objective_yuan": float(primary[selected]),
        "primary_minimum_yuan": primary_minimum,
        "selected_primary_gap_yuan": primary_gap,
        "primary_tie_tolerance_yuan": PRIMARY_TIE_ATOL,
        "primary_tie_count": int(len(eligible)),
        "primary_only_selected": primary_only_selected,
        "secondary_changed_candidate": bool(selected != primary_only_selected),
        "secondary_changed_plan": bool(np.max(np.abs(q1_plans[selected]-q1_plans[primary_only_selected]))>1e-7),
        "secondary_changed_feedback": bool(q1_reserve[selected]!=q1_reserve[primary_only_selected]),
        "expected_day1_emergency_yuan": float(emergency1[selected] @ first_prob),
        "expected_continuation_yuan": float(continuation[selected] @ first_prob),
        "base_plan_candidate_count": int(len(q1_base_plans)),
        "candidate_count": int(len(q1_plans)),
        "selected_candidate": selected,
        "second_best_gap_yuan": second - primary_minimum,
        "branch_count": int(len(branches)),
        "child_count_total": int(sum(len(branch.child_nets) for branch in branches)),
        "q2_candidate_counts": q2_counts,
        "selected_q2_candidate_by_branch": q2_selected[selected].tolist(),
        "expected_terminal_after_q2_kwh": float(q2_expected_terminal[selected] @ first_prob),
        "terminal_value_yuan_per_internal_kwh": float(terminal_value),
        "tree_audit": tree_audit,
        "branches": branches,
        "certificate": dict(plans=q1_plans, reserve_strengths=q1_reserve,
            day1_nets=day1_nets, probabilities=first_prob, normal_costs=normal1,
            emergency_costs=emergency1, day1_end_soc=terminal1, primary_costs=primary,
            eligible=eligible, secondary_costs=secondary, selected=np.array(selected),
            initial_soc=np.array(state), primary_tie_atol=np.array(PRIMARY_TIE_ATOL)),
    }


def execute_actual(
    plan: np.ndarray,
    net: np.ndarray,
    initial_soc: float,
    prices: np.ndarray,
    reserve_strength: float,
) -> dict[str, np.ndarray]:
    plan = np.asarray(plan, dtype=float)
    net = np.asarray(net, dtype=float)
    soc = np.empty(145); soc[0] = initial_soc
    charge = np.zeros(144); discharge = np.zeros(144)
    emergency = np.zeros(144); unused = np.zeros(144)
    future_peak = np.maximum.accumulate(prices[::-1])[::-1]
    spread = max(float(prices.max() - prices.min()), 1e-12)
    opportunity = np.maximum(future_peak - prices, 0.0) / spread
    reserve = SOC_MIN + reserve_strength * (SOC_MAX - SOC_MIN) * opportunity
    for k in range(144):
        residual = plan[k] - net[k]
        if residual >= 0:
            charge[k] = min(residual, FLOW_MAX, max((SOC_MAX - soc[k]) / ETA, 0.0))
            unused[k] = residual - charge[k]
        else:
            discharge[k] = min(-residual, FLOW_MAX, max(ETA * (soc[k] - reserve[k]), 0.0))
            emergency[k] = -residual - discharge[k]
        soc[k + 1] = soc[k] + ETA * charge[k] - discharge[k] / ETA
    return {
        "charge": charge,
        "discharge": discharge,
        "emergency": emergency,
        "unused": unused,
        "soc": soc,
        "reserve_floor": reserve,
    }



def next_day_value(initial_states, q2_plans, branch, prices, terminal_value):
    states, reverse = np.unique(np.asarray(initial_states, dtype=float), return_inverse=True)
    options = np.repeat(q2_plans, len(RESERVE_STRENGTHS), axis=0)
    strengths = np.tile(RESERVE_STRENGTHS, len(q2_plans))
    normal = options @ prices
    p = len(options)
    values = np.empty(len(states)); selected = np.empty(len(states), dtype=int)
    endings = np.empty(len(states))
    for lo in range(0, len(states), 16):
        s = states[lo:lo+16]
        ecost, terminal = simulate_feedback_matrix(
            np.tile(options, (len(s),1)), branch.child_nets,
            np.repeat(s,p), prices, np.tile(strengths,len(s)))
        expected = (ecost @ branch.child_probabilities + terminal_value *
            (np.maximum(TERMINAL_REFERENCE_SOC-terminal,0.) @ branch.child_probabilities)
            ).reshape(len(s),p) + normal[None,:]
        idx = np.argmin(expected,axis=1)
        values[lo:lo+len(s)] = expected[np.arange(len(s)),idx]
        selected[lo:lo+len(s)] = idx
        endings[lo:lo+len(s)] = (terminal @ branch.child_probabilities).reshape(len(s),p)[np.arange(len(s)),idx]
    return values[reverse], selected[reverse], endings[reverse]
