"""训练问题二的负荷 Ridge 与光伏 Lasso 模型，并执行因果滚动预测。
模型统一使用 62 列时序特征。"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import warnings

for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(key, '1')
ROOT = Path(__file__).resolve().parents[1]
DROP = ['pv_previous_day_last', 'weekday_6', 'load_mean_3', 'pv_mean_3']
GRID = {'load': [.1, 1., 10., 100., 1000.], 'pv': [.01, .03, .1, .3, 1., 3., 10.]}


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    def encode(x):
        if hasattr(x, 'tolist'): return x.tolist()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=encode), encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def historical_baseline(history):
    d = len(history)
    pred = np.zeros((2, 144, 2))
    if d == 0: return pred
    load_delta = history[7:, :, 0] - history[:-7, :, 0]
    ld = np.median(load_delta[-7:], axis=0) if len(load_delta) else np.zeros(144)
    pv_delta = np.diff(history[:, :, 1], axis=0)
    vd = np.median(pv_delta[-7:], axis=0) if len(pv_delta) else np.zeros(144)
    for h in range(2):
        base = history[d+h-7, :, 0] if d+h >= 7 else np.median(history[-3:, :, 0], axis=0)
        pred[h, :, 0] = np.maximum(0., base + ld)
        pred[h, :, 1] = np.maximum(0., history[-1, :, 1] + (h+1)*vd)
    return pred


def features(history, date):
    d = len(history)
    assert d >= 7
    baseline = historical_baseline(history)
    matrices = []
    for h in range(2):
        fields = {}
        def add(name, value):
            value = np.asarray(value)
            fields[name] = np.full(144, float(value)) if value.ndim == 0 else value
        slot = np.arange(144)
        target_date = pd.Timestamp(date) + pd.Timedelta(days=h)
        add('horizon_day', h+1)
        for f in (1, 2):
            add(f'slot_sin_{f}', np.sin(2*np.pi*f*slot/144))
            add(f'slot_cos_{f}', np.cos(2*np.pi*f*slot/144))
        for w in range(7): add(f'weekday_{w}', int(target_date.dayofweek == w))
        add('year_sin', np.sin(2*np.pi*(target_date.dayofyear-1)/365.25))
        add('year_cos', np.cos(2*np.pi*(target_date.dayofyear-1)/365.25))
        for v, name in enumerate(('load', 'pv')):
            add(f'{name}_baseline', baseline[h, :, v])
            for lag in (1, 2, 3, 7, 14, 21, 28):
                value = history[-lag, :, v] if d >= lag else np.median(history[:, :, v], axis=0)
                add(f'{name}_origin_lag_{lag}', value)
            for lag in (7, 14):
                idx = d+h-lag
                assert idx < d
                value = history[idx, :, v] if idx >= 0 else np.median(history[:, :, v], axis=0)
                add(f'{name}_target_lag_{lag}', value)
            for window in (3, 7, 14, 28):
                a = history[-window:, :, v]
                add(f'{name}_mean_{window}', a.mean(axis=0))
                add(f'{name}_median_{window}', np.median(a, axis=0))
                add(f'{name}_std_{window}', a.std(axis=0))
            for name_stat, value in [('mean', history[-1, :, v].mean()), ('max', history[-1, :, v].max()),
                                      ('std', history[-1, :, v].std()), ('last', history[-1, -1, v])]:
                add(f'{name}_previous_day_{name_stat}', value)
        names = list(fields)
        matrices.append(np.column_stack(list(fields.values())))
    return np.asarray(matrices, dtype=np.float32), names


def training(data, day):
    oo, hh = np.meshgrid(np.arange(max(7, day-91), day), np.arange(2), indexing='ij')
    oo, hh = oo.ravel(), hh.ravel()
    target = oo+hh
    valid = (target < day) & (target >= day-90)
    oo, hh, target = oo[valid], hh[valid], target[valid]
    x = data['features'][oo, hh].reshape(-1, 66).astype(np.float64)
    y = (data['actual'][target] - data['baseline'][oo, hh]).reshape(-1, 2)
    return x, y, {'fit_origin': day, 'fit_date': str(data['dates'][day]),
                   'earliest_training_target_day0': int(target.min()),
                   'latest_training_target_day0': int(target.max()),
                   'training_rows': len(x), 'training_target_days': int(target.max()-target.min()+1),
                   'training_targets_strictly_past': bool(np.all(target < day))}


def fit_one(x, y, variable, alpha, names):
    keep = np.array([j for j, name in enumerate(names) if name not in DROP])
    sd = x[:, keep].std(axis=0)
    constants = keep[sd <= 1e-12]
    keep = keep[sd > 1e-12]
    scaler = StandardScaler().fit(x[:, keep])
    xx = np.asfortranarray(scaler.transform(x[:, keep]))
    attempts = []
    for iterations in ([1] if variable == 'load' else [100000, 2000000]):
        model = Ridge(alpha=alpha, solver='auto') if variable == 'load' else Lasso(
            alpha=alpha, precompute=True, tol=1e-8, max_iter=iterations, selection='cyclic')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model.fit(xx, y)
        residual = y-model.predict(xx)
        gradient = xx.T @ residual / len(y)
        diag = {'intercept_gradient': float(abs(residual.mean())),
                'warnings': [str(w.message) for w in caught], 'n_iter': int(getattr(model, 'n_iter_', 0) or 0)}
        if variable == 'load':
            g = gradient-alpha/len(y)*model.coef_
            scale = max(1., float(np.max(abs(xx.T@y/len(y)))))
            diag.update(normal_equation_residual_max=float(np.max(abs(g))),
                        normal_equation_residual_relative=float(np.max(abs(g))/scale),
                        objective=float(np.sum(residual**2)+alpha*np.sum(model.coef_**2)))
            passed = diag['normal_equation_residual_relative'] < 1e-7 and diag['intercept_gradient'] < 1e-7
        else:
            active = model.coef_ != 0
            kkt = max(np.max(abs(gradient[active]-alpha*np.sign(model.coef_[active])), initial=0),
                      np.max(np.maximum(abs(gradient[~active])-alpha, 0), initial=0))
            diag.update(kkt_max_violation=float(kkt), dual_gap=float(model.dual_gap_),
                        dual_gap_relative=float(model.dual_gap_/max(np.var(y), 1e-30)),
                        objective=float(np.mean(residual**2)/2+alpha*np.sum(abs(model.coef_))))
            passed = kkt <= 1e-5*max(1., alpha) and abs(diag['dual_gap_relative']) <= 1.1e-8 and diag['intercept_gradient'] < 1e-8 and not caught
        diag['passed'] = passed
        attempts.append(diag)
        if passed: break
    if not passed: raise RuntimeError(f'Unconverged {variable} alpha={alpha}: {attempts}')
    snapshot = {'variable': variable, 'alpha': alpha, 'keep_original_indices': keep.tolist(),
                'feature_names': [names[j] for j in keep], 'constant_removed_features': [names[j] for j in constants],
                'fixed_removed_features': DROP, 'retained_count': len(keep),
                'scaler_mean': scaler.mean_, 'scaler_scale': scaler.scale_, 'scaler_variance': scaler.var_,
                'coefficients_standardized': model.coef_, 'intercept': float(model.intercept_),
                'coefficients_raw': model.coef_/scaler.scale_,
                'intercept_raw': float(model.intercept_-(model.coef_/scaler.scale_)@scaler.mean_),
                'solver': diag, 'solver_attempts': attempts}
    return scaler, model, keep, snapshot


def period(data, names, variable, alpha, start, end):
    v = {'load': 0, 'pv': 1}[variable]
    pred = data['baseline'][..., v].copy()
    logs = []
    began = time.perf_counter()
    for day in range(start, end):
        if (day-start) % 7 == 0:
            x, y, log = training(data, day)
            scaler, model, keep, snapshot = fit_one(x, y[:, v], variable, alpha, names)
            log.update(snapshot)
            logs.append(log)
        raw = data['features'][day].reshape(-1, 66).astype(np.float64)
        correction = model.predict(scaler.transform(raw[:, keep]))
        rebuilt = raw[:, keep]@snapshot['coefficients_raw']+snapshot['intercept_raw']
        assert np.max(abs(correction-rebuilt)) < 1e-7
        pred[day] = np.maximum(0., data['baseline'][day, :, :, v]+correction.reshape(2, 144))
        if start == 31 and ((day-start+1) % 56 == 0 or day == end-1):
            print(json.dumps({'variable': variable, 'formal_completed_origin': day, 'seconds': time.perf_counter()-began}), flush=True)
    return pred, logs


def metric_values(error):
    error = np.asarray(error).ravel()
    return {'n': len(error), 'rmse_kw': float(np.sqrt(np.mean(error**2))),
            'mae_kw': float(np.mean(abs(error))), 'bias_prediction_minus_actual_kw': float(error.mean())}


def calibration_metrics(pred, actual, variable):
    v = {'load': 0, 'pv': 1}[variable]
    errors = []
    by_horizon = []
    for h in (0, 1):
        targets = np.arange(14, 31)
        targets = targets[targets-h >= 14]
        error = 6*(pred[targets-h, h]-actual[targets, :, v])
        errors.extend(error.ravel())
        by_horizon.append({'horizon_days': h+1, **metric_values(error)})
    return {**metric_values(errors), 'by_horizon': by_horizon}


def formal_metrics(pred, actual, label):
    rows, daily = [], []
    days = np.arange(31, 365)
    pooled = {}
    for h in (0, 1):
        a, p = actual[days]*6, pred[days-h, h]*6
        for v, variable in enumerate(('load', 'pv', 'net')):
            error = p[:, :, v]-a[:, :, v] if v < 2 else (p[:, :, 0]-p[:, :, 1])-(a[:, :, 0]-a[:, :, 1])
            scopes = [('all_slots', error)]
            if variable == 'pv': scopes.append(('daytime_06_18', error[:, 36:108]))
            for scope, e in scopes:
                rows.append({'model': label, 'horizon_days': h+1, 'variable': variable, 'scope': scope, **metric_values(e)})
                pooled.setdefault((variable, scope), []).extend(e.ravel())
            for day, e in zip(days, error):
                daily.append({'model': label, 'target_day0': day, 'horizon_days': h+1, 'variable': variable, **metric_values(e)})
    for (variable, scope), error in pooled.items():
        rows.append({'model': label, 'horizon_days': 'pooled', 'variable': variable, 'scope': scope, **metric_values(error)})
    return rows, daily


def audit_inputs(data, names):
    checks = []
    for day in range(365):
        assert np.array_equal(historical_baseline(data['actual'][:day]), data['baseline'][day])
        if day >= 7:
            xx, nn = features(data['actual'][:day], data['dates'][day])
            assert nn == names and np.array_equal(xx, data['features'][day])
    for day in (14, 31, 171, 353):
        altered = {k: v.copy() for k, v in data.items()}
        altered['actual'][day:] = 123456.789
        x, y, log = training(data, day)
        xa, ya, la = training(altered, day)
        changed_features, _ = features(altered['actual'][:day], data['dates'][day])
        assert np.array_equal(x, xa) and np.array_equal(y, ya) and log == la
        assert np.array_equal(changed_features, data['features'][day])
        checks.append({'origin': day, 'training_and_features_unchanged_by_future_mutation': True})
    dump(ROOT/'qa/forecast_input_audit.json', {'passed': True, 'all_365_baselines_identical': True,
         'all_358_feature_designs_identical': True, 'future_mutation_checks': checks})


def rebuild_inputs_from_actual_csv():
    frame = pd.read_csv(ROOT/'data/source/actual.csv')
    assert len(frame) == 365*144
    assert np.array_equal(frame['slot'].to_numpy().reshape(365, 144), np.tile(np.arange(1, 145), (365, 1)))
    dates = pd.to_datetime(frame['sample_date']).drop_duplicates().to_numpy().astype('datetime64[D]')
    assert len(dates) == 365 and np.all(np.diff(dates).astype(int) == 1)
    actual = frame[['load_kw', 'pv_kw']].to_numpy().reshape(365, 144, 2)/6.
    baseline = np.array([historical_baseline(actual[:day]) for day in range(365)])
    feature_bank = np.zeros((365, 2, 144, 66), dtype=np.float32)
    for day in range(7, 365): feature_bank[day] = features(actual[:day], dates[day])[0]
    rebuilt = {'actual': actual, 'dates': dates, 'baseline': baseline, 'features': feature_bank}
    with np.load(ROOT/'data/source/prepared.npz', allow_pickle=False) as frozen:
        for key, value in rebuilt.items(): assert np.array_equal(value, frozen[key]), key
    return rebuilt


def main(check_inputs_only=False):
    data = rebuild_inputs_from_actual_csv()
    names = json.loads((ROOT/'data/source/protocol.json').read_text(encoding='utf-8'))['features']
    assert len(names) == 66
    for folder in ('qa/calibration', 'metadata', 'results', 'data'):
        (ROOT/folder).mkdir(parents=True, exist_ok=True)
    audit_inputs(data, names)
    if check_inputs_only:
        print('All inputs rebuilt from actual.csv and matched frozen source snapshots.', flush=True)
        return
    manifest = json.loads((ROOT/'data/source_manifest.json').read_text(encoding='utf-8'))
    assert all(digest(ROOT/row['copied_path']) == row['sha256'] for row in manifest['sources'])
    selection = {'retrospective_extension': True, 'calibration_targets': ['2025-01-15', '2025-01-31'],
        'calibration_origin_min_day0': 14, 'selection_rule': 'January pooled per-component RMSE; grid order resolves exact ties',
        'candidate_grids': GRID, 'fixed_drops': DROP, 'nighttime_gate': 'unchanged: only clip components nonnegative',
        'january_published_predictions': 'all origins 0..30 retain current complete-paper baseline exactly',
        'formal_first_origin_day0': 31, 'max_completed_target_days': 90, 'refit_every_days': 7,
        'versions': {'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__},
        'components': {}}
    with threadpool_limits(limits=4):
        for variable in ('load', 'pv'):
            candidates = []
            for index, alpha in enumerate(GRID[variable]):
                pred, logs = period(data, names, variable, alpha, 14, 31)
                metrics = calibration_metrics(pred, data['actual'], variable)
                file = f'qa/calibration/{variable}_alpha_{alpha:g}.npz'
                np.savez_compressed(ROOT/file, predictions=pred[:31], dates=data['dates'][:31], variable=variable, units='kWh')
                dump(ROOT/f'metadata/calibration_{variable}_alpha_{alpha:g}.json', {'fit_logs': logs})
                candidates.append({'candidate_index': index, 'alpha': alpha, 'metrics': metrics, 'prediction_file': file})
                print(json.dumps({'calibration_variable': variable, 'alpha': alpha, **metrics}), flush=True)
            best = min(candidates, key=lambda c: (c['metrics']['rmse_kw'], c['candidate_index']))
            selection['components'][variable] = {'selected': best, 'candidates': candidates}
        dump(ROOT/'selection.json', selection)
        final = data['baseline'].copy()
        all_logs = []
        for v, variable in enumerate(('load', 'pv')):
            alpha = selection['components'][variable]['selected']['alpha']
            pred, logs = period(data, names, variable, alpha, 31, 365)
            final[..., v] = pred
            all_logs.extend(logs)
            dump(ROOT/f'metadata/formal_{variable}.json', {'variable': variable, 'alpha': alpha, 'fit_logs': logs})
            assert len(logs) == 48
    reference = np.load(ROOT/'data/source/current_baseline.npz')['predictions']
    assert np.array_equal(final[:31], reference[:31])
    assert np.isfinite(final).all() and (final >= 0).all()
    np.savez_compressed(ROOT/'data/forecast_component_selected.npz', predictions=final, dates=data['dates'],
                        units='kWh per 10-min slot', model='62_features_load_Ridge_pv_Lasso',
                        alpha_load=selection['components']['load']['selected']['alpha'],
                        alpha_pv=selection['components']['pv']['selected']['alpha'])
    coefficients = []
    for fit in all_logs:
        for j, feature in enumerate(fit['feature_names']):
            coefficients.append({'variable': fit['variable'], 'fit_origin': fit['fit_origin'], 'fit_date': fit['fit_date'],
                'alpha': fit['alpha'], 'feature': feature, 'coefficient_standardized_kwh': fit['coefficients_standardized'][j],
                'coefficient_raw': fit['coefficients_raw'][j], 'intercept': fit['intercept'], 'intercept_raw': fit['intercept_raw'],
                'scaler_mean': fit['scaler_mean'][j], 'scaler_scale': fit['scaler_scale'][j], 'scaler_variance': fit['scaler_variance'][j]})
    pd.DataFrame(coefficients).to_csv(ROOT/'results/forecast_coefficients.csv', index=False, encoding='utf-8-sig')
    rows, daily = [], []
    for label, file in [('current_paper_lasso66', 'data/source/current_lasso.npz'),
                         ('reference_lasso62', 'data/source/structural_lasso.npz'),
                         ('component_selected62', 'data/forecast_component_selected.npz')]:
        rr, dd = formal_metrics(np.load(ROOT/file)['predictions'], data['actual'], label)
        rows.extend(rr); daily.extend(dd)
    pd.DataFrame(rows).to_csv(ROOT/'results/forecast_metrics.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(daily).to_csv(ROOT/'results/daily_forecast_metrics.csv', index=False, encoding='utf-8-sig')
    dump(ROOT/'qa/forecast_training_audit.json', {'passed': True, 'formal_component_fits': len(all_logs),
        'all_training_targets_strictly_past': all(log['training_targets_strictly_past'] for log in all_logs),
        'all_solver_checks_passed': all(log['solver']['passed'] for log in all_logs),
        'max_retained_count': max(log['retained_count'] for log in all_logs),
        'min_retained_count': min(log['retained_count'] for log in all_logs),
        'first31_origins_identical_to_current_paper': True,
        'forecast_sha256': digest(ROOT/'data/forecast_component_selected.npz'),
        'all_source_snapshots_unchanged': all(digest(ROOT/row['copied_path']) == row['sha256'] for row in manifest['sources'])})
    print(json.dumps({'complete': True, 'path': str(ROOT/'data/forecast_component_selected.npz'),
                      'alphas': {v: selection['components'][v]['selected']['alpha'] for v in ('load', 'pv')}}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', help='Optional already-installed project-local sklearn package directory')
    parser.add_argument('--check-inputs-only', action='store_true', help='Rebuild actual/baseline/features and compare source snapshots; do not retrain')
    args = parser.parse_args()
    if args.runtime: sys.path.insert(0, args.runtime)
    import numpy as np
    import pandas as pd
    import scipy
    import sklearn
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    main(check_inputs_only=args.check_inputs_only)
