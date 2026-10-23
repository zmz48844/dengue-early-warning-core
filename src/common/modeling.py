from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ.setdefault("DENGUE_CPU_THREADS", "6")

import joblib
import numpy as np
import pandas as pd
from scipy.stats import qmc
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

from common import core, paths
from common import data_builder
ROOT = paths.ROOT
core.CPU_THREADS = paths.CPU_THREADS
SEED = 20260909
ARMS = ("exponential", "direct_ttw")
ALPHAS = (.2, .5, .8)
LEVELS = {"blue": 12, "yellow": 8, "red": 4}
GATE_GRID = [0., .001, .002, .003, .005, .0075, .01, .0125, .015,
             .0175, .02, .025, .03, .035, .04, .05, .06, .08, .10,
             .12, .15, .20, .25, .30, .40, .50, .75, 1.]

PROTOCOL = {
    "expected_development": {"cities": 86, "city_years": 1204, "events": 36},
    "expected_test": {"cities": 86, "city_years": 430, "events": 35},
    "removed_feature": "local_cases_4w_linear_slope",
    "fold_balancing": "fixed-seed label-only greedy allocation and swaps; balance outbreak city-years, province, city-year and week counts",
    "stage1_model": "RandomForest discrete hazard t=0..12; balanced_subsample; 250 trees",
    "stage1_ranges": {"max_depth": [8,36], "min_samples_leaf": [5,120], "max_features_log": [.04,.60], "max_samples": [.55,1.]},
    "stage2_model": "three HistGradientBoosting quantile heads; no early stopping",
    "stage2_ranges": {"learning_rate_log": [.01,.18], "max_depth": [2,8], "max_leaf_nodes": [7,63],
        "min_samples_leaf": [2,100], "l2_regularization_log": [.001,30], "max_bins": [63,127,255],
        "max_iter": [250,1400,50], "tau_weeks": [2,3,4,5,6,8]},
    "transforms": {"direct_ttw": "z=y", "exponential": "z=12*(1-exp(-y/tau))/(1-exp(-12/tau)); analytic inverse"},
}


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, np.generic):
        return jsonable(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    if isinstance(x, Path):
        return str(x)
    return x


def dump(x, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(x), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def signature(x):
    return hashlib.sha256(json.dumps(jsonable(x), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def array_digest(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def city_ids(meta):
    return (meta.province.astype(str) + "|" + meta.city.astype(str)).to_numpy()


def year_ids(meta):
    return (meta.province.astype(str) + "|" + meta.city.astype(str) + "|" + meta.year.astype(str)).to_numpy()


def data_counts(meta):
    cy = meta.drop_duplicates(["province", "city", "year"])
    return {"cities": int(len(np.unique(city_ids(meta)))), "city_years": len(cy),
            "events": int((cy.onset_idx_raw >= 0).sum()), "weeks": len(meta),
            "exact_rows": int(meta.raw_ttw.between(0, 12).sum())}


def load_development():
    return data_builder.load_development()


def subset(d, idx):
    idx = np.asarray(idx, dtype=int)
    out = {k: d[k][idx] for k in ["X", "Y", "M", "ttw"] if k in d}
    out.update(meta=d["meta"].iloc[idx].reset_index(drop=True), features=list(d["features"]))
    return out


def balanced_cityyear_folds(meta, seed):

    cy = meta.drop_duplicates(["province", "city", "year"]).copy()
    cy["event"] = (cy.onset_idx_raw >= 0).astype(int)
    cities = cy[["province", "city", "year", "event"]].rename(columns={"event": "events"}).sort_values(
        ["province", "city", "year"]).reset_index(drop=True)
    cities["city_years"] = 1
    weeks = meta.groupby(["province", "city", "year"]).size()
    cities["weeks"] = [weeks.loc[(p, c, y)] for p, c, y in zip(cities.province, cities.city, cities.year)]
    prov = pd.get_dummies(cities.province, dtype=float).to_numpy()
    vec = np.column_stack([cities.events, (cities.events > 0).astype(int),
                           np.ones(len(cities)), cities.weeks, prov]).astype(float)
    target = vec.sum(axis=0) / 3
    weights = np.array([12., 2., 2., .5] + [2.] * prov.shape[1])

    def loss(load):
        return float(np.sum(weights * ((load - target) / np.maximum(target, 1)) ** 2))

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(32):
        order = np.lexsort((rng.random(len(cities)), -cities.events.to_numpy()))
        load = np.zeros((3, vec.shape[1]))
        assignment = np.full(len(cities), -1, int)
        for i in order:
            candidates = []
            for f in rng.permutation(3):
                trial = load.copy()
                trial[f] += vec[i]
                candidates.append((loss(trial), f))
            f = min(candidates, key=lambda z: z[0])[1]
            assignment[i] = f
            load[f] += vec[i]
        score = loss(load)
        for _ in range(1200):
            i, j = rng.choice(len(cities), 2, replace=False)
            a, b = assignment[i], assignment[j]
            if a == b:
                continue
            trial = load.copy()
            trial[a] += vec[j] - vec[i]
            trial[b] += vec[i] - vec[j]
            value = loss(trial)
            if value < score - 1e-12:
                assignment[i], assignment[j] = b, a
                load, score = trial, value
        if best is None or score < best[0]:
            best = (score, assignment.copy())
    cities["fold"] = best[1]
    lookup = {(r.province, r.city, int(r.year)): int(r.fold) for r in cities.itertuples()}
    rows = np.array([lookup[(p, c, int(y))] for p, c, y in zip(meta.province, meta.city, meta.year)])
    for f in range(3):
        assert set(year_ids(meta)[rows == f]).isdisjoint(set(year_ids(meta)[rows != f]))
        assert data_counts(meta.loc[rows == f])["events"] > 0
    return rows, cities


def exact_mask(y):
    return np.isfinite(y) & (y >= 0) & (y <= 12)


def label_and_valid(meta, horizon=12):
    ttw = meta.onset_idx_raw.to_numpy() - meta.sample_idx.to_numpy()
    y = (meta.onset_idx_raw.to_numpy() >= 0) & (ttw >= 0) & (ttw <= horizon)
    complete = meta.observed_time_points.to_numpy() >= horizon + 1
    return y, y | complete


def fit_probability_calibrator(meta, raw):
    y, valid = label_and_valid(meta)
    assert len(np.unique(y[valid])) == 2, "Calibration training must include positive and negative outcomes"
    model = IsotonicRegression(out_of_bounds="clip", y_min=0., y_max=1.)
    model.fit(np.clip(raw[valid], 1e-6, 1-1e-6), y[valid].astype(int))
    return model


def calibrated_probability(model, raw):
    return np.clip(model.predict(np.clip(raw, 1e-6, 1-1e-6)), 0, 1)


def fit_stage1(d, tr, va, params, seed):

    scaler = StandardScaler().fit(d["X"][tr, 0, :])
    model = core.Stage1SurvivalTrigger(dict(horizon_weeks=12, seed=seed, **params))
    xtr = scaler.transform(d["X"][tr, 0, :]).astype(np.float32)[:, None, :]
    model.fit(xtr, d["Y"][tr], d["M"][tr])
    raw = None
    if len(va):
        xva = scaler.transform(d["X"][va, 0, :]).astype(np.float32)[:, None, :]
        raw = model.predict_12w_raw(xva)["p_12w_raw"]
    return raw, {"model": model, "scaler": scaler}


def transform_target(y, arm, tau=4.):
    y = np.clip(np.asarray(y, dtype=float), 0, 12)
    return y if arm == "direct_ttw" else 12*(-np.expm1(-y/tau))/(-np.expm1(-12/tau))


def inverse_target(z, arm, tau=4.):
    z = np.clip(np.asarray(z, dtype=float), 0, 12)
    return z if arm == "direct_ttw" else np.clip(-tau*np.log1p(-(z/12)*(-np.expm1(-12/tau))), 0, 12)


def fit_stage2(d, tr, va, params, arm, seed):

    scaler = StandardScaler().fit(d["X"][tr, 0, :])
    exact = tr[exact_mask(d["ttw"][tr])]
    assert len(exact) > 0
    xtr = scaler.transform(d["X"][exact, 0, :]).astype(np.float32)
    xva = scaler.transform(d["X"][va, 0, :]).astype(np.float32) if len(va) else None
    tau = params.get("tau_weeks", 4.)
    z = transform_target(d["ttw"][exact], arm, tau)
    models, cols = [], []
    for alpha in ALPHAS:
        model = HistGradientBoostingRegressor(
            loss="quantile", quantile=alpha, early_stopping=False, random_state=seed,
            **{k: v for k, v in params.items() if k != "tau_weeks"})
        model.fit(xtr, z, sample_weight=np.ones(len(z)))
        models.append(model)
        if xva is not None:
            cols.append(inverse_target(model.predict(xva), arm, tau))
    raw = np.column_stack(cols) if cols else None
    return raw, {"models": models, "scaler": scaler, "arm": arm, "tau": tau,
                 "exact_training_rows": len(exact)}


def anchor_quantiles(raw):
    raw = np.clip(np.asarray(raw, dtype=float), 0, 12)
    return np.column_stack([np.minimum(raw[:, 0], raw[:, 1]), raw[:, 1],
                            np.maximum(raw[:, 2], raw[:, 1])])


def weighted_quantile(values, weights, q):
    order = np.argsort(values, kind="stable")
    return float(values[order[min(np.searchsorted(np.cumsum(weights[order]), q*np.sum(weights), side="left"), len(order)-1)]])


def fit_interval_calibrator(meta, raw):
    y = meta.raw_ttw.to_numpy(float)
    exact = exact_mask(y)
    q = anchor_quantiles(raw[exact])
    y, cities = y[exact], year_ids(meta)[exact]
    counts = pd.Series(cities).value_counts()
    weights = np.array([1./counts[c] for c in cities])
    assert len(y) and np.isfinite(q).all()
    return {"lower_offset": max(0., weighted_quantile(q[:, 0]-y, weights, .8)),
            "upper_offset": max(0., weighted_quantile(y-q[:, 2], weights, .8)),
            "calibration_exact_rows": len(y), "calibration_event_city_years": len(counts),
            "nominal_coverage": .60, "method": "empirical_city_year_equal_tail_offsets"}


def apply_interval_calibrator(raw, cal):
    q = anchor_quantiles(raw)
    return np.column_stack([np.maximum(0, q[:, 0]-cal["lower_offset"]),
                            np.minimum(12, q[:, 2]+cal["upper_offset"])])


def smooth_and_gate(meta, probability, thresholds):
    thresholds = np.asarray(thresholds)
    score = np.empty(len(meta), dtype=float)
    gates = np.zeros((len(thresholds), len(meta)), dtype=bool)
    for idx in meta.groupby(["province", "city", "year"], sort=False).indices.values():
        idx = np.asarray(idx)
        assert np.all(np.diff(meta.sample_idx.to_numpy()[idx]) > 0)
        sm = core.ewma(np.asarray(probability)[idx], span=3)
        score[idx] = sm
        state = np.zeros(len(thresholds), dtype=bool)
        for pos, value in zip(idx, sm):
            state = (state & (value >= thresholds*.90)) | (~state & (value >= thresholds))
            gates[:, pos] = state
    return score, gates


def alarm_summary(meta, alarm, horizon):
    positive, valid = label_and_valid(meta, horizon)
    negative = valid & ~positive
    event_rows, detected, lead1_detected, leads = [], 0, 0, []
    ttws = meta.onset_idx_raw.to_numpy() - meta.sample_idx.to_numpy()
    for idx in meta.groupby(["province", "city", "year"], sort=False).indices.values():
        idx = np.asarray(idx)
        target = idx[positive[idx]]
        if not len(target):
            continue
        hit = target[alarm[target]]
        detect = bool(len(hit))
        lead = int(ttws[hit].max()) if detect else None
        pre_hit = bool(np.any(alarm[target] & (ttws[target] >= 1)))
        detected += detect
        lead1_detected += pre_hit
        if detect:
            leads.append(lead)
        row = meta.iloc[idx[0]]
        event_rows.append(dict(province=row.province, city=row.city, year=int(row.year),
                               detected=int(detect), pre_onset_detected=int(pre_hit),
                               first_alarm_lead_in_window_weeks=lead))
    n = len(event_rows)
    return {"event_targets": n, "event_detected": int(detected),
            "event_recall": detected/n if n else np.nan,
            "pre_onset_detected": int(lead1_detected),
            "pre_onset_recall": lead1_detected/n if n else np.nan,
            "normal_weeks": int(negative.sum()),
            "false_alarm_weeks": int(np.sum(alarm & negative)),
            "normal_week_specificity": 1-float(np.mean(alarm[negative])) if negative.any() else np.nan,
            "median_first_alarm_lead_weeks": float(np.median(leads)) if leads else np.nan,
            "mean_first_alarm_lead_weeks": float(np.mean(leads)) if leads else np.nan,
            "alarmed_weeks": int(np.sum(alarm))}, event_rows


def interval_score(y, lo, hi):
    return hi-lo + 5*np.maximum(lo-y, 0) + 5*np.maximum(y-hi, 0)


def countdown_metrics(meta, raw, intervals, mask=None):
    y = meta.raw_ttw.to_numpy(float)
    use = exact_mask(y) & (np.ones(len(y), bool) if mask is None else mask)
    yy, q, ci = y[use], anchor_quantiles(raw[use]), intervals[use]
    native = np.clip(raw[use], 0, 12)
    result = {"rows": int(use.sum()), "event_cities": int(len(np.unique(city_ids(meta)[use])))}
    for name, subset_mask in [("all", np.ones(len(yy), bool)), ("0_4", yy <= 4),
                              ("5_8", (yy >= 5) & (yy <= 8)), ("9_12", yy >= 9)]:
        obs, qq, cc = yy[subset_mask], q[subset_mask], ci[subset_mask]
        result[f"rows_{name}"] = len(obs)
        if not len(obs):
            for field in ["mae", "raw_coverage", "raw_width", "raw_interval_score", "cal_coverage",
                          "cal_width", "cal_interval_score", "cal_coverage_error", "pinball_mean"]:
                result[f"{field}_{name}"] = np.nan
            continue
        result[f"mae_{name}"] = float(np.mean(np.abs(obs-qq[:, 1])))
        native_error = obs[:, None]-native[subset_mask]
        native_loss = np.maximum(np.array(ALPHAS)*native_error, (np.array(ALPHAS)-1)*native_error)
        repaired_error = obs[:, None]-qq
        repaired_loss = np.maximum(np.array(ALPHAS)*repaired_error, (np.array(ALPHAS)-1)*repaired_error)
        result[f"pinball_mean_{name}"] = float(np.mean(native_loss))
        result[f"repaired_pinball_mean_{name}"] = float(np.mean(repaired_loss))
        for j, alpha in enumerate(ALPHAS):
            result[f"pinball_q{int(100*alpha)}_{name}"] = float(np.mean(native_loss[:, j]))
        for tag, lo, hi in [("raw", qq[:, 0], qq[:, 2]), ("cal", cc[:, 0], cc[:, 1])]:
            coverage = float(np.mean((obs >= lo) & (obs <= hi)))
            result[f"{tag}_coverage_{name}"] = coverage
            result[f"{tag}_width_{name}"] = float(np.mean(hi-lo))
            result[f"{tag}_interval_score_{name}"] = float(np.mean(interval_score(obs, lo, hi)))
            result[f"{tag}_coverage_error_{name}"] = coverage - .60
            result[f"{tag}_absolute_coverage_error_{name}"] = abs(coverage - .60)
    a, b, c = [result[f"mae_{band}"] for band in ["0_4", "5_8", "9_12"]]
    result.update(objective_2_1_0p5=2*a+b+.5*c, normalized_objective_2_1_0p5=(2*a+b+.5*c)/3.5,
                  sensitivity_2_1_0=2*a+b, sensitivity_1_1_1=a+b+c,
                  raw_quantile_crossing_fraction=float(np.mean((np.diff(raw[use], axis=1) < 0).any(axis=1))) if use.any() else np.nan)
    return result


def probability_metrics(meta, raw, calibrated):
    y, valid = label_and_valid(meta)
    return {"evaluable_rows": int(valid.sum()),
            "average_precision_raw": float(average_precision_score(y[valid], raw[valid])),
            "brier_raw": float(brier_score_loss(y[valid], raw[valid])),
            "brier_calibrated": float(brier_score_loss(y[valid], calibrated[valid]))}


def fit_code_signature(stage):
    functions = [fit_stage1] if stage == 1 else [fit_stage2, transform_target, inverse_target]
    return signature([inspect.getsource(f) for f in functions] +
                     [hashlib.sha256((Path(core.__file__)).read_bytes()).hexdigest()])
