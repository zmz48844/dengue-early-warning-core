from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor
from common import paths

HERE = Path(__file__).resolve().parent
ROOT = paths.ROOT
from common import modeling as e
from common.modeling import np, pd, joblib
import sklearn
import scipy
from main.candidates import candidates, STAGE1_COUNT, STAGE2_COUNT, CANDIDATE_METADATA

OUT = paths.RUN_DIR
FOLD_SEED = e.SEED + 200
CONSTRAINTS = {"blue": .90, "q50_yellow": .90, "q50_red": .85,
               "q20_yellow": .90, "q20_red": .90,
               "q80_yellow": .70, "q80_red": .70}

FALSE_BURDEN_POLICIES = ("blue", "q50_yellow", "q50_red", "q20_yellow", "q20_red")
MIN_BLUE_SPEC = .85
POLICIES = {"blue": (None, 12), "q50_yellow": (1, 8), "q50_red": (1, 4),
            "q20_yellow": (0, 8), "q20_red": (0, 4),
            "q80_yellow": (2, 8), "q80_red": (2, 4)}
PROTOCOL = {
"version": "standalone_sobol_47_single_oof_v1",
    "development_years": [2005, 2018], "test_years": [2019, 2023],
    "expected_development": e.PROTOCOL["expected_development"],
    "expected_test": e.PROTOCOL["expected_test"],
    "group": "province|city|year; all weeks together; same city in different years may cross folds",
    "development_folds": 3, "outer_folds": 0, "fold_seed": FOLD_SEED,
    "fold_rule": e.PROTOCOL["fold_balancing"],
    "fold_origin": "recompute deterministic label-balanced city-year groups with the unchanged seed; never chosen using model performance",
    "feature_count": 47, "removed_feature": "local_cases_4w_linear_slope",
    "label": "existing Rc>1 for four consecutive weeks, onset at first week; existing censored hazard/TTW labels frozen",
    "stage1_model": e.PROTOCOL["stage1_model"],
    "stage1_ranges": e.PROTOCOL["stage1_ranges"], "stage1_candidates": STAGE1_COUNT,
    "stage1_selection": "highest raw 12-week-risk development OOF AUPRC; ties lowest raw Brier, then fixed candidate ID; NO provisional Gate is selected",
    "stage2_model": e.PROTOCOL["stage2_model"],
    "stage2_ranges": e.PROTOCOL["stage2_ranges"], "stage2_candidates_per_arm": STAGE2_COUNT,
    "candidate_generation": "prefix-stable scrambled Sobol: 96 S1 points, 144 matched structures per S2 arm; original generic anchors retained; tau shuffled in consecutive balanced 48-point blocks with one fixed RNG; no Test-adaptive points",
    "candidate_seed": e.SEED, "arms": list(e.ARMS),
    "stage2_training": "only true exact TTW=0..12, never Gate-filtered; weights all 1; three quantile heads share hyperparameters",
    "transforms": e.PROTOCOL["transforms"],
    "probability_calibration": "one isotonic calibrator fitted on pooled raw development OOF and its labels; reused unchanged after final base-model refit",
    "oof_interpretation": "development selection results, not unbiased independent validation; calibration fitted on this pooled OOF is part of development",
    "gate_grid": e.GATE_GRID, "gate_smooth_span": 3,
    "hysteresis_off_ratio": .90, "hysteresis_off_run": 1,
    "recent_case_rule": "current completed week plus previous 12 weeks, any indigenous or imported case; required only for Stage2",
    "warning_thresholds": {"blue": 12, "yellow": 8, "red": 4},
"warning_quantiles": "Q50 main; Q20 high-sensitivity policy; Q80 yellow and red event recall each >=70%; common Gate across quantiles within each arm",
    "quantile_crossing": "anchor Q50: operational Q20=min(native Q20,Q50), Q80=max(native Q80,Q50); report native pinball separately",
    "recall_constraints": CONSTRAINTS, "minimum_blue_normal_week_specificity": MIN_BLUE_SPEC,
"specificity_origin": "preserve the adopted Sobol operational rule: blue normal-week specificity >=85%",
    "event_definition": "any SAME-WEEK conjunction of Gate, recent cases and quantile<=h within true TTW=0..h; denominator ALL evaluable outbreak city-years",
    "pre_onset_definition": "TTW=1..h reported separately; onset-inclusive recall is not renamed advance warning",
    "negative_week_definition": "no onset within level-specific horizon and complete followup; not restricted to nonoutbreak city-years",
    "gate_selection": "per Stage2 candidate require all recalls including Q80 yellow/red and blue specificity; minimize blue false weeks, then original blue plus Q50/Q20 yellow/red false-week burden, then higher Gate",
    "false_burden_policies": list(FALSE_BURDEN_POLICIES),
    "q80_requirement": "Q80 yellow/red event recall >=70% within their respective windows, including onset week",
    "candidate_selection": "feasible first; then (3 IS0-4+2 IS5-8+IS9-12)/6 using anchored but UNCALIBRATED Q20-Q80 on ALL true 0..12 rows; ties (3 Q50MAE0-4+2 Q50MAE5-8+Q50MAE9-12)/6, native mean pinball, summed policy false weeks, fixed candidate ID",
    "countdown_weights": [3., 2., 1.], "sample_weights": "uniform 1; distinct from evaluation weights",
    "interval_selection_definition": "IS=(upper-lower)+5*max(lower-y,0)+5*max(y-upper,0); nominal60 interval; score using anchored raw endpoints, NEVER expanded calibration endpoints",
    "weighted_score_interpretation": "prespecified near-onset-prioritized operational loss; outcome-window weighting is not a guarantee of conditional quantile calibration; report each band's coverage, width, score and native pinball",
    "user_objective_amendment": "before protocol lock or selection, user approved replacing primary MAE selection by interval-score 3:2:1, while keeping quantile fitting and Q50 error reporting",
    "interval_calibration": "pooled raw development OOF, city-year equal weights, 80th percentile of each tail residual, expansion only; Q50 unchanged",
    "interval_nominal_coverage": .60,
    "interval_interpretation": "separately named empirical calibrated60 endpoints, NOT new Q20/Q80 alarm values; no finite-sample conformal guarantee",
    "primary_arm_rule": "select feasible arm using the same development candidate key; deterministic exponential then direct_ttw tie; both arms reported regardless of Test",
    "no_feasible_policy": "formal candidate/primary arm=null; diagnostic candidate minimizes sum of recall and specificity shortfalls then the same countdown key; may refit and evaluate for diagnosis, never qualified",
    "refit": "lock both arm decisions, calibrators and Gates BEFORE full-development base-model refitting; no fitted-in-sample threshold or calibration updates",
    "test_policy": "load Test arrays only after both system locks and primary-arm lock; no selection after first Test access in this run; report temporal retrospective results and known earlier exploratory access",
    "prior_test_history": "2019-2023 was examined by earlier project versions; this run is not a first-ever untouched confirmatory test",
    "confidence_intervals": "2000 city-cluster bootstrap draws retaining all events/years per city, conditional on fixed predictions, not full-algorithm refitting uncertainty",
    "cache_policy": "fit all candidate OOF models in this run; reuse only this run local checkpoints with identical data, code, fold, parameters and seed signatures; no external experiment dependency",
    "search_scope": "fixed 96/144/144 Sobol candidate universe; all choices on development OOF only",
    "implementation_origin": "standalone refactor of the adopted Sobol method; new source fingerprints and results, no claim of bit-identical old model artifact files",
    "raw_fit_execution": "three city-year-held-out fits per candidate; 1152 total blocks on a fresh full run; lock OOF choices and then refit shared Stage1 and one Stage2 per arm",
}


def stamp():
    return datetime.now().astimezone().isoformat()


def io_path(path):
    return e.core._windows_extended_path(str(Path(path).resolve()))


def exists(path):
    return os.path.exists(io_path(path))


def read_json(path):
    with open(io_path(path), encoding="utf-8-sig") as stream:
        return json.load(stream)


def dump(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".partial_{os.getpid()}")
    with open(io_path(temp), "w", encoding="utf-8") as stream:
        json.dump(e.jsonable(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(io_path(temp), io_path(path))


def load_joblib(path):
    return joblib.load(io_path(path))


def save_joblib(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".partial_{os.getpid()}")
    joblib.dump(value, io_path(temp), compress=3)
    os.replace(io_path(temp), io_path(path))


def save_csv(frame, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(io_path(path), index=False, encoding="utf-8-sig")


def file_hash(path):
    with open(io_path(path), "rb") as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def source_hashes():
    files = [paths.CODE_DIR / "common" / name for name in ("core.py", "modeling.py", "paths.py", "data_builder.py")]
    files += [HERE / name for name in ("engine.py", "runner.py", "candidates.py")]
    files.append(ROOT / "run.py")
    return {p.relative_to(ROOT).as_posix(): file_hash(p) for p in files}


def log(message):
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
    print(line, flush=True)
    with open(io_path(OUT / "runtime.log"), "a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def prepare():

    paths.require_configured()
    d = e.load_development()
    s1, s2 = candidates()
    assert len(s1) == STAGE1_COUNT and all(len(s2[a]) == STAGE2_COUNT for a in e.ARMS)
    folds, cityyears = e.balanced_cityyear_folds(d["meta"], FOLD_SEED)
    identity = dict(data_hashes={k: e.array_digest(d[k]) for k in ["X", "Y", "M", "ttw"]},
                    features=d["features"], sklearn_version=sklearn.__version__)
    metadata_hash = e.array_digest(pd.util.hash_pandas_object(d["meta"], index=False).to_numpy())
    fit_scope = e.signature(dict(**identity, global_idx=e.array_digest(np.arange(len(folds))),
                                 folds=e.array_digest(folds)))
    lock = dict(protocol=PROTOCOL, **identity, metadata_hash=metadata_hash,
                fold_hash=e.array_digest(folds), fit_scope=fit_scope,
                stage1_candidates=s1, stage2_candidates=s2, counts=e.data_counts(d["meta"]),
                source_hashes=source_hashes(), numpy_version=np.__version__, pandas_version=pd.__version__,
                scipy_version=scipy.__version__, candidate_metadata=CANDIDATE_METADATA)
    lock["data_config_sha256"] = file_hash(paths.DATA_CONFIG_PATH)
    lock["runtime_threads"] = paths.CPU_THREADS
    fingerprint = e.signature(lock)
    lock_path = OUT / "protocol_lock.json"
    if exists(lock_path):
        assert read_json(lock_path)["fingerprint"] == fingerprint, "Changed protocol/code/data: create a different version, never overwrite this run"
    else:
        dump(dict(lock, fingerprint=fingerprint, locked_at=stamp(), test_loaded_by_this_run=False), lock_path)
    ctx = dict(identity=identity, metadata_hash=metadata_hash, fit_scope=fit_scope,
               fingerprint=fingerprint, folds=folds,
               fit_codes={stage: e.fit_code_signature(stage) for stage in [1, 2]})
    dump(dict(policy="run-local checkpoints only", no_parent_experiment=True,
              expected_total_blocks=3*(len(s1)+sum(len(v) for v in s2.values()))),
         OUT / "cache_identity_audit.json")
    save_csv(cityyears.assign(fold=cityyears.fold+1), OUT / "cityyear_folds.csv")
    sample = d["meta"][["province", "city", "year", "sample_idx", "date"]].copy()
    sample["development_row"] = np.arange(len(folds))
    sample["fold"] = folds+1
    save_csv(sample, OUT / "sample_folds.csv")
    fold_counts = [dict(fold=i+1, **e.data_counts(d["meta"].loc[folds == i])) for i in range(3)]
    save_csv(pd.DataFrame(fold_counts), OUT / "fold_counts.csv")
    dump(dict(features=d["features"], feature_count=47, development=e.data_counts(d["meta"]),
              fold_counts=fold_counts, Stage2_exact_rows_per_event=d["meta"].loc[e.exact_mask(d["ttw"])].groupby(["province", "city", "year"]).size().tolist(),
              note="source feature/label definitions frozen; no historic Train/Validation role used"), OUT / "data_audit.json")
    log(f"Prepared fixed single 3-fold OOF: {e.data_counts(d['meta'])}; standalone 96/144/144 Sobol run; Test not loaded")
    return d, s1, s2, ctx


def raw_oof(d, candidate, stage, arm, ctx):

    paths.require_configured()
    folds = ctx["folds"]
    seeds = [e.SEED+(100 if stage == 1 else 200)+sum(10**f for f in range(3) if f != held) for held in range(3)]
    key = e.signature(dict(fit_scope=ctx["fit_scope"], metadata=ctx["metadata_hash"], stage=stage,
                           arm=arm, params=candidate["params"], fit_code=ctx["fit_codes"][stage], seeds=seeds))
    tag = "stage1" if stage == 1 else "stage2_" + arm
    own_path = OUT / "raw_oof" / tag / f"c{candidate['id']:03d}_{key[:12]}.joblib"
    shape = (len(folds),) if stage == 1 else (len(folds), 3)
    if exists(own_path):
        saved = load_joblib(own_path)
        assert saved["signature"] == key and saved["raw"].shape == shape
        assert np.isfinite(saved["raw"]).all()
        return saved["raw"], dict(saved["audit"], reused_new_oof=True, fitted_blocks=0, cached_blocks=3)

    def one(held):
        fit_folds = tuple(i for i in range(3) if i != held)
        tr, va = np.flatnonzero(folds != held), np.flatnonzero(folds == held)
        assert len(tr) and len(va)
        assert set(e.year_ids(d["meta"])[tr]).isdisjoint(set(e.year_ids(d["meta"])[va]))
        seed = seeds[held]
        fit_key = e.signature(dict(scope=ctx["fit_scope"], stage=stage, arm=arm,
                                   params=candidate["params"], fit_folds=fit_folds,
                                   seed=seed, fit_code=ctx["fit_codes"][stage]))
        name = "fit_"+"".join(str(f+1) for f in fit_folds)+"_"+fit_key[:12]+".joblib"
        cache = OUT / "fold_raw_cache" / tag / f"c{candidate['id']:03d}" / name
        began = time.monotonic()
        cached = exists(cache)
        if cached:
            saved = load_joblib(cache)
            assert saved["signature"] == fit_key and saved["metadata_hash"] == ctx["metadata_hash"]
            assert np.array_equal(saved["prediction_rows"], va)
            raw = saved["raw"]
        else:
            log(f"Fitting Stage{stage} {arm or ''} c{candidate['id']:03d}, held fold {held+1}/3")
            raw, fitted = e.fit_stage1(d, tr, va, candidate["params"], seed) if stage == 1 else e.fit_stage2(d, tr, va, candidate["params"], arm, seed)
            del fitted
        assert raw.shape == ((len(va),) if stage == 1 else (len(va), 3))
        assert np.isfinite(raw).all()
        if not cached:
            save_joblib(dict(signature=fit_key, metadata_hash=ctx["metadata_hash"],
                             prediction_rows=va, raw=raw), cache)
        return va, raw, dict(held_fold=held+1, fit_folds=[f+1 for f in fit_folds], seed=seed,
                             source="run_local_checkpoint" if cached else "fresh_grouped_fold_fit",
                             seconds=time.monotonic()-began, fit_signature=fit_key,
                             reused_fold_cache=cached, checkpoint=cache.relative_to(OUT).as_posix())

    with ThreadPoolExecutor(max_workers=3 if stage == 2 else 2) as pool:
        parts = list(pool.map(one, range(3)))
    raw = np.full(shape, np.nan)
    seen = np.zeros(len(folds), dtype=int)
    for va, values, _ in parts:
        raw[va], seen[va] = values, seen[va]+1
    assert np.all(seen == 1) and np.isfinite(raw).all()
    audit = dict(stage=stage, arm=arm, candidate_id=candidate["id"], blocks=[r[2] for r in parts],
                 fitted_blocks=sum(not r[2]["reused_fold_cache"] for r in parts),
                 cached_blocks=sum(r[2]["reused_fold_cache"] for r in parts),
                 oof_each_row_predicted_once=True)
    save_joblib(dict(signature=key, raw=raw, audit=audit), own_path)
    return raw, audit


def check_constraints(system):
    shortfalls = {}
    for name, threshold in CONSTRAINTS.items():
        value = float(system[name]["event_recall"])
        shortfalls[name] = max(0., threshold-value) if np.isfinite(value) else 1.
    value = float(system["blue"]["normal_week_specificity"])
    shortfalls["blue_specificity"] = max(0., MIN_BLUE_SPEC-value) if np.isfinite(value) else 1.
    return all(v <= 1e-12 for v in shortfalls.values()), float(sum(shortfalls.values())), shortfalls


def make_alarms(meta, gate_active, native):
    recent = ((meta.indigenous_gate_activity_sum.to_numpy() > 0) |
              (meta.imported_gate_activity_sum.to_numpy() > 0))
    active = np.asarray(gate_active, dtype=bool) & recent
    q = e.anchor_quantiles(native)
    alarms = {name: np.asarray(gate_active, dtype=bool) if column is None else active & (q[:, column] <= horizon)
              for name, (column, horizon) in POLICIES.items()}
    return alarms, active


class GateEvaluator:
    def __init__(self, meta, probability):
        self.meta = meta
        self.thresholds = e.GATE_GRID
        _, self.gates = e.smooth_and_gate(meta, probability, self.thresholds)
        self.recent = ((meta.indigenous_gate_activity_sum.to_numpy() > 0) |
                       (meta.imported_gate_activity_sum.to_numpy() > 0))
        groups = list(meta.groupby(["province", "city", "year"], sort=False).indices.values())
        self.negative, self.targets = {}, {}
        for horizon in [4, 8, 12]:
            positive, valid = e.label_and_valid(meta, horizon)
            self.negative[horizon] = valid & ~positive
            self.targets[horizon] = [np.asarray(idx)[positive[idx]] for idx in groups if positive[idx].any()]
        self.blue = self.level(np.ones(len(meta), bool), 12)

    def level(self, condition, horizon):
        alarm = self.gates & np.asarray(condition)[None, :]
        targets = self.targets[horizon]
        hits = np.stack([alarm[:, idx].any(axis=1) for idx in targets], axis=1) if targets else np.zeros((len(self.thresholds), 0))
        normal = int(self.negative[horizon].sum())
        false = alarm[:, self.negative[horizon]].sum(axis=1)
        return dict(recall=hits.mean(axis=1) if targets else np.full(len(self.thresholds), np.nan),
                    hits=hits.sum(axis=1), targets=len(targets), false=false, normal=normal,
                    specificity=1-false/normal if normal else np.full(len(self.thresholds), np.nan))

    def evaluate(self, native):
        q = e.anchor_quantiles(native)
        stats = {"blue": self.blue}
        for name, (column, horizon) in POLICIES.items():
            if column is not None:
                stats[name] = self.level(self.recent & (q[:, column] <= horizon), horizon)
        rows = []
        for i, gate in enumerate(self.thresholds):
            system = {name: dict(event_recall=float(s["recall"][i]), event_detected=int(s["hits"][i]),
                                 event_targets=s["targets"], false_alarm_weeks=int(s["false"][i]),
                                 normal_weeks=s["normal"], normal_week_specificity=float(s["specificity"][i])) for name, s in stats.items()}
            feasible, deficit, shortfalls = check_constraints(system)

            burden = sum(system[name]["false_alarm_weeks"] for name in FALSE_BURDEN_POLICIES)
            rows.append(dict(gate=float(gate), system=system, feasible=feasible,
                             total_shortfall=deficit, shortfalls=shortfalls, summed_policy_false_weeks=burden))
        best = min(rows, key=lambda r: (not r["feasible"], r["total_shortfall"],
                                        r["system"]["blue"]["false_alarm_weeks"],
                                        r["summed_policy_false_weeks"], -r["gate"]))
        return best, rows


def candidate_key(row):
    cm = row["countdown_potential"]
    return (not row["feasible"], row["total_shortfall"], cm["objective_interval_3_2_1"],
            cm["objective_mae_3_2_1"], cm["pinball_mean_all"],
            row["summed_policy_false_weeks"], row["candidate_id"])


def select_record(rows):
    assert rows, "Candidate pool must not be empty"
    best = min(rows, key=candidate_key)
    return dict(status="QUALIFIED" if best["feasible"] else "NO_FEASIBLE_CANDIDATE_DIAGNOSTIC_ONLY",
                selected_candidate_id=best["candidate_id"] if best["feasible"] else None,
                diagnostic_candidate_id=None if best["feasible"] else best["candidate_id"],
                selected_gate=best["selected_gate"], prediction_candidate=best,
                feasible_candidates=sum(r["feasible"] for r in rows), evaluated_candidates=len(rows))


def countdown_metrics(meta, native, intervals, active=None):
    result = e.countdown_metrics(meta, native, intervals, active)
    y = meta.raw_ttw.to_numpy(float)
    use = e.exact_mask(y) & (np.ones(len(y), bool) if active is None else active)
    q = e.anchor_quantiles(native)
    for band, part in [("all", use), ("0_4", use & (y <= 4)),
                       ("5_8", use & (y >= 5) & (y <= 8)), ("9_12", use & (y >= 9))]:
        error = q[part, 1]-y[part]
        result["bias_"+band] = float(np.mean(error)) if len(error) else np.nan
        for margin in [1, 2, 4]:
            result[f"within_{margin}_{band}"] = float(np.mean(np.abs(error) <= margin)) if len(error) else np.nan
        for column, alpha in enumerate([20, 50, 80]):
            result[f"q{alpha}_mae_{band}"] = float(np.mean(np.abs(q[part, column]-y[part]))) if len(error) else np.nan
            result[f"q{alpha}_empirical_cdf_{band}"] = float(np.mean(y[part] <= q[part, column])) if len(error) else np.nan
    bands = [(3., "0_4"), (2., "5_8"), (1., "9_12")]
    result["objective_interval_3_2_1"] = sum(w*result["raw_interval_score_"+band] for w, band in bands)/6.
    result["objective_mae_3_2_1"] = sum(w*result["mae_"+band] for w, band in bands)/6.
    result["weighted_raw_coverage_3_2_1"] = sum(w*result["raw_coverage_"+band] for w, band in bands)/6.
    result["weighted_raw_width_3_2_1"] = sum(w*result["raw_width_"+band] for w, band in bands)/6.
    result["weighted_cal_interval_score_3_2_1_descriptive_only"] = sum(w*result["cal_interval_score_"+band] for w, band in bands)/6.
    return result


def build_frame(d, raw1, probability, native, intervals, gate, arm, feasible, status):
    score, gates = e.smooth_and_gate(d["meta"], probability, [gate])
    alarms, active = make_alarms(d["meta"], gates[0], native)
    frame = d["meta"].copy()
    for j, alpha in enumerate([20, 50, 80]):
        frame[f"raw_q{alpha}"] = native[:, j]
        frame[f"operational_q{alpha}"] = e.anchor_quantiles(native)[:, j]
    frame["interval_lower_60"], frame["interval_upper_60"] = intervals[:, 0], intervals[:, 1]
    frame["p_12w_raw"], frame["p_12w"], frame["gate_score"] = raw1, probability, score
    frame["gate_threshold"] = gate
    frame["stage2_active"] = active.astype(int)
    for name, alarm in alarms.items():
        frame[name+"_alarm"] = alarm.astype(int)
    frame["yellow_alarm"], frame["red_alarm"] = frame.q50_yellow_alarm, frame.q50_red_alarm
    frame["arm"], frame["model_feasible"], frame["selection_status"] = arm, int(feasible), status
    return frame


def bootstrap_city_ci(frame, alarms, event_tables):
    cities = e.city_ids(frame)
    unique = np.unique(cities)
    rng = np.random.default_rng(e.SEED+500)
    counts = rng.multinomial(len(unique), np.full(len(unique), 1/len(unique)), size=2000)
    years = np.array([len(np.unique(e.year_ids(frame)[cities == c])) for c in unique])
    city_lookup = {c: i for i, c in enumerate(unique)}
    result = {}
    for name, (_, horizon) in POLICIES.items():
        events = pd.DataFrame(event_tables[name])
        positive, valid = e.label_and_valid(frame, horizon)
        negative = valid & ~positive
        totals, hits, pre, normals, false = [], [], [], [], []
        ec = (events.province.astype(str)+"|"+events.city.astype(str)).to_numpy()
        for city in unique:
            m, em = cities == city, ec == city
            totals.append(int(em.sum()))
            hits.append(int(events.loc[em, "detected"].sum()))
            pre.append(int(events.loc[em, "pre_onset_detected"].sum()))
            normals.append(int(np.sum(m & negative)))
            false.append(int(np.sum(m & negative & alarms[name])))
        den, nden = counts @ np.asarray(totals), counts @ np.asarray(normals)
        with np.errstate(divide="ignore", invalid="ignore"):
            values = dict(event_recall=(counts @ np.asarray(hits))/den,
                          pre_onset_recall=(counts @ np.asarray(pre))/den,
                          normal_week_specificity=1-(counts @ np.asarray(false))/nden,
                          false_alarm_weeks_per_city_year=(counts @ np.asarray(false))/(counts @ years))
        hit_mask = events.detected.to_numpy(bool)
        leads = events.loc[hit_mask, "first_alarm_lead_in_window_weeks"].to_numpy(float)
        hit_city = np.array([city_lookup[c] for c in ec[hit_mask]], int)
        medians, means = [], []
        for weights in counts:
            sample = np.repeat(leads, weights[hit_city])
            medians.append(float(np.median(sample)) if len(sample) else np.nan)
            means.append(float(np.mean(sample)) if len(sample) else np.nan)
        values.update(median_first_alarm_lead_weeks=np.asarray(medians), mean_first_alarm_lead_weeks=np.asarray(means))
        result[name] = {}
        for field, v in values.items():
            good = v[np.isfinite(v)]
            result[name][field] = dict(lower=float(np.quantile(good, .025)) if len(good) else np.nan,
                                       upper=float(np.quantile(good, .975)) if len(good) else np.nan,
                                       valid_bootstrap_draws=len(good))
    return result


def evaluate_frame(frame, path, label, bootstrap=True):
    path = Path(path)
    native = frame[["raw_q20", "raw_q50", "raw_q80"]].to_numpy()
    intervals = frame[["interval_lower_60", "interval_upper_60"]].to_numpy()
    alarms = {name: frame[name+"_alarm"].to_numpy(bool) for name in POLICIES}
    system, events = {}, {}
    ncy = e.data_counts(frame)["city_years"]
    for name, (_, horizon) in POLICIES.items():
        system[name], events[name] = e.alarm_summary(frame, alarms[name], horizon)
        system[name]["false_alarm_weeks_per_city_year"] = system[name]["false_alarm_weeks"]/ncy
        save_csv(pd.DataFrame(events[name]), path / ("events_"+name+".csv"))
    passed, deficit, shortfalls = check_constraints(system)
    exact = e.exact_mask(frame.raw_ttw.to_numpy())
    active = frame.stage2_active.to_numpy(bool)
    result = dict(label=label, arm=frame.arm.iloc[0], counts=e.data_counts(frame),
                  model_qualified=bool(frame.model_feasible.eq(1).all()),
                  evaluation_constraints_met=passed, constraint_shortfalls=shortfalls,
                  evaluation_total_shortfall=deficit, system=system,
                  confidence_intervals=bootstrap_city_ci(frame, alarms, events) if bootstrap else {},
                  countdown_potential=countdown_metrics(frame, native, intervals),
                  countdown_active=countdown_metrics(frame, native, intervals, active),
                  countdown_output_fraction_on_true_windows=float(active[exact].mean()),
                  probability=e.probability_metrics(frame, frame.p_12w_raw.to_numpy(), frame.p_12w.to_numpy()),
                  gate=float(frame.gate_threshold.iloc[0]),
                  interpretation="development tuning with pooled-OOF calibration, not independent validation" if label.startswith("development") else "temporal retrospective evaluation; earlier exploratory Test access acknowledged")
    dump(result, path / "metrics.json")
    save_csv(pd.DataFrame([dict(policy=k, **v) for k, v in system.items()]), path / "warning_metrics.csv")
    save_csv(pd.DataFrame([dict(domain="all_true_0_12", **result["countdown_potential"]),
                           dict(domain="system_active_only", **result["countdown_active"])]), path / "countdown_metrics.csv")
    return result
