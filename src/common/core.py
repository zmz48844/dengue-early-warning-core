from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import gamma
from sklearn.ensemble import RandomForestClassifier

CPU_THREADS = int(os.environ.get("DENGUE_CPU_THREADS", "6"))
METEO = ["prec", "pres", "rhum", "temp", "wind"]
CASE_RAW = ["total imported cases", "total indigenous cases"]
_BANDS = ["lag1w", "lag2_4w", "lag5_7w", "lag8_10w", "lag11_13w"]
WEEKLY_CASE_FEATURES = [
    f"{prefix}_{suffix}_log1p"
    for prefix in ["import", "local"] for suffix in _BANDS
]
WEEKLY_WEATHER_FEATURES = [
    f"{var}_{band}_{'sum' if var == 'prec' else 'mean'}"
    for var in ["temp", "prec", "rhum", "wind", "pres"] for band in _BANDS
]
WEEKLY_SEASONAL_FEATURES = ["week_sin", "week_cos"]
WEEKLY_SPATIAL_CASE_FEATURES = [
    f"province_other_{prefix}_{suffix}_log1p"
    for prefix in ["import", "local"] for suffix in _BANDS
]
MODEL_FEATURES_47 = (
    WEEKLY_CASE_FEATURES + WEEKLY_WEATHER_FEATURES
    + WEEKLY_SEASONAL_FEATURES + WEEKLY_SPATIAL_CASE_FEATURES
)

def _windows_extended_path(path):


    raw = os.fspath(path)
    if os.name != "nt" or raw.startswith("\\\\?\\"):
        return raw
    absolute = os.path.abspath(raw)
    if len(absolute) < 200:
        return raw
    if absolute.startswith("\\\\"):
        return "\\\\?\\UNC\\" + absolute[2:]
    return "\\\\?\\" + absolute


def ewma(vals, span=3):
    v = np.asarray(vals, dtype=np.float32)
    if len(v) == 0 or span <= 1:
        return v.copy()
    a = 2.0 / (span + 1.0)
    out = np.empty_like(v)
    out[0] = v[0]
    for i in range(1, len(v)):
        out[i] = a * v[i] + (1 - a) * out[i - 1]
    return out


def remove_short(mask, n=2):
    mask = np.asarray(mask, dtype=bool)
    if n <= 1 or not mask.any():
        return mask
    out = np.zeros_like(mask, dtype=bool)
    start = None
    for i in range(len(mask)):
        if mask[i]:
            if start is None:
                start = i
        else:
            if start is not None:
                if i - start >= n:
                    out[start:i] = True
                start = None
    if start is not None and len(mask) - start >= n:
        out[start:] = True
    return out


def complete_weekly_index(agg):
    if len(agg) == 0:
        return agg

    agg = agg.sort_values("time").reset_index(drop=True)
    full_weeks = pd.date_range(agg["time"].min(), agg["time"].max(), freq="W-MON")

    agg = (
        agg.set_index("time")
        .reindex(full_weeks)
        .rename_axis("time")
        .reset_index()
    )

    for c in CASE_RAW:
        if c not in agg.columns:
            agg[c] = 0.0
        agg[c] = pd.to_numeric(agg[c], errors="coerce").fillna(0.0)

    for c in METEO:
        if c not in agg.columns:
            agg[c] = 0.0
        agg[c] = pd.to_numeric(agg[c], errors="coerce")


        agg[c] = agg[c].ffill()

    if "n_days" not in agg.columns:
        agg["n_days"] = 0.0
    agg["n_days"] = pd.to_numeric(agg["n_days"], errors="coerce").fillna(0.0)

    return agg


def read_city(path, cfg):
    city = os.path.basename(path).replace(".csv", "")
    geo_city = re.sub(r"_\d{4}$", "", city)
    prov = "Unknown"

    aliases = {"\u5e7f\u897f\u58ee\u65cf\u81ea\u6cbb\u533a": "\u5e7f\u897f"}
    for r in cfg["regions"]:
        if r in city:
            prov = r
            break
    for k, v in aliases.items():
        if k in city:
            prov = v
            break

    df = pd.read_csv(path, engine="python", encoding="utf-8-sig")
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "time" not in df.columns:
        raise ValueError(f"No time column: {path}")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)


    for c in CASE_RAW:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in METEO:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["week_start"] = df["time"].dt.to_period("W-SUN").apply(lambda p: p.start_time)

    agg = (
        df.groupby("week_start", as_index=False)
        .agg(
            n_days=("time", "size"),
            prec=("prec", lambda s: s.sum(min_count=1)),
            pres=("pres", "mean"),
            rhum=("rhum", "mean"),
            temp=("temp", "mean"),
            wind=("wind", "mean"),
            **{
                "total imported cases": ("total imported cases", "sum"),
                "total indigenous cases": ("total indigenous cases", "sum"),
            },
        )
        .rename(columns={"week_start": "time"})
        .sort_values("time")
        .reset_index(drop=True)
    )

    agg = complete_weekly_index(agg)

    raw_ind = agg["total indigenous cases"].values.astype(np.float32)
    raw_imp = agg["total imported cases"].values.astype(np.float32)

    agg["raw_indigenous"] = raw_ind


    agg["total indigenous cases"] = (
        pd.Series(raw_ind).rolling(2, min_periods=1).mean().values.astype(np.float32)
    )

    agg["indigenous_4w_sum"] = pd.Series(raw_ind).rolling(4, min_periods=1).sum().values.astype(np.float32)
    agg["imported_4w_sum"] = pd.Series(raw_imp).rolling(4, min_periods=1).sum().values.astype(np.float32)
    gate_activity_w = int(
        cfg.get(
            "stage2_case_activity_window_weeks",
            cfg.get("gate_case_activity_window_weeks", 12),
        )
    )
    agg["indigenous_gate_activity_sum"] = (
        pd.Series(raw_ind).rolling(gate_activity_w, min_periods=1).sum().values.astype(np.float32)
    )
    agg["imported_gate_activity_sum"] = (
        pd.Series(raw_imp).rolling(gate_activity_w, min_periods=1).sum().values.astype(np.float32)
    )
    agg["indigenous_diff"] = np.concatenate([[0], np.diff(raw_ind)]).astype(np.float32)
    agg["indigenous_acc"] = np.concatenate([[0, 0], np.diff(raw_ind, 2)]).astype(np.float32)


    agg["year"] = agg["time"].dt.year
    agg["indigenous_year_cum"] = agg.groupby("year")["raw_indigenous"].cumsum().astype(np.float32)
    agg["imported_year_cum"] = agg.groupby("year")["total imported cases"].cumsum().astype(np.float32)


    iw = agg["time"].dt.isocalendar().week.astype(np.float32)
    agg["week_sin"] = np.sin(2 * np.pi * iw / 53).astype(np.float32)
    agg["week_cos"] = np.cos(2 * np.pi * iw / 53).astype(np.float32)

    agg["city"] = city
    agg["geo_city"] = geo_city
    agg["province"] = prov


    feat_cols = list(MODEL_FEATURES_47)
    return agg, feat_cols


def _weekly_band(src, start_week, end_week, method):

    width = int(end_week - start_week + 1)
    shifted = src.shift(int(start_week - 1))

    roller = shifted.rolling(width, min_periods=width)
    return roller.sum() if method == "sum" else roller.mean()


def add_weekly_protocol_features(g):

    g = g.sort_values("time").copy()
    common_bands = [(1, 1), (2, 4), (5, 7), (8, 10), (11, 13)]
    for raw_col, prefix in [("total imported cases", "import"), ("raw_indigenous", "local")]:
        src = pd.to_numeric(g[raw_col], errors="coerce").fillna(0.0)
        for start, end in common_bands:
            values = _weekly_band(src, start, end, "sum")
            tag = "lag1w" if start == end == 1 else f"lag{start}_{end}w"
            g[f"{prefix}_{tag}_log1p"] = np.log1p(values.clip(lower=0.0))

    for raw_col in ["temp", "prec", "rhum", "wind", "pres"]:
        src = pd.to_numeric(g[raw_col], errors="coerce")
        method = "sum" if raw_col == "prec" else "mean"
        for start, end in common_bands:
            values = _weekly_band(src, start, end, method)
            if start == end == 1:
                tag = "lag1w"
            else:
                tag = f"lag{start}_{end}w"
            g[f"{raw_col}_{tag}_{method}"] = values

    for raw_col, prefix in [
        ("province_other_imported_cases", "province_other_import"),
        ("province_other_local_cases", "province_other_local"),
    ]:
        src = pd.to_numeric(g[raw_col], errors="coerce").fillna(0.0)
        for start, end in common_bands:
            values = _weekly_band(src, start, end, "sum")
            tag = "lag1w" if start == end == 1 else f"lag{start}_{end}w"
            g[f"{prefix}_{tag}_log1p"] = np.log1p(values.clip(lower=0.0))
    return g


def attach_continuous_weekly_features(temp_records, cfg):


    parts = []
    for _, df in temp_records:
        cols = ["province", "geo_city", "time", "n_days", "prec", "pres", "rhum", "temp", "wind",
                "total imported cases", "raw_indigenous"]
        parts.append(df[cols].copy())
    base = pd.concat(parts, ignore_index=True)
    base["n_days"] = pd.to_numeric(base["n_days"], errors="coerce").fillna(0.0)
    for c in ["pres", "rhum", "temp", "wind"]:
        base[f"_{c}_weighted"] = pd.to_numeric(base[c], errors="coerce") * base["n_days"]
    agg = base.groupby(["province", "geo_city", "time"], as_index=False).agg(
        n_days=("n_days", "sum"),
        prec=("prec", lambda s: s.sum(min_count=1)),
        **{
            "total imported cases": ("total imported cases", "sum"),
            "raw_indigenous": ("raw_indigenous", "sum"),
            "_pres_weighted": ("_pres_weighted", lambda s: s.sum(min_count=1)),
            "_rhum_weighted": ("_rhum_weighted", lambda s: s.sum(min_count=1)),
            "_temp_weighted": ("_temp_weighted", lambda s: s.sum(min_count=1)),
            "_wind_weighted": ("_wind_weighted", lambda s: s.sum(min_count=1)),
        },
    )
    denom = agg["n_days"].replace(0, np.nan)
    for c in ["pres", "rhum", "temp", "wind"]:
        agg[c] = agg[f"_{c}_weighted"] / denom


    province_totals = agg.groupby(["province", "time"])[
        ["total imported cases", "raw_indigenous"]
    ].transform("sum")
    agg["province_other_imported_cases"] = (
        province_totals["total imported cases"] - agg["total imported cases"]
    ).clip(lower=0.0)
    agg["province_other_local_cases"] = (
        province_totals["raw_indigenous"] - agg["raw_indigenous"]
    ).clip(lower=0.0)
    feature_parts = []
    for (province, geo_city), city_g in agg.groupby(["province", "geo_city"], sort=False):
        city_g = city_g.sort_values("time").reset_index(drop=True).copy()

        for c in METEO:
            city_g[c] = pd.to_numeric(city_g[c], errors="coerce").ffill()

        raw_ind = pd.to_numeric(city_g["raw_indigenous"], errors="coerce").fillna(0.0)
        raw_imp = pd.to_numeric(city_g["total imported cases"], errors="coerce").fillna(0.0)
        city_g["raw_indigenous"] = raw_ind.astype(np.float32)
        city_g["total imported cases"] = raw_imp.astype(np.float32)
        city_g["total indigenous cases"] = raw_ind.rolling(2, min_periods=1).mean().astype(np.float32)
        city_g["indigenous_4w_sum"] = raw_ind.rolling(4, min_periods=1).sum().astype(np.float32)
        city_g["imported_4w_sum"] = raw_imp.rolling(4, min_periods=1).sum().astype(np.float32)
        gate_activity_w = int(
            cfg.get(
                "stage2_case_activity_window_weeks",
                cfg.get("gate_case_activity_window_weeks", 12),
            )
        )
        city_g["indigenous_gate_activity_sum"] = raw_ind.rolling(gate_activity_w, min_periods=1).sum().astype(np.float32)
        city_g["imported_gate_activity_sum"] = raw_imp.rolling(gate_activity_w, min_periods=1).sum().astype(np.float32)
        city_g["indigenous_diff"] = np.concatenate([[0.0], np.diff(raw_ind.values)]).astype(np.float32)
        city_g["indigenous_acc"] = np.concatenate([[0.0, 0.0], np.diff(raw_ind.values, 2)]).astype(np.float32)
        city_g["year"] = pd.to_datetime(city_g["time"]).dt.year.astype(int)
        city_g["indigenous_year_cum"] = city_g.groupby("year")["raw_indigenous"].cumsum().astype(np.float32)
        city_g["imported_year_cum"] = city_g.groupby("year")["total imported cases"].cumsum().astype(np.float32)
        iw = pd.to_datetime(city_g["time"]).dt.isocalendar().week.astype(np.float32)
        city_g["week_sin"] = np.sin(2 * np.pi * iw / 53).astype(np.float32)
        city_g["week_cos"] = np.cos(2 * np.pi * iw / 53).astype(np.float32)
        city_g["province"] = province
        city_g["geo_city"] = geo_city
        city_g["city"] = geo_city

        city_g = add_weekly_protocol_features(city_g)
        missing = [c for c in MODEL_FEATURES_47 if c not in city_g.columns]
        if missing:
            raise RuntimeError(f"Missing weekly protocol features: {missing}")
        feature_parts.append(city_g[["province", "geo_city", "time"] + list(MODEL_FEATURES_47)])

    feature_lookup = pd.concat(feature_parts, ignore_index=True)
    key_cols = ["province", "geo_city", "time"]
    if feature_lookup.duplicated(key_cols).any():
        raise RuntimeError("Continuous feature lookup contains duplicate city-week keys.")

    updated = []
    for annual_file_id, annual_df in temp_records:


        drop_cols = [c for c in MODEL_FEATURES_47 if c in annual_df.columns]
        base_df = annual_df.drop(columns=drop_cols)
        merged = base_df.merge(
            feature_lookup,
            on=key_cols,
            how="left",
            validate="many_to_one",
            indicator="_feature_merge",
        )
        if (merged["_feature_merge"] != "both").any():
            bad = merged.loc[merged["_feature_merge"] != "both", key_cols].head(5)
            raise RuntimeError(f"Annual rows were not matched to continuous features:\n{bad}")
        merged = merged.drop(columns=["_feature_merge"])
        missing = [c for c in MODEL_FEATURES_47 if c not in merged.columns]
        if missing:
            raise RuntimeError(f"Missing weekly protocol features after merge: {missing}")

        updated.append((annual_file_id, merged))
    return updated


def discretize_serial_interval_weekly(mean_days=18.2, sd_days=6.1, max_lag_weeks=8):

    max_lag_weeks = int(max_lag_weeks)
    if max_lag_weeks < 1:
        return np.array([1.0], dtype=np.float64)

    shape = (mean_days / sd_days) ** 2
    scale = (sd_days ** 2) / mean_days
    probs = []
    for lag in range(1, max_lag_weeks + 1):
        lo = max(0.0, (lag - 1) * 7.0)
        hi = lag * 7.0
        probs.append(float(gamma.cdf(hi, a=shape, scale=scale) - gamma.cdf(lo, a=shape, scale=scale)))
    w = np.asarray(probs, dtype=np.float64)

    w = np.maximum(w, 0.0)
    if w.sum() <= 0:
        w[:] = 1.0
    return (w / w.sum()).astype(np.float64)


def remove_short_true_runs(mask, min_run=1):

    return remove_short(mask, int(min_run))


def _forward_rolling_sum(arr, window):

    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    out = np.zeros(n, dtype=float)
    for i in range(n):
        out[i] = float(np.nansum(arr[i:min(n, i + int(window))]))
    return out


def compute_wt_rc_labels_for_city(df, cfg):


    g = df.sort_values("time").reset_index(drop=True).copy()
    n = len(g)
    local = pd.to_numeric(g.get("raw_indigenous", 0.0), errors="coerce").fillna(0.0).values.astype(float)
    imported = pd.to_numeric(g.get("total imported cases", 0.0), errors="coerce").fillna(0.0).values.astype(float)

    imported_weight = float(cfg.get("wt_imported_infectious_weight", 0.5))
    source = np.maximum(local, 0.0) + imported_weight * np.maximum(imported, 0.0)
    target = np.maximum(local, 0.0)

    w = discretize_serial_interval_weekly(
        cfg.get("wt_si_mean_days", 18.2),
        cfg.get("wt_si_sd_days", 6.1),
        cfg.get("wt_max_lag_weeks", 8),
    )
    L = len(w)
    secondary = np.zeros(n, dtype=float)


    for j in range(n):
        if target[j] <= 0:
            continue
        lo = max(0, j - L)
        if lo >= j:
            continue
        idxs = np.arange(lo, j)
        lags = j - idxs
        weights = source[idxs] * w[lags - 1]
        denom = float(weights.sum())
        if denom <= 0:
            continue
        secondary[idxs] += target[j] * weights / denom


    r_window = max(1, int(cfg.get("wt_rc_window_weeks", 1)))
    sec_w = pd.Series(secondary).rolling(r_window, center=False, min_periods=1).sum().values
    src_w = pd.Series(source).rolling(r_window, center=False, min_periods=1).sum().values
    wt_rc = np.divide(sec_w, np.maximum(src_w, 1e-8))

    right_censor = int(cfg.get("wt_exclude_right_censored_weeks", L))
    raw_wt_signal = (
        (wt_rc > float(cfg.get("wt_rc_threshold", 1.0)))
        & (src_w >= float(cfg.get("wt_min_source_cases_window", 1.0)))
    )
    if right_censor > 0 and n > right_censor:
        raw_wt_signal[n - right_censor:] = False
    wt_signal = remove_short_true_runs(raw_wt_signal, cfg.get("wt_min_signal_run_weeks", 4))


    confirmed_signal = wt_signal.copy()

    onset_marker = np.zeros(n, dtype=bool)
    in_period = False
    for i in range(n):
        if confirmed_signal[i] and not in_period:
            onset_marker[i] = True
            in_period = True
        elif not confirmed_signal[i]:
            in_period = False


    confirm_w = 4
    forward_local = _forward_rolling_sum(local, confirm_w)
    forward_nonzero = _forward_rolling_sum((local > 0).astype(float), confirm_w)
    forward_peak = np.zeros(n, dtype=float)
    for i in range(n):
        forward_peak[i] = float(np.nanmax(local[i:min(n, i + confirm_w)])) if i < n else 0.0

    g["WT_source_cases"] = source.astype(np.float32)
    g["WT_secondary_expected"] = secondary.astype(np.float32)
    g["WT_secondary_window"] = sec_w.astype(np.float32)
    g["WT_source_window"] = src_w.astype(np.float32)
    g["WT_secondary_smooth"] = sec_w.astype(np.float32)
    g["WT_source_smooth"] = src_w.astype(np.float32)
    g["WT_Rc"] = wt_rc.astype(np.float32)
    g["WT_Rc_smooth"] = wt_rc.astype(np.float32)
    g["wt_raw_signal"] = raw_wt_signal.astype(int)
    g["wt_signal"] = wt_signal.astype(int)
    g["wt_confirmed_signal"] = confirmed_signal.astype(int)
    g["wt_onset_marker"] = onset_marker.astype(int)
    g["wt_forward_local_cases_4w"] = forward_local.astype(np.float32)
    g["wt_forward_nonzero_weeks_4w"] = forward_nonzero.astype(np.float32)
    g["wt_forward_peak_cases_4w"] = forward_peak.astype(np.float32)
    g["wt_signal_candidate_onset_idx"] = np.where(onset_marker, np.arange(n), -1).astype(int)
    g["wt_candidate_pass_forward"] = confirmed_signal.astype(int)
    g["outbreak_threshold_used"] = float(cfg.get("wt_rc_threshold", 1.0))
    g["outbreak_threshold_comparator"] = ">"
    g["outbreak_min_consecutive_weeks"] = int(
        cfg.get("wt_min_signal_run_weeks", 4)
    )
    return g


def consolidate_physical_city_weeks(featured_records, cfg):


    if not featured_records:
        return []

    base_cols = [
        "province", "geo_city", "time", "n_days", "prec", "pres", "rhum", "temp", "wind",
        "total imported cases", "raw_indigenous",
    ]
    all_rows = pd.concat([df[base_cols + list(MODEL_FEATURES_47)].copy()
                          for _, df in featured_records], ignore_index=True)
    key = ["province", "geo_city", "time"]


    feature_lookup = all_rows[key + list(MODEL_FEATURES_47)].drop_duplicates(key, keep="first")

    for c in ["pres", "rhum", "temp", "wind"]:
        all_rows[f"_{c}_weighted"] = (
            pd.to_numeric(all_rows[c], errors="coerce")
            * pd.to_numeric(all_rows["n_days"], errors="coerce").fillna(0.0)
        )
    weekly = all_rows.groupby(key, as_index=False).agg(
        n_days=("n_days", "sum"),
        prec=("prec", lambda s: s.sum(min_count=1)),
        **{
            "total imported cases": ("total imported cases", "sum"),
            "raw_indigenous": ("raw_indigenous", "sum"),
            "_pres_weighted": ("_pres_weighted", lambda s: s.sum(min_count=1)),
            "_rhum_weighted": ("_rhum_weighted", lambda s: s.sum(min_count=1)),
            "_temp_weighted": ("_temp_weighted", lambda s: s.sum(min_count=1)),
            "_wind_weighted": ("_wind_weighted", lambda s: s.sum(min_count=1)),
        },
    )
    denom = weekly["n_days"].replace(0, np.nan)
    for c in ["pres", "rhum", "temp", "wind"]:
        weekly[c] = weekly[f"_{c}_weighted"] / denom
    weekly = weekly.drop(columns=[c for c in weekly.columns if c.startswith("_")])
    weekly = weekly.merge(feature_lookup, on=key, how="left", validate="one_to_one")

    weekly["city"] = weekly["geo_city"]
    weekly["year"] = pd.to_datetime(weekly["time"]).dt.year.astype(int)
    records = []
    dropped_short = 0
    for (province, city, year), g in weekly.groupby(["province", "geo_city", "year"], sort=True):
        g = g.sort_values("time").reset_index(drop=True).copy()

        if len(g) < 40:
            dropped_short += 1
            continue
        if g["time"].duplicated().any():
            raise RuntimeError(f"Duplicate physical city-week after consolidation: {province}/{city}/{year}")
        gaps = pd.to_datetime(g["time"]).diff().dropna().dt.days
        if len(gaps) and not gaps.eq(7).all():
            raise RuntimeError(f"Non-weekly gap after consolidation: {province}/{city}/{year}")


        g = compute_wt_rc_labels_for_city(g, cfg)
        raw_ind = pd.to_numeric(g["raw_indigenous"], errors="coerce").fillna(0.0)
        raw_imp = pd.to_numeric(g["total imported cases"], errors="coerce").fillna(0.0)
        g["total indigenous cases"] = raw_ind.rolling(2, min_periods=1).mean().astype(np.float32)
        g["indigenous_4w_sum"] = raw_ind.rolling(4, min_periods=1).sum().astype(np.float32)
        g["imported_4w_sum"] = raw_imp.rolling(4, min_periods=1).sum().astype(np.float32)
        gate_activity_w = int(
            cfg.get(
                "stage2_case_activity_window_weeks",
                cfg.get("gate_case_activity_window_weeks", 12),
            )
        )
        g["indigenous_gate_activity_sum"] = raw_ind.rolling(gate_activity_w, min_periods=1).sum().astype(np.float32)
        g["imported_gate_activity_sum"] = raw_imp.rolling(gate_activity_w, min_periods=1).sum().astype(np.float32)

        hit = np.flatnonzero(g["wt_onset_marker"].to_numpy(dtype=int) == 1)
        onset = int(hit[0]) if len(hit) else -1
        records.append(dict(
            province=province,
            city=city,
            geo_city=city,
            year=int(year),
            has_outbreak=onset >= 0,
            onset_idx=None if onset < 0 else onset,
            outbreak_threshold=float(cfg.get("wt_rc_threshold", 1.0)),
            data=g,
        ))

    expected = len({(r["province"], r["city"], r["year"]) for r in records})
    if expected != len(records):
        raise RuntimeError("Physical city-year consolidation did not produce unique records")
    print(f"  Consolidated physical city-years={len(records)}; dropped short boundary fragments={dropped_short}")
    return records


def collect_records(cfg):
    candidate_files = sorted([
        f for f in glob.glob(os.path.join(cfg["data_dir"], "*.csv"))
        if any(r in os.path.basename(f) for r in cfg["regions"])
           or "\u5e7f\u897f\u58ee\u65cf\u81ea\u6cbb\u533a" in os.path.basename(f)
    ])


    if "input_max_file_year" in cfg:
        lo = int(cfg.get("input_min_file_year", 2005))
        hi = int(cfg["input_max_file_year"])
        candidate_files = [
            path for path in candidate_files
            if re.search(r"_(\d{4})\.csv$", os.path.basename(path))
            and lo <= int(re.search(r"_(\d{4})\.csv$", os.path.basename(path)).group(1)) <= hi
        ]

    manifest_dir = str(cfg.get("file_manifest_dir", "") or "").strip()
    if manifest_dir:
        manifest_files = sorted(glob.glob(os.path.join(manifest_dir, "*.csv")))
        if not manifest_files:
            raise RuntimeError(f"No CSV files in the manifest directory: {manifest_dir}")
        allowed_names = {os.path.basename(f) for f in manifest_files}
        available = {os.path.basename(f): f for f in candidate_files}
        missing_names = sorted(allowed_names - set(available))
        extra_names = sorted(set(available) - allowed_names)
        if missing_names:
            preview = ", ".join(missing_names[:5])
            raise RuntimeError(
                f"The data directory is missing {len(missing_names)} CSV files listed in the manifest; examples: {preview}"
            )
        files = sorted(available[name] for name in allowed_names)
        print("\n[Input manifest filtering]")
        print(f"  Data directory: {cfg['data_dir']}")
        print(f"  Manifest directory: {manifest_dir}")
        print(f"  Manifest CSV files: {len(allowed_names)}")
        print(f"  CSV files selected: {len(files)}")
        print(f"  CSV files excluded from the data directory: {len(extra_names)}")
    else:
        files = candidate_files

    temp_records = []
    fcols = None
    failed = []

    for fp in files:
        try:
            df, fc = read_city(fp, cfg)
            if fcols is None:
                fcols = fc
            city_name = os.path.basename(fp).replace(".csv", "")
            temp_records.append((city_name, df))
        except Exception as e:
            failed.append((fp, str(e)))
            continue

    if failed:
        detail = "; ".join(f"{fp}: {err}" for fp, err in failed[:5])
        raise RuntimeError(f"Failed to read {len(failed)} input files: {detail}")
    if not temp_records:
        raise RuntimeError("No eligible raw city-year CSV files were found.")

    if temp_records:
        temp_records = attach_continuous_weekly_features(temp_records, cfg)
    records = consolidate_physical_city_weeks(temp_records, cfg)

    print("\n[WT-Rc Outbreak Label Definition]")
    print(
        "  Generation interval mean/sd days: "
        f"{cfg.get('wt_si_mean_days')} / {cfg.get('wt_si_sd_days')}"
    )
    print(f"  max lag weeks: {cfg.get('wt_max_lag_weeks')}")
    print(f"  Rc threshold: {cfg.get('wt_rc_threshold')}")
    print(f"  minimum source cases in Rc window: {cfg.get('wt_min_source_cases_window')}")
    print(f"  minimum consecutive Rc-signal weeks: {cfg.get('wt_min_signal_run_weeks')}")
    print("  EWARS/endemic-channel confirmation: disabled")
    print("  WT-Rc time-series scope: reset by consolidated physical-city calendar year")
    print("  Feature time-series scope: continuous physical-city history across years")
    print("  NOTE: WT-Rc variables are retrospective labels only and are not model features.")

    print(f"  Loaded physical city-year records={len(records)}, WT-Rc outbreak records={sum(int(r['has_outbreak']) for r in records)}")
    return records, fcols


def make_label(rec, idx, horizon):
    n = len(rec["data"])
    onset = rec["onset_idx"]

    if onset is not None and idx > onset:
        return None


    n_time_points = int(horizon) + 1
    y_haz = np.zeros(n_time_points, dtype=np.float32)
    m_haz = np.zeros(n_time_points, dtype=np.float32)
    ttw = np.nan

    if onset is not None:
        ttw = float(onset - idx)
        if 0 <= ttw <= horizon:
            b = int(ttw)
            m_haz[:b + 1] = 1.0
            y_haz[b] = 1.0
        else:
            m_haz[:] = 1.0
    else:
        ob = n - 1 - idx
        if ob >= horizon:
            m_haz[:] = 1.0
        elif ob >= 0:

            m_haz[:ob + 1] = 1.0
        else:
            return None

    return y_haz, m_haz, ttw


def iter_windows(records, rec, fcols, ws):
    lut = {(r["province"], r["city"], r["year"]): r for r in records}
    prev = lut.get((rec["province"], rec["city"], rec["year"] - 1))

    if prev is None:
        hist = np.empty((0, len(fcols)), dtype=np.float32)
    else:
        hist = prev["data"].sort_values("time")[fcols].values.astype(np.float32)[-ws:]

    g = rec["data"].sort_values("time").reset_index(drop=True)
    seq = np.concatenate([hist, g[fcols].values.astype(np.float32)])


    for i in range(max(0, ws - 1 - len(hist)), len(g)):
        end = len(hist) + i + 1
        win = seq[end - ws:end]
        if len(win) == ws and np.isfinite(win).all():
            yield i, win


def build_data(records, split, fcols, cfg):
    X, Y_haz, M_haz, Y_ttw, meta = [], [], [], [], []

    for r in records:
        if r.get("split") != split:
            continue

        g = r["data"].sort_values("time").reset_index(drop=True)

        for idx, win in iter_windows(records, r, fcols, cfg["window_size"]):
            lab = make_label(r, idx, cfg["horizon_weeks"])
            if lab is None:
                continue

            yh, mh, ttw = lab
            event_within_horizon = bool(np.any((yh > 0.5) & (mh > 0.5)))
            complete_horizon = bool(np.all(mh > 0.5))
            X.append(win)
            Y_haz.append(yh)
            M_haz.append(mh)
            Y_ttw.append(ttw)

            meta.append(dict(
                split=split,
                province=r["province"],
                city=r["city"],
                year=r["year"],
                sample_idx=idx,
                date=str(g.loc[idx, "time"]),
                onset_idx_raw=-1 if r["onset_idx"] is None else r["onset_idx"],
                raw_ttw=ttw,
                raw_total_indigenous_cases=float(g.loc[idx, "raw_indigenous"]),
                indigenous_4w_sum=float(g.loc[idx, "indigenous_4w_sum"]) if "indigenous_4w_sum" in g.columns else 0.0,
                imported_4w_sum=float(g.loc[idx, "imported_4w_sum"]) if "imported_4w_sum" in g.columns else 0.0,
                indigenous_gate_activity_sum=float(g.loc[idx, "indigenous_gate_activity_sum"]) if "indigenous_gate_activity_sum" in g.columns else 0.0,
                imported_gate_activity_sum=float(g.loc[idx, "imported_gate_activity_sum"]) if "imported_gate_activity_sum" in g.columns else 0.0,
                outbreak_threshold_used=float(r["outbreak_threshold"]),
                horizon_label_valid=int(event_within_horizon or complete_horizon),
                observed_time_points=int(np.sum(mh > 0.5)),
                record_last_idx=int(len(g) - 1),
            ))

    return (
        np.array(X, dtype=np.float32),
        np.array(Y_haz, dtype=np.float32),
        np.array(M_haz, dtype=np.float32),
        np.array(Y_ttw, dtype=np.float32),
        pd.DataFrame(meta),
    )


class Stage1SurvivalTrigger:


    def __init__(self, cfg):
        self.cfg = cfg
        self.horizon = int(cfg["horizon_weeks"])
        self.n_time_points = self.horizon + 1
        self.model = RandomForestClassifier(
            n_estimators=int(cfg.get("rf_n_estimators", 500)),
            max_depth=cfg.get("rf_max_depth", 12),
            min_samples_leaf=int(cfg.get("rf_min_samples_leaf", 4)),
            max_features=cfg.get("rf_max_features", "sqrt"),
            max_samples=cfg.get("rf_max_samples", None),
            class_weight="balanced_subsample",
            random_state=cfg["seed"],
            n_jobs=CPU_THREADS,
            warm_start=bool(cfg.get("rf_warm_start", False)),
        )

    def fit(self, X, Y_haz, M_haz):
        X_flat = X.reshape(X.shape[0], -1)
        X_exp, Y_exp = [], []

        if Y_haz.shape[1] != self.n_time_points or M_haz.shape[1] != self.n_time_points:
            raise ValueError(
                f"Stage1 expects {self.n_time_points} time points (TTW=0..{self.horizon}), "
                f"got Y={Y_haz.shape}, M={M_haz.shape}"
            )
        for t in range(self.n_time_points):
            valid_mask = M_haz[:, t] == 1.0
            if not np.any(valid_mask):
                continue
            X_t = X_flat[valid_mask]
            y_t = Y_haz[valid_mask, t]

            time_feat = np.full((len(X_t), 1), t, dtype=np.float32)
            X_expanded = np.hstack([X_t, time_feat])
            X_exp.append(X_expanded)
            Y_exp.append(y_t)

        X_exp = np.vstack(X_exp)
        Y_exp = np.concatenate(Y_exp)
        self.model.fit(X_exp, Y_exp)

    def _predict_hazard_at_t(self, X_flat, t):
        N = len(X_flat)
        time_feat = np.full((N, 1), t, dtype=np.float32)
        X_t = np.hstack([X_flat, time_feat])
        proba = self.model.predict_proba(X_t)

        if len(self.model.classes_) == 1:
            hazard_t = (
                np.ones(X_t.shape[0], dtype=np.float32)
                if int(self.model.classes_[0]) == 1
                else np.zeros(X_t.shape[0], dtype=np.float32)
            )
        else:
            pos_idx = list(self.model.classes_).index(1)
            hazard_t = proba[:, pos_idx].astype(np.float32)

        return np.clip(hazard_t, 0.0, 1.0)

    def predict_12w_raw(self, X):


        X_flat = X.reshape(X.shape[0], -1)
        N = len(X_flat)
        hazards = np.column_stack([
            self._predict_hazard_at_t(X_flat, t) for t in range(self.n_time_points)
        ]).astype(np.float64)
        surv_before = np.ones_like(hazards)
        if self.n_time_points > 1:
            surv_before[:, 1:] = np.cumprod(1.0 - hazards[:, :-1], axis=1)
        event_mass = surv_before * hazards
        p_12w = np.clip(event_mass.sum(axis=1), 0.0, 1.0)
        denom = np.maximum(p_12w[:, None], 1e-12)
        cond_pmf = event_mass / denom
        tiny = p_12w <= 1e-12
        if np.any(tiny):
            cond_pmf[tiny] = 0.0
            cond_pmf[tiny, -1] = 1.0
        cond_cdf = np.cumsum(cond_pmf, axis=1)

        out = {"p_12w_raw": p_12w.astype(np.float32)}
        for t in range(self.n_time_points):
            out[f"hazard_t{t}"] = hazards[:, t].astype(np.float32)
            out[f"event_mass_t{t}"] = event_mass[:, t].astype(np.float32)
            out[f"hazard_conditional_pmf_t{t}"] = cond_pmf[:, t].astype(np.float32)
        for alpha, tag in [(0.30, "q30"), (0.50, "q50"), (0.70, "q70")]:
            q = np.argmax(cond_cdf >= alpha, axis=1)
            out[f"hazard_pred_ttw_{tag}"] = q.astype(np.float32)
        grid = np.arange(self.n_time_points, dtype=float)
        out["hazard_pred_ttw_mean"] = (cond_pmf * grid[None, :]).sum(axis=1).astype(np.float32)
        return out

    def predict_multi_horizon_raw(self, X):

        out = self.predict_12w_raw(X)
        out["p_horizon_raw"] = out["p_12w_raw"]
        return out
