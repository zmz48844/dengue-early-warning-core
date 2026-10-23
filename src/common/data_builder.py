from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import core, paths

KEY = ["province", "city", "year", "sample_idx"]
CY = ["province", "city", "year"]
CACHE_NAMES = {
    "development": "development_47feature_cache.joblib",
    "test": "test_47feature_cache.joblib",
}
SCHEMA_VERSION = 1


def _digest(path):
    result = hashlib.sha256()
    with open(core._windows_extended_path(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _signature(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def load_config(max_year=None, raw_data_dir=None):

    with open(core._windows_extended_path(paths.DATA_CONFIG_PATH), encoding="utf-8-sig") as handle:
        cfg = json.load(handle)
    if cfg["window_size"] != 1 or cfg["horizon_weeks"] != 12 or cfg["feature_count"] != 47:
        raise ValueError("This frozen project requires window=1, horizon=12 and 47 features.")
    cfg["data_dir"] = str(Path(raw_data_dir) if raw_data_dir is not None else paths.RAW_DATA_DIR)
    cfg["input_min_file_year"] = int(cfg["study_start_year"])
    cfg["input_max_file_year"] = (
        int(max_year) if max_year is not None else int(cfg["test_start_year"]) - 1
    )
    cfg["file_manifest_dir"] = ""
    return cfg


def raw_input_files(cfg):

    lo, hi = cfg["input_min_file_year"], cfg["input_max_file_year"]
    result = []
    for path in sorted(Path(cfg["data_dir"]).glob("*.csv")):
        match = re.search(r"_(\d{4})\.csv$", path.name)
        if match is None or not lo <= int(match.group(1)) <= hi:
            continue
        if any(region in path.name for region in cfg["regions"]) or "\u5e7f\u897f\u58ee\u65cf\u81ea\u6cbb\u533a" in path.name:
            result.append(path)
    if not result:
        raise FileNotFoundError(f"No eligible raw city-year CSV files in {cfg['data_dir']}")
    return result


def _identity(split, cfg):

    stable_cfg = {k: v for k, v in cfg.items() if k not in ("data_dir", "file_manifest_dir")}
    sources = {path.name: _digest(path) for path in raw_input_files(cfg)}
    return dict(
        schema_version=SCHEMA_VERSION, split=split, config=stable_cfg,
        raw_inputs=sources, core_sha256=_digest(Path(core.__file__)),
        builder_sha256=_digest(Path(__file__)),
    )


def counts(meta):
    cityyears = meta.drop_duplicates(CY)
    return dict(
        cities=int(len(meta[["province", "city"]].drop_duplicates())),
        city_years=int(len(cityyears)),
        events=int((cityyears.onset_idx_raw >= 0).sum()),
        weeks=int(len(meta)),
        exact_rows=int(meta.raw_ttw.between(0, 12).sum()),
    )


def validate(data, split, cfg=None):
    cfg = load_config() if cfg is None else cfg
    required = {"X", "Y", "M", "ttw", "meta", "features"}
    if not required.issubset(data):
        raise ValueError(f"Input cache is missing keys: {sorted(required - set(data))}")
    meta = data["meta"]
    n = len(meta)
    if data["features"] != list(core.MODEL_FEATURES_47):
        raise ValueError("Feature names or ordering differ from the frozen 47-feature definition.")
    for key, shape in (("X", (n, 1, 47)), ("Y", (n, 13)), ("M", (n, 13)), ("ttw", (n,))):
        if data[key].shape != shape or data[key].dtype != np.float32:
            raise ValueError(f"Unexpected {key} shape/dtype: {data[key].shape}/{data[key].dtype}")
    if not np.isfinite(data["X"]).all() or meta.duplicated(KEY).any():
        raise ValueError("Nonfinite model inputs or duplicate city-year-week sample keys.")
    if not np.isfinite(data["Y"]).all() or not np.isfinite(data["M"]).all():
        raise ValueError("Nonfinite hazard labels.")
    if not np.allclose(data["ttw"], meta.raw_ttw, equal_nan=True):
        raise ValueError("TTW array and metadata differ.")
    lo, hi = (
        (cfg["study_start_year"], cfg["test_start_year"] - 1)
        if split == "development" else (cfg["test_start_year"], cfg["study_end_year"])
    )
    if not meta.year.between(lo, hi).all() or not meta["split"].eq(split).all():
        raise ValueError("A sample lies outside its requested temporal split.")
    if split == "development":
        cutoff = pd.Timestamp(year=int(hi), month=12, day=31)
        if not (pd.to_datetime(meta.date) + pd.Timedelta(days=6) <= cutoff).all():
            raise ValueError("Development samples include an incomplete cross-boundary week.")
    actual = counts(meta)
    expected = cfg.get("expected_" + split, {})
    if any(actual.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{split} data counts differ: expected={expected}, actual={actual}")
    return data


def _assemble(split, cfg):
    records, features = core.collect_records(cfg)
    if list(features) != list(core.MODEL_FEATURES_47):
        raise RuntimeError("Raw feature builder returned a different feature definition.")
    for record in records:
        record["split"] = "development" if record["year"] < cfg["test_start_year"] else "test"
    x, y, mask, ttw, meta = core.build_data(records, split, features, cfg)
    if split == "development":
        cutoff = pd.Timestamp(year=int(cfg["test_start_year"]) - 1, month=12, day=31)
        keep = (pd.to_datetime(meta.date) + pd.Timedelta(days=6) <= cutoff).to_numpy()
        x, y, mask, ttw = x[keep], y[keep], mask[keep], ttw[keep]
        meta = meta.loc[keep].reset_index(drop=True)
    meta = meta.drop(columns=["development_role"], errors="ignore").copy()
    meta["split"] = split
    return validate(dict(X=x, Y=y, M=mask, ttw=ttw, meta=meta, features=list(features)), split, cfg)


def _build(split, *, force=False, raw_data_dir=None, cache_dir=None):
    if split not in CACHE_NAMES:
        raise ValueError("Unknown temporal split")
    base = load_config(raw_data_dir=raw_data_dir)
    max_year = base["test_start_year"] - 1 if split == "development" else base["study_end_year"]
    cfg = load_config(max_year, raw_data_dir)
    folder = Path(cache_dir) if cache_dir is not None else paths.INPUT_CACHE_DIR
    path = folder / CACHE_NAMES[split]
    manifest_path = path.with_suffix(".manifest.json")
    identity = _identity(split, cfg)
    signature = _signature(identity)
    if path.exists() and manifest_path.exists() and not force:
        with open(core._windows_extended_path(manifest_path), encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("signature") != signature:
            raise RuntimeError(
                f"Input cache provenance changed: {path}. "
                "Use the data-build command with --force only after reviewing the source/configuration change."
            )
        if manifest.get("cache_sha256") != _digest(path):
            raise RuntimeError(f"Input cache content changed: {path}")
        return validate(joblib.load(core._windows_extended_path(path)), split, cfg)
    if (path.exists() or manifest_path.exists()) and not force:
        raise RuntimeError(f"Incomplete/untracked cache pair: {path}. Rebuild explicitly with --force.")
    data = _assemble(split, cfg)
    folder.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".building.joblib")
    joblib.dump(data, core._windows_extended_path(temporary), compress=3)
    temporary.replace(path)
    manifest = dict(identity=identity, signature=signature, counts=counts(data["meta"]),
                    cache_sha256=_digest(path), features=data["features"])
    temporary_manifest = manifest_path.with_suffix(".building.json")
    with open(core._windows_extended_path(temporary_manifest), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
    temporary_manifest.replace(manifest_path)
    return data


def build_development(*, force=False, raw_data_dir=None, cache_dir=None):

    return _build("development", force=force, raw_data_dir=raw_data_dir, cache_dir=cache_dir)


def build_test(*, force=False, raw_data_dir=None, cache_dir=None):

    return _build("test", force=force, raw_data_dir=raw_data_dir, cache_dir=cache_dir)


def load_development():
    return build_development()


def load_test():
    return build_test()
