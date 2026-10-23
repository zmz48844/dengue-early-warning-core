# Dengue outbreak early warning

Core implementation of a two-stage weekly early-warning framework. Stage 1 estimates 12-week outbreak risk using a discrete-time survival random forest. Stage 2 estimates Q20, Q50 and Q80 time-to-onset quantiles using histogram gradient boosting.

## Installation

Use Python 3.13.5 and install the dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

## Files

```text
run.py                    Command-line entry point
src/common/core.py        Weekly aggregation, outbreak labels, features and survival forest
src/common/data_builder.py Input construction, cohort checks and caches
src/common/modeling.py    Model fitting, calibration, warning rules and metrics
src/common/paths.py       Data and output paths
src/main/candidates.py   Fixed Sobol candidate generation
src/main/engine.py        Model selection and evaluation components
src/main/runner.py        Development selection, refitting and locked temporal evaluation
config/data.json         Data boundaries and label parameters
examples/daily_columns.csv Input-column template
```

## Data

Supply the original daily city-year CSV files using `--data-dir`, or place them in `data/raw/`. Files must be UTF-8 and have names ending in `_YYYY.csv`. Keep the original province and city labels in filenames. Province-matching labels are stored as Unicode escapes in `config/data.json` to preserve the original input selection.

Required columns are `time`, `prec`, `pres`, `rhum`, `temp`, `wind`, `total imported cases` and `total indigenous cases`. Observations are aggregated to Monday-Sunday weeks. Case counts and precipitation are summed; the other meteorological variables are averaged. Case values missing from the source are treated as zero; meteorological gaps use preceding observations.

The study configuration uses 47 predictors, development years 2005-2018 and temporal evaluation years 2019-2023. Expected cohort counts are fixed to the study dataset. Surveillance data, input caches and fitted models are not included.

## Run

```bash
python -X utf8 -B run.py prepare --run-id study_001 --data-dir /path/to/daily_csv
python -X utf8 -B run.py train --run-id study_001 --data-dir /path/to/daily_csv
python -X utf8 -B run.py test --run-id study_001 --data-dir /path/to/daily_csv
python -X utf8 -B run.py status --run-id study_001
```

`prepare` constructs development inputs and fixes grouped folds. `train` selects candidates using three-fold city-year development OOF predictions, locks both target-transformation systems, and refits on development data. `test` evaluates the locked systems. `all` performs the full sequence.

Outputs are written to `results/<run-id>/`, including weekly predictions, event-level warning metrics, countdown metrics, selection records and protocol locks. Use a fresh run ID after changing source files, configuration or data. Do not use `python -O`.

The search uses 96 Stage 1 candidates and 144 matched Stage 2 structures per target transformation. Warning rules retain the fitted probability calibration, EWMA span of three, hysteresis and recent-case requirement from the original analysis. Quantile ordering is anchored at Q50 before warning levels are assigned. The numerical analysis and recorded history of earlier exploratory temporal-data access are retained.

Development OOF results are used for selection and calibration. Temporal evaluation is retrospective and does not constitute a previously untouched confirmatory test. This distribution contains the main analysis only.
