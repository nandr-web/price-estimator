# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Price estimator for aerospace machine shop quotes. Takes 510 historical quotes from a CSV and builds ML models (M0 lookup table through M9 debiased XGBoost) to predict `TotalPrice_USD`. Part of an AORA take-home assessment with 5 deliverables: price prediction, feature extraction from part descriptions, estimator bias analysis, human-in-the-loop API, and missing variable identification.

## Commands

```bash
# Setup
uv venv && source .venv/bin/activate && uv pip install -e ".[dev]"

# Lint
ruff check src/ tests/ scripts/
ruff format --check src/ tests/ scripts/

# Test
pytest                                    # all tests
pytest --cov=price_estimator              # with coverage
pytest tests/test_features.py -v          # single file
pytest -m "not slow"                      # skip model training tests

# Full pipeline
python scripts/eda.py --data resources/aora_historical_quotes.csv --output outputs/ \
  && python scripts/train.py --data resources/aora_historical_quotes.csv --output outputs/ \
  && python scripts/evaluate.py --results outputs/results/cv_fold_results.csv --output outputs/ \
  && python scripts/bias_analysis.py --data resources/aora_historical_quotes.csv --output outputs/

# API
python scripts/serve.py --model outputs/models/M6.joblib --port 8000
```

## Architecture

- **`src/price_estimator/`** — All business logic as importable Python. Every function is testable, callable from scripts, notebooks, or the API.
  - `data.py` — CSV loading, schema validation, missing value analysis. Does NOT impute — imputation belongs inside sklearn Pipelines during CV to prevent data leakage.
  - `features.py` — Tier 1 PartDescription parser (deterministic dict registry + rapidfuzz for typo tolerance), feature matrix construction with `onehot` (linear models) and `label` (tree models) encoding modes. `build_feature_matrix_no_estimator()` is used for the debiased model M9.
  - `models.py` — M0–M9 model definitions, training, cross-validation
  - `bias.py` — Estimator bias analysis (residuals from debiased M9)
  - `predict.py` — Prediction with bands, SHAP explanations, OOD detection
  - `api.py` — FastAPI endpoints: POST /quote, POST /quote/{id}/override, GET /quote/{id}
- **`scripts/`** — CLI entry points that call into `src/`. Each is a self-contained workflow step.
- **`notebooks/`** — Presentation only; imports from `src/`, loads pre-computed results from `outputs/`. Contains no business logic.
- **`outputs/`** — Serialized models (joblib), figures, result CSVs/JSONs. Gitignored.

## Key Design Decisions

- **Log-target models require Jensen's correction**: `exp(log_pred + 0.5 * sigma²)` before computing metrics to avoid systematic underestimation.
- **Imputation within CV folds only** — use sklearn Pipeline, never impute before splitting.
- **PartDescription registry is a Python dict** (not YAML) — 10 known part types, 9 modifiers, fuzzy match threshold of 85.
- **Ordinal tiers**: `MATERIAL_COST_TIER` (Al6061=1 to Inconel=5) and `PROCESS_PRECISION_TIER` (Surface Grinding=1 to 5-Axis=5) in features.py.
- **Estimator bias is only meaningful after controlling for job assignment** — check confounding before labeling anyone "safe" or "aggressive."

## Testing Conventions

- `conftest.py` has `sample_data` fixture (3-row synthetic DataFrame) and `raw_data` fixture (loads actual CSV).
- Test markers: `slow` (model training), `api` (endpoint tests), `integration` (real CSV data).
- ruff lint rules: E, F, I, W, UP. Line length 99.
