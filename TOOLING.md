# Tooling, Workflow & Validation Analysis

## 1. Project Layout

All modeling, experimentation, and validation is CLI-executable. Notebooks are for presentation and visualization only — decisions are driven by numeric results from scripts.

```
part1-price-estimator/
├── pyproject.toml                  # Dependency management (uv)
├── PLAN.md
├── TOOLING.md                      # This file
├── resources/
│   └── aora_historical_quotes.csv
├── src/
│   └── price_estimator/
│       ├── __init__.py
│       ├── data.py                 # Loading, cleaning, validation, imputation
│       ├── features.py             # PartDescription parser (Tier 1 dict + rapidfuzz), feature matrix
│       ├── models.py               # M0–M9 definition, training, CV
│       ├── bias.py                 # Estimator bias analysis (Task 3)
│       ├── predict.py              # Prediction with bands, SHAP, OOD detection
│       └── api.py                  # FastAPI app (Task 4)
├── tests/
│   ├── conftest.py                 # Shared fixtures (sample data, trained model)
│   ├── test_data.py                # Schema validation, missing value handling
│   ├── test_features.py            # Parser correctness, edge cases, fuzzy matching
│   ├── test_models.py              # Smoke tests, reproducibility, metric bounds
│   ├── test_bias.py                # Bias computation sanity checks
│   └── test_api.py                 # Endpoint contracts, override storage
├── scripts/
│   ├── eda.py                      # CLI: python scripts/eda.py → stats + plots
│   ├── train.py                    # CLI: python scripts/train.py → train all models, CV results
│   ├── evaluate.py                 # CLI: python scripts/evaluate.py → comparison table, significance
│   ├── bias_analysis.py            # CLI: python scripts/bias_analysis.py → estimator profiles
│   └── serve.py                    # CLI: python scripts/serve.py → start FastAPI server
├── notebooks/
│   └── analysis.ipynb              # Presentation notebook (imports from src/, visualizes outputs/)
└── outputs/
    ├── models/                     # Serialized models (joblib)
    ├── figures/                    # EDA and result plots
    └── results/                    # CV tables, bias reports (CSV/JSON)
```

### Rationale

- **`src/price_estimator/`**: All logic lives here as importable Python. Every function is testable, every pipeline step is callable from CLI scripts, notebooks, or the API.
- **`scripts/`**: CLI entry points that parse arguments and call into `src/`. Each script is a complete, self-contained workflow step. The full pipeline is runnable as a single shell command chain.
- **`notebooks/`**: The deliverable notebook imports from `src/` and loads pre-computed results from `outputs/`. It focuses on narrative, plots, and conclusions — it is the primary artifact the assessors read, but contains no business logic. If the notebook disappears, the system still works.
- **`tests/`**: pytest suite covering data validation, feature parsing, model sanity, bias computation, and API contracts.
- **`outputs/`**: All artifacts (models, figures, result tables) written here. Gitignored except for final results.

### What changed from the original proposal

- Dropped `registry.yaml` — the Tier 1 part description registry is a Python dict in `features.py`. For 10 known part types and 9 modifiers, a dict literal is simpler, testable, and eliminates the `pyyaml` dependency. YAML would make sense if non-developers maintain the registry; for this prototype, it's unnecessary indirection.

---

## 2. Dependency Management

### Tool: `uv`

`uv` is the fastest Python package manager available. It handles virtual environments, dependency resolution, and lockfiles in one tool. Everything is CLI-driven.

```bash
# Initialize project
uv init
uv venv

# Install all dependencies
uv pip install -e ".[dev]"

# Reproducible install from lockfile
uv pip compile pyproject.toml -o requirements.lock
uv pip sync requirements.lock
```

### Why not poetry/pip/conda?

| Tool | Issue for this project |
|---|---|
| pip + requirements.txt | No lockfile, no resolution guarantees, manual venv management |
| poetry | Slower resolution, heavier config, overkill for a prototype |
| conda | Useful for binary deps (CUDA), but we have none here — pure Python suffices |
| uv | Fast, lockfile support, drop-in pip replacement, single binary |

### `pyproject.toml`

```toml
[project]
name = "price-estimator"
version = "0.1.0"
requires-python = ">=3.11"

dependencies = [
    # Data
    "pandas>=2.0,<3",
    "numpy>=1.26,<2",

    # ML
    "scikit-learn>=1.4,<2",
    "xgboost>=2.0,<3",
    "lightgbm>=4.0,<5",

    # Hyperparameter tuning
    "optuna>=3.5,<4",

    # Explainability
    "shap>=0.44,<1",

    # Prediction intervals
    "mapie>=0.8,<1",

    # Text parsing
    "rapidfuzz>=3.0,<4",

    # API
    "fastapi>=0.110,<1",
    "uvicorn[standard]>=0.29,<1",
    "pydantic>=2.0,<3",

    # Visualization
    "matplotlib>=3.8,<4",
    "seaborn>=0.13,<1",

    # Statistics
    "scipy>=1.12,<2",

    # Serialization
    "joblib>=1.3,<2",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "ruff>=0.3",
    "httpx>=0.27",          # FastAPI test client
    "jupyterlab>=4.0",
    "ipykernel>=6.0",
    "papermill>=2.5",       # CLI notebook execution
]

[tool.ruff]
line-length = 99
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "W", "UP"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
```

Also generate a `requirements.txt` for evaluators who don't use `uv`:

```bash
uv pip compile pyproject.toml -o requirements.txt
```

---

## 3. Library-by-Library Analysis

### 3.1 Libraries to KEEP (from the plan)

| Library | Role | Why it's right | Version concern |
|---|---|---|---|
| **pandas** | Data manipulation | 510 rows — pandas is the obvious choice. No need for polars/dask/spark. | >=2.0 for CoW and pyarrow backend |
| **scikit-learn** | Ridge, Lasso, RF, CV, metrics, pipelines | The backbone. Handles imputation within CV folds via `Pipeline`, provides `cross_val_score`, all standard metrics. | >=1.4 for improved `set_output` API |
| **XGBoost** | Primary tree model | Best-in-class for tabular data. Native handling of missing values. `xgb.XGBRegressor` fits into sklearn pipelines. | >=2.0 for improved categorical support |
| **LightGBM** | Alternative tree model | Leaf-wise splitting may perform differently on 510 rows. Native categorical feature support (no encoding needed). Quick to add since it shares the sklearn API. | >=4.0 for stable sklearn integration |
| **Optuna** | Hyperparameter tuning | Better than GridSearch (Bayesian optimization), better than manual tuning. `optuna.integration.XGBoostPruningCallback` for early stopping during tuning. 50–100 trials is reasonable for this dataset size. | >=3.5 for improved pruning |
| **SHAP** | Per-prediction explanations | Non-negotiable for the human-in-the-loop deliverable. TreeExplainer is fast for XGBoost/LightGBM. Produces the "$12,400 because: Inconel (+$2,800)..." breakdowns. | >=0.44 — watch for numpy 2.0 compatibility |
| **rapidfuzz** | Fuzzy string matching | Handles Tier 1 PartDescription parsing with typo tolerance. Written in C++ with Python bindings — fast even on large registries. Better than `thefuzz` (pure Python, slower). | >=3.0 |
| **FastAPI** | API endpoints | Right weight for the prototype. Auto-generates OpenAPI docs. Pydantic integration for request/response validation. Async support if needed later. | >=0.110 |
| **SQLite** | Override storage | Built into Python (`sqlite3`). No server to run. Perfect for prototype-scale override tracking. The plan correctly uses this — no need for Postgres at 510 rows + a trickle of overrides. | stdlib |
| **matplotlib + seaborn** | Visualization | Standard for EDA plots, residual charts, bias heatmaps. The plan needs ~15 plot types — these two cover all of them. | matplotlib >=3.8, seaborn >=0.13 |
| **scipy.stats** | Statistical tests | Wilcoxon signed-rank for pairwise model comparison, bootstrap CIs for estimator bias. Mentioned in the plan but missing from the stack table. | >=1.12 |

### 3.2 Libraries to ADD (missing from the original plan)

| Library | Role | Why it's needed |
|---|---|---|
| **mapie** | Conformal prediction intervals | The plan mentions conformal prediction for prediction bands (Phase 4) but names no library. `mapie` wraps any sklearn-compatible regressor and produces distribution-free prediction intervals. `MapieRegressor` with `method="plus"` is the right choice for 510 rows. Alternative: manual quantile regression with XGBoost (`objective="reg:quantile"`), but mapie is cleaner and model-agnostic. |
| **pytest** | Testing | Every component needs coverage — see Section 5 below. |
| **pytest-cov** | Coverage reporting | Ensures test coverage is measurable from CLI: `pytest --cov=price_estimator`. |
| **httpx** | API testing | FastAPI's `TestClient` is built on httpx. Required for `test_api.py`. |
| **ruff** | Linting + formatting | Single tool replaces flake8 + black + isort. Zero-config, extremely fast (written in Rust). Enforces consistent style across `src/`, `scripts/`, `tests/`. |
| **pydantic** | Data validation | Already installed via FastAPI, but should be used explicitly for input validation on the `/quote` endpoint and for data contracts between pipeline stages. Validates that `Quantity` is in `{1,5,10,20,50,100}`, `Material` is one of 5 known values, etc. |
| **joblib** | Model serialization | Serialize trained models to `outputs/models/`. sklearn's recommended serialization. Used by `scripts/train.py` (save) and `api.py` (load). |
| **papermill** | CLI notebook execution | Runs `analysis.ipynb` from the command line with parameters, ensuring the notebook is reproducible without manual interaction: `papermill notebooks/analysis.ipynb outputs/analysis_executed.ipynb`. Critical for the full pipeline — the notebook is the primary deliverable assessors read, and it must be regenerable from a single command. Without papermill, the notebook is a manual step that can silently diverge from `outputs/`. |

### 3.3 Libraries to DROP (in the plan but unnecessary)

| Library | Why drop it |
|---|---|
| **spaCy** | Tier 2 tooling for a Tier 1 prototype. Installing a 50MB+ NLP framework with model downloads for token matching we won't build is wasted complexity. The 10 known part types and 9 modifiers are fully handled by `rapidfuzz` + a Python dict. If Tier 2 is ever built, spaCy can be added then. |
| **flashtext** | Same reasoning — Tier 2 tooling. Also, with only ~20 keywords in the vocabulary, even a naive `str.contains()` loop runs in microseconds. flashtext's O(n)-in-keywords advantage is irrelevant at this scale. |
| **thefuzz** | `rapidfuzz` is a strict superset with 10x better performance. No reason to install both. |
| **pyyaml** | The Tier 1 part description registry is a Python dict in `features.py`, not a YAML file. For 10 part types and 9 modifiers, a dict literal is simpler, testable, and eliminates a dependency. |

### 3.4 Libraries that are CONDITIONAL

| Library | When to add | When to skip |
|---|---|---|
| **anthropic** (Claude API SDK) | If you actually build Tier 3 LLM parsing | Prototype scope is Tier 1 only — skip for now |
| **polars** | Never for this project | 510 rows doesn't benefit from columnar processing |
| **mlflow** | If you want experiment tracking across runs | Overkill for a take-home; Optuna's built-in study storage + CSV output is sufficient |
| **great_expectations** | If data validation needs to be production-grade | For prototype, manual pandas assertions in `data.py` + pytest is sufficient |
| **pre-commit** | If multiple developers will contribute | Nice-to-have for solo work; ruff in CI is sufficient |

---

## 4. CLI Workflow

All modeling, experimentation, and validation is CLI-executable. Decisions are driven by numeric results from scripts. The notebook is built afterward as a presentation layer for assessors.

### 4.1 Setup

```bash
cd part1-price-estimator

# Create environment and install
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Generate requirements.txt for evaluators who don't use uv
uv pip compile pyproject.toml -o requirements.txt
```

### 4.2 Lint & Format

```bash
# Check
ruff check src/ tests/ scripts/
ruff format --check src/ tests/ scripts/

# Fix
ruff check --fix src/ tests/ scripts/
ruff format src/ tests/ scripts/
```

### 4.3 Test

```bash
# All tests
pytest

# With coverage
pytest --cov=price_estimator --cov-report=term-missing

# Specific test file
pytest tests/test_features.py -v

# Only fast tests (skip slow model training)
pytest -m "not slow"
```

### 4.4 EDA

```bash
python scripts/eda.py --data resources/aora_historical_quotes.csv --output outputs/
```

The script calls functions from `src/price_estimator/data.py` and `features.py` and writes:
- `outputs/figures/eda_*.png` — distribution plots, scatter matrices, bias charts, confounding heatmaps
- `outputs/results/data_profile.json` — summary stats, missing value report, confounding analysis

### 4.5 Train Models

```bash
# Train all models with 5-fold CV
python scripts/train.py --data resources/aora_historical_quotes.csv --output outputs/

# Train a specific model
python scripts/train.py --model M6 --data resources/aora_historical_quotes.csv

# Train with Optuna tuning (tree models only)
python scripts/train.py --model M6 --tune --n-trials 100
```

The script:
1. Loads and cleans data via `data.py`
2. Engineers features via `features.py`
3. Trains the specified model(s) via `models.py`
4. Saves trained models to `outputs/models/{model_name}.joblib`
5. Writes CV results to `outputs/results/cv_results.csv`

### 4.6 Evaluate & Compare

```bash
# Generate comparison table, significance tests, feature importance
python scripts/evaluate.py --results outputs/results/cv_results.csv --output outputs/

# Run specific comparisons
python scripts/evaluate.py --compare M1,M2  # additive vs multiplicative
python scripts/evaluate.py --compare M6,M6b # lead time contribution
```

Outputs:
- `outputs/results/model_comparison.csv` — MAPE, Median APE, RMSE, R² per model
- `outputs/results/significance_tests.csv` — Wilcoxon p-values for pairwise comparisons
- `outputs/figures/feature_importance.png` — permutation importance
- `outputs/figures/shap_summary.png` — SHAP beeswarm plot

### 4.7 Estimator Bias Analysis

```bash
python scripts/bias_analysis.py --data resources/aora_historical_quotes.csv --output outputs/
```

Outputs:
- `outputs/results/estimator_bias.json` — per-estimator residuals, CIs, safe/aggressive labels
- `outputs/figures/bias_by_part_type.png`
- `outputs/figures/bias_over_time.png`
- `outputs/figures/estimator_job_distribution.png` — confounding check

### 4.8 Serve API

```bash
# Start the API server
python scripts/serve.py --model outputs/models/M6.joblib --port 8000

# Test endpoints
curl -X POST http://localhost:8000/quote \
  -H "Content-Type: application/json" \
  -d '{"part_description": "Sensor Housing - threaded", "material": "Inconel 718", "process": "Wire EDM", "quantity": 10, "rush_job": false, "lead_time_weeks": 4}'
```

### 4.9 Generate Deliverable Notebook

```bash
# Execute the notebook non-interactively (imports from src/, reads from outputs/)
papermill notebooks/analysis.ipynb outputs/analysis_executed.ipynb

# Convert to HTML for sharing
jupyter nbconvert --to html outputs/analysis_executed.ipynb
```

This ensures the notebook is always regenerable from a single command and stays in sync with the pipeline outputs. Without this step, the notebook can silently diverge from the actual model results.

### 4.10 Full Pipeline

```bash
# One command: lint, test, train, evaluate, analyze, generate notebook
ruff check src/ tests/ scripts/ \
  && pytest --cov=price_estimator \
  && python scripts/eda.py --data resources/aora_historical_quotes.csv --output outputs/ \
  && python scripts/train.py --data resources/aora_historical_quotes.csv --output outputs/ \
  && python scripts/evaluate.py --results outputs/results/cv_results.csv --output outputs/ \
  && python scripts/bias_analysis.py --data resources/aora_historical_quotes.csv --output outputs/ \
  && papermill notebooks/analysis.ipynb outputs/analysis_executed.ipynb
```

The final papermill step regenerates the deliverable notebook from the pipeline outputs, ensuring the presentation artifact always reflects the latest results.

---

## 5. Validation & Testing Strategy

### 5.1 What to Test

The plan currently has **no tests**. This section defines what needs coverage and why.

#### Data Layer (`test_data.py`)

| Test | What it validates | Priority |
|---|---|---|
| `test_load_csv_schema` | CSV loads with expected columns, dtypes, row count | High |
| `test_missing_values_identified` | The 15 missing Material and 15 missing Process rows are detected | High |
| `test_missing_values_same_rows` | Check whether missing Material and Process are the same 15 rows (impacts imputation strategy) | High |
| `test_no_duplicate_quote_ids` | QuoteID is unique | Medium |
| `test_value_ranges` | Quantity in {1,5,10,20,50,100}, RushJob in {Yes,No}, LeadTimeWeeks in [2,12], TotalPrice > 0 | High |
| `test_date_range` | Dates within 2023-01 to 2024-03 | Low |
| `test_imputation_within_fold` | Imputation does not leak across CV folds | High |

#### Feature Engineering (`test_features.py`)

| Test | What it validates | Priority |
|---|---|---|
| `test_parse_all_known_descriptions` | All 10 part types parse correctly from the actual data | High |
| `test_parse_exact_match` | "Sensor Housing - threaded" → `base_type="Sensor Housing"`, `threaded=True` | High |
| `test_parse_fuzzy_match` | "Sensor Housng - threaded" → still matches at ≥90% similarity | High |
| `test_parse_no_modifier` | "Actuator Linkage" → `base_type="Actuator Linkage"`, all flags False | Medium |
| `test_parse_unknown_description` | "Unknown Widget - custom" → returns unknown/fallback, does not crash | High |
| `test_complexity_score_ordering` | "Manifold Block - complex internal channels" scores higher than "Mounting Bracket - standard" | Medium |
| `test_log_quantity_values` | log(1)=0, log(100)=4.6, no NaN/Inf | Medium |
| `test_feature_matrix_shape` | Output matrix has expected number of columns | Medium |
| `test_feature_matrix_no_nulls` | After imputation, no NaN in feature matrix | High |
| `test_one_hot_completeness` | All categorical levels are represented in encoding | Medium |

#### Models (`test_models.py`)

| Test | What it validates | Priority |
|---|---|---|
| `test_model_fits_without_error` | Each model (M1–M9) trains on sample data without crashing | High |
| `test_predictions_positive` | All predictions > 0 (prices can't be negative) | High |
| `test_log_target_backtransform` | For log-target models, exp(prediction) matches expected scale | High |
| `test_cv_reproducibility` | Same seed → same CV splits → same scores (within float tolerance) | High |
| `test_ridge_baseline_mape` | Ridge MAPE < 30% (sanity — a useless model would be ~50%+) | Medium |
| `test_xgboost_beats_mean` | XGBoost R² > 0 (better than predicting the mean) | Medium |
| `test_shap_values_sum` | SHAP values sum to (prediction - base_value) for each sample | Medium |
| `test_prediction_bands_ordered` | lower_bound < point_estimate < upper_bound for all predictions | High |
| `test_quantile_coverage` | 80% prediction interval covers ~80% of actuals (±5% tolerance) | Medium |

#### Bias Analysis (`test_bias.py`)

| Test | What it validates | Priority |
|---|---|---|
| `test_three_estimators_present` | Bias computed for all 3 estimators | High |
| `test_residuals_sum_near_zero` | Weighted mean of estimator residuals ≈ 0 (biases are relative) | Medium |
| `test_safe_aggressive_labels` | Highest-residual estimator labeled "safe", lowest labeled "aggressive" | High |
| `test_confidence_intervals_computed` | Bootstrap CIs are non-degenerate (lower < upper) | Medium |

#### API (`test_api.py`)

| Test | What it validates | Priority |
|---|---|---|
| `test_quote_valid_input` | POST /quote with valid data returns 200 + expected fields | High |
| `test_quote_missing_field` | POST /quote missing required field returns 422 | High |
| `test_quote_invalid_material` | POST /quote with unknown material returns 422 or handled gracefully | High |
| `test_quote_response_schema` | Response contains `estimate`, `model_range`, `prediction_interval`, `shap_explanation` | High |
| `test_override_stores_correctly` | POST /quote/{id}/override saves to SQLite, retrievable via GET | High |
| `test_override_requires_reason` | Override without `override_reason` returns 422 | Medium |
| `test_get_quote_with_override` | GET /quote/{id} returns both original estimate and human override | Medium |
| `test_override_delta_computed` | Override response includes correct `delta_from_model` | Medium |

### 5.2 Test Fixtures (`conftest.py`)

```python
# Shared fixtures used across test files
@pytest.fixture(scope="session")
def raw_data():
    """Load the actual CSV once for all tests."""
    return pd.read_csv("resources/aora_historical_quotes.csv")

@pytest.fixture
def sample_data():
    """Small synthetic dataset for fast tests."""
    return pd.DataFrame({
        "QuoteID": ["Q-TEST-1", "Q-TEST-2", "Q-TEST-3"],
        "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "PartDescription": [
            "Sensor Housing - threaded",
            "Manifold Block - complex internal channels",
            "Mounting Bracket - standard",
        ],
        "Material": ["Inconel 718", "Titanium Grade 5", "Aluminum 6061"],
        "Process": ["Wire EDM", "5-Axis Milling", "CNC Turning"],
        "Quantity": [1, 10, 50],
        "LeadTimeWeeks": [4, 8, 6],
        "RushJob": ["Yes", "No", "No"],
        "Estimator": ["Sato-san", "Tanaka-san", "Suzuki-san"],
        "TotalPrice_USD": [703.11, 5000.00, 2500.00],
    })

@pytest.fixture(scope="session")
def trained_model(raw_data):
    """Train a lightweight model once for prediction tests."""
    # Uses M2 (Ridge log-linear) for speed
    ...
```

### 5.3 Test Markers

```python
# In pyproject.toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests that train full models (deselect with '-m not slow')",
    "api: marks API endpoint tests",
    "integration: marks tests that use the real CSV data",
]
```

This allows targeted test runs:
```bash
pytest -m "not slow"          # Fast feedback loop (~5 seconds)
pytest -m api                 # API tests only
pytest -m integration         # Tests against real data
pytest                        # Everything (~60 seconds)
```

### 5.4 Validation Beyond Unit Tests

#### Data Validation (runtime, in `data.py`)

Assertions that run every time data is loaded — not just in tests:

```python
def load_and_validate(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Schema checks
    assert set(EXPECTED_COLUMNS).issubset(df.columns)
    assert df["TotalPrice_USD"].gt(0).all(), "Negative prices found"
    assert df["Quantity"].isin([1, 5, 10, 20, 50, 100]).all()
    assert df["RushJob"].isin(["Yes", "No"]).all()

    # Log warnings for data quality issues
    n_missing_material = df["Material"].isna().sum()
    if n_missing_material > 0:
        logger.warning(f"{n_missing_material} rows with missing Material")

    return df
```

#### Model Validation (in `models.py`)

Guardrails that run after every training:

```python
def validate_model(model, X_test, y_test):
    preds = model.predict(X_test)

    # Sanity checks
    assert (preds > 0).all(), "Model produced negative price predictions"
    assert np.isfinite(preds).all(), "Model produced NaN/Inf predictions"

    # Performance floor
    mape = mean_absolute_percentage_error(y_test, preds)
    assert mape < 0.50, f"Model MAPE {mape:.1%} exceeds 50% — likely broken"

    # Prediction range sanity
    assert preds.min() > 10, "Suspiciously low prediction (< $10)"
    assert preds.max() < 500_000, "Suspiciously high prediction (> $500K)"
```

#### Cross-Validation Integrity

The plan correctly specifies imputation within CV folds. This must be enforced structurally, not just documented:

```python
# WRONG: impute before splitting (data leakage)
df["Material"].fillna(df["Material"].mode()[0], inplace=True)
X_train, X_test = train_test_split(df)

# RIGHT: impute within pipeline (no leakage)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("model", Ridge()),
])
cross_val_score(pipeline, X, y, cv=5)  # imputation happens per fold
```

---

## 6. Cloud Resources (If Needed)

The prototype runs entirely locally. Cloud is optional for these scenarios:

| Scenario | Service | When |
|---|---|---|
| Tier 3 LLM parsing | Claude API | Only if building LLM fallback (not in prototype scope) |
| Experiment tracking | Weights & Biases (free tier) | If you want persistent run comparison across sessions |
| CI/CD | GitHub Actions | If you want automated test runs on push |
| Demo deployment | Railway / Render / Fly.io | If you want to deploy the FastAPI endpoint for a live demo |

For the prototype, none of these are necessary. The entire pipeline runs on a laptop in under 2 minutes.

---

## 7. Summary: Resolved Decisions

| Area | Decision | Status |
|---|---|---|
| Dependency management | `pyproject.toml` + `uv`, with `requirements.txt` for evaluators | Adopted in plan |
| Project structure | `src/` for logic, `scripts/` for CLI, `notebooks/` for presentation | Adopted in plan |
| Testing | pytest suite across data, features, models, bias, API | Adopted in plan |
| CLI-first workflow | All modeling/experimentation/validation via scripts; notebook is presentation only | Adopted in plan |
| Linting | ruff for lint + format | Adopted in plan |
| Prediction intervals | mapie for conformal prediction | Adopted in plan |
| Data validation | Runtime assertions in `data.py` + pytest | Adopted in plan |
| Model serialization | joblib save/load | Adopted in plan |
| Dropped deps | spaCy, flashtext, thefuzz, pyyaml | Removed |
| Notebook execution | papermill for CLI-driven notebook regeneration | Adopted in plan |
| PartDescription registry | Python dict in `features.py` (not YAML) | Adopted in plan |
| API testing | httpx + `test_api.py` | Adopted in plan |
