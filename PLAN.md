# Price Estimator — Consolidated Plan

## Problem Statement

A precision machine shop's lead estimator quotes aerospace parts "by feel." We have 510 historical quotes and need to build a prototype that:

1. Predicts `TotalPrice_USD` for new quotes
2. Extracts complexity features from part descriptions
3. Detects and accounts for estimator biases
4. Supports human overrides that feed back into the system
5. Identifies missing variables that would improve accuracy

---

## Phase 1: EDA & Data Profile

### Dataset

- **510 rows**, 10 columns
- **Target**: `TotalPrice_USD` ($98 — $115K, mean ~$9K)

| Column | Type | Cardinality | Missing |
|---|---|---|---|
| QuoteID | ID | 510 | 0 |
| Date | Date | ~15 months (2023-01 to 2024-03) | 0 |
| PartDescription | Text | 10 distinct types | 0 |
| Material | Categorical | 5 levels | 15 |
| Process | Categorical | 5 levels | 15 |
| Quantity | Discrete | {1, 5, 10, 20, 50, 100} | 0 |
| LeadTimeWeeks | Numeric | 2–12 | 0 |
| RushJob | Binary | Yes/No | 0 |
| Estimator | Categorical | 3 levels | 0 |

### Preliminary Findings (to be validated by model comparison, not baked in)

| Finding | Observation | Validation Method |
|---|---|---|
| Rush premium ~1.55x | Controlling for part/material/process | M2 (uniform log-space) vs M4 (rush × material/part interactions) |
| Volume discount is log-linear, slope -0.19 | 10x qty => ~63% unit price | M2 (global curve) vs M3 (per-part-type curves) vs M6 (learned freely) |
| Lead time has no price correlation | Flat ~$400 unit price across 2–12 weeks | M6 vs M6-no-leadtime + permutation importance |
| Estimator bias is additive | Tanaka $474, Suzuki $409, Sato $369 mean unit | M6 (estimator as feature) vs M8 (per-estimator models) |
| Material cost tiers are ordinal | Inconel > Ti > SS > Al7075 > Al6061 | One-hot vs ordinal encoding comparison |
| Pricing is multiplicative | Material × hours × qty discount × rush | M1 (raw target) vs M2 (log target) |

### Additional EDA to Perform

- Check estimator job distribution for confounding (does Tanaka get harder jobs, or does he just quote higher?)
- Per-part-type volume discount curves (do complex parts have steeper discounts from setup amortization?)
- Rush premium by part type and material (uniform or variable?)
- Missing value patterns (are the 15 missing Material and 15 missing Process the same rows? Random or systematic?)

### Confounding Analysis Framework

The bias analysis is only meaningful if we can distinguish "quotes higher" from "gets harder jobs." Perform the following checks during EDA:

1. **Cross-tabs with chi-squared tests.** For every pair of categorical features (Estimator × PartType, Estimator × Material, Estimator × Process, RushJob × Estimator), test for independence. A heatmap of p-values shows where assignments are non-random.
2. **Conditional vs marginal means.** Show both: "Tanaka's average is $474" (marginal) vs "Tanaka's average controlling for job type is $X" (conditional). If these diverge significantly, confounding is present.
3. **Propensity-style balance check.** Train a classifier to predict which estimator was assigned a job from job features. If accuracy > 40% (vs 33% random for 3 estimators), assignments are non-random.
4. **Stratified residual analysis.** After fitting M9, plot residuals by estimator × part type, estimator × material, estimator × quantity band. If Tanaka's bias is +12% overall but +25% on Inconel and -5% on Aluminum, that's domain-specific conservatism, not uniform bias.

**Key insight to surface in narrative:** If an estimator says "I quote Inconel high because the scrap rate is terrible," that's missing-variable knowledge embedded in their bias, not irrational behavior. The bias analysis should distinguish correctable bias from encoded domain expertise.

---

## Phase 2: Feature Engineering

### Direct Features

| Feature | Encoding | Notes |
|---|---|---|
| Material | One-hot (5 levels) | Also try ordinal cost tier (1-5) |
| Process | One-hot (5 levels) | Also try ordinal precision tier |
| Estimator | One-hot (3 levels) | Excluded in debiased model (M9) |
| RushJob | Binary (0/1) | |
| Quantity | log(Quantity) | Power-law relationship |
| LeadTimeWeeks | Numeric | Included in all models; validated by comparison |
| Date | Numeric (days since epoch) | Kitchen-sink only |

### Extracted from PartDescription

| Feature | Source Text | Type |
|---|---|---|
| base_part_type | "Sensor Housing", "Manifold Block", etc. | Categorical (10 levels) |
| thin_walls | "thin walls" | Binary |
| complex_internal | "complex internal channels" | Binary |
| high_precision | "high precision" | Binary |
| hardened | "hardened" | Binary |
| threaded | "threaded" | Binary |
| aerospace_grade | "aerospace grade" | Binary |
| emi_shielded | "EMI shielded" | Binary |
| high_fin_density | "high fin density" | Binary |
| standard | "standard" | Binary |
| complexity_score | Sum of weighted binary flags | Numeric |

### Interaction Features (Kitchen-Sink Model Only)

- Material × Process
- Material × PartType
- Rush × Material
- Rush × PartType
- material_cost_tier × log(Quantity)
- Rush × LeadTime

### PartDescription Parsing Architecture

For the prototype, implement Tier 1 only. Describe Tiers 2–3 in documentation.

```
Tier 1: DETERMINISTIC REGISTRY (prototype scope)
  - Dict/regex mapping of known patterns to structured features
  - Fuzzy matching via rapidfuzz for typo tolerance
  - Handles all 10 known part types with 100% reproducibility

Tier 2: RULE-BASED NLP (described, not built)
  - Token-level matching for known modifiers on unknown base types
  - Partial extraction with confidence scores
  - Flags low-confidence fields for human review

Tier 3: LLM PROPOSAL + HUMAN GATE (described, not built)
  - LLM proposes structured extraction for fully unknown descriptions
  - Human approves/edits; approved mapping written to Tier 1 registry
  - System grows deterministic coverage over time
```

### Missing Value Handling

- Impute within CV folds (never on full dataset before splitting)
- Try: mode imputation for categoricals, and a "missing" indicator feature
- Compare: imputation vs dropping rows vs treating missing as its own category
- **Decision criteria:** Pick the strategy with best CV MAPE; break ties toward simplicity (treating missing as its own category is simplest and preserves the "missingness" signal)
- **EDA first:** Check whether the 15 missing Material and 15 missing Process are the same rows or different. Document whether missingness is random or systematic (correlated with estimator, part type, etc.)
- **API behavior:** When a new quote arrives with missing Material or Process, return the best estimate with wider confidence bands and a warning — never reject the request

---

## Phase 3: Model Matrix

Every EDA finding becomes a testable hypothesis via model comparison.

| # | Model | Target | Rush | Volume | Lead Time | Estimator | PartDescription |
|---|---|---|---|---|---|---|---|
| M0 | Lookup table / formula | raw price | fixed 1.55x multiplier | qty^0.81 (from EDA slope) | excluded | excluded | lookup table per part type |
| M1 | Ridge (additive) | raw price | additive feature | linear qty | included | one-hot | one-hot base + flags |
| M2 | Ridge (log-linear) | log(price) | learned in log-space (multiplicative) | log(qty) | included | one-hot | one-hot base + flags |
| M3a | Two-stage unit price (global curve) | unit price -> total | separate multiplier | global log-log curve | excluded | one-hot | one-hot base + flags |
| M3b | Two-stage unit price (per-part curves) | unit price -> total | separate multiplier | per-part-type log-log curves | excluded | one-hot | one-hot base + flags |
| M4 | Kitchen-sink Lasso | log(price) | interaction terms (rush x material, rush x part) | log(qty) + interactions | included | one-hot + interactions | one-hot + all flags + interactions |
| M5 | Random Forest | raw price | learned freely | learned freely | included | label-encoded | label-encoded base + flags |
| M6 | XGBoost | raw price | learned freely | learned freely | included | label-encoded | label-encoded base + flags |
| M6b | XGBoost (no lead time) | raw price | learned freely | learned freely | **excluded** | label-encoded | label-encoded base + flags |
| M7 | XGBoost (log target) | log(price) | learned freely | learned freely | included | label-encoded | label-encoded base + flags |
| M7b | LightGBM | raw price | learned freely | learned freely | included | label-encoded | label-encoded base + flags |
| M7c | LightGBM (log target) | log(price) | learned freely | learned freely | included | label-encoded | label-encoded base + flags |
| M8 | Per-estimator XGBoost (x3) | raw price | learned freely | learned freely | included | N/A (separate models) | label-encoded base + flags |
| M9 | Debiased XGBoost | raw price | learned freely | learned freely | included | **excluded** (bias computed post-hoc) | label-encoded base + flags |

### M0: Lookup Table Baseline

A pure deterministic formula with no ML:

```
base_price = PART_TYPE_TABLE[part] × MATERIAL_TABLE[material] × PROCESS_TABLE[process]
total = base_price × qty^0.81 × (1.55 if rush else 1.0)
```

The lookup tables are median unit prices from the training data. This could run in a spreadsheet — no Python needed on the shop floor. The assessors explicitly value pragmatism: "boring automation that saves 40 hours is often more valuable than a complex AI model." If M0 gets within ~15% MAPE, it contextualizes the marginal lift from more complex models.

### What Each Comparison Tests

| Comparison | Hypothesis |
|---|---|
| M0 vs M1 | How much does ML add over a simple formula? |
| M1 vs M2 | Is pricing additive or multiplicative? |
| M2 vs M3a | Does explicit unit-price decomposition help? |
| M3a vs M3b | Do discount curves differ by part type? |
| M2 vs M4 | Do interaction terms and kitchen-sink features add lift? |
| M5 vs M6 | RF vs XGBoost stability check (if XGBoost >> RF, likely overfitting) |
| M6 vs M6b | Does lead time contribute signal? |
| M6 vs M7 | Raw vs log target for trees |
| M6 vs M7b | XGBoost vs LightGBM on small data (level-wise vs leaf-wise) |
| M6 vs M8 | Is estimator bias additive (M6 wins) or structural (M8 wins)? |
| M6 vs M9 | Does including estimator help or hurt generalization? |

### Hyperparameter Tuning

- **Linear models**: alpha via CV (RidgeCV, LassoCV)
- **Tree models**: Optuna, 50–100 trials per model
  - `max_depth`: 3–6 (shallow — small dataset)
  - `n_estimators`: 50–300 with early stopping
  - `learning_rate`: 0.01–0.1
  - `min_child_weight` / `min_data_in_leaf`: high values
  - `subsample`, `colsample_bytree`: 0.6–0.9

---

## Phase 4: Evaluation

### Cross-Validation

- **5-fold CV**, same seeded splits across all models
- For log-target models, metrics computed on **back-transformed predictions** (exp), not in log-space
- For M8 (per-estimator), track whether CV variance increases vs M6

### Jensen's Inequality Correction

When training on `log(price)` and back-transforming with `exp()`, the result is the geometric mean, not the arithmetic mean — causing systematic underestimation. For all log-target models (M2, M4, M7, M7c), apply the correction:

```
corrected_prediction = exp(log_pred + 0.5 * σ²)
```

where σ² is the residual variance in log-space. This correction must be applied before computing MAPE on back-transformed predictions to ensure fair comparison with raw-target models.

### Metrics

| Metric | Purpose |
|---|---|
| MAPE | Primary — most intuitive ("off by X%") |
| Median APE | Robust to outliers |
| RMSE | Penalizes big misses |
| R² | Overall explanatory power |

### Statistical Significance

- **Wilcoxon signed-rank test** on per-fold MAPE for pairwise model comparisons
- Avoids "M6 got 8.2% vs M7 at 8.5%" ambiguity

### Feature Importance

- **Permutation importance** on best tree model
- **SHAP values** for per-prediction explanations
- **Lasso coefficients** from M4 for feature selection signal
- **Ablation**: measure lift from PartDescription parsing (with vs without complexity flags)

### Prediction Bands (Four Lenses)

| Band | What It Shows | Method |
|---|---|---|
| Model uncertainty (CI) | How sure is the model about the mean? | Linear: analytical. Trees: bootstrap |
| Prediction interval | Range for this specific job | Linear: analytical. Trees: quantile regression (tau=0.1, 0.5, 0.9) or conformal prediction |
| Estimator spread | What would each human quote? | Per-estimator predictions from M8 or bias-adjusted M9 |
| Multi-model disagreement | Do our models agree? | Min/max/median across all models |

**Display format:**
```
Quote Estimate: $12,400
+-- Model range:     $11,200 - $13,800  (across models)
+-- 80% prediction:  $9,800  - $15,600  (historical variance)
+-- Estimator range: $10,900 (Sato) - $14,200 (Tanaka)
```

### Confidence Flag Triggers (Consolidated)

The API's `confidence_flag` is triggered by any of the following:

1. Multi-model disagreement > 20%
2. Any feature value outside the training range (e.g., qty=500, unseen material)
3. Missing input fields (Material or Process)
4. PartDescription resolved at Tier 2 or Tier 3 (not fully recognized)
5. Low-confidence bias estimate (< 10 data points for estimator × job type cell)

Feature-range checks are trivial to implement and high value. Nearest-neighbor distance in feature space is a production enhancement (described, not built in prototype).

---

## Phase 5: Estimator Bias Analysis (Task 3)

### Method

1. Train M9 (debiased XGBoost, no estimator feature) to get "neutral" price
2. Compute residuals: `actual - neutral_prediction` per estimator
3. Statistical tests: bootstrap CIs on mean residual per estimator
4. Drill down: bias by part type, by material, over time

### Deliverables

- Who is the "safe" quoter (positive residuals = quotes high, protects margin)
- Who is the "aggressive" quoter (negative residuals = quotes low, wins bids)
- Whether bias is uniform or varies by job type
- Whether bias has drifted over time
- Check for confounding: are estimators assigned different job mixes?

---

## Phase 6: Human-in-the-Loop API (Task 4)

### Endpoint Design (FastAPI)

```
POST /quote
  Input:  { part_description, material, process, quantity, rush_job, lead_time_weeks }
  Output: { estimate, model_range, prediction_interval, estimator_range,
            shap_explanation, confidence_flag }

POST /quote/{quote_id}/override
  Input:  { human_price, override_reason, estimator_id }
  Output: { stored: true, delta_from_model, override_id }

GET /quote/{quote_id}
  Output: { original_estimate, human_override (if any), final_price }
```

### Learning from Overrides

- Store overrides in SQLite: `(features, model_price, human_price, override_reason_category, override_reason_text, timestamp, estimator_id)`
- **Structured override reasons:** Provide a dropdown of common categories (material hardness, geometry complexity, surface finish, tooling difficulty, customer relationship, scrap risk, certification requirements) plus a free-text field. Structured categories give usable signal from day one — "40% of overrides cite surface finish" directly identifies a missing variable.
- **Minimum sample threshold:** Require 30–50 overrides before incorporating into retraining. Validate that holdout performance improves after incorporating overrides — don't blindly add them to training data.
- Periodic retrain incorporates overrides as additional training data (with a flag distinguishing original quotes from overrides)
- Track override rate and magnitude over time — decreasing overrides = model is improving

### SHAP Integration

Each prediction includes a SHAP breakdown: "This quote is $12,400 because: Inconel (+$2,800), 5-axis (+$1,600), qty=1 (+$1,400), rush (+$950), thin walls (+$650)." This gives the estimator a reason to agree or disagree — informed overrides are more valuable than blind corrections.

---

## Phase 7: Missing Variables (Task 5)

Variables that would improve accuracy if available:

| Category | Variables | Impact |
|---|---|---|
| Geometry (from 3D model) | Total volume, surface area, bounding box, thinnest wall, tightest radius, number of features/holes | Directly drives machining time |
| Tolerances | GD&T callouts, surface finish (Ra values), positional tolerances | Tighter = slower = more expensive |
| Material pricing | Real-time cost per kg, current availability | Material is ~30-50% of part cost |
| Process details | Number of setups/fixtures, estimated cycle time, tool count | Core cost drivers |
| Post-processing | Heat treatment, coating, plating, NDT (non-destructive testing) | Additional operations |
| Certification | AS9100, NADCAP, ITAR requirements | Compliance overhead: documentation, inspection, traceability |
| Historical shop data | Scrap/rework rates per part type, per process | Risk pricing |
| Supplier data | Tooling wear rates by material/process combo | Consumables cost |

---

## Project Structure

All modeling, experimentation, and validation is CLI-executable. Notebooks are for presentation and visualization only — decisions are driven by numeric results from scripts.

```
part1-price-estimator/
├── pyproject.toml                  # Dependency management (uv)
├── PLAN.md
├── TOOLING.md
├── resources/
│   └── aora_historical_quotes.csv
├── src/
│   └── price_estimator/
│       ├── __init__.py
│       ├── data.py                 # Loading, cleaning, validation, imputation
│       ├── features.py             # PartDescription parser (Tier 1), feature matrix
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
│   ├── train.py                    # CLI: python scripts/train.py → train all models, output CV results
│   ├── evaluate.py                 # CLI: python scripts/evaluate.py → comparison table, significance tests
│   ├── bias_analysis.py            # CLI: python scripts/bias_analysis.py → estimator profiles
│   └── serve.py                    # CLI: uvicorn wrapper for API
├── notebooks/
│   └── analysis.ipynb              # Presentation notebook (imports from src/, visualizes outputs/)
└── outputs/
    ├── models/                     # Serialized models (joblib)
    ├── figures/                    # EDA and result plots
    └── results/                    # CV tables, bias reports (CSV/JSON)
```

### Design Principles

- **`src/price_estimator/`**: All logic lives here as importable Python. Every function is testable, callable from scripts, notebooks, or the API.
- **`scripts/`**: CLI entry points that parse arguments and call into `src/`. Each script is a complete, self-contained workflow step. The full pipeline is executable as a single shell command chain.
- **`notebooks/`**: The deliverable notebook imports from `src/` and focuses on narrative, plots, and conclusions. It contains no business logic — if the notebook disappears, the system still works.
- **`tests/`**: pytest suite covering data validation, feature parsing, model sanity, bias computation, and API contracts.
- **`outputs/`**: All artifacts (models, figures, result tables) written here. Gitignored except for final results.

### CLI Workflow

```bash
# Setup
uv venv && source .venv/bin/activate && uv pip install -e ".[dev]"

# Lint
ruff check src/ tests/ scripts/ && ruff format --check src/ tests/ scripts/

# Test
pytest --cov=price_estimator

# Full pipeline
python scripts/eda.py \
  && python scripts/train.py \
  && python scripts/evaluate.py \
  && python scripts/bias_analysis.py

# Serve API
python scripts/serve.py --model outputs/models/M6.joblib --port 8000
```

---

## Implementation Stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | Data science ecosystem |
| Dependency mgmt | uv + pyproject.toml | Fast, lockfile support; also generates requirements.txt for evaluators |
| Data | pandas | 510 rows, nothing heavier needed |
| ML | scikit-learn (Ridge, Lasso, RF), XGBoost, LightGBM | Best tabular ML libraries |
| Tuning | Optuna | Bayesian hyperparameter optimization |
| Explainability | SHAP | Per-prediction feature attributions |
| Prediction intervals | mapie | Conformal prediction, model-agnostic |
| Parsing | Python dict + rapidfuzz | Tier 1 PartDescription parsing (no YAML — dict literal is simpler) |
| Evaluation | scikit-learn metrics + scipy.stats (Wilcoxon) | Robust comparison |
| Visualization | matplotlib / seaborn | EDA, residual plots, bias charts |
| API | FastAPI + pydantic | Human-in-the-loop endpoint with input validation |
| Storage | SQLite (stdlib) | Override/correction storage |
| Serialization | joblib | Save/load trained models |
| Testing | pytest + pytest-cov + httpx | Unit/integration tests, API test client |
| Linting | ruff | Single tool for lint + format |
| Notebook | Jupyter (presentation only) | Narrative deliverable; imports from src/ |

---

## Risk Matrix

| Risk | Severity | Likelihood | Mitigation |
|---|---|---|---|
| Overfitting (510 rows, many features) | High | High | Aggressive regularization, nested CV, shallow trees, RF stability benchmark |
| Missing values create data leakage | Medium | Medium | Impute within CV folds, never on full dataset first |
| New part types at inference time | High | Certain (in production) | Three-tier parser architecture (Tier 1 built, 2-3 documented) |
| Extrapolation (qty=500, new material) | High | Medium | Prediction intervals; flag when input is outside training distribution |
| Estimator bias confounded with job difficulty | Medium | Medium | Check job assignment distributions before drawing conclusions |
| Stale model as prices/materials change | Medium | High over time | Scheduled retrain, drift monitoring on prediction residuals |
| Human overrides corrupt training data | Medium | Low | Track overrides separately; require reason field; review before retrain |

---

## Final Model Selection Criteria

After running the full model matrix, select the production model based on:

1. **Primary:** Best MAPE (most intuitive for pricing)
2. **Secondary:** Low fold-to-fold variance (stability — a model that's 9% MAPE with low variance beats 8.5% with high variance)
3. **Tertiary:** Explainability (if top models are within ~1% MAPE of each other, prefer the simpler/more interpretable one)

If several models are within ~1% MAPE, consider an ensemble (weighted average of top 2-3). The notebook conclusions section must explicitly name the "ship it" model(s) and justify the choice. Also report M0's MAPE prominently to contextualize the ML lift.

---

## Deliverable Structure

### CLI Scripts (where decisions are made)

| Script | What it produces | Artifacts |
|---|---|---|
| `scripts/eda.py` | Summary stats, missing value report, confounding checks | `outputs/results/data_profile.json`, `outputs/figures/eda_*.png` |
| `scripts/train.py` | Trains M0–M9, runs 5-fold CV, tunes hyperparameters | `outputs/models/*.joblib`, `outputs/results/cv_results.csv` |
| `scripts/evaluate.py` | Comparison table, significance tests, feature importance | `outputs/results/model_comparison.csv`, `outputs/results/significance_tests.csv`, `outputs/figures/shap_*.png` |
| `scripts/bias_analysis.py` | Estimator profiles, confounding assessment | `outputs/results/estimator_bias.json`, `outputs/figures/bias_*.png` |
| `scripts/serve.py` | Starts FastAPI server | Runtime only |

### Notebook (presentation deliverable)

The notebook imports from `src/` and loads pre-computed results from `outputs/`. It contains no training or evaluation logic — only visualization and narrative. But it is the **primary artifact the assessors will read**, so it must be rich, well-narrated, and tell a complete story.

```
1. Introduction & Problem Context
   - Problem statement: 30 years of "feel"-based quoting, 510 historical quotes
   - What we're building and why
   - Key design tenets (pragmatism, explainability, human-in-the-loop)

2. EDA
   - Univariate distributions (price, quantity, lead time histograms)
   - Unit price analysis by material, part type, process, estimator (bar charts with CIs)
   - Volume discount curves: log(qty) vs log(unit_price) scatter + fitted line
   - Rush premium analysis: controlled comparison with effect size
   - Lead time analysis: show the flat relationship, explain why we still test it
   - Missing value patterns: overlap check, missingness mechanism, distribution by estimator
   - Confounding analysis:
     - Cross-tab heatmaps (Estimator × PartType, Estimator × Material)
     - Chi-squared p-value summary
     - Conditional vs marginal mean comparison table
     - Propensity classifier accuracy

3. Feature Engineering
   - PartDescription parsing walkthrough: show input → output for all 10 types
   - Complexity flag extraction with examples
   - Tier 1 registry explanation + description of Tiers 2–3 for production
   - Feature matrix summary: final shape, column listing, correlation heatmap
   - Missing value strategy comparison (mode vs category vs drop — CV results)

4. Model Comparison
   - Full comparison table (MAPE, Median APE, RMSE, R² for M0–M9 + variants)
   - M0 prominently highlighted: "a spreadsheet formula achieves X% MAPE"
   - Bar chart of MAPE across all models with fold-to-fold error bars
   - Pairwise significance tests (Wilcoxon p-values, formatted as matrix)
   - Hypothesis resolution table:
     | Hypothesis | Comparison | Winner | Finding |
     (e.g., "Pricing is multiplicative" → M1 vs M2 → M2 wins → confirmed)
   - Feature importance: permutation importance bar chart + SHAP beeswarm plot
   - Ablation: MAPE with vs without complexity flags

5. Estimator Bias Analysis (Task 3)
   - Debiased model (M9) residuals by estimator: violin plots
   - Per-estimator summary: mean bias, CI, "safe"/"aggressive" labels
   - Stratified breakdown: bias heatmap by estimator × part type
   - Bias over time: rolling residual plot per estimator
   - Confounding assessment: narrative on whether bias = pricing style or job mix
   - Key insight: where bias encodes domain expertise (scrap risk, etc.)

6. Prediction Bands (Four Lenses)
   - Demo with 3–5 example quotes showing all four band types
   - Quantile regression calibration: does 80% interval cover ~80% of actuals?
   - Multi-model disagreement distribution: histogram of spread across models
   - Estimator spread visualization: Sato vs Suzuki vs Tanaka for same inputs
   - Display format mockup:
     Quote Estimate: $12,400
     +-- Model range:     $11,200 - $13,800
     +-- 80% prediction:  $9,800  - $15,600
     +-- Estimator range: $10,900 (Sato) - $14,200 (Tanaka)

7. API Design & Human-in-the-Loop (Task 4)
   - Endpoint spec with example request/response JSON
   - Override workflow diagram
   - Structured override reason categories
   - SHAP explanation demo: show breakdown for a specific prediction
   - Override feedback loop: how corrections feed back into retraining
   - OOD detection: confidence flag triggers with examples

8. Missing Variables Discussion (Task 5)
   - Table of variables that would improve accuracy (geometry, tolerances, etc.)
   - Mapping from override reason patterns to missing variables
   - What a 3D model / STEP file would unlock (wall thickness, feature count, etc.)
   - Prioritized roadmap: which variables to ingest first based on expected impact

9. Production Considerations
   - Parser monitoring (tier escalation rates, registry validation on new entries)
   - Model staleness / drift monitoring (residual tracking over time)
   - Override retraining thresholds and validation
   - New estimator onboarding (cold-start strategy)
   - Nearest-neighbor OOD detection (described, not built)
   - Assignment drift monitoring

10. Conclusions & Recommendations
    - Final model selection with justification (MAPE, stability, explainability)
    - M0 pragmatism assessment: spreadsheet vs ML tradeoff for a real shop
    - Recommended deployment path
    - What to build next
```
