# Implementation Review — Aspects to Analyze

## Purpose

Systematic review of the current implementation against our tenets, plan, and engineering standards. Organized by domain so reviewers can focus on their area of concern.

---

## 1. Modelling & Scientific Rigor

Correctness of the mathematical and statistical foundations.

### 1.1 Target Transform & Back-Transform

- **Jensen's correction**: Is `exp(log_pred + 0.5 * sigma^2)` applied consistently across all log-target models (M2, M4, M7, M7c)? Is sigma^2 computed from training residuals (correct) or full-dataset residuals (leakage)?
- **Residual variance scope**: sigma^2 is fitted on training data during `fit()`. During CV, this means each fold gets its own sigma^2 — is this intentional and correct?
- **Log of target safety**: `np.log(y)` assumes all prices are positive. Training data min is $98, but should there be an explicit guard?

### 1.2 Model-Specific Logic

- **M0 lookup table formula**: `unit_price * qty^(1+slope) * rush` — verify the algebraic derivation from unit price to total price with discount curve.
- **M3 two-stage round-trip**: Normalizes to qty=1 equivalent by subtracting `slope * log_qty`, then reconstructs with `(1+slope) * log_qty`. Verify the algebra preserves the relationship.
- **M4 interaction features**: Are interaction columns built identically in `fit()` and `predict()`? Column ordering and naming must match exactly.
- **M8 per-estimator CV**: During cross-validation, each fold splits the full dataset — then M8 internally splits by estimator. Could a test fold contain an estimator absent from the training fold?

### 1.3 Cross-Validation Methodology

- **KFold consistency**: All models use the same `KFold(n_splits=5, shuffle=True, random_state=42)`. Verify M8's effective test sets are identical to other models'.
- **Stratification**: Plain KFold with 510 rows and skewed price distribution. Should we stratify by price quartile or material to reduce fold variance?
- **Nested CV in M4**: LassoCV uses internal cv=5 for alpha selection inside our outer 5-fold CV. Is this properly nested (no leakage) or does it share folds?
- **Statistical power**: Wilcoxon on 5 fold-level MAPEs — we already saw 0 significant differences. Should we acknowledge this limitation more prominently?

### 1.4 Numerical Stability

- **MAPE computation**: `|y_true - y_pred| / |y_true|` — any near-zero actuals that could inflate MAPE?
- **Bias analysis denominator**: After the +inf% fix, we use actual prices as denominator. Confirm no edge cases remain.
- **Bootstrap CIs**: 5000 resamples on ~170 per-estimator samples — verify stability of the CI bounds.

---

## 2. Data Engineering — REVIEW COMPLETE

Correctness of data handling, feature construction, and pipeline integrity.

**Review date:** 2026-04-06

### 2.1 Data Leakage

#### 2.1.1 Imputation timing — PASS

All models impute within `fit()`/`predict()`, never before splitting:
- Linear models (M1, M2, M3, M4): `X.fillna(0)` inside `fit()` and `predict()`
- Tree models (M5, M6, M7, M8, M9): `X.fillna(-1)` inside `fit()` and `predict()`
- `build_feature_matrix()` returns NaN for missing values — does NOT impute
- `load_data()` converts empty strings to NaN — does NOT impute

No leakage from pre-split imputation.

#### 2.1.2 Feature encoding statefulness — PASS (label) / FINDING F1 (one-hot)

**Label encoding (tree models):** Correctly uses `LABEL_ENCODING_CATEGORIES` dict (`features.py:63-68`) with explicit fixed category lists. `pd.Categorical(series, categories=cats).codes` produces stable integer codes regardless of fold contents. Robust.

**One-hot encoding (linear models):** `pd.get_dummies()` is called fresh per fold (`features.py:309`). Columns depend on observed categories. See **Finding F1** below.

#### 2.1.3 M0 median computation — PASS

`M0LookupTable.fit()` calls `extract_description_features()` and `compute_unit_price()` only on the training DataFrame passed to it. The parser is deterministic (dict registry + rapidfuzz), no global state. No information smuggling.

#### 2.1.4 M4 date feature — OBSERVATION (not a bug)

`days_since_epoch` = `(df["Date"] - 2023-01-01).days`. Available at inference time (quote date is known). However, with 15 months of data and 510 rows, a date trend could capture noise. Lasso regularization should zero it if irrelevant. Not leakage, but an overfitting risk.

#### 2.1.5 Bias analysis: train-on-full-data residuals — FINDING F5

`compute_estimator_bias()` (`bias.py:37-39`) trains M9 on the **full dataset** then computes residuals on the **same data**. This is in-sample, not cross-validated. Consequences:
- Underestimates true prediction error (optimistic neutral predictions)
- May attribute model fitting noise to estimator bias
- Inflates confidence in bias labels ("safe"/"aggressive")

Should use OOF (out-of-fold) predictions for honest bias analysis.

**Severity:** MEDIUM — affects bias analysis interpretation (Task 3), not model selection.

---

### 2.2 Missing Value Handling

#### FINDING F2: One-hot encoding silently drops NaN; ordinal tiers default to 0.0

**Severity:** MEDIUM

**Location:** `features.py:309`, `features.py:291-292`, `models.py:311,318,458,475`

**Traced paths for linear models (M1, M2, M3, M4):**

```
NaN Material
  → pd.get_dummies(series)     → all-zero row (NaN silently dropped, no indicator)
  → fillna(0)                  → no-op (already int 0s)
  Result: indistinguishable from an unseen category

NaN Material
  → MATERIAL_COST_TIER.map()   → NaN
  → .astype(float)             → np.nan
  → fillna(0)                  → 0.0
  Result: "zero cost tier" — LOWER than Aluminum 6061 (tier 1)

NaN Process
  → PROCESS_PRECISION_TIER.map() → NaN → fillna(0) → 0.0
  Result: "zero precision tier" — LOWER than Surface Grinding (tier 1)
```

**Traced paths for tree models (M5, M6, M7, M8, M9):**

```
NaN Material
  → pd.Categorical(...).codes  → -1 (NaN code)
  → .astype(float)             → -1.0
  → fillna(-1)                 → no-op (-1.0 is not NaN)
  Result: distinct sentinel, tree can learn to split on it ✓

NaN Material
  → MATERIAL_COST_TIER.map()   → NaN
  → .astype(float)             → np.nan
  → fillna(-1)                 → -1.0
  Result: distinct sentinel ✓
```

**Problems with linear model path:**
1. `pd.get_dummies()` silently drops NaN — no `material_nan` indicator column
2. All-zero encoding is lossy: missing material = unknown material = absent category
3. Ordinal tiers get 0.0 for missing → implies cheaper/simpler than any known category
4. The plan mentions comparing strategies (mode vs category vs drop) — this was **not implemented**

**Impact:** 15/510 rows (2.9%) affected. Small MAPE impact but introduces systematic bias toward underpricing missing-material rows.

**Recommended fix:** Add `missing_material` and `missing_process` binary indicators. Impute ordinal tiers with median (3.0) rather than 0.

#### 2.2.2 Overlap analysis — NOT VERIFIED IN PIPELINE

`get_missing_report()` in `data.py` computes overlap between missing Material and Process rows. The result is diagnostic only — not used to inform feature engineering. The EDA script should surface this; whether the 15+15 are the same rows or different rows affects how many total rows lack information.

---

### 2.3 Feature Matrix Integrity

#### FINDING F1: One-hot column misalignment during CV

**Severity:** LOW-MEDIUM (mitigated by reindex, but lossy)

**Location:** `features.py:308-310`, linear models' `predict()` methods

**Mechanism:** `pd.get_dummies()` produces columns only for categories present. With 510 rows and 5 folds (~102 per fold):

| Category | Min count | Could vanish from a fold? |
|---|---|---|
| Material (5 levels) | ~87 rows | Very unlikely |
| Process (5 levels) | ~87 rows | Very unlikely |
| Base part type (10 levels) | 45 rows (Mounting Bracket) | Unlikely but possible |
| Estimator (3 levels) | ~160 rows | Impossible |

**When columns mismatch:**
- Training has column, test doesn't → `reindex(fill_value=0)` adds zero column → **correct**
- Test has column, training doesn't → `reindex()` drops it → **lossy** (model has no coefficient for it)

**In practice:** With seed=42 and these cardinalities, highly unlikely. But the mechanism is fragile — a different seed or smaller dataset would expose it. Label encoding with `LABEL_ENCODING_CATEGORIES` avoids this by design.

**Recommended fix:** Define fixed one-hot column lists (like label encoding does). Or accept risk given current cardinalities and document the assumption.

#### 2.3.2 Label encoding stability — PASS

`LABEL_ENCODING_CATEGORIES` (`features.py:63-68`) fixes category order. `pd.Categorical(series, categories=cats).codes` is stable across folds. Unseen categories get code `-1`. Correct and robust.

#### 2.3.3 One-hot vs label parity — PASS

Both encodings include: ordinal tiers, complexity flags, complexity_score, log_quantity, rush_job, lead_time_weeks, and all four categoricals (base_part_type, material, process, estimator). Verified in `build_feature_matrix()` lines 285-321.

---

### 2.4 M8 Per-Estimator Model Integrity

#### FINDING F3: M8 predict() silently returns zeros for unseen estimators

**Severity:** HIGH (during CV), LOW (in production)

**Location:** `models.py:791-797`

```python
def predict(self, df: pd.DataFrame) -> np.ndarray:
    preds = np.zeros(len(df))
    for estimator, model in self.models.items():
        mask = df["Estimator"] == estimator
        if mask.sum() > 0:
            preds[mask.values] = model.predict(df[mask])
    return preds
```

If a test fold contains an estimator absent from the training fold's `self.models`, those rows get prediction = **0.0**. No error, no warning, no fallback. The MAPE for those rows would be ~100%.

**Probability during CV:** With ~170 per estimator and ~102 per test fold, extremely unlikely with seed=42. But no defensive guard exists.

**Each sub-model trains on ~136 rows** (170 - 34 removed per fold). XGBoost with 200 estimators on 136 rows is aggressive — high overfitting risk. This is partly mitigated by `max_depth=4` and `min_child_weight=5`.

**Recommended fix:** Add guard in `predict()`:
```python
missing = set(df["Estimator"].unique()) - set(self.models.keys())
if missing:
    logger.warning("Estimators %s not in training, defaulting to zeros", missing)
```

---

### 2.5 Pipeline Reproducibility

#### 2.5.1 Random seeds — PASS

All randomness sources are seeded:
- `CV_SEED = 42` for KFold, RF, XGBoost, LightGBM, comparison stochastic probes
- `np.random.default_rng(42)` for bootstrap CIs in `bias.py`
- `np.random.RandomState(CV_SEED)` for stochastic probes in `comparison.py`

Assessors re-running the pipeline will get identical results.

#### 2.5.2 Non-deterministic operations — PASS (with caveat)

- `pd.get_dummies()` column order depends on data order, but `reindex()` normalizes
- `dict.items()` is insertion-ordered in Python 3.7+
- `df["Estimator"].unique()` returns in first-seen order — stable

Caveat: `set()` operations in `validate()` produce non-deterministic warning message ordering. Cosmetic only.

#### 2.5.3 Dependency pinning — OBSERVATION

`pyproject.toml` uses range specifiers (`pandas>=2.0,<3`, `xgboost>=2.0,<3`). An assessor's resolved versions may differ from ours. joblib-serialized models may not load across library version boundaries.

**Check:** Does a `uv.lock` file exist to pin exact versions?

#### 2.5.4 requirements.txt — PASS

`requirements.txt` exists, generated by `uv pip compile`, with exact pinned versions (e.g., `xgboost==2.1.4`, `scikit-learn==1.8.0`, `numpy==1.26.4`). Assessors without uv can `pip install -r requirements.txt`.

No `uv.lock` file exists — `requirements.txt` serves the same purpose for reproducibility.

---

### 2.6 Findings Summary

| ID | Finding | Severity | Affected | Fix Effort |
|---|---|---|---|---|
| **F1** | One-hot columns misalign across CV folds (lossy reindex) | LOW-MEDIUM | M1, M2, M3, M4 | Medium |
| **F2** | NaN silently dropped in one-hot; ordinal tiers get 0.0 for missing | MEDIUM | M1, M2, M3, M4 | Low |
| **F3** | M8 predict() returns 0 for unseen estimators (no guard) | HIGH (CV) | M8 | Low |
| **F5** | Bias analysis uses in-sample residuals (not OOF) | MEDIUM | Bias (Task 3) | Medium |

**Items that passed:** Imputation timing (no pre-split leakage), label encoding stability (fixed categories), M0 median computation (no smuggling), random seeds (all seeded), requirements.txt (exists with pinned versions).

**Recommended fix priority:** F3 > F2 > F5 > F1

---

## 3. Product Goals & Tenet Alignment

Does the implementation serve the assessment deliverables and our stated principles?

### 3.1 Deliverable Completeness

- **5 assessment tasks**: (1) price prediction, (2) feature extraction, (3) estimator bias, (4) human-in-the-loop API, (5) missing variables. Which are fully implemented vs described?
- **Notebook**: The primary artifact assessors will read — not yet created. Does the 10-section outline in PLAN.md still match the implemented capabilities?
- **M0 prominence**: Tenet says highlight M0 to contextualize ML lift. Is M0's result surfaced prominently in both evaluate.py and compare.py outputs?

### 3.2 Tenet Adherence

- **CLI-first**: Can the full pipeline run as a single shell command chain? Are there any manual steps or notebook-dependent operations?
- **Multiple competing models**: All 14 models implemented and comparable. But are the hypothesis comparisons (M1 vs M2, M6 vs M6b, etc.) explicitly surfaced in the output, or does the assessor have to derive them from a table?
- **Preliminary findings as hypotheses**: The evaluate.py `resolve_hypotheses()` function maps comparisons to findings. Verify it covers all hypotheses from PLAN.md.
- **Pragmatism**: Would an assessor see the spreadsheet-vs-ML tradeoff clearly? Is the "boring automation" value proposition communicated?

### 3.3 API Design

- **Estimator defaulting**: API defaults missing estimator to "Sato-san" in `_request_to_dataframe()`. This silently biases predictions for models that use estimator as a feature. Should it be documented or handled differently?
- **Structured override reasons**: Implemented as an enum. Are the categories well-chosen for this domain? Would an actual estimator find them natural to use?
- **SHAP explanation usefulness**: Top-10 features by SHAP value — but feature names like `material_cost_tier` or `base_part_type` (as an integer code) aren't human-readable. Is post-processing needed?

### 3.4 Missing Deliverables

- **Notebook (analysis.ipynb)**: Not yet created. This is the most important deliverable for assessors.
- **TOOLING.md**: Interrupted before completion. Does it reflect actual dependency state?
- **Optuna tuning**: Mentioned in plan, not implemented. Are the default hyperparameters defensible without tuning?
- **Prediction bands**: mapie/conformal prediction not integrated. Only model disagreement is available as uncertainty signal.

---

## 4. Validation & Automation

Test coverage, CI readiness, and validation framework integrity.

### 4.1 Test Coverage Gaps

- **Tested modules**: data.py (29), features.py (43), comparison.py (44) = 116 tests
- **Untested modules**: models.py, bias.py, predict.py, api.py have no dedicated test files
- **Critical untested paths**: Model `fit()`/`predict()` correctness, bias computation logic, API endpoint contracts, override storage integrity
- **Integration tests**: No test runs the full pipeline (load → feature → train → predict → evaluate). A broken intermediate step would only be caught manually.

### 4.2 Boundary & Domain Validation

- **Domain invariants tested for**: M0 and M2 (in test_comparison.py). Not tested for tree models (M5, M6, M7), which are more likely to violate economic coherence on edge cases.
- **Boundary safety**: Price floor/ceiling tested for M0 and M2. Tree models could extrapolate flat or produce unexpected values — untested.
- **Missing feature handling**: Tested that predictions don't crash with NaN material. Not tested that the degraded prediction is *reasonable* (within 3x of with-material prediction, per VALIDATION_PLAN.md).

### 4.3 Comparison Framework

- **Stochastic probe reproducibility**: Are the 50 random job configurations seeded? Will an assessor running `compare.py` get the same scorecard?
- **Economic coherence probe coverage**: Deterministic probes use fixed combos — do they cover enough of the 10 × 5 × 5 = 250 possible configurations?
- **Scorecard stability**: Quantile ranks with 14 models — ranks could shift if one model is added/removed. Is this sensitivity acceptable?
- **End-to-end smoke test**: `compare.py` has not been run against real trained models. The 44 unit tests pass, but full CLI execution is unverified.

### 4.4 Automation & CI Readiness

- **Single-command pipeline**: `eda.py && train.py && evaluate.py && bias_analysis.py && compare.py` — does this chain work without manual intervention?
- **Lint compliance**: All files pass `ruff check` and `ruff format --check`?
- **Test execution time**: Are `slow`-marked tests actually slow? Is the default `pytest` run fast enough for CI?
- **Artifact consistency**: If the pipeline is re-run, do outputs/ artifacts get cleanly overwritten or accumulate stale files?

---

## 5. Documentation & Deliverable Quality

Accuracy, completeness, and consistency of documentation.

### 5.1 Plan-to-Implementation Drift

- **PLAN.md**: Does the plan still accurately describe what's implemented? Check model matrix, feature list, CLI commands, project structure.
- **CLAUDE.md**: Does the "Commands" section match actual CLI interfaces? Are the architecture descriptions current?
- **COMPARISON.md / VALIDATION_PLAN.md**: Do these documents match the implementation in comparison.py and test_comparison.py?

### 5.2 Code Documentation

- **Docstrings**: All public functions in src/ have docstrings. Are they accurate after bug fixes (e.g., bias.py denominator change)?
- **Type hints**: Are return types and parameter types consistent and correct?
- **Inline comments**: Present where logic is non-obvious (Jensen's correction, M3 normalization). Any missing explanations for tricky code?

### 5.3 Assessor Experience

- **First-run experience**: Can an assessor clone the repo, run `uv venv && uv pip install -e ".[dev]"`, and immediately execute the pipeline?
- **Error messages**: If a step fails (missing data file, untrained models), are the error messages helpful?
- **Output readability**: Are CLI outputs (train.py, evaluate.py, compare.py) clear enough to follow without reading the code?
- **requirements.txt**: Generated for evaluators who don't use uv? Is it current?

---

## Review Order

1. **Modelling & Scientific Rigor** — highest consequence if wrong
2. **Data Engineering** — hardest to catch later
3. **Product Goals & Tenet Alignment** — ensures we're building the right thing
4. **Validation & Automation** — ensures we can verify what we built
5. **Documentation & Deliverable Quality** — final polish before submission
