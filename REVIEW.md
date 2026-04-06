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

**Review status: COMPLETED**

Does the implementation serve the assessment deliverables and our stated principles?

### 3.1 Deliverable Completeness

| # | Deliverable | Status | Notes |
|---|---|---|---|
| 1 | Price prediction model | **Implemented** | 14 models trained, CV'd, serialized. M2 wins at 10.8% MAPE. |
| 2 | Feature extraction from PartDescription | **Implemented** | Tier 1 parser with 10 types, 9 modifiers, fuzzy matching. Tiers 2-3 described in PLAN.md. |
| 3 | Estimator bias analysis | **Implemented** | M9 debiased residuals, bootstrap CIs, per-part-type and over-time breakdowns. |
| 4 | Human-in-the-loop API | **Implemented** | 3 endpoints, SQLite storage, structured override reasons, SHAP explanation, OOD detection. |
| 5 | Missing variables discussion | **Described only** | Table in PLAN.md §7. Not surfaced as a standalone artifact — needs notebook section or separate document. |

#### Finding P1: Notebook not created — CRITICAL

The notebook (`notebooks/analysis.ipynb`) is the **primary artifact assessors will read**. It's empty. The 10-section outline in PLAN.md is comprehensive, but the notebook directory contains nothing. This is the single highest-priority gap.

**Impact:** Assessors may not run CLI scripts. Without the notebook, the story isn't told.

#### Finding P2: Missing variables not a standalone artifact

Task 5 (missing variable identification) exists only as a table in PLAN.md §7. There's no script output, no JSON, and no dedicated section visible to someone who doesn't read PLAN.md. The notebook would surface this, but without the notebook, this deliverable is effectively invisible.

**Recommended action:** Ensure the notebook's Section 8 surfaces this prominently. Consider also emitting `outputs/results/missing_variables.json` from a script for completeness.

### 3.2 Tenet Adherence

#### Finding P3: Hypothesis resolution output is stale — MEDIUM

The saved `hypothesis_resolution.csv` contains `M6 vs M7b` ("XGBoost vs LightGBM on small data"), but `resolve_hypotheses()` in evaluate.py was updated to `M7 vs M7c` ("XGBoost vs LightGBM, log-target, fair comparison"). The CSV on disk was generated before the code change.

The old comparison (M6 vs M7b, both raw-target) is uninformative because raw-target trees perform poorly (~38-47% MAPE) on this multiplicative data. The updated comparison (M7 vs M7c, both log-target) is the fair test. **The code is correct; the saved artifact is stale.**

**Fix:** Re-run `scripts/evaluate.py` to regenerate with the corrected hypothesis.

#### Finding P4: M0 not called out prominently in stdout — LOW

M0 appears in the sorted comparison table, but no script prints a specific call-out like "A spreadsheet formula achieves 40.3% MAPE — ML (M2) reduces this to 10.8%, a 4x improvement." The pragmatism narrative is left to the notebook to tell.

train.py and evaluate.py treat M0 as just another row in the table. Per our tenet, M0's result should be highlighted as a contextualizing anchor.

**Recommended action:** Add a 1-2 line call-out in evaluate.py stdout after the comparison table: "M0 (lookup table): X% MAPE — ML best (M2): Y% MAPE — Z% relative improvement."

#### Finding P5: CLAUDE.md pipeline missing compare.py — LOW

The "Full pipeline" command in CLAUDE.md chains eda → train → evaluate → bias_analysis, but omits `compare.py`. The 7-lens comparison framework is a full pipeline step with its own artifacts.

**Fix:** Add `&& python scripts/compare.py --data ... --models outputs/models/ --output outputs/` to the pipeline chain.

#### Finding P6: CLI-first tenet — PASS

The full pipeline is executable as a single shell command chain. No manual steps. No notebook dependencies. Scripts produce all artifacts needed for analysis. The notebook (when created) will only visualize pre-computed results.

#### Finding P7: Hypothesis coverage — MOSTLY COMPLETE

`resolve_hypotheses()` covers 11 of the 11 planned comparisons from PLAN.md. The M7 vs M7c update (log-target fair comparison) is an improvement over the original M6 vs M7b. All hypotheses are surfaced with winner, p-value, and significance indicator.

One gap: the evaluate.py stdout prints hypotheses as terse one-liners ("M2 wins*") — it doesn't print the interpretive conclusion ("Pricing is multiplicative, not additive"). The assessor has to map model names to business meaning mentally.

**Recommended action:** Add a brief interpretation after each hypothesis line, e.g.: "Is pricing multiplicative? → M2 wins → Yes, log-linear model confirms multiplicative structure."

### 3.3 API Design

#### Finding P8: Silent estimator defaulting — MEDIUM

`_request_to_dataframe()` at `api.py:174` defaults `None` estimator to `"Sato-san"`. This means:
- For models that include estimator as a feature (M1-M8), the prediction is biased toward Sato's pricing style
- Sato is the "aggressive" estimator (-10.4% bias) — so the API systematically returns lower estimates for callers who don't specify an estimator
- This is undocumented in the API response

**Recommended fix options:**
1. Return the median prediction across all three estimator values (most neutral)
2. Document the default explicitly in the API response/confidence flags
3. Make estimator required, or flag "estimator defaulted to Sato-san" in confidence_flags

#### Finding P9: SHAP feature names are not human-readable — MEDIUM

The SHAP explanation returns raw feature names from the label-encoded matrix:
- `base_part_type` → integer code (e.g., 7), not "Sensor Housing"
- `material` → integer code (e.g., 2), not "Stainless Steel 17-4 PH"
- `material_cost_tier` → technically readable but domain-specific
- `estimator` → integer code, not "Tanaka-san"

An estimator reading the API response sees `{"feature": "material", "contribution": 2800.0}` — which is meaningless without a reverse lookup.

**Recommended fix:** Post-process SHAP results to replace integer codes with original category names. The feature matrix construction has the mappings available.

#### Finding P10: SHAP model selection priority may be wrong — LOW

`api.py:213` prioritizes M6 > M7 > M5 for SHAP explanations. But M6 has 38% MAPE (raw-target, poor). If the API loads all models and uses median as the estimate, the SHAP explanation comes from one of the worst models while the estimate comes from the median (which is heavily influenced by the better log-target models).

**Recommended fix:** Prioritize M2 (best overall) or M7 (best tree with log target). Linear model SHAP via coefficients is also an option for M2.

#### Finding P11: Override reason categories — PASS

The 8 categories (material_hardness, geometry_complexity, surface_finish, tooling_difficulty, customer_relationship, scrap_risk, certification_requirements, other) are well-chosen for aerospace machining. They map cleanly to the missing variables table in PLAN.md §7. The free-text field allows nuance. This is well-designed.

#### Finding P12: OOD detection coverage — PASS with note

`detect_ood()` checks: quantity range, lead time range, unknown material/process/part, missing material/process. This covers the VALIDATION_PLAN.md §3 requirements. However, it does **not** check for unusual feature *combinations* (e.g., Inconel + Surface Grinding, which may never appear in training data). The plan mentions nearest-neighbor OOD detection as a production enhancement — correctly described as out of scope.

### 3.4 Missing Deliverables & Gaps

#### Finding P13: Notebook — CRITICAL (repeat of P1)

Not created. Highest-priority gap. The 10-section outline in PLAN.md is ready to implement.

#### Finding P14: TOOLING.md — COMPLETE

Despite earlier concern that TOOLING.md was interrupted, it is in fact complete. Covers dependency analysis, CLI workflow (9 sections), and testing strategy. One discrepancy: it lists `papermill>=2.5` as a dev dependency, but pyproject.toml doesn't include it.

**Fix:** Add `papermill>=2.5` to pyproject.toml dev dependencies if we plan to use it for notebook execution, or remove the reference from TOOLING.md.

#### Finding P15: Optuna tuning not implemented — ACCEPTABLE

The plan mentions Optuna with 50-100 trials per tree model. Current implementation uses reasonable defaults (max_depth=4, lr=0.05, min_child_weight=5). Given that M2 (Ridge log-linear) wins at 10.8% MAPE and tree models on raw target perform poorly regardless of tuning, Optuna tuning would primarily benefit M7/M7c (log-target trees at ~12.5-13% MAPE).

**Assessment:** Defensible without tuning. M2 wins on simplicity anyway. The notebook should mention that tuning is a planned enhancement, not an oversight.

#### Finding P16: Prediction bands partially implemented — ACCEPTABLE

Of the 4 planned band types:
1. **Model disagreement** — implemented (`compute_model_disagreement()`)
2. **Prediction interval** — NOT implemented (mapie not integrated)
3. **Estimator spread** — implementable from bias analysis but not wired into API
4. **Model uncertainty (CI)** — NOT implemented

The API returns `model_range` (min/max across models). This is the most useful band for the prototype. The plan correctly describes the others as enhancements.

**Assessment:** Acceptable for prototype scope. The notebook should show the display format mockup from PLAN.md and note which bands are implemented vs planned.

#### Finding P17: compare.py not in CLAUDE.md pipeline — repeat of P5

#### Finding P18: requirements.txt exists and is current — PASS

Generated by `uv pip compile pyproject.toml -o requirements.txt`. 164 lines with pinned versions. Present for evaluators who don't use uv.

### 3.5 Findings Summary

| ID | Finding | Severity | Action |
|---|---|---|---|
| P1 | Notebook not created | **Critical** | Create analysis.ipynb per PLAN.md outline |
| P2 | Missing variables not standalone | Low | Surface in notebook §8; optionally emit JSON |
| P3 | Hypothesis resolution CSV stale | Medium | Re-run evaluate.py |
| P4 | M0 not called out in stdout | Low | Add 1-2 line call-out in evaluate.py |
| P5 | CLAUDE.md pipeline missing compare.py | Low | Add to command chain |
| P7 | Hypothesis lines lack interpretation | Low | Add business-meaning suffix |
| P8 | Silent estimator default to Sato-san | Medium | Document or neutralize |
| P9 | SHAP feature names not human-readable | Medium | Post-process integer codes to names |
| P10 | SHAP model priority wrong | Low | Prioritize M2 or M7 over M6 |
| P14 | papermill mismatch | Low | Sync pyproject.toml ↔ TOOLING.md |

**Items that passed:** CLI-first tenet, hypothesis coverage, override categories, OOD detection, TOOLING.md completeness, requirements.txt, Optuna deferral defensible, prediction bands scope acceptable.

---

## 4. Validation & Automation — REVIEW COMPLETE

Test coverage, CI readiness, and validation framework integrity.

### 4.1 Test Coverage Overview

**166 total tests** across 6 test files. 120 fast, 46 slow-marked.

| Test File | Tests | Module Covered | Focus |
|---|---|---|---|
| `test_data.py` | 29 | `data.py` | Schema, validation, missing values |
| `test_features.py` | 38 | `features.py` | Parser, encoding, feature matrix |
| `test_comparison.py` | 44 | `comparison.py` | 7-lens framework, domain invariants |
| `test_api.py` | 13 | `api.py` | Endpoints, overrides, validation |
| `test_bias.py` | 12 | `bias.py` | Bootstrap CI, bias computation, report |
| `test_models.py` | 9 (+14 parametrized) | `models.py` | Smoke fit/predict, factory, metrics |

#### Finding V1: `predict.py` has zero test coverage — MEDIUM

**Location:** `src/price_estimator/predict.py` (231 lines)

Four public functions completely untested:
1. **`TrainingBounds.from_dataframe()`** — computes bounds from training data. Untested that it extracts correct ranges.
2. **`detect_ood()`** — flags out-of-distribution inputs. Untested for: OOD quantity, OOD lead time, unknown material, missing material, unknown process, unknown part description, in-distribution inputs.
3. **`compute_model_disagreement()`** — computes min/max/median/spread across models. Untested for: single model, multiple models, flagging >20% spread.
4. **`compute_shap_explanation()` / `format_shap_explanation()`** — SHAP wrapper. Hard to unit test (requires trained tree model), but `format_shap_explanation()` is pure string formatting and trivially testable.

**Impact:** `detect_ood()` is used in the API (`api.py:256-259`). If it has a bug, OOD inputs pass silently without confidence flags. `compute_model_disagreement()` is used in the API's spread calculation — a bug would produce wrong confidence flags.

**Risk:** Medium. These functions are simple and readable, but "simple code with no tests" is exactly where subtle bugs hide (off-by-one in range checks, wrong dict key, etc.).

#### Finding V2: `analysis.py` has zero test coverage — LOW

**Location:** `src/price_estimator/analysis.py` (333 lines)

Six public functions untested: `compute_summary_stats()`, `compute_unit_price_analysis()`, `compute_volume_discount()`, `compute_rush_premium()`, `compute_lead_time_analysis()`, `compute_confounding_analysis()`.

**Impact:** Low — these are EDA/reporting functions called only by `scripts/eda.py`. They produce diagnostic output, not inputs to model training. A bug would surface as obviously wrong numbers in the EDA output, not as a silent pipeline corruption.

**Assessment:** Acceptable for prototype scope. These are pure computation functions that return dicts — easy to test but low consequence if deferred.

#### Finding V3: `test_models.py` has smoke tests only — acceptable but shallow

**Location:** `tests/test_models.py` (140 lines, 9 tests + 14 parametrized)

What's tested:
- ✅ Every model can fit/predict on 5-row sample without error
- ✅ Predictions are finite, positive, correct length
- ✅ M6 reproducibility (same data → same predictions)
- ✅ Factory returns 14 models with unique names
- ✅ `compute_metrics()` produces correct MAPE/MedAPE/RMSE/R²
- ✅ M0 and M6 CV on real data produces valid CVResults

What's NOT tested:
- ❌ **No correctness test for any specific model's predictions** — we verify predictions are positive and finite, but not that M2's predictions are in the right ballpark, or that Jensen's correction actually fires
- ❌ **No test that log-target models back-transform correctly** — a bug in Jensen's correction (`0.5 * sigma²`) would pass all current tests
- ❌ **No test for M3's two-stage decomposition** — the most algebraically complex model has only "does it not crash?" coverage
- ❌ **No cross-model sanity check** — e.g., "M2 should beat M1 by >50% MAPE on real data" (would catch if M2's log transform is silently disabled)
- ❌ **No test for model serialization round-trip** — joblib save/load is the production path but untested

**Assessment:** The smoke tests are valuable (they caught the StandardScaler overflow early). But they only test the interface contract ("returns array of floats"), not the model logic. A model that always returns the training mean would pass every current test.

#### Finding V4: `test_bias.py` covers structure but not correctness — LOW-MEDIUM

What's tested:
- ✅ `_bootstrap_ci()` correctness (constant values, ordering, single value, contains mean)
- ✅ `format_bias_report()` contains expected strings
- ✅ `compute_estimator_bias()` returns correct keys and structure
- ✅ Labels are valid (`safe`, `aggressive`, `neutral`)
- ✅ n_quotes sums to 510
- ✅ CI widths are reasonable (<50pp)

What's NOT tested:
- ❌ **Bias direction correctness on real data** — we don't verify that Tanaka is labeled "safe" and Sato is labeled "aggressive" (the known ground truth from our analysis). A sign flip in the residual calculation would pass all tests.
- ❌ **Bias magnitude reasonableness** — no test that mean bias is within [-30%, +30%] (the +inf% bug would have passed the current suite)
- ❌ **`by_part_type` and `over_time` structure** — tested that top-level keys exist, but not that sub-dicts have the right format
- ❌ **Denominator correctness** — the fix changed denominator from `neutral_preds` to `actual prices`. No regression test prevents reverting this.

#### Finding V5: `test_api.py` is solid but misses VALIDATION_PLAN.md §4 — LOW

What's tested:
- ✅ POST /quote success, 503 when no models, confidence flags for missing material/process
- ✅ POST /quote/{id}/override success, 404 for unknown quote, delta calculation
- ✅ GET /quote/{id} with and without override, 404 for unknown
- ✅ Input validation (quantity=0, negative, lead_time=0, >52)
- ✅ DB isolation via `tmp_path` monkeypatch

What's NOT tested (per VALIDATION_PLAN.md §4):
- ❌ **Override stores structured reason category** — the override test doesn't include `reason_category`; only sends `human_price`
- ❌ **Override stores estimator_id** — not tested
- ❌ **Override stores reason_text** — not tested
- ❌ **All override fields retrievable via GET** — GET test checks `human_price` but not `reason_category`, `reason_text`, `estimator_id`

**Impact:** Low. The Pydantic model and SQLite schema handle these fields. But the VALIDATION_PLAN explicitly requires testing that "every override is stored with: original model price, human price, structured reason category, free-text reason, estimator ID, timestamp."

---

### 4.2 Domain Invariant Coverage

#### Finding V6: Domain invariants tested only for M0 and M2 — MEDIUM

**Location:** `test_comparison.py` lines 247-323 (TestDomainInvariants, TestBoundarySafety)

Tests verify material ordering, quantity discount, rush premium, price floor/ceiling for M0 and M2. These are the **least likely** models to violate invariants:
- M0 encodes economics directly (lookup tables with material/process factors)
- M2 is a linear model in log-space — monotonic by construction if coefficients have correct signs

The **most likely violators** — tree models (M5, M6, M7) — are not tested for domain invariants. Trees can learn non-monotonic relationships:
- XGBoost could price Al7075 above Inconel if tree splits encode a spurious pattern
- Random Forest with max_depth=6 on 510 rows could overfit to noise that violates ordering
- Per-estimator M8 trains on ~170 rows per model — high overfitting risk

**Current safety net:** The comparison framework (`scripts/compare.py`) tests all models for economic coherence at runtime. But `compare.py` has never been run (Finding V9). And these checks aren't in CI — they run post-training, not on every commit.

**Recommended action:** Add at least M6 (primary tree candidate) to the TestDomainInvariants class. This catches violations in CI rather than relying on a manual compare.py run.

#### Finding V7: Missing degradation reasonableness check — LOW

VALIDATION_PLAN.md §3.3 requires: "prediction with missing Material should be within 3x of the with-material prediction." Current tests only check `pred > 0` (TestBoundarySafety lines 383-395). The 3x reasonableness bound is not tested.

**Impact:** Low — a 3x ratio is a very loose bound and unlikely to be violated. But it's an explicit requirement in our own validation plan that we're not enforcing.

---

### 4.3 Comparison Framework Validation

#### Finding V8: Stochastic probes are properly seeded — PASS

`comparison.py:389` uses `np.random.RandomState(CV_SEED)` where `CV_SEED = 42`. The 50 stochastic probe indices are deterministic. Assessors will get identical scorecard results.

#### Finding V9: `compare.py` never run against real models — MEDIUM

The 44 unit tests in `test_comparison.py` validate the framework's logic using M0 and M2 trained on real data. But `scripts/compare.py` (the CLI entry point that loads serialized models and produces the full report) has **never been executed end-to-end**.

Potential issues that unit tests don't catch:
- joblib-loaded models may behave differently than freshly-trained models (serialization gotchas)
- The 14-model scorecard with quantile ranks could produce unexpected rankings
- Figure generation (5 plots) is untested — matplotlib rendering bugs won't surface until first run
- JSON serialization of numpy types in the full report could fail on edge cases
- Model names from joblib filenames must match expected names in `_rate_interpretability()`

**Recommended action:** Run `scripts/compare.py --data resources/aora_historical_quotes.csv --models outputs/models/ --output outputs/` once and verify the output.

#### Finding V10: Deterministic probe coverage is narrow — ACCEPTABLE

Economic coherence deterministic probes use one fixed configuration per check:
- Material ordering: Sensor Housing, 3-Axis Milling, qty=10, no rush, Sato-san
- Process ordering: Sensor Housing, Aluminum 6061, qty=10, no rush, Sato-san
- Quantity discount: Sensor Housing, Aluminum 6061, 3-Axis Milling, no rush, Sato-san
- Rush premium: same config ± rush
- Complexity: Sensor Housing - standard vs high precision

This tests 1 of 250+ possible (part × material × process) configurations. A model could pass on "Sensor Housing + Al6061" but violate on "Turbine Blade Housing + Inconel."

**Mitigation:** The 50 stochastic probes sample from the training distribution, covering diverse real-world combinations. Together, deterministic (reproducible, debuggable) + stochastic (diverse, catches edge cases) provide reasonable coverage. Per our tenet: "evaluate both."

**Assessment:** Acceptable. Adding more deterministic probes (e.g., one per material) would be a marginal improvement.

#### Finding V11: Scorecard quantile rank stability — ACCEPTABLE

With 14 models, quantile ranks (1-5) mean each rank bucket has ~3 models. Adding or removing one model shifts ~3 other models' ranks. This is inherent to relative ranking.

**Mitigation:** The scorecard also shows absolute numbers (MAPE, signed error %, pass rates) alongside ranks. The narrative text uses absolute values. Ranks are a visual aid, not the sole decision basis.

---

### 4.4 Automation & CI Readiness

#### Finding V12: Pytest markers not registered — LOW (noisy, not broken)

`slow`, `integration`, and `api` markers are used but not declared in `pyproject.toml`. This produces **17 PytestUnknownMarkWarning** messages on every test run:

```
PytestUnknownMarkWarning: Unknown pytest.mark.slow - is this a typo?
```

Not broken — `pytest -m "not slow"` still works. But noisy, and an assessor might wonder if something is misconfigured.

**Fix:** Add to `pyproject.toml`:
```toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests that train models (deselect with '-m \"not slow\"')",
    "integration: marks tests using the full 510-row dataset",
    "api: marks API endpoint tests",
]
```

#### Finding V13: No coverage configuration — LOW

`pytest-cov` is installed (in dev dependencies) but not configured. No minimum coverage threshold. Running `pytest --cov=price_estimator` works but isn't in the default `addopts`.

**Assessment:** Acceptable for a prototype. Coverage thresholds on a research project can create perverse incentives (testing getters to hit 80% while complex logic goes untested).

#### Finding V14: Fast test run is fast — PASS

`pytest -m "not slow"` runs 120 tests in 1.87 seconds. Suitable for CI and rapid iteration. The `slow` marker correctly gates model-training tests.

#### Finding V15: No integration test for the full pipeline — MEDIUM

No test runs: load_data → build_feature_matrix → model.fit → model.predict → compute_metrics → results_to_dataframe. A broken intermediate step (e.g., feature column rename) would only surface when running scripts manually.

The smoke tests (`test_smoke_fit_predict`) test fit → predict on sample data, but:
- They don't test on real data shape/distributions (except the 2 slow M0/M6 CV tests)
- They don't test the evaluation pipeline (metrics → comparison table → hypothesis resolution)
- They don't test the bias pipeline (train M9 → compute residuals → bootstrap → labels)

**Recommended action:** One integration test that runs the core pipeline on real data:
```python
@pytest.mark.slow
@pytest.mark.integration
def test_full_pipeline(raw_data):
    """Full pipeline: load → feature → train → evaluate → bias."""
    # Train M0 and M2 (fast)
    results = run_all_cv(raw_data, model_names=["M0", "M2"])
    df = results_to_dataframe(results)
    assert df.iloc[0]["model"] == "M2"  # M2 should win
    assert df.iloc[0]["MAPE_mean"] < 20  # Sanity bound
    
    # Bias analysis
    bias = compute_estimator_bias(raw_data)
    assert len(bias["summary"]) == 3
```

#### Finding V16: Artifact overwrite behavior — PASS

Scripts use `mkdir(parents=True, exist_ok=True)` and write to fixed paths (`outputs/results/cv_results.csv`, etc.). Re-running overwrites cleanly. No stale file accumulation risk.

#### Finding V17: FutureWarning in test_data.py — LOW

```
FutureWarning: Setting an item of incompatible dtype is deprecated...
  sample_df.loc[sample_df.index[0], "RushJob"] = np.nan
```

`test_nan_rush_job_raises` sets a NaN into a boolean column. This works now but will break in a future pandas version. The test should use a different approach (e.g., construct a DataFrame with NaN RushJob from scratch rather than mutating a boolean column).

---

### 4.5 Findings Summary

| ID | Finding | Severity | Action |
|---|---|---|---|
| **V1** | `predict.py` has zero tests (OOD detection, model disagreement) | **Medium** | Write test_predict.py — detect_ood and compute_model_disagreement are critical API paths |
| **V2** | `analysis.py` has zero tests (EDA functions) | Low | Acceptable for prototype; low consequence |
| **V3** | `test_models.py` is smoke-only, no correctness tests | Low-Medium | Add at least: Jensen's correction fires, M2 beats M1 on real data |
| **V4** | `test_bias.py` doesn't verify bias direction | Low-Medium | Add regression test: Tanaka is safe, Sato is aggressive |
| **V5** | `test_api.py` doesn't test structured override fields | Low | Add reason_category and estimator_id to override test |
| **V6** | Domain invariants not tested for tree models | **Medium** | Add M6 to TestDomainInvariants |
| **V7** | Missing degradation 3x reasonableness check | Low | Add to TestBoundarySafety |
| **V8** | Stochastic probes seeded | Pass | — |
| **V9** | compare.py never run end-to-end | **Medium** | Run once and verify output |
| **V10** | Deterministic probe coverage narrow | Acceptable | Stochastic probes compensate |
| **V11** | Scorecard rank stability | Acceptable | Absolute numbers alongside ranks |
| **V12** | Pytest markers not registered (17 warnings) | Low | Add markers to pyproject.toml |
| **V13** | No coverage configuration | Low | Acceptable for prototype |
| **V14** | Fast tests are fast (1.87s) | Pass | — |
| **V15** | No full pipeline integration test | **Medium** | Add one integration test for core pipeline |
| **V16** | Artifact overwrite | Pass | — |
| **V17** | FutureWarning in test_data.py | Low | Refactor test to avoid boolean column mutation |

**Priority order for fixes:**

1. **V9** — Run compare.py once (zero code effort, high validation value)
2. **V1** — Write test_predict.py (detect_ood is an API-critical path)
3. **V6** — Add M6 to domain invariant tests (catches tree model economic violations in CI)
4. **V15** — Add one pipeline integration test
5. **V12** — Register pytest markers (noise reduction)
6. **V3/V4** — Add correctness tests for models and bias direction
7. **V5/V7/V17** — Low-severity cleanups

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
