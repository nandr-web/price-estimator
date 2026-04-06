# Agent 002 — Price Estimator: Analysis & Design

## Problem Context

AORA has acquired a precision machine shop whose lead estimator has been quoting aerospace parts by hand for 30 years using "feel" rather than a formula. We have 510 historical quotes in a messy CSV and need to build a prototype that can predict `TotalPrice_USD` for new quotes.

### The Five Tasks

1. **The Engine** — Predict `TotalPrice_USD` for a new quote
2. **Feature Extraction** — Extract "tribal" complexity indicators from `PartDescription`
3. **Bias Detection** — Identify and account for estimator-specific biases
4. **Human-in-the-Loop Workflow** — API endpoint for human overrides that feed back into the system
5. **Missing Variables** — What additional data would improve accuracy (3D models, measurements, etc.)

---

## Data Profile

- **510 rows**, tabular regression
- **Target**: `TotalPrice_USD`
- **Features**:

| Column | Type | Values |
|---|---|---|
| QuoteID | ID | Q-1000 to Q-1509 |
| Date | Date | 2023-01 to 2024-03 |
| PartDescription | Text | 11 base part types with complexity modifiers |
| Material | Categorical (5 + missing) | Aluminum 6061, Aluminum 7075, Inconel 718, Stainless Steel 17-4 PH, Titanium Grade 5 |
| Process | Categorical (5 + missing) | 3-Axis Milling, 5-Axis Milling, CNC Turning, Surface Grinding, Wire EDM |
| Quantity | Discrete | 1, 5, 10, 20, 50, 100 |
| LeadTimeWeeks | Numeric | Varies |
| RushJob | Binary | Yes / No |
| Estimator | Categorical (3) | Sato-san, Suzuki-san, Tanaka-san |

### PartDescription Values (11 types)

| Base Part Type | Modifier |
|---|---|
| Actuator Linkage | (none) |
| Electronic Chassis | EMI shielded |
| Fuel Injector Nozzle | high precision |
| Heat Sink | high fin density |
| Landing Gear Pin | hardened |
| Manifold Block | complex internal channels |
| Mounting Bracket | standard |
| Sensor Housing | threaded |
| Structural Rib | aerospace grade |
| Turbine Blade Housing | thin walls |

### Messiness

- Missing Material and Process values in some rows
- "Tribal knowledge" embedded in part description text (e.g., "thin walls", "high precision", "complex internal channels")
- Three different estimators with unknown individual biases

---

## Proposed Models

### Model 1: Ridge Regression (Interpretable Baseline)

**Purpose**: Establish a price floor/ceiling of explainability. If a linear model gets R^2 of 0.85, we know 85% of pricing is driven by simple additive factors. The gap between this and a nonlinear model tells us how much "interaction knowledge" the estimator carries.

**Features consumed**:
- One-hot: base part type (11), material (5), process (5), estimator (3)
- Numeric: log(quantity), lead_time_weeks
- Binary: rush_job
- Engineered: material_cost_tier (ordinal 1-4), complexity_score (from PartDescription parsing)
- Interaction: `material_cost_tier x log(quantity)` (expensive materials scale differently at volume)

**Data flow**:
```
Raw CSV -> Clean/impute -> Feature extraction -> One-hot encode
-> StandardScaler -> Ridge(alpha via CV) -> predictions
```

**What it tells us**: Feature coefficients are directly interpretable -- "Inconel adds ~$X per unit", "Sato-san quotes $Y higher on average." This becomes the explainability layer even if we use a better model for actual predictions.

**Risks**:
- Underfits nonlinear interactions (e.g., Inconel + 5-Axis + small qty = expensive)
- Assumes additive pricing -- real quoting has multiplicative relationships (material cost x machining hours x quantity discount)

**Mitigation**: Also try log-target regression (`log(TotalPrice_USD)`) which makes multiplicative relationships additive. Compare both.

---

### Model 2: XGBoost / LightGBM (Primary Predictor)

**Purpose**: Capture nonlinear interactions that drive real pricing -- material x process difficulty, quantity discount curves, rush premiums that vary by complexity. This is likely the production model.

**Features consumed**: Same as Model 1 but without one-hot encoding -- tree models handle categoricals directly (LightGBM) or via ordinal encoding (XGBoost). No need for scaling.

**Data flow**:
```
Raw CSV -> Clean/impute -> Feature extraction -> Label-encode categoricals
-> XGBoost/LightGBM with 5-fold CV -> hyperparameter tuning (Optuna)
-> predictions + SHAP values for explainability
```

**Why both XGBoost and LightGBM**: They have different splitting strategies. LightGBM uses leaf-wise growth (better with small data), XGBoost uses level-wise (more regularized). With 510 rows, the difference matters -- we pick whichever CVs better.

**Key hyperparameters to tune** (Optuna, 50-100 trials):
- `max_depth`: 3-6 (shallow -- small dataset overfits easily)
- `n_estimators`: 50-300 with early stopping
- `learning_rate`: 0.01-0.1
- `min_child_weight` / `min_data_in_leaf`: high values to prevent overfitting
- `subsample`, `colsample_bytree`: 0.6-0.9

**Explainability**: SHAP values per prediction -- "this quote is $2,400 because: Inconel (+$800), 5-axis (+$600), qty=1 (+$400), rush (+$350), Sato-san (+$250)." This is critical for the human-in-the-loop workflow -- the estimator can see *why* the model priced it that way.

**Risks**:
- 510 rows is small for GBMs -- overfitting is the primary risk
- Categorical features with 11 levels (part type) can dominate splits
- Doesn't extrapolate -- a new material or qty=500 would be unreliable

**Mitigation**: Aggressive regularization, cross-validation (not a single train/test split), prediction intervals via quantile regression (train models at quantiles 0.1, 0.5, 0.9 to give a range, not a point estimate).

---

### Model 3: Random Forest (Stability Benchmark)

**Purpose**: Less tuning-sensitive than GBMs, provides a stability benchmark. If XGBoost scores much higher than RF, it might be overfitting. If they're close, we have more confidence. Also useful for ensemble averaging.

**Data flow**: Same as Model 2 but with `RandomForestRegressor`. Key params: `n_estimators=500`, `max_depth=8-12`, `min_samples_leaf=5-10`.

**Unique value**: RF naturally produces prediction variance -- the spread across trees for a single prediction gives a confidence estimate. "This quote is $3,200 +/- $400" is far more useful to an estimator than just "$3,200."

**Risks**: Tends to predict toward the mean of training data -- extreme quotes (very cheap or very expensive) will be pulled toward center. Less of an issue for typical quotes, problematic for unusual ones.

---

### Model 4: Estimator Bias Model (Two-Stage)

**Purpose**: Directly answer Task 3 -- quantify and correct for each estimator's systematic bias.

**Architecture -- two-stage approach**:

```
Stage 1: Train a "neutral" model (XGBoost) WITHOUT the Estimator feature
         This learns the "objective" price given part/material/process/qty

Stage 2: Compute residuals per estimator:
         bias_i = mean(actual_price - neutral_prediction) for estimator i
         Also compute: std_dev, percentile distributions, bias by part type
```

**Data flow**:
```
Raw CSV -> Feature extraction (no estimator) -> Train neutral XGBoost
-> Predict on all data -> Compute per-estimator residuals
-> Statistical tests (t-test, bootstrap CI) for significance
-> Output: bias profile per estimator
```

**What it produces**:
- **Sato-san**: mean residual = +$X -> "safe" quoter (quotes high, protects margin)
- **Tanaka-san**: mean residual = -$Y -> "aggressive" quoter (quotes low, wins bids but thinner margin)
- **Suzuki-san**: mean residual ~ 0 -> "neutral"
- Per-estimator bias *by part type* -- maybe Sato-san is only conservative on Inconel parts
- Bias drift over time -- is someone getting more conservative recently?

**Production use**: When generating a new quote, the system can say: "Neutral estimate: $4,200. Adjusted for Sato-san's typical +12% bias: $3,740 (de-biased)."

**Risks**:
- With ~170 quotes per estimator, some part/material/estimator combinations will have <5 data points -- bias estimates for those cells are unreliable
- Confounding: maybe Sato-san genuinely gets harder jobs (selection bias, not pricing bias)

**Mitigation**: Only report bias with statistical confidence. Check for confounding by comparing the *distribution* of jobs across estimators (are they assigned randomly or by specialty?).

---

### Model 5: LLM-Augmented Feature Extraction + Tabular Model

**Purpose**: Handle *unseen* part descriptions that don't match known patterns. The current 11 part types are a closed vocabulary -- in production, new parts will arrive with descriptions like "Thrust Reverser Cowl - tight tolerances, thin leading edge." This model handles that.

**Data flow**:
```
New PartDescription -> Deterministic parser (see PartDescription Parsing section)
  -> If all features extracted: pass to XGBoost
  -> If unknown pattern flagged: send to LLM for proposed extraction
    -> Human reviews LLM proposal -> approved mapping added to registry
    -> Features passed to XGBoost
```

**When to use**: Only for the *feature extraction* step -- the actual price model is still XGBoost. The LLM is a fallback parser, not a price predictor. This is important: you don't want pricing decisions dependent on LLM stochasticity.

**Risks**:
- LLM hallucination -- might invent complexity features that don't exist
- Latency and cost for API calls on every unknown description
- LLM output is non-deterministic -- same input could produce different features

**Mitigation**: The LLM never goes directly to the model. It always proposes -> human approves -> mapping is saved deterministically. The LLM is an *accelerator for human review*, not an autonomous parser.

---

## Feature Engineering Strategy

| Source | Extracted Features |
|---|---|
| PartDescription | Base part type, complexity flags: `threaded`, `thin_walls`, `high_precision`, `complex_internal`, `hardened`, `high_fin_density`, `EMI_shielded`, `aerospace_grade`, `standard` |
| Material | Material category + material cost tier (Inconel > Ti > SS > Al) |
| Process | Process category + axis count / precision tier |
| Quantity | Log-quantity (price scales sub-linearly), quantity bins |
| RushJob | Binary flag |
| LeadTimeWeeks | Numeric + possible interaction with RushJob |
| Date | Time trend (inflation / pricing drift) |

---

## PartDescription Parsing: Three-Tier Architecture

### The Problem

Current descriptions follow a pattern: `{BasePartType} - {modifier1}, {modifier2}` or just `{BasePartType}`. But this is human-entered text -- in production we should expect:
- Typos: "Sensor Housng - threaded"
- New modifiers: "Landing Gear Pin - hardened, cryogenic treated"
- Entirely new part types: "Thrust Reverser Cowl"
- Inconsistent separators: "Manifold Block / complex internal channels"

### Architecture

```
Tier 1: DETERMINISTIC REGISTRY (handles ~95% of inputs)
  |  Exact and fuzzy match against known patterns
  |  -> Outputs structured features with 100% reproducibility
  |
  |  x No match
  v
Tier 2: RULE-BASED NLP (handles ~4% of inputs)
  |  Token-level matching against known vocabulary
  |  -> Partial extraction with confidence scores
  |  -> Flags low-confidence fields for review
  |
  |  x Below confidence threshold
  v
Tier 3: LLM PROPOSAL + HUMAN APPROVAL (handles ~1%)
  |  LLM proposes structured extraction
  |  -> Human approves / edits
  |  -> Approved mapping written back to Tier 1 registry
  |  -> System learns permanently (deterministically)
```

### Tier 1: Pattern Registry (Deterministic)

A JSON/YAML registry mapping known descriptions to structured features:

```yaml
# registry.yaml
patterns:
  - pattern: "Sensor Housing"
    base_type: "housing"
    geometry: "enclosed"
    base_complexity: 2

  - pattern: "Manifold Block"
    base_type: "block"
    geometry: "prismatic"
    base_complexity: 3

modifiers:
  - pattern: "threaded"
    feature: "has_threads"
    complexity_delta: +1

  - pattern: "thin walls"
    feature: "thin_wall"
    complexity_delta: +2

  - pattern: "complex internal channels"
    feature: "internal_channels"
    complexity_delta: +3

  - pattern: "high precision"
    feature: "high_precision"
    tolerance_class: "tight"
    complexity_delta: +2

  - pattern: "standard"
    feature: "standard"
    complexity_delta: 0
```

**Lookup logic**: split on ` - `, match base part against `patterns`, match each modifier against `modifiers`. Exact string match first, then fuzzy.

**Libraries for Tier 1**:
- **`rapidfuzz`** -- fast fuzzy string matching (Levenshtein, Jaro-Winkler). Handles typos: "Sensor Housng" -> matches "Sensor Housing" at 95% similarity. Set a threshold (e.g., 90%) -- below that, escalate to Tier 2.
- **`thefuzz`** (formerly fuzzywuzzy) -- similar but `rapidfuzz` is faster

### Tier 2: Token-Level NLP (Rule-Based)

For descriptions that don't match the registry but contain known vocabulary.

Example: "Custom Bracket Assembly - threaded, hardened" -- "Custom Bracket Assembly" doesn't match any known base_type, but "threaded" and "hardened" ARE known modifiers.

**Libraries for Tier 2**:
- **`spaCy`** with a custom `EntityRuler` -- define token patterns for manufacturing terms. spaCy's rule-based matching is deterministic and fast:

```python
import spacy
from spacy.language import Language

nlp = spacy.blank("en")
ruler = nlp.add_pipe("entity_ruler")
patterns = [
    {"label": "MODIFIER", "pattern": [{"LOWER": "thin"}, {"LOWER": "walls"}]},
    {"label": "MODIFIER", "pattern": [{"LOWER": "high"}, {"LOWER": "precision"}]},
    {"label": "MODIFIER", "pattern": [{"LOWER": "threaded"}]},
    {"label": "MODIFIER", "pattern": [{"LOWER": "hardened"}]},
    {"label": "MATERIAL_HINT", "pattern": [{"LOWER": "aerospace"}, {"LOWER": "grade"}]},
    {"label": "GEOMETRY", "pattern": [{"LOWER": "internal"}, {"LOWER": "channels"}]},
]
ruler.add_patterns(patterns)
```

- **`flashtext`** -- keyword extraction at scale, O(n) regardless of number of keywords. Good if the modifier vocabulary grows to hundreds of terms.

**Output**: partial feature vector + confidence. If base_type is unknown but modifiers are recognized, flag for human review with a pre-filled form ("We recognized: threaded, hardened. What is the base part type?").

### Tier 3: LLM Proposal + Human Gate

Only triggered when Tiers 1-2 fail. The LLM's job is narrow: propose a structured extraction, not make a pricing decision.

```python
prompt = f"""
Given this machining part description: "{description}"

Extract:
- base_part_type: (one of: housing, block, bracket, rib, pin, nozzle,
  blade_housing, chassis, linkage, heat_sink, OTHER)
- modifiers: list of complexity modifiers
- geometry: enclosed / prismatic / cylindrical / flat / complex
- estimated_complexity: 1-5

Known modifiers: threaded, thin_walls, high_precision,
complex_internal_channels, hardened, high_fin_density,
EMI_shielded, aerospace_grade, standard, cryogenic_treated

If you see a modifier not in this list, flag it as NEW.
Return JSON only.
"""
```

**Critical**: The LLM output goes to a human approval queue, not directly to the model. Once approved, the mapping is written to the Tier 1 registry -- so the same description is handled deterministically forever after.

### Tier Comparison

| Property | Tier 1 | Tier 2 | Tier 3 |
|---|---|---|---|
| Deterministic | Yes | Yes | No (but gated by human) |
| Speed | <1ms | <5ms | 1-2s + human wait |
| Handles typos | Yes (fuzzy) | Yes (token-level) | Yes |
| Handles new parts | No | Partial | Yes |
| Handles new modifiers | No | No | Yes |
| Self-improving | Via Tier 3 approvals | Via Tier 3 approvals | -- |

The system starts deterministic for all known patterns and *grows its deterministic coverage* over time as the LLM+human path feeds back into the registry. Eventually, Tier 3 is rarely triggered.

---

## Full End-to-End Data Flow

```
+-------------------------------------------------------------+
|  1. INGEST                                                   |
|  Raw CSV / new quote request                                 |
|  -> Validate schema (required fields present?)               |
|  -> Flag missing Material/Process for imputation             |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|  2. PARSE PartDescription (Three-Tier Pipeline)              |
|  -> Pattern registry lookup (Tier 1)                         |
|  -> Token-level NLP fallback (Tier 2)                        |
|  -> LLM proposal + human approval (Tier 3)                   |
|  -> Extract: base_part_type, complexity_modifiers[]          |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|  3. FEATURE ENGINEERING                                      |
|  -> material_cost_tier (Inconel=4, Ti=3, SS=2, Al=1)        |
|  -> process_precision_tier (5-axis=3, EDM=3, CNC=2, etc.)   |
|  -> log_quantity, rush_flag, lead_time                       |
|  -> complexity_score (sum of weighted modifiers)             |
|  -> Impute missing values (mode for categoricals)            |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|  4. PREDICT (Model Ensemble)                                 |
|  -> Ridge: interpretable baseline                            |
|  -> XGBoost: primary prediction                              |
|  -> RF: confidence interval via tree variance                |
|  -> Weighted average or stacking                             |
|  -> Output: point estimate + confidence range + SHAP         |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|  5. BIAS ADJUSTMENT                                          |
|  -> If estimator specified: apply de-biasing correction      |
|  -> Show: "Raw model: $X | De-biased for Sato-san: $Y"      |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|  6. HUMAN-IN-THE-LOOP                                        |
|  -> Present quote + SHAP explanation to estimator            |
|  -> Estimator accepts OR overrides with corrected price      |
|  -> Override stored in SQLite: (features, model_price,       |
|     human_price, override_reason)                            |
|  -> Periodic retrain incorporates overrides as training data |
|  -> Override reasons analyzed for missing variables          |
+-------------------------------------------------------------+
```

---

## Reliability & Risk Matrix

| Risk | Severity | Likelihood | Mitigation |
|---|---|---|---|
| **Overfitting** (510 rows, many features) | High | High | Aggressive regularization, nested CV, shallow trees, ensemble |
| **Missing values** create data leakage | Medium | Medium | Impute *within* CV folds, never on full dataset first |
| **New part types** at inference time | High | Certain | Deterministic parser + LLM fallback + human review |
| **Extrapolation** (qty=500, new material) | High | Medium | Prediction intervals; flag when input is outside training distribution |
| **Estimator bias confounded with job difficulty** | Medium | Medium | Check job assignment distributions before drawing conclusions |
| **Stale model** as prices/materials change | Medium | High over time | Scheduled retrain, drift monitoring on prediction residuals |
| **Human overrides corrupt training data** | Medium | Low | Track overrides separately; require reason field; review before retrain |

---

## Evaluation Plan

- **5-fold or 10-fold CV** (dataset is small -- no luxury of a large holdout)
- **Metrics**: RMSE, MAE, MAPE, R^2 -- compared across all models
- **Estimator bias analysis**: per-estimator mean residual and confidence intervals
- **Ablation**: measure lift from each feature engineering step (especially the PartDescription parsing)

---

## Recommended Stack

| Layer | Choice | Why |
|---|---|---|
| Language | **Python** | Data science ecosystem |
| Notebook | **Jupyter** | Required "well-documented notebook" deliverable |
| Data | **pandas** | 510 rows, no need for anything heavier |
| ML | **scikit-learn** (Ridge, RF), **XGBoost**, **LightGBM** | Best tabular ML libraries |
| Tuning | **Optuna** | Hyperparameter optimization |
| Explainability | **SHAP** | Per-prediction feature attributions |
| NLP/Parsing | **rapidfuzz**, **spaCy** (EntityRuler), **flashtext** | Deterministic PartDescription parsing |
| LLM fallback | **Claude API** | Tier 3 feature extraction proposals |
| Evaluation | **scikit-learn** metrics (MAE, RMSE, MAPE, R^2) + cross-validation | Robust comparison |
| Visualization | **matplotlib / seaborn** | EDA, residual plots, estimator bias charts |
| API | **FastAPI** | Task 4: human-in-the-loop override endpoint |
| Storage | **SQLite** | Store overrides/corrections for learning loop |

---

## Missing Variables (Task 5)

If provided with 3D models/renders/measurements, the following would improve accuracy:

- **Geometry from 3D model**: total volume, surface area, bounding box dimensions, thinnest wall section, tightest internal radius
- **Tolerance specifications**: GD&T callouts, surface finish requirements (Ra values)
- **Material cost per kg**: real-time or recent material pricing
- **Tooling wear rate**: how quickly tools degrade on each material/process combo
- **Certification requirements**: AS9100, NADCAP, ITAR -- these add compliance overhead
- **Supplier lead times**: current material availability
- **Historical rework/scrap rates**: per part type, per process
- **Setup time**: number of setups/fixtures required
- **Post-processing**: heat treatment, coating, plating, NDT requirements
