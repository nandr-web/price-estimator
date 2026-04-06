# Plan Review Feedback

The following items were identified during plan review and should be considered for incorporation into the plan. For each item, the planner should decide whether to integrate it into the implementation plan, add it as a documentation/narrative section, or note it as future work.

---

## 1. PartDescription Parser: Monitoring and Validation

The three-tier parser architecture is solid, but lacks validation and monitoring mechanisms as the registry grows over time.

**Recommendations:**
- **Tier escalation rate monitoring.** Track what percentage of quotes resolve at each tier (e.g., "98% Tier 1, 1.5% Tier 2, 0.5% Tier 3"). If Tier 2/3 rates increase, it means new part types are arriving faster than the registry grows.
- **Consistency validation for new registry entries.** When a Tier 3 approval adds a new pattern to the registry, validate that the assigned complexity score and base_type are consistent with how similar parts are priced. A human could approve a mapping that makes the model worse.
- **Revalidation after registry growth.** When new patterns are added to Tier 1, the model was trained without them. Define a trigger to retrain or at least measure whether the new features improve or degrade predictions.
- **Coverage reporting.** A simple dashboard or log showing Tier resolution rates over time, so the shop knows whether the system is maturing.

---

## 2. Add M0: Simple Lookup-Table / Multiplicative Formula Baseline

The model matrix starts at Ridge regression (M1), but there's no option simpler than that. Add an M0 that is a pure deterministic formula:

```
base_price = PART_TYPE_TABLE[part] × MATERIAL_TABLE[material] × PROCESS_TABLE[process]
total = base_price × qty^0.81 × (1.55 if rush else 1.0)
```

**Why this matters:**
- The assessors explicitly value pragmatism: "boring automation that saves 40 hours is often more valuable than a complex AI model."
- This could run in a spreadsheet — no Python needed on the shop floor.
- If M0 gets within ~15% MAPE, it might be the most *valuable* model for a dusty machine shop, even if XGBoost gets 8%.
- At minimum, it's a meaningful baseline that contextualizes the lift from more complex models.

---

## 3. Confounding Checks: Comprehensive Framework

The plan mentions "check estimator job distribution for confounding" but doesn't specify what to do if confounding exists, or how to check beyond estimator assignment. Here is a broader framework.

### Offline Checks (EDA / Model Development)

- **Cross-tabs with chi-squared tests.** For every pair of categorical features (Estimator × PartType, Estimator × Material, Material × Process, RushJob × Estimator, etc.), test for independence. A heatmap of p-values quickly shows where assignments are non-random.
- **Conditional vs marginal means.** Always show both. "Tanaka's average is $474" (marginal) vs "Tanaka's average controlling for job type is $420" (conditional). If these diverge, confounding is present. The current plan shows marginal means but doesn't contrast with conditional.
- **Propensity-style balance check.** Can you predict which estimator was assigned a job from the job features alone? If a classifier gets >40% accuracy (vs 33% random for 3 estimators), assignments are non-random.
- **Stratified residual analysis.** After fitting M9, plot residuals by estimator × part type, estimator × material, estimator × quantity band. If Tanaka's bias is +12% overall but +25% on Inconel and -5% on Aluminum, that's domain-specific conservatism, not uniform bias.

### Online Checks (Production)

- **Assignment drift monitoring.** Track whether the distribution of jobs per estimator changes over time. If the mix shifts, the bias model trained on historical data becomes unreliable. A simple chi-squared test on rolling windows.
- **Override correlation with job features.** If overrides cluster on specific part type + estimator combinations, the confounding structure may have shifted.
- **New estimator onboarding.** When a fourth estimator joins, the system has zero bias data. Options: start with no bias adjustment, collect N quotes before fitting their bias profile, or use the average of existing estimators as a prior.

### UX Implications

- **Show the estimator their bias context.** Display: "Your historical tendency on Inconel parts is +15% vs the model. On Aluminum parts you're within 3%." This lets the estimator self-calibrate and validates whether the bias model makes sense to them. If Tanaka says "I quote Inconel high because the scrap rate is terrible," that's missing-variable knowledge, not bias.
- **Flag low-confidence bias estimates.** If a particular estimator × part type cell has fewer than ~10 data points, indicate this: "Bias adjustment: +12% (based on 47 similar quotes)" vs "Bias adjustment: +8% (based on 3 quotes — low confidence)."

---

## 4. Missing Values: Decision Framework and Inference Behavior

The plan correctly specifies imputing within CV folds and comparing strategies, but is missing decision logic and production behavior.

**Recommendations:**
- **Decision criteria for imputation strategy.** After comparing mode imputation, "missing" category, and row dropping — what metric or criteria picks the winner? Best CV score? Preference for interpretability?
- **Document why values are missing.** The EDA should state the finding: 15 rows missing Material, 15 missing Process, only 3 overlap (27 distinct rows, ~5%). Distribution is roughly random across part types, lightly skewed toward Sato-san. Implication: missing-at-random → imputation is valid.
- **API behavior for missing fields at inference time.** When a new quote arrives with missing Material or Process: return the best estimate with correspondingly wider confidence bands and a clear warning that the estimate may not be reliable due to missing information. Do not reject the request.

---

## 5. Override Feedback Loop: Structure and Safety

The plan describes storing overrides in SQLite with free-text reasons and periodic retraining. Two additions:

- **Structured override reasons.** Provide a dropdown of common override reasons (material hardness, geometry complexity, surface finish, tooling difficulty, customer relationship, scrap risk, etc.) plus a free text field. Free text alone is hard to mine systematically. Structured categories give usable signal from day one and enable trend analysis ("40% of overrides cite surface finish → that's a missing variable").
- **Minimum sample threshold before retraining.** Define a minimum number of overrides (e.g., 30–50) before incorporating them into retraining. Additionally, validate that holdout performance improves after incorporating overrides — don't blindly add them to training data. This prevents a handful of noisy corrections from degrading the model.

---

## 6. Out-of-Distribution Detection

The risk matrix mentions "flag when input is outside training distribution" but no concrete mechanism is proposed. The API spec includes a `confidence_flag` field but doesn't define what triggers it beyond "multi-model disagreement > 20%."

**Recommendations:**
- **Feature-range checks.** Flag any input where a feature value falls outside the training range (e.g., qty=500 when max training qty is 100, or a new material not seen in training).
- **Nearest-neighbor distance.** Compute distance in feature space to the closest training examples. If the new quote is far from anything seen, widen confidence bands and flag for review.
- **Define all `confidence_flag` triggers.** Consolidate into a single list:
  - Multi-model disagreement > 20%
  - Any feature outside training range
  - Nearest-neighbor distance above threshold
  - Missing input fields
  - PartDescription resolved at Tier 2 or Tier 3 (not fully recognized)
  - Low-confidence bias estimate (small sample size for estimator × job type cell)

---

## 7. Jensen's Inequality Correction for Log-Target Models

Technical note: when training on `log(price)` and back-transforming predictions with `exp()`, the result is the geometric mean, not the arithmetic mean. This causes systematic underestimation due to Jensen's inequality.

**Fix:** Apply the correction `exp(log_pred + 0.5 * σ²)` where σ² is the residual variance in log-space. Alternatively, use median-based metrics (Median APE) which are unaffected. Ensure this correction is applied before computing MAPE on back-transformed predictions.

---

## 8. Final Model Selection Criteria

The plan builds a model matrix and comparison table but doesn't state how the production model is chosen.

**Recommendation:** After running the full matrix, select based on best MAPE (primary metric) with consideration for explainability and robustness (low CV variance across folds). If several models perform comparably, use an ensemble (e.g., weighted average of top 2-3 models). The notebook conclusions section should explicitly name the "ship it" model(s) and justify the choice.
