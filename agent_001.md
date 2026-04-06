# Agent 001: Price Estimator — Analysis & Plan

## Problem Summary

We have 510 historical quotes from a precision machine shop. The lead estimator quotes aerospace parts "by feel." We need to build a prototype that predicts `TotalPrice_USD` for new quotes, extracts complexity features from part descriptions, detects estimator biases, and supports human overrides.

## Data Overview

- **510 rows**, 10 columns
- **10 part types** with embedded complexity descriptors
- **5 materials**: Inconel 718, Ti Grade 5, SS 17-4 PH, Al 7075, Al 6061 (+ 15 missing)
- **5 processes**: Wire EDM, 5-Axis Milling, CNC Turning, Surface Grinding, 3-Axis Milling (+ 15 missing)
- **3 estimators**: Sato-san (180 quotes), Tanaka-san (170), Suzuki-san (160)
- **Quantities**: 1, 5, 10, 20, 50, 100
- **Prices**: $98 — $115K, mean ~$9K
- **Rush jobs**: ~13% of quotes
- **Lead time**: 2–12 weeks

## Key Data Observations

### Unit price by material (descending)
| Material | Mean Unit Price |
|---|---|
| Inconel 718 | $562 |
| Titanium Grade 5 | $479 |
| Stainless Steel 17-4 PH | $413 |
| Aluminum 7075 | $349 |
| Aluminum 6061 | $306 |

### Unit price by part type (descending)
| Part | Mean Unit Price |
|---|---|
| Manifold Block - complex internal channels | $757 |
| Fuel Injector Nozzle - high precision | $729 |
| Turbine Blade Housing - thin walls | $729 |
| Landing Gear Pin - hardened | $319 |
| Sensor Housing - threaded | $297 |
| Structural Rib - aerospace grade | $284 |
| Actuator Linkage | $265 |
| Heat Sink - high fin density | $261 |
| Electronic Chassis - EMI shielded | $254 |
| Mounting Bracket - standard | $190 |

### Unit price by process (descending)
| Process | Mean Unit Price |
|---|---|
| 5-Axis Milling | $702 |
| Wire EDM | $496 |
| 3-Axis Milling | $310 |
| CNC Turning | $286 |
| Surface Grinding | $215 |

### Unit price by estimator
| Estimator | Mean | Median |
|---|---|---|
| Tanaka-san | $474 | $357 |
| Suzuki-san | $409 | $328 |
| Sato-san | $369 | $273 |

### Other findings
- **Rush premium**: ~1.55x when controlling for part/material/process
- **Volume discount**: log-linear relationship (r = -0.38, slope = -0.19). 10x quantity => ~63% unit price
- **Lead time**: no meaningful trend across 2–12 weeks (~$400 flat). Likely a consequence of shop loading, not a price driver
- **Date**: only ~15 months of data, no obvious time trend

---

## Proposed Features

### Direct columns
| Feature | Encoding | Rationale |
|---|---|---|
| Material | One-hot (5 levels) | Huge unit price spread: Inconel $562 to Al 6061 $306 |
| Process | One-hot (5 levels) | 5-Axis Milling $702 vs Surface Grinding $215 |
| Estimator | One-hot (3 levels) | Tanaka $474, Suzuki $409, Sato $369 |
| RushJob | Binary (0/1) | ~1.55x premium |
| Quantity | log(Quantity) | Power-law volume discount |

### Extracted from PartDescription ("tribal knowledge")
| Feature | Rationale |
|---|---|
| Base part type (10 categories) | Strongest price signal after quantity |
| "thin walls" | Tight tolerances, slow feeds, high scrap risk |
| "complex internal channels" | Multi-setup, hard to inspect |
| "high precision" | Tighter tolerances = slower cycle time |
| "hardened" | Extra heat-treat step, harder to machine |
| "threaded" | Additional operations |
| "aerospace grade" | Certification/documentation overhead |
| "EMI shielded" | Special finishing requirements |
| "high fin density" | Toolpath complexity, fragile features |
| "standard" | Inverse signal — simplest work |

### Derived / interaction features
| Feature | Rationale |
|---|---|
| Material x Process interaction | Some combos are disproportionately expensive |
| Material difficulty score (ordinal 1-5) | Simpler alternative to one-hot for linear models |
| Complexity count (sum of flags) | Rough proxy for overall difficulty |

### Excluded (initially)
| Feature | Why |
|---|---|
| LeadTimeWeeks | No meaningful trend. Included in kitchen-sink approach for empirical testing |
| Date | Too short a window. Included in kitchen-sink approach |
| QuoteID | Identifier only |

---

## Modeling Approaches

### Approach 1: Baseline Linear (log-linear regression)
- Target: `log(TotalPrice)`
- Features: `log(Quantity)`, material one-hot, process one-hot, part type one-hot, complexity flags, rush binary, estimator one-hot
- Why: interpretable, gives coefficients ("Inconel adds 40%")

### Approach 2: Gradient Boosted Trees (XGBoost / LightGBM)
- Same feature set, handles interactions and non-linearities automatically
- Try both raw and log target
- Likely the best accuracy model

### Approach 3: One model per estimator
- Train 3 separate models (any algorithm), one per estimator
- Tests hypothesis: do estimators weight factors differently, or just shift prices up/down?
- If per-estimator doesn't beat single-model-with-estimator-feature, the bias is additive

### Approach 4: Kitchen sink — include everything
- Add LeadTimeWeeks, Date (numeric), month-of-year, day-of-week
- Interaction terms: Material x Process, Material x PartType, Rush x LeadTime
- Let regularization (Lasso/Ridge) or tree importance sort out significance
- Run feature importance / permutation importance to empirically settle "does lead time matter?"

### Approach 5: Two-stage model
- Stage 1: predict unit price from part/material/process/complexity/estimator
- Stage 2: apply quantity discount curve (`log(unit_price) = a + b*log(qty)` per part type or globally)
- Rush multiplier on top
- More physically interpretable

### Approach 6: Estimator-as-bias (debiased model)
- Train model WITHOUT estimator feature to get a "fair" price
- Fit estimator-specific bias terms as residuals
- Directly answers "who is safe, who is aggressive"
- Useful for human-in-the-loop: show AI price + estimator adjustment separately

---

## Evaluation Plan

- **Cross-validation**: 5-fold or 10-fold CV
- **Metrics**: MAPE, RMSE, R-squared, median absolute % error
- **Comparison table**: one row per approach, columns for each metric

---

## Prediction Bands

### A. Model uncertainty (confidence interval)
- "How uncertain is the model about the average price for this type of job?"
- Linear regression: analytical confidence intervals
- Trees: quantile regression (train at tau=0.1, 0.5, 0.9 for 80% interval)

### B. Prediction interval
- Wider — accounts for irreducible variance (the "feel" factor)
- Linear regression: analytical
- Trees: quantile models or conformal prediction

### C. Estimator spread as a band
- For any new job, show what each estimator would quote
- Most actionable band: shows range of human disagreement
- Display: `[aggressive] --- [AI estimate] --- [safe]`

### D. Multi-model disagreement band
- Run all approaches, show min/max/median prediction
- If models agree: high confidence. If they diverge: flag for human review
- Natural trigger for human-in-the-loop: "models disagree by >20%, please review"

### Recommended display
```
Quote Estimate: $12,400
+-- Model range:     $11,200 - $13,800  (6 models)
+-- 80% prediction:  $9,800  - $15,600  (historical variance)
+-- Estimator range: $10,900 (Sato) - $14,200 (Tanaka)
```

Three lenses: model consensus, statistical uncertainty, and historical human range.
