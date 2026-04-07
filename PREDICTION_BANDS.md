# Prediction Bands — Analysis & Design

## The Problem

A point estimate ("this job is $6,200") is insufficient for quoting. Estimators need to know how confident the model is and what range is reasonable. Without ranges, every prediction looks equally certain — which it isn't.

## Approach: Empirical Residual Bands

We use the simplest defensible method: **compute the model's historical error distribution from cross-validation, then apply those error percentiles to new predictions.**

During 5-fold CV, every row gets an out-of-fold prediction. We compute the signed percentage error for each:

```
signed_error_pct = (prediction - actual) / actual × 100
```

The 10th and 90th percentiles of this distribution define the "typical range" — the band within which 80% of historical predictions fell.

For M2 (our best model at 10.8% MAPE), the empirical band is **-14.5% to +17.9%**. This means:
- 80% of past jobs were priced within this band
- The model slightly tends to overestimate (asymmetric band)
- For a $10,000 prediction: typical range is ~$8,500 — $11,700

## Why Not Conformal Prediction (mapie)?

We have mapie as a dependency and could use it. But:

1. **Explainability**: "8 out of 10 similar past jobs fell in this range" is immediately understandable. Conformal prediction requires explaining coverage guarantees and exchangeability assumptions.
2. **510 rows**: Conformal calibration with small data can produce bands that are technically valid but too wide to be useful.
3. **Prototype scope**: Empirical bands are honest about what we know. We can upgrade to conformal prediction when we have more data and need tighter guarantees.

## Top-Tier Model Selection

The recommendation is the median of "top-tier" models — not just the single best MAPE model. The top tier is selected with three criteria:

### 1. Accuracy + Economic Coherence

MAPE < 14% and perfect economic coherence (10/10 on domain invariant checks). This gives us: M2, M3a, M3b, M4, M7.

### 2. Structural Diversity

The top tier must include both **linear** (M2, M3a, M3b, M4) and **tree** (M7) model families. Different model families have different failure modes — linear models assume log-additive pricing and miss superlinear interactions; tree models capture interactions but extrapolate flat. Including both means the consensus has a dissenting voice when one family's assumptions break down.

### 3. Segment-Weighted Performance

Overall MAPE can hide segment-specific blind spots. An expensive Inconel job matters more than a cheap Aluminum bracket (a 20% miss on $12K costs $2,400 vs $40). M7 (XGBoost log-target) is included partly because tree models tend to perform better on complex, high-value jobs where linear models underweight feature interactions.

**Observed example:** Row 20 (Manifold Block, Inconel, complex internal channels, qty=20). Linear models: ~$9,500. Tree models: ~$12,800. Actual: $11,958. The linear family systematically underquoted because the complexity × material interaction is superlinear. The family divergence flag caught this.

## Recommendation Format

```
Recommendation (consensus of top-tier models):
  Estimate:       $  9,602
  Win bid:        $  9,254
  Protect margin: $ 10,032

  Linear models:  $  9,452
  Tree models:    $ 12,849
  Divergence:     30% — linear and tree models disagree, review recommended
```

- **Estimate**: Best guess of true cost. Median of 5 top-tier models.
- **Win bid**: Shifted 75% toward the lower band edge (-10.9%). More competitive price, more likely to win work.
- **Protect margin**: Shifted 75% toward the upper band edge (+13.4%). Safer margin on jobs won.
- **Family divergence**: When linear and tree models disagree by >10%, it's flagged. This is often a signal that the job has characteristics (interaction effects, nonlinear pricing) that one family handles better.

## Range Types

```
Typical range (80% of similar past jobs):
  $8,100  —  $11,200

Model range (min/max across all models):
  $5,600  —  $17,700

Estimator range (if quoted by each estimator):
  Sato         $5,800
  Suzuki       $6,200
  Tanaka       $6,600
```

| Range | What it answers | Source |
|---|---|---|
| **Typical range** | "How far off could this estimate be?" | 10th/90th percentile of CV errors applied to prediction |
| **Model range** | "Do our models agree?" | Min/max across all 14 models |
| **Estimator range** | "What would each human quote?" | Same job run through estimator-aware models with each estimator |

## Interpretation for the Shop Floor

- **Tight typical range + no divergence flag**: High confidence. Quote near the estimate.
- **Wide typical range**: The model has historically been uncertain about this type of job. Quote conservatively or dig deeper.
- **Family divergence flag**: Linear and tree models see this job differently. Often happens on complex/expensive parts where interaction effects matter. The tree median may be more trustworthy for these jobs — but review the specific case.
- **Estimator range much wider than typical range**: Humans disagree more than the model is uncertain. The job may involve judgment calls the model can't capture (missing variables).

## Implementation

- **Computed during**: `scripts/train.py` → saved as `outputs/results/prediction_bands.json`
- **Applied in**: `scripts/predict_one.py` and `src/price_estimator/api.py`
- **Band math**: `lower = prediction / (1 + upper_error_pct/100)`, `upper = prediction / (1 + lower_error_pct/100)` (inverted because the error is from the model's perspective)
- **Default model for bands**: M2 (lowest median absolute error), or best available model with bands

## Win Bid / Protect Margin Calibration

The shift factor for Win bid and Protect margin was calibrated by running all 510 historical rows through the recommender and measuring how often the actual price falls outside the [Win bid, Protect margin] range.

### Factor sweep (M2 band: -14.5% / +17.9%)

| Factor | Win bid shift | PM shift | Actual in range | PM exposed (underquoted) | Total $ at risk |
|--------|-------------|----------|----------------|--------------------------|-----------------|
| 0.25 | -3.6% | +4.5% | 30.2% | 28.4% (145/510) | $197K |
| 0.50 | -7.2% | +9.0% | 56.7% | 17.5% (89/510) | $144K |
| **0.75** | **-10.9%** | **+13.4%** | **70.6%** | **12.4% (63/510)** | **$112K** |
| 1.00 | -14.5% | +17.9% | 84.5% | 7.3% (37/510) | $93K |

### Why 0.75?

At 0.25, "Protect margin" was misleading — it failed to protect on 28% of jobs. The dollar exposure was concentrated in Q4 (expensive jobs >$12K): 41% of Q4 jobs were exposed, accounting for $162K of the $197K total risk. A "protect" recommendation that underquotes 1 in 3 expensive jobs isn't protecting anything.

At 0.75:
- **12.4% exposure rate** — the remaining misses are genuine outliers (missing data, extreme complexity interactions) rather than routine jobs
- **~70% of actuals** land within the [Win bid, Protect margin] range
- **$112K total exposure** vs $197K — 43% reduction
- The range spread is ~24% of estimate (vs 8% at 0.25), which is meaningful but not as wide as the full typical range (32%)

At 1.00, the bid-margin range essentially collapses into the typical range (84.5% vs 83.7% coverage), making the two outputs redundant. 0.75 preserves a useful distinction: the typical range answers "how far off could this be?" while bid-margin answers "where should I price to win/protect?"

### PM underquoting by price quartile (at 0.75)

The remaining 12% exposure concentrates in:
- **Q4 (>$12K)**: highest exposure rate and dollar risk — expensive complex jobs where model uncertainty is inherently higher
- **Missing-data rows**: a single $115K outlier (row 223) drives ~$60K of residual risk regardless of factor

## Limitations

- Bands are **global** — the same percentages apply to all predictions. In practice, the model is probably more accurate on common jobs (Al bracket, qty=100) than rare ones (Inconel turbine housing, qty=1). Segment-specific bands would be more honest but require more data per segment.
- Bands assume **future error distribution matches historical CV error distribution**. If the model encounters genuinely novel jobs, the actual coverage could be worse than 80%.
- With 510 rows and 5 folds, each fold has ~102 test points. The 10th/90th percentiles are estimated from ~10 points in the tails. More data would tighten these estimates.
