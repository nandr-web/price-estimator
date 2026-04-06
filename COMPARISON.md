# Model Comparison Framework — Seven Lenses

## Motivation

`scripts/evaluate.py` answers: **"Which model has the lowest MAPE?"**

This framework answers a different question: **"Which model should we trust?"**

These are not the same question. A model can achieve the best MAPE while:
- Systematically underquoting expensive jobs (dangerous for margin)
- Violating basic economic relationships (Aluminum priced above Inconel)
- Being unstable across different data subsets (fragile with 510 rows)
- Producing catastrophic outlier predictions hidden by a good average

The 7-lens framework provides a **second, independent reference point** to cross-check our MAPE-based model selection. If the MAPE-best model fails sanity checks, something is wrong. If a model ranks well on both MAPE *and* the multi-lens scorecard, we have stronger confidence in shipping it.

## Design Principles

1. **Validation-first**: We start from "what properties must a good pricing model have?" and score backwards — not from "let's find more metrics to compute."
2. **Both deterministic and stochastic probes**: Economic coherence checks use fixed synthetic inputs (reproducible, interpretable) AND sampled inputs from the training distribution (covers real-world diversity). Per tenet: evaluate both.
3. **Absolute scoring for economic coherence, relative for everything else**: Either Inconel costs more than Aluminum or it doesn't — no grading on a curve. But for MAPE, stability, etc., we rank models against each other via quantile ranks (1-5).
4. **Readable by ML-familiar non-experts**: The scorecard uses ●○○○○ dot ratings. Narratives are one-sentence plain English. The assessor shouldn't need to interpret p-values to understand the tradeoffs.

---

## The Seven Lenses

### Lens 1: Error Profile

**Question**: *Where* does the model make mistakes, and how bad are the worst ones?

Two models at 10% MAPE can look very different:
- Model A: tight errors (all 8-12%) — reliable
- Model B: many 2% errors but a few 50%+ misses — "usually fine but occasionally catastrophic"

For a shop floor, Model A is better despite identical average accuracy.

**Metrics**:
| Metric | What it reveals |
|---|---|
| MAPE | Average accuracy |
| Median APE | Typical accuracy (robust to outliers) |
| P90 APE | How bad are the worst 10%? |
| P95 APE | How bad are the worst 5%? |
| Max APE | Single worst prediction |
| Skew | Tendency to over/under-predict |
| % under 5/10/20% | How often is the model "close enough"? |
| % over 50% | Catastrophic miss rate |

**Visualization**: Histogram of absolute percentage errors per model. Shape tells the story.

**Scoring**: Quantile rank on MAPE (lower is better → rank 5).

---

### Lens 2: Segment Fairness

**Question**: Does the model work equally well across all job types?

A model with 10% overall MAPE might be 5% on Aluminum/qty=100 (easy, common) but 25% on Inconel/qty=1 (hard, high-value). The overall metric hides the blind spot. If the shop floor quotes Inconel jobs wrong by 25%, the 10% average is misleading.

**Segments tested**:
| Segment | Why |
|---|---|
| Price quartile | Does it nail cheap jobs but miss expensive ones? |
| Material | Is it worse on hard-to-machine materials? |
| Part type | Any part types systematically mispriced? |
| Quantity band | qty=1 vs qty=100 accuracy gap? |
| Estimator | Does it fit one estimator's style better? |

**Visualization**: Heatmap — rows are segments, columns are models. Color = MAPE.

**Scoring**: Quantile rank on worst-segment MAPE (lower is better). A model that's good on average but terrible somewhere scores low.

---

### Lens 3: Economic Coherence

**Question**: Do predictions respect basic economic relationships?

This is the "shop floor smell test." An estimator would immediately reject a model that prices Aluminum above Inconel, even if its MAPE is great. Violations destroy trust.

**Checks (deterministic probes)**:
| Check | Expected | Method |
|---|---|---|
| Material ordering | Inconel > Ti > SS > Al7075 > Al6061 | Hold all else constant, vary material |
| Process ordering | 5-Axis > Wire EDM > 3-Axis > CNC > Grinding | Same |
| Quantity discount | More qty → lower unit price | Predict at qty=1,5,10,20,50,100 |
| Rush premium | Rush → higher price | Same job ± rush |
| Complexity premium | More modifiers → higher price | "standard" vs "high precision" |

**Checks (stochastic probes)**:
Same relationships tested on 50 randomly sampled job configurations from the training distribution. Catches violations that only appear with specific feature combinations.

**Scoring**: Absolute pass/fail. pass_rate / 20 → 1-5 scale. A model with economic violations is penalized regardless of how other models score.

---

### Lens 4: Calibration & Bias Direction

**Question**: When the model is wrong, does it err in a consistent direction?

In a machine shop, **underestimation is worse than overestimation**:
- Underquote → win the bid, lose money on the job
- Overquote → lose the bid, but protect margin on won jobs

A model that consistently overestimates by 3% is *safer* than one that consistently underestimates by 3%, even though both have 3% MAPE.

**Metrics**:
| Metric | What it reveals |
|---|---|
| Mean signed error (%) | Direction: positive = conservative, negative = aggressive |
| % overestimated | > 50% = protective |
| Signed error by price quartile | Is direction consistent or does it flip? |
| Label | "conservative" (>+3%), "aggressive" (<-3%), "balanced" |

**Visualization**: Bar chart of mean signed error per model.

**Scoring**: Quantile rank on |mean signed error %| — closer to zero is better.

---

### Lens 5: Stability & Robustness

**Question**: How much does model behavior change with small data perturbations?

With only 510 rows, instability is a real risk. A model with 8% MAPE ± 4% across folds is less trustworthy than 10% MAPE ± 0.5%.

**Tests**:
| Test | Method |
|---|---|
| Fold-to-fold variance | Std of MAPE across 5 CV folds |
| Bootstrap prediction stability | 20 resampled training sets → prediction spread (coefficient of variation) |

**Visualization**: Box plot of per-fold MAPEs per model. Tight box = stable.

**Scoring**: Quantile rank on fold MAPE std (lower is better).

---

### Lens 6: Boundary Behavior

**Question**: How does the model behave at distribution edges?

The API will receive inputs that push boundaries. Different model families have different failure modes:
- **Tree models**: extrapolate flat (repeat nearest leaf value)
- **Linear models**: extrapolate linearly (can go negative or explosive)
- **Lookup tables**: fall back to defaults

**Tests**:
| Test | What it checks |
|---|---|
| qty=200 extrapolation | Beyond training max (100). Reasonable ratio to qty=100? |
| qty=1000 extrapolation | Extreme. Does it degrade gracefully or explode? |
| Price floor | Does any cheap config predict < $50? (Below plausible minimum) |
| Price ceiling | Does the most expensive config stay < $500K? |
| Missing material | Does NaN material cause a crash or just degradation? |

**Visualization**: Table of stress test results per model, with failure mode annotations.

**Scoring**: Quantile rank on pass count (more passes is better).

**Note**: Boundary behavior tests are also available as a separate test/validation suite (`tests/test_comparison.py`) for CI integration. In production, boundary violations would trigger confidence flags with details on why the prediction is unreliable.

---

### Lens 7: Complexity & Interpretability

**Question**: Can we explain predictions to the shop floor?

Assessors value "a model the shop floor trusts" over maximum accuracy. Within ~1% MAPE, prefer the model that can show its work.

**Metrics**:
| Metric | Method |
|---|---|
| Effective parameter count | Non-zero coefficients (linear) or total leaves (trees) |
| Training time | Wall-clock seconds |
| Interpretability rating | High/Medium-High/Medium/Low-Medium/Low |
| Explanation method | Coefficients, SHAP, direct inspection |

**Rating scale**:
| Rating | Models | Why |
|---|---|---|
| High | M0, M1, M2, M3a, M3b | Lookup table / coefficients directly readable |
| Medium-High | M4 | Sparse Lasso coefficients, but many interaction terms |
| Medium | M5, M6, M6b, M7, M7b, M7c, M9 | SHAP available, but model internals are complex |
| Low-Medium | M8 | 3 separate models — harder to compare across estimators |

**Scoring**: Rating mapped to 1-5 directly.

---

## Scorecard Format

```
═══════════════════════════════════════════════════
  MODEL COMPARISON REPORT — 7 Lenses
═══════════════════════════════════════════════════

Model    Error   Segment  Econ.   Calib.  Stabil. Boundary Interp.   Avg
────────────────────────────────────────────────────────────────────────
M2       ●●●●○   ●●●●○   ●●●●○   ●●●●○   ●●●●●   ●●●○○   ●●●●○    4.0
M6       ●●●●●   ●●●●●   ●●●○○   ●●●●●   ●●●●○   ●●○○○   ●●●○○    3.6
...
────────────────────────────────────────────────────────────────────────

NARRATIVES

  M2: 10.8% MAPE, balanced. Fold stability: ±0.5%. Strong on: stability,
  calibration. Weak on: boundary behavior.

  M6: 12.5% MAPE, balanced. 2 economic coherence violation(s). Fold
  stability: ±1.2%. Strong on: error profile, segment fairness.
```

---

## Relationship to Other Components

| Component | Question it answers | Artifacts |
|---|---|---|
| `scripts/evaluate.py` | Which model has the best MAPE? | model_comparison.csv, significance_tests.csv |
| `scripts/compare.py` | Which model should we trust? | comparison_report.json, scorecard text, lens figures |
| `scripts/bias_analysis.py` | Which estimator is safe/aggressive? | estimator_bias.json, bias figures |

The comparison framework does **not replace** evaluate.py — it provides a complementary perspective. The final model selection in the notebook should reference both.

---

## Implementation

- **Library**: `src/price_estimator/comparison.py` — all 7 lens functions + scorecard generation
- **CLI**: `scripts/compare.py` — loads models, runs lenses, outputs reports
- **Tests**: `tests/test_comparison.py` — boundary behavior validation suite + unit tests
- **Presentation**: Notebook section 4 renders the scorecard from pre-computed JSON
