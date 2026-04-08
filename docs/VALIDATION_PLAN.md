# Validation Plan — Working Backwards from "What Should a Good Model Look Like?"

## Purpose

This document defines what a trustworthy pricing model *should* look like, independent of any specific model's results. We use these criteria as a second reference point to validate our implementation — if our best model fails these checks, we have a bug or a bad assumption, not just a model selection problem.

The approach: start from validation requirements → derive tests → check models against tests. This is the reverse of the typical workflow (train → evaluate → pick best MAPE).

---

## 1. Domain Invariants

These are economic relationships that **must hold** for any reasonable pricing model in a precision machine shop. Violations are bugs, not tradeoffs.

### 1.1 Material Cost Ordering

**Invariant**: For identical jobs, harder-to-machine materials must cost more.

```
Inconel 718 > Titanium Grade 5 > Stainless Steel 17-4 PH > Aluminum 7075 > Aluminum 6061
```

**Why**: Material difficulty drives machining time (Inconel dulls tools 10x faster than Aluminum), scrap risk, and raw material cost. This ordering is fundamental to aerospace manufacturing economics.

**Test**: Hold part type, process, quantity, rush, estimator constant. Vary only material. Check strict ordering of predictions.

**Pass criteria**: Zero violations across all probe configurations (absolute, not relative).

### 1.2 Process Precision Ordering

**Invariant**: More capable/precise processes cost more per unit.

```
5-Axis Milling > Wire EDM > 3-Axis Milling > CNC Turning > Surface Grinding
```

**Why**: Machine hour rates scale with capability. A 5-axis VMC costs $200+/hr; a surface grinder costs ~$60/hr.

**Test**: Same as material — hold everything else constant, vary process.

**Pass criteria**: Zero violations.

### 1.3 Volume Discount

**Invariant**: Unit price must decrease (or stay flat) as quantity increases.

**Why**: Setup costs (fixturing, programming, first-article inspection) are amortized across units. Raw material can be bought in bulk. Only learning-curve effects should drive unit cost down.

**Test**: Predict at qty={1, 5, 10, 20, 50, 100}. Compute unit price at each. Check monotonically non-increasing.

**Pass criteria**: Zero inversions.

### 1.4 Rush Premium

**Invariant**: Rush jobs cost at least as much as non-rush jobs.

**Why**: Rush requires schedule disruption, overtime, expedited material sourcing. No rational shop would charge less for rush.

**Test**: Same job, rush=True vs rush=False. Check rush prediction >= non-rush.

**Pass criteria**: Zero violations.

### 1.5 Complexity Premium

**Invariant**: Parts with more complexity modifiers should cost at least as much as simpler variants of the same base type.

**Why**: Thin walls require slower feeds, complex internal channels require EDM or 5-axis, high precision requires tighter QC.

**Test**: Compare "Part - standard" vs "Part - high precision" predictions.

**Pass criteria**: Complex >= Simple.

---

## 2. Statistical Properties

These are properties that a well-fitted model should exhibit. Unlike domain invariants, small violations are acceptable — they indicate room for improvement, not bugs.

### 2.1 Error Distribution Shape

**Requirement**: The error distribution should be:
- Unimodal (one peak, not bimodal — bimodal suggests the model has two distinct failure modes)
- Roughly symmetric (not heavily skewed toward over- or under-prediction)
- Light-tailed (90th percentile APE should be < 3x the median APE)

**Why a heavy tail is concerning**: If median APE is 8% but P90 is 40%, the model is unreliable — it's "usually fine but occasionally catastrophic." The shop floor won't trust it after a few big misses.

**Test**: Compute skewness, kurtosis, P90/median ratio from OOF predictions.

**Acceptable**: Skew within [-1, 1], P90/median ratio < 3.

### 2.2 No Systematic Segment Bias

**Requirement**: MAPE should not vary more than 2x between any two segments with sufficient data (n >= 20).

**Why**: If the model is 5% on cheap jobs but 25% on expensive jobs, the average MAPE is misleading. The high-value jobs (where accuracy matters most for margin) are poorly served.

**Test**: Compute segment MAPEs. Flag any segment where MAPE > 2x the overall MAPE.

**Acceptable**: Max segment MAPE < 2x overall MAPE for segments with n >= 20.

### 2.3 Calibration Consistency Across Price Tiers

**Requirement**: The model's bias direction (over/under-prediction) should be consistent across price quartiles, not flip-flopping.

**Why**: If a model overestimates cheap jobs but underestimates expensive ones, it's systematically distorting the quote book. Estimators can't develop intuition about "the model tends to be X% high."

**Test**: Compute mean signed error per price quartile. Check that the sign doesn't flip more than once.

**Acceptable**: At most one sign change across Q1→Q2→Q3→Q4.

### 2.4 Fold Stability

**Requirement**: Fold-to-fold MAPE standard deviation should be < 30% of mean MAPE.

**Why**: With 510 rows and 5 folds, each test fold has ~102 rows. High fold variance means the model is sensitive to which specific rows it trains on — unreliable for production.

**Test**: Std(fold MAPEs) / Mean(fold MAPEs) < 0.30.

**Acceptable**: CV coefficient < 0.30.

---

## 3. Boundary & Safety Properties

These protect the system in production when inputs push beyond the training distribution.

### 3.1 Price Floor

**Requirement**: No input combination should produce a prediction below $50.

**Why**: The minimum plausible job cost includes material ($10+), machine time ($30+/hr minimum), and setup. Even the cheapest possible part (Aluminum bracket, CNC turning, qty=100) has non-trivial cost.

**Test**: Predict cheapest plausible configuration. Check prediction >= $50.

### 3.2 Price Ceiling

**Requirement**: No input combination should produce a prediction above $500K.

**Why**: The training data maxes at ~$115K. A prediction 5x beyond the training maximum is almost certainly wrong. The API should flag this as low-confidence, not return it as a serious estimate.

**Test**: Predict most expensive plausible configuration. Check prediction <= $500K.

### 3.3 Graceful Degradation with Missing Features

**Requirement**: A missing Material or Process should produce a degraded-but-reasonable prediction, not a crash or a wildly different number.

**Why**: The API must accept incomplete quotes (15/510 training rows had missing material/process). Returning "error: missing field" is a worse UX than returning a wider-banded estimate.

**Test**: Predict with and without Material. Check: (1) no exception, (2) prediction within 3x of the with-material prediction.

### 3.4 Extrapolation Awareness

**Requirement**: Predictions for out-of-training-range inputs should be flagged, not silently returned.

**Why**: Tree models extrapolate flat (returning the nearest leaf), linear models extrapolate linearly (can go negative or explosive). Neither is reliable outside the training range.

**Test**: Predict at qty=200, qty=1000. Check that the API's OOD detection flags these.

**Note**: Extrapolation *behavior* (how bad the prediction is) is measured in the comparison framework's Lens 6. This validation plan only requires that extrapolation is *detected and flagged*.

---

## 4. Human-in-the-Loop Safety

These ensure the override workflow doesn't corrupt the system.

### 4.1 Override Storage Integrity

**Requirement**: Every override is stored with: original model price, human price, structured reason category, free-text reason, estimator ID, timestamp.

**Test**: Create a quote via API, override it, retrieve it. Verify all fields are present and correct.

### 4.2 Override Isolation

**Requirement**: Overrides are stored in a separate table from original quotes. An override does not modify the original model prediction — both are preserved.

**Test**: After override, GET /quote/{id} returns both `original_estimate` and `human_override`.

### 4.3 Delta Tracking

**Requirement**: The override response includes the delta (human_price - model_price) in both dollars and percentage.

**Test**: Override with a known price. Verify delta calculation.

---

## 5. Implementation: Where Each Check Lives

| Validation | Location | When |
|---|---|---|
| Domain invariants (§1) | `tests/test_comparison.py` (boundary suite) | CI on every commit |
| Domain invariants (§1) | `scripts/compare.py` → Lens 3 (economic coherence) | After training |
| Statistical properties (§2) | `scripts/compare.py` → Lenses 1, 2, 4, 5 | After training |
| Boundary safety (§3) | `tests/test_comparison.py` (boundary suite) | CI on every commit |
| Boundary safety (§3) | `scripts/compare.py` → Lens 6 (boundary behavior) | After training |
| Human-in-the-loop (§4) | `tests/test_api.py` | CI on every commit |

---

## 6. Decision Framework

After running both `evaluate.py` (MAPE-focused) and `compare.py` (7-lens):

1. **If MAPE-best model also ranks highest on the scorecard**: Ship it. Strong evidence.
2. **If MAPE-best model has economic coherence violations**: Investigate. Likely overfitting to data patterns that violate domain knowledge. Consider the next-best model without violations.
3. **If MAPE-best model is unstable (high fold variance)**: Prefer a more stable model within ~1-2% MAPE. The assessors will re-run your code — instability means their results may differ from yours.
4. **If MAPE-best model is aggressive (systematically underquotes)**: Flag this in the notebook. The shop might prefer a slightly less accurate but conservative model that protects margin.
5. **If multiple models are within ~1% MAPE**: Use the scorecard to break the tie. Prefer stability, then interpretability, then economic coherence.

This framework ensures we don't just chase MAPE — we ship a model that the shop floor can trust.
