## Deliverable

A quoting tool built as follows:
- An ensemble of models to generate price estimates with uncertainty ranges and explainability
- An API that serves predictions, handles human overrides, and tracks quote outcomes
- A web UI where estimators review AI prices, adjust them with structured reasons, and send quotes

---

## What It Does
### UX
An estimator opens the quoting tool, enters the part details, and gets:

- **A price estimate** with aggressive and conservative bounds — so they can choose their pricing strategy depending on whether they want to win the job or protect margin
- **"Why This Price"** — a chart showing which features drove the estimate (e.g. Inconel pushed it up, high quantity pulled it down)
- **Warnings when relevant** — missing fields, unusual inputs, high model disagreement, or parts the system hasn't seen before

The estimator reviews the AI price, optionally overrides it with a structured reason (material hardness, geometry complexity, scrap risk, etc.), then sends the quote. Outcomes (won/lost/expired) are tracked to close the feedback loop.

---

## How Reliable

We designed around three tenets:

1. **The conservative recommendation should not underquote.** When an estimator chooses the safe price to protect margin, the system must err on the side of pricing higher, not lower. On historical data, the conservative recommendation underquotes only 12% of jobs. Further optimization may be possible, but the remaining misses are genuine outliers — jobs with missing data that no model could price correctly.

2. **The system should know what it doesn't know.** When a part type, material, or quantity falls outside the training distribution, the system flags it for human review rather than guessing silently. A confident wrong answer is worse than an honest "please review."

3. **Economic rules are non-negotiable.** Every model in the consensus must pass basic sanity checks: Inconel costs more than Aluminum, larger quantities cost less per unit, rush jobs cost more. Models that got the average error right but violated these basics were rejected.

| Metric | Value | What it means |
|---|---|---|
| Average error | ~11% | For a $10,000 job, the estimate is typically within +/-$1,100 |
| Typical range coverage | 84% (target: 80%) | The quoted range captures the actual price 84% of the time — slightly conservative, which is the right direction |
| Economic coherence | 100% pass rate | All models in the consensus respect material ordering, volume discounts, and rush premiums |
*Given more time and data I'd work on optimizing this further.*

---

## How We Extract Features

The system takes in the following inputs and engineers features from them:

| Input | Extracted features |
|---|---|
| Part description (e.g. "Sensor Housing - threaded") | Base part type (10 known, fuzzy-matched for typos), complexity modifiers (9 known, each weighted), complexity score |
| Material | Material cost tier: 1 (Aluminum 6061) to 5 (Inconel 718); flag if missing |
| Process | Process precision tier: 1 (Surface Grinding) to 5 (5-Axis Milling); flag if missing |
| Quantity | Log-scaled quantity (captures diminishing returns on volume) |
| Rush job | Binary flag |
| Lead time | Included but found to have no effect on price — the shop charges for complexity, not schedule |

When a part description doesn't match anything in the registry, the system flags it as out-of-distribution and recommends human review.

---

## How We Pick the Models

We trained multiple model families — from a lookup table to regularized linear models to gradient-boosted trees. Accuracy alone isn't enough. We evaluated across seven dimensions:

1. **Error profile** — not just average error, but tail risk (how bad are the worst misses?)
2. **Segment fairness** — does accuracy hold across materials, part types, and estimators?
3. **Economic coherence** — does Inconel cost more than Aluminum? Does quantity 100 cost less per unit than quantity 1?
4. **Calibration** — is the model systematically over- or under-estimating?
5. **Stability** — how consistent across cross-validation folds?
6. **Boundary behavior** — what happens at edge cases?
7. **Interpretability** — can we explain the prediction?

The final consensus is a median across the top-performing models. It includes models that focus purely on job characteristics alongside models that account for historical estimator patterns — so the consensus reflects the job, not any one person's pricing habits.

---

## How It Handles Edge Cases

| Scenario | Behavior |
|---|---|
| Missing material or process | Flagged with a warning; prediction still generated using available features |
| Unknown part description | Fuzzy match attempted; falls back to "unknown" with a review recommendation |
| Unusual quantity or lead time | Out-of-distribution warning; estimate generated but flagged |
| Completely new product category | Warning shown; system recommends human pricing. With enough new quotes, the registry and models can be retrained |

---

## What We Found in the Data

- **Estimator bias is measurable.** Sato-san quotes ~7% below the group average (aggressive). Tanaka-san quotes ~9% above (conservative). The system controls for this so the AI price reflects the job, not the estimator's tendencies.
- **Pricing is multiplicative, not additive.** Material premiums and volume discounts compound — an Inconel part at qty=1 isn't just "more expensive," it scales differently.
- **Part descriptions encode real cost signals.** "Sensor Housing - complex internal channels" costs meaningfully more than "Sensor Housing - standard." We identified 10 base part types and 9 complexity modifiers.
- **Lead time doesn't affect price.** Removing it from the model made no difference.
- **~6% of rows are missing material or process.** Not random — correlated with specific estimators and part types. A data quality issue worth fixing at the source.

---

## What's Included

**Web UI** — quotes list, quote detail with estimate range and explainability chart, review and override workflow, outcome tracking.

- https://d1uq9qw97we8gx.cloudfront.net/

**API** — six endpoints covering the full quote lifecycle:

| Endpoint | Purpose |
|---|---|
| `POST /v1/quote` | Get an AI price estimate |
| `GET /v1/quote/{id}` | Retrieve quote with full history |
| `GET /v1/quotes` | List all quotes |
| `POST /v1/quote/{id}/override` | Override with a human price and reason |
| `POST /v1/quote/{id}/send` | Mark as sent to customer |
| `POST /v1/quote/{id}/outcome` | Record won/lost/expired |

**Infrastructure** — AWS CDK stack (Lambda, DynamoDB, S3, CloudFront, API Gateway, scheduled warmup). Deployed with `bash scripts/deploy.sh`.

**Pipeline** — EDA, training, evaluation, comparison, and bias analysis. Reproducible from a single chained command. 209 automated tests at 76% coverage, including domain invariants (material ordering, volume discounts) that break the build if violated.

The system runs three ways: locally, in AWS Lambda, or through the web UI. Same codebase, same models.

---

## Decisions and Tradeoffs

- **Human-in-the-loop by design.** The AI generates the starting point. The estimator has full authority to override, with structured reason tracking so we learn from disagreements.
- **Override reasons tell you what data to collect next.** If 40% of overrides cite "surface finish," that's exactly which variable to add to the input form.
- **Multiple models, not one.** No single model wins on every dimension. The ensemble covers more ground — when its members disagree on a specific job, that disagreement itself is a useful signal.

---

## Limitations

- **510 rows is small.** The models perform well within the training distribution, but confidence on rare combinations (Inconel + Wire EDM + qty 1) is limited. More data would let us build per-segment prediction bands instead of global ones.
- **Part description parsing is rule-based.** The 10-type registry covers the training data, but truly novel parts need human classification.
- **One $115K outlier job (10x the median) with missing material data.** The system would flag this for human review due to the missing field and unusual price range. Jobs like this need richer input data to price reliably.

---

## Given More Time

- **Automated retraining with drift detection.** The infrastructure skeleton is in place (scheduled retrain Lambda, S3 model versioning). Wire it to trigger when prediction-vs-outcome error exceeds a threshold.
- **Automated feature extraction from free text.** For novel part descriptions that don't match the registry, attempt to identify structured features automatically — using pattern matching, an LLM, or a combination — with human confirmation before adding to the training set.
- **Win/loss feedback loop.** Outcome tracking is built into the API and UI. Use it to retrain periodically on quotes that were actually won or lost, weighted by recency.

---

*Full API reference with curl examples: [docs/api-guide.md](api-guide.md)*
