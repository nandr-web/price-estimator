# TODO — Future Work

## Learning Loop (Post-Prototype)

Training data will live in DynamoDB. The dataset grows from three feedback types:

| Feedback Type | What it captures | Signal |
|---|---|---|
| **Override** | Estimator changed the model's price | Direct correction — strongest signal |
| **Accepted** | Quote sent to customer at model price, job won | Model price was competitive and profitable |
| **Rejected** | Quote sent, customer declined | Model price may have been too high (or non-price reasons) |

### Design Considerations

- Each record tagged with feedback type so the retraining pipeline can weight them differently (overrides > accepted > rejected)
- Accepted quotes are positive reinforcement — the model was right. These stabilize the model during retraining.
- Rejected quotes are ambiguous — rejection could be price, lead time, relationship, or scope. Consider a structured "rejection reason" field similar to override reasons.
- Minimum sample threshold: 30-50 new feedback records before triggering a retrain cycle
- Holdout validation required: retrained model must improve on a held-out set vs current model before deployment
- Track override rate and magnitude over time — decreasing overrides = model improving
- Store model version with each prediction so we can attribute feedback to the correct model generation

## UX and Deployment Considerations

After model selection is complete, address practical deployment concerns:
- What does the shop floor experience look like? (web form, spreadsheet plugin, etc.)
- How does the SHAP explanation render for the estimator? (bar chart, text breakdown, etc.)
- Wireframe or mockup of the estimator's workflow
- How does the override flow feel in practice?
- Assessors value "boring automation that saves 40 hours" — the narrative should reflect adoption/usability, not just accuracy
