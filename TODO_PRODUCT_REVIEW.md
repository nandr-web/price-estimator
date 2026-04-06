# Product Goals & Tenet Alignment — TODO Tracker

**Source:** REVIEW.md Section 3 (Session 4 review)
**Status:** Initial review completed. Re-run after notebook creation and pipeline regeneration.

---

## Context

This review checks whether the implementation serves the 5 assessment deliverables and adheres to our stated tenets:

1. **CLI-first development** — scripts produce results; notebooks visualize
2. **Multiple competing models** — EDA findings are hypotheses, not baked-in assumptions
3. **Pragmatism** — M0 contextualizes ML lift; boring automation > complex AI
4. **Evaluate both approaches** — when two strategies are valid, test both
5. **Rich notebook deliverable** — primary artifact assessors will read
6. **Validation-first** — define good model properties before evaluating
7. **No data leakage** — imputation inside CV folds only
8. **Preliminary findings are hypotheses** — validate before building in

### Deliverable Status (at review time)

| # | Deliverable | Status |
|---|---|---|
| 1 | Price prediction model | Implemented (14 models, M2 wins 10.8% MAPE) |
| 2 | Feature extraction from PartDescription | Implemented (Tier 1 parser, Tiers 2-3 described) |
| 3 | Estimator bias analysis | Implemented (M9 residuals, bootstrap CIs, breakdowns) |
| 4 | Human-in-the-loop API | Implemented (3 endpoints, SQLite, structured overrides) |
| 5 | Missing variables discussion | Described in PLAN.md only — not a standalone artifact |

---

## Open Items

### CRITICAL

#### P1: Notebook not created
- **File:** `notebooks/analysis.ipynb` (empty directory)
- **Why it matters:** The notebook is the primary artifact assessors will read. Without it, the story isn't told. Assessors may not run CLI scripts.
- **Action:** Create analysis.ipynb per the 10-section outline in PLAN.md §Deliverable Structure.
- **Re-check trigger:** After notebook creation, verify all 10 sections are populated and load pre-computed results from `outputs/`.

### MEDIUM

#### P3: Hypothesis resolution CSV is stale
- **File:** `outputs/results/hypothesis_resolution.csv`
- **What happened:** `resolve_hypotheses()` in evaluate.py was updated to compare M7 vs M7c (log-target fair comparison), but the saved CSV still has the old M6 vs M7b comparison (raw-target, uninformative since both perform ~38-47% MAPE).
- **Action:** Re-run `python scripts/evaluate.py --results outputs/results/cv_fold_results.csv --output outputs/`
- **Re-check trigger:** After any code change to evaluate.py or model retraining, verify artifacts match code.

#### P8: Silent estimator default biases API predictions
- **File:** `src/price_estimator/api.py:174`
- **What happens:** `_request_to_dataframe()` defaults `None` estimator to `"Sato-san"`. Sato is the aggressive estimator (-10.4% bias), so the API systematically returns lower estimates when callers don't specify an estimator. This is undocumented.
- **Options:**
  1. Return median prediction across all three estimator values (most neutral)
  2. Add `"Estimator defaulted to Sato-san"` to confidence_flags when estimator is None
  3. Make estimator required
- **Action:** At minimum, add a confidence flag when defaulting. Preferred: option 1 or 2.
- **Re-check trigger:** After API changes, verify the default behavior is documented in the response.

#### P9: SHAP feature names not human-readable
- **File:** `src/price_estimator/api.py:228-233`
- **What happens:** SHAP returns raw label-encoded feature names: `material` → integer code (e.g., 2), `base_part_type` → integer code (e.g., 7). An estimator reading the API response sees `{"feature": "material", "contribution": 2800.0}` — meaningless without a reverse lookup.
- **Action:** Post-process SHAP results to map integer-coded categoricals back to original names. The feature matrix construction has the mappings.
- **Re-check trigger:** After any SHAP or feature engineering changes.

### LOW

#### ~~P5: CLAUDE.md pipeline missing compare.py~~
- **FIXED** — Added compare.py to the pipeline chain.

#### ~~P10: SHAP model selection priority wrong~~
- **FIXED** — Updated priority to M2 > M7 > M7c > M6 > M5.

#### ~~P14: papermill mismatch~~
- **FIXED** — Removed papermill reference from TOOLING.md (we're not using it).

#### P4: M0 not called out prominently in stdout
- **File:** `scripts/evaluate.py`
- **What happens:** M0 appears in the sorted comparison table but isn't highlighted as a contextualizing anchor. Per our pragmatism tenet, the spreadsheet-vs-ML tradeoff should be explicit.
- **Action:** Add a 1-2 line call-out after the comparison table: "M0 (lookup table): X% MAPE — ML best (Y): Z% MAPE — W% relative improvement."
- **Re-check trigger:** After notebook creation — verify the notebook also highlights this.

#### P7: Hypothesis lines lack business interpretation
- **File:** `scripts/evaluate.py` stdout
- **What happens:** Prints `"Is pricing multiplicative? → M2 wins"` but not the interpretive conclusion. The assessor has to mentally map model names to business meaning.
- **Action:** Add a brief suffix, e.g.: `"→ Yes, log-linear model confirms multiplicative pricing structure."`
- **Re-check trigger:** After notebook creation — the notebook should carry the full narrative regardless.

#### P2: Missing variables not a standalone artifact
- **File:** PLAN.md §7 only
- **What happens:** Task 5 (missing variable identification) has no script output, no JSON, and is invisible to someone who doesn't read PLAN.md.
- **Action:** Ensure notebook Section 8 surfaces this prominently. Optionally emit `outputs/results/missing_variables.json`.
- **Re-check trigger:** After notebook creation.

---

## Passed Items (no action needed)

| ID | Item | Status |
|---|---|---|
| P6 | CLI-first tenet | PASS — full pipeline runs as single shell chain, no manual steps |
| P11 | Override reason categories | PASS — 8 categories well-chosen for aerospace machining |
| P12 | OOD detection coverage | PASS — covers VALIDATION_PLAN.md §3 requirements |
| P15 | Optuna tuning not implemented | ACCEPTABLE — M2 wins on simplicity; defensible without tuning |
| P16 | Prediction bands partially implemented | ACCEPTABLE — model disagreement implemented; others are enhancements |
| P18 | requirements.txt | PASS — exists, current, pinned versions |

---

## When to Re-Run This Review

Re-run Session 4 (Product Goals & Tenet Alignment) after:

1. **Notebook creation** — verify P1, P2, P4, P7 are resolved; check that all 10 sections match capabilities
2. **Pipeline regeneration** — verify P3 is resolved; check all artifacts are consistent with code
3. **API changes** — verify P8, P9 are resolved; check SHAP readability and estimator handling
4. **Before final submission** — full pass to verify all items are resolved or explicitly deferred
