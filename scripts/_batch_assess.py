"""Batch assessment of recommender quality across all 510 rows.

Compares Estimate, Win bid, Protect margin against typical range and actuals.
"""

import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from price_estimator.data import load_data
from price_estimator.predict import (
    apply_prediction_band,
    compute_recommendation,
    load_prediction_bands,
)

df = load_data("resources/aora_historical_quotes.csv")
models = {}
for p in sorted(Path("outputs/models").glob("*.joblib")):
    models[p.stem] = joblib.load(p)

bands = load_prediction_bands("outputs/results/prediction_bands.json")
available = [m for m in bands if m in models]
best_band_model = min(available, key=lambda m: bands[m]["median_abs_error_pct"])
best_band = bands[best_band_model]

print(f"Band model: {best_band_model}")
print(
    f"Band: lower_pct={best_band['lower_pct']:.1f}%, "
    f"upper_pct={best_band['upper_pct']:.1f}%, "
    f"coverage={best_band['coverage']:.0%}"
)

# Collect per-row data
rows_data = []

for i in range(len(df)):
    row = df.iloc[[i]].reset_index(drop=True)
    actual = float(row["TotalPrice_USD"].iloc[0])

    preds = {}
    for name, model in models.items():
        try:
            pred = model.predict(row)
            preds[name] = float(pred[0])
        except Exception:
            pass

    rec = compute_recommendation(preds, band=best_band)
    estimate = rec["estimate"]
    win_bid = rec["win_bid"]
    protect_margin = rec["protect_margin"]

    # Typical range from best band model prediction
    if best_band_model in preds:
        band_pred = preds[best_band_model]
    else:
        band_pred = estimate
    typical_low, typical_high = apply_prediction_band(band_pred, best_band)

    rows_data.append(
        {
            "actual": actual,
            "estimate": estimate,
            "win_bid": win_bid,
            "protect_margin": protect_margin,
            "typical_low": typical_low,
            "typical_high": typical_high,
            "div": rec.get("family_divergence"),
            "material": row["Material"].iloc[0],
            "quote_id": row["QuoteID"].iloc[0],
            "idx": i,
        }
    )

actuals = np.array([r["actual"] for r in rows_data])
estimates = np.array([r["estimate"] for r in rows_data])
win_bids = np.array([r["win_bid"] for r in rows_data])
protect_margins = np.array([r["protect_margin"] for r in rows_data])
typical_lows = np.array([r["typical_low"] for r in rows_data])
typical_highs = np.array([r["typical_high"] for r in rows_data])

n = len(df)

# --- Core error stats ---
est_errors = (estimates - actuals) / actuals * 100
abs_errors = np.abs(est_errors)

print("\n" + "=" * 70)
print("RECOMMENDER QUALITY ASSESSMENT (all 510 rows)")
print("=" * 70)

print("\nEstimate error:")
print(f"  MAPE:     {np.mean(abs_errors):.1f}%")
print(f"  Median:   {np.median(abs_errors):.1f}%")
print(f"  Mean signed: {np.mean(est_errors):+.1f}%")
print(f"  P90:      {np.percentile(abs_errors, 90):.1f}%")

# --- Win bid / Protect margin coverage ---
print(f"\n{'=' * 70}")
print("WIN BID vs PROTECT MARGIN vs ACTUAL")
print(f"{'=' * 70}")

# How often is actual between win_bid and protect_margin?
in_bid_margin = np.sum((actuals >= win_bids) & (actuals <= protect_margins))
print(
    f"\nActual within [Win bid, Protect margin]: "
    f"{in_bid_margin}/{n} ({in_bid_margin / n * 100:.1f}%)"
)

# How often is actual within typical range?
in_typical = np.sum((actuals >= typical_lows) & (actuals <= typical_highs))
print(f"Actual within typical range (80%):       {in_typical}/{n} ({in_typical / n * 100:.1f}%)")

# Win bid vs actual
wb_errors = (win_bids - actuals) / actuals * 100
pm_errors = (protect_margins - actuals) / actuals * 100

print("\nWin bid error vs actual:")
print(f"  Mean signed: {np.mean(wb_errors):+.1f}%")
print(f"  Median:      {np.median(wb_errors):+.1f}%")
print(
    f"  Win bid > actual (overpriced):    {np.sum(win_bids > actuals)}/{n} "
    f"({np.sum(win_bids > actuals) / n * 100:.0f}%)"
)
print(
    f"  Win bid < actual (competitive):   {np.sum(win_bids < actuals)}/{n} "
    f"({np.sum(win_bids < actuals) / n * 100:.0f}%)"
)

print("\nProtect margin error vs actual:")
print(f"  Mean signed: {np.mean(pm_errors):+.1f}%")
print(f"  Median:      {np.median(pm_errors):+.1f}%")
print(
    f"  Protect margin > actual (safe):   {np.sum(protect_margins > actuals)}/{n} "
    f"({np.sum(protect_margins > actuals) / n * 100:.0f}%)"
)
print(
    f"  Protect margin < actual (exposed): {np.sum(protect_margins < actuals)}/{n} "
    f"({np.sum(protect_margins < actuals) / n * 100:.0f}%)"
)

# --- Spread analysis ---
print(f"\n{'=' * 70}")
print("SPREAD ANALYSIS")
print(f"{'=' * 70}")

wb_pm_spread = (protect_margins - win_bids) / estimates * 100
typical_spread = (typical_highs - typical_lows) / estimates * 100

print("\nWin bid — Protect margin spread (as % of estimate):")
print(f"  Mean:   {np.mean(wb_pm_spread):.1f}%")
print(f"  Median: {np.median(wb_pm_spread):.1f}%")

print("\nTypical range spread (as % of estimate):")
print(f"  Mean:   {np.mean(typical_spread):.1f}%")
print(f"  Median: {np.median(typical_spread):.1f}%")

print(
    f"\nRatio: typical range is {np.mean(typical_spread) / np.mean(wb_pm_spread):.1f}x "
    f"wider than bid-margin range"
)

# --- Directional check: where does actual land relative to the 3 values? ---
print(f"\n{'=' * 70}")
print("WHERE DOES ACTUAL LAND?")
print(f"{'=' * 70}")

below_wb = np.sum(actuals < win_bids)
between_wb_est = np.sum((actuals >= win_bids) & (actuals < estimates))
between_est_pm = np.sum((actuals >= estimates) & (actuals < protect_margins))
above_pm = np.sum(actuals >= protect_margins)

print(f"  Below win bid:                {below_wb:4d} ({below_wb / n * 100:5.1f}%)")
print(f"  Between win bid & estimate:   {between_wb_est:4d} ({between_wb_est / n * 100:5.1f}%)")
print(f"  Between estimate & protect:   {between_est_pm:4d} ({between_est_pm / n * 100:5.1f}%)")
print(f"  Above protect margin:         {above_pm:4d} ({above_pm / n * 100:5.1f}%)")

# --- By price quartile ---
print(f"\n{'=' * 70}")
print("BY PRICE QUARTILE")
print(f"{'=' * 70}")

q1, q2, q3 = np.percentile(df["TotalPrice_USD"], [25, 50, 75])
for label, lo, hi in [
    (f"Q1 (<${q1:,.0f})", 0, q1),
    (f"Q2 (${q1:,.0f}-${q2:,.0f})", q1, q2),
    (f"Q3 (${q2:,.0f}-${q3:,.0f})", q2, q3),
    (f"Q4 (>${q3:,.0f})", q3, 1e9),
]:
    mask = (df["TotalPrice_USD"] >= lo) & (df["TotalPrice_USD"] < hi)
    if mask.sum() == 0:
        continue
    idx = mask.values
    q_actual = actuals[idx]
    q_est = estimates[idx]
    q_wb = win_bids[idx]
    q_pm = protect_margins[idx]
    q_tl = typical_lows[idx]
    q_th = typical_highs[idx]

    q_in_bm = np.sum((q_actual >= q_wb) & (q_actual <= q_pm))
    q_in_typ = np.sum((q_actual >= q_tl) & (q_actual <= q_th))
    q_n = mask.sum()
    q_mape = np.mean(np.abs((q_est - q_actual) / q_actual * 100))
    q_signed = np.mean((q_est - q_actual) / q_actual * 100)

    print(f"\n  {label}  (n={q_n})")
    print(f"    MAPE={q_mape:.1f}%  signed={q_signed:+.1f}%")
    print(f"    Actual in [WB,PM]: {q_in_bm}/{q_n} ({q_in_bm / q_n * 100:.0f}%)")
    print(f"    Actual in typical: {q_in_typ}/{q_n} ({q_in_typ / q_n * 100:.0f}%)")

# --- Cases where actual is outside typical range ---
outside_typical = ~((actuals >= typical_lows) & (actuals <= typical_highs))
n_outside = outside_typical.sum()
print(f"\n{'=' * 70}")
print(f"ROWS WHERE ACTUAL IS OUTSIDE TYPICAL RANGE ({n_outside}/{n})")
print(f"{'=' * 70}")

if n_outside > 0:
    outside_above = actuals > typical_highs
    outside_below = actuals < typical_lows
    print(f"  Actual > typical high (underquoted): {outside_above.sum()}")
    print(f"  Actual < typical low (overquoted):   {outside_below.sum()}")

    # Worst cases
    miss_pcts = []
    for r in rows_data:
        if r["actual"] > r["typical_high"]:
            miss_pcts.append(
                (
                    r["idx"],
                    r["quote_id"],
                    r["actual"],
                    r["estimate"],
                    r["typical_high"],
                    (r["actual"] - r["typical_high"]) / r["actual"] * 100,
                    "under",
                )
            )
        elif r["actual"] < r["typical_low"]:
            miss_pcts.append(
                (
                    r["idx"],
                    r["quote_id"],
                    r["actual"],
                    r["estimate"],
                    r["typical_low"],
                    (r["typical_low"] - r["actual"]) / r["actual"] * 100,
                    "over",
                )
            )

    miss_pcts.sort(key=lambda x: -x[5])
    print("\n  Top 10 worst outside-range misses:")
    print(
        f"  {'Row':>4s}  {'QuoteID':>8s}  {'Actual':>12s}  {'Estimate':>12s}  "
        f"{'RangeEdge':>12s}  {'Gap%':>6s}  Dir"
    )
    for row_idx, qid, actual, est, edge, gap, direction in miss_pcts[:10]:
        print(
            f"  {row_idx:4d}  {qid:>8s}  ${actual:>10,.2f}  ${est:>10,.2f}  "
            f"${edge:>10,.2f}  {gap:5.1f}%  {direction}"
        )
