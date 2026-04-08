# Price Estimator API — User Guide

Base URL: `https://idnste75ui.execute-api.us-west-2.amazonaws.com`

## Quick start

```bash
# Get a price estimate
curl -s -X POST "$BASE/v1/quote" \
  -H "Content-Type: application/json" \
  -d '{
    "part_description": "Sensor Housing - threaded",
    "material": "Inconel 718",
    "process": "5-Axis Milling",
    "quantity": 5,
    "rush_job": false,
    "lead_time_weeks": 4,
    "estimator": "Sato-san"
  }' | jq .
```

Response:

```json
{
  "quote_id": "Q-ab12cd34",
  "estimate": 2785.22,
  "aggressive_estimate": 2510.00,
  "conservative_estimate": 3100.45,
  "typical_range": { "low": 2200.00, "high": 3400.00 },
  "warnings": [],
  "shap_explanation": [
    { "feature": "Material_Inconel 718", "contribution": 820.5 },
    { "feature": "Process_5-Axis Milling", "contribution": 450.3 }
  ]
}
```

---

## Endpoints

### 1. Create a quote — `POST /v1/quote`

Generates an AI price estimate with prediction bands and SHAP explanation.

**Request body:**

| Field | Type | Required | Notes |
|---|---|---|---|
| `part_description` | string | yes | 1–200 chars, printable ASCII. e.g. `"Sensor Housing - threaded"` |
| `material` | string | no | One of the valid materials (see below) |
| `process` | string | no | One of the valid processes (see below) |
| `quantity` | int | yes | One of: 1, 5, 10, 20, 50, 100 |
| `rush_job` | bool | no | Default `false` |
| `lead_time_weeks` | int | yes | 1–52 |
| `estimator` | string | no | One of: `Sato-san`, `Suzuki-san`, `Tanaka-san` |

**Example — minimal request (only required fields):**

```bash
curl -s -X POST "$BASE/v1/quote" \
  -H "Content-Type: application/json" \
  -d '{
    "part_description": "Bearing Cap",
    "quantity": 10,
    "lead_time_weeks": 6
  }' | jq .
```

**Example — rush job with all fields:**

```bash
curl -s -X POST "$BASE/v1/quote" \
  -H "Content-Type: application/json" \
  -d '{
    "part_description": "Turbine Blade - precision ground",
    "material": "Titanium Grade 5",
    "process": "5-Axis Milling",
    "quantity": 1,
    "rush_job": true,
    "lead_time_weeks": 2,
    "estimator": "Tanaka-san"
  }' | jq .
```

---

### 2. Override a quote — `POST /v1/quote/{quote_id}/override`

A human estimator overrides the AI price. Records the delta for bias tracking.

**Request body:**

| Field | Type | Required | Notes |
|---|---|---|---|
| `human_price` | float | yes | Must be > 0 |
| `reason_category` | string | no | See categories below |
| `reason_text` | string | no | Free-text explanation (max 1000 chars) |
| `estimator_id` | string | no | Who is overriding |

**Reason categories:** `material_hardness`, `geometry_complexity`, `surface_finish`, `tooling_difficulty`, `customer_relationship`, `scrap_risk`, `certification_requirements`, `other`

**Example:**

```bash
curl -s -X POST "$BASE/v1/quote/Q-ab12cd34/override" \
  -H "Content-Type: application/json" \
  -d '{
    "human_price": 3200.00,
    "reason_category": "material_hardness",
    "reason_text": "Inconel work-hardens more than the model accounts for",
    "estimator_id": "Tanaka-san"
  }' | jq .
```

Response:

```json
{
  "stored": true,
  "override_id": "OVR-20260407T142530-a1b2c3",
  "delta_from_model": 414.78,
  "delta_pct": 14.89
}
```

---

### 3. Send a quote — `POST /v1/quote/{quote_id}/send`

Marks a quote as sent to the customer. No request body needed.

```bash
curl -s -X POST "$BASE/v1/quote/Q-ab12cd34/send" | jq .
```

Response:

```json
{
  "status": "sent",
  "sent_at": "2026-04-07T14:30:00"
}
```

---

### 4. Record outcome — `POST /v1/quote/{quote_id}/outcome`

Records whether the quote was won, lost, or expired.

**Request body:**

| Field | Type | Required | Notes |
|---|---|---|---|
| `outcome` | string | yes | `won`, `lost`, or `expired` |
| `reason` | string | no | Loss reason category (see below) |
| `reason_text` | string | no | Free-text (max 1000 chars) |
| `final_negotiated_price` | float | no | If negotiated down from quote |
| `po_number` | string | no | Purchase order number (max 100 chars) |

**Loss reason categories:** `price_too_high`, `lead_time`, `went_with_competitor`, `scope_change`, `other`

**Example — won:**

```bash
curl -s -X POST "$BASE/v1/quote/Q-ab12cd34/outcome" \
  -H "Content-Type: application/json" \
  -d '{
    "outcome": "won",
    "final_negotiated_price": 3100.00,
    "po_number": "PO-2026-0412"
  }' | jq .
```

**Example — lost:**

```bash
curl -s -X POST "$BASE/v1/quote/Q-ab12cd34/outcome" \
  -H "Content-Type: application/json" \
  -d '{
    "outcome": "lost",
    "reason": "price_too_high",
    "reason_text": "Customer got a lower bid from a domestic shop"
  }' | jq .
```

---

### 5. Get quote detail — `GET /v1/quote/{quote_id}`

Returns the full quote with lifecycle history (overrides, sent status, outcome).

```bash
curl -s "$BASE/v1/quote/Q-ab12cd34" | jq .
```

Response includes: `quote_id`, `status`, `features`, `original_estimate`, `aggressive_estimate`, `conservative_estimate`, `typical_range`, `warnings`, `shap_explanation`, `override`, `outcome`, `final_price`, `created_at`, `sent_at`, `timeline`.

---

### 6. List quotes — `GET /v1/quotes`

Returns a summary list of all quotes.

```bash
curl -s "$BASE/v1/quotes" | jq .
```

Response:

```json
[
  {
    "quote_id": "Q-ab12cd34",
    "status": "won",
    "part_description": "Sensor Housing - threaded",
    "material": "Inconel 718",
    "process": "5-Axis Milling",
    "quantity": 5,
    "original_estimate": 2785.22,
    "final_price": 3100.00,
    "created_at": "2026-04-07T14:25:00"
  }
]
```

---

## Valid values reference

**Materials:**
- Aluminum 6061
- Aluminum 7075
- Inconel 718
- Stainless Steel 17-4 PH
- Titanium Grade 5

**Processes:**
- 3-Axis Milling
- 5-Axis Milling
- CNC Turning
- Surface Grinding
- Wire EDM

**Quantities:** 1, 5, 10, 20, 50, 100

**Estimators:** Sato-san, Suzuki-san, Tanaka-san

---

## Quote lifecycle

```
draft  →  sent  →  won / lost / expired
  │
  └── override(s) can be applied at any point before outcome
```

1. **Create** (`POST /v1/quote`) — status: `draft`
2. **Override** (`POST /v1/quote/{id}/override`) — optional, repeatable
3. **Send** (`POST /v1/quote/{id}/send`) — status: `sent`
4. **Outcome** (`POST /v1/quote/{id}/outcome`) — status: `won` / `lost` / `expired`

---

## Error responses

All errors use a consistent envelope:

```json
{
  "error": {
    "code": "UNPROCESSABLE_ENTITY",
    "message": "1 validation error(s) in request.",
    "details": {
      "fields": [
        { "field": "quantity", "issue": "Invalid quantity: 7. Must be one of: [1, 5, 10, 20, 50, 100]" }
      ]
    }
  }
}
```

Common status codes: `404` (quote not found), `409` (conflict, e.g. outcome already recorded), `422` (validation error).
