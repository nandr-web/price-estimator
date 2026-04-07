# API Design — Price Estimator

## Overview

REST API for AI-assisted quoting of precision machined aerospace parts.
Three endpoints: create a quote, override a quote with human judgement,
retrieve a quote. All routes are prefixed with `/v1`.

This API is built for **review and demonstration**, not production traffic.
Rate limits are tight, there is no authentication, and the data is synthetic.

---

## Table of Contents

1. [Base URL and Versioning](#1-base-url-and-versioning)
2. [Rate Limiting](#2-rate-limiting)
3. [Authentication](#3-authentication)
4. [Common Response Envelope](#4-common-response-envelope)
5. [Error Handling](#5-error-handling)
6. [Endpoints](#6-endpoints)
7. [Input Validation and Sanitization](#7-input-validation-and-sanitization)
8. [Idempotency](#8-idempotency)
9. [CORS](#9-cors)
10. [Logging](#10-logging)
11. [OpenAPI Documentation](#11-openapi-documentation)

---

## 1. Base URL and Versioning

```
https://{api-gateway-id}.execute-api.{region}.amazonaws.com/v1
```

All routes live under `/v1`. If we make breaking changes to the response
schema, we introduce `/v2` as a new API Gateway stage while keeping `/v1`
alive. Non-breaking additions (new optional fields) do not require a
version bump.

**Local development:**

```
http://localhost:8000/v1
```

The FastAPI app mounts all routes under a `/v1` router prefix.

---

## 2. Rate Limiting

This API is intended for reviewers evaluating the system, not for
production quoting traffic. Rate limits are set to allow a reviewer
to use the API interactively and run an automated test suite, while
preventing abuse.

| Limit | Value | Enforced by |
|---|---|---|
| **Burst** | 50 requests | API Gateway throttling |
| **Sustained** | 20 requests/second | API Gateway throttling |
| **Daily** | 5,000 requests | API Gateway usage plan |

These limits are per-API (not per-client, since there is no auth).

**Rate limit responses:**

```
HTTP/1.1 429 Too Many Requests
Retry-After: 5

{
  "error": {
    "code": "RATE_LIMITED",
    "message": "Rate limit exceeded. Retry after 5 seconds."
  }
}
```

API Gateway returns 429 natively when throttling kicks in. We configure
the gateway response to match our error envelope.

**Why these numbers:** A typical automated test suite hitting all 3
endpoints with various inputs would make ~200-500 requests. A burst
of 50 allows parallel test execution. The 5,000 daily cap stops
runaway loops while leaving plenty of room for repeated test runs.

---

## 3. Authentication

**Decision: No authentication for now.**

Rationale:
- This is a review/demo deployment, not production.
- There is no real customer data — the dataset is 510 synthetic quotes.
- The Lambda IAM role is scoped to only its own DynamoDB tables and S3
  bucket — even if someone discovers the endpoint, the blast radius is
  limited to creating fake quote records.
- Rate limiting (Section 2) constrains abuse further.

**When to add auth:** If this moves toward production or handles real
pricing data, add a Cognito JWT authorizer at the API Gateway level.
The Lambda code requires zero changes — auth is handled entirely by
the gateway.

---

## 4. Common Response Envelope

All successful responses return the resource directly (REST convention).
No wrapper envelope — the HTTP status code and Content-Type header are
sufficient.

```
HTTP/1.1 200 OK
Content-Type: application/json

{
  "quote_id": "Q-API-a1b2c3d4",
  "estimate": 14573.68,
  ...
}
```

All error responses use a consistent envelope (Section 5).

---

## 5. Error Handling

Every error response uses this structure:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable description.",
    "details": {}
  }
}
```

`details` is optional and included only when there is structured
information (e.g., which fields failed validation).

### Error codes

| HTTP Status | Code | When |
|---|---|---|
| 400 | `INVALID_INPUT` | Request body fails validation (see Section 7) |
| 404 | `QUOTE_NOT_FOUND` | Quote ID does not exist |
| 422 | `UNPROCESSABLE_ENTITY` | Pydantic structural validation failure (missing required field, wrong type) |
| 429 | `RATE_LIMITED` | Rate limit exceeded |
| 500 | `PREDICTION_FAILED` | All models failed to produce a prediction |
| 503 | `SERVICE_UNAVAILABLE` | Models not loaded (cold start race or deployment error) |

### Validation error detail format (400)

```json
{
  "error": {
    "code": "INVALID_INPUT",
    "message": "1 validation error in request.",
    "details": {
      "fields": [
        {
          "field": "part_description",
          "issue": "Exceeds maximum length of 200 characters."
        }
      ]
    }
  }
}
```

### Distinction: 400 vs 422

- **422** — Pydantic can't parse the request at all (wrong type, missing
  required field). FastAPI returns this automatically.
- **400** — Pydantic parsed the request, but our business rules reject
  the values (e.g., unknown material, string too long). We raise these
  explicitly.

We override FastAPI's default 422 handler to match our error envelope
format, so both 400 and 422 responses look the same to clients.

---

## 6. Endpoints

### POST /v1/quote

Create a new price estimate.

**Request:**

```json
{
  "part_description": "Sensor Housing - threaded",
  "material": "Inconel 718",
  "process": "5-Axis Milling",
  "quantity": 5,
  "rush_job": false,
  "lead_time_weeks": 4,
  "estimator": "Sato-san"
}
```

| Field | Type | Required | Constraints |
|---|---|---|---|
| `part_description` | string | yes | 1-200 chars, stripped of leading/trailing whitespace |
| `material` | string \| null | no | Must be a known material if provided (see Section 7) |
| `process` | string \| null | no | Must be a known process if provided |
| `quantity` | integer | yes | One of: 1, 5, 10, 20, 50, 100 |
| `rush_job` | boolean | no | Defaults to false |
| `lead_time_weeks` | integer | yes | 1-52 inclusive |
| `estimator` | string \| null | no | Must be a known estimator if provided |

**Response (200):**

```json
{
  "quote_id": "Q-API-a1b2c3d4",
  "estimate": 14573.68,
  "aggressive_estimate": 14045.45,
  "conservative_estimate": 15226.37,
  "typical_range": {
    "low": 12333.81,
    "high": 17009.37,
    "coverage": 0.80
  },
  "warnings": [
    "No estimator provided — using debiased consensus"
  ],
  "shap_explanation": [
    {"feature": "log_quantity", "contribution": -2145.30},
    {"feature": "material_cost_tier", "contribution": 1832.10}
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `quote_id` | string | Unique identifier, format `Q-API-{hex8}` |
| `estimate` | float | Best estimate (median of top-tier model consensus) |
| `aggressive_estimate` | float \| null | Competitive bid price (40th percentile). Null if prediction bands unavailable. |
| `conservative_estimate` | float \| null | Safe margin price (60th percentile). Null if prediction bands unavailable. |
| `typical_range` | object \| null | Historical range at given coverage level. Null if bands unavailable. |
| `typical_range.low` | float | Lower bound of typical range |
| `typical_range.high` | float | Upper bound of typical range |
| `typical_range.coverage` | float | Coverage probability (0.80 = 80% of similar past jobs) |
| `warnings` | string[] | List of actionable warnings (empty if none) |
| `shap_explanation` | object[] \| null | Top features driving the estimate, sorted by abs contribution. Null if SHAP unavailable. |

**Warning categories:**

| Warning | Trigger |
|---|---|
| Missing material / process | Corresponding input field is null |
| Out-of-distribution | Quantity, lead time, material, or process outside training data range |
| Linear/tree divergence | Linear and tree model families disagree by >10% |
| Top-tier model spread | Top-tier models spread >15% |
| No estimator provided | Estimator field is null |

---

### POST /v1/quote/{quote_id}/override

Record a human price override for an existing quote. This does not
replace the original estimate — both are stored for audit and model
retraining purposes.

**Request:**

```json
{
  "human_price": 15500.00,
  "reason_category": "material_hardness",
  "reason_text": "Inconel work-hardens more than the model accounts for",
  "estimator_id": "Tanaka-san"
}
```

| Field | Type | Required | Constraints |
|---|---|---|---|
| `human_price` | float | yes | > 0, max 10,000,000 |
| `reason_category` | string \| null | no | Must be one of the enum values below |
| `reason_text` | string \| null | no | Max 1000 chars |
| `estimator_id` | string \| null | no | Must be a known estimator if provided |

**Reason categories (enum):**

| Value | Meaning |
|---|---|
| `material_hardness` | Material is harder to machine than model expects |
| `geometry_complexity` | Part geometry requires more setup/passes |
| `surface_finish` | Surface finish requirements add cost |
| `tooling_difficulty` | Special tooling required |
| `customer_relationship` | Pricing adjusted for customer context |
| `scrap_risk` | Higher scrap rate expected |
| `certification_requirements` | Additional certs/inspections needed |
| `other` | Freeform — `reason_text` should explain |

**Response (200):**

```json
{
  "override_id": "OVR-e5f6g7h8",
  "quote_id": "Q-API-a1b2c3d4",
  "stored": true,
  "delta_from_model": 926.32,
  "delta_pct": 6.35
}
```

| Field | Type | Description |
|---|---|---|
| `override_id` | string | Unique override identifier |
| `quote_id` | string | The quote that was overridden |
| `stored` | boolean | Always true on success |
| `delta_from_model` | float | human_price - model_estimate (positive = human priced higher) |
| `delta_pct` | float | Delta as percentage of model estimate |

**Errors:**

| Status | Code | When |
|---|---|---|
| 404 | `QUOTE_NOT_FOUND` | `quote_id` does not exist |

---

### GET /v1/quote/{quote_id}

Retrieve a quote with its original estimate and most recent override.

**Response (200):**

```json
{
  "quote_id": "Q-API-a1b2c3d4",
  "original_estimate": 14573.68,
  "final_price": 15500.00,
  "human_override": {
    "human_price": 15500.00,
    "reason_category": "material_hardness",
    "reason_text": "Inconel work-hardens more than the model accounts for",
    "estimator_id": "Tanaka-san",
    "delta_from_model": 926.32,
    "overridden_at": "2026-04-06T14:30:00"
  }
}
```

| Field | Type | Description |
|---|---|---|
| `quote_id` | string | The requested quote ID |
| `original_estimate` | float | The model's original estimate |
| `final_price` | float | The effective price (override if exists, else original) |
| `human_override` | object \| null | Most recent override, or null if never overridden |

**Errors:**

| Status | Code | When |
|---|---|---|
| 404 | `QUOTE_NOT_FOUND` | `quote_id` does not exist |

---

## 7. Input Validation and Sanitization

Validation happens in two layers:

### Layer 1: Pydantic (structural)

FastAPI + Pydantic handle type checking, required fields, and basic
constraints automatically. Failures return 422.

### Layer 2: Business rules (semantic)

Applied after Pydantic parsing, before model inference. Failures
return 400 with the `INVALID_INPUT` error code.

| Field | Rule | Rationale |
|---|---|---|
| `part_description` | Max 200 chars after strip | Prevents payload abuse. Longest known description is ~50 chars. |
| `part_description` | Must contain only printable ASCII + common punctuation | The PartDescription parser uses a fixed registry of English part names. Unicode, control chars, or emoji would never match. |
| `material` | If provided, must be one of the 5 known materials | Unknown materials produce unpredictable features. The OOD warning system already flags this, but we reject at the gate for known-invalid values to give a cleaner error. |
| `process` | If provided, must be one of the 5 known processes | Same rationale as material. |
| `estimator` | If provided, must be one of the 3 known estimators | Prevents label-encoding errors in tree models. |
| `quantity` | Must be one of: 1, 5, 10, 20, 50, 100 | Training data only contains these quantities. Arbitrary values would be out-of-distribution by definition. |
| `human_price` | Max 10,000,000 | Sanity cap. Highest quote in training data is ~$100K. |
| `reason_text` | Max 1000 chars after strip | Prevents payload abuse on freeform text. |

### Known valid values (for reference)

**Materials:** Aluminum 6061, Aluminum 7075, Inconel 718,
Stainless Steel 17-4 PH, Titanium Grade 5

**Processes:** 3-Axis Milling, 5-Axis Milling, CNC Turning,
Surface Grinding, Wire EDM

**Estimators:** Sato-san, Suzuki-san, Tanaka-san

**Quantities:** 1, 5, 10, 20, 50, 100

### What we intentionally do NOT validate

- **`part_description` against the part type registry.** The parser has
  fuzzy matching (rapidfuzz, threshold 85) specifically to handle typos
  and minor variations. Rejecting unknown descriptions would defeat that
  design. Instead, unrecognized descriptions produce an OOD warning in
  the response.

- **`material` and `process` when null.** Null is a valid signal —
  the feature matrix includes `missing_material` and `missing_process`
  indicator columns, and the models are trained on data with missing
  values. We warn but don't reject.

---

## 8. Idempotency

**POST /quote** is intentionally non-idempotent. Each call creates a
new quote with a unique `quote_id`, even if the inputs are identical.

Rationale: Every quote request is recorded for audit purposes.
If a reviewer or estimator asks for a quote on the same part twice,
we want both requests logged — the timestamps, any differences in
warnings, and the ability to override each independently are all
valuable for understanding estimator behavior over time.

**POST /quote/{id}/override** is also non-idempotent. Multiple overrides
on the same quote are allowed and recorded chronologically.
`GET /quote/{id}` returns the most recent override. Historical overrides
are preserved in the database for audit.

If clients need to avoid duplicate submissions (e.g., a UI with a submit
button), they should implement client-side deduplication (disable button
after click, check quote_id before re-submitting).

---

## 9. CORS

CORS is configured at the API Gateway level to support a future web UI.

```
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, POST, OPTIONS
Access-Control-Allow-Headers: Content-Type
Access-Control-Max-Age: 86400
```

**Why `Allow-Origin: *`:** There is no authentication, no cookies, and
no sensitive data. A wildcard origin is appropriate for a demo/review API.
If auth is added later, restrict this to the specific UI domain.

**Preflight caching:** `Max-Age: 86400` (24 hours) means the browser
caches the OPTIONS response and avoids a preflight on every request.

### Future UI sections (context for CORS scope)

The planned UI has three sections that will hit this API:

1. **Training management** — upload new training data, trigger retrain,
   view model metrics. (Hits a separate admin API, not these endpoints.)
2. **Quoting** — POST /v1/quote with a form, display estimate +
   aggressive/conservative + range + warnings + SHAP.
3. **Quote feedback** — POST /v1/quote/{id}/override with reason
   selection, GET /v1/quote/{id} to show current state, status tracking.

All three sections use the same CORS policy since they share a domain.

---

## 10. Logging

All request/response logging goes to CloudWatch Logs via two channels:

### API Gateway access logs

Enabled on the HTTP API stage. Log format (JSON):

```json
{
  "requestId": "$context.requestId",
  "ip": "$context.identity.sourceIp",
  "method": "$context.httpMethod",
  "path": "$context.path",
  "status": "$context.status",
  "latency": "$context.responseLatency",
  "integrationLatency": "$context.integrationLatency"
}
```

Log group: `/aws/apigateway/price-estimator`

These logs capture every request regardless of whether the Lambda
executes. Useful for monitoring rate limiting (429s), latency, and
traffic patterns.

### Lambda application logs

The Python `logging` module writes to CloudWatch Logs automatically
in Lambda. We configure structured JSON logging:

```json
{
  "timestamp": "2026-04-06T14:30:00.123Z",
  "level": "INFO",
  "logger": "price_estimator.api",
  "message": "Quote created",
  "quote_id": "Q-API-a1b2c3d4",
  "estimate": 14573.68,
  "n_warnings": 1,
  "latency_ms": 145
}
```

Log group: `/aws/lambda/price-estimator-predict`

**What we log:**

| Event | Level | Fields |
|---|---|---|
| Quote created | INFO | quote_id, estimate, n_warnings, latency_ms |
| Override recorded | INFO | quote_id, override_id, delta_pct |
| Quote retrieved | DEBUG | quote_id (low-value, debug only) |
| Model prediction failed | WARNING | model_name, error message |
| SHAP computation failed | WARNING | model_name, error message |
| All models failed | ERROR | input features (sanitized) |
| Validation rejected | WARNING | field, issue |

**What we do NOT log:**

- Full request/response bodies (contain pricing data — low risk now
  but bad habit to start).
- SHAP values (high cardinality, not useful in logs).
- Stack traces for expected errors (404, 400). Only for unexpected 500s.

**Retention:** 30 days for Lambda logs, 90 days for API Gateway access
logs. Configurable via CDK stack.

---

## 11. OpenAPI Documentation

FastAPI auto-generates an OpenAPI 3.1 spec. We expose it at:

- **Swagger UI:** `GET /v1/docs`
- **ReDoc:** `GET /v1/redoc`
- **Raw spec:** `GET /v1/openapi.json`

These are useful for reviewers exploring the API interactively. The
Swagger UI provides a "Try it out" button that sends real requests.

**Customizations to the default spec:**

- API title: "Price Estimator API"
- Version: "0.1.0"
- Description includes a brief overview, link to the architecture doc,
  and the list of known materials/processes/estimators for easy reference.
- Example values on all request fields so "Try it out" works immediately.
- Response schema descriptions match this document.

No additional tooling (Stoplight, Postman collections) is needed — the
auto-generated spec is the source of truth and stays in sync with the
code by definition.
