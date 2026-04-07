# AWS Architecture — Price Estimator

## Decision: Lambda, not SageMaker

Our models are scikit-learn Ridge regressions, XGBoost, and LightGBM trees.
Total serialized size is 3.4 MB. Inference on a single row takes <50ms for
all 14 models combined. SageMaker real-time endpoints run a minimum ~$50/month
for an always-on container and are designed for GPU workloads or
high-throughput inference. Lambda's pay-per-request model is the right fit
for a quoting tool that handles dozens to low hundreds of requests per day.

---

## High-Level Design

```
                          ┌──────────────────────────────────────────┐
                          │              S3 (artifacts)              │
                          │  models/*.joblib          (3.4 MB)      │
                          │  prediction_bands.json    (2 KB)        │
                          │  training_bounds.json     (1 KB)        │
                          │  training_data.csv        (source)      │
                          └────────┬──────────────────┬─────────────┘
                                   │                  │
                          cold start load        training input
                                   │                  │
┌────────┐    ┌──────────────┐    ┌▼──────────────┐  ┌▼──────────────────┐
│ Client │───▶│ API Gateway  │───▶│ Lambda        │  │ Lambda (retrain)  │
│        │◀───│ (HTTP API)   │◀───│ (predict)     │  │ triggered by      │
└────────┘    └──────────────┘    │               │  │ EventBridge or    │
                                  │ FastAPI +     │  │ manual invoke     │
                                  │ Mangum        │  │                   │
                                  └───┬───────────┘  └───────┬──────────┘
                                      │                      │
                                      ▼                      │ writes new
                                  ┌───────────┐              │ models to S3
                                  │ DynamoDB  │              │
                                  │ Quotes    │◀─────────────┘
                                  │ Overrides │  (optional: retrain
                                  └───────────┘   from override data)
```

### Components

| Component | Service | Purpose |
|---|---|---|
| **Predict** | API Gateway + Lambda | Serves `/quote`, `/quote/{id}/override`, `/quote/{id}` |
| **Storage** | DynamoDB | Quotes and overrides (replaces SQLite) |
| **Artifacts** | S3 (versioned) | Model files, prediction bands, training bounds |
| **Retrain** | Lambda (15-min timeout) | Re-runs training pipeline, writes new artifacts to S3 |
| **Schedule** | EventBridge | Triggers retrain on schedule or manual invoke |
| **Monitoring** | CloudWatch | Latency, error rates, override rate drift alarm |

---

## Low-Level Details

### 1. Predict Lambda

**Runtime:** Python 3.11, 512 MB memory, 30s timeout.

**Packaging:** Lambda container image (ECR). The dependency footprint
(scikit-learn + xgboost + lightgbm + shap + pandas + numpy + rapidfuzz)
exceeds the 250 MB layer limit when uncompressed, so a container image
(up to 10 GB) is the cleanest path. This also means the Dockerfile
pins exact versions and produces reproducible builds.

```dockerfile
FROM public.ecr.aws/lambda/python:3.11

COPY requirements-lambda.txt .
RUN pip install --no-cache-dir -r requirements-lambda.txt

COPY src/ ${LAMBDA_TASK_ROOT}/src/
ENV PYTHONPATH="${LAMBDA_TASK_ROOT}/src"

COPY lambda_handler.py ${LAMBDA_TASK_ROOT}/

CMD ["lambda_handler.handler"]
```

**Handler:** Mangum wraps the existing FastAPI app. Model loading happens
once per cold start and is cached in module-level globals (same pattern
as the current `set_models()` design).

```python
# lambda_handler.py
import json
import os
import boto3
import joblib
from io import BytesIO
from mangum import Mangum
from price_estimator.api import app, set_models
from price_estimator.predict import TrainingBounds, load_prediction_bands

S3_BUCKET = os.environ["ARTIFACTS_BUCKET"]
S3_PREFIX = os.environ.get("ARTIFACTS_PREFIX", "v1")

_initialized = False

def _init():
    """Load models from S3 on first invocation (cold start)."""
    global _initialized
    if _initialized:
        return
    s3 = boto3.client("s3")

    # List and load all .joblib model files
    models = {}
    resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=f"{S3_PREFIX}/models/")
    for obj in resp.get("Contents", []):
        if obj["Key"].endswith(".joblib"):
            body = s3.get_object(Bucket=S3_BUCKET, Key=obj["Key"])["Body"].read()
            model = joblib.load(BytesIO(body))
            models[model.name] = model

    # Load prediction bands
    bands = None
    try:
        body = s3.get_object(
            Bucket=S3_BUCKET,
            Key=f"{S3_PREFIX}/results/prediction_bands.json",
        )["Body"].read()
        bands = json.loads(body)
    except s3.exceptions.NoSuchKey:
        pass

    # Load training bounds
    bounds = None
    try:
        body = s3.get_object(
            Bucket=S3_BUCKET,
            Key=f"{S3_PREFIX}/results/training_bounds.json",
        )["Body"].read()
        bounds_data = json.loads(body)
        bounds = TrainingBounds()
        bounds.quantity_range = tuple(bounds_data["quantity_range"])
        bounds.lead_time_range = tuple(bounds_data["lead_time_range"])
        bounds.known_materials = set(bounds_data["known_materials"])
        bounds.known_processes = set(bounds_data["known_processes"])
        bounds.known_part_types = set(bounds_data["known_part_types"])
        bounds.known_estimators = set(bounds_data["known_estimators"])
    except s3.exceptions.NoSuchKey:
        pass

    set_models(models, bounds, prediction_bands=bands)
    _initialized = True

def handler(event, context):
    _init()
    return Mangum(app, lifespan="off")(event, context)
```

**Cold start budget:**
- S3 reads: 14 models x ~300 KB avg = ~4.2 MB total. At S3's typical
  throughput this takes <1s.
- Model deserialization (joblib.load): <500ms for all 14.
- Python import time (sklearn, xgboost, etc.): ~1-2s.
- Total cold start: ~3-4s. Acceptable for a quoting tool. If needed,
  provisioned concurrency ($) eliminates cold starts entirely.

**Warm invocation:** <200ms end-to-end (API Gateway overhead + feature
engineering + 14-model inference + SHAP).

### 2. DynamoDB Tables

Replace SQLite with two DynamoDB tables. The access patterns are simple
key lookups — no need for a relational database.

**Table: `PriceEstimator-Quotes`**

| Attribute | Type | Key |
|---|---|---|
| `quote_id` | String | Partition key |
| `features` | String (JSON) | — |
| `model_price` | Number | — |
| `aggressive_estimate` | Number | — |
| `conservative_estimate` | Number | — |
| `typical_range_low` | Number | — |
| `typical_range_high` | Number | — |
| `warnings` | List\<String\> | — |
| `created_at` | String (ISO 8601) | — |
| `ttl` | Number (epoch) | TTL attribute (auto-expire old quotes) |

**Table: `PriceEstimator-Overrides`**

| Attribute | Type | Key |
|---|---|---|
| `quote_id` | String | Partition key |
| `override_id` | String | Sort key |
| `human_price` | Number | — |
| `reason_category` | String | — |
| `reason_text` | String | — |
| `estimator_id` | String | — |
| `delta_from_model` | Number | — |
| `created_at` | String (ISO 8601) | — |

**Capacity:** On-demand mode. At quoting volumes (dozens/day), cost is
effectively $0. The free tier covers 25 WCU + 25 RCU perpetually.

**Code change:** Replace `get_db()` / raw SQL in `api.py` with a thin
`_get_dynamo_table()` wrapper using `boto3.resource("dynamodb")`. The
three endpoints each do 1-2 DynamoDB operations (PutItem, GetItem, Query).

### 3. S3 Artifact Bucket

```
s3://price-estimator-artifacts/
  v1/
    models/
      M0.joblib
      M2.joblib
      ...
    results/
      prediction_bands.json
      training_bounds.json
  v2/                          ← after retrain
    models/
      ...
```

**Versioning strategy:** Each training run writes to a new prefix
(`v1`, `v2`, ...). The predict Lambda reads from a prefix set via
environment variable `ARTIFACTS_PREFIX`. Deploying a new model version
means updating that env var and the Lambda picks it up on next cold start.
Rolling back = point `ARTIFACTS_PREFIX` back to the previous version.

**Bucket policy:** Versioning enabled. Lifecycle rule expires old versions
after 90 days. Server-side encryption (SSE-S3). No public access.

### 4. Retrain Lambda

**Runtime:** Python 3.11, 1024 MB memory, 15-min timeout.

**Trigger:** EventBridge rule (weekly schedule, or manual invoke via
console/CLI). Could also be triggered when new training data is uploaded
to S3 via S3 event notification.

**What it does:**
1. Pulls training CSV from S3
2. Optionally pulls override data from DynamoDB (for retraining with
   human feedback — future enhancement)
3. Runs `scripts/train.py` logic (all 14 models, 5-fold CV)
4. Writes new model artifacts + prediction bands + training bounds to
   S3 under a new version prefix
5. Optionally updates the predict Lambda's `ARTIFACTS_PREFIX` env var
   via `lambda:UpdateFunctionConfiguration`

**Training time:** With 510 rows and 14 models, full training takes ~60s
on a laptop. A 1024 MB Lambda (with 0.5 vCPU equivalent) should complete
in under 5 minutes.

### 5. API Gateway

**Type:** HTTP API (v2) — cheaper and lower latency than REST API (v1).
REST API is only needed for features like request validation, API keys,
or WAF integration. HTTP API supports JWT authorizers and is sufficient.

**Routes:**

| Method | Route | Lambda |
|---|---|---|
| POST | `/quote` | predict |
| POST | `/quote/{quote_id}/override` | predict |
| GET | `/quote/{quote_id}` | predict |

**CORS:** Configured at API Gateway level if a web frontend is added.

**Throttling:** Default 1000 req/s burst is more than sufficient.
Per-route throttling can be added if needed.

**Auth:** Start with API key (x-api-key header) for simplicity. Upgrade
to Cognito JWT authorizer if user-level auth is needed later.

### 6. Monitoring & Alarms

| Metric | Source | Alarm threshold |
|---|---|---|
| Lambda errors | CloudWatch Metrics | > 5 errors in 5 min |
| Lambda duration (p99) | CloudWatch Metrics | > 10s (cold start concern) |
| Override rate | Custom metric (emitted by override endpoint) | > 50% of quotes overridden in 24h |
| Override delta magnitude | Custom metric | Mean |delta| > 30% (model drift) |
| S3 model age | Custom metric (retrain Lambda emits) | > 30 days since last retrain |

**Override drift detection:** The override endpoint emits a custom
CloudWatch metric with the `delta_pct` value. A CloudWatch alarm on the
rolling 24-hour mean detects when humans are consistently correcting the
model in one direction — a signal to retrain.

### 7. Cost Estimate

| Component | Monthly cost at ~100 quotes/day |
|---|---|
| Lambda (predict) | ~$0.50 (3M free tier requests covers most of it) |
| Lambda (retrain) | ~$0.01 (runs once/week, 5 min each) |
| API Gateway | ~$1.00 (HTTP API at $1/million requests) |
| DynamoDB | ~$0.00 (free tier: 25 WCU/RCU) |
| S3 | ~$0.10 (a few MB of models) |
| CloudWatch | ~$0.00 (basic metrics free) |
| ECR | ~$0.10 (container image storage) |
| **Total** | **~$2/month** |

At 10,000 quotes/day (unlikely for a machine shop), total would still be
under $20/month.

### 8. Deployment

**IaC:** AWS SAM (Serverless Application Model). Single `template.yaml`
defines all resources: Lambda functions, API Gateway, DynamoDB tables,
S3 bucket, EventBridge rule, CloudWatch alarms.

**CI/CD pipeline (GitHub Actions or similar):**
1. `ruff check` + `pytest` (existing test suite)
2. `docker build` → push to ECR
3. `sam deploy` → CloudFormation stack update
4. Smoke test: POST /quote with a known input, assert 200

**Local development:** No change to current workflow. `scripts/serve.py`
continues to work locally with uvicorn + SQLite. The Lambda handler and
DynamoDB code are isolated behind a thin abstraction so the core
`price_estimator` package doesn't import boto3.

---

## Migration Path from Current Codebase

The current codebase needs minimal changes to deploy:

| Change | Scope | Description |
|---|---|---|
| **Add `lambda_handler.py`** | New file | Mangum wrapper + S3 model loading (see above) |
| **Abstract DB layer in `api.py`** | ~50 lines | Replace `get_db()` SQLite calls with a `QuoteStore` interface. Local impl uses SQLite, Lambda impl uses DynamoDB |
| **Serialize `TrainingBounds` to JSON** | ~20 lines | Add `to_json()`/`from_json()` to `TrainingBounds` so it can be stored in S3 alongside models |
| **Add `Dockerfile`** | New file | Python 3.11 Lambda base image + deps + source |
| **Add `template.yaml`** | New file | SAM template for all AWS resources |
| **Add `requirements-lambda.txt`** | New file | Trimmed deps (no dev, no jupyter, no matplotlib) |

The core `price_estimator` package, all 14 models, the feature engineering
pipeline, and the FastAPI endpoint logic remain unchanged.
