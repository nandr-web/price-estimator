# Deployment Checklist — Price Estimator

## What's done

| Layer | Status |
|---|---|
| Core ML pipeline | 14 models trained, 5-fold CV, all artifacts in `outputs/` |
| Prediction engine | Top-tier consensus, bands, SHAP, OOD detection |
| FastAPI endpoints | 3 endpoints, `/v1` prefix, error envelope, validation, CORS, structured logging, OpenAPI examples |
| Tests | 28 API tests + ~150 total, lint clean |
| CDK stack | Synthesizes — S3, DynamoDB, Lambda x2, API Gateway, EventBridge, CloudWatch |
| Design docs | AWS architecture (`docs/aws-architecture.md`) + API design (`docs/api-design.md`) |

## What remains before deploying

### 1. Lambda handler + Dockerfile (new files, ~50 lines each)

- `lambda_handler.py` — Mangum wrapper + S3 model loading (already drafted in `docs/aws-architecture.md`)
- `Dockerfile` — Lambda container image with deps
- `requirements-lambda.txt` — trimmed runtime deps (no jupyter, matplotlib, dev tools)

### 2. DB abstraction layer (~50 lines in `api.py`)

The current code calls `get_db()` which returns a SQLite connection. For Lambda we need DynamoDB. The cleanest path is a thin `QuoteStore` interface:

- `QuoteStore.save_quote()` / `QuoteStore.get_quote()` / `QuoteStore.save_override()` / `QuoteStore.get_latest_override()`
- `SqliteQuoteStore` (current behavior, for local dev)
- `DynamoQuoteStore` (for Lambda, uses boto3)

The endpoint code calls the store interface; the serve script or lambda handler picks the implementation.

### 3. TrainingBounds serialization (~20 lines in `predict.py`)

`TrainingBounds` currently only has `from_dataframe()`. We need `to_json()` / `from_json()` so the retrain Lambda can write bounds to S3 and the predict Lambda can load them without the training CSV.

### 4. CDK stack refinements

- Rate limiting config (throttle settings on HTTP API stage)
- CORS at gateway level
- CloudWatch log groups with retention policies
- API Gateway access log format

### 5. CI/CD pipeline (optional for first deploy)

- GitHub Actions: lint + test + docker build + push ECR + cdk deploy
- Smoke test post-deploy
