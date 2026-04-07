# Prerequisites

Tools and runtimes required to develop, build, test, and deploy the price estimator monorepo.

## Required for Local Development

| Tool | Min Version | Purpose | Install |
|------|-------------|---------|---------|
| Python | 3.11+ | Backend, scripts, CDK | [python.org](https://www.python.org/downloads/) or `brew install python@3.11` |
| uv | latest | Python package management | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Node.js | 18+ | Frontend build | [nodejs.org](https://nodejs.org/) or `brew install node` |
| npm | (bundled) | Frontend dependency management | Included with Node.js |
| ruff | 0.3+ | Python linting & formatting | Installed via `uv pip install -e ".[dev]"` |

### Setup

```bash
# Backend
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Frontend
cd frontend && npm install
```

## Required for Testing

| Tool | Min Version | Source | Purpose |
|------|-------------|--------|---------|
| pytest | 8.0+ | pyproject.toml `[dev]` | Test runner |
| pytest-cov | 5.0+ | pyproject.toml `[dev]` | Coverage reporting |
| httpx | 0.27+ | pyproject.toml `[dev]` | API test client (TestClient) |

All installed automatically via `uv pip install -e ".[dev]"`.

## Required for Deployment

| Tool | Min Version | Purpose | Install |
|------|-------------|---------|---------|
| Docker | any recent | Lambda container image build | [docker.com](https://docs.docker.com/get-docker/) |
| AWS CLI | v2 | CDK deployment, S3/ECR operations | [AWS docs](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) |
| AWS CDK CLI | 2.100+ | Infrastructure provisioning | `npm install -g aws-cdk` |

### AWS CDK Python dependencies

```bash
cd infra
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # aws-cdk-lib >=2.100, constructs >=10.0
```

### Lambda environment variables

| Variable | Description |
|----------|-------------|
| `ARTIFACTS_BUCKET` | S3 bucket containing serialized models and bounds |
| `ARTIFACTS_PREFIX` | S3 key prefix (default `v1`) |
| `QUOTES_TABLE` | DynamoDB table name for quotes |
| `OVERRIDES_TABLE` | DynamoDB table name for overrides |

## Optional

| Tool | Purpose |
|------|---------|
| JupyterLab 4.0+ | Notebook exploration (`uv pip install -e ".[dev]"`) |
| ipykernel 6.0+ | Jupyter kernel for this venv |

## Key Python Dependencies (reference)

Pinned in `pyproject.toml` — installed automatically, listed here for awareness:

| Package | Version Constraint | Role |
|---------|--------------------|------|
| pandas | >=2.0, <3 | Data loading and manipulation |
| scikit-learn | >=1.4, <2 | ML pipelines, Ridge, Lasso |
| xgboost | >=2.0, <3 | Gradient-boosted tree models (M5–M8) |
| lightgbm | >=4.0, <5 | Alternative tree model (M7c) |
| shap | >=0.44, <1 | Per-prediction explanations |
| fastapi | >=0.110, <1 | API framework |
| uvicorn | >=0.29, <1 | ASGI server |
| rapidfuzz | >=3.0, <4 | Fuzzy matching for part descriptions |
| mangum | >=0.17, <1 | Lambda ASGI adapter (requirements-lambda.txt) |
| boto3 | >=1.34, <2 | AWS SDK for Lambda deployment (requirements-lambda.txt) |

## Key Frontend Dependencies (reference)

Pinned in `frontend/package.json`:

| Package | Version | Role |
|---------|---------|------|
| React | ^19.2 | UI framework |
| Vite | ^8.0 | Build tooling |
| TypeScript | ~6.0 | Type checking |
| Tailwind CSS | ^4.2 | Styling |
| @tanstack/react-query | ^5.96 | API data fetching |
| react-router-dom | ^7.14 | Client-side routing |
| recharts | ^3.8 | SHAP chart visualization |
