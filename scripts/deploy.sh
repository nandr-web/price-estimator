#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> Activating venv"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"

echo "==> Building frontend"
cd "$ROOT/frontend"
npm run build

echo "==> Installing infra dependencies"
uv pip install -q -r "$ROOT/infra/requirements.txt"

echo "==> Deploying CDK stack"
cd "$ROOT/infra"
cdk deploy --require-approval never

echo "==> Done"
