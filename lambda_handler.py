"""AWS Lambda handler — wraps FastAPI with Mangum and loads models from S3."""

import json
import os
from io import BytesIO

import boto3
import joblib
from mangum import Mangum

from price_estimator.api import DynamoQuoteStore, app, set_models, set_store
from price_estimator.predict import TrainingBounds

S3_BUCKET = os.environ["ARTIFACTS_BUCKET"]
S3_PREFIX = os.environ.get("ARTIFACTS_PREFIX", "v1")

_initialized = False


def _init():
    """Load models from S3 on first invocation (cold start)."""
    global _initialized
    if _initialized:
        return

    s3 = boto3.client("s3")

    # Load all .joblib model files
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
        bounds = TrainingBounds.from_json(json.loads(body))
    except s3.exceptions.NoSuchKey:
        pass

    set_models(models, bounds, prediction_bands=bands)

    # Wire DynamoDB store
    set_store(
        DynamoQuoteStore(
            quotes_table_name=os.environ["QUOTES_TABLE"],
            overrides_table_name=os.environ["OVERRIDES_TABLE"],
        )
    )

    _initialized = True


_mangum = Mangum(app, lifespan="off")


def handler(event, context):
    _init()
    return _mangum(event, context)
