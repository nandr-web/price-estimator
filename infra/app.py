#!/usr/bin/env python3
"""CDK app entry point for the Price Estimator infrastructure."""

import aws_cdk as cdk

from stacks.price_estimator_stack import PriceEstimatorStack

app = cdk.App()

PriceEstimatorStack(
    app,
    "PriceEstimatorStack",
    description="Aerospace price estimator: Lambda API, DynamoDB, S3 artifacts, CloudWatch alarms",
)

app.synth()
