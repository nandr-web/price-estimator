"""Main CDK stack for the Price Estimator service.

Resources:
  - S3 bucket (versioned, SSE-S3, 90-day lifecycle for old versions)
  - DynamoDB tables: Quotes (with TTL) and Overrides (on-demand billing)
  - ECR repository for Lambda container images
  - Lambda function (predict): 512 MB, 30s timeout, serves FastAPI via Mangum
  - Lambda function (retrain): 1024 MB, 900s timeout, weekly EventBridge trigger
  - HTTP API (API Gateway v2) with routes for /quote endpoints
  - CloudFront + S3 frontend (React SPA)
  - CloudWatch alarm on predict Lambda errors
  - IAM permissions wired per least-privilege
"""

from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
    aws_apigatewayv2 as apigwv2,
    aws_apigatewayv2_integrations as apigwv2_integrations,
    aws_cloudfront as cloudfront,
    aws_cloudfront_origins as origins,
    aws_cloudwatch as cloudwatch,
    aws_cloudwatch_actions as cw_actions,
    aws_dynamodb as dynamodb,
    aws_ecr as ecr,
    aws_events as events,
    aws_events_targets as targets,
    aws_iam as iam,
    aws_lambda as lambda_,
    aws_logs as logs,
    aws_s3 as s3,
    aws_s3_deployment as s3_deployment,
    aws_sns as sns,
    aws_sns_subscriptions as sns_subs,
)
from constructs import Construct


class PriceEstimatorStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ------------------------------------------------------------------ #
        # S3 — model artifact storage                                         #
        # ------------------------------------------------------------------ #
        artifacts_bucket = s3.Bucket(
            self,
            "ArtifactsBucket",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=RemovalPolicy.RETAIN,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="ExpireOldVersions",
                    noncurrent_version_expiration=Duration.days(90),
                    enabled=True,
                ),
            ],
        )

        # ------------------------------------------------------------------ #
        # DynamoDB — Quotes table (with TTL)                                  #
        # ------------------------------------------------------------------ #
        quotes_table = dynamodb.Table(
            self,
            "QuotesTable",
            table_name="PriceEstimator-Quotes",
            partition_key=dynamodb.Attribute(
                name="quote_id", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.RETAIN,
            time_to_live_attribute="ttl",
        )

        # ------------------------------------------------------------------ #
        # DynamoDB — Overrides table                                          #
        # ------------------------------------------------------------------ #
        overrides_table = dynamodb.Table(
            self,
            "OverridesTable",
            table_name="PriceEstimator-Overrides",
            partition_key=dynamodb.Attribute(
                name="quote_id", type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="override_id", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.RETAIN,
        )

        # ------------------------------------------------------------------ #
        # ECR — container image repository                                    #
        # ------------------------------------------------------------------ #
        ecr_repo = ecr.Repository(
            self,
            "LambdaImageRepo",
            repository_name="price-estimator-lambda",
            removal_policy=RemovalPolicy.RETAIN,
            lifecycle_rules=[
                ecr.LifecycleRule(
                    description="Keep last 5 images",
                    max_image_count=5,
                ),
            ],
        )

        # ------------------------------------------------------------------ #
        # CloudWatch — log groups with retention                              #
        # ------------------------------------------------------------------ #
        predict_log_group = logs.LogGroup(
            self,
            "PredictLogGroup",
            log_group_name="/aws/lambda/price-estimator-predict",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.DESTROY,
        )

        retrain_log_group = logs.LogGroup(
            self,
            "RetrainLogGroup",
            log_group_name="/aws/lambda/price-estimator-retrain",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.DESTROY,
        )

        api_access_log_group = logs.LogGroup(
            self,
            "ApiAccessLogGroup",
            log_group_name="/aws/apigateway/price-estimator-api",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.DESTROY,
        )

        # ------------------------------------------------------------------ #
        # Lambda — predict (FastAPI + Mangum)                                 #
        # ------------------------------------------------------------------ #
        predict_fn = lambda_.DockerImageFunction(
            self,
            "PredictFunction",
            function_name="price-estimator-predict",
            code=lambda_.DockerImageCode.from_ecr(
                repository=ecr_repo,
                tag_or_digest="latest",
            ),
            memory_size=2048,
            timeout=Duration.seconds(60),
            environment={
                "ARTIFACTS_BUCKET": artifacts_bucket.bucket_name,
                "ARTIFACTS_PREFIX": "v1",
                "QUOTES_TABLE": quotes_table.table_name,
                "OVERRIDES_TABLE": overrides_table.table_name,
            },
            description="Serves /quote endpoints via FastAPI + Mangum",
            current_version_options=lambda_.VersionOptions(
                removal_policy=RemovalPolicy.RETAIN,
                description="Auto-published version for alias tracking",
            ),
        )

        # ------------------------------------------------------------------ #
        # Lambda alias — "live" with provisioned concurrency                  #
        # ------------------------------------------------------------------ #
        predict_alias = lambda_.Alias(
            self,
            "PredictLiveAlias",
            alias_name="live",
            version=predict_fn.current_version,
            provisioned_concurrent_executions=1,
        )

        # ------------------------------------------------------------------ #
        # Lambda — retrain                                                    #
        # ------------------------------------------------------------------ #
        retrain_fn = lambda_.DockerImageFunction(
            self,
            "RetrainFunction",
            function_name="price-estimator-retrain",
            code=lambda_.DockerImageCode.from_ecr(
                repository=ecr_repo,
                tag_or_digest="latest",
            ),
            memory_size=1024,
            timeout=Duration.seconds(900),
            environment={
                "ARTIFACTS_BUCKET": artifacts_bucket.bucket_name,
                "ARTIFACTS_PREFIX": "v1",
                "QUOTES_TABLE": quotes_table.table_name,
                "PREDICT_FUNCTION_NAME": "price-estimator-predict",
            },
            description="Retrains models on schedule, writes artifacts to S3",
        )

        # ------------------------------------------------------------------ #
        # IAM — predict Lambda permissions                                    #
        # ------------------------------------------------------------------ #
        artifacts_bucket.grant_read(predict_fn)
        quotes_table.grant_read_write_data(predict_fn)
        overrides_table.grant_read_write_data(predict_fn)

        # ------------------------------------------------------------------ #
        # IAM — retrain Lambda permissions                                    #
        # ------------------------------------------------------------------ #
        artifacts_bucket.grant_read_write(retrain_fn)
        quotes_table.grant_read_data(retrain_fn)
        overrides_table.grant_read_data(retrain_fn)

        # Allow retrain to update predict Lambda's env vars (for new model version)
        retrain_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["lambda:UpdateFunctionConfiguration"],
                resources=[predict_fn.function_arn],
            )
        )

        # ------------------------------------------------------------------ #
        # API Gateway — HTTP API v2                                           #
        # ------------------------------------------------------------------ #
        predict_integration = apigwv2_integrations.HttpLambdaIntegration(
            "PredictIntegration",
            handler=predict_alias,
        )

        http_api = apigwv2.HttpApi(
            self,
            "HttpApi",
            api_name="price-estimator-api",
            description="Price Estimator HTTP API",
            cors_preflight=apigwv2.CorsPreflightOptions(
                allow_origins=["*"],
                allow_methods=[
                    apigwv2.CorsHttpMethod.GET,
                    apigwv2.CorsHttpMethod.POST,
                    apigwv2.CorsHttpMethod.OPTIONS,
                ],
                allow_headers=["Content-Type"],
                max_age=Duration.hours(24),
            ),
        )

        # Throttling + access logging on the $default stage
        cfn_stage = http_api.default_stage.node.default_child
        cfn_stage.add_property_override("DefaultRouteSettings.ThrottlingBurstLimit", 50)
        cfn_stage.add_property_override("DefaultRouteSettings.ThrottlingRateLimit", 20)
        cfn_stage.add_property_override("AccessLogSettings.DestinationArn",
                                        api_access_log_group.log_group_arn)
        cfn_stage.add_property_override(
            "AccessLogSettings.Format",
            '{"requestId":"$context.requestId",'
            '"ip":"$context.identity.sourceIp",'
            '"method":"$context.httpMethod",'
            '"path":"$context.path",'
            '"status":"$context.status",'
            '"latency":"$context.responseLatency",'
            '"integrationLatency":"$context.integrationLatency"}',
        )

        # Grant API Gateway permission to write to the access log group
        api_access_log_group.grant_write(iam.ServicePrincipal("apigateway.amazonaws.com"))

        http_api.add_routes(
            path="/v1/quote",
            methods=[apigwv2.HttpMethod.POST],
            integration=predict_integration,
        )
        http_api.add_routes(
            path="/v1/quotes",
            methods=[apigwv2.HttpMethod.GET],
            integration=predict_integration,
        )
        http_api.add_routes(
            path="/v1/quote/{quote_id}",
            methods=[apigwv2.HttpMethod.GET],
            integration=predict_integration,
        )
        http_api.add_routes(
            path="/v1/quote/{quote_id}/override",
            methods=[apigwv2.HttpMethod.POST],
            integration=predict_integration,
        )
        http_api.add_routes(
            path="/v1/quote/{quote_id}/send",
            methods=[apigwv2.HttpMethod.POST],
            integration=predict_integration,
        )
        http_api.add_routes(
            path="/v1/quote/{quote_id}/outcome",
            methods=[apigwv2.HttpMethod.POST],
            integration=predict_integration,
        )

        # ------------------------------------------------------------------ #
        # EventBridge — weekly retrain schedule                               #
        # ------------------------------------------------------------------ #
        retrain_rule = events.Rule(
            self,
            "WeeklyRetrainRule",
            rule_name="price-estimator-weekly-retrain",
            schedule=events.Schedule.cron(
                week_day="MON",
                hour="6",
                minute="0",
            ),
            description="Triggers model retraining every Monday at 06:00 UTC",
        )
        retrain_rule.add_target(targets.LambdaFunction(retrain_fn))

        # ------------------------------------------------------------------ #
        # EventBridge — predict Lambda warm-up (every 5 minutes)              #
        # ------------------------------------------------------------------ #
        warmup_rule = events.Rule(
            self,
            "PredictWarmupRule",
            rule_name="price-estimator-predict-warmup",
            schedule=events.Schedule.rate(Duration.minutes(5)),
            description="Keeps predict Lambda warm by invoking alias every 5 minutes",
        )
        warmup_rule.add_target(targets.LambdaFunction(predict_alias))

        # ------------------------------------------------------------------ #
        # SNS topic for alarms (optional recipient — add subscription later)  #
        # ------------------------------------------------------------------ #
        alarm_topic = sns.Topic(
            self,
            "AlarmTopic",
            topic_name="price-estimator-alarms",
            display_name="Price Estimator Alarms",
        )

        # ------------------------------------------------------------------ #
        # CloudWatch — predict Lambda error alarm                             #
        # ------------------------------------------------------------------ #
        predict_errors_alarm = cloudwatch.Alarm(
            self,
            "PredictErrorsAlarm",
            alarm_name="price-estimator-predict-errors",
            metric=predict_fn.metric_errors(
                period=Duration.minutes(5),
                statistic="Sum",
            ),
            threshold=5,
            evaluation_periods=1,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
            alarm_description="Fires when predict Lambda has > 5 errors in 5 minutes",
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
        )
        predict_errors_alarm.add_alarm_action(cw_actions.SnsAction(alarm_topic))

        # ------------------------------------------------------------------ #
        # CloudWatch — predict Lambda p99 duration alarm                      #
        # ------------------------------------------------------------------ #
        predict_duration_alarm = cloudwatch.Alarm(
            self,
            "PredictDurationAlarm",
            alarm_name="price-estimator-predict-duration-p99",
            metric=predict_fn.metric_duration(
                period=Duration.minutes(5),
                statistic="p99",
            ),
            threshold=10_000,  # 10 seconds in milliseconds
            evaluation_periods=1,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
            alarm_description="Fires when predict Lambda p99 duration > 10s (cold start concern)",
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
        )
        predict_duration_alarm.add_alarm_action(cw_actions.SnsAction(alarm_topic))

        # ------------------------------------------------------------------ #
        # S3 — frontend static files                                          #
        # ------------------------------------------------------------------ #
        frontend_bucket = s3.Bucket(
            self,
            "FrontendBucket",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

        # ------------------------------------------------------------------ #
        # CloudFront — serves SPA + proxies /v1/* to API Gateway              #
        # ------------------------------------------------------------------ #
        api_origin = origins.HttpOrigin(
            f"{http_api.api_id}.execute-api.{self.region}.amazonaws.com",
        )

        distribution = cloudfront.Distribution(
            self,
            "Distribution",
            default_behavior=cloudfront.BehaviorOptions(
                origin=origins.S3BucketOrigin.with_origin_access_control(
                    frontend_bucket,
                ),
                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                cache_policy=cloudfront.CachePolicy.CACHING_OPTIMIZED,
            ),
            additional_behaviors={
                "/v1/*": cloudfront.BehaviorOptions(
                    origin=api_origin,
                    allowed_methods=cloudfront.AllowedMethods.ALLOW_ALL,
                    cache_policy=cloudfront.CachePolicy.CACHING_DISABLED,
                    origin_request_policy=cloudfront.OriginRequestPolicy.ALL_VIEWER_EXCEPT_HOST_HEADER,
                    viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                ),
            },
            default_root_object="index.html",
            error_responses=[
                cloudfront.ErrorResponse(
                    http_status=404,
                    response_http_status=200,
                    response_page_path="/index.html",
                    ttl=Duration.seconds(0),
                ),
                cloudfront.ErrorResponse(
                    http_status=403,
                    response_http_status=200,
                    response_page_path="/index.html",
                    ttl=Duration.seconds(0),
                ),
            ],
        )

        # ------------------------------------------------------------------ #
        # Deploy frontend build to S3 + invalidate CloudFront                 #
        # ------------------------------------------------------------------ #
        s3_deployment.BucketDeployment(
            self,
            "DeployFrontend",
            sources=[s3_deployment.Source.asset("../frontend/dist")],
            destination_bucket=frontend_bucket,
            distribution=distribution,
            distribution_paths=["/*"],
        )

        # ------------------------------------------------------------------ #
        # Outputs                                                              #
        # ------------------------------------------------------------------ #
        CfnOutput(self, "FrontendUrl",
                  value=f"https://{distribution.distribution_domain_name}",
                  description="CloudFront URL for the frontend")
        CfnOutput(self, "ApiUrl",
                  value=http_api.api_endpoint,
                  description="API Gateway endpoint URL")
