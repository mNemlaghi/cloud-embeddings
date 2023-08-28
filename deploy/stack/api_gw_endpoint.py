from aws_cdk import (
    Stack,
    CfnParameter,
    aws_sagemaker as sagemaker,
    aws_apigateway as apigateway,
    aws_lambda,
    aws_iam as iam,
    Fn,
    Duration,
    CfnOutput
)

from utils.hf_model_data import PopulatedBucketResource

from constructs import Construct



class SmApiGatewayStack(Stack):

    def __init__(self, scope: Construct, id: str, endpoint: sagemaker.CfnEndpoint, **kwargs) -> None:
        super().__init__(scope, id,  **kwargs)


        #create function
        lambda_fn = aws_lambda.Function(
            self,
            "sm_invoke",
            code=aws_lambda.Code.from_asset("lambda-sminvoke"),
            handler="handler.proxy",
            timeout=Duration.seconds(60),
            runtime=aws_lambda.Runtime.PYTHON_3_8,
            environment={"ENDPOINT_NAME": endpoint.endpoint_name})

        lambda_fn.node.add_dependency(endpoint)
        # add policy for invoking
        lambda_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "sagemaker:InvokeEndpoint",
                ],
                resources=[f"arn:aws:sagemaker:{self.region}:{self.account}:endpoint/{endpoint.endpoint_name}"]
            )
        )

        api = apigateway.LambdaRestApi(self, "hf_api_gw", proxy=True, handler=lambda_fn)
        api.node.add_dependency(lambda_fn)
        CfnOutput(scope=self, id=f"ApiUrl", value=f"{api.url}")
