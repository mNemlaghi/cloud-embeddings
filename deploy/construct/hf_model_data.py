from aws_cdk import (
    custom_resources as cr,
    aws_s3 as s3,
    aws_iam as iam,
    aws_lambda,
    RemovalPolicy,
    Duration,
    Size,
    Fn,
    CfnOutput
)


import os
from constructs import Construct
from datetime import datetime


class BucketPopulatorResource(Construct):
    def __init__(self, scope: Construct, id: str,  bucket_name, hf_model_id, model_name):
        super().__init__(scope, id)

        ecr_image = aws_lambda.EcrImageCode.from_asset_image(directory = os.path.join(os.getcwd(), "lambda-bundler"), asset_name="HfToS3")
        
        #Create a lambda function and invoke it
        self.lambda_fn = aws_lambda.Function(self,
            id= "lambdaContainerFunction",
            description   = "Sample Lambda Container Function", 
            code = ecr_image,
            handler = aws_lambda.Handler.FROM_IMAGE,
            runtime = aws_lambda.Runtime.FROM_IMAGE,
            function_name = "HFHubToS3",
            memory_size   = 6144,
            ephemeral_storage_size=Size.mebibytes(4096),
            timeout = Duration.seconds(900))


        invoker = cr.AwsCustomResource(self, 
                                       id= "PutModelInS3CustomResource",
                                       timeout=Duration.minutes(15), ##Tarring file can be expensive
                                       on_create = self.create(self.lambda_fn.function_name, bucket_name, hf_model_id, model_name),
                                       resource_type='Custom::InvokerResource',
                                       policy=cr.AwsCustomResourcePolicy.from_statements([iam.PolicyStatement(
                                           actions=['lambda:InvokeFunction'],
                                           effect=iam.Effect.ALLOW,
                                           resources=[self.lambda_fn.function_arn])]))
        
        self.core_model_name=hf_model_id.split('/')[-1]
        self.model_artefacts_s3_uri  = "s3://"+bucket_name + "/models/"+model_name+"/model.tar.gz"

        CfnOutput(scope=self, value=f"Invoke-{self.lambda_fn.function_name}-{hf_model_id}", id="HfLambdaInvoke")

        @property
        def lambda_name(self):
            return self.lambda_fn.function_name

        
    def create(self, lambda_name, bucket_name, hf_model_id, model_name):
        lambda_params = {}
        lambda_params['FunctionName'] = lambda_name
        lambda_params['InvocationType'] = "RequestResponse"
        payload = "".join( ["{\"hf_model_id\":\"", hf_model_id, "\", \"model_name\":\"",model_name,"\", \"bucket_name\":\"",bucket_name, "\"}"])
        lambda_params['Payload'] = payload

        now = "invoker"+datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        return cr.AwsSdkCall(service="Lambda",
                          action="invoke",
                          parameters = lambda_params,
                          physical_resource_id=cr.PhysicalResourceId.of("myAutomationExecution"))



class PopulatedBucketResource(Construct):

    def __init__(self, scope: Construct, id: str, bucket_name, hf_model_id, model_name):
        super().__init__(scope, id)

        #Create upload bucket
        self.default_bucket = s3.Bucket(self, id="bucket123", bucket_name=bucket_name, removal_policy=RemovalPolicy.DESTROY, auto_delete_objects=True)

        #Let's invoke the Lambda function beforehand

        bucket_populator= BucketPopulatorResource(self, "InvokerResource",  bucket_name, hf_model_id, model_name)
        self.default_bucket.grant_read_write(bucket_populator.lambda_fn)
        bucket_populator.lambda_fn.node.add_dependency(self.default_bucket)


        self.bucket_name=bucket_name
        self.model_artefacts_s3_uri=bucket_populator.model_artefacts_s3_uri
        CfnOutput(scope=self, id=f"BucketName", value=f"{self.default_bucket.bucket_name}")
        CfnOutput(scope=self, id=f"ModelArtefactsS3Uri", value=f"{self.model_artefacts_s3_uri}")


