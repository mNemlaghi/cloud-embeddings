from aws_cdk import (
    Stack,
    CfnParameter,
    aws_sagemaker as sagemaker,
    aws_iam as iam,
    Fn,
    CfnOutput
)

from utils.hf_model_data import PopulatedBucketResource

from constructs import Construct



class PretrainedEmbeddingEndpointStack(Stack):

    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)
        upload_bucket_name = CfnParameter(self, "uploadBucketName", type="String",description="Bucket where models are stored", default="sm-pretrained-embedding-bucket")
        hf_model_id= CfnParameter(self, "hfModelId", type="String",description="model from HF", default="sentence-transformers/all-mpnet-base-v2")
        model_name= CfnParameter(self, "ModelName", type="String",default="cfn-pretrained-embedding", description="Name for Embedding" )

        #First create a bucket populated with HF hub model parameters
        populated_bucket = PopulatedBucketResource(self, "PopulatedBucket", upload_bucket_name.value_as_string, 
                                                   hf_model_id.value_as_string, 
                                                   model_name.value_as_string)

        
        #Create SageMaker Role that has access to model artefacts
        sm_role = iam.Role(self, "Role",
                assumed_by=iam.CompositePrincipal(iam.ServicePrincipal("sagemaker.amazonaws.com")))
        
        sm_role.add_managed_policy(iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess"))
        populated_bucket.default_bucket.grant_read_write(sm_role)

        model = sagemaker.CfnModel(self, "model",
                                   model_name=model_name.value_as_string,
                                   primary_container={
                                       "image": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-cpu-py39-ubuntu20.04",
                                       "modelDataUrl": populated_bucket.model_artefacts_s3_uri
                                       },
                                    execution_role_arn=sm_role.role_arn
                                    )
        model.node.add_dependency(populated_bucket)


        #Finally creating Endpoint Config And Endpoint
        cfn_endpoint_config = sagemaker.CfnEndpointConfig(self, "MyCfnEndpointConfig",
            endpoint_config_name=Fn.join("", [model.model_name, "-endpoint-config"]),
			production_variants=[sagemaker.CfnEndpointConfig.ProductionVariantProperty(
                initial_variant_weight=1,
                model_name=model.model_name,
                variant_name="AllTraffic",
                serverless_config=sagemaker.CfnEndpointConfig.ServerlessConfigProperty(
                    max_concurrency=1,
                    memory_size_in_mb=6144))])

        cfn_endpoint_config.node.add_dependency(model)
        
        cfn_endpoint = sagemaker.CfnEndpoint(self, "MyCfnEndpoint", 
                                          endpoint_config_name=cfn_endpoint_config.endpoint_config_name,
                                          endpoint_name=Fn.join("", [model.model_name, "-endpoint"]))

        cfn_endpoint.node.add_dependency(cfn_endpoint_config)
        CfnOutput(scope=self, id=f"EndpointName", value=f"{cfn_endpoint.endpoint_name}")
