from aws_cdk import (
    Stack,
    CfnParameter,
    custom_resources as cr,
    aws_sagemaker as sm_cfn,
    aws_s3 as s3,
    aws_iam as iam,
    aws_lambda,
    RemovalPolicy,
    Duration,
    Size,
    Fn
)
import aws_cdk.aws_sagemaker_alpha as sagemaker
from utils.hf_model_data import PopulatedBucketResource

import os
from constructs import Construct
from datetime import datetime
import time



class PretrainedEmbeddingEndpointStack(Stack):

    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)
        upload_bucket_name = CfnParameter(self, "uploadBucketName", type="String",description="Bucket where models are stored", default="sm-pretrained-embedding-bucket")
        hf_model_id= CfnParameter(self, "hfModelId", type="String",description="model from HF", default="sentence-transformers/all-mpnet-base-v2")

        #First create a bucket populated with HF hub model parameters
        populated_bucket = PopulatedBucketResource(self, "PopulatedBucket", upload_bucket_name.value_as_string, hf_model_id.value_as_string)

        ##Now create the model
        image = sagemaker.ContainerImage.from_dlc("huggingface-pytorch-inference", "1.13.1-transformers4.26.0-cpu-py39-ubuntu20.04", account_id="763104351884")
        model_name="CFN-"+populated_bucket.core_model_name

        model_data = sagemaker.ModelData.from_bucket(populated_bucket.default_bucket, populated_bucket.model_archive_key)
        model = sagemaker.Model(self, "PrimaryContainerModel",containers=[sagemaker.ContainerDefinition(image=image,model_data=model_data)], model_name= model_name)
        model.node.add_dependency(populated_bucket)


        #Finally creating Endpoint Config And Endpoint
        cfn_endpoint_config = sm_cfn.CfnEndpointConfig(self, "MyCfnEndpointConfig",
            endpoint_config_name=Fn.join("", [model_name, "-endpoint-config"]),
			production_variants=[sm_cfn.CfnEndpointConfig.ProductionVariantProperty(
                initial_variant_weight=1,
                model_name=model.model_name,
                variant_name="AllTraffic",
                serverless_config=sm_cfn.CfnEndpointConfig.ServerlessConfigProperty(
                    max_concurrency=1,
                    memory_size_in_mb=6144))])

        cfn_endpoint_config.node.add_dependency(model)
        
        cfn_endpoint = sm_cfn.CfnEndpoint(self, "MyCfnEndpoint", 
                                          endpoint_config_name=cfn_endpoint_config.endpoint_config_name,
                                          endpoint_name=Fn.join("", [model_name, "-endpoint"]))

        cfn_endpoint.node.add_dependency(cfn_endpoint_config)
