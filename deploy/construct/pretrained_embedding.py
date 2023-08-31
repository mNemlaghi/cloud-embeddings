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

from construct.hf_model_data import PopulatedBucketResource

from constructs import Construct
import boto3
import json
from sagemaker import image_uris, model_uris, script_uris


class JumpStartPretrainedEmbedding(Construct):
    def __init__(self, scope: Construct, id: str, jumpstart_artefacts, model_name, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        sm_role = iam.Role(self, "Role",assumed_by=iam.CompositePrincipal(iam.ServicePrincipal("sagemaker.amazonaws.com")))
        sm_role.add_managed_policy(iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess"))
        sm_role.add_managed_policy(iam.ManagedPolicy.from_aws_managed_policy_name("AmazonS3FullAccess"))
        self.model = sagemaker.CfnModel(self, "model",
                                   model_name=model_name,
                                   primary_container={
                                       "image": jumpstart_artefacts.image_uri,
                                       "modelDataUrl": jumpstart_artefacts.model_archive
                                       },
                                    execution_role_arn=sm_role.role_arn)

    @property
    def model_name(self):
        return self.model.model_name

class HFPretrainedEmbeddingModel(Construct):

    def __init__(self, scope: Construct, id: str, upload_bucket_name: str, hf_model_id: str, model_name: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        #First create a bucket populated with HF hub model parameters
        populated_bucket = PopulatedBucketResource(self, "PopulatedBucket", upload_bucket_name, hf_model_id, model_name)

        #Create SageMaker Role that has access to model artefacts
        sm_role = iam.Role(self, "Role",
                assumed_by=iam.CompositePrincipal(iam.ServicePrincipal("sagemaker.amazonaws.com")))
        
        sm_role.add_managed_policy(iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess"))

        #Grant access to S3 bucket
        populated_bucket.default_bucket.grant_read_write(sm_role)
        self.model = sagemaker.CfnModel(self, "model",
                                   model_name=model_name,
                                   primary_container={
                                       "image": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-cpu-py39-ubuntu20.04",
                                       "modelDataUrl": populated_bucket.model_artefacts_s3_uri
                                       },
                                    execution_role_arn=sm_role.role_arn
                                    )
        self.model.node.add_dependency(populated_bucket)

    @property
    def model_name(self):
        return self.model.model_name


class HFPretrainedEmbeddingModel(Construct):

    def __init__(self, scope: Construct, id: str, upload_bucket_name: str, hf_model_id: str, model_name: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        #First create a bucket populated with HF hub model parameters
        populated_bucket = PopulatedBucketResource(self, "PopulatedBucket", upload_bucket_name, hf_model_id, model_name)

        #Create SageMaker Role that has access to model artefacts
        sm_role = iam.Role(self, "Role",
                assumed_by=iam.CompositePrincipal(iam.ServicePrincipal("sagemaker.amazonaws.com")))
        
        sm_role.add_managed_policy(iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess"))

        #Grant access to S3 bucket
        populated_bucket.default_bucket.grant_read_write(sm_role)
        self.model = sagemaker.CfnModel(self, "model",
                                   model_name=model_name,
                                   primary_container={
                                       "image": "763104351884.dkr.ecr.eu-west-2.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-cpu-py39-ubuntu20.04",
                                       "modelDataUrl": populated_bucket.model_artefacts_s3_uri
                                       },
                                    execution_role_arn=sm_role.role_arn
                                    )
        self.model.node.add_dependency(populated_bucket)

    @property
    def model_name(self):
        return self.model.model_name

class FinetunedEmbedding(Construct):
    def __init__(self, scope: Construct, id: str, finetuned_artefacts, model_name, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        sm_role = iam.Role(self, "Role",assumed_by=iam.CompositePrincipal(iam.ServicePrincipal("sagemaker.amazonaws.com")))
        sm_role.add_managed_policy(iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess"))
        sm_role.add_managed_policy(iam.ManagedPolicy.from_aws_managed_policy_name("AmazonS3FullAccess"))
        self.model = sagemaker.CfnModel(self, "model",
                                   model_name=model_name,
                                   primary_container={
                                       "image": finetuned_artefacts["image_uri"],
                                       "modelDataUrl": finetuned_artefacts["model_archive"],
                                       "environment":{"pretrained_hf_model_id":finetuned_artefacts["pretrained_hf_model_id"]}
                                       },
                                    execution_role_arn=sm_role.role_arn)

    @property
    def model_name(self):
        return self.model.model_name
