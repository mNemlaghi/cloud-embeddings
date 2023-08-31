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

from construct.pretrained_embedding import HFPretrainedEmbeddingModel, JumpStartPretrainedEmbedding, FinetunedEmbedding
from utils.jumpstart_uris import get_jumpstart_embeddings_model_list, JumpStartArtefacts

from constructs import Construct



class ServerlessEndpoint(Stack):

    def __init__(self, scope: Construct, id: str, model: sagemaker.CfnModel,  **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

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
        
        self.cfn_endpoint = sagemaker.CfnEndpoint(self, "MyCfnEndpoint", 
                                          endpoint_config_name=cfn_endpoint_config.endpoint_config_name,
                                          endpoint_name=Fn.join("", [model.model_name, "-endpoint"]))

        self.cfn_endpoint.node.add_dependency(cfn_endpoint_config)
        CfnOutput(scope=self, id=f"EndpointName", value=f"{self.cfn_endpoint.endpoint_name}")
