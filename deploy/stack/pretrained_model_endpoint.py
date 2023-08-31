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



class PretrainedEmbeddingEndpointStack(Stack):

    def __init__(self, scope: Construct, id: str,  **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        ##Parameters
        upload_bucket_name = CfnParameter(self, "uploadBucketName", type="String",description="Bucket where models are stored", default="sm-pretrained-embedding-bucket")
        model_name= CfnParameter(self, "ModelName", type="String",default="cfn-pretrained-embedding", description="Name for Embedding" )

        embedding_provider = self.node.try_get_context("provider")
        if embedding_provider == "jumpstart":
            jumpstart_embeddings = get_jumpstart_embeddings_model_list(self.region)
            jumpstart_model_id = self.node.try_get_context("jumpstart_model_id")
            jumpstart_artefacts = JumpStartArtefacts(jumpstart_model_id, self.region)
            self.model= JumpStartPretrainedEmbedding(self, "JumpstartPretrainedModel", jumpstart_artefacts, model_name.value_as_string)
        elif embedding_provider == "huggingface":
            #First create a bucket populated with HF hub model parameters
            hf_model_id = self.node.try_get_context("hf_model_id")
            self.model = HFPretrainedEmbeddingModel(self, "HFPretrainedModel",  upload_bucket_name.value_as_string, hf_model_id, model_name.value_as_string)
        elif embedding_provider == "finetuned":
            finetuned_artefacts = dict()
            finetuned_artefacts["image_uri"]= self.node.try_get_context("image_uri")
            finetuned_artefacts["model_archive"]=self.node.try_get_context("model_archive")
            finetuned_artefacts["pretrained_hf_model_id"]=self.node.try_get_context("pretrained_hf_model_id")
            self.model = FinetunedEmbedding(self, "FinetunedModel", finetuned_artefacts, model_name.value_as_string)
        else:
            raise NotImplementedError

        #Finally creating Endpoint Config And Endpoint
        #cfn_endpoint_config = sagemaker.CfnEndpointConfig(self, "MyCfnEndpointConfig",
        #    endpoint_config_name=Fn.join("", [model.model_name, "-endpoint-config"]),
        #    production_variants=[sagemaker.CfnEndpointConfig.ProductionVariantProperty(
        #        initial_variant_weight=1,
        #        model_name=model.model_name,
        #        variant_name="AllTraffic",
        #        serverless_config=sagemaker.CfnEndpointConfig.ServerlessConfigProperty(
        #            max_concurrency=1,
        #            memory_size_in_mb=6144))])

        #cfn_endpoint_config.node.add_dependency(model)
        #
        #self.cfn_endpoint = sagemaker.CfnEndpoint(self, "MyCfnEndpoint", 
        #                                  endpoint_config_name=cfn_endpoint_config.endpoint_config_name,
        #                                  endpoint_name=Fn.join("", [model.model_name, "-endpoint"]))

        #self.cfn_endpoint.node.add_dependency(cfn_endpoint_config)
        #CfnOutput(scope=self, id=f"EndpointName", value=f"{self.cfn_endpoint.endpoint_name}")
