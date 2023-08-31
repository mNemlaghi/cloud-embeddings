#!/usr/bin/env python3
import os

import aws_cdk as cdk
from aws_cdk import App, Environment

from stack.pretrained_model_endpoint import PretrainedEmbeddingEndpointStack
from stack.api_gw_endpoint import SmApiGatewayStack
from stack.serverless_endpoint import ServerlessEndpoint
from stack.storage import EmbeddingStorageStack
from utils.jumpstart_uris import get_jumpstart_embeddings_model_list, JumpStartArtefacts


app = App()
my_env=Environment(account=os.environ["CDK_DEFAULT_ACCOUNT"], region=os.environ["CDK_DEFAULT_REGION"]) 
provider = app.node.try_get_context("provider")

model_stack=PretrainedEmbeddingEndpointStack(app, "EmbeddingModelStack", env = my_env)
endpoint_stack=ServerlessEndpoint(app, "EmbeddingEndpointStack", model = model_stack.model, env = my_env)
api_stack=SmApiGatewayStack(app, "PretrainedEmbeddingApiGwStack", endpoint=endpoint_stack.cfn_endpoint, env=my_env)
embedding_storage_stack=EmbeddingStorageStack(app, "EmbeddingStorage", env=my_env)

app.synth()
