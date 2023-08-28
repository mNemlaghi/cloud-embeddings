#!/usr/bin/env python3
import os

import aws_cdk as cdk
from aws_cdk import App, Environment

from stack.pretrained_model_endpoint import PretrainedEmbeddingEndpointStack
from stack.api_gw_endpoint import SmApiGatewayStack
from stack.storage import EmbeddingStorageStack


app = App()
my_env=Environment(account=os.environ["CDK_DEFAULT_ACCOUNT"], region=os.environ["CDK_DEFAULT_REGION"]) 
endpoint_stack=PretrainedEmbeddingEndpointStack(app, "PretrainedHuggingFacModelEndpointeStack",env = my_env)
api_stack=SmApiGatewayStack(app, "PretrainedEmbeddingApiGwStack", endpoint=endpoint_stack.cfn_endpoint, env=my_env)
embeddingstorage_stack=EmbeddingStorageStack(app, "EmbeddingStorage", env=my_env)

app.synth()
