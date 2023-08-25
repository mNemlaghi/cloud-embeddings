#!/usr/bin/env python3
import os

import aws_cdk as cdk
from aws_cdk import App, Environment

from stack.pretrained_model_endpoint import PretrainedEmbeddingEndpointStack


app = App()
my_env=Environment(account=os.environ["CDK_DEFAULT_ACCOUNT"], region=os.environ["CDK_DEFAULT_REGION"]) 
PretrainedEmbeddingEndpointStack(app, "PretrainedHuggingFaceStack",env = my_env)

app.synth()
