import boto3
import json
import tempfile
from sagemaker import image_uris, model_uris, script_uris


def get_jumpstart_embeddings_model_list(aws_region):
    with tempfile.NamedTemporaryFile() as tmp:
        boto3.client("s3").download_file(f"jumpstart-cache-prod-{aws_region}", "models_manifest.json", tmp.name)
        with open(tmp.name, "rb") as json_file:
            model_list = json.load(json_file)

    # filter-out all the Text Embedding models from the manifest list.
    text_embedding_models = []
    for model in model_list:
        model_id = model["model_id"]
        if (("textembedding" in model_id) or ("tcembedding" in model_id)) and model_id not in text_embedding_models:
            text_embedding_models.append(model_id)
    return text_embedding_models



class JumpStartArtefacts:
    def __init__(self, jumpstart_value: str, aws_region, processor='cpu', **kwargs) -> None:
        self.model_archive = model_uris.retrieve(model_id=jumpstart_value, model_version='*', model_scope='inference')
        typical_instance = "ml.m5.xlarge" if processor=='cpu' else 'ml.g4dn.xlarge'
        self.image_uri = image_uris.retrieve(model_id=jumpstart_value, model_version='*', framework=None, region = aws_region, image_scope = "inference", instance_type=typical_instance)
