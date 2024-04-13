import json
import numpy as np

from abstract_embedder import AbstractAWSEmbedder, BinaryEncoder, ScalarEncoder, TernaryEncoder
from abstract_embedder import TernaryRotatedEncoder,BinaryRotatedEncoder, RotatedScalarEncoder


class CohereEncoder(AbstractAWSEmbedder):

    def __init__(self, boto_client, model_id = "cohere.embed-english-v3", task_name = "default", precomputed=False, embeddings_path="precomputed"):
        super().__init__(boto_client=boto_client, 
                         model_id=model_id, 
                         task_name=task_name, 
                         precomputed=precomputed , 
                         embeddings_path=embeddings_path, 
                         provider="cohere", 
                         batch_size = 90)
    
        
    def batch_encode(self, sentences, **kwargs):
        body = {"texts":sentences, "truncate":"NONE",  "input_type":"search_query"}
        res = self.client.invoke_model(modelId = self.model_id, body=json.dumps(body) )
        response = json.loads(res['body'].read().decode())
        embeddings = [np.array(embedding).astype(np.float32) for embedding in response['embeddings']]
        return embeddings
    

class BinaryCohereEncoder(CohereEncoder, BinaryEncoder):
    pass

class ScalarCohereEncoder(CohereEncoder, ScalarEncoder):
    pass

class TernaryCohereEncoder(CohereEncoder, TernaryEncoder):
    pass

class TernaryRotatedCohereEncoder(CohereEncoder, TernaryRotatedEncoder):
    pass

class BinaryRotatedCohereEncoder(CohereEncoder, BinaryRotatedEncoder):
    pass

class ScalarRotatedCohereEncoder(CohereEncoder, RotatedScalarEncoder):
    pass