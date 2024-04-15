import json
import numpy as np

from abstract_embedder import AbstractAWSEmbedder, BinaryEncoder, ScalarEncoder, TernaryEncoder, QuaternaryEncoder
from abstract_embedder import TernaryRotatedEncoder,BinaryRotatedEncoder, RotatedScalarEncoder, QuaternaryRotatedEncoder


class CohereEncoder(AbstractAWSEmbedder):

    def __init__(self, boto_client, model_id = "cohere.embed-english-v3", task_name = "default",  embeddings_path="precomputed"):
        super().__init__(boto_client=boto_client, 
                         model_id=model_id, 
                         task_name=task_name, 
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



#### For convenience during importation, we create a dict of classes.

cohere_encoders_scope = [CohereEncoder,
                         ScalarCohereEncoder,
                         BinaryCohereEncoder,
                         TernaryCohereEncoder,
                         TernaryRotatedCohereEncoder,
                         ScalarRotatedCohereEncoder,
                         BinaryRotatedCohereEncoder]

cohere_experiment_scope = {v.type:v for v in cohere_encoders_scope}