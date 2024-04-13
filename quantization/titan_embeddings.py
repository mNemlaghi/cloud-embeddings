import json
import numpy as np
from joblib import Parallel, delayed
from abstract_embedder import AbstractAWSEmbedder, BinaryEncoder, ScalarEncoder, TernaryEncoder
from abstract_embedder import TernaryRotatedEncoder,BinaryRotatedEncoder, RotatedScalarEncoder

class TitanEmbedder(AbstractAWSEmbedder):
    def __init__(self, boto_client, model_id = "amazon.titan-embed-text-v1", task_name = "default", precomputed=False, embeddings_path="precomputed"):
        super().__init__(boto_client=boto_client, 
                         model_id=model_id, 
                         task_name=task_name, 
                         precomputed=precomputed , 
                         embeddings_path=embeddings_path, 
                         provider="amazon", 
                         batch_size = 5)
        
    def single_encode(self, text, **kwargs):
        res = self.client.invoke_model(modelId = self.model_id, body=json.dumps({"inputText":text}) )
        r=json.loads(res['body'].read().decode('utf8'))
        return r['embedding']
    
    def batch_encode(self, sentences, **kwargs):
        parallel = Parallel(n_jobs=-1, prefer = "threads")(delayed(self.single_encode)(s) for s in sentences)
        embeddings = [np.array(embedding).astype(np.float32) for embedding in parallel]
        return embeddings
    

class BinaryTitanEncoder(TitanEmbedder, BinaryEncoder):
    pass

class ScalarTitanEncoder(TitanEmbedder, ScalarEncoder):
    pass

class TernaryTitanEncoder(TitanEmbedder, TernaryEncoder):
    pass

class TernaryRotatedTitanEncoder(TitanEmbedder, TernaryRotatedEncoder):
    pass

class BinaryRotatedTitanEncoder(TitanEmbedder, BinaryRotatedEncoder):
    pass

class ScalarRotatedTitanEncoder(TitanEmbedder, RotatedScalarEncoder):
    pass

