import pandas as pd
import numpy as np
from mteb import MTEB

from cohere_embeddings import cohere_experiment_scope
from titan_embeddings import titan_experiment_scope

np.random.seed(42)

## AWS BEDROCK MODELS
titan_models = ["amazon.titan-embed-text-v1", "amazon.titan-embed-image-v1"]
cohere_models = ["cohere.embed-multilingual-v3","cohere.embed-english-v3"]




class AbstractLabLabAssistant:
    def __init__(self, encoder_fns,  models, boto_client, core_task_name, metrics):
        self.client = boto_client
        self.models = models
        self.core_task_name = core_task_name
        self.metrics = metrics.split('.')

        self.encoder_fns = encoder_fns
        self.matrixes = {}


    def run(self, recap_pickle = "recap.pickle"):
        evaluations = []
        for model_id in self.models:
            for type, encoder_fn in self.encoder_fns.items():
                print(f"Handling {model_id}, quantization {type}")
                exp= self.run_generic_experiment(model_id=model_id, 
                    encoder_fn=encoder_fn,
                    core_task_name=self.core_task_name, 
                    boto_client=self.client, 
                    metrics = self.metrics)
                
                mx = exp.pop('matrix')
                if type=="classic":
                    self.matrixes[model_id]=mx
                evaluations.append(exp)
            
        final_df= pd.DataFrame(evaluations)
        final_df.to_pickle(recap_pickle)

        return final_df
    

    @staticmethod
    def run_generic_experiment(model_id, encoder_fn, core_task_name, boto_client,  metrics):
        encoder = AbstractLabLabAssistant.instantiate_encoder(model_id, encoder_fn,boto_client, core_task_name)
        encoder.calibrate_embeddings()

        evaluation = MTEB(tasks=[core_task_name], task_langs=["en"])
        res_all = evaluation.run(encoder, output_folder=f"results/{model_id}-{encoder.type}", eval_splits=["test"], overwrite_results=True)
        d = {"model_id":model_id, "dimension":encoder.my_dimension}

        perf = res_all[core_task_name]['test']

        for k in metrics:
            perf = perf[k]
            if isinstance(perf, dict):
                perf = perf.copy()
            elif isinstance(perf, float):
                break
            else:
                NotImplementedError

        d[k]=perf
        d["time"]=res_all[core_task_name]['test']['evaluation_time']
        d['type']=encoder.type
        d['size_in_bytes']= encoder.byte_size
        d['matrix'] = encoder.create_matrix()
        return d

    @staticmethod
    def instantiate_encoder(model_id, encoder_fn,boto_client, core_task_name):
        return encoder_fn(boto_client=boto_client, model_id = model_id, task_name = core_task_name+"test")


class Instantiator:
    @classmethod
    def instantiate_cohere_experiments(cls, boto_client):        
        return cls(encoder_fns=cohere_experiment_scope,
                         models=cohere_models, 
                         boto_client=boto_client)
    
    @classmethod
    def instantiate_titan_experiments(cls, boto_client):        
        return cls(encoder_fns=titan_experiment_scope,
                         models=titan_models, 
                         boto_client=boto_client)


##Having a good abstraction, now we can inherit simply by overloading.
class STSLabAssistant(AbstractLabLabAssistant, Instantiator):
    def __init__(self, encoder_fns, models, boto_client):
        super().__init__(encoder_fns, 
                         models, 
                         boto_client, 
                         core_task_name="STSBenchmark",
                         metrics ="cos_sim.spearman" )
    

class Banking77LabAssistant(AbstractLabLabAssistant, Instantiator):
    def __init__(self, encoder_fns, models, boto_client):
        super().__init__(encoder_fns, 
                         models, 
                         boto_client, 
                         core_task_name="Banking77Classification",
                         metrics ="accuracy" )
        