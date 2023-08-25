
import argparse
import os
from mteb import MTEB
from sentence_transformers import SentenceTransformer
from mteb.tasks import STSBenchmarkSTS

def stsb_mteb_evaluate_model(model, output_folder)->None:
    evaluation = MTEB(tasks=[STSBenchmarkSTS(langs=["en"])], task_langs=['en'])
    results = evaluation.run(model, output_folder=output_folder, eval_splits=['test'])
    return results

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name")
    os.path.join("/opt/ml/processing/evaluation")
    args, _ = parser.parse_known_args()
    print("Received arguments {}".format(args))
    output_path_folder = "/opt/ml/processing/eval/"
    model = SentenceTransformer(args.model_name, output_path_folder)
    res = stsb_mteb_evaluate_model(model, output_path_folder)
