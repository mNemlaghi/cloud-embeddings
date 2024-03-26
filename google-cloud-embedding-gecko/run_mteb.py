import vertexai
from gecko_embedding import GeckoEncoder
from argparse import ArgumentParser
from mteb import MTEB
from google.oauth2 import service_account

TASK_LIST_STS = [
    "BIOSSES",
 #   "SICK-R",
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STS16",
    "STS17",
    "STS22 (en)",
    "STSBenchmark",
]

class EvaluationRunner:
    def __init__(self,model_name, task_list=TASK_LIST_STS):
        self.task_list = task_list
        self.model_name = model_name
        self.encoder = GeckoEncoder(self.model_name)

    def run_task(self, task, model_name):
        print("Running task: ", task)
        eval_splits = ["validation"] if task == "MSMARCO" else ["test"]
        evaluation = MTEB(tasks=[task], task_langs=["en"])
        evaluation.run(self.encoder, output_folder=f"results/{self.model_name}", eval_splits=eval_splits, overwrite_results=True)

    def run(self):
        for task in self.task_list:
            self.run_task(task, self.model_name)

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="textembedding-gecko@003", help="GCP Model id")
    parser.add_argument("--project_id", type=str)
    parser.add_argument("--region", type=str)
    parser.add_argument("--service_account", type=str)
    parser.add_argument("--output_dir", type=str, default="./result")
    args = parser.parse_args()
    print("initiating")
    credentials = service_account.Credentials.from_service_account_file(args.service_account)

    vertexai.init(project=args.project_id, location=args.region, credentials=credentials)
    print("Let's go")
    runner = EvaluationRunner(model_name=args.model_name, task_list=TASK_LIST_STS)
    runner.run()