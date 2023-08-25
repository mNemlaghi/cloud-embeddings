
from huggingface_hub import hf_hub_download
import fasttext
from mteb import MTEB
import string
import torch
import numpy as np


class NaiveAvgFastTextModel():
    def encode(self, sentences, batch_size=32, **kwargs):
        """ Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """
        model_path = hf_hub_download(repo_id="facebook/fasttext-en-vectors", filename="model.bin")
        self.ftmodel = fasttext.load_model(model_path)
        res= []
        for sentence in sentences:
            unpunkt_sentence = sentence.translate(str.maketrans('', '', string.punctuation)).lower()
            res.append(self.ftmodel.get_sentence_vector(unpunkt_sentence))
        return res     

if __name__=='__main__':
    output_path_folder = "/opt/ml/processing/eval/"
    model = NaiveAvgFastTextModel()
    evaluation = MTEB(tasks=["STSBenchmark"])
    evaluation.run(model, eval_splits=["test"], output_folder=output_path_folder)
