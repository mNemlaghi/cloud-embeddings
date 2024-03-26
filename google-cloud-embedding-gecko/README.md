# What about Google Cloud Gecko Embeddings ?

This repository evaluates Gecko embeddings with MTEB. These embeddings provide pretty solid results, as shown below with MTEB on STS B.

## Embedding call.

Once configurations are done, main class is straight forward to implement.

```python
import numpy as np
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

class GeckoEncoder():
    def __init__(self, model_name):
        self.model_name = model_name
        print('fecthing model')
        self.model = TextEmbeddingModel.from_pretrained(self.model_name)

    def encode(self, sentences: list[str], n = 240,  **kwargs) -> list[np.ndarray] :
        res = []
        final = [sentences[i * n:(i + 1) * n] for i in range((len(sentences) + n - 1) // n )]
    
        for chunk in final:
            res.extend(self.batch_encode(chunk))
        return res

    def batch_encode(self, sentences: list[str],   **kwargs) -> list[np.ndarray] :
        inputs = []        
        for text in sentences:
            inputs.append(TextEmbeddingInput(task_type="SEMANTIC_SIMILARITY",text=text))
        embeddings = self.model.get_embeddings(inputs)
        return  [np.array(embedding.values) for embedding in embeddings]
```
I used `n=240` because of the 250 batch size limitation.

## Evaluate


### Create a VertexAI Service Account Key and download it to a folder


### Install the required packages

```bash
pip install -r requirements.txt
```

### Run `run_mteb.py`script to evaluate with MTEB

```bash
python run_mteb.py --region us-central1 --project mehdi-playground --service_account vertexmeftah.json
```

It will generate a `results` folder for each model tested.

### Run `mteb_meta.py`

It is taken from [MTEB scripts repository](https://github.com/embeddings-benchmark/mteb/blob/main/scripts/mteb_meta.py)

```bash
python mteb_meta.py ./results/textembedding-gecko@003
```

It will generate a `mteb_metadata.md` file.

