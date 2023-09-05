



import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from transformers import TrainingArguments, Trainer, logging
from tqdm import tqdm
import argparse


from peft import TaskType
from peft import LoraConfig, get_peft_model 
from datasets import load_dataset, Dataset

from torch.utils.data import DataLoader
from transformers import default_data_collator
import os
#import evaluate


#roc_auc_score = evaluate.load("roc_auc")

MAX_LENGTH=240


class EncoderForESCI(nn.Module):
    def __init__(self, pretrained_model, lora = True, normalize=True, lora_rank=8):
        super(EncoderForESCI, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.embedding_model = AutoModel.from_pretrained(pretrained_model)
        if lora:
            config = LoraConfig(
                r=lora_rank,
                lora_alpha=16,
                target_modules=["key","query", "value"],
                #target_modules=["encoder.layer.*"],
                bias="none",
                lora_dropout=0.05,
                inference_mode=False,
                task_type=TaskType.FEATURE_EXTRACTION
            )
        
            self.embedding_model = get_peft_model(self.embedding_model, config)
        self.normalize=normalize
        
    def forward(self, query_input_ids, query_attention_mask, product_input_ids, product_attention_mask,
                query_token_type_ids, product_token_type_ids, labels):
        q = {"input_ids":query_input_ids.squeeze(1), "attention_mask":query_attention_mask.squeeze(1),
            "token_type_ids":query_token_type_ids.squeeze(1)}
        p = {"input_ids":product_input_ids.squeeze(1), "attention_mask":product_attention_mask.squeeze(1),
            "token_type_ids":product_token_type_ids.squeeze(1)}
        
        q_emb_all = self.embedding_model(**q)
        p_emb_all = self.embedding_model(**p)
        
        q_emb = self.mean_pooling(q_emb_all, q["attention_mask"])
        p_emb = self.mean_pooling(p_emb_all, p["attention_mask"])
        #q_emb, p_emb = self.mean_pooling()), self.model(**p)
        
        
        if self.normalize:
            q_emb = torch.nn.functional.normalize(q_emb, p=2, dim=1)
            p_emb = torch.nn.functional.normalize(p_emb, p=2, dim=1)
        
        contig_labels=labels.reshape(-1).contiguous()
        return q_emb, p_emb, contig_labels
        
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    
def tokenize_examples_and_target(examples):
    queries = examples["query"]
    result = finetuned.tokenizer(queries, padding="max_length", max_length=MAX_LENGTH, truncation=True, return_tensors='pt')
    #result = finetuned.tokenizer(queries, return_tensors='pt')
    result = {f"query_{k}": v for k, v in result.items()}

    products = examples["product_title"]
    result_products = finetuned.tokenizer(products, padding="max_length", max_length=MAX_LENGTH, truncation=True, return_tensors='pt')
    #result_products = finetuned.tokenizer(products, return_tensors = 'pt')
    for k, v in result_products.items():
        result[f"product_{k}"] = v
        
    result["labels"] = torch.ByteTensor([examples["relevance_label"]])
    return result



emb_loss=nn.CosineEmbeddingLoss(reduction="mean")

class ESCITrainer(Trainer):  
    def compute_loss(self, model, inputs, return_outputs = False):
        outputs = model(**inputs)
        q_emb, p_emb, labels = outputs
        custom_loss =  emb_loss(q_emb, p_emb, 2*labels - 1)
        return (custom_loss, outputs) if return_outputs else custom_loss


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--eval_batch_size', type=int)

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--lora_rank', default=8, type=int)
    parser.add_argument('--lr', default=1e-6, type=float)

    
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    
        
    args, _ = parser.parse_known_args()

    default_args = {"output_dir": "tmp",
                    "evaluation_strategy": "steps",
                    "num_train_epochs": args.epochs,
                    "log_level": "error",
                    "report_to": "none"}

    
    finetuned = EncoderForESCI(args.pretrained_model, lora_rank=args.lora_rank)
    
    ds= load_dataset("smangrul/amazon_esci")
    processed_ds = ds.map(tokenize_examples_and_target,
                      num_proc=os.cpu_count(), 
                     remove_columns=ds['train'].column_names)
    
    
    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=4,
        fp16=True,
        remove_unused_columns=False, 
    **default_args,
    )

    trainer = ESCITrainer(model=finetuned, args=training_args, train_dataset=processed_ds['train'].with_format("torch")
                      , eval_dataset=processed_ds['validation'].with_format("torch"))
    trainer.train()

        
    ## Merge the models with trained Adapters, and save it to /opt/ml/code
    merged = finetuned.embedding_model.merge_and_unload()
    merged.save_pretrained(args.model_dir)
