

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import argparse


from peft import TaskType
from peft import LoraConfig, get_peft_model 
from datasets import load_dataset, Dataset

from torch.utils.data import DataLoader
from transformers import default_data_collator
import os


MAX_LENGTH=70


class EncoderForESCI(nn.Module):
    def __init__(self, pretrained_model, lora = True, normalize=True, lora_rank=8):
        super(EncoderForESCI, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.model = AutoModel.from_pretrained(pretrained_model)
        if lora:
            config = LoraConfig(
                r=lora_rank,
                lora_alpha=32,
                #target_modules=["key","query", "value"],
                #target_modules=["encoder.layer.*"],
                target_modules=None,
                bias="none",
                lora_dropout=0.05,
                inference_mode=False,
                task_type=TaskType.FEATURE_EXTRACTION
            )
        
            self.model = get_peft_model(self.model, config)
        self.normalize=normalize
        
    def forward(self, **kwargs):
        model_output = self.model(**kwargs)
        
        embeddings = self.mean_pooling(model_output, kwargs["attention_mask"])
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings
      
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable: {trainable_params} || all: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def tokenize_examples_and_target(examples):
    queries = examples["query"]
    result = finetuned.tokenizer(queries, padding="max_length", max_length=MAX_LENGTH, truncation=True, return_tensors='pt')
    result = {f"query_{k}": v.reshape(-1) for k, v in result.items()}

    products = examples["product_title"]
    result_products = finetuned.tokenizer(products, padding="max_length", max_length=MAX_LENGTH, truncation=True, return_tensors='pt')
    for k, v in result_products.items():
        result[f"product_{k}"] = v.reshape(-1)

    result["labels"] = torch.ByteTensor([examples["relevance_label"]]).reshape(-1)
    return result


def full_forward_pass(batch, finetuned):
    q = {k.replace("query_", ""):v.to('cuda') for k,v in batch.items() if 'query_' in k} 
    p = {k.replace("product_", ""):v.to('cuda') for k,v in batch.items() if 'product_' in k} 

    q_emb, p_emb = finetuned(**q), finetuned(**p)
    labels=batch['labels'].reshape(-1).to('cuda')
    return q_emb, p_emb, labels

emb_loss=nn.CosineEmbeddingLoss(reduction="mean")
def get_loss(q_e, p_e, labels):
    #In cosine loss, targets are 1 or -1, so we transform our 0/1 labels
    return emb_loss(q_e, p_e, 2*labels - 1)
    #return torch.square(emb_loss(q_e, p_e, 2*labels - 1)) 

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

    
    finetuned = EncoderForESCI(args.pretrained_model, lora_rank=args.lora_rank)
    
    ds= load_dataset("smangrul/amazon_esci")
    processed_ds = ds.map(tokenize_examples_and_target,
                      num_proc=os.cpu_count(), 
                     remove_columns=ds['train'].column_names)
    
    train_dataloader = DataLoader(
        processed_ds['train'],
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=args.batch_size,
        pin_memory=True)

    validation_dataloader = DataLoader(
        processed_ds['validation'],
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=args.eval_batch_size,
        pin_memory=True)
    
    starting_epoch=0
    finetuned.to('cuda')
    optimizer = torch.optim.AdamW(finetuned.parameters(), lr=args.lr)
    for epoch in range(starting_epoch, args.epochs):
    
        finetuned.train()
        running_loss = 0.0
        for step, batch in enumerate(tqdm(train_dataloader)):            

            q_emb, p_emb, labels = full_forward_pass(batch, finetuned)        
            loss = get_loss(q_emb, p_emb,labels)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.detach().item()
            if step%100==0:
                mean_train_loss= running_loss / (step+1)
                print(f"Epoch {epoch+1}, step {step+1} ==> training loss {mean_train_loss}")


        mean_train_loss= running_loss / (step+1)
        print(f"Epoch {epoch+1}, step {step+1} ==> training loss {mean_train_loss}")

        finetuned.eval()
        eval_loss=0.0
        for step, batch in enumerate(tqdm(validation_dataloader)):
            with torch.no_grad():
                q_emb, p_emb, labels = full_forward_pass(batch, finetuned)
                loss = get_loss(q_emb, p_emb,labels)
            eval_loss+=loss.detach().item()
            if step%100==0:
                mean_valid_loss= eval_loss / (step+1)
                print(f"Epoch {epoch+1}, step {step+1} ==> eval loss {mean_valid_loss}")


        mean_valid_loss= eval_loss / (step+1)
        print(f"Finished Epoch {epoch+1}, step {step+1} ==> eval loss {mean_valid_loss}")
    
    ## Merge the models with trained Adapters, and save it to /opt/ml/code
    merged = finetuned.model.merge_and_unload()
    merged.save_pretrained(args.model_dir)
    
