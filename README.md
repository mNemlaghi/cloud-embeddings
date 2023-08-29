# Cloud Embeddings: evaluate, finetune, deploy and store state-of-the-art pretrained embeddings

A repository for tackling cloud text pre-trained embeddings, from evaluation to deployment, including fine-tuning and vector stores, with an AWS cloud lens, with pretrained HuggingFace 🤗 embeddings and AWS.

A series of blog posts is coming soon to give more contexts to this part 👷🏻 

## Why do embeddings matter?

They're at the backbone of multiple ML systems we encounter every day; plus, as LLM encounter increasing popularity, the use-case of retrieval augmented generation (RAG) is a professional use of GenAI that heavily relies on embeddings.

## Objective of this repository.

* This collection of code showcases an end-to-end guide from selection, deployment to storage and retrieval embeddings with a cloud lens.
* This collection will hopefully allow you to get some perspective on every step froù seamlessly embark your organization in a seamless embedding journey, ensuring production readiness at every step.

## Repository structure

### Evaluate
Evaluate SOTA embeddings with [MTEB](https://huggingface.co/blog/mteb) and SageMaker Processing according to your needs and delights.

### Finetune

[Finetune](finetune/) is about modern, LoRA finetuning embeddings with 🤗 HuggingFace and SageMaker Training.

### Deploy

Automated pretrained embedding deployment with AWS CDK and SageMaker Model Hosting, in a Serverless way.

![Stacks proposition](https://github.com/mNemlaghi/cloud-embeddings/assets/12110853/dbc1689a-f050-4925-a334-cee70e04eb36)


### Store
__TO DO__
 

