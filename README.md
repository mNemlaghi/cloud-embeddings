# ☁️ Cloud Embeddings: evaluate, finetune, deploy and store state-of-the-art pretrained embeddings 🔢

A repository for tackling cloud text pre-trained embeddings, from evaluation to deployment, including fine-tuning and vector stores, with an AWS cloud lens, with pretrained HuggingFace 🤗 embeddings and AWS.

A series of blog posts is coming soon to give more contexts to this part 👷🏻 

## Why do embeddings matter?

They're at the backbone of multiple ML systems we encounter every day; plus, as LLM encounter increasing popularity, the use-case of retrieval augmented generation (RAG) is a professional use of GenAI that heavily relies on embeddings.

## Objective of this repository.

* This collection of code showcases an end-to-end guide from selection, evaluation, finetuning, deployment to storage and retrieval embeddings with a cloud lens.
* This collection will hopefully allow you to get some perspective on every step, allowing your organization to embark in a seamless embedding journey, ensuring production readiness at every step.

## Repository structure

### Evaluate
Thanks to the [evaluation](evaluate/) part of this repository, you can evaluate SOTA embeddings with [MTEB](https://huggingface.co/blog/mteb) and SageMaker Processing according to your needs and delights.

### Finetune

[Finetune](finetune/) part is about modern, LoRA finetuning embeddings with 🤗 HuggingFace and SageMaker Training.

### Deploy

[Deployment](deploy/). Automated pretrained embedding deployment with AWS CDK and SageMaker Model Hosting, in a Serverless way.

![Stacks proposition](https://github.com/mNemlaghi/cloud-embeddings/assets/12110853/e177f369-3276-4c3c-9cd4-1b4966309db6)


### Store

Deployment contains a ready-made DB instance with RDS!

🚧🚧 In construction 🚧🚧
We'll add resources on how to store and retrieve data, but you can already find excellent resources [in here](https://aws.amazon.com/fr/blogs/database/building-ai-powered-search-in-postgresql-using-amazon-sagemaker-and-pgvector/) for instance.
 

