# Welcome to your MLOPS for embeddings!

This project indicates on how to create a state-of-the-art embedding SageMaker Serverless endpoint with CDK from HuggingFace Hub.


## The big picture

We are going to create 3 stacks:
1. `PretrainedHFEmbeddingEndpointeStack`: a stack with a serverless model endpoint that directly populates the model within a new S3 bucket. 
2. `PretrainedEmbeddingApiGwStack`:  stack with a linked Api Gateway.
3. `EmbeddingStorage`: stack with an RDS Instance on PostgreSQL, backed by PGVECTOR extension.


## Inputs

1. `uploadBucketName`: Bucket used for hosting model parameters. Default: "cfn-pretrained-embedding";
2. `hfModel`. The Embedding model from HuggingFace Hub. Default: "sentence-transformers/all-mpnet-base-v2", e.g. the very handy [MPNET](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) ;  
3. Your model name. Default: Default: "cfn-pretrained-embedding".

![globalStackEmbeddings](https://github.com/mNemlaghi/cloud-embeddings/assets/12110853/dbc1689a-f050-4925-a334-cee70e04eb36)


## Using CDK.

The `cdk.json` file tells the CDK Toolkit how to execute your app.

This project is set up like a standard Python project.  The initialization
process also creates a virtualenv within this project, stored under the `.venv`
directory.  To create the virtualenv it assumes that there is a `python3`
(or `python` for Windows) executable in your path with access to the `venv`
package. If for any reason the automatic creation of the virtualenv fails,
you can create the virtualenv manually.

To manually create a virtualenv on MacOS and Linux:

```
$ python3 -m venv .venv
```

After the init process completes and the virtualenv is created, you can use the following
step to activate your virtualenv.

```
$ source .venv/bin/activate
```

If you are a Windows platform, you would activate the virtualenv like this:

```
% .venv\Scripts\activate.bat
```

Once the virtualenv is activated, you can install the required dependencies.

```
$ pip install -r requirements.txt
```

At this point you can now synthesize the CloudFormation template for this code.

```
$ cdk synth
```

To add additional dependencies, for example other CDK libraries, just add
them to your `setup.py` file and rerun the `pip install -r requirements.txt`
command.

## Useful commands

 * `cdk ls`          list all stacks in the app
 * `cdk synth`       emits the synthesized CloudFormation template
 * `cdk deploy`      deploy this stack to your default AWS account/region
 * `cdk diff`        compare deployed stack with current state
 * `cdk docs`        open CDK documentation

Enjoy!

## Creating the pretrained stack
Depending on your intention, you might want to push one of the three aforementioned stacks.

To deploy everything fast without review (⚠️️ ️be️war️e of️ the ️charge️s ⚠️)

```
$ cdk deploy --all --require-approval never --concurrency 2
```

If you want to deploy the endpoint stack, perform:

```
$ cdk deploy PretrainedHFEmbeddingEndpointeStack
```

Should you want to deploy the Endpoint and the Api Gateway API, perform:

```
$ cdk deploy PretrainedHFEmbeddingEndpointeStack && cdk deploy PretrainedEmbeddingApiGwStack
```

If you want to deploy only the RDS instance with PGVector, perform:

```
$ cdk deploy EmbeddingStorage
```
