# Welcome to your MLOPS for embeddings!

## Objective

This subdirectory is part of the `cloud-embeddings` repository. This section indicates on how to create a state-of-the-art embedding with Amazon SageMaker Serverless endpoint with CDK from HuggingFace Hub, alongside with a pre-configured RDS instance with PgVector installed.


## The big picture

You can create up to 4 stacks
1. `EmbeddingModelStack`: your embedding model
2. `EmbeddingEndpointStack`: a serverless endpoint hosting your embedding model
3. `PretrainedEmbeddingApiGwStack`:  stack with a linked Api Gateway.
4. `EmbeddingStorage`: stack with an RDS Instance on PostgreSQL, backed by PGVECTOR extension.

![Stacks proposition](https://github.com/mNemlaghi/cloud-embeddings/assets/12110853/e177f369-3276-4c3c-9cd4-1b4966309db6)

Then, we'll have everything set up for numerous use-cases.

## Flavors

I'm currently working on 3 types of models.

### Pretrained models from JumpStart

[SageMaker Jumpstart](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-jumpstart.html) provides you with a wide range of pretrained models and solutions, that you can already deploy, in a click-button manner! It removes the hassle of deploying models manually or losing time in configuring options. The good news is that we can also deploy models JumpStart from the CDK !

### Pretrained models from HuggingFace Hub

[HuggingFace](https://huggingface.co/) 🤗 is a stunning initiative that develops tools for machine learning models, mostly in open-source flavor. Amongst other features, they provide a hub where you can easily find an open-source model suited to your needs, including state-of-the-art embeddings!

### Your finetuned models !

You can put your already finetuned models and deploy them with a Serverless endpoint!
🚧🚧 In construction 🚧🚧


## Great, how can I do that ?

After installing and bootstrapping CDK (see below), you might need to change context fields depending on your need

### Context

* For Jumpstart embeddings, `provider` is equal to `jumpstart`. Furthermore, you'll need to provide:
    - `jumpstart_model_id`: the model id from jumpstart 
* For HuggingFace Hub, `provider` is equal to `huggingface`. Furthermore, you'll need to provide:
    - `hf_model_id`: the model id from HuggingFace Hub
* For your own finetuned embeddings, `provider` is equal to `finetuned`. Furthermore, you'll need to provide:
    -  `image_uri`: ECR image URI of your desired model
    -  `pretrained_hf_model_id`: the _pretrained_ model id from HuggingFace Hub from which you finetuned
    -  `model_archive` : S3 URI of the model artefacts


### Parameters (optional)
1. `uploadBucketName`: Bucket used for hosting model parameters. Default: "cfn-pretrained-embedding";
2. `ModelName`: your  model name. Default: Default: "cfn-pretrained-embedding".


### Creating the pretrained stack

Depending on your intention, you might want to push one of the three aforementioned stacks.

To deploy everything fast without review (⚠️️ ️be️war️e of️ the ️charge️s ⚠️)

```
$ cdk deploy --all --require-approval never --concurrency 2
```

If you want to deploy the endpoint stack, perform:

```
$ cdk deploy EmbeddingEndpointStack
```

Should you want to deploy the Endpoint and the Api Gateway API, perform:

```
$ cdk deploy PretrainedHFEmbeddingEndpointeStack && cdk deploy PretrainedEmbeddingApiGwStack
```

If you want to deploy only the RDS instance with PGVector, perform:

```
$ cdk deploy EmbeddingStorage
```

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


