import json
from helpers import HfModelBuilder

def handler(event, context):
    responseData = {}
    hf_model_id = event['hf_model_id']
    model_name = event['model_name']
    bucket_name = event['bucket_name']
    s3_uri= HfModelBuilder.run_from_hf_model_id(hf_model_id, bucket_name, model_name)
    responseData['message']='Done'
    responseData['s3uri']=s3_uri
    responseData['Status']="SUCCESS"
    print(responseData)
    return responseData
