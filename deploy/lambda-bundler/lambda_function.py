import json
from helpers import HfModelBuilder
import cfnresponse

def handler(event, context):
    print(event)
    print(context)
    responseData = {}
    hf_model_id = event['hf_model_id']
    model_name = event.get('model_name', None)
    bucket_name = event['bucket_name']
    s3_uri= HfModelBuilder.run_from_hf_model_id(hf_model_id, bucket_name)
    responseData['message']='Done'
    responseData['s3Uri']=s3_uri
    responseData['Status']="SUCCESS"
    #cfnresponse.send(event, context, cfnresponse.SUCCESS, responseData)
    return json.dumps(responseData)
