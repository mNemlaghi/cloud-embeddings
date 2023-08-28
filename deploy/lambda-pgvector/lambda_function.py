import os
import json
import boto3
import psycopg2

##Retrieve  DB secrets
sm_client = boto3.client('secretsmanager', endpoint_url = os.environ['SECRETS_MANAGER_ENDPOINT'])
secrets=json.loads(sm_client.get_secret_value(SecretId='postgres-credentials')['SecretString'])

def handler(event, context):
    print("connecting")
    #connection = psycopg2.connect(database=secrets.get("engine"), user=secrets.get("username"), password=secrets.get("password"), host=os.environ["RDS_HOST"], port=str(secrets.get("port")))
    connection = psycopg2.connect(database=secrets.get("engine"), user=secrets.get("username"), password=secrets.get("password"), host=secrets.get("host"), port=str(secrets.get("port")))
    cursor = connection.cursor()
    cursor.execute("CREATE EXTENSION vector;")
    connection.commit()
    print("creating extension")
    return json.dumps({"statusCode":"200"})
