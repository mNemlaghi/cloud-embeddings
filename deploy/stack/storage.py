from aws_cdk import (
    Stack,
    CfnParameter,
    aws_lambda,
    aws_iam as iam,
    aws_rds as rds,
    aws_ec2 as ec2,
    Fn,
    Duration,
    RemovalPolicy,
    CfnOutput
)

import os
from utils.hf_model_data import PopulatedBucketResource

from constructs import Construct



class EmbeddingStorageStack(Stack):

    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id,  **kwargs)
        
        self.vpc = ec2.Vpc(
            self,
            id="minimalVPC",
            vpc_name="MinimumVPC",
            #cidr="10.0.0.0/16",
            max_azs=2,
            nat_gateways=1,
            subnet_configuration=[
                ec2.SubnetConfiguration(name="public", cidr_mask=24,reserved=False, subnet_type=ec2.SubnetType.PUBLIC),
                ec2.SubnetConfiguration(name="private", cidr_mask=24,reserved=False, subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
                ec2.SubnetConfiguration(name="isolated", cidr_mask=24,reserved=False, subnet_type=ec2.SubnetType.PRIVATE_ISOLATED)
                ],
            enable_dns_hostnames=True,
            enable_dns_support=True
        )

        lambda_sg = ec2.SecurityGroup(self, "LambdaSecGroup", vpc = self.vpc, allow_all_outbound=True)
        db_connection_sg= ec2.SecurityGroup(self, "DBSecGroup", vpc = self.vpc, allow_all_outbound=False)
        db_connection_sg.add_ingress_rule(peer=db_connection_sg, connection=ec2.Port.tcp(5432), description="allow lambda connection")
        db_connection_sg.add_ingress_rule(peer=lambda_sg, connection=ec2.Port.tcp(5432), description="allow lambda connection")


        db_secrets = rds.DatabaseSecret(self, 'postgres-secret',
            username='postgres',
            secret_name='postgres-credentials'
            )

        # Create the database
        db = rds.DatabaseInstance(self, "db",
            engine=rds.DatabaseInstanceEngine.postgres(version=rds.PostgresEngineVersion.VER_15_2), #Check Support ! 
            instance_type=ec2.InstanceType.of(ec2.InstanceClass.BURSTABLE3, ec2.InstanceSize.MICRO),
            credentials=rds.Credentials.from_secret(db_secrets),
            vpc=self.vpc,
            deletion_protection=False,
            removal_policy=RemovalPolicy.DESTROY,
            security_groups= [db_connection_sg],
            publicly_accessible=False,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_ISOLATED))

        #db.connections.allow_default_port_from_any_ipv4()
        proxy = db.add_proxy(id+"-proxy", secrets = [db_secrets], debug_logging=True, vpc = self.vpc, require_tls = False, security_groups = [db_connection_sg])
        ecr_image = aws_lambda.EcrImageCode.from_asset_image(directory = os.path.join(os.getcwd(), "lambda-pgvector"), asset_name="InitializePgVectorDB")
        #Create a lambda function and invoke it
        self.lambda_fn = aws_lambda.Function(self,
            id= "lambdaContainerFunction2",
            description   = "Sample Lambda Container Function",
            code = ecr_image,
            handler = aws_lambda.Handler.FROM_IMAGE,
            runtime = aws_lambda.Runtime.FROM_IMAGE,
            function_name = "InitializePgVectorDbRuntime", 
            timeout = Duration.seconds(30),
            vpc = self.vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
            allow_public_subnet=True,
            security_groups = [lambda_sg],
            environment = {"RDS_HOST":proxy.endpoint}
        )
        db_secrets.grant_read(self.lambda_fn)
