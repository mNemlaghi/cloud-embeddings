from aws_cdk import (
    Stack,
    CfnParameter,
    aws_lambda,
    aws_iam as iam,
    aws_rds as rds,
    aws_ec2 as ec2,
    Fn,
    Duration,
    CfnOutput
)

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
            nat_gateways=0,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="public", cidr_mask=24,
                    reserved=False, subnet_type=ec2.SubnetType.PUBLIC),
                ec2.SubnetConfiguration(
                    name="private", cidr_mask=24,
                    reserved=False, subnet_type=ec2.SubnetType.PRIVATE_ISOLATED)
				],
            enable_dns_hostnames=True,
            enable_dns_support=True
        )
        db_secrets = rds.DatabaseSecret(self, 'postgres-secret',
            username='postgres',
            secret_name='postgres-credentials'
            )

        # Create the database
        db = rds.DatabaseInstance(self, "db",
            engine=rds.DatabaseInstanceEngine.postgres(version=rds.PostgresEngineVersion.VER_13_7), #Check Support ! 
            instance_type=ec2.InstanceType.of(ec2.InstanceClass.BURSTABLE3, ec2.InstanceSize.MICRO),
            credentials=rds.Credentials.from_secret(db_secrets),
            vpc=self.vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_ISOLATED)
        )

        db.connections.allow_default_port_from_any_ipv4()
