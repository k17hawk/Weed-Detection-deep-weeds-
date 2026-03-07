import json
import time
import base64
from pathlib import Path
from kafka import KafkaProducer
import boto3
from botocore.exceptions import ClientError
from weed_detection import logger
from weed_detection.entity.config_entity import KafkaProducerConfig
from weed_detection.config.configuration import ConfigurationManager
import os

class KafkaDataProducer:
    def __init__(self, config: KafkaProducerConfig):
        self.config = config
        self.producer = None
        self.sqs_client = None
        
    def initialize(self):
        """Initialize Kafka producer and SQS client"""
        logger.info("🚀 Initializing Kafka Producer...")
        
        # Initialize Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=[self.config.bootstrap_servers],
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            acks='all',
            retries=3
        )
        
        # Initialize SQS client
        self.sqs_client = boto3.client(
            'sqs',
            region_name=self.config.aws_region,
            aws_access_key_id=self.config.aws_access_key_id,
            aws_secret_access_key=self.config.aws_secret_access_key
        )
        
        logger.info(f"✅ Kafka Producer initialized for topic: {self.config.topic}")
        logger.info(f"📋 Listening to SQS queue: {self.config.queue_url}")
        
    def download_from_s3(self, bucket, key):
        """Download file from S3"""
        try:
            s3 = boto3.client(
                's3',
                region_name=self.config.aws_region,
                aws_access_key_id=self.config.aws_access_key_id,
                aws_secret_access_key=self.config.aws_secret_access_key
            )
            
            logger.info(f"⬇️ Downloading s3://{bucket}/{key}")
            response = s3.get_object(Bucket=bucket, Key=key)
            content = response['Body'].read()
            
            # Encode to base64 for transmission
            encoded = base64.b64encode(content).decode('utf-8')
            logger.info(f"✅ Downloaded {len(content)} bytes")
            
            return encoded, len(content)
        except Exception as e:
            logger.error(f"❌ Failed to download from S3: {str(e)}")
            return None, 0
    

    def process_sqs_message(self, sqs_message):
        try:
            body = json.loads(sqs_message['Body'])
            logger.info(f"🔍 SQS body keys: {list(body.keys())}")

            bucket = None
            key    = None

            # Format 1: Lambda sends 'bucket' + 'file_key'  ← YOUR CASE
            if 'bucket' in body and 'file_key' in body:
                bucket = body['bucket']
                key    = body['file_key']
                logger.info(f"📨 Lambda message → s3://{bucket}/{key}")

            # Format 2: direct bucket + key
            elif 'bucket' in body and 'key' in body:
                bucket = body['bucket']
                key    = body['key']

            # Format 3: SNS-wrapped
            elif 'Message' in body:
                try:
                    inner = json.loads(body['Message'])
                    if 'Records' in inner:
                        record = inner['Records'][0]
                        bucket = record['s3']['bucket']['name']
                        key    = record['s3']['object']['key']
                except (json.JSONDecodeError, KeyError):
                    pass

            # Format 4: raw S3 Records
            elif 'Records' in body:
                record = body['Records'][0]
                bucket = record['s3']['bucket']['name']
                key    = record['s3']['object']['key']

            kafka_message = {
                'timestamp'      : time.time(),
                'sqs_message_id' : sqs_message['MessageId'],
                'source'         : 'sqs_bridge',
                'file_name'      : body.get('file_name', ''),
                'file_type'      : body.get('file_type', ''),
                'file_size'      : body.get('file_size', 0),
                'event_time'     : body.get('event_time', ''),
                'aws_region'     : body.get('aws_region', ''),
            }

            if bucket and key:
                logger.info(f"📦 Downloading s3://{bucket}/{key} ...")
                content, size = self.download_from_s3(bucket, key)

                if content:
                    kafka_message.update({
                        'bucket'    : bucket,
                        'key'       : key,
                        'content'   : content,
                        'size_bytes': size,
                        'has_file'  : True,
                    })
                    logger.info(f"✅ File included ({size:,} bytes)")
                else:
                    kafka_message.update({
                        'bucket'  : bucket,
                        'key'     : key,
                        'has_file': False,
                    })
                    logger.warning("⚠️  Metadata only – S3 download failed")
            else:
                logger.warning(f"📄 No S3 info found\n   Keys: {list(body.keys())}")
                kafka_message.update({
                    'original_message': body,
                    'has_file'        : False,
                })

            return kafka_message

        except Exception as e:
            logger.error(f"❌ Error processing SQS message: {e}")
            return None 
    def run(self, poll_interval=1):
        """Main loop to poll SQS and send to Kafka"""
        if not self.producer or not self.sqs_client:
            self.initialize()
        
        logger.info("🔄 Starting SQS to Kafka bridge...")
        message_count = 0
        
        while True:
            try:
                # Poll SQS
                response = self.sqs_client.receive_message(
                    QueueUrl=self.config.queue_url,
                    MaxNumberOfMessages=10,
                    WaitTimeSeconds=5,
                    VisibilityTimeout=30
                )
                
                if 'Messages' in response:
                    logger.info(f"📥 Received {len(response['Messages'])} messages from SQS")
                    
                    for sqs_message in response['Messages']:
                        message_count += 1
                        logger.info(f"📦 Processing message #{message_count} (ID: {sqs_message['MessageId'][:8]}...)")
                        
                        # Process the message
                        kafka_message = self.process_sqs_message(sqs_message)
                        
                        if kafka_message:
                            # Send to Kafka
                            future = self.producer.send(
                                self.config.topic, 
                                value=kafka_message
                            )
                            result = future.get(timeout=10)
                            
                            logger.info(f"✅ Sent to Kafka: topic={result.topic}, partition={result.partition}, offset={result.offset}")
                            
                            # Delete from SQS
                            self.sqs_client.delete_message(
                                QueueUrl=self.config.queue_url,
                                ReceiptHandle=sqs_message['ReceiptHandle']
                            )
                            logger.info(f"🗑️ Deleted from SQS")
                    
                    self.producer.flush()
                    logger.info(f"✅ Flushed {message_count} messages")
                
            except KeyboardInterrupt:
                logger.info("👋 Shutting down...")
                break
            except Exception as e:
                logger.error(f"❌ Error: {str(e)}")
                time.sleep(5)
            
            time.sleep(poll_interval)
    
    def close(self):
        """Clean up resources"""
        if self.producer:
            self.producer.close()
            logger.info("✅ Kafka producer closed")

def get_kafka_producer_config() -> KafkaProducerConfig:
    """Get Kafka producer configuration from environment and config"""
    from dotenv import load_dotenv
    load_dotenv()
    
    # Load from environment variables
    config = KafkaProducerConfig(
        bootstrap_servers=os.getenv("KAFKA_BROKER", "localhost:9092"),
        topic=os.getenv("KAFKA_TOPIC", "drone-images-topic"),
        aws_region=os.getenv("REGION", "us-west-1"),
        queue_url=os.getenv("QUEUE_URL"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )
    
    return config

if __name__ == "__main__":
    try:
        # Get configuration
        config = get_kafka_producer_config()
        
        # Validate required config
        if not config.queue_url:
            logger.error("QUEUE_URL environment variable is required")
            exit(1)
            
        if not config.aws_access_key_id or not config.aws_secret_access_key:
            logger.error("AWS credentials are required")
            exit(1)
        
        # Create and run producer
        producer = KafkaDataProducer(config)
        producer.initialize()
        
        logger.info("Starting Kafka producer...")
        logger.info(f"Kafka Broker: {config.bootstrap_servers}")
        logger.info(f"Kafka Topic: {config.topic}")
        logger.info(f"SQS Queue URL: {config.queue_url}")
        
        producer.run()
        
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        exit(1)
    finally:
        if 'producer' in locals():
            producer.close()