from kafka import KafkaProducer
import json
import time
from dotenv import load_dotenv
import boto3
import os
import base64
from botocore.exceptions import ClientError

# Load environment variables
load_dotenv()

REGION_NAME = os.getenv("REGION")
KAFKA_BROKER = "localhost:9092"
TOPIC = "drone-images-topic"
QUEUE_URL = os.getenv("QUEUE_URL")

# Get AWS credentials from .env (after you've created new ones!)
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

def download_from_s3(bucket, key):
    """Download file from S3 and return base64 encoded content"""
    try:
        # Create S3 client with explicit credentials
        s3 = boto3.client(
            's3',
            region_name=REGION_NAME,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        
        print(f"Downloading s3://{bucket}/{key}")
        response = s3.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read()
        
        # Encode to base64 for JSON transmission
        encoded_content = base64.b64encode(content).decode('utf-8')
        print(f"Downloaded {len(content)} bytes from S3")
        
        return encoded_content, len(content)
    except ClientError as e:
        print(f"S3 Error: {e.response['Error']['Message']}")
        return None, 0
    except Exception as e:
        print(f"Error downloading from S3: {str(e)}")
        return None, 0

def extract_s3_info(message_body):
    """Extract bucket and key from various message formats"""
    try:
        # Parse the message
        if isinstance(message_body, str):
            data = json.loads(message_body)
        else:
            data = message_body
        
        # Handle different formats
        if 'Records' in data:
            # S3 event notification format
            record = data['Records'][0]
            bucket = record['s3']['bucket']['name']
            key = record['s3']['object']['key']
            return bucket, key
        
        elif 's3_data' in data:
            # Our enriched format
            s3_data = data['s3_data']
            if 'bucket' in s3_data and 'object' in s3_data:
                bucket = s3_data['bucket']['name']
                key = s3_data['object']['key']
                return bucket, key
        
        elif 'bucket' in data and 'key' in data:
            # Direct bucket/key format
            return data['bucket'], data['key']
        
        elif 'detail' in data and 'requestParameters' in data['detail']:
            # CloudTrail format
            params = data['detail']['requestParameters']
            if 'bucketName' in params and 'key' in params:
                return params['bucketName'], params['key']
        
        return None, None
    except Exception as e:
        print(f"Error extracting S3 info: {str(e)}")
        return None, None

def sqs_to_kafka():
    # SQS client with explicit credentials
    sqs = boto3.client(
        'sqs',
        region_name=REGION_NAME,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )
    
    # Kafka producer
    producer = KafkaProducer(
        bootstrap_servers=[KAFKA_BROKER],
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        acks='all',  # Wait for all replicas to acknowledge
        retries=3
    )
    
    print(f"🚀 Starting SQS to Kafka bridge")
    print(f"📋 Listening to queue: {QUEUE_URL}")
    print(f"📨 Sending to Kafka topic: {TOPIC}")
    print(f"☁️  AWS Region: {REGION_NAME}")
    print("-" * 50)
    
    message_count = 0
    
    while True:
        try:
            # Poll SQS
            response = sqs.receive_message(
                QueueUrl=QUEUE_URL,
                MaxNumberOfMessages=10,
                WaitTimeSeconds=5,
                AttributeNames=['All'],
                MessageAttributeNames=['All']
            )
            
            if 'Messages' in response:
                print(f"\n📥 Received {len(response['Messages'])} messages from SQS")
                
                for message in response['Messages']:
                    message_count += 1
                    print(f"\n📦 Processing message #{message_count} (ID: {message['MessageId'][:8]}...)")
                    
                    try:
                        # Parse SQS message body
                        sqs_content = json.loads(message['Body'])
                        
                        # Extract bucket and key
                        bucket, key = extract_s3_info(sqs_content)
                        
                        # Prepare enriched message
                        enriched_message = {
                            'timestamp': time.time(),
                            'message_id': message['MessageId'],
                            'sqs_receipt': message['ReceiptHandle'][:20] + "...",
                        }
                        
                        if bucket and key:
                            print(f"🔍 Found S3 object: s3://{bucket}/{key}")
                            
                            # Try to download the zip file
                            zip_content, size = download_from_s3(bucket, key)
                            
                            if zip_content:
                                enriched_message.update({
                                    'bucket': bucket,
                                    'key': key,
                                    'content': zip_content,  # Base64 encoded zip
                                    'size_bytes': size,
                                    'content_type': 'application/zip',
                                    'download_success': True
                                })
                                print(f"✅ Successfully downloaded zip ({size} bytes)")
                            else:
                                enriched_message.update({
                                    'bucket': bucket,
                                    'key': key,
                                    's3_data': sqs_content.get('s3_data', sqs_content),
                                    'download_success': False,
                                    'error': 'Failed to download from S3'
                                })
                                print(f"⚠️  Could not download from S3, sending metadata only")
                        else:
                            print("ℹ️  No S3 information found, sending original message")
                            enriched_message['original_message'] = sqs_content
                        
                        # Send to Kafka
                        future = producer.send(TOPIC, value=enriched_message)
                        result = future.get(timeout=15)
                        print(f"✅ Sent to Kafka: {result.topic}-{result.partition} @ offset {result.offset}")
                        
                        # Delete from SQS
                        sqs.delete_message(
                            QueueUrl=QUEUE_URL,
                            ReceiptHandle=message['ReceiptHandle']
                        )
                        print(f"🗑️  Deleted message from SQS")
                        
                    except Exception as e:
                        print(f"❌ Error processing message: {str(e)}")
                        # Don't delete - message will reappear after visibility timeout
                
                producer.flush()
                print(f"\n✅ Flushed {message_count} messages to Kafka")
            
        except ClientError as e:
            print(f"❌ AWS Error: {e.response['Error']['Message']}")
            time.sleep(10)
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
            time.sleep(5)
        
        time.sleep(1)

if __name__ == "__main__":
    try:
        sqs_to_kafka()
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")