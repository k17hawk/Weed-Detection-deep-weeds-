from kafka import KafkaConsumer
import json
import base64
import os
from datetime import datetime
import uuid

KAFKA_BROKER = "localhost:9092"
TOPIC = "drone-images-topic"

def safe_deserializer(m):
    try:
        return json.loads(m.decode("utf-8"))
    except:
        return m.decode("utf-8")

def save_zip_file(content, filename=None):
    """Save base64 encoded zip content to a file"""
    if not filename:
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"drone_images_{timestamp}_{unique_id}.zip"
    
    # Create downloads directory if it doesn't exist
    os.makedirs("downloaded_zips", exist_ok=True)
    
    # Decode and save
    try:
        zip_data = base64.b64decode(content)
        filepath = os.path.join("downloaded_zips", filename)
        with open(filepath, 'wb') as f:
            f.write(zip_data)
        print(f"✅ Saved zip file: {filepath} ({len(zip_data)} bytes)")
        return filepath
    except Exception as e:
        print(f"❌ Error saving zip file: {str(e)}")
        return None

consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers=KAFKA_BROKER,
    auto_offset_reset="latest",
    group_id="drone-group",
    value_deserializer=safe_deserializer
)

print("🚀 Kafka Consumer Started...")
print(f"Listening to topic: {TOPIC}")
print("-" * 50)

for message in consumer:
    try:
        print(f"\n📨 Received message from partition {message.partition} at offset {message.offset}")
        
        # Print message structure to understand format
        print("Message structure:", type(message.value))
        
        # Handle different message formats
        if isinstance(message.value, dict):
            # Check if this is the enriched message from your bridge
            if 's3_data' in message.value:
                print("📦 S3 Data found in message")
                s3_info = message.value['s3_data']
                
                # Extract bucket and key
                if 'bucket' in s3_info and 'object' in s3_info:
                    bucket = s3_info['bucket']['name']
                    key = s3_info['object']['key']
                    print(f"📍 S3 Location: s3://{bucket}/{key}")
            
            # Check if zip content is included
            if 'content' in message.value:
                print("📎 Zip content found in message")
                # Save the zip file
                saved_path = save_zip_file(message.value['content'])
                
                # Extract bucket/key if available
                if 'bucket' in message.value and 'key' in message.value:
                    print(f"📍 Original S3: s3://{message.value['bucket']}/{message.value['key']}")
            
            # Handle case where message is just the S3 event
            elif 'Records' in message.value:
                print("📋 Raw S3 event detected")
                for record in message.value['Records']:
                    if 's3' in record:
                        bucket = record['s3']['bucket']['name']
                        key = record['s3']['object']['key']
                        print(f"📍 S3 Location: s3://{bucket}/{key}")
                        
                        # You might want to download from S3 here if content isn't included
                        # This would require boto3 credentials
                        print("⚠️  Note: Zip content not included in message")
                        print("   To get the actual zip, you need to download from S3")
            
            else:
                print("📄 Other message format:", json.dumps(message.value, indent=2)[:200] + "...")
        
        elif isinstance(message.value, str):
            print("📝 String message:", message.value[:200] + "..." if len(message.value) > 200 else message.value)
        
        else:
            print(f"❓ Unknown message type: {type(message.value)}")
        
        print("-" * 50)
        
    except Exception as e:
        print(f"❌ Error processing message: {str(e)}")
        import traceback
        traceback.print_exc()