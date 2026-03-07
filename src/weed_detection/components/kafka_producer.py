import json
import time
import base64
import boto3
from weed_detection import logger
from weed_detection.entity.config_entity import KafkaProducerConfig
from weed_detection.config.configuration import ConfigurationManager
from kafka import KafkaProducer


class KafkaDataProducer:
    def __init__(self, config: KafkaProducerConfig):
        self.config     = config
        self.producer   = None
        self.sqs_client = None

    def initialize(self):
        logger.info("🚀 Initializing Kafka Producer…")

        self.producer = KafkaProducer(
            bootstrap_servers = [self.config.bootstrap_servers],
            value_serializer  = lambda v: json.dumps(v).encode("utf-8"),
            acks              = "all",
            retries           = 3,
        )

        self.sqs_client = boto3.client(
            "sqs",
            region_name           = self.config.aws_region,
            aws_access_key_id     = self.config.aws_access_key_id,
            aws_secret_access_key = self.config.aws_secret_access_key,
        )

        logger.info(f"✅ Producer ready – topic  : {self.config.topic}")
        logger.info(f"✅ Producer ready – queue  : {self.config.queue_url}")

    def download_from_s3(self, bucket: str, key: str):
        """Download file from S3, return (base64_str, size_bytes)."""
        try:
            s3 = boto3.client(
                "s3",
                region_name           = self.config.aws_region,
                aws_access_key_id     = self.config.aws_access_key_id,
                aws_secret_access_key = self.config.aws_secret_access_key,
            )
            logger.info(f"⬇️  Downloading s3://{bucket}/{key}")
            response = s3.get_object(Bucket=bucket, Key=key)
            content  = response["Body"].read()
            encoded  = base64.b64encode(content).decode("utf-8")
            logger.info(f"✅ Downloaded {len(content):,} bytes")
            return encoded, len(content)
        except Exception as e:
            logger.error(f"❌ S3 download failed: {e}")
            return None, 0

    def process_sqs_message(self, sqs_message: dict):
        """
        Parse SQS message → resolve bucket+key → download zip → build Kafka message.

        Handles 4 formats:
          1. Lambda flat  : body has 'bucket' + 'file_key'   ← current setup
          2. Direct       : body has 'bucket' + 'key'
          3. SNS-wrapped  : body has 'Message' (JSON string) with Records
          4. Raw S3 event : body has 'Records'
        """
        try:
            body = json.loads(sqs_message["Body"])
            logger.info(f"🔍 SQS body keys: {list(body.keys())}")

            bucket = None
            key    = None

            # Format 1 – Lambda direct (your current setup)
            if "bucket" in body and "file_key" in body:
                bucket = body["bucket"]
                key    = body["file_key"]
                logger.info(f"📨 Lambda message → s3://{bucket}/{key}")

            # Format 2 – direct bucket + key
            elif "bucket" in body and "key" in body:
                bucket = body["bucket"]
                key    = body["key"]

            # Format 3 – SNS wrapper
            elif "Message" in body:
                try:
                    inner = json.loads(body["Message"])
                    if "Records" in inner:
                        record = inner["Records"][0]
                        bucket = record["s3"]["bucket"]["name"]
                        key    = record["s3"]["object"]["key"]
                        logger.info(f"📨 SNS-wrapped → s3://{bucket}/{key}")
                except (json.JSONDecodeError, KeyError):
                    pass

            # Format 4 – raw S3 Records
            elif "Records" in body:
                record = body["Records"][0]
                bucket = record["s3"]["bucket"]["name"]
                key    = record["s3"]["object"]["key"]

            # ── base message ──────────────────────────────────────────────────
            kafka_message = {
                "timestamp"     : time.time(),
                "sqs_message_id": sqs_message["MessageId"],
                "source"        : "sqs_bridge",
                "file_name"     : body.get("file_name",  ""),
                "file_type"     : body.get("file_type",  ""),
                "file_size"     : body.get("file_size",   0),
                "event_time"    : body.get("event_time",  ""),
                "aws_region"    : body.get("aws_region",  ""),
            }

            if bucket and key:
                logger.info(f"📦 Downloading s3://{bucket}/{key} …")
                content, size = self.download_from_s3(bucket, key)

                if content:
                    kafka_message.update({
                        "bucket"    : bucket,
                        "key"       : key,
                        "content"   : content,
                        "size_bytes": size,
                        "has_file"  : True,
                    })
                    logger.info(f"✅ File included ({size:,} bytes)")
                else:
                    kafka_message.update({
                        "bucket"  : bucket,
                        "key"     : key,
                        "has_file": False,
                    })
                    logger.warning("⚠️  Metadata only – S3 download failed")
            else:
                logger.warning(f"📄 No S3 info found\n   Keys: {list(body.keys())}")
                kafka_message.update({
                    "original_message": body,
                    "has_file"        : False,
                })

            return kafka_message

        except Exception as e:
            logger.error(f"❌ Error processing SQS message: {e}")
            return None

    def run(self, poll_interval: int = 1):
        if not self.producer or not self.sqs_client:
            self.initialize()

        logger.info("🔄 Starting SQS → Kafka bridge…")
        message_count = 0

        while True:
            try:
                response = self.sqs_client.receive_message(
                    QueueUrl          = self.config.queue_url,
                    MaxNumberOfMessages= 10,
                    WaitTimeSeconds   = 5,
                    VisibilityTimeout = 30,
                )

                if "Messages" in response:
                    logger.info(f"📥 Received {len(response['Messages'])} message(s) from SQS")

                    for sqs_message in response["Messages"]:
                        message_count += 1
                        logger.info(f"📦 Processing #{message_count} (ID: {sqs_message['MessageId'][:8]}…)")

                        kafka_message = self.process_sqs_message(sqs_message)

                        if kafka_message:
                            future = self.producer.send(self.config.topic, value=kafka_message)
                            result = future.get(timeout=10)
                            logger.info(
                                f"✅ Sent → topic={result.topic} "
                                f"partition={result.partition} offset={result.offset}"
                            )

                            self.sqs_client.delete_message(
                                QueueUrl     = self.config.queue_url,
                                ReceiptHandle= sqs_message["ReceiptHandle"],
                            )
                            logger.info("🗑️  Deleted from SQS")

                    self.producer.flush()
                    logger.info(f"✅ Flushed {message_count} messages")

            except KeyboardInterrupt:
                logger.info("👋 Shutting down producer…")
                break
            except Exception as e:
                logger.error(f"❌ Error: {e}")
                time.sleep(5)

            time.sleep(poll_interval)

    def close(self):
        if self.producer:
            self.producer.close()
            logger.info("✅ Kafka producer closed")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("🚀 Starting Kafka Producer")
    logger.info("=" * 60)

    config_manager = ConfigurationManager()
    config         = config_manager.get_kafka_producer_config()

    if not config.queue_url:
        logger.error("❌ QUEUE_URL not set in .env")
        exit(1)
    if not config.aws_access_key_id or not config.aws_secret_access_key:
        logger.error("❌ AWS credentials not set in .env")
        exit(1)

    producer = KafkaDataProducer(config)
    producer.initialize()

    logger.info(f"   Broker : {config.bootstrap_servers}")
    logger.info(f"   Topic  : {config.topic}")
    logger.info(f"   Queue  : {config.queue_url}")

    try:
        producer.run()
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    finally:
        producer.close()


if __name__ == "__main__":
    main()