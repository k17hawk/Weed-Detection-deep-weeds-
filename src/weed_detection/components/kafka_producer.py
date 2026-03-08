
import json
import time
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

    # ── initialize ────────────────────────────────────────────────────────────

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

        logger.info(f"✅ Producer ready")
        logger.info(f"   Topic  : {self.config.topic}")
        logger.info(f"   Queue  : {self.config.queue_url}")
        logger.info(f"   Mode   : S3 pointer only — no file download")

    # ── process one SQS message ───────────────────────────────────────────────

    def process_sqs_message(self, sqs_message: dict) -> dict:
        """
        Parse SQS message body → extract bucket + key → return Kafka pointer.

        Handles 4 SQS body formats:
          Format 1 – Lambda flat   : { bucket, file_key }         ← current setup
          Format 2 – Direct        : { bucket, key }
          Format 3 – SNS-wrapped   : { Message: "{ Records: [...] }" }
          Format 4 – Raw S3 event  : { Records: [...] }

        Returns ~200 byte pointer dict. Zip file stays in S3.
        """
        try:
            body = json.loads(sqs_message["Body"])
            logger.info(f"🔍 SQS body keys: {list(body.keys())}")

            bucket = None
            key    = None

            # Format 1 – Lambda direct (current Lambda setup)
            if "bucket" in body and "file_key" in body:
                bucket = body["bucket"]
                key    = body["file_key"]
                logger.info(f"📨 Format 1 (Lambda) → s3://{bucket}/{key}")

            # Format 2 – direct bucket + key
            elif "bucket" in body and "key" in body:
                bucket = body["bucket"]
                key    = body["key"]
                logger.info(f"📨 Format 2 (Direct) → s3://{bucket}/{key}")

            # Format 3 – SNS-wrapped
            elif "Message" in body:
                try:
                    inner = json.loads(body["Message"])
                    if "Records" in inner:
                        record = inner["Records"][0]
                        bucket = record["s3"]["bucket"]["name"]
                        key    = record["s3"]["object"]["key"]
                        logger.info(f"📨 Format 3 (SNS) → s3://{bucket}/{key}")
                except (json.JSONDecodeError, KeyError):
                    pass

            # Format 4 – raw S3 Records
            elif "Records" in body:
                record = body["Records"][0]
                bucket = record["s3"]["bucket"]["name"]
                key    = record["s3"]["object"]["key"]
                logger.info(f"📨 Format 4 (S3 Records) → s3://{bucket}/{key}")

            if not bucket or not key:
                logger.warning(f"⚠️  Could not extract bucket/key from SQS body")
                logger.warning(f"   Keys present: {list(body.keys())}")
                return None

            # ── build pointer message ─────────────────────────────────────────
            # ~200 bytes regardless of zip file size
            kafka_message = {
                "bucket"        : bucket,
                "key"           : key,
                "has_file"      : True,
                "timestamp"     : time.time(),
                "sqs_message_id": sqs_message["MessageId"],
                "source"        : "sqs_bridge",
                "aws_region"    : self.config.aws_region,
            }

            logger.info(
                f"📦 Pointer ready → s3://{bucket}/{key}  "
                f"(~{len(json.dumps(kafka_message))} bytes)"
            )
            return kafka_message

        except Exception as e:
            logger.error(f"❌ Error processing SQS message: {e}")
            return None

    # ── main polling loop ─────────────────────────────────────────────────────

    def run(self, poll_interval: int = 1):
        if not self.producer or not self.sqs_client:
            self.initialize()

        logger.info("🔄 Polling SQS…  (Ctrl+C to stop)")
        message_count = 0

        while True:
            try:
                response = self.sqs_client.receive_message(
                    QueueUrl            = self.config.queue_url,
                    MaxNumberOfMessages = 10,
                    WaitTimeSeconds     = 5,    # long poll — reduces empty responses
                    VisibilityTimeout   = 30,
                )

                if "Messages" in response:
                    logger.info(
                        f"📥 {len(response['Messages'])} message(s) from SQS"
                    )

                    for sqs_msg in response["Messages"]:
                        message_count += 1
                        logger.info(
                            f"\n📦 Message #{message_count}  "
                            f"ID: {sqs_msg['MessageId'][:8]}…"
                        )

                        kafka_message = self.process_sqs_message(sqs_msg)

                        if kafka_message:
                            future = self.producer.send(
                                self.config.topic, value=kafka_message
                            )
                            result = future.get(timeout=10)
                            logger.info(
                                f"✅ Sent → topic={result.topic}  "
                                f"partition={result.partition}  "
                                f"offset={result.offset}"
                            )

                            # delete from SQS so it's not reprocessed
                            self.sqs_client.delete_message(
                                QueueUrl     = self.config.queue_url,
                                ReceiptHandle= sqs_msg["ReceiptHandle"],
                            )
                            logger.info("🗑️  Deleted from SQS")
                        else:
                            logger.warning("⚠️  Skipped — could not build Kafka message")

                    self.producer.flush()

            except KeyboardInterrupt:
                logger.info("\n👋 Shutting down producer…")
                break
            except Exception as e:
                logger.error(f"❌ Error in poll loop: {e}")
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
        logger.error("❌ QUEUE_URL not set in .env"); exit(1)
    if not config.aws_access_key_id or not config.aws_secret_access_key:
        logger.error("❌ AWS credentials not set in .env"); exit(1)

    producer = KafkaDataProducer(config)
    producer.initialize()

    try:
        producer.run()
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    finally:
        producer.close()


if __name__ == "__main__":
    main()