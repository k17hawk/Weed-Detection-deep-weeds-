# components/kafka_consumer.py
"""
Responsibility (ONLY):
  1. Connect to Kafka
  2. Receive message → fetch zip bytes (inline base64 OR from S3)
  3. Save zip  →  artifacts/data_ingestion/kafka_data/v_<timestamp>/
  4. Write KafkaArtifact  →  kafka_data/latest_artifact.json

DataIngestion reads latest_artifact.json to get zip_file_path.
This component does NOT unzip, normalize, or process anything.
"""

import json
import base64
import os
import hashlib
import boto3
from botocore.exceptions import ClientError
from pathlib import Path
from datetime import datetime
from typing import Optional

from kafka import KafkaConsumer

from weed_detection import logger
from weed_detection.entity.config_entity import KafkaConsumerConfig
from weed_detection.entity.artifact_entity import KafkaArtifact
from weed_detection.config.configuration import ConfigurationManager
from dotenv import load_dotenv

load_dotenv()

DATA_SOURCE    = os.getenv("DATA_SOURCE")
AWS_REGION     = os.getenv("REGION")
AWS_ACCESS_KEY = os.getenv("ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("SECRET_ACCESS_KEY")


# ── S3 ────────────────────────────────────────────────────────────────────────

def _s3_client():
    return boto3.client(
        "s3",
        region_name           = AWS_REGION,
        aws_access_key_id     = AWS_ACCESS_KEY,
        aws_secret_access_key = AWS_SECRET_KEY,
    )


def fetch_from_s3(bucket: str, key: str) -> Optional[bytes]:
    try:
        resp = _s3_client().get_object(Bucket=bucket, Key=key)
        data = resp["Body"].read()
        logger.info(f"☁️  Fetched s3://{bucket}/{key}  ({len(data):,} bytes)")
        return data
    except ClientError as e:
        logger.error(f"❌ S3 error: {e}")
        return None


def resolve_zip_bytes(msg: dict) -> Optional[bytes]:
    """
    Extract raw zip bytes from a Kafka message.
      - Pattern 1: msg has 'content' key  →  decode base64
      - Pattern 2: msg has 'bucket'+'key' →  fetch from S3
    """
    if "content" in msg:
        logger.info("📦 Inline base64 detected – decoding…")
        return base64.b64decode(msg["content"])

    bucket = msg.get("bucket")
    key    = msg.get("key")
    if bucket and key:
        return fetch_from_s3(bucket, key)

    logger.warning("⚠️  No 'content' and no bucket/key in message")
    return None


# ── Artifact helpers ──────────────────────────────────────────────────────────

def _sha256_short(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _artifact_to_dict(a: KafkaArtifact) -> dict:
    return {
        "kafka_data_dir"   : str(a.kafka_data_dir),
        "version_dir"      : str(a.version_dir),
        "zip_file_path"    : str(a.zip_file_path),
        "s3_bucket"        : a.s3_bucket,
        "s3_key"           : a.s3_key,
        "source_url"       : a.source_url,
        "file_hash"        : a.file_hash,
        "file_size_bytes"  : a.file_size_bytes,
        "original_filename": a.original_filename,
        "kafka_topic"      : a.kafka_topic,
        "kafka_partition"  : a.kafka_partition,
        "kafka_offset"     : a.kafka_offset,
        "received_at"      : a.received_at.isoformat(),
    }


def load_kafka_artifact(
    kafka_data_dir: Path,
) -> KafkaArtifact:
    """
    Load the latest KafkaArtifact from disk.
    Called by DataIngestion to get the zip path.

    Usage:
        artifact = load_kafka_artifact(Path("artifacts/data_ingestion/kafka_data"))
        zip_path = artifact.zip_file_path
    """
    latest_path = kafka_data_dir / "latest_artifact.json"
    if not latest_path.exists():
        raise FileNotFoundError(
            f"No KafkaArtifact found at {latest_path}. "
            "Run kafka_consumer.py first."
        )

    d = json.loads(latest_path.read_text())
    return KafkaArtifact(
        kafka_data_dir   = Path(d["kafka_data_dir"]),
        version_dir      = Path(d["version_dir"]),
        zip_file_path    = Path(d["zip_file_path"]),
        s3_bucket        = d["s3_bucket"],
        s3_key           = d["s3_key"],
        source_url       = d["source_url"],
        file_hash        = d["file_hash"],
        file_size_bytes  = d["file_size_bytes"],
        original_filename= d["original_filename"],
        kafka_topic      = d["kafka_topic"],
        kafka_partition  = d["kafka_partition"],
        kafka_offset     = d["kafka_offset"],
        received_at      = datetime.fromisoformat(d["received_at"]),
        artifact_path    = latest_path,
    )


# ── Core consumer class ───────────────────────────────────────────────────────

class KafkaZipConsumer:
    """
    Connects to Kafka, receives messages, saves zips, produces KafkaArtifact.
    """

    def __init__(self, config: KafkaConsumerConfig):
        self.config = config
        logger.info("=" * 60)
        logger.info("🚀 KafkaZipConsumer initialized")
        logger.info(f"   Broker        : {config.broker}")
        logger.info(f"   Topic         : {config.topic}")
        logger.info(f"   Group         : {config.group_id}")
        logger.info(f"   Kafka data dir: {config.kafka_data_dir}")
        logger.info("=" * 60)

    def _make_version_dir(self) -> Path:
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = self.config.kafka_data_dir / f"v_{ts}"
        version.mkdir(parents=True, exist_ok=True)
        return version

    def _save_zip(
        self,
        zip_bytes        : bytes,
        msg              : dict,
        partition        : int,
        offset           : int,
    ) -> KafkaArtifact:
        """Persist zip to versioned dir and return populated KafkaArtifact."""
        file_hash        = _sha256_short(zip_bytes)
        received_at      = datetime.now()
        version_dir      = self._make_version_dir()

        original_filename = (
            os.path.basename(msg.get("key", ""))
            or f"drone_{received_at.strftime('%Y%m%d_%H%M%S')}.zip"
        )
        if not original_filename.lower().endswith(".zip"):
            original_filename += ".zip"

        zip_path = version_dir / f"{file_hash}_{original_filename}"

        if not zip_path.exists():
            zip_path.write_bytes(zip_bytes)
            logger.info(f"💾 Zip saved : {zip_path}  ({len(zip_bytes):,} bytes)")
        else:
            logger.info(f"🔄 Duplicate : {zip_path}")

        artifact = KafkaArtifact(
            kafka_data_dir   = self.config.kafka_data_dir,
            version_dir      = version_dir,
            zip_file_path    = zip_path,
            s3_bucket        = msg.get("bucket", ""),
            s3_key           = msg.get("key",    ""),
            source_url       = DATA_SOURCE,
            file_hash        = file_hash,
            file_size_bytes  = len(zip_bytes),
            original_filename= original_filename,
            kafka_topic      = self.config.topic,
            kafka_partition  = partition,
            kafka_offset     = offset,
            received_at      = received_at,
        )

        # ── persist artifact JSON ─────────────────────────────────────────
        artifact_dict = _artifact_to_dict(artifact)

        # version-scoped copy  (historical record)
        (version_dir / "artifact.json").write_text(
            json.dumps(artifact_dict, indent=2)
        )

        # latest pointer  (DataIngestion reads this)
        latest_path = self.config.kafka_data_dir / "latest_artifact.json"
        latest_path.write_text(json.dumps(artifact_dict, indent=2))
        logger.info(f"📋 Artifact  : {latest_path}")

        return artifact

    def run(self) -> None:
        consumer = KafkaConsumer(
            self.config.topic,
            bootstrap_servers   = [self.config.broker],
            auto_offset_reset   = "latest",
            group_id            = self.config.group_id,
            value_deserializer  = self._deserialize,
            enable_auto_commit  = True,
            consumer_timeout_ms = -1,
        )

        logger.info("✅ Connected to Kafka. Waiting for messages… (Ctrl+C to stop)")

        try:
            for count, message in enumerate(consumer, start=1):
                logger.info(
                    f"\n📬 Message #{count}  "
                    f"partition={message.partition}  offset={message.offset}"
                )

                msg = message.value
                if not isinstance(msg, dict):
                    logger.warning("⚠️  Non-dict message – skipping")
                    continue

                logger.info(f"   Keys     : {list(msg.keys())}")
                logger.info(f"   has_file : {msg.get('has_file')}")

                if not msg.get("has_file"):
                    logger.info("   ℹ️  No file flag – skipping")
                    continue

                zip_bytes = resolve_zip_bytes(msg)
                if zip_bytes is None:
                    logger.error("❌ Could not retrieve zip bytes – skipping")
                    continue

                artifact = self._save_zip(
                    zip_bytes = zip_bytes,
                    msg       = msg,
                    partition = message.partition,
                    offset    = message.offset,
                )

                logger.info(f"   ✅ Done   : {artifact.zip_file_path.name}")
                logger.info(f"   🔑 Hash   : {artifact.file_hash}")
                logger.info(f"   📦 Size   : {artifact.file_size_bytes:,} bytes")
                logger.info(f"   🕐 Time   : {artifact.received_at}")
                logger.info("-" * 50)

        except KeyboardInterrupt:
            logger.info("\n👋 Shutdown requested – consumer stopped cleanly")

        except Exception as e:
            logger.error(f"❌ Fatal: {e}")
            import traceback
            logger.error(traceback.format_exc())

        finally:
            logger.info("🏁 KafkaZipConsumer finished.")

    @staticmethod
    def _deserialize(raw: bytes) -> dict:
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception as e:
            logger.error(f"❌ Deserialize error: {e}")
            return {"error": str(e)}


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("🚀 Starting Kafka Consumer")
    logger.info("=" * 60)

    config_manager = ConfigurationManager()
    kafka_config   = config_manager.get_kafka_consumer_config()

    consumer = KafkaZipConsumer(kafka_config)
    consumer.run()


if __name__ == "__main__":
    main()