
import json
import base64
import os
import hashlib
import boto3
from botocore.exceptions import ClientError
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

from kafka import KafkaConsumer

from weed_detection import logger
from weed_detection.entity.config_entity import KafkaConsumerConfig
from weed_detection.entity.artifact_entity import KafkaArtifact
from weed_detection.config.configuration import ConfigurationManager
from weed_detection.constants.constant import (
    AWS_ACCESS_KEY,
    AWS_SECRET_KEY,
    AWS_REGION,
    DATA_SOURCE,
    FILE_PATTERN,
)


# ── Filename validation ───────────────────────────────────────────────────────

def validate_filename(filename: str) -> Tuple[bool, str]:
    """
    Validate against FILE_PATTERN: drone_YYYYMMDD_HHMMSS.zip
    Returns (is_valid, reason).
    """
    if not filename.lower().endswith(".zip"):
        return False, f"Not a zip file: '{filename}'"

    match = FILE_PATTERN.match(filename)
    if not match:
        return False, (
            f"'{filename}' does not match required pattern. "
            f"Expected: drone_YYYYMMDD_HHMMSS.zip  e.g. drone_20260307_201234.zip"
        )

    try:
        datetime(
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
            int(match.group("hour")),
            int(match.group("minute")),
            int(match.group("second")),
        )
    except ValueError as e:
        return False, f"Invalid date/time in filename: {e}"

    return True, "OK"


# ── S3 helpers ────────────────────────────────────────────────────────────────

def _s3_client():
    return boto3.client(
        "s3",
        region_name           = AWS_REGION,
        aws_access_key_id     = AWS_ACCESS_KEY,
        aws_secret_access_key = AWS_SECRET_KEY,
    )


def fetch_from_s3(bucket: str, key: str) -> Optional[bytes]:
    """Download file from S3 and return raw bytes."""
    try:
        resp = _s3_client().get_object(Bucket=bucket, Key=key)
        data = resp["Body"].read()
        logger.info(f"☁️  Downloaded s3://{bucket}/{key}  ({len(data):,} bytes)")
        return data
    except ClientError as e:
        logger.error(f"❌ S3 download failed: {e}")
        return None


def resolve_zip_bytes(msg: dict) -> Optional[bytes]:
    """
    Get zip bytes from Kafka message.

    Pattern 1: 'content' key  → decode base64  (legacy — file embedded in message)
    Pattern 2: 'bucket' + 'key' → fetch from S3 (current — S3 pointer only)

    Current producer always sends Pattern 2 (pointer only) to stay under
    Kafka's 1MB message size limit.
    """
    if "content" in msg:
        logger.info("📦 Inline base64 content — decoding…")
        return base64.b64decode(msg["content"])

    bucket = msg.get("bucket")
    key    = msg.get("key")
    if bucket and key:
        logger.info(f"📦 S3 pointer — downloading from s3://{bucket}/{key}")
        return fetch_from_s3(bucket, key)

    logger.warning("⚠️  No 'content' and no bucket/key in message — cannot get file")
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


def load_kafka_artifact(kafka_data_dir: Path) -> KafkaArtifact:
    """
    Load KafkaArtifact from latest_artifact.json.
    Called by DataIngestion to find the zip path.

    Usage:
        artifact = load_kafka_artifact(Path("artifacts/data_ingestion/kafka_data"))
        zip_path = artifact.zip_file_path
    """
    latest_path = kafka_data_dir / "latest_artifact.json"
    if not latest_path.exists():
        raise FileNotFoundError(
            f"No KafkaArtifact found at {latest_path}\n"
            f"Run kafka_consumer.py first to receive a zip from Kafka."
        )
    d = json.loads(latest_path.read_text())
    return KafkaArtifact(
        kafka_data_dir    = Path(d["kafka_data_dir"]),
        version_dir       = Path(d["version_dir"]),
        zip_file_path     = Path(d["zip_file_path"]),
        s3_bucket         = d["s3_bucket"],
        s3_key            = d["s3_key"],
        source_url        = d["source_url"],
        file_hash         = d["file_hash"],
        file_size_bytes   = d["file_size_bytes"],
        original_filename = d["original_filename"],
        kafka_topic       = d["kafka_topic"],
        kafka_partition   = d["kafka_partition"],
        kafka_offset      = d["kafka_offset"],
        received_at       = datetime.fromisoformat(d["received_at"]),
        artifact_path     = latest_path,
    )


# ── Consumer class ────────────────────────────────────────────────────────────

class KafkaZipConsumer:
    """
    Listens to Kafka topic.
    Each message carries an S3 pointer { bucket, key }.
    Downloads zip from S3, validates filename, saves to disk, writes KafkaArtifact.
    Invalid filenames are quarantined and never processed further.
    """

    def __init__(self, config: KafkaConsumerConfig):
        self.config = config
        logger.info("=" * 60)
        logger.info("🚀 KafkaZipConsumer initialized")
        logger.info(f"   Broker    : {config.broker}")
        logger.info(f"   Topic     : {config.topic}")
        logger.info(f"   Group     : {config.group_id}")
        logger.info(f"   Good data : {config.kafka_data_dir}")
        logger.info(f"   Bad data  : {config.bad_raw_data_dir}")
        logger.info(f"   Pattern   : drone_YYYYMMDD_HHMMSS.zip")
        logger.info("=" * 60)

    # ── versioned directory ───────────────────────────────────────────────────

    def _make_version_dir(self) -> Path:
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = self.config.kafka_data_dir / f"v_{ts}"
        version.mkdir(parents=True, exist_ok=True)
        return version

    # ── quarantine invalid files ──────────────────────────────────────────────

    def _quarantine(
        self,
        zip_bytes        : bytes,
        original_filename: str,
        reason           : str,
        msg              : dict,
        partition        : int,
        offset           : int,
    ) -> None:
        """Save rejected zip + rejection log to bad_raw_data/. No artifact produced."""
        file_hash = _sha256_short(zip_bytes)
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest_path = self.config.bad_raw_data_dir / f"{ts}_{file_hash}_{original_filename}"
        log_path  = self.config.bad_raw_data_dir / f"{ts}_{file_hash}_rejection.json"

        dest_path.write_bytes(zip_bytes)

        rejection = {
            "original_filename": original_filename,
            "rejection_reason" : reason,
            "quarantined_at"   : datetime.now().isoformat(),
            "quarantined_to"   : str(dest_path),
            "file_hash"        : file_hash,
            "file_size_bytes"  : len(zip_bytes),
            "s3_bucket"        : msg.get("bucket", ""),
            "s3_key"           : msg.get("key",    ""),
            "kafka_topic"      : self.config.topic,
            "kafka_partition"  : partition,
            "kafka_offset"     : offset,
            "expected_pattern" : "drone_YYYYMMDD_HHMMSS.zip",
            "example"          : "drone_20260307_201234.zip",
        }
        log_path.write_text(json.dumps(rejection, indent=2))

        logger.warning("=" * 60)
        logger.warning("🚫 FILE QUARANTINED")
        logger.warning(f"   Filename : {original_filename}")
        logger.warning(f"   Reason   : {reason}")
        logger.warning(f"   Saved to : {dest_path}")
        logger.warning(f"   Log      : {log_path}")
        logger.warning("=" * 60)

    # ── save valid zip + write artifact ───────────────────────────────────────

    def _save_zip(
        self,
        zip_bytes        : bytes,
        original_filename: str,
        msg              : dict,
        partition        : int,
        offset           : int,
    ) -> KafkaArtifact:
        """Persist valid zip, write versioned + latest artifact JSON."""
        file_hash   = _sha256_short(zip_bytes)
        received_at = datetime.now()
        version_dir = self._make_version_dir()
        zip_path    = version_dir / f"{file_hash}_{original_filename}"

        if not zip_path.exists():
            zip_path.write_bytes(zip_bytes)
            logger.info(f"💾 Saved : {zip_path}  ({len(zip_bytes):,} bytes)")
        else:
            logger.info(f"🔄 Already exists (duplicate): {zip_path}")

        artifact = KafkaArtifact(
            kafka_data_dir    = self.config.kafka_data_dir,
            version_dir       = version_dir,
            zip_file_path     = zip_path,
            s3_bucket         = msg.get("bucket", ""),
            s3_key            = msg.get("key",    ""),
            source_url        = DATA_SOURCE,
            file_hash         = file_hash,
            file_size_bytes   = len(zip_bytes),
            original_filename = original_filename,
            kafka_topic       = self.config.topic,
            kafka_partition   = partition,
            kafka_offset      = offset,
            received_at       = received_at,
        )

        artifact_dict = _artifact_to_dict(artifact)

        # permanent version copy
        version_artifact_path = version_dir / "artifact.json"
        version_artifact_path.write_text(json.dumps(artifact_dict, indent=2))

        # latest pointer — data_ingestion reads this
        latest_path = self.config.kafka_data_dir / "latest_artifact.json"
        latest_path.write_text(json.dumps(artifact_dict, indent=2))

        logger.info(f"📋 Artifact written : {latest_path}")
        return artifact

    # ── process one Kafka message ─────────────────────────────────────────────

    def _process_message(self, msg: dict, partition: int, offset: int) -> None:
        # 1. download zip bytes from S3 using pointer in message
        zip_bytes = resolve_zip_bytes(msg)
        if zip_bytes is None:
            logger.error("❌ Could not retrieve zip bytes — skipping message")
            return

        # 2. resolve filename from S3 key
        original_filename = (
            os.path.basename(msg.get("key", ""))
            or os.path.basename(msg.get("file_name", ""))
            or f"unknown_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        )
        logger.info(f"   Filename  : {original_filename}")
        logger.info(f"   Size      : {len(zip_bytes):,} bytes")

        # 3. validate filename pattern
        is_valid, reason = validate_filename(original_filename)

        if not is_valid:
            # quarantine and stop — no artifact produced
            self._quarantine(zip_bytes, original_filename, reason, msg, partition, offset)
            return

        # 4. save + produce artifact
        logger.info(f"   ✅ Valid filename")
        artifact = self._save_zip(zip_bytes, original_filename, msg, partition, offset)

        logger.info(f"   ✅ Complete")
        logger.info(f"   Hash      : {artifact.file_hash}")
        logger.info(f"   Zip       : {artifact.zip_file_path}")
        logger.info(f"   Received  : {artifact.received_at}")

    # ── main consumer loop ────────────────────────────────────────────────────

    def run(self) -> None:
        consumer = KafkaConsumer(
            self.config.topic,
            bootstrap_servers   = [self.config.broker],
            auto_offset_reset   = "latest",
            group_id            = self.config.group_id,
            value_deserializer  = self._deserialize,
            enable_auto_commit  = True,
            consumer_timeout_ms = -1,   # run forever
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
                    logger.warning("⚠️  Non-dict message — skipping"); continue

                logger.info(f"   Keys     : {list(msg.keys())}")
                logger.info(f"   has_file : {msg.get('has_file')}")

                if not msg.get("has_file"):
                    logger.info("   ℹ️  has_file=False — skipping"); continue

                self._process_message(msg, message.partition, message.offset)
                logger.info("-" * 50)

        except KeyboardInterrupt:
            logger.info("\n👋 Shutdown — consumer stopped cleanly")
        except Exception as e:
            logger.error(f"❌ Fatal error: {e}")
            import traceback; logger.error(traceback.format_exc())
        finally:
            logger.info("🏁 KafkaZipConsumer finished")

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
    config         = config_manager.get_kafka_consumer_config()

    consumer = KafkaZipConsumer(config)
    consumer.run()


if __name__ == "__main__":
    main()