# entity/artifact_entity.py
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Optional, List


@dataclass(frozen=True)
class KafkaArtifact:
    """
    Produced by : kafka_consumer.py
    Consumed by : data_ingestion component  (passed as input)

    Represents one zip archive received from Kafka and saved to disk.
    DataIngestion reads this artifact to know where the zip lives.
    """
    # ── storage ───────────────────────────────────────────────────────────────
    kafka_data_dir  : Path          # artifacts/data_ingestion/kafka_data/
    version_dir     : Path          # kafka_data/v_YYYYMMDD_HHMMSS/
    zip_file_path   : Path          # version_dir/<hash>_<name>.zip

    # ── S3 origin ─────────────────────────────────────────────────────────────
    s3_bucket       : str
    s3_key          : str
    source_url      : str           # DATA_SOURCE env var

    # ── file metadata ─────────────────────────────────────────────────────────
    file_hash       : str           # SHA-256[:16] of zip bytes
    file_size_bytes : int
    original_filename: str

    # ── kafka provenance ──────────────────────────────────────────────────────
    kafka_topic     : str
    kafka_partition : int
    kafka_offset    : int

    # ── timing ────────────────────────────────────────────────────────────────
    received_at     : datetime

    # ── artifact JSON path (set after persistence) ────────────────────────────
    artifact_path   : Optional[Path] = None


@dataclass(frozen=True)
class DataIngestionArtifact:
    """Artifact produced by data ingestion component"""
    root_dir         : Path
    source_url       : str
    version_directory: Path
    ingested_files   : List[Path]
    metadata_files   : List[Path]
    timestamp        : datetime
    file_count       : int
    manifest_path    : Optional[Path] = None

    @property
    def processing_dir(self) -> Path:
        """Get the directory with symlinks for processing"""
        return self.root_dir / "processing"