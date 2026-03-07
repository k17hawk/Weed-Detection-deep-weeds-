from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import Optional, List


@dataclass(frozen=True)
class KafkaArtifact:
    """
    Produced by : kafka_consumer.py
    Consumed by : data_ingestion component

    Represents one valid zip received from Kafka and saved locally.
    Persisted as:  artifacts/data_ingestion/kafka_data/latest_artifact.json
    """
    # ── storage ───────────────────────────────────────────────────────────────
    kafka_data_dir   : Path     # artifacts/data_ingestion/kafka_data/
    version_dir      : Path     # kafka_data/v_YYYYMMDD_HHMMSS/
    zip_file_path    : Path     # version_dir/<hash>_<filename>.zip

    # ── S3 origin ─────────────────────────────────────────────────────────────
    s3_bucket        : str
    s3_key           : str
    source_url       : str

    # ── file info ─────────────────────────────────────────────────────────────
    file_hash        : str      # SHA-256[:16]
    file_size_bytes  : int
    original_filename: str      # e.g. drone_20260307_201234.zip

    # ── kafka provenance ──────────────────────────────────────────────────────
    kafka_topic      : str
    kafka_partition  : int
    kafka_offset     : int

    # ── timing ────────────────────────────────────────────────────────────────
    received_at      : datetime

    # ── self reference (set after JSON is written) ────────────────────────────
    artifact_path    : Optional[Path] = None


@dataclass(frozen=True)
class DataIngestionArtifact:
    """
    Produced by : data_ingestion component
    Consumed by : data_validation component

    Represents a fully extracted and normalized dataset.
    """
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
        return self.root_dir / "processing"