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
    kafka_artifact   : KafkaArtifact   # which zip this came from
    unzip_dir        : Path      
          # raw extracted contents
    normalized_dir   : Path            # normalized images + labels

    
    train_images_dir : Path
    train_labels_dir : Path
    val_images_dir   : Optional[Path]  = None
    val_labels_dir   : Optional[Path]  = None

    
    source_type      : str             = "unknown" 
    total_images     : int             = 0
    total_labels     : int             = 0
    splits           : List[str]       = field(default_factory=list)
    warnings         : List[str]       = field(default_factory=list)

    artifact_path    : Optional[Path]  = None
    
    @property
    def processing_dir(self) -> Path:
        return self.root_dir / "processing"