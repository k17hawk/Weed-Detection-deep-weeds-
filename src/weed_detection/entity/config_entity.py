from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KafkaProducerConfig:
    """
    Kafka producer + SQS poller config.
    Sends S3 pointer (bucket + key) to Kafka — no file download.
    Secrets injected from constant.py.
    """
    bootstrap_servers    : str
    topic                : str
    aws_region           : str
    queue_url            : str
    aws_access_key_id    : str
    aws_secret_access_key: str


@dataclass(frozen=True)
class KafkaConsumerConfig:
    """
    Kafka consumer config.
    Receives S3 pointer → downloads zip from S3 → validates → saves locally.
    kafka_data_dir   – valid zips saved here
    bad_raw_data_dir – invalid filenames quarantined here
    """
    broker          : str
    topic           : str
    group_id        : str
    kafka_data_dir  : Path   # artifacts/data_ingestion/kafka_data/
    bad_raw_data_dir: Path   # artifacts/data_ingestion/bad_raw_data/


@dataclass(frozen=True)
class DataIngestionConfig:
    """
    Data ingestion stage config.
    Version-aware: skips if latest zip already processed.
    All paths sourced from config.yaml via ConfigurationManager.
    """
    root_dir             : Path   # artifacts/data_ingestion/
    kafka_data_dir       : Path   # artifacts/data_ingestion/kafka_data/
    bad_raw_data_dir     : Path   # artifacts/data_ingestion/bad_raw_data/
    unzip_dir            : Path   # artifacts/data_ingestion/unzipped/
    normalized_dir       : Path   # artifacts/data_ingestion/normalized/
    local_data_file      : Path   # kafka_data/latest_artifact.json
    artifact_path        : Path   # artifacts/data_ingestion/data_ingestion_artifact.json
    ingestion_state_path : Path   # artifacts/data_ingestion/ingestion_state.json




@dataclass(frozen=True)
class DataValidationConfig:
    """
    Data validation stage config.
    Version-aware: skips if DataIngestionArtifact version already validated.

    Reads  : artifacts/data_ingestion/data_ingestion_artifact.json
    Writes : artifacts/data_validation/validation_report.json
             artifacts/data_validation/validation_state.json

    Hard failures  → is_valid=False  → pipeline stops
    Soft warnings  → logged          → pipeline continues

    deep-weed specific:
      valid_labels      : [0..8]   (0 = background, 1-8 = weed species)
      imbalance_threshold: 0.01    (class < 1% of split = warning)
      missing_file_threshold: 0.05 (>5% CSV rows missing on disk = hard fail)
    """
    root_dir                  : Path   # artifacts/data_validation/
    ingestion_artifact_path   : Path   # artifacts/data_ingestion/data_ingestion_artifact.json
    validation_report_path    : Path   # artifacts/data_validation/validation_report.json
    validation_state_path     : Path   # artifacts/data_validation/validation_state.json
    valid_label_min           : int    # 0
    valid_label_max           : int    # 8
    imbalance_threshold       : float  # 0.01  → class < 1% of split = warning
    missing_file_threshold    : float  # 0.05  → >5% CSV rows missing  = hard fail