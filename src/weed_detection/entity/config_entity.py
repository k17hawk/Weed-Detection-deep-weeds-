from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KafkaProducerConfig:
    """Kafka producer + SQS poller config. Secrets injected from constant.py."""
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
    kafka_data_dir  – where valid zips are saved
    bad_raw_data_dir – where invalid filenames are quarantined
    """
    broker          : str
    topic           : str
    group_id        : str
    kafka_data_dir  : Path   # artifacts/data_ingestion/kafka_data/
    bad_raw_data_dir: Path   # artifacts/data_ingestion/bad_raw_data/


@dataclass(frozen=True)
class DataIngestionConfig:
    """Data ingestion config – reads KafkaArtifact, unzips, normalizes."""
    root_dir        : Path   # artifacts/data_ingestion/
    kafka_data_dir  : Path   # artifacts/data_ingestion/kafka_data/
    bad_raw_data_dir: Path   # artifacts/data_ingestion/bad_raw_data/
    unzip_dir       : Path   # artifacts/data_ingestion/unzipped/
    local_data_file : Path   # kafka_data/latest_artifact.json  (pointer)