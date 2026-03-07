from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KafkaConsumerConfig:
    """
    Config for the Kafka consumer stage.
    Populated from config.yaml  kafka:  block.
    """
    kafka_data_dir : Path    # artifacts/data_ingestion/kafka_data/
    broker         : str     # localhost:9092
    topic          : str     # drone-images-topic
    group_id       : str     # drone-group


@dataclass(frozen=True)
class KafkaProducerConfig:
    bootstrap_servers    : str
    topic                : str
    aws_region           : str
    queue_url            : str
    aws_access_key_id    : str
    aws_secret_access_key: str


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir        : Path   # artifacts/data_ingestion/
    local_data_file : Path
    unzip_dir       : Path   # artifacts/data_ingestion/