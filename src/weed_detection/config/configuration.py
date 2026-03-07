import os
from pathlib import Path
from ensure import ensure_annotations
from weed_detection import logger
from weed_detection.constants.constant import DATA_SOURCE, CONFIG_FILE_PATH, PARAMS_FILE_PATH
from weed_detection.utils.utility import read_yaml, create_directories
from weed_detection.entity.config_entity import DataIngestionConfig


# configuration.py
import os
from pathlib import Path
from ensure import ensure_annotations
from weed_detection import logger
from weed_detection.constants.constant import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from weed_detection.utils.utility import read_yaml, create_directories
from weed_detection.entity.config_entity import DataIngestionConfig

# config/configuration.py
import os
from pathlib import Path
from weed_detection import logger
from weed_detection.constants.constant import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from weed_detection.utils.utility import read_yaml, create_directories
from weed_detection.entity.config_entity import (
    DataIngestionConfig,
    KafkaConsumerConfig,
)

# kafka_data lives inside data_ingestion artifacts
KAFKA_DATA_SUBDIR = "kafka_data"


class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH,
    ):
        self.config = read_yaml(config_filepath)

        # params.yaml is only needed for training stages – make it optional
        try:
            self.params = read_yaml(params_filepath)
        except FileNotFoundError:
            logger.warning(
                f"⚠️  params.yaml not found at {params_filepath} – "
                "skipping (not needed for ingestion stages)"
            )
            self.params = None

        create_directories([self.config.artifacts_root])

    # ── Kafka consumer config ─────────────────────────────────────────────────

    def get_kafka_consumer_config(self) -> KafkaConsumerConfig:
        kafka = self.config.kafka
        di    = self.config.data_ingestion

        # artifacts/data_ingestion/kafka_data/
        kafka_data_dir = Path(di.root_dir) / KAFKA_DATA_SUBDIR
        create_directories([kafka_data_dir])

        config = KafkaConsumerConfig(
            kafka_data_dir = kafka_data_dir,
            broker         = kafka.bootstrap_servers,
            topic          = kafka.topic,
            group_id       = kafka.consumer_group,
        )

        logger.info(f"✅ KafkaConsumerConfig ready – saving zips to: {kafka_data_dir}")
        return config

    # ── Data ingestion config ─────────────────────────────────────────────────

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir        = Path(config.root_dir),
            local_data_file = Path(config.root_dir) / "raw.zip",  # placeholder
            unzip_dir       = Path(config.unzip_dir),
        )

        logger.info("✅ DataIngestionConfig ready")
        return data_ingestion_config