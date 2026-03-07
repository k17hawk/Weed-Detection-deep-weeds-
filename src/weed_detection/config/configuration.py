from pathlib import Path
from weed_detection import logger
from weed_detection.constants.constant import (
    CONFIG_FILE_PATH,
    PARAMS_FILE_PATH,
    AWS_ACCESS_KEY,
    AWS_SECRET_KEY,
    AWS_REGION,
    DATA_SOURCE,
    QUEUE_URL,
    KAFKA_BROKER,
    KAFKA_TOPIC,
)
from weed_detection.utils.utility import read_yaml, create_directories
from weed_detection.entity.config_entity import (
    KafkaProducerConfig,
    KafkaConsumerConfig,
    DataIngestionConfig,
)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
    ):
        self.config = read_yaml(config_filepath)

        # params.yaml only needed for training stages
        try:
            self.params = read_yaml(params_filepath)
        except FileNotFoundError:
            logger.warning("⚠️  params.yaml not found – skipping (not needed for ingestion)")
            self.params = None

        create_directories([self.config.artifacts_root])

    # ── Kafka producer ────────────────────────────────────────────────────────

    def get_kafka_producer_config(self) -> KafkaProducerConfig:
        kafka = self.config.kafka

        config = KafkaProducerConfig(
            bootstrap_servers    = kafka.bootstrap_servers,
            topic                = kafka.topic,
            aws_region           = AWS_REGION,
            queue_url            = QUEUE_URL,
            aws_access_key_id    = AWS_ACCESS_KEY,
            aws_secret_access_key= AWS_SECRET_KEY,
        )

        logger.info(f"✅ KafkaProducerConfig ready – topic: {config.topic}")
        return config

    # ── Kafka consumer ────────────────────────────────────────────────────────

    def get_kafka_consumer_config(self) -> KafkaConsumerConfig:
        kafka = self.config.kafka
        di    = self.config.data_ingestion

        kafka_data_dir   = Path(di.kafka_data_dir)
        bad_raw_data_dir = Path(di.bad_raw_data_dir)

        create_directories([kafka_data_dir, bad_raw_data_dir])

        config = KafkaConsumerConfig(
            broker          = kafka.bootstrap_servers,
            topic           = kafka.topic,
            group_id        = kafka.consumer_group,
            kafka_data_dir  = kafka_data_dir,
            bad_raw_data_dir= bad_raw_data_dir,
        )

        logger.info(f"✅ KafkaConsumerConfig ready")
        logger.info(f"   Good data : {kafka_data_dir}")
        logger.info(f"   Bad data  : {bad_raw_data_dir}")
        return config

    # ── Data ingestion ────────────────────────────────────────────────────────

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        di = self.config.data_ingestion

        root_dir         = Path(di.root_dir)
        kafka_data_dir   = Path(di.kafka_data_dir)
        bad_raw_data_dir = Path(di.bad_raw_data_dir)
        unzip_dir        = Path(di.unzip_dir)
        local_data_file  = Path(di.local_data_file)

        create_directories([root_dir, kafka_data_dir, bad_raw_data_dir, unzip_dir])

        config = DataIngestionConfig(
            root_dir         = root_dir,
            kafka_data_dir   = kafka_data_dir,
            bad_raw_data_dir = bad_raw_data_dir,
            unzip_dir        = unzip_dir,
            local_data_file  = local_data_file,
        )

        logger.info(f"✅ DataIngestionConfig ready")
        logger.info(f"   Root      : {root_dir}")
        logger.info(f"   Kafka data: {kafka_data_dir}")
        logger.info(f"   Unzip     : {unzip_dir}")
        return config