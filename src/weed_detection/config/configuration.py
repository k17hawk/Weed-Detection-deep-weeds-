from pathlib import Path
from weed_detection import logger
from weed_detection.constants.constant import (
    CONFIG_FILE_PATH, PARAMS_FILE_PATH,
    AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION,
    DATA_SOURCE, QUEUE_URL,
)
from weed_detection.utils.utility import read_yaml, create_directories
from weed_detection.entity.config_entity import (
    KafkaProducerConfig,
    KafkaConsumerConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)


class ConfigurationManager:

    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
    ):
        self.config = read_yaml(config_filepath)

        try:
            self.params = read_yaml(params_filepath)
        except FileNotFoundError:
            logger.warning("⚠️  params.yaml not found – skipping")
            self.params = None

        create_directories([self.config.artifacts_root])

    # ── Kafka producer ────────────────────────────────────────────────────────

    def get_kafka_producer_config(self) -> KafkaProducerConfig:
        kafka  = self.config.kafka
        config = KafkaProducerConfig(
            bootstrap_servers     = kafka.bootstrap_servers,
            topic                 = kafka.topic,
            aws_region            = AWS_REGION,
            queue_url             = QUEUE_URL,
            aws_access_key_id     = AWS_ACCESS_KEY,
            aws_secret_access_key = AWS_SECRET_KEY,
        )
        logger.info(f"✅ KafkaProducerConfig")
        logger.info(f"   Broker : {config.bootstrap_servers}")
        logger.info(f"   Topic  : {config.topic}")
        return config

    # ── Kafka consumer ────────────────────────────────────────────────────────

    def get_kafka_consumer_config(self) -> KafkaConsumerConfig:
        kafka = self.config.kafka
        di    = self.config.data_ingestion

        kafka_data_dir   = Path(di.kafka_data_dir)
        bad_raw_data_dir = Path(di.bad_raw_data_dir)
        create_directories([kafka_data_dir, bad_raw_data_dir])

        config = KafkaConsumerConfig(
            broker           = kafka.bootstrap_servers,
            topic            = kafka.topic,
            group_id         = kafka.consumer_group,
            kafka_data_dir   = kafka_data_dir,
            bad_raw_data_dir = bad_raw_data_dir,
        )
        logger.info(f"✅ KafkaConsumerConfig")
        logger.info(f"   Broker : {config.broker}")
        logger.info(f"   Topic  : {config.topic}")
        logger.info(f"   Group  : {config.group_id}")
        return config

    # ── Data ingestion ────────────────────────────────────────────────────────

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        di = self.config.data_ingestion

        root_dir             = Path(di.root_dir)
        kafka_data_dir       = Path(di.kafka_data_dir)
        bad_raw_data_dir     = Path(di.bad_raw_data_dir)
        unzip_dir            = Path(di.unzip_dir)
        normalized_dir       = Path(di.normalized_dir)
        local_data_file      = Path(di.local_data_file)
        artifact_path        = Path(di.artifact_path)
        ingestion_state_path = Path(di.ingestion_state_path)

        create_directories([
            root_dir, kafka_data_dir, bad_raw_data_dir, unzip_dir, normalized_dir
        ])

        config = DataIngestionConfig(
            root_dir             = root_dir,
            kafka_data_dir       = kafka_data_dir,
            bad_raw_data_dir     = bad_raw_data_dir,
            unzip_dir            = unzip_dir,
            normalized_dir       = normalized_dir,
            local_data_file      = local_data_file,
            artifact_path        = artifact_path,
            ingestion_state_path = ingestion_state_path,
        )
        logger.info(f"✅ DataIngestionConfig")
        logger.info(f"   Root       : {root_dir}")
        logger.info(f"   Kafka data : {kafka_data_dir}")
        logger.info(f"   State      : {ingestion_state_path}")
        return config

    # ── Data validation ───────────────────────────────────────────────────────

    def get_data_validation_config(self) -> DataValidationConfig:
        dv = self.config.data_validation

        root_dir                = Path(dv.root_dir)
        ingestion_artifact_path = Path(dv.ingestion_artifact_path)
        validation_report_path  = Path(dv.validation_report_path)
        validation_state_path   = Path(dv.validation_state_path)

        create_directories([root_dir])

        config = DataValidationConfig(
            root_dir                = root_dir,
            ingestion_artifact_path = ingestion_artifact_path,
            validation_report_path  = validation_report_path,
            validation_state_path   = validation_state_path,
            valid_label_min         = int(dv.valid_label_min),
            valid_label_max         = int(dv.valid_label_max),
            imbalance_threshold     = float(dv.imbalance_threshold),
            missing_file_threshold  = float(dv.missing_file_threshold),
        )
        logger.info(f"✅ DataValidationConfig")
        logger.info(f"   Root               : {root_dir}")
        logger.info(f"   Valid labels       : [{config.valid_label_min}..{config.valid_label_max}]")
        logger.info(f"   Imbalance thresh   : {config.imbalance_threshold}")
        logger.info(f"   Missing file thresh: {config.missing_file_threshold}")
        return config

    # ── Data transformation ───────────────────────────────────────────────────

    def get_data_transformation_config(self) -> DataTransformationConfig:
        dt = self.config.data_transformation

        if self.params is None:
            raise RuntimeError("params.yaml is required for data transformation")
        model_params = self.params.model

        root_dir                  = Path(dt.root_dir)
        class_weights_path        = Path(dt.class_weights_path)
        transform_config_path     = Path(dt.transform_config_path)
        artifact_path             = Path(dt.artifact_path)
        transformation_state_path = Path(dt.transformation_state_path)

        create_directories([root_dir])

        config = DataTransformationConfig(
            root_dir                  = root_dir,
            class_weights_path        = class_weights_path,
            transform_config_path     = transform_config_path,
            artifact_path             = artifact_path,
            transformation_state_path = transformation_state_path,
            input_size                = int(model_params.input_size),
            batch_size                = int(model_params.batch_size),
            num_workers               = int(model_params.num_workers),
            sampler                   = str(model_params.sampler),
            pin_memory                = bool(model_params.pin_memory),
        )
        logger.info(f"✅ DataTransformationConfig")
        logger.info(f"   Root       : {root_dir}")
        logger.info(f"   Input size : {config.input_size}")
        logger.info(f"   Batch size : {config.batch_size}")
        logger.info(f"   Sampler    : {config.sampler}")
        return config

    # ── Model trainer ─────────────────────────────────────────────────────────

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        mt = self.config.model_trainer

        if self.params is None:
            raise RuntimeError("params.yaml is required for model training")
        p = self.params.model

        root_dir              = Path(mt.root_dir)
        checkpoints_dir       = Path(mt.checkpoints_dir)
        best_model_path       = Path(mt.best_model_path)
        final_model_path      = Path(mt.final_model_path)
        training_history_path = Path(mt.training_history_path)
        artifact_path         = Path(mt.artifact_path)
        trainer_state_path    = Path(mt.trainer_state_path)

        create_directories([root_dir, checkpoints_dir])

        config = ModelTrainerConfig(
            root_dir               = root_dir,
            checkpoints_dir        = checkpoints_dir,
            best_model_path        = best_model_path,
            final_model_path       = final_model_path,
            training_history_path  = training_history_path,
            artifact_path          = artifact_path,
            trainer_state_path     = trainer_state_path,
            architecture           = str(p.architecture),
            pretrained             = bool(p.pretrained),
            num_classes            = int(p.num_classes),
            input_size             = int(p.input_size),
            batch_size             = int(p.batch_size),
            num_workers            = int(p.num_workers),
            sampler                = str(p.sampler),
            pin_memory             = bool(p.pin_memory),
            epochs                 = int(p.epochs),
            learning_rate          = float(p.learning_rate),
            weight_decay           = float(p.weight_decay),
            lr_scheduler           = str(p.lr_scheduler),
            warmup_epochs          = int(p.warmup_epochs),
            early_stopping_patience= int(p.early_stopping_patience),
            dropout_rate           = float(p.dropout_rate),
            label_smoothing        = float(p.label_smoothing),
            save_top_k             = int(p.save_top_k),
            monitor_metric         = str(p.monitor_metric),
        )
        logger.info(f"✅ ModelTrainerConfig")
        logger.info(f"   Architecture : {config.architecture}")
        logger.info(f"   Pretrained   : {config.pretrained}")
        logger.info(f"   Num classes  : {config.num_classes}")
        logger.info(f"   Input size   : {config.input_size}")
        logger.info(f"   Epochs       : {config.epochs}")
        logger.info(f"   LR           : {config.learning_rate}")
        logger.info(f"   Scheduler    : {config.lr_scheduler}")
        logger.info(f"   Warmup       : {config.warmup_epochs} epochs")
        logger.info(f"   Early stop   : patience={config.early_stopping_patience}")
        logger.info(f"   Label smooth : {config.label_smoothing}")
        logger.info(f"   Dropout      : {config.dropout_rate}")
        logger.info(f"   Save top-k   : {config.save_top_k}")
        logger.info(f"   Monitor      : {config.monitor_metric}")
        logger.info(f"   Checkpoints  : {checkpoints_dir}")
        logger.info(f"   Best model   : {best_model_path}")
        return config