from pathlib import Path

from weed_detection import logger
from weed_detection.constants.constant import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from weed_detection.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    KafkaConfig,
    ModelEvaluationConfig,
    ModelRegistryConfig,
    ModelTrainerConfig,
    ModelExportConfig
)
from weed_detection.utils.utility import create_directories, read_yaml


class ConfigurationManager:
    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH,
        params_filepath: Path = PARAMS_FILE_PATH,
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([Path(self.config.artifacts_root)])

    # ── Kafka ─────────────────────────────────────────────────────────────────
    def get_kafka_config(self) -> KafkaConfig:
        k = self.config.kafka
        config = KafkaConfig(
            bootstrap_servers = k.bootstrap_servers,
            topic             = k.topic,
            consumer_group    = k.consumer_group,
        )
        logger.info(f"✅ KafkaConfig — topic={config.topic}")
        return config

    # ── Data Ingestion ────────────────────────────────────────────────────────
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        di = self.config.data_ingestion
        root_dir = Path(di.root_dir)
        create_directories([
            root_dir,
            Path(di.kafka_data_dir),
            Path(di.bad_raw_data_dir),
            Path(di.unzip_dir),
            Path(di.normalized_dir),
        ])
        config = DataIngestionConfig(
            root_dir             = root_dir,
            kafka_data_dir       = Path(di.kafka_data_dir),
            bad_raw_data_dir     = Path(di.bad_raw_data_dir),
            unzip_dir            = Path(di.unzip_dir),
            normalized_dir       = Path(di.normalized_dir),
            local_data_file      = Path(di.local_data_file),
            artifact_path        = Path(di.artifact_path),
            ingestion_state_path = Path(di.ingestion_state_path),
        )
        logger.info(f"✅ DataIngestionConfig — root={root_dir}")
        return config

    # ── Data Validation ───────────────────────────────────────────────────────
    def get_data_validation_config(self) -> DataValidationConfig:
        dv = self.config.data_validation
        root_dir = Path(dv.root_dir)
        create_directories([root_dir])
        config = DataValidationConfig(
            root_dir                = root_dir,
            ingestion_artifact_path = Path(dv.ingestion_artifact_path),
            validation_report_path  = Path(dv.validation_report_path),
            validation_state_path   = Path(dv.validation_state_path),
            valid_label_min         = int(dv.valid_label_min),
            valid_label_max         = int(dv.valid_label_max),
            imbalance_threshold     = float(dv.imbalance_threshold),
            missing_file_threshold  = float(dv.missing_file_threshold),
        )
        logger.info(f"✅ DataValidationConfig — root={root_dir}")
        return config

    # ── Data Transformation ───────────────────────────────────────────────────
    def get_data_transformation_config(self) -> DataTransformationConfig:
        dt = self.config.data_transformation
        p  = self.params.model
        root_dir = Path(dt.root_dir)
        create_directories([root_dir])
        config = DataTransformationConfig(
            root_dir                  = root_dir,
            class_weights_path        = Path(dt.class_weights_path),
            transform_config_path     = Path(dt.transform_config_path),
            artifact_path             = Path(dt.artifact_path),
            transformation_state_path = Path(dt.transformation_state_path),
            # ── from params.yaml ──────────────────────────────────────────────
            input_size                = int(p.input_size),      # 256
            batch_size                = int(p.batch_size),      # 16
            num_workers               = int(p.num_workers),
            sampler                   = str(p.sampler),
            pin_memory                = bool(p.pin_memory),
            drop_last                 = bool(p.drop_last),      # True
            weight_exponent           = float(p.weight_exponent),  # 1.0
        )
        logger.info(f"✅ DataTransformationConfig — input_size={config.input_size}  batch={config.batch_size}")
        return config

    # ── Model Trainer ─────────────────────────────────────────────────────────
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        mt = self.config.model_trainer
        p  = self.params.model
        root_dir = Path(mt.root_dir)
        create_directories([root_dir, Path(mt.checkpoints_dir)])

        config = ModelTrainerConfig(
            # ── paths ─────────────────────────────────────────────────────────
            root_dir               = root_dir,
            checkpoints_dir        = Path(mt.checkpoints_dir),
            best_model_path        = Path(mt.best_model_path),
            final_model_path       = Path(mt.final_model_path),
            training_history_path  = Path(mt.training_history_path),
            mlflow_db_path         = Path(mt.mlflow_db_path),
            artifact_path          = Path(mt.artifact_path),
            trainer_state_path     = Path(mt.trainer_state_path),
            # ── model arch ────────────────────────────────────────────────────
            architecture           = str(p.architecture),
            pretrained             = bool(p.pretrained),
            num_classes            = int(p.num_classes),
            input_size             = int(p.input_size),         # 256
            dropout_rate           = float(p.dropout_rate),
            # ── data loading ──────────────────────────────────────────────────
            batch_size             = int(p.batch_size),         # 16
            num_workers            = int(p.num_workers),
            sampler                = str(p.sampler),
            pin_memory             = bool(p.pin_memory),
            drop_last              = bool(p.drop_last),         # True
            weight_exponent        = float(p.weight_exponent),  # 1.0
            # ── optimisation ──────────────────────────────────────────────────
            epochs                 = int(p.epochs),
            learning_rate          = float(p.learning_rate),    # 1e-4
            weight_decay           = float(p.weight_decay),
            lr_scheduler           = str(p.lr_scheduler),
            warmup_epochs          = int(p.warmup_epochs),
            early_stopping_patience= int(p.early_stopping_patience),  # 5
            monitor_metric         = str(p.monitor_metric),
            grad_clip_norm         = float(p.grad_clip_norm),   # 0.5
            # ── loss ──────────────────────────────────────────────────────────
            use_focal_loss         = bool(p.use_focal_loss),    # True
            focal_gamma            = float(p.focal_gamma),      # 2.0
            label_smoothing        = float(p.label_smoothing),
            # ── AMP ───────────────────────────────────────────────────────────
            mixed_precision        = bool(p.mixed_precision),   # True
            # ── checkpointing ─────────────────────────────────────────────────
            save_top_k             = int(p.save_top_k),
            cm_log_interval        = int(p.cm_log_interval),    # 5
            # ── experiment tracking ───────────────────────────────────────────
            mlflow_tracking_uri    = str(p.mlflow_tracking_uri),
            mlflow_experiment_name = str(p.mlflow_experiment_name),
            wandb_project          = str(p.wandb_project),
            wandb_entity           = str(p.wandb_entity),
        )

        logger.info(f"✅ ModelTrainerConfig")
        logger.info(f"   Architecture    : {config.architecture}")
        logger.info(f"   Input size      : {config.input_size}")
        logger.info(f"   Batch size      : {config.batch_size}")
        logger.info(f"   Epochs          : {config.epochs}")
        logger.info(f"   LR              : {config.learning_rate}")
        logger.info(f"   Focal loss      : {config.use_focal_loss}  gamma={config.focal_gamma}")
        logger.info(f"   Mixed precision : {config.mixed_precision}")
        logger.info(f"   Grad clip       : {config.grad_clip_norm}")
        logger.info(f"   Early stop pat  : {config.early_stopping_patience}")
        logger.info(f"   MLflow URI      : {config.mlflow_tracking_uri}")
        logger.info(f"   W&B project     : {config.wandb_project}")
        return config

    # ── Model Registry ────────────────────────────────────────────────────────
    def get_model_registry_config(self) -> ModelRegistryConfig:
        mr = self.config.model_registry
        root_dir     = Path(mr.root_dir)
        champion_dir = Path(mr.champion_dir)
        runs_dir     = Path(mr.runs_dir)
        create_directories([root_dir, champion_dir, runs_dir])
        config = ModelRegistryConfig(
            root_dir               = root_dir,
            champion_dir           = champion_dir,
            champion_model_path    = Path(mr.champion_model_path),
            champion_metadata_path = Path(mr.champion_metadata_path),
            runs_dir               = runs_dir,
        )
        logger.info(f"✅ ModelRegistryConfig — champion={champion_dir}")
        return config

    # ── Model Evaluation ─────────────────────────────────────────────────────
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        me = self.config.model_evaluation
        p  = self.params.model
        root_dir = Path(me.root_dir)
        create_directories([root_dir])
        config = ModelEvaluationConfig(
            root_dir                 = root_dir,
            evaluation_report_path   = Path(me.evaluation_report_path),
            evaluation_history_path  = Path(me.evaluation_history_path),
            evaluation_state_path    = Path(me.evaluation_state_path),
            artifact_path            = Path(me.artifact_path),
            # ── hyperparams ───────────────────────────────────────────────────
            input_size               = int(p.input_size),         # 256
            eval_batch_size          = int(p.eval_batch_size),    # 64
            num_workers              = int(p.num_workers),
            pin_memory               = bool(p.pin_memory),
            eval_tta                 = bool(p.eval_tta),
            num_classes              = int(p.num_classes),
            # ── promotion ─────────────────────────────────────────────────────
            promotion_metric         = str(p.promotion_metric),
            min_promotion_threshold  = float(p.min_promotion_threshold),
            # ── experiment tracking ───────────────────────────────────────────
            mlflow_tracking_uri      = str(p.mlflow_tracking_uri),
            mlflow_experiment_name   = str(p.mlflow_experiment_name),
            wandb_project            = str(p.wandb_project),
            wandb_entity             = str(p.wandb_entity),
        )
        logger.info(f"✅ ModelEvaluationConfig")
        logger.info(f"   Input size         : {config.input_size}")
        logger.info(f"   Eval batch size    : {config.eval_batch_size}")
        logger.info(f"   Promotion metric   : {config.promotion_metric}")
        logger.info(f"   Min threshold      : {config.min_promotion_threshold}")
        logger.info(f"   TTA                : {config.eval_tta}")
        return config
    
    def get_model_export_config(self) -> ModelExportConfig:
        """Get model export configuration"""
        me = self.config.model_export
        p  = self.params.model

        root_dir    = Path(me.root_dir)
        exports_dir = Path(me.exports_dir)
        create_directories([root_dir, exports_dir])

        config = ModelExportConfig(
            root_dir             = root_dir,
            exports_dir          = exports_dir,
            onnx_model_path      = Path(me.onnx_model_path),
            onnx_fp16_model_path = Path(me.onnx_fp16_model_path),
            model_info_path      = Path(me.model_info_path),
            export_state_path    = Path(me.export_state_path),
            artifact_path        = Path(me.artifact_path),
            input_size           = int(p.input_size),
            num_classes          = int(p.num_classes),
            opset_version        = int(p.export_opset_version),
            export_fp16          = bool(p.export_fp16),
            validate_onnx        = bool(p.export_validate_onnx),
            dynamic_batch        = bool(p.export_dynamic_batch),
        )
        
        logger.info(f"✅ ModelExportConfig")
        logger.info(f"   ONNX path    : {config.onnx_model_path}")
        logger.info(f"   FP16         : {config.export_fp16}")
        logger.info(f"   Validate     : {config.validate_onnx}")
        return config