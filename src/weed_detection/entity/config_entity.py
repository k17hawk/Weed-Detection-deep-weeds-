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
    root_dir                  : Path   # artifacts/data_validation/
    ingestion_artifact_path   : Path   # artifacts/data_ingestion/data_ingestion_artifact.json
    validation_report_path    : Path   # artifacts/data_validation/validation_report.json
    validation_state_path     : Path   # artifacts/data_validation/validation_state.json
    valid_label_min           : int    # 0
    valid_label_max           : int    # 8
    imbalance_threshold       : float  # 0.01  → class < 1% of split = warning
    missing_file_threshold    : float  # 0.05  → >5% CSV rows missing  = hard fail

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir                  : Path   # artifacts/data_transformation/
    class_weights_path        : Path   # class_weights.json
    transform_config_path     : Path   # transform_config.json
    artifact_path             : Path   # data_transformation_artifact.json
    transformation_state_path : Path   # transformation_state.json
    # ── hyperparams from params.yaml ──────────────────────────────────────────
    input_size                : int    # 224
    batch_size                : int    # 32
    num_workers               : int    # 4
    sampler                   : str    # weighted | none
    pin_memory                : bool   # True

@dataclass(frozen=True)
class ModelTrainerConfig:
    # ── paths ─────────────────────────────────────────────────────────────────
    root_dir              : Path
    checkpoints_dir       : Path
    best_model_path       : Path
    final_model_path      : Path
    training_history_path : Path
    artifact_path         : Path
    trainer_state_path    : Path
    # ── architecture ──────────────────────────────────────────────────────────
    architecture          : str    # efficientnet_b3
    pretrained            : bool   # True
    num_classes           : int    # 9
    # ── data ──────────────────────────────────────────────────────────────────
    input_size            : int    # 224
    batch_size            : int    # 32
    num_workers           : int    # 4
    sampler               : str    # weighted | none
    pin_memory            : bool   # True
    # ── training ──────────────────────────────────────────────────────────────
    epochs                : int    # 30
    learning_rate         : float  # 0.0003
    weight_decay          : float  # 0.0001
    lr_scheduler          : str    # cosine | step | plateau
    warmup_epochs         : int    # 3
    early_stopping_patience: int   # 7
    # ── regularization ────────────────────────────────────────────────────────
    dropout_rate          : float  # 0.3
    label_smoothing       : float  # 0.1
    # ── checkpointing ─────────────────────────────────────────────────────────
    save_top_k            : int    # 3
    monitor_metric        : str    # val_acc | val_loss

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir               : Path
    evaluation_report_path : Path
    evaluation_state_path  : Path
    artifact_path          : Path
    # ── hyperparams ───────────────────────────────────────────────────────────
    input_size             : int    # 224  — must match trainer
    eval_batch_size        : int    # 64
    num_workers            : int    # 4
    pin_memory             : bool   # True
    eval_tta               : bool   # False
    num_classes            : int    # 9
