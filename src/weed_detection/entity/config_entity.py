from dataclasses import dataclass
from pathlib import Path


# ── Kafka ─────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class KafkaConfig:
    bootstrap_servers : str
    topic             : str
    consumer_group    : str


# ── Data Ingestion ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir             : Path
    kafka_data_dir       : Path
    bad_raw_data_dir     : Path
    unzip_dir            : Path
    normalized_dir       : Path
    local_data_file      : Path
    artifact_path        : Path
    ingestion_state_path : Path


# ── Data Validation ───────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DataValidationConfig:
    root_dir                : Path
    ingestion_artifact_path : Path
    validation_report_path  : Path
    validation_state_path   : Path
    valid_label_min         : int
    valid_label_max         : int
    imbalance_threshold     : float
    missing_file_threshold  : float


# ── Data Transformation ───────────────────────────────────────────────────────
@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir                  : Path
    class_weights_path        : Path
    transform_config_path     : Path
    artifact_path             : Path
    transformation_state_path : Path
    # ── from params.yaml ──────────────────────────────────────────────────────
    input_size                : int    # 256  — notebook Cell 6
    batch_size                : int    # 16
    num_workers               : int
    sampler                   : str
    pin_memory                : bool
    drop_last                 : bool   # notebook Cell 16
    weight_exponent           : float  # notebook Cell 13: 1.0 standard, 0.5 softer


# ── Model Trainer ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ModelTrainerConfig:
    """
    Ground truth for all trainer parameters mapped from notebook.

    Key differences from previous version:
      - input_size       : 256  (not 224)
      - batch_size       : 16   (not 32)
      - learning_rate    : 1e-4 (not 3e-4)
      - early_stopping_patience: 5 (not 7)
      - grad_clip_norm   : 0.5  (not 1.0)
      - use_focal_loss   : True (NEW — FocalLoss replaces CrossEntropyLoss)
      - focal_gamma      : 2.0  (NEW)
      - weight_exponent  : 1.0  (NEW — controls class weight strength)
      - mixed_precision  : True (NEW — AMP GradScaler)
      - cm_log_interval  : 5    (NEW — confusion matrix every 5 epochs)
      - drop_last        : True (NEW — train loader drop_last)
      - mlflow_tracking_uri / mlflow_experiment_name (NEW)
      - wandb_project / wandb_entity (NEW)
    """
    # ── paths ─────────────────────────────────────────────────────────────────
    root_dir              : Path
    checkpoints_dir       : Path
    best_model_path       : Path
    final_model_path      : Path
    training_history_path : Path
    mlflow_db_path        : Path    # local sqlite for MLflow tracking
    artifact_path         : Path
    trainer_state_path    : Path
    # ── model arch ────────────────────────────────────────────────────────────
    architecture          : str    # efficientnet_b3
    pretrained            : bool
    num_classes           : int    # 9
    input_size            : int    # 256
    dropout_rate          : float  # 0.3
    # ── data loading ──────────────────────────────────────────────────────────
    batch_size            : int    # 16
    num_workers           : int
    sampler               : str    # weighted
    pin_memory            : bool
    drop_last             : bool   # True
    weight_exponent       : float  # 1.0
    # ── optimisation ──────────────────────────────────────────────────────────
    epochs                : int    # 30
    learning_rate         : float  # 1e-4
    weight_decay          : float  # 1e-4
    lr_scheduler          : str    # cosine | plateau
    warmup_epochs         : int    # 3
    early_stopping_patience: int   # 5
    monitor_metric        : str    # val_acc | val_loss
    grad_clip_norm        : float  # 0.5
    # ── loss function ─────────────────────────────────────────────────────────
    use_focal_loss        : bool   # True  — FocalLoss
    focal_gamma           : float  # 2.0
    label_smoothing       : float  # 0.1 — used only when use_focal_loss=False
    # ── AMP ───────────────────────────────────────────────────────────────────
    mixed_precision       : bool   # True
    # ── checkpointing ─────────────────────────────────────────────────────────
    save_top_k            : int    # 3
    cm_log_interval       : int    # 5
    # ── experiment tracking ───────────────────────────────────────────────────
    mlflow_tracking_uri   : str    # sqlite:///mlflow.db
    mlflow_experiment_name: str    # weed-detection
    wandb_project         : str    # weed-detection
    wandb_entity          : str    # kumardahal788-loyalist-college-of-toronto


# ── Model Registry ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ModelRegistryConfig:
    """
    Cross-run model store.
    Every training run is registered under runs/<run_id>/.
    Champion = run with the highest promotion_metric across all runs.
    """
    root_dir               : Path
    champion_dir           : Path
    champion_model_path    : Path
    champion_metadata_path : Path
    runs_dir               : Path


# ── Model Evaluation ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ModelEvaluationConfig:
    """
    Model evaluation + champion selection stage.

    Evaluates best_model.pth from current training run on the test split.
    Compares weighted_f1 against the current champion.
    Promotes to champion/model.pth if this run is better.

    W&B logs:
      confusion matrix (wandb.plot.confusion_matrix)
      per-class F1 bar chart
      overall metrics table

    MLflow logs:
      test_accuracy, test_macro_f1, test_weighted_f1, test_top2_accuracy
      per_class_metrics.json artifact
      registers model as 'DeepWeedClassifier' → stage: Staging / Production
    """
    root_dir                 : Path
    evaluation_report_path   : Path
    evaluation_history_path  : Path
    evaluation_state_path    : Path
    artifact_path            : Path
    # ── hyperparams ───────────────────────────────────────────────────────────
    input_size               : int    # 256 — must match trainer
    eval_batch_size          : int    # 64
    num_workers              : int
    pin_memory               : bool
    eval_tta                 : bool
    num_classes              : int    # 9
    # ── promotion ─────────────────────────────────────────────────────────────
    promotion_metric         : str    # weighted_f1 | macro_f1 | accuracy
    min_promotion_threshold  : float  # 0.80
    # ── experiment tracking ───────────────────────────────────────────────────
    mlflow_tracking_uri      : str
    mlflow_experiment_name   : str
    wandb_project            : str
    wandb_entity             : str