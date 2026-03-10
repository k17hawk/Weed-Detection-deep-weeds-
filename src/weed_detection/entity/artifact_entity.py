# weed_detection/entity/artifact_entity.py
# ─────────────────────────────────────────────────────────────────────────────
# Immutable dataclasses that carry outputs between pipeline stages.
# Each stage produces one artifact and consumes the upstream artifact.
# ─────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


# ── Kafka ─────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class KafkaArtifact:
    topic          : str
    partition      : int
    offset         : int
    key            : Optional[str]
    artifact_path  : Path
    received_at    : datetime


# ── Data Ingestion ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DataIngestionArtifact:
    """
    Produced by  : data_ingestion.py
    Consumed by  : data_validation.py

    normalized_dir layout:
      <normalized_dir>/
        images/
          train/  val/  test/
        labels/
          train.csv  val.csv  test.csv    # columns: Filename, Label, Species
    """
    normalized_dir  : Path
    artifact_path   : Path
    ingested_at     : datetime
    version_id      : str    # e.g. v_20260307_192121_5a0b6671da93a189


# ── Data Validation ───────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DataValidationArtifact:
    """
    Produced by  : data_validation.py
    Consumed by  : data_transformation.py
    """
    ingestion_artifact     : DataIngestionArtifact
    is_valid               : bool
    failed_checks          : List[str]
    warnings               : List[str]
    split_stats            : Dict
    class_distribution     : Dict
    validated_at           : datetime
    validation_report_path : Path


# ── Data Transformation ───────────────────────────────────────────────────────
@dataclass(frozen=True)
class DataTransformationArtifact:
    """
    Produced by  : data_transformation.py
    Consumed by  : model_trainer.py

    Carries resolved paths to image folders + CSVs for all splits.
    Also carries pre-computed class weights for WeightedRandomSampler.

    class_weights : List[float]  — one per class, index matches label int
      Formula (notebook Cell 13):
        class_weights[c] = total_samples / (NUM_CLASSES * label_counts[c])
        optionally raised to weight_exponent (default 1.0)
    """
    validation_artifact   : DataValidationArtifact
    # ── split paths ───────────────────────────────────────────────────────────
    train_images_dir      : Path
    train_csv_path        : Path
    val_images_dir        : Optional[Path]
    val_csv_path          : Optional[Path]
    test_images_dir       : Optional[Path]
    test_csv_path         : Optional[Path]
    # ── class weights ─────────────────────────────────────────────────────────
    class_weights_path    : Path
    class_weights         : List[float]   # len == NUM_CLASSES
    # ── transform config ──────────────────────────────────────────────────────
    transform_config_path : Path
    input_size            : int           # 256
    batch_size            : int           # 16
    num_workers           : int
    sampler               : str
    pin_memory            : bool
    drop_last             : bool          # True for train loader
    weight_exponent       : float         # 1.0
    # ── timing ────────────────────────────────────────────────────────────────
    transformed_at        : datetime
    artifact_path         : Path


# ── Model Trainer ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ModelTrainerArtifact:
    """
    Produced by  : model_trainer.py
    Consumed by  : model_evaluation.py

    Stores every parameter from the notebook so downstream stages can
    reconstruct the exact same model architecture for inference.

    Key additions vs previous version:
      - use_focal_loss / focal_gamma    — FocalLoss support
      - mixed_precision                 — AMP flag
      - grad_clip_norm                  — 0.5 from notebook
      - weight_exponent                 — class weight formula exponent
      - drop_last                       — train loader flag
      - mlflow_run_id / wandb_run_id    — cross-link experiment runs
      - nan_batches_total               — NaN batch count for debugging
      - cm_log_interval                 — confusion matrix cadence
    """
    transformation_artifact : DataTransformationArtifact
    # ── model files ───────────────────────────────────────────────────────────
    best_model_path         : Path    # saved whenever val_acc improves
    final_model_path        : Path    # always saved in finally block
    checkpoints_dir         : Path    # top-K checkpoints
    # ── architecture ──────────────────────────────────────────────────────────
    architecture            : str
    num_classes             : int
    pretrained              : bool
    dropout_rate            : float
    total_params            : int
    trainable_params        : int
    # ── training config ───────────────────────────────────────────────────────
    epochs_trained          : int
    best_epoch              : int
    best_val_acc            : float
    best_val_loss           : float
    final_train_acc         : float
    final_train_loss        : float
    learning_rate           : float
    weight_decay            : float
    lr_scheduler            : str
    warmup_epochs           : int
    early_stopping_patience : int
    monitor_metric          : str
    batch_size              : int
    input_size              : int     # 256
    sampler                 : str
    drop_last               : bool
    weight_exponent         : float
    grad_clip_norm          : float   # 0.5
    # ── loss ──────────────────────────────────────────────────────────────────
    use_focal_loss          : bool    # True
    focal_gamma             : float   # 2.0
    label_smoothing         : float   # used only when use_focal_loss=False
    # ── AMP ───────────────────────────────────────────────────────────────────
    mixed_precision         : bool    # True
    nan_batches_total       : int     # total NaN batches skipped in training
    # ── per-class val accuracy at best epoch ──────────────────────────────────
    per_class_val_acc       : Dict[str, float]  # {"0": 0.91, ..., "8": 0.99}
    # ── experiment tracking ───────────────────────────────────────────────────
    mlflow_run_id           : str     # MLflow run ID for cross-linking
    wandb_run_id            : str     # W&B run ID for cross-linking
    wandb_run_url           : str     # W&B run URL
    mlflow_tracking_uri     : str
    mlflow_experiment_name  : str
    # ── hardware ──────────────────────────────────────────────────────────────
    device                  : str
    cuda_version            : Optional[str]
    # ── timing ────────────────────────────────────────────────────────────────
    training_history_path   : Path
    trained_at              : datetime
    total_training_time_s   : float
    artifact_path           : Path


# ── Model Registry ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ModelRegistryEntry:
    """
    A single registered training run in the model registry.
    Written to model_registry/runs/<run_id>/run_metadata.json.
    """
    run_id              : str
    run_number          : int
    model_path          : Path    # runs/<run_id>/model.pth
    architecture        : str
    epochs_trained      : int
    best_val_acc        : float
    # ── test metrics ──────────────────────────────────────────────────────────
    accuracy            : float
    top2_accuracy       : float
    macro_f1            : float
    weighted_f1         : float
    per_class_metrics   : Dict[str, Dict[str, float]]
    # ── mlflow ────────────────────────────────────────────────────────────────
    mlflow_run_id       : str
    mlflow_model_version: Optional[str]   # set after mlflow.register_model
    mlflow_stage        : str             # None | Staging | Production | Archived
    # ── registry state ────────────────────────────────────────────────────────
    is_champion         : bool
    promoted_at         : Optional[datetime]
    registered_at       : datetime


# ── Model Evaluation ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ModelEvaluationArtifact:

    trainer_artifact          : ModelTrainerArtifact
    # ── test metrics ──────────────────────────────────────────────────────────
    accuracy                  : float
    top2_accuracy             : float
    macro_f1                  : float
    weighted_f1               : float
    per_class_metrics         : Dict[str, Dict[str, float]]
    confusion_matrix          : List[List[int]]    # 9×9 raw counts
    # ── registry ──────────────────────────────────────────────────────────────
    registry_entry            : ModelRegistryEntry
    is_new_champion           : bool
    champion_model_path       : Path
    previous_champion_metric  : Optional[float]
    # ── eval config ───────────────────────────────────────────────────────────
    promotion_metric          : str
    tta_enabled               : bool
    eval_batch_size           : int
    # ── timing ────────────────────────────────────────────────────────────────
    evaluated_at              : datetime
    evaluation_report_path    : Path
    evaluation_history_path   : Path
    artifact_path             : Optional[Path] = None