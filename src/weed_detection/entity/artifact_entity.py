from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict


@dataclass(frozen=True)
class KafkaArtifact:
    kafka_data_dir   : Path
    version_dir      : Path
    zip_file_path    : Path
    s3_bucket        : str
    s3_key           : str
    source_url       : str
    file_hash        : str
    file_size_bytes  : int
    original_filename: str
    kafka_topic      : str
    kafka_partition  : int
    kafka_offset     : int
    received_at      : datetime
    artifact_path    : Optional[Path] = None


@dataclass(frozen=True)
class DataIngestionArtifact:
    kafka_artifact   : KafkaArtifact
    unzip_dir        : Path
    normalized_dir   : Path
    train_images_dir : Path
    train_labels_dir : Path
    val_images_dir   : Optional[Path] = None
    val_labels_dir   : Optional[Path] = None
    test_images_dir  : Optional[Path] = None
    test_labels_dir  : Optional[Path] = None
    source_type      : str            = "unknown"
    total_images     : int            = 0
    total_labels     : int            = 0
    splits           : List[str]      = field(default_factory=list)
    warnings         : List[str]      = field(default_factory=list)
    artifact_path    : Optional[Path] = None


@dataclass(frozen=True)
class DataValidationArtifact:
    ingestion_artifact    : DataIngestionArtifact
    is_valid              : bool
    failed_checks         : List[str]
    warnings              : List[str]
    split_stats           : Dict
    class_distribution    : Dict
    validated_at          : datetime
    validation_report_path: Optional[Path] = None


@dataclass(frozen=True)
class DataTransformationArtifact:
    validation_artifact   : DataValidationArtifact
    train_images_dir      : Path
    train_csv_path        : Path
    val_images_dir        : Optional[Path]
    val_csv_path          : Optional[Path]
    test_images_dir       : Optional[Path]
    test_csv_path         : Optional[Path]
    class_weights_path    : Path
    class_weights         : List[float]
    transform_config_path : Path
    input_size            : int
    batch_size            : int
    num_workers           : int
    sampler               : str
    pin_memory            : bool
    transformed_at        : datetime
    artifact_path         : Optional[Path] = None


@dataclass(frozen=True)
class ModelTrainerArtifact:
    # ── lineage ───────────────────────────────────────────────────────────────
    transformation_artifact : DataTransformationArtifact

    # ── model checkpoints ─────────────────────────────────────────────────────
    best_model_path         : Path       # highest val_acc checkpoint
    final_model_path        : Path       # last epoch weights
    checkpoints_dir         : Path       # all top-k checkpoints

    # ── architecture ──────────────────────────────────────────────────────────
    architecture            : str        # efficientnet_b3
    num_classes             : int        # 9
    pretrained              : bool       # True (ImageNet init)
    total_params            : int        # total model parameters
    trainable_params        : int        # trainable parameters

    # ── training config ───────────────────────────────────────────────────────
    epochs_trained          : int        # actual epochs run (may < config.epochs due to early stop)
    best_epoch              : int        # epoch with best val_acc
    best_val_acc            : float      # best validation accuracy
    best_val_loss           : float      # val loss at best epoch
    final_train_acc         : float      # train accuracy at last epoch
    final_train_loss        : float      # train loss at last epoch

    # ── per-class metrics ─────────────────────────────────────────────────────
    per_class_val_acc       : Dict[str, float]   # label → accuracy at best epoch

    # ── hyperparameters ───────────────────────────────────────────────────────
    learning_rate           : float
    weight_decay            : float
    lr_scheduler            : str
    batch_size              : int
    label_smoothing         : float
    dropout_rate            : float
    sampler                 : str

    # ── hardware ──────────────────────────────────────────────────────────────
    device                  : str        # cuda:0 | cpu
    cuda_version            : str        # e.g. 12.4 | N/A

    # ── timing ────────────────────────────────────────────────────────────────
    training_history_path   : Path       # JSON — full per-epoch metrics
    trained_at              : datetime
    total_training_time_s   : float

    # ── persistence ───────────────────────────────────────────────────────────
    artifact_path           : Optional[Path] = None

@dataclass(frozen=True)
class ModelEvaluationArtifact:
    """
    Produced by : model_evaluation.py
    Consumed by : model_export.py

    evaluation_report layout written to JSON:
      {
        "version_id"         : "v_20260307_...",
        "architecture"       : "efficientnet_b3",
        "best_model_path"    : "...",
        "overall"            : {
            "accuracy"       : 0.912,
            "top2_accuracy"  : 0.971,
            "macro_f1"       : 0.887,
            "weighted_f1"    : 0.911,
        },
        "per_class"          : {
            "0": { "precision": 0.91, "recall": 0.89, "f1": 0.90, "support": 1126 },
            ...
        },
        "confusion_matrix"   : [[...], ...],   # 9×9 list of lists
        "tta_enabled"        : false,
        "evaluated_at"       : "2026-03-08T...",
      }
    """
    # ── lineage ───────────────────────────────────────────────────────────────
    trainer_artifact        : ModelTrainerArtifact

    # ── overall metrics ───────────────────────────────────────────────────────
    accuracy                : float
    top2_accuracy           : float
    macro_f1                : float
    weighted_f1             : float

    # ── per-class metrics ─────────────────────────────────────────────────────
    per_class_metrics       : Dict[str, Dict[str, float]]
    # { "0": {"precision": 0.91, "recall": 0.89, "f1": 0.90, "support": 1126} }

    # ── confusion matrix ──────────────────────────────────────────────────────
    confusion_matrix        : List[List[int]]   # 9×9

    # ── eval config ───────────────────────────────────────────────────────────
    tta_enabled             : bool
    eval_batch_size         : int

    # ── timing + persistence ──────────────────────────────────────────────────
    evaluated_at            : datetime
    evaluation_report_path  : Path
    artifact_path           : Optional[Path] = None