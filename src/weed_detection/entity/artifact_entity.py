from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import Optional, List,Dict


@dataclass(frozen=True)
class KafkaArtifact:
    """
    Produced by : kafka_consumer.py
    Consumed by : data_ingestion.py

    Written to:
      kafka_data/latest_artifact.json          ← always overwritten (latest pointer)
      kafka_data/v_<ts>/artifact.json          ← permanent version copy

    Captures full lineage: S3 origin → Kafka metadata → local zip path.
    """
    # ── local storage ─────────────────────────────────────────────────────────
    kafka_data_dir   : Path     # artifacts/data_ingestion/kafka_data/
    version_dir      : Path     # kafka_data/v_YYYYMMDD_HHMMSS/
    zip_file_path    : Path     # version_dir/<hash>_drone_YYYYMMDD_HHMMSS.zip

    # ── S3 origin ─────────────────────────────────────────────────────────────
    s3_bucket        : str      # aws-drone-combined
    s3_key           : str      # drone_20260307_201234.zip
    source_url       : str      # DATA_SOURCE from .env

    # ── file integrity ────────────────────────────────────────────────────────
    file_hash        : str      # SHA-256[:16] — used as version fingerprint
    file_size_bytes  : int
    original_filename: str      # drone_20260307_201234.zip

    # ── kafka provenance ──────────────────────────────────────────────────────
    kafka_topic      : str
    kafka_partition  : int
    kafka_offset     : int

    # ── timing ────────────────────────────────────────────────────────────────
    received_at      : datetime

    # ── self reference ────────────────────────────────────────────────────────
    artifact_path    : Optional[Path] = None   # set after JSON is written


@dataclass(frozen=True)
class DataIngestionArtifact:
    """
    Produced by : data_ingestion.py
    Consumed by : data_validation.py

    deep-weed has train / val / test splits defined by subset CSVs.
    All three splits are captured here.

    Normalized layout per version:
        normalized/v_<timestamp>_<hash>/
        ├── images/
        │   ├── train/   images referenced in train_subset0..4.csv
        │   ├── val/     images referenced in val_subset0..4.csv
        │   └── test/    images referenced in test_subset0..4.csv
        └── labels/
            ├── train.csv   merged from train_subset0..4.csv
            ├── val.csv     merged from val_subset0..4.csv
            ├── test.csv    merged from test_subset0..4.csv
            └── labels.csv  master copy (all 17,509 images)

    Version is isolated — old versions are never touched when new data arrives.
    """
    # ── lineage ───────────────────────────────────────────────────────────────
    kafka_artifact   : KafkaArtifact   # full lineage back to the original zip

    # ── directories ───────────────────────────────────────────────────────────
    unzip_dir        : Path            # artifacts/data_ingestion/unzipped/
    normalized_dir   : Path            # artifacts/data_ingestion/normalized/v_.../

    # ── train split — always present ──────────────────────────────────────────
    train_images_dir : Path            # normalized_dir/images/train/
    train_labels_dir : Path            # normalized_dir/labels/  (train.csv lives here)

    # ── val split ─────────────────────────────────────────────────────────────
    val_images_dir   : Optional[Path] = None   # normalized_dir/images/val/
    val_labels_dir   : Optional[Path] = None   # normalized_dir/labels/

    # ── test split ────────────────────────────────────────────────────────────
    test_images_dir  : Optional[Path] = None   # normalized_dir/images/test/
    test_labels_dir  : Optional[Path] = None   # normalized_dir/labels/

    # ── stats ─────────────────────────────────────────────────────────────────
    source_type      : str            = "unknown"   # synthetic | real | flat
    total_images     : int            = 0
    total_labels     : int            = 0
    splits           : List[str]      = field(default_factory=list)
    warnings         : List[str]      = field(default_factory=list)

    # ── artifact persistence ──────────────────────────────────────────────────
    artifact_path    : Optional[Path] = None   # data_ingestion_artifact.json




@dataclass(frozen=True)
class DataValidationArtifact:
    # ── lineage ───────────────────────────────────────────────────────────────
    ingestion_artifact   : DataIngestionArtifact   # full lineage back to zip

    # ── overall verdict ───────────────────────────────────────────────────────
    is_valid             : bool           # False = pipeline must stop

    # ── findings ──────────────────────────────────────────────────────────────
    failed_checks        : List[str]      # hard failures  (non-empty → is_valid=False)
    warnings             : List[str]      # soft issues    (logged, not blocking)

    # ── per-split stats ───────────────────────────────────────────────────────
    split_stats          : Dict           # images_on_disk / csv_rows / corrupt per split
    class_distribution   : Dict           # label counts per class per split

    # ── timing ────────────────────────────────────────────────────────────────
    validated_at         : datetime

    # ── artifact persistence ──────────────────────────────────────────────────
    validation_report_path: Optional[Path] = None   # validation_report.json

  

@dataclass(frozen=True)
class DataTransformationArtifact:
    # ── lineage ───────────────────────────────────────────────────────────────
    validation_artifact      : DataValidationArtifact

    # ── split paths (mirrors DataIngestionArtifact for trainer convenience) ──
    train_images_dir         : Path
    train_csv_path           : Path
    val_images_dir           : Optional[Path]
    val_csv_path             : Optional[Path]
    test_images_dir          : Optional[Path]
    test_csv_path            : Optional[Path]

    # ── class imbalance ───────────────────────────────────────────────────────
    class_weights_path       : Path          # JSON list, index = label int
    class_weights            : List[float]   # in-memory copy for trainer

    # ── transforms ────────────────────────────────────────────────────────────
    transform_config_path    : Path          # JSON — full transform spec
    input_size               : int           # 224

    # ── dataloader params ─────────────────────────────────────────────────────
    batch_size               : int
    num_workers              : int
    sampler                  : str           # weighted | none
    pin_memory               : bool

    # ── timing + persistence ──────────────────────────────────────────────────
    transformed_at           : datetime
    artifact_path            : Optional[Path] = None