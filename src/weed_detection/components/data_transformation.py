import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from weed_detection import logger
from weed_detection.config.configuration import ConfigurationManager
from weed_detection.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact,
)
from weed_detection.entity.config_entity import DataTransformationConfig
from weed_detection.utils.utility import load_json, save_json

# ── Deep-weed label map ───────────────────────────────────────────────────────
SPECIES_MAP: Dict[int, str] = {
    0: "Chinee apple",
    1: "Lantana",
    2: "Parkinsonia",
    3: "Parthenium",
    4: "Prickly acacia",
    5: "Rubber vine",
    6: "Siam weed",
    7: "Snake weed",
    8: "Negative",
}
NUM_CLASSES = len(SPECIES_MAP)   # 9

# ── ImageNet normalisation stats ──────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Dataset ───────────────────────────────────────────────────────────────────

class DeepWeedDataset(Dataset):
    """
    PyTorch Dataset for deep-weed classification.

    Reads a merged split CSV (Filename | Label | Species) produced by
    data_ingestion._normalize_real().

    Args:
        csv_path   : Path to split CSV  (e.g. labels/train.csv)
        images_dir : Directory containing the image files for this split
        transform  : torchvision transform pipeline
    """

    def __init__(
        self,
        csv_path  : Path,
        images_dir: Path,
        transform : Optional[transforms.Compose] = None,
    ):
        self.images_dir = images_dir
        self.transform  = transform
        self.samples: List[Tuple[str, int]] = []   # (filename, label)
        self._missing: List[str] = []

        with open(csv_path, "r", newline="") as f:
            for row in csv.DictReader(f):
                filename = row["Filename"].strip()
                label    = int(row["Label"].strip())
                img_path = images_dir / filename
                if img_path.exists():
                    self.samples.append((filename, label))
                else:
                    self._missing.append(filename)

        if self._missing:
            logger.warning(
                f"⚠️  DeepWeedDataset: {len(self._missing)} file(s) in CSV "
                f"not found in {images_dir} — skipped. "
                f"First few: {self._missing[:3]}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        filename, label = self.samples[idx]
        img_path = self.images_dir / filename
        image    = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    @property
    def labels(self) -> List[int]:
        """All labels in order — used to build WeightedRandomSampler."""
        return [label for _, label in self.samples]


# ── DataTransformation ────────────────────────────────────────────────────────

class DataTransformation:

    def __init__(self, config: DataTransformationConfig):
        self.config = config

    # ── public entry point ────────────────────────────────────────────────────

    def run(self) -> Optional[DataTransformationArtifact]:
        logger.info("=" * 60)
        logger.info("🚀 Data Transformation — version check")
        logger.info("=" * 60)

        # 1. load validation artifact — abort if data invalid
        validation_artifact = self._load_validation_artifact()

        if not validation_artifact.is_valid:
            raise RuntimeError(
                "❌ DataValidationArtifact.is_valid=False — "
                "cannot transform invalid data. Fix validation failures first."
            )

        # 2. version check
        version_id = validation_artifact.ingestion_artifact.normalized_dir.name
        if self._already_transformed(version_id):
            logger.info(f"⏭️  Version '{version_id}' already transformed — skipping")
            return None

        logger.info(f"🆕 New version : {version_id}")

        ingestion = validation_artifact.ingestion_artifact

        # 3. compute class weights from train CSV
        train_csv  = ingestion.train_labels_dir / "train.csv"
        class_weights = self._compute_class_weights(train_csv)
        self._save_class_weights(class_weights)

        # 4. build transforms
        train_tf, val_tf, test_tf = self._build_transforms()
        self._save_transform_config(train_tf, val_tf, test_tf)

        # 5. build datasets
        train_dataset = DeepWeedDataset(train_csv, ingestion.train_images_dir, train_tf)
        logger.info(f"   [train] {len(train_dataset)} samples")

        val_dataset   = None
        val_csv       = None
        if ingestion.val_images_dir and ingestion.val_labels_dir:
            val_csv     = ingestion.val_labels_dir / "val.csv"
            val_dataset = DeepWeedDataset(val_csv, ingestion.val_images_dir, val_tf)
            logger.info(f"   [val]   {len(val_dataset)} samples")

        test_dataset  = None
        test_csv      = None
        if ingestion.test_images_dir and ingestion.test_labels_dir:
            test_csv     = ingestion.test_labels_dir / "test.csv"
            test_dataset = DeepWeedDataset(test_csv, ingestion.test_images_dir, test_tf)
            logger.info(f"   [test]  {len(test_dataset)} samples")

        # 6. build dataloaders
        train_loader = self._build_train_loader(train_dataset, class_weights)
        val_loader   = self._build_eval_loader(val_dataset,  "val")  if val_dataset  else None
        test_loader  = self._build_eval_loader(test_dataset, "test") if test_dataset else None

        # 7. validate one batch per loader
        self._validate_batch(train_loader, "train")
        if val_loader:
            self._validate_batch(val_loader,  "val")
        if test_loader:
            self._validate_batch(test_loader, "test")

        # 8. build + save artifact
        artifact = self._build_artifact(
            validation_artifact = validation_artifact,
            train_csv           = train_csv,
            val_csv             = val_csv,
            test_csv            = test_csv,
            class_weights       = class_weights,
        )

        # 9. update state
        self._update_state(version_id, artifact)

        self._log_summary(artifact)
        return artifact

    # ── step 3 : class weights ────────────────────────────────────────────────

    def _compute_class_weights(self, train_csv: Path) -> List[float]:
        """
        Inverse-frequency weights for all NUM_CLASSES (0..8).
        weight[i] = total_samples / (num_classes * count[i])

        If a class is absent from train (shouldn't happen but defensive),
        it gets weight 0.0.
        """
        logger.info("─" * 50)
        logger.info("⚖️  Step 3 — Computing class weights")

        counts: Dict[int, int] = Counter()
        total = 0

        with open(train_csv, "r", newline="") as f:
            for row in csv.DictReader(f):
                label = int(row["Label"].strip())
                counts[label] += 1
                total += 1

        weights: List[float] = []
        for cls in range(NUM_CLASSES):
            count = counts.get(cls, 0)
            w     = total / (NUM_CLASSES * count) if count > 0 else 0.0
            weights.append(round(w, 6))
            logger.info(
                f"   Label {cls} ({SPECIES_MAP[cls]:<16}) : "
                f"{count:>6} samples  weight={w:.4f}"
            )

        logger.info(f"   Total train samples : {total}")
        return weights

    def _save_class_weights(self, weights: List[float]) -> None:
        save_json(
            path = self.config.class_weights_path,
            data = {
                "weights"    : weights,
                "num_classes": NUM_CLASSES,
                "species_map": {str(k): v for k, v in SPECIES_MAP.items()},
                "strategy"   : "inverse_frequency",
                "formula"    : "total / (num_classes * count_per_class)",
            }
        )
        logger.info(f"💾 Class weights saved : {self.config.class_weights_path}")

    # ── step 4 : transforms ───────────────────────────────────────────────────

    def _build_transforms(
        self,
    ) -> Tuple[transforms.Compose, transforms.Compose, transforms.Compose]:
        """
        Train : augmentation pipeline suitable for top-down drone imagery
        Val   : deterministic resize + crop + normalize only
        Test  : same as val
        """
        logger.info("─" * 50)
        logger.info("🔄 Step 4 — Building transforms")

        size = self.config.input_size

        train_transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=90),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            ),
            transforms.RandomResizedCrop(
                size  = size,
                scale = (0.8, 1.0),
                ratio = (0.9, 1.1),
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        eval_transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        logger.info(f"   Train : {len(train_transform.transforms)} transform steps")
        logger.info(f"   Val   : {len(eval_transform.transforms)} transform steps")
        logger.info(f"   Test  : {len(eval_transform.transforms)} transform steps (same as val)")

        return train_transform, eval_transform, eval_transform

    def _save_transform_config(
        self,
        train_tf: transforms.Compose,
        val_tf  : transforms.Compose,
        test_tf : transforms.Compose,
    ) -> None:
        def _names(tf: transforms.Compose) -> List[str]:
            return [type(t).__name__ for t in tf.transforms]

        save_json(
            path = self.config.transform_config_path,
            data = {
                "input_size"     : self.config.input_size,
                "normalize_mean" : IMAGENET_MEAN,
                "normalize_std"  : IMAGENET_STD,
                "train"          : _names(train_tf),
                "val"            : _names(val_tf),
                "test"           : _names(test_tf),
                "color_jitter"   : {
                    "brightness": 0.2,
                    "contrast"  : 0.2,
                    "saturation": 0.2,
                    "hue"       : 0.05,
                },
                "random_resized_crop": {
                    "scale": [0.8, 1.0],
                    "ratio": [0.9, 1.1],
                },
            }
        )
        logger.info(f"💾 Transform config saved : {self.config.transform_config_path}")

    # ── step 6 : dataloaders ──────────────────────────────────────────────────

    def _build_train_loader(
        self,
        dataset      : DeepWeedDataset,
        class_weights: List[float],
    ) -> DataLoader:
        """
        weighted  → WeightedRandomSampler — over-samples minority weed classes
        none      → standard shuffle=True
        """
        if self.config.sampler == "weighted":
            sample_weights = [class_weights[label] for label in dataset.labels]
            sampler = WeightedRandomSampler(
                weights     = sample_weights,
                num_samples = len(sample_weights),
                replacement = True,
            )
            loader = DataLoader(
                dataset,
                batch_size  = self.config.batch_size,
                sampler     = sampler,
                num_workers = self.config.num_workers,
                pin_memory  = self.config.pin_memory,
                drop_last   = True,
            )
            logger.info(
                f"   [train] DataLoader — WeightedRandomSampler  "
                f"batch={self.config.batch_size}  workers={self.config.num_workers}"
            )
        else:
            loader = DataLoader(
                dataset,
                batch_size  = self.config.batch_size,
                shuffle     = True,
                num_workers = self.config.num_workers,
                pin_memory  = self.config.pin_memory,
                drop_last   = True,
            )
            logger.info(
                f"   [train] DataLoader — shuffle=True  "
                f"batch={self.config.batch_size}  workers={self.config.num_workers}"
            )
        return loader

    def _build_eval_loader(
        self,
        dataset: DeepWeedDataset,
        split  : str,
    ) -> DataLoader:
        loader = DataLoader(
            dataset,
            batch_size  = self.config.batch_size,
            shuffle     = False,
            num_workers = self.config.num_workers,
            pin_memory  = self.config.pin_memory,
        )
        logger.info(
            f"   [{split}]  DataLoader — shuffle=False  "
            f"batch={self.config.batch_size}  workers={self.config.num_workers}"
        )
        return loader

    # ── step 7 : batch validation ─────────────────────────────────────────────

    def _validate_batch(self, loader: DataLoader, split: str) -> None:
        """
        Pull one batch and assert:
          - tensor shape : (B, 3, input_size, input_size)
          - dtype        : float32
          - label range  : all in [0, NUM_CLASSES-1]
        """
        logger.info("─" * 50)
        logger.info(f"🔬 Step 7 — Batch validation [{split}]")

        images, labels = next(iter(loader))

        expected_shape = (
            images.shape[0],
            3,
            self.config.input_size,
            self.config.input_size,
        )

        assert images.shape == torch.Size(expected_shape), (
            f"[{split}] Unexpected tensor shape: {images.shape} "
            f"(expected {expected_shape})"
        )
        assert images.dtype == torch.float32, (
            f"[{split}] Unexpected dtype: {images.dtype} (expected float32)"
        )
        assert labels.min().item() >= 0, (
            f"[{split}] Label below 0: {labels.min().item()}"
        )
        assert labels.max().item() <= NUM_CLASSES - 1, (
            f"[{split}] Label above {NUM_CLASSES - 1}: {labels.max().item()}"
        )

        logger.info(f"   ✅ [{split}] shape  : {tuple(images.shape)}")
        logger.info(f"   ✅ [{split}] dtype  : {images.dtype}")
        logger.info(f"   ✅ [{split}] labels : min={labels.min().item()}  max={labels.max().item()}")
        logger.info(
            f"   ✅ [{split}] pixels : "
            f"min={images.min().item():.3f}  "
            f"max={images.max().item():.3f}  "
            f"mean={images.mean().item():.3f}"
        )

    # ── step 8 : build artifact ───────────────────────────────────────────────

    def _build_artifact(
        self,
        validation_artifact : DataValidationArtifact,
        train_csv           : Path,
        val_csv             : Optional[Path],
        test_csv            : Optional[Path],
        class_weights       : List[float],
    ) -> DataTransformationArtifact:
        ingestion        = validation_artifact.ingestion_artifact
        transformed_at   = datetime.now()

        artifact = DataTransformationArtifact(
            validation_artifact   = validation_artifact,
            train_images_dir      = ingestion.train_images_dir,
            train_csv_path        = train_csv,
            val_images_dir        = ingestion.val_images_dir,
            val_csv_path          = val_csv,
            test_images_dir       = ingestion.test_images_dir,
            test_csv_path         = test_csv,
            class_weights_path    = self.config.class_weights_path,
            class_weights         = class_weights,
            transform_config_path = self.config.transform_config_path,
            input_size            = self.config.input_size,
            batch_size            = self.config.batch_size,
            num_workers           = self.config.num_workers,
            sampler               = self.config.sampler,
            pin_memory            = self.config.pin_memory,
            transformed_at        = transformed_at,
            artifact_path         = self.config.artifact_path,
        )

        save_json(
            path = self.config.artifact_path,
            data = {
                "version_id"           : ingestion.normalized_dir.name,
                "transformed_at"       : transformed_at.isoformat(),
                "train_images_dir"     : str(ingestion.train_images_dir),
                "train_csv_path"       : str(train_csv),
                "val_images_dir"       : str(ingestion.val_images_dir)  if ingestion.val_images_dir  else None,
                "val_csv_path"         : str(val_csv)                   if val_csv                   else None,
                "test_images_dir"      : str(ingestion.test_images_dir) if ingestion.test_images_dir else None,
                "test_csv_path"        : str(test_csv)                  if test_csv                  else None,
                "class_weights_path"   : str(self.config.class_weights_path),
                "class_weights"        : class_weights,
                "transform_config_path": str(self.config.transform_config_path),
                "input_size"           : self.config.input_size,
                "batch_size"           : self.config.batch_size,
                "num_workers"          : self.config.num_workers,
                "sampler"              : self.config.sampler,
                "pin_memory"           : self.config.pin_memory,
                "validation_report"    : str(validation_artifact.validation_report_path),
                "is_valid"             : validation_artifact.is_valid,
            }
        )
        logger.info(f"📋 Artifact saved : {self.config.artifact_path}")
        return artifact

    # ── version control ───────────────────────────────────────────────────────

    def _already_transformed(self, version_id: str) -> bool:
        state_path = self.config.transformation_state_path
        if not state_path.exists():
            return False
        try:
            state = load_json(state_path)
            return state.get("last_version_id") == version_id
        except Exception as e:
            logger.warning(f"⚠️  Could not read transformation state: {e} — will re-transform")
            return False

    def _update_state(
        self, version_id: str, artifact: DataTransformationArtifact
    ) -> None:
        save_json(
            path = self.config.transformation_state_path,
            data = {
                "last_version_id"   : version_id,
                "last_transformed_at": artifact.transformed_at.isoformat(),
                "last_artifact_path" : str(artifact.artifact_path),
                "input_size"         : artifact.input_size,
                "batch_size"         : artifact.batch_size,
                "sampler"            : artifact.sampler,
            }
        )
        logger.info(f"💾 Transformation state saved : {self.config.transformation_state_path}")

    # ── load validation artifact ──────────────────────────────────────────────

    def _load_validation_artifact(self) -> DataValidationArtifact:
        from weed_detection.components.data_validation import load_ingestion_artifact
        from weed_detection.entity.artifact_entity import DataValidationArtifact as DVA

        config_manager    = ConfigurationManager()
        dv_config         = config_manager.get_data_validation_config()
        report_path       = dv_config.validation_report_path

        if not report_path.exists():
            raise FileNotFoundError(
                f"No validation report at {report_path}\n"
                f"Run data_validation.py first."
            )

        report = load_json(report_path)

        ingestion_artifact = load_ingestion_artifact(
            Path(report["ingestion_artifact"])
        )

        artifact = DVA(
            ingestion_artifact     = ingestion_artifact,
            is_valid               = report["is_valid"],
            failed_checks          = report.get("failed_checks", []),
            warnings               = report.get("warnings", []),
            split_stats            = report.get("split_stats", {}),
            class_distribution     = report.get("class_distribution", {}),
            validated_at           = datetime.fromisoformat(report["validated_at"]),
            validation_report_path = report_path,
        )
        logger.info(f"✅ ValidationArtifact loaded — is_valid={artifact.is_valid}")
        return artifact

    # ── summary ───────────────────────────────────────────────────────────────

    def _log_summary(self, artifact: DataTransformationArtifact) -> None:
        logger.info("=" * 60)
        logger.info("📊 DATA TRANSFORMATION COMPLETE")
        logger.info(f"   Version        : {artifact.validation_artifact.ingestion_artifact.normalized_dir.name}")
        logger.info(f"   Input size     : {artifact.input_size}×{artifact.input_size}")
        logger.info(f"   Batch size     : {artifact.batch_size}")
        logger.info(f"   Num workers    : {artifact.num_workers}")
        logger.info(f"   Sampler        : {artifact.sampler}")
        logger.info(f"   Train CSV      : {artifact.train_csv_path}")
        logger.info(f"   Val CSV        : {artifact.val_csv_path}")
        logger.info(f"   Test CSV       : {artifact.test_csv_path}")
        logger.info(f"   Class weights  : {[round(w, 4) for w in artifact.class_weights]}")
        logger.info(f"   Artifact       : {artifact.artifact_path}")
        logger.info("=" * 60)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("🚀 Starting Data Transformation")
    logger.info("=" * 60)

    config_manager = ConfigurationManager()
    config         = config_manager.get_data_transformation_config()

    transformation = DataTransformation(config)
    artifact       = transformation.run()

    if artifact is None:
        logger.info("✅ Nothing to do — already transformed (same version)")
    else:
        logger.info(f"✅ Transformation complete")

    return artifact


if __name__ == "__main__":
    main()