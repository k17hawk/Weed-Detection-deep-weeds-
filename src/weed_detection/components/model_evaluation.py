import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    import timm
except ImportError:
    raise ImportError("timm is required: pip install timm")

try:
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        f1_score,
    )
except ImportError:
    raise ImportError("scikit-learn is required: pip install scikit-learn")

from weed_detection import logger
from weed_detection.config.configuration import ConfigurationManager
from weed_detection.entity.artifact_entity import (
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
)
from weed_detection.entity.config_entity import ModelEvaluationConfig
from weed_detection.utils.utility import load_json, save_json

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
NUM_CLASSES   = 9
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Dataset ───────────────────────────────────────────────────────────────────

class DeepWeedDataset(Dataset):
    def __init__(
        self,
        csv_path  : Path,
        images_dir: Path,
        transform : Optional[transforms.Compose] = None,
    ):
        self.images_dir = images_dir
        self.transform  = transform
        self.samples: List[Tuple[str, int]] = []

        with open(csv_path, "r", newline="") as f:
            for row in csv.DictReader(f):
                filename = row["Filename"].strip()
                label    = int(row["Label"].strip())
                if (images_dir / filename).exists():
                    self.samples.append((filename, label))

        logger.info(f"   Test dataset : {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        filename, label = self.samples[idx]
        image = Image.open(self.images_dir / filename).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ── ModelEvaluation ───────────────────────────────────────────────────────────

class ModelEvaluation:

    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def run(self) -> Optional[ModelEvaluationArtifact]:
        logger.info("=" * 60)
        logger.info("🚀 Model Evaluation — version check")
        logger.info("=" * 60)

        # 1. load trainer artifact
        trainer_artifact = self._load_trainer_artifact()

        # 2. version check
        version_id = (
            trainer_artifact
            .transformation_artifact
            .validation_artifact
            .ingestion_artifact
            .normalized_dir.name
        )
        if self._already_evaluated(version_id):
            logger.info(f"⏭️  Version '{version_id}' already evaluated — skipping")
            return None

        logger.info(f"🆕 Evaluating version : {version_id}")
        logger.info(f"   Device            : {self.device}")
        logger.info(f"   Best model        : {trainer_artifact.best_model_path}")

        # 3. load model
        model = self._load_model(trainer_artifact)

        # 4. build test loader
        test_loader = self._build_test_loader(trainer_artifact)
        if test_loader is None:
            raise RuntimeError(
                "No test split found in TransformationArtifact. "
                "Cannot evaluate — test_csv_path is None."
            )

        # 5. run inference
        all_preds, all_labels, all_probs = self._run_inference(
            model, test_loader
        )

        # 6. compute metrics
        metrics = self._compute_metrics(all_preds, all_labels, all_probs)

        # 7. build + save artifact
        artifact = self._build_artifact(trainer_artifact, metrics, version_id)

        # 8. update state
        self._update_state(version_id, artifact)

        self._log_summary(artifact)
        return artifact

    # ── step 3 : load model ───────────────────────────────────────────────────

    def _load_model(self, trainer_artifact: ModelTrainerArtifact) -> nn.Module:
        logger.info("─" * 50)
        logger.info("🏗️  Step 3 — Loading best model")

        model = timm.create_model(
            trainer_artifact.architecture,
            pretrained  = False,
            num_classes = 0,
            global_pool = "avg",
        )
        num_features     = model.num_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=trainer_artifact.dropout_rate),
            nn.Linear(num_features, trainer_artifact.num_classes),
        )

        state_dict = torch.load(
            trainer_artifact.best_model_path,
            map_location=self.device,
        )
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()

        logger.info(f"   ✅ Loaded : {trainer_artifact.best_model_path.name}")
        logger.info(f"   Architecture : {trainer_artifact.architecture}")
        return model

    # ── step 4 : test loader ──────────────────────────────────────────────────

    def _build_test_loader(
        self, trainer_artifact: ModelTrainerArtifact
    ) -> Optional[DataLoader]:
        logger.info("─" * 50)
        logger.info("📦 Step 4 — Building test DataLoader")

        ta = trainer_artifact.transformation_artifact

        if not ta.test_csv_path or not ta.test_images_dir:
            logger.warning("⚠️  No test split in TransformationArtifact")
            return None

        size = self.config.input_size
        eval_tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        test_dataset = DeepWeedDataset(
            ta.test_csv_path, ta.test_images_dir, eval_tf
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size  = self.config.eval_batch_size,
            shuffle     = False,
            num_workers = self.config.num_workers,
            pin_memory  = self.config.pin_memory,
        )
        logger.info(
            f"   Test batches : {len(test_loader)}  "
            f"(batch={self.config.eval_batch_size})"
        )
        return test_loader

    # ── step 5 : inference ────────────────────────────────────────────────────

    def _run_inference(
        self,
        model : nn.Module,
        loader: DataLoader,
    ) -> Tuple[List[int], List[int], List[List[float]]]:
        """
        Returns all_preds, all_labels, all_probs (softmax).
        If TTA enabled: averages probs from original + hflip + vflip.
        """
        logger.info("─" * 50)
        logger.info(
            f"🔬 Step 5 — Running inference  "
            f"(TTA={'on' if self.config.eval_tta else 'off'})"
        )

        all_preds : List[int]         = []
        all_labels: List[int]         = []
        all_probs : List[List[float]] = []

        softmax = nn.Softmax(dim=1)

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(loader):
                images = images.to(self.device, non_blocking=True)

                if self.config.eval_tta:
                    # ensemble: original + hflip + vflip
                    probs = softmax(model(images))
                    probs += softmax(model(torch.flip(images, dims=[3])))  # hflip
                    probs += softmax(model(torch.flip(images, dims=[2])))  # vflip
                    probs /= 3.0
                else:
                    probs = softmax(model(images))

                preds = probs.argmax(dim=1)

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.tolist())
                all_probs.extend(probs.cpu().tolist())

                if (batch_idx + 1) % 50 == 0:
                    logger.info(
                        f"   Batch {batch_idx + 1}/{len(loader)} processed"
                    )

        logger.info(f"   ✅ Inference complete — {len(all_preds)} samples")
        return all_preds, all_labels, all_probs

    # ── step 6 : metrics ──────────────────────────────────────────────────────

    def _compute_metrics(
        self,
        all_preds : List[int],
        all_labels: List[int],
        all_probs : List[List[float]],
    ) -> Dict:
        logger.info("─" * 50)
        logger.info("📊 Step 6 — Computing metrics")

        # overall accuracy
        correct  = sum(p == l for p, l in zip(all_preds, all_labels))
        accuracy = correct / len(all_labels)

        # top-2 accuracy
        top2_correct = 0
        for probs, label in zip(all_probs, all_labels):
            top2 = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:2]
            if label in top2:
                top2_correct += 1
        top2_accuracy = top2_correct / len(all_labels)

        # macro + weighted F1
        macro_f1    = f1_score(all_labels, all_preds, average="macro",    zero_division=0)
        weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

        # per-class metrics via sklearn
        report = classification_report(
            all_labels, all_preds,
            target_names = [SPECIES_MAP[i] for i in range(NUM_CLASSES)],
            output_dict  = True,
            zero_division= 0,
        )

        per_class_metrics: Dict[str, Dict[str, float]] = {}
        for cls_idx in range(NUM_CLASSES):
            species_name = SPECIES_MAP[cls_idx]
            cls_report   = report.get(species_name, {})
            per_class_metrics[str(cls_idx)] = {
                "precision": round(cls_report.get("precision", 0.0), 6),
                "recall"   : round(cls_report.get("recall",    0.0), 6),
                "f1"       : round(cls_report.get("f1-score",  0.0), 6),
                "support"  : int(cls_report.get("support",     0)),
            }

        # confusion matrix
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))

        logger.info(f"   Accuracy     : {accuracy:.4f}")
        logger.info(f"   Top-2 acc    : {top2_accuracy:.4f}")
        logger.info(f"   Macro F1     : {macro_f1:.4f}")
        logger.info(f"   Weighted F1  : {weighted_f1:.4f}")
        logger.info("   Per-class:")
        for cls_str, m in per_class_metrics.items():
            logger.info(
                f"      Label {cls_str} ({SPECIES_MAP[int(cls_str)]:<16}) : "
                f"P={m['precision']:.3f}  R={m['recall']:.3f}  "
                f"F1={m['f1']:.3f}  support={m['support']}"
            )

        return {
            "accuracy"         : round(accuracy,      6),
            "top2_accuracy"    : round(top2_accuracy,  6),
            "macro_f1"         : round(macro_f1,       6),
            "weighted_f1"      : round(weighted_f1,    6),
            "per_class_metrics": per_class_metrics,
            "confusion_matrix" : cm.tolist(),
        }

    # ── step 7 : build artifact ───────────────────────────────────────────────

    def _build_artifact(
        self,
        trainer_artifact: ModelTrainerArtifact,
        metrics         : Dict,
        version_id      : str,
    ) -> ModelEvaluationArtifact:
        evaluated_at = datetime.now()

        artifact = ModelEvaluationArtifact(
            trainer_artifact       = trainer_artifact,
            accuracy               = metrics["accuracy"],
            top2_accuracy          = metrics["top2_accuracy"],
            macro_f1               = metrics["macro_f1"],
            weighted_f1            = metrics["weighted_f1"],
            per_class_metrics      = metrics["per_class_metrics"],
            confusion_matrix       = metrics["confusion_matrix"],
            tta_enabled            = self.config.eval_tta,
            eval_batch_size        = self.config.eval_batch_size,
            evaluated_at           = evaluated_at,
            evaluation_report_path = self.config.evaluation_report_path,
            artifact_path          = self.config.artifact_path,
        )

        report = {
            "version_id"        : version_id,
            "architecture"      : trainer_artifact.architecture,
            "best_model_path"   : str(trainer_artifact.best_model_path),
            "best_epoch"        : trainer_artifact.best_epoch,
            "best_val_acc"      : trainer_artifact.best_val_acc,
            "overall"           : {
                "accuracy"      : metrics["accuracy"],
                "top2_accuracy" : metrics["top2_accuracy"],
                "macro_f1"      : metrics["macro_f1"],
                "weighted_f1"   : metrics["weighted_f1"],
            },
            "per_class"         : metrics["per_class_metrics"],
            "confusion_matrix"  : metrics["confusion_matrix"],
            "tta_enabled"       : self.config.eval_tta,
            "eval_batch_size"   : self.config.eval_batch_size,
            "evaluated_at"      : evaluated_at.isoformat(),
            "trainer_artifact"  : str(trainer_artifact.artifact_path),
        }

        save_json(path=self.config.evaluation_report_path, data=report)
        save_json(
            path = self.config.artifact_path,
            data = {**report, "evaluation_report_path": str(self.config.evaluation_report_path)},
        )
        logger.info(f"📋 Evaluation report saved : {self.config.evaluation_report_path}")
        return artifact

    # ── version control ───────────────────────────────────────────────────────

    def _already_evaluated(self, version_id: str) -> bool:
        state_path = self.config.evaluation_state_path
        if not state_path.exists():
            return False
        try:
            state = load_json(state_path)
            return state.get("last_version_id") == version_id
        except Exception as e:
            logger.warning(f"⚠️  Could not read evaluation state: {e} — will re-evaluate")
            return False

    def _update_state(
        self, version_id: str, artifact: ModelEvaluationArtifact
    ) -> None:
        save_json(
            path = self.config.evaluation_state_path,
            data = {
                "last_version_id"    : version_id,
                "last_evaluated_at"  : artifact.evaluated_at.isoformat(),
                "last_accuracy"      : artifact.accuracy,
                "last_macro_f1"      : artifact.macro_f1,
                "last_artifact_path" : str(artifact.artifact_path),
            }
        )
        logger.info(f"💾 Evaluation state saved : {self.config.evaluation_state_path}")

    # ── load trainer artifact ─────────────────────────────────────────────────

    def _load_trainer_artifact(self) -> ModelTrainerArtifact:
        from weed_detection.components.data_validation import load_ingestion_artifact
        from weed_detection.entity.artifact_entity import (
            DataTransformationArtifact as DTA,
            DataValidationArtifact     as DVA,
            ModelTrainerArtifact       as MTA,
        )

        config_manager = ConfigurationManager()

        # trainer artifact JSON
        mt_config     = config_manager.get_model_trainer_config()
        artifact_path = mt_config.artifact_path
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"No ModelTrainerArtifact at {artifact_path}\n"
                f"Run model_trainer.py first."
            )
        d = load_json(artifact_path)

        # re-hydrate chain
        dv_config = config_manager.get_data_validation_config()
        dt_config = config_manager.get_data_transformation_config()

        report    = load_json(dv_config.validation_report_path)
        ingestion = load_ingestion_artifact(Path(report["ingestion_artifact"]))

        from datetime import datetime as _dt

        validation_artifact = DVA(
            ingestion_artifact     = ingestion,
            is_valid               = report["is_valid"],
            failed_checks          = report.get("failed_checks", []),
            warnings               = report.get("warnings", []),
            split_stats            = report.get("split_stats", {}),
            class_distribution     = report.get("class_distribution", {}),
            validated_at           = _dt.fromisoformat(report["validated_at"]),
            validation_report_path = dv_config.validation_report_path,
        )

        dt_d = load_json(dt_config.artifact_path)
        transformation_artifact = DTA(
            validation_artifact   = validation_artifact,
            train_images_dir      = Path(dt_d["train_images_dir"]),
            train_csv_path        = Path(dt_d["train_csv_path"]),
            val_images_dir        = Path(dt_d["val_images_dir"])  if dt_d.get("val_images_dir")  else None,
            val_csv_path          = Path(dt_d["val_csv_path"])    if dt_d.get("val_csv_path")    else None,
            test_images_dir       = Path(dt_d["test_images_dir"]) if dt_d.get("test_images_dir") else None,
            test_csv_path         = Path(dt_d["test_csv_path"])   if dt_d.get("test_csv_path")   else None,
            class_weights_path    = Path(dt_d["class_weights_path"]),
            class_weights         = dt_d["class_weights"],
            transform_config_path = Path(dt_d["transform_config_path"]),
            input_size            = dt_d["input_size"],
            batch_size            = dt_d["batch_size"],
            num_workers           = dt_d["num_workers"],
            sampler               = dt_d["sampler"],
            pin_memory            = dt_d["pin_memory"],
            transformed_at        = _dt.fromisoformat(dt_d["transformed_at"]),
            artifact_path         = dt_config.artifact_path,
        )

        artifact = MTA(
            transformation_artifact = transformation_artifact,
            best_model_path         = Path(d["best_model_path"]),
            final_model_path        = Path(d["final_model_path"]),
            checkpoints_dir         = Path(d["checkpoints_dir"]),
            architecture            = d["architecture"],
            num_classes             = d["num_classes"],
            pretrained              = d["pretrained"],
            total_params            = d["total_params"],
            trainable_params        = d["trainable_params"],
            epochs_trained          = d["epochs_trained"],
            best_epoch              = d["best_epoch"],
            best_val_acc            = d["best_val_acc"],
            best_val_loss           = d["best_val_loss"],
            final_train_acc         = d["final_train_acc"],
            final_train_loss        = d["final_train_loss"],
            per_class_val_acc       = d["per_class_val_acc"],
            learning_rate           = d["learning_rate"],
            weight_decay            = d["weight_decay"],
            lr_scheduler            = d["lr_scheduler"],
            batch_size              = d["batch_size"],
            label_smoothing         = d["label_smoothing"],
            dropout_rate            = d["dropout_rate"],
            sampler                 = d["sampler"],
            device                  = d["device"],
            cuda_version            = d["cuda_version"],
            training_history_path   = Path(d["training_history_path"]),
            trained_at              = _dt.fromisoformat(d["trained_at"]),
            total_training_time_s   = d["total_training_time_s"],
            artifact_path           = artifact_path,
        )
        logger.info(f"✅ ModelTrainerArtifact loaded")
        logger.info(f"   Best val acc : {artifact.best_val_acc:.4f}  (epoch {artifact.best_epoch})")
        return artifact

    # ── summary ───────────────────────────────────────────────────────────────

    def _log_summary(self, artifact: ModelEvaluationArtifact) -> None:
        logger.info("=" * 60)
        logger.info("📊 MODEL EVALUATION COMPLETE")
        logger.info(f"   Test accuracy  : {artifact.accuracy:.4f}")
        logger.info(f"   Top-2 accuracy : {artifact.top2_accuracy:.4f}")
        logger.info(f"   Macro F1       : {artifact.macro_f1:.4f}")
        logger.info(f"   Weighted F1    : {artifact.weighted_f1:.4f}")
        logger.info(f"   TTA            : {artifact.tta_enabled}")
        logger.info("─" * 50)
        logger.info("   Per-class (test):")
        for cls_str, m in sorted(
            artifact.per_class_metrics.items(), key=lambda x: int(x[0])
        ):
            species = SPECIES_MAP.get(int(cls_str), f"Class {cls_str}")
            logger.info(
                f"      Label {cls_str} ({species:<16}) : "
                f"P={m['precision']:.3f}  R={m['recall']:.3f}  "
                f"F1={m['f1']:.3f}  n={m['support']}"
            )
        logger.info("─" * 50)
        logger.info(f"   Report   : {artifact.evaluation_report_path}")
        logger.info(f"   Artifact : {artifact.artifact_path}")
        logger.info("=" * 60)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("🚀 Starting Model Evaluation")
    logger.info("=" * 60)

    config_manager = ConfigurationManager()
    config         = config_manager.get_model_evaluation_config()

    evaluator = ModelEvaluation(config)
    artifact  = evaluator.run()

    if artifact is None:
        logger.info("✅ Nothing to do — already evaluated (same version)")
    else:
        logger.info(f"✅ Evaluation complete — test accuracy: {artifact.accuracy:.4f}")
        if artifact.accuracy < 0.80:
            logger.warning("⚠️  Accuracy below 0.80 — consider retraining")

    return artifact


if __name__ == "__main__":
    main()