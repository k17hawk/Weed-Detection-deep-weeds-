import csv
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    import timm
except ImportError:
    raise ImportError("pip install timm")

try:
    import mlflow
    import mlflow.pytorch
    from mlflow.tracking import MlflowClient
except ImportError:
    raise ImportError("pip install mlflow")

try:
    import wandb
except ImportError:
    raise ImportError("pip install wandb")

from weed_detection import logger
from weed_detection.config.configuration import ConfigurationManager
from weed_detection.constants.constant import (
    CLASS_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MLFLOW_REGISTERED_MODEL_NAME,
    NUM_CLASSES,
    SPECIES_MAP,
)
from weed_detection.entity.artifact_entity import (
    ModelEvaluationArtifact,
    ModelRegistryEntry,
    ModelTrainerArtifact,
)
from weed_detection.entity.config_entity import (
    ModelEvaluationConfig,
    ModelRegistryConfig,
)
from weed_detection.utils.utility import load_json, save_json


# ── Dataset (test split) ──────────────────────────────────────────────────────
class DeepWeedDataset(Dataset):
    def __init__(
        self,
        csv_path  : Path,
        images_dir: Path,
        transform  = None,
    ):
        self.images_dir = images_dir
        self.transform  = transform
        self.samples: List[Tuple[str, int]] = []
        with open(csv_path, "r", newline="") as f:
            for row in csv.DictReader(f):
                fname = row["Filename"].strip()
                label = int(row["Label"].strip())
                if (images_dir / fname).exists():
                    self.samples.append((fname, label))
        logger.info(f"   Test dataset : {len(self.samples)} samples")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        img = Image.open(self.images_dir / fname).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ── ModelEvaluation ───────────────────────────────────────────────────────────
class ModelEvaluation:

    def __init__(
        self,
        config          : ModelEvaluationConfig,
        registry_config : ModelRegistryConfig,
    ):
        self.config          = config
        self.registry_config = registry_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self) -> Optional[ModelEvaluationArtifact]:
        logger.info("=" * 70)
        logger.info("🚀 Model Evaluation + Selection")
        logger.info("=" * 70)

        # 1. load trainer artifact
        trainer_artifact = self._load_trainer_artifact()

        # 2. version check
        run_id = (
            trainer_artifact
            .transformation_artifact
            .validation_artifact
            .ingestion_artifact
            .normalized_dir.name
        )
        if self._already_evaluated(run_id):
            logger.info(f"⏭️  Run '{run_id}' already evaluated — skipping")
            return None

        logger.info(f"🆕 Evaluating run : {run_id}")
        logger.info(f"   Device         : {self.device}")

        # 3 + 4. model + loader
        model       = self._load_model(trainer_artifact)
        test_loader = self._build_test_loader(trainer_artifact)
        if test_loader is None:
            raise RuntimeError("No test split — cannot evaluate.")

        # 5. inference
        all_preds, all_labels, all_probs = self._run_inference(model, test_loader)

        # 6. metrics
        metrics = self._compute_metrics(all_preds, all_labels, all_probs)

        # 7. register run
        run_number     = self._next_run_number()
        registry_entry = self._register_run(run_id, run_number, trainer_artifact, metrics)

        # 8. load previous champion metric
        prev_champion_metric = self._get_champion_metric()
        logger.info(
            f"   Current champion {self.config.promotion_metric} : "
            f"{prev_champion_metric if prev_champion_metric is not None else 'none'}"
        )

        # 9. promote if better
        current_metric  = metrics[self.config.promotion_metric]
        is_new_champion = self._evaluate_promotion(
            run_id, run_number, current_metric,
            prev_champion_metric, trainer_artifact, metrics,
        )
        champion_path = self.registry_config.champion_model_path

        # 10. W&B evaluation logging
        self._log_wandb_eval(
            trainer_artifact, metrics, run_id, run_number,
            all_labels, all_preds, is_new_champion,
        )

        # 11. MLflow evaluation logging + model registration
        mlflow_model_version, mlflow_stage = self._log_mlflow_eval(
            trainer_artifact, metrics, run_id, run_number,
            all_labels, all_preds, is_new_champion,
            current_metric, prev_champion_metric,
        )

        # update registry entry with MLflow details
        registry_entry = ModelRegistryEntry(
            **{
                **registry_entry.__dict__,
                "mlflow_model_version": mlflow_model_version,
                "mlflow_stage"        : mlflow_stage,
            }
        )

        # 12. build + save artifact
        artifact = self._build_artifact(
            trainer_artifact     = trainer_artifact,
            metrics              = metrics,
            registry_entry       = registry_entry,
            is_new_champion      = is_new_champion,
            champion_path        = champion_path,
            prev_champion_metric = prev_champion_metric,
            run_id               = run_id,
        )
        self._append_history(run_id, run_number, metrics, is_new_champion, mlflow_stage)
        self._update_state(run_id, artifact)
        self._log_summary(artifact)
        return artifact

    # ── step 3 : load model ───────────────────────────────────────────────────
    def _load_model(self, ta: ModelTrainerArtifact) -> nn.Module:
        logger.info("─" * 50)
        logger.info("🏗️  Loading best model from current run")
        model = timm.create_model(
            ta.architecture, pretrained=False, num_classes=0, global_pool="avg"
        )
        model.classifier = nn.Sequential(
            nn.Dropout(p=ta.dropout_rate),
            nn.Linear(model.num_features, ta.num_classes),
        )
        state = torch.load(ta.best_model_path, map_location=self.device)
        model.load_state_dict(state)
        model = model.to(self.device)
        model.eval()
        logger.info(f"   ✅ Loaded : {ta.best_model_path.name}")
        return model

    # ── step 4 : test loader ──────────────────────────────────────────────────
    def _build_test_loader(self, ta: ModelTrainerArtifact) -> Optional[DataLoader]:
        logger.info("─" * 50)
        logger.info("📦 Building test DataLoader")
        dta = ta.transformation_artifact
        if not dta.test_csv_path or not dta.test_images_dir:
            return None
        # same val_transform as notebook Cell 15
        size    = self.config.input_size   # 256
        eval_tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        ds = DeepWeedDataset(dta.test_csv_path, dta.test_images_dir, eval_tf)
        return DataLoader(
            ds,
            batch_size  = self.config.eval_batch_size,
            shuffle     = False,
            num_workers = self.config.num_workers,
            pin_memory  = self.config.pin_memory,
        )

    # ── step 5 : inference ────────────────────────────────────────────────────
    def _run_inference(
        self, model: nn.Module, loader: DataLoader
    ) -> Tuple[List[int], List[int], List[List[float]]]:
        logger.info("─" * 50)
        logger.info(f"🔬 Inference  (TTA={'on' if self.config.eval_tta else 'off'})")
        all_preds, all_labels, all_probs = [], [], []
        softmax = nn.Softmax(dim=1)
        with torch.no_grad():
            for i, (images, labels) in enumerate(loader):
                images = images.to(self.device, non_blocking=True)
                if self.config.eval_tta:
                    probs  = softmax(model(images))
                    probs += softmax(model(torch.flip(images, dims=[3])))
                    probs += softmax(model(torch.flip(images, dims=[2])))
                    probs /= 3.0
                else:
                    probs = softmax(model(images))
                preds = probs.argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.tolist())
                all_probs.extend(probs.cpu().tolist())
                if (i + 1) % 50 == 0:
                    logger.info(f"   Batch {i+1}/{len(loader)}")
        logger.info(f"   ✅ {len(all_preds)} samples")
        return all_preds, all_labels, all_probs

    # ── step 6 : metrics ──────────────────────────────────────────────────────
    def _compute_metrics(
        self,
        all_preds : List[int],
        all_labels: List[int],
        all_probs : List[List[float]],
    ) -> Dict:
        logger.info("─" * 50)
        logger.info("📊 Computing test metrics")

        accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
        top2_correct = sum(
            1 for probs, label in zip(all_probs, all_labels)
            if label in sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:2]
        )
        top2_accuracy = top2_correct / len(all_labels)
        macro_f1    = f1_score(all_labels, all_preds, average="macro",    zero_division=0)
        weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

        report = classification_report(
            all_labels, all_preds,
            target_names=CLASS_NAMES, output_dict=True, zero_division=0,
        )
        per_class: Dict[str, Dict[str, float]] = {}
        for cls_idx in range(NUM_CLASSES):
            r = report.get(CLASS_NAMES[cls_idx], {})
            per_class[str(cls_idx)] = {
                "precision": round(r.get("precision", 0.0), 6),
                "recall"   : round(r.get("recall",    0.0), 6),
                "f1"       : round(r.get("f1-score",  0.0), 6),
                "support"  : int(r.get("support",      0)),
            }

        cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))

        logger.info(f"   Accuracy     : {accuracy:.4f}")
        logger.info(f"   Top-2 acc    : {top2_accuracy:.4f}")
        logger.info(f"   Macro F1     : {macro_f1:.4f}")
        logger.info(f"   Weighted F1  : {weighted_f1:.4f}")
        for cls_str, m in per_class.items():
            logger.info(
                f"      Label {cls_str} ({SPECIES_MAP[int(cls_str)]:<16}) : "
                f"P={m['precision']:.3f}  R={m['recall']:.3f}  "
                f"F1={m['f1']:.3f}  n={m['support']}"
            )
        return {
            "accuracy"         : round(accuracy,     6),
            "top2_accuracy"    : round(top2_accuracy, 6),
            "macro_f1"         : round(macro_f1,      6),
            "weighted_f1"      : round(weighted_f1,   6),
            "per_class_metrics": per_class,
            "confusion_matrix" : cm.tolist(),
        }

    # ── step 7 : register run ─────────────────────────────────────────────────
    def _register_run(
        self, run_id: str, run_number: int,
        ta: ModelTrainerArtifact, metrics: Dict,
    ) -> ModelRegistryEntry:
        logger.info("─" * 50)
        logger.info(f"📝 Registering run #{run_number}  ({run_id})")
        run_dir   = self.registry_config.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        model_dst = run_dir / "model.pth"
        shutil.copy2(ta.best_model_path, model_dst)

        registered_at = datetime.now()
        entry = ModelRegistryEntry(
            run_id               = run_id,
            run_number           = run_number,
            model_path           = model_dst,
            architecture         = ta.architecture,
            epochs_trained       = ta.epochs_trained,
            best_val_acc         = ta.best_val_acc,
            accuracy             = metrics["accuracy"],
            top2_accuracy        = metrics["top2_accuracy"],
            macro_f1             = metrics["macro_f1"],
            weighted_f1          = metrics["weighted_f1"],
            per_class_metrics    = metrics["per_class_metrics"],
            mlflow_run_id        = ta.mlflow_run_id,
            mlflow_model_version = None,
            mlflow_stage         = "None",
            is_champion          = False,
            promoted_at          = None,
            registered_at        = registered_at,
        )
        save_json(path=run_dir / "run_metadata.json", data={
            "run_id"            : run_id,
            "run_number"        : run_number,
            "model_path"        : str(model_dst),
            "architecture"      : ta.architecture,
            "epochs_trained"    : ta.epochs_trained,
            "best_val_acc"      : ta.best_val_acc,
            "accuracy"          : metrics["accuracy"],
            "top2_accuracy"     : metrics["top2_accuracy"],
            "macro_f1"          : metrics["macro_f1"],
            "weighted_f1"       : metrics["weighted_f1"],
            "per_class_metrics" : metrics["per_class_metrics"],
            "mlflow_run_id"     : ta.mlflow_run_id,
            "mlflow_model_version": None,
            "mlflow_stage"      : "None",
            "is_champion"       : False,
            "promoted_at"       : None,
            "registered_at"     : registered_at.isoformat(),
        })
        logger.info(f"   ✅ Registered : {run_dir / 'run_metadata.json'}")
        return entry

    # ── step 8 : champion metric ──────────────────────────────────────────────
    def _get_champion_metric(self) -> Optional[float]:
        meta_path = self.registry_config.champion_metadata_path
        if not meta_path.exists():
            return None
        try:
            return load_json(meta_path).get(self.config.promotion_metric)
        except Exception:
            return None

    # ── step 9 : promotion ────────────────────────────────────────────────────
    def _evaluate_promotion(
        self, run_id: str, run_number: int, current_metric: float,
        prev_champion_metric: Optional[float], ta: ModelTrainerArtifact, metrics: Dict,
    ) -> bool:
        logger.info("─" * 50)
        logger.info("🏆 Champion selection")
        logger.info(f"   Current run {self.config.promotion_metric} : {current_metric:.4f}")
        logger.info(
            f"   Champion    {self.config.promotion_metric} : "
            f"{prev_champion_metric:.4f if prev_champion_metric is not None else 'none'}"
        )

        if current_metric < self.config.min_promotion_threshold:
            logger.warning(
                f"   ⚠️  Does not meet min threshold "
                f"({current_metric:.4f} < {self.config.min_promotion_threshold:.4f})"
            )
            return False

        if prev_champion_metric is not None and current_metric <= prev_champion_metric:
            logger.info(
                f"   ℹ️  Does not beat champion "
                f"({current_metric:.4f} <= {prev_champion_metric:.4f})"
            )
            return False

        promoted_at = datetime.now()
        run_model   = self.registry_config.runs_dir / run_id / "model.pth"
        shutil.copy2(run_model, self.registry_config.champion_model_path)
        logger.info(f"   🚀 Promoted → champion/model.pth")

        save_json(path=self.registry_config.champion_metadata_path, data={
            "champion_run_id"          : run_id,
            "champion_run_number"      : run_number,
            "champion_model_path"      : str(self.registry_config.champion_model_path),
            "architecture"             : ta.architecture,
            "accuracy"                 : metrics["accuracy"],
            "top2_accuracy"            : metrics["top2_accuracy"],
            "macro_f1"                 : metrics["macro_f1"],
            "weighted_f1"              : metrics["weighted_f1"],
            "per_class_metrics"        : metrics["per_class_metrics"],
            "previous_champion_metric" : prev_champion_metric,
            "promotion_metric"         : self.config.promotion_metric,
            "promoted_at"              : promoted_at.isoformat(),
        })

        # mark run_metadata as champion
        run_meta_path = self.registry_config.runs_dir / run_id / "run_metadata.json"
        run_meta = load_json(run_meta_path)
        run_meta["is_champion"] = True
        run_meta["promoted_at"] = promoted_at.isoformat()
        save_json(path=run_meta_path, data=run_meta)

        logger.info(
            f"   ✅ NEW CHAMPION  run=#{run_number}  "
            f"{self.config.promotion_metric}={current_metric:.4f}"
        )
        return True

    # ── step 10 : W&B eval logging ────────────────────────────────────────────
    def _log_wandb_eval(
        self,
        ta              : ModelTrainerArtifact,
        metrics         : Dict,
        run_id          : str,
        run_number      : int,
        all_labels      : List[int],
        all_preds       : List[int],
        is_new_champion : bool,
    ) -> None:
        try:
            wandb.init(
                project = self.config.wandb_project,
                entity  = self.config.wandb_entity,
                name    = f"eval_{ta.architecture}_{run_id[:12]}",
                tags    = ["evaluation", ta.architecture, "deepweeds"],
                resume  = "allow",
                id      = f"eval_{ta.wandb_run_id}",
            )

            # interactive confusion matrix
            wandb.log({
                "test/confusion_matrix": wandb.plot.confusion_matrix(
                    preds      = all_preds,
                    y_true     = all_labels,
                    class_names= CLASS_NAMES,
                )
            })

            # per-class F1 bar chart
            f1_data = [
                [SPECIES_MAP[int(k)], v["f1"]]
                for k, v in sorted(metrics["per_class_metrics"].items(), key=lambda x: int(x[0]))
            ]
            f1_table = wandb.Table(data=f1_data, columns=["species", "f1_score"])
            wandb.log({
                "test/per_class_f1": wandb.plot.bar(f1_table, "species", "f1_score", title="Per-Class F1 (Test)")
            })

            # overall metrics table
            overall_table = wandb.Table(
                columns = ["metric", "value"],
                data    = [
                    ["accuracy",     metrics["accuracy"]],
                    ["top2_accuracy",metrics["top2_accuracy"]],
                    ["macro_f1",     metrics["macro_f1"]],
                    ["weighted_f1",  metrics["weighted_f1"]],
                ],
            )

            # per-class metrics table
            per_class_table = wandb.Table(
                columns = ["label", "species", "precision", "recall", "f1", "support"],
                data    = [
                    [k, SPECIES_MAP[int(k)],
                     v["precision"], v["recall"], v["f1"], v["support"]]
                    for k, v in sorted(metrics["per_class_metrics"].items(), key=lambda x: int(x[0]))
                ],
            )

            wandb.log({
                "test/accuracy"         : metrics["accuracy"],
                "test/top2_accuracy"    : metrics["top2_accuracy"],
                "test/macro_f1"         : metrics["macro_f1"],
                "test/weighted_f1"      : metrics["weighted_f1"],
                "test/overall_table"    : overall_table,
                "test/per_class_table"  : per_class_table,
                "test/is_new_champion"  : int(is_new_champion),
                "test/run_number"       : run_number,
            })

            # seaborn confusion matrix image
            cm_arr = metrics["confusion_matrix"]
            import numpy as np
            cm_n   = (np.array(cm_arr).astype("float") /
                      (np.array(cm_arr).sum(axis=1, keepdims=True) + 1e-8))
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                cm_n, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"Test Confusion Matrix — Run #{run_number}")
            plt.tight_layout()
            wandb.log({"test/confusion_matrix_img": wandb.Image(fig)})
            plt.close(fig)

            if wandb.run is not None:
                wandb.finish()
            logger.info("   ✅ W&B eval logging complete")
        except Exception as e:
            logger.warning(f"W&B eval logging failed — {e}")

    # ── step 11 : MLflow eval logging + model registration ───────────────────
    def _log_mlflow_eval(
        self,
        ta                  : ModelTrainerArtifact,
        metrics             : Dict,
        run_id              : str,
        run_number          : int,
        all_labels          : List[int],
        all_preds           : List[int],
        is_new_champion     : bool,
        current_metric      : float,
        prev_champion_metric: Optional[float],
    ) -> Tuple[Optional[str], str]:
        """
        Returns (mlflow_model_version, mlflow_stage).
        """
        mlflow_model_version = None
        mlflow_stage         = "None"
        try:
            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
            mlflow.set_experiment(self.config.mlflow_experiment_name)

            # resume the original training run to attach eval metrics
            with mlflow.start_run(run_id=ta.mlflow_run_id):
                # log test metrics
                mlflow.log_metrics({
                    "test_accuracy"     : metrics["accuracy"],
                    "test_top2_accuracy": metrics["top2_accuracy"],
                    "test_macro_f1"     : metrics["macro_f1"],
                    "test_weighted_f1"  : metrics["weighted_f1"],
                })
                mlflow.log_metric("test_is_new_champion", int(is_new_champion))

                # per-class metrics
                for cls_str, m in metrics["per_class_metrics"].items():
                    mlflow.log_metric(f"test_f1_{SPECIES_MAP[int(cls_str)]}",  m["f1"])
                    mlflow.log_metric(f"test_prec_{SPECIES_MAP[int(cls_str)]}", m["precision"])
                    mlflow.log_metric(f"test_rec_{SPECIES_MAP[int(cls_str)]}", m["recall"])

                # confusion matrix artifact
                import numpy as np
                cm_arr = metrics["confusion_matrix"]
                cm_n   = (np.array(cm_arr).astype("float") /
                          (np.array(cm_arr).sum(axis=1, keepdims=True) + 1e-8))
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(
                    cm_n, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
                )
                ax.set_xlabel("Predicted"); ax.set_ylabel("True")
                ax.set_title(f"Test Confusion Matrix — Run #{run_number}")
                plt.tight_layout()
                cm_path = "confusion_matrix_test.png"
                fig.savefig(cm_path, dpi=100, bbox_inches="tight")
                mlflow.log_artifact(cm_path)
                plt.close(fig)
                os.remove(cm_path)

                # per-class metrics JSON
                import json
                pc_path = "per_class_metrics_test.json"
                with open(pc_path, "w") as f:
                    json.dump(metrics["per_class_metrics"], f, indent=2)
                mlflow.log_artifact(pc_path)
                os.remove(pc_path)

                # log champion model to registry
                model_uri = f"runs:/{ta.mlflow_run_id}/artifacts/best_model.pth"
                registered = mlflow.register_model(
                    model_uri  = model_uri,
                    name       = MLFLOW_REGISTERED_MODEL_NAME,
                )
                mlflow_model_version = registered.version
                client = MlflowClient()

                if current_metric < self.config.min_promotion_threshold:
                    # below floor — do not stage
                    mlflow_stage = "None"
                elif is_new_champion:
                    # transition current version to Production
                    client.transition_model_version_stage(
                        name    = MLFLOW_REGISTERED_MODEL_NAME,
                        version = mlflow_model_version,
                        stage   = "Production",
                    )
                    mlflow_stage = "Production"
                    # archive all other Production versions
                    for mv in client.search_model_versions(f"name='{MLFLOW_REGISTERED_MODEL_NAME}'"):
                        if (mv.version != mlflow_model_version
                                and mv.current_stage == "Production"):
                            client.transition_model_version_stage(
                                name    = MLFLOW_REGISTERED_MODEL_NAME,
                                version = mv.version,
                                stage   = "Archived",
                            )
                else:
                    client.transition_model_version_stage(
                        name    = MLFLOW_REGISTERED_MODEL_NAME,
                        version = mlflow_model_version,
                        stage   = "Staging",
                    )
                    mlflow_stage = "Staging"

                mlflow.set_tag("eval_mlflow_stage",        mlflow_stage)
                mlflow.set_tag("eval_mlflow_model_version", mlflow_model_version)
                mlflow.set_tag("eval_is_new_champion",     str(is_new_champion))

            logger.info(
                f"   ✅ MLflow model registered : "
                f"version={mlflow_model_version}  stage={mlflow_stage}"
            )
        except Exception as e:
            logger.warning(f"MLflow eval logging failed — {e}")

        return mlflow_model_version, mlflow_stage

    # ── step 12 : build + save artifact ──────────────────────────────────────
    def _build_artifact(
        self,
        trainer_artifact    : ModelTrainerArtifact,
        metrics             : Dict,
        registry_entry      : ModelRegistryEntry,
        is_new_champion     : bool,
        champion_path       : Path,
        prev_champion_metric: Optional[float],
        run_id              : str,
    ) -> ModelEvaluationArtifact:
        evaluated_at = datetime.now()
        artifact = ModelEvaluationArtifact(
            trainer_artifact         = trainer_artifact,
            accuracy                 = metrics["accuracy"],
            top2_accuracy            = metrics["top2_accuracy"],
            macro_f1                 = metrics["macro_f1"],
            weighted_f1              = metrics["weighted_f1"],
            per_class_metrics        = metrics["per_class_metrics"],
            confusion_matrix         = metrics["confusion_matrix"],
            registry_entry           = registry_entry,
            is_new_champion          = is_new_champion,
            champion_model_path      = champion_path,
            previous_champion_metric = prev_champion_metric,
            promotion_metric         = self.config.promotion_metric,
            tta_enabled              = self.config.eval_tta,
            eval_batch_size          = self.config.eval_batch_size,
            evaluated_at             = evaluated_at,
            evaluation_report_path   = self.config.evaluation_report_path,
            evaluation_history_path  = self.config.evaluation_history_path,
            artifact_path            = self.config.artifact_path,
        )
        report = {
            "run_id"                  : run_id,
            "run_number"              : registry_entry.run_number,
            "architecture"            : trainer_artifact.architecture,
            "best_model_path"         : str(trainer_artifact.best_model_path),
            "best_epoch"              : trainer_artifact.best_epoch,
            "best_val_acc"            : trainer_artifact.best_val_acc,
            "mlflow_run_id"           : trainer_artifact.mlflow_run_id,
            "wandb_run_id"            : trainer_artifact.wandb_run_id,
            "mlflow_model_version"    : registry_entry.mlflow_model_version,
            "mlflow_stage"            : registry_entry.mlflow_stage,
            "is_new_champion"         : is_new_champion,
            "champion_model_path"     : str(champion_path),
            "previous_champion_metric": prev_champion_metric,
            "promotion_metric"        : self.config.promotion_metric,
            "overall"                 : {
                "accuracy"    : metrics["accuracy"],
                "top2_accuracy": metrics["top2_accuracy"],
                "macro_f1"    : metrics["macro_f1"],
                "weighted_f1" : metrics["weighted_f1"],
            },
            "per_class"               : metrics["per_class_metrics"],
            "confusion_matrix"        : metrics["confusion_matrix"],
            "tta_enabled"             : self.config.eval_tta,
            "evaluated_at"            : evaluated_at.isoformat(),
        }
        save_json(path=self.config.evaluation_report_path, data=report)
        save_json(path=self.config.artifact_path,          data=report)
        logger.info(f"📋 Report   : {self.config.evaluation_report_path}")
        return artifact

    # ── helpers ───────────────────────────────────────────────────────────────
    def _next_run_number(self) -> int:
        runs_dir = self.registry_config.runs_dir
        if not runs_dir.exists():
            return 1
        return len([d for d in runs_dir.iterdir() if d.is_dir()]) + 1

    def _append_history(
        self, run_id: str, run_number: int, metrics: Dict,
        is_new_champion: bool, mlflow_stage: str,
    ) -> None:
        history: List[Dict] = []
        if self.config.evaluation_history_path.exists():
            try:
                history = load_json(self.config.evaluation_history_path)
            except Exception:
                history = []
        history.append({
            "run_id"        : run_id,
            "run_number"    : run_number,
            "accuracy"      : metrics["accuracy"],
            "top2_accuracy" : metrics["top2_accuracy"],
            "macro_f1"      : metrics["macro_f1"],
            "weighted_f1"   : metrics["weighted_f1"],
            "is_champion"   : is_new_champion,
            "mlflow_stage"  : mlflow_stage,
            "evaluated_at"  : datetime.now().isoformat(),
        })
        save_json(path=self.config.evaluation_history_path, data=history)
        logger.info(f"📋 History  : {self.config.evaluation_history_path}")

    def _already_evaluated(self, run_id: str) -> bool:
        if not self.config.evaluation_state_path.exists():
            return False
        try:
            return load_json(self.config.evaluation_state_path).get("last_run_id") == run_id
        except Exception:
            return False

    def _update_state(self, run_id: str, artifact: ModelEvaluationArtifact) -> None:
        save_json(path=self.config.evaluation_state_path, data={
            "last_run_id"          : run_id,
            "last_evaluated_at"    : artifact.evaluated_at.isoformat(),
            "last_accuracy"        : artifact.accuracy,
            "last_weighted_f1"     : artifact.weighted_f1,
            "last_is_new_champion" : artifact.is_new_champion,
            "champion_model_path"  : str(artifact.champion_model_path),
            "mlflow_stage"         : artifact.registry_entry.mlflow_stage,
        })
        logger.info(f"💾 State    : {self.config.evaluation_state_path}")

    def _load_trainer_artifact(self) -> ModelTrainerArtifact:
        from datetime import datetime as _dt
        from weed_detection.components.data_validation import load_ingestion_artifact
        from weed_detection.entity.artifact_entity import (
            DataTransformationArtifact as DTA,
            DataValidationArtifact     as DVA,
            ModelTrainerArtifact       as MTA,
        )
        config_manager = ConfigurationManager()
        mt_config      = config_manager.get_model_trainer_config()
        if not mt_config.artifact_path.exists():
            raise FileNotFoundError(
                f"No ModelTrainerArtifact at {mt_config.artifact_path}\n"
                "Run model_trainer.py first."
            )
        d         = load_json(mt_config.artifact_path)
        dv_config = config_manager.get_data_validation_config()
        dt_config = config_manager.get_data_transformation_config()
        report    = load_json(dv_config.validation_report_path)
        ingestion = load_ingestion_artifact(Path(report["ingestion_artifact"]))

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
            drop_last             = dt_d.get("drop_last", True),
            weight_exponent       = dt_d.get("weight_exponent", 1.0),
            transformed_at        = _dt.fromisoformat(dt_d["transformed_at"]),
            artifact_path         = dt_config.artifact_path,
        )
        return MTA(
            transformation_artifact  = transformation_artifact,
            best_model_path          = Path(d["best_model_path"]),
            final_model_path         = Path(d["final_model_path"]),
            checkpoints_dir          = Path(d["checkpoints_dir"]),
            architecture             = d["architecture"],
            num_classes              = d["num_classes"],
            pretrained               = d["pretrained"],
            dropout_rate             = d["dropout_rate"],
            total_params             = d["total_params"],
            trainable_params         = d["trainable_params"],
            epochs_trained           = d["epochs_trained"],
            best_epoch               = d["best_epoch"],
            best_val_acc             = d["best_val_acc"],
            best_val_loss            = d["best_val_loss"],
            final_train_acc          = d["final_train_acc"],
            final_train_loss         = d["final_train_loss"],
            learning_rate            = d["learning_rate"],
            weight_decay             = d["weight_decay"],
            lr_scheduler             = d["lr_scheduler"],
            warmup_epochs            = d["warmup_epochs"],
            early_stopping_patience  = d["early_stopping_patience"],
            monitor_metric           = d["monitor_metric"],
            batch_size               = d["batch_size"],
            input_size               = d["input_size"],
            sampler                  = d["sampler"],
            drop_last                = d.get("drop_last", True),
            weight_exponent          = d.get("weight_exponent", 1.0),
            grad_clip_norm           = d.get("grad_clip_norm", 0.5),
            use_focal_loss           = d.get("use_focal_loss", True),
            focal_gamma              = d.get("focal_gamma", 2.0),
            label_smoothing          = d.get("label_smoothing", 0.1),
            mixed_precision          = d.get("mixed_precision", True),
            nan_batches_total        = d.get("nan_batches_total", 0),
            per_class_val_acc        = d["per_class_val_acc"],
            mlflow_run_id            = d["mlflow_run_id"],
            wandb_run_id             = d["wandb_run_id"],
            wandb_run_url            = d["wandb_run_url"],
            mlflow_tracking_uri      = d["mlflow_tracking_uri"],
            mlflow_experiment_name   = d["mlflow_experiment_name"],
            device                   = d["device"],
            cuda_version             = d.get("cuda_version"),
            training_history_path    = Path(d["training_history_path"]),
            trained_at               = _dt.fromisoformat(d["trained_at"]),
            total_training_time_s    = d["total_training_time_s"],
            artifact_path            = mt_config.artifact_path,
        )

    def _log_summary(self, artifact: ModelEvaluationArtifact) -> None:
        logger.info("=" * 70)
        logger.info("📊 MODEL EVALUATION + SELECTION COMPLETE")
        logger.info(f"   Run #           : {artifact.registry_entry.run_number}")
        logger.info(f"   Test accuracy   : {artifact.accuracy:.4f}")
        logger.info(f"   Top-2 accuracy  : {artifact.top2_accuracy:.4f}")
        logger.info(f"   Macro F1        : {artifact.macro_f1:.4f}")
        logger.info(f"   Weighted F1     : {artifact.weighted_f1:.4f}")
        logger.info(f"   Is new champion : {'🏆 YES' if artifact.is_new_champion else 'NO'}")
        logger.info(f"   MLflow stage    : {artifact.registry_entry.mlflow_stage}")
        logger.info(f"   MLflow version  : {artifact.registry_entry.mlflow_model_version}")
        logger.info(f"   Champion model  : {artifact.champion_model_path}")
        logger.info(f"   Report          : {artifact.evaluation_report_path}")
        logger.info(f"   History         : {artifact.evaluation_history_path}")
        logger.info("=" * 70)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    logger.info("=" * 70)
    logger.info("🚀 Starting Model Evaluation + Selection")
    logger.info("=" * 70)

    config_manager  = ConfigurationManager()
    config          = config_manager.get_model_evaluation_config()
    registry_config = config_manager.get_model_registry_config()

    evaluator = ModelEvaluation(config, registry_config)
    artifact  = evaluator.run()

    if artifact is None:
        logger.info("✅ Nothing to do — already evaluated (same run)")
    else:
        logger.info(f"✅ Evaluation complete")
        logger.info(f"   Test accuracy : {artifact.accuracy:.4f}")
        logger.info(f"   Weighted F1   : {artifact.weighted_f1:.4f}")
        if artifact.is_new_champion:
            logger.info("   🏆 New champion promoted to model_registry/champion/")
        if artifact.accuracy < 0.80:
            logger.warning("⚠️  Accuracy below 0.80 — consider retraining")
    return artifact


if __name__ == "__main__":
    main()