import csv
import heapq
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    StepLR,
)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


from weed_detection import logger
from weed_detection.config.configuration import ConfigurationManager
from weed_detection.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from weed_detection.entity.config_entity import ModelTrainerConfig
from weed_detection.utils.utility import load_json, save_json
from weed_detection.constants.constant import SPECIES_MAP,NUM_CLASSES,IMAGENET_MEAN,IMAGENET_STD


class DeepWeedDataset(Dataset):
    """
    Minimal Dataset — reads split CSV, loads images on the fly.
    Identical to the one in data_transformation.py so trainer is self-contained.
    """

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

        logger.info(
            f"   Dataset [{csv_path.stem}] : "
            f"{len(self.samples)} samples from {images_dir}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        filename, label = self.samples[idx]
        image = Image.open(self.images_dir / filename).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    @property
    def labels(self) -> List[int]:
        return [label for _, label in self.samples]


# ── Early stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Stops training if monitored metric does not improve for `patience` epochs.
    monitor_metric: 'val_acc'  → higher is better
                    'val_loss' → lower is better
    """

    def __init__(self, patience: int, monitor: str = "val_acc"):
        self.patience  = patience
        self.monitor   = monitor
        self.counter   = 0
        self.best      = None
        self.triggered = False

    def step(self, value: float) -> bool:
        improved = (
            self.best is None
            or (self.monitor == "val_acc"  and value > self.best)
            or (self.monitor == "val_loss" and value < self.best)
        )
        if improved:
            self.best    = value
            self.counter = 0
        else:
            self.counter += 1
            logger.info(
                f"   EarlyStopping: no improvement for "
                f"{self.counter}/{self.patience} epochs "
                f"(best={self.best:.4f})"
            )
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered


# ── Top-K checkpoint manager ──────────────────────────────────────────────────

class TopKCheckpointManager:
    """
    Keeps only the top-k checkpoints by val_acc.
    Deletes the worst checkpoint when a new one displaces it.
    """

    def __init__(self, checkpoints_dir: Path, k: int):
        self.checkpoints_dir = checkpoints_dir
        self.k               = k
        self._heap: List[Tuple[float, Path]] = []   # min-heap (acc, path)

    def update(self, model: nn.Module, epoch: int, val_acc: float) -> Path:
        ckpt_path = (
            self.checkpoints_dir
            / f"epoch_{epoch:03d}_acc_{val_acc:.4f}.pth"
        )
        torch.save(model.state_dict(), ckpt_path)
        heapq.heappush(self._heap, (val_acc, ckpt_path))

        if len(self._heap) > self.k:
            worst_acc, worst_path = heapq.heappop(self._heap)
            if worst_path.exists():
                worst_path.unlink()
                logger.info(
                    f"   🗑️  Removed checkpoint: {worst_path.name} "
                    f"(acc={worst_acc:.4f})"
                )

        logger.info(f"   💾 Checkpoint saved: {ckpt_path.name}")
        return ckpt_path

    @property
    def best_path(self) -> Optional[Path]:
        if not self._heap:
            return None
        return max(self._heap, key=lambda x: x[0])[1]


# ── Model builder ─────────────────────────────────────────────────────────────

def build_model(
    architecture : str,
    num_classes  : int,
    pretrained   : bool,
    dropout_rate : float,
) -> nn.Module:
    """
    Build EfficientNet-B3 (or any timm model) with custom classification head.

    Custom head:
      AdaptiveAvgPool → Dropout(dropout_rate) → Linear(num_features, num_classes)

    Replaces timm's default classifier while keeping the pretrained backbone.
    """
    model = timm.create_model(
        architecture,
        pretrained  = pretrained,
        num_classes = 0,        # remove timm's default head
        global_pool = "avg",    # keep global average pooling
    )

    num_features = model.num_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(num_features, num_classes),
    )

    total_params    = sum(p.numel() for p in model.parameters())
    trainable_params= sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"   Architecture  : {architecture}")
    logger.info(f"   Pretrained    : {pretrained}")
    logger.info(f"   Num features  : {num_features}")
    logger.info(f"   Total params  : {total_params:,}")
    logger.info(f"   Trainable     : {trainable_params:,}")

    return model


# ── Scheduler builder ─────────────────────────────────────────────────────────

def build_scheduler(
    optimizer    : optim.Optimizer,
    scheduler    : str,
    epochs       : int,
    warmup_epochs: int,
) -> object:
    effective_epochs = max(epochs - warmup_epochs, 1)

    if scheduler == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max  = effective_epochs,
            eta_min= 1e-6,
        )
    elif scheduler == "step":
        return StepLR(optimizer, step_size=10, gamma=0.1)
    elif scheduler == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode    = "max",
            factor  = 0.5,
            patience= 3,
            min_lr  = 1e-6,
        )
    else:
        raise ValueError(
            f"Unknown scheduler '{scheduler}'. "
            f"Choose from: cosine | step | plateau"
        )


# ── ModelTrainer ──────────────────────────────────────────────────────────────

class ModelTrainer:

    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"   Device : {self.device}")
        if self.device.type == "cuda":
            logger.info(f"   GPU    : {torch.cuda.get_device_name(0)}")

    # ── public entry point ────────────────────────────────────────────────────

    def run(self) -> Optional[ModelTrainerArtifact]:
        logger.info("=" * 60)
        logger.info("🚀 Model Trainer — version check")
        logger.info("=" * 60)

        # 1. load transformation artifact
        transformation_artifact = self._load_transformation_artifact()

        # 2. version check
        version_id = (
            transformation_artifact
            .validation_artifact
            .ingestion_artifact
            .normalized_dir.name
        )
        if self._already_trained(version_id):
            logger.info(f"⏭️  Version '{version_id}' already trained — skipping")
            return None

        logger.info(f"🆕 Training version : {version_id}")

        # 3. build model
        logger.info("─" * 50)
        logger.info("🏗️  Step 3 — Building model")
        model = build_model(
            architecture = self.config.architecture,
            num_classes  = self.config.num_classes,
            pretrained   = self.config.pretrained,
            dropout_rate = self.config.dropout_rate,
        )
        model = model.to(self.device)
        total_params     = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # 4. build dataloaders
        logger.info("─" * 50)
        logger.info("📦 Step 4 — Building DataLoaders")
        train_loader, val_loader = self._build_loaders(transformation_artifact)

        # 5. loss function
        logger.info("─" * 50)
        logger.info("⚖️  Step 5 — Configuring loss")
        class_weights = torch.tensor(
            transformation_artifact.class_weights,
            dtype=torch.float32,
        ).to(self.device)

        criterion = nn.CrossEntropyLoss(
            weight          = class_weights,
            label_smoothing = self.config.label_smoothing,
        )
        logger.info(f"   Loss           : CrossEntropyLoss")
        logger.info(f"   Class weights  : ✅ applied")
        logger.info(f"   Label smoothing: {self.config.label_smoothing}")

        # 6. optimizer + scheduler
        logger.info("─" * 50)
        logger.info("🔧 Step 6 — Optimizer + Scheduler")
        optimizer = optim.AdamW(
            model.parameters(),
            lr           = self.config.learning_rate,
            weight_decay = self.config.weight_decay,
        )
        scheduler = build_scheduler(
            optimizer     = optimizer,
            scheduler     = self.config.lr_scheduler,
            epochs        = self.config.epochs,
            warmup_epochs = self.config.warmup_epochs,
        )
        logger.info(f"   Optimizer  : AdamW  lr={self.config.learning_rate}  wd={self.config.weight_decay}")
        logger.info(f"   Scheduler  : {self.config.lr_scheduler}")
        logger.info(f"   Warmup     : {self.config.warmup_epochs} epochs")

        # 7-9. training loop
        history, best_epoch, best_val_acc, best_val_loss, \
        per_class_val_acc, epochs_trained = self._train(
            model, train_loader, val_loader,
            criterion, optimizer, scheduler,
        )

        # 10. save final model
        torch.save(model.state_dict(), self.config.final_model_path)
        logger.info(f"   💾 Final model : {self.config.final_model_path}")

        # 11. build + save artifact
        artifact = self._build_artifact(
            transformation_artifact = transformation_artifact,
            history                 = history,
            best_epoch              = best_epoch,
            best_val_acc            = best_val_acc,
            best_val_loss           = best_val_loss,
            per_class_val_acc       = per_class_val_acc,
            epochs_trained          = epochs_trained,
            total_params            = total_params,
            trainable_params        = trainable_params,
        )
        self._update_state(version_id, artifact)
        self._log_summary(artifact)
        return artifact

    # ── step 4 : dataloaders ──────────────────────────────────────────────────

    def _build_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
        size = self.config.input_size

        train_tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=90),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2,
                saturation=0.2, hue=0.05,
            ),
            transforms.RandomResizedCrop(
                size=size, scale=(0.8, 1.0), ratio=(0.9, 1.1)
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        eval_tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        return train_tf, eval_tf

    def _build_loaders(
        self, ta: DataTransformationArtifact
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        train_tf, eval_tf = self._build_transforms()

        train_dataset = DeepWeedDataset(
            ta.train_csv_path, ta.train_images_dir, train_tf
        )

        if self.config.sampler == "weighted":
            sample_weights = [
                ta.class_weights[label] for label in train_dataset.labels
            ]
            sampler = WeightedRandomSampler(
                weights     = sample_weights,
                num_samples = len(sample_weights),
                replacement = True,
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size  = self.config.batch_size,
                sampler     = sampler,
                num_workers = self.config.num_workers,
                pin_memory  = self.config.pin_memory,
                drop_last   = True,
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size  = self.config.batch_size,
                shuffle     = True,
                num_workers = self.config.num_workers,
                pin_memory  = self.config.pin_memory,
                drop_last   = True,
            )

        val_loader = None
        if ta.val_csv_path and ta.val_images_dir:
            val_dataset = DeepWeedDataset(
                ta.val_csv_path, ta.val_images_dir, eval_tf
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size  = self.config.batch_size,
                shuffle     = False,
                num_workers = self.config.num_workers,
                pin_memory  = self.config.pin_memory,
            )

        logger.info(
            f"   Train batches : {len(train_loader)}  "
            f"(batch={self.config.batch_size}  "
            f"sampler={self.config.sampler})"
        )
        if val_loader:
            logger.info(f"   Val batches   : {len(val_loader)}")

        return train_loader, val_loader

    # ── steps 7-9 : training loop ─────────────────────────────────────────────

    def _train(
        self,
        model        : nn.Module,
        train_loader : DataLoader,
        val_loader   : Optional[DataLoader],
        criterion    : nn.Module,
        optimizer    : optim.Optimizer,
        scheduler    : object,
    ) -> Tuple[List[Dict], int, float, float, Dict, int]:
        """
        Full training loop with:
          - Linear warmup for warmup_epochs
          - Per-epoch train + val
          - Top-k checkpoint saving
          - Early stopping
          - Per-class val accuracy at best epoch

        Returns:
          history, best_epoch, best_val_acc, best_val_loss,
          per_class_val_acc, epochs_trained
        """
        logger.info("─" * 50)
        logger.info("🏋️  Steps 7-9 — Training loop")

        ckpt_manager  = TopKCheckpointManager(
            self.config.checkpoints_dir,
            self.config.save_top_k,
        )
        early_stopping = EarlyStopping(
            patience = self.config.early_stopping_patience,
            monitor  = self.config.monitor_metric,
        )

        history         : List[Dict] = []
        best_epoch      : int        = 0
        best_val_acc    : float      = 0.0
        best_val_loss   : float      = float("inf")
        per_class_val_acc: Dict      = {}
        training_start  = time.time()

        base_lr = self.config.learning_rate

        for epoch in range(1, self.config.epochs + 1):
            epoch_start = time.time()

            # ── warmup LR ─────────────────────────────────────────────────────
            if epoch <= self.config.warmup_epochs:
                warmup_lr = base_lr * epoch / self.config.warmup_epochs
                for pg in optimizer.param_groups:
                    pg["lr"] = warmup_lr
                logger.info(
                    f"   [Warmup {epoch}/{self.config.warmup_epochs}] "
                    f"lr={warmup_lr:.6f}"
                )

            # ── train one epoch ───────────────────────────────────────────────
            train_loss, train_acc = self._train_epoch(
                model, train_loader, criterion, optimizer
            )

            # ── validate ──────────────────────────────────────────────────────
            val_loss, val_acc, class_acc = (0.0, 0.0, {})
            if val_loader:
                val_loss, val_acc, class_acc = self._val_epoch(
                    model, val_loader, criterion
                )

            # ── scheduler step ────────────────────────────────────────────────
            if epoch > self.config.warmup_epochs:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_acc)
                else:
                    scheduler.step()

            current_lr  = optimizer.param_groups[0]["lr"]
            epoch_time  = time.time() - epoch_start

            row = {
                "epoch"       : epoch,
                "train_loss"  : round(train_loss, 6),
                "train_acc"   : round(train_acc,  6),
                "val_loss"    : round(val_loss,   6),
                "val_acc"     : round(val_acc,    6),
                "lr"          : round(current_lr, 8),
                "epoch_time_s": round(epoch_time, 2),
            }
            history.append(row)

            logger.info(
                f"   Epoch [{epoch:>3}/{self.config.epochs}] "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
                f"lr={current_lr:.6f}  time={epoch_time:.1f}s"
            )

            # ── checkpoint + best model tracking ─────────────────────────────
            is_best = (
                (self.config.monitor_metric == "val_acc"  and val_acc  > best_val_acc)
                or
                (self.config.monitor_metric == "val_loss" and val_loss < best_val_loss)
            )

            if is_best:
                best_epoch       = epoch
                best_val_acc     = val_acc
                best_val_loss    = val_loss
                per_class_val_acc= class_acc
                torch.save(model.state_dict(), self.config.best_model_path)
                logger.info(
                    f"   ⭐ New best model  "
                    f"val_acc={best_val_acc:.4f}  "
                    f"val_loss={best_val_loss:.4f}  "
                    f"→ {self.config.best_model_path.name}"
                )

            ckpt_manager.update(model, epoch, val_acc)

            # ── save history every epoch ──────────────────────────────────────
            save_json(
                path = self.config.training_history_path,
                data = history,
            )

            # ── early stopping ────────────────────────────────────────────────
            monitor_value = val_acc if self.config.monitor_metric == "val_acc" else val_loss
            if early_stopping.step(monitor_value):
                logger.info(
                    f"   🛑 Early stopping triggered at epoch {epoch} "
                    f"(patience={self.config.early_stopping_patience})"
                )
                break

        total_time = time.time() - training_start
        epochs_trained = len(history)

        logger.info("─" * 50)
        logger.info(f"✅ Training complete")
        logger.info(f"   Epochs trained : {epochs_trained}")
        logger.info(f"   Best epoch     : {best_epoch}  val_acc={best_val_acc:.4f}")
        logger.info(f"   Total time     : {total_time:.1f}s  ({total_time/60:.1f}min)")

        return (
            history, best_epoch, best_val_acc, best_val_loss,
            per_class_val_acc, epochs_trained,
        )

    # ── one train epoch ───────────────────────────────────────────────────────

    def _train_epoch(
        self,
        model      : nn.Module,
        loader     : DataLoader,
        criterion  : nn.Module,
        optimizer  : optim.Optimizer,
    ) -> Tuple[float, float]:
        model.train()
        total_loss    = 0.0
        correct       = 0
        total_samples = 0

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()

            # gradient clipping — stabilises training
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss    += loss.item() * images.size(0)
            preds          = outputs.argmax(dim=1)
            correct       += (preds == labels).sum().item()
            total_samples += images.size(0)

        avg_loss = total_loss    / total_samples
        accuracy = correct       / total_samples
        return avg_loss, accuracy

    # ── one val epoch ─────────────────────────────────────────────────────────

    def _val_epoch(
        self,
        model    : nn.Module,
        loader   : DataLoader,
        criterion: nn.Module,
    ) -> Tuple[float, float, Dict[str, float]]:
        model.eval()
        total_loss    = 0.0
        correct       = 0
        total_samples = 0

        # per-class correct / total for class-level accuracy
        class_correct : Dict[int, int] = defaultdict(int)
        class_total   : Dict[int, int] = defaultdict(int)

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                outputs = model(images)
                loss    = criterion(outputs, labels)

                total_loss    += loss.item() * images.size(0)
                preds          = outputs.argmax(dim=1)
                correct       += (preds == labels).sum().item()
                total_samples += images.size(0)

                for label, pred in zip(labels.cpu().tolist(), preds.cpu().tolist()):
                    class_total[label]   += 1
                    class_correct[label] += int(label == pred)

        avg_loss = total_loss    / total_samples
        accuracy = correct       / total_samples

        # per-class accuracy dict  { "0": 0.912, "1": 0.876, ... }
        class_acc = {
            str(cls): round(
                class_correct[cls] / class_total[cls], 6
            ) if class_total[cls] > 0 else 0.0
            for cls in range(NUM_CLASSES)
        }

        return avg_loss, accuracy, class_acc

    # ── build artifact ────────────────────────────────────────────────────────

    def _build_artifact(
        self,
        transformation_artifact: DataTransformationArtifact,
        history                : List[Dict],
        best_epoch             : int,
        best_val_acc           : float,
        best_val_loss          : float,
        per_class_val_acc      : Dict,
        epochs_trained         : int,
        total_params           : int,
        trainable_params       : int,
    ) -> ModelTrainerArtifact:

        total_time    = sum(r["epoch_time_s"] for r in history)
        final_row     = history[-1] if history else {}
        cuda_version  = (
            torch.version.cuda if torch.cuda.is_available() else "N/A"
        )

        artifact = ModelTrainerArtifact(
            transformation_artifact = transformation_artifact,
            best_model_path         = self.config.best_model_path,
            final_model_path        = self.config.final_model_path,
            checkpoints_dir         = self.config.checkpoints_dir,
            architecture            = self.config.architecture,
            num_classes             = self.config.num_classes,
            pretrained              = self.config.pretrained,
            total_params            = total_params,
            trainable_params        = trainable_params,
            epochs_trained          = epochs_trained,
            best_epoch              = best_epoch,
            best_val_acc            = best_val_acc,
            best_val_loss           = best_val_loss,
            final_train_acc         = final_row.get("train_acc", 0.0),
            final_train_loss        = final_row.get("train_loss", 0.0),
            per_class_val_acc       = per_class_val_acc,
            learning_rate           = self.config.learning_rate,
            weight_decay            = self.config.weight_decay,
            lr_scheduler            = self.config.lr_scheduler,
            batch_size              = self.config.batch_size,
            label_smoothing         = self.config.label_smoothing,
            dropout_rate            = self.config.dropout_rate,
            sampler                 = self.config.sampler,
            device                  = str(self.device),
            cuda_version            = cuda_version or "N/A",
            training_history_path   = self.config.training_history_path,
            trained_at              = datetime.now(),
            total_training_time_s   = round(total_time, 2),
            artifact_path           = self.config.artifact_path,
        )

        save_json(
            path = self.config.artifact_path,
            data = {
                "version_id"            : (
                    transformation_artifact
                    .validation_artifact
                    .ingestion_artifact
                    .normalized_dir.name
                ),
                "trained_at"            : artifact.trained_at.isoformat(),
                "architecture"          : artifact.architecture,
                "num_classes"           : artifact.num_classes,
                "pretrained"            : artifact.pretrained,
                "total_params"          : artifact.total_params,
                "trainable_params"      : artifact.trainable_params,
                "epochs_trained"        : artifact.epochs_trained,
                "best_epoch"            : artifact.best_epoch,
                "best_val_acc"          : artifact.best_val_acc,
                "best_val_loss"         : artifact.best_val_loss,
                "final_train_acc"       : artifact.final_train_acc,
                "final_train_loss"      : artifact.final_train_loss,
                "per_class_val_acc"     : artifact.per_class_val_acc,
                "learning_rate"         : artifact.learning_rate,
                "weight_decay"          : artifact.weight_decay,
                "lr_scheduler"          : artifact.lr_scheduler,
                "batch_size"            : artifact.batch_size,
                "label_smoothing"       : artifact.label_smoothing,
                "dropout_rate"          : artifact.dropout_rate,
                "sampler"               : artifact.sampler,
                "device"                : artifact.device,
                "cuda_version"          : artifact.cuda_version,
                "best_model_path"       : str(artifact.best_model_path),
                "final_model_path"      : str(artifact.final_model_path),
                "checkpoints_dir"       : str(artifact.checkpoints_dir),
                "training_history_path" : str(artifact.training_history_path),
                "total_training_time_s" : artifact.total_training_time_s,
                "transformation_artifact": str(
                    transformation_artifact.artifact_path
                ),
            }
        )
        logger.info(f"📋 Artifact saved : {self.config.artifact_path}")
        return artifact

    # ── version control ───────────────────────────────────────────────────────

    def _already_trained(self, version_id: str) -> bool:
        state_path = self.config.trainer_state_path
        if not state_path.exists():
            return False
        try:
            state = load_json(state_path)
            return state.get("last_version_id") == version_id
        except Exception as e:
            logger.warning(f"⚠️  Could not read trainer state: {e} — will retrain")
            return False

    def _update_state(
        self, version_id: str, artifact: ModelTrainerArtifact
    ) -> None:
        save_json(
            path = self.config.trainer_state_path,
            data = {
                "last_version_id"     : version_id,
                "last_trained_at"     : artifact.trained_at.isoformat(),
                "last_best_val_acc"   : artifact.best_val_acc,
                "last_best_epoch"     : artifact.best_epoch,
                "last_epochs_trained" : artifact.epochs_trained,
                "last_artifact_path"  : str(artifact.artifact_path),
            }
        )
        logger.info(f"💾 Trainer state saved : {self.config.trainer_state_path}")

    # ── load transformation artifact ──────────────────────────────────────────

    def _load_transformation_artifact(self) -> DataTransformationArtifact:
        from weed_detection.components.data_transformation import (
            DataTransformation,
        )
        from weed_detection.components.data_validation import (
            load_ingestion_artifact,
        )
        from weed_detection.entity.artifact_entity import (
            DataTransformationArtifact as DTA,
            DataValidationArtifact     as DVA,
        )

        config_manager = ConfigurationManager()
        dt_config      = config_manager.get_data_transformation_config()
        artifact_path  = dt_config.artifact_path

        if not artifact_path.exists():
            raise FileNotFoundError(
                f"No DataTransformationArtifact at {artifact_path}\n"
                f"Run data_transformation.py first."
            )

        d = load_json(artifact_path)

        # re-hydrate validation artifact
        dv_config  = config_manager.get_data_validation_config()
        report     = load_json(dv_config.validation_report_path)
        ingestion  = load_ingestion_artifact(
            Path(report["ingestion_artifact"])
        )

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

        artifact = DTA(
            validation_artifact   = validation_artifact,
            train_images_dir      = Path(d["train_images_dir"]),
            train_csv_path        = Path(d["train_csv_path"]),
            val_images_dir        = Path(d["val_images_dir"])  if d.get("val_images_dir")  else None,
            val_csv_path          = Path(d["val_csv_path"])    if d.get("val_csv_path")    else None,
            test_images_dir       = Path(d["test_images_dir"]) if d.get("test_images_dir") else None,
            test_csv_path         = Path(d["test_csv_path"])   if d.get("test_csv_path")   else None,
            class_weights_path    = Path(d["class_weights_path"]),
            class_weights         = d["class_weights"],
            transform_config_path = Path(d["transform_config_path"]),
            input_size            = d["input_size"],
            batch_size            = d["batch_size"],
            num_workers           = d["num_workers"],
            sampler               = d["sampler"],
            pin_memory            = d["pin_memory"],
            transformed_at        = _dt.fromisoformat(d["transformed_at"]),
            artifact_path         = artifact_path,
        )
        logger.info(f"✅ TransformationArtifact loaded")
        logger.info(f"   Train CSV : {artifact.train_csv_path}")
        logger.info(f"   Val CSV   : {artifact.val_csv_path}")
        return artifact

    # ── summary ───────────────────────────────────────────────────────────────

    def _log_summary(self, artifact: ModelTrainerArtifact) -> None:
        logger.info("=" * 60)
        logger.info("📊 MODEL TRAINING COMPLETE")
        logger.info(f"   Architecture      : {artifact.architecture}")
        logger.info(f"   Total params      : {artifact.total_params:,}")
        logger.info(f"   Trainable params  : {artifact.trainable_params:,}")
        logger.info(f"   Epochs trained    : {artifact.epochs_trained}")
        logger.info(f"   Best epoch        : {artifact.best_epoch}")
        logger.info(f"   Best val acc      : {artifact.best_val_acc:.4f}")
        logger.info(f"   Best val loss     : {artifact.best_val_loss:.4f}")
        logger.info(f"   Final train acc   : {artifact.final_train_acc:.4f}")
        logger.info(f"   Training time     : {artifact.total_training_time_s:.1f}s")
        logger.info(f"   Device            : {artifact.device}")
        logger.info("─" * 50)
        logger.info("   Per-class val accuracy (best epoch):")
        for cls_str, acc in sorted(
            artifact.per_class_val_acc.items(), key=lambda x: int(x[0])
        ):
            species = SPECIES_MAP.get(int(cls_str), f"Class {cls_str}")
            logger.info(f"      Label {cls_str} ({species:<16}) : {acc:.4f}")
        logger.info("─" * 50)
        logger.info(f"   Best model  : {artifact.best_model_path}")
        logger.info(f"   Final model : {artifact.final_model_path}")
        logger.info(f"   History     : {artifact.training_history_path}")
        logger.info(f"   Artifact    : {artifact.artifact_path}")
        logger.info("=" * 60)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("🚀 Starting Model Trainer")
    logger.info("=" * 60)

    config_manager = ConfigurationManager()
    config         = config_manager.get_model_trainer_config()

    trainer  = ModelTrainer(config)
    artifact = trainer.run()

    if artifact is None:
        logger.info("✅ Nothing to do — already trained (same version)")
    else:
        logger.info(f"✅ Training complete — best val_acc: {artifact.best_val_acc:.4f}")

    return artifact


if __name__ == "__main__":
    main()