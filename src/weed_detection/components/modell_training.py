import csv
import heapq
import os
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

try:
    import timm
except ImportError:
    raise ImportError("pip install timm")

try:
    import mlflow
    import mlflow.pytorch
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
    MLFLOW_DATASET,
    MLFLOW_MODEL_FAMILY,
    MLFLOW_TASK,
    NUM_CLASSES,
    PYTORCH_CUDA_ALLOC_CONF,
    SPECIES_MAP,
    WANDB_TAGS,
)
from weed_detection.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from weed_detection.entity.config_entity import ModelTrainerConfig
from weed_detection.utils.utility import load_json, save_json


# ── FocalLoss — notebook Cell 10 ──────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss — down-weights easy examples so the model focuses on hard ones.
    Works with class weights to further handle imbalance.

    inputs : float32 logits  (caller must cast from float16 before passing)
    targets: long labels
    """
    def __init__(self, weight=None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.weight    = weight
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss    = F.cross_entropy(inputs, targets, reduction="none", weight=self.weight)
        pt         = torch.exp(-ce_loss)          # pt = e^(-ce) = p_correct
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# ── Dataset — notebook Cell 14 ────────────────────────────────────────────────
class DeepWeedDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples   = samples
        self.transform = transform
        self.labels    = [label for _, label in samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ── ModelTrainer ──────────────────────────────────────────────────────────────
class ModelTrainer:

    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        # CUDA memory optimisation — notebook Cell 2
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = PYTORCH_CUDA_ALLOC_CONF
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device : {self.device}")
        if self.device.type == "cuda":
            logger.info(f"GPU    : {torch.cuda.get_device_name(0)}")
            torch.backends.cudnn.benchmark = True   # notebook Cell 25

    # ── public entry point ────────────────────────────────────────────────────
    def run(
        self, transformation_artifact: DataTransformationArtifact
    ) -> Optional[ModelTrainerArtifact]:
        logger.info("=" * 70)
        logger.info("🚀 Model Trainer — version check")
        logger.info("=" * 70)

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

        # ── build run name — notebook Cell 9 ──────────────────────────────────
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{self.config.architecture}_{ts}"
        logger.info(f"Run name : {run_name}")

        # ── load samples — notebook Cell 11 / 12 ──────────────────────────────
        train_samples = self._load_samples(
            transformation_artifact.train_csv_path,
            transformation_artifact.train_images_dir,
        )
        val_samples = self._load_samples(
            transformation_artifact.val_csv_path,
            transformation_artifact.val_images_dir,
        ) if transformation_artifact.val_csv_path else []

        # ── class weights — notebook Cell 13 ─────────────────────────────────
        class_weights = self._compute_class_weights(train_samples)

        # ── transforms + loaders — notebook Cell 15 / 16 ─────────────────────
        train_loader, val_loader = self._build_loaders(
            train_samples, val_samples, class_weights
        )

        # ── model — notebook Cell 18 ──────────────────────────────────────────
        model, scaler, total_params, trainable_params = self._build_model()

        # ── loss / optimizer / scheduler — notebook Cell 22 ──────────────────
        criterion = self._build_criterion(class_weights)
        optimizer = self._build_optimizer(model)
        scheduler = self._build_scheduler(optimizer)

        # ── MLflow setup — notebook Cell 23 / 24 ─────────────────────────────
        mlflow_run = self._setup_mlflow(
            run_name, train_samples, val_samples,
            total_params, trainable_params, class_weights,
        )
        mlflow_run_id = mlflow_run.info.run_id

        # ── W&B setup — notebook Cell 25 ─────────────────────────────────────
        wandb_run = self._setup_wandb(
            run_name, model, train_samples, val_samples, class_weights
        )
        # cross-link — notebook Cell 25
        mlflow.set_tag("wandb_url",    wandb_run.url)
        mlflow.set_tag("wandb_run_id", wandb_run.id)
        wandb_run_id  = wandb_run.id
        wandb_run_url = wandb_run.url

        # ── training loop — notebook Cell 27 / 28 ────────────────────────────
        result = self._train_loop(
            model, scaler, criterion, optimizer, scheduler,
            train_loader, val_loader,
        )

        # ── close trackers ────────────────────────────────────────────────────
        if wandb.run is not None:
            wandb.finish()
        mlflow.end_run()

        # ── build artifact ────────────────────────────────────────────────────
        artifact = self._build_artifact(
            transformation_artifact = transformation_artifact,
            result                  = result,
            total_params            = total_params,
            trainable_params        = trainable_params,
            mlflow_run_id           = mlflow_run_id,
            wandb_run_id            = wandb_run_id,
            wandb_run_url           = wandb_run_url,
            version_id              = version_id,
        )
        self._update_state(version_id, artifact)
        return artifact

    # ── load samples — notebook Cell 11 ───────────────────────────────────────
    def _load_samples(
        self, csv_path: Path, images_dir: Path
    ) -> List[Tuple[str, int]]:
        samples = []
        missing = 0
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in tqdm(reader, desc=f"Loading {csv_path.stem}"):
                filename   = row["Filename"]
                label      = int(row["Label"])
                image_path = str(images_dir / filename)
                if os.path.exists(image_path):
                    samples.append((image_path, label))
                else:
                    missing += 1
        logger.info(f"  Loaded : {len(samples)}  Missing : {missing}")
        return samples

    # ── class weights — notebook Cell 13 ─────────────────────────────────────
    def _compute_class_weights(
        self, train_samples: List[Tuple[str, int]]
    ) -> List[float]:
        """
        Formula: class_weights[c] = total / (NUM_CLASSES * count[c])
        Optionally raised to weight_exponent (1.0 = standard, 0.5 = softer).
        """
        train_labels  = [label for _, label in train_samples]
        label_counts  = Counter(train_labels)
        total_samples = len(train_labels)
        exp           = self.config.weight_exponent

        weights = []
        for c in range(NUM_CLASSES):
            w = total_samples / (NUM_CLASSES * label_counts[c])
            if exp != 1.0:
                w = w ** exp
            weights.append(w)

        logger.info("Class weights:")
        for c in range(NUM_CLASSES):
            logger.info(
                f"  {c} {SPECIES_MAP[c]:<18} : "
                f"count={label_counts[c]:>5}  weight={weights[c]:.4f}"
            )
        return weights

    # ── transforms — notebook Cell 15 ────────────────────────────────────────
    def _build_transforms(self):
        size = self.config.input_size  # 256
        train_tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=90),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomResizedCrop(size=size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        val_tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        return train_tf, val_tf

    # ── DataLoaders — notebook Cell 16 ───────────────────────────────────────
    def _build_loaders(
        self,
        train_samples : List[Tuple[str, int]],
        val_samples   : List[Tuple[str, int]],
        class_weights : List[float],
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        train_tf, val_tf = self._build_transforms()

        train_dataset   = DeepWeedDataset(train_samples, train_tf)
        sample_weights  = [class_weights[label] for label in train_dataset.labels]
        sampler         = WeightedRandomSampler(
            weights    = sample_weights,
            num_samples= len(sample_weights),
            replacement= True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size  = self.config.batch_size,
            sampler     = sampler,
            num_workers = self.config.num_workers,
            pin_memory  = self.config.pin_memory,
            drop_last   = self.config.drop_last,    # True — notebook Cell 16
        )
        logger.info(
            f"Train batches : {len(train_loader)}  "
            f"(batch={self.config.batch_size}  sampler=weighted  drop_last=True)"
        )

        val_loader = None
        if val_samples:
            val_dataset = DeepWeedDataset(val_samples, val_tf)
            val_loader  = DataLoader(
                val_dataset,
                batch_size  = self.config.batch_size,
                shuffle     = False,
                num_workers = self.config.num_workers,
                pin_memory  = self.config.pin_memory,
            )
            logger.info(f"Val batches   : {len(val_loader)}")
        return train_loader, val_loader

    # ── model — notebook Cell 18 ──────────────────────────────────────────────
    def _build_model(self):
        scaler = GradScaler("cuda") if self.config.mixed_precision else None

        model = timm.create_model(
            self.config.architecture,
            pretrained  = self.config.pretrained,
            num_classes = 0,
            global_pool = "avg",
        )
        num_features     = model.num_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=self.config.dropout_rate),
            nn.Linear(num_features, self.config.num_classes),
        )
        model = model.to(self.device)

        total_params     = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Architecture     : {self.config.architecture}")
        logger.info(f"Backbone features: {num_features}")
        logger.info(f"Total params     : {total_params:,}")
        logger.info(f"Trainable params : {trainable_params:,}")
        return model, scaler, total_params, trainable_params

    # ── loss — notebook Cell 22 ───────────────────────────────────────────────
    def _build_criterion(self, class_weights: List[float]) -> nn.Module:
        cw_tensor = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
        if self.config.use_focal_loss:
            criterion = FocalLoss(weight=cw_tensor, gamma=self.config.focal_gamma)
            logger.info(f"Loss : FocalLoss  gamma={self.config.focal_gamma}")
        else:
            criterion = nn.CrossEntropyLoss(
                weight          = cw_tensor,
                label_smoothing = self.config.label_smoothing,
            )
            logger.info(f"Loss : CrossEntropyLoss  smoothing={self.config.label_smoothing}")
        return criterion

    # ── optimizer — notebook Cell 22 ─────────────────────────────────────────
    def _build_optimizer(self, model: nn.Module) -> optim.Optimizer:
        optimizer = optim.AdamW(
            model.parameters(),
            lr           = self.config.learning_rate,
            weight_decay = self.config.weight_decay,
        )
        logger.info(
            f"Optimizer : AdamW  lr={self.config.learning_rate}  "
            f"wd={self.config.weight_decay}"
        )
        return optimizer

    # ── scheduler — notebook Cell 22 ─────────────────────────────────────────
    def _build_scheduler(self, optimizer: optim.Optimizer):
        effective_epochs = max(self.config.epochs - self.config.warmup_epochs, 1)
        if self.config.lr_scheduler == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=effective_epochs, eta_min=1e-6)
            logger.info(
                f"Scheduler : CosineAnnealingLR  T_max={effective_epochs}  eta_min=1e-6"
            )
        else:
            scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)
            logger.info("Scheduler : ReduceLROnPlateau")
        logger.info(f"Warmup    : {self.config.warmup_epochs} epochs (linear LR ramp)")
        return scheduler

    # ── MLflow setup — notebook Cell 23 / 24 ─────────────────────────────────
    def _setup_mlflow(
        self,
        run_name      : str,
        train_samples : List,
        val_samples   : List,
        total_params  : int,
        trainable_params: int,
        class_weights : List[float],
    ):
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        mlflow.set_experiment(self.config.mlflow_experiment_name)
        mlflow_run = mlflow.start_run(run_name=run_name)

        # log all params — notebook Cell 24
        mlflow.log_params({
            "architecture"    : self.config.architecture,
            "pretrained"      : self.config.pretrained,
            "num_classes"     : self.config.num_classes,
            "input_size"      : self.config.input_size,
            "dropout_rate"    : self.config.dropout_rate,
            "epochs"          : self.config.epochs,
            "batch_size"      : self.config.batch_size,
            "learning_rate"   : self.config.learning_rate,
            "weight_decay"    : self.config.weight_decay,
            "lr_scheduler"    : self.config.lr_scheduler,
            "warmup_epochs"   : self.config.warmup_epochs,
            "label_smoothing" : self.config.label_smoothing,
            "sampler"         : self.config.sampler,
            "drop_last"       : self.config.drop_last,
            "early_stop_pat"  : self.config.early_stopping_patience,
            "monitor_metric"  : self.config.monitor_metric,
            "grad_clip_norm"  : self.config.grad_clip_norm,
            "use_focal_loss"  : self.config.use_focal_loss,
            "focal_gamma"     : self.config.focal_gamma if self.config.use_focal_loss else None,
            "weight_exponent" : self.config.weight_exponent,
            "mixed_precision" : self.config.mixed_precision,
            "total_params"    : total_params,
            "trainable_params": trainable_params,
            "device"          : str(self.device),
            "num_train"       : len(train_samples),
            "num_val"         : len(val_samples),
        })
        mlflow.set_tags({
            "model_family": MLFLOW_MODEL_FAMILY,
            "dataset"     : MLFLOW_DATASET,
            "task"        : MLFLOW_TASK,
        })
        # per-class weight + count metrics — notebook Cell 25
        for c in range(NUM_CLASSES):
            mlflow.log_metric(f"class_weight_{SPECIES_MAP[c]}", class_weights[c])

        logger.info(f"MLflow run ID : {mlflow_run.info.run_id}")
        return mlflow_run

    # ── W&B setup — notebook Cell 25 ─────────────────────────────────────────
    def _setup_wandb(
        self,
        run_name      : str,
        model         : nn.Module,
        train_samples : List,
        val_samples   : List,
        class_weights : List[float],
    ):
        wandb_run = wandb.init(
            project = self.config.wandb_project,
            entity  = self.config.wandb_entity,
            name    = run_name,
            config  = {
                "architecture"    : self.config.architecture,
                "pretrained"      : self.config.pretrained,
                "num_classes"     : self.config.num_classes,
                "input_size"      : self.config.input_size,
                "dropout_rate"    : self.config.dropout_rate,
                "epochs"          : self.config.epochs,
                "batch_size"      : self.config.batch_size,
                "learning_rate"   : self.config.learning_rate,
                "weight_decay"    : self.config.weight_decay,
                "lr_scheduler"    : self.config.lr_scheduler,
                "warmup_epochs"   : self.config.warmup_epochs,
                "label_smoothing" : self.config.label_smoothing,
                "sampler"         : self.config.sampler,
                "drop_last"       : self.config.drop_last,
                "use_focal_loss"  : self.config.use_focal_loss,
                "focal_gamma"     : self.config.focal_gamma if self.config.use_focal_loss else None,
                "weight_exponent" : self.config.weight_exponent,
                "mixed_precision" : self.config.mixed_precision,
                "grad_clip_norm"  : self.config.grad_clip_norm,
                "num_train"       : len(train_samples),
                "num_val"         : len(val_samples),
            },
            tags = WANDB_TAGS + [self.config.architecture],
        )
        # gradient + weight histograms every 100 steps — notebook Cell 25
        wandb.watch(model, log="all", log_freq=100)

        # class weight table — notebook Cell 25
        cw_table = wandb.Table(
            columns = ["class_id", "species", "weight"],
            data    = [
                [c, SPECIES_MAP[c], round(class_weights[c], 4)]
                for c in range(NUM_CLASSES)
            ],
        )
        wandb.log({"class_weights": cw_table})
        logger.info(f"W&B run URL : {wandb_run.url}")
        return wandb_run

    @staticmethod
    def _plot_confusion_matrix(
        all_labels: List[int], all_preds: List[int], class_names: List[str], epoch: int
    ) -> plt.Figure:
        cm   = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
        cm_n = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-8)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm_n, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names, ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix — Epoch {epoch}")
        plt.tight_layout()
        return fig

    @staticmethod
    def _make_class_acc_table(epoch_cls_acc: Dict[int, float]) -> "wandb.Table":
        return wandb.Table(
            columns = ["class_id", "species", "val_acc"],
            data    = [
                [c, SPECIES_MAP[c], round(epoch_cls_acc.get(c, 0.0), 4)]
                for c in range(NUM_CLASSES)
            ],
        )

    # ── main training loop — notebook Cell 27 / 28 ───────────────────────────
    def _train_loop(
        self,
        model        : nn.Module,
        scaler       : Optional[GradScaler],
        criterion    : nn.Module,
        optimizer    : optim.Optimizer,
        scheduler,
        train_loader : DataLoader,
        val_loader   : Optional[DataLoader],
    ) -> Dict:
        cfg = self.config

        # state — notebook Cell 27
        history          : List[Dict] = []
        best_val_acc     = 0.0
        best_val_loss    = float("inf")
        best_epoch       = 0
        per_class_acc    : Dict[int, float] = {}
        no_improve_count = 0
        nan_batch_count  = 0
        training_start   = time.time()
        ckpt_heap        : List = []
        all_preds_list   : List[int] = []
        all_labels_list  : List[int] = []

        logger.info(f"\nStarting training — {cfg.epochs} epochs on {self.device}")
        logger.info("=" * 70)

        try:
            for epoch in range(1, cfg.epochs + 1):
                epoch_start      = time.time()
                epoch_nan_batches= 0

                # ── LR warmup — notebook Cell 28 ─────────────────────────────
                if epoch <= cfg.warmup_epochs:
                    warmup_lr = cfg.learning_rate * epoch / cfg.warmup_epochs
                    for pg in optimizer.param_groups:
                        pg["lr"] = warmup_lr

                # ══════════════════════════════════════════════════════════════
                # TRAIN
                # ══════════════════════════════════════════════════════════════
                model.train()
                train_loss_sum = 0.0
                train_correct  = 0
                train_total    = 0

                train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs} [Train]")
                for batch_idx, (images, labels) in enumerate(train_pbar):
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)

                    try:
                        if cfg.mixed_precision and scaler is not None:
                            with autocast("cuda"):
                                outputs = model(images)
                            # NaN FIX: cast float16 → float32 — notebook Cell 28
                            outputs = outputs.float()
                            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                                epoch_nan_batches += 1
                                nan_batch_count   += 1
                                optimizer.zero_grad(set_to_none=True)
                                continue
                            loss = criterion(outputs, labels)
                            scaler.scale(loss).backward()
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            outputs = model(images)
                            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                                epoch_nan_batches += 1
                                nan_batch_count   += 1
                                optimizer.zero_grad(set_to_none=True)
                                continue
                            loss = criterion(outputs, labels)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                            optimizer.step()

                        train_loss_sum += loss.item() * images.size(0)
                        preds           = outputs.argmax(dim=1)
                        train_correct  += (preds == labels).sum().item()
                        train_total    += images.size(0)
                        train_pbar.set_postfix({
                            "loss": f"{loss.item():.4f}",
                            "acc" : f"{(preds == labels).float().mean():.4f}",
                        })
                    except Exception as e:
                        logger.warning(f"Train batch {batch_idx} error: {e}")
                        optimizer.zero_grad(set_to_none=True)
                        continue

                train_loss = train_loss_sum / train_total if train_total > 0 else 0.0
                train_acc  = train_correct  / train_total if train_total > 0 else 0.0

                # ══════════════════════════════════════════════════════════════
                # VALIDATION
                # ══════════════════════════════════════════════════════════════
                val_loss, val_acc = 0.0, 0.0
                epoch_cls_acc: Dict[int, float] = {}
                all_preds_list, all_labels_list = [], []

                if val_loader is not None:
                    model.eval()
                    val_loss_sum   = 0.0
                    val_correct    = 0
                    val_total      = 0
                    class_correct  = defaultdict(int)
                    class_total_c  = defaultdict(int)
                    stable_batches = 0

                    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{cfg.epochs} [Val]")
                    with torch.no_grad():
                        for batch_idx, (images, labels) in enumerate(val_pbar):
                            images = images.to(self.device, non_blocking=True)
                            labels = labels.to(self.device, non_blocking=True)
                            try:
                                if cfg.mixed_precision:
                                    with autocast("cuda"):
                                        outputs = model(images)
                                    outputs = outputs.float()   # NaN FIX
                                else:
                                    outputs = model(images)

                                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                                    epoch_nan_batches += 1
                                    nan_batch_count   += 1
                                    continue

                                loss = criterion(outputs, labels)
                                if torch.isnan(loss).any() or torch.isinf(loss).any():
                                    epoch_nan_batches += 1
                                    nan_batch_count   += 1
                                    continue

                                val_loss_sum  += loss.item() * images.size(0)
                                preds          = outputs.argmax(dim=1)
                                val_correct   += (preds == labels).sum().item()
                                val_total     += images.size(0)
                                stable_batches += 1

                                all_preds_list.extend(preds.cpu().tolist())
                                all_labels_list.extend(labels.cpu().tolist())

                                for lbl, pred in zip(labels.cpu().tolist(), preds.cpu().tolist()):
                                    class_total_c[lbl] += 1
                                    class_correct[lbl] += int(lbl == pred)

                                val_pbar.set_postfix({
                                    "loss": f"{loss.item():.4f}",
                                    "acc" : f"{(preds == labels).float().mean():.4f}",
                                })
                            except Exception as e:
                                logger.warning(f"Val batch {batch_idx} error: {e}")
                                continue

                    if val_total > 0:
                        val_loss = val_loss_sum / val_total
                        val_acc  = val_correct  / val_total
                    else:
                        logger.warning("⚠️  No stable val batches")

                    epoch_cls_acc = {
                        cls: round(class_correct[cls] / class_total_c[cls], 4)
                        if class_total_c[cls] > 0 else 0.0
                        for cls in range(NUM_CLASSES)
                    }

                # ── scheduler step ────────────────────────────────────────────
                if epoch > cfg.warmup_epochs:
                    if isinstance(scheduler, ReduceLROnPlateau):
                        scheduler.step(val_acc)
                    else:
                        scheduler.step()

                current_lr = optimizer.param_groups[0]["lr"]
                epoch_time = time.time() - epoch_start

                logger.info(
                    f"\nEpoch [{epoch:>3}/{cfg.epochs}]  "
                    f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                    f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
                    f"lr={current_lr:.6f}  time={epoch_time:.1f}s  "
                    f"nan_batches={epoch_nan_batches}"
                )

                # ── W&B logging — notebook Cell 28 ───────────────────────────
                try:
                    log_dict = {
                        "epoch"                      : epoch,
                        "train/loss"                 : train_loss,
                        "train/acc"                  : train_acc,
                        "val/loss"                   : val_loss,
                        "val/acc"                    : val_acc,
                        "lr"                         : current_lr,
                        "epoch_time_s"               : epoch_time,
                        "debug/nan_batches_epoch"    : epoch_nan_batches,
                        "debug/nan_batches_total"    : nan_batch_count,
                        **{f"val/cls_acc/{SPECIES_MAP[c]}": acc
                           for c, acc in epoch_cls_acc.items()},
                        "val/per_class_table"        : self._make_class_acc_table(epoch_cls_acc),
                    }
                    if (epoch % cfg.cm_log_interval == 0 or epoch == cfg.epochs) and all_labels_list:
                        cm_fig = self._plot_confusion_matrix(
                            all_labels_list, all_preds_list, CLASS_NAMES, epoch
                        )
                        log_dict["val/confusion_matrix"] = wandb.Image(cm_fig)
                        plt.close(cm_fig)
                    wandb.log(log_dict)
                except Exception as e:
                    logger.warning(f"W&B logging failed — {e}")

                # ── MLflow logging — notebook Cell 28 ────────────────────────
                try:
                    mlflow.log_metrics({
                        "train_loss"        : train_loss,
                        "train_acc"         : train_acc,
                        "val_loss"          : val_loss,
                        "val_acc"           : val_acc,
                        "lr"                : current_lr,
                        "epoch_time_s"      : epoch_time,
                        "nan_batches_epoch" : epoch_nan_batches,
                        "nan_batches_total" : nan_batch_count,
                        **{f"cls_acc_{SPECIES_MAP[c]}": acc
                           for c, acc in epoch_cls_acc.items()},
                    }, step=epoch)
                    if (epoch % cfg.cm_log_interval == 0 or epoch == cfg.epochs) and all_labels_list:
                        cm_fig   = self._plot_confusion_matrix(
                            all_labels_list, all_preds_list, CLASS_NAMES, epoch
                        )
                        cm_path  = f"confusion_matrix_epoch_{epoch:03d}.png"
                        cm_fig.savefig(cm_path, dpi=100, bbox_inches="tight")
                        mlflow.log_artifact(cm_path)
                        plt.close(cm_fig)
                        os.remove(cm_path)
                except Exception as e:
                    logger.warning(f"MLflow logging failed — {e}")

                # ── best model tracking — notebook Cell 28 ───────────────────
                is_best = val_acc > best_val_acc
                if is_best:
                    best_val_acc      = val_acc
                    best_val_loss     = val_loss
                    best_epoch        = epoch
                    per_class_acc     = epoch_cls_acc
                    no_improve_count  = 0
                    torch.save(model.state_dict(), cfg.best_model_path)

                    try:
                        wb_artifact = wandb.Artifact(
                            name     = "best-model",
                            type     = "model",
                            metadata = {"val_acc": best_val_acc, "epoch": best_epoch},
                        )
                        wb_artifact.add_file(str(cfg.best_model_path))
                        wandb.log_artifact(wb_artifact)
                        wandb.log({
                            "best/val_acc" : best_val_acc,
                            "best/val_loss": best_val_loss,
                            "best/epoch"   : best_epoch,
                        })
                    except Exception as e:
                        logger.warning(f"W&B artifact logging failed — {e}")

                    try:
                        mlflow.log_metrics({
                            "best_val_acc" : best_val_acc,
                            "best_val_loss": best_val_loss,
                            "best_epoch"   : float(best_epoch),
                        }, step=epoch)
                        mlflow.log_artifact(str(cfg.best_model_path))
                    except Exception as e:
                        logger.warning(f"MLflow best-model logging failed — {e}")

                    logger.info(f"⭐ New best!  val_acc={best_val_acc:.4f}")
                else:
                    no_improve_count += 1

                # ── top-K checkpoints — notebook Cell 28 ─────────────────────
                ckpt_path = cfg.checkpoints_dir / f"epoch_{epoch:03d}_acc_{val_acc:.4f}.pth"
                torch.save(model.state_dict(), ckpt_path)
                heapq.heappush(ckpt_heap, (val_acc, str(ckpt_path)))
                if len(ckpt_heap) > cfg.save_top_k:
                    _, worst_path = heapq.heappop(ckpt_heap)
                    worst = Path(worst_path)
                    if worst.exists():
                        worst.unlink()

                # ── history — notebook Cell 28 ────────────────────────────────
                history.append({
                    "epoch"        : epoch,
                    "train_loss"   : train_loss,
                    "train_acc"    : train_acc,
                    "val_loss"     : val_loss,
                    "val_acc"      : val_acc,
                    "lr"           : current_lr,
                    "epoch_time_s" : epoch_time,
                    "nan_batches"  : epoch_nan_batches,
                })
                with open(cfg.training_history_path, "w") as f:
                    import json; json.dump(history, f, indent=2)

                # ── early stopping — notebook Cell 28 ────────────────────────
                if no_improve_count >= cfg.early_stopping_patience:
                    logger.info(
                        f"\n🛑 Early stopping at epoch {epoch} "
                        f"(no improvement for {cfg.early_stopping_patience} epochs)"
                    )
                    try:
                        mlflow.set_tag("stopped_early", "True")
                        mlflow.log_metric("stopped_at_epoch", float(epoch))
                    except Exception:
                        pass
                    break

        except KeyboardInterrupt:
            logger.info("🛑 Training interrupted by user")
            interrupted_path = cfg.checkpoints_dir / f"interrupted_epoch_{epoch}.pth"
            torch.save(model.state_dict(), interrupted_path)

        finally:
            torch.save(model.state_dict(), cfg.final_model_path)
            total_time = time.time() - training_start
            logger.info("=" * 70)
            logger.info(f"Training complete : {len(history)} epochs")
            logger.info(f"Total time        : {total_time / 60:.1f} min")
            logger.info(f"Best val_acc      : {best_val_acc:.4f} at epoch {best_epoch}")
            logger.info(f"Total NaN batches : {nan_batch_count}")

            # ── final classification report — notebook Cell 28 ────────────────
            if all_labels_list and all_preds_list:
                report = classification_report(
                    all_labels_list, all_preds_list,
                    target_names=CLASS_NAMES, output_dict=True
                )
                logger.info("\nFinal classification report (val):")
                logger.info(classification_report(
                    all_labels_list, all_preds_list, target_names=CLASS_NAMES
                ))
                for cls_name in CLASS_NAMES:
                    if cls_name in report:
                        try:
                            mlflow.log_metric(f"final_f1_{cls_name}",        report[cls_name]["f1-score"])
                            mlflow.log_metric(f"final_precision_{cls_name}", report[cls_name]["precision"])
                            mlflow.log_metric(f"final_recall_{cls_name}",    report[cls_name]["recall"])
                        except Exception:
                            pass
                try:
                    wandb.log({
                        "final/macro_f1"         : report["macro avg"]["f1-score"],
                        "final/weighted_f1"      : report["weighted avg"]["f1-score"],
                        "final/best_val_acc"     : best_val_acc,
                        "final/best_epoch"       : best_epoch,
                        "final/total_nan_batches": nan_batch_count,
                    })
                except Exception:
                    pass

            # ── final MLflow summary — notebook Cell 28 ───────────────────────
            try:
                mlflow.log_metrics({
                    "final_train_loss"       : history[-1]["train_loss"] if history else 0,
                    "final_val_acc"          : history[-1]["val_acc"]    if history else 0,
                    "total_training_time_min": total_time / 60,
                    "total_epochs_completed" : float(len(history)),
                    "total_nan_batches"      : float(nan_batch_count),
                })
                mlflow.log_artifact(str(cfg.training_history_path))
                mlflow.log_artifact(str(cfg.final_model_path))
            except Exception as e:
                logger.warning(f"Final MLflow logging failed — {e}")

        return {
            "history"            : history,
            "best_val_acc"       : best_val_acc,
            "best_val_loss"      : best_val_loss,
            "best_epoch"         : best_epoch,
            "per_class_val_acc"  : per_class_acc,
            "nan_batches_total"  : nan_batch_count,
            "total_time_s"       : time.time() - training_start,
            "final_train_loss"   : history[-1]["train_loss"] if history else 0.0,
            "final_train_acc"    : history[-1]["train_acc"]  if history else 0.0,
            "epochs_trained"     : len(history),
        }

    # ── build artifact ────────────────────────────────────────────────────────
    def _build_artifact(
        self,
        transformation_artifact : DataTransformationArtifact,
        result                  : Dict,
        total_params            : int,
        trainable_params        : int,
        mlflow_run_id           : str,
        wandb_run_id            : str,
        wandb_run_url           : str,
        version_id              : str,
    ) -> ModelTrainerArtifact:
        cfg          = self.config
        trained_at   = datetime.now()
        cuda_version = torch.version.cuda if torch.cuda.is_available() else None

        artifact = ModelTrainerArtifact(
            transformation_artifact  = transformation_artifact,
            best_model_path          = cfg.best_model_path,
            final_model_path         = cfg.final_model_path,
            checkpoints_dir          = cfg.checkpoints_dir,
            architecture             = cfg.architecture,
            num_classes              = cfg.num_classes,
            pretrained               = cfg.pretrained,
            dropout_rate             = cfg.dropout_rate,
            total_params             = total_params,
            trainable_params         = trainable_params,
            epochs_trained           = result["epochs_trained"],
            best_epoch               = result["best_epoch"],
            best_val_acc             = result["best_val_acc"],
            best_val_loss            = result["best_val_loss"],
            final_train_acc          = result["final_train_acc"],
            final_train_loss         = result["final_train_loss"],
            learning_rate            = cfg.learning_rate,
            weight_decay             = cfg.weight_decay,
            lr_scheduler             = cfg.lr_scheduler,
            warmup_epochs            = cfg.warmup_epochs,
            early_stopping_patience  = cfg.early_stopping_patience,
            monitor_metric           = cfg.monitor_metric,
            batch_size               = cfg.batch_size,
            input_size               = cfg.input_size,
            sampler                  = cfg.sampler,
            drop_last                = cfg.drop_last,
            weight_exponent          = cfg.weight_exponent,
            grad_clip_norm           = cfg.grad_clip_norm,
            use_focal_loss           = cfg.use_focal_loss,
            focal_gamma              = cfg.focal_gamma,
            label_smoothing          = cfg.label_smoothing,
            mixed_precision          = cfg.mixed_precision,
            nan_batches_total        = result["nan_batches_total"],
            per_class_val_acc        = {str(k): v for k, v in result["per_class_val_acc"].items()},
            mlflow_run_id            = mlflow_run_id,
            wandb_run_id             = wandb_run_id,
            wandb_run_url            = wandb_run_url,
            mlflow_tracking_uri      = cfg.mlflow_tracking_uri,
            mlflow_experiment_name   = cfg.mlflow_experiment_name,
            device                   = str(self.device),
            cuda_version             = cuda_version,
            training_history_path    = cfg.training_history_path,
            trained_at               = trained_at,
            total_training_time_s    = result["total_time_s"],
            artifact_path            = cfg.artifact_path,
        )

        save_json(path=cfg.artifact_path, data={
            "version_id"             : version_id,
            "architecture"           : artifact.architecture,
            "num_classes"            : artifact.num_classes,
            "pretrained"             : artifact.pretrained,
            "dropout_rate"           : artifact.dropout_rate,
            "input_size"             : artifact.input_size,
            "total_params"           : artifact.total_params,
            "trainable_params"       : artifact.trainable_params,
            "epochs_trained"         : artifact.epochs_trained,
            "best_epoch"             : artifact.best_epoch,
            "best_val_acc"           : artifact.best_val_acc,
            "best_val_loss"          : artifact.best_val_loss,
            "final_train_acc"        : artifact.final_train_acc,
            "final_train_loss"       : artifact.final_train_loss,
            "learning_rate"          : artifact.learning_rate,
            "weight_decay"           : artifact.weight_decay,
            "lr_scheduler"           : artifact.lr_scheduler,
            "warmup_epochs"          : artifact.warmup_epochs,
            "early_stopping_patience": artifact.early_stopping_patience,
            "monitor_metric"         : artifact.monitor_metric,
            "batch_size"             : artifact.batch_size,
            "input_size"             : artifact.input_size,
            "sampler"                : artifact.sampler,
            "drop_last"              : artifact.drop_last,
            "weight_exponent"        : artifact.weight_exponent,
            "grad_clip_norm"         : artifact.grad_clip_norm,
            "use_focal_loss"         : artifact.use_focal_loss,
            "focal_gamma"            : artifact.focal_gamma,
            "label_smoothing"        : artifact.label_smoothing,
            "mixed_precision"        : artifact.mixed_precision,
            "nan_batches_total"      : artifact.nan_batches_total,
            "per_class_val_acc"      : artifact.per_class_val_acc,
            "mlflow_run_id"          : artifact.mlflow_run_id,
            "wandb_run_id"           : artifact.wandb_run_id,
            "wandb_run_url"          : artifact.wandb_run_url,
            "mlflow_tracking_uri"    : artifact.mlflow_tracking_uri,
            "mlflow_experiment_name" : artifact.mlflow_experiment_name,
            "device"                 : artifact.device,
            "cuda_version"           : artifact.cuda_version,
            "best_model_path"        : str(artifact.best_model_path),
            "final_model_path"       : str(artifact.final_model_path),
            "checkpoints_dir"        : str(artifact.checkpoints_dir),
            "training_history_path"  : str(artifact.training_history_path),
            "trained_at"             : artifact.trained_at.isoformat(),
            "total_training_time_s"  : artifact.total_training_time_s,
            "artifact_path"          : str(artifact.artifact_path),
        })
        logger.info(f"💾 ModelTrainerArtifact saved : {cfg.artifact_path}")
        return artifact

    # ── version control ───────────────────────────────────────────────────────
    def _already_trained(self, version_id: str) -> bool:
        state_path = self.config.trainer_state_path
        if not state_path.exists():
            return False
        try:
            return load_json(state_path).get("last_version_id") == version_id
        except Exception:
            return False

    def _update_state(self, version_id: str, artifact: ModelTrainerArtifact) -> None:
        save_json(path=self.config.trainer_state_path, data={
            "last_version_id"      : version_id,
            "last_trained_at"      : artifact.trained_at.isoformat(),
            "last_best_val_acc"    : artifact.best_val_acc,
            "last_best_epoch"      : artifact.best_epoch,
            "last_mlflow_run_id"   : artifact.mlflow_run_id,
            "last_wandb_run_id"    : artifact.wandb_run_id,
            "last_artifact_path"   : str(artifact.artifact_path),
        })
        logger.info(f"💾 Trainer state saved : {self.config.trainer_state_path}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    from weed_detection.components.data_transformation import DataTransformation
    logger.info("=" * 70)
    logger.info("🚀 Starting Model Trainer")
    logger.info("=" * 70)

    config_manager          = ConfigurationManager()
    trainer_config          = config_manager.get_model_trainer_config()
    transformation_config   = config_manager.get_data_transformation_config()

    # load transformation artifact from disk
    transformation = DataTransformation(transformation_config)
    ta = transformation.load_artifact()

    trainer  = ModelTrainer(trainer_config)
    artifact = trainer.run(ta)

    if artifact is None:
        logger.info("✅ Nothing to do — already trained (same version)")
    else:
        logger.info(f"✅ Training complete")
        logger.info(f"   Best val acc   : {artifact.best_val_acc:.4f} (epoch {artifact.best_epoch})")
        logger.info(f"   MLflow run ID  : {artifact.mlflow_run_id}")
        logger.info(f"   W&B URL        : {artifact.wandb_run_url}")
    return artifact


if __name__ == "__main__":
    main()