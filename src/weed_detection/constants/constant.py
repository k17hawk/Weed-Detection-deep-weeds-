from pathlib import Path

# ── Config file paths ─────────────────────────────────────────────────────────
CONFIG_FILE_PATH : Path = Path("configs/config.yaml")
PARAMS_FILE_PATH : Path = Path("params.yaml")

# ── Species map — notebook Cell 7 ────────────────────────────────────────────
SPECIES_MAP: dict = {
    0: "Chinee Apple",
    1: "Lantana",
    2: "Parkinsonia",
    3: "Parthenium",
    4: "Prickly acacia",
    5: "Rubber Vine",
    6: "Siam weed",
    7: "Snake weed",
    8: "Negative",
}
NUM_CLASSES : int  = 9
CLASS_NAMES : list = [SPECIES_MAP[i] for i in range(NUM_CLASSES)]

# ── ImageNet normalisation — notebook Cell 6 ─────────────────────────────────
IMAGENET_MEAN : list = [0.485, 0.456, 0.406]
IMAGENET_STD  : list = [0.229, 0.224, 0.225]

# ── CUDA memory optimisation — notebook Cell 2 ───────────────────────────────
PYTORCH_CUDA_ALLOC_CONF : str = "expandable_segments:True"

# ── MLflow / W&B tags — notebook Cell 24 ─────────────────────────────────────
MLFLOW_MODEL_FAMILY : str = "efficientnet"
MLFLOW_DATASET      : str = "deepweeds"
MLFLOW_TASK         : str = "multiclass_classification"
WANDB_TAGS          : list = ["training", "deepweeds"]

# ── MLflow registered model name (used in model_evaluation.py) ───────────────
MLFLOW_REGISTERED_MODEL_NAME : str = "DeepWeedClassifier"