import os
import re
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict
load_dotenv()

# ── AWS credentials ───────────────────────────────────────────────────────────
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION     = os.getenv("REGION", "us-west-1")

# ── App config ────────────────────────────────────────────────────────────────
DATA_SOURCE  = os.getenv("DATA_SOURCE", "unknown_source")
QUEUE_URL    = os.getenv("QUEUE_URL")
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
KAFKA_TOPIC  = os.getenv("KAFKA_TOPIC",  "drone-images-topic")

# ── Config file paths ─────────────────────────────────────────────────────────
CONFIG_FILE_PATH = Path("configs/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")

# ── Filename validation pattern ───────────────────────────────────────────────
# Required format: drone_YYYYMMDD_HHMMSS.zip
# Example        : drone_20260307_201234.zip
FILE_PATTERN = re.compile(
    r"^drone_"
    r"(?P<year>\d{4})(?P<month>0[1-9]|1[0-2])(?P<day>0[1-9]|[12]\d|3[01])"
    r"_"
    r"(?P<hour>[01]\d|2[0-3])(?P<minute>[0-5]\d)(?P<second>[0-5]\d)"
    r"\.zip$"
)

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
