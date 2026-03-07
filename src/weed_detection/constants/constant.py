# weed_detection/constants/constant.py
# ─────────────────────────────────────────────────────────────────────────────
# Two things live here:
#   1. Secrets loaded from .env  (never hardcoded, never in config.yaml)
#   2. Static constants shared across the project (paths, regex patterns)
#
# MLOps best practice:
#   - config.yaml  → paths, infra settings  (committed to git)
#   - constant.py  → reads secrets from .env (never committed)
#   - .env         → actual secret values    (in .gitignore)
# ─────────────────────────────────────────────────────────────────────────────

import os
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── AWS credentials (from .env) ───────────────────────────────────────────────
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION     = os.getenv("REGION", "us-west-1")

# ── App config (from .env) ────────────────────────────────────────────────────
DATA_SOURCE    = os.getenv("DATA_SOURCE", "unknown_source")
QUEUE_URL      = os.getenv("QUEUE_URL")
KAFKA_BROKER   = os.getenv("KAFKA_BROKER", "localhost:9092")
KAFKA_TOPIC    = os.getenv("KAFKA_TOPIC",  "drone-images-topic")

# ── Config file paths ─────────────────────────────────────────────────────────
CONFIG_FILE_PATH = Path("configs/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")

# ── Filename validation pattern ───────────────────────────────────────────────
# Data engineers MUST name uploaded zips: drone_YYYYMMDD_HHMMSS.zip
# Example: drone_20260307_201234.zip
FILE_PATTERN = re.compile(
    r"^drone_"
    r"(?P<year>\d{4})(?P<month>0[1-9]|1[0-2])(?P<day>0[1-9]|[12]\d|3[01])"
    r"_"
    r"(?P<hour>[01]\d|2[0-3])(?P<minute>[0-5]\d)(?P<second>[0-5]\d)"
    r"\.zip$"
)