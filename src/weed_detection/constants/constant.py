import os
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()



ACCESS_KEY_ID = os.getenv("ACCESS_KEY_ID")
SECRET_ACCESS_KEY = os.getenv("SECRET_ACCESS_KEY")
DATA_SOURCE = os.getenv("DATA_SOURCE")
REGION =os.getenv("REGION")
CONFIG_FILE_PATH = Path("configs/config.yaml")

PARAMS_FILE_PATH = Path("params.yaml")



