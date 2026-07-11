import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

REDIS_BROKER_URL = os.getenv("REDIS_BROKER_URL", "redis://localhost:6379/0")
REDIS_BACKEND_URL = os.getenv("REDIS_BACKEND_URL", "redis://localhost:6379/1")
