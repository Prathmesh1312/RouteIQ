import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

# Detect and prioritize local virtualenv site-packages (for llama_cpp, Metal wheels, etc.)
for vp in [
    PROJECT_DIR.parent / "venv" / "lib" / "python3.13" / "site-packages",
    PROJECT_DIR / "venv" / "lib" / "python3.13" / "site-packages",
    BASE_DIR / "venv" / "lib" / "python3.13" / "site-packages",
]:
    if vp.exists() and str(vp) not in sys.path:
        sys.path.insert(0, str(vp))

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_REGISTRY_PATH = APP_DIR / "models" / "registry.json"
DB_PATH = DATA_DIR / "routeiq_audit.db"
ESTIMATOR_MODEL_PATH = DATA_DIR / "estimator_tfidf.joblib"
PROMPTS_PATH = DATA_DIR / "prompts_300.json"
BENCHMARK_PATH = DATA_DIR / "benchmark_30.json"
MODEL_DIR = Path(os.environ.get("MODEL_DIR", str(PROJECT_DIR / "model")))

# Auto-load .env if present
env_file = PROJECT_DIR / ".env"
if env_file.exists():
    try:
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k and k not in os.environ:
                        os.environ[k] = v
    except Exception:
        pass

# Hard Rule & Circuit Breaker Thresholds
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 3
CIRCUIT_BREAKER_COOLDOWN_SECONDS = 15.0
ESTIMATOR_CONFIDENCE_THRESHOLD = 0.70

# Independent Auditor Tolerances
AUDITOR_LATENCY_TOLERANCE_MS = 5.0
AUDITOR_COST_TOLERANCE_RATIO = 1.01

# System 1 / Laya Decision Engine Mapping
TIER_CAPABILITY_MAP = {
    "easy": 0.30,
    "medium": 0.65,
    "hard": 0.80
}

# Free / Low-Cost Online API Providers
GROQ_BASE_URL = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
GEMINI_BASE_URL = os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
