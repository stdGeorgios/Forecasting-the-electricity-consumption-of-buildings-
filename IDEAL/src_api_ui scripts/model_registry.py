# C:/IDEAL_Programming/src/model_registry.py
# ---------------------------------------------------------
# Model Registry for IDEAL API
# - Central place to declare available model families and artifact dirs
# - Each family can support one or more prediction modes
# ---------------------------------------------------------

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any

BASE_DIR = Path(os.getenv("IDEAL_BASE_DIR", "C:/IDEAL_Programming"))

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "rf": {
        "name": "Random Forest (final)",
        "artifact_dir": str(BASE_DIR / "processed" / "models" / "final_rf"),
        "type": "sklearn_rf",
        "supports": ["coldstart", "withhistory"],
    },
    "xgb": {
        "name": "XGBoost (final)",
        "artifact_dir": str(BASE_DIR / "processed" / "models" / "final_xgb"),
        "type": "xgboost",
        "supports": ["coldstart", "withhistory"],
    },
    "lgbm": {
        "name": "LightGBM (final)",
        "artifact_dir": str(BASE_DIR / "processed" / "models" / "final_lgbm"),
        "type": "lightgbm",
        "supports": ["coldstart", "withhistory"],
    },
}

DEFAULT_MODEL_ID = "rf"
DEFAULT_MODE = "coldstart"
