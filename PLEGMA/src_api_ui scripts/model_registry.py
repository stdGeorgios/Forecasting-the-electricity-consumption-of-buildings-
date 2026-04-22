# C:/Plegma_Programming/src/model_registry.py
# ---------------------------------------------------------
# Model Registry for PLEGMA API
# - Central place to declare available model families and artifact dirs
# - Each family supports coldstart & withhistory
# ---------------------------------------------------------

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any

# =========================
# Base path
# =========================
BASE_DIR = Path(os.getenv("PLEGMA_BASE_DIR", "C:/Plegma_Programming"))

# =========================
# Model Registry
# =========================
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "rf": {
        "name": "Random Forest (PLEGMA)",
        "artifact_dir": str(BASE_DIR / "models" / "final_rf"),
        "type": "sklearn_rf",
        "supports": ["coldstart", "withhistory"],
    },
    "xgb": {
        "name": "XGBoost (PLEGMA)",
        "artifact_dir": str(BASE_DIR / "models" / "final_xgb"),
        "type": "xgboost",
        "supports": ["coldstart", "withhistory"],
    },
    "lgbm": {
        "name": "LightGBM (PLEGMA)",
        "artifact_dir": str(BASE_DIR / "models" / "final_lgbm"),
        "type": "lightgbm",
        "supports": ["coldstart", "withhistory"],
    },
}

# =========================
# Defaults
# =========================
DEFAULT_MODEL_ID = "xgb"
DEFAULT_MODE = "coldstart"


# =========================
# Helpers
# =========================
def get_model_info(model_id: str) -> Dict[str, Any]:
    if model_id not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model_id: {model_id}")
    return MODEL_REGISTRY[model_id]


def get_artifact_dir(model_id: str) -> Path:
    info = get_model_info(model_id)
    return Path(info["artifact_dir"])


def list_models() -> Dict[str, str]:
    """For UI dropdowns"""
    return {k: v["name"] for k, v in MODEL_REGISTRY.items()}


def validate_model_mode(model_id: str, mode: str) -> None:
    info = get_model_info(model_id)
    if mode not in info["supports"]:
        raise ValueError(
            f"Model '{model_id}' does not support mode '{mode}'. "
            f"Supported: {info['supports']}"
        )