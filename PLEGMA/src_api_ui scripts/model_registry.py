# C:/Plegma_Programming/src/model_registry.py
# ---------------------------------------------------------
# Model Registry for PLEGMA API/UI - Generic user version
# ---------------------------------------------------------
#
# Final UI assumptions:
#   - The user does NOT provide home_id.
#   - No model selected by the UI/API requires or uses home_id as a feature.
#   - No-history models use only temporal, environmental and static household features.
#   - With-history models use recent user-provided consumption history through
#     lag/rolling features, but do not require or use home_id.
#
# Final PLEGMA artifact folders expected:
#
#   processed/models/final_api_models/RF/
#     ├── no_history/
#     └── with_history/
#
#   processed/models/final_api_models/LGBM/
#     ├── no_history_simple/
#     └── with_history_generic/
#
#   processed/models/final_api_models/XGB/
#     ├── no_history_simple/
#     └── with_history_generic/
#
# Notes:
#   - Canonical modes are "no_history" and "with_history".
#   - Legacy aliases such as "coldstart", "cold_start" and "withhistory"
#     are still accepted.
#   - Legacy model ids such as "rf", "lgbm" and "xgb" are resolved by mode.
# ---------------------------------------------------------

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(os.getenv("PLEGMA_BASE_DIR", "C:/Plegma_Programming"))
MODELS_ROOT = BASE_DIR / "processed" / "models" / "final_api_models"

RF_ROOT = MODELS_ROOT / "RF"
LGBM_ROOT = MODELS_ROOT / "LGBM"
XGB_ROOT = MODELS_ROOT / "XGB"


# ============================================================
# Standard API modes
# ============================================================

MODE_NO_HISTORY = "no_history"
MODE_WITH_HISTORY = "with_history"

# Backward-compatible aliases. Keep these constants so older imports do not break.
MODE_COLDSTART = MODE_NO_HISTORY
MODE_WITHHISTORY = MODE_WITH_HISTORY

OPT_BALANCED = "balanced"
OPT_DAILY = "daily"

DEFAULT_MODE = MODE_NO_HISTORY
DEFAULT_OPTIMIZATION = OPT_BALANCED
DEFAULT_HISTORY_HOURS = 168


# ============================================================
# Model registry
# ============================================================

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # --------------------------------------------------------
    # FINAL DEFAULT: NO-HISTORY
    # --------------------------------------------------------
    "lgbm_no_history_default": {
        "name": "LightGBM - no-history default",
        "short_name": "LGBM no-history default",
        "model_family": "LightGBM",
        "type": "lightgbm",
        "mode": MODE_NO_HISTORY,
        "optimization": OPT_BALANCED,
        "artifact_dir": str(LGBM_ROOT / "no_history_simple"),
        "is_default": True,
        "is_optional": False,
        "ui_visible": True,
        "supports": [MODE_NO_HISTORY],
        "requires_history": False,
        "requires_user_home_id": False,
        "uses_home_id_as_feature": False,
        "uses_lag_features": False,
        "uses_rolling_features": False,
        "uses_home_stats": False,
        "uses_consumption_regime": False,
        "uses_behavior_profiles": False,
        "uses_knn_profiles": False,
        "description": (
            "Final generic no-history LightGBM model. Uses temporal, environmental and static household "
            "features only. No home_id, no lag/history features, no behavior profiles and no KNN profiles."
        ),
    },

    # --------------------------------------------------------
    # FINAL DEFAULT: WITH-HISTORY
    # --------------------------------------------------------
    "lgbm_with_history_default": {
        "name": "LightGBM - with-history default",
        "short_name": "LGBM with-history default",
        "model_family": "LightGBM",
        "type": "lightgbm",
        "mode": MODE_WITH_HISTORY,
        "optimization": OPT_BALANCED,
        "artifact_dir": str(LGBM_ROOT / "with_history_generic"),
        "is_default": True,
        "is_optional": False,
        "ui_visible": True,
        "supports": [MODE_WITH_HISTORY],
        "requires_history": True,
        "min_history_hours": DEFAULT_HISTORY_HOURS,
        "requires_user_home_id": False,
        "uses_home_id_as_feature": False,
        "uses_lag_features": True,
        "uses_rolling_features": True,
        "uses_home_stats": False,
        "uses_consumption_regime": False,
        "uses_behavior_profiles": False,
        "uses_knn_profiles": False,
        "has_optional_daily_prediction": True,
        "default_prediction_column": "prediction_Wh",
        "optional_daily_prediction_column": "prediction_daily_calibrated_Wh",
        "description": (
            "Final generic with-history LightGBM model. Uses recent user-provided consumption history "
            "through lag and rolling features, but does not require or use home_id. The balanced prediction "
            "is the default; daily-calibrated output may exist as optional post-processing."
        ),
    },

    # --------------------------------------------------------
    # OPTIONAL WITH-HISTORY COMPARISON MODELS
    # --------------------------------------------------------
    "xgb_with_history_optional": {
        "name": "XGBoost - with-history optional",
        "short_name": "XGB with-history optional",
        "model_family": "XGBoost",
        "type": "xgboost",
        "mode": MODE_WITH_HISTORY,
        "optimization": OPT_BALANCED,
        "artifact_dir": str(XGB_ROOT / "with_history_generic"),
        "is_default": False,
        "is_optional": True,
        "ui_visible": True,
        "supports": [MODE_WITH_HISTORY],
        "requires_history": True,
        "min_history_hours": DEFAULT_HISTORY_HOURS,
        "requires_user_home_id": False,
        "uses_home_id_as_feature": False,
        "uses_lag_features": True,
        "uses_rolling_features": True,
        "uses_home_stats": False,
        "uses_consumption_regime": False,
        "uses_behavior_profiles": False,
        "uses_knn_profiles": False,
        "has_optional_daily_prediction": True,
        "default_prediction_column": "prediction_Wh",
        "optional_daily_prediction_column": "prediction_daily_calibrated_Wh",
        "description": (
            "Optional generic with-history XGBoost model. Uses lag/rolling history features and does not require home_id."
        ),
    },

    "rf_with_history_optional": {
        "name": "Random Forest - with-history optional",
        "short_name": "RF with-history optional",
        "model_family": "Random Forest",
        "type": "sklearn_rf",
        "mode": MODE_WITH_HISTORY,
        "optimization": OPT_BALANCED,
        "artifact_dir": str(RF_ROOT / "with_history"),
        "is_default": False,
        "is_optional": True,
        "ui_visible": True,
        "supports": [MODE_WITH_HISTORY],
        "requires_history": True,
        "min_history_hours": DEFAULT_HISTORY_HOURS,
        "requires_user_home_id": False,
        "uses_home_id_as_feature": False,
        "uses_lag_features": True,
        "uses_rolling_features": True,
        "uses_home_stats": False,
        "uses_consumption_regime": False,
        "uses_behavior_profiles": False,
        "uses_knn_profiles": False,
        "default_prediction_column": "prediction_Wh",
        "description": (
            "Optional generic with-history Random Forest model. The training script uses home_id only internally "
            "to construct lag/rolling features per home during dataset preparation; home_id is not a model feature "
            "and is not required by the user/API."
        ),
    },

    # --------------------------------------------------------
    # OPTIONAL NO-HISTORY COMPARISON MODELS
    # --------------------------------------------------------
    "xgb_no_history_optional": {
        "name": "XGBoost - no-history optional",
        "short_name": "XGB no-history optional",
        "model_family": "XGBoost",
        "type": "xgboost",
        "mode": MODE_NO_HISTORY,
        "optimization": OPT_BALANCED,
        "artifact_dir": str(XGB_ROOT / "no_history_simple"),
        "is_default": False,
        "is_optional": True,
        "ui_visible": True,
        "supports": [MODE_NO_HISTORY],
        "requires_history": False,
        "requires_user_home_id": False,
        "uses_home_id_as_feature": False,
        "uses_lag_features": False,
        "uses_rolling_features": False,
        "uses_home_stats": False,
        "uses_consumption_regime": False,
        "uses_behavior_profiles": False,
        "uses_knn_profiles": False,
        "default_prediction_column": "prediction_Wh",
        "description": (
            "Optional generic no-history XGBoost model. Uses temporal, environmental and static household features only."
        ),
    },

    "rf_no_history_optional": {
        "name": "Random Forest - no-history optional",
        "short_name": "RF no-history optional",
        "model_family": "Random Forest",
        "type": "sklearn_rf",
        "mode": MODE_NO_HISTORY,
        "optimization": OPT_BALANCED,
        "artifact_dir": str(RF_ROOT / "no_history"),
        "is_default": False,
        "is_optional": True,
        "ui_visible": True,
        "supports": [MODE_NO_HISTORY],
        "requires_history": False,
        "requires_user_home_id": False,
        "uses_home_id_as_feature": False,
        "uses_lag_features": False,
        "uses_rolling_features": False,
        "uses_home_stats": False,
        "uses_consumption_regime": False,
        "uses_behavior_profiles": False,
        "uses_knn_profiles": False,
        "default_prediction_column": "prediction_Wh",
        "description": (
            "Optional generic no-history Random Forest model. Uses temporal, environmental and static household features only."
        ),
    },
}


# ============================================================
# Defaults / aliases
# ============================================================

DEFAULT_MODEL_BY_MODE: Dict[str, str] = {
    MODE_NO_HISTORY: "lgbm_no_history_default",
    MODE_WITH_HISTORY: "lgbm_with_history_default",
}

DEFAULT_MODEL_BY_MODE_AND_OPTIMIZATION: Dict[tuple[str, str], str] = {
    (MODE_NO_HISTORY, OPT_BALANCED): "lgbm_no_history_default",
    (MODE_NO_HISTORY, OPT_DAILY): "lgbm_no_history_default",
    (MODE_WITH_HISTORY, OPT_BALANCED): "lgbm_with_history_default",
    (MODE_WITH_HISTORY, OPT_DAILY): "lgbm_with_history_default",
}

# Backward-compatible aliases for older API/UI values.
LEGACY_MODEL_ALIASES: Dict[str, Dict[str, str]] = {
    "rf": {
        MODE_NO_HISTORY: "rf_no_history_optional",
        MODE_WITH_HISTORY: "rf_with_history_optional",
    },
    "random_forest": {
        MODE_NO_HISTORY: "rf_no_history_optional",
        MODE_WITH_HISTORY: "rf_with_history_optional",
    },
    "randomforest": {
        MODE_NO_HISTORY: "rf_no_history_optional",
        MODE_WITH_HISTORY: "rf_with_history_optional",
    },
    "lgbm": {
        MODE_NO_HISTORY: "lgbm_no_history_default",
        MODE_WITH_HISTORY: "lgbm_with_history_default",
    },
    "lightgbm": {
        MODE_NO_HISTORY: "lgbm_no_history_default",
        MODE_WITH_HISTORY: "lgbm_with_history_default",
    },
    "xgb": {
        MODE_NO_HISTORY: "xgb_no_history_optional",
        MODE_WITH_HISTORY: "xgb_with_history_optional",
    },
    "xgboost": {
        MODE_NO_HISTORY: "xgb_no_history_optional",
        MODE_WITH_HISTORY: "xgb_with_history_optional",
    },

    # Older IDs from previous naming conventions.
    "rf_coldstart_default": {
        MODE_NO_HISTORY: "rf_no_history_optional",
        MODE_WITH_HISTORY: "rf_with_history_optional",
    },
    "rf_no_history_default": {
        MODE_NO_HISTORY: "rf_no_history_optional",
        MODE_WITH_HISTORY: "rf_with_history_optional",
    },
    "rf_withhistory_optional": {
        MODE_NO_HISTORY: "rf_no_history_optional",
        MODE_WITH_HISTORY: "rf_with_history_optional",
    },
    "rf_with_history_generic": {
        MODE_NO_HISTORY: "rf_no_history_optional",
        MODE_WITH_HISTORY: "rf_with_history_optional",
    },
    "lgbm_coldstart_optional": {
        MODE_NO_HISTORY: "lgbm_no_history_default",
        MODE_WITH_HISTORY: "lgbm_with_history_default",
    },
    "lgbm_no_history_simple": {
        MODE_NO_HISTORY: "lgbm_no_history_default",
        MODE_WITH_HISTORY: "lgbm_with_history_default",
    },
    "lgbm_withhistory_default": {
        MODE_NO_HISTORY: "lgbm_no_history_default",
        MODE_WITH_HISTORY: "lgbm_with_history_default",
    },
    "xgb_coldstart_optional": {
        MODE_NO_HISTORY: "xgb_no_history_optional",
        MODE_WITH_HISTORY: "xgb_with_history_optional",
    },
    "xgb_no_history_simple": {
        MODE_NO_HISTORY: "xgb_no_history_optional",
        MODE_WITH_HISTORY: "xgb_with_history_optional",
    },
    "xgb_withhistory_optional": {
        MODE_NO_HISTORY: "xgb_no_history_optional",
        MODE_WITH_HISTORY: "xgb_with_history_optional",
    },
}

DEFAULT_MODEL_ID = DEFAULT_MODEL_BY_MODE[DEFAULT_MODE]


# ============================================================
# Public helpers
# ============================================================

def normalize_mode(mode: Optional[str]) -> str:
    """Normalize public API/UI mode names to canonical internal values."""
    if not mode:
        return DEFAULT_MODE

    raw = str(mode).strip().lower()
    value = raw.replace("-", "_").replace(" ", "_")
    compact = value.replace("_", "")

    aliases = {
        "no_history": MODE_NO_HISTORY,
        "nohistory": MODE_NO_HISTORY,
        "coldstart": MODE_NO_HISTORY,
        "cold_start": MODE_NO_HISTORY,
        "cold": MODE_NO_HISTORY,
        "raw_frcst": MODE_NO_HISTORY,
        "rawfrcst": MODE_NO_HISTORY,
        "raw_forecast": MODE_NO_HISTORY,
        "rawforecast": MODE_NO_HISTORY,
        "without_history": MODE_NO_HISTORY,
        "withouthistory": MODE_NO_HISTORY,

        "with_history": MODE_WITH_HISTORY,
        "withhistory": MODE_WITH_HISTORY,
        "history": MODE_WITH_HISTORY,
        "withhist": MODE_WITH_HISTORY,
        "recent_history": MODE_WITH_HISTORY,
        "recenthistory": MODE_WITH_HISTORY,
    }

    if value in aliases:
        return aliases[value]

    if compact in aliases:
        return aliases[compact]

    raise ValueError(
        f"Unsupported prediction mode: {mode}. "
        "Use 'no_history' or 'with_history'. "
        "Legacy aliases 'coldstart' and 'withhistory' are also accepted."
    )


def normalize_optimization(optimization: Optional[str]) -> str:
    if not optimization:
        return DEFAULT_OPTIMIZATION

    value = str(optimization).strip().lower().replace("_", "-")

    if value in {"balanced", "default", "pointwise", "hourly"}:
        return OPT_BALANCED

    if value in {"daily", "daily-optimized", "dailyoptimized", "total", "daily-total", "daily_total"}:
        return OPT_DAILY

    raise ValueError(f"Unsupported optimization: {optimization}. Use 'balanced' or 'daily'.")


def get_default_model_id(mode: Optional[str] = None, optimization: Optional[str] = None) -> str:
    normalized_mode = normalize_mode(mode)
    normalized_optimization = normalize_optimization(optimization)

    key = (normalized_mode, normalized_optimization)

    if key in DEFAULT_MODEL_BY_MODE_AND_OPTIMIZATION:
        return DEFAULT_MODEL_BY_MODE_AND_OPTIMIZATION[key]

    return DEFAULT_MODEL_BY_MODE[normalized_mode]


def resolve_model_id(model_id: Optional[str], mode: Optional[str] = None, optimization: Optional[str] = None) -> str:
    """Resolve explicit, auto/default and legacy model IDs to a concrete registry ID."""
    normalized_mode = normalize_mode(mode)

    if model_id is None or str(model_id).strip().lower() in {"", "auto", "default"}:
        return get_default_model_id(normalized_mode, optimization)

    raw = str(model_id).strip()
    key = raw.lower()

    if raw in MODEL_REGISTRY:
        return raw

    if key in MODEL_REGISTRY:
        return key

    if key in LEGACY_MODEL_ALIASES:
        legacy_by_mode = LEGACY_MODEL_ALIASES[key]
        if normalized_mode not in legacy_by_mode:
            raise ValueError(f"Legacy model_id '{model_id}' is not available for mode '{normalized_mode}'.")
        return legacy_by_mode[normalized_mode]

    available = ", ".join(sorted(MODEL_REGISTRY.keys()))
    raise ValueError(f"Unknown model_id: {model_id}. Available model IDs: {available}")


def get_model_config(model_id: str, mode: Optional[str] = None, optimization: Optional[str] = None) -> Dict[str, Any]:
    resolved_id = resolve_model_id(model_id, mode=mode, optimization=optimization)
    cfg = dict(MODEL_REGISTRY[resolved_id])
    cfg["model_id"] = resolved_id
    cfg["resolved_model_id"] = resolved_id
    return cfg


def validate_model_for_mode(model_id: str, mode: Optional[str]) -> Dict[str, Any]:
    normalized_mode = normalize_mode(mode)
    cfg = get_model_config(model_id, mode=normalized_mode)

    if normalized_mode not in cfg.get("supports", []):
        raise ValueError(
            f"Model '{cfg['model_id']}' does not support mode '{normalized_mode}'. "
            f"Supported modes: {cfg.get('supports', [])}"
        )

    if cfg.get("requires_user_home_id", False):
        raise ValueError(
            f"Model '{cfg['model_id']}' requires user home_id and is not allowed in the generic UI."
        )

    if cfg.get("uses_home_id_as_feature", False):
        raise ValueError(
            f"Model '{cfg['model_id']}' uses home_id as a feature and is not allowed in the generic UI."
        )

    return cfg


def list_models(
    mode: Optional[str] = None,
    ui_visible_only: bool = True,
    include_optional: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Return available model metadata, optionally filtered by mode."""
    normalized_mode = normalize_mode(mode) if mode else None

    out: Dict[str, Dict[str, Any]] = {}

    for model_id, cfg in MODEL_REGISTRY.items():
        if normalized_mode and normalized_mode not in cfg.get("supports", []):
            continue

        if ui_visible_only and not cfg.get("ui_visible", True):
            continue

        if not include_optional and cfg.get("is_optional", False):
            continue

        item = dict(cfg)
        item["model_id"] = model_id
        out[model_id] = item

    return out


def list_model_options_for_ui(mode: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return compact dropdown-friendly model options."""
    models = list_models(mode=mode, ui_visible_only=True, include_optional=True)

    options: List[Dict[str, Any]] = [
        {
            "label": "Auto default",
            "value": "auto",
            "is_default": True,
            "is_optional": False,
            "description": "Automatically select the default model for the selected scenario.",
        }
    ]

    for model_id, cfg in models.items():
        options.append({
            "label": cfg.get("short_name", cfg.get("name", model_id)),
            "value": model_id,
            "mode": cfg.get("mode"),
            "is_default": bool(cfg.get("is_default", False)),
            "is_optional": bool(cfg.get("is_optional", False)),
            "requires_history": bool(cfg.get("requires_history", False)),
            "description": cfg.get("description", ""),
        })

    return options


def check_artifact_paths() -> Dict[str, Dict[str, Any]]:
    """Check whether expected artifact folders and core files exist."""
    result: Dict[str, Dict[str, Any]] = {}

    for model_id, cfg in MODEL_REGISTRY.items():
        artifact_dir = Path(cfg["artifact_dir"])
        model_path = artifact_dir / "model.joblib"
        preprocessor_path = artifact_dir / "preprocessor.pkl"
        feature_config_path = artifact_dir / "feature_config.json"
        metadata_path = artifact_dir / "metadata.json"

        result[model_id] = {
            "artifact_dir": str(artifact_dir),
            "artifact_dir_exists": artifact_dir.exists(),
            "model_exists": model_path.exists(),
            "preprocessor_exists": preprocessor_path.exists(),
            "feature_config_exists": feature_config_path.exists(),
            "metadata_exists": metadata_path.exists(),
            "ready": all([
                artifact_dir.exists(),
                model_path.exists(),
                preprocessor_path.exists(),
                feature_config_path.exists(),
            ]),
        }

    return result


if __name__ == "__main__":
    print("Default no_history:", get_default_model_id("no_history"))
    print("Default with_history:", get_default_model_id("with_history"))
    print("No-history UI options:", [x["value"] for x in list_model_options_for_ui("no_history")])
    print("With-history UI options:", [x["value"] for x in list_model_options_for_ui("with_history")])
