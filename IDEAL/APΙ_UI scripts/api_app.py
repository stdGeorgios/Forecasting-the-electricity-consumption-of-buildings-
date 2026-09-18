# C:/IDEAL_Programming/src/api_app.py
# ---------------------------------------------------------
# FastAPI app for IDEAL forecasting demo - Generic UI version
# ---------------------------------------------------------
#
# Unified endpoints:
#   POST /predict
#   GET  /models
#   GET  /health
#   GET  /history/info
#
# Final UI assumptions:
#   - The user does NOT provide home_id.
#   - No-history default:
#       RF2/cold_start_default
#   - With-history default:
#       LGBM2/with_history_generic
#   - Optional comparison models:
#       RF2/with_history_generic
#       XGB2/with_history_generic
#       LGBM2/cold_start_optional
#       XGB2/cold_start_optional
#
# Supports:
#   - model_id="auto" routing through model_registry.py
#   - canonical mode="no_history" or mode="with_history"
#   - backward-compatible aliases: "coldstart", "cold_start", "withhistory"
#   - optimization="balanced" or optimization="daily"
#   - external temperature from:
#       1) user-provided 24h vector
#       2) Open-Meteo forecast
#       3) t_min / t_max fallback
#   - with-history from:
#       1) user-provided history_consumption_Wh vector
#       2) user-selected single-home history CSV path
#       3) legacy default history_store.csv fallback, if enabled
#   - adaptive history-aware correction for with-history predictions
#
# Notes:
#   - The API never asks the user for home_id.
#   - Single-home history CSV must contain: timestamp, consumption_Wh.
#   - With-history generic models require recent hourly consumption history.
#   - Default min_history_hours = 168.
# ---------------------------------------------------------

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from src.history_store import get_history_from_csv, history_store_info
    from src.inference import load_model_artifacts, predict_dayahead
    from src.model_registry import (
        DEFAULT_MODE,
        DEFAULT_MODEL_ID,
        MODEL_REGISTRY,
        check_artifact_paths,
        get_default_model_id,
        list_model_options_for_ui,
        list_models as registry_list_models,
        normalize_mode,
        normalize_optimization,
        resolve_model_id,
        validate_model_for_mode,
    )
except ModuleNotFoundError:
    from history_store import get_history_from_csv, history_store_info
    from inference import load_model_artifacts, predict_dayahead
    from model_registry import (
        DEFAULT_MODE,
        DEFAULT_MODEL_ID,
        MODEL_REGISTRY,
        check_artifact_paths,
        get_default_model_id,
        list_model_options_for_ui,
        list_models as registry_list_models,
        normalize_mode,
        normalize_optimization,
        resolve_model_id,
        validate_model_for_mode,
    )
# ============================================================
# Paths / config
# ============================================================

BASE_DIR = Path(os.getenv("IDEAL_BASE_DIR", "C:/IDEAL_Programming"))

OUT_DIR = Path(
    os.getenv(
        "IDEAL_API_OUT_DIR",
        str(BASE_DIR / "processed" / "predictions" / "api_like"),
    )
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CITY = os.getenv("IDEAL_DEFAULT_CITY", "Edinburgh")
DEFAULT_MIN_HISTORY_HOURS = int(os.getenv("IDEAL_DEFAULT_HISTORY_HOURS", "168"))
OPENMETEO_TIMEOUT_SECONDS = int(os.getenv("OPENMETEO_TIMEOUT_SECONDS", "30"))

# Holidays CSV candidates
HOLIDAYS_CANDIDATES = [
    Path(os.getenv("HOLIDAYS_PATH", "")) if os.getenv("HOLIDAYS_PATH") else None,
    BASE_DIR / "holidays.csv",
    BASE_DIR / "metadata" / "holidays.csv",
    BASE_DIR / "processed" / "metadata" / "holidays.csv",
    BASE_DIR / "processed" / "holidays.csv",
]
HOLIDAYS_CANDIDATES = [p for p in HOLIDAYS_CANDIDATES if p is not None and str(p) != ""]


# ============================================================
# Holidays helper
# ============================================================

def _load_holidays_set() -> set:
    for p in HOLIDAYS_CANDIDATES:
        try:
            if p.exists():
                df = pd.read_csv(p)
                if df.shape[1] == 0:
                    continue

                if "date" in df.columns:
                    s = pd.to_datetime(df["date"], errors="coerce").dt.date.dropna()
                else:
                    s = pd.to_datetime(df.iloc[:, 0], errors="coerce").dt.date.dropna()

                return set(s.tolist())
        except Exception:
            continue

    return set()


HOLIDAYS_SET = _load_holidays_set()


def is_holiday_date(d: pd.Timestamp) -> int:
    try:
        return int(d.date() in HOLIDAYS_SET)
    except Exception:
        return 0


# ============================================================
# Weather helpers
# ============================================================

def _geocode_city(city: str) -> Optional[Dict[str, float]]:
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": city, "count": 1, "language": "en", "format": "json"}

    try:
        r = requests.get(url, params=params, timeout=20)
    except Exception:
        return None

    if r.status_code != 200:
        return None

    js = r.json()
    if "results" not in js or not js["results"]:
        return None

    lat = float(js["results"][0]["latitude"])
    lon = float(js["results"][0]["longitude"])
    return {"lat": lat, "lon": lon}


def fetch_openmeteo_hourly_temp(city: str, target_date: str) -> Optional[List[float]]:
    """Fetch 24h external temperature from Open-Meteo forecast API.

    For historical IDEAL test dates, this may fail because the forecast endpoint
    is intended for current/future periods. The API then falls back to user
    t_min/t_max or user external_temperature_24h.
    """
    geo = _geocode_city(city)
    if geo is None:
        return None

    target_day = str(pd.Timestamp(target_date).date())

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": geo["lat"],
        "longitude": geo["lon"],
        "hourly": "temperature_2m",
        "timezone": "Europe/London",
        "start_date": target_day,
        "end_date": target_day,
    }

    try:
        r = requests.get(url, params=params, timeout=OPENMETEO_TIMEOUT_SECONDS)
    except Exception:
        return None

    if r.status_code != 200:
        return None

    js = r.json()
    if "hourly" not in js or "temperature_2m" not in js["hourly"]:
        return None

    temps = js["hourly"]["temperature_2m"]
    if len(temps) < 24:
        return None

    return [float(x) for x in temps[:24]]


def build_hourly_temp_from_minmax(tmin: float, tmax: float) -> List[float]:
    tmin = float(tmin)
    tmax = float(tmax)

    if tmax < tmin:
        tmin, tmax = tmax, tmin

    mean = 0.5 * (tmin + tmax)
    amp = 0.5 * (tmax - tmin)

    temps = []
    for h in range(24):
        # Peak around 15:00, lower values overnight.
        val = mean + amp * math.cos(2 * math.pi * (h - 15) / 24.0)
        temps.append(float(val))

    return temps


def resolve_external_temperature_24h(
    city: str,
    target_date: str,
    external_temperature_24h: Optional[List[float]],
    tmin: Optional[float],
    tmax: Optional[float],
) -> Tuple[List[float], str, Optional[float], Optional[float]]:
    if external_temperature_24h is not None:
        if len(external_temperature_24h) != 24:
            raise HTTPException(status_code=422, detail="external_temperature_24h must have length 24.")
        return [float(x) for x in external_temperature_24h], "user_24h", None, None

    temps = fetch_openmeteo_hourly_temp(city, target_date)
    if temps is not None:
        return temps, "openmeteo_forecast", None, None

    if tmin is not None and tmax is not None:
        temps24 = build_hourly_temp_from_minmax(tmin, tmax)
        return temps24, "minmax_fallback", float(tmin), float(tmax)

    raise HTTPException(
        status_code=422,
        detail=(
            "External temperature could not be resolved automatically. "
            "Provide either t_min / t_max or external_temperature_24h with 24 values."
        ),
    )


def resolve_internal_temperature_24h(
    internal_temperature_24h: Optional[List[float]],
) -> Optional[List[float]]:
    if internal_temperature_24h is None:
        return None

    if len(internal_temperature_24h) != 24:
        raise HTTPException(status_code=422, detail="internal_temperature_24h must have length 24.")

    return [float(x) for x in internal_temperature_24h]


# ============================================================
# App / startup
# ============================================================

app = FastAPI(title="IDEAL Load Forecasting API", version="0.6.0-generic-ui")

# Cache loaded artifacts by resolved model_id.
ARTIFACT_CACHE: Dict[str, Dict[str, Any]] = {}


def _get_artifacts_for_model(
    model_id: str,
    mode: str,
    optimization: str = "balanced",
) -> Dict[str, Any]:
    try:
        resolved_id = resolve_model_id(model_id, mode=mode, optimization=optimization)
        validate_model_for_mode(resolved_id, mode)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    cache_key = f"{resolved_id}::{mode}::{optimization}"
    if cache_key in ARTIFACT_CACHE:
        return ARTIFACT_CACHE[cache_key]

    try:
        artifacts = load_model_artifacts(
            model_id=resolved_id,
            mode=mode,
            optimization=optimization,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load artifacts for model '{resolved_id}': {exc}",
        )

    ARTIFACT_CACHE[cache_key] = artifacts
    return artifacts


# ============================================================
# Response utilities
# ============================================================

CANONICAL_NO_HISTORY = "no_history"
CANONICAL_WITH_HISTORY = "with_history"
API_VERSION = "0.6.0-generic-ui"


def _total_kwh_from_preds(pred_df: pd.DataFrame) -> float:
    return float(pred_df["pred_consumption_Wh"].sum() / 1000.0)


def _derived_flags_for_day(target_date: str) -> Dict[str, int]:
    d0 = pd.Timestamp(target_date)
    return {
        "is_weekend": int(d0.dayofweek >= 5),
        "is_holiday": int(is_holiday_date(d0)),
    }


def _normalize_user_history(history: List[float], min_hours: int) -> List[float]:
    try:
        hist = [float(x) for x in history]
    except Exception:
        raise HTTPException(status_code=422, detail="history_consumption_Wh must be a numeric list.")

    if len(hist) < int(min_hours):
        raise HTTPException(
            status_code=422,
            detail=f"history_consumption_Wh length={len(hist)} but min_history_hours={min_hours}.",
        )

    return hist[-int(min_hours):]


def _json_safe_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (float, int, str, bool)):
        return value
    return str(value)


def _prediction_rows(pred: pd.DataFrame) -> List[Dict[str, Any]]:
    """Return API prediction rows, including optional correction diagnostics."""
    optional_cols = [
        "pred_model_raw_Wh",
        "history_median_7d_Wh",
        "history_shape_scaled_Wh",
        "adaptive_alpha",
        "history_correction_applied",
        "history_correction_reason",
    ]

    rows: List[Dict[str, Any]] = []

    for _, r in pred.iterrows():
        item = {
            "timestamp": pd.Timestamp(r["timestamp"]).isoformat(),
            "pred_consumption_Wh": float(r["pred_consumption_Wh"]),
        }

        for c in optional_cols:
            if c in pred.columns:
                item[c] = _json_safe_value(r[c])

        rows.append(item)

    return rows


def _get_default_history_store_info() -> Dict[str, Any]:
    try:
        return history_store_info()
    except Exception as exc:
        return {
            "available": False,
            "error": str(exc),
        }


def _history_schema_info() -> Dict[str, Any]:
    return {
        "single_home_history_csv": True,
        "required_columns": ["timestamp", "consumption_Wh"],
        "home_id_column_required": False,
        "selection_logic": (
            "The API treats the selected history CSV as one household only. "
            "To evaluate another household, select another history CSV."
        ),
    }


# ============================================================
# Endpoints
# ============================================================

@app.get("/health")
def health() -> Dict[str, Any]:
    artifact_status = check_artifact_paths()
    ready_models = [model_id for model_id, info in artifact_status.items() if info.get("ready")]

    return {
        "status": "ok",
        "api_version": API_VERSION,
        "base_dir": str(BASE_DIR),
        "out_dir": str(OUT_DIR),
        "default_mode": normalize_mode(DEFAULT_MODE),
        "default_model_id": DEFAULT_MODEL_ID,
        "default_no_history_model": get_default_model_id(CANONICAL_NO_HISTORY),
        "default_with_history_model": get_default_model_id(CANONICAL_WITH_HISTORY),
        # Legacy keys kept so older UI code does not break immediately.
        "default_coldstart_model": get_default_model_id(CANONICAL_NO_HISTORY),
        "default_withhistory_model": get_default_model_id(CANONICAL_WITH_HISTORY),
        "ready_models": ready_models,
        "num_ready_models": len(ready_models),
        "history_schema": _history_schema_info(),
        "legacy_default_history_store": _get_default_history_store_info(),
    }


@app.get("/models")
def list_models(
    mode: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        normalized_mode = normalize_mode(mode) if mode else None
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    models = registry_list_models(mode=normalized_mode, ui_visible_only=True, include_optional=True)
    artifact_status = check_artifact_paths()

    return {
        "default_model_by_mode": {
            CANONICAL_NO_HISTORY: get_default_model_id(CANONICAL_NO_HISTORY),
            CANONICAL_WITH_HISTORY: get_default_model_id(CANONICAL_WITH_HISTORY),
            # Legacy aliases for compatibility.
            "coldstart": get_default_model_id(CANONICAL_NO_HISTORY),
            "withhistory": get_default_model_id(CANONICAL_WITH_HISTORY),
        },
        "options_for_ui": list_model_options_for_ui(mode=normalized_mode),
        "models": [
            {
                "model_id": model_id,
                "name": cfg.get("name"),
                "short_name": cfg.get("short_name"),
                "model_family": cfg.get("model_family"),
                "type": cfg.get("type"),
                "mode": cfg.get("mode"),
                "supports": cfg.get("supports"),
                "is_default": cfg.get("is_default"),
                "is_optional": cfg.get("is_optional"),
                "requires_history": cfg.get("requires_history"),
                "min_history_hours": cfg.get("min_history_hours"),
                "requires_user_home_id": cfg.get("requires_user_home_id"),
                "artifact_dir": cfg.get("artifact_dir"),
                "artifact_status": artifact_status.get(model_id, {}),
                "description": cfg.get("description"),
            }
            for model_id, cfg in models.items()
        ],
    }


@app.get("/history/info")
def history_info() -> Dict[str, Any]:
    return {
        "history_schema": _history_schema_info(),
        "legacy_default_history_store": _get_default_history_store_info(),
        "note": (
            "For the updated UI, prefer passing a user-selected single-home CSV "
            "through history_csv_path. The legacy default history_store.csv remains "
            "available only as a fallback."
        ),
    }


# ============================================================
# Request model
# ============================================================

class PredictRequest(BaseModel):
    # 'auto' chooses the default model for the selected mode.
    model_id: str = Field(default="auto", example="auto")
    mode: str = Field(default=DEFAULT_MODE, example="no_history")
    optimization: str = Field(default="balanced", example="balanced")

    city: str = Field(default=DEFAULT_CITY, example="Edinburgh")
    target_date: str = Field(..., example="2026-02-21")

    total_floor_area_m2: float = Field(..., example=85, ge=0)
    residents: float = Field(..., example=2, ge=0)
    hometype: str = Field(..., example="flat")
    urban_rural_class: str = Field(..., example="1")
    num_electric_appliances: Optional[float] = Field(default=None, example=12, ge=0)

    # Weather / environmental inputs
    external_temperature_24h: Optional[List[float]] = None
    internal_temperature_24h: Optional[List[float]] = None
    t_min: Optional[float] = None
    t_max: Optional[float] = None

    # With-history sources, in priority order:
    # 1) history_consumption_Wh vector
    # 2) history_csv_path: selected single-home CSV anywhere on disk
    # 3) legacy default history_store.csv fallback, if enabled
    history_consumption_Wh: Optional[List[float]] = None
    history_csv_path: Optional[str] = Field(default=None, example="C:/IDEAL_Programming/processed/stores/history_store.csv")
    min_history_hours: int = Field(default=DEFAULT_MIN_HISTORY_HOURS, ge=1)

    # New preferred switch for the legacy default store.
    use_default_history_store: Optional[bool] = None
    # Backward-compatible old field name.
    use_proxy_history: bool = True

    # Adaptive correction for with_history.
    apply_history_correction: bool = True
    history_correction_days: int = Field(default=7, ge=1, le=30)
    history_correction_max_alpha: float = Field(default=0.20, ge=0.0, le=1.0)

    save_csv: bool = False


# ============================================================
# Unified predict endpoint
# ============================================================

@app.post("/predict")
def predict(req: PredictRequest) -> Dict[str, Any]:
    try:
        mode = normalize_mode(req.mode)
        optimization = normalize_optimization(req.optimization)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    try:
        resolved_model_id = resolve_model_id(req.model_id, mode=mode, optimization=optimization)
        model_cfg = validate_model_for_mode(resolved_model_id, mode)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    _ = _get_artifacts_for_model(
        model_id=resolved_model_id,
        mode=mode,
        optimization=optimization,
    )

    temps24, weather_source, tmin_used, tmax_used = resolve_external_temperature_24h(
        city=req.city,
        target_date=req.target_date,
        external_temperature_24h=req.external_temperature_24h,
        tmin=req.t_min,
        tmax=req.t_max,
    )

    internal24 = resolve_internal_temperature_24h(req.internal_temperature_24h)

    history = None
    history_csv_path = None
    history_source = None
    history_store_path = None
    history_window_start = None
    history_window_end = None
    min_history_hours = int(req.min_history_hours)

    if mode == CANONICAL_WITH_HISTORY:
        min_history_hours = int(
            max(
                req.min_history_hours,
                int(model_cfg.get("min_history_hours", DEFAULT_MIN_HISTORY_HOURS)),
            )
        )

        if req.history_consumption_Wh is not None:
            history = _normalize_user_history(req.history_consumption_Wh, min_history_hours)
            history_source = "user_vector"

        elif req.history_csv_path:
            history_csv_path = str(req.history_csv_path)
            history_source = "user_selected_csv"
            history_store_path = history_csv_path

        else:
            use_default_store = req.use_proxy_history if req.use_default_history_store is None else bool(req.use_default_history_store)

            if not use_default_store:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "with_history mode requires one of: history_consumption_Wh, "
                        "history_csv_path, or use_default_history_store=True."
                    ),
                )

            try:
                history, hs, ws, we = get_history_from_csv(
                    target_date=req.target_date,
                    history_hours=min_history_hours,
                    fallback_to_latest=True,
                )
            except Exception as exc:
                raise HTTPException(status_code=422, detail=f"Default CSV history failed: {exc}")

            history = _normalize_user_history(history, min_history_hours)
            history_source = f"legacy_default_store:{hs}"
            info = _get_default_history_store_info()
            history_store_path = info.get("history_csv_path")
            history_window_start = ws
            history_window_end = we

    try:
        pred = predict_dayahead(
            mode=mode,
            model_id=resolved_model_id,
            optimization=optimization,
            target_date=req.target_date,
            external_temperature_24h=temps24,
            history_consumption_Wh=history,
            history_csv_path=history_csv_path,
            total_floor_area_m2=req.total_floor_area_m2,
            residents=req.residents,
            hometype=req.hometype,
            urban_rural_class=req.urban_rural_class,
            internal_temperature_24h=internal24,
            num_electric_appliances=req.num_electric_appliances,
            min_history_hours=min_history_hours,
            apply_history_correction=req.apply_history_correction,
            history_correction_days=req.history_correction_days,
            history_correction_max_alpha=req.history_correction_max_alpha,
            save_csv=req.save_csv,
            out_dir=OUT_DIR,
            out_name=f"pred_{resolved_model_id}_{mode}_{req.city}_{req.target_date}.csv".replace(" ", "_"),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    history_input_info = pred.attrs.get("history_input_info", {}) if hasattr(pred, "attrs") else {}
    history_correction_info = pred.attrs.get("history_correction_info", {}) if hasattr(pred, "attrs") else {}

    if history_input_info:
        history_window_start = history_window_start or history_input_info.get("history_start")
        history_window_end = history_window_end or history_input_info.get("history_end")
        history_source = history_source or history_input_info.get("history_source")

    return {
        "requested_model_id": req.model_id,
        "model_id": resolved_model_id,
        "model_name": model_cfg.get("name"),
        "model_type": model_cfg.get("type"),
        "mode": mode,
        "optimization": optimization,
        "prediction_variant": pred["prediction_variant"].iloc[0] if "prediction_variant" in pred.columns else "balanced",

        "city": req.city,
        "target_date": req.target_date,
        "total_kWh_day": _total_kwh_from_preds(pred),

        "weather_source": weather_source,
        "t_min_used": tmin_used,
        "t_max_used": tmax_used,
        "external_temperature_24h": temps24,
        "internal_temperature_24h": internal24,

        "history_source": history_source,
        "history_csv_path": history_csv_path,
        "history_store_path": history_store_path,
        "history_window_start": history_window_start,
        "history_window_end": history_window_end,
        "history_hours_used": (
            int(history_input_info.get("required_history_hours"))
            if history_input_info and history_input_info.get("required_history_hours") is not None
            else (len(history) if history is not None else None)
        ),
        "history_input_info": history_input_info,
        "history_correction_info": history_correction_info,
        "history_correction_applied": bool(history_correction_info.get("history_correction_applied", False)),
        "history_correction_reason": history_correction_info.get("history_correction_reason"),

        "derived_flags": _derived_flags_for_day(req.target_date),

        "predictions": _prediction_rows(pred),
    }
