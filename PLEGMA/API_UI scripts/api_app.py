# C:/Plegma_Programming/src/api_app.py
# ---------------------------------------------------------
# FastAPI app for PLEGMA forecasting demo - Generic UI/API version
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
#   - No model selected by the UI/API requires or uses home_id as a feature.
#   - Model selection is handled by model_registry.py.
#   - Inference uses final API artifacts:
#       model.joblib
#       preprocessor.pkl
#       feature_config.json
#       metadata.json
#
# Supports:
#   - model_id="auto" routing through model_registry.py
#   - canonical mode="no_history" or mode="with_history"
#   - backward-compatible aliases: "coldstart", "cold_start", "withhistory"
#   - optimization="balanced" or optimization="daily"
#   - Open-Meteo hourly external temperature/humidity for selected Greek cities
#   - fallback to user-provided external_temperature_24h / external_humidity_24h
#   - fallback to t_min / t_max for external temperature
#   - default demo profiles for internal temperature / internal humidity
#   - with-history from:
#       1) user-provided history_consumption_Wh vector
#       2) user-selected single-home history CSV path
#       3) legacy default history_store.csv fallback, if enabled
#   - adaptive history-aware correction for with-history predictions
#
# Notes:
#   - API endpoints are intentionally kept the same as the current PLEGMA app.
#   - Ports/pages are not defined in this file; keep them in the existing
#     launch command / UI app so they do not conflict with IDEAL.
# ---------------------------------------------------------

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from src.history_store import get_history_from_csv, history_store_info
    from src.inference import (
        PRED_COL,
        _time_features,
        load_model_artifacts,
        predict_coldstart_dayahead,
        predict_withhistory_dayahead,
    )
    from src.model_registry import (
        DEFAULT_MODE,
        DEFAULT_MODEL_ID,
        MODEL_REGISTRY,
        MODE_NO_HISTORY,
        MODE_WITH_HISTORY,
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
    from inference import (
        PRED_COL,
        _time_features,
        load_model_artifacts,
        predict_coldstart_dayahead,
        predict_withhistory_dayahead,
    )
    from model_registry import (
        DEFAULT_MODE,
        DEFAULT_MODEL_ID,
        MODEL_REGISTRY,
        MODE_NO_HISTORY,
        MODE_WITH_HISTORY,
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
# Paths / Config
# ============================================================

BASE_DIR = Path(os.getenv("PLEGMA_BASE_DIR", "C:/Plegma_Programming"))

OUT_DIR = Path(
    os.getenv(
        "PLEGMA_API_OUT_DIR",
        str(BASE_DIR / "processed" / "predictions" / "api_like"),
    )
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CITY = os.getenv("PLEGMA_DEFAULT_CITY", "athens")
DEFAULT_MIN_HISTORY_HOURS = int(os.getenv("PLEGMA_DEFAULT_HISTORY_HOURS", "168"))
OPENMETEO_TIMEOUT_SECONDS = int(os.getenv("OPENMETEO_TIMEOUT_SECONDS", "30"))

API_VERSION = "0.6.0-plegma-generic-ui"
CANONICAL_NO_HISTORY = MODE_NO_HISTORY
CANONICAL_WITH_HISTORY = MODE_WITH_HISTORY


# ============================================================
# Supported Greek cities
# ============================================================

SUPPORTED_CITIES: Dict[str, Dict[str, Any]] = {
    "athens": {
        "label": "Athens",
        "latitude": 37.9838,
        "longitude": 23.7275,
    },
    "thessaloniki": {
        "label": "Thessaloniki",
        "latitude": 40.6401,
        "longitude": 22.9444,
    },
    "patra": {
        "label": "Patra",
        "latitude": 38.2466,
        "longitude": 21.7346,
    },
    "heraklion": {
        "label": "Heraklion",
        "latitude": 35.3387,
        "longitude": 25.1442,
    },
}


# ============================================================
# Default demo environmental profiles
# ============================================================

def _default_internal_temperature_24h() -> List[float]:
    return [
        21.0, 20.8, 20.7, 20.6, 20.5, 20.5,
        20.8, 21.2, 21.5, 21.7, 21.8, 21.9,
        22.0, 22.0, 21.9, 21.8, 21.8, 21.9,
        22.0, 22.0, 21.8, 21.5, 21.3, 21.1,
    ]


def _default_internal_humidity_24h() -> List[float]:
    return [
        50.0, 50.0, 49.5, 49.5, 49.0, 49.0,
        49.5, 50.0, 50.5, 51.0, 51.0, 51.5,
        52.0, 52.0, 51.5, 51.0, 50.5, 50.5,
        50.0, 50.0, 49.5, 49.5, 49.5, 50.0,
    ]


def _default_external_humidity_24h() -> List[float]:
    return [60.0] * 24


# ============================================================
# Weather helpers
# ============================================================

def fetch_openmeteo_hourly_weather(
    city: str,
    target_date: str,
) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """Fetch 24h external temperature and relative humidity for a supported city."""
    city_key = str(city or "").strip().lower()
    if city_key not in SUPPORTED_CITIES:
        return None, None

    geo = SUPPORTED_CITIES[city_key]
    target_day = str(pd.Timestamp(target_date).date())

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": geo["latitude"],
        "longitude": geo["longitude"],
        "hourly": "temperature_2m,relative_humidity_2m",
        "timezone": "Europe/Athens",
        "start_date": target_day,
        "end_date": target_day,
    }

    try:
        r = requests.get(url, params=params, timeout=OPENMETEO_TIMEOUT_SECONDS)
    except Exception:
        return None, None

    if r.status_code != 200:
        return None, None

    try:
        js = r.json()
    except Exception:
        return None, None

    hourly = js.get("hourly", {})
    temps = hourly.get("temperature_2m")
    hums = hourly.get("relative_humidity_2m")

    temps_out = None
    hums_out = None

    if isinstance(temps, list) and len(temps) >= 24:
        try:
            temps_out = [float(x) for x in temps[:24]]
        except Exception:
            temps_out = None

    if isinstance(hums, list) and len(hums) >= 24:
        try:
            hums_out = [float(x) for x in hums[:24]]
        except Exception:
            hums_out = None

    return temps_out, hums_out


def build_hourly_temp_from_minmax(tmin: float, tmax: float) -> List[float]:
    tmin = float(tmin)
    tmax = float(tmax)

    if tmax < tmin:
        tmin, tmax = tmax, tmin

    mean = 0.5 * (tmin + tmax)
    amp = 0.5 * (tmax - tmin)

    temps: List[float] = []
    for h in range(24):
        # Peak around 15:00, lower values overnight.
        val = mean + amp * math.cos(2 * math.pi * (h - 15) / 24.0)
        temps.append(float(val))
    return temps


def resolve_external_temperature_24h(
    city: Optional[str],
    target_date: str,
    external_temperature_24h: Optional[List[float]],
    tmin: Optional[float],
    tmax: Optional[float],
) -> Tuple[List[float], str, Optional[float], Optional[float]]:
    if external_temperature_24h is not None:
        if len(external_temperature_24h) != 24:
            raise HTTPException(status_code=422, detail="external_temperature_24h must have length 24.")
        return [float(x) for x in external_temperature_24h], "user_24h", None, None

    if city:
        temps, _ = fetch_openmeteo_hourly_weather(city, target_date)
        if temps is not None:
            return temps, "openmeteo_forecast", None, None

    if tmin is not None and tmax is not None:
        temps24 = build_hourly_temp_from_minmax(tmin, tmax)
        return temps24, "minmax_fallback", float(tmin), float(tmax)

    city_msg = f" for city '{city}'" if city else ""
    raise HTTPException(
        status_code=422,
        detail=(
            f"Could not resolve external_temperature_24h{city_msg}. "
            f"Provide one of: city among {list(SUPPORTED_CITIES.keys())}, "
            "external_temperature_24h with 24 values, or t_min and t_max."
        ),
    )


def resolve_external_humidity_24h(
    city: Optional[str],
    target_date: str,
    external_humidity_24h: Optional[List[float]],
) -> Tuple[List[float], str]:
    if external_humidity_24h is not None:
        if len(external_humidity_24h) != 24:
            raise HTTPException(status_code=422, detail="external_humidity_24h must have length 24.")
        return [float(x) for x in external_humidity_24h], "user_24h"

    if city:
        _, hums = fetch_openmeteo_hourly_weather(city, target_date)
        if hums is not None:
            return hums, "openmeteo_forecast"

    return _default_external_humidity_24h(), "default_profile"


def resolve_internal_temperature_24h(
    internal_temperature_24h: Optional[List[float]],
) -> Tuple[List[float], str]:
    if internal_temperature_24h is not None:
        if len(internal_temperature_24h) != 24:
            raise HTTPException(status_code=422, detail="internal_temperature_24h must have length 24.")
        return [float(x) for x in internal_temperature_24h], "user_24h"

    return _default_internal_temperature_24h(), "default_profile"


def resolve_internal_humidity_24h(
    internal_humidity_24h: Optional[List[float]],
) -> Tuple[List[float], str]:
    if internal_humidity_24h is not None:
        if len(internal_humidity_24h) != 24:
            raise HTTPException(status_code=422, detail="internal_humidity_24h must have length 24.")
        return [float(x) for x in internal_humidity_24h], "user_24h"

    return _default_internal_humidity_24h(), "default_profile"


# ============================================================
# App / startup
# ============================================================

app = FastAPI(title="PLEGMA Load Forecasting API", version=API_VERSION)

# Cache loaded artifacts by resolved model/mode/optimization.
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

def _total_kwh_from_preds(pred_df: pd.DataFrame) -> float:
    return float(pred_df[PRED_COL].sum() / 1000.0)


def _derived_flags_for_day(target_date: str) -> Dict[str, Any]:
    return _time_features(pd.Timestamp(target_date))


def _normalize_user_history(history: List[float], min_hours: int) -> List[float]:
    try:
        hist = [float(x) for x in history]
    except Exception:
        raise HTTPException(status_code=422, detail="history_consumption_Wh must be numeric (list of numbers).")

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
    optional_cols = [
        "pred_model_raw_Wh",
        "history_median_7d_Wh",
        "history_shape_scaled_Wh",
        "adaptive_alpha",
        "history_correction_applied",
        "history_correction_reason",
        "model_id",
        "mode",
        "prediction_variant",
        "history_source",
        "history_rows_used",
        "history_coverage",
    ]

    rows: List[Dict[str, Any]] = []
    for _, r in pred.iterrows():
        item = {
            "timestamp": pd.Timestamp(r["timestamp"]).isoformat(),
            "pred_consumption_Wh": float(r[PRED_COL]),
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
        return {"available": False, "error": str(exc)}


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


def _city_options_for_response() -> List[Dict[str, Any]]:
    return [
        {"city_id": k, "label": v["label"], "latitude": v["latitude"], "longitude": v["longitude"]}
        for k, v in SUPPORTED_CITIES.items()
    ]


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
        "supported_cities": _city_options_for_response(),
        "history_schema": _history_schema_info(),
        "legacy_default_history_store": _get_default_history_store_info(),
    }


@app.get("/models")
def list_models(mode: Optional[str] = None) -> Dict[str, Any]:
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
                "uses_home_id_as_feature": cfg.get("uses_home_id_as_feature"),
                "artifact_dir": cfg.get("artifact_dir"),
                "artifact_status": artifact_status.get(model_id, {}),
                "description": cfg.get("description"),
            }
            for model_id, cfg in models.items()
        ],
        "supported_cities": _city_options_for_response(),
    }


@app.get("/history/info")
def history_info(history_csv_path: Optional[str] = None) -> Dict[str, Any]:
    try:
        info = history_store_info(history_csv_path=history_csv_path)
    except TypeError:
        # Backward compatibility with older history_store.py versions.
        info = history_store_info()
    except Exception as exc:
        info = {"available": False, "error": str(exc), "history_csv_path": history_csv_path}

    return {
        "history_schema": _history_schema_info(),
        "history_store": info,
        "supported_cities": _city_options_for_response(),
        "note": (
            "For the updated UI, prefer passing a selected single-home CSV through history_csv_path. "
            "The legacy default history_store.csv remains available only as fallback."
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

    target_date: str = Field(..., example="2026-10-15")
    city: Optional[str] = Field(default=DEFAULT_CITY, example="athens")

    # Environmental inputs
    internal_temperature_24h: Optional[List[float]] = None
    external_temperature_24h: Optional[List[float]] = None
    internal_humidity_24h: Optional[List[float]] = None
    external_humidity_24h: Optional[List[float]] = None

    # Optional fallback if city forecast is unavailable
    t_min: Optional[float] = None
    t_max: Optional[float] = None

    # Numeric/static PLEGMA inputs
    num_rooms: float = Field(..., ge=0)
    residents: float = Field(..., ge=0)
    num_adults: float = Field(..., ge=0)
    num_children: float = Field(..., ge=0)
    num_elderly: float = Field(..., ge=0)

    has_ac: float = Field(..., ge=0)
    has_fridge_freezer: float = Field(..., ge=0)
    has_dryer: float = Field(..., ge=0)
    has_washing_machine: float = Field(..., ge=0)
    has_dishwasher: float = Field(..., ge=0)
    has_microwave: float = Field(..., ge=0)
    has_electric_oven: float = Field(..., ge=0)
    has_electric_hob: float = Field(..., ge=0)
    solar_panels: float = Field(..., ge=0)

    # Categorical/static PLEGMA inputs
    building_type: str
    build_era: str
    income_band: str
    heating_type: str
    water_heater_type: str
    years_in_house: str

    # Hidden/defaulted by system/UI
    homeowner_status: Optional[str] = None
    occupation: Optional[str] = None

    # With-history sources, in priority order:
    # 1) history_consumption_Wh vector
    # 2) history_csv_path: selected single-home CSV anywhere on disk
    # 3) legacy default history_store.csv fallback, if enabled
    history_consumption_Wh: Optional[List[float]] = None
    history_csv_path: Optional[str] = Field(
        default=None,
        example="C:/Plegma_Programming/processed/stores/selected_history_files/House_01_history.csv",
    )
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

    artifacts = _get_artifacts_for_model(
        model_id=resolved_model_id,
        mode=mode,
        optimization=optimization,
    )

    internal_temperature_24h, internal_temp_source = resolve_internal_temperature_24h(req.internal_temperature_24h)
    internal_humidity_24h, internal_hum_source = resolve_internal_humidity_24h(req.internal_humidity_24h)

    external_temperature_24h, weather_source, tmin_used, tmax_used = resolve_external_temperature_24h(
        city=req.city,
        target_date=req.target_date,
        external_temperature_24h=req.external_temperature_24h,
        tmin=req.t_min,
        tmax=req.t_max,
    )
    external_humidity_24h, external_hum_source = resolve_external_humidity_24h(
        city=req.city,
        target_date=req.target_date,
        external_humidity_24h=req.external_humidity_24h,
    )

    occupation_value = req.occupation if req.occupation is not None else "unknown"
    homeowner_status_value = req.homeowner_status if req.homeowner_status is not None else "renter"

    history = None
    history_csv_path = None
    history_source = None
    history_store_path = None
    history_window_start = None
    history_window_end = None
    min_history_hours = int(req.min_history_hours)

    if mode == CANONICAL_WITH_HISTORY:
        min_history_hours = int(max(req.min_history_hours, int(model_cfg.get("min_history_hours", DEFAULT_MIN_HISTORY_HOURS))))

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
            except TypeError:
                # Backward compatibility with older PLEGMA history_store.py.
                try:
                    history, hs, ws, we = get_history_from_csv(
                        target_ts=f"{req.target_date} 00:00:00",
                        history_hours=min_history_hours,
                        fallback_to_latest=True,
                    )
                except Exception as exc:
                    raise HTTPException(status_code=422, detail=f"Default CSV history failed: {exc}")
            except Exception as exc:
                raise HTTPException(status_code=422, detail=f"Default CSV history failed: {exc}")

            history = _normalize_user_history(history, min_history_hours)
            history_source = f"legacy_default_store:{hs}"
            info = _get_default_history_store_info()
            history_store_path = info.get("history_csv_path") or info.get("default_history_csv_path")
            history_window_start = ws
            history_window_end = we

    try:
        common_kwargs = dict(
            target_date=req.target_date,
            internal_temperature_24h=internal_temperature_24h,
            external_temperature_24h=external_temperature_24h,
            internal_humidity_24h=internal_humidity_24h,
            external_humidity_24h=external_humidity_24h,
            num_rooms=req.num_rooms,
            residents=req.residents,
            num_adults=req.num_adults,
            num_children=req.num_children,
            num_elderly=req.num_elderly,
            has_ac=req.has_ac,
            has_fridge_freezer=req.has_fridge_freezer,
            has_dryer=req.has_dryer,
            has_washing_machine=req.has_washing_machine,
            has_dishwasher=req.has_dishwasher,
            has_microwave=req.has_microwave,
            has_electric_oven=req.has_electric_oven,
            has_electric_hob=req.has_electric_hob,
            solar_panels=req.solar_panels,
            building_type=req.building_type,
            build_era=req.build_era,
            income_band=req.income_band,
            heating_type=req.heating_type,
            water_heater_type=req.water_heater_type,
            homeowner_status=homeowner_status_value,
            years_in_house=req.years_in_house,
            occupation=occupation_value,
            model_id=resolved_model_id,
            optimization=optimization,
            save_csv=req.save_csv,
            out_dir=OUT_DIR,
        )

        if mode == CANONICAL_NO_HISTORY:
            pred = predict_coldstart_dayahead(
                artifacts=artifacts,
                **common_kwargs,
                out_name=f"pred_{resolved_model_id}_{mode}_{req.city}_{req.target_date}.csv".replace(" ", "_"),
            )

        elif mode == CANONICAL_WITH_HISTORY:
            use_daily_calibrated = str(optimization).lower().startswith("daily") if optimization else False
            pred = predict_withhistory_dayahead(
                artifacts=artifacts,
                **common_kwargs,
                history_consumption_Wh=history,
                history_csv_path=history_csv_path,
                min_history_hours=min_history_hours,
                use_daily_calibrated=use_daily_calibrated,
                apply_history_correction=req.apply_history_correction,
                history_correction_days=req.history_correction_days,
                history_correction_max_alpha=req.history_correction_max_alpha,
                out_name=f"pred_{resolved_model_id}_{mode}_{req.city}_{req.target_date}.csv".replace(" ", "_"),
            )
        else:
            raise HTTPException(status_code=500, detail=f"Invalid normalized mode: {mode}")

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
        history_store_path = history_store_path or history_csv_path

    city_label = None
    if req.city:
        city_key = str(req.city).strip().lower()
        if city_key in SUPPORTED_CITIES:
            city_label = SUPPORTED_CITIES[city_key]["label"]

    day_info = _derived_flags_for_day(req.target_date)

    return {
        "requested_model_id": req.model_id,
        "model_id": resolved_model_id,
        "model_name": model_cfg.get("name"),
        "model_type": model_cfg.get("type"),
        "mode": mode,
        "optimization": optimization,
        "prediction_variant": pred["prediction_variant"].iloc[0] if "prediction_variant" in pred.columns else "balanced",

        "city": req.city,
        "city_label": city_label,
        "target_date": req.target_date,

        "day_of_week": day_info.get("day_of_week"),
        "month": day_info.get("month"),
        "season": day_info.get("season"),
        "is_weekend": day_info.get("is_weekend"),
        "is_holiday": day_info.get("is_holiday"),
        "derived_flags": day_info,

        "total_kWh_day": _total_kwh_from_preds(pred),

        "weather_source": weather_source,
        "external_humidity_source": external_hum_source,
        "internal_temperature_source": internal_temp_source,
        "internal_humidity_source": internal_hum_source,
        "t_min_used": tmin_used,
        "t_max_used": tmax_used,

        "external_temperature_24h": external_temperature_24h,
        "external_humidity_24h": external_humidity_24h,
        "internal_temperature_24h": internal_temperature_24h,
        "internal_humidity_24h": internal_humidity_24h,

        "occupation_used": occupation_value,
        "homeowner_status_used": homeowner_status_value,

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

        "predictions": _prediction_rows(pred),
    }
