# C:/Plegma_Programming/src/api_app.py
# ---------------------------------------------------------
# FastAPI app for PLEGMA forecasting demo
#
# Unified endpoint:
# POST /predict
# GET /models
# GET /health
# GET /history/info
#
# Supports:
# - Multiple model families via model_registry.py (RF, XGB, LGBM)
# - Separate prediction mode: coldstart / withhistory
# - Auto external temperature via Open-Meteo for selected Greek cities
# - Auto external humidity via Open-Meteo for selected Greek cities
# - Fallback to user-provided external_temperature_24h / external_humidity_24h
# - Fallback to t_min / t_max for external temperature
# - Default demo profiles for internal temperature / internal humidity
# - With-history:
#     1) user-provided history_consumption_Wh
#     2) proxy history from history_store.csv
# ---------------------------------------------------------

from __future__ import annotations

import os
import math
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Literal

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference import (
    load_model_artifacts,
    predict_coldstart_dayahead,
    predict_withhistory_dayahead,
    _time_features,
)
from src.history_store import get_history_from_csv, history_store_info
from src.model_registry import MODEL_REGISTRY, DEFAULT_MODEL_ID, DEFAULT_MODE


# =========================
# Paths / Config
# =========================
BASE_DIR = Path(os.getenv("PLEGMA_BASE_DIR", "C:/Plegma_Programming"))

OUT_DIR = Path(
    os.getenv(
        "PLEGMA_API_OUT_DIR",
        str(BASE_DIR / "predictions" / "api_like")
    )
)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Supported cities
# =========================
SUPPORTED_CITIES = {
    "athens": {"label": "Athens", "latitude": 37.9838, "longitude": 23.7275},
    "thessaloniki": {"label": "Thessaloniki", "latitude": 40.6401, "longitude": 22.9444},
    "patra": {"label": "Patra", "latitude": 38.2466, "longitude": 21.7346},
    "heraklion": {"label": "Heraklion", "latitude": 35.3387, "longitude": 25.1442},
}


# =========================
# Default demo profiles
# =========================
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


# =========================
# Weather helpers
# =========================
def fetch_openmeteo_hourly_weather(
    city: str,
    target_date: str,
) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    city_key = str(city).strip().lower()
    if city_key not in SUPPORTED_CITIES:
        return None, None

    geo = SUPPORTED_CITIES[city_key]

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": geo["latitude"],
        "longitude": geo["longitude"],
        "hourly": "temperature_2m,relative_humidity_2m",
        "timezone": "Europe/Athens",
        "start_date": str(pd.Timestamp(target_date).date()),
        "end_date": str(pd.Timestamp(target_date).date()),
    }

    try:
        r = requests.get(url, params=params, timeout=30)
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
        temps_out = [float(x) for x in temps[:24]]

    if isinstance(hums, list) and len(hums) >= 24:
        hums_out = [float(x) for x in hums[:24]]

    return temps_out, hums_out


def build_hourly_temp_from_minmax(tmin: float, tmax: float) -> List[float]:
    tmin = float(tmin)
    tmax = float(tmax)

    if tmax < tmin:
        tmin, tmax = tmax, tmin

    mean = 0.5 * (tmin + tmax)
    amp = 0.5 * (tmax - tmin)

    temps = []
    for h in range(24):
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
            f"Provide one of the following: "
            f"(a) city among {list(SUPPORTED_CITIES.keys())}, "
            f"(b) external_temperature_24h with 24 values, "
            f"(c) t_min and t_max."
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


# =========================
# App / Startup
# =========================
app = FastAPI(title="PLEGMA Load Forecasting API", version="0.3.0")

ARTIFACT_CACHE: Dict[str, Dict[str, Any]] = {}


def _get_artifacts_for_model(model_id: str) -> Dict[str, Any]:
    if model_id in ARTIFACT_CACHE:
        return ARTIFACT_CACHE[model_id]

    if model_id not in MODEL_REGISTRY:
        raise HTTPException(status_code=422, detail=f"Unknown model_id: {model_id}")

    cfg = MODEL_REGISTRY[model_id]
    art_dir = Path(cfg["artifact_dir"])
    model_type = cfg["type"]

    try:
        artifacts = load_model_artifacts(model_type=model_type, art_dir=art_dir)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load artifacts for model '{model_id}': {e}"
        )

    ARTIFACT_CACHE[model_id] = artifacts
    return artifacts


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def list_models():
    return {
        "models": [
            {
                "model_id": model_id,
                "name": cfg["name"],
                "type": cfg["type"],
                "supports": cfg["supports"],
            }
            for model_id, cfg in MODEL_REGISTRY.items()
        ],
        "supported_cities": [
            {"city_id": k, "label": v["label"]}
            for k, v in SUPPORTED_CITIES.items()
        ],
    }


@app.get("/history/info")
def history_info():
    info = history_store_info()
    info["supported_cities"] = [
        {"city_id": k, "label": v["label"]}
        for k, v in SUPPORTED_CITIES.items()
    ]
    return info


# =========================
# Request model
# =========================
class PredictRequest(BaseModel):
    model_id: str = Field(default=DEFAULT_MODEL_ID, example="rf")
    mode: Literal["coldstart", "withhistory"] = Field(default=DEFAULT_MODE, example="coldstart")

    target_date: str = Field(..., example="2026-10-15")
    city: Optional[str] = Field(default="athens", example="athens")

    # Environmental inputs
    internal_temperature_24h: Optional[List[float]] = None
    external_temperature_24h: Optional[List[float]] = None
    internal_humidity_24h: Optional[List[float]] = None
    external_humidity_24h: Optional[List[float]] = None

    # Optional fallback if city forecast is unavailable
    t_min: Optional[float] = None
    t_max: Optional[float] = None

    # Numeric/static inputs
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

    # Categorical/static inputs
    building_type: str
    build_era: str
    income_band: str
    heating_type: str
    water_heater_type: str
    years_in_house: str

    # hidden/defaulted by system
    homeowner_status: Optional[str] = None
    occupation: Optional[str] = None

    # With-history sources
    use_proxy_history: bool = True
    history_consumption_Wh: Optional[List[float]] = None
    min_history_hours: int = 168

    save_csv: bool = False


# =========================
# Utilities
# =========================
def _total_kwh_from_preds(pred_df: pd.DataFrame) -> float:
    return float(pred_df["pred_consumption_Wh"].sum() / 1000.0)


def _normalize_user_history(history: List[float], min_hours: int) -> List[float]:
    try:
        hist = [float(x) for x in history]
    except Exception:
        raise HTTPException(
            status_code=422,
            detail="history_consumption_Wh must be numeric (list of numbers)."
        )

    if len(hist) < int(min_hours):
        raise HTTPException(
            status_code=422,
            detail=f"history_consumption_Wh length={len(hist)} but min_history_hours={min_hours}.",
        )

    return hist[-int(min_hours):]


# =========================
# Unified endpoint
# =========================
@app.post("/predict")
def predict(req: PredictRequest) -> Dict[str, Any]:
    if req.model_id not in MODEL_REGISTRY:
        raise HTTPException(status_code=422, detail=f"Unknown model_id: {req.model_id}")

    model_cfg = MODEL_REGISTRY[req.model_id]
    mode = req.mode

    if mode not in model_cfg["supports"]:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Model '{req.model_id}' does not support mode '{mode}'. "
                f"Supported: {model_cfg['supports']}"
            ),
        )

    artifacts = _get_artifacts_for_model(req.model_id)

    internal_temperature_24h, internal_temp_source = resolve_internal_temperature_24h(
        req.internal_temperature_24h
    )
    internal_humidity_24h, internal_hum_source = resolve_internal_humidity_24h(
        req.internal_humidity_24h
    )
    external_humidity_24h, external_hum_source = resolve_external_humidity_24h(
        city=req.city,
        target_date=req.target_date,
        external_humidity_24h=req.external_humidity_24h,
    )
    external_temperature_24h, weather_source, tmin_used, tmax_used = resolve_external_temperature_24h(
        city=req.city,
        target_date=req.target_date,
        external_temperature_24h=req.external_temperature_24h,
        tmin=req.t_min,
        tmax=req.t_max,
    )

    occupation_value = req.occupation if req.occupation is not None else "Unknown"
    homeowner_status_value = req.homeowner_status if req.homeowner_status is not None else "renter"

    history_source = None
    history_store_path = None
    history_window_start = None
    history_window_end = None

    if mode == "coldstart":
        pred = predict_coldstart_dayahead(
            artifacts=artifacts,
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
            save_csv=req.save_csv,
            out_dir=OUT_DIR,
            out_name=f"pred_{req.model_id}_{mode}_{req.city}_{req.target_date}.csv".replace(" ", "_"),
        )

    elif mode == "withhistory":
        if req.history_consumption_Wh is not None:
            history = _normalize_user_history(req.history_consumption_Wh, req.min_history_hours)
            history_source = "user"
        else:
            if not req.use_proxy_history:
                raise HTTPException(
                    status_code=422,
                    detail="For withhistory give history_consumption_Wh or enable use_proxy_history.",
                )

            try:
                history, hs, ws, we = get_history_from_csv(
                    target_ts=f"{req.target_date} 00:00:00",
                    history_hours=req.min_history_hours,
                    fallback_to_latest=True,
                )
            except Exception as e:
                raise HTTPException(status_code=422, detail=f"CSV history failed: {e}")

            history = _normalize_user_history(history, req.min_history_hours)
            history_source = hs
            info = history_store_info()
            history_store_path = info.get("history_csv_path")
            history_window_start = ws
            history_window_end = we

        pred = predict_withhistory_dayahead(
            artifacts=artifacts,
            target_date=req.target_date,
            internal_temperature_24h=internal_temperature_24h,
            external_temperature_24h=external_temperature_24h,
            internal_humidity_24h=internal_humidity_24h,
            external_humidity_24h=external_humidity_24h,
            history_consumption_Wh=history,
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
            min_history_hours=req.min_history_hours,
            save_csv=req.save_csv,
            out_dir=OUT_DIR,
            out_name=f"pred_{req.model_id}_{mode}_{req.city}_{req.target_date}.csv".replace(" ", "_"),
        )

    else:
        raise HTTPException(status_code=500, detail=f"Invalid mode: {mode}")

    city_label = None
    if req.city:
        city_key = req.city.strip().lower()
        if city_key in SUPPORTED_CITIES:
            city_label = SUPPORTED_CITIES[city_key]["label"]

    day_info = _time_features(pd.Timestamp(req.target_date))

    return {
        "model_id": req.model_id,
        "model_name": model_cfg["name"],
        "model_type": model_cfg["type"],
        "mode": mode,

        "city": req.city,
        "city_label": city_label,
        "target_date": req.target_date,

        "day_of_week": day_info["day_of_week"],
        "month": day_info["month"],
        "season": day_info["season"],
        "is_weekend": day_info["is_weekend"],
        "is_holiday": day_info["is_holiday"],

        "total_kWh_day": _total_kwh_from_preds(pred),

        "weather_source": weather_source,
        "internal_temperature_source": internal_temp_source,
        "internal_humidity_source": internal_hum_source,
        "external_humidity_source": external_hum_source,
        "t_min_used": tmin_used,
        "t_max_used": tmax_used,

        "external_temperature_24h": external_temperature_24h,
        "external_humidity_24h": external_humidity_24h,
        "internal_temperature_24h": internal_temperature_24h,
        "internal_humidity_24h": internal_humidity_24h,

        "occupation_used": occupation_value,
        "homeowner_status_used": homeowner_status_value,

        "history_source": history_source,
        "history_store_path": history_store_path,
        "history_window_start": history_window_start,
        "history_window_end": history_window_end,

        "predictions": [
            {
                "timestamp": t.isoformat(),
                "pred_consumption_Wh": float(v),
                "is_weekend": int(day_info["is_weekend"]),
                "is_holiday": int(day_info["is_holiday"]),
            }
            for t, v in zip(pred["timestamp"], pred["pred_consumption_Wh"])
        ],
    }