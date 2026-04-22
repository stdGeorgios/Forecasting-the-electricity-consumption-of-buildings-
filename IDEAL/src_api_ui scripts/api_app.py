# C:/IDEAL_Programming/src/api_app.py
# ---------------------------------------------------------
# FastAPI app for IDEAL forecasting demo
# Unified endpoint:
# POST /predict
# GET /models
# GET /health
# GET /history/info
#
# Supports:
# - Multiple model families via model_registry.py (RF, XGB, ...)
# - Separate prediction mode: coldstart / withhistory
# - Auto external temperature via Open-Meteo, fallback to tmin/tmax
# - With-history: history from ONE CSV file (history_store.csv)
# OR user-provided history_consumption_Wh
# - Holidays flag computed from holidays.csv
#
# Returns external_temperature_24h always.
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
    load_model_artifacts, # <-- generic loader
    predict_coldstart_dayahead,
    predict_withhistory_dayahead,
)

from src.history_store import get_history_from_csv, history_store_info
from src.model_registry import MODEL_REGISTRY, DEFAULT_MODEL_ID, DEFAULT_MODE


# =========================
# Paths / Config
# =========================
BASE_DIR = Path(os.getenv("IDEAL_BASE_DIR", "C:/IDEAL_Programming"))

OUT_DIR = Path(os.getenv("IDEAL_API_OUT_DIR", str(BASE_DIR / "processed" / "predictions" / "api_like")))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Holidays CSV candidates
HOLIDAYS_CANDIDATES = [
    Path(os.getenv("HOLIDAYS_PATH", "")) if os.getenv("HOLIDAYS_PATH") else None,
    BASE_DIR / "holidays.csv",
    BASE_DIR / "metadata" / "holidays.csv",
    BASE_DIR / "processed" / "metadata" / "holidays.csv",
    BASE_DIR / "processed" / "holidays.csv",
]
HOLIDAYS_CANDIDATES = [p for p in HOLIDAYS_CANDIDATES if p is not None and str(p) != ""]


# =========================
# Holidays helper
# =========================
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


# =========================
# Weather helpers
# =========================
def _geocode_city(city: str) -> Optional[Dict[str, float]]:
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": city, "count": 1, "language": "en", "format": "json"}
    r = requests.get(url, params=params, timeout=20)
    if r.status_code != 200:
        return None
    js = r.json()
    if "results" not in js or not js["results"]:
        return None
    lat = float(js["results"][0]["latitude"])
    lon = float(js["results"][0]["longitude"])
    return {"lat": lat, "lon": lon}


def fetch_openmeteo_hourly_temp(city: str, target_date: str) -> Optional[List[float]]:
    geo = _geocode_city(city)
    if geo is None:
        return None

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": geo["lat"],
        "longitude": geo["lon"],
        "hourly": "temperature_2m",
        "timezone": "Europe/London",
        "start_date": str(pd.Timestamp(target_date).date()),
        "end_date": str(pd.Timestamp(target_date).date()),
    }
    r = requests.get(url, params=params, timeout=30)
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
            "Δεν βρέθηκαν αυτόματα δεδομένα εξωτερικής θερμοκρασίας για αυτή την ημερομηνία. "
            "Δώσε είτε t_min / t_max είτε external_temperature_24h (24 τιμές)."
        ),
    )


# =========================
# App / Startup
# =========================
app = FastAPI(title="IDEAL Load Forecasting API", version="0.4.0")

# cache loaded artifacts by model_id
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
        raise HTTPException(status_code=500, detail=f"Failed to load artifacts for model '{model_id}': {e}")

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
        ]
    }


@app.get("/history/info")
def history_info():
    return history_store_info()


# =========================
# Request model
# =========================
class PredictRequest(BaseModel):
    model_id: str = Field(default=DEFAULT_MODEL_ID, example="rf")
    mode: Literal["coldstart", "withhistory"] = Field(default=DEFAULT_MODE, example="coldstart")

    city: str = Field(..., example="Edinburgh")
    target_date: str = Field(..., example="2026-02-21")

    total_floor_area_m2: float = Field(..., example=85, ge=0)
    residents: float = Field(..., example=2, ge=0)
    hometype: str = Field(..., example="flat")
    urban_rural_class: str = Field(..., example="1")

    external_temperature_24h: Optional[List[float]] = None
    t_min: Optional[float] = None
    t_max: Optional[float] = None

    # with-history sources
    # 1) If history_consumption_Wh is provided -> use it
    # 2) Else if use_proxy_history True -> use history_store.csv
    use_proxy_history: bool = True
    history_consumption_Wh: Optional[List[float]] = None
    min_history_hours: int = 168

    save_csv: bool = False


# =========================
# Utilities
# =========================
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
        raise HTTPException(status_code=422, detail="history_consumption_Wh must be numeric (list of numbers).")

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
            detail=f"Model '{req.model_id}' does not support mode '{mode}'. Supported: {model_cfg['supports']}",
        )

    artifacts = _get_artifacts_for_model(req.model_id)

    temps24, weather_source, tmin_used, tmax_used = resolve_external_temperature_24h(
        city=req.city,
        target_date=req.target_date,
        external_temperature_24h=req.external_temperature_24h,
        tmin=req.t_min,
        tmax=req.t_max,
    )

    history_source = None
    history_store_path = None
    history_window_start = None
    history_window_end = None

    if mode == "coldstart":
        pred = predict_coldstart_dayahead(
            artifacts=artifacts,
            target_date=req.target_date,
            external_temperature_24h=temps24,
            total_floor_area_m2=req.total_floor_area_m2,
            residents=req.residents,
            hometype=req.hometype,
            urban_rural_class=req.urban_rural_class,
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
                    detail="Για with history δώσε history_consumption_Wh ή ενεργοποίησε 'use_proxy_history' (CSV store).",
                )

            try:
                history, hs, ws, we = get_history_from_csv(
                    target_date=req.target_date,
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
            external_temperature_24h=temps24,
            history_consumption_Wh=history,
            total_floor_area_m2=req.total_floor_area_m2,
            residents=req.residents,
            hometype=req.hometype,
            urban_rural_class=req.urban_rural_class,
            min_history_hours=req.min_history_hours,
            save_csv=req.save_csv,
            out_dir=OUT_DIR,
            out_name=f"pred_{req.model_id}_{mode}_{req.city}_{req.target_date}.csv".replace(" ", "_"),
        )

    else:
        raise HTTPException(status_code=500, detail=f"Invalid mode: {mode}")

    return {
        "model_id": req.model_id,
        "model_name": model_cfg["name"],
        "model_type": model_cfg["type"],
        "mode": mode,

        "city": req.city,
        "target_date": req.target_date,
        "total_kWh_day": _total_kwh_from_preds(pred),

        "weather_source": weather_source,
        "t_min_used": tmin_used,
        "t_max_used": tmax_used,
        "external_temperature_24h": temps24,

        "history_source": history_source,
        "history_store_path": history_store_path,
        "history_window_start": history_window_start,
        "history_window_end": history_window_end,

        "derived_flags": _derived_flags_for_day(req.target_date),

        "predictions": [
            {"timestamp": t.isoformat(), "pred_consumption_Wh": float(v)}
            for t, v in zip(pred["timestamp"], pred["pred_consumption_Wh"])
        ],
    }
