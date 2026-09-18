# C:/Plegma_Programming/src/ui_app.py
# ---------------------------------------------------------
# Gradio UI for PLEGMA Forecasting API - Generic UI version
# ---------------------------------------------------------
#
# Final API/UI assumptions:
#   - The user does NOT provide home_id.
#   - model_id="auto" selects the correct default by scenario:
#       no_history   -> LGBM/no_history_simple
#       with_history -> LGBM/with_history_generic
#   - Optional comparison models are selected through the API/model registry.
#   - With-history requires recent hourly consumption history, default 168 hours.
#   - History can come from a user-selected single-home CSV path/upload,
#     manual comma-separated values, or the legacy history_store.csv fallback.
#   - Weather can come from Open-Meteo, manual t_min/t_max, or manual 24h vectors.
#
# PLEGMA ports are intentionally kept separate from IDEAL:
#   API_BASE = http://127.0.0.1:8001
#   UI_PORT  = 7861
# ---------------------------------------------------------

from __future__ import annotations

import os

# Must be set before importing gradio / requests.
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import datetime as dt
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

# ============================================================
# GRADIO 4.44.x COMPATIBILITY PATCH
# ============================================================

try:
    from gradio_client import utils as _gradio_client_utils

    _orig_json_schema_to_python_type = _gradio_client_utils._json_schema_to_python_type

    def _safe_json_schema_to_python_type(schema, defs=None):
        if isinstance(schema, bool):
            return "Any"

        if isinstance(schema, dict):
            schema = dict(schema)
            if isinstance(schema.get("additionalProperties"), bool):
                schema["additionalProperties"] = {"type": "object"}

        return _orig_json_schema_to_python_type(schema, defs)

    _gradio_client_utils._json_schema_to_python_type = _safe_json_schema_to_python_type

except Exception as exc:
    print(f"[WARN] Gradio schema compatibility patch not applied: {exc}")


# ============================================================
# CONFIG
# ============================================================

API_BASE = "http://127.0.0.1:8001"
UI_HOST = "127.0.0.1"
UI_PORT = 7861

BASE_DIR = Path("C:/Plegma_Programming")
UI_EXPORT_DIR = BASE_DIR / "predictions" / "ui_exports"
UI_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_UPLOAD_DIR = BASE_DIR / "processed" / "stores" / "selected_history_files"
HISTORY_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

TIMEOUT = 180

DEFAULT_CITIES = ["athens", "thessaloniki", "patra", "heraklion"]

SUPPORTED_CITIES = {
    "athens": {"label": "Athens", "latitude": 37.9838, "longitude": 23.7275},
    "thessaloniki": {"label": "Thessaloniki", "latitude": 40.6401, "longitude": 22.9444},
    "patra": {"label": "Patra", "latitude": 38.2466, "longitude": 21.7346},
    "heraklion": {"label": "Heraklion", "latitude": 35.3387, "longitude": 25.1442},
}

DEFAULT_BUILDING_TYPES = ["apartment", "detached_house"]
DEFAULT_BUILD_ERAS = ["1950_1970", "1970_1990", "1990_2010"]
DEFAULT_HEATING_TYPES = [
    "radiator_oil",
    "radiator_gas",
    "air_conditioner",
    "air_to_air_heat_pump",
    "portable_electric_heaters",
]
DEFAULT_WATER_HEATER_TYPES = ["electric_boiler", "gas_boiler", "solar_boiler"]
DEFAULT_YEARS_IN_HOUSE = ["1_to_2_years", "3_to_4_years", "gt_5_years"]
DEFAULT_INCOME_BANDS = ["unknown", "low", "medium", "high"]
DEFAULT_HOMEOWNER_STATUS = ["renter", "owner", "unknown"]
DEFAULT_OCCUPATION = ["unknown", "employed", "student", "retired", "unemployed"]

# PLEGMA default ensemble weights. The user can override these in the UI.
DEFAULT_ENSEMBLE_WEIGHTS = {
    "rf": 0.25,
    "xgb": 0.35,
    "lgbm": 0.40,
}

MODE_NO_HISTORY = "no_history"
MODE_WITH_HISTORY = "with_history"

# Legacy names kept only for internal compatibility with old wording.
MODE_COLDSTART = MODE_NO_HISTORY
MODE_WITHHISTORY = MODE_WITH_HISTORY


# ============================================================
# DEFAULT ENVIRONMENTAL PROFILES
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


def _list24_to_text(vals: List[float]) -> str:
    return ", ".join(f"{float(v):.1f}" for v in vals)


# ============================================================
# BASIC HELPERS
# ============================================================

def _today_date() -> dt.date:
    return dt.date.today()


def _now_local() -> dt.datetime:
    return dt.datetime.now()


def _parse_date(s: str) -> dt.date:
    return dt.date.fromisoformat(str(s).strip())


def _next_full_hour(now: dt.datetime) -> dt.datetime:
    floor = now.replace(minute=0, second=0, microsecond=0)
    if now == floor:
        return floor
    return floor + dt.timedelta(hours=1)


def _api_health() -> str:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        if r.status_code == 200:
            js = r.json()
            ready = js.get("num_ready_models")
            nohist = js.get("default_no_history_model") or js.get("default_coldstart_model")
            hist = js.get("default_with_history_model") or js.get("default_withhistory_model")
            return f"✅ API: OK | Ready models: {ready} | No-history: {nohist} | With-history: {hist}"
        return f"⚠️ API: HTTP {r.status_code}"
    except Exception:
        return "❌ API: not reachable. Start FastAPI first on port 8001."


def _fetch_models_response(mode: Optional[str] = None) -> Dict[str, Any]:
    try:
        params = {"mode": mode} if mode else None
        r = requests.get(f"{API_BASE}/models", params=params, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def _fetch_model_options(mode: Optional[str] = None) -> List[Dict[str, Any]]:
    js = _fetch_models_response(mode=mode)
    options = js.get("options_for_ui") or []

    if options:
        return options

    # Fallback when API is not running yet.
    if mode == MODE_WITH_HISTORY:
        return [
            {"label": "Auto default", "value": "auto", "is_default": True},
            {"label": "LGBM with-history default", "value": "lgbm_with_history_default"},
            {"label": "XGB with-history optional", "value": "xgb_with_history_optional"},
            {"label": "RF with-history optional", "value": "rf_with_history_optional"},
        ]

    return [
        {"label": "Auto default", "value": "auto", "is_default": True},
        {"label": "LGBM no-history default", "value": "lgbm_no_history_default"},
        {"label": "XGB no-history optional", "value": "xgb_no_history_optional"},
        {"label": "RF no-history optional", "value": "rf_no_history_optional"},
    ]


def _option_to_choice(option: Dict[str, Any]) -> str:
    value = str(option.get("value", "auto"))
    label = str(option.get("label", value))
    marker = ""
    if option.get("is_default") and value != "auto":
        marker = " [default]"
    elif option.get("is_optional"):
        marker = " [optional]"
    return f"{value} — {label}{marker}"


def _choice_to_model_id(choice: str) -> str:
    if not choice:
        return "auto"
    return str(choice).split("—", 1)[0].strip()


def _model_choices_for_mode(mode: str) -> List[str]:
    return [_option_to_choice(o) for o in _fetch_model_options(mode=mode)]


def _default_choice_for_mode(mode: str) -> str:
    choices = _model_choices_for_mode(mode)
    return choices[0] if choices else "auto — Auto default"


def _needs_weather_fallback(detail_text: str) -> bool:
    t = (detail_text or "").lower()
    return (
        "external temperature" in t
        or "external_temperature_24h" in t
        or "t_min" in t
        or "t_max" in t
        or "could not resolve external_temperature_24h" in t
        or "δώσε είτε" in t
    )


def _normalize_temp_inputs(t_min: Optional[float], t_max: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
    try:
        if t_min is not None:
            t_min = float(t_min)
        if t_max is not None:
            t_max = float(t_max)
    except Exception:
        return None, None

    # Some Gradio environments return hidden Number fields as 0 / 0.0.
    if t_min == 0 and t_max == 0:
        return None, None

    if (t_min is None) != (t_max is None):
        return None, None

    return t_min, t_max


def _validate_inputs(
    target_date: str,
    num_rooms: float,
    residents: float,
    min_history_hours: int,
    t_min: Optional[float],
    t_max: Optional[float],
) -> Optional[str]:
    try:
        _ = _parse_date(target_date)
    except Exception:
        return "Σφάλμα: target_date πρέπει να είναι σε μορφή YYYY-MM-DD."

    # Past dates are intentionally allowed for backtesting/evaluation.
    if num_rooms is None or float(num_rooms) < 0:
        return "Σφάλμα: num_rooms πρέπει να είναι >= 0."

    if residents is None or float(residents) < 0:
        return "Σφάλμα: residents πρέπει να είναι >= 0."

    if min_history_hours is None or int(min_history_hours) <= 0:
        return "Σφάλμα: min_history_hours πρέπει να είναι > 0."

    if t_min is not None and t_max is not None:
        try:
            tmin = float(t_min)
            tmax = float(t_max)
        except Exception:
            return "Σφάλμα: t_min / t_max πρέπει να είναι αριθμοί."

        if tmin >= tmax:
            return "Σφάλμα: t_min πρέπει να είναι μικρότερο από t_max."

    return None


def _parse_csv_float_list(text: str, name: str, expected_len: Optional[int] = None) -> Optional[List[float]]:
    if not text or not str(text).strip():
        return None

    try:
        vals = [float(x.strip()) for x in str(text).split(",") if x.strip() != ""]
    except Exception:
        raise ValueError(f"Σφάλμα: {name} δεν είναι έγκυρη λίστα αριθμών comma-separated.")

    if expected_len is not None and len(vals) != expected_len:
        raise ValueError(f"Σφάλμα: {name} πρέπει να έχει {expected_len} τιμές. Έχει {len(vals)}.")

    return vals


def _extract_uploaded_file_path(file_obj: Any) -> Optional[str]:
    if file_obj is None:
        return None

    if isinstance(file_obj, str):
        return file_obj

    if isinstance(file_obj, dict):
        return file_obj.get("name") or file_obj.get("path") or file_obj.get("orig_name")

    return getattr(file_obj, "name", None) or getattr(file_obj, "path", None)


def register_history_csv_upload(file_obj: Any) -> Tuple[str, str]:
    """Copy a selected single-home history CSV to a stable PLEGMA project folder."""
    src_path = _extract_uploaded_file_path(file_obj)

    if not src_path or not str(src_path).strip():
        return "", "No history CSV selected."

    src = Path(str(src_path).strip())
    if not src.exists():
        return "", f"Selected file does not exist: {src}"

    if src.suffix.lower() != ".csv":
        return "", "Please select a CSV file."

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_stem = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in src.stem)
    dst = HISTORY_UPLOAD_DIR / f"{safe_stem}_{timestamp}.csv"

    try:
        shutil.copy2(src, dst)
    except Exception as exc:
        return "", f"Could not copy selected file: {exc}"

    return str(dst), f"Selected history CSV: {dst}"


def _resolve_history_csv_path(history_csv_file: Any, history_csv_path_text: str) -> Optional[str]:
    if history_csv_path_text and str(history_csv_path_text).strip():
        return str(history_csv_path_text).strip().strip('"').strip("'")

    file_path = _extract_uploaded_file_path(history_csv_file)
    if file_path and str(file_path).strip():
        return str(file_path).strip().strip('"').strip("'")

    return None


# ============================================================
# HISTORY STABILITY / MAX ALPHA RECOMMENDATION
# ============================================================

def _clamp01(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, x))


def _score_from_cv(cv_value: Optional[float], cv_bad: float) -> float:
    if cv_value is None or pd.isna(cv_value):
        return 0.50
    try:
        cv = float(cv_value)
    except Exception:
        return 0.50
    if cv < 0:
        return 0.50
    return _clamp01(1.0 - (cv / float(cv_bad)))


def _safe_cv(series: pd.Series) -> Optional[float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) <= 1:
        return None
    mean_val = float(s.mean())
    if abs(mean_val) < 1e-9:
        return None
    return float(s.std(ddof=0) / mean_val)


def _resolve_effective_history_alpha(
    use_history: bool,
    use_recommended_history_alpha: bool,
    recommended_history_correction_max_alpha: Optional[float],
    manual_history_correction_max_alpha: float,
) -> float:
    try:
        manual_alpha = float(manual_history_correction_max_alpha)
    except Exception:
        manual_alpha = 0.20

    if not use_history or not use_recommended_history_alpha:
        return manual_alpha

    if recommended_history_correction_max_alpha is None or str(recommended_history_correction_max_alpha).strip() == "":
        raise ValueError(
            "Έχεις ενεργό το 'Use recommended max alpha', αλλά δεν υπάρχει έγκυρη σύσταση. "
            "Πάτησε πρώτα Analyze history stability ή απενεργοποίησε το checkbox."
        )

    try:
        return float(recommended_history_correction_max_alpha)
    except Exception:
        raise ValueError("Η προτεινόμενη τιμή max_alpha δεν είναι έγκυρη. Πάτησε ξανά Analyze history stability.")


def _history_dataframe_from_inputs(
    history_csv_file: Any,
    history_csv_path_text: str,
    history_consumption_text: str,
    target_date: str,
    history_days: int,
) -> Tuple[pd.DataFrame, str]:
    target_start = pd.Timestamp(_parse_date(target_date))

    manual_values = _parse_csv_float_list(history_consumption_text, "history_consumption_Wh", expected_len=None)
    if manual_values is not None:
        if len(manual_values) < 1:
            raise ValueError("Το manual history_consumption_Wh είναι κενό.")
        start = target_start - pd.Timedelta(hours=len(manual_values))
        timestamps = pd.date_range(start, periods=len(manual_values), freq="h")
        hist = pd.DataFrame({"timestamp": timestamps, "consumption_Wh": [float(x) for x in manual_values]})
        return hist, "manual history_consumption_Wh"

    history_csv_path = _resolve_history_csv_path(history_csv_file, history_csv_path_text)
    if not history_csv_path:
        raise ValueError("Δεν έχει επιλεγεί history CSV και δεν δόθηκε manual history_consumption_Wh.")

    path = Path(str(history_csv_path).strip().strip('"').strip("'"))
    if not path.exists():
        raise ValueError(f"Δεν βρέθηκε το history CSV: {path}")

    hist = pd.read_csv(path, low_memory=False)
    col_map = {str(c).lower(): c for c in hist.columns}
    ts_col = col_map.get("timestamp") or col_map.get("datetime") or col_map.get("date_time") or col_map.get("time")
    val_col = (
        col_map.get("consumption_wh")
        or col_map.get("actual_consumption_wh")
        or col_map.get("pred_consumption_wh")
        or col_map.get("value")
        or col_map.get("wh")
    )

    if ts_col is None:
        raise ValueError(f"Το history CSV πρέπει να έχει στήλη timestamp. Columns: {list(hist.columns)}")
    if val_col is None:
        raise ValueError(f"Το history CSV πρέπει να έχει στήλη consumption_Wh. Columns: {list(hist.columns)}")

    hist = hist[[ts_col, val_col]].copy().rename(columns={ts_col: "timestamp", val_col: "consumption_Wh"})
    hist["timestamp"] = pd.to_datetime(hist["timestamp"], errors="coerce")
    hist["consumption_Wh"] = pd.to_numeric(hist["consumption_Wh"], errors="coerce")
    hist = hist.dropna(subset=["timestamp", "consumption_Wh"])
    hist["consumption_Wh"] = hist["consumption_Wh"].clip(lower=0)
    hist = hist.sort_values("timestamp").reset_index(drop=True)

    if hist.empty:
        raise ValueError("Το history CSV δεν περιέχει έγκυρες γραμμές.")

    return hist, str(path)


def analyze_history_stability_recommendation(
    history_csv_file: Any,
    history_csv_path_text: str,
    target_date: str,
    history_correction_days: int,
    min_history_hours: int,
    history_consumption_text: str,
):
    try:
        history_days = int(history_correction_days or 7)
        if history_days <= 0:
            history_days = 7

        target_start = pd.Timestamp(_parse_date(target_date))
        window_start = target_start - pd.Timedelta(days=history_days)
        expected_hours = history_days * 24

        hist, source = _history_dataframe_from_inputs(
            history_csv_file=history_csv_file,
            history_csv_path_text=history_csv_path_text,
            history_consumption_text=history_consumption_text,
            target_date=target_date,
            history_days=history_days,
        )

        hist = hist[hist["timestamp"] < target_start].copy()
        window_hist = hist[(hist["timestamp"] >= window_start) & (hist["timestamp"] < target_start)].copy()
        if window_hist.empty:
            window_hist = hist.tail(expected_hours).copy()
        if window_hist.empty:
            raise ValueError("Δεν υπάρχουν διαθέσιμες ιστορικές τιμές πριν από την target date.")

        window_hist = window_hist.groupby("timestamp", as_index=False).agg({"consumption_Wh": "mean"}).sort_values("timestamp")
        expected_index = pd.date_range(window_start, target_start - pd.Timedelta(hours=1), freq="h")
        aligned = window_hist.set_index("timestamp").reindex(expected_index)
        aligned.index.name = "timestamp"
        aligned = aligned.reset_index()

        observed_hours = int(aligned["consumption_Wh"].notna().sum())
        missing_hours = int(aligned["consumption_Wh"].isna().sum())
        completeness_score = observed_hours / expected_hours if expected_hours > 0 else 0.0

        aligned["consumption_filled_Wh"] = (
            aligned["consumption_Wh"]
            .interpolate(method="linear", limit_direction="both")
            .ffill()
            .bfill()
        )
        if aligned["consumption_filled_Wh"].isna().all():
            raise ValueError("Δεν μπορεί να υπολογιστεί σταθερότητα: όλες οι τιμές είναι κενές.")

        aligned["date"] = aligned["timestamp"].dt.date
        aligned["hour"] = aligned["timestamp"].dt.hour
        aligned["is_weekend"] = (aligned["timestamp"].dt.weekday >= 5).astype(int)

        daily = aligned.groupby("date", as_index=False).agg(
            daily_kWh=("consumption_filled_Wh", lambda s: float(pd.to_numeric(s, errors="coerce").sum() / 1000.0)),
            daily_mean_Wh=("consumption_filled_Wh", "mean"),
            daily_peak_Wh=("consumption_filled_Wh", "max"),
            observed_hours=("consumption_Wh", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
            is_weekend=("is_weekend", "max"),
        )

        daily_kwh_values = pd.to_numeric(daily["daily_kWh"], errors="coerce").dropna()
        mean_daily_kWh = float(daily_kwh_values.mean()) if not daily_kwh_values.empty else float("nan")
        median_daily_kWh = float(daily_kwh_values.median()) if not daily_kwh_values.empty else float("nan")
        min_daily_kWh = float(daily_kwh_values.min()) if not daily_kwh_values.empty else float("nan")
        max_daily_kWh = float(daily_kwh_values.max()) if not daily_kwh_values.empty else float("nan")

        daily_cv = _safe_cv(daily["daily_kWh"])
        daily_score = _score_from_cv(daily_cv, cv_bad=0.35)

        pivot = aligned.pivot_table(index="date", columns="hour", values="consumption_filled_Wh", aggfunc="mean")
        hourly_cvs = []
        for hour in range(24):
            if hour in pivot.columns:
                cv = _safe_cv(pivot[hour])
                if cv is not None and not pd.isna(cv):
                    hourly_cvs.append(float(cv))
        hourly_profile_cv = float(pd.Series(hourly_cvs).median()) if hourly_cvs else None
        hourly_score = _score_from_cv(hourly_profile_cv, cv_bad=0.75)

        daily["peak_ratio"] = daily["daily_peak_Wh"] / daily["daily_mean_Wh"].replace(0, pd.NA)
        peak_ratio_cv = _safe_cv(daily["peak_ratio"])
        peak_score = _score_from_cv(peak_ratio_cv, cv_bad=0.50)

        target_is_weekend = int(target_start.weekday() >= 5)
        same_type_days = int(daily[daily["is_weekend"] == target_is_weekend]["date"].nunique())
        if target_is_weekend:
            day_type_score = 1.0 if same_type_days >= 2 else (0.75 if same_type_days == 1 else 0.50)
        else:
            day_type_score = 1.0 if same_type_days >= 3 else (0.75 if same_type_days >= 1 else 0.50)

        base_score = 0.30 * daily_score + 0.35 * hourly_score + 0.20 * peak_score + 0.15 * completeness_score
        stability_score = _clamp01(base_score * (0.85 + 0.15 * day_type_score))

        if completeness_score < 0.85 or observed_hours < min(int(min_history_hours or expected_hours), expected_hours) * 0.80:
            recommended_alpha = 0.05
            category = "Low"
            alpha_range = "0.00–0.10"
            reason_alpha = "Το ιστορικό δεν είναι αρκετά πλήρες. Προτείνεται πολύ χαμηλό max_alpha."
        elif stability_score >= 0.93:
            recommended_alpha = 0.60
            category = "Excellent"
            alpha_range = "0.55–0.65"
            reason_alpha = "Το ιστορικό είναι εξαιρετικά σταθερό. Επιτρέπεται ισχυρότερη history-aware correction."
        elif stability_score >= 0.80:
            recommended_alpha = 0.40
            category = "Very High"
            alpha_range = "0.35–0.50"
            reason_alpha = "Το ιστορικό είναι πολύ σταθερό και επαναλαμβανόμενο."
        elif stability_score >= 0.60:
            recommended_alpha = 0.25
            category = "High"
            alpha_range = "0.20–0.30"
            reason_alpha = "Το ιστορικό είναι σταθερό. Προτείνεται μέτρια correction."
        elif stability_score >= 0.40:
            recommended_alpha = 0.10
            category = "Medium"
            alpha_range = "0.10–0.15"
            reason_alpha = "Το ιστορικό έχει μέτρια σταθερότητα. Προτείνεται χαμηλό max_alpha."
        else:
            recommended_alpha = 0.05
            category = "Low"
            alpha_range = "0.00–0.10"
            reason_alpha = "Το ιστορικό είναι ασταθές ή έχει μη επαναλαμβανόμενα peaks."

        explanation = (
            f"History source: {source}\n"
            f"Target date: {target_start.date().isoformat()}\n"
            f"History window checked: {window_start} έως {target_start} (exclusive)\n"
            f"Expected hours: {expected_hours} | Observed hours: {observed_hours} | Missing hours: {missing_hours}\n"
            f"Completeness score: {completeness_score:.3f}\n"
            f"Mean daily consumption: {mean_daily_kWh:.3f} kWh/day | Median: {median_daily_kWh:.3f} kWh/day\n"
            f"Daily consumption range: {min_daily_kWh:.3f}–{max_daily_kWh:.3f} kWh/day\n"
            f"Daily total CV: {daily_cv if daily_cv is not None else float('nan'):.3f} | Daily stability score: {daily_score:.3f}\n"
            f"Hourly profile median CV: {hourly_profile_cv if hourly_profile_cv is not None else float('nan'):.3f} | Hourly stability score: {hourly_score:.3f}\n"
            f"Peak-ratio CV: {peak_ratio_cv if peak_ratio_cv is not None else float('nan'):.3f} | Peak stability score: {peak_score:.3f}\n"
            f"Same day-type days in history: {same_type_days} | Day-type score: {day_type_score:.3f}\n"
            f"Recommended max_alpha range for category: {alpha_range}\n"
            f"Recommendation reason: {reason_alpha}"
        )

        status = (
            f"✅ History stability analyzed. Category={category}. "
            f"Recommended max_alpha={recommended_alpha:.2f} (range {alpha_range})."
        )

        return f"{stability_score:.3f}", category, float(recommended_alpha), explanation, status

    except Exception as exc:
        return "", "Not available", None, f"Could not analyze history stability: {exc}", "⚠️ History stability recommendation not available."


def _reset_history_stability_recommendation(*_):
    return (
        "",
        "Outdated / not analyzed",
        None,
        "History stability recommendation is outdated. Click Analyze history stability again.",
        "History stability recommendation needs refresh.",
        gr.update(value=False),
    )


# ============================================================
# BEHAVIORAL ADJUSTMENT
# ============================================================

def _parse_behavior_hours(text: str) -> List[int]:
    if text is None:
        return []

    text = str(text).strip()
    if text == "":
        return []

    hours = set()
    parts = [p.strip() for p in text.split(",") if p.strip()]

    for part in parts:
        if "-" in part:
            start_s, end_s = [x.strip() for x in part.split("-", 1)]
            start = int(start_s)
            end = int(end_s)
            if not (0 <= start <= 23 and 0 <= end <= 24):
                raise ValueError("Τα διαστήματα πρέπει να είναι μέσα στο 0-24.")
            if end <= start:
                raise ValueError("Σε κάθε διάστημα πρέπει να ισχύει τέλος > αρχή, π.χ. 18-23.")
            for h in range(start, end):
                if 0 <= h <= 23:
                    hours.add(h)
        else:
            h = int(part)
            if not (0 <= h <= 23):
                raise ValueError("Οι ώρες πρέπει να είναι μέσα στο 0-23.")
            hours.add(h)

    return sorted(hours)


def _normalize_behavior_inputs(
    enable_behavior_adjustment: bool,
    high_consumption_hours_text: str,
    behavior_factor: Optional[float],
) -> Tuple[bool, List[int], float]:
    if not enable_behavior_adjustment:
        return False, [], 1.0

    try:
        factor = 1.0 if behavior_factor is None else float(behavior_factor)
    except Exception:
        raise ValueError("Το behavior factor πρέπει να είναι αριθμός.")

    if factor <= 0:
        raise ValueError("Το behavior factor πρέπει να είναι > 0.")

    hours = _parse_behavior_hours(high_consumption_hours_text)
    if len(hours) == 0:
        return False, [], 1.0

    return True, hours, factor


def _apply_behavior_adjustment_single_df(
    df: pd.DataFrame,
    hours: List[int],
    factor: float,
    value_col: str = "pred_consumption_Wh",
) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out["hour"] = out["timestamp"].dt.hour
    out["behavior_adjusted"] = out["hour"].isin(hours).astype(int)
    out["behavior_factor"] = out["behavior_adjusted"].apply(lambda x: factor if x == 1 else 1.0)
    out[value_col] = out[value_col] * out["behavior_factor"]
    return out


def _apply_behavior_adjustment_curve_df(curve_df: pd.DataFrame, hours: List[int], factor: float, target_cols: List[str]) -> pd.DataFrame:
    if curve_df is None or curve_df.empty:
        return curve_df

    out = curve_df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out["hour"] = out["timestamp"].dt.hour
    mask = out["hour"].isin(hours)

    for col in target_cols:
        if col in out.columns:
            out.loc[mask, col] = out.loc[mask, col] * factor

    out["behavior_adjusted"] = mask.astype(int)
    out["behavior_factor"] = out["behavior_adjusted"].apply(lambda x: factor if x == 1 else 1.0)
    return out


def _behavior_summary_suffix(enabled: bool, hours: List[int], factor: float) -> str:
    if not enabled:
        return "Behavioral adjustment: OFF"

    if factor > 1:
        effect = f"increase by {(factor - 1) * 100:.1f}%"
    elif factor < 1:
        effect = f"decrease by {(1 - factor) * 100:.1f}%"
    else:
        effect = "no change"

    return (
        "Behavioral adjustment: ON\n"
        f"Adjusted hours: {hours}\n"
        f"Behavior factor: {factor:.2f} ({effect})"
    )


# ============================================================
# ENSEMBLE HELPERS
# ============================================================

def _resolve_ensemble_weights(
    use_custom_weights: bool,
    rf_weight_pct: Optional[float],
    xgb_weight_pct: Optional[float],
    lgbm_weight_pct: Optional[float],
) -> Dict[str, float]:
    if not use_custom_weights:
        return dict(DEFAULT_ENSEMBLE_WEIGHTS)

    try:
        rf = float(rf_weight_pct if rf_weight_pct is not None else 0.0)
        xgb = float(xgb_weight_pct if xgb_weight_pct is not None else 0.0)
        lgbm = float(lgbm_weight_pct if lgbm_weight_pct is not None else 0.0)
    except Exception:
        raise ValueError("Τα ensemble weights πρέπει να είναι αριθμοί.")

    if rf < 0 or xgb < 0 or lgbm < 0:
        raise ValueError("Τα ensemble weights δεν μπορούν να είναι αρνητικά.")

    total = rf + xgb + lgbm
    if abs(total - 100.0) > 1e-6:
        raise ValueError(f"Τα ensemble weights πρέπει να αθροίζουν σε 100%. Τρέχον άθροισμα: {total:.2f}%.")

    return {"rf": rf / 100.0, "xgb": xgb / 100.0, "lgbm": lgbm / 100.0}


def _weights_text(weights: Dict[str, float]) -> str:
    return f"Weights: RF={weights['rf']:.2f}, XGB={weights['xgb']:.2f}, LGBM={weights['lgbm']:.2f}"


def _comparison_model_ids(mode: str) -> Dict[str, str]:
    if mode == MODE_WITH_HISTORY:
        return {
            "rf": "rf_with_history_optional",
            "xgb": "xgb_with_history_optional",
            "lgbm": "lgbm_with_history_default",
        }

    return {
        "rf": "rf_no_history_optional",
        "xgb": "xgb_no_history_optional",
        "lgbm": "lgbm_no_history_default",
    }


# ============================================================
# SUMMARY / CSV / DISPLAY HELPERS
# ============================================================

def _flag_yes_no(value: Any) -> str:
    try:
        return "Yes" if int(value) == 1 else "No"
    except Exception:
        return "Unknown"


def _day_type_from_flags(flags: Dict[str, Any]) -> str:
    is_weekend = int(flags.get("is_weekend", 0) or 0)
    is_holiday = int(flags.get("is_holiday", 0) or 0)
    if is_weekend and is_holiday:
        return "Weekend and Holiday"
    if is_weekend:
        return "Weekend"
    if is_holiday:
        return "Holiday"
    return "Working day"


def _flags_summary_line(flags: Dict[str, Any]) -> str:
    return (
        f"Type of day: {_day_type_from_flags(flags)} | "
        f"Weekend: {_flag_yes_no(flags.get('is_weekend'))} | "
        f"Holiday: {_flag_yes_no(flags.get('is_holiday'))}"
    )


def _add_flag_columns_to_df(df: pd.DataFrame, flags: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    is_weekend = int(flags.get("is_weekend", 0) or 0)
    is_holiday = int(flags.get("is_holiday", 0) or 0)
    out["is_weekend"] = is_weekend
    out["is_holiday"] = is_holiday
    out["is_weekend_text"] = _flag_yes_no(is_weekend)
    out["is_holiday_text"] = _flag_yes_no(is_holiday)
    out["day_type"] = _day_type_from_flags(flags)
    return out


def _build_summary_single(out: Dict[str, Any], remaining_kwh_today: Optional[float] = None, cutoff_iso: Optional[str] = None, behavior_text: Optional[str] = None) -> str:
    flags = out.get("derived_flags", {}) or {}
    model_name = out.get("model_name") or out.get("label") or out.get("model_id")

    lines = [
        f"Requested model: {out.get('requested_model_id')}",
        f"Resolved model: {out.get('model_id')} — {model_name}",
        f"Mode: {out.get('mode')} | Optimization: {out.get('optimization')} | Variant: {out.get('prediction_variant')}",
        f"City: {out.get('city_label') or out.get('city')} | Date: {out.get('target_date')}",
        f"Season: {out.get('season')}",
        f"Total kWh/day (24h): {float(out.get('total_kWh_day', 0.0)):.3f}",
    ]

    if remaining_kwh_today is not None and cutoff_iso is not None:
        lines.append(f"Remaining kWh (from {cutoff_iso}): {float(remaining_kwh_today):.3f}")

    lines += [
        f"Weather source: {out.get('weather_source')}",
        f"External humidity source: {out.get('external_humidity_source')}",
        f"Internal temperature source: {out.get('internal_temperature_source')}",
        f"Internal humidity source: {out.get('internal_humidity_source')}",
        f"History source: {out.get('history_source')}",
        f"History hours used: {out.get('history_hours_used')}",
        f"History CSV path: {out.get('history_csv_path')}",
        f"Legacy history store path: {out.get('history_store_path')}",
        f"History correction applied: {out.get('history_correction_applied')}",
        f"History correction reason: {out.get('history_correction_reason')}",
        _flags_summary_line(flags),
    ]

    if behavior_text:
        lines += ["", behavior_text]

    return "\n".join(lines)


def _build_summary_compare(results: Dict[str, Dict[str, Any]], remaining_map: Dict[str, Optional[float]], cutoff_map: Dict[str, Optional[str]], behavior_text: Optional[str] = None) -> str:
    first = next(iter(results.values()))
    flags = first.get("derived_flags", {}) or {}
    lines = [
        f"City: {first.get('city_label') or first.get('city')} | Date: {first.get('target_date')}",
        f"Mode: {first.get('mode')} | Optimization: {first.get('optimization')}",
        _flags_summary_line(flags),
        "Model comparison:",
    ]

    for key, out in results.items():
        model_name = out.get("model_name") or out.get("label") or out.get("model_id")
        total_kwh = float(out.get("total_kWh_day", 0.0))
        rem = remaining_map.get(key)
        cutoff = cutoff_map.get(key)
        row = f"- {key.upper()} | {out.get('model_id')} — {model_name}: Total kWh/day = {total_kwh:.3f}"
        if rem is not None and cutoff is not None:
            row += f" | Remaining from {cutoff} = {float(rem):.3f}"
        lines.append(row)

    if behavior_text:
        lines += ["", behavior_text]

    return "\n".join(lines)


def _build_summary_combined(total_kwh_day: float, remaining_kwh_today: Optional[float], cutoff_iso: Optional[str], city: str, target_date: str, mode: str, optimization: str, weights: Dict[str, float], flags: Optional[Dict[str, Any]] = None, behavior_text: Optional[str] = None) -> str:
    flags = flags or {}
    lines = [
        "Combined prediction (weighted ensemble):",
        f"Mode: {mode} | Optimization: {optimization}",
        f"City: {city} | Date: {target_date}",
        _flags_summary_line(flags),
        f"Total kWh/day (24h): {float(total_kwh_day):.3f}",
        _weights_text(weights),
    ]

    if remaining_kwh_today is not None and cutoff_iso is not None:
        lines.append(f"Remaining kWh (from {cutoff_iso}): {float(remaining_kwh_today):.3f}")

    if behavior_text:
        lines += ["", behavior_text]

    return "\n".join(lines)


def _meteo24_to_df(temps24: Optional[List[float]], hum24: Optional[List[float]]) -> pd.DataFrame:
    if temps24 is None or len(temps24) != 24:
        return pd.DataFrame()
    out = {"hour": list(range(24)), "external_temp_C": [float(x) for x in temps24]}
    if hum24 is not None and len(hum24) == 24:
        out["external_humidity_pct"] = [float(x) for x in hum24]
    return pd.DataFrame(out)


def _compute_remaining_kwh_today(df_full: pd.DataFrame, target_date: str) -> Tuple[Optional[float], Optional[str]]:
    try:
        d = _parse_date(target_date)
    except Exception:
        return None, None

    if d != _today_date():
        return None, None

    if df_full is None or df_full.empty or "timestamp" not in df_full.columns or "pred_consumption_Wh" not in df_full.columns:
        return None, None

    cutoff = _next_full_hour(_now_local())
    cutoff_iso = cutoff.isoformat()

    df_tmp = df_full.copy()
    df_tmp["timestamp"] = pd.to_datetime(df_tmp["timestamp"])
    df_tmp = df_tmp[df_tmp["timestamp"] >= cutoff]

    if df_tmp.empty:
        return 0.0, cutoff_iso

    return float(df_tmp["pred_consumption_Wh"].sum() / 1000.0), cutoff_iso


def _compute_remaining_kwh_today_for_col(df_full: pd.DataFrame, target_date: str, value_col: str) -> Tuple[Optional[float], Optional[str]]:
    if df_full is None or df_full.empty or value_col not in df_full.columns:
        return None, None
    tmp = df_full.rename(columns={value_col: "pred_consumption_Wh"})[["timestamp", "pred_consumption_Wh"]].copy()
    return _compute_remaining_kwh_today(tmp, target_date)


def _filter_today_for_display(df_show: pd.DataFrame, meteo_df_show: pd.DataFrame, target_date: str):
    try:
        d = _parse_date(target_date)
        if d == _today_date() and not df_show.empty:
            cutoff = _next_full_hour(_now_local())
            df_show = df_show[df_show["timestamp"] >= cutoff].copy()
            if not meteo_df_show.empty:
                meteo_df_show = meteo_df_show[meteo_df_show["hour"] >= cutoff.hour].copy()
    except Exception:
        pass
    return df_show, meteo_df_show


def _filter_today_curve_only(df_show: pd.DataFrame, target_date: str):
    try:
        d = _parse_date(target_date)
        if d == _today_date() and not df_show.empty:
            cutoff = _next_full_hour(_now_local())
            df_show = df_show[df_show["timestamp"] >= cutoff].copy()
    except Exception:
        pass
    return df_show


def _save_csv_full_day(out: Dict[str, Any], preds_full: pd.DataFrame, temps24: Optional[List[float]], hum24: Optional[List[float]], remaining_kwh_today: Optional[float], cutoff_iso: Optional[str]) -> Path:
    df = preds_full.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if temps24 is not None and len(temps24) == 24 and len(df) == 24:
        df["external_temp_C"] = [float(x) for x in temps24]
    else:
        df["external_temp_C"] = pd.NA

    if hum24 is not None and len(hum24) == 24 and len(df) == 24:
        df["external_humidity_pct"] = [float(x) for x in hum24]
    else:
        df["external_humidity_pct"] = pd.NA

    for key in [
        "requested_model_id", "model_id", "model_name", "model_type", "mode", "optimization",
        "prediction_variant", "city", "city_label", "target_date", "weather_source",
        "external_humidity_source", "history_source", "history_csv_path", "history_store_path",
        "history_window_start", "history_window_end", "history_hours_used",
        "history_correction_applied", "history_correction_reason", "total_kWh_day",
    ]:
        df[key] = out.get(key)

    df = _add_flag_columns_to_df(df, out.get("derived_flags", {}) or {})
    df["remaining_kWh_today"] = remaining_kwh_today if remaining_kwh_today is not None else pd.NA
    df["cutoff_iso"] = cutoff_iso if cutoff_iso is not None else pd.NA

    safe_city = str(out.get("city", "city")).replace(" ", "_")
    safe_date = str(out.get("target_date", "date"))
    safe_mid = str(out.get("model_id", "model"))
    safe_mode = str(out.get("mode", "mode"))
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    fp = UI_EXPORT_DIR / f"ui_pred_{safe_mid}_{safe_mode}_{safe_city}_{safe_date}_{ts}.csv"
    df.to_csv(fp, index=False, encoding="utf-8")
    return fp


def _save_csv_compare(city: str, target_date: str, mode: str, optimization: str, curve_df: pd.DataFrame, comparison_df: pd.DataFrame, temps24: Optional[List[float]], hum24: Optional[List[float]], flags: Optional[Dict[str, Any]] = None) -> Path:
    flags = flags or {}
    curve = curve_df.copy()
    curve["timestamp"] = pd.to_datetime(curve["timestamp"])

    if temps24 is not None and len(temps24) == 24 and len(curve) == 24:
        curve["external_temp_C"] = [float(x) for x in temps24]
    else:
        curve["external_temp_C"] = pd.NA

    if hum24 is not None and len(hum24) == 24 and len(curve) == 24:
        curve["external_humidity_pct"] = [float(x) for x in hum24]
    else:
        curve["external_humidity_pct"] = pd.NA

    curve = _add_flag_columns_to_df(curve, flags)
    curve["mode"] = mode
    curve["optimization"] = optimization
    curve["city"] = city
    curve["target_date"] = target_date

    if comparison_df is not None and not comparison_df.empty:
        for _, row in comparison_df.iterrows():
            key = str(row.get("model_key", "")).strip()
            if key:
                curve[f"total_kWh_day_{key}"] = row.get("total_kWh_day")

    safe_city = str(city).replace(" ", "_")
    safe_date = str(target_date)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    fp = UI_EXPORT_DIR / f"ui_pred_compare_{mode}_{safe_city}_{safe_date}_{ts}.csv"
    curve.to_csv(fp, index=False, encoding="utf-8")
    return fp


def _save_csv_combined(city: str, target_date: str, mode: str, optimization: str, curve_df: pd.DataFrame, temps24: Optional[List[float]], hum24: Optional[List[float]], remaining_kwh_today: Optional[float], cutoff_iso: Optional[str], weights: Dict[str, float], flags: Optional[Dict[str, Any]] = None) -> Path:
    df = curve_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if temps24 is not None and len(temps24) == 24 and len(df) == 24:
        df["external_temp_C"] = [float(x) for x in temps24]
    else:
        df["external_temp_C"] = pd.NA

    if hum24 is not None and len(hum24) == 24 and len(df) == 24:
        df["external_humidity_pct"] = [float(x) for x in hum24]
    else:
        df["external_humidity_pct"] = pd.NA

    df = _add_flag_columns_to_df(df, flags or {})
    df["model_id"] = "combined"
    df["model_name"] = "Weighted Ensemble"
    df["model_type"] = "ensemble"
    df["mode"] = mode
    df["optimization"] = optimization
    df["city"] = city
    df["target_date"] = target_date
    df["ensemble_weight_rf"] = weights["rf"]
    df["ensemble_weight_xgb"] = weights["xgb"]
    df["ensemble_weight_lgbm"] = weights["lgbm"]
    df["total_kWh_day"] = float(df["ensemble_Wh"].sum() / 1000.0)
    df["remaining_kWh_today"] = remaining_kwh_today if remaining_kwh_today is not None else pd.NA
    df["cutoff_iso"] = cutoff_iso if cutoff_iso is not None else pd.NA

    safe_city = str(city).replace(" ", "_")
    safe_date = str(target_date)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    fp = UI_EXPORT_DIR / f"ui_pred_combined_{mode}_{safe_city}_{safe_date}_{ts}.csv"
    df.to_csv(fp, index=False, encoding="utf-8")
    return fp


# ============================================================
# CHARTS
# ============================================================

def _build_single_chart(df_show: pd.DataFrame):
    if df_show is None or df_show.empty:
        return None

    df = df_show.copy()
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    hours = list(range(24))
    base = pd.DataFrame({"hour": hours})
    plot_df = base.merge(df[["hour", "pred_consumption_Wh"]], on="hour", how="left")
    y = plot_df["pred_consumption_Wh"]

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.plot(plot_df["hour"], y, linewidth=2.5, marker="o", markersize=4, label="Prediction")
    if y.notna().any():
        ax.axhline(y.mean(), linestyle="--", linewidth=1, label="Daily average")
    ax.set_xticks(hours)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Predicted consumption (Wh)")
    ax.set_title("Predicted hourly electricity consumption")
    ax.grid(alpha=0.3)
    ax.legend()
    return fig


def _build_compare_chart(curve_df: pd.DataFrame):
    if curve_df is None or curve_df.empty:
        return None

    df = curve_df.copy()
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    hours = list(range(24))
    base = pd.DataFrame({"hour": hours})

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    for col in df.columns:
        if col in ["timestamp", "hour", "behavior_adjusted", "behavior_factor"]:
            continue
        if not col.endswith("_Wh"):
            continue
        temp = base.merge(df[["hour", col]], on="hour", how="left")
        ax.plot(temp["hour"], temp[col], linewidth=2, marker="o", markersize=4, label=col.replace("_Wh", "").upper())

    ax.set_xticks(hours)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Predicted consumption (Wh)")
    ax.set_title("Model comparison — hourly electricity prediction")
    ax.grid(alpha=0.3)
    ax.legend()
    return fig


# ============================================================
# PAYLOAD / API CALLS
# ============================================================

def _prepare_payload(
    use_history: bool,
    model_choice: str,
    optimization: str,
    city: str,
    target_date: str,
    internal_temperature_24h_text: str,
    external_temperature_24h_text: str,
    internal_humidity_24h_text: str,
    external_humidity_24h_text: str,
    t_min: Optional[float],
    t_max: Optional[float],
    num_rooms: float,
    residents: float,
    num_adults: Optional[float],
    num_children: Optional[float],
    num_elderly: Optional[float],
    has_ac: float,
    has_fridge_freezer: float,
    has_dryer: float,
    has_washing_machine: float,
    has_dishwasher: float,
    has_microwave: float,
    has_electric_oven: float,
    has_electric_hob: float,
    solar_panels: float,
    building_type: str,
    build_era: str,
    income_band: str,
    heating_type: str,
    water_heater_type: str,
    homeowner_status: str,
    years_in_house: str,
    occupation: str,
    history_csv_file: Any,
    history_csv_path_text: str,
    use_default_history_store: bool,
    min_history_hours: int,
    history_consumption_text: str,
    apply_history_correction: bool,
    history_correction_days: int,
    history_correction_max_alpha: float,
):
    t_min, t_max = _normalize_temp_inputs(t_min, t_max)

    err = _validate_inputs(
        target_date=target_date,
        num_rooms=num_rooms,
        residents=residents,
        min_history_hours=int(min_history_hours),
        t_min=t_min if (t_min is not None and t_max is not None) else None,
        t_max=t_max if (t_min is not None and t_max is not None) else None,
    )
    if err:
        raise ValueError(err)

    model_id = _choice_to_model_id(model_choice)
    mode = MODE_WITH_HISTORY if use_history else MODE_NO_HISTORY
    history_csv_path = _resolve_history_csv_path(history_csv_file, history_csv_path_text)

    if num_adults is None:
        num_adults = residents
    if num_children is None:
        num_children = 0.0
    if num_elderly is None:
        num_elderly = 0.0

    for name, val in {
        "has_ac": has_ac,
        "has_fridge_freezer": has_fridge_freezer,
        "has_dryer": has_dryer,
        "has_washing_machine": has_washing_machine,
        "has_dishwasher": has_dishwasher,
        "has_microwave": has_microwave,
        "has_electric_oven": has_electric_oven,
        "has_electric_hob": has_electric_hob,
        "solar_panels": solar_panels,
    }.items():
        if float(val) not in (0.0, 1.0):
            raise ValueError(f"Σφάλμα: το {name} πρέπει να είναι 0 ή 1.")

    payload: Dict[str, Any] = {
        "model_id": model_id,
        "mode": mode,
        "optimization": optimization,
        "city": city,
        "target_date": target_date,
        "num_rooms": float(num_rooms),
        "residents": float(residents),
        "num_adults": float(num_adults),
        "num_children": float(num_children),
        "num_elderly": float(num_elderly),
        "has_ac": float(has_ac),
        "has_fridge_freezer": float(has_fridge_freezer),
        "has_dryer": float(has_dryer),
        "has_washing_machine": float(has_washing_machine),
        "has_dishwasher": float(has_dishwasher),
        "has_microwave": float(has_microwave),
        "has_electric_oven": float(has_electric_oven),
        "has_electric_hob": float(has_electric_hob),
        "solar_panels": float(solar_panels),
        "building_type": str(building_type),
        "build_era": str(build_era),
        "income_band": str(income_band),
        "heating_type": str(heating_type),
        "water_heater_type": str(water_heater_type),
        "homeowner_status": str(homeowner_status),
        "years_in_house": str(years_in_house),
        "occupation": str(occupation),
        "save_csv": False,
        "min_history_hours": int(min_history_hours),
    }

    internal_temperature_24h = _parse_csv_float_list(internal_temperature_24h_text, "internal_temperature_24h", expected_len=24)
    external_temperature_24h = _parse_csv_float_list(external_temperature_24h_text, "external_temperature_24h", expected_len=24)
    internal_humidity_24h = _parse_csv_float_list(internal_humidity_24h_text, "internal_humidity_24h", expected_len=24)
    external_humidity_24h = _parse_csv_float_list(external_humidity_24h_text, "external_humidity_24h", expected_len=24)

    if internal_temperature_24h is not None:
        payload["internal_temperature_24h"] = internal_temperature_24h
    if external_temperature_24h is not None:
        payload["external_temperature_24h"] = external_temperature_24h
    elif t_min is not None and t_max is not None:
        payload["t_min"] = float(t_min)
        payload["t_max"] = float(t_max)
    if internal_humidity_24h is not None:
        payload["internal_humidity_24h"] = internal_humidity_24h
    if external_humidity_24h is not None:
        payload["external_humidity_24h"] = external_humidity_24h

    history_vals = _parse_csv_float_list(history_consumption_text, "history_consumption_Wh", expected_len=None)
    if use_history:
        payload["apply_history_correction"] = bool(apply_history_correction)
        payload["history_correction_days"] = int(history_correction_days)
        payload["history_correction_max_alpha"] = float(history_correction_max_alpha)

        if history_vals is not None:
            payload["history_consumption_Wh"] = history_vals
            payload["use_default_history_store"] = False
            payload["use_proxy_history"] = False
        elif history_csv_path:
            payload["history_csv_path"] = history_csv_path
            payload["use_default_history_store"] = False
            payload["use_proxy_history"] = False
        else:
            payload["use_default_history_store"] = bool(use_default_history_store)
            payload["use_proxy_history"] = bool(use_default_history_store)

    return payload, external_temperature_24h, external_humidity_24h


def _post_predict(payload: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(f"{API_BASE}/predict", json=payload, timeout=TIMEOUT)
    if r.status_code != 200:
        detail = ""
        try:
            js = r.json()
            detail = js.get("detail", "") if isinstance(js, dict) else str(js)
        except Exception:
            detail = r.text

        if r.status_code == 422 and _needs_weather_fallback(detail):
            raise RuntimeError(f"WEATHER_FALLBACK::{detail}")
        raise RuntimeError(f"API error ({r.status_code}): {detail}")
    return r.json()


def _predictions_to_df(out: Dict[str, Any]) -> pd.DataFrame:
    preds = out.get("predictions", []) or []
    df = pd.DataFrame(preds)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ============================================================
# BUTTON ACTIONS
# ============================================================

def do_predict(
    use_history: bool,
    model_choice: str,
    optimization: str,
    city: str,
    target_date: str,
    internal_temperature_24h_text: str,
    external_temperature_24h_text: str,
    internal_humidity_24h_text: str,
    external_humidity_24h_text: str,
    t_min: Optional[float],
    t_max: Optional[float],
    num_rooms: float,
    residents: float,
    num_adults: Optional[float],
    num_children: Optional[float],
    num_elderly: Optional[float],
    has_ac: float,
    has_fridge_freezer: float,
    has_dryer: float,
    has_washing_machine: float,
    has_dishwasher: float,
    has_microwave: float,
    has_electric_oven: float,
    has_electric_hob: float,
    solar_panels: float,
    building_type: str,
    build_era: str,
    income_band: str,
    heating_type: str,
    water_heater_type: str,
    homeowner_status: str,
    years_in_house: str,
    occupation: str,
    history_csv_file: Any,
    history_csv_path_text: str,
    use_default_history_store: bool,
    min_history_hours: int,
    history_consumption_text: str,
    apply_history_correction: bool,
    history_correction_days: int,
    history_correction_max_alpha: float,
    use_recommended_history_alpha: bool,
    recommended_history_correction_max_alpha: Optional[float],
    enable_behavior_adjustment: bool,
    high_consumption_hours_text: str,
    behavior_factor: Optional[float],
    use_custom_weights: bool,
    rf_weight_pct: Optional[float],
    xgb_weight_pct: Optional[float],
    lgbm_weight_pct: Optional[float],
    save_csv: bool,
):
    try:
        behavior_enabled, behavior_hours, behavior_factor_norm = _normalize_behavior_inputs(enable_behavior_adjustment, high_consumption_hours_text, behavior_factor)
        effective_alpha = _resolve_effective_history_alpha(use_history, use_recommended_history_alpha, recommended_history_correction_max_alpha, history_correction_max_alpha)

        payload, temps24_user, hum24_user = _prepare_payload(
            use_history, model_choice, optimization, city, target_date,
            internal_temperature_24h_text, external_temperature_24h_text,
            internal_humidity_24h_text, external_humidity_24h_text,
            t_min, t_max, num_rooms, residents, num_adults, num_children, num_elderly,
            has_ac, has_fridge_freezer, has_dryer, has_washing_machine, has_dishwasher,
            has_microwave, has_electric_oven, has_electric_hob, solar_panels,
            building_type, build_era, income_band, heating_type, water_heater_type,
            homeowner_status, years_in_house, occupation, history_csv_file, history_csv_path_text,
            use_default_history_store, min_history_hours, history_consumption_text,
            apply_history_correction, history_correction_days, effective_alpha,
        )

        out = _post_predict(payload)
        preds_full = _predictions_to_df(out)
        temps24 = out.get("external_temperature_24h") or temps24_user
        hum24 = out.get("external_humidity_24h") or hum24_user
        meteo_df_full = _meteo24_to_df(temps24, hum24)

        if behavior_enabled:
            preds_full = _apply_behavior_adjustment_single_df(preds_full, behavior_hours, behavior_factor_norm)

        remaining_kwh_today, cutoff_iso = _compute_remaining_kwh_today(preds_full[["timestamp", "pred_consumption_Wh"]], target_date)
        df_show = preds_full.copy()
        meteo_df_show = meteo_df_full.copy()
        df_show, meteo_df_show = _filter_today_for_display(df_show, meteo_df_show, target_date)

        out_for_summary = dict(out)
        out_for_summary["total_kWh_day"] = float(preds_full["pred_consumption_Wh"].sum() / 1000.0)
        behavior_text = _behavior_summary_suffix(behavior_enabled, behavior_hours, behavior_factor_norm)
        summary = _build_summary_single(out_for_summary, remaining_kwh_today, cutoff_iso, behavior_text)
        fig = _build_single_chart(df_show[["timestamp", "pred_consumption_Wh"]])

        status = "✅ Done"
        if save_csv and not preds_full.empty:
            fp = _save_csv_full_day(out_for_summary, preds_full, temps24, hum24, remaining_kwh_today, cutoff_iso)
            status = f"✅ Saved CSV (24h): {fp}"

        return summary, df_show, meteo_df_show, pd.DataFrame(), fig, status, gr.update(visible=False, value=None), gr.update(visible=False, value=None)

    except ValueError as e:
        return str(e), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None, "⚠️ Fix inputs and try again.", gr.update(visible=False, value=None), gr.update(visible=False, value=None)
    except RuntimeError as e:
        msg = str(e)
        if msg.startswith("WEATHER_FALLBACK::"):
            detail = msg.split("WEATHER_FALLBACK::", 1)[1]
            return (
                "⚠️ Δεν βρέθηκαν αυτόματα δεδομένα εξωτερικής θερμοκρασίας.\n"
                "Συμπλήρωσε t_min και t_max ή δώσε 24 ωριαίες τιμές και ξαναπάτα Predict.",
                pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None,
                f"API 422: {detail}",
                gr.update(visible=True, value=None), gr.update(visible=True, value=None),
            )
        return msg, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None, "❌ API error", gr.update(visible=False, value=None), gr.update(visible=False, value=None)
    except Exception as e:
        return f"Σφάλμα επικοινωνίας με API: {e}", pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None, "❌ API not reachable", gr.update(visible=False, value=None), gr.update(visible=False, value=None)


def _build_common_payload_for_multi(*args, **kwargs):
    return _prepare_payload(*args, **kwargs)


def do_compare(
    use_history: bool,
    model_choice: str,
    optimization: str,
    city: str,
    target_date: str,
    internal_temperature_24h_text: str,
    external_temperature_24h_text: str,
    internal_humidity_24h_text: str,
    external_humidity_24h_text: str,
    t_min: Optional[float],
    t_max: Optional[float],
    num_rooms: float,
    residents: float,
    num_adults: Optional[float],
    num_children: Optional[float],
    num_elderly: Optional[float],
    has_ac: float,
    has_fridge_freezer: float,
    has_dryer: float,
    has_washing_machine: float,
    has_dishwasher: float,
    has_microwave: float,
    has_electric_oven: float,
    has_electric_hob: float,
    solar_panels: float,
    building_type: str,
    build_era: str,
    income_band: str,
    heating_type: str,
    water_heater_type: str,
    homeowner_status: str,
    years_in_house: str,
    occupation: str,
    history_csv_file: Any,
    history_csv_path_text: str,
    use_default_history_store: bool,
    min_history_hours: int,
    history_consumption_text: str,
    apply_history_correction: bool,
    history_correction_days: int,
    history_correction_max_alpha: float,
    use_recommended_history_alpha: bool,
    recommended_history_correction_max_alpha: Optional[float],
    enable_behavior_adjustment: bool,
    high_consumption_hours_text: str,
    behavior_factor: Optional[float],
    use_custom_weights: bool,
    rf_weight_pct: Optional[float],
    xgb_weight_pct: Optional[float],
    lgbm_weight_pct: Optional[float],
    save_csv: bool,
):
    try:
        behavior_enabled, behavior_hours, behavior_factor_norm = _normalize_behavior_inputs(enable_behavior_adjustment, high_consumption_hours_text, behavior_factor)
        mode = MODE_WITH_HISTORY if use_history else MODE_NO_HISTORY
        compare_ids = _comparison_model_ids(mode)
        effective_alpha = _resolve_effective_history_alpha(use_history, use_recommended_history_alpha, recommended_history_correction_max_alpha, history_correction_max_alpha)

        payload, temps24_user, hum24_user = _prepare_payload(
            use_history, "auto — Auto default", optimization, city, target_date,
            internal_temperature_24h_text, external_temperature_24h_text,
            internal_humidity_24h_text, external_humidity_24h_text,
            t_min, t_max, num_rooms, residents, num_adults, num_children, num_elderly,
            has_ac, has_fridge_freezer, has_dryer, has_washing_machine, has_dishwasher,
            has_microwave, has_electric_oven, has_electric_hob, solar_panels,
            building_type, build_era, income_band, heating_type, water_heater_type,
            homeowner_status, years_in_house, occupation, history_csv_file, history_csv_path_text,
            use_default_history_store, min_history_hours, history_consumption_text,
            apply_history_correction, history_correction_days, effective_alpha,
        )

        results: Dict[str, Dict[str, Any]] = {}
        meteo_df_show = pd.DataFrame()
        primary_df_show = pd.DataFrame()

        for key, model_id in compare_ids.items():
            payload_mid = dict(payload)
            payload_mid["model_id"] = model_id
            out_mid = _post_predict(payload_mid)
            results[key] = out_mid

            if key == "lgbm":
                preds_primary = _predictions_to_df(out_mid)
                if behavior_enabled:
                    preds_primary = _apply_behavior_adjustment_single_df(preds_primary, behavior_hours, behavior_factor_norm)
                primary_df_show = preds_primary.copy()
                temps24 = out_mid.get("external_temperature_24h") or temps24_user
                hum24 = out_mid.get("external_humidity_24h") or hum24_user
                meteo_df_full = _meteo24_to_df(temps24, hum24)
                primary_df_show, meteo_df_show = _filter_today_for_display(primary_df_show, meteo_df_full.copy(), target_date)

        comparison_rows = []
        curve_df = None
        curve_df_for_save = None
        remaining_map: Dict[str, Optional[float]] = {}
        cutoff_map: Dict[str, Optional[str]] = {}

        for key, out_mid in results.items():
            preds_mid = _predictions_to_df(out_mid)
            if behavior_enabled:
                preds_mid = _apply_behavior_adjustment_single_df(preds_mid, behavior_hours, behavior_factor_norm)

            rem_mid, cutoff_mid = _compute_remaining_kwh_today(preds_mid[["timestamp", "pred_consumption_Wh"]], target_date)
            remaining_map[key] = rem_mid
            cutoff_map[key] = cutoff_mid
            total_kwh = float(preds_mid["pred_consumption_Wh"].sum() / 1000.0)

            comparison_rows.append({
                "model_key": key,
                "model_id": out_mid.get("model_id"),
                "model_name": out_mid.get("model_name"),
                "mode": out_mid.get("mode"),
                "optimization": out_mid.get("optimization"),
                "prediction_variant": out_mid.get("prediction_variant"),
                "total_kWh_day": total_kwh,
                "remaining_kWh_today": rem_mid,
                "history_correction_applied": out_mid.get("history_correction_applied"),
                "history_correction_reason": out_mid.get("history_correction_reason"),
                "history_csv_path": out_mid.get("history_csv_path"),
            })

            if not preds_mid.empty:
                temp_curve = preds_mid[["timestamp", "pred_consumption_Wh"]].copy()
                temp_curve.rename(columns={"pred_consumption_Wh": f"{key}_Wh"}, inplace=True)
                curve_df = temp_curve if curve_df is None else curve_df.merge(temp_curve, on="timestamp", how="outer")

        comparison_df = pd.DataFrame(comparison_rows)
        flags = next(iter(results.values())).get("derived_flags", {}) or {}
        if not comparison_df.empty:
            comparison_df = _add_flag_columns_to_df(comparison_df, flags)

        comparison_fig = None
        if curve_df is not None and not curve_df.empty:
            curve_df = curve_df.sort_values("timestamp").reset_index(drop=True)
            curve_df_for_save = curve_df.copy()
            curve_df = _filter_today_curve_only(curve_df, target_date)
            comparison_fig = _build_compare_chart(curve_df)

        behavior_text = _behavior_summary_suffix(behavior_enabled, behavior_hours, behavior_factor_norm)
        summary = _build_summary_compare(results, remaining_map, cutoff_map, behavior_text)

        status = "✅ Comparison done"
        if save_csv and curve_df_for_save is not None and not curve_df_for_save.empty:
            first = next(iter(results.values()))
            temps24_for_save = first.get("external_temperature_24h") or temps24_user
            hum24_for_save = first.get("external_humidity_24h") or hum24_user
            fp = _save_csv_compare(city, target_date, mode, optimization, curve_df_for_save, comparison_df, temps24_for_save, hum24_for_save, flags)
            status = f"✅ Saved compare CSV: {fp}"

        return summary, primary_df_show, meteo_df_show, comparison_df, comparison_fig, status, gr.update(visible=False, value=None), gr.update(visible=False, value=None)

    except ValueError as e:
        return str(e), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None, "⚠️ Fix inputs and try again.", gr.update(visible=False, value=None), gr.update(visible=False, value=None)
    except RuntimeError as e:
        msg = str(e)
        if msg.startswith("WEATHER_FALLBACK::"):
            detail = msg.split("WEATHER_FALLBACK::", 1)[1]
            return (
                "⚠️ Δεν βρέθηκαν αυτόματα δεδομένα εξωτερικής θερμοκρασίας.\nΣυμπλήρωσε t_min και t_max ή δώσε 24 ωριαίες τιμές και ξαναπάτα Compare.",
                pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None,
                f"API 422: {detail}",
                gr.update(visible=True, value=None), gr.update(visible=True, value=None),
            )
        return msg, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None, "❌ API error", gr.update(visible=False, value=None), gr.update(visible=False, value=None)
    except Exception as e:
        return f"Σφάλμα επικοινωνίας με API: {e}", pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None, "❌ API not reachable", gr.update(visible=False, value=None), gr.update(visible=False, value=None)


def do_combined(
    use_history: bool,
    model_choice: str,
    optimization: str,
    city: str,
    target_date: str,
    internal_temperature_24h_text: str,
    external_temperature_24h_text: str,
    internal_humidity_24h_text: str,
    external_humidity_24h_text: str,
    t_min: Optional[float],
    t_max: Optional[float],
    num_rooms: float,
    residents: float,
    num_adults: Optional[float],
    num_children: Optional[float],
    num_elderly: Optional[float],
    has_ac: float,
    has_fridge_freezer: float,
    has_dryer: float,
    has_washing_machine: float,
    has_dishwasher: float,
    has_microwave: float,
    has_electric_oven: float,
    has_electric_hob: float,
    solar_panels: float,
    building_type: str,
    build_era: str,
    income_band: str,
    heating_type: str,
    water_heater_type: str,
    homeowner_status: str,
    years_in_house: str,
    occupation: str,
    history_csv_file: Any,
    history_csv_path_text: str,
    use_default_history_store: bool,
    min_history_hours: int,
    history_consumption_text: str,
    apply_history_correction: bool,
    history_correction_days: int,
    history_correction_max_alpha: float,
    use_recommended_history_alpha: bool,
    recommended_history_correction_max_alpha: Optional[float],
    enable_behavior_adjustment: bool,
    high_consumption_hours_text: str,
    behavior_factor: Optional[float],
    use_custom_weights: bool,
    rf_weight_pct: Optional[float],
    xgb_weight_pct: Optional[float],
    lgbm_weight_pct: Optional[float],
    save_csv: bool,
):
    try:
        behavior_enabled, behavior_hours, behavior_factor_norm = _normalize_behavior_inputs(enable_behavior_adjustment, high_consumption_hours_text, behavior_factor)
        weights = _resolve_ensemble_weights(use_custom_weights, rf_weight_pct, xgb_weight_pct, lgbm_weight_pct)
        mode = MODE_WITH_HISTORY if use_history else MODE_NO_HISTORY
        compare_ids = _comparison_model_ids(mode)
        effective_alpha = _resolve_effective_history_alpha(use_history, use_recommended_history_alpha, recommended_history_correction_max_alpha, history_correction_max_alpha)

        payload, temps24_user, hum24_user = _prepare_payload(
            use_history, "auto — Auto default", optimization, city, target_date,
            internal_temperature_24h_text, external_temperature_24h_text,
            internal_humidity_24h_text, external_humidity_24h_text,
            t_min, t_max, num_rooms, residents, num_adults, num_children, num_elderly,
            has_ac, has_fridge_freezer, has_dryer, has_washing_machine, has_dishwasher,
            has_microwave, has_electric_oven, has_electric_hob, solar_panels,
            building_type, build_era, income_band, heating_type, water_heater_type,
            homeowner_status, years_in_house, occupation, history_csv_file, history_csv_path_text,
            use_default_history_store, min_history_hours, history_consumption_text,
            apply_history_correction, history_correction_days, effective_alpha,
        )

        results: Dict[str, Dict[str, Any]] = {}
        meteo_df_show = pd.DataFrame()
        curve_df = None
        temps24_for_save = None
        hum24_for_save = None

        for key, model_id in compare_ids.items():
            payload_mid = dict(payload)
            payload_mid["model_id"] = model_id
            out_mid = _post_predict(payload_mid)
            results[key] = out_mid
            preds_mid = _predictions_to_df(out_mid)

            if temps24_for_save is None:
                temps24_for_save = out_mid.get("external_temperature_24h") or temps24_user
                hum24_for_save = out_mid.get("external_humidity_24h") or hum24_user
                meteo_df_show = _meteo24_to_df(temps24_for_save, hum24_for_save)

            if not preds_mid.empty:
                temp_curve = preds_mid[["timestamp", "pred_consumption_Wh"]].copy()
                temp_curve.rename(columns={"pred_consumption_Wh": f"{key}_Wh"}, inplace=True)
                curve_df = temp_curve if curve_df is None else curve_df.merge(temp_curve, on="timestamp", how="outer")

        if curve_df is None or curve_df.empty:
            raise RuntimeError("No predictions available for combined forecast.")

        curve_df = curve_df.sort_values("timestamp").reset_index(drop=True)
        curve_df["ensemble_Wh"] = weights["rf"] * curve_df["rf_Wh"] + weights["xgb"] * curve_df["xgb_Wh"] + weights["lgbm"] * curve_df["lgbm_Wh"]

        if behavior_enabled:
            curve_df = _apply_behavior_adjustment_curve_df(curve_df, behavior_hours, behavior_factor_norm, ["rf_Wh", "xgb_Wh", "lgbm_Wh", "ensemble_Wh"])

        total_kwh_day = float(curve_df["ensemble_Wh"].sum() / 1000.0)
        display_curve = curve_df[["timestamp", "ensemble_Wh"]].copy().rename(columns={"ensemble_Wh": "pred_consumption_Wh"})
        remaining_kwh_today, cutoff_iso = _compute_remaining_kwh_today_for_col(display_curve, target_date, "pred_consumption_Wh")
        display_curve, meteo_df_show = _filter_today_for_display(display_curve, meteo_df_show, target_date)

        first = next(iter(results.values()))
        flags = first.get("derived_flags", {}) or {}
        behavior_text = _behavior_summary_suffix(behavior_enabled, behavior_hours, behavior_factor_norm)
        summary = _build_summary_combined(total_kwh_day, remaining_kwh_today, cutoff_iso, city, target_date, mode, optimization, weights, flags, behavior_text)

        comparison_df = pd.DataFrame([
            {"model_key": "rf", "weight": weights["rf"], "total_kWh_day": float(curve_df["rf_Wh"].sum() / 1000.0)},
            {"model_key": "xgb", "weight": weights["xgb"], "total_kWh_day": float(curve_df["xgb_Wh"].sum() / 1000.0)},
            {"model_key": "lgbm", "weight": weights["lgbm"], "total_kWh_day": float(curve_df["lgbm_Wh"].sum() / 1000.0)},
            {"model_key": "combined", "weight": 1.0, "total_kWh_day": total_kwh_day},
        ])
        comparison_df = _add_flag_columns_to_df(comparison_df, flags)

        fig_curve = _filter_today_curve_only(curve_df, target_date)
        fig = _build_compare_chart(fig_curve[["timestamp", "rf_Wh", "xgb_Wh", "lgbm_Wh", "ensemble_Wh"]])

        status = "✅ Combined prediction done"
        if save_csv:
            fp = _save_csv_combined(city, target_date, mode, optimization, curve_df, temps24_for_save, hum24_for_save, remaining_kwh_today, cutoff_iso, weights, flags)
            status = f"✅ Saved combined CSV: {fp}"

        return summary, display_curve, meteo_df_show, comparison_df, fig, status, gr.update(visible=False, value=None), gr.update(visible=False, value=None)

    except ValueError as e:
        return str(e), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None, "⚠️ Fix inputs and try again.", gr.update(visible=False, value=None), gr.update(visible=False, value=None)
    except RuntimeError as e:
        msg = str(e)
        if msg.startswith("WEATHER_FALLBACK::"):
            detail = msg.split("WEATHER_FALLBACK::", 1)[1]
            return (
                "⚠️ Δεν βρέθηκαν αυτόματα δεδομένα εξωτερικής θερμοκρασίας.\nΣυμπλήρωσε t_min και t_max ή δώσε 24 ωριαίες τιμές και ξαναπάτα Combined Prediction.",
                pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None,
                f"API 422: {detail}",
                gr.update(visible=True, value=None), gr.update(visible=True, value=None),
            )
        return msg, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None, "❌ API error", gr.update(visible=False, value=None), gr.update(visible=False, value=None)
    except Exception as e:
        return f"Σφάλμα επικοινωνίας με API: {e}", pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None, "❌ API not reachable", gr.update(visible=False, value=None), gr.update(visible=False, value=None)


# ============================================================
# UI EVENTS
# ============================================================

def _on_use_history_toggle(use_history: bool):
    mode = MODE_WITH_HISTORY if use_history else MODE_NO_HISTORY
    choices = _model_choices_for_mode(mode)
    value = choices[0] if choices else "auto — Auto default"
    return gr.update(visible=use_history), gr.update(choices=choices, value=value)


def _toggle_custom_weights(use_custom: bool):
    return gr.update(visible=use_custom)


def _reset_fallbacks_on_change(_):
    return gr.update(visible=False, value=None), gr.update(visible=False, value=None)


# ============================================================
# BUILD UI
# ============================================================

INITIAL_MODE = MODE_NO_HISTORY
MODEL_CHOICES = _model_choices_for_mode(INITIAL_MODE)
DEFAULT_MODEL = MODEL_CHOICES[0] if MODEL_CHOICES else "auto — Auto default"

with gr.Blocks(title="PLEGMA Forecasting UI") as demo:
    gr.Markdown("## PLEGMA Load Forecasting (Demo UI)")
    gr.Markdown(
        "Συμπλήρωσε τα πεδία και πάτα **Predict** για ένα μοντέλο, "
        "**Compare** για RF / XGB / LGBM ή **Combined Prediction** για weighted ensemble."
    )

    api_status = gr.Markdown(_api_health())

    with gr.Row():
        use_history = gr.Checkbox(
            value=False,
            label="Πρόβλεψη με χρήση ιστορικών δεδομένων",
            info="Αν το ενεργοποιήσεις, το API θα χρησιμοποιήσει mode='with_history'.",
        )

    with gr.Row():
        model_choice = gr.Dropdown(choices=MODEL_CHOICES, value=DEFAULT_MODEL, label="Μοντέλο")
        optimization = gr.Dropdown(choices=["balanced", "daily"], value="balanced", label="Optimization")

    with gr.Row():
        city = gr.Dropdown(choices=DEFAULT_CITIES, value="athens", label="City")
        target_date = gr.Textbox(value=_today_date().isoformat(), label="Target date (YYYY-MM-DD)")

    with gr.Accordion("Environmental inputs (24h)", open=True):
        internal_temperature_24h_text = gr.Textbox(
            value=_list24_to_text(_default_internal_temperature_24h()),
            label="internal_temperature_24h (default visible, editable)",
            lines=3,
        )
        external_temperature_24h_text = gr.Textbox(
            value="",
            label="external_temperature_24h (optional, 24 values)",
            placeholder="Άστο κενό για city-based lookup / t_min-t_max fallback",
            lines=3,
        )
        internal_humidity_24h_text = gr.Textbox(
            value=_list24_to_text(_default_internal_humidity_24h()),
            label="internal_humidity_24h (default visible, editable)",
            lines=3,
        )
        external_humidity_24h_text = gr.Textbox(
            value="",
            label="external_humidity_24h (optional, 24 values)",
            placeholder="Άστο κενό για Open-Meteo / default humidity profile από API",
            lines=3,
        )
        gr.Markdown("Αν δεν βρεθεί εξωτερική θερμοκρασία από Open-Meteo, συμπλήρωσε t_min και t_max.")
        with gr.Row():
            t_min = gr.Number(value=None, label="t_min (°C)", visible=False)
            t_max = gr.Number(value=None, label="t_max (°C)", visible=False)

    with gr.Accordion("Household inputs", open=True):
      
        # --------------------------------------------------------
        # Visible household inputs
        # --------------------------------------------------------

        with gr.Row():
            building_type = gr.Dropdown(
                choices=DEFAULT_BUILDING_TYPES,
                value="apartment",
                label="building_type",
            )
            build_era = gr.Dropdown(
                choices=DEFAULT_BUILD_ERAS,
                value="1970_1990",
                label="build_era",
            )
            num_rooms = gr.Number(
                value=3,
                label="num_rooms",
                minimum=0,
            )
            residents = gr.Number(
                value=2,
                label="residents",
                minimum=0,
            )

        with gr.Row():
            has_dryer = gr.Dropdown(
                choices=[0, 1],
                value=0,
                label="has_dryer",
            )
            has_ac = gr.Dropdown(
                choices=[0, 1],
                value=1,
                label="has_ac",
            )
            has_washing_machine = gr.Dropdown(
                choices=[0, 1],
                value=1,
                label="has_washing_machine",
            )
            has_dishwasher = gr.Dropdown(
                choices=[0, 1],
                value=1,
                label="has_dishwasher",
            )
            has_microwave = gr.Dropdown(
                choices=[0, 1],
                value=1,
                label="has_microwave",
            )

        with gr.Row():
            heating_type = gr.Dropdown(
                choices=DEFAULT_HEATING_TYPES,
                value="air_conditioner",
                label="heating_type",
            )
            water_heater_type = gr.Dropdown(
                choices=DEFAULT_WATER_HEATER_TYPES,
                value="electric_boiler",
                label="water_heater_type",
            )
            years_in_house = gr.Dropdown(
                choices=DEFAULT_YEARS_IN_HOUSE,
                value="3_to_4_years",
                label="years_in_house",
            )

        # --------------------------------------------------------
        # Hidden fixed / non-essential inputs
        # --------------------------------------------------------
        # These components must still exist because _prepare_payload,
        # do_predict, do_compare, do_combined and common_inputs expect them.
        # They are hidden from the user but their default values are sent to the API.

        num_adults = gr.Number(
            value=None,
            label="num_adults",
            visible=False,
        )
        num_children = gr.Number(
            value=0,
            label="num_children",
            visible=False,
        )
        num_elderly = gr.Number(
            value=0,
            label="num_elderly",
            visible=False,
        )

        # Fixed appliance assumptions for the selected PLEGMA case-study homes.
        has_fridge_freezer = gr.Dropdown(
            choices=[0, 1],
            value=1,
            label="has_fridge_freezer",
            visible=False,
        )
        has_electric_oven = gr.Dropdown(
            choices=[0, 1],
            value=1,
            label="has_electric_oven",
            visible=False,
        )
        has_electric_hob = gr.Dropdown(
            choices=[0, 1],
            value=1,
            label="has_electric_hob",
            visible=False,
        )
        solar_panels = gr.Dropdown(
            choices=[0, 1],
            value=0,
            label="solar_panels",
            visible=False,
        )

        # Non-essential metadata kept hidden to preserve API compatibility.
        income_band = gr.Dropdown(
            choices=DEFAULT_INCOME_BANDS,
            value="unknown",
            label="income_band",
            visible=False,
        )
        homeowner_status = gr.Dropdown(
            choices=DEFAULT_HOMEOWNER_STATUS,
            value="unknown",
            label="homeowner_status",
            visible=False,
        )
        occupation = gr.Dropdown(
            choices=DEFAULT_OCCUPATION,
            value="unknown",
            label="occupation",
            visible=False,
        )

    with gr.Group(visible=False) as history_group:
        with gr.Accordion("With-history inputs", open=True):
            gr.Markdown(
                "Το history CSV πρέπει να είναι single-home αρχείο με στήλες `timestamp` και `consumption_Wh`. "
                "Δεν χρειάζεται και δεν ζητείται `home_id`."
            )
            with gr.Row():
                history_csv_file = gr.UploadButton(
                    "Select history CSV",
                    file_types=[".csv"],
                    file_count="single",
                )
                history_csv_path_text = gr.Textbox(
                    value="",
                    label="Selected history CSV path",
                    placeholder="Θα συμπληρωθεί αυτόματα μετά την επιλογή αρχείου ή γράψε path χειροκίνητα",
                    scale=4,
                )
            history_upload_status = gr.Markdown("No history CSV selected.")

            use_default_history_store = gr.Checkbox(
                value=True,
                label="Use default history_store.csv if no CSV/manual history is provided",
            )
            min_history_hours = gr.Number(value=168, label="min_history_hours", precision=0, minimum=1)
            history_consumption_text = gr.Textbox(
                value="",
                label="history_consumption_Wh manual vector (optional, comma-separated)",
                placeholder="π.χ. 100, 120, 95, ...",
                lines=4,
            )

            with gr.Row():
                apply_history_correction = gr.Checkbox(value=True, label="Apply adaptive history correction")
                history_correction_days = gr.Number(value=7, label="history_correction_days", precision=0, minimum=1, maximum=30)
                history_correction_max_alpha = gr.Number(value=0.20, label="manual max_alpha", minimum=0.0, maximum=1.0)

            with gr.Accordion("History stability recommendation", open=False):
                analyze_history_btn = gr.Button("Analyze history stability")
                with gr.Row():
                    history_stability_score = gr.Textbox(value="", label="Stability score", interactive=False)
                    history_stability_category = gr.Textbox(value="Outdated / not analyzed", label="Category", interactive=False)
                    recommended_history_correction_max_alpha = gr.Number(value=None, label="Recommended max_alpha", interactive=False)
                history_stability_explanation = gr.Textbox(value="", label="Explanation", lines=10, interactive=False)
                history_stability_status = gr.Markdown("History stability recommendation not analyzed yet.")
                use_recommended_history_alpha = gr.Checkbox(value=False, label="Use recommended max_alpha in Predict/Compare/Combined")

    with gr.Accordion("Behavioral adjustment (optional)", open=False):
        enable_behavior_adjustment = gr.Checkbox(value=False, label="Enable behavioral adjustment")
        high_consumption_hours_text = gr.Textbox(
            value="",
            label="Hours to adjust",
            placeholder="π.χ. 18-23 ή 7,8,19,20",
        )
        behavior_factor = gr.Number(value=1.10, label="Behavior factor", minimum=0.01)

    with gr.Accordion("Weighted ensemble settings", open=False):
        use_custom_weights = gr.Checkbox(value=False, label="Use custom ensemble weights")
        with gr.Group(visible=False) as custom_weights_group:
            with gr.Row():
                rf_weight_pct = gr.Number(value=25, label="RF weight (%)", minimum=0, maximum=100)
                xgb_weight_pct = gr.Number(value=35, label="XGB weight (%)", minimum=0, maximum=100)
                lgbm_weight_pct = gr.Number(value=40, label="LGBM weight (%)", minimum=0, maximum=100)

    save_csv = gr.Checkbox(value=False, label="Save prediction CSV")

    with gr.Row():
        predict_btn = gr.Button("Predict", variant="primary")
        compare_btn = gr.Button("Compare RF / XGB / LGBM")
        combined_btn = gr.Button("Combined Prediction")

    summary_box = gr.Textbox(label="Summary", lines=18)
    status_box = gr.Textbox(label="Status", lines=2)
    plot_out = gr.Plot(label="Prediction chart")

    with gr.Tabs():
        with gr.Tab("Prediction / Display table"):
            pred_table = gr.Dataframe(label="Predictions")
        with gr.Tab("Weather table"):
            weather_table = gr.Dataframe(label="Weather / external environmental profile")
        with gr.Tab("Comparison table"):
            comparison_table = gr.Dataframe(label="Comparison / ensemble totals")

    common_inputs = [
        use_history,
        model_choice,
        optimization,
        city,
        target_date,
        internal_temperature_24h_text,
        external_temperature_24h_text,
        internal_humidity_24h_text,
        external_humidity_24h_text,
        t_min,
        t_max,
        num_rooms,
        residents,
        num_adults,
        num_children,
        num_elderly,
        has_ac,
        has_fridge_freezer,
        has_dryer,
        has_washing_machine,
        has_dishwasher,
        has_microwave,
        has_electric_oven,
        has_electric_hob,
        solar_panels,
        building_type,
        build_era,
        income_band,
        heating_type,
        water_heater_type,
        homeowner_status,
        years_in_house,
        occupation,
        history_csv_file,
        history_csv_path_text,
        use_default_history_store,
        min_history_hours,
        history_consumption_text,
        apply_history_correction,
        history_correction_days,
        history_correction_max_alpha,
        use_recommended_history_alpha,
        recommended_history_correction_max_alpha,
        enable_behavior_adjustment,
        high_consumption_hours_text,
        behavior_factor,
        use_custom_weights,
        rf_weight_pct,
        xgb_weight_pct,
        lgbm_weight_pct,
        save_csv,
    ]

    common_outputs = [
        summary_box,
        pred_table,
        weather_table,
        comparison_table,
        plot_out,
        status_box,
        t_min,
        t_max,
    ]

    predict_btn.click(fn=do_predict, inputs=common_inputs, outputs=common_outputs)
    compare_btn.click(fn=do_compare, inputs=common_inputs, outputs=common_outputs)
    combined_btn.click(fn=do_combined, inputs=common_inputs, outputs=common_outputs)

    use_history.change(fn=_on_use_history_toggle, inputs=[use_history], outputs=[history_group, model_choice])
    use_custom_weights.change(fn=_toggle_custom_weights, inputs=[use_custom_weights], outputs=[custom_weights_group])

    history_csv_file.upload(
        fn=register_history_csv_upload,
        inputs=[history_csv_file],
        outputs=[history_csv_path_text, history_upload_status],
    )

    analyze_history_btn.click(
        fn=analyze_history_stability_recommendation,
        inputs=[history_csv_file, history_csv_path_text, target_date, history_correction_days, min_history_hours, history_consumption_text],
        outputs=[
            history_stability_score,
            history_stability_category,
            recommended_history_correction_max_alpha,
            history_stability_explanation,
            history_stability_status,
        ],
    )

    for comp in [history_csv_path_text, history_consumption_text, target_date, history_correction_days, min_history_hours]:
        comp.change(
            fn=_reset_history_stability_recommendation,
            inputs=[],
            outputs=[
                history_stability_score,
                history_stability_category,
                recommended_history_correction_max_alpha,
                history_stability_explanation,
                history_stability_status,
                use_recommended_history_alpha,
            ],
        )

    for comp in [city, target_date, external_temperature_24h_text]:
        comp.change(fn=_reset_fallbacks_on_change, inputs=[comp], outputs=[t_min, t_max])


if __name__ == "__main__":
    demo.launch(server_name=UI_HOST, server_port=UI_PORT, show_error=True)
