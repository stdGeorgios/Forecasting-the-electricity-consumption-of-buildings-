# C:/IDEAL_Programming/src/ui_app.py
# ---------------------------------------------------------
# Gradio UI for IDEAL Forecasting API - Generic UI version
# ---------------------------------------------------------
#
# Final API/UI assumptions:
#   - The user does NOT provide home_id.
#   - model_id="auto" selects the correct default by scenario:
#       no_history   -> rf_coldstart_default (legacy artifact id)
#       with_history -> lgbm_withhistory_default
#   - Optional comparison models are selected through the new API registry.
#   - With-history requires recent hourly consumption history, default 168 hours.
#   - History can come from a user-selected single-home CSV path/upload,
#     manual comma-separated values, or the legacy history_store.csv fallback.
#   - Weather can come from Open-Meteo, manual t_min/t_max, or manual 24h vector.
#
# Main actions:
#   Predict              -> one selected model or auto default
#   Compare              -> RF / XGB / LGBM comparison for selected scenario
#   Combined Prediction  -> weighted ensemble of RF / XGB / LGBM predictions
#
# This UI is compatible with:
#   api_app.py
#   model_registry.py
#   inference.py
#   history_store.py
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

API_BASE = "http://127.0.0.1:8000"
UI_HOST = "127.0.0.1"
UI_PORT = 7860

BASE_DIR = Path("C:/IDEAL_Programming")
UI_EXPORT_DIR = BASE_DIR / "processed" / "predictions" / "ui_exports"
UI_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_UPLOAD_DIR = BASE_DIR / "processed" / "stores" / "selected_history_files"
HISTORY_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

TIMEOUT = 180

DEFAULT_CITIES = ["Edinburgh", "Glasgow", "Dundee", "Aberdeen"]

DEFAULT_HOMETYPES = [
    ("flat", "flat"),
    ("house_or_bungalow", "house_or_bungalow"),
]

URBAN_RURAL_OPTIONS = [
    ("1 — Large urban area", "1"),
    ("2 — Other urban area", "2"),
    ("3 — Accessible small town / rural", "3"),
    ("3+ — Remote small town / rural", "3+"),
]

DEFAULT_ENSEMBLE_WEIGHTS = {
    "rf": 0.40,
    "xgb": 0.30,
    "lgbm": 0.30,
}

MODE_NO_HISTORY = "no_history"
MODE_WITH_HISTORY = "with_history"

# Legacy names kept only for internal compatibility with old model IDs / wording.
MODE_COLDSTART = MODE_NO_HISTORY
MODE_WITHHISTORY = MODE_WITH_HISTORY


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
        return "❌ API: not reachable. Start FastAPI first."


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
            {"label": "LGBM with-history default", "value": "lgbm_withhistory_default"},
            {"label": "XGB with-history optional", "value": "xgb_withhistory_optional"},
            {"label": "RF with-history optional", "value": "rf_withhistory_optional"},
        ]

    return [
        {"label": "Auto default", "value": "auto", "is_default": True},
        {"label": "RF no-history default", "value": "rf_coldstart_default"},
        {"label": "LGBM no-history optional", "value": "lgbm_coldstart_optional"},
        {"label": "XGB no-history optional", "value": "xgb_coldstart_optional"},
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
        or "t_min" in t
        or "t_max" in t
        or "external_temperature_24h" in t
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
    total_floor_area_m2: float,
    residents: float,
    min_history_hours: int,
    t_min: Optional[float],
    t_max: Optional[float],
) -> Optional[str]:
    try:
        _ = _parse_date(target_date)
    except Exception:
        return "Σφάλμα: target_date πρέπει να είναι σε μορφή YYYY-MM-DD."

    if total_floor_area_m2 is None or float(total_floor_area_m2) <= 0:
        return "Σφάλμα: total_floor_area_m2 πρέπει να είναι > 0."

    if residents is None or float(residents) <= 0:
        return "Σφάλμα: residents πρέπει να είναι > 0."

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
    """Return a local path from a Gradio uploaded/selected file object."""
    if file_obj is None:
        return None

    if isinstance(file_obj, str):
        return file_obj

    if isinstance(file_obj, dict):
        return (
            file_obj.get("name")
            or file_obj.get("path")
            or file_obj.get("orig_name")
        )

    return (
        getattr(file_obj, "name", None)
        or getattr(file_obj, "path", None)
    )


def register_history_csv_upload(file_obj: Any) -> Tuple[str, str]:
    """Handle the UI history CSV selection button.

    The browser opens the normal Windows file picker. Gradio receives the
    selected CSV and stores it in a temporary location. We copy it to a stable
    project folder and write that path into the History CSV path textbox, so
    the FastAPI process can read the same file reliably.
    """
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
    """Resolve a single-home history CSV path.

    Priority:
      1. Textbox path populated by the Select history CSV button or typed manually.
      2. Direct Gradio file object fallback.

    The CSV is assumed to contain one home only and must include:
      timestamp, consumption_Wh
    """
    if history_csv_path_text and str(history_csv_path_text).strip():
        return str(history_csv_path_text).strip().strip('\"').strip("'")

    file_path = _extract_uploaded_file_path(history_csv_file)
    if file_path and str(file_path).strip():
        return str(file_path).strip().strip('\"').strip("'")

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
    """Convert coefficient of variation to a 0-1 stability score."""
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
    """Choose the alpha sent to the API.

    If the user checked 'Use recommended max alpha', the UI uses the latest
    recommendation from the stability check. Otherwise it uses the manual field.
    """
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
        rec_alpha = float(recommended_history_correction_max_alpha)
    except Exception:
        raise ValueError(
            "Η προτεινόμενη τιμή max_alpha δεν είναι έγκυρη. "
            "Πάτησε ξανά Analyze history stability."
        )

    return rec_alpha


def _history_dataframe_from_inputs(
    history_csv_file: Any,
    history_csv_path_text: str,
    history_consumption_text: str,
    target_date: str,
    history_days: int,
) -> Tuple[pd.DataFrame, str]:
    """Load history either from CSV/path or manual comma-separated values.

    Returns columns:
      timestamp, consumption_Wh

    The returned dataframe is not modified on disk.
    """
    target_start = pd.Timestamp(_parse_date(target_date))
    expected_hours = int(history_days) * 24

    manual_values = _parse_csv_float_list(
        history_consumption_text,
        "history_consumption_Wh",
        expected_len=None,
    )

    if manual_values is not None:
        if len(manual_values) < 1:
            raise ValueError("Το manual history_consumption_Wh είναι κενό.")

        # Create synthetic hourly timestamps ending exactly before target_start.
        start = target_start - pd.Timedelta(hours=len(manual_values))
        timestamps = pd.date_range(start, periods=len(manual_values), freq="h")
        hist = pd.DataFrame({
            "timestamp": timestamps,
            "consumption_Wh": [float(x) for x in manual_values],
        })
        return hist, "manual history_consumption_Wh"

    history_csv_path = _resolve_history_csv_path(history_csv_file, history_csv_path_text)

    if not history_csv_path:
        raise ValueError(
            "Δεν έχει επιλεγεί history CSV και δεν δόθηκε manual history_consumption_Wh."
        )

    path = Path(str(history_csv_path).strip().strip('"').strip("'"))

    if not path.exists():
        raise ValueError(f"Δεν βρέθηκε το history CSV: {path}")

    hist = pd.read_csv(path, low_memory=False)

    col_map = {str(c).lower(): c for c in hist.columns}
    ts_col = col_map.get("timestamp") or col_map.get("datetime") or col_map.get("date_time")
    val_col = (
        col_map.get("consumption_wh")
        or col_map.get("actual_consumption_wh")
        or col_map.get("pred_consumption_wh")
        or col_map.get("value")
    )

    if ts_col is None:
        raise ValueError(
            f"Το history CSV πρέπει να έχει στήλη timestamp. Columns: {list(hist.columns)}"
        )

    if val_col is None:
        raise ValueError(
            f"Το history CSV πρέπει να έχει στήλη consumption_Wh. Columns: {list(hist.columns)}"
        )

    hist = hist[[ts_col, val_col]].copy()
    hist = hist.rename(columns={ts_col: "timestamp", val_col: "consumption_Wh"})
    hist["timestamp"] = pd.to_datetime(hist["timestamp"], errors="coerce")
    hist["consumption_Wh"] = pd.to_numeric(hist["consumption_Wh"], errors="coerce")
    hist = hist.dropna(subset=["timestamp", "consumption_Wh"])
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
    """Analyze recent history stability and recommend a conservative max_alpha.

    The recommendation is based only on information before the target day.
    """
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

        # Strictly use history before the target day.
        hist = hist[hist["timestamp"] < target_start].copy()

        # Prefer the exact recent window if timestamps are available.
        window_hist = hist[
            (hist["timestamp"] >= window_start)
            & (hist["timestamp"] < target_start)
        ].copy()

        if window_hist.empty:
            # Fallback: last expected_hours before target_start.
            window_hist = hist.tail(expected_hours).copy()

        if window_hist.empty:
            raise ValueError("Δεν υπάρχουν διαθέσιμες ιστορικές τιμές πριν από την target date.")

        # One value per timestamp.
        window_hist = (
            window_hist
            .groupby("timestamp", as_index=False)
            .agg({"consumption_Wh": "mean"})
            .sort_values("timestamp")
        )

        expected_index = pd.date_range(
            window_start,
            target_start - pd.Timedelta(hours=1),
            freq="h",
        )

        aligned = (
            window_hist
            .set_index("timestamp")
            .reindex(expected_index)
        )
        aligned.index.name = "timestamp"
        aligned = aligned.reset_index()

        observed_hours = int(aligned["consumption_Wh"].notna().sum())
        missing_hours = int(aligned["consumption_Wh"].isna().sum())
        completeness_score = observed_hours / expected_hours if expected_hours > 0 else 0.0

        # Fill only for stability diagnostics, not for prediction input.
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

        daily = (
            aligned
            .groupby("date", as_index=False)
            .agg(
                daily_kWh=("consumption_filled_Wh", lambda s: float(pd.to_numeric(s, errors="coerce").sum() / 1000.0)),
                daily_mean_Wh=("consumption_filled_Wh", "mean"),
                daily_peak_Wh=("consumption_filled_Wh", "max"),
                observed_hours=("consumption_Wh", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
                is_weekend=("is_weekend", "max"),
            )
        )

        daily_kwh_values = pd.to_numeric(daily["daily_kWh"], errors="coerce").dropna()
        if daily_kwh_values.empty:
            mean_daily_kWh = float("nan")
            median_daily_kWh = float("nan")
            min_daily_kWh = float("nan")
            max_daily_kWh = float("nan")
        else:
            mean_daily_kWh = float(daily_kwh_values.mean())
            median_daily_kWh = float(daily_kwh_values.median())
            min_daily_kWh = float(daily_kwh_values.min())
            max_daily_kWh = float(daily_kwh_values.max())

        daily_cv = _safe_cv(daily["daily_kWh"])
        daily_score = _score_from_cv(daily_cv, cv_bad=0.35)

        pivot = aligned.pivot_table(
            index="date",
            columns="hour",
            values="consumption_filled_Wh",
            aggfunc="mean",
        )

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

        base_score = (
            0.30 * daily_score
            + 0.35 * hourly_score
            + 0.20 * peak_score
            + 0.15 * completeness_score
        )

        stability_score = _clamp01(base_score * (0.85 + 0.15 * day_type_score))

        # Data-driven recommendation for the maximum allowed adaptive alpha.
        # The alpha is still adaptive per hour inside the API; this value is only the safety cap.
        # Categories are intentionally conservative for low/medium stability and more permissive
        # only when the recent 7-day profile is highly repeatable.
        if completeness_score < 0.85 or observed_hours < min(int(min_history_hours or expected_hours), expected_hours) * 0.80:
            recommended_alpha = 0.05
            category = "Low"
            alpha_range = "0.00–0.10"
            reason_alpha = (
                "Το ιστορικό δεν είναι αρκετά πλήρες. Προτείνεται να μη χρησιμοποιηθεί "
                "adaptive correction ή να χρησιμοποιηθεί πολύ χαμηλό max_alpha."
            )
        elif stability_score >= 0.93:
            recommended_alpha = 0.60
            category = "Excellent"
            alpha_range = "0.55–0.65"
            reason_alpha = (
                "Το ιστορικό είναι εξαιρετικά σταθερό. Επιτρέπεται ισχυρότερη "
                "history-aware correction, αλλά τιμές πάνω από 0.65 θεωρούνται πειραματικές."
            )
        elif stability_score >= 0.80:
            recommended_alpha = 0.40
            category = "Very High"
            alpha_range = "0.35–0.50"
            reason_alpha = (
                "Το ιστορικό είναι πολύ σταθερό και επαναλαμβανόμενο. "
                "Μπορεί να χρησιμοποιηθεί πιο ισχυρό max_alpha με ελεγχόμενο ρίσκο."
            )
        elif stability_score >= 0.60:
            recommended_alpha = 0.25
            category = "High"
            alpha_range = "0.20–0.30"
            reason_alpha = "Το ιστορικό είναι σταθερό. Προτείνεται μέτρια history-aware correction."
        elif stability_score >= 0.40:
            recommended_alpha = 0.10
            category = "Medium"
            alpha_range = "0.10–0.15"
            reason_alpha = (
                "Το ιστορικό έχει μέτρια σταθερότητα. Προτείνεται χαμηλό max_alpha "
                "ώστε να περιοριστεί ο κίνδυνος υπερπροσαρμογής στο πρόσφατο history shape."
            )
        else:
            recommended_alpha = 0.05
            category = "Low"
            alpha_range = "0.00–0.10"
            reason_alpha = (
                "Το ιστορικό είναι ασταθές ή έχει μη επαναλαμβανόμενα peaks. "
                "Προτείνεται να μη χρησιμοποιηθεί adaptive correction ή να χρησιμοποιηθεί ελάχιστο max_alpha."
            )

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
            f"Recommended max_alpha={recommended_alpha:.2f} (range {alpha_range}). "
            "Tick 'Use recommended max alpha' to use it in Predict/Compare/Combined."
        )

        return (
            f"{stability_score:.3f}",
            category,
            float(recommended_alpha),
            explanation,
            status,
        )

    except Exception as exc:
        return (
            "",
            "Not available",
            None,
            f"Could not analyze history stability: {exc}",
            "⚠️ History stability recommendation not available.",
        )


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
            "rf": "rf_withhistory_optional",
            "xgb": "xgb_withhistory_optional",
            "lgbm": "lgbm_withhistory_default",
        }

    return {
        "rf": "rf_coldstart_default",
        "xgb": "xgb_coldstart_optional",
        "lgbm": "lgbm_coldstart_optional",
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
        return "Weekend and is_holiday"
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


def _build_summary_single(
    out: Dict[str, Any],
    remaining_kwh_today: Optional[float] = None,
    cutoff_iso: Optional[str] = None,
    behavior_text: Optional[str] = None,
) -> str:
    flags = out.get("derived_flags", {}) or {}
    model_name = out.get("model_name") or out.get("label") or out.get("model_id")

    lines = [
        f"Requested model: {out.get('requested_model_id')}",
        f"Resolved model: {out.get('model_id')} — {model_name}",
        f"Mode: {out.get('mode')} | Optimization: {out.get('optimization')} | Variant: {out.get('prediction_variant')}",
        f"City: {out.get('city')} | Date: {out.get('target_date')}",
        f"Total kWh/day (24h): {float(out.get('total_kWh_day', 0.0)):.3f}",
    ]

    if remaining_kwh_today is not None and cutoff_iso is not None:
        lines.append(f"Remaining kWh (from {cutoff_iso}): {float(remaining_kwh_today):.3f}")

    lines += [
        f"Weather source: {out.get('weather_source')}",
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


def _build_summary_compare(
    results: Dict[str, Dict[str, Any]],
    remaining_map: Dict[str, Optional[float]],
    cutoff_map: Dict[str, Optional[str]],
    behavior_text: Optional[str] = None,
) -> str:
    first = next(iter(results.values()))
    flags = first.get("derived_flags", {}) or {}
    lines = [
        f"City: {first.get('city')} | Date: {first.get('target_date')}",
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


def _build_summary_combined(
    total_kwh_day: float,
    remaining_kwh_today: Optional[float],
    cutoff_iso: Optional[str],
    city: str,
    target_date: str,
    mode: str,
    optimization: str,
    weights: Dict[str, float],
    flags: Optional[Dict[str, Any]] = None,
    behavior_text: Optional[str] = None,
) -> str:
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


def _temps24_to_df(temps24: Optional[List[float]]) -> pd.DataFrame:
    if temps24 is None or len(temps24) != 24:
        return pd.DataFrame()
    return pd.DataFrame({"hour": list(range(24)), "external_temp_C": [float(x) for x in temps24]})


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

    remaining_kwh = float(df_tmp["pred_consumption_Wh"].sum() / 1000.0)
    return remaining_kwh, cutoff_iso


def _compute_remaining_kwh_today_for_col(df_full: pd.DataFrame, target_date: str, value_col: str) -> Tuple[Optional[float], Optional[str]]:
    if df_full is None or df_full.empty or value_col not in df_full.columns:
        return None, None

    tmp = df_full.rename(columns={value_col: "pred_consumption_Wh"})[["timestamp", "pred_consumption_Wh"]].copy()
    return _compute_remaining_kwh_today(tmp, target_date)


def _filter_today_for_display(df_show: pd.DataFrame, temps_df_show: pd.DataFrame, target_date: str):
    try:
        d = _parse_date(target_date)
        if d == _today_date() and not df_show.empty:
            cutoff = _next_full_hour(_now_local())
            df_show = df_show[df_show["timestamp"] >= cutoff].copy()
            if not temps_df_show.empty:
                temps_df_show = temps_df_show[temps_df_show["hour"] >= cutoff.hour].copy()
    except Exception:
        pass

    return df_show, temps_df_show


def _filter_today_curve_only(df_show: pd.DataFrame, target_date: str):
    try:
        d = _parse_date(target_date)
        if d == _today_date() and not df_show.empty:
            cutoff = _next_full_hour(_now_local())
            df_show = df_show[df_show["timestamp"] >= cutoff].copy()
    except Exception:
        pass

    return df_show


def _save_csv_full_day(
    out: Dict[str, Any],
    preds_full: pd.DataFrame,
    temps24: Optional[List[float]],
    remaining_kwh_today: Optional[float],
    cutoff_iso: Optional[str],
) -> Path:
    df = preds_full.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if temps24 is not None and len(temps24) == 24 and len(df) == 24:
        df["external_temp_C"] = [float(x) for x in temps24]
    else:
        df["external_temp_C"] = pd.NA

    df["requested_model_id"] = out.get("requested_model_id")
    df["model_id"] = out.get("model_id")
    df["model_name"] = out.get("model_name") or out.get("label")
    df["model_type"] = out.get("model_type")
    df["mode"] = out.get("mode")
    df["optimization"] = out.get("optimization")
    df["prediction_variant"] = out.get("prediction_variant")
    df["city"] = out.get("city")
    df["target_date"] = out.get("target_date")
    df["weather_source"] = out.get("weather_source")
    df["history_source"] = out.get("history_source")
    df["history_csv_path"] = out.get("history_csv_path")
    df["history_store_path"] = out.get("history_store_path")
    df["history_window_start"] = out.get("history_window_start")
    df["history_window_end"] = out.get("history_window_end")
    df["history_hours_used"] = out.get("history_hours_used")
    df["history_correction_applied"] = out.get("history_correction_applied")
    df["history_correction_reason"] = out.get("history_correction_reason")

    flags = out.get("derived_flags", {}) or {}
    df = _add_flag_columns_to_df(df, flags)

    df["total_kWh_day"] = out.get("total_kWh_day")
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


def _save_csv_combined(
    city: str,
    target_date: str,
    mode: str,
    optimization: str,
    curve_df: pd.DataFrame,
    temps24: Optional[List[float]],
    remaining_kwh_today: Optional[float],
    cutoff_iso: Optional[str],
    weights: Dict[str, float],
    flags: Optional[Dict[str, Any]] = None,
) -> Path:
    df = curve_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if temps24 is not None and len(temps24) == 24 and len(df) == 24:
        df["external_temp_C"] = [float(x) for x in temps24]
    else:
        df["external_temp_C"] = pd.NA

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


def _save_csv_compare(
    city: str,
    target_date: str,
    mode: str,
    optimization: str,
    curve_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    temps24: Optional[List[float]],
    flags: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save full 24h comparison curves plus model-level totals.

    This is used by the Compare button when Save prediction CSV is checked.
    """
    flags = flags or {}

    curve = curve_df.copy()
    curve["timestamp"] = pd.to_datetime(curve["timestamp"])

    if temps24 is not None and len(temps24) == 24 and len(curve) == 24:
        curve["external_temp_C"] = [float(x) for x in temps24]
    else:
        curve["external_temp_C"] = pd.NA

    curve = _add_flag_columns_to_df(curve, flags)
    curve["mode"] = mode
    curve["optimization"] = optimization
    curve["city"] = city
    curve["target_date"] = target_date

    # Add total daily energy per model as repeated metadata columns for easier reading.
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
    total_floor_area_m2: float,
    residents: float,
    hometype: str,
    urban_rural_class: str,
    num_electric_appliances: Optional[float],
    t_min: Optional[float],
    t_max: Optional[float],
    external_temperature_24h_text: str,
    internal_temperature_24h_text: str,
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
        total_floor_area_m2=total_floor_area_m2,
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

    payload: Dict[str, Any] = {
        "model_id": model_id,
        "mode": mode,
        "optimization": optimization,
        "city": city,
        "target_date": target_date,
        "total_floor_area_m2": float(total_floor_area_m2),
        "residents": float(residents),
        "hometype": hometype,
        "urban_rural_class": str(urban_rural_class),
        "save_csv": False,
        "min_history_hours": int(min_history_hours),
    }

    if num_electric_appliances is not None:
        payload["num_electric_appliances"] = float(num_electric_appliances)

    temps24_user = _parse_csv_float_list(external_temperature_24h_text, "external_temperature_24h", expected_len=24)
    internal24_user = _parse_csv_float_list(internal_temperature_24h_text, "internal_temperature_24h", expected_len=24)

    if temps24_user is not None:
        payload["external_temperature_24h"] = temps24_user
    else:
        if t_min is not None and t_max is not None:
            payload["t_min"] = float(t_min)
            payload["t_max"] = float(t_max)

    if internal24_user is not None:
        payload["internal_temperature_24h"] = internal24_user

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

    return payload, temps24_user


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
    total_floor_area_m2: float,
    residents: float,
    hometype: str,
    urban_rural_class: str,
    num_electric_appliances: Optional[float],
    t_min: Optional[float],
    t_max: Optional[float],
    external_temperature_24h_text: str,
    internal_temperature_24h_text: str,
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
        behavior_enabled, behavior_hours, behavior_factor_norm = _normalize_behavior_inputs(
            enable_behavior_adjustment,
            high_consumption_hours_text,
            behavior_factor,
        )

        effective_history_correction_max_alpha = _resolve_effective_history_alpha(
            use_history=use_history,
            use_recommended_history_alpha=use_recommended_history_alpha,
            recommended_history_correction_max_alpha=recommended_history_correction_max_alpha,
            manual_history_correction_max_alpha=history_correction_max_alpha,
        )

        payload, temps24_user = _prepare_payload(
            use_history,
            model_choice,
            optimization,
            city,
            target_date,
            total_floor_area_m2,
            residents,
            hometype,
            urban_rural_class,
            num_electric_appliances,
            t_min,
            t_max,
            external_temperature_24h_text,
            internal_temperature_24h_text,
            history_csv_file,
            history_csv_path_text,
            use_default_history_store,
            min_history_hours,
            history_consumption_text,
            apply_history_correction,
            history_correction_days,
            effective_history_correction_max_alpha,
        )

        out = _post_predict(payload)
        preds_full = _predictions_to_df(out)

        temps24 = out.get("external_temperature_24h") or temps24_user
        temps_df_full = _temps24_to_df(temps24)

        if behavior_enabled:
            preds_full = _apply_behavior_adjustment_single_df(preds_full, behavior_hours, behavior_factor_norm)

        remaining_kwh_today, cutoff_iso = _compute_remaining_kwh_today(preds_full[["timestamp", "pred_consumption_Wh"]], target_date)

        df_show = preds_full.copy()
        temps_df_show = temps_df_full.copy()
        df_show, temps_df_show = _filter_today_for_display(df_show, temps_df_show, target_date)

        out_for_summary = dict(out)
        out_for_summary["total_kWh_day"] = float(preds_full["pred_consumption_Wh"].sum() / 1000.0)

        behavior_text = _behavior_summary_suffix(behavior_enabled, behavior_hours, behavior_factor_norm)
        summary = _build_summary_single(out_for_summary, remaining_kwh_today, cutoff_iso, behavior_text)
        fig = _build_single_chart(df_show[["timestamp", "pred_consumption_Wh"]])

        status = "✅ Done"
        if save_csv and not preds_full.empty:
            fp = _save_csv_full_day(out_for_summary, preds_full, temps24, remaining_kwh_today, cutoff_iso)
            status = f"✅ Saved CSV (24h): {fp}"

        return summary, df_show, temps_df_show, pd.DataFrame(), fig, status, gr.update(visible=False, value=None), gr.update(visible=False, value=None)

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


def do_compare(
    use_history: bool,
    model_choice: str,
    optimization: str,
    city: str,
    target_date: str,
    total_floor_area_m2: float,
    residents: float,
    hometype: str,
    urban_rural_class: str,
    num_electric_appliances: Optional[float],
    t_min: Optional[float],
    t_max: Optional[float],
    external_temperature_24h_text: str,
    internal_temperature_24h_text: str,
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

        effective_history_correction_max_alpha = _resolve_effective_history_alpha(
            use_history=use_history,
            use_recommended_history_alpha=use_recommended_history_alpha,
            recommended_history_correction_max_alpha=recommended_history_correction_max_alpha,
            manual_history_correction_max_alpha=history_correction_max_alpha,
        )

        payload, temps24_user = _prepare_payload(
            use_history,
            "auto — Auto default",
            optimization,
            city,
            target_date,
            total_floor_area_m2,
            residents,
            hometype,
            urban_rural_class,
            num_electric_appliances,
            t_min,
            t_max,
            external_temperature_24h_text,
            internal_temperature_24h_text,
            history_csv_file,
            history_csv_path_text,
            use_default_history_store,
            min_history_hours,
            history_consumption_text,
            apply_history_correction,
            history_correction_days,
            effective_history_correction_max_alpha,
        )

        results: Dict[str, Dict[str, Any]] = {}
        temps_df_show = pd.DataFrame()
        primary_df_show = pd.DataFrame()

        for key, model_id in compare_ids.items():
            payload_mid = dict(payload)
            payload_mid["model_id"] = model_id
            out_mid = _post_predict(payload_mid)
            results[key] = out_mid

            if key == "rf":
                preds_rf = _predictions_to_df(out_mid)
                if behavior_enabled:
                    preds_rf = _apply_behavior_adjustment_single_df(preds_rf, behavior_hours, behavior_factor_norm)
                primary_df_show = preds_rf.copy()

                temps24 = out_mid.get("external_temperature_24h") or temps24_user
                temps_df_full = _temps24_to_df(temps24)
                primary_df_show, temps_df_show = _filter_today_for_display(primary_df_show, temps_df_full.copy(), target_date)

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
            temps24_for_save = next(iter(results.values())).get("external_temperature_24h") or temps24_user
            fp = _save_csv_compare(
                city=city,
                target_date=target_date,
                mode=mode,
                optimization=optimization,
                curve_df=curve_df_for_save,
                comparison_df=comparison_df,
                temps24=temps24_for_save,
                flags=flags,
            )
            status = f"✅ Saved compare CSV: {fp}"

        return summary, primary_df_show, temps_df_show, comparison_df, comparison_fig, status, gr.update(visible=False, value=None), gr.update(visible=False, value=None)

    except ValueError as e:
        return str(e), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None, "⚠️ Fix inputs and try again.", gr.update(visible=False, value=None), gr.update(visible=False, value=None)

    except RuntimeError as e:
        msg = str(e)
        if msg.startswith("WEATHER_FALLBACK::"):
            detail = msg.split("WEATHER_FALLBACK::", 1)[1]
            return (
                "⚠️ Δεν βρέθηκαν αυτόματα δεδομένα εξωτερικής θερμοκρασίας.\n"
                "Συμπλήρωσε t_min και t_max ή δώσε 24 ωριαίες τιμές και ξαναπάτα Compare.",
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
    total_floor_area_m2: float,
    residents: float,
    hometype: str,
    urban_rural_class: str,
    num_electric_appliances: Optional[float],
    t_min: Optional[float],
    t_max: Optional[float],
    external_temperature_24h_text: str,
    internal_temperature_24h_text: str,
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

        effective_history_correction_max_alpha = _resolve_effective_history_alpha(
            use_history=use_history,
            use_recommended_history_alpha=use_recommended_history_alpha,
            recommended_history_correction_max_alpha=recommended_history_correction_max_alpha,
            manual_history_correction_max_alpha=history_correction_max_alpha,
        )

        payload, temps24_user = _prepare_payload(
            use_history,
            "auto — Auto default",
            optimization,
            city,
            target_date,
            total_floor_area_m2,
            residents,
            hometype,
            urban_rural_class,
            num_electric_appliances,
            t_min,
            t_max,
            external_temperature_24h_text,
            internal_temperature_24h_text,
            history_csv_file,
            history_csv_path_text,
            use_default_history_store,
            min_history_hours,
            history_consumption_text,
            apply_history_correction,
            history_correction_days,
            effective_history_correction_max_alpha,
        )

        results: Dict[str, Dict[str, Any]] = {}
        temps_df_show = pd.DataFrame()
        curve_df = None
        temps24_for_save = None

        for key, model_id in compare_ids.items():
            payload_mid = dict(payload)
            payload_mid["model_id"] = model_id
            out_mid = _post_predict(payload_mid)
            results[key] = out_mid

            preds_mid = _predictions_to_df(out_mid)

            if temps24_for_save is None:
                temps24_for_save = out_mid.get("external_temperature_24h") or temps24_user
                temps_df_show = _temps24_to_df(temps24_for_save)

            if not preds_mid.empty:
                temp_curve = preds_mid[["timestamp", "pred_consumption_Wh"]].copy()
                temp_curve.rename(columns={"pred_consumption_Wh": f"{key}_Wh"}, inplace=True)
                curve_df = temp_curve if curve_df is None else curve_df.merge(temp_curve, on="timestamp", how="outer")

        if curve_df is None or curve_df.empty:
            raise RuntimeError("No predictions available for combined forecast.")

        curve_df = curve_df.sort_values("timestamp").reset_index(drop=True)
        curve_df["ensemble_Wh"] = weights["rf"] * curve_df["rf_Wh"] + weights["xgb"] * curve_df["xgb_Wh"] + weights["lgbm"] * curve_df["lgbm_Wh"]

        if behavior_enabled:
            curve_df = _apply_behavior_adjustment_curve_df(
                curve_df,
                behavior_hours,
                behavior_factor_norm,
                target_cols=["rf_Wh", "xgb_Wh", "lgbm_Wh", "ensemble_Wh"],
            )

        total_kwh_day = float(curve_df["ensemble_Wh"].sum() / 1000.0)

        display_curve = curve_df[["timestamp", "ensemble_Wh"]].copy()
        display_curve.rename(columns={"ensemble_Wh": "pred_consumption_Wh"}, inplace=True)

        remaining_kwh_today, cutoff_iso = _compute_remaining_kwh_today_for_col(display_curve, target_date, "pred_consumption_Wh")
        display_curve, temps_df_show = _filter_today_for_display(display_curve, temps_df_show, target_date)

        behavior_text = _behavior_summary_suffix(behavior_enabled, behavior_hours, behavior_factor_norm)
        flags = results.get("rf", next(iter(results.values()))).get("derived_flags", {}) or {}
        summary = _build_summary_combined(total_kwh_day, remaining_kwh_today, cutoff_iso, city, target_date, mode, optimization, weights, flags, behavior_text)

        comparison_df = pd.DataFrame([
            {"model_key": "rf", "model_id": results["rf"].get("model_id"), "weight": weights["rf"], "total_kWh_day": float(curve_df["rf_Wh"].sum() / 1000.0), "history_correction_applied": results["rf"].get("history_correction_applied")},
            {"model_key": "xgb", "model_id": results["xgb"].get("model_id"), "weight": weights["xgb"], "total_kWh_day": float(curve_df["xgb_Wh"].sum() / 1000.0), "history_correction_applied": results["xgb"].get("history_correction_applied")},
            {"model_key": "lgbm", "model_id": results["lgbm"].get("model_id"), "weight": weights["lgbm"], "total_kWh_day": float(curve_df["lgbm_Wh"].sum() / 1000.0), "history_correction_applied": results["lgbm"].get("history_correction_applied")},
            {"model_key": "combined", "model_id": "combined", "weight": 1.0, "total_kWh_day": total_kwh_day, "history_correction_applied": None},
        ])

        fig_curve = _filter_today_curve_only(curve_df.copy(), target_date)
        fig = _build_compare_chart(fig_curve[["timestamp", "rf_Wh", "xgb_Wh", "lgbm_Wh", "ensemble_Wh"]])

        status = "✅ Combined prediction done"
        if save_csv:
            fp = _save_csv_combined(city, target_date, mode, optimization, curve_df, temps24_for_save, remaining_kwh_today, cutoff_iso, weights, flags)
            status = f"✅ Saved combined CSV: {fp}"

        return summary, display_curve, temps_df_show, comparison_df, fig, status, gr.update(visible=False, value=None), gr.update(visible=False, value=None)

    except ValueError as e:
        return str(e), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None, "⚠️ Fix inputs and try again.", gr.update(visible=False, value=None), gr.update(visible=False, value=None)

    except RuntimeError as e:
        msg = str(e)
        if msg.startswith("WEATHER_FALLBACK::"):
            detail = msg.split("WEATHER_FALLBACK::", 1)[1]
            return (
                "⚠️ Δεν βρέθηκαν αυτόματα δεδομένα εξωτερικής θερμοκρασίας.\n"
                "Συμπλήρωσε t_min και t_max ή δώσε 24 ωριαίες τιμές και ξαναπάτα Combined Prediction.",
                pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None,
                f"API 422: {detail}",
                gr.update(visible=True, value=None), gr.update(visible=True, value=None),
            )
        return msg, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None, "❌ API error", gr.update(visible=False, value=None), gr.update(visible=False, value=None)

    except Exception as e:
        return f"Σφάλμα επικοινωνίας με API: {e}", pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None, "❌ API not reachable", gr.update(visible=False, value=None), gr.update(visible=False, value=None)


# ============================================================
# UI CALLBACKS
# ============================================================

INITIAL_MODE = MODE_NO_HISTORY
INITIAL_MODEL_CHOICES = _model_choices_for_mode(INITIAL_MODE)
INITIAL_DEFAULT_MODEL = INITIAL_MODEL_CHOICES[0] if INITIAL_MODEL_CHOICES else "auto — Auto default"


def _on_use_history_toggle(use_history: bool):
    mode = MODE_WITH_HISTORY if use_history else MODE_NO_HISTORY
    choices = _model_choices_for_mode(mode)
    default_choice = choices[0] if choices else "auto — Auto default"
    return gr.update(visible=use_history), gr.update(choices=choices, value=default_choice)


def _toggle_custom_weights(use_custom: bool):
    return gr.update(visible=use_custom)


def _reset_temps_on_change(_):
    return gr.update(visible=False, value=None), gr.update(visible=False, value=None)


def _refresh_api_status():
    mode = MODE_NO_HISTORY
    choices = _model_choices_for_mode(mode)
    return _api_health(), gr.update(choices=choices, value=(choices[0] if choices else "auto — Auto default"))


# ============================================================
# UI BUILD
# ============================================================

with gr.Blocks(title="IDEAL Forecasting UI") as demo:
    gr.Markdown("## IDEAL Load Forecasting — Generic API/UI Demo")
    gr.Markdown(
        "Συμπλήρωσε τα χαρακτηριστικά της κατοικίας και πάτα **Predict** για ένα μοντέλο, "
        "**Compare** για σύγκριση RF / XGB / LGBM ή **Combined Prediction** για weighted ensemble. "
    )

    with gr.Row():
        api_status = gr.Markdown(_api_health())
        btn_refresh = gr.Button("Refresh API/models")

    with gr.Row():
        use_history = gr.Checkbox(
            value=False,
            label="Πρόβλεψη με χρήση ιστορικών δεδομένων",
            info="Αν ενεργοποιηθεί, το API χρησιμοποιεί mode='with_history' και απαιτεί πρόσφατο ιστορικό κατανάλωσης.",
        )

    with gr.Row():
        model_choice = gr.Dropdown(
            choices=INITIAL_MODEL_CHOICES,
            value=INITIAL_DEFAULT_MODEL,
            label="Μοντέλο",
            info="Auto default: RF για no-history, LGBM για with-history.",
        )
        optimization = gr.Dropdown(
            choices=["balanced", "daily"],
            value="balanced",
            label="Optimization",
            info="daily χρησιμοποιεί daily-calibrated output όπου υπάρχει διαθέσιμο.",
        )

    with gr.Row():
        city = gr.Dropdown(choices=DEFAULT_CITIES, value="Edinburgh", label="City")
        target_date = gr.Textbox(value=_today_date().isoformat(), label="Target date (YYYY-MM-DD)")

    with gr.Row():
        total_floor_area_m2 = gr.Number(value=85, minimum=1, label="Total floor area (m²)")
        residents = gr.Number(value=2, minimum=1, label="Residents")
        num_electric_appliances = gr.Number(value=None, minimum=0, label="Number of electric appliances [optional]")

    with gr.Row():
        hometype = gr.Dropdown(choices=[x[0] for x in DEFAULT_HOMETYPES], value="flat", label="Home type")
        urban_rural_class = gr.Dropdown(choices=URBAN_RURAL_OPTIONS, value="1", label="Urban/Rural class")

    with gr.Accordion("Weather / environmental inputs", open=True):
        gr.Markdown(
            "Το API προσπαθεί να πάρει αυτόματα εξωτερική θερμοκρασία από Open-Meteo. "
            "Αν δεν βρεθεί, συμπλήρωσε **t_min / t_max** ή δώσε 24 ωριαίες τιμές."
        )
        with gr.Row():
            t_min = gr.Number(value=None, label="t_min (°C)", visible=False)
            t_max = gr.Number(value=None, label="t_max (°C)", visible=False)

        external_temperature_24h_text = gr.Textbox(
            value="",
            label="external_temperature_24h (24 τιμές, comma-separated) [optional]",
            placeholder="π.χ. 3.1, 2.9, 2.7, ... (24 συνολικά)",
        )

        internal_temperature_24h_text = gr.Textbox(
            value="",
            label="internal_temperature_24h (24 τιμές, comma-separated) [optional]",
            placeholder="π.χ. 19.5, 19.4, 19.3, ... (24 συνολικά)",
        )

    with gr.Group(visible=False) as history_group:
        gr.Markdown("### History options — μόνο για with-history mode")
        gr.Markdown(
            "Το ιστορικό αρχείο θεωρείται ότι αφορά **μία μόνο κατοικία**. "
            "Αν θέλεις άλλη κατοικία, επίλεξε άλλο CSV. "
            "Απαιτούμενες στήλες: `timestamp`, `consumption_Wh`."
        )

        history_csv_file = gr.UploadButton(
            "Select history CSV",
            file_types=[".csv"],
            file_count="single",
            type="filepath",
        )
        history_csv_path_text = gr.Textbox(
            value="",
            label="Selected history CSV path [optional]",
            placeholder="Select a CSV using the button above or paste a local path manually.",
            interactive=True,
        )
        history_upload_status = gr.Markdown("No history CSV selected.")
        use_default_history_store = gr.Checkbox(
            value=True,
            label="Use legacy default history_store.csv if no file/path is selected",
            info="Fallback μόνο αν δεν δοθεί upload/path/manual history.",
        )
        min_history_hours = gr.Number(value=168, precision=0, minimum=1, label="min_history_hours")
        history_consumption_text = gr.Textbox(
            value="",
            label="history_consumption_Wh (comma-separated) [optional]",
            placeholder="π.χ. 120, 98, 105, ... (>=168 τιμές αν το δώσεις εσύ)",
        )

        with gr.Accordion("Adaptive history-aware correction", open=True):
            apply_history_correction = gr.Checkbox(
                value=True,
                label="Apply adaptive 7-day median shape correction",
                info="Εφαρμόζεται μόνο στο with-history mode, αν υπάρχει επαρκές πρόσφατο ιστορικό.",
            )
            with gr.Row():
                history_correction_days = gr.Number(
                    value=7,
                    precision=0,
                    minimum=1,
                    maximum=30,
                    label="history_correction_days",
                )
                history_correction_max_alpha = gr.Number(
                    value=0.20,
                    minimum=0.0,
                    maximum=1.0,
                    label="history_correction_max_alpha",
                    info="Manual value. If 'Use recommended max alpha' is active, the recommended value below is used instead.",
                )

            with gr.Row():
                btn_analyze_history_stability = gr.Button(
                    "Analyze history stability / Recommend max alpha",
                    variant="secondary",
                )
                use_recommended_history_alpha = gr.Checkbox(
                    value=False,
                    label="Use recommended max alpha",
                    info="If checked, Predict/Compare/Combined uses the recommended max_alpha instead of the manual field.",
                )

            with gr.Row():
                history_stability_score = gr.Textbox(
                    value="",
                    label="History stability score",
                    interactive=False,
                )
                history_stability_category = gr.Textbox(
                    value="Outdated / not analyzed",
                    label="History stability category",
                    interactive=False,
                )
                recommended_history_correction_max_alpha = gr.Number(
                    value=None,
                    minimum=0.0,
                    maximum=0.65,
                    label="Recommended max_alpha",
                    info="Computed from history stability. Excellent stability can recommend up to 0.60; values above 0.65 should remain manual/experimental.",
                    interactive=False,
                )

            history_stability_explanation = gr.Textbox(
                value="Click Analyze history stability after selecting history CSV and target date. Categories: Low, Medium, High, Very High, Excellent.",
                label="History stability explanation",
                lines=8,
                interactive=False,
            )
            history_stability_status = gr.Markdown("History stability recommendation not calculated yet.")

    with gr.Accordion("Behavioral adjustment (optional)", open=False):
        gr.Markdown(
            "Προαιρετικό post-processing των προβλέψεων. "
            "Δήλωσε ώρες, π.χ. **7-9,18-23**. "
            "Factor > 1 αυξάνει την κατανάλωση, factor < 1 τη μειώνει."
        )
        enable_behavior_adjustment = gr.Checkbox(value=False, label="Enable behavioral adjustment")
        high_consumption_hours_text = gr.Textbox(value="", label="Adjusted hours", placeholder="π.χ. 7-9, 13-15, 18-23")
        behavior_factor = gr.Number(value=1.15, minimum=0.01, label="Behavior factor")

    with gr.Accordion("Ensemble weights (optional)", open=False):
        gr.Markdown("Τα βάρη του combined prediction πρέπει να αθροίζουν σε 100%.")
        use_custom_weights = gr.Checkbox(value=False, label="Use custom ensemble weights")
        with gr.Row(visible=False) as weights_group:
            rf_weight_pct = gr.Number(value=40, minimum=0, maximum=100, label="RF weight (%)")
            xgb_weight_pct = gr.Number(value=30, minimum=0, maximum=100, label="XGB weight (%)")
            lgbm_weight_pct = gr.Number(value=30, minimum=0, maximum=100, label="LGBM weight (%)")

    with gr.Row():
        save_csv = gr.Checkbox(value=False, label="Save prediction CSV")

    with gr.Row():
        btn_predict = gr.Button("Predict", variant="primary")
        btn_compare = gr.Button("Compare")
        btn_combined = gr.Button("Combined Prediction")

    with gr.Row():
        summary = gr.Textbox(label="Summary", lines=18)

    with gr.Row():
        table = gr.Dataframe(label="Predictions — single model / combined / RF preview in compare", interactive=False)

    with gr.Row():
        temps_table = gr.Dataframe(label="External temperature used", interactive=False)

    with gr.Row():
        comparison_table = gr.Dataframe(label="Model comparison", interactive=False)

    with gr.Row():
        comparison_plot = gr.Plot(label="Prediction / comparison chart")

    with gr.Row():
        status = gr.Textbox(label="Status", lines=2)

    history_csv_file.upload(
        fn=register_history_csv_upload,
        inputs=[history_csv_file],
        outputs=[history_csv_path_text, history_upload_status],
    )

    btn_analyze_history_stability.click(
        fn=analyze_history_stability_recommendation,
        inputs=[
            history_csv_file,
            history_csv_path_text,
            target_date,
            history_correction_days,
            min_history_hours,
            history_consumption_text,
        ],
        outputs=[
            history_stability_score,
            history_stability_category,
            recommended_history_correction_max_alpha,
            history_stability_explanation,
            history_stability_status,
        ],
    )

    history_csv_path_text.change(
        fn=_reset_history_stability_recommendation,
        inputs=[history_csv_path_text],
        outputs=[
            history_stability_score,
            history_stability_category,
            recommended_history_correction_max_alpha,
            history_stability_explanation,
            history_stability_status,
            use_recommended_history_alpha,
        ],
    )
    target_date.change(
        fn=_reset_history_stability_recommendation,
        inputs=[target_date],
        outputs=[
            history_stability_score,
            history_stability_category,
            recommended_history_correction_max_alpha,
            history_stability_explanation,
            history_stability_status,
            use_recommended_history_alpha,
        ],
    )
    history_consumption_text.change(
        fn=_reset_history_stability_recommendation,
        inputs=[history_consumption_text],
        outputs=[
            history_stability_score,
            history_stability_category,
            recommended_history_correction_max_alpha,
            history_stability_explanation,
            history_stability_status,
            use_recommended_history_alpha,
        ],
    )
    history_correction_days.change(
        fn=_reset_history_stability_recommendation,
        inputs=[history_correction_days],
        outputs=[
            history_stability_score,
            history_stability_category,
            recommended_history_correction_max_alpha,
            history_stability_explanation,
            history_stability_status,
            use_recommended_history_alpha,
        ],
    )

    use_history.change(fn=_on_use_history_toggle, inputs=[use_history], outputs=[history_group, model_choice])
    use_custom_weights.change(fn=_toggle_custom_weights, inputs=[use_custom_weights], outputs=[weights_group])
    target_date.change(fn=_reset_temps_on_change, inputs=[target_date], outputs=[t_min, t_max])
    city.change(fn=_reset_temps_on_change, inputs=[city], outputs=[t_min, t_max])
    btn_refresh.click(fn=_refresh_api_status, inputs=[], outputs=[api_status, model_choice])

    common_inputs = [
        use_history,
        model_choice,
        optimization,
        city,
        target_date,
        total_floor_area_m2,
        residents,
        hometype,
        urban_rural_class,
        num_electric_appliances,
        t_min,
        t_max,
        external_temperature_24h_text,
        internal_temperature_24h_text,
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
        summary,
        table,
        temps_table,
        comparison_table,
        comparison_plot,
        status,
        t_min,
        t_max,
    ]

    btn_predict.click(fn=do_predict, inputs=common_inputs, outputs=common_outputs)
    btn_compare.click(fn=do_compare, inputs=common_inputs, outputs=common_outputs)
    btn_combined.click(fn=do_combined, inputs=common_inputs, outputs=common_outputs)


if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_name=UI_HOST,
        server_port=UI_PORT,
        share=False,
        inbrowser=False,
        show_error=True,
        quiet=False,
        show_api=False,
    )
