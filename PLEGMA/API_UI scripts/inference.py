# C:/Plegma_Programming/src/inference.py
# ============================================================
# Generic API/UI inference for final PLEGMA models
# ============================================================
#
# Final UI assumptions:
#   - The user does NOT provide home_id.
#   - Model artifacts are selected through model_registry.py.
#   - Every final artifact folder contains:
#       model.joblib
#       preprocessor.pkl
#       feature_config.json
#       metadata.json                  optional but recommended
#       daily_calibrator.json           optional for daily-calibrated output
#       global_daily_calibrator.json    optional legacy/alternative name
#
# Supported modes:
#   - no_history
#   - with_history
#   - backward-compatible aliases: coldstart, cold_start, withhistory
#
# Final PLEGMA routing is defined in model_registry.py.
# Ports/pages are intentionally not defined here; they remain in api_app.py/ui_app.py.
#
# Notes:
#   - home_id is not a user input and is not used as a model feature.
#   - With-history prediction is recursive for the 24-hour forecast horizon:
#     each predicted hour is appended to the history and used for later lags.
#   - This file keeps PLEGMA-specific inputs such as internal/external humidity,
#     appliance flags, income/heating/water-heater categories, etc.
# ============================================================

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

try:
    from src.model_registry import (
        DEFAULT_MODE,
        MODE_NO_HISTORY,
        MODE_WITH_HISTORY,
        normalize_mode,
        validate_model_for_mode,
    )
except ImportError:
    try:
        from model_registry import (
            DEFAULT_MODE,
            MODE_NO_HISTORY,
            MODE_WITH_HISTORY,
            normalize_mode,
            validate_model_for_mode,
        )
    except ImportError:
        # Backward-compatible minimal fallback.
        DEFAULT_MODE = "no_history"
        MODE_NO_HISTORY = "no_history"
        MODE_WITH_HISTORY = "with_history"

        def normalize_mode(mode: Optional[str]) -> str:
            if not mode:
                return DEFAULT_MODE
            value = str(mode).strip().lower().replace("-", "_").replace(" ", "_")
            compact = value.replace("_", "")
            if value in {"no_history", "cold_start", "coldstart"} or compact in {"nohistory", "coldstart"}:
                return MODE_NO_HISTORY
            if value in {"with_history", "withhistory"} or compact in {"withhistory"}:
                return MODE_WITH_HISTORY
            raise ValueError(f"Unsupported prediction mode: {mode}.")

        def validate_model_for_mode(model_id: str, mode: Optional[str]) -> Dict[str, Any]:
            raise ImportError(
                "model_registry.py is required for final PLEGMA inference. "
                "Place model_registry.py under C:/Plegma_Programming/src."
            )

# Compatibility constants for older imports/calls.
MODE_COLDSTART = MODE_NO_HISTORY
MODE_WITHHISTORY = MODE_WITH_HISTORY


# ============================================================
# Paths / constants
# ============================================================

DEFAULT_BASE_DIR = Path("C:/Plegma_Programming")
DEFAULT_OUT_DIR = DEFAULT_BASE_DIR / "processed" / "predictions" / "api_like"
DEFAULT_HOLIDAYS_CSV = DEFAULT_BASE_DIR / "holidays.csv"

TIMESTAMP_COL = "timestamp"
TARGET_COL = "consumption_Wh"
PRED_COL = "pred_consumption_Wh"
UNKNOWN_LABEL = "unknown"

DEFAULT_HISTORY_DAYS_FOR_CORRECTION = 7
DEFAULT_HISTORY_CORRECTION_MAX_ALPHA = 0.20
DEFAULT_HISTORY_CORRECTION_MIN_HOURLY_OBS = 3
DEFAULT_HISTORY_CORRECTION_MIN_TOTAL_ROWS = 120
DEFAULT_HISTORY_COVERAGE_THRESHOLD = 0.85


# ============================================================
# Holidays
# ============================================================

def _load_holidays_set(holidays_csv: Path = DEFAULT_HOLIDAYS_CSV) -> set:
    holidays_csv = Path(holidays_csv)
    if not holidays_csv.exists():
        return set()

    hdf = pd.read_csv(holidays_csv)
    if "date" not in hdf.columns:
        return set()

    dates = pd.to_datetime(hdf["date"], errors="coerce").dt.date.dropna()
    return set(dates.tolist())


_HOLIDAYS_SET = _load_holidays_set()


# ============================================================
# JSON / artifact helpers
# ============================================================

def _load_json_if_exists(path: Path, default: Any) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_joblib_load(path: Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    return joblib.load(path)


def load_model_artifacts(
    model_id: Optional[str] = "auto",
    mode: Optional[str] = DEFAULT_MODE,
    optimization: Optional[str] = "balanced",
) -> Dict[str, Any]:
    """Load one concrete final API model artifact bundle via model_registry.py."""
    cfg = validate_model_for_mode(model_id or "auto", mode)
    artifact_dir = Path(cfg["artifact_dir"])

    model_path = artifact_dir / "model.joblib"
    preprocessor_path = artifact_dir / "preprocessor.pkl"
    feature_config_path = artifact_dir / "feature_config.json"
    metadata_path = artifact_dir / "metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Missing preprocessor file: {preprocessor_path}")
    if not feature_config_path.exists():
        raise FileNotFoundError(f"Missing feature_config file: {feature_config_path}")

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    feature_config = _load_json_if_exists(feature_config_path, {})
    metadata = _load_json_if_exists(metadata_path, {})

    # Different training scripts may use either name. Prefer daily_calibrator.json.
    daily_calibrator = _load_json_if_exists(artifact_dir / "daily_calibrator.json", None)
    if daily_calibrator is None:
        daily_calibrator = _load_json_if_exists(artifact_dir / "global_daily_calibrator.json", None)

    return {
        "model": model,
        "preprocessor": preprocessor,
        "feature_config": feature_config,
        "metadata": metadata,
        "registry_config": cfg,
        "model_id": cfg["model_id"],
        "mode": cfg["mode"],
        "model_type": cfg.get("type"),
        "artifact_dir": str(artifact_dir),
        "daily_calibrator": daily_calibrator,
    }


# Backward-compatible wrappers. These load registry-selected final models.
def load_rf_artifacts(mode: str = MODE_NO_HISTORY) -> Dict[str, Any]:
    return load_model_artifacts(model_id="rf", mode=mode)


def load_lgbm_artifacts(mode: str = MODE_NO_HISTORY) -> Dict[str, Any]:
    return load_model_artifacts(model_id="lgbm", mode=mode)


def load_xgb_artifacts(mode: str = MODE_NO_HISTORY) -> Dict[str, Any]:
    return load_model_artifacts(model_id="xgb", mode=mode)


# ============================================================
# Time / vector helpers
# ============================================================

def _season_from_month(month: int) -> str:
    if month in [12, 1, 2]:
        return "winter"
    if month in [3, 4, 5]:
        return "spring"
    if month in [6, 7, 8]:
        return "summer"
    return "autumn"


def _time_features(ts: pd.Timestamp) -> Dict[str, Any]:
    ts = pd.Timestamp(ts)
    dow = ts.dayofweek

    return {
        "hour": int(ts.hour),
        "day_of_week": int(dow),
        "month": int(ts.month),
        "day_of_month": int(ts.day),
        "week_of_year": int(ts.isocalendar().week),
        "is_weekend": int(dow >= 5),
        "is_holiday": int(ts.date() in _HOLIDAYS_SET),
        "season": _season_from_month(int(ts.month)),
        "hour_sin": float(np.sin(2 * np.pi * ts.hour / 24)),
        "hour_cos": float(np.cos(2 * np.pi * ts.hour / 24)),
        "day_sin": float(np.sin(2 * np.pi * dow / 7)),
        "day_cos": float(np.cos(2 * np.pi * dow / 7)),
        "month_sin": float(np.sin(2 * np.pi * ts.month / 12)),
        "month_cos": float(np.cos(2 * np.pi * ts.month / 12)),
    }


def _build_24h_timestamps(target_date: str) -> pd.DatetimeIndex:
    start = pd.Timestamp(target_date).normalize()
    return pd.date_range(start=start, periods=24, freq="h")


def _ensure_len(arr: Iterable[Any], n: int, name: str) -> List[Any]:
    values = list(arr)
    if len(values) != n:
        raise ValueError(f"{name} must have length {n}, but has {len(values)}.")
    return values


def _normalize_24h_numeric(values, name: str, default: float = np.nan) -> List[float]:
    if values is None:
        return [float(default)] * 24
    values = _ensure_len(values, 24, name)
    return [float(x) if pd.notna(x) else float(default) for x in values]


def _as_float_or_nan(value) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except Exception:
        return float("nan")


def _as_label(value) -> str:
    if value is None:
        return UNKNOWN_LABEL
    text = str(value)
    if text.lower() in {"nan", "none", ""}:
        return UNKNOWN_LABEL
    return text


def _base_static_features(
    num_rooms: float = None,
    residents: float = None,
    num_adults: float = None,
    num_children: float = None,
    num_elderly: float = None,
    has_ac: float = None,
    has_fridge_freezer: float = None,
    has_dryer: float = None,
    has_washing_machine: float = None,
    has_dishwasher: float = None,
    has_microwave: float = None,
    has_electric_oven: float = None,
    has_electric_hob: float = None,
    solar_panels: float = None,
    building_type: str = UNKNOWN_LABEL,
    build_era: str = UNKNOWN_LABEL,
    income_band: str = UNKNOWN_LABEL,
    heating_type: str = UNKNOWN_LABEL,
    water_heater_type: str = UNKNOWN_LABEL,
    homeowner_status: str = UNKNOWN_LABEL,
    years_in_house: str = UNKNOWN_LABEL,
    occupation: str = UNKNOWN_LABEL,
) -> Dict[str, Any]:
    return {
        "num_rooms": _as_float_or_nan(num_rooms),
        "residents": _as_float_or_nan(residents),
        "num_adults": _as_float_or_nan(num_adults),
        "num_children": _as_float_or_nan(num_children),
        "num_elderly": _as_float_or_nan(num_elderly),
        "has_ac": _as_float_or_nan(has_ac),
        "has_fridge_freezer": _as_float_or_nan(has_fridge_freezer),
        "has_dryer": _as_float_or_nan(has_dryer),
        "has_washing_machine": _as_float_or_nan(has_washing_machine),
        "has_dishwasher": _as_float_or_nan(has_dishwasher),
        "has_microwave": _as_float_or_nan(has_microwave),
        "has_electric_oven": _as_float_or_nan(has_electric_oven),
        "has_electric_hob": _as_float_or_nan(has_electric_hob),
        "solar_panels": _as_float_or_nan(solar_panels),
        "building_type": _as_label(building_type),
        "build_era": _as_label(build_era),
        "income_band": _as_label(income_band),
        "heating_type": _as_label(heating_type),
        "water_heater_type": _as_label(water_heater_type),
        "homeowner_status": _as_label(homeowner_status),
        "years_in_house": _as_label(years_in_house),
        "occupation": _as_label(occupation),
    }


# ============================================================
# Target inverse / prediction helpers
# ============================================================

def _inverse_target_pred(y_pred, feature_config: Dict[str, Any]) -> np.ndarray:
    y = np.asarray(y_pred, dtype=np.float64)

    if feature_config.get("use_log_target", False):
        y = np.expm1(y)

    y = np.maximum(y, 0.0)
    return y.astype("float32")


def _predict_raw_model(model, X, model_type: Optional[str] = None) -> np.ndarray:
    """Predict while respecting XGBoost best_iteration when available."""
    if model_type == "xgboost":
        best_iteration = getattr(model, "best_iteration", None)
        if best_iteration is not None:
            try:
                return model.predict(X, iteration_range=(0, int(best_iteration) + 1))
            except TypeError:
                pass

    return model.predict(X)


def _apply_daily_calibrator_if_requested(
    preds: np.ndarray,
    artifacts: Dict[str, Any],
    use_daily_calibrated: bool = False,
) -> np.ndarray:
    preds = np.asarray(preds, dtype=np.float64)

    if not use_daily_calibrated:
        return np.maximum(preds, 0.0).astype("float32")

    calibrator = artifacts.get("daily_calibrator")
    if not calibrator:
        return np.maximum(preds, 0.0).astype("float32")

    ctype = calibrator.get("type", "")
    if ctype in {"global_scale", "daily_scale"}:
        scale = float(calibrator.get("scale", 1.0))
        return np.maximum(preds * scale, 0.0).astype("float32")

    return np.maximum(preds, 0.0).astype("float32")


# ============================================================
# Feature frame alignment for sklearn preprocessors
# ============================================================

def _feature_columns_from_config(feature_config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    numeric_cols = list(feature_config.get("numeric_cols", []))
    categorical_cols = list(feature_config.get("categorical_cols", []))
    return numeric_cols, categorical_cols


def _prepare_feature_frame_for_preprocessor(
    df_feat: pd.DataFrame,
    feature_config: Dict[str, Any],
) -> pd.DataFrame:
    """Ensure all expected feature columns exist before preprocessor.transform."""
    df = df_feat.copy()
    numeric_cols, categorical_cols = _feature_columns_from_config(feature_config)

    for c in numeric_cols:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in categorical_cols:
        if c not in df.columns:
            df[c] = UNKNOWN_LABEL
        df[c] = df[c].fillna(UNKNOWN_LABEL).astype(str).replace({"nan": UNKNOWN_LABEL, "None": UNKNOWN_LABEL})

    return df[numeric_cols + categorical_cols]


def _transform_features(artifacts: Dict[str, Any], df_feat: pd.DataFrame):
    feature_config = artifacts["feature_config"]
    preprocessor = artifacts["preprocessor"]
    X_df = _prepare_feature_frame_for_preprocessor(df_feat, feature_config)
    return preprocessor.transform(X_df)


# ============================================================
# History feature helpers
# ============================================================

def _history_value(hist_deque: deque, lag: int) -> float:
    if len(hist_deque) >= lag:
        return float(hist_deque[-lag])
    if len(hist_deque) > 0:
        return float(hist_deque[-1])
    return 0.0


def _history_window_values(hist_deque: deque, window: int) -> List[float]:
    values = list(hist_deque)
    if len(values) == 0:
        return [0.0]
    return [float(x) for x in values[-window:]]


def _add_recursive_history_features(row: Dict[str, Any], hist_deque: deque, feature_config: Dict[str, Any]) -> Dict[str, Any]:
    lag_hours = feature_config.get("lag_hours", [1, 2, 3, 6, 12, 24, 48, 72, 168])
    rolling_windows = feature_config.get("rolling_windows", [3, 6, 12, 24, 48, 168])
    rolling_extreme_windows = feature_config.get("rolling_extreme_windows", [24, 48, 168])

    for lag in lag_hours:
        lag = int(lag)
        row[f"lag_{lag}h"] = _history_value(hist_deque, lag)

    for w in rolling_windows:
        w = int(w)
        vals = _history_window_values(hist_deque, w)
        row[f"roll_mean_{w}h"] = float(np.mean(vals))

    for w in rolling_extreme_windows:
        w = int(w)
        vals = np.asarray(_history_window_values(hist_deque, w), dtype=np.float64)
        row[f"roll_std_{w}h"] = float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0
        row[f"roll_min_{w}h"] = float(np.min(vals))
        row[f"roll_max_{w}h"] = float(np.max(vals))

    # Derived lag/rolling features used by final training scripts.
    row["lag_1h_minus_24h"] = float(row.get("lag_1h", _history_value(hist_deque, 1))) - float(row.get("lag_24h", _history_value(hist_deque, 24)))
    row["lag_24h_minus_168h"] = float(row.get("lag_24h", _history_value(hist_deque, 24))) - float(row.get("lag_168h", _history_value(hist_deque, 168)))
    row["lag_48h_minus_168h"] = float(row.get("lag_48h", _history_value(hist_deque, 48))) - float(row.get("lag_168h", _history_value(hist_deque, 168)))
    row["roll_24h_div_168h"] = float(row.get("roll_mean_24h", np.mean(_history_window_values(hist_deque, 24)))) / (float(row.get("roll_mean_168h", np.mean(_history_window_values(hist_deque, 168)))) + 1.0)
    row["roll_3h_div_24h"] = float(row.get("roll_mean_3h", np.mean(_history_window_values(hist_deque, 3)))) / (float(row.get("roll_mean_24h", np.mean(_history_window_values(hist_deque, 24)))) + 1.0)

    return row


# ============================================================
# Single-home history CSV loading and adaptive correction helpers
# ============================================================

def load_single_home_history_csv(history_csv_path: Any) -> pd.DataFrame:
    """Load a single-home history CSV with timestamp and consumption_Wh columns."""
    if history_csv_path is None:
        raise ValueError("history_csv_path is required.")

    raw_path = history_csv_path
    if isinstance(raw_path, dict):
        raw_path = raw_path.get("name") or raw_path.get("path") or raw_path.get("orig_name")
    elif hasattr(raw_path, "name"):
        raw_path = raw_path.name

    path = Path(str(raw_path))
    if not path.exists():
        raise FileNotFoundError(f"History CSV not found: {path}")

    history_df = pd.read_csv(path, low_memory=False)
    return _prepare_history_dataframe(history_df)


def _prepare_history_dataframe(history_df: pd.DataFrame) -> pd.DataFrame:
    if history_df is None or len(history_df) == 0:
        raise ValueError("History dataframe is empty.")

    df = history_df.copy()

    if TIMESTAMP_COL not in df.columns:
        for alt in ["time", "datetime", "date_time", "date", "ds"]:
            if alt in df.columns:
                df = df.rename(columns={alt: TIMESTAMP_COL})
                break

    if TARGET_COL not in df.columns:
        for alt in ["consumption", "energy_Wh", "Wh", "value", "y"]:
            if alt in df.columns:
                df = df.rename(columns={alt: TARGET_COL})
                break

    if TIMESTAMP_COL not in df.columns:
        raise ValueError(f"History CSV must contain column '{TIMESTAMP_COL}'.")
    if TARGET_COL not in df.columns:
        raise ValueError(f"History CSV must contain column '{TARGET_COL}'.")

    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna(subset=[TIMESTAMP_COL, TARGET_COL]).copy()
    df[TARGET_COL] = df[TARGET_COL].clip(lower=0)

    if len(df) == 0:
        raise ValueError("History dataframe has no valid timestamp/consumption rows.")

    df = (
        df
        .assign(_hour=df[TIMESTAMP_COL].dt.floor("h"))
        .groupby("_hour", as_index=False)
        .agg(**{TARGET_COL: (TARGET_COL, "mean")})
        .rename(columns={"_hour": TIMESTAMP_COL})
        .sort_values(TIMESTAMP_COL)
        .reset_index(drop=True)
    )

    return df


def _history_df_from_vector(history_consumption_Wh, target_date: str) -> pd.DataFrame:
    values = [float(x) for x in list(history_consumption_Wh)]
    if len(values) == 0:
        raise ValueError("history_consumption_Wh is empty.")

    target_start = pd.Timestamp(target_date).normalize()
    timestamps = pd.date_range(
        end=target_start - pd.Timedelta(hours=1),
        periods=len(values),
        freq="h",
    )

    return pd.DataFrame({
        TIMESTAMP_COL: timestamps,
        TARGET_COL: np.maximum(np.asarray(values, dtype=np.float64), 0.0),
    })


def prepare_history_for_prediction_from_csv(
    history_csv_path: Any,
    target_date: str,
    min_history_hours: int = 168,
    coverage_threshold: float = DEFAULT_HISTORY_COVERAGE_THRESHOLD,
) -> Tuple[List[float], pd.DataFrame, Dict[str, Any]]:
    history_df = load_single_home_history_csv(history_csv_path)
    return prepare_history_for_prediction_from_dataframe(
        history_df=history_df,
        target_date=target_date,
        min_history_hours=min_history_hours,
        coverage_threshold=coverage_threshold,
    )


def prepare_history_for_prediction_from_dataframe(
    history_df: pd.DataFrame,
    target_date: str,
    min_history_hours: int = 168,
    coverage_threshold: float = DEFAULT_HISTORY_COVERAGE_THRESHOLD,
) -> Tuple[List[float], pd.DataFrame, Dict[str, Any]]:
    df = _prepare_history_dataframe(history_df)

    target_start = pd.Timestamp(target_date).normalize()
    history_start = target_start - pd.Timedelta(hours=int(min_history_hours))

    df = df[df[TIMESTAMP_COL] < target_start].copy()
    if len(df) == 0:
        raise ValueError("No history rows exist before target_date.")

    required_index = pd.date_range(
        start=history_start,
        end=target_start - pd.Timedelta(hours=1),
        freq="h",
    )

    hourly = df.set_index(TIMESTAMP_COL).reindex(required_index)
    observed_count = int(hourly[TARGET_COL].notna().sum())
    required_count = int(len(required_index))
    coverage = observed_count / max(required_count, 1)

    if observed_count == 0:
        raise ValueError("No usable hourly history rows found in the required window.")

    if coverage < float(coverage_threshold):
        raise ValueError(
            f"Insufficient recent history coverage before target_date. "
            f"Required window: {required_count} hours, observed: {observed_count} "
            f"({coverage:.1%})."
        )

    hourly[TARGET_COL] = (
        hourly[TARGET_COL]
        .astype(float)
        .interpolate(method="time", limit_direction="both")
        .ffill()
        .bfill()
        .clip(lower=0)
    )

    history_regularized_df = hourly.reset_index().rename(columns={"index": TIMESTAMP_COL})
    values = history_regularized_df[TARGET_COL].astype(float).tolist()

    daily = history_regularized_df.assign(date=history_regularized_df[TIMESTAMP_COL].dt.date).groupby("date")[TARGET_COL].sum() / 1000.0

    info = {
        "history_source": "csv_or_dataframe",
        "target_date": str(target_start.date()),
        "history_start": str(history_regularized_df[TIMESTAMP_COL].min()),
        "history_end": str(history_regularized_df[TIMESTAMP_COL].max()),
        "required_history_hours": required_count,
        "observed_history_hours": observed_count,
        "coverage": float(coverage),
        "filled_history_hours": int(required_count - observed_count),
        "history_total_kWh": float(history_regularized_df[TARGET_COL].sum() / 1000.0),
        "history_mean_Wh": float(history_regularized_df[TARGET_COL].mean()),
        "history_median_Wh": float(history_regularized_df[TARGET_COL].median()),
        "history_mean_daily_consumption_kWh": float(daily.mean()) if len(daily) else float("nan"),
    }

    return values, history_regularized_df, info


def _build_recent_history_profile(
    history_df: pd.DataFrame,
    target_date: str,
    history_days: int = DEFAULT_HISTORY_DAYS_FOR_CORRECTION,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    df = _prepare_history_dataframe(history_df)

    target_start = pd.Timestamp(target_date).normalize()
    start = target_start - pd.Timedelta(days=int(history_days))
    recent = df[(df[TIMESTAMP_COL] >= start) & (df[TIMESTAMP_COL] < target_start)].copy()

    if len(recent) < DEFAULT_HISTORY_CORRECTION_MIN_TOTAL_ROWS:
        raise ValueError(
            f"Insufficient rows for adaptive correction: {len(recent)}. "
            f"Minimum required: {DEFAULT_HISTORY_CORRECTION_MIN_TOTAL_ROWS}."
        )

    recent["date"] = recent[TIMESTAMP_COL].dt.date
    recent["hour"] = recent[TIMESTAMP_COL].dt.hour

    curve_matrix = (
        recent
        .pivot_table(index="date", columns="hour", values=TARGET_COL, aggfunc="mean")
        .reindex(columns=list(range(24)))
    )

    profile = pd.DataFrame({
        "hour": list(range(24)),
        "history_median_7d_Wh": curve_matrix.median(axis=0, skipna=True).values,
        "history_mean_7d_Wh": curve_matrix.mean(axis=0, skipna=True).values,
        "history_std_7d_Wh": curve_matrix.std(axis=0, skipna=True).values,
        "history_count_7d": curve_matrix.count(axis=0).values,
    })

    fallback_median = float(recent[TARGET_COL].median())
    profile["history_median_7d_Wh"] = pd.to_numeric(profile["history_median_7d_Wh"], errors="coerce").fillna(fallback_median).clip(lower=0)
    profile["history_mean_7d_Wh"] = pd.to_numeric(profile["history_mean_7d_Wh"], errors="coerce").fillna(fallback_median).clip(lower=0)
    profile["history_std_7d_Wh"] = pd.to_numeric(profile["history_std_7d_Wh"], errors="coerce").fillna(0.0).clip(lower=0)
    profile["history_count_7d"] = pd.to_numeric(profile["history_count_7d"], errors="coerce").fillna(0).astype(int)

    daily = recent.assign(date=recent[TIMESTAMP_COL].dt.date).groupby("date")[TARGET_COL].sum() / 1000.0

    info = {
        "history_days": int(history_days),
        "history_rows_used_for_correction": int(len(recent)),
        "history_window_start": str(start),
        "history_window_end_exclusive": str(target_start),
        "history_daily_median_kWh": float(daily.median()),
        "history_daily_mean_kWh": float(daily.mean()),
        "history_hourly_median_Wh": float(recent[TARGET_COL].median()),
        "history_hourly_p90_Wh": float(recent[TARGET_COL].quantile(0.90)),
    }

    return profile, curve_matrix, info


def apply_adaptive_history_correction(
    prediction_df: pd.DataFrame,
    history_df: pd.DataFrame,
    target_date: str,
    pred_col: str = PRED_COL,
    history_days: int = DEFAULT_HISTORY_DAYS_FOR_CORRECTION,
    max_alpha: float = DEFAULT_HISTORY_CORRECTION_MAX_ALPHA,
    min_hourly_obs: int = DEFAULT_HISTORY_CORRECTION_MIN_HOURLY_OBS,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Apply adaptive 7-day median shape correction to a 24h forecast."""
    out = prediction_df.copy()

    if pred_col not in out.columns:
        raise ValueError(f"Prediction dataframe must contain column '{pred_col}'.")
    if len(out) != 24:
        raise ValueError(f"Adaptive history correction expects 24 forecast rows, got {len(out)}.")

    base_pred = pd.to_numeric(out[pred_col], errors="coerce").fillna(0).clip(lower=0).to_numpy(dtype=float)
    pred_total = float(base_pred.sum())

    if pred_total <= 0:
        out["pred_model_raw_Wh"] = base_pred.astype("float32")
        out["history_correction_applied"] = False
        out["adaptive_alpha"] = 0.0
        return out, {
            "history_correction_applied": False,
            "history_correction_reason": "prediction_total_is_zero",
        }

    try:
        profile, _, profile_info = _build_recent_history_profile(
            history_df=history_df,
            target_date=target_date,
            history_days=history_days,
        )

        median_profile = profile["history_median_7d_Wh"].to_numpy(dtype=float)
        if float(median_profile.sum()) <= 0:
            raise ValueError("history_median_profile_sum_is_zero")

        history_shape = median_profile / float(median_profile.sum())
        history_shape_scaled = history_shape * pred_total

        cv = profile["history_std_7d_Wh"].to_numpy(dtype=float) / (np.abs(median_profile) + 1.0)
        stability = 1.0 - np.clip(cv / 1.0, 0.0, 1.0)

        alpha = float(max_alpha) * stability
        alpha = np.where(profile["history_count_7d"].to_numpy(dtype=int) >= int(min_hourly_obs), alpha, 0.0)
        alpha = np.where(median_profile > 0, alpha, 0.0)
        alpha = np.clip(alpha, 0.0, float(max_alpha))

        corrected = (1.0 - alpha) * base_pred + alpha * history_shape_scaled
        corrected = np.maximum(corrected, 0.0)

        out["pred_model_raw_Wh"] = base_pred.astype("float32")
        out["history_median_7d_Wh"] = median_profile.astype("float32")
        out["history_shape_scaled_Wh"] = history_shape_scaled.astype("float32")
        out["adaptive_alpha"] = alpha.astype("float32")
        out[pred_col] = corrected.astype("float32")
        out["history_correction_applied"] = True

        info = {
            "history_correction_applied": True,
            "history_correction_method": "adaptive_7d_median_shape",
            "history_correction_reason": "applied",
            "history_days": int(history_days),
            "max_alpha": float(max_alpha),
            "mean_alpha": float(np.mean(alpha)),
            "max_used_alpha": float(np.max(alpha)),
            "pred_total_kWh_before": float(base_pred.sum() / 1000.0),
            "pred_total_kWh_after": float(corrected.sum() / 1000.0),
            **profile_info,
        }
        return out, info

    except Exception as exc:
        out["pred_model_raw_Wh"] = base_pred.astype("float32")
        out["history_correction_applied"] = False
        out["adaptive_alpha"] = 0.0
        return out, {
            "history_correction_applied": False,
            "history_correction_reason": str(exc),
        }


# ============================================================
# Feature builders
# ============================================================

def build_no_history_feature_frame(
    target_date: str,
    internal_temperature_24h=None,
    external_temperature_24h=None,
    internal_humidity_24h=None,
    external_humidity_24h=None,
    **static_kwargs,
) -> pd.DataFrame:
    ts24 = _build_24h_timestamps(target_date)
    internal_temp = _normalize_24h_numeric(internal_temperature_24h, "internal_temperature_24h")
    external_temp = _normalize_24h_numeric(external_temperature_24h, "external_temperature_24h")
    internal_hum = _normalize_24h_numeric(internal_humidity_24h, "internal_humidity_24h")
    external_hum = _normalize_24h_numeric(external_humidity_24h, "external_humidity_24h")

    static = _base_static_features(**static_kwargs)

    rows = []
    for i, ts in enumerate(ts24):
        rows.append({
            TIMESTAMP_COL: ts,
            "internal_temperature": internal_temp[i],
            "external_temperature": external_temp[i],
            "internal_humidity": internal_hum[i],
            "external_humidity": external_hum[i],
            **static,
            **_time_features(ts),
        })

    return pd.DataFrame(rows)


# Backward-compatible alias.
build_coldstart_feature_frame = build_no_history_feature_frame


# ============================================================
# Public prediction functions
# ============================================================

def predict_coldstart_dayahead(
    artifacts: Optional[Dict[str, Any]] = None,
    target_date: str = None,
    internal_temperature_24h=None,
    external_temperature_24h=None,
    internal_humidity_24h=None,
    external_humidity_24h=None,
    num_rooms: float = None,
    residents: float = None,
    num_adults: float = None,
    num_children: float = None,
    num_elderly: float = None,
    has_ac: float = None,
    has_fridge_freezer: float = None,
    has_dryer: float = None,
    has_washing_machine: float = None,
    has_dishwasher: float = None,
    has_microwave: float = None,
    has_electric_oven: float = None,
    has_electric_hob: float = None,
    solar_panels: float = None,
    building_type: str = UNKNOWN_LABEL,
    build_era: str = UNKNOWN_LABEL,
    income_band: str = UNKNOWN_LABEL,
    heating_type: str = UNKNOWN_LABEL,
    water_heater_type: str = UNKNOWN_LABEL,
    homeowner_status: str = UNKNOWN_LABEL,
    years_in_house: str = UNKNOWN_LABEL,
    occupation: str = UNKNOWN_LABEL,
    model_id: Optional[str] = "auto",
    optimization: Optional[str] = "balanced",
    save_csv: bool = False,
    out_dir: Path = DEFAULT_OUT_DIR,
    out_name: str = "pred_no_history_dayahead.csv",
) -> pd.DataFrame:
    """Backward-compatible wrapper for no-history day-ahead prediction."""
    if target_date is None:
        raise ValueError("target_date is required.")

    if artifacts is None:
        artifacts = load_model_artifacts(model_id=model_id, mode=MODE_NO_HISTORY, optimization=optimization)

    if normalize_mode(artifacts.get("mode")) != MODE_NO_HISTORY:
        raise ValueError(f"Loaded artifact mode is {artifacts.get('mode')}, expected {MODE_NO_HISTORY}.")

    df_feat = build_no_history_feature_frame(
        target_date=target_date,
        internal_temperature_24h=internal_temperature_24h,
        external_temperature_24h=external_temperature_24h,
        internal_humidity_24h=internal_humidity_24h,
        external_humidity_24h=external_humidity_24h,
        num_rooms=num_rooms,
        residents=residents,
        num_adults=num_adults,
        num_children=num_children,
        num_elderly=num_elderly,
        has_ac=has_ac,
        has_fridge_freezer=has_fridge_freezer,
        has_dryer=has_dryer,
        has_washing_machine=has_washing_machine,
        has_dishwasher=has_dishwasher,
        has_microwave=has_microwave,
        has_electric_oven=has_electric_oven,
        has_electric_hob=has_electric_hob,
        solar_panels=solar_panels,
        building_type=building_type,
        build_era=build_era,
        income_band=income_band,
        heating_type=heating_type,
        water_heater_type=water_heater_type,
        homeowner_status=homeowner_status,
        years_in_house=years_in_house,
        occupation=occupation,
    )

    X = _transform_features(artifacts, df_feat)
    y_hat_tr = _predict_raw_model(artifacts["model"], X, artifacts.get("model_type"))
    y_hat = _inverse_target_pred(y_hat_tr, artifacts["feature_config"])

    out = pd.DataFrame({
        TIMESTAMP_COL: df_feat[TIMESTAMP_COL].values,
        PRED_COL: y_hat,
    })
    out["model_id"] = artifacts.get("model_id")
    out["mode"] = MODE_NO_HISTORY
    out["prediction_variant"] = "balanced"

    if save_csv:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_dir / out_name, index=False)

    return out


# Friendly canonical alias for new UI/API code.
predict_no_history_dayahead = predict_coldstart_dayahead


def predict_withhistory_dayahead(
    artifacts: Optional[Dict[str, Any]] = None,
    target_date: str = None,
    internal_temperature_24h=None,
    external_temperature_24h=None,
    internal_humidity_24h=None,
    external_humidity_24h=None,
    history_consumption_Wh=None,
    history_csv_path: Optional[Any] = None,
    history_df: Optional[pd.DataFrame] = None,
    num_rooms: float = None,
    residents: float = None,
    num_adults: float = None,
    num_children: float = None,
    num_elderly: float = None,
    has_ac: float = None,
    has_fridge_freezer: float = None,
    has_dryer: float = None,
    has_washing_machine: float = None,
    has_dishwasher: float = None,
    has_microwave: float = None,
    has_electric_oven: float = None,
    has_electric_hob: float = None,
    solar_panels: float = None,
    building_type: str = UNKNOWN_LABEL,
    build_era: str = UNKNOWN_LABEL,
    income_band: str = UNKNOWN_LABEL,
    heating_type: str = UNKNOWN_LABEL,
    water_heater_type: str = UNKNOWN_LABEL,
    homeowner_status: str = UNKNOWN_LABEL,
    years_in_house: str = UNKNOWN_LABEL,
    occupation: str = UNKNOWN_LABEL,
    min_history_hours: Optional[int] = None,
    model_id: Optional[str] = "auto",
    optimization: Optional[str] = "balanced",
    use_daily_calibrated: bool = False,
    apply_history_correction: bool = True,
    history_correction_days: int = DEFAULT_HISTORY_DAYS_FOR_CORRECTION,
    history_correction_max_alpha: float = DEFAULT_HISTORY_CORRECTION_MAX_ALPHA,
    save_csv: bool = False,
    out_dir: Path = DEFAULT_OUT_DIR,
    out_name: str = "pred_with_history_dayahead.csv",
) -> pd.DataFrame:
    if target_date is None:
        raise ValueError("target_date is required.")

    if artifacts is None:
        artifacts = load_model_artifacts(model_id=model_id, mode=MODE_WITH_HISTORY, optimization=optimization)

    if normalize_mode(artifacts.get("mode")) != MODE_WITH_HISTORY:
        raise ValueError(f"Loaded artifact mode is {artifacts.get('mode')}, expected {MODE_WITH_HISTORY}.")

    if optimization and str(optimization).lower() in {"daily", "daily-optimized", "dailyoptimized", "total", "daily-total"}:
        use_daily_calibrated = True

    feature_config = artifacts["feature_config"]
    if min_history_hours is None:
        min_history_hours = int(feature_config.get("min_required_history_hours", 168))

    history_regularized_df = None
    history_input_info: Dict[str, Any] = {}

    if history_df is not None:
        hist, history_regularized_df, history_input_info = prepare_history_for_prediction_from_dataframe(
            history_df=history_df,
            target_date=target_date,
            min_history_hours=int(min_history_hours),
        )
    elif history_csv_path is not None:
        hist, history_regularized_df, history_input_info = prepare_history_for_prediction_from_csv(
            history_csv_path=history_csv_path,
            target_date=target_date,
            min_history_hours=int(min_history_hours),
        )
    else:
        if history_consumption_Wh is None:
            raise ValueError(
                "with_history prediction requires one of: "
                "history_df, history_csv_path, or history_consumption_Wh."
            )

        hist = [float(x) for x in list(history_consumption_Wh)]
        if len(hist) < int(min_history_hours):
            raise ValueError(
                f"history_consumption_Wh must have at least {min_history_hours} hours. "
                f"Received: {len(hist)}"
            )

        history_regularized_df = _history_df_from_vector(hist[-int(min_history_hours):], target_date=target_date)
        history_input_info = {
            "history_source": "vector",
            "target_date": str(pd.Timestamp(target_date).normalize().date()),
            "required_history_hours": int(min_history_hours),
            "observed_history_hours": int(len(hist)),
            "coverage": 1.0,
            "filled_history_hours": 0,
        }

    ts24 = _build_24h_timestamps(target_date)
    internal_temp = _normalize_24h_numeric(internal_temperature_24h, "internal_temperature_24h")
    external_temp = _normalize_24h_numeric(external_temperature_24h, "external_temperature_24h")
    internal_hum = _normalize_24h_numeric(internal_humidity_24h, "internal_humidity_24h")
    external_hum = _normalize_24h_numeric(external_humidity_24h, "external_humidity_24h")

    hist_deque = deque([float(x) for x in hist], maxlen=200000)

    static = _base_static_features(
        num_rooms=num_rooms,
        residents=residents,
        num_adults=num_adults,
        num_children=num_children,
        num_elderly=num_elderly,
        has_ac=has_ac,
        has_fridge_freezer=has_fridge_freezer,
        has_dryer=has_dryer,
        has_washing_machine=has_washing_machine,
        has_dishwasher=has_dishwasher,
        has_microwave=has_microwave,
        has_electric_oven=has_electric_oven,
        has_electric_hob=has_electric_hob,
        solar_panels=solar_panels,
        building_type=building_type,
        build_era=build_era,
        income_band=income_band,
        heating_type=heating_type,
        water_heater_type=water_heater_type,
        homeowner_status=homeowner_status,
        years_in_house=years_in_house,
        occupation=occupation,
    )

    preds = []
    rows_used = []

    for h, ts in enumerate(ts24):
        row = {
            TIMESTAMP_COL: ts,
            "internal_temperature": internal_temp[h],
            "external_temperature": external_temp[h],
            "internal_humidity": internal_hum[h],
            "external_humidity": external_hum[h],
            **static,
            **_time_features(ts),
        }
        row = _add_recursive_history_features(row, hist_deque, feature_config)

        df_row = pd.DataFrame([row])
        X = _transform_features(artifacts, df_row)

        y_hat_tr = _predict_raw_model(artifacts["model"], X, artifacts.get("model_type"))
        y_hat = float(_inverse_target_pred(y_hat_tr, feature_config)[0])

        preds.append(y_hat)
        rows_used.append(row)
        hist_deque.append(y_hat)

    preds = np.asarray(preds, dtype=np.float32)
    preds_out = _apply_daily_calibrator_if_requested(preds, artifacts, use_daily_calibrated=use_daily_calibrated)

    out = pd.DataFrame({
        TIMESTAMP_COL: ts24,
        PRED_COL: preds_out,
    })

    out["model_id"] = artifacts.get("model_id")
    out["mode"] = MODE_WITH_HISTORY

    base_variant = "daily_calibrated" if use_daily_calibrated else "balanced"
    out["prediction_variant"] = base_variant

    correction_info: Dict[str, Any] = {
        "history_correction_applied": False,
        "history_correction_reason": "disabled",
    }

    if apply_history_correction:
        out, correction_info = apply_adaptive_history_correction(
            prediction_df=out,
            history_df=history_regularized_df,
            target_date=target_date,
            pred_col=PRED_COL,
            history_days=int(history_correction_days),
            max_alpha=float(history_correction_max_alpha),
        )

        if correction_info.get("history_correction_applied", False):
            out["prediction_variant"] = f"{base_variant}_adaptive_history_corrected"
        else:
            out["prediction_variant"] = f"{base_variant}_history_correction_not_applied"

    out["history_source"] = history_input_info.get("history_source", "unknown")
    out["history_rows_used"] = int(history_input_info.get("required_history_hours", len(hist)))
    out["history_coverage"] = float(history_input_info.get("coverage", 1.0))
    out["history_correction_reason"] = correction_info.get("history_correction_reason", "")

    out.attrs["history_input_info"] = history_input_info
    out.attrs["history_correction_info"] = correction_info
    out.attrs["feature_rows_used"] = rows_used

    if save_csv:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_dir / out_name, index=False)

    return out


def predict_dayahead(
    mode: str = MODE_NO_HISTORY,
    model_id: Optional[str] = "auto",
    optimization: Optional[str] = "balanced",
    target_date: str = None,
    internal_temperature_24h=None,
    external_temperature_24h=None,
    internal_humidity_24h=None,
    external_humidity_24h=None,
    history_consumption_Wh=None,
    history_csv_path: Optional[Any] = None,
    history_df: Optional[pd.DataFrame] = None,
    num_rooms: float = None,
    residents: float = None,
    num_adults: float = None,
    num_children: float = None,
    num_elderly: float = None,
    has_ac: float = None,
    has_fridge_freezer: float = None,
    has_dryer: float = None,
    has_washing_machine: float = None,
    has_dishwasher: float = None,
    has_microwave: float = None,
    has_electric_oven: float = None,
    has_electric_hob: float = None,
    solar_panels: float = None,
    building_type: str = UNKNOWN_LABEL,
    build_era: str = UNKNOWN_LABEL,
    income_band: str = UNKNOWN_LABEL,
    heating_type: str = UNKNOWN_LABEL,
    water_heater_type: str = UNKNOWN_LABEL,
    homeowner_status: str = UNKNOWN_LABEL,
    years_in_house: str = UNKNOWN_LABEL,
    occupation: str = UNKNOWN_LABEL,
    min_history_hours: Optional[int] = None,
    apply_history_correction: bool = True,
    history_correction_days: int = DEFAULT_HISTORY_DAYS_FOR_CORRECTION,
    history_correction_max_alpha: float = DEFAULT_HISTORY_CORRECTION_MAX_ALPHA,
    save_csv: bool = False,
    out_dir: Path = DEFAULT_OUT_DIR,
    out_name: Optional[str] = None,
) -> pd.DataFrame:
    """Unified day-ahead prediction entry point used by the API/UI."""
    normalized_mode = normalize_mode(mode)

    if normalized_mode == MODE_NO_HISTORY:
        return predict_coldstart_dayahead(
            artifacts=None,
            target_date=target_date,
            internal_temperature_24h=internal_temperature_24h,
            external_temperature_24h=external_temperature_24h,
            internal_humidity_24h=internal_humidity_24h,
            external_humidity_24h=external_humidity_24h,
            num_rooms=num_rooms,
            residents=residents,
            num_adults=num_adults,
            num_children=num_children,
            num_elderly=num_elderly,
            has_ac=has_ac,
            has_fridge_freezer=has_fridge_freezer,
            has_dryer=has_dryer,
            has_washing_machine=has_washing_machine,
            has_dishwasher=has_dishwasher,
            has_microwave=has_microwave,
            has_electric_oven=has_electric_oven,
            has_electric_hob=has_electric_hob,
            solar_panels=solar_panels,
            building_type=building_type,
            build_era=build_era,
            income_band=income_band,
            heating_type=heating_type,
            water_heater_type=water_heater_type,
            homeowner_status=homeowner_status,
            years_in_house=years_in_house,
            occupation=occupation,
            model_id=model_id,
            optimization=optimization,
            save_csv=save_csv,
            out_dir=out_dir,
            out_name=out_name or "pred_no_history_dayahead.csv",
        )

    if normalized_mode == MODE_WITH_HISTORY:
        return predict_withhistory_dayahead(
            artifacts=None,
            target_date=target_date,
            internal_temperature_24h=internal_temperature_24h,
            external_temperature_24h=external_temperature_24h,
            internal_humidity_24h=internal_humidity_24h,
            external_humidity_24h=external_humidity_24h,
            history_consumption_Wh=history_consumption_Wh,
            history_csv_path=history_csv_path,
            history_df=history_df,
            num_rooms=num_rooms,
            residents=residents,
            num_adults=num_adults,
            num_children=num_children,
            num_elderly=num_elderly,
            has_ac=has_ac,
            has_fridge_freezer=has_fridge_freezer,
            has_dryer=has_dryer,
            has_washing_machine=has_washing_machine,
            has_dishwasher=has_dishwasher,
            has_microwave=has_microwave,
            has_electric_oven=has_electric_oven,
            has_electric_hob=has_electric_hob,
            solar_panels=solar_panels,
            building_type=building_type,
            build_era=build_era,
            income_band=income_band,
            heating_type=heating_type,
            water_heater_type=water_heater_type,
            homeowner_status=homeowner_status,
            years_in_house=years_in_house,
            occupation=occupation,
            min_history_hours=min_history_hours,
            model_id=model_id,
            optimization=optimization,
            use_daily_calibrated=str(optimization).lower().startswith("daily") if optimization else False,
            apply_history_correction=apply_history_correction,
            history_correction_days=history_correction_days,
            history_correction_max_alpha=history_correction_max_alpha,
            save_csv=save_csv,
            out_dir=out_dir,
            out_name=out_name or "pred_with_history_dayahead.csv",
        )

    raise ValueError(f"Unsupported mode: {mode}. Use '{MODE_NO_HISTORY}' or '{MODE_WITH_HISTORY}'.")


