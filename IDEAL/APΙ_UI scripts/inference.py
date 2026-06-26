# C:/IDEAL_Programming/src/inference.py
# ============================================================
# Generic API/UI inference for final IDEAL models
# ============================================================
#
# Final UI assumptions:
#   - The user does NOT provide home_id.
#   - Model artifacts are selected through model_registry.py.
#   - Every final artifact folder contains:
#       model.joblib
#       preprocessor.pkl
#       feature_config.json
#       metadata.json              optional but recommended
#       global_daily_calibrator.json optional for generic with-history models
#       behavior_profiles.joblib     optional for no-history LGBM/XGB
#       knn_bundle.joblib            optional for no-history LGBM/XGB
#
# Supported modes:
#   - no_history
#   - with_history
#   - backward-compatible aliases: coldstart, cold_start, withhistory
#
# Final default routing, defined in model_registry.py:
#   - no_history   -> RF2/cold_start_default
#   - with_history -> LGBM2/with_history_generic
#
# Notes:
#   - home_id is not a user input and is not used as a model feature.
#   - For KNN/behavior optional no-history models, a pseudo home_id is used
#     internally only to reuse helper logic.
#   - With-history prediction is recursive for the 24-hour forecast horizon:
#     each predicted hour is appended to the history and used for later lags.
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
        # Backward-compatible fallback for older model_registry.py versions.
        try:
            from src.model_registry import (
                DEFAULT_MODE,
                MODE_COLDSTART,
                MODE_WITHHISTORY,
                validate_model_for_mode,
            )
        except ModuleNotFoundError:
            from model_registry import (
                DEFAULT_MODE,
                MODE_COLDSTART,
                MODE_WITHHISTORY,
                validate_model_for_mode,
            )

        MODE_NO_HISTORY = MODE_COLDSTART
        MODE_WITH_HISTORY = MODE_WITHHISTORY

        def normalize_mode(mode: Optional[str]) -> str:
            if not mode:
                return DEFAULT_MODE
            value = str(mode).strip().lower().replace("_", "").replace("-", "")
            if value in {"coldstart", "cold", "nohistory", "rawfrcst", "rawforecast"}:
                return MODE_NO_HISTORY
            if value in {"withhistory", "history", "withhist", "recenthistory"}:
                return MODE_WITH_HISTORY
            raise ValueError(f"Unsupported prediction mode: {mode}.")

# Compatibility constants for older imports/calls.
MODE_COLDSTART = MODE_NO_HISTORY
MODE_WITHHISTORY = MODE_WITH_HISTORY
# ============================================================
# Paths
# ============================================================

DEFAULT_BASE_DIR = Path("C:/IDEAL_Programming")
DEFAULT_OUT_DIR = DEFAULT_BASE_DIR / "processed" / "predictions" / "api_like"
DEFAULT_HOLIDAYS_CSV = DEFAULT_BASE_DIR / "metadata" / "holidays.csv"

TIMESTAMP_COL = "timestamp"
TARGET_COL = "consumption_Wh"
PRED_COL = "pred_consumption_Wh"
PSEUDO_HOME_ID = "ui_user"
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
    """Load one concrete model artifact bundle.

    Parameters
    ----------
    model_id:
        Concrete registry id or 'auto'. Legacy values such as 'rf', 'lgbm',
        and 'xgb' are resolved by model_registry.py.

    mode:
        'coldstart' or 'withhistory'.

    optimization:
        'balanced' or 'daily'. Currently this mainly controls registry default
        routing and optional calibrated output selection.
    """
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

    global_daily_calibrator = _load_json_if_exists(
        artifact_dir / "global_daily_calibrator.json",
        None,
    )

    behavior_profiles = _safe_joblib_load(artifact_dir / "behavior_profiles.joblib", default=None)
    knn_bundle = _safe_joblib_load(artifact_dir / "knn_bundle.joblib", default=None)

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
        "global_daily_calibrator": global_daily_calibrator,
        "behavior_profiles": behavior_profiles,
        "knn_bundle": knn_bundle,
    }


# Backward-compatible wrappers. These no longer load an old dual RF bundle;
# they load the registry default model for the requested mode.
def load_rf_artifacts(mode: str = MODE_COLDSTART) -> Dict[str, Any]:
    model_id = "rf" if mode == MODE_COLDSTART else "rf"
    return load_model_artifacts(model_id=model_id, mode=mode)


# ============================================================
# Time / vector helpers
# ============================================================

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


def _base_static_features(
    total_floor_area_m2: float,
    residents: float,
    hometype: str,
    urban_rural_class: str,
    num_electric_appliances: Optional[float] = None,
) -> Dict[str, Any]:
    return {
        "total_floor_area_m2": _as_float_or_nan(total_floor_area_m2),
        "residents": _as_float_or_nan(residents),
        "num_electric_appliances": _as_float_or_nan(num_electric_appliances),
        "hometype": str(hometype) if hometype is not None else UNKNOWN_LABEL,
        "urban_rural_class": str(urban_rural_class) if urban_rural_class is not None else UNKNOWN_LABEL,
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

    calibrator = artifacts.get("global_daily_calibrator")
    if not calibrator or calibrator.get("type") != "global_scale":
        return np.maximum(preds, 0.0).astype("float32")

    scale = float(calibrator.get("scale", 1.0))
    return np.maximum(preds * scale, 0.0).astype("float32")


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
# Cold-start behavior/KNN helpers for optional LGBM/XGB models
# ============================================================

def _add_static_bins(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    r = pd.to_numeric(df.get("residents"), errors="coerce")
    df["residents_bin"] = np.select(
        [r.isna(), r <= 0, r == 1, r == 2, r == 3, r == 4, r >= 5],
        [UNKNOWN_LABEL, UNKNOWN_LABEL, "1", "2", "3", "4", "5plus"],
        default=UNKNOWN_LABEL,
    )

    area = pd.to_numeric(df.get("total_floor_area_m2"), errors="coerce")
    df["area_bin"] = pd.cut(
        area,
        bins=[-np.inf, 50, 80, 110, 150, np.inf],
        labels=["area_0_50", "area_50_80", "area_80_110", "area_110_150", "area_150plus"],
    ).astype(str)
    df["area_bin"] = df["area_bin"].replace({"nan": UNKNOWN_LABEL, "None": UNKNOWN_LABEL})

    app = pd.to_numeric(df.get("num_electric_appliances"), errors="coerce")
    df["appliances_bin"] = pd.cut(
        app,
        bins=[-np.inf, 5, 10, 20, 30, np.inf],
        labels=["app_0_5", "app_5_10", "app_10_20", "app_20_30", "app_30plus"],
    ).astype(str)
    df["appliances_bin"] = df["appliances_bin"].replace({"nan": UNKNOWN_LABEL, "None": UNKNOWN_LABEL})

    for c in ["hometype", "urban_rural_class"]:
        df[c] = df[c].fillna(UNKNOWN_LABEL).astype(str).replace({"nan": UNKNOWN_LABEL, "None": UNKNOWN_LABEL})

    df["static_profile_key"] = (
        df["hometype"].astype(str) + "|" +
        df["urban_rural_class"].astype(str) + "|" +
        df["residents_bin"].astype(str) + "|" +
        df["area_bin"].astype(str) + "|" +
        df["appliances_bin"].astype(str)
    )

    return df


def _add_behavior_profile(df_part: pd.DataFrame, profiles: Optional[Dict[str, Any]]) -> pd.DataFrame:
    out = df_part.copy()

    if not profiles:
        out["expected_behavior_Wh"] = np.nan
        out["expected_behavior_log1p"] = np.nan
        out["behavior_profile_source"] = "missing"
        return out

    out = out.merge(
        profiles["static_dow_hour"],
        on=["static_profile_key", "day_of_week", "hour"],
        how="left",
    )
    out = out.merge(
        profiles["static_hour"],
        on=["static_profile_key", "hour"],
        how="left",
    )
    out = out.merge(
        profiles["static_overall"],
        on="static_profile_key",
        how="left",
    )
    out = out.merge(
        profiles["global_dow_hour"],
        on=["day_of_week", "hour"],
        how="left",
    )
    out = out.merge(
        profiles["global_hour"],
        on="hour",
        how="left",
    )

    fallback = [
        ("expected_static_dow_hour_Wh", "static_dow_hour"),
        ("expected_static_hour_Wh", "static_hour"),
        ("expected_static_overall_Wh", "static_overall"),
        ("expected_global_dow_hour_Wh", "global_dow_hour"),
        ("expected_global_hour_Wh", "global_hour"),
    ]

    out["expected_behavior_Wh"] = np.nan
    out["behavior_profile_source"] = "missing"

    for col, source in fallback:
        if col not in out.columns:
            continue
        mask = out["expected_behavior_Wh"].isna() & out[col].notna()
        out.loc[mask, "expected_behavior_Wh"] = out.loc[mask, col]
        out.loc[mask, "behavior_profile_source"] = source

    global_overall = float(profiles.get("global_overall", 0.0))
    out["expected_behavior_Wh"] = out["expected_behavior_Wh"].fillna(global_overall)
    out.loc[out["behavior_profile_source"] == "missing", "behavior_profile_source"] = "global_overall"
    out["expected_behavior_Wh"] = out["expected_behavior_Wh"].clip(lower=0)
    out["expected_behavior_log1p"] = np.log1p(out["expected_behavior_Wh"])

    return out


def _transform_home_static_for_knn(df_part: pd.DataFrame, knn_bundle: Dict[str, Any]):
    numeric_static = knn_bundle["numeric_static"]

    needed = [
        "home_id",
        "residents",
        "total_floor_area_m2",
        "num_electric_appliances",
        "hometype",
        "urban_rural_class",
    ]

    tmp = df_part[needed].drop_duplicates(subset=["home_id"]).copy()

    for c in numeric_static:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
        tmp[c] = tmp[c].fillna(knn_bundle["medians"].get(c, 0.0))

    for c in ["hometype", "urban_rural_class"]:
        tmp[c] = tmp[c].fillna(UNKNOWN_LABEL).astype(str).replace({"nan": UNKNOWN_LABEL, "None": UNKNOWN_LABEL})

    encoded = pd.get_dummies(
        tmp[numeric_static + ["hometype", "urban_rural_class"]],
        columns=["hometype", "urban_rural_class"],
        dummy_na=False,
    )

    encoded = encoded.reindex(columns=knn_bundle["static_columns"], fill_value=0.0)
    X = knn_bundle["scaler"].transform(encoded.values)

    return tmp[["home_id"]].copy(), X


def _add_knn_behavior_profile(df_part: pd.DataFrame, knn_bundle: Optional[Dict[str, Any]]) -> pd.DataFrame:
    out = df_part.copy()

    if not knn_bundle:
        out["knn_expected_dow_hour_Wh"] = np.nan
        out["knn_expected_hour_Wh"] = np.nan
        out["knn_expected_overall_Wh"] = np.nan
        out["knn_expected_Wh"] = out.get("expected_behavior_Wh", np.nan)
        out["knn_expected_log1p"] = np.log1p(out["knn_expected_Wh"].clip(lower=0))
        out["knn_minus_behavior_Wh"] = out["knn_expected_Wh"] - out.get("expected_behavior_Wh", 0.0)
        return out

    homes_df, X = _transform_home_static_for_knn(out, knn_bundle)
    _, indices = knn_bundle["knn"].kneighbors(X)

    train_homes = knn_bundle["home_static"]["home_id"].values
    home_to_neighbors = {}

    for i, home_id in enumerate(homes_df["home_id"].values):
        neighbors = train_homes[indices[i]].tolist()
        neighbors = [h for h in neighbors if h != home_id]
        if len(neighbors) == 0:
            neighbors = train_homes[indices[i]].tolist()
        home_to_neighbors[home_id] = neighbors

    rows = []
    for home_id, neighs in home_to_neighbors.items():
        for neigh in neighs:
            rows.append({"home_id": home_id, "neighbor_home_id": neigh})

    neighbor_map = pd.DataFrame(rows)

    dow_hour = knn_bundle["home_dow_hour_profile"].rename(columns={"home_id": "neighbor_home_id"})
    tmp_dow_hour = neighbor_map.merge(dow_hour, on="neighbor_home_id", how="left")
    knn_dow_hour = (
        tmp_dow_hour
        .groupby(["home_id", "day_of_week", "hour"], as_index=False)
        .agg(knn_expected_dow_hour_Wh=("knn_source_dow_hour_Wh", "mean"))
    )

    hour_prof = knn_bundle["home_hour_profile"].rename(columns={"home_id": "neighbor_home_id"})
    tmp_hour = neighbor_map.merge(hour_prof, on="neighbor_home_id", how="left")
    knn_hour = (
        tmp_hour
        .groupby(["home_id", "hour"], as_index=False)
        .agg(knn_expected_hour_Wh=("knn_source_hour_Wh", "mean"))
    )

    overall_prof = knn_bundle["home_overall_profile"].rename(columns={"home_id": "neighbor_home_id"})
    tmp_overall = neighbor_map.merge(overall_prof, on="neighbor_home_id", how="left")
    knn_overall = (
        tmp_overall
        .groupby("home_id", as_index=False)
        .agg(knn_expected_overall_Wh=("knn_source_overall_Wh", "mean"))
    )

    out = out.merge(knn_dow_hour, on=["home_id", "day_of_week", "hour"], how="left")
    out = out.merge(knn_hour, on=["home_id", "hour"], how="left")
    out = out.merge(knn_overall, on="home_id", how="left")

    out["knn_expected_Wh"] = out["knn_expected_dow_hour_Wh"]
    out["knn_expected_Wh"] = out["knn_expected_Wh"].fillna(out["knn_expected_hour_Wh"])
    out["knn_expected_Wh"] = out["knn_expected_Wh"].fillna(out["knn_expected_overall_Wh"])
    out["knn_expected_Wh"] = out["knn_expected_Wh"].fillna(out["expected_behavior_Wh"])

    out["knn_expected_Wh"] = out["knn_expected_Wh"].clip(lower=0)
    out["knn_expected_log1p"] = np.log1p(out["knn_expected_Wh"])
    out["knn_minus_behavior_Wh"] = out["knn_expected_Wh"] - out["expected_behavior_Wh"]

    return out


def _maybe_add_cold_optional_profiles(df_feat: pd.DataFrame, artifacts: Dict[str, Any]) -> pd.DataFrame:
    cfg = artifacts.get("feature_config", {})

    if not cfg.get("uses_behavior_profiles", False) and not cfg.get("uses_knn_profiles", False):
        return df_feat

    out = df_feat.copy()
    if "home_id" not in out.columns:
        out["home_id"] = PSEUDO_HOME_ID

    out = _add_static_bins(out)

    if cfg.get("uses_behavior_profiles", False):
        out = _add_behavior_profile(out, artifacts.get("behavior_profiles"))

    if cfg.get("uses_knn_profiles", False):
        out = _add_knn_behavior_profile(out, artifacts.get("knn_bundle"))

    return out


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
    lag_hours = feature_config.get("lag_hours", [1, 24, 168])
    rolling_windows = feature_config.get("rolling_windows", [24, 168])
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

    # Derived lag/rolling features used by the training scripts.
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
    """Load a single-home history CSV.

    Expected columns:
        timestamp
        consumption_Wh

    The file must refer to one household only. No home_id filtering is performed.
    """
    if history_csv_path is None:
        raise ValueError("history_csv_path is required.")

    # Gradio may provide either a string path, a tempfile-like object
    # with .name, or a small dict containing a name/path.
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
    """Validate and normalize a single-home history dataframe."""
    if history_df is None or len(history_df) == 0:
        raise ValueError("History dataframe is empty.")

    df = history_df.copy()

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

    # If duplicate timestamps exist, average them.
    df = (
        df
        .groupby(TIMESTAMP_COL, as_index=False)
        .agg(**{TARGET_COL: (TARGET_COL, "mean")})
        .sort_values(TIMESTAMP_COL)
        .reset_index(drop=True)
    )

    return df


def _history_df_from_vector(
    history_consumption_Wh,
    target_date: str,
) -> pd.DataFrame:
    """Create timestamped history dataframe from a plain vector.

    Assumption:
        the last value is the hour immediately before target_date 00:00.
    """
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
    """Load a single-home history CSV and return a regularized history vector.

    Returns:
        history_values:
            last min_history_hours hourly values before target_date.
        history_regularized_df:
            timestamped hourly history dataframe used for lags and correction.
        info:
            diagnostic information for API/UI display.
    """
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
    """Regularize single-home history to hourly values before target_date."""
    df = _prepare_history_dataframe(history_df)

    target_start = pd.Timestamp(target_date).normalize()
    history_start = target_start - pd.Timedelta(hours=int(min_history_hours))

    df = df[df[TIMESTAMP_COL] < target_start].copy()

    if len(df) == 0:
        raise ValueError("No history rows exist before target_date.")

    hourly = (
        df
        .assign(_hour=df[TIMESTAMP_COL].dt.floor("h"))
        .groupby("_hour", as_index=False)
        .agg(**{TARGET_COL: (TARGET_COL, "mean")})
        .rename(columns={"_hour": TIMESTAMP_COL})
        .sort_values(TIMESTAMP_COL)
    )

    required_index = pd.date_range(
        start=history_start,
        end=target_start - pd.Timedelta(hours=1),
        freq="h",
    )

    hourly = (
        hourly
        .set_index(TIMESTAMP_COL)
        .reindex(required_index)
    )

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

    history_regularized_df = (
        hourly
        .reset_index()
        .rename(columns={"index": TIMESTAMP_COL})
    )

    values = history_regularized_df[TARGET_COL].astype(float).tolist()

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
    }

    return values, history_regularized_df, info


def _build_recent_history_profile(
    history_df: pd.DataFrame,
    target_date: str,
    history_days: int = DEFAULT_HISTORY_DAYS_FOR_CORRECTION,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Build 7-day hourly median profile from timestamped single-home history."""
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

    profile["history_median_7d_Wh"] = (
        pd.to_numeric(profile["history_median_7d_Wh"], errors="coerce")
        .fillna(fallback_median)
        .clip(lower=0)
    )
    profile["history_mean_7d_Wh"] = (
        pd.to_numeric(profile["history_mean_7d_Wh"], errors="coerce")
        .fillna(fallback_median)
        .clip(lower=0)
    )
    profile["history_std_7d_Wh"] = (
        pd.to_numeric(profile["history_std_7d_Wh"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0)
    )
    profile["history_count_7d"] = (
        pd.to_numeric(profile["history_count_7d"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    info = {
        "history_days": int(history_days),
        "history_rows_used_for_correction": int(len(recent)),
        "history_window_start": str(start),
        "history_window_end_exclusive": str(target_start),
        "history_daily_median_kWh": float(
            recent.assign(date=recent[TIMESTAMP_COL].dt.date)
            .groupby("date")[TARGET_COL].sum()
            .median() / 1000.0
        ),
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
    """Apply adaptive 7-day median shape correction to a 24h forecast.

    The correction changes the hourly shape mildly using a recent-history
    median hourly profile. It does not apply daily-level scaling.
    """
    out = prediction_df.copy()

    if pred_col not in out.columns:
        raise ValueError(f"Prediction dataframe must contain column '{pred_col}'.")

    if len(out) != 24:
        raise ValueError(f"Adaptive history correction expects 24 forecast rows, got {len(out)}.")

    base_pred = pd.to_numeric(out[pred_col], errors="coerce").fillna(0).clip(lower=0).to_numpy(dtype=float)
    pred_total = float(base_pred.sum())

    if pred_total <= 0:
        info = {
            "history_correction_applied": False,
            "history_correction_reason": "prediction_total_is_zero",
        }
        out["pred_model_raw_Wh"] = base_pred
        out["history_correction_applied"] = False
        out["adaptive_alpha"] = 0.0
        return out, info

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

        info = {
            "history_correction_applied": False,
            "history_correction_reason": str(exc),
        }

        return out, info



# ============================================================
# Feature builders
# ============================================================

def build_coldstart_feature_frame(
    target_date: str,
    external_temperature_24h,
    total_floor_area_m2: float,
    residents: float,
    hometype: str,
    urban_rural_class: str,
    internal_temperature_24h=None,
    num_electric_appliances: Optional[float] = None,
) -> pd.DataFrame:
    ts24 = _build_24h_timestamps(target_date)
    ext = _normalize_24h_numeric(external_temperature_24h, "external_temperature_24h")
    internal = _normalize_24h_numeric(internal_temperature_24h, "internal_temperature_24h")

    static = _base_static_features(
        total_floor_area_m2=total_floor_area_m2,
        residents=residents,
        hometype=hometype,
        urban_rural_class=urban_rural_class,
        num_electric_appliances=num_electric_appliances,
    )

    rows = []
    for i, ts in enumerate(ts24):
        rows.append({
            TIMESTAMP_COL: ts,
            "external_temperature": ext[i],
            "internal_temperature": internal[i],
            **static,
            **_time_features(ts),
        })

    return pd.DataFrame(rows)


def build_withhistory_feature_rows_recursive(
    feature_config: Dict[str, Any],
    target_date: str,
    external_temperature_24h,
    history_consumption_Wh,
    total_floor_area_m2: float,
    residents: float,
    hometype: str,
    urban_rural_class: str,
    internal_temperature_24h=None,
    num_electric_appliances: Optional[float] = None,
    min_history_hours: Optional[int] = None,
) -> Tuple[pd.DatetimeIndex, deque, List[Dict[str, Any]]]:
    ts24 = _build_24h_timestamps(target_date)
    ext = _normalize_24h_numeric(external_temperature_24h, "external_temperature_24h")
    internal = _normalize_24h_numeric(internal_temperature_24h, "internal_temperature_24h")

    hist = [float(x) for x in list(history_consumption_Wh)]

    if min_history_hours is None:
        min_history_hours = int(feature_config.get("min_required_history_hours", 168))

    if len(hist) < int(min_history_hours):
        raise ValueError(
            f"history_consumption_Wh must have at least {min_history_hours} hours. "
            f"Received: {len(hist)}"
        )

    hist_deque = deque(hist, maxlen=200000)

    static = _base_static_features(
        total_floor_area_m2=total_floor_area_m2,
        residents=residents,
        hometype=hometype,
        urban_rural_class=urban_rural_class,
        num_electric_appliances=num_electric_appliances,
    )

    rows = []
    for i, ts in enumerate(ts24):
        row = {
            TIMESTAMP_COL: ts,
            "external_temperature": ext[i],
            "internal_temperature": internal[i],
            **static,
            **_time_features(ts),
        }
        row = _add_recursive_history_features(row, hist_deque, feature_config)
        rows.append(row)

        # Placeholder. Actual predicted value is appended during prediction loop,
        # not inside this builder. The caller will rebuild/transform each row one-by-one.

    return ts24, hist_deque, rows


# ============================================================
# Public prediction functions
# ============================================================

def predict_coldstart_dayahead(
    artifacts: Optional[Dict[str, Any]] = None,
    target_date: str = None,
    external_temperature_24h=None,
    total_floor_area_m2: float = None,
    residents: float = None,
    hometype: str = "unknown",
    urban_rural_class: str = "unknown",
    internal_temperature_24h=None,
    num_electric_appliances: Optional[float] = None,
    model_id: Optional[str] = "auto",
    optimization: Optional[str] = "balanced",
    save_csv: bool = False,
    out_dir: Path = DEFAULT_OUT_DIR,
    out_name: str = "pred_no_history_dayahead.csv",
) -> pd.DataFrame:
    """Backward-compatible wrapper for no-history day-ahead prediction."""
    if target_date is None:
        raise ValueError("target_date is required.")
    if external_temperature_24h is None:
        raise ValueError("external_temperature_24h is required.")

    if artifacts is None:
        artifacts = load_model_artifacts(model_id=model_id, mode=MODE_NO_HISTORY, optimization=optimization)

    if normalize_mode(artifacts.get("mode")) != MODE_NO_HISTORY:
        raise ValueError(f"Loaded artifact mode is {artifacts.get('mode')}, expected {MODE_NO_HISTORY}.")

    df_feat = build_coldstart_feature_frame(
        target_date=target_date,
        external_temperature_24h=external_temperature_24h,
        total_floor_area_m2=total_floor_area_m2,
        residents=residents,
        hometype=hometype,
        urban_rural_class=urban_rural_class,
        internal_temperature_24h=internal_temperature_24h,
        num_electric_appliances=num_electric_appliances,
    )

    df_feat = _maybe_add_cold_optional_profiles(df_feat, artifacts)

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


def predict_withhistory_dayahead(
    artifacts: Optional[Dict[str, Any]] = None,
    target_date: str = None,
    external_temperature_24h=None,
    history_consumption_Wh=None,
    history_csv_path: Optional[Any] = None,
    history_df: Optional[pd.DataFrame] = None,
    total_floor_area_m2: float = None,
    residents: float = None,
    hometype: str = "unknown",
    urban_rural_class: str = "unknown",
    internal_temperature_24h=None,
    num_electric_appliances: Optional[float] = None,
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
    if external_temperature_24h is None:
        raise ValueError("external_temperature_24h is required.")

    if artifacts is None:
        artifacts = load_model_artifacts(model_id=model_id, mode=MODE_WITH_HISTORY, optimization=optimization)

    if normalize_mode(artifacts.get("mode")) != MODE_WITH_HISTORY:
        raise ValueError(f"Loaded artifact mode is {artifacts.get('mode')}, expected {MODE_WITH_HISTORY}.")

    if optimization and str(optimization).lower() in {"daily", "daily-optimized", "dailyoptimized"}:
        use_daily_calibrated = True

    feature_config = artifacts["feature_config"]

    if min_history_hours is None:
        min_history_hours = int(feature_config.get("min_required_history_hours", 168))

    # History input priority:
    #   explicit dataframe > CSV path > plain consumption vector.
    # The CSV/dataframe format is single-home only: timestamp, consumption_Wh.
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
    ext = _normalize_24h_numeric(external_temperature_24h, "external_temperature_24h")
    internal = _normalize_24h_numeric(internal_temperature_24h, "internal_temperature_24h")

    if len(hist) < int(min_history_hours):
        raise ValueError(
            f"history must have at least {min_history_hours} hours. Received: {len(hist)}"
        )

    hist_deque = deque([float(x) for x in hist], maxlen=200000)

    static = _base_static_features(
        total_floor_area_m2=total_floor_area_m2,
        residents=residents,
        hometype=hometype,
        urban_rural_class=urban_rural_class,
        num_electric_appliances=num_electric_appliances,
    )

    preds = []
    rows_used = []

    for h, ts in enumerate(ts24):
        row = {
            TIMESTAMP_COL: ts,
            "external_temperature": ext[h],
            "internal_temperature": internal[h],
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

    # Store compact run-level diagnostics in DataFrame attrs for API/UI layers.
    out.attrs["history_input_info"] = history_input_info
    out.attrs["history_correction_info"] = correction_info

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
    external_temperature_24h=None,
    history_consumption_Wh=None,
    history_csv_path: Optional[Any] = None,
    history_df: Optional[pd.DataFrame] = None,
    total_floor_area_m2: float = None,
    residents: float = None,
    hometype: str = "unknown",
    urban_rural_class: str = "unknown",
    internal_temperature_24h=None,
    num_electric_appliances: Optional[float] = None,
    min_history_hours: Optional[int] = None,
    apply_history_correction: bool = True,
    history_correction_days: int = DEFAULT_HISTORY_DAYS_FOR_CORRECTION,
    history_correction_max_alpha: float = DEFAULT_HISTORY_CORRECTION_MAX_ALPHA,
    save_csv: bool = False,
    out_dir: Path = DEFAULT_OUT_DIR,
    out_name: Optional[str] = None,
) -> pd.DataFrame:
    """Unified day-ahead prediction entry point used by the API.

    Canonical modes:
        no_history
        with_history

    Backward-compatible aliases such as coldstart and withhistory are accepted
    through model_registry.normalize_mode().
    """
    normalized_mode = normalize_mode(mode)

    if normalized_mode == MODE_NO_HISTORY:
        return predict_coldstart_dayahead(
            artifacts=None,
            target_date=target_date,
            external_temperature_24h=external_temperature_24h,
            total_floor_area_m2=total_floor_area_m2,
            residents=residents,
            hometype=hometype,
            urban_rural_class=urban_rural_class,
            internal_temperature_24h=internal_temperature_24h,
            num_electric_appliances=num_electric_appliances,
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
            external_temperature_24h=external_temperature_24h,
            history_consumption_Wh=history_consumption_Wh,
            history_csv_path=history_csv_path,
            history_df=history_df,
            total_floor_area_m2=total_floor_area_m2,
            residents=residents,
            hometype=hometype,
            urban_rural_class=urban_rural_class,
            internal_temperature_24h=internal_temperature_24h,
            num_electric_appliances=num_electric_appliances,
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

    raise ValueError(
        f"Unsupported mode: {mode}. Use '{MODE_NO_HISTORY}' or '{MODE_WITH_HISTORY}'."
    )


# Friendly canonical alias for new UI/API code.
predict_no_history_dayahead = predict_coldstart_dayahead
