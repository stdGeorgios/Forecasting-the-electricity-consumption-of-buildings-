# C:/Plegma_Programming/src/inference.py

from __future__ import annotations

import json
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
import joblib

DEFAULT_BASE_DIR = Path("C:/Plegma_Programming")
DEFAULT_ART_DIR = DEFAULT_BASE_DIR / "models" / "final_rf"
DEFAULT_OUT_DIR = DEFAULT_BASE_DIR / "predictions" / "api_like"
DEFAULT_HOLIDAYS_CSV = DEFAULT_BASE_DIR / "holidays.csv"


# =========================
# Holidays
# =========================
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


# =========================
# JSON helpers
# =========================
def _load_json_if_exists(path: Path, default):
    path = Path(path)
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# Generic model artifacts loader
# =========================
def load_model_artifacts(model_type: str, art_dir: Path):
    """
    Generic loader for different model families.

    Returns fixed keys used by the prediction pipeline:
      - model_cold
      - model_hist
      - cold_cols
      - hist_cols
      - cold_medians
      - hist_medians
      - cold_extras
      - hist_extras
    """
    art_dir = Path(art_dir)

    if model_type == "sklearn_rf":
        prefix = "rf"
    elif model_type == "xgboost":
        prefix = "xgb"
    elif model_type == "lightgbm":
        prefix = "lgbm"
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    cold_model_path = art_dir / f"{prefix}_coldstart.joblib"
    hist_model_path = art_dir / f"{prefix}_withhistory.joblib"

    cold_cols_path = art_dir / f"{prefix}_coldstart_feature_columns.json"
    hist_cols_path = art_dir / f"{prefix}_withhistory_feature_columns.json"

    cold_medians_path = art_dir / f"{prefix}_coldstart_train_medians.json"
    hist_medians_path = art_dir / f"{prefix}_withhistory_train_medians.json"

    cold_extras_path = art_dir / f"{prefix}_coldstart_train_extras.json"
    hist_extras_path = art_dir / f"{prefix}_withhistory_train_extras.json"

    for p in [cold_model_path, hist_model_path, cold_cols_path, hist_cols_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing artifact: {p}")

    model_cold = joblib.load(cold_model_path)
    model_hist = joblib.load(hist_model_path)

    with open(cold_cols_path, "r", encoding="utf-8") as f:
        cold_cols = json.load(f)

    with open(hist_cols_path, "r", encoding="utf-8") as f:
        hist_cols = json.load(f)

    cold_medians = _load_json_if_exists(cold_medians_path, {})
    hist_medians = _load_json_if_exists(hist_medians_path, {})

    cold_extras = _load_json_if_exists(cold_extras_path, {})
    hist_extras = _load_json_if_exists(hist_extras_path, {})

    return {
        "model_cold": model_cold,
        "model_hist": model_hist,
        "cold_cols": cold_cols,
        "hist_cols": hist_cols,
        "cold_medians": cold_medians,
        "hist_medians": hist_medians,
        "cold_extras": cold_extras,
        "hist_extras": hist_extras,
        "model_type": model_type,
        "artifact_dir": str(art_dir),
    }


# Optional backward compatibility
def load_rf_artifacts(art_dir: Path = DEFAULT_ART_DIR):
    return load_model_artifacts(model_type="sklearn_rf", art_dir=art_dir)


# =========================
# Core feature helpers
# =========================
def _season_from_month(month: int) -> str:
    if month in [12, 1, 2]:
        return "winter"
    if month in [3, 4, 5]:
        return "spring"
    if month in [6, 7, 8]:
        return "summer"
    return "autumn"


def _time_features(ts: pd.Timestamp) -> dict:
    dow = ts.dayofweek
    return {
        "hour": int(ts.hour),
        "day_of_week": int(dow),
        "month": int(ts.month),
        "is_weekend": int(dow >= 5),
        "is_holiday": int(ts.date() in _HOLIDAYS_SET),
        "season": _season_from_month(ts.month),
    }


def _build_24h_timestamps(target_date: str) -> pd.DatetimeIndex:
    start = pd.Timestamp(target_date)
    return pd.date_range(start=start, periods=24, freq="h")


def _ensure_len(arr, n, name):
    if len(arr) != n:
        raise ValueError(f"{name} must have length {n}, but has {len(arr)}")


def _fill_numeric_medians(df_feat: pd.DataFrame, medians: dict) -> pd.DataFrame:
    df_feat = df_feat.copy()
    for col, med in medians.items():
        if col in df_feat.columns:
            df_feat[col] = pd.to_numeric(df_feat[col], errors="coerce").fillna(float(med))
    return df_feat


def _prepare_categoricals(df_feat: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
    df_feat = df_feat.copy()
    for c in categorical_cols:
        if c in df_feat.columns:
            df_feat[c] = df_feat[c].astype("string").fillna("Unknown")
    return df_feat


def _onehot_align(
    df_feat: pd.DataFrame,
    expected_cols: list,
    categorical_cols: list,
    medians: dict,
) -> pd.DataFrame:
    df_feat = _fill_numeric_medians(df_feat, medians)
    df_feat = _prepare_categoricals(df_feat, categorical_cols)

    cat_cols_present = [c for c in categorical_cols if c in df_feat.columns]
    X = pd.get_dummies(df_feat, columns=cat_cols_present, drop_first=False)
    X = X.reindex(columns=expected_cols, fill_value=0)

    return X.astype("float32")


# =========================
# Target inverse transform
# =========================
def _inverse_target_pred(y_pred, extras: dict) -> np.ndarray:
    y = np.asarray(y_pred, dtype=np.float64)

    if extras.get("use_log_target", False):
        y = np.expm1(y)

    y = np.maximum(y, 0.0)
    return y.astype("float32")


# =========================
# Predict: cold-start
# =========================
def predict_coldstart_dayahead(
    artifacts: dict,
    target_date: str,
    internal_temperature_24h,
    external_temperature_24h,
    internal_humidity_24h,
    external_humidity_24h,
    num_rooms: float,
    residents: float,
    num_adults: float,
    num_children: float,
    num_elderly: float,
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
    save_csv: bool = False,
    out_dir: Path = DEFAULT_OUT_DIR,
    out_name: str = "pred_coldstart_dayahead.csv",
) -> pd.DataFrame:
    model_cold = artifacts["model_cold"]
    cold_cols = artifacts["cold_cols"]
    cold_medians = artifacts.get("cold_medians", {})
    cold_extras = artifacts.get("cold_extras", {})
    categorical_cols = cold_extras.get("categorical", [])

    ts24 = _build_24h_timestamps(target_date)

    internal_temperature_24h = list(internal_temperature_24h)
    external_temperature_24h = list(external_temperature_24h)
    internal_humidity_24h = list(internal_humidity_24h)
    external_humidity_24h = list(external_humidity_24h)

    _ensure_len(internal_temperature_24h, 24, "internal_temperature_24h")
    _ensure_len(external_temperature_24h, 24, "external_temperature_24h")
    _ensure_len(internal_humidity_24h, 24, "internal_humidity_24h")
    _ensure_len(external_humidity_24h, 24, "external_humidity_24h")

    rows = []
    for i, ts in enumerate(ts24):
        tf = _time_features(ts)
        rows.append(
            {
                "internal_temperature": float(internal_temperature_24h[i]),
                "external_temperature": float(external_temperature_24h[i]),
                "internal_humidity": float(internal_humidity_24h[i]),
                "external_humidity": float(external_humidity_24h[i]),
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
                **tf,
            }
        )

    df_feat = pd.DataFrame(rows)
    X = _onehot_align(
        df_feat=df_feat,
        expected_cols=cold_cols,
        categorical_cols=categorical_cols,
        medians=cold_medians,
    )

    y_hat_tr = model_cold.predict(X).astype("float32")
    y_hat = _inverse_target_pred(y_hat_tr, cold_extras)

    out = pd.DataFrame(
        {
            "timestamp": ts24,
            "pred_consumption_Wh": y_hat,
        }
    )

    if save_csv:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_dir / out_name, index=False)

    return out


# =========================
# Predict: with-history
# =========================
def predict_withhistory_dayahead(
    artifacts: dict,
    target_date: str,
    internal_temperature_24h,
    external_temperature_24h,
    internal_humidity_24h,
    external_humidity_24h,
    history_consumption_Wh,
    num_rooms: float,
    residents: float,
    num_adults: float,
    num_children: float,
    num_elderly: float,
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
    min_history_hours: int = 168,
    save_csv: bool = False,
    out_dir: Path = DEFAULT_OUT_DIR,
    out_name: str = "pred_withhistory_dayahead.csv",
) -> pd.DataFrame:
    model_hist = artifacts["model_hist"]
    hist_cols = artifacts["hist_cols"]
    hist_medians = artifacts.get("hist_medians", {})
    hist_extras = artifacts.get("hist_extras", {})
    categorical_cols = hist_extras.get("categorical", [])

    ts24 = _build_24h_timestamps(target_date)

    internal_temperature_24h = list(internal_temperature_24h)
    external_temperature_24h = list(external_temperature_24h)
    internal_humidity_24h = list(internal_humidity_24h)
    external_humidity_24h = list(external_humidity_24h)

    _ensure_len(internal_temperature_24h, 24, "internal_temperature_24h")
    _ensure_len(external_temperature_24h, 24, "external_temperature_24h")
    _ensure_len(internal_humidity_24h, 24, "internal_humidity_24h")
    _ensure_len(external_humidity_24h, 24, "external_humidity_24h")

    hist = [float(x) for x in list(history_consumption_Wh)]
    if len(hist) < min_history_hours:
        raise ValueError(
            f"history_consumption_Wh must have >= {min_history_hours} hours. Has: {len(hist)}"
        )

    hist_deque = deque(hist, maxlen=200000)
    preds = []

    for h, ts in enumerate(ts24):
        tf = _time_features(ts)

        lag_1h = hist_deque[-1]
        lag_24h = hist_deque[-24] if len(hist_deque) >= 24 else lag_1h
        lag_168h = hist_deque[-168] if len(hist_deque) >= 168 else lag_24h

        if len(hist_deque) >= 24:
            roll_mean_24h = float(np.mean(list(hist_deque)[-24:]))
        else:
            roll_mean_24h = float(np.mean(list(hist_deque)))

        if len(hist_deque) >= 168:
            roll_mean_168h = float(np.mean(list(hist_deque)[-168:]))
        else:
            roll_mean_168h = float(np.mean(list(hist_deque)))

        row = {
            "internal_temperature": float(internal_temperature_24h[h]),
            "external_temperature": float(external_temperature_24h[h]),
            "internal_humidity": float(internal_humidity_24h[h]),
            "external_humidity": float(external_humidity_24h[h]),
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
            **tf,
            "lag_1h": float(lag_1h),
            "lag_24h": float(lag_24h),
            "lag_168h": float(lag_168h),
            "roll_mean_24h": float(roll_mean_24h),
            "roll_mean_168h": float(roll_mean_168h),
        }

        X = _onehot_align(
            df_feat=pd.DataFrame([row]),
            expected_cols=hist_cols,
            categorical_cols=categorical_cols,
            medians=hist_medians,
        )

        y_hat_tr = float(model_hist.predict(X)[0])
        y_hat = float(
            _inverse_target_pred(
                np.array([y_hat_tr], dtype="float32"),
                hist_extras,
            )[0]
        )

        preds.append(y_hat)
        hist_deque.append(y_hat)

    out = pd.DataFrame(
        {
            "timestamp": ts24,
            "pred_consumption_Wh": np.array(preds, dtype="float32"),
        }
    )

    if save_csv:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_dir / out_name, index=False)

    return out