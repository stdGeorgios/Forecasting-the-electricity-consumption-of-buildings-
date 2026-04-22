import json
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
import joblib

DEFAULT_BASE_DIR = Path("C:/IDEAL_Programming")
DEFAULT_ART_DIR = DEFAULT_BASE_DIR / "processed" / "models" / "final_rf"
DEFAULT_OUT_DIR = DEFAULT_BASE_DIR / "processed" / "predictions" / "api_like"
DEFAULT_HOLIDAYS_CSV = DEFAULT_BASE_DIR / "metadata" / "holidays.csv"


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

    Returns a dict with fixed keys used by the API/inference pipeline:
      - rf_cold
      - rf_hist
      - cold_cols
      - hist_cols
      - cold_extras
      - hist_extras

    Even for XGBoost / LightGBM we keep the same internal keys so the
    downstream prediction functions stay unchanged.
    """
    art_dir = Path(art_dir)

    if model_type == "sklearn_rf":
        cold_model_name = "rf_coldstart.joblib"
        hist_model_name = "rf_withhistory.joblib"
        cold_cols_name = "rf_coldstart_feature_columns.json"
        hist_cols_name = "rf_withhistory_feature_columns.json"
        cold_extras_name = "rf_coldstart_train_extras.json"
        hist_extras_name = "rf_withhistory_train_extras.json"

    elif model_type == "xgboost":
        cold_model_name = "xgb_coldstart.joblib"
        hist_model_name = "xgb_withhistory.joblib"
        cold_cols_name = "xgb_coldstart_feature_columns.json"
        hist_cols_name = "xgb_withhistory_feature_columns.json"
        cold_extras_name = "xgb_coldstart_train_extras.json"
        hist_extras_name = "xgb_withhistory_train_extras.json"

    elif model_type == "lightgbm":
        cold_model_name = "lgbm_coldstart.joblib"
        hist_model_name = "lgbm_withhistory.joblib"
        cold_cols_name = "lgbm_coldstart_feature_columns.json"
        hist_cols_name = "lgbm_withhistory_feature_columns.json"
        cold_extras_name = "lgbm_coldstart_train_extras.json"
        hist_extras_name = "lgbm_withhistory_train_extras.json"

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    cold_model_path = art_dir / cold_model_name
    hist_model_path = art_dir / hist_model_name
    cold_cols_path = art_dir / cold_cols_name
    hist_cols_path = art_dir / hist_cols_name
    cold_extras_path = art_dir / cold_extras_name
    hist_extras_path = art_dir / hist_extras_name

    if not cold_model_path.exists():
        raise FileNotFoundError(f"Missing model file: {cold_model_path}")
    if not hist_model_path.exists():
        raise FileNotFoundError(f"Missing model file: {hist_model_path}")
    if not cold_cols_path.exists():
        raise FileNotFoundError(f"Missing feature columns file: {cold_cols_path}")
    if not hist_cols_path.exists():
        raise FileNotFoundError(f"Missing feature columns file: {hist_cols_path}")

    model_cold = joblib.load(cold_model_path)
    model_hist = joblib.load(hist_model_path)

    with open(cold_cols_path, "r", encoding="utf-8") as f:
        cold_cols = json.load(f)

    with open(hist_cols_path, "r", encoding="utf-8") as f:
        hist_cols = json.load(f)

    cold_extras = _load_json_if_exists(cold_extras_path, {})
    hist_extras = _load_json_if_exists(hist_extras_path, {})

    return {
        "rf_cold": model_cold,
        "rf_hist": model_hist,
        "cold_cols": cold_cols,
        "hist_cols": hist_cols,
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
def _time_features(ts: pd.Timestamp) -> dict:
    dow = ts.dayofweek
    return {
        "hour": int(ts.hour),
        "day_of_week": int(dow),
        "month": int(ts.month),
        "is_weekend": int(dow >= 5),
        "is_holiday": int(ts.date() in _HOLIDAYS_SET),
    }


def _build_24h_timestamps(target_date: str) -> pd.DatetimeIndex:
    start = pd.Timestamp(target_date)
    return pd.date_range(start=start, periods=24, freq="h")


def _ensure_len(arr, n, name):
    if len(arr) != n:
        raise ValueError(f"{name} πρέπει να έχει μήκος {n}, αλλά έχει {len(arr)}")


def _onehot_align(df_feat: pd.DataFrame, expected_cols: list) -> pd.DataFrame:
    X = pd.get_dummies(df_feat, columns=["hometype", "urban_rural_class"], drop_first=False)
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
    external_temperature_24h,
    total_floor_area_m2: float,
    residents: float,
    hometype: str,
    urban_rural_class: str,
    save_csv: bool = False,
    out_dir: Path = DEFAULT_OUT_DIR,
    out_name: str = "pred_coldstart_dayahead.csv",
) -> pd.DataFrame:
    model_cold = artifacts["rf_cold"]
    cold_cols = artifacts["cold_cols"]
    cold_extras = artifacts.get("cold_extras", {})

    ts24 = _build_24h_timestamps(target_date)

    external_temperature_24h = list(external_temperature_24h)
    _ensure_len(external_temperature_24h, 24, "external_temperature_24h")

    rows = []
    for i, ts in enumerate(ts24):
        tf = _time_features(ts)
        rows.append(
            {
                "external_temperature": float(external_temperature_24h[i]),
                "total_floor_area_m2": float(total_floor_area_m2),
                "residents": float(residents),
                "hometype": str(hometype),
                "urban_rural_class": str(urban_rural_class),
                **tf,
            }
        )

    df_feat = pd.DataFrame(rows)
    X = _onehot_align(df_feat, cold_cols)

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
    external_temperature_24h,
    history_consumption_Wh,
    total_floor_area_m2: float,
    residents: float,
    hometype: str,
    urban_rural_class: str,
    min_history_hours: int = 48,
    save_csv: bool = False,
    out_dir: Path = DEFAULT_OUT_DIR,
    out_name: str = "pred_withhistory_dayahead.csv",
) -> pd.DataFrame:
    model_hist = artifacts["rf_hist"]
    hist_cols = artifacts["hist_cols"]
    hist_extras = artifacts.get("hist_extras", {})

    ts24 = _build_24h_timestamps(target_date)

    external_temperature_24h = list(external_temperature_24h)
    _ensure_len(external_temperature_24h, 24, "external_temperature_24h")

    hist = [float(x) for x in list(history_consumption_Wh)]
    if len(hist) < min_history_hours:
        raise ValueError(f"history_consumption_Wh πρέπει να έχει >= {min_history_hours} ώρες. Έχει: {len(hist)}")

    hist_deque = deque(hist, maxlen=200000)
    preds = []

    for h, ts in enumerate(ts24):
        tf = _time_features(ts)

        lag_1h = hist_deque[-1]
        lag_24h = hist_deque[-24] if len(hist_deque) >= 24 else lag_1h

        if len(hist_deque) >= 24:
            roll_mean_24h = float(np.mean(list(hist_deque)[-24:]))
        else:
            roll_mean_24h = float(np.mean(list(hist_deque)))

        if len(hist_deque) >= 168:
            roll_mean_168h = float(np.mean(list(hist_deque)[-168:]))
        else:
            roll_mean_168h = float(np.mean(list(hist_deque)))

        row = {
            "external_temperature": float(external_temperature_24h[h]),
            "total_floor_area_m2": float(total_floor_area_m2),
            "residents": float(residents),
            "hometype": str(hometype),
            "urban_rural_class": str(urban_rural_class),
            **tf,
            "lag_1h": float(lag_1h),
            "lag_24h": float(lag_24h),
            "roll_mean_24h": float(roll_mean_24h),
            "roll_mean_168h": float(roll_mean_168h),
        }

        X = _onehot_align(pd.DataFrame([row]), hist_cols)

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