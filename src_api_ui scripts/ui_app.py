# C:/IDEAL_Programming/src/ui_app.py
# ---------------------------------------------------------
# Gradio UI for IDEAL Forecasting API
#
# Features:
# ✅ Supports model families from API registry (rf / xgb / lgbm)
# ✅ Separate API fields: model_id + mode
# ✅ History mode controlled by checkbox
# ✅ Single-model prediction with Predict button
# ✅ Multi-model comparison with Compare button
# ✅ Combined prediction (weighted ensemble) with separate button
# ✅ Behavioral adjustment (optional, post-processing on UI side)
# ✅ User-defined ensemble weights for combined prediction
# ✅ Custom weight fields shown only when checkbox is enabled
# ✅ Fix for hidden t_min / t_max returning 0 instead of None
# ✅ Reset t_min/t_max when user changes target_date or city
# ✅ If target_date == today:
#    - UI shows Remaining kWh (from next full hour)
#    - CSV includes remaining_kWh_today and cutoff_iso
#    - Total kWh/day remains the FULL 24h day-ahead total
# ✅ Single-model prediction chart
# ✅ Comparison chart for hourly predictions
# ✅ Combined prediction chart
# ✅ 24 hours (0-23) always shown on horizontal axis
# ---------------------------------------------------------

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import gradio as gr
import pandas as pd
import requests
import matplotlib.pyplot as plt


# =========================
# CONFIG
# =========================
API_BASE = "http://127.0.0.1:8000"
UI_HOST = "127.0.0.1"
UI_PORT = 7860

BASE_DIR = Path("C:/IDEAL_Programming")
UI_EXPORT_DIR = BASE_DIR / "processed" / "predictions" / "ui_exports"
UI_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

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
    "rf": 0.50,
    "xgb": 0.25,
    "lgbm": 0.25,
}


# =========================
# HELPERS
# =========================
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
            return "✅ API: OK"
        return f"⚠️ API: HTTP {r.status_code}"
    except Exception:
        return "❌ API: not reachable (start API first)"


def _fetch_models() -> List[Dict[str, Any]]:
    try:
        r = requests.get(f"{API_BASE}/models", timeout=5)
        r.raise_for_status()
        models = r.json().get("models", [])
        if not models:
            raise RuntimeError("empty /models")
        return models
    except Exception:
        return [
            {
                "model_id": "rf",
                "name": "Random Forest (final)",
                "type": "sklearn_rf",
                "supports": ["coldstart", "withhistory"],
            },
            {
                "model_id": "xgb",
                "name": "XGBoost (final)",
                "type": "xgboost",
                "supports": ["coldstart", "withhistory"],
            },
            {
                "model_id": "lgbm",
                "name": "LightGBM (final)",
                "type": "lightgbm",
                "supports": ["coldstart", "withhistory"],
            },
        ]


def _format_model_choice(m: Dict[str, Any]) -> str:
    mid = m.get("model_id", "")
    name = m.get("name", mid)
    return f"{mid} — {name}"


def _choice_to_model_id(choice: str) -> str:
    return str(choice).split("—", 1)[0].strip()


def _needs_weather_fallback(detail_text: str) -> bool:
    t = (detail_text or "").lower()
    return ("δώσε είτε t_min / t_max" in t) or ("external_temperature" in t and "t_min" in t)


def _normalize_temp_inputs(
    t_min: Optional[float],
    t_max: Optional[float],
) -> Tuple[Optional[float], Optional[float]]:
    """
    Some Gradio environments return hidden Number fields as 0 / 0.0 instead of None.
    Convert (0,0) to (None,None) so validation doesn't fail on first run.
    """
    try:
        if t_min is not None:
            t_min = float(t_min)
        if t_max is not None:
            t_max = float(t_max)
    except Exception:
        return None, None

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
        d = _parse_date(target_date)
    except Exception:
        return "Σφάλμα: target_date πρέπει να είναι σε μορφή YYYY-MM-DD."

    if d < _today_date():
        return "Σφάλμα: Δεν επιτρέπεται ημερομηνία στο παρελθόν."

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


def _parse_behavior_hours(text: str) -> List[int]:
    """
    Examples:
      '7-9'
      '7-9,18-23'
      '6-8, 12-14, 19-22'
      '8'
    Convention:
      7-9  -> hours 7,8
      18-23 -> hours 18,19,20,21,22
    """
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
                raise ValueError("Τα διαστήματα high consumption πρέπει να είναι μέσα στο 0-24.")

            if end <= start:
                raise ValueError("Σε κάθε διάστημα πρέπει να ισχύει τέλος > αρχή, π.χ. 18-23.")

            for h in range(start, end):
                if 0 <= h <= 23:
                    hours.add(h)
        else:
            h = int(part)
            if not (0 <= h <= 23):
                raise ValueError("Οι ώρες high consumption πρέπει να είναι μέσα στο 0-23.")
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

    if factor < 1.0:
        raise ValueError("Το behavior factor πρέπει να είναι >= 1.0.")

    hours = _parse_behavior_hours(high_consumption_hours_text)

    if len(hours) == 0:
        return False, [], 1.0

    return True, hours, factor


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

    return {
        "rf": rf / 100.0,
        "xgb": xgb / 100.0,
        "lgbm": lgbm / 100.0,
    }


def _weights_text(weights: Dict[str, float]) -> str:
    return (
        f"Weights: RF={weights['rf']:.2f}, "
        f"XGB={weights['xgb']:.2f}, "
        f"LGBM={weights['lgbm']:.2f}"
    )


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


def _apply_behavior_adjustment_curve_df(
    curve_df: pd.DataFrame,
    hours: List[int],
    factor: float,
    target_cols: List[str],
) -> pd.DataFrame:
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


def _behavior_summary_suffix(
    enabled: bool,
    hours: List[int],
    factor: float,
) -> str:
    if not enabled:
        return "Behavioral adjustment: OFF"

    return (
        "Behavioral adjustment: ON\n"
        f"High consumption hours: {hours}\n"
        f"Behavior factor: {factor:.2f}"
    )


def _build_summary_single(
    out: Dict[str, Any],
    remaining_kwh_today: Optional[float] = None,
    cutoff_iso: Optional[str] = None,
    behavior_text: Optional[str] = None,
) -> str:
    flags = out.get("derived_flags", {}) or {}
    model_name = out.get("model_name") or out.get("label") or out.get("model_id")

    lines = [
        f"Model: {out.get('model_id')} — {model_name}",
        f"Mode: {out.get('mode')}",
        f"City: {out.get('city')} | Date: {out.get('target_date')}",
        f"Total kWh/day (24h): {float(out.get('total_kWh_day', 0.0)):.3f}",
    ]

    if remaining_kwh_today is not None and cutoff_iso is not None:
        lines.append(f"Remaining kWh (from {cutoff_iso}): {float(remaining_kwh_today):.3f}")

    lines += [
        f"Weather source: {out.get('weather_source')}",
        f"History source: {out.get('history_source')}",
        f"History store path: {out.get('history_store_path')}",
        f"Flags: weekend={flags.get('is_weekend')} holiday={flags.get('is_holiday')}",
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
    lines = [
        f"City: {first.get('city')} | Date: {first.get('target_date')}",
        f"Mode: {first.get('mode')}",
        "Model comparison:",
    ]

    for mid, out in results.items():
        model_name = out.get("model_name") or out.get("label") or mid
        total_kwh = float(out.get("total_kWh_day", 0.0))
        rem = remaining_map.get(mid)
        cutoff = cutoff_map.get(mid)
        row = f"- {mid} — {model_name}: Total kWh/day (24h) = {total_kwh:.3f}"
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
    weights: Dict[str, float],
    behavior_text: Optional[str] = None,
) -> str:
    lines = [
        "Combined prediction (weighted ensemble):",
        f"Mode: {mode}",
        f"City: {city} | Date: {target_date}",
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

    df["model_id"] = out.get("model_id")
    df["model_name"] = out.get("model_name") or out.get("label")
    df["model_type"] = out.get("model_type")
    df["mode"] = out.get("mode")
    df["city"] = out.get("city")
    df["target_date"] = out.get("target_date")
    df["weather_source"] = out.get("weather_source")
    df["history_source"] = out.get("history_source")
    df["history_store_path"] = out.get("history_store_path")
    df["history_window_start"] = out.get("history_window_start")
    df["history_window_end"] = out.get("history_window_end")

    flags = out.get("derived_flags", {}) or {}
    df["is_weekend"] = flags.get("is_weekend")
    df["is_holiday"] = flags.get("is_holiday")

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
    curve_df: pd.DataFrame,
    temps24: Optional[List[float]],
    remaining_kwh_today: Optional[float],
    cutoff_iso: Optional[str],
    weights: Dict[str, float],
) -> Path:
    df = curve_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if temps24 is not None and len(temps24) == 24 and len(df) == 24:
        df["external_temp_C"] = [float(x) for x in temps24]
    else:
        df["external_temp_C"] = pd.NA

    df["model_id"] = "combined"
    df["model_name"] = "Weighted Ensemble"
    df["model_type"] = "ensemble"
    df["mode"] = mode
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


def _compute_remaining_kwh_today(df_full: pd.DataFrame, target_date: str) -> Tuple[Optional[float], Optional[str]]:
    try:
        d = _parse_date(target_date)
    except Exception:
        return None, None

    if d != _today_date():
        return None, None

    if df_full is None or df_full.empty or "timestamp" not in df_full.columns or "pred_consumption_Wh" not in df_full.columns:
        return None, None

    now = _now_local()
    cutoff = _next_full_hour(now)
    cutoff_iso = cutoff.isoformat()

    df_tmp = df_full.copy()
    df_tmp["timestamp"] = pd.to_datetime(df_tmp["timestamp"])
    df_tmp = df_tmp[df_tmp["timestamp"] >= cutoff]

    if df_tmp.empty:
        return 0.0, cutoff_iso

    remaining_kwh = float(df_tmp["pred_consumption_Wh"].sum() / 1000.0)
    return remaining_kwh, cutoff_iso


def _compute_remaining_kwh_today_for_col(
    df_full: pd.DataFrame,
    target_date: str,
    value_col: str,
) -> Tuple[Optional[float], Optional[str]]:
    try:
        d = _parse_date(target_date)
    except Exception:
        return None, None

    if d != _today_date():
        return None, None

    if df_full is None or df_full.empty or "timestamp" not in df_full.columns or value_col not in df_full.columns:
        return None, None

    now = _now_local()
    cutoff = _next_full_hour(now)
    cutoff_iso = cutoff.isoformat()

    df_tmp = df_full.copy()
    df_tmp["timestamp"] = pd.to_datetime(df_tmp["timestamp"])
    df_tmp = df_tmp[df_tmp["timestamp"] >= cutoff]

    if df_tmp.empty:
        return 0.0, cutoff_iso

    remaining_kwh = float(df_tmp[value_col].sum() / 1000.0)
    return remaining_kwh, cutoff_iso


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

    ax.plot(
        plot_df["hour"],
        y,
        color="#1f77b4",
        linewidth=2.5,
        marker="o",
        markersize=4,
        label="Prediction",
    )

    if y.notna().any():
        avg = y.mean()
        ax.axhline(
            avg,
            linestyle="--",
            linewidth=1,
            color="gray",
            label="Daily average",
        )

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

    colors = {
        "rf_Wh": "#1f77b4",
        "xgb_Wh": "#ff7f0e",
        "lgbm_Wh": "#2ca02c",
        "ensemble_Wh": "#d62728",
    }

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)

    for col in df.columns:
        if col in ["timestamp", "hour"]:
            continue

        temp = base.merge(df[["hour", col]], on="hour", how="left")

        ax.plot(
            temp["hour"],
            temp[col],
            linewidth=2,
            marker="o",
            markersize=4,
            label=col.replace("_Wh", "").upper(),
            color=colors.get(col, "black"),
        )

    ax.set_xticks(hours)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Predicted consumption (Wh)")
    ax.set_title("Model comparison — hourly electricity prediction")
    ax.grid(alpha=0.3)
    ax.legend()

    return fig


def _prepare_payload(
    use_history: bool,
    model_choice: str,
    city: str,
    target_date: str,
    total_floor_area_m2: float,
    residents: float,
    hometype: str,
    urban_rural_class: str,
    t_min: Optional[float],
    t_max: Optional[float],
    external_temperature_24h_text: str,
    use_proxy_history: bool,
    min_history_hours: int,
    history_consumption_text: str,
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
    mode = "withhistory" if use_history else "coldstart"

    payload: Dict[str, Any] = {
        "model_id": model_id,
        "mode": mode,
        "city": city,
        "target_date": target_date,
        "total_floor_area_m2": float(total_floor_area_m2),
        "residents": float(residents),
        "hometype": hometype,
        "urban_rural_class": str(urban_rural_class),
        "save_csv": False,
        "use_proxy_history": bool(use_proxy_history),
        "min_history_hours": int(min_history_hours),
    }

    temps24_user: Optional[List[float]] = None

    if external_temperature_24h_text and external_temperature_24h_text.strip():
        try:
            vals = [float(x.strip()) for x in external_temperature_24h_text.split(",")]
        except Exception:
            raise ValueError("Σφάλμα: external_temperature_24h δεν είναι έγκυρη λίστα αριθμών (comma-separated).")

        if len(vals) != 24:
            raise ValueError(f"Σφάλμα: external_temperature_24h πρέπει να έχει 24 τιμές (έχει {len(vals)}).")

        payload["external_temperature_24h"] = vals
        temps24_user = vals
    else:
        if t_min is not None and t_max is not None:
            payload["t_min"] = float(t_min)
            payload["t_max"] = float(t_max)

    if history_consumption_text and history_consumption_text.strip():
        try:
            hvals = [float(x.strip()) for x in history_consumption_text.split(",")]
        except Exception:
            raise ValueError("Σφάλμα: history_consumption_Wh δεν είναι έγκυρη λίστα αριθμών (comma-separated).")
        payload["history_consumption_Wh"] = hvals

    return payload, temps24_user


def _run_single_model(payload: Dict[str, Any], temps24_user: Optional[List[float]], target_date: str, save_csv: bool):
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

    out = r.json()
    preds = out.get("predictions", []) or []

    df_full = pd.DataFrame(preds)
    if not df_full.empty:
        df_full["timestamp"] = pd.to_datetime(df_full["timestamp"])
        df_full = df_full.sort_values("timestamp").reset_index(drop=True)

    temps24 = out.get("external_temperature_24h")
    if temps24 is None:
        temps24 = temps24_user

    temps_df_full = _temps24_to_df(temps24)

    remaining_kwh_today, cutoff_iso = _compute_remaining_kwh_today(df_full, target_date)

    df_show = df_full.copy()
    temps_df_show = temps_df_full.copy()
    df_show, temps_df_show = _filter_today_for_display(df_show, temps_df_show, target_date)

    summary = _build_summary_single(out, remaining_kwh_today=remaining_kwh_today, cutoff_iso=cutoff_iso)

    status = "✅ Done"
    if save_csv and not df_full.empty:
        fp = _save_csv_full_day(
            out=out,
            preds_full=df_full,
            temps24=temps24,
            remaining_kwh_today=remaining_kwh_today,
            cutoff_iso=cutoff_iso,
        )
        status = f"✅ Saved CSV (24h): {fp}"

    return summary, df_show, temps_df_show, status, out


# =========================
# BUTTON ACTIONS
# =========================
def do_predict(
    use_history: bool,
    model_choice: str,
    city: str,
    target_date: str,
    total_floor_area_m2: float,
    residents: float,
    hometype: str,
    urban_rural_class: str,
    t_min: Optional[float],
    t_max: Optional[float],
    external_temperature_24h_text: str,
    use_proxy_history: bool,
    min_history_hours: int,
    history_consumption_text: str,
    enable_behavior_adjustment: bool,
    high_consumption_hours_text: str,
    behavior_factor: Optional[float],
    use_custom_weights: bool,       # ignored in predict
    rf_weight_pct: Optional[float],  # ignored in predict
    xgb_weight_pct: Optional[float],  # ignored in predict
    lgbm_weight_pct: Optional[float],  # ignored in predict
    save_csv: bool,
):
    try:
        behavior_enabled, behavior_hours, behavior_factor_norm = _normalize_behavior_inputs(
            enable_behavior_adjustment,
            high_consumption_hours_text,
            behavior_factor,
        )

        payload, temps24_user = _prepare_payload(
            use_history,
            model_choice,
            city,
            target_date,
            total_floor_area_m2,
            residents,
            hometype,
            urban_rural_class,
            t_min,
            t_max,
            external_temperature_24h_text,
            use_proxy_history,
            min_history_hours,
            history_consumption_text,
        )

        summary, df_show, temps_df_show, _status, out = _run_single_model(
            payload=payload,
            temps24_user=temps24_user,
            target_date=target_date,
            save_csv=False,
        )

        preds_full = pd.DataFrame(out.get("predictions", []) or [])
        if not preds_full.empty:
            preds_full["timestamp"] = pd.to_datetime(preds_full["timestamp"])
            preds_full = preds_full.sort_values("timestamp").reset_index(drop=True)

        temps24 = out.get("external_temperature_24h")
        if temps24 is None:
            temps24 = temps24_user

        if behavior_enabled:
            preds_full = _apply_behavior_adjustment_single_df(
                preds_full,
                behavior_hours,
                behavior_factor_norm,
                value_col="pred_consumption_Wh",
            )

        remaining_kwh_today, cutoff_iso = _compute_remaining_kwh_today(
            preds_full[["timestamp", "pred_consumption_Wh"]],
            target_date,
        )

        df_show = preds_full.copy()
        df_show, temps_df_show = _filter_today_for_display(df_show, temps_df_show, target_date)

        out_for_summary = dict(out)
        out_for_summary["total_kWh_day"] = float(preds_full["pred_consumption_Wh"].sum() / 1000.0)

        behavior_text = _behavior_summary_suffix(
            behavior_enabled,
            behavior_hours,
            behavior_factor_norm,
        )

        summary = _build_summary_single(
            out_for_summary,
            remaining_kwh_today=remaining_kwh_today,
            cutoff_iso=cutoff_iso,
            behavior_text=behavior_text,
        )

        fig = _build_single_chart(df_show[["timestamp", "pred_consumption_Wh"]])

        status = "✅ Done"
        if save_csv and not preds_full.empty:
            fp = _save_csv_full_day(
                out=out_for_summary,
                preds_full=preds_full,
                temps24=temps24,
                remaining_kwh_today=remaining_kwh_today,
                cutoff_iso=cutoff_iso,
            )
            status = f"✅ Saved CSV (24h): {fp}"

        return (
            summary,
            df_show,
            temps_df_show,
            pd.DataFrame(),
            fig,
            status,
            gr.update(visible=False, value=None),
            gr.update(visible=False, value=None),
        )

    except ValueError as e:
        return (
            str(e),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            None,
            "⚠️ Fix inputs and try again.",
            gr.update(visible=False, value=None),
            gr.update(visible=False, value=None),
        )

    except RuntimeError as e:
        msg = str(e)

        if msg.startswith("WEATHER_FALLBACK::"):
            detail = msg.split("WEATHER_FALLBACK::", 1)[1]
            return (
                "⚠️ Δεν βρέθηκαν αυτόματα δεδομένα εξωτερικής θερμοκρασίας για αυτή την ημερομηνία.\n"
                "Συμπλήρωσε t_min και t_max (ή δώσε 24 ωριαίες τιμές) και ξαναπάτα Predict.",
                pd.DataFrame(),
                pd.DataFrame(),
                pd.DataFrame(),
                None,
                f"API 422: {detail}",
                gr.update(visible=True, value=None),
                gr.update(visible=True, value=None),
            )

        return (
            msg,
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            None,
            "❌ API error",
            gr.update(visible=False, value=None),
            gr.update(visible=False, value=None),
        )

    except Exception as e:
        return (
            f"Σφάλμα επικοινωνίας με API: {e}",
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            None,
            "❌ API not reachable",
            gr.update(visible=False, value=None),
            gr.update(visible=False, value=None),
        )


def do_compare(
    use_history: bool,
    model_choice: str,  # ignored in compare
    city: str,
    target_date: str,
    total_floor_area_m2: float,
    residents: float,
    hometype: str,
    urban_rural_class: str,
    t_min: Optional[float],
    t_max: Optional[float],
    external_temperature_24h_text: str,
    use_proxy_history: bool,
    min_history_hours: int,
    history_consumption_text: str,
    enable_behavior_adjustment: bool,
    high_consumption_hours_text: str,
    behavior_factor: Optional[float],
    use_custom_weights: bool,        # ignored in compare
    rf_weight_pct: Optional[float],   # ignored in compare
    xgb_weight_pct: Optional[float],  # ignored in compare
    lgbm_weight_pct: Optional[float],  # ignored in compare
    save_csv: bool,  # ignored in compare
):
    try:
        behavior_enabled, behavior_hours, behavior_factor_norm = _normalize_behavior_inputs(
            enable_behavior_adjustment,
            high_consumption_hours_text,
            behavior_factor,
        )

        payload, temps24_user = _prepare_payload(
            use_history,
            "rf — Random Forest (final)",
            city,
            target_date,
            total_floor_area_m2,
            residents,
            hometype,
            urban_rural_class,
            t_min,
            t_max,
            external_temperature_24h_text,
            use_proxy_history,
            min_history_hours,
            history_consumption_text,
        )

        results: Dict[str, Dict[str, Any]] = {}
        temps_df_show = pd.DataFrame()
        primary_df_show = pd.DataFrame()

        for mid in ["rf", "xgb", "lgbm"]:
            payload_mid = dict(payload)
            payload_mid["model_id"] = mid

            r = requests.post(f"{API_BASE}/predict", json=payload_mid, timeout=TIMEOUT)

            if r.status_code != 200:
                detail = ""
                try:
                    js = r.json()
                    detail = js.get("detail", "") if isinstance(js, dict) else str(js)
                except Exception:
                    detail = r.text

                if r.status_code == 422 and _needs_weather_fallback(detail):
                    raise RuntimeError(f"WEATHER_FALLBACK::{detail}")

                raise RuntimeError(f"API error ({r.status_code}) for model '{mid}': {detail}")

            out_mid = r.json()
            results[mid] = out_mid

            if mid == "rf":
                preds_rf = pd.DataFrame(out_mid.get("predictions", []) or [])
                if not preds_rf.empty:
                    preds_rf["timestamp"] = pd.to_datetime(preds_rf["timestamp"])
                    preds_rf = preds_rf.sort_values("timestamp").reset_index(drop=True)

                if behavior_enabled:
                    preds_rf = _apply_behavior_adjustment_single_df(
                        preds_rf,
                        behavior_hours,
                        behavior_factor_norm,
                        value_col="pred_consumption_Wh",
                    )

                primary_df_show = preds_rf.copy()

                temps24 = out_mid.get("external_temperature_24h")
                if temps24 is None:
                    temps24 = temps24_user

                temps_df_full = _temps24_to_df(temps24)
                primary_df_show, temps_df_show = _filter_today_for_display(
                    primary_df_show, temps_df_full.copy(), target_date
                )

        comparison_rows = []
        curve_df = None
        remaining_map: Dict[str, Optional[float]] = {}
        cutoff_map: Dict[str, Optional[str]] = {}

        for mid, out_mid in results.items():
            preds_mid = pd.DataFrame(out_mid.get("predictions", []) or [])
            if not preds_mid.empty:
                preds_mid["timestamp"] = pd.to_datetime(preds_mid["timestamp"])
                preds_mid = preds_mid.sort_values("timestamp").reset_index(drop=True)

            if behavior_enabled:
                preds_mid = _apply_behavior_adjustment_single_df(
                    preds_mid,
                    behavior_hours,
                    behavior_factor_norm,
                    value_col="pred_consumption_Wh",
                )

            rem_mid, cutoff_mid = _compute_remaining_kwh_today(
                preds_mid[["timestamp", "pred_consumption_Wh"]],
                target_date,
            )
            remaining_map[mid] = rem_mid
            cutoff_map[mid] = cutoff_mid

            comparison_rows.append(
                {
                    "model": mid,
                    "model_name": out_mid.get("model_name") or out_mid.get("label"),
                    "mode": out_mid.get("mode"),
                    "total_kWh_day": float(preds_mid["pred_consumption_Wh"].sum() / 1000.0),
                    "remaining_kWh_today": rem_mid,
                }
            )

            if not preds_mid.empty:
                temp_curve = preds_mid[["timestamp", "pred_consumption_Wh"]].copy()
                temp_curve.rename(columns={"pred_consumption_Wh": f"{mid}_Wh"}, inplace=True)

                if curve_df is None:
                    curve_df = temp_curve
                else:
                    curve_df = curve_df.merge(temp_curve, on="timestamp", how="outer")

        comparison_df = pd.DataFrame(comparison_rows)
        comparison_fig = None

        if curve_df is not None and not curve_df.empty:
            curve_df = curve_df.sort_values("timestamp").reset_index(drop=True)
            curve_df = _filter_today_curve_only(curve_df, target_date)
            comparison_fig = _build_compare_chart(curve_df)

        behavior_text = _behavior_summary_suffix(
            behavior_enabled,
            behavior_hours,
            behavior_factor_norm,
        )

        summary = _build_summary_compare(
            results,
            remaining_map,
            cutoff_map,
            behavior_text=behavior_text,
        )

        return (
            summary,
            primary_df_show,
            temps_df_show,
            comparison_df,
            comparison_fig,
            "✅ Comparison done",
            gr.update(visible=False, value=None),
            gr.update(visible=False, value=None),
        )

    except ValueError as e:
        return (
            str(e),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            None,
            "⚠️ Fix inputs and try again.",
            gr.update(visible=False, value=None),
            gr.update(visible=False, value=None),
        )

    except RuntimeError as e:
        msg = str(e)

        if msg.startswith("WEATHER_FALLBACK::"):
            detail = msg.split("WEATHER_FALLBACK::", 1)[1]
            return (
                "⚠️ Δεν βρέθηκαν αυτόματα δεδομένα εξωτερικής θερμοκρασίας για αυτή την ημερομηνία.\n"
                "Συμπλήρωσε t_min και t_max (ή δώσε 24 ωριαίες τιμές) και ξαναπάτα Compare.",
                pd.DataFrame(),
                pd.DataFrame(),
                pd.DataFrame(),
                None,
                f"API 422: {detail}",
                gr.update(visible=True, value=None),
                gr.update(visible=True, value=None),
            )

        return (
            msg,
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            None,
            "❌ API error",
            gr.update(visible=False, value=None),
            gr.update(visible=False, value=None),
        )

    except Exception as e:
        return (
            f"Σφάλμα επικοινωνίας με API: {e}",
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            None,
            "❌ API not reachable",
            gr.update(visible=False, value=None),
            gr.update(visible=False, value=None),
        )


def do_combined(
    use_history: bool,
    model_choice: str,  # ignored in combined
    city: str,
    target_date: str,
    total_floor_area_m2: float,
    residents: float,
    hometype: str,
    urban_rural_class: str,
    t_min: Optional[float],
    t_max: Optional[float],
    external_temperature_24h_text: str,
    use_proxy_history: bool,
    min_history_hours: int,
    history_consumption_text: str,
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

        weights = _resolve_ensemble_weights(
            use_custom_weights,
            rf_weight_pct,
            xgb_weight_pct,
            lgbm_weight_pct,
        )

        payload, temps24_user = _prepare_payload(
            use_history,
            "rf — Random Forest (final)",
            city,
            target_date,
            total_floor_area_m2,
            residents,
            hometype,
            urban_rural_class,
            t_min,
            t_max,
            external_temperature_24h_text,
            use_proxy_history,
            min_history_hours,
            history_consumption_text,
        )

        results: Dict[str, Dict[str, Any]] = {}
        temps_df_show = pd.DataFrame()
        curve_df = None
        temps24_for_save = None

        for mid in ["rf", "xgb", "lgbm"]:
            payload_mid = dict(payload)
            payload_mid["model_id"] = mid

            r = requests.post(f"{API_BASE}/predict", json=payload_mid, timeout=TIMEOUT)

            if r.status_code != 200:
                detail = ""
                try:
                    js = r.json()
                    detail = js.get("detail", "") if isinstance(js, dict) else str(js)
                except Exception:
                    detail = r.text

                if r.status_code == 422 and _needs_weather_fallback(detail):
                    raise RuntimeError(f"WEATHER_FALLBACK::{detail}")

                raise RuntimeError(f"API error ({r.status_code}) for model '{mid}': {detail}")

            out_mid = r.json()
            results[mid] = out_mid

            preds_mid = pd.DataFrame(out_mid.get("predictions", []) or [])
            if not preds_mid.empty:
                preds_mid["timestamp"] = pd.to_datetime(preds_mid["timestamp"])
                preds_mid = preds_mid.sort_values("timestamp").reset_index(drop=True)

            if temps24_for_save is None:
                temps24_for_save = out_mid.get("external_temperature_24h")
                if temps24_for_save is None:
                    temps24_for_save = temps24_user
                temps_df_show = _temps24_to_df(temps24_for_save)

            if not preds_mid.empty:
                temp_curve = preds_mid[["timestamp", "pred_consumption_Wh"]].copy()
                temp_curve.rename(columns={"pred_consumption_Wh": f"{mid}_Wh"}, inplace=True)

                if curve_df is None:
                    curve_df = temp_curve
                else:
                    curve_df = curve_df.merge(temp_curve, on="timestamp", how="outer")

        if curve_df is None or curve_df.empty:
            raise RuntimeError("No predictions available for combined forecast.")

        curve_df = curve_df.sort_values("timestamp").reset_index(drop=True)

        curve_df["ensemble_Wh"] = (
            weights["rf"] * curve_df["rf_Wh"]
            + weights["xgb"] * curve_df["xgb_Wh"]
            + weights["lgbm"] * curve_df["lgbm_Wh"]
        )

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

        remaining_kwh_today, cutoff_iso = _compute_remaining_kwh_today_for_col(
            display_curve,
            target_date,
            "pred_consumption_Wh",
        )

        display_curve, temps_df_show = _filter_today_for_display(display_curve, temps_df_show, target_date)

        behavior_text = _behavior_summary_suffix(
            behavior_enabled,
            behavior_hours,
            behavior_factor_norm,
        )

        summary = _build_summary_combined(
            total_kwh_day=total_kwh_day,
            remaining_kwh_today=remaining_kwh_today,
            cutoff_iso=cutoff_iso,
            city=city,
            target_date=target_date,
            mode=("withhistory" if use_history else "coldstart"),
            weights=weights,
            behavior_text=behavior_text,
        )

        comparison_df = pd.DataFrame(
            [
                {
                    "model": "rf",
                    "weight": weights["rf"],
                    "total_kWh_day": float(curve_df["rf_Wh"].sum() / 1000.0),
                },
                {
                    "model": "xgb",
                    "weight": weights["xgb"],
                    "total_kWh_day": float(curve_df["xgb_Wh"].sum() / 1000.0),
                },
                {
                    "model": "lgbm",
                    "weight": weights["lgbm"],
                    "total_kWh_day": float(curve_df["lgbm_Wh"].sum() / 1000.0),
                },
                {
                    "model": "combined",
                    "weight": 1.0,
                    "total_kWh_day": total_kwh_day,
                },
            ]
        )

        fig_curve = curve_df.copy()
        fig_curve = _filter_today_curve_only(fig_curve, target_date)
        fig = _build_compare_chart(fig_curve[["timestamp", "rf_Wh", "xgb_Wh", "lgbm_Wh", "ensemble_Wh"]])

        status = "✅ Combined prediction done"

        if save_csv:
            fp = _save_csv_combined(
                city=city,
                target_date=target_date,
                mode=("withhistory" if use_history else "coldstart"),
                curve_df=curve_df,
                temps24=temps24_for_save,
                remaining_kwh_today=remaining_kwh_today,
                cutoff_iso=cutoff_iso,
                weights=weights,
            )
            status = f"✅ Saved combined CSV: {fp}"

        return (
            summary,
            display_curve,
            temps_df_show,
            comparison_df,
            fig,
            status,
            gr.update(visible=False, value=None),
            gr.update(visible=False, value=None),
        )

    except ValueError as e:
        return (
            str(e),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            None,
            "⚠️ Fix inputs and try again.",
            gr.update(visible=False, value=None),
            gr.update(visible=False, value=None),
        )

    except RuntimeError as e:
        msg = str(e)

        if msg.startswith("WEATHER_FALLBACK::"):
            detail = msg.split("WEATHER_FALLBACK::", 1)[1]
            return (
                "⚠️ Δεν βρέθηκαν αυτόματα δεδομένα εξωτερικής θερμοκρασίας για αυτή την ημερομηνία.\n"
                "Συμπλήρωσε t_min και t_max (ή δώσε 24 ωριαίες τιμές) και ξαναπάτα Combined Prediction.",
                pd.DataFrame(),
                pd.DataFrame(),
                pd.DataFrame(),
                None,
                f"API 422: {detail}",
                gr.update(visible=True, value=None),
                gr.update(visible=True, value=None),
            )

        return (
            msg,
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            None,
            "❌ API error",
            gr.update(visible=False, value=None),
            gr.update(visible=False, value=None),
        )

    except Exception as e:
        return (
            f"Σφάλμα επικοινωνίας με API: {e}",
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            None,
            "❌ API not reachable",
            gr.update(visible=False, value=None),
            gr.update(visible=False, value=None),
        )


# =========================
# UI BUILD
# =========================
MODELS = _fetch_models()
MODEL_CHOICES = [_format_model_choice(m) for m in MODELS]
DEFAULT_MODEL = MODEL_CHOICES[0] if MODEL_CHOICES else "rf — Random Forest (final)"


def _on_use_history_toggle(use_history: bool):
    return gr.update(visible=use_history)


def _toggle_custom_weights(use_custom: bool):
    return gr.update(visible=use_custom)


def _reset_temps_on_change(_):
    return gr.update(visible=False, value=None), gr.update(visible=False, value=None)


with gr.Blocks(title="IDEAL Forecasting UI") as demo:
    gr.Markdown("## IDEAL Load Forecasting (Demo UI)")
    gr.Markdown(
        "Συμπλήρωσε τα πεδία και πάτα **Predict** για ένα μοντέλο, "
        "**Compare** για RF / XGB / LGBM ή **Combined Prediction** για weighted ensemble."
    )

    api_status = gr.Markdown(_api_health())

    with gr.Row():
        use_history = gr.Checkbox(
            value=False,
            label="Πρόβλεψη με χρήση ιστορικών δεδομένων",
            info="Αν το ενεργοποιήσεις, το API θα χρησιμοποιήσει mode='withhistory'.",
        )

    with gr.Row():
        model_choice = gr.Dropdown(
            choices=MODEL_CHOICES,
            value=DEFAULT_MODEL,
            label="Μοντέλο",
            info="Επίλεξε οικογένεια μοντέλου για single run.",
        )

    with gr.Row():
        city = gr.Dropdown(choices=DEFAULT_CITIES, value="Edinburgh", label="City")
        target_date = gr.Textbox(
            value=_today_date().isoformat(),
            label="Target date (YYYY-MM-DD)",
            info="Δεν επιτρέπονται ημερομηνίες στο παρελθόν.",
        )

    with gr.Row():
        total_floor_area_m2 = gr.Number(value=85, minimum=1, label="Total floor area (m²)")
        residents = gr.Number(value=2, minimum=1, label="Residents")

    with gr.Row():
        hometype = gr.Dropdown(choices=[x[0] for x in DEFAULT_HOMETYPES], value="flat", label="Home type")
        urban_rural_class = gr.Dropdown(choices=URBAN_RURAL_OPTIONS, value="1", label="Urban/Rural class")

    with gr.Accordion("Weather", open=True):
        gr.Markdown(
            "Το σύστημα προσπαθεί να βρει αυτόματα την ωριαία εξωτερική θερμοκρασία (Open-Meteo).\n"
            "Αν **δεν** βρεθεί, θα σου ζητήσει **t_min / t_max** (ή 24 ωριαίες τιμές)."
        )

        with gr.Row():
            t_min = gr.Number(value=None, label="t_min (°C)", visible=False)
            t_max = gr.Number(value=None, label="t_max (°C)", visible=False)

        external_temperature_24h_text = gr.Textbox(
            value="",
            label="external_temperature_24h (24 τιμές, comma-separated) [optional]",
            placeholder="π.χ. 3.1, 2.9, 2.7, ... (24 συνολικά)",
        )

    with gr.Group(visible=False) as history_group:
        gr.Markdown("### History options (μόνο για with-history mode)")
        use_proxy_history = gr.Checkbox(
            value=True,
            label="Use history from CSV store",
            info="Αν δεν δώσεις manual history, το API θα χρησιμοποιήσει το history_store.csv.",
        )
        min_history_hours = gr.Number(value=168, precision=0, minimum=1, label="min_history_hours")
        history_consumption_text = gr.Textbox(
            value="",
            label="history_consumption_Wh (comma-separated) [optional]",
            placeholder="π.χ. 120, 98, 105, ... (>=min_history_hours τιμές αν το δώσεις εσύ)",
        )

    with gr.Accordion("Behavioral adjustment (optional)", open=False):
        gr.Markdown(
            "Προαιρετικό post-processing των προβλέψεων.\n"
            "Δήλωσε διαστήματα αυξημένης αναμενόμενης κατανάλωσης, π.χ. **7-9,18-23**.\n"
            "Το ML μοντέλο δεν αλλάζει· η προσαρμογή εφαρμόζεται μόνο στο UI."
        )

        enable_behavior_adjustment = gr.Checkbox(
            value=False,
            label="Enable behavioral adjustment",
        )

        high_consumption_hours_text = gr.Textbox(
            value="",
            label="High consumption hours",
            placeholder="π.χ. 7-9, 13-15, 18-23",
        )

        behavior_factor = gr.Number(
            value=1.15,
            minimum=1.0,
            label="Behavior factor",
            info="Παράδειγμα: 1.15 σημαίνει +15% στις δηλωμένες ώρες.",
        )

    with gr.Accordion("Ensemble weights (optional)", open=False):
        gr.Markdown(
            "Μπορείς να ορίσεις χειροκίνητα τα βάρη του combined prediction.\n"
            "Τα βάρη πρέπει να αθροίζουν σε 100%."
        )

        use_custom_weights = gr.Checkbox(
            value=False,
            label="Use custom ensemble weights",
        )

        with gr.Row(visible=False) as weights_group:
            rf_weight_pct = gr.Number(value=50, minimum=0, maximum=100, label="RF weight (%)")
            xgb_weight_pct = gr.Number(value=25, minimum=0, maximum=100, label="XGB weight (%)")
            lgbm_weight_pct = gr.Number(value=25, minimum=0, maximum=100, label="LGBM weight (%)")

    with gr.Row():
        save_csv = gr.Checkbox(value=False, label="Save prediction CSV")

    with gr.Row():
        btn_predict = gr.Button("Predict", variant="primary")
        btn_compare = gr.Button("Compare")
        btn_combined = gr.Button("Combined Prediction")

    with gr.Row():
        summary = gr.Textbox(label="Summary", lines=16)

    with gr.Row():
        table = gr.Dataframe(
            label="Predictions (hourly) — single model / combined prediction / RF preview in compare",
            interactive=False,
        )

    with gr.Row():
        temps_table = gr.Dataframe(label="External temperature used (hourly)", interactive=False)

    with gr.Row():
        comparison_table = gr.Dataframe(label="Model comparison", interactive=False)

    with gr.Row():
        comparison_plot = gr.Plot(label="Prediction / Comparison chart")

    with gr.Row():
        status = gr.Textbox(label="Status", lines=2)

    use_history.change(fn=_on_use_history_toggle, inputs=[use_history], outputs=[history_group])

    use_custom_weights.change(
        fn=_toggle_custom_weights,
        inputs=[use_custom_weights],
        outputs=[weights_group],
    )

    target_date.change(fn=_reset_temps_on_change, inputs=[target_date], outputs=[t_min, t_max])
    city.change(fn=_reset_temps_on_change, inputs=[city], outputs=[t_min, t_max])

    common_inputs = [
        use_history,
        model_choice,
        city,
        target_date,
        total_floor_area_m2,
        residents,
        hometype,
        urban_rural_class,
        t_min,
        t_max,
        external_temperature_24h_text,
        use_proxy_history,
        min_history_hours,
        history_consumption_text,
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

    btn_predict.click(
        fn=do_predict,
        inputs=common_inputs,
        outputs=common_outputs,
    )

    btn_compare.click(
        fn=do_compare,
        inputs=common_inputs,
        outputs=common_outputs,
    )

    btn_combined.click(
        fn=do_combined,
        inputs=common_inputs,
        outputs=common_outputs,
    )

demo.launch(server_name=UI_HOST, server_port=UI_PORT, share=False)