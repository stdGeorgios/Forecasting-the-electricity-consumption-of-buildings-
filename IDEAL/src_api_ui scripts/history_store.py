# C:/IDEAL_Programming/src/history_store.py
# ---------------------------------------------------------
# CSV-only history store for IDEAL API/UI
# ---------------------------------------------------------
#
# Expected CSV default path:
#   C:/IDEAL_Programming/processed/stores/history_store.csv
#
# Minimal columns:
#   - timestamp       hourly, parseable datetime
#   - consumption_Wh  numeric hourly consumption in Wh
#
# Purpose in the final generic UI:
#   - The user does NOT provide home_id.
#   - With-history models require recent hourly consumption history.
#   - This module returns the previous N hours before target_date 00:00.
#   - Default N = 168 hours, matching the generic UI models.
#
# Behavior:
#   1. Try exact requested window:
#      [target_date 00:00 - history_hours, target_date 00:00)
#   2. If the exact window is incomplete and fallback_to_latest=True:
#      return the latest contiguous history_hours available in the CSV.
#   3. If no valid window exists, raise a clear error.
#
# Notes:
#   - This file does not infer or require home_id.
#   - Timestamps are normalized to hourly timestamps.
#   - Duplicate timestamps are resolved by keeping the last row.
# ---------------------------------------------------------

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


# ============================================================
# Paths / defaults
# ============================================================

BASE_DIR = Path(os.getenv("IDEAL_BASE_DIR", "C:/IDEAL_Programming"))
STORES_DIR = Path(os.getenv("IDEAL_STORES_DIR", str(BASE_DIR / "processed" / "stores")))
HISTORY_CSV_PATH = Path(os.getenv("IDEAL_HISTORY_CSV", str(STORES_DIR / "history_store.csv")))

DEFAULT_HISTORY_HOURS = int(os.getenv("IDEAL_DEFAULT_HISTORY_HOURS", "168"))

TIMESTAMP_COL = "timestamp"
CONSUMPTION_COL = "consumption_Wh"

TIMESTAMP_ALIASES = ["time", "datetime", "date_time", "date", "ds"]
CONSUMPTION_ALIASES = ["consumption", "energy_Wh", "Wh", "value", "y"]


# ============================================================
# Internal helpers
# ============================================================

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize expected timestamp and consumption column names."""
    df = df.copy()

    if TIMESTAMP_COL not in df.columns:
        for alt in TIMESTAMP_ALIASES:
            if alt in df.columns:
                df = df.rename(columns={alt: TIMESTAMP_COL})
                break

    if TIMESTAMP_COL not in df.columns:
        raise KeyError(
            "history_store.csv must contain a 'timestamp' column "
            "or one of these alternatives: " + ", ".join(TIMESTAMP_ALIASES)
        )

    if CONSUMPTION_COL not in df.columns:
        for alt in CONSUMPTION_ALIASES:
            if alt in df.columns:
                df = df.rename(columns={alt: CONSUMPTION_COL})
                break

    if CONSUMPTION_COL not in df.columns:
        raise KeyError(
            "history_store.csv must contain 'consumption_Wh' "
            "or one of these alternatives: " + ", ".join(CONSUMPTION_ALIASES)
        )

    return df


def _load_history_csv() -> pd.DataFrame:
    """Load, normalize and clean the history CSV."""
    if not HISTORY_CSV_PATH.exists():
        raise FileNotFoundError(
            f"History CSV not found: {HISTORY_CSV_PATH}\n"
            f"Put a file named 'history_store.csv' under: {STORES_DIR}"
        )

    df = pd.read_csv(HISTORY_CSV_PATH)
    df = _normalize_columns(df)

    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
    df[CONSUMPTION_COL] = pd.to_numeric(df[CONSUMPTION_COL], errors="coerce")

    df = df.dropna(subset=[TIMESTAMP_COL, CONSUMPTION_COL]).copy()

    # Normalize to hourly timestamps. If a timestamp is already hourly, this is unchanged.
    # If non-hourly timestamps exist, they are floored to the hour and the last value is kept.
    df[TIMESTAMP_COL] = df[TIMESTAMP_COL].dt.floor("h")

    df = df.sort_values(TIMESTAMP_COL)
    df = df.drop_duplicates(subset=[TIMESTAMP_COL], keep="last")

    # Consumption should not be negative in this project.
    df[CONSUMPTION_COL] = df[CONSUMPTION_COL].clip(lower=0)

    return df[[TIMESTAMP_COL, CONSUMPTION_COL]].reset_index(drop=True)


def _expected_hourly_index(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Return hourly timestamps in [start, end)."""
    return pd.date_range(start=start, end=end - pd.Timedelta(hours=1), freq="h")


def _is_complete_hourly_window(df_window: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    expected = _expected_hourly_index(start, end)

    if len(df_window) != len(expected):
        return False

    actual = pd.DatetimeIndex(df_window[TIMESTAMP_COL].sort_values().values)
    return actual.equals(expected)


def _format_missing_hours(df_window: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, max_show: int = 10) -> List[str]:
    expected = set(_expected_hourly_index(start, end))
    actual = set(pd.to_datetime(df_window[TIMESTAMP_COL]).tolist())
    missing = sorted(expected - actual)
    return [pd.Timestamp(x).isoformat() for x in missing[:max_show]]


def _latest_contiguous_window(df: pd.DataFrame, history_hours: int) -> pd.DataFrame:
    """Find the latest contiguous hourly block of length history_hours."""
    if len(df) < history_hours:
        raise ValueError(
            f"CSV history too short: only {len(df)} rows, need >= {history_hours}."
        )

    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True).copy()

    # Consecutive hourly group id. A break occurs when difference != 1 hour.
    diffs = df[TIMESTAMP_COL].diff()
    new_group = diffs.ne(pd.Timedelta(hours=1)).fillna(True)
    df["_contiguous_group"] = new_group.cumsum()

    valid_groups = []
    for _, g in df.groupby("_contiguous_group", sort=False):
        if len(g) >= history_hours:
            valid_groups.append(g.tail(history_hours).copy())

    if not valid_groups:
        raise ValueError(
            f"No contiguous hourly history window with {history_hours} rows was found in {HISTORY_CSV_PATH}."
        )

    latest = valid_groups[-1].drop(columns=["_contiguous_group"])
    return latest.reset_index(drop=True)


def _history_tuple_from_window(df_window: pd.DataFrame, source: str) -> Tuple[List[float], str, str, str]:
    df_window = df_window.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    ws = pd.Timestamp(df_window[TIMESTAMP_COL].iloc[0])
    we = pd.Timestamp(df_window[TIMESTAMP_COL].iloc[-1]) + pd.Timedelta(hours=1)

    return (
        df_window[CONSUMPTION_COL].astype(float).tolist(),
        source,
        ws.isoformat(),
        we.isoformat(),
    )


# ============================================================
# Public API
# ============================================================

def get_history_from_csv(
    target_date: str,
    history_hours: int = DEFAULT_HISTORY_HOURS,
    fallback_to_latest: bool = True,
) -> Tuple[List[float], str, str, str]:
    """
    Return previous hourly consumption history from history_store.csv.

    Parameters
    ----------
    target_date:
        Date string, usually 'YYYY-MM-DD'. The function uses target_date 00:00
        as the forecast cutoff.

    history_hours:
        Number of previous hourly values to return. Default is 168.

    fallback_to_latest:
        If True, use the latest contiguous history_hours from the CSV when
        the exact target window is unavailable or incomplete.

    Returns
    -------
    Tuple:
        history_values, history_source, window_start_iso, window_end_iso
    """
    history_hours = int(history_hours)
    if history_hours <= 0:
        raise ValueError("history_hours must be > 0")

    df = _load_history_csv()

    window_end = pd.Timestamp(target_date).normalize()
    window_start = window_end - pd.Timedelta(hours=history_hours)

    w = df[(df[TIMESTAMP_COL] >= window_start) & (df[TIMESTAMP_COL] < window_end)].copy()
    w = w.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    if _is_complete_hourly_window(w, window_start, window_end):
        return _history_tuple_from_window(w, "csv_window")

    if fallback_to_latest:
        latest = _latest_contiguous_window(df, history_hours)
        return _history_tuple_from_window(latest, "csv_fallback_latest_contiguous")

    missing_examples = _format_missing_hours(w, window_start, window_end)
    raise ValueError(
        f"Not enough contiguous history for requested window [{window_start}, {window_end}). "
        f"Expected {history_hours} hourly rows, found {len(w)}. "
        f"Missing examples: {missing_examples}"
    )


def get_history_dataframe_from_csv(
    target_date: str,
    history_hours: int = DEFAULT_HISTORY_HOURS,
    fallback_to_latest: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Return history as a DataFrame plus metadata.

    Useful for API/inference code that wants timestamps and consumption values
    together instead of only a list.
    """
    history, source, window_start, window_end = get_history_from_csv(
        target_date=target_date,
        history_hours=history_hours,
        fallback_to_latest=fallback_to_latest,
    )

    start = pd.Timestamp(window_start)
    timestamps = pd.date_range(start=start, periods=len(history), freq="h")

    df = pd.DataFrame({
        TIMESTAMP_COL: timestamps,
        CONSUMPTION_COL: history,
    })

    meta = {
        "history_source": source,
        "window_start_iso": window_start,
        "window_end_iso": window_end,
        "history_hours": str(history_hours),
    }

    return df, meta


def history_store_info() -> dict:
    """Return file/status information for UI/debug/API health endpoints."""
    info = {
        "history_csv_path": str(HISTORY_CSV_PATH),
        "stores_dir": str(STORES_DIR),
        "default_history_hours": DEFAULT_HISTORY_HOURS,
        "exists": HISTORY_CSV_PATH.exists(),
    }

    if not HISTORY_CSV_PATH.exists():
        return info

    try:
        df = _load_history_csv()
        info.update({
            "rows": int(len(df)),
            "min_timestamp": str(df[TIMESTAMP_COL].min()) if len(df) else None,
            "max_timestamp": str(df[TIMESTAMP_COL].max()) if len(df) else None,
            "can_supply_default_history": bool(len(df) >= DEFAULT_HISTORY_HOURS),
        })

        if len(df) >= 2:
            diffs = df[TIMESTAMP_COL].diff().dropna()
            info.update({
                "median_step_hours": float(diffs.median() / pd.Timedelta(hours=1)),
                "non_hourly_gaps": int((diffs != pd.Timedelta(hours=1)).sum()),
            })
        else:
            info.update({
                "median_step_hours": None,
                "non_hourly_gaps": None,
            })

    except Exception as exc:
        info.update({
            "load_error": str(exc),
        })

    return info
