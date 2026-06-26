# C:/Plegma_Programming/src/history_store.py
# ---------------------------------------------------------
# CSV-only history store for PLEGMA API/UI
# ---------------------------------------------------------
#
# Expected CSV default path:
#   C:/Plegma_Programming/processed/stores/history_store.csv
#
# Minimal columns:
#   - timestamp       hourly, parseable datetime
#   - consumption_Wh  numeric hourly consumption in Wh
#
# Final PLEGMA UI/API design:
#   - The history CSV represents ONE home only.
#   - The user/API does NOT need to provide home_id.
#   - A selected CSV path can be passed directly from the UI.
#   - With-history models require recent hourly consumption history.
#   - Default N = 168 hours.
#
# Behavior:
#   1. Try exact requested window:
#      [target_date 00:00 - history_hours, target_date 00:00)
#   2. If the exact window is incomplete and fallback_to_latest=True:
#      return the latest contiguous history_hours available in the CSV.
#   3. If no valid contiguous window exists, raise a clear error.
#
# Notes:
#   - This file does not infer or require home_id.
#   - Optional home_id column is ignored for compatibility.
#   - Timestamps are normalized to hourly timestamps.
#   - Duplicate timestamps are resolved by keeping the last row.
#   - Consumption is clipped at zero.
# ---------------------------------------------------------

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd


# ============================================================
# Paths / defaults
# ============================================================

BASE_DIR = Path(os.getenv("PLEGMA_BASE_DIR", "C:/Plegma_Programming"))
PROCESSED_DIR = Path(os.getenv("PLEGMA_PROCESSED_DIR", str(BASE_DIR / "processed")))
STORES_DIR = Path(os.getenv("PLEGMA_STORES_DIR", str(PROCESSED_DIR / "stores")))
SELECTED_HISTORY_DIR = Path(
    os.getenv("PLEGMA_SELECTED_HISTORY_DIR", str(STORES_DIR / "selected_history_files"))
)

HISTORY_CSV_PATH = Path(os.getenv("PLEGMA_HISTORY_CSV", str(STORES_DIR / "history_store.csv")))
DEFAULT_HISTORY_HOURS = int(os.getenv("PLEGMA_DEFAULT_HISTORY_HOURS", "168"))

TIMESTAMP_COL = "timestamp"
CONSUMPTION_COL = "consumption_Wh"

TIMESTAMP_ALIASES = ["time", "datetime", "date_time", "date", "ds"]
CONSUMPTION_ALIASES = ["consumption", "consumption_wh", "energy_Wh", "energy_wh", "Wh", "wh", "value", "y"]


# ============================================================
# Internal helpers
# ============================================================

def _resolve_history_path(history_csv_path: Optional[Union[str, Path]] = None) -> Path:
    """Return the selected history CSV path or the default path."""
    if history_csv_path is None or str(history_csv_path).strip() == "":
        return HISTORY_CSV_PATH
    return Path(history_csv_path)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize expected timestamp and consumption column names."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Case-insensitive lookup for uploaded/user CSVs.
    lower_map = {str(c).lower(): c for c in df.columns}

    if TIMESTAMP_COL not in df.columns:
        for alt in TIMESTAMP_ALIASES:
            if alt in df.columns:
                df = df.rename(columns={alt: TIMESTAMP_COL})
                break
            if alt.lower() in lower_map:
                df = df.rename(columns={lower_map[alt.lower()]: TIMESTAMP_COL})
                break

    if TIMESTAMP_COL not in df.columns:
        raise KeyError(
            "history CSV must contain a 'timestamp' column "
            "or one of these alternatives: " + ", ".join(TIMESTAMP_ALIASES)
        )

    if CONSUMPTION_COL not in df.columns:
        for alt in CONSUMPTION_ALIASES:
            if alt in df.columns:
                df = df.rename(columns={alt: CONSUMPTION_COL})
                break
            if alt.lower() in lower_map:
                df = df.rename(columns={lower_map[alt.lower()]: CONSUMPTION_COL})
                break

    if CONSUMPTION_COL not in df.columns:
        raise KeyError(
            "history CSV must contain 'consumption_Wh' "
            "or one of these alternatives: " + ", ".join(CONSUMPTION_ALIASES)
        )

    return df


def load_history_csv(history_csv_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Load, normalize and clean a single-home history CSV.

    Returns a DataFrame with exactly:
      - timestamp
      - consumption_Wh
    """
    path = _resolve_history_path(history_csv_path)

    if not path.exists():
        raise FileNotFoundError(
            f"History CSV not found: {path}\n"
            f"Pass a valid selected CSV path from the UI or put history_store.csv under: {STORES_DIR}"
        )

    df = pd.read_csv(path)
    df = _normalize_columns(df)

    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
    df[CONSUMPTION_COL] = pd.to_numeric(df[CONSUMPTION_COL], errors="coerce")

    df = df.dropna(subset=[TIMESTAMP_COL, CONSUMPTION_COL]).copy()

    # Same policy as the final IDEAL history store:
    # normalize to hourly timestamps and keep the last row if duplicates exist.
    df[TIMESTAMP_COL] = df[TIMESTAMP_COL].dt.floor("h")

    df = df.sort_values(TIMESTAMP_COL)
    df = df.drop_duplicates(subset=[TIMESTAMP_COL], keep="last")

    # Consumption should not be negative in this project.
    df[CONSUMPTION_COL] = df[CONSUMPTION_COL].clip(lower=0)

    return df[[TIMESTAMP_COL, CONSUMPTION_COL]].reset_index(drop=True)


# Backward-compatible private name used by older code.
def _load_history_csv(history_csv_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    return load_history_csv(history_csv_path=history_csv_path)


def _expected_hourly_index(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Return hourly timestamps in [start, end)."""
    return pd.date_range(start=start, end=end - pd.Timedelta(hours=1), freq="h")


def _is_complete_hourly_window(df_window: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    expected = _expected_hourly_index(start, end)

    if len(df_window) != len(expected):
        return False

    actual = pd.DatetimeIndex(df_window[TIMESTAMP_COL].sort_values().values)
    return actual.equals(expected)


def _format_missing_hours(
    df_window: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    max_show: int = 10,
) -> List[str]:
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
            f"No contiguous hourly history window with {history_hours} rows was found."
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


def _resolve_target_cutoff(
    target_date: Optional[Union[str, pd.Timestamp]] = None,
    target_ts: Optional[Union[str, pd.Timestamp]] = None,
    normalize_to_midnight: bool = True,
) -> pd.Timestamp:
    """Resolve target_date/target_ts into a forecast cutoff timestamp."""
    raw = target_date if target_date is not None else target_ts
    if raw is None:
        raise ValueError("Either target_date or target_ts must be provided.")

    ts = pd.Timestamp(raw)
    if pd.isna(ts):
        raise ValueError(f"Invalid target date/timestamp: {raw}")

    return ts.normalize() if normalize_to_midnight else ts


# ============================================================
# Public API
# ============================================================

def get_history_from_csv(
    target_date: Optional[Union[str, pd.Timestamp]] = None,
    history_hours: int = DEFAULT_HISTORY_HOURS,
    fallback_to_latest: bool = True,
    history_csv_path: Optional[Union[str, Path]] = None,
    target_ts: Optional[Union[str, pd.Timestamp]] = None,
    normalize_to_midnight: bool = True,
) -> Tuple[List[float], str, str, str]:
    """
    Return previous hourly consumption history from a selected single-home CSV.

    Parameters
    ----------
    target_date:
        Date string, usually 'YYYY-MM-DD'. By default the function uses
        target_date 00:00 as the forecast cutoff.

    history_hours:
        Number of previous hourly values to return. Default is 168.

    fallback_to_latest:
        If True, use the latest contiguous history_hours from the CSV when
        the exact target window is unavailable or incomplete.

    history_csv_path:
        Optional selected CSV path from the UI/API. If omitted, the default
        PLEGMA history_store.csv path is used.

    target_ts:
        Backward-compatible alias for older PLEGMA code. If target_date is
        not provided, target_ts is used.

    normalize_to_midnight:
        True for normal day-ahead UI behavior. Set False only if a caller
        intentionally wants an arbitrary hourly cutoff.

    Returns
    -------
    Tuple:
        history_values, history_source, window_start_iso, window_end_iso
    """
    history_hours = int(history_hours)
    if history_hours <= 0:
        raise ValueError("history_hours must be > 0")

    df = load_history_csv(history_csv_path=history_csv_path)
    if df.empty:
        raise ValueError("History CSV contains no valid rows after parsing.")

    window_end = _resolve_target_cutoff(
        target_date=target_date,
        target_ts=target_ts,
        normalize_to_midnight=normalize_to_midnight,
    )
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
    target_date: Optional[Union[str, pd.Timestamp]] = None,
    history_hours: int = DEFAULT_HISTORY_HOURS,
    fallback_to_latest: bool = True,
    history_csv_path: Optional[Union[str, Path]] = None,
    target_ts: Optional[Union[str, pd.Timestamp]] = None,
    normalize_to_midnight: bool = True,
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
        history_csv_path=history_csv_path,
        target_ts=target_ts,
        normalize_to_midnight=normalize_to_midnight,
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


def analyze_history_csv(history_csv_path: Optional[Union[str, Path]] = None) -> dict:
    """Return a compact history report for UI display/debugging."""
    path = _resolve_history_path(history_csv_path)
    df = load_history_csv(history_csv_path=path)

    base = {
        "history_csv_path": str(path),
        "default_history_hours": DEFAULT_HISTORY_HOURS,
        "single_home_csv": True,
    }

    if df.empty:
        return {**base, "rows": 0, "status": "empty"}

    start = pd.Timestamp(df[TIMESTAMP_COL].min())
    end = pd.Timestamp(df[TIMESTAMP_COL].max())
    expected_hours = int(((end - start).total_seconds() / 3600) + 1) if end >= start else 0
    actual_hours = int(len(df))
    coverage = float(actual_hours / expected_hours) if expected_hours > 0 else 0.0

    diffs = df[TIMESTAMP_COL].diff().dropna()
    non_hourly_gaps = int((diffs != pd.Timedelta(hours=1)).sum()) if len(diffs) else 0
    median_step_hours = float(diffs.median() / pd.Timedelta(hours=1)) if len(diffs) else None

    daily = df.set_index(TIMESTAMP_COL)[CONSUMPTION_COL].resample("1D").sum()
    mean_daily_kwh = float(daily.mean() / 1000.0) if len(daily) else 0.0

    try:
        latest = _latest_contiguous_window(df, DEFAULT_HISTORY_HOURS)
        can_supply_default_history = True
        latest_contiguous_start = pd.Timestamp(latest[TIMESTAMP_COL].iloc[0]).isoformat()
        latest_contiguous_end = (
            pd.Timestamp(latest[TIMESTAMP_COL].iloc[-1]) + pd.Timedelta(hours=1)
        ).isoformat()
    except Exception:
        can_supply_default_history = False
        latest_contiguous_start = None
        latest_contiguous_end = None

    return {
        **base,
        "rows": actual_hours,
        "status": "ok",
        "min_timestamp": start.isoformat(),
        "max_timestamp": end.isoformat(),
        "expected_hours": expected_hours,
        "coverage": coverage,
        "can_supply_default_history": can_supply_default_history,
        "latest_contiguous_window_start": latest_contiguous_start,
        "latest_contiguous_window_end": latest_contiguous_end,
        "median_step_hours": median_step_hours,
        "non_hourly_gaps": non_hourly_gaps,
        "mean_consumption_Wh": float(df[CONSUMPTION_COL].mean()),
        "median_consumption_Wh": float(df[CONSUMPTION_COL].median()),
        "mean_daily_consumption_kWh": mean_daily_kwh,
        "min_consumption_Wh": float(df[CONSUMPTION_COL].min()),
        "max_consumption_Wh": float(df[CONSUMPTION_COL].max()),
    }


def copy_selected_history_file(
    source_csv_path: Union[str, Path],
    selected_name: Optional[str] = None,
) -> str:
    """
    Copy a user-selected history CSV into the managed selected-history folder.

    The UI can call this after upload/file selection and then pass the returned
    path to get_history_from_csv(..., history_csv_path=returned_path).
    """
    src = Path(source_csv_path)
    if not src.exists():
        raise FileNotFoundError(f"Selected history CSV not found: {src}")

    SELECTED_HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    safe_name = selected_name or src.name
    safe_name = Path(safe_name).name
    if not safe_name.lower().endswith(".csv"):
        safe_name += ".csv"

    dst = SELECTED_HISTORY_DIR / safe_name
    shutil.copy2(src, dst)
    return str(dst)


def history_store_info(history_csv_path: Optional[Union[str, Path]] = None) -> dict:
    """Return file/status information for UI/debug/API health endpoints."""
    path = _resolve_history_path(history_csv_path)
    info = {
        "history_csv_path": str(path),
        "default_history_csv_path": str(HISTORY_CSV_PATH),
        "stores_dir": str(STORES_DIR),
        "selected_history_dir": str(SELECTED_HISTORY_DIR),
        "default_history_hours": DEFAULT_HISTORY_HOURS,
        "exists": path.exists(),
        "single_home_csv": True,
        "required_columns": [TIMESTAMP_COL, CONSUMPTION_COL],
        "optional_columns": ["home_id"],
    }

    if not path.exists():
        return info

    try:
        info.update(analyze_history_csv(history_csv_path=path))
    except Exception as exc:
        info.update({"load_error": str(exc)})

    return info


# ============================================================
# Quick manual test
# ============================================================

if __name__ == "__main__":
    print(history_store_info())

    try:
        report = analyze_history_csv()
        print("history report:", report)

        vals, source, ws, we = get_history_from_csv(
            target_date="2026-10-15",
            history_hours=DEFAULT_HISTORY_HOURS,
            fallback_to_latest=True,
        )

        print("history_source:", source)
        print("window_start  :", ws)
        print("window_end    :", we)
        print("len(history)  :", len(vals))
        print("last 5 values :", vals[-5:])
    except Exception as exc:
        print("Manual test failed:", exc)
