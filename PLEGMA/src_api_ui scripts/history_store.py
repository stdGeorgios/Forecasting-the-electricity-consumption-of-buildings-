# C:/Plegma_Programming/src/history_store.py
# ---------------------------------------------------------
# CSV-only history store for PLEGMA demo
#
# Expected CSV (default):
# C:/Plegma_Programming/stores/history_store.csv
#
# Minimal columns:
# - timestamp
# - consumption_Wh
#
# Optional:
# - home_id
#
# Behavior:
# - For a target_date/time, returns the previous N hours:
#   [target_ts - N hours, target_ts)
# - If missing hours -> fallback to last available N hours in the CSV.
# ---------------------------------------------------------

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd


# =========================
# Paths
# =========================
BASE_DIR = Path(os.getenv("PLEGMA_BASE_DIR", "C:/Plegma_Programming"))
STORES_DIR = Path(os.getenv("PLEGMA_STORES_DIR", str(BASE_DIR / "stores")))

# Default fixed CSV name for demo history
HISTORY_CSV_PATH = Path(
    os.getenv(
        "PLEGMA_HISTORY_CSV",
        str(STORES_DIR / "history_store.csv")
    )
)


# =========================
# Load & normalize
# =========================
def _load_history_csv() -> pd.DataFrame:
    if not HISTORY_CSV_PATH.exists():
        raise FileNotFoundError(
            f"History CSV not found: {HISTORY_CSV_PATH}\n"
            f"Put a file named 'history_store.csv' under: {STORES_DIR}"
        )

    df = pd.read_csv(HISTORY_CSV_PATH)

    # Normalize timestamp column
    if "timestamp" not in df.columns:
        for alt in ["time", "datetime", "date_time"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "timestamp"})
                break
    if "timestamp" not in df.columns:
        raise KeyError("history CSV must contain a 'timestamp' column (or time/datetime/date_time).")

    # Normalize consumption column
    if "consumption_Wh" not in df.columns:
        for alt in ["consumption", "energy_Wh", "Wh", "value"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "consumption_Wh"})
                break
    if "consumption_Wh" not in df.columns:
        raise KeyError("history CSV must contain 'consumption_Wh' (or a known alternative column).")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["consumption_Wh"] = pd.to_numeric(df["consumption_Wh"], errors="coerce")

    if "home_id" in df.columns:
        df["home_id"] = df["home_id"].astype(str)

    df = df.dropna(subset=["timestamp", "consumption_Wh"]).copy()

    # Keep unique hourly timestamps
    df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset=["timestamp"], keep="last")

    return df


# =========================
# Public API
# =========================
def get_history_from_csv(
    target_ts: str,
    history_hours: int = 168,
    fallback_to_latest: bool = True,
) -> Tuple[List[float], str, str, str]:
    """
    Returns:
      history(list[Wh]), history_source, window_start_iso, window_end_iso

    Main window:
      [target_ts - history_hours, target_ts)

    Example:
      target_ts='2026-10-15 00:00:00', history_hours=168

    Fallback:
      last available history_hours in CSV.
    """
    history_hours = int(history_hours)
    if history_hours <= 0:
        raise ValueError("history_hours must be > 0")

    df = _load_history_csv()

    window_end = pd.Timestamp(target_ts)
    window_start = window_end - pd.Timedelta(hours=history_hours)

    w = df[(df["timestamp"] >= window_start) & (df["timestamp"] < window_end)].copy()

    if len(w) >= history_hours:
        w = w.sort_values("timestamp").tail(history_hours)
        return (
            w["consumption_Wh"].astype(float).tolist(),
            "csv_window",
            window_start.isoformat(),
            window_end.isoformat(),
        )

    if fallback_to_latest:
        if len(df) < history_hours:
            raise ValueError(
                f"CSV history too short: only {len(df)} rows, need >= {history_hours}."
            )

        tail = df.sort_values("timestamp").tail(history_hours).copy()
        ws = pd.Timestamp(tail["timestamp"].iloc[0])
        we = pd.Timestamp(tail["timestamp"].iloc[-1]) + pd.Timedelta(hours=1)

        return (
            tail["consumption_Wh"].astype(float).tolist(),
            "csv_fallback_latest",
            ws.isoformat(),
            we.isoformat(),
        )

    raise ValueError(
        f"Not enough history for requested window [{window_start}, {window_end}). "
        f"Found only {len(w)} rows."
    )


def history_store_info() -> dict:
    """Small helper for UI/info/debug."""
    return {
        "base_dir": str(BASE_DIR),
        "stores_dir": str(STORES_DIR),
        "history_csv_path": str(HISTORY_CSV_PATH),
    }


# =========================
# Quick manual test
# =========================
if __name__ == "__main__":
    print(history_store_info())

    vals, source, ws, we = get_history_from_csv(
        target_ts="2026-10-15 00:00:00",
        history_hours=168,
        fallback_to_latest=True,
    )

    print("history_source:", source)
    print("window_start  :", ws)
    print("window_end    :", we)
    print("len(history)  :", len(vals))
    print("last 5 values :", vals[-5:])