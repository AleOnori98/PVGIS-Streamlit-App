from __future__ import annotations
import numpy as np
import pandas as pd

def as_typical_year_index(n_hours: int) -> pd.Index:
    # Periods 1..n
    return pd.Index(range(1, n_hours + 1), name="period")

def add_hour_month_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds Hour [0..23] and Month [1..12] assuming the PVGIS TMY is ordered hourly.
    """
    out = df.copy()
    idx0 = np.arange(len(out))
    out["Hour"] = idx0 % 24
    out["Month"] = (idx0 // (24)) % 12 + 1
    return out

def shift_hourly_series_to_local_time(values: np.ndarray, tz_offset_hours: int) -> np.ndarray:
    """
    Shift a 24-hour daily profile to match local time.
    Positive offset shifts the curve forward (roll right).
    """
    shift = int(round(tz_offset_hours))
    return np.roll(values, shift)
