from __future__ import annotations
import numpy as np
import pandas as pd

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

def monthly_energy_from_hourly(series: pd.Series) -> pd.DataFrame:
    """
    Assumes typical-year ordering. Splits into 12 equal 30/31-day blocks is NOT correct.
    For a TMY, PVGIS already provides a constructed year; the simplest consistent approach:
    compute month index using sequential hours / 12 blocks (diagnostic).
    If you want real month lengths, you need timestamps. PVGIS TMY often includes 'time(UTC)'.
    """
    s = series.to_numpy(dtype=float)
    n = len(s)
    # naive month mapping: 12 equal chunks
    month = (np.arange(n) * 12 // n) + 1
    df = pd.DataFrame({"month": month, "value": s})
    out = df.groupby("month")["value"].sum().reset_index()
    out["month_name"] = out["month"].apply(lambda m: MONTH_NAMES[m-1])
    return out[["month","month_name","value"]]

def average_daily_profile(series: pd.Series) -> pd.Series:
    s = series.to_numpy(dtype=float)
    if len(s) % 24 != 0:
        raise ValueError("Series length must be multiple of 24.")
    d = s.reshape(-1, 24).mean(axis=0)
    return pd.Series(d, index=list(range(24)), name="avg_day")
