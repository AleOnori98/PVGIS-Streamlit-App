from __future__ import annotations
from io import BytesIO
import pandas as pd

from typing import Union
from pathlib import Path

def read_power_curve(source: Union[object, str, Path]) -> tuple[pd.Series, pd.Series]:
    """
    Accept:
      - Streamlit UploadedFile (CSV or Excel)
      - local path (CSV or Excel)

    Expected format (minimum):
      Column 1: wind speed [m/s]
      Column 2: power [kW]

    Returns:
      (ws_mps, p_kw) as float Series, cleaned + sorted by wind speed.
    """
    # --- local path
    if isinstance(source, (str, Path)):
        path = Path(source)
        name = path.name.lower()
        if name.endswith(".csv"):
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path, sheet_name=0)
    else:
        # --- Streamlit UploadedFile-like
        name = getattr(source, "name", "").lower()
        if name.endswith(".csv"):
            df = pd.read_csv(source)
        else:
            df = pd.read_excel(BytesIO(source.getvalue()), sheet_name=0)

    if df.shape[1] < 2:
        raise ValueError("Power curve must have >= 2 columns: wind speed, power.")

    ws = df.iloc[:, 0].astype(float)
    p = df.iloc[:, 1].astype(float)

    df2 = pd.DataFrame({"ws": ws, "p_kw": p}).dropna().sort_values("ws")

    # Optional sanity checks
    if (df2["ws"] < 0).any():
        raise ValueError("Wind speed values must be >= 0.")
    if (df2["p_kw"] < 0).any():
        raise ValueError("Power values must be >= 0.")

    return df2["ws"], df2["p_kw"]
