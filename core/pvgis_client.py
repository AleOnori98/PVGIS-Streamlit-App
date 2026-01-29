from __future__ import annotations
import requests
import pandas as pd

from config.settings import PVGIS_TMY_URL

def _data_download(url: str) -> dict:
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        raise ValueError(f"PVGIS request failed {resp.status_code}: {resp.text}")
    return resp.json()

def get_tmy_hourly(lat: float, lon: float) -> pd.DataFrame:
    """
    Download PVGIS typical meteorological year (TMY) hourly data as a DataFrame.

    Output columns typically include:
      - G(h)   global irradiance [Wh/m²] (hourly)
      - Gd(h)  diffuse irradiance [Wh/m²]
      - T2m    air temperature [°C]
      - WS10m  wind speed at 10m [m/s]
      - WD10m  wind direction [deg]
    """
    url = f"{PVGIS_TMY_URL}lat={lat}&lon={lon}&outputformat=json"
    js = _data_download(url)
    hourly = js["outputs"]["tmy_hourly"]
    df = pd.DataFrame(hourly)
    if df.empty:
        raise ValueError("PVGIS returned empty TMY.")
    return df
