from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PVGIS_TMY_URL = "https://re.jrc.ec.europa.eu/api/tmy?"

# ---------- defaults ----------
DEFAULT_LAT = -1.2915
DEFAULT_LON = 36.8165
DEFAULT_TZ_OFFSET = 2  # hours

# PV defaults (kWp-based)
DEFAULT_PV_KWP = 1.0
DEFAULT_TILT_DEG = 20.0
DEFAULT_AZIMUTH_DEG = 0.0
DEFAULT_ALBEDO = 0.2
DEFAULT_KT_PCT_PER_C = -0.4
DEFAULT_NMOT_C = 45.0
DEFAULT_T_NMOT_C = 20.0
DEFAULT_G_NMOT_WM2 = 800.0

# Wind defaults
DEFAULT_HUB_HEIGHT_M = 37.0
DEFAULT_ROTOR_DIAM_M = 21.0
DEFAULT_DRIVETRAIN_EFF = 0.90
DEFAULT_SURFACE_ROUGHNESS_M = 0.10  # z0

@dataclass(frozen=True)
class PathManager:
    ROOT: Path = Path(__file__).resolve().parents[1]

    # UI / assets
    ASSETS: Path = ROOT / "assets"

    # Internal turbine library
    DATA: Path = ROOT / "data"
    TURBINE_LIBRARY_DIR: Path = DATA / "wind_turbines"
    TURBINE_INDEX_CSV: Path = TURBINE_LIBRARY_DIR / "index.csv"
