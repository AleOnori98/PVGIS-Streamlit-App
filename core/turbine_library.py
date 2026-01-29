from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from config.settings import PathManager


@dataclass(frozen=True)
class TurbineSpec:
    turbine_id: str
    display_name: str
    curve_file: str

    hub_height_m_default: float
    rotor_diam_m: float
    rated_kw: float
    drivetrain_eff_default: float

    curve_ws: np.ndarray
    curve_p_kw: np.ndarray

    notes: str = ""


def _read_curve_csv(curve_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(curve_path)
    if df.shape[1] < 2:
        raise ValueError(f"{curve_path.name}: needs at least 2 columns (ws, p_kw).")

    ws = df.iloc[:, 0].astype(float).to_numpy()
    p = df.iloc[:, 1].astype(float).to_numpy()

    m = np.isfinite(ws) & np.isfinite(p)
    ws, p = ws[m], p[m]
    idx = np.argsort(ws)
    ws, p = ws[idx], p[idx]

    if (ws < 0).any():
        raise ValueError(f"{curve_path.name}: wind speed must be >= 0.")
    if (p < 0).any():
        raise ValueError(f"{curve_path.name}: power must be >= 0.")

    return ws, p


def load_turbine_library(library_dir: Path | None = None) -> Dict[str, TurbineSpec]:
    """
    Uses your existing index.csv schema:

      model,turbine_type,rated_power_kw,rotor_diameter_m,hub_height_m,curve_csv,(optional...)...

    Expected structure:
      <library_dir>/index.csv
      <library_dir>/curves/<curve_csv>.csv
    """
    library_dir = library_dir or PathManager.TURBINE_LIBRARY_DIR
    index_path = library_dir / "index.csv"
    curves_dir = library_dir / "curves"

    if not index_path.exists():
        return {}
    if not curves_dir.exists():
        raise FileNotFoundError(f"Curves folder not found: {curves_dir}")

    df = pd.read_csv(index_path)

    required = [
        "model",
        "turbine_type",
        "rated_power_kw",
        "rotor_diameter_m",
        "hub_height_m",
        "curve_csv",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{index_path.name} missing columns: {missing}")

    out: Dict[str, TurbineSpec] = {}

    for _, r in df.iterrows():
        model = str(r["model"]).strip()
        turbine_type = str(r["turbine_type"]).strip()

        # stable ID (no spaces, consistent)
        turbine_id = f"{model}__{turbine_type}".replace(" ", "_")

        curve_file = str(r["curve_csv"]).strip()
        curve_path = curves_dir / curve_file
        if not curve_path.exists():
            raise FileNotFoundError(f"Curve file not found for {turbine_id}: {curve_path}")

        ws, p_kw = _read_curve_csv(curve_path)

        rated_kw = float(r["rated_power_kw"])
        rotor_d = float(r["rotor_diameter_m"])
        hub_h = float(r["hub_height_m"])

        def _format_rated_kw(rated_kw: float) -> str:
            if rated_kw < 10:
                return f"{rated_kw:.1f} kW"
            return f"{rated_kw:.0f} kW"

        # you can store this in index later; for now set a reasonable default
        drivetrain_eff_default = float(r["drivetrain_eff_default"]) if "drivetrain_eff_default" in df.columns and pd.notna(r["drivetrain_eff_default"]) else 0.90

        # nice label for UI
        display_name = f"{model} ({turbine_type}) — {_format_rated_kw(rated_kw)}"

        notes_parts = []
        for col in ["source", "sheet"]:
            if col in df.columns and pd.notna(r[col]):
                notes_parts.append(f"{col}: {str(r[col]).strip()}")
        notes = " | ".join(notes_parts)

        out[turbine_id] = TurbineSpec(
            turbine_id=turbine_id,
            display_name=display_name,
            curve_file=curve_file,
            hub_height_m_default=hub_h,
            rotor_diam_m=rotor_d,
            rated_kw=rated_kw,
            drivetrain_eff_default=drivetrain_eff_default,
            curve_ws=ws,
            curve_p_kw=p_kw,
            notes=notes,
        )

    return out
