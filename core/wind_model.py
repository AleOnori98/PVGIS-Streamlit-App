from __future__ import annotations
import math
import numpy as np
import pandas as pd

def rotor_wind_speed(ws10: np.ndarray, alpha: float, hub_height_m: float) -> np.ndarray:
    return ws10 * (hub_height_m / 10.0) ** alpha

def air_density(hub_height_m: float, t2m_C: np.ndarray) -> np.ndarray:
    DT = -0.0066 * (hub_height_m - 2.0)
    P = 101.29 - 0.011837 * hub_height_m + 4.793e-7 * hub_height_m**2  # kPa
    MM = 28.96
    R = 8.314
    R_molar = R / MM
    rho = P / (R_molar * (t2m_C + 273.15 + DT))
    return rho

def turbine_power_from_curve(
    ws_curve: np.ndarray,
    p_curve_kw: np.ndarray,
    ws_rotor: np.ndarray,
    drivetrain_eff: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate power curve; return power [kW] and Cp [-] estimate (optional diagnostic).
    Cp is approximate because we do not apply air density correction to the curve here.
    """
    p_kw_raw = np.interp(ws_rotor, ws_curve, p_curve_kw, left=0.0, right=0.0)
    p_kw = p_kw_raw * drivetrain_eff

    # Cp diagnostic (uses ideal wind power)
    # If you want Cp: require rotor area + rho + convert to W
    cp = np.full_like(p_kw, np.nan, dtype=float)
    return p_kw, cp

def wind_from_tmy(
    tmy_df: pd.DataFrame,
    *,
    hub_height_m: float,
    rotor_diam_m: float,
    drivetrain_eff: float,
    surface_roughness_m: float,
    ws_curve: np.ndarray,
    p_curve_kw: np.ndarray,
) -> pd.DataFrame:
    """
    Compute wind turbine hourly power [kW] from PVGIS TMY wind speed and uploaded curve.
    Returns DataFrame indexed by period (1..8760):
      - wt_power_kw
      - wt_power_kw_per_kw_rated (if rated is max of curve)
    """
    df = tmy_df.copy()
    ws10 = df["WS10m"].to_numpy(dtype=float)
    t2m = df["T2m"].to_numpy(dtype=float)

    # power-law exponent alpha from z0 (your existing correlation)
    z0 = float(surface_roughness_m)
    if z0 <= 0:
        raise ValueError("Surface roughness must be > 0.")
    alpha = 0.096 * math.log10(z0) + 0.16 * (math.log10(z0)) ** 2 + 0.24

    ws_rotor = rotor_wind_speed(ws10, alpha, hub_height_m)

    # rated power from curve
    rated_kw = float(np.nanmax(p_curve_kw)) if len(p_curve_kw) else 0.0
    p_kw, _cp = turbine_power_from_curve(ws_curve, p_curve_kw, ws_rotor, drivetrain_eff)

    out = pd.DataFrame(
        {
            "period": np.arange(1, len(p_kw) + 1),
            "wt_power_kw": p_kw,
            "wt_power_per_kw_rated": p_kw / max(rated_kw, 1e-9),
            "ws10m": ws10,
            "ws_rotor": ws_rotor,
        }
    ).set_index("period")
    return out
