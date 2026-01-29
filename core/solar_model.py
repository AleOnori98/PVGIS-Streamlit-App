from __future__ import annotations
import math
import numpy as np
import pandas as pd

def _I_tilt_f(beta: float, I_tot: float, I_diff: float, ro_g: float, theta_z: float, theta_i: float) -> float:
    # I_tot/I_diff are kWh/m² (hourly), angles in radians
    denom = max(math.cos(theta_z), 1e-6)
    I_tilt_iso = (
        I_diff * (1 + math.cos(beta)) / 2
        + I_tot * ro_g * (1 - math.cos(beta)) / 2
        + (I_tot - I_diff) * math.cos(theta_i) / denom
    )
    return max(I_tilt_iso, 0.0)

def _hourly_solar(H_lst, I_diff_lst, lat: float, lon: float, day_year: int, tilt: float, azimuth: float, albedo: float):
    B = (day_year - 1) * 2 * math.pi / 365
    delta = math.radians(23.45 * (math.sin(math.radians((day_year + 284) * 360 / 365))))
    phi = math.radians(lat)
    beta = math.radians(tilt)
    gamma = math.radians(azimuth)

    EoT = 229.2 * (
        0.000075
        + 0.001868 * math.cos(B)
        - 0.032077 * math.sin(B)
        - 0.014615 * math.cos(2 * B)
        - 0.04089 * math.sin(2 * B)
    )

    I_tilt = []
    ro_g = albedo

    for hour_day in range(24):
        utc_time = hour_day
        t_s = utc_time + 4 * (lon) / 60 + EoT / 60
        omega = math.radians(15 * (t_s - 12))

        I_tot = float(H_lst[hour_day])
        I_diff = float(I_diff_lst[hour_day])
        if I_tot - I_diff < 0:
            I_tot = I_diff

        theta_z = abs(math.acos(math.cos(phi) * math.cos(delta) * math.cos(omega) + math.sin(phi) * math.sin(delta)))

        denom = max(math.sin(theta_z) * math.cos(phi), 1e-6)
        gamma_s = np.sign(omega) * abs(math.acos((math.cos(theta_z) * math.sin(phi) - math.sin(delta)) / denom))

        theta_i = math.acos(
            math.cos(theta_z) * math.cos(beta)
            + math.sin(theta_z) * math.sin(beta) * math.cos(gamma_s - gamma)
        )
        if math.cos(theta_z) < 0.1:
            theta_i = math.pi / 2

        I_tilt.append(_I_tilt_f(beta, I_tot, I_diff, ro_g, theta_z, theta_i))

    return I_tilt

def pv_from_tmy(
    tmy_df: pd.DataFrame,
    *,
    lat: float,
    lon: float,
    nom_power_kwp: float,
    tilt_deg: float,
    azimuth_deg: float,
    albedo: float,
    k_T_pct_per_C: float,
    NMOT_C: float,
    T_NMOT_C: float,
    G_NMOT_Wm2: float,
) -> pd.DataFrame:
    """
    Compute PV hourly energy from PVGIS TMY.
    Returns a DataFrame indexed by period (1..8760) with:
      - pv_kwh
      - pv_kwh_per_kwp
      - t_cell_C (optional diagnostic)
      - g_tilt_kwh_m2 (tilted irradiance)
    """
    df = tmy_df.copy()
    n = len(df)
    if n % 24 != 0:
        raise ValueError(f"TMY length must be multiple of 24, got {n}.")
    n_days = n // 24

    # Convert PVGIS Wh/m² to kWh/m² per hour
    GHI = (df["G(h)"].to_numpy(dtype=float) / 1000.0).reshape(n_days, 24)
    DHI = (df["Gd(h)"].to_numpy(dtype=float) / 1000.0).reshape(n_days, 24)
    T2m = df["T2m"].to_numpy(dtype=float).reshape(n_days, 24)

    I_tilt = np.zeros((n_days, 24), dtype=float)
    for d in range(n_days):
        I_tilt[d, :] = _hourly_solar(
            H_lst=GHI[d, :].tolist(),
            I_diff_lst=DHI[d, :].tolist(),
            lat=lat,
            lon=lon,
            day_year=d + 1,
            tilt=tilt_deg,
            azimuth=azimuth_deg,
            albedo=albedo,
        )

    # Cell temperature + production
    kT = k_T_pct_per_C / 100.0
    T_cell = T2m + ((NMOT_C - T_NMOT_C) / G_NMOT_Wm2) * (I_tilt * 1000.0)  # W/m²

    pv_kwh = I_tilt * nom_power_kwp * (1 + kT * (T_cell - 25.0))
    pv_kwh = np.clip(pv_kwh, 0.0, None)

    pv_kwh_flat = pv_kwh.reshape(-1)
    out = pd.DataFrame(
        {
            "period": np.arange(1, len(pv_kwh_flat) + 1),
            "pv_kwh": pv_kwh_flat,
            "pv_kwh_per_kwp": pv_kwh_flat / max(nom_power_kwp, 1e-9),
            "t_cell_C": T_cell.reshape(-1),
            "g_tilt_kwh_m2": I_tilt.reshape(-1),
        }
    ).set_index("period")

    return out
