from __future__ import annotations

import time
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
import folium
import plotly.graph_objects as go

from config.settings import (
    DEFAULT_LAT, DEFAULT_LON, DEFAULT_TZ_OFFSET,
    DEFAULT_PV_KWP, DEFAULT_TILT_DEG, DEFAULT_AZIMUTH_DEG, DEFAULT_ALBEDO,
    DEFAULT_KT_PCT_PER_C, DEFAULT_NMOT_C, DEFAULT_T_NMOT_C, DEFAULT_G_NMOT_WM2,
    DEFAULT_HUB_HEIGHT_M, DEFAULT_ROTOR_DIAM_M, DEFAULT_DRIVETRAIN_EFF, DEFAULT_SURFACE_ROUGHNESS_M,
)

from config.settings import PathManager
from core.pvgis_client import get_tmy_hourly
from core.solar_model import pv_from_tmy
from core.wind_model import wind_from_tmy
from core.power_curve_io import read_power_curve
from core.time_utils import shift_hourly_series_to_local_time
from core.summaries import average_daily_profile, monthly_energy_from_hourly
from core.export_io import zip_results
from core.turbine_library import load_turbine_library
from utils.wind_resource_plots import weibull_figure, wind_rose_figure, weibull_fit_moments, wind_power_density_wm2


# -------------------------
# Streamlit config
# -------------------------
st.set_page_config(
    page_title="PVGIS Resource Assessment",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------
# Session state init
# -------------------------
SS = st.session_state
SS.setdefault("lat", DEFAULT_LAT)
SS.setdefault("lon", DEFAULT_LON)
SS.setdefault("tz", DEFAULT_TZ_OFFSET)

SS.setdefault("tmy_df", None)
SS.setdefault("pv_out", None)
SS.setdefault("wind_out", None)

SS.setdefault("power_curve_ws", None)
SS.setdefault("power_curve_pkw", None)


# -------------------------
# Caching: PVGIS download
# -------------------------
@st.cache_data(show_spinner=False)
def _cached_get_tmy(lat: float, lon: float) -> pd.DataFrame:
    return get_tmy_hourly(lat, lon)


# -------------------------
# Helpers
# -------------------------
def _map_picker(lat: float, lon: float) -> tuple[float, float]:
    m = folium.Map(location=[lat, lon], zoom_start=6)
    folium.Marker([lat, lon], tooltip="Selected location").add_to(m)
    out = st_folium(m, height=420, width=700)
    if out and out.get("last_clicked"):
        return float(out["last_clicked"]["lat"]), float(out["last_clicked"]["lng"])
    return lat, lon


def _kpi_row(label: str, value: float, unit: str = ""):
    st.metric(label, f"{value:,.2f} {unit}".strip())


# -------------------------
# Sidebar navigation
# -------------------------
with st.sidebar:
    st.title("PVGIS Resource Tool")
    page = st.radio(
        "Workflow",
        options=["1) Location", "2) Solar PV", "3) Wind Turbine", "4) Hydro Power", "5) Export"],
        index=0,
    )
    st.markdown("---")
    st.caption("Tip: run PVGIS TMY once, then PV & wind reuse the same dataset.")


# -------------------------
# Page 1: Location
# -------------------------
if page.startswith("1"):
    st.header("Set location and time zone")
    st.markdown(
        """
        This tool retrieves solar irradiation and wind data from the **PVGIS (Photovoltaic Geographical Information System)** developed by the **European Commission – Joint Research Centre (JRC)**.

        **PVGIS** is an open-access platform providing long-term, quality-controlled climate and energy resource data
        for any location worldwide.  
        👉 https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis_en

        For the selected location, the app downloads a **Typical Meteorological Year (TMY)** dataset.
        A TMY represents a *synthetic but statistically representative year* built from many years of historical weather data.
        It is commonly used in energy system modeling to estimate **average, long-term electricity production**
        without simulating multiple decades.

        Enter the site coordinates below (or adjust them using the map).
        The **UTC offset** is used to shift hourly data into **local time**, ensuring that daily production profiles
        correctly align with daylight hours and local wind patterns.
        """
    )

    # --- Inputs on one line -------------------------------------------------
    c1, c2 = st.columns([1, 1])
    with c1:
        SS.lat = st.number_input("Latitude [°]", value=float(SS.lat), format="%.6f")
        SS.tz = st.number_input("UTC offset [h]", value=int(SS.tz), step=1)
    with c2:
        SS.lon = st.number_input("Longitude [°]", value=float(SS.lon), format="%.6f")
    

    st.caption(
        "The UTC offset is used to shift **average daily profiles** into local time, "
        "so PV peaks align with real daylight hours."
    )

    # --- Map below: visualize / pick coordinates ---------------------------
    st.subheader("Visualize or pick coordinates")
    new_lat, new_lon = _map_picker(float(SS.lat), float(SS.lon))
    if (new_lat, new_lon) != (float(SS.lat), float(SS.lon)):
        SS.lat, SS.lon = new_lat, new_lon
        st.success(f"Updated to: {SS.lat:.6f}, {SS.lon:.6f}")

    st.markdown("---")

    # --- Download TMY + preview --------------------------------------------
    if st.button("Download PVGIS typical-year data", type="primary", use_container_width=True):
        with st.spinner("Downloading PVGIS TMY..."):
            t0 = time.perf_counter()
            SS.tmy_df = _cached_get_tmy(float(SS.lat), float(SS.lon))
            SS.pv_out = None
            SS.wind_out = None
            st.success(f"TMY downloaded in {time.perf_counter() - t0:.2f}s")

    if SS.tmy_df is not None:
        st.subheader("TMY preview")
        st.dataframe(SS.tmy_df.head(10), use_container_width=True)
    else:
        st.warning("No PVGIS data yet. Click **Download PVGIS typical-year data**.")


# -------------------------
# Page 2: Solar PV
# -------------------------
elif page.startswith("2"):
    st.header("Solar PV resource assessment")

    if SS.tmy_df is None:
        st.warning("First run **Step 1** to download PVGIS TMY data.")
        st.stop()

    # ------------------------------------------------------------------
    # A) Solar resource insights (from PVGIS TMY irradiance components)
    # ------------------------------------------------------------------
    st.subheader("Solar resource")

    st.markdown(
        "Before simulating PV production, you can inspect the underlying **irradiance components** "
        "from the PVGIS Typical Meteorological Year (TMY). This helps you understand *how much sun* "
        "is available and how it is distributed across the day and the year."
    )

    tmy = SS.tmy_df.copy()

    # PVGIS typical columns:
    # - G(h): Global horizontal irradiance [Wh/m²]
    # - Gd(h): Diffuse horizontal irradiance [Wh/m²]
    col_ghi = "G(h)" if "G(h)" in tmy.columns else None
    col_dhi = "Gd(h)" if "Gd(h)" in tmy.columns else None

    if col_ghi is None:
        st.warning(
            "TMY dataset does not contain `G(h)` (global horizontal irradiance). "
            "Solar resource plots are skipped."
        )
    else:
        ghi_whm2 = tmy[col_ghi].astype(float).clip(lower=0.0)
        dhi_whm2 = tmy[col_dhi].astype(float).clip(lower=0.0) if col_dhi else None

        if dhi_whm2 is None:
            # Only GHI available → show as a single component
            diffuse_whm2 = pd.Series(np.zeros(len(ghi_whm2)), index=ghi_whm2.index)
            beam_whm2 = ghi_whm2.copy()
            resource_note = "Only **GHI** is available in this TMY dataset; beam/diffuse are approximated."
        else:
            diffuse_whm2 = dhi_whm2
            beam_whm2 = (ghi_whm2 - diffuse_whm2).clip(lower=0.0)
            resource_note = "Beam is computed as **GHI − DHI** (horizontal components)."

        st.caption(resource_note)

        # Convert to kW/m² (Wh/m² per hour ≈ W/m² average over hour; /1000 → kW/m²)
        diffuse_kwm2 = diffuse_whm2 / 1000.0
        beam_kwm2 = beam_whm2 / 1000.0

        # --- Pastel palette (requested) ---
        COLOR_BEAM = "#F4A261"     # pastel orange
        COLOR_DIFFUSE = "#F6E58D"  # light pastel yellow

        # ---------- (1) Annual stacked area (hourly) ----------
        x = np.arange(1, len(beam_kwm2) + 1)

        fig_year = go.Figure()
        fig_year.add_trace(
            go.Scatter(
                x=x,
                y=diffuse_kwm2.values,
                mode="lines",
                name="Diffuse (Gd,h)",
                line=dict(color=COLOR_DIFFUSE, width=1),
                fill="tozeroy",
            )
        )
        fig_year.add_trace(
            go.Scatter(
                x=x,
                y=beam_kwm2.values,
                mode="lines",
                name="Beam (GHI−DHI)",
                line=dict(color=COLOR_BEAM, width=1),
                fill="tonexty",
            )
        )
        fig_year.update_layout(
            height=240,
            margin=dict(l=20, r=20, t=10, b=10),
            xaxis_title="Hour of typical year",
            yaxis_title="Irradiance [kW/m²]",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig_year, use_container_width=True)

        # KPIs (based on GHI)
        annual_ghi_kwh_m2 = float(ghi_whm2.sum()) / 1000.0
        mean_ghi_w_m2 = float(ghi_whm2.mean())
        peak_ghi_w_m2 = float(ghi_whm2.max())

        c1, c2, c3 = st.columns(3)
        c1.metric("Annual GHI", f"{annual_ghi_kwh_m2:,.0f} kWh/m²")
        c2.metric("Average GHI (hourly mean)", f"{mean_ghi_w_m2:,.0f} W/m²")
        c3.metric("Peak GHI (hourly max)", f"{peak_ghi_w_m2:,.0f} W/m²")

        # ---------- (2) Average daily solar resource profile (24h) ----------
        st.markdown("**Average daily solar resource profile (24h)**")

        # Build hour-of-day index (0..23) assuming TMY is hourly contiguous
        n = len(ghi_whm2)
        hour_of_day = np.arange(n) % 24

        # Mean over all days for each hour
        daily_df = pd.DataFrame(
            {
                "hour": hour_of_day,
                "diffuse_kwm2": diffuse_kwm2.values,
                "beam_kwm2": beam_kwm2.values,
            }
        )
        daily_mean = daily_df.groupby("hour")[["diffuse_kwm2", "beam_kwm2"]].mean().reindex(range(24))

        # Optional: shift to local time for the resource daily profile as well
        # (keeps consistency with PV/Wind daily profiles)
        shift = int(SS.tz) % 24
        daily_mean_local = daily_mean.copy()
        daily_mean_local.index = range(24)
        daily_mean_local = pd.DataFrame(
            {
                "diffuse_kwm2": np.roll(daily_mean["diffuse_kwm2"].values, shift),
                "beam_kwm2": np.roll(daily_mean["beam_kwm2"].values, shift),
            },
            index=range(24),
        )

        fig_day = go.Figure()
        fig_day.add_trace(
            go.Scatter(
                x=list(daily_mean_local.index),
                y=daily_mean_local["diffuse_kwm2"].values,
                mode="lines",
                name="Diffuse (Gd,h)",
                line=dict(color=COLOR_DIFFUSE, width=2),
                fill="tozeroy",
            )
        )
        fig_day.add_trace(
            go.Scatter(
                x=list(daily_mean_local.index),
                y=daily_mean_local["beam_kwm2"].values,
                mode="lines",
                name="Beam (GHI−DHI)",
                line=dict(color=COLOR_BEAM, width=2),
                fill="tonexty",
            )
        )
        fig_day.update_layout(
            height=220,
            margin=dict(l=20, r=20, t=10, b=10),
            xaxis_title="Hour of day (local time)",
            yaxis_title="Irradiance [kW/m²]",
            xaxis=dict(dtick=1),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig_day, use_container_width=True)

        st.caption("Daily resource profile is shown in **local time** using the UTC offset.")

    st.markdown("---")

    # ------------------------------------------------------------------
    # B) PV system parameters (cleaner layout)
    # ------------------------------------------------------------------
    st.subheader("PV system and performance model")

    st.markdown(
        "Define the PV system size and geometry. Tilt and azimuth affect how much of the available "
        "irradiance is captured by the panel plane."
    )

    # Nominal power on a single line
    pv_kwp = st.number_input(
        "Nominal PV size [kWp]",
        min_value=0.1,
        value=float(DEFAULT_PV_KWP),
        step=0.1,
        help="Installed DC peak capacity. Results include both total production [kWh] and normalized yield [kWh/kWp].",
    )

    # Angles in one row
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        tilt = st.slider(
            "Tilt angle [°]",
            min_value=0,
            max_value=60,
            value=int(DEFAULT_TILT_DEG),
            step=1,
            help="Panel inclination from horizontal. A good first guess is close to latitude (site-dependent).",
        )
    with c2:
        azim = st.slider(
            "Azimuth [°] (0 = south, +west, −east)",
            min_value=-180,
            max_value=180,
            value=int(DEFAULT_AZIMUTH_DEG),
            step=5,
            help="Panel facing direction. Convention here: 0 = south, +90 = west, -90 = east.",
        )
    with c3:
        albedo = st.slider(
            "Ground albedo [-]",
            min_value=0.0,
            max_value=0.9,
            value=float(DEFAULT_ALBEDO),
            step=0.05,
            help="Fraction of light reflected by the ground (affects tilted-plane irradiance). Typical: 0.15–0.30.",
        )

    st.markdown(
        "Temperature reduces PV efficiency at high cell temperature. You can keep default values "
        "unless you have module-specific datasheet parameters."
    )

    t1, t2, t3, t4 = st.columns(4)
    with t1:
        kT = st.number_input(
            "Temp. coefficient [%/°C]",
            value=float(DEFAULT_KT_PCT_PER_C),
            step=0.05,
            help="Typical range: about -0.2 to -0.5 %/°C (negative means output decreases as temperature increases).",
        )
    with t2:
        nmot = st.number_input(
            "NMOT [°C]",
            value=float(DEFAULT_NMOT_C),
            step=1.0,
            help="Nominal Module Operating Temperature (used to estimate cell temperature).",
        )
    with t3:
        t_nmot = st.number_input(
            "Ambient T at NMOT [°C]",
            value=float(DEFAULT_T_NMOT_C),
            step=1.0,
            help="Standard test reference for NMOT (often 20°C).",
        )
    with t4:
        g_nmot = st.number_input(
            "Irradiance at NMOT [W/m²]",
            value=float(DEFAULT_G_NMOT_WM2),
            step=50.0,
            help="Standard test reference for NMOT (often 800 W/m²).",
        )

    st.markdown("---")

    # ------------------------------------------------------------------
    # C) Run PV simulation
    # ------------------------------------------------------------------
    run = st.button("Run PV simulation", type="primary", use_container_width=True)
    if run:
        with st.spinner("Computing PV production..."):
            SS.pv_out = pv_from_tmy(
                SS.tmy_df,
                lat=float(SS.lat),
                lon=float(SS.lon),
                nom_power_kwp=float(pv_kwp),
                tilt_deg=float(tilt),
                azimuth_deg=float(azim),
                albedo=float(albedo),
                k_T_pct_per_C=float(kT),
                NMOT_C=float(nmot),
                T_NMOT_C=float(t_nmot),
                G_NMOT_Wm2=float(g_nmot),
            )
        st.success("PV simulation completed.")

    # ------------------------------------------------------------------
    # D) Results (monthly bar chart instead of table)
    # ------------------------------------------------------------------
    if SS.pv_out is not None:
        st.subheader("PV results")

        annual_kwh_per_kwp = float(SS.pv_out["pv_kwh_per_kwp"].sum())
        annual_kwh = float(SS.pv_out["pv_kwh"].sum())
        cap_factor = annual_kwh_per_kwp / 8760.0

        a, b, c = st.columns(3)
        a.metric("Annual PV yield", f"{annual_kwh_per_kwp:,.0f} kWh/kWp")
        b.metric("Annual energy", f"{annual_kwh:,.0f} kWh")
        c.metric("Capacity factor (approx.)", f"{cap_factor:.3f}")

        st.markdown("**Average daily PV profile**")

        # Hourly PV (per kWp), expected length = 8760 (TMY)
        pv_hourly = SS.pv_out["pv_kwh_per_kwp"]
        pv_vals = np.asarray(pv_hourly, dtype=float)

        if pv_vals.size < 24:
            st.warning("PV output is too short to compute a daily profile.")
        else:
            # Make sure we can reshape into full days (ignore leftover hours if any)
            n_days = pv_vals.size // 24
            pv_vals = pv_vals[: n_days * 24]
            pv_by_day = pv_vals.reshape(n_days, 24)

            mean_24 = pv_by_day.mean(axis=0)
            min_24 = pv_by_day.min(axis=0)
            max_24 = pv_by_day.max(axis=0)

            # Shift to local time (same logic as shift_hourly_series_to_local_time)
            shift = int(SS.tz) % 24
            mean_local = np.roll(mean_24, shift)
            min_local = np.roll(min_24, shift)
            max_local = np.roll(max_24, shift)

            hours = np.arange(24)

            fig = go.Figure()

            # Variability band: max (upper) then min (lower) with fill
            fig.add_trace(
                go.Scatter(
                    x=hours,
                    y=max_local,
                    mode="lines",
                    line=dict(width=0),
                    name="Daily max",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=hours,
                    y=min_local,
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(120,120,120,0.25)",  # gray band
                    name="Min–max range",
                    hoverinfo="skip",
                )
            )

            # Mean curve on top
            fig.add_trace(
                go.Scatter(
                    x=hours,
                    y=mean_local,
                    mode="lines",
                    line=dict(
                        width=3,
                        color="#1f77b4",  # blue 
                    ),
                    name="Average",
                )
            )

            fig.update_layout(
                height=340,  # slightly taller
                margin=dict(l=20, r=20, t=10, b=10),
                xaxis_title="Hour of day (local time)",
                yaxis_title="PV energy [kWh/kWp]",
                xaxis=dict(dtick=1),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0,
                ),
            )

            st.plotly_chart(fig, use_container_width=True)
            st.caption("Daily profile shown in **local time** (using UTC offset). Gray band shows day-to-day variability (min to max).")


        st.markdown("**Monthly PV yield (kWh/kWp)**")
        mon = monthly_energy_from_hourly(SS.pv_out["pv_kwh_per_kwp"])

        # Expect mon as a table-like object; convert to a clean 12-month Series when possible
        # Common patterns: DataFrame with index month or a column named 'value'
        mon_series = None
        if isinstance(mon, pd.Series):
            mon_series = mon
        elif isinstance(mon, pd.DataFrame):
            if mon.shape[1] == 1:
                mon_series = mon.iloc[:, 0]
            else:
                # try common names
                for cand in ["kwh_per_kwp", "value", "pv_kwh_per_kwp"]:
                    if cand in mon.columns:
                        mon_series = mon[cand]
                        break

        if mon_series is None:
            # fallback: show whatever comes back
            st.bar_chart(mon, height=260)
        else:
            # Ensure 12 ordered months if index is numeric
            try:
                mon_series = mon_series.copy()
                mon_series.index = pd.Index(mon_series.index)
                st.bar_chart(mon_series, height=260)
            except Exception:
                st.bar_chart(mon_series.values, height=260)

        # Optional: if you still want a table for export/debug
        with st.expander("Show monthly table (optional)", expanded=False):
            st.dataframe(mon, use_container_width=True)


# -------------------------
# Page 3: Wind
# -------------------------
elif page.startswith("3"):
    st.header("Wind resource & turbine assessment")

    if SS.tmy_df is None:
        st.warning("First run **Step 1** to download PVGIS TMY data.")
        st.stop()

    # ------------------------------------------------------------------
    # A) Wind resource (from TMY): speed distribution + wind rose + KPIs
    # ------------------------------------------------------------------
    st.subheader("Wind resource potential")

    ws10 = SS.tmy_df["WS10m"].to_numpy(dtype=float) if "WS10m" in SS.tmy_df.columns else None
    wd10 = SS.tmy_df["WD10m"].to_numpy(dtype=float) if "WD10m" in SS.tmy_df.columns else None

    if ws10 is None:
        st.error("TMY dataset does not include 'WS10m'. Check PVGIS response/columns.")
        st.stop()

    r1, r2 = st.columns([1, 1])
    with r1:
        fig_wb = weibull_figure(ws10)
        st.plotly_chart(fig_wb, use_container_width=True)
    with r2:
        if wd10 is not None:
            fig_wr = wind_rose_figure(wd10, ws10)
            st.plotly_chart(fig_wr, use_container_width=True)
        else:
            st.info("Wind direction (WD10m) not available in this TMY response — skipping wind rose.")

    # Resource KPIs
    mean_ws = float(np.nanmean(ws10))
    p95_ws = float(np.nanpercentile(ws10, 95))
    wpd = wind_power_density_wm2(ws10, rho=1.225)  # at 10m (rough, but useful diagnostic)

    k, c = weibull_fit_moments(ws10)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Mean wind speed (10 m)", f"{mean_ws:.2f} m/s")
    k2.metric("P95 wind speed (10 m)", f"{p95_ws:.2f} m/s")
    k3.metric("Weibull k (shape)", f"{k:.2f}" if np.isfinite(k) else "n/a")
    k4.metric("Wind power density (10 m)", f"{wpd:,.0f} W/m²")

    st.caption(
        "These plots describe the **resource** only. Turbine production depends on hub height, roughness, and the power curve."
    )

    st.markdown("---")

    # ------------------------------------------------------------------
    # B) Turbine selection + parameters (clean layout)
    # ------------------------------------------------------------------
    st.subheader("Turbine selection and performance")

    st.markdown(
        "Choose a turbine from the **built-in library** (recommended) or **upload** your own power curve. "
        "The power curve must contain two columns: **wind speed [m/s]** and **power [kW]**."
    )

    # Load turbine library
    # You should define PathManager.TURBINE_LIBRARY_DIR = Path("data/turbines") (or similar)
    try:
        turbine_lib = load_turbine_library(PathManager.TURBINE_LIBRARY_DIR)
    except Exception as e:
        turbine_lib = {}
        st.warning(f"Turbine library not available ({e}). You can still upload a power curve.")

    source = st.radio(
        "Power curve source",
        options=["Library (predefined turbines)", "Upload (CSV/Excel)"],
        horizontal=True,
    )

    # Defaults (will be overwritten by library selection if used)
    hub_h_default = float(DEFAULT_HUB_HEIGHT_M)
    rotor_d_default = float(DEFAULT_ROTOR_DIAM_M)
    eta_default = float(DEFAULT_DRIVETRAIN_EFF)

    st.markdown("**Turbine geometry & drivetrain**")

    selected_spec = None
    if source.startswith("Library"):
        if not turbine_lib:
            st.error("No turbines found in the library. Switch to Upload.")
        else:
            # Build select options
            options = list(turbine_lib.keys())
            labels = {k: turbine_lib[k].display_name for k in options}

            selected_id = st.selectbox(
                "Select turbine model",
                options=options,
                format_func=lambda k: labels.get(k, k),
            )
            selected_spec = turbine_lib[selected_id]

            # Prefill defaults
            hub_h_default = float(selected_spec.hub_height_m_default)
            rotor_d_default = float(selected_spec.rotor_diam_m)
            eta_default = float(selected_spec.drivetrain_eff_default)

            # Store curve in session for simulation
            SS.power_curve_ws = np.asarray(selected_spec.curve_ws, float)
            SS.power_curve_pkw = np.asarray(selected_spec.curve_p_kw, float)

            st.caption(
                f"Rated power (library): **{selected_spec.rated_kw:.1f} kW**  •  "
                f"Curve points: **{len(selected_spec.curve_ws)}**"
                + (f"  •  Notes: {selected_spec.notes}" if selected_spec.notes else "")
            )

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        # Inputs (user can still override library defaults if they want)
        hub_h = st.number_input(
            "Hub height [m]",
            min_value=0.0,
            value=hub_h_default,
            step=1.0,
            help="Used to extrapolate wind speed from 10 m to hub height via a power-law profile.",
        )
    with c2:
        rotor_d = st.number_input(
            "Rotor diameter [m]",
            min_value=0.0,
            value=rotor_d_default,
            step=0.5,
            help="Used for swept area and diagnostics. Production comes primarily from the power curve.",
        )
    with c3:
        eta = st.slider(
            "Drivetrain efficiency [-]",
            0.50,
            1.00,
            eta_default,
            0.01,
            help="Applies as a multiplicative factor to the electrical power from the curve.",
        )

    st.markdown("**Site / wind profile settings**")
    z0 = st.number_input(
        "Surface roughness z0 [m]",
        min_value=1e-5,
        value=float(DEFAULT_SURFACE_ROUGHNESS_M),
        step=0.01,
        format="%.5f",
        help=(
            "Controls the vertical wind shear (power-law exponent). "
            "Higher roughness → stronger shear → lower wind speed near ground and higher at hub height."
        ),
    )

    # ------------------------------------------------------------------
    # C) Power curve input section (below)
    # ------------------------------------------------------------------
    st.markdown("### Power curve")
    if source.startswith("Upload"):
        curve_file = st.file_uploader(
            "Upload power curve (CSV or Excel) — columns: wind speed [m/s], power [kW]",
            type=["csv", "xlsx", "xls"],
        )
        if curve_file is not None:
            try:
                ws, pkw = read_power_curve(curve_file)
                SS.power_curve_ws = ws.to_numpy(dtype=float)
                SS.power_curve_pkw = pkw.to_numpy(dtype=float)
                st.success(
                    f"Loaded curve with {len(ws)} points. Rated: {np.nanmax(SS.power_curve_pkw):.1f} kW"
                )
            except Exception as e:
                st.error(f"Invalid power curve: {e}")

    # Always show a tiny preview if curve is present
    if SS.power_curve_ws is not None and SS.power_curve_pkw is not None:
        pc_df = pd.DataFrame({"ws_mps": SS.power_curve_ws, "p_kw": SS.power_curve_pkw})
        with st.expander("Show power curve table (preview)", expanded=False):
            st.dataframe(pc_df, use_container_width=True)

    st.markdown("---")

    # ------------------------------------------------------------------
    # D) Run wind simulation
    # ------------------------------------------------------------------
    can_run = SS.power_curve_ws is not None and SS.power_curve_pkw is not None
    if st.button("Run wind simulation", type="primary", disabled=not can_run):
        with st.spinner("Computing wind turbine production..."):
            SS.wind_out = wind_from_tmy(
                SS.tmy_df,
                hub_height_m=float(hub_h),
                rotor_diam_m=float(rotor_d),
                drivetrain_eff=float(eta),
                surface_roughness_m=float(z0),
                ws_curve=np.asarray(SS.power_curve_ws, float),
                p_curve_kw=np.asarray(SS.power_curve_pkw, float),
            )
        st.success("Wind simulation completed.")

    # ------------------------------------------------------------------
    # E) Results
    # ------------------------------------------------------------------
    if SS.wind_out is not None:
        st.subheader("Wind turbine results")

        annual_kwh = float(SS.wind_out["wt_power_kw"].sum())  # kW per hour -> kWh
        rated_kw = float(np.nanmax(SS.power_curve_pkw)) if SS.power_curve_pkw is not None else np.nan
        annual_kwh_per_kw = annual_kwh / max(rated_kw, 1e-9)
        cap_factor = annual_kwh_per_kw / 8760.0

        a, b, c = st.columns(3)
        a.metric("Annual energy", f"{annual_kwh:,.0f} kWh")
        b.metric("Annual yield", f"{annual_kwh_per_kw:,.0f} kWh/kW_rated")
        c.metric("Capacity factor (approx.)", f"{cap_factor:.3f}")

        # Daily profile (local time)
        st.markdown("**Average daily turbine output (shifted to local time)**")
        prof = average_daily_profile(SS.wind_out["wt_power_per_kw_rated"])
        prof_local = shift_hourly_series_to_local_time(prof.values, int(SS.tz))
        st.line_chart(pd.Series(prof_local, index=range(24), name="kW/kW_rated"), height=320)
        st.caption("Daily profile shown in **local time** (using UTC offset).")

        # Monthly yield as bar chart (similar to PV section)
        st.markdown("**Monthly wind yield (kWh/kW_rated)**")
        mon = monthly_energy_from_hourly(SS.wind_out["wt_power_per_kw_rated"])

        # Make it bar-friendly
        mon_series = None
        if isinstance(mon, pd.Series):
            mon_series = mon
        elif isinstance(mon, pd.DataFrame):
            if mon.shape[1] == 1:
                mon_series = mon.iloc[:, 0]
            else:
                for cand in ["value", "kwh_per_kw", "wt_power_per_kw_rated"]:
                    if cand in mon.columns:
                        mon_series = mon[cand]
                        break

        if mon_series is None:
            st.bar_chart(mon, height=320)
        else:
            st.bar_chart(mon_series, height=320)

        with st.expander("Show monthly table (optional)", expanded=False):
            st.dataframe(mon, use_container_width=True)

# -------------------------
# Page 4: Hydro Power
# --------------------------
# (not implemented)
elif page.startswith("4"):
    st.header("Hydro Power resource assessment")
    st.info("Hydro Power assessment is not implemented yet.")


# -------------------------
# Page 5: Export
# -------------------------
else:
    st.header("Export results")

    if SS.tmy_df is None:
        st.warning("Nothing to export yet. Run Step 1 at least.")
        st.stop()

    # -------------------------
    # A) Always export raw PVGIS TMY (hourly typical year)
    # -------------------------
    tables: dict[str, pd.DataFrame] = {
        "tmy_hourly_raw": SS.tmy_df.copy(),
    }

    # -------------------------
    # B) Export PV hourly production (clean) + full table (optional)
    # -------------------------
    if SS.pv_out is not None:
        pv_df = SS.pv_out.copy()

        # Full (for traceability)
        tables["pv_hourly_full"] = pv_df

        # Clean electricity production only (what users typically need)
        pv_cols = []
        if "pv_kwh" in pv_df.columns:
            pv_cols.append("pv_kwh")
        if "pv_kwh_per_kwp" in pv_df.columns:
            pv_cols.append("pv_kwh_per_kwp")

        if pv_cols:
            tables["pv_hourly_production"] = pv_df[pv_cols].copy()

    # -------------------------
    # C) Export Wind hourly production (clean) + full table (optional)
    # -------------------------
    if SS.wind_out is not None:
        w_df = SS.wind_out.copy()

        # Full (for traceability)
        tables["wind_hourly_full"] = w_df

        # Clean electricity production only
        wind_cols = []
        if "wt_power_kw" in w_df.columns:
            # NOTE: this is power [kW] at 1h steps; numerically equals kWh per hour-step
            wind_cols.append("wt_power_kw")
        if "wt_power_per_kw_rated" in w_df.columns:
            wind_cols.append("wt_power_per_kw_rated")

        if wind_cols:
            tables["wind_hourly_production"] = w_df[wind_cols].copy()

    # -------------------------
    # D) Show summary of what will be exported
    # -------------------------
    st.write("The ZIP will contain:")

    for k in tables.keys():
        st.write(f"- `{k}.csv`")

    # -------------------------
    # E) Download ZIP
    # -------------------------
    zip_buf = zip_results(tables)
    st.download_button(
        "Download ZIP (TMY + hourly production)",
        data=zip_buf,
        file_name="resource_assessment_outputs.zip",
        mime="application/zip",
        type="primary",
    )

    st.caption(
        "Exports include the **original PVGIS TMY (hourly)** and, when available, the **hourly electricity production** "
        "for PV and wind. Annual totals are computed by summing hourly values of the typical year."
    )
