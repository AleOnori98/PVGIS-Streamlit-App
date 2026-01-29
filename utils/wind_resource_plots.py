import numpy as np
import math
import plotly.graph_objects as go


def weibull_fit_moments(ws: np.ndarray) -> tuple[float, float]:
    """
    Simple Weibull fit via method of moments.
    Returns (k, c) where:
      - k = shape
      - c = scale
    Works well enough for UI analytics without scipy.
    """
    ws = np.asarray(ws, float)
    ws = ws[np.isfinite(ws) & (ws >= 0)]
    if ws.size < 10:
        return (np.nan, np.nan)

    mean = ws.mean()
    std = ws.std(ddof=0)
    if mean <= 0 or std <= 0:
        return (np.nan, np.nan)

    # Approx for k using CV
    cv = std / mean
    # common approximation: k ≈ (cv)^(-1.086)
    k = cv ** (-1.086) if cv > 0 else np.nan

    # c from mean: mean = c * Gamma(1 + 1/k)
    # Use math.gamma
    c = mean / math.gamma(1.0 + 1.0 / k) if np.isfinite(k) and k > 0 else np.nan
    return float(k), float(c)


def weibull_pdf(v: np.ndarray, k: float, c: float) -> np.ndarray:
    v = np.asarray(v, float)
    if not (np.isfinite(k) and np.isfinite(c) and k > 0 and c > 0):
        return np.full_like(v, np.nan, dtype=float)
    return (k / c) * (v / c) ** (k - 1) * np.exp(-(v / c) ** k)


def wind_power_density_wm2(ws: np.ndarray, rho: float = 1.225) -> float:
    """
    Theoretical wind power per swept area: 0.5 * rho * E[v^3]
    """
    ws = np.asarray(ws, float)
    ws = ws[np.isfinite(ws) & (ws >= 0)]
    if ws.size == 0:
        return float("nan")
    return float(0.5 * rho * np.mean(ws ** 3))


def wind_rose_figure(wd_deg: np.ndarray, ws: np.ndarray) -> go.Figure:
    """
    Simple wind rose: frequency by direction sector (optionally weighted by speed bins).
    Here: plot frequency by direction only, colored by mean speed per sector.
    """
    wd = np.asarray(wd_deg, float) % 360.0
    sp = np.asarray(ws, float)
    m = np.isfinite(wd) & np.isfinite(sp)
    wd, sp = wd[m], sp[m]
    if wd.size == 0:
        return go.Figure()

    # 16 sectors (22.5°)
    n_sectors = 16
    edges = np.linspace(0, 360, n_sectors + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    freq = np.zeros(n_sectors)
    mean_sp = np.zeros(n_sectors)

    for i in range(n_sectors):
        mask = (wd >= edges[i]) & (wd < edges[i + 1])
        freq[i] = mask.mean() * 100.0
        mean_sp[i] = float(np.mean(sp[mask])) if mask.any() else 0.0

    fig = go.Figure()
    fig.add_trace(
        go.Barpolar(
            r=freq,
            theta=centers,
            width=np.full(n_sectors, 360 / n_sectors),
            marker=dict(color=mean_sp, colorscale="Blues"),
            name="Frequency [%]",
            opacity=0.9,
        )
    )
    fig.update_layout(
        template="plotly_white",
        polar=dict(
            angularaxis=dict(direction="clockwise"),
            radialaxis=dict(title="Frequency [%]"),
        ),
        margin=dict(l=10, r=10, t=30, b=10),
        height=360,
        title="Wind rose",
        showlegend=False,
    )
    return fig


def weibull_figure(ws: np.ndarray) -> go.Figure:
    ws = np.asarray(ws, float)
    ws = ws[np.isfinite(ws) & (ws >= 0)]
    if ws.size == 0:
        return go.Figure()

    k, c = weibull_fit_moments(ws)
    v = np.linspace(0, max(1.0, np.percentile(ws, 99) * 1.2), 200)
    pdf = weibull_pdf(v, k, c)

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=ws, nbinsx=30, histnorm="probability density", name="TMY histogram"))
    if np.all(np.isfinite(pdf)):
        fig.add_trace(go.Scatter(x=v, y=pdf, mode="lines", name=f"Weibull fit (k={k:.2f}, c={c:.2f})"))

    fig.update_layout(
        template="plotly_white",
        height=360,
        margin=dict(l=10, r=10, t=30, b=10),
        title="Wind speed distribution",
        xaxis_title="Wind speed [m/s]",
        yaxis_title="Probability density [-]",
    )
    return fig
