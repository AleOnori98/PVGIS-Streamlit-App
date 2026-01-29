from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", s).strip("_")


def build_library_from_excel(excel_path: Path, out_dir: Path) -> None:
    """
    Excel expected format per sheet (0-index rows, col B = index 1):
      row 0: ["Rated Power [kW]", <value>]
      row 1: ["Rotor Diameter [m]", <value>]
      row 2: ["Hub Height [m]", <value>]
      row 3: ["Wind Speed [m/s]", "Power [kW]"]
      row 4+: curve data
    Sheet name expected: "<model> - <turbine_type>"
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    xls = pd.ExcelFile(excel_path)
    index_rows = []

    for sheet in xls.sheet_names:
        df = pd.read_excel(excel_path, sheet_name=sheet, header=None)

        parts = sheet.split(" - ")
        model = parts[0].strip()
        turbine_type = parts[1].strip() if len(parts) > 1 else ""

        rated_kw = float(df.iloc[0, 1])
        rotor_m = float(df.iloc[1, 1])
        hub_m = float(df.iloc[2, 1])

        curve = df.iloc[4:, :2].copy()
        curve.columns = ["wind_speed_mps", "power_kw"]
        curve = curve.dropna()
        curve["wind_speed_mps"] = curve["wind_speed_mps"].astype(float)
        curve["power_kw"] = curve["power_kw"].astype(float)
        curve = curve.sort_values("wind_speed_mps")

        fname = f"{_safe(model)}__{_safe(turbine_type)}.csv" if turbine_type else f"{_safe(model)}.csv"
        curve.to_csv(out_dir / fname, index=False)

        index_rows.append(
            {
                "model": model,
                "turbine_type": turbine_type,
                "rated_power_kw": rated_kw,
                "rotor_diameter_m": rotor_m,
                "hub_height_m": hub_m,
                "curve_csv": fname,
                "sheet": sheet,
                "source": excel_path.name,
            }
        )

    index_df = pd.DataFrame(index_rows).sort_values(["turbine_type", "model"])
    index_df.to_csv(out_dir / "turbines_index.csv", index=False)
    print(f"✅ Wrote {len(index_df)} turbines to: {out_dir}")
    print(f"✅ Index: {out_dir / 'turbines_index.csv'}")


if __name__ == "__main__":
    # adjust to your repo layout
    repo_root = Path(__file__).resolve().parents[1]
    excel = repo_root / "config" / "Wind Turbine Power Curves.xlsx"
    out = repo_root / "data" / "turbines"
    build_library_from_excel(excel, out)
