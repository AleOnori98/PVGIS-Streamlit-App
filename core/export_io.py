from __future__ import annotations
import io
import zipfile
import pandas as pd

def zip_results(tables: dict[str, pd.DataFrame]) -> io.BytesIO:
    """
    Build an in-memory ZIP containing CSVs.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, df in tables.items():
            csv_bytes = df.to_csv(index=True).encode("utf-8")
            zf.writestr(f"{name}.csv", csv_bytes)
    buf.seek(0)
    return buf
