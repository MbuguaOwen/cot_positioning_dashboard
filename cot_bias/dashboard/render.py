from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd


def render_dashboard(
    out_dir: Path,
    as_of: str,
    max_report_date: Optional[str],
    warnings: List[str],
    latest_df: pd.DataFrame,
    history_df: pd.DataFrame,
    pairs_df: Optional[pd.DataFrame] = None,
    gate_df: Optional[pd.DataFrame] = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "dashboard.html"

    def _fmt(df: pd.DataFrame) -> pd.DataFrame:
        df2 = df.copy()
        for c in df2.columns:
            if pd.api.types.is_bool_dtype(df2[c]):
                df2[c] = df2[c].map(
                    lambda x: "" if pd.isna(x) else ("true" if bool(x) else "false")
                )
                continue
            if pd.api.types.is_numeric_dtype(df2[c]):
                df2[c] = df2[c].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
        return df2

    latest_tbl = _fmt(latest_df)

    parts = []
    parts.append("<html><head><meta charset='utf-8'><title>COT Dashboard</title>")
    parts.append("<style>body{font-family:Arial,Helvetica,sans-serif;padding:16px;} table{border-collapse:collapse;margin:12px 0;} th,td{border:1px solid #ddd;padding:6px 8px;font-size:12px;} th{background:#f5f5f5;} .warn{color:#b00020;font-weight:bold;} .note{color:#444;}</style>")
    parts.append("</head><body>")
    parts.append("<h1>COT Positioning Dashboard</h1>")
    parts.append(f"<p><strong>As-of:</strong> {as_of}</p>")
    parts.append(f"<p><strong>Data max report_date:</strong> {max_report_date or 'N/A'}</p>")
    if warnings:
        parts.append("<div class='warn'>")
        for w in warnings:
            parts.append(f"<p>{w}</p>")
        parts.append("</div>")
    parts.append("<h2>Instruments (latest as-of)</h2>")
    parts.append(latest_tbl.to_html(index=False, escape=True))
    if pairs_df is not None and not pairs_df.empty:
        pairs_tbl = _fmt(pairs_df)
        parts.append("<h2>FX Pairs (all available)</h2>")
        parts.append(pairs_tbl.to_html(index=False, escape=True))
    parts.append("<h2>FX Pairs (passed COT filters)</h2>")
    if gate_df is None or gate_df.empty:
        parts.append("<p class='note'>No FX pairs passed the current COT filter set.</p>")
    else:
        gate_tbl = _fmt(gate_df)
        parts.append(gate_tbl.to_html(index=False, escape=True))
    parts.append("</body></html>")

    html_path.write_text("\n".join(parts), encoding="utf-8")
    return html_path
