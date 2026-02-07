from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .render_updated import render_dashboard_updated


def render_dashboard(
    out_dir: Path,
    as_of: str,
    max_report_date: Optional[str],
    warnings: List[str],
    latest_df: pd.DataFrame,
    history_df: pd.DataFrame,
    pairs_df: Optional[pd.DataFrame] = None,
    gate_df: Optional[pd.DataFrame] = None,
    tier_tables: Optional[Dict[str, pd.DataFrame]] = None,
    tier_meta: Optional[Dict[str, object]] = None,
) -> Path:
    del as_of, max_report_date, warnings, latest_df, history_df, pairs_df, gate_df, tier_tables, tier_meta
    return render_dashboard_updated(out_dir, out_dir / "dashboard.html")
