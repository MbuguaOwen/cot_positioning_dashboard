from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

from .utils import Config, ensure_dirs

def write_sqlite(df: pd.DataFrame, sqlite_path: Path, table: str) -> None:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(sqlite_path)) as conn:
        df.to_sql(table, conn, if_exists="replace", index=False)

def try_write_parquet(df: pd.DataFrame, parquet_path: Path) -> Optional[Path]:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception:
        return None

def store_raw(cfg: Config, fin_df: pd.DataFrame, dis_df: pd.DataFrame, *, combined: bool) -> None:
    ensure_dirs(cfg)
    # SQLite
    write_sqlite(fin_df, cfg.sqlite_path, "financial_raw")
    write_sqlite(dis_df, cfg.sqlite_path, "disagg_raw")

    # Parquet (best-effort)
    try_write_parquet(fin_df, cfg.parquet_dir / ("financial_raw.parquet"))
    try_write_parquet(dis_df, cfg.parquet_dir / ("disagg_raw.parquet"))

def load_raw(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load from Parquet if available, else from SQLite."""
    fin_pq = cfg.parquet_dir / "financial_raw.parquet"
    dis_pq = cfg.parquet_dir / "disagg_raw.parquet"

    if fin_pq.exists() and dis_pq.exists():
        return pd.read_parquet(fin_pq), pd.read_parquet(dis_pq)

    # SQLite fallback
    import sqlite3
    with sqlite3.connect(str(cfg.sqlite_path)) as conn:
        fin = pd.read_sql_query("SELECT * FROM financial_raw", conn)
        dis = pd.read_sql_query("SELECT * FROM disagg_raw", conn)
    return fin, dis
