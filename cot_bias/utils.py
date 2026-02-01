from __future__ import annotations

import os
import re
import json
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

DEFAULT_URLS = {
    "financial_futures_only_url": "https://www.cftc.gov/dea/newcot/FinFutWk.txt",
    "financial_combined_url": "https://www.cftc.gov/dea/newcot/FinComWk.txt",
    "disagg_futures_only_url": "https://www.cftc.gov/dea/newcot/f_disagg.txt",
    "disagg_combined_url": "https://www.cftc.gov/dea/newcot/c_disagg.txt",
}

DEFAULT_CONTRACT_PATTERNS = {
    # FX (TFF)
    # Anchor at start to avoid cross-rate contracts like "EURO FX/JAPANESE YEN XRATE".
    "USD": r"(?i)^(?:U\\.S\\.\s*)?DOLLAR\s+INDEX\b",
    "EUR": r"(?i)^EURO\s+FX\b(?!/)" ,
    "JPY": r"(?i)^JAPANESE\s+YEN\b(?!/)" ,
    "GBP": r"(?i)^BRITISH\s+POUND\b" ,
    "AUD": r"(?i)^AUSTRALIAN\s+DOLLAR\b" ,
    "CHF": r"(?i)^SWISS\s+FRANC\b" ,
    "CAD": r"(?i)^CANADIAN\s+DOLLAR\b" ,
    "NZD": r"(?i)^NEW\s+ZEALAND\s+DOLLAR\b" ,

    # Metals (Disaggregated)
    # Prefer COMEX venues (avoids smaller alternative listings like Coinbase micro contracts).
    "GOLD": r"(?i)^GOLD\b.*COMMODITY\s+EXCHANGE" ,
    "SILVER": r"(?i)^SILVER\b.*COMMODITY\s+EXCHANGE" ,
}

@dataclass(frozen=True)
class Config:
    data_dir: Path
    hist_dir: Path
    processed_dir: Path
    sqlite_path: Path
    parquet_dir: Path

    urls: Dict[str, str]
    contract_patterns: Dict[str, str]

    fx_group: str
    metals_group: str

    rolling_weeks: int
    delta_weeks: int

def _load_yaml_if_available(path: Path) -> Optional[Dict[str, Any]]:
    # PyYAML isn't required. If present, we use it. If not, we fall back to JSON only.
    try:
        import yaml  # type: ignore
    except Exception:
        return None

    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path) -> None:
    """Create directory if missing."""
    path.mkdir(parents=True, exist_ok=True)

def load_config(config_path: Optional[str] = None) -> Config:
    # Search order:
    # 1) explicit path
    # 2) ./config.yaml
    # 3) defaults
    cfg_file = Path(config_path) if config_path else Path("config.yaml")
    doc = _load_yaml_if_available(cfg_file)

    # Defaults
    data_dir = Path("data")
    hist_dir = data_dir / "historical"
    processed_dir = data_dir / "processed"
    sqlite_path = data_dir / "cot.sqlite"
    parquet_dir = data_dir / "parquet"
    urls = dict(DEFAULT_URLS)
    contracts = dict(DEFAULT_CONTRACT_PATTERNS)
    fx_group = "lev_money"
    metals_group = "m_money"
    rolling_weeks = 156
    delta_weeks = 4

    if doc:
        storage = doc.get("storage", {}) if isinstance(doc, dict) else {}
        sources = doc.get("sources", {}) if isinstance(doc, dict) else {}
        contract_patterns = doc.get("contracts", {}) if isinstance(doc, dict) else {}
        bias_driver = doc.get("bias_driver", {}) if isinstance(doc, dict) else {}
        metrics = doc.get("metrics", {}) if isinstance(doc, dict) else {}

        if "data_dir" in storage:
            data_dir = Path(storage["data_dir"])
            hist_dir = data_dir / "historical"
        if "hist_dir" in storage:
            hist_dir = Path(storage["hist_dir"])
        if "processed_dir" in storage:
            processed_dir = Path(storage["processed_dir"])
        if "sqlite_path" in storage:
            sqlite_path = Path(storage["sqlite_path"])
        if "parquet_dir" in storage:
            parquet_dir = Path(storage["parquet_dir"])

        for k, v in (sources or {}).items():
            if isinstance(v, str):
                urls[k] = v

        for k, v in (contract_patterns or {}).items():
            if isinstance(v, str):
                contracts[k] = v

        if isinstance(bias_driver, dict):
            fx_group = bias_driver.get("fx_group", fx_group)
            metals_group = bias_driver.get("metals_group", metals_group)

        if isinstance(metrics, dict):
            rolling_weeks = int(metrics.get("rolling_weeks", rolling_weeks))
            delta_weeks = int(metrics.get("delta_weeks", delta_weeks))

    # Normalize derived paths
    if not sqlite_path.is_absolute():
        sqlite_path = sqlite_path
    if not parquet_dir.is_absolute():
        parquet_dir = parquet_dir

    return Config(
        data_dir=data_dir,
        hist_dir=hist_dir,
        processed_dir=processed_dir,
        sqlite_path=sqlite_path,
        parquet_dir=parquet_dir,
        urls=urls,
        contract_patterns=contracts,
        fx_group=fx_group,
        metals_group=metals_group,
        rolling_weeks=rolling_weeks,
        delta_weeks=delta_weeks,
    )

def ensure_dirs(cfg: Config) -> None:
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.hist_dir.mkdir(parents=True, exist_ok=True)
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    cfg.parquet_dir.mkdir(parents=True, exist_ok=True)
    if cfg.sqlite_path.parent != Path("."):
        cfg.sqlite_path.parent.mkdir(parents=True, exist_ok=True)

def slugify(s: str) -> str:
    s2 = re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_")
    return s2.lower()

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def parse_iso_date(value: Optional[str]) -> Optional[dt.date]:
    if value is None:
        return None
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid date '{value}'. Use YYYY-MM-DD.") from exc

def most_recent_tuesday(d: dt.date) -> dt.date:
    # Tuesday is 1 (Mon=0)
    delta = (d.weekday() - 1) % 7
    return d - dt.timedelta(days=delta)
