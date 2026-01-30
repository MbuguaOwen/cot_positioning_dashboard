from __future__ import annotations

import csv
import re
import hashlib
import io
import zipfile
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import pandas as pd
import requests

from .utils import ensure_dir

ReportType = Literal["tff", "legacy", "disagg"]

ZIP_URLS: Dict[ReportType, str] = {
    "tff": "https://www.cftc.gov/files/dea/history/fut_fin_txt_{year}.zip",
    "legacy": "https://www.cftc.gov/files/dea/history/deacot{year}.zip",
    "disagg": "https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip",
}

# Official variable-name list for TFF futures-only (from CFTC).
TFF_FUTURES_COLS: List[str] = [
    "Market_and_Exchange_Names",
    "As_of_Date_In_Form_YYMMDD",
    "Report_Date_as_YYYY-MM-DD",
    "CFTC_Contract_Market_Code",
    "CFTC_Market_Code",
    "CFTC_Region_Code",
    "CFTC_Commodity_Code",
    "Open_Interest_All",
    "Dealer_Positions_Long_All",
    "Dealer_Positions_Short_All",
    "Dealer_Positions_Spread_All",
    "Asset_Mgr_Positions_Long_All",
    "Asset_Mgr_Positions_Short_All",
    "Asset_Mgr_Positions_Spread_All",
    "Lev_Money_Positions_Long_All",
    "Lev_Money_Positions_Short_All",
    "Lev_Money_Positions_Spread_All",
    "Other_Rept_Positions_Long_All",
    "Other_Rept_Positions_Short_All",
    "Other_Rept_Positions_Spread_All",
    "Tot_Rept_Positions_Long_All",
    "Tot_Rept_Positions_Short_All",
    "NonRept_Positions_Long_All",
    "NonRept_Positions_Short_All",
    "Change_in_Open_Interest_All",
    "Change_in_Dealer_Long_All",
    "Change_in_Dealer_Short_All",
    "Change_in_Dealer_Spread_All",
    "Change_in_Asset_Mgr_Long_All",
    "Change_in_Asset_Mgr_Short_All",
    "Change_in_Asset_Mgr_Spread_All",
    "Change_in_Lev_Money_Long_All",
    "Change_in_Lev_Money_Short_All",
    "Change_in_Lev_Money_Spread_All",
    "Change_in_Other_Rept_Long_All",
    "Change_in_Other_Rept_Short_All",
    "Change_in_Other_Rept_Spread_All",
    "Change_in_Tot_Rept_Long_All",
    "Change_in_Tot_Rept_Short_All",
    "Change_in_NonRept_Long_All",
    "Change_in_NonRept_Short_All",
    "Pct_of_Open_Interest_All",
    "Pct_of_OI_Dealer_Long_All",
    "Pct_of_OI_Dealer_Short_All",
    "Pct_of_OI_Dealer_Spread_All",
    "Pct_of_OI_Asset_Mgr_Long_All",
    "Pct_of_OI_Asset_Mgr_Short_All",
    "Pct_of_OI_Asset_Mgr_Spread_All",
    "Pct_of_OI_Lev_Money_Long_All",
    "Pct_of_OI_Lev_Money_Short_All",
    "Pct_of_OI_Lev_Money_Spread_All",
    "Pct_of_OI_Other_Rept_Long_All",
    "Pct_of_OI_Other_Rept_Short_All",
    "Pct_of_OI_Other_Rept_Spread_All",
    "Pct_of_OI_Tot_Rept_Long_All",
    "Pct_of_OI_Tot_Rept_Short_All",
    "Pct_of_OI_NonRept_Long_All",
    "Pct_of_OI_NonRept_Short_All",
    "Traders_Tot_All",
    "Traders_Dealer_Long_All",
    "Traders_Dealer_Short_All",
    "Traders_Dealer_Spread_All",
    "Traders_Asset_Mgr_Long_All",
    "Traders_Asset_Mgr_Short_All",
    "Traders_Asset_Mgr_Spread_All",
    "Traders_Lev_Money_Long_All",
    "Traders_Lev_Money_Short_All",
    "Traders_Lev_Money_Spread_All",
    "Traders_Other_Rept_Long_All",
    "Traders_Other_Rept_Short_All",
    "Traders_Other_Rept_Spread_All",
    "Traders_Tot_Rept_Long_All",
    "Traders_Tot_Rept_Short_All",
    "Conc_Gross_LE_4_TDR_Long_All",
    "Conc_Gross_LE_4_TDR_Short_All",
    "Conc_Gross_LE_8_TDR_Long_All",
    "Conc_Gross_LE_8_TDR_Short_All",
    "Conc_Net_LE_4_TDR_Long_All",
    "Conc_Net_LE_4_TDR_Short_All",
    "Conc_Net_LE_8_TDR_Long_All",
    "Conc_Net_LE_8_TDR_Short_All",
    "Contract_Units",
    "CFTC_Contract_Market_Code_Quotes",
    "CFTC_Market_Code_Quotes",
    "CFTC_Commodity_Code_Quotes",
    "CFTC_SubGroup_Code",
    "FutOnly_or_Combined",
]

OFFICIAL_SCHEMAS: Dict[ReportType, Optional[List[str]]] = {
    "tff": TFF_FUTURES_COLS,
    # Legacy/Disagg futures-only files include headers in official zips.
    # If headers are missing, we fail loudly to avoid silent misalignment.
    "legacy": None,
    "disagg": None,
}

REQUIRED_COLUMNS: Dict[ReportType, List[str]] = {
    # Normalized column names (lowercase, underscores)
    "tff": [
        "market_and_exchange_names",
        "cftc_contract_market_code",
        "open_interest_all",
        "lev_money_positions_long_all",
        "lev_money_positions_short_all",
    ],
    "legacy": [
        "market_and_exchange_names",
        "cftc_contract_market_code",
        "open_interest_all",
    ],
    "disagg": [
        "market_and_exchange_names",
        "cftc_contract_market_code",
        "open_interest_all",
    ],
}


class ValidationError(RuntimeError):
    pass


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_dir(report_type: ReportType, base_dir: Path) -> Path:
    p = base_dir / report_type
    ensure_dir(p)
    return p


def download_year_zip(
    report_type: ReportType,
    year: int,
    base_dir: Path,
    refresh: bool = False,
    verbose: bool = False,
) -> Tuple[Path, str]:
    if report_type not in ZIP_URLS:
        raise ValueError(f"Unsupported report_type={report_type}")
    url = ZIP_URLS[report_type].format(year=year)
    cache_dir = _cache_dir(report_type, base_dir)
    zip_path = cache_dir / f"{year}.zip"

    if refresh or not zip_path.exists():
        ensure_dir(zip_path.parent)
        try:
            r = requests.get(url, timeout=60)
        except requests.RequestException as exc:
            raise RuntimeError(f"Download failed: {url} ({exc})") from exc
        if not r.ok:
            raise RuntimeError(f"Download failed: {url} (status {r.status_code})")
        zip_path.write_bytes(r.content)

    sha = _sha256_file(zip_path)
    sha_path = zip_path.with_suffix(".sha256")
    sha_path.write_text(f"{sha}  {zip_path.name}\n", encoding="utf-8")

    if verbose:
        print(f"SHA256 {zip_path.name}: {sha}")
    return zip_path, sha


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        s = line.strip()
        if s:
            return s
    return ""


def _looks_like_header(line: str) -> bool:
    if not line:
        return False
    s = line.lower()
    return ("market" in s and "exchange" in s) or ("cftc" in s and "code" in s)


def _read_zip_text(zip_path: Path) -> str:
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        pick = None
        for ext in (".txt", ".csv"):
            for n in names:
                if n.lower().endswith(ext):
                    pick = n
                    break
            if pick:
                break
        if not pick:
            raise FileNotFoundError(f"No .txt/.csv found inside {zip_path.name}")
        raw = zf.read(pick)
    return raw.decode("utf-8", errors="replace")


def _normalize_col(name: str) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    if "report_date_as_mm_dd_yyyy" in s:
        s = s.replace("report_date_as_mm_dd_yyyy", "report_date_as_yyyy_mm_dd")
    return s


def _validate_schema(columns: List[str], report_type: ReportType) -> None:
    expected = OFFICIAL_SCHEMAS.get(report_type)
    if expected is None:
        return
    if len(columns) != len(expected):
        raise ValidationError(
            f"{report_type} column count mismatch: expected {len(expected)} got {len(columns)}"
        )
    for i, (got, exp) in enumerate(zip(columns, expected)):
        if _normalize_col(got) != _normalize_col(exp):
            raise ValidationError(
                f"{report_type} column mismatch at position {i+1}: expected '{exp}' got '{got}'"
            )


def read_cot_zip(zip_path: Path, report_type: ReportType) -> pd.DataFrame:
    text = _read_zip_text(zip_path)
    first = _first_nonempty_line(text)
    has_header = _looks_like_header(first)
    if has_header:
        df = pd.read_csv(io.StringIO(text), dtype=str)
        _validate_schema(list(df.columns), report_type)
    else:
        schema = OFFICIAL_SCHEMAS.get(report_type)
        if not schema:
            raise ValidationError(
                f"{report_type} file appears headerless; no official schema configured."
            )
        df = pd.read_csv(io.StringIO(text), header=None, names=schema, dtype=str)
    return df


def _pick_report_date_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        s = _normalize_col(c)
        if s.startswith("report_date"):
            return c
    for c in df.columns:
        s = _normalize_col(c)
        if s.startswith("as_of_date"):
            return c
    raise ValidationError("No report/as-of date column found.")


def _parse_report_date(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    if s.str.contains("-").any():
        return pd.to_datetime(s, errors="coerce").dt.date
    return pd.to_datetime(s, format="%y%m%d", errors="coerce").dt.date


def extract_report_dates(df: pd.DataFrame) -> pd.Series:
    dcol = _pick_report_date_col(df)
    return _parse_report_date(df[dcol])


def validate_df(df: pd.DataFrame, report_type: ReportType) -> pd.DataFrame:
    norm_cols = {_normalize_col(c) for c in df.columns}
    missing = [c for c in REQUIRED_COLUMNS.get(report_type, []) if c not in norm_cols]
    if missing:
        raise ValidationError(f"{report_type} missing required columns: {missing}")

    dcol = _pick_report_date_col(df)
    dates = _parse_report_date(df[dcol])
    if dates.isna().any():
        raise ValidationError(f"{report_type} has unparseable report dates.")

    # Ensure Tuesday (holiday weeks can be Monday)
    bad = dates[dates.map(lambda d: d.weekday() not in (0, 1))]
    if not bad.empty:
        raise ValidationError(f"{report_type} has report dates outside Mon/Tue.")

    # Duplicate keys
    market_col = None
    for c in df.columns:
        if _normalize_col(c) == "cftc_contract_market_code":
            market_col = c
            break
    if market_col is None:
        raise ValidationError(f"{report_type} missing CFTC_Contract_Market_Code.")
    keys = pd.DataFrame({"market_code": df[market_col].astype(str), "report_date": dates})
    if keys.duplicated().any():
        raise ValidationError(f"{report_type} has duplicate (market_code, report_date) rows.")

    # Row count sanity
    if len(df) < 100:
        raise ValidationError(f"{report_type} row count too small: {len(df)}")

    # Monotonic dates per market
    tmp = keys.copy()
    tmp["report_date"] = pd.to_datetime(tmp["report_date"])
    for code, grp in tmp.groupby("market_code"):
        if not grp["report_date"].is_monotonic_increasing:
            grp_sorted = grp.sort_values("report_date")
            if not grp_sorted["report_date"].is_monotonic_increasing:
                raise ValidationError(f"{report_type} dates not monotonic for market_code={code}")

    return df


def load_year_df(
    report_type: ReportType,
    year: int,
    base_dir: Path,
    refresh: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    cache_dir = _cache_dir(report_type, base_dir)
    parquet_path = cache_dir / f"{year}.parquet"
    zip_path = cache_dir / f"{year}.zip"

    if parquet_path.exists() and not refresh:
        if not zip_path.exists():
            raise ValidationError(f"Missing raw zip for {report_type} {year}; expected {zip_path}.")
        sha = _sha256_file(zip_path)
        sha_path = zip_path.with_suffix(".sha256")
        sha_path.write_text(f"{sha}  {zip_path.name}\n", encoding="utf-8")
        if verbose:
            print(f"SHA256 {zip_path.name}: {sha}")
        return pd.read_parquet(parquet_path)

    zip_path, _ = download_year_zip(report_type, year, base_dir, refresh=refresh, verbose=verbose)
    df = read_cot_zip(zip_path, report_type)
    df = validate_df(df, report_type)
    df.to_parquet(parquet_path, index=False)
    return df


def load_years(
    report_type: ReportType,
    years: Iterable[int],
    base_dir: Path,
    refresh: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for y in years:
        frames.append(load_year_df(report_type, y, base_dir, refresh=refresh, verbose=verbose))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)
