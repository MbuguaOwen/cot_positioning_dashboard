from __future__ import annotations

import csv
import io
import zipfile
import math
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd
import requests

from .utils import ensure_dir, Config


@dataclass(frozen=True)
class SourceURLs:
    """CFTC official weekly comma-delimited datasets."""

    # Traders in Financial Futures (TFF) — Futures-and-Options Combined
    tff_combined: str = "https://www.cftc.gov/dea/newcot/FinComWk.txt"

    # Disaggregated COT — Futures-and-Options Combined
    disagg_combined: str = "https://www.cftc.gov/dea/newcot/c_disagg.txt"


TFF_NCOLS = 87
DISAGG_NCOLS = 191


def fetch_datasets(cfg: Config, combined: bool = True, force: bool = False) -> tuple[Path, Path]:
    """Download the official weekly CFTC datasets into ``cfg.data_dir``.

    Notes:
    - The CFTC weekly downloads are often *headerless* comma-delimited text files.
      Parsing/typing is handled by :func:`read_cot_csv`.
    - ``combined=True`` uses Futures+Options combined datasets (recommended).
    """

    fin_url = cfg.urls["financial_combined_url"] if combined else cfg.urls["financial_futures_only_url"]
    dis_url = cfg.urls["disagg_combined_url"] if combined else cfg.urls["disagg_futures_only_url"]

    fin_name = "FinComWk.txt" if combined else "FinFutWk.txt"
    dis_name = "c_disagg.txt" if combined else "f_disagg.txt"

    fin_path = cfg.data_dir / fin_name
    dis_path = cfg.data_dir / dis_name

    if force or not fin_path.exists():
        download(fin_url, fin_path)
    if force or not dis_path.exists():
        download(dis_url, dis_path)

    return fin_path, dis_path


def fetch_historical_datasets(
    cfg: Config,
    combined: bool = True,
    years_back: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, List[Path]]:
    """Download Historical Compressed (zipped) datasets and return concatenated DataFrames.

    Why this exists
    - The "newcot" weekly text files (e.g. FinComWk.txt) contain **only the latest report date**.
      You cannot compute 3-year percentiles/zscores from them.
    - The Historical Compressed zips contain the full weekly history (by year).

    This function downloads the last N years (default: inferred from cfg.rolling_weeks) and concatenates them.
    The downloaded zip files are cached under ``cfg.hist_dir`` (default: data/historical).
    """

    now_year = dt.date.today().year
    lookback_weeks = max(int(getattr(cfg, "rolling_weeks", 156)), int(getattr(cfg, "delta_weeks", 4))) + 12
    if years_back is None:
        years_back = int(math.ceil(lookback_weeks / 52.0)) + 1
        years_back = max(2, min(years_back, 12))  # safety cap

    start_year = now_year - years_back + 1
    years = list(range(start_year, now_year + 1))

    base = "https://www.cftc.gov/files/dea/history/"
    if combined:
        fin_tmpl = base + "com_fin_txt_{year}.zip"
        dis_tmpl = base + "com_disagg_txt_{year}.zip"
    else:
        fin_tmpl = base + "fut_fin_txt_{year}.zip"
        dis_tmpl = base + "fut_disagg_txt_{year}.zip"

    hist_dir = _resolve_hist_dir(cfg)
    downloaded: List[Path] = []

    def _get_zip(url: str, name: str, year: int) -> Path:
        p = hist_dir / name
        # Redownload current year (it changes weekly). Cache prior years.
        if (not p.exists()) or (year == now_year):
            download(url, p)
            downloaded.append(p)
        return p

    def _read_zip(path: Path) -> pd.DataFrame:
        with zipfile.ZipFile(path, "r") as zf:
            # Prefer .txt payloads.
            names = [n for n in zf.namelist() if not n.endswith("/")]
            txts = [n for n in names if n.lower().endswith(".txt")]
            target = txts[0] if txts else names[0]
            raw = zf.read(target)
        text = raw.decode("utf-8", errors="replace")
        return read_cot_csv_text(text)

    fin_frames = []
    dis_frames = []
    for y in years:
        fin_zip = _get_zip(fin_tmpl.format(year=y), f"{Path(fin_tmpl.format(year=y)).name}", y)
        dis_zip = _get_zip(dis_tmpl.format(year=y), f"{Path(dis_tmpl.format(year=y)).name}", y)
        fin_frames.append(_read_zip(fin_zip))
        dis_frames.append(_read_zip(dis_zip))

    fin_df = pd.concat(fin_frames, ignore_index=True, sort=False)
    dis_df = pd.concat(dis_frames, ignore_index=True, sort=False)
    return fin_df, dis_df, downloaded


def download(url: str, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    try:
        r = requests.get(url, timeout=60)
    except requests.RequestException as exc:
        raise RuntimeError(f"Download failed: {url} ({exc})") from exc
    if not r.ok:
        raise RuntimeError(f"Download failed: {url} (status {r.status_code})")
    out_path.write_bytes(r.content)


def _resolve_hist_dir(cfg: Config) -> Path:
    hist_dir = getattr(cfg, "hist_dir", None)
    if hist_dir is None:
        hist_dir = cfg.data_dir / "historical"
    hist_dir = Path(hist_dir)
    ensure_dir(hist_dir)
    return hist_dir


def _count_fields(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline()
    return len(next(csv.reader([first_line])))


def _tff_schema(ncols: int) -> list[str]:
    """Minimal schema (first ~25 columns) for FinComWk.txt when file has no header row."""
    cols = [f"col_{i:03d}" for i in range(1, ncols + 1)]
    # Column order is per the CFTC "Variable Names" list for TFF combined.
    mapping = {
        0: "Market_and_Exchange_Names",
        1: "As_of_Date_In_Form_YYMMDD",
        2: "Report_Date_as_MM_DD_YYYY",
        3: "CFTC_Contract_Market_Code",
        4: "CFTC_Market_Code",
        5: "CFTC_Region_Code",
        6: "CFTC_Commodity_Code",
        7: "Open_Interest_All",
        8: "Dealer_Positions_Long_All",
        9: "Dealer_Positions_Short_All",
        10: "Dealer_Positions_Spread_All",
        11: "Asset_Mgr_Positions_Long_All",
        12: "Asset_Mgr_Positions_Short_All",
        13: "Asset_Mgr_Positions_Spread_All",
        14: "Lev_Money_Positions_Long_All",
        15: "Lev_Money_Positions_Short_All",
        16: "Lev_Money_Positions_Spread_All",
        17: "Other_Rept_Positions_Long_All",
        18: "Other_Rept_Positions_Short_All",
        19: "Other_Rept_Positions_Spread_All",
        20: "Tot_Rept_Positions_Long_All",
        21: "Tot_Rept_Positions_Short_All",
        22: "NonRept_Positions_Long_All",
        23: "NonRept_Positions_Short_All",
    }
    for idx, name in mapping.items():
        if idx < ncols:
            cols[idx] = name
    return cols


def _disagg_schema(ncols: int) -> list[str]:
    """Minimal schema (first ~25 columns) for c_disagg.txt when file has no header row."""
    cols = [f"col_{i:03d}" for i in range(1, ncols + 1)]
    # Column order is per the CFTC "Variable Names" list for Disaggregated combined.
    mapping = {
        0: "Market_and_Exchange_Names",
        1: "As_of_Date_In_Form_YYMMDD",
        2: "As_of_Date_Form_YYYY-MM-DD",
        3: "CFTC_Contract_Market_Code",
        4: "CFTC_Market_Code",
        5: "CFTC_Region_Code",
        6: "CFTC_Commodity_Code",
        7: "Open_Interest_All",
        8: "Prod_Merc_Positions_Long_All",
        9: "Prod_Merc_Positions_Short_All",
        10: "Swap_Positions_Long_All",
        11: "Swap_Positions_Short_All",
        12: "Swap_Positions_Spread_All",
        13: "M_Money_Positions_Long_All",
        14: "M_Money_Positions_Short_All",
        15: "M_Money_Positions_Spread_All",
        16: "Other_Rept_Positions_Long_All",
        17: "Other_Rept_Positions_Short_All",
        18: "Other_Rept_Positions_Spread_All",
        19: "Tot_Rept_Positions_Long_All",
        20: "Tot_Rept_Positions_Short_All",
        21: "NonRept_Positions_Long_All",
        22: "NonRept_Positions_Short_All",
    }
    for idx, name in mapping.items():
        if idx < ncols:
            cols[idx] = name
    return cols


def _looks_like_header(columns: list[object]) -> bool:
    # If a file *has* headers, we expect at least some official tokens.
    joined = " ".join([str(c).lower() for c in columns])
    return (
        ("market" in joined and "exchange" in joined)
        or ("open_interest" in joined)
        or ("cftc" in joined and "code" in joined)
        or ("report_date" in joined)
    )


def read_cot_csv(path: str | Path) -> pd.DataFrame:
    """Read a CFTC comma-delimited file.

    Important: the CFTC weekly downloads (e.g., FinComWk.txt, c_disagg.txt) are
    commonly *headerless*. Pandas will otherwise treat the first data row as the
    header and everything downstream breaks.
    """

    path = Path(path)

    # 1) Try as headered (some historical exports may include a header row)
    try:
        df_try = pd.read_csv(path, dtype=str)
        if _looks_like_header(list(df_try.columns)):
            return df_try
    except Exception:
        pass

    # 2) Headerless: infer by field count and apply a minimal schema
    ncols = _count_fields(path)
    if ncols == TFF_NCOLS:
        names = _tff_schema(ncols)
    elif ncols == DISAGG_NCOLS:
        names = _disagg_schema(ncols)
    else:
        # Unknown schema length (rare, but possible with older exports).
        # Still label the key early fields defensively so downstream code can function.
        names = [f"col_{i:03d}" for i in range(1, ncols + 1)]
        if ncols >= 1:
            names[0] = "Market_and_Exchange_Names"
        if ncols >= 2:
            names[1] = "As_of_Date_In_Form_YYMMDD"
        if ncols >= 8:
            names[7] = "Open_Interest_All"

    return pd.read_csv(path, header=None, names=names, dtype=str)


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        s = line.strip()
        if s:
            return s
    return ""


def read_cot_csv_text(text: str) -> pd.DataFrame:
    """Read a CFTC comma-delimited string (headered or headerless)."""

    # 1) Try headered
    try:
        df_try = pd.read_csv(io.StringIO(text), dtype=str)
        if _looks_like_header(list(df_try.columns)):
            return df_try
    except Exception:
        pass

    # 2) Headerless: infer by field count from the first line
    first = _first_nonempty_line(text)
    ncols = len(list(csv.reader([first]))[0]) if first else 0
    if ncols == TFF_NCOLS:
        names = _tff_schema(ncols)
    elif ncols == DISAGG_NCOLS:
        names = _disagg_schema(ncols)
    else:
        names = [f"col_{i:03d}" for i in range(1, max(1, ncols) + 1)]
        if ncols >= 1:
            names[0] = "Market_and_Exchange_Names"
        if ncols >= 2:
            names[1] = "As_of_Date_In_Form_YYMMDD"
        if ncols >= 8:
            names[7] = "Open_Interest_All"

    return pd.read_csv(io.StringIO(text), header=None, names=names, dtype=str)


def read_cot_from_zip(zip_path: str | Path) -> pd.DataFrame:
    """Read a CFTC historical-compressed zip (Text/CSV)."""

    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Prefer .txt, fallback to .csv; ignore dirs.
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
    text = raw.decode("utf-8", errors="replace")
    return read_cot_csv_text(text)
