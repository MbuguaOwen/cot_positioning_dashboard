from __future__ import annotations

import re
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .utils import Config, clamp, slugify

# ---------------------------
# Column normalization
# ---------------------------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [re.sub(r"[^a-zA-Z0-9]+", "_", c).strip("_").lower() for c in out.columns]
    return out

def _pick_date_col(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    # Prefer explicit report date if present (TFF)
    for c in cols:
        if str(c).startswith("report_date"):
            return c
    # Disaggregated uses As_of_Date_Form_* (format may differ; we parse flexibly)
    for c in cols:
        s = str(c)
        if s.startswith("as_of_date_form") or s.startswith("as_of_date"):
            return c
    # Common fallbacks
    for cand in (
        "report_date_as_mm_dd_yyyy",
        "report_date_as_yyyy_mm_dd",
        "as_of_date_form_mm_dd_yyyy",
        "as_of_date_form_yyyy_mm_dd",
    ):
        if cand in df.columns:
            return cand
    raise KeyError("Could not locate report/as-of date column")

def _to_date(s: pd.Series) -> pd.Series:
    # handles 'YYYY-MM-DD' or 'YYMMDD'
    s = s.astype(str)
    # if looks like YYYY-MM-DD
    if s.str.contains("-").any():
        return pd.to_datetime(s, errors="coerce").dt.date
    # try yymmdd
    return pd.to_datetime(s, format="%y%m%d", errors="coerce").dt.date

def _to_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

# ---------------------------
# Dataset schemas
# ---------------------------

TFF_GROUPS = {
    # Traders in Financial Futures (TFF) official variable names:
    # Dealer / Asset_Mgr / Lev_Money / Other_Rept / NonRept
    "dealer": ("dealer_positions_long_all", "dealer_positions_short_all"),
    "asset_mgr": ("asset_mgr_positions_long_all", "asset_mgr_positions_short_all"),
    "lev_money": ("lev_money_positions_long_all", "lev_money_positions_short_all"),
    "other_rept": ("other_rept_positions_long_all", "other_rept_positions_short_all"),
    "nonrept": ("nonrept_positions_long_all", "nonrept_positions_short_all"),
}

DISAGG_GROUPS = {
    # Disaggregated report official variable names:
    # Prod_Merc / Swap / M_Money / Other_Rept / NonRept
    "prod_merc": ("prod_merc_positions_long_all", "prod_merc_positions_short_all"),
    "swap": ("swap_positions_long_all", "swap_positions_short_all"),
    "m_money": ("m_money_positions_long_all", "m_money_positions_short_all"),
    "other_rept": ("other_rept_positions_long_all", "other_rept_positions_short_all"),
    "nonrept": ("nonrept_positions_long_all", "nonrept_positions_short_all"),
}

def detect_dataset_type(df: pd.DataFrame) -> str:
    cols = set(df.columns)
    # TFF (financial) has Lev_Money / Asset_Mgr columns
    if "lev_money_positions_long_all" in cols or "asset_mgr_positions_long_all" in cols:
        return "tff"
    # Disaggregated has Prod_Merc / M_Money columns
    if "prod_merc_positions_long_all" in cols or "m_money_positions_long_all" in cols:
        return "disagg"
    raise ValueError("Could not detect dataset type from columns; expected TFF or Disaggregated fields.")

def _open_interest_col(df: pd.DataFrame) -> str:
    for c in ["open_interest_all", "open_interest"]:
        if c in df.columns:
            return c
    # defensive
    for c in df.columns:
        if c.startswith("open_interest"):
            return c
    raise KeyError("Could not locate open interest column")

def _market_col(df: pd.DataFrame) -> str:
    for c in ["market_and_exchange_names", "market_and_exchange_name", "market_and_exchange" ]:
        if c in df.columns:
            return c
    # fallback: first column that includes 'market'
    for c in df.columns:
        if "market" in c and "exchange" in c:
            return c
    raise KeyError("Could not locate market name column")

# ---------------------------
# Rolling metrics
# ---------------------------

def rolling_zscore(x: pd.Series, window: int) -> pd.Series:
    mu = x.rolling(window=window, min_periods=window).mean()
    sd = x.rolling(window=window, min_periods=window).std(ddof=0)
    return (x - mu) / (sd.replace(0, np.nan))

def rolling_percentile_of_last(x: pd.Series, window: int) -> pd.Series:
    def _pct(arr: np.ndarray) -> float:
        if len(arr) == 0 or np.isnan(arr[-1]):
            return np.nan
        v = arr[-1]
        arr2 = arr[~np.isnan(arr)]
        if len(arr2) == 0:
            return np.nan
        return float((arr2 <= v).sum()) / float(len(arr2)) * 100.0

    return x.rolling(window=window, min_periods=window).apply(_pct, raw=True)

# ---------------------------
# Labels
# ---------------------------

def bias_label(z: float, pct: float) -> str:
    if np.isnan(z) or np.isnan(pct):
        return "UNKNOWN"
    if z >= 0.5 and pct >= 55:
        return "BULLISH"
    if z <= -0.5 and pct <= 45:
        return "BEARISH"
    return "NEUTRAL"

def reversal_risk_label(pct: float) -> str:
    if np.isnan(pct):
        return "UNKNOWN"
    if pct >= 95 or pct <= 5:
        return "EXTREME"
    if pct >= 90 or pct <= 10:
        return "HIGH"
    if pct >= 80 or pct <= 20:
        return "MED"
    return "LOW"

# ---------------------------
# Core computation
# ---------------------------

def _filter_contract(df: pd.DataFrame, pattern: str) -> pd.DataFrame:
    """Filter the dataset to the contract(s) matching the pattern.

    If multiple contract markets match (common when patterns are broad, e.g.
    matching cross-rate contracts, micro contracts, or alternative venues), we
    automatically select the *dominant* market by average open interest over the
    last ~52 weeks (fallback: whole history). This makes the dashboard more
    robust without forcing you to hand-tune regexes.
    """

    mcol = _market_col(df)
    mask = df[mcol].astype(str).str.contains(pattern, regex=True, na=False)
    sub = df.loc[mask].copy()
    if sub.empty:
        return sub

    # Disambiguate if multiple markets match
    markets = sub[mcol].astype(str)
    uniq = markets.unique()
    if len(uniq) <= 1:
        return sub

    try:
        dcol = _pick_date_col(sub)
        oicol = _open_interest_col(sub)

        tmp = sub[[mcol, dcol, oicol]].copy()
        tmp["report_date"] = _to_date(tmp[dcol])
        tmp["open_interest"] = _to_float(tmp[oicol])
        tmp = tmp.dropna(subset=["report_date", "open_interest"])
        if tmp.empty:
            return sub

        last = tmp["report_date"].max()
        cut = last - dt.timedelta(weeks=52)
        tmp_recent = tmp[tmp["report_date"] >= cut]
        if tmp_recent.empty:
            tmp_recent = tmp

        score = tmp_recent.groupby(mcol)["open_interest"].mean()
        best_market = score.idxmax()
        return sub[sub[mcol].astype(str) == str(best_market)].copy()
    except Exception:
        # If anything goes wrong, keep the broad match (better than hard-failing)
        return sub

def _build_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    dcol = _pick_date_col(df)
    mcol = _market_col(df)
    oicol = _open_interest_col(df)

    out = df.copy()
    out["report_date"] = _to_date(out[dcol])
    out["open_interest"] = _to_float(out[oicol])
    out["market"] = out[mcol].astype(str)
    out = out.dropna(subset=["report_date"]).sort_values("report_date").reset_index(drop=True)
    return out

def compute_instrument_metrics(df_raw: pd.DataFrame, dataset_type: str, pattern: str, driver_group: str, rolling_weeks: int, delta_weeks: int, symbol: str) -> pd.DataFrame:
    df = normalize_columns(df_raw)
    df = _filter_contract(df, pattern)
    if df.empty:
        return pd.DataFrame()

    ts = _build_timeseries(df)
    if dataset_type == "tff":
        groups = TFF_GROUPS
    elif dataset_type == "disagg":
        groups = DISAGG_GROUPS
    else:
        return pd.DataFrame()

    # compute nets for all available groups
    for g, (lc, sc) in groups.items():
        if lc in ts.columns and sc in ts.columns:
            ts[f"{g}_long"] = _to_float(ts[lc])
            ts[f"{g}_short"] = _to_float(ts[sc])
            ts[f"{g}_net"] = ts[f"{g}_long"] - ts[f"{g}_short"]
            ts[f"{g}_net_pct_oi"] = (ts[f"{g}_net"] / ts["open_interest"]) * 100.0

    if f"{driver_group}_net_pct_oi" not in ts.columns:
        # fallback to any computed group
        computed = [c for c in ts.columns if c.endswith("_net_pct_oi")]
        if not computed:
            return pd.DataFrame()
        driver_col = computed[0]
        driver_group = driver_col.replace("_net_pct_oi", "")
    else:
        driver_col = f"{driver_group}_net_pct_oi"

    ts["driver_group"] = driver_group
    ts["driver_net_pct_oi"] = ts[driver_col]
    ts["driver_net"] = ts.get(f"{driver_group}_net", np.nan)

    ts["net_pct_oi_3y_z"] = rolling_zscore(ts["driver_net_pct_oi"], rolling_weeks)
    ts["net_pct_oi_3y_pctile"] = rolling_percentile_of_last(ts["driver_net_pct_oi"], rolling_weeks)
    ts["net_pct_oi_4w_delta"] = ts["driver_net_pct_oi"].diff(delta_weeks)

    ts["bias"] = ts.apply(lambda r: bias_label(float(r["net_pct_oi_3y_z"]) if pd.notna(r["net_pct_oi_3y_z"]) else np.nan,
                                                float(r["net_pct_oi_3y_pctile"]) if pd.notna(r["net_pct_oi_3y_pctile"]) else np.nan), axis=1)
    ts["reversal_risk"] = ts["net_pct_oi_3y_pctile"].apply(lambda v: reversal_risk_label(float(v)) if pd.notna(v) else "UNKNOWN")

    ts["symbol"] = symbol
    # keep a clean output schema
    keep = [
        "symbol", "market", "report_date", "open_interest",
        "driver_group", "driver_net", "driver_net_pct_oi",
        "net_pct_oi_3y_z", "net_pct_oi_3y_pctile", "net_pct_oi_4w_delta",
        "bias", "reversal_risk",
    ]
    # include group nets for transparency
    for g in (TFF_GROUPS.keys() if dataset_type == "tff" else DISAGG_GROUPS.keys()):
        for suffix in ["_long", "_short", "_net", "_net_pct_oi"]:
            c = f"{g}{suffix}"
            if c in ts.columns:
                keep.append(c)

    return ts[keep].copy()

def latest_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # latest per symbol
    df2 = df.sort_values(["symbol", "report_date"]).groupby("symbol", as_index=False).tail(1)
    return df2.sort_values("symbol").reset_index(drop=True)

def compute_all(cfg: Config, fin_raw: pd.DataFrame, dis_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fin_norm = normalize_columns(fin_raw)
    dis_norm = normalize_columns(dis_raw)
    fin_type = detect_dataset_type(fin_norm)
    dis_type = detect_dataset_type(dis_norm)

    instruments: List[pd.DataFrame] = []
    for sym, pat in cfg.contract_patterns.items():
        if sym in {"GOLD", "SILVER"}:
            dfm = compute_instrument_metrics(dis_raw, dis_type, pat, cfg.metals_group, cfg.rolling_weeks, cfg.delta_weeks, sym)
        else:
            dfm = compute_instrument_metrics(fin_raw, fin_type, pat, cfg.fx_group, cfg.rolling_weeks, cfg.delta_weeks, sym)
        if not dfm.empty:
            instruments.append(dfm)

    inst_df = pd.concat(instruments, ignore_index=True) if instruments else pd.DataFrame()

    # FX pairs dashboard (derived)
    pairs_df = compute_pairs(latest_rows(inst_df))
    return inst_df, pairs_df

def compute_pairs(inst_latest: pd.DataFrame) -> pd.DataFrame:
    """Derive bias for common FX pairs using currency positioning scores.
    For USD-base pairs (USDJPY, USDCHF), we use quote currency positioning as proxy.
    """
    if inst_latest.empty:
        return pd.DataFrame()

    # Use z-score as the strength score
    score = {row["symbol"]: float(row["net_pct_oi_3y_z"]) if pd.notna(row["net_pct_oi_3y_z"]) else np.nan
             for _, row in inst_latest.iterrows()}
    pct = {row["symbol"]: float(row["net_pct_oi_3y_pctile"]) if pd.notna(row["net_pct_oi_3y_pctile"]) else np.nan
           for _, row in inst_latest.iterrows()}

    has_usd = ("USD" in score) and (not np.isnan(score.get("USD", np.nan)))

    def _pair(name: str, base: str, quote: str, usd_base: bool = False) -> Dict[str, object]:
        # If we have USD index positioning, compute a proper relative score.
        if usd_base:
            # USDJPY / USDCHF style.
            if has_usd:
                s = score.get("USD", np.nan) - score.get(quote, np.nan)
                pb = pct.get("USD", np.nan)
                pq = pct.get(quote, np.nan)
                p = np.nanmean([pb, 100.0 - pq]) if not (np.isnan(pb) and np.isnan(pq)) else np.nan
            else:
                # Fallback: proxy USD strength as the inverse of quote strength.
                s = -score.get(quote, np.nan)
                p = 100.0 - pct.get(quote, np.nan) if not np.isnan(pct.get(quote, np.nan)) else np.nan
        else:
            if quote == "USD" and not has_usd:
                # Fallback when USD index isn't tracked: use base currency only.
                s = score.get(base, np.nan)
                p = pct.get(base, np.nan)
            else:
                s = score.get(base, np.nan) - score.get(quote, np.nan)
                # percentile isn't linear under subtraction; we use mean as heuristic
                pb = pct.get(base, np.nan)
                pq = pct.get(quote, np.nan)
                p = np.nanmean([pb, 100.0 - pq]) if not (np.isnan(pb) and np.isnan(pq)) else np.nan

        bias = bias_label(s, p)
        risk = reversal_risk_label(p)
        return {
            "pair": name,
            "score_z": s,
            "pctile_proxy": p,
            "bias": bias,
            "reversal_risk": risk,
            "base": base,
            "quote": quote,
        }

    rows = [
        _pair("EURUSD", "EUR", "USD", usd_base=False),
        _pair("AUDUSD", "AUD", "USD", usd_base=False),
        _pair("USDJPY", "USD", "JPY", usd_base=True),
        _pair("USDCHF", "USD", "CHF", usd_base=True),
        _pair("GBPJPY", "GBP", "JPY", usd_base=False),
    ]

    return pd.DataFrame(rows)
