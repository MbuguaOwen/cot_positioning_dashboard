from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd


G10_CCY = ["USD", "EUR", "JPY", "GBP", "CHF", "AUD", "NZD", "CAD"]
VALID_TFS = {"W1", "D1", "H4", "H1"}
_CCY_PATTERN = "(?:USD|EUR|JPY|GBP|CHF|AUD|NZD|CAD)"


def clean_pair(value: str) -> str:
    s = re.sub(r"[^A-Za-z]", "", str(value or "")).upper()
    if len(s) >= 6:
        return s[:6]
    return ""


def pair_parts(pair: str) -> Tuple[str, str]:
    p = clean_pair(pair)
    if len(p) == 6:
        return p[:3], p[3:]
    return "", ""


def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [re.sub(r"[^a-zA-Z0-9]+", "_", str(c)).strip("_").lower() for c in out.columns]
    return out


def pick_col(df: pd.DataFrame, names: Iterable[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in names:
        if c in cols:
            return c
    return None


def to_ts(series: pd.Series) -> pd.Series:
    s = series
    parsed = pd.to_datetime(s, errors="coerce", utc=True)
    s_num = pd.to_numeric(s, errors="coerce")
    non_null = s_num.dropna()
    if non_null.empty:
        return parsed

    med = float(non_null.abs().median())
    if med > 1e12:
        epoch = pd.to_datetime(s_num, unit="ms", utc=True, errors="coerce")
    elif med > 1e9:
        epoch = pd.to_datetime(s_num, unit="s", utc=True, errors="coerce")
    else:
        return parsed
    return parsed.where(s_num.isna(), epoch)


def parse_as_of(value: Optional[str]) -> pd.Timestamp:
    if value is None:
        return pd.Timestamp.now(tz="UTC")
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def safe_float(v: object) -> float:
    try:
        out = float(v)
        if math.isnan(out):
            return float("nan")
        return out
    except Exception:
        return float("nan")


def load_generic_csv(path: Optional[Path]) -> pd.DataFrame:
    if path is None or (not path.exists()) or (not path.is_file()):
        return pd.DataFrame()
    try:
        return norm_cols(pd.read_csv(path, sep=None, engine="python"))
    except Exception:
        return pd.DataFrame()


def _pair_tf_from_filename(path: Path) -> Tuple[str, str]:
    stem = path.stem.upper()
    pair_match = re.search(rf"({_CCY_PATTERN})({_CCY_PATTERN})", stem)
    tf_match = re.search(r"(?:^|[_\-\s])(W1|D1|H4|H1)(?:$|[_\-\s])", stem)
    if tf_match is None:
        tf_match = re.search(r"(W1|D1|H4|H1)", stem)
    pair = f"{pair_match.group(1)}{pair_match.group(2)}" if pair_match else ""
    tf = tf_match.group(1) if tf_match else ""
    return pair, tf


def load_price_bars(path: Optional[Path], timeframes: Iterable[str]) -> pd.DataFrame:
    cols = ["pair", "timeframe", "ts", "open", "high", "low", "close", "volume", "source"]
    if path is None:
        return pd.DataFrame(columns=cols)
    files = sorted(path.rglob("*.csv")) if path.is_dir() else [path] if path.is_file() else []
    allowed_tf = {str(tf).upper() for tf in timeframes}
    rows = []
    for fp in files:
        try:
            raw = pd.read_csv(fp, sep=None, engine="python")
        except Exception:
            continue
        df = norm_cols(raw)
        ts_col = pick_col(df, ["ts", "timestamp", "datetime", "date", "time"])
        close_col = pick_col(df, ["close", "c"])
        if ts_col is None or close_col is None:
            continue
        pair_col = pick_col(df, ["pair", "symbol", "instrument"])
        tf_col = pick_col(df, ["timeframe", "tf"])
        p_file, tf_file = _pair_tf_from_filename(fp)
        out = pd.DataFrame()
        out["ts"] = to_ts(df[ts_col])
        out["close"] = pd.to_numeric(df[close_col], errors="coerce")
        open_col = pick_col(df, ["open", "o"])
        high_col = pick_col(df, ["high", "h"])
        low_col = pick_col(df, ["low", "l"])
        vol_col = pick_col(df, ["volume", "vol", "tick_volume"])
        out["open"] = pd.to_numeric(df[open_col], errors="coerce") if open_col else out["close"]
        out["high"] = pd.to_numeric(df[high_col], errors="coerce") if high_col else out[["open", "close"]].max(axis=1)
        out["low"] = pd.to_numeric(df[low_col], errors="coerce") if low_col else out[["open", "close"]].min(axis=1)
        out["volume"] = pd.to_numeric(df[vol_col], errors="coerce") if vol_col else 0.0
        out["pair"] = df[pair_col].astype(str).apply(clean_pair) if pair_col else p_file
        out["timeframe"] = df[tf_col].astype(str).str.upper().str.strip() if tf_col else tf_file
        out["source"] = fp.name
        out = out.dropna(subset=["ts", "close"])
        out = out[(out["pair"].str.len() == 6) & (out["timeframe"].isin(allowed_tf))]
        if not out.empty:
            rows.append(out)
    if not rows:
        return pd.DataFrame(columns=cols)
    ret = pd.concat(rows, ignore_index=True)
    return ret.sort_values(["pair", "timeframe", "ts"]).reset_index(drop=True)


def _resample_ohlcv(
    source_df: pd.DataFrame,
    *,
    pair: str,
    source_tf: str,
    target_tf: str,
    as_of_ts: pd.Timestamp,
) -> pd.DataFrame:
    cols = ["pair", "timeframe", "ts", "open", "high", "low", "close", "volume", "source"]
    if source_df.empty:
        return pd.DataFrame(columns=cols)
    s = source_df.sort_values("ts").copy()
    s = s.dropna(subset=["ts", "open", "high", "low", "close"])
    if s.empty:
        return pd.DataFrame(columns=cols)
    s = s.set_index("ts")

    if target_tf == "D1":
        rule = "1D"
        label = "left"
        closed = "left"
        min_count = 12 if source_tf == "H1" else 4
    elif target_tf == "W1":
        rule = "W-FRI"
        label = "right"
        closed = "right"
        min_count = 4
    else:
        return pd.DataFrame(columns=cols)

    agg = s.resample(rule, label=label, closed=closed).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    counts = s["close"].resample(rule, label=label, closed=closed).count().rename("bar_count")
    out = agg.join(counts, how="left")
    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out[out["bar_count"] >= min_count]
    out = out.reset_index()

    if target_tf == "D1":
        out = out[out["ts"] < as_of_ts.floor("D")]
    elif target_tf == "W1":
        out = out[out["ts"] < as_of_ts.floor("D")]

    if out.empty:
        return pd.DataFrame(columns=cols)

    out["pair"] = pair
    out["timeframe"] = target_tf
    out["source"] = f"resampled_from_{source_tf}"
    out = out.drop(columns=["bar_count"], errors="ignore")
    return out[cols].copy()


def augment_price_timeframes(price_bars: pd.DataFrame, as_of_ts: pd.Timestamp) -> Tuple[pd.DataFrame, List[str]]:
    cols = ["pair", "timeframe", "ts", "open", "high", "low", "close", "volume", "source"]
    notes: List[str] = []
    if price_bars.empty:
        return pd.DataFrame(columns=cols), notes

    bars = price_bars.copy()
    bars = bars[bars["timeframe"].isin(VALID_TFS)].copy()
    additions: List[pd.DataFrame] = []

    for pair, grp in bars.groupby("pair", sort=False):
        pair_df = grp.copy()
        tf_set = set(pair_df["timeframe"].astype(str).str.upper().tolist())

        if "D1" not in tf_set:
            src_tf = "H1" if "H1" in tf_set else "H4" if "H4" in tf_set else ""
            if src_tf:
                src = pair_df[pair_df["timeframe"] == src_tf].copy()
                d1 = _resample_ohlcv(src, pair=str(pair), source_tf=src_tf, target_tf="D1", as_of_ts=as_of_ts)
                if not d1.empty:
                    additions.append(d1)
                    notes.append(f"PRICE D1 AUTO-BUILT: {pair} from {src_tf}")
                    tf_set.add("D1")

        if "W1" not in tf_set:
            if "D1" in tf_set:
                src_tf = "D1"
                src = pd.concat(
                    [
                        pair_df[pair_df["timeframe"] == "D1"].copy(),
                        *[a[(a["pair"] == pair) & (a["timeframe"] == "D1")] for a in additions],
                    ],
                    ignore_index=True,
                )
            else:
                src_tf = "H4" if "H4" in tf_set else "H1" if "H1" in tf_set else ""
                src = pair_df[pair_df["timeframe"] == src_tf].copy() if src_tf else pd.DataFrame()
            if src_tf and not src.empty:
                w1 = _resample_ohlcv(src, pair=str(pair), source_tf=src_tf, target_tf="W1", as_of_ts=as_of_ts)
                if not w1.empty:
                    additions.append(w1)
                    notes.append(f"PRICE W1 AUTO-BUILT: {pair} from {src_tf}")

    if additions:
        bars = pd.concat([bars] + additions, ignore_index=True)
        bars = bars.sort_values(["pair", "timeframe", "ts", "source"]).drop_duplicates(
            subset=["pair", "timeframe", "ts"], keep="last"
        )
    return bars[cols].sort_values(["pair", "timeframe", "ts"]).reset_index(drop=True), notes


def load_rates(path: Optional[Path]) -> pd.DataFrame:
    df = load_generic_csv(path)
    cols = ["currency", "ts", "y2_value", "source"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    c_col = pick_col(df, ["currency", "ccy", "symbol"])
    ts_col = pick_col(df, ["ts", "timestamp", "datetime", "date", "time"])
    y_col = pick_col(df, ["y2_value", "y2", "yield_2y", "yield", "value", "rate"])
    src_col = pick_col(df, ["source", "provider"])
    if c_col is None or ts_col is None or y_col is None:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame()
    out["currency"] = df[c_col].astype(str).str.upper().str.strip()
    out["ts"] = to_ts(df[ts_col])
    out["y2_value"] = pd.to_numeric(df[y_col], errors="coerce")
    out["source"] = df[src_col].astype(str) if src_col else ""
    out = out[out["currency"].isin(G10_CCY)].dropna(subset=["ts", "y2_value"])
    return out.sort_values(["currency", "ts"]).reset_index(drop=True)


def load_risk(path: Optional[Path]) -> pd.DataFrame:
    df = load_generic_csv(path)
    cols = ["asset", "ts", "value", "source"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    a_col = pick_col(df, ["asset", "symbol", "proxy"])
    ts_col = pick_col(df, ["ts", "timestamp", "datetime", "date", "time"])
    v_col = pick_col(df, ["value", "close", "price", "level", "v"])
    src_col = pick_col(df, ["source", "provider"])
    if a_col is None or ts_col is None or v_col is None:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame()
    out["asset"] = df[a_col].astype(str).str.upper().str.strip()
    out["ts"] = to_ts(df[ts_col])
    out["value"] = pd.to_numeric(df[v_col], errors="coerce")
    out["source"] = df[src_col].astype(str) if src_col else ""
    out = out.dropna(subset=["asset", "ts", "value"])
    return out.sort_values(["asset", "ts"]).reset_index(drop=True)


def load_carry(path: Optional[Path]) -> pd.DataFrame:
    df = load_generic_csv(path)
    cols = ["pair", "ts", "forward_points", "source"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    p_col = pick_col(df, ["pair", "symbol", "instrument"])
    ts_col = pick_col(df, ["ts", "timestamp", "datetime", "date", "time"])
    f_col = pick_col(df, ["forward_points", "fwd_points", "swap", "value"])
    src_col = pick_col(df, ["source", "provider"])
    if p_col is None or ts_col is None or f_col is None:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame()
    out["pair"] = df[p_col].astype(str).apply(clean_pair)
    out["ts"] = to_ts(df[ts_col])
    out["forward_points"] = pd.to_numeric(df[f_col], errors="coerce")
    out["source"] = df[src_col].astype(str) if src_col else ""
    out = out[(out["pair"].str.len() == 6)].dropna(subset=["ts", "forward_points"])
    return out.sort_values(["pair", "ts"]).reset_index(drop=True)


def load_skew(path: Optional[Path]) -> pd.DataFrame:
    df = load_generic_csv(path)
    cols = ["symbol_or_ccy", "ts", "rr25", "quality_flags", "source"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    s_col = pick_col(df, ["symbol_or_ccy", "symbol", "currency", "pair"])
    ts_col = pick_col(df, ["ts", "timestamp", "datetime", "date", "time"])
    r_col = pick_col(df, ["rr25", "rr_25", "risk_reversal_25", "value"])
    q_col = pick_col(df, ["quality_flags", "flags", "quality"])
    src_col = pick_col(df, ["source", "provider"])
    if s_col is None or ts_col is None or r_col is None:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame()
    out["symbol_or_ccy"] = df[s_col].astype(str).str.upper().str.strip()
    out["ts"] = to_ts(df[ts_col])
    out["rr25"] = pd.to_numeric(df[r_col], errors="coerce")
    out["quality_flags"] = df[q_col].astype(str) if q_col else ""
    out["source"] = df[src_col].astype(str) if src_col else ""
    out = out.dropna(subset=["ts", "rr25"])
    return out.sort_values(["symbol_or_ccy", "ts"]).reset_index(drop=True)


def load_cot_flags(path: Optional[Path]) -> pd.DataFrame:
    df = load_generic_csv(path)
    cols = ["currency", "ts", "extreme_flag", "persistence_flag", "source"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    c_col = pick_col(df, ["currency", "ccy", "symbol"])
    ts_col = pick_col(df, ["ts", "timestamp", "datetime", "date", "time"])
    e_col = pick_col(df, ["extreme_flag", "cot_extremeflag", "extreme"])
    p_col = pick_col(df, ["persistence_flag", "cot_persistenceflag", "persistence"])
    src_col = pick_col(df, ["source", "provider"])
    if c_col is None or e_col is None or p_col is None:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame()
    out["currency"] = df[c_col].astype(str).str.upper().str.strip()
    out["ts"] = to_ts(df[ts_col]) if ts_col else pd.NaT
    out["extreme_flag"] = df[e_col].astype(str).str.lower().isin({"1", "true", "yes", "y"})
    out["persistence_flag"] = df[p_col].astype(str).str.lower().isin({"1", "true", "yes", "y"})
    out["source"] = df[src_col].astype(str) if src_col else ""
    return out[out["currency"].isin(G10_CCY)].reset_index(drop=True)
