from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .cftc_official import load_years, extract_report_dates, ReportType
from .compute import compute_instrument_metrics
from .fx import DXY_WEIGHTS, pair_regime, pair_reversal_risk_from_abs_z, usd_proxy_from_z
from .utils import Config, most_recent_tuesday

BASKET_CURRENCIES = ["EUR", "JPY", "GBP", "AUD", "CAD", "CHF", "NZD"]


def resolve_report_date(
    requested_date: dt.date,
    available_report_dates: Iterable[dt.date],
    max_weeks: int = 104,
) -> dt.date:
    available = set(available_report_dates)
    candidate = requested_date
    for _ in range(max_weeks + 1):
        tuesday = most_recent_tuesday(candidate)
        if tuesday in available:
            return tuesday
        monday = tuesday - dt.timedelta(days=1)
        if monday in available:
            return monday
        candidate = tuesday - dt.timedelta(days=7)
    raise ValueError(
        f"No report_date found within {max_weeks} weeks at/before {requested_date.isoformat()}."
    )


def _years_for_window(target_date: dt.date, window_weeks: int, extra_weeks: int = 12) -> List[int]:
    start_date = target_date - dt.timedelta(weeks=window_weeks + extra_weeks)
    return list(range(start_date.year, target_date.year + 1))


def _parse_pairs(pairs_csv: str) -> List[Tuple[str, str, str]]:
    pairs: List[Tuple[str, str, str]] = []
    for raw in pairs_csv.split(","):
        p = raw.strip().upper()
        if not p:
            continue
        if len(p) != 6:
            raise ValueError(f"Invalid pair '{p}'. Expected format like EURUSD.")
        base, quote = p[:3], p[3:]
        pairs.append((p, base, quote))
    if not pairs:
        raise ValueError("No FX pairs provided.")
    return pairs


def _currency_symbols_for_pairs(pairs: Sequence[Tuple[str, str, str]]) -> List[str]:
    syms = set()
    for _, base, quote in pairs:
        syms.add(base)
        syms.add(quote)
    return sorted(syms)


def _build_currency_timeseries(
    df_raw: pd.DataFrame,
    dataset_type: str,
    cfg: Config,
    symbols: Sequence[str],
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for sym in symbols:
        pat = cfg.contract_patterns.get(sym)
        if not pat:
            continue
        dfm = compute_instrument_metrics(
            df_raw=df_raw,
            dataset_type=dataset_type,
            pattern=pat,
            driver_group=cfg.fx_group,
            rolling_weeks=cfg.rolling_weeks,
            delta_weeks=cfg.delta_weeks,
            symbol=sym,
        )
        if not dfm.empty:
            frames.append(dfm)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _pivot_zscores(inst_df: pd.DataFrame) -> pd.DataFrame:
    df = inst_df[["report_date", "symbol", "net_pct_oi_3y_z"]].copy()
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce").dt.date
    return df.pivot_table(index="report_date", columns="symbol", values="net_pct_oi_3y_z", aggfunc="last")


def _usd_series_from_pivot(
    pivot: pd.DataFrame,
    usd_mode: str,
    usd_weights: str,
) -> Tuple[pd.Series, str]:
    weights = None if usd_weights == "equal" else DXY_WEIGHTS
    basket = pivot.apply(lambda row: usd_proxy_from_z(row.to_dict(), weights=weights), axis=1)

    if usd_mode == "direct" and "USD" in pivot.columns:
        direct = pivot["USD"]
        usd_series = direct.copy()
        missing = direct.isna()
        if missing.any():
            usd_series[missing] = basket[missing]
            return usd_series, "direct_fallback"
        return usd_series, "direct"

    return basket, "basket"


def run_report(
    cfg: Config,
    requested_date: dt.date,
    pairs_csv: str,
    report_type: ReportType,
    usd_mode: str = "basket",
    usd_weights: str = "equal",
    refresh: bool = False,
    verbose: bool = False,
) -> Dict[str, object]:
    pairs = _parse_pairs(pairs_csv)
    if report_type == "disagg":
        raise ValueError("Disaggregated futures-only report does not include FX currencies.")

    years = _years_for_window(requested_date, cfg.rolling_weeks)
    base_dir = Path("data") / "cftc_cache"
    df_raw = load_years(report_type, years, base_dir, refresh=refresh, verbose=verbose)
    if df_raw.empty:
        raise ValueError("No data loaded from CFTC official sources.")

    report_dates = extract_report_dates(df_raw)
    resolved = resolve_report_date(requested_date, report_dates)

    dataset_type = report_type
    # Build the currency universe: pairs + basket currencies (+ USD for direct)
    symbols = set(_currency_symbols_for_pairs(pairs))
    symbols.update([c for c in BASKET_CURRENCIES if c in cfg.contract_patterns])
    if usd_mode == "direct":
        symbols.add("USD")

    inst_df = _build_currency_timeseries(df_raw, dataset_type, cfg, sorted(symbols))
    if inst_df.empty:
        raise ValueError("No matching currency contracts found for requested pairs.")

    pivot = _pivot_zscores(inst_df)
    usd_series, usd_mode_used = _usd_series_from_pivot(pivot, usd_mode, usd_weights)

    if resolved not in pivot.index:
        raise ValueError(f"Resolved report_date {resolved.isoformat()} not present in computed series.")

    row = pivot.loc[resolved]
    z_by_ccy: Dict[str, float] = {k: (float(v) if pd.notna(v) else float("nan")) for k, v in row.items()}
    usd_z = float(usd_series.loc[resolved]) if resolved in usd_series.index else float("nan")

    results = []
    for pair, base, quote in pairs:
        base_z = usd_z if base == "USD" else z_by_ccy.get(base, float("nan"))
        quote_z = usd_z if quote == "USD" else z_by_ccy.get(quote, float("nan"))
        pair_z = base_z - quote_z if not (np.isnan(base_z) or np.isnan(quote_z)) else float("nan")
        results.append(
            {
                "pair": pair,
                "base": base,
                "quote": quote,
                "base_z": base_z,
                "quote_z": quote_z,
                "pair_z": pair_z,
                "regime": pair_regime(pair_z),
                "reversal_risk": pair_reversal_risk_from_abs_z(abs(pair_z)) if not np.isnan(pair_z) else "UNKNOWN",
            }
        )

    return {
        "requested_date": requested_date.isoformat(),
        "resolved_report_date": resolved.isoformat(),
        "report_type": report_type,
        "usd_mode_requested": usd_mode,
        "usd_mode_used": usd_mode_used,
        "note": "COT positions are as-of Tuesday and released Friday.",
        "pairs": results,
    }


def format_report_output(report: Dict[str, object], out: str, out_dir: Path) -> str:
    if out == "json":
        return json.dumps(report, indent=2)
    if out != "html":
        raise ValueError("out must be json or html")

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "report.html"

    rows = report.get("pairs", [])
    df = pd.DataFrame(rows)
    for c in ["base_z", "quote_z", "pair_z"]:
        if c in df.columns:
            df[c] = df[c].map(lambda x: "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}")

    parts = []
    parts.append("<html><head><meta charset='utf-8'><title>COT Report</title>")
    parts.append("<style>body{font-family:Arial,Helvetica,sans-serif;padding:16px;} table{border-collapse:collapse;margin:12px 0;} th,td{border:1px solid #ddd;padding:6px 8px;font-size:12px;} th{background:#f5f5f5;} h2{margin-top:24px;}</style>")
    parts.append("</head><body>")
    parts.append("<h1>COT FX Report</h1>")
    parts.append(f"<p><strong>Requested date:</strong> {report['requested_date']}</p>")
    parts.append(f"<p><strong>Resolved report date:</strong> {report['resolved_report_date']}</p>")
    parts.append(f"<p><strong>USD mode:</strong> {report['usd_mode_used']}</p>")
    parts.append(f"<p><em>{report['note']}</em></p>")
    parts.append("<h2>Pairs</h2>")
    parts.append(df.to_html(index=False, escape=True))
    parts.append("</body></html>")

    path.write_text("\n".join(parts), encoding="utf-8")
    return str(path)
