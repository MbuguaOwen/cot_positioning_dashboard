from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

from ..filters.cot_filter import COTFilter, COTFilterInput, default_cot_filter_config
from ..fx import build_pairs_df, usd_proxy_from_z
from ..reporting import (
    BASKET_CURRENCIES,
    build_cot_components_table,
    build_crowdedness_policy_table,
    build_decision_summary_table,
    build_macro_gate_table,
    build_pine_join_table,
    build_spread_flow_table,
    build_tier_thresholds_table,
    resolve_report_date_by_release,
)


def _release_dt_for_report(report_date: dt.date, release_cfg: Dict[str, Any]) -> dt.datetime:
    wd_map = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }
    target = wd_map.get(str(release_cfg.get("release_weekday", "friday")).strip().lower())
    if target is None:
        raise ValueError(f"Invalid release_weekday: {release_cfg.get('release_weekday')}")

    wd = report_date.weekday()
    delta = (target - wd) % 7
    if delta == 0:
        delta = 7
    release_date = report_date + dt.timedelta(days=delta)

    hour, minute = [int(x) for x in str(release_cfg.get("release_time", "15:30")).split(":")]
    tz = ZoneInfo(release_cfg.get("timezone", "America/New_York"))
    return dt.datetime(
        release_date.year,
        release_date.month,
        release_date.day,
        hour,
        minute,
        tzinfo=tz,
    ).astimezone(dt.timezone.utc)


def _add_usd_proxy(panel: pd.DataFrame, basket: Sequence[str]) -> pd.DataFrame:
    if "USD" in set(panel["currency"]):
        return panel

    basket = [c for c in basket if c in set(panel["currency"])]
    if not basket:
        return panel

    px = panel[panel["currency"].isin(basket)].copy()
    if px.empty:
        return panel

    agg = (
        px.groupby("report_date", as_index=False)
        .agg(
            net_pctile=("net_pctile", "mean"),
            net_z=("net_z", "mean"),
            net_pct_oi=("net_pct_oi", "mean"),
            release_dt=("release_dt", "first"),
        )
        .copy()
    )
    agg["currency"] = "USD"
    agg["net_pctile"] = 100.0 - agg["net_pctile"]
    agg["net_z"] = -agg["net_z"]
    agg["net_pct_oi"] = -agg["net_pct_oi"]

    cols = ["currency", "report_date", "net_pctile", "net_z", "net_pct_oi", "release_dt"]
    return pd.concat([panel, agg[cols]], ignore_index=True)


def _metric_at(panel: pd.DataFrame, currency: str, report_date: dt.date, metric: str) -> float:
    rows = panel[(panel["currency"] == currency) & (panel["report_date"] == report_date)]
    if rows.empty or metric not in rows.columns:
        return float("nan")
    return float(rows.iloc[-1][metric])


def _spread_at(panel: pd.DataFrame, base: str, quote: str, report_date: dt.date, metric: str) -> float:
    base_v = _metric_at(panel, base, report_date, metric)
    quote_v = _metric_at(panel, quote, report_date, metric)
    if np.isnan(base_v) or np.isnan(quote_v):
        return float("nan")
    return float(base_v - quote_v)


def _dspread_at(
    panel: pd.DataFrame,
    base: str,
    quote: str,
    report_date: dt.date,
    metric: str,
    lag: int,
) -> float:
    if lag <= 0:
        return float("nan")
    dates = sorted(panel["report_date"].dropna().unique().tolist())
    if report_date not in dates:
        return float("nan")
    idx = dates.index(report_date)
    if idx - lag < 0:
        return float("nan")
    prev = dates[idx - lag]
    s_now = _spread_at(panel, base, quote, report_date, metric)
    s_prev = _spread_at(panel, base, quote, prev, metric)
    if np.isnan(s_now) or np.isnan(s_prev):
        return float("nan")
    return float(s_now - s_prev)


def compute_tiered_tables(
    history_df: pd.DataFrame,
    pairs_df: Optional[pd.DataFrame],
    as_of: Optional[dt.date],
    cot_filter_cfg: Optional[Dict[str, Any]] = None,
    report_date_override: Optional[dt.date] = None,
) -> tuple[Optional[Dict[str, pd.DataFrame]], Optional[Dict[str, Any]]]:
    if pairs_df is None or pairs_df.empty:
        pairs_df = None
    if history_df is None or history_df.empty:
        return None, None

    required = {"symbol", "report_date", "net_pct_oi", "z_3y", "pctile_3y"}
    if not required.issubset(set(history_df.columns)):
        return None, None

    cot_filter = COTFilter(cot_filter_cfg)
    filter_cfg = cot_filter.cfg
    release_cfg = filter_cfg.get("release_alignment", default_cot_filter_config()["release_alignment"])

    panel = history_df.copy()
    if "report_type" in panel.columns:
        panel = panel[panel["report_type"] == "tff"].copy()
    if panel.empty:
        return None, None

    panel = panel.rename(
        columns={
            "symbol": "currency",
            "z_3y": "net_z",
            "pctile_3y": "net_pctile",
        }
    )
    panel["currency"] = panel["currency"].astype(str).str.upper()
    panel["report_date"] = pd.to_datetime(panel["report_date"], errors="coerce").dt.date
    panel = panel.dropna(subset=["currency", "report_date"])
    if panel.empty:
        return None, None

    panel["release_dt"] = panel["report_date"].apply(lambda d: _release_dt_for_report(d, release_cfg))
    panel = _add_usd_proxy(panel, BASKET_CURRENCIES)

    dates = sorted(panel["report_date"].dropna().unique().tolist())
    if not dates:
        return None, None

    if report_date_override is not None:
        if report_date_override not in dates:
            raise RuntimeError(
                f"report_date_override {report_date_override} not available in history."
            )
        use_report = report_date_override
    else:
        if as_of is None:
            as_of = dates[-1]
        use_report = resolve_report_date_by_release(as_of, dates, release_cfg)

    release_rows = panel[panel["report_date"] == use_report]
    if not release_rows.empty:
        release_dt = release_rows.iloc[-1]["release_dt"]
    else:
        release_dt = dt.datetime.combine(use_report, dt.time(0, 0), tzinfo=dt.timezone.utc)
    signal_ts = release_dt + dt.timedelta(minutes=1)

    metric_kind = str(filter_cfg["metric"]["kind"]).strip().lower()
    tier = str(filter_cfg["strictness_tier"]).strip().lower()

    pair_contexts = []
    fx_release = history_df.copy()
    if "report_type" in fx_release.columns:
        fx_release = fx_release[fx_release["report_type"] == "tff"].copy()
    fx_release["report_date"] = pd.to_datetime(fx_release["report_date"], errors="coerce").dt.date
    fx_release = fx_release[fx_release["report_date"] == use_report].copy()

    if not fx_release.empty and "z_3y" in fx_release.columns:
        z_by = {
            row["symbol"]: float(row["z_3y"]) if pd.notna(row["z_3y"]) else float("nan")
            for _, row in fx_release.iterrows()
        }
        usd_z = usd_proxy_from_z(z_by, weights=None)
        pairs = build_pairs_df(z_by, usd_z, usd_mode="basket")
    else:
        if pairs_df is None or pairs_df.empty:
            return None, None
        pairs = pairs_df.copy()

    if pairs.empty:
        return None, None

    pair_required = {"pair", "base", "quote"}
    if not pair_required.issubset(set(pairs.columns)):
        return None, None

    pairs["pair"] = pairs["pair"].astype(str).str.upper()
    pairs["base"] = pairs["base"].astype(str).str.upper()
    pairs["quote"] = pairs["quote"].astype(str).str.upper()
    if "pair_z" in pairs.columns:
        pairs["pair_z"] = pd.to_numeric(pairs["pair_z"], errors="coerce")

    for _, row in pairs.iterrows():
        pair = str(row["pair"]).strip()
        if len(pair) != 6:
            continue
        base = pair[:3]
        quote = pair[3:]
        if not pd.isna(row["base"]):
            base = str(row["base"]).strip() or base
        if not pd.isna(row["quote"]):
            quote = str(row["quote"]).strip() or quote
        pz = row["pair_z"] if "pair_z" in row else float("nan")
        if isinstance(pz, float) and not np.isnan(pz):
            direction = "long" if pz >= 0 else "short"
        else:
            direction = "long"

        decision = cot_filter.evaluate(
            COTFilterInput(pair=pair, signal_direction=direction, signal_ts=signal_ts),
            panel,
        )
        spread_val = _spread_at(panel, base, quote, use_report, metric_kind)
        dspread_1w = _dspread_at(panel, base, quote, use_report, metric_kind, 1)
        base_metric = _metric_at(panel, base, use_report, metric_kind)
        quote_metric = _metric_at(panel, quote, use_report, metric_kind)
        base_pctile = _metric_at(panel, base, use_report, "net_pctile")
        quote_pctile = _metric_at(panel, quote, use_report, "net_pctile")

        pair_contexts.append(
            {
                "pair": pair,
                "base": base,
                "quote": quote,
                "signal_direction": direction,
                "decision": decision,
                "spread": spread_val,
                "dspread_1w": dspread_1w,
                "base_metric": base_metric,
                "quote_metric": quote_metric,
                "base_pctile": base_pctile,
                "quote_pctile": quote_pctile,
            }
        )

    inst_df = history_df.copy()
    if "report_type" in inst_df.columns:
        inst_df = inst_df[inst_df["report_type"] == "tff"].copy()
    inst_df = inst_df.rename(
        columns={
            "net_pct_oi": "driver_net_pct_oi",
            "z_3y": "net_pct_oi_3y_z",
            "pctile_3y": "net_pct_oi_3y_pctile",
        }
    )
    for col in [
        "symbol",
        "report_date",
        "driver_net",
        "driver_net_pct_oi",
        "net_pct_oi_3y_pctile",
        "net_pct_oi_3y_z",
        "open_interest",
    ]:
        if col not in inst_df.columns:
            inst_df[col] = np.nan

    decisions_summary = build_decision_summary_table(use_report, tier, pair_contexts)
    cot_components = build_cot_components_table(use_report, filter_cfg, inst_df)
    spread_flow = build_spread_flow_table(use_report, filter_cfg, pair_contexts)
    crowdedness_policy = build_crowdedness_policy_table(use_report, filter_cfg, pair_contexts)
    macro_gate = build_macro_gate_table(use_report, filter_cfg, pair_contexts)
    tier_thresholds = build_tier_thresholds_table(filter_cfg)
    pine_join = build_pine_join_table(signal_ts, use_report, tier, pair_contexts)

    tables = {
        "decisions_summary": decisions_summary,
        "cot_components": cot_components,
        "spread_flow": spread_flow,
        "crowdedness_policy": crowdedness_policy,
        "macro_gate": macro_gate,
        "tier_thresholds": tier_thresholds,
        "pine_join": pine_join,
    }

    meta = {
        "resolved_report_date": use_report.isoformat(),
        "resolved_release_dt": release_dt.isoformat(),
        "signal_ts": signal_ts.isoformat(),
        "tier_mode": tier.upper(),
        "metric_used": metric_kind,
    }

    return tables, meta
