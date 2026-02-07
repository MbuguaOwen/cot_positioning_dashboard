from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from zoneinfo import ZoneInfo

from .cftc_official import load_years, extract_report_dates, ReportType
from .compute import compute_instrument_metrics
from .filters.cot_filter import COTFilter, COTFilterInput, default_cot_filter_config
from .fx import DXY_WEIGHTS, pair_regime, pair_reversal_risk_from_abs_z, usd_proxy_from_z
from .utils import Config, most_recent_tuesday

BASKET_CURRENCIES = ["EUR", "JPY", "GBP", "AUD", "CAD", "CHF", "NZD"]


def _metric_col_from_kind(kind: str) -> str:
    kind = str(kind).strip().lower()
    if kind == "net_pctile":
        return "net_pct_oi_3y_pctile"
    if kind == "net_z":
        return "net_pct_oi_3y_z"
    if kind == "net_pct_oi":
        return "driver_net_pct_oi"
    raise ValueError(f"Unsupported metric kind: {kind}")


def _release_dt_for_report(
    report_date: dt.date,
    release_tz: str,
    release_weekday: str,
    release_time: str,
) -> dt.datetime:
    # Convert weekday name to Python weekday int.
    wd_map = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }
    target = wd_map.get(str(release_weekday).strip().lower())
    if target is None:
        raise ValueError(f"Invalid release_weekday: {release_weekday}")

    wd = report_date.weekday()
    delta = (target - wd) % 7
    if delta == 0:
        delta = 7
    release_date = report_date + dt.timedelta(days=delta)

    hour, minute = [int(x) for x in str(release_time).split(":")]
    tz = ZoneInfo(release_tz)
    return dt.datetime(
        release_date.year,
        release_date.month,
        release_date.day,
        hour,
        minute,
        tzinfo=tz,
    ).astimezone(dt.timezone.utc)


def resolve_report_date_by_release(
    requested_date: dt.date,
    available_report_dates: Iterable[dt.date],
    release_cfg: Dict[str, Any],
) -> dt.date:
    report_dates = sorted(set(available_report_dates))
    if not report_dates:
        raise ValueError("No report_date values available.")

    release_tz = release_cfg.get("timezone", "America/New_York")
    release_weekday = release_cfg.get("release_weekday", "friday")
    release_time = release_cfg.get("release_time", "15:30")

    # Treat requested_date as end-of-day in release timezone to avoid lookahead.
    tz = ZoneInfo(release_tz)
    requested_dt = dt.datetime(
        requested_date.year,
        requested_date.month,
        requested_date.day,
        23,
        59,
        59,
        tzinfo=tz,
    ).astimezone(dt.timezone.utc)

    eligible: List[dt.date] = []
    for rd in report_dates:
        rel = _release_dt_for_report(rd, release_tz, release_weekday, release_time)
        if rel <= requested_dt:
            eligible.append(rd)

    if not eligible:
        raise ValueError("No report_date has a release_dt <= requested_date.")
    return max(eligible)


def _parse_signals(signals_csv: Optional[str]) -> Dict[str, str]:
    if not signals_csv:
        return {}
    out: Dict[str, str] = {}
    for raw in signals_csv.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if "=" not in raw:
            raise ValueError(f"Invalid signal '{raw}'. Expected PAIR=long|short.")
        pair, direction = raw.split("=", 1)
        pair = pair.strip().upper()
        direction = direction.strip().lower()
        if direction not in {"long", "short"}:
            raise ValueError(f"Invalid direction for {pair}: {direction}")
        out[pair] = direction
    return out


def _primary_blocked_by(reasons: Sequence[str]) -> Optional[str]:
    priority = [
        "direction_fail",
        "spread_fail",
        "flow_fail",
        "flow_accel_fail",
        "crowded_block",
        "crowded_override_fail",
        "macro_block",
        "macro_misaligned",
        "macro_reversal_fail",
        "news_blackout_block",
        "score_gate_fail",
        "no_report_released_yet",
    ]
    for key in priority:
        if key in reasons:
            return key
    return None


def _crowded_flag(crowded_base: Optional[bool], crowded_quote: Optional[bool]) -> str:
    if crowded_base and crowded_quote:
        return "BOTH_EXTREMES"
    if crowded_base:
        return "BASE_CROWDED"
    if crowded_quote:
        return "QUOTE_CROWDED"
    return "NONE"


def _macro_ok_from_reasons(reasons: Sequence[str]) -> bool:
    if "macro_block" in reasons or "macro_misaligned" in reasons or "macro_reversal_fail" in reasons:
        return False
    return True


def _news_ok_from_reasons(reasons: Sequence[str]) -> bool:
    if "news_blackout_block" in reasons:
        return False
    return True


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


def _build_filter_panel(
    inst_df: pd.DataFrame,
    release_cfg: Dict[str, Any],
) -> pd.DataFrame:
    panel = inst_df[
        [
            "symbol",
            "report_date",
            "net_pct_oi_3y_pctile",
            "net_pct_oi_3y_z",
            "driver_net_pct_oi",
        ]
    ].copy()
    panel = panel.rename(
        columns={
            "symbol": "currency",
            "net_pct_oi_3y_pctile": "net_pctile",
            "net_pct_oi_3y_z": "net_z",
            "driver_net_pct_oi": "net_pct_oi",
        }
    )
    panel["currency"] = panel["currency"].astype(str).str.upper()
    panel["report_date"] = pd.to_datetime(panel["report_date"], errors="coerce").dt.date

    tz = release_cfg.get("timezone", "America/New_York")
    weekday = release_cfg.get("release_weekday", "friday")
    release_time = release_cfg.get("release_time", "15:30")
    panel["release_dt"] = panel["report_date"].apply(
        lambda d: _release_dt_for_report(d, tz, weekday, release_time)
    )

    return panel.dropna(subset=["currency", "report_date"]).copy()


def _add_usd_proxy(panel: pd.DataFrame) -> pd.DataFrame:
    if "USD" in set(panel["currency"]):
        return panel

    basket = [c for c in BASKET_CURRENCIES if c in set(panel["currency"])]
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

    return pd.concat(
        [
            panel,
            agg[["currency", "report_date", "net_pctile", "net_z", "net_pct_oi", "release_dt"]],
        ],
        ignore_index=True,
    )


def _metric_at(panel: pd.DataFrame, currency: str, report_date: dt.date, metric: str) -> float:
    rows = panel[(panel["currency"] == currency) & (panel["report_date"] == report_date)]
    if rows.empty or metric not in rows.columns:
        return float("nan")
    return float(rows.iloc[-1][metric])


def _spread_at(
    panel: pd.DataFrame,
    base: str,
    quote: str,
    report_date: dt.date,
    metric: str,
) -> float:
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


def build_decision_summary_table(
    report_date: dt.date,
    tier: str,
    pair_contexts: Sequence[Dict[str, Any]],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for ctx in pair_contexts:
        dec = ctx["decision"]
        reasons = dec.reasons
        rows.append(
            {
                "week_end": report_date.isoformat(),
                "pair": ctx["pair"],
                "base": ctx["base"],
                "quote": ctx["quote"],
                "tier_mode": tier.upper(),
                "direction_allowed": str(dec.direction).upper(),
                "allow": bool(dec.allow),
                "confidence_score": round(float(dec.score), 2),
                "spread": ctx.get("spread"),
                "dSpread_1w": ctx.get("dspread_1w"),
                "crowded_flag": _crowded_flag(
                    dec.components.get("crowded_base"), dec.components.get("crowded_quote")
                ),
                "macro_ok": _macro_ok_from_reasons(reasons),
                "news_ok": _news_ok_from_reasons(reasons),
                "blocked_by": _primary_blocked_by(reasons),
                "reasons": ";".join(reasons),
            }
        )
    return pd.DataFrame(rows)


def build_cot_components_table(
    report_date: dt.date,
    tier_cfg: Dict[str, Any],
    inst_df: pd.DataFrame,
) -> pd.DataFrame:
    df = inst_df.copy()
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce").dt.date
    df = df.sort_values(["symbol", "report_date"])
    df["dNet_1w"] = df.groupby("symbol")["driver_net"].diff(1)
    df["dNet_4w"] = df.groupby("symbol")["driver_net"].diff(4)

    tier = str(tier_cfg["strictness_tier"]).strip().lower()
    high = float(tier_cfg["crowdedness"]["high_by_tier"][tier])
    low = float(tier_cfg["crowdedness"]["low_by_tier"][tier])

    latest = df[df["report_date"] == report_date].copy()
    rows: List[Dict[str, object]] = []
    for _, row in latest.iterrows():
        pct = float(row["net_pct_oi_3y_pctile"]) if pd.notna(row["net_pct_oi_3y_pctile"]) else float("nan")
        crowded_side = "NONE"
        if not np.isnan(pct):
            if pct >= high:
                crowded_side = "HIGH"
            elif pct <= low:
                crowded_side = "LOW"
        rows.append(
            {
                "week_end": report_date.isoformat(),
                "ccy": row["symbol"],
                "net": row["driver_net"],
                "net_pct_oi": row["driver_net_pct_oi"],
                "pctile_3y": row["net_pct_oi_3y_pctile"],
                "z_3y": row["net_pct_oi_3y_z"],
                "dNet_1w": row["dNet_1w"],
                "dNet_4w": row["dNet_4w"],
                "oi": row["open_interest"],
                "crowded_side": crowded_side,
            }
        )
    return pd.DataFrame(rows)


def build_spread_flow_table(
    report_date: dt.date,
    tier_cfg: Dict[str, Any],
    pair_contexts: Sequence[Dict[str, Any]],
) -> pd.DataFrame:
    tier = str(tier_cfg["strictness_tier"]).strip().lower()
    metric_kind = tier_cfg["metric"]["kind"]
    metric_used = {
        "net_pctile": "pctile_3y",
        "net_z": "z_3y",
        "net_pct_oi": "net_pct_oi",
    }[metric_kind]
    spread_thr = float(tier_cfg["spread_gate"]["threshold_by_metric"][metric_kind][tier])
    flow_required = bool(tier_cfg["flow_gate"]["enabled_by_tier"][tier])
    dmin = float(tier_cfg["flow_gate"]["min_dspread_by_metric"][metric_kind][tier])

    rows: List[Dict[str, object]] = []
    for ctx in pair_contexts:
        direction = ctx["signal_direction"].upper()
        spread = ctx.get("spread")
        spread_pass = True
        if isinstance(spread, float) and not np.isnan(spread):
            if direction == "LONG":
                spread_pass = spread >= spread_thr
            else:
                spread_pass = spread <= -spread_thr
        else:
            spread_pass = False

        flow_pass = True
        if flow_required:
            flow_pass = "flow_ok" in ctx["decision"].reasons

        dspread_1w = ctx.get("dspread_1w")
        if dmin <= 0:
            flow_strength = "WEAK" if not dspread_1w else "STRONG"
        else:
            if dspread_1w is None or (isinstance(dspread_1w, float) and np.isnan(dspread_1w)):
                flow_strength = "WEAK"
            elif abs(float(dspread_1w)) >= 2 * dmin:
                flow_strength = "STRONG"
            elif abs(float(dspread_1w)) >= dmin:
                flow_strength = "MIXED"
            else:
                flow_strength = "WEAK"

        accel_required = bool(tier_cfg["flow_gate"]["acceleration_required_by_tier"][tier])
        accel = None
        if accel_required:
            if "flow_accel_ok" in ctx["decision"].reasons:
                accel = True
            elif "flow_accel_fail" in ctx["decision"].reasons:
                accel = False

        rows.append(
            {
                "week_end": report_date.isoformat(),
                "pair": ctx["pair"],
                "metric_used": metric_used,
                "spread": spread,
                "spread_gate_X": spread_thr,
                "spread_pass": spread_pass,
                "dSpread_1w": dspread_1w,
                "flow_pass": flow_pass,
                "accel_2w": accel,
                "flow_strength": flow_strength,
            }
        )
    return pd.DataFrame(rows)


def build_crowdedness_policy_table(
    report_date: dt.date,
    tier_cfg: Dict[str, Any],
    pair_contexts: Sequence[Dict[str, Any]],
) -> pd.DataFrame:
    tier = str(tier_cfg["strictness_tier"]).strip().lower()
    metric_kind = tier_cfg["metric"]["kind"]
    policy = str(tier_cfg["crowdedness"]["policy_by_tier"][tier]).strip().lower()
    policy_label = {
        "allow": "ALLOW",
        "reduce_risk": "ALLOW_REDUCED_SIZE",
        "allow_if_strong_flow": "REQUIRE_STRONG_FLOW",
        "block": "BLOCK",
    }.get(policy, "ALLOW")
    high = float(tier_cfg["crowdedness"]["high_by_tier"][tier])
    low = float(tier_cfg["crowdedness"]["low_by_tier"][tier])
    req_flow_min = float(tier_cfg["crowdedness"]["strong_flow_override_min_by_metric"][metric_kind][tier])

    rows: List[Dict[str, object]] = []
    for ctx in pair_contexts:
        base_pct = ctx.get("base_pctile")
        quote_pct = ctx.get("quote_pctile")
        direction = ctx["signal_direction"].lower()

        if direction == "long":
            base_extreme = base_pct is not None and not np.isnan(base_pct) and base_pct >= high
            quote_extreme = quote_pct is not None and not np.isnan(quote_pct) and quote_pct <= low
        else:
            base_extreme = base_pct is not None and not np.isnan(base_pct) and base_pct <= low
            quote_extreme = quote_pct is not None and not np.isnan(quote_pct) and quote_pct >= high

        crowded_type = "NONE"
        if base_extreme and quote_extreme:
            crowded_type = "BOTH_EXTREMES"
        elif base_extreme:
            crowded_type = "BASE_EXTREME"
        elif quote_extreme:
            crowded_type = "QUOTE_EXTREME"

        allowed_under_policy = True
        if "crowded_block" in ctx["decision"].reasons or "crowded_override_fail" in ctx["decision"].reasons:
            allowed_under_policy = False

        rows.append(
            {
                "week_end": report_date.isoformat(),
                "pair": ctx["pair"],
                "base_metric": ctx.get("base_metric"),
                "quote_metric": ctx.get("quote_metric"),
                "base_pctile": base_pct,
                "quote_pctile": quote_pct,
                "crowded_type": crowded_type,
                "policy": policy_label,
                "allowed_under_policy": allowed_under_policy,
                "size_multiplier": ctx["decision"].risk_multiplier,
                "required_dSpread_min": req_flow_min if policy_label == "REQUIRE_STRONG_FLOW" else 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_macro_gate_table(
    report_date: dt.date,
    tier_cfg: Dict[str, Any],
    pair_contexts: Sequence[Dict[str, Any]],
) -> pd.DataFrame:
    method = str(tier_cfg["macro"]["method"]).strip().lower()
    macro_source = "USD_BASKET_SCORE" if method == "usd_vs_basket" else "DXY_COT_PROXY"

    rows: List[Dict[str, object]] = []
    for ctx in pair_contexts:
        dec = ctx["decision"]
        macro_score = dec.components.get("macro_score")
        macro_regime = "N/A"
        if isinstance(macro_score, float) and not np.isnan(macro_score):
            if macro_score > 0:
                macro_regime = "USD_BULL"
            elif macro_score < 0:
                macro_regime = "USD_BEAR"
            else:
                macro_regime = "USD_NEUTRAL"
        macro_ok = _macro_ok_from_reasons(dec.reasons)
        if "macro_align_ok" in dec.reasons:
            macro_reason = "macro_align_ok"
        elif "macro_misaligned" in dec.reasons or "macro_block" in dec.reasons:
            macro_reason = "macro_misaligned"
        else:
            macro_reason = "macro_skipped"
        rows.append(
            {
                "week_end": report_date.isoformat(),
                "pair": ctx["pair"],
                "macro_source": macro_source,
                "macro_regime": macro_regime,
                "macro_score": macro_score,
                "trade_dir": ctx["signal_direction"].upper(),
                "macro_ok": macro_ok,
                "macro_reason": macro_reason,
            }
        )
    return pd.DataFrame(rows)


def build_tier_thresholds_table(tier_cfg: Dict[str, Any]) -> pd.DataFrame:
    metric_kind = tier_cfg["metric"]["kind"]
    rows: List[Dict[str, object]] = []
    for tier in ["loose", "balanced", "strict", "sniper"]:
        rows.append(
            {
                "tier": tier.upper(),
                "metric_used": metric_kind,
                "spread_X": tier_cfg["spread_gate"]["threshold_by_metric"][metric_kind][tier],
                "require_flow": tier_cfg["flow_gate"]["enabled_by_tier"][tier],
                "dSpread_min": tier_cfg["flow_gate"]["min_dspread_by_metric"][metric_kind][tier],
                "crowded_policy": tier_cfg["crowdedness"]["policy_by_tier"][tier],
                "macro_required": bool(tier_cfg["macro"].get("macro_gate_required", False))
                or bool(tier_cfg["macro"]["macro_gate_required_by_tier"][tier]),
                "news_blackout": "ON"
                if bool(tier_cfg["news_blackout"]["enabled"])
                and bool(tier_cfg["news_blackout"]["required_by_tier"][tier])
                else "OFF",
            }
        )
    return pd.DataFrame(rows)


def build_pine_join_table(
    signal_ts: dt.datetime,
    report_date: dt.date,
    tier: str,
    pair_contexts: Sequence[Dict[str, Any]],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    ts_str = signal_ts.isoformat()
    for ctx in pair_contexts:
        dec = ctx["decision"]
        rows.append(
            {
                "ts": ts_str,
                "pair": ctx["pair"],
                "pine_signal": ctx["signal_direction"].upper(),
                "cot_direction": str(dec.direction).upper(),
                "tier_mode": tier.upper(),
                "allow": bool(dec.allow),
                "score": round(float(dec.score), 2),
                "size_mult": dec.risk_multiplier,
                "reasons": ";".join(dec.reasons),
            }
        )
    return pd.DataFrame(rows)



def run_report(
    cfg: Config,
    requested_date: dt.date,
    pairs_csv: str,
    report_type: ReportType,
    usd_mode: str = "basket",
    usd_weights: str = "equal",
    refresh: bool = False,
    verbose: bool = False,
    signals_csv: Optional[str] = None,
    cot_filter_cfg: Optional[Dict[str, Any]] = None,
    tier_override: Optional[str] = None,
) -> Dict[str, object]:
    pairs = _parse_pairs(pairs_csv)
    if report_type == "disagg":
        raise ValueError("Disaggregated futures-only report does not include FX currencies.")

    cot_cfg_input: Dict[str, Any] = cot_filter_cfg or {}
    if tier_override:
        cot_cfg_input = dict(cot_cfg_input)
        cot_cfg_input["strictness_tier"] = tier_override
    cot_filter = COTFilter(cot_cfg_input)
    filter_cfg = cot_filter.cfg
    release_cfg = filter_cfg.get("release_alignment", default_cot_filter_config()["release_alignment"])

    years = _years_for_window(requested_date, cfg.rolling_weeks)
    base_dir = Path("data") / "cftc_cache"
    df_raw = load_years(report_type, years, base_dir, refresh=refresh, verbose=verbose)
    if df_raw.empty:
        raise ValueError("No data loaded from CFTC official sources.")

    report_dates = extract_report_dates(df_raw)
    resolved = resolve_report_date_by_release(requested_date, report_dates, release_cfg)

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

    # Build filter tables
    resolved_release_dt = _release_dt_for_report(
        resolved,
        release_cfg.get("timezone", "America/New_York"),
        release_cfg.get("release_weekday", "friday"),
        release_cfg.get("release_time", "15:30"),
    )
    signal_ts = resolved_release_dt + dt.timedelta(minutes=1)

    panel = _build_filter_panel(inst_df, release_cfg)
    panel = _add_usd_proxy(panel)

    signal_map = _parse_signals(signals_csv)
    signal_source = "provided" if signal_map else "cot_proxy"
    pair_z_map = {row["pair"]: row["pair_z"] for row in results}

    metric_kind = str(filter_cfg["metric"]["kind"]).strip().lower()
    tier = str(filter_cfg["strictness_tier"]).strip().lower()

    pair_contexts: List[Dict[str, Any]] = []
    for pair, base, quote in pairs:
        direction = signal_map.get(pair)
        if not direction:
            pz = pair_z_map.get(pair, float("nan"))
            if isinstance(pz, float) and not np.isnan(pz):
                direction = "long" if pz >= 0 else "short"
            else:
                direction = "long"

        decision = cot_filter.evaluate(
            COTFilterInput(pair=pair, signal_direction=direction, signal_ts=signal_ts),
            panel,
        )
        spread_val = _spread_at(panel, base, quote, resolved, metric_kind)
        dspread_1w = _dspread_at(panel, base, quote, resolved, metric_kind, 1)
        base_metric = _metric_at(panel, base, resolved, metric_kind)
        quote_metric = _metric_at(panel, quote, resolved, metric_kind)
        base_pctile = _metric_at(panel, base, resolved, "net_pctile")
        quote_pctile = _metric_at(panel, quote, resolved, "net_pctile")

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

    decisions_summary = build_decision_summary_table(resolved, tier, pair_contexts)
    cot_components = build_cot_components_table(resolved, filter_cfg, inst_df)
    spread_flow = build_spread_flow_table(resolved, filter_cfg, pair_contexts)
    crowdedness_policy = build_crowdedness_policy_table(resolved, filter_cfg, pair_contexts)
    macro_gate = build_macro_gate_table(resolved, filter_cfg, pair_contexts)
    tier_thresholds = build_tier_thresholds_table(filter_cfg)
    pine_join = build_pine_join_table(signal_ts, resolved, tier, pair_contexts)

    filter_tables = {
        "decisions_summary": decisions_summary,
        "cot_components": cot_components,
        "spread_flow": spread_flow,
        "crowdedness_policy": crowdedness_policy,
        "macro_gate": macro_gate,
        "tier_thresholds": tier_thresholds,
        "pine_join": pine_join,
    }

    allow_count = int(decisions_summary["allow"].sum()) if not decisions_summary.empty else 0
    total_count = int(len(decisions_summary))
    blocked_count = total_count - allow_count

    notes = [
        f"Release alignment: report_date {resolved.isoformat()} released at {resolved_release_dt.isoformat()} (UTC).",
        f"Pine signals: {signal_source} ({'PAIR=long/short' if signal_source == 'provided' else 'proxy by pair_z sign'}).",
        f"News blackout: {'ON' if filter_cfg['news_blackout']['enabled'] else 'OFF'}.",
    ]
    flow_lag = filter_cfg["flow_gate"]["lag_weeks_by_tier"].get(tier, 1)
    if flow_lag != 1:
        notes.append(f"Flow lag weeks (engine): {flow_lag}. dSpread_1w is shown for reference.")

    return {
        "requested_date": requested_date.isoformat(),
        "resolved_report_date": resolved.isoformat(),
        "resolved_release_dt": resolved_release_dt.isoformat(),
        "report_type": report_type,
        "usd_mode_requested": usd_mode,
        "usd_mode_used": usd_mode_used,
        "note": "COT positions are as-of Tuesday and released Friday.",
        "pairs": results,
        "filter_tables": filter_tables,
        "executive_summary": {
            "total_pairs": total_count,
            "allowed": allow_count,
            "blocked": blocked_count,
            "tier_mode": tier.upper(),
        },
        "notes": notes,
    }


def write_report_tables(filter_tables: Dict[str, pd.DataFrame], out_dir: Path) -> Dict[str, Path]:
    filenames = {
        "decisions_summary": "decisions_summary.csv",
        "cot_components": "cot_components.csv",
        "spread_flow": "spread_flow.csv",
        "crowdedness_policy": "crowdedness_policy.csv",
        "macro_gate": "macro_gate.csv",
        "tier_thresholds": "tier_thresholds.csv",
        "pine_join": "pine_join.csv",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    for key, df in filter_tables.items():
        fname = filenames.get(key, f"{key}.csv")
        path = out_dir / fname
        df.to_csv(path, index=False)
        paths[key] = path
    return paths


def _serialize_tables(filter_tables: Dict[str, pd.DataFrame]) -> Dict[str, List[Dict[str, Any]]]:
    return {k: v.to_dict(orient="records") for k, v in filter_tables.items()}


def format_report_output(report: Dict[str, object], out: str, out_dir: Path) -> str:
    if out == "json":
        payload = dict(report)
        if "filter_tables" in payload and isinstance(payload["filter_tables"], dict):
            payload["filter_tables"] = _serialize_tables(payload["filter_tables"])
        return json.dumps(payload, indent=2, default=str)
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
    summary = report.get("executive_summary", {})
    parts.append("<h2>Executive Summary</h2>")
    if isinstance(summary, dict):
        parts.append("<ul>")
        for k in ["total_pairs", "allowed", "blocked", "tier_mode"]:
            if k in summary:
                parts.append(f"<li><strong>{k}:</strong> {summary[k]}</li>")
        parts.append("</ul>")

    parts.append("<h2>Legacy Tables</h2>")
    parts.append("<h3>Pairs</h3>")
    parts.append(df.to_html(index=False, escape=True))

    filter_tables = report.get("filter_tables")
    if isinstance(filter_tables, dict):
        parts.append("<h2>COT Filter (Tiered) — Decision & Diagnostics</h2>")
        table_order = [
            ("decisions_summary", "Table 1: Tier Decision Summary"),
            ("cot_components", "Table 2: COT Components Breakdown"),
            ("spread_flow", "Table 3: Spread + Flow Table"),
            ("crowdedness_policy", "Table 4: Crowdedness Policy Table"),
            ("macro_gate", "Table 5: Macro Gate Table"),
            ("tier_thresholds", "Table 6: Tier Ladder Thresholds"),
            ("pine_join", "Table 7: Pine Signal Join"),
        ]
        for key, title in table_order:
            df_tbl = filter_tables.get(key)
            if isinstance(df_tbl, pd.DataFrame):
                parts.append(f"<h3>{title}</h3>")
                parts.append(df_tbl.to_html(index=False, escape=True))

    notes = report.get("notes", [])
    parts.append("<h2>Notes / Assumptions</h2>")
    if isinstance(notes, list):
        parts.append("<ul>")
        for n in notes:
            parts.append(f"<li>{n}</li>")
        parts.append("</ul>")
    parts.append("</body></html>")

    path.write_text("\n".join(parts), encoding="utf-8")
    return str(path)
