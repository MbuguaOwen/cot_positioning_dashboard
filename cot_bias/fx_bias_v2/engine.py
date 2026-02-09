from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .compose import (
    build_currency_polarity_pairs,
    build_diagnostics,
    build_pair_rows,
    component_row,
    compose_bias_confidence,
    jsonable,
)
from .config import FxBiasV2Config, config_hash
from .features import (
    compute_carry_scores,
    compute_pair_trends,
    compute_price_strength,
    compute_rate_scores,
    compute_skew_scores,
    currency_regime_from_pairs,
    d1_flips_last_10,
    detect_risk_regime,
    latest_cot_flags,
    price_composite,
    risk_overlay_scores,
)
from .io import (
    G10_CCY,
    augment_price_timeframes,
    clean_pair,
    load_carry,
    load_cot_flags,
    load_price_bars,
    load_rates,
    load_risk,
    load_skew,
    parse_as_of,
)
from .render import render_fx_bias_v2_dashboard


ENGINE_VERSION = "2.1.0"


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None


def _iso(ts: Optional[pd.Timestamp]) -> Optional[str]:
    if ts is None or pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat()


def _src(df: pd.DataFrame, fallback: str) -> str:
    if df.empty or "source" not in df.columns or df["source"].dropna().empty:
        return fallback
    return str(df["source"].dropna().iloc[0])


def _latest_pair_carry_signs(
    carry: pd.DataFrame,
    *,
    as_of_ts: pd.Timestamp,
    freshness_limit_minutes: int,
) -> Dict[str, int]:
    required = {"pair", "ts", "forward_points"}
    if carry.empty or not required.issubset(set(carry.columns)):
        return {}

    data = carry[["pair", "ts", "forward_points"]].copy()
    data["ts"] = pd.to_datetime(data["ts"], errors="coerce", utc=True)
    data["forward_points"] = pd.to_numeric(data["forward_points"], errors="coerce")
    data = data.dropna(subset=["pair", "ts", "forward_points"])
    if data.empty:
        return {}

    data = data[data["ts"] <= as_of_ts]
    if data.empty:
        return {}

    cutoff = as_of_ts - pd.Timedelta(minutes=max(int(freshness_limit_minutes), 0))
    data = data[data["ts"] >= cutoff]
    if data.empty:
        return {}

    latest = data.sort_values(["pair", "ts"]).groupby("pair", as_index=False).tail(1)
    out: Dict[str, int] = {}
    for _, row in latest.iterrows():
        fp = float(row["forward_points"])
        out[str(row["pair"])] = 1 if fp > 0 else -1 if fp < 0 else 0
    return out


def run_fx_bias_engine_v2(
    *,
    as_of_ts: pd.Timestamp,
    cfg: FxBiasV2Config,
    prices_path: Optional[Path],
    rates_path: Optional[Path],
    risk_path: Optional[Path],
    carry_path: Optional[Path],
    skew_path: Optional[Path],
    cot_path: Optional[Path],
    out_dir: Path,
    pairs_filter: Optional[List[str]] = None,
) -> Dict[str, object]:
    warnings: List[str] = []

    if prices_path is not None and not prices_path.exists():
        warnings.append(f"PRICE PATH NOT FOUND: {prices_path}")
    if rates_path is not None and not rates_path.exists():
        warnings.append(f"RATES PATH NOT FOUND: {rates_path}")
    if risk_path is not None and not risk_path.exists():
        warnings.append(f"RISK PATH NOT FOUND: {risk_path}")
    if carry_path is not None and not carry_path.exists():
        warnings.append(f"CARRY PATH NOT FOUND: {carry_path}")
    if skew_path is not None and not skew_path.exists():
        warnings.append(f"SKEW PATH NOT FOUND: {skew_path}")
    if cot_path is not None and not cot_path.exists():
        warnings.append(f"COT PATH NOT FOUND: {cot_path}")

    bars = load_price_bars(prices_path, cfg.timeframes)
    rates = load_rates(rates_path)
    risk = load_risk(risk_path)
    carry = load_carry(carry_path)
    skew = load_skew(skew_path)
    cot = load_cot_flags(cot_path)

    if prices_path is not None and prices_path.exists() and bars.empty:
        warnings.append("PRICE INPUT PRESENT BUT NO VALID OHLCV BARS PARSED")
    if rates_path is not None and rates_path.exists() and rates.empty:
        warnings.append("RATES INPUT PRESENT BUT NO VALID ROWS PARSED")
    if risk_path is not None and risk_path.exists() and risk.empty:
        warnings.append("RISK INPUT PRESENT BUT NO VALID ROWS PARSED")
    if carry_path is not None and carry_path.exists() and carry.empty:
        warnings.append("CARRY INPUT PRESENT BUT NO VALID ROWS PARSED")
    if skew_path is not None and skew_path.exists() and skew.empty:
        warnings.append("SKEW INPUT PRESENT BUT NO VALID ROWS PARSED")
    if cot_path is not None and cot_path.exists() and cot.empty:
        warnings.append("COT INPUT PRESENT BUT NO VALID ROWS PARSED")

    if not bars.empty:
        bars = bars[bars["ts"] <= as_of_ts]
    if not rates.empty:
        rates = rates[rates["ts"] <= as_of_ts]
    if not risk.empty:
        risk = risk[risk["ts"] <= as_of_ts]
    if not carry.empty:
        carry = carry[carry["ts"] <= as_of_ts]
    if not skew.empty:
        skew = skew[skew["ts"] <= as_of_ts]
    if not cot.empty and "ts" in cot.columns:
        cot = cot[(cot["ts"].isna()) | (cot["ts"] <= as_of_ts)]
    bars, augment_notes = augment_price_timeframes(bars, as_of_ts)
    warnings.extend(augment_notes)

    pair_tf_cov = 0.0
    if not bars.empty:
        pair_count = bars["pair"].nunique()
        tf_count = bars[["pair", "timeframe"]].drop_duplicates().shape[0]
        pair_tf_cov = float(tf_count / max(pair_count * len(cfg.timeframes), 1))

    latest_trends, trend_hist = compute_pair_trends(bars, as_of_ts, cfg)
    w1_reg = currency_regime_from_pairs(latest_trends[latest_trends["timeframe"] == "W1"]) if not latest_trends.empty else {c: "RANGE" for c in G10_CCY}
    d1_reg = currency_regime_from_pairs(latest_trends[latest_trends["timeframe"] == "D1"]) if not latest_trends.empty else {c: "RANGE" for c in G10_CCY}
    d1_flips = d1_flips_last_10(trend_hist, as_of_ts)

    strength, price_cov_series, price_warns = compute_price_strength(bars, as_of_ts, cfg)
    warnings.extend(price_warns)
    price_comp = price_composite(strength, w1_reg, d1_reg)
    rate_scores, rate_cov = compute_rate_scores(rates, as_of_ts)
    risk_regime, risk_cov = detect_risk_regime(risk, as_of_ts)
    risk_scores = risk_overlay_scores(risk_regime, cfg)
    carry_scores, pair_carry_sign, carry_cov = compute_carry_scores(carry, as_of_ts)
    pair_carry_reason_signs = _latest_pair_carry_signs(
        carry,
        as_of_ts=as_of_ts,
        freshness_limit_minutes=int(cfg.freshness_minutes["carry"]),
    )
    skew_scores, skew_cov, skew_reliable_internal = compute_skew_scores(skew, as_of_ts)
    cot_flags = latest_cot_flags(cot, as_of_ts)

    caps = []
    data_max_ts: Dict[str, Optional[str]] = {}

    price_row, mx, w = component_row(component="PRICE", df=bars, as_of_ts=as_of_ts, freshness_limit=int(cfg.freshness_minutes["prices"]), coverage_ratio=max(pair_tf_cov, price_cov_series), source=_src(bars, "prices"))
    caps.append(price_row); data_max_ts["prices"] = _iso(mx); warnings.extend(w)
    rates_row, mx, w = component_row(component="RATES", df=rates, as_of_ts=as_of_ts, freshness_limit=int(cfg.freshness_minutes["rates"]), coverage_ratio=rate_cov, source=_src(rates, "rates"))
    caps.append(rates_row); data_max_ts["rates"] = _iso(mx); warnings.extend(w)
    risk_row, mx, w = component_row(component="RISK", df=risk, as_of_ts=as_of_ts, freshness_limit=int(cfg.freshness_minutes["risk"]), coverage_ratio=risk_cov, source=_src(risk, "risk"))
    caps.append(risk_row); data_max_ts["risk"] = _iso(mx); warnings.extend(w)
    carry_row, mx, w = component_row(component="CARRY", df=carry, as_of_ts=as_of_ts, freshness_limit=int(cfg.freshness_minutes["carry"]), coverage_ratio=carry_cov, source=_src(carry, "carry"))
    caps.append(carry_row); data_max_ts["carry"] = _iso(mx); warnings.extend(w)
    skew_row, mx, w = component_row(component="SKEW", df=skew, as_of_ts=as_of_ts, freshness_limit=int(cfg.freshness_minutes["skew"]), coverage_ratio=skew_cov, source=_src(skew, "skew"), reliability_override=skew_reliable_internal)
    caps.append(skew_row); data_max_ts["skew"] = _iso(mx); warnings.extend(w)
    cot_row, mx, w = component_row(component="COT", df=cot, as_of_ts=as_of_ts, freshness_limit=int(cfg.freshness_minutes["cot"]), coverage_ratio=float(cot["currency"].nunique() / len(G10_CCY)) if not cot.empty else 0.0, source=_src(cot, "cot"))
    caps.append(cot_row); data_max_ts["cot"] = _iso(mx); warnings.extend(w)

    price_used = bool(price_row["Reliable"])
    rates_used = bool(rates_row["Reliable"])
    risk_used = bool(risk_row["Reliable"]) and bool(cfg.risk_overlay["enabled"])
    carry_used = bool(carry_row["Reliable"])
    skew_used = bool(skew_row["Reliable"])

    price_row["UsedInBias"] = price_used
    rates_row["UsedInBias"] = rates_used
    risk_row["UsedInBias"] = risk_used
    carry_row["UsedInBias"] = carry_used
    skew_row["UsedInBias"] = skew_used
    cot_row["UsedInBias"] = False
    cot_row["ReasonIfNotUsed"] = "overlay_only"
    if not price_used:
        warnings.append("PRICE STALE OR MISSING -> GATES WILL BLOCK")
        price_row["ReasonIfNotUsed"] = "stale_or_missing"
    if not rates_used:
        rates_row["ReasonIfNotUsed"] = "stale_or_missing"
        rate_scores = {c: None for c in G10_CCY}
    if not risk_used:
        risk_row["ReasonIfNotUsed"] = "stale_or_missing"
        risk_scores = {c: 0.0 for c in G10_CCY}
    if not carry_used:
        carry_row["ReasonIfNotUsed"] = "stale_or_missing"
        carry_scores = {c: None for c in G10_CCY}
    if not skew_used:
        skew_row["ReasonIfNotUsed"] = "stale_or_missing_or_unreliable"
        skew_scores = {c: None for c in G10_CCY}
    if not price_used:
        price_comp = {c: None for c in G10_CCY}

    bias_scores, conf_scores, contributions = compose_bias_confidence(
        cfg=cfg,
        price_comp=price_comp,
        rate_scores=rate_scores,
        risk_scores=risk_scores,
        carry_scores=carry_scores,
        skew_scores=skew_scores,
        regime_w1=w1_reg,
        regime_d1=d1_reg,
        d1_flips_10d=d1_flips,
        cot_flags=cot_flags,
        rates_used=rates_used,
        risk_used=risk_used,
        carry_used=carry_used,
        skew_used=skew_used,
    )

    currency_rows = []
    for ccy in G10_CCY:
        s = strength.get(ccy, {})
        currency_rows.append(
            {
                "Currency": ccy,
                "PriceStrength_5D": None if s.get(5) is None else round(float(s.get(5)), 3),
                "PriceStrength_20D": None if s.get(20) is None else round(float(s.get(20)), 3),
                "PriceStrength_60D": None if s.get(60) is None else round(float(s.get(60)), 3),
                "PriceRegime_W1": w1_reg.get(ccy, "RANGE"),
                "PriceRegime_D1": d1_reg.get(ccy, "RANGE"),
                "RateScore": None if rate_scores.get(ccy) is None else round(float(rate_scores.get(ccy)), 3),
                "RiskOverlayScore": round(float(risk_scores.get(ccy, 0.0)), 3),
                "CarryScore": None if carry_scores.get(ccy) is None else round(float(carry_scores.get(ccy)), 3),
                "SkewScore": None if skew_scores.get(ccy) is None else round(float(skew_scores.get(ccy)), 3),
                "BiasScore_Final": round(float(bias_scores.get(ccy, 0.0)), 1),
                "Confidence": int(conf_scores.get(ccy, 0)),
            }
        )

    all_pairs = sorted(set(bars["pair"].tolist())) if not bars.empty else []
    allowed_pairs: Optional[set[str]] = None
    if pairs_filter:
        allowed_pairs = {clean_pair(p) for p in pairs_filter}
        all_pairs = [p for p in all_pairs if p in allowed_pairs]
    pair_rows = build_pair_rows(
        pairs=all_pairs,
        latest_trends=latest_trends,
        bias_scores=bias_scores,
        rate_scores=rate_scores,
        risk_regime=risk_regime,
        pair_carry_sign=pair_carry_sign,
        carry_used=carry_used,
        carry_reason_signs=pair_carry_reason_signs,
        skew_scores=skew_scores,
        cfg=cfg,
        block_all=not price_used,
    )
    diagnostics = build_diagnostics(currency_rows, pair_rows, contributions, trend_hist, as_of_ts)
    polarity_cfg = cfg.currency_polarity_pairs
    polarity_pairs = list(cfg.tradable_pairs)
    if allowed_pairs is not None:
        polarity_pairs = [p for p in polarity_pairs if p in allowed_pairs]
    polarity_rows = build_currency_polarity_pairs(
        currency_rows=currency_rows,
        tradable_pairs=polarity_pairs,
        spread_threshold=float(polarity_cfg["spread_threshold"]),
        min_confidence=int(polarity_cfg["min_confidence"]),
        top_n=int(polarity_cfg["top_n"]),
    )

    bundle = {
        "RunMeta": {
            "as_of_ts": as_of_ts.isoformat(),
            "engine_version": ENGINE_VERSION,
            "git_commit": _git_commit(),
            "config_hash": config_hash(cfg),
            "data_max_ts": data_max_ts,
            "warnings": warnings,
        },
        "CapabilityMatrix": caps,
        "A_Currency_Strength_Bias": currency_rows,
        "B_Pair_Bias_Trade_Gate": pair_rows,
        "C_Diagnostics": diagnostics,
        "D_Currency_Polarity_Pairs": polarity_rows,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fx_bias_v2_run.json").write_text(json.dumps(jsonable(bundle), indent=2), encoding="utf-8")
    pd.DataFrame(currency_rows).to_csv(out_dir / "currency_strength_bias.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(out_dir / "pair_bias_trade_gate.csv", index=False)
    pd.DataFrame(caps).to_csv(out_dir / "capability_matrix.csv", index=False)
    pd.DataFrame(
        polarity_rows,
        columns=[
            "Pair",
            "Base",
            "Quote",
            "BaseBias",
            "QuoteBias",
            "Spread",
            "Opposition",
            "ConvictionScore",
            "ImpliedDirection",
            "PolarityBucket",
            "BaseConf",
            "QuoteConf",
        ],
    ).to_csv(out_dir / "currency_polarity_pairs.csv", index=False)
    (out_dir / "diagnostics.json").write_text(json.dumps(jsonable(diagnostics), indent=2), encoding="utf-8")
    render_fx_bias_v2_dashboard(base_dir=out_dir, output_path=out_dir / "fx_bias_v2_dashboard.html")
    return bundle


def run_fx_bias_engine_v2_from_paths(
    *,
    as_of: Optional[str],
    cfg: FxBiasV2Config,
    prices_path: Optional[str],
    rates_path: Optional[str],
    risk_path: Optional[str],
    carry_path: Optional[str],
    skew_path: Optional[str],
    cot_path: Optional[str],
    out_dir: str,
    pairs_csv: Optional[str],
) -> Dict[str, object]:
    as_of_ts = parse_as_of(as_of)
    pairs = [p.strip().upper() for p in (pairs_csv or "").split(",") if p.strip()] if pairs_csv else None
    return run_fx_bias_engine_v2(
        as_of_ts=as_of_ts,
        cfg=cfg,
        prices_path=Path(prices_path) if prices_path else None,
        rates_path=Path(rates_path) if rates_path else None,
        risk_path=Path(risk_path) if risk_path else None,
        carry_path=Path(carry_path) if carry_path else None,
        skew_path=Path(skew_path) if skew_path else None,
        cot_path=Path(cot_path) if cot_path else None,
        out_dir=Path(out_dir),
        pairs_filter=pairs,
    )
