from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .config import FxBiasV2Config
from .io import G10_CCY, pair_parts


def sign(v: Optional[float]) -> int:
    if v is None:
        return 0
    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0


def clip(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def component_row(
    *,
    component: str,
    df: pd.DataFrame,
    as_of_ts: pd.Timestamp,
    freshness_limit: int,
    coverage_ratio: float,
    source: str,
    reliability_override: Optional[bool] = None,
) -> Tuple[Dict[str, Any], Optional[pd.Timestamp], List[str]]:
    warnings: List[str] = []
    available = not df.empty
    max_ts: Optional[pd.Timestamp] = None
    freshness: Optional[int] = None
    if available and "ts" in df.columns and df["ts"].notna().any():
        max_ts = pd.to_datetime(df["ts"], errors="coerce", utc=True).max()
    if max_ts is not None and pd.notna(max_ts):
        freshness = int(max((as_of_ts - max_ts).total_seconds() // 60, 0))
    if reliability_override is None:
        reliable = bool(available)
        if available and freshness is not None and freshness > int(freshness_limit):
            reliable = False
        if available and coverage_ratio <= 0.0:
            reliable = False
    else:
        reliable = bool(reliability_override)
    if available and freshness is not None and freshness > int(freshness_limit):
        warnings.append(f"{component} STALE: age={freshness}m > {freshness_limit}m")
    row = {
        "Component": component,
        "Available": bool(available),
        "Reliable": bool(reliable),
        "UsedInBias": False,
        "Source": source,
        "FreshnessMinutes": freshness,
        "CoverageRatio": round(float(coverage_ratio), 3),
        "ReasonIfNotUsed": None,
    }
    return row, max_ts, warnings


def compose_bias_confidence(
    *,
    cfg: FxBiasV2Config,
    price_comp: Dict[str, Optional[float]],
    rate_scores: Dict[str, Optional[float]],
    risk_scores: Dict[str, Optional[float]],
    carry_scores: Dict[str, Optional[float]],
    skew_scores: Dict[str, Optional[float]],
    regime_w1: Dict[str, str],
    regime_d1: Dict[str, str],
    d1_flips_10d: Dict[str, int],
    cot_flags: Dict[str, Dict[str, bool]],
    rates_used: bool,
    risk_used: bool,
    carry_used: bool,
    skew_used: bool,
) -> Tuple[Dict[str, float], Dict[str, int], Dict[str, Dict[str, float]]]:
    w_nom = cfg.weights
    w = {
        "price": float(w_nom["price"]),
        "rates": float(w_nom["rates"]),
        "risk": float(w_nom["risk"]),
        "carry": float(w_nom["carry"]),
    }
    if skew_used:
        if carry_used:
            w["carry"] = float(w_nom["skew_if_available"]["carry"])
            w["skew"] = float(w_nom["skew_if_available"]["skew"])
        else:
            w["carry"] = 0.0
            w["skew"] = 0.10
    nominal_total = float(w_nom["price"] + w_nom["rates"] + w_nom["risk"] + w_nom["carry"])

    bias: Dict[str, float] = {}
    conf: Dict[str, int] = {}
    contribs: Dict[str, Dict[str, float]] = {}
    for ccy in G10_CCY:
        comp = {
            "price": price_comp.get(ccy),
            "rates": rate_scores.get(ccy) if rates_used else None,
            "risk": risk_scores.get(ccy) if risk_used else None,
            "carry": carry_scores.get(ccy) if carry_used else None,
            "skew": skew_scores.get(ccy) if skew_used else None,
        }
        used = {}
        weighted = 0.0
        for k, v in comp.items():
            wk = float(w.get(k, 0.0))
            if wk <= 0 or v is None or math.isnan(float(v)):
                continue
            used[k] = wk
            weighted += wk * float(v)
        denom = float(sum(used.values()))
        if denom <= 0:
            bias[ccy] = 0.0
            contribs[ccy] = {}
        else:
            raw = clip(weighted / denom, -1.0, 1.0)
            bias[ccy] = round(100.0 * raw, 1)
            contribs[ccy] = {k: round(100.0 * (used[k] * float(comp[k]) / denom), 1) for k in used}

        coverage = (sum(used.values()) / nominal_total) if nominal_total > 0 else 0.0
        non_price = [n for n in ["rates", "risk", "carry", "skew"] if n in used]
        aligned_num = 0
        aligned_den = 0
        p_sign = sign(comp.get("price"))
        for n in non_price:
            v = comp.get(n)
            if v is None or abs(float(v)) < 0.15:
                continue
            aligned_den += 1
            if p_sign == 0 or sign(float(v)) == p_sign:
                aligned_num += 1
        alignment = float(aligned_num / aligned_den) if aligned_den > 0 else 0.5
        w1 = regime_w1.get(ccy, "RANGE")
        d1 = regime_d1.get(ccy, "RANGE")
        if w1 == d1 and w1 != "RANGE":
            tf_alignment = 1.0
        elif w1 == "RANGE" or d1 == "RANGE":
            tf_alignment = 0.5
        else:
            tf_alignment = 0.0
        flips = int(d1_flips_10d.get(ccy, 0))
        stability = 1.0 - min(1.0, flips / 4.0)
        conf_score = round(100.0 * (0.45 * coverage + 0.25 * alignment + 0.20 * tf_alignment + 0.10 * stability))
        flags = cot_flags.get(ccy, {})
        if cfg.cot_overlay["enabled"] and flags.get("extreme_flag") and flags.get("persistence_flag"):
            conf_score -= int(cfg.cot_overlay["confidence_penalty_if_extreme_and_persistent"])
        conf[ccy] = int(clip(float(conf_score), 0.0, 100.0))
    return bias, conf, contribs


def build_pair_rows(
    *,
    pairs: List[str],
    latest_trends: pd.DataFrame,
    bias_scores: Dict[str, float],
    rate_scores: Dict[str, Optional[float]],
    risk_regime: str,
    pair_carry_sign: Dict[str, int],
    skew_scores: Dict[str, Optional[float]],
    cfg: FxBiasV2Config,
    block_all: bool,
    carry_used: bool = True,
    carry_reason_signs: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    t_bias = float(cfg.gate["t_bias"])
    opp = float(cfg.gate["opposing_strength_block"])
    rows = []
    reason_carry_signs = pair_carry_sign if carry_reason_signs is None else carry_reason_signs

    def trend(pair: str, tf: str) -> Tuple[str, float]:
        sub = latest_trends[(latest_trends["pair"] == pair) & (latest_trends["timeframe"] == tf)]
        if sub.empty:
            return "RANGE", 0.0
        r = sub.iloc[-1]
        return str(r["trend"]), float(r["strength"])

    for pair in sorted(set(pairs)):
        base, quote = pair_parts(pair)
        if base not in G10_CCY or quote not in G10_CCY:
            continue
        pair_bias = clip((float(bias_scores.get(base, 0.0)) - float(bias_scores.get(quote, 0.0))) / 2.0, -100.0, 100.0)
        tw1, _ = trend(pair, "W1")
        td1, _ = trend(pair, "D1")
        th4, sh4 = trend(pair, "H4")
        th1, _ = trend(pair, "H1")
        strong_oppose = (pair_bias > 0 and th4 == "BEAR" and sh4 >= opp) or (pair_bias < 0 and th4 == "BULL" and sh4 >= opp)
        d1_align = (pair_bias > 0 and td1 == "BULL") or (pair_bias < 0 and td1 == "BEAR")
        alignment = "PASS" if (d1_align and not strong_oppose) else "FAIL"
        if td1 == "BULL" and pair_bias > t_bias and not strong_oppose and not block_all:
            gate = "ALLOW_LONG"
        elif td1 == "BEAR" and pair_bias < -t_bias and not strong_oppose and not block_all:
            gate = "ALLOW_SHORT"
        else:
            gate = "BLOCK"
        suggested = "LONG" if gate == "ALLOW_LONG" else "SHORT" if gate == "ALLOW_SHORT" else "NEUTRAL"
        rb = rate_scores.get(base)
        rq = rate_scores.get(quote)
        rate_impact = None if rb is None or rq is None else float(rb - rq)
        sb = skew_scores.get(base)
        sq = skew_scores.get(quote)
        skew_impact = None if sb is None or sq is None else float(sb - sq)
        carry_sign = reason_carry_signs.get(pair) if carry_used else None
        reasons = []
        reasons.append("PRICE_OK" if d1_align else "PRICE_CONFLICT")
        if rate_impact is None:
            reasons.append("RATES_NA")
        else:
            reasons.append("RATES_OK" if sign(rate_impact) == sign(pair_bias) or abs(pair_bias) < 1e-9 else "RATES_CONFLICT")
        if risk_regime == "NEUTRAL":
            reasons.append("RISK_NA")
        elif "JPY" in (base, quote):
            reasons.append("RISK_OFF_JPY_BOOST" if risk_regime == "RISK_OFF" else "RISK_ON_JPY_PENALTY")
        if carry_sign is None:
            reasons.append("CARRY_NA")
        else:
            reasons.append("CARRY_TAILWIND" if sign(float(carry_sign)) == sign(pair_bias) else "CARRY_HEADWIND")
        if skew_impact is None:
            reasons.append("SKEW_NA")
        else:
            reasons.append("SKEW_OK" if sign(skew_impact) == sign(pair_bias) or sign(pair_bias) == 0 else "SKEW_CONFLICT")
        if strong_oppose:
            reasons.append("TF_CONFLICT")
        if abs(pair_bias) <= t_bias:
            reasons.append("BIAS_TOO_WEAK")
        if block_all:
            reasons.append("PRICE_STALE_BLOCK")
        reasons = list(dict.fromkeys(reasons))
        rows.append(
            {
                "Pair": pair,
                "PairBias": round(float(pair_bias), 1),
                "Trend_W1": tw1,
                "Trend_D1": td1,
                "Trend_H4": th4,
                "Trend_H1": th1,
                "AlignmentFlag": alignment,
                "ReasonCodes": reasons,
                "SuggestedBias": suggested,
                "GatingRuleForBOT": gate,
            }
        )
    return rows


def build_diagnostics(currency_rows: List[Dict[str, Any]], pair_rows: List[Dict[str, Any]], contributions: Dict[str, Dict[str, float]], pair_hist: pd.DataFrame, as_of_ts: pd.Timestamp) -> Dict[str, Any]:
    top_drivers = []
    for row in currency_rows:
        c = row["Currency"]
        ordered = sorted(contributions.get(c, {}).items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
        drivers = [{"Component": k, "Contribution": round(float(v), 1), "Note": "supports bias" if v >= 0 else "offsets bias"} for k, v in ordered]
        top_drivers.append({"Currency": c, "Drivers": drivers})
    shifts = []
    if not pair_hist.empty:
        cutoff = as_of_ts - pd.Timedelta(days=10)
        hist = pair_hist[pair_hist["ts"] <= as_of_ts].sort_values(["pair", "timeframe", "ts"])
        for (pair, tf), grp in hist.groupby(["pair", "timeframe"]):
            prev = None
            for _, r in grp.iterrows():
                cur = str(r["trend"])
                if prev is not None and cur != prev and r["ts"] >= cutoff:
                    shifts.append({"Pair": pair, "Timeframe": str(tf), "ShiftDate": r["ts"].date().isoformat(), "From": prev, "To": cur})
                prev = cur
    top_pairs = sorted([r for r in pair_rows if r["AlignmentFlag"] == "PASS"], key=lambda x: abs(float(x["PairBias"])), reverse=True)[:10]
    conviction = [{"Pair": r["Pair"], "PairBiasAbs": round(abs(float(r["PairBias"])), 1), "AlignmentFlag": r["AlignmentFlag"], "SuggestedBias": r["SuggestedBias"]} for r in top_pairs]
    disagree = []
    for r in pair_rows:
        dis = [x for x in r.get("ReasonCodes", []) if "CONFLICT" in x or "HEADWIND" in x]
        if dis:
            disagree.append({"Pair": r["Pair"], "PairBias": r["PairBias"], "DisagreementCodes": dis})
    return {
        "Top3DriversByCurrency": top_drivers,
        "RegimeShiftsLast10TradingDays": shifts,
        "HighestConvictionPairsTop10": conviction,
        "ComponentDisagreementPairs": disagree,
    }


def build_currency_polarity_pairs(
    *,
    currency_rows: List[Dict[str, Any]],
    tradable_pairs: List[str],
    spread_threshold: float,
    min_confidence: int,
    top_n: int,
) -> List[Dict[str, Any]]:
    ccy_map: Dict[str, Dict[str, Any]] = {}
    for row in currency_rows:
        ccy = str(row.get("Currency", "")).upper().strip()
        if not ccy:
            continue
        ccy_map[ccy] = row

    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for raw_pair in tradable_pairs:
        pair = str(raw_pair).upper().strip()
        if not pair:
            continue
        base, quote = pair_parts(pair)
        pair = f"{base}{quote}" if len(base) == 3 and len(quote) == 3 else ""
        if not pair or pair in seen:
            continue
        seen.add(pair)
        if base not in ccy_map or quote not in ccy_map:
            continue

        base_row = ccy_map[base]
        quote_row = ccy_map[quote]
        try:
            base_bias = float(base_row.get("BiasScore_Final"))
            quote_bias = float(quote_row.get("BiasScore_Final"))
            base_conf = int(base_row.get("Confidence"))
            quote_conf = int(quote_row.get("Confidence"))
        except Exception:
            continue
        if min(base_conf, quote_conf) < int(min_confidence):
            continue

        spread = float(base_bias - quote_bias)
        opposition = float(min(abs(base_bias), abs(quote_bias)))
        spread_sign = 1.0 if spread > 0 else -1.0 if spread < 0 else 0.0
        conviction = float(opposition * spread_sign)

        if spread >= float(spread_threshold):
            implied = "LONG"
        elif spread <= -float(spread_threshold):
            implied = "SHORT"
        else:
            implied = "NEUTRAL"

        if base_bias > 0 and quote_bias < 0:
            bucket = "BASE_BULL__QUOTE_BEAR"
        elif base_bias < 0 and quote_bias > 0:
            bucket = "BASE_BEAR__QUOTE_BULL"
        elif base_bias > 0 and quote_bias > 0:
            bucket = "BOTH_BULL"
        elif base_bias < 0 and quote_bias < 0:
            bucket = "BOTH_BEAR"
        else:
            bucket = "MIXED_WEAK"

        rows.append(
            {
                "Pair": pair,
                "Base": base,
                "Quote": quote,
                "BaseBias": round(base_bias, 1),
                "QuoteBias": round(quote_bias, 1),
                "Spread": round(spread, 1),
                "Opposition": round(opposition, 1),
                "ConvictionScore": round(conviction, 1),
                "ImpliedDirection": implied,
                "PolarityBucket": bucket,
                "BaseConf": int(base_conf),
                "QuoteConf": int(quote_conf),
            }
        )

    rows.sort(key=lambda r: (-abs(float(r["ConvictionScore"])), -abs(float(r["Spread"])), str(r["Pair"])))
    keep = max(int(top_n), 0)
    return rows[:keep]


def jsonable(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: jsonable(x) for k, x in v.items()}
    if isinstance(v, list):
        return [jsonable(x) for x in v]
    if isinstance(v, float):
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(v, pd.Timestamp):
        if v.tzinfo is None:
            v = v.tz_localize("UTC")
        return v.tz_convert("UTC").isoformat()
    return v
