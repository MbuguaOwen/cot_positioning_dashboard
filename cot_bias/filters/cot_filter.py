from __future__ import annotations

import copy
import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

Direction = Literal["long", "short"]
DecisionDirection = Literal["long", "short", "none"]
Tier = Literal["loose", "balanced", "strict", "sniper"]


def default_cot_filter_config() -> Dict[str, Any]:
    """Default, conservative thresholds intended as a starting point."""
    return {
        "enabled": True,
        "strictness_tier": "balanced",
        "release_alignment": {
            "timezone": "America/New_York",
            "release_weekday": "friday",
            "release_time": "15:30",
            "freeze_midweek": True,
        },
        "metric": {
            "kind": "net_pctile",
            "lookback_weeks": 156,
            "min_history_weeks": 104,
        },
        "directional_bias": {
            "method": "spread_sign",
            "neutral_band": {
                "net_pctile": 2.0,
                "net_z": 0.05,
                "net_pct_oi": 0.25,
            },
        },
        "spread_gate": {
            "enabled": True,
            "threshold_by_metric": {
                "net_pctile": {"loose": 8.0, "balanced": 15.0, "strict": 22.0, "sniper": 30.0},
                "net_z": {"loose": 0.35, "balanced": 0.60, "strict": 0.90, "sniper": 1.20},
                "net_pct_oi": {"loose": 2.0, "balanced": 3.5, "strict": 5.0, "sniper": 6.5},
            },
        },
        "flow_gate": {
            "enabled_by_tier": {"loose": False, "balanced": True, "strict": True, "sniper": True},
            "lag_weeks_by_tier": {"loose": 1, "balanced": 2, "strict": 2, "sniper": 3},
            "min_dspread_by_metric": {
                "net_pctile": {"loose": 0.0, "balanced": 2.0, "strict": 3.0, "sniper": 5.0},
                "net_z": {"loose": 0.0, "balanced": 0.12, "strict": 0.20, "sniper": 0.30},
                "net_pct_oi": {"loose": 0.0, "balanced": 0.4, "strict": 0.7, "sniper": 1.0},
            },
            "acceleration_required_by_tier": {"loose": False, "balanced": False, "strict": True, "sniper": True},
        },
        "crowdedness": {
            "enabled_by_tier": {"loose": False, "balanced": True, "strict": True, "sniper": True},
            "metric_kind": "net_pctile",
            "high_by_tier": {"loose": 95.0, "balanced": 92.0, "strict": 90.0, "sniper": 88.0},
            "low_by_tier": {"loose": 5.0, "balanced": 8.0, "strict": 10.0, "sniper": 12.0},
            "policy_by_tier": {
                "loose": "allow",
                "balanced": "allow_if_strong_flow",
                "strict": "block",
                "sniper": "block",
            },
            "strong_flow_override_min_by_metric": {
                "net_pctile": {"loose": 0.0, "balanced": 4.0, "strict": 6.0, "sniper": 8.0},
                "net_z": {"loose": 0.0, "balanced": 0.25, "strict": 0.35, "sniper": 0.50},
                "net_pct_oi": {"loose": 0.0, "balanced": 1.0, "strict": 1.4, "sniper": 2.0},
            },
            "risk_multiplier_if_policy_c": {
                "one_side_crowded": 0.75,
                "both_sides_crowded": 0.50,
            },
        },
        "macro": {
            "enabled": True,
            "macro_gate_required": False,
            "macro_gate_required_by_tier": {"loose": False, "balanced": False, "strict": True, "sniper": True},
            "method": "usd_vs_basket",
            "metric_kind": "net_z",
            "usd_basket": ["EUR", "JPY", "GBP", "AUD", "CAD", "CHF", "NZD"],
            "align_threshold": 0.25,
            "reversal_mode_enabled": False,
            "reversal_min_flow_multiplier": 1.5,
        },
        "news_blackout": {
            "enabled": False,
            "calendar_path": None,
            "tier1_only": True,
            "pre_hours": 4,
            "post_hours": 2,
            "required_by_tier": {"loose": False, "balanced": False, "strict": False, "sniper": True},
        },
        "scoring": {
            "enabled": True,
            "weights": {
                "spread_strength": 40.0,
                "flow_alignment": 25.0,
                "crowdedness_penalty": 20.0,
                "macro_alignment": 15.0,
                "news_penalty": 10.0,
            },
            "baseline": 50.0,
            "clamp_min": 0.0,
            "clamp_max": 100.0,
            "min_score_to_allow_by_tier": {"loose": 30.0, "balanced": 40.0, "strict": 50.0, "sniper": 65.0},
            "enforce_min_score_gate": False,
        },
        "sizing": {
            "use_score_for_size": True,
            "score_bands": [
                {"min": 0.0, "max": 39.0, "multiplier": 0.00},
                {"min": 40.0, "max": 54.0, "multiplier": 0.50},
                {"min": 55.0, "max": 69.0, "multiplier": 0.75},
                {"min": 70.0, "max": 100.0, "multiplier": 1.00},
            ],
        },
    }


@dataclass(frozen=True)
class COTFilterInput:
    pair: str
    signal_direction: Direction
    signal_ts: dt.datetime


@dataclass(frozen=True)
class EventWindow:
    start_dt: dt.datetime
    end_dt: dt.datetime
    currencies: Set[str]
    tier: int = 1
    event_id: str = ""


@dataclass
class COTFilterDecision:
    allow: bool
    direction: DecisionDirection
    score: float
    reasons: List[str]
    strictness_tier: Tier
    components: Dict[str, Any] = field(default_factory=dict)
    risk_multiplier: float = 0.0
    effective_report_date: Optional[dt.date] = None
    effective_release_dt: Optional[dt.datetime] = None


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _as_utc(ts: dt.datetime) -> dt.datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=dt.timezone.utc)
    return ts.astimezone(dt.timezone.utc)


def _safe_float(value: Any) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return f


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class COTFilter:
    """
    Multi-layer COT + macro regime filter for Pine entry gating.

    Required cot_panel columns:
    - currency (str)
    - report_date (date-like)
    - release_dt (datetime-like, compared in UTC)
    - net_pctile / net_z / net_pct_oi (one or more metrics)
    """

    METRIC_ALIASES = {
        "net_pctile": "net_pctile",
        "netpctile": "net_pctile",
        "NetPctile": "net_pctile",
        "net_z": "net_z",
        "netz": "net_z",
        "NetZ": "net_z",
        "net_pct_oi": "net_pct_oi",
        "net_%oi": "net_pct_oi",
        "Net_%OI": "net_pct_oi",
    }

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        cfg_in = dict(config or {})
        if "cot_filter" in cfg_in and isinstance(cfg_in["cot_filter"], Mapping):
            cfg_in = dict(cfg_in["cot_filter"])
        self.cfg = _deep_merge(default_cot_filter_config(), cfg_in)

    def evaluate(
        self,
        cot_input: COTFilterInput,
        cot_panel: pd.DataFrame,
        event_windows: Optional[Sequence[EventWindow]] = None,
        macro_override: Optional[float] = None,
    ) -> COTFilterDecision:
        tier = self._tier()
        reasons: List[str] = []
        components: Dict[str, Any] = {}
        risk_multiplier = 1.0

        if not bool(self.cfg.get("enabled", True)):
            return COTFilterDecision(
                allow=True,
                direction=cot_input.signal_direction,
                score=100.0,
                reasons=["filter_disabled"],
                strictness_tier=tier,
                risk_multiplier=1.0,
            )

        panel = self._prepare_panel(cot_panel)
        if panel.empty:
            return COTFilterDecision(
                allow=False,
                direction="none",
                score=0.0,
                reasons=["empty_panel"],
                strictness_tier=tier,
                risk_multiplier=0.0,
            )

        metric_kind = self._metric_name(self.cfg["metric"]["kind"])
        spread_metric = metric_kind
        crowded_metric = self._metric_name(self.cfg["crowdedness"]["metric_kind"])

        signal_ts = _as_utc(cot_input.signal_ts)
        report_date, release_dt = self._effective_report(panel, signal_ts)
        if report_date is None:
            return COTFilterDecision(
                allow=False,
                direction="none",
                score=0.0,
                reasons=["no_report_released_yet"],
                strictness_tier=tier,
                risk_multiplier=0.0,
            )

        pair = cot_input.pair.upper().strip()
        if len(pair) != 6:
            return COTFilterDecision(
                allow=False,
                direction="none",
                score=0.0,
                reasons=[f"invalid_pair:{pair}"],
                strictness_tier=tier,
                risk_multiplier=0.0,
                effective_report_date=report_date,
                effective_release_dt=release_dt,
            )

        base, quote = pair[:3], pair[3:]
        base_value = self._value(panel, base, report_date, spread_metric)
        quote_value = self._value(panel, quote, report_date, spread_metric)
        spread = base_value - quote_value
        components["spread"] = spread
        components["base_value"] = base_value
        components["quote_value"] = quote_value

        # 1) Directional bias (hard gate)
        bias_long, bias_short = self._directional_bias(base_value, quote_value, spread, spread_metric)
        direction_ok = (
            cot_input.signal_direction == "long" and bias_long
        ) or (
            cot_input.signal_direction == "short" and bias_short
        )
        reasons.append("direction_ok" if direction_ok else "direction_fail")

        # 2) Relative spread threshold (hard gate)
        spread_gate_enabled = bool(self.cfg["spread_gate"]["enabled"])
        spread_threshold = float(self.cfg["spread_gate"]["threshold_by_metric"][spread_metric][tier])
        if cot_input.signal_direction == "long":
            spread_ok = spread >= spread_threshold
        else:
            spread_ok = spread <= -spread_threshold
        if not spread_gate_enabled:
            spread_ok = True
        reasons.append("spread_ok" if spread_ok else "spread_fail")

        # 3) Flow confirmation (hard gate when enabled)
        flow_enabled = bool(self.cfg["flow_gate"]["enabled_by_tier"][tier])
        flow_ok = True
        dspread = float("nan")
        dspread_prev = float("nan")
        if flow_enabled:
            lag = int(self.cfg["flow_gate"]["lag_weeks_by_tier"][tier])
            spread_prev, spread_prev2 = self._lagged_spreads(panel, base, quote, report_date, spread_metric, lag)
            if np.isfinite(spread_prev):
                dspread = spread - spread_prev
            if np.isfinite(spread_prev2):
                dspread_prev = spread_prev - spread_prev2
            components["dspread"] = dspread
            components["dspread_prev"] = dspread_prev

            dmin = float(self.cfg["flow_gate"]["min_dspread_by_metric"][spread_metric][tier])
            if cot_input.signal_direction == "long":
                flow_sign_ok = dspread > 0
            else:
                flow_sign_ok = dspread < 0
            flow_mag_ok = np.isfinite(dspread) and abs(dspread) >= dmin
            flow_ok = bool(flow_sign_ok and flow_mag_ok)
            reasons.append("flow_ok" if flow_ok else "flow_fail")

            accel_required = bool(self.cfg["flow_gate"]["acceleration_required_by_tier"][tier])
            if accel_required:
                if not np.isfinite(dspread_prev):
                    flow_ok = False
                    reasons.append("flow_accel_missing")
                else:
                    if cot_input.signal_direction == "long":
                        accel_ok = dspread > dspread_prev
                    else:
                        accel_ok = dspread < dspread_prev
                    flow_ok = bool(flow_ok and accel_ok)
                    reasons.append("flow_accel_ok" if accel_ok else "flow_accel_fail")
        else:
            reasons.append("flow_skipped")

        # 4) Crowdedness / squeeze risk
        crowd_enabled = bool(self.cfg["crowdedness"]["enabled_by_tier"][tier])
        crowded_block = False
        crowded_penalty = 0.0
        if crowd_enabled:
            base_crowd_value = self._value(panel, base, report_date, crowded_metric)
            quote_crowd_value = self._value(panel, quote, report_date, crowded_metric)
            hi = float(self.cfg["crowdedness"]["high_by_tier"][tier])
            lo = float(self.cfg["crowdedness"]["low_by_tier"][tier])

            if cot_input.signal_direction == "long":
                base_crowded = base_crowd_value >= hi
                quote_crowded = quote_crowd_value <= lo
            else:
                base_crowded = base_crowd_value <= lo
                quote_crowded = quote_crowd_value >= hi

            components["crowded_base"] = bool(base_crowded)
            components["crowded_quote"] = bool(quote_crowded)

            crowded_any = bool(base_crowded or quote_crowded)
            crowded_both = bool(base_crowded and quote_crowded)
            if crowded_any:
                crowded_penalty = 1.0 if crowded_both else 0.5
            policy = str(self.cfg["crowdedness"]["policy_by_tier"][tier]).strip().lower()

            if crowded_any:
                if policy == "block":
                    crowded_block = True
                    reasons.append("crowded_block")
                elif policy == "allow_if_strong_flow":
                    strong_min = float(
                        self.cfg["crowdedness"]["strong_flow_override_min_by_metric"][spread_metric][tier]
                    )
                    strong_flow = np.isfinite(dspread) and abs(dspread) >= strong_min
                    crowded_block = not strong_flow
                    reasons.append("crowded_override_ok" if strong_flow else "crowded_override_fail")
                elif policy == "reduce_risk":
                    if crowded_both:
                        risk_multiplier *= float(self.cfg["crowdedness"]["risk_multiplier_if_policy_c"]["both_sides_crowded"])
                    else:
                        risk_multiplier *= float(self.cfg["crowdedness"]["risk_multiplier_if_policy_c"]["one_side_crowded"])
                    reasons.append("crowded_reduce_risk")
                else:
                    reasons.append("crowded_allow")
            else:
                reasons.append("crowded_clear")
        else:
            reasons.append("crowded_skipped")

        # 5) Macro gate
        macro_enabled = bool(self.cfg["macro"]["enabled"])
        macro_required = bool(self.cfg["macro"].get("macro_gate_required", False)) or bool(
            self.cfg["macro"]["macro_gate_required_by_tier"][tier]
        )
        macro_ok = True
        macro_score = float("nan")
        usd_exposure = self._usd_exposure(base, quote, cot_input.signal_direction)

        if macro_enabled and usd_exposure != 0:
            macro_score = _safe_float(macro_override)
            if not np.isfinite(macro_score):
                macro_score = self._macro_score(panel, report_date)
            components["macro_score"] = macro_score
            components["usd_exposure"] = usd_exposure

            align_threshold = float(self.cfg["macro"]["align_threshold"])
            macro_ok = (usd_exposure * macro_score) >= align_threshold
            reasons.append("macro_align_ok" if macro_ok else "macro_misaligned")

            if macro_required and not macro_ok:
                reversal_enabled = bool(self.cfg["macro"]["reversal_mode_enabled"])
                if reversal_enabled:
                    dmin = float(self.cfg["flow_gate"]["min_dspread_by_metric"][spread_metric][tier])
                    required_flow = dmin * float(self.cfg["macro"]["reversal_min_flow_multiplier"])
                    reversal_ok = np.isfinite(dspread) and abs(dspread) >= required_flow
                    macro_ok = bool(reversal_ok)
                    reasons.append("macro_reversal_ok" if reversal_ok else "macro_reversal_fail")
                else:
                    reasons.append("macro_block")
        else:
            reasons.append("macro_skipped")

        # 6) News blackout
        news_hit = False
        news_block = False
        if bool(self.cfg["news_blackout"]["enabled"]) and event_windows:
            news_hit = self._news_blackout_hit(signal_ts, base, quote, event_windows)
            if news_hit:
                reasons.append("news_blackout_hit")
                if bool(self.cfg["news_blackout"]["required_by_tier"][tier]):
                    news_block = True
                    reasons.append("news_blackout_block")
        else:
            reasons.append("news_skipped")

        # Hard-gate verdict
        allow = bool(direction_ok and spread_ok and flow_ok and (not crowded_block) and macro_ok and (not news_block))

        # Soft score
        score = self._score(
            tier=tier,
            spread=spread,
            spread_threshold=spread_threshold,
            dspread=dspread,
            metric_kind=spread_metric,
            macro_score=macro_score,
            macro_ok=macro_ok,
            crowded_penalty=crowded_penalty,
            news_hit=news_hit,
        )
        components["score"] = score

        if bool(self.cfg["scoring"]["enforce_min_score_gate"]):
            score_min = float(self.cfg["scoring"]["min_score_to_allow_by_tier"][tier])
            if score < score_min:
                allow = False
                reasons.append("score_gate_fail")

        # Risk multiplier
        if not allow:
            risk_multiplier = 0.0
        elif bool(self.cfg["sizing"]["use_score_for_size"]):
            risk_multiplier *= self._score_to_size_multiplier(score)

        reasons.append("allow" if allow else "blocked")
        direction: DecisionDirection = cot_input.signal_direction if allow else "none"
        return COTFilterDecision(
            allow=allow,
            direction=direction,
            score=score,
            reasons=reasons,
            strictness_tier=tier,
            components=components,
            risk_multiplier=risk_multiplier,
            effective_report_date=report_date,
            effective_release_dt=release_dt,
        )

    def _tier(self) -> Tier:
        tier = str(self.cfg.get("strictness_tier", "balanced")).strip().lower()
        if tier not in {"loose", "balanced", "strict", "sniper"}:
            raise ValueError(f"Invalid strictness tier: {tier}")
        return tier  # type: ignore[return-value]

    def _prepare_panel(self, df: pd.DataFrame) -> pd.DataFrame:
        required = {"currency", "report_date", "release_dt"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"cot_panel missing required columns: {sorted(missing)}")

        panel = df.copy()
        panel["currency"] = panel["currency"].astype(str).str.upper()
        panel["report_date"] = pd.to_datetime(panel["report_date"], errors="coerce").dt.date
        panel["release_dt"] = pd.to_datetime(panel["release_dt"], errors="coerce", utc=True)
        panel = panel.dropna(subset=["currency", "report_date", "release_dt"])
        return panel.sort_values(["release_dt", "currency"]).reset_index(drop=True)

    def _effective_report(self, panel: pd.DataFrame, signal_ts_utc: dt.datetime) -> Tuple[Optional[dt.date], Optional[dt.datetime]]:
        releases = panel[["report_date", "release_dt"]].drop_duplicates().sort_values(["release_dt", "report_date"])
        eligible = releases[releases["release_dt"] <= pd.Timestamp(signal_ts_utc)]
        if eligible.empty:
            return None, None
        row = eligible.iloc[-1]
        rd = row["report_date"]
        rel = row["release_dt"].to_pydatetime()
        return rd, rel

    def _metric_name(self, raw_name: str) -> str:
        name = str(raw_name).strip()
        if name in self.METRIC_ALIASES:
            return self.METRIC_ALIASES[name]
        lowered = name.lower()
        if lowered in self.METRIC_ALIASES:
            return self.METRIC_ALIASES[lowered]
        raise ValueError(f"Unsupported metric name: {raw_name}")

    def _value(self, panel: pd.DataFrame, currency: str, report_date: dt.date, metric_name: str) -> float:
        if metric_name not in panel.columns:
            return float("nan")
        rows = panel[(panel["currency"] == currency) & (panel["report_date"] == report_date)]
        if rows.empty:
            return float("nan")
        return _safe_float(rows.iloc[-1][metric_name])

    def _directional_bias(
        self,
        base_value: float,
        quote_value: float,
        spread: float,
        metric_kind: str,
    ) -> Tuple[bool, bool]:
        method = str(self.cfg["directional_bias"]["method"]).strip().lower()
        neutral = float(self.cfg["directional_bias"]["neutral_band"][metric_kind])
        center = 50.0 if metric_kind == "net_pctile" else 0.0

        if method == "spread_sign":
            return bool(spread > neutral), bool(spread < -neutral)

        if method in {"pctile_sign", "net_sign", "z_sign"}:
            bullish = (base_value >= center + neutral) and (quote_value <= center - neutral)
            bearish = (base_value <= center - neutral) and (quote_value >= center + neutral)
            return bool(bullish), bool(bearish)

        raise ValueError(f"Unsupported directional_bias.method: {method}")

    def _lagged_spreads(
        self,
        panel: pd.DataFrame,
        base: str,
        quote: str,
        report_date: dt.date,
        metric_name: str,
        lag: int,
    ) -> Tuple[float, float]:
        if lag <= 0:
            return float("nan"), float("nan")
        dates = sorted(panel["report_date"].dropna().unique().tolist())
        if report_date not in dates:
            return float("nan"), float("nan")

        idx = dates.index(report_date)
        if idx - lag < 0:
            return float("nan"), float("nan")
        rd_prev = dates[idx - lag]
        spread_prev = self._value(panel, base, rd_prev, metric_name) - self._value(panel, quote, rd_prev, metric_name)

        if idx - (2 * lag) < 0:
            return spread_prev, float("nan")
        rd_prev2 = dates[idx - (2 * lag)]
        spread_prev2 = self._value(panel, base, rd_prev2, metric_name) - self._value(panel, quote, rd_prev2, metric_name)
        return spread_prev, spread_prev2

    def _usd_exposure(self, base: str, quote: str, signal_direction: Direction) -> int:
        if base == "USD":
            return 1 if signal_direction == "long" else -1
        if quote == "USD":
            return -1 if signal_direction == "long" else 1
        return 0

    def _macro_score(self, panel: pd.DataFrame, report_date: dt.date) -> float:
        method = str(self.cfg["macro"]["method"]).strip().lower()
        metric_name = self._metric_name(self.cfg["macro"]["metric_kind"])
        basket = [str(x).upper() for x in self.cfg["macro"]["usd_basket"]]

        def _usd_vs_basket() -> float:
            usd_value = self._value(panel, "USD", report_date, metric_name)
            basket_values = [self._value(panel, c, report_date, metric_name) for c in basket]
            basket_values = [x for x in basket_values if np.isfinite(x)]
            basket_avg = float(np.mean(basket_values)) if basket_values else 0.0
            if np.isfinite(usd_value):
                return usd_value - basket_avg
            # If USD contract is unavailable, invert basket average as proxy.
            return -basket_avg

        if method == "usd_vs_basket":
            return _usd_vs_basket()
        if method == "dxy_cot":
            dxy_value = self._value(panel, "DXY", report_date, metric_name)
            if np.isfinite(dxy_value):
                return dxy_value
            return _usd_vs_basket()
        raise ValueError(f"Unsupported macro.method: {method}")

    def _news_blackout_hit(
        self,
        signal_ts_utc: dt.datetime,
        base: str,
        quote: str,
        event_windows: Sequence[EventWindow],
    ) -> bool:
        pre_h = float(self.cfg["news_blackout"]["pre_hours"])
        post_h = float(self.cfg["news_blackout"]["post_hours"])
        require_tier1 = bool(self.cfg["news_blackout"]["tier1_only"])
        watched = {base, quote}

        for window in event_windows:
            if require_tier1 and window.tier != 1:
                continue
            if not (window.currencies & watched):
                continue
            start_utc = _as_utc(window.start_dt) - dt.timedelta(hours=pre_h)
            end_utc = _as_utc(window.end_dt) + dt.timedelta(hours=post_h)
            if start_utc <= signal_ts_utc <= end_utc:
                return True
        return False

    def _score(
        self,
        tier: Tier,
        spread: float,
        spread_threshold: float,
        dspread: float,
        metric_kind: str,
        macro_score: float,
        macro_ok: bool,
        crowded_penalty: float,
        news_hit: bool,
    ) -> float:
        if not bool(self.cfg["scoring"]["enabled"]):
            return 100.0

        weights = self.cfg["scoring"]["weights"]
        baseline = float(self.cfg["scoring"]["baseline"])

        spread_ref = max(abs(spread_threshold), 1e-9)
        spread_strength = _clamp(abs(spread) / spread_ref, 0.0, 1.0)

        flow_ref = float(self.cfg["flow_gate"]["min_dspread_by_metric"][metric_kind][tier])
        if flow_ref <= 0:
            flow_ref = 1.0
        flow_strength = _clamp(abs(dspread) / flow_ref, 0.0, 1.0) if np.isfinite(dspread) else 0.0

        if np.isfinite(macro_score):
            macro_strength = 1.0 if macro_ok else 0.0
        else:
            macro_strength = 0.5

        score = (
            baseline
            + float(weights["spread_strength"]) * spread_strength
            + float(weights["flow_alignment"]) * flow_strength
            + float(weights["macro_alignment"]) * macro_strength
            - float(weights["crowdedness_penalty"]) * _clamp(crowded_penalty, 0.0, 1.0)
        )
        if news_hit:
            score -= float(weights["news_penalty"])

        lo = float(self.cfg["scoring"]["clamp_min"])
        hi = float(self.cfg["scoring"]["clamp_max"])
        return _clamp(score, lo, hi)

    def _score_to_size_multiplier(self, score: float) -> float:
        for band in self.cfg["sizing"]["score_bands"]:
            if float(band["min"]) <= score <= float(band["max"]):
                return float(band["multiplier"])
        return 0.0

