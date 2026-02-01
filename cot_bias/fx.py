from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np

# DXY-style weights (only used if explicitly requested)
DXY_WEIGHTS = {
    "EUR": 0.576,
    "JPY": 0.136,
    "GBP": 0.119,
    "CAD": 0.091,
    "SEK": 0.042,
    "CHF": 0.036,
}


def _normalize_weights(weights: Dict[str, float], available: Iterable[str]) -> Dict[str, float]:
    avail = [c for c in available if c in weights]
    if not avail:
        return {}
    total = float(sum(weights[c] for c in avail))
    if total == 0:
        return {}
    return {c: float(weights[c]) / total for c in avail}


def usd_proxy_from_z(
    z_by_ccy: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Compute USD proxy as negative weighted mean of available currency z-scores."""
    # Exclude USD itself from the basket
    basket = {k: v for k, v in z_by_ccy.items() if k != "USD" and not np.isnan(v)}
    if not basket:
        return float("nan")

    if weights is None:
        # Equal weights across available currencies
        vals = list(basket.values())
        return float(-np.mean(vals))

    w = _normalize_weights(weights, basket.keys())
    if not w:
        vals = list(basket.values())
        return float(-np.mean(vals))

    total = 0.0
    for c, z in basket.items():
        if c in w:
            total += w[c] * float(z)
    return float(-total)


def pair_regime(pair_z: float, threshold: float = 0.5) -> str:
    if np.isnan(pair_z):
        return "UNKNOWN"
    if pair_z > threshold:
        return "BULLISH"
    if pair_z < -threshold:
        return "BEARISH"
    return "NEUTRAL"


def pair_reversal_risk_from_abs_z(abs_z: float) -> str:
    if np.isnan(abs_z):
        return "UNKNOWN"
    if abs_z >= 2.0:
        return "EXTREME"
    if abs_z >= 1.5:
        return "HIGH"
    if abs_z >= 1.0:
        return "MED"
    return "LOW"


def build_pairs_df(
    z_by_ccy: Dict[str, float],
    usd_z: float,
    usd_mode: str = "basket",
) -> "pd.DataFrame":
    import pandas as pd

    currencies = [c for c, z in z_by_ccy.items() if not np.isnan(z)]
    if not np.isnan(usd_z) and "USD" not in currencies:
        currencies.append("USD")
    currencies = sorted(set(currencies))

    rows = []
    for base in currencies:
        for quote in currencies:
            if base == quote:
                continue
            base_z = usd_z if base == "USD" else z_by_ccy.get(base, float("nan"))
            quote_z = usd_z if quote == "USD" else z_by_ccy.get(quote, float("nan"))
            if np.isnan(base_z) or np.isnan(quote_z):
                continue
            pair_z = base_z - quote_z
            rows.append(
                {
                    "pair": f"{base}{quote}",
                    "pair_z": pair_z,
                    "bias": pair_regime(pair_z),
                    "reversal_risk": pair_reversal_risk_from_abs_z(abs(pair_z)),
                    "base": base,
                    "quote": quote,
                    "base_z": base_z,
                    "quote_z": quote_z,
                    "usd_mode": usd_mode,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    bias_order = {"BULLISH": 0, "NEUTRAL": 1, "BEARISH": 2, "UNKNOWN": 3}
    risk_order = {"EXTREME": 0, "HIGH": 1, "MED": 2, "LOW": 3, "UNKNOWN": 4}
    df["_bias_rank"] = df["bias"].map(bias_order).fillna(9)
    df["_risk_rank"] = df["reversal_risk"].map(risk_order).fillna(9)

    # For bullish pairs, higher z should appear first; for bearish, lower z first.
    df["_pair_rank"] = df.apply(
        lambda r: -r["pair_z"] if r["bias"] == "BULLISH" else r["pair_z"], axis=1
    )

    df = df.sort_values(["_bias_rank", "_risk_rank", "_pair_rank", "pair"]).drop(
        columns=["_bias_rank", "_risk_rank", "_pair_rank"]
    )
    return df
