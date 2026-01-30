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

