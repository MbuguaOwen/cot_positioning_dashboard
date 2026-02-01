from __future__ import annotations

import numpy as np


def bias_label(score: float) -> str:
    if np.isnan(score):
        return "UNKNOWN"
    if score >= 0.75:
        return "BULLISH"
    if score <= -0.75:
        return "BEARISH"
    return "NEUTRAL"


def reversal_risk_label(pctile_3y: float, z_3y: float) -> str:
    if np.isnan(pctile_3y) or np.isnan(z_3y):
        return "UNKNOWN"
    if pctile_3y >= 95 or pctile_3y <= 5 or abs(z_3y) >= 2.0:
        return "EXTREME"
    if pctile_3y >= 85 or pctile_3y <= 15 or abs(z_3y) >= 1.5:
        return "MED"
    return "LOW"
