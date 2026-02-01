from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from .bias import bias_label, reversal_risk_label


def rolling_zscore(x: pd.Series, window: int) -> pd.Series:
    mu = x.rolling(window=window, min_periods=window).mean()
    sd = x.rolling(window=window, min_periods=window).std(ddof=0)
    return (x - mu) / sd.replace(0, np.nan)


def rolling_percentile_of_last(x: pd.Series, window: int) -> pd.Series:
    def _pct(arr: np.ndarray) -> float:
        if len(arr) == 0 or np.isnan(arr[-1]):
            return np.nan
        v = arr[-1]
        arr2 = arr[~np.isnan(arr)]
        if len(arr2) == 0:
            return np.nan
        return float((arr2 <= v).sum()) / float(len(arr2)) * 100.0

    return x.rolling(window=window, min_periods=window).apply(_pct, raw=True)


def compute_metrics(df: pd.DataFrame, window: int = 156) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.sort_values(["symbol", "report_date"]).copy()

    def _calc(group: pd.DataFrame) -> pd.DataFrame:
        g = group.copy()
        if "symbol" not in g.columns:
            g["symbol"] = group.name
        g["net_pct_oi"] = g["net_pct_oi"].astype(float)
        g["z_3y"] = rolling_zscore(g["net_pct_oi"], window)
        g["pctile_3y"] = rolling_percentile_of_last(g["net_pct_oi"], window)
        g["delta_4w"] = g["net_pct_oi"] - g["net_pct_oi"].shift(4)
        g["z_delta_4w"] = rolling_zscore(g["delta_4w"], window)
        g["score"] = g["z_3y"] + 0.35 * g["z_delta_4w"]
        g["bias"] = g["score"].apply(bias_label)
        g["reversal_risk"] = g.apply(lambda r: reversal_risk_label(r["pctile_3y"], r["z_3y"]), axis=1)
        return g

    try:
        out = df.groupby("symbol", group_keys=False).apply(_calc, include_groups=False)
    except TypeError:
        out = df.groupby("symbol", group_keys=False).apply(_calc)
    return out
