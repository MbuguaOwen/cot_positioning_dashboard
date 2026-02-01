from __future__ import annotations

import datetime as dt
from typing import Optional, Set

import numpy as np
import pandas as pd

from ..metrics.bias import bias_label

ALLOWED_STRENGTHS = {"MED", "HIGH", "EXTREME"}


def strength_label_from_abs_z(abs_z: float) -> str:
    if np.isnan(abs_z):
        return "UNKNOWN"
    if abs_z >= 2.0:
        return "EXTREME"
    if abs_z >= 1.5:
        return "HIGH"
    if abs_z >= 1.0:
        return "MED"
    return "LOW"


def _bias_from_value(value: float) -> str:
    if value is None or np.isnan(value):
        return "UNKNOWN"
    return bias_label(float(value))


def _decorate_pairs(pairs_df: pd.DataFrame, currency_df: pd.DataFrame) -> pd.DataFrame:
    if pairs_df is None or pairs_df.empty:
        return pd.DataFrame()

    pairs = pairs_df.copy()
    for col in ["pair_z", "base_z", "quote_z"]:
        if col in pairs.columns:
            pairs[col] = pd.to_numeric(pairs[col], errors="coerce")

    pairs["pair_bias"] = pairs["bias"]
    pairs["pair_strength"] = pairs["pair_z"].abs().apply(strength_label_from_abs_z)
    pairs["reversal_risk_extreme"] = pairs["reversal_risk"].astype(str).eq("EXTREME")

    bias_map = {}
    strength_map = {}
    if currency_df is not None and not currency_df.empty:
        cur = currency_df.copy()
        if "z_3y" in cur.columns:
            cur["strength"] = cur["z_3y"].abs().apply(strength_label_from_abs_z)
        if "symbol" in cur.columns and "bias" in cur.columns:
            bias_map = {
                str(sym): str(bias) for sym, bias in zip(cur["symbol"], cur["bias"])
            }
        if "symbol" in cur.columns and "strength" in cur.columns:
            strength_map = {
                str(sym): str(strength) for sym, strength in zip(cur["symbol"], cur["strength"])
            }

    pairs["base_bias"] = pairs["base"].map(bias_map)
    pairs["quote_bias"] = pairs["quote"].map(bias_map)
    pairs["base_strength"] = pairs["base"].map(strength_map)
    pairs["quote_strength"] = pairs["quote"].map(strength_map)

    pairs["base_bias"] = pairs["base_bias"].where(
        pairs["base_bias"].notna(), pairs["base_z"].apply(_bias_from_value)
    )
    pairs["quote_bias"] = pairs["quote_bias"].where(
        pairs["quote_bias"].notna(), pairs["quote_z"].apply(_bias_from_value)
    )
    pairs["base_strength"] = pairs["base_strength"].where(
        pairs["base_strength"].notna(),
        pairs["base_z"].abs().apply(strength_label_from_abs_z),
    )
    pairs["quote_strength"] = pairs["quote_strength"].where(
        pairs["quote_strength"].notna(),
        pairs["quote_z"].abs().apply(strength_label_from_abs_z),
    )

    for col in [
        "base_bias",
        "quote_bias",
        "base_strength",
        "quote_strength",
        "pair_bias",
        "pair_strength",
    ]:
        if col in pairs.columns:
            pairs[col] = pairs[col].fillna("UNKNOWN")

    pairs["passes_gate"] = (
        (pairs["base_bias"] == "BULLISH")
        & (pairs["base_strength"].isin(ALLOWED_STRENGTHS))
        & (pairs["quote_bias"] == "BEARISH")
        & (pairs["quote_strength"].isin(ALLOWED_STRENGTHS))
        & (pairs["pair_bias"] == "BULLISH")
        & (pairs["pair_strength"].isin(ALLOWED_STRENGTHS))
    )

    return pairs


def compute_gated_fx_pairs(
    currency_latest: pd.DataFrame,
    currency_previous: pd.DataFrame,
    pairs_latest: pd.DataFrame,
    pairs_previous: pd.DataFrame,
    latest_release_date: dt.date,
    previous_release_date: Optional[dt.date],
) -> pd.DataFrame:
    """Return gated FX pairs for latest release, plus 2-report consistency flag."""
    empty_cols = [
        "pair",
        "base",
        "quote",
        "base_bias",
        "base_strength",
        "quote_bias",
        "quote_strength",
        "pair_bias",
        "pair_strength",
        "reversal_risk",
        "reversal_risk_extreme",
        "passes_2_reports",
        "latest_release_date",
        "previous_release_date",
    ]

    latest_decor = _decorate_pairs(pairs_latest, currency_latest)
    if latest_decor.empty:
        return pd.DataFrame(columns=empty_cols)

    prev_pass: Set[str] = set()
    if previous_release_date and pairs_previous is not None and not pairs_previous.empty:
        prev_decor = _decorate_pairs(pairs_previous, currency_previous)
        if not prev_decor.empty:
            prev_pass = set(prev_decor.loc[prev_decor["passes_gate"], "pair"].astype(str))

    latest_filtered = latest_decor[latest_decor["passes_gate"]].copy()
    if latest_filtered.empty:
        return pd.DataFrame(columns=empty_cols)

    latest_filtered["passes_2_reports"] = latest_filtered["pair"].astype(str).apply(
        lambda p: p in prev_pass
    )
    latest_filtered["reversal_risk_extreme"] = (
        latest_filtered["reversal_risk_extreme"].fillna(False).astype(bool)
    )
    latest_filtered["passes_2_reports"] = (
        latest_filtered["passes_2_reports"].fillna(False).astype(bool)
    )
    latest_filtered["latest_release_date"] = (
        latest_release_date.isoformat() if latest_release_date else ""
    )
    latest_filtered["previous_release_date"] = (
        previous_release_date.isoformat() if previous_release_date else ""
    )

    for col in empty_cols:
        if col not in latest_filtered.columns:
            latest_filtered[col] = ""

    return latest_filtered[empty_cols].reset_index(drop=True)
