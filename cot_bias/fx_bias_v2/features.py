from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import FxBiasV2Config
from .io import G10_CCY, clean_pair, pair_parts


def rolling_z(series: pd.Series, window: int) -> pd.Series:
    mu = series.rolling(window=window, min_periods=window).mean()
    sd = series.rolling(window=window, min_periods=window).std(ddof=0)
    return (series - mu) / sd.replace(0, np.nan)


def clip(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev).abs(),
            (df["low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=n, min_periods=n).mean()


def trend_series(
    bars: pd.DataFrame,
    *,
    ema_len: int,
    atr_len: int,
    slope_k: int,
    range_strength_lt: float,
    slope_abs_min: float,
    pos_abs_min: float,
) -> pd.DataFrame:
    b = bars.sort_values("ts").copy()
    if b.empty:
        return pd.DataFrame(columns=["ts", "trend", "strength", "slope_norm", "pos_norm"])
    ema = b["close"].ewm(span=ema_len, adjust=False, min_periods=ema_len).mean()
    a = atr(b, atr_len)
    slope_norm = (ema - ema.shift(slope_k)) / (float(max(slope_k, 1)) * a).replace(0, np.nan)
    pos_norm = (b["close"] - ema) / a.replace(0, np.nan)
    strength = (100.0 * (0.6 * slope_norm.abs() + 0.4 * pos_norm.abs()) / 1.5).clip(0.0, 100.0)
    trend = pd.Series("RANGE", index=b.index, dtype="object")
    trend.loc[(slope_norm > slope_abs_min) & (pos_norm > pos_abs_min)] = "BULL"
    trend.loc[(slope_norm < -slope_abs_min) & (pos_norm < -pos_abs_min)] = "BEAR"
    trend.loc[strength < range_strength_lt] = "RANGE"
    out = pd.DataFrame(
        {"ts": b["ts"], "trend": trend, "strength": strength, "slope_norm": slope_norm, "pos_norm": pos_norm}
    )
    return out.dropna(subset=["strength"])


def compute_pair_trends(price_bars: pd.DataFrame, as_of_ts: pd.Timestamp, cfg: FxBiasV2Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["pair", "timeframe", "ts", "trend", "strength", "slope_norm", "pos_norm"]
    if price_bars.empty:
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=cols)
    bars = price_bars[price_bars["ts"] <= as_of_ts].copy()
    if bars.empty:
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=cols)
    reg = cfg.price_regime
    thresholds = reg["thresholds"]
    slope_k = reg["slope_k"]
    latest_rows = []
    hist_rows = []
    for (pair, tf), grp in bars.groupby(["pair", "timeframe"], sort=False):
        ts = trend_series(
            grp,
            ema_len=int(reg["ema_len"]),
            atr_len=int(reg["atr_len"]),
            slope_k=int(slope_k.get(str(tf).upper(), 12)),
            range_strength_lt=float(thresholds["range_strength_lt"]),
            slope_abs_min=float(thresholds["slope_abs_min"]),
            pos_abs_min=float(thresholds["pos_abs_min"]),
        )
        if ts.empty:
            continue
        ts = ts.copy()
        ts["pair"] = pair
        ts["timeframe"] = tf
        hist_rows.append(ts)
        last = ts.iloc[-1]
        latest_rows.append(
            {
                "pair": pair,
                "timeframe": tf,
                "ts": last["ts"],
                "trend": str(last["trend"]),
                "strength": float(last["strength"]),
                "slope_norm": float(last["slope_norm"]),
                "pos_norm": float(last["pos_norm"]),
            }
        )
    latest = pd.DataFrame(latest_rows)
    hist = pd.concat(hist_rows, ignore_index=True) if hist_rows else pd.DataFrame(columns=cols)
    return latest, hist


def currency_regime_from_pairs(pair_snapshot: pd.DataFrame) -> Dict[str, str]:
    accum: Dict[str, List[float]] = {c: [] for c in G10_CCY}
    if pair_snapshot.empty:
        return {c: "RANGE" for c in G10_CCY}
    for _, row in pair_snapshot.iterrows():
        pair = str(row.get("pair", ""))
        trend = str(row.get("trend", "RANGE")).upper()
        strength = float(row.get("strength", 0.0))
        base, quote = pair_parts(pair)
        if base not in accum or quote not in accum or trend == "RANGE":
            continue
        s = 1.0 if trend == "BULL" else -1.0
        w = clip(strength / 100.0, 0.0, 1.0)
        accum[base].append(s * w)
        accum[quote].append(-s * w)
    out = {}
    for ccy in G10_CCY:
        vals = accum.get(ccy, [])
        score = float(np.mean(vals)) if vals else 0.0
        if score > 0.10:
            out[ccy] = "BULL"
        elif score < -0.10:
            out[ccy] = "BEAR"
        else:
            out[ccy] = "RANGE"
    return out


def d1_flips_last_10(pair_hist: pd.DataFrame, as_of_ts: pd.Timestamp) -> Dict[str, int]:
    out = {c: 0 for c in G10_CCY}
    if pair_hist.empty:
        return out
    d1 = pair_hist[(pair_hist["timeframe"] == "D1") & (pair_hist["ts"] <= as_of_ts)].copy()
    if d1.empty:
        return out
    rows = []
    for ts, snap in d1.groupby("ts", sort=True):
        reg = currency_regime_from_pairs(snap)
        for ccy, label in reg.items():
            rows.append({"ts": ts, "currency": ccy, "regime": label})
    if not rows:
        return out
    hist = pd.DataFrame(rows)
    hist["date"] = hist["ts"].dt.floor("D")
    for ccy in G10_CCY:
        seq = (
            hist[hist["currency"] == ccy]
            .sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")["regime"]
            .tolist()[-10:]
        )
        flips = 0
        for i in range(1, len(seq)):
            if seq[i] != seq[i - 1]:
                flips += 1
        out[ccy] = flips
    return out


def _solve_currency_returns(pair_returns: Dict[str, float], ridge_lambda: float) -> Optional[Dict[str, float]]:
    rows = []
    vals = []
    for pair, ret in pair_returns.items():
        b, q = pair_parts(pair)
        if b in G10_CCY and q in G10_CCY and not math.isnan(ret):
            rows.append((b, q))
            vals.append(float(ret))
    if len(vals) < 3:
        return None
    m = len(G10_CCY)
    A = np.zeros((len(vals), m), dtype=float)
    for i, (b, q) in enumerate(rows):
        A[i, G10_CCY.index(b)] = 1.0
        A[i, G10_CCY.index(q)] = -1.0
    r = np.array(vals, dtype=float)
    eta = 10.0
    A_aug = np.vstack([A, np.sqrt(eta) * np.ones((1, m), dtype=float)])
    r_aug = np.append(r, 0.0)
    lhs = A_aug.T @ A_aug + float(ridge_lambda) * np.eye(m)
    rhs = A_aug.T @ r_aug
    try:
        s = np.linalg.solve(lhs, rhs)
    except Exception:
        s = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return {ccy: float(s[i]) for i, ccy in enumerate(G10_CCY)}


def _proxy_currency_returns(pair_returns: Dict[str, float]) -> Dict[str, float]:
    obs: Dict[str, List[float]] = {c: [] for c in G10_CCY}
    for pair, ret in pair_returns.items():
        b, q = pair_parts(pair)
        if b in obs and q in obs and not math.isnan(ret):
            obs[b].append(float(ret))
            obs[q].append(-float(ret))
    out = {c: (float(np.mean(v)) if v else 0.0) for c, v in obs.items()}
    avg = float(np.mean(list(out.values())))
    return {c: out[c] - avg for c in G10_CCY}


def compute_price_strength(price_bars: pd.DataFrame, as_of_ts: pd.Timestamp, cfg: FxBiasV2Config) -> Tuple[Dict[str, Dict[int, Optional[float]]], float, List[str]]:
    warns: List[str] = []
    out = {c: {} for c in G10_CCY}
    cs = cfg.currency_strength
    horizons = [int(x) for x in cs["horizons_days"]]
    d1 = price_bars[(price_bars["timeframe"] == "D1") & (price_bars["ts"] <= as_of_ts)].copy()
    if d1.empty:
        for c in G10_CCY:
            for h in horizons:
                out[c][h] = None
        warns.append("PRICE D1 SERIES MISSING")
        return out, 0.0, warns
    d1["date"] = d1["ts"].dt.floor("D")
    d1 = d1.sort_values(["pair", "date", "ts"]).groupby(["pair", "date"], as_index=False).tail(1)
    close = d1.pivot(index="date", columns="pair", values="close").sort_index()
    pairs = [p for p in close.columns if len(clean_pair(str(p))) == 6]
    if not pairs:
        for c in G10_CCY:
            for h in horizons:
                out[c][h] = None
        warns.append("NO VALID FX PAIRS FOR PRICE STRENGTH")
        return out, 0.0, warns
    coverage = []
    for h in horizons:
        rets = np.log(close / close.shift(h))
        rows = []
        for date, r in rets.iterrows():
            pair_returns = {str(p): float(r[p]) for p in pairs if pd.notna(r[p])}
            if not pair_returns:
                continue
            cov = len(pair_returns) / float(max(len(pairs), 1))
            coverage.append(cov)
            solved = None
            if cov >= float(cs["min_pair_coverage_ratio"]) and len(pair_returns) >= 3:
                solved = _solve_currency_returns(pair_returns, float(cs["ridge_lambda"]))
            if solved is None:
                solved = _proxy_currency_returns(pair_returns)
            item = {"date": date}
            item.update(solved)
            rows.append(item)
        if not rows:
            for c in G10_CCY:
                out[c][h] = None
            warns.append(f"PRICE STRENGTH H{h}D MISSING")
            continue
        hist = pd.DataFrame(rows).sort_values("date")
        hist = hist[hist["date"] <= as_of_ts.floor("D")]
        for c in G10_CCY:
            z = rolling_z(pd.to_numeric(hist[c], errors="coerce"), int(cs["zscore_lookback_days"]))
            v = z.iloc[-1] if len(z) else np.nan
            out[c][h] = None if pd.isna(v) else float(clip(float(v), -float(cs["clip_z"]), float(cs["clip_z"])))
    return out, float(np.mean(coverage)) if coverage else 0.0, warns


def price_composite(
    strength: Dict[str, Dict[int, Optional[float]]],
    regime_w1: Dict[str, str],
    regime_d1: Dict[str, str],
) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {c: None for c in G10_CCY}

    def reg(v: str) -> float:
        if v == "BULL":
            return 1.0
        if v == "BEAR":
            return -1.0
        return 0.0

    for c in G10_CCY:
        s5 = strength.get(c, {}).get(5)
        s20 = strength.get(c, {}).get(20)
        s60 = strength.get(c, {}).get(60)
        if s5 is None or s20 is None or s60 is None:
            out[c] = None
            continue
        z5 = clip(float(s5) / 3.0, -1.0, 1.0)
        z20 = clip(float(s20) / 3.0, -1.0, 1.0)
        z60 = clip(float(s60) / 3.0, -1.0, 1.0)
        momentum = 0.30 * z5 + 0.40 * z20 + 0.30 * z60
        regime = 0.40 * reg(regime_w1.get(c, "RANGE")) + 0.60 * reg(regime_d1.get(c, "RANGE"))
        out[c] = float(clip(0.65 * momentum + 0.35 * regime, -1.0, 1.0))
    return out


def compute_rate_scores(rates: pd.DataFrame, as_of_ts: pd.Timestamp) -> Tuple[Dict[str, Optional[float]], float]:
    out: Dict[str, Optional[float]] = {c: None for c in G10_CCY}
    if rates.empty:
        return out, 0.0
    data = rates[rates["ts"] <= as_of_ts].copy()
    if data.empty:
        return out, 0.0
    for ccy, grp in data.groupby("currency"):
        if ccy not in out:
            continue
        g = grp.sort_values("ts").copy()
        g["d2y"] = g["y2_value"].diff(1)
        g["z"] = rolling_z(g["d2y"], 60)
        if len(g) and pd.notna(g["z"].iloc[-1]):
            out[ccy] = float(clip(float(g["z"].iloc[-1]) / 3.0, -1.0, 1.0))
    coverage = float(sum(v is not None for v in out.values()) / len(G10_CCY))
    return out, coverage


def detect_risk_regime(risk: pd.DataFrame, as_of_ts: pd.Timestamp) -> Tuple[str, float]:
    if risk.empty:
        return "NEUTRAL", 0.0
    data = risk[risk["ts"] <= as_of_ts].copy()
    if data.empty:
        return "NEUTRAL", 0.0
    eq_candidates = ["SPX", "SP500", "S&P500", "US100", "NASDAQ100", "NDX"]
    vol_candidates = ["VIX", "ATR_VOL", "ATRVOL", "VOL", "VOLATILITY"]
    eq = pd.DataFrame()
    vol = pd.DataFrame()
    for name in eq_candidates:
        sub = data[data["asset"] == name]
        if not sub.empty:
            eq = sub.sort_values("ts")
            break
    for name in vol_candidates:
        sub = data[data["asset"] == name]
        if not sub.empty:
            vol = sub.sort_values("ts")
            break
    if eq.empty or vol.empty or len(eq) < 5 or len(vol) < 5:
        return "NEUTRAL", 0.0
    eq_n = min(20, len(eq) - 1)
    vol_n = min(10, len(vol) - 1)
    if eq_n <= 0 or vol_n <= 0:
        return "NEUTRAL", 0.0
    eq_ret = float(eq["value"].iloc[-1] / eq["value"].iloc[-(eq_n + 1)] - 1.0)
    vol_ret = float(vol["value"].iloc[-1] / vol["value"].iloc[-(vol_n + 1)] - 1.0)
    if eq_ret < 0.0 and vol_ret > 0.0:
        return "RISK_OFF", 1.0
    if eq_ret > 0.0 and vol_ret < 0.0:
        return "RISK_ON", 1.0
    return "NEUTRAL", 1.0


def risk_overlay_scores(risk_regime: str, cfg: FxBiasV2Config) -> Dict[str, float]:
    out = {c: 0.0 for c in G10_CCY}
    if not cfg.risk_overlay["enabled"]:
        return out
    x = float(cfg.risk_overlay["jpy_chf_overlay_abs"])
    if risk_regime == "RISK_OFF":
        out["JPY"] = x
        out["CHF"] = x
    elif risk_regime == "RISK_ON":
        out["JPY"] = -x
        out["CHF"] = -x
    return out


def compute_carry_scores(carry: pd.DataFrame, as_of_ts: pd.Timestamp) -> Tuple[Dict[str, Optional[float]], Dict[str, int], float]:
    scores: Dict[str, Optional[float]] = {c: None for c in G10_CCY}
    pair_signs: Dict[str, int] = {}
    if carry.empty:
        return scores, pair_signs, 0.0
    data = carry[carry["ts"] <= as_of_ts].copy()
    if data.empty:
        return scores, pair_signs, 0.0
    data = data.sort_values(["pair", "ts"]).copy()
    data["date"] = data["ts"].dt.floor("D")
    data["sign"] = np.sign(pd.to_numeric(data["forward_points"], errors="coerce")).fillna(0).astype(int)
    latest = data.groupby("pair", as_index=False).tail(1)
    pair_signs = {str(r["pair"]): int(r["sign"]) for _, r in latest.iterrows()}
    rows = []
    for date, grp in data.groupby("date", sort=True):
        obs: Dict[str, List[float]] = {c: [] for c in G10_CCY}
        for _, row in grp.iterrows():
            b, q = pair_parts(str(row["pair"]))
            if b in obs and q in obs:
                s = float(row["sign"])
                obs[b].append(s)
                obs[q].append(-s)
        item = {"date": date}
        for ccy in G10_CCY:
            vals = obs[ccy]
            item[ccy] = float(np.mean(vals)) if vals else np.nan
        rows.append(item)
    if not rows:
        return scores, pair_signs, 0.0
    hist = pd.DataFrame(rows).sort_values("date")
    hist = hist[hist["date"] <= as_of_ts.floor("D")]
    for c in G10_CCY:
        z = rolling_z(pd.to_numeric(hist[c], errors="coerce"), 252)
        if len(z) and pd.notna(z.iloc[-1]):
            scores[c] = float(clip(float(z.iloc[-1]) / 3.0, -1.0, 1.0))
    coverage = float(sum(v is not None for v in scores.values()) / len(G10_CCY))
    return scores, pair_signs, coverage


def compute_skew_scores(skew: pd.DataFrame, as_of_ts: pd.Timestamp) -> Tuple[Dict[str, Optional[float]], float, bool]:
    out: Dict[str, Optional[float]] = {c: None for c in G10_CCY}
    if skew.empty:
        return out, 0.0, False
    data = skew[skew["ts"] <= as_of_ts].copy()
    if data.empty:
        return out, 0.0, False
    bad = bool(data["quality_flags"].astype(str).str.lower().str.contains("stale|bad|invalid", regex=True).any())
    rows = []
    for _, row in data.iterrows():
        sym = str(row.get("symbol_or_ccy", "")).upper().strip()
        rr = float(row["rr25"])
        if sym in G10_CCY:
            rows.append({"ts": row["ts"], "currency": sym, "rr25": rr})
        else:
            pair = clean_pair(sym)
            if len(pair) == 6:
                b, q = pair_parts(pair)
                if b in G10_CCY and q in G10_CCY:
                    rows.append({"ts": row["ts"], "currency": b, "rr25": rr})
                    rows.append({"ts": row["ts"], "currency": q, "rr25": -rr})
    if not rows:
        return out, 0.0, False
    mapped = pd.DataFrame(rows)
    mapped["date"] = mapped["ts"].dt.floor("D")
    agg = mapped.groupby(["currency", "date"], as_index=False)["rr25"].mean().sort_values(["currency", "date"])
    enough_history = True
    for c in G10_CCY:
        sub = agg[agg["currency"] == c]
        if len(sub) < 126:
            enough_history = False
        abs_z = rolling_z(sub["rr25"].abs(), 252)
        raw = np.sign(sub["rr25"]) * abs_z
        if len(raw) and pd.notna(raw.iloc[-1]):
            out[c] = float(clip(float(raw.iloc[-1]) / 3.0, -1.0, 1.0))
    coverage = float(sum(v is not None for v in out.values()) / len(G10_CCY))
    return out, coverage, bool((not bad) and enough_history)


def latest_cot_flags(cot: pd.DataFrame, as_of_ts: pd.Timestamp) -> Dict[str, Dict[str, bool]]:
    out = {c: {"extreme_flag": False, "persistence_flag": False} for c in G10_CCY}
    if cot.empty:
        return out
    data = cot.copy()
    if "ts" in data.columns:
        data = data[(data["ts"].isna()) | (data["ts"] <= as_of_ts)]
    for ccy, grp in data.groupby("currency"):
        if ccy in out and not grp.empty:
            row = grp.sort_values("ts").iloc[-1] if "ts" in grp.columns else grp.iloc[-1]
            out[ccy] = {
                "extreme_flag": bool(row.get("extreme_flag", False)),
                "persistence_flag": bool(row.get("persistence_flag", False)),
            }
    return out
