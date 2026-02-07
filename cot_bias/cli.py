from __future__ import annotations

import argparse
import datetime as dt
import subprocess
from pathlib import Path
from typing import Optional

import pandas as pd
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None

from .utils import load_config, ensure_dirs, parse_iso_date, most_recent_tuesday
from .data.store import update_store, load_store, snapshot_df, latest_per_symbol, data_max_report_date, write_manifest
from .metrics.features import compute_metrics
from .dashboard.render_updated import render_dashboard_updated
from .dashboard.tiered import compute_tiered_tables
from .dashboard.gate import compute_gated_fx_pairs
from .fx import usd_proxy_from_z, build_pairs_df
from .reporting import run_report, format_report_output, write_report_tables, resolve_report_date_by_release
from .filters.cot_filter import COTFilter, default_cot_filter_config


QUALITY_STRENGTH_MODES = {"B1_MIN_MAG", "B2_NOT_FADING", "B3_Z_GAP"}
QUALITY_POLICY_MODES = {"HARD_BLOCK", "SOFT_PENALTY"}


def cmd_fetch(args: argparse.Namespace) -> None:
    if not (args.update or args.combined or args.futures_only):
        raise ValueError("Use --update to refresh the local store.")
    if args.combined or args.futures_only:
        print("Note: --combined/--futures-only are deprecated; using the default historical futures-only data pipeline.")
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    info = update_store(cfg, force_refresh=args.force, verbose=args.verbose)
    print(f"Updated store: {info.store_path}")
    print(f"Rows: {info.row_count}")
    print(f"Max report_date: {info.max_report_date}")


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None


def _bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def _filter_long_allowed(decisions: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty:
        return decisions.iloc[0:0].copy()
    allow = (
        _bool_series(decisions["allow"]) if "allow" in decisions.columns else pd.Series([False] * len(decisions))
    )
    direction = (
        decisions.get("direction_allowed", pd.Series([""] * len(decisions))).astype(str).str.upper()
    )
    return decisions[allow & (direction == "LONG")].copy()


def _build_persistent_bullish_2w(
    current_longs: pd.DataFrame,
    previous_longs: pd.DataFrame,
) -> pd.DataFrame:
    persistent_cols = [
        "pair",
        "confidence_score",
        "spread",
        "dSpread_1w",
        "crowded_flag",
        "prev_spread",
        "prev_dSpread_1w",
        "prev_crowded_flag",
    ]
    if current_longs.empty or previous_longs.empty:
        return pd.DataFrame(columns=persistent_cols)

    current_view = current_longs[
        ["pair", "confidence_score", "spread", "dSpread_1w", "crowded_flag"]
    ].copy()
    prev_view = previous_longs[["pair", "spread", "dSpread_1w", "crowded_flag"]].copy()
    prev_view = prev_view.rename(
        columns={
            "spread": "prev_spread",
            "dSpread_1w": "prev_dSpread_1w",
            "crowded_flag": "prev_crowded_flag",
        }
    )

    persistent_df = current_view.merge(prev_view, on="pair", how="inner")
    if persistent_df.empty:
        return pd.DataFrame(columns=persistent_cols)

    persistent_df["confidence_score"] = pd.to_numeric(
        persistent_df["confidence_score"], errors="coerce"
    )
    persistent_df["spread"] = pd.to_numeric(persistent_df["spread"], errors="coerce")
    persistent_df["_abs_spread"] = persistent_df["spread"].abs()
    persistent_df = persistent_df.sort_values(
        ["confidence_score", "_abs_spread"], ascending=[False, False]
    ).drop(columns=["_abs_spread"])

    for col in persistent_cols:
        if col not in persistent_df.columns:
            persistent_df[col] = ""
    return persistent_df[persistent_cols].copy()


def _build_strong_persistent_bullish_3w(
    bullish_r0: pd.DataFrame,
    bullish_r1: pd.DataFrame,
    bullish_r2: pd.DataFrame,
) -> pd.DataFrame:
    strong_cols = [
        "pair",
        "confidence_score",
        "spread",
        "dSpread_1w",
        "crowded_flag",
        "prev1_spread",
        "prev2_spread",
    ]
    if bullish_r0.empty or bullish_r1.empty or bullish_r2.empty:
        return pd.DataFrame(columns=strong_cols)

    r0_pairs = set(bullish_r0["pair"].astype(str))
    r1_pairs = set(bullish_r1["pair"].astype(str))
    r2_pairs = set(bullish_r2["pair"].astype(str))
    persistent_pairs = r0_pairs & r1_pairs & r2_pairs
    if not persistent_pairs:
        return pd.DataFrame(columns=strong_cols)

    r0 = bullish_r0[bullish_r0["pair"].astype(str).isin(persistent_pairs)][
        ["pair", "confidence_score", "spread", "dSpread_1w", "crowded_flag"]
    ].copy()
    r1 = bullish_r1[bullish_r1["pair"].astype(str).isin(persistent_pairs)][["pair", "spread"]].copy()
    r2 = bullish_r2[bullish_r2["pair"].astype(str).isin(persistent_pairs)][["pair", "spread"]].copy()
    r1 = r1.rename(columns={"spread": "prev1_spread"})
    r2 = r2.rename(columns={"spread": "prev2_spread"})

    r0 = r0.drop_duplicates(subset=["pair"], keep="first")
    r1 = r1.drop_duplicates(subset=["pair"], keep="first")
    r2 = r2.drop_duplicates(subset=["pair"], keep="first")

    out = r0.merge(r1, on="pair", how="inner").merge(r2, on="pair", how="inner")
    if out.empty:
        return pd.DataFrame(columns=strong_cols)

    out["confidence_score"] = pd.to_numeric(out["confidence_score"], errors="coerce")
    out["spread"] = pd.to_numeric(out["spread"], errors="coerce")
    out["_abs_spread"] = out["spread"].abs()
    out = out.sort_values(["confidence_score", "_abs_spread"], ascending=[False, False]).drop(
        columns=["_abs_spread"]
    )

    for col in strong_cols:
        if col not in out.columns:
            out[col] = ""
    return out[strong_cols].copy()


def _pair_parts(pair: str) -> tuple[str, str]:
    p = str(pair).strip().upper().replace("/", "")
    if len(p) >= 6:
        return p[:3], p[-3:]
    return "", ""


def _to_float(value: object) -> float:
    try:
        out = float(value)
        return out if pd.notna(out) else float("nan")
    except Exception:
        return float("nan")


def _quality_tier(score: float) -> str:
    if score >= 75.0:
        return "ELITE"
    if score >= 55.0:
        return "GOOD"
    if score >= 40.0:
        return "OK"
    if score >= 1.0:
        return "WEAK"
    return "FAIL"


def _load_quality_cfg(raw_cfg: dict) -> dict:
    cfg = {
        "quality_strength_mode": "B1_MIN_MAG",
        "quality_crowding_policy": "SOFT_PENALTY",
        "quality_collapse_policy": "SOFT_PENALTY",
    }
    if not isinstance(raw_cfg, dict):
        return cfg
    block = raw_cfg.get("quality_gates")
    merged = dict(cfg)
    if isinstance(block, dict):
        merged.update(block)
    for key in cfg:
        if key in raw_cfg:
            merged[key] = raw_cfg[key]

    strength_mode = str(merged.get("quality_strength_mode", cfg["quality_strength_mode"])).strip().upper()
    if strength_mode not in QUALITY_STRENGTH_MODES:
        strength_mode = cfg["quality_strength_mode"]
    crowding_policy = str(merged.get("quality_crowding_policy", cfg["quality_crowding_policy"])).strip().upper()
    if crowding_policy not in QUALITY_POLICY_MODES:
        crowding_policy = cfg["quality_crowding_policy"]
    collapse_policy = str(merged.get("quality_collapse_policy", cfg["quality_collapse_policy"])).strip().upper()
    if collapse_policy not in QUALITY_POLICY_MODES:
        collapse_policy = cfg["quality_collapse_policy"]

    return {
        "quality_strength_mode": strength_mode,
        "quality_crowding_policy": crowding_policy,
        "quality_collapse_policy": collapse_policy,
    }


def _release_eligible_as_of(
    as_of: dt.date,
    resolved_release_dt: Optional[dt.datetime],
    release_cfg: dict,
) -> bool:
    if resolved_release_dt is None:
        return False
    rel = resolved_release_dt
    if rel.tzinfo is None:
        rel = rel.replace(tzinfo=dt.timezone.utc)
    try:
        from zoneinfo import ZoneInfo

        tz = ZoneInfo(str(release_cfg.get("timezone", "America/New_York")))
        requested_dt = dt.datetime(
            as_of.year,
            as_of.month,
            as_of.day,
            23,
            59,
            59,
            tzinfo=tz,
        ).astimezone(dt.timezone.utc)
        return rel.astimezone(dt.timezone.utc) <= requested_dt
    except Exception:
        return rel.date() <= as_of


def _build_quality_gates_score(
    *,
    strong_persistent: pd.DataFrame,
    decisions_r0: pd.DataFrame,
    bullish_r1: pd.DataFrame,
    bullish_r2: pd.DataFrame,
    pairs_snapshot: pd.DataFrame,
    instruments_snapshot: pd.DataFrame,
    as_of: dt.date,
    resolved_release_dt: Optional[dt.datetime],
    release_cfg: dict,
    quality_cfg: dict,
) -> pd.DataFrame:
    quality_cols = [
        "pair",
        "direction_allowed",
        "quality_score",
        "quality_tier",
        "spread",
        "dSpread_1w",
        "crowded_flag",
        "prev1_spread",
        "prev2_spread",
        "gap_z",
        "A_release_ok",
        "A_consecutive_ok",
        "A_sign_ok",
        "B_mode",
        "B_strength_ok",
        "B_strength_points",
        "C_crowding_policy",
        "C_crowding_ok",
        "C_crowding_points",
        "C_base_reversal_risk",
        "C_quote_reversal_risk",
        "C_reversal_points",
        "D_no_collapse_ok",
        "D_points",
    ]
    if strong_persistent.empty:
        return pd.DataFrame(columns=quality_cols)

    mode = str(quality_cfg.get("quality_strength_mode", "B1_MIN_MAG")).strip().upper()
    crowding_policy = str(quality_cfg.get("quality_crowding_policy", "SOFT_PENALTY")).strip().upper()
    collapse_policy = str(quality_cfg.get("quality_collapse_policy", "SOFT_PENALTY")).strip().upper()

    release_ok_global = _release_eligible_as_of(as_of, resolved_release_dt, release_cfg)

    decisions = decisions_r0.copy() if isinstance(decisions_r0, pd.DataFrame) else pd.DataFrame()
    if not decisions.empty:
        decisions["pair"] = decisions["pair"].astype(str).str.upper()
        decisions["direction_allowed"] = (
            decisions.get("direction_allowed", pd.Series(["LONG"] * len(decisions)))
            .astype(str)
            .str.upper()
        )
        direction_map = dict(zip(decisions["pair"], decisions["direction_allowed"]))
    else:
        direction_map = {}

    r1_pairs = (
        set(bullish_r1["pair"].astype(str).str.upper().tolist())
        if isinstance(bullish_r1, pd.DataFrame) and not bullish_r1.empty and "pair" in bullish_r1.columns
        else set()
    )
    r2_pairs = (
        set(bullish_r2["pair"].astype(str).str.upper().tolist())
        if isinstance(bullish_r2, pd.DataFrame) and not bullish_r2.empty and "pair" in bullish_r2.columns
        else set()
    )

    pair_map = {}
    if isinstance(pairs_snapshot, pd.DataFrame) and not pairs_snapshot.empty and "pair" in pairs_snapshot.columns:
        pairs = pairs_snapshot.copy()
        pairs["pair"] = pairs["pair"].astype(str).str.upper()
        if "base" not in pairs.columns or "quote" not in pairs.columns:
            parts = pairs["pair"].apply(_pair_parts)
            pairs["base"] = [p[0] for p in parts]
            pairs["quote"] = [p[1] for p in parts]
        if "gap_z" not in pairs.columns:
            if "base_z" in pairs.columns and "quote_z" in pairs.columns:
                pairs["base_z"] = pd.to_numeric(pairs["base_z"], errors="coerce")
                pairs["quote_z"] = pd.to_numeric(pairs["quote_z"], errors="coerce")
                pairs["gap_z"] = pairs["base_z"] - pairs["quote_z"]
            else:
                pairs["gap_z"] = float("nan")
        pairs = pairs.drop_duplicates(subset=["pair"], keep="first")
        pair_map = {
            str(row["pair"]).upper(): {
                "base": str(row.get("base", "")).upper(),
                "quote": str(row.get("quote", "")).upper(),
                "gap_z": _to_float(row.get("gap_z")),
            }
            for _, row in pairs.iterrows()
        }

    risk_map = {}
    if isinstance(instruments_snapshot, pd.DataFrame) and not instruments_snapshot.empty and "symbol" in instruments_snapshot.columns:
        inst = instruments_snapshot.copy()
        inst["symbol"] = inst["symbol"].astype(str).str.upper()
        inst["reversal_risk"] = inst.get("reversal_risk", pd.Series(["UNKNOWN"] * len(inst))).astype(str).str.upper()
        inst = inst.drop_duplicates(subset=["symbol"], keep="first")
        risk_map = dict(zip(inst["symbol"], inst["reversal_risk"]))

    rows = []
    strong = strong_persistent.copy()
    strong["pair"] = strong["pair"].astype(str).str.upper()
    for _, row in strong.iterrows():
        pair = str(row.get("pair", "")).upper()
        direction = str(direction_map.get(pair, "LONG")).upper()
        if direction not in {"LONG", "SHORT"}:
            direction = "LONG"

        spread = _to_float(row.get("spread"))
        dspread = _to_float(row.get("dSpread_1w"))
        prev1 = _to_float(row.get("prev1_spread"))
        prev2 = _to_float(row.get("prev2_spread"))
        crowded_flag = str(row.get("crowded_flag", "")).strip().upper()
        if not crowded_flag:
            crowded_flag = "UNKNOWN"

        pair_info = pair_map.get(pair, {})
        base = str(pair_info.get("base", "")).upper()
        quote = str(pair_info.get("quote", "")).upper()
        if not base or not quote:
            base, quote = _pair_parts(pair)
        gap_z = _to_float(pair_info.get("gap_z"))

        base_risk = str(risk_map.get(base, "UNKNOWN")).upper()
        quote_risk = str(risk_map.get(quote, "UNKNOWN")).upper()

        a_release_ok = bool(release_ok_global)
        a_consecutive_ok = pair in r1_pairs and pair in r2_pairs
        if direction == "SHORT":
            a_sign_ok = pd.notna(spread) and pd.notna(prev1) and pd.notna(prev2) and spread < 0 and prev1 < 0 and prev2 < 0
        else:
            a_sign_ok = pd.notna(spread) and pd.notna(prev1) and pd.notna(prev2) and spread > 0 and prev1 > 0 and prev2 > 0
        a_hard_fail = not (a_release_ok and a_consecutive_ok and a_sign_ok)

        b_ok = False
        b_points = 0
        if mode == "B1_MIN_MAG":
            mag = abs(spread) if pd.notna(spread) else float("nan")
            b_ok = pd.notna(mag) and mag >= 35.0
            if pd.notna(mag):
                if mag >= 65.0:
                    b_points = 30
                elif mag >= 50.0:
                    b_points = 20
                elif mag >= 35.0:
                    b_points = 10
        elif mode == "B2_NOT_FADING":
            mags = [abs(v) for v in [prev2, prev1, spread] if pd.notna(v)]
            min_mag = min(mags) if len(mags) == 3 else float("nan")
            b_ok = pd.notna(min_mag) and min_mag >= 30.0
            if pd.notna(min_mag):
                if min_mag >= 55.0:
                    b_points = 30
                elif min_mag >= 40.0:
                    b_points = 20
                elif min_mag >= 30.0:
                    b_points = 10
        else:  # B3_Z_GAP
            if pd.notna(gap_z):
                if direction == "SHORT":
                    b_ok = gap_z <= -2.3
                    if gap_z <= -3.0:
                        b_points = 30
                    elif gap_z <= -2.6:
                        b_points = 20
                    elif gap_z <= -2.3:
                        b_points = 10
                else:
                    b_ok = gap_z >= 2.3
                    if gap_z >= 3.0:
                        b_points = 30
                    elif gap_z >= 2.6:
                        b_points = 20
                    elif gap_z >= 2.3:
                        b_points = 10

        crowding_block = False
        if crowded_flag == "NONE":
            c_crowding_ok = True
            c_crowding_points = 20
        else:
            if crowding_policy == "HARD_BLOCK":
                c_crowding_ok = False
                c_crowding_points = 0
                crowding_block = True
            else:
                c_crowding_ok = True
                c_crowding_points = -10

        reversal_extreme = (base_risk == "EXTREME") or (quote_risk == "EXTREME")
        c_reversal_points = -15 if reversal_extreme else 0
        reversal_block = bool(reversal_extreme and crowding_policy == "HARD_BLOCK")
        c_points = c_crowding_points + c_reversal_points

        collapse_block = False
        d_no_collapse_ok = False
        if pd.notna(spread) and pd.notna(dspread):
            bound = 0.15 * abs(spread)
            if direction == "SHORT":
                d_no_collapse_ok = dspread <= bound
            else:
                d_no_collapse_ok = dspread >= -bound
        if d_no_collapse_ok:
            d_points = 15
        else:
            if collapse_policy == "HARD_BLOCK":
                d_points = 0
                collapse_block = True
            else:
                d_points = -10

        hard_fail = a_hard_fail or crowding_block or reversal_block or collapse_block
        if hard_fail:
            quality_score = 0.0
            tier = "FAIL"
        else:
            quality_score = float(b_points + c_points + d_points)
            quality_score = max(0.0, min(100.0, quality_score))
            tier = _quality_tier(quality_score)

        rows.append(
            {
                "pair": pair,
                "direction_allowed": direction,
                "quality_score": quality_score,
                "quality_tier": tier,
                "spread": spread,
                "dSpread_1w": dspread,
                "crowded_flag": crowded_flag,
                "prev1_spread": prev1,
                "prev2_spread": prev2,
                "gap_z": gap_z,
                "A_release_ok": a_release_ok,
                "A_consecutive_ok": a_consecutive_ok,
                "A_sign_ok": a_sign_ok,
                "B_mode": mode,
                "B_strength_ok": b_ok,
                "B_strength_points": int(b_points),
                "C_crowding_policy": crowding_policy,
                "C_crowding_ok": c_crowding_ok,
                "C_crowding_points": int(c_crowding_points),
                "C_base_reversal_risk": base_risk,
                "C_quote_reversal_risk": quote_risk,
                "C_reversal_points": int(c_reversal_points),
                "D_no_collapse_ok": d_no_collapse_ok,
                "D_points": int(d_points),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=quality_cols)
    out = out.sort_values(["quality_score", "pair"], ascending=[False, True]).reset_index(drop=True)
    for col in quality_cols:
        if col not in out.columns:
            out[col] = ""
    return out[quality_cols].copy()


def _load_raw_config(config_path: Optional[str]) -> dict:
    if yaml is None:
        return {}
    path = Path(config_path) if config_path else Path("config.yaml")
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    return doc if isinstance(doc, dict) else {}


def _release_dt_for_report(report_date: dt.date, release_cfg: dict) -> dt.datetime:
    wd_map = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }
    target = wd_map.get(str(release_cfg.get("release_weekday", "friday")).strip().lower())
    if target is None:
        raise ValueError(f"Invalid release_weekday: {release_cfg.get('release_weekday')}")

    wd = report_date.weekday()
    delta = (target - wd) % 7
    if delta == 0:
        delta = 7
    release_date = report_date + dt.timedelta(days=delta)

    hour, minute = [int(x) for x in str(release_cfg.get("release_time", "15:30")).split(":")]
    tz = dt.timezone.utc
    try:
        from zoneinfo import ZoneInfo

        tz = ZoneInfo(release_cfg.get("timezone", "America/New_York"))
    except Exception:
        tz = dt.timezone.utc
    return dt.datetime(
        release_date.year,
        release_date.month,
        release_date.day,
        hour,
        minute,
        tzinfo=tz,
    ).astimezone(dt.timezone.utc)


def _fx_release_dates(metrics: pd.DataFrame) -> list[dt.date]:
    if metrics.empty or "report_date" not in metrics.columns:
        return []
    df = metrics.copy()
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce").dt.date
    if "report_type" in df.columns:
        df = df[df["report_type"] == "tff"]
    dates = sorted(d for d in df["report_date"].dropna().unique())
    return dates


def _resolve_report_date(
    metrics: pd.DataFrame,
    as_of: dt.date,
    config_path: Optional[str],
    resolve_mode: str,
    requested_report_date: Optional[dt.date],
) -> tuple[dt.date, dt.datetime, dict, Optional[dt.date], Optional[dt.datetime]]:
    raw_cfg = _load_raw_config(config_path)
    cot_filter_cfg = raw_cfg.get("cot_filter") if isinstance(raw_cfg, dict) else None
    cot_filter = COTFilter(cot_filter_cfg)
    release_cfg = cot_filter.cfg.get(
        "release_alignment", default_cot_filter_config()["release_alignment"]
    )

    if metrics.empty or "report_date" not in metrics.columns:
        raise RuntimeError("Missing report_date column; cannot resolve release-aligned date.")
    df = metrics.copy()
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce").dt.date
    dates = sorted(d for d in df["report_date"].dropna().unique())
    if not dates:
        raise RuntimeError("No report_date values available to resolve release alignment.")

    mode = str(resolve_mode or "release_aligned").strip().lower()
    if mode == "report_date_direct":
        if requested_report_date:
            target = requested_report_date
        else:
            target = most_recent_tuesday(as_of)
        if target not in dates:
            raise RuntimeError(
                f"Requested report_date {target} not available in dataset."
            )
        resolved_report_date = target
    else:
        resolved_report_date = resolve_report_date_by_release(as_of, dates, release_cfg)

    resolved_release_dt = _release_dt_for_report(resolved_report_date, release_cfg)
    next_report_date = None
    next_release_dt = None
    for d in dates:
        if d > resolved_report_date:
            next_report_date = d
            next_release_dt = _release_dt_for_report(d, release_cfg)
            break

    return resolved_report_date, resolved_release_dt, release_cfg, next_report_date, next_release_dt


def _assert_no_lookahead(df: pd.DataFrame, resolved_report_date: dt.date, label: str) -> None:
    if df.empty or "report_date" not in df.columns:
        return
    dates = pd.to_datetime(df["report_date"], errors="coerce").dt.date
    max_date = dates.max()
    if max_date and max_date > resolved_report_date:
        raise RuntimeError(
            f"Lookahead detected: {label} report_date {max_date} exceeds resolved_report_date {resolved_report_date}"
        )


def _filter_metrics_for_report_date(
    metrics: pd.DataFrame, resolved_report_date: dt.date
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if metrics.empty or "report_date" not in metrics.columns:
        raise RuntimeError("Missing report_date column; cannot filter metrics.")
    df = metrics.copy()
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce").dt.date
    hist = df[df["report_date"] <= resolved_report_date].copy()
    latest = hist[hist["report_date"] == resolved_report_date].copy()
    if latest.empty:
        raise RuntimeError("Resolved report_date not present in metrics; cannot build dashboard.")
    _assert_no_lookahead(hist, resolved_report_date, "metrics_history")
    _assert_no_lookahead(latest, resolved_report_date, "metrics_latest")
    return hist, latest


def _fx_snapshot_for_release(metrics: pd.DataFrame, release_date: dt.date) -> pd.DataFrame:
    if metrics.empty or "report_date" not in metrics.columns:
        return pd.DataFrame()
    df = metrics.copy()
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce").dt.date
    if "report_type" in df.columns:
        df = df[df["report_type"] == "tff"]
    return df[df["report_date"] == release_date].copy()


def _pairs_for_release(fx_release: pd.DataFrame) -> pd.DataFrame:
    if fx_release.empty or "z_3y" not in fx_release.columns:
        return pd.DataFrame()
    z_by = {
        row["symbol"]: float(row["z_3y"]) if pd.notna(row["z_3y"]) else float("nan")
        for _, row in fx_release.iterrows()
    }
    usd_z = usd_proxy_from_z(z_by, weights=None)
    return build_pairs_df(z_by, usd_z, usd_mode="basket")


def _build_dashboard_context(
    *,
    cfg,
    as_of: Optional[dt.date],
    force_refresh: bool,
    verbose: bool,
    config_path: Optional[str],
    resolve_mode: str,
    requested_report_date: Optional[dt.date],
) -> dict:
    if force_refresh:
        update_store(cfg, force_refresh=True, verbose=verbose)

    df = load_store(cfg)
    max_date = data_max_report_date(df)
    warnings = []

    if as_of is None:
        if max_date is None:
            raise RuntimeError("No rows available in store; run `cot_bias fetch --update` first.")
        as_of = max_date

    if max_date and max_date < as_of:
        try:
            update_store(cfg, force_refresh=True, verbose=verbose)
            df = load_store(cfg)
            max_date = data_max_report_date(df)
        except Exception as exc:
            warnings.append(f"AUTO-REFRESH FAILED: {exc}")

    snap = snapshot_df(df, as_of)
    if snap.empty:
        raise RuntimeError("No rows available at or before as_of date.")

    metrics = compute_metrics(snap, window=cfg.rolling_weeks)
    resolved_report_date, resolved_release_dt, release_cfg, next_report_date, next_release_dt = _resolve_report_date(
        metrics, as_of, config_path, resolve_mode, requested_report_date
    )
    metrics_hist, metrics_latest = _filter_metrics_for_report_date(metrics, resolved_report_date)

    latest = latest_per_symbol(metrics_latest)
    if "report_date" in latest.columns:
        latest["report_date_used"] = latest["report_date"]

    pairs_df = None
    if "report_type" in metrics_latest.columns:
        fx_latest = metrics_latest[metrics_latest["report_type"] == "tff"].copy()
        if not fx_latest.empty and "z_3y" in fx_latest.columns:
            z_by = {
                row["symbol"]: float(row["z_3y"]) if pd.notna(row["z_3y"]) else float("nan")
                for _, row in fx_latest.iterrows()
            }
            usd_z = usd_proxy_from_z(z_by, weights=None)
            pairs_df = build_pairs_df(z_by, usd_z, usd_mode="basket")

    fx_dates = _fx_release_dates(metrics_hist)
    latest_release_date = resolved_report_date
    previous_release_date = None
    previous2_release_date = None
    if fx_dates:
        prior = [d for d in fx_dates if d < resolved_report_date]
        previous_release_date = prior[-1] if prior else None
        previous2_release_date = prior[-2] if len(prior) >= 2 else None
    previous_release_dt = (
        _release_dt_for_report(previous_release_date, release_cfg)
        if previous_release_date
        else None
    )
    fx_latest_release = _fx_snapshot_for_release(metrics_hist, latest_release_date)
    fx_prev_release = (
        _fx_snapshot_for_release(metrics_hist, previous_release_date)
        if previous_release_date
        else pd.DataFrame()
    )
    pairs_latest_gate = _pairs_for_release(fx_latest_release)
    pairs_prev_gate = _pairs_for_release(fx_prev_release) if previous_release_date else pd.DataFrame()

    gated_pairs_df = compute_gated_fx_pairs(
        currency_latest=fx_latest_release,
        currency_previous=fx_prev_release,
        pairs_latest=pairs_latest_gate,
        pairs_previous=pairs_prev_gate,
        latest_release_date=latest_release_date,
        previous_release_date=previous_release_date,
    )

    if max_date and max_date < as_of:
        warnings.append(
            f"DATA STALE: max report_date is {max_date.isoformat()}, as_of is {as_of.isoformat()}"
        )

    return {
        "as_of": as_of,
        "max_date": max_date,
        "warnings": warnings,
        "snap": snap,
        "metrics_hist": metrics_hist,
        "metrics_latest": metrics_latest,
        "latest": latest,
        "pairs_df": pairs_df,
        "gated_pairs_df": gated_pairs_df,
        "fx_latest_release": fx_latest_release,
        "pairs_latest_gate": pairs_latest_gate,
        "latest_release_date": latest_release_date,
        "previous_release_date": previous_release_date,
        "previous2_release_date": previous2_release_date,
        "previous_release_dt": previous_release_dt,
        "resolved_report_date": resolved_report_date,
        "resolved_release_dt": resolved_release_dt,
        "release_cfg": release_cfg,
        "next_report_date": next_report_date,
        "next_release_dt": next_release_dt,
        "resolve_mode": resolve_mode,
        "requested_report_date": requested_report_date,
    }


def cmd_dashboard(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    ensure_dirs(cfg)

    as_of = parse_iso_date(args.as_of)
    requested_report_date = parse_iso_date(args.report_date) if args.report_date else None

    ctx = _build_dashboard_context(
        cfg=cfg,
        as_of=as_of,
        force_refresh=args.force_refresh,
        verbose=args.verbose,
        config_path=args.config,
        resolve_mode=args.resolve_mode,
        requested_report_date=requested_report_date,
    )
    as_of = ctx["as_of"]
    max_date = ctx["max_date"]
    warnings = ctx["warnings"]
    snap = ctx["snap"]
    metrics_hist = ctx["metrics_hist"]
    metrics_latest = ctx["metrics_latest"]
    latest = ctx["latest"]
    pairs_df = ctx["pairs_df"]
    gated_pairs_df = ctx["gated_pairs_df"]
    fx_latest_release = ctx.get("fx_latest_release")
    pairs_latest_gate = ctx.get("pairs_latest_gate")
    latest_release_date = ctx.get("latest_release_date")
    resolved_report_date = ctx.get("resolved_report_date")
    resolved_release_dt = ctx.get("resolved_release_dt")
    next_report_date = ctx.get("next_report_date")
    next_release_dt = ctx.get("next_release_dt")
    previous_release_date = ctx.get("previous_release_date")
    previous2_release_date = ctx.get("previous2_release_date")
    previous_release_dt = ctx.get("previous_release_dt")
    resolve_mode = ctx.get("resolve_mode")
    requested_report_date = ctx.get("requested_report_date")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    history_path = out_dir / "instruments_history.csv"
    latest_path = out_dir / "instruments_latest.csv"
    metrics_latest.to_csv(history_path, index=False)
    latest.to_csv(latest_path, index=False)
    if pairs_df is not None and not pairs_df.empty:
        pairs_df.to_csv(out_dir / "pairs_latest.csv", index=False)

    tier_tables = None
    tier_meta = None
    raw_cfg = _load_raw_config(args.config)
    cot_filter_cfg = raw_cfg.get("cot_filter") if isinstance(raw_cfg, dict) else None
    quality_cfg = _load_quality_cfg(raw_cfg)
    if args.tier:
        if cot_filter_cfg is None:
            cot_filter_cfg = {}
        cot_filter_cfg = dict(cot_filter_cfg)
        cot_filter_cfg["strictness_tier"] = args.tier
    try:
        tier_tables, tier_meta = compute_tiered_tables(
            history_df=metrics_hist,
            pairs_df=pairs_latest_gate,
            as_of=as_of,
            cot_filter_cfg=cot_filter_cfg,
            report_date_override=resolved_report_date,
        )
        if tier_tables:
            write_report_tables(tier_tables, out_dir)
    except Exception as exc:
        warnings.append(f"TIERED TABLES FAILED: {exc}")

    persistent_df = pd.DataFrame(
        columns=[
            "pair",
            "confidence_score",
            "spread",
            "dSpread_1w",
            "crowded_flag",
            "prev_spread",
            "prev_dSpread_1w",
            "prev_crowded_flag",
        ]
    )
    strong_persistent_df = pd.DataFrame(
        columns=[
            "pair",
            "confidence_score",
            "spread",
            "dSpread_1w",
            "crowded_flag",
            "prev1_spread",
            "prev2_spread",
        ]
    )
    current_longs = pd.DataFrame()
    decisions_r0 = pd.DataFrame()
    if tier_tables and "decisions_summary" in tier_tables:
        decisions_r0 = tier_tables["decisions_summary"]
        current_longs = _filter_long_allowed(decisions_r0)

    if previous_release_date:
        try:
            prev_tables, _ = compute_tiered_tables(
                history_df=metrics_hist,
                pairs_df=pairs_latest_gate,
                as_of=as_of,
                cot_filter_cfg=cot_filter_cfg,
                report_date_override=previous_release_date,
            )
        except Exception as exc:
            prev_tables = None
            warnings.append(f"PERSISTENT BULLISH FAILED: {exc}")
    else:
        prev_tables = None

    if previous2_release_date:
        try:
            prev2_tables, _ = compute_tiered_tables(
                history_df=metrics_hist,
                pairs_df=pairs_latest_gate,
                as_of=as_of,
                cot_filter_cfg=cot_filter_cfg,
                report_date_override=previous2_release_date,
            )
        except Exception as exc:
            prev2_tables = None
            warnings.append(f"STRONG PERSISTENCE FAILED: {exc}")
    else:
        prev2_tables = None

    prev_longs = pd.DataFrame()
    if prev_tables and "decisions_summary" in prev_tables:
        prev_longs = _filter_long_allowed(prev_tables["decisions_summary"])
    prev2_longs = pd.DataFrame()
    if prev2_tables and "decisions_summary" in prev2_tables:
        prev2_longs = _filter_long_allowed(prev2_tables["decisions_summary"])

    if not current_longs.empty and not prev_longs.empty:
        persistent_df = _build_persistent_bullish_2w(current_longs, prev_longs)
    if previous2_release_date:
        strong_persistent_df = _build_strong_persistent_bullish_3w(
            bullish_r0=current_longs,
            bullish_r1=prev_longs,
            bullish_r2=prev2_longs,
        )

    persistent_path = out_dir / "persistent_bullish_2w.csv"
    persistent_df.to_csv(persistent_path, index=False)
    strong_persistent_path = out_dir / "strong_persistent_bullish_3w.csv"
    strong_persistent_df.to_csv(strong_persistent_path, index=False)
    quality_df = _build_quality_gates_score(
        strong_persistent=strong_persistent_df,
        decisions_r0=decisions_r0,
        bullish_r1=prev_longs,
        bullish_r2=prev2_longs,
        pairs_snapshot=pairs_latest_gate if isinstance(pairs_latest_gate, pd.DataFrame) else pd.DataFrame(),
        instruments_snapshot=latest if isinstance(latest, pd.DataFrame) else pd.DataFrame(),
        as_of=as_of,
        resolved_release_dt=resolved_release_dt,
        release_cfg=ctx.get("release_cfg") or {},
        quality_cfg=quality_cfg,
    )
    quality_path = out_dir / "quality_gates_score.csv"
    quality_df.to_csv(quality_path, index=False)

    snapshot_max = snap["report_date"].max() if not snap.empty else None
    write_manifest(
        out_dir=out_dir,
        as_of=as_of,
        store_max_report_date=max_date,
        snapshot_max_report_date=snapshot_max,
        row_count=len(snap),
        instruments=sorted(latest["symbol"].astype(str).unique().tolist()),
        warnings=warnings,
        store_path=cfg.processed_dir / "cot.parquet",
        git_commit=_git_commit(),
        resolved_report_date=resolved_report_date,
        resolved_release_dt=resolved_release_dt,
        requested_report_date=requested_report_date,
        next_report_date=next_report_date,
        next_release_dt=next_release_dt,
        resolve_mode=resolve_mode,
        previous_report_date=previous_release_date,
        previous2_report_date=previous2_release_date,
        previous_release_dt=previous_release_dt,
    )

    html_path = render_dashboard_updated(
        base_dir=out_dir,
        output_path=out_dir / "dashboard.html",
    )
    updated_path = render_dashboard_updated(
        base_dir=out_dir,
        output_path=out_dir / "dashboard_updated.html",
    )

    print(f"Wrote: {html_path}")
    print(f"Wrote: {updated_path}")


def cmd_compute(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    ensure_dirs(cfg)

    as_of = parse_iso_date(args.as_of)
    requested_report_date = parse_iso_date(args.report_date) if args.report_date else None
    ctx = _build_dashboard_context(
        cfg=cfg,
        as_of=as_of,
        force_refresh=args.force_refresh,
        verbose=args.verbose,
        config_path=args.config,
        resolve_mode=args.resolve_mode,
        requested_report_date=requested_report_date,
    )
    as_of = ctx["as_of"]
    max_date = ctx["max_date"]
    warnings = ctx["warnings"]
    snap = ctx["snap"]
    metrics_latest = ctx["metrics_latest"]
    latest = ctx["latest"]
    pairs_df = ctx["pairs_df"]
    resolved_report_date = ctx.get("resolved_report_date")
    resolved_release_dt = ctx.get("resolved_release_dt")
    next_report_date = ctx.get("next_report_date")
    next_release_dt = ctx.get("next_release_dt")
    previous_release_date = ctx.get("previous_release_date")
    previous2_release_date = ctx.get("previous2_release_date")
    previous_release_dt = ctx.get("previous_release_dt")
    resolve_mode = ctx.get("resolve_mode")
    requested_report_date = ctx.get("requested_report_date")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    history_path = out_dir / "instruments_history.csv"
    latest_path = out_dir / "instruments_latest.csv"
    metrics_latest.to_csv(history_path, index=False)
    latest.to_csv(latest_path, index=False)
    if pairs_df is not None and not pairs_df.empty:
        pairs_df.to_csv(out_dir / "pairs_latest.csv", index=False)

    snapshot_max = snap["report_date"].max() if not snap.empty else None
    write_manifest(
        out_dir=out_dir,
        as_of=as_of,
        store_max_report_date=max_date,
        snapshot_max_report_date=snapshot_max,
        row_count=len(snap),
        instruments=sorted(latest["symbol"].astype(str).unique().tolist()),
        warnings=warnings,
        store_path=cfg.processed_dir / "cot.parquet",
        git_commit=_git_commit(),
        resolved_report_date=resolved_report_date,
        resolved_release_dt=resolved_release_dt,
        requested_report_date=requested_report_date,
        next_report_date=next_report_date,
        next_release_dt=next_release_dt,
        resolve_mode=resolve_mode,
        previous_report_date=previous_release_date,
        previous2_report_date=previous2_release_date,
        previous_release_dt=previous_release_dt,
    )

    print(f"Wrote: {latest_path}")


def cmd_report(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    requested = parse_iso_date(args.date)
    if requested is None:
        raise ValueError("Missing --date")
    raw_cfg = _load_raw_config(args.config)
    cot_filter_cfg = raw_cfg.get("cot_filter") if isinstance(raw_cfg, dict) else None
    if args.tier:
        if cot_filter_cfg is None:
            cot_filter_cfg = {}
        cot_filter_cfg = dict(cot_filter_cfg)
        cot_filter_cfg["strictness_tier"] = args.tier
    report = run_report(
        cfg=cfg,
        requested_date=requested,
        pairs_csv=args.pairs,
        report_type=args.report_type,
        usd_mode=args.usd_mode,
        usd_weights=args.usd_weights,
        refresh=args.refresh,
        verbose=args.verbose,
        signals_csv=args.signals,
        cot_filter_cfg=cot_filter_cfg,
        tier_override=args.tier,
    )
    out_dir = Path(args.out_dir)
    if "filter_tables" in report and isinstance(report["filter_tables"], dict):
        write_report_tables(report["filter_tables"], out_dir)
    output = format_report_output(report, args.out, out_dir)
    if args.out == "json":
        print(output)
    else:
        print(f"Wrote: {output}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cot_bias", description="COT Positioning Dashboard (FX + Metals)")
    p.add_argument("--config", default=None, help="Optional path to config.yaml (requires PyYAML).")

    sub = p.add_subparsers(dest="cmd", required=True)

    p_fetch = sub.add_parser("fetch", help="Download and update local CFTC store.")
    p_fetch.add_argument("--update", action="store_true", help="Update processed store (required).")
    p_fetch.add_argument("--combined", action="store_true", help="Deprecated alias for --update (ignored).")
    p_fetch.add_argument("--futures-only", action="store_true", help="Deprecated alias for --update (ignored).")
    p_fetch.add_argument("--force", action="store_true", help="Force refresh of current-year zip.")
    p_fetch.add_argument("--verbose", action="store_true", help="Verbose output (prints SHA256).")
    p_fetch.set_defaults(func=cmd_fetch)

    p_compute = sub.add_parser("compute", help="Compute CSV outputs without rendering HTML.")
    p_compute.add_argument("--out", default="outputs", help="Output directory for CSV output.")
    p_compute.add_argument(
        "--as-of",
        required=False,
        help="As-of date (YYYY-MM-DD). Defaults to most recent available release if omitted.",
    )
    p_compute.add_argument(
        "--as-at",
        dest="as_of",
        required=False,
        help="Deprecated alias for --as-of.",
    )
    p_compute.add_argument(
        "--resolve-mode",
        choices=["release_aligned", "report_date_direct"],
        default="release_aligned",
        help="Resolve report_date using release alignment (default) or direct report_date selection.",
    )
    p_compute.add_argument(
        "--report-date",
        default=None,
        help="Explicit report_date (YYYY-MM-DD) for report_date_direct mode.",
    )
    p_compute.add_argument(
        "--force-refresh", action="store_true", help="Force refresh of CFTC data before computing."
    )
    p_compute.add_argument("--verbose", action="store_true", help="Verbose output.")
    p_compute.set_defaults(func=cmd_compute)

    p_dash = sub.add_parser("dashboard", help="Generate HTML dashboard for as-of date.")
    p_dash.add_argument("--out", default="outputs", help="Output directory for CSV/HTML.")
    p_dash.add_argument(
        "--as-of",
        required=False,
        help="As-of date (YYYY-MM-DD). Defaults to most recent available release if omitted.",
    )
    p_dash.add_argument(
        "--as-at",
        dest="as_of",
        required=False,
        help="Deprecated alias for --as-of.",
    )
    p_dash.add_argument(
        "--release-time",
        default=None,
        help="Deprecated; ignored.",
    )
    p_dash.add_argument(
        "--resolve-mode",
        choices=["release_aligned", "report_date_direct"],
        default="release_aligned",
        help="Resolve report_date using release alignment (default) or direct report_date selection.",
    )
    p_dash.add_argument(
        "--report-date",
        default=None,
        help="Explicit report_date (YYYY-MM-DD) for report_date_direct mode.",
    )
    p_dash.add_argument(
        "--tier",
        default=None,
        choices=["loose", "balanced", "strict", "sniper"],
        help="Override cot_filter.strictness_tier for dashboard tables.",
    )
    p_dash.add_argument("--force-refresh", action="store_true", help="Force refresh of CFTC data before building dashboard.")
    p_dash.add_argument("--verbose", action="store_true", help="Verbose output.")
    p_dash.set_defaults(func=cmd_dashboard)

    p_rep = sub.add_parser("report", help="Generate an FX COT report for any input date.")
    p_rep.add_argument("--date", required=True, help="Requested calendar date (YYYY-MM-DD).")
    p_rep.add_argument("--pairs", required=True, help="Comma-separated FX pairs, e.g. EURUSD,AUDUSD,USDJPY.")
    p_rep.add_argument(
        "--signals",
        default=None,
        help="Optional Pine signals: PAIR=long,PAIR=short (overrides direction proxy).",
    )
    p_rep.add_argument(
        "--tier",
        default=None,
        choices=["loose", "balanced", "strict", "sniper"],
        help="Override cot_filter.strictness_tier for report tables.",
    )
    p_rep.add_argument("--report-type", choices=["legacy", "tff", "disagg"], default="tff", help="CFTC report type (futures-only).")
    p_rep.add_argument("--usd-mode", choices=["basket", "direct"], default="basket", help="USD mode: basket (default) or direct USD contract when available.")
    p_rep.add_argument("--usd-weights", choices=["equal", "dxy"], default="equal", help="USD basket weights: equal or DXY-like.")
    p_rep.add_argument("--out", choices=["json", "html"], default="json", help="Output format.")
    p_rep.add_argument("--out-dir", default="outputs", help="Output directory for HTML output.")
    p_rep.add_argument("--refresh", action="store_true", help="Force refresh of current-year CFTC zip.")
    p_rep.add_argument("--verbose", action="store_true", help="Verbose output (prints SHA256).")
    p_rep.set_defaults(func=cmd_report)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
