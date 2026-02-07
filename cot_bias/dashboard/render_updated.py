from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def _bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def _fmt(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_bool_dtype(out[c]):
            out[c] = out[c].map(lambda x: "" if pd.isna(x) else ("true" if bool(x) else "false"))
            continue
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    return out


def _missing_files(required: Dict[str, Path]) -> List[str]:
    missing = []
    for name, path in required.items():
        if not path.exists():
            missing.append(name)
    return missing


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _read_csv_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _quality_summary_line(quality_view: pd.DataFrame) -> str:
    if quality_view.empty:
        return "ELITE count: 0 | GOOD count: 0 | FAIL count: 0 | Top failure gates: none"

    tiers = quality_view.get("quality_tier", pd.Series([""] * len(quality_view))).astype(str).str.upper()
    elite_count = int((tiers == "ELITE").sum())
    good_count = int((tiers == "GOOD").sum())
    fail_count = int((tiers == "FAIL").sum())

    fail_rows = quality_view[tiers == "FAIL"].copy()
    if fail_rows.empty:
        return (
            f"ELITE count: {elite_count} | GOOD count: {good_count} | FAIL count: {fail_count} "
            "| Top failure gates: none"
        )

    counts = {
        "A_release": int((~_bool_series(fail_rows.get("A_release_ok", pd.Series([True] * len(fail_rows))))).sum()),
        "A_consecutive": int((~_bool_series(fail_rows.get("A_consecutive_ok", pd.Series([True] * len(fail_rows))))).sum()),
        "A_sign": int((~_bool_series(fail_rows.get("A_sign_ok", pd.Series([True] * len(fail_rows))))).sum()),
        "B_strength": int((~_bool_series(fail_rows.get("B_strength_ok", pd.Series([True] * len(fail_rows))))).sum()),
        "C_crowding": int((~_bool_series(fail_rows.get("C_crowding_ok", pd.Series([True] * len(fail_rows))))).sum()),
        "C_reversal": int((pd.to_numeric(fail_rows.get("C_reversal_points", pd.Series([0] * len(fail_rows))), errors="coerce") < 0).sum()),
        "D_collapse": int(
            (
                (~_bool_series(fail_rows.get("D_no_collapse_ok", pd.Series([True] * len(fail_rows)))))
                & (pd.to_numeric(fail_rows.get("D_points", pd.Series([0] * len(fail_rows))), errors="coerce") >= 0)
            ).sum()
        ),
    }
    max_count = max(counts.values()) if counts else 0
    if max_count <= 0:
        top = "none"
    else:
        top = ", ".join(sorted(k for k, v in counts.items() if v == max_count))
    return (
        f"ELITE count: {elite_count} | GOOD count: {good_count} | FAIL count: {fail_count} "
        f"| Top failure gates: {top}"
    )


def _pick_first(series: pd.Series) -> str:
    if series.empty:
        return ""
    vals = [v for v in series.dropna().astype(str).tolist() if v.strip()]
    return vals[0] if vals else ""


def _pick_latest_date(series: pd.Series) -> str:
    if series.empty:
        return ""
    vals = [v for v in series.dropna().astype(str).tolist() if v.strip()]
    if not vals:
        return ""
    return sorted(set(vals))[-1]


def _load_strength_cfg(config_path: Path) -> Dict[str, object]:
    cfg = {
        "enabled": True,
        "strong_threshold": 2.0,
        "sniper_threshold": 2.5,
        "mode": "strong",
    }
    try:
        import yaml  # type: ignore
    except Exception:
        return cfg
    if not config_path.exists():
        return cfg
    try:
        with config_path.open("r", encoding="utf-8") as f:
            doc = yaml.safe_load(f)
        if not isinstance(doc, dict):
            return cfg
        block = doc.get("cot_strength_filter")
        if not isinstance(block, dict):
            return cfg
        cfg["enabled"] = bool(block.get("enabled", cfg["enabled"]))
        cfg["strong_threshold"] = float(block.get("strong_threshold", cfg["strong_threshold"]))
        cfg["sniper_threshold"] = float(block.get("sniper_threshold", cfg["sniper_threshold"]))
        mode = str(block.get("mode", cfg["mode"])).strip().lower()
        if mode in {"strong", "sniper"}:
            cfg["mode"] = mode
        return cfg
    except Exception:
        return cfg


def render_dashboard_updated(base_dir: Path, output_path: Optional[Path] = None) -> Path:
    base_dir = Path(base_dir)
    output_path = output_path or (base_dir / "dashboard_updated.html")

    required = {
        "instruments_latest.csv": base_dir / "instruments_latest.csv",
        "pairs_latest.csv": base_dir / "pairs_latest.csv",
        "decisions_summary.csv": base_dir / "decisions_summary.csv",
        "spread_flow.csv": base_dir / "spread_flow.csv",
        "crowdedness_policy.csv": base_dir / "crowdedness_policy.csv",
        "macro_gate.csv": base_dir / "macro_gate.csv",
        "pine_join.csv": base_dir / "pine_join.csv",
        "run_manifest.json": base_dir / "run_manifest.json",
    }
    missing = _missing_files(required)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise RuntimeError(
            f"Missing required files in {base_dir}: {missing_list}. "
            "Run fetch/compute/dashboard to generate outputs."
        )

    manifest = _load_json(required["run_manifest.json"])
    instruments = pd.read_csv(required["instruments_latest.csv"])
    pairs = pd.read_csv(required["pairs_latest.csv"])
    decisions = pd.read_csv(required["decisions_summary.csv"])
    persistent_path = base_dir / "persistent_bullish_2w.csv"
    persistent = _read_csv_optional(persistent_path)
    strong_persistent_path = base_dir / "strong_persistent_bullish_3w.csv"
    strong_persistent = _read_csv_optional(strong_persistent_path)
    quality_path = base_dir / "quality_gates_score.csv"
    quality = _read_csv_optional(quality_path)

    allow = _bool_series(decisions["allow"]) if "allow" in decisions.columns else pd.Series([False] * len(decisions))
    direction = decisions.get("direction_allowed", pd.Series([""] * len(decisions))).astype(str).str.upper()

    total_pairs = int(len(decisions))
    allowed = int(allow.sum())
    blocked = int(total_pairs - allowed)
    allowed_longs = int(((allow) & (direction == "LONG")).sum())
    allowed_shorts = int(((allow) & (direction == "SHORT")).sum())

    week_end_used = _pick_latest_date(decisions.get("week_end", pd.Series(dtype=str)))
    tier_mode = _pick_first(decisions.get("tier_mode", pd.Series(dtype=str))).upper()
    resolve_mode = str(manifest.get("resolve_mode", "release_aligned") or "release_aligned").strip().lower()
    requested_report_date = str(manifest.get("requested_report_date", "") or "")
    next_report_date = str(manifest.get("next_report_date", "") or "")
    next_release_dt = str(manifest.get("next_release_dt", "") or "")
    previous_report_date = str(manifest.get("previous_report_date", "") or "")
    previous2_report_date = str(manifest.get("previous2_report_date", "") or "")

    # B) Instrument Snapshot
    inst_cols = [
        "symbol",
        "report_date",
        "z_3y",
        "pctile_3y",
        "bias",
        "reversal_risk",
        "score",
        "net_pct_oi",
    ]
    inst_view = instruments.copy()
    if "score" in inst_view.columns:
        inst_view["score"] = pd.to_numeric(inst_view["score"], errors="coerce")
        inst_view = inst_view.sort_values("score", ascending=False)
    if "z_3y" in inst_view.columns:
        inst_view["z_3y"] = pd.to_numeric(inst_view["z_3y"], errors="coerce")
        inst_view["_abs_z"] = inst_view["z_3y"].abs()
        inst_view = inst_view.sort_values("_abs_z", ascending=False).drop(columns=["_abs_z"])
    elif "pctile_3y" in inst_view.columns:
        inst_view["pctile_3y"] = pd.to_numeric(inst_view["pctile_3y"], errors="coerce")
        inst_view["_pctile_dist"] = (inst_view["pctile_3y"] - 50.0).abs()
        inst_view = inst_view.sort_values("_pctile_dist", ascending=False).drop(columns=["_pctile_dist"])
    inst_cols_present = [c for c in inst_cols if c in inst_view.columns]
    inst_view = inst_view[inst_cols_present].head(30)

    # C) Actionable List — Tier Passed
    passed = decisions[allow].copy() if "allow" in decisions.columns else decisions.iloc[0:0].copy()
    passed["direction_allowed"] = passed.get("direction_allowed", "").astype(str).str.upper()
    passed["confidence_score"] = pd.to_numeric(passed.get("confidence_score"), errors="coerce")
    passed["spread"] = pd.to_numeric(passed.get("spread"), errors="coerce")
    passed["dSpread_1w"] = pd.to_numeric(passed.get("dSpread_1w"), errors="coerce")
    passed["bias_hint"] = passed["direction_allowed"].map(
        {
            "LONG": "BASE↑ / QUOTE↓ (bullish base)",
            "SHORT": "BASE↓ / QUOTE↑ (bearish base)",
        }
    ).fillna("")

    passed["dir_rank"] = passed["direction_allowed"].map(
        {"LONG": 0, "SHORT": 1, "BULLISH": 0, "BEARISH": 1}
    ).fillna(2)
    passed = passed.sort_values(["dir_rank", "confidence_score"], ascending=[True, False])

    tier_cols = [
        "pair",
        "direction_allowed",
        "confidence_score",
        "spread",
        "dSpread_1w",
        "crowded_flag",
        "macro_ok",
        "news_ok",
        "bias_hint",
    ]
    for col in tier_cols:
        if col not in passed.columns:
            passed[col] = ""
    tier_view = passed[tier_cols].copy()
    for col in ("macro_ok", "news_ok"):
        if col in tier_view.columns:
            tier_view[col] = _bool_series(tier_view[col])

    # D) Bullish Bias Focus (LONGs)
    bullish = passed[passed["direction_allowed"] == "LONG"].copy()
    bullish_cols = ["pair", "confidence_score", "spread", "dSpread_1w", "crowded_flag"]
    for col in bullish_cols:
        if col not in bullish.columns:
            bullish[col] = ""
    bullish_view = bullish[bullish_cols].copy()

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
    persistent_view = persistent.copy()
    for col in persistent_cols:
        if col not in persistent_view.columns:
            persistent_view[col] = ""
    persistent_view = persistent_view[persistent_cols].copy() if not persistent_view.empty else persistent_view[persistent_cols]

    strong_persistent_cols = [
        "pair",
        "confidence_score",
        "spread",
        "dSpread_1w",
        "crowded_flag",
        "prev1_spread",
        "prev2_spread",
    ]
    strong_persistent_view = strong_persistent.copy()
    for col in strong_persistent_cols:
        if col not in strong_persistent_view.columns:
            strong_persistent_view[col] = ""
    strong_persistent_view = (
        strong_persistent_view[strong_persistent_cols].copy()
        if not strong_persistent_view.empty
        else strong_persistent_view[strong_persistent_cols]
    )

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
    quality_view = quality.copy()
    for col in quality_cols:
        if col not in quality_view.columns:
            quality_view[col] = ""
    quality_view = quality_view[quality_cols].copy() if not quality_view.empty else quality_view[quality_cols]
    if not quality_view.empty:
        quality_view["quality_score"] = pd.to_numeric(quality_view["quality_score"], errors="coerce")
        quality_view = quality_view.sort_values(["quality_score", "pair"], ascending=[False, True]).reset_index(drop=True)
    quality_summary_line = _quality_summary_line(quality_view)

    # E) Context — Top Pair Z
    pairs_view = pairs.copy()
    if "pair_bias" not in pairs_view.columns and "bias" in pairs_view.columns:
        pairs_view["pair_bias"] = pairs_view["bias"]
    if "method" not in pairs_view.columns and "usd_mode" in pairs_view.columns:
        pairs_view["method"] = pairs_view["usd_mode"]
    if "pair_z" in pairs_view.columns:
        pairs_view["pair_z"] = pd.to_numeric(pairs_view["pair_z"], errors="coerce")
        pairs_view["abs_pair_z"] = pairs_view["pair_z"].abs()
        pairs_view = pairs_view.sort_values("abs_pair_z", ascending=False)
        pairs_view = pairs_view.drop(columns=["abs_pair_z"])
    pairs_view = pairs_view.head(12)
    pair_cols = [
        "pair",
        "pair_z",
        "pair_bias",
        "pair_strength",
        "base",
        "quote",
        "base_z",
        "quote_z",
        "method",
    ]
    pair_cols_present = [c for c in pair_cols if c in pairs_view.columns]
    pairs_view = pairs_view[pair_cols_present].copy()

    strength_cfg = _load_strength_cfg(Path("config.yaml"))
    strength_enabled = bool(strength_cfg.get("enabled", True))
    strength_mode = str(strength_cfg.get("mode", "strong")).strip().lower()
    strong_thr = float(strength_cfg.get("strong_threshold", 2.0))
    sniper_thr = float(strength_cfg.get("sniper_threshold", 2.5))
    threshold = sniper_thr if strength_mode == "sniper" else strong_thr

    strength_cols = [
        "pair",
        "base",
        "quote",
        "base_z",
        "quote_z",
        "gap_z",
    ]
    if "pair_z" in pairs.columns:
        pairs_for_strength = pairs.copy()
        if "pair_bias" not in pairs_for_strength.columns and "bias" in pairs_for_strength.columns:
            pairs_for_strength["pair_bias"] = pairs_for_strength["bias"]
        if "method" not in pairs_for_strength.columns and "usd_mode" in pairs_for_strength.columns:
            pairs_for_strength["method"] = pairs_for_strength["usd_mode"]
        if "base_z" not in pairs_for_strength.columns:
            pairs_for_strength["base_z"] = pd.Series([float("nan")] * len(pairs_for_strength))
        if "quote_z" not in pairs_for_strength.columns:
            pairs_for_strength["quote_z"] = pd.Series([float("nan")] * len(pairs_for_strength))
        pairs_for_strength["base_z"] = pd.to_numeric(pairs_for_strength["base_z"], errors="coerce")
        pairs_for_strength["quote_z"] = pd.to_numeric(pairs_for_strength["quote_z"], errors="coerce")
        pairs_for_strength["gap_z"] = pairs_for_strength["base_z"] - pairs_for_strength["quote_z"]
        bullish_strong = pairs_for_strength[
            (pairs_for_strength["base_z"] > 0)
            & (pairs_for_strength["quote_z"] < 0)
            & (pairs_for_strength["gap_z"] >= threshold)
        ].copy()
        bearish_strong = pairs_for_strength[
            (pairs_for_strength["base_z"] < 0)
            & (pairs_for_strength["quote_z"] > 0)
            & (pairs_for_strength["gap_z"] <= -threshold)
        ].copy()
        bullish_strong = bullish_strong.sort_values("gap_z", ascending=False)
        bearish_strong = bearish_strong.sort_values("gap_z", ascending=True)
        bullish_strong = bullish_strong.head(20)
        bearish_strong = bearish_strong.head(20)
    else:
        bullish_strong = pd.DataFrame()
        bearish_strong = pd.DataFrame()

    parts: List[str] = []
    parts.append("<html><head><meta charset='utf-8'><title>COT Positioning Dashboard</title>")
    parts.append(
        "<style>"
        "body{font-family:Arial,Helvetica,sans-serif;background:#fff;color:#111;margin:0;}"
        ".container{max-width:1100px;margin:0 auto;padding:20px 18px;}"
        ".badges{display:flex;flex-wrap:wrap;gap:8px;margin:6px 0 14px 0;}"
        ".badge{background:#f1f3f5;border:1px solid #e1e4e8;padding:4px 8px;border-radius:999px;font-size:12px;}"
        ".kpis{display:flex;flex-wrap:wrap;gap:12px;margin:10px 0 20px 0;}"
        ".kpi{border:1px solid #ddd;padding:10px 12px;border-radius:8px;min-width:150px;background:#fafafa;}"
        ".kpi .label{font-size:11px;color:#666;text-transform:uppercase;letter-spacing:0.3px;}"
        ".kpi .value{font-size:16px;font-weight:bold;margin-top:4px;}"
        "table{border-collapse:collapse;margin:10px 0 18px 0;width:100%;}"
        "th,td{border:1px solid #ddd;padding:6px 8px;font-size:12px;}"
        "th{background:#f5f5f5;}"
        ".helper{font-size:12px;color:#666;margin:4px 0 8px 0;}"
        ".note{font-size:12px;color:#555;margin:6px 0;}"
        ".warning{background:#fff3cd;border:1px solid #ffeeba;padding:8px 10px;border-radius:6px;font-size:12px;margin:8px 0 12px 0;}"
        ".meta{font-size:12px;color:#444;margin:6px 0 12px 0;}"
        ".meta div{margin:2px 0;}"
        "</style>"
    )
    parts.append("</head><body><div class='container'>")

    # A) Header + KPI cards
    parts.append("<h1>COT Positioning Dashboard</h1>")
    parts.append("<div class='badges'>")
    parts.append(f"<span class='badge'>As-of: {manifest.get('as_of','')}</span>")
    parts.append(
        f"<span class='badge'>Resolved report_date: {manifest.get('resolved_report_date','')}</span>"
    )
    parts.append(
        f"<span class='badge'>Release_dt (UTC): {manifest.get('resolved_release_dt','')}</span>"
    )
    parts.append(f"<span class='badge'>Max report_date: {manifest.get('store_max_report_date','')}</span>")
    parts.append(f"<span class='badge'>Commit: {manifest.get('git_commit','')}</span>")
    parts.append(f"<span class='badge'>Tier mode: {tier_mode}</span>")
    parts.append(f"<span class='badge'>Strength mode: {strength_mode}</span>")
    parts.append("</div>")
    if resolve_mode == "report_date_direct":
        parts.append("<div class='warning'>Warning: report_date_direct mode can look ahead of release timing.</div>")
    parts.append("<div class='meta'>")
    parts.append(f"<div>As-of: {manifest.get('as_of','')}</div>")
    if requested_report_date:
        parts.append(f"<div>Requested report_date: {requested_report_date}</div>")
    mode_label = "release aligned" if resolve_mode != "report_date_direct" else "report_date_direct"
    parts.append(
        f"<div>Resolved report_date ({mode_label}): {manifest.get('resolved_report_date','')}</div>"
    )
    parts.append(f"<div>Release_dt (UTC): {manifest.get('resolved_release_dt','')}</div>")
    if next_report_date:
        parts.append(
            f"<div>Next report_date {next_report_date} releases at {next_release_dt}; "
            f"not eligible as-of {manifest.get('as_of','')}.</div>"
        )
    parts.append("</div>")
    parts.append("<div class='kpis'>")
    parts.append(f"<div class='kpi'><div class='label'>total_pairs</div><div class='value'>{len(pairs)}</div></div>")
    parts.append(
        f"<div class='kpi'><div class='label'>bullish_strong_count</div><div class='value'>{len(bullish_strong)}</div></div>"
    )
    parts.append(
        f"<div class='kpi'><div class='label'>bearish_strong_count</div><div class='value'>{len(bearish_strong)}</div></div>"
    )
    parts.append("</div>")

    # B) Instrument Snapshot
    parts.append("<h2>1) Instrument Snapshot (what’s crowded / what’s weak)</h2>")
    parts.append("<div class='helper'>Top 30 by |z_3y| (or pctile distance from 50) for quick sanity checks.</div>")
    parts.append(_fmt(inst_view).to_html(index=False, escape=True))

    # Strength-First Shortlist
    if strength_enabled:
        parts.append("<h2>Strength-First Shortlist (pair_z)</h2>")
        parts.append(
            f"<div class='helper'>Mode: {strength_mode} | strong_threshold: {threshold:.2f} | gap_z = base_z - quote_z</div>"
        )
        parts.append(f"<h3>A) Bullish Strong (gap_z >= {threshold:.2f})</h3>")
        if bullish_strong.empty:
            parts.append("<div class='note'>No bullish pairs met the strong threshold.</div>")
        else:
            cols = [c for c in strength_cols if c in bullish_strong.columns]
            if cols:
                parts.append(_fmt(bullish_strong[cols]).to_html(index=False, escape=True))
            else:
                parts.append(_fmt(bullish_strong).to_html(index=False, escape=True))

        parts.append(f"<h3>B) Bearish Strong (gap_z <= -{threshold:.2f})</h3>")
        if bearish_strong.empty:
            parts.append("<div class='note'>No bearish pairs met the strong threshold.</div>")
        else:
            cols = [c for c in strength_cols if c in bearish_strong.columns]
            if cols:
                parts.append(_fmt(bearish_strong[cols]).to_html(index=False, escape=True))
            else:
                parts.append(_fmt(bearish_strong).to_html(index=False, escape=True))

    # C) Actionable List — Tier Passed
    parts.append("<h2>2) Actionable List — Tier Passed (this is the only shortlist)</h2>")
    if tier_view.empty:
        parts.append("<div class='note'>No pairs passed the current tier filters.</div>")
    else:
        parts.append(_fmt(tier_view).to_html(index=False, escape=True))

    # D) Bullish Bias Focus (LONGs)
    parts.append("<h2>2a) Bullish Bias Focus (LONGs)</h2>")
    if bullish_view.empty:
        parts.append("<div class='note'>No LONGs passed this week under the active tier.</div>")
    else:
        parts.append(_fmt(bullish_view).to_html(index=False, escape=True))

    parts.append("<h2>2b) Persistent Bullish Bias (Current + Previous report)</h2>")
    if not previous_report_date:
        parts.append("<div class='note'>No previous report available.</div>")
    else:
        parts.append(
            f"<div class='helper'>Persistent bullish count: {len(persistent_view)} / current bullish count: {len(bullish_view)}</div>"
        )
        if persistent_view.empty:
            parts.append("<div class='note'>No pairs persisted bullish across two reports.</div>")
        else:
            parts.append(_fmt(persistent_view).to_html(index=False, escape=True))

    parts.append("<h2>2c) Strong Persistence (3 consecutive reports — LONG)</h2>")
    if not previous2_report_date:
        parts.append("<div class='note'>Strong persistence unavailable (need 3 reports).</div>")
    elif strong_persistent_view.empty:
        parts.append("<div class='note'>No pairs met 3-week strong persistence.</div>")
    else:
        parts.append(_fmt(strong_persistent_view).to_html(index=False, escape=True))

    parts.append("<h2>2d) Quality Gates Score (Scored Filter — derived from 2c Strong Persistence)</h2>")
    if not previous2_report_date:
        parts.append("<div class='note'>Quality score unavailable (need 3 reports).</div>")
    elif quality_view.empty:
        parts.append("<div class='note'>No strong-persistence pairs available for quality scoring.</div>")
        parts.append(f"<div class='helper'>{quality_summary_line}</div>")
    else:
        parts.append(_fmt(quality_view).to_html(index=False, escape=True))
        parts.append(f"<div class='helper'>{quality_summary_line}</div>")

    # E) Context — Top Pair Z
    parts.append("<h2>3) Context — Top Pair Z (raw ranking, not a trade list)</h2>")
    parts.append("<div class='note'>Scan-only list (not a trade list).</div>")
    parts.append(_fmt(pairs_view).to_html(index=False, escape=True))

    parts.append("</div></body></html>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")
    return output_path

