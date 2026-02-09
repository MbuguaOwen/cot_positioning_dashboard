from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _rows_to_df(rows: Any) -> pd.DataFrame:
    if isinstance(rows, list):
        safe_rows = [r for r in rows if isinstance(r, dict)]
        return pd.DataFrame(safe_rows)
    return pd.DataFrame()


def _is_true(v: object) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in {"true", "1", "yes", "y"}


def _fmt_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = out[c].map(_fmt_cell)
    return out


def _fmt_cell(v: object) -> str:
    if isinstance(v, list):
        return ", ".join(str(x) for x in v)
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if isinstance(v, float) and abs(v - round(v)) > 1e-9:
            return f"{v:.3f}"
        return str(int(v)) if float(v).is_integer() else str(v)
    return str(v)


def _safe_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "<div class='empty'>No data.</div>"
    return _fmt_table(df).to_html(index=False, escape=True, classes="data-table")


def _kpi_card(label: str, value: object, subtitle: str = "") -> str:
    value_s = html.escape(_fmt_cell(value))
    subtitle_s = html.escape(subtitle) if subtitle else ""
    subtitle_html = f"<div class='kpi-sub'>{subtitle_s}</div>" if subtitle_s else ""
    return (
        "<div class='kpi-card'>"
        f"<div class='kpi-label'>{html.escape(label)}</div>"
        f"<div class='kpi-value'>{value_s}</div>"
        f"{subtitle_html}"
        "</div>"
    )


def _drivers_table(diag: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    for item in diag.get("Top3DriversByCurrency", []) if isinstance(diag, dict) else []:
        if not isinstance(item, dict):
            continue
        ccy = str(item.get("Currency", ""))
        drivers = item.get("Drivers", [])
        parts: List[str] = []
        if isinstance(drivers, list):
            for d in drivers:
                if not isinstance(d, dict):
                    continue
                comp = str(d.get("Component", ""))
                contrib = _fmt_cell(d.get("Contribution"))
                parts.append(f"{comp} ({contrib})")
        rows.append({"Currency": ccy, "TopDrivers": " | ".join(parts)})
    return pd.DataFrame(rows)


def _warnings_html(warnings: Any) -> str:
    if not isinstance(warnings, list) or not warnings:
        return "<div class='ok'>No warnings.</div>"
    items = "".join(f"<li>{html.escape(str(w))}</li>" for w in warnings)
    return f"<div class='warn'><ul>{items}</ul></div>"


def render_fx_bias_v2_dashboard(base_dir: Path, output_path: Path | None = None) -> Path:
    base_dir = Path(base_dir)
    output_path = output_path or (base_dir / "fx_bias_v2_dashboard.html")
    run_path = base_dir / "fx_bias_v2_run.json"
    if not run_path.exists():
        raise RuntimeError(f"Missing required file: {run_path}")

    bundle = _load_json(run_path)
    run_meta = bundle.get("RunMeta", {}) if isinstance(bundle, dict) else {}
    cap_df = _rows_to_df(bundle.get("CapabilityMatrix", []))
    ccy_df = _rows_to_df(bundle.get("A_Currency_Strength_Bias", []))
    pair_df = _rows_to_df(bundle.get("B_Pair_Bias_Trade_Gate", []))
    polarity_df = _rows_to_df(bundle.get("D_Currency_Polarity_Pairs", []))
    diag = bundle.get("C_Diagnostics", {}) if isinstance(bundle, dict) else {}

    if "ReasonCodes" in pair_df.columns:
        pair_df["ReasonCodes"] = pair_df["ReasonCodes"].map(_fmt_cell)

    if "PairBias" in pair_df.columns:
        pair_df["PairBias"] = pd.to_numeric(pair_df["PairBias"], errors="coerce")
        pair_df["_abs"] = pair_df["PairBias"].abs()
        pair_df = pair_df.sort_values(["_abs", "Pair"], ascending=[False, True]).drop(columns=["_abs"])
    if "ConvictionScore" in polarity_df.columns:
        polarity_df["ConvictionScore"] = pd.to_numeric(polarity_df["ConvictionScore"], errors="coerce")
        polarity_df["_abs"] = polarity_df["ConvictionScore"].abs()
        polarity_df = polarity_df.sort_values(["_abs", "Pair"], ascending=[False, True]).drop(columns=["_abs"])

    if "BiasScore_Final" in ccy_df.columns:
        ccy_df["BiasScore_Final"] = pd.to_numeric(ccy_df["BiasScore_Final"], errors="coerce")
        ccy_df["_abs"] = ccy_df["BiasScore_Final"].abs()
        ccy_df = ccy_df.sort_values(["_abs", "Currency"], ascending=[False, True]).drop(columns=["_abs"])

    pair_total = int(len(pair_df))
    allow_long = int((pair_df.get("GatingRuleForBOT", pd.Series(dtype=str)).astype(str) == "ALLOW_LONG").sum())
    allow_short = int((pair_df.get("GatingRuleForBOT", pd.Series(dtype=str)).astype(str) == "ALLOW_SHORT").sum())
    block = int((pair_df.get("GatingRuleForBOT", pd.Series(dtype=str)).astype(str) == "BLOCK").sum())
    reliable = int(cap_df.get("Reliable", pd.Series(dtype=bool)).map(_is_true).sum()) if not cap_df.empty else 0
    warnings = run_meta.get("warnings", []) if isinstance(run_meta, dict) else []

    top_conv_df = _rows_to_df(diag.get("HighestConvictionPairsTop10", []) if isinstance(diag, dict) else [])
    disagreement_df = _rows_to_df(diag.get("ComponentDisagreementPairs", []) if isinstance(diag, dict) else [])
    if "DisagreementCodes" in disagreement_df.columns:
        disagreement_df["DisagreementCodes"] = disagreement_df["DisagreementCodes"].map(_fmt_cell)
    shifts_df = _rows_to_df(diag.get("RegimeShiftsLast10TradingDays", []) if isinstance(diag, dict) else [])
    drivers_df = _drivers_table(diag if isinstance(diag, dict) else {})

    parts: List[str] = []
    parts.append("<!doctype html>")
    parts.append("<html><head><meta charset='utf-8'>")
    parts.append("<meta name='viewport' content='width=device-width, initial-scale=1'>")
    parts.append("<title>FX Bias Engine v2 Dashboard</title>")
    parts.append(
        "<style>"
        ":root{"
        "--bg:#f3f6f8;--card:#ffffff;--ink:#13232d;--muted:#4d6978;--line:#d7e0e6;"
        "--accent:#0d7b8f;--accent2:#d68536;--ok:#1f9a5b;--warn:#9f5b00;}"
        "*{box-sizing:border-box}"
        "body{margin:0;background:radial-gradient(circle at 10% 0%,#dcecf2 0%,#f3f6f8 45%,#eef2f4 100%);"
        "color:var(--ink);font-family:'IBM Plex Sans','Trebuchet MS','Segoe UI',sans-serif;}"
        ".wrap{max-width:1200px;margin:0 auto;padding:28px 18px 36px;}"
        "h1{margin:0 0 4px;font-size:30px;letter-spacing:.2px}"
        ".sub{color:var(--muted);margin:0 0 16px;font-size:14px}"
        ".meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px;margin:0 0 16px;}"
        ".meta div{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:10px 12px;font-size:12px;}"
        ".kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:0 0 18px;}"
        ".kpi-card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px;"
        "animation:rise .45s ease both;}"
        ".kpi-label{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.6px}"
        ".kpi-value{font-size:24px;font-weight:700;margin-top:6px}"
        ".kpi-sub{font-size:12px;color:var(--muted);margin-top:3px}"
        ".panel{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px 14px 6px;margin:12px 0;"
        "animation:rise .45s ease both;}"
        "h2{margin:0 0 10px;font-size:18px}"
        ".warn{background:#fff1dd;border:1px solid #ffd9a8;border-radius:10px;padding:10px 12px;color:var(--warn)}"
        ".ok{background:#e9f8ee;border:1px solid #c4ebd0;border-radius:10px;padding:10px 12px;color:var(--ok)}"
        ".warn ul{margin:0;padding-left:18px}"
        ".empty{padding:10px 0;color:var(--muted);font-size:13px}"
        ".data-table{width:100%;border-collapse:collapse;margin:0 0 8px;font-size:12px}"
        ".data-table th{background:linear-gradient(0deg,#0d7b8f,#1494aa);color:#fff;border:1px solid #0b6c7d;padding:7px;text-align:left}"
        ".data-table td{border:1px solid var(--line);padding:6px;vertical-align:top}"
        ".grid{display:grid;grid-template-columns:1fr;gap:12px}"
        "@media(min-width:980px){.grid{grid-template-columns:1fr 1fr}}"
        "@keyframes rise{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}"
        "</style>"
    )
    parts.append("</head><body><div class='wrap'>")

    parts.append("<h1>FX Bias Engine v2 Dashboard</h1>")
    parts.append("<p class='sub'>Price-led directional bias with rates, risk, carry, and skew overlays.</p>")

    parts.append("<section class='meta'>")
    parts.append(f"<div><strong>as_of_ts</strong><br>{html.escape(str(run_meta.get('as_of_ts', '')))}</div>")
    parts.append(f"<div><strong>engine_version</strong><br>{html.escape(str(run_meta.get('engine_version', '')))}</div>")
    parts.append(f"<div><strong>git_commit</strong><br>{html.escape(str(run_meta.get('git_commit', '')))}</div>")
    parts.append(f"<div><strong>config_hash</strong><br>{html.escape(str(run_meta.get('config_hash', '')))}</div>")
    parts.append("</section>")

    parts.append("<section class='kpis'>")
    parts.append(_kpi_card("Pairs Total", pair_total))
    parts.append(_kpi_card("Allow Long", allow_long))
    parts.append(_kpi_card("Allow Short", allow_short))
    parts.append(_kpi_card("Blocked", block))
    parts.append(_kpi_card("Reliable Components", reliable, f"of {len(cap_df)}"))
    parts.append(_kpi_card("Warnings", len(warnings)))
    parts.append("</section>")

    parts.append("<section class='panel'><h2>Run Warnings</h2>")
    parts.append(_warnings_html(warnings))
    parts.append("</section>")

    parts.append("<section class='panel'><h2>Capability Matrix</h2>")
    parts.append(_safe_table(cap_df))
    parts.append("</section>")

    parts.append("<section class='panel'><h2>Currency Strength Bias</h2>")
    parts.append(_safe_table(ccy_df))
    parts.append("</section>")

    parts.append("<section class='panel'><h2>Pair Bias and Trade Gate</h2>")
    parts.append(_safe_table(pair_df))
    parts.append("</section>")

    parts.append("<section class='panel'><h2>Currency Polarity Pairs</h2>")
    parts.append(
        "<div class='empty'>Orientation: pair direction is BASE minus QUOTE (Spread = BaseBias - QuoteBias); "
        "a bullish QUOTE makes the pair bearish.</div>"
    )
    parts.append(_safe_table(polarity_df))
    parts.append("</section>")

    parts.append("<section class='grid'>")
    parts.append("<div class='panel'><h2>Top Conviction Pairs</h2>")
    parts.append(_safe_table(top_conv_df))
    parts.append("</div>")
    parts.append("<div class='panel'><h2>Component Disagreements</h2>")
    parts.append(_safe_table(disagreement_df))
    parts.append("</div>")
    parts.append("</section>")

    parts.append("<section class='grid'>")
    parts.append("<div class='panel'><h2>Top Drivers by Currency</h2>")
    parts.append(_safe_table(drivers_df))
    parts.append("</div>")
    parts.append("<div class='panel'><h2>Regime Shifts (Last 10 Trading Days)</h2>")
    parts.append(_safe_table(shifts_df))
    parts.append("</div>")
    parts.append("</section>")

    parts.append("</div></body></html>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")
    return output_path
