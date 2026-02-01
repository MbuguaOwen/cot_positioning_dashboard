from __future__ import annotations

import argparse
import datetime as dt
import subprocess
from pathlib import Path
from typing import Optional

import pandas as pd

from .utils import load_config, ensure_dirs, parse_iso_date
from .data.store import update_store, load_store, snapshot_df, latest_per_symbol, data_max_report_date, write_manifest
from .metrics.features import compute_metrics
from .dashboard.render import render_dashboard
from .dashboard.gate import compute_gated_fx_pairs
from .fx import usd_proxy_from_z, build_pairs_df
from .reporting import run_report, format_report_output


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


def _fx_release_dates(metrics: pd.DataFrame) -> list[dt.date]:
    if metrics.empty or "report_date" not in metrics.columns:
        return []
    df = metrics.copy()
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce").dt.date
    if "report_type" in df.columns:
        df = df[df["report_type"] == "tff"]
    dates = sorted(d for d in df["report_date"].dropna().unique())
    return dates


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
    latest = latest_per_symbol(metrics)
    if "report_date" in latest.columns:
        latest["report_date_used"] = latest["report_date"]

    pairs_df = None
    if "report_type" in latest.columns:
        fx_latest = latest[latest["report_type"] == "tff"].copy()
        if not fx_latest.empty and "z_3y" in fx_latest.columns:
            z_by = {
                row["symbol"]: float(row["z_3y"]) if pd.notna(row["z_3y"]) else float("nan")
                for _, row in fx_latest.iterrows()
            }
            usd_z = usd_proxy_from_z(z_by, weights=None)
            pairs_df = build_pairs_df(z_by, usd_z, usd_mode="basket")

    fx_dates = _fx_release_dates(metrics)
    latest_release_date = fx_dates[-1] if fx_dates else as_of
    previous_release_date = fx_dates[-2] if len(fx_dates) >= 2 else None
    fx_latest_release = (
        _fx_snapshot_for_release(metrics, latest_release_date) if latest_release_date else pd.DataFrame()
    )
    fx_prev_release = (
        _fx_snapshot_for_release(metrics, previous_release_date)
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
        "metrics": metrics,
        "latest": latest,
        "pairs_df": pairs_df,
        "gated_pairs_df": gated_pairs_df,
    }


def cmd_dashboard(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    ensure_dirs(cfg)

    as_of = parse_iso_date(args.as_of)

    ctx = _build_dashboard_context(
        cfg=cfg,
        as_of=as_of,
        force_refresh=args.force_refresh,
        verbose=args.verbose,
    )
    as_of = ctx["as_of"]
    max_date = ctx["max_date"]
    warnings = ctx["warnings"]
    snap = ctx["snap"]
    metrics = ctx["metrics"]
    latest = ctx["latest"]
    pairs_df = ctx["pairs_df"]
    gated_pairs_df = ctx["gated_pairs_df"]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    history_path = out_dir / "instruments_history.csv"
    latest_path = out_dir / "instruments_latest.csv"
    metrics.to_csv(history_path, index=False)
    latest.to_csv(latest_path, index=False)
    if pairs_df is not None and not pairs_df.empty:
        pairs_df.to_csv(out_dir / "pairs_latest.csv", index=False)

    html_path = render_dashboard(
        out_dir=out_dir,
        as_of=as_of.isoformat(),
        max_report_date=max_date.isoformat() if max_date else None,
        warnings=warnings,
        latest_df=latest,
        history_df=metrics,
        pairs_df=pairs_df,
        gate_df=gated_pairs_df,
    )

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
    )

    print(f"Wrote: {html_path}")


def cmd_compute(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    ensure_dirs(cfg)

    as_of = parse_iso_date(args.as_of)
    ctx = _build_dashboard_context(
        cfg=cfg,
        as_of=as_of,
        force_refresh=args.force_refresh,
        verbose=args.verbose,
    )
    as_of = ctx["as_of"]
    max_date = ctx["max_date"]
    warnings = ctx["warnings"]
    snap = ctx["snap"]
    metrics = ctx["metrics"]
    latest = ctx["latest"]
    pairs_df = ctx["pairs_df"]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    history_path = out_dir / "instruments_history.csv"
    latest_path = out_dir / "instruments_latest.csv"
    metrics.to_csv(history_path, index=False)
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
    )

    print(f"Wrote: {latest_path}")


def cmd_report(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    requested = parse_iso_date(args.date)
    if requested is None:
        raise ValueError("Missing --date")
    report = run_report(
        cfg=cfg,
        requested_date=requested,
        pairs_csv=args.pairs,
        report_type=args.report_type,
        usd_mode=args.usd_mode,
        usd_weights=args.usd_weights,
        refresh=args.refresh,
        verbose=args.verbose,
    )
    out_dir = Path(args.out_dir)
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
    p_dash.add_argument("--force-refresh", action="store_true", help="Force refresh of CFTC data before building dashboard.")
    p_dash.add_argument("--verbose", action="store_true", help="Verbose output.")
    p_dash.set_defaults(func=cmd_dashboard)

    p_rep = sub.add_parser("report", help="Generate an FX COT report for any input date.")
    p_rep.add_argument("--date", required=True, help="Requested calendar date (YYYY-MM-DD).")
    p_rep.add_argument("--pairs", required=True, help="Comma-separated FX pairs, e.g. EURUSD,AUDUSD,USDJPY.")
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
