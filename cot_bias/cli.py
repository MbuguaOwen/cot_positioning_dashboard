from __future__ import annotations

import argparse
import datetime as dt
from typing import Optional
from pathlib import Path

import pandas as pd

from .utils import load_config, ensure_dirs, parse_iso_date, most_recent_tuesday
from .sources import fetch_datasets, fetch_historical_datasets, read_cot_csv
from .storage import store_raw, load_raw
from .compute import compute_all, latest_rows, compute_pairs
from .dashboard import write_dashboard_html
from .reporting import run_report, format_report_output

def _resolve_dates(args: argparse.Namespace) -> tuple[Optional[dt.date], Optional[dt.date]]:
    as_at = parse_iso_date(getattr(args, "as_at", None))
    as_of = parse_iso_date(getattr(args, "as_of", None))
    if as_at and not as_of:
        as_of = most_recent_tuesday(as_at)
    return as_at, as_of

def cmd_fetch(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    if getattr(args, "weekly_only", False):
        fin_path, dis_path = fetch_datasets(cfg, combined=args.combined)
        fin_df = read_cot_csv(fin_path)
        dis_df = read_cot_csv(dis_path)
        store_raw(cfg, fin_df, dis_df, combined=args.combined)
        print(f"Downloaded + stored (current week only): {fin_path} and {dis_path}")
    else:
        fin_df, dis_df, downloaded = fetch_historical_datasets(
            cfg, combined=args.combined, years_back=args.years_back
        )
        store_raw(cfg, fin_df, dis_df, combined=args.combined)
        print(f"Downloaded + stored (historical compressed): {len(downloaded)} zip file(s)")
    print(f"SQLite: {cfg.sqlite_path}")
    print(f"Parquet: {cfg.parquet_dir}")

def cmd_compute(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    fin_raw, dis_raw = load_raw(cfg)
    _as_at, as_of = _resolve_dates(args)
    inst_df, pairs_df = compute_all(cfg, fin_raw, dis_raw, as_of=as_of)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    inst_latest = latest_rows(inst_df, as_of=as_of)
    inst_latest.to_csv(out_dir / "instruments_latest.csv", index=False)
    pairs_df.to_csv(out_dir / "pairs_latest.csv", index=False)

    # also store full history (optional)
    inst_df.to_csv(out_dir / "instruments_history.csv", index=False)
    pairs_df.to_csv(out_dir / "pairs_latest_only.csv", index=False)

    print(f"Wrote: {out_dir / 'instruments_latest.csv'}")
    print(f"Wrote: {out_dir / 'pairs_latest.csv'}")
    print(f"Wrote: {out_dir / 'instruments_history.csv'}")

def cmd_dashboard(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    ensure_dirs(cfg)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    inst_latest_path = out_dir / "instruments_latest.csv"
    pairs_latest_path = out_dir / "pairs_latest.csv"
    inst_history_path = out_dir / "instruments_history.csv"

    as_at, as_of = _resolve_dates(args)

    if as_of is not None:
        if inst_history_path.exists():
            inst_df = pd.read_csv(inst_history_path)
        else:
            fin_raw, dis_raw = load_raw(cfg)
            inst_df, _ = compute_all(cfg, fin_raw, dis_raw)
            inst_df.to_csv(inst_history_path, index=False)
        inst_latest = latest_rows(inst_df, as_of=as_of)
        pairs_df = compute_pairs(inst_latest)
    else:
        if not inst_latest_path.exists() or not pairs_latest_path.exists():
            # compute first
            fin_raw, dis_raw = load_raw(cfg)
            inst_df, pairs_df = compute_all(cfg, fin_raw, dis_raw)
            inst_latest = latest_rows(inst_df)
            inst_latest.to_csv(inst_latest_path, index=False)
            pairs_df.to_csv(pairs_latest_path, index=False)
        else:
            inst_latest = pd.read_csv(inst_latest_path)
            pairs_df = pd.read_csv(pairs_latest_path)

    html_path = write_dashboard_html(inst_latest, pairs_df, out_dir, as_at=as_at, as_of=as_of, release_time=args.release_time)
    print(f"Wrote: {html_path}")


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

    p_fetch = sub.add_parser("fetch", help="Download latest COT datasets and store locally.")
    p_fetch.add_argument("--combined", action="store_true", help="Use futures+options combined datasets (recommended). Default is futures-only if not set.")
    p_fetch.add_argument("--years-back", type=int, default=None, help="How many years of historical compressed data to pull (default: derived from rolling window).")
    p_fetch.add_argument("--weekly-only", action="store_true", help="Fetch only the current-week files (fast), skipping historical compressed.")
    p_fetch.set_defaults(func=cmd_fetch)

    p_compute = sub.add_parser("compute", help="Compute metrics + write CSV outputs.")
    p_compute.add_argument("--out", default="outputs", help="Output directory for CSV/HTML.")
    g_compute_dates = p_compute.add_mutually_exclusive_group()
    g_compute_dates.add_argument("--as-at", help="Release date (YYYY-MM-DD). Will resolve to most recent Tuesday report date.")
    g_compute_dates.add_argument("--as-of", help="Report date (YYYY-MM-DD). Use to backfill a specific report date.")
    p_compute.set_defaults(func=cmd_compute)

    p_dash = sub.add_parser("dashboard", help="Generate HTML dashboard.")
    p_dash.add_argument("--out", default="outputs", help="Output directory for CSV/HTML.")
    p_dash.add_argument("--release-time", default=None, help="Optional release time label, e.g. '3:30 p.m. ET'.")
    g_dash_dates = p_dash.add_mutually_exclusive_group()
    g_dash_dates.add_argument("--as-at", help="Release date (YYYY-MM-DD). Will resolve to most recent Tuesday report date.")
    g_dash_dates.add_argument("--as-of", help="Report date (YYYY-MM-DD). Use to backfill a specific report date.")
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
