from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .utils import load_config, ensure_dirs
from .sources import fetch_datasets, fetch_historical_datasets, read_cot_csv
from .storage import store_raw, load_raw
from .compute import compute_all, latest_rows
from .dashboard import write_dashboard_html

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
    inst_df, pairs_df = compute_all(cfg, fin_raw, dis_raw)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    inst_latest = latest_rows(inst_df)
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

    html_path = write_dashboard_html(inst_latest, pairs_df, out_dir)
    print(f"Wrote: {html_path}")

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
    p_compute.set_defaults(func=cmd_compute)

    p_dash = sub.add_parser("dashboard", help="Generate HTML dashboard.")
    p_dash.add_argument("--out", default="outputs", help="Output directory for CSV/HTML.")
    p_dash.set_defaults(func=cmd_dashboard)

    return p

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
