from __future__ import annotations

import argparse
import calendar
import datetime as dt
import subprocess
import sys
from pathlib import Path
from typing import List

import pandas as pd


def subtract_months(date_value: dt.date, months: int) -> dt.date:
    month_index = date_value.month - 1 - months
    year = date_value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(date_value.day, calendar.monthrange(year, month)[1])
    return dt.date(year, month, day)


def load_report_dates(processed_parquet: Path) -> List[dt.date]:
    if not processed_parquet.exists():
        raise FileNotFoundError(
            f"Missing processed store at {processed_parquet}. Run fetch/compute first."
        )
    df = pd.read_parquet(processed_parquet, columns=["report_date"])
    if "report_date" not in df.columns:
        raise ValueError(f"{processed_parquet} does not contain report_date.")
    series = pd.to_datetime(df["report_date"], errors="coerce").dt.date.dropna()
    return sorted(set(series.tolist()))


def run_dashboard(as_of: dt.date, out_root: Path) -> None:
    out_dir = out_root / as_of.isoformat()
    cmd = [
        sys.executable,
        "-m",
        "cot_bias",
        "dashboard",
        "--out",
        str(out_dir),
        "--as-of",
        as_of.isoformat(),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate dashboard folders for all report dates in the last N months."
    )
    parser.add_argument("--months", type=int, default=4, help="Months of history to include.")
    parser.add_argument(
        "--processed-parquet",
        default="data/processed/cot.parquet",
        help="Path to processed parquet store.",
    )
    parser.add_argument("--out", default="outputs", help="Root output directory.")
    args = parser.parse_args()

    if args.months <= 0:
        raise ValueError("--months must be >= 1.")

    report_dates = load_report_dates(Path(args.processed_parquet))
    if not report_dates:
        raise ValueError("No report dates found in processed store.")

    latest = report_dates[-1]
    cutoff = subtract_months(latest, args.months)
    selected = [d for d in report_dates if d >= cutoff]
    if not selected:
        print("No report dates in selected window.")
        return

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    print(
        f"Generating {len(selected)} dashboards from {selected[0].isoformat()} "
        f"to {selected[-1].isoformat()} (latest={latest.isoformat()}, months={args.months}).",
        flush=True,
    )
    for report_date in selected:
        print(f"- {report_date.isoformat()}", flush=True)
        run_dashboard(report_date, out_root)


if __name__ == "__main__":
    main()
