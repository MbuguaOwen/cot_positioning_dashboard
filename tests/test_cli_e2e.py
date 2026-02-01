import datetime as dt
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


class TestCliE2E(unittest.TestCase):
    def test_dashboard_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            processed = tmp_path / "processed"
            processed.mkdir(parents=True, exist_ok=True)
            store_path = processed / "cot.parquet"

            df = pd.DataFrame(
                {
                    "symbol": ["EUR", "EUR"],
                    "report_date": [dt.date(2026, 1, 6), dt.date(2026, 1, 13)],
                    "net_pct_oi": [1.0, 2.0],
                }
            )
            df.to_parquet(store_path, index=False)

            cfg = tmp_path / "config.yaml"
            cfg.write_text(
                f"storage:\n  data_dir: {tmp_path.as_posix()}\n  processed_dir: {processed.as_posix()}\n"
                "metrics:\n  rolling_weeks: 2\n  delta_weeks: 4\n",
                encoding="utf-8",
            )

            out_dir = tmp_path / "out"
            env = os.environ.copy()
            env["PYTHONPATH"] = str(Path.cwd())

            cmd = [
                sys.executable,
                "-m",
                "cot_bias",
                "--config",
                str(cfg),
                "dashboard",
                "--as-of",
                "2026-01-20",
                "--out",
                str(out_dir),
            ]
            subprocess.check_call(cmd, env=env, cwd=Path.cwd())

            html_path = out_dir / "dashboard.html"
            self.assertTrue(html_path.exists())
            html = html_path.read_text(encoding="utf-8")
            self.assertNotIn("DATA STALE", html)

            hist_path = out_dir / "instruments_history.csv"
            hist = pd.read_csv(hist_path)
            self.assertTrue((pd.to_datetime(hist["report_date"]).dt.date <= dt.date(2026, 1, 20)).all())


if __name__ == "__main__":
    unittest.main()
