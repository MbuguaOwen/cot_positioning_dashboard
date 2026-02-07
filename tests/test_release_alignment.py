import datetime as dt
import unittest

import pandas as pd

from cot_bias.cli import _filter_metrics_for_report_date, _resolve_report_date


class TestReleaseAlignment(unittest.TestCase):
    def test_resolve_report_date_release_aligned(self) -> None:
        metrics = pd.DataFrame(
            {
                "report_date": [
                    dt.date(2025, 1, 7),
                    dt.date(2025, 1, 14),
                    dt.date(2025, 1, 21),
                ]
            }
        )

        resolved, _, _, _, _ = _resolve_report_date(
            metrics, dt.date(2025, 1, 21), None, "release_aligned", None
        )
        self.assertEqual(resolved, dt.date(2025, 1, 14))

        resolved, _, _, _, _ = _resolve_report_date(
            metrics, dt.date(2025, 1, 25), None, "release_aligned", None
        )
        self.assertEqual(resolved, dt.date(2025, 1, 21))

    def test_filter_excludes_future_report_dates(self) -> None:
        metrics = pd.DataFrame(
            {
                "symbol": ["EUR", "EUR", "EUR"],
                "report_date": [dt.date(2025, 12, 30), dt.date(2026, 1, 20), dt.date(2026, 2, 3)],
                "net_pct_oi": [1.0, 2.0, 3.0],
            }
        )

        hist, latest = _filter_metrics_for_report_date(metrics, dt.date(2026, 1, 20))

        self.assertTrue((pd.to_datetime(hist["report_date"]).dt.date <= dt.date(2026, 1, 20)).all())
        self.assertTrue((pd.to_datetime(latest["report_date"]).dt.date == dt.date(2026, 1, 20)).all())


if __name__ == "__main__":
    unittest.main()
