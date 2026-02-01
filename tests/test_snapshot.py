import datetime as dt
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from cot_bias.data.store import snapshot_df


class TestSnapshot(unittest.TestCase):
    def test_snapshot_cutoff(self) -> None:
        df = pd.DataFrame(
            {
                "symbol": ["EUR", "EUR", "EUR"],
                "report_date": [dt.date(2026, 1, 6), dt.date(2026, 1, 13), dt.date(2026, 1, 20)],
                "net_pct_oi": [1.0, 2.0, 3.0],
            }
        )
        snap = snapshot_df(df, dt.date(2026, 1, 6))
        self.assertTrue((snap["report_date"] <= dt.date(2026, 1, 6)).all())

    def test_history_max_le_asof(self) -> None:
        df = pd.DataFrame(
            {
                "symbol": ["EUR", "EUR"],
                "report_date": [dt.date(2026, 1, 6), dt.date(2026, 1, 13)],
                "net_pct_oi": [1.0, 2.0],
            }
        )
        snap = snapshot_df(df, dt.date(2026, 1, 6))
        self.assertLessEqual(snap["report_date"].max(), dt.date(2026, 1, 6))


if __name__ == "__main__":
    unittest.main()
