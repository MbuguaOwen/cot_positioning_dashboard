import datetime as dt
import unittest

import pandas as pd

from cot_bias.dashboard.tiered import compute_tiered_tables


class TestTieredTables(unittest.TestCase):
    def test_release_alignment_avoids_lookahead(self) -> None:
        history_df = pd.DataFrame(
            [
                {
                    "symbol": "EUR",
                    "report_date": dt.date(2026, 1, 13),
                    "net_pct_oi": 1.0,
                    "z_3y": 1.2,
                    "pctile_3y": 60.0,
                    "driver_net": 100.0,
                    "open_interest": 1000.0,
                    "report_type": "tff",
                },
                {
                    "symbol": "JPY",
                    "report_date": dt.date(2026, 1, 13),
                    "net_pct_oi": -1.0,
                    "z_3y": -1.0,
                    "pctile_3y": 40.0,
                    "driver_net": -90.0,
                    "open_interest": 900.0,
                    "report_type": "tff",
                },
                {
                    "symbol": "EUR",
                    "report_date": dt.date(2026, 1, 20),
                    "net_pct_oi": 1.5,
                    "z_3y": 1.4,
                    "pctile_3y": 62.0,
                    "driver_net": 110.0,
                    "open_interest": 980.0,
                    "report_type": "tff",
                },
                {
                    "symbol": "JPY",
                    "report_date": dt.date(2026, 1, 20),
                    "net_pct_oi": -1.2,
                    "z_3y": -1.1,
                    "pctile_3y": 38.0,
                    "driver_net": -95.0,
                    "open_interest": 920.0,
                    "report_type": "tff",
                },
            ]
        )

        # 2026-01-22 is before the Friday release of the 2026-01-20 report.
        tables, meta = compute_tiered_tables(
            history_df=history_df,
            pairs_df=None,
            as_of=dt.date(2026, 1, 22),
            cot_filter_cfg=None,
        )

        self.assertIsNotNone(tables)
        self.assertIsNotNone(meta)
        self.assertEqual(meta["resolved_report_date"], "2026-01-13")


if __name__ == "__main__":
    unittest.main()
