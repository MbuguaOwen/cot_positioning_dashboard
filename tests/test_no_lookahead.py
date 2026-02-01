import datetime as dt
import unittest

import pandas as pd

from cot_bias.metrics.features import compute_metrics


class TestNoLookahead(unittest.TestCase):
    def test_no_lookahead_metrics(self) -> None:
        dates = [dt.date(2026, 1, 6), dt.date(2026, 1, 13), dt.date(2026, 1, 20)]
        df = pd.DataFrame(
            {
                "symbol": ["EUR"] * 3,
                "report_date": dates,
                "net_pct_oi": [1.0, 2.0, 3.0],
            }
        )
        base = compute_metrics(df, window=2)
        base_row = base[base["report_date"] == dt.date(2026, 1, 13)].iloc[0]

        df2 = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "symbol": ["EUR"],
                        "report_date": [dt.date(2026, 1, 27)],
                        "net_pct_oi": [100.0],
                    }
                ),
            ],
            ignore_index=True,
        )
        updated = compute_metrics(df2, window=2)
        updated_row = updated[updated["report_date"] == dt.date(2026, 1, 13)].iloc[0]

        for col in ["z_3y", "pctile_3y", "delta_4w", "score"]:
            b = base_row[col]
            u = updated_row[col]
            if pd.isna(b) and pd.isna(u):
                continue
            self.assertEqual(b, u)


if __name__ == "__main__":
    unittest.main()
