import datetime as dt
import unittest

import numpy as np
import pandas as pd

from cot_bias.fx import usd_proxy_from_z
from cot_bias.compute import compute_pairs
from cot_bias.reporting import resolve_report_date


class TestUsdProxy(unittest.TestCase):
    def test_usd_proxy_sign(self) -> None:
        z1 = {"EUR": 1.0, "JPY": 0.0, "GBP": 0.0}
        z2 = {"EUR": 2.0, "JPY": 0.0, "GBP": 0.0}
        usd1 = usd_proxy_from_z(z1, weights=None)
        usd2 = usd_proxy_from_z(z2, weights=None)
        self.assertLess(usd2, usd1)


class TestPairs(unittest.TestCase):
    def test_pair_identity(self) -> None:
        inst_latest = pd.DataFrame(
            [
                {"symbol": "EUR", "net_pct_oi_3y_z": 1.2},
                {"symbol": "JPY", "net_pct_oi_3y_z": -0.5},
                {"symbol": "GBP", "net_pct_oi_3y_z": 0.1},
            ]
        )
        pairs_df = compute_pairs(inst_latest, usd_mode="basket", usd_weights="equal")
        z_by = {"EUR": 1.2, "JPY": -0.5, "GBP": 0.1}
        usd = usd_proxy_from_z(z_by, weights=None)

        eurusd = pairs_df[pairs_df["pair"] == "EURUSD"].iloc[0]
        usdjpy = pairs_df[pairs_df["pair"] == "USDJPY"].iloc[0]

        self.assertTrue(np.isclose(eurusd["pair_z"], z_by["EUR"] - usd, equal_nan=False))
        self.assertTrue(np.isclose(usdjpy["pair_z"], usd - z_by["JPY"], equal_nan=False))


class TestDateResolution(unittest.TestCase):
    def test_resolve_report_date(self) -> None:
        available = [
            dt.date(2026, 1, 6),
            dt.date(2026, 1, 12),
            dt.date(2026, 1, 20),
        ]
        # Requested Friday; closest Tuesday is 2026-01-13, which is missing.
        resolved = resolve_report_date(dt.date(2026, 1, 16), available, max_weeks=10)
        # Monday holiday adjustment should select 2026-01-12.
        self.assertEqual(resolved, dt.date(2026, 1, 12))


if __name__ == "__main__":
    unittest.main()
