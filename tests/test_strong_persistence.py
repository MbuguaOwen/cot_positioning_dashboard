import unittest

import pandas as pd

from cot_bias.cli import _build_strong_persistent_bullish_3w, _filter_long_allowed


class TestStrongPersistence(unittest.TestCase):
    def test_filter_long_allowed_uses_only_allow_and_direction(self) -> None:
        decisions = pd.DataFrame(
            [
                {
                    "pair": "EURJPY",
                    "allow": True,
                    "direction_allowed": "LONG",
                    "macro_ok": False,
                    "news_ok": False,
                    "crowded_flag": "HIGH",
                },
                {
                    "pair": "GBPJPY",
                    "allow": False,
                    "direction_allowed": "LONG",
                },
                {
                    "pair": "AUDUSD",
                    "allow": True,
                    "direction_allowed": "SHORT",
                },
            ]
        )

        out = _filter_long_allowed(decisions)
        self.assertEqual(out["pair"].tolist(), ["EURJPY"])

    def test_build_strong_persistent_bullish_3w_exact_intersection(self) -> None:
        bullish_r0 = pd.DataFrame(
            [
                {
                    "pair": "EURJPY",
                    "confidence_score": 58.0,
                    "spread": 1.2,
                    "dSpread_1w": 0.2,
                    "crowded_flag": "NONE",
                },
                {
                    "pair": "GBPJPY",
                    "confidence_score": 51.0,
                    "spread": 1.9,
                    "dSpread_1w": 0.1,
                    "crowded_flag": "LOW",
                },
                {
                    "pair": "AUDUSD",
                    "confidence_score": 50.0,
                    "spread": 0.8,
                    "dSpread_1w": 0.3,
                    "crowded_flag": "NONE",
                },
            ]
        )
        bullish_r1 = pd.DataFrame(
            [
                {"pair": "EURJPY", "spread": 1.0},
                {"pair": "GBPJPY", "spread": 1.6},
                {"pair": "USDJPY", "spread": 1.1},
            ]
        )
        bullish_r2 = pd.DataFrame(
            [
                {"pair": "EURJPY", "spread": 0.9},
                {"pair": "GBPJPY", "spread": 1.4},
                {"pair": "NZDUSD", "spread": 0.7},
            ]
        )

        out = _build_strong_persistent_bullish_3w(bullish_r0, bullish_r1, bullish_r2)

        self.assertEqual(out["pair"].tolist(), ["EURJPY", "GBPJPY"])
        self.assertEqual(
            out.columns.tolist(),
            [
                "pair",
                "confidence_score",
                "spread",
                "dSpread_1w",
                "crowded_flag",
                "prev1_spread",
                "prev2_spread",
            ],
        )
        eur = out[out["pair"] == "EURJPY"].iloc[0]
        gbp = out[out["pair"] == "GBPJPY"].iloc[0]
        self.assertEqual(float(eur["prev1_spread"]), 1.0)
        self.assertEqual(float(eur["prev2_spread"]), 0.9)
        self.assertEqual(float(gbp["prev1_spread"]), 1.6)
        self.assertEqual(float(gbp["prev2_spread"]), 1.4)


if __name__ == "__main__":
    unittest.main()
