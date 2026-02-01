import datetime as dt
import unittest

import pandas as pd

from cot_bias.dashboard.gate import compute_gated_fx_pairs


class TestFxGate(unittest.TestCase):
    def test_gate_inclusion_and_extreme_allowed(self) -> None:
        latest_currency = pd.DataFrame(
            [
                {"symbol": "AAA", "bias": "BULLISH", "z_3y": 1.2},
                {"symbol": "BBB", "bias": "BEARISH", "z_3y": -1.6},
                {"symbol": "CCC", "bias": "BULLISH", "z_3y": 0.4},
            ]
        )
        pairs_latest = pd.DataFrame(
            [
                {
                    "pair": "AAABBB",
                    "pair_z": 1.8,
                    "bias": "BULLISH",
                    "reversal_risk": "EXTREME",
                    "base": "AAA",
                    "quote": "BBB",
                    "base_z": 1.2,
                    "quote_z": -1.6,
                },
                {
                    "pair": "AAACCC",
                    "pair_z": 1.1,
                    "bias": "BULLISH",
                    "reversal_risk": "MED",
                    "base": "AAA",
                    "quote": "CCC",
                    "base_z": 1.2,
                    "quote_z": 0.4,
                },
                {
                    "pair": "BBBCCC",
                    "pair_z": -2.0,
                    "bias": "BEARISH",
                    "reversal_risk": "EXTREME",
                    "base": "BBB",
                    "quote": "CCC",
                    "base_z": -1.6,
                    "quote_z": 0.4,
                },
            ]
        )

        out = compute_gated_fx_pairs(
            currency_latest=latest_currency,
            currency_previous=pd.DataFrame(),
            pairs_latest=pairs_latest,
            pairs_previous=pd.DataFrame(),
            latest_release_date=dt.date(2026, 1, 20),
            previous_release_date=None,
        )

        pairs = out["pair"].tolist()
        self.assertIn("AAABBB", pairs)
        self.assertNotIn("AAACCC", pairs)
        self.assertNotIn("BBBCCC", pairs)

        row = out[out["pair"] == "AAABBB"].iloc[0]
        self.assertEqual(row["reversal_risk"], "EXTREME")
        self.assertTrue(bool(row["reversal_risk_extreme"]))

    def test_passes_2_reports_flag(self) -> None:
        latest_currency = pd.DataFrame(
            [
                {"symbol": "AAA", "bias": "BULLISH", "z_3y": 1.2},
                {"symbol": "BBB", "bias": "BEARISH", "z_3y": -1.6},
                {"symbol": "CCC", "bias": "BEARISH", "z_3y": -1.1},
            ]
        )
        pairs_latest = pd.DataFrame(
            [
                {
                    "pair": "AAABBB",
                    "pair_z": 1.8,
                    "bias": "BULLISH",
                    "reversal_risk": "HIGH",
                    "base": "AAA",
                    "quote": "BBB",
                    "base_z": 1.2,
                    "quote_z": -1.6,
                },
                {
                    "pair": "AAACCC",
                    "pair_z": 1.2,
                    "bias": "BULLISH",
                    "reversal_risk": "MED",
                    "base": "AAA",
                    "quote": "CCC",
                    "base_z": 1.2,
                    "quote_z": -1.1,
                },
            ]
        )

        prev_currency = pd.DataFrame(
            [
                {"symbol": "AAA", "bias": "BULLISH", "z_3y": 1.4},
                {"symbol": "BBB", "bias": "BEARISH", "z_3y": -1.5},
                {"symbol": "CCC", "bias": "NEUTRAL", "z_3y": -0.2},
            ]
        )
        pairs_prev = pd.DataFrame(
            [
                {
                    "pair": "AAABBB",
                    "pair_z": 1.7,
                    "bias": "BULLISH",
                    "reversal_risk": "HIGH",
                    "base": "AAA",
                    "quote": "BBB",
                    "base_z": 1.4,
                    "quote_z": -1.5,
                },
                {
                    "pair": "AAACCC",
                    "pair_z": 1.3,
                    "bias": "BULLISH",
                    "reversal_risk": "MED",
                    "base": "AAA",
                    "quote": "CCC",
                    "base_z": 1.4,
                    "quote_z": -0.2,
                },
            ]
        )

        out = compute_gated_fx_pairs(
            currency_latest=latest_currency,
            currency_previous=prev_currency,
            pairs_latest=pairs_latest,
            pairs_previous=pairs_prev,
            latest_release_date=dt.date(2026, 1, 20),
            previous_release_date=dt.date(2026, 1, 13),
        )

        row1 = out[out["pair"] == "AAABBB"].iloc[0]
        row2 = out[out["pair"] == "AAACCC"].iloc[0]
        self.assertTrue(bool(row1["passes_2_reports"]))
        self.assertFalse(bool(row2["passes_2_reports"]))


if __name__ == "__main__":
    unittest.main()
