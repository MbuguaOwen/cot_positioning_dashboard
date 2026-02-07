import datetime as dt
import unittest

import pandas as pd

from cot_bias.cli import _build_quality_gates_score, _load_quality_cfg


class TestQualityGatesScore(unittest.TestCase):
    def _base_inputs(self):
        strong = pd.DataFrame(
            [
                {
                    "pair": "EURJPY",
                    "confidence_score": 88.0,
                    "spread": 52.0,
                    "dSpread_1w": 1.0,
                    "crowded_flag": "NONE",
                    "prev1_spread": 48.0,
                    "prev2_spread": 46.0,
                }
            ]
        )
        decisions = pd.DataFrame(
            [
                {
                    "pair": "EURJPY",
                    "allow": True,
                    "direction_allowed": "LONG",
                }
            ]
        )
        prev1 = pd.DataFrame([{"pair": "EURJPY", "allow": True, "direction_allowed": "LONG"}])
        prev2 = pd.DataFrame([{"pair": "EURJPY", "allow": True, "direction_allowed": "LONG"}])
        pairs = pd.DataFrame(
            [
                {
                    "pair": "EURJPY",
                    "base": "EUR",
                    "quote": "JPY",
                    "base_z": 1.4,
                    "quote_z": -1.6,
                }
            ]
        )
        instruments = pd.DataFrame(
            [
                {"symbol": "EUR", "reversal_risk": "LOW"},
                {"symbol": "JPY", "reversal_risk": "LOW"},
            ]
        )
        return strong, decisions, prev1, prev2, pairs, instruments

    def test_quality_score_default_mode(self) -> None:
        strong, decisions, prev1, prev2, pairs, instruments = self._base_inputs()
        cfg = _load_quality_cfg({})

        out = _build_quality_gates_score(
            strong_persistent=strong,
            decisions_r0=decisions,
            bullish_r1=prev1,
            bullish_r2=prev2,
            pairs_snapshot=pairs,
            instruments_snapshot=instruments,
            as_of=dt.date(2026, 1, 27),
            resolved_release_dt=dt.datetime(2026, 1, 24, 0, 0, tzinfo=dt.timezone.utc),
            release_cfg={"timezone": "America/New_York"},
            quality_cfg=cfg,
        )

        self.assertEqual(len(out), 1)
        row = out.iloc[0]
        self.assertTrue(bool(row["A_release_ok"]))
        self.assertTrue(bool(row["A_consecutive_ok"]))
        self.assertTrue(bool(row["A_sign_ok"]))
        self.assertEqual(row["B_mode"], "B1_MIN_MAG")
        self.assertEqual(int(row["B_strength_points"]), 20)
        self.assertEqual(int(row["C_crowding_points"]), 20)
        self.assertEqual(int(row["D_points"]), 15)
        self.assertEqual(float(row["quality_score"]), 55.0)
        self.assertEqual(row["quality_tier"], "GOOD")

    def test_quality_fails_on_gate_a_sign(self) -> None:
        strong, decisions, prev1, prev2, pairs, instruments = self._base_inputs()
        strong.loc[0, "prev1_spread"] = -10.0

        out = _build_quality_gates_score(
            strong_persistent=strong,
            decisions_r0=decisions,
            bullish_r1=prev1,
            bullish_r2=prev2,
            pairs_snapshot=pairs,
            instruments_snapshot=instruments,
            as_of=dt.date(2026, 1, 27),
            resolved_release_dt=dt.datetime(2026, 1, 24, 0, 0, tzinfo=dt.timezone.utc),
            release_cfg={"timezone": "America/New_York"},
            quality_cfg=_load_quality_cfg({}),
        )

        row = out.iloc[0]
        self.assertFalse(bool(row["A_sign_ok"]))
        self.assertEqual(float(row["quality_score"]), 0.0)
        self.assertEqual(row["quality_tier"], "FAIL")

    def test_quality_fails_hard_block_crowding_policy(self) -> None:
        strong, decisions, prev1, prev2, pairs, instruments = self._base_inputs()
        strong.loc[0, "crowded_flag"] = "BASE_CROWDED"

        out = _build_quality_gates_score(
            strong_persistent=strong,
            decisions_r0=decisions,
            bullish_r1=prev1,
            bullish_r2=prev2,
            pairs_snapshot=pairs,
            instruments_snapshot=instruments,
            as_of=dt.date(2026, 1, 27),
            resolved_release_dt=dt.datetime(2026, 1, 24, 0, 0, tzinfo=dt.timezone.utc),
            release_cfg={"timezone": "America/New_York"},
            quality_cfg=_load_quality_cfg({"quality_crowding_policy": "HARD_BLOCK"}),
        )

        row = out.iloc[0]
        self.assertFalse(bool(row["C_crowding_ok"]))
        self.assertEqual(float(row["quality_score"]), 0.0)
        self.assertEqual(row["quality_tier"], "FAIL")


if __name__ == "__main__":
    unittest.main()
