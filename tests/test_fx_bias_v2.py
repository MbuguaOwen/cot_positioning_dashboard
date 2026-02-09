import copy
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from cot_bias.fx_bias_v2.compose import build_currency_polarity_pairs, build_pair_rows
from cot_bias.fx_bias_v2.config import DEFAULTS, FxBiasV2Config
from cot_bias.fx_bias_v2.engine import run_fx_bias_engine_v2_from_paths
from cot_bias.fx_bias_v2.io import load_price_bars, load_rates, to_ts


def _write_price_csv(path: Path, pair: str, tf: str, periods: int, freq: str, start: str, slope: float) -> None:
    ts = pd.date_range(start=start, periods=periods, freq=freq, tz="UTC")
    base = pd.Series(range(periods), dtype=float)
    close = 1.0 + slope * base
    df = pd.DataFrame(
        {
            "ts": ts,
            "open": close - 0.0001,
            "high": close + 0.0002,
            "low": close - 0.0002,
            "close": close,
            "volume": 1000.0,
        }
    )
    df.to_csv(path / f"{pair}_{tf}.csv", index=False)


def _base_cfg() -> FxBiasV2Config:
    return FxBiasV2Config(raw=copy.deepcopy(DEFAULTS))


class TestFxBiasV2(unittest.TestCase):
    def test_to_ts_handles_epoch_seconds_and_milliseconds(self) -> None:
        expected = pd.Timestamp("2024-01-01T00:00:00Z")
        parsed_ms = to_ts(pd.Series([1704067200000]))
        parsed_sec = to_ts(pd.Series([1704067200]))
        parsed_mixed = to_ts(pd.Series(["2024-01-02T00:00:00Z", "1704067200"]))
        self.assertEqual(parsed_ms.iloc[0], expected)
        self.assertEqual(parsed_sec.iloc[0], expected)
        self.assertEqual(parsed_mixed.iloc[0], pd.Timestamp("2024-01-02T00:00:00Z"))
        self.assertEqual(parsed_mixed.iloc[1], expected)

    def test_semicolon_csv_auto_detect_for_rates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rates_path = tmp_path / "rates_semicolon.csv"
            pd.DataFrame(
                {
                    "currency": ["USD", "EUR"],
                    "ts": [1704067200000, 1704067200000],
                    "y2_value": [4.5, 2.5],
                }
            ).to_csv(rates_path, sep=";", index=False)

            loaded = load_rates(rates_path)
            self.assertEqual(len(loaded), 2)
            self.assertTrue((loaded["ts"] == pd.Timestamp("2024-01-01T00:00:00Z")).all())

    def test_semicolon_csv_auto_detect_for_price_bars(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            prices_path = tmp_path / "EURUSD_D1_semicolon.csv"
            pd.DataFrame(
                {
                    "pair": ["EUR/USD", "EUR/USD"],
                    "timeframe": ["D1", "D1"],
                    "ts": [1704067200, 1704153600],
                    "open": [1.1, 1.2],
                    "high": [1.3, 1.4],
                    "low": [1.0, 1.1],
                    "close": [1.2, 1.3],
                    "volume": [1000, 1001],
                }
            ).to_csv(prices_path, sep=";", index=False)

            loaded = load_price_bars(prices_path, ["D1"])
            self.assertEqual(len(loaded), 2)
            self.assertTrue((loaded["pair"] == "EURUSD").all())
            self.assertEqual(loaded["ts"].min(), pd.Timestamp("2024-01-01T00:00:00Z"))

    def test_schema_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            prices_dir = tmp_path / "prices"
            prices_dir.mkdir(parents=True, exist_ok=True)
            for tf, freq, n in [("D1", "1D", 360), ("W1", "7D", 260), ("H4", "4h", 500), ("H1", "1h", 500)]:
                _write_price_csv(prices_dir, "EURUSD", tf, n, freq, "2023-01-01", 0.0002)
                _write_price_csv(prices_dir, "USDJPY", tf, n, freq, "2023-01-01", -0.0002)

            out_dir = tmp_path / "out"
            bundle = run_fx_bias_engine_v2_from_paths(
                as_of="2026-01-15T20:00:00Z",
                cfg=_base_cfg(),
                prices_path=str(prices_dir),
                rates_path=None,
                risk_path=None,
                carry_path=None,
                skew_path=None,
                cot_path=None,
                out_dir=str(out_dir),
                pairs_csv=None,
            )

            self.assertIn("RunMeta", bundle)
            self.assertIn("CapabilityMatrix", bundle)
            self.assertIn("A_Currency_Strength_Bias", bundle)
            self.assertIn("B_Pair_Bias_Trade_Gate", bundle)
            self.assertIn("C_Diagnostics", bundle)
            self.assertIn("D_Currency_Polarity_Pairs", bundle)

            ccy_rows = bundle["A_Currency_Strength_Bias"]
            self.assertEqual(len(ccy_rows), 8)
            self.assertTrue((out_dir / "fx_bias_v2_run.json").exists())
            self.assertTrue((out_dir / "fx_bias_v2_dashboard.html").exists())
            self.assertTrue((out_dir / "currency_strength_bias.csv").exists())
            self.assertTrue((out_dir / "pair_bias_trade_gate.csv").exists())
            self.assertTrue((out_dir / "capability_matrix.csv").exists())
            self.assertTrue((out_dir / "currency_polarity_pairs.csv").exists())

            loaded = json.loads((out_dir / "fx_bias_v2_run.json").read_text(encoding="utf-8"))
            self.assertIn("RunMeta", loaded)
            self.assertIn("CapabilityMatrix", loaded)
            self.assertIn("D_Currency_Polarity_Pairs", loaded)

    def test_currency_polarity_pairs_orientation_filter_and_ranking(self) -> None:
        currency_rows = [
            {"Currency": "EUR", "BiasScore_Final": 85.0, "Confidence": 90},
            {"Currency": "USD", "BiasScore_Final": -55.0, "Confidence": 88},
            {"Currency": "JPY", "BiasScore_Final": 70.0, "Confidence": 85},
            {"Currency": "GBP", "BiasScore_Final": -40.0, "Confidence": 50},  # filtered by confidence
        ]
        rows = build_currency_polarity_pairs(
            currency_rows=currency_rows,
            tradable_pairs=["EURUSD", "USDJPY", "GBPUSD"],
            spread_threshold=20.0,
            min_confidence=60,
            top_n=20,
        )
        self.assertEqual([r["Pair"] for r in rows], ["EURUSD", "USDJPY"])

        eurusd = rows[0]
        self.assertEqual(eurusd["Base"], "EUR")
        self.assertEqual(eurusd["Quote"], "USD")
        self.assertEqual(eurusd["Spread"], 140.0)
        self.assertEqual(eurusd["Opposition"], 55.0)
        self.assertEqual(eurusd["ConvictionScore"], 55.0)
        self.assertEqual(eurusd["ImpliedDirection"], "LONG")
        self.assertEqual(eurusd["PolarityBucket"], "BASE_BULL__QUOTE_BEAR")
        self.assertEqual(eurusd["BaseConf"], 90)
        self.assertEqual(eurusd["QuoteConf"], 88)

        usdjpy = rows[1]
        self.assertEqual(usdjpy["Spread"], -125.0)
        self.assertEqual(usdjpy["ConvictionScore"], -55.0)
        self.assertEqual(usdjpy["ImpliedDirection"], "SHORT")
        self.assertEqual(usdjpy["PolarityBucket"], "BASE_BEAR__QUOTE_BULL")

    def test_engine_uses_configured_tradable_pairs_for_section_d(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            prices_dir = tmp_path / "prices"
            prices_dir.mkdir(parents=True, exist_ok=True)
            for tf, freq, n in [("D1", "1D", 360), ("W1", "7D", 260), ("H4", "4h", 500), ("H1", "1h", 500)]:
                _write_price_csv(prices_dir, "EURUSD", tf, n, freq, "2023-01-01", 0.0002)
                _write_price_csv(prices_dir, "USDJPY", tf, n, freq, "2023-01-01", -0.0002)

            raw_cfg = copy.deepcopy(DEFAULTS)
            raw_cfg["tradable_pairs"] = ["EURJPY", "USDCHF"]
            raw_cfg["currency_polarity_pairs"] = {
                "spread_threshold": 20.0,
                "min_confidence": 0,
                "top_n": 20,
            }
            cfg = FxBiasV2Config(raw=raw_cfg)

            bundle = run_fx_bias_engine_v2_from_paths(
                as_of="2026-01-15T20:00:00Z",
                cfg=cfg,
                prices_path=str(prices_dir),
                rates_path=None,
                risk_path=None,
                carry_path=None,
                skew_path=None,
                cot_path=None,
                out_dir=str(tmp_path / "out"),
                pairs_csv=None,
            )

            d_pairs = [r["Pair"] for r in bundle["D_Currency_Polarity_Pairs"]]
            self.assertEqual(set(d_pairs), {"EURJPY", "USDCHF"})
            self.assertEqual(len(d_pairs), 2)

    def test_auto_build_d1_from_intraday_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            prices_root = tmp_path / "prices"
            intraday_dir = prices_root / "nested" / "intraday"
            intraday_dir.mkdir(parents=True, exist_ok=True)

            # No D1 files on purpose. Loader must recurse and engine must auto-build D1.
            _write_price_csv(intraday_dir, "EURUSD", "H4", 3000, "4h", "2025-01-01", 0.0002)
            _write_price_csv(intraday_dir, "USDJPY", "H4", 3000, "4h", "2025-01-01", -0.0002)
            _write_price_csv(intraday_dir, "EURUSD", "H1", 12000, "1h", "2025-01-01", 0.00002)
            _write_price_csv(intraday_dir, "USDJPY", "H1", 12000, "1h", "2025-01-01", -0.00002)

            out_dir = tmp_path / "out"
            bundle = run_fx_bias_engine_v2_from_paths(
                as_of="2026-01-15T20:00:00Z",
                cfg=_base_cfg(),
                prices_path=str(prices_root),
                rates_path=None,
                risk_path=None,
                carry_path=None,
                skew_path=None,
                cot_path=None,
                out_dir=str(out_dir),
                pairs_csv=None,
            )

            warnings = bundle["RunMeta"]["warnings"]
            self.assertFalse(any("PRICE D1 SERIES MISSING" in w for w in warnings))
            self.assertFalse(any("PRICE STALE OR MISSING" in w for w in warnings))
            self.assertTrue(any("PRICE D1 AUTO-BUILT" in w for w in warnings))

            price_row = [r for r in bundle["CapabilityMatrix"] if r["Component"] == "PRICE"][0]
            self.assertTrue(bool(price_row["UsedInBias"]))

    def test_no_lookahead_rates_component(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            prices_dir = tmp_path / "prices"
            prices_dir.mkdir(parents=True, exist_ok=True)
            for tf, freq, n in [("D1", "1D", 360), ("W1", "7D", 260), ("H4", "4h", 500), ("H1", "1h", 500)]:
                _write_price_csv(prices_dir, "EURUSD", tf, n, freq, "2023-01-01", 0.0002)
                _write_price_csv(prices_dir, "USDJPY", tf, n, freq, "2023-01-01", -0.0002)

            base_ts = pd.date_range(start="2025-01-01", periods=80, freq="1D", tz="UTC")
            rates = pd.DataFrame(
                {
                    "currency": ["USD"] * len(base_ts),
                    "ts": base_ts,
                    "y2_value": [3.0 + i * 0.001 for i in range(len(base_ts))],
                }
            )
            rates_path = tmp_path / "rates.csv"
            rates.to_csv(rates_path, index=False)

            as_of = "2025-03-10T00:00:00Z"
            base = run_fx_bias_engine_v2_from_paths(
                as_of=as_of,
                cfg=_base_cfg(),
                prices_path=str(prices_dir),
                rates_path=str(rates_path),
                risk_path=None,
                carry_path=None,
                skew_path=None,
                cot_path=None,
                out_dir=str(tmp_path / "out1"),
                pairs_csv=None,
            )

            rates2 = pd.concat(
                [
                    rates,
                    pd.DataFrame(
                        {
                            "currency": ["USD"],
                            "ts": [pd.Timestamp("2027-01-01", tz="UTC")],
                            "y2_value": [100.0],
                        }
                    ),
                ],
                ignore_index=True,
            )
            rates2_path = tmp_path / "rates2.csv"
            rates2.to_csv(rates2_path, index=False)

            updated = run_fx_bias_engine_v2_from_paths(
                as_of=as_of,
                cfg=_base_cfg(),
                prices_path=str(prices_dir),
                rates_path=str(rates2_path),
                risk_path=None,
                carry_path=None,
                skew_path=None,
                cot_path=None,
                out_dir=str(tmp_path / "out2"),
                pairs_csv=None,
            )

            base_usd = [r for r in base["A_Currency_Strength_Bias"] if r["Currency"] == "USD"][0]
            upd_usd = [r for r in updated["A_Currency_Strength_Bias"] if r["Currency"] == "USD"][0]
            self.assertEqual(base_usd["RateScore"], upd_usd["RateScore"])

    def test_stale_rates_marked_not_used(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            prices_dir = tmp_path / "prices"
            prices_dir.mkdir(parents=True, exist_ok=True)
            for tf, freq, n in [("D1", "1D", 360), ("W1", "7D", 260), ("H4", "4h", 500), ("H1", "1h", 500)]:
                _write_price_csv(prices_dir, "EURUSD", tf, n, freq, "2023-01-01", 0.0002)
                _write_price_csv(prices_dir, "USDJPY", tf, n, freq, "2023-01-01", -0.0002)

            stale_rates = pd.DataFrame(
                {
                    "currency": ["USD", "EUR"],
                    "ts": [pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC")],
                    "y2_value": [3.0, 2.0],
                }
            )
            rates_path = tmp_path / "rates.csv"
            stale_rates.to_csv(rates_path, index=False)

            bundle = run_fx_bias_engine_v2_from_paths(
                as_of="2026-01-15T20:00:00Z",
                cfg=_base_cfg(),
                prices_path=str(prices_dir),
                rates_path=str(rates_path),
                risk_path=None,
                carry_path=None,
                skew_path=None,
                cot_path=None,
                out_dir=str(tmp_path / "out"),
                pairs_csv=None,
            )

            rates_row = [r for r in bundle["CapabilityMatrix"] if r["Component"] == "RATES"][0]
            self.assertFalse(bool(rates_row["UsedInBias"]))
            usd_row = [r for r in bundle["A_Currency_Strength_Bias"] if r["Currency"] == "USD"][0]
            self.assertIsNone(usd_row["RateScore"])

    def test_stale_carry_marks_pair_reason_as_na(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            prices_dir = tmp_path / "prices"
            prices_dir.mkdir(parents=True, exist_ok=True)
            for tf, freq, n in [("D1", "1D", 360), ("W1", "7D", 260), ("H4", "4h", 500), ("H1", "1h", 500)]:
                _write_price_csv(prices_dir, "EURUSD", tf, n, freq, "2023-01-01", 0.0002)
                _write_price_csv(prices_dir, "USDJPY", tf, n, freq, "2023-01-01", -0.0002)

            stale_carry = pd.DataFrame(
                {
                    "pair": ["EURUSD", "USDJPY"],
                    "ts": [pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC")],
                    "forward_points": [12.0, -10.0],
                }
            )
            carry_path = tmp_path / "carry.csv"
            stale_carry.to_csv(carry_path, index=False)

            bundle = run_fx_bias_engine_v2_from_paths(
                as_of="2026-01-15T20:00:00Z",
                cfg=_base_cfg(),
                prices_path=str(prices_dir),
                rates_path=None,
                risk_path=None,
                carry_path=str(carry_path),
                skew_path=None,
                cot_path=None,
                out_dir=str(tmp_path / "out"),
                pairs_csv=None,
            )

            carry_row = [r for r in bundle["CapabilityMatrix"] if r["Component"] == "CARRY"][0]
            self.assertFalse(bool(carry_row["UsedInBias"]))
            for row in bundle["B_Pair_Bias_Trade_Gate"]:
                reasons = row["ReasonCodes"]
                self.assertIn("CARRY_NA", reasons)
                self.assertNotIn("CARRY_TAILWIND", reasons)
                self.assertNotIn("CARRY_HEADWIND", reasons)

    def test_fresh_pair_carry_reason_is_na_if_carry_not_used(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            prices_dir = tmp_path / "prices"
            prices_dir.mkdir(parents=True, exist_ok=True)
            for tf, freq, n in [("D1", "1D", 360), ("W1", "7D", 260), ("H4", "4h", 500), ("H1", "1h", 500)]:
                _write_price_csv(prices_dir, "EURUSD", tf, n, freq, "2023-01-01", 0.0002)
                _write_price_csv(prices_dir, "USDJPY", tf, n, freq, "2023-01-01", -0.0002)

            as_of = "2026-01-15T20:00:00Z"
            fresh_carry = pd.DataFrame(
                {
                    "pair": ["EURUSD", "USDJPY"],
                    "ts": [pd.Timestamp(as_of), pd.Timestamp(as_of)],
                    "forward_points": [8.0, -6.0],
                }
            )
            carry_path = tmp_path / "carry.csv"
            fresh_carry.to_csv(carry_path, index=False)

            bundle = run_fx_bias_engine_v2_from_paths(
                as_of=as_of,
                cfg=_base_cfg(),
                prices_path=str(prices_dir),
                rates_path=None,
                risk_path=None,
                carry_path=str(carry_path),
                skew_path=None,
                cot_path=None,
                out_dir=str(tmp_path / "out"),
                pairs_csv=None,
            )

            carry_row = [r for r in bundle["CapabilityMatrix"] if r["Component"] == "CARRY"][0]
            self.assertFalse(bool(carry_row["UsedInBias"]))
            for row in bundle["B_Pair_Bias_Trade_Gate"]:
                reasons = row["ReasonCodes"]
                self.assertIn("CARRY_NA", reasons)
                self.assertNotIn("CARRY_TAILWIND", reasons)
                self.assertNotIn("CARRY_HEADWIND", reasons)

    def test_carry_reason_emitted_when_carry_used(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            prices_dir = tmp_path / "prices"
            prices_dir.mkdir(parents=True, exist_ok=True)
            for tf, freq, n in [("D1", "1D", 360), ("W1", "7D", 260), ("H4", "4h", 500), ("H1", "1h", 500)]:
                _write_price_csv(prices_dir, "EURUSD", tf, n, freq, "2023-01-01", 0.0002)
                _write_price_csv(prices_dir, "USDJPY", tf, n, freq, "2023-01-01", -0.0002)

            as_of = "2026-01-15T20:00:00Z"
            carry_ts = pd.date_range(end=pd.Timestamp(as_of), periods=320, freq="1D", tz="UTC")
            carry_rows = []
            for i, ts in enumerate(carry_ts):
                carry_rows.append({"pair": "EURUSD", "ts": ts, "forward_points": 8.0 if i % 2 == 0 else -8.0})
                carry_rows.append({"pair": "USDJPY", "ts": ts, "forward_points": -6.0 if i % 3 == 0 else 6.0})
            carry_path = tmp_path / "carry.csv"
            pd.DataFrame(carry_rows).to_csv(carry_path, index=False)

            bundle = run_fx_bias_engine_v2_from_paths(
                as_of=as_of,
                cfg=_base_cfg(),
                prices_path=str(prices_dir),
                rates_path=None,
                risk_path=None,
                carry_path=str(carry_path),
                skew_path=None,
                cot_path=None,
                out_dir=str(tmp_path / "out"),
                pairs_csv=None,
            )

            carry_row = [r for r in bundle["CapabilityMatrix"] if r["Component"] == "CARRY"][0]
            self.assertTrue(bool(carry_row["UsedInBias"]))
            for row in bundle["B_Pair_Bias_Trade_Gate"]:
                reasons = row["ReasonCodes"]
                self.assertTrue(("CARRY_TAILWIND" in reasons) or ("CARRY_HEADWIND" in reasons))
                self.assertNotIn("CARRY_NA", reasons)

    def test_gate_rule_behavior(self) -> None:
        latest_trends = pd.DataFrame(
            [
                {"pair": "EURUSD", "timeframe": "D1", "trend": "BULL", "strength": 70.0},
                {"pair": "EURUSD", "timeframe": "H4", "trend": "RANGE", "strength": 20.0},
                {"pair": "EURUSD", "timeframe": "W1", "trend": "BULL", "strength": 80.0},
                {"pair": "EURUSD", "timeframe": "H1", "trend": "BULL", "strength": 65.0},
                {"pair": "GBPJPY", "timeframe": "D1", "trend": "BULL", "strength": 70.0},
                {"pair": "GBPJPY", "timeframe": "H4", "trend": "BEAR", "strength": 80.0},
                {"pair": "GBPJPY", "timeframe": "W1", "trend": "BULL", "strength": 75.0},
                {"pair": "GBPJPY", "timeframe": "H1", "trend": "BEAR", "strength": 70.0},
            ]
        )
        bias_scores = {"EUR": 40.0, "USD": 0.0, "GBP": 40.0, "JPY": 0.0}
        cfg = _base_cfg()
        rows = build_pair_rows(
            pairs=["EURUSD", "GBPJPY"],
            latest_trends=latest_trends,
            bias_scores=bias_scores,
            rate_scores={c: None for c in ["USD", "EUR", "JPY", "GBP", "CHF", "AUD", "NZD", "CAD"]},
            risk_regime="NEUTRAL",
            pair_carry_sign={},
            skew_scores={c: None for c in ["USD", "EUR", "JPY", "GBP", "CHF", "AUD", "NZD", "CAD"]},
            cfg=cfg,
            block_all=False,
        )
        eurusd = [r for r in rows if r["Pair"] == "EURUSD"][0]
        gbpjpy = [r for r in rows if r["Pair"] == "GBPJPY"][0]
        self.assertEqual(eurusd["GatingRuleForBOT"], "ALLOW_LONG")
        self.assertEqual(gbpjpy["GatingRuleForBOT"], "BLOCK")


if __name__ == "__main__":
    unittest.main()
