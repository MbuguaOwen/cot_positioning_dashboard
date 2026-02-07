import datetime as dt
import unittest

import pandas as pd

from cot_bias.filters.cot_filter import COTFilter, COTFilterInput, default_cot_filter_config


def _row(
    currency: str,
    report_date: dt.date,
    release_dt: dt.datetime,
    net_pctile: float,
    net_z: float,
    net_pct_oi: float,
) -> dict:
    return {
        "currency": currency,
        "report_date": report_date,
        "release_dt": release_dt,
        "net_pctile": net_pctile,
        "net_z": net_z,
        "net_pct_oi": net_pct_oi,
    }


class TestCOTFilter(unittest.TestCase):
    def _base_cfg(self) -> dict:
        cfg = default_cot_filter_config()
        cfg["strictness_tier"] = "loose"
        cfg["metric"]["kind"] = "net_pctile"
        cfg["spread_gate"]["threshold_by_metric"]["net_pctile"]["loose"] = 1.0
        cfg["flow_gate"]["enabled_by_tier"]["loose"] = False
        cfg["crowdedness"]["enabled_by_tier"]["loose"] = False
        cfg["macro"]["enabled"] = False
        cfg["news_blackout"]["enabled"] = False
        return {"cot_filter": cfg}

    def test_date_alignment_no_lookahead(self) -> None:
        cfg = self._base_cfg()
        filt = COTFilter(cfg)

        rd1 = dt.date(2026, 1, 6)
        rd2 = dt.date(2026, 1, 13)
        rel1 = dt.datetime(2026, 1, 9, 20, 30, tzinfo=dt.timezone.utc)
        rel2 = dt.datetime(2026, 1, 16, 20, 30, tzinfo=dt.timezone.utc)

        panel = pd.DataFrame(
            [
                _row("EUR", rd1, rel1, 65, 0.5, 2.0),
                _row("USD", rd1, rel1, 40, -0.2, -1.0),
                _row("EUR", rd2, rel2, 80, 1.0, 3.5),
                _row("USD", rd2, rel2, 30, -0.5, -2.0),
            ]
        )

        signal_before_rel2 = COTFilterInput(
            pair="EURUSD",
            signal_direction="long",
            signal_ts=dt.datetime(2026, 1, 16, 20, 29, tzinfo=dt.timezone.utc),
        )
        out = filt.evaluate(signal_before_rel2, panel)
        self.assertEqual(out.effective_report_date, rd1)
        self.assertAlmostEqual(float(out.components["spread"]), 25.0, places=6)

    def test_spread_computation_base_minus_quote(self) -> None:
        cfg = self._base_cfg()
        filt = COTFilter(cfg)

        rd = dt.date(2026, 1, 20)
        rel = dt.datetime(2026, 1, 23, 20, 30, tzinfo=dt.timezone.utc)
        panel = pd.DataFrame(
            [
                _row("EUR", rd, rel, 70, 0.7, 2.5),
                _row("USD", rd, rel, 40, -0.4, -1.8),
            ]
        )
        signal = COTFilterInput(
            pair="EURUSD",
            signal_direction="long",
            signal_ts=dt.datetime(2026, 1, 24, 0, 0, tzinfo=dt.timezone.utc),
        )
        out = filt.evaluate(signal, panel)
        self.assertAlmostEqual(float(out.components["spread"]), 30.0, places=6)

    def test_dspread_sign_for_long_and_short(self) -> None:
        cfg = self._base_cfg()
        cfg["cot_filter"]["strictness_tier"] = "balanced"
        cfg["cot_filter"]["flow_gate"]["enabled_by_tier"]["balanced"] = True
        cfg["cot_filter"]["flow_gate"]["lag_weeks_by_tier"]["balanced"] = 1
        cfg["cot_filter"]["flow_gate"]["min_dspread_by_metric"]["net_pctile"]["balanced"] = 1.0
        cfg["cot_filter"]["spread_gate"]["threshold_by_metric"]["net_pctile"]["balanced"] = 5.0
        filt = COTFilter(cfg)

        rd1 = dt.date(2026, 1, 6)
        rd2 = dt.date(2026, 1, 13)
        rel1 = dt.datetime(2026, 1, 9, 20, 30, tzinfo=dt.timezone.utc)
        rel2 = dt.datetime(2026, 1, 16, 20, 30, tzinfo=dt.timezone.utc)

        panel_long = pd.DataFrame(
            [
                _row("EUR", rd1, rel1, 60, 0.4, 1.5),
                _row("USD", rd1, rel1, 50, 0.0, 0.0),
                _row("EUR", rd2, rel2, 75, 0.9, 3.2),
                _row("USD", rd2, rel2, 50, 0.0, 0.0),
            ]
        )
        out_long = filt.evaluate(
            COTFilterInput(
                pair="EURUSD",
                signal_direction="long",
                signal_ts=dt.datetime(2026, 1, 17, 0, 0, tzinfo=dt.timezone.utc),
            ),
            panel_long,
        )
        self.assertIn("flow_ok", out_long.reasons)
        self.assertGreater(float(out_long.components["dspread"]), 0.0)

        panel_short = pd.DataFrame(
            [
                _row("EUR", rd1, rel1, 45, -0.2, -0.5),
                _row("USD", rd1, rel1, 55, 0.2, 0.8),
                _row("EUR", rd2, rel2, 30, -0.8, -2.0),
                _row("USD", rd2, rel2, 60, 0.5, 1.2),
            ]
        )
        out_short = filt.evaluate(
            COTFilterInput(
                pair="EURUSD",
                signal_direction="short",
                signal_ts=dt.datetime(2026, 1, 17, 0, 0, tzinfo=dt.timezone.utc),
            ),
            panel_short,
        )
        self.assertIn("flow_ok", out_short.reasons)
        self.assertLess(float(out_short.components["dspread"]), 0.0)

    def test_crowdedness_flags_and_block(self) -> None:
        cfg = self._base_cfg()
        cfg["cot_filter"]["strictness_tier"] = "strict"
        cfg["cot_filter"]["flow_gate"]["enabled_by_tier"]["strict"] = False
        cfg["cot_filter"]["spread_gate"]["threshold_by_metric"]["net_pctile"]["strict"] = 5.0
        cfg["cot_filter"]["crowdedness"]["enabled_by_tier"]["strict"] = True
        cfg["cot_filter"]["crowdedness"]["policy_by_tier"]["strict"] = "block"
        filt = COTFilter(cfg)

        rd = dt.date(2026, 1, 27)
        rel = dt.datetime(2026, 1, 30, 20, 30, tzinfo=dt.timezone.utc)
        panel = pd.DataFrame(
            [
                _row("EUR", rd, rel, 95, 1.2, 4.5),
                _row("USD", rd, rel, 5, -1.1, -3.9),
            ]
        )
        out = filt.evaluate(
            COTFilterInput(
                pair="EURUSD",
                signal_direction="long",
                signal_ts=dt.datetime(2026, 1, 31, 0, 0, tzinfo=dt.timezone.utc),
            ),
            panel,
        )
        self.assertFalse(out.allow)
        self.assertTrue(bool(out.components["crowded_base"]))
        self.assertTrue(bool(out.components["crowded_quote"]))
        self.assertIn("crowded_block", out.reasons)

    def test_macro_alignment_logic_for_usd_pairs(self) -> None:
        cfg = self._base_cfg()
        cfg["cot_filter"]["strictness_tier"] = "balanced"
        cfg["cot_filter"]["macro"]["enabled"] = True
        cfg["cot_filter"]["macro"]["macro_gate_required"] = True
        cfg["cot_filter"]["spread_gate"]["threshold_by_metric"]["net_pctile"]["balanced"] = 5.0
        cfg["cot_filter"]["flow_gate"]["enabled_by_tier"]["balanced"] = False
        filt = COTFilter(cfg)

        rd = dt.date(2026, 1, 27)
        rel = dt.datetime(2026, 1, 30, 20, 30, tzinfo=dt.timezone.utc)
        panel = pd.DataFrame(
            [
                _row("EUR", rd, rel, 75, 0.9, 3.0),
                _row("USD", rd, rel, 35, -0.7, -2.8),
            ]
        )
        out = filt.evaluate(
            COTFilterInput(
                pair="EURUSD",
                signal_direction="long",
                signal_ts=dt.datetime(2026, 1, 31, 0, 0, tzinfo=dt.timezone.utc),
            ),
            panel,
            macro_override=1.0,  # USD-bullish macro: should block EURUSD long.
        )
        self.assertFalse(out.allow)
        self.assertIn("macro_misaligned", out.reasons)
        self.assertIn("macro_block", out.reasons)


if __name__ == "__main__":
    unittest.main()
