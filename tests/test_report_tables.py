import datetime as dt
import unittest

import pandas as pd

from cot_bias.filters.cot_filter import COTFilterDecision, default_cot_filter_config
from cot_bias.reporting import (
    _crowded_flag,
    _primary_blocked_by,
    build_spread_flow_table,
    build_tier_thresholds_table,
    resolve_report_date_by_release,
)


class TestReportTables(unittest.TestCase):
    def test_resolve_report_date_by_release_no_lookahead(self) -> None:
        available = [dt.date(2026, 1, 6), dt.date(2026, 1, 13)]
        cfg = default_cot_filter_config()["release_alignment"]
        # Thursday before Friday release of 2026-01-13
        resolved = resolve_report_date_by_release(dt.date(2026, 1, 15), available, cfg)
        self.assertEqual(resolved, dt.date(2026, 1, 6))
        # Friday should allow the 2026-01-13 report
        resolved2 = resolve_report_date_by_release(dt.date(2026, 1, 16), available, cfg)
        self.assertEqual(resolved2, dt.date(2026, 1, 13))

    def test_primary_blocked_by_priority(self) -> None:
        reasons = ["spread_fail", "direction_fail", "flow_fail"]
        blocked_by = _primary_blocked_by(reasons)
        self.assertEqual(blocked_by, "direction_fail")

    def test_crowded_flag_logic(self) -> None:
        self.assertEqual(_crowded_flag(True, True), "BOTH_EXTREMES")
        self.assertEqual(_crowded_flag(True, False), "BASE_CROWDED")
        self.assertEqual(_crowded_flag(False, True), "QUOTE_CROWDED")
        self.assertEqual(_crowded_flag(False, False), "NONE")

    def test_tier_thresholds_table(self) -> None:
        cfg = default_cot_filter_config()
        cfg["strictness_tier"] = "balanced"
        df = build_tier_thresholds_table(cfg)
        self.assertEqual(len(df), 4)
        row = df[df["tier"] == "BALANCED"].iloc[0]
        self.assertEqual(row["metric_used"], cfg["metric"]["kind"])
        self.assertEqual(
            row["spread_X"],
            cfg["spread_gate"]["threshold_by_metric"][cfg["metric"]["kind"]]["balanced"],
        )

    def test_spread_flow_table_values(self) -> None:
        cfg = default_cot_filter_config()
        cfg["strictness_tier"] = "balanced"
        cfg["spread_gate"]["threshold_by_metric"]["net_pctile"]["balanced"] = 1.0
        cfg["flow_gate"]["enabled_by_tier"]["balanced"] = True
        cfg["flow_gate"]["min_dspread_by_metric"]["net_pctile"]["balanced"] = 0.5

        decision = COTFilterDecision(
            allow=True,
            direction="long",
            score=80.0,
            reasons=["direction_ok", "spread_ok", "flow_ok"],
            strictness_tier="balanced",
        )
        ctx = {
            "pair": "EURUSD",
            "signal_direction": "long",
            "spread": 2.0,
            "dspread_1w": 0.75,
            "decision": decision,
        }
        df = build_spread_flow_table(dt.date(2026, 1, 27), cfg, [ctx])
        row = df.iloc[0]
        self.assertTrue(row["spread_pass"])
        self.assertTrue(row["flow_pass"])
        self.assertEqual(row["flow_strength"], "MIXED")


if __name__ == "__main__":
    unittest.main()
