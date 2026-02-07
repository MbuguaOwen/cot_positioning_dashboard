import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from cot_bias.dashboard.render_updated import render_dashboard_updated


class TestDashboardRenderUpdated(unittest.TestCase):
    def test_updated_layout_headings_and_bias_hint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)

            instruments = pd.DataFrame(
                [
                    {
                        "symbol": "EUR",
                        "report_date": "2026-01-20",
                        "z_3y": 1.2,
                        "pctile_3y": 60.0,
                        "bias": "BULLISH",
                        "reversal_risk": "LOW",
                        "score": 1.8,
                        "net_pct_oi": 2.1,
                    }
                ]
            )
            pairs = pd.DataFrame(
                [
                    {
                        "pair": "EURJPY",
                        "pair_z": 1.5,
                        "pair_bias": "BULLISH",
                        "pair_strength": "HIGH",
                        "base": "EUR",
                        "quote": "JPY",
                        "base_z": 1.2,
                        "quote_z": -0.3,
                        "method": "basket",
                    }
                ]
            )
            decisions = pd.DataFrame(
                [
                    {
                        "week_end": "2026-01-20",
                        "tier_mode": "BALANCED",
                        "pair": "EURJPY",
                        "direction_allowed": "LONG",
                        "confidence_score": 55.0,
                        "spread": 1.2,
                        "dSpread_1w": 0.3,
                        "crowded_flag": "NONE",
                        "macro_ok": True,
                        "news_ok": True,
                        "allow": True,
                    }
                ]
            )

            instruments.to_csv(out_dir / "instruments_latest.csv", index=False)
            pairs.to_csv(out_dir / "pairs_latest.csv", index=False)
            decisions.to_csv(out_dir / "decisions_summary.csv", index=False)
            pd.DataFrame().to_csv(out_dir / "spread_flow.csv", index=False)
            pd.DataFrame().to_csv(out_dir / "crowdedness_policy.csv", index=False)
            pd.DataFrame().to_csv(out_dir / "macro_gate.csv", index=False)
            pd.DataFrame().to_csv(out_dir / "pine_join.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "pair": "EURJPY",
                        "confidence_score": 55.0,
                        "spread": 1.2,
                        "dSpread_1w": 0.3,
                        "crowded_flag": "NONE",
                        "prev_spread": 0.9,
                        "prev_dSpread_1w": 0.2,
                        "prev_crowded_flag": "NONE",
                    }
                ]
            ).to_csv(out_dir / "persistent_bullish_2w.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "pair": "EURJPY",
                        "confidence_score": 55.0,
                        "spread": 1.2,
                        "dSpread_1w": 0.3,
                        "crowded_flag": "NONE",
                        "prev1_spread": 0.9,
                        "prev2_spread": 0.8,
                    }
                ]
            ).to_csv(out_dir / "strong_persistent_bullish_3w.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "pair": "EURJPY",
                        "direction_allowed": "LONG",
                        "quality_score": 75.0,
                        "quality_tier": "ELITE",
                        "spread": 1.2,
                        "dSpread_1w": 0.3,
                        "crowded_flag": "NONE",
                        "prev1_spread": 0.9,
                        "prev2_spread": 0.8,
                        "gap_z": 2.8,
                        "A_release_ok": True,
                        "A_consecutive_ok": True,
                        "A_sign_ok": True,
                        "B_mode": "B3_Z_GAP",
                        "B_strength_ok": True,
                        "B_strength_points": 20,
                        "C_crowding_policy": "SOFT_PENALTY",
                        "C_crowding_ok": True,
                        "C_crowding_points": 20,
                        "C_base_reversal_risk": "LOW",
                        "C_quote_reversal_risk": "LOW",
                        "C_reversal_points": 0,
                        "D_no_collapse_ok": True,
                        "D_points": 15,
                    }
                ]
            ).to_csv(out_dir / "quality_gates_score.csv", index=False)
            (out_dir / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "as_of": "2026-01-27",
                        "store_max_report_date": "2026-01-20",
                        "git_commit": "abc123",
                        "previous_report_date": "2026-01-13",
                        "previous2_report_date": "2026-01-06",
                        "resolved_report_date": "2026-01-20",
                        "resolved_release_dt": "2026-01-23T20:30:00Z",
                    }
                ),
                encoding="utf-8",
            )

            html_path = render_dashboard_updated(out_dir, out_dir / "dashboard.html")
            html = html_path.read_text(encoding="utf-8")

            self.assertIn("1) Instrument Snapshot", html)
            self.assertIn("Strength-First Shortlist (pair_z)", html)
            self.assertIn("2) Actionable List", html)
            self.assertIn("2d) Quality Gates Score (Scored Filter", html)
            self.assertIn("2a) Bullish Bias Focus (LONGs)", html)
            self.assertIn("2c) Strong Persistence (3 consecutive reports", html)
            self.assertIn("2b) Persistent Bullish Bias (Current + Previous report)", html)
            self.assertIn("3) Context", html)
            self.assertIn("bias_hint", html)
            self.assertIn("prev1_spread", html)
            self.assertIn("prev2_spread", html)
            self.assertIn("ELITE count:", html)
            self.assertIn("Commit: abc123", html)
            self.assertNotIn("FX Pairs (passed COT filters)", html)

    def test_strong_persistence_unavailable_without_three_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            pd.DataFrame([{"symbol": "EUR", "report_date": "2026-01-20"}]).to_csv(
                out_dir / "instruments_latest.csv", index=False
            )
            pd.DataFrame([{"pair": "EURJPY", "pair_z": 1.5}]).to_csv(
                out_dir / "pairs_latest.csv", index=False
            )
            pd.DataFrame(
                [
                    {
                        "pair": "EURJPY",
                        "direction_allowed": "LONG",
                        "confidence_score": 55.0,
                        "spread": 1.2,
                        "dSpread_1w": 0.3,
                        "crowded_flag": "NONE",
                        "allow": True,
                    }
                ]
            ).to_csv(out_dir / "decisions_summary.csv", index=False)
            pd.DataFrame().to_csv(out_dir / "spread_flow.csv", index=False)
            pd.DataFrame().to_csv(out_dir / "crowdedness_policy.csv", index=False)
            pd.DataFrame().to_csv(out_dir / "macro_gate.csv", index=False)
            pd.DataFrame().to_csv(out_dir / "pine_join.csv", index=False)
            pd.DataFrame().to_csv(out_dir / "persistent_bullish_2w.csv", index=False)
            (out_dir / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "as_of": "2026-01-27",
                        "store_max_report_date": "2026-01-20",
                        "resolved_report_date": "2026-01-20",
                    }
                ),
                encoding="utf-8",
            )

            html_path = render_dashboard_updated(out_dir, out_dir / "dashboard.html")
            html = html_path.read_text(encoding="utf-8")
            self.assertIn("2d) Quality Gates Score (Scored Filter", html)
            self.assertIn("Quality score unavailable (need 3 reports).", html)
            self.assertIn("2c) Strong Persistence (3 consecutive reports", html)
            self.assertIn("Strong persistence unavailable (need 3 reports).", html)


if __name__ == "__main__":
    unittest.main()
