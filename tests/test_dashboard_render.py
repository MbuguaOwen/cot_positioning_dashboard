import tempfile
import unittest
from pathlib import Path

import pandas as pd

from cot_bias.dashboard.render import render_dashboard


class TestDashboardRender(unittest.TestCase):
    def test_gate_section_renders_empty_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            latest_df = pd.DataFrame([{"symbol": "EUR", "bias": "NEUTRAL"}])
            gate_df = pd.DataFrame(
                columns=[
                    "pair",
                    "base",
                    "quote",
                    "base_bias",
                    "base_strength",
                    "quote_bias",
                    "quote_strength",
                    "pair_bias",
                    "pair_strength",
                    "reversal_risk",
                    "reversal_risk_extreme",
                    "passes_2_reports",
                    "latest_release_date",
                    "previous_release_date",
                ]
            )

            html_path = render_dashboard(
                out_dir=out_dir,
                as_of="2026-01-20",
                max_report_date="2026-01-20",
                warnings=[],
                latest_df=latest_df,
                history_df=pd.DataFrame(),
                pairs_df=None,
                gate_df=gate_df,
            )

            html = html_path.read_text(encoding="utf-8")
            self.assertIn("FX Pairs (passed COT filters)", html)
            self.assertIn("No FX pairs passed the current COT filter set.", html)


if __name__ == "__main__":
    unittest.main()
