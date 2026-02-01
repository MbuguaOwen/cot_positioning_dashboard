import datetime as dt
import unittest
from unittest import mock

import pandas as pd

import cot_bias.cli as cli


class TestCacheRefresh(unittest.TestCase):
    def test_force_refresh_calls_update(self) -> None:
        args = mock.Mock()
        args.config = None
        args.as_of = "2026-01-20"
        args.out = "outputs"
        args.force_refresh = True
        args.verbose = False

        df = pd.DataFrame(
            {
                "symbol": ["EUR"],
                "report_date": [dt.date(2026, 1, 13)],
                "net_pct_oi": [1.0],
            }
        )

        with mock.patch.object(cli, "update_store") as upd, \
             mock.patch.object(cli, "load_store", return_value=df) as ld, \
             mock.patch.object(cli, "render_dashboard") as rd, \
             mock.patch.object(cli, "write_manifest") as wm:
            rd.return_value = "outputs/dashboard.html"
            cli.cmd_dashboard(args)
            self.assertTrue(upd.called)
            self.assertTrue(ld.called)
            self.assertTrue(rd.called)
            self.assertTrue(wm.called)

    def test_refresh_failure_warns(self) -> None:
        args = mock.Mock()
        args.config = None
        args.as_of = "2026-01-20"
        args.out = "outputs"
        args.force_refresh = False
        args.verbose = False

        df = pd.DataFrame(
            {
                "symbol": ["EUR"],
                "report_date": [dt.date(2026, 1, 6)],
                "net_pct_oi": [1.0],
            }
        )

        captured = {}

        def _render(**kwargs):
            captured["warnings"] = kwargs.get("warnings", [])
            return "outputs/dashboard.html"

        with mock.patch.object(cli, "update_store", side_effect=RuntimeError("offline")) as upd, \
             mock.patch.object(cli, "load_store", return_value=df) as ld, \
             mock.patch.object(cli, "render_dashboard", side_effect=_render) as rd, \
             mock.patch.object(cli, "write_manifest") as wm:
            cli.cmd_dashboard(args)
            warns = captured.get("warnings", [])
            self.assertTrue(any("AUTO-REFRESH FAILED" in w for w in warns))
            self.assertTrue(any("DATA STALE" in w for w in warns))


if __name__ == "__main__":
    unittest.main()
