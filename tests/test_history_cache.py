import tempfile
import unittest
from pathlib import Path

from cot_bias.sources import _resolve_hist_dir
from cot_bias.utils import Config, DEFAULT_CONTRACT_PATTERNS, DEFAULT_URLS


class TestHistoricalCache(unittest.TestCase):
    def test_resolve_hist_dir_creates_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            cfg = Config(
                data_dir=data_dir,
                hist_dir=data_dir / "historical",
                sqlite_path=data_dir / "cot.sqlite",
                parquet_dir=data_dir / "parquet",
                urls=dict(DEFAULT_URLS),
                contract_patterns=dict(DEFAULT_CONTRACT_PATTERNS),
                fx_group="lev_money",
                metals_group="m_money",
                rolling_weeks=156,
                delta_weeks=4,
            )
            hist_dir = _resolve_hist_dir(cfg)
            self.assertTrue(hist_dir.exists())
            self.assertEqual(hist_dir, data_dir / "historical")
            self.assertEqual(hist_dir / "dummy.zip", data_dir / "historical" / "dummy.zip")


if __name__ == "__main__":
    unittest.main()
