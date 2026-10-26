"""User-facing demo integrity: label synthetic evidence and restore driver state."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import v6_5_intraday_replay as driver
from tools.run_offline_simulation import run_demo


class OfflineSimulationTests(unittest.TestCase):
    def test_demo_is_labeled_and_restores_production_globals(self):
        original=(driver.CFG, driver.SYMBOLS, driver.DATA_DIR, driver.EXT_DIR)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"demo"
            result=run_demo(root)
            self.assertEqual(result["status"], "completed")
            self.assertFalse(result["profitability_validated"])
            self.assertFalse(result["is_100_stock_test"])
            self.assertEqual(result["daily_rows"],18)
            manifest=json.loads((root/"replay/manifest.json").read_text())
            self.assertEqual(manifest["data_kind"],"synthetic")
            self.assertEqual(manifest["config"]["probability_threshold"],driver.CFG["probability_threshold"])
            for name, digest in manifest["outputs"].items():
                self.assertEqual(hashlib.sha256((root/"replay"/name).read_bytes()).hexdigest(),digest)
            comparison=json.loads((root/"replay/spy_comparison.json").read_text())
            self.assertEqual(comparison["classification"],"synthetic_demonstration_only")
        for before,after in zip(original,(driver.CFG,driver.SYMBOLS,driver.DATA_DIR,driver.EXT_DIR)):
            self.assertIs(before,after)

    def test_existing_run_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            sentinel=Path(tmp)/"sentinel.txt";sentinel.write_text("keep")
            with self.assertRaises(FileExistsError):run_demo(tmp)
            self.assertEqual(sentinel.read_text(),"keep")

    def test_failed_driver_leaves_failed_report_and_restores_config(self):
        original=driver.CFG
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"demo"
            with patch.object(driver,"main",side_effect=ValueError("injected failure")):
                with self.assertRaises(ValueError):run_demo(root)
            self.assertEqual(json.loads((root/"run_report.json").read_text())["status"],"failed")
        self.assertIs(driver.CFG,original)


if __name__=="__main__":unittest.main()
