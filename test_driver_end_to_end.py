"""Tiny generated two-symbol end-to-end run: wiring only, NOT market evidence."""
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import v6_5_intraday_replay as driver


def fixture(days, scale=1.):
    frames = []
    for i, date in enumerate(days):
        ts = pd.date_range(f"{date} 09:30", f"{date} 15:55", freq="5min", tz="America/New_York").tz_convert("UTC")
        c = (100 + i*.1 + np.sin(np.arange(len(ts))*.7)) * scale
        frames.append(pd.DataFrame({"timestamp":ts,"open":c,"high":c+.4,"low":c-.4,"close":c,"volume":1000.}))
    return pd.concat(frames, ignore_index=True)


class DriverTests(unittest.TestCase):
    def test_complete_synthetic_run_writes_hashes_daily_and_comparison(self):
        train_days = ["2026-08-20","2026-08-21","2026-08-24","2026-08-25","2026-08-26","2026-08-27","2026-08-28","2026-08-31"]
        test_days = ["2026-09-01","2026-09-02","2026-09-03","2026-09-04","2026-09-08","2026-09-09","2026-09-10","2026-09-11","2026-10-19"]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); old=root/"old"; new=root/"new"; out=root/"run"
            old.mkdir(); new.mkdir()
            for symbol,scale in [("NVDA",1.),("AMD",2.),("SPY",3.)]:
                fixture(train_days,scale).to_csv(old/f"{symbol}_5min.csv",index=False)
                fixture(test_days,scale).to_csv(new/f"{symbol}_5min.csv",index=False)
            sessions=root/"sessions.csv"
            pd.DataFrame({"date":test_days,"close":"16:00"}).to_csv(sessions,index=False)
            args=["v6_5_intraday_replay.py","--data-dir",str(old),"--extension-dir",str(new),"--sessions",str(sessions),"--output-dir",str(out)]
            with patch.object(driver,"SYMBOLS",["NVDA","AMD"]), patch.object(driver,"CFG",dict(driver.CFG,symbols=["NVDA","AMD"])), patch.object(driver,"DATA_DIR",old), patch.object(driver,"EXT_DIR",new), patch("sys.argv",args), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                driver.main()
                with self.assertRaises(SystemExit):
                    driver.main()  # existing outputs must not be overwritten
            manifest=json.loads((out/"manifest.json").read_text())
            self.assertEqual(manifest["status"],"completed_development_comparison")
            self.assertEqual(len(manifest["input_sha256"]),7)
            self.assertFalse(manifest["profitability_validated"])
            daily=pd.read_csv(out/"v6_5_intraday_daily.csv")
            self.assertEqual(len(daily),2*len(test_days))
            comparison=json.loads((out/"spy_comparison.json").read_text())
            self.assertFalse(comparison["independent_evidence"])
            self.assertEqual(comparison["sessions"],len(test_days))

    def test_missing_input_fails_before_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/"run"
            args=["v6_5_intraday_replay.py","--data-dir",tmp,"--extension-dir",tmp,"--sessions",str(Path(tmp)/"missing.csv"),"--output-dir",str(out)]
            with patch("sys.argv",args), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                driver.main()
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
