"""Synthetic fixtures verify wiring/failure policy, never historical profits."""
from contextlib import ExitStack, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from tools import run_august_simulation as runner
from tools.run_offline_simulation import synthetic_bars


TRAIN = ["2026-07-22", "2026-07-23", "2026-07-24", "2026-07-27",
         "2026-07-28", "2026-07-29", "2026-07-30", "2026-07-31"]


class AugustSimulationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / "source"
        self.data.mkdir()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(runner.reference, "SYMBOLS", ["NVDA", "AMD"]))
        self.stack.enter_context(patch.object(runner.reference, "CFG", dict(runner.reference.CFG, symbols=["NVDA", "AMD"])))
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        for s, scale in [("NVDA", 1.), ("AMD", 2.), ("SPY", 3.)]:
            synthetic_bars(TRAIN + list(runner.load_sessions()), scale).to_csv(self.data/f"{s}_5min.csv", index=False)

    def remove(self, symbol, stamp):
        p = self.data/f"{symbol}_5min.csv"
        x = pd.read_csv(p)
        x.loc[pd.to_datetime(x.timestamp, utc=True) != pd.Timestamp(stamp)].to_csv(p, index=False)

    def test_preflight_keeps_gaps_sources_and_config_without_training(self):
        self.remove("AMD", "2026-08-11T18:55Z")
        hashes = {p.name:runner.digest(p) for p in self.data.iterdir()}
        config = dict(runner.reference.CFG)
        with patch.object(runner, "fit_bundle", side_effect=AssertionError("No fitting in preflight")):
            result = runner.run(self.data, self.root/"check")
        self.assertEqual(result["status"], "preflight_complete_no_backtest")
        self.assertEqual(result["sessions"], 21)
        row = next(r for r in result["coverage"] if r["symbol"] == "AMD")
        self.assertEqual(row["missing_bars"], ["2026-08-11T18:55:00+00:00"])
        self.assertEqual(row["expected_regular_bars"], 1638)
        self.assertTrue(result["source_hashes_unchanged_at_end"])
        self.assertEqual(hashes, {p.name:runner.digest(p) for p in self.data.iterdir()})
        self.assertEqual(config, runner.reference.CFG)
        manifest = json.loads((self.root/"check/manifest.json").read_text())
        self.assertEqual(manifest["config"]["feed"], "sip")
        self.assertFalse((self.root/"check/summary.csv").exists())

    def test_benchmark_gap_or_missing_session_stops_before_training(self):
        self.remove("SPY", "2026-08-03T13:30Z")
        with patch.object(runner, "fit_bundle") as fit:
            with self.assertRaisesRegex(ValueError, "Benchmark"):
                runner.run(self.data, self.root/"bad-spy", execute=True)
            fit.assert_not_called()
        p = self.data/"NVDA_5min.csv"
        x = pd.read_csv(p)
        x.loc[~x.timestamp.str.startswith("2026-08-04")].to_csv(p, index=False)
        with self.assertRaisesRegex(ValueError, "missing/unexpected"):
            runner.run(self.data, self.root/"bad-day")
        self.assertEqual(json.loads((self.root/"bad-day/run_report.json").read_text())["status"], "failed_not_validated")

    def test_future_prices_do_not_enter_training_or_earlier_features(self):
        market, _ = runner.read_source(self.data/"SPY_5min.csv")
        raw, _ = runner.read_source(self.data/"NVDA_5min.csv")
        train, test = runner.prepare(raw, market, runner.reference.CFG)
        changed = raw.copy()
        mask = changed.timestamp >= pd.Timestamp("2026-08-04", tz=runner.NY)
        changed.loc[mask, ["open", "high", "low", "close"]] *= 2
        train2, test2 = runner.prepare(changed, market, runner.reference.CFG)
        assert_frame_equal(train, train2)
        # Labels themselves can change with future prices; inference features cannot.
        cols = ["timestamp", *runner.FEATURES]
        before = test.timestamp < pd.Timestamp("2026-08-04", tz=runner.NY)
        assert_frame_equal(test.loc[before, cols], test2.loc[before, cols])
        self.assertLess(train.timestamp.max(), runner.CUTOFF)
        with_future = pd.concat([raw, synthetic_bars(["2026-09-01"], 1.)], ignore_index=True)
        with_future.to_csv(self.data/"NVDA_5min.csv", index=False)
        reread, _ = runner.read_source(self.data/"NVDA_5min.csv")
        assert_frame_equal(raw, reread)

    def test_real_ensemble_and_replay_keep_all_sessions_and_hashes(self):
        result = runner.run(self.data, self.root/"complete", execute=True)
        self.assertEqual(result["status"], "completed_development_comparison")
        self.assertFalse(result["independent_evidence"])
        self.assertFalse(result["profitability_validated"])
        daily = pd.read_csv(self.root/"complete/daily.csv")
        self.assertEqual(len(daily), 42)
        for _, group in daily.groupby("strategy"):
            self.assertEqual(set(group.date), set(runner.load_sessions()))
        manifest = json.loads((self.root/"complete/manifest.json").read_text())
        for name, sha in manifest["outputs"].items():
            self.assertEqual(runner.digest(self.root/"complete"/name), sha)
        comparison = json.loads((self.root/"complete/spy_comparison.json").read_text())
        self.assertEqual(comparison["sessions"], 21)
        self.assertEqual(comparison["gate_allowed_sessions"], 0)

    def force_signals(self):
        self.stack.enter_context(patch.object(runner, "fit_bundle", return_value={"feature_contract": {}}))
        self.stack.enter_context(patch.object(runner, "predict_bundle", side_effect=lambda b,x,c:np.full(len(x), .9)))
        self.stack.enter_context(patch.object(runner, "trade_score", side_effect=lambda p,*a:np.full(len(p), 100.)))

    def test_missing_held_bar_invalidates_whole_comparison_without_fill(self):
        self.remove("NVDA", "2026-08-03T13:40Z")
        self.force_signals()
        with self.assertRaisesRegex(ValueError, "Unobservable held position NVDA"):
            runner.run(self.data, self.root/"gap", execute=True)
        self.assertFalse((self.root/"gap/summary.csv").exists())
        self.assertFalse((self.root/"gap/spy_comparison.json").exists())
        report = json.loads((self.root/"gap/run_report.json").read_text())
        self.assertEqual(report["status"], "failed_not_validated")
        manifest = json.loads((self.root/"gap/manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed_not_validated")

    def test_source_mutation_during_training_cannot_produce_valid_result(self):
        self.force_signals()
        def mutate(*args, **kwargs):
            p = self.data/"AMD_5min.csv"
            p.write_bytes(p.read_bytes()+b"\n")
            return {"feature_contract": {}}
        with patch.object(runner, "fit_bundle", side_effect=mutate):
            with self.assertRaisesRegex(ValueError, "Input changed"):
                runner.run(self.data, self.root/"mutation", execute=True)
        self.assertFalse((self.root/"mutation/summary.csv").exists())

    def test_network_is_blocked_during_training(self):
        def attempt(*args, **kwargs):
            with socket.socket() as sock:
                sock.connect(("127.0.0.1", 443))
        with patch.object(runner, "fit_bundle", side_effect=attempt):
            with self.assertRaisesRegex(RuntimeError, "Network forbidden"):
                runner.run(self.data, self.root/"network", execute=True)

    def test_no_overwrite_missing_inputs_and_calendar_tampering(self):
        with self.assertRaises(ValueError):
            runner.run(self.data, self.data/"nested")
        self.root.joinpath("existing").mkdir()
        with self.assertRaises(FileExistsError):
            runner.run(self.data, self.root/"existing")
        (self.data/"AMD_5min.csv").unlink()
        with self.assertRaises(FileNotFoundError):
            runner.run(self.data, self.root/"missing")
        self.assertFalse((self.root/"missing").exists())
        calendar = self.root/"calendar.csv"
        pd.DataFrame({"date": ["2026-08-03"], "close": ["16:00"]}).to_csv(calendar, index=False)
        with patch.object(runner, "CALENDAR", calendar):
            with self.assertRaisesRegex(ValueError, "calendar"):
                runner.load_sessions()


if __name__ == "__main__":
    unittest.main()
