"""Numerical/behavior parity and failure tests, using synthetic inputs only."""
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

import disk_training as disk
from features import FEATURES
from ml_models import LogisticGD
from model_pipeline import fit_bundle, predict_bundle
from tools import run_august_simulation as runner
from tools.run_offline_simulation import synthetic_bars


def training_frame(n=257, seed=5):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    frame[FEATURES[2]] = 0.  # Constant feature scale must match the reference.
    frame["target"] = ((frame[FEATURES[0]]+.3*frame[FEATURES[1]]) > 0).astype(float)
    frame["timestamp"] = pd.date_range("2026-07-01", periods=n, freq="5min", tz="UTC")
    return frame


class DiskTrainingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def store(self, frame, name="store"):
        store = disk.TrainingStore(self.root/name)
        store.append(frame)
        return store.finish()

    def test_stable_external_merge_preserves_every_value_and_tie_order(self):
        a, b = training_frame(51), training_frame(51, 9)
        store = disk.TrainingStore(self.root/"merge")
        store.append(a); store.append(b); store.finish()
        expected = pd.concat([a, b], ignore_index=True).sort_values("timestamp", kind="stable")
        actual = np.concatenate(list(disk.blocks(store.path, disk.RECORD, 7)))
        np.testing.assert_array_equal(actual["x"], expected[FEATURES].to_numpy())
        np.testing.assert_array_equal(actual["y"], expected.target.to_numpy())
        np.testing.assert_array_equal(actual["timestamp"], expected.timestamp.dt.as_unit("ns").astype("int64"))
        self.assertEqual(len(actual), 102)
        self.assertFalse(list(store.root.glob("shard-*")))
        with self.assertRaises(FileExistsError): store.finish()

    def test_full_fixed_bundle_matches_reference_predictions_and_bootstraps(self):
        frame = training_frame()
        store = self.store(frame)
        cfg = runner.reference.CFG
        reference = fit_bundle(frame, cfg, "neutral")
        streamed = disk.fit_disk_bundle(store, cfg, "neutral", block_rows=37, progress=lambda s:None)
        test = training_frame(113, 10)
        np.testing.assert_allclose(predict_bundle(reference, test, cfg),
                                   predict_bundle(streamed, test, cfg), rtol=0, atol=1e-11)
        for (ac, am), (bc, bm) in zip(reference["models"][1].models_, streamed["models"][1].models_):
            np.testing.assert_array_equal(ac, bc)
            np.testing.assert_allclose(am.scaler_.mean_, bm.scaler_.mean_, atol=1e-12)
            np.testing.assert_allclose(am.scaler_.std_, bm.scaler_.std_, atol=1e-12)
            np.testing.assert_allclose(am.w_, bm.w_, atol=1e-11)
            self.assertEqual(am.epochs, bm.epochs)
        self.assertEqual(len(streamed["models"][1].models_), 25)
        self.assertEqual(streamed["models"][0].epochs, 500)
        self.assertFalse((store.root/"standardized.bin").exists())

    def test_weighted_full_batch_updates_match_materialized_bootstrap(self):
        frame = training_frame(83)
        store = self.store(frame)
        rng = np.random.default_rng(42)
        idx = rng.integers(0, len(frame), len(frame))
        cols = rng.choice(len(FEATURES), 8, replace=False)
        counts = disk.bootstrap_counts(np.random.default_rng(42), len(frame))
        np.testing.assert_array_equal(counts, np.bincount(idx, minlength=len(frame)))
        a = LogisticGD(lr=.02, epochs=7, l2=.03).fit(frame[FEATURES].to_numpy()[idx][:, cols], frame.target.to_numpy()[idx])
        b = disk.fit_logistic(store.path, LogisticGD(lr=.02, epochs=7, l2=.03),
                              store.root, cols, counts, block_rows=6)
        np.testing.assert_allclose(a.w_, b.w_, rtol=0, atol=1e-12)
        self.assertAlmostEqual(a.b_, b.b_, places=12)

    def test_invalid_training_and_truncated_store_fail_closed(self):
        frame = training_frame(33)
        bad = frame.copy(); bad.iloc[0, 0] = np.nan
        store = disk.TrainingStore(self.root/"bad")
        with self.assertRaises(ValueError): store.append(bad)
        with self.assertRaises(ValueError): store.append(frame.iloc[::-1])
        one = frame.copy(); one["target"] = 1.
        store = self.store(one, "one")
        with self.assertRaisesRegex(ValueError, "Both binary"):
            disk.fit_disk_bundle(store, runner.reference.CFG, "neutral")
        with store.path.open("ab") as f: f.write(b"x")
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            disk.fit_disk_bundle(store, runner.reference.CFG, "neutral")

    def test_cleanup_preserves_unknown_files_and_reports_retained_directory(self):
        store = self.store(training_frame(33))
        extra = store.root/"keep.txt"
        extra.write_text("not owned by the trainer")
        self.assertFalse(store.cleanup())
        self.assertEqual(extra.read_text(), "not owned by the trainer")
        self.assertFalse(store.path.exists())

    def test_interrupt_removes_only_own_standardized_cache(self):
        store = self.store(training_frame(43))
        before = runner.digest(store.path)
        def stop(message): raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            disk.fit_logistic(store.path, LogisticGD(epochs=25), store.root, progress=stop)
        self.assertFalse((store.root/"standardized.bin").exists())
        self.assertEqual(before, runner.digest(store.path))
        cache = store.root/"standardized.bin"; cache.write_bytes(b"existing")
        with self.assertRaises(FileExistsError):
            disk.fit_logistic(store.path, LogisticGD(epochs=1), store.root)
        self.assertEqual(cache.read_bytes(), b"existing")

    def test_session_replay_matches_full_month_with_compounding_and_zero_days(self):
        days = ["2026-08-03", "2026-08-04", "2026-08-05"]
        prices = {s: synthetic_bars(days, scale) for s, scale in [("NVDA", 1.), ("AMD", 2.)]}
        scored = {s: x.assign(ml_prob=.9, trade_score=100., atr=.3, news_score=0.) for s,x in prices.items()}
        cfg = dict(runner.reference.CFG)
        kwargs = dict(spy_opening_gate=dict(zip(days, [True, False, True])),
                      session_closes=dict.fromkeys(days, "16:00"))
        for gate in [False, True]:
            a = runner.replay_intraday(scored, prices, list(prices), {}, cfg, require_spy=gate, **kwargs)
            b = runner.replay_by_session(scored, prices, list(prices), {}, cfg, require_spy=gate, **kwargs)
            self.assertEqual(a[0], b[0])
            for av, bv in zip(a[1:], b[1:]): assert_frame_equal(av, bv)
        self.assertEqual(b[2].iloc[1].entries, 0)

    def historical_fixture(self, stack):
        data = self.root/"source"; data.mkdir()
        stack.enter_context(patch.object(runner.reference, "SYMBOLS", ["NVDA", "AMD"]))
        stack.enter_context(patch.object(runner.reference, "CFG", dict(runner.reference.CFG, symbols=["NVDA", "AMD"])))
        stack.enter_context(redirect_stdout(io.StringIO()))
        days = ["2026-07-22", "2026-07-23", "2026-07-24", "2026-07-27", "2026-07-28", "2026-07-29", "2026-07-30", "2026-07-31"]
        for s, scale in [("NVDA", 1.), ("AMD", 2.), ("SPY", 3.)]:
            synthetic_bars(days + list(runner.load_sessions()), scale).to_csv(data/f"{s}_5min.csv", index=False)
        return data

    def test_low_memory_full_run_matches_reference_outputs(self):
        with ExitStack() as stack:
            data = self.historical_fixture(stack)
            a = runner.run(data, self.root/"ram", execute=True)
            b = runner.run(data, self.root/"disk", execute=True, low_memory=True)
        self.assertEqual(a["training_rows"], b["training_rows"])
        self.assertEqual(b["status"], "completed_development_comparison")
        self.assertTrue(b["source_hashes_unchanged_at_end"])
        self.assertFalse((self.root/"disk/training_cache").exists())
        for name in ["summary", "trades", "daily", "equity"]:
            av, bv = [pd.read_csv(self.root/p/f"{name}.csv") for p in ["ram", "disk"]]
            assert_frame_equal(av, bv, atol=1e-9, rtol=1e-10)

    def test_low_memory_held_gap_still_invalidates_result(self):
        with ExitStack() as stack:
            data = self.historical_fixture(stack)
            p = data/"NVDA_5min.csv"; x = pd.read_csv(p)
            x.loc[pd.to_datetime(x.timestamp, utc=True) != pd.Timestamp("2026-08-03T13:40Z")].to_csv(p, index=False)
            stack.enter_context(patch.object(runner, "fit_disk_bundle", return_value={"feature_contract": {}}))
            stack.enter_context(patch.object(runner, "predict_bundle", side_effect=lambda b,x,c:np.full(len(x), .9)))
            stack.enter_context(patch.object(runner, "trade_score", side_effect=lambda p,*a:np.full(len(p), 100.)))
            with self.assertRaisesRegex(ValueError, "Unobservable held position"):
                runner.run(data, self.root/"gap", execute=True, low_memory=True)
        self.assertFalse((self.root/"gap/summary.csv").exists())
        report = json.loads((self.root/"gap/run_report.json").read_text())
        self.assertEqual(report["status"], "failed_not_validated")

    def test_insufficient_disk_stops_before_training(self):
        with ExitStack() as stack:
            data = self.historical_fixture(stack)
            stack.enter_context(patch.object(runner.shutil, "disk_usage", return_value=SimpleNamespace(free=0)))
            fit = stack.enter_context(patch.object(runner, "fit_disk_bundle"))
            with self.assertRaisesRegex(ValueError, "free disk"):
                runner.run(data, self.root/"no-space", execute=True, low_memory=True)
            fit.assert_not_called()
        self.assertFalse((self.root/"no-space/training_cache").exists())


if __name__ == "__main__": unittest.main()
