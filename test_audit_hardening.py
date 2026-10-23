"""Regression tests for critical audit fixes; no API keys or network needed."""
import unittest

import numpy as np
import pandas as pd

from ml_models import CentroidClassifier
from features import enrich
from backtest_execution import next_entry_and_path, simulate_long_exit


class AuditFixTests(unittest.TestCase):
    def test_centroid_prediction_invariant_to_future_or_extra_rows(self):
        X = np.array([
            [-2.0, 0.2], [-1.5, -0.5], [-0.8, 0.9], [0.0, -0.2],
            [0.9, -0.9], [1.4, 0.7], [2.1, -0.6], [3.0, 0.3],
        ])
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        model = CentroidClassifier().fit(X, y)
        candidate = np.array([[0.3, 0.2]])
        single = model.predict_proba(candidate)[0]
        extra = np.array([[20.0, -50.0], [100.0, 999.0], [-12.0, 80.0]])
        combined = model.predict_proba(np.vstack([candidate, extra]))[0]
        np.testing.assert_allclose(single, combined, atol=1e-14, rtol=0)

    def test_old_centroid_requires_explicit_retraining(self):
        X = np.array([[-2.], [-1.], [1.], [2.]])
        model = CentroidClassifier().fit(X, np.array([0, 0, 1, 1]))
        del model.margin_scale_
        with self.assertRaisesRegex(RuntimeError, "Retrain"):
            model.predict_proba(np.array([[0.]]))

    def test_news_without_a_matching_story_is_neutral_not_nan(self):
        # The latest historical headline is outside its 30-minute tolerance.
        times = pd.date_range("2026-01-05T14:30:00Z", periods=30, freq="5min")
        price = pd.DataFrame({
            "timestamp": times,
            "open": np.linspace(100.0, 103.0, len(times)),
            "high": np.linspace(100.5, 103.5, len(times)),
            "low": np.linspace(99.5, 102.5, len(times)),
            "close": np.linspace(100.1, 103.1, len(times)),
            "volume": np.repeat(1000., len(times)),
        })
        news = pd.DataFrame({
            "timestamp": [pd.Timestamp("2026-01-05T12:00:00Z")],
            "sentiment": [1.0],
        })
        x = enrich(price, news=news, news_lookback_minutes=30)
        self.assertEqual(len(x), len(price))
        self.assertTrue(x["news_score"].notna().all())
        self.assertTrue((x["news_score"] == 0).all())

    def test_same_symbol_next_bar_not_concurrent_other_symbol(self):
        ts = pd.to_datetime([
            "2026-10-01T13:30:00Z", "2026-10-01T13:35:00Z",
            "2026-10-01T13:40:00Z",
        ])
        aapl = pd.DataFrame({
            "timestamp": ts, "open": [10., 10.5, 10.7],
            "high": [11., 11., 11.], "low": [9., 10., 10.],
            "close": [10.2, 10.6, 10.8],
        })
        nvda = aapl.copy()
        nvda["open"] *= 100
        entry, path = next_entry_and_path(aapl, ts[0])
        self.assertAlmostEqual(float(entry.open), 10.5)
        self.assertEqual(len(path), 2)
        self.assertNotEqual(float(entry.open), float(nvda.iloc[1].open))
        self.assertTrue((path["timestamp"] > ts[0]).all())

    def test_no_overnight_entry(self):
        ts = pd.to_datetime([
            "2026-10-01T19:55:00Z", "2026-10-02T13:30:00Z",
        ])
        bars = pd.DataFrame({
            "timestamp": ts, "open": [20., 22.],
            "high": [21., 23.], "low": [19., 21.],
            "close": [20., 22.],
        })
        entry, path = next_entry_and_path(bars, ts[0])
        self.assertIsNone(entry)
        self.assertTrue(path.empty)

    def test_stop_gap_uses_open_not_optimistic_stop_price(self):
        ts = pd.to_datetime(["2026-10-01T13:35:00Z"])
        bars = pd.DataFrame({
            "timestamp": ts, "open": [95.],
            "high": [110.], "low": [90.], "close": [104.],
        })
        price, reason, _ = simulate_long_exit(
            bars, stop=98., target=108., slippage_fraction=0.001
        )
        self.assertEqual(reason, "stop_gap")
        self.assertAlmostEqual(price, 95. * 0.999)

    def test_both_stop_and_target_in_bar_stop_wins(self):
        bars = pd.DataFrame({
            "timestamp": pd.to_datetime(["2026-10-01T13:35:00Z"]),
            "open": [100.], "high": [120.], "low": [90.], "close": [105.]
        })
        price, reason, _ = simulate_long_exit(
            bars, stop=95., target=115., slippage_fraction=0.0
        )
        self.assertEqual(reason, "stop")
        self.assertAlmostEqual(price, 95.)


if __name__ == "__main__":
    unittest.main()
