"""Offline causality, feature parity and news event tests (no broker import)."""
import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from features import FEATURES, enrich, label, completed_bars
from news_features import news_scores
from model_pipeline import (fit_bundle, latest_signal, predict_bundle, training_rows,
                            validate_bundle, feature_contract, trade_score)
import v6_5_intraday_replay as replay_driver

CFG = json.loads(Path(__file__).with_name("config.json").read_text())


def price_frame(n=400):
    rng = np.random.default_rng(101)
    c = 100 + rng.normal(0, .2, n).cumsum()
    return pd.DataFrame({"timestamp": pd.date_range("2026-10-01 08:00", periods=n, freq="5min", tz="UTC"),
                         "open": c, "high": c+.3, "low": c-.3, "close": c,
                         "volume": rng.integers(100, 1000, n)})


class FeatureContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = price_frame()
        cls.frame = label(enrich(cls.raw, cls.raw))
        training = cls.frame.dropna(subset=FEATURES+["target"])
        cls.bundle = fit_bundle(training, CFG, "neutral")

    def test_full_history_equals_last_250_bar_inference(self):
        cutoff = self.raw.timestamp.iloc[-1] + pd.Timedelta(minutes=5)
        probability, row = latest_signal(self.raw, self.raw, None, self.bundle, CFG, cutoff)
        np.testing.assert_allclose(row[FEATURES].to_numpy(float), self.frame.iloc[-1][FEATURES].to_numpy(float), atol=1e-12)
        expected = predict_bundle(self.bundle, self.frame.iloc[[-1]], CFG)[0]
        self.assertAlmostEqual(probability, expected, places=12)

    def test_ensemble_single_batch_and_future_rows_equal(self):
        x = self.frame.dropna(subset=FEATURES)
        whole = predict_bundle(self.bundle, x, CFG)
        single = predict_bundle(self.bundle, x.iloc[[0]], CFG)[0]
        self.assertAlmostEqual(whole[0], single, places=12)
        np.testing.assert_allclose(whole[:10], predict_bundle(self.bundle, x.iloc[:10], CFG), atol=1e-12)

    def test_future_stock_and_market_do_not_change_past_features(self):
        cutoff = 330
        old = enrich(self.raw.iloc[:cutoff], self.raw.iloc[:cutoff])
        new = enrich(self.raw, self.raw)
        pd.testing.assert_frame_equal(old, new.iloc[:cutoff].reset_index(drop=True))

    def test_benchmark_features_independent_of_missing_stock_bars(self):
        sparse = self.raw.drop(index=[270, 280, 290])
        x = enrich(sparse, self.raw)
        y = enrich(self.raw, self.raw).set_index("timestamp")
        cols = [c for c in FEATURES if c.startswith("market_")]
        np.testing.assert_allclose(x.iloc[-1][cols].to_numpy(float), y.loc[x.timestamp.iloc[-1], cols].to_numpy(float))

    def test_missing_latest_benchmark_fails_closed(self):
        cutoff = self.raw.timestamp.iloc[-1] + pd.Timedelta(minutes=5)
        self.assertEqual(latest_signal(self.raw, self.raw.iloc[:-1], None, self.bundle, CFG, cutoff), (None, None))

    def test_incomplete_bar_never_used_and_stale_bar_rejected(self):
        cutoff = self.raw.timestamp.iloc[-1] + pd.Timedelta(minutes=2)
        clean = completed_bars(self.raw, cutoff)
        self.assertEqual(clean.timestamp.iloc[-1], self.raw.timestamp.iloc[-2])
        prob, row = latest_signal(self.raw, self.raw, None, self.bundle, CFG, cutoff)
        self.assertIsNotNone(prob)
        self.assertEqual(row.timestamp, self.raw.timestamp.iloc[-2])
        self.assertEqual(latest_signal(self.raw, self.raw, None, self.bundle, CFG, cutoff+pd.Timedelta(minutes=15)), (None, None))

    def test_invalid_latest_features_do_not_fall_back(self):
        raw = self.raw.copy()
        raw.loc[raw.index[-20:], "volume"] = 0
        cutoff = raw.timestamp.iloc[-1] + pd.Timedelta(minutes=5)
        self.assertEqual(latest_signal(raw, self.raw, None, self.bundle, CFG, cutoff), (None, None))

    def test_old_or_changed_contract_rejected(self):
        with self.assertRaisesRegex(ValueError, "Retrain"):
            validate_bundle({"models": []}, CFG)
        with self.assertRaisesRegex(ValueError, "Retrain"):
            validate_bundle(self.bundle, dict(CFG, news_decay_minutes=60))
        with self.assertRaisesRegex(ValueError, "Publication-only"):
            validate_bundle(dict(self.bundle, feature_contract=feature_contract(CFG, "publication_proxy")), CFG, live=True)

    def test_global_time_cutoff_purges_labels(self):
        cutoff = self.raw.timestamp.iloc[350]
        tr = training_rows(self.frame, self.raw.timestamp.iloc[0], cutoff)
        self.assertTrue((tr.label_end <= cutoff).all())
        self.assertTrue((tr.timestamp < cutoff).all())
        changed = self.raw.copy()
        changed.loc[changed.timestamp >= cutoff, ["open", "high", "low", "close"]] *= 2
        new = training_rows(label(enrich(changed, changed)), self.raw.timestamp.iloc[0], cutoff)
        pd.testing.assert_frame_equal(tr, new)

    def test_label_gap_and_multiple_symbols_rejected(self):
        x = self.raw.iloc[:8].drop(index=2).reset_index(drop=True)
        self.assertTrue(pd.isna(label(x).iloc[0].target))
        x["symbol"] = ["A"]*3+["B"]*4
        with self.assertRaisesRegex(ValueError, "one symbol"):
            label(x)

    def test_inference_rows_do_not_depend_on_target_availability(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "NVDA_5min.csv").touch()
            with patch.object(replay_driver, "DATA_DIR", Path(tmp)), patch.object(replay_driver, "read_extended_csv", return_value=self.raw):
                x = replay_driver.load_symbol("NVDA", self.raw)
        self.assertEqual(x.timestamp.iloc[-1], self.raw.timestamp.iloc[-1])
        self.assertTrue(pd.isna(x.target.iloc[-1]))

    def test_rsi_flat_and_rising_are_finite(self):
        for values, expected in [(np.ones(400)*100, 50), (np.arange(400)+100., 100)]:
            x = self.raw.copy()
            for name in ["open", "close"]:
                x[name] = values
            x["high"], x["low"] = values+.5, values-.5
            self.assertEqual(enrich(x).rsi14.iloc[-1], expected)

    def test_score_parity(self):
        node = next(n for n in ast.parse(Path("paper_trader_integrated.py").read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == "calculate_trade_score")
        scope = {"trade_score": trade_score, "CFG": CFG}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "paper-score", "exec"), scope)
        row = self.frame.iloc[-1]
        self.assertAlmostEqual(scope["calculate_trade_score"](.7, row, .2),
                               float(trade_score(.7, row.market_ret_3, .2, CFG)))


class NewsTests(unittest.TestCase):
    def test_receipt_revision_future_and_decay(self):
        news = pd.DataFrame({"created_at": ["2026-10-01T13:30Z"], "updated_at": ["2026-10-01T13:40Z"],
                             "received_at": ["2026-10-01T13:45Z"], "sentiment": [1.]})
        result = news_scores(["2026-10-01T13:44Z", "2026-10-01T13:45Z", "2026-10-01T20:00Z"], news, require_received=True)
        np.testing.assert_allclose(result, [0., np.exp(-15/120), 0.])

    def test_unknown_receipt_rejected_in_strict_mode(self):
        news = pd.DataFrame({"timestamp": ["2026-10-01T13:30Z"], "sentiment": [1.]})
        with self.assertRaisesRegex(ValueError, "received_at"):
            news_scores(["2026-10-01T13:35Z"], news, require_received=True)
        self.assertGreater(news_scores(["2026-10-01T13:35Z"], news)[0], 0.)

    def test_future_and_duplicate_news_do_not_change_past(self):
        old = {"timestamp": "2026-10-01T13:30Z", "received_at": "2026-10-01T13:31Z", "sentiment": .5}
        future = dict(old, timestamp="2026-10-01T13:40Z", received_at="2026-10-01T13:42Z", sentiment=-1.)
        a = news_scores(["2026-10-01T13:35Z"], [old], require_received=True)
        b = news_scores(["2026-10-01T13:35Z"], [old, old, future], require_received=True)
        np.testing.assert_array_equal(a, b)

    def test_batch_and_single_same_news(self):
        n = [{"timestamp": "2026-10-01T13:30Z", "received_at": "2026-10-01T13:31Z", "sentiment": .5}]
        times = ["2026-10-01T13:35Z", "2026-10-01T13:40Z"]
        np.testing.assert_allclose(news_scores(times, n), [news_scores([t], n)[0] for t in times])

    def test_news_at_bar_completion_is_same_for_history_and_inference(self):
        raw = price_frame()
        end = raw.timestamp.iloc[-1]+pd.Timedelta(minutes=5)
        n = [{"created_at": end-pd.Timedelta(minutes=2), "received_at": end-pd.Timedelta(minutes=1), "sentiment": .5}]
        full = enrich(raw, raw, n, require_news_received=True)
        short = enrich(raw.tail(250), raw.tail(250), n, require_news_received=True)
        self.assertEqual(full.news_score.iloc[-1], short.news_score.iloc[-1])

    def test_invalid_sentiment_is_not_silent_neutral(self):
        with self.assertRaises(ValueError):
            news_scores(["2026-10-01T13:35Z"], [{"timestamp": "2026-10-01T13:30Z", "sentiment": float("nan")}])


if __name__ == "__main__":
    unittest.main()
