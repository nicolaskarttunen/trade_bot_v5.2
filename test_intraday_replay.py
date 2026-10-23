"""Offline tests for the paper-like intraday research simulator."""
import unittest

import pandas as pd

from intraday_replay import replay_intraday, session_bounded_label


CFG = {
    "slippage_bps": 0,
    "spread_bps": 0,
    "commission_per_trade": 0,
    "initial_cash": 100000,
    "max_trades_per_day": 5,
    "max_daily_loss_pct": 0.03,
    "max_open_positions": 5,
    "max_total_exposure_pct": 0.50,
    "max_position_pct": 0.10,
    "risk_per_trade_pct": 0.01,
    "probability_threshold": 0.55,
    "trade_score_threshold": 55,
    "stop_atr_multiplier": 1.5,
    "reward_risk_ratio": 2,
}


def bars(symbol_price=100., day="2026-10-01", low_override=None):
    times = [t.strftime("%H:%M") for t in pd.date_range("2000-01-01 09:30", "2000-01-01 15:50", freq="5min")]
    timestamps = pd.to_datetime(
        [f"{day} {t}" for t in times]
    ).tz_localize("America/New_York").tz_convert("UTC")
    rows = []
    for clock, ts in zip(times, timestamps):
        price = symbol_price
        lo = price - 0.25
        if low_override and clock in low_override:
            lo = low_override[clock]
        rows.append({
            "timestamp": ts, "open": price, "high": price + .25,
            "low": lo, "close": price, "volume": 1000.
        })
    return pd.DataFrame(rows)


def signals(price_bars, clocks=("09:30",), qualified=True):
    x = price_bars.copy()
    et = x["timestamp"].dt.tz_convert("America/New_York")
    x["trade_date"] = et.dt.strftime("%Y-%m-%d")
    x["clock"] = et.dt.strftime("%H:%M")
    x = x.loc[x["clock"].isin(clocks)].copy()
    x["ml_prob"] = 0.70 if qualified else 0.45
    x["trade_score"] = 75.0 if qualified else 30.0
    x["news_score"] = 0.0
    x["atr"] = 1.0
    return x


class IntradayReplayTests(unittest.TestCase):
    def run_one(self, prices, scored, **kwargs):
        return replay_intraday(
            scored, prices, sorted(prices), {s: "Technology" for s in prices},
            dict(CFG, **kwargs.pop("config", {})), **kwargs
        )

    def test_label_does_not_use_next_session_close(self):
        timestamps = pd.to_datetime([
            "2026-10-01 19:55:00+00:00",
            "2026-10-02 13:30:00+00:00",
            "2026-10-02 13:35:00+00:00",
        ])
        df = pd.DataFrame({
            "timestamp": timestamps,
            "close": [100.0, 150.0, 151.0],
        })
        labels = session_bounded_label(df, horizon=1, threshold=0.002)
        self.assertTrue(pd.isna(labels.iloc[0]["target"]))
        self.assertEqual(labels.iloc[1]["target"], 1.0)
        self.assertTrue(pd.isna(labels.iloc[2]["target"]))

    def test_buy_uses_next_bar_open_not_signal_bar(self):
        price = bars()
        only_later = signals(price, clocks=("09:35",))
        equity, trades, daily, curve = self.run_one({"A": price}, {"A": only_later})
        self.assertEqual(len(trades), 1)
        self.assertEqual(
            trades.iloc[0]["entry_ts"],
            price.iloc[2]["timestamp"],
        )
        self.assertEqual(int(daily.iloc[0]["entries"]), 1)

    def test_0930_signal_can_trade_at_0935_but_not_0930(self):
        price = bars()
        _, trades, _, _ = self.run_one({"A": price}, {"A": signals(price)})
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0]["entry_ts"], price.iloc[1]["timestamp"])

    def test_no_future_signal_from_same_bar(self):
        price = bars()
        sig = signals(price, clocks=("09:40",))
        _, trades, _, _ = self.run_one({"A": price}, {"A": sig})
        self.assertEqual(trades.iloc[0]["entry_ts"], price.iloc[3]["timestamp"])

    def test_spy_gate_skips_all_intraday_entries_on_bad_day(self):
        price = bars()
        _, trades, daily, _ = self.run_one(
            {"A": price}, {"A": signals(price)},
            require_spy=True, spy_opening_gate={"2026-10-01": False})
        self.assertTrue(trades.empty)
        self.assertEqual(int(daily.iloc[0]["entries"]), 0)

    def test_spy_gate_keeps_good_day(self):
        price = bars()
        _, trades, _, _ = self.run_one(
            {"A": price}, {"A": signals(price)},
            require_spy=True, spy_opening_gate={"2026-10-01": True})
        self.assertEqual(len(trades), 1)

    def test_stop_exit_and_no_second_buy_of_same_symbol(self):
        price = bars(low_override={"09:35": 96.})
        sig = signals(price, clocks=("09:30", "09:35", "09:40"))
        equity, trades, daily, _ = self.run_one({"A": price}, {"A": sig})
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0]["reason"], "stop")
        self.assertLess(equity, CFG["initial_cash"])
        self.assertEqual(int(daily.iloc[0]["entries"]), 1)

    def test_max_positions_and_order_of_symbol_scans(self):
        price_a = bars()
        price_b = bars(symbol_price=200.)
        _, trades, _, _ = self.run_one(
            {"A": price_a, "B": price_b},
            {"A": signals(price_a), "B": signals(price_b)},
            config={"max_trades_per_day": 1, "max_open_positions": 1},
        )
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0]["symbol"], "A")

    def test_all_positions_flat_at_eod(self):
        price = bars()
        equity, trades, daily, curve = self.run_one(
            {"A": price}, {"A": signals(price)}
        )
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0]["reason"], "eod")
        self.assertAlmostEqual(equity, CFG["initial_cash"])
        self.assertAlmostEqual(float(daily.iloc[-1]["end_equity"]), equity)
        self.assertEqual(curve.iloc[-1]["clock"], "15:50")

    def test_no_signal_means_no_trade(self):
        price = bars()
        equity, trades, _, _ = self.run_one(
            {"A": price}, {"A": signals(price, qualified=False)}
        )
        self.assertTrue(trades.empty)
        self.assertEqual(equity, CFG["initial_cash"])


if __name__ == "__main__":
    unittest.main()
