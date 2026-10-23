import unittest
import numpy as np
import pandas as pd
from test_intraday_replay import bars, signals, CFG
from intraday_replay import replay_intraday
from research_validation import build_spy_gate, paired_daily_comparison
from v6_5_intraday_replay import metrics
from backtest_execution import simulate_long_exit


class ReplayRiskTests(unittest.TestCase):
    def run_case(self, prices, scores, config=None, **kwargs):
        return replay_intraday(scores, prices, list(prices), {}, dict(CFG, **(config or {})), **kwargs)

    def test_risk_position_and_total_exposure_include_costs(self):
        prices = {str(i): bars() for i in range(8)}
        config = {"slippage_bps": 20, "spread_bps": 10, "commission_per_trade": 2.,
                  "risk_per_trade_pct": .001, "max_position_pct": .1, "max_total_exposure_pct": .25}
        final, trades, daily, curve = self.run_case(prices, {s: signals(p) for s,p in prices.items()}, config)
        self.assertTrue((trades.planned_stop_loss <= trades.entry_equity*.001+1e-8).all())
        self.assertTrue((trades.entry*trades.qty <= trades.post_entry_equity*.1+1e-8).all())
        self.assertTrue((trades.entry_exposure <= trades.post_entry_equity*.25+1e-8).all())
        self.assertAlmostEqual(final-CFG["initial_cash"], trades.pnl.sum(), places=8)
        self.assertAlmostEqual(daily.pnl.sum(), trades.pnl.sum(), places=8)
        self.assertAlmostEqual(curve.equity.iloc[-1], final)

    def test_fee_is_one_roundtrip_and_cash_never_borrowed(self):
        price = bars()
        final, trades, _, _ = self.run_case({"A":price}, {"A":signals(price)},
                                           {"initial_cash": 1000., "commission_per_trade": 3.})
        self.assertEqual(len(trades), 1)
        self.assertEqual(final, 997.)
        self.assertLessEqual(trades.entry.iloc[0]*trades.qty.iloc[0]+3, 1000.)

    def test_stop_gap_and_target_gap_precede_intrabar_extremes(self):
        for opening, expected in [(90., "stop_gap"), (105., "target_gap")]:
            price = bars()
            price.loc[2, ["open", "high", "low", "close"]] = [opening, 110., 85., opening]
            final, trades, _, _ = self.run_case({"A":price}, {"A":signals(price)})
            self.assertEqual(trades.reason.iloc[0], expected)
            self.assertEqual(trades.exit.iloc[0], 90. if opening==90 else 103.)
        path = price.iloc[[2]]
        self.assertEqual(simulate_long_exit(path, 98, 103, 0)[1], "target_gap")

    def test_same_bar_stop_wins_and_event_time_is_not_before_entry(self):
        price = bars(low_override={"09:35": 95.})
        price.loc[1,"high"] = 105.
        _, trades, _, _ = self.run_case({"A":price}, {"A":signals(price)})
        self.assertEqual(trades.reason.iloc[0], "stop")
        self.assertGreater(trades.exit_ts.iloc[0], trades.entry_ts.iloc[0])

    def test_missing_held_bar_and_missing_eod_are_errors(self):
        for index in [2, -1]:
            price = bars()
            scores = signals(price)
            price = price.drop(price.index[index])
            with self.assertRaisesRegex(ValueError, "Unobservable"):
                self.run_case({"A":price}, {"A":scores})

    def test_early_close_uses_supplied_exchange_calendar(self):
        price = bars(day="2026-11-27")
        price = price[price.timestamp.dt.tz_convert("America/New_York").dt.strftime("%H:%M") <= "12:50"]
        _, trades, _, curve = self.run_case({"A":price}, {"A":signals(price)}, session_closes={"2026-11-27":"13:00"})
        self.assertEqual(trades.reason.iloc[0], "eod")
        self.assertEqual(trades.exit_ts.iloc[0].tz_convert("America/New_York").strftime("%H:%M"), "12:50")
        self.assertEqual(curve.clock.iloc[-1], "12:50")

    def test_daily_loss_blocks_later_entries_and_resets_next_session(self):
        a = bars(low_override={"09:35":96.})
        b = bars()
        tomorrow = bars(day="2026-10-02")
        pa = pd.concat([a,tomorrow], ignore_index=True)
        pb = pd.concat([b,tomorrow], ignore_index=True)
        sa = pd.concat([signals(a),signals(tomorrow)], ignore_index=True)
        sb = signals(pb, clocks=("09:35",))
        _, trades, daily, _ = self.run_case({"A":pa,"B":pb}, {"A":sa,"B":sb}, {"max_daily_loss_pct":.001})
        self.assertEqual(daily.entries.iloc[0], 1)
        self.assertGreater(daily.entries.iloc[1], 0)

    def test_intraday_equity_captures_unrealized_drawdown(self):
        price = bars()
        price.loc[2, ["open","high","low","close"]] = [100.,100.25,99.,99.]
        final,trades,daily,curve = self.run_case({"A":price}, {"A":signals(price)})
        self.assertEqual(final, CFG["initial_cash"])
        self.assertLess(metrics(CFG["initial_cash"],final,trades,daily,curve)["max_dd_pct"], 0)

    def test_bad_config_duplicate_prices_and_forged_signal_clock(self):
        price = bars()
        with self.assertRaises(ValueError):
            self.run_case({"A":price}, {"A":signals(price)}, {"risk_per_trade_pct":np.nan})
        with self.assertRaises(ValueError):
            self.run_case({"A":pd.concat([price,price.iloc[[0]]])}, {"A":signals(price)})
        sig = signals(price, clocks=("09:40",))
        sig["clock"] = "09:30"
        _,trades,_,_ = self.run_case({"A":price}, {"A":sig})
        self.assertEqual(trades.entry_ts.iloc[0], price.timestamp.iloc[3])

    def test_future_volume_does_not_control_open_entry(self):
        price = bars()
        scores = signals(price)
        _,original,_,_ = self.run_case({"A":price},{"A":scores})
        price.loc[1,"volume"] = 0
        _,changed,_,_ = self.run_case({"A":price},{"A":scores})
        pd.testing.assert_frame_equal(original,changed)


class SpyProtocolTests(unittest.TestCase):
    def test_threshold_is_inclusive_and_uses_only_first_completed_bar(self):
        price = bars()
        price.loc[0,"close"] = 100.10
        gate = build_spy_gate(price)
        self.assertTrue(gate["2026-10-01"])
        price.loc[1:, ["open","high","low","close"]] *= 2
        self.assertEqual(gate, build_spy_gate(price))
        price.loc[0,"close"] = 100.099
        self.assertFalse(build_spy_gate(price)["2026-10-01"])

    def test_missing_opening_bar_gate_fails_closed(self):
        self.assertEqual(build_spy_gate(bars().iloc[1:]), {})
        price = bars()
        _,trades,_,_ = replay_intraday({"A":signals(price)}, {"A":price},["A"],{},CFG,{},True)
        self.assertTrue(trades.empty)

    def test_paired_bootstrap_includes_zero_days_and_is_reproducible(self):
        a = pd.DataFrame({"date":pd.date_range("2026-01-01", periods=20).strftime("%Y-%m-%d"),
                          "pnl": [0.,1.,-1.,2.,-2.]*4, "start_equity":1000.})
        b = a.copy(); b["pnl"] = 0.
        r = paired_daily_comparison(a,b)
        self.assertEqual(r, paired_daily_comparison(a,b))
        self.assertEqual(r["sessions"],20)
        self.assertFalse(r["independent_evidence"])
        with self.assertRaises(ValueError):
            paired_daily_comparison(a,b.iloc[1:])


if __name__ == "__main__":
    unittest.main()
