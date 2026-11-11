"""Entry-selection regressions with synthetic signals and a fake paper broker."""
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from types import SimpleNamespace
import threading
import tempfile
import unittest
from unittest.mock import Mock

import pandas as pd

from entry_selection import long_trade_plan, rank_entry_candidates, valid_opening_selection, SELECTION_POLICY
from test_paper_boundaries import functions

CFG = json.loads(Path(__file__).with_name("config.json").read_text())


class TradePlanTests(unittest.TestCase):
    def test_legacy_or_incomplete_opening_selection_requires_rescan(self):
        valid = {"policy": SELECTION_POLICY, "symbols": ["A", "B"],
                 "scanned_count": 2, "complete": True, "rank": 1}
        self.assertTrue(valid_opening_selection(valid, ["A", "B"]))
        for selection in (None, {}, dict(valid, complete=False), dict(valid, scanned_count=1),
                          dict(valid, rank=6), dict(valid, symbols=["A"])):
            self.assertFalse(valid_opening_selection(selection, ["A", "B"]))

    def test_scenarios_match_bracket_sizing_and_both_sides_costs(self):
        plan = long_trade_plan(100, 2, 100000, dict(CFG, commission_per_trade=1))
        self.assertEqual(plan["quantity"], 100)
        self.assertEqual(plan["stop_price"], 97)
        self.assertEqual(plan["target_price"], 106)
        self.assertAlmostEqual(plan["target_net_pnl_usd"], 588.7)
        self.assertAlmostEqual(plan["stop_net_pnl_usd"], -310.85)
        self.assertEqual(plan["commission_round_trip_usd"], 1)
        self.assertIsNone(plan["expected_pnl_usd"])

    def test_atr_floor_and_portfolio_cap_apply_to_scenarios(self):
        plan = long_trade_plan(100, .1, 100000, CFG, quantity_cap=7)
        self.assertEqual(plan["stop_price"], 98.75)
        self.assertEqual(plan["target_price"], 102.50)
        self.assertEqual(plan["quantity"], 7)
        self.assertAlmostEqual(plan["target_net_pnl_usd"], 17.5 - .70875)
        empty = long_trade_plan(100, .1, 100000, dict(CFG, commission_per_trade=1), quantity_cap=0)
        self.assertEqual(empty["target_net_pnl_usd"], 0)
        self.assertEqual(empty["stop_net_pnl_usd"], 0)

    def test_invalid_prices_atr_and_costs_cannot_create_plans(self):
        for price, atr in [(float("nan"), 2), (0, 2), (100, -1), (100, float("inf")), (1, 2)]:
            with self.subTest(price=price, atr=atr), self.assertRaises(ValueError):
                long_trade_plan(price, atr, 100000, CFG)
        with self.assertRaises(ValueError):
            long_trade_plan(100, 2, 100000, dict(CFG, commission_per_trade=float("nan")))

    def test_ranking_ignores_input_order_and_excludes_blocked_and_invalid_signals(self):
        items = [{"symbol": f"S{i:03}", "probability": .7, "score": 70} for i in range(8)]
        items += [{"symbol": "BAD", "probability": float("nan"), "score": 100},
                  {"symbol": "LOW", "probability": .54, "score": 100}]
        expected = [f"S{i:03}" for i in range(1, 6)]
        for values in (items, list(reversed(items))):
            ranked = rank_entry_candidates(values, CFG, {"S000"})
            self.assertEqual([item["symbol"] for item in ranked], expected)


class FullScanTests(unittest.TestCase):
    def test_actual_submission_refreshes_scenario_after_price_and_quantity_change(self):
        broker = Mock()
        broker.submit_once.return_value = SimpleNamespace(id="paper-1")
        review = {}
        scope = functions(
            "submit_paper_long", CFG=CFG, pd=pd, datetime=datetime, timezone=timezone,
            telegram_trading_enabled=True, require_entry_ready=Mock(),
            get_daily_state=lambda: {"date": "2026-11-12", "trades": 0, "traded_symbols": []},
            daily_loss_reached=lambda state: (False, 100000, 100000),
            get_positions=lambda: {}, get_latest_trade_price=lambda symbol: 105.,
            has_open_buy_order=lambda symbol: False, is_fresh_intraday_row=lambda *args: True,
            long_trade_plan=long_trade_plan, enforce_entry_limits=lambda **kwargs: 9,
            MarketOrderRequest=SimpleNamespace, TakeProfitRequest=SimpleNamespace,
            StopLossRequest=SimpleNamespace, OrderSide=SimpleNamespace(BUY="buy"),
            TimeInForce=SimpleNamespace(DAY="day"), OrderClass=SimpleNamespace(BRACKET="bracket"),
            ORDER_LIFECYCLE=broker, save_state=Mock(), write_trade=Mock(),
        )
        order = scope["submit_paper_long"]("A", 100., .8, 90., .1, {"atr": 2.}, review=review)
        self.assertEqual(order.id, "paper-1")
        self.assertEqual(review["plan"], long_trade_plan(105, 2, 100000, CFG, quantity_cap=9))
        request = broker.submit_once.call_args.args[3]("paper-entry-id")
        self.assertEqual(request.qty, 9)
        self.assertEqual(request.stop_loss.stop_price, review["plan"]["stop_price"])
        self.assertEqual(request.take_profit.limit_price, review["plan"]["target_price"])

    def test_old_persisted_queue_cannot_bypass_complete_universe_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "opening_orders.json"
            path.write_text(json.dumps({"A": {"status": "queued"}}))
            broker = Mock()
            scope = functions(
                "submit_queued_opening_orders", BASE=Path(tmp), json=json, SYMBOLS=["A", "B"],
                telegram_trading_enabled=True, valid_opening_selection=valid_opening_selection,
                get_daily_state=lambda: {"trades": 0}, daily_loss_reached=lambda _: (False, 100000, 100000),
                atomic_json=lambda destination, value: destination.write_text(json.dumps(value)),
                ORDER_LIFECYCLE=broker,
            )
            scope["submit_queued_opening_orders"]()
            broker.submit_once.assert_not_called()
            self.assertEqual(json.loads(path.read_text())["A"]["status"], "selection_required")

    def scope(self, **overrides):
        events = []
        writes = []
        symbols = [f"S{i:03}" for i in range(100)]
        benchmark = pd.DataFrame({"close": [100]})

        def signal(symbol, **kwargs):
            events.append(("signal", symbol, kwargs))
            probability = .56 + int(symbol[1:]) * .003
            row = pd.Series({"timestamp": kwargs["as_of"].floor("5min") - pd.Timedelta(minutes=5),
                             "close": 100., "atr": 2., "news_score": .1,
                             "ret_3": .003, "market_ret_3": .002,
                             "rsi14": 58., "volume_ratio": 1.2})
            return probability, row

        def latest(symbol):
            self.assertEqual(sum(event[0] == "signal" for event in events), 100)
            events.append(("review", symbol))
            return 100.

        def submit(symbol, *args, **kwargs):
            self.assertEqual(sum(event[0] == "review" for event in events), 5)
            self.assertEqual(writes[-1]["status"], "reviewed")
            self.assertTrue(writes[-1]["complete"])
            events.append(("submit", symbol, kwargs))
            return SimpleNamespace(id=f"paper-{symbol}")

        scope = dict(
            pd=pd, math=math, CFG=copy.deepcopy(CFG), SYMBOLS=symbols, BENCHMARK="SPY",
            ENTRY_SCAN_FILE=Path("unused-test-scan.json"),
            atomic_json=lambda path, report: writes.append(copy.deepcopy(report)),
            get_daily_state=lambda: {"trades": 0, "traded_symbols": []},
            daily_loss_reached=lambda _: (False, 100000., 100000.),
            get_portfolio_entry_usage=lambda: (set(), 0),
            get_bars=Mock(return_value=benchmark),
            news_lock=threading.Lock(), news_items=[{"symbols": ["S099"], "sentiment": .1}],
            get_ml_probability=signal, calculate_trade_score=lambda probability, row, news: 100 * probability,
            maintain_order_lifecycle=Mock(), entry_scan_is_current=Mock(return_value=True),
            rank_entry_candidates=rank_entry_candidates, long_trade_plan=long_trade_plan,
            SELECTION_POLICY=SELECTION_POLICY,
            get_latest_trade_price=latest, submit_paper_long=submit,
            telegram_trading_enabled=True, telegram_lock=threading.Lock(), telegram_signals=[],
        )
        scope.update(overrides)
        return functions("run_entry_scan", **scope), events, writes, benchmark

    def test_all_100_signals_and_all_five_reviews_precede_first_order(self):
        scope, events, writes, benchmark = self.scope()
        report = scope["run_entry_scan"]()
        self.assertEqual(report["scanned_count"], 100)
        self.assertEqual(report["submitted_count"], 5)
        self.assertEqual(report["status"], "complete")
        self.assertEqual([item["symbol"] for item in report["top5"]],
                         ["S099", "S098", "S097", "S096", "S095"])
        self.assertEqual([event[0] for event in events], ["signal"] * 100 + ["review"] * 5 + ["submit"] * 5)
        self.assertEqual(scope["get_bars"].call_count, 1)
        times = {event[2]["as_of"] for event in events[:100]}
        self.assertEqual(len(times), 1)
        self.assertTrue(all(event[2]["benchmark"] is benchmark for event in events[:100]))
        self.assertEqual(writes[-1], report)
        json.dumps(report, allow_nan=False)

    def test_occupied_and_previously_traded_symbols_are_scanned_but_not_selected(self):
        scope, events, _, _ = self.scope(
            get_daily_state=lambda: {"trades": 1, "traded_symbols": ["S099"]},
            get_portfolio_entry_usage=lambda: ({"S098"}, 10000),
        )
        report = scope["run_entry_scan"]()
        self.assertEqual(report["scanned_count"], 100)
        self.assertEqual(report["top5"][0]["symbol"], "S097")
        self.assertEqual(report["submitted_count"], 4)
        self.assertEqual(len([event for event in events if event[0] == "review"]), 5)

    def test_one_symbol_api_failure_is_recorded_and_the_rest_are_scanned(self):
        scope, events, _, _ = self.scope()
        original = scope["get_ml_probability"]

        def failure(symbol, **kwargs):
            result = original(symbol, **kwargs)
            if symbol == "S099":
                raise ConnectionError("synthetic price service failure")
            return result

        scope["get_ml_probability"] = failure
        report = scope["run_entry_scan"]()
        self.assertEqual(report["scanned_count"], 100)
        self.assertEqual(report["top5"][0]["symbol"], "S098")
        self.assertIn("error", report["signals"][-1])

    def test_expired_partial_scan_cannot_submit_or_mark_complete(self):
        scope, events, _, _ = self.scope()
        scope["entry_scan_is_current"] = lambda *args: len(events) < 40
        report = scope["run_entry_scan"]()
        self.assertEqual(report["status"], "expired")
        self.assertFalse(report["complete"])
        self.assertEqual(report["scanned_count"], 40)
        self.assertFalse(any(event[0] == "submit" for event in events))

    def test_expiry_or_stop_during_research_blocks_the_entire_submission_phase(self):
        for expire in (True, False):
            with self.subTest(expire=expire):
                scope, events, _, _ = self.scope()
                latest = scope["get_latest_trade_price"]

                def review(symbol):
                    result = latest(symbol)
                    if sum(event[0] == "review" for event in events) == 5:
                        if expire:
                            scope["entry_scan_is_current"] = lambda *args: False
                        else:
                            scope["telegram_trading_enabled"] = False
                    return result

                scope["get_latest_trade_price"] = review
                report = scope["run_entry_scan"]()
                self.assertEqual(report["status"], "expired" if expire else "paused")
                self.assertEqual(report["submitted_count"], 0)

    def test_ambiguous_submission_stops_later_orders_and_persists_review(self):
        scope, _, writes, _ = self.scope()
        scope["submit_paper_long"] = Mock(side_effect=RuntimeError("unknown broker acknowledgement"))
        with self.assertRaisesRegex(RuntimeError, "unknown broker"):
            scope["run_entry_scan"]()
        self.assertEqual(scope["submit_paper_long"].call_count, 1)
        self.assertEqual(writes[-1]["status"], "submission_error")
        self.assertEqual(writes[-1]["top5"][0]["decision"], "submission_error")

    def test_missing_price_blocks_only_that_review_and_does_not_select_a_sixth(self):
        scope, events, _, _ = self.scope()
        latest = scope["get_latest_trade_price"]

        def price(symbol):
            value = latest(symbol)
            return None if symbol == "S099" else value

        scope["get_latest_trade_price"] = price
        report = scope["run_entry_scan"]()
        self.assertEqual(len(report["top5"]), 5)
        self.assertEqual(report["submitted_count"], 4)
        self.assertEqual(report["top5"][0]["decision"], "unavailable")

    def test_opening_scan_marks_done_only_after_completed_selection(self):
        for status in ("complete", "expired", "paused"):
            with self.subTest(status=status):
                saved = Mock()
                scope = functions("run_opening_scan",
                                  get_daily_state=lambda: {"opening_scan_done": False},
                                  run_entry_scan=Mock(return_value={"complete": True, "status": status}),
                                  save_state=saved)
                scope["run_opening_scan"]()
                self.assertEqual(saved.call_count, int(status == "complete"))

    def test_real_clock_rejects_crossed_bar_and_opening_cutoff(self):
        clock = Mock(wraps=pd.Timestamp)
        scope = functions("entry_scan_is_current", pd=SimpleNamespace(Timestamp=clock))
        clock.now.return_value = pd.Timestamp("2026-11-12T14:27:59Z")
        self.assertTrue(scope["entry_scan_is_current"](pd.Timestamp("2026-11-12T14:25:02Z"), True))
        clock.now.return_value = pd.Timestamp("2026-11-12T14:28:00Z")
        self.assertFalse(scope["entry_scan_is_current"](pd.Timestamp("2026-11-12T14:25:02Z"), True))
        clock.now.return_value = pd.Timestamp("2026-11-12T15:00:00Z")
        self.assertFalse(scope["entry_scan_is_current"](pd.Timestamp("2026-11-12T14:55:02Z")))


if __name__ == "__main__":
    unittest.main()
