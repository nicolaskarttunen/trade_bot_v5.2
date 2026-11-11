"""Extract individual functions; NEVER import the process or contact Alpaca."""
import ast
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import pandas as pd
from features import completed_bars
from test_pipeline_contract import price_frame


def functions(*names, **scope):
    source = Path(__file__).with_name("paper_trader_integrated.py")
    tree = ast.parse(source.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if len(nodes) != len(names):
        raise AssertionError("Function missing from paper process")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
    return scope


class Status(Enum):
    NEW = "new"
    PARTIAL = "partially_filled"


class PaperBoundaryTests(unittest.TestCase):
    def test_enum_status_pending_buy_is_detected(self):
        broker = Mock()
        broker.get_orders.return_value = [SimpleNamespace(symbol="A", side="buy", status=Status.NEW)]
        scope = functions("has_open_buy_order", "_enum_text", trading=broker)
        self.assertTrue(scope["has_open_buy_order"]("A"))
        self.assertFalse(scope["has_open_buy_order"]("B"))
        broker.get_orders.side_effect = ConnectionError("offline failure")
        self.assertTrue(scope["has_open_buy_order"]("B"))

    def test_positions_and_pending_remainder_are_counted(self):
        broker = Mock()
        broker.get_all_positions.return_value = [SimpleNamespace(symbol="A", market_value=200., qty=2)]
        broker.get_orders.return_value = [SimpleNamespace(symbol="A", side="buy", status=Status.PARTIAL,
                                                         qty=5, filled_qty=2, limit_price=100.)]
        scope = functions("get_portfolio_entry_usage", "_active_buy_orders", "_enum_text", trading=broker)
        occupied, exposure = scope["get_portfolio_entry_usage"]()
        self.assertEqual(occupied, {"A"})
        self.assertEqual(exposure, 500.)

    def test_portfolio_lookup_failure_blocks_entry(self):
        scope = functions("enforce_entry_limits", CFG={"max_open_positions":5,"max_total_exposure_pct":.5},
                          get_portfolio_entry_usage=Mock(side_effect=ConnectionError("offline failure")))
        self.assertEqual(scope["enforce_entry_limits"]("A",100,10,100000),0)

    def test_descending_request_and_completed_bar_selection(self):
        raw = price_frame(300)
        now = raw.timestamp.iloc[-1] + pd.Timedelta(minutes=2)
        class Clock:
            @staticmethod
            def now(tz):
                return now.to_pydatetime()
        broker = Mock()
        broker.get_stock_bars.return_value = SimpleNamespace(df=raw.iloc[::-1].set_index("timestamp"))
        request = Mock(side_effect=lambda **kw: SimpleNamespace(**kw))
        scope = functions("get_bars", datetime=Clock, timezone=timezone, timedelta=timedelta,
                          StockBarsRequest=request, TimeFrame=lambda *args:args,
                          TimeFrameUnit=SimpleNamespace(Minute="Minute"), Sort=SimpleNamespace(DESC="desc"),
                          CFG={"feed":"iex"}, data_client=broker, pd=pd, completed_bars=completed_bars)
        result = scope["get_bars"]("A")
        self.assertEqual(request.call_args.kwargs["sort"], "desc")
        self.assertEqual(request.call_args.kwargs["limit"],251)
        self.assertEqual(len(result),250)
        self.assertEqual(result.timestamp.iloc[-1], raw.timestamp.iloc[-2])
        self.assertTrue(result.timestamp.is_monotonic_increasing)

    def test_freshness_rejects_forming_old_and_future_bars(self):
        clock = Mock(wraps=pd.Timestamp)
        clock.now.return_value = pd.Timestamp("2026-10-23T13:52:00Z")
        scope = functions("is_fresh_intraday_row", pd=SimpleNamespace(Timestamp=clock, Timedelta=pd.Timedelta))
        now = pd.Timestamp("2026-10-23T13:50:00Z")
        self.assertTrue(scope["is_fresh_intraday_row"]("A",{"timestamp":now-pd.Timedelta(minutes=5)}))
        for ts in (now, now+pd.Timedelta(minutes=5), now-pd.Timedelta(minutes=10)):
            self.assertFalse(scope["is_fresh_intraday_row"]("A",{"timestamp":ts}))

    def test_flat_eod_with_pending_buy_is_not_confirmed_flat(self):
        broker = Mock()
        broker.get_all_positions.return_value = []
        broker.get_orders.return_value = [SimpleNamespace(side="buy")]
        scope = functions("flatten_end_of_day", trading=broker)
        self.assertFalse(scope["flatten_end_of_day"]())
        broker.cancel_orders.assert_called_once()
        broker.get_orders.return_value = []
        self.assertTrue(scope["flatten_end_of_day"]())

    def test_unknown_and_pending_cancel_orders_still_reserve_exposure(self):
        broker = Mock()
        for status in ["pending_cancel", "held", "unknown_new_sdk_status"]:
            broker.get_orders.return_value = [SimpleNamespace(symbol="A", side="buy", status=status)]
            scope = functions("_active_buy_orders", "_enum_text", "has_open_buy_order", trading=broker)
            self.assertEqual(len(scope["_active_buy_orders"]()), 1)
            self.assertTrue(scope["has_open_buy_order"]("A"))

    def test_eod_after_close_still_requires_empty_order_book(self):
        broker = Mock()
        broker.get_all_positions.side_effect = [[SimpleNamespace(symbol="A")], []]
        broker.get_orders.return_value = [SimpleNamespace(side="buy")]
        scope = functions("flatten_end_of_day", trading=broker, time=SimpleNamespace(sleep=lambda _:None))
        self.assertFalse(scope["flatten_end_of_day"]())
        broker.close_all_positions.assert_called_once_with(cancel_orders=True)

    def test_eod_cancel_failure_fails_closed(self):
        broker = Mock()
        broker.get_all_positions.return_value = []
        broker.cancel_orders.side_effect = ConnectionError("offline failure")
        scope = functions("flatten_end_of_day", trading=broker)
        self.assertFalse(scope["flatten_end_of_day"]())


if __name__ == "__main__":
    unittest.main()
