"""Stateful broker faults against the real SQLite EOD gateway, offline only."""
import copy
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from order_journal import OrderJournal, OrderBlocked, TERMINAL, client_id
from order_lifecycle import OrderLifecycle, family, snapshot
from test_order_lifecycle import Broker, acknowledged, request, SESSION


def closing(cid, symbol, qty):
    return request(cid, symbol, qty, "sell", type="market", order_class="simple")


class VisibleBroker(Broker):
    def get_orders(self):
        return [copy.deepcopy(o) for o in self.orders.values()
                if any(n["status"] not in TERMINAL for n in family(snapshot(o)))] + copy.deepcopy(self.foreign)


class EodTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "journal.sqlite3"
        self.journal = OrderJournal(self.path, "paper-account")
        self.broker = VisibleBroker()
        self.life = OrderLifecycle(self.journal, self.broker)

    def restart(self):
        self.journal = OrderJournal(self.path, "paper-account")
        self.life = OrderLifecycle(self.journal, self.broker)

    def seed(self, qty=10, status="filled", exits=True):
        self.life.submit_once(SESSION, "A", "entry", lambda cid:request(
            cid, qty=10, type="market", order_class="bracket",
            stop_loss={"stop_price":98}, take_profit={"limit_price":104}), 100,
            {"mode":"intraday", "stop_distance":2, "reward_risk_ratio":2})
        root = next(iter(self.broker.orders.values()))
        root.update(status=status, filled_qty=qty, filled_avg_price=100 if qty else None)
        if exits:
            target = acknowledged(request("target", qty=10, side="sell", type="limit"))
            stop = acknowledged(request("stop", qty=10, side="sell", type="stop"))
            target.update(limit_price=104)
            stop.update(stop_price=98)
            if status != "filled":
                target["status"] = stop["status"] = "held"
            root["legs"] = [target, stop]
        self.broker.positions = [dict(symbol="A",qty=qty)] if qty else []
        return root

    def liquidate(self):
        return self.life.liquidate(SESSION, closing)

    def cancel_exits(self, root):
        for leg in root["legs"]:
            if leg["status"] not in TERMINAL:
                leg["status"] = "canceled"

    def close_order(self):
        rows = [r for r in self.journal.rows() if r["role"] == "close"]
        return self.broker.orders[rows[-1]["client_id"]] if rows else None

    def test_waits_for_both_terminal_exits_and_never_repeats_cancels(self):
        root = self.seed()
        self.assertFalse(self.liquidate()["complete"])
        self.assertEqual(len(self.broker.cancels),2)
        self.liquidate(); self.restart(); self.liquidate()
        self.assertEqual(len(self.broker.cancels),2)
        self.assertEqual(len(self.broker.sends),1)
        root["legs"][0]["status"] = "canceled"
        self.liquidate()
        self.assertIsNone(self.close_order())
        self.cancel_exits(root)
        self.liquidate()
        self.assertEqual(self.close_order()["qty"],10)
        for _ in range(3):
            self.liquidate()
        self.assertEqual(len(self.broker.sends),2)
        self.assertEqual(len(self.broker.cancels),2)

    def test_partial_entry_late_fill_is_counted_before_closing(self):
        root = self.seed(4,"partially_filled")
        self.liquidate()
        self.assertEqual(self.broker.cancels,[root["id"]])
        root.update(status="pending_cancel",filled_qty=6)
        self.broker.positions = [dict(symbol="A",qty=6)]
        self.liquidate()
        self.assertIsNone(self.close_order())
        root["status"] = "canceled"
        self.cancel_exits(root)
        self.liquidate()
        self.assertEqual(self.close_order()["qty"],6)

    def test_stop_fill_during_cancel_reduces_close_and_flat_is_confirmed(self):
        root = self.seed()
        self.liquidate()
        self.cancel_exits(root)
        root["legs"][1].update(filled_qty=4,filled_avg_price=97)
        self.broker.positions = [dict(symbol="A",qty=6)]
        self.liquidate()
        close = self.close_order()
        self.assertEqual(close["qty"],6)
        close.update(status="filled",filled_qty=6,filled_avg_price=99)
        # Filled order with a lagging nonzero position is NOT confirmed flat.
        self.assertFalse(self.liquidate()["complete"])
        self.broker.positions = []
        self.assertTrue(self.liquidate()["complete"])
        self.assertTrue(all(r["settled"] for r in self.journal.rows()))
        self.assertTrue(self.life.entry_blockers(SESSION))
        with self.assertRaises(OrderBlocked):
            self.life.submit_once(SESSION,"B","entry",lambda cid:request(cid,"B"),100)
        self.assertFalse(self.journal.is_liquidating("2026-10-27"))

    def test_lost_close_ack_restart_reconciles_without_second_sell(self):
        self.seed(exits=False)
        self.broker.fault = "after"
        self.liquidate()
        self.assertEqual(len(self.broker.sends),2)
        self.restart(); self.broker.fault = None
        self.liquidate()
        self.assertEqual(len(self.broker.sends),2)
        self.assertEqual(self.broker.cancels,[])
        close = self.close_order()
        close.update(status="filled",filled_qty=10,filled_avg_price=99)
        self.broker.positions = []
        self.assertTrue(self.liquidate()["complete"])

    def test_timeout_before_acceptance_and_404_never_resend(self):
        self.seed(exits=False)
        self.broker.fault = "before"
        self.liquidate()
        self.restart(); self.broker.fault = None
        for _ in range(3):
            self.assertFalse(self.liquidate()["complete"])
        self.assertEqual(len(self.broker.sends),2)

    def test_partial_close_is_never_canceled_or_duplicated(self):
        self.seed(exits=False)
        self.liquidate()
        close = self.close_order()
        close.update(status="partially_filled",filled_qty=4,filled_avg_price=99)
        self.broker.positions = [dict(symbol="A",qty=6)]
        self.restart()
        for _ in range(3):
            self.assertFalse(self.liquidate()["complete"])
        self.assertEqual(len(self.broker.sends),2)
        self.assertEqual(self.broker.cancels,[])
        close["status"] = "canceled"
        result = self.liquidate()
        self.assertTrue(any("never cancel or resend" in b for b in result["blockers"]))
        self.assertEqual(len(self.broker.sends),2)

    def test_foreign_orders_positions_and_lookup_errors_block_all_mutation(self):
        self.seed()
        self.broker.foreign = [acknowledged(request("manual",symbol="B"))]
        self.liquidate()
        self.broker.foreign = []
        self.broker.positions.append(dict(symbol="B",qty=1))
        self.liquidate()
        self.broker.positions.pop()
        self.broker.lookup_error = True
        self.liquidate()
        self.assertEqual(len(self.broker.sends),1)
        self.assertEqual(self.broker.cancels,[])

    def test_cancel_timeout_and_crash_before_rpc_do_not_repeat_attempt(self):
        self.seed()
        self.broker.fault = "cancel"
        self.liquidate()
        self.restart(); self.liquidate()
        self.assertEqual(len(self.broker.cancels),2)
        self.assertIsNone(self.close_order())

    def test_crash_between_cancel_reservation_and_rpc_never_resends(self):
        root = self.seed(qty=4,status="partially_filled")
        self.life.reconcile()
        cid = root["client_order_id"]
        self.journal.begin_liquidation(SESSION)
        self.journal.claim_order_cancel(cid,root["id"],SESSION)
        self.restart(); self.liquidate()
        self.assertEqual(self.broker.cancels,[])
        self.assertIsNone(self.close_order())

    def test_cancel_response_alone_and_lagging_position_cannot_authorize_close(self):
        root = self.seed()
        self.liquidate()
        self.cancel_exits(root)
        self.broker.positions = [dict(symbol="A",qty=9)]
        self.liquidate()
        self.assertIsNone(self.close_order())
        self.broker.positions = [dict(symbol="A",qty=-1)]
        with self.assertRaises(OrderBlocked):
            self.liquidate()

    def test_order_and_position_lookup_outage_prevent_close(self):
        self.seed(exits=False)
        with patch.object(self.broker,"get_all_positions",side_effect=TimeoutError):
            with self.assertRaises(TimeoutError):
                self.liquidate()
        with patch.object(self.life,"open_orders",side_effect=TimeoutError):
            with self.assertRaises(TimeoutError):
                self.liquidate()
        self.assertEqual(len(self.broker.sends),1)

    def test_exit_fills_cannot_regress_disappear_or_change_symbol(self):
        root = self.seed()
        root["legs"][1].update(status="partially_filled",filled_qty=4,filled_avg_price=98)
        self.life.reconcile()
        valid = copy.deepcopy(root["legs"])
        for corruption in ("regress","disappear","symbol"):
            with self.subTest(corruption=corruption):
                root["legs"] = copy.deepcopy(valid)
                if corruption == "regress": root["legs"][1]["filled_qty"] = 3
                if corruption == "disappear": root["legs"].pop()
                if corruption == "symbol": root["legs"][1]["symbol"] = "B"
                result = self.liquidate()
                self.assertTrue(any("Unresolved" in b for b in result["blockers"]))
                self.assertEqual(self.broker.cancels,[])
                self.assertIsNone(self.close_order())

    def test_flat_account_with_queued_buy_waits_for_terminal_cancel(self):
        root = self.seed(0,"new",exits=False)
        self.assertFalse(self.liquidate()["complete"])
        self.assertEqual(self.broker.cancels,[root["id"]])
        root["status"] = "canceled"
        self.assertTrue(self.liquidate()["complete"])
        self.assertEqual(len(self.broker.sends),1)

    def test_visible_active_order_overrides_terminal_direct_lookup(self):
        root = self.seed(exits=False)
        stale = copy.deepcopy(root)
        stale.update(status="new",filled_qty=0,filled_avg_price=None)
        self.broker.foreign = [stale]
        self.assertFalse(self.liquidate()["complete"])
        self.assertIsNone(self.close_order())

    def test_settled_previous_day_close_does_not_block_next_days_exit(self):
        first = self.seed(exits=False)
        self.liquidate()
        self.close_order().update(status="filled",filled_qty=10,filled_avg_price=100)
        self.broker.positions = []
        self.assertTrue(self.liquidate()["complete"])
        next_day = "2026-10-27"
        order = self.life.submit_once(next_day,"A","entry",lambda cid:request(cid,type="market"),100)
        self.broker.orders[order["client_order_id"]].update(status="filled",filled_qty=10,filled_avg_price=100)
        self.broker.positions = [dict(symbol="A",qty=10)]
        self.life.liquidate(next_day,closing)
        self.assertEqual(len([r for r in self.journal.rows() if r["role"] == "close"]),2)

    def test_new_opening_protection_is_disabled_during_liquidation(self):
        self.life.submit_once(SESSION,"A","entry",lambda cid:request(cid),100,
                             {"mode":"opening","stop_distance":2,"reward_risk_ratio":2})
        root = next(iter(self.broker.orders.values()))
        root.update(status="filled",filled_qty=10,filled_avg_price=100)
        self.broker.positions = [dict(symbol="A",qty=10)]
        self.life.reconcile(); self.journal.begin_liquidation(SESSION)
        self.life.maintain_opening_protection(lambda *args: self.fail("must not create protection"))
        self.assertEqual(len(self.broker.sends),1)

    def test_v1_migration_preserves_intents_and_account_mismatch_writes_nothing(self):
        self.seed(exits=False)
        with sqlite3.connect(self.path) as db:
            before = db.execute("SELECT * FROM intents").fetchall()
            db.execute("UPDATE metadata SET value='1' WHERE key='schema_version'")
            db.execute("DROP TABLE liquidations")
            db.execute("DROP TABLE cancellations")
        original = self.path.read_bytes()
        with self.assertRaises(OrderBlocked): OrderJournal(self.path,"other-account")
        self.assertEqual(self.path.read_bytes(),original)
        self.restart()
        with sqlite3.connect(self.path) as db:
            self.assertEqual(db.execute("SELECT * FROM intents").fetchall(),before)
            self.assertEqual(db.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()[0],"2")
        self.liquidate()
        self.assertEqual(self.close_order()["qty"],10)

    def test_sdk_close_is_simple_day_sell_with_the_durable_identity(self):
        from paper_order_adapter import closing_request
        cid = client_id("paper-account",SESSION,"A","close")
        payload = closing_request(cid,"A",6).model_dump(mode="json",exclude_none=True)
        self.assertEqual({k:payload[k] for k in ("symbol","side","type","order_class","time_in_force","client_order_id")},
                         dict(symbol="A",side="sell",type="market",order_class="simple",time_in_force="day",client_order_id=cid))
        self.assertEqual(payload["qty"],6)
        self.assertNotIn("stop_loss",payload)
        self.assertNotIn("take_profit",payload)

    def test_read_only_inspection_reports_unfinished_liquidation(self):
        from tools.check_oracle_rollout import journal_summary
        self.seed(exits=False)
        self.liquidate()
        before = self.path.read_bytes()
        report, _ = journal_summary(self.path,"paper-account")
        self.assertEqual(report["pending_liquidations"],1)
        self.assertTrue(report["account_matches"])
        self.assertEqual(before,self.path.read_bytes())


if __name__ == "__main__":
    unittest.main()
