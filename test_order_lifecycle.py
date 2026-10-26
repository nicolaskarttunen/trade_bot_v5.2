"""Fault injection against the actual durable gateway. No clients/credentials/network."""
import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from order_journal import OrderJournal, OrderBlocked, SubmissionUnknown, ProcessLock, client_id
from order_lifecycle import OrderLifecycle, snapshot, exit_state
from test_paper_boundaries import functions

SESSION = "2026-10-26"


def request(cid, symbol="A", qty=10, side="buy", **kw):
    return dict(client_order_id=cid, symbol=symbol, qty=qty, side=side, **kw)


def acknowledged(req):
    return dict(req, id="broker-"+req["client_order_id"], status="new", filled_qty=0,
                filled_avg_price=None, legs=[], order_type=req.get("type", "market"))


class Broker:
    def __init__(self):
        self.orders = {}
        self.positions = []
        self.sends = []
        self.cancels = []
        self.fault = None
        self.foreign = []
        self.lookup_error = False

    def submit_order(self, req):
        self.sends.append(copy.deepcopy(req))
        if self.fault == "before":
            raise TimeoutError()
        order = acknowledged(req)
        self.orders[req["client_order_id"]] = order
        if self.fault == "after":
            raise TimeoutError()
        return copy.deepcopy(order)

    def get_order_by_client_id(self, cid):
        if self.lookup_error:
            raise TimeoutError()
        return copy.deepcopy(self.orders[cid])

    def get_all_positions(self):
        return copy.deepcopy(self.positions)

    def get_orders(self):
        # Fixtures explicitly supply external SELLs when needed; journal
        # lookups provide owned parents/legs without a paginated list.
        return copy.deepcopy(self.foreign)

    def cancel_order_by_id(self, oid):
        self.cancels.append(oid)
        if self.fault == "cancel":
            raise TimeoutError()


def oco(cid, symbol, qty, stop, target):
    return request(cid, symbol, qty, "sell", type="limit", order_class="oco",
                   stop_loss={"stop_price":stop}, take_profit={"limit_price":target})


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/"journal.sqlite3"
        self.journal = OrderJournal(self.path, "paper-account")
        self.broker = Broker()
        self.life = OrderLifecycle(self.journal, self.broker)

    def send(self, symbol="A", session=SESSION, mode="opening"):
        return self.life.submit_once(session, symbol, "entry", lambda cid:request(cid,symbol),100,
                {"mode":mode,"stop_distance":2,"reward_risk_ratio":2})

    def entry(self):
        return next(r for r in self.journal.rows() if r["role"] == "entry")

    def fill(self, qty, status="partially_filled", avg=101):
        cid = self.entry()["client_id"]
        self.broker.orders[cid].update(filled_qty=qty, status=status, filled_avg_price=avg)
        self.broker.positions = [dict(symbol="A",qty=qty)] if qty else []
        self.life.reconcile()

    def test_lost_ack_recovered_by_id_after_restart_never_resent(self):
        self.broker.fault="after"
        with self.assertRaises(SubmissionUnknown): self.send()
        self.assertEqual(self.entry()["state"],"unknown")
        self.assertTrue(self.life.entry_blockers(SESSION))
        self.life=OrderLifecycle(OrderJournal(self.path,"paper-account"),self.broker)
        self.life.reconcile()
        self.assertEqual(self.entry()["state"],"observed")
        with self.assertRaises(OrderBlocked): self.send()
        self.assertEqual(len(self.broker.sends),1)

    def test_timeout_before_acceptance_and_404_do_not_authorize_retry(self):
        self.broker.fault="before"
        with self.assertRaises(SubmissionUnknown): self.send()
        self.life.reconcile()
        with self.assertRaises(OrderBlocked): self.send()
        with self.assertRaises(OrderBlocked): self.send("B")
        self.assertEqual(len(self.broker.sends),1)

    def test_crash_between_durable_reservation_and_rpc_never_resends(self):
        cid=client_id("paper-account",SESSION,"A","entry")
        self.journal.reserve(SESSION,"A","entry",request(cid),100)
        self.life.reconcile()
        with self.assertRaises(OrderBlocked): self.send()
        self.assertEqual(self.broker.sends,[])

    def test_disk_failure_before_reservation_means_no_rpc(self):
        with patch.object(self.journal,"reserve",side_effect=OSError("disk full")):
            with self.assertRaises(OSError): self.send()
        self.assertEqual(self.broker.sends,[])

    def test_snapshot_write_failure_after_send_preserves_unknown_intent(self):
        with patch.object(self.journal,"observe",side_effect=OSError("disk full")):
            with self.assertRaises(SubmissionUnknown): self.send()
        self.assertEqual(self.entry()["state"],"unknown")
        with self.assertRaises(OrderBlocked): self.send()
        self.assertEqual(len(self.broker.sends),1)

    def test_account_mismatch_and_corrupt_database_fail_closed(self):
        with self.assertRaises(OrderBlocked): OrderJournal(self.path,"different-account")
        broken=Path(self.temp.name)/"broken"
        broken.write_bytes(b"not sqlite")
        with self.assertRaises(sqlite3.DatabaseError): OrderJournal(broken,"paper-account")

    def test_two_instances_cannot_reserve_same_intent(self):
        other=OrderJournal(self.path,"paper-account")
        cid=client_id("paper-account",SESSION,"A","entry")
        def reserve(j):
            try: return j.reserve(SESSION,"A","entry",request(cid),100)
            except OrderBlocked: return None
        with ThreadPoolExecutor(2) as pool:
            outcomes=list(pool.map(reserve,[self.journal,other]))
        self.assertEqual(sum(x is not None for x in outcomes),1)

    def test_process_lock_prevents_second_writer(self):
        path=Path(self.temp.name)/"writer.lock"
        first=ProcessLock(path)
        try:
            with self.assertRaises(OrderBlocked): ProcessLock(path)
        finally: first.close()
        second=ProcessLock(path); second.close()

    def test_mismatched_and_unsupported_acknowledgements_block(self):
        self.send()
        cid=self.entry()["client_id"]
        original=copy.deepcopy(self.broker.orders[cid])
        for change in [dict(symbol="B"),dict(side="sell"),dict(id="other"),dict(qty=11),
                       dict(status="replaced"),dict(status="mystery"),dict(filled_qty=11),
                       dict(status="filled",filled_qty=5),dict(filled_qty=1,filled_avg_price=None)]:
            with self.subTest(change=change):
                self.broker.orders[cid]=dict(original,**change)
                self.life.reconcile()
                self.assertEqual(self.entry()["state"],"unknown")
        self.broker.orders[cid]=original
        self.life.reconcile()
        self.assertEqual(self.entry()["state"],"observed")

    def test_cumulative_fill_regression_is_rejected(self):
        self.send(); self.fill(4)
        self.fill(3)
        self.assertEqual(self.entry()["state"],"unknown")
        self.assertEqual(self.entry()["filled_qty"],"4")

    def test_partial_cancel_ack_is_not_final_and_late_fill_is_included(self):
        self.send(); self.fill(4)
        self.life.maintain_opening_protection(oco)
        self.life.maintain_opening_protection(oco)
        self.assertEqual(len(self.broker.cancels),1)
        self.assertEqual(len(self.broker.sends),1)
        self.assertTrue(self.life.entry_blockers(SESSION))
        self.fill(6,"pending_cancel",102)
        self.life.maintain_opening_protection(oco)
        self.assertEqual(len(self.broker.sends),1)
        self.fill(6,"canceled",102)
        self.life.maintain_opening_protection(oco)
        sent=self.broker.sends[-1]
        self.assertEqual(sent["qty"],6)
        self.assertEqual(sent["stop_loss"]["stop_price"],100)
        self.assertEqual(sent["take_profit"]["limit_price"],106)
        self.life.maintain_opening_protection(oco)
        self.assertEqual(len(self.broker.sends),2)

    def test_cancel_timeout_blocks_until_parent_reconciled(self):
        self.send(); self.fill(4); self.broker.fault="cancel"
        self.life.maintain_opening_protection(oco)
        self.assertEqual(self.entry()["state"],"unknown")
        self.life.reconcile(); self.life.maintain_opening_protection(oco)
        self.assertEqual(len(self.broker.cancels),1)
        self.assertEqual(len(self.broker.sends),1)

    def test_position_lag_and_existing_sell_prevent_protection(self):
        self.send(); self.fill(10,"filled")
        self.broker.positions=[]
        self.assertTrue(self.life.maintain_opening_protection(oco))
        self.broker.positions=[dict(symbol="A",qty=10)]
        sell=acknowledged(request("foreign",side="sell"))
        self.broker.foreign=[sell]
        self.assertTrue(self.life.maintain_opening_protection(oco))
        self.assertEqual(len(self.broker.sends),1)

    def test_unknown_oco_ack_never_retries_or_market_sells(self):
        self.send(); self.fill(10,"filled"); self.broker.fault="after"
        self.assertTrue(self.life.maintain_opening_protection(oco))
        self.life.reconcile()
        self.life.maintain_opening_protection(oco)
        self.assertEqual(len(self.broker.sends),2)
        self.assertEqual(self.broker.sends[-1]["order_class"],"oco")
        self.assertTrue(self.life.entry_blockers(SESSION))  # no active stop reported yet

    def test_canceled_partial_bracket_needs_manual_protection_reconciliation(self):
        self.send(mode="intraday"); self.fill(4,"canceled")
        self.life.maintain_opening_protection(oco)
        self.assertEqual(len(self.broker.sends),1)
        self.assertTrue(self.life.entry_blockers(SESSION))

    def test_fractional_fill_is_not_truncated_into_protection(self):
        self.send(); self.fill(3.5,"canceled")
        self.assertTrue(self.life.maintain_opening_protection(oco))
        self.assertEqual(len(self.broker.sends),1)

    def test_unknown_previous_day_order_blocks_new_session(self):
        self.broker.fault="after"
        with self.assertRaises(SubmissionUnknown): self.send(session="2026-10-23")
        self.life.reconcile()
        self.assertTrue(self.life.entry_blockers(SESSION))

    def test_daily_attempts_survive_restart_and_rejection(self):
        self.send(); self.fill(0,"rejected")
        recovered=OrderJournal(self.path,"paper-account")
        self.assertEqual(recovered.daily_symbols(SESSION),{"A"})
        with self.assertRaises(OrderBlocked): self.send()

    def test_exposure_floor_covers_invisible_ack_without_double_counting(self):
        self.send()
        self.assertEqual(self.life.reserve_exposure(SESSION,{}),{"A":1000})
        self.assertEqual(self.life.reserve_exposure(SESSION,{"A":1000,"B":200}),{"A":1000,"B":200})
        self.assertEqual(self.life.reserve_exposure(SESSION,{"A":1100}),{"A":1100})
        self.fill(0,"rejected")
        self.assertEqual(self.life.reserve_exposure(SESSION,{}),{})

    def protective_fixture(self):
        self.send(); self.fill(10,"filled")
        self.life.maintain_opening_protection(oco)
        protected=next(r for r in self.journal.rows() if r["role"]=="protection")
        root=self.broker.orders[protected["client_id"]]
        stop=acknowledged(request("stop-leg",side="sell"))
        stop.update(order_type="stop",stop_price=99,order_class="oco")
        root["legs"]=[stop]
        self.life.reconcile()
        return root,stop

    def test_active_stop_requires_correct_price_and_quantity(self):
        root,stop=self.protective_fixture()
        self.assertEqual(self.life.entry_blockers(SESSION),[])
        for change in [dict(stop_price=1),dict(qty=9),dict(status="held"),dict(status="pending_cancel")]:
            orig=copy.deepcopy(stop); stop.update(change)
            self.life.reconcile()
            self.assertTrue(self.life.entry_blockers(SESSION))
            stop.clear();stop.update(orig)

    def test_partial_take_profit_requires_reduced_stop_quantity(self):
        root,stop=self.protective_fixture()
        root.update(filled_qty=3,filled_avg_price=105,status="partially_filled")
        self.broker.positions=[dict(symbol="A",qty=7)]
        self.life.reconcile()
        self.assertTrue(self.life.entry_blockers(SESSION))
        stop["qty"]=7
        self.life.reconcile()
        self.assertEqual(self.life.entry_blockers(SESSION),[])

    def test_flat_requires_terminal_sibling_and_settles_before_symbol_reuse(self):
        root,stop=self.protective_fixture()
        root.update(filled_qty=10,filled_avg_price=105,status="filled")
        self.broker.positions=[]
        self.life.reconcile()
        self.assertTrue(self.life.entry_blockers(SESSION))
        stop["status"]="canceled"
        self.life.reconcile()
        self.assertEqual(self.life.entry_blockers(SESSION),[])
        self.assertTrue(all(r["settled"] for r in self.journal.rows()))
        self.send(session="2026-10-27")
        self.assertEqual(self.life.entry_blockers("2026-10-27"),[])

    def test_unowned_position_order_and_short_position_fail_closed(self):
        self.broker.positions=[dict(symbol="B",qty=1)]
        self.assertTrue(self.life.entry_blockers(SESSION))
        self.broker.positions=[dict(symbol="B",qty=-1)]
        with self.assertRaises(OrderBlocked): self.life.entry_blockers(SESSION)
        self.broker.positions=[]
        self.broker.foreign=[acknowledged(request("foreign"))]
        self.assertTrue(self.life.entry_blockers(SESSION))

    def test_exit_overfill_and_snapshot_mismatch_do_not_report_protected(self):
        root,stop=self.protective_fixture()
        root.update(filled_qty=10,filled_avg_price=105,status="filled")
        stop.update(filled_qty=10,filled_avg_price=99,status="filled")
        self.broker.positions=[]
        self.life.reconcile()
        self.assertTrue(self.life.entry_blockers(SESSION))


class IntegrationTests(unittest.TestCase):
    def test_only_durable_gateway_submits_and_notifications_have_no_order_side_effects(self):
        import ast
        tree=ast.parse(Path(__file__).with_name("paper_trader_integrated.py").read_text())
        submitters=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=="submit_order":
                submitters.append(node.lineno)
        self.assertEqual(submitters,[])
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="check_filled_orders")
        calls={n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id for n in ast.walk(node)
               if isinstance(n,ast.Call) and isinstance(n.func,(ast.Name,ast.Attribute))}
        self.assertFalse(calls & {"submit_once","submit_order","close_position","close_all_positions","save_state"})

    def test_corrupt_or_missing_risk_state_never_resets_existing_journal(self):
        import json, math
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"state.json"
            lifecycle=SimpleNamespace(journal=SimpleNamespace(rows=lambda:["existing"]))
            scope=functions("load_state",STATE_FILE=path,ORDER_LIFECYCLE=lifecycle,
                            json=json,math=math,OrderBlocked=OrderBlocked)
            with self.assertRaises(OrderBlocked): scope["load_state"]()
            path.write_text('{broken')
            with self.assertRaises(ValueError): scope["load_state"]()
            path.write_text('{"date":"2026-10-26","trades":1,"start_equity":0}')
            with self.assertRaises(OrderBlocked): scope["load_state"]()

    def test_atomic_state_write_failure_keeps_previous_state(self):
        import os,json
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"state.json";path.write_text('{"trades":1}')
            scope=functions("atomic_json",os=os,json=json,tempfile=tempfile)
            with patch.object(os,"replace",side_effect=OSError("disk failure")):
                with self.assertRaises(OSError):scope["atomic_json"](path,{"trades":2})
            self.assertEqual(json.loads(path.read_text()),{"trades":1})
            scope["atomic_json"](path,{"trades":2})
            self.assertEqual(json.loads(path.read_text()),{"trades":2})

    def test_sdk_transport_does_not_retry_ambiguous_post(self):
        from importlib.metadata import version
        import alpaca
        self.assertEqual(version("alpaca-py"), "0.44.0")
        self.assertEqual(alpaca.__version__, "0.44.0")
        from paper_order_adapter import PaperTradingClient
        from requests import Response, Timeout
        from alpaca.common.exceptions import APIError
        client=PaperTradingClient("offline-key", "offline-secret")
        self.assertIn("paper-api", client._base_url.value)
        for status in [429,504]:
            response=Response();response.status_code=status
            response._content=b'{"code":504,"message":"ambiguous"}'
            client._session.request=Mock(return_value=response)
            with self.assertRaises(APIError): client.post("/orders", {"client_order_id":"cid"})
            client._session.request.assert_called_once()
            self.assertEqual(client._session.request.call_args.kwargs["timeout"], (5,10))
        client._session.request=Mock(side_effect=Timeout())
        with self.assertRaises(Timeout): client.post("/orders", {})
        client._session.request.assert_called_once()
        client._session.close()

    def test_maintenance_runs_with_notifications_unavailable_and_skips_eod_sends(self):
        import pandas as pd
        life=Mock()
        life.maintain_opening_protection.return_value=[]
        life.entry_blockers.return_value=[]
        broker=Mock()
        broker.get_clock.return_value=SimpleNamespace(is_open=True,next_close=pd.Timestamp.now(tz="UTC")+pd.Timedelta(hours=1))
        factory=Mock()
        scope=functions("maintain_order_lifecycle",ORDER_LIFECYCLE=life,trading=broker,
                        pd=pd,protective_request=factory,OrderBlocked=OrderBlocked,
                        time=__import__("time"),ORDER_LAST_MAINTENANCE=0.,ORDER_LAST_ISSUES=[])
        self.assertEqual(scope["maintain_order_lifecycle"](force=True),[])
        life.reconcile.assert_called_once()
        life.maintain_opening_protection.assert_called_once_with(factory)
        life.reset_mock()
        broker.get_clock.return_value.next_close=pd.Timestamp.now(tz="UTC")+pd.Timedelta(minutes=5)
        scope["maintain_order_lifecycle"](force=True)
        life.maintain_opening_protection.assert_not_called()

    def test_entry_guard_rechecks_close_boundary_and_unknown_orders(self):
        import pandas as pd
        broker=Mock()
        broker.get_clock.return_value=SimpleNamespace(is_open=True,next_close=pd.Timestamp.now(tz="UTC")+pd.Timedelta(minutes=5))
        maintenance=Mock(return_value=[])
        scope=functions("require_entry_ready",maintain_order_lifecycle=maintenance,
                        trading=broker,pd=pd,OrderBlocked=OrderBlocked)
        with self.assertRaises(OrderBlocked):scope["require_entry_ready"]()
        broker.get_clock.return_value.is_open=False
        with self.assertRaises(OrderBlocked):scope["require_entry_ready"]()
        scope["require_entry_ready"](allow_queue=True)
        maintenance.return_value=["unresolved"]
        with self.assertRaises(OrderBlocked):scope["require_entry_ready"](allow_queue=True)

    def test_daily_state_recovers_attempt_count_without_queue_deletion(self):
        import pandas as pd
        today=pd.Timestamp.now(tz="America/New_York").date().isoformat()
        state={"date":today,"start_equity":100000,"trades":0,"opening_scan_done":False,"traded_symbols":[]}
        save=Mock()
        broker=Mock();broker.get_account.return_value=SimpleNamespace(equity=100000)
        life=SimpleNamespace(journal=SimpleNamespace(daily_symbols=lambda day:{"A","B"}))
        scope=functions("get_daily_state",load_state=lambda:copy.deepcopy(state),save_state=save,
                        ORDER_LIFECYCLE=life,trading=broker,pd=pd)
        result=scope["get_daily_state"]()
        self.assertEqual(result["trades"],2)
        self.assertEqual(result["traded_symbols"],["A","B"])
        self.assertEqual(scope["get_daily_state"]()["trades"],2)

    def test_real_opening_submission_uses_gateway_and_cannot_duplicate_lost_ack(self):
        import json
        import pandas as pd
        from datetime import datetime,timezone
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide,TimeInForce
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            journal=OrderJournal(base/"journal", "paper-account")
            broker=Broker();broker.fault="after"
            # Accept actual SDK requests without constructing a broker client.
            original=broker.submit_order
            broker.submit_order=lambda req: original(req.model_dump(mode="json",exclude_none=True))
            life=OrderLifecycle(journal,broker)
            queue={"A":{"status":"queued","created_at":"2026-10-26T13:00:00+00:00",
                        "quantity":10,"reference_price":100,"stop_price":98}}
            (base/"opening_orders.json").write_text(json.dumps(queue))
            def state():
                attempts=journal.daily_symbols(SESSION)
                return {"date":SESSION,"trades":len(attempts),"traded_symbols":sorted(attempts)}
            scope=functions("submit_queued_opening_orders", BASE=base,json=json,pd=pd,
                get_daily_state=state,daily_loss_reached=lambda _: (False,100000,100000),
                require_entry_ready=lambda:None,get_positions=lambda:{},has_open_buy_order=lambda _:False,
                get_latest_trade_price=lambda _:100,CFG={"max_position_pct":.1,"max_trades_per_day":5,"reward_risk_ratio":2},
                enforce_entry_limits=lambda **kw:kw["requested_quantity"],MarketOrderRequest=MarketOrderRequest,
                OrderSide=OrderSide,TimeInForce=TimeInForce,ORDER_LIFECYCLE=life,datetime=datetime,timezone=timezone,
                atomic_json=lambda p,v:p.write_text(json.dumps(v)))
            scope["submit_queued_opening_orders"]()
            scope["submit_queued_opening_orders"]()
            self.assertEqual(len(broker.sends),1)
            self.assertEqual(journal.rows()[0]["state"],"unknown")
            # A candidate left over from another session must never be sent.
            queue["B"]=dict(queue["A"],created_at="2026-10-23T13:00:00+00:00")
            (base/"opening_orders.json").write_text(json.dumps(queue))
            scope["submit_queued_opening_orders"]()
            self.assertEqual(len(broker.sends),1)

    def test_real_intraday_submission_persists_bracket_id_before_ack(self):
        from alpaca.trading.requests import MarketOrderRequest,TakeProfitRequest,StopLossRequest
        from alpaca.trading.enums import OrderSide,TimeInForce,OrderClass
        with tempfile.TemporaryDirectory() as tmp:
            journal=OrderJournal(Path(tmp)/"journal", "paper-account")
            broker=Broker();broker.fault="after"
            original=broker.submit_order
            def send(req):
                self.assertEqual(journal.rows()[0]["state"], "prepared")
                return original(req.model_dump(mode="json",exclude_none=True))
            broker.submit_order=send
            life=OrderLifecycle(journal,broker)
            saved=Mock()
            scope=functions("submit_paper_long",require_entry_ready=Mock(),
                get_daily_state=lambda:{"date":SESSION,"trades":0,"traded_symbols":[]},
                daily_loss_reached=lambda _: (False,100000,100000),get_positions=lambda:{},
                get_latest_trade_price=lambda _:100,has_open_buy_order=lambda _:False,
                enforce_entry_limits=lambda **kw:kw["requested_quantity"],
                CFG={"max_trades_per_day":5,"risk_per_trade_pct":.001,"stop_atr_multiplier":1.5,
                     "max_position_pct":.1,"reward_risk_ratio":2},save_state=saved,
                MarketOrderRequest=MarketOrderRequest,TakeProfitRequest=TakeProfitRequest,
                StopLossRequest=StopLossRequest,OrderSide=OrderSide,TimeInForce=TimeInForce,
                OrderClass=OrderClass,ORDER_LIFECYCLE=life)
            with self.assertRaises(SubmissionUnknown):scope["submit_paper_long"]("A",100,.8,90,0,{"atr":2})
            with self.assertRaises(OrderBlocked):scope["submit_paper_long"]("A",100,.8,90,0,{"atr":2})
            self.assertEqual(len(broker.sends),1)
            payload=broker.sends[0]
            self.assertEqual(payload["order_class"],"bracket")
            self.assertEqual(payload["stop_loss"]["stop_price"],97)
            self.assertEqual(payload["take_profit"]["limit_price"],106)
            saved.assert_not_called()
            self.assertEqual(journal.daily_symbols(SESSION),{"A"})

    def test_background_poll_cache_and_forced_entry_reconciliation(self):
        import pandas as pd
        life=Mock();life.maintain_opening_protection.return_value=[];life.entry_blockers.return_value=["blocked"]
        broker=Mock()
        broker.get_clock.return_value=SimpleNamespace(is_open=True,next_close=pd.Timestamp.now(tz="UTC")+pd.Timedelta(hours=1))
        timer=SimpleNamespace(monotonic=lambda:100.)
        scope=functions("maintain_order_lifecycle",ORDER_LIFECYCLE=life,trading=broker,pd=pd,
            protective_request=Mock(),OrderBlocked=OrderBlocked,time=timer,
            ORDER_LAST_MAINTENANCE=99.,ORDER_LAST_ISSUES=["cached"])
        self.assertEqual(scope["maintain_order_lifecycle"](),["cached"])
        life.reconcile.assert_not_called()
        self.assertEqual(scope["maintain_order_lifecycle"](force=True),["blocked"])
        life.reconcile.assert_called_once()

    def test_alpaca_request_and_nested_lookup_contract_offline(self):
        from paper_order_adapter import protective_request,lookup_nested,open_orders
        from alpaca.trading.client import TradingClient
        import inspect
        self.assertIn("client_id",inspect.signature(TradingClient.get_order_by_client_id).parameters)
        req=protective_request("cid","A",6,99,105)
        payload=req.model_dump(mode="json",exclude_none=True)
        self.assertEqual(payload["client_order_id"],"cid")
        self.assertEqual(payload["order_class"],"oco")
        self.assertEqual(payload["stop_loss"]["stop_price"],99)
        broker=Mock()
        broker.get_order_by_client_id.return_value=SimpleNamespace(id="id")
        lookup_nested(broker,"cid")
        self.assertTrue(broker.get_order_by_id.call_args.kwargs["filter"].nested)
        broker.get_orders.return_value=[None]*500
        with self.assertRaises(OrderBlocked):open_orders(broker)


if __name__=="__main__":unittest.main()
