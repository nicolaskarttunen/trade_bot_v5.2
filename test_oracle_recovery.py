"""Recovery fault injection: fake paper broker and service manager; no network."""
import copy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import Mock, patch

from tools import recover_oracle_legacy as recovery
from order_journal import OrderJournal, client_id


class Broker:
    def __init__(self):
        self.position = [dict(symbol="A", qty="4", side="long")]
        self.open = []
        self.buy = dict(id="entry-id", client_order_id="entry-cid", symbol="A", side="buy",
                        status="canceled", filled_qty="4", qty="10", legs=[
                            dict(symbol="A", side="sell", status="canceled", filled_qty="0"),
                            dict(symbol="A", side="sell", status="canceled", filled_qty="0")])
        self.order = None
        self.sent = []
        self.fault = None
        self.account_id = "paper-account"
        self.is_open = True
        self.stale_clock = False

    def account(self):
        return dict(id=self.account_id, status="ACTIVE", trading_blocked=False)

    def positions(self):
        return copy.deepcopy(self.position)

    def orders(self):
        return copy.deepcopy(self.open)

    def parent(self, oid):
        return copy.deepcopy(self.buy)

    def clock(self):
        now = datetime.now(timezone.utc) - timedelta(hours=int(self.stale_clock))
        return dict(is_open=self.is_open, timestamp=now.isoformat())

    def lookup(self, cid):
        if not self.order:
            raise recovery.RecoveryError("Paper GET: HTTP 404; no automatic resend")
        return copy.deepcopy(self.order)

    def submit(self, payload):
        self.sent.append(copy.deepcopy(payload))
        if self.fault == "before":
            raise TimeoutError("private response must never authorize resend")
        self.order = dict(payload, id="close-id", status="filled", filled_qty="4", filled_avg_price="101")
        self.position = []
        if self.fault == "partial":
            self.order.update(status="partially_filled", filled_qty="2")
            self.position = [dict(symbol="A", qty="2", side="long")]
        if self.fault == "after":
            raise TimeoutError("acknowledgement lost")
        return copy.deepcopy(self.order)


class CloseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "intent.json"
        self.broker = Broker()
        self.entry = dict(client_id="entry-cid", broker_id="entry-id")

    def close(self, **kw):
        return recovery.close_once(self.broker, self.path, "paper-account", self.entry,
                                   "A", 4, attempts=2, sleeper=lambda _: None, **kw)

    def test_exact_sell_and_flat_confirmation_then_repeated_launch_does_not_resend(self):
        result = self.close()
        self.assertEqual(result["filled_qty"], "4")
        self.assertEqual(self.broker.sent[0]["side"], "sell")
        self.assertEqual(self.broker.sent[0]["qty"], "4")
        self.assertEqual(self.broker.sent[0]["order_class"], "simple")
        self.assertFalse(self.broker.sent[0]["extended_hours"])
        if os.name != "nt":
            self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        self.close()
        self.assertEqual(len(self.broker.sent), 1)

    def test_lost_ack_is_recovered_read_only(self):
        self.broker.fault = "after"
        with self.assertRaises(TimeoutError): self.close()
        self.assertTrue(recovery.read(self.path)["claimed"])
        self.assertEqual(self.close()["filled_qty"], "4")
        self.assertEqual(len(self.broker.sent), 1)

    def test_timeout_before_acceptance_and_404_never_retry_post(self):
        self.broker.fault = "before"
        with self.assertRaises(TimeoutError): self.close()
        with self.assertRaises(recovery.RecoveryError): self.close()
        self.assertEqual(len(self.broker.sent), 1)

    def test_claim_write_failure_prevents_post(self):
        with patch.object(recovery, "save", side_effect=OSError("disk full")):
            with self.assertRaises(OSError): self.close()
        self.assertEqual(self.broker.sent, [])

    def test_claim_is_durable_before_submission(self):
        original = self.broker.submit
        def submit(payload):
            self.assertEqual(recovery.read(self.path)["client_order_id"], payload["client_order_id"])
            return original(payload)
        self.broker.submit = submit
        self.close()

    def test_pending_partial_sell_is_not_filled_or_resent(self):
        self.broker.fault = "partial"
        with self.assertRaises(recovery.RecoveryError): self.close()
        self.broker.order.update(status="filled", filled_qty="4")
        self.broker.position = []
        self.close()
        self.assertEqual(len(self.broker.sent), 1)

    def test_wrong_position_open_order_or_active_original_parent_blocks_post(self):
        changes = [lambda: self.broker.position[0].update(qty="5"),
                   lambda: self.broker.position[0].update(side="short"),
                   lambda: self.broker.open.append(dict(symbol="A", side="sell")),
                   lambda: self.broker.buy.update(status="pending_cancel"),
                   lambda: self.broker.buy["legs"][0].update(status="new"),
                   lambda: self.broker.buy["legs"][0].update(filled_qty="1"),
                   lambda: setattr(self.broker, "account_id", "different-account"),
                   lambda: setattr(self.broker, "is_open", False),
                   lambda: setattr(self.broker, "stale_clock", True)]
        for change in changes:
            with self.subTest(change=change):
                self.broker = Broker()
                change()
                with self.assertRaises(recovery.RecoveryError): self.close()
                self.assertEqual(self.broker.sent, [])
                self.assertFalse(self.path.exists())

    def test_position_change_at_final_check_prevents_claim_and_post(self):
        before = self.broker.positions
        count = 0
        def positions():
            nonlocal count
            count += 1
            if count == 2:
                self.broker.position[0]["qty"] = "3"
            return before()
        self.broker.positions = positions
        with self.assertRaises(recovery.RecoveryError): self.close()
        self.assertFalse(self.path.exists())
        self.assertEqual(self.broker.sent, [])

    def test_repeated_intent_cannot_be_rebound_to_another_entry(self):
        self.close()
        self.entry["client_id"] = "different-entry"
        with self.assertRaises(recovery.RecoveryError): self.close()
        self.assertEqual(len(self.broker.sent), 1)

    def test_flat_account_with_terminal_original_needs_no_sell(self):
        self.broker.position = []
        self.assertTrue(self.close()["already_flat"])
        self.assertEqual(self.broker.sent, [])
        self.broker.buy["status"] = "partially_filled"
        with self.assertRaises(recovery.RecoveryError): self.close()


class RestoreTests(unittest.TestCase):
    def test_daily_state_keeps_consumed_slot_instead_of_yesterdays_count(self):
        current = dict(date="2026-01-06", trades=1, start_equity=1000, traded_symbols=["A"])
        old = dict(date="2026-01-05", trades=5, start_equity=1000)
        result = recovery.merge_daily(current, old, {"A"}, 5)
        self.assertEqual(result["trades"], 1)
        self.assertEqual(result["traded_symbols"], ["A"])

    def test_same_day_count_and_symbol_union_never_go_backwards(self):
        current = dict(date="2026-01-06", trades=1, start_equity=1000, traded_symbols=["A"])
        old = dict(date=current["date"], trades=3, start_equity=1000, traded_symbols=["B"])
        result = recovery.merge_daily(current, old, {"A"}, 5)
        self.assertEqual(result["trades"], 3)
        self.assertEqual(result["traded_symbols"], ["A", "B"])
        old["start_equity"] = 999
        with self.assertRaises(recovery.RecoveryError): recovery.merge_daily(current, old, {"A"}, 5)

    def test_corrupt_or_newer_daily_state_is_not_reset(self):
        current = dict(date="2026-01-06", trades=0, start_equity=1000, traded_symbols=[])
        with self.assertRaises(recovery.RecoveryError): recovery.merge_daily(current, {}, {"A"}, 5)
        current["trades"] = 1
        with self.assertRaises(recovery.RecoveryError):
            recovery.merge_daily(current, dict(date="2026-01-07"), {"A"}, 5)

    def test_original_model_and_code_must_match_retained_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, release, backup = (root / name for name in ("old", "new", "backup"))
            for p in (source, release, backup): p.mkdir()
            names = ["paper_trader_integrated.py", "features.py", "ml_models.py",
                     "models/ensemble.pkl", "config.json", ".env.v5"]
            for name in names:
                (source / name).parent.mkdir(parents=True, exist_ok=True)
                (source / name).write_text("trusted-old-" + name)
            (release / "config.json").write_bytes((source / "config.json").read_bytes())
            with tarfile.open(backup / "original-tree.tar", "w") as archive:
                archive.add(source, arcname="original")
            recovery.verify_old_files(source, release, backup)
            (source / "models/ensemble.pkl").write_text("changed-model")
            with self.assertRaises(recovery.RecoveryError): recovery.verify_old_files(source, release, backup)

    def test_original_is_never_started_before_flat_account_and_exact_unit_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, backup, output = (root / name for name in ("old", "backup", "out"))
            for p in (source, backup, output): p.mkdir()
            override = root / "new.conf"; override.write_text("replacement")
            (backup / "original-unit.txt").write_bytes(b"original-unit")
            broker = Broker()
            host = Mock()
            host.run.return_value = b"original-unit"
            host.info.side_effect = [dict(WorkingDirectory=str(source), MainPID="0")] + [
                dict(WorkingDirectory=str(source), MainPID="123", ActiveState="active")] * 5
            with self.assertRaises(recovery.RecoveryError):
                recovery.start_original(host, override, source, backup, output, broker,
                                        "paper-account", {}, sleeper=lambda _: None)
            host.sudo.assert_not_called()
            broker.position = []
            host.run.return_value = b"changed-unit"
            with self.assertRaises(recovery.RecoveryError):
                recovery.start_original(host, override, source, backup, output, broker,
                                        "paper-account", {}, sleeper=lambda _: None)
            self.assertNotIn(unittest.mock.call("systemctl", "start", recovery.SERVICE), host.sudo.call_args_list)
            host.run.return_value = b"original-unit"
            info = recovery.start_original(host, override, source, backup, output, broker,
                                           "paper-account", {}, sleeper=lambda _: None)
            self.assertEqual(info["MainPID"], "123")
            self.assertIn(unittest.mock.call("systemctl", "start", recovery.SERVICE), host.sudo.call_args_list)


class TransportTests(unittest.TestCase):
    def test_paper_only_one_post_no_redirect_proxy_or_transport_retry(self):
        import requests
        session = Mock()
        session.request.side_effect = requests.Timeout("private-server-data")
        with patch.object(requests, "Session", return_value=session):
            broker = recovery.Paper(("key", "secret"))
            with self.assertRaises(recovery.RecoveryError) as error:
                broker.submit(dict(symbol="A", side="sell", qty="4"))
        self.assertNotIn("private-server-data", str(error.exception))
        self.assertFalse(session.trust_env)
        self.assertEqual(session.request.call_count, 1)
        self.assertEqual(session.request.call_args.args[:2], ("POST", recovery.API + "/orders"))
        self.assertFalse(session.request.call_args.kwargs["allow_redirects"])
        self.assertEqual(session.mount.call_args.args[1].max_retries.total, 0)


class FullRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.source, self.release, self.backup = (root / name for name in (
            "original", "20260106T120000Z-example", "backup"))
        self.output = self.backup / "recovery"
        for p in (self.source, self.release, self.backup, self.output):
            p.mkdir(parents=True, exist_ok=True)
        for name in ("paper_trader_integrated.py", "features.py", "ml_models.py", "models/ensemble.pkl", ".env.v5"):
            p = self.source / name; p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("retained-original-" + name)
        self.old_model = (self.source / "models/ensemble.pkl").read_bytes()
        (self.source / ".env.v5").write_text("ALPACA_API_KEY=K\nALPACA_SECRET_KEY=S\n")
        recovery.save(self.source / "config.json", dict(max_trades_per_day=5))
        shutil_copy = (self.source / "config.json").read_bytes()
        (self.release / "config.json").write_bytes(shutil_copy)
        recovery.save(self.source / "paper_live_state.json", dict(
            date="2026-01-05", trades=5, start_equity=1000, traded_symbols=[]))
        recovery.save(self.release / "paper_live_state.json", dict(
            date="2026-01-06", trades=1, start_equity=1000, traded_symbols=["A"], opening_scan_done=True))
        recovery.save(self.source / "opening_orders.json", {})
        recovery.save(self.release / "opening_orders.json", {})
        with tarfile.open(self.backup / "original-tree.tar", "w") as archive:
            archive.add(self.source, arcname="original")
        recovery.save(self.backup / "manifest.json", dict(source=str(self.source), release=str(self.release),
            base_interpreter="/usr/bin/python3.14", override="/etc/systemd/system/v52-paper-bot.service.d/zz-oracle-top5-" + self.release.name + ".conf"))
        (self.release / ".paper-service.env").write_text("ALPACA_API_KEY=K\nALPACA_SECRET_KEY=S\n")
        self.journal = OrderJournal(self.release / "paper_order_intents.sqlite3", "paper-account")
        cid = client_id("paper-account", "2026-01-06", "A", "entry")
        self.journal.reserve("2026-01-06", "A", "entry", dict(
            client_order_id=cid, symbol="A", side="buy", qty=10), 100, {"mode": "intraday"})
        with self.journal.connection() as db:
            db.execute("UPDATE intents SET state='observed', broker_id='entry-id', status='canceled', filled_qty='4' WHERE client_id=?", (cid,))
        self.broker = Broker(); self.broker.buy["client_order_id"] = cid
        self.state = dict(WorkingDirectory=str(self.release), ActiveState="active", MainPID="111")
        self.calls = []
        outer = self
        class Host:
            def __init__(self, output): pass
            def info(self): return dict(outer.state)
            def sudo(self, *args):
                outer.calls.append(args)
                if args == ("cat", "/etc/v52-paper-bot.env"):
                    return b"PYTHONPATH=/example\n"
                if args == ("systemctl", "stop", recovery.SERVICE):
                    outer.state.update(ActiveState="inactive", MainPID="0")
        self.Host = Host

    def test_stop_backup_single_close_risk_preservation_and_original_start(self):
        submit = self.broker.submit
        def checked_submit(payload):
            self.assertEqual(self.state["ActiveState"], "inactive")
            marker = recovery.read(self.output / "runtime-backup-complete.json")
            self.assertTrue(Path(marker["runtime_archive"]).is_file())
            with recovery.closing(recovery.sqlite3.connect(marker["journal"])) as db:
                self.assertEqual(db.execute("SELECT count(*) FROM intents").fetchone()[0], 1)
            return submit(payload)
        self.broker.submit = checked_submit
        def start(*args):
            state = recovery.read(self.source / "paper_live_state.json")
            self.assertEqual(state["trades"], 1)
            self.assertEqual(state["traded_symbols"], ["A"])
            self.assertTrue(recovery.read(self.output / "close-intent.json")["complete"])
            self.assertEqual(self.broker.positions(), [])
            self.state.update(WorkingDirectory=str(self.source), ActiveState="active", MainPID="222")
            return dict(self.state)
        with patch.object(recovery, "Host", self.Host), patch.object(recovery, "Paper", return_value=self.broker), \
             patch.object(recovery, "capture_old_runtime"), patch.object(recovery, "assert_no_writer"), \
             patch.object(recovery, "start_original", side_effect=start):
            recovery.execute(self.release, self.backup, self.output, "A", 4)
            recovery.execute(self.release, self.backup, self.output, "A", 4)
        self.assertEqual(len(self.broker.sent), 1)
        self.assertEqual(len(self.journal.rows()), 1)
        self.assertEqual((self.source / "models/ensemble.pkl").read_bytes(), self.old_model)
        self.assertEqual(recovery.read(self.output / "restored.json")["retained_daily_trades"], 1)

    def test_old_runtime_failure_stops_before_service_stop_or_sell(self):
        with patch.object(recovery, "Host", self.Host), patch.object(recovery, "Paper", return_value=self.broker), \
             patch.object(recovery, "capture_old_runtime", side_effect=recovery.RecoveryError("old runtime invalid")):
            with self.assertRaises(recovery.RecoveryError):
                recovery.execute(self.release, self.backup, self.output, "A", 4)
        self.assertNotIn(("systemctl", "stop", recovery.SERVICE), self.calls)
        self.assertEqual(self.broker.sent, [])


if __name__ == "__main__":
    unittest.main()
