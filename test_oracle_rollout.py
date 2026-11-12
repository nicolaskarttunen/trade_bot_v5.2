"""Read-only Oracle inspection tests: fake broker data and blocked sockets."""
from contextlib import closing, redirect_stdout
import io
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tools import check_oracle_rollout as check


def empty_broker():
    return dict(account=dict(id="synthetic-paper-id", status="ACTIVE", trading_blocked=False),
                positions=[], orders=[])


class StateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = Path(self.tmp.name)

    def test_empty_account_creates_no_journal_and_is_not_a_deployment_approval(self):
        before = set(self.source.iterdir())
        report = check.account_state(self.source, empty_broker())
        self.assertEqual(report["reconciliation_findings"], [])
        self.assertEqual(before, set(self.source.iterdir()))
        self.assertFalse(report["journal"]["present"])

    def test_positions_nested_parent_orders_and_legacy_queue_require_reconciliation(self):
        queue = {"AAA": dict(status="submitted", order_id="legacy-id")}
        path = self.source / "opening_orders.json"
        path.write_text(json.dumps(queue))
        original = path.read_bytes()
        broker = empty_broker()
        broker["positions"] = [dict(symbol="AAA", qty="3")]
        broker["orders"] = [dict(symbol="AAA", legs=[dict(symbol="AAA")])]
        report = check.account_state(self.source, broker)
        self.assertEqual(report["positions"], 1)
        self.assertEqual(report["open_parent_orders"], 1)
        self.assertEqual(report["legacy_submitted_records"], 1)
        self.assertEqual(len(report["reconciliation_findings"]), 2)
        self.assertEqual(path.read_bytes(), original)

    def test_journal_is_read_only_and_account_binding_and_unknown_intents_are_reported(self):
        dbpath = self.source / "paper_order_intents.sqlite3"
        with closing(sqlite3.connect(dbpath)) as db:
            db.executescript("CREATE TABLE metadata(key TEXT,value TEXT);"
                "INSERT INTO metadata VALUES('schema_version','1'),('account_id','different-id');"
                "CREATE TABLE intents(broker_id TEXT,state TEXT);"
                "INSERT INTO intents VALUES('known-order','unknown');")
            db.commit()
        before = dbpath.read_bytes()
        report = check.account_state(self.source, empty_broker())
        self.assertFalse(report["journal"]["account_matches"])
        self.assertEqual(report["journal"]["unresolved"], 1)
        self.assertEqual(len(report["reconciliation_findings"]), 3)
        self.assertEqual(before, dbpath.read_bytes())

    def test_unknown_schema_corrupt_queue_and_truncated_orders_stop(self):
        with self.subTest("orders"), self.assertRaisesRegex(check.CheckError, "truncated"):
            broker = empty_broker(); broker["orders"] = [dict(symbol="AAA")] * 500
            check.account_state(self.source, broker)
        (self.source / "opening_orders.json").write_text("[]")
        with self.assertRaisesRegex(check.CheckError, "queue"):
            check.account_state(self.source, empty_broker())
        (self.source / "opening_orders.json").unlink()
        with closing(sqlite3.connect(self.source / "paper_order_intents.sqlite3")) as db:
            db.executescript("CREATE TABLE metadata(key TEXT,value TEXT);"
                             "INSERT INTO metadata VALUES('schema_version','unsupported');")
            db.commit()
        with self.assertRaisesRegex(check.CheckError, "schema"):
            check.account_state(self.source, empty_broker())

    def test_paper_transport_has_only_fixed_gets_and_does_not_export_secrets(self):
        import requests
        session = Mock()
        session.headers = {}
        session.__enter__ = Mock(return_value=session)
        session.__exit__ = Mock(return_value=False)
        broker = empty_broker()
        session.get.side_effect = [SimpleNamespace(status_code=200, json=lambda x=x: x)
                                  for x in broker.values()]
        with patch.object(requests, "Session", return_value=session), \
                patch("socket.socket.connect", side_effect=AssertionError("Network forbidden")):
            result = check.paper_gets(("fake-key", "fake-secret"))
        self.assertEqual(result, broker)
        self.assertEqual(session.get.call_count, 3)
        for args in session.get.call_args_list:
            self.assertTrue(args.args[0].startswith("https://paper-api.alpaca.markets/v2/"))
            self.assertFalse(args.kwargs["allow_redirects"])
            self.assertEqual(args.kwargs["timeout"], (5, 15))
        session.post.assert_not_called(); session.delete.assert_not_called()
        self.assertNotIn("fake-key", json.dumps(result))
        self.assertFalse(session.trust_env)
        session.get.side_effect = None
        session.get.return_value = SimpleNamespace(status_code=403, text="fake-secret must stay private")
        with patch.object(requests, "Session", return_value=session), self.assertRaises(check.CheckError) as caught:
            check.paper_gets(("fake-key", "fake-secret"))
        self.assertNotIn("fake-secret", str(caught.exception))


class StartupCredentialTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = Path(self.tmp.name).resolve()
        (self.source / "paper_trader_integrated.py").write_text('load_dotenv(".env.v5")\n')
        (self.source / ".env.v5").write_text("ALPACA_API_KEY=file-key\nALPACA_SECRET_KEY=file-secret\n")

    def test_startup_dotenv_pair_is_resolved_without_using_callers_environment_or_writing_files(self):
        before = {p.name: p.read_bytes() for p in self.source.iterdir()}
        with patch.object(check, "service_startup", return_value=({}, time.time() + 2)), \
                patch.dict(os.environ, {"ALPACA_API_KEY": "other-key", "ALPACA_SECRET_KEY": "other-secret"}):
            pair, origin = check.running_service_pair(self.source, {})
        self.assertEqual(pair, ("file-key", "file-secret"))
        self.assertEqual(origin, "verified_service_startup_dotenv_file")
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.source.iterdir()})

    def test_initial_service_pair_takes_precedence_and_partial_pairs_never_mix(self):
        values = dict(ALPACA_API_KEY="startup-key", ALPACA_SECRET_KEY="startup-secret")
        with patch.object(check, "service_startup", return_value=(values, 0)):
            pair, origin = check.running_service_pair(self.source, {})
        self.assertEqual(pair, ("startup-key", "startup-secret"))
        self.assertEqual(origin, "verified_service_initial_environment")
        for values in ({"ALPACA_API_KEY": "only-key"}, {"ALPACA_API_KEY": ""}):
            with patch.object(check, "service_startup", return_value=(values, time.time() + 2)), \
                    self.assertRaisesRegex(check.CheckError, "mixing sources"):
                check.running_service_pair(self.source, {})

    def test_changed_startup_files_interpolation_and_unsupported_loads_stop(self):
        with patch.object(check, "service_startup", return_value=({"PYTHON_DOTENV_DISABLED": "true"}, time.time() + 2)), \
                self.assertRaisesRegex(check.CheckError, "disabled at startup"):
            check.running_service_pair(self.source, {})
        with patch.object(check, "service_startup", return_value=({}, 0)), \
                self.assertRaisesRegex(check.CheckError, "changed since startup"):
            check.running_service_pair(self.source, {})
        (self.source / ".env.v5").write_text("ALPACA_API_KEY=${OTHER}\nALPACA_SECRET_KEY=file-secret\n")
        with patch.object(check, "service_startup", return_value=({}, time.time() + 2)), \
                self.assertRaisesRegex(check.CheckError, "literal credential pair"):
            check.running_service_pair(self.source, {})
        (self.source / "paper_trader_integrated.py").write_text('load_dotenv("different-file")\n')
        with patch.object(check, "service_startup", return_value=({}, time.time() + 2)), \
                self.assertRaisesRegex(check.CheckError, "supported literal"):
            check.running_service_pair(self.source, {})

    @unittest.skipUnless(os.name == "posix", "Oracle verifies Linux process ownership and /proc")
    def test_process_identity_and_selected_initial_keys_are_verified(self):
        proc = self.source / "fake-proc"; process = proc / "123"
        process.mkdir(parents=True)
        (process / "cmdline").write_bytes(os.fsencode(str(self.source / "paper_trader_integrated.py")) + b"\0")
        (process / "cwd").symlink_to(self.source, target_is_directory=True)
        (process / "environ").write_bytes(b"ALPACA_API_KEY=key\0ALPACA_SECRET_KEY=secret\0UNRELATED=never-export\0")
        (process / "stat").write_text("123 (python with spaces) " + " ".join(["0"] * 19 + [str(os.sysconf("SC_CLK_TCK") * 25)]))
        (proc / "stat").write_text("btime 1000\n")
        values, started = check.service_startup(self.source, dict(ActiveState="active", MainPID="123"), proc)
        self.assertEqual(values, {"ALPACA_API_KEY": "key", "ALPACA_SECRET_KEY": "secret"})
        self.assertEqual(started, 1025)
        (process / "cmdline").write_bytes(b"unrelated_process\0")
        with self.assertRaisesRegex(check.CheckError, "expected source file"):
            check.service_startup(self.source, dict(ActiveState="active", MainPID="123"), proc)


@unittest.skipUnless(os.name == "posix", "Oracle existing preparation lock uses flock")
class CandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_oracle_candidate import CFG, bars
        from tools import prepare_oracle_candidate as candidate
        cls.candidate = candidate
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.source = cls.root / "source"; cls.source.mkdir()
        cls.output = cls.root / "candidate"
        cfg = dict(CFG, symbols=["AAA", "BBB"], benchmark_symbols=["SPY"],
                   history_start="2026-01-01", history_end="2026-02-01")
        (cls.source / "config.json").write_text(json.dumps(cfg))
        rows = {s: bars(seed=i + 30) for i, s in enumerate(["AAA", "BBB", "SPY"])}
        with redirect_stdout(io.StringIO()), patch("socket.socket.connect", side_effect=AssertionError("Network forbidden")):
            candidate.run(cls.source, cls.output, fetch=lambda p: dict(bars={p["symbols"]: rows[p["symbols"]]}))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_real_candidate_and_fake_empty_account_preserve_all_inputs(self):
        before = {p.relative_to(self.root): self.candidate.digest(p) for p in self.root.rglob("*") if p.is_file()}
        with patch.object(check, "service_info", return_value=dict(ActiveState="active", MainPID="123")), \
                patch("socket.socket.connect", side_effect=AssertionError("Network forbidden")):
            report = check.inspect(self.source, self.output, get_broker=empty_broker)
        self.assertTrue(report["candidate"]["verified"])
        self.assertEqual(report["candidate"]["predictions_checked"], 2)
        self.assertFalse(report["deployment_performed"])
        self.assertEqual(report["assessment"], "inspection_only_repeat_at_cutover")
        after = {p.relative_to(self.root): self.candidate.digest(p) for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_changed_pickle_is_rejected_before_loading_and_no_broker_query(self):
        path = self.output / "candidate.pkl"; original = path.read_bytes()
        try:
            path.write_bytes(b"changed pickle must not execute")
            with patch.object(check.pickle, "load", side_effect=AssertionError("Untrusted pickle")) as load, \
                    self.assertRaisesRegex(check.CheckError, "model hash"):
                check.verify_candidate(self.source, self.output)
            load.assert_not_called()
        finally:
            path.write_bytes(original)

    def test_active_training_and_changed_service_stop_inspection(self):
        with self.candidate.exclusive(self.output), self.assertRaisesRegex(check.CheckError, "lock"):
            check.inspect(self.source, self.output, get_broker=empty_broker)
        with patch.object(check, "service_info", side_effect=[dict(MainPID="1"), dict(MainPID="2")]), \
                self.assertRaisesRegex(check.CheckError, "service changed"):
            check.inspect(self.source, self.output, get_broker=empty_broker)

    def test_changed_source_config_or_data_prevents_broker_access(self):
        for path in (self.source / "config.json", self.output / "data/AAA_5min.csv"):
            original = path.read_bytes()
            try:
                if path.name == "config.json":
                    cfg = json.loads(original); cfg["max_position_pct"] = .02
                    path.write_text(json.dumps(cfg))
                else:
                    path.write_bytes(original + b"changed")
                broker = Mock(side_effect=AssertionError("Must not query broker"))
                with self.assertRaises(check.CheckError):
                    check.inspect(self.source, self.output, get_broker=broker)
                broker.assert_not_called()
            finally:
                path.write_bytes(original)

    def test_no_unverified_credential_fallback_or_exception_secret_output(self):
        with patch.object(check, "service_info", return_value=dict(MainPID="1")), \
                patch.object(check, "running_service_pair", side_effect=check.CheckError("Startup sources unavailable")), \
                patch.object(check, "paper_gets", side_effect=AssertionError("Credentials unverified")):
            with self.assertRaisesRegex(check.CheckError, "Startup sources unavailable"):
                check.inspect(self.source, self.output)
        output = io.StringIO()
        with patch.object(check, "inspect", side_effect=RuntimeError("fake-secret")), \
                patch("sys.argv", ["check", "--source-dir", str(self.source), "--candidate-dir", str(self.output)]), \
                redirect_stdout(output):
            self.assertEqual(check.main(), 1)
        self.assertNotIn("fake-secret", output.getvalue())


if __name__ == "__main__":
    unittest.main()
