"""Cutover failure/recovery and actual scan-function checks with no real service/broker."""
from contextlib import closing, redirect_stdout
import copy
import io
import json
import os
from pathlib import Path, PurePosixPath
import pickle
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd
from tools import deploy_oracle_paper as deploy
from tools import probe_oracle_entry_scan as probe


def snapshot():
    now = pd.Timestamp.now(tz="UTC")
    return dict(account=dict(id="synthetic-id", status="ACTIVE", trading_blocked=False, equity="100000"),
                positions=[], orders=[], clock=dict(timestamp=now.isoformat(), is_open=False,
                                                    next_open=(now+pd.Timedelta(days=1)).isoformat()))


@unittest.skipUnless(os.name == "posix", "Systemd cutover uses Linux process and filesystem paths")
class CutoverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source, self.release, self.backup = [self.root / n for n in ("original", "release", "backup")]
        for path in (self.source, self.release, self.backup): path.mkdir()
        (self.source / "config.json").write_text('{"max_position_pct":0.2,"max_total_exposure_pct":1.0}')
        (self.release / "config.json").write_bytes((self.source / "config.json").read_bytes())
        (self.source / "tax_report.py").write_text("original untracked report")
        (self.source / "opening_orders.json").write_text("{}")
        (self.source / "paper_live_state.json").write_text('{"date":"2026-11-11","trades":5}')
        self.before = {p.name:p.read_bytes() for p in self.source.iterdir()}
        original, release = self.source, self.release
        class FakeHost:
            def __init__(self):
                self.selected = original; self.active = True; self.events = []
                self.environment_file = PurePosixPath("/etc/v52-paper-test.env")
                self.base_interpreter = PurePosixPath("/usr/bin/python3.14")
            def stop(self): self.events.append("stop"); self.active = False
            def info(self):
                return dict(WorkingDirectory=str(self.selected), MainPID="1" if self.active else "0",
                            ActiveState="active" if self.active else "inactive")
            def install(self, path): self.events.append("install"); self.selected = release
            def verify_unit(self, *args): self.events.append("verify_unit")
            def start(self): self.events.append("start"); self.active = True
            def remove_override(self): self.events.append("remove_override"); self.selected = original
            def healthy(self, path, account):
                self.events.append("healthy")
                return self.info()
            def restored(self, source):
                return self.active and self.selected == source
        self.host = FakeHost()
        self.scan = Mock(return_value=dict(scanned=100, universe=100, submitted=0))
        self.window = patch.object(deploy, "require_closed_window")
        self.window.start(); self.addCleanup(self.window.stop)
        self.writer = patch.object(deploy, "assert_no_source_writer")
        self.writer.start(); self.addCleanup(self.writer.stop)

    def cut(self, broker=None, probe=None):
        with redirect_stdout(io.StringIO()), patch("socket.socket.connect", side_effect=AssertionError("Network forbidden")):
            return deploy.cutover(self.source, self.release, self.backup, Path("/python"),
                ("fake-key", "fake-secret"), self.host, broker or snapshot, probe=probe or self.scan)

    def test_success_preserves_original_and_copies_daily_counters_without_resetting(self):
        result = self.cut()
        self.assertTrue(result["deployment_performed"])
        self.assertEqual(self.host.events, ["stop","install","verify_unit","start","healthy"])
        self.assertEqual({p.name:p.read_bytes() for p in self.source.iterdir()}, self.before)
        self.assertEqual((self.release / "paper_live_state.json").read_bytes(), self.before["paper_live_state.json"])
        with tarfile.open(self.backup / "original-tree.tar") as archive:
            self.assertEqual(archive.extractfile("original/tax_report.py").read(), self.before["tax_report.py"])
        self.assertEqual(result["scan_before_first_entry"]["submitted"], 0)
        self.assertNotIn("fake-secret", json.dumps(result))

    def test_new_position_after_stop_restores_original_and_never_installs(self):
        changed = snapshot(); changed["positions"] = [dict(symbol="AAA",qty="1")]
        with self.assertRaises(deploy.DeployError): self.cut(Mock(side_effect=[snapshot(),changed]))
        self.assertEqual(self.host.events, ["stop","start"])
        self.assertEqual(self.host.selected, self.source)
        self.assertTrue(self.host.active)

    def test_expired_probe_restores_original_without_unit_change(self):
        with self.assertRaises(probe.ProbeError):
            self.cut(probe=Mock(side_effect=probe.ProbeError("expired")))
        self.assertEqual(self.host.events, ["stop","start"])
        self.assertTrue(json.loads((self.backup / "failure-state.json").read_text())["original_service_restored"])

    def test_start_failure_restores_original_only_after_flat_and_empty_journal_checks(self):
        self.host.healthy = Mock(side_effect=deploy.DeployError("failed startup"))
        with self.assertRaises(deploy.DeployError): self.cut()
        self.assertEqual(self.host.events, ["stop","install","verify_unit","start","stop","remove_override","start"])
        self.assertEqual(self.host.selected, self.source)

    def test_new_intents_prevent_blind_rollback_even_if_broker_looks_flat(self):
        def new_intent(*args):
            with closing(sqlite3.connect(self.release / "paper_order_intents.sqlite3")) as db:
                db.executescript("CREATE TABLE metadata(key TEXT,value TEXT);"
                    "INSERT INTO metadata VALUES('schema_version','1'),('account_id','synthetic-id');"
                    "CREATE TABLE intents(broker_id TEXT,state TEXT);"
                    "INSERT INTO intents VALUES(NULL,'unknown');")
                db.commit()
            raise deploy.DeployError("startup created order knowledge")
        self.host.healthy = new_intent
        with self.assertRaisesRegex(deploy.DeployError, "restoration was not confirmed"): self.cut()
        self.assertNotIn("remove_override", self.host.events)
        self.assertEqual(self.host.selected, self.release)
        self.assertTrue((self.release / "paper_order_intents.sqlite3").exists())
        self.assertFalse(json.loads((self.backup / "failure-state.json").read_text())["original_service_restored"])

    def test_partial_stop_failure_still_attempts_unchanged_original_service(self):
        def stop(): self.host.events.append("stop"); self.host.active=False; raise deploy.DeployError("stop status failed")
        self.host.stop=stop
        with self.assertRaises(deploy.DeployError): self.cut()
        self.assertEqual(self.host.events, ["stop","start"])
        self.assertTrue(self.host.active)


class GateTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "posix", "Original runtime validation reproduces Linux systemd startup")
    def test_original_runtime_check_selects_import_variables_and_never_imports_process_or_forwards_keys(self):
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp); unit_file=source/"original.service"; unit_file.write_text("unit fixture")
            unit=dict(FragmentPath=str(unit_file),DropInPaths="",EnvironmentFiles="",
                      ExecStart=f"{{ path=/python ; argv[]=/python -u {source}/paper_trader_integrated.py ; }}")
            with patch.object(deploy.check,"service_startup",return_value=({},time.time()+60)), \
                    patch.object(Path,"read_bytes",return_value=b"HOME=/fake\0PYTHONPATH=/old/site\0ALPACA_SECRET_KEY=never-forward\0"), \
                    patch.object(deploy.subprocess,"run") as run:
                deploy.verify_original_runtime(source,dict(MainPID="123"),unit)
                self.assertEqual(run.call_args.kwargs["env"],dict(HOME="/fake",PYTHONPATH="/old/site"))
                self.assertEqual(run.call_args.args[0][:2],["/python","-c"])
                self.assertNotIn("paper_trader_integrated.py",run.call_args.args[0][2])
                run.side_effect=subprocess.CalledProcessError(1,["/python"],stderr=b"never-export")
                with self.assertRaisesRegex(deploy.DeployError,"original service remains running"):
                    deploy.verify_original_runtime(source,dict(MainPID="123"),unit)

    def test_clock_rejects_open_stale_and_imminent_open(self):
        now = pd.Timestamp("2026-11-12T12:00:00Z")
        clock = dict(timestamp=now.isoformat(), is_open=False, next_open="2026-11-12T14:30:00Z")
        deploy.require_closed_window(clock, now)
        for changed in (dict(clock,is_open=True),dict(clock,timestamp=(now-pd.Timedelta(minutes=2)).isoformat()),
                        dict(clock,next_open=(now+pd.Timedelta(minutes=20)).isoformat())):
            with self.assertRaises(deploy.DeployError): deploy.require_closed_window(changed, now)

    def test_paper_environment_and_unit_are_explicit_and_secret_encoding_is_restricted(self):
        self.assertIn("ALPACA_PAPER=true", deploy.service_env(("fake-key","fake-secret")))
        with self.assertRaises(deploy.DeployError): deploy.service_env(("bad\nkey","secret"))
        unit=deploy.override_text(PurePosixPath("/home/opc/release"),PurePosixPath("/home/opc/release/.venv/bin/python"),
                                  PurePosixPath("/etc/v52-paper-test.env"),PurePosixPath("/usr/bin/python3.14"))
        self.assertIn("ExecStart=\n",unit); self.assertIn(" -E -s -u ",unit)
        self.assertIn("EnvironmentFile=/etc/v52-paper-test.env",unit)
        self.assertIn("ExecStart=@/usr/bin/python3.14 /home/opc/release/.venv/bin/python",unit)
        self.assertNotIn("fake-secret",unit)

    def test_runtime_probe_preserves_service_restrictions_without_unit_relationships(self):
        text=("[Unit]\nRequires=original.service\nOnFailure=notify.service\n"
              "[Service]\nUser=opc\nProtectHome=read-only\nNoNewPrivileges=yes\n"
              "EnvironmentFile=/etc/original.env\n[Install]\nWantedBy=multi-user.target\n"
              "# /etc/drop-in.conf\n[Service]\nPrivateTmp=yes\n")
        result=deploy.service_settings(text)
        for value in ("User=opc","ProtectHome=read-only","NoNewPrivileges=yes",
                      "EnvironmentFile=/etc/original.env","PrivateTmp=yes"):
            self.assertIn(value,result)
        for value in ("Requires=","OnFailure=","WantedBy="):
            self.assertNotIn(value,result)

    def test_data_transport_rejects_writes_and_bounds_gets_without_redirects(self):
        from alpaca.data.historical import StockHistoricalDataClient
        client=probe.make_data_client(("fake-key","fake-secret"))
        with patch.object(StockHistoricalDataClient,"_one_request",return_value={}) as request:
            for method in ("POST","DELETE","PATCH"):
                with self.assertRaises(probe.ProbeError): client._one_request(method,"https://data.alpaca.markets/test",{},0)
            request.assert_not_called()
            client._one_request("GET","https://data.alpaca.markets/test",{},4)
            self.assertEqual(request.call_args.args[2]["timeout"],(5,10))
            self.assertFalse(request.call_args.args[2]["allow_redirects"])
            self.assertEqual(request.call_args.args[3],0)

    def test_probe_rejects_partial_expired_errorful_or_nonzero_submission_reports(self):
        good=dict(complete=True,status="research_only",research_only=True,
                  daily_risk=dict(date="2026-11-12",trades=5,max_trades_per_day=5,loss_limit_reached=False),
                  scanned_count=2,universe_count=2,submitted_count=0,
                  signals=[dict(symbol="A",status="unavailable"),dict(symbol="B",status="evaluated")],
                  top5=[],as_of="2026-11-12T12:00:00Z")
        self.assertEqual(probe.validate_report(good,["A","B"])["scanned"],2)
        for changed in (dict(good,status="expired"),dict(good,status="complete"),dict(good,research_only=False),
                        dict(good,daily_risk=None),
                        dict(good,scanned_count=1),dict(good,submitted_count=1),
                        dict(good,signals=[dict(symbol="A",status="unavailable",error="API error"),good["signals"][1]])):
            with self.assertRaises(probe.ProbeError): probe.validate_report(changed,["A","B"])


class LiveFunctionProbeTests(unittest.TestCase):
    def test_actual_100_stock_production_functions_use_data_only_and_never_import_process(self):
        self.exercise_probe()

    def test_actual_probe_preserves_exhausted_daily_budget_and_reached_loss_limit(self):
        self.exercise_probe(capped=True)

    def exercise_probe(self, capped=False):
        from test_pipeline_contract import CFG, price_frame
        from model_pipeline import fit_bundle
        from features import FEATURES, enrich, label
        raw=price_frame(); train=label(enrich(raw,raw)).dropna(subset=FEATURES+["target"])
        cfg=dict(CFG,symbols=[f"S{i:03}" for i in range(100)],probability_threshold=0.,trade_score_threshold=0.,
                 max_trades_per_day=5)
        bundle=fit_bundle(train,cfg,"neutral")
        fixed=pd.Timestamp.now(tz="UTC").floor("5min") + pd.Timedelta(seconds=20)
        fresh=raw.copy(); fresh.timestamp=pd.date_range(end=fixed.floor("5min")-pd.Timedelta(minutes=5),periods=len(raw),freq="5min")
        data=Mock()
        data.get_stock_bars.return_value=SimpleNamespace(df=fresh.iloc[-251:].set_index("timestamp"))
        data.get_stock_latest_trade.side_effect=lambda r:{
            r.symbol_or_symbols[0] if isinstance(r.symbol_or_symbols,list) else r.symbol_or_symbols:SimpleNamespace(price=100.)}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); (root/"models").mkdir()
            (root/"config.json").write_text(json.dumps(cfg))
            with (root/"models/ensemble.pkl").open("wb") as f: pickle.dump(bundle,f)
            text=(Path(__file__).with_name("paper_trader_integrated.py")).read_text()
            (root/"paper_trader_integrated.py").write_text("raise AssertionError('module body must never run')\n"+text)
            if capped:
                state=dict(date=pd.Timestamp.now(tz="America/New_York").date().isoformat(),trades=5,
                           start_equity=100000.,traded_symbols=["S099"])
                (root/"paper_live_state.json").write_text(json.dumps(state))
            before={str(p.relative_to(root)):p.read_bytes() for p in root.rglob("*") if p.is_file()}
            clock=Mock(wraps=pd.Timestamp); clock.now.return_value=fixed
            proxy=SimpleNamespace(**dict(pd.__dict__,Timestamp=clock))
            loader=probe.load_functions
            def load(path,scope):
                scope["pd"]=proxy
                scope["telegram_trading_enabled"]=True  # Explicit research return must independently prevent orders.
                return loader(path,scope)
            with patch.object(probe,"load_functions",side_effect=load),redirect_stdout(io.StringIO()), \
                    patch.object(probe,"deny_submission",side_effect=AssertionError("No submission call allowed")) as submit, \
                    patch("socket.socket.connect",side_effect=AssertionError("Network forbidden")):
                report=probe.run_probe(root,root/"evidence",("fake-key","fake-secret"),
                                       dict(equity="95000" if capped else "100000"),data_client=data)
                submit.assert_not_called()
            self.assertEqual(report["scanned"],100)
            self.assertEqual(report["submitted"],0)
            self.assertEqual(report["shortlisted"],5)
            self.assertEqual(data.get_stock_bars.call_count,101)
            self.assertEqual(data.get_stock_latest_trade.call_count,5)
            self.assertTrue(report["research_only"])
            self.assertEqual(report["daily_risk"]["trades"],5 if capped else 0)
            self.assertEqual(report["daily_risk"]["loss_limit_reached"],capped)
            self.assertEqual({p: (root/p).read_bytes() for p in before},before)
            self.assertEqual((root/"paper_live_state.json").exists(),capped)
            self.assertFalse((root/"paper_order_intents.sqlite3").exists())


ISOLATED = (sys.prefix != sys.base_prefix and (Path(sys.prefix)/"pyvenv.cfg").is_file()
            and "include-system-site-packages = false" in (Path(sys.prefix)/"pyvenv.cfg").read_text())


@unittest.skipUnless(os.name == "posix" and ISOLATED, "Explicit same-host packaging check needs an isolated Linux venv")
class RuntimePackagingTests(unittest.TestCase):
    def test_real_release_has_independent_runtime_and_unchanged_model_configuration(self):
        from test_oracle_candidate import CFG, bars
        from tools import prepare_oracle_candidate as prep
        original = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); stage=root/"stage"; source=root/"source"; candidate=root/"candidate"; release=root/"release"
            stage.mkdir(); source.mkdir(); (stage/"tools").mkdir(); (stage/"models").mkdir(); (source/"models").mkdir()
            for name in [*original.glob("*.py"),*(original/"tools").glob("*.py")]:
                target=stage/name.relative_to(original); target.write_bytes(name.read_bytes())
            (stage/"models/ensemble.pkl").write_bytes(b"legacy model must be replaced in the release only")
            (source/"models/ensemble.pkl").write_bytes(b"original model must remain unchanged")
            cfg=dict(CFG,symbols=["AAA","BBB"],benchmark_symbols=["SPY"],history_start="2026-01-01",history_end="2026-02-01",
                     max_position_pct=.2,max_total_exposure_pct=1.)
            (source/"config.json").write_text(json.dumps(cfg)); (stage/"config.json").write_text(json.dumps(CFG))
            for args in (["init","-q"],["add","--","."],
                         ["-c","user.name=Test","-c","user.email=test@example.invalid","commit","-qm","fixture","--no-gpg-sign"]):
                subprocess.run(["git","-C",str(stage),*args],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            frames={s:bars(seed=i+51) for i,s in enumerate(["AAA","BBB","SPY"])}
            before={str(p.relative_to(source)):p.read_bytes() for p in source.rglob("*") if p.is_file()}
            with redirect_stdout(io.StringIO()),patch("socket.socket.connect",side_effect=AssertionError("Network forbidden")):
                prep.run(source,candidate,fetch=lambda p:dict(bars={p["symbols"]:frames[p["symbols"]]}))
                runtime=deploy.prepare_release(stage,source,candidate,release,Path(sys.executable), ("fake-key","fake-secret"))
            command=[str(runtime),"-E","-s","-u",str(release/"tools/check_oracle_service_runtime.py"),
                     "--release-dir",str(release),"--expected-uid",str(os.geteuid()),
                     "--base-interpreter",str(Path(sys._base_executable).resolve())]
            env=dict(os.environ,ALPACA_API_KEY="fake-key",ALPACA_SECRET_KEY="fake-secret",ALPACA_PAPER="true")
            result=subprocess.run(command,executable=str(Path(sys._base_executable).resolve()),env=env,
                                  check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=60)
            self.assertIn("SERVICE_RUNTIME_VERIFIED",result.stdout)
            proof=json.loads((release/".service-runtime-proof.json").read_text())
            self.assertTrue(proof["environment_matches"])
            self.assertTrue(proof["network_blocked"])
            self.assertEqual(proof["prefix"],str(release/".venv"))
            self.assertFalse((release/"paper_order_intents.sqlite3").exists())
            mismatch=subprocess.run(command,executable=str(Path(sys._base_executable).resolve()),
                                    env=dict(env,ALPACA_SECRET_KEY="wrong-secret"),
                                    stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=60)
            self.assertEqual(mismatch.returncode,1)
            self.assertNotIn("wrong-secret",mismatch.stdout+mismatch.stderr)
            self.assertEqual(runtime,release/".venv/bin/python")
            self.assertEqual((release/"config.json").read_bytes(),(source/"config.json").read_bytes())
            self.assertEqual((release/"models/ensemble.pkl").read_bytes(),(candidate/"candidate.pkl").read_bytes())
            self.assertEqual(before,{str(p.relative_to(source)):p.read_bytes() for p in source.rglob("*") if p.is_file()})
            self.assertNotEqual(runtime.parent.parent.resolve(),Path(sys.prefix).resolve())


@unittest.skipUnless(os.name == "posix", "Root-owned environment and system service startup are Linux-specific")
class SystemServiceRuntimeTests(unittest.TestCase):
    def exercise(self, *, fail_start=False, mode=b"0 0 600\n"):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);backup=root/"backup";release=root/"release";backup.mkdir();release.mkdir()
            (release/"models").mkdir();(release/"models/ensemble.pkl").write_bytes(b"model")
            (release/"config.json").write_text("{}")
            (release/".paper-service.env").write_text("ALPACA_API_KEY=fake-key\nALPACA_SECRET_KEY=fake-secret\n")
            host=deploy.Host(PurePosixPath("/etc/systemd/system/v52-paper-bot.service.d/zz-test-runtime.conf"))
            host.backup=backup;host.original_settings="[Service]\nUser=opc\nNoNewPrivileges=yes\n"
            events=[]
            def sudo(*args):
                events.append(tuple(map(str,args)))
                if args[0]=="stat": return mode
                if args[:2]==("systemctl","start"):
                    self.assertNotEqual(args[2],deploy.SERVICE)
                    if fail_start: raise subprocess.CalledProcessError(1,["systemctl","start"],stderr=b"Permission denied")
                    deploy.save_json(release/".service-runtime-proof.json",dict(
                        uid=os.geteuid(),prefix=str(release/".venv"),base_interpreter=str(host.base_interpreter),
                        model_sha256=deploy.digest(release/"models/ensemble.pkl"),
                        config_sha256=deploy.digest(release/"config.json"),
                        environment_matches=True,paper=True,write_test=True,network_blocked=True))
                return b"runtime journal\n" if args[0]=="journalctl" else b""
            with patch.object(host,"sudo",side_effect=sudo),patch.object(host,"restore_context"), \
                    patch.object(host,"command",return_value=b"Result=success\nExecMainStatus=0\n"), \
                    patch("socket.socket.connect",side_effect=AssertionError("Network forbidden")):
                if fail_start:
                    with self.assertRaisesRegex(deploy.DeployError,"before stopping the original"):
                        host.verify_runtime(release,release/".venv/bin/python")
                    self.assertTrue((backup/"runtime-check-log.txt").is_file())
                elif mode!=b"0 0 600\n":
                    with self.assertRaisesRegex(deploy.DeployError,"root-owned mode 0600"):
                        host.verify_runtime(release,release/".venv/bin/python")
                else:
                    result=host.verify_runtime(release,release/".venv/bin/python")
                    self.assertTrue(result["paper"])
                    unit=(backup/"runtime-check.service").read_text()
                    self.assertIn("NoNewPrivileges=yes",unit)
                    self.assertIn("User=opc",unit)
                    self.assertIn(f"EnvironmentFile={host.environment_file}",unit)
                    self.assertNotIn("fake-secret",unit)
                    self.assertTrue((backup/"runtime-check.json").is_file())
                self.assertFalse(any(e[:2]==("systemctl","stop") and e[2]==deploy.SERVICE for e in events))
                installs=[e for e in events if e[0]=="install"]
                self.assertTrue(all(e[1:8]==("-T","-m","0600","-o","root","-g","root") for e in installs))
                self.assertTrue(all("fake-secret" not in str(e) for e in events))

    def test_private_environment_and_actual_launch_are_verified_before_original_stop(self): self.exercise()

    def test_manager_cannot_read_environment_original_is_not_stopped(self): self.exercise(fail_start=True)

    def test_incorrect_environment_permissions_stop_before_probe_start(self): self.exercise(mode=b"0 0 644\n")

    def test_failed_host_command_retains_private_diagnostics(self):
        with tempfile.TemporaryDirectory() as temp:
            host=deploy.Host(PurePosixPath("/etc/test.conf"));host.backup=Path(temp)
            error=subprocess.CalledProcessError(1,["systemctl","start"],stderr=b"private failure details")
            with patch.object(deploy.subprocess,"check_output",side_effect=error):
                with self.assertRaises(subprocess.CalledProcessError): host.command(["systemctl","start","fixture.service"])
            path=Path(temp)/"failed-host-command.json"
            self.assertEqual(path.stat().st_mode & 0o777,0o600)
            self.assertEqual(json.loads(path.read_text())["returncode"],1)


if __name__ == "__main__": unittest.main()
