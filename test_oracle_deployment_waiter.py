"""Bounded Oracle cutover waiter: synthetic clocks, no service changes or network."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd
from tools import wait_oracle_deployment as wait


def sample():
    now = pd.Timestamp.now(tz="UTC")
    return dict(account=dict(id="synthetic-account", status="ACTIVE", trading_blocked=False), positions=[], orders=[],
                clock=dict(timestamp=now.isoformat(),is_open=True,next_close=(now+pd.Timedelta(hours=2)).isoformat(),
                           next_open=(now+pd.Timedelta(days=1)).isoformat()))


class ClockTests(unittest.TestCase):
    def test_open_session_waits_fifteen_minutes_after_reported_close(self):
        broker=sample()
        when=wait.earliest_attempt(broker["clock"])
        self.assertEqual(when,pd.Timestamp(broker["clock"]["next_close"])+pd.Timedelta(minutes=15))
        self.assertFalse(wait.ready(broker["clock"],when))

    def test_stale_clock_is_not_mistaken_for_an_open_window(self):
        broker=sample(); broker["clock"]["timestamp"]=(pd.Timestamp.now(tz="UTC")-pd.Timedelta(minutes=5)).isoformat()
        with self.assertRaisesRegex(wait.WaitError,"clock is not fresh"):
            wait.earliest_attempt(broker["clock"])

    def test_original_closed_market_gate_is_still_required(self):
        clock=sample()["clock"]; clock["is_open"]=False
        with patch.object(wait.deploy,"require_closed_window",side_effect=wait.deploy.DeployError("not safe")) as gate:
            self.assertFalse(wait.ready(clock,pd.Timestamp.now(tz="UTC")-pd.Timedelta(seconds=1)))
            gate.assert_called_once()


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.output=Path(self.tmp.name)
        self.broker=sample()
        self.intent=dict(source_dir="/original",candidate_dir="/candidate",candidate_sha256="modelhash",
                         config_sha256="cfghash",account_sha256=wait.fingerprint(self.broker["account"]),
                         code_commit="frozen-code",not_before="2026-11-12T21:15:00Z",deadline_epoch=1000.)

    def test_while_open_no_deployment_until_ready_and_then_only_one_attempt(self):
        run=Mock(return_value=0); sleeps=[]
        with patch.object(wait,"guard"),patch.object(wait,"ready",side_effect=[False,True]), \
                patch("socket.socket.connect",side_effect=AssertionError("Network forbidden")):
            result=wait.supervise(self.intent,self.output,snapshot=lambda p:self.broker,deploy_once=run,
                                  sleeper=sleeps.append,now=lambda:0.)
        self.assertEqual(sleeps,[60]); run.assert_called_once()
        self.assertEqual(result["phase"],"deployment_command_completed")

    def test_failed_cutover_is_not_retried(self):
        run=Mock(return_value=1)
        with patch.object(wait,"guard"),patch.object(wait,"ready",return_value=True):
            result=wait.supervise(self.intent,self.output,snapshot=lambda p:self.broker,deploy_once=run,now=lambda:0.)
        self.assertEqual(result["phase"],"deployment_command_failed"); run.assert_called_once()

    def test_open_positions_after_close_are_waited_out_without_closing_them(self):
        occupied=sample();occupied["positions"]=[dict(symbol="AAA",qty="1")]
        snap=Mock(side_effect=[occupied,self.broker]);run=Mock(return_value=0);sleeps=[]
        with patch.object(wait,"guard"),patch.object(wait,"ready",return_value=True):
            wait.supervise(self.intent,self.output,snapshot=snap,deploy_once=run,sleeper=sleeps.append,now=lambda:0.)
        self.assertEqual(sleeps,[60]);run.assert_called_once()

    def test_changed_account_or_code_stops_before_deployment(self):
        run=Mock(); changed=sample(); changed["account"]["id"]="different-account"
        with patch.object(wait,"guard"),self.assertRaisesRegex(wait.WaitError,"identity changed"):
            wait.supervise(self.intent,self.output,snapshot=lambda p:changed,deploy_once=run,now=lambda:0.)
        with patch.object(wait,"guard",side_effect=wait.WaitError("code changed")),self.assertRaises(wait.WaitError):
            wait.supervise(self.intent,self.output,snapshot=lambda p:self.broker,deploy_once=run,now=lambda:0.)
        run.assert_not_called()

    def test_transient_read_error_retries_but_authentication_error_does_not(self):
        run=Mock(return_value=0); snap=Mock(side_effect=[wait.check.CheckError("Paper GET clock: HTTP 503"),self.broker])
        with patch.object(wait,"guard"),patch.object(wait,"ready",return_value=True):
            wait.supervise(self.intent,self.output,snapshot=snap,deploy_once=run,sleeper=lambda n:None,now=lambda:0.)
        self.assertEqual(snap.call_count,2);run.assert_called_once()
        with patch.object(wait,"guard"),self.assertRaises(wait.check.CheckError):
            wait.supervise(self.intent,self.output,snapshot=Mock(side_effect=wait.check.CheckError("HTTP 403")),now=lambda:0.)

    def test_deadline_expires_without_attempt(self):
        run=Mock()
        with self.assertRaisesRegex(wait.WaitError,"deadline"):
            wait.supervise(self.intent,self.output,deploy_once=run,now=lambda:2000.)
        run.assert_not_called()


@unittest.skipUnless(os.name == "posix", "Detached Oracle waiter uses Linux file locks")
class LaunchTests(unittest.TestCase):
    def test_launch_uses_new_session_no_stdin_and_no_credentials_in_arguments(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root=Path(temp);source=root/"source";candidate=root/"candidate";output=root/"waiter"
            source.mkdir();candidate.mkdir();(source/"config.json").write_text("{}")
            (candidate/"run.lock").write_text("0")
            broker=sample()
            stack.enter_context(patch.object(wait,"active_status",side_effect=[None,dict(phase="waiting_for_safe_window",pid=555)]))
            stack.enter_context(patch.object(wait,"clean"))
            stack.enter_context(patch.object(wait,"head",return_value="frozen-code"))
            stack.enter_context(patch.object(wait.check,"verify_candidate",return_value=({},dict(stocks=100,candidate_sha256="modelhash"))))
            stack.enter_context(patch.object(wait.check,"service_info",return_value=dict(WorkingDirectory=str(source))))
            stack.enter_context(patch.object(wait.check,"running_service_pair",return_value=(("private-key","private-secret"),"source")))
            stack.enter_context(patch.object(wait.check,"paper_gets",return_value=broker))
            prepare=stack.enter_context(patch.object(wait.deploy.Host,"prepare"))
            launch=stack.enter_context(patch.object(wait.subprocess,"Popen",return_value=Mock(pid=555)))
            stack.enter_context(patch.object(wait.time,"sleep"))
            stack.enter_context(patch("socket.socket.connect",side_effect=AssertionError("Network forbidden")))
            result=wait.start(source,candidate,output,"modelhash")
            self.assertTrue(result["running"]);self.assertEqual(result["pid"],555)
            self.assertTrue(launch.call_args.kwargs["start_new_session"])
            self.assertEqual(launch.call_args.kwargs["stdin"],wait.subprocess.DEVNULL)
            self.assertNotIn("private-key",str(launch.call_args.args));self.assertNotIn("private-secret",str(launch.call_args.args))
            self.assertNotIn("private-secret",(output/"intent.json").read_text())
            prepare.assert_called_once()

    def test_completed_waiter_does_not_schedule_another_legacy_migration(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);output=root/"waiter";output.mkdir()
            (output/"status.json").write_text(json.dumps(dict(phase="deployment_command_completed")))
            with patch.object(wait,"active_status",return_value=None),patch.object(wait.check,"service_info",return_value={}), \
                    patch.object(wait.subprocess,"Popen") as launch:
                result=wait.start(root/"source",root/"candidate",output,"modelhash")
            self.assertTrue(result["previous_attempt_completed"]);launch.assert_not_called()


if __name__ == "__main__": unittest.main()
