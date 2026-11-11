"""Fault injection and observational-log parity. Synthetic data only."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from intraday_replay import replay_intraday
from replay_audit import audit_frames
from test_intraday_replay import CFG, bars, signals
from tools.audit_simulation_run import audit_run, ARMS, WINDOWS, FILES
from tools.run_august_simulation import replay_by_session
from v6_5_intraday_replay import metrics


class ReplayAuditTests(unittest.TestCase):
    def fixture(self, **changes):
        cfg = dict(CFG, slippage_bps=3, spread_bps=2, commission_per_trade=1, **changes)
        price = bars()
        scores = signals(price)
        prices, scored = {"NVDA":price, "AMD":price}, {"NVDA":scores, "AMD":scores}
        sectors = {"NVDA":"Technology", "AMD":"Technology"}
        cash,t,d,c = replay_intraday(scored, prices, list(prices), sectors, cfg)
        s = pd.DataFrame([metrics(cfg["initial_cash"], cash,t,d,c)])
        return [t,d,c,s,cfg,list(prices),{"2026-10-01":"16:00"},sectors]

    def test_valid_ledger_concentration_and_costs(self):
        args = self.fixture()
        report = audit_frames(*args)
        self.assertEqual(report["issues"], [])
        self.assertEqual(report["top20_trade_pct"], 100)
        self.assertEqual(report["sectors"][0]["trades"], 2)
        self.assertEqual(report["recorded_max_open_positions"], 2)
        self.assertAlmostEqual(args[0].pnl.sum(), args[1].pnl.sum())

    def test_faults_in_money_limits_and_metadata_are_detected(self):
        mutations = [("pnl", 50, "trade_pnl_identity"),
                     ("qty", 1.5, "positive_integer_quantity"),
                     ("planned_stop_loss", 99999, "planned_risk_limit"),
                     ("entry_exposure", 99999, "entry_exposure_limit"),
                     ("post_entry_equity", 1, "position_limit"),
                     ("symbol_order", 99, "symbol_order"),
                     ("sector", "Energy", "sector_label"),
                     ("ml_prob", .1, "entry_thresholds"),
                     ("exit_ts", "2026-10-01 13:30:00+00:00", "trade_time_order"),
                     ("reason", "missing_eod_approx", "unvalidated_exit_reason")]
        for column,value,code in mutations:
            with self.subTest(column=column):
                args=self.fixture()
                if column == "qty": args[0][column]=args[0][column].astype(float)
                args[0].loc[0,column]=value
                self.assertIn(code, audit_frames(*args)["issues"])
        for index,column,value,code in [(1,"pnl",123,"daily_trade_pnl_reconciliation"),
                                       (1,"entries",6,"daily_entry_limit"),
                                       (1,"start_equity",99999,"initial_cash"),
                                       (2,"equity",99999,"equity_day_end_reconciliation"),
                                       (3,"return_pct",100,"summary_return_pct")]:
            args=self.fixture()
            args[index].loc[args[index].index[-1],column]=value
            self.assertIn(code, audit_frames(*args)["issues"])

    def test_nonfinite_missing_rows_and_duplicate_trades_fail(self):
        args=self.fixture(); args[0].loc[0,"entry"]=np.nan
        with self.assertRaisesRegex(ValueError,"Nonfinite"):
            audit_frames(*args)
        args=self.fixture(); args[0]=pd.concat([args[0], args[0].iloc[[0]]])
        self.assertIn("repeat_symbol_same_day",audit_frames(*args)["issues"])
        args=self.fixture(); args[2]=args[2].iloc[1:]
        self.assertIn("equity_grid",audit_frames(*args)["issues"])
        args=self.fixture(); args[0]=args[0].drop(columns="qty")
        with self.assertRaisesRegex(ValueError,"missing columns"):
            audit_frames(*args)

    def test_zero_trade_sessions_and_gap_losses_are_valid(self):
        price=bars()
        for gap in [False, True]:
            with self.subTest(gap=gap):
                p=price.copy()
                if gap:
                    p.loc[2,["open","high","low","close"]]=[90,91,89,90]
                cash,t,d,c=replay_intraday({"A":signals(p,qualified=gap)}, {"A":p}, ["A"],{},CFG)
                s=pd.DataFrame([metrics(CFG["initial_cash"],cash,t,d,c)])
                self.assertEqual(audit_frames(t,d,c,s,CFG,["A"],{"2026-10-01":"16:00"})["issues"], [])
                if gap:
                    self.assertGreater(-t.pnl.iloc[0],t.planned_stop_loss.iloc[0])


class DecisionLogTests(unittest.TestCase):
    def test_logging_preserves_outputs_and_marks_blocked_tail_unassessed(self):
        p=bars()
        prices={s:p for s in ["A","B","C"]}
        scored={s:signals(p) for s in prices}
        cfg=dict(CFG,max_trades_per_day=1)
        events=[]
        a=replay_intraday(scored,prices,list(prices),{},cfg)
        b=replay_intraday(scored,prices,list(prices),{},cfg,decision_sink=events.append)
        self.assertEqual(a[0],b[0])
        for x,y in zip(a[1:],b[1:]): assert_frame_equal(x,y)
        self.assertEqual([e["reason"] for e in events[:3]], ["entered", "daily_trade_limit_not_evaluated", "daily_trade_limit_not_evaluated"])
        self.assertFalse(events[1]["signal_evaluated"])
        self.assertNotIn("ml_prob",events[1])
        self.assertEqual(events[0]["signal_bar_ts"],p.timestamp.iloc[0].isoformat())
        self.assertEqual(events[0]["qty"],min(events[0][k] for k in ["risk_qty","position_qty","exposure_qty","cash_qty"]))

    def test_gated_unassessed_and_open_limit_threshold_missing_and_size(self):
        p=bars()
        for config,scored,gate,expected in [
            ({},{"A":signals(p)},True,"spy_gate_not_evaluated"),
            ({},{"A":signals(p,qualified=False)},False,"threshold_not_met"),
            ({},{},False,"missing_signal"),
            ({"max_position_pct":.00001},{"A":signals(p)},False,"size_below_one"),
        ]:
            events=[]
            replay_intraday(scored,{"A":p},["A"],{},dict(CFG,**config),require_spy=gate,decision_sink=events.append)
            self.assertEqual(events[0]["reason"],expected)
        events=[]
        replay_intraday({s:signals(p) for s in ["A","B"]},{"A":p,"B":p},["A","B"],{},dict(CFG,max_open_positions=1),decision_sink=events.append)
        self.assertEqual(events[1]["reason"],"open_position_limit")
        self.assertEqual(events[2]["reason"],"already_held_or_traded")
        events=[]
        replay_intraday({"A":signals(p)}, {"A":p.drop(index=1), "B":p}, ["A", "B"], {},CFG,decision_sink=events.append)
        self.assertEqual(events[0]["reason"],"missing_entry_bar")

    def test_daily_loss_stops_tail_without_claiming_eligibility(self):
        p=bars(low_override={"09:35":95})
        q=bars()
        events=[]
        replay_intraday({"A":signals(p),"B":signals(q,clocks=("09:35",)),"C":signals(q,clocks=("09:35",))},
                        {"A":p,"B":q,"C":q},["A","B","C"],{},dict(CFG,max_daily_loss_pct=.001),decision_sink=events.append)
        hit=next(i for i,e in enumerate(events) if e["reason"]=="daily_loss_limit")
        self.assertEqual(events[hit+1]["reason"],"daily_loss_limit_not_evaluated")
        self.assertFalse(events[hit+1]["signal_evaluated"])

    def test_sink_failure_propagates_and_session_wrapper_preserves_events(self):
        p=pd.concat([bars(),bars(day="2026-10-02")],ignore_index=True)
        scored={"A":signals(p)}; prices={"A":p}
        def fail(event): raise OSError("disk full")
        with self.assertRaisesRegex(OSError,"disk full"):
            replay_intraday(scored,prices,["A"],{},CFG,decision_sink=fail)
        first=[]; second=[]
        kw=dict(spy_opening_gate={},require_spy=False,session_closes={"2026-10-01":"16:00","2026-10-02":"16:00"})
        a=replay_intraday(scored,prices,["A"],{},CFG,decision_sink=first.append,**kw)
        b=replay_by_session(scored,prices,["A"],{},CFG,decision_sink=second.append,**kw)
        self.assertEqual(first,second)
        self.assertEqual(a[0],b[0])
        for x,y in zip(a[1:],b[1:]): assert_frame_equal(x,y)


class SavedRunAuditTests(unittest.TestCase):
    def create_old_run(self, root):
        source=root/"original"; source.mkdir()
        sessions=pd.bdate_range("2026-08-03","2026-08-31").strftime("%Y-%m-%d")
        price=pd.concat([bars(day=day) for day in sessions],ignore_index=True)
        cfg=dict(CFG,symbols=["NVDA"])
        collections={name:[] for name in ["trades","daily","equity","summary"]}
        for i,arm in enumerate(ARMS):
            cash,t,d,c=replay_intraday({"NVDA":signals(price)},{"NVDA":price},["NVDA"],{"NVDA":"Technology"},cfg,require_spy=bool(i))
            for name,frame in zip(collections,[t,d,c,pd.DataFrame([metrics(cfg["initial_cash"],cash,t,d,c)])]):
                collections[name].append(frame.assign(strategy=arm))
        for name,frames in collections.items(): pd.concat(frames,ignore_index=True).to_csv(source/f"{name}.csv",index=False)
        (source/"run_report.json").write_text(json.dumps(dict(status="completed_development_comparison",source_hashes_unchanged_at_end=True,symbols=["NVDA"])))
        (source/"spy_comparison.json").write_text("{}")
        manifest=dict(status="completed_development_comparison",windows=WINDOWS,config=cfg,
                      outputs={n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in FILES})
        (source/"manifest.json").write_text(json.dumps(manifest))
        return source,manifest

    def test_existing_run_no_retrain_no_overwrite_and_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root=Path(temp); source,manifest=self.create_old_run(root)
            before={p.name:p.read_bytes() for p in source.iterdir()}
            with patch("socket.socket.connect",side_effect=AssertionError("network")):
                report=audit_run(source,root/"audit")
            self.assertEqual(report["status"],"passed_ledger_checks")
            self.assertFalse(report["decision_log_available"])
            self.assertEqual(before,{p.name:p.read_bytes() for p in source.iterdir()})
            self.assertEqual(report["arms"][ARMS[1]]["zero_days"],21)
            with self.assertRaises(FileExistsError): audit_run(source,root/"audit")
            with self.assertRaises(ValueError): audit_run(source,source/"inside")
            (source/"trades.csv").write_bytes(before["trades.csv"]+b"\n")
            bad=audit_run(source,root/"tampered")
            self.assertEqual(bad["status"],"failed_ledger_checks")
            self.assertIn("hash mismatch",bad["error"])

    def test_failed_run_manifest_traversal_and_rehashed_bad_ledger_rejected(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root=Path(temp); source,manifest=self.create_old_run(root)
            for i,change in enumerate([{"status":"failed_not_validated"}, {"outputs":dict(manifest["outputs"], **{"../outside":"bad"})}]):
                (source/"manifest.json").write_text(json.dumps(dict(manifest,**change)))
                self.assertEqual(audit_run(source,root/f"bad{i}")["status"],"failed_ledger_checks")
            t=pd.read_csv(source/"trades.csv"); t.loc[0,"pnl"]=500
            t.to_csv(source/"trades.csv",index=False)
            manifest["outputs"]["trades.csv"]=hashlib.sha256((source/"trades.csv").read_bytes()).hexdigest()
            (source/"manifest.json").write_text(json.dumps(manifest))
            result=audit_run(source,root/"badledger")
            self.assertIn("trade_pnl_identity",result["arms"][ARMS[0]]["issues"])
            self.assertEqual(result["status"],"failed_ledger_checks")


if __name__ == "__main__": unittest.main()
