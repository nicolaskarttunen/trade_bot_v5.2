import io
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from model_pipeline import trade_score
from tools import trial_ml_filter as trial
from tools.run_offline_simulation import synthetic_bars
from test_locked_period import WARM_DAYS
from test_august_simulation import TRAIN
from test_intraday_replay import bars, signals


class MlFilterTests(unittest.TestCase):
    def test_only_stock_threshold_removed_and_no_future_return(self):
        day = "2026-08-03"
        px = bars(day=day)
        symbols = ["AMD", "NVDA"]
        cfg = dict(trial.aug.reference.CFG, symbols=symbols,max_trades_per_day=1)
        market = px[["timestamp","close"]].copy()
        # make the market return zero, negative, then positive at 09:55.
        market.loc[4,"close"]=99
        market.loc[5,"close"]=101
        rows = {}
        for s,p in zip(symbols,(.2,.9)):
            x=signals(px,clocks=("09:45","09:50","09:55"))
            x["signal_available_at"]=x.timestamp+pd.Timedelta(minutes=5)
            x["ml_prob"]=p
            x["trade_score"]=trade_score(p,[0,-.01,.01],0,cfg)
            rows[s]=x
        selected=trial.market_rows(rows,market,cfg)
        for s in symbols:
            self.assertEqual(selected[s].clock.tolist(),["09:55"])
        self.assertEqual(selected["AMD"].ml_prob.item(),.2)
        args=({s:px for s in symbols}, {day:"16:00"})
        def replay(r,c):
            return trial.prior.replay(r,args[0],c,args[1],{day:False},{},False,False)[0]
        orig=replay(rows,cfg)
        alt=replay(selected,dict(cfg,probability_threshold=0,trade_score_threshold=0))
        self.assertEqual(orig["trades"].symbol.tolist(),["NVDA"])
        self.assertEqual(alt["trades"].symbol.tolist(),["AMD"])
        self.assertEqual(alt["trades"].entry_ts.iloc[0],market.timestamp.iloc[6])
        market.loc[6:,"close"]=1.
        self.assertTrue(selected["AMD"].equals(trial.market_rows(rows,market,cfg)["AMD"]))
        rows["AMD"].loc[:,"trade_score"]=88
        with self.assertRaisesRegex(ValueError,"mismatch"):
            trial.market_rows(rows,market,cfg)

    def test_exposure_and_turnover_include_idle_sessions(self):
        day="2026-08-03"; other="2026-08-04"
        frames={"trades":pd.DataFrame([dict(qty=10,entry=10,exit=11,pnl=10,date=day,
                    entry_ts="2026-08-03T14:00Z",exit_ts="2026-08-03T14:39Z")]),
                "daily":pd.DataFrame([dict(date=day,pnl=10,start_equity=1000),
                                      dict(date=other,pnl=0,start_equity=1010)])}
        a=trial.activity(frames,{day:"16:00",other:"16:00"})
        self.assertAlmostEqual(a["avg_session_exposure_pct"],.5)
        self.assertAlmostEqual(a["avg_daily_two_sided_turnover_pct"],10.5)
        self.assertAlmostEqual(a["net_extra_3bp_usd"],9.937)

    def test_control_reproduction_and_frozen_archives(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            stack.enter_context(redirect_stdout(io.StringIO()))
            root=Path(tmp); history=root/"history"; history.mkdir()
            ref,old,state=root/"ref",root/"old",root/"state"
            syms=["NVDA","AMD"]; aug,locked,fw=trial.aug,trial.locked,trial.forward
            stack.enter_context(patch.object(aug.reference,"SYMBOLS",syms))
            stack.enter_context(patch.object(aug.reference,"CFG",dict(aug.reference.CFG,symbols=syms)))
            stack.enter_context(patch.object(fw,"utcnow",return_value=pd.Timestamp("2026-11-09T22:00Z")))
            for s,k in [("NVDA",1),("AMD",2),("SPY",3)]:
                synthetic_bars(TRAIN+list(aug.load_sessions()),k).to_csv(history/f"{s}_5min.csv",index=False)
            aug.run(history,ref,execute=True,low_memory=True,retain_replay_inputs=True)
            env=root/"dummy"; env.write_text("ALPACA_API_KEY=dummy\nALPACA_SECRET_KEY=dummy\n")
            with patch.object(locked,"fetch_bars",side_effect=lambda s,h:synthetic_bars(
                        WARM_DAYS+locked.DAYS,{"NVDA":1,"AMD":2,"SPY":3}[s])):
                locked.run(history,ref,old,env_file=env)
            fw.initialize(state,history,ref)
            frozen={f.name:aug.digest(f) for f in state.iterdir() if f.is_file()}
            with patch.object(locked,"fit_disk_bundle",side_effect=AssertionError("unexpected training")):
                result=trial.run(ref,old,state,history,root/"result")
            self.assertEqual(result["status"],"completed_exploratory_ml_ablation")
            self.assertEqual(frozen,{f.name:aug.digest(f) for f in state.iterdir() if f.is_file()})
            for name,n in zip(trial.PERIODS,(21,14)):
                self.assertEqual(len(pd.read_csv(root/"result"/name/"daily.csv")),2*n)
                pair=fw.read(root/"result"/name/"audit.json")["ml_minus_market_only"]
                self.assertFalse(pair["independent_evidence"]); self.assertEqual(pair["sessions"],n)
            with self.assertRaises(FileExistsError): trial.run(ref,old,state,history,root/"result")
            with (history/"SPY_5min.csv").open("ab") as f: f.write(b"\n")
            with self.assertRaisesRegex(ValueError,"Source changed"):
                trial.run(ref,old,state,history,root/"bad")
            self.assertEqual(fw.read(root/"bad/report.json")["status"],"failed_not_validated")
            self.assertFalse((root/"bad/august/summary.csv").exists())
