"""An isolated exploratory study; fixtures are synthetic, not trading evidence."""
from contextlib import ExitStack, redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from tools import trial_open_priority as trial
from test_intraday_replay import CFG, bars, signals
from test_august_simulation import TRAIN
from test_locked_period import WARM_DAYS
from tools.run_offline_simulation import synthetic_bars


class OpeningPriorityTests(unittest.TestCase):
    def fixture(self):
        day="2026-08-03"; symbols=["AMD","NVDA","ORCL","QCOM"]
        prices={s:bars(day=day) for s in symbols}
        scores={}
        for s,value in zip(symbols,[.7,.9,.9,.99]):
            x=signals(prices[s],clocks=("09:30","09:35"))
            x["signal_available_at"]=x.timestamp+pd.Timedelta(minutes=5)
            x["ml_prob"]=value
            scores[s]=x if s!="QCOM" else x.iloc[1:].copy()
        return day,symbols,scores,prices

    def test_missing_ties_and_future_rows_do_not_change_opening_priority(self):
        day,symbols,scores,_=self.fixture()
        order,probs=trial.opening_order(symbols,scores,day)
        self.assertEqual(order,["NVDA","ORCL","AMD","QCOM"])
        self.assertNotIn("QCOM",probs)
        for s in symbols:
            x=scores[s]; x.loc[x.clock!="09:30","ml_prob"]=1. if s=="AMD" else 0.
        self.assertEqual(trial.opening_order(symbols,scores,day),(order,probs))
        scores["NVDA"].loc[:,"signal_available_at"]+=pd.Timedelta(minutes=5)
        with self.assertRaisesRegex(ValueError,"availability"): trial.opening_order(symbols,scores,day)

    def test_priority_changes_fill_preserves_cap_and_spy_gate(self):
        day,symbols,scores,prices=self.fixture()
        order,_=trial.opening_order(symbols,scores,day)
        cfg=dict(CFG,symbols=symbols,max_trades_per_day=1)
        args=(scores,prices,cfg,{day:"16:00"},{day:False},{day:order})
        base,_=trial.replay(*args,False,False)
        alt,audit=trial.replay(*args,False,True)
        self.assertEqual(base["trades"].symbol.tolist(),["AMD"])
        self.assertEqual(alt["trades"].symbol.tolist(),["NVDA"])
        self.assertEqual(alt["trades"].symbol_order.tolist(),[2])
        self.assertEqual(alt["trades"].scan_order.tolist(),[1])
        self.assertEqual(audit["issues"],[])
        gated,_=trial.replay(*args,True,True)
        self.assertTrue(gated["trades"].empty)
        self.assertEqual(len(gated["daily"]),1)

    def test_both_archives_no_refit_and_mismatch_is_not_a_result(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as st:
            root=Path(tmp); hist=root/"hist"; hist.mkdir()
            syms=["NVDA","AMD"]; aug=trial.aug; locked=trial.locked; fw=trial.forward
            st.enter_context(redirect_stdout(io.StringIO()))
            st.enter_context(patch.object(aug.reference,"SYMBOLS",syms))
            st.enter_context(patch.object(aug.reference,"CFG",dict(aug.reference.CFG,symbols=syms)))
            st.enter_context(patch.object(fw,"utcnow",return_value=pd.Timestamp("2026-11-09T22:00Z")))
            for s,k in [("NVDA",1),("AMD",2),("SPY",3)]:
                synthetic_bars(TRAIN+list(aug.load_sessions()),k).to_csv(hist/f"{s}_5min.csv",index=False)
            def market(s,_headers):
                x=synthetic_bars(WARM_DAYS+locked.DAYS,{"NVDA":1.,"AMD":2.,"SPY":3.}[s])
                if s=="SPY":
                    mask=x.timestamp.dt.tz_convert(locked.NY).dt.strftime("%H:%M")=="09:30"
                    x.loc[mask,"open"]*=.999
                return x
            ref,old,state=root/"reference",root/"old",root/"state"
            aug.run(hist,ref,execute=True,low_memory=True,retain_replay_inputs=True)
            # Fixed synthetic API fixture; never issue a historical data request.
            env=root/"dummy"; env.write_text("ALPACA_API_KEY=dummy\nALPACA_SECRET_KEY=dummy\n")
            with patch.object(locked,"fetch_bars",side_effect=market):
                locked.run(hist,ref,old,env_file=env)
            fw.initialize(state,hist,ref)
            before={p.name:aug.digest(p) for p in state.iterdir() if p.is_file()}
            with patch.object(locked,"fit_disk_bundle",side_effect=AssertionError("refit")):
                info=trial.run(ref,old,state,root/"result")
            self.assertTrue(info["all_controls_reproduced"])
            self.assertEqual(info["training_runs"],0)
            self.assertEqual(before,{p.name:aug.digest(p) for p in state.iterdir() if p.is_file()})
            for label,days in [("august",21),("october_november",14)]:
                self.assertEqual(len(pd.read_csv(root/"result"/label/"daily.csv")),4*days)
            # A manifest hash that agrees with deliberately incorrect results must
            # still fail actual reproduction before performance is published.
            summary=ref/"summary.csv"; df=pd.read_csv(summary); df.loc[0,"trades"]+=1; df.to_csv(summary,index=False)
            m=fw.read(ref/"manifest.json"); m["outputs"][summary.name]=aug.digest(summary); aug.write_json(ref/"manifest.json",m)
            r=fw.read(old/"report.json"); r["reference_manifest_sha256"]=aug.digest(ref/"manifest.json"); aug.write_json(old/"report.json",r)
            with self.assertRaisesRegex(ValueError,"control differs"):
                trial.run(ref,old,state,root/"bad")
            self.assertEqual(fw.read(root/"bad/report.json")["status"],"failed_not_validated")
            self.assertFalse((root/"bad/august/summary.csv").exists())
            with self.assertRaisesRegex(ValueError,"separate"): trial.run(ref,old,state,state/"nested")
