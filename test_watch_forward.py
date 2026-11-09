from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from model_pipeline import predict_bundle
from tools import watch_forward as fw
from tools.run_offline_simulation import synthetic_bars
from test_august_simulation import TRAIN


class ForwardTests(unittest.TestCase):
    def test_model_json_roundtrip_all_members_and_malformed_state(self):
        from model_pipeline import fit_bundle
        from features import FEATURES
        rng=np.random.default_rng(1)
        x=pd.DataFrame(rng.normal(size=(80,len(FEATURES))),columns=FEATURES)
        x["target"]=np.arange(80)%2
        cfg=fw.protocol()["config"]
        model=fit_bundle(x,cfg,"neutral")
        state=json.loads(json.dumps(fw.encode(model),allow_nan=False))
        new=fw.decode(state,cfg)
        np.testing.assert_array_equal(predict_bundle(new,x,cfg),predict_bundle(model,x,cfg))
        state["members"][0]["std"][0]=0
        with self.assertRaisesRegex(ValueError,"scale"): fw.decode(state,cfg)

    def test_calendar_holiday_and_early_close(self):
        self.assertEqual(len(fw.DATES),20); self.assertNotIn("2026-11-26",fw.CAL)
        day="2026-11-27"
        x=self.bars_for(day,1.)
        fw.check_bars(x,day,"SPY","SPY")
        with self.assertRaisesRegex(ValueError,"benchmark"):
            fw.check_bars(x.iloc[:-1],day,"SPY","SPY")
        self.assertEqual(fw.bounds(day)[2].tz_convert("UTC").hour,18)

    @staticmethod
    def bars_for(day,scale):
        days=pd.bdate_range(end=day,periods=7).strftime("%Y-%m-%d").tolist()
        x=synthetic_bars(days,scale)
        _,_,end=fw.bounds(day)
        return x.loc[x.timestamp<end].copy()

    def test_initialization_collection_restart_cash_and_snapshot(self):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            root=Path(folder); state=root/"forward"; history=root/"history"; history.mkdir()
            aug=fw.aug
            syms=["NVDA","AMD"]
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(patch.object(aug.reference,"SYMBOLS",syms))
            stack.enter_context(patch.object(aug.reference,"CFG",dict(aug.reference.CFG,symbols=syms)))
            now=stack.enter_context(patch.object(fw,"utcnow",return_value=pd.Timestamp("2026-11-09T22:00Z")))
            for s,k in [("NVDA",1.),("AMD",2.),("SPY",3.)]:
                synthetic_bars(TRAIN+list(aug.load_sessions()),k).to_csv(history/f"{s}_5min.csv",index=False)
            ref=root/"ref"; aug.run(history,ref,execute=True,low_memory=True,retain_replay_inputs=True)
            fw.initialize(state,history,ref)
            with patch.object(fw.locked,"fit_disk_bundle",side_effect=AssertionError("no refit")):
                fw.initialize(state,history,ref)
            calls=[]
            def fetch(symbol,start,end):
                day=end.tz_convert(fw.NY).strftime("%Y-%m-%d")
                calls.append((day,symbol))
                return self.bars_for(day,{"NVDA":1.,"AMD":2.,"SPY":3.}[symbol])
            for stamp in ["2026-11-10T20:59Z","2026-11-10T21:29Z"]:
                now.return_value=pd.Timestamp(stamp)
                self.assertEqual(fw.refresh(state,None,fetch=fetch)["status"],"waiting_first_close")
                self.assertFalse(calls)
            now.return_value=pd.Timestamp("2026-11-10T21:45Z")
            with patch.object(fw.locked,"one_arm",side_effect=ValueError("held gap")):
                with self.assertRaisesRegex(ValueError,"held gap"):
                    fw.refresh(state,None,fetch=fetch)
            self.assertFalse((state/"days/2026-11-10").exists())
            self.assertTrue((state/"snapshots/2026-11-10/bars.zip").is_file())
            self.assertEqual(len(calls),3)
            self.assertEqual(fw.refresh(state,None,fetch=fetch)["completed_sessions"],["2026-11-10"])
            self.assertEqual(len(calls),3)
            now.return_value=pd.Timestamp("2026-11-13T12:00Z")
            r=fw.refresh(state,None,fetch=fetch)
            self.assertEqual(len(r["completed_sessions"]),3)
            self.assertEqual(r["late_sessions"],["2026-11-11","2026-11-12"])
            self.assertEqual(len(calls),9)
            fw.refresh(state,None,fetch=fetch); self.assertEqual(len(calls),9)
            summary=pd.read_csv(state/"summary.csv")
            self.assertEqual(set(summary.strategy),set(a for a,_,_ in fw.locked.ARMS))
            self.assertNotIn("paired_daily_comparisons",r)
            now.return_value=pd.Timestamp("2026-12-09T06:00Z")
            final=fw.refresh(state,None,fetch=fetch)
            self.assertEqual(final["status"],"complete")
            self.assertEqual(final["completed_sessions"],fw.DATES)
            self.assertEqual(len(final["paired_daily_comparisons"]),4)
            self.assertEqual(len(calls),20*3)
            self.assertTrue(all(v["sessions"]==20 for v in final["paired_daily_comparisons"].values()))
            with self.assertRaisesRegex(ValueError,"Registration is closed"): fw.initialize(root/"late",history,ref)
            day=state/"days/2026-11-10/daily.csv"; old=day.read_bytes(); day.write_bytes(old+b"\n")
            with self.assertRaisesRegex(ValueError,"ledger chain"): fw.refresh(state,None,fetch=fetch)
            day.write_bytes(old)
            with (state/"model.json").open("ab") as f: f.write(b"\n")
            with self.assertRaisesRegex(ValueError,"changed"): fw.refresh(state,None,fetch=fetch)
